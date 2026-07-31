#include "adaiso/AdaIsoRemesher.hpp"
#include "adaiso/IO.hpp"
#include "adaiso/Metrics.hpp"

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void printUsage(std::ostream& out) {
  out << "Usage:\n"
      << "  adaiso_cli <input.obj|input.ply|input.off> -o <output.obj|output.ply> [options]\n\n"
      << "Options:\n"
      << "  --variant <code|paper>     Default: code\n"
      << "  --curvature-mode <normal-angle|robust-mean|principal-max> Default: principal-max\n"
      << "  --csf-alpha <value>        Screened CSF smoothing strength. Default: 2.0\n"
      << "  --target-error-delta <v>   Absolute error tolerance. Default: auto from input resolution and CSF curvature\n"
      << "  --auto-delta-quantile <v>  Area-weighted local delta quantile in [0, 1]. Default: 0.5\n"
      << "  --target-h-max-factor <v>  h_max as a multiple of mean edge length. Default: 5.0\n"
      << "  --export-target-field <dir> Export target-field CSV/PLY diagnostics\n"
      << "  --export-iteration-meshes <dir> Export per-iteration remesh snapshots\n"
      << "  --export-stage-meshes <dir> Export first-iteration remesh stage snapshots\n"
      << "  --export-all-stage-meshes <dir> Export every iteration's remesh stage snapshots\n"
      << "  --preprocess <0|1>         Default: 0\n"
      << "  --iters <n>                Default: 5\n"
      << "  --stage-a-policy <name>    Experimental: split_collapse_no_guard|split_collapse_weak_guard|collapse_split_weak_guard\n"
      << "  --split-factor <v>        Override split hysteresis factor. Default: variant value\n"
      << "  --collapse-factor <v>     Override collapse hysteresis factor. Default: 0.8\n"
      << "  --max-flip-dihedral-deg <v> Maximum allowed flip dihedral angle. Default: 45\n"
      << "  --feature-aware-collapse <0|1> Feature-aware collapse placement. Default: 0\n"
      << "  --collapse-feature-omega <v> Feature similarity threshold for collapse. Default: 0.15\n"
      << "  --stage-a-projection <0|1> Enable guarded closest-point projection after Stage A. Default: 1\n"
      << "  --relocation-mls <0|1>    Enable MLS projection inside relocation. Default: 0\n"
      << "  --gt <mesh>                Compute area-sampled Chamfer Distance to GT mesh\n"
      << "  --cd-samples <n>           Samples per CD direction. Default: 100000\n"
      << "  --summary-json <path>      Write JSON summary to file\n"
      << "  --quiet                    Suppress progress logs\n"
      << "  -h, --help                 Show this help\n";
}

std::string lowerString(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  return value;
}

bool parseBool(const std::string& value) {
  const std::string lower = lowerString(value);
  if (lower == "1" || lower == "true" || lower == "yes" || lower == "on") {
    return true;
  }
  if (lower == "0" || lower == "false" || lower == "no" || lower == "off") {
    return false;
  }
  throw std::runtime_error("expected boolean value 0/1, got: " + value);
}

std::string requireValue(int& i, int argc, char** argv, const std::string& option) {
  if (i + 1 >= argc) {
    throw std::runtime_error("missing value after " + option);
  }
  ++i;
  return argv[i];
}

std::filesystem::path defaultOutputPath(const std::filesystem::path& input) {
  const auto stem = input.stem().string();
  return input.parent_path() / (stem + "_cppadaiso.obj");
}

std::size_t parsePositiveSize(const std::string& value, const std::string& option) {
  const long long parsed = std::stoll(value);
  if (parsed <= 0) {
    throw std::runtime_error(option + " must be positive");
  }
  return static_cast<std::size_t>(parsed);
}

double parsePositiveDouble(const std::string& value, const std::string& option) {
  const double parsed = std::stod(value);
  if (!std::isfinite(parsed) || parsed <= 0.0) {
    throw std::runtime_error(option + " must be finite and positive");
  }
  return parsed;
}

double parseUnitIntervalDouble(const std::string& value, const std::string& option) {
  const double parsed = std::stod(value);
  if (!std::isfinite(parsed) || parsed < 0.0 || parsed > 1.0) {
    throw std::runtime_error(option + " must be finite and in [0, 1]");
  }
  return parsed;
}

std::string jsonEscape(const std::string& value) {
  std::ostringstream out;
  for (char c : value) {
    switch (c) {
      case '\\':
        out << "\\\\";
        break;
      case '"':
        out << "\\\"";
        break;
      case '\n':
        out << "\\n";
        break;
      case '\r':
        out << "\\r";
        break;
      case '\t':
        out << "\\t";
        break;
      default:
        out << c;
        break;
    }
  }
  return out.str();
}

std::string iterationsToJson(const std::vector<adaiso::AdaIsoStats::IterationStats>& iterations, int indent) {
  const std::string pad(static_cast<std::size_t>(indent), ' ');
  const std::string pad2(static_cast<std::size_t>(indent + 2), ' ');
  std::ostringstream out;
  out << std::setprecision(10);
  out << "[";
  if (!iterations.empty()) {
    out << "\n";
  }
  for (std::size_t i = 0; i < iterations.size(); ++i) {
    const auto& item = iterations[i];
    out << pad2 << "{\n";
    out << pad2 << "  \"stage\": \"" << jsonEscape(item.stage) << "\",\n";
    out << pad2 << "  \"iteration\": " << item.iteration << ",\n";
    out << pad2 << "  \"split_count\": " << item.split_count << ",\n";
    out << pad2 << "  \"collapse_count\": " << item.collapse_count << ",\n";
    out << pad2 << "  \"flip_count\": " << item.flip_count << ",\n";
    out << pad2 << "  \"relocation_count\": " << item.relocation_count << ",\n";
    out << pad2 << "  \"vertices\": " << item.vertices << ",\n";
    out << pad2 << "  \"faces\": " << item.faces << ",\n";
    out << pad2 << "  \"qavg\": " << item.qavg << ",\n";
    out << pad2 << "  \"qmin\": " << item.qmin << ",\n";
    out << pad2 << "  \"min_angle_degrees\": " << item.min_angle_degrees << ",\n";
    out << pad2 << "  \"angle_lt_10_pct\": " << item.angle_lt_10_pct << ",\n";
    out << pad2 << "  \"angle_lt_20_pct\": " << item.angle_lt_20_pct << ",\n";
    out << pad2 << "  \"mean_ratio\": " << item.mean_ratio << ",\n";
    out << pad2 << "  \"std_ratio\": " << item.std_ratio << ",\n";
    out << pad2 << "  \"p05_ratio\": " << item.p05_ratio << ",\n";
    out << pad2 << "  \"p95_ratio\": " << item.p95_ratio << ",\n";
    out << pad2 << "  \"violation_rate\": " << item.violation_rate << ",\n";
    out << pad2 << "  \"short_violation_rate\": " << item.short_violation_rate << ",\n";
    out << pad2 << "  \"long_violation_rate\": " << item.long_violation_rate << ",\n";
    out << pad2 << "  \"target_log_energy\": " << item.target_log_energy << "\n";
    out << pad2 << "}";
    if (i + 1 < iterations.size()) {
      out << ",";
    }
    out << "\n";
  }
  if (!iterations.empty()) {
    out << pad;
  }
  out << "]";
  return out.str();
}

std::string chamferToJson(const adaiso::ChamferDistanceSummary& summary, int indent) {
  const std::string pad(static_cast<std::size_t>(indent), ' ');
  const std::string pad2(static_cast<std::size_t>(indent + 2), ' ');
  std::ostringstream out;
  out << std::setprecision(10);
  out << pad << "{\n";
  out << pad2 << "\"method\": \"area_sampled\",\n";
  out << pad2 << "\"samples_per_direction\": " << summary.samples_per_direction << ",\n";
  out << pad2 << "\"output_to_gt_msd\": " << summary.output_to_gt_msd << ",\n";
  out << pad2 << "\"gt_to_output_msd\": " << summary.gt_to_output_msd << ",\n";
  out << pad2 << "\"chamfer_distance\": " << summary.chamfer_distance << ",\n";
  out << pad2 << "\"symmetric_rmse\": " << summary.symmetric_rmse << "\n";
  out << pad << "}";
  return out.str();
}

std::string buildSummaryJson(
    const std::filesystem::path& input_path,
    const std::filesystem::path& output_path,
    const std::filesystem::path& summary_path,
    const std::filesystem::path& gt_path,
    const adaiso::AdaIsoConfig& config,
    const adaiso::AdaIsoStats& stats,
    const adaiso::MeshSummary& input_summary,
    const adaiso::MeshSummary& output_summary,
    const adaiso::MeshSummary* gt_summary,
    const adaiso::ChamferDistanceSummary* chamfer_summary,
    double runtime_sec) {
  std::ostringstream out;
  out << std::setprecision(10);
  out << "{\n";
  out << "  \"input\": \"" << jsonEscape(input_path.string()) << "\",\n";
  out << "  \"output\": \"" << jsonEscape(output_path.string()) << "\",\n";
  if (!gt_path.empty()) {
    out << "  \"ground_truth\": \"" << jsonEscape(gt_path.string()) << "\",\n";
  }
  if (!summary_path.empty()) {
    out << "  \"summary_json\": \"" << jsonEscape(summary_path.string()) << "\",\n";
  }
  out << "  \"runtime_sec\": " << runtime_sec << ",\n";
  out << "  \"parameters\": {\n";
  out << "    \"variant\": \"" << adaiso::toString(config.variant) << "\",\n";
  out << "    \"curvature_mode\": \"" << adaiso::toString(config.curvature_mode) << "\",\n";
  out << "    \"target_lookup\": \"surface_barycentric\",\n";
  out << "    \"csf_alpha\": " << config.csf_smoothing_alpha << ",\n";
  out << "    \"target_error_delta\": " << stats.target_error_delta << ",\n";
  out << "    \"auto_delta_source\": \"input_resolution_csf_curvature\",\n";
  out << "    \"auto_delta_quantile\": " << config.auto_delta_quantile << ",\n";
  out << "    \"h_min_safe_factor\": " << stats.target_h_min_safe_factor << ",\n";
  out << "    \"target_error_bbox_ratio\": " << stats.target_error_bbox_ratio << ",\n";
  out << "    \"bbox_diagonal\": " << stats.bbox_diagonal << ",\n";
  out << "    \"h_min\": " << stats.target_h_min << ",\n";
  out << "    \"h_max\": " << stats.target_h_max << ",\n";
  out << "    \"h_max_factor\": " << config.target_h_max_factor << ",\n";
  out << "    \"estimated_faces\": " << stats.target_estimated_faces << ",\n";
  out << "    \"kappa_max\": " << stats.target_kappa_max << ",\n";
  out << "    \"flat_fallback\": " << (stats.target_flat_fallback ? "true" : "false") << ",\n";
  if (!config.target_field_export_dir.empty()) {
    out << "    \"target_field_export_dir\": \"" << jsonEscape(config.target_field_export_dir) << "\",\n";
  }
  if (!config.iteration_mesh_export_dir.empty()) {
    out << "    \"iteration_mesh_export_dir\": \"" << jsonEscape(config.iteration_mesh_export_dir) << "\",\n";
  }
  if (!config.stage_mesh_export_dir.empty()) {
    out << "    \"stage_mesh_export_dir\": \"" << jsonEscape(config.stage_mesh_export_dir) << "\",\n";
  }
  if (!config.all_stage_mesh_export_dir.empty()) {
    out << "    \"all_stage_mesh_export_dir\": \"" << jsonEscape(config.all_stage_mesh_export_dir) << "\",\n";
  }
  out << "    \"preprocess\": " << (config.preprocess ? "true" : "false") << ",\n";
  out << "    \"global_iterations\": " << config.global_iterations << ",\n";
  out << "    \"stage_a_policy\": \"" << adaiso::toString(config.stage_a_policy) << "\",\n";
  out << "    \"split_factor\": " << config.splitFactor() << ",\n";
  out << "    \"collapse_factor\": " << config.collapseFactor() << ",\n";
  out << "    \"split_min_angle_floor_degrees\": " << config.split_min_angle_floor_degrees << ",\n";
  out << "    \"tangent_lambda\": " << config.tangent_lambda << ",\n";
  out << "    \"max_flip_dihedral_degrees\": " << config.max_flip_dihedral_degrees << ",\n";
  out << "    \"min_angle_floor_degrees\": " << config.min_angle_floor_degrees << ",\n";
  out << "    \"feature_aware_collapse\": " << (config.feature_aware_collapse ? "true" : "false") << ",\n";
  out << "    \"collapse_feature_omega\": " << config.collapse_feature_omega << ",\n";
  out << "    \"stage_a_projection\": " << (config.stage_a_projection ? "true" : "false") << ",\n";
  out << "    \"relocation_mls\": " << (config.enable_relocation_mls ? "true" : "false") << ",\n";
  out << "    \"mls_k\": " << config.mls_k << ",\n";
  out << "    \"mls_iterations\": " << config.mls_iterations << "\n";
  out << "  },\n";
  out << "  \"input_summary\": " << adaiso::summaryToJson(input_summary, 2) << ",\n";
  out << "  \"output_summary\": " << adaiso::summaryToJson(output_summary, 2) << ",\n";
  out << "  \"iterations\": " << iterationsToJson(stats.iterations, 2) << ",\n";
  if (gt_summary != nullptr && chamfer_summary != nullptr) {
    out << "  \"gt_summary\": " << adaiso::summaryToJson(*gt_summary, 2) << ",\n";
    out << "  \"chamfer_distance\": " << chamferToJson(*chamfer_summary, 2) << ",\n";
  }
  out << "  \"stats\": {\n";
  out << "    \"vertices\": " << stats.vertices << ",\n";
  out << "    \"faces\": " << stats.faces << ",\n";
  out << "    \"size_stage_iterations\": " << stats.size_stage_iterations << ",\n";
  out << "    \"size_stage_splits\": " << stats.size_stage_splits << ",\n";
  out << "    \"size_stage_collapses\": " << stats.size_stage_collapses << ",\n";
  out << "    \"size_stage_long_edge_candidates\": " << stats.size_stage_long_edge_candidates << ",\n";
  out << "    \"size_stage_split_accepted\": " << stats.size_stage_split_accepted << ",\n";
  out << "    \"size_stage_split_rejected_by_angle\": " << stats.size_stage_split_rejected_by_angle << ",\n";
  out << "    \"size_stage_short_edge_candidates\": " << stats.size_stage_short_edge_candidates << ",\n";
  out << "    \"size_stage_collapse_accepted\": " << stats.size_stage_collapse_accepted << ",\n";
  out << "    \"size_stage_collapse_rejected\": " << stats.size_stage_collapse_rejected << ",\n";
  out << "    \"size_stage_collapse_rejected_by_boundary\": "
      << stats.size_stage_collapse_rejected_by_boundary << ",\n";
  out << "    \"size_stage_collapse_rejected_by_non_manifold\": "
      << stats.size_stage_collapse_rejected_by_non_manifold << ",\n";
  out << "    \"size_stage_collapse_rejected_by_batch_conflict\": "
      << stats.size_stage_collapse_rejected_by_batch_conflict << ",\n";
  out << "    \"size_stage_collapse_rejected_by_link_condition\": "
      << stats.size_stage_collapse_rejected_by_link_condition << ",\n";
  out << "    \"size_stage_collapse_rejected_by_degenerate_patch\": "
      << stats.size_stage_collapse_rejected_by_degenerate_patch << ",\n";
  out << "    \"size_stage_collapse_rejected_by_target_length\": "
      << stats.size_stage_collapse_rejected_by_target_length << ",\n";
  out << "    \"size_stage_collapse_rejected_by_angle\": "
      << stats.size_stage_collapse_rejected_by_angle << ",\n";
  out << "    \"size_stage_collapse_rejected_by_batch_duplicate\": "
      << stats.size_stage_collapse_rejected_by_batch_duplicate << ",\n";
  out << "    \"size_stage_collapse_conflict_mini_rounds\": "
      << stats.size_stage_collapse_conflict_mini_rounds << ",\n";
  out << "    \"size_stage_collapse_conflict_retry_candidates\": "
      << stats.size_stage_collapse_conflict_retry_candidates << ",\n";
  out << "    \"size_stage_collapse_conflict_retry_accepted\": "
      << stats.size_stage_collapse_conflict_retry_accepted << ",\n";
  out << "    \"size_stage_collapse_conflict_retry_rejected\": "
      << stats.size_stage_collapse_conflict_retry_rejected << ",\n";
  out << "    \"size_stage_final_violation\": " << stats.size_stage_final_violation << ",\n";
  out << "    \"size_stage_final_short_violation\": " << stats.size_stage_final_short_violation << ",\n";
  out << "    \"size_stage_final_long_violation\": " << stats.size_stage_final_long_violation << ",\n";
  out << "    \"size_stage_final_target_log_energy\": " << stats.size_stage_final_target_log_energy << ",\n";
  out << "    \"size_stage_final_min_angle_degrees\": " << stats.size_stage_final_min_angle_degrees << ",\n";
  out << "    \"size_stage_final_angle_lt_10_pct\": " << stats.size_stage_final_angle_lt_10_pct << ",\n";
  out << "    \"size_stage_final_angle_lt_20_pct\": " << stats.size_stage_final_angle_lt_20_pct << ",\n";
  out << "    \"stage_a_projection_candidates\": " << stats.stage_a_projection_candidates << ",\n";
  out << "    \"stage_a_projection_accepted\": " << stats.stage_a_projection_accepted << ",\n";
  out << "    \"stage_a_projection_rejected_by_invalid\": "
      << stats.stage_a_projection_rejected_by_invalid << ",\n";
  out << "    \"stage_a_projection_rejected_by_size_energy\": "
      << stats.stage_a_projection_rejected_by_size_energy << ",\n";
  out << "    \"stage_a_projection_rejected_by_angle\": "
      << stats.stage_a_projection_rejected_by_angle << ",\n";
  out << "    \"stage_a_projection_mean_displacement\": "
      << stats.stage_a_projection_mean_displacement << ",\n";
  out << "    \"stage_a_projection_max_displacement\": "
      << stats.stage_a_projection_max_displacement << ",\n";
  out << "    \"quality_stage_iterations\": " << stats.quality_stage_iterations << ",\n";
  out << "    \"quality_stage_flips\": " << stats.quality_stage_flips << ",\n";
  out << "    \"quality_stage_relocations\": " << stats.quality_stage_relocations << ",\n";
  out << "    \"csf_assembly_time_ms\": " << stats.csf_assembly_time_ms << ",\n";
  out << "    \"csf_factorization_time_ms\": " << stats.csf_factorization_time_ms << ",\n";
  out << "    \"csf_solve_time_ms\": " << stats.csf_solve_time_ms << ",\n";
  out << "    \"csf_residual_abs\": " << stats.csf_residual_abs << ",\n";
  out << "    \"csf_residual_scaled\": " << stats.csf_residual_scaled << ",\n";
  out << "    \"csf_matrix_nonzeros\": " << stats.csf_matrix_nonzeros << ",\n";
  out << "    \"csf_solver_success\": " << (stats.csf_solver_success ? "true" : "false") << "\n";
  out << "  }\n";
  out << "}\n";
  return out.str();
}

void writeTextFile(const std::filesystem::path& path, const std::string& text) {
  if (path.has_parent_path()) {
    std::filesystem::create_directories(path.parent_path());
  }
  std::ofstream out(path);
  if (!out) {
    throw std::runtime_error("failed to write file: " + path.string());
  }
  out << text;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    adaiso::AdaIsoConfig config;
    config.variant = adaiso::Variant::Code;

    std::filesystem::path input_path;
    std::filesystem::path output_path;
    std::filesystem::path summary_path;
    std::filesystem::path gt_path;
    std::size_t cd_samples = 100000;

    if (argc <= 1) {
      printUsage(std::cerr);
      return 1;
    }

    for (int i = 1; i < argc; ++i) {
      const std::string arg = argv[i];
      if (arg == "-h" || arg == "--help") {
        printUsage(std::cout);
        return 0;
      }
      if (arg == "-o" || arg == "--output") {
        output_path = requireValue(i, argc, argv, arg);
      } else if (arg == "--variant") {
        config.variant = adaiso::parseVariant(requireValue(i, argc, argv, arg));
      } else if (arg == "--curvature-mode") {
        config.curvature_mode = adaiso::parseCurvatureMode(requireValue(i, argc, argv, arg));
      } else if (arg == "--csf-alpha") {
        config.csf_smoothing_alpha = std::stod(requireValue(i, argc, argv, arg));
      } else if (arg == "--target-error-delta") {
        config.target_error_delta = std::stod(requireValue(i, argc, argv, arg));
      } else if (arg == "--auto-delta-quantile") {
        config.auto_delta_quantile = parseUnitIntervalDouble(requireValue(i, argc, argv, arg), arg);
      } else if (arg == "--target-h-max-factor") {
        config.target_h_max_factor = parsePositiveDouble(requireValue(i, argc, argv, arg), arg);
      } else if (arg == "--export-target-field") {
        config.target_field_export_dir = requireValue(i, argc, argv, arg);
      } else if (arg == "--export-iteration-meshes") {
        config.iteration_mesh_export_dir = requireValue(i, argc, argv, arg);
      } else if (arg == "--export-stage-meshes") {
        config.stage_mesh_export_dir = requireValue(i, argc, argv, arg);
      } else if (arg == "--export-all-stage-meshes") {
        config.all_stage_mesh_export_dir = requireValue(i, argc, argv, arg);
      } else if (arg == "--preprocess") {
        config.preprocess = parseBool(requireValue(i, argc, argv, arg));
      } else if (arg == "--iters") {
        config.global_iterations = std::stoi(requireValue(i, argc, argv, arg));
      } else if (arg == "--stage-a-policy") {
        config.stage_a_policy = adaiso::parseStageAPolicy(requireValue(i, argc, argv, arg));
      } else if (arg == "--split-factor") {
        config.split_factor_override = parsePositiveDouble(requireValue(i, argc, argv, arg), arg);
      } else if (arg == "--collapse-factor") {
        config.collapse_factor_override = parsePositiveDouble(requireValue(i, argc, argv, arg), arg);
      } else if (arg == "--max-flip-dihedral-deg") {
        config.max_flip_dihedral_degrees = std::stod(requireValue(i, argc, argv, arg));
      } else if (arg == "--feature-aware-collapse") {
        config.feature_aware_collapse = parseBool(requireValue(i, argc, argv, arg));
      } else if (arg == "--collapse-feature-omega") {
        config.collapse_feature_omega = std::stod(requireValue(i, argc, argv, arg));
      } else if (arg == "--stage-a-projection") {
        config.stage_a_projection = parseBool(requireValue(i, argc, argv, arg));
      } else if (arg == "--relocation-mls") {
        config.enable_relocation_mls = parseBool(requireValue(i, argc, argv, arg));
      } else if (arg == "--gt") {
        gt_path = requireValue(i, argc, argv, arg);
      } else if (arg == "--cd-samples") {
        cd_samples = parsePositiveSize(requireValue(i, argc, argv, arg), arg);
      } else if (arg == "--summary-json") {
        summary_path = requireValue(i, argc, argv, arg);
      } else if (arg == "--quiet") {
        config.verbose = false;
      } else if (!arg.empty() && arg[0] == '-') {
        throw std::runtime_error("unknown option: " + arg);
      } else if (input_path.empty()) {
        input_path = arg;
      } else {
        throw std::runtime_error("unexpected positional argument: " + arg);
      }
    }

    if (input_path.empty()) {
      throw std::runtime_error("missing input mesh path");
    }
    if (output_path.empty()) {
      output_path = defaultOutputPath(input_path);
    }
    if (config.global_iterations < 0) {
      throw std::runtime_error("--iters must be non-negative");
    }
    if (config.csf_smoothing_alpha <= 0.0 || !std::isfinite(config.csf_smoothing_alpha)) {
      throw std::runtime_error("--csf-alpha must be positive");
    }
    if (config.target_error_delta < 0.0 || !std::isfinite(config.target_error_delta)) {
      throw std::runtime_error("--target-error-delta must be finite and non-negative");
    }
    if (config.target_h_max_factor <= 0.0 || !std::isfinite(config.target_h_max_factor)) {
      throw std::runtime_error("--target-h-max-factor must be finite and positive");
    }
    if (!std::isfinite(config.auto_delta_quantile) ||
        config.auto_delta_quantile < 0.0 ||
        config.auto_delta_quantile > 1.0) {
      throw std::runtime_error("--auto-delta-quantile must be finite and in [0, 1]");
    }
    if (!std::isfinite(config.min_angle_floor_degrees) ||
        config.min_angle_floor_degrees < 0.0 ||
        config.min_angle_floor_degrees > 60.0) {
      throw std::runtime_error("min angle floor must be finite and in [0, 60]");
    }
    if (!std::isfinite(config.split_min_angle_floor_degrees) ||
        config.split_min_angle_floor_degrees < 0.0 ||
        config.split_min_angle_floor_degrees > 60.0) {
      throw std::runtime_error("split min angle floor must be finite and in [0, 60]");
    }
    if (config.split_factor_override < 0.0 || !std::isfinite(config.split_factor_override)) {
      throw std::runtime_error("--split-factor must be finite and non-negative");
    }
    if (config.collapse_factor_override < 0.0 || !std::isfinite(config.collapse_factor_override)) {
      throw std::runtime_error("--collapse-factor must be finite and non-negative");
    }
    if (config.collapse_feature_omega < 0.0 || !std::isfinite(config.collapse_feature_omega)) {
      throw std::runtime_error("--collapse-feature-omega must be finite and non-negative");
    }

    const auto started = std::chrono::steady_clock::now();
    adaiso::Mesh input = adaiso::loadMesh(input_path);
    const adaiso::MeshSummary input_summary = adaiso::summarizeMesh(adaiso::removeDegenerateFaces(input));

    adaiso::AdaIsoRemesher remesher(config);
    adaiso::Mesh output = remesher.remesh(input);
    output.validate();
    adaiso::writeMesh(output_path, output);

    const adaiso::MeshSummary output_summary = adaiso::summarizeMesh(output);
    adaiso::MeshSummary gt_summary;
    adaiso::ChamferDistanceSummary chamfer_summary;
    const adaiso::MeshSummary* gt_summary_ptr = nullptr;
    const adaiso::ChamferDistanceSummary* chamfer_summary_ptr = nullptr;
    if (!gt_path.empty()) {
      adaiso::Mesh gt = adaiso::removeDegenerateFaces(adaiso::loadMesh(gt_path));
      gt.validate();
      gt_summary = adaiso::summarizeMesh(gt);
      chamfer_summary = adaiso::areaSampledChamferDistance(output, gt, cd_samples);
      gt_summary_ptr = &gt_summary;
      chamfer_summary_ptr = &chamfer_summary;
    }

    const auto ended = std::chrono::steady_clock::now();
    const double runtime_sec = std::chrono::duration<double>(ended - started).count();
    const std::string json = buildSummaryJson(
        input_path,
        output_path,
        summary_path,
        gt_path,
        config,
        remesher.stats(),
        input_summary,
        output_summary,
        gt_summary_ptr,
        chamfer_summary_ptr,
        runtime_sec);

    if (!summary_path.empty()) {
      writeTextFile(summary_path, json);
    }
    std::cout << json;
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "adaiso_cli: " << e.what() << "\n";
    return 2;
  }
}
