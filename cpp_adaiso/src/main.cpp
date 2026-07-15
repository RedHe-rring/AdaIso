#include "adaiso/AdaIsoRemesher.hpp"
#include "adaiso/IO.hpp"
#include "adaiso/Metrics.hpp"
#include "adaiso/detail/TargetFaceSizing.hpp"

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
      << "  --curvature-outlier-mode <none|topk-median|sampling-cap> Default: none\n"
      << "  --curvature-outlier-top-k <n> Top-k diagnostic replacements. Default: 1\n"
      << "  --curvature-cap-beta <v>  Sampling-aware curvature cap. Default: 5\n"
      << "  --curvature-smoothing-alpha <value> Screened curvature smoothing strength. Default: 0.75; 0 disables\n"
      << "  --target-gradation <0|1>   Enable downward log-size gradation. Default: 1\n"
      << "  --target-gradation-ratio <value> Maximum size ratio per mean-edge distance. Default: 1.30\n"
      << "  --target-face-representative <mean|min> Face h used for calibration. Default: mean\n"
      << "  --target-max-factor <value> Maximum target h / mean edge length. Default: 5\n"
      << "  --target-faces <n>         Target face budget. Default: cleaned input face count\n"
      << "  --target-bisection-iters <n> Target sizing/calibration bisection iterations. Default: 64\n"
      << "  --export-target-field <dir> Export target-field CSV/PLY diagnostics\n"
      << "  --export-iteration-meshes <dir> Export per-iteration remesh snapshots\n"
      << "  --export-stage-meshes <dir> Export first-iteration remesh stage snapshots\n"
      << "  --export-all-stage-meshes <dir> Export every iteration's remesh stage snapshots\n"
      << "  --preprocess <0|1>         Default: 0\n"
      << "  --iters <n>                Default: 5\n"
      << "  --max-flip-dihedral-deg <v> Maximum allowed flip dihedral angle. Default: 45\n"
      << "  --feature-aware-collapse <0|1> Feature-aware collapse placement. Default: 1\n"
      << "  --collapse-feature-omega <v> Feature similarity threshold for collapse. Default: 0.15\n"
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

adaiso::CurvatureOutlierMode parseCurvatureOutlierMode(const std::string& value) {
  const std::string lower = lowerString(value);
  if (lower == "none") {
    return adaiso::CurvatureOutlierMode::None;
  }
  if (lower == "topk-median" || lower == "topk_median") {
    return adaiso::CurvatureOutlierMode::TopKMedian;
  }
  if (lower == "sampling-cap" || lower == "sampling_cap") {
    return adaiso::CurvatureOutlierMode::SamplingCap;
  }
  throw std::runtime_error("unknown curvature outlier mode: " + value);
}

const char* curvatureOutlierModeName(adaiso::CurvatureOutlierMode mode) {
  switch (mode) {
    case adaiso::CurvatureOutlierMode::None:
      return "none";
    case adaiso::CurvatureOutlierMode::TopKMedian:
      return "topk_median";
    case adaiso::CurvatureOutlierMode::SamplingCap:
      return "sampling_cap";
  }
  return "unknown";
}

adaiso::TargetFaceRepresentative parseTargetFaceRepresentative(
    const std::string& value) {
  const std::string lower = lowerString(value);
  if (lower == "mean") {
    return adaiso::TargetFaceRepresentative::Mean;
  }
  if (lower == "min" || lower == "minimum") {
    return adaiso::TargetFaceRepresentative::Minimum;
  }
  throw std::runtime_error("unknown target face representative: " + value);
}

const char* targetFaceRepresentativeName(adaiso::TargetFaceRepresentative value) {
  return value == adaiso::TargetFaceRepresentative::Minimum ? "min" : "mean";
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
    out << pad2 << "  \"iteration\": " << item.iteration << ",\n";
    out << pad2 << "  \"split_count\": " << item.split_count << ",\n";
    out << pad2 << "  \"collapse_count\": " << item.collapse_count << ",\n";
    out << pad2 << "  \"flip_count\": " << item.flip_count << ",\n";
    out << pad2 << "  \"vertices\": " << item.vertices << ",\n";
    out << pad2 << "  \"faces\": " << item.faces << ",\n";
    out << pad2 << "  \"qavg\": " << item.qavg << ",\n";
    out << pad2 << "  \"qmin\": " << item.qmin << ",\n";
    out << pad2 << "  \"mean_ratio\": " << item.mean_ratio << ",\n";
    out << pad2 << "  \"std_ratio\": " << item.std_ratio << ",\n";
    out << pad2 << "  \"p05_ratio\": " << item.p05_ratio << ",\n";
    out << pad2 << "  \"p95_ratio\": " << item.p95_ratio << ",\n";
    out << pad2 << "  \"violation_rate\": " << item.violation_rate << "\n";
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
  out << "    \"principal_curvature_estimator\": \"weighted_quadratic_patch\",\n";
  out << "    \"principal_curvature_scalar\": \"mean_edge_length_times_max_abs_k1_k2\",\n";
  out << "    \"target_lookup\": \"surface_barycentric\",\n";
  out << "    \"target_size_mapping\": \"principal_curvature_screened_kappa_inverse_power_log_downward_gradation\",\n";
  out << "    \"curvature_outlier_mode\": \""
      << curvatureOutlierModeName(config.curvature_outlier_mode) << "\",\n";
  out << "    \"curvature_outlier_top_k\": " << config.curvature_outlier_top_k << ",\n";
  out << "    \"curvature_cap_beta\": " << config.curvature_cap_beta << ",\n";
  out << "    \"curvature_smoothing_alpha\": " << config.curvature_smoothing_alpha << ",\n";
  out << "    \"target_gradation_enabled\": "
      << (config.enable_target_gradation ? "true" : "false") << ",\n";
  out << "    \"target_gradation_ratio\": " << config.target_gradation_ratio << ",\n";
  out << "    \"target_face_representative\": \""
      << targetFaceRepresentativeName(config.target_face_representative) << "\",\n";
  out << "    \"target_max_factor\": " << config.target_max_factor << ",\n";
  out << "    \"target_faces\": "
      << (config.target_faces > 0 ? config.target_faces : input_summary.faces) << ",\n";
  out << "    \"target_curvature_exponent\": " << adaiso::detail::kTargetCurvatureExponent << ",\n";
  out << "    \"target_bisection_iterations\": "
      << config.target_bisection_iterations << ",\n";
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
  out << "    \"split_factor\": " << config.splitFactor() << ",\n";
  out << "    \"collapse_factor\": " << config.collapseFactor() << ",\n";
  out << "    \"tangent_lambda\": " << config.tangent_lambda << ",\n";
  out << "    \"max_flip_dihedral_degrees\": " << config.max_flip_dihedral_degrees << ",\n";
  out << "    \"feature_aware_collapse\": " << (config.feature_aware_collapse ? "true" : "false") << ",\n";
  out << "    \"collapse_feature_omega\": " << config.collapse_feature_omega << ",\n";
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
  out << "    \"curvature_smoothing_applied\": "
      << (stats.curvature_smoothing_applied ? "true" : "false") << ",\n";
  out << "    \"curvature_smoothing_assembly_time_ms\": "
      << stats.curvature_smoothing_assembly_time_ms << ",\n";
  out << "    \"curvature_smoothing_factorization_time_ms\": "
      << stats.curvature_smoothing_factorization_time_ms << ",\n";
  out << "    \"curvature_smoothing_solve_time_ms\": "
      << stats.curvature_smoothing_solve_time_ms << ",\n";
  out << "    \"curvature_smoothing_residual_abs\": "
      << stats.curvature_smoothing_residual_abs << ",\n";
  out << "    \"curvature_smoothing_residual_scaled\": "
      << stats.curvature_smoothing_residual_scaled << ",\n";
  out << "    \"curvature_smoothing_matrix_nonzeros\": "
      << stats.curvature_smoothing_matrix_nonzeros << ",\n";
  out << "    \"curvature_smoothing_solver_success\": "
      << (stats.curvature_smoothing_solver_success ? "true" : "false") << ",\n";
  out << "    \"principal_curvature_fallback_vertices\": "
      << stats.principal_curvature_fallback_vertices << ",\n";
  out << "    \"principal_curvature_fallback_rate\": "
      << stats.principal_curvature_fallback_rate << ",\n";
  out << "    \"principal_fit_min_singular_ratio\": "
      << stats.principal_fit_min_singular_ratio << ",\n";
  out << "    \"principal_fit_max_residual_rms\": "
      << stats.principal_fit_max_residual_rms << ",\n";
  out << "    \"principal_fit_max_curvature_times_local_scale\": "
      << stats.principal_fit_max_curvature_times_local_scale << ",\n";
  out << "    \"curvature_filter_changed_vertices\": "
      << stats.curvature_filter_changed_vertices << ",\n";
  out << "    \"curvature_filter_changed_vertex_rate\": "
      << stats.curvature_filter_changed_vertex_rate << ",\n";
  out << "    \"raw_kappa_max\": " << stats.raw_kappa_max << ",\n";
  out << "    \"filtered_kappa_max\": " << stats.filtered_kappa_max << ",\n";
  out << "    \"smoothed_kappa_max\": " << stats.smoothed_kappa_max << ",\n";
  out << "    \"pre_limiter_estimated_faces\": " << stats.pre_limiter_estimated_faces << ",\n";
  out << "    \"post_limiter_estimated_faces\": " << stats.post_limiter_estimated_faces << ",\n";
  out << "    \"gradation_runtime_ms\": " << stats.gradation_runtime_ms << ",\n";
  out << "    \"gradation_queue_pops\": " << stats.gradation_queue_pops << ",\n";
  out << "    \"gradation_relaxations\": " << stats.gradation_relaxations << ",\n";
  out << "    \"gradation_changed_vertices\": " << stats.gradation_changed_vertices << ",\n";
  out << "    \"gradation_changed_vertex_rate\": " << stats.gradation_changed_vertex_rate << ",\n";
  out << "    \"gradation_min_log_reduction\": " << stats.gradation_min_log_reduction << ",\n";
  out << "    \"gradation_mean_log_reduction\": " << stats.gradation_mean_log_reduction << ",\n";
  out << "    \"gradation_max_log_reduction\": " << stats.gradation_max_log_reduction << ",\n";
  out << "    \"gradation_max_normalized_log_slope_before\": "
      << stats.gradation_max_normalized_log_slope_before << ",\n";
  out << "    \"gradation_max_normalized_log_slope_after\": "
      << stats.gradation_max_normalized_log_slope_after << ",\n";
  out << "    \"gradation_p95_normalized_log_slope_after\": "
      << stats.gradation_p95_normalized_log_slope_after << ",\n";
  out << "    \"gradation_p99_normalized_log_slope_after\": "
      << stats.gradation_p99_normalized_log_slope_after << ",\n";
  out << "    \"gradation_max_adjacent_ratio_before\": "
      << stats.gradation_max_adjacent_ratio_before << ",\n";
  out << "    \"gradation_max_adjacent_ratio_after\": "
      << stats.gradation_max_adjacent_ratio_after << ",\n";
  out << "    \"final_gradation_max_normalized_log_slope\": "
      << stats.final_gradation_max_normalized_log_slope << ",\n";
  out << "    \"final_gradation_p95_normalized_log_slope\": "
      << stats.final_gradation_p95_normalized_log_slope << ",\n";
  out << "    \"final_gradation_p99_normalized_log_slope\": "
      << stats.final_gradation_p99_normalized_log_slope << ",\n";
  out << "    \"final_gradation_max_adjacent_ratio\": "
      << stats.final_gradation_max_adjacent_ratio << ",\n";
  out << "    \"final_estimated_faces\": " << stats.final_estimated_faces << ",\n";
  out << "    \"final_estimated_faces_mean\": "
      << stats.final_estimated_faces_mean << ",\n";
  out << "    \"final_estimated_faces_min\": "
      << stats.final_estimated_faces_min << ",\n";
  out << "    \"final_min_mean_estimate_ratio\": "
      << stats.final_min_mean_estimate_ratio << ",\n";
  out << "    \"final_face_estimate_relative_error\": "
      << stats.final_face_estimate_relative_error << ",\n";
  out << "    \"target_h_min_saturation_rate\": "
      << stats.target_h_min_saturation_rate << ",\n";
  out << "    \"target_h_max_saturation_rate\": "
      << stats.target_h_max_saturation_rate << ",\n";
  out << "    \"smallest_five_percent_face_budget_rate\": "
      << stats.smallest_five_percent_face_budget_rate << ",\n";
  out << "    \"input_edge_target_violation_rate\": "
      << stats.input_edge_target_violation_rate << ",\n";
  out << "    \"target_scale\": " << stats.target_scale << ",\n";
  out << "    \"target_faces_reachable\": " << (stats.target_faces_reachable ? "true" : "false") << "\n";
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
      } else if (arg == "--curvature-outlier-mode") {
        config.curvature_outlier_mode =
            parseCurvatureOutlierMode(requireValue(i, argc, argv, arg));
      } else if (arg == "--curvature-outlier-top-k") {
        config.curvature_outlier_top_k =
            parsePositiveSize(requireValue(i, argc, argv, arg), arg);
      } else if (arg == "--curvature-cap-beta") {
        config.curvature_cap_beta = std::stod(requireValue(i, argc, argv, arg));
      } else if (arg == "--curvature-smoothing-alpha") {
        config.curvature_smoothing_alpha = std::stod(requireValue(i, argc, argv, arg));
      } else if (arg == "--target-gradation") {
        config.enable_target_gradation = parseBool(requireValue(i, argc, argv, arg));
      } else if (arg == "--target-gradation-ratio") {
        config.target_gradation_ratio = std::stod(requireValue(i, argc, argv, arg));
      } else if (arg == "--target-face-representative") {
        config.target_face_representative =
            parseTargetFaceRepresentative(requireValue(i, argc, argv, arg));
      } else if (arg == "--target-max-factor") {
        config.target_max_factor = std::stod(requireValue(i, argc, argv, arg));
      } else if (arg == "--target-faces") {
        config.target_faces = parsePositiveSize(requireValue(i, argc, argv, arg), arg);
      } else if (arg == "--target-bisection-iters") {
        config.target_bisection_iterations = std::stoi(requireValue(i, argc, argv, arg));
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
      } else if (arg == "--max-flip-dihedral-deg") {
        config.max_flip_dihedral_degrees = std::stod(requireValue(i, argc, argv, arg));
      } else if (arg == "--feature-aware-collapse") {
        config.feature_aware_collapse = parseBool(requireValue(i, argc, argv, arg));
      } else if (arg == "--collapse-feature-omega") {
        config.collapse_feature_omega = std::stod(requireValue(i, argc, argv, arg));
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
    if (config.curvature_smoothing_alpha < 0.0 ||
        !std::isfinite(config.curvature_smoothing_alpha)) {
      throw std::runtime_error("--curvature-smoothing-alpha must be finite and non-negative");
    }
    if (!std::isfinite(config.target_gradation_ratio) || config.target_gradation_ratio <= 1.0) {
      throw std::runtime_error("--target-gradation-ratio must be finite and greater than one");
    }
    if (config.target_bisection_iterations < 0) {
      throw std::runtime_error("--target-bisection-iters must be non-negative");
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
