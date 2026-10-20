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
      << "  --target-faces <n>         Target face budget. Default: cleaned input face count\n"
      << "  --curvature-signal <mode>  log_dimless|dimless|physical. Default: log_dimless\n"
      << "  --use-csf <0|1>            Use screened CSF smoothing. Default: 1\n"
      << "  --curvature-cap <q|off>    Curvature cap quantile or off. Default: 0.995\n"
      << "  --export-target-field <dir> Write target-field CSV diagnostics\n"
      << "  --preprocess <0|1>         Default: 0\n"
      << "  --iters <n>                Default: 5\n"
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

adaiso::AdaIsoConfig::CurvatureSignal parseCurvatureSignal(const std::string& value) {
  const std::string lower = lowerString(value);
  if (lower == "log_dimless" || lower == "log-dimless" || lower == "log") {
    return adaiso::AdaIsoConfig::CurvatureSignal::LogDimensionless;
  }
  if (lower == "dimless" || lower == "dimensionless") {
    return adaiso::AdaIsoConfig::CurvatureSignal::Dimensionless;
  }
  if (lower == "physical") {
    return adaiso::AdaIsoConfig::CurvatureSignal::Physical;
  }
  throw std::runtime_error("unknown curvature signal mode: " + value);
}

std::string curvatureSignalName(adaiso::AdaIsoConfig::CurvatureSignal mode) {
  switch (mode) {
    case adaiso::AdaIsoConfig::CurvatureSignal::LogDimensionless:
      return "log_dimless";
    case adaiso::AdaIsoConfig::CurvatureSignal::Dimensionless:
      return "dimless";
    case adaiso::AdaIsoConfig::CurvatureSignal::Physical:
      return "physical";
  }
  return "log_dimless";
}

double parseCurvatureCap(const std::string& value) {
  const std::string lower = lowerString(value);
  if (lower == "off" || lower == "none" || lower == "no") {
    return -1.0;
  }
  const double parsed = std::stod(value);
  if (!std::isfinite(parsed) || parsed <= 0.0 || parsed > 1.0) {
    throw std::runtime_error("--curvature-cap must be in (0,1] or off");
  }
  return parsed;
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
  out << "    \"target_lookup\": \"surface_barycentric\",\n";
  out << "    \"target_size_mapping\": \"target_faces_inverse_sqrt_"
      << curvatureSignalName(config.curvature_signal)
      << (config.use_csf ? "_csf" : "_raw")
      << (config.curvature_cap_quantile >= 0.0 ? "_cap" : "_nocap")
      << "\",\n";
  out << "    \"raw_curvature\": \"robust_mean\",\n";
  out << "    \"curvature_signal\": \"" << curvatureSignalName(config.curvature_signal) << "\",\n";
  out << "    \"use_csf\": " << (config.use_csf ? "true" : "false") << ",\n";
  if (config.curvature_cap_quantile >= 0.0) {
    out << "    \"curvature_cap_quantile\": " << config.curvature_cap_quantile << ",\n";
  } else {
    out << "    \"curvature_cap_quantile\": \"off\",\n";
  }
  if (!config.export_target_field_dir.empty()) {
    out << "    \"export_target_field\": \"" << jsonEscape(config.export_target_field_dir) << "\",\n";
  }
  out << "    \"target_faces\": "
      << (config.target_faces > 0 ? config.target_faces : input_summary.faces) << ",\n";
  out << "    \"preprocess\": " << (config.preprocess ? "true" : "false") << ",\n";
  out << "    \"global_iterations\": " << config.global_iterations << ",\n";
  out << "    \"split_factor\": " << config.splitFactor() << ",\n";
  out << "    \"collapse_factor\": " << config.collapseFactor() << ",\n";
  out << "    \"tangent_lambda\": " << config.tangent_lambda << "\n";
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
  out << "    \"target_lambda\": " << stats.target_lambda << ",\n";
  out << "    \"target_estimated_faces\": " << stats.target_estimated_faces << ",\n";
  out << "    \"target_h_min\": " << stats.target_h_min << ",\n";
  out << "    \"target_h_max\": " << stats.target_h_max << ",\n";
  out << "    \"target_observed_h_min\": " << stats.target_observed_h_min << ",\n";
  out << "    \"target_observed_h_max\": " << stats.target_observed_h_max << ",\n";
  out << "    \"curvature_cap_value\": " << stats.curvature_cap_value << ",\n";
  out << "    \"curvature_cap_changed_rate\": " << stats.curvature_cap_changed_rate << ",\n";
  out << "    \"split_total\": " << stats.split_total << ",\n";
  out << "    \"collapse_total\": " << stats.collapse_total << ",\n";
  out << "    \"flip_total\": " << stats.flip_total << ",\n";
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
      } else if (arg == "--curvature-signal") {
        config.curvature_signal = parseCurvatureSignal(requireValue(i, argc, argv, arg));
      } else if (arg == "--use-csf") {
        config.use_csf = parseBool(requireValue(i, argc, argv, arg));
      } else if (arg == "--curvature-cap") {
        config.curvature_cap_quantile = parseCurvatureCap(requireValue(i, argc, argv, arg));
      } else if (arg == "--export-target-field") {
        config.export_target_field_dir = requireValue(i, argc, argv, arg);
      } else if (arg == "--target-faces") {
        config.target_faces = parsePositiveSize(requireValue(i, argc, argv, arg), arg);
      } else if (arg == "--preprocess") {
        config.preprocess = parseBool(requireValue(i, argc, argv, arg));
      } else if (arg == "--iters") {
        config.global_iterations = std::stoi(requireValue(i, argc, argv, arg));
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
