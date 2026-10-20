#include "adaiso/IO.hpp"
#include "adaiso/Mesh.hpp"
#include "adaiso/Metrics.hpp"

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace {

void printUsage(std::ostream& out) {
  out << "Usage:\n"
      << "  adaiso_eval --mesh <mesh.obj|mesh.ply|mesh.off> --gt <mesh.obj|mesh.ply|mesh.off> [options]\n\n"
      << "Options:\n"
      << "  --cd-samples <n>    Samples per CD direction. Default: 100000\n"
      << "  --json <path>       Write JSON metrics to file\n"
      << "  -h, --help          Show this help\n";
}

std::string requireValue(int& i, int argc, char** argv, const std::string& option) {
  if (i + 1 >= argc) {
    throw std::runtime_error("missing value after " + option);
  }
  ++i;
  return argv[i];
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

std::string buildMetricsJson(
    const std::filesystem::path& mesh_path,
    const std::filesystem::path& gt_path,
    const std::filesystem::path& json_path,
    const adaiso::MeshSummary& mesh_summary,
    const adaiso::MeshSummary& gt_summary,
    const adaiso::ChamferDistanceSummary& chamfer_summary) {
  std::ostringstream out;
  out << std::setprecision(10);
  out << "{\n";
  out << "  \"mesh\": \"" << jsonEscape(mesh_path.string()) << "\",\n";
  out << "  \"ground_truth\": \"" << jsonEscape(gt_path.string()) << "\",\n";
  if (!json_path.empty()) {
    out << "  \"metrics_json\": \"" << jsonEscape(json_path.string()) << "\",\n";
  }
  out << "  \"cd_samples\": " << chamfer_summary.samples_per_direction << ",\n";
  out << "  \"mesh_summary\": " << adaiso::summaryToJson(mesh_summary, 2) << ",\n";
  out << "  \"gt_summary\": " << adaiso::summaryToJson(gt_summary, 2) << ",\n";
  out << "  \"chamfer_distance\": " << chamferToJson(chamfer_summary, 2) << "\n";
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
    std::filesystem::path mesh_path;
    std::filesystem::path gt_path;
    std::filesystem::path json_path;
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
      if (arg == "--mesh") {
        mesh_path = requireValue(i, argc, argv, arg);
      } else if (arg == "--gt") {
        gt_path = requireValue(i, argc, argv, arg);
      } else if (arg == "--cd-samples") {
        cd_samples = parsePositiveSize(requireValue(i, argc, argv, arg), arg);
      } else if (arg == "--json") {
        json_path = requireValue(i, argc, argv, arg);
      } else {
        throw std::runtime_error("unknown option: " + arg);
      }
    }

    if (mesh_path.empty()) {
      throw std::runtime_error("missing --mesh path");
    }
    if (gt_path.empty()) {
      throw std::runtime_error("missing --gt path");
    }

    const adaiso::Mesh mesh = adaiso::removeDegenerateFaces(adaiso::loadMesh(mesh_path));
    mesh.validate();
    const adaiso::Mesh gt = adaiso::removeDegenerateFaces(adaiso::loadMesh(gt_path));
    gt.validate();

    const adaiso::MeshSummary mesh_summary = adaiso::summarizeMesh(mesh);
    const adaiso::MeshSummary gt_summary = adaiso::summarizeMesh(gt);
    const adaiso::ChamferDistanceSummary chamfer_summary = adaiso::areaSampledChamferDistance(mesh, gt, cd_samples);
    const std::string json = buildMetricsJson(mesh_path, gt_path, json_path, mesh_summary, gt_summary, chamfer_summary);

    if (!json_path.empty()) {
      writeTextFile(json_path, json);
    }
    std::cout << json;
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "adaiso_eval: " << e.what() << "\n";
    return 2;
  }
}
