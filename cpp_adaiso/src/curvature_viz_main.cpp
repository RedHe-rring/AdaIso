#include "adaiso/AdaIsoRemesher.hpp"
#include "adaiso/Curvature.hpp"
#include "adaiso/IO.hpp"
#include "adaiso/Mesh.hpp"

#include <algorithm>
#include <cctype>
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
      << "  adaiso_curvature_viz <input.obj|input.ply|input.off> --output-prefix <path-prefix> [options]\n\n"
      << "Options:\n"
      << "  --variant <code|paper>     Default: code\n"
      << "  --summary-json <path>      Write JSON summary to file\n"
      << "  -h, --help                 Show this help\n";
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

std::string requireValue(int& i, int argc, char** argv, const std::string& option) {
  if (i + 1 >= argc) {
    throw std::runtime_error("missing value after " + option);
  }
  ++i;
  return argv[i];
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

std::string buildSummaryJson(
    const std::filesystem::path& input_path,
    const std::filesystem::path& raw_path,
    const std::filesystem::path& diffused_path,
    const std::filesystem::path& summary_path,
    adaiso::Variant variant,
    const adaiso::Mesh& mesh,
    const adaiso::ScalarRange& raw_range,
    const adaiso::ScalarRange& diffused_range,
    const adaiso::detail::CsfSolveStats& stats) {
  std::ostringstream out;
  out << std::setprecision(10);
  out << "{\n";
  out << "  \"input\": \"" << jsonEscape(input_path.string()) << "\",\n";
  out << "  \"raw_output\": \"" << jsonEscape(raw_path.string()) << "\",\n";
  out << "  \"diffused_output\": \"" << jsonEscape(diffused_path.string()) << "\",\n";
  if (!summary_path.empty()) {
    out << "  \"summary_json\": \"" << jsonEscape(summary_path.string()) << "\",\n";
  }
  out << "  \"parameters\": {\n";
  out << "    \"variant\": \"" << adaiso::toString(variant) << "\",\n";
  out << "    \"color_map\": \"blue-green-red\",\n";
  out << "    \"normalization\": \"independent\"\n";
  out << "  },\n";
  out << "  \"mesh\": {\n";
  out << "    \"vertices\": " << mesh.vertexCount() << ",\n";
  out << "    \"faces\": " << mesh.faceCount() << "\n";
  out << "  },\n";
  out << "  \"raw\": {\n";
  out << "    \"min\": " << raw_range.min << ",\n";
  out << "    \"max\": " << raw_range.max << "\n";
  out << "  },\n";
  out << "  \"diffused\": {\n";
  out << "    \"min\": " << diffused_range.min << ",\n";
  out << "    \"max\": " << diffused_range.max << "\n";
  out << "  },\n";
  out << "  \"stats\": {\n";
  out << "    \"csf_assembly_time_ms\": " << stats.assembly_time_ms << ",\n";
  out << "    \"csf_factorization_time_ms\": " << stats.factorization_time_ms << ",\n";
  out << "    \"csf_solve_time_ms\": " << stats.solve_time_ms << ",\n";
  out << "    \"csf_residual_abs\": " << stats.residual_abs << ",\n";
  out << "    \"csf_residual_scaled\": " << stats.residual_scaled << ",\n";
  out << "    \"csf_free_vertices\": " << stats.free_vertices << ",\n";
  out << "    \"csf_fixed_vertices\": " << stats.fixed_vertices << ",\n";
  out << "    \"csf_matrix_nonzeros\": " << stats.matrix_nonzeros << ",\n";
  out << "    \"csf_solver_success\": " << (stats.solver_success ? "true" : "false") << "\n";
  out << "  }\n";
  out << "}\n";
  return out.str();
}

}  // namespace

int main(int argc, char** argv) {
  try {
    adaiso::Variant variant = adaiso::Variant::Code;
    std::filesystem::path input_path;
    std::filesystem::path output_prefix;
    std::filesystem::path summary_path;

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
      if (arg == "--output-prefix") {
        output_prefix = requireValue(i, argc, argv, arg);
      } else if (arg == "--variant") {
        variant = adaiso::parseVariant(requireValue(i, argc, argv, arg));
      } else if (arg == "--summary-json") {
        summary_path = requireValue(i, argc, argv, arg);
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
    if (output_prefix.empty()) {
      throw std::runtime_error("missing --output-prefix");
    }

    adaiso::Mesh mesh = adaiso::removeDegenerateFaces(adaiso::loadMesh(input_path));
    mesh.validate();
    const adaiso::CurvatureFields fields = adaiso::computeCurvatureFields(mesh, variant);
    const adaiso::ScalarRange raw_range = adaiso::scalarRange(fields.raw);
    const adaiso::ScalarRange diffused_range = adaiso::scalarRange(fields.diffused);

    const std::filesystem::path raw_path = output_prefix.string() + "_raw.ply";
    const std::filesystem::path diffused_path = output_prefix.string() + "_diffused.ply";
    adaiso::writeColoredAsciiPly(
        raw_path,
        mesh,
        adaiso::colorizeScalars(fields.raw, raw_range.min, raw_range.max));
    adaiso::writeColoredAsciiPly(
        diffused_path,
        mesh,
        adaiso::colorizeScalars(fields.diffused, diffused_range.min, diffused_range.max));

    const std::string json = buildSummaryJson(
        input_path,
        raw_path,
        diffused_path,
        summary_path,
        variant,
        mesh,
        raw_range,
        diffused_range,
        fields.stats);
    if (!summary_path.empty()) {
      writeTextFile(summary_path, json);
    }
    std::cout << json;
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "adaiso_curvature_viz: " << e.what() << "\n";
    return 2;
  }
}
