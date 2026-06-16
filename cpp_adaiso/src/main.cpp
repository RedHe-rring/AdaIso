#include "adaiso/AdaIsoRemesher.hpp"
#include "adaiso/IO.hpp"
#include "adaiso/Metrics.hpp"

#include <algorithm>
#include <chrono>
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
      << "  adaiso_cli <input.obj|input.ply|input.off> -o <output.obj> [options]\n\n"
      << "Options:\n"
      << "  --mesh-scale <value>       Default: 1.0\n"
      << "  --variant <code|paper>     Default: code\n"
      << "  --preprocess <0|1>         Default: 0\n"
      << "  --iters <n>                Default: 5\n"
      << "  --project <0|1>            Default: 1\n"
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

std::string buildSummaryJson(
    const std::filesystem::path& input_path,
    const std::filesystem::path& output_path,
    const std::filesystem::path& summary_path,
    const adaiso::AdaIsoConfig& config,
    const adaiso::AdaIsoStats& stats,
    const adaiso::MeshSummary& input_summary,
    const adaiso::MeshSummary& output_summary,
    double runtime_sec) {
  std::ostringstream out;
  out << std::setprecision(10);
  out << "{\n";
  out << "  \"input\": \"" << jsonEscape(input_path.string()) << "\",\n";
  out << "  \"output\": \"" << jsonEscape(output_path.string()) << "\",\n";
  if (!summary_path.empty()) {
    out << "  \"summary_json\": \"" << jsonEscape(summary_path.string()) << "\",\n";
  }
  out << "  \"runtime_sec\": " << runtime_sec << ",\n";
  out << "  \"parameters\": {\n";
  out << "    \"mesh_scale\": " << config.mesh_scale << ",\n";
  out << "    \"variant\": \"" << adaiso::toString(config.variant) << "\",\n";
  out << "    \"preprocess\": " << (config.preprocess ? "true" : "false") << ",\n";
  out << "    \"global_iterations\": " << config.global_iterations << ",\n";
  out << "    \"project_to_surface\": " << (config.project_to_surface ? "true" : "false") << ",\n";
  out << "    \"histogram_bins\": " << config.histogram_bins << ",\n";
  out << "    \"split_factor\": " << config.splitFactor() << ",\n";
  out << "    \"collapse_factor\": " << config.collapseFactor() << ",\n";
  out << "    \"tangent_lambda\": " << config.tangent_lambda << ",\n";
  out << "    \"mls_k\": " << config.mls_k << ",\n";
  out << "    \"mls_iterations\": " << config.mls_iterations << "\n";
  out << "  },\n";
  out << "  \"input_summary\": " << adaiso::summaryToJson(input_summary, 2) << ",\n";
  out << "  \"output_summary\": " << adaiso::summaryToJson(output_summary, 2) << ",\n";
  out << "  \"stats\": {\n";
  out << "    \"vertices\": " << stats.vertices << ",\n";
  out << "    \"faces\": " << stats.faces << ",\n";
  out << "    \"csf_iterations\": " << stats.csf_iterations << "\n";
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
      } else if (arg == "--mesh-scale") {
        config.mesh_scale = std::stod(requireValue(i, argc, argv, arg));
      } else if (arg == "--variant") {
        config.variant = adaiso::parseVariant(requireValue(i, argc, argv, arg));
      } else if (arg == "--preprocess") {
        config.preprocess = parseBool(requireValue(i, argc, argv, arg));
      } else if (arg == "--iters") {
        config.global_iterations = std::stoi(requireValue(i, argc, argv, arg));
      } else if (arg == "--project") {
        config.project_to_surface = parseBool(requireValue(i, argc, argv, arg));
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
    if (config.mesh_scale <= 0.0) {
      throw std::runtime_error("--mesh-scale must be positive");
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
    adaiso::writeObj(output_path, output);

    const auto ended = std::chrono::steady_clock::now();
    const double runtime_sec = std::chrono::duration<double>(ended - started).count();
    const adaiso::MeshSummary output_summary = adaiso::summarizeMesh(output);
    const std::string json = buildSummaryJson(
        input_path,
        output_path,
        summary_path,
        config,
        remesher.stats(),
        input_summary,
        output_summary,
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
