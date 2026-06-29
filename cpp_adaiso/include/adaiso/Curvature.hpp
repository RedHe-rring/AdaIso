#pragma once

#include "adaiso/AdaIsoRemesher.hpp"
#include "adaiso/Mesh.hpp"
#include "adaiso/detail/CsfSolver.hpp"

#include <cstdint>
#include <filesystem>
#include <vector>

namespace adaiso {

struct CurvatureFields {
  std::vector<double> raw;
  std::vector<double> diffused;
  detail::CsfSolveStats stats;
};

struct ScalarRange {
  double min = 0.0;
  double max = 0.0;
};

struct Rgb8 {
  std::uint8_t r = 0;
  std::uint8_t g = 0;
  std::uint8_t b = 0;
};

[[nodiscard]] std::vector<double> computeRawCurvature(const Mesh& mesh, Variant variant);
[[nodiscard]] CurvatureFields computeCurvatureFields(const Mesh& mesh, Variant variant);
[[nodiscard]] ScalarRange scalarRange(const std::vector<double>& values);
[[nodiscard]] Rgb8 blueGreenRedColor(double value, double min_value, double max_value);
[[nodiscard]] std::vector<Rgb8> colorizeScalars(
    const std::vector<double>& values,
    double min_value,
    double max_value);
void writeColoredAsciiPly(
    const std::filesystem::path& path,
    const Mesh& mesh,
    const std::vector<Rgb8>& colors);

}  // namespace adaiso
