#pragma once

#include "adaiso/Mesh.hpp"

#include <vector>

namespace adaiso {
namespace detail {

enum class CurvatureSignalMode {
  LogDimensionless,
  Dimensionless,
  Physical,
};

struct RobustMeanCurvatureFields {
  std::vector<double> physical_mean_curvature;
  std::vector<double> raw_signal;
};

[[nodiscard]] RobustMeanCurvatureFields computeRobustMeanCurvatureFields(
    const Mesh& mesh,
    CurvatureSignalMode mode);

[[nodiscard]] std::vector<double> computeRobustMeanCurvatureField(
    const Mesh& mesh);

}  // namespace detail
}  // namespace adaiso
