#pragma once

#include "adaiso/Mesh.hpp"

#include <cstddef>
#include <vector>

namespace adaiso::detail {

constexpr double kScreenedLaplacianEta = 1e-4;
constexpr double kMinScreenedEdgeLength = 1e-12;
constexpr double kScreenedResidualThreshold = 1e-10;

struct ScreenedNeighbor {
  int vertex = -1;
  double weight = 0.0;
};

using ScreenedAdjacency = std::vector<std::vector<ScreenedNeighbor>>;

struct ScreenedLaplacianStats {
  double assembly_time_ms = 0.0;
  double factorization_time_ms = 0.0;
  double solve_time_ms = 0.0;
  double residual_abs = 0.0;
  double residual_scaled = 0.0;
  std::size_t matrix_nonzeros = 0;
  bool solver_success = false;
};

struct ScreenedLaplacianResult {
  std::vector<double> values;
  ScreenedLaplacianStats stats;
};

[[nodiscard]] ScreenedAdjacency buildScreenedWeightGraph(const Mesh& mesh);

[[nodiscard]] ScreenedLaplacianResult solveScreenedScalarField(
    const Mesh& mesh,
    const std::vector<double>& initial_values,
    double smoothing_alpha,
    double residual_threshold = kScreenedResidualThreshold);

}  // namespace adaiso::detail
