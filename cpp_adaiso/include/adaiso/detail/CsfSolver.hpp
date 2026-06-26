#pragma once

#include "adaiso/Mesh.hpp"

#include <cstddef>
#include <vector>

namespace adaiso::detail {

constexpr double kCsfEta = 1e-4;
constexpr double kMinEdgeLength = 1e-12;
constexpr double kResidualThreshold = 1e-10;

struct CsfNeighbor {
  int vertex = -1;
  double weight = 0.0;
};

using CsfAdjacency = std::vector<std::vector<CsfNeighbor>>;

struct CsfSolveStats {
  double assembly_time_ms = 0.0;
  double factorization_time_ms = 0.0;
  double solve_time_ms = 0.0;
  double residual_abs = 0.0;
  double residual_scaled = 0.0;
  std::size_t matrix_nonzeros = 0;
  bool solver_success = false;
};

struct CsfSolveResult {
  std::vector<double> values;
  CsfSolveStats stats;
};

[[nodiscard]] CsfAdjacency buildCsfWeightGraph(const Mesh& mesh);

[[nodiscard]] CsfSolveResult solveScreenedCsf(
    const Mesh& mesh,
    const std::vector<double>& initial_values,
    double smoothing_alpha,
    double residual_threshold = kResidualThreshold);

[[nodiscard]] CsfSolveResult solveScreenedScalarField(
    const Mesh& mesh,
    const std::vector<double>& initial_values,
    double smoothing_alpha,
    double residual_threshold = kResidualThreshold);

}  // namespace adaiso::detail
