#pragma once

#include "adaiso/Mesh.hpp"

#include <cstddef>
#include <vector>

namespace adaiso::detail {

inline constexpr double kPrincipalFitMinSingularRatio = 1e-6;

struct PrincipalCurvatureFitDiagnostics {
  int sample_ring = 0;
  std::size_t sample_count = 0;
  double local_scale = 0.0;
  double singular_ratio = 0.0;
  double residual_rms = 0.0;
  bool fallback = true;
};

struct PrincipalCurvatureResult {
  std::vector<double> k1;
  std::vector<double> k2;
  std::vector<double> kappa;
  std::vector<PrincipalCurvatureFitDiagnostics> fit_diagnostics;
  std::size_t fallback_vertices = 0;
};

[[nodiscard]] PrincipalCurvatureResult computePrincipalCurvatures(
    const Mesh& mesh,
    double mean_edge_length);

}  // namespace adaiso::detail
