#pragma once

#include "adaiso/Mesh.hpp"

#include <cstddef>
#include <vector>

namespace adaiso::detail {

inline constexpr double kPrincipalFitMinSingularRatio = 1e-6;

struct PrincipalCurvatureResult {
  std::vector<double> k1;
  std::vector<double> k2;
  std::vector<double> kappa;
  std::size_t fallback_vertices = 0;
};

[[nodiscard]] PrincipalCurvatureResult computePrincipalCurvatures(
    const Mesh& mesh,
    double mean_edge_length);

}  // namespace adaiso::detail
