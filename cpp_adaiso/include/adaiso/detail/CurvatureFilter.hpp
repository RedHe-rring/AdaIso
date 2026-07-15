#pragma once

#include "adaiso/Mesh.hpp"
#include "adaiso/TargetSizingOptions.hpp"
#include "adaiso/detail/PrincipalCurvature.hpp"

#include <cstddef>
#include <vector>

namespace adaiso::detail {

struct CurvatureFilterResult {
  std::vector<double> filtered_kappa;
  std::vector<double> applied_caps;
  std::size_t changed_vertices = 0;
  double raw_max = 0.0;
  double filtered_max = 0.0;
};

[[nodiscard]] CurvatureFilterResult filterCurvatureOutliers(
    const Mesh& mesh,
    const PrincipalCurvatureResult& principal,
    double mean_edge_length,
    CurvatureOutlierMode mode,
    std::size_t top_k,
    double sampling_cap_beta);

}  // namespace adaiso::detail
