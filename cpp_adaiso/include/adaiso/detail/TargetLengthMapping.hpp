#pragma once

#include <vector>

namespace adaiso::detail {

[[nodiscard]] std::vector<double> continuousTargetLengths(
    const std::vector<double>& curvature,
    double base_length,
    double mesh_scale,
    double target_min_factor,
    double target_max_factor,
    double target_gamma);

}  // namespace adaiso::detail
