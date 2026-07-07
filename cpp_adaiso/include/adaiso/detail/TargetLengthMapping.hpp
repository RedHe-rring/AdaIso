#pragma once

#include <vector>

namespace adaiso::detail {

[[nodiscard]] std::vector<double> linearTargetLengths(
    const std::vector<double>& curvature,
    double base_length,
    double mesh_scale,
    double target_min_factor,
    double target_max_factor);

}  // namespace adaiso::detail
