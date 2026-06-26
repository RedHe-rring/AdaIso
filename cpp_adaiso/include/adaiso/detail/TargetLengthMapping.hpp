#pragma once

#include "adaiso/Mesh.hpp"

#include <vector>

namespace adaiso::detail {

[[nodiscard]] std::vector<double> continuousTargetLogLengths(
    const std::vector<double>& curvature,
    double base_length,
    double mesh_scale,
    double target_min_factor,
    double target_max_factor,
    double target_gamma);

void clampTargetLogLengths(
    std::vector<double>& log_lengths,
    double min_log_length,
    double max_log_length);

void projectTargetLogGradation(
    const Mesh& mesh,
    std::vector<double>& log_lengths,
    double min_log_length,
    double max_log_length,
    double max_adjacent_ratio,
    int iterations);

}  // namespace adaiso::detail
