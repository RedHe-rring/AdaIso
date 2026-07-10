#pragma once

#include "adaiso/Mesh.hpp"

#include <cstddef>
#include <vector>

namespace adaiso::detail {

struct TargetFaceSizingResult {
  std::vector<double> target_lengths;
  double lambda = 0.0;
  double estimated_faces = 0.0;
  double h_min = 0.0;
  double h_max = 0.0;
  std::size_t target_faces = 0;
};

[[nodiscard]] double estimateTargetFaceCount(
    const Mesh& mesh,
    const std::vector<double>& target_lengths);

[[nodiscard]] TargetFaceSizingResult targetFaceCurvatureLengths(
    const Mesh& mesh,
    const std::vector<double>& diffused_curvature,
    std::size_t target_faces,
    double mean_edge_length);

}  // namespace adaiso::detail
