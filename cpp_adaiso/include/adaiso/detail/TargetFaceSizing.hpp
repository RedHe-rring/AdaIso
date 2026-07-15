#pragma once

#include "adaiso/Mesh.hpp"

#include <cstddef>
#include <vector>

namespace adaiso::detail {

inline constexpr double kTargetKappaMin = 1e-12;
inline constexpr double kTargetMinEdgeFactor = 0.05;
inline constexpr double kTargetMaxEdgeFactor = 5.0;
inline constexpr double kEquilateralAreaFactor = 0.43301270189221932338;
inline constexpr int kTargetLambdaBisectionIterations = 64;

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
