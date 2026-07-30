#pragma once

#include "adaiso/Mesh.hpp"

#include <vector>

namespace adaiso {

namespace detail {

struct TargetErrorSizingResult {
  std::vector<double> target_lengths;
  double estimated_faces = 0.0;
  double h_min = 0.0;
  double h_max = 0.0;
  double target_error_delta = 0.0;
  double target_error_bbox_ratio = 0.0;
  double bbox_diagonal = 0.0;
  double kappa_max = 0.0;
  double auto_delta_quantile = 0.5;
  double h_min_safe_factor = 0.25;
  std::vector<double> local_edge_medians;
  std::vector<double> local_deltas;
  bool flat_fallback = false;
};

[[nodiscard]] double estimateTargetFaceCount(
    const Mesh& mesh,
    const std::vector<double>& target_lengths);

[[nodiscard]] TargetErrorSizingResult targetErrorCurvatureLengths(
    const Mesh& mesh,
    const std::vector<double>& diffused_curvature,
    double mean_edge_length,
    double bbox_diagonal,
    double target_error_delta,
    double target_h_max_factor,
    double auto_delta_quantile);

}  // namespace detail
}  // namespace adaiso
