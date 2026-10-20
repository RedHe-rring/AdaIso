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
  std::vector<double> capped_curvature;
  double lambda = 0.0;
  double estimated_faces = 0.0;
  double h_min = 0.0;
  double h_max = 0.0;
  double curvature_cap_value = 0.0;
  double curvature_cap_changed_rate = 0.0;
  std::size_t target_faces = 0;
};

struct CurvatureCapResult {
  std::vector<double> curvature;
  double cap_value = 0.0;
  double changed_rate = 0.0;
};

[[nodiscard]] double estimateTargetFaceCount(
    const Mesh& mesh,
    const std::vector<double>& target_lengths);

[[nodiscard]] CurvatureCapResult capCurvatureQuantile(
    const std::vector<double>& curvature,
    double quantile);

[[nodiscard]] std::vector<double> smoothLogDomainSmallSizeOutliers(
    const Mesh& mesh,
    const std::vector<double>& target_lengths,
    double h_min,
    double h_max,
    std::size_t target_faces);

[[nodiscard]] TargetFaceSizingResult targetFaceCurvatureLengths(
    const Mesh& mesh,
    const std::vector<double>& diffused_curvature,
    std::size_t target_faces,
    double mean_edge_length,
    double cap_quantile = 0.995);

}  // namespace adaiso::detail
