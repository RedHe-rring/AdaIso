#pragma once

#include "adaiso/Mesh.hpp"
#include "adaiso/TargetSizingOptions.hpp"

#include <cstddef>
#include <vector>

namespace adaiso::detail {

inline constexpr double kTargetKappaMin = 1e-12;
inline constexpr double kTargetMinEdgeFactor = 0.05;
inline constexpr double kTargetMaxEdgeFactor = 5.0;
inline constexpr double kTargetCurvatureExponent = 0.75;
inline constexpr double kEquilateralAreaFactor = 0.43301270189221932338;
inline constexpr int kDefaultTargetBisectionIterations = 64;

struct InitialTargetSizingResult {
  std::vector<double> initial_lengths;
  double h_min = 0.0;
  double h_max = 0.0;
};

struct FinalTargetCalibrationResult {
  std::vector<double> target_lengths;
  double scale = 1.0;
  double estimated_faces = 0.0;
  double relative_error = 0.0;
  std::size_t target_faces = 0;
  bool target_faces_reachable = true;
};

[[nodiscard]] double estimateTargetFaceCount(
    const Mesh& mesh,
    const std::vector<double>& target_lengths,
    TargetFaceRepresentative representative = TargetFaceRepresentative::Mean);

[[nodiscard]] InitialTargetSizingResult buildInitialPrincipalCurvatureLengths(
    const Mesh& mesh,
    const std::vector<double>& smoothed_dimensionless_kappa,
    double mean_edge_length,
    double target_max_factor = kTargetMaxEdgeFactor);

[[nodiscard]] FinalTargetCalibrationResult calibrateTargetLengthsToFaceCount(
    const Mesh& mesh,
    const std::vector<double>& shape_lengths,
    std::size_t target_faces,
    double h_min,
    double h_max,
    int bisection_iterations = kDefaultTargetBisectionIterations,
    TargetFaceRepresentative representative = TargetFaceRepresentative::Mean);

}  // namespace adaiso::detail
