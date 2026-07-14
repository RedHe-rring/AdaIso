#include "adaiso/detail/TargetFaceSizing.hpp"

#include "adaiso/Geometry.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace adaiso::detail {
namespace {

void validateSizingInputs(
    const Mesh& mesh,
    std::size_t target_faces,
    int bisection_iterations) {
  mesh.validate();
  if (mesh.vertices.empty() || mesh.faces.empty()) {
    throw std::runtime_error("target face sizing requires a non-empty mesh");
  }
  if (target_faces == 0) {
    throw std::runtime_error("target face count must be positive");
  }
  if (bisection_iterations < 0) {
    throw std::runtime_error("target bisection iterations must be non-negative");
  }
}

std::vector<double> uniformLengths(std::size_t count, double value) {
  return std::vector<double>(count, value);
}

}  // namespace

double estimateTargetFaceCount(
    const Mesh& mesh,
    const std::vector<double>& target_lengths) {
  mesh.validate();
  if (mesh.vertices.empty() || mesh.faces.empty()) {
    throw std::runtime_error("target face estimate requires a non-empty mesh");
  }
  if (target_lengths.size() != mesh.vertices.size()) {
    throw std::runtime_error("target face estimate length size mismatch");
  }

  double estimate = 0.0;
  for (const auto& face : mesh.faces) {
    const double area = triangleArea(mesh.vertices[face[0]], mesh.vertices[face[1]], mesh.vertices[face[2]]);
    if (!std::isfinite(area) || area <= 0.0) {
      throw std::runtime_error("target face estimate requires positive finite face areas");
    }
    const double h =
        (target_lengths[static_cast<std::size_t>(face[0])] +
         target_lengths[static_cast<std::size_t>(face[1])] +
         target_lengths[static_cast<std::size_t>(face[2])]) /
        3.0;
    if (!std::isfinite(h) || h <= 0.0) {
      throw std::runtime_error("target face estimate requires positive finite lengths");
    }
    estimate += area / (kEquilateralAreaFactor * h * h);
  }
  if (!std::isfinite(estimate)) {
    throw std::runtime_error("target face estimate produced a non-finite value");
  }
  return estimate;
}

InitialTargetSizingResult buildInitialPrincipalCurvatureLengths(
    const Mesh& mesh,
    const std::vector<double>& dimensionless_kappa,
    std::size_t target_faces,
    double mean_edge_length,
    int bisection_iterations) {
  validateSizingInputs(mesh, target_faces, bisection_iterations);
  if (dimensionless_kappa.size() != mesh.vertices.size()) {
    throw std::runtime_error("principal curvature sizing field size mismatch");
  }
  if (!std::isfinite(mean_edge_length) || mean_edge_length <= 0.0) {
    throw std::runtime_error("target face sizing requires a positive mean edge length");
  }

  std::vector<double> safe_kappa(dimensionless_kappa.size(), kTargetKappaMin);
  double max_kappa = kTargetKappaMin;
  for (std::size_t i = 0; i < dimensionless_kappa.size(); ++i) {
    const double value = dimensionless_kappa[i];
    if (!std::isfinite(value) || value < 0.0) {
      throw std::runtime_error("principal curvature sizing values must be finite and non-negative");
    }
    safe_kappa[i] = std::max(value, kTargetKappaMin);
    max_kappa = std::max(max_kappa, safe_kappa[i]);
  }

  InitialTargetSizingResult result;
  result.h_min = kTargetMinEdgeFactor * mean_edge_length;
  result.h_max = kTargetMaxEdgeFactor * mean_edge_length;
  result.target_faces = target_faces;
  if (!std::isfinite(result.h_min) || !std::isfinite(result.h_max) ||
      result.h_min <= 0.0 || result.h_max <= result.h_min) {
    throw std::runtime_error("target face sizing produced invalid length bounds");
  }

  auto lengthsForLambda = [&](double lambda) {
    std::vector<double> lengths(safe_kappa.size(), result.h_min);
    for (std::size_t i = 0; i < safe_kappa.size(); ++i) {
      const double h = lambda / std::pow(safe_kappa[i], kTargetCurvatureExponent);
      lengths[i] = std::clamp(h, result.h_min, result.h_max);
    }
    return lengths;
  };

  const auto min_lengths = uniformLengths(mesh.vertexCount(), result.h_min);
  const auto max_lengths = uniformLengths(mesh.vertexCount(), result.h_max);
  const double max_density_faces = estimateTargetFaceCount(mesh, min_lengths);
  const double min_density_faces = estimateTargetFaceCount(mesh, max_lengths);
  const double target = static_cast<double>(target_faces);
  const double hi_lambda = result.h_max * std::pow(max_kappa, kTargetCurvatureExponent);

  if (target >= max_density_faces) {
    result.lambda = 0.0;
    result.initial_lengths = min_lengths;
    result.estimated_faces = max_density_faces;
    return result;
  }
  if (target <= min_density_faces) {
    result.lambda = hi_lambda;
    result.initial_lengths = max_lengths;
    result.estimated_faces = min_density_faces;
    return result;
  }

  double lo = 0.0;
  double hi = hi_lambda;
  for (int iteration = 0; iteration < bisection_iterations; ++iteration) {
    const double mid = 0.5 * (lo + hi);
    const double estimate = estimateTargetFaceCount(mesh, lengthsForLambda(mid));
    if (estimate > target) {
      lo = mid;
    } else {
      hi = mid;
    }
  }
  result.lambda = 0.5 * (lo + hi);
  result.initial_lengths = lengthsForLambda(result.lambda);
  result.estimated_faces = estimateTargetFaceCount(mesh, result.initial_lengths);
  return result;
}

FinalTargetCalibrationResult calibrateTargetLengthsToFaceCount(
    const Mesh& mesh,
    const std::vector<double>& smoothed_lengths,
    std::size_t target_faces,
    double h_min,
    double h_max,
    int bisection_iterations) {
  validateSizingInputs(mesh, target_faces, bisection_iterations);
  if (smoothed_lengths.size() != mesh.vertexCount()) {
    throw std::runtime_error("target calibration field size mismatch");
  }
  if (!std::isfinite(h_min) || !std::isfinite(h_max) || h_min <= 0.0 || h_max <= h_min) {
    throw std::runtime_error("target calibration requires valid length bounds");
  }
  double min_smoothed = h_max;
  for (double value : smoothed_lengths) {
    if (!std::isfinite(value) || value <= 0.0) {
      throw std::runtime_error("target calibration lengths must be positive and finite");
    }
    min_smoothed = std::min(min_smoothed, value);
  }

  FinalTargetCalibrationResult result;
  result.target_faces = target_faces;
  const auto min_lengths = uniformLengths(mesh.vertexCount(), h_min);
  const auto max_lengths = uniformLengths(mesh.vertexCount(), h_max);
  const double max_density_faces = estimateTargetFaceCount(mesh, min_lengths);
  const double min_density_faces = estimateTargetFaceCount(mesh, max_lengths);
  const double target = static_cast<double>(target_faces);

  if (target >= max_density_faces) {
    result.scale = 0.0;
    result.target_lengths = min_lengths;
    result.estimated_faces = max_density_faces;
    result.target_faces_reachable = target <= max_density_faces;
    return result;
  }
  if (target <= min_density_faces) {
    result.scale = h_max / min_smoothed;
    result.target_lengths = max_lengths;
    result.estimated_faces = min_density_faces;
    result.target_faces_reachable = target >= min_density_faces;
    return result;
  }

  auto lengthsForScale = [&](double scale) {
    std::vector<double> lengths(smoothed_lengths.size(), h_min);
    for (std::size_t i = 0; i < smoothed_lengths.size(); ++i) {
      lengths[i] = std::clamp(scale * smoothed_lengths[i], h_min, h_max);
    }
    return lengths;
  };

  double lo = 0.0;
  double hi = h_max / min_smoothed;
  for (int iteration = 0; iteration < bisection_iterations; ++iteration) {
    const double mid = 0.5 * (lo + hi);
    const double estimate = estimateTargetFaceCount(mesh, lengthsForScale(mid));
    if (estimate > target) {
      lo = mid;
    } else {
      hi = mid;
    }
  }
  result.scale = 0.5 * (lo + hi);
  result.target_lengths = lengthsForScale(result.scale);
  result.estimated_faces = estimateTargetFaceCount(mesh, result.target_lengths);
  result.target_faces_reachable = true;
  return result;
}

}  // namespace adaiso::detail
