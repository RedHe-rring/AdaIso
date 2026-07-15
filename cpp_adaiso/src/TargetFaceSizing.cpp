#include "adaiso/detail/TargetFaceSizing.hpp"

#include "adaiso/Geometry.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace adaiso::detail {
namespace {

void validateMesh(const Mesh& mesh) {
  mesh.validate();
  if (mesh.vertices.empty() || mesh.faces.empty()) {
    throw std::runtime_error("target face sizing requires a non-empty mesh");
  }
}

void validateCalibrationInputs(const Mesh& mesh, std::size_t target_faces,
                               int bisection_iterations) {
  validateMesh(mesh);
  if (target_faces == 0) {
    throw std::runtime_error("target face count must be positive");
  }
  if (bisection_iterations < 0) {
    throw std::runtime_error(
        "target bisection iterations must be non-negative");
  }
}

std::vector<double> uniformLengths(std::size_t count, double value) {
  return std::vector<double>(count, value);
}

double relativeFaceError(double estimate, std::size_t target_faces) {
  return std::abs(estimate - static_cast<double>(target_faces)) /
         static_cast<double>(target_faces);
}

}  // namespace

double estimateTargetFaceCount(const Mesh& mesh,
                               const std::vector<double>& target_lengths,
                               TargetFaceRepresentative representative) {
  validateMesh(mesh);
  if (target_lengths.size() != mesh.vertices.size()) {
    throw std::runtime_error("target face estimate length size mismatch");
  }

  double estimate = 0.0;
  for (const auto& face : mesh.faces) {
    const double area = triangleArea(
        mesh.vertices[face[0]], mesh.vertices[face[1]], mesh.vertices[face[2]]);
    if (!std::isfinite(area) || area <= 0.0) {
      throw std::runtime_error(
          "target face estimate requires positive finite face areas");
    }
    const double h0 = target_lengths[static_cast<std::size_t>(face[0])];
    const double h1 = target_lengths[static_cast<std::size_t>(face[1])];
    const double h2 = target_lengths[static_cast<std::size_t>(face[2])];
    const double h = representative == TargetFaceRepresentative::Minimum
                         ? std::min({h0, h1, h2})
                         : (h0 + h1 + h2) / 3.0;
    if (!std::isfinite(h) || h <= 0.0) {
      throw std::runtime_error(
          "target face estimate requires positive finite lengths");
    }
    estimate += area / (kEquilateralAreaFactor * h * h);
  }
  if (!std::isfinite(estimate)) {
    throw std::runtime_error(
        "target face estimate produced a non-finite value");
  }
  return estimate;
}

InitialTargetSizingResult buildInitialPrincipalCurvatureLengths(
    const Mesh& mesh, const std::vector<double>& smoothed_dimensionless_kappa,
    double mean_edge_length, double target_max_factor) {
  validateMesh(mesh);
  if (smoothed_dimensionless_kappa.size() != mesh.vertices.size()) {
    throw std::runtime_error("principal curvature sizing field size mismatch");
  }
  if (!std::isfinite(mean_edge_length) || mean_edge_length <= 0.0) {
    throw std::runtime_error(
        "target face sizing requires a positive mean edge length");
  }

  InitialTargetSizingResult result;
  result.h_min = kTargetMinEdgeFactor * mean_edge_length;
  if (!std::isfinite(target_max_factor) || target_max_factor <= kTargetMinEdgeFactor) {
    throw std::runtime_error("target max factor must be finite and greater than the minimum factor");
  }
  result.h_max = target_max_factor * mean_edge_length;
  if (!std::isfinite(result.h_min) || !std::isfinite(result.h_max) ||
      result.h_min <= 0.0 || result.h_max <= result.h_min) {
    throw std::runtime_error(
        "target face sizing produced invalid length bounds");
  }

  result.initial_lengths.resize(smoothed_dimensionless_kappa.size(),
                                mean_edge_length);
  for (std::size_t i = 0; i < smoothed_dimensionless_kappa.size(); ++i) {
    const double kappa = smoothed_dimensionless_kappa[i];
    if (!std::isfinite(kappa) || kappa < 0.0) {
      throw std::runtime_error(
          "principal curvature sizing values must be finite and non-negative");
    }
    const double safe_kappa = std::max(kappa, kTargetKappaMin);
    const double length =
        mean_edge_length / std::pow(safe_kappa, kTargetCurvatureExponent);
    if (!std::isfinite(length) || length <= 0.0) {
      throw std::runtime_error(
          "principal curvature sizing produced an invalid length");
    }
    result.initial_lengths[i] = length;
  }
  return result;
}

FinalTargetCalibrationResult
calibrateTargetLengthsToFaceCount(const Mesh& mesh,
                                  const std::vector<double>& shape_lengths,
                                  std::size_t target_faces, double h_min,
                                  double h_max, int bisection_iterations,
                                  TargetFaceRepresentative representative) {
  validateCalibrationInputs(mesh, target_faces, bisection_iterations);
  if (shape_lengths.size() != mesh.vertexCount()) {
    throw std::runtime_error("target calibration field size mismatch");
  }
  if (!std::isfinite(h_min) || !std::isfinite(h_max) || h_min <= 0.0 ||
      h_max <= h_min) {
    throw std::runtime_error("target calibration requires valid length bounds");
  }

  double min_shape = std::numeric_limits<double>::infinity();
  for (double value : shape_lengths) {
    if (!std::isfinite(value) || value <= 0.0) {
      throw std::runtime_error(
          "target calibration lengths must be positive and finite");
    }
    min_shape = std::min(min_shape, value);
  }
  const double maximum_scale = h_max / min_shape;
  if (!std::isfinite(maximum_scale) || maximum_scale <= 0.0) {
    throw std::runtime_error(
        "target calibration produced an invalid scale bound");
  }

  FinalTargetCalibrationResult result;
  result.target_faces = target_faces;
  const auto min_lengths = uniformLengths(mesh.vertexCount(), h_min);
  const auto max_lengths = uniformLengths(mesh.vertexCount(), h_max);
  const double max_density_faces =
      estimateTargetFaceCount(mesh, min_lengths, representative);
  const double min_density_faces =
      estimateTargetFaceCount(mesh, max_lengths, representative);
  const double target = static_cast<double>(target_faces);

  if (target >= max_density_faces) {
    result.scale = 0.0;
    result.target_lengths = min_lengths;
    result.estimated_faces = max_density_faces;
    result.relative_error =
        relativeFaceError(result.estimated_faces, target_faces);
    result.target_faces_reachable = target <= max_density_faces;
    return result;
  }
  if (target <= min_density_faces) {
    result.scale = maximum_scale;
    result.target_lengths = max_lengths;
    result.estimated_faces = min_density_faces;
    result.relative_error =
        relativeFaceError(result.estimated_faces, target_faces);
    result.target_faces_reachable = target >= min_density_faces;
    return result;
  }

  auto lengthsForScale = [&](double scale) {
    std::vector<double> lengths(shape_lengths.size(), h_min);
    for (std::size_t i = 0; i < shape_lengths.size(); ++i) {
      lengths[i] = std::clamp(scale * shape_lengths[i], h_min, h_max);
    }
    return lengths;
  };

  double lo = 0.0;
  double hi = maximum_scale;
  for (int iteration = 0; iteration < bisection_iterations; ++iteration) {
    const double mid = 0.5 * (lo + hi);
    const double estimate =
        estimateTargetFaceCount(mesh, lengthsForScale(mid), representative);
    if (estimate > target) {
      lo = mid;
    } else {
      hi = mid;
    }
  }
  result.scale = 0.5 * (lo + hi);
  result.target_lengths = lengthsForScale(result.scale);
  result.estimated_faces =
      estimateTargetFaceCount(mesh, result.target_lengths, representative);
  result.relative_error =
      relativeFaceError(result.estimated_faces, target_faces);
  result.target_faces_reachable = true;
  return result;
}

}  // namespace adaiso::detail
