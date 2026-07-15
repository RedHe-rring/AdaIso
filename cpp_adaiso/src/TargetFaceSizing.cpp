#include "adaiso/detail/TargetFaceSizing.hpp"

#include "adaiso/Geometry.hpp"

#include <algorithm>
#include <cmath>
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

}  // namespace

double estimateTargetFaceCount(
    const Mesh& mesh,
    const std::vector<double>& target_lengths) {
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
    const double h = (h0 + h1 + h2) / 3.0;
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

TargetFaceSizingResult targetFaceCurvatureLengths(
    const Mesh& mesh,
    const std::vector<double>& diffused_curvature,
    std::size_t target_faces,
    double mean_edge_length) {
  validateMesh(mesh);
  if (diffused_curvature.size() != mesh.vertices.size()) {
    throw std::runtime_error("target face sizing curvature size mismatch");
  }
  if (target_faces == 0) {
    throw std::runtime_error("target face count must be positive");
  }
  if (!std::isfinite(mean_edge_length) || mean_edge_length <= 0.0) {
    throw std::runtime_error(
        "target face sizing requires a positive mean edge length");
  }

  std::vector<double> safe_kappa(diffused_curvature.size(), kTargetKappaMin);
  double max_kappa = kTargetKappaMin;
  for (std::size_t i = 0; i < diffused_curvature.size(); ++i) {
    const double value = diffused_curvature[i];
    if (!std::isfinite(value)) {
      throw std::runtime_error("target face sizing curvature values must be finite");
    }
    safe_kappa[i] = std::max(value, kTargetKappaMin);
    max_kappa = std::max(max_kappa, safe_kappa[i]);
  }

  TargetFaceSizingResult result;
  result.h_min = kTargetMinEdgeFactor * mean_edge_length;
  result.h_max = kTargetMaxEdgeFactor * mean_edge_length;
  result.target_faces = target_faces;
  if (!std::isfinite(result.h_min) || !std::isfinite(result.h_max) ||
      result.h_min <= 0.0 || result.h_max <= result.h_min) {
    throw std::runtime_error("target face sizing produced invalid length bounds");
  }

  auto lengths_for_lambda = [&](double lambda) {
    std::vector<double> lengths(safe_kappa.size(), result.h_min);
    for (std::size_t i = 0; i < safe_kappa.size(); ++i) {
      const double h = lambda / std::sqrt(safe_kappa[i]);
      lengths[i] = std::clamp(h, result.h_min, result.h_max);
    }
    return lengths;
  };

  const auto min_lengths = lengths_for_lambda(0.0);
  const double max_density_faces = estimateTargetFaceCount(mesh, min_lengths);
  const double hi_lambda = result.h_max * std::sqrt(max_kappa);
  const auto max_lengths = lengths_for_lambda(hi_lambda);
  const double min_density_faces = estimateTargetFaceCount(mesh, max_lengths);
  const double target = static_cast<double>(target_faces);

  if (target >= max_density_faces) {
    result.lambda = 0.0;
    result.target_lengths = min_lengths;
    result.estimated_faces = max_density_faces;
    return result;
  }
  if (target <= min_density_faces) {
    result.lambda = hi_lambda;
    result.target_lengths = max_lengths;
    result.estimated_faces = min_density_faces;
    return result;
  }

  double lo = 0.0;
  double hi = hi_lambda;
  for (int iter = 0; iter < kTargetLambdaBisectionIterations; ++iter) {
    const double mid = 0.5 * (lo + hi);
    const auto lengths = lengths_for_lambda(mid);
    const double estimate = estimateTargetFaceCount(mesh, lengths);
    if (estimate > target) {
      lo = mid;
    } else {
      hi = mid;
    }
  }

  result.lambda = 0.5 * (lo + hi);
  result.target_lengths = lengths_for_lambda(result.lambda);
  result.estimated_faces = estimateTargetFaceCount(mesh, result.target_lengths);
  return result;
}

}  // namespace adaiso::detail
