#include "adaiso/detail/TargetFaceSizing.hpp"

#include "adaiso/Geometry.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <unordered_set>
#include <vector>

namespace adaiso::detail {
namespace {

constexpr double kLogSmoothEtaBase = 0.35;
constexpr double kLogSmoothTau = 0.35;
constexpr int kLogSmoothIterations = 1;

void validateMesh(const Mesh& mesh) {
  mesh.validate();
  if (mesh.vertices.empty() || mesh.faces.empty()) {
    throw std::runtime_error("target face sizing requires a non-empty mesh");
  }
}

double quantileOfSorted(const std::vector<double>& sorted, double q) {
  if (sorted.empty()) {
    return 0.0;
  }
  if (sorted.size() == 1) {
    return sorted.front();
  }
  q = std::clamp(q, 0.0, 1.0);
  const double pos = q * static_cast<double>(sorted.size() - 1);
  const auto lo = static_cast<std::size_t>(std::floor(pos));
  const auto hi = static_cast<std::size_t>(std::ceil(pos));
  const double t = pos - static_cast<double>(lo);
  return sorted[lo] * (1.0 - t) + sorted[hi] * t;
}

double medianValue(std::vector<double> values) {
  if (values.empty()) {
    return 0.0;
  }
  std::sort(values.begin(), values.end());
  return quantileOfSorted(values, 0.5);
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

CurvatureCapResult capCurvatureQuantile(
    const std::vector<double>& curvature,
    double quantile) {
  if (curvature.empty()) {
    throw std::runtime_error("curvature cap requires a non-empty field");
  }
  std::vector<double> finite_values;
  finite_values.reserve(curvature.size());
  for (double value : curvature) {
    if (!std::isfinite(value)) {
      throw std::runtime_error("curvature cap values must be finite");
    }
    finite_values.push_back(std::max(value, kTargetKappaMin));
  }
  std::sort(finite_values.begin(), finite_values.end());
  const double cap_value = std::max(
      kTargetKappaMin,
      quantileOfSorted(finite_values, quantile));

  CurvatureCapResult result;
  result.curvature.reserve(curvature.size());
  result.cap_value = cap_value;
  std::size_t changed = 0;
  for (double value : curvature) {
    const double safe = std::max(value, kTargetKappaMin);
    const double capped = std::min(safe, cap_value);
    if (capped < safe) {
      ++changed;
    }
    result.curvature.push_back(capped);
  }
  result.changed_rate =
      static_cast<double>(changed) / static_cast<double>(curvature.size());
  return result;
}

std::vector<double> smoothLogDomainSmallSizeOutliers(
    const Mesh& mesh,
    const std::vector<double>& target_lengths,
    double h_min,
    double h_max,
    std::size_t target_faces) {
  validateMesh(mesh);
  if (target_lengths.size() != mesh.vertices.size()) {
    throw std::runtime_error("log-domain smoothing length size mismatch");
  }
  if (!std::isfinite(h_min) || !std::isfinite(h_max) ||
      h_min <= 0.0 || h_max <= h_min) {
    throw std::runtime_error("log-domain smoothing requires valid h bounds");
  }
  if (target_faces == 0) {
    throw std::runtime_error("log-domain smoothing requires target faces");
  }

  std::vector<double> x(target_lengths.size(), 0.0);
  for (std::size_t i = 0; i < target_lengths.size(); ++i) {
    const double h = target_lengths[i];
    if (!std::isfinite(h) || h <= 0.0) {
      throw std::runtime_error("log-domain smoothing requires positive lengths");
    }
    x[i] = std::log(std::clamp(h, h_min, h_max));
  }

  const auto neigh = vertexNeighbors(mesh);
  for (int iter = 0; iter < kLogSmoothIterations; ++iter) {
    std::vector<double> next = x;
    for (std::size_t i = 0; i < x.size(); ++i) {
      if (neigh[i].empty()) {
        continue;
      }
      std::vector<double> neighbor_logs;
      neighbor_logs.reserve(neigh[i].size());
      double mean = 0.0;
      for (int n : neigh[i]) {
        const double value = x[static_cast<std::size_t>(n)];
        neighbor_logs.push_back(value);
        mean += value;
      }
      mean /= static_cast<double>(neighbor_logs.size());
      const double median = medianValue(std::move(neighbor_logs));
      const double outlier_weight =
          std::clamp((median - x[i] - kLogSmoothTau) / kLogSmoothTau, 0.0, 1.0);
      const double eta = kLogSmoothEtaBase * outlier_weight;
      next[i] = (1.0 - eta) * x[i] + eta * mean;
    }
    x = std::move(next);
  }

  std::vector<double> smoothed(x.size(), h_min);
  for (std::size_t i = 0; i < x.size(); ++i) {
    smoothed[i] = std::clamp(std::exp(x[i]), h_min, h_max);
  }

  const double estimate = estimateTargetFaceCount(mesh, smoothed);
  const double scale =
      std::sqrt(estimate / static_cast<double>(target_faces));
  if (!std::isfinite(scale) || scale <= 0.0) {
    throw std::runtime_error("log-domain smoothing produced invalid rescale");
  }
  for (double& value : smoothed) {
    value = std::clamp(scale * value, h_min, h_max);
  }
  return smoothed;
}

TargetFaceSizingResult targetFaceCurvatureLengths(
    const Mesh& mesh,
    const std::vector<double>& diffused_curvature,
    std::size_t target_faces,
    double mean_edge_length,
    double cap_quantile) {
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

  CurvatureCapResult cap;
  if (cap_quantile >= 0.0) {
    cap = capCurvatureQuantile(diffused_curvature, cap_quantile);
  } else {
    if (diffused_curvature.empty()) {
      throw std::runtime_error("curvature cap requires a non-empty field");
    }
    cap.curvature.reserve(diffused_curvature.size());
    cap.cap_value = 0.0;
    cap.changed_rate = 0.0;
    for (double value : diffused_curvature) {
      if (!std::isfinite(value)) {
        throw std::runtime_error("curvature cap values must be finite");
      }
      const double safe = std::max(value, kTargetKappaMin);
      cap.curvature.push_back(safe);
      cap.cap_value = std::max(cap.cap_value, safe);
    }
  }

  std::vector<double> safe_kappa(diffused_curvature.size(), kTargetKappaMin);
  double max_kappa = kTargetKappaMin;
  for (std::size_t i = 0; i < cap.curvature.size(); ++i) {
    const double value = cap.curvature[i];
    safe_kappa[i] = std::max(value, kTargetKappaMin);
    max_kappa = std::max(max_kappa, safe_kappa[i]);
  }

  TargetFaceSizingResult result;
  result.h_min = kTargetMinEdgeFactor * mean_edge_length;
  result.h_max = kTargetMaxEdgeFactor * mean_edge_length;
  result.curvature_cap_value = cap.cap_value;
  result.curvature_cap_changed_rate = cap.changed_rate;
  result.capped_curvature = cap.curvature;
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
