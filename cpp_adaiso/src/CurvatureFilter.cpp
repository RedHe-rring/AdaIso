#include "adaiso/detail/CurvatureFilter.hpp"

#include "adaiso/Geometry.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <unordered_set>
#include <vector>

namespace adaiso::detail {
namespace {

double median(std::vector<double> values) {
  if (values.empty()) {
    throw std::runtime_error("curvature outlier median requires samples");
  }
  std::sort(values.begin(), values.end());
  const std::size_t middle = values.size() / 2;
  if (values.size() % 2 == 0) {
    return 0.5 * (values[middle - 1] + values[middle]);
  }
  return values[middle];
}

std::vector<int> twoRing(
    std::size_t vertex,
    const std::vector<std::unordered_set<int>>& neighbors) {
  std::unordered_set<int> gathered = neighbors[vertex];
  for (int adjacent : neighbors[vertex]) {
    const auto& second = neighbors[static_cast<std::size_t>(adjacent)];
    gathered.insert(second.begin(), second.end());
  }
  gathered.erase(static_cast<int>(vertex));
  std::vector<int> result(gathered.begin(), gathered.end());
  std::sort(result.begin(), result.end());
  return result;
}

}  // namespace

CurvatureFilterResult filterCurvatureOutliers(
    const Mesh& mesh,
    const PrincipalCurvatureResult& principal,
    double mean_edge_length,
    CurvatureOutlierMode mode,
    std::size_t top_k,
    double sampling_cap_beta) {
  mesh.validate();
  if (principal.kappa.size() != mesh.vertexCount() ||
      principal.fit_diagnostics.size() != mesh.vertexCount()) {
    throw std::runtime_error("curvature filter field size mismatch");
  }
  if (!std::isfinite(mean_edge_length) || mean_edge_length <= 0.0) {
    throw std::runtime_error("curvature filter requires a positive mean edge length");
  }
  if (mode == CurvatureOutlierMode::SamplingCap &&
      (!std::isfinite(sampling_cap_beta) || sampling_cap_beta <= 0.0)) {
    throw std::runtime_error("curvature sampling cap beta must be positive and finite");
  }

  CurvatureFilterResult result;
  result.filtered_kappa = principal.kappa;
  result.applied_caps.assign(mesh.vertexCount(),
                             std::numeric_limits<double>::infinity());
  for (double value : principal.kappa) {
    if (!std::isfinite(value) || value < 0.0) {
      throw std::runtime_error("curvature filter requires finite non-negative values");
    }
    result.raw_max = std::max(result.raw_max, value);
  }

  if (mode == CurvatureOutlierMode::TopKMedian && top_k > 0) {
    std::vector<std::size_t> order(mesh.vertexCount(), 0);
    for (std::size_t i = 0; i < order.size(); ++i) {
      order[i] = i;
    }
    std::sort(order.begin(), order.end(), [&](std::size_t lhs, std::size_t rhs) {
      if (principal.kappa[lhs] != principal.kappa[rhs]) {
        return principal.kappa[lhs] > principal.kappa[rhs];
      }
      return lhs < rhs;
    });
    const auto neighbors = vertexNeighbors(mesh);
    const std::size_t count = std::min(top_k, order.size());
    for (std::size_t rank = 0; rank < count; ++rank) {
      const std::size_t vertex = order[rank];
      std::vector<double> samples;
      for (int adjacent : twoRing(vertex, neighbors)) {
        samples.push_back(principal.kappa[static_cast<std::size_t>(adjacent)]);
      }
      if (samples.empty()) {
        continue;
      }
      const double replacement = median(std::move(samples));
      result.applied_caps[vertex] = replacement;
      if (replacement < result.filtered_kappa[vertex]) {
        result.filtered_kappa[vertex] = replacement;
        ++result.changed_vertices;
      }
    }
  } else if (mode == CurvatureOutlierMode::SamplingCap) {
    for (std::size_t i = 0; i < result.filtered_kappa.size(); ++i) {
      const double local_scale = principal.fit_diagnostics[i].local_scale;
      if (!std::isfinite(local_scale) || local_scale <= 0.0) {
        continue;
      }
      const double cap = sampling_cap_beta * mean_edge_length / local_scale;
      if (!std::isfinite(cap) || cap <= 0.0) {
        throw std::runtime_error("curvature sampling cap produced an invalid value");
      }
      result.applied_caps[i] = cap;
      if (result.filtered_kappa[i] > cap) {
        result.filtered_kappa[i] = cap;
        ++result.changed_vertices;
      }
    }
  }

  for (double value : result.filtered_kappa) {
    result.filtered_max = std::max(result.filtered_max, value);
  }
  return result;
}

}  // namespace adaiso::detail
