#include "adaiso/detail/TargetGradation.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <functional>
#include <limits>
#include <numeric>
#include <queue>
#include <stdexcept>
#include <utility>

namespace adaiso::detail {
namespace {

using Clock = std::chrono::steady_clock;
using Neighbor = std::pair<int, double>;
using Adjacency = std::vector<std::vector<Neighbor>>;

void validateLengths(const Mesh& mesh, const std::vector<double>& lengths,
                     double mean_edge_length) {
  mesh.validate();
  if (mesh.vertices.empty() || mesh.faces.empty()) {
    throw std::runtime_error("target gradation requires a non-empty mesh");
  }
  if (lengths.size() != mesh.vertexCount()) {
    throw std::runtime_error("target gradation length size mismatch");
  }
  if (!std::isfinite(mean_edge_length) || mean_edge_length <= 0.0) {
    throw std::runtime_error(
        "target gradation requires a positive mean edge length");
  }
  for (double length : lengths) {
    if (!std::isfinite(length) || length <= 0.0) {
      throw std::runtime_error(
          "target gradation lengths must be positive and finite");
    }
  }
}

Adjacency buildLengthAdjacency(const Mesh& mesh) {
  Adjacency adjacency(mesh.vertexCount());
  const auto edges = uniqueEdges(mesh);
  if (edges.empty()) {
    throw std::runtime_error(
        "target gradation requires at least one mesh edge");
  }
  for (const Edge& edge : edges) {
    const int a = edge.first;
    const int b = edge.second;
    const double length = (mesh.vertices[static_cast<std::size_t>(a)] -
                           mesh.vertices[static_cast<std::size_t>(b)])
                              .norm();
    if (!std::isfinite(length) || length <= 0.0) {
      throw std::runtime_error(
          "target gradation requires positive finite edge lengths");
    }
    adjacency[static_cast<std::size_t>(a)].push_back({b, length});
    adjacency[static_cast<std::size_t>(b)].push_back({a, length});
  }
  for (auto& neighbors : adjacency) {
    std::sort(neighbors.begin(), neighbors.end(),
              [](const Neighbor& lhs, const Neighbor& rhs) {
                return lhs.first < rhs.first;
              });
  }
  return adjacency;
}

double percentile(std::vector<double> values, double fraction) {
  if (values.empty()) {
    return 0.0;
  }
  std::sort(values.begin(), values.end());
  const double position = fraction * static_cast<double>(values.size() - 1);
  const std::size_t lower = static_cast<std::size_t>(std::floor(position));
  const std::size_t upper = static_cast<std::size_t>(std::ceil(position));
  const double t = position - static_cast<double>(lower);
  return values[lower] * (1.0 - t) + values[upper] * t;
}

TargetGradationMetrics
measureTargetGradationFromAdjacency(const Adjacency& adjacency,
                                    const std::vector<double>& lengths,
                                    double mean_edge_length) {
  std::vector<double> slopes;
  slopes.reserve(lengths.size() * 3);
  TargetGradationMetrics metrics;
  for (std::size_t i = 0; i < adjacency.size(); ++i) {
    for (const Neighbor& neighbor : adjacency[i]) {
      const std::size_t j = static_cast<std::size_t>(neighbor.first);
      if (j <= i) {
        continue;
      }
      const double log_difference =
          std::abs(std::log(lengths[i]) - std::log(lengths[j]));
      const double normalized_slope =
          log_difference * mean_edge_length / neighbor.second;
      const double ratio =
          std::max(lengths[i], lengths[j]) / std::min(lengths[i], lengths[j]);
      if (!std::isfinite(normalized_slope) || !std::isfinite(ratio)) {
        throw std::runtime_error(
            "target gradation diagnostics produced a non-finite value");
      }
      slopes.push_back(normalized_slope);
      metrics.max_normalized_log_slope =
          std::max(metrics.max_normalized_log_slope, normalized_slope);
      metrics.max_adjacent_ratio = std::max(metrics.max_adjacent_ratio, ratio);
    }
  }
  metrics.p95_normalized_log_slope = percentile(slopes, 0.95);
  metrics.p99_normalized_log_slope = percentile(slopes, 0.99);
  return metrics;
}

}  // namespace

TargetGradationMetrics
measureTargetGradation(const Mesh& mesh, const std::vector<double>& lengths,
                       double mean_edge_length) {
  validateLengths(mesh, lengths, mean_edge_length);
  return measureTargetGradationFromAdjacency(buildLengthAdjacency(mesh),
                                             lengths, mean_edge_length);
}

TargetGradationResult applyDownwardLogGradationLimiter(
    const Mesh& mesh, const std::vector<double>& initial_lengths,
    double mean_edge_length, double gradation_ratio) {
  validateLengths(mesh, initial_lengths, mean_edge_length);
  if (!std::isfinite(gradation_ratio) || gradation_ratio <= 1.0) {
    throw std::runtime_error(
        "target gradation ratio must be finite and greater than one");
  }

  const Adjacency adjacency = buildLengthAdjacency(mesh);
  const double log_ratio = std::log(gradation_ratio);
  const auto started = Clock::now();

  std::vector<double> initial_log(initial_lengths.size(), 0.0);
  std::vector<double> limited_log(initial_lengths.size(), 0.0);
  std::vector<double> source_distances(initial_lengths.size(), 0.0);
  std::vector<int> source_vertices(initial_lengths.size(), -1);
  using QueueEntry = std::pair<double, int>;
  std::priority_queue<QueueEntry, std::vector<QueueEntry>,
                      std::greater<QueueEntry>>
      queue;
  for (std::size_t i = 0; i < initial_lengths.size(); ++i) {
    initial_log[i] = std::log(initial_lengths[i]);
    limited_log[i] = initial_log[i];
    source_vertices[i] = static_cast<int>(i);
    queue.push({limited_log[i], static_cast<int>(i)});
  }

  TargetGradationResult result;
  while (!queue.empty()) {
    const QueueEntry current = queue.top();
    queue.pop();
    ++result.stats.queue_pops;
    const double value = current.first;
    const std::size_t i = static_cast<std::size_t>(current.second);
    if (value != limited_log[i]) {
      continue;
    }
    for (const Neighbor& neighbor : adjacency[i]) {
      const std::size_t j = static_cast<std::size_t>(neighbor.first);
      const double edge_cost = log_ratio * neighbor.second / mean_edge_length;
      const double candidate = value + edge_cost;
      if (candidate < limited_log[j]) {
        limited_log[j] = candidate;
        source_vertices[j] = source_vertices[i];
        source_distances[j] =
            source_distances[i] + neighbor.second / mean_edge_length;
        queue.push({candidate, neighbor.first});
        ++result.stats.relaxations;
      }
    }
  }

  result.limited_lengths.resize(initial_lengths.size(), 0.0);
  result.source_vertices = std::move(source_vertices);
  result.normalized_source_distances = std::move(source_distances);
  std::vector<double> reductions(initial_lengths.size(), 0.0);
  const double change_tolerance = 64.0 * std::numeric_limits<double>::epsilon();
  for (std::size_t i = 0; i < limited_log.size(); ++i) {
    const double limited_length = std::exp(limited_log[i]);
    if (!std::isfinite(limited_length) || limited_length <= 0.0) {
      throw std::runtime_error("target gradation produced an invalid length");
    }
    if (limited_length > initial_lengths[i] * (1.0 + change_tolerance)) {
      throw std::runtime_error("target gradation increased an input length");
    }
    result.limited_lengths[i] = std::min(limited_length, initial_lengths[i]);
    reductions[i] = initial_log[i] - std::log(result.limited_lengths[i]);
    const double scaled_tolerance =
        change_tolerance * std::max(1.0, std::abs(initial_log[i]));
    if (reductions[i] > scaled_tolerance) {
      ++result.stats.changed_vertices;
    }
  }

  result.stats.changed_vertex_rate =
      static_cast<double>(result.stats.changed_vertices) /
      static_cast<double>(initial_lengths.size());
  result.stats.min_log_reduction =
      *std::min_element(reductions.begin(), reductions.end());
  result.stats.max_log_reduction =
      *std::max_element(reductions.begin(), reductions.end());
  result.stats.mean_log_reduction =
      std::accumulate(reductions.begin(), reductions.end(), 0.0) /
      static_cast<double>(reductions.size());
  result.stats.before = measureTargetGradationFromAdjacency(
      adjacency, initial_lengths, mean_edge_length);
  result.stats.after = measureTargetGradationFromAdjacency(
      adjacency, result.limited_lengths, mean_edge_length);
  result.stats.runtime_ms =
      std::chrono::duration<double, std::milli>(Clock::now() - started).count();
  return result;
}

}  // namespace adaiso::detail
