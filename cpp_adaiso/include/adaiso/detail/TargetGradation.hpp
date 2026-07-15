#pragma once

#include "adaiso/Mesh.hpp"

#include <cstddef>
#include <vector>

namespace adaiso::detail {

struct TargetGradationMetrics {
  double max_normalized_log_slope = 0.0;
  double p95_normalized_log_slope = 0.0;
  double p99_normalized_log_slope = 0.0;
  double max_adjacent_ratio = 1.0;
};

struct TargetGradationStats {
  double runtime_ms = 0.0;
  std::size_t queue_pops = 0;
  std::size_t relaxations = 0;
  std::size_t changed_vertices = 0;
  double changed_vertex_rate = 0.0;
  double min_log_reduction = 0.0;
  double mean_log_reduction = 0.0;
  double max_log_reduction = 0.0;
  TargetGradationMetrics before;
  TargetGradationMetrics after;
};

struct TargetGradationResult {
  std::vector<double> limited_lengths;
  std::vector<int> source_vertices;
  std::vector<double> normalized_source_distances;
  TargetGradationStats stats;
};

[[nodiscard]] TargetGradationMetrics
measureTargetGradation(const Mesh& mesh, const std::vector<double>& lengths,
                       double mean_edge_length);

[[nodiscard]] TargetGradationResult applyDownwardLogGradationLimiter(
    const Mesh& mesh, const std::vector<double>& initial_lengths,
    double mean_edge_length, double gradation_ratio);

}  // namespace adaiso::detail
