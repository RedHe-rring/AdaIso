#pragma once

#include "adaiso/Geometry.hpp"
#include "adaiso/Mesh.hpp"
#include "adaiso/detail/FeatureClassification.hpp"

#include <cstddef>
#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

namespace adaiso {

struct EdgeRatioStats {
  double mean_ratio = 0.0;
  double std_ratio = 0.0;
  double p05_ratio = 0.0;
  double p95_ratio = 0.0;
  double violation_rate = 0.0;
};

struct TargetLengthFieldImpl;

class TargetLengthField {
 public:
  TargetLengthField();
  ~TargetLengthField();

  TargetLengthField(const TargetLengthField&) = delete;
  TargetLengthField& operator=(const TargetLengthField&) = delete;
  TargetLengthField(TargetLengthField&& other) noexcept;
  TargetLengthField& operator=(TargetLengthField&& other) noexcept;

  void build(const Mesh& input, const std::vector<double>& target_lengths);
  void clear();
  [[nodiscard]] bool empty() const;
  [[nodiscard]] double valueAt(const Eigen::Vector3d& point) const;

 private:
  std::unique_ptr<TargetLengthFieldImpl> impl_;
};

struct AdaIsoConfig {
  enum class CurvatureSignal {
    LogDimensionless,
    Dimensionless,
    Physical,
  };

  std::size_t target_faces = 0;
  bool preprocess = false;
  int global_iterations = 5;
  CurvatureSignal curvature_signal = CurvatureSignal::LogDimensionless;
  bool use_csf = true;
  double curvature_cap_quantile = 0.995;
  std::string export_target_field_dir;
  double subremesh_multiplier = 1.5;
  int max_subremesh_rounds = 8;
  int max_split_passes = 5;
  int max_collapse_passes = 5;
  int max_flip_passes = 5;
  double tangent_lambda = 0.6;
  double min_quality_after_collapse = 0.05;
  bool verbose = true;

  [[nodiscard]] double splitFactor() const;
  [[nodiscard]] double collapseFactor() const;
};

struct AdaIsoStats {
  struct IterationStats {
    int iteration = 0;
    int split_count = 0;
    int collapse_count = 0;
    int flip_count = 0;
    std::size_t vertices = 0;
    std::size_t faces = 0;
    double qavg = 0.0;
    double qmin = 0.0;
    double mean_ratio = 0.0;
    double std_ratio = 0.0;
    double p05_ratio = 0.0;
    double p95_ratio = 0.0;
    double violation_rate = 0.0;
  };

  std::size_t vertices = 0;
  std::size_t faces = 0;
  std::vector<IterationStats> iterations;
  double csf_assembly_time_ms = 0.0;
  double csf_factorization_time_ms = 0.0;
  double csf_solve_time_ms = 0.0;
  double csf_residual_abs = 0.0;
  double csf_residual_scaled = 0.0;
  std::size_t csf_matrix_nonzeros = 0;
  bool csf_solver_success = false;
  int split_total = 0;
  int collapse_total = 0;
  int flip_total = 0;
  double target_lambda = 0.0;
  double target_estimated_faces = 0.0;
  double target_h_min = 0.0;
  double target_h_max = 0.0;
  double target_observed_h_min = 0.0;
  double target_observed_h_max = 0.0;
  double curvature_cap_value = 0.0;
  double curvature_cap_changed_rate = 0.0;
};

class AdaIsoRemesher {
 public:
  explicit AdaIsoRemesher(AdaIsoConfig config = {});

  [[nodiscard]] Mesh remesh(const Mesh& input);
  [[nodiscard]] const AdaIsoStats& stats() const { return stats_; }

 private:
  [[nodiscard]] Mesh subRemesh(Mesh mesh) const;
  void buildTargetLengths(const Mesh& mesh, std::size_t target_faces);
  [[nodiscard]] std::vector<double> solveCsf(
      const Mesh& mesh,
      const std::vector<double>& raw_curvature);
  [[nodiscard]] double targetAt(const Eigen::Vector3d& point) const;
  [[nodiscard]] double edgeTarget(const Mesh& mesh, int a, int b, bool use_max) const;
  [[nodiscard]] EdgeRatioStats edgeRatioStats(const Mesh& mesh) const;
  [[nodiscard]] int splitLongEdges(Mesh& mesh);
  void splitOneEdge(Mesh& mesh, int a, int b) const;
  [[nodiscard]] int collapseShortEdges(Mesh& mesh);
  [[nodiscard]] bool collapseValid(
      const Mesh& mesh,
      int a,
      int b,
      const std::vector<std::unordered_set<int>>& neigh,
      const EdgeFaceMap& e2f,
      const std::vector<std::vector<int>>& vf,
      const std::vector<double>& feature_strengths) const;
  [[nodiscard]] Eigen::Vector3d collapsePoint(
      const Mesh& mesh,
      int a,
      int b,
      const std::vector<double>& feature_strengths) const;
  [[nodiscard]] int collapseEdgesBatch(
      Mesh& mesh,
      const std::vector<Edge>& selected,
      const std::vector<double>& feature_strengths) const;
  [[nodiscard]] int flipEdges(Mesh& mesh);
  void relocateVertices(Mesh& mesh);
  void tangentSmoothing(Mesh& mesh, const std::vector<detail::VertexRelocationClass>& classes);

  AdaIsoConfig config_;
  AdaIsoStats stats_;
  TargetLengthField target_field_;
};

[[nodiscard]] EdgeRatioStats computeEdgeRatioStats(
    const Mesh& mesh,
    const TargetLengthField& target_field,
    double collapse_factor,
    double split_factor);

}  // namespace adaiso
