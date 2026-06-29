#pragma once

#include "adaiso/Geometry.hpp"
#include "adaiso/Mesh.hpp"

#include <cstddef>
#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

namespace adaiso {

enum class Variant {
  Paper,
  Code,
};

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

  void build(const Mesh& input, const std::vector<double>& log_lengths);
  void clear();
  [[nodiscard]] bool empty() const;
  [[nodiscard]] double valueAt(const Eigen::Vector3d& point) const;

 private:
  std::unique_ptr<TargetLengthFieldImpl> impl_;
};

struct AdaIsoConfig {
  double mesh_scale = 1.0;
  Variant variant = Variant::Code;
  double csf_smoothing_alpha = 2.0;
  double target_min_factor = 0.6;
  double target_max_factor = 1.8;
  double target_gamma = 1.0;
  double target_smooth_alpha = 0.3;
  double target_max_adjacent_ratio = 1.30;
  int target_gradation_iterations = 20;
  std::string target_field_export_dir;
  bool preprocess = false;
  int global_iterations = 5;
  double subremesh_multiplier = 1.5;
  int max_subremesh_rounds = 8;
  int max_split_passes = 10;
  int max_collapse_passes = 10;
  int max_flip_passes = 20;
  double tangent_lambda = 0.6;
  bool project_to_surface = true;
  int mls_k = 8;
  int mls_iterations = 10;
  double mls_tolerance = 1e-4;
  bool avoid_obtuse_flip = true;
  double min_quality_after_collapse = 0.05;
  bool verbose = true;

  [[nodiscard]] double splitFactor() const;
  [[nodiscard]] double collapseFactor() const;
  [[nodiscard]] bool collapseUsesMaxReference() const;
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
};

class AdaIsoRemesher {
 public:
  explicit AdaIsoRemesher(AdaIsoConfig config = {});

  [[nodiscard]] Mesh remesh(const Mesh& input);
  [[nodiscard]] const AdaIsoStats& stats() const { return stats_; }

 private:
  [[nodiscard]] Mesh subRemesh(Mesh mesh) const;
  void buildTargetLengths(const Mesh& mesh);
  [[nodiscard]] std::vector<double> computeCsf(const Mesh& mesh);
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
      const std::vector<std::vector<int>>& vf) const;
  [[nodiscard]] int collapseEdgesBatch(Mesh& mesh, const std::vector<Edge>& selected) const;
  [[nodiscard]] int flipEdges(Mesh& mesh);
  [[nodiscard]] bool flipImprovesValence(
      const std::vector<int>& valence,
      const std::unordered_set<int>& boundary,
      int a,
      int b,
      int c,
      int d) const;
  void tangentSmoothing(Mesh& mesh);
  [[nodiscard]] std::vector<Eigen::Vector3d> mlsProject(
      const std::vector<Eigen::Vector3d>& points,
      const Mesh& original) const;

  AdaIsoConfig config_;
  AdaIsoStats stats_;
  TargetLengthField target_field_;
};

[[nodiscard]] Variant parseVariant(const std::string& value);
[[nodiscard]] std::string toString(Variant variant);
[[nodiscard]] EdgeRatioStats computeEdgeRatioStats(
    const Mesh& mesh,
    const TargetLengthField& target_field,
    double collapse_factor,
    double split_factor);

}  // namespace adaiso
