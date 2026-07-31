#pragma once

#include "adaiso/Geometry.hpp"
#include "adaiso/Mesh.hpp"
#include "adaiso/detail/TargetErrorSizing.hpp"

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

enum class CurvatureMode {
  NormalAngle,
  RobustMean,
  PrincipalMax,
};

enum class StageAPolicy {
  SplitCollapseNoGuard,
  SplitCollapseWeakGuard,
  CollapseSplitWeakGuard,
};

struct EdgeRatioStats {
  double mean_ratio = 0.0;
  double std_ratio = 0.0;
  double p05_ratio = 0.0;
  double p95_ratio = 0.0;
  double violation_rate = 0.0;
  double short_violation_rate = 0.0;
  double long_violation_rate = 0.0;
  double target_log_energy = 0.0;
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
  Variant variant = Variant::Code;
  CurvatureMode curvature_mode = CurvatureMode::PrincipalMax;
  double csf_smoothing_alpha = 2.0;
  double target_error_delta = 0.0;
  std::string target_field_export_dir;
  std::string iteration_mesh_export_dir;
  std::string stage_mesh_export_dir;
  std::string all_stage_mesh_export_dir;
  bool preprocess = false;
  int global_iterations = 5;
  double subremesh_multiplier = 1.5;
  int max_subremesh_rounds = 8;
  int max_split_passes = 5;
  int max_collapse_passes = 5;
  int max_flip_passes = 5;
  double tangent_lambda = 0.6;
  bool enable_relocation_mls = false;
  int mls_k = 8;
  int mls_iterations = 5;
  double mls_tolerance = 1e-4;
  bool avoid_obtuse_flip = true;
  double max_flip_dihedral_degrees = 45.0;
  bool feature_aware_collapse = true;
  double collapse_feature_omega = 0.15;
  double split_factor_override = 0.0;
  double collapse_factor_override = 0.0;
  double target_h_max_factor = 5.0;
  double auto_delta_quantile = 0.5;
  double min_angle_floor_degrees = 10.0;
  double split_min_angle_floor_degrees = 1.0;
  StageAPolicy stage_a_policy = StageAPolicy::SplitCollapseNoGuard;
  bool verbose = true;

  [[nodiscard]] double splitFactor() const;
  [[nodiscard]] double collapseFactor() const;
  [[nodiscard]] bool collapseUsesMaxReference() const;
};

struct AdaIsoStats {
  struct IterationStats {
    std::string stage;
    int iteration = 0;
    int split_count = 0;
    int collapse_count = 0;
    int flip_count = 0;
    int relocation_count = 0;
    std::size_t vertices = 0;
    std::size_t faces = 0;
    double qavg = 0.0;
    double qmin = 0.0;
    double min_angle_degrees = 0.0;
    double angle_lt_10_pct = 0.0;
    double angle_lt_20_pct = 0.0;
    double mean_ratio = 0.0;
    double std_ratio = 0.0;
    double p05_ratio = 0.0;
    double p95_ratio = 0.0;
    double violation_rate = 0.0;
    double short_violation_rate = 0.0;
    double long_violation_rate = 0.0;
    double target_log_energy = 0.0;
  };

  std::size_t vertices = 0;
  std::size_t faces = 0;
  std::vector<IterationStats> iterations;
  int size_stage_iterations = 0;
  int size_stage_splits = 0;
  int size_stage_collapses = 0;
  std::size_t size_stage_long_edge_candidates = 0;
  std::size_t size_stage_split_accepted = 0;
  std::size_t size_stage_split_rejected_by_angle = 0;
  std::size_t size_stage_short_edge_candidates = 0;
  std::size_t size_stage_collapse_accepted = 0;
  std::size_t size_stage_collapse_rejected = 0;
  std::size_t size_stage_collapse_rejected_by_boundary = 0;
  std::size_t size_stage_collapse_rejected_by_non_manifold = 0;
  std::size_t size_stage_collapse_rejected_by_batch_conflict = 0;
  std::size_t size_stage_collapse_rejected_by_link_condition = 0;
  std::size_t size_stage_collapse_rejected_by_degenerate_patch = 0;
  std::size_t size_stage_collapse_rejected_by_target_length = 0;
  std::size_t size_stage_collapse_rejected_by_angle = 0;
  std::size_t size_stage_collapse_rejected_by_batch_duplicate = 0;
  std::size_t size_stage_collapse_conflict_mini_rounds = 0;
  std::size_t size_stage_collapse_conflict_retry_candidates = 0;
  std::size_t size_stage_collapse_conflict_retry_accepted = 0;
  std::size_t size_stage_collapse_conflict_retry_rejected = 0;
  double size_stage_final_violation = 0.0;
  double size_stage_final_short_violation = 0.0;
  double size_stage_final_long_violation = 0.0;
  double size_stage_final_target_log_energy = 0.0;
  double size_stage_final_min_angle_degrees = 0.0;
  double size_stage_final_angle_lt_10_pct = 0.0;
  double size_stage_final_angle_lt_20_pct = 0.0;
  int quality_stage_iterations = 0;
  int quality_stage_flips = 0;
  int quality_stage_relocations = 0;
  double csf_assembly_time_ms = 0.0;
  double csf_factorization_time_ms = 0.0;
  double csf_solve_time_ms = 0.0;
  double csf_residual_abs = 0.0;
  double csf_residual_scaled = 0.0;
  std::size_t csf_matrix_nonzeros = 0;
  bool csf_solver_success = false;
  double target_error_delta = 0.0;
  double target_error_bbox_ratio = 0.0;
  double bbox_diagonal = 0.0;
  double target_h_min = 0.0;
  double target_h_max = 0.0;
  double target_estimated_faces = 0.0;
  double target_kappa_max = 0.0;
  double target_auto_delta_quantile = 0.5;
  double target_h_min_safe_factor = 0.25;
  bool target_flat_fallback = false;
};

class AdaIsoRemesher {
 public:
  explicit AdaIsoRemesher(AdaIsoConfig config = {});

  [[nodiscard]] Mesh remesh(const Mesh& input);
  [[nodiscard]] const AdaIsoStats& stats() const { return stats_; }

 private:
  enum class VertexRelocationClass {
    Corner,
    Crease,
    Smooth
  };

  enum class CollapseRejectReason {
    None,
    NonManifold,
    LinkCondition,
    DegeneratePatch,
    TargetLength,
    Angle
  };

  [[nodiscard]] Mesh subRemesh(Mesh mesh) const;
  void buildTargetLengths(const Mesh& mesh);
  [[nodiscard]] std::vector<double> computeCsf(const Mesh& mesh);
  [[nodiscard]] std::vector<double> solveCsf(
      const Mesh& mesh,
      const std::vector<double>& raw_curvature);
  [[nodiscard]] double targetAt(const Eigen::Vector3d& point) const;
  [[nodiscard]] EdgeRatioStats edgeRatioStats(const Mesh& mesh) const;
  void runSizeFieldFittingStage(Mesh& mesh);
  void runQualityRepairStage(Mesh& mesh, const Mesh& original_for_projection);
  void recordIterationStats(
      const std::string& stage,
      int iteration,
      int split_count,
      int collapse_count,
      int flip_count,
      int relocation_count,
      const Mesh& mesh);
  void writeStageSnapshot(
      const Mesh& mesh,
      const std::string& stage,
      int iteration,
      const std::string& operation) const;
  void writeIterationSnapshot(
      const Mesh& mesh,
      const std::string& stage,
      int iteration) const;
  [[nodiscard]] int splitLongEdges(Mesh& mesh);
  [[nodiscard]] bool splitAcceptedByAngleGuard(
      const Mesh& mesh,
      const EdgeFaceMap& e2f,
      int a,
      int b) const;
  void splitOneEdge(Mesh& mesh, int a, int b) const;
  [[nodiscard]] int collapseShortEdges(Mesh& mesh);
  [[nodiscard]] CollapseRejectReason collapseRejectReason(
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
  [[nodiscard]] bool flipAcceptedByAngleMetric(
      const Mesh& mesh,
      const std::vector<int>& valence,
      const std::unordered_set<int>& boundary,
      const Eigen::Vector3i& old1,
      const Eigen::Vector3i& old2,
      const Eigen::Vector3i& new1,
      const Eigen::Vector3i& new2,
      int a,
      int b,
      int c,
      int d) const;
  [[nodiscard]] std::vector<VertexRelocationClass> classifyVerticesForRelocation(const Mesh& mesh) const;
  [[nodiscard]] int relocateVertices(Mesh& mesh, const Mesh& original_for_projection);
  [[nodiscard]] int tangentSmoothing(Mesh& mesh, const std::vector<VertexRelocationClass>& classes);
  [[nodiscard]] bool relocationAcceptedByAngleMetric(
      const Mesh& mesh,
      const std::vector<std::vector<int>>& vf,
      int vertex,
      const Eigen::Vector3d& candidate) const;
  [[nodiscard]] std::vector<Eigen::Vector3d> mlsProject(
      const std::vector<Eigen::Vector3d>& points,
      const Mesh& original) const;

  AdaIsoConfig config_;
  AdaIsoStats stats_;
  TargetLengthField target_field_;
  double target_min_length_ = 0.0;
  double target_max_length_ = 0.0;
};

[[nodiscard]] Variant parseVariant(const std::string& value);
[[nodiscard]] std::string toString(Variant variant);
[[nodiscard]] CurvatureMode parseCurvatureMode(const std::string& value);
[[nodiscard]] std::string toString(CurvatureMode mode);
[[nodiscard]] StageAPolicy parseStageAPolicy(const std::string& value);
[[nodiscard]] std::string toString(StageAPolicy policy);
[[nodiscard]] EdgeRatioStats computeEdgeRatioStats(
    const Mesh& mesh,
    const TargetLengthField& target_field,
    double collapse_factor,
    double split_factor);

}  // namespace adaiso
