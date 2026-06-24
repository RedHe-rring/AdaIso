#pragma once

#include "adaiso/Geometry.hpp"
#include "adaiso/KDTree.hpp"
#include "adaiso/Mesh.hpp"

#include <cstddef>
#include <string>
#include <unordered_set>
#include <vector>

namespace adaiso {

enum class Variant {
  Paper,
  Code,
};

enum class TargetLookupMode {
  Nearest,
  GaussianKnn,
};

struct EdgeRatioStats {
  double mean_ratio = 0.0;
  double std_ratio = 0.0;
  double p05_ratio = 0.0;
  double p95_ratio = 0.0;
  double violation_rate = 0.0;
};

class TargetLengthField {
 public:
  TargetLengthField() = default;
  ~TargetLengthField() = default;

  TargetLengthField(const TargetLengthField&) = delete;
  TargetLengthField& operator=(const TargetLengthField&) = delete;
  TargetLengthField(TargetLengthField&& other);
  TargetLengthField& operator=(TargetLengthField&& other);

  void build(const std::vector<Eigen::Vector3d>& positions, const std::vector<double>& lengths);
  void clear();
  [[nodiscard]] bool empty() const;
  [[nodiscard]] double valueAt(
      const Eigen::Vector3d& point,
      TargetLookupMode mode,
      int k,
      double sigma_factor) const;

 private:
  std::vector<Eigen::Vector3d> positions_;
  std::vector<double> lengths_;
  NearestNeighborIndex index_;
};

struct AdaIsoConfig {
  double mesh_scale = 1.0;
  Variant variant = Variant::Code;
  TargetLookupMode target_lookup_mode = TargetLookupMode::GaussianKnn;
  int target_knn_k = 8;
  double target_gaussian_sigma_factor = 1.0;
  double csf_smoothing_alpha = 0.5;
  double target_min_factor = 0.6;
  double target_max_factor = 1.8;
  double target_gamma = 1.0;
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
[[nodiscard]] TargetLookupMode parseTargetLookupMode(const std::string& value);
[[nodiscard]] std::string toString(TargetLookupMode mode);
[[nodiscard]] EdgeRatioStats computeEdgeRatioStats(
    const Mesh& mesh,
    const TargetLengthField& target_field,
    TargetLookupMode mode,
    int target_knn_k,
    double target_gaussian_sigma_factor,
    double collapse_factor,
    double split_factor);

}  // namespace adaiso
