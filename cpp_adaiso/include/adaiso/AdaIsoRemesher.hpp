#pragma once

#include "adaiso/Geometry.hpp"
#include "adaiso/KDTree.hpp"
#include "adaiso/Mesh.hpp"

#include <array>
#include <string>
#include <unordered_set>

namespace adaiso {

enum class Variant {
  Paper,
  Code,
};

struct AdaIsoConfig {
  double mesh_scale = 1.0;
  Variant variant = Variant::Code;
  bool preprocess = false;
  int global_iterations = 5;
  int histogram_bins = 40;
  std::array<double, 5> multipliers = {1.8, 1.4, 1.0, 0.8, 0.6};
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
  std::size_t vertices = 0;
  std::size_t faces = 0;
  double csf_assembly_time_ms = 0.0;
  double csf_factorization_time_ms = 0.0;
  double csf_solve_time_ms = 0.0;
  double csf_residual_abs = 0.0;
  double csf_residual_scaled = 0.0;
  std::size_t csf_free_vertices = 0;
  std::size_t csf_fixed_vertices = 0;
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
  [[nodiscard]] std::vector<double> histogramMultipliers(const std::vector<double>& curvature) const;
  [[nodiscard]] double targetAt(const Eigen::Vector3d& point) const;
  [[nodiscard]] double edgeTarget(const Mesh& mesh, int a, int b, bool use_max) const;
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
  std::vector<Eigen::Vector3d> target_positions_;
  std::vector<double> target_lengths_;
  NearestNeighborIndex target_index_;
};

[[nodiscard]] Variant parseVariant(const std::string& value);
[[nodiscard]] std::string toString(Variant variant);

}  // namespace adaiso
