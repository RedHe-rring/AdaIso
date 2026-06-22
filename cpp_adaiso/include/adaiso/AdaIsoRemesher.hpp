#pragma once

#include "adaiso/Geometry.hpp"
#include "adaiso/Mesh.hpp"

#include <array>
#include <string>
#include <unordered_set>
#include <vector>

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
  double approximationTolerance = 0.0;
  double minTargetLength = 0.0;
  double maxTargetLength = 0.0;
  double subremesh_multiplier = 1.5;
  int max_subremesh_rounds = 8;
  int max_split_passes = 100;
  int max_collapse_passes = 100;
  int max_flip_passes = 200;
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

struct SizingStats {
  double epsilon = 0.0;
  double min_target_length = 0.0;
  double max_target_length = 0.0;
  double min = 0.0;
  double max = 0.0;
  double avg = 0.0;
};

struct IterationStats {
  int iteration = 0;
  int split_count = 0;
  int collapse_count = 0;
  int flip_count = 0;
  std::size_t vertices = 0;
  std::size_t faces = 0;
  double target_error_avg = 0.0;
  double target_error_max = 0.0;
  double min_angle_degrees = 0.0;
  double qavg = 0.0;
  double qmin = 0.0;
  double geometry_error_avg = 0.0;
  double geometry_error_max = 0.0;
};

struct AdaIsoStats {
  std::size_t vertices = 0;
  std::size_t faces = 0;
  SizingStats sizing;
  std::vector<IterationStats> iterations;
};

class AdaIsoRemesher {
 public:
  explicit AdaIsoRemesher(AdaIsoConfig config = {});

  [[nodiscard]] Mesh remesh(const Mesh& input);
  [[nodiscard]] const AdaIsoStats& stats() const { return stats_; }

 private:
  [[nodiscard]] Mesh subRemesh(Mesh mesh) const;
  void buildSizingField(const Mesh& mesh);
  void cleanupMesh(Mesh& mesh);
  [[nodiscard]] double targetAt(int vertex) const;
  [[nodiscard]] double edgeTarget(const Mesh& mesh, int a, int b, bool use_max) const;
  [[nodiscard]] int splitLongEdges(Mesh& mesh);
  void splitOneEdge(Mesh& mesh, int a, int b) const;
  [[nodiscard]] int collapseShortEdges(Mesh& mesh);
  [[nodiscard]] int collapseEdgesBatch(Mesh& mesh, const std::vector<Edge>& edges);
  [[nodiscard]] bool collapseValid(
      const Mesh& mesh,
      int a,
      int b,
      const std::vector<std::unordered_set<int>>& neigh,
      const EdgeFaceMap& e2f,
      const std::vector<std::vector<int>>& vf) const;
  void collapseOneEdge(Mesh& mesh, int a, int b);
  [[nodiscard]] int flipEdges(Mesh& mesh);
  [[nodiscard]] bool flipImprovesValence(
      const std::vector<int>& valence,
      const std::unordered_set<int>& boundary,
      int a,
      int b,
      int c,
      int d) const;
  [[nodiscard]] IterationStats collectIterationStats(
      const Mesh& mesh,
      const Mesh& original,
      int iteration,
      int split_count,
      int collapse_count,
      int flip_count) const;
  void tangentSmoothing(Mesh& mesh);
  [[nodiscard]] std::vector<Eigen::Vector3d> mlsProject(
      const std::vector<Eigen::Vector3d>& points,
      const Mesh& original) const;

  AdaIsoConfig config_;
  AdaIsoStats stats_;
  std::vector<double> target_lengths_;
};

[[nodiscard]] Variant parseVariant(const std::string& value);
[[nodiscard]] std::string toString(Variant variant);

}  // namespace adaiso
