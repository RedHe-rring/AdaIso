#pragma once

#include <Eigen/Core>

#include <cstddef>
#include <utility>
#include <vector>

namespace adaiso {

struct Mesh {
  std::vector<Eigen::Vector3d> vertices;
  std::vector<Eigen::Vector3i> faces;

  [[nodiscard]] std::size_t vertexCount() const { return vertices.size(); }
  [[nodiscard]] std::size_t faceCount() const { return faces.size(); }
  void validate() const;
};

using Edge = std::pair<int, int>;

struct EdgeHash {
  std::size_t operator()(const Edge& e) const noexcept;
};

[[nodiscard]] Edge makeEdge(int a, int b);
[[nodiscard]] Mesh compactVertices(const Mesh& mesh);
[[nodiscard]] Mesh removeDegenerateFaces(const Mesh& mesh, double area_eps = 1e-14);
[[nodiscard]] std::vector<Edge> uniqueEdges(const Mesh& mesh);

}  // namespace adaiso
