#pragma once

#include "adaiso/Mesh.hpp"

#include <unordered_map>
#include <unordered_set>

namespace adaiso {

using EdgeFaceMap = std::unordered_map<Edge, std::vector<int>, EdgeHash>;

[[nodiscard]] Eigen::Vector3d safeNormalize(const Eigen::Vector3d& v, double eps = 1e-14);
[[nodiscard]] std::vector<double> faceAreas(const Mesh& mesh);
[[nodiscard]] std::vector<Eigen::Vector3d> faceNormals(const Mesh& mesh);
[[nodiscard]] std::vector<Eigen::Vector3d> vertexNormals(const Mesh& mesh);
[[nodiscard]] EdgeFaceMap edgeFaces(const Mesh& mesh);
[[nodiscard]] std::vector<std::unordered_set<int>> vertexNeighbors(const Mesh& mesh);
[[nodiscard]] std::vector<std::vector<int>> vertexFaces(const Mesh& mesh);
[[nodiscard]] std::unordered_set<int> boundaryVertices(const Mesh& mesh);
[[nodiscard]] double meanEdgeLength(const Mesh& mesh);
[[nodiscard]] std::unordered_map<Edge, double, EdgeHash> cotangentEdgeWeights(
    const Mesh& mesh,
    double min_weight);
[[nodiscard]] std::vector<double> maximumAbsolutePrincipalCurvatures(const Mesh& mesh);
[[nodiscard]] std::vector<double> curvatureSizingField(
    const Mesh& mesh,
    double epsilon,
    double min_length,
    double max_length);
[[nodiscard]] double triangleArea(
    const Eigen::Vector3d& a,
    const Eigen::Vector3d& b,
    const Eigen::Vector3d& c);
[[nodiscard]] double triangleQuality(
    const Eigen::Vector3d& a,
    const Eigen::Vector3d& b,
    const Eigen::Vector3d& c);
[[nodiscard]] std::vector<double> triangleQualities(const Mesh& mesh);
[[nodiscard]] double maxInnerAngle(
    const Eigen::Vector3d& a,
    const Eigen::Vector3d& b,
    const Eigen::Vector3d& c);

}  // namespace adaiso
