#include "adaiso/Geometry.hpp"

#include <Eigen/Geometry>

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>

namespace adaiso {

Eigen::Vector3d safeNormalize(const Eigen::Vector3d& v, double eps) {
  const double n = v.norm();
  if (n <= eps || !std::isfinite(n)) {
    return Eigen::Vector3d::Zero();
  }
  return v / n;
}

std::vector<double> faceAreas(const Mesh& mesh) {
  std::vector<double> areas;
  areas.reserve(mesh.faces.size());
  for (const auto& face : mesh.faces) {
    areas.push_back(triangleArea(mesh.vertices[face[0]], mesh.vertices[face[1]], mesh.vertices[face[2]]));
  }
  return areas;
}

std::vector<Eigen::Vector3d> faceNormals(const Mesh& mesh) {
  std::vector<Eigen::Vector3d> normals;
  normals.reserve(mesh.faces.size());
  for (const auto& face : mesh.faces) {
    const auto n = (mesh.vertices[face[1]] - mesh.vertices[face[0]])
                       .cross(mesh.vertices[face[2]] - mesh.vertices[face[0]]);
    normals.push_back(safeNormalize(n));
  }
  return normals;
}

std::vector<Eigen::Vector3d> vertexNormals(const Mesh& mesh) {
  std::vector<Eigen::Vector3d> normals(mesh.vertices.size(), Eigen::Vector3d::Zero());
  const auto fn = faceNormals(mesh);
  const auto fa = faceAreas(mesh);
  for (std::size_t i = 0; i < mesh.faces.size(); ++i) {
    const auto weighted = fn[i] * fa[i];
    const auto& f = mesh.faces[i];
    normals[f[0]] += weighted;
    normals[f[1]] += weighted;
    normals[f[2]] += weighted;
  }
  for (auto& n : normals) {
    n = safeNormalize(n);
  }
  return normals;
}

EdgeFaceMap edgeFaces(const Mesh& mesh) {
  EdgeFaceMap result;
  for (int i = 0; i < static_cast<int>(mesh.faces.size()); ++i) {
    const auto& f = mesh.faces[static_cast<std::size_t>(i)];
    result[makeEdge(f[0], f[1])].push_back(i);
    result[makeEdge(f[1], f[2])].push_back(i);
    result[makeEdge(f[2], f[0])].push_back(i);
  }
  return result;
}

std::vector<std::unordered_set<int>> vertexNeighbors(const Mesh& mesh) {
  std::vector<std::unordered_set<int>> result(mesh.vertices.size());
  for (const auto& f : mesh.faces) {
    result[f[0]].insert(f[1]);
    result[f[0]].insert(f[2]);
    result[f[1]].insert(f[0]);
    result[f[1]].insert(f[2]);
    result[f[2]].insert(f[0]);
    result[f[2]].insert(f[1]);
  }
  return result;
}

std::vector<std::vector<int>> vertexFaces(const Mesh& mesh) {
  std::vector<std::vector<int>> result(mesh.vertices.size());
  for (int i = 0; i < static_cast<int>(mesh.faces.size()); ++i) {
    const auto& f = mesh.faces[static_cast<std::size_t>(i)];
    result[f[0]].push_back(i);
    result[f[1]].push_back(i);
    result[f[2]].push_back(i);
  }
  return result;
}

std::unordered_set<int> boundaryVertices(const Mesh& mesh) {
  std::unordered_set<int> result;
  for (const auto& item : edgeFaces(mesh)) {
    if (item.second.size() == 1) {
      result.insert(item.first.first);
      result.insert(item.first.second);
    }
  }
  return result;
}

double meanEdgeLength(const Mesh& mesh) {
  const auto edges = uniqueEdges(mesh);
  if (edges.empty()) {
    return 0.0;
  }
  double sum = 0.0;
  for (const auto& [a, b] : edges) {
    sum += (mesh.vertices[a] - mesh.vertices[b]).norm();
  }
  return sum / static_cast<double>(edges.size());
}

std::unordered_map<Edge, double, EdgeHash> cotangentEdgeWeights(const Mesh& mesh, double min_weight) {
  std::unordered_map<Edge, double, EdgeHash> weights;

  auto cotAt = [&](int a, int b, int c) {
    const Eigen::Vector3d u = mesh.vertices[b] - mesh.vertices[a];
    const Eigen::Vector3d v = mesh.vertices[c] - mesh.vertices[a];
    const double denom = u.cross(v).norm();
    if (denom <= 1e-14) {
      return 0.0;
    }
    return u.dot(v) / denom;
  };

  for (const auto& f : mesh.faces) {
    weights[makeEdge(f[1], f[2])] += cotAt(f[0], f[1], f[2]);
    weights[makeEdge(f[2], f[0])] += cotAt(f[1], f[2], f[0]);
    weights[makeEdge(f[0], f[1])] += cotAt(f[2], f[0], f[1]);
  }

  for (auto& item : weights) {
    item.second *= 0.5;
    if (!std::isfinite(item.second) || item.second <= 0.0) {
      item.second = min_weight;
    }
  }
  return weights;
}

double triangleArea(const Eigen::Vector3d& a, const Eigen::Vector3d& b, const Eigen::Vector3d& c) {
  return 0.5 * (b - a).cross(c - a).norm();
}

double triangleQuality(const Eigen::Vector3d& a, const Eigen::Vector3d& b, const Eigen::Vector3d& c) {
  const double l0 = (b - a).norm();
  const double l1 = (c - b).norm();
  const double l2 = (a - c).norm();
  const double half_p = 0.5 * (l0 + l1 + l2);
  const double longest = std::max({l0, l1, l2});
  const double denom = std::sqrt(3.0) * half_p * longest;
  if (denom <= 1e-14) {
    return 0.0;
  }
  return 6.0 * triangleArea(a, b, c) / denom;
}

std::vector<double> triangleQualities(const Mesh& mesh) {
  std::vector<double> result;
  result.reserve(mesh.faces.size());
  for (const auto& f : mesh.faces) {
    result.push_back(triangleQuality(mesh.vertices[f[0]], mesh.vertices[f[1]], mesh.vertices[f[2]]));
  }
  return result;
}

double maxInnerAngle(const Eigen::Vector3d& a, const Eigen::Vector3d& b, const Eigen::Vector3d& c) {
  auto angle = [](const Eigen::Vector3d& p, const Eigen::Vector3d& q, const Eigen::Vector3d& r) {
    const Eigen::Vector3d u = q - p;
    const Eigen::Vector3d v = r - p;
    const double denom = u.norm() * v.norm();
    if (denom <= 1e-14) {
      return std::acos(-1.0);
    }
    const double cosv = std::clamp(u.dot(v) / denom, -1.0, 1.0);
    return std::acos(cosv);
  };
  return std::max({angle(a, b, c), angle(b, c, a), angle(c, a, b)});
}

}  // namespace adaiso
