#include "adaiso/Geometry.hpp"

#include <Eigen/Geometry>

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace adaiso {
namespace {

constexpr double kGeomEps = 1e-14;
constexpr double kTwoPi = 6.283185307179586476925286766559005768;

bool finiteDouble(double value) {
  return std::isfinite(value);
}

double angleAt(
    const Eigen::Vector3d& center,
    const Eigen::Vector3d& a,
    const Eigen::Vector3d& b) {
  const Eigen::Vector3d u = a - center;
  const Eigen::Vector3d v = b - center;
  const double denom = u.norm() * v.norm();
  if (denom <= kGeomEps) {
    return 0.0;
  }
  return std::acos(std::clamp(u.dot(v) / denom, -1.0, 1.0));
}

double rawCotAt(
    const Eigen::Vector3d& center,
    const Eigen::Vector3d& a,
    const Eigen::Vector3d& b) {
  const Eigen::Vector3d u = a - center;
  const Eigen::Vector3d v = b - center;
  const double denom = u.cross(v).norm();
  if (denom <= kGeomEps) {
    return 0.0;
  }
  const double value = u.dot(v) / denom;
  return finiteDouble(value) ? value : 0.0;
}

std::unordered_map<Edge, double, EdgeHash> rawCotangentEdgeWeights(const Mesh& mesh) {
  std::unordered_map<Edge, double, EdgeHash> weights;
  for (const auto& f : mesh.faces) {
    weights[makeEdge(f[1], f[2])] += rawCotAt(mesh.vertices[f[0]], mesh.vertices[f[1]], mesh.vertices[f[2]]);
    weights[makeEdge(f[2], f[0])] += rawCotAt(mesh.vertices[f[1]], mesh.vertices[f[2]], mesh.vertices[f[0]]);
    weights[makeEdge(f[0], f[1])] += rawCotAt(mesh.vertices[f[2]], mesh.vertices[f[0]], mesh.vertices[f[1]]);
  }
  for (auto& item : weights) {
    item.second *= 0.5;
    if (!finiteDouble(item.second)) {
      item.second = 0.0;
    }
  }
  return weights;
}

}  // namespace

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

std::vector<double> maximumAbsolutePrincipalCurvatures(const Mesh& mesh) {
  std::vector<double> curvature(mesh.vertexCount(), 0.0);
  if (mesh.vertices.empty() || mesh.faces.empty()) {
    return curvature;
  }

  const auto boundary = boundaryVertices(mesh);
  const auto weights = rawCotangentEdgeWeights(mesh);
  std::vector<double> area(mesh.vertexCount(), 0.0);
  std::vector<double> angle_sum(mesh.vertexCount(), 0.0);
  std::vector<Eigen::Vector3d> laplace_sum(mesh.vertexCount(), Eigen::Vector3d::Zero());

  for (const auto& face : mesh.faces) {
    const int a = face[0];
    const int b = face[1];
    const int c = face[2];
    const double tri_area = triangleArea(mesh.vertices[a], mesh.vertices[b], mesh.vertices[c]);
    if (tri_area <= kGeomEps || !finiteDouble(tri_area)) {
      continue;
    }

    area[static_cast<std::size_t>(a)] += tri_area / 3.0;
    area[static_cast<std::size_t>(b)] += tri_area / 3.0;
    area[static_cast<std::size_t>(c)] += tri_area / 3.0;

    angle_sum[static_cast<std::size_t>(a)] += angleAt(mesh.vertices[a], mesh.vertices[b], mesh.vertices[c]);
    angle_sum[static_cast<std::size_t>(b)] += angleAt(mesh.vertices[b], mesh.vertices[c], mesh.vertices[a]);
    angle_sum[static_cast<std::size_t>(c)] += angleAt(mesh.vertices[c], mesh.vertices[a], mesh.vertices[b]);
  }

  for (const auto& item : weights) {
    const int a = item.first.first;
    const int b = item.first.second;
    const double w = item.second;
    if (!finiteDouble(w) || static_cast<std::size_t>(a) >= mesh.vertices.size() ||
        static_cast<std::size_t>(b) >= mesh.vertices.size()) {
      continue;
    }
    const Eigen::Vector3d delta = mesh.vertices[static_cast<std::size_t>(b)] - mesh.vertices[static_cast<std::size_t>(a)];
    laplace_sum[static_cast<std::size_t>(a)] += w * delta;
    laplace_sum[static_cast<std::size_t>(b)] -= w * delta;
  }

  for (std::size_t i = 0; i < mesh.vertices.size(); ++i) {
    if (boundary.find(static_cast<int>(i)) != boundary.end() || area[i] <= kGeomEps || !finiteDouble(area[i])) {
      continue;
    }

    const Eigen::Vector3d delta_x = laplace_sum[i] / area[i];
    const double mean_curvature = 0.5 * delta_x.norm();
    const double gaussian_curvature = (kTwoPi - angle_sum[i]) / area[i];
    if (!finiteDouble(mean_curvature) || !finiteDouble(gaussian_curvature)) {
      continue;
    }

    const double discriminant = std::max(mean_curvature * mean_curvature - gaussian_curvature, 0.0);
    const double value = mean_curvature + std::sqrt(discriminant);
    curvature[i] = finiteDouble(value) && value > 0.0 ? value : 0.0;
  }
  return curvature;
}

std::vector<double> curvatureSizingField(
    const Mesh& mesh,
    double epsilon,
    double min_length,
    double max_length) {
  if (!finiteDouble(epsilon) || !finiteDouble(min_length) || !finiteDouble(max_length) || epsilon <= 0.0 ||
      min_length <= 0.0 || max_length <= 0.0 || min_length > max_length) {
    throw std::runtime_error("invalid curvature sizing parameters");
  }

  const auto curvature = maximumAbsolutePrincipalCurvatures(mesh);
  std::vector<double> sizing(curvature.size(), max_length);
  constexpr double kCurvatureEps = 1e-14;

  for (std::size_t i = 0; i < curvature.size(); ++i) {
    const double kappa = curvature[i];
    if (!finiteDouble(kappa) || kappa <= kCurvatureEps) {
      sizing[i] = max_length;
      continue;
    }

    const double radicand = (6.0 * epsilon / kappa) - (3.0 * epsilon * epsilon);
    double length = min_length;
    if (finiteDouble(radicand) && radicand > 0.0) {
      length = std::sqrt(radicand);
    }
    if (!finiteDouble(length)) {
      length = min_length;
    }
    sizing[i] = std::clamp(length, min_length, max_length);
  }
  return sizing;
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
