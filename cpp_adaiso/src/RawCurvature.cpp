#include "adaiso/detail/RawCurvature.hpp"

#include "adaiso/Geometry.hpp"

#include <Eigen/Geometry>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <stdexcept>
#include <unordered_map>

namespace adaiso {
namespace {

constexpr double kEps = 1e-14;
constexpr double kRobustMeanCotMax = 10.0;
constexpr double kRobustMeanAreaFloorFactor = 1e-8;

std::string lowerString(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  return value;
}

double clampedPositiveCotangent(
    const Eigen::Vector3d& a,
    const Eigen::Vector3d& b,
    const Eigen::Vector3d& c) {
  const Eigen::Vector3d u = b - a;
  const Eigen::Vector3d v = c - a;
  const double denom = u.cross(v).norm();
  if (denom <= kEps || !std::isfinite(denom)) {
    return 0.0;
  }
  const double cot = u.dot(v) / denom;
  if (!std::isfinite(cot) || cot <= 0.0) {
    return 0.0;
  }
  return std::min(cot, kRobustMeanCotMax);
}

std::vector<double> barycentricVertexAreas(const Mesh& mesh) {
  std::vector<double> areas(mesh.vertexCount(), 0.0);
  for (const auto& face : mesh.faces) {
    const double area = triangleArea(
        mesh.vertices[face[0]], mesh.vertices[face[1]], mesh.vertices[face[2]]);
    if (!std::isfinite(area) || area <= 0.0) {
      throw std::runtime_error(
          "robust mean curvature requires positive finite face areas");
    }
    const double share = area / 3.0;
    areas[static_cast<std::size_t>(face[0])] += share;
    areas[static_cast<std::size_t>(face[1])] += share;
    areas[static_cast<std::size_t>(face[2])] += share;
  }
  return areas;
}

std::unordered_map<Edge, double, EdgeHash> robustMeanCotangentWeights(
    const Mesh& mesh) {
  std::unordered_map<Edge, double, EdgeHash> weights;
  for (const auto& f : mesh.faces) {
    weights[makeEdge(f[1], f[2])] +=
        0.5 * clampedPositiveCotangent(
                  mesh.vertices[f[0]], mesh.vertices[f[1]], mesh.vertices[f[2]]);
    weights[makeEdge(f[2], f[0])] +=
        0.5 * clampedPositiveCotangent(
                  mesh.vertices[f[1]], mesh.vertices[f[2]], mesh.vertices[f[0]]);
    weights[makeEdge(f[0], f[1])] +=
        0.5 * clampedPositiveCotangent(
                  mesh.vertices[f[2]], mesh.vertices[f[0]], mesh.vertices[f[1]]);
  }
  return weights;
}

}  // namespace

CurvatureMode parseCurvatureMode(const std::string& value) {
  const std::string lower = lowerString(value);
  if (lower == "normal-angle" || lower == "normal_angle") {
    return CurvatureMode::NormalAngle;
  }
  if (lower == "robust-mean" || lower == "robust_mean") {
    return CurvatureMode::RobustMean;
  }
  throw std::runtime_error("unknown curvature mode: " + value);
}

std::string toString(CurvatureMode mode) {
  switch (mode) {
    case CurvatureMode::NormalAngle:
      return "normal-angle";
    case CurvatureMode::RobustMean:
      return "robust-mean";
  }
  return "unknown";
}

namespace detail {

std::vector<double> computeNormalAngleCurvatureField(const Mesh& mesh) {
  mesh.validate();
  if (mesh.vertices.empty()) {
    throw std::runtime_error("normal-angle curvature requires a non-empty mesh");
  }

  const auto normals = vertexNormals(mesh);
  const auto neigh = vertexNeighbors(mesh);
  std::vector<double> curvature(mesh.vertexCount(), 0.0);

  for (std::size_t i = 0; i < neigh.size(); ++i) {
    if (neigh[i].empty()) {
      continue;
    }
    double accum = 0.0;
    int count = 0;
    for (int j : neigh[i]) {
      const double dot = std::clamp(
          normals[i].dot(normals[static_cast<std::size_t>(j)]), -1.0, 1.0);
      accum += std::acos(dot);
      ++count;
    }
    curvature[i] = accum / static_cast<double>(count);
  }
  return curvature;
}

std::vector<double> computeRobustMeanCurvatureField(const Mesh& mesh) {
  mesh.validate();
  if (mesh.vertices.empty()) {
    throw std::runtime_error("robust mean curvature requires a non-empty mesh");
  }

  const double mean_length = meanEdgeLength(mesh);
  if (!std::isfinite(mean_length) || mean_length <= 0.0) {
    throw std::runtime_error(
        "robust mean curvature requires a positive mean edge length");
  }

  const double area_floor =
      kRobustMeanAreaFloorFactor * mean_length * mean_length;
  if (!std::isfinite(area_floor) || area_floor <= 0.0) {
    throw std::runtime_error("robust mean curvature produced an invalid area floor");
  }

  const std::vector<double> areas = barycentricVertexAreas(mesh);
  const auto weights = robustMeanCotangentWeights(mesh);
  std::vector<Eigen::Vector3d> laplacian(
      mesh.vertexCount(), Eigen::Vector3d::Zero());

  for (const auto& item : weights) {
    const int a = item.first.first;
    const int b = item.first.second;
    const double weight = item.second;
    if (!std::isfinite(weight) || weight <= 0.0) {
      continue;
    }
    const auto& pa = mesh.vertices[static_cast<std::size_t>(a)];
    const auto& pb = mesh.vertices[static_cast<std::size_t>(b)];
    laplacian[static_cast<std::size_t>(a)] += weight * (pb - pa);
    laplacian[static_cast<std::size_t>(b)] += weight * (pa - pb);
  }

  std::vector<double> curvature(mesh.vertexCount(), 0.0);
  for (std::size_t i = 0; i < curvature.size(); ++i) {
    const double area = std::max(areas[i], area_floor);
    if (!std::isfinite(area) || area <= 0.0) {
      throw std::runtime_error("robust mean curvature produced an invalid vertex area");
    }
    const double mean_curvature = 0.5 * laplacian[i].norm() / area;
    const double signal = std::log1p(mean_curvature * mean_length);
    if (!std::isfinite(signal)) {
      throw std::runtime_error("robust mean curvature produced a non-finite value");
    }
    curvature[i] = signal;
  }
  return curvature;
}

std::vector<double> computeRawCurvatureField(
    const Mesh& mesh,
    CurvatureMode mode) {
  switch (mode) {
    case CurvatureMode::NormalAngle:
      return computeNormalAngleCurvatureField(mesh);
    case CurvatureMode::RobustMean:
      return computeRobustMeanCurvatureField(mesh);
  }
  throw std::runtime_error("unknown curvature mode");
}

}  // namespace detail
}  // namespace adaiso
