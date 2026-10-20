#include "adaiso/detail/RawCurvature.hpp"

#include "adaiso/Geometry.hpp"

#include <Eigen/Geometry>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <unordered_map>

namespace adaiso {
namespace {

constexpr double kEps = 1e-14;
constexpr double kRobustMeanCotMax = 10.0;
constexpr double kRobustMeanAreaFloorFactor = 1e-8;

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

namespace detail {

RobustMeanCurvatureFields computeRobustMeanCurvatureFields(
    const Mesh& mesh,
    CurvatureSignalMode mode) {
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

  RobustMeanCurvatureFields result;
  result.physical_mean_curvature.assign(mesh.vertexCount(), 0.0);
  result.raw_signal.assign(mesh.vertexCount(), 0.0);
  for (std::size_t i = 0; i < result.raw_signal.size(); ++i) {
    const double area = std::max(areas[i], area_floor);
    if (!std::isfinite(area) || area <= 0.0) {
      throw std::runtime_error("robust mean curvature produced an invalid vertex area");
    }
    const double mean_curvature = 0.5 * laplacian[i].norm() / area;
    double signal = 0.0;
    switch (mode) {
      case CurvatureSignalMode::LogDimensionless:
        signal = std::log1p(mean_curvature * mean_length);
        break;
      case CurvatureSignalMode::Dimensionless:
        signal = mean_curvature * mean_length;
        break;
      case CurvatureSignalMode::Physical:
        signal = mean_curvature;
        break;
    }
    if (!std::isfinite(mean_curvature)) {
      throw std::runtime_error("robust mean curvature produced a non-finite physical value");
    }
    if (!std::isfinite(signal)) {
      throw std::runtime_error("robust mean curvature produced a non-finite value");
    }
    result.physical_mean_curvature[i] = mean_curvature;
    result.raw_signal[i] = signal;
  }
  return result;
}

std::vector<double> computeRobustMeanCurvatureField(const Mesh& mesh) {
  return computeRobustMeanCurvatureFields(
             mesh,
             CurvatureSignalMode::LogDimensionless)
      .raw_signal;
}

}  // namespace detail
}  // namespace adaiso
