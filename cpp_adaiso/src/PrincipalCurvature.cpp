#include "adaiso/detail/PrincipalCurvature.hpp"

#include "adaiso/Geometry.hpp"

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/SVD>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <unordered_set>
#include <vector>

namespace adaiso::detail {
namespace {

constexpr double kNormalEpsilon = 1e-12;

bool finiteVector(const Eigen::Vector3d& value) {
  return value.allFinite();
}

std::vector<int> sortedOneRing(
    std::size_t vertex,
    const std::vector<std::unordered_set<int>>& neighbors) {
  std::vector<int> result(neighbors[vertex].begin(), neighbors[vertex].end());
  std::sort(result.begin(), result.end());
  return result;
}

std::vector<int> sortedTwoRing(
    std::size_t vertex,
    const std::vector<std::unordered_set<int>>& neighbors) {
  std::unordered_set<int> gathered = neighbors[vertex];
  for (int adjacent : neighbors[vertex]) {
    const auto& second = neighbors[static_cast<std::size_t>(adjacent)];
    gathered.insert(second.begin(), second.end());
  }
  gathered.erase(static_cast<int>(vertex));
  std::vector<int> result(gathered.begin(), gathered.end());
  std::sort(result.begin(), result.end());
  return result;
}

bool tangentBasis(
    const Eigen::Vector3d& input_normal,
    Eigen::Vector3d& normal,
    Eigen::Vector3d& tangent1,
    Eigen::Vector3d& tangent2) {
  if (!finiteVector(input_normal)) {
    return false;
  }
  const double length = input_normal.norm();
  if (!std::isfinite(length) || length <= kNormalEpsilon) {
    return false;
  }
  normal = input_normal / length;

  int axis_index = 0;
  if (std::abs(normal[1]) < std::abs(normal[axis_index])) {
    axis_index = 1;
  }
  if (std::abs(normal[2]) < std::abs(normal[axis_index])) {
    axis_index = 2;
  }
  Eigen::Vector3d axis = Eigen::Vector3d::Zero();
  axis[axis_index] = 1.0;
  tangent1 = normal.cross(axis);
  const double tangent_length = tangent1.norm();
  if (!std::isfinite(tangent_length) || tangent_length <= kNormalEpsilon) {
    return false;
  }
  tangent1 /= tangent_length;
  tangent2 = normal.cross(tangent1);
  return finiteVector(tangent2) && tangent2.norm() > kNormalEpsilon;
}

double localScale(
    const Mesh& mesh,
    std::size_t vertex,
    const std::vector<int>& one_ring) {
  double sum = 0.0;
  std::size_t count = 0;
  for (int adjacent : one_ring) {
    const double length =
        (mesh.vertices[static_cast<std::size_t>(adjacent)] - mesh.vertices[vertex]).norm();
    if (std::isfinite(length) && length > 0.0) {
      sum += length;
      ++count;
    }
  }
  return count == 0 ? 0.0 : sum / static_cast<double>(count);
}

bool fitQuadraticPatch(
    const Mesh& mesh,
    std::size_t vertex,
    const std::vector<int>& samples,
    const Eigen::Vector3d& normal,
    const Eigen::Vector3d& tangent1,
    const Eigen::Vector3d& tangent2,
    double local_scale,
    double& k1,
    double& k2) {
  if (samples.size() < 3 || !std::isfinite(local_scale) || local_scale <= 0.0) {
    return false;
  }

  std::vector<Eigen::Vector3d> rows;
  std::vector<double> rhs_values;
  rows.reserve(samples.size());
  rhs_values.reserve(samples.size());
  for (int sample : samples) {
    const Eigen::Vector3d delta =
        mesh.vertices[static_cast<std::size_t>(sample)] - mesh.vertices[vertex];
    if (!finiteVector(delta)) {
      continue;
    }
    const double x = delta.dot(tangent1) / local_scale;
    const double y = delta.dot(tangent2) / local_scale;
    const double z = delta.dot(normal) / local_scale;
    const double radius_squared = x * x + y * y;
    if (!std::isfinite(radius_squared) || radius_squared <= kNormalEpsilon || !std::isfinite(z)) {
      continue;
    }
    const double sqrt_weight = std::exp(-0.25 * radius_squared);
    if (!std::isfinite(sqrt_weight) || sqrt_weight <= 0.0) {
      continue;
    }
    rows.push_back(sqrt_weight * Eigen::Vector3d(0.5 * x * x, x * y, 0.5 * y * y));
    rhs_values.push_back(sqrt_weight * z);
  }
  if (rows.size() < 3) {
    return false;
  }

  Eigen::MatrixXd matrix(static_cast<Eigen::Index>(rows.size()), 3);
  Eigen::VectorXd rhs(static_cast<Eigen::Index>(rows.size()));
  for (std::size_t row = 0; row < rows.size(); ++row) {
    matrix.row(static_cast<Eigen::Index>(row)) = rows[row].transpose();
    rhs[static_cast<Eigen::Index>(row)] = rhs_values[row];
  }

  Eigen::JacobiSVD<Eigen::MatrixXd> svd(matrix, Eigen::ComputeThinU | Eigen::ComputeThinV);
  const Eigen::VectorXd singular = svd.singularValues();
  if (singular.size() != 3 || !singular.allFinite() || singular[0] <= 0.0 ||
      singular[2] / singular[0] < kPrincipalFitMinSingularRatio) {
    return false;
  }
  const Eigen::Vector3d coefficients = svd.solve(rhs);
  if (!finiteVector(coefficients)) {
    return false;
  }

  const double a = coefficients[0];
  const double b = coefficients[1];
  const double c = coefficients[2];
  const double trace_half = 0.5 * (a + c);
  const double discriminant = std::sqrt(std::max(0.0, 0.25 * (a - c) * (a - c) + b * b));
  k1 = (trace_half + discriminant) / local_scale;
  k2 = (trace_half - discriminant) / local_scale;
  if (!std::isfinite(k1) || !std::isfinite(k2)) {
    return false;
  }
  if (std::abs(k2) > std::abs(k1)) {
    std::swap(k1, k2);
  }
  return true;
}

}  // namespace

PrincipalCurvatureResult computePrincipalCurvatures(
    const Mesh& mesh,
    double mean_edge_length) {
  mesh.validate();
  if (mesh.vertices.empty() || mesh.faces.empty()) {
    throw std::runtime_error("principal curvature requires a non-empty mesh");
  }
  if (!std::isfinite(mean_edge_length) || mean_edge_length <= 0.0) {
    throw std::runtime_error("principal curvature requires a positive mean edge length");
  }

  const auto normals = vertexNormals(mesh);
  const auto neighbors = vertexNeighbors(mesh);
  PrincipalCurvatureResult result;
  result.k1.assign(mesh.vertexCount(), 0.0);
  result.k2.assign(mesh.vertexCount(), 0.0);
  result.kappa.assign(mesh.vertexCount(), 0.0);

  for (std::size_t vertex = 0; vertex < mesh.vertexCount(); ++vertex) {
    Eigen::Vector3d normal;
    Eigen::Vector3d tangent1;
    Eigen::Vector3d tangent2;
    const std::vector<int> one_ring = sortedOneRing(vertex, neighbors);
    const double local_scale = localScale(mesh, vertex, one_ring);
    double k1 = 0.0;
    double k2 = 0.0;
    bool fitted = tangentBasis(normals[vertex], normal, tangent1, tangent2) &&
        std::isfinite(local_scale) &&
        local_scale > kNormalEpsilon * mean_edge_length &&
        fitQuadraticPatch(
            mesh,
            vertex,
            one_ring,
            normal,
            tangent1,
            tangent2,
            local_scale,
            k1,
            k2);
    if (!fitted && tangentBasis(normals[vertex], normal, tangent1, tangent2) &&
        std::isfinite(local_scale) && local_scale > kNormalEpsilon * mean_edge_length) {
      fitted = fitQuadraticPatch(
          mesh,
          vertex,
          sortedTwoRing(vertex, neighbors),
          normal,
          tangent1,
          tangent2,
          local_scale,
          k1,
          k2);
    }
    if (!fitted) {
      ++result.fallback_vertices;
      continue;
    }
    result.k1[vertex] = k1;
    result.k2[vertex] = k2;
    result.kappa[vertex] = mean_edge_length * std::max(std::abs(k1), std::abs(k2));
  }
  return result;
}

}  // namespace adaiso::detail
