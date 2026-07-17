#include "adaiso/detail/FeatureClassification.hpp"

#include "adaiso/Geometry.hpp"

#include <Eigen/Geometry>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <unordered_set>

namespace adaiso::detail {
namespace {

constexpr double kEps = 1e-14;
constexpr double kPi = 3.141592653589793238462643383279502884;
constexpr double kRelocationFeatureEps = 1e-12;
constexpr double kRelocationZeta = 0.5;

Eigen::Vector3d safeFaceNormal(const std::vector<Eigen::Vector3d>& vertices, const Eigen::Vector3i& face) {
  return safeNormalize((vertices[face[1]] - vertices[face[0]]).cross(vertices[face[2]] - vertices[face[0]]));
}

double faceNormalAngle(const std::vector<Eigen::Vector3d>& vertices, const Eigen::Vector3i& a, const Eigen::Vector3i& b) {
  const Eigen::Vector3d n0 = safeFaceNormal(vertices, a);
  const Eigen::Vector3d n1 = safeFaceNormal(vertices, b);
  if (n0.norm() <= kEps || n1.norm() <= kEps) {
    return std::numeric_limits<double>::infinity();
  }
  return std::acos(std::clamp(n0.dot(n1), -1.0, 1.0));
}

double cornerAngle(
    const std::vector<Eigen::Vector3d>& vertices,
    const Eigen::Vector3i& face,
    int corner) {
  const Eigen::Vector3d& p = vertices[static_cast<std::size_t>(face[corner])];
  const Eigen::Vector3d& q = vertices[static_cast<std::size_t>(face[(corner + 1) % 3])];
  const Eigen::Vector3d& r = vertices[static_cast<std::size_t>(face[(corner + 2) % 3])];
  const Eigen::Vector3d u = q - p;
  const Eigen::Vector3d v = r - p;
  const double denom = u.norm() * v.norm();
  if (denom <= kEps || !std::isfinite(denom)) {
    return 0.0;
  }
  return std::acos(std::clamp(u.dot(v) / denom, -1.0, 1.0));
}

}  // namespace

std::vector<double> computeVertexFeatureStrengths(const Mesh& mesh) {
  const std::size_t n = mesh.vertexCount();
  std::vector<double> theta_sum(n, 0.0);
  for (const auto& face : mesh.faces) {
    for (int corner = 0; corner < 3; ++corner) {
      theta_sum[static_cast<std::size_t>(face[corner])] += cornerAngle(mesh.vertices, face, corner);
    }
  }

  const auto e2f = edgeFaces(mesh);
  std::unordered_set<int> boundary;
  std::vector<double> max_dihedral(n, 0.0);
  for (const auto& item : e2f) {
    const Edge& edge = item.first;
    const auto& incident = item.second;
    if (incident.size() == 1) {
      boundary.insert(edge.first);
      boundary.insert(edge.second);
      continue;
    }
    if (incident.size() != 2) {
      continue;
    }
    const int f0 = incident[0];
    const int f1 = incident[1];
    if (f0 < 0 || f1 < 0 || static_cast<std::size_t>(f0) >= mesh.faces.size() ||
        static_cast<std::size_t>(f1) >= mesh.faces.size()) {
      continue;
    }
    const double angle = faceNormalAngle(
        mesh.vertices,
        mesh.faces[static_cast<std::size_t>(f0)],
        mesh.faces[static_cast<std::size_t>(f1)]);
    if (!std::isfinite(angle)) {
      continue;
    }
    max_dihedral[static_cast<std::size_t>(edge.first)] =
        std::max(max_dihedral[static_cast<std::size_t>(edge.first)], angle);
    max_dihedral[static_cast<std::size_t>(edge.second)] =
        std::max(max_dihedral[static_cast<std::size_t>(edge.second)], angle);
  }

  auto tau = [](double value) {
    if (!std::isfinite(value) || value <= 0.0) {
      return 0.0;
    }
    return std::min(kPi, 2.0 * value);
  };

  std::vector<double> strengths(n, 0.0);
  for (std::size_t i = 0; i < n; ++i) {
    const double reference_angle =
        boundary.find(static_cast<int>(i)) == boundary.end() ? 2.0 * kPi : kPi;
    const double angle_defect = reference_angle - theta_sum[i];
    const double k_feature = tau(std::abs(angle_defect));
    const double e_feature = tau(max_dihedral[i]);
    strengths[i] = (k_feature + 1.0) * (e_feature + 1.0) - 1.0;
  }
  return strengths;
}

std::vector<VertexRelocationClass> classifyVerticesForRelocation(const Mesh& mesh) {
  const auto boundary = boundaryVertices(mesh);
  const auto neigh = vertexNeighbors(mesh);
  const auto feature_strengths = computeVertexFeatureStrengths(mesh);
  std::vector<VertexRelocationClass> classes(mesh.vertexCount(), VertexRelocationClass::Smooth);

  for (std::size_t i = 0; i < mesh.vertexCount(); ++i) {
    if (boundary.find(static_cast<int>(i)) != boundary.end()) {
      classes[i] = VertexRelocationClass::Corner;
      continue;
    }

    const double center_strength = feature_strengths[i];
    if (!std::isfinite(center_strength) || center_strength <= kRelocationFeatureEps) {
      classes[i] = VertexRelocationClass::Smooth;
      continue;
    }

    const std::size_t n = neigh[i].size();
    if (n == 0) {
      classes[i] = VertexRelocationClass::Corner;
      continue;
    }

    const double threshold = kRelocationZeta * center_strength;
    std::size_t important_count = 0;
    for (int neighbor : neigh[i]) {
      if (neighbor < 0 || static_cast<std::size_t>(neighbor) >= feature_strengths.size()) {
        continue;
      }
      const double neighbor_strength = feature_strengths[static_cast<std::size_t>(neighbor)];
      if (std::isfinite(neighbor_strength) && neighbor_strength >= threshold) {
        ++important_count;
      }
    }

    const std::size_t d_corner = important_count;
    const std::size_t d_crease = important_count > 2 ? important_count - 2 : 2 - important_count;
    const std::size_t d_smooth = important_count > n ? important_count - n : n - important_count;
    if (d_corner <= d_crease && d_corner <= d_smooth) {
      classes[i] = VertexRelocationClass::Corner;
    } else if (d_crease <= d_smooth) {
      classes[i] = VertexRelocationClass::Crease;
    } else {
      classes[i] = VertexRelocationClass::Smooth;
    }
  }

  return classes;
}

}  // namespace adaiso::detail
