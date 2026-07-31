#include "adaiso/AdaIsoRemesher.hpp"

#include "adaiso/Geometry.hpp"
#include "adaiso/IO.hpp"
#include "adaiso/KDTree.hpp"
#include "adaiso/Metrics.hpp"
#include "adaiso/detail/CsfSolver.hpp"
#include "adaiso/detail/TargetErrorSizing.hpp"

#include <Eigen/Geometry>

#include <igl/barycenter.h>
#include <igl/box_surface_area.h>
#include <igl/doublearea.h>
#include <igl/EPS.h>
#include <igl/pad_box.h>
#include <igl/point_simplex_squared_distance.h>
#include <igl/principal_curvature.h>
#include <igl/project_to_line_segment.h>
#include <igl/ray_box_intersect.h>
#include <igl/volume.h>

// vcpkg's libigl is header-only here. Keep ray_mesh_intersect declarations
// visible, but do not include its body because this package lacks iglraytri.c.
#ifndef IGL_STATIC_LIBRARY
#define ADAISO_DEFINED_IGL_STATIC_LIBRARY
#define IGL_STATIC_LIBRARY
#endif
#include <igl/ray_mesh_intersect.h>
#include <igl/AABB.h>
#ifdef ADAISO_DEFINED_IGL_STATIC_LIBRARY
#undef IGL_STATIC_LIBRARY
#undef ADAISO_DEFINED_IGL_STATIC_LIBRARY
#endif
#include <igl/AABB.cpp>

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace adaiso {
namespace {

constexpr double kEps = 1e-14;
constexpr double kPi = 3.141592653589793238462643383279502884;
constexpr double kRobustMeanCotMax = 10.0;
constexpr double kRobustMeanAreaFloorFactor = 1e-8;
constexpr double kRelocationFeatureEps = 1e-12;
constexpr double kRelocationZeta = 0.5;
constexpr double kTargetKappaMin = 1e-12;
constexpr double kTargetHMinSafeFactor = 0.25;
constexpr double kEquilateralAreaFactor = 0.43301270189221932338;
constexpr double kMinAngleTolDegrees = 1e-8;
constexpr double kTargetLogEnergyTol = 1e-8;
constexpr double kStageAProjectionSizeEnergyTol = 1e-4;
constexpr double kStageAProjectionMinDistanceEps = kEps;
constexpr int kCollapseConflictMiniRounds = 2;

using SurfaceAabb = igl::AABB<Eigen::MatrixXd, 3>;

bool faceContains(const Eigen::Vector3i& face, int v) {
  return face[0] == v || face[1] == v || face[2] == v;
}

bool faceContainsEdge(const Eigen::Vector3i& face, int a, int b) {
  for (int i = 0; i < 3; ++i) {
    const int x = face[i];
    const int y = face[(i + 1) % 3];
    if ((x == a && y == b) || (x == b && y == a)) {
      return true;
    }
  }
  return false;
}

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

Eigen::Vector3i orientTriangle(
    Eigen::Vector3i tri,
    const std::vector<Eigen::Vector3d>& vertices,
    const Eigen::Vector3d& ref_normal) {
  if (ref_normal.norm() <= kEps) {
    return tri;
  }
  const Eigen::Vector3d n = safeFaceNormal(vertices, tri);
  if (n.dot(ref_normal) < 0.0) {
    std::swap(tri[1], tri[2]);
  }
  return tri;
}

double quantileOfSorted(const std::vector<double>& sorted, double q) {
  if (sorted.empty()) {
    return 0.0;
  }
  if (sorted.size() == 1) {
    return sorted.front();
  }
  q = std::clamp(q, 0.0, 1.0);
  const double pos = q * static_cast<double>(sorted.size() - 1);
  const auto lo = static_cast<std::size_t>(std::floor(pos));
  const auto hi = static_cast<std::size_t>(std::ceil(pos));
  const double t = pos - static_cast<double>(lo);
  return sorted[lo] * (1.0 - t) + sorted[hi] * t;
}

bool unorderedSetsEqual(const std::unordered_set<int>& a, const std::unordered_set<int>& b) {
  if (a.size() != b.size()) {
    return false;
  }
  for (int v : a) {
    if (b.find(v) == b.end()) {
      return false;
    }
  }
  return true;
}

bool finiteVector(const Eigen::Vector3d& v) {
  return std::isfinite(v[0]) && std::isfinite(v[1]) && std::isfinite(v[2]);
}

double targetMetricSplitParameter(double h_a, double h_b) {
  if (!std::isfinite(h_a) || !std::isfinite(h_b) || h_a <= 0.0 || h_b <= 0.0) {
    return 0.5;
  }
  const double sqrt_a = std::sqrt(h_a);
  const double sqrt_b = std::sqrt(h_b);
  const double denom = sqrt_a + sqrt_b;
  if (!std::isfinite(denom) || denom <= kEps) {
    return 0.5;
  }
  return std::clamp(sqrt_a / denom, 0.25, 0.75);
}

double targetMetricLength(double length, double h_a, double h_b) {
  if (!std::isfinite(length) || !std::isfinite(h_a) || !std::isfinite(h_b) ||
      length < 0.0 || h_a <= 0.0 || h_b <= 0.0) {
    return std::numeric_limits<double>::quiet_NaN();
  }

  const double dh = h_b - h_a;
  const double scale = std::max({1.0, std::abs(h_a), std::abs(h_b)});
  if (std::abs(dh) <= kEps * scale) {
    return length / h_a;
  }

  const double metric = length * (std::log(h_b) - std::log(h_a)) / dh;
  return metric > 0.0 && std::isfinite(metric) ? metric : std::numeric_limits<double>::quiet_NaN();
}

double edgeTargetMetricLength(const Mesh& mesh, const TargetLengthField& target_field, int a, int b) {
  const auto ai = static_cast<std::size_t>(a);
  const auto bi = static_cast<std::size_t>(b);
  const double length = (mesh.vertices[ai] - mesh.vertices[bi]).norm();
  return targetMetricLength(
      length,
      target_field.valueAt(mesh.vertices[ai]),
      target_field.valueAt(mesh.vertices[bi]));
}

double innerAngleDegrees(const Eigen::Vector3d& p, const Eigen::Vector3d& q, const Eigen::Vector3d& r) {
  const Eigen::Vector3d u = q - p;
  const Eigen::Vector3d v = r - p;
  const double denom = u.norm() * v.norm();
  if (denom <= kEps || !std::isfinite(denom)) {
    return 0.0;
  }
  return std::acos(std::clamp(u.dot(v) / denom, -1.0, 1.0)) * 180.0 / kPi;
}

double minInnerAngleDegrees(const Eigen::Vector3d& a, const Eigen::Vector3d& b, const Eigen::Vector3d& c) {
  return std::min({
      innerAngleDegrees(a, b, c),
      innerAngleDegrees(b, c, a),
      innerAngleDegrees(c, a, b),
  });
}

struct AnglePatchMetrics {
  bool valid = false;
  double min_angle_degrees = 0.0;
};

Eigen::Vector3d localPatchPoint(
    const Mesh& mesh,
    const std::unordered_map<int, Eigen::Vector3d>& overrides,
    int vertex);

AnglePatchMetrics evaluateAnglePatch(
    const Mesh& mesh,
    const std::vector<Eigen::Vector3i>& faces,
    const std::unordered_map<int, Eigen::Vector3d>& overrides = {}) {
  AnglePatchMetrics metrics;
  metrics.valid = true;
  metrics.min_angle_degrees = std::numeric_limits<double>::infinity();
  bool checked_face = false;

  for (const auto& face : faces) {
    if (face[0] == face[1] || face[1] == face[2] || face[2] == face[0]) {
      metrics.valid = false;
      return metrics;
    }
    const Eigen::Vector3d p0 = localPatchPoint(mesh, overrides, face[0]);
    const Eigen::Vector3d p1 = localPatchPoint(mesh, overrides, face[1]);
    const Eigen::Vector3d p2 = localPatchPoint(mesh, overrides, face[2]);
    const double area = triangleArea(p0, p1, p2);
    if (!std::isfinite(area) || area <= kEps) {
      metrics.valid = false;
      return metrics;
    }
    checked_face = true;
    metrics.min_angle_degrees =
        std::min(metrics.min_angle_degrees, minInnerAngleDegrees(p0, p1, p2));
  }

  if (!checked_face) {
    metrics.valid = false;
    metrics.min_angle_degrees = 0.0;
  }
  return metrics;
}

struct LocalPatchMetrics {
  bool valid = false;
  double min_angle_degrees = 0.0;
  double target_metric_violation = 0.0;
  double target_log_energy = 0.0;
};

Eigen::MatrixXd verticesToMatrix(const Mesh& mesh);
Eigen::MatrixXi facesToMatrix(const Mesh& mesh);

int remappedVertex(const std::unordered_map<int, int>& replacements, int vertex) {
  const auto it = replacements.find(vertex);
  return it == replacements.end() ? vertex : it->second;
}

Eigen::Vector3d localPatchPoint(
    const Mesh& mesh,
    const std::unordered_map<int, Eigen::Vector3d>& overrides,
    int vertex) {
  const auto it = overrides.find(vertex);
  if (it != overrides.end()) {
    return it->second;
  }
  return mesh.vertices[static_cast<std::size_t>(vertex)];
}

double localEdgeTargetRatio(
    const TargetLengthField& target_field,
    const Eigen::Vector3d& a,
    const Eigen::Vector3d& b) {
  return targetMetricLength(
      (a - b).norm(),
      target_field.valueAt(a),
      target_field.valueAt(b));
}

double localEdgeMetricViolation(double ratio, double collapse_factor, double split_factor) {
  if (!std::isfinite(ratio)) {
    return std::numeric_limits<double>::infinity();
  }
  return std::max(0.0, collapse_factor - ratio) + std::max(0.0, ratio - split_factor);
}

double localEdgeTargetLogEnergy(double ratio) {
  if (!std::isfinite(ratio) || ratio <= 0.0) {
    return std::numeric_limits<double>::infinity();
  }
  const double log_ratio = std::log(ratio);
  return log_ratio * log_ratio;
}

LocalPatchMetrics evaluateLocalPatch(
    const Mesh& mesh,
    const TargetLengthField& target_field,
    const std::vector<Eigen::Vector3i>& faces,
    const std::unordered_map<int, int>& replacements,
    const std::unordered_map<int, Eigen::Vector3d>& overrides,
    double collapse_factor,
    double split_factor,
    bool skip_collapsed_faces) {
  LocalPatchMetrics metrics;
  metrics.valid = true;
  metrics.min_angle_degrees = std::numeric_limits<double>::infinity();

  bool checked_face = false;
  std::size_t checked_edge_count = 0;
  std::unordered_set<Edge, EdgeHash> checked_edges;
  for (const auto& face : faces) {
    const int v0 = remappedVertex(replacements, face[0]);
    const int v1 = remappedVertex(replacements, face[1]);
    const int v2 = remappedVertex(replacements, face[2]);
    if (v0 == v1 || v1 == v2 || v2 == v0) {
      if (skip_collapsed_faces) {
        continue;
      }
      metrics.valid = false;
      return metrics;
    }

    const Eigen::Vector3d p0 = localPatchPoint(mesh, overrides, v0);
    const Eigen::Vector3d p1 = localPatchPoint(mesh, overrides, v1);
    const Eigen::Vector3d p2 = localPatchPoint(mesh, overrides, v2);
    const double area = triangleArea(p0, p1, p2);
    if (!std::isfinite(area) || area <= kEps) {
      metrics.valid = false;
      return metrics;
    }

    checked_face = true;
    metrics.min_angle_degrees =
        std::min(metrics.min_angle_degrees, minInnerAngleDegrees(p0, p1, p2));

    const std::array<std::pair<int, int>, 3> edges = {
        std::make_pair(v0, v1),
        std::make_pair(v1, v2),
        std::make_pair(v2, v0),
    };
    for (const auto& item : edges) {
      const Edge edge = makeEdge(item.first, item.second);
      if (!checked_edges.insert(edge).second) {
        continue;
      }
      const Eigen::Vector3d edge_start = localPatchPoint(mesh, overrides, edge.first);
      const Eigen::Vector3d edge_end = localPatchPoint(mesh, overrides, edge.second);
      const double ratio = localEdgeTargetRatio(target_field, edge_start, edge_end);
      if (!std::isfinite(ratio)) {
        metrics.valid = false;
        return metrics;
      }
      const double violation = localEdgeMetricViolation(ratio, collapse_factor, split_factor);
      if (!std::isfinite(violation)) {
        metrics.valid = false;
        return metrics;
      }
      const double log_energy = localEdgeTargetLogEnergy(ratio);
      if (!std::isfinite(log_energy)) {
        metrics.valid = false;
        return metrics;
      }
      metrics.target_metric_violation += violation;
      metrics.target_log_energy += log_energy;
      ++checked_edge_count;
    }
  }

  if (!checked_face || checked_edge_count == 0) {
    metrics.valid = false;
    metrics.min_angle_degrees = 0.0;
    metrics.target_log_energy = 0.0;
  } else {
    metrics.target_log_energy /= static_cast<double>(checked_edge_count);
  }
  return metrics;
}

LocalPatchMetrics evaluateLocalPatch(
    const Mesh& mesh,
    const TargetLengthField& target_field,
    const std::vector<Eigen::Vector3i>& faces,
    double collapse_factor,
    double split_factor) {
  return evaluateLocalPatch(
      mesh,
      target_field,
      faces,
      {},
      {},
      collapse_factor,
      split_factor,
      false);
}

int valenceObjective(
    const std::vector<int>& valence,
    const std::unordered_set<int>& boundary,
    int a,
    int b,
    int c,
    int d,
    bool after_flip) {
  auto target = [&](int v) {
    return boundary.find(v) == boundary.end() ? 6 : 4;
  };
  auto current = [&](int v) {
    return valence[static_cast<std::size_t>(v)];
  };
  const int da = after_flip ? -1 : 0;
  const int db = after_flip ? -1 : 0;
  const int dc = after_flip ? 1 : 0;
  const int dd = after_flip ? 1 : 0;
  return std::abs(current(a) + da - target(a)) +
         std::abs(current(b) + db - target(b)) +
         std::abs(current(c) + dc - target(c)) +
         std::abs(current(d) + dd - target(d));
}

bool minAngleSafetyAccepted(double before_min_angle, double after_min_angle, double floor_degrees) {
  if (!std::isfinite(before_min_angle) || !std::isfinite(after_min_angle)) {
    return false;
  }
  const double required_min_angle = std::min(before_min_angle, floor_degrees);
  return after_min_angle + kMinAngleTolDegrees >= required_min_angle;
}

bool minAngleQualityAccepted(double before_min_angle, double after_min_angle, double floor_degrees) {
  if (!std::isfinite(before_min_angle) || !std::isfinite(after_min_angle)) {
    return false;
  }
  if (before_min_angle < floor_degrees) {
    return after_min_angle > before_min_angle + kMinAngleTolDegrees;
  }
  return after_min_angle + kMinAngleTolDegrees >= before_min_angle;
}

double minQuality(const Mesh& mesh) {
  const auto qualities = triangleQualities(mesh);
  if (qualities.empty()) {
    return 0.0;
  }
  return *std::min_element(qualities.begin(), qualities.end());
}

Eigen::MatrixXd verticesToMatrix(const Mesh& mesh) {
  Eigen::MatrixXd vertices(static_cast<Eigen::Index>(mesh.vertices.size()), 3);
  for (Eigen::Index i = 0; i < vertices.rows(); ++i) {
    vertices.row(i) = mesh.vertices[static_cast<std::size_t>(i)].transpose();
  }
  return vertices;
}

Eigen::MatrixXi facesToMatrix(const Mesh& mesh) {
  Eigen::MatrixXi faces(static_cast<Eigen::Index>(mesh.faces.size()), 3);
  for (Eigen::Index i = 0; i < faces.rows(); ++i) {
    const auto& face = mesh.faces[static_cast<std::size_t>(i)];
    faces(i, 0) = face[0];
    faces(i, 1) = face[1];
    faces(i, 2) = face[2];
  }
  return faces;
}

double boundingBoxDiagonal(const Mesh& mesh) {
  if (mesh.vertices.empty()) {
    return 0.0;
  }
  Eigen::Vector3d lo = mesh.vertices.front();
  Eigen::Vector3d hi = mesh.vertices.front();
  for (const auto& vertex : mesh.vertices) {
    lo = lo.cwiseMin(vertex);
    hi = hi.cwiseMax(vertex);
  }
  return (hi - lo).norm();
}

std::vector<double> incidentMedianEdgeLengths(const Mesh& mesh) {
  std::vector<std::vector<double>> incident_lengths(mesh.vertexCount());
  for (const Edge& edge : uniqueEdges(mesh)) {
    const auto a = static_cast<std::size_t>(edge.first);
    const auto b = static_cast<std::size_t>(edge.second);
    const double length = (mesh.vertices[a] - mesh.vertices[b]).norm();
    if (!std::isfinite(length) || length <= 0.0) {
      throw std::runtime_error("target error sizing requires positive finite incident edge lengths");
    }
    incident_lengths[a].push_back(length);
    incident_lengths[b].push_back(length);
  }

  std::vector<double> medians(mesh.vertexCount(), 0.0);
  for (std::size_t i = 0; i < incident_lengths.size(); ++i) {
    auto& values = incident_lengths[i];
    if (values.empty()) {
      throw std::runtime_error("target error sizing found an isolated vertex without incident edges");
    }
    std::sort(values.begin(), values.end());
    medians[i] = quantileOfSorted(values, 0.5);
  }
  return medians;
}

double areaWeightedQuantile(
    const std::vector<double>& values,
    const std::vector<double>& weights,
    double q) {
  if (values.size() != weights.size()) {
    throw std::runtime_error("weighted quantile size mismatch");
  }
  if (!std::isfinite(q) || q < 0.0 || q > 1.0) {
    throw std::runtime_error("weighted quantile must be finite and in [0, 1]");
  }

  std::vector<std::pair<double, double>> items;
  items.reserve(values.size());
  double total_weight = 0.0;
  for (std::size_t i = 0; i < values.size(); ++i) {
    const double value = values[i];
    const double weight = weights[i];
    if (!std::isfinite(value) || !std::isfinite(weight) || weight <= 0.0) {
      continue;
    }
    items.emplace_back(value, weight);
    total_weight += weight;
  }
  if (items.empty() || !std::isfinite(total_weight) || total_weight <= 0.0) {
    return 0.0;
  }

  std::sort(items.begin(), items.end(), [](const auto& lhs, const auto& rhs) {
    return lhs.first < rhs.first;
  });
  const double threshold = q * total_weight;
  double cumulative = 0.0;
  for (const auto& item : items) {
    cumulative += item.second;
    if (cumulative + kEps >= threshold) {
      return item.first;
    }
  }
  return items.back().first;
}

Eigen::Vector3d barycentricOnTriangle(
    const Eigen::Vector3d& point,
    const Eigen::Vector3d& a,
    const Eigen::Vector3d& b,
    const Eigen::Vector3d& c) {
  const Eigen::Vector3d ab = b - a;
  const Eigen::Vector3d ac = c - a;
  const Eigen::Vector3d ap = point - a;
  const double d00 = ab.dot(ab);
  const double d01 = ab.dot(ac);
  const double d11 = ac.dot(ac);
  const double d20 = ap.dot(ab);
  const double d21 = ap.dot(ac);
  const double denom = d00 * d11 - d01 * d01;
  if (std::abs(denom) <= kEps || !std::isfinite(denom)) {
    const double da = (point - a).squaredNorm();
    const double db = (point - b).squaredNorm();
    const double dc = (point - c).squaredNorm();
    if (da <= db && da <= dc) {
      return {1.0, 0.0, 0.0};
    }
    if (db <= dc) {
      return {0.0, 1.0, 0.0};
    }
    return {0.0, 0.0, 1.0};
  }
  const double v = (d11 * d20 - d01 * d21) / denom;
  const double w = (d00 * d21 - d01 * d20) / denom;
  Eigen::Vector3d bary(1.0 - v - w, v, w);
  for (int i = 0; i < 3; ++i) {
    bary[i] = std::clamp(bary[i], 0.0, 1.0);
  }
  const double sum = bary.sum();
  if (sum <= kEps || !std::isfinite(sum)) {
    return {1.0, 0.0, 0.0};
  }
  return bary / sum;
}

struct Rgb {
  int r = 0;
  int g = 0;
  int b = 0;
};

struct TargetFieldStageData {
  std::vector<double> raw_curvature;
  std::vector<double> csf_curvature;
  std::vector<double> target_lengths;
  std::vector<double> local_edge_medians;
  std::vector<double> local_deltas;
  double base_length = 0.0;
  double min_length = 0.0;
  double max_length = 0.0;
  double target_error_delta = 0.0;
  double target_error_bbox_ratio = 0.0;
  double bbox_diagonal = 0.0;
  double h_max_factor = 5.0;
  double auto_delta_quantile = 0.5;
  double h_min_safe_factor = kTargetHMinSafeFactor;
  double kappa_max = 0.0;
  double estimated_faces = 0.0;
  bool flat_fallback = false;
};

struct EdgeFieldRecord {
  Edge edge;
  double edge_length = 0.0;
  double abs_dh = 0.0;
  double target_ratio = 1.0;
  double h_slope = 0.0;
};

struct FaceFieldRecord {
  double grad_h_norm = 0.0;
  double max_edge_target_ratio = 1.0;
};

std::ofstream openOutputFile(const std::filesystem::path& path) {
  if (path.has_parent_path()) {
    std::filesystem::create_directories(path.parent_path());
  }
  std::ofstream out(path);
  if (!out) {
    throw std::runtime_error("failed to write file: " + path.string());
  }
  out << std::setprecision(17);
  return out;
}

void writeTextFile(const std::filesystem::path& path, const std::string& text) {
  auto out = openOutputFile(path);
  out << text;
}

std::pair<double, double> scalarRange(const std::vector<double>& values) {
  double lo = std::numeric_limits<double>::infinity();
  double hi = -std::numeric_limits<double>::infinity();
  for (double value : values) {
    if (!std::isfinite(value)) {
      continue;
    }
    lo = std::min(lo, value);
    hi = std::max(hi, value);
  }
  if (!std::isfinite(lo) || !std::isfinite(hi)) {
    return {0.0, 1.0};
  }
  if (hi <= lo) {
    const double pad = std::max(1.0, std::abs(lo)) * 0.5;
    return {lo - pad, hi + pad};
  }
  return {lo, hi};
}

double normalizedScalar(double value, double lo, double hi) {
  if (!std::isfinite(value)) {
    return 0.0;
  }
  if (!std::isfinite(lo) || !std::isfinite(hi) || hi <= lo) {
    return 0.5;
  }
  return std::clamp((value - lo) / (hi - lo), 0.0, 1.0);
}

int colorByte(double value) {
  return static_cast<int>(std::round(std::clamp(value, 0.0, 1.0) * 255.0));
}

Rgb lerpColor(const Rgb& a, const Rgb& b, double t) {
  t = std::clamp(t, 0.0, 1.0);
  return {
      colorByte((1.0 - t) * static_cast<double>(a.r) / 255.0 + t * static_cast<double>(b.r) / 255.0),
      colorByte((1.0 - t) * static_cast<double>(a.g) / 255.0 + t * static_cast<double>(b.g) / 255.0),
      colorByte((1.0 - t) * static_cast<double>(a.b) / 255.0 + t * static_cast<double>(b.b) / 255.0)};
}

Rgb heatColor(double t) {
  t = std::clamp(t, 0.0, 1.0);
  const std::array<Rgb, 5> palette = {{
      {45, 70, 180},
      {0, 170, 210},
      {80, 190, 80},
      {245, 210, 70},
      {210, 40, 40},
  }};
  const double scaled = t * static_cast<double>(palette.size() - 1);
  const auto index = static_cast<std::size_t>(std::min<double>(
      std::floor(scaled),
      static_cast<double>(palette.size() - 2)));
  return lerpColor(palette[index], palette[index + 1], scaled - static_cast<double>(index));
}

void writeVertexColorPly(
    const std::filesystem::path& path,
    const Mesh& mesh,
    const std::vector<Rgb>& colors) {
  if (colors.size() != mesh.vertices.size()) {
    throw std::runtime_error("vertex color count does not match mesh vertices");
  }
  auto out = openOutputFile(path);
  out << "ply\n";
  out << "format ascii 1.0\n";
  out << "element vertex " << mesh.vertices.size() << '\n';
  out << "property float x\n";
  out << "property float y\n";
  out << "property float z\n";
  out << "property uchar red\n";
  out << "property uchar green\n";
  out << "property uchar blue\n";
  out << "element face " << mesh.faces.size() << '\n';
  out << "property list uchar int vertex_indices\n";
  out << "end_header\n";
  for (std::size_t i = 0; i < mesh.vertices.size(); ++i) {
    const auto& v = mesh.vertices[i];
    const Rgb& color = colors[i];
    out << v[0] << ' ' << v[1] << ' ' << v[2] << ' '
        << color.r << ' ' << color.g << ' ' << color.b << '\n';
  }
  for (const auto& f : mesh.faces) {
    out << "3 " << f[0] << ' ' << f[1] << ' ' << f[2] << '\n';
  }
}

void writeVertexHeatPly(
    const std::filesystem::path& path,
    const Mesh& mesh,
    const std::vector<double>& values,
    double lo,
    double hi,
    bool inverted = false) {
  if (values.size() != mesh.vertices.size()) {
    throw std::runtime_error("vertex scalar count does not match mesh vertices");
  }
  std::vector<Rgb> colors;
  colors.reserve(values.size());
  for (double value : values) {
    double t = normalizedScalar(value, lo, hi);
    if (inverted) {
      t = 1.0 - t;
    }
    colors.push_back(heatColor(t));
  }
  writeVertexColorPly(path, mesh, colors);
}

double clampedPositiveCotangent(const Eigen::Vector3d& a, const Eigen::Vector3d& b, const Eigen::Vector3d& c) {
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
    const double area = triangleArea(mesh.vertices[face[0]], mesh.vertices[face[1]], mesh.vertices[face[2]]);
    if (!std::isfinite(area) || area <= 0.0) {
      throw std::runtime_error("robust mean curvature requires positive finite face areas");
    }
    const double share = area / 3.0;
    areas[static_cast<std::size_t>(face[0])] += share;
    areas[static_cast<std::size_t>(face[1])] += share;
    areas[static_cast<std::size_t>(face[2])] += share;
  }
  return areas;
}

std::unordered_map<Edge, double, EdgeHash> robustMeanCotangentWeights(const Mesh& mesh) {
  std::unordered_map<Edge, double, EdgeHash> weights;
  for (const auto& f : mesh.faces) {
    weights[makeEdge(f[1], f[2])] +=
        0.5 * clampedPositiveCotangent(mesh.vertices[f[0]], mesh.vertices[f[1]], mesh.vertices[f[2]]);
    weights[makeEdge(f[2], f[0])] +=
        0.5 * clampedPositiveCotangent(mesh.vertices[f[1]], mesh.vertices[f[2]], mesh.vertices[f[0]]);
    weights[makeEdge(f[0], f[1])] +=
        0.5 * clampedPositiveCotangent(mesh.vertices[f[2]], mesh.vertices[f[0]], mesh.vertices[f[1]]);
  }
  return weights;
}

std::vector<double> computeNormalAngleCurvatureField(const Mesh& mesh) {
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
      const double dot = std::clamp(normals[i].dot(normals[static_cast<std::size_t>(j)]), -1.0, 1.0);
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
    throw std::runtime_error("robust mean curvature requires a positive mean edge length");
  }

  const double area_floor = kRobustMeanAreaFloorFactor * mean_length * mean_length;
  if (!std::isfinite(area_floor) || area_floor <= 0.0) {
    throw std::runtime_error("robust mean curvature produced an invalid area floor");
  }

  const std::vector<double> areas = barycentricVertexAreas(mesh);
  const auto weights = robustMeanCotangentWeights(mesh);
  std::vector<Eigen::Vector3d> laplacian(mesh.vertexCount(), Eigen::Vector3d::Zero());

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
    const double signal = mean_curvature * mean_length;
    if (!std::isfinite(signal)) {
      throw std::runtime_error("robust mean curvature produced a non-finite value");
    }
    curvature[i] = signal;
  }
  return curvature;
}

std::vector<double> computePrincipalMaxCurvatureField(const Mesh& mesh) {
  mesh.validate();
  if (mesh.vertices.empty() || mesh.faces.empty()) {
    throw std::runtime_error("principal curvature requires a non-empty mesh");
  }

  const Eigen::MatrixXd vertices = verticesToMatrix(mesh);
  const Eigen::MatrixXi faces = facesToMatrix(mesh);
  Eigen::MatrixXd pd1;
  Eigen::MatrixXd pd2;
  Eigen::VectorXd pv1;
  Eigen::VectorXd pv2;
  std::vector<int> bad_vertices;
  igl::principal_curvature(vertices, faces, pd1, pd2, pv1, pv2, bad_vertices);

  if (pv1.size() != static_cast<Eigen::Index>(mesh.vertexCount()) ||
      pv2.size() != static_cast<Eigen::Index>(mesh.vertexCount())) {
    throw std::runtime_error("principal curvature returned an unexpected vertex count");
  }

  std::vector<double> curvature(mesh.vertexCount(), kTargetKappaMin);
  for (std::size_t i = 0; i < curvature.size(); ++i) {
    const double k1 = std::abs(pv1(static_cast<Eigen::Index>(i)));
    const double k2 = std::abs(pv2(static_cast<Eigen::Index>(i)));
    const double value = std::max(k1, k2);
    curvature[i] = std::isfinite(value) ? value : kTargetKappaMin;
  }
  return curvature;
}

std::vector<double> computeRawCurvatureField(const Mesh& mesh, CurvatureMode mode) {
  switch (mode) {
    case CurvatureMode::NormalAngle:
      return computeNormalAngleCurvatureField(mesh);
    case CurvatureMode::RobustMean:
      return computeRobustMeanCurvatureField(mesh);
    case CurvatureMode::PrincipalMax:
      return computePrincipalMaxCurvatureField(mesh);
  }
  throw std::runtime_error("unknown curvature mode");
}

std::vector<EdgeFieldRecord> computeEdgeFieldRecords(
    const Mesh& mesh,
    const std::vector<double>& target_lengths) {
  if (target_lengths.size() != mesh.vertices.size()) {
    throw std::runtime_error("target diagnostic length size mismatch");
  }

  std::vector<EdgeFieldRecord> records;
  const auto edges = uniqueEdges(mesh);
  records.reserve(edges.size());
  for (const auto& edge : edges) {
    const int a = edge.first;
    const int b = edge.second;
    const double length =
        (mesh.vertices[static_cast<std::size_t>(a)] - mesh.vertices[static_cast<std::size_t>(b)]).norm();
    const double ha = target_lengths[static_cast<std::size_t>(a)];
    const double hb = target_lengths[static_cast<std::size_t>(b)];
    if (ha <= 0.0 || hb <= 0.0 || !std::isfinite(ha) || !std::isfinite(hb)) {
      throw std::runtime_error("target diagnostic length must be positive and finite");
    }
    const double abs_dh = std::abs(ha - hb);
    const double ratio = std::max(ha, hb) / std::min(ha, hb);
    EdgeFieldRecord record;
    record.edge = edge;
    record.edge_length = length;
    record.abs_dh = abs_dh;
    record.target_ratio = ratio;
    record.h_slope = length > kEps ? abs_dh / length : 0.0;
    records.push_back(record);
  }
  return records;
}

double faceTargetGradientNorm(
    const Mesh& mesh,
    const Eigen::Vector3i& face,
    const std::vector<double>& target_lengths) {
  const Eigen::Vector3d& p0 = mesh.vertices[static_cast<std::size_t>(face[0])];
  const Eigen::Vector3d& p1 = mesh.vertices[static_cast<std::size_t>(face[1])];
  const Eigen::Vector3d& p2 = mesh.vertices[static_cast<std::size_t>(face[2])];
  const Eigen::Vector3d e1 = p1 - p0;
  const Eigen::Vector3d e2 = p2 - p0;
  const double g00 = e1.dot(e1);
  const double g01 = e1.dot(e2);
  const double g11 = e2.dot(e2);
  const double det = g00 * g11 - g01 * g01;
  if (std::abs(det) <= kEps || !std::isfinite(det)) {
    return 0.0;
  }
  const double rhs0 =
      target_lengths[static_cast<std::size_t>(face[1])] - target_lengths[static_cast<std::size_t>(face[0])];
  const double rhs1 =
      target_lengths[static_cast<std::size_t>(face[2])] - target_lengths[static_cast<std::size_t>(face[0])];
  const double a = (rhs0 * g11 - rhs1 * g01) / det;
  const double b = (g00 * rhs1 - g01 * rhs0) / det;
  return (a * e1 + b * e2).norm();
}

std::vector<FaceFieldRecord> computeFaceFieldRecords(
    const Mesh& mesh,
    const std::vector<double>& target_lengths,
    const std::unordered_map<Edge, double, EdgeHash>& edge_ratios) {
  std::vector<FaceFieldRecord> records;
  records.reserve(mesh.faces.size());
  for (const auto& face : mesh.faces) {
    double max_ratio = 1.0;
    for (const Edge& edge : {
             makeEdge(face[0], face[1]),
             makeEdge(face[1], face[2]),
             makeEdge(face[2], face[0]),
         }) {
      const auto found = edge_ratios.find(edge);
      if (found != edge_ratios.end()) {
        max_ratio = std::max(max_ratio, found->second);
      }
    }
    records.push_back({faceTargetGradientNorm(mesh, face, target_lengths), max_ratio});
  }
  return records;
}

void writeVerticesCsv(
    const std::filesystem::path& path,
    const Mesh& mesh,
    const TargetFieldStageData& data) {
  auto out = openOutputFile(path);
  out << "vertex_id,x,y,z,raw_curvature,csf_curvature,local_edge_median,local_delta,target_h\n";
  for (std::size_t i = 0; i < mesh.vertices.size(); ++i) {
    const auto& p = mesh.vertices[i];
    out << i << ','
        << p[0] << ',' << p[1] << ',' << p[2] << ','
        << data.raw_curvature[i] << ','
        << data.csf_curvature[i] << ','
        << data.local_edge_medians[i] << ','
        << data.local_deltas[i] << ','
        << data.target_lengths[i] << '\n';
  }
}

void writeEdgesCsv(
    const std::filesystem::path& path,
    const std::vector<EdgeFieldRecord>& records) {
  auto out = openOutputFile(path);
  out << "i,j,edge_length,abs_dh,target_ratio,h_slope\n";
  for (const auto& record : records) {
    out << record.edge.first << ','
        << record.edge.second << ','
        << record.edge_length << ','
        << record.abs_dh << ','
        << record.target_ratio << ','
        << record.h_slope << '\n';
  }
}

void writeFacesCsv(
    const std::filesystem::path& path,
    const std::vector<FaceFieldRecord>& records) {
  auto out = openOutputFile(path);
  out << "face_id,grad_h_norm,max_edge_target_ratio\n";
  for (std::size_t i = 0; i < records.size(); ++i) {
    out << i << ','
        << records[i].grad_h_norm << ','
        << records[i].max_edge_target_ratio << '\n';
  }
}

std::string targetFieldStatsJson(
    const TargetFieldStageData& data,
    const std::vector<EdgeFieldRecord>& edge_records,
    const std::vector<FaceFieldRecord>& face_records) {
  std::vector<double> ratios;
  ratios.reserve(edge_records.size());
  for (const auto& record : edge_records) {
    ratios.push_back(record.target_ratio);
  }
  std::sort(ratios.begin(), ratios.end());

  std::vector<double> gradients;
  gradients.reserve(face_records.size());
  for (const auto& record : face_records) {
    gradients.push_back(record.grad_h_norm);
  }
  std::sort(gradients.begin(), gradients.end());

  std::vector<double> local_deltas;
  local_deltas.reserve(data.local_deltas.size());
  for (double value : data.local_deltas) {
    if (std::isfinite(value)) {
      local_deltas.push_back(value);
    }
  }
  std::sort(local_deltas.begin(), local_deltas.end());

  double observed_min = std::numeric_limits<double>::infinity();
  double observed_max = -std::numeric_limits<double>::infinity();
  for (double value : data.target_lengths) {
    observed_min = std::min(observed_min, value);
    observed_max = std::max(observed_max, value);
  }
  const bool range_ok =
      observed_min + 1e-12 >= data.min_length &&
      observed_max <= data.max_length + 1e-12;
  const double max_ratio = ratios.empty() ? 0.0 : ratios.back();

  std::ostringstream out;
  out << std::setprecision(17);
  out << "{\n";
  out << "  \"base_length\": " << data.base_length << ",\n";
  out << "  \"auto_delta_source\": \"input_resolution_csf_curvature\",\n";
  out << "  \"auto_delta_quantile\": " << data.auto_delta_quantile << ",\n";
  out << "  \"h_min_safe_factor\": " << data.h_min_safe_factor << ",\n";
  out << "  \"target_error_delta\": " << data.target_error_delta << ",\n";
  out << "  \"target_error_bbox_ratio\": " << data.target_error_bbox_ratio << ",\n";
  out << "  \"bbox_diagonal\": " << data.bbox_diagonal << ",\n";
  out << "  \"h_max_factor\": " << data.h_max_factor << ",\n";
  out << "  \"kappa_max\": " << data.kappa_max << ",\n";
  out << "  \"local_delta_min\": " << (local_deltas.empty() ? 0.0 : local_deltas.front()) << ",\n";
  out << "  \"local_delta_q50\": " << quantileOfSorted(local_deltas, 0.5) << ",\n";
  out << "  \"local_delta_q95\": " << quantileOfSorted(local_deltas, 0.95) << ",\n";
  out << "  \"local_delta_max\": " << (local_deltas.empty() ? 0.0 : local_deltas.back()) << ",\n";
  out << "  \"h_min\": " << data.min_length << ",\n";
  out << "  \"h_max\": " << data.max_length << ",\n";
  out << "  \"estimated_faces\": " << data.estimated_faces << ",\n";
  out << "  \"flat_fallback\": " << (data.flat_fallback ? "true" : "false") << ",\n";
  out << "  \"observed_h_min\": " << observed_min << ",\n";
  out << "  \"observed_h_max\": " << observed_max << ",\n";
  out << "  \"range_ok\": " << (range_ok ? "true" : "false") << ",\n";
  out << "  \"edge_count\": " << edge_records.size() << ",\n";
  out << "  \"max_edge_target_ratio\": " << max_ratio << ",\n";
  out << "  \"p95_edge_target_ratio\": " << quantileOfSorted(ratios, 0.95) << ",\n";
  out << "  \"p99_edge_target_ratio\": " << quantileOfSorted(ratios, 0.99) << ",\n";
  out << "  \"face_count\": " << face_records.size() << ",\n";
  out << "  \"max_grad_h_norm\": " << (gradients.empty() ? 0.0 : gradients.back()) << ",\n";
  out << "  \"p95_grad_h_norm\": " << quantileOfSorted(gradients, 0.95) << ",\n";
  out << "  \"p99_grad_h_norm\": " << quantileOfSorted(gradients, 0.99) << "\n";
  out << "}\n";
  return out.str();
}

std::string colorScalesJson(
    double shared_curvature_min,
    double shared_curvature_max,
    double raw_curvature_min,
    double raw_curvature_max,
    double csf_curvature_min,
    double csf_curvature_max,
    double local_delta_min,
    double local_delta_max,
    double target_h_min,
    double target_h_max,
    double observed_target_h_min,
    double observed_target_h_max) {
  std::ostringstream out;
  out << std::setprecision(17);
  out << "{\n";
  out << "  \"palette\": \"heat\",\n";
  out << "  \"normalization\": \"linear_clamp\",\n";
  out << "  \"scales\": {\n";
  out << "    \"raw_curvature_shared\": {\n";
  out << "      \"file\": \"raw_curvature_heat_shared.ply\",\n";
  out << "      \"field\": \"raw_curvature\",\n";
  out << "      \"min\": " << shared_curvature_min << ",\n";
  out << "      \"max\": " << shared_curvature_max << "\n";
  out << "    },\n";
  out << "    \"csf_curvature_shared\": {\n";
  out << "      \"file\": \"csf_curvature_heat_shared.ply\",\n";
  out << "      \"field\": \"csf_curvature\",\n";
  out << "      \"min\": " << shared_curvature_min << ",\n";
  out << "      \"max\": " << shared_curvature_max << "\n";
  out << "    },\n";
  out << "    \"raw_curvature_independent\": {\n";
  out << "      \"file\": \"raw_curvature_heat_independent.ply\",\n";
  out << "      \"field\": \"raw_curvature\",\n";
  out << "      \"min\": " << raw_curvature_min << ",\n";
  out << "      \"max\": " << raw_curvature_max << "\n";
  out << "    },\n";
  out << "    \"csf_curvature_independent\": {\n";
  out << "      \"file\": \"csf_curvature_heat_independent.ply\",\n";
  out << "      \"field\": \"csf_curvature\",\n";
  out << "      \"min\": " << csf_curvature_min << ",\n";
  out << "      \"max\": " << csf_curvature_max << "\n";
  out << "    },\n";
  out << "    \"local_delta\": {\n";
  out << "      \"file\": \"local_delta_heat.ply\",\n";
  out << "      \"field\": \"local_delta\",\n";
  out << "      \"min\": " << local_delta_min << ",\n";
  out << "      \"max\": " << local_delta_max << "\n";
  out << "    },\n";
  out << "    \"target_h\": {\n";
  out << "      \"file\": \"target_h_heat.ply\",\n";
  out << "      \"field\": \"target_h\",\n";
  out << "      \"min\": " << target_h_min << ",\n";
  out << "      \"max\": " << target_h_max << ",\n";
  out << "      \"inverted\": true\n";
  out << "    },\n";
  out << "    \"target_h_observed\": {\n";
  out << "      \"file\": \"target_h_heat_observed.ply\",\n";
  out << "      \"field\": \"target_h\",\n";
  out << "      \"min\": " << observed_target_h_min << ",\n";
  out << "      \"max\": " << observed_target_h_max << ",\n";
  out << "      \"inverted\": true\n";
  out << "    }\n";
  out << "  }\n";
  out << "}\n";
  return out.str();
}

void writeTargetFieldDiagnostics(
    const std::filesystem::path& out_dir,
    const Mesh& mesh,
    const TargetFieldStageData& data) {
  if (out_dir.empty()) {
    return;
  }
  const std::size_t n = mesh.vertices.size();
  if (data.raw_curvature.size() != n || data.csf_curvature.size() != n ||
      data.target_lengths.size() != n ||
      data.local_edge_medians.size() != n ||
      data.local_deltas.size() != n) {
    throw std::runtime_error("target diagnostic field size mismatch");
  }
  std::filesystem::create_directories(out_dir);

  const auto edge_records = computeEdgeFieldRecords(mesh, data.target_lengths);
  std::unordered_map<Edge, double, EdgeHash> edge_ratios;
  edge_ratios.reserve(edge_records.size());
  for (const auto& record : edge_records) {
    edge_ratios.emplace(record.edge, record.target_ratio);
  }
  const auto face_records = computeFaceFieldRecords(mesh, data.target_lengths, edge_ratios);

  writeVerticesCsv(out_dir / "vertices.csv", mesh, data);
  writeEdgesCsv(out_dir / "edges.csv", edge_records);
  writeFacesCsv(out_dir / "faces.csv", face_records);
  writeTextFile(
      out_dir / "field_stats.json",
      targetFieldStatsJson(
          data,
          edge_records,
          face_records));

  const auto [raw_min, raw_max] = scalarRange(data.raw_curvature);
  const auto [csf_min, csf_max] = scalarRange(data.csf_curvature);
  const auto [local_delta_min, local_delta_max] = scalarRange(data.local_deltas);
  const auto [target_h_observed_min, target_h_observed_max] = scalarRange(data.target_lengths);
  const double curvature_min = std::min(raw_min, csf_min);
  const double curvature_max = std::max(raw_max, csf_max);
  writeTextFile(
      out_dir / "color_scales.json",
      colorScalesJson(
          curvature_min,
          curvature_max,
          raw_min,
          raw_max,
          csf_min,
          csf_max,
          local_delta_min,
          local_delta_max,
          data.min_length,
          data.max_length,
          target_h_observed_min,
          target_h_observed_max));
  writeVertexHeatPly(
      out_dir / "raw_curvature_heat_shared.ply",
      mesh,
      data.raw_curvature,
      curvature_min,
      curvature_max);
  writeVertexHeatPly(
      out_dir / "csf_curvature_heat_shared.ply",
      mesh,
      data.csf_curvature,
      curvature_min,
      curvature_max);
  writeVertexHeatPly(
      out_dir / "raw_curvature_heat_independent.ply",
      mesh,
      data.raw_curvature,
      raw_min,
      raw_max);
  writeVertexHeatPly(
      out_dir / "csf_curvature_heat_independent.ply",
      mesh,
      data.csf_curvature,
      csf_min,
      csf_max);
  writeVertexHeatPly(
      out_dir / "local_delta_heat.ply",
      mesh,
      data.local_deltas,
      local_delta_min,
      local_delta_max);
  writeVertexHeatPly(
      out_dir / "target_h_heat.ply",
      mesh,
      data.target_lengths,
      data.min_length,
      data.max_length,
      true);
  writeVertexHeatPly(
      out_dir / "target_h_heat_observed.ply",
      mesh,
      data.target_lengths,
      target_h_observed_min,
      target_h_observed_max,
      true);
}

}  // namespace

namespace detail {

double estimateTargetFaceCount(
    const Mesh& mesh,
    const std::vector<double>& target_lengths) {
  mesh.validate();
  if (mesh.vertices.empty() || mesh.faces.empty()) {
    throw std::runtime_error("target face estimate requires a non-empty mesh");
  }
  if (target_lengths.size() != mesh.vertices.size()) {
    throw std::runtime_error("target face estimate length size mismatch");
  }

  double estimate = 0.0;
  for (const auto& face : mesh.faces) {
    const double area = triangleArea(mesh.vertices[face[0]], mesh.vertices[face[1]], mesh.vertices[face[2]]);
    if (!std::isfinite(area) || area <= 0.0) {
      throw std::runtime_error("target face estimate requires positive finite face areas");
    }
    const double h =
        (target_lengths[static_cast<std::size_t>(face[0])] +
         target_lengths[static_cast<std::size_t>(face[1])] +
         target_lengths[static_cast<std::size_t>(face[2])]) /
        3.0;
    if (!std::isfinite(h) || h <= 0.0) {
      throw std::runtime_error("target face estimate requires positive finite lengths");
    }
    estimate += area / (kEquilateralAreaFactor * h * h);
  }
  if (!std::isfinite(estimate)) {
    throw std::runtime_error("target face estimate produced a non-finite value");
  }
  return estimate;
}

TargetErrorSizingResult targetErrorCurvatureLengths(
    const Mesh& mesh,
    const std::vector<double>& diffused_curvature,
    double mean_edge_length,
    double bbox_diagonal,
    double target_error_delta,
    double target_h_max_factor,
    double auto_delta_quantile) {
  mesh.validate();
  if (mesh.vertices.empty() || mesh.faces.empty()) {
    throw std::runtime_error("target error sizing requires a non-empty mesh");
  }
  if (diffused_curvature.size() != mesh.vertices.size()) {
    throw std::runtime_error("target error sizing curvature size mismatch");
  }
  if (!std::isfinite(mean_edge_length) || mean_edge_length <= 0.0) {
    throw std::runtime_error("target error sizing requires a positive mean edge length");
  }
  if (!std::isfinite(bbox_diagonal) || bbox_diagonal <= 0.0) {
    throw std::runtime_error("target error sizing requires a positive bbox diagonal");
  }
  if (!std::isfinite(target_error_delta) || target_error_delta < 0.0) {
    throw std::runtime_error("target error delta must be finite and non-negative");
  }
  if (!std::isfinite(target_h_max_factor) || target_h_max_factor <= 0.0) {
    throw std::runtime_error("target h max factor must be finite and positive");
  }
  if (!std::isfinite(auto_delta_quantile) || auto_delta_quantile < 0.0 || auto_delta_quantile > 1.0) {
    throw std::runtime_error("auto delta quantile must be finite and in [0, 1]");
  }

  std::vector<double> safe_kappa(diffused_curvature.size(), kTargetKappaMin);
  double observed_max_kappa = 0.0;
  for (std::size_t i = 0; i < diffused_curvature.size(); ++i) {
    const double value = diffused_curvature[i];
    if (std::isfinite(value)) {
      observed_max_kappa = std::max(observed_max_kappa, value);
      safe_kappa[i] = std::max(value, kTargetKappaMin);
    }
  }

  TargetErrorSizingResult result;
  result.h_max = target_h_max_factor * mean_edge_length;
  result.bbox_diagonal = bbox_diagonal;
  result.kappa_max = observed_max_kappa;
  result.auto_delta_quantile = auto_delta_quantile;
  result.h_min_safe_factor = kTargetHMinSafeFactor;
  if (!std::isfinite(result.h_max) || result.h_max <= 0.0) {
    throw std::runtime_error("target error sizing produced an invalid h_max");
  }

  result.local_edge_medians = incidentMedianEdgeLengths(mesh);
  result.local_deltas.reserve(safe_kappa.size());
  for (std::size_t i = 0; i < safe_kappa.size(); ++i) {
    const double local_length = result.local_edge_medians[i];
    const double local_delta = safe_kappa[i] * local_length * local_length / 6.0;
    result.local_deltas.push_back(std::isfinite(local_delta) && local_delta > 0.0 ? local_delta : 0.0);
  }

  if (target_error_delta == 0.0) {
    target_error_delta =
        areaWeightedQuantile(result.local_deltas, barycentricVertexAreas(mesh), auto_delta_quantile);
  }
  if (!std::isfinite(target_error_delta) || target_error_delta <= 0.0) {
    target_error_delta = kTargetKappaMin * mean_edge_length * mean_edge_length / 6.0;
  }
  result.target_error_delta = target_error_delta;
  result.target_error_bbox_ratio = target_error_delta / bbox_diagonal;

  result.h_min = kTargetHMinSafeFactor * mean_edge_length;
  if (!std::isfinite(result.h_min) || result.h_min <= 0.0) {
    throw std::runtime_error("target error sizing produced an invalid h_min");
  }

  if (observed_max_kappa <= kTargetKappaMin) {
    result.flat_fallback = true;
    result.target_lengths.assign(diffused_curvature.size(), result.h_max);
    result.estimated_faces = estimateTargetFaceCount(mesh, result.target_lengths);
    return result;
  }

  if (result.h_min > result.h_max) {
    result.h_min = result.h_max;
  }
  result.target_lengths.reserve(safe_kappa.size());
  for (double kappa : safe_kappa) {
    const double h_squared = 6.0 * target_error_delta / kappa - 3.0 * target_error_delta * target_error_delta;
    double h = result.h_max;
    if (std::isfinite(h_squared) && h_squared > 0.0) {
      h = std::sqrt(h_squared);
    }
    result.target_lengths.push_back(std::clamp(h, result.h_min, result.h_max));
  }
  result.estimated_faces = estimateTargetFaceCount(mesh, result.target_lengths);
  return result;
}

}  // namespace detail

std::vector<Eigen::Vector3i> splitFaceOnEdge(const Eigen::Vector3i& face, int a, int b, int m);
void splitEdgesBatch(
    Mesh& mesh,
    const std::vector<Edge>& edges,
    const TargetLengthField& target_field,
    std::unordered_set<int>* affected_vertices = nullptr);

class ReferenceSurfaceProjector {
 public:
  explicit ReferenceSurfaceProjector(const Mesh& mesh)
      : vertices_(verticesToMatrix(mesh)),
        faces_(facesToMatrix(mesh)) {
    if (mesh.vertices.empty() || mesh.faces.empty()) {
      throw std::runtime_error("reference surface projector requires a non-empty mesh");
    }
    mesh.validate();
    tree_.init(vertices_, faces_);
  }

  [[nodiscard]] bool closestPoint(const Eigen::Vector3d& point, Eigen::Vector3d& closest) const {
    if (!finiteVector(point) || faces_.rows() == 0) {
      return false;
    }
    const Eigen::RowVector3d query = point.transpose();
    Eigen::RowVector3d closest_row = Eigen::RowVector3d::Zero();
    int face_index = -1;
    (void)tree_.squared_distance(vertices_, faces_, query, face_index, closest_row);
    if (face_index < 0 || face_index >= faces_.rows()) {
      return false;
    }
    closest = closest_row.transpose();
    return finiteVector(closest);
  }

 private:
  Eigen::MatrixXd vertices_;
  Eigen::MatrixXi faces_;
  SurfaceAabb tree_;
};

struct TargetLengthFieldImpl {
  Mesh mesh;
  std::vector<double> target_lengths;
  Eigen::MatrixXd vertices;
  Eigen::MatrixXi faces;
  SurfaceAabb tree;
  bool built = false;
};

TargetLengthField::TargetLengthField() : impl_(std::make_unique<TargetLengthFieldImpl>()) {}

TargetLengthField::~TargetLengthField() = default;

TargetLengthField::TargetLengthField(TargetLengthField&& other) noexcept = default;

TargetLengthField& TargetLengthField::operator=(TargetLengthField&& other) noexcept = default;

void TargetLengthField::build(const Mesh& input, const std::vector<double>& target_lengths) {
  if (input.vertices.empty() || input.faces.empty()) {
    throw std::runtime_error("target field requires a non-empty input mesh");
  }
  input.validate();
  if (input.vertices.size() != target_lengths.size()) {
    throw std::runtime_error("target field vertex/length size mismatch");
  }
  for (double value : target_lengths) {
    if (!std::isfinite(value) || value <= 0.0) {
      throw std::runtime_error("target length must be positive and finite");
    }
  }
  impl_ = std::make_unique<TargetLengthFieldImpl>();
  impl_->mesh = input;
  impl_->target_lengths = target_lengths;
  impl_->vertices = verticesToMatrix(input);
  impl_->faces = facesToMatrix(input);
  impl_->tree.init(impl_->vertices, impl_->faces);
  impl_->built = true;
}

void TargetLengthField::clear() {
  impl_ = std::make_unique<TargetLengthFieldImpl>();
}

bool TargetLengthField::empty() const {
  return !impl_ || !impl_->built || impl_->target_lengths.empty() || impl_->faces.rows() == 0;
}

double TargetLengthField::valueAt(const Eigen::Vector3d& point) const {
  if (empty()) {
    throw std::runtime_error("target lengths have not been initialized");
  }
  if (!finiteVector(point)) {
    throw std::runtime_error("target query point is not finite");
  }

  const Eigen::RowVector3d query = point.transpose();
  Eigen::RowVector3d closest = Eigen::RowVector3d::Zero();
  int face_index = -1;
  (void)impl_->tree.squared_distance(impl_->vertices, impl_->faces, query, face_index, closest);
  if (face_index < 0 || face_index >= impl_->faces.rows()) {
    throw std::runtime_error("target field closest-triangle query failed");
  }

  const Eigen::Vector3i face = impl_->mesh.faces[static_cast<std::size_t>(face_index)];
  const Eigen::Vector3d bary = barycentricOnTriangle(
      closest.transpose(),
      impl_->mesh.vertices[static_cast<std::size_t>(face[0])],
      impl_->mesh.vertices[static_cast<std::size_t>(face[1])],
      impl_->mesh.vertices[static_cast<std::size_t>(face[2])]);
  const double value =
      bary[0] * impl_->target_lengths[static_cast<std::size_t>(face[0])] +
      bary[1] * impl_->target_lengths[static_cast<std::size_t>(face[1])] +
      bary[2] * impl_->target_lengths[static_cast<std::size_t>(face[2])];
  if (!std::isfinite(value) || value <= 0.0) {
    throw std::runtime_error("target field query produced an invalid length");
  }
  return value;
}

EdgeRatioStats computeEdgeRatioStats(
    const Mesh& mesh,
    const TargetLengthField& target_field,
    double collapse_factor,
    double split_factor) {
  EdgeRatioStats stats;
  const auto edges = uniqueEdges(mesh);
  if (edges.empty()) {
    return stats;
  }

  std::vector<double> ratios;
  ratios.reserve(edges.size());
  int violations = 0;
  int short_violations = 0;
  int long_violations = 0;
  double target_log_energy_sum = 0.0;
  for (const auto& edge : edges) {
    const int a = edge.first;
    const int b = edge.second;
    const double ratio = edgeTargetMetricLength(mesh, target_field, a, b);
    if (!std::isfinite(ratio) || ratio <= 0.0) {
      continue;
    }
    ratios.push_back(ratio);
    if (ratio < collapse_factor) {
      ++violations;
      ++short_violations;
    } else if (ratio > split_factor) {
      ++violations;
      ++long_violations;
    }
    const double log_ratio = std::log(ratio);
    target_log_energy_sum += log_ratio * log_ratio;
  }

  if (ratios.empty()) {
    return stats;
  }

  const double sum = std::accumulate(ratios.begin(), ratios.end(), 0.0);
  stats.mean_ratio = sum / static_cast<double>(ratios.size());
  double variance = 0.0;
  for (double ratio : ratios) {
    const double diff = ratio - stats.mean_ratio;
    variance += diff * diff;
  }
  stats.std_ratio = std::sqrt(variance / static_cast<double>(ratios.size()));
  std::sort(ratios.begin(), ratios.end());
  stats.p05_ratio = quantileOfSorted(ratios, 0.05);
  stats.p95_ratio = quantileOfSorted(ratios, 0.95);
  stats.violation_rate = static_cast<double>(violations) / static_cast<double>(ratios.size());
  stats.short_violation_rate = static_cast<double>(short_violations) / static_cast<double>(ratios.size());
  stats.long_violation_rate = static_cast<double>(long_violations) / static_cast<double>(ratios.size());
  stats.target_log_energy = target_log_energy_sum / static_cast<double>(ratios.size());
  return stats;
}

void writeOutputTargetDiagnostics(
    const std::filesystem::path& out_dir,
    const Mesh& mesh,
    const TargetLengthField& target_field,
    double target_h_min,
    double target_h_max,
    double collapse_factor,
    double split_factor) {
  if (out_dir.empty() || target_field.empty()) {
    return;
  }
  std::filesystem::create_directories(out_dir);

  std::vector<double> vertex_target_h(mesh.vertexCount(), 0.0);
  for (std::size_t i = 0; i < mesh.vertices.size(); ++i) {
    vertex_target_h[i] = target_field.valueAt(mesh.vertices[i]);
  }

  std::vector<double> vertex_min_ratio(mesh.vertexCount(), std::numeric_limits<double>::infinity());
  std::vector<double> vertex_max_ratio(mesh.vertexCount(), -std::numeric_limits<double>::infinity());
  std::vector<double> ratios;
  const auto edges = uniqueEdges(mesh);
  ratios.reserve(edges.size());
  int short_violations = 0;
  int long_violations = 0;
  double target_log_energy_sum = 0.0;

  auto edge_csv = openOutputFile(out_dir / "output_edge_ratios.csv");
  edge_csv << "i,j,edge_length,target_i,target_j,target_min,target_max,metric_length\n";
  for (const auto& edge : edges) {
    const int a = edge.first;
    const int b = edge.second;
    const auto ai = static_cast<std::size_t>(a);
    const auto bi = static_cast<std::size_t>(b);
    const double edge_length = (mesh.vertices[ai] - mesh.vertices[bi]).norm();
    const double target_i = vertex_target_h[ai];
    const double target_j = vertex_target_h[bi];
    const double target_min = std::min(target_i, target_j);
    const double target_max = std::max(target_i, target_j);
    if (target_min <= kEps || !std::isfinite(edge_length) || !std::isfinite(target_min) ||
        !std::isfinite(target_max)) {
      continue;
    }
    const double ratio = targetMetricLength(edge_length, target_i, target_j);
    if (!std::isfinite(ratio) || ratio <= 0.0) {
      continue;
    }
    edge_csv << a << ',' << b << ','
             << edge_length << ','
             << target_i << ','
             << target_j << ','
             << target_min << ','
             << target_max << ','
             << ratio << '\n';
    ratios.push_back(ratio);
    if (ratio < collapse_factor) {
      ++short_violations;
    } else if (ratio > split_factor) {
      ++long_violations;
    }
    const double log_ratio = std::log(ratio);
    target_log_energy_sum += log_ratio * log_ratio;
    vertex_min_ratio[ai] = std::min(vertex_min_ratio[ai], ratio);
    vertex_min_ratio[bi] = std::min(vertex_min_ratio[bi], ratio);
    vertex_max_ratio[ai] = std::max(vertex_max_ratio[ai], ratio);
    vertex_max_ratio[bi] = std::max(vertex_max_ratio[bi], ratio);
  }

  for (std::size_t i = 0; i < mesh.vertexCount(); ++i) {
    if (!std::isfinite(vertex_min_ratio[i])) {
      vertex_min_ratio[i] = 1.0;
    }
    if (!std::isfinite(vertex_max_ratio[i])) {
      vertex_max_ratio[i] = 1.0;
    }
  }

  writeVertexHeatPly(
      out_dir / "output_target_h_samebar.ply",
      mesh,
      vertex_target_h,
      target_h_min,
      target_h_max,
      true);
  writeVertexHeatPly(
      out_dir / "output_edge_ratio_min_heat.ply",
      mesh,
      vertex_min_ratio,
      collapse_factor,
      split_factor);
  writeVertexHeatPly(
      out_dir / "output_edge_ratio_max_heat.ply",
      mesh,
      vertex_max_ratio,
      collapse_factor,
      split_factor);

  std::sort(ratios.begin(), ratios.end());
  int violations = 0;
  double sum = 0.0;
  for (double ratio : ratios) {
    sum += ratio;
    if (ratio < collapse_factor || ratio > split_factor) {
      ++violations;
    }
  }
  const double mean = ratios.empty() ? 0.0 : sum / static_cast<double>(ratios.size());
  double variance = 0.0;
  for (double ratio : ratios) {
    const double diff = ratio - mean;
    variance += diff * diff;
  }
  const double stddev = ratios.empty() ? 0.0 : std::sqrt(variance / static_cast<double>(ratios.size()));

  std::ostringstream stats;
  stats << std::setprecision(17);
  stats << "{\n";
  stats << "  \"palette\": \"heat\",\n";
  stats << "  \"normalization\": \"linear_clamp\",\n";
  stats << "  \"target_h_samebar\": {\n";
  stats << "    \"file\": \"output_target_h_samebar.ply\",\n";
  stats << "    \"field\": \"target_h_sampled_on_output\",\n";
  stats << "    \"min\": " << target_h_min << ",\n";
  stats << "    \"max\": " << target_h_max << ",\n";
  stats << "    \"inverted\": true\n";
  stats << "  },\n";
  stats << "  \"edge_ratio\": {\n";
  stats << "    \"files\": [\"output_edge_ratio_min_heat.ply\", \"output_edge_ratio_max_heat.ply\"],\n";
  stats << "    \"field\": \"target_metric_edge_length\",\n";
  stats << "    \"min\": " << collapse_factor << ",\n";
  stats << "    \"max\": " << split_factor << ",\n";
  stats << "    \"edge_count\": " << ratios.size() << ",\n";
  stats << "    \"ratio_min\": " << (ratios.empty() ? 0.0 : ratios.front()) << ",\n";
  stats << "    \"ratio_max\": " << (ratios.empty() ? 0.0 : ratios.back()) << ",\n";
  stats << "    \"mean_ratio\": " << mean << ",\n";
  stats << "    \"std_ratio\": " << stddev << ",\n";
  stats << "    \"p05_ratio\": " << quantileOfSorted(ratios, 0.05) << ",\n";
  stats << "    \"p95_ratio\": " << quantileOfSorted(ratios, 0.95) << ",\n";
  stats << "    \"p99_ratio\": " << quantileOfSorted(ratios, 0.99) << ",\n";
  stats << "    \"violation_rate\": "
        << (ratios.empty() ? 0.0 : static_cast<double>(violations) / static_cast<double>(ratios.size())) << ",\n";
  stats << "    \"short_violation_rate\": "
        << (ratios.empty() ? 0.0 : static_cast<double>(short_violations) / static_cast<double>(ratios.size())) << ",\n";
  stats << "    \"long_violation_rate\": "
        << (ratios.empty() ? 0.0 : static_cast<double>(long_violations) / static_cast<double>(ratios.size())) << ",\n";
  stats << "    \"target_log_energy\": "
        << (ratios.empty() ? 0.0 : target_log_energy_sum / static_cast<double>(ratios.size())) << "\n";
  stats << "  }\n";
  stats << "}\n";
  writeTextFile(out_dir / "output_visualization_stats.json", stats.str());
}

double AdaIsoConfig::splitFactor() const {
  if (split_factor_override > 0.0) {
    return split_factor_override;
  }
  return variant == Variant::Paper ? 5.0 / 3.0 : 1.33;
}

double AdaIsoConfig::collapseFactor() const {
  if (collapse_factor_override > 0.0) {
    return collapse_factor_override;
  }
  return 4.0 / 5.0;
}

bool AdaIsoConfig::collapseUsesMaxReference() const {
  return variant == Variant::Paper;
}

Variant parseVariant(const std::string& value) {
  std::string lower = value;
  std::transform(lower.begin(), lower.end(), lower.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  if (lower == "paper") {
    return Variant::Paper;
  }
  if (lower == "code") {
    return Variant::Code;
  }
  throw std::runtime_error("unknown variant: " + value);
}

std::string toString(Variant variant) {
  return variant == Variant::Paper ? "paper" : "code";
}

CurvatureMode parseCurvatureMode(const std::string& value) {
  std::string lower = value;
  std::transform(lower.begin(), lower.end(), lower.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  if (lower == "normal-angle") {
    return CurvatureMode::NormalAngle;
  }
  if (lower == "robust-mean") {
    return CurvatureMode::RobustMean;
  }
  if (lower == "principal-max") {
    return CurvatureMode::PrincipalMax;
  }
  throw std::runtime_error("unknown curvature mode: " + value);
}

std::string toString(CurvatureMode mode) {
  switch (mode) {
    case CurvatureMode::NormalAngle:
      return "normal-angle";
    case CurvatureMode::RobustMean:
      return "robust-mean";
    case CurvatureMode::PrincipalMax:
      return "principal-max";
  }
  return "unknown";
}

StageAPolicy parseStageAPolicy(const std::string& value) {
  std::string lower = value;
  std::transform(lower.begin(), lower.end(), lower.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  if (lower == "split-collapse-no-guard" || lower == "split_collapse_no_guard") {
    return StageAPolicy::SplitCollapseNoGuard;
  }
  if (lower == "split-collapse-weak-guard" || lower == "split_collapse_weak_guard") {
    return StageAPolicy::SplitCollapseWeakGuard;
  }
  if (lower == "collapse-split-weak-guard" || lower == "collapse_split_weak_guard") {
    return StageAPolicy::CollapseSplitWeakGuard;
  }
  throw std::runtime_error("unknown Stage A policy: " + value);
}

std::string toString(StageAPolicy policy) {
  switch (policy) {
    case StageAPolicy::SplitCollapseNoGuard:
      return "split_collapse_no_guard";
    case StageAPolicy::SplitCollapseWeakGuard:
      return "split_collapse_weak_guard";
    case StageAPolicy::CollapseSplitWeakGuard:
      return "collapse_split_weak_guard";
  }
  return "unknown";
}

StageAProjectionMode parseStageAProjectionMode(const std::string& value) {
  std::string lower = value;
  std::transform(lower.begin(), lower.end(), lower.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  if (lower == "final-only" || lower == "final_only") {
    return StageAProjectionMode::FinalOnly;
  }
  if (lower == "per-iteration-local" || lower == "per_iteration_local") {
    return StageAProjectionMode::PerIterationLocal;
  }
  throw std::runtime_error("unknown Stage A projection mode: " + value);
}

std::string toString(StageAProjectionMode mode) {
  switch (mode) {
    case StageAProjectionMode::FinalOnly:
      return "final_only";
    case StageAProjectionMode::PerIterationLocal:
      return "per_iteration_local";
  }
  return "unknown";
}

AdaIsoRemesher::AdaIsoRemesher(AdaIsoConfig config) : config_(config) {}

Mesh AdaIsoRemesher::remesh(const Mesh& input) {
  if (config_.csf_smoothing_alpha <= 0.0 || !std::isfinite(config_.csf_smoothing_alpha)) {
    throw std::runtime_error("CSF smoothing alpha must be positive");
  }
  if (config_.max_flip_dihedral_degrees <= 0.0 || config_.max_flip_dihedral_degrees > 180.0 ||
      !std::isfinite(config_.max_flip_dihedral_degrees)) {
    throw std::runtime_error("max flip dihedral angle must be finite and in (0, 180]");
  }
  if (config_.collapse_feature_omega < 0.0 || !std::isfinite(config_.collapse_feature_omega)) {
    throw std::runtime_error("collapse feature omega must be finite and non-negative");
  }
  if (config_.split_factor_override < 0.0 || !std::isfinite(config_.split_factor_override)) {
    throw std::runtime_error("split factor override must be finite and non-negative");
  }
  if (config_.collapse_factor_override < 0.0 || !std::isfinite(config_.collapse_factor_override)) {
    throw std::runtime_error("collapse factor override must be finite and non-negative");
  }
  if (config_.target_h_max_factor <= 0.0 || !std::isfinite(config_.target_h_max_factor)) {
    throw std::runtime_error("target h max factor must be finite and positive");
  }
  if (!std::isfinite(config_.auto_delta_quantile) ||
      config_.auto_delta_quantile < 0.0 ||
      config_.auto_delta_quantile > 1.0) {
    throw std::runtime_error("auto delta quantile must be finite and in [0, 1]");
  }
  if (!std::isfinite(config_.min_angle_floor_degrees) ||
      config_.min_angle_floor_degrees < 0.0 ||
      config_.min_angle_floor_degrees > 60.0) {
    throw std::runtime_error("min angle floor must be finite and in [0, 60]");
  }
  if (!std::isfinite(config_.split_min_angle_floor_degrees) ||
      config_.split_min_angle_floor_degrees < 0.0 ||
      config_.split_min_angle_floor_degrees > 60.0) {
    throw std::runtime_error("split min angle floor must be finite and in [0, 60]");
  }

  if (config_.target_error_delta < 0.0 || !std::isfinite(config_.target_error_delta)) {
    throw std::runtime_error("target error delta must be finite and non-negative");
  }

  stats_ = {};
  target_min_length_ = 0.0;
  target_max_length_ = 0.0;
  Mesh mesh = removeDegenerateFaces(input);


  Mesh original_for_projection = mesh;

  if (config_.verbose) {
    std::cerr << "input: " << mesh.vertexCount() << " vertices, " << mesh.faceCount() << " faces\n";
  }

  if (config_.preprocess) {
    mesh = subRemesh(mesh);
    mesh = removeDegenerateFaces(mesh);
    if (config_.verbose) {
      std::cerr << "after sub-remesh: " << mesh.vertexCount() << " vertices, " << mesh.faceCount() << " faces\n";
    }
  }

  buildTargetLengths(mesh);
  stats_.iterations.reserve(static_cast<std::size_t>(2 * std::max(0, config_.global_iterations)));
  if (!config_.iteration_mesh_export_dir.empty()) {
    writeMesh(std::filesystem::path(config_.iteration_mesh_export_dir) / "iter000.ply", mesh);
  }

  const ReferenceSurfaceProjector stage_a_projector(original_for_projection);
  runSizeFieldFittingStage(mesh, stage_a_projector);
  if (config_.stage_a_projection) {
    (void)runStageAProjection(mesh, stage_a_projector, nullptr, true);
  }
  recordSizeStageFinalStats(mesh);
  runQualityRepairStage(mesh, original_for_projection);
  mesh = removeDegenerateFaces(mesh);

  stats_.vertices = mesh.vertexCount();
  stats_.faces = mesh.faceCount();
  if (!config_.target_field_export_dir.empty()) {
    writeOutputTargetDiagnostics(
        std::filesystem::path(config_.target_field_export_dir),
        mesh,
        target_field_,
        target_min_length_,
        target_max_length_,
        config_.collapseFactor(),
        config_.splitFactor());
  }
  return mesh;
}

void AdaIsoRemesher::runSizeFieldFittingStage(Mesh& mesh, const ReferenceSurfaceProjector& projector) {
  for (int it = 0; it < config_.global_iterations; ++it) {
    if (config_.verbose) {
      std::cerr << "size stage iteration " << (it + 1) << "/" << config_.global_iterations << "\n";
    }

    int split_count = 0;
    int collapse_count = 0;
    std::unordered_set<int> affected_vertices;
    if (config_.stage_a_policy == StageAPolicy::CollapseSplitWeakGuard) {
      collapse_count = collapseShortEdges(mesh, &affected_vertices);
      mesh = removeDegenerateFaces(mesh);
      writeStageSnapshot(mesh, "size", it + 1, "collapse");
      split_count = splitLongEdges(mesh, &affected_vertices);
      writeStageSnapshot(mesh, "size", it + 1, "split");
    } else {
      split_count = splitLongEdges(mesh, &affected_vertices);
      writeStageSnapshot(mesh, "size", it + 1, "split");
      collapse_count = collapseShortEdges(mesh, &affected_vertices);
      mesh = removeDegenerateFaces(mesh);
      writeStageSnapshot(mesh, "size", it + 1, "collapse");
    }

    if (config_.stage_a_projection &&
        config_.stage_a_projection_mode == StageAProjectionMode::PerIterationLocal &&
        !affected_vertices.empty()) {
      (void)runStageAProjection(mesh, projector, &affected_vertices, false);
      writeStageSnapshot(mesh, "size", it + 1, "projection");
    }

    stats_.size_stage_splits += split_count;
    stats_.size_stage_collapses += collapse_count;
    ++stats_.size_stage_iterations;
    recordIterationStats("size", it + 1, split_count, collapse_count, 0, 0, mesh);
    writeIterationSnapshot(mesh, "size", it + 1);

    if (split_count == 0 && collapse_count == 0) {
      break;
    }
  }
}

void AdaIsoRemesher::recordSizeStageFinalStats(const Mesh& mesh) {
  const MeshSummary size_summary = summarizeMesh(mesh);
  const EdgeRatioStats size_ratio_stats = edgeRatioStats(mesh);
  stats_.size_stage_final_violation = size_ratio_stats.violation_rate;
  stats_.size_stage_final_short_violation = size_ratio_stats.short_violation_rate;
  stats_.size_stage_final_long_violation = size_ratio_stats.long_violation_rate;
  stats_.size_stage_final_target_log_energy = size_ratio_stats.target_log_energy;
  stats_.size_stage_final_min_angle_degrees = size_summary.min_angle_degrees;
  stats_.size_stage_final_angle_lt_10_pct = size_summary.angle_lt_10_pct;
  stats_.size_stage_final_angle_lt_20_pct = size_summary.angle_lt_20_pct;
}

int AdaIsoRemesher::runStageAProjection(
    Mesh& mesh,
    const ReferenceSurfaceProjector& projector,
    const std::unordered_set<int>* candidate_vertices,
    bool final_pass) {
  if (!config_.stage_a_projection) {
    return 0;
  }

  const auto boundary = boundaryVertices(mesh);
  const auto vf = vertexFaces(mesh);

  int accepted = 0;
  double displacement_sum = 0.0;
  double displacement_max = 0.0;
  std::vector<int> vertices_to_visit;
  if (candidate_vertices == nullptr) {
    vertices_to_visit.reserve(mesh.vertices.size());
    for (std::size_t i = 0; i < mesh.vertices.size(); ++i) {
      vertices_to_visit.push_back(static_cast<int>(i));
    }
  } else {
    vertices_to_visit.reserve(candidate_vertices->size());
    for (int vertex : *candidate_vertices) {
      vertices_to_visit.push_back(vertex);
    }
    std::sort(vertices_to_visit.begin(), vertices_to_visit.end());
    vertices_to_visit.erase(std::unique(vertices_to_visit.begin(), vertices_to_visit.end()), vertices_to_visit.end());
  }

  ++stats_.stage_a_projection_passes;
  if (final_pass) {
    ++stats_.stage_a_projection_final_passes;
  } else {
    ++stats_.stage_a_projection_local_passes;
  }

  const std::size_t accepted_before = stats_.stage_a_projection_accepted;
  const double displacement_sum_before =
      stats_.stage_a_projection_mean_displacement * static_cast<double>(accepted_before);

  for (int vertex : vertices_to_visit) {
    if (vertex < 0 || static_cast<std::size_t>(vertex) >= mesh.vertices.size()) {
      continue;
    }
    const std::size_t i = static_cast<std::size_t>(vertex);
    if (boundary.find(vertex) != boundary.end()) {
      continue;
    }
    if (i >= vf.size() || vf[i].empty()) {
      continue;
    }

    Eigen::Vector3d candidate = Eigen::Vector3d::Zero();
    if (!projector.closestPoint(mesh.vertices[i], candidate)) {
      ++stats_.stage_a_projection_rejected_by_invalid;
      continue;
    }
    const double displacement = (candidate - mesh.vertices[i]).norm();
    if (!std::isfinite(displacement) || displacement <= kStageAProjectionMinDistanceEps) {
      continue;
    }

    ++stats_.stage_a_projection_candidates;
    if (final_pass) {
      ++stats_.stage_a_projection_final_candidates;
    } else {
      ++stats_.stage_a_projection_local_candidates;
    }
    bool rejected_by_size_energy = false;
    bool rejected_by_angle = false;
    if (!stageAProjectionAccepted(
            mesh,
            vf,
            static_cast<int>(i),
            candidate,
            rejected_by_size_energy,
            rejected_by_angle)) {
      if (rejected_by_size_energy) {
        ++stats_.stage_a_projection_rejected_by_size_energy;
      } else if (rejected_by_angle) {
        ++stats_.stage_a_projection_rejected_by_angle;
      } else {
        ++stats_.stage_a_projection_rejected_by_invalid;
      }
      continue;
    }

    mesh.vertices[i] = candidate;
    ++accepted;
    ++stats_.stage_a_projection_accepted;
    if (final_pass) {
      ++stats_.stage_a_projection_final_accepted;
    } else {
      ++stats_.stage_a_projection_local_accepted;
    }
    displacement_sum += displacement;
    displacement_max = std::max(displacement_max, displacement);
  }

  const std::size_t accepted_after = stats_.stage_a_projection_accepted;
  stats_.stage_a_projection_mean_displacement =
      accepted_after > 0
          ? (displacement_sum_before + displacement_sum) / static_cast<double>(accepted_after)
          : 0.0;
  stats_.stage_a_projection_max_displacement =
      std::max(stats_.stage_a_projection_max_displacement, displacement_max);

  if (final_pass && !config_.stage_mesh_export_dir.empty()) {
    writeMesh(std::filesystem::path(config_.stage_mesh_export_dir) / "size_after_projection.ply", mesh);
  }
  if (final_pass && !config_.all_stage_mesh_export_dir.empty()) {
    writeMesh(std::filesystem::path(config_.all_stage_mesh_export_dir) / "size_after_projection.ply", mesh);
  }
  return accepted;
}

void AdaIsoRemesher::runQualityRepairStage(Mesh& mesh, const Mesh& original_for_projection) {
  for (int it = 0; it < config_.global_iterations; ++it) {
    if (config_.verbose) {
      std::cerr << "quality stage iteration " << (it + 1) << "/" << config_.global_iterations << "\n";
    }

    const int flip_count = flipEdges(mesh);
    writeStageSnapshot(mesh, "quality", it + 1, "flip");
    const int relocation_count = relocateVertices(mesh, original_for_projection);
    mesh = removeDegenerateFaces(mesh);
    writeStageSnapshot(mesh, "quality", it + 1, "smooth");

    stats_.quality_stage_flips += flip_count;
    stats_.quality_stage_relocations += relocation_count;
    ++stats_.quality_stage_iterations;
    recordIterationStats("quality", it + 1, 0, 0, flip_count, relocation_count, mesh);
    writeIterationSnapshot(mesh, "quality", it + 1);

    if (flip_count == 0 && relocation_count == 0) {
      break;
    }
  }
}

void AdaIsoRemesher::recordIterationStats(
    const std::string& stage,
    int iteration,
    int split_count,
    int collapse_count,
    int flip_count,
    int relocation_count,
    const Mesh& mesh) {
  const MeshSummary iteration_summary = summarizeMesh(mesh);
  const EdgeRatioStats ratio_stats = edgeRatioStats(mesh);
  stats_.iterations.push_back({
      stage,
      iteration,
      split_count,
      collapse_count,
      flip_count,
      relocation_count,
      mesh.vertexCount(),
      mesh.faceCount(),
      iteration_summary.qavg,
      iteration_summary.qmin,
      iteration_summary.min_angle_degrees,
      iteration_summary.angle_lt_10_pct,
      iteration_summary.angle_lt_20_pct,
      ratio_stats.mean_ratio,
      ratio_stats.std_ratio,
      ratio_stats.p05_ratio,
      ratio_stats.p95_ratio,
      ratio_stats.violation_rate,
      ratio_stats.short_violation_rate,
      ratio_stats.long_violation_rate,
      ratio_stats.target_log_energy});
  if (config_.verbose) {
    std::cerr << "  split=" << split_count
              << ", collapse=" << collapse_count
              << ", flip=" << flip_count
              << ", relocate=" << relocation_count
              << ", violation=" << ratio_stats.violation_rate
              << ", size_energy=" << ratio_stats.target_log_energy
              << ", min_angle=" << iteration_summary.min_angle_degrees
              << ", mesh=" << mesh.vertexCount() << "v/" << mesh.faceCount() << "f\n";
  }
}

void AdaIsoRemesher::writeStageSnapshot(
    const Mesh& mesh,
    const std::string& stage,
    int iteration,
    const std::string& operation) const {
  std::ostringstream filename;
  filename << stage << "_iter" << std::setw(3) << std::setfill('0') << iteration
           << "_after_" << operation << ".ply";
  if (iteration == 1 && !config_.stage_mesh_export_dir.empty()) {
    writeMesh(std::filesystem::path(config_.stage_mesh_export_dir) / filename.str(), mesh);
  }
  if (!config_.all_stage_mesh_export_dir.empty()) {
    writeMesh(std::filesystem::path(config_.all_stage_mesh_export_dir) / filename.str(), mesh);
  }
}

void AdaIsoRemesher::writeIterationSnapshot(
    const Mesh& mesh,
    const std::string& stage,
    int iteration) const {
  if (config_.iteration_mesh_export_dir.empty()) {
    return;
  }
  std::ostringstream filename;
  filename << stage << "_iter" << std::setw(3) << std::setfill('0') << iteration << ".ply";
  writeMesh(std::filesystem::path(config_.iteration_mesh_export_dir) / filename.str(), mesh);
}

Mesh AdaIsoRemesher::subRemesh(Mesh mesh) const {
  const double base = meanEdgeLength(mesh);
  if (base <= 0.0) {
    return mesh;
  }
  const double threshold = 2.0 * config_.subremesh_multiplier * base;
  for (int round = 0; round < config_.max_subremesh_rounds; ++round) {
    std::vector<std::tuple<double, int, int>> candidates;
    for (const auto& item : edgeFaces(mesh)) {
      const int a = item.first.first;
      const int b = item.first.second;
      const double length = (mesh.vertices[a] - mesh.vertices[b]).norm();
      if (length > threshold) {
        candidates.emplace_back(length, a, b);
      }
    }
    if (candidates.empty()) {
      break;
    }
    std::sort(candidates.begin(), candidates.end(), [](const auto& lhs, const auto& rhs) {
      return std::get<0>(lhs) > std::get<0>(rhs);
    });
    splitOneEdge(mesh, std::get<1>(candidates.front()), std::get<2>(candidates.front()));
    mesh = removeDegenerateFaces(mesh);
  }
  return mesh;
}

void AdaIsoRemesher::buildTargetLengths(const Mesh& mesh) {
  const double base_len = meanEdgeLength(mesh);
  const double bbox_diag = boundingBoxDiagonal(mesh);
  const auto raw_curvature = computeRawCurvatureField(mesh, config_.curvature_mode);
  const auto curvature = solveCsf(mesh, raw_curvature);
  const auto sizing = detail::targetErrorCurvatureLengths(
      mesh,
      curvature,
      base_len,
      bbox_diag,
      config_.target_error_delta,
      config_.target_h_max_factor,
      config_.auto_delta_quantile);

  if (!config_.target_field_export_dir.empty()) {
    writeTargetFieldDiagnostics(
        std::filesystem::path(config_.target_field_export_dir),
        mesh,
        {raw_curvature,
         curvature,
         sizing.target_lengths,
         sizing.local_edge_medians,
         sizing.local_deltas,
         base_len,
         sizing.h_min,
         sizing.h_max,
         sizing.target_error_delta,
         sizing.target_error_bbox_ratio,
         sizing.bbox_diagonal,
         config_.target_h_max_factor,
         config_.auto_delta_quantile,
         sizing.h_min_safe_factor,
         sizing.kappa_max,
         sizing.estimated_faces,
         sizing.flat_fallback});
  }

  target_field_.build(mesh, sizing.target_lengths);
  target_min_length_ = sizing.h_min;
  target_max_length_ = sizing.h_max;
  stats_.target_error_delta = sizing.target_error_delta;
  stats_.target_error_bbox_ratio = sizing.target_error_bbox_ratio;
  stats_.bbox_diagonal = sizing.bbox_diagonal;
  stats_.target_h_min = sizing.h_min;
  stats_.target_h_max = sizing.h_max;
  stats_.target_estimated_faces = sizing.estimated_faces;
  stats_.target_kappa_max = sizing.kappa_max;
  stats_.target_auto_delta_quantile = sizing.auto_delta_quantile;
  stats_.target_h_min_safe_factor = sizing.h_min_safe_factor;
  stats_.target_flat_fallback = sizing.flat_fallback;

  if (config_.verbose && !sizing.target_lengths.empty()) {
    const auto [min_it, max_it] = std::minmax_element(sizing.target_lengths.begin(), sizing.target_lengths.end());
    std::cerr << "target length: base=" << base_len
              << ", delta=" << sizing.target_error_delta
              << ", delta/bbox=" << sizing.target_error_bbox_ratio
              << ", estimated_faces=" << sizing.estimated_faces
              << ", auto_delta_quantile=" << sizing.auto_delta_quantile
              << ", kappa_max=" << sizing.kappa_max
              << ", min=" << *min_it
              << ", max=" << *max_it << "\n";
  }
}

std::vector<double> AdaIsoRemesher::computeCsf(const Mesh& mesh) {
  return solveCsf(mesh, computeRawCurvatureField(mesh, config_.curvature_mode));
}

std::vector<double> AdaIsoRemesher::solveCsf(
    const Mesh& mesh,
    const std::vector<double>& raw_curvature) {
  const auto solved = detail::solveScreenedCsf(mesh, raw_curvature, config_.csf_smoothing_alpha);
  stats_.csf_assembly_time_ms = solved.stats.assembly_time_ms;
  stats_.csf_factorization_time_ms = solved.stats.factorization_time_ms;
  stats_.csf_solve_time_ms = solved.stats.solve_time_ms;
  stats_.csf_residual_abs = solved.stats.residual_abs;
  stats_.csf_residual_scaled = solved.stats.residual_scaled;
  stats_.csf_matrix_nonzeros = solved.stats.matrix_nonzeros;
  stats_.csf_solver_success = solved.stats.solver_success;
  return solved.values;
}

double AdaIsoRemesher::targetAt(const Eigen::Vector3d& point) const {
  return target_field_.valueAt(point);
}

EdgeRatioStats AdaIsoRemesher::edgeRatioStats(const Mesh& mesh) const {
  return computeEdgeRatioStats(
      mesh,
      target_field_,
      config_.collapseFactor(),
      config_.splitFactor());
}

int AdaIsoRemesher::splitLongEdges(Mesh& mesh, std::unordered_set<int>* affected_vertices) {
  int total = 0;
  for (int pass = 0; pass < config_.max_split_passes; ++pass) {
    // Collect candidate edges that are longer than the split threshold
    std::vector<std::tuple<double, int, int>> candidates;
    const auto e2f = edgeFaces(mesh);
    const auto neigh = vertexNeighbors(mesh);
    for (const auto& item : e2f) {
      const int a = item.first.first;
      const int b = item.first.second;
      const double metric_length = edgeTargetMetricLength(mesh, target_field_, a, b);
      if (metric_length > config_.splitFactor()) {
        candidates.emplace_back(metric_length, a, b);
      }
    }
    stats_.size_stage_long_edge_candidates += candidates.size();
    if (candidates.empty()) {
      break;
    }
    std::sort(candidates.begin(), candidates.end(), [](const auto& lhs, const auto& rhs) {
      return std::get<0>(lhs) > std::get<0>(rhs);
    });


    std::vector<Edge> selected;
    selected.reserve(candidates.size());
    std::unordered_set<int> used_vertices;
    for (const auto& [old_metric_length, a, b] : candidates) {
      (void)old_metric_length;
      if (used_vertices.find(a) != used_vertices.end() || used_vertices.find(b) != used_vertices.end()) {
        continue;
      }
      const double metric_length = edgeTargetMetricLength(mesh, target_field_, a, b);
      if (!std::isfinite(metric_length) || metric_length <= config_.splitFactor()) {
        continue;
      }
      if (config_.stage_a_policy != StageAPolicy::SplitCollapseNoGuard &&
          !splitAcceptedByAngleGuard(mesh, e2f, a, b)) {
        ++stats_.size_stage_split_rejected_by_angle;
        continue;
      }
      selected.push_back(makeEdge(a, b));
      used_vertices.insert(a);
      used_vertices.insert(b);
    }
    if (selected.empty()) {
      break;
    }
    if (affected_vertices != nullptr) {
      for (const auto& edge : selected) {
        affected_vertices->insert(edge.first);
        affected_vertices->insert(edge.second);
        if (static_cast<std::size_t>(edge.first) < neigh.size()) {
          affected_vertices->insert(
              neigh[static_cast<std::size_t>(edge.first)].begin(),
              neigh[static_cast<std::size_t>(edge.first)].end());
        }
        if (static_cast<std::size_t>(edge.second) < neigh.size()) {
          affected_vertices->insert(
              neigh[static_cast<std::size_t>(edge.second)].begin(),
              neigh[static_cast<std::size_t>(edge.second)].end());
        }
      }
    }
    splitEdgesBatch(mesh, selected, target_field_, affected_vertices);
    stats_.size_stage_split_accepted += selected.size();
    total += static_cast<int>(selected.size());
  }
  return total;
}

bool AdaIsoRemesher::splitAcceptedByAngleGuard(
    const Mesh& mesh,
    const EdgeFaceMap& e2f,
    int a,
    int b) const {
  if (a < 0 || b < 0 ||
      static_cast<std::size_t>(a) >= mesh.vertices.size() ||
      static_cast<std::size_t>(b) >= mesh.vertices.size()) {
    return false;
  }
  const auto incident_it = e2f.find(makeEdge(a, b));
  if (incident_it == e2f.end() || incident_it->second.empty()) {
    return false;
  }

  std::vector<Eigen::Vector3i> before_faces;
  before_faces.reserve(incident_it->second.size());
  for (int fi : incident_it->second) {
    if (fi < 0 || static_cast<std::size_t>(fi) >= mesh.faces.size()) {
      return false;
    }
    const auto& face = mesh.faces[static_cast<std::size_t>(fi)];
    if (!faceContainsEdge(face, a, b)) {
      return false;
    }
    before_faces.push_back(face);
  }

  const Eigen::Vector3d& pa = mesh.vertices[static_cast<std::size_t>(a)];
  const Eigen::Vector3d& pb = mesh.vertices[static_cast<std::size_t>(b)];
  const double t = targetMetricSplitParameter(target_field_.valueAt(pa), target_field_.valueAt(pb));
  const Eigen::Vector3d split_point = (1.0 - t) * pa + t * pb;
  if (!finiteVector(split_point)) {
    return false;
  }

  const int midpoint_idx = static_cast<int>(mesh.vertices.size());
  std::vector<Eigen::Vector3i> after_faces;
  after_faces.reserve(before_faces.size() * 2);
  for (const auto& face : before_faces) {
    auto split = splitFaceOnEdge(face, a, b, midpoint_idx);
    after_faces.insert(after_faces.end(), split.begin(), split.end());
  }

  const AnglePatchMetrics before = evaluateAnglePatch(mesh, before_faces);
  const AnglePatchMetrics after =
      evaluateAnglePatch(mesh, after_faces, {{midpoint_idx, split_point}});
  if (!before.valid || !after.valid) {
    return false;
  }
  (void)before;
  return after.min_angle_degrees + kMinAngleTolDegrees >= config_.split_min_angle_floor_degrees;
}

std::vector<Eigen::Vector3i> splitFaceOnEdge(const Eigen::Vector3i& face, int a, int b, int m) {
  for (int offset = 0; offset < 3; ++offset) {
    const int x = face[offset];
    const int y = face[(offset + 1) % 3];
    const int z = face[(offset + 2) % 3];
    if (x == a && y == b) {
      return {Eigen::Vector3i(a, m, z), Eigen::Vector3i(m, b, z)};
    }
    if (x == b && y == a) {
      return {Eigen::Vector3i(b, m, z), Eigen::Vector3i(m, a, z)};
    }
  }
  throw std::runtime_error("internal error: face does not contain requested edge");
}

void splitEdgesBatch(
    Mesh& mesh,
    const std::vector<Edge>& edges,
    const TargetLengthField& target_field,
    std::unordered_set<int>* affected_vertices) {
  if (edges.empty()) {
    return;
  }

  std::unordered_map<Edge, int, EdgeHash> midpoint_indices;
  midpoint_indices.reserve(edges.size());
  for (const auto& edge : edges) {
    const int a = edge.first;
    const int b = edge.second;
    const int m = static_cast<int>(mesh.vertices.size());
    const Eigen::Vector3d& pa = mesh.vertices[static_cast<std::size_t>(a)];
    const Eigen::Vector3d& pb = mesh.vertices[static_cast<std::size_t>(b)];
    const double t = targetMetricSplitParameter(target_field.valueAt(pa), target_field.valueAt(pb));
    mesh.vertices.push_back((1.0 - t) * pa + t * pb);
    midpoint_indices.emplace(edge, m);
    if (affected_vertices != nullptr) {
      affected_vertices->insert(m);
    }
  }

  std::vector<Eigen::Vector3i> new_faces;
  new_faces.reserve(mesh.faces.size() + edges.size());
  for (const auto& face : mesh.faces) {
    bool did_split = false;
    for (int i = 0; i < 3; ++i) {
      const int a = face[i];
      const int b = face[(i + 1) % 3];
      const Edge edge = makeEdge(a, b);
      const auto it = midpoint_indices.find(edge);
      if (it != midpoint_indices.end()) {
        auto split = splitFaceOnEdge(face, a, b, it->second);
        new_faces.insert(new_faces.end(), split.begin(), split.end());
        did_split = true;
        break;
      }
    }
    if (!did_split) {
      new_faces.push_back(face);
    }
  }
  mesh.faces = std::move(new_faces);
}

void AdaIsoRemesher::splitOneEdge(Mesh& mesh, int a, int b) const {
  const int midpoint_idx = static_cast<int>(mesh.vertices.size());
  mesh.vertices.push_back(0.5 * (mesh.vertices[static_cast<std::size_t>(a)] + mesh.vertices[static_cast<std::size_t>(b)]));

  std::vector<Eigen::Vector3i> new_faces;
  new_faces.reserve(mesh.faces.size() + 2);
  for (const auto& face : mesh.faces) {
    if (faceContains(face, a) && faceContains(face, b)) {
      auto split = splitFaceOnEdge(face, a, b, midpoint_idx);
      new_faces.insert(new_faces.end(), split.begin(), split.end());
    } else {
      new_faces.push_back(face);
    }
  }
  mesh.faces = std::move(new_faces);
}

int AdaIsoRemesher::collapseShortEdges(Mesh& mesh, std::unordered_set<int>* affected_vertices) {
  struct CollapseCandidate {
    double ratio = 0.0;
    double length = 0.0;
    int a = 0;
    int b = 0;
  };

  int total = 0;
  auto record_reject = [this](CollapseRejectReason reason, bool retry_round) {
    ++stats_.size_stage_collapse_rejected;
    if (retry_round) {
      ++stats_.size_stage_collapse_conflict_retry_rejected;
    }
    switch (reason) {
      case CollapseRejectReason::None:
        break;
      case CollapseRejectReason::NonManifold:
        ++stats_.size_stage_collapse_rejected_by_non_manifold;
        break;
      case CollapseRejectReason::LinkCondition:
        ++stats_.size_stage_collapse_rejected_by_link_condition;
        break;
      case CollapseRejectReason::DegeneratePatch:
        ++stats_.size_stage_collapse_rejected_by_degenerate_patch;
        break;
      case CollapseRejectReason::TargetLength:
        ++stats_.size_stage_collapse_rejected_by_target_length;
        break;
      case CollapseRejectReason::Angle:
        ++stats_.size_stage_collapse_rejected_by_angle;
        break;
    }
  };
  auto record_boundary_reject = [this](bool retry_round) {
    ++stats_.size_stage_collapse_rejected;
    ++stats_.size_stage_collapse_rejected_by_boundary;
    if (retry_round) {
      ++stats_.size_stage_collapse_conflict_retry_rejected;
    }
  };
  auto record_conflict_reject = [this](bool retry_round) {
    ++stats_.size_stage_collapse_rejected;
    ++stats_.size_stage_collapse_rejected_by_batch_conflict;
    if (retry_round) {
      ++stats_.size_stage_collapse_conflict_retry_rejected;
    }
  };
  auto record_duplicate_rejects = [this](std::size_t count, bool retry_round) {
    stats_.size_stage_collapse_rejected += count;
    stats_.size_stage_collapse_rejected_by_batch_duplicate += count;
    if (retry_round) {
      stats_.size_stage_collapse_conflict_retry_rejected += count;
    }
  };

  for (int pass = 0; pass < config_.max_collapse_passes; ++pass) {
    std::unordered_set<int> conflict_seed_vertices;
    bool pass_had_candidates = false;
    bool pass_accepted = false;

    for (int mini_round = 0; mini_round < kCollapseConflictMiniRounds; ++mini_round) {
      const bool retry_round = mini_round > 0;
      if (retry_round && conflict_seed_vertices.empty()) {
        break;
      }
      if (retry_round) {
        ++stats_.size_stage_collapse_conflict_mini_rounds;
      }

      const auto boundary = boundaryVertices(mesh);
      const auto e2f = edgeFaces(mesh);
      const auto neigh = vertexNeighbors(mesh);
      const auto vf = vertexFaces(mesh);
      const auto feature_strengths = computeVertexFeatureStrengths(mesh);
      std::vector<CollapseCandidate> candidates;
      for (const auto& item : e2f) {
        const int a = item.first.first;
        const int b = item.first.second;
        if (retry_round &&
            conflict_seed_vertices.find(a) == conflict_seed_vertices.end() &&
            conflict_seed_vertices.find(b) == conflict_seed_vertices.end()) {
          continue;
        }
        const double length = (mesh.vertices[a] - mesh.vertices[b]).norm();
        const double metric_length = edgeTargetMetricLength(mesh, target_field_, a, b);
        if (metric_length < config_.collapseFactor()) {
          candidates.push_back({metric_length, length, a, b});
        }
      }
      stats_.size_stage_short_edge_candidates += candidates.size();
      if (retry_round) {
        stats_.size_stage_collapse_conflict_retry_candidates += candidates.size();
      }
      if (candidates.empty()) {
        if (!retry_round) {
          pass_had_candidates = false;
        }
        break;
      }
      pass_had_candidates = true;
      std::sort(candidates.begin(), candidates.end(), [](const auto& lhs, const auto& rhs) {
        if (lhs.ratio != rhs.ratio) {
          return lhs.ratio < rhs.ratio;
        }
        return lhs.length < rhs.length;
      });

      std::vector<Edge> selected;
      selected.reserve(candidates.size());
      std::unordered_set<int> blocked_vertices;
      std::unordered_set<int> next_conflict_seed_vertices;
      for (const auto& candidate : candidates) {
        const int a = candidate.a;
        const int b = candidate.b;
        if (boundary.find(a) != boundary.end() || boundary.find(b) != boundary.end()) {
          record_boundary_reject(retry_round);
          continue;
        }
        auto it = e2f.find(makeEdge(a, b));
        if (it != e2f.end() && it->second.size() > 2) {
          record_reject(CollapseRejectReason::NonManifold, retry_round);
          continue;
        }
        std::unordered_set<int> footprint;
        footprint.insert(a);
        footprint.insert(b);
        footprint.insert(neigh[static_cast<std::size_t>(a)].begin(), neigh[static_cast<std::size_t>(a)].end());
        footprint.insert(neigh[static_cast<std::size_t>(b)].begin(), neigh[static_cast<std::size_t>(b)].end());

        bool conflicts = false;
        for (int v : footprint) {
          if (blocked_vertices.find(v) != blocked_vertices.end()) {
            conflicts = true;
            break;
          }
        }
        if (conflicts) {
          record_conflict_reject(retry_round);
          next_conflict_seed_vertices.insert(footprint.begin(), footprint.end());
          continue;
        }
        const CollapseRejectReason reason =
            collapseRejectReason(mesh, a, b, neigh, e2f, vf, feature_strengths);
        if (reason != CollapseRejectReason::None) {
          record_reject(reason, retry_round);
          continue;
        }
        selected.push_back(makeEdge(a, b));
        blocked_vertices.insert(footprint.begin(), footprint.end());
      }

      conflict_seed_vertices = std::move(next_conflict_seed_vertices);
      if (selected.empty()) {
        continue;
      }
      if (affected_vertices != nullptr) {
        for (const auto& edge : selected) {
          const int a = edge.first;
          const int b = edge.second;
          affected_vertices->insert(a);
          affected_vertices->insert(b);
          if (static_cast<std::size_t>(a) < neigh.size()) {
            affected_vertices->insert(
                neigh[static_cast<std::size_t>(a)].begin(),
                neigh[static_cast<std::size_t>(a)].end());
          }
          if (static_cast<std::size_t>(b) < neigh.size()) {
            affected_vertices->insert(
                neigh[static_cast<std::size_t>(b)].begin(),
                neigh[static_cast<std::size_t>(b)].end());
          }
        }
      }
      const int accepted = collapseEdgesBatch(mesh, selected, feature_strengths, affected_vertices);
      stats_.size_stage_collapse_accepted += static_cast<std::size_t>(accepted);
      if (retry_round) {
        stats_.size_stage_collapse_conflict_retry_accepted += static_cast<std::size_t>(accepted);
      }
      if (selected.size() > static_cast<std::size_t>(accepted)) {
        record_duplicate_rejects(selected.size() - static_cast<std::size_t>(accepted), retry_round);
      }
      total += accepted;
      pass_accepted = pass_accepted || accepted > 0;
    }

    if (!pass_had_candidates || !pass_accepted) {
      break;
    }
  }
  return total;
}

AdaIsoRemesher::CollapseRejectReason AdaIsoRemesher::collapseRejectReason(
    const Mesh& mesh,
    int a,
    int b,
    const std::vector<std::unordered_set<int>>& neigh,
    const EdgeFaceMap& e2f,
    const std::vector<std::vector<int>>& vf,
    const std::vector<double>& feature_strengths) const {
  const Eigen::Vector3d candidate = collapsePoint(mesh, a, b, feature_strengths);
  if (!candidate.allFinite()) {
    return CollapseRejectReason::TargetLength;
  }
  const auto incident_it = e2f.find(makeEdge(a, b));
  if (incident_it == e2f.end()) {
    return CollapseRejectReason::NonManifold;
  }

  std::unordered_set<int> opposite;
  for (int fi : incident_it->second) {
    const auto& face = mesh.faces[static_cast<std::size_t>(fi)];
    for (int k = 0; k < 3; ++k) {
      const int v = face[k];
      if (v != a && v != b) {
        opposite.insert(v);
      }
    }
  }

  std::unordered_set<int> link_intersection;
  for (int n : neigh[static_cast<std::size_t>(a)]) {
    if (n != a && n != b && neigh[static_cast<std::size_t>(b)].find(n) != neigh[static_cast<std::size_t>(b)].end()) {
      link_intersection.insert(n);
    }
  }
  if (!unorderedSetsEqual(link_intersection, opposite)) {
    return CollapseRejectReason::LinkCondition;
  }

  std::unordered_set<int> related = neigh[static_cast<std::size_t>(a)];
  related.insert(neigh[static_cast<std::size_t>(b)].begin(), neigh[static_cast<std::size_t>(b)].end());
  related.erase(a);
  related.erase(b);
  if (related.size() < 3) {
    return CollapseRejectReason::DegeneratePatch;
  }
  {
    const int keep = std::min(a, b);
    const int remove = std::max(a, b);

    std::unordered_set<int> affected_ids;
    affected_ids.insert(vf[static_cast<std::size_t>(a)].begin(), vf[static_cast<std::size_t>(a)].end());
    affected_ids.insert(vf[static_cast<std::size_t>(b)].begin(), vf[static_cast<std::size_t>(b)].end());

    std::vector<Eigen::Vector3i> affected_faces;
    affected_faces.reserve(affected_ids.size());
    for (int fi : affected_ids) {
      affected_faces.push_back(mesh.faces[static_cast<std::size_t>(fi)]);
    }

    const LocalPatchMetrics before = evaluateLocalPatch(
        mesh,
        target_field_,
        affected_faces,
        config_.collapseFactor(),
        config_.splitFactor());

    const LocalPatchMetrics after = evaluateLocalPatch(
        mesh,
        target_field_,
        affected_faces,
        {{remove, keep}},
        {{keep, candidate}},
        config_.collapseFactor(),
        config_.splitFactor(),
        true);

    if (!before.valid || !after.valid ||
        !std::isfinite(before.min_angle_degrees) ||
        !std::isfinite(after.min_angle_degrees) ||
        !std::isfinite(before.target_metric_violation) ||
        !std::isfinite(after.target_metric_violation) ||
        !std::isfinite(before.target_log_energy) ||
        !std::isfinite(after.target_log_energy)) {
      return CollapseRejectReason::DegeneratePatch;
    }
    if (after.target_log_energy >= before.target_log_energy - kTargetLogEnergyTol) {
      return CollapseRejectReason::TargetLength;
    }
    if (!minAngleSafetyAccepted(
            before.min_angle_degrees,
            after.min_angle_degrees,
            config_.min_angle_floor_degrees)) {
      return CollapseRejectReason::Angle;
    }
  }

  return CollapseRejectReason::None;
}

Eigen::Vector3d AdaIsoRemesher::collapsePoint(
    const Mesh& mesh,
    int a,
    int b,
    const std::vector<double>& feature_strengths) const {
  const Eigen::Vector3d midpoint =
      0.5 * (mesh.vertices[static_cast<std::size_t>(a)] + mesh.vertices[static_cast<std::size_t>(b)]);
  if (!config_.feature_aware_collapse || feature_strengths.size() != mesh.vertices.size()) {
    return midpoint;
  }

  const double fa = feature_strengths[static_cast<std::size_t>(a)];
  const double fb = feature_strengths[static_cast<std::size_t>(b)];
  if (!std::isfinite(fa) || !std::isfinite(fb)) {
    return midpoint;
  }

  const double fmax = std::max(fa, fb);
  constexpr double kFeatureEps = 1e-12;
  if (fmax <= kFeatureEps || std::abs(fa - fb) < config_.collapse_feature_omega * fmax) {
    return midpoint;
  }
  return fa > fb ? mesh.vertices[static_cast<std::size_t>(a)] : mesh.vertices[static_cast<std::size_t>(b)];
}

int AdaIsoRemesher::collapseEdgesBatch(
    Mesh& mesh,
    const std::vector<Edge>& selected,
    const std::vector<double>& feature_strengths,
    std::unordered_set<int>* affected_vertices) const {
  struct CollapseOp {
    int keep = 0;
    int remove = 0;
    Eigen::Vector3d point = Eigen::Vector3d::Zero();
  };

  std::vector<CollapseOp> ops;
  ops.reserve(selected.size());
  std::unordered_map<int, int> replacements;
  replacements.reserve(selected.size());

  for (const auto& edge : selected) {
    const int a = edge.first;
    const int b = edge.second;
    const int keep = std::min(a, b);
    const int remove = std::max(a, b);
    if (replacements.find(remove) != replacements.end()) {
      continue;
    }
    replacements.emplace(remove, keep);
    ops.push_back({keep, remove, collapsePoint(mesh, a, b, feature_strengths)});
    if (affected_vertices != nullptr) {
      affected_vertices->insert(keep);
    }
  }

  if (ops.empty()) {
    return 0;
  }

  for (const auto& op : ops) {
    mesh.vertices[static_cast<std::size_t>(op.keep)] = op.point;
  }
  for (auto& face : mesh.faces) {
    for (int k = 0; k < 3; ++k) {
      const auto it = replacements.find(face[k]);
      if (it != replacements.end()) {
        face[k] = it->second;
      }
    }
  }
  mesh = removeDegenerateFaces(mesh);
  return static_cast<int>(ops.size());
}

int AdaIsoRemesher::flipEdges(Mesh& mesh) {
  int total = 0;
  for (int pass = 0; pass < config_.max_flip_passes; ++pass) {
    const auto e2f = edgeFaces(mesh);
    std::unordered_set<Edge, EdgeHash> current_edges;
    current_edges.reserve(e2f.size());
    for (const auto& item : e2f) {
      current_edges.insert(item.first);
    }
    const auto boundary = boundaryVertices(mesh);
    const auto neigh = vertexNeighbors(mesh);
    std::vector<int> valence(neigh.size(), 0);
    for (std::size_t i = 0; i < neigh.size(); ++i) {
      valence[i] = static_cast<int>(neigh[i].size());
    }

    int flipped = 0;
    for (const auto& item : e2f) {
      const int a = item.first.first;
      const int b = item.first.second;
      const auto& incident = item.second;
      if (incident.size() != 2) {
        continue;
      }
      const int f1 = incident[0];
      const int f2 = incident[1];
      if (f1 < 0 || f2 < 0 || static_cast<std::size_t>(f1) >= mesh.faces.size() ||
          static_cast<std::size_t>(f2) >= mesh.faces.size()) {
        continue;
      }
      const Eigen::Vector3i tri1 = mesh.faces[static_cast<std::size_t>(f1)];
      const Eigen::Vector3i tri2 = mesh.faces[static_cast<std::size_t>(f2)];
      if (!faceContainsEdge(tri1, a, b) || !faceContainsEdge(tri2, a, b)) {
        continue;
      }

      const double max_flip_dihedral = config_.max_flip_dihedral_degrees * kPi / 180.0;
      if (faceNormalAngle(mesh.vertices, tri1, tri2) > max_flip_dihedral) {
        continue;
      }

      int c = -1;
      int d = -1;
      for (int k = 0; k < 3; ++k) {
        if (tri1[k] != a && tri1[k] != b) {
          c = tri1[k];
        }
        if (tri2[k] != a && tri2[k] != b) {
          d = tri2[k];
        }
      }
      if (c < 0 || d < 0 || c == d || current_edges.find(makeEdge(c, d)) != current_edges.end()) {
        continue;
      }

      const double metric_length = edgeTargetMetricLength(mesh, target_field_, c, d);
      if (!std::isfinite(metric_length) ||
          metric_length < config_.collapseFactor() ||
          metric_length > config_.splitFactor()) {
        continue;
      }
      const Eigen::Vector3d ref_normal = safeFaceNormal(mesh.vertices, tri1) + safeFaceNormal(mesh.vertices, tri2);
      Eigen::Vector3i new1 = orientTriangle(Eigen::Vector3i(c, d, b), mesh.vertices, ref_normal);
      Eigen::Vector3i new2 = orientTriangle(Eigen::Vector3i(d, c, a), mesh.vertices, ref_normal);
      if (faceNormalAngle(mesh.vertices, new1, new2) > max_flip_dihedral) {
        continue;
      }
      if (config_.avoid_obtuse_flip) {
        if (maxInnerAngle(mesh.vertices[new1[0]], mesh.vertices[new1[1]], mesh.vertices[new1[2]]) > 0.5 * kPi ||
            maxInnerAngle(mesh.vertices[new2[0]], mesh.vertices[new2[1]], mesh.vertices[new2[2]]) > 0.5 * kPi) {
          continue;
        }
      }
      if (!flipAcceptedByAngleMetric(mesh, valence, boundary, tri1, tri2, new1, new2, a, b, c, d)) {
        continue;
      }

      mesh.faces[static_cast<std::size_t>(f1)] = new1;
      mesh.faces[static_cast<std::size_t>(f2)] = new2;
      current_edges.erase(makeEdge(a, b));
      current_edges.insert(makeEdge(c, d));
      --valence[static_cast<std::size_t>(a)];
      --valence[static_cast<std::size_t>(b)];
      ++valence[static_cast<std::size_t>(c)];
      ++valence[static_cast<std::size_t>(d)];
      ++flipped;
      ++total;
    }
    if (flipped == 0) {
      break;
    }
  }
  return total;
}

bool AdaIsoRemesher::flipAcceptedByAngleMetric(
    const Mesh& mesh,
    const std::vector<int>& valence,
    const std::unordered_set<int>& boundary,
    const Eigen::Vector3i& old1,
    const Eigen::Vector3i& old2,
    const Eigen::Vector3i& new1,
    const Eigen::Vector3i& new2,
    int a,
    int b,
    int c,
    int d) const {
  const LocalPatchMetrics before = evaluateLocalPatch(
      mesh,
      target_field_,
      {old1, old2},
      config_.collapseFactor(),
      config_.splitFactor());
  const LocalPatchMetrics after = evaluateLocalPatch(
      mesh,
      target_field_,
      {new1, new2},
      config_.collapseFactor(),
      config_.splitFactor());
  if (!before.valid || !after.valid) {
    return false;
  }
  if (!minAngleQualityAccepted(
          before.min_angle_degrees,
          after.min_angle_degrees,
          config_.min_angle_floor_degrees)) {
    return false;
  }
  if (!std::isfinite(before.target_log_energy) ||
      !std::isfinite(after.target_log_energy) ||
      after.target_log_energy > before.target_log_energy + kTargetLogEnergyTol) {
    return false;
  }
  if (before.min_angle_degrees < config_.min_angle_floor_degrees) {
    return true;
  }
  if (after.min_angle_degrees > before.min_angle_degrees + kMinAngleTolDegrees) {
    return true;
  }
  if (after.target_log_energy + kTargetLogEnergyTol < before.target_log_energy) {
    return true;
  }
  if (std::abs(after.target_log_energy - before.target_log_energy) <= kTargetLogEnergyTol) {
    return valenceObjective(valence, boundary, a, b, c, d, true) <
           valenceObjective(valence, boundary, a, b, c, d, false);
  }
  return false;
}

std::vector<AdaIsoRemesher::VertexRelocationClass> AdaIsoRemesher::classifyVerticesForRelocation(
    const Mesh& mesh) const {
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

int AdaIsoRemesher::relocateVertices(Mesh& mesh, const Mesh& original_for_projection) {
  const auto classes = classifyVerticesForRelocation(mesh);
  int accepted = tangentSmoothing(mesh, classes);

  if (!config_.enable_relocation_mls) {
    return accepted;
  }

  const auto projected = mlsProject(mesh.vertices, original_for_projection);
  if (projected.size() != mesh.vertices.size()) {
    return accepted;
  }

  const auto vf = vertexFaces(mesh);
  for (std::size_t i = 0; i < mesh.vertices.size(); ++i) {
    if (classes[i] == VertexRelocationClass::Smooth &&
        finiteVector(projected[i]) &&
        (projected[i] - mesh.vertices[i]).norm() > kEps &&
        relocationAcceptedByAngleMetric(mesh, vf, static_cast<int>(i), projected[i])) {
      mesh.vertices[i] = projected[i];
      ++accepted;
    }
  }
  return accepted;
}

int AdaIsoRemesher::tangentSmoothing(Mesh& mesh, const std::vector<VertexRelocationClass>& classes) {
  const auto vf = vertexFaces(mesh);
  const auto normals = vertexNormals(mesh);
  const auto areas = faceAreas(mesh);
  int accepted = 0;

  for (std::size_t i = 0; i < mesh.vertices.size(); ++i) {
    if (i >= classes.size() || classes[i] != VertexRelocationClass::Smooth || vf[i].empty()) {
      continue;
    }
    std::unordered_map<int, double> weights;
    for (int fi : vf[i]) {
      const auto& face = mesh.faces[static_cast<std::size_t>(fi)];
      for (int k = 0; k < 3; ++k) {
        const int n = face[k];
        if (n != static_cast<int>(i)) {
          weights[n] += areas[static_cast<std::size_t>(fi)];
        }
      }
    }
    if (weights.empty()) {
      continue;
    }

    double sum_w = 0.0;
    Eigen::Vector3d centroid = Eigen::Vector3d::Zero();
    for (const auto& item : weights) {
      const int n = item.first;
      const double w = item.second * targetAt(mesh.vertices[static_cast<std::size_t>(n)]);
      sum_w += w;
      centroid += w * mesh.vertices[static_cast<std::size_t>(n)];
    }
    if (sum_w <= kEps) {
      continue;
    }
    centroid /= sum_w;

    const Eigen::Vector3d disp = centroid - mesh.vertices[i];
    const Eigen::Vector3d normal = normals[i];
    const Eigen::Vector3d tangent = disp - disp.dot(normal) * normal;
    const Eigen::Vector3d candidate = mesh.vertices[i] + config_.tangent_lambda * tangent;
    if (finiteVector(candidate) &&
        (candidate - mesh.vertices[i]).norm() > kEps &&
        relocationAcceptedByAngleMetric(mesh, vf, static_cast<int>(i), candidate)) {
      mesh.vertices[i] = candidate;
      ++accepted;
    }
  }
  return accepted;
}

bool AdaIsoRemesher::relocationAcceptedByAngleMetric(
    const Mesh& mesh,
    const std::vector<std::vector<int>>& vf,
    int vertex,
    const Eigen::Vector3d& candidate) const {
  if (vertex < 0 || static_cast<std::size_t>(vertex) >= vf.size() || vf[static_cast<std::size_t>(vertex)].empty() ||
      !finiteVector(candidate)) {
    return false;
  }

  std::vector<Eigen::Vector3i> faces;
  faces.reserve(vf[static_cast<std::size_t>(vertex)].size());
  for (int fi : vf[static_cast<std::size_t>(vertex)]) {
    faces.push_back(mesh.faces[static_cast<std::size_t>(fi)]);
  }

  const LocalPatchMetrics before = evaluateLocalPatch(
      mesh,
      target_field_,
      faces,
      config_.collapseFactor(),
      config_.splitFactor());
  const LocalPatchMetrics after = evaluateLocalPatch(
      mesh,
      target_field_,
      faces,
      {},
      {{vertex, candidate}},
      config_.collapseFactor(),
      config_.splitFactor(),
      false);
  if (!before.valid || !after.valid) {
    return false;
  }
  if (!std::isfinite(before.target_log_energy) ||
      !std::isfinite(after.target_log_energy) ||
      after.target_log_energy > before.target_log_energy + kTargetLogEnergyTol) {
    return false;
  }
  return minAngleQualityAccepted(
      before.min_angle_degrees,
      after.min_angle_degrees,
      config_.min_angle_floor_degrees);
}

bool AdaIsoRemesher::stageAProjectionAccepted(
    const Mesh& mesh,
    const std::vector<std::vector<int>>& vf,
    int vertex,
    const Eigen::Vector3d& candidate,
    bool& rejected_by_size_energy,
    bool& rejected_by_angle) const {
  rejected_by_size_energy = false;
  rejected_by_angle = false;
  if (vertex < 0 || static_cast<std::size_t>(vertex) >= vf.size() ||
      vf[static_cast<std::size_t>(vertex)].empty() || !finiteVector(candidate)) {
    return false;
  }

  std::vector<Eigen::Vector3i> faces;
  faces.reserve(vf[static_cast<std::size_t>(vertex)].size());
  for (int fi : vf[static_cast<std::size_t>(vertex)]) {
    faces.push_back(mesh.faces[static_cast<std::size_t>(fi)]);
  }

  const LocalPatchMetrics before = evaluateLocalPatch(
      mesh,
      target_field_,
      faces,
      config_.collapseFactor(),
      config_.splitFactor());
  const LocalPatchMetrics after = evaluateLocalPatch(
      mesh,
      target_field_,
      faces,
      {},
      {{vertex, candidate}},
      config_.collapseFactor(),
      config_.splitFactor(),
      false);
  if (!before.valid || !after.valid ||
      !std::isfinite(before.target_log_energy) ||
      !std::isfinite(after.target_log_energy) ||
      !std::isfinite(before.min_angle_degrees) ||
      !std::isfinite(after.min_angle_degrees)) {
    return false;
  }
  if (after.target_log_energy > before.target_log_energy + kStageAProjectionSizeEnergyTol) {
    rejected_by_size_energy = true;
    return false;
  }
  if (!minAngleSafetyAccepted(
          before.min_angle_degrees,
          after.min_angle_degrees,
          config_.min_angle_floor_degrees)) {
    rejected_by_angle = true;
    return false;
  }
  return true;
}

std::vector<Eigen::Vector3d> AdaIsoRemesher::mlsProject(
    const std::vector<Eigen::Vector3d>& points,
    const Mesh& original) const {
  if (original.vertices.empty()) {
    return points;
  }

  const auto normals = vertexNormals(original);
  NearestNeighborIndex tree(original.vertices);
  const std::size_t k = std::min<std::size_t>(std::max(1, config_.mls_k), original.vertexCount());
  auto out = points;

  for (std::size_t pi = 0; pi < points.size(); ++pi) {
    const Eigen::Vector3d& point = points[pi];
    auto initial = tree.nearestK(point, k);
    if (initial.empty() || std::sqrt(initial.front().second) <= kEps) {
      out[pi] = point;
      continue;
    }

    Eigen::Vector3d px = point;
    double best_err = std::numeric_limits<double>::infinity();
    for (int it = 0; it < config_.mls_iterations; ++it) {
      auto knn = tree.nearestK(px, k);
      if (knn.empty()) {
        break;
      }

      std::vector<double> dists;
      dists.reserve(knn.size());
      for (const auto& item : knn) {
        dists.push_back(std::sqrt(std::max(0.0, item.second)));
      }

      double h = dists.back();
      if (dists.size() > 1) {
        h += std::max(0.0, h - dists[dists.size() - 2]);
      }
      h = std::max(h, 1e-12);

      double sum_w = 0.0;
      Eigen::Vector3d a = Eigen::Vector3d::Zero();
      Eigen::Vector3d n = Eigen::Vector3d::Zero();
      for (std::size_t i = 0; i < knn.size(); ++i) {
        const double ratio = dists[i] / h;
        const double w = std::exp(-(ratio * ratio));
        const std::size_t idx = knn[i].first;
        sum_w += w;
        a += w * original.vertices[idx];
        n += w * normals[idx];
      }
      if (sum_w <= kEps) {
        break;
      }
      a /= sum_w;
      n /= sum_w;
      const double n_norm = n.norm();
      if (n_norm <= kEps) {
        break;
      }
      n /= n_norm;

      const double err = std::abs(n.dot(point - a));
      if (err > best_err) {
        break;
      }
      best_err = err;
      px = point - n.dot(point - a) * n;
      if (err < config_.mls_tolerance) {
        break;
      }
    }
    if (finiteVector(px)) {
      out[pi] = px;
    }
  }

  return out;
}

}  // namespace adaiso
