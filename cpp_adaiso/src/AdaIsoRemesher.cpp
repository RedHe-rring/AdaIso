#include "adaiso/AdaIsoRemesher.hpp"

#include "adaiso/Geometry.hpp"
#include "adaiso/IO.hpp"
#include "adaiso/KDTree.hpp"
#include "adaiso/Metrics.hpp"
#include "adaiso/detail/PrincipalCurvature.hpp"
#include "adaiso/detail/CurvatureFilter.hpp"
#include "adaiso/detail/ScreenedLaplacian.hpp"
#include "adaiso/detail/TargetFaceSizing.hpp"
#include "adaiso/detail/TargetGradation.hpp"

#include <Eigen/Geometry>

#include <igl/barycenter.h>
#include <igl/box_surface_area.h>
#include <igl/doublearea.h>
#include <igl/EPS.h>
#include <igl/pad_box.h>
#include <igl/point_simplex_squared_distance.h>
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
constexpr double kRelocationFeatureEps = 1e-12;
constexpr double kRelocationZeta = 0.5;
constexpr double kScalarHeatClipLowPercentile = 2.0;
constexpr double kScalarHeatClipHighPercentile = 98.0;

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
  std::vector<double> k1;
  std::vector<double> k2;
  std::vector<double> raw_kappa;
  std::vector<double> filtered_kappa;
  std::vector<double> curvature_caps;
  std::vector<double> smoothed_kappa;
  std::vector<double> initial_lengths;
  std::vector<double> limited_lengths;
  std::vector<double> target_lengths;
  std::vector<detail::PrincipalCurvatureFitDiagnostics> fit_diagnostics;
  std::vector<int> gradation_source_vertices;
  std::vector<double> gradation_source_distances;
  double base_length = 0.0;
  double min_length = 0.0;
  double max_length = 0.0;
  double pre_limiter_estimated_faces = 0.0;
  double post_limiter_estimated_faces = 0.0;
  double scale = 1.0;
  double estimated_faces = 0.0;
  double estimated_faces_mean = 0.0;
  double estimated_faces_min = 0.0;
  double estimated_faces_relative_error = 0.0;
  std::size_t target_faces = 0;
  int bisection_iterations = detail::kDefaultTargetBisectionIterations;
  std::size_t fallback_vertices = 0;
  std::size_t curvature_filter_changed_vertices = 0;
  CurvatureOutlierMode curvature_outlier_mode = CurvatureOutlierMode::None;
  TargetFaceRepresentative target_face_representative = TargetFaceRepresentative::Mean;
  bool target_faces_reachable = true;
  bool gradation_enabled = true;
  double gradation_ratio = 1.0;
  detail::TargetGradationStats gradation;
  detail::TargetGradationMetrics final_gradation;
};

struct EdgeFieldRecord {
  Edge edge;
  double edge_length = 0.0;
  double abs_dh = 0.0;
  double target_ratio = 1.0;
  double edge_length_to_min_target = 0.0;
  double h_slope = 0.0;
  double normalized_log_slope = 0.0;
};

struct FaceFieldRecord {
  Eigen::Vector3i face = Eigen::Vector3i::Zero();
  double area = 0.0;
  double mean_h = 0.0;
  double min_h = 0.0;
  double mean_estimate_contribution = 0.0;
  double min_estimate_contribution = 0.0;
  double grad_h_norm = 0.0;
  double max_edge_target_ratio = 1.0;
};

const char* curvatureOutlierModeName(CurvatureOutlierMode mode) {
  switch (mode) {
    case CurvatureOutlierMode::None:
      return "none";
    case CurvatureOutlierMode::TopKMedian:
      return "topk_median";
    case CurvatureOutlierMode::SamplingCap:
      return "sampling_cap";
  }
  return "unknown";
}

const char* targetFaceRepresentativeName(TargetFaceRepresentative value) {
  return value == TargetFaceRepresentative::Minimum ? "min" : "mean";
}

struct TargetFieldHealth {
  double estimated_faces_mean = 0.0;
  double estimated_faces_min = 0.0;
  double h_min_saturation_rate = 0.0;
  double h_max_saturation_rate = 0.0;
  double smallest_five_percent_face_budget_rate = 0.0;
  double input_edge_target_violation_rate = 0.0;
};

TargetFieldHealth measureTargetFieldHealth(
    const Mesh& mesh,
    const std::vector<double>& lengths,
    double h_min,
    double h_max,
    double collapse_factor,
    double split_factor) {
  TargetFieldHealth health;
  health.estimated_faces_mean = detail::estimateTargetFaceCount(
      mesh, lengths, TargetFaceRepresentative::Mean);
  health.estimated_faces_min = detail::estimateTargetFaceCount(
      mesh, lengths, TargetFaceRepresentative::Minimum);
  std::size_t min_saturated = 0;
  std::size_t max_saturated = 0;
  for (double value : lengths) {
    min_saturated += static_cast<std::size_t>(
        std::abs(value - h_min) <= 1e-12 * std::max(1.0, h_min));
    max_saturated += static_cast<std::size_t>(
        std::abs(value - h_max) <= 1e-12 * std::max(1.0, h_max));
  }
  health.h_min_saturation_rate =
      static_cast<double>(min_saturated) / static_cast<double>(lengths.size());
  health.h_max_saturation_rate =
      static_cast<double>(max_saturated) / static_cast<double>(lengths.size());

  struct FaceBudget {
    double h = 0.0;
    double contribution = 0.0;
  };
  std::vector<FaceBudget> budgets;
  budgets.reserve(mesh.faceCount());
  double total_budget = 0.0;
  for (const auto& face : mesh.faces) {
    const double face_h =
        (lengths[static_cast<std::size_t>(face[0])] +
         lengths[static_cast<std::size_t>(face[1])] +
         lengths[static_cast<std::size_t>(face[2])]) /
        3.0;
    const double area = triangleArea(
        mesh.vertices[static_cast<std::size_t>(face[0])],
        mesh.vertices[static_cast<std::size_t>(face[1])],
        mesh.vertices[static_cast<std::size_t>(face[2])]);
    const double contribution =
        area / (detail::kEquilateralAreaFactor * face_h * face_h);
    budgets.push_back({face_h, contribution});
    total_budget += contribution;
  }
  std::sort(budgets.begin(), budgets.end(), [](const FaceBudget& lhs, const FaceBudget& rhs) {
    return lhs.h < rhs.h;
  });
  const std::size_t count = static_cast<std::size_t>(
      std::ceil(0.05 * static_cast<double>(budgets.size())));
  double small_budget = 0.0;
  for (std::size_t i = 0; i < count; ++i) {
    small_budget += budgets[i].contribution;
  }
  health.smallest_five_percent_face_budget_rate =
      total_budget > 0.0 ? small_budget / total_budget : 0.0;

  std::size_t violations = 0;
  const auto edges = uniqueEdges(mesh);
  for (const auto& edge : edges) {
    const double edge_length =
        (mesh.vertices[static_cast<std::size_t>(edge.first)] -
         mesh.vertices[static_cast<std::size_t>(edge.second)])
            .norm();
    const double target = std::min(
        lengths[static_cast<std::size_t>(edge.first)],
        lengths[static_cast<std::size_t>(edge.second)]);
    const double ratio = edge_length / target;
    if (ratio < collapse_factor || ratio > split_factor) {
      ++violations;
    }
  }
  health.input_edge_target_violation_rate =
      edges.empty() ? 0.0
                    : static_cast<double>(violations) /
                          static_cast<double>(edges.size());
  return health;
}

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

std::pair<double, double> paddedScalarRange(double lo, double hi) {
  if (!std::isfinite(lo) || !std::isfinite(hi)) {
    return {0.0, 1.0};
  }
  if (hi <= lo) {
    const double pad = std::max(1.0, std::abs(lo)) * 0.5;
    return {lo - pad, hi + pad};
  }
  return {lo, hi};
}

std::vector<double> finiteSortedValues(const std::vector<double>& values) {
  std::vector<double> finite;
  finite.reserve(values.size());
  for (double value : values) {
    if (std::isfinite(value)) {
      finite.push_back(value);
    }
  }
  std::sort(finite.begin(), finite.end());
  return finite;
}

std::pair<double, double> percentileScalarRange(
    const std::vector<double>& values,
    double low_percentile,
    double high_percentile) {
  const std::vector<double> sorted = finiteSortedValues(values);
  if (sorted.empty()) {
    return {0.0, 1.0};
  }
  const double lo = quantileOfSorted(sorted, low_percentile / 100.0);
  const double hi = quantileOfSorted(sorted, high_percentile / 100.0);
  return paddedScalarRange(lo, hi);
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

std::vector<EdgeFieldRecord> computeEdgeFieldRecords(
    const Mesh& mesh,
    const std::vector<double>& target_lengths,
    double mean_edge_length) {
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
    record.edge_length_to_min_target = length / std::min(ha, hb);
    record.h_slope = length > kEps ? abs_dh / length : 0.0;
    record.normalized_log_slope =
        length > kEps ? std::abs(std::log(ha) - std::log(hb)) * mean_edge_length / length : 0.0;
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
    const double area = triangleArea(
        mesh.vertices[static_cast<std::size_t>(face[0])],
        mesh.vertices[static_cast<std::size_t>(face[1])],
        mesh.vertices[static_cast<std::size_t>(face[2])]);
    const double h0 = target_lengths[static_cast<std::size_t>(face[0])];
    const double h1 = target_lengths[static_cast<std::size_t>(face[1])];
    const double h2 = target_lengths[static_cast<std::size_t>(face[2])];
    const double mean_h = (h0 + h1 + h2) / 3.0;
    const double min_h = std::min({h0, h1, h2});
    FaceFieldRecord record;
    record.face = face;
    record.area = area;
    record.mean_h = mean_h;
    record.min_h = min_h;
    record.mean_estimate_contribution =
        area / (detail::kEquilateralAreaFactor * mean_h * mean_h);
    record.min_estimate_contribution =
        area / (detail::kEquilateralAreaFactor * min_h * min_h);
    record.grad_h_norm = faceTargetGradientNorm(mesh, face, target_lengths);
    record.max_edge_target_ratio = max_ratio;
    records.push_back(record);
  }
  return records;
}

void writeVerticesCsv(
    const std::filesystem::path& path,
    const Mesh& mesh,
    const TargetFieldStageData& data) {
  auto out = openOutputFile(path);
  out << "vertex_id,x,y,z,k1,k2,raw_kappa,filtered_kappa,curvature_cap,"
         "smoothed_kappa,fit_sample_ring,fit_sample_count,fit_local_scale,"
         "fit_singular_ratio,fit_residual_rms,fit_fallback,initial_h,limited_h,"
         "target_h,limiter_source_vertex,limiter_normalized_distance,"
         "limiter_log_reduction,hit_h_min,hit_h_max\n";
  for (std::size_t i = 0; i < mesh.vertices.size(); ++i) {
    const auto& p = mesh.vertices[i];
    out << i << ','
        << p[0] << ',' << p[1] << ',' << p[2] << ','
        << data.k1[i] << ','
        << data.k2[i] << ','
        << data.raw_kappa[i] << ','
        << data.filtered_kappa[i] << ','
        << data.curvature_caps[i] << ','
        << data.smoothed_kappa[i] << ','
        << data.fit_diagnostics[i].sample_ring << ','
        << data.fit_diagnostics[i].sample_count << ','
        << data.fit_diagnostics[i].local_scale << ','
        << data.fit_diagnostics[i].singular_ratio << ','
        << data.fit_diagnostics[i].residual_rms << ','
        << (data.fit_diagnostics[i].fallback ? 1 : 0) << ','
        << data.initial_lengths[i] << ','
        << data.limited_lengths[i] << ','
        << data.target_lengths[i] << ','
        << data.gradation_source_vertices[i] << ','
        << data.gradation_source_distances[i] << ','
        << std::log(data.initial_lengths[i] / data.limited_lengths[i]) << ','
        << (std::abs(data.target_lengths[i] - data.min_length) <=
                    1e-12 * std::max(1.0, data.min_length)
                ? 1
                : 0)
        << ','
        << (std::abs(data.target_lengths[i] - data.max_length) <=
                    1e-12 * std::max(1.0, data.max_length)
                ? 1
                : 0)
        << '\n';
  }
}

void writeEdgesCsv(
    const std::filesystem::path& path,
    const std::vector<EdgeFieldRecord>& records) {
  auto out = openOutputFile(path);
  out << "i,j,edge_length,abs_dh,target_ratio,edge_length_to_min_target,"
         "h_slope,normalized_log_slope\n";
  for (const auto& record : records) {
    out << record.edge.first << ','
        << record.edge.second << ','
        << record.edge_length << ','
        << record.abs_dh << ','
        << record.target_ratio << ','
        << record.edge_length_to_min_target << ','
        << record.h_slope << ','
        << record.normalized_log_slope << '\n';
  }
}

void writeFacesCsv(
    const std::filesystem::path& path,
    const std::vector<FaceFieldRecord>& records) {
  auto out = openOutputFile(path);
  out << "face_id,v0,v1,v2,area,mean_h,min_h,mean_estimate_contribution,"
         "min_estimate_contribution,grad_h_norm,max_edge_target_ratio\n";
  for (std::size_t i = 0; i < records.size(); ++i) {
    out << i << ','
        << records[i].face[0] << ','
        << records[i].face[1] << ','
        << records[i].face[2] << ','
        << records[i].area << ','
        << records[i].mean_h << ','
        << records[i].min_h << ','
        << records[i].mean_estimate_contribution << ','
        << records[i].min_estimate_contribution << ','
        << records[i].grad_h_norm << ','
        << records[i].max_edge_target_ratio << '\n';
  }
}

void writeLimiterSourcesCsv(
    const std::filesystem::path& path,
    const TargetFieldStageData& data,
    const std::vector<FaceFieldRecord>& faces) {
  struct SourceAggregate {
    std::size_t influenced_vertices = 0;
    double assigned_area = 0.0;
    double assigned_mean_estimated_faces = 0.0;
  };
  std::unordered_map<int, SourceAggregate> aggregates;
  for (int source : data.gradation_source_vertices) {
    ++aggregates[source].influenced_vertices;
  }
  for (const auto& record : faces) {
    int min_vertex = record.face[0];
    for (int k = 1; k < 3; ++k) {
      const int candidate = record.face[k];
      if (data.limited_lengths[static_cast<std::size_t>(candidate)] <
          data.limited_lengths[static_cast<std::size_t>(min_vertex)]) {
        min_vertex = candidate;
      }
    }
    const int source =
        data.gradation_source_vertices[static_cast<std::size_t>(min_vertex)];
    aggregates[source].assigned_area += record.area;
    aggregates[source].assigned_mean_estimated_faces +=
        record.mean_estimate_contribution;
  }
  std::vector<int> sources;
  sources.reserve(aggregates.size());
  for (const auto& item : aggregates) {
    sources.push_back(item.first);
  }
  std::sort(sources.begin(), sources.end());
  auto out = openOutputFile(path);
  out << "source_vertex,source_initial_h,influenced_vertices,assigned_area,"
         "assigned_mean_estimated_faces\n";
  for (int source : sources) {
    const auto& aggregate = aggregates.at(source);
    const double initial_h = source >= 0
                                 ? data.initial_lengths[static_cast<std::size_t>(source)]
                                 : 0.0;
    out << source << ',' << initial_h << ',' << aggregate.influenced_vertices
        << ',' << aggregate.assigned_area << ','
        << aggregate.assigned_mean_estimated_faces << '\n';
  }
}

std::string targetFieldStatsJson(
    const TargetFieldStageData& data,
    const std::vector<EdgeFieldRecord>& edge_records,
    const std::vector<FaceFieldRecord>& face_records) {
  std::vector<double> ratios;
  std::vector<double> normalized_log_slopes;
  ratios.reserve(edge_records.size());
  normalized_log_slopes.reserve(edge_records.size());
  for (const auto& record : edge_records) {
    ratios.push_back(record.target_ratio);
    normalized_log_slopes.push_back(record.normalized_log_slope);
  }
  std::sort(ratios.begin(), ratios.end());
  std::sort(normalized_log_slopes.begin(), normalized_log_slopes.end());

  std::vector<double> gradients;
  gradients.reserve(face_records.size());
  for (const auto& record : face_records) {
    gradients.push_back(record.grad_h_norm);
  }
  std::sort(gradients.begin(), gradients.end());

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

  const double raw_kappa_max = data.raw_kappa.empty()
                                   ? 0.0
                                   : *std::max_element(data.raw_kappa.begin(), data.raw_kappa.end());
  const double filtered_kappa_max = data.filtered_kappa.empty()
                                        ? 0.0
                                        : *std::max_element(
                                              data.filtered_kappa.begin(),
                                              data.filtered_kappa.end());
  const double smoothed_kappa_max = data.smoothed_kappa.empty()
                                        ? 0.0
                                        : *std::max_element(
                                              data.smoothed_kappa.begin(),
                                              data.smoothed_kappa.end());
  double min_singular_ratio = std::numeric_limits<double>::infinity();
  double max_residual_rms = 0.0;
  double max_curvature_times_local_scale = 0.0;
  for (std::size_t i = 0; i < data.fit_diagnostics.size(); ++i) {
    const auto& diagnostics = data.fit_diagnostics[i];
    if (!diagnostics.fallback && diagnostics.singular_ratio > 0.0) {
      min_singular_ratio = std::min(min_singular_ratio, diagnostics.singular_ratio);
    }
    max_residual_rms = std::max(max_residual_rms, diagnostics.residual_rms);
    if (data.base_length > 0.0) {
      max_curvature_times_local_scale = std::max(
          max_curvature_times_local_scale,
          data.raw_kappa[i] * diagnostics.local_scale / data.base_length);
    }
  }
  if (!std::isfinite(min_singular_ratio)) {
    min_singular_ratio = 0.0;
  }

  std::size_t min_saturated = 0;
  std::size_t max_saturated = 0;
  for (double value : data.target_lengths) {
    if (std::abs(value - data.min_length) <=
        1e-12 * std::max(1.0, data.min_length)) {
      ++min_saturated;
    }
    if (std::abs(value - data.max_length) <=
        1e-12 * std::max(1.0, data.max_length)) {
      ++max_saturated;
    }
  }
  std::vector<std::size_t> face_order(face_records.size(), 0);
  for (std::size_t i = 0; i < face_order.size(); ++i) {
    face_order[i] = i;
  }
  std::sort(face_order.begin(), face_order.end(), [&](std::size_t lhs, std::size_t rhs) {
    return face_records[lhs].mean_h < face_records[rhs].mean_h;
  });
  const std::size_t small_face_count =
      static_cast<std::size_t>(std::ceil(0.05 * static_cast<double>(face_order.size())));
  double small_face_budget = 0.0;
  double total_mean_face_budget = 0.0;
  for (const auto& record : face_records) {
    total_mean_face_budget += record.mean_estimate_contribution;
  }
  for (std::size_t i = 0; i < small_face_count; ++i) {
    small_face_budget += face_records[face_order[i]].mean_estimate_contribution;
  }
  std::size_t input_edge_violations = 0;
  for (const auto& record : edge_records) {
    if (record.edge_length_to_min_target < 0.8 ||
        record.edge_length_to_min_target > 1.65) {
      ++input_edge_violations;
    }
  }
  std::unordered_set<int> limiter_sources(
      data.gradation_source_vertices.begin(), data.gradation_source_vertices.end());

  std::ostringstream out;
  out << std::setprecision(17);
  out << "{\n";
  out << "  \"base_length\": " << data.base_length << ",\n";
  out << "  \"h_min\": " << data.min_length << ",\n";
  out << "  \"h_max\": " << data.max_length << ",\n";
  out << "  \"target_faces\": " << data.target_faces << ",\n";
  out << "  \"target_bisection_iterations\": " << data.bisection_iterations << ",\n";
  out << "  \"curvature_outlier_mode\": \""
      << curvatureOutlierModeName(data.curvature_outlier_mode) << "\",\n";
  out << "  \"curvature_filter_changed_vertices\": "
      << data.curvature_filter_changed_vertices << ",\n";
  out << "  \"raw_kappa_max\": " << raw_kappa_max << ",\n";
  out << "  \"filtered_kappa_max\": " << filtered_kappa_max << ",\n";
  out << "  \"smoothed_kappa_max\": " << smoothed_kappa_max << ",\n";
  out << "  \"principal_fit_min_singular_ratio\": " << min_singular_ratio << ",\n";
  out << "  \"principal_fit_max_residual_rms\": " << max_residual_rms << ",\n";
  out << "  \"principal_fit_max_curvature_times_local_scale\": "
      << max_curvature_times_local_scale << ",\n";
  out << "  \"pre_limiter_estimated_faces\": " << data.pre_limiter_estimated_faces << ",\n";
  out << "  \"post_limiter_estimated_faces\": " << data.post_limiter_estimated_faces << ",\n";
  out << "  \"scale\": " << data.scale << ",\n";
  out << "  \"estimated_faces\": " << data.estimated_faces << ",\n";
  out << "  \"target_face_representative\": \""
      << targetFaceRepresentativeName(data.target_face_representative) << "\",\n";
  out << "  \"estimated_faces_mean\": " << data.estimated_faces_mean << ",\n";
  out << "  \"estimated_faces_min\": " << data.estimated_faces_min << ",\n";
  out << "  \"min_mean_estimate_ratio\": "
      << (data.estimated_faces_mean > 0.0
              ? data.estimated_faces_min / data.estimated_faces_mean
              : 0.0)
      << ",\n";
  out << "  \"estimated_faces_error\": " << (data.estimated_faces - static_cast<double>(data.target_faces)) << ",\n";
  out << "  \"estimated_faces_relative_error\": " << data.estimated_faces_relative_error << ",\n";
  out << "  \"target_faces_reachable\": " << (data.target_faces_reachable ? "true" : "false") << ",\n";
  out << "  \"target_gradation_enabled\": " << (data.gradation_enabled ? "true" : "false") << ",\n";
  out << "  \"target_gradation_ratio\": " << data.gradation_ratio << ",\n";
  out << "  \"gradation_runtime_ms\": " << data.gradation.runtime_ms << ",\n";
  out << "  \"gradation_queue_pops\": " << data.gradation.queue_pops << ",\n";
  out << "  \"gradation_relaxations\": " << data.gradation.relaxations << ",\n";
  out << "  \"gradation_changed_vertices\": " << data.gradation.changed_vertices << ",\n";
  out << "  \"gradation_changed_vertex_rate\": " << data.gradation.changed_vertex_rate << ",\n";
  out << "  \"gradation_source_count\": " << limiter_sources.size() << ",\n";
  out << "  \"gradation_min_log_reduction\": " << data.gradation.min_log_reduction << ",\n";
  out << "  \"gradation_mean_log_reduction\": " << data.gradation.mean_log_reduction << ",\n";
  out << "  \"gradation_max_log_reduction\": " << data.gradation.max_log_reduction << ",\n";
  out << "  \"gradation_max_normalized_log_slope_before\": "
      << data.gradation.before.max_normalized_log_slope << ",\n";
  out << "  \"gradation_max_normalized_log_slope_after\": "
      << data.gradation.after.max_normalized_log_slope << ",\n";
  out << "  \"final_gradation_max_normalized_log_slope\": "
      << data.final_gradation.max_normalized_log_slope << ",\n";
  out << "  \"final_gradation_p95_normalized_log_slope\": "
      << data.final_gradation.p95_normalized_log_slope << ",\n";
  out << "  \"final_gradation_p99_normalized_log_slope\": "
      << data.final_gradation.p99_normalized_log_slope << ",\n";
  out << "  \"principal_curvature_fallback_vertices\": " << data.fallback_vertices << ",\n";
  out << "  \"principal_curvature_fallback_rate\": "
      << static_cast<double>(data.fallback_vertices) / static_cast<double>(data.target_lengths.size()) << ",\n";
  out << "  \"observed_h_min\": " << observed_min << ",\n";
  out << "  \"observed_h_max\": " << observed_max << ",\n";
  out << "  \"h_min_saturation_rate\": "
      << static_cast<double>(min_saturated) / static_cast<double>(data.target_lengths.size())
      << ",\n";
  out << "  \"h_max_saturation_rate\": "
      << static_cast<double>(max_saturated) / static_cast<double>(data.target_lengths.size())
      << ",\n";
  out << "  \"smallest_five_percent_face_budget_rate\": "
      << (total_mean_face_budget > 0.0 ? small_face_budget / total_mean_face_budget : 0.0)
      << ",\n";
  out << "  \"input_edge_target_violation_rate\": "
      << (edge_records.empty()
              ? 0.0
              : static_cast<double>(input_edge_violations) /
                    static_cast<double>(edge_records.size()))
      << ",\n";
  out << "  \"range_ok\": " << (range_ok ? "true" : "false") << ",\n";
  out << "  \"edge_count\": " << edge_records.size() << ",\n";
  out << "  \"max_edge_target_ratio\": " << max_ratio << ",\n";
  out << "  \"p95_edge_target_ratio\": " << quantileOfSorted(ratios, 0.95) << ",\n";
  out << "  \"p99_edge_target_ratio\": " << quantileOfSorted(ratios, 0.99) << ",\n";
  out << "  \"max_normalized_log_slope\": "
      << (normalized_log_slopes.empty() ? 0.0 : normalized_log_slopes.back()) << ",\n";
  out << "  \"p95_normalized_log_slope\": "
      << quantileOfSorted(normalized_log_slopes, 0.95) << ",\n";
  out << "  \"p99_normalized_log_slope\": "
      << quantileOfSorted(normalized_log_slopes, 0.99) << ",\n";
  out << "  \"face_count\": " << face_records.size() << ",\n";
  out << "  \"max_grad_h_norm\": " << (gradients.empty() ? 0.0 : gradients.back()) << ",\n";
  out << "  \"p95_grad_h_norm\": " << quantileOfSorted(gradients, 0.95) << ",\n";
  out << "  \"p99_grad_h_norm\": " << quantileOfSorted(gradients, 0.99) << "\n";
  out << "}\n";
  return out.str();
}

std::string colorScalesJson(
    const std::pair<double, double>& k1_range,
    const std::pair<double, double>& k2_range,
    const std::pair<double, double>& kappa_range,
    double target_h_min,
    double target_h_max,
    double clip_low_percentile,
    double clip_high_percentile) {
  std::ostringstream out;
  out << std::setprecision(17);
  out << "{\n";
  out << "  \"palette\": \"heat\",\n";
  out << "  \"normalization\": \"linear_clamp_percentile\",\n";
  out << "  \"clip_low_percentile\": " << clip_low_percentile << ",\n";
  out << "  \"clip_high_percentile\": " << clip_high_percentile << ",\n";
  out << "  \"scales\": {\n";
  out << "    \"k1_abs\": {\n";
  out << "      \"file\": \"k1_abs_heat.ply\",\n";
  out << "      \"field\": \"abs(k1)\",\n";
  out << "      \"min\": " << k1_range.first << ",\n";
  out << "      \"max\": " << k1_range.second << "\n";
  out << "    },\n";
  out << "    \"k2_abs\": {\n";
  out << "      \"file\": \"k2_abs_heat.ply\",\n";
  out << "      \"field\": \"abs(k2)\",\n";
  out << "      \"min\": " << k2_range.first << ",\n";
  out << "      \"max\": " << k2_range.second << "\n";
  out << "    },\n";
  out << "    \"raw_kappa\": {\n";
  out << "      \"file\": \"raw_kappa_heat.ply\",\n";
  out << "      \"field\": \"raw_dimensionless_kappa\",\n";
  out << "      \"min\": " << kappa_range.first << ",\n";
  out << "      \"max\": " << kappa_range.second << "\n";
  out << "    },\n";
  out << "    \"filtered_kappa\": {\n";
  out << "      \"file\": \"filtered_kappa_heat.ply\",\n";
  out << "      \"field\": \"filtered_dimensionless_kappa\",\n";
  out << "      \"min\": " << kappa_range.first << ",\n";
  out << "      \"max\": " << kappa_range.second << "\n";
  out << "    },\n";
  out << "    \"smoothed_kappa\": {\n";
  out << "      \"file\": \"smoothed_kappa_heat.ply\",\n";
  out << "      \"field\": \"smoothed_dimensionless_kappa\",\n";
  out << "      \"min\": " << kappa_range.first << ",\n";
  out << "      \"max\": " << kappa_range.second << "\n";
  out << "    },\n";
  out << "    \"initial_h\": {\n";
  out << "      \"file\": \"initial_h_heat.ply\",\n";
  out << "      \"field\": \"initial_h\",\n";
  out << "      \"min\": " << target_h_min << ",\n";
  out << "      \"max\": " << target_h_max << ",\n";
  out << "      \"inverted\": true\n";
  out << "    },\n";
  out << "    \"limited_h\": {\n";
  out << "      \"file\": \"limited_h_heat.ply\",\n";
  out << "      \"field\": \"limited_h\",\n";
  out << "      \"min\": " << target_h_min << ",\n";
  out << "      \"max\": " << target_h_max << ",\n";
  out << "      \"inverted\": true\n";
  out << "    },\n";
  out << "    \"target_h\": {\n";
  out << "      \"file\": \"target_h_heat.ply\",\n";
  out << "      \"field\": \"target_h\",\n";
  out << "      \"min\": " << target_h_min << ",\n";
  out << "      \"max\": " << target_h_max << ",\n";
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
  if (data.k1.size() != n || data.k2.size() != n || data.raw_kappa.size() != n ||
      data.filtered_kappa.size() != n || data.curvature_caps.size() != n ||
      data.smoothed_kappa.size() != n || data.fit_diagnostics.size() != n ||
      data.gradation_source_vertices.size() != n ||
      data.gradation_source_distances.size() != n ||
      data.initial_lengths.size() != n ||
      data.limited_lengths.size() != n ||
      data.target_lengths.size() != n) {
    throw std::runtime_error("target diagnostic field size mismatch");
  }
  std::filesystem::create_directories(out_dir);

  const auto edge_records = computeEdgeFieldRecords(mesh, data.target_lengths, data.base_length);
  std::unordered_map<Edge, double, EdgeHash> edge_ratios;
  edge_ratios.reserve(edge_records.size());
  for (const auto& record : edge_records) {
    edge_ratios.emplace(record.edge, record.target_ratio);
  }
  const auto face_records = computeFaceFieldRecords(mesh, data.target_lengths, edge_ratios);

  writeVerticesCsv(out_dir / "vertices.csv", mesh, data);
  writeEdgesCsv(out_dir / "edges.csv", edge_records);
  writeFacesCsv(out_dir / "faces.csv", face_records);
  writeLimiterSourcesCsv(out_dir / "limiter_sources.csv", data, face_records);
  writeTextFile(
      out_dir / "field_stats.json",
      targetFieldStatsJson(
          data,
          edge_records,
          face_records));

  std::vector<double> k1_abs(data.k1.size(), 0.0);
  std::vector<double> k2_abs(data.k2.size(), 0.0);
  std::transform(data.k1.begin(), data.k1.end(), k1_abs.begin(), [](double value) { return std::abs(value); });
  std::transform(data.k2.begin(), data.k2.end(), k2_abs.begin(), [](double value) { return std::abs(value); });
  const auto k1_range = percentileScalarRange(
      k1_abs, kScalarHeatClipLowPercentile, kScalarHeatClipHighPercentile);
  const auto k2_range = percentileScalarRange(
      k2_abs, kScalarHeatClipLowPercentile, kScalarHeatClipHighPercentile);
  std::vector<double> combined_kappa = data.raw_kappa;
  combined_kappa.insert(
      combined_kappa.end(), data.filtered_kappa.begin(), data.filtered_kappa.end());
  combined_kappa.insert(
      combined_kappa.end(), data.smoothed_kappa.begin(), data.smoothed_kappa.end());
  const auto kappa_range = percentileScalarRange(
      combined_kappa, kScalarHeatClipLowPercentile, kScalarHeatClipHighPercentile);
  writeTextFile(
      out_dir / "color_scales.json",
      colorScalesJson(
          k1_range,
          k2_range,
          kappa_range,
          data.min_length,
          data.max_length,
          kScalarHeatClipLowPercentile,
          kScalarHeatClipHighPercentile));
  writeVertexHeatPly(
      out_dir / "k1_abs_heat.ply",
      mesh,
      k1_abs,
      k1_range.first,
      k1_range.second);
  writeVertexHeatPly(
      out_dir / "k2_abs_heat.ply",
      mesh,
      k2_abs,
      k2_range.first,
      k2_range.second);
  writeVertexHeatPly(
      out_dir / "raw_kappa_heat.ply",
      mesh,
      data.raw_kappa,
      kappa_range.first,
      kappa_range.second);
  writeVertexHeatPly(
      out_dir / "filtered_kappa_heat.ply",
      mesh,
      data.filtered_kappa,
      kappa_range.first,
      kappa_range.second);
  writeVertexHeatPly(
      out_dir / "smoothed_kappa_heat.ply",
      mesh,
      data.smoothed_kappa,
      kappa_range.first,
      kappa_range.second);
  writeVertexHeatPly(
      out_dir / "initial_h_heat.ply",
      mesh,
      data.initial_lengths,
      data.min_length,
      data.max_length,
      true);
  writeVertexHeatPly(
      out_dir / "limited_h_heat.ply",
      mesh,
      data.limited_lengths,
      data.min_length,
      data.max_length,
      true);
  writeVertexHeatPly(
      out_dir / "target_h_heat.ply",
      mesh,
      data.target_lengths,
      data.min_length,
      data.max_length,
      true);
}

}  // namespace

std::vector<Eigen::Vector3i> splitFaceOnEdge(const Eigen::Vector3i& face, int a, int b, int m);
void splitEdgesBatch(Mesh& mesh, const std::vector<Edge>& edges);

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
  for (const auto& edge : edges) {
    const int a = edge.first;
    const int b = edge.second;
    const double length = (mesh.vertices[static_cast<std::size_t>(a)] - mesh.vertices[static_cast<std::size_t>(b)]).norm();
    const double target = std::min(
        target_field.valueAt(mesh.vertices[static_cast<std::size_t>(a)]),
        target_field.valueAt(mesh.vertices[static_cast<std::size_t>(b)]));
    if (target <= kEps || !std::isfinite(length) || !std::isfinite(target)) {
      continue;
    }
    const double ratio = length / target;
    if (!std::isfinite(ratio)) {
      continue;
    }
    ratios.push_back(ratio);
    if (ratio < collapse_factor || ratio > split_factor) {
      ++violations;
    }
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
  return stats;
}

double AdaIsoConfig::splitFactor() const {
  return variant == Variant::Paper ? 5.0 / 3.0 : 1.65;
}

double AdaIsoConfig::collapseFactor() const {
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

AdaIsoRemesher::AdaIsoRemesher(AdaIsoConfig config) : config_(config) {}

Mesh AdaIsoRemesher::remesh(const Mesh& input) {
  if (config_.curvature_outlier_mode == CurvatureOutlierMode::SamplingCap &&
      (!std::isfinite(config_.curvature_cap_beta) ||
       config_.curvature_cap_beta <= 0.0)) {
    throw std::runtime_error("curvature cap beta must be positive and finite");
  }
  if (config_.curvature_smoothing_alpha < 0.0 ||
      !std::isfinite(config_.curvature_smoothing_alpha)) {
    throw std::runtime_error("curvature smoothing alpha must be finite and non-negative");
  }
  if (!std::isfinite(config_.target_gradation_ratio) || config_.target_gradation_ratio <= 1.0) {
    throw std::runtime_error("target gradation ratio must be finite and greater than one");
  }
  if (config_.target_bisection_iterations < 0) {
    throw std::runtime_error("target bisection iterations must be non-negative");
  }
  if (!std::isfinite(config_.target_max_factor) ||
      config_.target_max_factor <= detail::kTargetMinEdgeFactor) {
    throw std::runtime_error("target max factor must exceed the target minimum factor");
  }
  if (config_.max_flip_dihedral_degrees <= 0.0 || config_.max_flip_dihedral_degrees > 180.0 ||
      !std::isfinite(config_.max_flip_dihedral_degrees)) {
    throw std::runtime_error("max flip dihedral angle must be finite and in (0, 180]");
  }
  if (config_.collapse_feature_omega < 0.0 || !std::isfinite(config_.collapse_feature_omega)) {
    throw std::runtime_error("collapse feature omega must be finite and non-negative");
  }

  stats_ = {};
  Mesh mesh = removeDegenerateFaces(input);
  const std::size_t effective_target_faces =
      config_.target_faces > 0 ? config_.target_faces : mesh.faceCount();
  // TODO: 入口添加一个开关，用于输出mesh清理后的结果


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

  buildTargetLengths(mesh, effective_target_faces);
  stats_.iterations.reserve(static_cast<std::size_t>(std::max(0, config_.global_iterations)));
  if (!config_.iteration_mesh_export_dir.empty()) {
    writeMesh(std::filesystem::path(config_.iteration_mesh_export_dir) / "iter000.ply", mesh);
  }

  for (int it = 0; it < config_.global_iterations; ++it) {
    if (config_.verbose) {
      std::cerr << "iteration " << (it + 1) << "/" << config_.global_iterations << "\n";
    }
    const bool export_first_iteration_stages = it == 0 && !config_.stage_mesh_export_dir.empty();
    auto write_stage_mesh = [&](const std::string& stage) {
      if (export_first_iteration_stages) {
        writeMesh(
            std::filesystem::path(config_.stage_mesh_export_dir) / ("iter001_after_" + stage + ".ply"),
            mesh);
      }
      if (!config_.all_stage_mesh_export_dir.empty()) {
        std::ostringstream filename;
        filename << "iter" << std::setw(3) << std::setfill('0') << (it + 1)
                 << "_after_" << stage << ".ply";
        writeMesh(std::filesystem::path(config_.all_stage_mesh_export_dir) / filename.str(), mesh);
      }
    };

    const int split_count = splitLongEdges(mesh);
    write_stage_mesh("split");
    const int collapse_count = collapseShortEdges(mesh);
    mesh = removeDegenerateFaces(mesh);
    write_stage_mesh("collapse");
    const int flip_count = flipEdges(mesh);
    write_stage_mesh("flip");
    relocateVertices(mesh, original_for_projection);
    mesh = removeDegenerateFaces(mesh);
    write_stage_mesh("smooth");
    const MeshSummary iteration_summary = summarizeMesh(mesh);
    const EdgeRatioStats ratio_stats = edgeRatioStats(mesh);
    if (!config_.iteration_mesh_export_dir.empty()) {
      std::ostringstream filename;
      filename << "iter" << std::setw(3) << std::setfill('0') << (it + 1) << ".ply";
      writeMesh(std::filesystem::path(config_.iteration_mesh_export_dir) / filename.str(), mesh);
    }
    stats_.iterations.push_back({
        it + 1,
        split_count,
        collapse_count,
        flip_count,
        mesh.vertexCount(),
        mesh.faceCount(),
        iteration_summary.qavg,
        iteration_summary.qmin,
        ratio_stats.mean_ratio,
        ratio_stats.std_ratio,
        ratio_stats.p05_ratio,
        ratio_stats.p95_ratio,
        ratio_stats.violation_rate});
    if (config_.verbose) {
      std::cerr << "  split=" << split_count << ", collapse=" << collapse_count << ", flip=" << flip_count
                << ", violation=" << ratio_stats.violation_rate
                << ", mesh=" << mesh.vertexCount() << "v/" << mesh.faceCount() << "f\n";
    }
  }

  stats_.vertices = mesh.vertexCount();
  stats_.faces = mesh.faceCount();
  return mesh;
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

void AdaIsoRemesher::buildTargetLengths(const Mesh& mesh, std::size_t target_faces) {
  const double base_len = meanEdgeLength(mesh);
  const auto principal = detail::computePrincipalCurvatures(mesh, base_len);
  const auto filtered = detail::filterCurvatureOutliers(
      mesh,
      principal,
      base_len,
      config_.curvature_outlier_mode,
      config_.curvature_outlier_top_k,
      config_.curvature_cap_beta);

  std::vector<double> smoothed_kappa = filtered.filtered_kappa;
  detail::ScreenedLaplacianResult curvature_smoothing;
  if (config_.curvature_smoothing_alpha > 0.0) {
    curvature_smoothing = detail::solveScreenedScalarField(
        mesh,
        filtered.filtered_kappa,
        config_.curvature_smoothing_alpha);
    smoothed_kappa = curvature_smoothing.values;
    for (double& value : smoothed_kappa) {
      if (!std::isfinite(value)) {
        throw std::runtime_error("curvature smoothing produced a non-finite value");
      }
      value = std::max(value, 0.0);
    }
  }

  const auto initial = detail::buildInitialPrincipalCurvatureLengths(
      mesh,
      smoothed_kappa,
      base_len,
      config_.target_max_factor);

  detail::TargetGradationResult gradation;
  if (config_.enable_target_gradation) {
    gradation = detail::applyDownwardLogGradationLimiter(
        mesh,
        initial.initial_lengths,
        base_len,
        config_.target_gradation_ratio);
  } else {
    gradation.limited_lengths = initial.initial_lengths;
    gradation.source_vertices.resize(mesh.vertexCount(), -1);
    gradation.normalized_source_distances.assign(mesh.vertexCount(), 0.0);
    for (std::size_t i = 0; i < mesh.vertexCount(); ++i) {
      gradation.source_vertices[i] = static_cast<int>(i);
    }
    gradation.stats.before =
        detail::measureTargetGradation(mesh, initial.initial_lengths, base_len);
    gradation.stats.after = gradation.stats.before;
  }

  const double pre_limiter_estimated_faces =
      detail::estimateTargetFaceCount(mesh, initial.initial_lengths);
  const double post_limiter_estimated_faces =
      detail::estimateTargetFaceCount(mesh, gradation.limited_lengths);
  const auto final = detail::calibrateTargetLengthsToFaceCount(
      mesh,
      gradation.limited_lengths,
      target_faces,
      initial.h_min,
      initial.h_max,
      config_.target_bisection_iterations,
      config_.target_face_representative);
  const TargetFieldHealth field_health = measureTargetFieldHealth(
      mesh,
      final.target_lengths,
      initial.h_min,
      initial.h_max,
      config_.collapseFactor(),
      config_.splitFactor());

  const detail::TargetGradationMetrics final_gradation =
      detail::measureTargetGradation(mesh, final.target_lengths, base_len);
  if (config_.enable_target_gradation) {
    const double allowed_slope = std::log(config_.target_gradation_ratio);
    const double tolerance = 1e-10 * std::max(1.0, std::abs(allowed_slope));
    if (final_gradation.max_normalized_log_slope > allowed_slope + tolerance) {
      throw std::runtime_error("final target field violates the configured gradation ratio");
    }
  }

  stats_.curvature_smoothing_applied = config_.curvature_smoothing_alpha > 0.0;
  stats_.curvature_smoothing_assembly_time_ms = curvature_smoothing.stats.assembly_time_ms;
  stats_.curvature_smoothing_factorization_time_ms = curvature_smoothing.stats.factorization_time_ms;
  stats_.curvature_smoothing_solve_time_ms = curvature_smoothing.stats.solve_time_ms;
  stats_.curvature_smoothing_residual_abs = curvature_smoothing.stats.residual_abs;
  stats_.curvature_smoothing_residual_scaled = curvature_smoothing.stats.residual_scaled;
  stats_.curvature_smoothing_matrix_nonzeros = curvature_smoothing.stats.matrix_nonzeros;
  stats_.curvature_smoothing_solver_success = curvature_smoothing.stats.solver_success;
  stats_.principal_curvature_fallback_vertices = principal.fallback_vertices;
  stats_.principal_curvature_fallback_rate =
      static_cast<double>(principal.fallback_vertices) / static_cast<double>(mesh.vertexCount());
  stats_.principal_fit_min_singular_ratio = std::numeric_limits<double>::infinity();
  for (std::size_t i = 0; i < principal.fit_diagnostics.size(); ++i) {
    const auto& diagnostics = principal.fit_diagnostics[i];
    if (!diagnostics.fallback && diagnostics.singular_ratio > 0.0) {
      stats_.principal_fit_min_singular_ratio = std::min(
          stats_.principal_fit_min_singular_ratio, diagnostics.singular_ratio);
    }
    stats_.principal_fit_max_residual_rms = std::max(
        stats_.principal_fit_max_residual_rms, diagnostics.residual_rms);
    stats_.principal_fit_max_curvature_times_local_scale = std::max(
        stats_.principal_fit_max_curvature_times_local_scale,
        principal.kappa[i] * diagnostics.local_scale / base_len);
  }
  if (!std::isfinite(stats_.principal_fit_min_singular_ratio)) {
    stats_.principal_fit_min_singular_ratio = 0.0;
  }
  stats_.curvature_filter_changed_vertices = filtered.changed_vertices;
  stats_.curvature_filter_changed_vertex_rate =
      static_cast<double>(filtered.changed_vertices) /
      static_cast<double>(mesh.vertexCount());
  stats_.raw_kappa_max = filtered.raw_max;
  stats_.filtered_kappa_max = filtered.filtered_max;
  stats_.smoothed_kappa_max = smoothed_kappa.empty()
                                  ? 0.0
                                  : *std::max_element(
                                        smoothed_kappa.begin(), smoothed_kappa.end());
  stats_.pre_limiter_estimated_faces = pre_limiter_estimated_faces;
  stats_.post_limiter_estimated_faces = post_limiter_estimated_faces;
  stats_.gradation_runtime_ms = gradation.stats.runtime_ms;
  stats_.gradation_queue_pops = gradation.stats.queue_pops;
  stats_.gradation_relaxations = gradation.stats.relaxations;
  stats_.gradation_changed_vertices = gradation.stats.changed_vertices;
  stats_.gradation_changed_vertex_rate = gradation.stats.changed_vertex_rate;
  stats_.gradation_min_log_reduction = gradation.stats.min_log_reduction;
  stats_.gradation_mean_log_reduction = gradation.stats.mean_log_reduction;
  stats_.gradation_max_log_reduction = gradation.stats.max_log_reduction;
  stats_.gradation_max_normalized_log_slope_before =
      gradation.stats.before.max_normalized_log_slope;
  stats_.gradation_max_normalized_log_slope_after =
      gradation.stats.after.max_normalized_log_slope;
  stats_.gradation_p95_normalized_log_slope_after =
      gradation.stats.after.p95_normalized_log_slope;
  stats_.gradation_p99_normalized_log_slope_after =
      gradation.stats.after.p99_normalized_log_slope;
  stats_.gradation_max_adjacent_ratio_before = gradation.stats.before.max_adjacent_ratio;
  stats_.gradation_max_adjacent_ratio_after = gradation.stats.after.max_adjacent_ratio;
  stats_.final_gradation_max_normalized_log_slope =
      final_gradation.max_normalized_log_slope;
  stats_.final_gradation_p95_normalized_log_slope =
      final_gradation.p95_normalized_log_slope;
  stats_.final_gradation_p99_normalized_log_slope =
      final_gradation.p99_normalized_log_slope;
  stats_.final_gradation_max_adjacent_ratio = final_gradation.max_adjacent_ratio;
  stats_.final_estimated_faces = final.estimated_faces;
  stats_.final_estimated_faces_mean = field_health.estimated_faces_mean;
  stats_.final_estimated_faces_min = field_health.estimated_faces_min;
  stats_.final_min_mean_estimate_ratio =
      field_health.estimated_faces_mean > 0.0
          ? field_health.estimated_faces_min / field_health.estimated_faces_mean
          : 0.0;
  stats_.final_face_estimate_relative_error = final.relative_error;
  stats_.target_h_min_saturation_rate = field_health.h_min_saturation_rate;
  stats_.target_h_max_saturation_rate = field_health.h_max_saturation_rate;
  stats_.smallest_five_percent_face_budget_rate =
      field_health.smallest_five_percent_face_budget_rate;
  stats_.input_edge_target_violation_rate =
      field_health.input_edge_target_violation_rate;
  stats_.target_scale = final.scale;
  stats_.target_faces_reachable = final.target_faces_reachable;

  if (!config_.target_field_export_dir.empty()) {
    TargetFieldStageData diagnostic;
    diagnostic.k1 = principal.k1;
    diagnostic.k2 = principal.k2;
    diagnostic.raw_kappa = principal.kappa;
    diagnostic.filtered_kappa = filtered.filtered_kappa;
    diagnostic.curvature_caps = filtered.applied_caps;
    diagnostic.smoothed_kappa = smoothed_kappa;
    diagnostic.fit_diagnostics = principal.fit_diagnostics;
    diagnostic.initial_lengths = initial.initial_lengths;
    diagnostic.limited_lengths = gradation.limited_lengths;
    diagnostic.target_lengths = final.target_lengths;
    diagnostic.gradation_source_vertices = gradation.source_vertices;
    diagnostic.gradation_source_distances = gradation.normalized_source_distances;
    diagnostic.base_length = base_len;
    diagnostic.min_length = initial.h_min;
    diagnostic.max_length = initial.h_max;
    diagnostic.pre_limiter_estimated_faces = pre_limiter_estimated_faces;
    diagnostic.post_limiter_estimated_faces = post_limiter_estimated_faces;
    diagnostic.scale = final.scale;
    diagnostic.estimated_faces = final.estimated_faces;
    diagnostic.estimated_faces_mean = field_health.estimated_faces_mean;
    diagnostic.estimated_faces_min = field_health.estimated_faces_min;
    diagnostic.estimated_faces_relative_error = final.relative_error;
    diagnostic.target_faces = final.target_faces;
    diagnostic.bisection_iterations = config_.target_bisection_iterations;
    diagnostic.fallback_vertices = principal.fallback_vertices;
    diagnostic.curvature_filter_changed_vertices = filtered.changed_vertices;
    diagnostic.curvature_outlier_mode = config_.curvature_outlier_mode;
    diagnostic.target_face_representative = config_.target_face_representative;
    diagnostic.target_faces_reachable = final.target_faces_reachable;
    diagnostic.gradation_enabled = config_.enable_target_gradation;
    diagnostic.gradation_ratio = config_.target_gradation_ratio;
    diagnostic.gradation = gradation.stats;
    diagnostic.final_gradation = final_gradation;
    writeTargetFieldDiagnostics(
        std::filesystem::path(config_.target_field_export_dir),
        mesh,
        diagnostic);
  }

  target_field_.build(mesh, final.target_lengths);

  if (config_.verbose && !final.target_lengths.empty()) {
    const auto [min_it, max_it] = std::minmax_element(final.target_lengths.begin(), final.target_lengths.end());
    std::cerr << "target length: base=" << base_len
              << ", target_faces=" << target_faces
              << ", bisection_iters=" << config_.target_bisection_iterations
              << ", pre_limiter_estimated_faces=" << pre_limiter_estimated_faces
              << ", post_limiter_estimated_faces=" << post_limiter_estimated_faces
              << ", final_estimated_faces=" << final.estimated_faces
              << ", scale=" << final.scale
              << ", max_log_slope=" << final_gradation.max_normalized_log_slope
              << ", curvature_fallback=" << principal.fallback_vertices << "/" << mesh.vertexCount()
              << ", min=" << *min_it
              << ", max=" << *max_it << "\n";
  }
}

double AdaIsoRemesher::targetAt(const Eigen::Vector3d& point) const {
  return target_field_.valueAt(point);
}

// Returns the target length for an edge defined by vertices a and b. If use_max is true, returns the maximum of the two vertex targets; otherwise, returns the minimum.
double AdaIsoRemesher::edgeTarget(const Mesh& mesh, int a, int b, bool use_max) const {
  const double ta = targetAt(mesh.vertices[static_cast<std::size_t>(a)]);
  const double tb = targetAt(mesh.vertices[static_cast<std::size_t>(b)]);
  return use_max ? std::max(ta, tb) : std::min(ta, tb);
}

EdgeRatioStats AdaIsoRemesher::edgeRatioStats(const Mesh& mesh) const {
  return computeEdgeRatioStats(
      mesh,
      target_field_,
      config_.collapseFactor(),
      config_.splitFactor());
}

int AdaIsoRemesher::splitLongEdges(Mesh& mesh) {
  int total = 0;
  for (int pass = 0; pass < config_.max_split_passes; ++pass) {
    // Collect candidate edges that are longer than the split threshold
    std::vector<std::tuple<double, int, int>> candidates;
    for (const auto& item : edgeFaces(mesh)) {
      const int a = item.first.first;
      const int b = item.first.second;
      const double length = (mesh.vertices[a] - mesh.vertices[b]).norm();
      const double target = edgeTarget(mesh, a, b, false);
      if (length > config_.splitFactor() * target) {
        candidates.emplace_back(length, a, b);
      }
    }
    if (candidates.empty()) {
      break;
    }
    std::sort(candidates.begin(), candidates.end(), [](const auto& lhs, const auto& rhs) {
      return std::get<0>(lhs) > std::get<0>(rhs);
    });


    std::vector<Edge> selected;
    selected.reserve(candidates.size());
    std::unordered_set<int> used_vertices;
    for (const auto& [old_length, a, b] : candidates) {
      (void)old_length;
      if (used_vertices.find(a) != used_vertices.end() || used_vertices.find(b) != used_vertices.end()) {
        continue;
      }
      const double length = (mesh.vertices[a] - mesh.vertices[b]).norm();
      const double target = edgeTarget(mesh, a, b, false);
      if (length <= config_.splitFactor() * target) {
        continue;
      }
      selected.push_back(makeEdge(a, b));
      used_vertices.insert(a);
      used_vertices.insert(b);
    }
    if (selected.empty()) {
      break;
    }
    splitEdgesBatch(mesh, selected);
    total += static_cast<int>(selected.size());
  }
  return total;
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

void splitEdgesBatch(Mesh& mesh, const std::vector<Edge>& edges) {
  if (edges.empty()) {
    return;
  }

  std::unordered_map<Edge, int, EdgeHash> midpoint_indices;
  midpoint_indices.reserve(edges.size());
  for (const auto& edge : edges) {
    const int a = edge.first;
    const int b = edge.second;
    const int m = static_cast<int>(mesh.vertices.size());
    mesh.vertices.push_back(
        0.5 * (mesh.vertices[static_cast<std::size_t>(a)] + mesh.vertices[static_cast<std::size_t>(b)]));
    midpoint_indices.emplace(edge, m);
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

int AdaIsoRemesher::collapseShortEdges(Mesh& mesh) {
  int total = 0;
  for (int pass = 0; pass < config_.max_collapse_passes; ++pass) {
    const auto boundary = boundaryVertices(mesh);
    const auto e2f = edgeFaces(mesh);
    const auto neigh = vertexNeighbors(mesh);
    const auto vf = vertexFaces(mesh);
    const auto feature_strengths = computeVertexFeatureStrengths(mesh);
    std::vector<std::tuple<double, double, int, int>> candidates;
    for (const auto& item : e2f) {
      const int a = item.first.first;
      const int b = item.first.second;
      if (boundary.find(a) != boundary.end() || boundary.find(b) != boundary.end()) {
        continue;
      }
      const double length = (mesh.vertices[a] - mesh.vertices[b]).norm();
      const double target = edgeTarget(mesh, a, b, config_.collapseUsesMaxReference());
      if (length < config_.collapseFactor() * target) {
        candidates.emplace_back(length / target, length, a, b);
      }
    }
    if (candidates.empty()) {
      break;
    }
    std::sort(candidates.begin(), candidates.end(), [](const auto& lhs, const auto& rhs) {
      if (std::get<0>(lhs) != std::get<0>(rhs)) {
        return std::get<0>(lhs) < std::get<0>(rhs);
      }
      return std::get<1>(lhs) < std::get<1>(rhs);
    });

    std::vector<Edge> selected;
    selected.reserve(candidates.size());
    std::unordered_set<int> blocked_vertices;
    for (const auto& [ratio, old_length, a, b] : candidates) {
      (void)ratio;
      (void)old_length;
      auto it = e2f.find(makeEdge(a, b));
      if (it != e2f.end() && it->second.size() > 2) {
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
        continue;
      }
      if (!collapseValid(mesh, a, b, neigh, e2f, vf, feature_strengths)) {
        continue;
      }
      selected.push_back(makeEdge(a, b));
      blocked_vertices.insert(footprint.begin(), footprint.end());
    }
    if (selected.empty()) {
      break;
    }
    total += collapseEdgesBatch(mesh, selected, feature_strengths);
  }
  return total;
}

bool AdaIsoRemesher::collapseValid(
    const Mesh& mesh,
    int a,
    int b,
    const std::vector<std::unordered_set<int>>& neigh,
    const EdgeFaceMap& e2f,
    const std::vector<std::vector<int>>& vf,
    const std::vector<double>& feature_strengths) const {
  const Eigen::Vector3d candidate = collapsePoint(mesh, a, b, feature_strengths);
  const auto incident_it = e2f.find(makeEdge(a, b));
  if (incident_it == e2f.end()) {
    return false;
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
    return false;
  }

  std::unordered_set<int> related = neigh[static_cast<std::size_t>(a)];
  related.insert(neigh[static_cast<std::size_t>(b)].begin(), neigh[static_cast<std::size_t>(b)].end());
  related.erase(a);
  related.erase(b);
  if (related.size() < 3) {
    return false;
  }
  for (int n : related) {
    const double length = (candidate - mesh.vertices[static_cast<std::size_t>(n)]).norm();
    const double target = std::min(targetAt(candidate), targetAt(mesh.vertices[static_cast<std::size_t>(n)]));
    if (length > config_.splitFactor() * target) {
      return false;
    }
  }

  if (config_.min_quality_after_collapse > 0.0) {
    const int keep = std::min(a, b);
    const int remove = std::max(a, b);

    std::unordered_set<int> affected;
    affected.insert(vf[static_cast<std::size_t>(a)].begin(), vf[static_cast<std::size_t>(a)].end());
    affected.insert(vf[static_cast<std::size_t>(b)].begin(), vf[static_cast<std::size_t>(b)].end());

    std::size_t surviving_faces = mesh.faceCount();
    double local_min_quality = std::numeric_limits<double>::infinity();
    bool checked_any = false;

    auto pointAt = [&](int v) -> Eigen::Vector3d {
      return v == keep ? candidate : mesh.vertices[static_cast<std::size_t>(v)];
    };

    for (int fi : affected) {
      Eigen::Vector3i face = mesh.faces[static_cast<std::size_t>(fi)];
      for (int k = 0; k < 3; ++k) {
        if (face[k] == remove) {
          face[k] = keep;
        }
      }
      if (face[0] == face[1] || face[1] == face[2] || face[2] == face[0]) {
        --surviving_faces;
        continue;
      }
      const Eigen::Vector3d p0 = pointAt(face[0]);
      const Eigen::Vector3d p1 = pointAt(face[1]);
      const Eigen::Vector3d p2 = pointAt(face[2]);
      if (triangleArea(p0, p1, p2) <= 1e-14) {
        --surviving_faces;
        continue;
      }
      local_min_quality = std::min(local_min_quality, triangleQuality(p0, p1, p2));
      checked_any = true;
    }

    if (surviving_faces == 0 ||
        (checked_any && local_min_quality < config_.min_quality_after_collapse)) {
      return false;
    }
  }

  return true;
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
    const std::vector<double>& feature_strengths) const {
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

      const double length = (mesh.vertices[static_cast<std::size_t>(c)] - mesh.vertices[static_cast<std::size_t>(d)]).norm();
      const double target = edgeTarget(mesh, c, d, false);
      if (length < config_.collapseFactor() * target || length > config_.splitFactor() * target) {
        continue;
      }
      if (!flipImprovesValence(valence, boundary, a, b, c, d)) {
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

      mesh.faces[static_cast<std::size_t>(f1)] = new1;
      mesh.faces[static_cast<std::size_t>(f2)] = new2;
      current_edges.erase(makeEdge(a, b));
      current_edges.insert(makeEdge(c, d));
      ++flipped;
      ++total;
    }
    if (flipped == 0) {
      break;
    }
  }
  return total;
}

bool AdaIsoRemesher::flipImprovesValence(
    const std::vector<int>& valence,
    const std::unordered_set<int>& boundary,
    int a,
    int b,
    int c,
    int d) const {
  auto target = [&](int v) {
    return boundary.find(v) == boundary.end() ? 6 : 4;
  };
  const int before = std::abs(valence[static_cast<std::size_t>(a)] - target(a)) +
                     std::abs(valence[static_cast<std::size_t>(b)] - target(b)) +
                     std::abs(valence[static_cast<std::size_t>(c)] - target(c)) +
                     std::abs(valence[static_cast<std::size_t>(d)] - target(d));
  const int after = std::abs(valence[static_cast<std::size_t>(a)] - 1 - target(a)) +
                    std::abs(valence[static_cast<std::size_t>(b)] - 1 - target(b)) +
                    std::abs(valence[static_cast<std::size_t>(c)] + 1 - target(c)) +
                    std::abs(valence[static_cast<std::size_t>(d)] + 1 - target(d));
  return after < before;
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

void AdaIsoRemesher::relocateVertices(Mesh& mesh, const Mesh& original_for_projection) {
  const auto classes = classifyVerticesForRelocation(mesh);
  tangentSmoothing(mesh, classes);

  if (!config_.enable_relocation_mls) {
    return;
  }

  const auto projected = mlsProject(mesh.vertices, original_for_projection);
  if (projected.size() != mesh.vertices.size()) {
    return;
  }

  for (std::size_t i = 0; i < mesh.vertices.size(); ++i) {
    if (classes[i] == VertexRelocationClass::Smooth && finiteVector(projected[i])) {
      mesh.vertices[i] = projected[i];
    }
  }
}

void AdaIsoRemesher::tangentSmoothing(Mesh& mesh, const std::vector<VertexRelocationClass>& classes) {
  const auto vf = vertexFaces(mesh);
  const auto normals = vertexNormals(mesh);
  const auto areas = faceAreas(mesh);
  auto new_vertices = mesh.vertices;

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
    if (finiteVector(candidate)) {
      new_vertices[i] = candidate;
    }
  }

  mesh.vertices = std::move(new_vertices);
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
