#include "adaiso/AdaIsoRemesher.hpp"

#include "adaiso/Geometry.hpp"
#include "adaiso/Metrics.hpp"
#include "adaiso/detail/FeatureClassification.hpp"
#include "adaiso/detail/LocalOperators.hpp"
#include "adaiso/detail/RawCurvature.hpp"
#include "adaiso/detail/ScreenedLaplacian.hpp"
#include "adaiso/detail/TargetFaceSizing.hpp"

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
constexpr double kCsfSmoothingAlpha = 2.0;
constexpr double kMaxFlipDihedralRadians = 45.0 * kPi / 180.0;

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

detail::CurvatureSignalMode toDetailCurvatureMode(AdaIsoConfig::CurvatureSignal mode) {
  switch (mode) {
    case AdaIsoConfig::CurvatureSignal::LogDimensionless:
      return detail::CurvatureSignalMode::LogDimensionless;
    case AdaIsoConfig::CurvatureSignal::Dimensionless:
      return detail::CurvatureSignalMode::Dimensionless;
    case AdaIsoConfig::CurvatureSignal::Physical:
      return detail::CurvatureSignalMode::Physical;
  }
  return detail::CurvatureSignalMode::LogDimensionless;
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

void writeTargetFieldDiagnostics(
    const Mesh& mesh,
    const std::vector<double>& physical_curvature,
    const std::vector<double>& raw_curvature,
    const std::vector<double>& csf_curvature,
    const detail::TargetFaceSizingResult& sizing,
    double base_length,
    const std::filesystem::path& directory) {
  if (directory.empty()) {
    return;
  }
  mesh.validate();
  if (physical_curvature.size() != mesh.vertexCount() ||
      raw_curvature.size() != mesh.vertexCount() ||
      csf_curvature.size() != mesh.vertexCount() ||
      sizing.capped_curvature.size() != mesh.vertexCount() ||
      sizing.target_lengths.size() != mesh.vertexCount()) {
    throw std::runtime_error("target-field diagnostic size mismatch");
  }
  if (!std::isfinite(base_length) || base_length <= 0.0) {
    throw std::runtime_error("target-field diagnostics require positive base length");
  }

  std::filesystem::create_directories(directory);

  {
    std::ofstream out(directory / "vertices.csv");
    if (!out) {
      throw std::runtime_error("failed to write target-field vertices.csv");
    }
    out << std::setprecision(17);
    out << "vertex_id,x,y,z,H_physical,raw_kappa,csf_kappa,capped_kappa,target_h,h_over_base,is_capped\n";
    for (std::size_t i = 0; i < mesh.vertexCount(); ++i) {
      const Eigen::Vector3d& p = mesh.vertices[i];
      const double h = sizing.target_lengths[i];
      const bool capped = csf_curvature[i] > sizing.capped_curvature[i] + 1e-14;
      out << i << ','
          << p.x() << ',' << p.y() << ',' << p.z() << ','
          << physical_curvature[i] << ','
          << raw_curvature[i] << ','
          << csf_curvature[i] << ','
          << sizing.capped_curvature[i] << ','
          << h << ','
          << (h / base_length) << ','
          << (capped ? 1 : 0) << '\n';
    }
  }

  const auto edges = uniqueEdges(mesh);
  {
    std::ofstream out(directory / "edges.csv");
    if (!out) {
      throw std::runtime_error("failed to write target-field edges.csv");
    }
    out << std::setprecision(17);
    out << "i,j,edge_length,target_h_min,target_ratio,h_slope\n";
    for (const auto& edge : edges) {
      const int a = edge.first;
      const int b = edge.second;
      const double length = (mesh.vertices[static_cast<std::size_t>(a)] -
                             mesh.vertices[static_cast<std::size_t>(b)])
                                .norm();
      const double ha = sizing.target_lengths[static_cast<std::size_t>(a)];
      const double hb = sizing.target_lengths[static_cast<std::size_t>(b)];
      const double target = std::min(ha, hb);
      const double ratio = length / target;
      const double slope = std::abs(ha - hb) / std::max(length, kEps);
      out << a << ',' << b << ','
          << length << ','
          << target << ','
          << ratio << ','
          << slope << '\n';
    }
  }

  {
    std::ofstream out(directory / "faces.csv");
    if (!out) {
      throw std::runtime_error("failed to write target-field faces.csv");
    }
    out << std::setprecision(17);
    out << "face_id,mean_csf,mean_target_h,grad_h_norm,max_edge_target_ratio\n";
    for (std::size_t i = 0; i < mesh.faceCount(); ++i) {
      const Eigen::Vector3i& f = mesh.faces[i];
      const int a = f[0];
      const int b = f[1];
      const int c = f[2];
      const double ha = sizing.target_lengths[static_cast<std::size_t>(a)];
      const double hb = sizing.target_lengths[static_cast<std::size_t>(b)];
      const double hc = sizing.target_lengths[static_cast<std::size_t>(c)];
      const double mean_csf =
          (csf_curvature[static_cast<std::size_t>(a)] +
           csf_curvature[static_cast<std::size_t>(b)] +
           csf_curvature[static_cast<std::size_t>(c)]) /
          3.0;
      const double mean_h = (ha + hb + hc) / 3.0;
      auto edgeRatio = [&](int u, int v, double hu, double hv) {
        const double length = (mesh.vertices[static_cast<std::size_t>(u)] -
                               mesh.vertices[static_cast<std::size_t>(v)])
                                  .norm();
        return length / std::min(hu, hv);
      };
      auto edgeSlope = [&](int u, int v, double hu, double hv) {
        const double length = (mesh.vertices[static_cast<std::size_t>(u)] -
                               mesh.vertices[static_cast<std::size_t>(v)])
                                  .norm();
        return std::abs(hu - hv) / std::max(length, kEps);
      };
      const double max_ratio = std::max({
          edgeRatio(a, b, ha, hb),
          edgeRatio(b, c, hb, hc),
          edgeRatio(c, a, hc, ha)});
      const double grad_norm = std::max({
          edgeSlope(a, b, ha, hb),
          edgeSlope(b, c, hb, hc),
          edgeSlope(c, a, hc, ha)});
      out << i << ','
          << mean_csf << ','
          << mean_h << ','
          << grad_norm << ','
          << max_ratio << '\n';
    }
  }

  auto sortedCopy = [](const std::vector<double>& values) {
    std::vector<double> sorted = values;
    std::sort(sorted.begin(), sorted.end());
    return sorted;
  };
  const auto target_sorted = sortedCopy(sizing.target_lengths);
  const auto raw_sorted = sortedCopy(raw_curvature);
  const auto csf_sorted = sortedCopy(csf_curvature);
  const auto capped_sorted = sortedCopy(sizing.capped_curvature);

  std::ofstream out(directory / "field_stats.json");
  if (!out) {
    throw std::runtime_error("failed to write target-field field_stats.json");
  }
  out << std::setprecision(17);
  out << "{\n";
  out << "  \"base_length\": " << base_length << ",\n";
  out << "  \"target_faces\": " << sizing.target_faces << ",\n";
  out << "  \"target_estimated_faces\": " << sizing.estimated_faces << ",\n";
  out << "  \"lambda\": " << sizing.lambda << ",\n";
  out << "  \"h_min\": " << sizing.h_min << ",\n";
  out << "  \"h_max\": " << sizing.h_max << ",\n";
  out << "  \"target_h_min\": " << quantileOfSorted(target_sorted, 0.0) << ",\n";
  out << "  \"target_h_p05\": " << quantileOfSorted(target_sorted, 0.05) << ",\n";
  out << "  \"target_h_p50\": " << quantileOfSorted(target_sorted, 0.5) << ",\n";
  out << "  \"target_h_p95\": " << quantileOfSorted(target_sorted, 0.95) << ",\n";
  out << "  \"target_h_max\": " << quantileOfSorted(target_sorted, 1.0) << ",\n";
  out << "  \"raw_kappa_p95\": " << quantileOfSorted(raw_sorted, 0.95) << ",\n";
  out << "  \"raw_kappa_max\": " << quantileOfSorted(raw_sorted, 1.0) << ",\n";
  out << "  \"csf_kappa_p95\": " << quantileOfSorted(csf_sorted, 0.95) << ",\n";
  out << "  \"csf_kappa_max\": " << quantileOfSorted(csf_sorted, 1.0) << ",\n";
  out << "  \"capped_kappa_p95\": " << quantileOfSorted(capped_sorted, 0.95) << ",\n";
  out << "  \"capped_kappa_max\": " << quantileOfSorted(capped_sorted, 1.0) << ",\n";
  out << "  \"curvature_cap_value\": " << sizing.curvature_cap_value << ",\n";
  out << "  \"curvature_cap_changed_rate\": " << sizing.curvature_cap_changed_rate << ",\n";
  out << "  \"vertex_count\": " << mesh.vertexCount() << ",\n";
  out << "  \"edge_count\": " << edges.size() << ",\n";
  out << "  \"face_count\": " << mesh.faceCount() << "\n";
  out << "}\n";
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
  return 1.65;
}

double AdaIsoConfig::collapseFactor() const {
  return 4.0 / 5.0;
}

AdaIsoRemesher::AdaIsoRemesher(AdaIsoConfig config) : config_(config) {}

Mesh AdaIsoRemesher::remesh(const Mesh& input) {
  stats_ = {};
  Mesh mesh = removeDegenerateFaces(input);
  const std::size_t effective_target_faces =
      config_.target_faces > 0 ? config_.target_faces : mesh.faceCount();

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

  for (int it = 0; it < config_.global_iterations; ++it) {
    if (config_.verbose) {
      std::cerr << "iteration " << (it + 1) << "/" << config_.global_iterations << "\n";
    }
    const int split_count = splitLongEdges(mesh);
    const int collapse_count = collapseShortEdges(mesh);
    mesh = removeDegenerateFaces(mesh);
    const int flip_count = flipEdges(mesh);
    stats_.split_total += split_count;
    stats_.collapse_total += collapse_count;
    stats_.flip_total += flip_count;
    relocateVertices(mesh);
    mesh = removeDegenerateFaces(mesh);
    const MeshSummary iteration_summary = summarizeMesh(mesh);
    const EdgeRatioStats ratio_stats = edgeRatioStats(mesh);
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
  const auto curvature_fields = detail::computeRobustMeanCurvatureFields(
      mesh,
      toDetailCurvatureMode(config_.curvature_signal));
  const auto& raw_curvature = curvature_fields.raw_signal;
  const auto csf_curvature =
      config_.use_csf ? solveCsf(mesh, raw_curvature) : raw_curvature;
  if (!config_.use_csf) {
    stats_.csf_solver_success = true;
  }
  const auto sizing = detail::targetFaceCurvatureLengths(
      mesh,
      csf_curvature,
      target_faces,
      base_len,
      config_.curvature_cap_quantile);

  target_field_.build(mesh, sizing.target_lengths);
  writeTargetFieldDiagnostics(
      mesh,
      curvature_fields.physical_mean_curvature,
      raw_curvature,
      csf_curvature,
      sizing,
      base_len,
      config_.export_target_field_dir);
  stats_.target_lambda = sizing.lambda;
  stats_.target_estimated_faces = sizing.estimated_faces;
  stats_.target_h_min = sizing.h_min;
  stats_.target_h_max = sizing.h_max;
  stats_.curvature_cap_value = sizing.curvature_cap_value;
  stats_.curvature_cap_changed_rate = sizing.curvature_cap_changed_rate;

  if (!sizing.target_lengths.empty()) {
    const auto [min_it, max_it] =
        std::minmax_element(sizing.target_lengths.begin(), sizing.target_lengths.end());
    stats_.target_observed_h_min = *min_it;
    stats_.target_observed_h_max = *max_it;
  }

  if (config_.verbose && !sizing.target_lengths.empty()) {
    const auto [min_it, max_it] =
        std::minmax_element(sizing.target_lengths.begin(), sizing.target_lengths.end());
    std::cerr << "target length: base=" << base_len
              << ", target_faces=" << target_faces
              << ", estimated_faces=" << sizing.estimated_faces
              << ", lambda=" << sizing.lambda
              << ", min=" << *min_it
              << ", max=" << *max_it << "\n";
  }
}

std::vector<double> AdaIsoRemesher::solveCsf(
    const Mesh& mesh,
    const std::vector<double>& raw_curvature) {
  const auto solved = detail::solveScreenedScalarField(
      mesh,
      raw_curvature,
      kCsfSmoothingAlpha);
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
    const auto feature_strengths = detail::computeVertexFeatureStrengths(mesh);
    std::vector<std::tuple<double, double, int, int>> candidates;
    for (const auto& item : e2f) {
      const int a = item.first.first;
      const int b = item.first.second;
      if (boundary.find(a) != boundary.end() || boundary.find(b) != boundary.end()) {
        continue;
      }
      const double length = (mesh.vertices[a] - mesh.vertices[b]).norm();
      const double target = edgeTarget(mesh, a, b, false);
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
  return detail::featureAwareCollapsePoint(mesh, a, b, feature_strengths);
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

      if (faceNormalAngle(mesh.vertices, tri1, tri2) > kMaxFlipDihedralRadians) {
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
      if (!detail::flipImprovesValence(valence, boundary, a, b, c, d)) {
        continue;
      }

      const Eigen::Vector3d ref_normal = safeFaceNormal(mesh.vertices, tri1) + safeFaceNormal(mesh.vertices, tri2);
      Eigen::Vector3i new1 = orientTriangle(Eigen::Vector3i(c, d, b), mesh.vertices, ref_normal);
      Eigen::Vector3i new2 = orientTriangle(Eigen::Vector3i(d, c, a), mesh.vertices, ref_normal);
      if (faceNormalAngle(mesh.vertices, new1, new2) > kMaxFlipDihedralRadians) {
        continue;
      }
      if (maxInnerAngle(mesh.vertices[new1[0]], mesh.vertices[new1[1]], mesh.vertices[new1[2]]) > 0.5 * kPi ||
          maxInnerAngle(mesh.vertices[new2[0]], mesh.vertices[new2[1]], mesh.vertices[new2[2]]) > 0.5 * kPi) {
        continue;
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

void AdaIsoRemesher::relocateVertices(Mesh& mesh) {
  const auto classes = detail::classifyVerticesForRelocation(mesh);
  tangentSmoothing(mesh, classes);
}

void AdaIsoRemesher::tangentSmoothing(Mesh& mesh, const std::vector<detail::VertexRelocationClass>& classes) {
  const auto vf = vertexFaces(mesh);
  const auto normals = vertexNormals(mesh);
  const auto areas = faceAreas(mesh);
  auto new_vertices = mesh.vertices;

  for (std::size_t i = 0; i < mesh.vertices.size(); ++i) {
    if (i >= classes.size() || classes[i] != detail::VertexRelocationClass::Smooth || vf[i].empty()) {
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

}  // namespace adaiso
