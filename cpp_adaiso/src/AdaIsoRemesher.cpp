#include "adaiso/AdaIsoRemesher.hpp"

#include "adaiso/Geometry.hpp"
#include "adaiso/Metrics.hpp"
#include "adaiso/detail/CsfSolver.hpp"

#include <Eigen/Geometry>

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace adaiso {
namespace {

constexpr double kEps = 1e-14;
constexpr double kPi = 3.141592653589793238462643383279502884;

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

Eigen::Vector3d safeFaceNormal(const std::vector<Eigen::Vector3d>& vertices, const Eigen::Vector3i& face) {
  return safeNormalize((vertices[face[1]] - vertices[face[0]]).cross(vertices[face[2]] - vertices[face[0]]));
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

std::size_t clampIndex(long long idx, std::size_t n) {
  if (n == 0) {
    return 0;
  }
  if (idx < 0) {
    return 0;
  }
  const auto max_index = static_cast<long long>(n - 1);
  if (idx > max_index) {
    return n - 1;
  }
  return static_cast<std::size_t>(idx);
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

}  // namespace

TargetLengthField::TargetLengthField(TargetLengthField&& other)
    : positions_(std::move(other.positions_)), lengths_(std::move(other.lengths_)) {
  if (!positions_.empty()) {
    index_.build(positions_);
  }
  other.clear();
}

TargetLengthField& TargetLengthField::operator=(TargetLengthField&& other) {
  if (this == &other) {
    return *this;
  }
  positions_ = std::move(other.positions_);
  lengths_ = std::move(other.lengths_);
  index_ = NearestNeighborIndex();
  if (!positions_.empty()) {
    index_.build(positions_);
  }
  other.clear();
  return *this;
}

void TargetLengthField::build(const std::vector<Eigen::Vector3d>& positions, const std::vector<double>& lengths) {
  if (positions.size() != lengths.size()) {
    throw std::runtime_error("target field positions and lengths have different sizes");
  }
  positions_ = positions;
  lengths_ = lengths;
  if (positions_.empty()) {
    index_ = NearestNeighborIndex();
    return;
  }
  index_.build(positions_);
}

void TargetLengthField::clear() {
  positions_.clear();
  lengths_.clear();
  index_ = NearestNeighborIndex();
}

bool TargetLengthField::empty() const {
  return index_.empty() || lengths_.empty();
}

double TargetLengthField::valueAt(
    const Eigen::Vector3d& point,
    TargetLookupMode mode,
    int k,
    double sigma_factor) const {
  if (empty()) {
    throw std::runtime_error("target lengths have not been initialized");
  }
  if (k <= 0) {
    throw std::runtime_error("target k must be positive");
  }

  const std::size_t nearest_idx = index_.nearest(point);
  if (mode == TargetLookupMode::Nearest || k == 1 || lengths_.size() == 1) {
    return lengths_[nearest_idx];
  }
  if (sigma_factor <= 0.0 || !std::isfinite(sigma_factor)) {
    throw std::runtime_error("target gaussian sigma factor must be positive");
  }

  const std::size_t query_k = std::min<std::size_t>(static_cast<std::size_t>(k), lengths_.size());
  const auto knn = index_.nearestK(point, query_k);
  if (knn.empty()) {
    return lengths_[nearest_idx];
  }
  if (knn.size() == 1) {
    return lengths_[knn.front().first];
  }

  std::vector<double> dists;
  dists.reserve(knn.size());
  for (const auto& item : knn) {
    dists.push_back(std::sqrt(std::max(0.0, item.second)));
  }

  const double dk = dists.back();
  const double dk_prev = dists[dists.size() - 2];
  const double raw_h = dk + std::max(0.0, dk - dk_prev);
  const double h = std::max(sigma_factor * std::max(raw_h, 1e-12), 1e-12);
  if (!std::isfinite(h) || h <= 0.0) {
    return lengths_[nearest_idx];
  }

  double sum_w = 0.0;
  double sum_l = 0.0;
  for (std::size_t i = 0; i < knn.size(); ++i) {
    const std::size_t idx = knn[i].first;
    if (idx >= lengths_.size()) {
      continue;
    }
    const double ratio = dists[i] / h;
    const double w = std::exp(-(ratio * ratio));
    if (!std::isfinite(w)) {
      continue;
    }
    sum_w += w;
    sum_l += w * lengths_[idx];
  }
  if (sum_w <= kEps) {
    return lengths_[nearest_idx];
  }
  const double value = sum_l / sum_w;
  return std::isfinite(value) ? value : lengths_[nearest_idx];
}

EdgeRatioStats computeEdgeRatioStats(
    const Mesh& mesh,
    const TargetLengthField& target_field,
    TargetLookupMode mode,
    int target_knn_k,
    double target_gaussian_sigma_factor,
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
        target_field.valueAt(mesh.vertices[static_cast<std::size_t>(a)], mode, target_knn_k, target_gaussian_sigma_factor),
        target_field.valueAt(mesh.vertices[static_cast<std::size_t>(b)], mode, target_knn_k, target_gaussian_sigma_factor));
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
  return variant == Variant::Paper ? 5.0 / 3.0 : 1.33;
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

TargetLookupMode parseTargetLookupMode(const std::string& value) {
  std::string lower = value;
  std::transform(lower.begin(), lower.end(), lower.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  if (lower == "nearest" || lower == "nn") {
    return TargetLookupMode::Nearest;
  }
  if (lower == "gaussian" || lower == "gaussian-knn" || lower == "gaussianknn") {
    return TargetLookupMode::GaussianKnn;
  }
  throw std::runtime_error("unknown target lookup mode: " + value);
}

std::string toString(TargetLookupMode mode) {
  return mode == TargetLookupMode::Nearest ? "nearest" : "gaussian";
}

AdaIsoRemesher::AdaIsoRemesher(AdaIsoConfig config) : config_(config) {}

Mesh AdaIsoRemesher::remesh(const Mesh& input) {
  if (config_.target_knn_k <= 0) {
    throw std::runtime_error("target k must be positive");
  }
  if (config_.target_gaussian_sigma_factor <= 0.0 || !std::isfinite(config_.target_gaussian_sigma_factor)) {
    throw std::runtime_error("target gaussian sigma factor must be positive");
  }
  if (config_.csf_smoothing_alpha <= 0.0 || !std::isfinite(config_.csf_smoothing_alpha)) {
    throw std::runtime_error("CSF smoothing alpha must be positive");
  }

  stats_ = {};
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
  stats_.iterations.reserve(static_cast<std::size_t>(std::max(0, config_.global_iterations)));

  for (int it = 0; it < config_.global_iterations; ++it) {
    if (config_.verbose) {
      std::cerr << "iteration " << (it + 1) << "/" << config_.global_iterations << "\n";
    }
    const int split_count = splitLongEdges(mesh);
    const int collapse_count = collapseShortEdges(mesh);
    mesh = removeDegenerateFaces(mesh);
    const int flip_count = flipEdges(mesh);
    tangentSmoothing(mesh);
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

  if (config_.project_to_surface) {
    const Mesh before_projection = mesh;
    const double before_qmin = minQuality(before_projection);
    Mesh projected = mesh;
    projected.vertices = mlsProject(mesh.vertices, original_for_projection);
    projected = removeDegenerateFaces(projected);
    const double after_qmin = minQuality(projected);
    if (projected.faceCount() > 0 && (before_qmin <= 0.0 || after_qmin + 1e-12 >= before_qmin)) {
      mesh = std::move(projected);
    } else if (config_.verbose) {
      std::cerr << "projection rejected: qmin would drop from " << before_qmin << " to " << after_qmin << "\n";
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

void AdaIsoRemesher::buildTargetLengths(const Mesh& mesh) {
  const double base_len = meanEdgeLength(mesh);
  const auto curvature = computeCsf(mesh);
  const auto multipliers = histogramMultipliers(curvature);
  std::vector<double> target_lengths(mesh.vertices.size(), 1.0);
  for (std::size_t i = 0; i < target_lengths.size(); ++i) {
    const double multiplier = i < multipliers.size() ? multipliers[i] : 1.0;
    target_lengths[i] = std::max(1e-12, base_len * config_.mesh_scale * multiplier);
  }
  target_field_.build(mesh.vertices, target_lengths);

  if (config_.verbose && !target_lengths.empty()) {
    const auto [min_it, max_it] = std::minmax_element(target_lengths.begin(), target_lengths.end());
    std::cerr << "target length: base=" << base_len << ", min=" << *min_it << ", max=" << *max_it << "\n";
  }
}

std::vector<double> AdaIsoRemesher::computeCsf(const Mesh& mesh) {
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
      const double angle = std::acos(dot);
      accum += angle;
      ++count;
    }
    curvature[i] = accum / static_cast<double>(count);
  }

  const auto solved = detail::solveScreenedCsf(mesh, curvature, config_.csf_smoothing_alpha);
  stats_.csf_assembly_time_ms = solved.stats.assembly_time_ms;
  stats_.csf_factorization_time_ms = solved.stats.factorization_time_ms;
  stats_.csf_solve_time_ms = solved.stats.solve_time_ms;
  stats_.csf_residual_abs = solved.stats.residual_abs;
  stats_.csf_residual_scaled = solved.stats.residual_scaled;
  stats_.csf_matrix_nonzeros = solved.stats.matrix_nonzeros;
  stats_.csf_solver_success = solved.stats.solver_success;
  return solved.values;
}

std::vector<double> AdaIsoRemesher::histogramMultipliers(const std::vector<double>& curvature) const {
  std::vector<double> result(curvature.size(), 1.0);
  if (curvature.empty()) {
    return result;
  }

  const auto [min_it, max_it] = std::minmax_element(curvature.begin(), curvature.end());
  const double min_value = *min_it;
  const double max_value = *max_it;
  if (std::abs(max_value - min_value) <= 1e-14) {
    return result;
  }

  const int bins = std::max(1, config_.histogram_bins);
  const double width = (max_value - min_value) / static_cast<double>(bins);
  std::vector<int> counts(static_cast<std::size_t>(bins), 0);
  for (double value : curvature) {
    int bin = static_cast<int>(std::floor((value - min_value) / width));
    bin = std::clamp(bin, 0, bins - 1);
    counts[static_cast<std::size_t>(bin)] += 1;
  }

  const int max_count = *std::max_element(counts.begin(), counts.end());
  std::vector<int> max_bins;
  for (int i = 0; i < bins; ++i) {
    if (counts[static_cast<std::size_t>(i)] == max_count) {
      max_bins.push_back(i);
    }
  }
  const int base_bin = max_bins[max_bins.size() / 2];
  const double base = min_value + (static_cast<double>(base_bin) + 0.5) * width;

  std::vector<double> sorted = curvature;
  std::sort(sorted.begin(), sorted.end());
  std::array<double, 4> cuts{};

  if (config_.variant == Variant::Code) {
    auto lower = std::lower_bound(sorted.begin(), sorted.end(), base);
    long long idx = static_cast<long long>(std::distance(sorted.begin(), lower));
    idx = std::max<long long>(1, std::min<long long>(idx, static_cast<long long>(sorted.size() - 1)));
    const long long unit_right = std::max<long long>(1, (static_cast<long long>(sorted.size()) - idx) / 5);
    cuts[0] = sorted[clampIndex((idx / 5) * 2, sorted.size())];
    cuts[1] = sorted[clampIndex((idx / 5) * 4, sorted.size())];
    cuts[2] = sorted[clampIndex(idx + unit_right, sorted.size())];
    cuts[3] = sorted[clampIndex(idx + unit_right * 3, sorted.size())];
  } else {
    std::vector<double> left;
    std::vector<double> right;
    for (double value : curvature) {
      if (value <= base) {
        left.push_back(value);
      } else {
        right.push_back(value);
      }
    }
    std::sort(left.begin(), left.end());
    std::sort(right.begin(), right.end());
    if (left.empty()) {
      left.assign(sorted.begin(), sorted.begin() + std::max<std::size_t>(1, sorted.size() / 2));
    }
    if (right.empty()) {
      right.assign(sorted.begin() + static_cast<std::ptrdiff_t>(sorted.size() / 2), sorted.end());
    }
    cuts[0] = quantileOfSorted(left, 2.0 / 5.0);
    cuts[1] = quantileOfSorted(left, 4.0 / 5.0);
    cuts[2] = quantileOfSorted(right, 1.0 / 5.0);
    cuts[3] = quantileOfSorted(right, 3.0 / 5.0);
  }

  for (std::size_t i = 0; i < curvature.size(); ++i) {
    const double c = curvature[i];
    if (c < cuts[0]) {
      result[i] = config_.multipliers[0];
    } else if (c < cuts[1]) {
      result[i] = config_.multipliers[1];
    } else if (c < cuts[2]) {
      result[i] = config_.multipliers[2];
    } else if (c < cuts[3]) {
      result[i] = config_.multipliers[3];
    } else {
      result[i] = config_.multipliers[4];
    }
  }
  return result;
}

double AdaIsoRemesher::targetAt(const Eigen::Vector3d& point) const {
  return target_field_.valueAt(
      point,
      config_.target_lookup_mode,
      config_.target_knn_k,
      config_.target_gaussian_sigma_factor);
}

double AdaIsoRemesher::edgeTarget(const Mesh& mesh, int a, int b, bool use_max) const {
  const double ta = targetAt(mesh.vertices[static_cast<std::size_t>(a)]);
  const double tb = targetAt(mesh.vertices[static_cast<std::size_t>(b)]);
  return use_max ? std::max(ta, tb) : std::min(ta, tb);
}

EdgeRatioStats AdaIsoRemesher::edgeRatioStats(const Mesh& mesh) const {
  return computeEdgeRatioStats(
      mesh,
      target_field_,
      config_.target_lookup_mode,
      config_.target_knn_k,
      config_.target_gaussian_sigma_factor,
      config_.collapseFactor(),
      config_.splitFactor());
}

int AdaIsoRemesher::splitLongEdges(Mesh& mesh) {
  int total = 0;
  for (int pass = 0; pass < config_.max_split_passes; ++pass) {
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
      if (!collapseValid(mesh, a, b, neigh, e2f, vf)) {
        continue;
      }
      selected.push_back(makeEdge(a, b));
      blocked_vertices.insert(footprint.begin(), footprint.end());
    }
    if (selected.empty()) {
      break;
    }
    total += collapseEdgesBatch(mesh, selected);
  }
  return total;
}

bool AdaIsoRemesher::collapseValid(
    const Mesh& mesh,
    int a,
    int b,
    const std::vector<std::unordered_set<int>>& neigh,
    const EdgeFaceMap& e2f,
    const std::vector<std::vector<int>>& vf) const {
  const Eigen::Vector3d midpoint = 0.5 * (mesh.vertices[static_cast<std::size_t>(a)] + mesh.vertices[static_cast<std::size_t>(b)]);
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
    const double length = (midpoint - mesh.vertices[static_cast<std::size_t>(n)]).norm();
    const double target = std::min(targetAt(midpoint), targetAt(mesh.vertices[static_cast<std::size_t>(n)]));
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
      return v == keep ? midpoint : mesh.vertices[static_cast<std::size_t>(v)];
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

int AdaIsoRemesher::collapseEdgesBatch(Mesh& mesh, const std::vector<Edge>& selected) const {
  struct CollapseOp {
    int keep = 0;
    int remove = 0;
    Eigen::Vector3d midpoint = Eigen::Vector3d::Zero();
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
    ops.push_back({keep, remove,
        0.5 * (mesh.vertices[static_cast<std::size_t>(a)] + mesh.vertices[static_cast<std::size_t>(b)])});
  }

  if (ops.empty()) {
    return 0;
  }

  for (const auto& op : ops) {
    mesh.vertices[static_cast<std::size_t>(op.keep)] = op.midpoint;
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

void AdaIsoRemesher::tangentSmoothing(Mesh& mesh) {
  const auto boundary = boundaryVertices(mesh);
  const auto vf = vertexFaces(mesh);
  const auto normals = vertexNormals(mesh);
  const auto areas = faceAreas(mesh);
  auto new_vertices = mesh.vertices;

  for (std::size_t i = 0; i < mesh.vertices.size(); ++i) {
    if (boundary.find(static_cast<int>(i)) != boundary.end() || vf[i].empty()) {
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
