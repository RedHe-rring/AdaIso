#include "adaiso/AdaIsoRemesher.hpp"

#include "adaiso/Geometry.hpp"
#include "adaiso/KDTree.hpp"

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
#include <vector>

namespace adaiso {
namespace {

constexpr double kEps = 1e-14;
constexpr double kPi = 3.141592653589793238462643383279502884;

struct ResolvedSizingParameters {
  double epsilon = 0.0;
  double min_length = 0.0;
  double max_length = 0.0;
};

struct FaceKey {
  int a = 0;
  int b = 0;
  int c = 0;

  bool operator==(const FaceKey& other) const {
    return a == other.a && b == other.b && c == other.c;
  }
};

struct FaceKeyHash {
  std::size_t operator()(const FaceKey& key) const noexcept {
    std::size_t h = 1469598103934665603ull;
    for (int v : {key.a, key.b, key.c}) {
      h ^= static_cast<std::size_t>(v + 0x9e3779b9);
      h *= 1099511628211ull;
    }
    return h;
  }
};

FaceKey makeFaceKey(const Eigen::Vector3i& face) {
  std::array<int, 3> values = {face[0], face[1], face[2]};
  std::sort(values.begin(), values.end());
  return {values[0], values[1], values[2]};
}

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

void splitEdgesBatch(Mesh& mesh, std::vector<double>& target_lengths, const std::vector<Edge>& edges) {
  if (edges.empty()) {
    return;
  }
  if (target_lengths.size() != mesh.vertices.size()) {
    throw std::runtime_error("target length count does not match mesh vertices before split");
  }

  std::unordered_map<Edge, int, EdgeHash> midpoint_indices;
  midpoint_indices.reserve(edges.size());
  for (const auto& edge : edges) {
    const int a = edge.first;
    const int b = edge.second;
    const int m = static_cast<int>(mesh.vertices.size());
    mesh.vertices.push_back(
        0.5 * (mesh.vertices[static_cast<std::size_t>(a)] + mesh.vertices[static_cast<std::size_t>(b)]));
    target_lengths.push_back(
        0.5 * (target_lengths[static_cast<std::size_t>(a)] + target_lengths[static_cast<std::size_t>(b)]));
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

Mesh removeDegenerateFacesWithTargets(
    const Mesh& mesh,
    const std::vector<double>& target_lengths,
    std::vector<double>& remapped_targets) {
  if (target_lengths.size() != mesh.vertices.size()) {
    throw std::runtime_error("target length count does not match mesh vertices during cleanup");
  }

  Mesh filtered;
  filtered.vertices = mesh.vertices;
  std::unordered_set<FaceKey, FaceKeyHash> seen;

  for (const auto& face : mesh.faces) {
    if (face[0] == face[1] || face[1] == face[2] || face[2] == face[0]) {
      continue;
    }
    if (face[0] < 0 || face[1] < 0 || face[2] < 0) {
      continue;
    }
    if (static_cast<std::size_t>(face[0]) >= mesh.vertices.size() ||
        static_cast<std::size_t>(face[1]) >= mesh.vertices.size() ||
        static_cast<std::size_t>(face[2]) >= mesh.vertices.size()) {
      continue;
    }
    if (triangleArea(mesh.vertices[face[0]], mesh.vertices[face[1]], mesh.vertices[face[2]]) <= kEps) {
      continue;
    }

    const FaceKey key = makeFaceKey(face);
    if (!seen.insert(key).second) {
      continue;
    }
    filtered.faces.push_back(face);
  }

  if (filtered.faces.empty()) {
    remapped_targets = target_lengths;
    return filtered;
  }

  std::vector<int> remap(filtered.vertices.size(), -1);
  std::vector<int> used;
  used.reserve(filtered.vertices.size());
  for (const auto& face : filtered.faces) {
    for (int k = 0; k < 3; ++k) {
      const int idx = face[k];
      if (remap[static_cast<std::size_t>(idx)] == -1) {
        remap[static_cast<std::size_t>(idx)] = static_cast<int>(used.size());
        used.push_back(idx);
      }
    }
  }

  Mesh out;
  out.vertices.reserve(used.size());
  remapped_targets.clear();
  remapped_targets.reserve(used.size());
  for (int idx : used) {
    out.vertices.push_back(filtered.vertices[static_cast<std::size_t>(idx)]);
    remapped_targets.push_back(target_lengths[static_cast<std::size_t>(idx)]);
  }

  out.faces.reserve(filtered.faces.size());
  for (const auto& face : filtered.faces) {
    out.faces.emplace_back(
        remap[static_cast<std::size_t>(face[0])],
        remap[static_cast<std::size_t>(face[1])],
        remap[static_cast<std::size_t>(face[2])]);
  }
  return out;
}

double minQuality(const Mesh& mesh) {
  const auto qualities = triangleQualities(mesh);
  if (qualities.empty()) {
    return 0.0;
  }
  return *std::min_element(qualities.begin(), qualities.end());
}

double minInnerAngleDegrees(const Mesh& mesh) {
  if (mesh.faces.empty()) {
    return 0.0;
  }
  double best = std::numeric_limits<double>::infinity();
  auto angle = [](const Eigen::Vector3d& p, const Eigen::Vector3d& q, const Eigen::Vector3d& r) {
    const Eigen::Vector3d u = q - p;
    const Eigen::Vector3d v = r - p;
    const double denom = u.norm() * v.norm();
    if (denom <= kEps) {
      return 0.0;
    }
    return std::acos(std::clamp(u.dot(v) / denom, -1.0, 1.0));
  };
  for (const auto& face : mesh.faces) {
    best = std::min(best, angle(mesh.vertices[face[0]], mesh.vertices[face[1]], mesh.vertices[face[2]]));
    best = std::min(best, angle(mesh.vertices[face[1]], mesh.vertices[face[2]], mesh.vertices[face[0]]));
    best = std::min(best, angle(mesh.vertices[face[2]], mesh.vertices[face[0]], mesh.vertices[face[1]]));
  }
  if (!std::isfinite(best)) {
    return 0.0;
  }
  return best * 180.0 / kPi;
}

ResolvedSizingParameters resolveSizingParameters(const AdaIsoConfig& config, const Mesh& mesh) {
  double base = meanEdgeLength(mesh);
  if (!std::isfinite(base) || base <= kEps) {
    base = 1.0;
  }
  ResolvedSizingParameters params;
  params.epsilon = config.approximationTolerance > 0.0 ? config.approximationTolerance : 0.05 * base;
  params.min_length = config.minTargetLength > 0.0 ? config.minTargetLength : 0.8 * base;
  params.max_length = config.maxTargetLength > 0.0 ? config.maxTargetLength : (4.0 / 3.0) * base;
  if (!std::isfinite(params.epsilon) || !std::isfinite(params.min_length) || !std::isfinite(params.max_length) ||
      params.epsilon <= 0.0 || params.min_length <= 0.0 || params.max_length <= 0.0 ||
      params.min_length > params.max_length) {
    throw std::runtime_error("invalid resolved sizing parameters");
  }
  return params;
}

}  // namespace

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

AdaIsoRemesher::AdaIsoRemesher(AdaIsoConfig config) : config_(config) {}

Mesh AdaIsoRemesher::remesh(const Mesh& input) {
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

  buildSizingField(mesh);

  for (int it = 0; it < config_.global_iterations; ++it) {
    if (config_.verbose) {
      std::cerr << "iteration " << (it + 1) << "/" << config_.global_iterations << "\n";
    }
    const int split_count = splitLongEdges(mesh);
    const int collapse_count = collapseShortEdges(mesh);
    cleanupMesh(mesh);
    const int flip_count = flipEdges(mesh);
    tangentSmoothing(mesh);
    cleanupMesh(mesh);
    stats_.iterations.push_back(collectIterationStats(
        mesh,
        original_for_projection,
        it + 1,
        split_count,
        collapse_count,
        flip_count));
    if (config_.verbose) {
      const IterationStats& iter_stats = stats_.iterations.back();
      std::cerr << "  split=" << split_count << ", collapse=" << collapse_count << ", flip=" << flip_count
                << ", mesh=" << mesh.vertexCount() << "v/" << mesh.faceCount() << "f"
                << ", target_err_avg=" << iter_stats.target_error_avg
                << ", qmin=" << iter_stats.qmin
                << ", min_angle=" << iter_stats.min_angle_degrees
                << ", geom_err_avg=" << iter_stats.geometry_error_avg << "\n";
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

void AdaIsoRemesher::buildSizingField(const Mesh& mesh) {
  if (mesh.vertices.empty()) {
    target_lengths_.clear();
    stats_.sizing = {};
    return;
  }

  const ResolvedSizingParameters params = resolveSizingParameters(config_, mesh);
  target_lengths_ = curvatureSizingField(mesh, params.epsilon, params.min_length, params.max_length);
  if (target_lengths_.size() != mesh.vertices.size()) {
    throw std::runtime_error("sizing field size mismatch");
  }

  stats_.sizing.epsilon = params.epsilon;
  stats_.sizing.min_target_length = params.min_length;
  stats_.sizing.max_target_length = params.max_length;
  if (!target_lengths_.empty()) {
    const auto [min_it, max_it] = std::minmax_element(target_lengths_.begin(), target_lengths_.end());
    stats_.sizing.min = *min_it;
    stats_.sizing.max = *max_it;
    stats_.sizing.avg = std::accumulate(target_lengths_.begin(), target_lengths_.end(), 0.0) /
                        static_cast<double>(target_lengths_.size());
  }

  if (config_.verbose && !target_lengths_.empty()) {
    std::cerr << "sizing field: epsilon=" << params.epsilon
              << ", bounds=[" << params.min_length << ", " << params.max_length << "]"
              << ", min=" << stats_.sizing.min
              << ", max=" << stats_.sizing.max
              << ", avg=" << stats_.sizing.avg << "\n";
  }
}

void AdaIsoRemesher::cleanupMesh(Mesh& mesh) {
  if (target_lengths_.empty()) {
    mesh = removeDegenerateFaces(mesh);
    return;
  }
  std::vector<double> remapped_targets;
  mesh = removeDegenerateFacesWithTargets(mesh, target_lengths_, remapped_targets);
  target_lengths_ = std::move(remapped_targets);
}

double AdaIsoRemesher::targetAt(int vertex) const {
  if (vertex < 0 || static_cast<std::size_t>(vertex) >= target_lengths_.size()) {
    throw std::runtime_error("target length vertex index is out of range");
  }
  return target_lengths_[static_cast<std::size_t>(vertex)];
}

double AdaIsoRemesher::edgeTarget(const Mesh& mesh, int a, int b, bool use_max) const {
  (void)mesh;
  (void)use_max;
  const double ta = targetAt(a);
  const double tb = targetAt(b);
  return std::min(ta, tb);
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
    splitEdgesBatch(mesh, target_lengths_, selected);
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
      return std::get<0>(lhs) < std::get<0>(rhs);
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
    const double midpoint_target = 0.5 * (targetAt(a) + targetAt(b));
    const double target = std::min(midpoint_target, targetAt(n));
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

void AdaIsoRemesher::collapseOneEdge(Mesh& mesh, int a, int b) {
  const int keep = std::min(a, b);
  const int remove = std::max(a, b);
  mesh.vertices[static_cast<std::size_t>(keep)] =
      0.5 * (mesh.vertices[static_cast<std::size_t>(a)] + mesh.vertices[static_cast<std::size_t>(b)]);
  target_lengths_[static_cast<std::size_t>(keep)] =
      0.5 * (target_lengths_[static_cast<std::size_t>(a)] + target_lengths_[static_cast<std::size_t>(b)]);
  for (auto& face : mesh.faces) {
    for (int k = 0; k < 3; ++k) {
      if (face[k] == remove) {
        face[k] = keep;
      }
    }
  }
  cleanupMesh(mesh);
}

int AdaIsoRemesher::collapseEdgesBatch(Mesh& mesh, const std::vector<Edge>& edges) {
  if (edges.empty()) {
    return 0;
  }
  if (target_lengths_.size() != mesh.vertices.size()) {
    throw std::runtime_error("target length count does not match mesh vertices before collapse");
  }

  std::vector<int> redirect(mesh.vertices.size(), -1);
  int applied = 0;
  for (const auto& edge : edges) {
    const int a = edge.first;
    const int b = edge.second;
    if (a < 0 || b < 0 || static_cast<std::size_t>(a) >= mesh.vertices.size() ||
        static_cast<std::size_t>(b) >= mesh.vertices.size()) {
      continue;
    }
    const int keep = std::min(a, b);
    const int remove = std::max(a, b);
    if (redirect[static_cast<std::size_t>(keep)] != -1 ||
        redirect[static_cast<std::size_t>(remove)] != -1) {
      continue;
    }
    mesh.vertices[static_cast<std::size_t>(keep)] =
        0.5 * (mesh.vertices[static_cast<std::size_t>(a)] + mesh.vertices[static_cast<std::size_t>(b)]);
    target_lengths_[static_cast<std::size_t>(keep)] =
        0.5 * (target_lengths_[static_cast<std::size_t>(a)] + target_lengths_[static_cast<std::size_t>(b)]);
    redirect[static_cast<std::size_t>(remove)] = keep;
    ++applied;
  }

  if (applied == 0) {
    return 0;
  }

  for (auto& face : mesh.faces) {
    for (int k = 0; k < 3; ++k) {
      const int v = face[k];
      if (v >= 0 && static_cast<std::size_t>(v) < redirect.size() && redirect[static_cast<std::size_t>(v)] != -1) {
        face[k] = redirect[static_cast<std::size_t>(v)];
      }
    }
  }
  cleanupMesh(mesh);
  return applied;
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

IterationStats AdaIsoRemesher::collectIterationStats(
    const Mesh& mesh,
    const Mesh& original,
    int iteration,
    int split_count,
    int collapse_count,
    int flip_count) const {
  IterationStats result;
  result.iteration = iteration;
  result.split_count = split_count;
  result.collapse_count = collapse_count;
  result.flip_count = flip_count;
  result.vertices = mesh.vertexCount();
  result.faces = mesh.faceCount();

  const auto edges = uniqueEdges(mesh);
  double target_error_sum = 0.0;
  for (const auto& edge : edges) {
    const int a = edge.first;
    const int b = edge.second;
    if (static_cast<std::size_t>(a) >= mesh.vertices.size() || static_cast<std::size_t>(b) >= mesh.vertices.size()) {
      continue;
    }
    const double target = edgeTarget(mesh, a, b, false);
    if (target <= kEps) {
      continue;
    }
    const double length = (mesh.vertices[static_cast<std::size_t>(a)] - mesh.vertices[static_cast<std::size_t>(b)]).norm();
    const double error = std::abs(length - target) / target;
    if (std::isfinite(error)) {
      target_error_sum += error;
      result.target_error_max = std::max(result.target_error_max, error);
    }
  }
  if (!edges.empty()) {
    result.target_error_avg = target_error_sum / static_cast<double>(edges.size());
  }

  const auto qualities = triangleQualities(mesh);
  if (!qualities.empty()) {
    result.qavg = std::accumulate(qualities.begin(), qualities.end(), 0.0) / static_cast<double>(qualities.size());
    result.qmin = *std::min_element(qualities.begin(), qualities.end());
  }
  result.min_angle_degrees = minInnerAngleDegrees(mesh);

  if (!original.vertices.empty() && !mesh.vertices.empty()) {
    NearestNeighborIndex tree(original.vertices);
    double error_sum = 0.0;
    for (const auto& vertex : mesh.vertices) {
      const std::size_t idx = tree.nearest(vertex);
      const double error = (vertex - original.vertices[idx]).norm();
      if (std::isfinite(error)) {
        error_sum += error;
        result.geometry_error_max = std::max(result.geometry_error_max, error);
      }
    }
    result.geometry_error_avg = error_sum / static_cast<double>(mesh.vertices.size());
  }

  return result;
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

    double sum_w = 0.0;
    Eigen::Vector3d centroid = Eigen::Vector3d::Zero();
    for (int fi : vf[i]) {
      const auto& face = mesh.faces[static_cast<std::size_t>(fi)];
      const double face_area = areas[static_cast<std::size_t>(fi)];
      if (face_area <= kEps || !std::isfinite(face_area)) {
        continue;
      }
      const double face_target = (targetAt(face[0]) + targetAt(face[1]) + targetAt(face[2])) / 3.0;
      const double w = face_area * face_target;
      const Eigen::Vector3d barycenter =
          (mesh.vertices[face[0]] + mesh.vertices[face[1]] + mesh.vertices[face[2]]) / 3.0;
      sum_w += w;
      centroid += w * barycenter;
    }
    if (sum_w <= kEps) {
      continue;
    }
    centroid /= sum_w;

    const Eigen::Vector3d normal = normals[i];
    if (normal.norm() <= kEps) {
      continue;
    }
    const Eigen::Vector3d projected = centroid + normal * normal.dot(mesh.vertices[i] - centroid);
    const Eigen::Vector3d candidate = mesh.vertices[i] + config_.tangent_lambda * (projected - mesh.vertices[i]);
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
