#include "adaiso/Mesh.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <Eigen/Geometry>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

namespace adaiso {
namespace {

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

FaceKey makeFaceKey(const Eigen::Vector3i& f) {
  std::array<int, 3> values = {f[0], f[1], f[2]};
  std::sort(values.begin(), values.end());
  return {values[0], values[1], values[2]};
}

double doubleArea(
    const Eigen::Vector3d& a,
    const Eigen::Vector3d& b,
    const Eigen::Vector3d& c) {
  return (b - a).cross(c - a).norm();
}

}  // namespace

void Mesh::validate() const {
  for (const auto& face : faces) {
    for (int k = 0; k < 3; ++k) {
      if (face[k] < 0 || static_cast<std::size_t>(face[k]) >= vertices.size()) {
        throw std::runtime_error("mesh contains out-of-range face index");
      }
    }
  }
}

std::size_t EdgeHash::operator()(const Edge& e) const noexcept {
  const auto a = static_cast<std::size_t>(e.first);
  const auto b = static_cast<std::size_t>(e.second);
  return (a * 73856093u) ^ (b * 19349663u);
}

Edge makeEdge(int a, int b) {
  if (a > b) {
    std::swap(a, b);
  }
  return {a, b};
}

Mesh compactVertices(const Mesh& mesh) {
  Mesh out;
  if (mesh.faces.empty()) {
    out.vertices = mesh.vertices;
    return out;
  }

  std::vector<int> remap(mesh.vertices.size(), -1);
  std::vector<int> used;
  used.reserve(mesh.vertices.size());
  for (const auto& face : mesh.faces) {
    for (int k = 0; k < 3; ++k) {
      const int idx = face[k];
      if (remap[idx] == -1) {
        remap[idx] = static_cast<int>(used.size());
        used.push_back(idx);
      }
    }
  }

  out.vertices.reserve(used.size());
  for (int idx : used) {
    out.vertices.push_back(mesh.vertices[idx]);
  }

  out.faces.reserve(mesh.faces.size());
  for (const auto& face : mesh.faces) {
    out.faces.emplace_back(remap[face[0]], remap[face[1]], remap[face[2]]);
  }
  return out;
}

Mesh removeDegenerateFaces(const Mesh& mesh, double area_eps) {
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
    if (doubleArea(mesh.vertices[face[0]], mesh.vertices[face[1]], mesh.vertices[face[2]]) <= area_eps) {
      continue;
    }

    const FaceKey key = makeFaceKey(face);
    if (seen.find(key) != seen.end()) {
      continue;
    }
    seen.insert(key);
    filtered.faces.push_back(face);
  }

  return compactVertices(filtered);
}

std::vector<Edge> uniqueEdges(const Mesh& mesh) {
  std::unordered_set<Edge, EdgeHash> seen;
  std::vector<Edge> edges;
  edges.reserve(mesh.faces.size() * 3);

  for (const auto& face : mesh.faces) {
    for (const auto& edge : {makeEdge(face[0], face[1]), makeEdge(face[1], face[2]), makeEdge(face[2], face[0])}) {
      if (seen.insert(edge).second) {
        edges.push_back(edge);
      }
    }
  }
  return edges;
}

}  // namespace adaiso
