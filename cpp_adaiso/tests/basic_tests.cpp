#include "adaiso/AdaIsoRemesher.hpp"
#include "adaiso/Geometry.hpp"
#include "adaiso/IO.hpp"
#include "adaiso/Mesh.hpp"
#include "adaiso/Metrics.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <filesystem>
#include <fstream>

namespace {

adaiso::Mesh makePyramid() {
  adaiso::Mesh mesh;
  mesh.vertices = {
      {0.0, 0.0, 0.0},
      {2.0, 0.0, 0.0},
      {2.0, 2.0, 0.0},
      {0.0, 2.0, 0.0},
      {1.0, 1.0, 1.2},
  };
  mesh.faces = {
      {0, 1, 4},
      {1, 2, 4},
      {2, 3, 4},
      {3, 0, 4},
      {0, 3, 2},
      {0, 2, 1},
  };
  return mesh;
}

void testGeometryAndMetrics() {
  const adaiso::Mesh mesh = makePyramid();
  mesh.validate();
  assert(mesh.vertexCount() == 5);
  assert(mesh.faceCount() == 6);
  assert(!adaiso::uniqueEdges(mesh).empty());
  assert(adaiso::meanEdgeLength(mesh) > 0.0);
  const auto summary = adaiso::summarizeMesh(mesh);
  assert(summary.vertices == 5);
  assert(summary.faces == 6);
  assert(summary.qavg > 0.0);
  assert(summary.qmin > 0.0);
  assert(adaiso::boundaryVertices(mesh).empty());
}

void testCleanupAndAdjacency() {
  adaiso::Mesh mesh = makePyramid();
  mesh.faces.push_back({0, 0, 1});
  mesh.faces.push_back({0, 1, 4});
  const adaiso::Mesh clean = adaiso::removeDegenerateFaces(mesh);
  assert(clean.faceCount() == 6);
  const auto e2f = adaiso::edgeFaces(clean);
  const auto it = e2f.find(adaiso::makeEdge(0, 1));
  assert(it != e2f.end());
  assert(!it->second.empty());
}

void testObjRoundtrip() {
  const adaiso::Mesh mesh = makePyramid();
  const std::filesystem::path dir = std::filesystem::temp_directory_path() / "cpp_adaiso_tests";
  const std::filesystem::path path = dir / "pyramid.obj";
  adaiso::writeObj(path, mesh);
  const adaiso::Mesh loaded = adaiso::loadObj(path);
  assert(loaded.vertexCount() == mesh.vertexCount());
  assert(loaded.faceCount() == mesh.faceCount());
  loaded.validate();
}

void testOffLoad() {
  const std::filesystem::path dir = std::filesystem::temp_directory_path() / "cpp_adaiso_tests";
  std::filesystem::create_directories(dir);
  const std::filesystem::path path = dir / "quad.off";
  {
    std::ofstream out(path);
    out << "OFF\n";
    out << "4 1 0\n";
    out << "0 0 0\n";
    out << "1 0 0\n";
    out << "1 1 0\n";
    out << "0 1 0\n";
    out << "4 0 1 2 3\n";
  }
  const adaiso::Mesh loaded = adaiso::loadOff(path);
  loaded.validate();
  assert(loaded.vertexCount() == 4);
  assert(loaded.faceCount() == 2);
}

void testRemeshSmoke() {
  const adaiso::Mesh input = makePyramid();
  const auto input_summary = adaiso::summarizeMesh(input);
  adaiso::AdaIsoConfig config;
  config.variant = adaiso::Variant::Code;
  config.mesh_scale = 0.8;
  config.global_iterations = 2;
  config.preprocess = true;
  config.project_to_surface = false;
  config.verbose = false;

  adaiso::AdaIsoRemesher remesher(config);
  const adaiso::Mesh output = remesher.remesh(input);
  output.validate();
  assert(output.vertexCount() > 0);
  assert(output.faceCount() > 0);
  const auto output_summary = adaiso::summarizeMesh(output);
  assert(output_summary.qmin > 0.0);
  assert(output_summary.qmin + 1e-12 >= std::min(0.01, input_summary.qmin));
}

}  // namespace

int main() {
  testGeometryAndMetrics();
  testCleanupAndAdjacency();
  testObjRoundtrip();
  testOffLoad();
  testRemeshSmoke();
  return 0;
}
