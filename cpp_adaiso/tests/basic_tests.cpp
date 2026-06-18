#include "adaiso/AdaIsoRemesher.hpp"
#include "adaiso/Geometry.hpp"
#include "adaiso/IO.hpp"
#include "adaiso/Mesh.hpp"
#include "adaiso/Metrics.hpp"
#include "adaiso/detail/CsfSolver.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <numeric>

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

adaiso::Mesh makeSingleTriangle() {
  adaiso::Mesh mesh;
  mesh.vertices = {
      {0.0, 0.0, 0.0},
      {1.0, 0.0, 0.0},
      {0.0, 1.0, 0.0},
  };
  mesh.faces = {{0, 1, 2}};
  return mesh;
}

adaiso::Mesh makeDisconnectedTriangles() {
  adaiso::Mesh mesh;
  mesh.vertices = {
      {0.0, 0.0, 0.0},
      {1.0, 0.0, 0.0},
      {0.0, 1.0, 0.0},
      {3.0, 0.0, 0.0},
      {4.0, 0.0, 0.0},
      {3.0, 1.0, 0.2},
  };
  mesh.faces = {{0, 1, 2}, {3, 4, 5}};
  return mesh;
}

adaiso::Mesh makePlanePatch() {
  adaiso::Mesh mesh;
  mesh.vertices = {
      {-1.0, -1.0, 0.0},
      {1.0, -1.0, 0.0},
      {1.0, 1.0, 0.0},
      {-1.0, 1.0, 0.0},
  };
  mesh.faces = {{0, 1, 2}, {0, 2, 3}};
  return mesh;
}

adaiso::Mesh makeUvSphere(double radius, int rings, int segments) {
  adaiso::Mesh mesh;
  const double pi = std::acos(-1.0);
  mesh.vertices.push_back({0.0, 0.0, radius});
  for (int i = 1; i < rings; ++i) {
    const double theta = pi * static_cast<double>(i) / static_cast<double>(rings);
    const double z = radius * std::cos(theta);
    const double r = radius * std::sin(theta);
    for (int j = 0; j < segments; ++j) {
      const double phi = 2.0 * pi * static_cast<double>(j) / static_cast<double>(segments);
      mesh.vertices.push_back({r * std::cos(phi), r * std::sin(phi), z});
    }
  }
  const int bottom = static_cast<int>(mesh.vertices.size());
  mesh.vertices.push_back({0.0, 0.0, -radius});

  auto idx = [&](int ring, int segment) {
    return 1 + (ring - 1) * segments + ((segment + segments) % segments);
  };

  for (int j = 0; j < segments; ++j) {
    mesh.faces.push_back({0, idx(1, j), idx(1, j + 1)});
  }
  for (int i = 1; i + 1 < rings; ++i) {
    for (int j = 0; j < segments; ++j) {
      const int a = idx(i, j);
      const int b = idx(i, j + 1);
      const int c = idx(i + 1, j);
      const int d = idx(i + 1, j + 1);
      mesh.faces.push_back({a, c, b});
      mesh.faces.push_back({b, c, d});
    }
  }
  for (int j = 0; j < segments; ++j) {
    mesh.faces.push_back({bottom, idx(rings - 1, j + 1), idx(rings - 1, j)});
  }
  return mesh;
}

adaiso::Mesh makeCylinder(double radius, double height, int rows, int segments) {
  adaiso::Mesh mesh;
  const double pi = std::acos(-1.0);
  for (int i = 0; i < rows; ++i) {
    const double z = -0.5 * height + height * static_cast<double>(i) / static_cast<double>(rows - 1);
    for (int j = 0; j < segments; ++j) {
      const double phi = 2.0 * pi * static_cast<double>(j) / static_cast<double>(segments);
      mesh.vertices.push_back({radius * std::cos(phi), radius * std::sin(phi), z});
    }
  }
  const int bottom_center = static_cast<int>(mesh.vertices.size());
  mesh.vertices.push_back({0.0, 0.0, -0.5 * height});
  const int top_center = static_cast<int>(mesh.vertices.size());
  mesh.vertices.push_back({0.0, 0.0, 0.5 * height});

  auto idx = [&](int row, int segment) {
    return row * segments + ((segment + segments) % segments);
  };
  for (int i = 0; i + 1 < rows; ++i) {
    for (int j = 0; j < segments; ++j) {
      const int a = idx(i, j);
      const int b = idx(i, j + 1);
      const int c = idx(i + 1, j);
      const int d = idx(i + 1, j + 1);
      mesh.faces.push_back({a, c, b});
      mesh.faces.push_back({b, c, d});
    }
  }
  for (int j = 0; j < segments; ++j) {
    mesh.faces.push_back({bottom_center, idx(0, j + 1), idx(0, j)});
    mesh.faces.push_back({top_center, idx(rows - 1, j), idx(rows - 1, j + 1)});
  }
  return mesh;
}

adaiso::Mesh reverseFaces(adaiso::Mesh mesh) {
  for (auto& face : mesh.faces) {
    std::swap(face[1], face[2]);
  }
  return mesh;
}

adaiso::detail::CsfAdjacency makeChainAdjacency(int vertices) {
  adaiso::detail::CsfAdjacency adjacency(static_cast<std::size_t>(vertices));
  for (int i = 0; i + 1 < vertices; ++i) {
    adjacency[static_cast<std::size_t>(i)].push_back({i + 1, 1.0});
    adjacency[static_cast<std::size_t>(i + 1)].push_back({i, 1.0});
  }
  return adjacency;
}

void assertNear(double actual, double expected, double eps = 1e-10) {
  assert(std::abs(actual - expected) <= eps);
}

void assertFiniteValues(const std::vector<double>& values) {
  for (double value : values) {
    assert(std::isfinite(value));
  }
}

double averagePositive(const std::vector<double>& values) {
  double sum = 0.0;
  int count = 0;
  for (double value : values) {
    if (std::isfinite(value) && value > 0.0) {
      sum += value;
      ++count;
    }
  }
  return count == 0 ? 0.0 : sum / static_cast<double>(count);
}

void assertSymmetricPositive(const adaiso::detail::CsfAdjacency& adjacency) {
  for (std::size_t i = 0; i < adjacency.size(); ++i) {
    for (const auto& neighbor : adjacency[i]) {
      assert(std::isfinite(neighbor.weight));
      assert(neighbor.weight > 0.0);
      bool found_reverse = false;
      for (const auto& reverse : adjacency[static_cast<std::size_t>(neighbor.vertex)]) {
        if (reverse.vertex == static_cast<int>(i)) {
          assertNear(reverse.weight, neighbor.weight, 1e-12);
          found_reverse = true;
          break;
        }
      }
      assert(found_reverse);
    }
  }
}

void assertHarmonicResidual(
    const adaiso::detail::CsfAdjacency& adjacency,
    const std::vector<double>& values,
    const std::vector<int>& free_vertices,
    double eps = 1e-10) {
  for (int i : free_vertices) {
    double residual = 0.0;
    for (const auto& neighbor : adjacency[static_cast<std::size_t>(i)]) {
      residual += neighbor.weight * (values[static_cast<std::size_t>(i)] - values[static_cast<std::size_t>(neighbor.vertex)]);
    }
    assert(std::abs(residual) <= eps);
  }
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

void testCurvatureAndSizingField() {
  const adaiso::Mesh plane = makePlanePatch();
  const auto plane_curvature = adaiso::maximumAbsolutePrincipalCurvatures(plane);
  assert(plane_curvature.size() == plane.vertexCount());
  for (double value : plane_curvature) {
    assert(std::abs(value) < 1e-12);
  }
  const auto plane_sizing = adaiso::curvatureSizingField(plane, 0.01, 0.1, 2.0);
  for (double value : plane_sizing) {
    assertNear(value, 2.0);
  }

  const adaiso::Mesh sphere = makeUvSphere(1.0, 12, 24);
  const auto sphere_curvature = adaiso::maximumAbsolutePrincipalCurvatures(sphere);
  assertFiniteValues(sphere_curvature);
  const double sphere_avg = averagePositive(sphere_curvature);
  assert(sphere_avg > 0.5 && sphere_avg < 1.8);

  const adaiso::Mesh flipped_sphere = reverseFaces(sphere);
  const double flipped_avg = averagePositive(adaiso::maximumAbsolutePrincipalCurvatures(flipped_sphere));
  assert(std::abs(sphere_avg - flipped_avg) < 1e-10);

  const adaiso::Mesh cylinder = makeCylinder(1.0, 3.0, 7, 32);
  const auto cylinder_curvature = adaiso::maximumAbsolutePrincipalCurvatures(cylinder);
  assertFiniteValues(cylinder_curvature);
  double cylinder_mid_sum = 0.0;
  int cylinder_mid_count = 0;
  const int middle_row = 3;
  for (int j = 0; j < 32; ++j) {
    cylinder_mid_sum += cylinder_curvature[static_cast<std::size_t>(middle_row * 32 + j)];
    ++cylinder_mid_count;
  }
  const double cylinder_mid_avg = cylinder_mid_sum / static_cast<double>(cylinder_mid_count);
  assert(cylinder_mid_avg > 0.4 && cylinder_mid_avg < 1.8);

  const auto small_sphere_sizing = adaiso::curvatureSizingField(sphere, 0.01, 0.05, 2.0);
  const auto large_sphere_sizing = adaiso::curvatureSizingField(makeUvSphere(2.0, 12, 24), 0.01, 0.05, 2.0);
  const double small_sphere_avg = std::accumulate(
      small_sphere_sizing.begin(),
      small_sphere_sizing.end(),
      0.0) / static_cast<double>(small_sphere_sizing.size());
  const double large_sphere_avg = std::accumulate(
      large_sphere_sizing.begin(),
      large_sphere_sizing.end(),
      0.0) / static_cast<double>(large_sphere_sizing.size());
  assert(small_sphere_avg < large_sphere_avg);

  const auto tight_sizing = adaiso::curvatureSizingField(sphere, 0.005, 0.05, 2.0);
  const auto loose_sizing = adaiso::curvatureSizingField(sphere, 0.02, 0.05, 2.0);
  const double tight_avg = std::accumulate(tight_sizing.begin(), tight_sizing.end(), 0.0) /
                           static_cast<double>(tight_sizing.size());
  const double loose_avg = std::accumulate(loose_sizing.begin(), loose_sizing.end(), 0.0) /
                           static_cast<double>(loose_sizing.size());
  assert(tight_avg < loose_avg);
  for (double value : tight_sizing) {
    assert(std::isfinite(value));
    assert(value >= 0.05 - 1e-12 && value <= 2.0 + 1e-12);
  }
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

void testCsfKnownChainSolution() {
  const auto adjacency = makeChainAdjacency(4);
  const auto result = adaiso::detail::solveDirichletCsf(adjacency, {0.0, 0.0, 0.0, 1.0});
  assert(result.stats.solver_success);
  assert(result.stats.fixed_vertices == 2);
  assert(result.stats.free_vertices == 2);
  assert(result.stats.matrix_nonzeros == 4);
  assert(result.stats.residual_scaled < adaiso::detail::kResidualThreshold);
  assertNear(result.values[0], 0.0);
  assertNear(result.values[1], 1.0 / 3.0);
  assertNear(result.values[2], 2.0 / 3.0);
  assertNear(result.values[3], 1.0);
  assertHarmonicResidual(adjacency, result.values, {1, 2});
  for (double value : result.values) {
    assert(value >= -1e-12 && value <= 1.0 + 1e-12);
  }
}

void testCsfDisconnectedAndConstantComponents() {
  auto adjacency = makeChainAdjacency(9);
  adjacency[3].clear();
  adjacency[4].clear();
  adjacency[3].push_back({2, 1.0});
  adjacency[4].push_back({5, 1.0});
  adjacency[2].clear();
  adjacency[2].push_back({1, 1.0});
  adjacency[2].push_back({3, 1.0});
  adjacency[5].clear();
  adjacency[5].push_back({4, 1.0});
  adjacency[5].push_back({6, 1.0});
  adjacency[6].clear();
  adjacency[6].push_back({5, 1.0});
  adjacency[6].push_back({7, 1.0});
  adjacency[7].clear();
  adjacency[7].push_back({6, 1.0});
  adjacency[8].clear();

  const auto result = adaiso::detail::solveDirichletCsf(adjacency, {0.0, 0.0, 0.0, 1.0, 2.0, 2.0, 2.0, 5.0, 7.0});
  assert(result.stats.solver_success);
  assert(result.stats.fixed_vertices == 5);
  assert(result.stats.free_vertices == 4);
  assertNear(result.values[1], 1.0 / 3.0);
  assertNear(result.values[2], 2.0 / 3.0);
  assertNear(result.values[5], 3.0);
  assertNear(result.values[6], 4.0);
  assertNear(result.values[8], 7.0);
  assert(result.values[4] != result.values[5]);
  assertHarmonicResidual(adjacency, result.values, {1, 2, 5, 6});

  const auto constant = adaiso::detail::solveDirichletCsf(makeChainAdjacency(4), {2.0, 2.0, 2.0, 2.0});
  assert(constant.stats.solver_success);
  assert(constant.stats.free_vertices == 0);
  assert(constant.stats.fixed_vertices == 4);
  for (double value : constant.values) {
    assertNear(value, 2.0);
  }
}

void testCsfVariantWeightGraphs() {
  const adaiso::Mesh mesh = makePyramid();
  const auto paper = adaiso::detail::buildCsfWeightGraph(mesh, false);
  const auto code = adaiso::detail::buildCsfWeightGraph(mesh, true);
  assertSymmetricPositive(paper);
  assertSymmetricPositive(code);

  const std::vector<double> initial = {0.0, 0.2, 0.4, 0.6, 1.0};
  const auto paper_result = adaiso::detail::solveDirichletCsf(paper, initial);
  const auto code_result = adaiso::detail::solveDirichletCsf(code, initial);
  assert(paper_result.stats.solver_success);
  assert(code_result.stats.solver_success);
  assertFiniteValues(paper_result.values);
  assertFiniteValues(code_result.values);
  assert(paper_result.stats.residual_scaled < adaiso::detail::kResidualThreshold);
  assert(code_result.stats.residual_scaled < adaiso::detail::kResidualThreshold);
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
  assert(remesher.stats().sizing.epsilon > 0.0);
  assert(remesher.stats().sizing.min > 0.0);
  assert(remesher.stats().sizing.max >= remesher.stats().sizing.min);
  assert(remesher.stats().iterations.size() == 2);
  for (const auto& iter : remesher.stats().iterations) {
    assert(iter.vertices > 0);
    assert(iter.faces > 0);
    assert(std::isfinite(iter.target_error_avg));
    assert(std::isfinite(iter.geometry_error_avg));
  }
}

void testRemeshEdgeCases() {
  auto run = [](const adaiso::Mesh& input) {
    adaiso::AdaIsoConfig config;
    config.global_iterations = 0;
    config.project_to_surface = false;
    config.verbose = false;
    adaiso::AdaIsoRemesher remesher(config);
    const adaiso::Mesh output = remesher.remesh(input);
    output.validate();
    assert(std::isfinite(remesher.stats().sizing.epsilon));
    assert(std::isfinite(remesher.stats().sizing.min));
    assert(std::isfinite(remesher.stats().sizing.max));
  };

  run(makeSingleTriangle());
  run(makeDisconnectedTriangles());

  adaiso::Mesh empty;
  run(empty);

  adaiso::Mesh degenerate;
  degenerate.vertices = {{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {2.0, 0.0, 0.0}};
  degenerate.faces = {{0, 1, 2}};
  run(degenerate);

  adaiso::Mesh short_edge;
  short_edge.vertices = {{0.0, 0.0, 0.0}, {1e-15, 0.0, 0.0}, {0.0, 1.0, 0.0}};
  short_edge.faces = {{0, 1, 2}};
  run(short_edge);

}

}  // namespace

int main() {
  testGeometryAndMetrics();
  testCurvatureAndSizingField();
  testCleanupAndAdjacency();
  testObjRoundtrip();
  testOffLoad();
  testCsfKnownChainSolution();
  testCsfDisconnectedAndConstantComponents();
  testCsfVariantWeightGraphs();
  testRemeshSmoke();
  testRemeshEdgeCases();
  return 0;
}
