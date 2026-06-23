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
#include <stdexcept>

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

adaiso::Mesh translatedMesh(adaiso::Mesh mesh, const Eigen::Vector3d& offset) {
  for (auto& vertex : mesh.vertices) {
    vertex += offset;
  }
  return mesh;
}

adaiso::Mesh makeTwoDisconnectedOctahedra() {
  adaiso::Mesh mesh;

  auto append = [&](double x_offset) {
    const int offset = static_cast<int>(mesh.vertices.size());
    mesh.vertices.insert(mesh.vertices.end(), {
        {x_offset + 0.0, 0.0, 1.0},
        {x_offset + 0.0, 0.0, -1.0},
        {x_offset + 1.0, 0.0, 0.0},
        {x_offset - 1.0, 0.0, 0.0},
        {x_offset + 0.0, 1.0, 0.0},
        {x_offset + 0.0, -1.0, 0.0},
    });
    const std::vector<Eigen::Vector3i> faces = {
        {offset + 0, offset + 2, offset + 4},
        {offset + 0, offset + 4, offset + 3},
        {offset + 0, offset + 3, offset + 5},
        {offset + 0, offset + 5, offset + 2},
        {offset + 1, offset + 4, offset + 2},
        {offset + 1, offset + 3, offset + 4},
        {offset + 1, offset + 5, offset + 3},
        {offset + 1, offset + 2, offset + 5},
    };
    mesh.faces.insert(mesh.faces.end(), faces.begin(), faces.end());
  };

  append(0.0);
  append(5.0);
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

void testPlyRoundtripAndWriteMeshDispatch() {
  const adaiso::Mesh mesh = makePyramid();
  const std::filesystem::path dir = std::filesystem::temp_directory_path() / "cpp_adaiso_tests";

  const std::filesystem::path ply_path = dir / "pyramid.ply";
  adaiso::writeAsciiPly(ply_path, mesh);
  const adaiso::Mesh loaded_ply = adaiso::loadAsciiPly(ply_path);
  loaded_ply.validate();
  assert(loaded_ply.vertexCount() == mesh.vertexCount());
  assert(loaded_ply.faceCount() == mesh.faceCount());

  const std::filesystem::path dispatch_path = dir / "pyramid_dispatch.ply";
  adaiso::writeMesh(dispatch_path, mesh);
  const adaiso::Mesh loaded_dispatch = adaiso::loadMesh(dispatch_path);
  loaded_dispatch.validate();
  assert(loaded_dispatch.vertexCount() == mesh.vertexCount());
  assert(loaded_dispatch.faceCount() == mesh.faceCount());
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
  assert(remesher.stats().csf_solver_success);
  assert(remesher.stats().csf_residual_scaled < adaiso::detail::kResidualThreshold);
  const auto& iterations = remesher.stats().iterations;
  assert(iterations.size() == 2);
  for (std::size_t i = 0; i < iterations.size(); ++i) {
    assert(iterations[i].iteration == static_cast<int>(i + 1));
    assert(iterations[i].split_count >= 0);
    assert(iterations[i].collapse_count >= 0);
    assert(iterations[i].flip_count >= 0);
    assert(iterations[i].vertices > 0);
    assert(iterations[i].faces > 0);
    assert(std::isfinite(iterations[i].qavg));
    assert(std::isfinite(iterations[i].qmin));
  }
}

void testBatchCollapsePassCollapsesMultipleComponents() {
  const adaiso::Mesh input = makeTwoDisconnectedOctahedra();
  input.validate();

  adaiso::AdaIsoConfig config;
  config.variant = adaiso::Variant::Code;
  config.mesh_scale = 10.0;
  config.global_iterations = 1;
  config.max_split_passes = 0;
  config.max_collapse_passes = 1;
  config.max_flip_passes = 0;
  config.preprocess = false;
  config.project_to_surface = false;
  config.min_quality_after_collapse = 0.0;
  config.verbose = false;

  adaiso::AdaIsoRemesher remesher(config);
  const adaiso::Mesh output = remesher.remesh(input);
  output.validate();
  assert(output.vertexCount() + 2 <= input.vertexCount());
  assert(output.faceCount() > 0);
  assert(remesher.stats().csf_solver_success);
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
    assert(remesher.stats().csf_solver_success);
    assert(std::isfinite(remesher.stats().csf_residual_abs));
    assert(std::isfinite(remesher.stats().csf_residual_scaled));
    assert(remesher.stats().iterations.empty());
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

void testAreaSampledChamferDistance() {
  const adaiso::Mesh mesh = makePyramid();
  const auto same = adaiso::areaSampledChamferDistance(mesh, mesh, 128);
  assert(same.samples_per_direction == 128);
  assert(same.output_to_gt_msd <= 1e-20);
  assert(same.gt_to_output_msd <= 1e-20);
  assert(same.chamfer_distance <= 1e-20);
  assert(same.symmetric_rmse <= 1e-10);

  const adaiso::Mesh shifted = translatedMesh(mesh, Eigen::Vector3d(4.0, 0.0, 0.0));
  const auto shifted_cd = adaiso::areaSampledChamferDistance(mesh, shifted, 128);
  assert(shifted_cd.chamfer_distance > 1.0);
  assert(std::isfinite(shifted_cd.symmetric_rmse));

  bool threw = false;
  try {
    (void)adaiso::areaSampledChamferDistance(mesh, mesh, 0);
  } catch (const std::runtime_error&) {
    threw = true;
  }
  assert(threw);

  threw = false;
  try {
    adaiso::Mesh empty;
    (void)adaiso::areaSampledChamferDistance(empty, mesh, 16);
  } catch (const std::runtime_error&) {
    threw = true;
  }
  assert(threw);

  threw = false;
  try {
    adaiso::Mesh degenerate;
    degenerate.vertices = {{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {2.0, 0.0, 0.0}};
    degenerate.faces = {{0, 1, 2}};
    (void)adaiso::areaSampledChamferDistance(mesh, degenerate, 16);
  } catch (const std::runtime_error&) {
    threw = true;
  }
  assert(threw);
}

}  // namespace

int main() {
  testGeometryAndMetrics();
  testCleanupAndAdjacency();
  testObjRoundtrip();
  testPlyRoundtripAndWriteMeshDispatch();
  testOffLoad();
  testCsfKnownChainSolution();
  testCsfDisconnectedAndConstantComponents();
  testCsfVariantWeightGraphs();
  testRemeshSmoke();
  testBatchCollapsePassCollapsesMultipleComponents();
  testRemeshEdgeCases();
  testAreaSampledChamferDistance();
  return 0;
}
