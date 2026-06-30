#include "adaiso/AdaIsoRemesher.hpp"
#include "adaiso/Geometry.hpp"
#include "adaiso/IO.hpp"
#include "adaiso/Mesh.hpp"
#include "adaiso/Metrics.hpp"
#include "adaiso/detail/CsfSolver.hpp"
#include "adaiso/detail/TargetLengthMapping.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>

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

adaiso::Mesh makeSliverTriangle() {
  adaiso::Mesh mesh;
  mesh.vertices = {
      {0.0, 0.0, 0.0},
      {1.0, 0.0, 0.0},
      {1e-6, 1e-4, 0.0},
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

std::vector<double> vertexMasses(const adaiso::Mesh& mesh) {
  std::vector<double> masses(mesh.vertices.size(), 0.0);
  for (const auto& face : mesh.faces) {
    const double area = adaiso::triangleArea(mesh.vertices[face[0]], mesh.vertices[face[1]], mesh.vertices[face[2]]);
    const double share = area / 3.0;
    masses[static_cast<std::size_t>(face[0])] += share;
    masses[static_cast<std::size_t>(face[1])] += share;
    masses[static_cast<std::size_t>(face[2])] += share;
  }
  return masses;
}

void assertScreenedResidual(
    const adaiso::Mesh& mesh,
    const std::vector<double>& raw,
    const std::vector<double>& values,
    double alpha,
    double eps = 1e-10) {
  const auto adjacency = adaiso::detail::buildCsfWeightGraph(mesh);
  const auto masses = vertexMasses(mesh);
  const double h = adaiso::meanEdgeLength(mesh);
  const double lambda = alpha * h * h;
  for (std::size_t i = 0; i < values.size(); ++i) {
    double laplacian = 0.0;
    for (const auto& neighbor : adjacency[i]) {
      laplacian += neighbor.weight * (values[i] - values[static_cast<std::size_t>(neighbor.vertex)]);
    }
    const double residual = masses[i] * (values[i] - raw[i]) + lambda * laplacian;
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

void testCsfScreenedKnownTriangleSolution() {
  const adaiso::Mesh mesh = makeSingleTriangle();
  const std::vector<double> raw = {0.0, 0.0, 1.0};
  const double alpha = 1.0;
  const auto result = adaiso::detail::solveScreenedCsf(mesh, raw, alpha);
  assert(result.stats.solver_success);
  assert(result.stats.matrix_nonzeros == 9);
  assert(result.stats.residual_scaled < adaiso::detail::kResidualThreshold);
  assertFiniteValues(result.values);
  assertNear(result.values[0], 0.30699714, 1e-8);
  assertNear(result.values[1], 0.24419278, 1e-8);
  assertNear(result.values[2], 0.44881008, 1e-8);
  assertScreenedResidual(mesh, raw, result.values, alpha);
}

void testCsfScreenedConstantField() {
  const adaiso::Mesh mesh = makePyramid();
  const std::vector<double> raw(mesh.vertexCount(), 2.0);
  const auto result = adaiso::detail::solveScreenedCsf(mesh, raw, 0.5);
  assert(result.stats.solver_success);
  assertFiniteValues(result.values);
  for (double value : result.values) {
    assertNear(value, 2.0, 1e-10);
  }
  assertScreenedResidual(mesh, raw, result.values, 0.5);
}

void testCsfScreenedWeightGraphAndStrictValidation() {
  const adaiso::Mesh mesh = makePyramid();
  const auto graph = adaiso::detail::buildCsfWeightGraph(mesh);
  assertSymmetricPositive(graph);

  const std::vector<double> raw = {0.0, 0.2, 0.4, 0.6, 1.0};
  const auto result = adaiso::detail::solveScreenedCsf(mesh, raw, 0.5);
  assert(result.stats.solver_success);
  assertFiniteValues(result.values);
  assert(result.stats.residual_scaled < adaiso::detail::kResidualThreshold);

  bool threw = false;
  try {
    (void)adaiso::detail::solveScreenedCsf(mesh, raw, 0.0);
  } catch (const std::runtime_error&) {
    threw = true;
  }
  assert(threw);

  threw = false;
  try {
    adaiso::Mesh empty;
    (void)adaiso::detail::solveScreenedCsf(empty, {}, 0.5);
  } catch (const std::runtime_error&) {
    threw = true;
  }
  assert(threw);

  threw = false;
  try {
    adaiso::Mesh point_cloud;
    point_cloud.vertices = {{0.0, 0.0, 0.0}};
    (void)adaiso::detail::solveScreenedCsf(point_cloud, {0.0}, 0.5);
  } catch (const std::runtime_error&) {
    threw = true;
  }
  assert(threw);

  threw = false;
  try {
    adaiso::Mesh degenerate;
    degenerate.vertices = {{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {2.0, 0.0, 0.0}};
    degenerate.faces = {{0, 1, 2}};
    (void)adaiso::detail::solveScreenedCsf(degenerate, {0.0, 0.0, 0.0}, 0.5);
  } catch (const std::runtime_error&) {
    threw = true;
  }
  assert(threw);
}

void testTargetLengthFieldQueries() {
  adaiso::TargetLengthField field;
  const adaiso::Mesh triangle = makeSingleTriangle();
  field.build(triangle, {std::log(1.0), std::log(4.0), std::log(16.0)});

  assertNear(field.valueAt(triangle.vertices[0]), 1.0);
  assertNear(field.valueAt(0.5 * (triangle.vertices[0] + triangle.vertices[1])), 2.0);
  assertNear(field.valueAt({0.25, 0.25, 0.0}), std::sqrt(8.0));
  assertNear(field.valueAt({0.25, 0.25, 2.0}), std::sqrt(8.0));

  adaiso::Mesh square;
  square.vertices = {
      {0.0, 0.0, 0.0},
      {1.0, 0.0, 0.0},
      {0.0, 1.0, 0.0},
      {1.0, 1.0, 0.0},
  };
  square.faces = {{0, 1, 2}, {1, 3, 2}};
  field.build(square, {std::log(1.0), std::log(4.0), std::log(9.0), std::log(16.0)});
  assertNear(field.valueAt({0.5, 0.5, 0.0}), 6.0);
}

void testContinuousTargetLengthMapping() {
  const auto lengths = adaiso::detail::continuousTargetLogLengths({0.0, 1.0, 2.0}, 2.0, 1.0, 0.6, 1.8, 1.0);
  assert(lengths.size() == 3);
  assertNear(std::exp(lengths[0]), 3.6);
  assertNear(std::exp(lengths[1]), std::sqrt(3.6 * 1.2));
  assertNear(std::exp(lengths[2]), 1.2);

  const auto tied = adaiso::detail::continuousTargetLogLengths({0.0, 1.0, 1.0, 3.0}, 2.0, 1.0, 0.6, 1.8, 1.0);
  assert(tied.size() == 4);
  assertNear(tied[1], tied[2]);
  assert(tied[0] > tied[1]);
  assert(tied[1] > tied[3]);

  const auto constant = adaiso::detail::continuousTargetLogLengths({5.0, 5.0, 5.0}, 2.0, 1.5, 0.6, 1.8, 1.0);
  assert(constant.size() == 3);
  for (double value : constant) {
    assertNear(std::exp(value), 3.0);
  }

  const auto monotonic = adaiso::detail::continuousTargetLogLengths({0.0, 2.0, 4.0, 8.0}, 1.0, 1.0, 0.6, 1.8, 2.0);
  for (std::size_t i = 1; i < monotonic.size(); ++i) {
    assert(monotonic[i - 1] >= monotonic[i]);
  }
}

void testTargetLogGradationProjection() {
  adaiso::Mesh mesh;
  mesh.vertices = {
      {0.0, 0.0, 0.0},
      {1.0, 0.0, 0.0},
      {0.0, 1.0, 0.0},
      {1.0, 1.0, 0.0},
  };
  mesh.faces = {{0, 1, 2}, {1, 3, 2}};
  std::vector<double> logs = {std::log(1.0), std::log(100.0), std::log(100.0), std::log(100.0)};
  const double lo = std::log(1.0);
  const double hi = std::log(100.0);
  const double ratio = 2.0;
  adaiso::detail::projectTargetLogGradation(mesh, logs, lo, hi, ratio, 8);

  for (double value : logs) {
    assert(value + 1e-12 >= lo);
    assert(value <= hi + 1e-12);
  }
  for (const auto& [a, b] : adaiso::uniqueEdges(mesh)) {
    const double diff = std::abs(logs[static_cast<std::size_t>(a)] - logs[static_cast<std::size_t>(b)]);
    assert(diff <= std::log(ratio) + 1e-12);
  }
}

void testEdgeRatioStats() {
  adaiso::Mesh mesh = makeSingleTriangle();
  adaiso::TargetLengthField field;
  field.build(mesh, {std::log(1.0), std::log(1.0), std::log(1.0)});

  const auto stats = adaiso::computeEdgeRatioStats(
      mesh,
      field,
      0.8,
      1.33);

  const double sqrt2 = std::sqrt(2.0);
  const double expected_mean = (2.0 + sqrt2) / 3.0;
  const double expected_variance =
      ((1.0 - expected_mean) * (1.0 - expected_mean) * 2.0 +
          (sqrt2 - expected_mean) * (sqrt2 - expected_mean)) /
      3.0;
  assertNear(stats.mean_ratio, expected_mean);
  assertNear(stats.std_ratio, std::sqrt(expected_variance));
  assertNear(stats.p05_ratio, 1.0);
  assertNear(stats.p95_ratio, 0.1 + 0.9 * sqrt2);
  assertNear(stats.violation_rate, 1.0 / 3.0);
}

void testCurvatureModeParsing() {
  assert(adaiso::parseCurvatureMode("normal-angle") == adaiso::CurvatureMode::NormalAngle);
  assert(adaiso::parseCurvatureMode("ROBUST-MEAN") == adaiso::CurvatureMode::RobustMean);
  assert(adaiso::toString(adaiso::CurvatureMode::NormalAngle) == "normal-angle");
  assert(adaiso::toString(adaiso::CurvatureMode::RobustMean) == "robust-mean");

  bool threw = false;
  try {
    (void)adaiso::parseCurvatureMode("gaussian");
  } catch (const std::runtime_error&) {
    threw = true;
  }
  assert(threw);
}

void testRobustMeanCurvatureRemeshInitialization() {
  auto run = [](const adaiso::Mesh& input) {
    adaiso::AdaIsoConfig config;
    config.curvature_mode = adaiso::CurvatureMode::RobustMean;
    config.global_iterations = 0;
    config.project_to_surface = false;
    config.verbose = false;

    adaiso::AdaIsoRemesher remesher(config);
    const adaiso::Mesh output = remesher.remesh(input);
    output.validate();
    assert(output.vertexCount() > 0);
    assert(output.faceCount() > 0);
    assert(remesher.stats().csf_solver_success);
    assert(std::isfinite(remesher.stats().csf_residual_abs));
    assert(std::isfinite(remesher.stats().csf_residual_scaled));
  };

  run(makeSingleTriangle());
  run(makeSliverTriangle());
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
    assert(std::isfinite(iterations[i].mean_ratio));
    assert(std::isfinite(iterations[i].std_ratio));
    assert(std::isfinite(iterations[i].p05_ratio));
    assert(std::isfinite(iterations[i].p95_ratio));
    assert(std::isfinite(iterations[i].violation_rate));
    assert(iterations[i].violation_rate >= 0.0 && iterations[i].violation_rate <= 1.0);
  }
}

void testTargetFieldDiagnosticExport() {
  const std::filesystem::path dir =
      std::filesystem::temp_directory_path() / "adaiso_target_field_diagnostic_export";
  std::filesystem::remove_all(dir);

  adaiso::AdaIsoConfig config;
  config.global_iterations = 0;
  config.project_to_surface = false;
  config.verbose = false;
  config.target_field_export_dir = dir.string();

  adaiso::AdaIsoRemesher remesher(config);
  const adaiso::Mesh output = remesher.remesh(makePyramid());
  output.validate();
  assert(remesher.stats().csf_solver_success);

  const std::vector<std::filesystem::path> expected = {
      dir / "vertices.csv",
      dir / "edges.csv",
      dir / "faces.csv",
      dir / "edge_ratio_histogram.csv",
      dir / "field_stats.json",
      dir / "raw_curvature_heat.ply",
      dir / "csf_curvature_heat.ply",
      dir / "raw_target_h_heat.ply",
      dir / "smoothed_target_h_heat.ply",
      dir / "final_target_h_heat.ply",
      dir / "target_delta_log_heat.ply",
      dir / "edge_ratio_hotspot.ply",
      dir / "violation_mask.ply",
      dir / "face_log_gradient_heat.ply",
  };
  for (const auto& path : expected) {
    assert(std::filesystem::exists(path));
    assert(std::filesystem::file_size(path) > 0);
  }

  {
    std::ifstream vertices(dir / "vertices.csv");
    std::string header;
    std::getline(vertices, header);
    assert(header == "vertex_id,x,y,z,raw_curvature,csf_curvature,x0_log,xsmooth_log,xfinal_log,h0,hsmooth,hfinal");

    std::ifstream edges(dir / "edges.csv");
    std::getline(edges, header);
    assert(header == "i,j,edge_length,abs_dx,ratio,log_slope,violated");

    std::ifstream faces(dir / "faces.csv");
    std::getline(faces, header);
    assert(header == "face_id,grad_log_h_norm,max_edge_ratio");
  }

  std::filesystem::remove_all(dir);
}

void testTargetFieldConfigValidation() {
  bool threw = false;
  try {
    adaiso::AdaIsoConfig config;
    config.target_smooth_alpha = -1.0;
    config.project_to_surface = false;
    config.verbose = false;
    adaiso::AdaIsoRemesher remesher(config);
    (void)remesher.remesh(makeSingleTriangle());
  } catch (const std::runtime_error&) {
    threw = true;
  }
  assert(threw);

  threw = false;
  try {
    adaiso::AdaIsoConfig config;
    config.target_max_adjacent_ratio = 0.0;
    config.project_to_surface = false;
    config.verbose = false;
    adaiso::AdaIsoRemesher remesher(config);
    (void)remesher.remesh(makeSingleTriangle());
  } catch (const std::runtime_error&) {
    threw = true;
  }
  assert(threw);

  threw = false;
  try {
    adaiso::AdaIsoConfig config;
    config.csf_smoothing_alpha = 0.0;
    config.project_to_surface = false;
    config.verbose = false;
    adaiso::AdaIsoRemesher remesher(config);
    (void)remesher.remesh(makeSingleTriangle());
  } catch (const std::runtime_error&) {
    threw = true;
  }
  assert(threw);

  auto assert_invalid_target_config = [](adaiso::AdaIsoConfig config) {
    config.project_to_surface = false;
    config.verbose = false;
    adaiso::AdaIsoRemesher remesher(config);
    bool invalid_threw = false;
    try {
      (void)remesher.remesh(makeSingleTriangle());
    } catch (const std::runtime_error&) {
      invalid_threw = true;
    }
    assert(invalid_threw);
  };

  {
    adaiso::AdaIsoConfig config;
    config.target_min_factor = 0.0;
    assert_invalid_target_config(config);
  }
  {
    adaiso::AdaIsoConfig config;
    config.target_max_factor = 0.0;
    assert_invalid_target_config(config);
  }
  {
    adaiso::AdaIsoConfig config;
    config.target_min_factor = 2.0;
    config.target_max_factor = 1.0;
    assert_invalid_target_config(config);
  }
  {
    adaiso::AdaIsoConfig config;
    config.target_gamma = 0.0;
    assert_invalid_target_config(config);
  }
  {
    adaiso::AdaIsoConfig config;
    config.target_gradation_iterations = -1;
    assert_invalid_target_config(config);
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
  auto run_valid = [](const adaiso::Mesh& input) {
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

  run_valid(makeSingleTriangle());
  run_valid(makeDisconnectedTriangles());

  auto assert_invalid = [](const adaiso::Mesh& input) {
    adaiso::AdaIsoConfig config;
    config.global_iterations = 0;
    config.project_to_surface = false;
    config.verbose = false;
    adaiso::AdaIsoRemesher remesher(config);
    bool threw = false;
    try {
      (void)remesher.remesh(input);
    } catch (const std::runtime_error&) {
      threw = true;
    }
    assert(threw);
  };

  adaiso::Mesh empty;
  assert_invalid(empty);

  adaiso::Mesh degenerate;
  degenerate.vertices = {{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {2.0, 0.0, 0.0}};
  degenerate.faces = {{0, 1, 2}};
  assert_invalid(degenerate);

  adaiso::Mesh short_edge;
  short_edge.vertices = {{0.0, 0.0, 0.0}, {1e-15, 0.0, 0.0}, {0.0, 1.0, 0.0}};
  short_edge.faces = {{0, 1, 2}};
  assert_invalid(short_edge);

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
  testCsfScreenedKnownTriangleSolution();
  testCsfScreenedConstantField();
  testCsfScreenedWeightGraphAndStrictValidation();
  testTargetLengthFieldQueries();
  testContinuousTargetLengthMapping();
  testTargetLogGradationProjection();
  testEdgeRatioStats();
  testCurvatureModeParsing();
  testRobustMeanCurvatureRemeshInitialization();
  testRemeshSmoke();
  testTargetFieldDiagnosticExport();
  testTargetFieldConfigValidation();
  testBatchCollapsePassCollapsesMultipleComponents();
  testRemeshEdgeCases();
  testAreaSampledChamferDistance();
  return 0;
}
