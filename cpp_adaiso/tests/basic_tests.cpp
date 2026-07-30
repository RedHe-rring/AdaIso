#include "adaiso/AdaIsoRemesher.hpp"
#include "adaiso/Geometry.hpp"
#include "adaiso/IO.hpp"
#include "adaiso/Mesh.hpp"
#include "adaiso/Metrics.hpp"
#include "adaiso/detail/CsfSolver.hpp"
#include "adaiso/detail/TargetErrorSizing.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

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

adaiso::Mesh makeEquilateralTriangle() {
  adaiso::Mesh mesh;
  mesh.vertices = {
      {0.0, 0.0, 0.0},
      {1.0, 0.0, 0.0},
      {0.5, std::sqrt(3.0) * 0.5, 0.0},
  };
  mesh.faces = {{0, 1, 2}};
  return mesh;
}

adaiso::Mesh makeTargetMetricSplitTriangle() {
  adaiso::Mesh mesh;
  mesh.vertices = {
      {0.0, 0.0, 0.0},
      {100.0, 0.0, 0.0},
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

adaiso::Mesh makeFeatureCollapseMesh() {
  adaiso::Mesh mesh;
  mesh.vertices = {
      {0.0, 0.0, 0.0},
      {0.08, 0.0, 0.0},
      {0.04, 1.0, 0.0},
      {0.04, -1.0, 0.0},
      {-1.0, 0.0, 0.0},
      {0.08, 0.0, 2.5},
  };
  mesh.faces = {
      {0, 1, 2},
      {0, 3, 1},
      {0, 2, 4},
      {0, 4, 3},
      {1, 5, 2},
      {1, 3, 5},
      {2, 5, 4},
      {3, 4, 5},
  };
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

bool containsPointNear(const adaiso::Mesh& mesh, const Eigen::Vector3d& point, double eps = 1e-9) {
  for (const auto& vertex : mesh.vertices) {
    if ((vertex - point).norm() <= eps) {
      return true;
    }
  }
  return false;
}

Eigen::Vector3d findAddedVertex(const adaiso::Mesh& output, const adaiso::Mesh& input) {
  bool found = false;
  Eigen::Vector3d added = Eigen::Vector3d::Zero();
  for (const auto& vertex : output.vertices) {
    if (!containsPointNear(input, vertex)) {
      assert(!found);
      added = vertex;
      found = true;
    }
  }
  assert(found);
  return added;
}

double segmentParameter(const Eigen::Vector3d& a, const Eigen::Vector3d& b, const Eigen::Vector3d& p) {
  const Eigen::Vector3d ab = b - a;
  return (p - a).dot(ab) / ab.squaredNorm();
}

double distanceToSegmentLine(const Eigen::Vector3d& a, const Eigen::Vector3d& b, const Eigen::Vector3d& p) {
  const double t = segmentParameter(a, b, p);
  return (p - ((1.0 - t) * a + t * b)).norm();
}

adaiso::Mesh runOneCollapse(const adaiso::Mesh& input, bool feature_aware, double omega = 0.15) {
  adaiso::AdaIsoConfig config;
  config.variant = adaiso::Variant::Code;
  config.curvature_mode = adaiso::CurvatureMode::RobustMean;
  config.target_error_delta = 0.01;
  config.global_iterations = 1;
  config.max_split_passes = 0;
  config.max_collapse_passes = 1;
  config.max_flip_passes = 0;
  config.preprocess = false;
  config.tangent_lambda = 0.0;
  config.enable_relocation_mls = false;
  config.min_angle_floor_degrees = 0.0;
  config.feature_aware_collapse = feature_aware;
  config.collapse_feature_omega = omega;
  config.verbose = false;

  adaiso::AdaIsoRemesher remesher(config);
  adaiso::Mesh output = remesher.remesh(input);
  output.validate();
  assert(remesher.stats().size_stage_iterations == 1);
  assert(remesher.stats().size_stage_collapses == 1);
  assert(!remesher.stats().iterations.empty());
  assert(remesher.stats().iterations.front().stage == "size");
  assert(remesher.stats().iterations.front().collapse_count == 1);
  return output;
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
  assert(summary.min_angle_degrees > 0.0);
  assert(summary.angle_lt_10_pct >= 0.0 && summary.angle_lt_10_pct <= 100.0);
  assert(summary.angle_lt_20_pct >= 0.0 && summary.angle_lt_20_pct <= 100.0);
  const std::string summary_json = adaiso::summaryToJson(summary);
  assert(summary_json.find("\"min_angle_degrees\"") != std::string::npos);
  assert(summary_json.find("\"angle_lt_10_pct\"") != std::string::npos);
  assert(summary_json.find("\"angle_lt_20_pct\"") != std::string::npos);
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
  field.build(triangle, {1.0, 4.0, 16.0});

  assertNear(field.valueAt(triangle.vertices[0]), 1.0);
  assertNear(field.valueAt(0.5 * (triangle.vertices[0] + triangle.vertices[1])), 2.5);
  assertNear(field.valueAt({0.25, 0.25, 0.0}), 5.5);
  assertNear(field.valueAt({0.25, 0.25, 2.0}), 5.5);

  adaiso::Mesh square;
  square.vertices = {
      {0.0, 0.0, 0.0},
      {1.0, 0.0, 0.0},
      {0.0, 1.0, 0.0},
      {1.0, 1.0, 0.0},
  };
  square.faces = {{0, 1, 2}, {1, 3, 2}};
  field.build(square, {1.0, 4.0, 9.0, 16.0});
  assertNear(field.valueAt({0.5, 0.5, 0.0}), 6.5);
}

void testTargetErrorCurvatureSizing() {
  const adaiso::Mesh triangle = makeSingleTriangle();
  const double mean_length = adaiso::meanEdgeLength(triangle);
  const double bbox_diagonal = std::sqrt(2.0);
  const double delta = 0.1;
  const double h_max_factor = 3.0;
  const auto varying = adaiso::detail::targetErrorCurvatureLengths(
      triangle,
      {1.0, 4.0, 16.0},
      mean_length,
      bbox_diagonal,
      delta,
      h_max_factor,
      0.5);
  assert(varying.target_lengths.size() == triangle.vertexCount());
  assert(varying.target_lengths[0] > varying.target_lengths[1]);
  assert(varying.target_lengths[1] > varying.target_lengths[2]);
  assert(varying.local_edge_medians.size() == triangle.vertexCount());
  assert(varying.local_deltas.size() == triangle.vertexCount());
  assert(std::isfinite(varying.estimated_faces));
  assert(varying.estimated_faces > 0.0);
  assertNear(varying.h_min, 0.25 * mean_length);
  assertNear(varying.h_max, h_max_factor * mean_length);
  assertNear(varying.kappa_max, 16.0);
  assertNear(varying.auto_delta_quantile, 0.5);
  assertNear(varying.h_min_safe_factor, 0.25);
  assertNear(varying.target_error_delta, delta);
  assertNear(varying.target_error_bbox_ratio, delta / bbox_diagonal);
  assertNear(varying.target_lengths[0], std::sqrt(6.0 * delta / 1.0 - 3.0 * delta * delta));
  assertNear(varying.target_lengths[1], std::sqrt(6.0 * delta / 4.0 - 3.0 * delta * delta));
  assertNear(varying.target_lengths[2], 0.25 * mean_length);
  assertNear(
      adaiso::detail::estimateTargetFaceCount(triangle, varying.target_lengths),
      varying.estimated_faces,
      1e-12);

  const auto constant = adaiso::detail::targetErrorCurvatureLengths(
      triangle,
      {5.0, 5.0, 5.0},
      mean_length,
      bbox_diagonal,
      delta,
      h_max_factor,
      0.5);
  assert(constant.target_lengths.size() == triangle.vertexCount());
  assertNear(constant.target_lengths[0], constant.target_lengths[1], 1e-12);
  assertNear(constant.target_lengths[1], constant.target_lengths[2], 1e-12);
  assert(std::isfinite(constant.estimated_faces));
  assert(constant.estimated_faces > 0.0);

  const auto automatic = adaiso::detail::targetErrorCurvatureLengths(
      triangle,
      {1.0, 1.0, 1.0},
      mean_length,
      bbox_diagonal,
      0.0,
      h_max_factor,
      0.5);
  const double diagonal_median = (1.0 + std::sqrt(2.0)) * 0.5;
  const double expected_auto_delta = diagonal_median * diagonal_median / 6.0;
  assertNear(automatic.target_error_delta, expected_auto_delta);
  assertNear(automatic.local_edge_medians[0], 1.0);
  assertNear(automatic.local_edge_medians[1], diagonal_median);
  assertNear(automatic.local_edge_medians[2], diagonal_median);

  const auto q95 = adaiso::detail::targetErrorCurvatureLengths(
      triangle,
      {1.0, 1.0, 1.0},
      mean_length,
      bbox_diagonal,
      0.0,
      h_max_factor,
      0.95);
  assertNear(q95.target_error_delta, expected_auto_delta);
  assertNear(q95.auto_delta_quantile, 0.95);

  const auto flat = adaiso::detail::targetErrorCurvatureLengths(
      triangle,
      {0.0, 0.0, 0.0},
      mean_length,
      bbox_diagonal,
      delta,
      h_max_factor,
      0.5);
  assert(flat.flat_fallback);
  assert(flat.target_lengths.size() == triangle.vertexCount());
  assertNear(flat.h_min, 0.25 * mean_length);
  assertNear(flat.target_lengths[0], flat.h_max);
  assertNear(flat.target_lengths[1], flat.h_max);
  assertNear(flat.target_lengths[2], flat.h_max);

  auto expect_throw = [](auto&& fn) {
    bool threw = false;
    try {
      fn();
    } catch (const std::runtime_error&) {
      threw = true;
    }
    assert(threw);
  };

  expect_throw([] {
    (void)adaiso::detail::targetErrorCurvatureLengths(
        makeSingleTriangle(),
        {1.0, 2.0},
        1.0,
        1.0,
        0.01,
        5.0,
        0.5);
  });
  const auto nonfinite = adaiso::detail::targetErrorCurvatureLengths(
      triangle,
      {1.0, std::numeric_limits<double>::infinity(), 2.0},
      mean_length,
      bbox_diagonal,
      delta,
      h_max_factor,
      0.5);
  assertFiniteValues(nonfinite.target_lengths);
  expect_throw([] {
    (void)adaiso::detail::targetErrorCurvatureLengths(
        makeSingleTriangle(),
        {1.0, 2.0, 3.0},
        0.0,
        1.0,
        0.01,
        5.0,
        0.5);
  });
  expect_throw([] {
    (void)adaiso::detail::targetErrorCurvatureLengths(
        makeSingleTriangle(),
        {1.0, 2.0, 3.0},
        1.0,
        0.0,
        0.01,
        5.0,
        0.5);
  });
  expect_throw([] {
    (void)adaiso::detail::targetErrorCurvatureLengths(
        makeSingleTriangle(),
        {1.0, 2.0, 3.0},
        1.0,
        1.0,
        0.01,
        5.0,
        -0.1);
  });
  expect_throw([] {
    (void)adaiso::detail::targetErrorCurvatureLengths(
        makeSingleTriangle(),
        {1.0, 2.0, 3.0},
        1.0,
        1.0,
        0.01,
        5.0,
        1.1);
  });
  expect_throw([] {
    (void)adaiso::detail::targetErrorCurvatureLengths(
        makeSingleTriangle(),
        {1.0, 2.0, 3.0},
        1.0,
        1.0,
        0.01,
        0.0,
        0.5);
  });
  expect_throw([] {
    adaiso::Mesh empty;
    (void)adaiso::detail::targetErrorCurvatureLengths(
        empty,
        {},
        1.0,
        1.0,
        0.01,
        5.0,
        0.5);
  });
}

void testEdgeRatioStats() {
  {
    adaiso::Mesh mesh = makeSingleTriangle();
    adaiso::TargetLengthField field;
    field.build(mesh, {1.0, 1.0, 1.0});

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

  {
    adaiso::Mesh mesh = makeSingleTriangle();
    adaiso::TargetLengthField field;
    field.build(mesh, {1.0, 4.0, 4.0});

    const auto stats = adaiso::computeEdgeRatioStats(
        mesh,
        field,
        0.4,
        0.8);

    const double sqrt2 = std::sqrt(2.0);
    const double equal_target_metric = sqrt2 / 4.0;
    const double varying_target_metric = std::log(4.0) / 3.0;
    const double expected_mean = (equal_target_metric + 2.0 * varying_target_metric) / 3.0;
    const double expected_variance =
        ((equal_target_metric - expected_mean) * (equal_target_metric - expected_mean) +
            2.0 * (varying_target_metric - expected_mean) * (varying_target_metric - expected_mean)) /
        3.0;

    assertNear(stats.mean_ratio, expected_mean);
    assertNear(stats.std_ratio, std::sqrt(expected_variance));
    assertNear(stats.p05_ratio, 0.9 * equal_target_metric + 0.1 * varying_target_metric);
    assertNear(stats.p95_ratio, varying_target_metric);
    assertNear(stats.violation_rate, 1.0 / 3.0);
  }
}

void testCurvatureModeParsing() {
  assert(adaiso::parseCurvatureMode("normal-angle") == adaiso::CurvatureMode::NormalAngle);
  assert(adaiso::parseCurvatureMode("ROBUST-MEAN") == adaiso::CurvatureMode::RobustMean);
  assert(adaiso::parseCurvatureMode("principal-max") == adaiso::CurvatureMode::PrincipalMax);
  assert(adaiso::toString(adaiso::CurvatureMode::NormalAngle) == "normal-angle");
  assert(adaiso::toString(adaiso::CurvatureMode::RobustMean) == "robust-mean");
  assert(adaiso::toString(adaiso::CurvatureMode::PrincipalMax) == "principal-max");

  bool threw = false;
  try {
    (void)adaiso::parseCurvatureMode("gaussian");
  } catch (const std::runtime_error&) {
    threw = true;
  }
  assert(threw);
}

void testStageAPolicyParsing() {
  assert(adaiso::parseStageAPolicy("split_collapse_no_guard") ==
         adaiso::StageAPolicy::SplitCollapseNoGuard);
  assert(adaiso::parseStageAPolicy("split-collapse-weak-guard") ==
         adaiso::StageAPolicy::SplitCollapseWeakGuard);
  assert(adaiso::parseStageAPolicy("COLLAPSE_SPLIT_WEAK_GUARD") ==
         adaiso::StageAPolicy::CollapseSplitWeakGuard);
  assert(adaiso::toString(adaiso::StageAPolicy::SplitCollapseNoGuard) == "split_collapse_no_guard");
  assert(adaiso::toString(adaiso::StageAPolicy::SplitCollapseWeakGuard) == "split_collapse_weak_guard");
  assert(adaiso::toString(adaiso::StageAPolicy::CollapseSplitWeakGuard) == "collapse_split_weak_guard");
  assert(adaiso::AdaIsoConfig{}.stage_a_policy == adaiso::StageAPolicy::SplitCollapseNoGuard);

  bool threw = false;
  try {
    (void)adaiso::parseStageAPolicy("strong-guard");
  } catch (const std::runtime_error&) {
    threw = true;
  }
  assert(threw);
}

void testHysteresisFactorOverrides() {
  {
    adaiso::AdaIsoConfig config;
    config.variant = adaiso::Variant::Code;
    assertNear(config.splitFactor(), 1.33);
    assertNear(config.collapseFactor(), 0.8);
  }
  {
    adaiso::AdaIsoConfig config;
    config.variant = adaiso::Variant::Paper;
    assertNear(config.splitFactor(), 5.0 / 3.0);
    assertNear(config.collapseFactor(), 0.8);
  }
  {
    adaiso::AdaIsoConfig config;
    config.variant = adaiso::Variant::Paper;
    config.split_factor_override = 1.2;
    config.collapse_factor_override = 0.6;
    assertNear(config.splitFactor(), 1.2);
    assertNear(config.collapseFactor(), 0.6);
  }
}

void testRobustMeanCurvatureRemeshInitialization() {
  auto run = [](const adaiso::Mesh& input) {
    adaiso::AdaIsoConfig config;
    config.curvature_mode = adaiso::CurvatureMode::RobustMean;
    config.global_iterations = 0;
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
  config.global_iterations = 2;
  config.preprocess = true;
  config.verbose = false;

  adaiso::AdaIsoRemesher remesher(config);
  const adaiso::Mesh output = remesher.remesh(input);
  output.validate();
  assert(output.vertexCount() > 0);
  assert(output.faceCount() > 0);
  const auto output_summary = adaiso::summarizeMesh(output);
  assert(output_summary.qmin > 0.0);
  assert(output_summary.qmin + 1e-12 >= std::min(0.01, input_summary.qmin));
  assert(std::isfinite(output_summary.min_angle_degrees));
  assert(std::isfinite(output_summary.angle_lt_10_pct));
  assert(std::isfinite(output_summary.angle_lt_20_pct));
  assert(remesher.stats().csf_solver_success);
  assert(remesher.stats().csf_residual_scaled < adaiso::detail::kResidualThreshold);
  const auto& iterations = remesher.stats().iterations;
  assert(!iterations.empty());
  assert(remesher.stats().size_stage_iterations >= 1);
  assert(remesher.stats().size_stage_iterations <= config.global_iterations);
  assert(remesher.stats().size_stage_split_accepted ==
         static_cast<std::size_t>(remesher.stats().size_stage_splits));
  assert(remesher.stats().size_stage_collapse_accepted ==
         static_cast<std::size_t>(remesher.stats().size_stage_collapses));
  const std::size_t collapse_reject_reasons =
      remesher.stats().size_stage_collapse_rejected_by_boundary +
      remesher.stats().size_stage_collapse_rejected_by_non_manifold +
      remesher.stats().size_stage_collapse_rejected_by_batch_conflict +
      remesher.stats().size_stage_collapse_rejected_by_link_condition +
      remesher.stats().size_stage_collapse_rejected_by_degenerate_patch +
      remesher.stats().size_stage_collapse_rejected_by_target_length +
      remesher.stats().size_stage_collapse_rejected_by_angle +
      remesher.stats().size_stage_collapse_rejected_by_batch_duplicate;
  assert(collapse_reject_reasons == remesher.stats().size_stage_collapse_rejected);
  assert(remesher.stats().size_stage_collapse_accepted +
             remesher.stats().size_stage_collapse_rejected ==
         remesher.stats().size_stage_short_edge_candidates);
  assert(std::isfinite(remesher.stats().size_stage_final_violation));
  assert(std::isfinite(remesher.stats().size_stage_final_min_angle_degrees));
  assert(std::isfinite(remesher.stats().size_stage_final_angle_lt_10_pct));
  assert(std::isfinite(remesher.stats().size_stage_final_angle_lt_20_pct));
  assert(remesher.stats().quality_stage_iterations >= 1);
  assert(remesher.stats().quality_stage_iterations <= config.global_iterations);
  for (const auto& iteration : iterations) {
    assert(iteration.stage == "size" || iteration.stage == "quality");
    assert(iteration.iteration >= 1);
    assert(iteration.iteration <= config.global_iterations);
    assert(iteration.split_count >= 0);
    assert(iteration.collapse_count >= 0);
    assert(iteration.flip_count >= 0);
    assert(iteration.relocation_count >= 0);
    if (iteration.stage == "size") {
      assert(iteration.flip_count == 0);
      assert(iteration.relocation_count == 0);
    } else {
      assert(iteration.split_count == 0);
      assert(iteration.collapse_count == 0);
    }
    assert(iteration.vertices > 0);
    assert(iteration.faces > 0);
    assert(std::isfinite(iteration.qavg));
    assert(std::isfinite(iteration.qmin));
    assert(std::isfinite(iteration.min_angle_degrees));
    assert(std::isfinite(iteration.angle_lt_10_pct));
    assert(std::isfinite(iteration.angle_lt_20_pct));
    assert(std::isfinite(iteration.mean_ratio));
    assert(std::isfinite(iteration.std_ratio));
    assert(std::isfinite(iteration.p05_ratio));
    assert(std::isfinite(iteration.p95_ratio));
    assert(std::isfinite(iteration.violation_rate));
    assert(iteration.violation_rate >= 0.0 && iteration.violation_rate <= 1.0);
  }
}

void testTargetFieldDiagnosticExport() {
  const std::filesystem::path dir =
      std::filesystem::temp_directory_path() / "adaiso_target_field_diagnostic_export";
  std::filesystem::remove_all(dir);

  adaiso::AdaIsoConfig config;
  config.global_iterations = 0;
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
      dir / "field_stats.json",
      dir / "raw_curvature_heat_shared.ply",
      dir / "csf_curvature_heat_shared.ply",
      dir / "raw_curvature_heat_independent.ply",
      dir / "csf_curvature_heat_independent.ply",
      dir / "local_delta_heat.ply",
      dir / "target_h_heat.ply",
      dir / "target_h_heat_observed.ply",
      dir / "output_target_h_samebar.ply",
      dir / "output_edge_ratio_min_heat.ply",
      dir / "output_edge_ratio_max_heat.ply",
      dir / "output_edge_ratios.csv",
      dir / "output_visualization_stats.json",
      dir / "color_scales.json",
  };
  for (const auto& path : expected) {
    assert(std::filesystem::exists(path));
    assert(std::filesystem::file_size(path) > 0);
  }

  {
    std::ifstream vertices(dir / "vertices.csv");
    std::string header;
    std::getline(vertices, header);
    assert(header == "vertex_id,x,y,z,raw_curvature,csf_curvature,local_edge_median,local_delta,target_h");

    std::ifstream edges(dir / "edges.csv");
    std::getline(edges, header);
    assert(header == "i,j,edge_length,abs_dh,target_ratio,h_slope");

    std::ifstream faces(dir / "faces.csv");
    std::getline(faces, header);
    assert(header == "face_id,grad_h_norm,max_edge_target_ratio");

    std::ifstream output_edges(dir / "output_edge_ratios.csv");
    std::getline(output_edges, header);
    assert(header == "i,j,edge_length,target_i,target_j,target_min,target_max,metric_length");
  }

  {
    std::ifstream field_stats(dir / "field_stats.json");
    std::string field_stats_text;
    std::string line;
    while (std::getline(field_stats, line)) {
      field_stats_text += line;
      field_stats_text += '\n';
    }
    assert(field_stats_text.find("\"auto_delta_source\": \"input_resolution_csf_curvature\"") != std::string::npos);
    assert(field_stats_text.find("\"auto_delta_quantile\"") != std::string::npos);
    assert(field_stats_text.find("\"h_min_safe_factor\"") != std::string::npos);
    assert(field_stats_text.find("\"local_delta_min\"") != std::string::npos);
    assert(field_stats_text.find("\"local_delta_q50\"") != std::string::npos);
    assert(field_stats_text.find("\"local_delta_q95\"") != std::string::npos);
    assert(field_stats_text.find("\"local_delta_max\"") != std::string::npos);
    assert(field_stats_text.find("\"kappa_max\"") != std::string::npos);
  }

  {
    std::ifstream color_scales(dir / "color_scales.json");
    std::string color_scales_text;
    std::string line;
    while (std::getline(color_scales, line)) {
      color_scales_text += line;
      color_scales_text += '\n';
    }
    assert(color_scales_text.find("\"raw_curvature_shared\"") != std::string::npos);
    assert(color_scales_text.find("\"csf_curvature_shared\"") != std::string::npos);
    assert(color_scales_text.find("\"raw_curvature_independent\"") != std::string::npos);
    assert(color_scales_text.find("\"csf_curvature_independent\"") != std::string::npos);
    assert(color_scales_text.find("\"local_delta\"") != std::string::npos);
    assert(color_scales_text.find("\"target_h\"") != std::string::npos);
    assert(color_scales_text.find("\"target_h_observed\"") != std::string::npos);
    assert(color_scales_text.find("\"raw_curvature_heat_shared.ply\"") != std::string::npos);
    assert(color_scales_text.find("\"csf_curvature_heat_shared.ply\"") != std::string::npos);
    assert(color_scales_text.find("\"raw_curvature_heat_independent.ply\"") != std::string::npos);
    assert(color_scales_text.find("\"csf_curvature_heat_independent.ply\"") != std::string::npos);
    assert(color_scales_text.find("\"local_delta_heat.ply\"") != std::string::npos);
    assert(color_scales_text.find("\"target_h_heat.ply\"") != std::string::npos);
    assert(color_scales_text.find("\"target_h_heat_observed.ply\"") != std::string::npos);
    assert(color_scales_text.find("\"inverted\": true") != std::string::npos);
  }

  {
    std::ifstream output_stats(dir / "output_visualization_stats.json");
    std::string output_stats_text;
    std::string line;
    while (std::getline(output_stats, line)) {
      output_stats_text += line;
      output_stats_text += '\n';
    }
    assert(output_stats_text.find("\"output_target_h_samebar.ply\"") != std::string::npos);
    assert(output_stats_text.find("\"output_edge_ratio_min_heat.ply\"") != std::string::npos);
    assert(output_stats_text.find("\"output_edge_ratio_max_heat.ply\"") != std::string::npos);
    assert(output_stats_text.find("\"target_metric_edge_length\"") != std::string::npos);
    assert(output_stats_text.find("\"violation_rate\"") != std::string::npos);
  }

  {
    std::ifstream vertices(dir / "vertices.csv");
    std::string line;
    std::getline(vertices, line);
    std::vector<double> target_h;
    while (std::getline(vertices, line)) {
      std::stringstream row(line);
      std::string cell;
      for (int column = 0; column < 8; ++column) {
        std::getline(row, cell, ',');
      }
      std::getline(row, cell, ',');
      target_h.push_back(std::stod(cell));
    }
    assert(!target_h.empty());
    const auto min_it = std::min_element(target_h.begin(), target_h.end());
    const auto max_it = std::max_element(target_h.begin(), target_h.end());
    assert(*min_it < *max_it);
    const std::size_t min_index = static_cast<std::size_t>(std::distance(target_h.begin(), min_it));
    const std::size_t max_index = static_cast<std::size_t>(std::distance(target_h.begin(), max_it));

    std::ifstream target_h_ply(dir / "target_h_heat.ply");
    while (std::getline(target_h_ply, line) && line != "end_header") {
    }

    std::vector<int> red;
    std::vector<int> blue;
    red.reserve(target_h.size());
    blue.reserve(target_h.size());
    for (std::size_t i = 0; i < target_h.size(); ++i) {
      double x = 0.0;
      double y = 0.0;
      double z = 0.0;
      int r = 0;
      int g = 0;
      int b = 0;
      target_h_ply >> x >> y >> z >> r >> g >> b;
      red.push_back(r);
      blue.push_back(b);
    }
    assert(red[min_index] > red[max_index]);
    assert(blue[min_index] < blue[max_index]);
  }

  assert(!std::filesystem::exists(dir / "raw_curvature_heat.ply"));
  assert(!std::filesystem::exists(dir / "csf_curvature_heat.ply"));
  assert(!std::filesystem::exists(dir / "raw_target_h_heat.ply"));
  assert(!std::filesystem::exists(dir / "smoothed_target_h_heat.ply"));
  assert(!std::filesystem::exists(dir / "final_target_h_heat.ply"));
  assert(!std::filesystem::exists(dir / "target_delta_log_heat.ply"));

  std::filesystem::remove_all(dir);
}

void testIterationMeshExport() {
  const std::filesystem::path dir =
      std::filesystem::temp_directory_path() / "adaiso_iteration_mesh_export";
  std::filesystem::remove_all(dir);

  adaiso::AdaIsoConfig config;
  config.global_iterations = 2;
  config.verbose = false;
  config.iteration_mesh_export_dir = dir.string();

  adaiso::AdaIsoRemesher remesher(config);
  const adaiso::Mesh output = remesher.remesh(makePyramid());
  output.validate();

  const std::vector<std::filesystem::path> expected = {
      dir / "iter000.ply",
      dir / "size_iter001.ply",
      dir / "quality_iter001.ply",
  };
  for (const auto& path : expected) {
    assert(std::filesystem::exists(path));
    assert(std::filesystem::file_size(path) > 0);
    const adaiso::Mesh loaded = adaiso::loadMesh(path);
    loaded.validate();
  }

  assert(!std::filesystem::exists(dir / "iter001.ply"));
  std::filesystem::remove_all(dir);
}

void testStageMeshExport() {
  const std::filesystem::path dir =
      std::filesystem::temp_directory_path() / "adaiso_stage_mesh_export";
  std::filesystem::remove_all(dir);

  adaiso::AdaIsoConfig config;
  config.global_iterations = 1;
  config.verbose = false;
  config.stage_mesh_export_dir = dir.string();

  adaiso::AdaIsoRemesher remesher(config);
  const adaiso::Mesh output = remesher.remesh(makePyramid());
  output.validate();

  const std::vector<std::filesystem::path> expected = {
      dir / "size_iter001_after_split.ply",
      dir / "size_iter001_after_collapse.ply",
      dir / "quality_iter001_after_flip.ply",
      dir / "quality_iter001_after_smooth.ply",
  };
  for (const auto& path : expected) {
    assert(std::filesystem::exists(path));
    assert(std::filesystem::file_size(path) > 0);
    const adaiso::Mesh loaded = adaiso::loadMesh(path);
    loaded.validate();
  }

  std::filesystem::remove_all(dir);
}

void testAllStageMeshExport() {
  const std::filesystem::path dir =
      std::filesystem::temp_directory_path() / "adaiso_all_stage_mesh_export";
  std::filesystem::remove_all(dir);

  adaiso::AdaIsoConfig config;
  config.global_iterations = 2;
  config.verbose = false;
  config.all_stage_mesh_export_dir = dir.string();

  adaiso::AdaIsoRemesher remesher(config);
  const adaiso::Mesh output = remesher.remesh(makePyramid());
  output.validate();

  const std::vector<std::filesystem::path> expected = {
      dir / "size_iter001_after_split.ply",
      dir / "size_iter001_after_collapse.ply",
      dir / "quality_iter001_after_flip.ply",
      dir / "quality_iter001_after_smooth.ply",
  };
  for (const auto& path : expected) {
    assert(std::filesystem::exists(path));
    assert(std::filesystem::file_size(path) > 0);
    const adaiso::Mesh loaded = adaiso::loadMesh(path);
    loaded.validate();
  }

  assert(!std::filesystem::exists(dir / "iter001_after_split.ply"));
  std::filesystem::remove_all(dir);
}

void testRelocationKeepsBoundaryVerticesFixed() {
  const adaiso::Mesh input = makeSingleTriangle();

  adaiso::AdaIsoConfig config;
  config.global_iterations = 1;
  config.max_split_passes = 0;
  config.max_collapse_passes = 0;
  config.max_flip_passes = 0;
  config.verbose = false;

  adaiso::AdaIsoRemesher remesher(config);
  const adaiso::Mesh output = remesher.remesh(input);
  output.validate();

  assert(output.vertexCount() == input.vertexCount());
  assert(output.faceCount() == input.faceCount());
  for (std::size_t i = 0; i < input.vertices.size(); ++i) {
    assert((output.vertices[i] - input.vertices[i]).norm() <= 1e-12);
  }
}

void testTargetFieldConfigValidation() {
  bool threw = false;
  try {
    adaiso::AdaIsoConfig config;
    config.csf_smoothing_alpha = 0.0;
    config.verbose = false;
    adaiso::AdaIsoRemesher remesher(config);
    (void)remesher.remesh(makeSingleTriangle());
  } catch (const std::runtime_error&) {
    threw = true;
  }
  assert(threw);

  auto assert_invalid_config = [](adaiso::AdaIsoConfig config) {
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
    config.collapse_feature_omega = -0.1;
    assert_invalid_config(config);
  }
  {
    adaiso::AdaIsoConfig config;
    config.collapse_feature_omega = std::numeric_limits<double>::infinity();
    assert_invalid_config(config);
  }
  {
    adaiso::AdaIsoConfig config;
    config.target_error_delta = -1.0;
    assert_invalid_config(config);
  }
  {
    adaiso::AdaIsoConfig config;
    config.split_factor_override = -1.0;
    assert_invalid_config(config);
  }
  {
    adaiso::AdaIsoConfig config;
    config.split_factor_override = std::numeric_limits<double>::infinity();
    assert_invalid_config(config);
  }
  {
    adaiso::AdaIsoConfig config;
    config.collapse_factor_override = -1.0;
    assert_invalid_config(config);
  }
  {
    adaiso::AdaIsoConfig config;
    config.collapse_factor_override = std::numeric_limits<double>::infinity();
    assert_invalid_config(config);
  }
  {
    adaiso::AdaIsoConfig config;
    config.target_h_max_factor = 0.0;
    assert_invalid_config(config);
  }
  {
    adaiso::AdaIsoConfig config;
    config.target_h_max_factor = std::numeric_limits<double>::infinity();
    assert_invalid_config(config);
  }
  {
    adaiso::AdaIsoConfig config;
    config.auto_delta_quantile = -0.1;
    assert_invalid_config(config);
  }
  {
    adaiso::AdaIsoConfig config;
    config.auto_delta_quantile = 1.1;
    assert_invalid_config(config);
  }
  {
    adaiso::AdaIsoConfig config;
    config.auto_delta_quantile = std::numeric_limits<double>::infinity();
    assert_invalid_config(config);
  }
  {
    adaiso::AdaIsoConfig config;
    config.min_angle_floor_degrees = -0.1;
    assert_invalid_config(config);
  }
  {
    adaiso::AdaIsoConfig config;
    config.min_angle_floor_degrees = 60.1;
    assert_invalid_config(config);
  }
  {
    adaiso::AdaIsoConfig config;
    config.min_angle_floor_degrees = std::numeric_limits<double>::infinity();
    assert_invalid_config(config);
  }
  {
    adaiso::AdaIsoConfig config;
    config.split_min_angle_floor_degrees = -0.1;
    assert_invalid_config(config);
  }
  {
    adaiso::AdaIsoConfig config;
    config.split_min_angle_floor_degrees = 60.1;
    assert_invalid_config(config);
  }
  {
    adaiso::AdaIsoConfig config;
    config.split_min_angle_floor_degrees = std::numeric_limits<double>::infinity();
    assert_invalid_config(config);
  }
}

void testBatchCollapsePassCollapsesMultipleComponents() {
  const adaiso::Mesh input = makeTwoDisconnectedOctahedra();
  input.validate();

  adaiso::AdaIsoConfig config;
  config.variant = adaiso::Variant::Code;
  config.curvature_mode = adaiso::CurvatureMode::RobustMean;
  config.target_error_delta = 0.05;
  config.global_iterations = 1;
  config.max_split_passes = 0;
  config.max_collapse_passes = 1;
  config.max_flip_passes = 0;
  config.preprocess = false;
  config.tangent_lambda = 0.0;
  config.enable_relocation_mls = false;
  config.min_angle_floor_degrees = 0.0;
  config.verbose = false;

  adaiso::AdaIsoRemesher remesher(config);
  const adaiso::Mesh output = remesher.remesh(input);
  output.validate();
  assert(output.vertexCount() + 2 <= input.vertexCount());
  assert(output.faceCount() > 0);
  assert(remesher.stats().csf_solver_success);
}

void testCollapseTargetEnergyAcceptanceDiagnostics() {
  const adaiso::Mesh input = makeFeatureCollapseMesh();
  input.validate();

  adaiso::AdaIsoConfig config;
  config.variant = adaiso::Variant::Code;
  config.curvature_mode = adaiso::CurvatureMode::RobustMean;
  config.target_error_delta = 0.01;
  config.global_iterations = 1;
  config.max_split_passes = 0;
  config.max_collapse_passes = 1;
  config.max_flip_passes = 0;
  config.preprocess = false;
  config.tangent_lambda = 0.0;
  config.enable_relocation_mls = false;
  config.min_angle_floor_degrees = 0.0;
  config.split_factor_override = 0.85;
  config.collapse_factor_override = 0.8;
  config.verbose = false;

  adaiso::AdaIsoRemesher remesher(config);
  const adaiso::Mesh output = remesher.remesh(input);
  output.validate();

  const auto& stats = remesher.stats();
  assert(stats.size_stage_collapses > 0);
  assert(stats.size_stage_collapse_accepted == static_cast<std::size_t>(stats.size_stage_collapses));

  const std::size_t collapse_reject_reasons =
      stats.size_stage_collapse_rejected_by_boundary +
      stats.size_stage_collapse_rejected_by_non_manifold +
      stats.size_stage_collapse_rejected_by_batch_conflict +
      stats.size_stage_collapse_rejected_by_link_condition +
      stats.size_stage_collapse_rejected_by_degenerate_patch +
      stats.size_stage_collapse_rejected_by_target_length +
      stats.size_stage_collapse_rejected_by_angle +
      stats.size_stage_collapse_rejected_by_batch_duplicate;
  assert(collapse_reject_reasons == stats.size_stage_collapse_rejected);
  assert(stats.size_stage_collapse_accepted + stats.size_stage_collapse_rejected ==
         stats.size_stage_short_edge_candidates);
}

void testTargetMetricSplitPlacement() {
  {
    const adaiso::Mesh input = makeTargetMetricSplitTriangle();
    input.validate();
    const std::filesystem::path dir =
        std::filesystem::temp_directory_path() / "adaiso_target_metric_split_placement";
    std::filesystem::remove_all(dir);

    adaiso::AdaIsoConfig config;
    config.variant = adaiso::Variant::Code;
    config.curvature_mode = adaiso::CurvatureMode::RobustMean;
    config.target_error_delta = 1e-8;
    config.global_iterations = 1;
    config.max_split_passes = 1;
    config.max_collapse_passes = 0;
    config.max_flip_passes = 0;
    config.preprocess = false;
    config.tangent_lambda = 0.0;
    config.enable_relocation_mls = false;
    config.stage_mesh_export_dir = dir.string();
    config.verbose = false;

    adaiso::AdaIsoRemesher remesher(config);
    const adaiso::Mesh output = remesher.remesh(input);
    output.validate();
    assert(remesher.stats().size_stage_iterations == 1);
    assert(remesher.stats().size_stage_splits == 1);
    assert(!remesher.stats().iterations.empty());
    assert(remesher.stats().iterations.front().stage == "size");
    assert(remesher.stats().iterations.front().split_count == 1);

    const adaiso::Mesh split_stage = adaiso::loadMesh(dir / "size_iter001_after_split.ply");
    split_stage.validate();
    assert(split_stage.vertexCount() == input.vertexCount() + 1);
    const Eigen::Vector3d added = findAddedVertex(split_stage, input);
    const double t = segmentParameter(input.vertices[1], input.vertices[2], added);

    assert(distanceToSegmentLine(input.vertices[1], input.vertices[2], added) <= 1e-8);
    assert(t >= 0.25 - 1e-10);
    assert(t <= 0.75 + 1e-10);
    assert(std::abs(t - 0.5) > 1e-3);
  }

  {
    const adaiso::Mesh input = makeEquilateralTriangle();
    input.validate();
    const std::filesystem::path dir =
        std::filesystem::temp_directory_path() / "adaiso_target_metric_equal_split";
    std::filesystem::remove_all(dir);

    adaiso::AdaIsoConfig config;
    config.variant = adaiso::Variant::Code;
    config.curvature_mode = adaiso::CurvatureMode::RobustMean;
    config.target_error_delta = 1e-8;
    config.global_iterations = 1;
    config.max_split_passes = 1;
    config.max_collapse_passes = 0;
    config.max_flip_passes = 0;
    config.preprocess = false;
    config.tangent_lambda = 0.0;
    config.enable_relocation_mls = false;
    config.stage_mesh_export_dir = dir.string();
    config.verbose = false;

    adaiso::AdaIsoRemesher remesher(config);
    const adaiso::Mesh output = remesher.remesh(input);
    output.validate();
    assert(remesher.stats().size_stage_iterations == 1);
    assert(remesher.stats().size_stage_splits == 1);
    assert(!remesher.stats().iterations.empty());
    assert(remesher.stats().iterations.front().stage == "size");
    assert(remesher.stats().iterations.front().split_count == 1);

    const adaiso::Mesh split_stage = adaiso::loadMesh(dir / "size_iter001_after_split.ply");
    split_stage.validate();
    assert(split_stage.vertexCount() == input.vertexCount() + 1);
    const Eigen::Vector3d added = findAddedVertex(split_stage, input);

    bool midpoint_found = false;
    for (const auto& edge : adaiso::uniqueEdges(input)) {
      const Eigen::Vector3d midpoint =
          0.5 * (input.vertices[static_cast<std::size_t>(edge.first)] +
                 input.vertices[static_cast<std::size_t>(edge.second)]);
      if ((added - midpoint).norm() <= 1e-8) {
        midpoint_found = true;
        break;
      }
    }
    assert(midpoint_found);
  }
}

void testWeakSplitAngleGuard() {
  const adaiso::Mesh input = makeEquilateralTriangle();
  input.validate();

  {
    adaiso::AdaIsoConfig config;
    config.variant = adaiso::Variant::Code;
    config.curvature_mode = adaiso::CurvatureMode::RobustMean;
    config.target_error_delta = 1e-8;
    config.global_iterations = 1;
    config.max_split_passes = 1;
    config.max_collapse_passes = 0;
    config.max_flip_passes = 0;
    config.preprocess = false;
    config.tangent_lambda = 0.0;
    config.enable_relocation_mls = false;
    config.verbose = false;

    adaiso::AdaIsoRemesher remesher(config);
    const adaiso::Mesh output = remesher.remesh(input);
    output.validate();
    assert(remesher.stats().size_stage_long_edge_candidates > 0);
    assert(remesher.stats().size_stage_split_accepted == 1);
    assert(remesher.stats().size_stage_split_rejected_by_angle == 0);
    assert(output.vertexCount() == input.vertexCount() + 1);
  }

  {
    adaiso::AdaIsoConfig config;
    config.variant = adaiso::Variant::Code;
    config.curvature_mode = adaiso::CurvatureMode::RobustMean;
    config.target_error_delta = 1e-8;
    config.global_iterations = 1;
    config.max_split_passes = 1;
    config.max_collapse_passes = 0;
    config.max_flip_passes = 0;
    config.preprocess = false;
    config.tangent_lambda = 0.0;
    config.enable_relocation_mls = false;
    config.split_min_angle_floor_degrees = 60.0;
    config.verbose = false;

    adaiso::AdaIsoRemesher remesher(config);
    const adaiso::Mesh output = remesher.remesh(input);
    output.validate();
    assert(remesher.stats().size_stage_long_edge_candidates > 0);
    assert(remesher.stats().size_stage_split_accepted == 0);
    assert(remesher.stats().size_stage_split_rejected_by_angle > 0);
    assert(output.vertexCount() == input.vertexCount());
  }
}

void testFeatureAwareCollapsePlacement() {
  const adaiso::Mesh input = makeFeatureCollapseMesh();
  input.validate();

  const Eigen::Vector3d midpoint = 0.5 * (input.vertices[0] + input.vertices[1]);
  const adaiso::Mesh midpoint_output = runOneCollapse(input, false);
  assert(containsPointNear(midpoint_output, midpoint));
  assert(!containsPointNear(midpoint_output, input.vertices[0]));
  assert(!containsPointNear(midpoint_output, input.vertices[1]));

  const adaiso::Mesh feature_output = runOneCollapse(input, true);
  assert(!containsPointNear(feature_output, midpoint));
  assert(containsPointNear(feature_output, input.vertices[0]) ||
         containsPointNear(feature_output, input.vertices[1]));

  const adaiso::Mesh high_omega_output = runOneCollapse(input, true, 1000.0);
  assert(containsPointNear(high_omega_output, midpoint));
}

void testRemeshEdgeCases() {
  auto run_valid = [](const adaiso::Mesh& input) {
    adaiso::AdaIsoConfig config;
    config.global_iterations = 0;
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
  testTargetErrorCurvatureSizing();
  testEdgeRatioStats();
  testCurvatureModeParsing();
  testStageAPolicyParsing();
  testHysteresisFactorOverrides();
  testRobustMeanCurvatureRemeshInitialization();
  testRemeshSmoke();
  testTargetFieldDiagnosticExport();
  testIterationMeshExport();
  testStageMeshExport();
  testAllStageMeshExport();
  testRelocationKeepsBoundaryVerticesFixed();
  testTargetFieldConfigValidation();
  testBatchCollapsePassCollapsesMultipleComponents();
  testCollapseTargetEnergyAcceptanceDiagnostics();
  testTargetMetricSplitPlacement();
  testWeakSplitAngleGuard();
  testFeatureAwareCollapsePlacement();
  testRemeshEdgeCases();
  testAreaSampledChamferDistance();
  return 0;
}
