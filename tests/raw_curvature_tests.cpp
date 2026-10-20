#include "TestSupport.hpp"

#include "adaiso/Geometry.hpp"
#include "adaiso/detail/RawCurvature.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

namespace {

adaiso::Mesh squareFanMesh() {
  adaiso::Mesh mesh;
  mesh.vertices = {
      {0.0, 0.0, 0.0},
      {1.0, 0.0, 0.0},
      {1.0, 1.0, 0.0},
      {0.0, 1.0, 0.0},
      {0.5, 0.5, 0.0},
  };
  mesh.faces = {{0, 1, 4}, {1, 2, 4}, {2, 3, 4}, {3, 0, 4}};
  return mesh;
}

adaiso::Mesh tetraMesh() {
  adaiso::Mesh mesh;
  mesh.vertices = {
      {0.0, 0.0, 0.0},
      {1.0, 0.0, 0.0},
      {0.0, 1.0, 0.0},
      {0.0, 0.0, 1.0},
  };
  mesh.faces = {{0, 2, 1}, {0, 1, 3}, {1, 2, 3}, {2, 0, 3}};
  return mesh;
}

}  // namespace

int main() {
  try {
    const adaiso::Mesh square = test_support::squareMesh();

    const auto robust = adaiso::detail::computeRobustMeanCurvatureField(square);
    for (double value : robust) {
      test_support::require(
          std::isfinite(value) && value >= 0.0,
          "robust-mean curvature produced invalid values");
    }

    const adaiso::Mesh fan = squareFanMesh();
    const auto fan_robust =
        adaiso::detail::computeRobustMeanCurvatureField(fan);
    test_support::requireNear(
        fan_robust[4], 0.0, 1e-12,
        "symmetric planar interior robust-mean curvature was not zero");

    const adaiso::Mesh tetra = tetraMesh();
    const double tetra_mean = adaiso::meanEdgeLength(tetra);
    const auto physical = adaiso::detail::computeRobustMeanCurvatureFields(
        tetra,
        adaiso::detail::CurvatureSignalMode::Physical);
    const auto dimless = adaiso::detail::computeRobustMeanCurvatureFields(
        tetra,
        adaiso::detail::CurvatureSignalMode::Dimensionless);
    const auto log_dimless = adaiso::detail::computeRobustMeanCurvatureFields(
        tetra,
        adaiso::detail::CurvatureSignalMode::LogDimensionless);
    for (std::size_t i = 0; i < tetra.vertexCount(); ++i) {
      test_support::requireNear(
          dimless.raw_signal[i],
          physical.physical_mean_curvature[i] * tetra_mean,
          1e-12,
          "dimensionless robust curvature did not equal H times mean edge");
      test_support::requireNear(
          log_dimless.raw_signal[i],
          std::log1p(dimless.raw_signal[i]),
          1e-12,
          "log-dimensionless robust curvature did not equal log1p(dimless)");
    }

    bool rejected = false;
    try {
      adaiso::Mesh empty;
      static_cast<void>(
          adaiso::detail::computeRobustMeanCurvatureField(empty));
    } catch (const std::runtime_error&) {
      rejected = true;
    }
    test_support::require(rejected, "empty mesh was not rejected");

    std::cout << "raw_curvature_tests: passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "raw_curvature_tests: " << error.what() << '\n';
    return 1;
  }
}
