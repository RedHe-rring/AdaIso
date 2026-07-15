#include "TestSupport.hpp"

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

    const auto normal =
        adaiso::detail::computeNormalAngleCurvatureField(square);
    for (double value : normal) {
      test_support::requireNear(
          value, 0.0, 1e-12,
          "planar normal-angle curvature was not zero");
    }

    const auto via_mode = adaiso::detail::computeRawCurvatureField(
        square, adaiso::CurvatureMode::RobustMean);
    test_support::require(
        via_mode.size() == square.vertexCount(),
        "raw curvature mode dispatch returned wrong size");

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
