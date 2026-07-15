#include "TestSupport.hpp"

#include "adaiso/Geometry.hpp"
#include "adaiso/detail/TargetFaceSizing.hpp"

#include <cmath>
#include <iostream>
#include <limits>
#include <vector>

namespace {

adaiso::Mesh repeatedEquilateralTriangles(int count) {
  adaiso::Mesh mesh;
  const double height = std::sqrt(3.0) * 0.5;
  for (int i = 0; i < count; ++i) {
    const int base = static_cast<int>(mesh.vertices.size());
    const double offset = static_cast<double>(i) * 3.0;
    mesh.vertices.push_back({offset, 0.0, 0.0});
    mesh.vertices.push_back({offset + 1.0, 0.0, 0.0});
    mesh.vertices.push_back({offset + 0.5, height, 0.0});
    mesh.faces.push_back({base, base + 1, base + 2});
  }
  return mesh;
}

}  // namespace

int main() {
  try {
    const adaiso::Mesh mesh = test_support::equilateralTriangleMesh();
    const double mean_length = adaiso::meanEdgeLength(mesh);

    const std::vector<double> kappa(mesh.vertexCount(), 1.0);
    const auto sizing = adaiso::detail::targetFaceCurvatureLengths(
        mesh, kappa, 1, mean_length);
    test_support::requireNear(
        sizing.estimated_faces, 1.0, 1e-12,
        "target-face sizing did not recover the requested estimate");
    for (double value : sizing.target_lengths) {
      test_support::requireNear(
          value, mean_length, 1e-12,
          "unit CSF curvature did not calibrate to mean edge length");
    }

    const auto dense = adaiso::detail::targetFaceCurvatureLengths(
        mesh, kappa, 1000, mean_length);
    for (double value : dense.target_lengths) {
      test_support::requireNear(
          value,
          adaiso::detail::kTargetMinEdgeFactor * mean_length,
          1e-12,
          "unreachable dense target did not clamp to h_min");
    }

    const adaiso::Mesh many = repeatedEquilateralTriangles(30);
    const std::vector<double> many_kappa(many.vertexCount(), 1.0);
    const auto coarse = adaiso::detail::targetFaceCurvatureLengths(
        many, many_kappa, 1, adaiso::meanEdgeLength(many));
    for (double value : coarse.target_lengths) {
      test_support::requireNear(
          value,
          adaiso::detail::kTargetMaxEdgeFactor * adaiso::meanEdgeLength(many),
          1e-12,
          "unreachable coarse target did not clamp to h_max");
    }

    bool rejected = false;
    try {
      std::vector<double> bad = kappa;
      bad[0] = std::numeric_limits<double>::infinity();
      static_cast<void>(adaiso::detail::targetFaceCurvatureLengths(
          mesh, bad, 1, mean_length));
    } catch (const std::runtime_error&) {
      rejected = true;
    }
    test_support::require(rejected, "non-finite curvature was not rejected");

    std::cout << "target_face_sizing_tests: passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "target_face_sizing_tests: " << error.what() << '\n';
    return 1;
  }
}
