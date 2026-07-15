#include "TestSupport.hpp"

#include "adaiso/Geometry.hpp"
#include "adaiso/detail/TargetFaceSizing.hpp"
#include "adaiso/detail/TargetGradation.hpp"

#include <iostream>
#include <vector>

int main() {
  try {
    const adaiso::Mesh mesh = test_support::equilateralTriangleMesh();
    const double mean_length = adaiso::meanEdgeLength(mesh);

    const std::vector<double> kappa(mesh.vertexCount(), 1.0);
    const auto initial = adaiso::detail::buildInitialPrincipalCurvatureLengths(
        mesh, kappa, mean_length);
    for (double value : initial.initial_lengths) {
      test_support::requireNear(
          value, 1.0, 1e-12, "unit curvature did not map to mean edge length");
    }

    const auto calibrated = adaiso::detail::calibrateTargetLengthsToFaceCount(
        mesh, initial.initial_lengths, 1, initial.h_min, initial.h_max, 64);
    test_support::require(calibrated.target_faces_reachable,
                          "reachable target was marked unreachable");
    test_support::require(
        calibrated.relative_error <= 1e-12,
        "target-face calibration did not recover the requested estimate");

    const std::vector<double> shape = {1.0, 1.2, 1.1};
    const double mean_estimate = adaiso::detail::estimateTargetFaceCount(
        mesh, shape, adaiso::TargetFaceRepresentative::Mean);
    const double min_estimate = adaiso::detail::estimateTargetFaceCount(
        mesh, shape, adaiso::TargetFaceRepresentative::Minimum);
    test_support::require(min_estimate > mean_estimate,
                          "minimum-h estimator did not increase density");
    const auto before =
        adaiso::detail::measureTargetGradation(mesh, shape, mean_length);
    const auto shape_calibrated =
        adaiso::detail::calibrateTargetLengthsToFaceCount(
            mesh, shape, 1, initial.h_min, initial.h_max, 64);
    const auto after = adaiso::detail::measureTargetGradation(
        mesh, shape_calibrated.target_lengths, mean_length);
    test_support::require(
        after.max_normalized_log_slope <=
            before.max_normalized_log_slope + 1e-12,
        "global calibration and clamp increased the gradation slope");

    const auto unreachable = adaiso::detail::calibrateTargetLengthsToFaceCount(
        mesh, initial.initial_lengths, 1000, initial.h_min, initial.h_max, 64);
    test_support::require(!unreachable.target_faces_reachable,
                          "unreachable target was marked reachable");

    const auto min_calibrated =
        adaiso::detail::calibrateTargetLengthsToFaceCount(
            mesh, shape, 1, initial.h_min, initial.h_max, 64,
            adaiso::TargetFaceRepresentative::Minimum);
    test_support::require(min_calibrated.relative_error <= 1e-12,
                          "minimum-h calibration missed its target");

    std::cout << "target_face_sizing_tests: passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "target_face_sizing_tests: " << error.what() << '\n';
    return 1;
  }
}
