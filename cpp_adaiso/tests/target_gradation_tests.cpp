#include "TestSupport.hpp"

#include "adaiso/Geometry.hpp"
#include "adaiso/detail/TargetGradation.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

int main() {
  try {
    const adaiso::Mesh mesh = test_support::squareMesh();
    const double mean_length = adaiso::meanEdgeLength(mesh);
    const double ratio = 1.30;

    const std::vector<double> constant(mesh.vertexCount(), 3.0);
    const auto unchanged = adaiso::detail::applyDownwardLogGradationLimiter(
        mesh, constant, mean_length, ratio);
    test_support::require(unchanged.limited_lengths == constant,
                          "already feasible constant field changed");

    const std::vector<double> initial = {1.0, 100.0, 100.0, 100.0};
    const auto limited = adaiso::detail::applyDownwardLogGradationLimiter(
        mesh, initial, mean_length, ratio);
    test_support::require(limited.stats.changed_vertices > 0,
                          "limiter changed no vertices");
    test_support::require(limited.source_vertices.size() == initial.size(),
                          "limiter did not return source provenance");
    test_support::require(limited.normalized_source_distances.size() == initial.size(),
                          "limiter did not return source distances");
    test_support::require(limited.source_vertices[0] == 0,
                          "small-size seed lost its source identity");
    for (std::size_t i = 1; i < initial.size(); ++i) {
      test_support::require(limited.source_vertices[i] == 0,
                            "propagated vertex has the wrong source");
      test_support::require(limited.normalized_source_distances[i] > 0.0,
                            "propagated vertex has no source distance");
    }
    for (std::size_t i = 0; i < initial.size(); ++i) {
      test_support::require(limited.limited_lengths[i] <=
                                initial[i] * (1.0 + 1e-12),
                            "limiter increased a length");
    }
    test_support::require(limited.stats.after.max_normalized_log_slope <=
                              std::log(ratio) + 1e-12,
                          "limiter did not enforce the normalized log slope");

    std::vector<double> scaled(initial.size(), 0.0);
    std::transform(initial.begin(), initial.end(), scaled.begin(),
                   [](double value) { return 7.0 * value; });
    const auto scaled_limited =
        adaiso::detail::applyDownwardLogGradationLimiter(mesh, scaled,
                                                         mean_length, ratio);
    for (std::size_t i = 0; i < initial.size(); ++i) {
      test_support::requireNear(scaled_limited.limited_lengths[i],
                                7.0 * limited.limited_lengths[i], 1e-11,
                                "limiter is not scale equivariant");
    }

    adaiso::Mesh disconnected;
    disconnected.vertices = {
        {0.0, 0.0, 0.0},  {1.0, 0.0, 0.0},  {0.0, 1.0, 0.0},
        {10.0, 0.0, 0.0}, {11.0, 0.0, 0.0}, {10.0, 1.0, 0.0},
    };
    disconnected.faces = {{0, 1, 2}, {3, 4, 5}};
    const std::vector<double> disconnected_lengths = {1.0,  100.0, 100.0,
                                                      10.0, 10.0,  10.0};
    const auto disconnected_result =
        adaiso::detail::applyDownwardLogGradationLimiter(
            disconnected, disconnected_lengths,
            adaiso::meanEdgeLength(disconnected), ratio);
    for (std::size_t i = 3; i < disconnected_lengths.size(); ++i) {
      test_support::requireNear(disconnected_result.limited_lengths[i], 10.0,
                                1e-12,
                                "gradation crossed disconnected components");
    }

    bool rejected = false;
    try {
      static_cast<void>(adaiso::detail::applyDownwardLogGradationLimiter(
          mesh, initial, mean_length, 1.0));
    } catch (const std::runtime_error&) {
      rejected = true;
    }
    test_support::require(
        rejected,
        "limiter accepted a ratio that collapses components to one size");

    std::cout << "target_gradation_tests: passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "target_gradation_tests: " << error.what() << '\n';
    return 1;
  }
}
