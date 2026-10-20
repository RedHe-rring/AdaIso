#include "TestSupport.hpp"

#include "adaiso/detail/ScreenedLaplacian.hpp"

#include <algorithm>
#include <iostream>
#include <vector>

int main() {
  try {
    const adaiso::Mesh mesh = test_support::squareMesh();

    const std::vector<double> constant(mesh.vertexCount(), 2.5);
    const auto constant_result =
        adaiso::detail::solveScreenedScalarField(mesh, constant, 0.75);
    test_support::require(constant_result.stats.solver_success,
                          "constant solve did not succeed");
    for (double value : constant_result.values) {
      test_support::requireNear(value, 2.5, 1e-12,
                                "constant field was not preserved");
    }

    const std::vector<double> impulse = {1.0, 0.0, 0.0, 0.0};
    const auto impulse_result =
        adaiso::detail::solveScreenedScalarField(mesh, impulse, 0.75);
    const double maximum = *std::max_element(impulse_result.values.begin(),
                                             impulse_result.values.end());
    test_support::require(maximum < 1.0, "impulse peak was not reduced");
    for (double value : impulse_result.values) {
      test_support::require(value >= -1e-12,
                            "non-negative field produced a negative value");
    }

    bool rejected = false;
    try {
      static_cast<void>(
          adaiso::detail::solveScreenedScalarField(mesh, impulse, 0.0));
    } catch (const std::runtime_error&) {
      rejected = true;
    }
    test_support::require(
        rejected,
        "solver accepted zero alpha instead of leaving bypass to the caller");

    std::cout << "screened_laplacian_tests: passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "screened_laplacian_tests: " << error.what() << '\n';
    return 1;
  }
}
