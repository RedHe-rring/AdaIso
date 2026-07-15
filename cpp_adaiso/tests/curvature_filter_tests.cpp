#include "TestSupport.hpp"

#include "adaiso/Geometry.hpp"
#include "adaiso/detail/CurvatureFilter.hpp"

#include <iostream>

int main() {
  try {
    const adaiso::Mesh mesh = test_support::squareMesh();
    adaiso::detail::PrincipalCurvatureResult principal;
    principal.kappa = {1000.0, 1.0, 2.0, 3.0};
    principal.fit_diagnostics.resize(mesh.vertexCount());
    for (auto& diagnostics : principal.fit_diagnostics) {
      diagnostics.local_scale = 1.0;
      diagnostics.fallback = false;
    }
    const double mean_length = adaiso::meanEdgeLength(mesh);

    const auto none = adaiso::detail::filterCurvatureOutliers(
        mesh, principal, mean_length, adaiso::CurvatureOutlierMode::None, 1,
        5.0);
    test_support::require(none.filtered_kappa == principal.kappa,
                          "none filter changed the field");

    const auto topk = adaiso::detail::filterCurvatureOutliers(
        mesh, principal, mean_length,
        adaiso::CurvatureOutlierMode::TopKMedian, 1, 5.0);
    test_support::require(topk.changed_vertices == 1,
                          "top-k filter changed the wrong number of vertices");
    test_support::require(topk.filtered_kappa[0] < principal.kappa[0],
                          "top-k filter did not remove the largest spike");
    test_support::require(topk.filtered_kappa[1] == principal.kappa[1],
                          "top-k filter changed an unselected vertex");

    const auto capped = adaiso::detail::filterCurvatureOutliers(
        mesh, principal, mean_length,
        adaiso::CurvatureOutlierMode::SamplingCap, 1, 2.0);
    test_support::require(capped.changed_vertices > 0,
                          "sampling cap changed no vertices");
    for (std::size_t i = 0; i < capped.filtered_kappa.size(); ++i) {
      test_support::require(
          capped.filtered_kappa[i] <= capped.applied_caps[i] + 1e-12,
          "sampling cap was not enforced");
    }

    std::cout << "curvature_filter_tests: passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "curvature_filter_tests: " << error.what() << '\n';
    return 1;
  }
}
