#include "TestSupport.hpp"

#include "adaiso/detail/FeatureClassification.hpp"
#include "adaiso/detail/LocalOperators.hpp"

#include <Eigen/Core>

#include <iostream>
#include <unordered_set>
#include <vector>

int main() {
  try {
    const adaiso::Mesh mesh = test_support::equilateralTriangleMesh();

    const std::vector<double> equal_features = {1.0, 1.05, 0.0};
    const Eigen::Vector3d midpoint = 0.5 * (mesh.vertices[0] + mesh.vertices[1]);
    const Eigen::Vector3d equal_point =
        adaiso::detail::featureAwareCollapsePoint(mesh, 0, 1, equal_features);
    test_support::require(
        (equal_point - midpoint).norm() < 1e-12,
        "feature-aware collapse should keep midpoint for similar strengths");

    const std::vector<double> strong_features = {3.0, 1.0, 0.0};
    const Eigen::Vector3d strong_point =
        adaiso::detail::featureAwareCollapsePoint(mesh, 0, 1, strong_features);
    test_support::require(
        (strong_point - mesh.vertices[0]).norm() < 1e-12,
        "feature-aware collapse did not preserve stronger endpoint");

    const auto strengths = adaiso::detail::computeVertexFeatureStrengths(mesh);
    test_support::require(
        strengths.size() == mesh.vertexCount(),
        "feature strength output size mismatch");
    for (double value : strengths) {
      test_support::require(std::isfinite(value) && value >= 0.0, "feature strength is invalid");
    }

    const std::vector<int> valence = {7, 7, 5, 5};
    const std::unordered_set<int> boundary;
    test_support::require(
        adaiso::detail::flipImprovesValence(valence, boundary, 0, 1, 2, 3),
        "flip valence test should accept an improving flip");

    std::cout << "local_operators_tests: passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "local_operators_tests: " << error.what() << '\n';
    return 1;
  }
}
