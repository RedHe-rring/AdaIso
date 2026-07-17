#include "adaiso/detail/LocalOperators.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>

namespace adaiso::detail {

Eigen::Vector3d featureAwareCollapsePoint(
    const Mesh& mesh,
    int a,
    int b,
    const std::vector<double>& feature_strengths) {
  const Eigen::Vector3d midpoint =
      0.5 * (mesh.vertices[static_cast<std::size_t>(a)] + mesh.vertices[static_cast<std::size_t>(b)]);
  if (feature_strengths.size() != mesh.vertices.size()) {
    return midpoint;
  }

  const double fa = feature_strengths[static_cast<std::size_t>(a)];
  const double fb = feature_strengths[static_cast<std::size_t>(b)];
  if (!std::isfinite(fa) || !std::isfinite(fb)) {
    return midpoint;
  }

  const double fmax = std::max(fa, fb);
  constexpr double kFeatureEps = 1e-12;
  if (fmax <= kFeatureEps || std::abs(fa - fb) < kCollapseFeatureOmega * fmax) {
    return midpoint;
  }
  return fa > fb ? mesh.vertices[static_cast<std::size_t>(a)] : mesh.vertices[static_cast<std::size_t>(b)];
}

bool flipImprovesValence(
    const std::vector<int>& valence,
    const std::unordered_set<int>& boundary,
    int a,
    int b,
    int c,
    int d) {
  auto target = [&](int v) {
    return boundary.find(v) == boundary.end() ? 6 : 4;
  };
  const int before = std::abs(valence[static_cast<std::size_t>(a)] - target(a)) +
                     std::abs(valence[static_cast<std::size_t>(b)] - target(b)) +
                     std::abs(valence[static_cast<std::size_t>(c)] - target(c)) +
                     std::abs(valence[static_cast<std::size_t>(d)] - target(d));
  const int after = std::abs(valence[static_cast<std::size_t>(a)] - 1 - target(a)) +
                    std::abs(valence[static_cast<std::size_t>(b)] - 1 - target(b)) +
                    std::abs(valence[static_cast<std::size_t>(c)] + 1 - target(c)) +
                    std::abs(valence[static_cast<std::size_t>(d)] + 1 - target(d));
  return after < before;
}

}  // namespace adaiso::detail
