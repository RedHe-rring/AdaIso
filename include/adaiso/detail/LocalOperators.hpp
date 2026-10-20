#pragma once

#include "adaiso/Geometry.hpp"
#include "adaiso/Mesh.hpp"

#include <Eigen/Core>

#include <unordered_set>
#include <vector>

namespace adaiso::detail {

inline constexpr double kCollapseFeatureOmega = 0.15;

[[nodiscard]] Eigen::Vector3d featureAwareCollapsePoint(
    const Mesh& mesh,
    int a,
    int b,
    const std::vector<double>& feature_strengths);

[[nodiscard]] bool flipImprovesValence(
    const std::vector<int>& valence,
    const std::unordered_set<int>& boundary,
    int a,
    int b,
    int c,
    int d);

}  // namespace adaiso::detail
