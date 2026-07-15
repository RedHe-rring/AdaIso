#pragma once

#include "adaiso/Mesh.hpp"

#include <string>
#include <vector>

namespace adaiso {

enum class CurvatureMode {
  NormalAngle,
  RobustMean,
};

[[nodiscard]] CurvatureMode parseCurvatureMode(const std::string& value);
[[nodiscard]] std::string toString(CurvatureMode mode);

namespace detail {

[[nodiscard]] std::vector<double> computeNormalAngleCurvatureField(
    const Mesh& mesh);
[[nodiscard]] std::vector<double> computeRobustMeanCurvatureField(
    const Mesh& mesh);
[[nodiscard]] std::vector<double> computeRawCurvatureField(
    const Mesh& mesh,
    CurvatureMode mode);

}  // namespace detail
}  // namespace adaiso
