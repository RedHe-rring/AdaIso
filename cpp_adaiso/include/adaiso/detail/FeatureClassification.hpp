#pragma once

#include "adaiso/Mesh.hpp"

#include <vector>

namespace adaiso::detail {

enum class VertexRelocationClass {
  Corner,
  Crease,
  Smooth
};

[[nodiscard]] std::vector<double> computeVertexFeatureStrengths(const Mesh& mesh);

[[nodiscard]] std::vector<VertexRelocationClass> classifyVerticesForRelocation(const Mesh& mesh);

}  // namespace adaiso::detail
