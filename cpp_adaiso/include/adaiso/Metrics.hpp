#pragma once

#include "adaiso/Mesh.hpp"

#include <string>

namespace adaiso {

struct MeshSummary {
  std::size_t vertices = 0;
  std::size_t faces = 0;
  double qavg = 0.0;
  double qmin = 0.0;
};

[[nodiscard]] MeshSummary summarizeMesh(const Mesh& mesh);
[[nodiscard]] std::string summaryToJson(const MeshSummary& summary, int indent = 0);

}  // namespace adaiso
