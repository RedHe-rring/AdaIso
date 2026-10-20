#pragma once

#include "adaiso/Mesh.hpp"

#include <cstddef>
#include <string>

namespace adaiso {

struct MeshSummary {
  std::size_t vertices = 0;
  std::size_t faces = 0;
  double qavg = 0.0;
  double qmin = 0.0;
};

struct ChamferDistanceSummary {
  std::size_t samples_per_direction = 0;
  double output_to_gt_msd = 0.0;
  double gt_to_output_msd = 0.0;
  double chamfer_distance = 0.0;
  double symmetric_rmse = 0.0;
};

[[nodiscard]] MeshSummary summarizeMesh(const Mesh& mesh);
[[nodiscard]] std::string summaryToJson(const MeshSummary& summary, int indent = 0);
[[nodiscard]] ChamferDistanceSummary areaSampledChamferDistance(
    const Mesh& output,
    const Mesh& gt,
    std::size_t samples_per_direction);

}  // namespace adaiso
