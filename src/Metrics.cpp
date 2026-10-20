#include "adaiso/Metrics.hpp"

#include "adaiso/Geometry.hpp"
#include "adaiso/KDTree.hpp"

#include <Eigen/Geometry>

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace adaiso {
namespace {

struct AreaFace {
  int face_index = 0;
  double cumulative_area = 0.0;
};

double radicalInverse(std::size_t index, unsigned int base) {
  double result = 0.0;
  double factor = 1.0 / static_cast<double>(base);
  while (index > 0) {
    result += factor * static_cast<double>(index % base);
    index /= base;
    factor /= static_cast<double>(base);
  }
  return result;
}

std::vector<AreaFace> buildAreaCdf(const Mesh& mesh, double& total_area) {
  total_area = 0.0;
  std::vector<AreaFace> cdf;
  cdf.reserve(mesh.faces.size());
  for (std::size_t i = 0; i < mesh.faces.size(); ++i) {
    const auto& f = mesh.faces[i];
    const Eigen::Vector3d e0 = mesh.vertices[static_cast<std::size_t>(f[1])] - mesh.vertices[static_cast<std::size_t>(f[0])];
    const Eigen::Vector3d e1 = mesh.vertices[static_cast<std::size_t>(f[2])] - mesh.vertices[static_cast<std::size_t>(f[0])];
    const double area = 0.5 * e0.cross(e1).norm();
    if (std::isfinite(area) && area > 0.0) {
      total_area += area;
      cdf.push_back({static_cast<int>(i), total_area});
    }
  }
  if (cdf.empty() || !(total_area > 0.0)) {
    throw std::runtime_error("cannot compute area-sampled CD: mesh has no positive-area triangles");
  }
  return cdf;
}

Eigen::Vector3d sampleTriangle(const Mesh& mesh, int face_index, double u, double v) {
  if (u + v > 1.0) {
    u = 1.0 - u;
    v = 1.0 - v;
  }
  const auto& face = mesh.faces[static_cast<std::size_t>(face_index)];
  const Eigen::Vector3d& a = mesh.vertices[static_cast<std::size_t>(face[0])];
  const Eigen::Vector3d& b = mesh.vertices[static_cast<std::size_t>(face[1])];
  const Eigen::Vector3d& c = mesh.vertices[static_cast<std::size_t>(face[2])];
  return a + u * (b - a) + v * (c - a);
}

std::vector<Eigen::Vector3d> sampleMeshByArea(const Mesh& mesh, std::size_t samples) {
  if (samples == 0) {
    throw std::runtime_error("area-sampled CD requires a positive sample count");
  }

  double total_area = 0.0;
  const std::vector<AreaFace> cdf = buildAreaCdf(mesh, total_area);

  std::vector<Eigen::Vector3d> points;
  points.reserve(samples);
  for (std::size_t i = 0; i < samples; ++i) {
    const std::size_t seq = i + 1;
    const double area_position = radicalInverse(seq, 2) * total_area;
    const auto it = std::lower_bound(cdf.begin(), cdf.end(), area_position, [](const AreaFace& item, double value) {
      return item.cumulative_area < value;
    });
    const int face_index = (it == cdf.end()) ? cdf.back().face_index : it->face_index;
    points.push_back(sampleTriangle(mesh, face_index, radicalInverse(seq, 3), radicalInverse(seq, 5)));
  }
  return points;
}

double meanSquaredNearestDistance(
    const std::vector<Eigen::Vector3d>& query_points,
    const std::vector<Eigen::Vector3d>& target_points) {
  NearestNeighborIndex index(target_points);
  double sum = 0.0;
  for (const auto& point : query_points) {
    const auto nearest = index.nearestK(point, 1);
    if (nearest.empty()) {
      throw std::runtime_error("cannot compute area-sampled CD: empty sampled target point set");
    }
    sum += nearest.front().second;
  }
  return sum / static_cast<double>(query_points.size());
}

}  // namespace

MeshSummary summarizeMesh(const Mesh& mesh) {
  MeshSummary summary;
  summary.vertices = mesh.vertexCount();
  summary.faces = mesh.faceCount();
  const auto qualities = triangleQualities(mesh);
  if (qualities.empty()) {
    summary.qavg = 0.0;
    summary.qmin = 0.0;
    return summary;
  }
  double sum = 0.0;
  for (double q : qualities) {
    sum += q;
  }
  summary.qavg = sum / static_cast<double>(qualities.size());
  summary.qmin = *std::min_element(qualities.begin(), qualities.end());
  return summary;
}

std::string summaryToJson(const MeshSummary& summary, int indent) {
  const std::string pad(static_cast<std::size_t>(indent), ' ');
  const std::string pad2(static_cast<std::size_t>(indent + 2), ' ');
  std::ostringstream out;
  out << std::setprecision(10);
  out << pad << "{\n";
  out << pad2 << "\"vertices\": " << summary.vertices << ",\n";
  out << pad2 << "\"faces\": " << summary.faces << ",\n";
  out << pad2 << "\"qavg\": " << summary.qavg << ",\n";
  out << pad2 << "\"qmin\": " << summary.qmin << "\n";
  out << pad << "}";
  return out.str();
}

ChamferDistanceSummary areaSampledChamferDistance(
    const Mesh& output,
    const Mesh& gt,
    std::size_t samples_per_direction) {
  if (samples_per_direction == 0) {
    throw std::runtime_error("area-sampled CD requires a positive sample count");
  }

  const std::vector<Eigen::Vector3d> output_points = sampleMeshByArea(output, samples_per_direction);
  const std::vector<Eigen::Vector3d> gt_points = sampleMeshByArea(gt, samples_per_direction);

  ChamferDistanceSummary summary;
  summary.samples_per_direction = samples_per_direction;
  summary.output_to_gt_msd = meanSquaredNearestDistance(output_points, gt_points);
  summary.gt_to_output_msd = meanSquaredNearestDistance(gt_points, output_points);
  summary.chamfer_distance = summary.output_to_gt_msd + summary.gt_to_output_msd;
  summary.symmetric_rmse = std::sqrt(0.5 * summary.chamfer_distance);
  return summary;
}

}  // namespace adaiso
