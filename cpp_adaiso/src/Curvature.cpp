#include "adaiso/Curvature.hpp"

#include "adaiso/Geometry.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <iomanip>
#include <limits>
#include <stdexcept>

namespace adaiso {
namespace {

constexpr double kRangeEps = 1e-14;

std::uint8_t colorChannel(double value) {
  const int channel = std::clamp(static_cast<int>(std::lround(value)), 0, 255);
  return static_cast<std::uint8_t>(channel);
}

void validateFiniteScalar(double value) {
  if (!std::isfinite(value)) {
    throw std::runtime_error("curvature scalar is not finite");
  }
}

}  // namespace

std::vector<double> computeRawCurvature(const Mesh& mesh, Variant variant) {
  const auto normals = vertexNormals(mesh);
  const auto neigh = vertexNeighbors(mesh);
  std::vector<double> curvature(mesh.vertexCount(), 0.0);

  for (std::size_t i = 0; i < neigh.size(); ++i) {
    if (neigh[i].empty()) {
      continue;
    }
    double accum = 0.0;
    double max_angle = 0.0;
    int count = 0;
    for (int j : neigh[i]) {
      const double dot = std::clamp(normals[i].dot(normals[static_cast<std::size_t>(j)]), -1.0, 1.0);
      const double angle = std::acos(dot);
      accum += angle;
      max_angle = std::max(max_angle, angle);
      ++count;
    }
    curvature[i] = variant == Variant::Paper ? max_angle : accum / static_cast<double>(count);
  }

  return curvature;
}

CurvatureFields computeCurvatureFields(const Mesh& mesh, Variant variant) {
  CurvatureFields fields;
  fields.raw = computeRawCurvature(mesh, variant);
  fields.diffused = fields.raw;

  if (fields.raw.empty()) {
    fields.stats.solver_success = true;
    return fields;
  }

  const auto adjacency = detail::buildCsfWeightGraph(mesh, variant == Variant::Code);
  const auto solved = detail::solveDirichletCsf(adjacency, fields.raw);
  fields.diffused = solved.values;
  fields.stats = solved.stats;
  return fields;
}

ScalarRange scalarRange(const std::vector<double>& values) {
  if (values.empty()) {
    return {};
  }

  double min_value = std::numeric_limits<double>::infinity();
  double max_value = -std::numeric_limits<double>::infinity();
  for (double value : values) {
    validateFiniteScalar(value);
    min_value = std::min(min_value, value);
    max_value = std::max(max_value, value);
  }
  return {min_value, max_value};
}

Rgb8 blueGreenRedColor(double value, double min_value, double max_value) {
  validateFiniteScalar(value);
  validateFiniteScalar(min_value);
  validateFiniteScalar(max_value);
  if (max_value < min_value) {
    throw std::runtime_error("color range max is smaller than min");
  }
  if (max_value - min_value <= kRangeEps) {
    return {0, 255, 0};
  }

  const double t = std::clamp((value - min_value) / (max_value - min_value), 0.0, 1.0);
  if (t <= 0.5) {
    const double local = t * 2.0;
    return {0, colorChannel(255.0 * local), colorChannel(255.0 * (1.0 - local))};
  }

  const double local = (t - 0.5) * 2.0;
  return {colorChannel(255.0 * local), colorChannel(255.0 * (1.0 - local)), 0};
}

std::vector<Rgb8> colorizeScalars(
    const std::vector<double>& values,
    double min_value,
    double max_value) {
  std::vector<Rgb8> colors;
  colors.reserve(values.size());
  for (double value : values) {
    colors.push_back(blueGreenRedColor(value, min_value, max_value));
  }
  return colors;
}

void writeColoredAsciiPly(
    const std::filesystem::path& path,
    const Mesh& mesh,
    const std::vector<Rgb8>& colors) {
  if (colors.size() != mesh.vertices.size()) {
    throw std::runtime_error("colored PLY color count does not match vertex count");
  }
  if (path.has_parent_path()) {
    std::filesystem::create_directories(path.parent_path());
  }

  std::ofstream out(path);
  if (!out) {
    throw std::runtime_error("failed to write colored PLY file: " + path.string());
  }

  out << std::setprecision(17);
  out << "ply\n";
  out << "format ascii 1.0\n";
  out << "element vertex " << mesh.vertices.size() << '\n';
  out << "property float x\n";
  out << "property float y\n";
  out << "property float z\n";
  out << "property uchar red\n";
  out << "property uchar green\n";
  out << "property uchar blue\n";
  out << "element face " << mesh.faces.size() << '\n';
  out << "property list uchar int vertex_indices\n";
  out << "end_header\n";
  for (std::size_t i = 0; i < mesh.vertices.size(); ++i) {
    const auto& v = mesh.vertices[i];
    const Rgb8 color = colors[i];
    out << v[0] << ' ' << v[1] << ' ' << v[2] << ' '
        << static_cast<int>(color.r) << ' '
        << static_cast<int>(color.g) << ' '
        << static_cast<int>(color.b) << '\n';
  }
  for (const auto& f : mesh.faces) {
    out << "3 " << f[0] << ' ' << f[1] << ' ' << f[2] << '\n';
  }
}

}  // namespace adaiso
