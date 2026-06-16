#include "adaiso/Metrics.hpp"

#include "adaiso/Geometry.hpp"

#include <algorithm>
#include <iomanip>
#include <sstream>

namespace adaiso {

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

}  // namespace adaiso
