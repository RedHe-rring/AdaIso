#include "adaiso/detail/ScreenedLaplacian.hpp"

#include "adaiso/Geometry.hpp"

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace adaiso::detail {
namespace {

using SparseMatrix = Eigen::SparseMatrix<double>;
using Triplet = Eigen::Triplet<double>;
using Clock = std::chrono::steady_clock;

double elapsedMs(Clock::time_point start, Clock::time_point end) {
  return std::chrono::duration<double, std::milli>(end - start).count();
}

double positiveFiniteWeight(double weight) {
  return std::isfinite(weight) && weight > 0.0 ? weight : kScreenedLaplacianEta;
}

void validateValues(const Mesh& mesh, const std::vector<double>& values) {
  mesh.validate();
  if (mesh.vertices.empty()) {
    throw std::runtime_error("screened scalar solve requires a non-empty mesh");
  }
  if (values.size() != mesh.vertices.size()) {
    throw std::runtime_error("screened scalar solve input size mismatch");
  }
  for (double value : values) {
    if (!std::isfinite(value)) {
      throw std::runtime_error("screened scalar solve input is not finite");
    }
  }
}

void validateAdjacency(const ScreenedAdjacency& adjacency) {
  if (adjacency.empty()) {
    throw std::runtime_error("screened scalar solve requires a non-empty adjacency");
  }
  bool has_edge = false;
  for (std::size_t i = 0; i < adjacency.size(); ++i) {
    for (const ScreenedNeighbor& neighbor : adjacency[i]) {
      has_edge = true;
      if (neighbor.vertex < 0 || static_cast<std::size_t>(neighbor.vertex) >= adjacency.size()) {
        throw std::runtime_error("screened adjacency contains an out-of-range vertex");
      }
      if (!std::isfinite(neighbor.weight) || neighbor.weight <= 0.0) {
        throw std::runtime_error("screened adjacency contains an invalid weight");
      }
    }
  }
  if (!has_edge) {
    throw std::runtime_error("screened scalar solve requires at least one mesh edge");
  }
}

std::vector<double> barycentricVertexMasses(const Mesh& mesh) {
  std::vector<double> masses(mesh.vertices.size(), 0.0);
  for (const auto& face : mesh.faces) {
    const double area = triangleArea(mesh.vertices[face[0]], mesh.vertices[face[1]], mesh.vertices[face[2]]);
    const double share = area / 3.0;
    masses[static_cast<std::size_t>(face[0])] += share;
    masses[static_cast<std::size_t>(face[1])] += share;
    masses[static_cast<std::size_t>(face[2])] += share;
  }
  for (double mass : masses) {
    if (!std::isfinite(mass) || mass <= 0.0) {
      throw std::runtime_error("screened scalar solve requires positive finite vertex masses");
    }
  }
  return masses;
}

double sparseInfinityNorm(const SparseMatrix& matrix) {
  std::vector<double> row_sums(static_cast<std::size_t>(matrix.rows()), 0.0);
  for (int outer = 0; outer < matrix.outerSize(); ++outer) {
    for (SparseMatrix::InnerIterator it(matrix, outer); it; ++it) {
      row_sums[static_cast<std::size_t>(it.row())] += std::abs(it.value());
    }
  }
  return row_sums.empty() ? 0.0 : *std::max_element(row_sums.begin(), row_sums.end());
}

}  // namespace

ScreenedAdjacency buildScreenedWeightGraph(const Mesh& mesh) {
  mesh.validate();
  ScreenedAdjacency adjacency(mesh.vertexCount());
  const auto weights = cotangentEdgeWeights(mesh, kScreenedLaplacianEta);
  for (const auto& item : weights) {
    const int a = item.first.first;
    const int b = item.first.second;
    if (a < 0 || b < 0 || static_cast<std::size_t>(a) >= mesh.vertices.size() ||
        static_cast<std::size_t>(b) >= mesh.vertices.size() || a == b) {
      continue;
    }
    const double weight = positiveFiniteWeight(item.second);
    adjacency[static_cast<std::size_t>(a)].push_back({b, weight});
    adjacency[static_cast<std::size_t>(b)].push_back({a, weight});
  }
  for (auto& neighbors : adjacency) {
    std::sort(neighbors.begin(), neighbors.end(), [](const ScreenedNeighbor& lhs, const ScreenedNeighbor& rhs) {
      return lhs.vertex < rhs.vertex;
    });
  }
  return adjacency;
}

ScreenedLaplacianResult solveScreenedScalarField(
    const Mesh& mesh,
    const std::vector<double>& initial_values,
    double smoothing_alpha,
    double residual_threshold) {
  validateValues(mesh, initial_values);
  if (!std::isfinite(smoothing_alpha) || smoothing_alpha <= 0.0) {
    throw std::runtime_error("screened scalar smoothing alpha must be positive");
  }
  const double mean_length = meanEdgeLength(mesh);
  if (!std::isfinite(mean_length) || mean_length <= 0.0) {
    throw std::runtime_error("screened scalar solve requires a positive mean edge length");
  }

  const ScreenedAdjacency adjacency = buildScreenedWeightGraph(mesh);
  validateAdjacency(adjacency);
  const std::vector<double> masses = barycentricVertexMasses(mesh);
  const double diffusion = smoothing_alpha * mean_length * mean_length;
  if (!std::isfinite(diffusion) || diffusion <= 0.0) {
    throw std::runtime_error("screened scalar solve produced an invalid diffusion strength");
  }

  ScreenedLaplacianResult result;
  const auto assembly_start = Clock::now();
  std::vector<Triplet> triplets;
  triplets.reserve(mesh.vertices.size() * 8);
  Eigen::VectorXd rhs = Eigen::VectorXd::Zero(static_cast<Eigen::Index>(mesh.vertices.size()));
  for (std::size_t i = 0; i < adjacency.size(); ++i) {
    double diagonal = masses[i];
    rhs[static_cast<Eigen::Index>(i)] = masses[i] * initial_values[i];
    for (const ScreenedNeighbor& neighbor : adjacency[i]) {
      diagonal += diffusion * neighbor.weight;
      triplets.emplace_back(static_cast<int>(i), neighbor.vertex, -diffusion * neighbor.weight);
    }
    if (!std::isfinite(diagonal) || diagonal <= 0.0) {
      throw std::runtime_error("screened scalar matrix has an invalid diagonal");
    }
    triplets.emplace_back(static_cast<int>(i), static_cast<int>(i), diagonal);
  }

  SparseMatrix matrix(
      static_cast<Eigen::Index>(mesh.vertices.size()),
      static_cast<Eigen::Index>(mesh.vertices.size()));
  matrix.setFromTriplets(triplets.begin(), triplets.end());
  matrix.makeCompressed();
  result.stats.matrix_nonzeros = static_cast<std::size_t>(matrix.nonZeros());
  const auto assembly_end = Clock::now();

  Eigen::SimplicialLDLT<SparseMatrix> solver;
  const auto factor_start = Clock::now();
  solver.compute(matrix);
  const auto factor_end = Clock::now();
  if (solver.info() != Eigen::Success) {
    throw std::runtime_error("screened scalar factorization failed");
  }

  const auto solve_start = Clock::now();
  const Eigen::VectorXd solution = solver.solve(rhs);
  const auto solve_end = Clock::now();
  if (solver.info() != Eigen::Success || !solution.allFinite()) {
    throw std::runtime_error("screened scalar solve failed");
  }

  const Eigen::VectorXd residual = matrix * solution - rhs;
  result.stats.residual_abs = residual.norm();
  const double matrix_inf = sparseInfinityNorm(matrix);
  const double solution_inf = solution.size() == 0 ? 0.0 : solution.cwiseAbs().maxCoeff();
  const double rhs_inf = rhs.size() == 0 ? 0.0 : rhs.cwiseAbs().maxCoeff();
  result.stats.residual_scaled =
      2.0 * result.stats.residual_abs /
      (matrix_inf * solution_inf + rhs_inf + std::numeric_limits<double>::epsilon());
  if (result.stats.residual_scaled > residual_threshold) {
    throw std::runtime_error(
        "screened scalar residual too high: " + std::to_string(result.stats.residual_scaled));
  }

  result.values.resize(static_cast<std::size_t>(solution.size()));
  for (Eigen::Index i = 0; i < solution.size(); ++i) {
    result.values[static_cast<std::size_t>(i)] = solution[i];
  }
  result.stats.assembly_time_ms = elapsedMs(assembly_start, assembly_end);
  result.stats.factorization_time_ms = elapsedMs(factor_start, factor_end);
  result.stats.solve_time_ms = elapsedMs(solve_start, solve_end);
  result.stats.solver_success = true;
  return result;
}

}  // namespace adaiso::detail
