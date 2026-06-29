#include "adaiso/detail/CsfSolver.hpp"

#include "adaiso/Geometry.hpp"

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
  return std::isfinite(weight) && weight > 0.0 ? weight : kCsfEta;
}

void validateInputs(const CsfAdjacency& adjacency, const std::vector<double>& initial_values) {
  if (adjacency.size() != initial_values.size()) {
    throw std::runtime_error("CSF solve input size mismatch");
  }
  for (std::size_t i = 0; i < initial_values.size(); ++i) {
    if (!std::isfinite(initial_values[i])) {
      throw std::runtime_error("CSF initial value is not finite");
    }
    for (const CsfNeighbor& neighbor : adjacency[i]) {
      if (neighbor.vertex < 0 || static_cast<std::size_t>(neighbor.vertex) >= adjacency.size()) {
        throw std::runtime_error("CSF adjacency contains an out-of-range vertex");
      }
      if (!std::isfinite(neighbor.weight) || neighbor.weight <= 0.0) {
        throw std::runtime_error("CSF adjacency contains a non-positive or non-finite weight");
      }
    }
  }
}

std::vector<std::vector<int>> connectedComponents(const CsfAdjacency& adjacency) {
  std::vector<std::vector<int>> components;
  std::vector<char> visited(adjacency.size(), 0);

  for (std::size_t start = 0; start < adjacency.size(); ++start) {
    if (visited[start]) {
      continue;
    }
    std::vector<int> component;
    std::vector<int> stack = {static_cast<int>(start)};
    visited[start] = 1;

    while (!stack.empty()) {
      const int v = stack.back();
      stack.pop_back();
      component.push_back(v);
      for (const CsfNeighbor& neighbor : adjacency[static_cast<std::size_t>(v)]) {
        const int n = neighbor.vertex;
        if (!visited[static_cast<std::size_t>(n)]) {
          visited[static_cast<std::size_t>(n)] = 1;
          stack.push_back(n);
        }
      }
    }

    std::sort(component.begin(), component.end());
    components.push_back(std::move(component));
  }
  return components;
}

bool componentHasConstantValue(const std::vector<int>& component, const std::vector<double>& values) {
  double min_value = std::numeric_limits<double>::infinity();
  double max_value = -std::numeric_limits<double>::infinity();
  for (int v : component) {
    const double value = values[static_cast<std::size_t>(v)];
    min_value = std::min(min_value, value);
    max_value = std::max(max_value, value);
  }
  return max_value - min_value <= 1e-14;
}

double quantileOfSorted(const std::vector<double>& sorted, double q) {
  if (sorted.empty()) {
    return 0.0;
  }
  if (sorted.size() == 1) {
    return sorted.front();
  }
  q = std::clamp(q, 0.0, 1.0);
  const double pos = q * static_cast<double>(sorted.size() - 1);
  const auto lo = static_cast<std::size_t>(std::floor(pos));
  const auto hi = static_cast<std::size_t>(std::ceil(pos));
  const double t = pos - static_cast<double>(lo);
  return sorted[lo] * (1.0 - t) + sorted[hi] * t;
}

double componentStability(
    int vertex,
    const CsfAdjacency& adjacency,
    const std::vector<double>& values,
    double component_range) {
  const auto& neighbors = adjacency[static_cast<std::size_t>(vertex)];
  double sum_weight = 0.0;
  double sum_value = 0.0;
  for (const CsfNeighbor& neighbor : neighbors) {
    sum_weight += neighbor.weight;
    sum_value += neighbor.weight * values[static_cast<std::size_t>(neighbor.vertex)];
  }
  if (sum_weight <= 0.0 || !std::isfinite(sum_weight)) {
    return std::numeric_limits<double>::infinity();
  }

  const double average = sum_value / sum_weight;
  return std::abs(values[static_cast<std::size_t>(vertex)] - average) / (component_range + 1e-14);
}

bool stableAnchorLess(
    int candidate,
    int current,
    const std::vector<double>& values,
    const std::vector<double>& stability,
    bool high_anchor) {
  if (current < 0) {
    return true;
  }

  const double candidate_stability = stability[static_cast<std::size_t>(candidate)];
  const double current_stability = stability[static_cast<std::size_t>(current)];
  if (candidate_stability < current_stability) {
    return true;
  }
  if (candidate_stability > current_stability) {
    return false;
  }

  const double candidate_value = values[static_cast<std::size_t>(candidate)];
  const double current_value = values[static_cast<std::size_t>(current)];
  if (high_anchor) {
    if (candidate_value > current_value) {
      return true;
    }
    if (candidate_value < current_value) {
      return false;
    }
  } else {
    if (candidate_value < current_value) {
      return true;
    }
    if (candidate_value > current_value) {
      return false;
    }
  }

  return candidate < current;
}

template <typename CandidatePredicate>
int selectStableAnchor(
    const std::vector<int>& component,
    const std::vector<double>& values,
    const std::vector<double>& stability,
    CandidatePredicate is_candidate,
    bool high_anchor) {
  int best = -1;
  for (int v : component) {
    if (!is_candidate(v)) {
      continue;
    }
    if (stableAnchorLess(v, best, values, stability, high_anchor)) {
      best = v;
    }
  }
  return best;
}

std::vector<char> selectFixedVertices(const CsfAdjacency& adjacency, const std::vector<double>& values) {
  std::vector<char> fixed(values.size(), 0);
  for (const std::vector<int>& component : connectedComponents(adjacency)) {
    if (component.empty()) {
      continue;
    }
    if (component.size() == 1 || componentHasConstantValue(component, values)) {
      for (int v : component) {
        fixed[static_cast<std::size_t>(v)] = 1;
      }
      continue;
    }
    if (component.size() == 2) {
      fixed[static_cast<std::size_t>(component[0])] = 1;
      fixed[static_cast<std::size_t>(component[1])] = 1;
      continue;
    }

    std::vector<double> sorted_values;
    sorted_values.reserve(component.size());
    for (int v : component) {
      sorted_values.push_back(values[static_cast<std::size_t>(v)]);
    }
    std::sort(sorted_values.begin(), sorted_values.end());
    const double min_value = sorted_values.front();
    const double max_value = sorted_values.back();
    const double range = max_value - min_value;
    const double q30 = quantileOfSorted(sorted_values, 0.30);
    const double q70 = quantileOfSorted(sorted_values, 0.70);

    std::vector<double> stability(values.size(), std::numeric_limits<double>::infinity());
    for (int v : component) {
      stability[static_cast<std::size_t>(v)] = componentStability(v, adjacency, values, range);
    }

    const int vlow = selectStableAnchor(
        component,
        values,
        stability,
        [&](int v) {
          return values[static_cast<std::size_t>(v)] <= q30;
        },
        false);
    if (vlow < 0) {
      throw std::runtime_error("CSF anchor selection failed");
    }

    int vhigh = selectStableAnchor(
        component,
        values,
        stability,
        [&](int v) {
          return values[static_cast<std::size_t>(v)] >= q70;
        },
        true);

    const double low_value = values[static_cast<std::size_t>(vlow)];
    if (vhigh < 0 || vhigh == vlow || values[static_cast<std::size_t>(vhigh)] <= low_value + 1e-14) {
      vhigh = selectStableAnchor(
          component,
          values,
          stability,
          [&](int v) {
            return values[static_cast<std::size_t>(v)] > low_value + 1e-14;
          },
          true);
    }
    if (vhigh < 0) {
      for (int v : component) {
        fixed[static_cast<std::size_t>(v)] = 1;
      }
      continue;
    }

    fixed[static_cast<std::size_t>(vlow)] = 1;
    fixed[static_cast<std::size_t>(vhigh)] = 1;
  }
  return fixed;
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

CsfAdjacency buildCsfWeightGraph(const Mesh& mesh, bool length_scaled) {
  CsfAdjacency adjacency(mesh.vertexCount());
  const auto weights = cotangentEdgeWeights(mesh, kCsfEta);
  for (const auto& item : weights) {
    const int a = item.first.first;
    const int b = item.first.second;
    if (a < 0 || b < 0 || static_cast<std::size_t>(a) >= mesh.vertices.size() ||
        static_cast<std::size_t>(b) >= mesh.vertices.size() || a == b) {
      continue;
    }

    double weight = positiveFiniteWeight(item.second);
    if (length_scaled) {
      const double length = (mesh.vertices[static_cast<std::size_t>(a)] - mesh.vertices[static_cast<std::size_t>(b)]).norm();
      weight /= std::max(length, kMinEdgeLength);
      weight = positiveFiniteWeight(weight);
    }

    adjacency[static_cast<std::size_t>(a)].push_back({b, weight});
    adjacency[static_cast<std::size_t>(b)].push_back({a, weight});
  }

  for (auto& neighbors : adjacency) {
    std::sort(neighbors.begin(), neighbors.end(), [](const CsfNeighbor& lhs, const CsfNeighbor& rhs) {
      return lhs.vertex < rhs.vertex;
    });
  }
  return adjacency;
}

CsfSolveResult solveDirichletCsf(
    const CsfAdjacency& adjacency,
    const std::vector<double>& initial_values,
    double residual_threshold) {
  validateInputs(adjacency, initial_values);

  CsfSolveResult result;
  result.values = initial_values;

  if (initial_values.empty()) {
    result.stats.solver_success = true;
    return result;
  }

  const std::vector<char> fixed = selectFixedVertices(adjacency, initial_values);
  std::vector<int> free_index(initial_values.size(), -1);
  for (std::size_t i = 0; i < initial_values.size(); ++i) {
    if (fixed[i]) {
      ++result.stats.fixed_vertices;
    } else {
      free_index[i] = static_cast<int>(result.stats.free_vertices++);
    }
  }

  if (result.stats.free_vertices == 0) {
    result.stats.solver_success = true;
    return result;
  }

  const auto assembly_start = Clock::now();
  std::vector<Triplet> triplets;
  triplets.reserve(result.stats.free_vertices * 8);
  Eigen::VectorXd rhs = Eigen::VectorXd::Zero(static_cast<Eigen::Index>(result.stats.free_vertices));

  for (std::size_t i = 0; i < adjacency.size(); ++i) {
    const int row = free_index[i];
    if (row < 0) {
      continue;
    }

    double diagonal = 0.0;
    for (const CsfNeighbor& neighbor : adjacency[i]) {
      const double weight = neighbor.weight;
      diagonal += weight;
      const int col = free_index[static_cast<std::size_t>(neighbor.vertex)];
      if (col >= 0) {
        triplets.emplace_back(row, col, -weight);
      } else {
        rhs[row] += weight * initial_values[static_cast<std::size_t>(neighbor.vertex)];
      }
    }
    if (diagonal <= 0.0 || !std::isfinite(diagonal)) {
      throw std::runtime_error("CSF free vertex has no positive finite Laplace diagonal");
    }
    triplets.emplace_back(row, row, diagonal);
  }

  SparseMatrix matrix(
      static_cast<Eigen::Index>(result.stats.free_vertices),
      static_cast<Eigen::Index>(result.stats.free_vertices));
  matrix.setFromTriplets(triplets.begin(), triplets.end());
  matrix.makeCompressed();
  result.stats.matrix_nonzeros = static_cast<std::size_t>(matrix.nonZeros());
  const auto assembly_end = Clock::now();

  Eigen::SimplicialLDLT<SparseMatrix> solver;
  const auto factor_start = Clock::now();
  solver.compute(matrix);
  const auto factor_end = Clock::now();
  if (solver.info() != Eigen::Success) {
    throw std::runtime_error("CSF Laplace factorization failed");
  }

  const auto solve_start = Clock::now();
  const Eigen::VectorXd solution = solver.solve(rhs);
  const auto solve_end = Clock::now();
  if (solver.info() != Eigen::Success) {
    throw std::runtime_error("CSF Laplace solve failed");
  }
  if (!solution.allFinite()) {
    throw std::runtime_error("CSF Laplace solve produced non-finite values");
  }

  const Eigen::VectorXd residual = matrix * solution - rhs;
  result.stats.residual_abs = residual.norm();
  const double matrix_inf = sparseInfinityNorm(matrix);
  const double solution_inf = solution.size() == 0 ? 0.0 : solution.cwiseAbs().maxCoeff();
  const double rhs_inf = rhs.size() == 0 ? 0.0 : rhs.cwiseAbs().maxCoeff();
  result.stats.residual_scaled =
      2.0 * result.stats.residual_abs / (matrix_inf * solution_inf + rhs_inf + std::numeric_limits<double>::epsilon());
  if (result.stats.residual_scaled > residual_threshold) {
    throw std::runtime_error(
        "CSF Laplace residual too high: " + std::to_string(result.stats.residual_scaled));
  }

  for (std::size_t i = 0; i < free_index.size(); ++i) {
    const int idx = free_index[i];
    if (idx >= 0) {
      result.values[i] = solution[idx];
    }
  }

  result.stats.assembly_time_ms = elapsedMs(assembly_start, assembly_end);
  result.stats.factorization_time_ms = elapsedMs(factor_start, factor_end);
  result.stats.solve_time_ms = elapsedMs(solve_start, solve_end);
  result.stats.solver_success = true;
  return result;
}

}  // namespace adaiso::detail
