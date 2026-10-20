#pragma once

#include "adaiso/Mesh.hpp"

#include <Eigen/Core>

#include <cmath>
#include <stdexcept>
#include <string>

namespace test_support {

inline void require(bool condition, const std::string& message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

inline void requireNear(double actual, double expected, double tolerance,
                        const std::string& message) {
  if (!std::isfinite(actual) || std::abs(actual - expected) > tolerance) {
    throw std::runtime_error(message + ": actual=" + std::to_string(actual) +
                             ", expected=" + std::to_string(expected));
  }
}

inline adaiso::Mesh squareMesh() {
  adaiso::Mesh mesh;
  mesh.vertices = {
      {0.0, 0.0, 0.0},
      {1.0, 0.0, 0.0},
      {1.0, 1.0, 0.0},
      {0.0, 1.0, 0.0},
  };
  mesh.faces = {{0, 1, 2}, {0, 2, 3}};
  return mesh;
}

inline adaiso::Mesh equilateralTriangleMesh() {
  adaiso::Mesh mesh;
  mesh.vertices = {
      {0.0, 0.0, 0.0},
      {1.0, 0.0, 0.0},
      {0.5, std::sqrt(3.0) * 0.5, 0.0},
  };
  mesh.faces = {{0, 1, 2}};
  return mesh;
}

}  // namespace test_support
