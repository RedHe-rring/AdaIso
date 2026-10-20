#pragma once

#include <Eigen/Core>

#include <cstddef>
#include <memory>
#include <vector>

namespace adaiso {

class NearestNeighborIndex {
 public:
  NearestNeighborIndex();
  explicit NearestNeighborIndex(const std::vector<Eigen::Vector3d>& points);
  ~NearestNeighborIndex();

  NearestNeighborIndex(const NearestNeighborIndex&) = delete;
  NearestNeighborIndex& operator=(const NearestNeighborIndex&) = delete;
  NearestNeighborIndex(NearestNeighborIndex&&) noexcept;
  NearestNeighborIndex& operator=(NearestNeighborIndex&&) noexcept;

  void build(const std::vector<Eigen::Vector3d>& points);
  [[nodiscard]] std::size_t nearest(const Eigen::Vector3d& point) const;
  [[nodiscard]] std::vector<std::pair<std::size_t, double>> nearestK(
      const Eigen::Vector3d& point,
      std::size_t k) const;
  [[nodiscard]] bool empty() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace adaiso
