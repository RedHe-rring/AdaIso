#include "adaiso/KDTree.hpp"

#include <nanoflann.hpp>

#include <algorithm>
#include <stdexcept>

namespace adaiso {
namespace {

struct PointCloudAdaptor {
  const std::vector<Eigen::Vector3d>* points = nullptr;

  [[nodiscard]] inline std::size_t kdtree_get_point_count() const {
    return points ? points->size() : 0;
  }

  [[nodiscard]] inline double kdtree_get_pt(const std::size_t idx, const std::size_t dim) const {
    return (*points)[idx][static_cast<int>(dim)];
  }

  template <class BBOX>
  bool kdtree_get_bbox(BBOX&) const {
    return false;
  }
};

using Distance = nanoflann::L2_Simple_Adaptor<double, PointCloudAdaptor>;
using Tree = nanoflann::KDTreeSingleIndexAdaptor<Distance, PointCloudAdaptor, 3>;

}  // namespace

struct NearestNeighborIndex::Impl {
  PointCloudAdaptor adaptor;
  std::unique_ptr<Tree> tree;
};

NearestNeighborIndex::NearestNeighborIndex() : impl_(std::make_unique<Impl>()) {}

NearestNeighborIndex::NearestNeighborIndex(const std::vector<Eigen::Vector3d>& points)
    : impl_(std::make_unique<Impl>()) {
  build(points);
}

NearestNeighborIndex::~NearestNeighborIndex() = default;
NearestNeighborIndex::NearestNeighborIndex(NearestNeighborIndex&&) noexcept = default;
NearestNeighborIndex& NearestNeighborIndex::operator=(NearestNeighborIndex&&) noexcept = default;

void NearestNeighborIndex::build(const std::vector<Eigen::Vector3d>& points) {
  impl_->adaptor.points = &points;
  impl_->tree = std::make_unique<Tree>(3, impl_->adaptor, nanoflann::KDTreeSingleIndexAdaptorParams(10));
  impl_->tree->buildIndex();
}

std::size_t NearestNeighborIndex::nearest(const Eigen::Vector3d& point) const {
  if (empty()) {
    throw std::runtime_error("nearest-neighbor query on empty point set");
  }
  std::size_t index = 0;
  double dist = 0.0;
  nanoflann::KNNResultSet<double> result(1);
  result.init(&index, &dist);
  const double query[3] = {point[0], point[1], point[2]};
  impl_->tree->findNeighbors(result, query, nanoflann::SearchParameters());
  return index;
}

std::vector<std::pair<std::size_t, double>> NearestNeighborIndex::nearestK(
    const Eigen::Vector3d& point,
    std::size_t k) const {
  if (empty()) {
    return {};
  }
  k = std::min(k, impl_->adaptor.kdtree_get_point_count());
  std::vector<std::size_t> indices(k);
  std::vector<double> distances(k);
  nanoflann::KNNResultSet<double> result(k);
  result.init(indices.data(), distances.data());
  const double query[3] = {point[0], point[1], point[2]};
  impl_->tree->findNeighbors(result, query, nanoflann::SearchParameters());

  std::vector<std::pair<std::size_t, double>> out;
  out.reserve(k);
  for (std::size_t i = 0; i < k; ++i) {
    out.emplace_back(indices[i], distances[i]);
  }
  return out;
}

bool NearestNeighborIndex::empty() const {
  return !impl_ || !impl_->tree || impl_->adaptor.kdtree_get_point_count() == 0;
}

}  // namespace adaiso
