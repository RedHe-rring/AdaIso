#include "TestSupport.hpp"

#include "adaiso/AdaIsoRemesher.hpp"
#include "adaiso/Geometry.hpp"
#include "adaiso/detail/TargetFaceSizing.hpp"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <string>
#include <vector>

namespace {

adaiso::Mesh repeatedEquilateralTriangles(int count) {
  adaiso::Mesh mesh;
  const double height = std::sqrt(3.0) * 0.5;
  for (int i = 0; i < count; ++i) {
    const int base = static_cast<int>(mesh.vertices.size());
    const double offset = static_cast<double>(i) * 3.0;
    mesh.vertices.push_back({offset, 0.0, 0.0});
    mesh.vertices.push_back({offset + 1.0, 0.0, 0.0});
    mesh.vertices.push_back({offset + 0.5, height, 0.0});
    mesh.faces.push_back({base, base + 1, base + 2});
  }
  return mesh;
}

std::size_t countLines(const std::filesystem::path& path) {
  std::ifstream in(path);
  if (!in) {
    throw std::runtime_error("failed to open file: " + path.string());
  }
  std::size_t lines = 0;
  std::string line;
  while (std::getline(in, line)) {
    ++lines;
  }
  return lines;
}

}  // namespace

int main() {
  try {
    const adaiso::Mesh mesh = test_support::equilateralTriangleMesh();
    const double mean_length = adaiso::meanEdgeLength(mesh);

    const std::vector<double> kappa(mesh.vertexCount(), 1.0);
    const auto sizing = adaiso::detail::targetFaceCurvatureLengths(
        mesh, kappa, 1, mean_length);
    test_support::requireNear(
        sizing.estimated_faces, 1.0, 1e-12,
        "target-face sizing did not recover the requested estimate");
    for (double value : sizing.target_lengths) {
      test_support::requireNear(
          value, mean_length, 1e-12,
          "unit CSF curvature did not calibrate to mean edge length");
    }

    const auto dense = adaiso::detail::targetFaceCurvatureLengths(
        mesh, kappa, 1000, mean_length);
    for (double value : dense.target_lengths) {
      test_support::requireNear(
          value,
          adaiso::detail::kTargetMinEdgeFactor * mean_length,
          1e-12,
          "unreachable dense target did not clamp to h_min");
    }

    const adaiso::Mesh many = repeatedEquilateralTriangles(30);
    const std::vector<double> many_kappa(many.vertexCount(), 1.0);
    const auto coarse = adaiso::detail::targetFaceCurvatureLengths(
        many, many_kappa, 1, adaiso::meanEdgeLength(many));
    for (double value : coarse.target_lengths) {
      test_support::requireNear(
          value,
          adaiso::detail::kTargetMaxEdgeFactor * adaiso::meanEdgeLength(many),
          1e-12,
          "unreachable coarse target did not clamp to h_max");
    }

    bool rejected = false;
    try {
      std::vector<double> bad = kappa;
      bad[0] = std::numeric_limits<double>::infinity();
      static_cast<void>(adaiso::detail::targetFaceCurvatureLengths(
          mesh, bad, 1, mean_length));
    } catch (const std::runtime_error&) {
      rejected = true;
    }
    test_support::require(rejected, "non-finite curvature was not rejected");

    std::vector<double> capped_input(1000, 1.0);
    capped_input.back() = 1000.0;
    const auto capped = adaiso::detail::capCurvatureQuantile(capped_input, 0.995);
    test_support::requireNear(
        capped.cap_value, 1.0, 1e-12,
        "curvature cap did not clamp isolated peak to the quantile");
    test_support::requireNear(
        capped.changed_rate, 0.001, 1e-12,
        "curvature cap changed rate is incorrect");
    test_support::requireNear(
        capped.curvature.back(), 1.0, 1e-12,
        "curvature cap did not replace the peak value");

    const std::vector<double> constant_curvature(8, 2.5);
    const auto constant_cap = adaiso::detail::capCurvatureQuantile(constant_curvature, 0.995);
    test_support::requireNear(
        constant_cap.cap_value, 2.5, 1e-12,
        "constant curvature cap value changed");
    test_support::requireNear(
        constant_cap.changed_rate, 0.0, 1e-12,
        "constant curvature field should not be changed");

    bool cap_rejected = false;
    try {
      std::vector<double> bad_cap = {1.0, std::numeric_limits<double>::quiet_NaN()};
      static_cast<void>(adaiso::detail::capCurvatureQuantile(bad_cap, 0.995));
    } catch (const std::runtime_error&) {
      cap_rejected = true;
    }
    test_support::require(cap_rejected, "curvature cap accepted non-finite values");

    const std::vector<double> varying_kappa = {1.0, 4.0, 9.0};
    std::vector<double> scaled_kappa;
    scaled_kappa.reserve(varying_kappa.size());
    for (double value : varying_kappa) {
      scaled_kappa.push_back(10.0 * value);
    }
    const auto unscaled = adaiso::detail::targetFaceCurvatureLengths(
        mesh, varying_kappa, 1, mean_length, -1.0);
    const auto scaled = adaiso::detail::targetFaceCurvatureLengths(
        mesh, scaled_kappa, 1, mean_length, -1.0);
    for (std::size_t i = 0; i < unscaled.target_lengths.size(); ++i) {
      test_support::requireNear(
          scaled.target_lengths[i],
          unscaled.target_lengths[i],
          1e-12,
          "constant curvature rescale changed calibrated target lengths");
    }
    test_support::requireNear(
        scaled.estimated_faces,
        unscaled.estimated_faces,
        1e-12,
        "constant curvature rescale changed estimated face count");

    const adaiso::Mesh square = test_support::squareMesh();
    const std::vector<double> lengths = {0.1, 1.0, 1.0, 1.0};
    const double initial_estimate = adaiso::detail::estimateTargetFaceCount(square, lengths);
    const std::size_t target_count = static_cast<std::size_t>(std::llround(initial_estimate));
    const double baseline_scale =
        std::sqrt(initial_estimate / static_cast<double>(target_count));
    const auto smoothed = adaiso::detail::smoothLogDomainSmallSizeOutliers(
        square, lengths, 0.05, 5.0, target_count);
    test_support::require(
        smoothed[0] > baseline_scale * lengths[0],
        "log-domain smoothing did not raise the isolated small-size outlier");
    test_support::require(
        std::abs(smoothed[1] - baseline_scale * lengths[1]) < 0.08,
        "log-domain smoothing changed an ordinary neighbor too much");
    const double smoothed_estimate =
        adaiso::detail::estimateTargetFaceCount(square, smoothed);
    test_support::require(
        std::abs(smoothed_estimate - static_cast<double>(target_count)) /
                static_cast<double>(target_count) < 0.05,
        "log-domain smoothing did not preserve the requested face estimate");

    const std::filesystem::path export_dir =
        std::filesystem::temp_directory_path() / "adaiso_target_field_export_test";
    std::filesystem::remove_all(export_dir);
    adaiso::AdaIsoConfig config;
    config.global_iterations = 0;
    config.verbose = false;
    config.export_target_field_dir = export_dir.string();
    adaiso::AdaIsoRemesher remesher(config);
    static_cast<void>(remesher.remesh(square));
    test_support::require(
        countLines(export_dir / "vertices.csv") == square.vertexCount() + 1,
        "vertices.csv row count mismatch");
    test_support::require(
        countLines(export_dir / "edges.csv") == adaiso::uniqueEdges(square).size() + 1,
        "edges.csv row count mismatch");
    test_support::require(
        countLines(export_dir / "faces.csv") == square.faceCount() + 1,
        "faces.csv row count mismatch");
    test_support::require(
        std::filesystem::exists(export_dir / "field_stats.json"),
        "field_stats.json was not written");
    std::filesystem::remove_all(export_dir);

    std::cout << "target_face_sizing_tests: passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "target_face_sizing_tests: " << error.what() << '\n';
    return 1;
  }
}
