#pragma once

namespace adaiso {

enum class CurvatureOutlierMode {
  None,
  TopKMedian,
  SamplingCap,
};

enum class TargetFaceRepresentative {
  Mean,
  Minimum,
};

}  // namespace adaiso
