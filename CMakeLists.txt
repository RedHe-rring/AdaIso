cmake_minimum_required(VERSION 3.20)

project(cpp_adaiso VERSION 2.12.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Eigen3 CONFIG REQUIRED)
find_package(nanoflann CONFIG REQUIRED)
find_package(Libigl CONFIG REQUIRED)

add_library(adaiso_core
  src/AdaIsoRemesher.cpp
  src/FeatureClassification.cpp
  src/Geometry.cpp
  src/IO.cpp
  src/KDTree.cpp
  src/LocalOperators.cpp
  src/Mesh.cpp
  src/Metrics.cpp
  src/RawCurvature.cpp
  src/ScreenedLaplacian.cpp
  src/TargetFaceSizing.cpp
)

target_include_directories(adaiso_core
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(adaiso_core PUBLIC Eigen3::Eigen nanoflann::nanoflann igl::igl_core)

if(MSVC)
  target_compile_options(adaiso_core PRIVATE /W4 /permissive-)
else()
  target_compile_options(adaiso_core PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_executable(adaiso_cli src/main.cpp)
target_link_libraries(adaiso_cli PRIVATE adaiso_core)

add_executable(adaiso_eval src/eval.cpp)
target_link_libraries(adaiso_eval PRIVATE adaiso_core)

option(ADAISO_BUILD_TESTS "Build focused target-field tests" ON)
if(ADAISO_BUILD_TESTS)
  enable_testing()

  add_executable(screened_laplacian_tests tests/screened_laplacian_tests.cpp)
  target_link_libraries(screened_laplacian_tests PRIVATE adaiso_core)
  add_test(NAME screened_laplacian_tests COMMAND screened_laplacian_tests)

  add_executable(target_face_sizing_tests tests/target_face_sizing_tests.cpp)
  target_link_libraries(target_face_sizing_tests PRIVATE adaiso_core)
  add_test(NAME target_face_sizing_tests COMMAND target_face_sizing_tests)

  add_executable(raw_curvature_tests tests/raw_curvature_tests.cpp)
  target_link_libraries(raw_curvature_tests PRIVATE adaiso_core)
  add_test(NAME raw_curvature_tests COMMAND raw_curvature_tests)

  add_executable(local_operators_tests tests/local_operators_tests.cpp)
  target_link_libraries(local_operators_tests PRIVATE adaiso_core)
  add_test(NAME local_operators_tests COMMAND local_operators_tests)
endif()
