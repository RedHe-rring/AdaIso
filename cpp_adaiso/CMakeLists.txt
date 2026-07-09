cmake_minimum_required(VERSION 3.20)

project(cpp_adaiso VERSION 2.8.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(FetchContent)

find_package(Eigen3 CONFIG QUIET)
if(NOT TARGET Eigen3::Eigen)
  FetchContent_Declare(
    eigen
    URL https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz
  )
  set(EIGEN_BUILD_DOC OFF CACHE BOOL "" FORCE)
  set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(eigen)
endif()

find_package(nanoflann CONFIG QUIET)
if(NOT TARGET nanoflann::nanoflann)
  FetchContent_Declare(
    nanoflann
    URL https://github.com/jlblancoc/nanoflann/archive/refs/tags/v1.5.5.tar.gz
  )
  FetchContent_MakeAvailable(nanoflann)
endif()

find_package(Libigl CONFIG REQUIRED)

add_library(adaiso_core
  src/AdaIsoRemesher.cpp
  src/CsfSolver.cpp
  src/Geometry.cpp
  src/IO.cpp
  src/KDTree.cpp
  src/Mesh.cpp
  src/Metrics.cpp
)

target_include_directories(adaiso_core
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

if(TARGET Eigen3::Eigen)
  target_link_libraries(adaiso_core PUBLIC Eigen3::Eigen)
else()
  target_include_directories(adaiso_core PUBLIC ${eigen_SOURCE_DIR})
endif()

if(TARGET nanoflann::nanoflann)
  target_link_libraries(adaiso_core PUBLIC nanoflann::nanoflann)
else()
  target_include_directories(adaiso_core PUBLIC ${nanoflann_SOURCE_DIR}/include)
endif()

target_link_libraries(adaiso_core PUBLIC igl::igl_core)

if(MSVC)
  target_compile_options(adaiso_core PRIVATE /W4 /permissive-)
else()
  target_compile_options(adaiso_core PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_executable(adaiso_cli src/main.cpp)
target_link_libraries(adaiso_cli PRIVATE adaiso_core)

add_executable(adaiso_eval src/eval.cpp)
target_link_libraries(adaiso_eval PRIVATE adaiso_core)

option(ADAISO_BUILD_TESTS "Build basic smoke tests" ON)
if(ADAISO_BUILD_TESTS)
  enable_testing()
  add_executable(adaiso_tests tests/basic_tests.cpp)
  target_link_libraries(adaiso_tests PRIVATE adaiso_core)
  add_test(NAME adaiso_basic_tests COMMAND adaiso_tests)
endif()
