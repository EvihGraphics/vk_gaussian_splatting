# gps_cuda: builds the Gaussian Point Splatting CUDA renderer from the git submodule
# (cherry-picking only the rendering core, none of its OpenGL/GLFW/ImGui front-end)
# plus a thin C-ABI adapter, as a shared library.
#
# Can be used standalone (for testing) or through add_subdirectory() from the
# vk_gaussian_splatting root CMakeLists.txt (option USE_GPS_CUDA).

cmake_minimum_required(VERSION 3.24)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(gps_cuda LANGUAGES C CXX CUDA)
  set(GPS_CUDA_STANDALONE ON)
else()
  set(GPS_CUDA_STANDALONE OFF)
endif()

set(GPS_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../3rdparty/gaussian-point-splatting"
    CACHE PATH "Path to the gaussian-point-splatting sources (git submodule)")
set(GPS_CUDA_ARCHITECTURES "75;86;89" CACHE STRING "CUDA architectures for gps_cuda")
option(GPS_CUDA_BUILD_SMOKETEST "Build the standalone gps_cuda_smoketest executable" ${GPS_CUDA_STANDALONE})

if(NOT EXISTS "${GPS_SOURCE_DIR}/src/core/rendering/passes/gaussian_point_splatting.cu")
  message(FATAL_ERROR "gps_cuda: GPS sources not found in '${GPS_SOURCE_DIR}'. "
                      "Run: git submodule update --init 3rdparty/gaussian-point-splatting")
endif()

find_package(CUDAToolkit REQUIRED)

set(GPS_SRC "${GPS_SOURCE_DIR}/src")

# Rendering core only. Deliberately excluded: main.cu, core/window.*, core/menu.*,
# core/input_manager.*, core/rendering/{render_server,quad_renderer,shader,grid}.*,
# core/rendering/passes/frustum_renderer.*, core/gaussians/editor/*, gaussian_test.*
set(GPS_CORE_SOURCES
  ${GPS_SRC}/core/rendering/passes/gaussian_point_splatting.cu
  ${GPS_SRC}/core/gaussians/gaussian_loader.cu
  ${GPS_SRC}/core/random/workload/workload_distributor.cu
  ${GPS_SRC}/core/rendering/occlusion/depth_mip_chain.cu
  ${GPS_SRC}/core/rendering/occlusion/gaussian_bvh.cu
  ${GPS_SRC}/timings/stochastic_timings.cu
  ${GPS_SRC}/utils/camera_loader.cu
  ${GPS_SRC}/world/camera.cpp
  ${GPS_SRC}/world/node.cpp
  ${GPS_SRC}/world/transform.cpp
)

add_library(gps_cuda SHARED
  ${GPS_CORE_SOURCES}
  ${CMAKE_CURRENT_SOURCE_DIR}/gps_cuda_api.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/gps_cuda_api.h
  ${CMAKE_CURRENT_SOURCE_DIR}/stubs/core/input_manager.h
)
source_group("GPS core" FILES ${GPS_CORE_SOURCES})

# Do not inherit the parent project's directory-level include paths (vk_gaussian_splatting
# uses include_directories()), GPS must only see its own headers and its own glm.
set_property(TARGET gps_cuda PROPERTY INCLUDE_DIRECTORIES "")
target_include_directories(gps_cuda
  PRIVATE
    # stubs first: shadows core/input_manager.h (GLFW dependency of world/camera.cpp)
    ${CMAKE_CURRENT_SOURCE_DIR}/stubs
    ${GPS_SRC}
    ${GPS_SOURCE_DIR}/packages
    ${GPS_SOURCE_DIR}/packages/glm
    ${GPS_SOURCE_DIR}/packages/stb
    ${GPS_SOURCE_DIR}/packages/tinybvh
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
)

set_target_properties(gps_cuda PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON
  CXX_EXTENSIONS OFF
  CUDA_STANDARD 17
  CUDA_STANDARD_REQUIRED ON
  CUDA_SEPARABLE_COMPILATION ON
  CUDA_RESOLVE_DEVICE_SYMBOLS ON
  CUDA_ARCHITECTURES "${GPS_CUDA_ARCHITECTURES}"
  CUDA_RUNTIME_LIBRARY Static
  FOLDER "ThirdParty/gps_cuda"
)

target_compile_definitions(gps_cuda PRIVATE
  GPS_CUDA_BUILD
  # GPS resolves relative asset paths against this; we always pass absolute paths.
  PROJECT_ROOT="${GPS_SOURCE_DIR}"
  _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
)
if(WIN32)
  target_compile_definitions(gps_cuda PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
endif()

# Same CUDA flags as upstream gaussian-point-splatting/CMakeLists.txt
target_compile_options(gps_cuda PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:
    -O3
    -use_fast_math
    --expt-extended-lambda
    --expt-relaxed-constexpr
    -lineinfo
    -allow-unsupported-compiler
    -Xcudafe=--diag_suppress=20012
  >
)
if(MSVC)
  # Note: no /utf-8, some GPS headers are not UTF-8 encoded (C4828)
  target_compile_options(gps_cuda PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:/EHsc /W3>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/EHsc>
  )
endif()

target_link_libraries(gps_cuda PRIVATE CUDA::cudart_static)
if(MSVC)
  # cudart_static drags a LIBCMT default-lib directive into an /MD build
  target_link_options(gps_cuda PRIVATE /NODEFAULTLIB:LIBCMT)
endif()

if(GPS_CUDA_BUILD_SMOKETEST)
  add_executable(gps_cuda_smoketest ${CMAKE_CURRENT_SOURCE_DIR}/gps_cuda_smoketest.cpp)
  set_property(TARGET gps_cuda_smoketest PROPERTY INCLUDE_DIRECTORIES "")
  target_include_directories(gps_cuda_smoketest PRIVATE
    ${GPS_SOURCE_DIR}/packages        # nlohmann/json.hpp
    ${GPS_SOURCE_DIR}/packages/stb)
  set_target_properties(gps_cuda_smoketest PROPERTIES CXX_STANDARD 17 FOLDER "ThirdParty/gps_cuda")
  target_link_libraries(gps_cuda_smoketest PRIVATE gps_cuda)
endif()
