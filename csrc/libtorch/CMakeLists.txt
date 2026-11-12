cmake_minimum_required(VERSION 3.22)
project(TorchLeanLibTorch LANGUAGES CXX)

if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
  message(FATAL_ERROR "TorchLean's LibTorch CUDA backend currently requires Linux.")
endif()
if(NOT EXISTS "${TORCHLEAN_LEAN_INCLUDE}/lean/lean.h")
  message(FATAL_ERROR "TORCHLEAN_LEAN_INCLUDE must name the pinned Lean include directory.")
endif()
if(NOT EXISTS "${TORCHLEAN_LIBTORCH_HOME}/share/cmake/Torch/TorchConfig.cmake")
  message(FATAL_ERROR "TORCHLEAN_LIBTORCH_HOME must contain share/cmake/Torch/TorchConfig.cmake.")
endif()

# Use the selected SDK exclusively. Torch's exported targets carry its ABI flags and dependencies.
set(Torch_DIR "${TORCHLEAN_LIBTORCH_HOME}/share/cmake/Torch")
set(Caffe2_DIR "${TORCHLEAN_LIBTORCH_HOME}/share/cmake/Caffe2")
list(PREPEND CMAKE_PREFIX_PATH "${TORCHLEAN_LIBTORCH_HOME}")
find_package(Torch REQUIRED CONFIG PATHS "${Torch_DIR}" NO_DEFAULT_PATH)
find_package(CUDAToolkit REQUIRED)
if(NOT TARGET torch_cuda)
  message(FATAL_ERROR "cuda=true requires a CUDA-enabled LibTorch SDK (torch_cuda is missing).")
endif()

get_target_property(torch_cxx_standard torch CXX_STANDARD)
# Require C++20 or a stricter standard advertised by the selected SDK.
if(NOT torch_cxx_standard OR torch_cxx_standard LESS 20)
  set(torch_cxx_standard 20)
endif()
get_target_property(torch_compile_features torch INTERFACE_COMPILE_FEATURES)
get_target_property(torch_compile_options torch INTERFACE_COMPILE_OPTIONS)
get_target_property(torch_compile_definitions torch INTERFACE_COMPILE_DEFINITIONS)
foreach(feature IN LISTS torch_compile_features)
  if(feature MATCHES "^cxx_std_([0-9]+)$" AND CMAKE_MATCH_1 GREATER torch_cxx_standard)
    set(torch_cxx_standard "${CMAKE_MATCH_1}")
  endif()
endforeach()
set(CMAKE_CXX_STANDARD "${torch_cxx_standard}")
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(torch_sdk_version "${Torch_VERSION}")
foreach(include_dir IN LISTS TORCH_INCLUDE_DIRS)
  if(EXISTS "${include_dir}/torch/version.h")
    file(STRINGS "${include_dir}/torch/version.h" version_line
      REGEX "^#define TORCH_VERSION \"")
    if(version_line)
      string(REGEX REPLACE "^#define TORCH_VERSION \"([^\"]+)\".*" "\\1"
        torch_sdk_version "${version_line}")
    endif()
  endif()
endforeach()
# torchlean.cpp, conv_pool.cpp, tree.cpp and kernels.cpp call torch only through the LibTorch Stable
# ABI of torch 2.13 (TORCH_TARGET_VERSION is TORCH_VERSION_2_13_0), so their calls also run with
# later releases. unstable.cpp holds what that ABI cannot express, and docs/TRUST_BOUNDARIES.md
# lists both.
if(torch_sdk_version VERSION_LESS 2.13)
  message(FATAL_ERROR "The LibTorch bridge needs torch 2.13 or later, not ${torch_sdk_version}.")
endif()

# Keep the compile/link probe in this project. A try_compile scratch project cannot reliably
# re-export SDK alias targets (the NVIDIA SDK's torch_cpu_library uses protobuf::libprotobuf).
# This executable is built as a dependency of the backend and is never run.
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/sdk_link_check.cpp" [=[
  #include <ATen/ATen.h>
  #include <c10/cuda/CUDAFunctions.h>
  #include <torch/version.h>
  #include <string>
  int main() {
    const c10::Device device(std::string("cpu"));
    auto tensor = at::zeros({1}, at::TensorOptions().device(device));
    return tensor.numel() == 1 && c10::cuda::device_count() >= 0 ? 0 : 1;
  }
]=])
add_executable(torchlean_sdk_link_check EXCLUDE_FROM_ALL
  "${CMAKE_CURRENT_BINARY_DIR}/sdk_link_check.cpp")
target_link_libraries(torchlean_sdk_link_check PRIVATE ${TORCH_LIBRARIES})
separate_arguments(torch_sdk_cxx_flags NATIVE_COMMAND "${TORCH_CXX_FLAGS}")
target_compile_options(torchlean_sdk_link_check PRIVATE ${torch_sdk_cxx_flags})
target_link_options(torchlean_sdk_link_check PRIVATE "LINKER:--disable-new-dtags")

add_library(torchlean_libtorch SHARED
  torchlean.cpp
  conv_pool.cpp
  tree.cpp
  kernels.cpp
  unstable.cpp
)
# USE_CUDA declares the CUDA shims (device guards and streams).
set_source_files_properties(torchlean.cpp conv_pool.cpp tree.cpp kernels.cpp PROPERTIES
  COMPILE_DEFINITIONS "TORCH_TARGET_VERSION=0x020d000000000000;USE_CUDA")
add_dependencies(torchlean_libtorch torchlean_sdk_link_check)
target_compile_definitions(torchlean_libtorch PRIVATE TORCHLEAN_LIBTORCH)
target_compile_options(torchlean_libtorch PRIVATE ${torch_sdk_cxx_flags})
target_include_directories(torchlean_libtorch PRIVATE
  "${TORCHLEAN_LEAN_INCLUDE}"
  "${CMAKE_CURRENT_SOURCE_DIR}"
)
target_link_libraries(torchlean_libtorch PRIVATE ${TORCH_LIBRARIES}
  CUDA::nvrtc CUDA::cuda_driver)
set_target_properties(torchlean_libtorch PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  LIBRARY_OUTPUT_DIRECTORY "${TORCHLEAN_OUTPUT_DIR}"
  # Lake links this private library by absolute path. Omitting SONAME retains that path in
  # the executable, including for downstream packages whose build directory differs.
  NO_SONAME ON
  BUILD_WITH_INSTALL_RPATH ON
  INSTALL_RPATH_USE_LINK_PATH ON
)
# RUNPATH, not RPATH, so LD_LIBRARY_PATH can select another torch runtime for the built library.
# A RUNPATH serves only the bridge's own dependencies, so each SDK library must find its own,
# libc10 among them, through its own search path, as pip torch's do with `$ORIGIN`, or through
# LD_LIBRARY_PATH or ldconfig (scripts/README.md). The bridge calls no CUDA runtime function, and
# --as-needed keeps libcudart, which torch's targets add, out of its dependencies: the torch runtime
# loads its own. -z now binds every import at load, so a torch runtime that lacks a function of
# docs/TRUST_BOUNDARIES.md fails to load, naming the function.
target_link_options(torchlean_libtorch PRIVATE "LINKER:--enable-new-dtags" "LINKER:--as-needed"
  "LINKER:-z,now")

file(WRITE "${CMAKE_BINARY_DIR}/sdk.txt"
  "Torch_VERSION=${torch_sdk_version}
Torch_DIR=${Torch_DIR}
TORCH_CXX_FLAGS=${TORCH_CXX_FLAGS}
TORCH_COMPILE_FEATURES=${torch_compile_features}
TORCH_COMPILE_OPTIONS=${torch_compile_options}
TORCH_COMPILE_DEFINITIONS=${torch_compile_definitions}
TORCH_LIBRARIES=${TORCH_LIBRARIES}
CXX_STANDARD=${CMAKE_CXX_STANDARD}
CXX_COMPILER=${CMAKE_CXX_COMPILER}
CXX_COMPILER_ID=${CMAKE_CXX_COMPILER_ID}
CXX_COMPILER_VERSION=${CMAKE_CXX_COMPILER_VERSION}
")
message(STATUS "TorchLean: LibTorch ${torch_sdk_version}, C++${CMAKE_CXX_STANDARD}")
message(STATUS "TorchLean SDK flags: ${TORCH_CXX_FLAGS}; ${torch_compile_options}")
