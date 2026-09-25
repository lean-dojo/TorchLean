cmake_minimum_required(VERSION 3.22)
project(TorchLeanElementwiseRegression LANGUAGES CXX)

# Configure, compile and run this project only in the cluster.
# Link the production backend so that the tests exercise its actual C ABI and invoke guard.
foreach(required TORCHLEAN_LIBTORCH_HOME TORCHLEAN_LEAN_PREFIX TORCHLEAN_BACKEND_LIBRARY)
  if(NOT DEFINED ${required})
    message(FATAL_ERROR "${required} is required; see README.md")
  endif()
endforeach()
if(NOT EXISTS "${TORCHLEAN_BACKEND_LIBRARY}")
  message(FATAL_ERROR "Build the production backend before configuring this regression.")
endif()
if(NOT EXISTS "${TORCHLEAN_LEAN_PREFIX}/include/lean/lean.h")
  message(FATAL_ERROR "TORCHLEAN_LEAN_PREFIX must contain include/lean/lean.h.")
endif()

set(Torch_DIR "${TORCHLEAN_LIBTORCH_HOME}/share/cmake/Torch")
set(Caffe2_DIR "${TORCHLEAN_LIBTORCH_HOME}/share/cmake/Caffe2")
list(PREPEND CMAKE_PREFIX_PATH "${TORCHLEAN_LIBTORCH_HOME}")
find_package(Torch REQUIRED CONFIG PATHS "${Torch_DIR}" NO_DEFAULT_PATH)
if(NOT TARGET torch_cuda)
  message(FATAL_ERROR "These regressions require the CUDA-enabled SDK.")
endif()
find_library(LEAN_RUNTIME NAMES leanshared
  PATHS "${TORCHLEAN_LEAN_PREFIX}/lib/lean" NO_DEFAULT_PATH REQUIRED)

# This test project selects C++20; honor any stricter standard exported by the SDK.
get_target_property(CMAKE_CXX_STANDARD torch CXX_STANDARD)
if(NOT CMAKE_CXX_STANDARD OR CMAKE_CXX_STANDARD LESS 20)
  set(CMAKE_CXX_STANDARD 20)
endif()
get_target_property(torch_features torch INTERFACE_COMPILE_FEATURES)
foreach(feature IN LISTS torch_features)
  if(feature MATCHES "^cxx_std_([0-9]+)$" AND CMAKE_MATCH_1 GREATER CMAKE_CXX_STANDARD)
    set(CMAKE_CXX_STANDARD "${CMAKE_MATCH_1}")
  endif()
endforeach()
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(torchlean_elementwise_regression
  regression.cpp
  ../../../../scripts/checks/cuda_float32_parity.cpp)
target_compile_definitions(torchlean_elementwise_regression PRIVATE TORCHLEAN_LIBTORCH)
target_include_directories(torchlean_elementwise_regression PRIVATE
  "${TORCHLEAN_LEAN_PREFIX}/include")
target_link_libraries(torchlean_elementwise_regression PRIVATE
  "${TORCHLEAN_BACKEND_LIBRARY}" ${TORCH_LIBRARIES} "${LEAN_RUNTIME}")
separate_arguments(torch_sdk_cxx_flags NATIVE_COMMAND "${TORCH_CXX_FLAGS}")
if(NOT CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  message(FATAL_ERROR "The staged host oracle needs GCC/Clang floating-point flags.")
endif()
target_compile_options(torchlean_elementwise_regression PRIVATE
  ${torch_sdk_cxx_flags} -fno-fast-math -ffp-contract=off)
target_link_options(torchlean_elementwise_regression PRIVATE "LINKER:--disable-new-dtags")
get_filename_component(backend_directory "${TORCHLEAN_BACKEND_LIBRARY}" DIRECTORY)
set_target_properties(torchlean_elementwise_regression PROPERTIES
  BUILD_RPATH "${backend_directory};${TORCHLEAN_LIBTORCH_HOME}/lib;${TORCHLEAN_LEAN_PREFIX}/lib/lean")
