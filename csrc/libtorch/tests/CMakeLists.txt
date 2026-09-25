cmake_minimum_required(VERSION 3.22)
project(TorchLeanConvPoolRegression LANGUAGES CXX)

# Configure against the same LibTorch SDK and Lean toolchain as the production backend build.
foreach(required TORCHLEAN_LIBTORCH_HOME TORCHLEAN_LEAN_PREFIX)
  if(NOT DEFINED ${required})
    message(FATAL_ERROR "${required} is required.")
  endif()
endforeach()
if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
  message(FATAL_ERROR "TorchLean's LibTorch CUDA backend currently requires Linux.")
endif()
if(NOT EXISTS "${TORCHLEAN_LEAN_PREFIX}/include/lean/lean.h")
  message(FATAL_ERROR "TORCHLEAN_LEAN_PREFIX must contain include/lean/lean.h.")
endif()
if(NOT EXISTS "${TORCHLEAN_LIBTORCH_HOME}/share/cmake/Torch/TorchConfig.cmake")
  message(FATAL_ERROR "TORCHLEAN_LIBTORCH_HOME must contain share/cmake/Torch/TorchConfig.cmake.")
endif()

set(Torch_DIR "${TORCHLEAN_LIBTORCH_HOME}/share/cmake/Torch")
set(Caffe2_DIR "${TORCHLEAN_LIBTORCH_HOME}/share/cmake/Caffe2")
list(PREPEND CMAKE_PREFIX_PATH "${TORCHLEAN_LIBTORCH_HOME}")
find_package(Torch REQUIRED CONFIG PATHS "${Torch_DIR}" NO_DEFAULT_PATH)
if(NOT TARGET torch_cuda)
  message(FATAL_ERROR "Use the CUDA-enabled SDK selected for the production backend.")
endif()
find_library(LEAN_RUNTIME NAMES leanshared
  PATHS "${TORCHLEAN_LEAN_PREFIX}/lib/lean" NO_DEFAULT_PATH REQUIRED)

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

# conv_pool.cpp includes ../conv_pool.cpp with FFI wrappers disabled. Do not add the
# implementation as another translation unit or link the production backend here.
# The Lean CUDA suite separately exercises the production exports and tape.
add_executable(torchlean_conv_pool_regression conv_pool.cpp)
target_compile_definitions(torchlean_conv_pool_regression PRIVATE TORCHLEAN_LIBTORCH)
target_include_directories(torchlean_conv_pool_regression PRIVATE
  "${TORCHLEAN_LEAN_PREFIX}/include")
separate_arguments(torch_sdk_cxx_flags NATIVE_COMMAND "${TORCH_CXX_FLAGS}")
target_compile_options(torchlean_conv_pool_regression PRIVATE ${torch_sdk_cxx_flags})
target_link_libraries(torchlean_conv_pool_regression PRIVATE ${TORCH_LIBRARIES} "${LEAN_RUNTIME}")
target_link_options(torchlean_conv_pool_regression PRIVATE "LINKER:--disable-new-dtags")
set_target_properties(torchlean_conv_pool_regression PROPERTIES
  BUILD_RPATH "${TORCHLEAN_LIBTORCH_HOME}/lib;${TORCHLEAN_LEAN_PREFIX}/lib/lean")

enable_testing()
add_test(NAME conv_pool_cpu COMMAND torchlean_conv_pool_regression cpu)
add_test(NAME conv_pool_cuda COMMAND torchlean_conv_pool_regression cuda)
set_tests_properties(conv_pool_cpu PROPERTIES LABELS "conv_pool;cpu")
set_tests_properties(conv_pool_cuda PROPERTIES LABELS "conv_pool;cuda")
