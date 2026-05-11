/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

import Lake
open Lake DSL
open System

/-- Whether native CUDA archives should be built instead of portable CPU stubs. -/
private def cudaEnabled : Bool :=
  match get_config? cuda with
  | some v => v == "true" || v == "1"
  | none => false

/-- Normalize and lightly validate a CUDA toolkit root passed through `-K cuda_home=...`. -/
private def cleanCudaHome (p : String) : String :=
  let h := p.trimAscii.toString
  if h.isEmpty then
    "/usr/local/cuda"
  else if h.startsWith "-" then
    panic! s!"cuda_home must be a path, not an option-like value: {h}"
  else
    h

/-- CUDA toolkit root used for includes, libraries, and runtime search path. -/
private def cudaHome : String :=
  match get_config? cuda_home with
  | some p => cleanCudaHome p
  | none => "/usr/local/cuda"

/-- Extra native link flags for TorchLean's optional CUDA backend. -/
private def nativeLinkArgs : Array String :=
  if cudaEnabled then
    #[
      "-L", s!"{cudaHome}/lib64",
      "-lcudart", "-lcublas", "-lcufft",
      "-Wl,-rpath," ++ s!"{cudaHome}/lib64"
    ]
  else if Platform.isWindows then
    #[]
  else
    -- CPU stubs call functions from `math.h`; Linux keeps these in `libm`.
    #["-lm"]

package TorchLean where
  version := v!"0.1.0"
  description := "Neural network specification, execution, and verification in Lean 4."
  keywords := #["machine-learning", "neural-networks", "verification", "autograd", "cuda"]
  homepage := "https://lean-dojo.github.io/TorchLean/"
  license := "MIT"
  readmeFile := "README.md"
  testDriver := "nn_tests_suite"
  lintDriver := "torchlean_lint"
  leanOptions := #[
    ⟨`pp.unicode.fun, true⟩,
    ⟨`autoImplicit, false⟩,
    ⟨`relaxedAutoImplicit, false⟩,
    ⟨`backward.privateInPublic, false⟩,
    ⟨`backward.privateInPublic.warn, false⟩]
  moreLinkArgs := nativeLinkArgs

@[default_target]
lean_lib NN where
  -- `NN:docs` should document the whole maintained Lean surface, including examples and CLI
  -- dispatchers. Keep tests out of this library surface; they build through `nn_tests_suite`.
  roots := #[
    `NN,
    `NN.Examples.Zoo,
    `NN.CI.SlowProofs,
    `NN.Examples.Models.Runner,
    `NN.Verification.CLI
  ]
  globs := #[
    .one `NN,
    .one `NN.Library,
    .submodules `NN.Examples,
    .submodules `NN.Verification
  ]

private structure CudaArchive where
  stem : String
  cudaSrc : String
  stubSrc : String

/-- Shared include paths for TorchLean native CUDA/stub sources. -/
private def cudaIncludeArgs (pkg : Package) : Array String :=
  #[
    "-I", (pkg.dir / "csrc/cuda/common").toString,
    "-I", (pkg.dir / "csrc/cuda/conv_pool").toString
  ]

/-- Build a CUDA-backed archive when `-K cuda=true`, otherwise build its CPU stub. -/
private def buildCudaArchive (pkg : Package) (spec : CudaArchive) := do
  let lean ← getLeanInstall
  let includeArgs := cudaIncludeArgs pkg
  let libFile := pkg.buildDir / nameToStaticLib spec.stem
  if cudaEnabled then
    let srcJob ← inputFile (pkg.dir / spec.cudaSrc) false
    let oFile := pkg.buildDir / s!"{spec.stem}.o"
    let oJob ← buildO oFile srcJob
      (#[
        "-I", lean.includeDir.toString,
        "-I", s!"{cudaHome}/include",
        "-c", "--std=c++17", "-O2", "-Xcompiler", "-fPIC"
      ] ++ includeArgs) #[] "nvcc"
    buildStaticLib libFile #[oJob]
  else
    let srcJob ← inputFile (pkg.dir / spec.stubSrc) false
    let oFile := pkg.buildDir / s!"{spec.stem}_stub.o"
    let oJob ← buildO oFile srcJob
      (#["-I", lean.includeDir.toString] ++ includeArgs ++ #["-O2", "-fPIC"])
      #[] "cc"
    buildStaticLib libFile #[oJob]

/-- Static archive for `torchlean_dgemm_cuda`: CUDA+cuBLAS when `-K cuda=true`, else CPU stub. -/
extern_lib torchlean_dgemm_cuda (pkg) :=
  buildCudaArchive pkg {
    stem := "torchlean_dgemm_cuda"
    cudaSrc := "csrc/cuda/blas/torchlean_dgemm_cuda.cu"
    stubSrc := "csrc/cuda/blas/torchlean_dgemm_cuda_stub.c"
  }

/-- Static archive for `torchlean_cuda_kernels`: CUDA kernels when `-K cuda=true`, else CPU stub. -/
extern_lib torchlean_cuda_kernels (pkg) :=
  buildCudaArchive pkg {
    stem := "torchlean_cuda_kernels"
    cudaSrc := "csrc/cuda/kernels/torchlean_cuda_kernels.cu"
    stubSrc := "csrc/cuda/kernels/torchlean_cuda_kernels_stub.c"
  }

/-- Static archive for `torchlean_cuda_conv_pool`: CUDA conv/pool when `-K cuda=true`, else CPU stub. -/
extern_lib torchlean_cuda_conv_pool (pkg) :=
  buildCudaArchive pkg {
    stem := "torchlean_cuda_conv_pool"
    cudaSrc := "csrc/cuda/conv_pool/torchlean_cuda_conv_pool.cu"
    stubSrc := "csrc/cuda/conv_pool/torchlean_cuda_conv_pool_stub.c"
  }

/-- Static archive for `torchlean_cuda_tensor`: CUDA buffer runtime when `-K cuda=true`, else CPU stub. -/
extern_lib torchlean_cuda_tensor (pkg) :=
  buildCudaArchive pkg {
    stem := "torchlean_cuda_tensor"
    cudaSrc := "csrc/cuda/tensor/torchlean_cuda_tensor.cu"
    stubSrc := "csrc/cuda/tensor/torchlean_cuda_tensor_stub.c"
  }

-- Unified verification CLI registry: `lake exe verify -- <tool> [args...]`
lean_exe verify where
  root := `NN.Verification.CLI

-- Curated test suite runner (native executable).
-- We run this via `lake exe nn_tests_suite` instead of `lean --run ...` because the Lean
-- interpreter cannot execute definitions from precompiled `.olean`s unless the whole dependency
-- closure is built with interpreter support.
lean_exe nn_tests_suite where
  root := `NN.Tests.Suite

-- Repo-policy lints (header hygiene, banned constructs, etc.) via `lake lint`.
lean_exe torchlean_lint where
  srcDir := "scripts/checks"
  root := `TorchLeanLint

-- Device-agnostic runnable examples (CPU by default; pass `--cuda` after building with CUDA).
--
-- This single executable supports all runnable examples (MLP/CNN/Transformer/Vit/ResNet/GPT2/PPO)
-- via a simple
-- subcommand interface:
--   `lake exe torchlean <example> [args...]`
--
-- CUDA build: `lake build -R -K cuda=true`
lean_exe torchlean where
  root := `NN.Examples.Models.Runner

-- API documentation (HTML) via `lake build NN:docs`.
require «doc-gen4» from git
  "https://github.com/leanprover/doc-gen4" @ "v4.29.0"

-- Comparator: a sandboxed judge for untrusted Lean proof submissions.
-- We pin versions compatible with TorchLean's Lean toolchain (v4.29.0).
require lean4export from git
  "https://github.com/leanprover/lean4export" @ "ca36c44858e2d7ba40996203d2f08a69113d1211"

require Comparator from git
  "https://github.com/leanprover/comparator" @ "10033e381ff7f2146859e21ab99ce01f9ed61c36"

-- Keep `mathlib` last so Mathlib’s dependency versions win, which is required for cache tooling.
require mathlib from git
  "https://github.com/leanprover-community/mathlib4" @ "v4.29.0"
