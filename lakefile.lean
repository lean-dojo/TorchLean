/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

import Lake
open Lake DSL
open System

/-- Whether Lake should compile the native CUDA sources instead of the portable C stubs. -/
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

/-- Native link flags selected by the `cuda` Lake option. -/
private def nativeLinkArgs : Array String :=
  if cudaEnabled then
    #[
      "-L", s!"{cudaHome}/lib64",
      "-lcudart", "-lcublas", "-lcufft",
      "-lstdc++",
      "-Wl,-rpath," ++ s!"{cudaHome}/lib64"
    ]
  else if Platform.isWindows || Platform.isOSX then
    -- Windows and macOS provide libm via the default C runtime
    #[]
  else
    -- CPU stubs call functions from `math.h`; Linux keeps these in `libm`.
    -- Keep libstdc++ for mixed native objects when switching between CPU and CUDA builds.
    #["-lm", "-lstdc++"]

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

/-!
## Native backend libraries

TorchLean has a small amount of native code behind Lean `extern` declarations. Each component has
the same build shape: compile the CUDA implementation when the package is built with
`-K cuda=true`; otherwise compile the matching C stub so the Lean package still builds on machines
without a CUDA toolkit.
-/

private structure NativeBackendLib where
  stem : String
  cudaSrc : String
  stubSrc : String

/-- Include paths shared by the CUDA implementations and the portable C stubs. -/
private def nativeIncludeArgs (pkg : Package) : Array String :=
  #[
    "-I", (pkg.dir / "csrc/cuda/common").toString,
    "-I", (pkg.dir / "csrc/cuda/conv_pool").toString
  ]

/-- Build one native backend library for the current Lake configuration. -/
private def buildNativeBackendLib (pkg : Package) (spec : NativeBackendLib) := do
  let lean ← getLeanInstall
  let includeArgs := nativeIncludeArgs pkg
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

/-- Native backend for `torchlean_dgemm_cuda`: CUDA+cuBLAS when `-K cuda=true`, else C stub. -/
extern_lib torchlean_dgemm_cuda (pkg) :=
  buildNativeBackendLib pkg {
    stem := "torchlean_dgemm_cuda"
    cudaSrc := "csrc/cuda/blas/torchlean_dgemm_cuda.cu"
    stubSrc := "csrc/cuda/blas/torchlean_dgemm_cuda_stub.c"
  }

/-- Native backend for `torchlean_cuda_kernels`: CUDA kernels when `-K cuda=true`, else C stub. -/
extern_lib torchlean_cuda_kernels (pkg) :=
  buildNativeBackendLib pkg {
    stem := "torchlean_cuda_kernels"
    cudaSrc := "csrc/cuda/kernels/torchlean_cuda_kernels.cu"
    stubSrc := "csrc/cuda/kernels/torchlean_cuda_kernels_stub.c"
  }

/-- Native backend for `torchlean_cuda_conv_pool`: CUDA conv/pool when `-K cuda=true`, else C stub. -/
extern_lib torchlean_cuda_conv_pool (pkg) :=
  buildNativeBackendLib pkg {
    stem := "torchlean_cuda_conv_pool"
    cudaSrc := "csrc/cuda/conv_pool/torchlean_cuda_conv_pool.cu"
    stubSrc := "csrc/cuda/conv_pool/torchlean_cuda_conv_pool_stub.c"
  }

/-- Native backend for `torchlean_cuda_tensor`: CUDA buffer runtime when `-K cuda=true`, else C stub. -/
extern_lib torchlean_cuda_tensor (pkg) :=
  buildNativeBackendLib pkg {
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
  "https://github.com/leanprover/doc-gen4" @ "v4.30.0"

-- Comparator: a sandboxed judge for untrusted Lean proof submissions.
-- We pin versions compatible with TorchLean's Lean toolchain.
require lean4export from git
  "https://github.com/leanprover/lean4export" @ "ca36c44858e2d7ba40996203d2f08a69113d1211"

require Comparator from git
  "https://github.com/leanprover/comparator" @ "10033e381ff7f2146859e21ab99ce01f9ed61c36"

-- Keep `mathlib` last so Mathlib’s dependency versions win, which is required for cache tooling.
require mathlib from git
  "https://github.com/leanprover-community/mathlib4" @ "v4.30.0"
