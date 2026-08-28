/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

import Lake
import Lake.Util.Proc
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

/-- CUDA compilation target for the native kernels, from `-K cuda_arch=...`.

Device code is compiled *for* an architecture. With no target, `nvcc` applies its own built-in
default, which changes with the toolkit version (CUDA 13.0 emits `sm_75` SASS plus `compute_75`
PTX). A binary then runs natively on that architecture only, reaching newer GPUs through forward
PTX JIT and older ones not at all, so the default is a portability choice rather than a
performance one.

Accepted values:
* a bare architecture, such as `sm_86`, `compute_86`, `native`, `all`, or `all-major`, passed as
  `-arch=<value>` — `nvcc` shorthand for that architecture's SASS *and* its PTX;
* a value starting with `-`, such as `-gencode arch=compute_86,code=[sm_86,compute_86]`, split on
  spaces and passed verbatim, which is how a multi-architecture binary is requested.

The `TORCHLEAN_CUDA_ARCH` environment variable supplies the same value when the Lake option is
absent, so a container or CI job can select the target without rewriting its `lake` invocation.
Prefer either spelling over `nvcc`'s own `NVCC_APPEND_FLAGS`: both reach `buildO`'s traced
arguments below, whereas a variable Lake never reads leaves the existing objects looking current,
and the build then links kernels compiled for the previous architecture. -/
private def cudaArchConfig : Option String :=
  match get_config? cuda_arch with
  | some v =>
      let t := v.trimAscii.toString
      if t.isEmpty then none else some t
  | none => none

/-- Resolve the CUDA compilation target: the `cuda_arch` Lake option, else `TORCHLEAN_CUDA_ARCH`,
else none (leaving `nvcc` on its built-in default). -/
private def resolveCudaArch : SpawnM (Option String) := do
  match cudaArchConfig with
  | some spec => return some spec
  | none =>
      let env ← IO.getEnv "TORCHLEAN_CUDA_ARCH"
      return env.bind fun v =>
        let t := v.trimAscii.toString
        if t.isEmpty then none else some t

/-- `nvcc` flags for a resolved CUDA compilation target; empty when no target was requested. -/
private def cudaArchArgs : Option String → Array String
  | none => #[]
  | some spec =>
      if spec.startsWith "-" then
        ((spec.splitOn " ").filter (!·.isEmpty)).toArray
      else
        #[s!"-arch={spec}"]

/-- Optional explicit LibTorch root from `-K libtorch_home=...`. -/
private def libtorchHomeConfig : Option String :=
  match get_config? libtorch_home with
  | some p =>
      let t := p.trimAscii.toString
      if t.isEmpty then none else some t
  | none => none

/-- Whether to build the optional LibTorch-backed backend capsules. -/
private def libtorchEnabled : Bool :=
  match get_config? libtorch with
  | some v => v == "true" || v == "1"
  | none => false

/-- Native link flags selected by the `cuda` Lake option. -/
private def nativeLinkArgs : Array String :=
  if cudaEnabled then
    let lt := match libtorchHomeConfig with | some h => h | none => "libtorch"
    let cudaArgs := #[
      "-L", s!"{cudaHome}/lib64", "-lcudart", "-lcublas", "-lcufft",
      "-Wl,-rpath," ++ s!"{cudaHome}/lib64"
    ]
    if libtorchEnabled then
      cudaArgs.push ("-Wl,-rpath," ++ s!"{lt}/lib")
    else
      cudaArgs
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

/-- LibTorch root for `-I` / `-L` (must match `resolve_libtorch.sh`). -/
private def libtorchHome (pkg : Package) : String :=
  match libtorchHomeConfig with
  | some h => h
  | none => (pkg.dir / "libtorch").toString

/-- g++ compile flags for LibTorch C++ sources. -/
private def libtorchCppCompileArgs (pkg : Package) (lean : LeanInstall) (lt : String) : Array String :=
  #[
    "-I", lean.includeDir.toString,
    "-I", s!"{pkg.dir}/csrc/cuda/common",
    "-I", s!"{cudaHome}/include",
    "-I", s!"{lt}/include",
    "-I", s!"{lt}/include/torch/csrc/api/include",
    "-c", "-O2", "-fPIC", "-std=c++17", "-D_GLIBCXX_USE_CXX11_ABI=1"
  ]

/-- g++ link flags for the LibTorch SDPA shared library. -/
private def libtorchSDPALinkArgs (lt : String) : Array String :=
  #[
    "-L", s!"{lt}/lib",
    "-Wl,--no-as-needed",
    "-ltorch", "-ltorch_cpu", "-ltorch_cuda", "-lc10", "-lc10_cuda",
    "-L", s!"{cudaHome}/lib64", "-lcudart",
    "-lstdc++",
    "-Wl,-rpath," ++ s!"{lt}/lib",
    "-Wl,-rpath," ++ s!"{cudaHome}/lib64"
  ]

/-- Include paths shared by the CUDA implementations and the portable C stubs. -/
private def nativeIncludeArgs (pkg : Package) : Array String :=
  #[
    "-I", (pkg.dir / "csrc/cuda/common").toString,
    "-I", (pkg.dir / "csrc/cuda/conv_pool").toString
  ]

/-- Track project-owned native headers so a header-only edit invalidates every dependent object. -/
private def nativeHeaderDeps (pkg : Package) : SpawnM (Job Unit) := do
  let isHeader := fun path : FilePath => path.extension == some "h"
  let common ← inputDir (pkg.dir / "csrc/cuda/common") true isHeader
  let convPool ← inputDir (pkg.dir / "csrc/cuda/conv_pool") true isHeader
  pure <| common.mix convPool

/-- Resolve LibTorch; caches `.lake/build/libtorch.path`. -/
private def libtorchResolveJob (pkg : Package) : SpawnM (Job FilePath) := do
  let stamp := pkg.buildDir / "libtorch.path"
  let resolver := pkg.dir / "scripts" / "setup" / "resolve_libtorch.sh"
  let resolverJob ← inputFile resolver false
  let args :=
    match libtorchHomeConfig with
    | some home => #[resolver.toString, home]
    | none => #[resolver.toString]
  buildFileAfterDep stamp resolverJob fun _ =>
    proc { cmd := "bash", args := args, cwd := some pkg.dir }

/-- LibTorch SDPA forward/backward bridge as a shared library. -/
private def buildLibtorchSDPASo (pkg : Package) := do
  let lean ← getLeanInstall
  let _ ← libtorchResolveJob pkg
  let headerDeps ← nativeHeaderDeps pkg
  let lt := libtorchHome pkg
  let cppJob ← inputFile (pkg.dir / "csrc/cuda/kernels/torchlean_libtorch_sdpa.cpp") false
  let cppO := pkg.buildDir / "torchlean_libtorch_sdpa.o"
  let cppOJob ← buildO cppO cppJob (libtorchCppCompileArgs pkg lean lt) #[] "c++"
    (pure headerDeps.getTrace)
  let soFile := pkg.buildDir / nameToSharedLib "torchlean_libtorch_sdpa"
  cppOJob.mapM fun o => do
    let art ← buildArtifactUnlessUpToDate soFile (ext := sharedLibExt) (restore := true) do
      compileSharedLib soFile (#[o.toString] ++ libtorchSDPALinkArgs lt) "g++"
    return art.path

/-- Linkable error-returning symbols when the CUDA LibTorch provider is unavailable. -/
private def buildLibtorchSDPAStub (pkg : Package) := do
  let lean ← getLeanInstall
  let srcJob ← inputFile (pkg.dir / "csrc/cuda/kernels/torchlean_libtorch_sdpa_stub.c") false
  let oFile := pkg.buildDir / "torchlean_libtorch_sdpa_stub.o"
  let oJob ← buildO oFile srcJob
    #["-I", lean.includeDir.toString, "-O2", "-fPIC"] #[] "cc"
  let libFile := pkg.buildDir / nameToStaticLib "torchlean_libtorch_sdpa_stub"
  buildStaticLib libFile #[oJob]

target torchlean_libtorch_sdpa_so pkg : FilePath :=
  if cudaEnabled && libtorchEnabled then
    buildLibtorchSDPASo pkg
  else
    pure (Job.pure (pkg.buildDir / "torchlean_libtorch_sdpa_skipped"))

target torchlean_libtorch_sdpa_stub pkg : FilePath :=
  if !cudaEnabled || !libtorchEnabled then
    buildLibtorchSDPAStub pkg
  else
    pure (Job.pure (pkg.buildDir / "torchlean_libtorch_sdpa_stub_skipped"))

@[default_target]
lean_lib NN where
  moreLinkObjs :=
    if cudaEnabled && libtorchEnabled then #[torchlean_libtorch_sdpa_so]
    else #[torchlean_libtorch_sdpa_stub]
  -- The reusable library follows its canonical umbrella. Examples, tests, CI-only modules,
  -- documentation, and executable roots have separate targets below.
  roots := #[`NN]

/-- Runnable and narrative examples, kept out of the reusable `NN` library target. -/
lean_lib NNExamples where
  roots := #[`NN.Examples.Zoo]
  globs := #[.submodules `NN.Examples]

/-- Test modules used by the curated native test runner. -/
lean_lib NNTests where
  roots := #[`NN.Tests.Suite]
  globs := #[.submodules `NN.Tests]

/-- Ordinary CI-only imports omitted from the downstream `NN` umbrella. -/
lean_lib NNCI where
  roots := #[`NN.CI.All]

/-- Proof-heavy modules typechecked by the docs build or an explicit local target. -/
lean_lib NNSlowProofs where
  roots := #[`NN.CI.SlowProofs]

/-- Complete maintained API documentation surface. -/
lean_lib TorchLeanDocs where
  roots := #[`NN.Docs]

/-- Build one native backend library for the current Lake configuration. -/
private def buildNativeBackendLib (pkg : Package) (spec : NativeBackendLib) := do
  let lean ← getLeanInstall
  let headerDeps ← nativeHeaderDeps pkg
  let includeArgs := nativeIncludeArgs pkg
  let libFile := pkg.buildDir / nameToStaticLib spec.stem
  -- Include paths stay in `weakArgs`, where a moved checkout does not invalidate every object.
  -- The flags that change what the compiler emits belong in `traceArgs`: `buildO` hashes those,
  -- so switching optimization level or CUDA compilation target rebuilds instead of silently
  -- reusing objects built for the previous one.
  if cudaEnabled then
    let srcJob ← inputFile (pkg.dir / spec.cudaSrc) false
    let oFile := pkg.buildDir / s!"{spec.stem}.o"
    let archArgs := cudaArchArgs (← resolveCudaArch)
    let oJob ← buildO oFile srcJob
      (weakArgs := #["-I", lean.includeDir.toString, "-I", s!"{cudaHome}/include"] ++ includeArgs)
      (traceArgs := #["-c", "--std=c++17", "-O2", "-Xcompiler", "-fPIC"] ++ archArgs)
      (compiler := "nvcc") (extraDepTrace := pure headerDeps.getTrace)
    buildStaticLib libFile #[oJob]
  else
    let srcJob ← inputFile (pkg.dir / spec.stubSrc) false
    let oFile := pkg.buildDir / s!"{spec.stem}_stub.o"
    let oJob ← buildO oFile srcJob
      (weakArgs := #["-I", lean.includeDir.toString] ++ includeArgs)
      (traceArgs := #["-O2", "-fPIC"])
      (compiler := "cc") (extraDepTrace := pure headerDeps.getTrace)
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
  root := `NN.Verification.Main

-- Curated test suite runner (native executable).
-- We run this via `lake exe nn_tests_suite` instead of `lean --run ...` because the Lean
-- interpreter cannot execute definitions from precompiled `.olean`s unless the whole dependency
-- closure is built with interpreter support.
lean_exe nn_tests_suite where
  root := `NN.Tests.Suite

-- Optional LibTorch SDPA bridge test. Requires:
--   lake exe -K cuda=true -K libtorch=true libtorch_sdpa_test
lean_exe libtorch_sdpa_test where
  root := `NN.Tests.Runtime.Cuda.LibTorchSDPA

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
  root := `NN.Examples.Models.RunnerMain

-- Self-checking positive/negative example for the functional transcendental +
-- scalar-affine ops (`nn.functional.{exp,log,scale,shift,affine}`). Runs the
-- autograd checks compiled; exits non-zero on any regression.
--   `lake exe transcendentals_check`
lean_exe transcendentals_check where
  root := `NN.Examples.Functional.Transcendentals

-- Complete API documentation (HTML) via `lake build TorchLeanDocs:docs`.
require «doc-gen4» from git
  "https://github.com/leanprover/doc-gen4" @ "v4.33.0"

-- Keep `mathlib` last so Mathlib’s dependency versions win, which is required for cache tooling.
require mathlib from git
  "https://github.com/leanprover-community/mathlib4" @ "v4.33.0"
