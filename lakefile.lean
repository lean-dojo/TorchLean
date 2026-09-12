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
  let value := (get_config? cuda).getD "false"
  value == "true" || value == "1"

/-- CUDA toolkit root used for includes, libraries, and runtime search paths. -/
private def cudaHome : String := Id.run do
  let home := ((get_config? cuda_home).getD "").trimAscii.toString
  if home.startsWith "-" then
    panic! s!"cuda_home must be a path, not an option-like value: {home}"
  return if home.isEmpty then "/usr/local/cuda" else home

/-- Optional LibTorch root; relative paths are resolved against the package directory. -/
private def libtorchHomeConfig : Option String :=
  (get_config? libtorch_home).bind fun path =>
    let path := path.trimAscii.toString
    if path.isEmpty then none else some path

/-- Whether to build the optional LibTorch-backed backend capsules. -/
private def libtorchEnabled : Bool :=
  let value := (get_config? libtorch).getD "false"
  value == "true" || value == "1"

/-- Native link flags selected by the `cuda` Lake option. -/
private def nativeLinkArgs : Array String :=
  if cudaEnabled then
    let lt := libtorchHomeConfig.getD "libtorch"
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
  buildDir := FilePath.mk ((get_config? torchleanBuildDir).getD ".lake/build")
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
    ⟨`warningAsError, true⟩]
  dynlibs := #[`@/torchlean_tensor_cpu_shared]
  moreLinkArgs := nativeLinkArgs

/-!
## Native backend libraries

TorchLean has a small amount of native code behind Lean `extern` declarations. Each component has
the same build shape: compile the CUDA implementation when the package is built with
`-K cuda=true`; otherwise compile the matching C stub so the Lean package still builds on machines
without a CUDA toolkit.
-/

/-- LibTorch root for native include and library paths. -/
private def libtorchHome (pkg : Package) : String :=
  (pkg.dir / libtorchHomeConfig.getD "libtorch").toString

/-- g++ compile flags for LibTorch C++ sources. -/
private def libtorchCppCompileArgs (pkg : Package) (lean : LeanInstall) (lt : String) :
    Array String :=
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

/-- Validate the configured SDK and track its path for native builds. -/
private def libtorchResolveJob (pkg : Package) : SpawnM (Job FilePath) := do
  let stamp := pkg.buildDir / "libtorch.path"
  let home : FilePath := libtorchHome pkg
  let configJob ← inputFile (pkg.dir / "lakefile.lean") false
  buildFileAfterDep stamp configJob (fun _ => do
    unless (← (home / "include").isDir) && (← (home / "lib").isDir) do
      error <| s!"LibTorch home must contain include/ and lib/: {home}. " ++
        "Pass -Klibtorch_home=/path/to/libtorch when enabling the optional bridge."
    IO.FS.createDirAll pkg.buildDir
    IO.FS.writeFile stamp (home.toString ++ "\n"))
    (pure <| .ofHash (pureHash home.toString) "LibTorch configuration")

/-- LibTorch SDPA forward/backward bridge as a shared library. -/
private def buildLibtorchSDPASo (pkg : Package) := do
  let lean ← getLeanInstall
  let resolveJob ← libtorchResolveJob pkg
  let headerDeps ← nativeHeaderDeps pkg
  let lt := libtorchHome pkg
  let cppJob ← inputFile (pkg.dir / "csrc/cuda/kernels/torchlean_libtorch_sdpa.cpp") false
  let cppO := pkg.buildDir / "torchlean_libtorch_sdpa.o"
  let deps := cppJob.zipWith (fun src _ => src) (resolveJob.mix headerDeps)
  let cppOJob ← buildO cppO deps #[] (libtorchCppCompileArgs pkg lean lt) "c++"
    getLeanTrace
  let soFile := pkg.buildDir / nameToSharedLib "torchlean_libtorch_sdpa"
  cppOJob.mapM fun o => do
    addPureTrace (libtorchSDPALinkArgs lt) "link flags"
    let art ← buildArtifactUnlessUpToDate soFile (ext := sharedLibExt) (restore := true) do
      compileSharedLib soFile (#[o.toString] ++ libtorchSDPALinkArgs lt) "g++"
    return art.path

/-- Linkable error-returning symbols when the CUDA LibTorch provider is unavailable. -/
private def buildLibtorchSDPAStub (pkg : Package) := do
  let lean ← getLeanInstall
  let srcJob ← inputFile (pkg.dir / "csrc/cuda/kernels/torchlean_libtorch_sdpa_stub.c") false
  let oFile := pkg.buildDir / "torchlean_libtorch_sdpa_stub.o"
  let oJob ← buildO oFile srcJob
    #["-I", lean.includeDir.toString] #["-O2", "-fPIC"] "cc" getLeanTrace
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

/-- Compile the native bulk operations for packed host tensor storage once. -/
private def buildTensorCpuObject (pkg : Package) := do
  let lean ← getLeanInstall
  let srcJob ← inputFile
    (pkg.dir / "csrc/cpu/torchlean_tensor.c") false
  let oFile := pkg.buildDir / "torchlean_tensor_cpu.o"
  buildO oFile srcJob
    #["-I", lean.includeDir.toString] #["-O3", "-fPIC"] "cc" getLeanTrace

/-- Object shared by the static executable link and the dynamic elaborator library. -/
target torchlean_tensor_cpu_object pkg : FilePath :=
  buildTensorCpuObject pkg

/-- Packed host tensor primitives linked into compiled executables. -/
target torchlean_tensor_cpu pkg : FilePath := do
  let oJob ← torchlean_tensor_cpu_object.fetch
  let libFile := pkg.buildDir / nameToStaticLib "torchlean_tensor_cpu"
  buildStaticLib libFile #[oJob]

/-- Packed host tensor primitives loaded by Lean for `#eval` and documentation examples. -/
target torchlean_tensor_cpu_shared pkg : Dynlib := do
  let oJob ← torchlean_tensor_cpu_object.fetch
  let libName := "torchlean_tensor_cpu"
  let libFile := pkg.sharedLibDir / nameToSharedLib libName
  buildLeanSharedLib libName libFile #[oJob] #[]

/-- Compile a CUDA component or its portable stub, tracking headers and compiler settings. -/
private def buildNativeBackendLib (pkg : Package) (dir stem : String) := do
  let lean ← getLeanInstall
  let headerDeps ← nativeHeaderDeps pkg
  let source := if cudaEnabled then s!"{stem}.cu" else s!"{stem}_stub.c"
  let srcJob ← inputFile (pkg.dir / "csrc/cuda" / dir / source) false
  let srcJob := srcJob.zipWith (fun src _ => src) headerDeps
  let objectStem := if cudaEnabled then stem else s!"{stem}_stub"
  let libraryStem := if cudaEnabled then s!"{stem}_cuda" else objectStem
  let flags := if cudaEnabled then
    #["-I", s!"{cudaHome}/include", "--std=c++17", "-O2", "-Xcompiler", "-fPIC"]
  else
    #["-O2", "-fPIC"]
  let oJob ← buildO (pkg.buildDir / s!"{objectStem}.o") srcJob
    (#["-I", lean.includeDir.toString] ++ nativeIncludeArgs pkg) flags
    (if cudaEnabled then s!"{cudaHome}/bin/nvcc" else "cc") getLeanTrace
  buildStaticLib (pkg.buildDir / nameToStaticLib libraryStem) #[oJob]

/-- CUDA+cuBLAS matrix multiplication, or portable stubs. -/
target torchlean_dgemm_cuda pkg : FilePath :=
  buildNativeBackendLib pkg "blas" "torchlean_dgemm_cuda"

/-- CUDA kernels, or portable stubs. -/
target torchlean_cuda_kernels pkg : FilePath :=
  buildNativeBackendLib pkg "kernels" "torchlean_cuda_kernels"

/-- CUDA convolution and pooling, or portable stubs. -/
target torchlean_cuda_conv_pool pkg : FilePath :=
  buildNativeBackendLib pkg "conv_pool" "torchlean_cuda_conv_pool"

/-- CUDA tensor buffers, or portable stubs. -/
target torchlean_cuda_tensor pkg : FilePath :=
  buildNativeBackendLib pkg "tensor" "torchlean_cuda_tensor"

@[default_target]
lean_lib NN where
  moreLinkObjs :=
    (#[
      torchlean_tensor_cpu,
      torchlean_dgemm_cuda,
      torchlean_cuda_kernels,
      torchlean_cuda_conv_pool,
      torchlean_cuda_tensor
    ] : TargetArray FilePath) ++
      if cudaEnabled && libtorchEnabled then
        (#[torchlean_libtorch_sdpa_so] : TargetArray FilePath)
      else
        (#[torchlean_libtorch_sdpa_stub] : TargetArray FilePath)
  -- The reusable library follows its canonical umbrella. Examples, tests, CI-only modules,
  -- documentation, and executable roots have separate targets below.
  roots := #[`NN]

/-- Runnable and narrative examples, kept out of the reusable `NN` library target. -/
lean_lib NNExamples where
  roots := #[`NN.Examples]
  globs := #[.one `NN.Examples, .submodules `NN.Examples]

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

-- Unified verification CLI registry: `lake exe verify -- <tool> [args...]`
lean_exe verify where
  root := `NN.Verification.Main

-- Native runner for `lake test`, including tests that call backend externs.
lean_exe nn_tests_suite where
  root := `NN.Tests.Suite

-- Cross-runtime numerical regression tools.
lean_exe pytorch_export_check where
  root := `NN.Tests.Interop.PyTorchMain

lean_exe native_float32_parity where
  root := `NN.Tests.Floats.NativePrimitiveParityMain

-- Optional LibTorch SDPA bridge test. Requires:
--   lake exe -K cuda=true -K libtorch=true libtorch_sdpa_test
lean_exe libtorch_sdpa_test where
  root := `NN.Tests.Runtime.Cuda.LibTorchSDPA

-- Repo-policy lints (header hygiene, banned constructs, etc.) via `lake lint`.
lean_exe torchlean_lint where
  srcDir := "scripts/checks"
  root := `TorchLeanLint

-- Runnable examples: `lake exe torchlean <example> [args...]`.
-- Build with `lake -R -K cuda=true build` before passing `--cuda` to an example.
lean_exe torchlean where
  root := `NN.Examples.RunnerMain

-- Complete API documentation (HTML) via `lake build TorchLeanDocs:docs`.
require «doc-gen4» from git
  "https://github.com/leanprover/doc-gen4" @ "v4.33.0"

-- Keep `mathlib` last so Mathlib’s dependency versions win, which is required for cache tooling.
require mathlib from git
  "https://github.com/leanprover-community/mathlib4" @ "v4.33.0"
