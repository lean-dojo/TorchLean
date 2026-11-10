/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

import Lake
import Lake.Util.Proc
open Lake DSL
open System

/-- Whether the build links the LibTorch CUDA backend, selected by `-K cuda=true` or `-K cuda=1`.
Any other value, and the default, links the unavailable-backend shim. -/
private def cudaEnabled : Bool :=
  let value := (get_config? cuda).getD "false"
  value == "true" || value == "1"

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
  -- Windows and macOS include the math functions in their default C runtime.
  moreLinkArgs := if Platform.isWindows || Platform.isOSX then #[] else #["-lm"]

/-!
## Native backend libraries

`-K cuda=true` builds one LibTorch C++ library containing the numerical C ABI exports, and CMake
obtains its ABI, language standard, libraries, and runtime paths from the selected SDK. The default
build needs neither LibTorch nor a CUDA toolkit: it links a small C file that reports the backend
as not linked and rejects GPU computation. Status and memory queries work in both builds, and
without LibTorch the memory statistics read zero.
-/

/--
Finds the compiler `name` (on `PATH` when it is a bare name) and traces its invocation path,
resolved path and file hash, so replacing the compiler invalidates cached native objects. The job
takes the first candidate that runs `--version` with exit code 0, so a file that cannot be executed
is skipped. It returns the invocation path with symlinks unresolved, because a `clang++` symlink can
select a different mode from `clang`.
-/
private def nativeCompilerJob (name : String) : SpawnM (Job FilePath) := Job.async do
  let path := FilePath.mk name
  let cwd ← IO.currentDir
  let candidates ←
    if path.isAbsolute || path.components.length > 1 then
      pure [cwd / path]
    else
      pure <| (SearchPath.parse ((← IO.getEnv "PATH").getD "")).map (cwd / · / path)
  for candidate in candidates do
    if (← candidate.pathExists) && !(← candidate.isDir) then
      let compiler := candidate.normalize
      if (← testProc { cmd := compiler.toString, args := #["--version"] }) then
        let resolved ← IO.FS.realPath compiler
        addPureTrace (compiler.toString, resolved.toString) "native tool paths"
        addTrace (.ofHash (← computeFileHash resolved) resolved.toString)
        return compiler
  error s!"native compiler not found: no candidate for {name} runs `--version`"

/-- Numerical CUDA primitives built and linked with the selected LibTorch SDK. -/
target torchlean_libtorch pkg : FilePath := do
  let lean ← getLeanInstall
  let scriptJob ← inputFile (pkg.dir / "scripts/libtorch_build.py") false
  scriptJob.mapM fun scriptPath => do
    unless cudaEnabled do
      error "torchlean_libtorch requires -Kcuda=true; the default build does not link LibTorch"
    let mut args := #[scriptPath.toString, "--package-dir", pkg.dir.toString,
      "--build-dir", pkg.buildDir.toString, "--lean-include", lean.includeDir.toString]
    for (flag, config) in #[
        ("libtorch-home", get_config? libtorch_home), ("cuda-home", get_config? cuda_home)] do
      let path := (config.getD "").trimAscii.toString
      if !path.isEmpty then
        args := args.push s!"--{flag}={path}"
    -- Each time a build needs this target, cache hits included, the helper runs and rebuilds
    -- through CMake when the SDK, a tool, a flag or a source file has changed.
    let fingerprint ← captureProc { cmd := "python3", args := args }
    addPureTrace fingerprint "LibTorch SDK, compiler, flags, and native sources"
    addTrace (← getLeanTrace)
    let output ← IO.FS.realPath
      (pkg.buildDir / "libtorch" / nameToSharedLib "torchlean_libtorch")
    addTrace (.ofHash (← computeFileHash output) output.toString)
    return output

/-- `csrc/cpu/torchlean_tensor.c` compiled once with `-fPIC`, as one object file shared by the
static executable link and the dynamic elaborator library. -/
target torchlean_tensor_cpu_object pkg : FilePath := do
  let lean ← getLeanInstall
  let srcJob ← inputFile
    (pkg.dir / "csrc/cpu/torchlean_tensor.c") false
  let oFile := pkg.buildDir / "torchlean_tensor_cpu.o"
  let compilerJob ← nativeCompilerJob "cc"
  compilerJob.bindM fun compiler =>
    buildO oFile srcJob #["-I", lean.includeDir.toString] #["-O3", "-fPIC"] compiler getLeanTrace

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

/-- GPU ABI exports for builds without LibTorch: the status query reports the backend as not
linked, IO constructors return an error and pure operations panic. -/
target torchlean_libtorch_unavailable pkg : FilePath := do
  let lean ← getLeanInstall
  let srcJob ← inputFile (pkg.dir / "csrc/libtorch/unavailable.c") false
  let operationsJob ← inputFile (pkg.dir / "csrc/libtorch/operations.h") false
  let source := srcJob.zipWith (fun src _ => src) operationsJob
  let oFile := pkg.buildDir / "torchlean_libtorch_unavailable.o"
  let compilerJob ← nativeCompilerJob "cc"
  let oJob ← compilerJob.bindM fun compiler =>
    buildO oFile source #["-I", lean.includeDir.toString] #["-O2", "-fPIC"] compiler getLeanTrace
  buildStaticLib (pkg.buildDir / nameToStaticLib "torchlean_libtorch_unavailable") #[oJob]

/-- Lean's pinned mimalloc for Linux links, compiled by `scripts/lean_allocator.py` with PIC,
initial-exec TLS (so shared libraries can use it) and the arena purge-wakeup repair. -/
target torchlean_allocator pkg : FilePath := do
  let lean ← getLeanInstall
  let buildScript := pkg.dir / "scripts/lean_allocator.py"
  let scriptJob ← inputFile buildScript false
  let runtimeJob ← inputFile (pkg.dir / "csrc/runtime/allocator.cpp") false
  let runtimeHeaderJob ← inputFile (lean.includeDir / "lean/lean.h") false
  let compatJob ← inputFile (pkg.dir / "csrc/runtime/lean_libc_compat.h") false
  let headerJob ← inputFile (lean.includeDir / "lean/mimalloc.h") false
  let deps := scriptJob.mix (runtimeJob.mix (runtimeHeaderJob.mix (compatJob.mix headerJob)))
  let compilerJob ← nativeCompilerJob "c++"
  compilerJob.bindM fun compiler => do
    let output := pkg.buildDir / "torchlean_allocator.o"
    buildFileAfterDep output deps (fun _ => do
      proc {
        cmd := "python3"
        args := #[buildScript.toString, "--lean-include", lean.includeDir.toString,
          "--compiler", compiler.toString, "--output", output.toString]
      }) getLeanTrace

-- Libraries: one per module root, in the order TorchLean.Tensor < TorchLean.CUDA < TorchLean.NN
-- < TorchLean.Verify < TorchLeanExamples. A module imports only modules of its own library or of
-- earlier ones, and TorchLeanTest may import any library. The layering gate of
-- `scripts/checks/dependency_audit.py`, run by `scripts/lake.sh lint` and CI, checks this order.

/-- Tensors, shapes, the operator table and its specifications, and the packed CPU kernels. -/
@[default_target]
lean_lib TorchLean.Tensor where
  roots := #[]
  globs := #[.submodules `TorchLean.Tensor]
  moreLinkObjs :=
    -- Embedders that link Lean's shared runtime (`-K sharedRuntime=true`) must use its allocator,
    -- so only Linux builds against the static runtime link the pinned one.
    (if Platform.isLinux && (get_config? sharedRuntime).getD "false" != "true" then
      (#[torchlean_allocator] : TargetArray FilePath) else #[]) ++
    (#[torchlean_tensor_cpu] : TargetArray FilePath)

/-- The LibTorch/CUDA boundary: GPU `@[extern]` declarations, CUDA source generation and the native
link. Without `-Kcuda=true` it links the stub that reports the backend as not linked. -/
@[default_target]
lean_lib TorchLean.CUDA where
  roots := #[]
  globs := #[.submodules `TorchLean.CUDA]
  moreLinkObjs :=
    if cudaEnabled then (#[torchlean_libtorch] : TargetArray FilePath)
    else (#[torchlean_libtorch_unavailable] : TargetArray FilePath)

/-- Autograd and the tape, the runtime and API, the IR with import and export, and their proofs. -/
@[default_target]
lean_lib TorchLean.NN where
  roots := #[]
  globs := #[.submodules `TorchLean.NN]

/-- CROWN and IBP, certificates, and roundoff proofs. -/
@[default_target]
lean_lib TorchLean.Verify where
  roots := #[]
  globs := #[.submodules `TorchLean.Verify]

/-- Runnable and narrative examples, and the roots of the `verify` and `torchlean` executables. -/
lean_lib TorchLeanExamples where
  roots := #[]
  globs := #[.submodules `TorchLeanExamples]

/-- Tests run by `scripts/lake.sh test`. -/
lean_lib TorchLeanTest where
  roots := #[]
  globs := #[.submodules `TorchLeanTest]

-- Verification tools: `scripts/lake.sh exe verify -- <tool> [args...]`.
lean_exe verify where
  root := `TorchLeanExamples.Verification.Main

-- Native runner for `scripts/lake.sh test`, including tests that call backend externs.
lean_exe nn_tests_suite where
  root := `TorchLeanTest.Suite

-- Developer-only arithmetic check.
lean_exe native_float32_parity where
  root := `TorchLeanTest.Floats.NativePrimitivesMain

-- Tree kernel checks on the host and, with LibTorch, on the device.
lean_exe tree_tests where
  root := `TorchLeanTest.Tree.Main

-- Repository policy, layering and library checks: `scripts/lake.sh lint`. Lake builds the
-- libraries first, because the library check imports them.
lean_exe torchlean_lint where
  srcDir := "scripts/checks"
  root := `TorchLeanLint
  needs := #[TorchLean.Tensor, TorchLean.CUDA, TorchLean.NN, TorchLean.Verify, TorchLeanExamples,
    TorchLeanTest]

-- Runnable examples: `scripts/lake.sh exe torchlean <example> [args...]`.
-- Build with `scripts/lake.sh -Kcuda=true build` before passing `--device cuda` to an example.
lean_exe torchlean where
  root := `TorchLeanExamples.RunnerMain

-- Shared executable numerical formats and refinement proofs.
require floatlib from git
  "https://github.com/lean-dojo/FloatLib" @ "main"

-- API documentation (HTML): `build TorchLean.Tensor:docInfo TorchLean.CUDA:docInfo
-- TorchLean.NN:docInfo TorchLean.Verify:docInfo TorchLeanExamples:docInfo +FloatLib:docInfo`,
-- then `exe doc-gen4 fromDb` with no module roots (scripts/docs/build_site.sh). A
-- library's `:docs` facet renders only its root modules.
require «doc-gen4» from git
  "https://github.com/leanprover/doc-gen4" @ "v4.35.0"

-- Keep `mathlib` last so Mathlib’s dependency versions win, which is required for cache tooling.
require mathlib from git
  "https://github.com/leanprover-community/mathlib4" @ "v4.35.0"
