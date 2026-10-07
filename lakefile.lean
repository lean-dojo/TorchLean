/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

import Lake
import Lake.Util.Proc
open Lake DSL
open System

/-- Whether Lake should link the LibTorch CUDA backend instead of the unavailable-backend shim. -/
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

`-K cuda=true` builds one LibTorch C++ library containing the numerical C ABI exports. CMake obtains
the ABI, language standard, libraries, and runtime paths from the selected SDK. The default build
needs neither LibTorch nor a CUDA toolkit: it links a small C file that reports the backend as not
linked and rejects GPU computation. Status and memory queries remain available.
-/

/--
Resolve a compiler and trace its contents, so replacing it invalidates cached native objects.
Preserve the invocation path: a `clang++` symlink can select a different mode from `clang`.
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
      let resolved ← IO.FS.realPath compiler
      addPureTrace (compiler.toString, resolved.toString) "native tool paths"
      addTrace (.ofHash (← computeFileHash resolved) resolved.toString)
      return compiler
  error s!"native compiler not found: {name}"

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
    -- The helper checks SDK/tool/source contents even when Lake previously built this target.
    let fingerprint ← captureProc { cmd := "python3", args := args }
    addPureTrace fingerprint "LibTorch SDK, compiler, flags, and native sources"
    addTrace (← getLeanTrace)
    let output ← IO.FS.realPath
      (pkg.buildDir / "libtorch" / nameToSharedLib "torchlean_libtorch")
    addTrace (.ofHash (← computeFileHash output) output.toString)
    return output

/-- Object shared by the static executable link and the dynamic elaborator library. -/
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

/-- GPU ABI exports for builds without LibTorch: status reports "not linked", calls fail. -/
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

/-- Build the pinned Linux allocator with shared-library TLS and the arena purge-wakeup repair. -/
target torchlean_allocator pkg : FilePath := do
  let lean ← getLeanInstall
  let buildScript := pkg.dir / "scripts/lean_allocator.py"
  let scriptJob ← inputFile buildScript false
  let compatJob ← inputFile (pkg.dir / "csrc/runtime/lean_libc_compat.h") false
  let headerJob ← inputFile (lean.includeDir / "lean/mimalloc.h") false
  let deps := scriptJob.mix (compatJob.mix headerJob)
  let compilerJob ← nativeCompilerJob "c++"
  compilerJob.bindM fun compiler => do
    let output := pkg.buildDir / "torchlean_allocator.o"
    buildFileAfterDep output deps (fun _ => do
      proc {
        cmd := "python3"
        args := #[buildScript.toString, "--lean-include", lean.includeDir.toString,
          "--compiler", compiler.toString, "--output", output.toString]
      }) getLeanTrace

@[default_target]
lean_lib NN where
  moreLinkObjs :=
    (if Platform.isLinux then (#[torchlean_allocator] : TargetArray FilePath) else #[]) ++
    (#[torchlean_tensor_cpu] : TargetArray FilePath) ++
      if cudaEnabled then
        (#[torchlean_libtorch] : TargetArray FilePath)
      else
        (#[torchlean_libtorch_unavailable] : TargetArray FilePath)
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

-- Unified verification CLI registry: `scripts/lake.sh exe verify -- <tool> [args...]`
lean_exe verify where
  root := `NN.Verification.Main

-- Native runner for `scripts/lake.sh test`, including tests that call backend externs.
lean_exe nn_tests_suite where
  root := `NN.Tests.Suite

-- Cross-runtime numerical regression tools.
lean_exe pytorch_export_check where
  root := `NN.Tests.Interop.PyTorchMain

lean_exe native_float32_parity where
  root := `NN.Tests.Floats.NativePrimitiveParityMain

-- Repo-policy lints (header hygiene, banned constructs, etc.) via `scripts/lake.sh lint`.
lean_exe torchlean_lint where
  srcDir := "scripts/checks"
  root := `TorchLeanLint

-- Runnable examples: `scripts/lake.sh exe torchlean <example> [args...]`.
-- Build with `scripts/lake.sh -Kcuda=true build` before passing `--device cuda` to an example.
lean_exe torchlean where
  root := `NN.Examples.RunnerMain

-- Shared executable numerical formats and refinement proofs.
require floatlib from git
  "https://github.com/lean-dojo/FloatLib" @ "main"

-- Complete API documentation (HTML) via `scripts/lake.sh build TorchLeanDocs:docs`.
require «doc-gen4» from git
  "https://github.com/leanprover/doc-gen4" @ "v4.34.0"

-- Keep `mathlib` last so Mathlib’s dependency versions win, which is required for cache tooling.
require mathlib from git
  "https://github.com/leanprover-community/mathlib4" @ "v4.34.0"
