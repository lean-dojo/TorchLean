/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLeanExamples.DeepDives.Floats.RoundingBugs
public import TorchLean.NN.Session.Lifecycle
public import TorchLeanTest.Data.NPY
public import TorchLeanTest.Data.Text
public import TorchLeanTest.IR.Flow
public import TorchLeanTest.IR.Pointwise
public import TorchLeanTest.IR.MoEImport
public import TorchLeanTest.Verification.BuilderImport
public import TorchLeanTest.Verification.CameraCertificates
public import TorchLeanTest.Verification.CertificateParsing
public import TorchLeanTest.CUDA.MoE
public import TorchLeanTest.CUDA.Scheduler
public import TorchLeanTest.CUDA.KernelGradient
public import TorchLeanTest.CUDA.KernelIndex
public import TorchLeanTest.Interop.Suite
public import TorchLeanTest.CUDA.Suite
public import TorchLeanTest.Tree.Suite
public import TorchLeanTest.Tensor.Elementwise
public import TorchLeanTest.Tensor.EinopsLexer
public import TorchLeanTest.Tensor.Elementary
public import TorchLeanTest.Tensor.Operations
public import TorchLeanTest.Tensor.Storage
public import TorchLeanTest.RL.PolicyKL

/-!
# Suite

`run` calls the checks of TorchLean against outside references: PyTorch, JAX and ONNX, files in the
NPY, CSV, PPM and certificate formats, results of CPython, mpmath and numpy, the C kernels in
`csrc/`, LibTorch on CUDA and the host's floating point. With `TORCHLEAN_LIBTORCH_MEMORY_PROBE` set
it runs one CUDA memory probe instead.
-/

@[expose] public section

open Std

namespace TorchLean.Test

/-- Help text of the `nn_tests_suite` executable. -/
def usage : String :=
  String.intercalate "\n"
    [ "TorchLean test suite"
    , ""
    , "Usage:"
    , "  scripts/lake.sh build nn_tests_suite && scripts/lake.sh exe nn_tests_suite"
    , ""
    , "Notes:"
    , "  Heavier verification certificate checkers are separate executables:"
    , "    scripts/lake.sh exe verify -- all    # run bundled cert checkers"
    , "    scripts/lake.sh exe verify -- list   # list all verifier tools"
    ]

/-- Run the curated tests, or the one CUDA memory probe that `TORCHLEAN_LIBTORCH_MEMORY_PROBE`
names. -/
def run : IO Unit := do
  -- `TORCHLEAN_REQUIRE_CUDA=1` means that a GPU ran the suite, which the host bridge does not
  -- satisfy. `TORCHLEAN_REQUIRE_NATIVE=1` means that the bridge ran it on the device it selected,
  -- the check for a CPU-only SDK, which a build without LibTorch fails.
  if (← IO.getEnv "TORCHLEAN_REQUIRE_CUDA") == some "1" then
    TorchLean.LibTorch.Buffer.requireCudaDevice
  if (← IO.getEnv "TORCHLEAN_REQUIRE_NATIVE") == some "1" then
    TorchLean.LibTorch.Buffer.requireNativeRuntime
  -- Fresh subprocesses isolate native memory accounting and restore allocator limits on exit.
  -- These probes are selected before the ordinary suite to avoid unrelated live GPU owners.
  match ← IO.getEnv "TORCHLEAN_LIBTORCH_MEMORY_PROBE" with
  | some "accounting" => TorchLean.Test.CUDA.Stress.runMemoryAccountingProbe
  | some "attention-buffers" => TorchLean.Test.CUDA.Stress.runAttentionMemoryProbe
  | some "oom-recovery" => TorchLean.Test.CUDA.Stress.runMemoryOOMProbe
  | some other => throw <| IO.userError s!"unknown LibTorch memory probe: {other}"
  | none =>
    IO.println "== TorchLean: curated tests =="
    TorchLean.Test.NPY.run
    TorchLean.Test.TextData.run
    TorchLean.Test.MoEImport.run
    TorchLean.Test.Pointwise.run
    TorchLean.Test.Flow.run
    TorchLean.Test.BuilderImport.run
    TorchLean.Test.CameraCertificates.run
    TorchLean.Test.CertificateParsing.run
    TorchLean.Test.Elementwise.run
    TorchLean.Test.EinopsLexer.run
    TorchLean.Test.Elementary.run
    TorchLean.Test.Tensor.run
    TorchLean.Examples.SoftmaxCrossEntropy.run
    TorchLean.Test.Storage.run
    TorchLean.Test.PolicyKL.run
    TorchLean.Test.Floats.run
    TorchLean.Test.Tree.run
    TorchLean.Test.KernelGradient.run .cpu
    TorchLean.Test.KernelIndex.run .cpu
    match (← TorchLean.LibTorch.Buffer.RuntimeStatus.query) with
    | .notLinked =>
        -- A build without LibTorch refuses a CUDA session.
        try
          let _ ← TorchLean.Session.create (α := Float)
            ({ device := .cuda } : TorchLean.Session.Config)
          throw <| IO.userError "build without LibTorch unexpectedly admitted a user CUDA session"
        catch e =>
          unless e.toString.contains "built without LibTorch" do
            throw <| IO.userError s!"no-LibTorch CUDA session rejection: {e}"
        TorchLean.Test.Tree.runUnlinked
        IO.println "  CUDA kernels: skipped (LibTorch not linked)"
    | .nativeAvailable =>
        -- The selected device agrees with the visible devices and with the request.
        let kind ← TorchLean.LibTorch.deviceKind
        let count ← TorchLean.LibTorch.deviceCount
        let requested ← IO.getEnv "TORCHLEAN_LIBTORCH_DEVICE"
        if kind == .cuda && count == 0 then
          throw <| IO.userError "LibTorch selected a CUDA device while none is visible"
        if requested == some "cpu" && kind != .host then
          throw <| IO.userError "TORCHLEAN_LIBTORCH_DEVICE=cpu did not select the host"
        if requested == some "cuda" && kind != .cuda then
          throw <| IO.userError "TORCHLEAN_LIBTORCH_DEVICE=cuda did not select a CUDA device"
        IO.println s!"  LibTorch device: {match kind with | .cuda => "cuda" | .host => "host"}"
        TorchLean.Test.MoE.run .cuda
        TorchLean.Test.Scheduler.run .cuda
        if kind == .cuda then
          TorchLean.Test.KernelGradient.run .cuda
          TorchLean.Test.KernelIndex.run .cuda
        else TorchLean.Test.KernelGradient.checkHostRefusal
        TorchLean.Test.CUDA.run
    | .nativeUnavailable =>
        throw <| IO.userError
          "TorchLean was built with LibTorch, but no usable device is selected"
    IO.println "== TorchLean: all curated tests passed =="

/-- Run the suite without arguments, print `usage` for `--help` or `-h`, and fail on any other
arguments. -/
def main (args : List String) : IO Unit := do
  match args with
  | [] => run
  | ["--help"] | ["-h"] => IO.println usage
  | _ =>
      IO.eprintln s!"Unknown args: {args}"
      IO.eprintln ""
      IO.eprintln usage
      throw <| IO.userError "bad CLI args"

end TorchLean.Test

/-- The `nn_tests_suite` executable: `TorchLean.Test.main`. -/
def main (args : List String) : IO Unit :=
  TorchLean.Test.main args
