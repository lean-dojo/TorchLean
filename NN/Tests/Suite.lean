/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Tests.Backend.Profile
public import NN.Tests.IR.ShapeContracts
public import NN.Tests.MLTheory.CROWNSoundnessGuardrails
public import NN.Tests.MLTheory.CROWNTransferFailures
public import NN.Tests.Verification.CameraCertificates
public import NN.Tests.Verification.CertificateParsing
public import NN.Tests.Verification.FiniteArtifact
public import NN.Tests.Verification.GraphNumericalCertificate
public import NN.Tests.Runtime.EinsumDynamic
public import NN.Tests.Runtime.Floats.Suite
public import NN.Tests.Runtime.Cuda.Suite
public import NN.Tests.Tensor.Lexer
public import NN.Tests.Tensor.NumericContracts
public import NN.Tests.Tensor.Operations
public import NN.Tests.Tensor.Storage
public import NN.Tests.Tensor.Stack

/-!
# Suite

Top-level executable test entrypoint for TorchLean.

This regression suite complements the theorems in `NN/Proofs` by exercising runtime trust
boundaries: the LibTorch CUDA backend, FFI buffers, floating-point execution, executable parsers,
and API runtime checks.
-/

@[expose] public section

open Std

namespace NN.Tests

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

def run : IO Unit := do
  if (← IO.getEnv "TORCHLEAN_REQUIRE_CUDA") == some "1" then
    Runtime.Autograd.LibTorch.Buffer.requireNativeRuntime
  -- Fresh subprocesses isolate native memory accounting and restore allocator limits on exit.
  -- These probes are selected before the ordinary suite to avoid unrelated live GPU owners.
  match ← IO.getEnv "TORCHLEAN_LIBTORCH_MEMORY_PROBE" with
  | some "accounting" => Tests.Cuda.Stress.runMemoryAccountingProbe
  | some "attention-buffers" => Tests.Cuda.Stress.runAttentionMemoryProbe
  | some "oom-recovery" => Tests.Cuda.Stress.runMemoryOOMProbe
  | some other => throw <| IO.userError s!"unknown LibTorch memory probe: {other}"
  | none =>
    IO.println "== TorchLean: curated tests =="
    NN.Tests.Backend.Profile.run
    NN.Tests.MLTheory.CROWNSoundnessGuardrails.run
    NN.Tests.MLTheory.CROWNTransferFailures.run
    NN.Tests.Verification.CameraCertificates.run
    NN.Tests.Verification.CertificateParsing.run
    NN.Tests.Verification.FiniteArtifact.run
    NN.Tests.Verification.GraphNumericalCertificate.run
    NN.Tests.Runtime.EinsumDynamic.run
    NN.Tests.Tensor.Lexer.run
    NN.Tests.Tensor.NumericContracts.run
    NN.Tests.Tensor.Operations.run
    NN.Tests.Tensor.Storage.run
    NN.Tests.Tensor.Stack.run
    Tests.Floats.run
    match Runtime.Autograd.LibTorch.Buffer.runtimeStatus with
    | .notLinked =>
        IO.println "  CUDA kernels: skipped (LibTorch not linked)"
    | .nativeAvailable =>
        NN.Tests.Runtime.EinsumDynamic.run .cuda
        Tests.Cuda.run
    | .nativeUnavailable =>
        throw <| IO.userError
          "TorchLean was built with CUDA, but no usable CUDA device is visible"
    IO.println "== TorchLean: all curated tests passed =="

def main (args : List String) : IO Unit := do
  match args with
  | [] => run
  | ["--help"] | ["-h"] => IO.println usage
  | _ =>
      IO.eprintln s!"Unknown args: {args}"
      IO.eprintln ""
      IO.eprintln usage
      throw <| IO.userError "bad CLI args"

end NN.Tests

def main (args : List String) : IO Unit :=
  NN.Tests.main args
