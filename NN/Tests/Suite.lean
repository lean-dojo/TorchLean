/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Tests.API.BufferUpdates
public import NN.Tests.API.BuilderSeeds
public import NN.Tests.API.CLI
public import NN.Tests.API.Command
public import NN.Tests.API.Complex
public import NN.Tests.API.Data
public import NN.Tests.API.Diffusion
public import NN.Tests.API.Differential
public import NN.Tests.API.Gpt2Corpus
public import NN.Tests.API.Macros
public import NN.Tests.API.Optim
public import NN.Tests.API.Precision
public import NN.Tests.API.PrecisionTrainer
public import NN.Tests.API.TypedTraining
public import NN.Tests.API.PublicSurface
public import NN.Tests.API.RLCritic
public import NN.Tests.API.SelfSupervised.BlockMask
public import NN.Tests.API.Text
public import NN.Tests.API.TrainerCheckpoint
public import NN.Tests.API.TrainerReport
public import NN.Tests.API.TrainerRun
public import NN.Tests.API.Verification
public import NN.Tests.API.GradientAccumulation
public import NN.Tests.API.Init
public import NN.Tests.API.ModelContracts
public import NN.Tests.API.ModelEdgeCases
public import NN.Tests.API.Models
public import NN.Tests.Backend.Profile
public import NN.Tests.IR.ShapeContracts
public import NN.Tests.MLTheory.IBPRefinement
public import NN.Tests.MLTheory.BinaryMatmul
public import NN.Tests.MLTheory.CROWNLayerNormDerivatives
public import NN.Tests.MLTheory.CROWNOperators
public import NN.Tests.MLTheory.CROWNSoundnessGuardrails
public import NN.Tests.MLTheory.CROWNTransferFailures
public import NN.Tests.MLTheory.Diagnostics
public import NN.Tests.MLTheory.DirectedReductions
public import NN.Tests.MLTheory.BatchNormBounds
public import NN.Tests.Verification.CameraCertificates
public import NN.Tests.Verification.CertificateParsing
public import NN.Tests.Verification.DigitsCrown
public import NN.Tests.Verification.FiniteArtifact
public import NN.Tests.Verification.GraphNumericalCertificate
public import NN.Tests.MLTheory.CROWNQuery
public import NN.Tests.MLTheory.Monotonicity
public import NN.Tests.MLTheory.SoftmaxDerivatives
public import NN.Tests.Runtime.EinsumDynamic
public import NN.Tests.Runtime.PerfFastPaths
public import NN.Tests.Runtime.Floats.Suite
public import NN.Tests.Runtime.Rationals.Suite
public import NN.Tests.Runtime.Cuda.Suite
public import NN.Tests.Tensor.EinsumPlanning
public import NN.Tests.Tensor.Lexer
public import NN.Tests.Tensor.LinearAlgebra
public import NN.Tests.Tensor.Operations
public import NN.Tests.Tensor.Storage

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
    , "  lake build nn_tests_suite && lake exe nn_tests_suite"
    , ""
    , "Notes:"
    , "  Heavier verification certificate checkers are separate executables:"
    , "    lake exe verify -- all    # run bundled cert checkers"
    , "    lake exe verify -- list   # list all verifier tools"
    ]

def run : IO Unit := do
  if (← IO.getEnv "TORCHLEAN_REQUIRE_CUDA") == some "1" then
    Runtime.Autograd.Cuda.Buffer.requireNativeRuntime
  -- Fresh subprocesses isolate native memory accounting and restore allocator limits on exit.
  -- These probes are selected before the ordinary suite to avoid unrelated live GPU owners.
  match ← IO.getEnv "TORCHLEAN_LIBTORCH_MEMORY_PROBE" with
  | some "accounting" => Tests.Cuda.Stress.runMemoryAccountingProbe
  | some "attention-context" => Tests.Cuda.Stress.runAttentionMemoryProbe
  | some "oom-recovery" => Tests.Cuda.Stress.runMemoryOOMProbe
  | some other => throw <| IO.userError s!"unknown LibTorch memory probe: {other}"
  | none =>
    IO.println "== TorchLean: curated tests =="
    NN.Tests.API.BufferUpdates.run
    NN.Tests.API.BuilderSeeds.run
    NN.Tests.API.CLI.run
    NN.Tests.API.Command.run
    NN.Tests.API.Complex.run
    NN.Tests.API.Data.run
    NN.Tests.API.Diffusion.run
    NN.Tests.API.Differential.run
    NN.Tests.API.Fourier.run
    NN.Tests.API.Gpt2Corpus.run
    NN.Tests.API.Optim.run
    NN.Tests.API.Precision.run
    NN.Tests.API.PrecisionTrainer.run
    NN.Tests.API.TypedTraining.run
    NN.Tests.API.PublicSurface.run
    NN.Tests.API.RLCritic.run
    NN.Tests.API.SelfSupervised.BlockMask.run
    NN.Tests.API.Text.run
    NN.Tests.API.TrainerCheckpoint.run
    NN.Tests.API.TrainerReport.run
    NN.Tests.API.TrainerRun.run
    NN.Tests.API.Verification.run
    NN.Tests.API.GradientAccumulation.run
    NN.Tests.API.Init.run
    NN.Tests.API.ModelContracts.run
    NN.Tests.API.ModelEdgeCases.run
    NN.Tests.API.Models.run
    NN.Tests.Backend.Profile.run
    NN.Tests.MLTheory.IBPRefinement.run
    NN.Tests.MLTheory.BinaryMatmul.run
    NN.Tests.MLTheory.CROWNLayerNormDerivatives.run
    NN.Tests.MLTheory.CROWNOperators.run
    NN.Tests.MLTheory.CROWNSoundnessGuardrails.run
    NN.Tests.MLTheory.CROWNTransferFailures.run
    NN.Tests.MLTheory.Diagnostics.run
    NN.Tests.MLTheory.DirectedReductions.run
    NN.Tests.MLTheory.BatchNormBounds.run
    NN.Tests.Verification.CameraCertificates.run
    NN.Tests.Verification.CertificateParsing.run
    NN.Tests.Verification.DigitsCrown.run
    NN.Tests.Verification.FiniteArtifact.run
    NN.Tests.Verification.GraphNumericalCertificate.run
    NN.Tests.MLTheory.CROWNQuery.run
    NN.Tests.MLTheory.Monotonicity.run
    NN.Tests.MLTheory.SoftmaxDerivatives.run
    NN.Tests.Runtime.EinsumDynamic.run
    NN.Tests.Runtime.PerfFastPaths.run
    NN.Tests.Tensor.Lexer.run
    NN.Tests.Tensor.LinearAlgebra.run
    NN.Tests.Tensor.Operations.run
    NN.Tests.Tensor.Storage.run
    Tests.Floats.run
    Tests.Rationals.Suite.run
    match Runtime.Autograd.Cuda.Buffer.runtimeStatus with
    | .notLinked =>
        IO.println "  CUDA kernels: skipped (LibTorch not linked)"
    | .nativeAvailable =>
        NN.Tests.API.BufferUpdates.checkStochasticBuffers (device := .cuda)
        NN.Tests.Runtime.EinsumDynamic.run .cuda
        Tests.Cuda.run
    | .nativeUnavailable =>
        throw <| IO.userError
          "TorchLean was built with CUDA, but no usable CUDA device is visible"
    IO.println "== TorchLean: all curated tests passed =="

def main (args : List String) : IO Unit := do
  match args with
  | List.nil =>
      run
  | List.cons arg List.nil =>
      if arg == "--help" || arg == "-h" then
        IO.println usage
      else
        IO.eprintln s!"Unknown args: {args}"
        IO.eprintln ""
        IO.eprintln usage
        throw <| IO.userError "bad CLI args"
  | _ =>
      IO.eprintln s!"Unknown args: {args}"
      IO.eprintln ""
      IO.eprintln usage
      throw <| IO.userError "bad CLI args"

end NN.Tests

def main (args : List String) : IO Unit :=
  NN.Tests.main args
