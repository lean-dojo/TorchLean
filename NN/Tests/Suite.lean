/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import Std

public import NN.Tests.API.BuilderSeeds
public import NN.Tests.API.SelfSupervised.BlockMask
public import NN.Tests.API.GradientAccumulation
public import NN.Tests.Backend.Profile
public import NN.Tests.GraphSpec.Generality
public import NN.Tests.IR.ShapeContracts
public import NN.Tests.MLTheory.CROWNOperators
public import NN.Tests.MLTheory.CROWNSoundnessGuardrails
public import NN.Tests.MLTheory.Diagnostics
public import NN.Tests.Runtime.Floats.Suite
public import NN.Tests.Runtime.Rationals.Suite
public import NN.Tests.Runtime.Cuda.Suite

/-!
# Suite

Top-level executable test entrypoint for TorchLean.

This regression suite complements the theorems in `NN/Proofs` by exercising runtime trust
boundaries: native CUDA kernels, FFI buffers, floating-point execution, executable parsers, and API
runtime checks.
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
  -- Fork-child mode for the block-cache byte cap. A child launched by
  -- `Tests.Cuda.Stress.runCacheCapTest` re-enters here with the cap fixed in its environment and
  -- runs only the cache probe, then exits, so the parent can inspect the outcome.
  match ← IO.getEnv "TORCHLEAN_CUDA_CACHE_PROBE" with
  | some "cache-cap" => Tests.Cuda.Stress.runCacheCapProbe
  | _ =>
  -- Death-test child modes for the arena use-after-free detector. A forked child (see
  -- `Tests.Cuda.Stress.runArenaDetectorDeathTest`) re-enters here with `TORCHLEAN_ARENA_UAF_PROBE` set
  -- and runs only the planted UAF (`uaf` — expected to panic under `TORCHLEAN_ARENA_DEBUG=1`) or a
  -- valid promotion (`valid` — expected to be left alone), then exits, so the parent can inspect it.
  match ← IO.getEnv "TORCHLEAN_ARENA_UAF_PROBE" with
  | some "uaf" => Tests.Cuda.Stress.runArenaUseAfterFreeProbe
  | some "valid" => Tests.Cuda.Stress.runArenaValidPromotionProbe
  | _ =>
    IO.println "== TorchLean: curated tests =="
    NN.Tests.API.BuilderSeeds.run
    NN.Tests.API.SelfSupervised.BlockMask.run
    NN.Tests.API.GradientAccumulation.run
    NN.Tests.Backend.Profile.run
    NN.Tests.MLTheory.CROWNOperators.run
    NN.Tests.MLTheory.CROWNSoundnessGuardrails.run
    NN.Tests.MLTheory.Diagnostics.run
    Tests.Floats.run
    Tests.Rationals.Suite.run
    match Runtime.Autograd.Cuda.Buffer.runtimeStatus with
    | .cpuStub =>
        IO.println "  CUDA kernels: skipped (CPU build)"
    | .nativeAvailable =>
        Tests.Cuda.run
    | .nativeUnavailable =>
        throw <| IO.userError
          "TorchLean was built with CUDA, but no usable CUDA device is visible"
    IO.println "== TorchLean: all curated tests passed =="

def main (args : List String) : IO Unit := do
  match args with
  | ["--help"] | ["-h"] =>
      IO.println usage
  | [] =>
      run
  | _ =>
      IO.eprintln s!"Unknown args: {args}"
      IO.eprintln ""
      IO.eprintln usage
      throw <| IO.userError "bad CLI args"

end NN.Tests

def main (args : List String) : IO Unit :=
  NN.Tests.main args
