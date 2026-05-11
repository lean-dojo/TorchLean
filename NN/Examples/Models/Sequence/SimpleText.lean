/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN
public import NN.Examples.Models.Common.RealData

/-!
# Shared Simple Sequence Text Runner

This file is the shared runner for the small RNN/LSTM/Transformer text smoke tests.

Both examples have the same public shape:

- read a real text corpus, usually Tiny Shakespeare;
- turn one short byte window into a supervised sequence sample;
- train a reusable API model for a few steps; and
- optionally write a before/after loss log.

The actual architecture still lives in the calling file. This helper owns only the runnable example
infrastructure, so readers can compare `Rnn.lean`, `Lstm.lean`, and `Transformer.lean` without reading
three copies of the same runtime wrapper.
-/

@[expose] public section

open Spec Tensor
open NN.API

namespace NN.Examples.Models.Sequence.SimpleText

/-- Configuration for a small real-text sequence training example. -/
structure RunnerConfig (σ τ : Shape) where
  /-- CLI subcommand name, e.g. `torchlean rnn`. -/
  exeName : String
  /-- Default JSON log path. -/
  defaultLogJson : System.FilePath
  /-- Human-readable model name used in banners. -/
  modelName : String
  /-- Human-readable training log title. -/
  logTitle : String
  /-- Construct the model under test. -/
  mkModel : nn.M (nn.Sequential σ τ)
  /-- Build the supervised sample from corpus text. -/
  mkSample : {α : Type} → [Semantics.Scalar α] → [Runtime.Scalar α] →
    String → API.sample.Supervised α σ τ
  /-- SGD learning rate. Kept per-example because attention/recurrent examples have different scale. -/
  lr : Float

/--
Train one small sequence model for `steps` optimizer updates.

This is intentionally a smoke-scale routine: one fixed corpus-derived sample, one optimizer, and a
before/after loss. More realistic streaming/minibatch examples live elsewhere; this helper is for
architecture sanity checks that should run quickly on CPU or CUDA.
-/
def unitTrainSteps {σ τ : Shape} {α : Type}
    [Semantics.Scalar α] [DecidableEq Shape] [ToString α]
    [Runtime.Scalar α] [_root_.Runtime.Autograd.Torch.Internal.CudaBridge.TensorConv α]
    (cfg : RunnerConfig σ τ) (cast : Float → α)
    (opts : Runtime.Autograd.Torch.Options) (input : String) (steps : Nat) :
    IO (α × α) := do
  nn.withModel cfg.mkModel fun model => do
    let modDef := nn.mseScalarModuleDef model
    let m ← TorchLean.Module.instantiateWithOptions (α := α) modDef cast opts
    let sample := cfg.mkSample (α := α) input
    let loss0 ← TorchLean.Module.forward (α := α) m sample
    let L0 := Tensor.toScalar loss0
    let opt :=
      TorchLean.Optim.sgd (α := α) (paramShapes := nn.paramShapes model)
        (lr := Runtime.ofFloat (α := α) cfg.lr)
    let optH ← TorchLean.Optim.handle (α := α) m opt
    for _ in [0:steps] do
      optH.step sample
    let loss1 ← TorchLean.Module.forward (α := α) m sample
    let L1 := Tensor.toScalar loss1
    IO.println s!"  steps={steps} loss0={L0} loss1={L1}"
    pure (L0, L1)

/-- Shared `main` implementation for the small RNN/LSTM text examples. -/
def main {σ τ : Shape} (cfg : RunnerConfig σ τ) (args : List String) : IO UInt32 := do
  let banner := fun (opts : Runtime.Autograd.Torch.Options) =>
    s!"{cfg.exeName}: {cfg.modelName} text example (device={if opts.useGpu then "cuda" else "cpu"})"
  let runAny :=
    TorchLean.Module.run cfg.exeName args
      (.any (fun {α} _ _ _ _ cast opts rest => do
        let (path, rest) ← Common.orThrow cfg.exeName <| RealData.parseTextFlags rest
        let (steps, rest) ← Common.orThrow cfg.exeName <| CLI.takeStepsOrEpochs rest 1
        if steps = 0 then
          throw <| IO.userError s!"{cfg.exeName}: --steps/--epochs must be > 0"
        Common.orThrow cfg.exeName <| CLI.requireNoArgs rest
        let input ← RealData.readTextCorpus cfg.exeName path
        let _ ← unitTrainSteps (α := α) cfg cast opts input (steps := steps)
        pure ()
      ))
      { banner? := some banner, printOk := true }
  if args.contains "--cuda" || CLI.hasFlagValue args "log" then
    -- CUDA eager supports `Float` upload/download. We route logging through the
    -- Float path for the same reason: JSON logs need concrete scalar values.
    TorchLean.Module.run cfg.exeName args
      (.float (fun opts rest => do
        let (path, rest) ← Common.orThrow cfg.exeName <| RealData.parseTextFlags rest
        let (log?, rest) ← Common.orThrow cfg.exeName <| CLI.takePathFlagOnce rest "log"
        let logPath := log?.getD cfg.defaultLogJson
        let (steps, rest) ← Common.orThrow cfg.exeName <| CLI.takeStepsOrEpochs rest 1
        if steps = 0 then
          throw <| IO.userError s!"{cfg.exeName}: --steps/--epochs must be > 0"
        Common.orThrow cfg.exeName <| CLI.requireNoArgs rest
        let input ← RealData.readTextCorpus cfg.exeName path
        let (L0, L1) ← unitTrainSteps (α := Float) cfg (cast := id) opts input (steps := steps)
        Common.writeBeforeAfterLossLog logPath cfg.logTitle steps L0 L1
          #[s!"corpus={path}", s!"device={if opts.useGpu then "cuda" else "cpu"}"]
      ))
      { banner? := some banner, printOk := true }
  else
    runAny

end NN.Examples.Models.Sequence.SimpleText
