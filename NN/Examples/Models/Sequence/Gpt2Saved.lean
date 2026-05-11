/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN
public import NN.Examples.Models.Sequence.Gpt2
public import NN.API.Runtime

/-!
# GPT-2 Saved-Weights Demo

This file is the "load + sample" half of the GPT-2 tutorial.

1. Train and save parameters:

```bash
lake build -R -K cuda=true torchlean:exe
lake exe torchlean gpt2 --cuda --fast-kernels --tiny-shakespeare --steps 200 \
  --prompt "First Citizen:" --generate 96 \
  --save-params data/model_zoo/gpt2_shakespeare.params.json
```

2. Load the saved weights and sample text (no training loop, no optimizer state):

```bash
lake exe torchlean gpt2_saved --cuda --fast-kernels \
  --params data/model_zoo/gpt2_shakespeare.params.json \
  --prompt "First Citizen:" --generate 160
```

## What A "Checkpoint" Is In TorchLean

TorchLean's simplest checkpoint format is intentionally explicit:

- a **typed parameter pack**: `TList Float (nn.paramShapes model)`,
- encoded as **exact IEEE-754 bit patterns** (`Float.toBits`) in JSON, and
- validated by shape on load.

So "save/load" is model-agnostic: if you can name the model, you can name its
`paramShapes`, and you can save/load the parameters.

## Why This Is A Separate Example

TorchLean's checkpoint format is shape-indexed and architecture-agnostic: it is just a typed
parameter pack (`TList Float (nn.paramShapes model)`). This file exists to show the simplest
"inference-only" workflow: load a checkpoint and run sampling, without building a training loop.
-/

@[expose] public section

open Spec Tensor
open NN.API

namespace NN.Examples.Models.Sequence.Gpt2Saved

def exeName : String := "torchlean gpt2_saved"

structure LoadOptions where
  /-- JSON bits checkpoint produced by `torchlean gpt2 --save-params ...`. -/
  paramsPath : System.FilePath
  /-- Prompt string (byte-tokenized by the same tokenizer as `Gpt2`). -/
  prompt : String
  /-- Number of tokens to generate past the prompt. -/
  generate : Nat
  /-- Softmax temperature used during sampling (must be > 0). -/
  temperature : Float
  /-- Top-k sampling cutoff; smaller values are more conservative. -/
  topK : Nat
  /-- Penalize repeating tokens in the recent window. `1.0` disables the penalty. -/
  repeatPenalty : Float
  /-- Size of the repeat-penalty window. -/
  repeatWindow : Nat
  /-- RNG seed for sampling. -/
  seed : Nat
  /-- If `true`, replace non-ASCII bytes by escapes when displaying the sampled string. -/
  asciiOnly : Bool
deriving Repr

def parseLoadOptions (args : List String) : Except String (LoadOptions × List String) := do
  let (paramsRaw?, args) ← CLI.takeFlagValueOnce args "params"
  let paramsRaw ←
    match paramsRaw? with
    | some p => pure p
    | none => throw s!"{exeName}: missing required --params <path>"
  let (prompt?, args) ← CLI.takeFlagValueOnce args "prompt"
  let (generate?, args) ← CLI.takeNatFlagOnce args "generate"
  let (temperature?, args) ← CLI.takeFloatFlagOnce args "temperature"
  let (topK?, args) ← CLI.takeNatFlagOnce args "top-k"
  let (repeatPenalty?, args) ← CLI.takeFloatFlagOnce args "repeat-penalty"
  let (repeatWindow?, args) ← CLI.takeNatFlagOnce args "repeat-window"
  let (seed?, args) ← CLI.takeNatFlagOnce args "sample-seed"
  let (asciiOnlyRaw?, args) ← CLI.takeFlagValueOnce args "ascii-only"
  let (asciiOnlyFlag, args) ← CLI.takeBoolFlagOnce args "ascii-only"
  let asciiOnly :=
    match asciiOnlyRaw? with
    | none => asciiOnlyFlag
    | some v =>
        if v = "true" || v = "1" then true
        else if v = "false" || v = "0" then false
        else asciiOnlyFlag
  match asciiOnlyRaw? with
  | none => pure ()
  | some v =>
      if v = "true" || v = "1" || v = "false" || v = "0" then
        pure ()
      else
        throw s!"{exeName}: --ascii-only expects true/false (or 1/0), got {v}"
  let temperature := temperature?.getD 0.85
  if temperature <= 0.0 then
    throw s!"{exeName}: --temperature must be > 0"
  let repeatPenalty := repeatPenalty?.getD 1.25
  if repeatPenalty < 0.0 then
    throw s!"{exeName}: --repeat-penalty must be >= 0"
  pure ({ paramsPath := (paramsRaw : System.FilePath)
          prompt := prompt?.getD "First Citizen:"
          generate := generate?.getD 96
          temperature := temperature
          topK := topK?.getD 12
          repeatPenalty := repeatPenalty
          repeatWindow := repeatWindow?.getD 24
          seed := seed?.getD 7
          asciiOnly := asciiOnly }, args)

/--
Load parameters from disk and run sampling with the fixed tutorial architecture.

Important: the checkpoint must match `Gpt2.mkModel`'s parameter shapes. If the model configuration
in `Gpt2.lean` changes (heads, width, layers, etc.), mismatched checkpoints fail the shape check
before sampling starts.
-/
def sampleWithSavedParams (opts : Runtime.Autograd.Torch.Options) (load : LoadOptions) :
    IO String := do
  nn.withModel NN.Examples.Models.Sequence.Gpt2.mkModel fun model => do
    -- This is the generic “load parameters for any TorchLean model” helper:
    -- a checkpoint is just a shape-indexed `TList Float (nn.paramShapes model)`.
    let ps ← TorchLean.ParamIO.loadTListBits (paramShapes := nn.paramShapes model) load.paramsPath
    let params ← _root_.Runtime.Autograd.Torch.ParamList.ofTList (α := Float) (ss := nn.paramShapes model) ps
    let outIds ←
      NN.Examples.Models.Sequence.Gpt2.generateSampled opts model params load.prompt load.generate
        load.temperature load.topK load.seed load.repeatWindow load.repeatPenalty load.asciiOnly
    let txt := NN.Examples.Models.Sequence.Gpt2.escapeByteIdsForDisplay outIds
    IO.println s!"  loaded={load.paramsPath}"
    IO.println s!"  prompt={text.escapeForDisplay load.prompt}"
    IO.println s!"  sampled={txt}"
    pure txt

def main (args : List String) : IO UInt32 := do
  TorchLean.Module.run exeName args
    (.float (fun opts rest => do
      let (load, rest) ← Common.orThrow exeName <| parseLoadOptions rest
      Common.orThrow exeName <| CLI.requireNoArgs rest
      let _ ← sampleWithSavedParams opts load
      pure ()
    ))
    { banner? := some (fun opts =>
        s!"{exeName}: sample from saved params (device={if opts.useGpu then "cuda" else "cpu"})")
      printOk := true }

end NN.Examples.Models.Sequence.Gpt2Saved
