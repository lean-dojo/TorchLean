/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLeanExamples.Models.Sequence.GPT2

/-!
# GPT-2 Checkpoint Example

The `gpt2_saved` command loads a checkpoint written by the byte-level GPT example and samples text
from it.

1. Train and save a model checkpoint:

```bash
scripts/lake.sh -Kcuda=true build torchlean:exe
scripts/lake.sh -Kcuda=true exe torchlean gpt2 --device cuda \
    --tiny-shakespeare --steps 1 --windows 1 \
  --prompt "First Citizen:" --generate 0 \
  --save-checkpoint data/examples/gpt2_shakespeare.state
```

2. Load the checkpoint and sample text without optimizer updates:

```bash
scripts/lake.sh -Kcuda=true exe torchlean gpt2_saved --device cuda \
  --checkpoint data/examples/gpt2_shakespeare.state \
  --prompt "First Citizen:" --generate 0
```

## What A Checkpoint Is Here

The example uses the simplest TorchLean checkpoint format:

- a shape-indexed pack of model state tensors,
- stored as 64-bit `Float.toBits` words in TorchLean's binary state format, and
- checked against the model's state layout before inference starts.

Save and load are therefore model-agnostic. Given the model, TorchLean computes the expected state
shapes and rejects files with incompatible shapes. Matching shapes alone do not identify the
architecture or the meaning of the parameters.

## Why This Is A Separate Example

The inference-only workflow needs no training loop: it loads a checkpoint, converts it into runtime
handles, and samples text.
-/

@[expose] public section

open TorchLean.Trainer.Objective

namespace TorchLean.Examples.GPT2Saved

/-- CLI subcommand name used in terminal banners and error messages. -/
def exeName : String := "gpt2_saved"

/-- Help text for checkpoint-only GPT-2 sampling. -/
def usage : String :=
  String.intercalate "\n"
    [ "Usage:"
    , "  scripts/lake.sh -Kcuda=true exe torchlean gpt2_saved --device cuda --checkpoint PATH "
        ++ "[generation flags]"
    , ""
    , "Required:"
    , "  --checkpoint PATH    model state written by `torchlean gpt2 --save-checkpoint`"
    , ""
    , "Common generation flags:"
    , "  --prompt TEXT        prompt prefix"
    , "  --generate N         number of bytes to generate"
    , "  --temperature X      sampling temperature"
    , "  --top-k N            top-k cutoff"
    , "  --sample-seed N      sampling seed"
    ]

/-- Command-local options for loading one checkpoint and sampling from it. -/
structure Options where
  /-- Prompt and token-sampling policy. -/
  generation : Text.GenerationOptions
  /-- Model checkpoint loaded before sampling starts. -/
  checkpointPath : System.FilePath
deriving Repr

namespace Options

/-- Parse the checkpoint path followed by the shared generation flags. -/
def parse (args : List String) (defaults : Text.GenerationOptions) :
    Except String (Options × List String) := do
  let (checkpointPath, args) ←
    CLI.requirePathFlag args "checkpoint" (exeName := exeName)
  let (generation, args) ← Text.GenerationOptions.parse exeName args defaults
  pure ({ checkpointPath, generation }, args)

end Options

/--
Load model state from disk and run sampling with the fixed byte-level GPT architecture.

The checkpoint must match `GPT2.model`'s state shapes. If the model configuration
in `GPT2.lean` changes (heads, width, layers, etc.), mismatched checkpoints fail the shape check
before sampling starts.
-/
def sampleCheckpoint
    (load : Options) (runtime : Session.Config := {}) :
    IO String := do
  let trainer := Trainer.create Examples.GPT2.model
    { runtime with objective := mse }
  let session ← trainer.open
  session.load load.checkpointPath
  let predict : Examples.GPT2.Predictor := session.predict
  let outIds ←
    GPT2.generate predict
      (Tensor.from (Text.Tokenizer.byte.encode load.generation.prompt)) load.generation
  let txt := Text.formatBytes outIds
  IO.println s!"  loaded={load.checkpointPath}"
  IO.println s!"  prompt={Text.escape load.generation.prompt}"
  IO.println s!"  sampled={txt}"
  pure txt

/-- CLI entrypoint for checkpoint sampling. -/
def main (args : List String) : IO UInt32 := do
  if CLI.hasHelp args then
    IO.println usage
    return 0
  Module.Command.run
    (config := {
      banner? := some fun _ => s!"{exeName}: sample from a model checkpoint"
      printSuccess := true })
    exeName args
    (fun runtime rest => do
      let (load, rest) ← CLI.orThrow exeName <|
        Options.parse rest
          { prompt := "First Citizen:"
            continuationLength := 96
            temperature := 0.85
            topK := 12
            repeatPenalty := 1.25
            repeatWindow := 24
            seed := 7
            asciiOnly := false }
      CLI.requireNoArgs exeName rest
      let _ ← sampleCheckpoint load runtime)

end TorchLean.Examples.GPT2Saved
