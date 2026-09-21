/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team

Device-agnostic example:
  lake exe torchlean rnn --device cpu
  lake -R -K cuda=true exe torchlean rnn --device cuda

This example trains a tiny byte-level RNN on real text:
- load a corpus through `--tiny-shakespeare` or `--data-file`,
- turn the first few bytes into a next-token training window,
- train `nn.rnn` plus a time-distributed linear head.
-/

module

public import NN.Examples.Models.Common.RealData

/-!
# RNN Text Example

Runnable `torchlean rnn` example. It reads a local text corpus, takes a short byte window from the
front, and trains a vanilla RNN plus a time-distributed linear head.

The model constructor is exposed as `TorchLean.nn.models.rnn`. The local code names the
architecture, builds the text dataset, and trains through the public `Trainer` surface.

## Scope

This is the plain recurrent baseline. It keeps the text window short so the example stays focused on
the recurrent cell, the time-distributed head, and the public `Trainer` API. For generation and
longer contexts, use `chargpt`, `gpt2`, or `text_gpt2`.

```bash
python3 scripts/datasets/download_example_data.py --tiny-shakespeare
lake -R -K cuda=true exe torchlean rnn --device cuda --tiny-shakespeare --steps 1
```
-/

@[expose] public section

open TorchLean

namespace NN.Examples.Models.Sequence.Rnn

/-- CLI subcommand name used in terminal banners and error messages. -/
def exeName : String := "rnn"

/-- Default JSON loss-curve path for this command. -/
def defaultLogPath : System.FilePath := Support.trainLogPath "rnn"

/-- Number of byte-level timesteps in the training window. -/
def contextLength : Nat := 4
/-- Tiny one-hot token width for the example dataset. -/
def vocabularySize : Nat := 8
/-- Default number of distinct corpus windows exposed to the trainer. -/
def defaultWindows : Nat := 16

local instance : NeZero vocabularySize := ⟨by decide⟩

/-- Compact byte vocabulary: encode byte id `b` as `b % 8`; collisions are intentional. -/
def byteBucket (id : Nat) : Fin vocabularySize :=
  Fin.ofNat vocabularySize id
/-- Hidden state width of the vanilla recurrent cell. -/
def hiddenWidth : Nat := 4

/-- Shared shape/config record for the reusable RNN-with-head constructor. -/
abbrev modelConfig : nn.models.Recurrent.Config :=
  { sequenceLength := contextLength
    inputWidth := vocabularySize
    hiddenWidth := hiddenWidth
    outputWidth := vocabularySize }

/-- Input shape: one token vector per timestep. -/
abbrev input :=
  modelConfig.inputShape

/-- Output shape: one prediction row per timestep. -/
abbrev output :=
  modelConfig.outputShape

/-- Vanilla RNN followed by a time-distributed linear output head. -/
def model : nn.Builder (nn.Sequential input output) :=
  nn.models.rnn modelConfig

/-- Build a finite next-token dataset from evenly spaced corpus windows. -/
def samples (corpus : String) (windows : Nat) :
    Data.SampleStream (Sample.Supervised Float input output) :=
  Data.CausalLM.byteSamples
    (α := Float) contextLength vocabularySize byteBucket windows corpus

/-- Train the vanilla RNN with the public `Trainer` surface. -/
def train (runtime : Runtime.Config) (data : RealData.TextWindowFlags)
    (flags : CLI.Training.OptimizerOptions) : IO Unit := do
  let corpus ← RealData.TextWindowFlags.read exeName data
  let trainer :=
    Trainer.new model <|
      Trainer.RunConfig.forObjective
        (Trainer.RunConfig.fromRuntime runtime
          { optimizer := optim.sgd { learningRate := flags.learningRate } })
        (.oneHotCrossEntropy 1) (seed := runtime.seed)
  let trainData := Data.fromStream (samples corpus data.windows)
  let trained ← trainer.train
    trainData
    (flags.trainOptions
      (logTitle := "RNN text training")
      (logNotes := #[s!"corpus={data.corpus.path}", s!"windows={data.windows}"]))
  trained.printSummary

/-- CLI entrypoint for the vanilla RNN text command. -/
def main (args : List String) : IO UInt32 := do
  CLI.Training.Command.run
    { exeName := exeName
      defaultLogPath := defaultLogPath
      defaultSteps := 1
      defaultLearningRate := 1e-2
      description := "vanilla RNN"
      dataOptions := RealData.TextWindowFlags.help defaultWindows
      parseData := RealData.TextWindowFlags.parse exeName defaultWindows
      train := train }
    args

end NN.Examples.Models.Sequence.Rnn
