/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team

Device-agnostic example:
  scripts/lake.sh exe torchlean lstm --device cpu
  scripts/lake.sh -Kcuda=true exe torchlean lstm --device cuda

This example trains a tiny byte-level LSTM on real text:
- load a corpus through `--tiny-shakespeare` or `--data-file`,
- build next-token samples from evenly spaced corpus windows,
- train `nn.lstm` plus a time-distributed linear head.
-/

module

public import NN.Examples.Models.Common.RealData

/-!
# LSTM Text Example

Runnable `torchlean lstm` example. It reads a local text corpus, samples short
byte windows across the
corpus, and trains an LSTM plus a time-distributed linear head.

The model constructor is exposed as `TorchLean.nn.models.lstm`. The local code names the
architecture, builds the text dataset, and trains through the public `Trainer` surface.

## Scope

This is the gated recurrent baseline. It keeps the text window short so the example stays focused on
the LSTM cell, the time-distributed head, and the public `Trainer` API. For generation and
longer-context language-model behavior, use one of:
- `torchlean chargpt` (Karpathy-style, single-file char-level GPT),
- `torchlean gpt2` (byte-level GPT-2-style model + save/reload),
- `torchlean text_gpt2` (CUDA corpus trainer).

```bash
python3 scripts/datasets/download_example_data.py --tiny-shakespeare
scripts/lake.sh -Kcuda=true exe torchlean lstm --device cuda --tiny-shakespeare --steps 1
```
-/

@[expose] public section

open TorchLean

namespace NN.Examples.Models.Sequence.Lstm

/-- CLI subcommand name used in terminal banners and error messages. -/
def exeName : String := "lstm"

/-- Default JSON loss-curve path for this command. -/
def defaultLogPath : System.FilePath := Support.trainLogPath "lstm"

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
/-- Hidden state width of the LSTM cell. -/
def hiddenWidth : Nat := 4

/-- Shared shape/config record for the reusable LSTM-with-head constructor. -/
abbrev modelConfig : nn.models.Recurrent.Config :=
  { sequenceLength := contextLength
    inputWidth := vocabularySize
    hiddenWidths := [hiddenWidth]
    outputWidth := vocabularySize }

/-- Input shape: one token vector per timestep. -/
abbrev input :=
  modelConfig.inputShape

/-- Output shape: one prediction row per timestep. -/
abbrev output :=
  modelConfig.outputShape

/-- LSTM followed by a time-distributed linear output head. -/
def model : nn.Builder (nn.Sequential input output) :=
  nn.models.lstm modelConfig

/-- Build a finite next-token dataset from evenly spaced corpus windows. -/
def samples (corpus : String) (windows : Nat) :
    Data.SampleStream (Sample.Supervised Float input output) :=
  Data.CausalLM.byteSamples
    (α := Float) contextLength vocabularySize byteBucket windows corpus

/-- Train the LSTM with the public `Trainer` surface. -/
def train (runtime : Runtime.Config) (data : RealData.Corpus.Options)
    (flags : CLI.Training.OptimizerOptions) : IO Unit := do
  let corpus ← RealData.Corpus.Options.read exeName data
  let trainer :=
    Trainer.new model <|
      { runtime with
          optimizer := optim.sgd { learningRate := flags.learningRate }
          objective := oneHotCrossEntropy 1 }
  let trainData := Data.fromStream (samples corpus data.windows)
  let trained ← trainer.train
    trainData
    (flags.trainOptions
      (logTitle := "LSTM text training")
      (logNotes := #[s!"corpus={data.corpus}", s!"windows={data.windows}"]))
  trained.printSummary

/-- CLI entrypoint for the LSTM text command. -/
def main (args : List String) : IO UInt32 := do
  CLI.Training.Command.run
    { exeName := exeName
      defaultLogPath := defaultLogPath
      defaultSteps := 1
      defaultLearningRate := 1e-2
      description := "LSTM"
      dataOptions := RealData.Corpus.Options.help defaultWindows
      parseData := RealData.Corpus.Options.parse exeName defaultWindows
      train := train }
    args

end NN.Examples.Models.Sequence.Lstm
