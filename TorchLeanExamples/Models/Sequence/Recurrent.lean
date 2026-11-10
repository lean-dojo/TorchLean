/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team

Device-agnostic example:
  scripts/lake.sh exe torchlean rnn --device cpu
  scripts/lake.sh -Kcuda=true exe torchlean lstm --device cuda

This example trains a tiny byte-level recurrent network on real text:
- load a corpus through `--tiny-shakespeare` or `--data-file`,
- build next-token samples from evenly spaced corpus windows,
- train `nn.rnn` (command `rnn`) or `nn.lstm` (command `lstm`) plus a time-distributed linear head.
-/

module

public import TorchLeanExamples.Models.RealData
public import TorchLean.NN.Data.CausalLM
public import TorchLean.NN.Models.Recurrent
public import TorchLeanExamples.Models.Training

/-!
# Recurrent Text Examples

Runnable `torchlean rnn` and `torchlean lstm` examples. Each reads a local text corpus, samples
short byte windows across the corpus, and trains a recurrent cell plus a time-distributed linear
head. The two commands differ only in the cell: `rnn` builds a vanilla RNN with
`TorchLean.nn.Recurrent.rnn`, and `lstm` builds an LSTM with `TorchLean.nn.Recurrent.lstm`.

## Scope

These are the plain and the gated recurrent baselines. They keep the text window short so the
examples stay focused on the recurrent cell, the time-distributed head, and the public `Trainer`
API. For generation and longer contexts, use one of:
- `torchlean chargpt` (Karpathy-style, single-file char-level GPT),
- `torchlean gpt2` (byte-level GPT-2-style model + save/reload),
- `torchlean text_gpt2` (CUDA corpus trainer).

```bash
python3 scripts/datasets/download_example_data.py --tiny-shakespeare
scripts/lake.sh -Kcuda=true exe torchlean rnn --device cuda --tiny-shakespeare --steps 1
scripts/lake.sh -Kcuda=true exe torchlean lstm --device cuda --tiny-shakespeare --steps 1
```
-/

@[expose] public section

open TorchLean.Trainer.Objective

namespace TorchLean.Examples.Recurrent

/-- The recurrent cell that a text command trains. -/
inductive Cell where
  /-- The vanilla RNN of the `rnn` command. -/
  | rnn
  /-- The LSTM of the `lstm` command. -/
  | lstm

namespace Cell

/-- CLI subcommand name used in terminal banners and error messages. -/
def exeName : Cell → String
  | .rnn => "rnn"
  | .lstm => "lstm"

/-- Title of the command's training log. -/
def logTitle : Cell → String
  | .rnn => "RNN text training"
  | .lstm => "LSTM text training"

/-- Model description used in the command's banner and help text. -/
def description : Cell → String
  | .rnn => "vanilla RNN"
  | .lstm => "LSTM"

/-- Default JSON loss-curve path of the cell's command. -/
def defaultLogPath (cell : Cell) : System.FilePath :=
  Training.artifactPath cell.exeName

end Cell

/-- Number of byte-level timesteps in the training window. -/
def contextLength : Nat := 4
/-- Tiny one-hot token width for the example dataset. -/
def vocabularySize : Nat := 8
/-- Default number of distinct corpus windows exposed to the trainer. -/
def defaultWindows : Nat := 16

local instance : NeZero vocabularySize := ⟨by decide⟩

/-- Compact byte vocabulary: encode byte id `b` as `b % 8`. Collisions are intentional. -/
def byteBucket (id : Nat) : Fin vocabularySize :=
  Fin.ofNat vocabularySize id
/-- Hidden state width of the recurrent cell. -/
def hiddenWidth : Nat := 4

/-- Shape record shared by the RNN-with-head and LSTM-with-head constructors. -/
abbrev modelConfig : nn.Recurrent.Config :=
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

/-- The cell followed by a time-distributed linear output head. -/
def model : Cell → nn.Builder (nn.Sequential input output)
  | .rnn => nn.Recurrent.rnn modelConfig
  | .lstm => nn.Recurrent.lstm modelConfig

/-- Build a finite next-token dataset from evenly spaced corpus windows. -/
def samples (corpus : String) (windows : Nat) :
    Data.SampleStream (Sample.Supervised Float input output) :=
  Data.CausalLM.byteSamples
    (α := Float) contextLength vocabularySize byteBucket windows corpus

/-- Train the cell's model with the public `Trainer` surface. -/
def train (cell : Cell) (runtime : Session.Config) (data : Corpus.Options)
    (flags : CLI.Training.OptimizerOptions) : IO Unit := do
  let corpus ← Corpus.Options.read cell.exeName data
  let trainer :=
    Trainer.create (model cell) <|
      { runtime with
          optimizer := optim.sgd { lr := flags.lr }
          objective := oneHotCrossEntropy 1 }
  let trainData := Data.fromStream (samples corpus data.windows)
  let trained ← trainer.train
    trainData
    (flags.trainOptions
      (logTitle := cell.logTitle)
      (logNotes := #[s!"corpus={data.corpus}", s!"windows={data.windows}"]))
  trained.printSummary

/-- CLI entrypoint of the cell's text command. -/
def main (cell : Cell) (args : List String) : IO UInt32 := do
  CLI.Training.run
    { exeName := cell.exeName
      defaultLogPath := cell.defaultLogPath
      defaultSteps := 1
      defaultLr := 1e-2
      description := cell.description
      dataOptions := Corpus.Options.help defaultWindows
      parseData := Corpus.Options.parse cell.exeName defaultWindows
      train := train cell }
    args

end TorchLean.Examples.Recurrent
