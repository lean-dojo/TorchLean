/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLeanExamples.Models.RealData
public import TorchLean.NN.Data.CausalLM
public import TorchLean.NN.Models.CausalTransformer
public import TorchLeanExamples.Models.Training

/-!
# Causal Transformer Text Example

The `torchlean transformer` command reads a local text corpus, cuts it into evenly spaced
next-byte windows (`--windows`, 16 by default) and trains a small causal Transformer on them. Each
byte enters as its bucket `b % 4`, and the model belongs to the `nn.CausalTransformer` family of
the GPT examples. Future positions are masked, so a position cannot read the byte it is trained
to predict.

```bash
python3 scripts/datasets/download_example_data.py --tiny-shakespeare
scripts/lake.sh -Kcuda=true exe torchlean transformer --device cuda --tiny-shakespeare --steps 1
```
-/

@[expose] public section

open TorchLean.Trainer.Objective

namespace TorchLean.Examples.Transformer

/-- CLI subcommand name used in terminal banners and error messages. -/
def exeName : String := "transformer"

/-- Default JSON loss-curve path for this command. -/
def defaultLogPath : System.FilePath := Training.artifactPath "transformer"

/-- Window length in bytes for next-byte training. -/
def contextLength : Nat := 4
/-- Number of byte buckets and the Transformer width (`attentionHeads * attentionHeadWidth`). -/
def modelWidth : Nat := 4
/-- Default `--windows`: the number of evenly spaced corpus windows in the training set. -/
def defaultWindows : Nat := 16

local instance : NeZero modelWidth := ⟨by decide⟩

/-- Byte id `b` as bucket `b % 4`. Distinct bytes share buckets on purpose, to keep four classes. -/
def byteBucket (id : Nat) : Fin modelWidth :=
  Fin.ofNat modelWidth id
/-- Number of attention heads. -/
def attentionHeads : Nat := 2
/-- Per-head width, chosen so that `attentionHeads * attentionHeadWidth = modelWidth`. -/
def attentionHeadWidth : Nat := 2
/-- Feed-forward hidden width inside the encoder block. -/
def feedForwardWidth : Nat := 8

/-- Causal language-model configuration with one Transformer block. -/
abbrev modelConfig : nn.CausalTransformer.Config :=
  { sequenceLength := contextLength
    vocabularySize := modelWidth
    headCount := attentionHeads
    headWidth := attentionHeadWidth
    feedForwardWidth := feedForwardWidth
    layerCount := 1 }

/-- Input shape `[4, 4]`: one window of one-hot byte buckets. -/
abbrev input : Shape :=
  modelConfig.vocabularyShape

/-- Output shape `[4, 4]`: next-bucket logits at each position. -/
abbrev output : Shape :=
  modelConfig.vocabularyShape

/-- Compact causal Transformer used by the runnable text example. -/
def model : nn.Builder (nn.Sequential input output) :=
  nn.CausalTransformer.oneHot modelConfig

/-- Build a finite next-byte dataset from evenly spaced corpus windows. -/
def samples (corpus : String) (windows : Nat) :
    Data.SampleStream (Sample.Supervised Float input output) :=
  Data.CausalLM.byteSamples
    (α := Float) contextLength modelWidth byteBucket windows corpus

/-- Train the causal Transformer with SGD on cross-entropy over the bucket axis. -/
def train (runtime : Session.Config) (data : Corpus.Options)
    (flags : CLI.Training.OptimizerOptions) : IO Unit := do
  let corpus ← Corpus.Options.read exeName data
  let trainer :=
    Trainer.create model <|
      { runtime with
          optimizer := optim.sgd { lr := flags.lr }
          objective := oneHotCrossEntropy 1 }
  let trainData := Data.fromStream (samples corpus data.windows)
  let trained ← trainer.train
    trainData
    (flags.trainOptions
      (logTitle := "Transformer next-byte training")
      (logNotes := #[s!"corpus={data.corpus}", s!"windows={data.windows}"]))
  trained.printSummary

/-- CLI entry point, defaulting to one step at learning rate `1e-4`. -/
def main (args : List String) : IO UInt32 := do
  CLI.Training.run
    { exeName := exeName
      defaultLogPath := defaultLogPath
      defaultSteps := 1
      defaultLr := 1e-4
      description := "Causal Transformer next-byte model"
      dataOptions := Corpus.Options.help defaultWindows
      parseData := Corpus.Options.parse exeName defaultWindows
      train := train }
    args

end TorchLean.Examples.Transformer
