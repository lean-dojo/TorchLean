/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team

Real-data CUDA example:
  python3 scripts/datasets/download_example_data.py --tiny-shakespeare
  lake -R -K cuda=true exe torchlean transformer --device cuda --tiny-shakespeare --steps 1
-/

module

public import NN.Examples.Models.Common.RealData

/-!
# Causal Transformer Text Example

Runnable `torchlean transformer` example. It reads a local text corpus, builds a shifted next-byte
feature sample, and trains a compact causal Transformer on that real text window.

This command uses the same public causal model family as the GPT examples. Future tokens are masked,
so a position cannot read the byte it is being trained to predict.

```bash
python3 scripts/datasets/download_example_data.py --tiny-shakespeare
lake -R -K cuda=true exe torchlean transformer --device cuda --tiny-shakespeare --steps 1
```
-/

@[expose] public section

open TorchLean

namespace NN.Examples.Models.Sequence.Transformer

/-- CLI subcommand name used in terminal banners and error messages. -/
def exeName : String := "transformer"

/-- Default JSON loss-curve path for this command. -/
def defaultLogPath : System.FilePath := Support.trainLogPath "transformer"

/-- Short multi-token window for the quick encoder training run. -/
def contextLength : Nat := 4
/-- Transformer feature width. -/
def modelWidth : Nat := 4
/-- Default number of distinct corpus windows exposed to the trainer. -/
def defaultWindows : Nat := 16

local instance : NeZero modelWidth := ⟨by decide⟩

/-- Compact byte vocabulary: encode byte id `b` as `b % 4`; collisions are intentional. -/
def byteBucket (id : Nat) : Fin modelWidth :=
  Fin.ofNat modelWidth id
/-- Number of attention heads. -/
def attentionHeads : Nat := 2
/-- Per-head width; `attentionHeads * attentionHeadWidth = modelWidth`. -/
def attentionHeadWidth : Nat := 2
/-- Feed-forward hidden width inside the encoder block. -/
def feedForwardWidth : Nat := 8

/-- Causal language-model configuration. -/
abbrev modelConfig : nn.models.CausalTransformer.Config :=
  { sequenceLength := contextLength
    vocabularySize := modelWidth
    headCount := attentionHeads
    headWidth := attentionHeadWidth
    feedForwardWidth := feedForwardWidth
    layerCount := 1 }

/-- Input shape: batched one-hot byte buckets. -/
abbrev input : Shape :=
  modelConfig.vocabularyShape

/-- Output shape: one next-byte logit row per token position. -/
abbrev output : Shape :=
  modelConfig.vocabularyShape

/-- Compact causal Transformer used by the runnable text example. -/
def model : nn.Builder (nn.Sequential input output) :=
  nn.models.CausalTransformer.oneHot modelConfig

/-- Build a finite next-byte dataset from evenly spaced corpus windows. -/
def samples (corpus : String) (windows : Nat) :
    Data.SampleStream (Sample.Supervised Float input output) :=
  Data.CausalLM.byteSamples
    (α := Float) contextLength modelWidth byteBucket windows corpus

/-- Train the causal Transformer with the public `Trainer` surface. -/
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
      (logTitle := "Transformer next-byte training")
      (logNotes := #[s!"corpus={data.corpus.path}", s!"windows={data.windows}"]))
  trained.printSummary

/-- CLI entrypoint for the causal Transformer text command. -/
def main (args : List String) : IO UInt32 := do
  CLI.Training.Command.run
    { exeName := exeName
      defaultLogPath := defaultLogPath
      defaultSteps := 1
      defaultLearningRate := 1e-4
      description := "Causal Transformer next-byte model"
      dataOptions := RealData.TextWindowFlags.help defaultWindows
      parseData := RealData.TextWindowFlags.parse exeName defaultWindows
      train := train }
    args

end NN.Examples.Models.Sequence.Transformer
