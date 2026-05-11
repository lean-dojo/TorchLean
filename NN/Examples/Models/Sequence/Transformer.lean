/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team

Device-agnostic real-data example:
  python3 scripts/datasets/download_example_data.py --tiny-shakespeare
  lake exe torchlean transformer --cpu --steps 1
  lake build -R -K cuda=true && lake exe torchlean transformer --cuda --steps 1
-/

module


public import NN
public import NN.API.Models.Transformer
public import NN.Examples.Models.Common.RealData
public import NN.Examples.Models.Sequence.SimpleText

/-!
# Transformer Text Example

Runnable `torchlean transformer` example. It reads a local text corpus, builds a byte-level
sequence reconstruction sample, and trains one transformer encoder block on that real text window.

The reusable model wiring lives in `NN.API.Models.Transformer`
(`nn.models.transformerEncoder`). This file is the runnable wrapper.

```bash
python3 scripts/datasets/download_example_data.py --tiny-shakespeare
lake build -R -K cuda=true && lake exe torchlean transformer --cuda --tiny-shakespeare --steps 1
```
-/

@[expose] public section

open Spec Tensor
open NN.API

namespace NN.Examples.Models.Sequence.Transformer

def exeName : String := "torchlean transformer"
def defaultLogJson : System.FilePath := "data/model_zoo/transformer_trainlog.json"

/-- Number of identical rows in the small batch used by this encoder check. -/
def batch : Nat := 4
/-- Short sequence length: enough to exercise attention without making CPU runs painful. -/
def seqLen : Nat := 8
/-- Transformer feature width. -/
def dModel : Nat := 32
/-- Number of attention heads. -/
def numHeads : Nat := 4
/-- Per-head width; `numHeads * headDim` matches `dModel`. -/
def headDim : Nat := 8
/-- Feed-forward hidden width inside the encoder block. -/
def ffnHidden : Nat := 128

/-- API-level encoder configuration shared by shapes and the constructor. -/
def cfg : nn.models.TransformerEncoderConfig :=
  { batch := batch
    seqLen := seqLen
    dModel := dModel
    numHeads := numHeads
    headDim := headDim
    ffnHidden := ffnHidden }

abbrev σ : Shape :=
  nn.models.transformerEncoderShape cfg

abbrev τ : Shape :=
  σ

def mkModel : nn.M (nn.Sequential σ τ) :=
  nn.models.transformerEncoder cfg

/--
Build one batch by repeating a real-text causal sample.

This is intentionally an encoder-block reconstruction example, not autoregressive generation. The
causal GPT/Mamba files cover language-model decoding; this file keeps the attention block itself
small and easy to sanity-check.
-/
def mkSample {α : Type} [Semantics.Scalar α] [Runtime.Scalar α] (input : String) :
    API.sample.Supervised α σ τ :=
  let row : API.sample.Supervised α (NN.Tensor.Shape.Mat seqLen dModel)
      (NN.Tensor.Shape.Mat seqLen dModel) :=
    RealData.textCausalSample (α := α) seqLen dModel input
  match row with
  | .cons x (.cons y .nil) =>
      API.sample.mk (Tensor.dim (fun _ => x)) (Tensor.dim (fun _ => y))

/--
Shared runner configuration for `torchlean transformer`.

We intentionally reuse the same training infrastructure as `torchlean rnn` and `torchlean lstm`:
the goal here is to compare the *architecture* (attention/norm/FFN) rather than read three copies
of the same CLI/runtime wrapper.
-/
def runner : SimpleText.RunnerConfig σ τ :=
  { exeName := exeName
    defaultLogJson := defaultLogJson
    modelName := "Transformer encoder"
    logTitle := "Transformer text training"
    mkModel := mkModel
    mkSample := fun {α} _ _ input => mkSample (α := α) input
    -- Attention + LayerNorm is more sensitive than the RNN/LSTM checks; keep the default LR small.
    lr := 1e-4 }

def main (args : List String) : IO UInt32 := do
  SimpleText.main runner args

end NN.Examples.Models.Sequence.Transformer
