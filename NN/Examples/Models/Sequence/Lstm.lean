/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team

Device-agnostic example:
  lake exe torchlean lstm --cpu
  lake build -R -K cuda=true && lake exe torchlean lstm --cuda

This is a real-data sequence run:
- reads a local text corpus (default: `data/real/text/tiny_shakespeare.txt`),
- builds a byte-level causal-LM one-hot window,
- trains `nn.lstm` plus a time-distributed linear head for one or more steps.
-/

module

public import NN
public import NN.API.Models.SimpleSeq
public import NN.Examples.Models.Sequence.SimpleText

/-!
# LSTM Text Example

Runnable `torchlean lstm` example. It reads a local text corpus, creates a byte-level
causal-language-model window, and trains an LSTM plus time-distributed linear head.

The model constructor lives in `NN.API.Models.SimpleSeq` so other examples can reuse it. This file
keeps only the architecture-specific declarations; the shared corpus loading, CLI parsing, logging,
and train loop live in `NN.Examples.Models.Sequence.SimpleText`.

## What This Example Is (And Is Not)

This is a **small layer smoke test** for the LSTM cell plus the TorchLean training loop. It uses a
single fixed text window and a simple MSE-on-one-hot objective to keep runs short and predictable.

If you want a real language-model tutorial (proper autoregressive loss + longer context + sampling),
use one of:
- `torchlean chargpt` (Karpathy-style, single-file char-level GPT),
- `torchlean gpt2` (byte-level GPT-2-style model + save/reload),
- `torchlean text_gpt2` (CUDA corpus trainer).

```bash
python3 scripts/datasets/download_example_data.py --tiny-shakespeare
lake build -R -K cuda=true && lake exe torchlean lstm --cuda --tiny-shakespeare --steps 1
```
-/

@[expose] public section

open Spec Tensor
open NN.API

namespace NN.Examples.Models.Sequence.Lstm

def exeName : String := "torchlean lstm"
def defaultLogJson : System.FilePath := "data/model_zoo/lstm_trainlog.json"

/-- Short byte-window length used for a quick gated-recurrent smoke test. -/
def seqLen : Nat := 8
/--
Byte vocabulary size.

This example uses byte-level tokens (`0..255`) rather than hashing bytes down to a smaller bucket
count. Earlier smoke tests used `32` here for speed, but the full byte vocab avoids unnecessary
aliasing and makes the tutorial behavior easier to reason about.
-/
def inputSize : Nat := 256
/-- Hidden state width of the LSTM cell. -/
def hiddenSize : Nat := 64

/-- Shared shape/config record consumed by the reusable API constructor. -/
def cfg : nn.models.SeqRnnHeadConfig :=
  { seqLen := seqLen, inputSize := inputSize, hiddenSize := hiddenSize }

abbrev σ : Shape :=
  nn.models.seqRnnHeadInShape cfg

abbrev τ : Shape :=
  nn.models.seqRnnHeadOutShape cfg

def mkModel : nn.M (nn.Sequential σ τ) :=
  nn.models.lstmWithLinearHead cfg

/-- Convert corpus text into one supervised causal sequence window. -/
def mkSample {α : Type} [Semantics.Scalar α] [Runtime.Scalar α] (input : String) :
    API.sample.Supervised α σ τ :=
  RealData.textCausalSample (α := α) seqLen inputSize input

/-- Shared runner configuration for `torchlean lstm`. -/
def runner : SimpleText.RunnerConfig σ τ :=
  { exeName := exeName
    defaultLogJson := defaultLogJson
    modelName := "LSTM"
    logTitle := "LSTM text training"
    mkModel := mkModel
    mkSample := fun {α} _ _ input => mkSample (α := α) input
    lr := 1e-2 }

def main (args : List String) : IO UInt32 := do
  SimpleText.main runner args

end NN.Examples.Models.Sequence.Lstm
