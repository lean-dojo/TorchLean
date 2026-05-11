/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Public

/-!
# Simple Sequence Model Helpers (API)

Small, reusable constructors for “sequence core + time-distributed linear head” models used by the
`rnn` and `lstm` runnable examples.

These helpers live in the API layer so examples can stay focused on:
corpus loading, sample construction, and training loops.
-/

@[expose] public section

namespace NN
namespace API

open Spec Tensor

namespace nn
namespace models

/--
Configuration for an RNN/LSTM sequence model with a time-distributed linear head.

Shapes follow the convention used by the runnable examples:
- input: `(seqLen × inputSize)`
- output logits/targets: `(seqLen × inputSize)`
-/
structure SeqRnnHeadConfig where
  seqLen : Nat
  inputSize : Nat
  hiddenSize : Nat
deriving Repr

/-- Input shape `(seqLen × inputSize)` for `SeqRnnHeadConfig`. -/
abbrev seqRnnHeadInShape (cfg : SeqRnnHeadConfig) : Shape :=
  NN.Tensor.Shape.Mat cfg.seqLen cfg.inputSize

/-- Output shape `(seqLen × inputSize)` for `SeqRnnHeadConfig`. -/
abbrev seqRnnHeadOutShape (cfg : SeqRnnHeadConfig) : Shape :=
  NN.Tensor.Shape.Mat cfg.seqLen cfg.inputSize

/--
Vanilla RNN core plus time-distributed linear head:

`rnn(seqLen,inputSize,hiddenSize) → linear(hiddenSize → inputSize)`.
-/
def rnnWithLinearHead (cfg : SeqRnnHeadConfig) :
    nn.M (nn.Sequential (seqRnnHeadInShape cfg) (seqRnnHeadOutShape cfg)) :=
  nn.sequential![
    nn.rnn cfg.seqLen cfg.inputSize cfg.hiddenSize,
    nn.linear cfg.hiddenSize cfg.inputSize (pfx := NN.Tensor.Shape.Vec cfg.seqLen)
  ]

/--
LSTM core plus time-distributed linear head:

`lstm(seqLen,inputSize,hiddenSize) → linear(hiddenSize → inputSize)`.
-/
def lstmWithLinearHead (cfg : SeqRnnHeadConfig) :
    nn.M (nn.Sequential (seqRnnHeadInShape cfg) (seqRnnHeadOutShape cfg)) :=
  nn.sequential![
    nn.lstm cfg.seqLen cfg.inputSize cfg.hiddenSize,
    nn.linear cfg.hiddenSize cfg.inputSize (pfx := NN.Tensor.Shape.Vec cfg.seqLen)
  ]

end models
end nn

end API
end NN

