/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.TorchLean.NN
import Mathlib.Algebra.Order.Algebra

/-!
# TorchLean-executable model family: MLPs

This file defines the compact feed-forward model family using the
`Runtime.Autograd.TorchLean.NN` builder layer:

- `mlp`: two-layer regression/general-purpose MLP,
- `mlpClassifier`: the same hidden block with a class-logit output,
- `softmaxRegression`: a single linear class-logit model.

Keeping these together avoids an overly granular module layout while still separating genuinely
different architecture families (CNN, FNO, ResNet, Transformer) into their own modules.

## Spec vs TorchLean views

TorchLean exposes two related layers:

1. `NN.Spec.Models.*`: proof-friendly specifications, evaluated as functions on `Tensor α s`.

2. `NN.GraphSpec.Models.TorchLean.*`: executable architecture constructors used by runtime
   training/evaluation utilities (with `.eager` / `.compiled` backends).

In `NN.Tests.Runtime.Floats.TorchLeanSpecMlpEquivSmoke` we assert that (for the same
initialized parameters) TorchLean’s forward pass agrees with the Spec forward pass.

For an **opt-in** executable that trains this MLP with `Torch.Options.fastKernels` and
`useGpu` (GEMM path for every `linear`), run the demo executable:

- CPU: `lake exe torchlean mlp --cpu --steps 10`
- CUDA: `lake build -R -K cuda=true && lake exe torchlean mlp --cuda --fast-kernels --steps 10`
-/

@[expose] public section


namespace NN
namespace GraphSpec
namespace Models
namespace TorchLean

open Spec
open Tensor
open NN.Tensor

/-!
## `Linear → ReLU → Linear`

This is the smallest MLP that exercises parameterized layers (`Linear`), a nonlinearity (`ReLU`),
and sequential composition (`Seq`).

Seeds are explicit so initialization stays deterministic and tests can lock in a reference behavior.
-/

/-- 2-layer MLP: `Linear(inDim,hidDim) → ReLU → Linear(hidDim,outDim)`. -/
def mlp
    (inDim hidDim outDim : Nat)
    (seedW1 seedB1 seedW2 seedB2 : Nat := 0) :
    _root_.Runtime.Autograd.TorchLean.NN.Seq
      (NN.Tensor.Shape.Vec inDim) (NN.Tensor.Shape.Vec outDim) :=
  _root_.Runtime.Autograd.TorchLean.NN.seq1
      (_root_.Runtime.Autograd.TorchLean.NN.linear inDim hidDim
        (seedW := seedW1) (seedB := seedB1)) >>>
  _root_.Runtime.Autograd.TorchLean.NN.seq1 _root_.Runtime.Autograd.TorchLean.NN.relu >>>
  _root_.Runtime.Autograd.TorchLean.NN.seq1
      (_root_.Runtime.Autograd.TorchLean.NN.linear hidDim outDim
        (seedW := seedW2) (seedB := seedB2))

/-!
## Classifier variants

These return logits. Loss choice stays outside the constructor, so callers can use cross-entropy,
margin losses, calibration losses, or verification objectives without changing the architecture.
-/

/-- 2-layer MLP classifier: `Linear(inDim,hidDim) → ReLU → Linear(hidDim,numClasses)`. -/
def mlpClassifier
    (inDim hidDim numClasses : Nat)
    (seedW1 seedB1 seedW2 seedB2 : Nat := 0) :
    _root_.Runtime.Autograd.TorchLean.NN.Seq
      (NN.Tensor.Shape.Vec inDim) (NN.Tensor.Shape.Vec numClasses) :=
  mlp inDim hidDim numClasses
    (seedW1 := seedW1) (seedB1 := seedB1) (seedW2 := seedW2) (seedB2 := seedB2)

/-- Multiclass logistic regression: a single linear layer producing logits. -/
def softmaxRegression
    (inDim numClasses : Nat)
    (seedW seedB : Nat := 0) :
    _root_.Runtime.Autograd.TorchLean.NN.Seq
      (NN.Tensor.Shape.Vec inDim) (NN.Tensor.Shape.Vec numClasses) :=
  _root_.Runtime.Autograd.TorchLean.NN.seq1
    (_root_.Runtime.Autograd.TorchLean.NN.linear inDim numClasses (seedW := seedW) (seedB := seedB))

end TorchLean
end Models
end GraphSpec
end NN
