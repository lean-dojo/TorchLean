/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Model.Functional.Core
public import NN.Runtime.Autograd.Model.Functional.ShapeOps

/-!
# Loss

TorchLean loss helpers in the style of `torch.nn.functional`.

These helpers keep training loops close to the familiar `torch.nn.functional` style:

```
loss = Loss.mse yhat y
loss = Loss.mse yhat y (reduction := .sum)
```

They are execution-mode generic: eager tape and typed SSA/DAG both work.

### PyTorch references

- `torch.nn.functional` (losses overview): https://pytorch.org/docs/stable/nn.functional.html
- `mse_loss`: https://pytorch.org/docs/stable/generated/torch.nn.functional.mse_loss.html
- `cross_entropy`: https://pytorch.org/docs/stable/generated/torch.nn.functional.cross_entropy.html
- `nll_loss`: https://pytorch.org/docs/stable/generated/torch.nn.functional.nll_loss.html
- `binary_cross_entropy_with_logits`:
https://pytorch.org/docs/stable/generated/torch.nn.functional.binary_cross_entropy_with_logits.html
-/

@[expose] public section


namespace TorchLean

open Spec TorchLean
open TorchLean TorchLean.Tensor
open Runtime.Autograd.Torch
open Runtime.Autograd.Model

namespace Loss

/--
Reduction mode for losses that start as elementwise tensors.

PyTorch analogy: `reduction="mean"` or `reduction="sum"`.
-/
inductive Reduction where
  | mean
  | sum
  deriving Repr, DecidableEq

/--
Reduce an elementwise loss tensor to a scalar according to `reduction`.

This is the common final step for losses like MSE and cross-entropy.
-/
def reduceLoss {α : Type} [TorchLean.Storage α] [Context α]
    {m : Type → Type} [Monad m] [Ops (m := m) (α := α)]
    {s : Shape} (x : RefTy (m := m) (α := α) s) (reduction : Reduction) :
    m (RefTy (m := m) (α := α) Shape.scalar) := by
  cases reduction with
  | mean => exact F.mean (m := m) (α := α) (s := s) x
  | sum => exact sum (m := m) (α := α) (s := s) x

/--
Mean squared error (MSE) loss between predictions and targets.

This is backend-generic and supports both `mean` and `sum` reduction.
-/
def mse {α : Type} [TorchLean.Storage α] [Context α]
    {m : Type → Type} [Monad m] [Ops (m := m) (α := α)]
    {s : Shape}
    (yhat y : RefTy (m := m) (α := α) s)
    (reduction : Reduction := .mean) :
    m (RefTy (m := m) (α := α) Shape.scalar) := by
  cases reduction with
  | mean =>
      exact mseLoss (m := m) (α := α) (s := s) yhat y
  | sum =>
      exact (do
        let diff ← sub (m := m) (α := α) (s := s) yhat y
        let sq ← F.square (m := m) (α := α) (s := s) diff
        sum (m := m) (α := α) (s := s) sq
      )

/--
Weighted mean-squared error.

This returns `sum (weights * (prediction - target)^2)` without implicit normalization. Weights
whose sum is one therefore define a weighted mean, and zero weights exclude coordinates without
requiring a separate masking operation.
-/
def mseWeighted {α : Type} [TorchLean.Storage α] [Context α]
    {m : Type → Type} [Monad m] [Ops (m := m) (α := α)]
    {s : Shape}
    (yhat y weights : RefTy (m := m) (α := α) s) :
    m (RefTy (m := m) (α := α) Shape.scalar) := do
  let diff ← sub (m := m) (α := α) (s := s) yhat y
  let squared ← F.square (m := m) (α := α) (s := s) diff
  let weighted ← mul (m := m) (α := α) (s := s) squared weights
  sum (m := m) (α := α) (s := s) weighted

/-- Negative log-likelihood (one-hot targets), assuming inputs are log-probabilities. -/
def nllOneHot {α : Type} [TorchLean.Storage α] [Context α]
    {m : Type → Type} [Monad m] [Ops (m := m) (α := α)]
    {s : Shape} (axis : Nat) [Shape.AxisInBounds axis s]
    (logProbs targetOneHot : RefTy (m := m) (α := α) s)
    (reduction : Reduction := .mean) :
    m (RefTy (m := m) (α := α) Shape.scalar) := do
  let prod ← mul (m := m) (α := α) (s := s) targetOneHot logProbs
  let negProd ← scale (m := m) (α := α) (s := s) prod (-1)
  match reduction with
  | .sum =>
      -- `sum` is already the correct reduction: `∑_{prefix,cls} -y * logp = ∑_{prefix} -logp_true`.
      reduceLoss (m := m) (α := α) (s := s) negProd .sum
  | .mean =>
      -- Mean over samples, not classes: undo the class-dimension factor introduced by averaging
      -- every tensor entry.
      let avgAll ← reduceLoss (m := m) (α := α) (s := s) negProd .mean
      scale (m := m) (α := α) (s := Shape.scalar) avgAll (Shape.axisSize s axis)

/--
Cross-entropy (one-hot targets), computed as `-sum(y * log(softmax(logits)))`.

Note: this is one-hot only. For integer class labels, use `crossEntropy`, whose
`Fin classes` target tensor records the class bound in its element type.
-/
def oneHotCrossEntropy {α : Type} [TorchLean.Storage α] [Context α]
    {m : Type → Type} [Monad m] [Ops (m := m) (α := α)]
    {s : Shape} (axis : Nat) [Shape.AxisInBounds axis s]
    (logits targetOneHot : RefTy (m := m) (α := α) s)
    (reduction : Reduction := .mean) :
    m (RefTy (m := m) (α := α) Shape.scalar) := do
  let logp ← F.logSoftmax (m := m) (α := α) (s := s) axis logits
  nllOneHot (m := m) (α := α) (s := s) axis logp targetOneHot (reduction := reduction)

/--
Unreduced indexed negative log-likelihood along an arbitrary class axis.

The shape contract is:

* `logProbs` has shape `prefix ++ [classes] ++ suffix`;
* `classAxis = rank prefix` identifies the `classes` dimension;
* `target` has shape `prefix ++ suffix` and element type `Fin classes`;
* the result has the same shape as `target`, with the class axis removed.

Each output coordinate is `-logProbs[..., target[...], ...]`. In particular, suffix coordinates
remain aligned with their corresponding labels. Invalid class labels are unrepresentable.
-/
def nllUnreduced {α : Type} [TorchLean.Storage α] [Context α]
    {m : Type → Type} [Monad m] [Ops (m := m) (α := α)]
    {leading trailing : Shape} {classes : Nat}
    (classAxis : Nat) (_hClassAxis : classAxis = leading.rank)
    (logProbs : RefTy (m := m) (α := α) (leading.concat (trailing.prependDim classes)))
    (target : Runtime.Autograd.Torch.DataRef (m := m) (α := α) (Fin classes)
      (leading.concat trailing)) :
    m (RefTy (m := m) (α := α) (leading.concat trailing)) := do
  let toIndices := fun (target : Tensor (Fin classes) (leading.concat trailing)) =>
    (Tensor.ofFn fun i =>
      let targetClass := target.getFlat ⟨i.val, by
        rw [Spec.Shape.internalSize_eq]
        exact i.isLt⟩
      let inner := Shape.size trailing
      let outer := i.val / inner
      let flatIndex := outer * (classes * inner) + targetClass.val * inner + i.val % inner
      ⟨flatIndex, by
        have hInner : 0 < inner := by
          by_contra h
          have hZero : Shape.size trailing = 0 := by
            simpa [inner] using Nat.eq_zero_of_not_pos h
          simpa [Shape.size_concat, hZero] using i.isLt
        have hOuter : outer < Shape.size leading := by
          apply Nat.div_lt_of_lt_mul
          simpa [Shape.size_concat, Nat.mul_comm] using i.isLt
        have hRemainder : i.val % inner < inner := Nat.mod_lt _ hInner
        have hClassInner : targetClass.val * inner + i.val % inner < classes * inner := by
          calc
            targetClass.val * inner + i.val % inner < targetClass.val * inner + inner :=
              Nat.add_lt_add_left hRemainder _
            _ = (targetClass.val + 1) * inner := by simp [Nat.add_mul]
            _ ≤ classes * inner :=
              Nat.mul_le_mul_right inner (Nat.succ_le_iff.mpr targetClass.isLt)
        calc
          flatIndex < outer * (classes * inner) + classes * inner := by
            simpa [flatIndex, Nat.add_assoc] using
              Nat.add_lt_add_left hClassInner (outer * (classes * inner))
          _ = (outer + 1) * (classes * inner) := by simp [Nat.add_mul]
          _ ≤ Shape.size leading * (classes * inner) :=
            Nat.mul_le_mul_right _ (Nat.succ_le_iff.mpr hOuter)
          _ = Shape.size (leading.concat (trailing.prependDim classes)) := by
            simp [inner, Shape.size_concat, Shape.size]⟩ :
      Tensor (Fin (Shape.size (leading.concat (trailing.prependDim classes))))
        [Shape.size (leading.concat trailing)])
  let flat ← reshape (m := m) (α := α)
    (s₁ := leading.concat (trailing.prependDim classes))
    (s₂ := [Shape.size (leading.concat (trailing.prependDim classes))]) logProbs
      (by simp [Shape.size])
  let flatTarget := Runtime.Autograd.Torch.mapData (m := m) (α := α)
    toIndices target
  let picked ← indexSelect (m := m) (α := α)
    (s := [Shape.size (leading.concat (trailing.prependDim classes))]) 0
    (Shape.size (leading.concat trailing)) flat flatTarget
  let losses ← scale (m := m) (α := α)
    (s := [Shape.size (leading.concat trailing)]) picked (-1)
  reshape (m := m) (α := α) (s₁ := [Shape.size (leading.concat trailing)])
    (s₂ := leading.concat trailing) losses (by simp [Shape.size])

/--
Indexed negative log-likelihood along an arbitrary class axis.

The logits, labels, and class-axis contract are those of `nllUnreduced`. The requested reduction is
applied over every coordinate of the class-erased output shape.
-/
def nll {α : Type} [TorchLean.Storage α] [Context α]
    {m : Type → Type} [Monad m] [Ops (m := m) (α := α)]
    {leading trailing : Shape} {classes : Nat}
    (classAxis : Nat) (hClassAxis : classAxis = leading.rank)
    (logProbs : RefTy (m := m) (α := α) (leading.concat (trailing.prependDim classes)))
    (target : Runtime.Autograd.Torch.DataRef (m := m) (α := α) (Fin classes)
      (leading.concat trailing))
    (reduction : Reduction := .mean) :
    m (RefTy (m := m) (α := α) Shape.scalar) := do
  let losses ← nllUnreduced (m := m) (α := α) classAxis hClassAxis logProbs target
  reduceLoss (m := m) (α := α) (s := leading.concat trailing) losses reduction

/--
Weighted indexed negative log-likelihood along an arbitrary class axis.

The logits, labels, weights, and output coordinates obey the `nllUnreduced` shape contract. This
returns `sum (weights * losses)` without implicit normalization; normalized weights therefore give
a weighted mean, while zero weights mask coordinates without a division-by-zero convention.
-/
def nllWeighted {α : Type} [TorchLean.Storage α] [Context α]
    {m : Type → Type} [Monad m] [Ops (m := m) (α := α)]
    {leading trailing : Shape} {classes : Nat}
    (classAxis : Nat) (hClassAxis : classAxis = leading.rank)
    (logProbs : RefTy (m := m) (α := α) (leading.concat (trailing.prependDim classes)))
    (target : Runtime.Autograd.Torch.DataRef (m := m) (α := α) (Fin classes)
      (leading.concat trailing))
    (weights : RefTy (m := m) (α := α) (leading.concat trailing)) :
    m (RefTy (m := m) (α := α) Shape.scalar) := do
  let losses ← nllUnreduced (m := m) (α := α) classAxis hClassAxis logProbs target
  let weighted ← mul (m := m) (α := α) (s := leading.concat trailing) losses weights
  sum (m := m) (α := α) (s := leading.concat trailing) weighted

/--
Indexed cross-entropy along an arbitrary class axis.

This applies log-softmax along `classAxis` and then uses `nll`. Shapes follow the
`prefix ++ [classes] ++ suffix` contract documented on `nllUnreduced`.
-/
def crossEntropy {α : Type} [TorchLean.Storage α] [Context α]
    {m : Type → Type} [Monad m] [Ops (m := m) (α := α)]
    {leading trailing : Shape} {classes : Nat}
    (classAxis : Nat) (hClassAxis : classAxis = leading.rank)
    (logits : RefTy (m := m) (α := α) (leading.concat (trailing.prependDim classes)))
    (target : Runtime.Autograd.Torch.DataRef (m := m) (α := α) (Fin classes)
      (leading.concat trailing))
    (reduction : Reduction := .mean) :
    m (RefTy (m := m) (α := α) Shape.scalar) :=
  letI : Shape.AxisInBounds classAxis (leading.concat (trailing.prependDim classes)) :=
    Shape.AxisInBounds.mk <| by
      rw [hClassAxis]
      simp [Tensor.LinearAlgebra.Internal.rank_concat, Shape.rank]
  do
  let logp ← F.logSoftmax (m := m) (α := α)
    (s := leading.concat (trailing.prependDim classes)) classAxis logits
  nll (m := m) (α := α) classAxis hClassAxis logp target (reduction := reduction)

/--
Weighted indexed cross-entropy along an arbitrary class axis.

This applies log-softmax along `classAxis`, multiplies each class-erased loss by the corresponding
weight, and sums without implicit normalization. Shapes follow the contract on `nllUnreduced`.
-/
def crossEntropyWeighted {α : Type} [TorchLean.Storage α] [Context α]
    {m : Type → Type} [Monad m] [Ops (m := m) (α := α)]
    {leading trailing : Shape} {classes : Nat}
    (classAxis : Nat) (hClassAxis : classAxis = leading.rank)
    (logits : RefTy (m := m) (α := α) (leading.concat (trailing.prependDim classes)))
    (target : Runtime.Autograd.Torch.DataRef (m := m) (α := α) (Fin classes)
      (leading.concat trailing))
    (weights : RefTy (m := m) (α := α) (leading.concat trailing)) :
    m (RefTy (m := m) (α := α) Shape.scalar) :=
  letI : Shape.AxisInBounds classAxis (leading.concat (trailing.prependDim classes)) :=
    Shape.AxisInBounds.mk <| by
      rw [hClassAxis]
      simp [Tensor.LinearAlgebra.Internal.rank_concat, Shape.rank]
  do
  let logp ← F.logSoftmax (m := m) (α := α)
    (s := leading.concat (trailing.prependDim classes)) classAxis logits
  nllWeighted (m := m) (α := α) classAxis hClassAxis logp target weights

/--
Binary cross-entropy with logits (elementwise), using the stable identity:
`BCEWithLogits(x,y) = y * softplus(-x) + (1-y) * softplus(x)`.

Targets are expected in `[0,1]` (typically 0/1), same shape as `logits`.
-/
def bceWithLogits {α : Type} [TorchLean.Storage α] [Context α]
    {m : Type → Type} [Monad m] [Ops (m := m) (α := α)]
    {s : Shape}
    (logits target : RefTy (m := m) (α := α) s)
    (reduction : Reduction := .mean) :
    m (RefTy (m := m) (α := α) Shape.scalar) := do
  let onesT : Tensor α s := Tensor.full s (1 : α)
  let ones ← const (m := m) (α := α) (s := s) onesT
  let oneMinusY ← sub (m := m) (α := α) (s := s) ones target
  let negLogits ← scale (m := m) (α := α) (s := s) logits (-1)
  let spNeg ← softplus (m := m) (α := α) (s := s) negLogits
  let spPos ← softplus (m := m) (α := α) (s := s) logits
  let t1 ← mul (m := m) (α := α) (s := s) target spNeg
  let t2 ← mul (m := m) (α := α) (s := s) oneMinusY spPos
  let lossVec ← add (m := m) (α := α) (s := s) t1 t2
  reduceLoss (m := m) (α := α) (s := s) lossVec reduction

end Loss

end TorchLean
