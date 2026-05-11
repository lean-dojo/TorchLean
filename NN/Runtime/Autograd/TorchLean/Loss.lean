/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.TorchLean.Functional

import Mathlib.Algebra.Order.Algebra

/-!
# Loss

TorchLean loss helpers in the style of `torch.nn.functional`.

These are small conveniences for writing readable training loops:

```
loss = Loss.mse yhat y
loss = Loss.mse yhat y (reduction := .sum)
```

They are backend-generic: eager tape and compiled SSA/DAG both work.

### PyTorch references

- `torch.nn.functional` (losses overview): https://pytorch.org/docs/stable/nn.functional.html
- `mse_loss`: https://pytorch.org/docs/stable/generated/torch.nn.functional.mse_loss.html
- `cross_entropy`: https://pytorch.org/docs/stable/generated/torch.nn.functional.cross_entropy.html
- `nll_loss`: https://pytorch.org/docs/stable/generated/torch.nn.functional.nll_loss.html
- `binary_cross_entropy_with_logits`:
  https://pytorch.org/docs/stable/generated/torch.nn.functional.binary_cross_entropy_with_logits.html
-/

@[expose] public section


namespace Runtime
namespace Autograd
namespace TorchLean

open Spec
open Tensor

namespace Loss

/-- Size of the innermost dimension (or `1` for scalar). -/
def lastDim : Shape → Nat
  | .scalar => 1
  | .dim n .scalar => n
  | .dim _ rest => lastDim rest

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
  def reduce {α : Type} [Context α] [DecidableEq Shape]
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
def mse {α : Type} [Context α] [DecidableEq Shape]
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

/-- Negative log-likelihood (one-hot targets), assuming inputs are log-probabilities. -/
def nllOneHot {α : Type} [Context α] [DecidableEq Shape]
    {m : Type → Type} [Monad m] [Ops (m := m) (α := α)]
    {s : Shape}
    (logProbs targetOneHot : RefTy (m := m) (α := α) s)
    (reduction : Reduction := .mean) :
    m (RefTy (m := m) (α := α) Shape.scalar) := do
  let prod ← mul (m := m) (α := α) (s := s) targetOneHot logProbs
  let negProd ← scale (m := m) (α := α) (s := s) prod (-1)
  match reduction with
  | .sum =>
      -- `sum` is already the correct reduction: `∑_{prefix,cls} -y * logp = ∑_{prefix} -logp_true`.
      reduce (m := m) (α := α) (s := s) negProd .sum
  | .mean =>
      -- For one-hot (or probability-simplex) targets, the correct mean is over the prefix dims only.
      -- `mean(negProd)` averages over *all* elements, including the class dimension, so we undo that
      -- extra `1 / nClasses` factor by multiplying by the innermost dimension size.
      let avgAll ← reduce (m := m) (α := α) (s := s) negProd .mean
      scale (m := m) (α := α) (s := Shape.scalar) avgAll (lastDim s)

/--
Cross-entropy (one-hot targets), computed as `-sum(y * log(softmax(logits)))`.

Note: this is one-hot only. If your label is an integer class index, use `crossEntropyIndex`
(`Fin n`) or `crossEntropyNat` (`Nat`).
-/
def crossEntropyOneHot {α : Type} [Context α] [DecidableEq Shape]
    {m : Type → Type} [Monad m] [Ops (m := m) (α := α)]
    {s : Shape}
    (logits targetOneHot : RefTy (m := m) (α := α) s)
    (reduction : Reduction := .mean) (ε : α := Numbers.epsilon) :
    m (RefTy (m := m) (α := α) Shape.scalar) := do
  let logp ← logSoftmax (m := m) (α := α) (s := s) logits (ε := ε)
  nllOneHot (m := m) (α := α) (s := s) logp targetOneHot (reduction := reduction)

/-- Negative log-likelihood for a single class index (vector logits/log-probs). -/
def nllIndex {α : Type} [Context α] [DecidableEq Shape]
    {m : Type → Type} [Monad m] [Ops (m := m) (α := α)]
    {n : Nat}
    (logProbs : RefTy (m := m) (α := α) (.dim n .scalar))
    (target : Fin n) :
    m (RefTy (m := m) (α := α) Shape.scalar) := do
  let picked ← gatherScalar (m := m) (α := α) (n := n) logProbs target
  scale (m := m) (α := α) (s := Shape.scalar) picked (-1)

/--
Negative log-likelihood for a single class index.

Unlike `nllIndex`, the target is a `Nat` (useful when labels come from data).
Out-of-bounds indices contribute `0` (so this stays total).
-/
def nllNat {α : Type} [Context α] [DecidableEq Shape]
    {m : Type → Type} [Monad m] [Ops (m := m) (α := α)]
    {n : Nat}
    (logProbs : RefTy (m := m) (α := α) (.dim n .scalar))
    (target : Nat) :
    m (RefTy (m := m) (α := α) Shape.scalar) := do
  let picked ← gatherScalarNat (m := m) (α := α) (n := n) logProbs target
  scale (m := m) (α := α) (s := Shape.scalar) picked (-1)

/--
Cross-entropy for a single class index, computed as `-log(softmax(logits)[target])`.

This avoids one-hot targets, but the target index is a Lean `Fin n` (not a tensor value).
-/
def crossEntropyIndex {α : Type} [Context α] [DecidableEq Shape]
    {m : Type → Type} [Monad m] [Ops (m := m) (α := α)]
    {n : Nat}
    (logits : RefTy (m := m) (α := α) (.dim n .scalar))
    (target : Fin n)
    (ε : α := Numbers.epsilon) :
    m (RefTy (m := m) (α := α) Shape.scalar) := do
  let logp ← logSoftmax (m := m) (α := α) (s := .dim n .scalar) logits (ε := ε)
  nllIndex (m := m) (α := α) (n := n) logp target

/--
Cross-entropy for a single class index, with a `Nat` target (useful for labels from data).

Out-of-bounds indices contribute `0` (so this stays total).
-/
def crossEntropyNat {α : Type} [Context α] [DecidableEq Shape]
    {m : Type → Type} [Monad m] [Ops (m := m) (α := α)]
    {n : Nat}
    (logits : RefTy (m := m) (α := α) (.dim n .scalar))
    (target : Nat)
    (ε : α := Numbers.epsilon) :
    m (RefTy (m := m) (α := α) Shape.scalar) := do
  let logp ← logSoftmax (m := m) (α := α) (s := .dim n .scalar) logits (ε := ε)
  nllNat (m := m) (α := α) (n := n) logp target

/--
Binary cross-entropy with logits (elementwise), using the stable identity:
`BCEWithLogits(x,y) = y * softplus(-x) + (1-y) * softplus(x)`.

Targets are expected in `[0,1]` (typically 0/1), same shape as `logits`.
-/
def bceWithLogits {α : Type} [Context α] [DecidableEq Shape]
    {m : Type → Type} [Monad m] [Ops (m := m) (α := α)]
    {s : Shape}
    (logits target : RefTy (m := m) (α := α) s)
    (reduction : Reduction := .mean) :
    m (RefTy (m := m) (α := α) Shape.scalar) := do
  let onesT : Tensor α s := Spec.fill (1 : α) s
  let ones ← const (m := m) (α := α) (s := s) onesT
  let oneMinusY ← sub (m := m) (α := α) (s := s) ones target
  let negLogits ← scale (m := m) (α := α) (s := s) logits (-1)
  let spNeg ← softplus (m := m) (α := α) (s := s) negLogits
  let spPos ← softplus (m := m) (α := α) (s := s) logits
  let t1 ← mul (m := m) (α := α) (s := s) target spNeg
  let t2 ← mul (m := m) (α := α) (s := s) oneMinusY spPos
  let lossVec ← add (m := m) (α := α) (s := s) t1 t2
  reduce (m := m) (α := α) (s := s) lossVec reduction

/--
Binary cross-entropy on probabilities (elementwise):
`- (y * log(p) + (1-y) * log(1-p))`.

If you have logits, prefer `bceWithLogits`.
-/
def bce {α : Type} [Context α] [DecidableEq Shape]
    {m : Type → Type} [Monad m] [Ops (m := m) (α := α)]
    {s : Shape}
    (probs target : RefTy (m := m) (α := α) s)
    (reduction : Reduction := .mean) (ε : α := Numbers.epsilon) :
    m (RefTy (m := m) (α := α) Shape.scalar) := do
  let onesT : Tensor α s := Spec.fill (1 : α) s
  let ones ← const (m := m) (α := α) (s := s) onesT
  let oneMinusP ← sub (m := m) (α := α) (s := s) ones probs
  let oneMinusY ← sub (m := m) (α := α) (s := s) ones target
  let logP ← safeLog (m := m) (α := α) (s := s) probs (ε := ε)
  let logOneMinusP ← safeLog (m := m) (α := α) (s := s) oneMinusP (ε := ε)
  let t1 ← mul (m := m) (α := α) (s := s) target logP
  let t2 ← mul (m := m) (α := α) (s := s) oneMinusY logOneMinusP
  let sumT ← add (m := m) (α := α) (s := s) t1 t2
  let negSumT ← scale (m := m) (α := α) (s := s) sumT (-1)
  reduce (m := m) (α := α) (s := s) negSumT reduction

end Loss

end TorchLean
end Autograd
end Runtime
