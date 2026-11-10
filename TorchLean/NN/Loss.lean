/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.Functional
public import TorchLean.Tensor.Layers.Loss

/-!
# Loss

TorchLean loss helpers in the style of `torch.nn.functional`, which keep training loops close to
PyTorch:

```
loss = Loss.mse yhat y
loss = Loss.mse yhat y (reduction := .sum)
```

They use the shared operation interface, so the eager tape and the typed SSA/DAG recorder both run
them.

### PyTorch references

- `torch.nn.functional` (losses overview): https://pytorch.org/docs/stable/nn.functional.html
- `mse_loss`: https://pytorch.org/docs/stable/generated/torch.nn.functional.mse_loss.html
- `cross_entropy`: https://pytorch.org/docs/stable/generated/torch.nn.functional.cross_entropy.html
- `nll_loss`: https://pytorch.org/docs/stable/generated/torch.nn.functional.nll_loss.html
- `binary_cross_entropy_with_logits`:
https://pytorch.org/docs/stable/generated/torch.nn.functional.binary_cross_entropy_with_logits.html
-/

@[expose] public section


open TorchLean.Shape (size size_append)

namespace TorchLean

open TorchLean.Tensor
open Ops
open nn

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
Reduce an elementwise loss tensor to a scalar according to `reduction`, the common final step of
losses like MSE and cross-entropy.
-/
def «reduce» {α : Type} [TorchLean.Storage α] [Context α]
    {m : Type → Type} [Monad m] [Ops (m := m) (α := α)]
    {s : Shape} (x : Ref (m := m) (α := α) s) (reduction : Reduction) :
    m (Ref (m := m) (α := α) Shape.scalar) := by
  cases reduction with
  | mean => exact functional.mean (m := m) (α := α) (s := s) x
  | sum => exact (Ref.cast (Shape.reduce_finRange s) <$> Ops.apply (m := m)
        (α := α) (.reduce (s := s) .sum (List.finRange _) false
          (fun h => by rcases h with h | h <;> cases h)) (x, ()))

/--
Mean squared error (MSE) loss between predictions and targets.

It is backend-generic and supports both `mean` and `sum` reduction.
-/
def mse {α : Type} [TorchLean.Storage α] [Context α]
    {m : Type → Type} [Monad m] [Ops (m := m) (α := α)]
    {s : Shape}
    (yhat y : Ref (m := m) (α := α) s)
    (reduction : Reduction := .mean) :
    m (Ref (m := m) (α := α) Shape.scalar) := by
  cases reduction with
  | mean =>
      exact Ops.apply (m := m) (α := α) (.mseLoss (s := s)) (yhat, y, ())
  | sum =>
      exact (do
        let diff ← Ops.apply (m := m) (α := α) (.sub (s := s)) (yhat, y, ())
        let sq ← functional.square (m := m) (α := α) (s := s) diff
        (Ref.cast (Shape.reduce_finRange s) <$> Ops.apply (m := m) (α := α)
              (.reduce (s := s) .sum (List.finRange _) false
                (fun h => by rcases h with h | h <;> cases h)) (sq, ()))
      )

/--
Weighted mean-squared error.

It returns `sum (weights * (prediction - target)^2)` without implicit normalization, so weights
whose sum is one define a weighted mean. A zero weight excludes a finite coordinate loss, while a
NaN or infinite one gives NaN, because each loss is evaluated before the multiplication.
-/
def mseWeighted {α : Type} [TorchLean.Storage α] [Context α]
    {m : Type → Type} [Monad m] [Ops (m := m) (α := α)]
    {s : Shape}
    (yhat y weights : Ref (m := m) (α := α) s) :
    m (Ref (m := m) (α := α) Shape.scalar) := do
  let diff ← Ops.apply (m := m) (α := α) (.sub (s := s)) (yhat, y, ())
  let squared ← functional.square (m := m) (α := α) (s := s) diff
  let weighted ← Ops.apply (m := m) (α := α) (.mul (s := s)) (squared, weights, ())
  (Ref.cast (Shape.reduce_finRange s) <$> Ops.apply (m := m) (α := α)
        (.reduce (s := s) .sum (List.finRange _) false
          (fun h => by rcases h with h | h <;> cases h)) (weighted, ()))

/--
Cross-entropy on logits with one-hot targets. Eager and typed-graph backends keep the original
logits inside the loss, so a zero target need not multiply an overflowing log-probability.
The same weighted formula accepts soft targets, and both inputs keep their gradients.
`reduction` selects the sum or mean over samples, without normalizing target weights.

For integer class labels, use `crossEntropy`, whose `Fin classes` target tensor records the class
bound in its element type.
-/
def oneHotCrossEntropy {α : Type} [TorchLean.Storage α] [Context α]
    {m : Type → Type} [Monad m] [Ops (m := m) (α := α)]
    {s : Shape} (axis : Nat) [Shape.AxisInBounds axis s]
    (logits targetOneHot : Ref (m := m) (α := α) s)
    (reduction : Reduction := .mean) :
    m (Ref (m := m) (α := α) Shape.scalar) := do
  Ops.apply (.crossEntropy ⟨axis, Shape.AxisInBounds.proof⟩ (reduction == .mean))
      (logits, targetOneHot, ())

/-- Cross-entropy from a fixed teacher's logits along `axis`.

The loss keeps the teacher's logit differences available for a cancellation-resistant student
gradient, so the teacher's predictions need no rounding to probabilities first. The teacher is
detached in both forward and reverse mode, and `mean` divides the summed loss by the number of
class slices. Eager CPU, LibTorch CUDA and typed graphs implement this loss, which has no
canonical IR export.
-/
def distill {α : Type} [TorchLean.Storage α] [Context α]
    {m : Type → Type} [Monad m] [Ops (m := m) (α := α)]
    {s : Shape} (axis : Nat) [Shape.AxisInBounds axis s]
    (student teacher : Ref (m := m) (α := α) s) (reduction : Reduction := .mean) :
    m (Ref (m := m) (α := α) Shape.scalar) := do
  let teacher ← Ops.apply .detach (teacher, ())
  let loss ← Ops.apply (.distill ⟨axis, Shape.AxisInBounds.proof⟩) (student, teacher, ())
  match reduction with
  | .mean => pure loss
  | .sum => Ops.apply .scale (loss, .data (Ops.dataConst
      (Tensor.scalar (Tensor.axisMeanDenom s ⟨axis, Shape.AxisInBounds.proof⟩ : α))), ())

/--
Unreduced indexed negative log-likelihood along an arbitrary class axis.

The shape contract is:

* `logProbs` has shape `prefix ++ [classes] ++ suffix`,
* `classAxis = rank prefix` identifies the `classes` dimension,
* `target` has shape `prefix ++ suffix` and element type `Fin classes`, and
* the result has the same shape as `target`, with the class axis removed.

Each output coordinate is `-logProbs[..., target[...], ...]`, which keeps every suffix coordinate
aligned with its label. Invalid class labels are unrepresentable.
-/
def nllUnreduced {α : Type} [TorchLean.Storage α] [Context α]
    {m : Type → Type} [Monad m] [Ops (m := m) (α := α)]
    {leading trailing : Shape} {classes : Nat}
    (classAxis : Nat) (_hClassAxis : classAxis = leading.rank)
    (logProbs : Ref (m := m) (α := α) (leading ++ (Shape.dim classes trailing)))
    (target : Ops.DataRef (m := m) (α := α) (Fin classes)
      (leading ++ trailing)) :
    m (Ref (m := m) (α := α) (leading ++ trailing)) := do
  let toIndices := fun (target : Tensor (Fin classes) (leading ++ trailing)) =>
    (Tensor.ofFn fun i =>
      let targetClass := target.getFlat i
      let inner := Shape.size trailing
      let outer := i.val / inner
      let flatIndex := outer * (classes * inner) + targetClass.val * inner + i.val % inner
      ⟨flatIndex, by
        have hInner : 0 < inner := by
          by_contra h
          have hZero : Shape.size trailing = 0 := by
            simpa [inner] using Nat.eq_zero_of_not_pos h
          simpa [Shape.size_append, hZero] using i.isLt
        have hOuter : outer < Shape.size leading := by
          apply Nat.div_lt_of_lt_mul
          simpa [Shape.size_append, Nat.mul_comm] using i.isLt
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
          _ = Shape.size (leading ++ (Shape.dim classes trailing)) := by
            simp [inner, Shape.size_append]⟩ :
      Tensor (Fin (Shape.size (leading ++ (Shape.dim classes trailing))))
        [Shape.size (leading ++ trailing)])
  let flat ← Ops.apply (m := m) (α := α)
      (.reshape (s := leading ++ (Shape.dim classes trailing))
        (t := [Shape.size (leading ++ (Shape.dim classes trailing))]) (by simp))
      (logProbs, ())
  -- The flattened log-probabilities at the flattened targets: a gather along axis 0.
  let flatTarget := Ops.mapData (m := m) (α := α)
    (fun x => Tensor.map (fun tuple => tuple ⟨0, Nat.one_pos⟩)
      (Tensor.selectIndices (s := [Shape.size (leading ++ (Shape.dim classes trailing))])
        ⟨0, Shape.AxisInBounds.proof⟩ (toIndices x))) target
  let picked : Ref (m := m) (α := α) [Shape.size (leading ++ trailing)] ←
    Ref.cast (m := m) (α := α)
        (by rw [Shape.eraseAxes_selectBatch, List.append_nil]; rfl) <$>
      Ops.apply (m := m) (α := α)
        (.gather
          (Shape.selectDims [Shape.size (leading ++ (Shape.dim classes trailing))]
            ⟨0, Shape.AxisInBounds.proof⟩)
          (Shape.selectBatch [Shape.size (leading ++ (Shape.dim classes trailing))]
            ⟨0, Shape.AxisInBounds.proof⟩ (Shape.size (leading ++ trailing)))
          (.select [Shape.size (leading ++ (Shape.dim classes trailing))]
            ⟨0, Shape.AxisInBounds.proof⟩ (Shape.size (leading ++ trailing))))
        (flat, flatTarget, ())
  let losses ← Ops.apply (m := m) (α := α) (.scale (s := [Shape.size (leading ++ trailing)]))
      (picked, .data (Ops.dataConst (m := m) (α := α) (Tensor.scalar (-1))), ())
  Ops.apply (m := m) (α := α)
      (.reshape (s := [Shape.size (leading ++ trailing)]) (t := leading ++ trailing)
        (by simp)) (losses, ())

/--
Indexed negative log-likelihood along an arbitrary class axis.

The log-probabilities, labels and class-axis contract are those of `nllUnreduced`. The requested
reduction is applied over every coordinate of the class-erased output shape.
-/
def nll {α : Type} [TorchLean.Storage α] [Context α]
    {m : Type → Type} [Monad m] [Ops (m := m) (α := α)]
    {leading trailing : Shape} {classes : Nat}
    (classAxis : Nat) (hClassAxis : classAxis = leading.rank)
    (logProbs : Ref (m := m) (α := α) (leading ++ (Shape.dim classes trailing)))
    (target : Ops.DataRef (m := m) (α := α) (Fin classes)
      (leading ++ trailing))
    (reduction : Reduction := .mean) :
    m (Ref (m := m) (α := α) Shape.scalar) := do
  let losses ← nllUnreduced (m := m) (α := α) classAxis hClassAxis logProbs target
  Loss.reduce (m := m) (α := α) (s := leading ++ trailing) losses reduction

/--
Indexed cross-entropy along an arbitrary class axis.

It applies log-softmax along `classAxis` and then `nll`. Shapes follow the
`prefix ++ [classes] ++ suffix` contract documented on `nllUnreduced`.
-/
def crossEntropy {α : Type} [TorchLean.Storage α] [Context α]
    {m : Type → Type} [Monad m] [Ops (m := m) (α := α)]
    {leading trailing : Shape} {classes : Nat}
    (classAxis : Nat) (hClassAxis : classAxis = leading.rank)
    (logits : Ref (m := m) (α := α) (leading ++ (Shape.dim classes trailing)))
    (target : Ops.DataRef (m := m) (α := α) (Fin classes)
      (leading ++ trailing))
    (reduction : Reduction := .mean) :
    m (Ref (m := m) (α := α) Shape.scalar) :=
  letI : Shape.AxisInBounds classAxis (leading ++ (Shape.dim classes trailing)) :=
    Shape.AxisInBounds.mk <| by
      rw [hClassAxis]
      simp [Shape.rank_concat]
  do
  let logp ← Ops.apply (m := m) (α := α)
      (.logSoftmax (s := leading ++ (Shape.dim classes trailing))
        ⟨classAxis, Shape.AxisInBounds.proof⟩) (logits, ())
  nll (m := m) (α := α) classAxis hClassAxis logp target (reduction := reduction)

/--
Binary cross-entropy with logits (elementwise), using the stable identity:
`BCEWithLogits(x,y) = y * softplus(-x) + (1-y) * softplus(x)`.

Targets are expected in `[0,1]` (typically 0/1), same shape as `logits`.
-/
def bceWithLogits {α : Type} [TorchLean.Storage α] [Context α]
    {m : Type → Type} [Monad m] [Ops (m := m) (α := α)]
    {s : Shape}
    (logits target : Ref (m := m) (α := α) s)
    (reduction : Reduction := .mean) :
    m (Ref (m := m) (α := α) Shape.scalar) := do
  let onesT : Tensor α s := Tensor.full s (1 : α)
  let ones ← Ops.const (m := m) (α := α) (s := s) onesT
  let oneMinusY ← Ops.apply (m := m) (α := α) (.sub (s := s)) (ones, target, ())
  let negLogits ← Ops.apply (m := m) (α := α) (.scale (s := s))
      (logits, .data (Ops.dataConst (m := m) (α := α) (Tensor.scalar (-1))), ())
  let spNeg ← Ops.apply (m := m) (α := α) (.softplus (s := s)) (negLogits, ())
  let spPos ← Ops.apply (m := m) (α := α) (.softplus (s := s)) (logits, ())
  let t1 ← Ops.apply (m := m) (α := α) (.mul (s := s)) (target, spNeg, ())
  let t2 ← Ops.apply (m := m) (α := α) (.mul (s := s)) (oneMinusY, spPos, ())
  let lossVec ← Ops.apply (m := m) (α := α) (.add (s := s)) (t1, t2, ())
  Loss.reduce (m := m) (α := α) (s := s) lossVec reduction

end Loss

end TorchLean
