/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Model.Functional.Einsum

@[expose] public section


namespace Runtime
namespace Autograd
namespace Model

open Spec TorchLean
open TorchLean TorchLean.Tensor

namespace F

/-! ## Shape and axis operations -/

/--
Dynamic permutation: like `permute`, but returns an existential output shape.

PyTorch analogue: `torch.permute` / `Tensor.permute` (with runtime checks).
-/
def permute? {α : Type} [TorchLean.Storage α] [Context α]
    {m : Type → Type} [Monad m] [Ops (m := m) (α := α)]
    {s : Shape}
    (axes : Array Nat)
    (x : Ref (m := m) (α := α) s) :
    m (Option (Σ s' : Shape, Ref (m := m) (α := α) s')) := do
  let r := Spec.Shape.rank s
  if axes.size != r then
    return none
  if !axes.allDiff then
    return none
  if !(axes.all (fun a => a < r)) then
    return none
  let some swaps := Einsum.swapDepthsForPerm? axes.toList r | return none
  let out ← Einsum.permuteBySwaps (α := α) (m := m) ⟨s, x⟩ swaps
  pure (some out)

/--
Permutation with an expected output shape.

This calls `permute?` and checks that the computed shape equals `sOut`.
-/
def permute {α : Type} [TorchLean.Storage α] [Context α]
    {m : Type → Type} [Monad m] [Ops (m := m) (α := α)]
    {s sOut : Shape}
    (axes : Array Nat)
    (x : Ref (m := m) (α := α) s) :
    m (Option (Ref (m := m) (α := α) sOut)) := do
  let y? ← permute? (α := α) (m := m) (s := s) axes x
  match y? with
  | none => pure none
  | some ⟨s', y⟩ =>
      if h : s' = sOut then
        pure (some (h ▸ y))
      else
        pure none

/--
Exchange two arbitrary axes and check the statically expected output shape.

The result is `none` when either axis is invalid or the transposed shape is not `sOut`.
-/
def transpose {α : Type} [TorchLean.Storage α] [Context α]
    {m : Type → Type} [Monad m] [Ops (m := m) (α := α)]
    {s sOut : Shape} (axis₁ axis₂ : Nat)
    (x : Ref (m := m) (α := α) s) :
    m (Option (Ref (m := m) (α := α) sOut)) :=
  if axis₁ < s.rank && axis₂ < s.rank then
    permute (α := α) (m := m) (s := s) (sOut := sOut)
      (Shape.transposePermutation s.rank axis₁ axis₂).toArray x
  else
    pure none

namespace Internal

/-- Reduce the last axis after checking that the tensor has nonempty dimensions. -/
def reduceAlongLast {α : Type} [TorchLean.Storage α] [Context α]
    {m : Type → Type} [Monad m] [Ops (m := m) (α := α)]
    (operation : {s : Shape} → (axis : Nat) →
      [Shape.HasNonemptyAxis axis s] → [Shape.WellFormed s] →
      Ref (m := m) (α := α) s →
        m (Ref (m := m) (α := α) (TorchLean.Tensor.shapeAfterSum s axis)))
    (x : Σ s : Shape, Ref (m := m) (α := α) s) :
    m (Option (Σ s' : Shape, Ref (m := m) (α := α) s')) := do
  let s := x.fst
  if hw : s.wellFormed then
    letI : Shape.WellFormed s := ⟨hw⟩
    if hRank : Spec.Shape.rank s > 0 then
      let axis := Spec.Shape.rank s - 1
      haveI : Shape.HasNonemptyAxis axis s :=
        Shape.inferNonemptyAxis (by grind)
      operation axis x.snd >>= fun y =>
        pure (some ⟨TorchLean.Tensor.shapeAfterSum s axis, y⟩)
    else
      pure none
  else
    pure none

/--
Core implementation for reductions over a runtime array of axes.

This lowers “reduce along axis k” to:
1. permute axis `k` to the last position,
2. call `reduceLast`, and
3. optionally re-insert a singleton dimension when `keepdim = true`.

`reduceSumDims?` and `reduceMeanDims?` specialize this operation.
-/
def reduceAxes {α : Type} [TorchLean.Storage α] [Context α]
    {m : Type → Type} [Monad m] [Ops (m := m) (α := α)]
    (reduceLast :
      (Σ s : Shape, Ref (m := m) (α := α) s) →
        m (Option (Σ s' : Shape, Ref (m := m) (α := α) s')))
    {s : Shape}
    (axes : Array Nat)
    (keepdim : Bool)
    (x : Ref (m := m) (α := α) s) :
    m (Option (Σ s' : Shape, Ref (m := m) (α := α) s')) := do
  let r0 := Spec.Shape.rank s
  if !axes.allDiff then
    return none
  if !(axes.all (fun a => a < r0)) then
    return none
  let axes' := if keepdim then axes else (axes.toList.mergeSort (· ≥ ·)).toArray
  let mut cur : Σ s : Shape, Ref (m := m) (α := α) s := ⟨s, x⟩
  for axis in axes' do
    let r := Spec.Shape.rank cur.fst
    if axis ≥ r then
      return none
    let swaps := Shape.moveAxisToInnermostSwaps r axis
    let curMoved ← Einsum.permuteBySwaps (α := α) (m := m) cur swaps
    let some curRed ← reduceLast curMoved | return none
    if keepdim then
      let sReshape : Shape := Shape.appendDim curRed.fst 1
      have hSz : Spec.Shape.size curRed.fst = Spec.Shape.size sReshape := by
        simpa [sReshape] using (Spec.Shape.size_appendDim curRed.fst 1).symm
      let xReshaped ← reshape (m := m) (α := α) (s₁ := curRed.fst) (s₂ := sReshape) curRed.snd hSz
      let curKeep : Σ s : Shape, Ref (m := m) (α := α) s := ⟨sReshape, xReshaped⟩
      let curBack ← Einsum.permuteBySwaps (α := α) (m := m) curKeep swaps.reverse
      cur := curBack
    else
      cur := curRed
  pure (some cur)

end Internal

/-- Dynamic multi-axis sum reduction (like `torch.sum(x, dim=axes, keepdim=...)`). -/
def reduceSumDims? {α : Type} [TorchLean.Storage α] [Context α]
    {m : Type → Type} [Monad m] [Ops (m := m) (α := α)]
    {s : Shape}
    (axes : Array Nat)
    (x : Ref (m := m) (α := α) s)
    (keepdim : Bool := false) :
    m (Option (Σ s' : Shape, Ref (m := m) (α := α) s')) :=
  Internal.reduceAxes (α := α) (m := m)
    (Internal.reduceAlongLast (Runtime.Autograd.Torch.Ops.reduceSum (m := m) (α := α)))
    (s := s) axes keepdim x

/-- Dynamic multi-axis mean reduction (like `torch.mean(x, dim=axes, keepdim=...)`). -/
def reduceMeanDims? {α : Type} [TorchLean.Storage α] [Context α]
    {m : Type → Type} [Monad m] [Ops (m := m) (α := α)]
    {s : Shape}
    (axes : Array Nat)
    (x : Ref (m := m) (α := α) s)
    (keepdim : Bool := false) :
    m (Option (Σ s' : Shape, Ref (m := m) (α := α) s')) :=
  Internal.reduceAxes (α := α) (m := m)
    (Internal.reduceAlongLast (Runtime.Autograd.Torch.Ops.reduceMean (m := m) (α := α)))
    (s := s) axes keepdim x

/-- Softmax along any valid tensor dimension.

The selected dimension is moved to the end for the backend's row-softmax primitive and then moved
back. The reverse-swap theorem makes the result shape exactly `s`; no runtime shape check is needed.
-/
def softmax {α : Type} [TorchLean.Storage α] [Context α]
    {m : Type → Type} [Monad m] [Ops (m := m) (α := α)]
    {s : Shape} (axis : Nat) [Shape.AxisInBounds axis s]
    (x : Ref (m := m) (α := α) s) :
    m (Ref (m := m) (α := α) s) := do
  let swaps := Shape.moveAxisToInnermostSwaps (Spec.Shape.rank s) axis
  let moved ← Einsum.permuteBySwapsTyped (α := α) (m := m) x swaps
  let y ← Runtime.Autograd.Torch.softmaxLast (m := m) (α := α) moved
  let restored ← Einsum.permuteBySwapsTyped (α := α) (m := m) y swaps.reverse
  pure ((Shape.applyAdjacentSwaps_reverse s swaps) ▸ restored)

/-- Log-softmax along any valid tensor dimension. -/
def logSoftmax {α : Type} [TorchLean.Storage α] [Context α]
    {m : Type → Type} [Monad m] [Ops (m := m) (α := α)]
    {s : Shape} (axis : Nat) [Shape.AxisInBounds axis s]
    (x : Ref (m := m) (α := α) s) :
    m (Ref (m := m) (α := α) s) := do
  let swaps := Shape.moveAxisToInnermostSwaps (Spec.Shape.rank s) axis
  let moved ← Einsum.permuteBySwapsTyped (α := α) (m := m) x swaps
  let y ← Runtime.Autograd.Torch.logSoftmaxLast (m := m) (α := α) moved
  let restored ← Einsum.permuteBySwapsTyped (α := α) (m := m) y swaps.reverse
  pure ((Shape.applyAdjacentSwaps_reverse s swaps) ▸ restored)

end F
end Model
end Autograd
end Runtime
