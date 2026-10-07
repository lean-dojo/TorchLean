/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.IRExec.Core

/-!
# IR Lowering Primitives

Typed state, indices, and shape operations used by the checked IR lowering pass. These declarations
remain under `IRExec.Internal`; the public lowering entry point lives in `IRExec.Lowering`.
-/

@[expose] public section

namespace Runtime
namespace Autograd
namespace IRExec

open Spec
open TorchLean TorchLean.Tensor
open Proofs.Autograd.Algebra
-- Typed context indices come from `NN.Proofs.Autograd.Tape.Util.Idx`, the one place
-- `Idx` and `getIdx` are defined.
open Proofs (Idx getIdx)
open NN.IR

namespace Internal

/--
Internal lowering state used by `buildFrom`.

It is a dependent pair of:
- `ss`: shapes of already-lowered IR nodes,
- `ForwardData α [inShape] ss`: forward closures for exactly that shape list.
-/
abbrev State (α : Type) [TorchLean.Storage α] (inShape : Shape) : Type :=
  Σ ss : List Shape, ForwardData α [inShape] ss

/--
Build a typed runtime index (`Idx`) for a numeric IR parent id.

The forward executor's context is typed by `[inShape] ++ ss`, matching `ForwardData`'s
input-plus-node representation. `mkIdx` checks that:
- `id` is in bounds, and
- the context shape at that position matches the expected shape `s`.

On failure, this returns a descriptive error string used directly by `buildFrom`.
-/
def mkIdx
    (inShape : Shape) (ss : List Shape) (id : Nat) (s : Shape) :
    Except String (Idx ([inShape] ++ ss) s) := by
  let ctxShapes : List Shape := [inShape] ++ ss
  if h : id < ctxShapes.length then
    let fin : Fin ctxShapes.length := ⟨id, h⟩
    let got : Shape := ctxShapes.get fin
    if hg : got = s then
      exact .ok ⟨fin, hg⟩
    else
      exact .error
        s!"IRExec: shape mismatch at id={id}: expected {Shape.pretty s}, got {Shape.pretty got}"
  else
    exact .error s!"IRExec: invalid id={id} for ctxLen={ctxShapes.length}"

/-- Package a typed forward closure as one node of the executable IR graph. -/
def mkForwardNode {α : Type} [TorchLean.Storage α] {Γ : List Shape} {τ : Shape}
    (forward : TensorLookup α Γ → Tensor α τ) : ForwardNode α Γ τ :=
  ⟨forward⟩

/-- Running a constructed forward node applies its reader closure. -/
@[simp] theorem mkForwardNode_run {α : Type} [TorchLean.Storage α] {Γ : List Shape} {τ : Shape}
    (f : TensorLookup α Γ → Tensor α τ) (ctx : TensorLookup α Γ) :
    (mkForwardNode f).run ctx = f ctx := rfl

/--
Evaluation projection for `mkForwardNode`.
-/
@[simp] theorem mkForwardNode_eval {α : Type} [TorchLean.Storage α] {Γ : List Shape} {τ : Shape}
    (f : TensorLookup α Γ → Tensor α τ) (ctx : TorchLean.TensorPack α Γ) :
    (mkForwardNode (α := α) (Γ := Γ) (τ := τ) f).eval ctx = f ctx := rfl

/-- Internal list recursion used to track the dependent output shape of adjacent swaps. -/
def swapShapeList (s : Shape) : List Nat → Shape
  | [] => s
  | d :: ds => swapShapeList (s.swapAdjacentAtDepth d) ds

/-- Apply adjacent swaps, represented by their axis depths, to a shape. -/
def swapShape (s : Shape) (swaps : Array Nat) : Shape :=
  swapShapeList s swaps.toList

/-- Internal dependent recursion underlying `swapAxes`. -/
def swapAxesList {α : Type} [TorchLean.Storage α] :
    {s : Shape} → (swaps : List Nat) → Tensor α s → Tensor α (swapShapeList s swaps)
  | _s, [], t => t
  | s, d :: ds, t =>
      let t' : Tensor α (s.swapAdjacentAtDepth d) := Tensor.swapAdjacentAxes (tensor := t) d
      swapAxesList (s := s.swapAdjacentAtDepth d) (swaps := ds) t'

/-- Apply the same adjacent-swap sequence as `swapShape` to a tensor value. -/
def swapAxes {α : Type} [TorchLean.Storage α] {s : Shape} (swaps : Array Nat)
    (tensor : Tensor α s) : Tensor α (swapShape s swaps) :=
  swapAxesList swaps.toList tensor

/--
One typed concat input: a leading extent together with a closure that reads the tensor with that
extent from the runtime context. Inputs for a nonzero concat axis permute the parent before
returning it, so the closure is the common shape for every concat branch.
-/
abbrev ConcatInput (α : Type) [TorchLean.Storage α] (Γ : List Shape) (rest : Shape) : Type :=
  Sigma fun nP => TensorLookup α Γ → Tensor α (.dim nP rest)

/--
Concatenate typed tensors along their leading axis, folding from the first tensor.

The empty list yields the empty tensor with leading extent `0`. This is the same fold shape as
the IR evaluator's `NN.IR.Graph.evalConcatLeadingAxisFold`.
-/
def concatList {α : Type} [TorchLean.Storage α] [Zero α] {rest : Shape} :
    List (Sigma fun n => Tensor α (.dim n rest)) → Sigma fun nSum => Tensor α (.dim nSum rest)
  | [] => ⟨0, Tensor.full (α := α) (.dim 0 rest) 0⟩
  | first :: others =>
      others.foldl
        (fun acc nxt =>
          ⟨acc.1 + nxt.1, Tensor.concatAxisSpec .scalar (α := α) (n := acc.1) (m := nxt.1)
            (suffix := rest) acc.2 nxt.2⟩)
        first

/-- The leading extent of the fold in `concatList` is a plain sum of extents. -/
theorem concatList_fst {α : Type} [TorchLean.Storage α] [Zero α] {rest : Shape}
    (tensors : List (Sigma fun n => Tensor α (.dim n rest))) :
    (concatList (α := α) (rest := rest) tensors).1 =
      tensors.foldl (fun acc t => acc + t.1) 0 := by
  cases tensors with
  | nil => simp [concatList]
  | cons first others =>
      simp only [concatList, List.foldl_cons, Nat.zero_add]
      exact (List.foldl_hom Sigma.fst (fun _ _ => rfl)).symm

/-- Concatenate the tensors produced by concat inputs along their leading axis. -/
def concatInputs
    {α : Type} [TorchLean.Storage α] [Zero α] {Γ : List Shape} {rest : Shape}
    (ctx : TensorLookup α Γ) (inputs : Array (ConcatInput α Γ rest)) :
    Sigma fun nSum => Tensor α (.dim nSum rest) :=
  concatList (inputs.toList.map fun input => ⟨input.1, input.2 ctx⟩)

/--
The concatenated size reported by `concatInputs` is the sum of the input extents.

This theorem justifies the output-shape cast in the concat lowering branches.
-/
theorem concatInputs_size_eq_sum
    {α : Type} [TorchLean.Storage α] [Zero α] {Γ : List Shape} {rest : Shape}
    (ctx : TensorLookup α Γ) (inputs : Array (ConcatInput α Γ rest)) :
    (concatInputs (α := α) (Γ := Γ) (rest := rest) ctx inputs).1 =
      inputs.foldl (fun acc input => acc + input.1) 0 := by
  simpa only [concatInputs, concatList_fst, List.foldl_map] using
    (Array.foldl_toList (xs := inputs) (f := fun acc input => acc + input.1) (init := 0))

end Internal
end IRExec
end Autograd
end Runtime
