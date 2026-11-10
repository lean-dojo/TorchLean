/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Tensor.Pack

/-!
# Typed context indices

A tape-style graph names its inputs and intermediates by position. `Idx Γ s` is such a name: a
position in the shape context `Γ` with a proof that the entry there has shape `s`, so a node that
reads its `i`th saved tensor keeps the shape invariant that makes it typecheck. The type does not
depend on tensor elements, and the runtime typed graphs, the tape proofs over `ℝ` and over any
`CommSemiring`, and the runtime-approximation graphs all use it.

## Main declarations

- `TorchLean.Idx` and `TorchLean.TensorPack.getIdx`: a typed position and the tensor of a pack at
  it.
- `TorchLean.Idx.weaken`: the same position in a context extended by more intermediates.
- `TorchLean.Idx.last`: the freshly appended entry of `Γ ++ ss ++ [τ]`.
- `TorchLean.TensorPack.ext_getIdx`: two packs agree when every typed selection agrees.
-/

@[expose] public section

namespace TorchLean

/--
A typed index into a heterogeneous context `Γ`, carrying a proof that the selected entry has the
expected shape `s`.
-/
structure Idx (Γ : List Shape) (s : Shape) where
  /-- Position in the heterogeneous context. -/
  i : Fin Γ.length
  /-- Proof that the selected context entry has shape `s`. -/
  h : Γ.get i = s

/--
Read a tensor out of a context at a typed index, casting along the shape equality the index
carries.

The cast is what makes the result `Tensor α s` instead of `Tensor α (Γ.get idx.i)`, so callers
never have to rewrite the ambient shape by hand.
-/
def TensorPack.getIdx {α : Type} [Storage α] {Γ : List Shape} {s : Shape}
    (xs : TensorPack α Γ) (idx : Idx Γ s) : Tensor α s :=
  Tensor.castShape (xs.get idx.i) idx.h

namespace Idx

/--
Weaken a typed index when the context is extended by appending more shapes.

If `idx : Idx Γ s`, then `weaken idx rest : Idx (Γ ++ rest) s`.
-/
def weaken {Γ : List Shape} {s : Shape} (idx : Idx Γ s) (rest : List Shape) :
    Idx (Γ ++ rest) s :=
  let i' : Fin (Γ ++ rest).length := ⟨idx.i.1, by
    simpa [List.length_append] using
      Nat.lt_of_lt_of_le idx.i.2 (Nat.le_add_right Γ.length rest.length)⟩
  have hget : (Γ ++ rest).get i' = s := by
    simpa only [List.get_eq_getElem, i', List.getElem_append_left idx.i.isLt] using idx.h
  ⟨i', hget⟩

/--
Typed index for the last element of an appended shape list.

`Idx.last` is the canonical index of `τ` in `Γ ++ ss ++ [τ]`.
-/
def last {Γ : List Shape} {ss : List Shape} {τ : Shape} : Idx (Γ ++ ss ++ [τ]) τ :=
  ⟨⟨(Γ ++ ss).length, by
      simp [List.length_append]⟩, by
    simp [List.append_assoc]⟩

end Idx

end TorchLean

namespace TorchLean.TensorPack

/-- Two packs agree when every typed tensor selection agrees. -/
theorem ext_getIdx {α : Type} [Storage α] {shapes : List Shape}
    {a b : TensorPack α shapes} (h : ∀ {shape : Shape} (index : Idx shapes shape),
      getIdx a index = getIdx b index) : a = b := by
  induction shapes with
  | nil => cases a; cases b; rfl
  | cons shape shapes ih =>
    cases a with
    | cons x xs =>
      cases b with
      | cons y ys =>
        have head : x = y := h ⟨0, rfl⟩
        have tail : xs = ys := ih (fun index => h ⟨index.i.succ, index.h⟩)
        rw [head, tail]

end TorchLean.TensorPack
