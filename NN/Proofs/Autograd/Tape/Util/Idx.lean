/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Tensor.Pack

/-!
# Typed context indices

A tape-style graph names its inputs and intermediates by position, so every node has to say "the
`i`th saved tensor" without losing the shape invariant that makes the node typecheck. `Idx Γ s`
is that name: a position in the context `Γ` bundled with a proof that the entry sitting there has
shape `s`.

The type carries no element type, which is why it lives here rather than beside any one soundness
development. The real-valued tape proofs, the `CommSemiring`-generic ones, and the
runtime-approximation graphs all index contexts the same way, and they used to do it through three
byte-identical copies of this structure. One definition means a lemma about indices proved in one
of those developments is usable in the others.

Alongside the structure are the two operations every graph construction needs:

- `Idx.weaken` extends the context with more intermediates and keeps the index valid;
- `Idx.last` names the freshly appended entry of `Γ ++ ss ++ [τ]`.

Both are pure list arithmetic, and centralizing them keeps that boilerplate out of every op graph
(LayerNorm, BatchNorm, attention, …).
-/

@[expose] public section


namespace Proofs

open Spec TorchLean

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
def getIdx {α : Type} [TorchLean.Storage α] {Γ : List Shape} {s : Shape}
    (xs : TorchLean.TensorPack α Γ) (idx : Idx Γ s) : Tensor α s :=
  Tensor.castShape (xs.get (α := α) idx.i) idx.h

namespace Idx

private theorem get_append_last {α : Type} (l : List α) (a : α) :
    (l ++ [a]).get ⟨l.length, by simp⟩ = a := by
  induction l with
  | nil => simp
  | cons _ xs ih =>
      simp [List.length]

private theorem get_append_left {α : Type} (l₁ l₂ : List α) (i : Fin l₁.length) :
    (l₁ ++ l₂).get ⟨i.1, by
        -- `i.1 < l₁.length` and `l₁.length ≤ l₁.length + l₂.length`.
        simpa [List.length_append] using
          Nat.lt_of_lt_of_le i.2 (Nat.le_add_right l₁.length l₂.length)⟩ =
      l₁.get i := by
  induction l₁ with
  | nil =>
      cases i with
      | mk _ hk => cases hk
  | cons _ tl ih =>
      classical
      cases i using Fin.cases with
      | zero =>
          simp
      | succ i =>
          simp

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
    have hleft := get_append_left (l₁ := Γ) (l₂ := rest) (i := idx.i)
    have hi' :
        (⟨idx.i.1, by
          simpa [List.length_append] using
            Nat.lt_of_lt_of_le idx.i.2 (Nat.le_add_right Γ.length rest.length)⟩ :
          Fin (Γ ++ rest).length) = i' := by
      ext; rfl
    simpa [hi'] using (hleft.trans idx.h)
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

end Proofs
