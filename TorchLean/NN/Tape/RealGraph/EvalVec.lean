/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.Tape.RealNode.Context

/-!
# Reading intermediate values out of an evaluated tape graph

`Tape.GraphData.evalVec` returns the whole flattened context, inputs followed by every saved
intermediate. The pointwise normalization proofs need to know what a specific block of that vector
is: the variance block of the LayerNorm prefix, for instance, must be shown nonnegative before
`sqrt` and `inv` can be differentiated.

The lemmas here describe `CtxVec.get` in coordinates and give the three rules that let a proof
walk back through a `snoc` chain:

* `Tape.GraphData.get_evalVec_snoc_last`: the block just appended is the node's forward value.
* `Tape.GraphData.get_evalVec_snoc_of_lt`: any earlier block is unchanged by appending a node.
* `Tape.GraphData.get_evalVec_input`: an input block is unchanged by the whole graph.
-/

@[expose] public section

open TorchLean.Shape (size)

namespace TorchLean

noncomputable section

namespace CtxVec

/-- Offset of the block at list position `k` inside the flattened context vector. -/
def blockOffset : List Shape → Nat → Nat
  | [], _ => 0
  | _ :: _, 0 => 0
  | s :: ss, k + 1 => Shape.size s + blockOffset ss k

/-- Every coordinate of a block lies inside the flattened context. -/
theorem blockOffset_add_lt :
    ∀ {Γ : List Shape} (i : Fin Γ.length) (j : Fin (Shape.size (Γ.get i))),
      blockOffset Γ i.val + j.val < ctxSize Γ
  | [], i, _ => nomatch i
  | s :: ss, ⟨0, _⟩, j => by
      have hj := j.isLt
      simp only [List.get_eq_getElem, List.getElem_cons_zero] at hj
      simp only [blockOffset, ctxSize, Nat.zero_add]
      exact Nat.lt_of_lt_of_le hj (Nat.le_add_right _ _)
  | s :: ss, ⟨k + 1, hk⟩, j => by
      have ih := blockOffset_add_lt (Γ := ss) ⟨k, Nat.lt_of_succ_lt_succ hk⟩ j
      simp only [blockOffset, ctxSize] at ih ⊢
      rw [Nat.add_assoc]
      exact Nat.add_lt_add_left ih _

/-- Coordinates of `getBlock` are coordinates of the context, shifted by the block offset. -/
theorem getBlock_ofLp :
    ∀ {Γ : List Shape} (i : Fin Γ.length) (x : CtxVec Γ)
      (j : Fin (Shape.size (Γ.get i))),
      (getBlock (Γ := Γ) i x).ofLp j =
        x.ofLp ⟨blockOffset Γ i.val + j.val, blockOffset_add_lt i j⟩
  | [], i, _, _ => nomatch i
  | s :: ss, ⟨0, _⟩, x, j => by
      simp only [getBlock, PiLp.toLp_apply]
      exact congrArg x.ofLp (Fin.ext (by simp [blockOffset]))
  | s :: ss, ⟨k + 1, hk⟩, x, j => by
      simp only [getBlock]
      rw [getBlock_ofLp (Γ := ss) ⟨k, Nat.lt_of_succ_lt_succ hk⟩ _ j]
      simp only
      exact congrArg x.ofLp (Fin.ext (by simp [blockOffset, Nat.add_assoc]))

/-- Every coordinate of the block selected by a typed index lies inside the context. -/
theorem blockOffset_add_lt' {Γ : List Shape} {s : Shape} (idx : Idx Γ s)
    (j : Fin (Shape.size s)) :
    blockOffset Γ idx.i.val + j.val < ctxSize Γ := by
  have h := blockOffset_add_lt idx.i (Fin.cast (congrArg Shape.size idx.h).symm j)
  simpa using h

/-- Coordinate description of `CtxVec.get`. -/
theorem get_ofLp {Γ : List Shape} {s : Shape} (idx : Idx Γ s) (x : CtxVec Γ)
    (j : Fin (Shape.size s)) :
    (get (Γ := Γ) (s := s) idx x).ofLp j =
      x.ofLp ⟨blockOffset Γ idx.i.val + j.val, blockOffset_add_lt' idx j⟩ := by
  simp only [get, LinearIsometryEquiv.piLpCongrLeft_apply, Equiv.piCongrLeft'_apply,
    finCongr_symm_apply]
  rw [getBlock_ofLp]
  rfl

/-- The block right after a prefix `Γ` starts at `ctxSize Γ`. -/
theorem blockOffset_append_length :
    ∀ (Γ rest : List Shape), blockOffset (Γ ++ rest) Γ.length = ctxSize Γ
  | [], rest => by cases rest <;> simp [blockOffset, ctxSize]
  | s :: Γ, rest => by
      simp [blockOffset, ctxSize, blockOffset_append_length Γ rest]

/-- Appending shapes does not move the blocks of the prefix. -/
theorem blockOffset_append_of_lt :
    ∀ (Γ rest : List Shape) (k : Nat), k < Γ.length →
      blockOffset (Γ ++ rest) k = blockOffset Γ k
  | [], _, k, hk => absurd hk (Nat.not_lt_zero k)
  | _ :: _, _, 0, _ => by simp [blockOffset]
  | s :: Γ, rest, k + 1, hk => by
      simp [blockOffset, blockOffset_append_of_lt Γ rest k (Nat.lt_of_succ_lt_succ hk)]

/-- The first block of a flattened tensor pack is the first tensor. -/
theorem getBlock_toVec_zero {s : Shape} {ss : List Shape} (X : Tensor ℝ s)
    (xs : TensorPack ℝ ss) (h : 0 < (s :: ss).length) :
    getBlock (Γ := s :: ss) ⟨0, h⟩ (TensorPack.toVec (Γ := s :: ss) (.cons X xs)) =
      Tensor.toVec X := by
  rw [getBlock_toVec, Pack.get_cons_zero]

/-- Later blocks of a flattened tensor pack are blocks of the tail. -/
theorem getBlock_toVec_succ {s : Shape} {ss : List Shape} (X : Tensor ℝ s)
    (xs : TensorPack ℝ ss) (k : Nat) (hk : k + 1 < (s :: ss).length) :
    getBlock (Γ := s :: ss) ⟨k + 1, hk⟩ (TensorPack.toVec (Γ := s :: ss) (.cons X xs)) =
      getBlock (Γ := ss) ⟨k, Nat.lt_of_succ_lt_succ hk⟩ (TensorPack.toVec (Γ := ss) xs) := by
  -- Both sides are `Tensor.toVec` of a pack lookup, and the lookup at `k + 1` is the tail lookup
  -- at `k` by definition of `Pack.get`.
  rw [getBlock_toVec, getBlock_toVec]
  rfl

end CtxVec

/-- Coordinates of `cast` are coordinates of the original vector. -/
theorem CtxVec.cast_ofLp {Γ₁ Γ₂ : List Shape} (h : Γ₁ = Γ₂) (x : CtxVec Γ₁)
    (k : Fin (ctxSize Γ₂)) :
    (cast (Γ₁ := Γ₁) (Γ₂ := Γ₂) h x).ofLp k =
      x.ofLp (Fin.cast (congrArg ctxSize h).symm k) := by
  simp [cast]

/-- Coordinates of `snoc` below the prefix size read the prefix. -/
theorem CtxVec.snoc_ofLp_of_lt {Γ : List Shape} {τ : Shape} (ctx : CtxVec Γ)
    (t : Vec (Shape.size τ)) (k : Fin (ctxSize (Γ ++ [τ]))) (hk : k.val < ctxSize Γ) :
    (snoc (Γ := Γ) (τ := τ) ctx t).ofLp k = ctx.ofLp ⟨k.val, hk⟩ := by
  simp only [snoc, LinearIsometryEquiv.piLpCongrLeft_apply, Equiv.piCongrLeft'_apply,
    finCongr_symm_apply]
  have hcast :
      Fin.cast (ctxSize_snoc Γ τ).symm.symm k = Fin.castAdd (Shape.size τ) ⟨k.val, hk⟩ := by
    ext
    simp
  rw [hcast, Vec.append_ofLp_castAdd]

/-- Coordinates of `snoc` at or above the prefix size read the appended block. -/
theorem CtxVec.snoc_ofLp_last {Γ : List Shape} {τ : Shape} (ctx : CtxVec Γ)
    (t : Vec (Shape.size τ)) (k : Fin (ctxSize (Γ ++ [τ]))) (j : Fin (Shape.size τ))
    (hk : k.val = ctxSize Γ + j.val) :
    (snoc (Γ := Γ) (τ := τ) ctx t).ofLp k = t.ofLp j := by
  simp only [snoc, LinearIsometryEquiv.piLpCongrLeft_apply, Equiv.piCongrLeft'_apply,
    finCongr_symm_apply]
  have hcast : Fin.cast (ctxSize_snoc Γ τ).symm.symm k = Fin.natAdd (ctxSize Γ) j := by
    ext
    simp [hk]
  rw [hcast, Vec.append_ofLp_natAdd]

namespace Tape

namespace GraphData

/-- Evaluating the empty graph leaves every block in place. -/
theorem get_evalVec_nil {Δ : Type} {Γ : List Shape} {s : Shape} (d : Δ) (xV : CtxVec Γ)
    (idx : Idx (Γ ++ []) s) (idx' : Idx Γ s) (hv : idx.i.val = idx'.i.val) :
    CtxVec.get (Γ := Γ ++ []) (s := s) idx (evalVec (Γ := Γ) (ss := []) .nil d xV) =
      CtxVec.get (Γ := Γ) (s := s) idx' xV := by
  apply PiLp.ext
  intro j
  rw [CtxVec.get_ofLp, CtxVec.get_ofLp]
  simp only [evalVec, CtxVec.cast_ofLp]
  exact congrArg xV.ofLp (Fin.ext (by simp [hv]))

/-- The block appended by `snoc` is the node's forward value on the prefix evaluation. -/
theorem get_evalVec_snoc_last {Δ : Type} {Γ ss : List Shape} {τ : Shape}
    (g : GraphData ℝ Δ Γ ss) (node : NodeData ℝ Δ (Γ ++ ss) τ) (d : Δ) (xV : CtxVec Γ)
    (idx : Idx (Γ ++ (ss ++ [τ])) τ) (hv : idx.i.val = (Γ ++ ss).length) :
    CtxVec.get (Γ := Γ ++ (ss ++ [τ])) (s := τ) idx
        (evalVec (Γ := Γ) (ss := ss ++ [τ]) (.snoc g node) d xV) =
      node.forwardVec (Γ := Γ ++ ss) (τ := τ) d (evalVec (Γ := Γ) (ss := ss) g d xV) := by
  apply PiLp.ext
  intro j
  rw [CtxVec.get_ofLp]
  simp only [evalVec, CtxVec.cast_ofLp]
  rw [CtxVec.snoc_ofLp_last _ _ _ j]
  simp only [Fin.val_cast]
  rw [hv, ← List.append_assoc, CtxVec.blockOffset_append_length]

/-- Blocks below the appended one are unchanged by `snoc`. -/
theorem get_evalVec_snoc_of_lt {Δ : Type} {Γ ss : List Shape} {τ s : Shape}
    (g : GraphData ℝ Δ Γ ss) (node : NodeData ℝ Δ (Γ ++ ss) τ) (d : Δ) (xV : CtxVec Γ)
    (idx : Idx (Γ ++ (ss ++ [τ])) s) (idx' : Idx (Γ ++ ss) s) (hv : idx.i.val = idx'.i.val) :
    CtxVec.get (Γ := Γ ++ (ss ++ [τ])) (s := s) idx
        (evalVec (Γ := Γ) (ss := ss ++ [τ]) (.snoc g node) d xV) =
      CtxVec.get (Γ := Γ ++ ss) (s := s) idx' (evalVec (Γ := Γ) (ss := ss) g d xV) := by
  apply PiLp.ext
  intro j
  rw [CtxVec.get_ofLp, CtxVec.get_ofLp]
  simp only [evalVec, CtxVec.cast_ofLp]
  have hoff :
      CtxVec.blockOffset (Γ ++ (ss ++ [τ])) idx.i.val =
        CtxVec.blockOffset (Γ ++ ss) idx'.i.val := by
    rw [hv, ← List.append_assoc]
    exact CtxVec.blockOffset_append_of_lt _ _ _ idx'.i.isLt
  rw [CtxVec.snoc_ofLp_of_lt _ _ _ (by simpa [hoff] using CtxVec.blockOffset_add_lt' idx' j)]
  exact congrArg _ (Fin.ext (by simp [hoff]))

/-- Read through the weakened index `Idx.weaken idx ss`, an input block is unchanged by evaluating
any graph. -/
theorem get_evalVec_weaken {Δ : Type} {Γ : List Shape} {s : Shape} (idx : Idx Γ s) :
    ∀ {ss : List Shape} (g : GraphData ℝ Δ Γ ss) (d : Δ) (xV : CtxVec Γ),
      CtxVec.get (Γ := Γ ++ ss) (s := s) (Idx.weaken idx ss)
          (evalVec (Γ := Γ) (ss := ss) g d xV) =
        CtxVec.get (Γ := Γ) (s := s) idx xV
  | _, .nil, d, xV => get_evalVec_nil d xV _ idx rfl
  | _, .snoc g node, d, xV => by
      rw [get_evalVec_snoc_of_lt g node d xV _ (Idx.weaken idx _) rfl]
      exact get_evalVec_weaken idx g d xV

/-- An index into the evaluated context at the position of an input index `idx'` reads the input
block of `idx'`, whatever the graph. -/
theorem get_evalVec_input {Δ : Type} {Γ ss : List Shape} {s : Shape}
    (g : GraphData ℝ Δ Γ ss) (d : Δ) (xV : CtxVec Γ)
    (idx : Idx (Γ ++ ss) s) (idx' : Idx Γ s) (hv : idx.i.val = idx'.i.val) :
    CtxVec.get (Γ := Γ ++ ss) (s := s) idx (evalVec (Γ := Γ) (ss := ss) g d xV) =
      CtxVec.get (Γ := Γ) (s := s) idx' xV := by
  have hidx : idx = Idx.weaken idx' ss := Idx.ext (Fin.ext hv)
  rw [hidx]
  exact get_evalVec_weaken idx' g d xV

end GraphData

end Tape

end

end TorchLean
