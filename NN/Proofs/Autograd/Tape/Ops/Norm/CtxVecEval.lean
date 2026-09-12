/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Proofs.Autograd.Tape.Util.Idx
public import NN.Proofs.Autograd.Tape.Nodes.Context

/-!
# Reading intermediate values out of an evaluated tape graph

`Graph.evalVec` returns the whole flattened context, inputs followed by every saved intermediate.
The pointwise normalization proofs need to know what a specific block of that vector is: the
variance block of the LayerNorm prefix, for instance, must be shown nonnegative before `sqrt` and
`inv` can be differentiated.

This file gives the coordinate description of `CtxVec.get` and the three rules that let a proof
walk back through a `snoc` chain:

* `Graph.get_evalVec_snoc_last`: the block just appended is the node's forward value;
* `Graph.get_evalVec_snoc_of_lt`: any earlier block is unchanged by appending a node;
* `Graph.get_evalVec_input`: an input block is unchanged by the whole graph.
-/

@[expose] public section

namespace Proofs
namespace Autograd

open Spec TorchLean

noncomputable section

namespace Idx

/-- Two typed indices with the same position are equal. -/
theorem ext' {Γ : List Shape} {s : Shape} {a b : Idx Γ s} (h : a.i = b.i) : a = b := by
  cases a
  cases b
  cases h
  rfl

end Idx

namespace CtxVec

/-- Offset of the block at list position `k` inside the flattened context vector. -/
def blockOffset : List Shape → Nat → Nat
  | [], _ => 0
  | _ :: _, 0 => 0
  | s :: ss, k + 1 => Spec.Shape.size s + blockOffset ss k

/-- Every coordinate of a block lies inside the flattened context. -/
theorem blockOffset_add_lt :
    ∀ {Γ : List Shape} (i : Fin Γ.length) (j : Fin (Spec.Shape.size (Γ.get i))),
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
      (j : Fin (Spec.Shape.size (Γ.get i))),
      (getBlock (Γ := Γ) i x).ofLp j =
        x.ofLp ⟨blockOffset Γ i.val + j.val, blockOffset_add_lt i j⟩
  | [], i, _, _ => nomatch i
  | s :: ss, ⟨0, _⟩, x, j => by
      simp only [getBlock, vecOfFun_ofLp]
      exact congrArg x.ofLp (Fin.ext (by simp [blockOffset]))
  | s :: ss, ⟨k + 1, hk⟩, x, j => by
      simp only [getBlock]
      rw [getBlock_ofLp (Γ := ss) ⟨k, Nat.lt_of_succ_lt_succ hk⟩ _ j]
      simp only [vecOfFun_ofLp]
      exact congrArg x.ofLp (Fin.ext (by simp [blockOffset, Nat.add_assoc]))

/-- Every coordinate of the block selected by a typed index lies inside the context. -/
theorem blockOffset_add_lt' {Γ : List Shape} {s : Shape} (idx : Idx Γ s)
    (j : Fin (Spec.Shape.size s)) :
    blockOffset Γ idx.i.val + j.val < ctxSize Γ := by
  have h := blockOffset_add_lt idx.i (Fin.cast (congrArg Spec.Shape.size idx.h).symm j)
  simpa using h

/-- Coordinate description of `CtxVec.get`. -/
theorem get_ofLp {Γ : List Shape} {s : Shape} (idx : Idx Γ s) (x : CtxVec Γ)
    (j : Fin (Spec.Shape.size s)) :
    (get (Γ := Γ) (s := s) idx x).ofLp j =
      x.ofLp ⟨blockOffset Γ idx.i.val + j.val, blockOffset_add_lt' idx j⟩ := by
  simp only [get, castVec_ofLp]
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
theorem getBlock_flattenCtx_zero {s : Shape} {ss : List Shape} (X : Tensor ℝ s)
    (xs : TorchLean.TensorPack ℝ ss) (h : 0 < (s :: ss).length) :
    getBlock (Γ := s :: ss) ⟨0, h⟩ (flattenCtx (Γ := s :: ss) (.cons X xs)) = tensorToVec X := by
  apply PiLp.ext
  intro j
  simp [getBlock, flattenCtx_cons]

/-- Later blocks of a flattened tensor pack are blocks of the tail. -/
theorem getBlock_flattenCtx_succ {s : Shape} {ss : List Shape} (X : Tensor ℝ s)
    (xs : TorchLean.TensorPack ℝ ss) (k : Nat) (hk : k + 1 < (s :: ss).length) :
    getBlock (Γ := s :: ss) ⟨k + 1, hk⟩ (flattenCtx (Γ := s :: ss) (.cons X xs)) =
      getBlock (Γ := ss) ⟨k, Nat.lt_of_succ_lt_succ hk⟩ (flattenCtx (Γ := ss) xs) := by
  simp only [getBlock]
  congr 1
  apply PiLp.ext
  intro j
  simp [flattenCtx_cons]

end CtxVec

/-- Coordinates of `castCtxVec` are coordinates of the original vector. -/
theorem castCtxVec_ofLp {Γ₁ Γ₂ : List Shape} (h : Γ₁ = Γ₂) (x : CtxVec Γ₁)
    (k : Fin (ctxSize Γ₂)) :
    (castCtxVec (Γ₁ := Γ₁) (Γ₂ := Γ₂) h x).ofLp k =
      x.ofLp (Fin.cast (congrArg ctxSize h).symm k) := by
  simp [castCtxVec]

/-- Coordinates of `snocCtx` below the prefix size read the prefix. -/
theorem snocCtx_ofLp_of_lt {Γ : List Shape} {τ : Shape} (ctx : CtxVec Γ)
    (t : Vec (Spec.Shape.size τ)) (k : Fin (ctxSize (Γ ++ [τ]))) (hk : k.val < ctxSize Γ) :
    (snocCtx (Γ := Γ) (τ := τ) ctx t).ofLp k = ctx.ofLp ⟨k.val, hk⟩ := by
  simp only [snocCtx, castVec_ofLp, appendVec, vecOfFun_ofLp]
  have hcast :
      Fin.cast (ctxSize_snoc Γ τ).symm.symm k = Fin.castAdd (Spec.Shape.size τ) ⟨k.val, hk⟩ := by
    ext
    simp
  rw [hcast, Fin.append_left]

/-- Coordinates of `snocCtx` at or above the prefix size read the appended block. -/
theorem snocCtx_ofLp_last {Γ : List Shape} {τ : Shape} (ctx : CtxVec Γ)
    (t : Vec (Spec.Shape.size τ)) (k : Fin (ctxSize (Γ ++ [τ]))) (j : Fin (Spec.Shape.size τ))
    (hk : k.val = ctxSize Γ + j.val) :
    (snocCtx (Γ := Γ) (τ := τ) ctx t).ofLp k = t.ofLp j := by
  simp only [snocCtx, castVec_ofLp, appendVec, vecOfFun_ofLp]
  have hcast : Fin.cast (ctxSize_snoc Γ τ).symm.symm k = Fin.natAdd (ctxSize Γ) j := by
    ext
    simp [hk]
  rw [hcast, Fin.append_right]

namespace Graph

/-- Evaluating the empty graph leaves every block in place. -/
theorem get_evalVec_nil {Γ : List Shape} {s : Shape} (xV : CtxVec Γ)
    (idx : Idx (Γ ++ []) s) (idx' : Idx Γ s) (hv : idx.i.val = idx'.i.val) :
    CtxVec.get (Γ := Γ ++ []) (s := s) idx (evalVec (Γ := Γ) (ss := []) .nil xV) =
      CtxVec.get (Γ := Γ) (s := s) idx' xV := by
  apply PiLp.ext
  intro j
  rw [CtxVec.get_ofLp, CtxVec.get_ofLp]
  simp only [evalVec, castCtxVec_ofLp]
  exact congrArg xV.ofLp (Fin.ext (by simp [hv]))

/-- The block appended by `snoc` is the node's forward value on the prefix evaluation. -/
theorem get_evalVec_snoc_last {Γ ss : List Shape} {τ : Shape}
    (g : Graph Γ ss) (node : Node (Γ ++ ss) τ) (xV : CtxVec Γ)
    (idx : Idx (Γ ++ (ss ++ [τ])) τ) (hv : idx.i.val = (Γ ++ ss).length) :
    CtxVec.get (Γ := Γ ++ (ss ++ [τ])) (s := τ) idx
        (evalVec (Γ := Γ) (ss := ss ++ [τ]) (.snoc g node) xV) =
      node.forwardVec (Γ := Γ ++ ss) (τ := τ) (evalVec (Γ := Γ) (ss := ss) g xV) := by
  apply PiLp.ext
  intro j
  rw [CtxVec.get_ofLp]
  simp only [evalVec, castCtxVec_ofLp]
  rw [snocCtx_ofLp_last _ _ _ j]
  simp only [Fin.val_cast]
  rw [hv, ← List.append_assoc, CtxVec.blockOffset_append_length]

/-- Blocks below the appended one are unchanged by `snoc`. -/
theorem get_evalVec_snoc_of_lt {Γ ss : List Shape} {τ s : Shape}
    (g : Graph Γ ss) (node : Node (Γ ++ ss) τ) (xV : CtxVec Γ)
    (idx : Idx (Γ ++ (ss ++ [τ])) s) (idx' : Idx (Γ ++ ss) s) (hv : idx.i.val = idx'.i.val) :
    CtxVec.get (Γ := Γ ++ (ss ++ [τ])) (s := s) idx
        (evalVec (Γ := Γ) (ss := ss ++ [τ]) (.snoc g node) xV) =
      CtxVec.get (Γ := Γ ++ ss) (s := s) idx' (evalVec (Γ := Γ) (ss := ss) g xV) := by
  apply PiLp.ext
  intro j
  rw [CtxVec.get_ofLp, CtxVec.get_ofLp]
  simp only [evalVec, castCtxVec_ofLp]
  have hoff :
      CtxVec.blockOffset (Γ ++ (ss ++ [τ])) idx.i.val =
        CtxVec.blockOffset (Γ ++ ss) idx'.i.val := by
    rw [hv, ← List.append_assoc]
    exact CtxVec.blockOffset_append_of_lt _ _ _ idx'.i.isLt
  rw [snocCtx_ofLp_of_lt _ _ _ (by simpa [hoff] using CtxVec.blockOffset_add_lt' idx' j)]
  exact congrArg _ (Fin.ext (by simp [hoff]))

/-- Input blocks are unchanged by evaluating any graph. -/
theorem get_evalVec_weaken {Γ : List Shape} {s : Shape} (idx : Idx Γ s) :
    ∀ {ss : List Shape} (g : Graph Γ ss) (xV : CtxVec Γ),
      CtxVec.get (Γ := Γ ++ ss) (s := s) (Idx.weaken idx ss) (evalVec (Γ := Γ) (ss := ss) g xV) =
        CtxVec.get (Γ := Γ) (s := s) idx xV
  | _, .nil, xV => get_evalVec_nil xV _ idx rfl
  | _, .snoc g node, xV => by
      rw [get_evalVec_snoc_of_lt g node xV _ (Idx.weaken idx _) rfl]
      exact get_evalVec_weaken idx g xV

/-- Input blocks are unchanged by evaluating any graph, for any index into the input prefix. -/
theorem get_evalVec_input {Γ ss : List Shape} {s : Shape} (g : Graph Γ ss) (xV : CtxVec Γ)
    (idx : Idx (Γ ++ ss) s) (idx' : Idx Γ s) (hv : idx.i.val = idx'.i.val) :
    CtxVec.get (Γ := Γ ++ ss) (s := s) idx (evalVec (Γ := Γ) (ss := ss) g xV) =
      CtxVec.get (Γ := Γ) (s := s) idx' xV := by
  have hidx : idx = Idx.weaken idx' ss := Idx.ext' (Fin.ext hv)
  rw [hidx]
  exact get_evalVec_weaken idx' g xV

end Graph

end

end Autograd
end Proofs
