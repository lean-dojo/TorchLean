/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.MLTheory.CROWN.Proofs.GraphCertSoundness
public import NN.MLTheory.CROWN.Proofs.GraphCrownCertSoundness
public import NN.MLTheory.CROWN.Proofs.AlphaBetaReLUScalarSoundness
public import NN.Proofs.Tensor.Basic

/-!
# Shared α-CROWN Transfer Lemmas

Common definitions and local proof lemmas for the graph-dialect α-CROWN and α/β-CROWN transfer
rules over `ℝ`.  These facts connect affine bounds, IBP boxes, ReLU relaxations, dimension casts,
and pointwise graph semantics.
-/

@[expose] public section


namespace NN.MLTheory.CROWN.Graph

open _root_.Spec _root_.TorchLean
open _root_.TorchLean.Tensor
open scoped BigOperators
open Proofs.TensorAlgebra

open NN.MLTheory.CROWN
open NN.MLTheory.CROWN.Cert
open NN.MLTheory.CROWN.Proofs

namespace AlphaCrownTransferSoundness

noncomputable section

open CrownCertSoundness
open CertSoundness

-- The graph dialect’s `FlatBox` is a dependent record (the tensor shapes depend on `dim`),
-- so Lean does not automatically register a usable extensionality lemma for the `ext` tactic.
-- We add a small `[ext]` lemma locally.
/-- Extensionality for `FlatBox`: equal dimension and heterogeneously equal endpoints. -/
@[ext] theorem FlatBox.ext' {α : Type} [TorchLean.Storage α] [Context α] {B1 B2 : FlatBox α}
    (hDim : B1.dim = B2.dim)
    (hLo : HEq B1.lo B2.lo)
    (hHi : HEq B1.hi B2.hi) : B1 = B2 := by
  cases B1
  cases B2
  cases hDim
  cases hLo
  cases hHi
  rfl

/-! ## Helper assumptions -/

/-- The designated input entry in `inputs` matches the concrete point `x` (up to a `castDimScalar`).
  -/
def InputsMatch (inputs : Std.HashMap Nat Val) (ctx : AffineCtx)
    (x : Tensor ℝ [ctx.inputDim]) : Prop :=
  ∃ v : Val,
    inputs[ctx.inputId]? = some v ∧
    ∃ h : v.n = ctx.inputDim,
      castDimScalar (α := ℝ) (n := v.n) (n' := ctx.inputDim) h v.v = x

/-- Pointwise: whenever both arrays contain entries at `id`, the IBP box encloses the semantic
  value. -/
def IBPEnclosesVals (ibp : Array (Option (FlatBox ℝ))) (vals : Array (Option Val)) : Prop :=
  ∀ id : Nat, id < vals.size →
    match ibp[id]!, vals[id]! with
    | some B, some v => CertSoundness.EnclosesBox B v
    | _, _ => True

/-- Well-formedness condition for α vectors: each component lies in `[0,1]`. -/
def AlphaOK (alpha : Array (Option (FlatTensor ℝ))) : Prop :=
  ∀ id : Nat, id < alpha.size →
    match alpha[id]! with
    | none => True
    | some a => ∀ i : Fin a.n, (0 : ℝ) ≤ getScalar a.v i ∧ getScalar a.v i ≤ (1 : ℝ)

/-! ## `Theorems.Semantics.encloses` ↔ componentwise inequalities (via `getScalar`) -/

theorem encloses_iff_getScalar {n : Nat}
    (lo hi x : Tensor ℝ [n]) :
    Theorems.Semantics.encloses (α := ℝ) { dim := n, lo := lo, hi := hi } x ↔
      ∀ i : Fin n, getScalar lo i ≤ getScalar x i ∧ getScalar x i ≤ getScalar hi i := by
  rfl

/-! ## Small tensor algebra helpers -/

theorem add_spec_full_zero_right {n : Nat}
    (t : Tensor ℝ [n]) :
    Tensor.addSpec (α := ℝ) t (Tensor.full (α := ℝ) (.dim n .scalar) (0 : ℝ)) = t := by
  apply Tensor.ext_vector
  intro i
  simp [Tensor.addSpec]

/-- A linear layer with zero bias is a plain matrix-vector product.

The certificate builders emit zero biases for the layers that have none, so without this the
transfer proofs would carry a `+ 0` through every step. -/
theorem linear_spec_bias_zero_eq_matvec {m n : Nat}
    (W : Tensor ℝ [m, n])
    (x : Tensor ℝ [n]) :
    Spec.linearSpec (α := ℝ)
        { weights := W
          bias := Tensor.full (α := ℝ) (.dim m .scalar) (0 : ℝ) } x
      =
      Spec.matVecMulSpec (α := ℝ) W x := by
  -- `linear_spec` is `mat_vec_mul + bias`, so the zero bias disappears.
  simp [Spec.linearSpec]

/-! ## Small cast lemmas (avoid `cases` on equalities mentioning record fields) -/

/-!
`castDimScalar_trans` comes from `CertSoundness` (opened above) and `castDimScalar_self` from the
engine module where the cast is defined; this file already depends on both for the rest of the
interval lemmas. Each of them was stated a second time here, word for word, before this note
replaced them.
-/
/-- `castDimScalar` is proof-irrelevant in its equality argument. -/
theorem castDimScalar_proof_irrel {n n' : Nat}
    (h₁ h₂ : n = n') (t : Tensor ℝ [n]) :
    castDimScalar (α := ℝ) h₁ t = castDimScalar (α := ℝ) h₂ t := by
  have : h₁ = h₂ := Subsingleton.elim _ _
  cases this
  rfl

/-- `getScalar` commutes with `castDimScalar` (up to `Fin.cast`). -/
theorem getScalar_castDimScalar {n n' : Nat} (h : n = n') (t : Tensor ℝ [n]) (i : Fin n') :
    getScalar (castDimScalar (α := ℝ) (n := n) (n' := n') h t) i =
      getScalar t (Fin.cast h.symm i) := by
  cases h
  simp [castDimScalar]

/-- `Activation.reluSpec` commutes with `castDimScalar`. -/
theorem relu_spec_castDimScalar {n n' : Nat} (h : n = n') (t : Tensor ℝ [n]) :
    castDimScalar (α := ℝ) (n := n) (n' := n') h (Activation.reluSpec (α := ℝ) t)
      =
    Activation.reluSpec (α := ℝ) (castDimScalar (α := ℝ) (n := n) (n' := n') h t) := by
  cases h
  rfl

/-- A small `matVecMulSpec` cast lemma used for single-row “sum” encodings. -/
theorem mat_vec_mul_full_one_castDimScalar {n n' : Nat} (h : n = n') (v : Tensor ℝ [n]) :
    Spec.matVecMulSpec (α := ℝ)
        (Tensor.full (α := ℝ) (.dim 1 (.dim n .scalar)) (1 : ℝ)) v
      =
    Spec.matVecMulSpec (α := ℝ)
        (Tensor.full (α := ℝ) (.dim 1 (.dim n' .scalar)) (1 : ℝ))
        (castDimScalar (α := ℝ) (n := n) (n' := n') h v) := by
  cases h
  simp [castDimScalar]

/-- `affineEvalAt` commutes with casting the output dimension of an affine form. -/
theorem affineEvalAt_castAffineOut {inDim outDim outDim' : Nat}
    (h : outDim = outDim') (aff : AffineVec ℝ inDim outDim) (x : Tensor ℝ [inDim]) :
    CrownCertSoundness.affineEvalAt (α := ℝ) (inDim := inDim) (outDim := outDim')
        (NN.MLTheory.CROWN.Graph.castAffineOut (α := ℝ) (n := inDim) (m := outDim) (m' := outDim') h
          aff) x
      =
      castDimScalar (α := ℝ) h
        (CrownCertSoundness.affineEvalAt (α := ℝ) (inDim := inDim) (outDim := outDim) aff x) := by
  cases h
  rfl

/-- `boundsEvalAt` commutes with casting the output dimension of affine bounds. -/
theorem boundsEvalAt_castAffineOut (xin : FlatAffineBounds ℝ) {outDim' : Nat}
    (h : xin.outDim = outDim') (x : Tensor ℝ [xin.inDim]) :
    CrownCertSoundness.boundsEvalAt (α := ℝ)
        { inDim := xin.inDim
          outDim := outDim'
          loAff := NN.MLTheory.CROWN.Graph.castAffineOut (α := ℝ) (n := xin.inDim) (m := xin.outDim)
            (m' := outDim') h xin.loAff
          hiAff := NN.MLTheory.CROWN.Graph.castAffineOut (α := ℝ) (n := xin.inDim) (m := xin.outDim)
            (m' := outDim') h xin.hiAff } x
      =
      { dim := outDim'
        lo := castDimScalar (α := ℝ) h (CrownCertSoundness.boundsEvalAt (α := ℝ) xin x).lo
        hi := castDimScalar (α := ℝ) h (CrownCertSoundness.boundsEvalAt (α := ℝ) xin x).hi } := by
  cases h
  rfl

/-- `Semantics.encloses` is preserved under casting a box and point to an equal dimension. -/
theorem sem_encloses_castDim {B : FlatBox ℝ} {n' : Nat}
    (h : B.dim = n') (x : Tensor ℝ [B.dim]) :
    Theorems.Semantics.encloses (α := ℝ) B x →
      Theorems.Semantics.encloses (α := ℝ)
        { dim := n'
          lo := castDimScalar (α := ℝ) h B.lo
          hi := castDimScalar (α := ℝ) h B.hi }
        (castDimScalar (α := ℝ) h x) := by
  intro hx
  cases B with
  | mk n lo hi =>
      cases h
      simpa [Theorems.Semantics.encloses, castDimScalar, getDimScalarFn] using hx

/-- `Semantics.encloses` respects definitional equality of boxes. -/
theorem sem_encloses_of_eq {B1 B2 : FlatBox ℝ}
    (h : B1 = B2) (x : Tensor ℝ [B1.dim]) :
    Theorems.Semantics.encloses (α := ℝ) B1 x →
      Theorems.Semantics.encloses (α := ℝ) B2
        (castDimScalar (α := ℝ) (congrArg FlatBox.dim h) x) := by
  intro hx
  cases h
  simpa [castDimScalar] using hx

/-- `Semantics.encloses` respects definitional equality of values. -/
theorem sem_encloses_value_eq {B : FlatBox ℝ}
    {x y : Tensor ℝ [B.dim]} (hxy : x = y) :
    Theorems.Semantics.encloses (α := ℝ) B x →
      Theorems.Semantics.encloses (α := ℝ) B y := by
  intro hx
  cases hxy
  simpa using hx

/-- `EnclosesAtInput` is preserved under casting the output dimension of bounds and value payloads.
  -/
theorem enclosesAtInput_castOut (ctx : AffineCtx) (x : Tensor ℝ [ctx.inputDim])
    (xin : FlatAffineBounds ℝ) (vp : FlatTensor ℝ) {outDim' : Nat}
    (hout : xin.outDim = outDim') (hvout : vp.n = outDim') :
    CrownCertSoundness.EnclosesAtInput (α := ℝ) ctx x xin vp →
      CrownCertSoundness.EnclosesAtInput (α := ℝ) ctx x
        { inDim := xin.inDim
          outDim := outDim'
          loAff := NN.MLTheory.CROWN.Graph.castAffineOut (α := ℝ) (n := xin.inDim) (m := xin.outDim)
            (m' := outDim') hout xin.loAff
          hiAff := NN.MLTheory.CROWN.Graph.castAffineOut (α := ℝ) (n := xin.inDim) (m := xin.outDim)
            (m' := outDim') hout xin.hiAff }
        { n := outDim', v := castDimScalar (α := ℝ) hvout vp.v } := by
  intro hpar
  rcases hpar with ⟨hinDim, hvec⟩
  refine ⟨hinDim, ?_⟩
  -- The `x'` used to evaluate `xin` and the casted bound is the same, since `inDim` is unchanged.
  dsimp
  rcases hvec with ⟨hdim, henc⟩
  -- Cast the enclosure result from `xin.outDim` to `outDim'` via `hout`.
  have hencCast :=
    (sem_encloses_castDim (B := CrownCertSoundness.boundsEvalAt (α := ℝ) xin (castDimScalar (α :=
      ℝ) hinDim.symm x))
      (h := hout) (x := castDimScalar (α := ℝ) hdim.symm vp.v) henc)
  -- Simplify the RHS cast: `hout ∘ hdim.symm` is a proof of `vp.n = outDim'`, so it matches
  -- `hvout`.
  have hvout' : Eq.trans hdim.symm hout = hvout := by
    exact Subsingleton.elim _ _
  have hxCast :
      castDimScalar (α := ℝ) hout (castDimScalar (α := ℝ) hdim.symm vp.v)
        = castDimScalar (α := ℝ) hvout vp.v := by
    calc
      castDimScalar (α := ℝ) hout (castDimScalar (α := ℝ) hdim.symm vp.v)
          = castDimScalar (α := ℝ) (Eq.trans hdim.symm hout) vp.v := by
              exact (castDimScalar_trans (h₁ := hdim.symm) (h₂ := hout) (t := vp.v)).symm
      _ = castDimScalar (α := ℝ) hvout vp.v := by
              exact castDimScalar_proof_irrel (Eq.trans hdim.symm hout) hvout vp.v
  -- Rewrite the enclosure `henc'` to target `castDimScalar hvout vp.v`.
  have henc1 :
      Theorems.Semantics.encloses (α := ℝ)
        { dim := outDim'
          lo := castDimScalar (α := ℝ) hout (CrownCertSoundness.boundsEvalAt (α := ℝ) xin
            (castDimScalar (α := ℝ) hinDim.symm x)).lo
          hi := castDimScalar (α := ℝ) hout (CrownCertSoundness.boundsEvalAt (α := ℝ) xin
            (castDimScalar (α := ℝ) hinDim.symm x)).hi }
        (castDimScalar (α := ℝ) hvout vp.v) := by
    exact sem_encloses_value_eq hxCast hencCast

  -- Avoid rewriting dependent `boundsEvalAt` equalities: transfer componentwise between the
  -- explicit cast box
  -- and `boundsEvalAt` of the casted affine bound.
  let x0 : Tensor ℝ [xin.inDim] :=
    castDimScalar (α := ℝ) (n := ctx.inputDim) (n' := xin.inDim) hinDim.symm x
  let B1 : FlatBox ℝ :=
    boundsEvalAt (α := ℝ)
      { inDim := xin.inDim
        outDim := outDim'
        loAff := NN.MLTheory.CROWN.Graph.castAffineOut (α := ℝ) (n := xin.inDim) (m := xin.outDim)
          (m' := outDim') hout xin.loAff
        hiAff := NN.MLTheory.CROWN.Graph.castAffineOut (α := ℝ) (n := xin.inDim) (m := xin.outDim)
          (m' := outDim') hout xin.hiAff } x0
  let B2 : FlatBox ℝ :=
    { dim := outDim'
      lo := castDimScalar (α := ℝ) hout (boundsEvalAt (α := ℝ) xin x0).lo
      hi := castDimScalar (α := ℝ) hout (boundsEvalAt (α := ℝ) xin x0).hi }

  have hB2 : Theorems.Semantics.encloses (α := ℝ) B2 (castDimScalar (α := ℝ) hvout vp.v) := by
    simpa [B2, x0] using henc1

  have hlo : B1.lo = B2.lo := by
    -- `B1.lo` is `affineEvalAt` of the casted affine map; `B2.lo` is the cast of the original
    -- `boundsEvalAt` lower.
    simpa [B1, B2, x0, CrownCertSoundness.boundsEvalAt, CrownCertSoundness.affineEvalAt] using
      (affineEvalAt_castAffineOut (h := hout) (aff := xin.loAff) (x := x0))
  have hhi : B1.hi = B2.hi := by
    simpa [B1, B2, x0, CrownCertSoundness.boundsEvalAt, CrownCertSoundness.affineEvalAt] using
      (affineEvalAt_castAffineOut (h := hout) (aff := xin.hiAff) (x := x0))

  have hB1 : Theorems.Semantics.encloses (α := ℝ) B1 (castDimScalar (α := ℝ) hvout vp.v) := by
    have hcomp :=
      (encloses_iff_getScalar (n := outDim') (lo := B2.lo) (hi := B2.hi)
        (x := castDimScalar (α := ℝ) hvout vp.v)).1 hB2
    refine (encloses_iff_getScalar (n := outDim') (lo := B1.lo) (hi := B1.hi)
      (x := castDimScalar (α := ℝ) hvout vp.v)).2 ?_
    intro i
    have hi := hcomp i
    constructor
    · simpa [hlo] using hi.1
    · simpa [hhi] using hi.2

  -- Finish by packaging as `EnclosesVec` (the outer cast is definitional).
  refine ⟨rfl, ?_⟩
  simpa [B1, x0, castDimScalar] using hB1

/-! ## Matrix sign-splitting bound (pointwise, over `ℝ`) -/

theorem get2_mat_pos {m n : Nat}
    (W : Tensor ℝ [m, n]) (i : Fin m) (j : Fin n) :
    Spec.get2 (NN.MLTheory.CROWN.IBP.matPos (α := ℝ) (m := m) (n := n) W) i j =
      (if Spec.get2 W i j > 0 then Spec.get2 W i j else 0) := by
  simp [NN.MLTheory.CROWN.IBP.matPos]

/-- Entries of the negative part of a matrix: the entry where it is not positive, else zero. -/
theorem get2_mat_neg {m n : Nat}
    (W : Tensor ℝ [m, n]) (i : Fin m) (j : Fin n) :
    Spec.get2 (NN.MLTheory.CROWN.IBP.matNeg (α := ℝ) (m := m) (n := n) W) i j =
      (if Spec.get2 W i j > 0 then 0 else Spec.get2 W i j) := by
  simp [NN.MLTheory.CROWN.IBP.matNeg]

/-- One term of the interval upper bound: `w * x` is at most `w⁺ * u + w⁻ * l`.

This is the whole idea of sign splitting. A positive weight is maximized at the upper endpoint and a
negative one at the lower endpoint, so pairing each sign with the right endpoint is what makes the
matrix version below a coordinatewise consequence. -/
theorem signSplit_term_upper (w l u x : ℝ) (hlx : l ≤ x) (hxu : x ≤ u) :
    w * x ≤ (if 0 < w then w else 0) * u + (if 0 < w then 0 else w) * l := by
  by_cases hw : 0 < w
  · have hw0 : 0 ≤ w := le_of_lt hw
    have : w * x ≤ w * u := mul_le_mul_of_nonneg_left hxu hw0
    simpa [hw, add_assoc, add_left_comm, add_comm] using this
  · have hw0 : w ≤ 0 := le_of_not_gt hw
    have : w * x ≤ w * l := mul_le_mul_of_nonpos_left hlx hw0
    simpa [hw, add_assoc, add_left_comm, add_comm] using this

/-- The lower-bound companion: `w⁺ * l + w⁻ * u` is at most `w * x`. -/
theorem signSplit_term_lower (w l u x : ℝ) (hlx : l ≤ x) (hxu : x ≤ u) :
    (if 0 < w then w else 0) * l + (if 0 < w then 0 else w) * u ≤ w * x := by
  by_cases hw : 0 < w
  · have hw0 : 0 ≤ w := le_of_lt hw
    have : w * l ≤ w * x := mul_le_mul_of_nonneg_left hlx hw0
    simpa [hw, add_assoc, add_left_comm, add_comm] using this
  · have hw0 : w ≤ 0 := le_of_not_gt hw
    have : w * u ≤ w * x := mul_le_mul_of_nonpos_left hxu hw0
    simpa [hw, add_assoc, add_left_comm, add_comm] using this

/-- Interval bound propagation through an affine layer is sound.

The positive and negative parts of `W` are applied to opposite endpoints, so the resulting box
encloses `W x + b` for every `x` in the input box. This is plain IBP; α-CROWN uses it for the
intermediate boxes that its ReLU relaxations are built from. -/
theorem encloses_linear_signSplit {m n : Nat}
    (W : Tensor ℝ [m, n])
    (b : Tensor ℝ [m])
    (lo hi x : Tensor ℝ [n])
    (hx : Theorems.Semantics.encloses (α := ℝ) { dim := n, lo := lo, hi := hi } x) :
    Theorems.Semantics.encloses (α := ℝ)
      { dim := m
        lo :=
          Tensor.addSpec (α := ℝ)
            (Tensor.addSpec (α := ℝ)
              (Spec.matVecMulSpec (α := ℝ) (NN.MLTheory.CROWN.IBP.matPos (α := ℝ) (m := m) (n :=
                n) W) lo)
              (Spec.matVecMulSpec (α := ℝ) (NN.MLTheory.CROWN.IBP.matNeg (α := ℝ) (m := m) (n :=
                n) W) hi))
            b
        hi :=
          Tensor.addSpec (α := ℝ)
            (Tensor.addSpec (α := ℝ)
              (Spec.matVecMulSpec (α := ℝ) (NN.MLTheory.CROWN.IBP.matPos (α := ℝ) (m := m) (n :=
                n) W) hi)
              (Spec.matVecMulSpec (α := ℝ) (NN.MLTheory.CROWN.IBP.matNeg (α := ℝ) (m := m) (n :=
                n) W) lo))
            b }
      (Tensor.addSpec (α := ℝ) (Spec.matVecMulSpec (α := ℝ) W x) b) := by
  classical
  have hx' := (encloses_iff_getScalar (lo := lo) (hi := hi) (x := x)).1 hx
  refine (encloses_iff_getScalar (n := m) (lo := _) (hi := _) (x := _)).2 ?_
  intro i
  -- Expand all mat-vec products into finite sums.
  have hW :
      TorchLean.Tensor.getScalar (Spec.matVecMulSpec (α := ℝ) W x) i =
        ∑ k : Fin n, (Spec.get2 W i k) * (TorchLean.Tensor.getScalar x k) := by
    simpa using (Proofs.TensorAlgebra.getScalar_mat_vec_mul_spec (A := W) (v := x) (i := i))
  have hPos_lo :
      TorchLean.Tensor.getScalar (Spec.matVecMulSpec (α := ℝ)
          (NN.MLTheory.CROWN.IBP.matPos (α := ℝ) (m := m) (n := n) W) lo) i =
        ∑ k : Fin n,
          (Spec.get2 (NN.MLTheory.CROWN.IBP.matPos (α := ℝ) (m := m) (n := n) W) i k) *
            (TorchLean.Tensor.getScalar lo k) := by
    simpa using
      (Proofs.TensorAlgebra.getScalar_mat_vec_mul_spec
        (A := NN.MLTheory.CROWN.IBP.matPos (α := ℝ) (m := m) (n := n) W) (v := lo) (i := i))
  have hNeg_hi :
      TorchLean.Tensor.getScalar (Spec.matVecMulSpec (α := ℝ)
          (NN.MLTheory.CROWN.IBP.matNeg (α := ℝ) (m := m) (n := n) W) hi) i =
        ∑ k : Fin n,
          (Spec.get2 (NN.MLTheory.CROWN.IBP.matNeg (α := ℝ) (m := m) (n := n) W) i k) *
            (TorchLean.Tensor.getScalar hi k) := by
    simpa using
      (Proofs.TensorAlgebra.getScalar_mat_vec_mul_spec
        (A := NN.MLTheory.CROWN.IBP.matNeg (α := ℝ) (m := m) (n := n) W) (v := hi) (i := i))
  have hPos_hi :
      TorchLean.Tensor.getScalar (Spec.matVecMulSpec (α := ℝ)
          (NN.MLTheory.CROWN.IBP.matPos (α := ℝ) (m := m) (n := n) W) hi) i =
        ∑ k : Fin n,
          (Spec.get2 (NN.MLTheory.CROWN.IBP.matPos (α := ℝ) (m := m) (n := n) W) i k) *
            (TorchLean.Tensor.getScalar hi k) := by
    simpa using
      (Proofs.TensorAlgebra.getScalar_mat_vec_mul_spec
        (A := NN.MLTheory.CROWN.IBP.matPos (α := ℝ) (m := m) (n := n) W) (v := hi) (i := i))
  have hNeg_lo :
      TorchLean.Tensor.getScalar (Spec.matVecMulSpec (α := ℝ)
          (NN.MLTheory.CROWN.IBP.matNeg (α := ℝ) (m := m) (n := n) W) lo) i =
        ∑ k : Fin n,
          (Spec.get2 (NN.MLTheory.CROWN.IBP.matNeg (α := ℝ) (m := m) (n := n) W) i k) *
            (TorchLean.Tensor.getScalar lo k) := by
    simpa using
      (Proofs.TensorAlgebra.getScalar_mat_vec_mul_spec
        (A := NN.MLTheory.CROWN.IBP.matNeg (α := ℝ) (m := m) (n := n) W) (v := lo) (i := i))

  have hUpperTerm :
      ∀ k : Fin n,
        (Spec.get2 W i k) * (TorchLean.Tensor.getScalar x k) ≤
          (Spec.get2 (NN.MLTheory.CROWN.IBP.matPos (α := ℝ) (m := m) (n := n) W) i k) *
            (TorchLean.Tensor.getScalar hi k) +
          (Spec.get2 (NN.MLTheory.CROWN.IBP.matNeg (α := ℝ) (m := m) (n := n) W) i k) *
            (TorchLean.Tensor.getScalar lo k) := by
    intro k
    have hk := hx' k
    have hpos := get2_mat_pos (W := W) i k
    have hneg := get2_mat_neg (W := W) i k
    have := signSplit_term_upper (w := Spec.get2 W i k) (l := TorchLean.Tensor.getScalar lo k)
      (u := TorchLean.Tensor.getScalar hi k)
      (x := TorchLean.Tensor.getScalar x k) hk.1 hk.2
    simpa [hpos, hneg, mul_add, add_mul, add_assoc, add_left_comm, add_comm] using this

  have hLowerTerm :
      ∀ k : Fin n,
        (Spec.get2 (NN.MLTheory.CROWN.IBP.matPos (α := ℝ) (m := m) (n := n) W) i k) *
          (TorchLean.Tensor.getScalar lo k) +
          (Spec.get2 (NN.MLTheory.CROWN.IBP.matNeg (α := ℝ) (m := m) (n := n) W) i k) *
            (TorchLean.Tensor.getScalar hi k)
            ≤ (Spec.get2 W i k) * (TorchLean.Tensor.getScalar x k) := by
    intro k
    have hk := hx' k
    have hpos := get2_mat_pos (W := W) i k
    have hneg := get2_mat_neg (W := W) i k
    have := signSplit_term_lower (w := Spec.get2 W i k) (l := TorchLean.Tensor.getScalar lo k)
      (u := TorchLean.Tensor.getScalar hi k)
      (x := TorchLean.Tensor.getScalar x k) hk.1 hk.2
    simpa [hpos, hneg, mul_add, add_mul, add_assoc, add_left_comm, add_comm] using this

  have hUpperSum :
      (∑ k : Fin n, (Spec.get2 W i k) * (TorchLean.Tensor.getScalar x k)) ≤
        (∑ k : Fin n,
          ((Spec.get2 (NN.MLTheory.CROWN.IBP.matPos (α := ℝ) (m := m) (n := n) W) i k) *
            (TorchLean.Tensor.getScalar hi k) +
           (Spec.get2 (NN.MLTheory.CROWN.IBP.matNeg (α := ℝ) (m := m) (n := n) W) i k) *
             (TorchLean.Tensor.getScalar lo k))) := by
    classical
    simpa using (Finset.sum_le_sum (s := Finset.univ) (fun k _ => hUpperTerm k))

  have hLowerSum :
      (∑ k : Fin n,
        ((Spec.get2 (NN.MLTheory.CROWN.IBP.matPos (α := ℝ) (m := m) (n := n) W) i k) *
          (TorchLean.Tensor.getScalar lo k) +
         (Spec.get2 (NN.MLTheory.CROWN.IBP.matNeg (α := ℝ) (m := m) (n := n) W) i k) *
           (TorchLean.Tensor.getScalar hi k)))
          ≤ (∑ k : Fin n, (Spec.get2 W i k) * (TorchLean.Tensor.getScalar x k)) := by
    classical
    simpa using (Finset.sum_le_sum (s := Finset.univ) (fun k _ => hLowerTerm k))

  have hlo :
      TorchLean.Tensor.getScalar
          (Tensor.addSpec (α := ℝ)
            (Tensor.addSpec (α := ℝ)
              (Spec.matVecMulSpec (α := ℝ) (NN.MLTheory.CROWN.IBP.matPos (α := ℝ) (m := m) (n :=
                n) W) lo)
              (Spec.matVecMulSpec (α := ℝ) (NN.MLTheory.CROWN.IBP.matNeg (α := ℝ) (m := m) (n :=
                n) W) hi))
            b) i
        ≤
  TorchLean.Tensor.getScalar (Tensor.addSpec (α := ℝ) (Spec.matVecMulSpec (α := ℝ) W x) b) i := by
    -- Rewrite `sum (a+b)` into `sum a + sum b` to match `getScalar` expansions.
    have hLowerSum' :
        (∑ k : Fin n,
            (Spec.get2 (NN.MLTheory.CROWN.IBP.matPos (α := ℝ) (m := m) (n := n) W) i k) *
              (TorchLean.Tensor.getScalar lo k)) +
          (∑ k : Fin n,
            (Spec.get2 (NN.MLTheory.CROWN.IBP.matNeg (α := ℝ) (m := m) (n := n) W) i k) *
              (TorchLean.Tensor.getScalar hi k))
            ≤ (∑ k : Fin n, (Spec.get2 W i k) * (TorchLean.Tensor.getScalar x k)) := by
      -- Start from `hLowerSum` and distribute the sum.
      let f : Fin n → ℝ :=
        fun k =>
          (Spec.get2 (NN.MLTheory.CROWN.IBP.matPos (α := ℝ) (m := m) (n := n) W) i k) *
            (TorchLean.Tensor.getScalar lo k)
      let g : Fin n → ℝ :=
        fun k =>
          (Spec.get2 (NN.MLTheory.CROWN.IBP.matNeg (α := ℝ) (m := m) (n := n) W) i k) *
            (TorchLean.Tensor.getScalar hi k)
      have hLowerSum_fg :
          (∑ k : Fin n, (f k + g k)) ≤
            (∑ k : Fin n, (Spec.get2 W i k) * (TorchLean.Tensor.getScalar x k)) := by
        simpa [f, g] using hLowerSum
      have hdist : (∑ k : Fin n, (f k + g k)) = (∑ k : Fin n, f k) + (∑ k : Fin n, g k) := by
        simp [Finset.sum_add_distrib, f, g]
      simpa [hdist, f, g] using hLowerSum_fg
    have hLowerSum_swapped :
        (∑ k : Fin n,
            (Spec.get2 (NN.MLTheory.CROWN.IBP.matNeg (α := ℝ) (m := m) (n := n) W) i k) *
              (TorchLean.Tensor.getScalar hi k)) +
          (∑ k : Fin n,
            (Spec.get2 (NN.MLTheory.CROWN.IBP.matPos (α := ℝ) (m := m) (n := n) W) i k) *
              (TorchLean.Tensor.getScalar lo k))
          ≤ (∑ k : Fin n, (Spec.get2 W i k) * (TorchLean.Tensor.getScalar x k)) := by
      simpa [add_comm, add_left_comm, add_assoc] using hLowerSum'
    simp [getScalar_add_spec, hW, hPos_lo, hNeg_hi, hLowerSum_swapped, add_comm]

  have hhi :
      TorchLean.Tensor.getScalar (Tensor.addSpec (α := ℝ) (Spec.matVecMulSpec (α := ℝ) W x) b) i
        ≤
      TorchLean.Tensor.getScalar
          (Tensor.addSpec (α := ℝ)
            (Tensor.addSpec (α := ℝ)
              (Spec.matVecMulSpec (α := ℝ) (NN.MLTheory.CROWN.IBP.matPos (α := ℝ) (m := m) (n :=
                n) W) hi)
              (Spec.matVecMulSpec (α := ℝ) (NN.MLTheory.CROWN.IBP.matNeg (α := ℝ) (m := m) (n :=
                n) W) lo))
            b) i := by
    have hUpperSum' :
        (∑ k : Fin n, (Spec.get2 W i k) * (TorchLean.Tensor.getScalar x k)) ≤
          (∑ k : Fin n,
              (Spec.get2 (NN.MLTheory.CROWN.IBP.matPos (α := ℝ) (m := m) (n := n) W) i k) *
                (TorchLean.Tensor.getScalar hi k)) +
            (∑ k : Fin n,
              (Spec.get2 (NN.MLTheory.CROWN.IBP.matNeg (α := ℝ) (m := m) (n := n) W) i k) *
                (TorchLean.Tensor.getScalar lo k)) := by
      let f : Fin n → ℝ :=
        fun k =>
          (Spec.get2 (NN.MLTheory.CROWN.IBP.matPos (α := ℝ) (m := m) (n := n) W) i k) *
            (TorchLean.Tensor.getScalar hi k)
      let g : Fin n → ℝ :=
        fun k =>
          (Spec.get2 (NN.MLTheory.CROWN.IBP.matNeg (α := ℝ) (m := m) (n := n) W) i k) *
            (TorchLean.Tensor.getScalar lo k)
      have hUpperSum_fg :
          (∑ k : Fin n, (Spec.get2 W i k) * (TorchLean.Tensor.getScalar x k)) ≤
            (∑ k : Fin n, (f k + g k)) := by
        simpa [f, g] using hUpperSum
      have hdist : (∑ k : Fin n, (f k + g k)) = (∑ k : Fin n, f k) + (∑ k : Fin n, g k) := by
        simp [Finset.sum_add_distrib, f, g]
      simpa [hdist, f, g] using hUpperSum_fg
    have hUpperSum_swapped :
        (∑ k : Fin n, (Spec.get2 W i k) * (TorchLean.Tensor.getScalar x k)) ≤
          (∑ k : Fin n,
              (Spec.get2 (NN.MLTheory.CROWN.IBP.matNeg (α := ℝ) (m := m) (n := n) W) i k) *
                (TorchLean.Tensor.getScalar lo k)) +
            (∑ k : Fin n,
              (Spec.get2 (NN.MLTheory.CROWN.IBP.matPos (α := ℝ) (m := m) (n := n) W) i k) *
                (TorchLean.Tensor.getScalar hi k)) := by
      simpa [add_comm, add_left_comm, add_assoc] using hUpperSum'
    simp [getScalar_add_spec, hW, hPos_hi, hNeg_lo, hUpperSum_swapped, add_comm]

  exact ⟨hlo, hhi⟩

/-! ## ReLU relaxations used by α-CROWN -/

theorem relu_relax_scalar_upper_real_runtime
  (l u x : ℝ)
  (hlx : l ≤ x) (hxu : x ≤ u) :
  let rp := NN.MLTheory.CROWN.Runtime.Ops.ReLU.relaxScalar (α:=ℝ) l u
  Activation.Math.reluSpec (α:=ℝ) x ≤ rp.slope * x + rp.bias := by
  -- Same structure as `Models/mlp.lean`, but for `Runtime/Ops`.
  unfold NN.MLTheory.CROWN.Runtime.Ops.ReLU.relaxScalar
  by_cases hu : u > 0
  · by_cases hlpos : l > 0
    · have hxpos : 0 < x := lt_of_lt_of_le hlpos hlx
      have hxnonneg : 0 ≤ x := le_of_lt hxpos
      simp [hu, hlpos, Activation.Math.reluSpec_eq_max, max_eq_left hxnonneg]
    · have hle0 : l ≤ 0 := le_of_not_gt hlpos
      have hden : 0 < (u - l) := by linarith
      simp only [hu, hlpos, ite_true, ite_false]
      by_cases hxpos : 0 < x
      · have hxnonneg : 0 ≤ x := le_of_lt hxpos
        simp [Activation.Math.reluSpec_eq_max, max_eq_left hxnonneg]
        have hx_to_goal : x ≤ u / (u - l) * (x - l) := by
          have hrewrite : (u - l) * x - u * (x - l) = l * (u - x) := by ring
          have hxux : 0 ≤ u - x := sub_nonneg.mpr hxu
          have hxmul_le : l * (u - x) ≤ 0 := mul_nonpos_of_nonpos_of_nonneg hle0 hxux
          have hmul_goal : (u - l) * x ≤ u * (x - l) := by
            have : (u - l) * x - u * (x - l) ≤ 0 := by simpa [hrewrite] using hxmul_le
            exact sub_nonpos.mp this
          have hx_to_goal' : x ≤ (u * (x - l)) / (u - l) := by
            have : x * (u - l) ≤ u * (x - l) := by simpa [mul_comm] using hmul_goal
            exact (le_div_iff₀ (G₀ := ℝ) hden).mpr this
          simpa [div_eq_mul_inv, mul_comm, mul_left_comm, mul_assoc] using hx_to_goal'
        have h2 : u / (u - l) * (x - l) = u / (u - l) * x + -(u / (u - l)) * l := by ring
        simpa [h2] using hx_to_goal
      · have hxle : x ≤ 0 := le_of_not_gt hxpos
        have h1 : u / (u - l) * x + -(u / (u - l) * l) = u / (u - l) * (x - l) := by ring
        have : 0 ≤ u / (u - l) * (x - l) := by
          apply mul_nonneg
          · have : 0 ≤ u := le_of_lt hu
            exact div_nonneg this (le_of_lt hden)
          · linarith
        simpa [Activation.Math.reluSpec_eq_max, max_eq_right hxle, h1] using this
  · have hxle : x ≤ 0 := le_trans hxu (le_of_not_gt hu)
    simp [hu, Activation.Math.reluSpec_eq_max, max_eq_right hxle]

/-- The upper ReLU relaxation has nonnegative slope, in all three phase branches.

Monotonicity of the relaxation is what lets the transfer proofs multiply an inequality by the slope
without flipping it, so this small fact is used pervasively downstream. -/
theorem relax_scalar_slope_nonneg (l u : ℝ) :
    0 ≤ (NN.MLTheory.CROWN.Runtime.Ops.ReLU.relaxScalar (α := ℝ) l u).slope := by
  unfold NN.MLTheory.CROWN.Runtime.Ops.ReLU.relaxScalar
  by_cases hu : u > 0
  · by_cases hlpos : l > 0
    · simp [hu, hlpos]
    · have hden : 0 < (u - l) := by
        have hl0 : l ≤ 0 := le_of_not_gt hlpos
        linarith
      have : 0 ≤ u / (u - l) := by
        have : 0 ≤ u := le_of_lt hu
        exact div_nonneg this (le_of_lt hden)
      simp [hu, hlpos, this]
  · simp [hu]

/-- The α-parameterized lower relaxation also has nonnegative slope, for any `α ≥ 0`. -/
theorem alphaRelaxLowerScalar_slope_nonneg (l u a : ℝ) (ha0 : 0 ≤ a) :
    0 ≤ (alphaRelaxLowerScalar (α := ℝ) l u a).slope := by
  unfold alphaRelaxLowerScalar
  by_cases hu : u > 0
  · by_cases hlpos : l > 0
    · simp [hu, hlpos]
    · simp [hu, hlpos, ha0]
  · simp [hu]

/-! ## ReLU relaxations used by α/β-CROWN (β phase constraints) -/

theorem phaseConsistentScalar?_inactive {l u : ℝ} :
    phaseConsistentScalar? (α := ℝ) l u ReLUPhase.inactive = some () → u ≤ 0 := by
  intro h
  unfold phaseConsistentScalar? at h
  by_cases hu : u > 0
  · simp [hu] at h
  ·
    have : ¬ (0 : ℝ) < u := by simpa using hu
    exact (not_lt).1 this

/-- Accepting an `active` phase constraint forces `0 ≤ l`, so the neuron really is unambiguously on.

This is how the β-CROWN branch decisions are validated: the checker refuses a phase assignment that
the interval bounds do not already support, rather than trusting the search that proposed it. -/
theorem phaseConsistentScalar?_active {l u : ℝ} :
    phaseConsistentScalar? (α := ℝ) l u ReLUPhase.active = some () → 0 ≤ l := by
  intro h
  unfold phaseConsistentScalar? at h
  by_cases hl : l < 0
  · simp [hl] at h
  ·
    have : ¬ l < (0 : ℝ) := by simpa using hl
    exact (not_lt).1 this

/-- Nonnegative slope for the upper relaxation under any phase constraint. -/
theorem phaseRelaxUpperScalar_slope_nonneg (l u : ℝ) (ph : ReLUPhase) :
    0 ≤ (phaseRelaxUpperScalar (α := ℝ) l u ph).slope := by
  cases ph <;> simp [phaseRelaxUpperScalar, relax_scalar_slope_nonneg]

/-- Nonnegative slope for the lower relaxation under any phase constraint and `α ≥ 0`. -/
theorem phaseRelaxLowerScalar_slope_nonneg (l u a : ℝ) (ph : ReLUPhase) (ha0 : 0 ≤ a) :
    0 ≤ (phaseRelaxLowerScalar (α := ℝ) l u a ph).slope := by
  cases ph <;> simp [phaseRelaxLowerScalar, alphaRelaxLowerScalar_slope_nonneg, ha0]

/-! ## ReLU transfer helpers (getScalar-level) -/

theorem defaultAlphaVec_range {n : Nat}
    (lo hi : Tensor ℝ [n]) :
    ∀ i : Fin n, (0 : ℝ) ≤ getScalar (defaultAlphaVec (α := ℝ) (n := n) lo hi) i ∧
      getScalar (defaultAlphaVec (α := ℝ) (n := n) lo hi) i ≤ (1 : ℝ) := by
  classical
  intro i
  by_cases h : getScalar hi i > -getScalar lo i
  · simp [defaultAlphaVec, h]
  · simp [defaultAlphaVec, h]

/-- The vector ReLU is the scalar ReLU coordinatewise. -/
theorem getScalar_relu_spec {n : Nat} (t : Tensor ℝ [n]) (i : Fin n) :
    getScalar (Activation.reluSpec (α := ℝ) t) i =
      Activation.Math.reluSpec (α := ℝ) (getScalar t i) := by
  simp [Activation.reluSpec]

/-- The vector relaxation is the scalar relaxation coordinatewise, which is what turns the scalar
soundness lemmas above into statements about whole layers. -/
theorem getScalar_runtime_relu_relax_vector {n : Nat}
    (lo hi : Tensor ℝ [n]) (i : Fin n) :
    getScalar (NN.MLTheory.CROWN.Runtime.Ops.ReLU.relaxVector (α := ℝ) (n := n) lo hi) i =
      NN.MLTheory.CROWN.Runtime.Ops.ReLU.relaxScalar (α := ℝ) (getScalar lo i)
        (getScalar hi i) := by
  simp [NN.MLTheory.CROWN.Runtime.Ops.ReLU.relaxVector]

/-- Same coordinatewise reading for the α-parameterized lower relaxation. -/
theorem getScalar_alphaRelaxLowerVec {n : Nat}
    (lo hi αv : Tensor ℝ [n]) (i : Fin n) :
    getScalar (alphaRelaxLowerVec (α := ℝ) (n := n) lo hi αv) i =
      alphaRelaxLowerScalar (α := ℝ) (getScalar lo i) (getScalar hi i) (getScalar αv i) := by
  simp [alphaRelaxLowerVec]

/-- Propagating an affine form through a ReLU relaxation acts coordinatewise as `slope * value +
bias`.

The implementation scales the whole matrix and shifts the whole constant, so this lemma is the
bridge
from that global operation to the per-neuron reasoning the soundness proof needs. -/
theorem getScalar_affineEvalAt_relu_propagate_affine
    {inDim hidDim : Nat}
    (relax : Tensor (NN.MLTheory.CROWN.Runtime.Ops.ReLURelax ℝ) [hidDim])
    (aff : AffineVec ℝ inDim hidDim)
    (x : Tensor ℝ [inDim]) (i : Fin hidDim) :
    getScalar
        (CrownCertSoundness.affineEvalAt (α := ℝ) (inDim := inDim) (outDim := hidDim)
          (NN.MLTheory.CROWN.Runtime.Ops.ReLU.propagateAffine (α := ℝ)
            (inDim := inDim) (hidDim := hidDim) relax aff) x) i
      =
      (let rp := getScalar relax i
       rp.slope *
          getScalar
            (CrownCertSoundness.affineEvalAt (α := ℝ) (inDim := inDim) (outDim := hidDim) aff x)
            i +
        rp.bias) := by
  classical
  let rp := getScalar relax i
  let propagated :=
    NN.MLTheory.CROWN.Runtime.Ops.ReLU.propagateAffine (α := ℝ)
      (inDim := inDim) (hidDim := hidDim) relax aff
  have hget2 (j : Fin inDim) :
      Spec.get2 propagated.A i j = Spec.get2 aff.A i j * rp.slope := by
    simp [propagated, rp, NN.MLTheory.CROWN.Runtime.Ops.ReLU.propagateAffine]
  have hc :
      getScalar propagated.c i = rp.slope * getScalar aff.c i + rp.bias := by
    simp [propagated, rp, NN.MLTheory.CROWN.Runtime.Ops.ReLU.propagateAffine]
  have hscale :
      (∑ k : Fin inDim, (Spec.get2 aff.A i k * rp.slope) * getScalar x k) =
        rp.slope * (∑ k : Fin inDim, Spec.get2 aff.A i k * getScalar x k) := by
    calc
      (∑ k : Fin inDim, (Spec.get2 aff.A i k * rp.slope) * getScalar x k) =
          ∑ k : Fin inDim, rp.slope * (Spec.get2 aff.A i k * getScalar x k) := by
            apply Finset.sum_congr rfl
            intro k _
            ring
      _ = rp.slope * (∑ k : Fin inDim, Spec.get2 aff.A i k * getScalar x k) := by
            exact (Finset.mul_sum _ _ _).symm
  change getScalar
      (CrownCertSoundness.affineEvalAt (α := ℝ) propagated x) i =
    rp.slope * getScalar (CrownCertSoundness.affineEvalAt (α := ℝ) aff x) i + rp.bias
  simp only [CrownCertSoundness.affineEvalAt]
  rw [congrFun (Spec.getScalar_add_spec _ _) i,
    congrFun (Spec.getScalar_add_spec _ _) i]
  rw [Proofs.TensorAlgebra.getScalar_mat_vec_mul_spec
      (A := propagated.A) (v := x) (i := i),
    Proofs.TensorAlgebra.getScalar_mat_vec_mul_spec
      (A := aff.A) (v := x) (i := i)]
  simp_rw [hget2]
  rw [hc, hscale]
  ring

/-!
β phase vectors (`AlphaBetaCROWN.phaseRelaxVec?`) are executable, so to reason about them we
extract their per-index consequences from the fact they returned `some ...`.
-/

theorem List.all_eq_true_of_mem {α : Type} (p : α → Bool) (xs : List α) :
    xs.all p = true → ∀ x : α, x ∈ xs → p x = true := by
  intro hall
  induction xs with
  | nil =>
      intro x hx
      cases hx
  | cons a xs ih =>
      -- Unfold `List.all` without rewriting it into a `∀`-statement.
      have ha' : p a = true ∧ xs.all p = true := by
        simpa [List.all, Bool.and_eq_true] using hall
      rcases ha' with ⟨ha, hxs⟩
      intro x hx
      have hx' : x = a ∨ x ∈ xs := by
        simpa [List.mem_cons] using hx
      cases hx' with
      | inl hxa =>
          cases hxa
          simpa using ha
      | inr hxmem =>
          exact ih hxs x hxmem

/-- Unpack a successful `phaseRelaxVec?`: the phase array has the right length, and every coordinate
carries a phase that is consistent with its interval and whose two relaxations are the scalar ones.

The checker returns only the relaxations, so this is the lemma that recovers the phase witnesses the
soundness argument has to case on. -/
theorem phaseRelaxVec?_some_getScalar {n : Nat}
    (lo hi αv : Tensor ℝ [n]) (phases : Array Int)
    (relaxLo relaxHi : Tensor (NN.MLTheory.CROWN.Runtime.Ops.ReLURelax ℝ) [n])
    (h : phaseRelaxVec? (α := ℝ) (n := n) lo hi αv phases = some (relaxLo, relaxHi)) :
    phases.size = n ∧
      ∀ i : Fin n,
        ∃ ph : ReLUPhase,
          phaseConsistentScalar? (α := ℝ) (getScalar lo i) (getScalar hi i) ph = some () ∧
          getScalar relaxHi i =
            phaseRelaxUpperScalar (α := ℝ) (getScalar lo i) (getScalar hi i) ph ∧
          getScalar relaxLo i =
            phaseRelaxLowerScalar (α := ℝ) (getScalar lo i) (getScalar hi i) (getScalar αv i) ph
            := by
  classical
  by_cases hlen : phases.size = n
  · refine ⟨hlen, ?_⟩
    have hS := h
    simp [phaseRelaxVec?, hlen] at hS
    rcases hS with ⟨hOkAll, hLoEq, hHiEq⟩
    intro i
    have hpTrue := hOkAll i
    cases hph : ReLUPhase.ofInt? (betaAt phases (↑i)) with
    | none =>
        have : False := by
          simp [hph] at hpTrue
        exact False.elim this
    | some ph =>
        cases hcons :
            phaseConsistentScalar? (α := ℝ) (getScalar lo i) (getScalar hi i) ph with
        | none =>
            have : False := by
              simp [hph, hcons] at hpTrue
            exact False.elim this
        | some result =>
            cases result
            refine ⟨ph, ?_, ?_, ?_⟩
            · simpa using hcons
            · rw [← hHiEq]
              simp [hph]
            · rw [← hLoEq]
              simp [hph]
  ·
    have hnone : phaseRelaxVec? (α := ℝ) (n := n) lo hi αv phases = none := by
      simp [phaseRelaxVec?, hlen]
    have : False := by
      simp [hnone] at h
    exact False.elim this

/-! ## Evaluating `linearBoundsFromAffine` at a point -/

theorem get2_add_spec {m n : Nat}
    (A B : Tensor ℝ [m, n]) (i : Fin m) (j : Fin n) :
    Spec.get2 (Tensor.addSpec (α := ℝ) A B) i j = Spec.get2 A i j + Spec.get2 B i j := by
  simp [Tensor.addSpec]

/-- Matrix-vector multiplication is additive in the matrix. -/
theorem mat_vec_add_matrix {m n : Nat}
    (A B : Tensor ℝ [m, n])
    (x : Tensor ℝ [n]) :
    Spec.matVecMulSpec (α := ℝ) (Tensor.addSpec (α := ℝ) A B) x =
      Tensor.addSpec (α := ℝ)
        (Spec.matVecMulSpec (α := ℝ) A x)
        (Spec.matVecMulSpec (α := ℝ) B x) := by
  classical
  have hgetScalar :
      TorchLean.Tensor.getScalar (Spec.matVecMulSpec (α := ℝ) (Tensor.addSpec (α := ℝ) A B) x) =
        TorchLean.Tensor.getScalar
          (Tensor.addSpec (α := ℝ)
            (Spec.matVecMulSpec (α := ℝ) A x)
            (Spec.matVecMulSpec (α := ℝ) B x)) := by
    funext i
    rw [Proofs.TensorAlgebra.getScalar_mat_vec_mul_spec (A := Tensor.addSpec (α := ℝ) A B) (v := x)
      (i := i)]
    simp [Spec.getScalar_add_spec]
    rw [Proofs.TensorAlgebra.getScalar_mat_vec_mul_spec (A := A) (v := x) (i := i)]
    rw [Proofs.TensorAlgebra.getScalar_mat_vec_mul_spec (A := B) (v := x) (i := i)]
    -- Distribute `get2 (A+B)` and split the sum.
    have :
        (∑ k : Fin n,
            (Spec.get2 (Tensor.addSpec (α := ℝ) A B) i k) * (TorchLean.Tensor.getScalar x k)) =
          (∑ k : Fin n, (Spec.get2 A i k) * (TorchLean.Tensor.getScalar x k)) +
          (∑ k : Fin n, (Spec.get2 B i k) * (TorchLean.Tensor.getScalar x k)) := by
      classical
      calc
        (∑ k : Fin n,
            (Spec.get2 (Tensor.addSpec (α := ℝ) A B) i k) * (TorchLean.Tensor.getScalar x k))
            = ∑ k : Fin n,
                ((Spec.get2 A i k + Spec.get2 B i k) * (TorchLean.Tensor.getScalar x k)) := by
                refine Finset.sum_congr rfl ?_
                intro k _
                simp [get2_add_spec]
        _ = ∑ k : Fin n, ((Spec.get2 A i k) * (TorchLean.Tensor.getScalar x k) +
              (Spec.get2 B i k) * (TorchLean.Tensor.getScalar x k)) := by
              simp [add_mul]
        _ = (∑ k : Fin n, (Spec.get2 A i k) * (TorchLean.Tensor.getScalar x k)) +
            (∑ k : Fin n, (Spec.get2 B i k) * (TorchLean.Tensor.getScalar x k)) := by
              simp [Finset.sum_add_distrib]
    simp [this]
  have hTensor := congrArg TorchLean.Tensor.ofFn hgetScalar
  simpa using
    (Eq.trans (TorchLean.Tensor.ofFn_getScalar
        (t := Spec.matVecMulSpec (α := ℝ) (Tensor.addSpec (α := ℝ) A B) x)).symm
      (Eq.trans hTensor (TorchLean.Tensor.ofFn_getScalar (t := Tensor.addSpec (α := ℝ)
        (Spec.matVecMulSpec (α := ℝ) A x)
        (Spec.matVecMulSpec (α := ℝ) B x)))))

/-- The zero matrix sends every vector to zero, which is what makes constant bounds constant. -/
theorem mat_vec_mul_spec_full_zero {m n : Nat}
    (x : Tensor ℝ [n]) :
    Spec.matVecMulSpec (α := ℝ) (Tensor.full (α := ℝ) (.dim m (.dim n .scalar)) (0 : ℝ)) x =
      Tensor.full (α := ℝ) (.dim m .scalar) (0 : ℝ) := by
  classical
  have hgetScalar :
      TorchLean.Tensor.getScalar
          (Spec.matVecMulSpec (α := ℝ) (Tensor.full (α := ℝ) (.dim m (.dim n .scalar)) (0 : ℝ)) x) =
        TorchLean.Tensor.getScalar (Tensor.full (α := ℝ) (.dim m .scalar) (0 : ℝ)) := by
    funext i
    -- Expand the mat-vec coordinate as a finite sum; all terms are zero.
    rw [Proofs.TensorAlgebra.getScalar_mat_vec_mul_spec
      (A := Tensor.full (α := ℝ) (.dim m (.dim n .scalar)) (0 : ℝ)) (v := x) (i := i)]
    simp
  have hTensor := congrArg TorchLean.Tensor.ofFn hgetScalar
  simpa using
    (Eq.trans (TorchLean.Tensor.ofFn_getScalar (t := Spec.matVecMulSpec (α := ℝ)
        (Tensor.full (α := ℝ) (.dim m (.dim n .scalar)) (0 : ℝ)) x)).symm
      (Eq.trans hTensor
        (TorchLean.Tensor.ofFn_getScalar (t := Tensor.full (α := ℝ) (.dim m .scalar) (0 : ℝ)))))

/-- The identity affine certificate acts as the identity on inputs. -/
theorem mat_vec_mul_spec_aff_identity {n : Nat}
    (x : Tensor ℝ [n]) :
    Spec.matVecMulSpec (α := ℝ) (Cert.affIdentity (α := ℝ) n).A x = x := by
  classical
  have hgetScalar :
      TorchLean.Tensor.getScalar (Spec.matVecMulSpec (α := ℝ) (Cert.affIdentity (α := ℝ) n).A x) =
        TorchLean.Tensor.getScalar x := by
    funext i
    -- Expand mat-vec coordinate; only the diagonal term survives.
    rw [Proofs.TensorAlgebra.getScalar_mat_vec_mul_spec
      (A := (Cert.affIdentity (α := ℝ) n).A) (v := x) (i := i)]
    simp only [Cert.affIdentity, Spec.get2_dim]
    rw [Finset.sum_eq_single i]
    · simp
    · intro j _ hj
      have hji : i ≠ j := fun h => hj h.symm
      simp [hji]
    · intro hnot
      exact False.elim (hnot (Finset.mem_univ i))
  have hTensor := congrArg TorchLean.Tensor.ofFn hgetScalar
  simpa using
    (Eq.trans (TorchLean.Tensor.ofFn_getScalar
        (t := Spec.matVecMulSpec (α := ℝ) (Cert.affIdentity (α := ℝ) n).A x)).symm
      (Eq.trans hTensor (TorchLean.Tensor.ofFn_getScalar (t := x))))

/-- The identity bounds certificate evaluates to the degenerate box `[x, x]`.

This is the base case of every certificate chain: the input is bounded by itself exactly. -/
theorem boundsEvalAt_bounds_identity {n : Nat} (x : Tensor ℝ [n]) :
    boundsEvalAt (α := ℝ) (Cert.boundsIdentity (α := ℝ) n) x = { dim := n, lo := x, hi := x } := by
  classical
  have hMat :
      Spec.matVecMulSpec (α := ℝ) (Cert.affIdentity (α := ℝ) n).A x = x :=
    mat_vec_mul_spec_aff_identity (n := n) x
  have hC : (Cert.affIdentity (α := ℝ) n).c = Tensor.full (α := ℝ) (.dim n .scalar) (0 : ℝ) := by
    simp [Cert.affIdentity]
  ext <;> simp [boundsEvalAt, Cert.boundsIdentity, affineEvalAt, hMat, hC]

/-- A constant bounds certificate evaluates to its own endpoints, ignoring the input. -/
theorem boundsEvalAt_bounds_const {inDim outDim : Nat}
    (lo hi : Tensor ℝ [outDim]) (x : Tensor ℝ [inDim]) :
    boundsEvalAt (α := ℝ) (Cert.boundsConst (α := ℝ) inDim outDim lo hi) x =
      { dim := outDim
        lo := lo
        hi := hi } := by
  classical
  ext <;> simp [boundsEvalAt, Cert.boundsConst, affineEvalAt, mat_vec_mul_spec_full_zero]

/-- Left commutativity of tensor addition, derived from associativity and commutativity. -/
theorem add_spec_left_comm {s : Shape}
    (a b c : Tensor ℝ s) :
    Tensor.addSpec (α := ℝ) a (Tensor.addSpec (α := ℝ) b c) =
      Tensor.addSpec (α := ℝ) b (Tensor.addSpec (α := ℝ) a c) := by
  -- Derive left-commutativity from `add_spec_assoc` and `add_spec_comm`.
  calc
    Tensor.addSpec (α := ℝ) a (Tensor.addSpec (α := ℝ) b c)
        = Tensor.addSpec (α := ℝ) (Tensor.addSpec (α := ℝ) a b) c := by
            simpa using (add_spec_assoc (a := a) (b := b) (c := c)).symm
    _ = Tensor.addSpec (α := ℝ) (Tensor.addSpec (α := ℝ) b a) c := by
            simp [add_spec_comm]
    _ = Tensor.addSpec (α := ℝ) b (Tensor.addSpec (α := ℝ) a c) := by
            simpa using (add_spec_assoc (a := b) (b := a) (c := c))

/-- Regroup `(a + b) + (c + d)` as `(a + c) + (b + d)`.

Certificate composition produces sums in the order the layers were visited, while the goal groups
them by which affine form they came from; this is the reassociation that reconciles the two. -/
theorem add_spec_pair_distrib {s : Shape}
    (a b c d : Tensor ℝ s) :
    Tensor.addSpec (α := ℝ) (Tensor.addSpec (α := ℝ) a b) (Tensor.addSpec (α := ℝ) c d) =
      Tensor.addSpec (α := ℝ) (Tensor.addSpec (α := ℝ) a c) (Tensor.addSpec (α := ℝ) b d) := by
  calc
    Tensor.addSpec (α := ℝ) (Tensor.addSpec (α := ℝ) a b) (Tensor.addSpec (α := ℝ) c d)
        = Tensor.addSpec (α := ℝ) a (Tensor.addSpec (α := ℝ) b (Tensor.addSpec (α := ℝ) c d)) :=
          by
            simpa using (add_spec_assoc (a := a) (b := b) (c := Tensor.addSpec (α := ℝ) c d))
    _ = Tensor.addSpec (α := ℝ) a (Tensor.addSpec (α := ℝ) c (Tensor.addSpec (α := ℝ) b d)) := by
            -- swap `b` and `c` inside `b + (c + d)`
            have hswap :
                Tensor.addSpec (α := ℝ) b (Tensor.addSpec (α := ℝ) c d) =
                  Tensor.addSpec (α := ℝ) c (Tensor.addSpec (α := ℝ) b d) :=
              add_spec_left_comm (a := b) (b := c) (c := d)
            simp [hswap]
    _ = Tensor.addSpec (α := ℝ) (Tensor.addSpec (α := ℝ) a c) (Tensor.addSpec (α := ℝ) b d) := by
            simpa using (add_spec_assoc (a := a) (b := c) (c := Tensor.addSpec (α := ℝ) b d)).symm

/-- Evaluating the composition of a linear layer with two incoming affine forms distributes over the
pair: matrices compose, constants are pushed through, and the bias is added once at the end. -/
theorem affineEvalAt_linear_pair
    {inDim n m : Nat}
    (W₁ W₂ : Tensor ℝ [m, n])
    (aff₁ aff₂ : AffineVec ℝ inDim n)
    (b : Tensor ℝ [m])
    (x : Tensor ℝ [inDim]) :
    affineEvalAt (α := ℝ)
        { A := Tensor.addSpec (Spec.matMulSpec W₁ aff₁.A) (Spec.matMulSpec W₂ aff₂.A)
          c := Tensor.addSpec
            (Tensor.addSpec (Spec.matVecMulSpec W₁ aff₁.c) (Spec.matVecMulSpec W₂ aff₂.c)) b } x =
      Tensor.addSpec
        (Tensor.addSpec
          (Spec.matVecMulSpec W₁ (affineEvalAt (α := ℝ) aff₁ x))
          (Spec.matVecMulSpec W₂ (affineEvalAt (α := ℝ) aff₂ x)))
        b := by
  simp only [affineEvalAt, mat_vec_add_matrix, Spec.mat_vec_assoc, Spec.mat_vec_add]
  rw [(add_spec_assoc (a := _) (b := _) (c := b)).symm]
  apply congrArg (fun z => Tensor.addSpec (α := ℝ) z b)
  exact add_spec_pair_distrib _ _ _ _

/-- Evaluating the bounds certificate that a linear layer builds from an incoming affine bound gives
exactly the sign-split box of the evaluated endpoints.

This is the lemma that connects the certificate the checker writes down to the interval arithmetic
that `encloses_linear_signSplit` proves sound, and it carries the output-dimension cast that the
graph dialect needs because layer widths are only equal up to a proof. -/
theorem boundsEvalAt_linear_bounds_from_affine
    {n m : Nat}
    (W : Tensor ℝ [m, n])
    (b : Tensor ℝ [m])
    (xB : FlatAffineBounds ℝ)
    (hout : xB.outDim = n)
    (x : Tensor ℝ [xB.inDim]) :
    boundsEvalAt (α := ℝ)
        (Cert.linearBoundsFromAffine (α := ℝ) (inDim := xB.inDim) (n := n) (m := m) W b xB hout) x =
      { dim := m
        lo :=
          let l := affineEvalAt (α := ℝ) (inDim := xB.inDim) (outDim := n)
            (Graph.castAffineOut (α := ℝ) (n := xB.inDim) (m := xB.outDim) (m' := n) hout
              xB.loAff) x
          let u := affineEvalAt (α := ℝ) (inDim := xB.inDim) (outDim := n)
            (Graph.castAffineOut (α := ℝ) (n := xB.inDim) (m := xB.outDim) (m' := n) hout
              xB.hiAff) x
          Tensor.addSpec (α := ℝ)
            (Tensor.addSpec (α := ℝ)
              (Spec.matVecMulSpec (α := ℝ) (NN.MLTheory.CROWN.IBP.matPos (α := ℝ) (m := m) (n :=
                n) W) l)
              (Spec.matVecMulSpec (α := ℝ) (NN.MLTheory.CROWN.IBP.matNeg (α := ℝ) (m := m) (n :=
                n) W) u))
            b
        hi :=
          let l := affineEvalAt (α := ℝ) (inDim := xB.inDim) (outDim := n)
            (Graph.castAffineOut (α := ℝ) (n := xB.inDim) (m := xB.outDim) (m' := n) hout
              xB.loAff) x
          let u := affineEvalAt (α := ℝ) (inDim := xB.inDim) (outDim := n)
            (Graph.castAffineOut (α := ℝ) (n := xB.inDim) (m := xB.outDim) (m' := n) hout
              xB.hiAff) x
          Tensor.addSpec (α := ℝ)
            (Tensor.addSpec (α := ℝ)
              (Spec.matVecMulSpec (α := ℝ) (NN.MLTheory.CROWN.IBP.matPos (α := ℝ) (m := m) (n :=
                n) W) u)
              (Spec.matVecMulSpec (α := ℝ) (NN.MLTheory.CROWN.IBP.matNeg (α := ℝ) (m := m) (n :=
                n) W) l))
            b } := by
  classical
  let xLo :=
    Graph.castAffineOut (α := ℝ) (n := xB.inDim) (m := xB.outDim) (m' := n) hout xB.loAff
  let xHi :=
    Graph.castAffineOut (α := ℝ) (n := xB.inDim) (m := xB.outDim) (m' := n) hout xB.hiAff
  let Wpos := NN.MLTheory.CROWN.IBP.matPos (α := ℝ) (m := m) (n := n) W
  let Wneg := NN.MLTheory.CROWN.IBP.matNeg (α := ℝ) (m := m) (n := n) W
  have hlo := affineEvalAt_linear_pair Wpos Wneg xLo xHi b x
  have hhi := affineEvalAt_linear_pair Wpos Wneg xHi xLo b x
  unfold boundsEvalAt Cert.linearBoundsFromAffine
  dsimp only
  refine FlatBox.ext' rfl (heq_of_eq ?_) (heq_of_eq ?_)
  · exact hlo
  · exact hhi

/-! ## Step wrapper -/

/-- Wrapper around `alphaCrownStepNode?` in the `CrownTransferSound` “step function” shape. -/
def stepAlpha (g : Graph) (ps : ParamStore ℝ)
    (ibp : Array (Option (FlatBox ℝ))) (alpha : Array (Option (FlatTensor ℝ))) (ctx : AffineCtx) :
    Array (Option (FlatAffineBounds ℝ)) → Nat → Option (FlatAffineBounds ℝ) :=
  fun cert id => alphaCrownStepNode? (α := ℝ) g.nodes ps ibp alpha cert ctx id

/-- Wrapper around `alphaBetaCrownStepNode?` in the `CrownTransferSound` “step function” shape. -/
def stepAlphaBeta (g : Graph) (ps : ParamStore ℝ)
    (ibp : Array (Option (FlatBox ℝ)))
    (alpha : Array (Option (FlatTensor ℝ)))
    (beta : Array (Option (Array Int)))
    (ctx : AffineCtx) :
    Array (Option (FlatAffineBounds ℝ)) → Nat → Option (FlatAffineBounds ℝ) :=
  fun cert id => alphaBetaCrownStepNode? (α := ℝ) g.nodes ps ibp alpha beta cert ctx id

end

end AlphaCrownTransferSoundness

end NN.MLTheory.CROWN.Graph
