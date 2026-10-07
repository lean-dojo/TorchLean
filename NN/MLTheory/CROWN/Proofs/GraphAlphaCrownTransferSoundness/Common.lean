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

/-- Enclosure in a box is the conjunction of the two coordinatewise inequalities. -/
theorem encloses_iff_getScalar {n : Nat}
    (lo hi x : Tensor ℝ [n]) :
    Theorems.Semantics.encloses (α := ℝ) { dim := n, lo := lo, hi := hi } x ↔
      ∀ i : Fin n, getScalar lo i ≤ getScalar x i ∧ getScalar x i ≤ getScalar hi i := by
  rfl

/-! ## Small tensor algebra helpers -/

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

/-- `castDimScalar` is proof-irrelevant in its equality argument. -/
theorem castDimScalar_proof_irrel {n n' : Nat}
    (h₁ h₂ : n = n') (t : Tensor ℝ [n]) :
    castDimScalar (α := ℝ) h₁ t = castDimScalar (α := ℝ) h₂ t :=
  rfl

/-- `getScalar` commutes with `castDimScalar` (up to `Fin.cast`). -/
theorem getScalar_castDimScalar {n n' : Nat} (h : n = n') (t : Tensor ℝ [n]) (i : Fin n') :
    getScalar (castDimScalar (α := ℝ) (n := n) (n' := n') h t) i =
      getScalar t (Fin.cast h.symm i) := by
  cases h
  simp [castDimScalar]

/-- Casting the dimension of a tensor preserves the unit-interval range of its entries. -/
theorem castDimScalar_unit_range {n n' : Nat} (h : n = n') (t : Tensor ℝ [n])
    (hr : ∀ i : Fin n, (0 : ℝ) ≤ getScalar t i ∧ getScalar t i ≤ (1 : ℝ)) :
    ∀ i : Fin n', (0 : ℝ) ≤ getScalar (castDimScalar (α := ℝ) h t) i ∧
      getScalar (castDimScalar (α := ℝ) h t) i ≤ (1 : ℝ) := by
  intro i
  rw [getScalar_castDimScalar]
  exact hr _

/-! ## Safe lookups -/

/-- A successful `Cert.getAff?` lookup is an in-bounds array read. -/
theorem getElem!_of_getAff?_eq_some {cert : Array (Option (FlatAffineBounds ℝ))} {p : Nat}
    {xin : FlatAffineBounds ℝ} (h : NN.MLTheory.CROWN.Cert.getAff? (α := ℝ) cert p = some xin) :
    cert[p]! = some xin := by
  by_cases hlt : p < cert.size
  · simpa [NN.MLTheory.CROWN.Cert.getAff?, Array.getD, hlt] using h
  · simp [NN.MLTheory.CROWN.Cert.getAff?, Array.getD, hlt] at h

/-- A successful `getAlpha?` lookup is an in-bounds array read. -/
theorem getElem!_of_getAlpha?_eq_some {alpha : Array (Option (FlatTensor ℝ))} {id : Nat}
    {αv : FlatTensor ℝ} (h : NN.MLTheory.CROWN.Cert.getAlpha? (α := ℝ) alpha id = some αv) :
    id < alpha.size ∧ alpha[id]! = some αv := by
  by_cases hlt : id < alpha.size
  · exact ⟨hlt, by simpa [NN.MLTheory.CROWN.Cert.getAlpha?, Array.getD, hlt] using h⟩
  · simp [NN.MLTheory.CROWN.Cert.getAlpha?, Array.getD, hlt] at h

/-- Under `AlphaOK`, an α vector read by `getAlpha?` has every entry in `[0, 1]`. -/
theorem getAlpha?_unit_range {alpha : Array (Option (FlatTensor ℝ))} {id : Nat}
    {αv : FlatTensor ℝ} (halpha : AlphaOK (alpha := alpha))
    (h : NN.MLTheory.CROWN.Cert.getAlpha? (α := ℝ) alpha id = some αv) :
    ∀ i : Fin αv.n, (0 : ℝ) ≤ getScalar αv.v i ∧ getScalar αv.v i ≤ (1 : ℝ) := by
  obtain ⟨hlt, hentry⟩ := getElem!_of_getAlpha?_eq_some h
  simpa [hentry] using halpha id hlt

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
  obtain ⟨hinDim, hdim, henc⟩ := hpar
  -- Once `outDim'` is identified with `xin.outDim`, both casts run along `rfl`: the cast bounds
  -- are `xin` itself and the two casts of `vp.v` agree by proof irrelevance, so the parent
  -- enclosure is the goal up to definitional unfolding.
  subst hout
  exact ⟨hinDim, rfl, henc⟩

/-! ## Matrix sign-splitting bound (pointwise, over `ℝ`) -/

/-- Entries of the positive part of a matrix: the entry where it is positive, else zero. -/
theorem get2_mat_pos {m n : Nat}
    (W : Tensor ℝ [m, n]) (i : Fin m) (j : Fin n) :
    Spec.get2 (NN.MLTheory.CROWN.IBP.matPos (α := ℝ) (m := m) (n := n) W) i j =
      (if Spec.get2 W i j > 0 then Spec.get2 W i j else 0) := by
  simpa only [NN.MLTheory.CROWN.IBP.matPos, Tensor.map] using
    Tensor.get2_map (fun w : ℝ => if w > 0 then w else 0) W i j

/-- Entries of the negative part of a matrix: the entry where it is not positive, else zero. -/
theorem get2_mat_neg {m n : Nat}
    (W : Tensor ℝ [m, n]) (i : Fin m) (j : Fin n) :
    Spec.get2 (NN.MLTheory.CROWN.IBP.matNeg (α := ℝ) (m := m) (n := n) W) i j =
      (if Spec.get2 W i j > 0 then 0 else Spec.get2 W i j) := by
  simpa only [NN.MLTheory.CROWN.IBP.matNeg, Tensor.map] using
    Tensor.get2_map (fun w : ℝ => if w > 0 then 0 else w) W i j

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
      simpa only [Finset.sum_add_distrib] using hLowerSum
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
      simpa only [Finset.sum_add_distrib] using hUpperSum
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

export NN.MLTheory.CROWN.Proofs (relu_relax_scalar_upper_real_runtime)

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

/-- Nonnegative slope for the upper relaxation under any phase constraint. -/
theorem phaseRelaxUpperScalar_slope_nonneg (l u : ℝ) (ph : ReLUPhase) :
    0 ≤ (phaseRelaxUpperScalar (α := ℝ) l u ph).slope := by
  cases ph <;> simp [phaseRelaxUpperScalar, relax_scalar_slope_nonneg]

/-- Nonnegative slope for the lower relaxation under any phase constraint and `α ≥ 0`. -/
theorem phaseRelaxLowerScalar_slope_nonneg (l u a : ℝ) (ph : ReLUPhase) (ha0 : 0 ≤ a) :
    0 ≤ (phaseRelaxLowerScalar (α := ℝ) l u a ph).slope := by
  cases ph <;> simp [phaseRelaxLowerScalar, alphaRelaxLowerScalar_slope_nonneg, ha0]

/-! ## ReLU transfer helpers (getScalar-level) -/

/-- The default α vector is a `0`/`1` vector, hence lies in `[0, 1]` coordinatewise. -/
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

/-! ## Evaluating `propagateLinearBounds` at a point -/

/-- Matrix-vector multiplication is additive in the matrix. -/
theorem mat_vec_add_matrix {m n : Nat}
    (A B : Tensor ℝ [m, n])
    (x : Tensor ℝ [n]) :
    Spec.matVecMulSpec (α := ℝ) (Tensor.addSpec (α := ℝ) A B) x =
      Tensor.addSpec (α := ℝ)
        (Spec.matVecMulSpec (α := ℝ) A x)
        (Spec.matVecMulSpec (α := ℝ) B x) := by
  classical
  apply Tensor.ext_vector
  intro i
  rw [congrFun (Spec.getScalar_add_spec _ _) i]
  simp only [Proofs.TensorAlgebra.getScalar_mat_vec_mul_spec]
  simp [Tensor.addSpec, add_mul, Finset.sum_add_distrib]

/-- The zero matrix sends every vector to zero, which is what makes constant bounds constant. -/
theorem mat_vec_mul_spec_full_zero {m n : Nat}
    (x : Tensor ℝ [n]) :
    Spec.matVecMulSpec (α := ℝ) (Tensor.full (α := ℝ) (.dim m (.dim n .scalar)) (0 : ℝ)) x =
      Tensor.full (α := ℝ) (.dim m .scalar) (0 : ℝ) := by
  classical
  apply Tensor.ext_vector
  intro i
  -- Expand the mat-vec coordinate as a finite sum; all terms are zero.
  rw [Proofs.TensorAlgebra.getScalar_mat_vec_mul_spec
    (A := Tensor.full (α := ℝ) (.dim m (.dim n .scalar)) (0 : ℝ)) (v := x) (i := i)]
  simp

/-- The identity affine certificate acts as the identity on inputs. -/
theorem mat_vec_mul_spec_aff_identity {n : Nat}
    (x : Tensor ℝ [n]) :
    Spec.matVecMulSpec (α := ℝ) (Graph.affIdentity (α := ℝ) n).A x = x := by
  classical
  apply Tensor.ext_vector
  intro i
  rw [Proofs.TensorAlgebra.getScalar_mat_vec_mul_spec
    (A := (Graph.affIdentity (α := ℝ) n).A) (v := x) (i := i)]
  simp [Graph.affIdentity, Spec.get2_identityTensorSpec]

/-- The identity bounds certificate evaluates to the degenerate box `[x, x]`.

This is the base case of every certificate chain: the input is bounded by itself exactly. -/
theorem boundsEvalAt_bounds_identity {n : Nat} (x : Tensor ℝ [n]) :
    boundsEvalAt (α := ℝ) (Graph.boundsIdentity (α := ℝ) n) x = { dim := n, lo := x, hi := x } := by
  classical
  have hMat :
      Spec.matVecMulSpec (α := ℝ) (Graph.affIdentity (α := ℝ) n).A x = x :=
    mat_vec_mul_spec_aff_identity (n := n) x
  have hC : (Graph.affIdentity (α := ℝ) n).c = Tensor.full (α := ℝ) (.dim n .scalar) (0 : ℝ) := by
    simp [Graph.affIdentity]
  ext <;> simp [boundsEvalAt, Graph.boundsIdentity, affineEvalAt, hMat, hC]

/-- A constant bounds certificate evaluates to its own endpoints, ignoring the input. -/
theorem boundsEvalAt_bounds_const {inDim outDim : Nat}
    (lo hi : Tensor ℝ [outDim]) (x : Tensor ℝ [inDim]) :
    boundsEvalAt (α := ℝ) (Graph.boundsConst (α := ℝ) inDim outDim lo hi) x =
      { dim := outDim
        lo := lo
        hi := hi } := by
  classical
  ext <;> simp [boundsEvalAt, Graph.boundsConst, affineEvalAt, mat_vec_mul_spec_full_zero]

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
        (Graph.propagateLinearBounds (α := ℝ) (n := n) (m := m) W b xB hout) x =
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
  unfold boundsEvalAt Graph.propagateLinearBounds
  dsimp only
  refine FlatBox.ext rfl (heq_of_eq ?_) (heq_of_eq ?_)
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
