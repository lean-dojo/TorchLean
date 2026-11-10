/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Verify.CROWN.FlatBox.Affine
public import TorchLean.Verify.CROWN.FlatBox.Directed
public import TorchLean.Verify.CROWN.FlatBox.Enclosure
public import TorchLean.Verify.CROWN.Activation.ReLU
public import TorchLean.Tensor.Algebra.Real
public import TorchLean.Tensor.LinearAlgebra.Real
public import TorchLean.Tensor.Layers.Linear

/-!
# Affine bounds at an input point and the real α-CROWN relaxation lemmas

`boundsEvalAt` and `EnclosesAtInput` read affine bounds at a designated input point, where
`Linear.forward` evaluates each affine form. The remaining declarations are the real lemmas behind
α-CROWN soundness: the sign-split rule of a linear layer (`encloses_linear_signSplit`), the α lower
and triangle upper relaxations of ReLU and their phase versions (`phaseRelaxLowerScalar_sound`,
`phaseRelaxUpperScalar_sound`), and their propagation through affine bounds
(`enclosesAtInput_relu_propagate`, `enclosesAtInput_relu_phase`). The statements involve tensors,
boxes and affine forms, and no operator graph appears in them.
-/

@[expose] public section

namespace TorchLean.CROWN.Graph
open Tensor
noncomputable section

/-- Evaluate affine lower/upper bounds at a concrete input point, yielding a `FlatBox`. -/
def boundsEvalAt {α : Type} [Storage α] [Context α]
    (b : FlatAffineBounds α) (x : Tensor α [b.inDim]) : FlatBox α :=
  { dim := b.outDim
    lo := Linear.forward (α := α) (inDim := b.inDim) (outDim := b.outDim) b.loAff x
    hi := Linear.forward (α := α) (inDim := b.inDim) (outDim := b.outDim) b.hiAff x }

/-- Enclosure of a real node value `v` by an affine bound `b` evaluated at the designated input `x`.

It holds when `b.inDim = ctx.inputDim` and the box `boundsEvalAt b x'` has width `v.n` and encloses
each entry of `v` (`Directed.RowEncloses`), for the cast `x'` of `x` along that equation. Every
bound of a well-formed CROWN certificate meets the dimension condition.
-/
def EnclosesAtInput (ctx : AffineCtx) (x : Tensor ℝ [ctx.inputDim])
    (b : FlatAffineBounds ℝ) (v : FlatTensor ℝ) : Prop :=
  ∃ h : b.inDim = ctx.inputDim,
    Directed.RowEncloses (boundsEvalAt (α := ℝ) b (castDimScalar (α := ℝ) h.symm x)) v.n
      fun i => getAtOrZero v.v [i]

end
end TorchLean.CROWN.Graph

namespace TorchLean.CROWN
open Cert
open TorchLean.Tensor
noncomputable section

/-- Scaling an input by a coefficient in `[0, 1]` never exceeds its ReLU. -/
theorem _root_.TorchLean.Activation.mul_le_relu (alpha input : Real)
    (alphaNonnegative : 0 ≤ alpha) (alphaAtMostOne : alpha ≤ 1) :
    alpha * input ≤ Activation.relu (α := Real) input := by
  by_cases inputNonpositive : input ≤ 0
  · have : alpha * input ≤ 0 :=
      mul_nonpos_of_nonneg_of_nonpos alphaNonnegative inputNonpositive
    simpa [Activation.relu_eq_max, max_eq_right inputNonpositive] using this
  · have inputNonnegative : 0 ≤ input := le_of_not_ge inputNonpositive
    simpa [Activation.relu_eq_max, max_eq_left inputNonnegative] using
      mul_le_of_le_one_left inputNonnegative alphaAtMostOne

/-- The alpha-CROWN scalar lower relaxation under-approximates ReLU throughout `[lower, upper]`. -/
theorem Cert.alphaRelaxLowerScalar_sound
    (lower upper alpha input : Real)
    (lowerBound : lower ≤ input) (upperBound : input ≤ upper)
    (alphaNonnegative : 0 ≤ alpha) (alphaAtMostOne : alpha ≤ 1) :
    let relaxation := alphaRelaxLowerScalar (α := Real) lower upper alpha
    relaxation.slope * input + relaxation.bias ≤
      Activation.relu (α := Real) input := by
  unfold alphaRelaxLowerScalar
  by_cases upperPositive : upper > 0
  · by_cases lowerPositive : lower > 0
    · have inputPositive : 0 < input := lt_of_lt_of_le lowerPositive lowerBound
      have inputNonnegative : 0 ≤ input := le_of_lt inputPositive
      simp [upperPositive, lowerPositive, Activation.relu_eq_max,
        max_eq_left inputNonnegative]
    · simpa [upperPositive, lowerPositive] using
        Activation.mul_le_relu alpha input alphaNonnegative alphaAtMostOne
  · have inputNonpositive : input ≤ 0 := le_trans upperBound (le_of_not_gt upperPositive)
    simp [upperPositive, Activation.relu_eq_max, max_eq_right inputNonpositive]

/-- An accepted inactive phase has nonpositive upper endpoint. -/
theorem Cert.upper_nonpos_of_inactive (l u : ℝ)
    (h : phaseConsistentScalar? (α := ℝ) l u ReLUPhase.inactive = some ()) :
    u ≤ 0 := by
  -- `inactive` checks `¬ (0 < u)` via the executable `if u > 0 then none else some ()`.
  unfold phaseConsistentScalar? at h
  by_cases hu : u > 0
  · simp [hu] at h
  · exact le_of_not_gt hu

/-- An accepted active phase has nonnegative lower endpoint. -/
theorem Cert.lower_nonneg_of_active (l u : ℝ)
    (h : phaseConsistentScalar? (α := ℝ) l u ReLUPhase.active = some ()) :
    0 ≤ l := by
  -- `active` checks `¬ (l < 0)` via `if l < 0 then none else some ()`.
  unfold phaseConsistentScalar? at h
  by_cases hl : l < 0
  · simp [hl] at h
  · exact le_of_not_gt hl

/-- For `x ∈ [l, u]` and a phase that `phaseConsistentScalar?` accepts on `[l, u]`, the upper
relaxation of that phase lies above `relu x`.

Both stable phases need the check. The inactive relaxation is the zero line, which lies above
`relu` on `[l, u]` because the check gives `u ≤ 0`. The active relaxation is the identity, which
lies above `relu` there because the check gives `0 ≤ l`. -/
theorem Cert.phaseRelaxUpperScalar_sound
    (l u x : ℝ) (hlx : l ≤ x) (hxu : x ≤ u) (ph : ReLUPhase)
    (hcons : phaseConsistentScalar? (α := ℝ) l u ph = some ()) :
    let rp := phaseRelaxUpperScalar (α := ℝ) l u ph
    Activation.relu (α := ℝ) x ≤ rp.slope * x + rp.bias := by
  cases ph with
  | inactive =>
      have hu0 : u ≤ 0 := upper_nonpos_of_inactive (l := l) (u := u) hcons
      have hxle : x ≤ 0 := le_trans hxu hu0
      simp [phaseRelaxUpperScalar, Activation.relu_eq_max, max_eq_right hxle]
  | active =>
      have hl0 : 0 ≤ l := lower_nonneg_of_active (l := l) (u := u) hcons
      have hxnonneg : 0 ≤ x := le_trans hl0 hlx
      simp [phaseRelaxUpperScalar, Activation.relu_eq_max, max_eq_left hxnonneg]
  | unstable =>
      simpa [phaseRelaxUpperScalar] using
        (ReLU.relaxScalar_sound (l := l) (u := u) (x := x) hlx hxu)

/-- For `x ∈ [l, u]`, a slope `a ∈ [0, 1]` and a phase that `phaseConsistentScalar?` accepts on
`[l, u]`, the lower relaxation of that phase lies below `relu x`.

The slope `a` is the α of α-CROWN. On an unstable neuron every slope in the unit interval gives a
valid lower line, so an optimizer may pick whichever one tightens the final bound. -/
theorem Cert.phaseRelaxLowerScalar_sound
    (l u a x : ℝ) (hlx : l ≤ x) (hxu : x ≤ u)
    (ha0 : 0 ≤ a) (ha1 : a ≤ 1)
    (ph : ReLUPhase) (hcons : phaseConsistentScalar? (α := ℝ) l u ph = some ()) :
    let rp := phaseRelaxLowerScalar (α := ℝ) l u a ph
    rp.slope * x + rp.bias ≤ Activation.relu (α := ℝ) x := by
  cases ph with
  | inactive =>
      have hu0 : u ≤ 0 := upper_nonpos_of_inactive (l := l) (u := u) hcons
      have hxle : x ≤ 0 := le_trans hxu hu0
      -- The relaxation is the zero line, and `relu x = 0` because `x ≤ 0`.
      simp [phaseRelaxLowerScalar, Activation.relu_eq_max, max_eq_right hxle]
  | active =>
      have hl0 : 0 ≤ l := lower_nonneg_of_active (l := l) (u := u) hcons
      have hxnonneg : 0 ≤ x := le_trans hl0 hlx
      simp [phaseRelaxLowerScalar, Activation.relu_eq_max, max_eq_left hxnonneg]
  | unstable =>
      simpa [phaseRelaxLowerScalar] using
        (alphaRelaxLowerScalar_sound
          (lower := l) (upper := u) (alpha := a) (input := x) hlx hxu ha0 ha1)

end
end TorchLean.CROWN

namespace TorchLean
open Tensor
open scoped BigOperators
open CROWN
open Cert
noncomputable section
open Graph
open CROWN.Directed (RowEncloses)

/-- Well-formedness condition for α vectors: each component lies in `[0,1]`. -/
def CROWN.Graph.AlphaOK (alpha : Array (Option (FlatTensor ℝ))) : Prop :=
  ∀ id : Nat, id < alpha.size →
    match alpha[id]! with
    | none => True
    | some a => ∀ i : Fin a.n, (0 : ℝ) ≤ getScalar a.v i ∧ getScalar a.v i ≤ (1 : ℝ)

/-- The box with endpoints `lo` and `hi` encloses the entries of `x` (`RowEncloses`) exactly when
every coordinate of `x` lies between those of `lo` and `hi`. -/
theorem CROWN.Graph.encloses_iff_getScalar {n : Nat}
    (lo hi x : Tensor ℝ [n]) :
    RowEncloses { dim := n, lo := lo, hi := hi } n (fun i => getAtOrZero x [i]) ↔
      ∀ i : Fin n, getScalar lo i ≤ getScalar x i ∧ getScalar x i ≤ getScalar hi i := by
  rw [Directed.rowEncloses_iff]
  simp only [getAtOrZero_eq_getScalar]
  rfl

/-- A linear layer with zero bias is a plain matrix-vector product.

The forward pass bounds a sum over all axes as a linear layer with a row of ones and a zero bias,
and `alpha?_encloses` removes that bias with this lemma. -/
theorem Linear.forward_zero_bias {m n : Nat}
    (W : Tensor ℝ [m, n])
    (x : Tensor ℝ [n]) :
    forward (α := ℝ)
        { weights := W
          bias := Tensor.full (α := ℝ) (.dim m .scalar) (0 : ℝ) } x
      =
      Tensor.contract W x _ _ [m] .matVec := by
  -- `Linear.forward` adds the bias to the matrix-vector product, so a zero bias drops out.
  simp [forward]

/-- `castDimScalar` is proof-irrelevant in its equality argument. -/
theorem CROWN.Graph.castDimScalar_proof_irrel {n n' : Nat}
    (h₁ h₂ : n = n') (t : Tensor ℝ [n]) :
    castDimScalar (α := ℝ) h₁ t = castDimScalar (α := ℝ) h₂ t :=
  rfl

/-- Casting the dimension of a tensor preserves the unit-interval range of its entries. -/
theorem CROWN.Graph.castDimScalar_unit_range {n n' : Nat} (h : n = n') (t : Tensor ℝ [n])
    (hr : ∀ i : Fin n, (0 : ℝ) ≤ getScalar t i ∧ getScalar t i ≤ (1 : ℝ)) :
    ∀ i : Fin n', (0 : ℝ) ≤ getScalar (castDimScalar (α := ℝ) h t) i ∧
      getScalar (castDimScalar (α := ℝ) h t) i ≤ (1 : ℝ) := by
  intro i
  rw [getScalar_castDimScalar]
  exact hr _

/-- A successful `Cert.getAff?` lookup is an in-bounds array read. -/
theorem CROWN.Cert.getElem!_of_getAff?_eq_some
    {cert : Array (Option (FlatAffineBounds ℝ))} {p : Nat}
    {xin : FlatAffineBounds ℝ} (h : CROWN.Cert.getAff? (α := ℝ) cert p = some xin) :
    cert[p]! = some xin := by
  by_cases hlt : p < cert.size
  · simpa [CROWN.Cert.getAff?, Array.getD, hlt] using h
  · simp [CROWN.Cert.getAff?, Array.getD, hlt] at h

/-- A successful `getAlpha?` lookup is an in-bounds array read. -/
theorem CROWN.Cert.getElem!_of_getAlpha?_eq_some {alpha : Array (Option (FlatTensor ℝ))} {id : Nat}
    {αv : FlatTensor ℝ} (h : CROWN.Cert.getAlpha? (α := ℝ) alpha id = some αv) :
    id < alpha.size ∧ alpha[id]! = some αv := by
  by_cases hlt : id < alpha.size
  · exact ⟨hlt, by simpa [CROWN.Cert.getAlpha?, Array.getD, hlt] using h⟩
  · simp [CROWN.Cert.getAlpha?, Array.getD, hlt] at h

/-- Under `AlphaOK`, an α vector read by `getAlpha?` has every entry in `[0, 1]`. -/
theorem CROWN.Cert.getAlpha?_unit_range {alpha : Array (Option (FlatTensor ℝ))} {id : Nat}
    {αv : FlatTensor ℝ} (halpha : AlphaOK (alpha := alpha))
    (h : CROWN.Cert.getAlpha? (α := ℝ) alpha id = some αv) :
    ∀ i : Fin αv.n, (0 : ℝ) ≤ getScalar αv.v i ∧ getScalar αv.v i ≤ (1 : ℝ) := by
  obtain ⟨hlt, hentry⟩ := getElem!_of_getAlpha?_eq_some h
  simpa [hentry] using halpha id hlt

/-- `Tensor.relu` commutes with `castDimScalar`. -/
theorem CROWN.Graph.castDimScalar_relu {n n' : Nat} (h : n = n') (t : Tensor ℝ [n]) :
    castDimScalar (α := ℝ) (n := n) (n' := n') h (Tensor.relu (α := ℝ) t)
      =
    Tensor.relu (α := ℝ) (castDimScalar (α := ℝ) (n := n) (n' := n') h t) := by
  cases h
  rfl

/-- Casting the output dimension of the bounds and the width of the value to a common `outDim'`
preserves `EnclosesAtInput`. -/
theorem CROWN.Graph.enclosesAtInput_castOut (ctx : AffineCtx) (x : Tensor ℝ [ctx.inputDim])
    (xin : FlatAffineBounds ℝ) (vp : FlatTensor ℝ) {outDim' : Nat}
    (hout : xin.outDim = outDim') (hvout : vp.n = outDim') :
    EnclosesAtInput ctx x xin vp →
      EnclosesAtInput ctx x
        { inDim := xin.inDim
          outDim := outDim'
          loAff := CROWN.Graph.castAffineOut (α := ℝ) (n := xin.inDim) (m := xin.outDim)
            (m' := outDim') hout xin.loAff
          hiAff := CROWN.Graph.castAffineOut (α := ℝ) (n := xin.inDim) (m := xin.outDim)
            (m' := outDim') hout xin.hiAff }
        { n := outDim', v := castDimScalar (α := ℝ) hvout vp.v } := by
  intro hpar
  obtain ⟨hinDim, henc⟩ := hpar
  obtain ⟨n, v⟩ := vp
  change n = outDim' at hvout
  -- Once `outDim'` and the width are identified with `xin.outDim`, the casts run along `rfl`, so
  -- the cast bounds are `xin` itself and the parent enclosure is the goal up to unfolding.
  subst hvout hout
  exact ⟨hinDim, henc⟩

/-- Entries of the positive part of a matrix: the entry where it is positive, else zero. -/
theorem CROWN.IBP.get2_matPos {m n : Nat}
    (W : Tensor ℝ [m, n]) (i : Fin m) (j : Fin n) :
    Tensor.get2 (CROWN.IBP.matPos (α := ℝ) (m := m) (n := n) W) i j =
      (if Tensor.get2 W i j > 0 then Tensor.get2 W i j else 0) := by
  simpa only [CROWN.IBP.matPos] using
    Tensor.get2_map (fun w : ℝ => if w > 0 then w else 0) W i j

/-- Entries of the negative part of a matrix: the entry where it is not positive, else zero. -/
theorem CROWN.IBP.get2_matNeg {m n : Nat}
    (W : Tensor ℝ [m, n]) (i : Fin m) (j : Fin n) :
    Tensor.get2 (CROWN.IBP.matNeg (α := ℝ) (m := m) (n := n) W) i j =
      (if Tensor.get2 W i j > 0 then 0 else Tensor.get2 W i j) := by
  simpa only [CROWN.IBP.matNeg] using
    Tensor.get2_map (fun w : ℝ => if w > 0 then 0 else w) W i j

/-- One term of the interval upper bound: `w * x` is at most `w⁺ * u + w⁻ * l`.

Over `x ∈ [l, u]` the product `w * x` is largest at `u` when `w > 0` and at `l` otherwise. Summed
over a row, this bound and its lower companion give `encloses_linear_signSplit`. -/
theorem CROWN.Graph.signSplit_term_upper (w l u x : ℝ) (hlx : l ≤ x) (hxu : x ≤ u) :
    w * x ≤ (if 0 < w then w else 0) * u + (if 0 < w then 0 else w) * l := by
  by_cases hw : 0 < w
  · have hw0 : 0 ≤ w := le_of_lt hw
    have : w * x ≤ w * u := mul_le_mul_of_nonneg_left hxu hw0
    simpa [hw, add_assoc, add_left_comm, add_comm] using this
  · have hw0 : w ≤ 0 := le_of_not_gt hw
    have : w * x ≤ w * l := mul_le_mul_of_nonpos_left hlx hw0
    simpa [hw, add_assoc, add_left_comm, add_comm] using this

/-- The lower-bound companion: `w⁺ * l + w⁻ * u` is at most `w * x`. -/
theorem CROWN.Graph.signSplit_term_lower (w l u x : ℝ) (hlx : l ≤ x) (hxu : x ≤ u) :
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
encloses `W x + b` for every `x` in the input box. The affine transfer through a linear layer
applies it to the box of the incoming affine bounds at the input point
(`enclosesAtInput_linear`). -/
theorem CROWN.Graph.encloses_linear_signSplit {m n : Nat}
    (W : Tensor ℝ [m, n])
    (b : Tensor ℝ [m])
    (lo hi x : Tensor ℝ [n])
    (hx : RowEncloses { dim := n, lo := lo, hi := hi } n fun i => getAtOrZero x [i]) :
    RowEncloses
      { dim := m
        lo :=
          ((Tensor.contract (CROWN.IBP.matPos (α := ℝ) (m := m) (n :=
                n) W) lo _ _ [m] .matVec) +
              (Tensor.contract (CROWN.IBP.matNeg (α := ℝ) (m := m) (n :=
                n) W) hi _ _ [m] .matVec)) +
            b
        hi :=
          ((Tensor.contract (CROWN.IBP.matPos (α := ℝ) (m := m) (n :=
                n) W) hi _ _ [m] .matVec) +
              (Tensor.contract (CROWN.IBP.matNeg (α := ℝ) (m := m) (n :=
                n) W) lo _ _ [m] .matVec)) +
            b }
      m fun i => getAtOrZero (Tensor.contract W x _ _ [m] .matVec + b) [i] := by
  classical
  have hx' := (encloses_iff_getScalar (lo := lo) (hi := hi) (x := x)).1 hx
  refine (encloses_iff_getScalar (n := m) (lo := _) (hi := _) (x := _)).2 ?_
  intro i
  -- Expand all mat-vec products into finite sums.
  have hW :
      Tensor.getScalar (Tensor.contract W x _ _ [m] .matVec) i =
        ∑ k : Fin n, (Tensor.get2 W i k) * (Tensor.getScalar x k) := by
    simpa using (Tensor.getScalar_contract_matVec_eq_sum (A := W) (v := x) (i := i))
  have hPos_lo :
      Tensor.getScalar
          (Tensor.contract (CROWN.IBP.matPos (α := ℝ) (m := m) (n := n) W) lo _ _ [m]
              .matVec) i =
        ∑ k : Fin n,
          (Tensor.get2 (CROWN.IBP.matPos (α := ℝ) (m := m) (n := n) W) i k) *
            (Tensor.getScalar lo k) := by
    simpa using
      (Tensor.getScalar_contract_matVec_eq_sum
        (A := CROWN.IBP.matPos (α := ℝ) (m := m) (n := n) W) (v := lo) (i := i))
  have hNeg_hi :
      Tensor.getScalar
          (Tensor.contract (CROWN.IBP.matNeg (α := ℝ) (m := m) (n := n) W) hi _ _ [m]
              .matVec) i =
        ∑ k : Fin n,
          (Tensor.get2 (CROWN.IBP.matNeg (α := ℝ) (m := m) (n := n) W) i k) *
            (Tensor.getScalar hi k) := by
    simpa using
      (Tensor.getScalar_contract_matVec_eq_sum
        (A := CROWN.IBP.matNeg (α := ℝ) (m := m) (n := n) W) (v := hi) (i := i))
  have hPos_hi :
      Tensor.getScalar
          (Tensor.contract (CROWN.IBP.matPos (α := ℝ) (m := m) (n := n) W) hi _ _ [m]
              .matVec) i =
        ∑ k : Fin n,
          (Tensor.get2 (CROWN.IBP.matPos (α := ℝ) (m := m) (n := n) W) i k) *
            (Tensor.getScalar hi k) := by
    simpa using
      (Tensor.getScalar_contract_matVec_eq_sum
        (A := CROWN.IBP.matPos (α := ℝ) (m := m) (n := n) W) (v := hi) (i := i))
  have hNeg_lo :
      Tensor.getScalar
          (Tensor.contract (CROWN.IBP.matNeg (α := ℝ) (m := m) (n := n) W) lo _ _ [m]
              .matVec) i =
        ∑ k : Fin n,
          (Tensor.get2 (CROWN.IBP.matNeg (α := ℝ) (m := m) (n := n) W) i k) *
            (Tensor.getScalar lo k) := by
    simpa using
      (Tensor.getScalar_contract_matVec_eq_sum
        (A := CROWN.IBP.matNeg (α := ℝ) (m := m) (n := n) W) (v := lo) (i := i))

  have hUpperTerm :
      ∀ k : Fin n,
        (Tensor.get2 W i k) * (Tensor.getScalar x k) ≤
          (Tensor.get2 (CROWN.IBP.matPos (α := ℝ) (m := m) (n := n) W) i k) *
            (Tensor.getScalar hi k) +
          (Tensor.get2 (CROWN.IBP.matNeg (α := ℝ) (m := m) (n := n) W) i k) *
            (Tensor.getScalar lo k) := by
    intro k
    have hk := hx' k
    have hpos := IBP.get2_matPos (W := W) i k
    have hneg := IBP.get2_matNeg (W := W) i k
    have := signSplit_term_upper (w := Tensor.get2 W i k) (l := Tensor.getScalar lo k)
      (u := Tensor.getScalar hi k)
      (x := Tensor.getScalar x k) hk.1 hk.2
    simpa [hpos, hneg, mul_add, add_mul, add_assoc, add_left_comm, add_comm] using this

  have hLowerTerm :
      ∀ k : Fin n,
        (Tensor.get2 (CROWN.IBP.matPos (α := ℝ) (m := m) (n := n) W) i k) *
          (Tensor.getScalar lo k) +
          (Tensor.get2 (CROWN.IBP.matNeg (α := ℝ) (m := m) (n := n) W) i k) *
            (Tensor.getScalar hi k)
            ≤ (Tensor.get2 W i k) * (Tensor.getScalar x k) := by
    intro k
    have hk := hx' k
    have hpos := IBP.get2_matPos (W := W) i k
    have hneg := IBP.get2_matNeg (W := W) i k
    have := signSplit_term_lower (w := Tensor.get2 W i k) (l := Tensor.getScalar lo k)
      (u := Tensor.getScalar hi k)
      (x := Tensor.getScalar x k) hk.1 hk.2
    simpa [hpos, hneg, mul_add, add_mul, add_assoc, add_left_comm, add_comm] using this

  have hUpperSum :
      (∑ k : Fin n, (Tensor.get2 W i k) * (Tensor.getScalar x k)) ≤
        (∑ k : Fin n,
          ((Tensor.get2 (CROWN.IBP.matPos (α := ℝ) (m := m) (n := n) W) i k) *
            (Tensor.getScalar hi k) +
           (Tensor.get2 (CROWN.IBP.matNeg (α := ℝ) (m := m) (n := n) W) i k) *
             (Tensor.getScalar lo k))) := by
    classical
    simpa using (Finset.sum_le_sum (s := Finset.univ) (fun k _ => hUpperTerm k))

  have hLowerSum :
      (∑ k : Fin n,
        ((Tensor.get2 (CROWN.IBP.matPos (α := ℝ) (m := m) (n := n) W) i k) *
          (Tensor.getScalar lo k) +
         (Tensor.get2 (CROWN.IBP.matNeg (α := ℝ) (m := m) (n := n) W) i k) *
           (Tensor.getScalar hi k)))
          ≤ (∑ k : Fin n, (Tensor.get2 W i k) * (Tensor.getScalar x k)) := by
    classical
    simpa using (Finset.sum_le_sum (s := Finset.univ) (fun k _ => hLowerTerm k))

  have hlo :
      Tensor.getScalar
          (((Tensor.contract (CROWN.IBP.matPos (α := ℝ) (m := m) (n :=
                n) W) lo _ _ [m] .matVec) +
              (Tensor.contract (CROWN.IBP.matNeg (α := ℝ) (m := m) (n :=
                n) W) hi _ _ [m] .matVec)) +
            b) i
        ≤
  Tensor.getScalar (Tensor.contract W x _ _ [m] .matVec + b) i := by
    -- Rewrite `sum (a+b)` into `sum a + sum b` to match `getScalar` expansions.
    have hLowerSum' :
        (∑ k : Fin n,
            (Tensor.get2 (CROWN.IBP.matPos (α := ℝ) (m := m) (n := n) W) i k) *
              (Tensor.getScalar lo k)) +
          (∑ k : Fin n,
            (Tensor.get2 (CROWN.IBP.matNeg (α := ℝ) (m := m) (n := n) W) i k) *
              (Tensor.getScalar hi k))
            ≤ (∑ k : Fin n, (Tensor.get2 W i k) * (Tensor.getScalar x k)) := by
      simpa only [Finset.sum_add_distrib] using hLowerSum
    have hLowerSum_swapped :
        (∑ k : Fin n,
            (Tensor.get2 (CROWN.IBP.matNeg (α := ℝ) (m := m) (n := n) W) i k) *
              (Tensor.getScalar hi k)) +
          (∑ k : Fin n,
            (Tensor.get2 (CROWN.IBP.matPos (α := ℝ) (m := m) (n := n) W) i k) *
              (Tensor.getScalar lo k))
          ≤ (∑ k : Fin n, (Tensor.get2 W i k) * (Tensor.getScalar x k)) := by
      simpa [add_comm, add_left_comm, add_assoc] using hLowerSum'
    simp [getScalar_add, hW, hPos_lo, hNeg_hi, hLowerSum_swapped, add_comm]

  have hhi :
      Tensor.getScalar (Tensor.contract W x _ _ [m] .matVec + b) i
        ≤
      Tensor.getScalar
          (((Tensor.contract (CROWN.IBP.matPos (α := ℝ) (m := m) (n :=
                n) W) hi _ _ [m] .matVec) +
              (Tensor.contract (CROWN.IBP.matNeg (α := ℝ) (m := m) (n :=
                n) W) lo _ _ [m] .matVec)) +
            b) i := by
    have hUpperSum' :
        (∑ k : Fin n, (Tensor.get2 W i k) * (Tensor.getScalar x k)) ≤
          (∑ k : Fin n,
              (Tensor.get2 (CROWN.IBP.matPos (α := ℝ) (m := m) (n := n) W) i k) *
                (Tensor.getScalar hi k)) +
            (∑ k : Fin n,
              (Tensor.get2 (CROWN.IBP.matNeg (α := ℝ) (m := m) (n := n) W) i k) *
                (Tensor.getScalar lo k)) := by
      simpa only [Finset.sum_add_distrib] using hUpperSum
    have hUpperSum_swapped :
        (∑ k : Fin n, (Tensor.get2 W i k) * (Tensor.getScalar x k)) ≤
          (∑ k : Fin n,
              (Tensor.get2 (CROWN.IBP.matNeg (α := ℝ) (m := m) (n := n) W) i k) *
                (Tensor.getScalar lo k)) +
            (∑ k : Fin n,
              (Tensor.get2 (CROWN.IBP.matPos (α := ℝ) (m := m) (n := n) W) i k) *
                (Tensor.getScalar hi k)) := by
      simpa [add_comm, add_left_comm, add_assoc] using hUpperSum'
    simp [getScalar_add, hW, hPos_hi, hNeg_lo, hUpperSum_swapped, add_comm]

  exact ⟨hlo, hhi⟩

/-- The upper ReLU relaxation `relaxScalar l u` has nonnegative slope in each of its three cases.

Multiplying both sides of an inequality by a nonnegative slope keeps its direction, which the
transfer proofs need (`relaxScalar_sound_of_le`). -/
theorem CROWN.ReLU.relaxScalar_slope_nonneg (l u : ℝ) :
    0 ≤ (CROWN.ReLU.relaxScalar (α := ℝ) l u).slope := by
  unfold CROWN.ReLU.relaxScalar
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

/-- The α lower relaxation `alphaRelaxLowerScalar l u a` has nonnegative slope when `a ≥ 0`. -/
theorem CROWN.Cert.alphaRelaxLowerScalar_slope_nonneg (l u a : ℝ) (ha0 : 0 ≤ a) :
    0 ≤ (alphaRelaxLowerScalar (α := ℝ) l u a).slope := by
  unfold alphaRelaxLowerScalar
  by_cases hu : u > 0
  · by_cases hlpos : l > 0
    · simp [hu, hlpos]
    · simp [hu, hlpos, ha0]
  · simp [hu]

/-- Nonnegative slope for the upper relaxation under any phase constraint. -/
theorem CROWN.Cert.phaseRelaxUpperScalar_slope_nonneg (l u : ℝ) (ph : ReLUPhase) :
    0 ≤ (phaseRelaxUpperScalar (α := ℝ) l u ph).slope := by
  cases ph <;> simp [phaseRelaxUpperScalar, ReLU.relaxScalar_slope_nonneg]

/-- Nonnegative slope for the lower relaxation under any phase constraint and `a ≥ 0`. -/
theorem CROWN.Cert.phaseRelaxLowerScalar_slope_nonneg (l u a : ℝ) (ph : ReLUPhase) (ha0 : 0 ≤ a) :
    0 ≤ (phaseRelaxLowerScalar (α := ℝ) l u a ph).slope := by
  cases ph <;> simp [phaseRelaxLowerScalar, alphaRelaxLowerScalar_slope_nonneg, ha0]

/-- The default α vector is a `0`/`1` vector, hence lies in `[0, 1]` coordinatewise. -/
theorem CROWN.Cert.defaultAlphaVec_range {n : Nat}
    (lo hi : Tensor ℝ [n]) :
    ∀ i : Fin n, (0 : ℝ) ≤ getScalar (defaultAlphaVec (α := ℝ) (n := n) lo hi) i ∧
      getScalar (defaultAlphaVec (α := ℝ) (n := n) lo hi) i ≤ (1 : ℝ) := by
  classical
  intro i
  by_cases h : getScalar hi i > -getScalar lo i
  · simp [defaultAlphaVec, h]
  · simp [defaultAlphaVec, h]

/-- Coordinate `i` of `relaxVector lo hi` is `relaxScalar` of the endpoints at `i`, so the scalar
soundness lemmas above apply to whole layers. -/
theorem CROWN.ReLU.getScalar_relaxVector {n : Nat}
    (lo hi : Tensor ℝ [n]) (i : Fin n) :
    getScalar (CROWN.ReLU.relaxVector (α := ℝ) (n := n) lo hi) i =
      CROWN.ReLU.relaxScalar (α := ℝ) (getScalar lo i)
        (getScalar hi i) := by
  simp [CROWN.ReLU.relaxVector]

/-- Same coordinatewise reading for the α-parameterized lower relaxation. -/
theorem CROWN.Cert.getScalar_alphaRelaxLowerVec {n : Nat}
    (lo hi αv : Tensor ℝ [n]) (i : Fin n) :
    getScalar (alphaRelaxLowerVec (α := ℝ) (n := n) lo hi αv) i =
      alphaRelaxLowerScalar (α := ℝ) (getScalar lo i) (getScalar hi i) (getScalar αv i) := by
  simp [alphaRelaxLowerVec]

/-- Propagating an affine form through a ReLU relaxation acts at each coordinate as
`slope * value + bias`, with the slope and bias of the relaxation at that coordinate.

`propagateAffine` scales each row of the matrix by its slope and maps the constant through the
relaxation, and this lemma brings that whole-matrix operation down to the per-neuron reasoning of
the soundness proofs. -/
theorem Linear.forward_propagateAffine
    {inDim hidDim : Nat}
    (relax : Tensor (CROWN.ReLURelax ℝ) [hidDim])
    (aff : Linear ℝ inDim hidDim)
    (x : Tensor ℝ [inDim]) (i : Fin hidDim) :
    getScalar
        (Linear.forward (α := ℝ) (inDim := inDim) (outDim := hidDim)
          (CROWN.ReLU.propagateAffine (α := ℝ)
            (inDim := inDim) (hidDim := hidDim) relax aff) x) i
      =
      (let rp := getScalar relax i
       rp.slope *
          getScalar
            (Linear.forward (α := ℝ) (inDim := inDim) (outDim := hidDim) aff x)
            i +
        rp.bias) := by
  classical
  let rp := getScalar relax i
  let propagated :=
    CROWN.ReLU.propagateAffine (α := ℝ)
      (inDim := inDim) (hidDim := hidDim) relax aff
  have hget2 (j : Fin inDim) :
      Tensor.get2 propagated.weights i j = Tensor.get2 aff.weights i j * rp.slope := by
    simp [propagated, rp, CROWN.ReLU.propagateAffine]
  have hc :
      getScalar propagated.bias i = rp.slope * getScalar aff.bias i + rp.bias := by
    simp [propagated, rp, CROWN.ReLU.propagateAffine]
  have hscale :
      (∑ k : Fin inDim, (Tensor.get2 aff.weights i k * rp.slope) * getScalar x k) =
        rp.slope * (∑ k : Fin inDim, Tensor.get2 aff.weights i k * getScalar x k) := by
    calc
      (∑ k : Fin inDim, (Tensor.get2 aff.weights i k * rp.slope) * getScalar x k) =
          ∑ k : Fin inDim, rp.slope * (Tensor.get2 aff.weights i k * getScalar x k) := by
            apply Finset.sum_congr rfl
            intro k _
            ring
      _ = rp.slope * (∑ k : Fin inDim, Tensor.get2 aff.weights i k * getScalar x k) := by
            exact (Finset.mul_sum _ _ _).symm
  change getScalar
      (Linear.forward (α := ℝ) propagated x) i =
    rp.slope * getScalar (Linear.forward (α := ℝ) aff x) i + rp.bias
  simp only [Linear.forward]
  rw [congrFun (Tensor.getScalar_add _ _) i,
    congrFun (Tensor.getScalar_add _ _) i]
  rw [Tensor.getScalar_contract_matVec_eq_sum
      (A := propagated.weights) (v := x) (i := i),
    Tensor.getScalar_contract_matVec_eq_sum
      (A := aff.weights) (v := x) (i := i)]
  simp_rw [hget2]
  rw [hc, hscale]
  ring

/-- Unpack a successful `phaseRelaxVec?`: the phase array has the right length, and every coordinate
carries a phase that is consistent with its interval and whose two relaxations are the scalar ones.

`phaseRelaxVec?` returns only the relaxations, and this lemma recovers the phase of each coordinate
for the soundness proof to case on. -/
theorem CROWN.Cert.phaseRelaxVec?_some_getScalar {n : Nat}
    (lo hi αv : Tensor ℝ [n]) (phases : Array Int)
    (relaxLo relaxHi : Tensor (CROWN.ReLURelax ℝ) [n])
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
    cases hph : ReLUPhase.ofInt? (phaseAt phases (↑i)) with
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

/-- Matrix-vector multiplication is additive in the matrix. -/
theorem CROWN.Graph.mat_vec_add_matrix {m n : Nat}
    (A B : Tensor ℝ [m, n])
    (x : Tensor ℝ [n]) :
    Tensor.contract (A + B) x _ _ [m] .matVec =
      Tensor.contract A x _ _ [m] .matVec +
        Tensor.contract B x _ _ [m] .matVec := by
  classical
  apply Tensor.ext_vector
  intro i
  rw [congrFun (Tensor.getScalar_add _ _) i]
  simp only [Tensor.getScalar_contract_matVec_eq_sum]
  simp [Tensor.add_def, add_mul, Finset.sum_add_distrib]

/-- The zero matrix sends every vector to zero. `CROWN.Graph.boundsEvalAt_boundsConst` applies this
to the zero coefficient matrices of `boundsConst`. -/
theorem CROWN.Graph.contract_matVec_full_zero {m n : Nat}
    (x : Tensor ℝ [n]) :
    Tensor.contract (Tensor.full (α := ℝ) (.dim m (.dim n .scalar)) (0 : ℝ)) x _ _ [m] .matVec =
      Tensor.full (α := ℝ) (.dim m .scalar) (0 : ℝ) := by
  classical
  apply Tensor.ext_vector
  intro i
  -- Expand the matrix-vector coordinate as a finite sum, whose terms are all zero.
  rw [Tensor.getScalar_contract_matVec_eq_sum
    (A := Tensor.full (α := ℝ) (.dim m (.dim n .scalar)) (0 : ℝ)) (v := x) (i := i)]
  simp

/-- The identity affine certificate acts as the identity on inputs. -/
theorem CROWN.Graph.contract_matVec_affIdentity {n : Nat}
    (x : Tensor ℝ [n]) :
    Tensor.contract (Graph.affIdentity (α := ℝ) n).weights x _ _ [n] .matVec = x := by
  classical
  apply Tensor.ext_vector
  intro i
  rw [Tensor.getScalar_contract_matVec_eq_sum
    (A := (Graph.affIdentity (α := ℝ) n).weights) (v := x) (i := i)]
  simp [Graph.affIdentity, Tensor.get2_identity]

/-- The identity bounds certificate evaluates to the degenerate box `[x, x]`.

It is the base case of every certificate chain, in which the input bounds itself. -/
theorem CROWN.Graph.boundsEvalAt_boundsIdentity {n : Nat} (x : Tensor ℝ [n]) :
    boundsEvalAt (α := ℝ) (Graph.boundsIdentity (α := ℝ) n) x = { dim := n, lo := x, hi := x } := by
  classical
  have hMat :
      Tensor.contract (Graph.affIdentity (α := ℝ) n).weights x _ _ [n] .matVec = x :=
    contract_matVec_affIdentity (n := n) x
  have hC :
      (Graph.affIdentity (α := ℝ) n).bias = Tensor.full (α := ℝ) (.dim n .scalar) (0 : ℝ) := by
    simp [Graph.affIdentity]
  ext <;> simp [boundsEvalAt, Graph.boundsIdentity, Linear.forward, hMat, hC]

/-- A constant bounds certificate evaluates to its own endpoints, ignoring the input. -/
theorem CROWN.Graph.boundsEvalAt_boundsConst {inDim outDim : Nat}
    (lo hi : Tensor ℝ [outDim]) (x : Tensor ℝ [inDim]) :
    boundsEvalAt (α := ℝ) (Graph.boundsConst (α := ℝ) inDim outDim lo hi) x =
      { dim := outDim
        lo := lo
        hi := hi } := by
  classical
  ext <;> simp [boundsEvalAt, Graph.boundsConst, Linear.forward, contract_matVec_full_zero]

/-- Evaluating the composition of a linear layer with two incoming affine forms distributes over the
pair: matrices compose, constants are pushed through, and the bias is added once at the end. -/
theorem Linear.forward_linear_pair
    {inDim n m : Nat}
    (W₁ W₂ : Tensor ℝ [m, n])
    (aff₁ aff₂ : Linear ℝ inDim n)
    (b : Tensor ℝ [m])
    (x : Tensor ℝ [inDim]) :
    Linear.forward (α := ℝ)
        { weights := (Tensor.matmul W₁ aff₁.weights + Tensor.matmul W₂ aff₂.weights)
          bias := (Tensor.contract W₁ aff₁.bias _ _ [m] .matVec +
                Tensor.contract W₂ aff₂.bias _ _ [m] .matVec) + b } x =
      (Tensor.contract W₁ (Linear.forward (α := ℝ) aff₁ x) _ _ [m] .matVec +
          Tensor.contract W₂ (Linear.forward (α := ℝ) aff₂ x) _ _ [m] .matVec) +
        b := by
  simp only [Linear.forward, mat_vec_add_matrix, Tensor.contract_matVec_assoc,
    Tensor.contract_matVec_add]
  rw [(add_assoc (a := _) (b := _) (c := b)).symm]
  apply congrArg (fun z => z + b)
  exact add_add_add_comm _ _ _ _

/-- Evaluating at `x` the bounds that `propagateLinearBounds` builds from an incoming affine bound
gives the sign-split box of the evaluated endpoints.

With it, the affine transfer reduces to the interval rule that `encloses_linear_signSplit` proves
sound. The incoming bound is cast along `hout`, because its width equals `n` only up to a proof. -/
theorem CROWN.Graph.boundsEvalAt_linear
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
          let l := Linear.forward (α := ℝ) (inDim := xB.inDim) (outDim := n)
            (Graph.castAffineOut (α := ℝ) (n := xB.inDim) (m := xB.outDim) (m' := n) hout
              xB.loAff) x
          let u := Linear.forward (α := ℝ) (inDim := xB.inDim) (outDim := n)
            (Graph.castAffineOut (α := ℝ) (n := xB.inDim) (m := xB.outDim) (m' := n) hout
              xB.hiAff) x
          ((Tensor.contract (CROWN.IBP.matPos (α := ℝ) (m := m) (n :=
                n) W) l _ _ [m] .matVec) +
              (Tensor.contract (CROWN.IBP.matNeg (α := ℝ) (m := m) (n :=
                n) W) u _ _ [m] .matVec)) +
            b
        hi :=
          let l := Linear.forward (α := ℝ) (inDim := xB.inDim) (outDim := n)
            (Graph.castAffineOut (α := ℝ) (n := xB.inDim) (m := xB.outDim) (m' := n) hout
              xB.loAff) x
          let u := Linear.forward (α := ℝ) (inDim := xB.inDim) (outDim := n)
            (Graph.castAffineOut (α := ℝ) (n := xB.inDim) (m := xB.outDim) (m' := n) hout
              xB.hiAff) x
          ((Tensor.contract (CROWN.IBP.matPos (α := ℝ) (m := m) (n :=
                n) W) u _ _ [m] .matVec) +
              (Tensor.contract (CROWN.IBP.matNeg (α := ℝ) (m := m) (n :=
                n) W) l _ _ [m] .matVec)) +
            b } := by
  classical
  let xLo :=
    Graph.castAffineOut (α := ℝ) (n := xB.inDim) (m := xB.outDim) (m' := n) hout xB.loAff
  let xHi :=
    Graph.castAffineOut (α := ℝ) (n := xB.inDim) (m := xB.outDim) (m' := n) hout xB.hiAff
  let Wpos := CROWN.IBP.matPos (α := ℝ) (m := m) (n := n) W
  let Wneg := CROWN.IBP.matNeg (α := ℝ) (m := m) (n := n) W
  have hlo := Linear.forward_linear_pair Wpos Wneg xLo xHi b x
  have hhi := Linear.forward_linear_pair Wpos Wneg xHi xLo b x
  unfold boundsEvalAt Graph.propagateLinearBounds
  dsimp only
  refine FlatBox.ext rfl (heq_of_eq ?_) (heq_of_eq ?_)
  · exact hlo
  · exact hhi

end
end TorchLean

namespace TorchLean.CROWN
open Tensor
open Graph
open Cert
open CROWN.Directed (RowEncloses)

/-- The constant affine bounds of a box `B0` enclose a flat value `v` at every input point when
`B0` encloses `v`. -/
theorem Graph.enclosesAtInput_boundsConst {ctx : AffineCtx} {x : Tensor ℝ [ctx.inputDim]}
    {B0 : FlatBox ℝ} {v : FlatTensor ℝ} (hEnc : RowEncloses B0 v.n fun i => getAtOrZero v.v [i]) :
    EnclosesAtInput ctx x (Graph.boundsConst (α := ℝ) ctx.inputDim B0.dim B0.lo B0.hi) v := by
  refine ⟨rfl, ?_⟩
  have hBoxEval :
      boundsEvalAt (α := ℝ) (Graph.boundsConst (α := ℝ) ctx.inputDim B0.dim B0.lo B0.hi)
        (castDimScalar (α := ℝ) rfl x) = B0 := by
    cases B0
    exact boundsEvalAt_boundsConst _ _ _
  exact (congrArg (fun B => RowEncloses B v.n fun i => getAtOrZero v.v [i]) hBoxEval).mpr hEnc

/-- If `xin` encloses `vp` at `x`, the box of `xin` cast to width `n` and evaluated at the cast
input encloses the entries of `vp` cast to width `n`. It is `enclosesAtInput_castOut` unpacked to
one box, the form that the linear and ReLU transfers use. -/
theorem Graph.encloses_castOut {ctx : AffineCtx} {x : Tensor ℝ [ctx.inputDim]}
    {xin : FlatAffineBounds ℝ} {vp : FlatTensor ℝ} {n : Nat}
    (hout : xin.outDim = n) (hvn : vp.n = n) (hinDim : xin.inDim = ctx.inputDim)
    (hpar : EnclosesAtInput ctx x xin vp) :
    RowEncloses
      { dim := n
        lo := Linear.forward (α := ℝ) (inDim := xin.inDim) (outDim := n)
          (Graph.castAffineOut (α := ℝ) (n := xin.inDim) (m := xin.outDim) (m' := n) hout xin.loAff)
          (castDimScalar (α := ℝ) hinDim.symm x)
        hi := Linear.forward (α := ℝ) (inDim := xin.inDim) (outDim := n)
          (Graph.castAffineOut (α := ℝ) (n := xin.inDim) (m := xin.outDim) (m' := n) hout xin.hiAff)
          (castDimScalar (α := ℝ) hinDim.symm x) }
      n fun i => getAtOrZero (castDimScalar (α := ℝ) hvn vp.v) [i] := by
  obtain ⟨_, henc⟩ := enclosesAtInput_castOut ctx x xin vp hout hvn hpar
  exact henc

/-- Sign-splitting transfer through `y = W x + bv`: if the affine bounds `xin` enclose the parent
value `vp` at `x`, then `propagateLinearBounds W bv xin hout` encloses `W · vp + bv`. -/
theorem Graph.enclosesAtInput_linear {ctx : AffineCtx} {x : Tensor ℝ [ctx.inputDim]}
    {xin : FlatAffineBounds ℝ} {vp : FlatTensor ℝ} {n m : Nat}
    (W : Tensor ℝ [m, n]) (bv : Tensor ℝ [m])
    (hout : xin.outDim = n) (hvIn : vp.n = n)
    (hpar : EnclosesAtInput ctx x xin vp) :
    EnclosesAtInput ctx x
      (Graph.propagateLinearBounds (α := ℝ) (n := n) (m := m) W bv xin hout)
      { n := m
        v := Linear.forward (α := ℝ) { weights := W, bias := bv }
          (castDimScalar (α := ℝ) hvIn vp.v) } := by
  have hinDim := hpar.1
  have hxCast := encloses_castOut hout hvIn hinDim hpar
  have hy := encloses_linear_signSplit (m := m) (n := n) W bv _ _ _ hxCast
  refine ⟨hinDim, ?_⟩
  have hbox := boundsEvalAt_linear W bv xin hout
    (castDimScalar (α := ℝ) hinDim.symm x)
  exact (congrArg (fun B => RowEncloses B m fun i => getAtOrZero (Linear.forward (α := ℝ)
    { weights := W, bias := bv } (castDimScalar (α := ℝ) hvIn vp.v)) [i]) hbox).mpr hy

/-- The α lower relaxation on `[l, u]`, evaluated at a lower estimate `lAff ≤ z`, lies below
`relu z`. -/
theorem Cert.alphaRelaxLowerScalar_sound_of_le (l u a lAff z : ℝ) (hl : l ≤ z) (hu : z ≤ u)
    (ha0 : 0 ≤ a) (ha1 : a ≤ 1) (hlz : lAff ≤ z) :
    (alphaRelaxLowerScalar (α := ℝ) l u a).slope * lAff +
        (alphaRelaxLowerScalar (α := ℝ) l u a).bias ≤
      Activation.relu (α := ℝ) z := by
  have hslope := alphaRelaxLowerScalar_slope_nonneg l u a ha0
  have hsound := CROWN.Cert.alphaRelaxLowerScalar_sound l u a z hl hu ha0 ha1
  exact le_trans (add_le_add (mul_le_mul_of_nonneg_left hlz hslope) le_rfl) hsound

/-- `relu z` lies below the CROWN upper relaxation on `[l, u]` evaluated at an upper estimate
`z ≤ uAff`. -/
theorem ReLU.relaxScalar_sound_of_le (l u uAff z : ℝ) (hl : l ≤ z) (hu : z ≤ u) (hzu : z ≤ uAff) :
    Activation.relu (α := ℝ) z ≤
      (ReLU.relaxScalar (α := ℝ) l u).slope * uAff +
        (ReLU.relaxScalar (α := ℝ) l u).bias := by
  have hslope := relaxScalar_slope_nonneg l u
  have hsound := CROWN.ReLU.relaxScalar_sound l u z hl hu
  exact le_trans hsound (add_le_add (mul_le_mul_of_nonneg_left hzu hslope) le_rfl)

/-- Componentwise ReLU bound: if `z` lies in both the affine estimate box and the IBP box, and the
α vector lies in `[0, 1]`, then the propagated lower and upper affine forms bracket `relu z` at
every index. -/
theorem Graph.alphaRelax_getScalar_bounds {inDim n : Nat} (lo hi αt z : Tensor ℝ [n])
    (xLo xHi : Linear ℝ inDim n) (x' : Tensor ℝ [inDim])
    (hαrange : ∀ i : Fin n, (0 : ℝ) ≤ getScalar αt i ∧ getScalar αt i ≤ 1)
    (hzAff : RowEncloses
      { dim := n, lo := Linear.forward (α := ℝ) xLo x', hi := Linear.forward (α := ℝ) xHi x' } n
      fun j => getAtOrZero z [j])
    (hzIbp : RowEncloses { dim := n, lo := lo, hi := hi } n fun j => getAtOrZero z [j])
    (i : Fin n) :
    getScalar (Linear.forward (α := ℝ)
        (ReLU.propagateAffine (α := ℝ) (alphaRelaxLowerVec (α := ℝ) lo hi αt) xLo)
        x') i ≤ getScalar (Tensor.relu (α := ℝ) z) i ∧
      getScalar (Tensor.relu (α := ℝ) z) i ≤
        getScalar (Linear.forward (α := ℝ)
          (ReLU.propagateAffine (α := ℝ) (ReLU.relaxVector (α := ℝ) lo hi)
            xHi) x') i := by
  obtain ⟨hzLo, hzHi⟩ := (encloses_iff_getScalar _ _ _).1 hzAff i
  obtain ⟨hzIlo, hzIhi⟩ := (encloses_iff_getScalar _ _ _).1 hzIbp i
  simp only [Linear.forward_propagateAffine, getScalar_alphaRelaxLowerVec,
    ReLU.getScalar_relaxVector, Tensor.relu, Tensor.getScalar_map]
  exact ⟨alphaRelaxLowerScalar_sound_of_le _ _ _ _ _ hzIlo hzIhi (hαrange i).1 (hαrange i).2 hzLo,
    ReLU.relaxScalar_sound_of_le _ _ _ _ hzIlo hzIhi hzHi⟩

/-- ReLU transfer core: if `xin` encloses the parent value `vp` at `x`, the IBP box `preB`
encloses `vp`, and the α vector lies in `[0, 1]`, then propagating `xin` through the α lower and
CROWN upper relaxations encloses `relu vp`. -/
theorem Graph.enclosesAtInput_relu_propagate {ctx : AffineCtx} {x : Tensor ℝ [ctx.inputDim]}
    {xin : FlatAffineBounds ℝ} {vp : FlatTensor ℝ} {preB : FlatBox ℝ} (αt : Tensor ℝ [preB.dim])
    (hαrange : ∀ i : Fin preB.dim, (0 : ℝ) ≤ getScalar αt i ∧ getScalar αt i ≤ 1)
    (hout : xin.outDim = preB.dim)
    (hpar : EnclosesAtInput ctx x xin vp)
    (hEncIbp : RowEncloses preB vp.n fun i => getAtOrZero vp.v [i]) :
    EnclosesAtInput ctx x
      { inDim := xin.inDim
        outDim := preB.dim
        loAff := ReLU.propagateAffine (α := ℝ) (inDim := xin.inDim)
          (hidDim := preB.dim) (alphaRelaxLowerVec (α := ℝ) (n := preB.dim) preB.lo preB.hi αt)
          (Graph.castAffineOut (α := ℝ) (n := xin.inDim) (m := xin.outDim) (m' := preB.dim) hout
            xin.loAff)
        hiAff := ReLU.propagateAffine (α := ℝ) (inDim := xin.inDim)
          (hidDim := preB.dim)
          (ReLU.relaxVector (α := ℝ) (n := preB.dim) preB.lo preB.hi)
          (Graph.castAffineOut (α := ℝ) (n := xin.inDim) (m := xin.outDim) (m' := preB.dim) hout
            xin.hiAff) }
      { n := vp.n, v := Tensor.relu (α := ℝ) vp.v } := by
  obtain ⟨hinDim, -⟩ := id hpar
  have hdimIbp : preB.dim = vp.n := hEncIbp.1
  have hzAff := encloses_castOut hout hdimIbp.symm hinDim hpar
  have hzIbp : RowEncloses { dim := preB.dim, lo := preB.lo, hi := preB.hi } preB.dim
      fun i => getAtOrZero (castDimScalar (α := ℝ) hdimIbp.symm vp.v) [i] := by
    obtain ⟨n, v⟩ := vp
    obtain rfl : preB.dim = n := hdimIbp
    rw [castDimScalar_self]
    exact hEncIbp
  refine ⟨hinDim, ?_⟩
  obtain ⟨n, v⟩ := vp
  obtain rfl : preB.dim = n := hdimIbp
  have hr := (encloses_iff_getScalar (n := preB.dim) _ _ _).2 fun i =>
    alphaRelax_getScalar_bounds preB.lo preB.hi αt _ _ _ _ hαrange hzAff hzIbp i
  exact hr

/-- Under `AlphaOK`, when `getAlpha?` finds an entry `αv`, the α vector that `reluBounds?` selects
without phases has every component in `[0, 1]`. That vector is `αv` cast to the width of `preB`
when the widths agree, and `defaultAlphaVec` otherwise. -/
theorem Graph.selected_alpha_range {alpha : Array (Option (FlatTensor ℝ))} {id : Nat}
    {αv : FlatTensor ℝ} {preB : FlatBox ℝ} (halpha : AlphaOK (alpha := alpha))
    (hαopt : Cert.getAlpha? (α := ℝ) alpha id = some αv) :
    ∀ i : Fin preB.dim,
      (0 : ℝ) ≤ getScalar (if hα : αv.n = preB.dim then castDimScalar (α := ℝ) hα αv.v
        else defaultAlphaVec (α := ℝ) preB.lo preB.hi) i ∧
      getScalar (if hα : αv.n = preB.dim then castDimScalar (α := ℝ) hα αv.v
        else defaultAlphaVec (α := ℝ) preB.lo preB.hi) i ≤ 1 := by
  by_cases hα : αv.n = preB.dim
  · rw [dite_eq_left hα]
    exact castDimScalar_unit_range hα αv.v (getAlpha?_unit_range halpha hαopt)
  · rw [dite_eq_right hα]
    exact defaultAlphaVec_range preB.lo preB.hi

end TorchLean.CROWN

namespace TorchLean.CROWN.Graph
open Tensor
open Cert
noncomputable section
open CROWN.Directed (RowEncloses)

/--
Index-wise bounds delivered by a consistent phase relaxation.

At index `i`, the lower relaxation applied to the affine lower bound sits below `relu (z i)`, and
the upper relaxation applied to the affine upper bound sits above it. Both slopes are nonnegative,
so the affine bounds on `z` pass through the relaxations.
-/
theorem phaseRelax_getScalar_bounds {n : Nat}
    (lAff uAff lo hi z αt : Tensor ℝ [n]) (phases : Array Int)
    (relaxLo relaxHi : Tensor (CROWN.ReLURelax ℝ) [n])
    (hαrange : ∀ i : Fin n, (0 : ℝ) ≤ getScalar αt i ∧ getScalar αt i ≤ (1 : ℝ))
    (hrelax : phaseRelaxVec? (α := ℝ) (n := n) lo hi αt phases = some (relaxLo, relaxHi))
    (hzAff : RowEncloses { dim := n, lo := lAff, hi := uAff } n fun j => getAtOrZero z [j])
    (hzIbp : RowEncloses { dim := n, lo := lo, hi := hi } n fun j => getAtOrZero z [j])
    (i : Fin n) :
    (getScalar relaxLo i).slope * getScalar lAff i + (getScalar relaxLo i).bias ≤
        Activation.relu (α := ℝ) (getScalar z i) ∧
      Activation.relu (α := ℝ) (getScalar z i) ≤
        (getScalar relaxHi i).slope * getScalar uAff i + (getScalar relaxHi i).bias := by
  obtain ⟨hzLo, hzHi⟩ := (encloses_iff_getScalar (n := n) lAff uAff z).1 hzAff i
  obtain ⟨hzIlo, hzIhi⟩ := (encloses_iff_getScalar (n := n) lo hi z).1 hzIbp i
  obtain ⟨ph, hcons, hrHi, hrLo⟩ :=
    (phaseRelaxVec?_some_getScalar lo hi αt phases relaxLo relaxHi hrelax).2 i
  obtain ⟨hai0, hai1⟩ := hαrange i
  rw [hrLo, hrHi]
  have hsLo := phaseRelaxLowerScalar_slope_nonneg (getScalar lo i) (getScalar hi i)
    (getScalar αt i) ph hai0
  have hsHi := phaseRelaxUpperScalar_slope_nonneg (getScalar lo i) (getScalar hi i) ph
  have hlo := CROWN.Cert.phaseRelaxLowerScalar_sound (getScalar lo i)
    (getScalar hi i) (getScalar αt i) (getScalar z i) hzIlo hzIhi hai0 hai1 ph hcons
  have hhi := CROWN.Cert.phaseRelaxUpperScalar_sound (getScalar lo i)
    (getScalar hi i) (getScalar z i) hzIlo hzIhi ph hcons
  constructor
  · exact le_trans (add_le_add (mul_le_mul_of_nonneg_left hzLo hsLo) le_rfl) hlo
  · exact le_trans hhi (add_le_add (mul_le_mul_of_nonneg_left hzHi hsHi) le_rfl)

/--
Enclosure of `relu z` by phase relaxations propagated through affine bounds on `z`.

Here `xLo` and `xHi` are the parent's affine lower and upper bounds, already cast to the IBP
dimension `n` and evaluated at the input `x'`, and `lo` and `hi` form the IBP box on which
`phaseRelaxVec?` checked the phases.
-/
theorem encloses_relu_propagateAffine {inDim n : Nat}
    (xLo xHi : Linear ℝ inDim n) (x' : Tensor ℝ [inDim])
    (lo hi z αt : Tensor ℝ [n]) (phases : Array Int)
    (relaxLo relaxHi : Tensor (CROWN.ReLURelax ℝ) [n])
    (hαrange : ∀ i : Fin n, (0 : ℝ) ≤ getScalar αt i ∧ getScalar αt i ≤ (1 : ℝ))
    (hrelax : phaseRelaxVec? (α := ℝ) (n := n) lo hi αt phases = some (relaxLo, relaxHi))
    (hzAff : RowEncloses
      { dim := n
        lo := Linear.forward (α := ℝ) (inDim := inDim) (outDim := n) xLo x'
        hi := Linear.forward (α := ℝ) (inDim := inDim) (outDim := n) xHi x' } n
      fun j => getAtOrZero z [j])
    (hzIbp : RowEncloses { dim := n, lo := lo, hi := hi } n fun j => getAtOrZero z [j]) :
    RowEncloses
      { dim := n
        lo := Linear.forward (α := ℝ) (inDim := inDim) (outDim := n)
          (CROWN.ReLU.propagateAffine (α := ℝ)
            (inDim := inDim) (hidDim := n) relaxLo xLo) x'
        hi := Linear.forward (α := ℝ) (inDim := inDim) (outDim := n)
          (CROWN.ReLU.propagateAffine (α := ℝ)
            (inDim := inDim) (hidDim := n) relaxHi xHi) x' }
      n fun j => getAtOrZero (Tensor.relu (α := ℝ) z) [j] := by
  refine (encloses_iff_getScalar (n := n) _ _ _).2 ?_
  intro i
  rw [Linear.forward_propagateAffine, Linear.forward_propagateAffine,
    Tensor.relu, Tensor.getScalar_map]
  exact phaseRelax_getScalar_bounds _ _ lo hi z αt phases relaxLo relaxHi hαrange hrelax hzAff
    hzIbp i

/--
Soundness of the phased ReLU rule at one node.

The parent value `vp` is enclosed by its certificate entry `xin` (evaluated at `x`) and by its
IBP box `preB`. The node's bounds are the phase relaxations for `preB` propagated through `xin`
cast to `preB.dim`, and they enclose `relu vp`.
-/
theorem enclosesAtInput_relu_phase
    (ctx : AffineCtx) (x : Tensor ℝ [ctx.inputDim])
    (xin : FlatAffineBounds ℝ) (vp : FlatTensor ℝ) (preB : FlatBox ℝ)
    (hout : xin.outDim = preB.dim)
    (αt : Tensor ℝ [preB.dim]) (phases : Array Int)
    (relaxLo relaxHi : Tensor (CROWN.ReLURelax ℝ) [preB.dim])
    (hαrange : ∀ i : Fin preB.dim, (0 : ℝ) ≤ getScalar αt i ∧ getScalar αt i ≤ (1 : ℝ))
    (hrelax : phaseRelaxVec? (α := ℝ) (n := preB.dim) preB.lo preB.hi αt phases =
      some (relaxLo, relaxHi))
    (hpar : EnclosesAtInput ctx x xin vp)
    (hibp : RowEncloses preB vp.n fun i => getAtOrZero vp.v [i]) :
    EnclosesAtInput ctx x
      { inDim := xin.inDim
        outDim := preB.dim
        loAff := CROWN.ReLU.propagateAffine (α := ℝ)
          (inDim := xin.inDim) (hidDim := preB.dim) relaxLo
          (CROWN.Graph.castAffineOut (α := ℝ) hout xin.loAff)
        hiAff := CROWN.ReLU.propagateAffine (α := ℝ)
          (inDim := xin.inDim) (hidDim := preB.dim) relaxHi
          (CROWN.Graph.castAffineOut (α := ℝ) hout xin.hiAff) }
      { n := vp.n, v := Tensor.relu (α := ℝ) vp.v } := by
  obtain ⟨n, v⟩ := vp
  obtain rfl : preB.dim = n := hibp.1
  obtain ⟨hinDim, hzAff⟩ := enclosesAtInput_castOut ctx x xin ⟨preB.dim, v⟩ hout rfl hpar
  rw [castDimScalar_self] at hzAff
  refine ⟨hinDim, ?_⟩
  exact encloses_relu_propagateAffine
    (CROWN.Graph.castAffineOut (α := ℝ) hout xin.loAff)
    (CROWN.Graph.castAffineOut (α := ℝ) hout xin.hiAff)
    (castDimScalar (α := ℝ) hinDim.symm x) preB.lo preB.hi v αt phases relaxLo relaxHi hαrange
    hrelax hzAff hibp

end
end TorchLean.CROWN.Graph
