/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.MLTheory.CROWN.Proofs.DirectedIBPNormalizationTensor
public import NN.MLTheory.CROWN.Proofs.LayerNormEnclosure
import all NN.MLTheory.CROWN.Graph.Engine.Base

/-!
# Rounded LayerNorm transfers

The checked matrix view normalizes the complete suffix beginning at the requested axis.
Stored gamma, beta, and epsilon are interpreted as real scalars before applying the real
specification. Both the directed row path and the uniform default-parameter path are covered.
-/

@[expose] public section

namespace NN.MLTheory.CROWN.Graph.DirectedBackward

open Spec TorchLean TorchLean.Tensor NN.IR
open NN.MLTheory.CROWN.Graph.Internal
open _root_.Proofs.Autograd.Norm
open scoped BigOperators

noncomputable section

/-- A successful matrix-layout check has positive width and preserves the element count. -/
theorem layerNormMatrixDims_properties {s : Shape} {axis rows width : Nat}
    (h : (OpContracts.layerNormMatrixDims axis s).toOption = some (rows, width)) :
    0 < width ∧ width = (s.toList.drop axis).prod ∧ s.size = Shape.size [rows, width] := by
  have hok : OpContracts.layerNormMatrixDims axis s = .ok (rows, width) := by
    cases he : OpContracts.layerNormMatrixDims axis s <;> simp_all [Except.toOption]
  unfold OpContracts.layerNormMatrixDims at hok
  cases ha : OpContracts.checkAxisValid axis s with
  | error message => simp [ha, Bind.bind, Except.bind] at hok
  | ok result =>
      simp only [ha, Bind.bind, Except.bind, OpContracts.checkPositive,
        ← List.prod_eq_foldl] at hok
      split at hok
      · contradiction
      · rename_i hwidth
        simp only [Pure.pure, Except.pure,
          Except.ok.injEq, Prod.mk.injEq] at hok
        obtain ⟨hrows, hwidth'⟩ := hok
        subst rows width
        have hw : (s.toList.drop axis).prod ≠ 0 := by
          intro hz
          simp [hz] at hwidth
        refine ⟨Nat.pos_of_ne_zero hw, rfl, ?_⟩
        simpa only [Shape.size, Nat.mul_one, Shape.size_eq_prod, Shape.toList] using
          (List.prod_take_mul_prod_drop s axis).symm

/-- Dividing a centered row entry by its stabilized standard deviation bounds its magnitude
by the square root of the row width. Zero denominators follow Lean's real division convention. -/
theorem layerNorm_normalized_abs_le_sqrt {m n : Nat} (hn : 0 < n)
    (x : Tensor ℝ [m, n]) (epsilon : ℝ) (hepsilon : 0 ≤ epsilon)
    (i : Fin m) (j : Fin n) :
    |(Spec.get2 x i j - rowMeanE x i) /
        Real.sqrt (max (rowVarE x i + epsilon) 0)| ≤ Real.sqrt n := by
  have hnreal : (0 : ℝ) < n := by exact_mod_cast hn
  have hentry :
      (Spec.get2 x i j - rowMeanE x i) ^ 2 ≤ (n : ℝ) * rowVarE x i := by
    have hs := Finset.single_le_sum
      (fun k (_ : k ∈ (Finset.univ : Finset (Fin n))) =>
        mul_self_nonneg (Spec.get2 x i k - rowMeanE x i))
      (Finset.mem_univ j)
    simpa only [rowVarE, mul_div_cancel₀ _ hnreal.ne', pow_two] using hs
  have ht : rowVarE x i ≤ max (rowVarE x i + epsilon) 0 :=
    (le_add_of_nonneg_right hepsilon).trans (le_max_left _ _)
  have hsquare := hentry.trans (mul_le_mul_of_nonneg_left ht hnreal.le)
  have habs := Real.abs_le_sqrt hsquare
  rw [Real.sqrt_mul hnreal.le] at habs
  by_cases hz : Real.sqrt (max (rowVarE x i + epsilon) 0) = 0
  · simp [hz, Real.sqrt_nonneg]
  · have hp : 0 < Real.sqrt (max (rowVarE x i + epsilon) 0) :=
      lt_of_le_of_ne (Real.sqrt_nonneg _) (Ne.symm hz)
    rw [abs_div, abs_of_nonneg (Real.sqrt_nonneg _)]
    exact (div_le_iff₀ hp).mpr habs

variable {α : Type} [Storage α] [Context α] [BoundOps α] [LawfulBoundOps α]

local notation "value" => LawfulBoundOps.toReal (α := α)

/-- Resolve the optional stored payload using the same suffix validation as IR evaluation. -/
def normalizationAffine? (s : Shape) (axis width : Nat)
    (parameters : Option (NN.IR.LayerNormParams α)) :
    Option (NN.IR.Graph.LayerNormAffine α width) :=
  (NN.IR.Graph.resolveLayerNormAffine
    { layerNorm? := fun _ => parameters } 0 axis s width).toOption

/-- The unit scale, zero bias, and stored-format default epsilon used by IR evaluation. -/
def defaultNormalizationAffine (width : Nat) : NN.IR.Graph.LayerNormAffine α width :=
  { gamma := Tensor.full [width] 1
    beta := Tensor.full [width] 0
    epsilon := TorchLean.normalizationEpsilon }

/-- The payload synthesized by the executable default-parameter fallback. -/
def defaultLayerNormParams (s : Shape) (axis : Nat) : NN.IR.LayerNormParams α :=
  { normalizedShape := Shape.ofList (s.toList.drop axis)
    gamma := Tensor.full (Shape.ofList (s.toList.drop axis)) 1
    beta := Tensor.full (Shape.ofList (s.toList.drop axis)) 0
    eps := TorchLean.normalizationEpsilon }

omit [BoundOps α] [LawfulBoundOps α] in
@[simp] theorem normalizationAffine?_none (s : Shape) (axis width : Nat) :
    normalizationAffine? (α := α) s axis width none =
      some (defaultNormalizationAffine width) := rfl

omit [BoundOps α] [LawfulBoundOps α] in
/-- Resolving the synthesized default payload gives exactly the absent-payload affine data. -/
theorem normalizationAffine?_defaultLayerNormParams (s : Shape) (axis width : Nat)
    (hwidth : width = (s.toList.drop axis).prod) :
    normalizationAffine? s axis width (some (defaultLayerNormParams (α := α) s axis)) =
      some (defaultNormalizationAffine width) := by
  have hsize : (Shape.ofList (s.toList.drop axis)).size = width := by
    simpa only [Shape.size_eq_prod, Shape.ofList, Shape.toList] using hwidth.symm
  cases hdec : @decEq Shape inferInstance (Shape.ofList (s.toList.drop axis))
      (Shape.ofList (s.toList.drop axis)) with
  | isFalse h => exact (h rfl).elim
  | isTrue h =>
      simp [normalizationAffine?, defaultLayerNormParams, NN.IR.Graph.resolveLayerNormAffine,
        hdec, hsize, defaultNormalizationAffine, normalization_reshapeSpec_full, Except.toOption,
        Pure.pure, Except.pure]

/-- The actual real matrix LayerNorm, with every affine parameter and epsilon interpreted
from its stored scalar. Empty leading batches retain the IR's empty-tensor convention. -/
def layerNormRealValue (rows width : Nat) (f : Nat → ℝ)
    (affine : NN.IR.Graph.LayerNormAffine α width) (hwidth : 0 < width) :
    Tensor ℝ [rows, width] :=
  NN.IR.Graph.layerNormMatrixValue rows width (tensorOfFlatValues [rows, width] f)
    (Tensor.ofFn fun j => value (affine.gamma.getScalar j))
    (Tensor.ofFn fun j => value (affine.beta.getScalar j))
    (value affine.epsilon) hwidth

/-- Coordinate equation for suffix LayerNorm. Its premises are successful layout and payload
resolution, independently of any interval box or interval transfer. -/
def LayerNormRealEquation (s : Shape) (axis : Nat)
    (parameters : Option (NN.IR.LayerNormParams α)) (f g : Nat → ℝ) : Prop :=
  ∀ rows width, (OpContracts.layerNormMatrixDims axis s).toOption = some (rows, width) →
    ∀ affine, normalizationAffine? s axis width parameters = some affine →
      ∀ hwidth : 0 < width, ∀ c : Shape.Coord [rows, width],
        g (Shape.Coord.linearize c).val = layerNormRealValue rows width f affine hwidth c

/-- The real coordinate equation is unchanged by materializing the default affine payload. -/
theorem LayerNormRealEquation.defaultPayload {s : Shape} {axis : Nat} {f g : Nat → ℝ}
    (heq : LayerNormRealEquation (α := α) s axis none f g) :
    LayerNormRealEquation s axis (some (defaultLayerNormParams (α := α) s axis)) f g := by
  intro rows width hlayout affine haffine hwidth c
  rw [normalizationAffine?_defaultLayerNormParams s axis width
    (layerNormMatrixDims_properties hlayout).2.1, Option.some.injEq] at haffine
  subst affine
  exact heq rows width hlayout _ (normalizationAffine?_none s axis width) hwidth c

/-- The default matrix specification has the uniform square-root width bound. -/
theorem layerNormRealValue_default_abs_le {rows width : Nat} (hwidth : 0 < width)
    (f : Nat → ℝ) (hepsilon : 0 ≤ value (TorchLean.normalizationEpsilon : α))
    (i : Fin rows) (j : Fin width) :
    |Spec.get2 (layerNormRealValue rows width f
        (defaultNormalizationAffine (α := α) width) hwidth) i j| ≤ Real.sqrt width := by
  have hm : 0 < rows := Nat.zero_lt_of_lt i.isLt
  simp only [layerNormRealValue, NN.IR.Graph.layerNormMatrixValue, dite_eq_left hm,
    get2_layerNorm, defaultNormalizationAffine, Tensor.getScalar_ofFn, Tensor.getScalar_full,
    LawfulBoundOps.toReal_one, LawfulBoundOps.toReal_zero, mul_one, add_zero]
  exact layerNorm_normalized_abs_le_sqrt hwidth _ _ hepsilon i j

/-- Default LayerNorm on a singleton suffix is exactly zero, even if epsilon rounds to zero. -/
theorem layerNormRealValue_default_singleton {rows : Nat} (f : Nat → ℝ) (i : Fin rows)
    (j : Fin 1) :
    Spec.get2 (layerNormRealValue rows 1 f
      (defaultNormalizationAffine (α := α) 1) (by omega)) i j = 0 := by
  have hm : 0 < rows := Nat.zero_lt_of_lt i.isLt
  have hj : j = 0 := Subsingleton.elim _ _
  subst j
  simp [layerNormRealValue, NN.IR.Graph.layerNormMatrixValue, hm, get2_layerNorm,
    defaultNormalizationAffine, rowMeanE, LawfulBoundOps.toReal_one,
    LawfulBoundOps.toReal_zero]

variable [NonlinearBoundOps α] [LawfulNonlinearBoundOps α]

/-- A successful stored-parameter transfer encloses the IR matrix specification, including
its resolved scale, bias, and epsilon, for every accepted normalized suffix. -/
theorem ibpLayerNormPayloadBox?_encloses {s : Shape} {axis : Nat}
    {parameters : NN.IR.LayerNormParams α} {input output : FlatBox α} {f g : Nat → ℝ}
    (hx : RowEncloses input s.size f)
    (heq : LayerNormRealEquation s axis (some parameters) f g)
    (hout : ibpLayerNormPayloadBox? s axis parameters input = some output) :
    RowEncloses output s.size g := by
  unfold ibpLayerNormPayloadBox? at hout
  obtain ⟨⟨rows, width⟩, hlayout, hout⟩ := Option.bind_eq_some_iff.mp hout
  obtain ⟨affine, haffine, hout⟩ := Option.bind_eq_some_iff.mp hout
  change normalizationAffine? s axis width (some parameters) = some affine at haffine
  obtain ⟨_, _, hout⟩ := Option.bind_eq_some_iff.mp hout
  have hepsilon : affine.epsilon > 0 := by
    by_contra h
    simp [h] at hout
  simp only [hepsilon, ↓reduceIte, Option.bind_eq_bind, Option.bind_some] at hout
  obtain ⟨_, _, hout⟩ := Option.bind_eq_some_iff.mp hout
  split at hout
  · rename_i hinput
    split at hout
    · rename_i hmatrix
      obtain ⟨bounds, hbounds, hout⟩ := Option.bind_eq_some_iff.mp hout
      have houtEq := Option.some.inj hout
      cases houtEq
      rw [hmatrix, rowEncloses_flatten_iff]
      rintro ⟨i, j, ⟨⟩⟩
      have hw := (layerNormMatrixDims_properties hlayout).1
      have hm : 0 < rows := Nat.zero_lt_of_lt i.isLt
      rw [heq rows width hlayout affine haffine hw]
      have hxmatrix : RowEncloses input (Shape.size [rows, width]) f := by
        simpa only [← hmatrix] using hx
      have hrow := LayerNormDirected.directedLayerNormRow?_encloses hm hw
        _ _ affine.gamma affine.beta affine.epsilon
        (tensorOfFlatValues [rows, width] f) i
        (fun k => by
          simpa only [Tensor.getScalar_eq_apply, Tensor.unstack,
            TorchLean.Tensor.Internal.Rep.unstack_apply, Spec.get2_eq_apply] using
            rowEncloses_unflatten hxmatrix (i, k, PUnit.unit))
        (traverseFin_eq_some_iff.mp hbounds i) j
      simpa only [layerNormRealValue, NN.IR.Graph.layerNormMatrixValue, dite_eq_left hm,
        Tensor.dim, TorchLean.Tensor.Internal.Rep.stack_apply, Tensor.getScalar_eq_apply,
        Spec.get2_eq_apply] using hrow
    · contradiction
  · contradiction

/-- The backend's uniform default-parameter box encloses the actual real normalization.
The lower radius is rounded downward; nonnegativity of the stored default epsilon is the only
additional scalar fact. -/
theorem ibpLayerNormRange?_encloses
    (hepsilon : 0 ≤ value (TorchLean.normalizationEpsilon : α))
    {s : Shape} {axis rows width : Nat} {output : FlatBox α} {f g : Nat → ℝ}
    (hlayout : (OpContracts.layerNormMatrixDims axis s).toOption = some (rows, width))
    (heq : LayerNormRealEquation (α := α) s axis none f g)
    (hout : ibpLayerNormRange? (α := α) s s.size axis = some output) :
    RowEncloses output s.size g := by
  obtain ⟨hwpos, hwidth, hmatrix⟩ := layerNormMatrixDims_properties hlayout
  have hvalue (k : Fin s.size) :
      g k.val = layerNormRealValue rows width f (defaultNormalizationAffine (α := α) width)
        hwpos (Shape.Coord.unlinearize (Fin.cast hmatrix k)) := by
    simpa only [Shape.Coord.linearize_unlinearize, Fin.val_cast] using
      heq rows width hlayout _ (normalizationAffine?_none s axis width) hwpos
        (Shape.Coord.unlinearize (Fin.cast hmatrix k))
  simp only [ibpLayerNormRange?, ← hwidth, hwpos.ne', ↓reduceIte] at hout
  split at hout
  · rename_i hwone
    clear hwidth
    subst width
    have houtEq := Option.some.inj hout
    cases houtEq
    rw [rowEncloses_iff]
    intro k
    rw [hvalue k]
    have hz := layerNormRealValue_default_singleton (α := α) f
      (Shape.Coord.unlinearize (Fin.cast hmatrix k)).1
      (Shape.Coord.unlinearize (Fin.cast hmatrix k)).2.1
    rw [Spec.get2_eq_apply] at hz
    simp [Tensor.getScalar_full, LawfulBoundOps.toReal_zero, hz]
  · obtain ⟨radius, hradius, hout⟩ := Option.bind_eq_some_iff.mp hout
    have houtEq := Option.some.inj hout
    cases houtEq
    rw [rowEncloses_iff]
    intro k
    rw [hvalue k]
    have habs := layerNormRealValue_default_abs_le (α := α) hwpos f hepsilon
      (Shape.Coord.unlinearize (Fin.cast hmatrix k)).1
      (Shape.Coord.unlinearize (Fin.cast hmatrix k)).2.1
    rw [Spec.get2_eq_apply] at habs
    have hb := abs_le.mp habs
    have hr := LawfulNonlinearBoundOps.layerNormAbsBound_sound hradius
    have hsub := LawfulBoundOps.subDown_le (0 : α) radius
    simp only [LawfulBoundOps.toReal_zero, zero_sub] at hsub
    simp only [Tensor.getScalar_full]
    exact ⟨hsub.trans ((neg_le_neg hr).trans hb.1), hb.2.trans hr⟩

/-- Both successful default-parameter execution paths enclose the actual real LayerNorm. -/
theorem ibpLayerNormBox?_encloses
    (hepsilon : 0 ≤ value (TorchLean.normalizationEpsilon : α))
    {s : Shape} {axis : Nat} {input output : FlatBox α} {f g : Nat → ℝ}
    (hx : RowEncloses input s.size f)
    (heq : LayerNormRealEquation (α := α) s axis none f g)
    (hout : ibpLayerNormBox? s input axis = some output) :
    RowEncloses output s.size g := by
  unfold ibpLayerNormBox? at hout
  obtain ⟨⟨rows, width⟩, hlayout, hout⟩ := Option.bind_eq_some_iff.mp hout
  simp only [hx.1, ↓reduceIte] at hout
  obtain ⟨_, _, hout⟩ := Option.bind_eq_some_iff.mp hout
  cases hrange : ibpLayerNormRange? (α := α) s s.size axis with
  | none =>
      simp only [hrange] at hout
      exact ibpLayerNormPayloadBox?_encloses hx heq.defaultPayload hout
  | some result =>
      simp only [hrange, Option.pure_def, Option.some.injEq] at hout
      subst output
      exact ibpLayerNormRange?_encloses hepsilon hlayout heq hrange

end

end NN.MLTheory.CROWN.Graph.DirectedBackward
