/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/
module

public import TorchLean.Verify.RuntimeApprox.Roundoff.Recovery
public import TorchLean.Tensor.ConfiguredBinary
public import TorchLean.Tensor.Layers.Attention
public meta import TorchLean.Verify.RuntimeApprox.Roundoff.Rules
public import FloatLib.Floats.Formats.BinaryInterchange.Automation.Roundoff.Extension

/-!
# Scaled products for `roundoff`

`Attention.scaledProductsSlice`, the kernel that `Attention.scaledProducts` reads at every
batch slice, adds matrix products in checked arithmetic. When an operation loses range or a
nonzero term is absorbed, it replays the exact rational coefficients of the whole contraction,
the scale included, and rounds once (`Attention.replayProducts?`). The analysis reads the
ordinary schedule through its loops, and the theorems here read a successful replay of a nonempty
sum of products. At finite inputs, a finite replay is the exact scaled sum of the products of the
inputs' values rounded once, stated absolutely and as a mixed error. The exact sum is written in
the grouping of the ordinary schedule (each product from zero, then the products in order), so
both branches have the same reference.

The attention value and multiplier cotangents replay in the same way a contraction whose terms also
carry pointwise factors (`TorchLean.Attention.replayWeightedValues?`,
`TorchLean.Attention.replayWeightedProducts?`). At finite entries, their finite replays are the
exact sums rounded once.
-/

public section

namespace TorchLean.Roundoff

open FloatLib.Floats FloatLib.Floats.Formats.BinaryInterchange FloatLib.Numerics
open TorchLean.Numeric
open ExecFloat.Binary (toReal toRat? isFinite)
open Tensor (get2)

variable {format : FloatFormat} {plan : Configured.StoragePlan format} {code : Type}
  [ExecFloat.ModelCodec plan (Model format) code]

local notation "Value" => ExecFloat (Configured.Family format code plan)

/-- A running sum of finite encoded products is the encoding of the rational running sum. -/
private theorem foldl_encode_products {ι : Type} (indices : List ι) (x y : ι → Value)
    (hx : ∀ i, isFinite (x i) = true)
    (hy : ∀ i, isFinite (y i) = true) (initial : ℚ) :
    indices.foldl (fun accumulated i => do
        let total ← accumulated
        let a ← QuotientArithmetic.encode (x i)
        let b ← QuotientArithmetic.encode (y i)
        pure (QuotientCoefficients.add total (QuotientCoefficients.mul a b)))
      (some (.scalar initial) : Option (QuotientCoefficients 0)) =
      some (.scalar (indices.foldl (fun total i =>
        total + (toRat? (x i)).getD 0 * (toRat? (y i)).getD 0)
          initial)) := by
  induction indices generalizing initial with
  | nil => rfl
  | cons i rest ih =>
      simp only [List.foldl_cons, configured_encode_of_isFinite (hx i),
        configured_encode_of_isFinite (hy i)]
      exact ih _

/-- At finite inputs the coefficient of a scaled contraction is the rational running sum of the
products of the inputs' values, scaled. -/
private theorem productCoefficients_eq [Storage Value] {m n p groups : Nat}
    (left : Fin groups → Tensor Value [m, n]) (right : Fin groups → Tensor Value [n, p])
    (scale : Value) (row : Fin m) (column : Fin p)
    (hleft : ∀ group index, isFinite (get2 (left group) row index) = true)
    (hright : ∀ group index,
      isFinite (get2 (right group) index column) = true)
    (hscale : isFinite scale = true) :
    Attention.productCoefficients? left right scale row column =
      some (Coefficients.scalar ((List.finRange groups).foldl (fun total group =>
        (List.finRange n).foldl (fun total index =>
          total + (toRat? (get2 (left group) row index)).getD 0 *
            (toRat? (get2 (right group) index column)).getD 0) total)
          ((toRat? (0 : Value)).getD 0) *
        (toRat? scale).getD 0)) := by
  have outer : ∀ (groupList : List (Fin groups)) (initial : ℚ),
      groupList.foldl (fun accumulated group => do
          let initial ← accumulated
          (List.finRange n).foldl (fun accumulated index => do
            let total ← accumulated
            let a ← QuotientArithmetic.encode (get2 (left group) row index)
            let b ← QuotientArithmetic.encode (get2 (right group) index column)
            pure (QuotientCoefficients.add total (QuotientCoefficients.mul a b))) (some initial))
        (some (.scalar initial) : Option (QuotientCoefficients 0)) =
        some (.scalar (groupList.foldl (fun total group =>
          (List.finRange n).foldl (fun total index =>
            total + (toRat? (get2 (left group) row index)).getD 0 *
              (toRat? (get2 (right group) index column)).getD 0) total)
          initial)) := by
    intro groupList
    induction groupList with
    | nil => intro initial; rfl
    | cons group rest ih =>
        intro initial
        have inner := foldl_encode_products (List.finRange n) _ _ (hleft group) (hright group)
        simp only [Option.bind_eq_bind] at inner ih
        simp only [List.foldl_cons, Option.bind_eq_bind, Option.bind_some, inner]
        exact ih _
  have zero := configured_encode_of_isFinite (format := format) (plan := plan) (code := code)
    (ExecFloat.Binary.isFinite_zero (format := format) (plan := plan) (code := code))
  unfold Attention.productCoefficients?
  simp only [configured_encode_of_isFinite hscale, zero, Option.bind_eq_bind, Option.bind_some]
  simp only [Option.bind_eq_bind] at outer
  erw [outer]
  rfl


/-- The real value of the rational running sum of a nonempty scaled contraction, in the grouping
of the ordinary schedule. -/
private theorem cast_productSum [Storage Value] {m n p groups : Nat}
    (left : Fin (groups + 1) → Tensor Value [m, n])
    (right : Fin (groups + 1) → Tensor Value [n, p]) (scale : Value) (row : Fin m)
    (column : Fin p) :
    (((List.finRange (groups + 1)).foldl (fun total group =>
        (List.finRange n).foldl (fun total index =>
          total + (toRat? (get2 (left group) row index)).getD 0 *
            (toRat? (get2 (right group) index column)).getD 0) total)
          ((toRat? (0 : Value)).getD 0) * (toRat? scale).getD 0 : ℚ) : ℝ) =
      (List.finRange groups).foldl
          (fun total group => total + (List.finRange n).foldl (fun total index =>
            total + toReal (get2 (left group.succ) row index) *
              toReal (get2 (right group.succ) index column)) 0)
          ((List.finRange n).foldl (fun total index =>
            total + toReal (get2 (left 0) row index) * toReal (get2 (right 0) index column)) 0) *
        toReal scale := by
  have inner : ∀ group (initial : ℚ), (((List.finRange n).foldl (fun total index =>
      total + (toRat? (get2 (left group) row index)).getD 0 *
        (toRat? (get2 (right group) index column)).getD 0) initial : ℚ) : ℝ) =
        (initial : ℝ) + (List.finRange n).foldl (fun total index =>
          total + toReal (get2 (left group) row index) *
            toReal (get2 (right group) index column)) 0 := by
    intro group initial
    rw [← List.foldl_map, ← List.foldl_hom (Rat.cast : ℚ → ℝ) (g₂ := fun (x : ℝ) (y : ℚ) => x + y)
      fun x y => (Rat.cast_add x y).symm, ← add_zero (initial : ℝ), ← List.foldl_map,
      List.foldl_assoc]
    simp only [List.foldl_map, add_zero, Rat.cast_mul, cast_toRat_eq_toReal]
  have outer : ∀ (groupList : List (Fin (groups + 1))) (initial : ℚ),
      ((groupList.foldl (fun total group => (List.finRange n).foldl (fun total index =>
        total + (toRat? (get2 (left group) row index)).getD 0 *
          (toRat? (get2 (right group) index column)).getD 0) total) initial : ℚ) : ℝ) =
        groupList.foldl (fun total group => total + (List.finRange n).foldl (fun total index =>
          total + toReal (get2 (left group) row index) *
            toReal (get2 (right group) index column)) 0) (initial : ℝ) := by
    intro groupList
    induction groupList with
    | nil => intro initial; rfl
    | cons group rest ih => intro initial; rw [List.foldl_cons, List.foldl_cons, ih, inner]
  rw [Rat.cast_mul, outer, List.finRange_succ, List.foldl_cons, List.foldl_map,
    cast_toRat_eq_toReal, cast_toRat_eq_toReal, ExecFloat.Binary.toReal_zero,
    zero_add]

/-- A finite replayed coefficient `q` is `q` rounded once, also when a zero coefficient keeps the
ordinary zero's sign. -/
private theorem toReal_replaySelect (q : ℚ) (ordinary : Value) (hformat : format.isIEEE = true)
    (hfinite : isFinite (if (q == 0 && ordinary == 0) = true then
      QuotientArithmetic.copyPrimal ordinary (ExecFloat.Binary.ofModel (Model.ofRat format q))
      else ExecFloat.Binary.ofModel (Model.ofRat format q) : Value) = true) :
    toReal (if (q == 0 && ordinary == 0) = true then
      QuotientArithmetic.copyPrimal ordinary (ExecFloat.Binary.ofModel (Model.ofRat format q))
      else ExecFloat.Binary.ofModel (Model.ofRat format q) : Value) =
      Model.roundAt format q := by
  by_cases zero : (q == 0 && ordinary == 0) = true
  · rw [ite_eq_left zero]
    obtain ⟨equal, ordinaryZero⟩ := Bool.and_eq_true_iff.mp zero
    rw [beq_iff_eq.mp equal, Rat.cast_zero, Model.roundAt_zero]
    exact (ExecFloat.Binary.beq_zero_iff.mp ordinaryZero).2
  · rw [ite_eq_right zero] at hfinite ⊢
    simp only [isFinite, toReal, ExecFloat.Binary.toModel_ofModel] at hfinite ⊢
    exact Model.toReal_ofRat_eq_roundAt _ hformat hfinite

/-- At finite inputs, a finite replay of a nonempty scaled contraction is its exact value rounded
once: the scaled sum of the products of the inputs' values. -/
theorem toReal_replayProducts [Storage Value] {m n p groups : Nat}
    (left : Fin (groups + 1) → Tensor Value [m, n])
    (right : Fin (groups + 1) → Tensor Value [n, p]) (scale ordinary : Value) (row : Fin m)
    (column : Fin p) (hformat : format.isIEEE = true)
    (hleft : ∀ group index, isFinite (get2 (left group) row index) = true)
    (hright : ∀ group index, isFinite (get2 (right group) index column) = true)
    (hscale : isFinite scale = true)
    (found : (Attention.replayProducts? left right scale ordinary row column).isSome)
    (hresult : isFinite ((Attention.replayProducts? left right scale ordinary row column).get
      found) = true) :
    toReal ((Attention.replayProducts? left right scale ordinary row column).get found) =
      Model.roundAt format ((List.finRange groups).foldl
          (fun total group => total + (List.finRange n).foldl (fun total index =>
            total + toReal (get2 (left group.succ) row index) *
              toReal (get2 (right group.succ) index column)) 0)
          ((List.finRange n).foldl (fun total index =>
            total + toReal (get2 (left 0) row index) * toReal (get2 (right 0) index column)) 0) *
        toReal scale) := by
  rw [Option.get_eq_getD (fallback := ordinary)] at hresult ⊢
  rw [← cast_productSum left right scale row column]
  unfold Attention.replayProducts? at hresult ⊢
  rw [productCoefficients_eq left right scale row column hleft hright hscale] at hresult ⊢
  dsimp only at hresult ⊢
  simp only [Option.bind_eq_bind, Option.bind_some] at hresult ⊢
  erw [configured_decode] at hresult ⊢
  simp only [Option.bind_some, Option.pure_def, Option.getD_some,
    QuotientCoefficients.primal] at hresult ⊢
  exact toReal_replaySelect _ ordinary hformat hresult

/-- The absolute form of `toReal_replayProducts`. -/
@[roundoff_rule]
theorem abs_replayProducts_sub_le [Storage Value] {m n p groups : Nat}
    (left : Fin (groups + 1) → Tensor Value [m, n])
    (right : Fin (groups + 1) → Tensor Value [n, p]) (scale ordinary : Value) (row : Fin m)
    (column : Fin p) (hformat : format.isIEEE = true)
    (hleft : ∀ group index, isFinite (get2 (left group) row index) = true)
    (hright : ∀ group index, isFinite (get2 (right group) index column) = true)
    (hscale : isFinite scale = true)
    (found : (Attention.replayProducts? left right scale ordinary row column).isSome)
    (hresult : isFinite ((Attention.replayProducts? left right scale ordinary row column).get
      found) = true) :
    |toReal ((Attention.replayProducts? left right scale ordinary row column).get found) -
        ((List.finRange groups).foldl
          (fun total group => total + (List.finRange n).foldl (fun total index =>
            total + toReal (get2 (left group.succ) row index) *
              toReal (get2 (right group.succ) index column)) 0)
          ((List.finRange n).foldl (fun total index =>
            total + toReal (get2 (left 0) row index) * toReal (get2 (right 0) index column)) 0) *
        toReal scale)| ≤
      Model.halfUlpAt format ((List.finRange groups).foldl
          (fun total group => total + (List.finRange n).foldl (fun total index =>
            total + toReal (get2 (left group.succ) row index) *
              toReal (get2 (right group.succ) index column)) 0)
          ((List.finRange n).foldl (fun total index =>
            total + toReal (get2 (left 0) row index) * toReal (get2 (right 0) index column)) 0) *
        toReal scale) := by
  rw [toReal_replayProducts left right scale ordinary row column hformat hleft hright hscale
    found hresult]
  exact Model.abs_roundAt_sub_le _ _

/-- The mixed form of `toReal_replayProducts`: one rounding. -/
@[roundoff_rule]
theorem mixedError_replayProducts [Storage Value] {m n p groups : Nat}
    (left : Fin (groups + 1) → Tensor Value [m, n])
    (right : Fin (groups + 1) → Tensor Value [n, p]) (scale ordinary : Value) (row : Fin m)
    (column : Fin p) (hformat : format.isIEEE = true)
    (hleft : ∀ group index, isFinite (get2 (left group) row index) = true)
    (hright : ∀ group index, isFinite (get2 (right group) index column) = true)
    (hscale : isFinite scale = true)
    (found : (Attention.replayProducts? left right scale ordinary row column).isSome)
    (hresult : isFinite ((Attention.replayProducts? left right scale ordinary row column).get
      found) = true) :
    MixedError
      (toReal ((Attention.replayProducts? left right scale ordinary row column).get found))
      ((List.finRange groups).foldl
          (fun total group => total + (List.finRange n).foldl (fun total index =>
            total + toReal (get2 (left group.succ) row index) *
              toReal (get2 (right group.succ) index column)) 0)
          ((List.finRange n).foldl (fun total index =>
            total + toReal (get2 (left 0) row index) * toReal (get2 (right 0) index column)) 0) *
        toReal scale)
      (Model.unitRoundoffAt format) (Model.underflowErrorAt format) :=
  ExecFloat.Binary.mixedError_toReal_of_eq_roundAt <| toReal_replayProducts left right scale
    ordinary row column hformat hleft hright hscale found hresult

/-- A running sum of finite encoded triple products is the encoding of the rational running sum. -/
private theorem foldl_encode_triples {ι : Type} (indices : List ι) (x y z : ι → Value)
    (hx : ∀ i, isFinite (x i) = true) (hy : ∀ i, isFinite (y i) = true)
    (hz : ∀ i, isFinite (z i) = true) (initial : ℚ) :
    indices.foldl (fun accumulated i => do
        let total ← accumulated
        let a ← QuotientArithmetic.encode (x i)
        let b ← QuotientArithmetic.encode (y i)
        let c ← QuotientArithmetic.encode (z i)
        pure (QuotientCoefficients.add total
          (QuotientCoefficients.mul (QuotientCoefficients.mul a b) c)))
      (some (.scalar initial) : Option (QuotientCoefficients 0)) =
      some (.scalar (indices.foldl (fun total i =>
        total + (toRat? (x i)).getD 0 * (toRat? (y i)).getD 0 * (toRat? (z i)).getD 0)
          initial)) := by
  induction indices generalizing initial with
  | nil => rfl
  | cons i rest ih =>
      simp only [List.foldl_cons, configured_encode_of_isFinite (hx i),
        configured_encode_of_isFinite (hy i), configured_encode_of_isFinite (hz i)]
      exact ih _

/-- At finite entries the coefficient of a weighted contraction is the rational running sum of the
products of the entries' values, times the unit scale and the factor. -/
private theorem weightedProductCoefficients_eq [Storage Value] {m n p : Nat}
    (left : Tensor Value [m, n]) (right : Tensor Value [n, p]) (factor : Tensor Value [m, p])
    (row : Fin m) (column : Fin p) (hformat : format.isIEEE = true)
    (hleft : ∀ index, isFinite (get2 left row index) = true)
    (hright : ∀ index, isFinite (get2 right index column) = true)
    (hfactor : isFinite (get2 factor row column) = true) :
    Attention.weightedProductCoefficients? left right factor row column =
      some (Coefficients.scalar ((List.finRange n).foldl (fun total index =>
          total + (toRat? (get2 left row index)).getD 0 *
            (toRat? (get2 right index column)).getD 0) ((toRat? (0 : Value)).getD 0) *
        (toRat? (1 : Value)).getD 0 * (toRat? (get2 factor row column)).getD 0)) := by
  have zero := configured_encode_of_isFinite (format := format) (plan := plan) (code := code)
    (ExecFloat.Binary.isFinite_zero (format := format) (plan := plan) (code := code))
  have one := configured_encode_of_isFinite (format := format) (plan := plan) (code := code)
    (ExecFloat.Binary.isFinite_one (format := format) (plan := plan) (code := code) hformat)
  have inner := foldl_encode_products (List.finRange n) (fun index => get2 left row index)
    (fun index => get2 right index column) hleft hright ((toRat? (0 : Value)).getD 0)
  unfold Attention.weightedProductCoefficients? Attention.productCoefficients?
  simp only [one, zero, configured_encode_of_isFinite hfactor, Option.bind_eq_bind,
    Option.bind_some, List.finRange_succ, List.finRange_zero, List.map_nil, List.foldl_cons,
    List.foldl_nil] at inner ⊢
  erw [inner]
  rfl

/-- At finite entries the coefficient of a transposed value-gradient contraction is the rational
running sum of the products of the entries' values. -/
private theorem weightedValueCoefficients_eq [Storage Value] {rows keys width : Nat}
    (weights multiplier : Tensor Value [rows, keys]) (gradient : Tensor Value [rows, width])
    (key : Fin keys) (column : Fin width)
    (hweights : ∀ row, isFinite (get2 weights row key) = true)
    (hmultiplier : ∀ row, isFinite (get2 multiplier row key) = true)
    (hgradient : ∀ row, isFinite (get2 gradient row column) = true) :
    Attention.weightedValueCoefficients? weights multiplier gradient key column =
      some (Coefficients.scalar ((List.finRange rows).foldl (fun total row =>
        total + (toRat? (get2 weights row key)).getD 0 *
          (toRat? (get2 multiplier row key)).getD 0 * (toRat? (get2 gradient row column)).getD 0)
        ((toRat? (0 : Value)).getD 0))) := by
  have zero := configured_encode_of_isFinite (format := format) (plan := plan) (code := code)
    (ExecFloat.Binary.isFinite_zero (format := format) (plan := plan) (code := code))
  have folded := foldl_encode_triples (List.finRange rows) (fun row => get2 weights row key)
    (fun row => get2 multiplier row key) (fun row => get2 gradient row column) hweights
    hmultiplier hgradient ((toRat? (0 : Value)).getD 0)
  unfold Attention.weightedValueCoefficients?
  simp only [zero, Option.bind_eq_bind, Option.bind_some] at folded ⊢
  erw [folded]

/-- At finite entries, a finite replay of a weighted contraction is its exact value rounded once:
the sum of the products of the entries' values, times the factor. -/
theorem toReal_replayWeightedProducts [Storage Value] {m n p : Nat}
    (left : Tensor Value [m, n]) (right : Tensor Value [n, p]) (factor : Tensor Value [m, p])
    (ordinary : Value) (row : Fin m) (column : Fin p) (hformat : format.isIEEE = true)
    (hleft : ∀ index, isFinite (get2 left row index) = true)
    (hright : ∀ index, isFinite (get2 right index column) = true)
    (hfactor : isFinite (get2 factor row column) = true)
    (found : (Attention.replayWeightedProducts? left right factor ordinary row column).isSome)
    (hresult : isFinite ((Attention.replayWeightedProducts? left right factor ordinary row
      column).get found) = true) :
    toReal ((Attention.replayWeightedProducts? left right factor ordinary row column).get
        found) =
      Model.roundAt format ((List.finRange n).foldl (fun total index =>
          total + toReal (get2 left row index) * toReal (get2 right index column)) 0 *
        toReal (get2 factor row column)) := by
  have cast : (((List.finRange n).foldl (fun total index =>
          total + (toRat? (get2 left row index)).getD 0 *
            (toRat? (get2 right index column)).getD 0) ((toRat? (0 : Value)).getD 0) *
        (toRat? (1 : Value)).getD 0 * (toRat? (get2 factor row column)).getD 0 : ℚ) : ℝ) =
      (List.finRange n).foldl (fun total index =>
          total + toReal (get2 left row index) * toReal (get2 right index column)) 0 *
        toReal (get2 factor row column) := by
    rw [Rat.cast_mul, Rat.cast_mul, ← List.foldl_map,
      ← List.foldl_hom (Rat.cast : ℚ → ℝ) (g₂ := fun (x : ℝ) (y : ℚ) => x + y)
        fun x y => (Rat.cast_add x y).symm, List.foldl_map]
    simp only [Rat.cast_mul, cast_toRat_eq_toReal, ExecFloat.Binary.toReal_zero,
      ExecFloat.Binary.toReal_one (hformat := hformat), mul_one]
  rw [Option.get_eq_getD (fallback := ordinary)] at hresult ⊢
  rw [← cast]
  unfold Attention.replayWeightedProducts? at hresult ⊢
  rw [weightedProductCoefficients_eq left right factor row column hformat hleft hright hfactor]
    at hresult ⊢
  dsimp only at hresult ⊢
  simp only [Option.bind_eq_bind, Option.bind_some] at hresult ⊢
  erw [configured_decode] at hresult ⊢
  simp only [Option.bind_some, Option.pure_def, Option.getD_some,
    QuotientCoefficients.primal] at hresult ⊢
  exact toReal_replaySelect _ ordinary hformat hresult

/-- At finite entries, a finite replay of a transposed value-gradient contraction is its exact
value rounded once: the sum of the products of both factors and the gradient entries. -/
theorem toReal_replayWeightedValues [Storage Value] {rows keys width : Nat}
    (weights multiplier : Tensor Value [rows, keys]) (gradient : Tensor Value [rows, width])
    (ordinary : Value) (key : Fin keys) (column : Fin width) (hformat : format.isIEEE = true)
    (hweights : ∀ row, isFinite (get2 weights row key) = true)
    (hmultiplier : ∀ row, isFinite (get2 multiplier row key) = true)
    (hgradient : ∀ row, isFinite (get2 gradient row column) = true)
    (found : (Attention.replayWeightedValues? weights multiplier gradient ordinary key
      column).isSome)
    (hresult : isFinite ((Attention.replayWeightedValues? weights multiplier gradient ordinary
      key column).get found) = true) :
    toReal ((Attention.replayWeightedValues? weights multiplier gradient ordinary key
        column).get found) =
      Model.roundAt format ((List.finRange rows).foldl (fun total row =>
        total + toReal (get2 weights row key) * toReal (get2 multiplier row key) *
          toReal (get2 gradient row column)) 0) := by
  have cast : (((List.finRange rows).foldl (fun total row =>
        total + (toRat? (get2 weights row key)).getD 0 *
          (toRat? (get2 multiplier row key)).getD 0 * (toRat? (get2 gradient row column)).getD 0)
        ((toRat? (0 : Value)).getD 0) : ℚ) : ℝ) =
      (List.finRange rows).foldl (fun total row =>
        total + toReal (get2 weights row key) * toReal (get2 multiplier row key) *
          toReal (get2 gradient row column)) 0 := by
    rw [← List.foldl_map, ← List.foldl_hom (Rat.cast : ℚ → ℝ) (g₂ := fun (x : ℝ) (y : ℚ) => x + y)
      fun x y => (Rat.cast_add x y).symm, List.foldl_map]
    simp only [Rat.cast_mul, cast_toRat_eq_toReal, ExecFloat.Binary.toReal_zero]
  rw [Option.get_eq_getD (fallback := ordinary)] at hresult ⊢
  rw [← cast]
  unfold Attention.replayWeightedValues? at hresult ⊢
  rw [weightedValueCoefficients_eq weights multiplier gradient key column hweights hmultiplier
    hgradient] at hresult ⊢
  dsimp only at hresult ⊢
  simp only [Option.bind_eq_bind, Option.bind_some] at hresult ⊢
  erw [configured_decode] at hresult ⊢
  simp only [Option.bind_some, Option.pure_def, Option.getD_some,
    QuotientCoefficients.primal] at hresult ⊢
  exact toReal_replaySelect _ ordinary hformat hresult

/-- The absolute form of `toReal_replayWeightedProducts`. -/
@[roundoff_rule]
theorem abs_replayWeightedProducts_sub_le [Storage Value] {m n p : Nat}
    (left : Tensor Value [m, n]) (right : Tensor Value [n, p]) (factor : Tensor Value [m, p])
    (ordinary : Value) (row : Fin m) (column : Fin p) (hformat : format.isIEEE = true)
    (hleft : ∀ index, isFinite (get2 left row index) = true)
    (hright : ∀ index, isFinite (get2 right index column) = true)
    (hfactor : isFinite (get2 factor row column) = true)
    (found : (Attention.replayWeightedProducts? left right factor ordinary row column).isSome)
    (hresult : isFinite ((Attention.replayWeightedProducts? left right factor ordinary row
      column).get found) = true) :
    |toReal ((Attention.replayWeightedProducts? left right factor ordinary row column).get
        found) -
        ((List.finRange n).foldl (fun total index =>
          total + toReal (get2 left row index) * toReal (get2 right index column)) 0 *
        toReal (get2 factor row column))| ≤
      Model.halfUlpAt format ((List.finRange n).foldl (fun total index =>
          total + toReal (get2 left row index) * toReal (get2 right index column)) 0 *
        toReal (get2 factor row column)) := by
  rw [toReal_replayWeightedProducts left right factor ordinary row column hformat hleft hright
    hfactor found hresult]
  exact Model.abs_roundAt_sub_le _ _

/-- The mixed form of `toReal_replayWeightedProducts`: one rounding. -/
@[roundoff_rule]
theorem mixedError_replayWeightedProducts [Storage Value] {m n p : Nat}
    (left : Tensor Value [m, n]) (right : Tensor Value [n, p]) (factor : Tensor Value [m, p])
    (ordinary : Value) (row : Fin m) (column : Fin p) (hformat : format.isIEEE = true)
    (hleft : ∀ index, isFinite (get2 left row index) = true)
    (hright : ∀ index, isFinite (get2 right index column) = true)
    (hfactor : isFinite (get2 factor row column) = true)
    (found : (Attention.replayWeightedProducts? left right factor ordinary row column).isSome)
    (hresult : isFinite ((Attention.replayWeightedProducts? left right factor ordinary row
      column).get found) = true) :
    MixedError
      (toReal ((Attention.replayWeightedProducts? left right factor ordinary row column).get
        found))
      ((List.finRange n).foldl (fun total index =>
          total + toReal (get2 left row index) * toReal (get2 right index column)) 0 *
        toReal (get2 factor row column))
      (Model.unitRoundoffAt format) (Model.underflowErrorAt format) :=
  ExecFloat.Binary.mixedError_toReal_of_eq_roundAt <| toReal_replayWeightedProducts left
    right factor ordinary row column hformat hleft hright hfactor found hresult

/-- The absolute form of `toReal_replayWeightedValues`. -/
@[roundoff_rule]
theorem abs_replayWeightedValues_sub_le [Storage Value] {rows keys width : Nat}
    (weights multiplier : Tensor Value [rows, keys]) (gradient : Tensor Value [rows, width])
    (ordinary : Value) (key : Fin keys) (column : Fin width) (hformat : format.isIEEE = true)
    (hweights : ∀ row, isFinite (get2 weights row key) = true)
    (hmultiplier : ∀ row, isFinite (get2 multiplier row key) = true)
    (hgradient : ∀ row, isFinite (get2 gradient row column) = true)
    (found : (Attention.replayWeightedValues? weights multiplier gradient ordinary key
      column).isSome)
    (hresult : isFinite ((Attention.replayWeightedValues? weights multiplier gradient ordinary
      key column).get found) = true) :
    |toReal ((Attention.replayWeightedValues? weights multiplier gradient ordinary key
        column).get found) -
        ((List.finRange rows).foldl (fun total row =>
        total + toReal (get2 weights row key) * toReal (get2 multiplier row key) *
          toReal (get2 gradient row column)) 0)| ≤
      Model.halfUlpAt format ((List.finRange rows).foldl (fun total row =>
        total + toReal (get2 weights row key) * toReal (get2 multiplier row key) *
          toReal (get2 gradient row column)) 0) := by
  rw [toReal_replayWeightedValues weights multiplier gradient ordinary key column hformat
    hweights hmultiplier hgradient found hresult]
  exact Model.abs_roundAt_sub_le _ _

/-- The mixed form of `toReal_replayWeightedValues`: one rounding. -/
@[roundoff_rule]
theorem mixedError_replayWeightedValues [Storage Value] {rows keys width : Nat}
    (weights multiplier : Tensor Value [rows, keys]) (gradient : Tensor Value [rows, width])
    (ordinary : Value) (key : Fin keys) (column : Fin width) (hformat : format.isIEEE = true)
    (hweights : ∀ row, isFinite (get2 weights row key) = true)
    (hmultiplier : ∀ row, isFinite (get2 multiplier row key) = true)
    (hgradient : ∀ row, isFinite (get2 gradient row column) = true)
    (found : (Attention.replayWeightedValues? weights multiplier gradient ordinary key
      column).isSome)
    (hresult : isFinite ((Attention.replayWeightedValues? weights multiplier gradient ordinary
      key column).get found) = true) :
    MixedError
      (toReal ((Attention.replayWeightedValues? weights multiplier gradient ordinary key
        column).get found))
      ((List.finRange rows).foldl (fun total row =>
        total + toReal (get2 weights row key) * toReal (get2 multiplier row key) *
          toReal (get2 gradient row column)) 0)
      (Model.unitRoundoffAt format) (Model.underflowErrorAt format) :=
  ExecFloat.Binary.mixedError_toReal_of_eq_roundAt <| toReal_replayWeightedValues weights
    multiplier gradient ordinary key column hformat hweights hmultiplier hgradient found hresult

end TorchLean.Roundoff
