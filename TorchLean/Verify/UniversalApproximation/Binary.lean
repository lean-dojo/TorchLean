/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Verify.Floats.IEEE754
public import TorchLean.Verify.UniversalApproximation.Rounded
public import TorchLean.Tensor.ConfiguredBinary

/-!
# Executable ReLU approximation for configured IEEE formats

The hinge network `hingeFun` is evaluated at the values of a configured `FloatFormat`, storage
plan and `ModelCodec`. With `format.isIEEE = true`, finite intermediates and a finite output, the
evaluation read as reals is `hingeFun` in the rounded-real model of `Rounded`, in the
same fold order. The error bounds combine real approximation, parameter quantization and
arithmetic rounding error. On configured values `relu x` is IEEE `maximum x 0`
(`relu_eq_max_binary`).

`HingeSumFinite` records the finite intermediates. `HingeSumBounds` gives sufficient bounds on
the exact subtractions, products and partial sums, from which FloatLib's range theorems build
that witness. The dyadic quantization bounds use the rounding of the selected format and its
half-ulp bound.
-/

@[expose] public section

open FloatLib.Floats (ExecFloat)
open FloatLib.Floats.ExecFloat.Binary (isFinite ofModel toModel)
open FloatLib.Floats.Formats.BinaryInterchange (Model FloatFormat)

open FloatLib FloatLib.Numerics FloatLib.Floats.Formats
open Flocq

namespace TorchLean.UniversalApproximation

open FloatLib.Floats.Formats.BinaryInterchange

open Rounded
open Activation (relu)

noncomputable section

variable {format : FloatFormat} {plan : Configured.StoragePlan format} {code : Type}
    [ExecFloat.ModelCodec plan (Model format) code]

local notation "Value" => ExecFloat (Configured.Family format code plan)
local notation "RoundedValue" => Rounded format

open IEEE754

-- Executable values are read as reals only through the bridge lemmas of
-- `TorchLean.Verify.Floats.IEEE754`, never by unfolding `toModel` and `Model.toReal`
-- in `simp`.

/-! ## Embedding configured binary values into the rounded-real model -/

/-- Read an executable configured IEEE value as a point of the rounded-real model, through
`Model.toReal` (which sends NaN and infinities to `0`). -/
@[inline] def toRounded (x : Value) : RoundedValue := ⟨(toModel x).toReal⟩

/-- The value of `toRounded x` is `(toModel x).toReal`. -/
@[simp] theorem toRounded_val (x : Value) : (toRounded x).val = (toModel x).toReal := rfl

/-- Pointwise embedding of a vector of `Value` values into the rounded-real model. -/
@[inline] def toRoundedVec {n : ℕ} (v : Fin n → Value) : Fin n → RoundedValue := fun i =>
  toRounded (v i)

/-! ## ReLU on configured binary values -/

/-- On configured binary values `relu x` is IEEE 754 `maximum x 0`. The zero test of
`relu` is IEEE equality, which also holds at `-0`, and `maximum (-0) (+0) = +0`. -/
theorem relu_eq_max_binary (x : Value) : relu x = max x 0 := by
  unfold Activation.relu
  split
  · rename_i h
    obtain ⟨hfin, hreal⟩ := ExecFloat.Binary.beq_zero_iff.1 h
    have hzero : Model.isZero (toModel x) = true :=
      (ExecFloat.Binary.isZero_eq_true_iff_toReal_eq_zero hfin).2 hreal
    apply ExecFloat.Binary.toModel_inj.1
    rw [ExecFloat.Binary.toModel_max, Model.maximum_of_isZero hzero ExecFloat.Binary.isZero_zero]
    have h0 : toModel (0 : Value) = Model.zero format false := by
      change Configured.Family.toModel
        (Configured.Family.ofModel (Model.ofRat format (0 : Rat))) = _
      rw [Configured.Family.toModel_ofModel]
      simp [Model.ofRat, Model.Internal.Directed.roundRat, Model.Internal.Directed.roundRatScaled]
    rw [h0, Model.signBit_zero]
    simp
  · rfl

/-! ## A finiteness witness for IEEE evaluation (no NaN/Inf intermediates) -/

/--
Every intermediate of the hinge fold is finite: no subtraction, ReLU, product, or partial sum
overflows to an infinity or produces a NaN.

The inductive form follows the fold one index at a time, which lets `instDecHingeSumFinite`
decide it on concrete inputs.
-/
inductive HingeSumFinite {n : ℕ} (t c : Fin n → Value)
    (x : Value) : Value → List (Fin n) → Prop where
  /-- With no index left, the accumulator is finite. -/
  | nil {acc : Value} (hacc : isFinite acc = true) :
      HingeSumFinite t c x acc []
  /-- At index `i`, the difference, the term and the next accumulator are finite, and so is the
  rest of the fold. -/
  | cons {acc : Value} {i : Fin n} {xs : List (Fin n)}
      (hsub : isFinite (x - t i) = true)
      (hmul : isFinite (hingeTerm c t x i) = true)
      (hadd : isFinite (acc + hingeTerm c t x i) = true)
      (hrest : HingeSumFinite t c x (acc + hingeTerm c t x i) xs) :
      HingeSumFinite t c x acc (i :: xs)

/-!
### Discharging `HingeSumFinite` for *concrete* networks by computation

In a typical verification workflow `t`, `c` and `x` are concrete configured binary constants from
a lowered model artifact. Then the easiest way to satisfy a `HingeSumFinite …` hypothesis is to
run the configured binary kernel and check finiteness at every intermediate.
`instDecHingeSumFinite` walks the fold, so a true concrete instance closes by computation:

```lean
  have hSum : HingeSumFinite t c x (0 : Value) (List.finRange n) := by
    decide
```

A computation covers one input at a time. A proof for every `x` goes through `HingeSumBounds`
and `hingeSumFinite_of_bounds`.
-/

/-- Decides `HingeSumFinite` by checking the subtraction, product and partial sum at each index
in fold order, then the final accumulator. -/
instance instDecHingeSumFinite {n : ℕ} (t c : Fin n → Value)
    (x : Value) :
    ∀ (acc : Value) (xs : List (Fin n)), Decidable (HingeSumFinite t c x acc xs)
    | acc, [] =>
        if hacc : isFinite acc = true then
          isTrue (HingeSumFinite.nil (t := t) (c := c) (x := x) (acc := acc) hacc)
        else
          isFalse (by
            intro h
            cases h with
            | nil hacc' =>
                exact hacc hacc')
    | acc, i :: xs =>
        let sub := x - t i
        let term := hingeTerm c t x i
        let acc' := acc + term
        if hsub : isFinite sub = true then
          if hmul : isFinite term = true then
            if hadd : isFinite acc' = true then
              match instDecHingeSumFinite t c x acc' xs with
              | isTrue hrest =>
                  isTrue
                    (HingeSumFinite.cons
                      (t := t) (c := c) (x := x) (acc := acc) (i := i) (xs := xs)
                      (hsub := by simpa [sub] using hsub)
                      (hmul := by simpa [term] using hmul)
                      (hadd := by simpa [acc', term] using hadd)
                      (hrest := by simpa [acc', term] using hrest))
              | isFalse hrest =>
                  isFalse (by
                    intro h
                    cases h with
                    | cons _ _ _ hrest' =>
                        exact hrest (by simpa [acc', term] using hrest'))
            else
              isFalse (by
                intro h
                cases h with
                | cons _ _ hadd' _ =>
                    dsimp [acc', term] at hadd
                    exact hadd hadd')
          else
            isFalse (by
              intro h
              cases h with
              | cons _ hmul' _ _ =>
                  dsimp [term] at hmul
                  exact hmul hmul')
        else
          isFalse (by
            intro h
            cases h with
            | cons hsub' _ _ _ =>
                dsimp [sub] at hsub
                exact hsub hsub')

/-! ## Sufficient range bounds for finite intermediates -/

/-- Range bounds along the executable fold: at each index the exact difference, product and sum
of the executed operands are at most the largest finite value `Model.posMaxFinite format` in
magnitude. -/
inductive HingeSumBounds {n : ℕ} (t c : Fin n → Value) (x : Value) :
    Value → List (Fin n) → Prop where
  /-- With no index left, there is nothing to bound. -/
  | nil {acc : Value} : HingeSumBounds t c x acc []
  /-- At index `i`, the exact difference, product and sum are at most `Model.posMaxFinite format`
  in magnitude, and the rest of the fold has range bounds. -/
  | cons {acc : Value} {i : Fin n} {xs : List (Fin n)}
      (hsub : |(toModel x).toReal - (toModel (t i)).toReal| ≤
        (Model.posMaxFinite format).toReal)
      (hmul : |(toModel (c i)).toReal| * |(toModel (relu (x - t i))).toReal| ≤
          (Model.posMaxFinite format).toReal)
      (hadd : |(toModel acc).toReal + (toModel (hingeTerm c t x i)).toReal| ≤
        (Model.posMaxFinite format).toReal)
      (hrest : HingeSumBounds t c x (acc + hingeTerm c t x i) xs) :
      HingeSumBounds t c x acc (i :: xs)

/-- For an IEEE format, a finite input, finite knots and coefficients, and a finite starting
accumulator, the range bounds of `HingeSumBounds` rule out every exceptional intermediate. -/
theorem hingeSumFinite_of_bounds (hformat : format.isIEEE = true) {n : ℕ}
    {t c : Fin n → Value} {x acc : Value} {xs : List (Fin n)}
    (hx : isFinite x = true) (ht : ∀ i, isFinite (t i) = true)
    (hc : ∀ i, isFinite (c i) = true) (hbound : HingeSumBounds t c x acc xs) :
    isFinite acc = true → HingeSumFinite t c x acc xs := by
  induction hbound with
  | nil => exact HingeSumFinite.nil
  | @cons acc i xs hsub hmul hadd hrest ih =>
      intro hacc
      have hs : isFinite (x - t i) = true := by
        change Model.isFinite (toModel (x - t i)) = true
        rw [ExecFloat.Binary.toModel_hSub x (t i)]
        exact Model.isFinite_sub_of_abs_toReal_sub_le_posMaxFinite
          (toModel x) (toModel (t i)) hformat hx (ht i) hsub
      have hr : isFinite (relu (x - t i)) = true := by
        rw [relu_eq_max_binary]
        exact ExecFloat.Binary.isFinite_max hs ExecFloat.Binary.isFinite_zero
      have hm : isFinite (hingeTerm c t x i) = true := by
        change Model.isFinite (toModel (c i * relu (x - t i))) = true
        rw [ExecFloat.Binary.toModel_hMul (c i) (relu (x - t i))]
        exact Model.isFinite_mul_of_abs_mul_le_posMaxFinite _ _ hformat (hc i) hr hmul
      have ha : isFinite (acc + hingeTerm c t x i) = true := by
        change Model.isFinite (toModel (acc + hingeTerm c t x i)) = true
        rw [ExecFloat.Binary.toModel_hAdd acc (hingeTerm c t x i)]
        exact Model.isFinite_add_of_abs_toReal_add_le_posMaxFinite _ _ hformat hacc hm hadd
      exact HingeSumFinite.cons hs hm ha (ih ha)

/-- A finite-fold witness guarantees that its final accumulator is finite. -/
theorem HingeSumFinite.isFinite_fold {n : ℕ} {t c : Fin n → Value}
    {x acc : Value} {xs : List (Fin n)} (h : HingeSumFinite t c x acc xs) :
    isFinite (xs.foldl (fun acc i => acc + hingeTerm c t x i) acc) = true := by
  induction h with
  | nil hacc => exact hacc
  | cons _ _ _ _ ih => exact ih

/-- Range bounds on the fold and on the final bias addition give `HingeSumFinite` over the fold
and a finite output `hingeFun t c b x`, for an IEEE format with finite input and parameters. -/
theorem isFinite_hingeFun_of_bounds (hformat : format.isIEEE = true) {n : ℕ}
    {t c : Fin n → Value} {b x : Value}
    (hx : isFinite x = true) (ht : ∀ i, isFinite (t i) = true)
    (hc : ∀ i, isFinite (c i) = true) (hb : isFinite b = true)
    (hbound : HingeSumBounds t c x (0 : Value) (List.finRange n))
    (hout : |(toModel (hingeSum c t x)).toReal + (toModel b).toReal| ≤
      (Model.posMaxFinite format).toReal) :
    HingeSumFinite t c x (0 : Value) (List.finRange n) ∧
      isFinite (hingeFun t c b x) = true := by
  have hsum :=
    hingeSumFinite_of_bounds hformat hx ht hc hbound ExecFloat.Binary.isFinite_zero
  refine ⟨hsum, ?_⟩
  change Model.isFinite (toModel (hingeSum c t x + b)) = true
  rw [ExecFloat.Binary.toModel_hAdd (hingeSum c t x) b]
  exact Model.isFinite_add_of_abs_toReal_add_le_posMaxFinite _ _ hformat
    (by
      change isFinite (hingeSum c t x) = true
      simpa only [hingeSum, Fin.foldl_eq_foldl_finRange] using hsum.isFinite_fold) hb hout

/-! ## Refinement to rounded-real execution -/

/-- Addition in `UniversalApproximation.Rounded format` rounds the real sum with
`Model.roundAt format`, the rounding that finite configured binary addition performs. -/
private theorem rounded_add_val (a b : RoundedValue) :
    (a + b).val = Model.roundAt format (a.val + b.val) := rfl

/-- Subtraction in `UniversalApproximation.Rounded format` rounds the real difference with
`Model.roundAt format`. -/
private theorem rounded_sub_val (a b : RoundedValue) :
    (a - b).val = Model.roundAt format (a.val - b.val) := rfl

/-- Multiplication in `UniversalApproximation.Rounded format` rounds the real product with
`Model.roundAt format`. -/
private theorem rounded_mul_val (a b : RoundedValue) :
    (a * b).val = Model.roundAt format (a.val * b.val) := rfl

/-- Extensionality for the rounded-real wrapper. -/
private theorem rounded_ext {u v : RoundedValue} (h : u.val = v.val) : u = v := by
  cases u; cases v; cases h; rfl

/-- On a finite configured value `x`, `relu x` read as a real is the real ReLU of `x`: by
`relu_eq_max_binary` it is IEEE `maximum x 0`, which is the real `max` on finite operands. -/
private theorem toReal_relu {x : Value}
    (hx : isFinite x = true) :
    (toModel (relu x)).toReal = relu ((toModel x).toReal) := by
  have h0 : isFinite (0 : Value) = true := ExecFloat.Binary.isFinite_zero
  have hmax :
      (toModel (max x (0 : Value))).toReal =
        max ((toModel x).toReal) ((toModel (0 : Value)).toReal) :=
    toReal_maximum_eq_max (x := x)
      (y := (0 : Value)) hx h0
  -- `relu` is `max _ 0` on reals.
  simpa only [relu_eq_max_binary, Activation.relu_eq_max, toModel_zero, Model.toReal_zero]
    using hmax

/--
Refine one executable hinge term to the corresponding rounded-real hinge term.

When the difference `x - tᵢ` and the product are finite, subtraction and multiplication each
round once with `Model.roundAt format`, and ReLU of the finite difference is exact.
-/
private theorem toReal_hingeTerm (hformat : format.isIEEE = true) {n : ℕ}
    (t c : Fin n → Value) (x : Value) (i : Fin n)
    (hsub : isFinite (x - t i) = true)
    (hmul : isFinite (hingeTerm c t x i) = true) :
    (toModel (hingeTerm c t x i)).toReal =
      (hingeTerm (toRoundedVec c) (toRoundedVec t) (toRounded x) i).val := by
  have hsubR :
      (toModel (x - t i)).toReal =
        Model.roundAt format ((toModel x).toReal - (toModel (t i)).toReal) :=
    ExecFloat.Binary.toReal_sub_eq_roundAt (left := x) (right := t i) hsub hformat
  -- ReLU is `maximum _ 0`, which is exact as `max` on reals for finite inputs.
  have hreluR :
      (toModel (relu (x - t i))).toReal = relu ((toModel (x - t i)).toReal) :=
    toReal_relu hsub
  have hmulR :
      (toModel (c i * relu (x - t i))).toReal =
        Model.roundAt format
          ((toModel (c i)).toReal * (toModel (relu (x - t i))).toReal) :=
    ExecFloat.Binary.toReal_mul_eq_roundAt (left := c i)
      (right := relu (x - t i)) hmul hformat
  -- The rounded-real hinge term is the same `Model.roundAt format` expression: `relu` is
  -- exact on `.val`, and RoundedValue `-`/`*` round with the same operation.
  have htermRounded :
      (hingeTerm (toRoundedVec c) (toRoundedVec t) (toRounded x) i).val =
        Model.roundAt format
          ((toModel (c i)).toReal *
            relu
              (Model.roundAt format ((toModel x).toReal -
                (toModel (t i)).toReal))) := by
    simp only [hingeTerm, rounded_mul_val, relu_val, rounded_sub_val, toRoundedVec, toRounded]
  calc
    (toModel (hingeTerm c t x i)).toReal
        = Model.roundAt format
            ((toModel (c i)).toReal * (toModel (relu (x - t i))).toReal) :=
          hmulR
    _ = Model.roundAt format
            ((toModel (c i)).toReal * relu ((toModel (x - t i)).toReal)) := by
          rw [hreluR]
    _ = Model.roundAt format
            ((toModel (c i)).toReal *
              relu
                (Model.roundAt format ((toModel x).toReal -
                  (toModel (t i)).toReal))) := by
          rw [hsubR]
    _ = (hingeTerm (toRoundedVec c) (toRoundedVec t) (toRounded x) i).val :=
          htermRounded.symm

/--
Refine an executable hinge-term fold to the rounded-real fold with the same order.

Given a `HingeSumFinite` witness for the index list `xs` and start accumulator `acc`, the two
folds agree after `toReal`. Floating-point addition is not associative, so the statement keeps
the fold order of `xs` and never passes to an unordered finite sum.
-/
theorem HingeSumFinite.toReal_fold (hformat : format.isIEEE = true) {n : ℕ}
    (t c : Fin n → Value) (x : Value)
    {acc : Value} {xs : List (Fin n)}
    (hfin : HingeSumFinite t c x acc xs) :
    (toModel (xs.foldl (fun acc i => acc + hingeTerm c t x i) acc)).toReal =
      (xs.foldl
        (fun accRounded i => accRounded + hingeTerm (toRoundedVec c) (toRoundedVec t)
          (toRounded x) i) (toRounded acc)).val := by
  induction hfin with
  | nil hacc =>
      simp [List.foldl]
  | cons hsub hmul hadd hrest ih =>
      rename_i acc i xs
      let termI : Value := hingeTerm c t x i
      let termRounded : RoundedValue := hingeTerm (toRoundedVec c) (toRoundedVec t) (toRounded x) i
      have hterm : (toModel termI).toReal = termRounded.val := by
        simpa [termI, termRounded] using
          (toReal_hingeTerm hformat t c x i hsub hmul)
      have haddR :
          (toModel (acc + termI)).toReal =
            Model.roundAt format ((toModel acc).toReal + (toModel termI).toReal) :=
        ExecFloat.Binary.toReal_add_eq_roundAt (left := acc) (right := termI) hadd hformat
      have hroundedStep :
          ((toRounded acc) + termRounded).val =
            Model.roundAt format ((toModel acc).toReal + termRounded.val) :=
        rounded_add_val (toRounded acc) termRounded
      -- `toRounded` exposes `toReal`, and both additions round with the same operation.
      have hstart : toRounded (acc + termI) = (toRounded acc) + termRounded :=
        rounded_ext (by rw [toRounded_val, haddR, hroundedStep, hterm])
      -- Unfold one fold step and apply the IH (whose start accumulator is
      -- `toRounded (add acc termI)`), then rewrite that start accumulator to
      -- `toRounded acc + termRounded`.
      simpa [List.foldl, termI, termRounded, hstart] using ih

/--
Refine the whole executable hinge network to the rounded-real network.

With a `HingeSumFinite` witness for the fold and a finite output, the executable result read through
`toReal` is the value of `hingeFun` in `UniversalApproximation.Rounded format` on the embedded
parameters and input. The proof composes the fold refinement with the rounded bias addition.
-/
theorem toReal_hingeFun (hformat : format.isIEEE = true) {n : ℕ}
    (t c : Fin n → Value) (b x : Value)
    (hSum :
      HingeSumFinite t c x (0 : Value) (List.finRange n))
    (hOut : isFinite (hingeFun t c b x) = true) :
    (toModel (hingeFun t c b x)).toReal =
      (hingeFun (toRoundedVec t) (toRoundedVec c) (toRounded b) (toRounded x)).val := by
  -- First refine the hinge sum.
  have hsum :
      (toModel (hingeSum c t x)).toReal =
      (hingeSum (toRoundedVec c) (toRoundedVec t) (toRounded x)).val := by
    have hstart : toRounded (0 : Value) = (0 : RoundedValue) := rounded_ext (by simp [toRounded])
    simpa only [hingeSum, Fin.foldl_eq_foldl_finRange, hstart] using
      HingeSumFinite.toReal_fold hformat t c x hSum
  -- Finally refine the last `+ b`.
  have haddR :
      (toModel (hingeSum c t x + b)).toReal =
        Model.roundAt format ((toModel (hingeSum c t x)).toReal + (toModel b).toReal) :=
    ExecFloat.Binary.toReal_add_eq_roundAt (left := hingeSum c t x) (right := b) hOut hformat
  -- `hingeFun` is the hinge sum plus `b`, and `+` on `RoundedValue` rounds with
  -- `Model.roundAt format`.
  have hfp :
      (hingeFun (toRoundedVec t) (toRoundedVec c) (toRounded b) (toRounded x)).val =
        Model.roundAt format
          ((hingeSum (toRoundedVec c) (toRoundedVec t) (toRounded x)).val + (toRounded b).val) := by
    rw [hingeFun, rounded_add_val]
  rw [hingeFun, haddR, hfp, hsum, toRounded_val]

/-! ## Configured binary error bound inherited from the rounded-real bound -/

/--
Lift the rounded-real hinge-network error bound to executable evaluation.

Under the finiteness witnesses, `toReal_hingeFun` identifies the
executable result with the rounded-real model, whose bound `hingeFunErrorBound format` against
`hingeFunReal format` then applies unchanged.
-/
theorem hingeFun_abs_error_binary (hformat : format.isIEEE = true) {n : ℕ}
    (t c : Fin n → Value) (b x : Value)
    (hSum :
      HingeSumFinite t c x (0 : Value) (List.finRange n))
    (hOut : isFinite (hingeFun t c b x) = true) :
    |(toModel (hingeFun t c b x)).toReal -
        hingeFunReal format (toRoundedVec t) (toRoundedVec c) (toRounded b) (toRounded x)| ≤
      hingeFunErrorBound format (toRoundedVec t) (toRoundedVec c) (toRounded b) (toRounded x) := by
  rw [toReal_hingeFun hformat t c b x hSum hOut]
  exact hingeFun_abs_error format (t := toRoundedVec t) (c := toRoundedVec c) (b := toRounded b)
    (x := toRounded x)

/-! ## “Approximation error + rounding error” (pointwise) -/

/--
Pointwise triangle bound: under the finiteness witnesses, the distance from `f` to the executable
output is at most the real approximation error plus the IEEE rounding bound
`hingeFunErrorBound format`.
-/
theorem abs_sub_hingeFun_le_binary (hformat : format.isIEEE = true) {n : ℕ} (f : ℝ → ℝ)
    (t c : Fin n → Value) (b x : Value)
    (hSum :
      HingeSumFinite t c x (0 : Value) (List.finRange n))
    (hOut : isFinite (hingeFun t c b x) = true) :
    |f ((toModel x).toReal) - (toModel (hingeFun t c b x)).toReal| ≤
      |f ((toModel x).toReal) - hingeFunReal format (toRoundedVec t) (toRoundedVec c) (toRounded b)
          (toRounded x)| +
        hingeFunErrorBound format (toRoundedVec t) (toRoundedVec c) (toRounded b)
          (toRounded x) := by
  have hround :
      |hingeFunReal format (toRoundedVec t) (toRoundedVec c) (toRounded b) (toRounded x) -
          (toModel (hingeFun t c b x)).toReal| ≤
        hingeFunErrorBound format (toRoundedVec t) (toRoundedVec c) (toRounded b)
          (toRounded x) := by
    rw [abs_sub_comm]
    exact hingeFun_abs_error_binary hformat t c b x hSum hOut
  have htri :
      |f ((toModel x).toReal) - (toModel (hingeFun t c b x)).toReal| ≤
        |f ((toModel x).toReal) -
          hingeFunReal format (toRoundedVec t) (toRoundedVec c) (toRounded b) (toRounded x)| +
          |hingeFunReal format (toRoundedVec t) (toRoundedVec c) (toRounded b) (toRounded x) -
              (toModel (hingeFun t c b x)).toReal| :=
    abs_sub_le _ _ _
  exact le_trans htri (add_le_add_right hround _)

/-- Strict version of `abs_sub_hingeFun_le_binary`: if the real network is less than `ε`
from `f`, the executable output is less than `ε + hingeFunErrorBound format` from it. -/
theorem abs_sub_hingeFun_lt_binary (hformat : format.isIEEE = true) {n : ℕ} (f : ℝ → ℝ)
    (t c : Fin n → Value) (b x : Value)
    (hSum :
      HingeSumFinite t c x (0 : Value) (List.finRange n))
    (hOut : isFinite (hingeFun t c b x) = true)
    {ε : ℝ}
    (hε :
      |f ((toModel x).toReal) -
          hingeFunReal format (toRoundedVec t) (toRoundedVec c) (toRounded b) (toRounded x)| < ε) :
    |f ((toModel x).toReal) - (toModel (hingeFun t c b x)).toReal| <
      ε + hingeFunErrorBound format (toRoundedVec t) (toRoundedVec c) (toRounded b) (toRounded x) :=
  lt_of_le_of_lt (abs_sub_hingeFun_le_binary hformat f t c b x hSum hOut)
    (add_lt_add_of_lt_of_le hε le_rfl)

/-! ## Configured binary pointwise ReLU approximation packaging (1D) -/

/--
One-dimensional ReLU approximation over configured binary values.

If some executable hinge parameters `(t, c, b0)` have finite intermediates and a finite output at
every finite input in `[a, b]`, and their real network `hingeFunReal format` on the embedded
values is less than `ε` from `f` there, then executable evaluation with those parameters is less
than `ε + hingeFunErrorBound format` from `f` on the same inputs. Their existence is a
hypothesis: the theorem repackages `abs_sub_hingeFun_lt_binary` and builds no IEEE
weights from `f`.
-/
theorem relu_Icc_binary_of_hinge (hformat : format.isIEEE = true)
    {f : ℝ → ℝ} {a b : ℝ} :
    ∀ ε : ℝ,
      (∃ (hidDim : ℕ) (t c : Fin hidDim → Value) (b0 : Value),
          (∀ x : Value,
              isFinite x = true →
              (toModel x).toReal ∈ Set.Icc a b →
                HingeSumFinite t c x (0 : Value) (List.finRange hidDim) ∧
                isFinite (hingeFun t c b0 x) = true) ∧
          (∀ x : Value,
              isFinite x = true →
              (toModel x).toReal ∈ Set.Icc a b →
                |f ((toModel x).toReal) -
                    hingeFunReal format (toRoundedVec t) (toRoundedVec c) (toRounded b0)
                      (toRounded x)| < ε))
      →
      ∃ (hidDim : ℕ) (t c : Fin hidDim → Value) (b0 : Value),
        ∀ x : Value,
          isFinite x = true →
          (toModel x).toReal ∈ Set.Icc a b →
            |f ((toModel x).toReal) -
                (toModel (hingeFun t c b0 x)).toReal| <
              ε +
                hingeFunErrorBound format (toRoundedVec t) (toRoundedVec c) (toRounded b0)
                  (toRounded x) := by
  rintro ε ⟨hidDim, t, c, b0, hFinite, hApprox⟩
  refine ⟨hidDim, t, c, b0, fun x hx hxIn => ?_⟩
  obtain ⟨hSum, hOut⟩ := hFinite x hx hxIn
  exact abs_sub_hingeFun_lt_binary hformat f t c b0 x hSum hOut
    (hApprox x hx hxIn)

/-! ## Real approximation + quantization + IEEE rounding (1D, pointwise) -/

/--
One-dimensional ReLU approximation by an executable hinge network, with the error split into
three terms around a real reference network with caller-supplied bias `bR`:

1. **Real approximation error**:
   $|f(r)-\operatorname{hinge\_fun}(\ldots,r)|<\varepsilon_{\mathrm{approx}}$.
2. **Quantization/reference error**:
   $|\operatorname{hinge\_fun}(\ldots,r)
   -\operatorname{hinge\_fun}_{\mathbb R}(\operatorname{embed}(\text{IEEE parameters}),
     \operatorname{embed}(r))|\leq\varepsilon_Q$.
3. **IEEE rounding error** (proved): `hingeFunErrorBound format`.

The caller supplies the executable parameters `(t, c, b0)` and proves the first two bounds and
the finiteness witnesses (`HingeSumFinite` and a finite output) at every finite input in
`[a, b]`.
-/
theorem relu_Icc_binary_bias (hformat : format.isIEEE = true)
    {f : ℝ → ℝ} {a b : ℝ} {hidDim : ℕ}
    (bR : ℝ) (tR cR : Fin hidDim → ℝ)
    (t c : Fin hidDim → Value) (b0 : Value) :
    ∀ εApprox εQ : ℝ,
      (∀ x : Value,
          isFinite x = true →
          (toModel x).toReal ∈ Set.Icc a b →
            HingeSumFinite t c x (0 : Value) (List.finRange hidDim) ∧
            isFinite (hingeFun t c b0 x) = true) →
      (∀ x : Value,
          isFinite x = true →
          (toModel x).toReal ∈ Set.Icc a b →
            |f ((toModel x).toReal) - hingeFun tR cR bR ((toModel x).toReal)| < εApprox) →
      (∀ x : Value,
          isFinite x = true →
          (toModel x).toReal ∈ Set.Icc a b →
            |hingeFun tR cR bR ((toModel x).toReal) -
                hingeFunReal format (toRoundedVec t) (toRoundedVec c) (toRounded b0)
                    (toRounded x)| ≤
                  εQ) →
      ∀ x : Value,
        isFinite x = true →
        (toModel x).toReal ∈ Set.Icc a b →
          |f ((toModel x).toReal) -
              (toModel (hingeFun t c b0 x)).toReal| <
            (εApprox + εQ) +
              hingeFunErrorBound format (toRoundedVec t) (toRoundedVec c) (toRounded b0)
                (toRounded x) := by
  intro εApprox εQ hFinite hApprox hQ x hx hxIn
  obtain ⟨hSum, hOut⟩ := hFinite x hx hxIn
  -- Triangle inequality through the real hinge network with the reference bias `bR`.
  have hApproxEmbed :
      |f ((toModel x).toReal) -
          hingeFunReal format (toRoundedVec t) (toRoundedVec c) (toRounded b0) (toRounded x)| <
        εApprox + εQ :=
    lt_of_le_of_lt (abs_sub_le _ (hingeFun tR cR bR ((toModel x).toReal)) _)
      (add_lt_add_of_lt_of_le (hApprox x hx hxIn) (hQ x hx hxIn))
  exact abs_sub_hingeFun_lt_binary hformat f t c b0 x hSum hOut hApproxEmbed

/--
Specialization of `relu_Icc_binary_bias` to the real bias `f a`, which is
the bias the constructive one-dimensional interpolation theorem emits.
-/
theorem relu_Icc_binary_three_term (hformat : format.isIEEE = true)
    {f : ℝ → ℝ} {a b : ℝ} {hidDim : ℕ}
    (tR cR : Fin hidDim → ℝ)
    (t c : Fin hidDim → Value) (b0 : Value) :
    ∀ εApprox εQ : ℝ,
      (∀ x : Value,
          isFinite x = true →
          (toModel x).toReal ∈ Set.Icc a b →
            HingeSumFinite t c x (0 : Value) (List.finRange hidDim) ∧
            isFinite (hingeFun t c b0 x) = true) →
      (∀ x : Value,
          isFinite x = true →
          (toModel x).toReal ∈ Set.Icc a b →
            |f ((toModel x).toReal) - hingeFun tR cR (f a) ((toModel x).toReal)| <
              εApprox) →
      (∀ x : Value,
          isFinite x = true →
          (toModel x).toReal ∈ Set.Icc a b →
            |hingeFun tR cR (f a) ((toModel x).toReal) -
                hingeFunReal format (toRoundedVec t) (toRoundedVec c) (toRounded b0)
                    (toRounded x)| ≤
                  εQ) →
      ∀ x : Value,
        isFinite x = true →
        (toModel x).toReal ∈ Set.Icc a b →
          |f ((toModel x).toReal) -
              (toModel (hingeFun t c b0 x)).toReal| <
            (εApprox + εQ) +
              hingeFunErrorBound format (toRoundedVec t) (toRoundedVec c) (toRounded b0)
                (toRounded x) :=
  relu_Icc_binary_bias hformat (f a) tR cR t c b0


/-! ## Dyadic quantization helpers -/

/--
Half-ulp error bound for dyadic values rounded into executable configured binary values.

Finiteness of the rounded value rules out overflow, so the executable rounding of a dyadic agrees
with `Model.roundAt format` and FloatLib's rounding bound `Model.abs_roundAt_sub_le` applies.
-/
theorem abs_roundDyadic_sub_le (hformat : format.isIEEE = true)
    (d : FloatLib.Numerics.Dyadic)
    (hfin :
      isFinite (ofModel (Model.roundDyadic format d) : Value) =
        true) :
    abs ((toModel (ofModel (Model.roundDyadic format d) :
          Value)).toReal - Dyadic.toReal d) ≤
      ulp binaryRadix (Model.fexpOf format) (Dyadic.toReal d) / 2 := by
  rw [toReal_roundDyadic_eq_round hfin]
  exact Model.abs_roundAt_sub_le format _

/-!
### Real hinge network sensitivity to parameter rounding (for a compact input domain)

These lemmas bound the quantization term `εQ` of the three-term theorems.
The lemma below uses the compact-domain assumption that the *reference* knot locations `tR`
lie in the input interval `[a,b]`, then bounds the output perturbation using:
- a width term `|b-a|` to bound `relu (x - tR i)` on the interval, and
- the 1-Lipschitz bound `abs_max_sub_max_le_abs` to control the sensitivity to perturbing `t`.
-/

/-- On `[a,b]`, a hinge activation `relu (x - t)` is bounded by the interval width. -/
private theorem abs_relu_sub_le {a b x t : ℝ}
    (hx : x ∈ Set.Icc a b) (ht : t ∈ Set.Icc a b) :
    abs (relu (x - t)) ≤ abs (b - a) := by
  calc
    abs (relu (x - t)) = relu (x - t) :=
        abs_of_nonneg (by rw [Activation.relu_eq_max]; exact le_max_right _ _)
    _ ≤ abs (x - t) := by
        rw [Activation.relu_eq_max]; exact max_le (le_abs_self _) (abs_nonneg _)
    -- Two points of `[a,b]` are at most the interval width apart.
    _ ≤ abs (b - a) :=
        Set.abs_sub_le_of_uIcc_subset_uIcc
          (Set.Subset.trans (Set.uIcc_subset_Icc ht hx) Set.Icc_subset_uIcc)

/--
Sensitivity of a real hinge network to perturbing all parameters on a compact interval.

For `x` and the reference knots `tR` in `[a,b]`, the output moves by at most the bias change
plus, per neuron, the coefficient change times the width `|b-a|` and the absolute perturbed
coefficient `|cI i|` times the knot change.
-/
theorem abs_hingeFun_sub_le
    {n : ℕ} {a b x : ℝ}
    (tR cR tI cI : Fin n → ℝ) (bR bI : ℝ)
    (hx : x ∈ Set.Icc a b)
    (htR : ∀ i, tR i ∈ Set.Icc a b) :
    abs (hingeFun tR cR bR x - hingeFun tI cI bI x) ≤
      abs (bR - bI) +
        ∑ i : Fin n,
          (abs (cR i - cI i) * abs (b - a) + abs (cI i) * abs (tR i - tI i)) := by
  classical
  -- Separate bias and sum: `hingeFun = b + sum`.
  have htri0 :
      abs (hingeFun tR cR bR x - hingeFun tI cI bI x) ≤
        abs (bR - bI) +
          abs ((∑ i : Fin n, cR i * relu (x - tR i)) -
            (∑ i : Fin n, cI i * relu (x - tI i))) := by
    simpa [hingeFun_eq_add_sum, sub_eq_add_neg, add_assoc, add_left_comm, add_comm] using
      (abs_add_le (bR - bI)
        ((∑ i : Fin n, cR i * relu (x - tR i)) - (∑ i : Fin n, cI i * relu (x - tI i))))
  -- Bound the difference of sums by summing per-term bounds.
  have hsum0 :
      abs ((∑ i : Fin n, cR i * relu (x - tR i)) - (∑ i : Fin n, cI i * relu (x - tI i))) ≤
        ∑ i : Fin n, abs (cR i * relu (x - tR i) - cI i * relu (x - tI i)) := by
    rw [← Finset.sum_sub_distrib]
    exact Finset.abs_sum_le_sum_abs _ _
  have hterm :
      ∀ i : Fin n,
        abs (cR i * relu (x - tR i) - cI i * relu (x - tI i)) ≤
          abs (cR i - cI i) * abs (b - a) + abs (cI i) * abs (tR i - tI i) := by
    intro i
    -- Decompose into a coefficient perturbation term and a knot perturbation term.
    have hdecomp :
        cR i * relu (x - tR i) - cI i * relu (x - tI i) =
          (cR i - cI i) * relu (x - tR i) +
            cI i * (relu (x - tR i) - relu (x - tI i)) := by
      ring
    have htri :
        abs (cR i * relu (x - tR i) - cI i * relu (x - tI i)) ≤
          abs ((cR i - cI i) * relu (x - tR i)) +
            abs (cI i * (relu (x - tR i) - relu (x - tI i))) := by
      rw [hdecomp]
      exact abs_add_le _ _
    have hreluWidth : abs (relu (x - tR i)) ≤ abs (b - a) :=
      abs_relu_sub_le (hx := hx) (ht := htR i)
    have hreluLip :
        abs (relu (x - tR i) - relu (x - tI i)) ≤ abs (tR i - tI i) := by
      have hdiff : (x - tR i) - (x - tI i) = tI i - tR i := by ring
      calc
        abs (relu (x - tR i) - relu (x - tI i))
            ≤ abs ((x - tR i) - (x - tI i)) := by
              simpa only [Activation.relu_eq_max] using
                abs_max_sub_max_le_abs (x - tR i) (x - tI i) 0
        _ = abs (tR i - tI i) := by rw [hdiff, abs_sub_comm]
    have h1 :
        abs ((cR i - cI i) * relu (x - tR i)) ≤ abs (cR i - cI i) * abs (b - a) := by
      rw [abs_mul]
      exact mul_le_mul_of_nonneg_left hreluWidth (abs_nonneg _)
    have h2 :
        abs (cI i * (relu (x - tR i) - relu (x - tI i))) ≤
          abs (cI i) * abs (tR i - tI i) := by
      rw [abs_mul]
      exact mul_le_mul_of_nonneg_left hreluLip (abs_nonneg _)
    linarith
  -- Chain the bias+sum triangle bound with the termwise bounds.
  exact le_trans htri0 (add_le_add le_rfl (le_trans hsum0 (Finset.sum_le_sum fun i _ => hterm i)))

/--
Uniform version of `abs_hingeFun_sub_le`. If every coefficient change is at
most `Δc`, every perturbed coefficient is at most `C` in magnitude (with `0 ≤ C`) and every knot
change is at most `Δt`, the output moves by at most `|bR-bI| + n * (Δc * |b-a| + C * Δt)`.
-/
theorem abs_hingeFun_sub_le_uniform
    {n : ℕ} {a b x : ℝ}
    (tR cR tI cI : Fin n → ℝ) (bR bI : ℝ)
    (hx : x ∈ Set.Icc a b)
    (htR : ∀ i, tR i ∈ Set.Icc a b)
    {Δc C Δt : ℝ}
    (hC0 : 0 ≤ C)
    (hΔc : ∀ i, abs (cR i - cI i) ≤ Δc)
    (hC : ∀ i, abs (cI i) ≤ C)
    (hΔt : ∀ i, abs (tR i - tI i) ≤ Δt) :
    abs (hingeFun tR cR bR x - hingeFun tI cI bI x) ≤
      abs (bR - bI) + (n : ℝ) * (Δc * abs (b - a) + C * Δt) := by
  classical
  -- Bound the per-neuron summand uniformly.
  have hterm :
      ∀ i : Fin n,
        abs (cR i - cI i) * abs (b - a) + abs (cI i) * abs (tR i - tI i) ≤
          Δc * abs (b - a) + C * Δt := by
    intro i
    have h1 : abs (cR i - cI i) * abs (b - a) ≤ Δc * abs (b - a) :=
      mul_le_mul_of_nonneg_right (hΔc i) (abs_nonneg (b - a))
    have h2 : abs (cI i) * abs (tR i - tI i) ≤ C * Δt :=
      le_trans (mul_le_mul_of_nonneg_right (hC i) (abs_nonneg (tR i - tI i)))
        (mul_le_mul_of_nonneg_left (hΔt i) hC0)
    exact add_le_add h1 h2
  have hsum' :
      (∑ _i : Fin n, (Δc * abs (b - a) + C * Δt)) = (n : ℝ) * (Δc * abs (b - a) + C * Δt) := by
    simp [mul_add]
  calc
    abs (hingeFun tR cR bR x - hingeFun tI cI bI x)
        ≤ abs (bR - bI) +
            ∑ i : Fin n,
              (abs (cR i - cI i) * abs (b - a) + abs (cI i) * abs (tR i - tI i)) :=
          abs_hingeFun_sub_le tR cR tI cI bR bI hx htR
    _ ≤ abs (bR - bI) + ∑ _i : Fin n, (Δc * abs (b - a) + C * Δt) :=
          add_le_add le_rfl (Finset.sum_le_sum fun i _ => hterm i)
    _ = abs (bR - bI) + (n : ℝ) * (Δc * abs (b - a) + C * Δt) := by
          rw [hsum']

/-! ## Removing the quantization term when reals are exactly representable -/

/--
The real network `hingeFunReal format` on embedded executable values is the real hinge network
`hingeFun` on the `toReal` values of the parameters and the input.
-/
theorem hingeFunReal_toRounded {n : ℕ}
    (t c : Fin n → Value) (b x : Value) :
    hingeFunReal format (toRoundedVec t) (toRoundedVec c) (toRounded b) (toRounded x) =
      hingeFun
        (fun i => (toModel (t i)).toReal)
        (fun i => (toModel (c i)).toReal)
        ((toModel b).toReal)
        ((toModel x).toReal) := by
  classical
  -- `hingeSumReal format` is a sum of `hingeTermReal format`, and the embeddings expose the
  -- `toReal` values.
  have hsum :
      hingeSumReal format (c := toRoundedVec c) (t := toRoundedVec t) (x := toRounded x) =
        ∑ i : Fin n,
          ((toModel (c i)).toReal) * relu ((toModel x).toReal - (toModel (t i)).toReal) := by
    simpa [hingeTermReal, toRoundedVec, toRounded] using
      (hingeSumReal_eq_sum format (c := toRoundedVec c) (t := toRoundedVec t) (x := toRounded x))
  -- Commute the final `+ b`.
  rw [hingeFunReal, hingeFun_eq_add_sum, hsum, toRounded_val, add_comm]

/--
Uniform bound on the quantization term `εQ` between a real reference network and executable
parameters. For an input and reference knots in `[a, b]`, if every coefficient and knot error is
at most `Δc` and `Δt` and every executable coefficient is at most `C` in magnitude, the two
networks differ by at most `|bR - b0| + n * (Δc * |b-a| + C * Δt)`, with `b0` read through
`toReal`.
-/
theorem hingeFun_quantization_le
    {n : ℕ} {a b : ℝ}
    (tR cR : Fin n → ℝ) (bR : ℝ)
    (t c : Fin n → Value) (b0 x : Value)
    (hxIn : (toModel x).toReal ∈ Set.Icc a b)
    (htR : ∀ i, tR i ∈ Set.Icc a b)
    {Δc C Δt : ℝ}
    (hC0 : 0 ≤ C)
    (hΔc : ∀ i, abs (cR i - (toModel (c i)).toReal) ≤ Δc)
    (hC : ∀ i, abs ((toModel (c i)).toReal) ≤ C)
    (hΔt : ∀ i, abs (tR i - (toModel (t i)).toReal) ≤ Δt) :
    abs (hingeFun tR cR bR ((toModel x).toReal) -
          hingeFunReal format (toRoundedVec t) (toRoundedVec c) (toRounded b0) (toRounded x)) ≤
      abs (bR - (toModel b0).toReal) + (n : ℝ) * (Δc * abs (b - a) + C * Δt) := by
  -- Rewrite the embedded reference as `hingeFun` on the `toReal` parameters, then apply the
  -- uniform parameter-perturbation bound on `[a,b]`.
  rw [hingeFunReal_toRounded]
  exact abs_hingeFun_sub_le_uniform (tR := tR) (cR := cR)
    (tI := fun i => (toModel (t i)).toReal) (cI := fun i => (toModel (c i)).toReal)
    (bR := bR) (bI := (toModel b0).toReal) hxIn htR hC0 hΔc hC hΔt

/-!
### Dyadic-to-configured binary quantization bound (`εQ`)

When the executable parameters are dyadic rationals rounded with `Model.roundDyadic format`, these
lemmas discharge the `εQ` premise of `relu_Icc_binary_bias`, with the
dyadic parameters as the real reference.
-/

/--
Dyadic quantization error for a hinge network on `[a,b]`.

The real reference uses the exact dyadic parameters and `hingeFunReal format` uses those dyadics
rounded to configured binary values. Their gap is at most the bias rounding error plus, per
neuron, the coefficient rounding error times `|b-a|` and the absolute rounded coefficient times
the knot rounding error.
-/
theorem quantization_dyadic_le
    {n : ℕ} {a b : ℝ}
    (tD cD : Fin n → FloatLib.Numerics.Dyadic) (bD : FloatLib.Numerics.Dyadic)
    (x : Value)
    (hxIn : (toModel x).toReal ∈ Set.Icc a b)
    (htIn : ∀ i, Dyadic.toReal (tD i) ∈ Set.Icc a b) :
    abs
        (hingeFun
            (fun i => Dyadic.toReal (tD i))
            (fun i => Dyadic.toReal (cD i))
            (Dyadic.toReal bD)
            ((toModel x).toReal) -
          hingeFunReal format
            (t := toRoundedVec (fun i =>
              (ofModel (Model.roundDyadic format (tD i)) : Value)))
            (c := toRoundedVec (fun i =>
              (ofModel (Model.roundDyadic format (cD i)) : Value)))
            (b := toRounded (ofModel (Model.roundDyadic format bD) :
              Value))
            (x := toRounded x)) ≤
      abs (Dyadic.toReal bD -
          (toModel (ofModel (Model.roundDyadic format bD) :
            Value)).toReal) +
        ∑ i : Fin n,
          (abs (Dyadic.toReal (cD i) -
              (toModel (ofModel (Model.roundDyadic format (cD i)) :
                Value)).toReal) *
              abs (b - a) +
            abs ((toModel (ofModel (Model.roundDyadic format (cD i)) :
                Value)).toReal) *
              abs (Dyadic.toReal (tD i) -
                (toModel (ofModel (Model.roundDyadic format (tD i)) :
                  Value)).toReal)) := by
  classical
  -- Rewrite the `hingeFunReal format` reference as `hingeFun` over the embedded reals, then apply
  -- the general parameter-sensitivity bound on `[a,b]`.
  rw [hingeFunReal_toRounded]
  exact abs_hingeFun_sub_le
    (tR := fun i => Dyadic.toReal (tD i))
    (cR := fun i => Dyadic.toReal (cD i))
    (tI := fun i =>
      (toModel (ofModel (Model.roundDyadic format (tD i)) :
        Value)).toReal)
    (cI := fun i =>
      (toModel (ofModel (Model.roundDyadic format (cD i)) :
        Value)).toReal)
    (bR := Dyadic.toReal bD)
    (bI := (toModel (ofModel (Model.roundDyadic format bD) :
      Value)).toReal)
    hxIn htIn

/--
Dyadic quantization error with each per-parameter rounding error bounded by half an ulp.

Callers supply finiteness of every rounded dyadic, and `abs_roundDyadic_sub_le`
bounds each term of `quantization_dyadic_le`.
-/
theorem quantization_half_ulp_le (hformat : format.isIEEE = true)
    {n : ℕ} {a b : ℝ}
    (tD cD : Fin n → FloatLib.Numerics.Dyadic) (bD : FloatLib.Numerics.Dyadic)
    (x : Value)
    (hxIn : (toModel x).toReal ∈ Set.Icc a b)
    (htIn : ∀ i, Dyadic.toReal (tD i) ∈ Set.Icc a b)
    (htfin : ∀ i,
      isFinite (ofModel (Model.roundDyadic format (tD i)) : Value) =
        true)
    (hcfin : ∀ i,
      isFinite (ofModel (Model.roundDyadic format (cD i)) : Value) =
        true)
    (hbfin :
      isFinite (ofModel (Model.roundDyadic format bD) : Value) =
        true) :
    abs
        (hingeFun
            (fun i => Dyadic.toReal (tD i))
            (fun i => Dyadic.toReal (cD i))
            (Dyadic.toReal bD)
            ((toModel x).toReal) -
          hingeFunReal format
            (t := toRoundedVec (fun i =>
              (ofModel (Model.roundDyadic format (tD i)) : Value)))
            (c := toRoundedVec (fun i =>
              (ofModel (Model.roundDyadic format (cD i)) : Value)))
            (b := toRounded (ofModel (Model.roundDyadic format bD) :
              Value))
            (x := toRounded x)) ≤
      ulp binaryRadix (Model.fexpOf format) (Dyadic.toReal bD) / 2 +
        ∑ i : Fin n,
          (ulp binaryRadix (Model.fexpOf format) (Dyadic.toReal (cD i)) / 2
            *
              abs (b - a) +
            abs ((toModel (ofModel (Model.roundDyadic format (cD i)) :
                Value)).toReal) *
              (ulp binaryRadix (Model.fexpOf format) (Dyadic.toReal (tD i))
                / 2)) := by
  classical
  -- Start from the raw parameter-sensitivity bound.
  have hraw :=
    quantization_dyadic_le (tD := tD) (cD := cD) (bD := bD) (x := x) hxIn htIn
  -- Bound each parameter-difference term by a half-ulp using
  -- `abs_roundDyadic_sub_le`.
  have hb :
      abs (Dyadic.toReal bD -
          (toModel (ofModel (Model.roundDyadic format bD) :
            Value)).toReal) ≤
        ulp binaryRadix (Model.fexpOf format) (Dyadic.toReal bD) / 2 := by
    rw [abs_sub_comm]
    exact abs_roundDyadic_sub_le hformat bD hbfin
  have hterm :
      ∀ i : Fin n,
        (abs (Dyadic.toReal (cD i) -
            (toModel (ofModel (Model.roundDyadic format (cD i)) :
              Value)).toReal) *
              abs (b - a) +
            abs ((toModel (ofModel (Model.roundDyadic format (cD i)) :
                Value)).toReal) *
              abs (Dyadic.toReal (tD i) -
                (toModel (ofModel (Model.roundDyadic format (tD i)) :
                  Value)).toReal))
          ≤
            (ulp binaryRadix (Model.fexpOf format) (Dyadic.toReal (cD i)) /
              2 *
                abs (b - a) +
              abs ((toModel (ofModel (Model.roundDyadic format (cD i)) :
                  Value)).toReal) *
                (ulp binaryRadix (Model.fexpOf format)
                  (Dyadic.toReal (tD i)) / 2)) := by
    intro i
    have hc :
        abs (Dyadic.toReal (cD i) -
            (toModel (ofModel (Model.roundDyadic format (cD i)) :
              Value)).toReal) ≤
          ulp binaryRadix (Model.fexpOf format) (Dyadic.toReal (cD i)) / 2
            := by
      rw [abs_sub_comm]
      exact abs_roundDyadic_sub_le hformat (cD i) (hcfin i)
    have ht :
        abs (Dyadic.toReal (tD i) -
            (toModel (ofModel (Model.roundDyadic format (tD i)) :
              Value)).toReal) ≤
          ulp binaryRadix (Model.fexpOf format) (Dyadic.toReal (tD i)) / 2
            := by
      rw [abs_sub_comm]
      exact abs_roundDyadic_sub_le hformat (tD i) (htfin i)
    exact add_le_add (mul_le_mul_of_nonneg_right hc (abs_nonneg (b - a)))
      (mul_le_mul_of_nonneg_left ht (abs_nonneg _))
  -- Combine the termwise bounds under the sum, then add the bias bound.
  exact le_trans hraw (add_le_add hb (Finset.sum_le_sum fun i _ => hterm i))

/-!
### Packaged “dyadic quantization + IEEE rounding” theorem (1D, pointwise)

`relu_Icc_binary_dyadic` replaces the `εQ` premise of
`relu_Icc_binary_bias` with a bound built from two facts:
- each dyadic parameter rounds to within half an ulp (`≤ 1/2 ulp`), and
- a Lipschitz-style sensitivity bound holds for the real hinge network on `x ∈ [a,b]`.

The finiteness witnesses for evaluation (`HingeSumFinite` and a finite output) stay hypotheses.
-/

/--
Three-term approximation bound for parameters rounded from dyadics, with the quantization term
replaced by the half-ulp budget of `quantization_half_ulp_le`.
-/
theorem relu_Icc_binary_dyadic (hformat : format.isIEEE = true)
    {f : ℝ → ℝ} {a b : ℝ} {hidDim : ℕ}
    (tD cD : Fin hidDim → FloatLib.Numerics.Dyadic) (bD : FloatLib.Numerics.Dyadic)
    (htIn : ∀ i, Dyadic.toReal (tD i) ∈ Set.Icc a b)
    (htfin : ∀ i,
      isFinite (ofModel (Model.roundDyadic format (tD i)) : Value) =
        true)
    (hcfin : ∀ i,
      isFinite (ofModel (Model.roundDyadic format (cD i)) : Value) =
        true)
    (hbfin :
      isFinite (ofModel (Model.roundDyadic format bD) : Value) =
        true) :
    ∀ εApprox : ℝ,
      (∀ x : Value,
          isFinite x = true →
          (toModel x).toReal ∈ Set.Icc a b →
            HingeSumFinite
                (fun i =>
                  (ofModel (Model.roundDyadic format (tD i)) : Value))
                (fun i =>
                  (ofModel (Model.roundDyadic format (cD i)) : Value))
                x (0 : Value) (List.finRange hidDim) ∧
              isFinite
                (hingeFun
                  (t := fun i =>
                    (ofModel (Model.roundDyadic format (tD i)) :
                      Value))
                  (c := fun i =>
                    (ofModel (Model.roundDyadic format (cD i)) :
                      Value))
                  (b := (ofModel (Model.roundDyadic format bD) :
                    Value)) x) = true) →
      (∀ x : Value,
          isFinite x = true →
          (toModel x).toReal ∈ Set.Icc a b →
            |f ((toModel x).toReal) -
                hingeFun
                  (fun i => Dyadic.toReal (tD i))
                  (fun i => Dyadic.toReal (cD i))
                  (Dyadic.toReal bD)
                  ((toModel x).toReal)| < εApprox) →
      ∀ x : Value,
        isFinite x = true →
        (toModel x).toReal ∈ Set.Icc a b →
          |f ((toModel x).toReal) -
              (toModel (hingeFun
                  (t := fun i =>
                    (ofModel (Model.roundDyadic format (tD i)) :
                      Value))
                  (c := fun i =>
                    (ofModel (Model.roundDyadic format (cD i)) :
                      Value))
                  (b := (ofModel (Model.roundDyadic format bD) :
                    Value)) x)).toReal| <
            (εApprox +
              (ulp binaryRadix (Model.fexpOf format) (Dyadic.toReal bD) / 2
                +
                ∑ i : Fin hidDim,
                  (ulp binaryRadix (Model.fexpOf format)
                    (Dyadic.toReal (cD i)) / 2 *
                      abs (b - a) +
                    abs ((toModel (ofModel (Model.roundDyadic format (cD i)) :
                        Value)).toReal) *
                      (ulp binaryRadix (Model.fexpOf format)
                        (Dyadic.toReal (tD i)) / 2)))) +
              hingeFunErrorBound format
                (t := toRoundedVec (fun i =>
                  (ofModel (Model.roundDyadic format (tD i)) :
                    Value)))
                (c := toRoundedVec (fun i =>
                  (ofModel (Model.roundDyadic format (cD i)) :
                    Value)))
                (b := toRounded (ofModel (Model.roundDyadic format bD) :
                  Value))
                (x := toRounded x) := by
  intro εApprox hFinite hApprox x hx hxIn
  -- The half-ulp quantization budget, playing the role of `εQ` in the bias-parametric theorem.
  let εQdy : ℝ :=
    ulp binaryRadix (Model.fexpOf format) (Dyadic.toReal bD) / 2 +
      ∑ i : Fin hidDim,
        (ulp binaryRadix (Model.fexpOf format) (Dyadic.toReal (cD i)) / 2 *
            abs (b - a) +
          abs ((toModel (ofModel (Model.roundDyadic format (cD i)) :
              Value)).toReal) *
            (ulp binaryRadix (Model.fexpOf format) (Dyadic.toReal (tD i)) /
              2))
  have :=
    (relu_Icc_binary_bias hformat
        (f := f) (a := a) (b := b) (hidDim := hidDim)
        (bR := Dyadic.toReal bD)
        (tR := fun i => Dyadic.toReal (tD i))
        (cR := fun i => Dyadic.toReal (cD i))
        (t := fun i =>
          (ofModel (Model.roundDyadic format (tD i)) : Value))
        (c := fun i =>
          (ofModel (Model.roundDyadic format (cD i)) : Value))
        (b0 := (ofModel (Model.roundDyadic format bD) : Value)))
      εApprox εQdy
      hFinite
      hApprox
      (fun x _ hxIn =>
        quantization_half_ulp_le hformat tD cD bD x
          hxIn htIn htfin hcfin hbfin)
      x hx hxIn
  -- The conclusion matches, up to unfolding `εQdy` and reassociation.
  simpa [εQdy, add_assoc] using this

/--
Two-term one-dimensional ReLU approximation by an executable hinge network. If the real hinge
network on the `toReal` values of the executable parameters is less than `εApprox` from `f` at
every finite input in `[a,b]`, and the finiteness witnesses hold there, then IEEE execution is
less than `εApprox + hingeFunErrorBound format` from `f`.

It is the three-term bound with `εQ = 0`: `hingeFunReal_toRounded` identifies
that real network with `hingeFunReal format` on the embedded parameters.
-/
theorem relu_Icc_binary_two_term (hformat : format.isIEEE = true)
    {f : ℝ → ℝ} {a b : ℝ} {hidDim : ℕ}
    (t c : Fin hidDim → Value) (b0 : Value) :
    ∀ εApprox : ℝ,
      (∀ x : Value,
          isFinite x = true →
          (toModel x).toReal ∈ Set.Icc a b →
            HingeSumFinite t c x (0 : Value) (List.finRange hidDim) ∧
            isFinite (hingeFun t c b0 x) = true) →
      (∀ x : Value,
          isFinite x = true →
          (toModel x).toReal ∈ Set.Icc a b →
            |f ((toModel x).toReal) -
                hingeFun
                  (fun i => (toModel (t i)).toReal)
                  (fun i => (toModel (c i)).toReal)
                  ((toModel b0).toReal)
                  ((toModel x).toReal)| < εApprox) →
      ∀ x : Value,
        isFinite x = true →
        (toModel x).toReal ∈ Set.Icc a b →
          |f ((toModel x).toReal) -
              (toModel (hingeFun t c b0 x)).toReal| <
            εApprox +
              hingeFunErrorBound format (toRoundedVec t) (toRoundedVec c) (toRounded b0)
                (toRounded x) := by
  intro εApprox hFinite hApprox x hx hxIn
  obtain ⟨hSum, hOut⟩ := hFinite x hx hxIn
  -- Replace `hingeFunReal format` by `hingeFun` on the `toReal` parameters and input.
  have hApproxEmbed :
      |f ((toModel x).toReal) -
          hingeFunReal format (toRoundedVec t) (toRoundedVec c) (toRounded b0) (toRounded x)| <
        εApprox := by
    rw [hingeFunReal_toRounded]
    exact hApprox x hx hxIn
  exact abs_sub_hingeFun_lt_binary hformat f t c b0 x hSum hOut hApproxEmbed

end

end TorchLean.UniversalApproximation
