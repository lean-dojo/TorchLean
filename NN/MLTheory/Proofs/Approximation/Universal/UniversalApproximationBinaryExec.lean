/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Floats.IEEEExec.Bridge.Finite
public import NN.MLTheory.Proofs.Approximation.Universal.UniversalApproximationRounded

/-!
# Executable ReLU approximation for configured IEEE formats

`hingeFunBinary` evaluates a hinge network with a configured `FloatFormat`, storage plan, and
`ModelCodec`. Refinement to `RoundedReLUApprox` requires `format.isIEEE = true` and finite
intermediates. The result preserves the exact fold order and combines real approximation,
parameter quantization, and arithmetic rounding error.

`HingeSumFinite` records the finite intermediates. `HingeSumBounds` supplies sufficient bounds
on exact subtractions, products, and partial sums to establish that witness through FloatLib's
range theorems. Dyadic quantization bounds use the selected format's rounding and half-ulp API.
-/

@[expose] public section

open FloatLib.Floats (ExecFloat)
open FloatLib.Floats.ExecFloat.Binary (isFinite ofModel toModel)
open FloatLib.Floats.Formats.BinaryInterchange (Model FloatFormat)

open FloatLib FloatLib.Numerics FloatLib.Floats.Formats
open Flocq

namespace NN.MLTheory.Proofs.UniversalApproximation

open FloatLib.Floats.Formats.BinaryInterchange

namespace BinaryExecReLUApprox

open RoundedReLUApprox

noncomputable section

variable {format : FloatFormat} {plan : Configured.StoragePlan format} {code : Type}
    [ExecFloat.ModelCodec plan (Model format) code]

local notation "Value" => ExecFloat (Configured.Family format code plan)
local notation "RoundedValue" => Rounded format

open TorchLean.Floats.IEEEExec

-- Executable values are read as reals only through the dedicated bridge lemmas below, never by
-- simp-unfolding `toModel` and `Model.toReal`.

/-! ## Embedding configured binary values into the rounded-real model -/

/-- Read an executable configured IEEE value as a point of the rounded-real model. -/
@[inline] def embed (x : Value) : RoundedValue := ⟨(toModel x).toReal⟩

@[simp] theorem embed_val (x : Value) : (embed x).val = (toModel x).toReal := rfl

/-- Pointwise embedding of a vector of `Value` values into the rounded-real model. -/
@[inline] def embedVec {n : ℕ} (v : Fin n → Value) : Fin n → RoundedValue := fun i =>
  embed (v i)

/-! ## Configured binary hinge network (same shape as `RoundedReLUApprox.hingeFun format`) -/

/-- ReLU in the executable backend, as `maximum x 0` rather than a branch, so that the IEEE 754
treatment of signed zeros and NaN comes from `maximum` itself. -/
@[inline] def reluBinary (x : Value) : Value :=
  max x (0 : Value)

/-- One hinge term `cᵢ * ReLU(x - tᵢ)` evaluated in the executable configured binary backend. -/
@[inline] def hingeTermBinary {n : ℕ} (c t : Fin n → Value)
    (x : Value) (i : Fin n) : Value :=
  ExecFloat.mul (c i) (reluBinary (ExecFloat.sub x (t i)))

/-- Fold step for summing hinge terms, used by `hingeSumBinary`. -/
@[inline] def hingeSumStepBinary {n : ℕ} (c t : Fin n → Value)
    (x : Value) : Value → Fin n → Value :=
  fun acc i => ExecFloat.add acc (hingeTermBinary c t x i)

/-- Executable configured binary sum of hinge terms, in ascending index order. -/
def hingeSumBinary {n : ℕ} (c t : Fin n → Value) (x : Value) :
    Value :=
  Fin.foldl n (hingeSumStepBinary c t x) (0 : Value)

/-- Executable configured binary hinge network: sum hinge terms, then add the executable bias. -/
def hingeFunBinary {n : ℕ} (t c : Fin n → Value) (b x : Value) :
    Value :=
  ExecFloat.add (hingeSumBinary (c := c) (t := t) x) b

/-! ## A finiteness witness for IEEE evaluation (no NaN/Inf intermediates) -/

/--
Every intermediate of the hinge fold is finite: no subtraction, ReLU, product, or partial sum
overflows to an infinity or produces a NaN.

Stating it as an inductive over the remaining index list rather than as a conjunction is what makes
it checkable by `decide` on a concrete input, via the instance `instDecHingeSumFinite`.
-/
inductive HingeSumFinite {n : ℕ} (t c : Fin n → Value)
    (x : Value) : Value → List (Fin n) → Prop where
  | nil {acc : Value} (hacc : isFinite acc = true) :
      HingeSumFinite t c x acc []
  | cons {acc : Value} {i : Fin n} {xs : List (Fin n)}
      (hsub : isFinite (ExecFloat.sub x (t i)) = true)
      (hmax : isFinite (max (ExecFloat.sub x (t i)) (0 : Value)) = true)
      (hmul : isFinite (hingeTermBinary (c := c) (t := t) x i) = true)
      (hadd : isFinite (ExecFloat.add acc (hingeTermBinary (c := c) (t := t) x i)) = true)
      (hrest :
        HingeSumFinite t c x (ExecFloat.add acc (hingeTermBinary (c := c) (t := t) x i)) xs) :
      HingeSumFinite t c x acc (i :: xs)

/-!
### Discharging `HingeSumFinite` for *concrete* networks by computation

For typical verification workflows, `t`, `c`, and `x` are *concrete* configured binary constants
coming from a lowered model artifact. In that setting, the easiest way to satisfy the
`HingeSumFinite …` hypotheses is to compute the configured binary kernel and check finiteness at
every intermediate.

`instDecHingeSumFinite` is a `Decidable` instance for `HingeSumFinite …` that walks the fold, so a
true concrete instance closes by computation:

```lean
  have hSum : HingeSumFinite t c x (0 : Value) (List.finRange n) := by
    decide
```

This does *not* solve the symbolic “no overflow for all x” problem, but it makes the pointwise
theorems in this file directly usable for concrete executions.
-/

/-- Decide `HingeSumFinite` by walking the fold, which is what lets concrete runs use `decide`. -/
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
        let sub := ExecFloat.sub x (t i)
        let mx := max sub (0 : Value)
        let term := hingeTermBinary (c := c) (t := t) x i
        let acc' := ExecFloat.add acc term
      if hsub : isFinite sub = true then
        if hmax : isFinite mx = true then
          if hmul : isFinite term = true then
            if hadd : isFinite acc' = true then
              match instDecHingeSumFinite t c x acc' xs with
              | isTrue hrest =>
                  isTrue
                    (HingeSumFinite.cons
                      (t := t) (c := c) (x := x) (acc := acc) (i := i) (xs := xs)
                      (hsub := by simpa [sub] using hsub)
                      (hmax := by simpa [mx, sub] using hmax)
                      (hmul := by simpa [term] using hmul)
                      (hadd := by simpa [acc', term] using hadd)
                      (hrest := by simpa [acc', term] using hrest))
              | isFalse hrest =>
                  isFalse (by
                    intro h
                    cases h with
                    | cons _ _ _ _ hrest' =>
                        exact hrest (by simpa [acc', term] using hrest'))
                else
                  isFalse (by
                    intro h
                    cases h with
                    | cons _ _ _ hadd' _ =>
                        dsimp [acc', term] at hadd
                        exact hadd hadd')
              else
                isFalse (by
                  intro h
                  cases h with
                  | cons _ _ hmul' _ _ =>
                      dsimp [term] at hmul
                      exact hmul hmul')
          else
            isFalse (by
              intro h
              cases h with
              | cons _ hmax' _ _ _ =>
                    dsimp [mx, sub] at hmax
                    exact hmax hmax')
        else
          isFalse (by
            intro h
            cases h with
            | cons hsub' _ _ _ _ =>
                  dsimp [sub] at hsub
                  exact hsub hsub')

/-! ### A compact finiteness hypothesis bundle (pointwise) -/

/-- The finiteness hypotheses the pointwise theorems need, bundled into one conjunction. -/
def HingeEvalFiniteProp {n : ℕ} (t c : Fin n → Value)
    (b x : Value) : Prop :=
  isFinite x = true ∧
    (∀ i, isFinite (t i) = true) ∧
    (∀ i, isFinite (c i) = true) ∧
    HingeSumFinite t c x (0 : Value) (List.finRange n) ∧
    isFinite (hingeFunBinary (t := t) (c := c) (b := b) x) = true


/-! ## Sufficient range bounds for finite intermediates -/

/-- Exact arithmetic bounds along the executable fold. Each bound compares the unrounded
operation with the selected format's largest finite value. -/
inductive HingeSumBounds {n : ℕ} (t c : Fin n → Value) (x : Value) :
    Value → List (Fin n) → Prop where
  | nil {acc : Value} : HingeSumBounds t c x acc []
  | cons {acc : Value} {i : Fin n} {xs : List (Fin n)}
      (hsub : |(toModel x).toReal - (toModel (t i)).toReal| ≤
        (Model.posMaxFinite format).toReal)
      (hmul : |(toModel (c i)).toReal| *
        |(toModel (reluBinary (ExecFloat.sub x (t i)))).toReal| ≤
          (Model.posMaxFinite format).toReal)
      (hadd : |(toModel acc).toReal + (toModel (hingeTermBinary c t x i)).toReal| ≤
        (Model.posMaxFinite format).toReal)
      (hrest : HingeSumBounds t c x (ExecFloat.add acc (hingeTermBinary c t x i)) xs) :
      HingeSumBounds t c x acc (i :: xs)

private theorem isFinite_maximum {x y : Value}
    (hx : isFinite x = true) (hy : isFinite y = true) : isFinite (max x y) = true := by
  change Model.isFinite (toModel (max x y)) = true
  rw [ExecFloat.Binary.toModel_max]
  have hxNaN := Model.isNaN_eq_false_of_isFinite_eq_true (toModel x) hx
  have hyNaN := Model.isNaN_eq_false_of_isFinite_eq_true (toModel y) hy
  have hchoose := Model.chooseNaN2_none_of_not_isNaN (toModel x) (toModel y) hxNaN hyNaN
  simp only [Model.maximum, Model.withNaNSelection_of_none _ _ hchoose]
  split
  · exact hy
  · exact hx
  · split
    · exact Model.isFinite_eq_true_of_isZero_eq_true _ (Model.isZero_zero format _)
    · exact hx

/-- Bounds on each exact subtraction, product, and partial sum rule out exceptional
intermediates, starting from finite inputs, parameters, and accumulator. -/
theorem hinge_sum_finite_of_bounds (hformat : format.isIEEE = true) {n : ℕ}
    {t c : Fin n → Value} {x acc : Value} {xs : List (Fin n)}
    (hx : isFinite x = true) (ht : ∀ i, isFinite (t i) = true)
    (hc : ∀ i, isFinite (c i) = true) (hbound : HingeSumBounds t c x acc xs) :
    isFinite acc = true → HingeSumFinite t c x acc xs := by
  induction hbound with
  | nil => exact HingeSumFinite.nil
  | @cons acc i xs hsub hmul hadd hrest ih =>
      intro hacc
      have hs : isFinite (ExecFloat.sub x (t i)) = true := by
        change Model.isFinite (toModel (ExecFloat.sub x (t i))) = true
        rw [toModel_sub]
        exact Model.isFinite_sub_of_abs_toReal_sub_le_posMaxFinite
          (toModel x) (toModel (t i)) hformat hx (ht i) hsub
      have hr : isFinite (reluBinary (ExecFloat.sub x (t i))) = true :=
        isFinite_maximum hs isFinite_zero
      have hm : isFinite (hingeTermBinary c t x i) = true := by
        change Model.isFinite (toModel (ExecFloat.mul (c i)
          (reluBinary (ExecFloat.sub x (t i))))) = true
        rw [toModel_mul]
        exact Model.isFinite_mul_of_abs_mul_le_posMaxFinite _ _ hformat (hc i) hr hmul
      have ha : isFinite (ExecFloat.add acc (hingeTermBinary c t x i)) = true := by
        change Model.isFinite (toModel (ExecFloat.add acc (hingeTermBinary c t x i))) = true
        rw [toModel_add]
        exact Model.isFinite_add_of_abs_toReal_add_le_posMaxFinite _ _ hformat hacc hm hadd
      exact HingeSumFinite.cons hs hr hm ha (ih ha)

/-- A finite-fold witness guarantees that its final accumulator is finite. -/
theorem HingeSumFinite.isFinite_fold {n : ℕ} {t c : Fin n → Value}
    {x acc : Value} {xs : List (Fin n)} (h : HingeSumFinite t c x acc xs) :
    isFinite (xs.foldl (hingeSumStepBinary c t x) acc) = true := by
  induction h with
  | nil hacc => exact hacc
  | cons _ _ _ _ _ ih => exact ih

/-- Range bounds on the fold and final bias addition discharge the complete execution witness. -/
theorem hinge_eval_finite_of_bounds (hformat : format.isIEEE = true) {n : ℕ}
    {t c : Fin n → Value} {b x : Value}
    (hx : isFinite x = true) (ht : ∀ i, isFinite (t i) = true)
    (hc : ∀ i, isFinite (c i) = true) (hb : isFinite b = true)
    (hbound : HingeSumBounds t c x (0 : Value) (List.finRange n))
    (hout : |(toModel (hingeSumBinary c t x)).toReal + (toModel b).toReal| ≤
      (Model.posMaxFinite format).toReal) : HingeEvalFiniteProp t c b x := by
  have hsum := hinge_sum_finite_of_bounds hformat hx ht hc hbound isFinite_zero
  refine ⟨hx, ht, hc, hsum, ?_⟩
  change Model.isFinite (toModel (ExecFloat.add (hingeSumBinary c t x) b)) = true
  rw [toModel_add]
  exact Model.isFinite_add_of_abs_toReal_add_le_posMaxFinite _ _ hformat
    (by
      change isFinite (hingeSumBinary c t x) = true
      simpa only [hingeSumBinary, Fin.foldl_eq_foldl_finRange] using hsum.isFinite_fold) hb hout

/-! ## Refinement to rounded-real execution -/

/-- RoundedValue addition in the rounded-`ℝ` model uses the same `Model.roundAt format`
operation as configured binary. -/
private theorem rounded_add_val (a b : RoundedValue) :
    (a + b).val = Model.roundAt format (a.val + b.val) := rfl

/-- RoundedValue subtraction in the rounded-`ℝ` model uses the same `Model.roundAt format`
operation as configured binary. -/
private theorem rounded_sub_val (a b : RoundedValue) :
    (a - b).val = Model.roundAt format (a.val - b.val) := rfl

/-- RoundedValue multiplication in the rounded-`ℝ` model uses the same `Model.roundAt format`
operation as configured binary. -/
private theorem rounded_mul_val (a b : RoundedValue) :
    (a * b).val = Model.roundAt format (a.val * b.val) := rfl

/-- Extensionality for the rounded-real wrapper. -/
private theorem rounded_ext {u v : RoundedValue} (h : u.val = v.val) : u = v := by
  cases u; cases v; cases h; rfl

/--
Finite executable ReLU agrees with real ReLU after `toReal`.

The IEEE operation here is `maximum x 0`; once NaN/Inf paths are ruled out, the bridge theorem for
`maximum` turns it into real `max`, which is exactly TorchLean's real ReLU specification.
-/
private theorem toReal_relu_binary_eq_relu {x : Value}
    (hx : isFinite x = true) :
    (toModel (reluBinary x)).toReal = relu ((toModel x).toReal) := by
  have h0 : isFinite (0 : Value) = true := isFinite_zero
  have hmax :
      (toModel (max x (0 : Value))).toReal =
        max ((toModel x).toReal) ((toModel (0 : Value)).toReal) :=
    toReal_maximum_eq_max_of_isFinite (x := x)
      (y := (0 : Value)) hx h0
  -- `relu` is `max _ 0` on reals.
  simpa only [reluBinary, relu, Activation.Math.reluSpec_eq_max, toReal_zero] using hmax

/--
Refine one executable hinge term to the corresponding rounded-real hinge term.

This is the local IEEE-754 step in the proof: subtraction rounds, ReLU is exact under finiteness,
and multiplication rounds.  The statement is deliberately per-neuron so the fold refinement below
can reuse it uniformly across `List.finRange n`.
-/
private theorem toReal_hinge_term_eq_rounded_val (hformat : format.isIEEE = true) {n : ℕ}
    (t c : Fin n → Value) (x : Value) (i : Fin n)
    (hsub : isFinite (ExecFloat.sub x (t i)) = true)
    (hmul : isFinite (hingeTermBinary (c := c) (t := t) x i) = true)
    (hx : isFinite x = true)
    (ht : isFinite (t i) = true) (_hc : isFinite (c i) = true) :
    (toModel (hingeTermBinary (c := c) (t := t) x i)).toReal =
      (hingeTerm format (embedVec c) (embedVec t) (embed x) i).val := by
  -- Subtraction refinement.
  have hsubR :
      (toModel (ExecFloat.sub x (t i))).toReal =
        Model.roundAt format ((toModel x).toReal - (toModel (t i)).toReal) :=
    toReal_sub_eq_round_of_isFinite hformat (x := x) (y := t i) hx ht hsub
  -- ReLU is `maximum _ 0`, which is exact as `max` on reals for finite inputs.
  have hreluR :
      (toModel (reluBinary (ExecFloat.sub x (t i)))).toReal =
        relu ((toModel (ExecFloat.sub x (t i))).toReal) :=
    toReal_relu_binary_eq_relu hsub
  -- Multiplication refinement.
  have hmulR :
      (toModel (ExecFloat.mul (c i) (reluBinary (ExecFloat.sub x (t i))))).toReal =
        Model.roundAt format
          ((toModel (c i)).toReal * (toModel (reluBinary (ExecFloat.sub x (t i)))).toReal) :=
    toReal_mul_eq_round_of_isFinite hformat (x := c i)
      (y := reluBinary (ExecFloat.sub x (t i)))
      (by simpa [hingeTermBinary] using hmul)
  -- The rounded-real hinge term is the same `Model.roundAt format` expression: `reluSpec` is
  -- exact on `.val`, and RoundedValue `-`/`*` round with the same operation.
  have htermRounded :
      (hingeTerm format (embedVec c) (embedVec t) (embed x) i).val =
        Model.roundAt format
          ((toModel (c i)).toReal *
            relu
              (Model.roundAt format ((toModel x).toReal -
                (toModel (t i)).toReal))) := by
    simp only [hingeTerm, rounded_mul_val, relu_rounded_val, rounded_sub_val, embedVec, embed]
  calc
    (toModel (hingeTermBinary (c := c) (t := t) x i)).toReal
        = Model.roundAt format
            ((toModel (c i)).toReal * (toModel (reluBinary (ExecFloat.sub x (t i)))).toReal) :=
          hmulR
    _ = Model.roundAt format
            ((toModel (c i)).toReal * relu ((toModel (ExecFloat.sub x (t i))).toReal)) := by
          rw [hreluR]
    _ = Model.roundAt format
            ((toModel (c i)).toReal *
              relu
                (Model.roundAt format ((toModel x).toReal -
                  (toModel (t i)).toReal))) := by
          rw [hsubR]
    _ = (hingeTerm format (embedVec c) (embedVec t) (embed x) i).val :=
          htermRounded.symm

/-- The rounded-real hinge-sum definition is the plain fold over rounded-real hinge terms. -/
private theorem hinge_sum_rounded_eq_fold {n : ℕ} (c t : Fin n → RoundedValue) (x : RoundedValue) :
    hingeSum format c t x =
      (List.finRange n).foldl (fun acc i => acc + hingeTerm format c t x i) (0 : RoundedValue) := by
  -- Peel the irrelevant components of the hinge-sum state by induction over the list.
  have :
      ∀ (xs : List (Fin n)) (accRounded : RoundedValue) (accR err : ℝ),
        (xs.foldl (hingeSumStateStep format c t x) (accRounded, accR, err)).1 =
          xs.foldl (fun acc i => acc + hingeTerm format c t x i) accRounded := by
    intro xs
    induction xs with
    | nil =>
        intro accRounded accR err
        simp [List.foldl]
    | cons i xs ih =>
        intro accRounded accR err
        simp [List.foldl, hingeSumStateStep, ih]
  simpa [hingeSum, hingeSumState, Fin.foldl_eq_foldl_finRange] using
    (this (List.finRange n) (0 : RoundedValue) 0 0)

/--
Refine an executable hinge-term fold to the rounded-real fold with the same order.

Floating-point addition is not associative, so the theorem preserves the exact `List.finRange`
evaluation order.  This is why the proof is fold-based rather than rewriting directly to an
unordered finite sum.

-/
theorem toReal_hinge_sum_binary_eq_rounded_val (hformat : format.isIEEE = true) {n : ℕ}
    (t c : Fin n → Value) (x : Value)
    {acc : Value} {xs : List (Fin n)}
    (hfin : HingeSumFinite t c x acc xs)
    (hx : isFinite x = true)
    (ht : ∀ i, isFinite (t i) = true)
    (hc : ∀ i, isFinite (c i) = true) :
    (toModel (xs.foldl (hingeSumStepBinary (c := c) (t := t) x) acc)).toReal =
      (xs.foldl
        (fun accRounded i => accRounded + hingeTerm format (embedVec c) (embedVec t) (embed x) i)
        (embed acc)).val := by
  induction hfin with
  | nil hacc =>
      simp [List.foldl]
  | cons hsub hmax hmul hadd hrest ih =>
      rename_i acc i xs
      -- Abbreviations for this step.
      let termI : Value := hingeTermBinary (c := c) (t := t) x i
      let termRounded : RoundedValue := hingeTerm format (embedVec c) (embedVec t) (embed x) i
      have hterm : (toModel termI).toReal = termRounded.val := by
        simpa [termI, termRounded] using
          (toReal_hinge_term_eq_rounded_val hformat t c x i hsub hmul hx (ht i) (hc i))
      have haddR :
          (toModel (ExecFloat.add acc termI)).toReal =
            Model.roundAt format ((toModel acc).toReal + (toModel termI).toReal) :=
        toReal_add_eq_round_of_isFinite hformat (x := acc)
          (y := termI) (by simpa [termI] using hadd)
      have hroundedStep :
          ((embed acc) + termRounded).val =
            Model.roundAt format ((toModel acc).toReal + termRounded.val) :=
        rounded_add_val (embed acc) termRounded
      -- `embed` exposes `toReal`, and both additions round with the same operation.
      have hstart : embed (ExecFloat.add acc termI) = (embed acc) + termRounded :=
        rounded_ext (by rw [embed_val, haddR, hroundedStep, hterm])
      -- Unfold one fold step and apply the IH (whose start accumulator is `embed (add acc termI)`),
      -- then rewrite that start accumulator to `embed acc + termRounded`.
      simpa [List.foldl, hingeSumStepBinary, termI, termRounded, hstart] using ih

/--
Refine the whole executable hinge network to the rounded-real network.

The theorem composes the fold refinement with the final rounded bias addition.  Its hypotheses are
exactly the finiteness obligations needed by the IEEE-754 bridge lemmas.
-/
theorem toReal_hinge_fun_binary_eq_rounded_val (hformat : format.isIEEE = true) {n : ℕ}
    (t c : Fin n → Value) (b x : Value)
    (hx : isFinite x = true)
    (ht : ∀ i, isFinite (t i) = true)
    (hc : ∀ i, isFinite (c i) = true)
    (hSum :
      HingeSumFinite t c x (0 : Value) (List.finRange n))
    (hOut : isFinite (hingeFunBinary (t := t) (c := c) (b := b) x) = true) :
    (toModel (hingeFunBinary (t := t) (c := c) (b := b) x)).toReal =
      (RoundedReLUApprox.hingeFun format (embedVec t) (embedVec c) (embed b) (embed x)).val := by
  -- First refine the hinge sum.
  have hsum :
      (toModel (hingeSumBinary (c := c) (t := t) x)).toReal =
        (hingeSum format (embedVec c) (embedVec t) (embed x)).val := by
    rw [hinge_sum_rounded_eq_fold]
    have hstart : embed (0 : Value) = (0 : RoundedValue) := rounded_ext (by simp [embed])
    simpa only [hingeSumBinary, Fin.foldl_eq_foldl_finRange, hstart] using
      toReal_hinge_sum_binary_eq_rounded_val hformat t c x hSum hx ht hc
  -- Finally refine the last `+ b`.
  have haddR :
      (toModel (ExecFloat.add (hingeSumBinary (c := c) (t := t) x) b)).toReal =
        Model.roundAt format
          ((toModel (hingeSumBinary (c := c) (t := t) x)).toReal + (toModel b).toReal) :=
    toReal_add_eq_round_of_isFinite hformat
      (x := hingeSumBinary (c := c) (t := t) x) (y := b) (by simpa [hingeFunBinary] using hOut)
  -- `RoundedReLUApprox.hingeFun format` is `sum + b`, and RoundedValue `+` rounds with
  -- `Model.roundAt format`.
  have hfp :
      (RoundedReLUApprox.hingeFun format (embedVec t) (embedVec c) (embed b) (embed x)).val =
        Model.roundAt format
          ((hingeSum format (embedVec c) (embedVec t) (embed x)).val +
            (embed b).val) := by
    rw [RoundedReLUApprox.hingeFun, rounded_add_val]
  rw [hingeFunBinary, haddR, hfp, hsum, embed_val]

/-! ## Configured binary error bound inherited from the rounded-real bound -/

/--
Lift the rounded-real hinge-network error bound to executable evaluation.

The refinement `toReal_hinge_fun_binary_eq_rounded_val` identifies the executable result
with the rounded-real model, so its error bound transfers directly.
-/
theorem hinge_fun_binary_abs_error_le (hformat : format.isIEEE = true) {n : ℕ}
    (t c : Fin n → Value) (b x : Value)
    (hx : isFinite x = true)
    (ht : ∀ i, isFinite (t i) = true)
    (hc : ∀ i, isFinite (c i) = true)
    (hSum :
      HingeSumFinite t c x (0 : Value) (List.finRange n))
    (hOut : isFinite (hingeFunBinary (t := t) (c := c) (b := b) x) = true) :
    |(toModel (hingeFunBinary (t := t) (c := c) (b := b) x)).toReal -
        hingeFunReal format (embedVec t) (embedVec c) (embed b) (embed x)| ≤
      hingeFunErrorBound format (embedVec t) (embedVec c) (embed b) (embed x) := by
  rw [toReal_hinge_fun_binary_eq_rounded_val hformat t c b x hx ht hc hSum hOut]
  exact hinge_fun_abs_error format (t := embedVec t) (c := embedVec c) (b := embed b) (x := embed x)

/-! ## “Approximation error + rounding error” (pointwise) -/

/--
Pointwise triangle bound: target error to executable output is bounded by real approximation error
plus the certified IEEE arithmetic rounding error.
-/
theorem hinge_fun_total_abs_error_binary_le (hformat : format.isIEEE = true) {n : ℕ} (f : ℝ → ℝ)
    (t c : Fin n → Value) (b x : Value)
    (hx : isFinite x = true)
    (ht : ∀ i, isFinite (t i) = true)
    (hc : ∀ i, isFinite (c i) = true)
    (hSum :
      HingeSumFinite t c x (0 : Value) (List.finRange n))
    (hOut : isFinite (hingeFunBinary (t := t) (c := c) (b := b) x) = true) :
    |f ((toModel x).toReal) - (toModel (hingeFunBinary (t := t) (c := c) (b := b) x)).toReal| ≤
      |f ((toModel x).toReal) - hingeFunReal format (embedVec t) (embedVec c) (embed b) (embed x)| +
        hingeFunErrorBound format (embedVec t) (embedVec c) (embed b) (embed x) := by
  have hround :
      |hingeFunReal format (embedVec t) (embedVec c) (embed b) (embed x) -
          (toModel (hingeFunBinary (t := t) (c := c) (b := b) x)).toReal| ≤
        hingeFunErrorBound format (embedVec t) (embedVec c) (embed b) (embed x) := by
    rw [abs_sub_comm]
    exact hinge_fun_binary_abs_error_le hformat t c b x hx ht hc hSum hOut
  have htri :
      |f ((toModel x).toReal) - (toModel (hingeFunBinary (t := t) (c := c) (b := b) x)).toReal| ≤
        |f ((toModel x).toReal) -
          hingeFunReal format (embedVec t) (embedVec c) (embed b) (embed x)| +
          |hingeFunReal format (embedVec t) (embedVec c) (embed b) (embed x) -
              (toModel (hingeFunBinary (t := t) (c := c) (b := b) x)).toReal| :=
    abs_sub_le _ _ _
  exact le_trans htri (add_le_add_right hround _)

/-- Strict version of `hinge_fun_total_abs_error_binary_le`, useful for approximation theorems. -/
theorem hinge_fun_total_abs_error_binary_lt (hformat : format.isIEEE = true) {n : ℕ} (f : ℝ → ℝ)
    (t c : Fin n → Value) (b x : Value)
    (hx : isFinite x = true)
    (ht : ∀ i, isFinite (t i) = true)
    (hc : ∀ i, isFinite (c i) = true)
    (hSum :
      HingeSumFinite t c x (0 : Value) (List.finRange n))
    (hOut : isFinite (hingeFunBinary (t := t) (c := c) (b := b) x) = true)
    {ε : ℝ}
    (hε :
      |f ((toModel x).toReal) -
          hingeFunReal format (embedVec t) (embedVec c) (embed b) (embed x)| < ε) :
    |f ((toModel x).toReal) - (toModel (hingeFunBinary (t := t) (c := c) (b := b) x)).toReal| <
      ε + hingeFunErrorBound format (embedVec t) (embedVec c) (embed b) (embed x) :=
  lt_of_le_of_lt (hinge_fun_total_abs_error_binary_le hformat f t c b x hx ht hc hSum hOut)
    (add_lt_add_of_lt_of_le hε le_rfl)

/-! ## Configured binary pointwise ReLU approximation packaging (1D) -/

/--
1D ReLU approximation statement over configured binary values.

This is **not** a full universal approximation theorem, because it does not construct IEEE weights
from a real target. Instead it packages the already-proved pointwise inequality
`hinge_fun_total_abs_error_binary_lt` into an existence/for-all form:

- assume there exist configured binary hinge parameters `(t,c,b0)` that approximate `f` at the real
level
  (via `hingeFunReal format` on the embedded reals),
- and assume a finiteness/no-NaN/no-Inf witness for configured binary evaluation,
- then configured binary evaluation approximates `f` with an explicit extra rounding term
  `hingeFunErrorBound format`.
-/
theorem relu_approximation_Icc_binary_of_hinge (hformat : format.isIEEE = true)
    {f : ℝ → ℝ} {a b : ℝ} :
    ∀ ε > 0,
      (∃ (hidDim : ℕ) (t c : Fin hidDim → Value) (b0 : Value),
          (∀ i, isFinite (t i) = true) ∧
          (∀ i, isFinite (c i) = true) ∧
          (∀ x : Value,
              isFinite x = true →
              (toModel x).toReal ∈ Set.Icc a b →
                HingeSumFinite t c x (0 : Value) (List.finRange hidDim) ∧
                isFinite (hingeFunBinary (t := t) (c := c) (b := b0) x) = true) ∧
          (∀ x : Value,
              isFinite x = true →
              (toModel x).toReal ∈ Set.Icc a b →
                |f ((toModel x).toReal) -
                    hingeFunReal format (embedVec t) (embedVec c) (embed b0) (embed x)| < ε))
      →
      ∃ (hidDim : ℕ) (t c : Fin hidDim → Value) (b0 : Value),
        ∀ x : Value,
          isFinite x = true →
          (toModel x).toReal ∈ Set.Icc a b →
            |f ((toModel x).toReal) -
                (toModel (hingeFunBinary (t := t) (c := c) (b := b0) x)).toReal| <
              ε +
                hingeFunErrorBound format (embedVec t) (embedVec c) (embed b0) (embed x) := by
  rintro ε hε ⟨hidDim, t, c, b0, ht, hc, hFinite, hApprox⟩
  refine ⟨hidDim, t, c, b0, fun x hx hxIn => ?_⟩
  obtain ⟨hSum, hOut⟩ := hFinite x hx hxIn
  exact hinge_fun_total_abs_error_binary_lt hformat f t c b0 x hx ht hc hSum hOut
    (hApprox x hx hxIn)

/-! ## Real approximation + quantization + IEEE rounding (1D, pointwise) -/

/--
1D configured binary ReLU approximation with an explicit 3-term error decomposition, relative to a
caller-supplied real reference bias `bR`:

1. **Real approximation error**:
   $|f(r)-\operatorname{hinge\_fun}(\ldots,r)|<\varepsilon_{\mathrm{approx}}$.
2. **Quantization/reference error**:
   $|\operatorname{hinge\_fun}(\ldots,r)
   -\operatorname{hinge\_fun}_{\mathbb R}(\operatorname{embed}(\text{IEEE parameters}),
     \operatorname{embed}(r))|\leq\varepsilon_Q$.
3. **IEEE rounding error** (proved): `hingeFunErrorBound format`.

To obtain a fully synthesized configured binary approximation theorem, callers must additionally:
- construct configured binary parameters `(t,c,b0)` from the real hinge parameters, and
- prove the finiteness/no-NaN/no-Inf witnesses (`HingeSumFinite` + finite output), and
- prove a uniform bound `εQ` for the parameter-quantization step.
-/
theorem relu_approximation_Icc_binary_three_term_bias (hformat : format.isIEEE = true)
    {f : ℝ → ℝ} {a b : ℝ} {hidDim : ℕ}
    (bR : ℝ) (tR cR : Fin hidDim → ℝ)
    (t c : Fin hidDim → Value) (b0 : Value)
    (ht : ∀ i, isFinite (t i) = true)
    (hc : ∀ i, isFinite (c i) = true) :
    ∀ εApprox εQ : ℝ,
      (∀ x : Value,
          isFinite x = true →
          (toModel x).toReal ∈ Set.Icc a b →
            HingeSumFinite t c x (0 : Value) (List.finRange hidDim) ∧
            isFinite (hingeFunBinary (t := t) (c := c) (b := b0) x) = true) →
      (∀ x : Value,
          isFinite x = true →
          (toModel x).toReal ∈ Set.Icc a b →
            |f ((toModel x).toReal) - hingeFun hidDim tR cR bR ((toModel x).toReal)| < εApprox) →
      (∀ x : Value,
          isFinite x = true →
          (toModel x).toReal ∈ Set.Icc a b →
            |hingeFun hidDim tR cR bR ((toModel x).toReal) -
                hingeFunReal format (embedVec t) (embedVec c) (embed b0) (embed x)| ≤
                  εQ) →
      ∀ x : Value,
        isFinite x = true →
        (toModel x).toReal ∈ Set.Icc a b →
          |f ((toModel x).toReal) -
              (toModel (hingeFunBinary (t := t) (c := c) (b := b0) x)).toReal| <
            (εApprox + εQ) +
              hingeFunErrorBound format (embedVec t) (embedVec c) (embed b0) (embed x) := by
  intro εApprox εQ hFinite hApprox hQ x hx hxIn
  obtain ⟨hSum, hOut⟩ := hFinite x hx hxIn
  -- Triangle inequality through the real hinge network with the reference bias `bR`.
  have hApproxEmbed :
      |f ((toModel x).toReal) -
          hingeFunReal format (embedVec t) (embedVec c) (embed b0) (embed x)| <
        εApprox + εQ :=
    lt_of_le_of_lt (abs_sub_le _ (hingeFun hidDim tR cR bR ((toModel x).toReal)) _)
      (add_lt_add_of_lt_of_le (hApprox x hx hxIn) (hQ x hx hxIn))
  exact hinge_fun_total_abs_error_binary_lt hformat f t c b0 x hx ht hc hSum hOut hApproxEmbed

/--
Specialization of `relu_approximation_Icc_binary_three_term_bias` to the real bias `f a`, which is
the bias the constructive one-dimensional interpolation theorem emits.
-/
theorem relu_approximation_Icc_binary_three_term (hformat : format.isIEEE = true)
    {f : ℝ → ℝ} {a b : ℝ} {hidDim : ℕ}
    (tR cR : Fin hidDim → ℝ)
    (t c : Fin hidDim → Value) (b0 : Value)
    (ht : ∀ i, isFinite (t i) = true)
    (hc : ∀ i, isFinite (c i) = true) :
    ∀ εApprox εQ : ℝ,
      (∀ x : Value,
          isFinite x = true →
          (toModel x).toReal ∈ Set.Icc a b →
            HingeSumFinite t c x (0 : Value) (List.finRange hidDim) ∧
            isFinite (hingeFunBinary (t := t) (c := c) (b := b0) x) = true) →
      (∀ x : Value,
          isFinite x = true →
          (toModel x).toReal ∈ Set.Icc a b →
            |f ((toModel x).toReal) - hingeFun hidDim tR cR (f a) ((toModel x).toReal)| <
              εApprox) →
      (∀ x : Value,
          isFinite x = true →
          (toModel x).toReal ∈ Set.Icc a b →
            |hingeFun hidDim tR cR (f a) ((toModel x).toReal) -
                hingeFunReal format (embedVec t) (embedVec c) (embed b0) (embed x)| ≤
                  εQ) →
      ∀ x : Value,
        isFinite x = true →
        (toModel x).toReal ∈ Set.Icc a b →
          |f ((toModel x).toReal) -
              (toModel (hingeFunBinary (t := t) (c := c) (b := b0) x)).toReal| <
            (εApprox + εQ) +
              hingeFunErrorBound format (embedVec t) (embedVec c) (embed b0) (embed x) :=
  relu_approximation_Icc_binary_three_term_bias hformat (f a) tR cR t c b0 ht hc


/-! ## Dyadic quantization helpers -/

/--
Half-ulp error bound for dyadic values rounded into executable configured binary values.

The finiteness hypothesis excludes overflow/NaN paths, after which the executable rounding of a
dyadic agrees with the real `Model.roundAt format` operation used by the rounded-real model,
and FloatLib's local rounding bound `Model.abs_roundAt_sub_le` applies.
-/
theorem abs_toReal_roundDyadic_sub_le_half_ulp (hformat : format.isIEEE = true)
    (d : FloatLib.Numerics.Dyadic)
    (hfin :
      isFinite (ofModel (Model.roundDyadic format d) : Value) =
        true) :
    abs ((toModel (ofModel (Model.roundDyadic format d) :
          Value)).toReal - Dyadic.toReal d) ≤
      ulp binaryRadix (Model.fexpOf format) (Dyadic.toReal d) / 2 := by
  rw [toReal_roundDyadic_eq_round hformat hfin]
  exact Model.abs_roundAt_sub_le format _

/-!
### Real hinge network sensitivity to parameter rounding (for a compact input domain)

This is the “εQ quantization” step for `relu_approximation_Icc_binary_three_term`.
The lemma below uses the compact-domain assumption that the *reference* knot locations `tR`
lie in the input interval `[a,b]`, then bounds the output perturbation using:
- a width term `|b-a|` to bound `relu(x - tR i)` on the interval, and
- `relu_lipschitz` to control the sensitivity to perturbing `t`.
-/

/-- Nonnegativity of the real ReLU used in the compact-domain perturbation bound. -/
private theorem relu_nonneg (u : ℝ) : 0 ≤ relu u := by
  rw [relu, Activation.Math.reluSpec_eq_max]
  exact le_max_right u 0

/-- Since ReLU is nonnegative, its absolute value is itself. -/
private theorem abs_relu (u : ℝ) : abs (relu u) = relu u :=
  abs_of_nonneg (relu_nonneg u)

/-- ReLU is pointwise bounded by absolute value. -/
private theorem relu_le_abs (u : ℝ) : relu u ≤ abs u := by
  rw [relu, Activation.Math.reluSpec_eq_max]
  exact max_le (le_abs_self u) (abs_nonneg u)

/-- On `[a,b]`, a hinge activation `relu (x - t)` is bounded by the interval width. -/
private theorem abs_relu_sub_le_abs_width_Icc {a b x t : ℝ}
    (hx : x ∈ Set.Icc a b) (ht : t ∈ Set.Icc a b) :
    abs (relu (x - t)) ≤ abs (b - a) := by
  calc
    abs (relu (x - t)) = relu (x - t) := abs_relu (x - t)
    _ ≤ abs (x - t) := relu_le_abs (x - t)
    -- Two points of `[a,b]` are at most the interval width apart.
    _ ≤ abs (b - a) :=
        Set.abs_sub_le_of_uIcc_subset_uIcc
          (Set.Subset.trans (Set.uIcc_subset_Icc ht hx) Set.Icc_subset_uIcc)

/--
Sensitivity of a real hinge network to perturbing all parameters on a compact interval.

The bound decomposes into a bias perturbation, a coefficient perturbation weighted by the interval
width, and a knot perturbation weighted by the absolute executable coefficients.  This is the real
analysis step that supplies the quantization term `εQ`.
-/
theorem hinge_fun_abs_error_le_of_params_Icc
    {n : ℕ} {a b x : ℝ}
    (tR cR tI cI : Fin n → ℝ) (bR bI : ℝ)
    (hx : x ∈ Set.Icc a b)
    (htR : ∀ i, tR i ∈ Set.Icc a b) :
    abs (hingeFun n tR cR bR x - hingeFun n tI cI bI x) ≤
      abs (bR - bI) +
        ∑ i : Fin n,
          (abs (cR i - cI i) * abs (b - a) + abs (cI i) * abs (tR i - tI i)) := by
  classical
  -- Separate bias and sum: `hingeFun = b + sum`.
  have htri0 :
      abs (hingeFun n tR cR bR x - hingeFun n tI cI bI x) ≤
        abs (bR - bI) +
          abs ((∑ i : Fin n, cR i * relu (x - tR i)) - (∑ i : Fin n, cI i * relu (x - tI i))) := by
    simpa [hingeFun, sub_eq_add_neg, add_assoc, add_left_comm, add_comm] using
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
          (cR i - cI i) * relu (x - tR i) + cI i * (relu (x - tR i) - relu (x - tI i)) := by
      ring
    have htri :
        abs (cR i * relu (x - tR i) - cI i * relu (x - tI i)) ≤
          abs ((cR i - cI i) * relu (x - tR i)) +
            abs (cI i * (relu (x - tR i) - relu (x - tI i))) := by
      rw [hdecomp]
      exact abs_add_le _ _
    have hreluWidth : abs (relu (x - tR i)) ≤ abs (b - a) :=
      abs_relu_sub_le_abs_width_Icc (hx := hx) (ht := htR i)
    have hreluLip :
        abs (relu (x - tR i) - relu (x - tI i)) ≤ abs (tR i - tI i) := by
      have hdiff : (x - tR i) - (x - tI i) = tI i - tR i := by ring
      calc
        abs (relu (x - tR i) - relu (x - tI i))
            ≤ abs ((x - tR i) - (x - tI i)) := relu_lipschitz (x - tR i) (x - tI i)
        _ = abs (tR i - tI i) := by rw [hdiff, abs_sub_comm]
    have h1 :
        abs ((cR i - cI i) * relu (x - tR i)) ≤ abs (cR i - cI i) * abs (b - a) := by
      rw [abs_mul]
      exact mul_le_mul_of_nonneg_left hreluWidth (abs_nonneg _)
    have h2 :
        abs (cI i * (relu (x - tR i) - relu (x - tI i))) ≤ abs (cI i) * abs (tR i - tI i) := by
      rw [abs_mul]
      exact mul_le_mul_of_nonneg_left hreluLip (abs_nonneg _)
    linarith
  -- Substitute the improved sum bound into the bias+sum triangle bound.
  exact le_trans htri0 (add_le_add le_rfl (le_trans hsum0 (Finset.sum_le_sum fun i _ => hterm i)))

/--
Uniform version of `hinge_fun_abs_error_le_of_params_Icc`.

Instead of summing per-neuron perturbation bounds, this theorem uses uniform coefficient and knot
budgets `Δc`, `C`, and `Δt`, producing the simpler expression
`|bR-bI| + n * (Δc * |b-a| + C * Δt)`.
-/
theorem hinge_fun_abs_error_le_of_params_Icc_uniform
    {n : ℕ} {a b x : ℝ}
    (tR cR tI cI : Fin n → ℝ) (bR bI : ℝ)
    (hx : x ∈ Set.Icc a b)
    (htR : ∀ i, tR i ∈ Set.Icc a b)
    {Δc C Δt : ℝ}
    (hC0 : 0 ≤ C)
    (hΔc : ∀ i, abs (cR i - cI i) ≤ Δc)
    (hC : ∀ i, abs (cI i) ≤ C)
    (hΔt : ∀ i, abs (tR i - tI i) ≤ Δt) :
    abs (hingeFun n tR cR bR x - hingeFun n tI cI bI x) ≤
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
    abs (hingeFun n tR cR bR x - hingeFun n tI cI bI x)
        ≤ abs (bR - bI) +
            ∑ i : Fin n,
              (abs (cR i - cI i) * abs (b - a) + abs (cI i) * abs (tR i - tI i)) :=
          hinge_fun_abs_error_le_of_params_Icc tR cR tI cI bR bI hx htR
    _ ≤ abs (bR - bI) + ∑ _i : Fin n, (Δc * abs (b - a) + C * Δt) :=
          add_le_add le_rfl (Finset.sum_le_sum fun i _ => hterm i)
    _ = abs (bR - bI) + (n : ℝ) * (Δc * abs (b - a) + C * Δt) := by
          rw [hsum']

/-! ## Removing the quantization term when reals are exactly representable -/

/--
The embedded real reference is exactly the ordinary real hinge network evaluated on
entrywise `ExecFloat.Binary.toModel` followed by `Model.toReal` parameters.
-/
theorem hinge_fun_real_embed_eq_hinge_fun_toReal {n : ℕ}
    (t c : Fin n → Value) (b x : Value) :
    hingeFunReal format (embedVec t) (embedVec c) (embed b) (embed x) =
      hingeFun n
        (fun i => (toModel (t i)).toReal)
        (fun i => (toModel (c i)).toReal)
        ((toModel b).toReal)
        ((toModel x).toReal) := by
  classical
  -- `hingeSumReal format` is a sum of `hingeTermReal format`; embeddings expose `toReal` values.
  have hsum :
      hingeSumReal format (c := embedVec c) (t := embedVec t) (x := embed x) =
        ∑ i : Fin n,
          ((toModel (c i)).toReal) * relu ((toModel x).toReal - (toModel (t i)).toReal) := by
    simpa [hingeTermReal, embedVec, embed] using
      (hinge_sum_real_eq_sum format (c := embedVec c) (t := embedVec t) (x := embed x))
  -- Finish (commute the final `+ b`).
  rw [hingeFunReal, hingeFun, hsum, embed_val, add_comm]

/--
Uniform quantization-error bound between a real hinge network and executable IEEE parameters.

This is the user-facing `εQ` discharge lemma when callers can bound coefficient and knot rounding
errors uniformly.
-/
theorem hinge_fun_quantization_error_le_Icc_uniform
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
    abs (hingeFun n tR cR bR ((toModel x).toReal) -
          hingeFunReal format (embedVec t) (embedVec c) (embed b0) (embed x)) ≤
      abs (bR - (toModel b0).toReal) + (n : ℝ) * (Δc * abs (b - a) + C * Δt) := by
  -- Rewrite the embedded-real reference into an explicit `hingeFun` on `toReal` parameters, then
  -- apply the uniform parameter-perturbation bound on `[a,b]`.
  rw [hinge_fun_real_embed_eq_hinge_fun_toReal]
  exact hinge_fun_abs_error_le_of_params_Icc_uniform (tR := tR) (cR := cR)
    (tI := fun i => (toModel (t i)).toReal) (cI := fun i => (toModel (c i)).toReal)
    (bR := bR) (bI := (toModel b0).toReal) hxIn htR hC0 hΔc hC hΔt

/-!
### Dyadic-to-configured binary quantization bound (`εQ`)

This lemma is a drop-in way to discharge the `εQ` premise of
`relu_approximation_Icc_binary_three_term` when IEEE parameters are obtained by rounding dyadic
rationals with `Model.roundDyadic format`.
-/

/--
Dyadic quantization error for a hinge network on `[a,b]`.

The real reference uses exact dyadic parameters, while the executable reference uses those dyadics
rounded to configured binary values.  The result is still expressed as a sum of concrete
per-parameter
rounding errors.
-/
theorem hinge_fun_dyadic_quantization_error_le_Icc
    {n : ℕ} {a b : ℝ}
    (tD cD : Fin n → FloatLib.Numerics.Dyadic) (bD : FloatLib.Numerics.Dyadic)
    (x : Value)
    (hxIn : (toModel x).toReal ∈ Set.Icc a b)
    (htIn : ∀ i, Dyadic.toReal (tD i) ∈ Set.Icc a b) :
    abs
        (hingeFun n
            (fun i => Dyadic.toReal (tD i))
            (fun i => Dyadic.toReal (cD i))
            (Dyadic.toReal bD)
            ((toModel x).toReal) -
          hingeFunReal format
            (t := embedVec (fun i =>
              (ofModel (Model.roundDyadic format (tD i)) : Value)))
            (c := embedVec (fun i =>
              (ofModel (Model.roundDyadic format (cD i)) : Value)))
            (b := embed (ofModel (Model.roundDyadic format bD) :
              Value))
            (x := embed x)) ≤
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
  -- Rewrite the `hingeFunReal format` reference into a `hingeFun` over the embedded reals, then
  -- apply the
  -- general “parameter sensitivity on `[a,b]`” bound.
  rw [hinge_fun_real_embed_eq_hinge_fun_toReal]
  exact hinge_fun_abs_error_le_of_params_Icc
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
Dyadic quantization error with each per-parameter rounding error bounded by a half-ulp term.

This is the more automated version of `hinge_fun_dyadic_quantization_error_le_Icc`: callers provide
finiteness of every rounded dyadic value, and the theorem substitutes the standard half-ulp bounds.
-/
theorem hinge_fun_dyadic_quantization_error_le_Icc_half_ulp (hformat : format.isIEEE = true)
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
        (hingeFun n
            (fun i => Dyadic.toReal (tD i))
            (fun i => Dyadic.toReal (cD i))
            (Dyadic.toReal bD)
            ((toModel x).toReal) -
          hingeFunReal format
            (t := embedVec (fun i =>
              (ofModel (Model.roundDyadic format (tD i)) : Value)))
            (c := embedVec (fun i =>
              (ofModel (Model.roundDyadic format (cD i)) : Value)))
            (b := embed (ofModel (Model.roundDyadic format bD) :
              Value))
            (x := embed x)) ≤
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
    hinge_fun_dyadic_quantization_error_le_Icc (tD := tD) (cD := cD) (bD := bD) (x := x) hxIn htIn
  -- Bound each parameter-difference term by a half-ulp using
  -- `abs_toReal_roundDyadic_sub_le_half_ulp`.
  have hb :
      abs (Dyadic.toReal bD -
          (toModel (ofModel (Model.roundDyadic format bD) :
            Value)).toReal) ≤
        ulp binaryRadix (Model.fexpOf format) (Dyadic.toReal bD) / 2 := by
    rw [abs_sub_comm]
    exact abs_toReal_roundDyadic_sub_le_half_ulp hformat bD hbfin
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
      exact abs_toReal_roundDyadic_sub_le_half_ulp hformat (cD i) (hcfin i)
    have ht :
        abs (Dyadic.toReal (tD i) -
            (toModel (ofModel (Model.roundDyadic format (tD i)) :
              Value)).toReal) ≤
          ulp binaryRadix (Model.fexpOf format) (Dyadic.toReal (tD i)) / 2
            := by
      rw [abs_sub_comm]
      exact abs_toReal_roundDyadic_sub_le_half_ulp hformat (tD i) (htfin i)
    exact add_le_add (mul_le_mul_of_nonneg_right hc (abs_nonneg (b - a)))
      (mul_le_mul_of_nonneg_left ht (abs_nonneg _))
  -- Combine the termwise bounds under the sum, then add the bias bound.
  exact le_trans hraw (add_le_add hb (Finset.sum_le_sum fun i _ => hterm i))

/-!
### Packaged “dyadic quantization + IEEE rounding” theorem (1D, pointwise)

This is the “dyadic next step”: it replaces the abstract `εQ` premise of
`relu_approximation_Icc_binary_three_term_bias` with a concrete bound coming from:
- dyadic→arithmetic rounding (`≤ 1/2 ulp`), and
- a Lipschitz-style sensitivity bound for the real hinge network on `x ∈ [a,b]`.

It still assumes the IEEE finiteness witnesses for evaluation (`HingeSumFinite` + finite output).
-/

/--
Three-term configured binary approximation bound with the quantization term instantiated by the
half-ulp
dyadic rounding budget of `hinge_fun_dyadic_quantization_error_le_Icc_half_ulp`.
-/
theorem relu_approximation_Icc_binary_dyadic_half_ulp (hformat : format.isIEEE = true)
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
                (hingeFunBinary
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
                hingeFun hidDim
                  (fun i => Dyadic.toReal (tD i))
                  (fun i => Dyadic.toReal (cD i))
                  (Dyadic.toReal bD)
                  ((toModel x).toReal)| < εApprox) →
      ∀ x : Value,
        isFinite x = true →
        (toModel x).toReal ∈ Set.Icc a b →
          |f ((toModel x).toReal) -
              (toModel (hingeFunBinary
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
                (t := embedVec (fun i =>
                  (ofModel (Model.roundDyadic format (tD i)) :
                    Value)))
                (c := embedVec (fun i =>
                  (ofModel (Model.roundDyadic format (cD i)) :
                    Value)))
                (b := embed (ofModel (Model.roundDyadic format bD) :
                  Value))
                (x := embed x) := by
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
    (relu_approximation_Icc_binary_three_term_bias hformat
        (f := f) (a := a) (b := b) (hidDim := hidDim)
        (bR := Dyadic.toReal bD)
        (tR := fun i => Dyadic.toReal (tD i))
        (cR := fun i => Dyadic.toReal (cD i))
        (t := fun i =>
          (ofModel (Model.roundDyadic format (tD i)) : Value))
        (c := fun i =>
          (ofModel (Model.roundDyadic format (cD i)) : Value))
        (b0 := (ofModel (Model.roundDyadic format bD) : Value))
        (ht := htfin) (hc := hcfin))
      εApprox εQdy
      hFinite
      hApprox
      (fun x _ hxIn =>
        hinge_fun_dyadic_quantization_error_le_Icc_half_ulp hformat tD cD bD x
          hxIn htIn htfin hcfin hbfin)
      x hx hxIn
  -- The conclusion matches, up to unfolding `εQdy` and reassociation.
  simpa [εQdy, add_assoc] using this

/--
Two-term 1D configured binary ReLU approximation statement:

- assume the real hinge network built from the IEEE parameters’ `toReal` values already
  approximates `f` on `toReal` inputs in `[a,b]`,
- and assume finiteness/no-NaN/no-Inf witnesses,
- then IEEE execution approximates `f` with an explicit IEEE rounding term.

This is a specialization of the 3-term theorem with `εQ = 0`.
-/
theorem relu_approximation_Icc_binary_two_term (hformat : format.isIEEE = true)
    {f : ℝ → ℝ} {a b : ℝ} {hidDim : ℕ}
    (t c : Fin hidDim → Value) (b0 : Value)
    (ht : ∀ i, isFinite (t i) = true)
    (hc : ∀ i, isFinite (c i) = true) :
    ∀ εApprox : ℝ,
      (∀ x : Value,
          isFinite x = true →
          (toModel x).toReal ∈ Set.Icc a b →
            HingeSumFinite t c x (0 : Value) (List.finRange hidDim) ∧
            isFinite (hingeFunBinary (t := t) (c := c) (b := b0) x) = true) →
      (∀ x : Value,
          isFinite x = true →
          (toModel x).toReal ∈ Set.Icc a b →
            |f ((toModel x).toReal) -
                hingeFun hidDim
                  (fun i => (toModel (t i)).toReal)
                  (fun i => (toModel (c i)).toReal)
                  ((toModel b0).toReal)
                  ((toModel x).toReal)| < εApprox) →
      ∀ x : Value,
        isFinite x = true →
        (toModel x).toReal ∈ Set.Icc a b →
          |f ((toModel x).toReal) -
              (toModel (hingeFunBinary (t := t) (c := c) (b := b0) x)).toReal| <
            εApprox +
              hingeFunErrorBound format (embedVec t) (embedVec c) (embed b0) (embed x) := by
  intro εApprox hFinite hApprox x hx hxIn
  obtain ⟨hSum, hOut⟩ := hFinite x hx hxIn
  -- Replace `hingeFunReal format` by the explicit real hinge function on `toReal`
  -- parameters/inputs.
  have hApproxEmbed :
      |f ((toModel x).toReal) -
          hingeFunReal format (embedVec t) (embedVec c) (embed b0) (embed x)| <
        εApprox := by
    rw [hinge_fun_real_embed_eq_hinge_fun_toReal]
    exact hApprox x hx hxIn
  exact hinge_fun_total_abs_error_binary_lt hformat f t c b0 x hx ht hc hSum hOut hApproxEmbed

end

end BinaryExecReLUApprox

end NN.MLTheory.Proofs.UniversalApproximation
