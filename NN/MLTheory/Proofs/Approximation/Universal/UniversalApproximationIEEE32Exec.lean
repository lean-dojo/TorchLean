/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.MLTheory.Proofs.Approximation.Universal.IEEE32ExecCore
public import NN.MLTheory.Proofs.Approximation.Universal.UniversalApproximationFP32

/-!
# IEEE32 executable ReLU approximation

IEEE32Exec-facing theorems for combining:

- real-valued hinge-network approximation results, and
- a proved rounding/error bound for executing that hinge network with IEEE binary32 rounding.

The file is organized as a refinement stack:

- `hingeFunIeee` defines the executable network using IEEE-754 binary32 operations;
- `HingeSumFinite` records the exact finiteness obligations needed to rule out NaN/Inf paths;
- bridge lemmas connect the executable values, via `IEEE32Exec.toReal`, to the rounded-`ℝ`
  `FP32` semantics; and
- the final theorem packages real approximation, FP32 rounding, and IEEE execution into one
  certified error bound.

The construction exposes the finite-execution hypotheses at theorem boundaries: they are the
precise trust boundary between mathematical approximation and concrete floating-point execution.
The floating-point viewpoint follows IEEE Std 754-2019 and the standard analyses of
Goldberg and Higham; the approximation side reuses the constructive hinge-network development in
`UniversalApproximation` and `UniversalApproximationFP32`.
-/

@[expose] public section


namespace NN.MLTheory.Proofs.UniversalApproximation

open TorchLean.Floats
open TorchLean.Floats.IEEE754

namespace IEEE32ExecReLUApprox

open IEEE32Exec

noncomputable section

/-- FP32 model type (rounded `ℝ`) used as the comparison target for IEEE32Exec error bounds. -/
abbrev FP32 : Type := TorchLean.Floats.FP32

-- We avoid simp-unfolding `IEEE32Exec.toReal`; this file uses the dedicated bridge lemmas instead.

/-! ## Embedding IEEE32Exec values into the FP32 (rounded-ℝ) model -/

@[inline] def embed (x : IEEE32Exec) : FP32 := ⟨IEEE32Exec.toReal x⟩

@[simp] theorem embed_val (x : IEEE32Exec) : (embed x).val = IEEE32Exec.toReal x := rfl

/-- Pointwise embedding of a vector of `IEEE32Exec` values into the FP32 model. -/
@[inline] def embedVec {n : ℕ} (v : Fin n → IEEE32Exec) : Fin n → FP32 := fun i => embed (v i)

/-! ## IEEE32Exec hinge network (same shape as `hinge_fun_fp32`) -/

@[inline] def reluIeee (x : IEEE32Exec) : IEEE32Exec :=
  IEEE32Exec.maximum x (Numbers.zero : IEEE32Exec)

/-- One hinge term `cᵢ * ReLU(x - tᵢ)` evaluated in the executable IEEE32Exec backend. -/
@[inline] def hingeTermIeee {n : ℕ} (c t : Fin n → IEEE32Exec) (x : IEEE32Exec) (i : Fin n) :
  IEEE32Exec :=
  IEEE32Exec.mul (c i) (reluIeee (IEEE32Exec.sub x (t i)))

/-- Fold step for summing hinge terms, used by `hinge_sum_ieee`. -/
@[inline] def hingeSumStepIeee {n : ℕ} (c t : Fin n → IEEE32Exec) (x : IEEE32Exec) :
    IEEE32Exec → Fin n → IEEE32Exec :=
  fun acc i => IEEE32Exec.add acc (hingeTermIeee c t x i)

/-- Executable IEEE32Exec sum of hinge terms, in the fixed `List.finRange` order. -/
noncomputable def hingeSumIeee {n : ℕ} (c t : Fin n → IEEE32Exec) (x : IEEE32Exec) : IEEE32Exec :=
  (List.finRange n).foldl (hingeSumStepIeee c t x) (Numbers.zero : IEEE32Exec)

/-- Executable IEEE32Exec hinge network: sum hinge terms, then add the executable bias. -/
noncomputable def hingeFunIeee {n : ℕ} (t c : Fin n → IEEE32Exec) (b x : IEEE32Exec) : IEEE32Exec
  :=
  IEEE32Exec.add (hingeSumIeee (c := c) (t := t) x) b

/-! ## A finiteness witness for IEEE evaluation (no NaN/Inf intermediates) -/

inductive HingeSumFinite {n : ℕ} (t c : Fin n → IEEE32Exec) (x : IEEE32Exec) :
    IEEE32Exec → List (Fin n) → Prop where
  | nil {acc : IEEE32Exec} (hacc : IEEE32Exec.isFinite acc = true) :
      HingeSumFinite t c x acc []
  | cons {acc : IEEE32Exec} {i : Fin n} {xs : List (Fin n)}
      (hsub : IEEE32Exec.isFinite (IEEE32Exec.sub x (t i)) = true)
      (hmax : IEEE32Exec.isFinite (IEEE32Exec.maximum (IEEE32Exec.sub x (t i)) (Numbers.zero :
        IEEE32Exec)) = true)
      (hmul : IEEE32Exec.isFinite (hingeTermIeee (c := c) (t := t) x i) = true)
      (hadd : IEEE32Exec.isFinite (IEEE32Exec.add acc (hingeTermIeee (c := c) (t := t) x i)) =
        true)
      (hrest :
        HingeSumFinite t c x (IEEE32Exec.add acc (hingeTermIeee (c := c) (t := t) x i)) xs) :
      HingeSumFinite t c x acc (i :: xs)

/-!
### Discharging `HingeSumFinite` for *concrete* networks by computation

For typical verification workflows, `t`, `c`, and `x` are *concrete* IEEE32Exec constants
coming from a compiled model artifact. In that setting, the easiest way to satisfy the
`HingeSumFinite …` hypotheses is to compute the IEEE32Exec kernel and check finiteness at
every intermediate.

`instDecHingeSumFinite` provides a `Decidable` instance generator for `HingeSumFinite …`.
This enables proofs like:

```lean
  classical
  haveI := IEEE32ExecReLUApprox.instDecHingeSumFinite (t := t) (c := c) (x := x)
    (acc := (Numbers.zero : IEEE32Exec)) (xs := List.finRange n)
  have hSum : HingeSumFinite t c x (Numbers.zero : IEEE32Exec) (List.finRange n) := by
    decide
```

This does *not* solve the symbolic “no overflow for all x” problem, but it makes the pointwise
theorems in this file directly usable for concrete executions.
-/

def instDecHingeSumFinite {n : ℕ} (t c : Fin n → IEEE32Exec) (x : IEEE32Exec) :
    ∀ (acc : IEEE32Exec) (xs : List (Fin n)),
      Decidable (HingeSumFinite t c x acc xs)
    | acc, [] =>
        if hacc : IEEE32Exec.isFinite acc = true then
          isTrue (HingeSumFinite.nil (t := t) (c := c) (x := x) (acc := acc) hacc)
        else
          isFalse (by
            intro h
            cases h with
            | nil hacc' =>
                exact hacc hacc')
    | acc, i :: xs =>
        let sub := IEEE32Exec.sub x (t i)
        let mx := IEEE32Exec.maximum sub (Numbers.zero : IEEE32Exec)
        let term := hingeTermIeee (c := c) (t := t) x i
        let acc' := IEEE32Exec.add acc term
      if hsub : IEEE32Exec.isFinite sub = true then
        if hmax : IEEE32Exec.isFinite mx = true then
          if hmul : IEEE32Exec.isFinite term = true then
            if hadd : IEEE32Exec.isFinite acc' = true then
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

def HingeEvalFiniteProp {n : ℕ} (t c : Fin n → IEEE32Exec) (b x : IEEE32Exec) : Prop :=
  IEEE32Exec.isFinite x = true ∧
    (∀ i, IEEE32Exec.isFinite (t i) = true) ∧
    (∀ i, IEEE32Exec.isFinite (c i) = true) ∧
    HingeSumFinite t c x (Numbers.zero : IEEE32Exec) (List.finRange n) ∧
    IEEE32Exec.isFinite (hingeFunIeee (t := t) (c := c) (b := b) x) = true

/--
Unpack the compact pointwise finiteness bundle.

The executable bridge lemmas need separate finiteness facts for the input, parameters, fold state,
and final output. Keeping the bundled form at theorem boundaries makes user-facing statements
shorter, while this projection feeds the stepwise IEEE-754 refinement proofs.
-/
theorem hingeEvalFiniteProp_to_witness {n : ℕ} {t c : Fin n → IEEE32Exec} {b x : IEEE32Exec} :
    HingeEvalFiniteProp t c b x →
      (IEEE32Exec.isFinite x = true) ∧
      (∀ i, IEEE32Exec.isFinite (t i) = true) ∧
      (∀ i, IEEE32Exec.isFinite (c i) = true) ∧
      HingeSumFinite t c x (Numbers.zero : IEEE32Exec) (List.finRange n) ∧
      IEEE32Exec.isFinite (hingeFunIeee (t := t) (c := c) (b := b) x) = true := by
  intro h; simpa [HingeEvalFiniteProp] using h

/-! ## Refinement: IEEE32Exec execution equals FP32 rounded-ℝ execution (as reals) -/

/-- FP32 addition in the rounded-`ℝ` model uses the same `fp32Round` operation as IEEE32Exec. -/
private theorem fp32_add_val (a b : FP32) :
    (a + b).val = TorchLean.Floats.IEEE754.IEEE32Exec.fp32Round (a.val + b.val) := by
  change TorchLean.Floats.round32 (a.val + b.val) =
    TorchLean.Floats.IEEE754.IEEE32Exec.fp32Round (a.val + b.val)
  rfl

/-- FP32 subtraction in the rounded-`ℝ` model uses the same `fp32Round` operation as IEEE32Exec. -/
private theorem fp32_sub_val (a b : FP32) :
    (a - b).val = TorchLean.Floats.IEEE754.IEEE32Exec.fp32Round (a.val - b.val) := by
  change TorchLean.Floats.round32 (a.val - b.val) =
    TorchLean.Floats.IEEE754.IEEE32Exec.fp32Round (a.val - b.val)
  rfl

/-- FP32 multiplication in the rounded-`ℝ` model uses the same `fp32Round` operation as IEEE32Exec. -/
private theorem fp32_mul_val (a b : FP32) :
    (a * b).val = TorchLean.Floats.IEEE754.IEEE32Exec.fp32Round (a.val * b.val) := by
  change TorchLean.Floats.round32 (a.val * b.val) =
    TorchLean.Floats.IEEE754.IEEE32Exec.fp32Round (a.val * b.val)
  rfl

/-- ReLU is exact at the FP32 value level because it is a comparison with zero, not arithmetic. -/
private theorem fp32_relu_val' (x : FP32) : (reluFp32 x).val = relu x.val := by
  simp [relu_fp32_val]

/-- Extensionality for the rounded-`ℝ` FP32 wrapper. -/
private theorem fp32_ext {u v : FP32} (h : u.val = v.val) : u = v := by
  cases u with
  | mk uval =>
      cases v with
      | mk vval =>
          cases h
          rfl

/--
Finite IEEE32Exec ReLU agrees with real ReLU after `toReal`.

The IEEE operation here is `maximum x 0`; once NaN/Inf paths are ruled out, the bridge theorem for
`maximum` turns it into real `max`, which is exactly TorchLean's real ReLU specification.
-/
private theorem toReal_relu_ieee_eq_relu {x : IEEE32Exec}
    (hx : IEEE32Exec.isFinite x = true) :
    IEEE32Exec.toReal (reluIeee x) = relu (IEEE32Exec.toReal x) := by
  have h0 : IEEE32Exec.isFinite (Numbers.zero : IEEE32Exec) = true := by decide
  have hto0 : IEEE32Exec.toReal (Numbers.zero : IEEE32Exec) = 0 := by
    -- `Numbers.zero` is `posZero`.
    simp [Numbers.zero, TorchLean.Floats.IEEE754.IEEE32Exec.toReal_posZero]
  have hmax :
      IEEE32Exec.toReal (IEEE32Exec.maximum x (Numbers.zero : IEEE32Exec)) =
        max (IEEE32Exec.toReal x) (IEEE32Exec.toReal (Numbers.zero : IEEE32Exec)) :=
    TorchLean.Floats.IEEE754.IEEE32Exec.toReal_maximum_eq_max_of_isFinite (x := x)
      (y := (Numbers.zero : IEEE32Exec)) hx h0
  -- `relu` is `max _ 0` on reals.
  simpa [reluIeee, relu, Activation.Math.reluSpec, hto0] using hmax

/--
Refine one executable hinge term to the corresponding FP32 rounded-`ℝ` hinge term.

This is the local IEEE-754 step in the proof: subtraction rounds, ReLU is exact under finiteness,
and multiplication rounds.  The statement is deliberately per-neuron so the fold refinement below
can reuse it uniformly across `List.finRange n`.
-/
private theorem toReal_hinge_term_eq_fp32_val {n : ℕ}
    (t c : Fin n → IEEE32Exec) (x : IEEE32Exec) (i : Fin n)
    (hsub : IEEE32Exec.isFinite (IEEE32Exec.sub x (t i)) = true)
    (hmul : IEEE32Exec.isFinite (hingeTermIeee (c := c) (t := t) x i) = true)
    (hx : IEEE32Exec.isFinite x = true)
    (ht : IEEE32Exec.isFinite (t i) = true) :
    IEEE32Exec.toReal (hingeTermIeee (c := c) (t := t) x i) =
      (hingeTermFp32 (c := embedVec c) (t := embedVec t) (x := embed x) i).val := by
  -- Subtraction refinement.
  have hsubR :
      IEEE32Exec.toReal (IEEE32Exec.sub x (t i)) =
        TorchLean.Floats.IEEE754.IEEE32Exec.fp32Round (IEEE32Exec.toReal x - IEEE32Exec.toReal (t
          i)) :=
    toReal_sub_eq_fp32Round_of_isFinite (x := x) (y := t i) hx ht (by simpa using hsub)
  -- ReLU is `maximum _ 0`, which is exact as `max` on reals for finite inputs.
  have h0 : IEEE32Exec.isFinite (Numbers.zero : IEEE32Exec) = true := by decide
  have hsubFinite : IEEE32Exec.isFinite (IEEE32Exec.sub x (t i)) = true := hsub
  have hreluR :
      IEEE32Exec.toReal (reluIeee (IEEE32Exec.sub x (t i))) =
        relu (IEEE32Exec.toReal (IEEE32Exec.sub x (t i))) := by
    exact toReal_relu_ieee_eq_relu (x := IEEE32Exec.sub x (t i)) hsubFinite
  -- Multiplication refinement.
  have hmulR :
      IEEE32Exec.toReal (IEEE32Exec.mul (c i) (reluIeee (IEEE32Exec.sub x (t i)))) =
        TorchLean.Floats.IEEE754.IEEE32Exec.fp32Round
          (IEEE32Exec.toReal (c i) * IEEE32Exec.toReal (reluIeee (IEEE32Exec.sub x (t i)))) :=
    TorchLean.Floats.IEEE754.IEEE32Exec.toReal_mul_eq_fp32Round_of_isFinite (x := c i)
      (y := reluIeee (IEEE32Exec.sub x (t i))) (by simpa [hingeTermIeee] using hmul)
  -- Compute the FP32 hinge term as the same `fp32Round` expression.
  have htermFP :
      (hingeTermFp32 (c := embedVec c) (t := embedVec t) (x := embed x) i).val =
        TorchLean.Floats.IEEE754.IEEE32Exec.fp32Round
          (IEEE32Exec.toReal (c i) *
            relu
              (TorchLean.Floats.IEEE754.IEEE32Exec.fp32Round (IEEE32Exec.toReal x -
                IEEE32Exec.toReal (t i)))) := by
    -- `reluFp32` is exact on `.val`, and FP32 `-`/`*` round with the same `fp32Round`.
    simp [hingeTermFp32, embedVec, embed, fp32_mul_val, fp32_sub_val, relu,
      Activation.Math.reluSpec]
  -- Rewrite IEEE32Exec `mul` and `reluIeee` using the bridge lemmas, then match `htermFP`.
  calc
    IEEE32Exec.toReal (hingeTermIeee (c := c) (t := t) x i)
        = TorchLean.Floats.IEEE754.IEEE32Exec.fp32Round
            (IEEE32Exec.toReal (c i) * IEEE32Exec.toReal (reluIeee (IEEE32Exec.sub x (t i)))) := by
              simpa [hingeTermIeee] using hmulR
    _ = TorchLean.Floats.IEEE754.IEEE32Exec.fp32Round
            (IEEE32Exec.toReal (c i) * relu (IEEE32Exec.toReal (IEEE32Exec.sub x (t i)))) := by
              simp [hreluR]
    _ = TorchLean.Floats.IEEE754.IEEE32Exec.fp32Round
            (IEEE32Exec.toReal (c i) *
              relu
                (TorchLean.Floats.IEEE754.IEEE32Exec.fp32Round (IEEE32Exec.toReal x -
                  IEEE32Exec.toReal (t i)))) := by
              simp [hsubR]
    _ = (hingeTermFp32 (c := embedVec c) (t := embedVec t) (x := embed x) i).val := by
              simp [htermFP]

/-- The FP32 hinge-sum definition is the plain fold over FP32 hinge terms. -/
private theorem hinge_sum_fp32_eq_fold {n : ℕ} (c t : Fin n → FP32) (x : FP32) :
    hingeSumFp32 c t x =
      (List.finRange n).foldl (fun acc i => acc + hingeTermFp32 c t x i) (0 : FP32) := by
  -- Peel the irrelevant components of the hinge-sum state by induction over the list.
  have :
      ∀ (xs : List (Fin n)) (acc32 : FP32) (accR err : ℝ),
        (xs.foldl (hingeSumStateStep c t x) (acc32, accR, err)).1 =
          xs.foldl (fun acc i => acc + hingeTermFp32 c t x i) acc32 := by
    intro xs
    induction xs with
    | nil =>
        intro acc32 accR err
        simp [List.foldl]
    | cons i xs ih =>
        intro acc32 accR err
        simp [List.foldl, hingeSumStateStep, ih]
  simpa [hingeSumFp32, hingeSumState] using
    (this (xs := List.finRange n) (acc32 := (0 : FP32)) (accR := (0 : ℝ)) (err := (0 : ℝ)))

/--
Refine an executable IEEE hinge-term fold to the FP32 rounded-`ℝ` fold with the same order.

Floating-point addition is not associative, so the theorem preserves the exact `List.finRange`
evaluation order.  This is why the proof is fold-based rather than rewriting directly to an
unordered finite sum.
-/
theorem toReal_hinge_sum_ieee_eq_fp32_val {n : ℕ}
    (t c : Fin n → IEEE32Exec) (x : IEEE32Exec)
    {acc : IEEE32Exec} {xs : List (Fin n)}
    (hfin : HingeSumFinite t c x acc xs)
    (hx : IEEE32Exec.isFinite x = true)
    (ht : ∀ i, IEEE32Exec.isFinite (t i) = true)
    (_hc : ∀ i, IEEE32Exec.isFinite (c i) = true) :
    IEEE32Exec.toReal (xs.foldl (hingeSumStepIeee (c := c) (t := t) x) acc) =
      (xs.foldl (fun acc32 i => acc32 + hingeTermFp32 (c := embedVec c) (t := embedVec t) (x :=
        embed x) i)
        (embed acc)).val := by
  induction hfin with
  | nil hacc =>
      simp [List.foldl]
  | cons hsub hmax hmul hadd hrest ih =>
      rename_i acc i xs
      -- Abbreviations for this step.
      let termI : IEEE32Exec := hingeTermIeee (c := c) (t := t) x i
      let termFP : FP32 := hingeTermFp32 (c := embedVec c) (t := embedVec t) (x := embed x) i
      have hterm : IEEE32Exec.toReal termI = termFP.val := by
        simpa [termI, termFP] using
          (toReal_hinge_term_eq_fp32_val (t := t) (c := c) (x := x) (i := i) hsub hmul hx (ht i))
      have haddR :
          IEEE32Exec.toReal (IEEE32Exec.add acc termI) =
            TorchLean.Floats.IEEE754.IEEE32Exec.fp32Round (IEEE32Exec.toReal acc + IEEE32Exec.toReal
              termI) :=
        TorchLean.Floats.IEEE754.IEEE32Exec.toReal_add_eq_fp32Round_of_isFinite (x := acc) (y :=
          termI)
          (by simpa [termI] using hadd)
      have hfpStep :
          ((embed acc) + termFP).val =
            TorchLean.Floats.IEEE754.IEEE32Exec.fp32Round (IEEE32Exec.toReal acc + termFP.val) := by
        simp [embed, termFP, fp32_add_val]
      have hstartVal : (embed (IEEE32Exec.add acc termI)).val = ((embed acc) + termFP).val := by
        -- `embed` exposes `toReal`, and both additions round with the same `fp32Round`.
        simpa [embed, haddR, hfpStep, hterm]
      have hstart : embed (IEEE32Exec.add acc termI) = (embed acc) + termFP :=
        fp32_ext hstartVal
      -- Unfold one fold step and apply the IH (whose start accumulator is `embed (add acc termI)`).
      -- Then rewrite that start accumulator to `embed acc + termFP`.
      simpa [List.foldl, hingeSumStepIeee, termI, termFP, hstart] using ih

/--
Refine the whole executable hinge network to the FP32 rounded-`ℝ` network.

The theorem composes the fold refinement with the final rounded bias addition.  Its hypotheses are
exactly the finiteness obligations needed by the IEEE-754 bridge lemmas.
-/
theorem toReal_hinge_fun_ieee_eq_fp32_val {n : ℕ}
    (t c : Fin n → IEEE32Exec) (b x : IEEE32Exec)
    (hx : IEEE32Exec.isFinite x = true)
    (ht : ∀ i, IEEE32Exec.isFinite (t i) = true)
    (hc : ∀ i, IEEE32Exec.isFinite (c i) = true)
    (hSum :
      HingeSumFinite t c x (Numbers.zero : IEEE32Exec) (List.finRange n))
    (hOut : IEEE32Exec.isFinite (hingeFunIeee (t := t) (c := c) (b := b) x) = true) :
    IEEE32Exec.toReal (hingeFunIeee (t := t) (c := c) (b := b) x) =
      (hingeFunFp32 (t := embedVec t) (c := embedVec c) (b := embed b) (x := embed x)).val := by
  -- First refine the hinge sum.
  have hsum :
      IEEE32Exec.toReal (hingeSumIeee (c := c) (t := t) x) =
        (hingeSumFp32 (c := embedVec c) (t := embedVec t) (x := embed x)).val := by
    -- Rewrite `hinge_sum_fp32` into a fold on the FP32 accumulator component.
    have hsumFold :
        (hingeSumFp32 (c := embedVec c) (t := embedVec t) (x := embed x)) =
          (List.finRange n).foldl
            (fun acc32 i =>
              acc32 + hingeTermFp32 (c := embedVec c) (t := embedVec t) (x := embed x) i)
            (0 : FP32) := hinge_sum_fp32_eq_fold (c := embedVec c) (t := embedVec t) (x := embed x)
    -- Use the fold refinement lemma on the same list.
    have hfold :=
      toReal_hinge_sum_ieee_eq_fp32_val (t := t) (c := c) (x := x)
        (acc := (Numbers.zero : IEEE32Exec)) (xs := List.finRange n) hSum hx ht hc
    have hstartVal : (embed (Numbers.zero : IEEE32Exec)).val = (0 : FP32).val := by
      -- Both are real zero.
      simp [embed, Numbers.zero, TorchLean.Floats.IEEE754.IEEE32Exec.toReal_posZero]
    have hstart : embed (Numbers.zero : IEEE32Exec) = (0 : FP32) := fp32_ext hstartVal
    -- Convert the RHS fold's start accumulator using `hstart`, then rewrite via `hsumFold`.
    have hfold' :
        IEEE32Exec.toReal ((List.finRange n).foldl (hingeSumStepIeee c t x) (Numbers.zero :
          IEEE32Exec)) =
          ((List.finRange n).foldl
              (fun acc32 i =>
                acc32 + hingeTermFp32 (c := embedVec c) (t := embedVec t) (x := embed x) i)
              (0 : FP32)).val := by
        simpa [hingeSumIeee, hingeSumStepIeee, hstart] using hfold
    -- Finish by rewriting the FP32 fold back to `hinge_sum_fp32`.
    simpa [hsumFold] using hfold'
  -- Finally refine the last `+ b`.
  have haddR :
      IEEE32Exec.toReal (IEEE32Exec.add (hingeSumIeee (c := c) (t := t) x) b) =
        TorchLean.Floats.IEEE754.IEEE32Exec.fp32Round
          (IEEE32Exec.toReal (hingeSumIeee (c := c) (t := t) x) + IEEE32Exec.toReal b) :=
    TorchLean.Floats.IEEE754.IEEE32Exec.toReal_add_eq_fp32Round_of_isFinite
      (x := hingeSumIeee (c := c) (t := t) x) (y := b) (by simpa [hingeFunIeee] using hOut)
  have hfp :
      (hingeFunFp32 (t := embedVec t) (c := embedVec c) (b := embed b) (x := embed x)).val =
        TorchLean.Floats.IEEE754.IEEE32Exec.fp32Round
          ((hingeSumFp32 (c := embedVec c) (t := embedVec t) (x := embed x)).val + (embed b).val)
            := by
    -- `hinge_fun_fp32` is `sum + b`, and FP32 `+` rounds with `fp32Round`.
    simp [hingeFunFp32, fp32_add_val, embed]
  -- Combine.
  simp [hingeFunIeee, haddR, hfp, hsum]

/-! ## IEEE32Exec error bound inherited from the FP32 bound -/

/--
Lift the FP32 hinge-network rounding error bound to executable IEEE32Exec evaluation.

Once `toReal_hinge_fun_ieee_eq_fp32_val` identifies the executable result with the FP32 model, the
already-proved FP32 error bound transfers directly.
-/
theorem hinge_fun_ieee_abs_error_le {n : ℕ}
    (t c : Fin n → IEEE32Exec) (b x : IEEE32Exec)
    (hx : IEEE32Exec.isFinite x = true)
    (ht : ∀ i, IEEE32Exec.isFinite (t i) = true)
    (hc : ∀ i, IEEE32Exec.isFinite (c i) = true)
    (hSum :
      HingeSumFinite t c x (Numbers.zero : IEEE32Exec) (List.finRange n))
    (hOut : IEEE32Exec.isFinite (hingeFunIeee (t := t) (c := c) (b := b) x) = true) :
    |IEEE32Exec.toReal (hingeFunIeee (t := t) (c := c) (b := b) x) -
        hingeFunReal (t := embedVec t) (c := embedVec c) (b := embed b) (x := embed x)| ≤
      hingeFunErrorBound (t := embedVec t) (c := embedVec c) (b := embed b) (x := embed x) := by
  have href :
      IEEE32Exec.toReal (hingeFunIeee (t := t) (c := c) (b := b) x) =
        (hingeFunFp32 (t := embedVec t) (c := embedVec c) (b := embed b) (x := embed x)).val :=
    toReal_hinge_fun_ieee_eq_fp32_val (t := t) (c := c) (b := b) (x := x) hx ht hc hSum hOut
  -- Reuse the existing FP32 bound.
  simpa [href] using
    (hinge_fun_abs_error (t := embedVec t) (c := embedVec c) (b := embed b) (x := embed x))

/-! ## “Approximation error + rounding error” (pointwise) -/

/--
Pointwise triangle bound: target error to executable output is bounded by real approximation error
plus the certified FP32/IEEE rounding error.
-/
lemma hinge_fun_total_abs_error_ieee_le {n : ℕ} (f : ℝ → ℝ)
    (t c : Fin n → IEEE32Exec) (b x : IEEE32Exec)
    (hx : IEEE32Exec.isFinite x = true)
    (ht : ∀ i, IEEE32Exec.isFinite (t i) = true)
    (hc : ∀ i, IEEE32Exec.isFinite (c i) = true)
    (hSum :
      HingeSumFinite t c x (Numbers.zero : IEEE32Exec) (List.finRange n))
    (hOut : IEEE32Exec.isFinite (hingeFunIeee (t := t) (c := c) (b := b) x) = true) :
    |f (IEEE32Exec.toReal x) - IEEE32Exec.toReal (hingeFunIeee (t := t) (c := c) (b := b) x)| ≤
      |f (IEEE32Exec.toReal x) - hingeFunReal (t := embedVec t) (c := embedVec c) (b := embed b)
        (x := embed x)| +
        hingeFunErrorBound (t := embedVec t) (c := embedVec c) (b := embed b) (x := embed x) :=
          by
  have hround :
      |hingeFunReal (t := embedVec t) (c := embedVec c) (b := embed b) (x := embed x) -
          IEEE32Exec.toReal (hingeFunIeee (t := t) (c := c) (b := b) x)| ≤
        hingeFunErrorBound (t := embedVec t) (c := embedVec c) (b := embed b) (x := embed x) :=
          by
    simpa [abs_sub_comm] using
      (hinge_fun_ieee_abs_error_le (t := t) (c := c) (b := b) (x := x) hx ht hc hSum hOut)
  have htri :
      |f (IEEE32Exec.toReal x) - IEEE32Exec.toReal (hingeFunIeee (t := t) (c := c) (b := b) x)| ≤
        |f (IEEE32Exec.toReal x) - hingeFunReal (t := embedVec t) (c := embedVec c) (b := embed b)
          (x := embed x)| +
          |hingeFunReal (t := embedVec t) (c := embedVec c) (b := embed b) (x := embed x) -
              IEEE32Exec.toReal (hingeFunIeee (t := t) (c := c) (b := b) x)| := by
    simpa using
      (abs_sub_le (a := f (IEEE32Exec.toReal x))
        (b := hingeFunReal (t := embedVec t) (c := embedVec c) (b := embed b) (x := embed x))
        (c := IEEE32Exec.toReal (hingeFunIeee (t := t) (c := c) (b := b) x)))
  exact le_trans htri (add_le_add_right hround _)

/-- Strict version of `hinge_fun_total_abs_error_ieee_le`, useful for approximation theorems. -/
lemma hinge_fun_total_abs_error_ieee_lt {n : ℕ} (f : ℝ → ℝ)
    (t c : Fin n → IEEE32Exec) (b x : IEEE32Exec)
    (hx : IEEE32Exec.isFinite x = true)
    (ht : ∀ i, IEEE32Exec.isFinite (t i) = true)
    (hc : ∀ i, IEEE32Exec.isFinite (c i) = true)
    (hSum :
      HingeSumFinite t c x (Numbers.zero : IEEE32Exec) (List.finRange n))
    (hOut : IEEE32Exec.isFinite (hingeFunIeee (t := t) (c := c) (b := b) x) = true)
    {ε : ℝ}
    (hε :
      |f (IEEE32Exec.toReal x) -
          hingeFunReal (t := embedVec t) (c := embedVec c) (b := embed b) (x := embed x)| < ε) :
    |f (IEEE32Exec.toReal x) - IEEE32Exec.toReal (hingeFunIeee (t := t) (c := c) (b := b) x)| <
      ε + hingeFunErrorBound (t := embedVec t) (c := embedVec c) (b := embed b) (x := embed x) :=
        by
  have hle :=
    hinge_fun_total_abs_error_ieee_le (f := f) (t := t) (c := c) (b := b) (x := x) hx ht hc hSum
      hOut
  have hlt :
      |f (IEEE32Exec.toReal x) -
          hingeFunReal (t := embedVec t) (c := embedVec c) (b := embed b) (x := embed x)| +
        hingeFunErrorBound (t := embedVec t) (c := embedVec c) (b := embed b) (x := embed x) <
        ε + hingeFunErrorBound (t := embedVec t) (c := embedVec c) (b := embed b) (x := embed x)
          :=
    add_lt_add_of_lt_of_le hε (le_rfl)
  exact lt_of_le_of_lt hle hlt

/-! ## IEEE32Exec pointwise ReLU approximation packaging (1D) -/

/--
1D ReLU approximation statement over IEEE32Exec values.

This is **not** a full universal approximation theorem, because it does not construct IEEE weights
from a real target. Instead it packages the already-proved pointwise inequality
`hinge_fun_total_abs_error_ieee_lt` into an existence/for-all form:

- assume there exist IEEE32Exec hinge parameters `(t,c,b0)` that approximate `f` at the real level
  (via `hinge_fun_real` on the embedded reals),
- and assume a finiteness/no-NaN/no-Inf witness for IEEE32Exec evaluation,
- then IEEE32Exec evaluation approximates `f` with an explicit extra rounding term
  `hinge_fun_error_bound`.
-/
theorem reluApproximationIccIEEE32Exec_fromHinge
    {f : ℝ → ℝ} {a b : ℝ} :
    ∀ ε > 0,
      (∃ (hidDim : ℕ) (t c : Fin hidDim → IEEE32Exec) (b0 : IEEE32Exec),
          (∀ i, IEEE32Exec.isFinite (t i) = true) ∧
          (∀ i, IEEE32Exec.isFinite (c i) = true) ∧
          (∀ x : IEEE32Exec,
              IEEE32Exec.isFinite x = true →
              IEEE32Exec.toReal x ∈ Set.Icc a b →
                HingeSumFinite t c x (Numbers.zero : IEEE32Exec) (List.finRange hidDim) ∧
                IEEE32Exec.isFinite (hingeFunIeee (t := t) (c := c) (b := b0) x) = true) ∧
          (∀ x : IEEE32Exec,
              IEEE32Exec.isFinite x = true →
              IEEE32Exec.toReal x ∈ Set.Icc a b →
                |f (IEEE32Exec.toReal x) -
                    hingeFunReal (t := embedVec t) (c := embedVec c) (b := embed b0) (x := embed
                      x)| < ε))
      →
      ∃ (hidDim : ℕ) (t c : Fin hidDim → IEEE32Exec) (b0 : IEEE32Exec),
        ∀ x : IEEE32Exec,
          IEEE32Exec.isFinite x = true →
          IEEE32Exec.toReal x ∈ Set.Icc a b →
            |f (IEEE32Exec.toReal x) -
                IEEE32Exec.toReal (hingeFunIeee (t := t) (c := c) (b := b0) x)| <
              ε +
                hingeFunErrorBound (t := embedVec t) (c := embedVec c) (b := embed b0) (x :=
                  embed x) := by
  intro ε hε hex
  rcases hex with ⟨hidDim, t, c, b0, ht, hc, hFinite, hApprox⟩
  refine ⟨hidDim, t, c, b0, ?_⟩
  intro x hx hxIn
  rcases hFinite x hx hxIn with ⟨hSum, hOut⟩
  have hApproxX :=
    hApprox x hx hxIn
  -- Apply the proved pointwise bound.
  simpa using
    (hinge_fun_total_abs_error_ieee_lt (f := f) (t := t) (c := c) (b := b0) (x := x)
      (hx := hx) (ht := ht) (hc := hc) (hSum := hSum) (hOut := hOut) (ε := ε) hApproxX)

/-! ## Real approximation + quantization + IEEE rounding (1D, pointwise) -/

/--
1D IEEE32Exec ReLU approximation with an explicit 3-term error decomposition:

1. **Real approximation error**: `|f(r) - hinge_fun … r| < εApprox`
2. **Quantization/reference error**: `|hinge_fun … r - hinge_fun_real (embed IEEE-params) (embed r)|
  ≤ εQ`
3. **IEEE rounding error** (proved): `hinge_fun_error_bound …`

To obtain a fully synthesized IEEE32Exec approximation theorem, callers must additionally:
- construct IEEE32Exec parameters `(t,c,b0)` from the real hinge parameters, and
- prove the finiteness/no-NaN/no-Inf witnesses (`HingeSumFinite` + finite output), and
- prove a uniform bound `εQ` for the parameter-quantization step.
-/
theorem reluApproximationIccIEEE32Exec_threeTerm
    {f : ℝ → ℝ} {a b : ℝ} {hidDim : ℕ}
    (tR cR : Fin hidDim → ℝ) (t c : Fin hidDim → IEEE32Exec) (b0 : IEEE32Exec)
    (ht : ∀ i, IEEE32Exec.isFinite (t i) = true)
    (hc : ∀ i, IEEE32Exec.isFinite (c i) = true) :
    ∀ εApprox εQ : ℝ,
      (∀ x : IEEE32Exec,
          IEEE32Exec.isFinite x = true →
          IEEE32Exec.toReal x ∈ Set.Icc a b →
            HingeSumFinite t c x (Numbers.zero : IEEE32Exec) (List.finRange hidDim) ∧
            IEEE32Exec.isFinite (hingeFunIeee (t := t) (c := c) (b := b0) x) = true) →
      (∀ x : IEEE32Exec,
          IEEE32Exec.isFinite x = true →
          IEEE32Exec.toReal x ∈ Set.Icc a b →
            |f (IEEE32Exec.toReal x) - hingeFun hidDim tR cR (f a) (IEEE32Exec.toReal x)| < εApprox) →
      (∀ x : IEEE32Exec,
          IEEE32Exec.isFinite x = true →
          IEEE32Exec.toReal x ∈ Set.Icc a b →
            |hingeFun hidDim tR cR (f a) (IEEE32Exec.toReal x) -
                hingeFunReal (t := embedVec t) (c := embedVec c) (b := embed b0) (x := embed x)| ≤
                  εQ) →
      ∀ x : IEEE32Exec,
        IEEE32Exec.isFinite x = true →
        IEEE32Exec.toReal x ∈ Set.Icc a b →
          |f (IEEE32Exec.toReal x) -
              IEEE32Exec.toReal (hingeFunIeee (t := t) (c := c) (b := b0) x)| <
            (εApprox + εQ) +
              hingeFunErrorBound (t := embedVec t) (c := embedVec c) (b := embed b0) (x := embed
                x) := by
  intro εApprox εQ hFinite hApprox hQ x hx hxIn
  rcases hFinite x hx hxIn with ⟨hSum, hOut⟩
  have hApproxx : |f (IEEE32Exec.toReal x) - hingeFun hidDim tR cR (f a) (IEEE32Exec.toReal x)| < εApprox
    :=
    hApprox x hx hxIn
  have hQx :
      |hingeFun hidDim tR cR (f a) (IEEE32Exec.toReal x) -
          hingeFunReal (t := embedVec t) (c := embedVec c) (b := embed b0) (x := embed x)| ≤ εQ :=
    hQ x hx hxIn
  have hApproxEmbed :
      |f (IEEE32Exec.toReal x) -
          hingeFunReal (t := embedVec t) (c := embedVec c) (b := embed b0) (x := embed x)| <
        εApprox + εQ := by
    have htri :
        |f (IEEE32Exec.toReal x) -
            hingeFunReal (t := embedVec t) (c := embedVec c) (b := embed b0) (x := embed x)| ≤
          |f (IEEE32Exec.toReal x) - hingeFun hidDim tR cR (f a) (IEEE32Exec.toReal x)| +
            |hingeFun hidDim tR cR (f a) (IEEE32Exec.toReal x) -
                hingeFunReal (t := embedVec t) (c := embedVec c) (b := embed b0) (x := embed x)|
                  := by
      simpa using
        (abs_sub_le
          (a := f (IEEE32Exec.toReal x))
          (b := hingeFun hidDim tR cR (f a) (IEEE32Exec.toReal x))
          (c := hingeFunReal (t := embedVec t) (c := embedVec c) (b := embed b0) (x := embed x)))
    have hsum :
        |f (IEEE32Exec.toReal x) - hingeFun hidDim tR cR (f a) (IEEE32Exec.toReal x)| +
            |hingeFun hidDim tR cR (f a) (IEEE32Exec.toReal x) -
                hingeFunReal (t := embedVec t) (c := embedVec c) (b := embed b0) (x := embed x)| <
          εApprox + εQ :=
      add_lt_add_of_lt_of_le hApproxx hQx
    exact lt_of_le_of_lt htri hsum
  -- Apply the already-proved pointwise “real approximation + IEEE rounding” combination lemma.
  have := hinge_fun_total_abs_error_ieee_lt (f := f) (t := t) (c := c) (b := b0) (x := x)
    (hx := hx) (ht := ht) (hc := hc) (hSum := hSum) (hOut := hOut) (ε := εApprox + εQ) hApproxEmbed
  -- Rearrange the RHS as `(εApprox + εQ) + roundErr`.
  simpa [add_assoc] using this

/-! ## Pointwise wrappers that take `HingeEvalFiniteProp` -/

/--
Pointwise strict error theorem using the compact finiteness bundle.

Use this form when a checker or proof generator has produced one `HingeEvalFiniteProp` certificate
instead of separate hypotheses for input, parameter, fold, and output finiteness.
-/
theorem hinge_fun_total_abs_error_ieee_lt_of_finiteProp {n : ℕ}
    {f : ℝ → ℝ} (t c : Fin n → IEEE32Exec) (b x : IEEE32Exec)
    {ε : ℝ}
    (hFin : HingeEvalFiniteProp t c b x)
    (hε :
      |f (IEEE32Exec.toReal x) -
          hingeFunReal (t := embedVec t) (c := embedVec c) (b := embed b) (x := embed x)| < ε) :
    |f (IEEE32Exec.toReal x) - IEEE32Exec.toReal (hingeFunIeee (t := t) (c := c) (b := b) x)| <
      ε + hingeFunErrorBound (t := embedVec t) (c := embedVec c) (b := embed b) (x := embed x) :=
        by
  rcases (hingeEvalFiniteProp_to_witness (t := t) (c := c) (b := b) (x := x) hFin) with
    ⟨hx, ht, hc, hSum, hOut⟩
  simpa using
    (hinge_fun_total_abs_error_ieee_lt (f := f) (t := t) (c := c) (b := b) (x := x)
      (hx := hx) (ht := ht) (hc := hc) (hSum := hSum) (hOut := hOut) (ε := ε) hε)

/-! ## Same 3-term theorem, but with an explicit real bias parameter -/

/--
Three-term IEEE32Exec approximation theorem with a caller-supplied real bias.

`reluApproximationIccIEEE32Exec_threeTerm` uses `f a` as the real hinge-network bias because that
is what the constructive one-dimensional interpolation theorem emits.  This variant is the more
general numerical-analysis statement: any real reference bias `bR` may be compared with the
executable bias `b0`.
-/
theorem reluApproximationIccIEEE32Exec_threeTerm_bias
    {f : ℝ → ℝ} {a b : ℝ} {hidDim : ℕ}
    (bR : ℝ)
    (tR cR : Fin hidDim → ℝ) (t c : Fin hidDim → IEEE32Exec) (b0 : IEEE32Exec)
    (ht : ∀ i, IEEE32Exec.isFinite (t i) = true)
    (hc : ∀ i, IEEE32Exec.isFinite (c i) = true) :
    ∀ εApprox εQ : ℝ,
      (∀ x : IEEE32Exec,
          IEEE32Exec.isFinite x = true →
          IEEE32Exec.toReal x ∈ Set.Icc a b →
            HingeSumFinite t c x (Numbers.zero : IEEE32Exec) (List.finRange hidDim) ∧
            IEEE32Exec.isFinite (hingeFunIeee (t := t) (c := c) (b := b0) x) = true) →
      (∀ x : IEEE32Exec,
          IEEE32Exec.isFinite x = true →
          IEEE32Exec.toReal x ∈ Set.Icc a b →
            |f (IEEE32Exec.toReal x) - hingeFun hidDim tR cR bR (IEEE32Exec.toReal x)| < εApprox) →
      (∀ x : IEEE32Exec,
          IEEE32Exec.isFinite x = true →
          IEEE32Exec.toReal x ∈ Set.Icc a b →
            |hingeFun hidDim tR cR bR (IEEE32Exec.toReal x) -
                hingeFunReal (t := embedVec t) (c := embedVec c) (b := embed b0) (x := embed x)| ≤
                  εQ) →
      ∀ x : IEEE32Exec,
        IEEE32Exec.isFinite x = true →
        IEEE32Exec.toReal x ∈ Set.Icc a b →
          |f (IEEE32Exec.toReal x) -
              IEEE32Exec.toReal (hingeFunIeee (t := t) (c := c) (b := b0) x)| <
            (εApprox + εQ) +
              hingeFunErrorBound (t := embedVec t) (c := embedVec c) (b := embed b0) (x := embed
                x) := by
  intro εApprox εQ hFinite hApprox hQ x hx hxIn
  rcases hFinite x hx hxIn with ⟨hSum, hOut⟩
  have hApproxx : |f (IEEE32Exec.toReal x) - hingeFun hidDim tR cR bR (IEEE32Exec.toReal x)| < εApprox :=
    hApprox x hx hxIn
  have hQx :
      |hingeFun hidDim tR cR bR (IEEE32Exec.toReal x) -
          hingeFunReal (t := embedVec t) (c := embedVec c) (b := embed b0) (x := embed x)| ≤ εQ :=
    hQ x hx hxIn
  have hApproxEmbed :
      |f (IEEE32Exec.toReal x) -
          hingeFunReal (t := embedVec t) (c := embedVec c) (b := embed b0) (x := embed x)| <
        εApprox + εQ := by
    have htri :
        |f (IEEE32Exec.toReal x) -
            hingeFunReal (t := embedVec t) (c := embedVec c) (b := embed b0) (x := embed x)| ≤
          |f (IEEE32Exec.toReal x) - hingeFun hidDim tR cR bR (IEEE32Exec.toReal x)| +
            |hingeFun hidDim tR cR bR (IEEE32Exec.toReal x) -
                hingeFunReal (t := embedVec t) (c := embedVec c) (b := embed b0) (x := embed x)|
                  := by
      simpa using
        (abs_sub_le
          (a := f (IEEE32Exec.toReal x))
          (b := hingeFun hidDim tR cR bR (IEEE32Exec.toReal x))
          (c := hingeFunReal (t := embedVec t) (c := embedVec c) (b := embed b0) (x := embed x)))
    have hsum :
        |f (IEEE32Exec.toReal x) - hingeFun hidDim tR cR bR (IEEE32Exec.toReal x)| +
            |hingeFun hidDim tR cR bR (IEEE32Exec.toReal x) -
                hingeFunReal (t := embedVec t) (c := embedVec c) (b := embed b0) (x := embed x)| <
          εApprox + εQ :=
      add_lt_add_of_lt_of_le hApproxx hQx
    exact lt_of_le_of_lt htri hsum
  -- Apply the already-proved pointwise “real approximation + IEEE rounding” combination lemma.
  have := hinge_fun_total_abs_error_ieee_lt (f := f) (t := t) (c := c) (b := b0) (x := x)
    (hx := hx) (ht := ht) (hc := hc) (hSum := hSum) (hOut := hOut) (ε := εApprox + εQ) hApproxEmbed
  -- Rearrange the RHS as `(εApprox + εQ) + roundErr`.
  simpa [add_assoc] using this

/-! ## Dyadic (`roundDyadicToIEEE32`) quantization helpers -/

/--
Generic half-ulp absolute-error bound for the rounded-`ℝ` binary32 model.

This is the standard floating-point local rounding statement specialized to TorchLean's FP32
format parameters.  It is the bridge from abstract approximation coefficients to explicit
quantization budgets.
-/
theorem fp32Round_abs_error_bound (x : ℝ) :
    abs (TorchLean.Floats.IEEE754.IEEE32Exec.fp32Round x - x) ≤
      neuralUlp binaryRadix fexp32 x TrainingPhase.forward / 2 := by
  simpa [TorchLean.Floats.IEEE754.IEEE32Exec.fp32Round] using
    (neural_error_bound_ulp
      (β := binaryRadix) (fexp := fexp32) (rnd := TorchLean.Floats.rnd32) x)

/--
Half-ulp error bound for dyadic values rounded into executable IEEE32Exec values.

The finiteness hypothesis excludes overflow/NaN paths, after which `roundDyadicToIEEE32` agrees
with the same real `fp32Round` operation used by the FP32 model.
-/
theorem toReal_roundDyadicToIEEE32_abs_error_bound (d : IEEE32Exec.Dyadic)
    (hfin : IEEE32Exec.isFinite (IEEE32Exec.roundDyadicToIEEE32 d) = true) :
    abs (IEEE32Exec.toReal (IEEE32Exec.roundDyadicToIEEE32 d) - IEEE32Exec.dyadicToReal d) ≤
      neuralUlp binaryRadix fexp32 (IEEE32Exec.dyadicToReal d) TrainingPhase.forward / 2 := by
  have hbridge :=
    TorchLean.Floats.IEEE754.IEEE32Exec.toReal_roundDyadicToIEEE32_eq_fp32Round (d := d) hfin
  -- Reduce to the generic rounding bound for `fp32Round`.
  simpa [hbridge] using (fp32Round_abs_error_bound (x := IEEE32Exec.dyadicToReal d))

/-!
### Real hinge network sensitivity to parameter rounding (for a compact input domain)

This is the “εQ quantization” step for
  `reluApproximationIccIEEE32Exec_threeTerm`.
The lemma below uses the compact-domain assumption that the *reference* knot locations `tR`
lie in the input interval `[a,b]`, then bounds the output perturbation using:
- a width term `|b-a|` to bound `relu(x - tR i)` on the interval, and
- `relu_lipschitz` to control the sensitivity to perturbing `t`.
-/

/-- Real ReLU is nonnegative. -/
private lemma relu_nonneg (u : ℝ) : 0 ≤ relu u := by
  -- `relu u = max u 0`.
  simp [relu, Activation.Math.reluSpec]

/-- Since ReLU is nonnegative, its absolute value is itself. -/
private lemma abs_relu (u : ℝ) : abs (relu u) = relu u := by
  simp [abs_of_nonneg (relu_nonneg u)]

/-- ReLU is pointwise bounded by absolute value. -/
private lemma relu_le_abs (u : ℝ) : relu u ≤ abs u := by
  by_cases hu : 0 ≤ u
  · simp [relu, Activation.Math.reluSpec, max_eq_left hu, abs_of_nonneg hu]
  · have hu' : u ≤ 0 := le_of_not_ge hu
    -- `relu u = 0` for `u ≤ 0`, so the goal is `0 ≤ |u|`.
    simp [relu, Activation.Math.reluSpec, max_eq_right hu']

/--
If both points lie in `[a,b]`, their distance is bounded by the interval width.

This is the compact-domain geometric fact used to control the size of each hinge activation under
parameter perturbations.
-/
private lemma abs_sub_le_abs_width_Icc {a b x t : ℝ}
    (hx : x ∈ Set.Icc a b) (ht : t ∈ Set.Icc a b) :
    abs (x - t) ≤ abs (b - a) := by
  have hab : a ≤ b := le_trans hx.1 hx.2
  have h1 : x - t ≤ b - a := by linarith [hx.2, ht.1]
  have h2 : -(b - a) ≤ x - t := by
    have : t - x ≤ b - a := by linarith [ht.2, hx.1]
    linarith
  have habs : abs (x - t) ≤ b - a := (abs_le.2 ⟨h2, h1⟩)
  have hbnonneg : 0 ≤ b - a := sub_nonneg.mpr hab
  simpa [abs_of_nonneg hbnonneg] using habs

/-- On `[a,b]`, a hinge activation `relu (x - t)` is bounded by the interval width. -/
private lemma abs_relu_sub_le_abs_width_Icc {a b x t : ℝ}
    (hx : x ∈ Set.Icc a b) (ht : t ∈ Set.Icc a b) :
    abs (relu (x - t)) ≤ abs (b - a) := by
  calc
    abs (relu (x - t)) = relu (x - t) := abs_relu (x - t)
    _ ≤ abs (x - t) := relu_le_abs (x - t)
    _ ≤ abs (b - a) := abs_sub_le_abs_width_Icc (hx := hx) (ht := ht)

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
  -- Separate bias and sum.
  have htri0 :
      abs (hingeFun n tR cR bR x - hingeFun n tI cI bI x) ≤
        abs (bR - bI) +
          abs ((∑ i : Fin n, cR i * relu (x - tR i)) - (∑ i : Fin n, cI i * relu (x - tI i))) := by
    -- `hinge_fun = b + sum`.
    simpa [hingeFun, sub_eq_add_neg, add_assoc, add_left_comm, add_comm] using
      (abs_add_le (bR - bI)
        ((∑ i : Fin n, cR i * relu (x - tR i)) - (∑ i : Fin n, cI i * relu (x - tI i))))
  -- Bound the difference of sums by summing per-term bounds.
  have hsum0 :
      abs ((∑ i : Fin n, cR i * relu (x - tR i)) - (∑ i : Fin n, cI i * relu (x - tI i))) ≤
        ∑ i : Fin n, abs (cR i * relu (x - tR i) - cI i * relu (x - tI i)) := by
    -- Rewrite the difference of sums as a sum of differences, then use `abs_sum_le_sum_abs`.
    have :
        (∑ i : Fin n, cR i * relu (x - tR i)) - (∑ i : Fin n, cI i * relu (x - tI i)) =
          ∑ i : Fin n, (cR i * relu (x - tR i) - cI i * relu (x - tI i)) := by
      exact
        (Finset.sum_sub_distrib (s := (Finset.univ : Finset (Fin n)))
          (f := fun i => cR i * relu (x - tR i))
          (g := fun i => cI i * relu (x - tI i))).symm
    -- Apply the absolute-sum inequality on `Finset.univ`.
    rw [this]
    simpa using
      (Finset.abs_sum_le_sum_abs (s := (Finset.univ : Finset (Fin n)))
        (f := fun i => cR i * relu (x - tR i) - cI i * relu (x - tI i)))
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
      simpa [hdecomp] using
        (abs_add_le ((cR i - cI i) * relu (x - tR i)) (cI i * (relu (x - tR i) - relu (x - tI i))))
    have hreluWidth : abs (relu (x - tR i)) ≤ abs (b - a) :=
      abs_relu_sub_le_abs_width_Icc (hx := hx) (ht := htR i)
    have hreluLip :
        abs (relu (x - tR i) - relu (x - tI i)) ≤ abs (tR i - tI i) := by
      have h := relu_lipschitz (x - tR i) (x - tI i)
      have hdiff : (x - tR i) - (x - tI i) = tI i - tR i := by ring
      calc
        abs (relu (x - tR i) - relu (x - tI i))
            ≤ abs ((x - tR i) - (x - tI i)) := h
        _ = abs (tI i - tR i) := by simp [hdiff]
        _ = abs (tR i - tI i) := by
              simpa using (abs_sub_comm (tI i) (tR i))
    -- Put the pieces together, using `|ab| = |a||b|`.
    have h1 :
        abs ((cR i - cI i) * relu (x - tR i)) ≤ abs (cR i - cI i) * abs (b - a) := by
      -- `abs_mul` + `relu` width bound.
      have : abs ((cR i - cI i) * relu (x - tR i)) = abs (cR i - cI i) * abs (relu (x - tR i)) := by
        simp [abs_mul]
      -- Replace `abs (relu …)` with its bound.
      calc
        abs ((cR i - cI i) * relu (x - tR i))
            = abs (cR i - cI i) * abs (relu (x - tR i)) := this
        _ ≤ abs (cR i - cI i) * abs (b - a) := by
            exact mul_le_mul_of_nonneg_left hreluWidth (abs_nonneg (cR i - cI i))
    have h2 :
        abs (cI i * (relu (x - tR i) - relu (x - tI i))) ≤ abs (cI i) * abs (tR i - tI i) := by
      have : abs (cI i * (relu (x - tR i) - relu (x - tI i))) =
          abs (cI i) * abs (relu (x - tR i) - relu (x - tI i)) := by
        simp [abs_mul]
      calc
        abs (cI i * (relu (x - tR i) - relu (x - tI i)))
            = abs (cI i) * abs (relu (x - tR i) - relu (x - tI i)) := this
        _ ≤ abs (cI i) * abs (tR i - tI i) := by
            exact mul_le_mul_of_nonneg_left hreluLip (abs_nonneg (cI i))
    -- Final bound for this term.
    linarith
  -- Combine `hsum0` with the per-term bounds.
  have hsum :
      abs ((∑ i : Fin n, cR i * relu (x - tR i)) - (∑ i : Fin n, cI i * relu (x - tI i))) ≤
        ∑ i : Fin n,
          (abs (cR i - cI i) * abs (b - a) + abs (cI i) * abs (tR i - tI i)) := by
    have hsum1 : (∑ i : Fin n, abs (cR i * relu (x - tR i) - cI i * relu (x - tI i))) ≤
        ∑ i : Fin n,
          (abs (cR i - cI i) * abs (b - a) + abs (cI i) * abs (tR i - tI i)) := by
      exact Finset.sum_le_sum (fun i _ => hterm i)
    exact le_trans hsum0 hsum1
  -- Finish: substitute the improved sum bound into the bias+sum triangle bound.
  exact le_trans htri0 (by gcongr)

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
  have hbase :=
    hinge_fun_abs_error_le_of_params_Icc (tR := tR) (cR := cR) (tI := tI) (cI := cI)
      (bR := bR) (bI := bI) (hx := hx) (htR := htR)
  -- Bound the per-neuron summand uniformly.
  have hterm :
      ∀ i : Fin n,
        abs (cR i - cI i) * abs (b - a) + abs (cI i) * abs (tR i - tI i) ≤
          Δc * abs (b - a) + C * Δt := by
    intro i
    have h1 : abs (cR i - cI i) * abs (b - a) ≤ Δc * abs (b - a) := by
      exact mul_le_mul_of_nonneg_right (hΔc i) (abs_nonneg (b - a))
    have h2 : abs (cI i) * abs (tR i - tI i) ≤ C * Δt := by
      have h2a : abs (cI i) * abs (tR i - tI i) ≤ C * abs (tR i - tI i) := by
        exact mul_le_mul_of_nonneg_right (hC i) (abs_nonneg (tR i - tI i))
      have h2b : C * abs (tR i - tI i) ≤ C * Δt := by
        exact mul_le_mul_of_nonneg_left (hΔt i) hC0
      exact le_trans h2a h2b
    exact add_le_add h1 h2
  have hsum :
      (∑ i : Fin n,
          (abs (cR i - cI i) * abs (b - a) + abs (cI i) * abs (tR i - tI i))) ≤
        ∑ _i : Fin n, (Δc * abs (b - a) + C * Δt) := by
    exact Finset.sum_le_sum (fun i _ => hterm i)
  have hsum' :
      (∑ _i : Fin n, (Δc * abs (b - a) + C * Δt)) = (n : ℝ) * (Δc * abs (b - a) + C * Δt) := by
    simp [mul_add]
  -- Combine.
  calc
    abs (hingeFun n tR cR bR x - hingeFun n tI cI bI x)
        ≤ abs (bR - bI) +
            ∑ i : Fin n,
              (abs (cR i - cI i) * abs (b - a) + abs (cI i) * abs (tR i - tI i)) := hbase
    _ ≤ abs (bR - bI) + ∑ _i : Fin n, (Δc * abs (b - a) + C * Δt) := by
          gcongr
    _ = abs (bR - bI) + (n : ℝ) * (Δc * abs (b - a) + C * Δt) := by
          simp [hsum']

/-! ## Removing the quantization term when reals are exactly representable -/

/--
The embedded FP32/IEEE real reference is exactly the ordinary real hinge network evaluated on
entrywise `IEEE32Exec.toReal` parameters.
-/
theorem hinge_fun_real_embed_eq_hinge_fun_toReal {n : ℕ}
    (t c : Fin n → IEEE32Exec) (b x : IEEE32Exec) :
    hingeFunReal (t := embedVec t) (c := embedVec c) (b := embed b) (x := embed x) =
      hingeFun n
        (fun i => IEEE32Exec.toReal (t i))
        (fun i => IEEE32Exec.toReal (c i))
        (IEEE32Exec.toReal b)
        (IEEE32Exec.toReal x) := by
  classical
  -- Expand `hinge_fun_real` via `hinge_sum_real_eq_sum`, then match `hinge_fun`.
  have hsum :
      hingeSumReal (c := embedVec c) (t := embedVec t) (x := embed x) =
        ∑ i : Fin n,
          (IEEE32Exec.toReal (c i)) * relu (IEEE32Exec.toReal x - IEEE32Exec.toReal (t i)) := by
    -- `hinge_sum_real` is a sum of `hinge_term_real`; embeddings expose `toReal` values.
    simpa [hingeTermReal, embedVec, embed] using
      (hinge_sum_real_eq_sum (c := embedVec c) (t := embedVec t) (x := embed x))
  -- Finish (commute the final `+ b`).
  simpa [hingeFunReal, hingeFun, hsum, embed, add_comm, add_left_comm, add_assoc]

/--
Uniform quantization-error bound between a real hinge network and executable IEEE parameters.

This is the user-facing `εQ` discharge lemma when callers can bound coefficient and knot rounding
errors uniformly.
-/
  theorem hinge_fun_quantization_error_le_Icc_uniform
      {n : ℕ} {a b : ℝ}
      (tR cR : Fin n → ℝ) (bR : ℝ)
      (t c : Fin n → IEEE32Exec) (b0 x : IEEE32Exec)
      (hxIn : IEEE32Exec.toReal x ∈ Set.Icc a b)
      (htR : ∀ i, tR i ∈ Set.Icc a b)
      {Δc C Δt : ℝ}
      (hC0 : 0 ≤ C)
      (hΔc : ∀ i, abs (cR i - IEEE32Exec.toReal (c i)) ≤ Δc)
      (hC : ∀ i, abs (IEEE32Exec.toReal (c i)) ≤ C)
      (hΔt : ∀ i, abs (tR i - IEEE32Exec.toReal (t i)) ≤ Δt) :
    abs (hingeFun n tR cR bR (IEEE32Exec.toReal x) -
          hingeFunReal (t := embedVec t) (c := embedVec c) (b := embed b0) (x := embed x)) ≤
      abs (bR - IEEE32Exec.toReal b0) + (n : ℝ) * (Δc * abs (b - a) + C * Δt) := by
  -- Rewrite the embedded-real reference into an explicit `hinge_fun` on `toReal` parameters.
  have hreal :
      hingeFunReal (t := embedVec t) (c := embedVec c) (b := embed b0) (x := embed x) =
        hingeFun n
          (fun i => IEEE32Exec.toReal (t i))
          (fun i => IEEE32Exec.toReal (c i))
          (IEEE32Exec.toReal b0)
          (IEEE32Exec.toReal x) := by
    simpa using (hinge_fun_real_embed_eq_hinge_fun_toReal (t := t) (c := c) (b := b0) (x := x))
  -- Apply the uniform parameter-perturbation bound on `[a,b]`.
  have :=
    hinge_fun_abs_error_le_of_params_Icc_uniform
      (tR := tR) (cR := cR)
      (tI := fun i => IEEE32Exec.toReal (t i))
      (cI := fun i => IEEE32Exec.toReal (c i))
      (bR := bR) (bI := IEEE32Exec.toReal b0)
      (x := IEEE32Exec.toReal x)
      (a := a) (b := b)
      (hx := hxIn) (htR := htR)
      (hC0 := hC0)
      (hΔc := hΔc) (hC := hC) (hΔt := hΔt)
  simpa [hreal] using this

/-!
### Dyadic-to-IEEE32Exec quantization bound (`εQ`)

This lemma is a drop-in way to discharge the `εQ` premise of
`reluApproximationIccIEEE32Exec_threeTerm` when IEEE parameters are obtained by
rounding dyadic rationals via `roundDyadicToIEEE32`.
-/

/--
Dyadic quantization error for a hinge network on `[a,b]`.

The real reference uses exact dyadic parameters, while the executable reference uses those dyadics
rounded to IEEE32Exec values.  The result is still expressed as a sum of concrete per-parameter
rounding errors.
-/
theorem hinge_fun_dyadic_quantization_error_le_Icc
    {n : ℕ} {a b : ℝ}
    (tD cD : Fin n → IEEE32Exec.Dyadic) (bD : IEEE32Exec.Dyadic)
    (x : IEEE32Exec)
    (hxIn : IEEE32Exec.toReal x ∈ Set.Icc a b)
    (htIn : ∀ i, IEEE32Exec.dyadicToReal (tD i) ∈ Set.Icc a b) :
    abs
        (hingeFun n
            (fun i => IEEE32Exec.dyadicToReal (tD i))
            (fun i => IEEE32Exec.dyadicToReal (cD i))
            (IEEE32Exec.dyadicToReal bD)
            (IEEE32Exec.toReal x) -
          hingeFunReal
            (t := embedVec (fun i => IEEE32Exec.roundDyadicToIEEE32 (tD i)))
            (c := embedVec (fun i => IEEE32Exec.roundDyadicToIEEE32 (cD i)))
            (b := embed (IEEE32Exec.roundDyadicToIEEE32 bD))
            (x := embed x)) ≤
      abs (IEEE32Exec.dyadicToReal bD - IEEE32Exec.toReal (IEEE32Exec.roundDyadicToIEEE32 bD)) +
        ∑ i : Fin n,
          (abs (IEEE32Exec.dyadicToReal (cD i) - IEEE32Exec.toReal (IEEE32Exec.roundDyadicToIEEE32
            (cD i))) *
              abs (b - a) +
            abs (IEEE32Exec.toReal (IEEE32Exec.roundDyadicToIEEE32 (cD i))) *
              abs (IEEE32Exec.dyadicToReal (tD i) - IEEE32Exec.toReal
                (IEEE32Exec.roundDyadicToIEEE32 (tD i)))) := by
  classical
  -- Rewrite the `hinge_fun_real` reference into a `hinge_fun` over the embedded reals.
  have hreal :
      hingeFunReal
          (t := embedVec (fun i => IEEE32Exec.roundDyadicToIEEE32 (tD i)))
          (c := embedVec (fun i => IEEE32Exec.roundDyadicToIEEE32 (cD i)))
          (b := embed (IEEE32Exec.roundDyadicToIEEE32 bD))
          (x := embed x) =
        hingeFun n
          (fun i => IEEE32Exec.toReal (IEEE32Exec.roundDyadicToIEEE32 (tD i)))
          (fun i => IEEE32Exec.toReal (IEEE32Exec.roundDyadicToIEEE32 (cD i)))
          (IEEE32Exec.toReal (IEEE32Exec.roundDyadicToIEEE32 bD))
          (IEEE32Exec.toReal x) := by
    simpa using
      (hinge_fun_real_embed_eq_hinge_fun_toReal
        (t := fun i => IEEE32Exec.roundDyadicToIEEE32 (tD i))
        (c := fun i => IEEE32Exec.roundDyadicToIEEE32 (cD i))
        (b := IEEE32Exec.roundDyadicToIEEE32 bD)
        (x := x))
  -- Apply the general “parameter sensitivity on `[a,b]`” bound.
  have :=
    hinge_fun_abs_error_le_of_params_Icc
      (tR := fun i => IEEE32Exec.dyadicToReal (tD i))
      (cR := fun i => IEEE32Exec.dyadicToReal (cD i))
      (tI := fun i => IEEE32Exec.toReal (IEEE32Exec.roundDyadicToIEEE32 (tD i)))
      (cI := fun i => IEEE32Exec.toReal (IEEE32Exec.roundDyadicToIEEE32 (cD i)))
      (bR := IEEE32Exec.dyadicToReal bD)
      (bI := IEEE32Exec.toReal (IEEE32Exec.roundDyadicToIEEE32 bD))
      (x := IEEE32Exec.toReal x)
      (a := a) (b := b) hxIn htIn
  simpa [hreal] using this

/--
Dyadic quantization error with each per-parameter rounding error bounded by a half-ulp term.

This is the more automated version of `hinge_fun_dyadic_quantization_error_le_Icc`: callers provide
finiteness of every rounded dyadic value, and the theorem substitutes the standard half-ulp bounds.
-/
theorem hinge_fun_dyadic_quantization_error_le_Icc_halfUlp
    {n : ℕ} {a b : ℝ}
    (tD cD : Fin n → IEEE32Exec.Dyadic) (bD : IEEE32Exec.Dyadic)
    (x : IEEE32Exec)
    (hxIn : IEEE32Exec.toReal x ∈ Set.Icc a b)
    (htIn : ∀ i, IEEE32Exec.dyadicToReal (tD i) ∈ Set.Icc a b)
    (htfin : ∀ i, IEEE32Exec.isFinite (IEEE32Exec.roundDyadicToIEEE32 (tD i)) = true)
    (hcfin : ∀ i, IEEE32Exec.isFinite (IEEE32Exec.roundDyadicToIEEE32 (cD i)) = true)
    (hbfin : IEEE32Exec.isFinite (IEEE32Exec.roundDyadicToIEEE32 bD) = true) :
    abs
        (hingeFun n
            (fun i => IEEE32Exec.dyadicToReal (tD i))
            (fun i => IEEE32Exec.dyadicToReal (cD i))
            (IEEE32Exec.dyadicToReal bD)
            (IEEE32Exec.toReal x) -
          hingeFunReal
            (t := embedVec (fun i => IEEE32Exec.roundDyadicToIEEE32 (tD i)))
            (c := embedVec (fun i => IEEE32Exec.roundDyadicToIEEE32 (cD i)))
            (b := embed (IEEE32Exec.roundDyadicToIEEE32 bD))
            (x := embed x)) ≤
      neuralUlp binaryRadix fexp32 (IEEE32Exec.dyadicToReal bD) TrainingPhase.forward / 2 +
        ∑ i : Fin n,
          (neuralUlp binaryRadix fexp32 (IEEE32Exec.dyadicToReal (cD i)) TrainingPhase.forward / 2
            *
              abs (b - a) +
            abs (IEEE32Exec.toReal (IEEE32Exec.roundDyadicToIEEE32 (cD i))) *
              (neuralUlp binaryRadix fexp32 (IEEE32Exec.dyadicToReal (tD i)) TrainingPhase.forward
                / 2)) := by
  classical
  -- Start from the raw parameter-sensitivity bound.
  have hraw :=
    hinge_fun_dyadic_quantization_error_le_Icc (tD := tD) (cD := cD) (bD := bD) (x := x) hxIn htIn
  -- Bound each parameter-difference term by a half-ULP using
  -- `toReal_roundDyadicToIEEE32_abs_error_bound`.
  have hb :
      abs (IEEE32Exec.dyadicToReal bD - IEEE32Exec.toReal (IEEE32Exec.roundDyadicToIEEE32 bD)) ≤
        neuralUlp binaryRadix fexp32 (IEEE32Exec.dyadicToReal bD) TrainingPhase.forward / 2 := by
    simpa [abs_sub_comm] using
      (toReal_roundDyadicToIEEE32_abs_error_bound (d := bD) (hfin := hbfin))
  have hterm :
      ∀ i : Fin n,
        (abs (IEEE32Exec.dyadicToReal (cD i) - IEEE32Exec.toReal (IEEE32Exec.roundDyadicToIEEE32 (cD
          i))) *
              abs (b - a) +
            abs (IEEE32Exec.toReal (IEEE32Exec.roundDyadicToIEEE32 (cD i))) *
              abs (IEEE32Exec.dyadicToReal (tD i) - IEEE32Exec.toReal
                (IEEE32Exec.roundDyadicToIEEE32 (tD i))))
          ≤
            (neuralUlp binaryRadix fexp32 (IEEE32Exec.dyadicToReal (cD i)) TrainingPhase.forward /
              2 *
                abs (b - a) +
              abs (IEEE32Exec.toReal (IEEE32Exec.roundDyadicToIEEE32 (cD i))) *
                (neuralUlp binaryRadix fexp32 (IEEE32Exec.dyadicToReal (tD i))
                  TrainingPhase.forward / 2)) := by
    intro i
    have hc :
        abs (IEEE32Exec.dyadicToReal (cD i) - IEEE32Exec.toReal (IEEE32Exec.roundDyadicToIEEE32 (cD
          i))) ≤
          neuralUlp binaryRadix fexp32 (IEEE32Exec.dyadicToReal (cD i)) TrainingPhase.forward / 2
            := by
      simpa [abs_sub_comm] using
        (toReal_roundDyadicToIEEE32_abs_error_bound (d := cD i) (hfin := hcfin i))
    have ht :
        abs (IEEE32Exec.dyadicToReal (tD i) - IEEE32Exec.toReal (IEEE32Exec.roundDyadicToIEEE32 (tD
          i))) ≤
          neuralUlp binaryRadix fexp32 (IEEE32Exec.dyadicToReal (tD i)) TrainingPhase.forward / 2
            := by
      simpa [abs_sub_comm] using
        (toReal_roundDyadicToIEEE32_abs_error_bound (d := tD i) (hfin := htfin i))
    have hc' :
        abs (IEEE32Exec.dyadicToReal (cD i) - IEEE32Exec.toReal (IEEE32Exec.roundDyadicToIEEE32 (cD
          i))) *
            abs (b - a) ≤
          (neuralUlp binaryRadix fexp32 (IEEE32Exec.dyadicToReal (cD i)) TrainingPhase.forward /
            2) *
            abs (b - a) := by
      exact mul_le_mul_of_nonneg_right hc (abs_nonneg (b - a))
    have ht' :
        abs (IEEE32Exec.toReal (IEEE32Exec.roundDyadicToIEEE32 (cD i))) *
            abs (IEEE32Exec.dyadicToReal (tD i) - IEEE32Exec.toReal (IEEE32Exec.roundDyadicToIEEE32
              (tD i))) ≤
          abs (IEEE32Exec.toReal (IEEE32Exec.roundDyadicToIEEE32 (cD i))) *
            (neuralUlp binaryRadix fexp32 (IEEE32Exec.dyadicToReal (tD i)) TrainingPhase.forward /
              2) := by
      exact mul_le_mul_of_nonneg_left ht (abs_nonneg _)
    exact add_le_add hc' ht'
  -- Combine the termwise bounds under the sum, then add the bias bound.
  have hsum :
      (∑ i : Fin n,
          (abs (IEEE32Exec.dyadicToReal (cD i) - IEEE32Exec.toReal (IEEE32Exec.roundDyadicToIEEE32
            (cD i))) *
                abs (b - a) +
              abs (IEEE32Exec.toReal (IEEE32Exec.roundDyadicToIEEE32 (cD i))) *
                abs (IEEE32Exec.dyadicToReal (tD i) - IEEE32Exec.toReal
                  (IEEE32Exec.roundDyadicToIEEE32 (tD i))))) ≤
        ∑ i : Fin n,
          (neuralUlp binaryRadix fexp32 (IEEE32Exec.dyadicToReal (cD i)) TrainingPhase.forward / 2
            *
                abs (b - a) +
              abs (IEEE32Exec.toReal (IEEE32Exec.roundDyadicToIEEE32 (cD i))) *
                (neuralUlp binaryRadix fexp32 (IEEE32Exec.dyadicToReal (tD i))
                  TrainingPhase.forward / 2)) := by
    exact Finset.sum_le_sum (fun i _ => hterm i)
  -- Finish by transitivity.
  have := add_le_add hb hsum
  exact le_trans hraw this

/-!
### Packaged “dyadic quantization + IEEE rounding” theorem (1D, pointwise)

This is the “dyadic next step”: it replaces the abstract `εQ` premise of
`reluApproximationIccIEEE32Exec_threeTerm_bias` with a concrete bound coming from:
- dyadic→FP32 rounding (`≤ 1/2 ulp`), and
- a Lipschitz-style sensitivity bound for the real hinge network on `x ∈ [a,b]`.

It still assumes the IEEE finiteness witnesses for evaluation (`HingeSumFinite` + finite output).
-/

theorem reluApproximationIccIEEE32Exec_dyadicHalfUlp
    {f : ℝ → ℝ} {a b : ℝ} {hidDim : ℕ}
    (tD cD : Fin hidDim → IEEE32Exec.Dyadic) (bD : IEEE32Exec.Dyadic)
    (htIn : ∀ i, IEEE32Exec.dyadicToReal (tD i) ∈ Set.Icc a b)
    (htfin : ∀ i, IEEE32Exec.isFinite (IEEE32Exec.roundDyadicToIEEE32 (tD i)) = true)
    (hcfin : ∀ i, IEEE32Exec.isFinite (IEEE32Exec.roundDyadicToIEEE32 (cD i)) = true)
    (hbfin : IEEE32Exec.isFinite (IEEE32Exec.roundDyadicToIEEE32 bD) = true) :
    ∀ εApprox : ℝ,
      (∀ x : IEEE32Exec,
          IEEE32Exec.isFinite x = true →
          IEEE32Exec.toReal x ∈ Set.Icc a b →
            HingeSumFinite
                (fun i => IEEE32Exec.roundDyadicToIEEE32 (tD i))
                (fun i => IEEE32Exec.roundDyadicToIEEE32 (cD i))
                x (Numbers.zero : IEEE32Exec) (List.finRange hidDim) ∧
              IEEE32Exec.isFinite
                (hingeFunIeee
                  (t := fun i => IEEE32Exec.roundDyadicToIEEE32 (tD i))
                  (c := fun i => IEEE32Exec.roundDyadicToIEEE32 (cD i))
                  (b := IEEE32Exec.roundDyadicToIEEE32 bD) x) = true) →
      (∀ x : IEEE32Exec,
          IEEE32Exec.isFinite x = true →
          IEEE32Exec.toReal x ∈ Set.Icc a b →
            |f (IEEE32Exec.toReal x) -
                hingeFun hidDim
                  (fun i => IEEE32Exec.dyadicToReal (tD i))
                  (fun i => IEEE32Exec.dyadicToReal (cD i))
                  (IEEE32Exec.dyadicToReal bD)
                  (IEEE32Exec.toReal x)| < εApprox) →
      ∀ x : IEEE32Exec,
        IEEE32Exec.isFinite x = true →
        IEEE32Exec.toReal x ∈ Set.Icc a b →
          |f (IEEE32Exec.toReal x) -
              IEEE32Exec.toReal
                (hingeFunIeee
                  (t := fun i => IEEE32Exec.roundDyadicToIEEE32 (tD i))
                  (c := fun i => IEEE32Exec.roundDyadicToIEEE32 (cD i))
                  (b := IEEE32Exec.roundDyadicToIEEE32 bD) x)| <
            (εApprox +
              (neuralUlp binaryRadix fexp32 (IEEE32Exec.dyadicToReal bD) TrainingPhase.forward / 2
                +
                ∑ i : Fin hidDim,
                  (neuralUlp binaryRadix fexp32 (IEEE32Exec.dyadicToReal (cD i))
                    TrainingPhase.forward / 2 *
                      abs (b - a) +
                    abs (IEEE32Exec.toReal (IEEE32Exec.roundDyadicToIEEE32 (cD i))) *
                      (neuralUlp binaryRadix fexp32 (IEEE32Exec.dyadicToReal (tD i))
                        TrainingPhase.forward / 2)))) +
              hingeFunErrorBound
                (t := embedVec (fun i => IEEE32Exec.roundDyadicToIEEE32 (tD i)))
                (c := embedVec (fun i => IEEE32Exec.roundDyadicToIEEE32 (cD i)))
                (b := embed (IEEE32Exec.roundDyadicToIEEE32 bD))
                (x := embed x) := by
  intro εApprox hFinite hApprox x hx hxIn
  have ht : ∀ i, IEEE32Exec.isFinite (IEEE32Exec.roundDyadicToIEEE32 (tD i)) = true := htfin
  have hc : ∀ i, IEEE32Exec.isFinite (IEEE32Exec.roundDyadicToIEEE32 (cD i)) = true := hcfin
  -- Now apply the general 3-term (bias-parametric) theorem.
  let εQdy : ℝ :=
    neuralUlp binaryRadix fexp32 (IEEE32Exec.dyadicToReal bD) TrainingPhase.forward / 2 +
      ∑ i : Fin hidDim,
        (neuralUlp binaryRadix fexp32 (IEEE32Exec.dyadicToReal (cD i)) TrainingPhase.forward / 2 *
            abs (b - a) +
          abs (IEEE32Exec.toReal (IEEE32Exec.roundDyadicToIEEE32 (cD i))) *
            (neuralUlp binaryRadix fexp32 (IEEE32Exec.dyadicToReal (tD i)) TrainingPhase.forward /
              2))
  have :=
    (reluApproximationIccIEEE32Exec_threeTerm_bias
        (f := f) (a := a) (b := b) (hidDim := hidDim)
        (bR := IEEE32Exec.dyadicToReal bD)
        (tR := fun i => IEEE32Exec.dyadicToReal (tD i))
        (cR := fun i => IEEE32Exec.dyadicToReal (cD i))
        (t := fun i => IEEE32Exec.roundDyadicToIEEE32 (tD i))
        (c := fun i => IEEE32Exec.roundDyadicToIEEE32 (cD i))
        (b0 := IEEE32Exec.roundDyadicToIEEE32 bD)
        (ht := ht) (hc := hc))
      εApprox εQdy
      hFinite
      hApprox
      (by
        intro x hx hxIn
        exact
          hinge_fun_dyadic_quantization_error_le_Icc_halfUlp
            (tD := tD) (cD := cD) (bD := bD) (x := x)
            (hxIn := hxIn) (htIn := htIn) (htfin := htfin) (hcfin := hcfin) (hbfin := hbfin))
      x hx hxIn
  -- The conclusion matches, up to reassociation.
  simpa [εQdy, add_assoc] using this

/--
Two-term 1D IEEE32Exec ReLU approximation statement:

- assume the real hinge network built from the IEEE parameters’ `toReal` values already
  approximates `f` on `toReal` inputs in `[a,b]`,
- and assume finiteness/no-NaN/no-Inf witnesses,
- then IEEE execution approximates `f` with an explicit IEEE rounding term.

This is a specialization of the 3-term theorem with `εQ = 0`.
-/
theorem reluApproximationIccIEEE32Exec_twoTerm
    {f : ℝ → ℝ} {a b : ℝ} {hidDim : ℕ}
    (t c : Fin hidDim → IEEE32Exec) (b0 : IEEE32Exec)
    (ht : ∀ i, IEEE32Exec.isFinite (t i) = true)
    (hc : ∀ i, IEEE32Exec.isFinite (c i) = true) :
    ∀ εApprox : ℝ,
      (∀ x : IEEE32Exec,
          IEEE32Exec.isFinite x = true →
          IEEE32Exec.toReal x ∈ Set.Icc a b →
            HingeSumFinite t c x (Numbers.zero : IEEE32Exec) (List.finRange hidDim) ∧
            IEEE32Exec.isFinite (hingeFunIeee (t := t) (c := c) (b := b0) x) = true) →
      (∀ x : IEEE32Exec,
          IEEE32Exec.isFinite x = true →
          IEEE32Exec.toReal x ∈ Set.Icc a b →
            |f (IEEE32Exec.toReal x) -
                hingeFun hidDim
                  (fun i => IEEE32Exec.toReal (t i))
                  (fun i => IEEE32Exec.toReal (c i))
                  (IEEE32Exec.toReal b0)
                  (IEEE32Exec.toReal x)| < εApprox) →
      ∀ x : IEEE32Exec,
        IEEE32Exec.isFinite x = true →
        IEEE32Exec.toReal x ∈ Set.Icc a b →
          |f (IEEE32Exec.toReal x) -
              IEEE32Exec.toReal (hingeFunIeee (t := t) (c := c) (b := b0) x)| <
            εApprox +
              hingeFunErrorBound (t := embedVec t) (c := embedVec c) (b := embed b0) (x := embed
                x) := by
  intro εApprox hFinite hApprox x hx hxIn
  rcases hFinite x hx hxIn with ⟨hSum, hOut⟩
  have hApproxx :=
    hApprox x hx hxIn
  have hApproxEmbed :
      |f (IEEE32Exec.toReal x) -
          hingeFunReal (t := embedVec t) (c := embedVec c) (b := embed b0) (x := embed x)| < εApprox
            := by
    -- Replace `hinge_fun_real` by the explicit real hinge function on `toReal` parameters/inputs.
    simpa [hinge_fun_real_embed_eq_hinge_fun_toReal (t := t) (c := c) (b := b0) (x := x)] using
      hApproxx
  -- Apply the pointwise bound with `ε = εApprox`.
  simpa [add_comm, add_left_comm, add_assoc] using
    (hinge_fun_total_abs_error_ieee_lt (f := f) (t := t) (c := c) (b := b0) (x := x)
      (hx := hx) (ht := ht) (hc := hc) (hSum := hSum) (hOut := hOut) (ε := εApprox) hApproxEmbed)

end

end IEEE32ExecReLUApprox

end NN.MLTheory.Proofs.UniversalApproximation
