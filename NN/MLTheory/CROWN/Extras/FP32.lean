/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Floats.FP32
public import NN.Spec.Core.FloatInstances.NF
public import FloatLib.Floats.Interval.Rounders
public import FloatLib.Floats.Formats.Flocq.Theory.Rounding.Properties
public import NN.MLTheory.CROWN.BoundOps
public import NN.MLTheory.CROWN.Extras.IntervalLemmas
public import NN.MLTheory.CROWN.Graph
public import NN.MLTheory.CROWN.BoundOps.Lawful

/-!
# FP32

FP32-specialized entrypoints for the CROWN/LiRPA graph engine.

This is *not* an executable backend (it is `noncomputable` in general, because `FP32` is modeled on
`ℝ`). It exists so proofs can state “sound w.r.t. float32 semantics” without mentioning Lean’s
builtin `Float`.

This module is an optional convenience layer and lives under `NN/MLTheory/CROWN/Extras/`.
-/

@[expose] public section

open FloatLib FloatLib.Numerics FloatLib.Floats.Formats
open Flocq


namespace NN.MLTheory.CROWN

/-! ## FP32 entrypoints -/

/-- FP32 scalar type used for FP32-specialized CROWN/LiRPA statements. -/
abbrev FP32 := TorchLean.Floats.FP32

namespace FP32

open NN.MLTheory.CROWN.Graph

/--
Directed endpoint arithmetic for the rounded-real FP32 model.

Each operation is performed on the underlying real values and rounded directly toward the
appropriate side of the binary32 grid. The enclosure laws are
`FloatLib.Floats.Interval.roundDown_le` and `FloatLib.Floats.Interval.le_roundUp`.
-/
noncomputable instance : BoundOps FP32 where
  addDown a b :=
    ⟨FloatLib.Floats.Interval.roundDown
      (β := binaryRadix) (fexp := TorchLean.Floats.fexp32) (a.val + b.val)⟩
  addUp a b :=
    ⟨FloatLib.Floats.Interval.roundUp
      (β := binaryRadix) (fexp := TorchLean.Floats.fexp32) (a.val + b.val)⟩
  subDown a b :=
    ⟨FloatLib.Floats.Interval.roundDown
      (β := binaryRadix) (fexp := TorchLean.Floats.fexp32) (a.val - b.val)⟩
  subUp a b :=
    ⟨FloatLib.Floats.Interval.roundUp
      (β := binaryRadix) (fexp := TorchLean.Floats.fexp32) (a.val - b.val)⟩
  mulDown a b :=
    ⟨FloatLib.Floats.Interval.roundDown
      (β := binaryRadix) (fexp := TorchLean.Floats.fexp32) (a.val * b.val)⟩
  mulUp a b :=
    ⟨FloatLib.Floats.Interval.roundUp
      (β := binaryRadix) (fexp := TorchLean.Floats.fexp32) (a.val * b.val)⟩

/--
The proof-oriented FP32 endpoint operations enclose exact real arithmetic.

This follows directly from the format-generic floor and ceiling rounding theorems. It is the law
dictionary used by sound rounded CROWN statements over `FP32`.
-/
noncomputable instance : LawfulBoundOps FP32 where
  toReal := TorchLean.Floats.FP32.toReal
  lt_iff _ _ := Iff.rfl
  addDown_le a b := by
    change Flocq.round
        (β := binaryRadix) (fexp := TorchLean.Floats.fexp32)
        floorRound (a.val + b.val) ≤ a.val + b.val
    exact round_floor_le _
  le_addUp a b := by
    change a.val + b.val ≤
      Flocq.round
        (β := binaryRadix) (fexp := TorchLean.Floats.fexp32)
        ceilRound (a.val + b.val)
    exact le_round_ceil _
  subDown_le a b := by
    change Flocq.round
        (β := binaryRadix) (fexp := TorchLean.Floats.fexp32)
        floorRound (a.val - b.val) ≤ a.val - b.val
    exact round_floor_le _
  le_subUp a b := by
    change a.val - b.val ≤
      Flocq.round
        (β := binaryRadix) (fexp := TorchLean.Floats.fexp32)
        ceilRound (a.val - b.val)
    exact le_round_ceil _
  mulDown_le a b := by
    change Flocq.round
        (β := binaryRadix) (fexp := TorchLean.Floats.fexp32)
        floorRound (a.val * b.val) ≤ a.val * b.val
    exact round_floor_le _
  le_mulUp a b := by
    change a.val * b.val ≤
      Flocq.round
        (β := binaryRadix) (fexp := TorchLean.Floats.fexp32)
        ceilRound (a.val * b.val)
    exact le_round_ceil _

/-- Embed a real endpoint after rounding it downward to the binary32 grid. -/
noncomputable def roundDownEndpoint (x : ℝ) : FP32 :=
  ⟨FloatLib.Floats.Interval.roundDown
    (β := binaryRadix) (fexp := TorchLean.Floats.fexp32) x⟩

/-- Embed a real endpoint after rounding it upward to the binary32 grid. -/
noncomputable def roundUpEndpoint (x : ℝ) : FP32 :=
  ⟨FloatLib.Floats.Interval.roundUp
    (β := binaryRadix) (fexp := TorchLean.Floats.fexp32) x⟩

/-- Exact-real nonlinear operations rounded outward to the binary32 grid. -/
noncomputable instance : NonlinearBoundOps FP32 where
  divBounds aLo aHi bLo bHi :=
    if bLo.val > 0 || 0 > bHi.val then
      let p1 := aLo.val / bLo.val
      let p2 := aLo.val / bHi.val
      let p3 := aHi.val / bLo.val
      let p4 := aHi.val / bHi.val
      some (roundDownEndpoint (min (min p1 p2) (min p3 p4)),
        roundUpEndpoint (max (max p1 p2) (max p3 p4)))
    else
      none
  expBounds lo hi :=
    some (roundDownEndpoint (Real.exp lo.val), roundUpEndpoint (Real.exp hi.val))
  logBounds lo hi :=
    if lo.val > 0 then
      some (roundDownEndpoint (Real.log lo.val), roundUpEndpoint (Real.log hi.val))
    else
      none
  sqrtBounds lo hi :=
    if hi.val < 0 then
      none
    else
      some (roundDownEndpoint (Real.sqrt (max lo.val 0)), roundUpEndpoint (Real.sqrt hi.val))
  sigmoidBounds lo hi :=
    some (roundDownEndpoint (1 / (1 + Real.exp (-lo.val))),
      roundUpEndpoint (1 / (1 + Real.exp (-hi.val))))
  tanhBounds lo hi :=
    some (roundDownEndpoint (Real.tanh lo.val), roundUpEndpoint (Real.tanh hi.val))
  sinBounds := fun _ _ => some (roundDownEndpoint (-1), roundUpEndpoint 1)
  cosBounds := fun _ _ => some (roundDownEndpoint (-1), roundUpEndpoint 1)
  layerNormAbsBound n := some (roundUpEndpoint (Real.sqrt n))
  supportsIdealCoupledDerivatives := false

/-- Downward-rounded proof endpoints do not exceed their exact real inputs. -/
theorem roundDownEndpoint_le (x : ℝ) : (roundDownEndpoint x).val ≤ x := by
  exact round_floor_le x

/-- Upward-rounded proof endpoints do not fall below their exact real inputs. -/
theorem le_roundUpEndpoint (x : ℝ) : x ≤ (roundUpEndpoint x).val := by
  exact le_round_ceil x

private theorem roundedUnaryEnclosure_of_monotone (f : ℝ → ℝ) (hf : Monotone f) :
    UnaryEnclosure (α := FP32) f
      (fun lo hi ↦ some (roundDownEndpoint (f lo.val), roundUpEndpoint (f hi.val))) := by
  intro lo hi outLo outHi x hout hxLo hxHi
  have hpair : outLo = roundDownEndpoint (f lo.val) ∧
      outHi = roundUpEndpoint (f hi.val) := by
    simpa using Option.some.inj hout.symm
  rcases hpair with ⟨rfl, rfl⟩
  change (roundDownEndpoint (f lo.val)).val ≤ f x ∧ f x ≤ (roundUpEndpoint (f hi.val)).val
  exact ⟨(roundDownEndpoint_le _).trans (hf hxLo), (hf hxHi).trans (le_roundUpEndpoint _)⟩

private theorem roundedUnaryEnclosure_of_unit_range (f : ℝ → ℝ)
    (hf : ∀ x, -1 ≤ f x ∧ f x ≤ 1) :
    UnaryEnclosure (α := FP32) f
      (fun _ _ ↦ some (roundDownEndpoint (-1), roundUpEndpoint 1)) := by
  intro lo hi outLo outHi x hout _ _
  have hpair : outLo = roundDownEndpoint (-1) ∧ outHi = roundUpEndpoint 1 := by
    simpa using Option.some.inj hout.symm
  rcases hpair with ⟨rfl, rfl⟩
  change (roundDownEndpoint (-1)).val ≤ f x ∧ f x ≤ (roundUpEndpoint 1).val
  exact ⟨(roundDownEndpoint_le _).trans (hf x).1, (hf x).2.trans (le_roundUpEndpoint _)⟩

/-- The rounded-real FP32 nonlinear transfers enclose their exact real meanings. -/
noncomputable instance : LawfulNonlinearBoundOps FP32 where
  divBounds_enclosure := by
    intro aLo aHi bLo bHi outLo outHi x y hout hxLo hxHi hyLo hyHi
    change
      (if bLo.val > 0 || 0 > bHi.val then
        some
          (roundDownEndpoint
            (min (min (aLo.val / bLo.val) (aLo.val / bHi.val))
              (min (aHi.val / bLo.val) (aHi.val / bHi.val))),
            roundUpEndpoint
              (max (max (aLo.val / bLo.val) (aLo.val / bHi.val))
                (max (aHi.val / bLo.val) (aHi.val / bHi.val))))
      else none) = some (outLo, outHi) at hout
    split at hout
    next hAvoidsZero =>
      have hpair :
          outLo = roundDownEndpoint
            (min (min (aLo.val / bLo.val) (aLo.val / bHi.val))
              (min (aHi.val / bLo.val) (aHi.val / bHi.val))) ∧
          outHi = roundUpEndpoint
            (max (max (aLo.val / bLo.val) (aLo.val / bHi.val))
              (max (aHi.val / bLo.val) (aHi.val / bHi.val))) := by
        simpa using Option.some.inj hout.symm
      rcases hpair with ⟨rfl, rfl⟩
      have hside : bHi.val < 0 ∨ 0 < bLo.val := by
        have hz : 0 < bLo.val ∨ bHi.val < 0 := by simpa using hAvoidsZero
        exact hz.elim Or.inr Or.inl
      have hExact := FloatLib.Floats.Interval.div_bounds_Icc
        aLo.val aHi.val bLo.val bHi.val x y ⟨hxLo, hxHi⟩ ⟨hyLo, hyHi⟩ hside
      change
        (roundDownEndpoint
            (min (min (aLo.val / bLo.val) (aLo.val / bHi.val))
              (min (aHi.val / bLo.val) (aHi.val / bHi.val)))).val ≤ x / y ∧
          x / y ≤
            (roundUpEndpoint
              (max (max (aLo.val / bLo.val) (aLo.val / bHi.val))
                (max (aHi.val / bLo.val) (aHi.val / bHi.val)))).val
      constructor
      · exact (roundDownEndpoint_le _).trans <| by
          simpa only [FloatLib.Floats.Interval.minOfFour] using hExact.1
      · have hUpper :
            x / y ≤ max (max (aLo.val / bLo.val) (aLo.val / bHi.val))
              (max (aHi.val / bLo.val) (aHi.val / bHi.val)) := by
          simpa only [FloatLib.Floats.Interval.maxOfFour] using hExact.2
        exact hUpper.trans (le_roundUpEndpoint _)
    next hIncludesZero => simp at hout
  expBounds_enclosure := by
    change UnaryEnclosure (α := FP32) Real.exp
      (fun lo hi ↦ some
        (roundDownEndpoint (Real.exp lo.val), roundUpEndpoint (Real.exp hi.val)))
    exact roundedUnaryEnclosure_of_monotone Real.exp Real.exp_monotone
  logBounds_enclosure := by
    intro lo hi outLo outHi x hout hxLo hxHi
    change
      (if lo.val > 0 then
        some (roundDownEndpoint (Real.log lo.val), roundUpEndpoint (Real.log hi.val))
      else none) = some (outLo, outHi) at hout
    split at hout
    next hlo =>
      have hpair : outLo = roundDownEndpoint (Real.log lo.val) ∧
          outHi = roundUpEndpoint (Real.log hi.val) := by
        simpa using Option.some.inj hout.symm
      rcases hpair with ⟨rfl, rfl⟩
      change (roundDownEndpoint (Real.log lo.val)).val ≤ Real.log x ∧
        Real.log x ≤ (roundUpEndpoint (Real.log hi.val)).val
      exact
        ⟨(roundDownEndpoint_le _).trans (Real.log_le_log hlo hxLo),
          (Real.log_le_log (hlo.trans_le hxLo) hxHi).trans (le_roundUpEndpoint _)⟩
    next hnlo => simp at hout
  sqrtBounds_enclosure := by
    intro lo hi outLo outHi x hout hxLo hxHi
    change
      (if hi.val < 0 then none else
        some (roundDownEndpoint (Real.sqrt (max lo.val 0)),
          roundUpEndpoint (Real.sqrt hi.val))) = some (outLo, outHi) at hout
    split at hout
    next hhi => simp at hout
    next hnhi =>
      have hpair : outLo = roundDownEndpoint (Real.sqrt (max lo.val 0)) ∧
          outHi = roundUpEndpoint (Real.sqrt hi.val) := by
        simpa using Option.some.inj hout.symm
      rcases hpair with ⟨rfl, rfl⟩
      have hLower : Real.sqrt (max lo.val 0) = Real.sqrt lo.val := by
        by_cases hlo : lo.val ≤ 0
        · simp [max_eq_right hlo, Real.sqrt_eq_zero_of_nonpos hlo]
        · simp [max_eq_left (le_of_not_ge hlo)]
      change (roundDownEndpoint (Real.sqrt (max lo.val 0))).val ≤ Real.sqrt x ∧
        Real.sqrt x ≤ (roundUpEndpoint (Real.sqrt hi.val)).val
      rw [hLower]
      exact
        ⟨(roundDownEndpoint_le _).trans (Real.sqrt_le_sqrt hxLo),
          (Real.sqrt_le_sqrt hxHi).trans (le_roundUpEndpoint _)⟩
  sigmoidBounds_enclosure := by
    change UnaryEnclosure (α := FP32) IntervalLemmas.realSigmoid
      (fun lo hi ↦ some
        (roundDownEndpoint (IntervalLemmas.realSigmoid lo.val),
          roundUpEndpoint (IntervalLemmas.realSigmoid hi.val)))
    exact roundedUnaryEnclosure_of_monotone _ IntervalLemmas.monotone_realSigmoid
  tanhBounds_enclosure := by
    change UnaryEnclosure (α := FP32) Real.tanh
      (fun lo hi ↦ some
        (roundDownEndpoint (Real.tanh lo.val), roundUpEndpoint (Real.tanh hi.val)))
    exact roundedUnaryEnclosure_of_monotone Real.tanh IntervalLemmas.monotone_real_tanh
  sinBounds_enclosure := by
    change UnaryEnclosure (α := FP32) Real.sin
      (fun _ _ ↦ some (roundDownEndpoint (-1), roundUpEndpoint 1))
    exact roundedUnaryEnclosure_of_unit_range Real.sin fun x ↦
      ⟨Real.neg_one_le_sin x, Real.sin_le_one x⟩
  cosBounds_enclosure := by
    change UnaryEnclosure (α := FP32) Real.cos
      (fun _ _ ↦ some (roundDownEndpoint (-1), roundUpEndpoint 1))
    exact roundedUnaryEnclosure_of_unit_range Real.cos fun x ↦
      ⟨Real.neg_one_le_cos x, Real.cos_le_one x⟩
  layerNormAbsBound_sound := by
    intro n radius hout
    change some (roundUpEndpoint (Real.sqrt n)) = some radius at hout
    change Real.sqrt n ≤ radius.val
    rw [← Option.some.inj hout]
    exact le_roundUpEndpoint _
  coupledDerivatives_exact := by simp

/-- Run IBP over `FP32` graph semantics. -/
noncomputable def runIBP (g : Graph) (ps : NN.MLTheory.CROWN.Graph.ParamStore FP32) :
    Array (Option (NN.MLTheory.CROWN.FlatBox FP32)) :=
  NN.MLTheory.CROWN.Graph.runIBP (α := FP32) g ps

/-- Run the scalar-input derivative IBP pass over `FP32` graph semantics. -/
noncomputable def runScalarDerivative (g : Graph) (ps : NN.MLTheory.CROWN.Graph.ParamStore FP32)
    (ibp : Array (Option (NN.MLTheory.CROWN.FlatBox FP32))) :
    Array (Option (NN.MLTheory.CROWN.FlatBox FP32)) :=
  NN.MLTheory.CROWN.Graph.runScalarDerivative (α := FP32) g ps ibp

/-- Run a first-derivative pass from an arbitrary interval-valued direction. -/
noncomputable def runDirectionalDerivative (g : Graph)
    (ps : NN.MLTheory.CROWN.Graph.ParamStore FP32)
    (ibp : Array (Option (NN.MLTheory.CROWN.FlatBox FP32)))
    (seed : NN.MLTheory.CROWN.FlatBox FP32) :
    Array (Option (NN.MLTheory.CROWN.FlatBox FP32)) :=
  NN.MLTheory.CROWN.Graph.runDirectionalDerivative (α := FP32) g ps ibp seed

/-- Run the mixed second-derivative pass `D²f[u, v]` over `FP32` graph semantics. -/
noncomputable def runMixedSecondDerivative (g : Graph)
    (ps : NN.MLTheory.CROWN.Graph.ParamStore FP32)
    (ibp dLeft dRight : Array (Option (NN.MLTheory.CROWN.FlatBox FP32))) :
    Array (Option (NN.MLTheory.CROWN.FlatBox FP32)) :=
  NN.MLTheory.CROWN.Graph.runMixedSecondDerivative (α := FP32) g ps ibp dLeft dRight

/-- Run the second-derivative IBP pass over `FP32` graph semantics. -/
noncomputable def runScalarSecondDerivative (g : Graph)
    (ps : NN.MLTheory.CROWN.Graph.ParamStore FP32)
    (ibp : Array (Option (NN.MLTheory.CROWN.FlatBox FP32)))
    (d1 : Array (Option (NN.MLTheory.CROWN.FlatBox FP32))) :
    Array (Option (NN.MLTheory.CROWN.FlatBox FP32)) :=
  NN.MLTheory.CROWN.Graph.runScalarSecondDerivative (α := FP32) g ps ibp d1

/-- Run the forward affine CROWN pass over `FP32` graph semantics. -/
noncomputable def runAffine (g : Graph) (ps : NN.MLTheory.CROWN.Graph.ParamStore FP32)
    (ctx : NN.MLTheory.CROWN.Graph.AffineCtx)
    (ibp : Array (Option (NN.MLTheory.CROWN.FlatBox FP32))) :
    Array (Option (NN.MLTheory.CROWN.Graph.FlatAffine FP32)) :=
  NN.MLTheory.CROWN.Graph.runAffine (α := FP32) g ps ctx ibp

/-- Run the forward CROWN lower/upper affine-bounds pass over `FP32` graph semantics. -/
noncomputable def runCROWN (g : Graph) (ps : NN.MLTheory.CROWN.Graph.ParamStore FP32)
    (ctx : NN.MLTheory.CROWN.Graph.AffineCtx)
    (ibp : Array (Option (NN.MLTheory.CROWN.FlatBox FP32))) :
    Array (Option (NN.MLTheory.CROWN.Graph.FlatAffineBounds FP32)) :=
  NN.MLTheory.CROWN.Graph.runCROWN (α := FP32) g ps ctx ibp

end FP32

end NN.MLTheory.CROWN
