/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Floats.NeuralFloat.Rounding.Core
public import NN.Floats.NeuralFloat.Rounding.Predicates
public import NN.Floats.NeuralFloat.Rounding.Order
public import NN.Floats.NeuralFloat.Format.Formats

/-!
# Double Rounding

Directed rounding through a finer intermediate format gives the same answer as direct rounding to
the coarser format.  The order-theoretic proof only needs inclusion of representable values, so it
applies beyond the standard FIX, FLX, and FLT families.
-/

@[expose] public section

namespace TorchLean.Floats

variable {β : NeuralRadix}

/-- Downward rounding through a containing format collapses to direct downward rounding. -/
theorem neuralRoundDownPoint_double {fine coarse : ℝ → Prop} {x fineValue coarseValue : ℝ}
    (hsubset : ∀ z, coarse z → fine z)
    (hf : NeuralRoundDownPoint fine x fineValue)
    (hc : NeuralRoundDownPoint coarse x coarseValue) :
    NeuralRoundDownPoint coarse fineValue coarseValue := by
  have hcoarseFine : coarseValue ≤ fineValue :=
    hf.2.2 coarseValue (hsubset coarseValue hc.1) hc.2.1
  refine ⟨hc.1, hcoarseFine, ?_⟩
  intro z hz hzFine
  exact hc.2.2 z hz (hzFine.trans hf.2.1)

/-- Upward rounding through a containing format collapses to direct upward rounding. -/
theorem neuralRoundUpPoint_double {fine coarse : ℝ → Prop} {x fineValue coarseValue : ℝ}
    (hsubset : ∀ z, coarse z → fine z)
    (hf : NeuralRoundUpPoint fine x fineValue)
    (hc : NeuralRoundUpPoint coarse x coarseValue) :
    NeuralRoundUpPoint coarse fineValue coarseValue := by
  have hFineCoarse : fineValue ≤ coarseValue :=
    hf.2.2 coarseValue (hsubset coarseValue hc.1) hc.2.1
  refine ⟨hc.1, hFineCoarse, ?_⟩
  intro z hz hFineZ
  exact hc.2.2 z hz (hf.2.1.trans hFineZ)

/-- Increasing FLX precision preserves every exactly representable value. -/
theorem neural_generic_format_FLX_mono {coarsePrec finePrec : ℤ}
    (hcoarse : 0 < coarsePrec) (hfine : 0 < finePrec) (hprec : coarsePrec ≤ finePrec)
    {x : ℝ}
    (hx : @neuralGenericFormat β (FLXExp coarsePrec)
      (flxValidExp coarsePrec hcoarse) x) :
    @neuralGenericFormat β (FLXExp finePrec) (flxValidExp finePrec hfine) x := by
  let _ : NeuralValidExp (FLXExp coarsePrec) := flxValidExp coarsePrec hcoarse
  let _ : NeuralValidExp (FLXExp finePrec) := flxValidExp finePrec hfine
  apply neural_generic_inclusion (fexp₁ := FLXExp coarsePrec) (fexp₂ := FLXExp finePrec)
  · intro e
    simp [FLXExp]
    linarith
  · exact hx

/-- Downward FLX double rounding equals direct rounding to the coarser precision. -/
theorem neuralRound_floor_double_FLX {coarsePrec finePrec : ℤ}
    (hcoarse : 0 < coarsePrec) (hfine : 0 < finePrec) (hprec : coarsePrec ≤ finePrec)
    (x : ℝ) :
    @neuralRound β (FLXExp coarsePrec) (flxValidExp coarsePrec hcoarse)
        neuralFloorRound
        (@neuralRound β (FLXExp finePrec) (flxValidExp finePrec hfine)
          neuralFloorRound x) =
      @neuralRound β (FLXExp coarsePrec) (flxValidExp coarsePrec hcoarse)
        neuralFloorRound x := by
  let _ : NeuralValidExp (FLXExp coarsePrec) := flxValidExp coarsePrec hcoarse
  let _ : NeuralValidExp (FLXExp finePrec) := flxValidExp finePrec hfine
  let fineValue := @neuralRound β (FLXExp finePrec) (flxValidExp finePrec hfine)
    neuralFloorRound x
  let coarseValue := @neuralRound β (FLXExp coarsePrec) (flxValidExp coarsePrec hcoarse)
    neuralFloorRound x
  have hf : NeuralRoundDownPoint
      (@neuralGenericFormat β (FLXExp finePrec) (flxValidExp finePrec hfine)) x fineValue :=
    neuralRound_floor_point x
  have hc : NeuralRoundDownPoint
      (@neuralGenericFormat β (FLXExp coarsePrec) (flxValidExp coarsePrec hcoarse)) x coarseValue :=
    neuralRound_floor_point x
  have hdouble := neuralRoundDownPoint_double
    (fun _ h => neural_generic_format_FLX_mono hcoarse hfine hprec h) hf hc
  exact neuralRoundDownPoint_unique
    (neuralRound_floor_point (β := β) (fexp := FLXExp coarsePrec) fineValue) hdouble

/-- Upward FLX double rounding equals direct rounding to the coarser precision. -/
theorem neuralRound_ceil_double_FLX {coarsePrec finePrec : ℤ}
    (hcoarse : 0 < coarsePrec) (hfine : 0 < finePrec) (hprec : coarsePrec ≤ finePrec)
    (x : ℝ) :
    @neuralRound β (FLXExp coarsePrec) (flxValidExp coarsePrec hcoarse)
        neuralCeilRound
        (@neuralRound β (FLXExp finePrec) (flxValidExp finePrec hfine)
          neuralCeilRound x) =
      @neuralRound β (FLXExp coarsePrec) (flxValidExp coarsePrec hcoarse)
        neuralCeilRound x := by
  let _ : NeuralValidExp (FLXExp coarsePrec) := flxValidExp coarsePrec hcoarse
  let _ : NeuralValidExp (FLXExp finePrec) := flxValidExp finePrec hfine
  let fineValue := @neuralRound β (FLXExp finePrec) (flxValidExp finePrec hfine)
    neuralCeilRound x
  let coarseValue := @neuralRound β (FLXExp coarsePrec) (flxValidExp coarsePrec hcoarse)
    neuralCeilRound x
  have hf : NeuralRoundUpPoint
      (@neuralGenericFormat β (FLXExp finePrec) (flxValidExp finePrec hfine)) x fineValue :=
    neuralRound_ceil_point x
  have hc : NeuralRoundUpPoint
      (@neuralGenericFormat β (FLXExp coarsePrec) (flxValidExp coarsePrec hcoarse)) x coarseValue :=
    neuralRound_ceil_point x
  have hdouble := neuralRoundUpPoint_double
    (fun _ h => neural_generic_format_FLX_mono hcoarse hfine hprec h) hf hc
  exact neuralRoundUpPoint_unique
    (neuralRound_ceil_point (β := β) (fexp := FLXExp coarsePrec) fineValue) hdouble

end TorchLean.Floats
