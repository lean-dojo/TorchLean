/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Models.Diffusion
public import NN.Spec.Generative.Diffusion.ImageDDIM
public import NN.Spec.Generative.Diffusion.ForwardProcess

/-!
# Image diffusion API contracts

These equalities connect the public image helpers to their scalar-polymorphic specifications.
The Float tensor operations use specialized kernels, so their coordinate laws supply the
connection to the spec's pointwise operations. No real-field identities or tolerances are used.

The reverse update agrees with `ImageDDIM.stepFromEps`. Agreement with the separate `ddimStep`
formula would require changing clipping, denominator, and time conventions, so it is not the
contract stated here.
-/

@[expose] public section

namespace Generative.Diffusion.ImageDDIM

open Spec TorchLean

variable {T : Nat} {s : Shape}

/-- Public Float addition and the spec addition have the same coordinate values. -/
theorem float_add_eq_addSpec (left right : Tensor Float s) :
    Tensor.add left right = Tensor.addSpec left right := by
  apply TorchLean.Tensor.Internal.Rep.ext
  intro coordinate
  simp only [Tensor.add, Tensor.addSpec, Tensor.map2Spec,
    TorchLean.Tensor.Internal.Rep.add_apply, TorchLean.Tensor.Internal.Rep.zipWith_apply]
  rfl

/-- Public Float subtraction and the spec subtraction have the same coordinate values. -/
theorem float_sub_eq_subSpec (left right : Tensor Float s) :
    Tensor.sub left right = Tensor.subSpec left right := by
  apply TorchLean.Tensor.Internal.Rep.ext
  intro coordinate
  simp only [Tensor.sub, Tensor.subSpec, Tensor.map2Spec,
    TorchLean.Tensor.Internal.Rep.sub_apply, TorchLean.Tensor.Internal.Rep.zipWith_apply]
  rfl

/--
The public reverse update is the image DDIM formula with denominator floor `1e-12`.

The equality preserves clipping before remixing and reuse of the supplied epsilon. It applies
to the actual Float expression, with its original multiplication and addition order.
-/
theorem ddimPrev_eq_stepFromEps (previousAlpha alpha : Float)
    (sample epsilon : Tensor Float s) :
    TorchLean.diffusion.ddimPrev previousAlpha alpha sample epsilon =
      stepFromEps 1e-12 previousAlpha alpha sample epsilon := by
  -- Both sides now select the same scalar operations. The remaining bridge is between
  -- specialized Float tensor addition/subtraction and their coordinate specifications.
  simp only [TorchLean.diffusion.ddimPrev, stepFromEps, sqrtNonneg,
    Tensor.scale, Tensor.clamp, float_sub_eq_subSpec, float_add_eq_addSpec]
  rfl

/-- Cycling a public training step selects noisy state `index + 1` in the spec view. -/
theorem schedule_alphaBar_eq (schedule : TorchLean.diffusion.Schedule T) (step : Nat) :
    schedule.alphaBar step = alphaBar schedule.alphaBars (schedule.index step).succ := by
  rw [alphaBar_succ, Tensor.getScalar_eq_apply]
  rfl

/-- The image time embedding agrees with the public schedule, including its single-state case. -/
theorem schedule_normalizedTime_eq (schedule : TorchLean.diffusion.Schedule T) (step : Nat) :
    schedule.normalizedTime step = timeOfIndex (α := Float) (schedule.index step) := by
  rfl

/--
Matching cumulative coefficients gives the same forward-corrupted sample and image time channel.

The hypothesis compares the actual stored coefficients with the chosen VP schedule. It does not
identify the two linear constructors: their one-step endpoint conventions differ.
-/
theorem noisedSampleFromNoise_input_eq_qSample (batchShape : Shape) {d c : Nat}
    (spatial : Tensor Nat [d]) (schedule : TorchLean.diffusion.Schedule T)
    (specSchedule : VPSchedule Float T)
    (hCoefficients : ∀ index : Fin T,
      schedule.alphaBars[index] = specSchedule.alphaBar index.succ)
    (clean epsilon : Tensor Float (TorchLean.diffusion.sampleShape batchShape c spatial))
    (step : Nat) :
    (TorchLean.diffusion.noisedSampleFromNoise
      batchShape spatial schedule clean epsilon step).input =
      TorchLean.diffusion.appendTimeChannel batchShape spatial
        (qSample specSchedule clean (schedule.index step).succ epsilon)
        (timeOfIndex (α := Float) (schedule.index step)) := by
  have hAlpha : schedule.alphaBar step =
      specSchedule.alphaBar (schedule.index step).succ :=
    hCoefficients (schedule.index step)
  simp only [TorchLean.diffusion.noisedSampleFromNoise, qSample, sqrtNonneg,
    hAlpha, schedule_normalizedTime_eq, Tensor.scale, float_add_eq_addSpec]
  rfl

end Generative.Diffusion.ImageDDIM
