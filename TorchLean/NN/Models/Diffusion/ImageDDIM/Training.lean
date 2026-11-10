/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.Models.Diffusion.NoisePredictor
public import TorchLean.NN.Models.Diffusion.ImageDDIM
public import TorchLean.NN.Models.Diffusion.ForwardProcess

/-!
# Image diffusion API contracts

These equalities connect the public diffusion schedule and training sample to the image
specification's coefficient indexing and forward noising. No real-field identities or tolerances
are used. Runnable code reverses a schedule with `ImageDDIM.sample` itself, so the reverse update
needs no separate contract.

When the stored table matches a `VPSchedule`, `sample_input_eq_qSample` says the training input
for `step` is the `x_t` of eq. (4) of Ho, Jain and Abbeel (2020) at `t = step % T + 1`, with time
channel `(step % T) / (T - 1)` once `T ≥ 2`. Both proofs unfold the definitions.
-/

@[expose] public section

namespace TorchLean.Diffusion.ImageDDIM

variable {T : Nat}

/-- The public coefficient of training step `step` is the spec coefficient of noisy state
`schedule.index step + 1`, where `schedule.index` takes the step modulo the schedule length. -/
theorem schedule_alphaBar_eq (schedule : Schedule T) (step : Nat) :
    schedule.alphaBar step = alphaBar schedule.alphaBars (schedule.index step).succ := by
  rw [alphaBar_succ, Tensor.getScalar_eq_apply]
  rfl

/--
If the stored coefficients equal `specSchedule`'s at every noisy state, the input of the public
training sample is `qSample` at state `schedule.index step + 1` with the image time channel
appended.
-/
theorem sample_input_eq_qSample (batchShape : Shape) {d c : Nat}
    (spatial : Tensor Nat [d]) (schedule : Schedule T)
    (specSchedule : VPSchedule Float T)
    (hCoefficients : ∀ index : Fin T,
      schedule.alphaBars[index] = specSchedule.alphaBar index.succ)
    (clean epsilon : Tensor Float (sampleShape batchShape c spatial))
    (step : Nat) :
    (Diffusion.sample
      batchShape spatial schedule clean step (noise := epsilon)).input =
      appendTimeChannel batchShape spatial
        (qSample specSchedule clean (schedule.index step).succ epsilon)
        (timeOfIndex (α := Float) (schedule.index step)) := by
  have hAlpha : schedule.alphaBar step =
      specSchedule.alphaBar (schedule.index step).succ :=
    hCoefficients (schedule.index step)
  simp only [Diffusion.sample, qSample, hAlpha]

end TorchLean.Diffusion.ImageDDIM
