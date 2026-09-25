/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Generative.Diffusion.Core
public import NN.Spec.Generative.Diffusion.Schedule
public import NN.Spec.Core.TensorOps

/-!
# Image DDIM

The image API stores one cumulative coefficient per noisy state. Its index zero is the first
noisy state, while `VPSchedule` reserves state zero for the clean sample. Reading the coefficient
table directly preserves loaded schedules, including a zero coefficient; no division is needed
to recover per-step beta values.

Image sampling floors the reconstruction denominator, then applies a shape-preserving
postprocessor. The default clips the reconstructed sample to `[-1, 1]`. The separate `ddimStep`
specification retains its additive epsilon and unclipped reconstruction.
-/

@[expose] public section

namespace Generative.Diffusion

open Spec TorchLean

variable {α : Type} [Storage α] [Context α]
variable {T : Nat} {s : Shape}

namespace VPSchedule

/-- The cumulative coefficients of the noisy states, excluding the initial clean coefficient. -/
def noisyAlphaBars (schedule : VPSchedule α T) : Tensor α [T] :=
  Tensor.ofFn fun index => schedule.alphaBar index.succ

/-- Entry `index` describes spec state `index + 1`. -/
@[simp] theorem getScalar_noisyAlphaBars (schedule : VPSchedule α T) (index : Fin T) :
    Tensor.getScalar schedule.noisyAlphaBars index = schedule.alphaBar index.succ := by
  simp [noisyAlphaBars]

end VPSchedule

namespace ImageDDIM

/--
Read a noisy-state coefficient table using the spec's clean-state indexing.

State zero has coefficient one; state `index + 1` reads table entry `index`. This also defines
the sole coefficient of an empty schedule without trying to index its empty table.
-/
def alphaBar (coefficients : Tensor α [T]) (state : Fin (T + 1)) : α :=
  Fin.cases 1 (fun index => Tensor.getScalar coefficients index) state

/-- The clean state precedes every entry of the stored noisy-state table. -/
@[simp] theorem alphaBar_zero (coefficients : Tensor α [T]) :
    alphaBar coefficients 0 = 1 := by
  rfl

/-- Noisy state `index + 1` uses table entry `index`. -/
@[simp] theorem alphaBar_succ (coefficients : Tensor α [T]) (index : Fin T) :
    alphaBar coefficients index.succ = Tensor.getScalar coefficients index := by
  rfl

/-- Dropping and restoring the clean coefficient preserves every VP schedule state. -/
theorem alphaBar_noisyAlphaBars (schedule : VPSchedule α T) (state : Fin (T + 1)) :
    alphaBar schedule.noisyAlphaBars state = schedule.alphaBar state := by
  refine Fin.cases ?_ (fun index => ?_) state
  · rfl
  · simp

/--
Time conditioning used by the image API.

The first noisy state has time zero and the last has time one when there are at least two
states. A single noisy state has time zero. The state index stays explicit, so this convention
never requires recovering an integer index from a rounded scalar time.
-/
def timeOfIndex (index : Fin T) : α :=
  if T <= 1 then 0 else (index.val : α) / ((T - 1 : Nat) : α)

/--
One image DDIM update from an already evaluated epsilon prediction.

First reconstruct with the explicitly branched denominator below, then apply `postprocess`
before remixing with the same epsilon prediction at the previous coefficient. The default
postprocessor clamps to `[-1, 1]`; it never recomputes epsilon. The image API defaults to the
floor `1e-12`. The formula itself makes no positivity or finiteness assumption on the floor.
-/
def stepFromEps (denominatorFloor previousAlpha alpha : α)
    (sample epsilon : Tensor α s)
    (postprocess : Tensor α s → Tensor α s :=
      fun reconstruction => Tensor.clampSpec reconstruction (-1) 1) : Tensor α s :=
  let sqrtAlpha := sqrtNonneg alpha
  let denominator := if sqrtAlpha > denominatorFloor then sqrtAlpha else denominatorFloor
  let reconstruction := Tensor.scaleSpec
    (Tensor.subSpec sample (Tensor.scaleSpec epsilon (sqrtNonneg (1 - alpha))))
    (1 / denominator)
  let processed := postprocess reconstruction
  Tensor.addSpec
    (Tensor.scaleSpec processed (sqrtNonneg previousAlpha))
    (Tensor.scaleSpec epsilon (sqrtNonneg (1 - previousAlpha)))

/-- Evaluate epsilon once at the noisy-state index, then move to the preceding state. -/
def step (denominatorFloor : α) (coefficients : Tensor α [T])
    (predict : Fin T → Tensor α s → Tensor α s) (index : Fin T)
    (sample : Tensor α s)
    (postprocess : Tensor α s → Tensor α s :=
      fun reconstruction => Tensor.clampSpec reconstruction (-1) 1) : Tensor α s :=
  let epsilon := predict index sample
  stepFromEps denominatorFloor
    (alphaBar coefficients (Fin.castSucc index)) (alphaBar coefficients index.succ)
    sample epsilon postprocess

/-- Reverse every noisy state, ending at the clean coefficient one. -/
def sample (denominatorFloor : α) (coefficients : Tensor α [T])
    (predict : Fin T → Tensor α s → Tensor α s) (initial : Tensor α s)
    (postprocess : Tensor α s → Tensor α s :=
      fun reconstruction => Tensor.clampSpec reconstruction (-1) 1) : Tensor α s :=
  (List.finRange T).foldr
    (fun index x => step denominatorFloor coefficients predict index x postprocess)
    initial

end ImageDDIM
end Generative.Diffusion
