/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.Models.Diffusion
public import TorchLean.NN.Models.Diffusion.Schedule
public import TorchLean.Tensor.Elementwise

/-!
# Image DDIM

The image API stores one cumulative coefficient per noisy state. Its index zero is the first
noisy state, while `VPSchedule` reserves state zero for the clean sample. Sampling reads the
stored coefficients as they are, so a loaded schedule keeps every value, a zero coefficient
included, and no per-step beta has to be recovered by division.

Image sampling floors the reconstruction denominator, then applies a shape-preserving
postprocessor. The default clips the reconstructed sample to `[-1, 1]`.

Song, Meng and Ermon (2021) give the step as their eq. (12), and `stepFromEps` is its `σ_t = 0`
case, written by hand. That's the DDIM case of their Section 4.1, so a step adds no fresh noise
and there's no `η` to set. Their `α_t` is DDPM's `ᾱ_t`, and their `α_0 := 1` is our clean state.
The floor and the clamp are ours, and eq. (12) has neither. On the last step the previous
coefficient is `1`, so the step returns the clamped reconstruction. `step_eq` proves that
`predict` runs once per step. No torch test covers the sampler.

References:
- Jiaming Song, Chenlin Meng, and Stefano Ermon (2021), "Denoising Diffusion Implicit Models",
  ICLR 2021, arXiv:2010.02502, eq. (12) and Section 4.1.
-/

@[expose] public section

namespace TorchLean.Diffusion

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

State zero has coefficient one and state `index + 1` reads table entry `index`, which also gives
an empty schedule its single coefficient without indexing the empty table.
-/
def alphaBar (coefficients : Tensor α [T]) (state : Fin (T + 1)) : α :=
  Fin.cases 1 (fun index => Tensor.getScalar coefficients index) state

/-- State zero, the clean state, has coefficient one whatever the stored table holds. -/
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
Time conditioning used by the image API: noisy state `index` has time `index / (T - 1)`,
computed in `Rat` and cast once into the scalar.

The first noisy state has time zero and the last has time one when there are at least two
states, and a single noisy state has time zero. The predictor receives the state index itself, so
no integer index is ever recovered from a rounded scalar time. `VPSchedule.timeOfIndex` puts the
same state at `(index + 1) / T`, so with `T = 1000` the first noisy state sits at `0` here and at
`0.001` there.
-/
def timeOfIndex (index : Fin T) : α :=
  if T <= 1 then 0 else ((index.val / (T - 1 : Nat) : Rat) : α)

/--
One image DDIM update from an epsilon prediction that has already been evaluated.

The update subtracts the scaled prediction from `sample` and divides by $\sqrt{\bar\alpha}$, or by
`denominatorFloor` when the root does not exceed the floor. It then applies `postprocess` and
remixes the result with the same epsilon prediction at the previous coefficient. The default
postprocessor clamps to `[-1, 1]` and never recomputes epsilon. The runnable diffusion example
passes the floor `1e-12`, and the formula makes no positivity or finiteness assumption on the
floor.
-/
def stepFromEps (denominatorFloor previousAlpha alpha : α)
    (sample epsilon : Tensor α s)
    (postprocess : Tensor α s → Tensor α s :=
      fun reconstruction => Tensor.clamp reconstruction (Tensor.full _ (-1)) (Tensor.full _
          1) .interior) : Tensor α s :=
  let sqrtAlpha := sqrtNonneg alpha
  let denominator := if sqrtAlpha > denominatorFloor then sqrtAlpha else denominatorFloor
  let reconstruction := Tensor.scale
    (sample - Tensor.scale epsilon (sqrtNonneg (1 - alpha)))
    (1 / denominator)
  let processed := postprocess reconstruction
  Tensor.scale processed (sqrtNonneg previousAlpha) +
    Tensor.scale epsilon (sqrtNonneg (1 - previousAlpha))

/-- Evaluate epsilon once at the noisy-state index, then move to the preceding state. -/
def step {m : Type → Type} [Monad m] (denominatorFloor : α) (coefficients : Tensor α [T])
    (predict : Fin T → Tensor α s → m (Tensor α s)) (index : Fin T)
    (sample : Tensor α s)
    (postprocess : Tensor α s → Tensor α s :=
      fun reconstruction => Tensor.clamp reconstruction (Tensor.full _ (-1)) (Tensor.full _
          1) .interior) : m (Tensor α s) := do
  let epsilon ← predict index sample
  pure <| stepFromEps denominatorFloor
    (alphaBar coefficients (Fin.castSucc index)) (alphaBar coefficients index.succ)
    sample epsilon postprocess

/-- One update calls `predict` once and passes the same prediction to the reconstruction and to
the remix. The equality is definitional, so it holds in every scalar context and monad. -/
theorem step_eq {m : Type → Type} [Monad m] (denominatorFloor : α) (coefficients : Tensor α [T])
    (predict : Fin T → Tensor α s → m (Tensor α s)) (index : Fin T) (sample : Tensor α s)
    (postprocess : Tensor α s → Tensor α s) :
    step denominatorFloor coefficients predict index sample postprocess =
      predict index sample >>= fun epsilon =>
        pure (stepFromEps denominatorFloor (alphaBar coefficients (Fin.castSucc index))
          (alphaBar coefficients index.succ) sample epsilon postprocess) := by
  rfl

/--
Reverse the noisy states from `start` (by default the last) down to the first, ending at the
clean coefficient one. An empty table returns `initial` without calling `predict`.
-/
def sample {m : Type → Type} [Monad m] (denominatorFloor : α) (coefficients : Tensor α [T])
    (predict : Fin T → Tensor α s → m (Tensor α s)) (initial : Tensor α s)
    (postprocess : Tensor α s → Tensor α s :=
      fun reconstruction => Tensor.clamp reconstruction (Tensor.full _ (-1)) (Tensor.full _
          1) .interior)
    (start : Option (Fin T) := none) : m (Tensor α s) :=
  let reverse (count : Nat) (h : count ≤ T) : m (Tensor α s) :=
    Fin.foldrM count
      (fun index x =>
        step denominatorFloor coefficients predict (Fin.castLE h index) x postprocess)
      initial
  match start with
  | some index => reverse (index.val + 1) index.isLt
  | none => reverse T (Nat.le_refl T)

end ImageDDIM
end TorchLean.Diffusion
