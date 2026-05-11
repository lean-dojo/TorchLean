/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Core.TensorReductionShape
-- We need `Array.qsort`'s definitional unfolding (module-system hides non-`@[expose]` bodies by default).
import all Init.Data.Array.QSort.Basic

/-!
# Tensor gradient utilities (spec layer)

These are small, generic helpers that operate on gradient tensors:

- norm-based clipping (`clip_gradients_spec`)
- value-based clipping (`clip_by_value_spec`)
- percentile-based clipping (`clip_by_percentile_spec`)

They are defined at the spec layer so they can be used both:

- in executable training examples (instantiated at `Float` / NF backends), and
- in proofs (instantiated at `ℝ`).

Why clipping utilities belong in the spec layer:

- Gradient clipping is part of the *algorithmic definition* of many training loops, not just an
  implementation detail. If we want to reason about "the training step we ran", we need clipping
  to be part of the pure model of that step.
- We also want to reuse the same clipping logic across scalar backends: `Float` for executable runs,
  and proof-friendly scalars (`ℝ`, `NF`, etc.) for theorems and approximation statements.

Design note:
- These definitions are written for clarity and reuse across scalar backends. Backend-specific
  implementations (for example, fused kernels) belong in the runtime layer.
-/

@[expose] public section


namespace Spec
open Tensor

variable {α : Type} [Context α] [DecidableRel ((· > ·) : α → α → Prop)]

-- Gradient clipping functions

/--
Clip gradients by L2 norm (global norm over all elements).

This implements the common "global norm clipping" used in many optimizers:

1. compute `||g||_2`
2. if it exceeds `max_norm`, rescale `g` so that `||g||_2 = max_norm`.

Implementation detail:
- We compare squared norms first so we only compute `sqrt` in the clipping branch.
- We treat `max_norm` as a magnitude, so we use `abs max_norm` as the threshold.
-/
def clipGradientsSpec {s : Shape} (gradients : Tensor α s) (max_norm : α) : Tensor α s :=
  let sumsq := sumSpec (squareSpec gradients)
  let max_norm' := MathFunctions.abs max_norm
  let maxsq := max_norm' * max_norm'
  if sumsq > maxsq then
    -- If `sumsq > 0` then `sqrt sumsq` is safe as a divisor in the usual numeric backends.
    -- (If `max_norm' = 0`, this scales everything to 0.)
    let norm := MathFunctions.sqrt sumsq
    scaleSpec gradients (max_norm' / norm)
  else
    gradients

/--
Clip gradients by value (elementwise clamp).

PyTorch analogy: `torch.clamp(g, min=min_val, max=max_val)`.
-/
def clipByValueSpec {s : Shape} (gradients : Tensor α s) (min_val max_val : α) : Tensor α s :=
  clampSpec gradients min_val max_val

/--
Clip gradients by percentile of absolute values.

This is a *value* clipping rule driven by the data:
- Flatten `abs(g)` to an array.
- Take the `pct` percentile (0..100) as a bound `b`.
- Return `clamp(g, -b, b)`.

Notes:
- This definition sorts values, so it requires *decidable* comparison (`DecidableLT α`).
- In practice this is meant for executable scalars like `Float` or `IEEE32Exec`.

PyTorch analogy (conceptual): compute `b = quantile(abs(g), pct/100)` and clamp to `[-b, b]`.
-/
def clipByPercentileSpec {s : Shape} (gradients : Tensor α s) (pct : Nat)
    [DecidableLT α] : Tensor α s :=
  let pct' := min pct 100
  let absVals : Array α :=
    tensorFoldlSpec (β := Array α) (fun acc x => acc.push (MathFunctions.abs x)) #[] gradients
  if _h0 : absVals.size = 0 then
    gradients
 else
    let sorted := absVals.qsort (fun a b => decide (a < b))
    let len := sorted.size
    let idx := (pct' * (len - 1)) / 100
    -- `idx < len` since `pct' ≤ 100` and `len > 0` (from `h0`).
    have hpct : pct' ≤ 100 := by
      simp [pct']
    have hlen_pos : 0 < len := by
      have hqsort_size : sorted.size = absVals.size := by
        -- `Array.qsort` is in-place sorting, so it preserves array size.
        simpa [sorted] using (by
          unfold Array.qsort
          by_cases h : absVals.size = 0 <;> simp [h])
      have hlen_eq : len = absVals.size := by
        simpa [len] using hqsort_size
      have hlen_ne : len ≠ 0 := by
        intro hlen0
        apply _h0
        exact (hlen_eq.symm ▸ hlen0)
      exact Nat.pos_of_ne_zero hlen_ne
    have hidx : idx < len := by
      have hmul : pct' * (len - 1) ≤ 100 * (len - 1) :=
        Nat.mul_le_mul_right (len - 1) hpct
      have hdiv :
          (pct' * (len - 1)) / 100 ≤ (100 * (len - 1)) / 100 :=
        Nat.div_le_div_right (c := 100) hmul
      have hidx_le : idx ≤ (100 * (len - 1)) / 100 := by
        simpa [idx] using hdiv
      have hdiv100 : (100 * (len - 1)) / 100 = len - 1 := by
        calc
          (100 * (len - 1)) / 100 = ((len - 1) * 100) / 100 := by
            simp [Nat.mul_comm]
          _ = len - 1 := by
            exact Nat.mul_div_left (len - 1) (n := 100) (Nat.succ_pos 99)
      have hidx_le' : idx ≤ len - 1 := by
        simpa [hdiv100] using hidx_le
      have hlenm1_lt : len - 1 < len := Nat.sub_lt hlen_pos (by decide : 0 < (1 : Nat))
      exact lt_of_le_of_lt hidx_le' hlenm1_lt
    have hidx_sorted : idx < sorted.size := by
      simpa [len] using hidx
    let bound := sorted[idx]'hidx_sorted
    clampSpec gradients (-bound) bound

end Spec
