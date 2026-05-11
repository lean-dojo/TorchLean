/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Core.Tensor.Constructors
public import NN.Spec.Core.Tensor.Vec
public import NN.Spec.Generative.Diffusion.Core

/-!
# VP diffusion schedules (spec layer)

This file defines a discrete-time variance-preserving (VP) schedule for diffusion models.

We follow the common DDPM-style discrete schedule:

- choose `T` steps and a sequence of `β₀, …, β_{T-1}` with `0 ≤ β_t < 1`,
- define `α_t := 1 - β_t`,
- define the cumulative product `ᾱ_0 := 1` and `ᾱ_{t+1} := ᾱ_t * α_t`.

Then the forward noising kernel is (informally):

`x_t = sqrt(ᾱ_t) x_0 + sqrt(1-ᾱ_t) ε` where `ε ~ N(0, I)`.

We keep the schedule scalar-polymorphic (`Context α`) so the same definitions can be reused under:

- `Float` (fast runtime execution),
- `IEEE32Exec` / `NeuralFloat` (proof-relevant floating-point models),
- interval-like scalars (verification), and
- `ℝ` (mathematical proofs).

References (informal pointers):

- Ho, Jain, Abbeel (2020), "Denoising Diffusion Probabilistic Models" (DDPM).
-/

@[expose] public section

namespace Generative.Diffusion

open Spec
open Tensor

variable {α : Type} [Context α]

/-- Discrete VP schedule with `T` diffusion steps. -/
structure VPSchedule (α : Type) (T : Nat) [Context α] where
  /-- Per-step variances `β_t` for `t = 0..T-1`. -/
  betas : Spec.Vec T α

namespace VPSchedule

variable {T : Nat}

/-- Fetch `β_t` as a scalar. -/
def beta (sched : VPSchedule α T) (t : Fin T) : α :=
  Tensor.vecGet sched.betas t

/-- `α_t := 1 - β_t`. -/
def alpha (sched : VPSchedule α T) (t : Fin T) : α :=
  1 - sched.beta t

/--
Compute `ᾱ_t` for `t : Fin (T+1)` with the convention:

- `ᾱ_0 = 1`,
- `ᾱ_{t+1} = ᾱ_t * α_t`.

Implementation note: we define an auxiliary recursion on `Nat` and then package it as a `Fin`
function; this keeps the definition total and easy to evaluate.
-/
def alphaBar (sched : VPSchedule α T) (t : Fin (T + 1)) : α :=
  let rec go : (k : Nat) → k < T + 1 → α
    | 0, _ => 1
    | k + 1, hk =>
        -- `k < T` so we can index `α_k`.
        let i : Fin T := ⟨k, Nat.lt_of_succ_lt_succ hk⟩
        have hk' : k < T + 1 := Nat.lt_trans (Nat.lt_succ_self k) hk
        go k hk' * sched.alpha i
  go t.1 t.2

/-- Vector form of `alphaBar` (length `T+1`). -/
def alphaBarVec (sched : VPSchedule α T) : Spec.Vec (T + 1) α :=
  Spec.vectorTensor (fun t => sched.alphaBar t)

/--
Convert a discrete time index `t : Fin (T+1)` into a scalar time `t/T ∈ [0,1]` (when `T > 0`).

If `T = 0`, we define the time as `0` (the only index is `t = 0`).
-/
def timeOfIndex (t : Fin (T + 1)) : α :=
  match T with
  | 0 => 0
  | Nat.succ _ => (t.1 : α) / (T : α)

/-!
## Simple constructors

These are convenience constructors for examples and demos.
We intentionally keep them small and deterministic; large-scale training pipelines usually want
explicit control over schedules.
-/

/--
Linear `β` schedule over `T` steps: `β_t` interpolates from `β_start` to `β_end`.

Note: this is a small spec helper. Popular schedules in the diffusion literature often use
variants such as cosine schedules or continuous VP schedules; add those as separate named specs
when a model or theorem needs them.
-/
def linearBetas (T : Nat) (β_start β_end : α) : Spec.Vec T α :=
  match T with
  | 0 => Spec.vectorTensor (fun i : Fin 0 => (False.elim (by simpa using i.2)))
  | Nat.succ T' =>
      match T' with
      | 0 =>
          -- `T = 1`: by convention, return the endpoint.
          Spec.vectorTensor (fun _i : Fin 1 => β_end)
      | Nat.succ T'' =>
          -- `T = T'' + 2`: interpolate using denominator `(T-1) = T'' + 1`, so the last beta is
          -- exactly `β_end`.
          Spec.vectorTensor (fun i : Fin (Nat.succ (Nat.succ T'')) =>
            let denom : α := (Nat.succ T'' : α) -- = T - 1
            let frac : α := (i.1 : α) / denom
            β_start + frac * (β_end - β_start))

/-- Build a schedule from a linear beta ramp. -/
def linear (T : Nat) (β_start β_end : α) : VPSchedule α T :=
  { betas := linearBetas (α := α) T β_start β_end }

end VPSchedule

end Generative.Diffusion
