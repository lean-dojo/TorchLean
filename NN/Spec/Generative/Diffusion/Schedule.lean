/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Core.Tensor.Constructors
public import NN.Spec.Core.Context

/-!
# VP diffusion schedules (spec layer)

This file defines a discrete-time variance-preserving (VP) schedule for diffusion models.

We follow the common DDPM-style discrete schedule:

- choose $T$ steps and a sequence $\beta_0,\ldots,\beta_{T-1}$ with $0\le\beta_t<1$,
- define $\alpha_t:=1-\beta_t$,
- define the cumulative product $\bar\alpha_0:=1$ and
  $\bar\alpha_{t+1}:=\bar\alpha_t\alpha_t$.

Then the forward noising kernel is (informally):

$$
x_t=\sqrt{\bar\alpha_t}\,x_0+\sqrt{1-\bar\alpha_t}\,\varepsilon,
\qquad \varepsilon\sim\mathcal{N}(0,I).
$$

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

open Spec TorchLean
open TorchLean TorchLean.Tensor

variable {α : Type} [TorchLean.Storage α] [Context α]

/-- Discrete VP schedule with `T` diffusion steps. -/
structure VPSchedule (α : Type) (T : Nat) [TorchLean.Storage α] [Context α] where
  /-- Per-step variances $\beta_t$ for $t=0,\ldots,T-1$. -/
  betas : TorchLean.Tensor α [T]

namespace VPSchedule

variable {T : Nat}

/-- Fetch $\beta_t$ as a scalar. -/
def beta (sched : VPSchedule α T) (t : Fin T) : α :=
  Tensor.getScalar sched.betas t

/-- $\alpha_t:=1-\beta_t$. -/
def alpha (sched : VPSchedule α T) (t : Fin T) : α :=
  1 - sched.beta t

/--
Compute $\bar\alpha_t$ for `t : Fin (T+1)` with the convention:

- $\bar\alpha_0=1$,
- $\bar\alpha_{t+1}=\bar\alpha_t\alpha_t$.

Implementation note: define an auxiliary recursion on `Nat`, then package it as a `Fin` function.
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

/-- The $T+1$ accumulated coefficients as a vector. -/
def alphaBarTensor (sched : VPSchedule α T) : TorchLean.Tensor α [T + 1] :=
  TorchLean.Tensor.ofFn (fun t => sched.alphaBar t)

/--
Convert a discrete time index `t : Fin (T+1)` into a scalar time $t/T\in[0,1]$ (when $T>0$).

If $T=0$, we define the time as $0$ (the only index is $t=0$).
-/
def timeOfIndex (t : Fin (T + 1)) : α :=
  match T with
  | 0 => 0
  | Nat.succ _ => (t.1 : α) / (T : α)

/-!
## Simple constructors

These are convenience constructors for examples and examples.
We intentionally keep them small and deterministic; large-scale training pipelines usually want
explicit control over schedules.
-/

/--
Linear $\beta$ schedule over $T$ steps: $\beta_t$ interpolates from $\beta_{\mathrm{start}}$ to
$\beta_{\mathrm{end}}$.

Note: this is a small spec helper. Popular schedules in the diffusion literature often use
variants such as cosine schedules or continuous VP schedules; add those as separate named specs
when a model or theorem needs them.
-/
def linearBetas (T : Nat) (β_start β_end : α) : TorchLean.Tensor α [T] :=
  match T with
  | 0 => TorchLean.Tensor.ofFn (fun i : Fin 0 => (False.elim (by simpa using i.2)))
  | Nat.succ T' =>
      match T' with
      | 0 =>
          -- `T = 1`: by convention, return the endpoint.
          TorchLean.Tensor.ofFn (fun _i : Fin 1 => β_end)
      | Nat.succ T'' =>
          -- `T = T'' + 2`: interpolate using denominator `(T-1) = T'' + 1`, so the last beta is
          -- exactly `β_end`.
          TorchLean.Tensor.ofFn (fun i : Fin (Nat.succ (Nat.succ T'')) =>
            let denom : α := (Nat.succ T'' : α) -- = T - 1
            let frac : α := (i.1 : α) / denom
            β_start + frac * (β_end - β_start))

/-- Build a schedule from a linear beta ramp. -/
def linear (T : Nat) (β_start β_end : α) : VPSchedule α T :=
  { betas := linearBetas (α := α) T β_start β_end }

end VPSchedule

end Generative.Diffusion
