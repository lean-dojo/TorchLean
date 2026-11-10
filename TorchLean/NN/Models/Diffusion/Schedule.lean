/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Tensor.OfFn
public import TorchLean.Tensor.Context

/-!
# VP diffusion schedules (spec layer)

A discrete-time variance-preserving (VP) schedule for diffusion models, following the common
DDPM-style construction:

- choose $T$ steps and a sequence $\beta_0,\ldots,\beta_{T-1}$ with $0\le\beta_t<1$,
- define $\alpha_t:=1-\beta_t$,
- define the cumulative product $\bar\alpha_0:=1$ and
  $\bar\alpha_{t+1}:=\bar\alpha_t\alpha_t$.

`VPSchedule` stores the coefficients without a range proof, and the inequalities are assumptions
of the probabilistic interpretation. Under them the forward noising kernel is (informally):

$$
x_t=\sqrt{\bar\alpha_t}\,x_0+\sqrt{1-\bar\alpha_t}\,\varepsilon,
\qquad \varepsilon\sim\mathcal{N}(0,I).
$$

The schedule is scalar-polymorphic (`Context α`), so the same definitions serve:

- `Float` / `Float32` (native runtime execution),
- `ExecFloat.Binary 8 23` (executable binary32 semantics),
- `FloatLib.Floats.ExecFloat.Binary` (CPU software arithmetic at a chosen precision),
- `FloatLib.Floats.Formats.Flocq.NF` (noncomputable rounded-real proofs), and
- `ℝ` (mathematical proofs).

We took the schedule from Ho, Jain and Abbeel and shifted the step index down by one. Their steps
run `1..T` and ours run `0..T-1`, so our `β_t` is their `β_{t+1}`, and `alphaBar t` is their
`ᾱ_t` for `1 ≤ t ≤ T`. Their experiments use `T = 1000` with `β` rising linearly from `1e-4` to
`0.02` (Section 4), and `linear 1000` with those two endpoints gives the same ramp, `1e-4` at
index `0` and `0.02` at index `999`.

References:

- Jonathan Ho, Ajay Jain, and Pieter Abbeel (2020), "Denoising Diffusion Probabilistic Models",
  NeurIPS 2020, arXiv:2006.11239, eq. (4) and Section 4.
-/

@[expose] public section

namespace TorchLean.Diffusion

open TorchLean.Tensor

variable {α : Type} [Storage α] [Context α]

/-- Discrete VP schedule with `T` diffusion steps. -/
structure VPSchedule (α : Type) (T : Nat) [Storage α] where
  /-- Per-step variances $\beta_t$ for $t=0,\ldots,T-1$. -/
  betas : Tensor α [T]

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
  let rec
    /-- $\bar\alpha_k$, the product of $\alpha_0, \dots, \alpha_{k-1}$, for `k < T + 1`. -/
    go : (k : Nat) → k < T + 1 → α
      | 0, _ => 1
      | k + 1, hk =>
          -- `k < T` so we can index `α_k`.
          let i : Fin T := ⟨k, Nat.lt_of_succ_lt_succ hk⟩
          have hk' : k < T + 1 := Nat.lt_trans (Nat.lt_succ_self k) hk
          go k hk' * sched.alpha i
  go t.1 t.2

/-- The $T+1$ accumulated coefficients as a vector. -/
def alphaBarTensor (sched : VPSchedule α T) : Tensor α [T + 1] :=
  Tensor.ofFn (fun t => sched.alphaBar t)

/--
Convert a discrete time index `t : Fin (T+1)` into a scalar time $t/T\in[0,1]$ (when $T>0$),
the exact ratio cast once into the scalar (binary backends round it once).

If $T=0$, we define the time as $0$ (the only index is $t=0$).
-/
def timeOfIndex (t : Fin (T + 1)) : α :=
  match T with
  | 0 => 0
  | Nat.succ _ => ((t.1 / T : Rat) : α)

/-!
## Simple constructors

Convenience constructors for examples, kept small and deterministic. Large-scale training
pipelines usually specify their own schedules.
-/

/--
Linear $\beta$ schedule over $T$ steps: $\beta_t$ interpolates from $\beta_{\mathrm{start}}$ to
$\beta_{\mathrm{end}}$. A one-step schedule holds only the start value $\beta_{\mathrm{start}}$.

Cosine schedules, common in the diffusion literature, have no constructor here. The
continuous-time linear schedule is `VPLinearSchedule` in
`TorchLean.NN.Models.Diffusion.PFODE`.
-/
def linearBetas (T : Nat) (β_start β_end : α) : Tensor α [T] :=
  match T with
  | 0 => Tensor.full [0] β_end
  | Nat.succ T' =>
      match T' with
      | 0 =>
          -- `T = 1`: the ramp's first entry.
          Tensor.full [1] β_start
      | Nat.succ T'' =>
          -- `T = T'' + 2`: interpolate using denominator `(T-1) = T'' + 1`.
          -- The last beta equals `β_end` in exact real arithmetic, and rounding can change it.
          Tensor.ofFn (fun i : Fin (Nat.succ (Nat.succ T'')) =>
            let denom : α := (Nat.succ T'' : α) -- = T - 1
            let frac : α := (i.1 : α) / denom
            β_start + frac * (β_end - β_start))

/-- Build a schedule from a linear beta ramp. -/
def linear (T : Nat) (β_start β_end : α) : VPSchedule α T :=
  { betas := linearBetas (α := α) T β_start β_end }

end VPSchedule

end TorchLean.Diffusion
