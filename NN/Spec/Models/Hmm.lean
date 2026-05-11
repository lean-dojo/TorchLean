/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Core.TensorReductionShape

/-!
# Hidden Markov Model (HMM) (spec model)

This file defines an HMM with discrete observations:

- hidden states: `nStates`
- observations: `nObservations` (discrete symbols)

The model parameters are:
- initial distribution `π`
- transition matrix `A`
- emission matrix `B`

We represent observations as `List (Fin nObservations)` to keep the observation alphabet explicit
and avoid mixing “probabilities” with “indices” in the scalar type `α`.

## Notation and shapes

We use the conventional HMM notation:

- `π : nStates` initial state distribution
- `A : nStates × nStates` transition matrix (`A[i,j] = P(z_{t+1}=j | z_t=i)`)
- `B : nStates × nObservations` emission matrix (`B[i,o] = P(x_t=o | z_t=i)`)

An observation sequence is `o₀, o₁, ..., o_{T-1}` where each `o_t : Fin nObservations`.

References:

- Rabiner (1989),
  "A Tutorial on Hidden Markov Models and Selected Applications in Speech Recognition":
  https://ieeexplore.ieee.org/document/18626
- Baum and Petrie (1966),
  "Statistical Inference for Probabilistic Functions of Finite State Markov Chains":
  https://projecteuclid.org/journals/annals-of-mathematical-statistics/volume-37/issue-6/Statistical
    -Inference-for-Probabilistic-Functions-of-Finite-State-Markov-Chains/10.1214/aoms/1177699147.ful
    l

PyTorch analogy:

- emissions are categorical distributions (`torch.distributions.Categorical`),
- the forward algorithm corresponds to multiplying by `A` and reweighting by `B[:, obs_t]`,
  then summing over previous states (often implemented in log-space in practice).

In practice, PyTorch users often reach for a dedicated HMM library (e.g. `hmmlearn`) or implement
HMMs in log-space with `logsumexp`; TorchLean keeps the spec in a simple, explicit form that is
good for reading and proofs.
-/

public section


namespace Spec

open Tensor

variable {α : Type} [Context α]

/-- A discrete-observation HMM.

We do not enforce probabilistic validity (nonnegativity / rows summing to `1`) at the type level;
that is a modeling assumption, similar to how PyTorch will happily store unconstrained tensors
until you feed them to a distribution or a loss.
-/
structure HMMSpec (α : Type) (nStates nObservations : Nat) where
  /-- Initial distribution `π`. -/
  init_prob : Tensor α (.dim nStates .scalar)
  /-- Transition matrix `A`. -/
  trans_prob : Tensor α (.dim nStates (.dim nStates .scalar))
  /-- Emission matrix `B`. -/
  emission_prob : Tensor α (.dim nStates (.dim nObservations .scalar))

/-- Observation sequence as a list of discrete symbols (indices into the observation alphabet). -/
abbrev ObservationSeq (nObservations : Nat) := List (Fin nObservations)

/-! ## Basic helpers -/

/-- Get emission probability `B[state, obs]` for a discrete observation symbol. -/
def getEmissionProbDiscrete
  {nStates nObservations : Nat}
  (m : HMMSpec α nStates nObservations)
  (state : Fin nStates)
  (obs : Fin nObservations) : α :=
  match get m.emission_prob state with
  | Tensor.dim emit_vals =>
    match emit_vals obs with
    | Tensor.scalar prob => prob

/-!
## Baum–Welch (EM) training

The forward-pass APIs above are enough to *use* a fixed HMM, but a “fully implemented” baseline
should also include classical training. For discrete-observation HMMs, the standard training
procedure is the Baum–Welch algorithm (an EM procedure):

- **E-step**: run forward–backward to compute expected state occupancies (`γ`) and expected
  transition counts (`ξ`).
- **M-step**: normalize those expected counts to update `π`, `A`, and `B`.

This implementation uses *scaled* forward–backward to reduce numerical underflow:
each forward message `α_t` is normalized by a scalar `c_t`, and the backward messages divide by
those same scalars. The sequence likelihood is then `∏_t c_t`, so the log-likelihood is
`Σ_t log c_t`.

Concretely:

- forward recursion (unnormalized): `α̃_{t+1}(j) = B[j, o_{t+1}] * Σ_i α_t(i) * A[i,j]`
- scaling: `c_t = Σ_j α̃_t(j)` and `α_t = α̃_t / c_t` so that `Σ_j α_t(j) = 1`

This is the same basic idea used in many practical HMM implementations (sometimes also expressed as
log-space forward–backward).

This is deterministic and written for clarity; it is not intended to be a high-performance HMM
trainer.
-/

/--
Uniform distribution vector of length `n`.

This is used as a safe fallback when a normalization sum is `0`.
-/
private def uniformVec {n : Nat} : Tensor α (.dim n .scalar) :=
  match n with
  | 0 => Tensor.dim (fun k => nomatch k)
  | Nat.succ _ => Tensor.dim (fun _ => Tensor.scalar (1 / (n : α)))

/-- Normalize a nonnegative vector `v` to sum to `1`, returning `(v / sum(v), sum(v))`.

If the sum is `0`, we fall back to a uniform distribution. This keeps the forward pass total and
avoids propagating `NaN`/division-by-zero behavior into later computations.
-/
def normalizeVec {n : Nat} (v : Tensor α (.dim n .scalar)) : (Tensor α (.dim n .scalar) × α) :=
  let s := sumSpec v
  if s > 0 then
    (scaleSpec v (1 / s), s)
  else
    (uniformVec (α := α) (n := n), 1)

/-- Emission probabilities `B[:, obs]` as a vector over states. -/
def emissionVec {nStates nObservations : Nat}
  (m : HMMSpec α nStates nObservations) (obs : Fin nObservations) : Tensor α (.dim nStates .scalar)
    :=
  Tensor.dim (fun s => Tensor.scalar (getEmissionProbDiscrete m s obs))

/--
One forward step (unnormalized) of the scaled forward algorithm.

Given the previous normalized forward message `prev_alpha` and the next observation `obs`, compute
the next unnormalized message `alphaTilde`.
-/
private def forwardStep {nStates nObservations : Nat}
  (m : HMMSpec α nStates nObservations)
  (prev_alpha : Tensor α (.dim nStates .scalar))
  (obs : Fin nObservations) : Tensor α (.dim nStates .scalar) :=
  -- One forward update (without scaling): apply transitions, then reweight by emissions.
  let emission_probs := emissionVec (α := α) m obs
  Tensor.dim (fun s =>
    let trans_sum := Tensor.dim (fun s' =>
      match get prev_alpha s', get m.trans_prob s' with
      | Tensor.scalar alpha_val, Tensor.dim trans_vals =>
        match trans_vals s with
        | Tensor.scalar trans_val => Tensor.scalar (alpha_val * trans_val))
    let trans_total := sumSpec trans_sum
    match get emission_probs s with
    | Tensor.scalar emit_val => Tensor.scalar (emit_val * trans_total))

/-- Scaled forward pass, returning `(α_t, c_t)` for each timestep.

- Each `α_t` is normalized to sum to `1`.
- Each `c_t` is the normalization constant used at step `t`.

If you need the total likelihood, multiply the scales: `p(o₀:T-1) = ∏_t c_t`.
-/
def hmmForwardScaled
  {nStates nObservations : Nat} [Inhabited (Fin nObservations)]
  (m : HMMSpec α nStates nObservations)
  (observations : ObservationSeq nObservations) :
  (List (Tensor α (.dim nStates .scalar)) × List α) :=
  match observations with
  | [] => ([], [])
  | o0 :: os =>
      let alpha0_raw := mulSpec m.init_prob (emissionVec (α := α) m o0)
      let (alpha0, c0) := normalizeVec (α := α) alpha0_raw
      let rec go (prev : Tensor α (.dim nStates .scalar))
        (rest : ObservationSeq nObservations)
        (accA : List (Tensor α (.dim nStates .scalar)))
        (accC : List α) :
        (List (Tensor α (.dim nStates .scalar)) × List α) :=
        match rest with
        | [] => (accA.reverse, accC.reverse)
        | o :: os =>
            let alpha_raw := forwardStep (α := α) m prev o
            let (alpha, c) := normalizeVec (α := α) alpha_raw
            go alpha os (alpha :: accA) (c :: accC)
      go alpha0 os [alpha0] [c0]

/-- Scaled backward pass, producing normalized backward messages `β_t`.

The standard backward recursion is:

`β_t(i) = Σ_j A[i,j] * B[j, o_{t+1}] * β_{t+1}(j)`.

In the scaled variant, we divide by the forward scale `c_{t+1}` so that `α_t ⊙ β_t` stays
well-conditioned.
-/
private def hmmBackwardScaled
  {nStates nObservations : Nat} [Inhabited (Fin nObservations)]
  (m : HMMSpec α nStates nObservations)
  (observations : ObservationSeq nObservations)
  (scales : List α) :
  List (Tensor α (.dim nStates .scalar)) :=
  let T := observations.length
  if hT : T = 0 then
    []
  else
    let betaLast : Tensor α (.dim nStates .scalar) := fill 1 (.dim nStates .scalar)
    let rec step (t : Nat)
      (betaNext : Tensor α (.dim nStates .scalar))
      (acc : List (Tensor α (.dim nStates .scalar))) :
      List (Tensor α (.dim nStates .scalar)) :=
      if ht : t > 0 then
        let time_idx := t - 1
        let obsNext := observations.getD t default
        let cNext := scales.getD t 1
        let emitNext := emissionVec (α := α) m obsNext
        let betaRaw : Tensor α (.dim nStates .scalar) :=
          Tensor.dim (fun i =>
            let sumOverJ : Tensor α (.dim nStates .scalar) :=
              Tensor.dim (fun j =>
                match get (get m.trans_prob i) j, get emitNext j, get betaNext j with
                | Tensor.scalar aij, Tensor.scalar bj, Tensor.scalar bnext =>
                    Tensor.scalar (aij * bj * bnext))
            Tensor.scalar (sumSpec sumOverJ))
        let beta :=
          if cNext > 0 then
            scaleSpec betaRaw (1 / cNext)
          else
            betaRaw
        step time_idx beta (beta :: acc)
      else
        acc
    -- start from t = T-1, with acc containing beta_{T-1}
    step (T - 1) betaLast [betaLast]

 /-- Elementwise multiplication for state-probability vectors. -/
private def elementwiseMul {n : Nat} (a b : Tensor α (.dim n .scalar)) : Tensor α (.dim n .scalar)
  :=
  mulSpec a b

 /--
Compute the normalized state posterior `gamma_t` from forward/backward messages.

`gamma_t(i) ∝ alpha_t(i) * beta_t(i)`.
 -/
private def gammaAt {nStates : Nat}
  (alpha : Tensor α (.dim nStates .scalar)) (beta : Tensor α (.dim nStates .scalar)) : Tensor α
    (.dim nStates .scalar) :=
  -- γ_t(i) ∝ α_t(i) * β_t(i)
  let g := elementwiseMul (α := α) alpha beta
  (normalizeVec (α := α) g).1

 /--
Compute the normalized transition posterior `xi_t` for a single time step.

Unnormalized:
`xi(i,j) = alpha_t(i) * A(i,j) * B(j, obs_{t+1}) * beta_{t+1}(j)`.
 -/
private def xiAt {nStates nObservations : Nat}
  (m : HMMSpec α nStates nObservations)
  (alpha_t : Tensor α (.dim nStates .scalar))
  (beta_next : Tensor α (.dim nStates .scalar))
  (obs_next : Fin nObservations) : Tensor α (.dim nStates (.dim nStates .scalar)) :=
  let emitNext := emissionVec (α := α) m obs_next
  -- Unnormalized ξ(i,j) = α_t(i) * A(i,j) * B(j, obs_{t+1}) * β_{t+1}(j)
  let xiRaw :=
    Tensor.dim (fun i =>
      Tensor.dim (fun j =>
        match get alpha_t i, get (get m.trans_prob i) j, get emitNext j, get beta_next j with
        | Tensor.scalar ai, Tensor.scalar aij, Tensor.scalar bj, Tensor.scalar bnext =>
            Tensor.scalar (ai * aij * bj * bnext)))
  -- Normalize so each ξ_t sums to 1 (helps control roundoff).
  let s := sumSpec xiRaw
  if s > 0 then scaleSpec xiRaw (1 / s) else xiRaw

 /-- Sum `xi_t(i,j)` over a list of `xi` matrices (expected transition count). -/
private def sumXi
  {nStates : Nat}
  (xis : List (Tensor α (.dim nStates (.dim nStates .scalar))))
  (i : Fin nStates) (j : Fin nStates) : α :=
  xis.foldl (fun acc xi =>
    match get (get xi i) j with
    | Tensor.scalar v => acc + v) 0

 /--
Sum `gamma_t(state)` over timesteps where the observation equals a given symbol.

This yields the expected emission count for `(state, symbol)`.
 -/
private def sumGammaWhereObs
  {nStates nObservations : Nat} [DecidableEq (Fin nObservations)]
  (gammas : List (Tensor α (.dim nStates .scalar)))
  (observations : ObservationSeq nObservations)
  (state : Fin nStates) (sym : Fin nObservations) : α :=
  (List.finRange observations.length).foldl (fun acc t =>
    let ot := observations.getD t.val sym
    if ot = sym then
      match gammas.getD t.val (fill (0 : α) (.dim nStates .scalar)) |> fun g => get g state with
      | Tensor.scalar v => acc + v
    else
      acc) 0

 /--
Normalize each row of a nonnegative matrix to sum to `1`.

Rows with sum `0` fall back to a uniform row (keeps the EM update total).
 -/
private def normalizeRows {nRows nCols : Nat} (m : Tensor α (.dim nRows (.dim nCols .scalar))) :
    Tensor α (.dim nRows (.dim nCols .scalar)) :=
  Tensor.dim (fun i =>
    let row := get m i
    let s := sumSpec row
    if s > 0 then scaleSpec row (1 / s) else uniformVec (α := α) (n := nCols))

 /--
Compute expected sufficient statistics for one observation sequence.

Returns `(initCounts, transCounts, emitCounts, loglik)` suitable for a Baum-Welch M-step.
 -/
private def expectedCounts
  {nStates nObservations : Nat} [Inhabited (Fin nObservations)] [DecidableEq (Fin nObservations)]
  (m : HMMSpec α nStates nObservations)
  (observations : ObservationSeq nObservations) :
  (Tensor α (.dim nStates .scalar) ×
   Tensor α (.dim nStates (.dim nStates .scalar)) ×
   Tensor α (.dim nStates (.dim nObservations .scalar)) ×
   α) :=
  -- Returns:
  -- - initial expected occupancies (for π),
  -- - expected transition counts (for A),
  -- - expected emission counts (for B),
  -- - scaled log-likelihood Σ log c_t.
  match observations with
  | [] =>
      (fill (0 : α) (.dim nStates .scalar),
       fill (0 : α) (.dim nStates (.dim nStates .scalar)),
       fill (0 : α) (.dim nStates (.dim nObservations .scalar)),
       0)
  | _ =>
      let (alphas, scales) := hmmForwardScaled (α := α) m observations
      let betas := hmmBackwardScaled (α := α) m observations scales
      let gammas :=
        (List.finRange observations.length).map (fun t =>
          gammaAt (α := α)
            (alphas.getD t.val (fill (0 : α) (.dim nStates .scalar)))
            (betas.getD t.val (fill (0 : α) (.dim nStates .scalar))))
      let xis :=
        (List.finRange (observations.length - 1)).map (fun t =>
          let alpha_t := alphas.getD t.val (fill (0 : α) (.dim nStates .scalar))
          let beta_next := betas.getD (t.val + 1) (fill (0 : α) (.dim nStates .scalar))
          let obs_next := observations.getD (t.val + 1) default
          xiAt (α := α) m alpha_t beta_next obs_next)
      let initCounts := gammas.getD 0 (fill (0 : α) (.dim nStates .scalar))
      let transCounts :=
        Tensor.dim (fun i =>
          Tensor.dim (fun j =>
            Tensor.scalar (sumXi (α := α) xis i j)))
      let emitCounts :=
        Tensor.dim (fun i =>
          Tensor.dim (fun o =>
            Tensor.scalar (sumGammaWhereObs (α := α) gammas observations i o)))
      let loglik :=
        scales.foldl (fun acc c => if c > 0 then acc + MathFunctions.log c else acc) 0
      (initCounts, transCounts, emitCounts, loglik)

/-- One Baum–Welch (EM) step on a single sequence. -/
def baumWelchStepSpec
  {nStates nObservations : Nat} [Inhabited (Fin nObservations)] [DecidableEq (Fin nObservations)]
  (m : HMMSpec α nStates nObservations)
  (observations : ObservationSeq nObservations) :
  (HMMSpec α nStates nObservations × α) :=
  let (initCounts, transCounts, emitCounts, loglik) := expectedCounts (α := α) m observations
  let init_prob :=
    let (v, _) := normalizeVec (α := α) initCounts
    v
  let trans_prob := normalizeRows (α := α) transCounts
  let emission_prob := normalizeRows (α := α) emitCounts
  ({ init_prob := init_prob, trans_prob := trans_prob, emission_prob := emission_prob }, loglik)

/-- One Baum–Welch epoch over a dataset of observation sequences (sums expected counts). -/
def baumWelchEpochSpec
  {nStates nObservations : Nat} [Inhabited (Fin nObservations)] [DecidableEq (Fin nObservations)]
  (m : HMMSpec α nStates nObservations)
  (dataset : List (ObservationSeq nObservations)) :
  (HMMSpec α nStates nObservations × α) :=
  let init0 := fill (0 : α) (.dim nStates .scalar)
  let trans0 := fill (0 : α) (.dim nStates (.dim nStates .scalar))
  let emit0 := fill (0 : α) (.dim nStates (.dim nObservations .scalar))
  let (initSum, transSum, emitSum, ll) :=
    dataset.foldl (fun (acc : Tensor α (.dim nStates .scalar) ×
                        Tensor α (.dim nStates (.dim nStates .scalar)) ×
                        Tensor α (.dim nStates (.dim nObservations .scalar)) × α) obs =>
      let (accInit, accTrans, accEmit, accLL) := acc
      let (iC, tC, eC, llik) := expectedCounts (α := α) m obs
      (addSpec accInit iC, addSpec accTrans tC, addSpec accEmit eC, accLL + llik)
    ) (init0, trans0, emit0, 0)
  let init_prob := (normalizeVec (α := α) initSum).1
  let trans_prob := normalizeRows (α := α) transSum
  let emission_prob := normalizeRows (α := α) emitSum
  ({ init_prob := init_prob, trans_prob := trans_prob, emission_prob := emission_prob }, ll)

/-! ## Forward / likelihood -/

/-- Forward algorithm (scaled) returning the total sequence likelihood.

Implementation note:
we compute the likelihood from the per-timestep scaling factors produced by
`hmm_forward_scaled`. This avoids the worst underflow behavior of multiplying many small
probabilities directly.
-/
def hmmForwardSpec
  {nStates nObservations : Nat} [Inhabited (Fin nObservations)]
  (m : HMMSpec α nStates nObservations)
  (observations : ObservationSeq nObservations) :
  α :=
  match observations with
  | [] => sumSpec m.init_prob
  | _ =>
      let (_alphas, scales) := hmmForwardScaled (α := α) m observations
      scales.foldl (fun acc c => acc * c) 1

/-- Batched forward pass: compute likelihood for each observation sequence in a list. -/
def hmmBatchedForwardSpec {nStates nObservations : Nat} [Inhabited (Fin nObservations)]
  (m : HMMSpec α nStates nObservations)
  (observations : List (ObservationSeq nObservations)) :
  List α :=
  observations.map (hmmForwardSpec m)

/--
Initialize an HMM with uniform (uninformative) parameters.

This is a deterministic uniform initializer (useful for examples/tests); it is not intended as a
statistically meaningful random initialization.
 -/
def hmmInitSpec {nStates nObservations : Nat} :
  HMMSpec α nStates nObservations :=
  let init_prob : Tensor α (.dim nStates .scalar) := uniformVec (α := α) (n := nStates)
  let trans_prob : Tensor α (.dim nStates (.dim nStates .scalar)) :=
    Tensor.dim (fun _ => uniformVec (α := α) (n := nStates))
  let emission_prob : Tensor α (.dim nStates (.dim nObservations .scalar)) :=
    Tensor.dim (fun _ => uniformVec (α := α) (n := nObservations))
  {
    init_prob := init_prob,
    trans_prob := trans_prob,
    emission_prob := emission_prob
  }

/-- Log-likelihood of an observation sequence.

We compute this from the same scaling factors used in the EM implementation:
`log p(x_{0:T-1}) = Σ_t log c_t`.
-/
def hmmLogLikelihoodSpec {nStates nObservations : Nat} [Inhabited (Fin nObservations)]
  (m : HMMSpec α nStates nObservations)
  (observations : ObservationSeq nObservations) :
  α :=
  match observations with
  | [] =>
      let s := sumSpec m.init_prob
      if s > 0 then MathFunctions.log s else 0
  | _ =>
      let (_alphas, scales) := hmmForwardScaled (α := α) m observations
      scales.foldl (fun acc c => if c > 0 then acc + MathFunctions.log c else acc) 0

end Spec
