/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Tensor.Sequence
public import TorchLean.Tensor.LinearAlgebra.Identity
public import TorchLean.Tensor.Reduce
public import TorchLean.Tensor.Contract

/-!
# Hidden Markov Model (HMM) (spec model)

The HMM here has discrete observations:

- hidden states: `nStates`
- observations: `nObservations` (discrete symbols)

The model parameters are:
- initial distribution $\pi$,
- transition matrix $A$, and
- emission matrix $B$.

A length-`T` observation sequence is an `ObservationSeq nObservations T`, which abbreviates
`Tensor (Fin nObservations) [T]`. The element type keeps observation symbols distinct from
probabilities, while the tensor shape records the sequence length.

## Notation and shapes

The notation is the conventional HMM one:

- $\pi$: initial state distribution,
- $A$: transition matrix, with $A_{ij}=P(z_{t+1}=j\mid z_t=i)$, and
- $B$: emission matrix, with $B_{io}=P(x_t=o\mid z_t=i)$.

An observation sequence is $o_0,o_1,\ldots,o_{T-1}$, where each observation is represented by
`Fin nObservations`.

References:

- Rabiner (1989),
  "A Tutorial on Hidden Markov Models and Selected Applications in Speech Recognition",
  Proceedings of the IEEE 77(2), 257-286, Section V.A (Scaling) and Section V.B (Multiple
  Observation Sequences), doi:10.1109/5.18626:
  https://ieeexplore.ieee.org/document/18626
- Baum and Petrie (1966),
  "Statistical Inference for Probabilistic Functions of Finite State Markov Chains",
  Annals of Mathematical Statistics 37(6), 1554-1563:
  https://doi.org/10.1214/aoms/1177699147

PyTorch analogy:

- emissions are categorical distributions (`torch.distributions.Categorical`),
- a forward step is a vector-matrix product with $A$, which sums over the previous states, followed
  by reweighting with the emission vector $B_{\mathord{:},o_t}$.

PyTorch users often reach for a dedicated HMM library (e.g. `hmmlearn`) or write the recursions in
log space with `logsumexp`. The spec keeps probabilities in linear space with per-step scaling, a
simple form that suits reading and proofs.

## Implementation status

No API builder implements this model. The module holds definitions only, with no correctness
theorems for the forward–backward or Baum–Welch code.
-/

public section


open FloatLib.Numerics (MathFunctions)

namespace TorchLean

open TorchLean.Tensor

variable {α : Type} [Storage α] [Context α]

/-- A discrete-observation HMM.

Probabilistic validity (nonnegativity, rows summing to $1$) is a modeling assumption that neither
the type nor any function in this file checks, much as PyTorch stores unconstrained tensors until
they reach a distribution or a loss.
-/
structure HMM (α : Type) [Storage α]
    (nStates nObservations : Nat) where
  /-- Initial distribution $\pi$. -/
  initial : Tensor α [nStates]
  /-- Transition matrix $A$. -/
  transition : Tensor α [nStates, nStates]
  /-- Emission matrix $B$. -/
  emission : Tensor α [nStates, nObservations]

/-- A fixed-length sequence of symbols from the discrete observation alphabet. -/
abbrev HMM.ObservationSeq (nObservations length : Nat) :=
  Tensor (Fin nObservations) [length]

/-!
## Baum–Welch (EM) training

Discrete-observation HMMs are classically trained with the Baum–Welch algorithm, an EM procedure:

- **E-step**: run forward–backward to compute expected state occupancies $\gamma$ and expected
  transition counts $\xi$.
- **M-step**: normalize those expected counts to update $\pi$, $A$, and $B$. Counts whose sum does
  not compare positive give a uniform row.

Forward–backward runs in *scaled* form to limit underflow: each forward message $\alpha_t$ is
normalized by a scalar $c_t$, and the backward messages divide by those same scalars. The sequence
likelihood is then $\prod_t c_t$, so the log-likelihood is $\sum_t\log c_t$.

Concretely:

- forward recursion (unnormalized):
  $$
  \widetilde{\alpha}_{t+1}(j)
    = B_{j,o_{t+1}}\sum_i \alpha_t(i)A_{ij};
  $$
- scaling:
  $$
  c_t=\sum_j\widetilde{\alpha}_t(j),
  \qquad
  \alpha_t=\frac{\widetilde{\alpha}_t}{c_t},
  \qquad
  \sum_j\alpha_t(j)=1.
  $$

Many practical HMM implementations use the same scaling, and others run forward–backward in log
space. Every step here is deterministic, and the code is written for clarity over speed.

We wrote the scaled recursions by hand from Rabiner's Section V.A. His scale (91) is
$1/\sum_j\widetilde{\alpha}_t(j)$ in our notation, the reciprocal of ours, so a first message of
total mass $0.25$ has $c_0=0.25$ here and $4$ there. That turns his likelihood (102),
$P=1/\prod_t c_t$, into our $\prod_t c_t$ and drops the minus sign from his (103). In his (94) the
scale of step $t$ multiplies $\beta_t$. We divide $\beta_t$ by the forward scale of step $t+1$, and
that one-step shift is why the product in `gammaAt` already sums to one.
-/

/-- Normalize a nonnegative vector $v$ to sum to $1$, returning
$(v/\sum_i v_i,\sum_i v_i)$.

If the computed sum does not compare strictly positive, the result is the uniform vector with
entries $1/n$ and scale $0$. For a valid nonnegative model in exact arithmetic, a zero scale in the
forward pass marks an impossible observation prefix. Under rounding, underflow can also produce it,
and a normalized vector need not have a floating-point sum of exactly one.
-/
def HMM.normalizeVec {n : Nat} (v : Tensor α [n]) : (Tensor α [n] × α) :=
  let s := Tensor.sum v
  if s > 0 then
    (v / Tensor.full [n] s, s)
  else
    (Tensor.full [n] (Context.divNat 1 n), 0)

/-- Sum of the logarithms of the scale factors, taken left to right. A factor that does not compare
positive, such as the zero scale of an impossible prefix, makes the result `none`. -/
private def HMM.logScales? {length : Nat} (scales : Tensor α [length]) : Option α :=
  scales.data.foldl
    (fun acc c =>
      match acc with
      | none => none
      | some total => if c > 0 then some (total + MathFunctions.log c) else none)
    (some 0)

/-- Emission probabilities $B_{\mathord{:},\mathtt{obs}}$ as a vector over states. -/
def HMM.emissionVec {nStates nObservations : Nat}
  (m : HMM α nStates nObservations) (obs : Fin nObservations) : Tensor α [nStates]
    :=
  Tensor.ofFn (fun s => get2 m.emission s obs)

/--
One unnormalized step of the scaled forward algorithm.

Given the normalized message `prevAlpha` and the next observation `obs`, returns the unnormalized
message $\widetilde{\alpha}_{t+1}$: the product of `prevAlpha` with $A$, reweighted elementwise by
the emission probabilities of `obs`.
-/
private def HMM.forwardUpdate {nStates nObservations : Nat}
  (m : HMM α nStates nObservations)
  (prevAlpha : Tensor α [nStates])
  (obs : Fin nObservations) : Tensor α [nStates] :=
  -- `.vecMat` contracts over the first axis of `A`, giving `Σ_i prevAlpha(i) * A(i, j)`.
  let emissionProbs := emissionVec (α := α) m obs
  emissionProbs * Tensor.contract prevAlpha m.transition _ _ [nStates] .vecMat

/-- One timestep of a scaled HMM forward trace. -/
structure HMM.ForwardStep (α : Type) [Storage α]
    (nStates nObservations : Nat) where
  /-- Observation consumed at this timestep. -/
  observation : Fin nObservations
  /-- Normalized forward message $\alpha_t$. -/
  message : Tensor α [nStates]
  /-- Normalization constant $c_t$. -/
  scale : α

/-- Scaled forward pass, returning the observation, $\alpha_t$, and $c_t$ at each timestep.

The first unnormalized message is $\pi\odot B_{\mathord{:},o_0}$, and each later one comes from
`HMM.forwardUpdate`. `normalizeVec` turns each into $\alpha_t$ and $c_t$, so a message whose sum
does not compare positive becomes uniform with $c_t=0$.

For valid probabilities in exact arithmetic, the total likelihood is
$p(o_{0:T-1})=\prod_t c_t$.
-/
def HMM.forwardScaled
  {nStates nObservations length : Nat}
  (m : HMM α nStates nObservations)
  (observations : ObservationSeq nObservations length) :
  Tensor (ForwardStep α nStates nObservations) [length] :=
  (Sequence.mapAccum length (none : Option (Tensor α [nStates])) fun t previous =>
    let observation := observations.getScalar t
    let raw := match previous with
      | none => m.initial * emissionVec (α := α) m observation
      | some message => forwardUpdate (α := α) m message observation
    let (message, scale) := normalizeVec (α := α) raw
    (some message, { observation, message, scale })).2

/-- Scaled backward pass, producing the backward messages $\beta_t$, with $\beta_{T-1}$ all ones.

The standard backward recursion is:

$$
\beta_t(i)=\sum_j A_{ij}B_{j,o_{t+1}}\beta_{t+1}(j).
$$

The scaled variant divides each message by the forward scale $c_{t+1}$, which keeps
$\alpha_t\odot\beta_t$ well-conditioned, and skips the division when $c_{t+1}$ does not compare
positive.
-/
private def HMM.backwardScaled
  {nStates nObservations length : Nat}
  (m : HMM α nStates nObservations)
  (steps : Tensor (ForwardStep α nStates nObservations) [length]) :
  Tensor (Tensor α [nStates]) [length] :=
  let betaLast : Tensor α [nStates] := Tensor.full [nStates] 1
  let initial : Option (Tensor α [nStates] × ForwardStep α nStates nObservations) :=
    none
  (Sequence.mapAccumRight length initial fun index next =>
    let current := steps.getScalar index
    match next with
    | none =>
        (some (betaLast, current), betaLast)
    | some (betaNext, nextStep) =>
        let emitNext := emissionVec (α := α) m nextStep.observation
        let betaRaw : Tensor α [nStates] :=
          Tensor.dim (fun i =>
            let sumOverJ : Tensor α [nStates] :=
              Tensor.dim (fun j =>
                let aij := (Tensor.unstack m.transition i).getScalar j
                Tensor.scalar (aij * emitNext.getScalar j * betaNext.getScalar j))
            Tensor.scalar (Tensor.sum sumOverJ))
        let beta :=
          if nextStep.scale > 0 then
            betaRaw / Tensor.full [nStates] nextStep.scale
          else
            betaRaw
        (some (beta, current), beta)).2

 /--
Normalized state posterior $\gamma_t$ from the forward and backward messages:

$$
\gamma_t(i)\propto\alpha_t(i)\beta_t(i).
$$

The product goes through `normalizeVec`, so it becomes uniform when its sum does not compare
positive. With positive scales in exact arithmetic the product already sums to one.
-/
private def HMM.gammaAt {nStates : Nat}
  (alpha : Tensor α [nStates]) (beta : Tensor α [nStates]) : Tensor α
    [nStates] :=
  let g := alpha * beta
  (normalizeVec (α := α) g).1

 /--
Transition posterior $\xi_t$ for one time step, from the unnormalized matrix
$$
\xi_t(i,j)
  =\alpha_t(i)A_{ij}B_{j,o_{t+1}}\beta_{t+1}(j).
$$
The matrix is divided by its sum when that sum compares positive and returned unscaled otherwise.
-/
private def HMM.xiAt {nStates nObservations : Nat}
  (m : HMM α nStates nObservations)
  (alpha_t : Tensor α [nStates])
  (beta_next : Tensor α [nStates])
  (obsNext : Fin nObservations) : Tensor α [nStates, nStates] :=
  let emitNext := emissionVec (α := α) m obsNext
  let xiRaw :=
    Tensor.dim (fun i =>
      Tensor.dim (fun j =>
        let aij := (Tensor.unstack m.transition i).getScalar j
        Tensor.scalar
          (alpha_t.getScalar i * aij * emitNext.getScalar j * beta_next.getScalar j)))
  -- With positive scales in exact arithmetic the raw entries sum to c_{t+1}, so dividing by the
  -- sum is the division by c_{t+1} that turns them into the posterior ξ_t.
  let s := Tensor.sum xiRaw
  if s > 0 then xiRaw / Tensor.full [nStates, nStates] s else xiRaw

 /-- Sum $\xi_t(i,j)$ over an array of $\xi$ matrices (expected transition count). -/
private def HMM.sumXi
  {nStates : Nat}
  (xis : Array (Tensor α [nStates, nStates]))
  (i : Fin nStates) (j : Fin nStates) : α :=
  xis.foldl (fun acc xi =>
    acc + (Tensor.unstack xi i).getScalar j) 0

 /--
Expected emission count for `(state, sym)`: the sum of $\gamma_t(\mathtt{state})$ over the
timesteps whose observation is `sym`.
-/
private def HMM.sumGammaWhereObs
  {nStates nObservations length : Nat} [DecidableEq (Fin nObservations)]
  (observations : ObservationSeq nObservations length)
  (gammas : Tensor (Tensor α [nStates])
    [length])
  (state : Fin nStates) (sym : Fin nObservations) : α :=
  (Array.finRange length).foldl (fun acc t =>
    if observations.getScalar t = sym then
      acc + (gammas.getScalar t).getScalar state
    else
      acc) 0

 /--
Apply `normalizeVec` independently to each row.

Rows whose computed sum does not compare positive fall back to a uniform row.
 -/
private def HMM.normalizeRows {nRows nCols : Nat} (m : Tensor α [nRows, nCols]) :
    Tensor α [nRows, nCols] :=
  Tensor.dim (fun i => (normalizeVec (Tensor.unstack m i)).1)

 /--
Expected sufficient statistics of one observation sequence, returned as
`(initCounts, transCounts, emitCounts, loglik)`: the posterior $\gamma_0$ for $\pi$, the summed
$\xi_t$ for $A$, the per-symbol sums of $\gamma_t$ for $B$, and $\sum_t\log c_t$.

`loglik` is `none` when some scale does not compare positive. An empty sequence has zero counts and
`loglik` equal to the log of the total initial mass $\sum_i\pi_i$, or `none` if that total does not
compare positive.
-/
private def HMM.expectedCounts
  {nStates nObservations length : Nat} [DecidableEq (Fin nObservations)]
  (m : HMM α nStates nObservations)
  (observations : ObservationSeq nObservations length) :
  (Tensor α [nStates] ×
   Tensor α [nStates, nStates] ×
   Tensor α [nStates, nObservations] ×
   Option α) :=
  if hLength : length = 0 then
      (Tensor.full (.dim nStates .scalar) (0 : α),
       Tensor.full (.dim nStates (.dim nStates .scalar)) (0 : α),
       Tensor.full (.dim nStates (.dim nObservations .scalar)) (0 : α),
       let s := Tensor.sum m.initial
       if s > 0 then some (MathFunctions.log s) else none)
  else
      let steps := forwardScaled (α := α) m observations
      let betas := backwardScaled (α := α) m steps
      let gammas : Tensor (Tensor α [nStates]) [length] :=
        Tensor.ofFn fun t =>
          gammaAt (α := α) (steps.getScalar t).message (betas.getScalar t)
      let xis :=
        (Array.finRange length).foldl (fun xis current =>
          if hNext : current.val + 1 < length then
            let next : Fin length := ⟨current.val + 1, hNext⟩
            let nextStep := steps.getScalar next
            xis.push <| xiAt (α := α) m (steps.getScalar current).message
              (betas.getScalar next) nextStep.observation
          else
            xis) #[]
      let first : Fin length := ⟨0, Nat.pos_of_ne_zero hLength⟩
      let initCounts := gammas.getScalar first
      let transCounts :=
        Tensor.dim (fun i =>
          Tensor.dim (fun j =>
            Tensor.scalar (sumXi (α := α) xis i j)))
      let emitCounts :=
        Tensor.dim (fun i =>
          Tensor.dim (fun o =>
            Tensor.scalar (sumGammaWhereObs (α := α) observations gammas i o)))
      let scales : Tensor α [length] := steps.map (·.scale)
      let loglik := logScales? (α := α) scales
      (initCounts, transCounts, emitCounts, loglik)

/-- One Baum–Welch (EM) step on a single sequence, returning the re-estimated model and the
log-likelihood of `m` before the update.

An empty sequence has only zero counts, so the result has the uniform parameters of `HMM.init`.
-/
def HMM.baumWelchStep
  {nStates nObservations length : Nat} [DecidableEq (Fin nObservations)]
  (m : HMM α nStates nObservations)
  (observations : ObservationSeq nObservations length) :
  (HMM α nStates nObservations × Option α) :=
  let (initCounts, transCounts, emitCounts, loglik) := expectedCounts (α := α) m observations
  let initial :=
    let (v, _) := normalizeVec (α := α) initCounts
    v
  let transition := normalizeRows (α := α) transCounts
  let emission := normalizeRows (α := α) emitCounts
  ({ initial, transition, emission }, loglik)

/-- One Baum–Welch epoch over a dataset of observation sequences: the expected counts of all
sequences are summed before a single M-step. The log-likelihood is the sum over the sequences, or
`none` if any of them is `none`. The counts are posteriors, so they already carry the factor
$1/P_k$ of Rabiner's multiple-sequence formulas (109) and (110). He keeps $\pi$ fixed there because
his left-right models start in state 1, and we re-estimate it from the summed $\gamma_0$. -/
def HMM.baumWelchEpoch
  {nStates nObservations batch length : Nat} [DecidableEq (Fin nObservations)]
  (m : HMM α nStates nObservations)
  (dataset : Tensor (Fin nObservations) [batch, length]) :
  (HMM α nStates nObservations × Option α) :=
  let init0 := Tensor.full (.dim nStates .scalar) (0 : α)
  let trans0 := Tensor.full (.dim nStates (.dim nStates .scalar)) (0 : α)
  let emit0 := Tensor.full (.dim nStates (.dim nObservations .scalar)) (0 : α)
  let (initSum, transSum, emitSum, ll) :=
    (Array.finRange batch).foldl (fun (acc : Tensor α [nStates] ×
                        Tensor α [nStates, nStates] ×
                        Tensor α [nStates, nObservations] × Option α) obs =>
      let (accInit, accTrans, accEmit, accLL) := acc
      let (iC, tC, eC, llik) := expectedCounts (α := α) m (Tensor.unstack dataset obs)
      let nextLL :=
        match accLL, llik with
        | some a, some b => some (a + b)
        | _, _ => none
      (accInit + iC, accTrans + tC, accEmit + eC, nextLL)
    ) (init0, trans0, emit0, some 0)
  let initial := (normalizeVec (α := α) initSum).1
  let transition := normalizeRows (α := α) transSum
  let emission := normalizeRows (α := α) emitSum
  ({ initial, transition, emission }, ll)

/-! ## Forward / likelihood -/

/-- Sequence likelihood from the scaled forward pass: the product of the scales $c_t$ of
`forwardScaled`, or $\sum_i\pi_i$ for an empty sequence.

Each message is rescaled to sum to one, but over a long sequence the product of the scales can
underflow to zero. `HMM.logLikelihood` sums logarithms of the same scales and avoids that.
-/
def HMM.forward
  {nStates nObservations length : Nat}
  (m : HMM α nStates nObservations)
  (observations : ObservationSeq nObservations length) :
  α :=
  if length = 0 then
    Tensor.sum m.initial
  else
      let steps := forwardScaled (α := α) m observations
      steps.data.foldl (fun acc step => acc * step.scale) 1

/--
Deterministic uniform parameters: $\pi_i=1/\mathtt{nStates}$, $A_{ij}=1/\mathtt{nStates}$ and
$B_{io}=1/\mathtt{nObservations}$.

All states are interchangeable in this model, and a Baum–Welch step preserves that symmetry. The
parameters suit examples and tests, and training that should tell states apart needs a nonuniform
start.
-/
def HMM.init {nStates nObservations : Nat} :
  HMM α nStates nObservations :=
  let initial : Tensor α [nStates] := Tensor.full [nStates] (Context.divNat 1 nStates)
  let transition : Tensor α [nStates, nStates] :=
    Tensor.dim (fun _ => Tensor.full [nStates] (Context.divNat 1 nStates))
  let emission : Tensor α [nStates, nObservations] :=
    Tensor.dim (fun _ => Tensor.full [nObservations] (Context.divNat 1 nObservations))
  { initial, transition, emission }

/-- Log-likelihood of an observation sequence, from the scales $c_t$ that the Baum–Welch code also
uses:
$$
\log p(x_{0:T-1})=\sum_t\log c_t.
$$
The result is `none` when some $c_t$ does not compare positive. An empty sequence gives
$\log\sum_i\pi_i$, or `none` if that sum does not compare positive.
-/
def HMM.logLikelihood {nStates nObservations : Nat}
  (m : HMM α nStates nObservations)
  {length : Nat} (observations : ObservationSeq nObservations length) :
  Option α :=
  if length = 0 then
      let s := Tensor.sum m.initial
      if s > 0 then some (MathFunctions.log s) else none
  else
      let steps := forwardScaled (α := α) m observations
      logScales? (α := α) (steps.map (·.scale))

end TorchLean
