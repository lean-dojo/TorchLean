/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Core.Sequence
public import NN.Spec.Core.TensorReductionShape.Reductions

/-!
# Hidden Markov Model (HMM) (spec model)

This file defines an HMM with discrete observations:

- hidden states: `nStates`
- observations: `nObservations` (discrete symbols)

The model parameters are:
- initial distribution $\pi$,
- transition matrix $A$, and
- emission matrix $B$.

We represent a length-`T` observation sequence as a `Tensor (Fin nObservations) [T]`. The element
type keeps observation symbols distinct from probabilities, while the tensor shape records the
sequence length.

## Notation and shapes

We use the conventional HMM notation:

- $\pi$: initial state distribution,
- $A$: transition matrix, with $A_{ij}=P(z_{t+1}=j\mid z_t=i)$, and
- $B$: emission matrix, with $B_{io}=P(x_t=o\mid z_t=i)$.

An observation sequence is $o_0,o_1,\ldots,o_{T-1}$, where each observation is represented by
`Fin nObservations`.

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
- the forward algorithm corresponds to multiplying by $A$ and reweighting by the emission vector
  $B_{\mathord{:},o_t}$,
  then summing over previous states (often implemented in log-space in practice).

In practice, PyTorch users often reach for a dedicated HMM library (e.g. `hmmlearn`) or implement
HMMs in log-space with `logsumexp`; TorchLean keeps the spec in a simple, explicit form that is
good for reading and proofs.

## Implementation status

No API builder implements this model. `NN/Spec/Module/Hmm.lean` wraps it as a `Spec.Module`, and
`NN/Tests/Runtime/Floats/TorchLeanOpsCheck.lean` checks it numerically; no theorem is proved about
it.
-/

public section


open TorchLean

namespace Spec

open TorchLean TorchLean.Tensor

variable {α : Type} [TorchLean.Storage α] [Context α]

/-- A discrete-observation HMM.

We do not enforce probabilistic validity (nonnegativity or rows summing to $1$) at the type level;
that is a modeling assumption, similar to how PyTorch will happily store unconstrained tensors
until you feed them to a distribution or a loss.
-/
structure HMMSpec (α : Type) [TorchLean.Storage α]
    (nStates nObservations : Nat) where
  /-- Initial distribution $\pi$. -/
  initial : Tensor α [nStates]
  /-- Transition matrix $A$. -/
  transition : Tensor α [nStates, nStates]
  /-- Emission matrix $B$. -/
  emission : Tensor α [nStates, nObservations]

/-- A fixed-length sequence of symbols from the discrete observation alphabet. -/
abbrev ObservationSeq (nObservations length : Nat) :=
  Tensor (Fin nObservations) [length]

/-! ## Basic helpers -/

/-- Get the emission probability $B_{\mathtt{state},\mathtt{obs}}$ for a discrete symbol. -/
def getEmissionProbDiscrete
  {nStates nObservations : Nat}
  (m : HMMSpec α nStates nObservations)
  (state : Fin nStates)
  (obs : Fin nObservations) : α :=
  (Tensor.unstack m.emission state).getScalar obs

/-!
## Baum–Welch (EM) training

The forward-pass APIs above are enough to *use* a fixed HMM, but a “fully implemented” baseline
should also include classical training. For discrete-observation HMMs, the standard training
procedure is the Baum–Welch algorithm (an EM procedure):

- **E-step**: run forward–backward to compute expected state occupancies $\gamma$ and expected
  transition counts $\xi$.
- **M-step**: normalize those expected counts to update $\pi$, $A$, and $B$.

This implementation uses *scaled* forward–backward to reduce numerical underflow:
each forward message $\alpha_t$ is normalized by a scalar $c_t$, and the backward messages divide
by those same scalars. The sequence likelihood is then $\prod_t c_t$, so the log-likelihood is
$\sum_t\log c_t$.

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

This is the same basic idea used in many practical HMM implementations (sometimes also expressed as
log-space forward–backward).

This is deterministic and written for clarity; it is not intended to be a high-performance HMM
trainer.
-/

/--
Uniform distribution vector of length `n`.

This is used to keep posterior messages total after an impossible observation has already been
recorded by a zero scaling factor.
-/
private def uniformVec {n : Nat} : Tensor α [n] :=
  match n with
  | 0 => Tensor.dim (fun k => nomatch k)
  | Nat.succ _ => Tensor.dim (fun _ => Tensor.scalar (1 / (n : α)))

/-- Normalize a nonnegative vector $v$ to sum to $1$, returning
$(v/\sum_i v_i,\sum_i v_i)$.

If the sum is $0$, the normalized message is totalized to a uniform vector, while the returned
scale remains $0$. The zero scale is essential: it records that the observation prefix has
probability zero.
-/
def normalizeVec {n : Nat} (v : Tensor α [n]) : (Tensor α [n] × α) :=
  let s := sumSpec v
  if s > 0 then
    (scaleSpec v (1 / s), s)
  else
    (uniformVec (α := α) (n := n), 0)

/-- Sum logarithms of positive scale factors. A nonpositive factor represents zero likelihood. -/
private def logScales? {length : Nat} (scales : Tensor α [length]) : Option α :=
  scales.data.foldl
    (fun acc c =>
      match acc with
      | none => none
      | some total => if c > 0 then some (total + MathFunctions.log c) else none)
    (some 0)

/-- Emission probabilities $B_{\mathord{:},\mathtt{obs}}$ as a vector over states. -/
def emissionVec {nStates nObservations : Nat}
  (m : HMMSpec α nStates nObservations) (obs : Fin nObservations) : Tensor α [nStates]
    :=
  Tensor.dim (fun s => Tensor.scalar (getEmissionProbDiscrete m s obs))

/--
One forward step (unnormalized) of the scaled forward algorithm.

Given the previous normalized forward message `prevAlpha` and the next observation `obs`, compute
the next unnormalized message `alphaTilde`.
-/
private def forwardStep {nStates nObservations : Nat}
  (m : HMMSpec α nStates nObservations)
  (prevAlpha : Tensor α [nStates])
  (obs : Fin nObservations) : Tensor α [nStates] :=
  -- One forward update (without scaling): apply transitions, then reweight by emissions.
  let emissionProbs := emissionVec (α := α) m obs
  Tensor.dim (fun s =>
    let trans_sum := Tensor.dim (fun s' =>
      let alpha_val := prevAlpha.getScalar s'
      let transVal := (Tensor.unstack m.transition s').getScalar s
      Tensor.scalar (alpha_val * transVal))
    let transTotal := sumSpec trans_sum
    Tensor.scalar (emissionProbs.getScalar s * transTotal))

/-- One timestep of a scaled HMM forward trace. -/
structure HMMForwardStep (α : Type) [TorchLean.Storage α]
    (nStates nObservations : Nat) where
  /-- Observation consumed at this timestep. -/
  observation : Fin nObservations
  /-- Normalized forward message $\alpha_t$. -/
  message : Tensor α [nStates]
  /-- Normalization constant $c_t$. -/
  scale : α

/-- Scaled forward pass, returning the observation, $\alpha_t$, and $c_t$ at each timestep.

- Each $\alpha_t$ is normalized to sum to $1$.
- Each $c_t$ is the normalization constant used at step $t$.

If you need the total likelihood, multiply the scales:
$p(o_{0:T-1})=\prod_t c_t$.
-/
def hmmForwardScaled
  {nStates nObservations length : Nat}
  (m : HMMSpec α nStates nObservations)
  (observations : ObservationSeq nObservations length) :
  Tensor (HMMForwardStep α nStates nObservations) [length] :=
  (Sequence.mapAccum length (none : Option (Tensor α [nStates])) fun t previous =>
    let observation := observations.getScalar t
    let raw := match previous with
      | none => mulSpec m.initial (emissionVec (α := α) m observation)
      | some message => forwardStep (α := α) m message observation
    let (message, scale) := normalizeVec (α := α) raw
    (some message, { observation, message, scale })).2

/-- Scaled backward pass, producing normalized backward messages $\beta_t$.

The standard backward recursion is:

$$
\beta_t(i)=\sum_j A_{ij}B_{j,o_{t+1}}\beta_{t+1}(j).
$$

In the scaled variant, we divide by the forward scale $c_{t+1}$ so that
$\alpha_t\odot\beta_t$ stays
well-conditioned.
-/
private def hmmBackwardScaled
  {nStates nObservations length : Nat}
  (m : HMMSpec α nStates nObservations)
  (steps : Tensor (HMMForwardStep α nStates nObservations) [length]) :
  Tensor (Tensor α [nStates]) [length] :=
  let betaLast : Tensor α [nStates] := Tensor.full [nStates] 1
  let initial : Option (Tensor α [nStates] × HMMForwardStep α nStates nObservations) :=
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
            Tensor.scalar (sumSpec sumOverJ))
        let beta :=
          if nextStep.scale > 0 then
            scaleSpec betaRaw (1 / nextStep.scale)
          else
            betaRaw
        (some (beta, current), beta)).2

 /-- Elementwise multiplication for state-probability vectors. -/
private def elementwiseMul {n : Nat} (a b : Tensor α [n]) : Tensor α [n]
  :=
  mulSpec a b

 /--
Compute the normalized state posterior $\gamma_t$ from forward/backward messages.

$$
\gamma_t(i)\propto\alpha_t(i)\beta_t(i).
$$
 -/
private def gammaAt {nStates : Nat}
  (alpha : Tensor α [nStates]) (beta : Tensor α [nStates]) : Tensor α
    [nStates] :=
  -- γ_t(i) ∝ α_t(i) * β_t(i)
  let g := elementwiseMul (α := α) alpha beta
  (normalizeVec (α := α) g).1

 /--
Compute the normalized transition posterior $\xi_t$ for a single time step.

Unnormalized:
$$
\xi_t(i,j)
  =\alpha_t(i)A_{ij}B_{j,o_{t+1}}\beta_{t+1}(j).
$$
 -/
private def xiAt {nStates nObservations : Nat}
  (m : HMMSpec α nStates nObservations)
  (alpha_t : Tensor α [nStates])
  (beta_next : Tensor α [nStates])
  (obsNext : Fin nObservations) : Tensor α [nStates, nStates] :=
  let emitNext := emissionVec (α := α) m obsNext
  -- Unnormalized ξ(i,j) = α_t(i) * A(i,j) * B(j, obs_{t+1}) * β_{t+1}(j)
  let xiRaw :=
    Tensor.dim (fun i =>
      Tensor.dim (fun j =>
        let aij := (Tensor.unstack m.transition i).getScalar j
        Tensor.scalar
          (alpha_t.getScalar i * aij * emitNext.getScalar j * beta_next.getScalar j)))
  -- Normalize so each ξ_t sums to 1 (helps control roundoff).
  let s := sumSpec xiRaw
  if s > 0 then scaleSpec xiRaw (1 / s) else xiRaw

 /-- Sum $\xi_t(i,j)$ over an array of $\xi$ matrices (expected transition count). -/
private def sumXi
  {nStates : Nat}
  (xis : Array (Tensor α [nStates, nStates]))
  (i : Fin nStates) (j : Fin nStates) : α :=
  xis.foldl (fun acc xi =>
    acc + (Tensor.unstack xi i).getScalar j) 0

 /--
Sum $\gamma_t(\mathtt{state})$ over timesteps where the observation equals a given symbol.

This yields the expected emission count for $(\mathtt{state},\mathtt{symbol})$.
 -/
private def sumGammaWhereObs
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
Normalize each row of a nonnegative matrix to sum to `1`.

Rows with sum `0` fall back to a uniform row (keeps the EM update total).
 -/
private def normalizeRows {nRows nCols : Nat} (m : Tensor α [nRows, nCols]) :
    Tensor α [nRows, nCols] :=
  Tensor.dim (fun i =>
    let row := get m i
    let s := sumSpec row
    if s > 0 then scaleSpec row (1 / s) else uniformVec (α := α) (n := nCols))

 /--
Compute expected sufficient statistics for one observation sequence.

Returns `(initCounts, transCounts, emitCounts, loglik)` suitable for a Baum-Welch M-step.
 -/
private def expectedCounts
  {nStates nObservations length : Nat} [DecidableEq (Fin nObservations)]
  (m : HMMSpec α nStates nObservations)
  (observations : ObservationSeq nObservations length) :
  (Tensor α [nStates] ×
   Tensor α [nStates, nStates] ×
   Tensor α [nStates, nObservations] ×
   Option α) :=
  -- Returns:
  -- - initial expected occupancies (for π),
  -- - expected transition counts (for A),
  -- - expected emission counts (for B),
  -- - scaled log-likelihood Σ log c_t.
  if hLength : length = 0 then
      (Tensor.full (.dim nStates .scalar) (0 : α),
       Tensor.full (.dim nStates (.dim nStates .scalar)) (0 : α),
       Tensor.full (.dim nStates (.dim nObservations .scalar)) (0 : α),
       let s := sumSpec m.initial
       if s > 0 then some (MathFunctions.log s) else none)
  else
      let steps := hmmForwardScaled (α := α) m observations
      let betas := hmmBackwardScaled (α := α) m steps
      let gammas : Tensor (Tensor α [nStates]) [length] :=
        TorchLean.Tensor.ofFn fun t =>
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

/-- One Baum–Welch (EM) step on a single sequence. -/
def baumWelchStepSpec
  {nStates nObservations length : Nat} [DecidableEq (Fin nObservations)]
  (m : HMMSpec α nStates nObservations)
  (observations : ObservationSeq nObservations length) :
  (HMMSpec α nStates nObservations × Option α) :=
  let (initCounts, transCounts, emitCounts, loglik) := expectedCounts (α := α) m observations
  let initial :=
    let (v, _) := normalizeVec (α := α) initCounts
    v
  let transition := normalizeRows (α := α) transCounts
  let emission := normalizeRows (α := α) emitCounts
  ({ initial, transition, emission }, loglik)

/-- One Baum–Welch epoch over a dataset of observation sequences (sums expected counts). -/
def baumWelchEpochSpec
  {nStates nObservations batch length : Nat} [DecidableEq (Fin nObservations)]
  (m : HMMSpec α nStates nObservations)
  (dataset : Tensor (Fin nObservations) [batch, length]) :
  (HMMSpec α nStates nObservations × Option α) :=
  let init0 := Tensor.full (.dim nStates .scalar) (0 : α)
  let trans0 := Tensor.full (.dim nStates (.dim nStates .scalar)) (0 : α)
  let emit0 := Tensor.full (.dim nStates (.dim nObservations .scalar)) (0 : α)
  let (initSum, transSum, emitSum, ll) :=
    (Array.finRange batch).foldl (fun (acc : Tensor α [nStates] ×
                        Tensor α [nStates, nStates] ×
                        Tensor α [nStates, nObservations] × Option α) obs =>
      let (accInit, accTrans, accEmit, accLL) := acc
      let (iC, tC, eC, llik) := expectedCounts (α := α) m (get dataset obs)
      let nextLL :=
        match accLL, llik with
        | some a, some b => some (a + b)
        | _, _ => none
      (addSpec accInit iC, addSpec accTrans tC, addSpec accEmit eC, nextLL)
    ) (init0, trans0, emit0, some 0)
  let initial := (normalizeVec (α := α) initSum).1
  let transition := normalizeRows (α := α) transSum
  let emission := normalizeRows (α := α) emitSum
  ({ initial, transition, emission }, ll)

/-! ## Forward / likelihood -/

/-- Forward algorithm (scaled) returning the total sequence likelihood.

Implementation note:
we compute the likelihood from the per-timestep scaling factors produced by
`hmmForwardScaled`. This avoids the worst underflow behavior of multiplying many small
probabilities directly.
-/
def hmmForwardSpec
  {nStates nObservations length : Nat}
  (m : HMMSpec α nStates nObservations)
  (observations : ObservationSeq nObservations length) :
  α :=
  if length = 0 then
    sumSpec m.initial
  else
      let steps := hmmForwardScaled (α := α) m observations
      steps.data.foldl (fun acc step => acc * step.scale) 1

/-- Batched forward pass with statically matched batch and sequence dimensions. -/
def hmmBatchedForwardSpec {nStates nObservations batch length : Nat}
  (m : HMMSpec α nStates nObservations)
  (observations : Tensor (Fin nObservations) [batch, length]) :
  Tensor α [batch] :=
  Tensor.dim fun row => Tensor.scalar (hmmForwardSpec m (get observations row))

/--
Initialize an HMM with uniform (uninformative) parameters.

This is a deterministic uniform initializer (useful for examples/tests); it is not intended as a
statistically meaningful random initialization.
 -/
def hmmInitSpec {nStates nObservations : Nat} :
  HMMSpec α nStates nObservations :=
  let initial : Tensor α [nStates] := uniformVec (α := α) (n := nStates)
  let transition : Tensor α [nStates, nStates] :=
    Tensor.dim (fun _ => uniformVec (α := α) (n := nStates))
  let emission : Tensor α [nStates, nObservations] :=
    Tensor.dim (fun _ => uniformVec (α := α) (n := nObservations))
  { initial, transition, emission }

/-- Log-likelihood of an observation sequence.

We compute this from the same scaling factors used in the EM implementation:
$$
\log p(x_{0:T-1})=\sum_t\log c_t.
$$
-/
def hmmLogLikelihoodSpec {nStates nObservations : Nat}
  (m : HMMSpec α nStates nObservations)
  {length : Nat} (observations : ObservationSeq nObservations length) :
  Option α :=
  if length = 0 then
      let s := sumSpec m.initial
      if s > 0 then some (MathFunctions.log s) else none
  else
      let steps := hmmForwardScaled (α := α) m observations
      logScales? (α := α) (steps.map (·.scale))

end Spec
