/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Models.Hmm
public import NN.Spec.Module.Core

/-!
# HMM adapters as `Spec.Module`s

The HMM spec model (`NN/Spec/Models/Hmm.lean`) uses discrete observations (`Fin nObservations`).

For composition and examples, it is sometimes convenient to accept a tensor of scores/probabilities
over the observation alphabet and decode each timestep via `argmax`. The wrappers in this file
provide that bridge and package the resulting behavior as `Spec.Module`s.
-/

@[expose] public section


namespace Spec.Module

open TorchLean TorchLean.Tensor

variable {α : Type} [TorchLean.Storage α] [Context α]

/-- Decode a single observation vector into a discrete symbol by taking `argmax`. -/
def decodeObservation
  {nObservations : Nat} (hObservations : nObservations > 0)
  (scores : Tensor α [nObservations]) : Fin nObservations :=
  Fin.cast (by simp [Shape.size])
    (argmax (s := [nObservations]) (by simpa [Shape.size] using hObservations) scores)

/-- Convert a tensor of per-symbol scores/probabilities into a discrete observation sequence by
decoding each timestep with `argmax`. -/
def decodeObservations
  {seqLen nObservations : Nat} (hObservations : nObservations > 0)
  (scores : Tensor α [seqLen, nObservations]) :
  Tensor (Fin nObservations) [seqLen] :=
  TorchLean.Tensor.ofFn fun t => decodeObservation hObservations (get scores t)

/-- A one-step HMM module: map an observation distribution to a filtered state distribution. -/
def hmm {nStates nObservations : Nat}
  (hObservations : nObservations > 0) (m : HMMSpec α nStates nObservations) :
  Spec.Module α ([nObservations]) ([nStates]) :=
{
  forward := fun scores =>
    let observation := decodeObservation hObservations scores
    let unnormalized := mulSpec m.initial (emissionVec (α := α) m observation)
    (normalizeVec (α := α) unnormalized).1,
  kind := "HMM",
  pythonExpr := "UnsupportedLayer(\"HMM\", \"torch.distributions.Categorical\")"
}

/-- Forward messages `α_t` for each timestep (scaled). -/
def forwardMessages
  {nStates nObservations length : Nat}
  (m : HMMSpec α nStates nObservations)
  (observations : ObservationSeq nObservations length) :
  Tensor (Tensor α [nStates]) [length] :=
  (hmmForwardScaled (α := α) m observations).map (·.message)

/-- Sequence module: compute forward messages `α_t` for each timestep. -/
def hmmSequence {seqLen nStates nObservations : Nat}
  (hObservations : nObservations > 0) (m : HMMSpec α nStates nObservations) :
  Spec.Module α ([seqLen, nObservations]) ([seqLen, nStates]) :=
{
  forward := fun scores =>
    let observations := decodeObservations hObservations scores
    let messages := forwardMessages (α := α) m observations
    Tensor.dim fun t => messages.getScalar t,
  kind := "HMMSequence",
  pythonExpr := "UnsupportedLayer(\"HMMSequence\", \"torch.distributions.Categorical\")"
}

/-- Sequence module: compute prefix likelihoods `p(o₀:t)` for each timestep `t`. -/
def hmmPrefixLikelihoods {seqLen nStates nObservations : Nat}
  (hObservations : nObservations > 0) (m : HMMSpec α nStates nObservations) :
  Spec.Module α ([seqLen, nObservations]) ([seqLen]) :=
{
  forward := fun scores =>
    let observations := decodeObservations hObservations scores
    Tensor.dim (fun t =>
      let prefixObservations : ObservationSeq nObservations (t.val + 1) :=
        Tensor.ofFn fun i => observations.getScalar ⟨i.val, by grind⟩
      Tensor.scalar (hmmForwardSpec (α := α) m prefixObservations)
    ),
  kind := "HMMPrefixLikelihoods",
  pythonExpr := "UnsupportedLayer(\"HMMPrefixLikelihoods\", \"torch.distributions.Categorical\")"
}

/-- Sequence module: normalized state probabilities at each timestep. -/
def hmmStateProbabilities {seqLen nStates nObservations : Nat}
  (hObservations : nObservations > 0) (m : HMMSpec α nStates nObservations) :
  Spec.Module α ([seqLen, nObservations]) ([seqLen, nStates]) :=
{
  forward := fun scores =>
    let observations := decodeObservations hObservations scores
    let messages := forwardMessages (α := α) m observations
    Tensor.dim (fun t =>
      let message := messages.getScalar t
      let total := sumSpec message
      if total > 0 then
        Tensor.dim (fun s =>
          Tensor.scalar (message.getScalar s / total)
        )
      else
        Tensor.dim (fun _ => Tensor.scalar (1 / nStates))
    ),
  kind := "HMMStateProbabilities",
  pythonExpr := "UnsupportedLayer(\"HMMStateProbabilities\", \"torch.distributions.Categorical\")"
}

/-- Apply `hmm` independently at each timestep, using the initial distribution for every row.

Each row is decoded with `argmax`, as in the one-step module. The output contains filtered state
probabilities, with the same totalization for impossible observations as `hmm`.
-/
def hmmIndependent {seqLen nStates nObservations : Nat}
  (hObservations : nObservations > 0) (m : HMMSpec α nStates nObservations) :
  Spec.Module α ([seqLen, nObservations]) ([seqLen, nStates]) :=
{
  forward := (liftLeading (n := seqLen) (hmm hObservations m)).forward,
  kind := "HMMIndependent",
  pythonExpr := "UnsupportedLayer(\"HMMIndependent\", \"torch.distributions.Categorical\")"
}

end Spec.Module
