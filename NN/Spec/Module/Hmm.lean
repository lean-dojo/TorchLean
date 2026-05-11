/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Models.Hmm
public import NN.Spec.Module.SpecModule

/-!
# HMM adapters as `NNModuleSpec`s

The HMM spec model (`NN/Spec/Models/Hmm.lean`) uses discrete observations (`Fin nObservations`).

For composition and demos, it is sometimes convenient to accept a tensor of scores/probabilities
over the observation alphabet and decode each timestep via `argmax`. The wrappers in this file
provide that bridge and package the resulting behavior as `NNModuleSpec`s.
-/

@[expose] public section


namespace Spec

open Tensor
open ModSpec

variable {α : Type} [Context α]

/-- Decode a single observation vector into a discrete symbol by taking `argmax`. -/
def tensorToDiscreteObs
  {nObservations : Nat} [Inhabited (Fin nObservations)] (h_nobs : nObservations > 0)
  (obs_tensor : Tensor α (.dim nObservations .scalar)) : Fin nObservations :=
  let rec find_argmax (i : Nat) (max_val : α) (max_idx : Fin nObservations) : Fin nObservations :=
    if h : i < nObservations then
      let current_val :=
        match get obs_tensor ⟨i, h⟩ with
        | Tensor.scalar val => val
      if current_val > max_val then
        find_argmax (i + 1) current_val ⟨i, h⟩
      else
        find_argmax (i + 1) max_val max_idx
    else
      max_idx
  let first_val :=
    match get obs_tensor ⟨0, h_nobs⟩ with
    | Tensor.scalar val => val
  find_argmax 1 first_val ⟨0, h_nobs⟩

/-- Convert a tensor of per-symbol scores/probabilities into a discrete observation sequence by
decoding each timestep with `argmax`. -/
def tensorSeqToDiscreteSeq
  {seqLen nObservations : Nat} [Inhabited (Fin nObservations)] (h_nobs : nObservations > 0)
  (obs_seq_tensor : Tensor α (.dim seqLen (.dim nObservations .scalar))) : ObservationSeq
    nObservations :=
  let rec convert_seq (t : Nat) (acc : List (Fin nObservations)) : List (Fin nObservations) :=
    if h : t < seqLen then
      let obs_tensor := get obs_seq_tensor ⟨t, h⟩
      let discrete_obs := tensorToDiscreteObs h_nobs obs_tensor
      convert_seq (t + 1) (discrete_obs :: acc)
    else
      acc.reverse
  convert_seq 0 []

/-- A one-step HMM module: map an observation distribution to a filtered state distribution. -/
def HMMModuleSpec {nStates nObservations : Nat} [Inhabited (Fin nObservations)]
  (h_nobs : nObservations > 0) (m : HMMSpec α nStates nObservations) :
  NNModuleSpec α (.dim nObservations .scalar) (.dim nStates .scalar) :=
{
  forward := fun obs_tensor =>
    let discrete_obs := tensorToDiscreteObs h_nobs obs_tensor
    let alpha0_raw := mulSpec m.init_prob (emissionVec (α := α) m discrete_obs)
    (normalizeVec (α := α) alpha0_raw).1,
  kind := "HMM",
  export_func := {
    toPyTorch := "UnsupportedLayer(\"HMM\", \"torch.distributions.Categorical\")",
    dimensions := (nObservations, nStates)
  }
}

/-- Forward messages `α_t` for each timestep (scaled). -/
def hmmForwardPassSpec
  {nStates nObservations : Nat} [Inhabited (Fin nObservations)]
  (m : HMMSpec α nStates nObservations)
  (observations : ObservationSeq nObservations) :
  List (Tensor α (.dim nStates .scalar)) :=
  (hmmForwardScaled (α := α) m observations).1

/-- Sequence module: compute forward messages `α_t` for each timestep. -/
def HMMSeqModuleSpec {seqLen nStates nObservations : Nat} [Inhabited (Fin nObservations)]
  (h_nobs : nObservations > 0) (m : HMMSpec α nStates nObservations) :
  NNModuleSpec α (.dim seqLen (.dim nObservations .scalar)) (.dim seqLen (.dim nStates .scalar)) :=
{
  forward := fun obs_seq_tensor =>
    let discrete_seq := tensorSeqToDiscreteSeq h_nobs obs_seq_tensor
    let alpha_values := hmmForwardPassSpec (α := α) m discrete_seq
    Tensor.dim (fun t =>
      if _ : t.val < seqLen then
        match alpha_values[t.val]? with
        | some alpha => alpha
        | none => Tensor.dim (fun _ => Tensor.scalar (0 : α))
      else
        Tensor.dim (fun _ => Tensor.scalar (0 : α))
    ),
  kind := "HMMSeq",
  export_func := {
    toPyTorch := "UnsupportedLayer(\"HMMSeq\", \"torch.distributions.Categorical\")",
    dimensions := (nObservations, nStates)
  }
}

/-- Sequence module: compute prefix likelihoods `p(o₀:t)` for each timestep `t`. -/
def HMMSeqLikelihoodModuleSpec {seqLen nStates nObservations : Nat} [Inhabited (Fin nObservations)]
  (h_nobs : nObservations > 0) (m : HMMSpec α nStates nObservations) :
  NNModuleSpec α (.dim seqLen (.dim nObservations .scalar)) (.dim seqLen .scalar) :=
{
  forward := fun obs_seq_tensor =>
    let discrete_seq := tensorSeqToDiscreteSeq h_nobs obs_seq_tensor
    Tensor.dim (fun t =>
      if _ : t.val < seqLen then
        let partial_seq := discrete_seq.take (t.val + 1)
        let likelihood := hmmForwardSpec (α := α) m partial_seq
        Tensor.scalar likelihood
      else
        Tensor.scalar (0 : α)
    ),
  kind := "HMMSeqLikelihood",
  export_func := {
    toPyTorch := "UnsupportedLayer(\"HMMSeqLikelihood\", \"torch.distributions.Categorical\")",
    dimensions := (nObservations, nStates)
  }
}

/-- Sequence module: normalized state probabilities at each timestep. -/
def HMMSeqStateProbModuleSpec {seqLen nStates nObservations : Nat} [Inhabited (Fin nObservations)]
  (h_nobs : nObservations > 0) (m : HMMSpec α nStates nObservations) :
  NNModuleSpec α (.dim seqLen (.dim nObservations .scalar)) (.dim seqLen (.dim nStates .scalar)) :=
{
  forward := fun obs_seq_tensor =>
    let discrete_seq := tensorSeqToDiscreteSeq h_nobs obs_seq_tensor
    let alpha_values := hmmForwardPassSpec (α := α) m discrete_seq
    Tensor.dim (fun t =>
      if _ : t.val < seqLen then
        match alpha_values[t.val]? with
        | some alpha =>
          let total := sumSpec alpha
          if total > 0 then
            Tensor.dim (fun s =>
              match get alpha s with
              | Tensor.scalar val => Tensor.scalar (val / total)
            )
          else
            Tensor.dim (fun _ => Tensor.scalar (1 / nStates))
        | none => Tensor.dim (fun _ => Tensor.scalar (1 / nStates))
      else
        Tensor.dim (fun _ => Tensor.scalar (1 / nStates))
    ),
  kind := "HMMSeqStateProb",
  export_func := {
    toPyTorch := "UnsupportedLayer(\"HMMSeqStateProb\", \"torch.distributions.Categorical\")",
    dimensions := (nObservations, nStates)
  }
}

/-- Sequence module: apply the one-step update independently at each timestep. -/
def HMMSeqIndependentModuleSpec {seqLen nStates nObservations : Nat} [Inhabited (Fin nObservations)]
  (h_nobs : nObservations > 0) (m : HMMSpec α nStates nObservations) :
  NNModuleSpec α (.dim seqLen (.dim nObservations .scalar)) (.dim seqLen (.dim nStates .scalar)) :=
{
  forward := fun obs_seq_tensor =>
    Tensor.dim (fun t =>
      let obs_tensor := get obs_seq_tensor t
      let discrete_obs := tensorToDiscreteObs h_nobs obs_tensor
      let obs_seq : ObservationSeq nObservations := [discrete_obs]
      let likelihood := hmmForwardSpec (α := α) m obs_seq
      Tensor.dim (fun _s => Tensor.scalar likelihood)
    ),
  kind := "HMMSeqIndependent",
  export_func := {
    toPyTorch := "UnsupportedLayer(\"HMMSeqIndependent\", \"torch.distributions.Categorical\")",
    dimensions := (nObservations, nStates)
  }
}

end Spec
