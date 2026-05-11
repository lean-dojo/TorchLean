/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.RL.Gymnasium
public import NN.Proofs.RL.Boundary

/-!
# Gymnasium Trust-Boundary Wrappers (Proof Layer)

`NN.Runtime.RL.Gymnasium.Session.stepChecked` returns a contract-validated transition, but the
runtime layer intentionally does not *carry* the Prop-level proof object in its return type.

This file provides a proof-layer wrapper that returns the same transition bundled with a proof
that it satisfies the Lean-side trust-boundary contract (`Runtime.RL.Boundary.ContractHolds`).

This is useful when you want a “checked preconditions” API: instead of assuming numeric/shape
safety of externally collected rollouts, you explicitly check and obtain a usable hypothesis.

References:

- Gymnasium API docs (reset/step, terminated vs truncated): https://gymnasium.farama.org/
-/

@[expose] public section

namespace Runtime
namespace RL
namespace Gymnasium

open Spec
open Tensor

namespace Session

/--
Like `stepChecked`, but returns the transition bundled with a Prop-level proof that it satisfies
the client’s trust-boundary contract.
-/
def stepCheckedWithProof {obsShape : Shape} {nActions : Nat}
    (s : Session obsShape nActions) (action : Fin nActions) (resetOnDone : Bool := true) :
    IO ({t : Boundary.Transition obsShape nActions //
          Boundary.ContractHolds (obsShape := obsShape) (nActions := nActions) s.client.contract t} ×
        Session obsShape nActions) := do
  let obs := s.observation
  let (obs', reward, terminated, truncated) ← s.client.stepRaw (action := action.1)
  match h :
      Boundary.checkTransitionFin (obsShape := obsShape) (nActions := nActions) s.client.contract
        obs obs' action reward terminated truncated with
  | .ok t =>
      have ht :
          Boundary.ContractHolds (obsShape := obsShape) (nActions := nActions) s.client.contract t :=
        Proofs.RL.Boundary.contractHolds_of_checkTransitionFin_eq_ok (c := s.client.contract)
          (observation := obs) (nextObservation := obs') (action := action) (reward := reward)
          (terminated := terminated) (truncated := truncated) (t := t) h

      let done : Bool := Boundary.Transition.done t
      let nextObs ←
        if resetOnDone && done then
          s.client.reset
        else
          pure obs'
      pure (⟨t, ht⟩, { s with observation := nextObs })
  | .error e =>
      throw <| IO.userError e

end Session

end Gymnasium
end RL
end Runtime
