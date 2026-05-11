/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.RL.Boundary.Core
public import NN.Proofs.RL.Tactics

/-!
# RL Trust-Boundary Proofs

`NN.Runtime.RL.Boundary.Core` provides executable “trust-boundary” checkers for externally supplied
RL data. Those checkers return `Except String ...` so runtime code can fail fast with clear error
messages. JSON rollout parsing lives one layer higher in `NN.Runtime.RL.Boundary.Json`; the proof
bridge here only needs the contract and checker semantics.

For formal reasoning, we also want a Prop-level view of exactly what was checked:

- `Runtime.RL.Boundary.ObservationHolds`
- `Runtime.RL.Boundary.RewardHolds`
- `Runtime.RL.Boundary.DoneFlagsHolds`
- `Runtime.RL.Boundary.ContractHolds`

This module provides the bridge lemmas that turn a successful executable check (`= .ok ...`) into
the corresponding Prop-level statement.

References:

- Meyer, *Object-Oriented Software Construction* (2nd ed., 1997): “Design by Contract” as a general
  specification pattern at software boundaries.
- Gymnasium API docs (reset/step, terminated vs truncated): https://gymnasium.farama.org/
- Sutton and Barto, *Reinforcement Learning: An Introduction* (2nd ed., 2018),
  discussion of episodic termination semantics:
  http://incompleteideas.net/book/the-book-2nd.html
-/

@[expose] public section

namespace Proofs
namespace RL
namespace Boundary

open Runtime.RL.Boundary

/-!
## Internal helper lemmas

These facts are purely about the small executable helpers in `Runtime.RL.Boundary.Internal`.
-/

private theorem tensorFinite_of_checkTensorFinite_eq_ok {s : Spec.Shape} (field : String)
    (t : Spec.Tensor Float s)
    (h : Internal.checkTensorFinite (s := s) (field := field) t = .ok ()) :
    tensorFinite (s := s) t = true := by
  -- `checkTensorFinite` is just an `if` on `tensorFinite`.
  cases hFinite : tensorFinite (s := s) t with
  | false =>
      simp [Internal.checkTensorFinite, hFinite] at h
  | true =>
      simp

private theorem tensorInClosedInterval_of_checkTensorRange_eq_ok {s : Spec.Shape} (field : String)
    (lo hi : Float) (t : Spec.Tensor Float s)
    (h : Internal.checkTensorRange (s := s) (field := field) lo hi t = .ok ()) :
    tensorInClosedInterval (s := s) lo hi t = true := by
  cases hOk : tensorInClosedInterval (s := s) lo hi t with
  | false =>
      simp [Internal.checkTensorRange, hOk] at h
  | true =>
      simp

private theorem isFiniteFloat_of_checkFloatFinite_eq_ok (field : String) (x : Float)
    (h : Internal.checkFloatFinite (field := field) x = .ok ()) :
    isFiniteFloat x = true := by
  cases hOk : isFiniteFloat x with
  | false =>
      simp [Internal.checkFloatFinite, hOk] at h
  | true =>
      simp

private theorem floatInClosedInterval_of_checkFloatRange_eq_ok (field : String) (lo hi x : Float)
    (h : Internal.checkFloatRange (field := field) lo hi x = .ok ()) :
    lo ≤ x ∧ x ≤ hi := by
  have hBool : (lo ≤ x && x ≤ hi) = true := by
    cases hOk : (lo ≤ x && x ≤ hi) with
    | false =>
        simp [Internal.checkFloatRange, hOk] at h
    | true =>
        simp
  have hAnd : decide (lo ≤ x) = true ∧ decide (x ≤ hi) = true := by
    have : (decide (lo ≤ x) && decide (x ≤ hi)) = true := by
      simpa using hBool
    exact (Bool.and_eq_true_iff).1 this
  exact ⟨of_decide_eq_true hAnd.1, of_decide_eq_true hAnd.2⟩

/-!
## Bridge lemmas: executable checks -> Prop contract
-/

/-- If the executable checker `checkDoneFlags` succeeds, then the Prop-level done-flag contract holds. -/
theorem doneFlagsHolds_of_checkDoneFlags_eq_ok {obsShape : Spec.Shape} {nActions : Nat}
    (c : Contract obsShape nActions) (terminated truncated : Bool)
    (h : checkDoneFlags (obsShape := obsShape) (nActions := nActions) c terminated truncated = .ok ()) :
    DoneFlagsHolds (obsShape := obsShape) (nActions := nActions) c terminated truncated := by
  intro hReq hBoth
  -- Under these hypotheses, the boolean guard in `checkDoneFlags` is true, so the checker cannot be `.ok ()`.
  have hNe : checkDoneFlags (obsShape := obsShape) (nActions := nActions) c terminated truncated ≠ .ok () := by
    -- Unfold and simplify the guard to `true`.
    simp [checkDoneFlags, hReq, hBoth.1, hBoth.2]
  exact hNe h

/-- If the executable checker `checkObservation` succeeds, then the Prop-level observation contract holds. -/
theorem observationHolds_of_checkObservation_eq_ok {obsShape : Spec.Shape} {nActions : Nat}
    (c : Contract obsShape nActions) (field : String) (obs : Spec.Tensor Float obsShape)
    (h : checkObservation (obsShape := obsShape) (nActions := nActions) c (field := field) (obs := obs) = .ok ()) :
    ObservationHolds (obsShape := obsShape) (nActions := nActions) c obs := by
  constructor
  · intro hFiniteSwitch
    -- With `checkObsFinite=true`, `checkObservation` must have successfully run `checkTensorFinite`.
    have h' :
        (do
          Internal.checkTensorFinite (s := obsShape) (field := field) obs
          match c.obsRange? with
          | none => pure ()
          | some (lo, hi) => Internal.checkTensorRange (s := obsShape) (field := field) lo hi obs) = .ok () := by
      simpa [checkObservation, hFiniteSwitch] using h
    except_cases hFinite : Internal.checkTensorFinite (s := obsShape) (field := field) obs using h' with u =>
      cases u
      exact
        tensorFinite_of_checkTensorFinite_eq_ok (s := obsShape) (field := field) (t := obs)
          (by simpa using hFinite)
  · -- Range check component.
    cases hRange : c.obsRange? with
    | none =>
        simp
    | some lohi =>
        rcases lohi with ⟨lo, hi⟩
        -- Extract the success of `checkTensorRange` from the overall success of `checkObservation`.
        cases hCF : c.checkObsFinite with
        | false =>
            have hRangeOk :
                Internal.checkTensorRange (s := obsShape) (field := field) lo hi obs = .ok () := by
              simpa [checkObservation, hCF, hRange] using h
            exact
              tensorInClosedInterval_of_checkTensorRange_eq_ok (s := obsShape) (field := field)
                (lo := lo) (hi := hi) (t := obs) hRangeOk
        | true =>
            have h' :
                (do
                  Internal.checkTensorFinite (s := obsShape) (field := field) obs
                  Internal.checkTensorRange (s := obsShape) (field := field) lo hi obs) = .ok () := by
              simpa [checkObservation, hCF, hRange] using h
            cases hFinite : Internal.checkTensorFinite (s := obsShape) (field := field) obs with
            | error e =>
                cases (by simpa [hFinite] using h')
            | ok u =>
                cases u
                have hRangeOk :
                    Internal.checkTensorRange (s := obsShape) (field := field) lo hi obs = .ok () := by
                  -- After the first check succeeds, the do-chain reduces to the second check.
                  have : Internal.checkTensorRange (s := obsShape) (field := field) lo hi obs = .ok () := by
                    simpa [hFinite] using h'
                  exact this
                exact
                  tensorInClosedInterval_of_checkTensorRange_eq_ok (s := obsShape) (field := field)
                    (lo := lo) (hi := hi) (t := obs) hRangeOk

/-- If the executable checker `checkReward` succeeds, then the Prop-level reward contract holds. -/
theorem rewardHolds_of_checkReward_eq_ok {obsShape : Spec.Shape} {nActions : Nat}
    (c : Contract obsShape nActions) (reward : Float)
    (h : checkReward (obsShape := obsShape) (nActions := nActions) c reward = .ok ()) :
    RewardHolds (obsShape := obsShape) (nActions := nActions) c reward := by
  constructor
  · intro hFiniteSwitch
    have h' :
        (do
          Internal.checkFloatFinite (field := "reward") reward
          match c.rewardRange? with
          | none => pure ()
          | some (lo, hi) => Internal.checkFloatRange (field := "reward") lo hi reward) = .ok () := by
      simpa [checkReward, hFiniteSwitch] using h
    except_cases hFinite : Internal.checkFloatFinite (field := "reward") reward using h' with u =>
      cases u
      exact
        isFiniteFloat_of_checkFloatFinite_eq_ok (field := "reward") (x := reward)
          (by simpa using hFinite)
  · cases hRange : c.rewardRange? with
    | none =>
        simp
    | some lohi =>
        rcases lohi with ⟨lo, hi⟩
        -- Extract `checkFloatRange` success.
        cases hCF : c.checkRewardFinite with
        | false =>
            have hRangeOk : Internal.checkFloatRange (field := "reward") lo hi reward = .ok () := by
              simpa [checkReward, hCF, hRange] using h
            exact floatInClosedInterval_of_checkFloatRange_eq_ok (field := "reward")
              (lo := lo) (hi := hi) (x := reward) hRangeOk
        | true =>
            have h' :
                (do
                  Internal.checkFloatFinite (field := "reward") reward
                  Internal.checkFloatRange (field := "reward") lo hi reward) = .ok () := by
              simpa [checkReward, hCF, hRange] using h
            except_cases hFinite : Internal.checkFloatFinite (field := "reward") reward using h' with u =>
              cases u
              have hRangeOk : Internal.checkFloatRange (field := "reward") lo hi reward = .ok () := by
                simpa [hFinite] using h'
              exact floatInClosedInterval_of_checkFloatRange_eq_ok (field := "reward")
                (lo := lo) (hi := hi) (x := reward) hRangeOk

/--
If the executable checker `checkTransitionFin` succeeds, then the Prop-level contract holds.

This is the main “checked preconditions” bridge used by downstream RL theorems.
-/
theorem contractHolds_of_checkTransitionFin_eq_ok {obsShape : Spec.Shape} {nActions : Nat}
    (c : Contract obsShape nActions)
    (observation nextObservation : Spec.Tensor Float obsShape)
    (action : Fin nActions)
    (reward : Float)
    (terminated truncated : Bool)
    (t : Transition obsShape nActions)
    (h :
      checkTransitionFin (obsShape := obsShape) (nActions := nActions) c observation nextObservation action
        reward terminated truncated = .ok t) :
    ContractHolds (obsShape := obsShape) (nActions := nActions) c t := by
  -- Peel the `Except` do-chain to recover each successful sub-check and the returned record.
  unfold checkTransitionFin at h
  cases hDone : checkDoneFlags (obsShape := obsShape) (nActions := nActions) c terminated truncated with
  | error e =>
      cases (by simpa [hDone] using h)
  | ok u =>
      cases u
      have h' :
          (do
            checkReward (obsShape := obsShape) (nActions := nActions) c reward
            checkObservation (obsShape := obsShape) (nActions := nActions) c (field := "observation") observation
            checkObservation (obsShape := obsShape) (nActions := nActions) c (field := "nextObservation") nextObservation
            pure
              ({ observation := observation
                 action := action
                 reward := reward
                 nextObservation := nextObservation
                 terminated := terminated
                 truncated := truncated } : Transition obsShape nActions)) = .ok t := by
        simpa [hDone] using h
      cases hReward : checkReward (obsShape := obsShape) (nActions := nActions) c reward with
      | error e =>
          cases (by simpa [hReward] using h')
      | ok uR =>
          cases uR
          have h'' :
              (do
                checkObservation (obsShape := obsShape) (nActions := nActions) c (field := "observation") observation
                checkObservation (obsShape := obsShape) (nActions := nActions) c (field := "nextObservation") nextObservation
                pure
                  ({ observation := observation
                     action := action
                     reward := reward
                     nextObservation := nextObservation
                     terminated := terminated
                     truncated := truncated } : Transition obsShape nActions)) = .ok t := by
            simpa [hReward] using h'
          cases hObs : checkObservation (obsShape := obsShape) (nActions := nActions) c
              (field := "observation") observation with
          | error e =>
              cases (by simpa [hObs] using h'')
          | ok uO =>
              cases uO
              have h''' :
                  (do
                    checkObservation (obsShape := obsShape) (nActions := nActions) c (field := "nextObservation") nextObservation
                    pure
                      ({ observation := observation
                         action := action
                         reward := reward
                         nextObservation := nextObservation
                         terminated := terminated
                         truncated := truncated } : Transition obsShape nActions)) = .ok t := by
                simpa [hObs] using h''
              cases hNextObs : checkObservation (obsShape := obsShape) (nActions := nActions) c
                  (field := "nextObservation") nextObservation with
              | error e =>
                  have hContra := h'''
                  simp [hNextObs] at hContra
              | ok uNO =>
                  cases uNO
                  have hFinal :
                      (.ok
                        { observation := observation
                          action := action
                          reward := reward
                          nextObservation := nextObservation
                          terminated := terminated
                          truncated := truncated } :
                        Except String (Transition obsShape nActions)) = .ok t := by
                    simpa [hNextObs] using h'''
                  have ht :
                      t =
                        { observation := observation
                          action := action
                          reward := reward
                          nextObservation := nextObservation
                          terminated := terminated
                          truncated := truncated } := by
                    cases hFinal
                    rfl

                  -- Assemble `ContractHolds`.
                  refine ht ▸ ?_
                  refine
                    { doneFlags :=
                        doneFlagsHolds_of_checkDoneFlags_eq_ok (c := c) (terminated := terminated)
                          (truncated := truncated)
                          (by simpa using hDone)
                      reward :=
                        rewardHolds_of_checkReward_eq_ok (c := c) (reward := reward) (by simpa using hReward)
                      observation :=
                        observationHolds_of_checkObservation_eq_ok (c := c) (field := "observation")
                          (obs := observation) (by simpa using hObs)
                      nextObservation :=
                        observationHolds_of_checkObservation_eq_ok (c := c) (field := "nextObservation")
                          (obs := nextObservation) (by simpa using hNextObs) }

/--
Convenience wrapper: run the executable checker and, on success, return the transition bundled
with the Prop-level contract proof.

This is the “checked preconditions” interface for downstream proofs/programs:
instead of assuming a contract, you explicitly *check* it and obtain a usable hypothesis.
-/
def checkTransitionFinWithProof {obsShape : Spec.Shape} {nActions : Nat}
    (c : Contract obsShape nActions)
    (observation nextObservation : Spec.Tensor Float obsShape)
    (action : Fin nActions)
    (reward : Float)
    (terminated truncated : Bool) :
    Except String {t : Transition obsShape nActions // ContractHolds (obsShape := obsShape) (nActions := nActions) c t} :=
  match h :
      checkTransitionFin (obsShape := obsShape) (nActions := nActions) c observation nextObservation action
        reward terminated truncated with
  | .ok t => .ok ⟨t, contractHolds_of_checkTransitionFin_eq_ok (c := c) (observation := observation)
      (nextObservation := nextObservation) (action := action) (reward := reward)
      (terminated := terminated) (truncated := truncated) (t := t) h⟩
  | .error e => .error e

end Boundary
end RL
end Proofs
