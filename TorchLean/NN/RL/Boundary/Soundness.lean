/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.RL.Boundary
public import TorchLean.Tensor.Tactic.Except

/-!
# RL Trust-Boundary Proofs

`TorchLean.NN.RL.Boundary` provides executable “trust-boundary” checkers for externally
supplied RL data. They return `Except String ...`, so runtime code can fail fast with a clear error
message. JSON rollout parsing lives one layer higher, in `TorchLean.NN.RL.Boundary.Json`,
and the proofs here need only the contract and what each checker tests.

Proofs need a Prop-level statement of what was checked, which `RL.Boundary` defines:

- `TorchLean.RL.Boundary.ObservationHolds`
- `TorchLean.RL.Boundary.RewardHolds`
- `TorchLean.RL.Boundary.DoneFlagsHolds`
- `TorchLean.RL.Boundary.ContractHolds`

Each bridge lemma turns a successful executable check (`= .ok ...`) into the matching Prop-level
statement. We proved them by hand by unfolding the checkers.

References:

- Meyer, *Object-Oriented Software Construction* (2nd ed., Prentice Hall, 1997): “Design by
  Contract” as a general specification pattern at software boundaries.
- Gymnasium API reference (reset/step, terminated vs truncated): https://gymnasium.farama.org/
- Sutton and Barto, *Reinforcement Learning: An Introduction* (2nd ed., MIT Press, 2018),
  Sections 3.3 and 3.4, on episodes and the absorbing terminal state:
  http://incompleteideas.net/book/the-book-2nd.html
-/

@[expose] public section

namespace TorchLean
namespace RL
namespace Boundary

/-!
## Bridge lemmas: executable checks -> Prop contract
-/

/-- If the executable checker `checkDoneFlags` succeeds, then the Prop-level done-flag contract
holds. -/
theorem checkDoneFlags_sound {obsShape : Shape} {nActions : Nat}
    (c : Contract obsShape nActions) (terminated truncated : Bool)
    (h : checkDoneFlags (obsShape := obsShape) (nActions := nActions) c terminated truncated
      = .ok ()) :
    DoneFlagsHolds (obsShape := obsShape) (nActions := nActions) c terminated truncated := by
  intro hReq hBoth
  -- Under these hypotheses, the boolean guard in `checkDoneFlags` is true, so the checker
  -- cannot be `.ok ()`.
  have hNe :
      checkDoneFlags (obsShape := obsShape) (nActions := nActions) c terminated truncated
        ≠ .ok () := by
    simp [checkDoneFlags, hReq, hBoth.1, hBoth.2]
  exact hNe h

/-- If the executable checker `checkObservation` succeeds, then the Prop-level observation
contract holds. -/
theorem checkObservation_sound {obsShape : Shape} {nActions : Nat}
    (c : Contract obsShape nActions) (field : String) (obs : Tensor Float obsShape)
    (h : checkObservation (obsShape := obsShape) (nActions := nActions) c (field := field)
      (obs := obs) = .ok ()) :
    ObservationHolds (obsShape := obsShape) (nActions := nActions) c obs := by
  constructor
  · intro hFiniteSwitch
    cases hFinite : obs.all Context.isFinite with
    | false =>
        simp [checkObservation, hFiniteSwitch, hFinite] at h
        cases h
    | true => rfl
  · cases hRange : c.obsRange? with
    | none => trivial
    | some lohi =>
        obtain ⟨lo, hi⟩ := lohi
        cases hIn : obs.all (fun x => lo ≤ x && x ≤ hi) with
        | true => exact hIn
        | false =>
            cases hCF : c.checkObsFinite
            · simp [checkObservation, hCF, hRange, hIn] at h
            · cases hFinite : obs.all Context.isFinite
              · simp [checkObservation, hCF, hFinite] at h
                cases h
              · simp [checkObservation, hCF, hFinite, hRange, hIn] at h

/-- If the executable checker `checkReward` succeeds, then the Prop-level reward contract holds. -/
theorem checkReward_sound {obsShape : Shape} {nActions : Nat}
    (c : Contract obsShape nActions) (reward : Float)
    (h : checkReward (obsShape := obsShape) (nActions := nActions) c reward = .ok ()) :
    RewardHolds (obsShape := obsShape) (nActions := nActions) c reward := by
  constructor
  · intro hFiniteSwitch
    cases hFinite : Context.requireWithin "RL boundary: reward" none none reward with
    | error e =>
        simp [checkReward, hFiniteSwitch, hFinite] at h
        cases h
    | ok u => simpa using Context.requireWithin_eq_ok_iff.mp hFinite
  · cases hRange : c.rewardRange? with
    | none => trivial
    | some lohi =>
        obtain ⟨lo, hi⟩ := lohi
        show lo ≤ reward ∧ reward ≤ hi
        cases hIn : Context.requireWithin "RL boundary: reward" (some lo) (some hi) reward with
        | ok u => simpa using Context.requireWithin_eq_ok_iff.mp hIn
        | error e =>
            cases hCF : c.checkRewardFinite <;>
              cases hFinite : Context.requireWithin "RL boundary: reward" none none reward <;>
              simp [checkReward, hCF, hFinite, hRange, hIn] at h <;> cases h

/--
If the executable checker `checkTransitionFin` returns `.ok t`, then `t` satisfies the Prop-level
contract `ContractHolds c t`.

`checkTransitionFinWithProof` returns this proof together with the checked transition.
-/
theorem checkTransitionFin_sound {obsShape : Shape} {nActions : Nat}
    (c : Contract obsShape nActions)
    (observation nextObservation : Tensor Float obsShape)
    (action : Fin nActions)
    (reward : Float)
    (terminated truncated : Bool)
    (t : Transition obsShape nActions)
    (h :
      checkTransitionFin (obsShape := obsShape) (nActions := nActions) c observation
        nextObservation action reward terminated truncated = .ok t) :
    ContractHolds (obsShape := obsShape) (nActions := nActions) c t := by
  unfold checkTransitionFin at h
  except_cases hDone : checkDoneFlags c terminated truncated using h with u =>
    cases u
    simp only [hDone, Bind.bind, Except.bind] at h
    except_cases hReward : checkReward c reward using h with u =>
      cases u
      simp only [hReward] at h
      except_cases hObs : checkObservation c (field := "observation") observation
          using h with u =>
        cases u
        simp only [hObs] at h
        except_cases hNextObs : checkObservation c (field := "nextObservation") nextObservation
            using h with u =>
          cases u
          simp only [hNextObs, Pure.pure, Except.pure, Except.ok.injEq] at h
          cases h
          exact
            { doneFlags := checkDoneFlags_sound c terminated truncated hDone
              reward := checkReward_sound c reward hReward
              observation := checkObservation_sound c "observation"
                observation hObs
              nextObservation := checkObservation_sound c "nextObservation"
                nextObservation hNextObs }

/--
Run `checkTransitionFin` and, on success, return the transition bundled with its `ContractHolds`
proof.

A caller checks the contract at run time and gets a usable hypothesis back, so no downstream proof
or program has to assume the contract.
-/
def checkTransitionFinWithProof {obsShape : Shape} {nActions : Nat}
    (c : Contract obsShape nActions)
    (observation nextObservation : Tensor Float obsShape)
    (action : Fin nActions)
    (reward : Float)
    (terminated truncated : Bool) :
    Except String
      {t : Transition obsShape nActions //
        ContractHolds (obsShape := obsShape) (nActions := nActions) c t} :=
  match h :
      checkTransitionFin (obsShape := obsShape) (nActions := nActions) c observation
        nextObservation action reward terminated truncated with
  | .ok t => .ok ⟨t, checkTransitionFin_sound (c := c) (observation := observation)
      (nextObservation := nextObservation) (action := action) (reward := reward)
      (terminated := terminated) (truncated := truncated) (t := t) h⟩
  | .error e => .error e

end Boundary
end RL
end TorchLean
