/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Backend.Profile
import Batteries.Lean.Except

/-!
# Backend selection soundness

The production planner selects registered capsules according to availability and policy.
These proofs concern selection of declared contracts. They do not establish native kernel
eligibility, numerical correctness, or the behavior of foreign implementations.
-/

@[expose] public section

namespace NN.Backend

/-- A successful choice is registered, implements the requested operation, and is admissible. -/
theorem chooseCapsuleFor_sound
    {policy : KernelPolicy} {op : BackendOp} {registry : Array KernelCapsule}
    {selected : KernelCapsule}
    (hselected : chooseCapsuleFor? policy op registry = some selected) :
    selected ∈ registry ∧ selected.op = op ∧ selected.admissible policy = true := by
  have ordinary
      (hfind : registry.find? (fun capsule =>
        capsule.op == op && capsule.admissible policy) = some selected) :
      selected ∈ registry ∧ selected.op = op ∧ selected.admissible policy = true := by
    refine ⟨Array.mem_of_find?_eq_some hfind, ?_⟩
    simpa only [Bool.and_eq_true, beq_iff_eq] using Array.find?_some hfind
  cases hpolicy : policy.provider with
  | prefer preferred =>
    simp only [chooseCapsuleFor?, hpolicy] at hselected
    cases hfind : registry.find? (fun capsule =>
        (capsule.op == op && capsule.admissible policy) &&
          capsule.provider == preferred) with
    | none =>
        exact ordinary (by simpa [hfind] using hselected)
    | some found =>
        have heq : found = selected := by simpa [hfind] using hselected
        subst found
        refine ⟨Array.mem_of_find?_eq_some hfind, ?_⟩
        have hpredicate := Array.find?_some hfind
        have eligible := (Bool.and_eq_true_iff.mp hpredicate).1
        simpa only [Bool.and_eq_true, beq_iff_eq] using eligible
  | auto => exact ordinary (by simpa [chooseCapsuleFor?, hpolicy] using hselected)
  | only provider => exact ordinary (by simpa [chooseCapsuleFor?, hpolicy] using hselected)

/-- An exclusive provider request never selects a capsule from another provider. -/
theorem chooseCapsuleFor_only_provider
    {policy : KernelPolicy} {op : BackendOp} {registry : Array KernelCapsule}
    {selected : KernelCapsule} {provider : Provider}
    (honly : policy.provider = .only provider)
    (hselected : chooseCapsuleFor? policy op registry = some selected) :
    selected.provider = provider := by
  have hadmissible := (chooseCapsuleFor_sound hselected).2.2
  simp only [KernelCapsule.admissible, Bool.and_eq_true, and_assoc] at hadmissible
  have hprovider := hadmissible.2.2.2.1
  simpa [KernelCapsule.matchesPreference, honly, eq_comm] using hprovider

/--
A successful choice leaves the preferred provider exactly when no available, admissible
candidate of that provider implements the requested operation.
-/
theorem chooseCapsuleFor_prefer_fallback_iff
    (policy : KernelPolicy) (availability : Availability)
    (registry : Array KernelCapsule) (op : BackendOp)
    (preferred : Provider) (selected : KernelCapsule)
    (hprefer : policy.provider = .prefer preferred)
    (hselected :
      chooseCapsuleFor? policy op (availability.filterCapsules registry) = some selected) :
    selected.provider ≠ preferred ↔
      ¬ ∃ capsule ∈ registry,
        availability.admitsCapsule capsule = true ∧
        capsule.op = op ∧
        capsule.admissible policy = true ∧
        capsule.provider = preferred := by
  constructor
  · intro hother hexists
    rcases hexists with ⟨candidate, hmem, havailable, hop, hadmissible, hprovider⟩
    have hfiltered : candidate ∈ availability.filterCapsules registry :=
      Array.mem_filter.mpr ⟨hmem, havailable⟩
    have hsome :
        ((availability.filterCapsules registry).find? fun capsule =>
          (capsule.op == op && capsule.admissible policy) &&
            capsule.provider == preferred).isSome := by
      apply Array.find?_isSome.mpr
      exact ⟨candidate, hfiltered, by simp [hop, hadmissible, hprovider]⟩
    simp only [chooseCapsuleFor?, hprefer] at hselected
    cases hfind : (availability.filterCapsules registry).find? (fun capsule =>
        (capsule.op == op && capsule.admissible policy) &&
          capsule.provider == preferred) with
    | none => simp [hfind] at hsome
    | some found =>
        have heq : found = selected := by simpa [hfind] using hselected
        subst found
        have hpredicate := Array.find?_some hfind
        have hprovider := (Bool.and_eq_true_iff.mp hpredicate).2
        exact hother (by simpa only [beq_iff_eq] using hprovider)
  · intro hmissing heq
    rcases chooseCapsuleFor_sound hselected with ⟨hmem, hop, hadmissible⟩
    rcases Array.mem_filter.mp hmem with ⟨hregistered, havailable⟩
    exact hmissing ⟨selected, hregistered, havailable, hop, hadmissible, heq⟩

/-- Planning one operation preserves its tag and selects a registered, admissible capsule. -/
theorem planOp_sound
    {policy : KernelPolicy} {registry : Array KernelCapsule}
    {op : BackendOp} {kernel : PlannedKernel}
    (hplan : planOp policy registry op = .ok kernel) :
    kernel.op = op ∧ kernel.capsule ∈ registry ∧
      kernel.capsule.op = kernel.op ∧ kernel.capsule.admissible policy = true := by
  cases hchoice : chooseCapsuleFor? policy op registry with
  | none => simp [planOp, hchoice] at hplan
  | some capsule =>
      have heq : ({ op, capsule } : PlannedKernel) = kernel := by
        simpa [planOp, hchoice, Pure.pure, Except.pure] using hplan
      subst kernel
      exact ⟨rfl, chooseCapsuleFor_sound hchoice⟩

private theorem planOps_list_sound
    {policy : KernelPolicy} {registry : Array KernelCapsule}
    {ops : List BackendOp} {kernels : List PlannedKernel}
    (hplan : ops.mapM (planOp policy registry) = .ok kernels) :
    kernels.map (·.op) = ops ∧
      ∀ kernel ∈ kernels, kernel.capsule ∈ registry ∧
        kernel.capsule.op = kernel.op ∧ kernel.capsule.admissible policy = true := by
  induction ops generalizing kernels with
  | nil =>
      have heq : kernels = [] := by
        simpa [Pure.pure, Except.pure] using hplan.symm
      subst kernels
      simp
  | cons op ops ih =>
      simp only [List.mapM_cons] at hplan
      cases hhead : planOp policy registry op with
      | error err => simp [hhead, Bind.bind, Except.bind] at hplan
      | ok kernel =>
          cases htail : ops.mapM (planOp policy registry) with
          | error err => simp [hhead, htail, Bind.bind, Except.bind] at hplan
          | ok tail =>
              have heq : kernel :: tail = kernels := by
                simpa [hhead, htail, Bind.bind, Except.bind, Pure.pure, Except.pure] using hplan
              subst kernels
              rcases planOp_sound hhead with ⟨hop, hkernel⟩
              rcases ih htail with ⟨hops, htail⟩
              constructor
              · simp [hop, hops]
              · intro selected hmem
                rcases List.mem_cons.mp hmem with heq | hmem
                · subst selected
                  exact hkernel
                · exact htail selected hmem

/--
Successful planning preserves operation order and multiplicity, and each selected capsule is
registered, implements its planned operation, and is admissible under the requested policy.
-/
theorem planOps_sound
    {policy : KernelPolicy} {registry : Array KernelCapsule}
    {ops : Array BackendOp} {plan : KernelPlan}
    (hplan : planOps policy registry ops = .ok plan) :
    plan.kernels.map (·.op) = ops ∧
      ∀ kernel ∈ plan.kernels, kernel.capsule ∈ registry ∧
        kernel.capsule.op = kernel.op ∧ kernel.capsule.admissible policy = true := by
  cases hmap : ops.mapM (planOp policy registry) with
  | error err => simp [planOps, hmap] at hplan
  | ok kernels =>
      have heq : ({ kernels } : KernelPlan) = plan := by
        simpa [planOps, hmap] using hplan
      subst plan
      have hlist : ops.toList.mapM (planOp policy registry) = .ok kernels.toList := by
        simpa using congrArg (fun result => Array.toList <$> result) hmap
      rcases planOps_list_sound hlist with ⟨hops, hcapsules⟩
      constructor
      · apply Array.ext'
        simpa using hops
      · simpa using hcapsules

/--
A successful profile plan preserves the requests and selects only registered, available,
operation-matching capsules admissible under the profile policy.
-/
theorem BackendProfile.planOps_sound
    {profile : BackendProfile} {ops : Array BackendOp} {plan : KernelPlan}
    (hplan : profile.planOps ops = .ok plan) :
    plan.kernels.map (·.op) = ops ∧
      ∀ kernel ∈ plan.kernels,
        kernel.capsule ∈ profile.registry ∧
        kernel.capsule.op = kernel.op ∧
        profile.availability.admitsCapsule kernel.capsule = true ∧
        kernel.capsule.admissible profile.policy = true := by
  unfold BackendProfile.planOps at hplan
  cases hvalid : Registry.validateModules profile.capsuleModules with
  | error err => simp [hvalid, Bind.bind, Except.bind] at hplan
  | ok valid =>
      have havailable :
          NN.Backend.planOps profile.policy
            (profile.availability.filterCapsules profile.registry) ops = .ok plan := by
        simpa [hvalid, planOpsAvailable, Bind.bind, Except.bind] using hplan
      rcases NN.Backend.planOps_sound havailable with ⟨hops, hcapsules⟩
      refine ⟨hops, ?_⟩
      intro kernel hmem
      rcases hcapsules kernel hmem with ⟨hfiltered, hop, hadmissible⟩
      rcases Array.mem_filter.mp hfiltered with ⟨hregistered, havailable⟩
      exact ⟨hregistered, hop, havailable, hadmissible⟩

end NN.Backend
