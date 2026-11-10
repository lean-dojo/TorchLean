/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.Backend.Profile
import Batteries.Lean.Except

/-!
# Backend selection soundness

The production planner selects, for each requested operator, a capsule that an available
implementation offers and the policy admits. The proofs cover this choice among declared contracts
and leave out native kernel eligibility, numerical correctness and the behavior of foreign
implementations.
-/

@[expose] public section

namespace TorchLean.Backend

/-- For every contract, provider, device and operator, the capsule built from the contract has
aligned contracts: each of its contract descriptors (shape, layout, value and VJP) claims its own
field's obligation for that operator. -/
theorem KernelContract.contractsAligned_capsule (c : KernelContract) (provider : Provider)
    (device : Device) {ins outs : List Port} (op : Op ins outs) :
    (c.capsule provider device op).contractsAligned = true := by
  cases hmode : c.vjpMode <;>
    simp [KernelContract.capsule, KernelCapsule.contractsAligned,
      Contract.Claim.matchesObligation, hmode]

/-- Every capsule offered by an implementation built from contracts has aligned contracts. -/
theorem Implementation.contractsAligned_ofContracts {name : String} {provider : Provider}
    {device : Device} {contract? : {ins outs : List Port} → Op ins outs → Option KernelContract}
    {ins outs : List Port} {op : Op ins outs} {c : KernelCapsule}
    (h : (Implementation.ofContracts name provider device contract?).capsule? op = some c) :
    c.contractsAligned = true := by
  obtain ⟨contract, -, rfl⟩ := Option.map_eq_some_iff.mp h
  exact KernelContract.contractsAligned_capsule contract provider device op

/-- A successful choice is offered for the requested operator, names it, and is admissible. -/
theorem chooseCapsuleFor?_sound
    {policy : KernelPolicy} {ins outs : List Port} {op : Op ins outs}
    {impls : Array Implementation} {selected : KernelCapsule}
    (hselected : chooseCapsuleFor? policy op impls = some selected) :
    Implementation.Offers impls op selected ∧ selected.op = Op.encode ⟨ins, outs, op⟩ ∧
      selected.admissible policy = true := by
  have ordinary
      (hfind : (Implementation.candidates impls op).find? (fun capsule =>
        capsule.op == Op.encode ⟨ins, outs, op⟩ && capsule.admissible policy) = some selected) :
      Implementation.Offers impls op selected ∧ selected.op = Op.encode ⟨ins, outs, op⟩ ∧
        selected.admissible policy = true := by
    refine ⟨Implementation.mem_candidates.mp (Array.mem_of_find?_eq_some hfind), ?_⟩
    simpa only [Bool.and_eq_true, beq_iff_eq] using Array.find?_some hfind
  cases hpolicy : policy.provider with
  | prefer preferred =>
    simp only [chooseCapsuleFor?, hpolicy] at hselected
    cases hfind : (Implementation.candidates impls op).find? (fun capsule =>
        (capsule.op == Op.encode ⟨ins, outs, op⟩ && capsule.admissible policy) &&
          capsule.provider == preferred) with
    | none =>
        exact ordinary (by simpa [hfind] using hselected)
    | some found =>
        have heq : found = selected := by simpa [hfind] using hselected
        subst found
        refine ⟨Implementation.mem_candidates.mp (Array.mem_of_find?_eq_some hfind), ?_⟩
        have hpredicate := Array.find?_some hfind
        have eligible := (Bool.and_eq_true_iff.mp hpredicate).1
        simpa only [Bool.and_eq_true, beq_iff_eq] using eligible
  | auto => exact ordinary (by simpa [chooseCapsuleFor?, hpolicy] using hselected)
  | only provider => exact ordinary (by simpa [chooseCapsuleFor?, hpolicy] using hselected)

/-- An exclusive provider request never selects a capsule from another provider. -/
theorem chooseCapsuleFor?_only_provider
    {policy : KernelPolicy} {ins outs : List Port} {op : Op ins outs}
    {impls : Array Implementation} {selected : KernelCapsule} {provider : Provider}
    (honly : policy.provider = .only provider)
    (hselected : chooseCapsuleFor? policy op impls = some selected) :
    selected.provider = provider := by
  have hadmissible := (chooseCapsuleFor?_sound hselected).2.2
  simp only [KernelCapsule.admissible, Bool.and_eq_true, and_assoc] at hadmissible
  have hprovider := hadmissible.2.2.2.1
  simpa [KernelCapsule.matchesPreference, honly, eq_comm] using hprovider

/--
A successful choice leaves the preferred provider exactly when no available, admissible capsule of
that provider is offered for the requested operator.
-/
theorem chooseCapsuleFor?_fallback_iff
    {policy : KernelPolicy} {availability : Availability}
    {impls : Array Implementation} {ins outs : List Port} {op : Op ins outs}
    {preferred : Provider} {selected : KernelCapsule}
    (hprefer : policy.provider = .prefer preferred)
    (hselected : chooseCapsuleFor? policy op (availability.filter impls) = some selected) :
    selected.provider ≠ preferred ↔
      ¬ ∃ impl ∈ impls, ∃ capsule, impl.capsule? op = some capsule ∧
        availability.admitsCapsule capsule = true ∧
        capsule.op = Op.encode ⟨ins, outs, op⟩ ∧
        capsule.admissible policy = true ∧
        capsule.provider = preferred := by
  constructor
  · intro hother hexists
    rcases hexists with
      ⟨impl, himpl, candidate, hcandidate, havailable, hop, hadmissible, hprovider⟩
    have hfiltered : candidate ∈ Implementation.candidates (availability.filter impls) op :=
      Implementation.mem_candidates.mpr
        (Availability.offers_filter.mpr ⟨⟨impl, himpl, hcandidate⟩, havailable⟩)
    have hsome :
        ((Implementation.candidates (availability.filter impls) op).find? fun capsule =>
          (capsule.op == Op.encode ⟨ins, outs, op⟩ && capsule.admissible policy) &&
            capsule.provider == preferred).isSome := by
      apply Array.find?_isSome.mpr
      exact ⟨candidate, hfiltered, by simp [hop, hadmissible, hprovider]⟩
    simp only [chooseCapsuleFor?, hprefer] at hselected
    cases hfind : (Implementation.candidates (availability.filter impls) op).find? (fun capsule =>
        (capsule.op == Op.encode ⟨ins, outs, op⟩ && capsule.admissible policy) &&
          capsule.provider == preferred) with
    | none => simp [hfind] at hsome
    | some found =>
        have heq : found = selected := by simpa [hfind] using hselected
        subst found
        have hpredicate := Array.find?_some hfind
        have hprovider := (Bool.and_eq_true_iff.mp hpredicate).2
        exact hother (by simpa only [beq_iff_eq] using hprovider)
  · intro hmissing heq
    rcases chooseCapsuleFor?_sound hselected with ⟨hoffers, hop, hadmissible⟩
    rcases Availability.offers_filter.mp hoffers with ⟨⟨impl, himpl, hcandidate⟩, havailable⟩
    exact hmissing ⟨impl, himpl, selected, hcandidate, havailable, hop, hadmissible, heq⟩

/-- Planning one operator records its wire record and selects an offered, admissible capsule. -/
theorem planOp_sound
    {policy : KernelPolicy} {impls : Array Implementation} {ins outs : List Port}
    {op : Op ins outs} {kernel : PlannedKernel}
    (hplan : planOp policy impls op = .ok kernel) :
    kernel.op = Op.encode ⟨ins, outs, op⟩ ∧ Implementation.Offers impls op kernel.capsule ∧
      kernel.capsule.op = kernel.op ∧ kernel.capsule.admissible policy = true := by
  unfold planOp at hplan
  cases hduplicate : firstDuplicateCapsuleIdentity? (Implementation.candidates impls op) with
  | some name =>
      simp [hduplicate, throw, throwThe, MonadExceptOf.throw, Bind.bind, Except.bind] at hplan
  | none =>
    cases hchoice : chooseCapsuleFor? policy op impls with
    | none =>
        simp [hduplicate, hchoice, throw, throwThe, MonadExceptOf.throw] at hplan
    | some capsule =>
        have heq : ({ op := Op.encode ⟨ins, outs, op⟩, capsule } : PlannedKernel) = kernel := by
          simpa [hduplicate, hchoice, Pure.pure, Except.pure, Bind.bind, Except.bind] using hplan
        subst kernel
        rcases chooseCapsuleFor?_sound hchoice with ⟨hoffers, hop, hadmissible⟩
        exact ⟨rfl, hoffers, hop, hadmissible⟩

private theorem planOps_list_sound
    {policy : KernelPolicy} {impls : Array Implementation}
    {ops : List (Σ ins outs, Op ins outs)} {kernels : List PlannedKernel}
    (hplan : ops.mapM (fun e => planOp policy impls e.2.2) = .ok kernels) :
    kernels.map (·.op) = ops.map Op.encode ∧
      ∀ kernel ∈ kernels, (∃ e ∈ ops, kernel.op = Op.encode e ∧
          Implementation.Offers impls e.2.2 kernel.capsule) ∧
        kernel.capsule.op = kernel.op ∧ kernel.capsule.admissible policy = true := by
  induction ops generalizing kernels with
  | nil =>
      have heq : kernels = [] := by
        simpa [Pure.pure, Except.pure] using hplan.symm
      subst kernels
      simp
  | cons e ops ih =>
      simp only [List.mapM_cons] at hplan
      cases hhead : planOp policy impls e.2.2 with
      | error err => simp [hhead, Bind.bind, Except.bind] at hplan
      | ok kernel =>
          cases htail : ops.mapM (fun e => planOp policy impls e.2.2) with
          | error err => simp [hhead, htail, Bind.bind, Except.bind] at hplan
          | ok tail =>
              have heq : kernel :: tail = kernels := by
                simpa [hhead, htail, Bind.bind, Except.bind, Pure.pure, Except.pure] using hplan
              subst kernels
              rcases planOp_sound hhead with ⟨hop, hoffers, hcapsule, hadmissible⟩
              rcases ih htail with ⟨hops, htail⟩
              constructor
              · simp only [List.map_cons, hops]
                exact congrArg (· :: _) hop
              · intro selected hmem
                rcases List.mem_cons.mp hmem with heq | hmem
                · subst selected
                  exact ⟨⟨e, List.mem_cons_self, hop, hoffers⟩, hcapsule, hadmissible⟩
                · rcases htail selected hmem with ⟨⟨e', he', hop', hoffers'⟩, hrest⟩
                  exact ⟨⟨e', List.mem_cons_of_mem _ he', hop', hoffers'⟩, hrest⟩

/--
Successful planning preserves operator order and multiplicity, and each selected capsule is offered
for its request, names its planned operator, and is admissible under the requested policy.
-/
theorem planOps_sound
    {policy : KernelPolicy} {impls : Array Implementation}
    {ops : Array (Σ ins outs, Op ins outs)} {plan : KernelPlan}
    (hplan : planOps policy impls ops = .ok plan) :
    plan.kernels.map (·.op) = ops.map Op.encode ∧
      ∀ kernel ∈ plan.kernels, (∃ e ∈ ops, kernel.op = Op.encode e ∧
          Implementation.Offers impls e.2.2 kernel.capsule) ∧
        kernel.capsule.op = kernel.op ∧ kernel.capsule.admissible policy = true := by
  cases hmap : ops.mapM (fun e => planOp policy impls e.2.2) with
  | error err => simp [planOps, hmap] at hplan
  | ok kernels =>
      have heq : ({ kernels } : KernelPlan) = plan := by
        simpa [planOps, hmap] using hplan
      subst plan
      have hlist : ops.toList.mapM (fun e => planOp policy impls e.2.2) =
          .ok kernels.toList := by
        simpa using congrArg (fun result => Array.toList <$> result) hmap
      rcases planOps_list_sound hlist with ⟨hops, hcapsules⟩
      constructor
      · apply Array.ext'
        simpa using hops
      · simpa using hcapsules

/--
A successful profile plan preserves the requests and selects only capsules offered for them by the
profile's implementations, available on the build, naming their operators, and admissible under
the profile policy.
-/
theorem Profile.planOps_sound
    {profile : Profile} {ops : Array (Σ ins outs, Op ins outs)} {plan : KernelPlan}
    (hplan : profile.planOps ops = .ok plan) :
    plan.kernels.map (·.op) = ops.map Op.encode ∧
      ∀ kernel ∈ plan.kernels,
        (∃ e ∈ ops, kernel.op = Op.encode e ∧
          Implementation.Offers profile.implementations e.2.2 kernel.capsule) ∧
        kernel.capsule.op = kernel.op ∧
        profile.availability.admitsCapsule kernel.capsule = true ∧
        kernel.capsule.admissible profile.policy = true := by
  unfold Profile.planOps at hplan
  cases hvalid : Registry.validate profile.implementations with
  | error err => simp [hvalid, Bind.bind, Except.bind] at hplan
  | ok valid =>
      have havailable :
          Backend.planOps profile.policy
            (profile.availability.filter profile.implementations) ops = .ok plan := by
        simpa [hvalid, planOpsAvailable, Bind.bind, Except.bind] using hplan
      rcases Backend.planOps_sound havailable with ⟨hops, hcapsules⟩
      refine ⟨hops, ?_⟩
      intro kernel hmem
      rcases hcapsules kernel hmem with ⟨⟨e, he, hop, hoffers⟩, hcapsule, hadmissible⟩
      rcases Availability.offers_filter.mp hoffers with ⟨hoffers, havailable⟩
      exact ⟨⟨e, he, hop, hoffers⟩, hcapsule, havailable, hadmissible⟩

end TorchLean.Backend
