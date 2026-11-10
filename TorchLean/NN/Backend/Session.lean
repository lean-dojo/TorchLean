/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.Session.Apply
import all TorchLean.NN.Session.Backend
import all TorchLean.NN.Session
import TorchLean.NN.LibTorch.Kernel
import TorchLean.NN.Data.Monad
import TorchLean.NN.Tape.Apply.Correctness
import TorchLean.NN.Backend.Planner.Soundness

/-!
# The capsule `Session.apply` selects and the program it runs

We prove which capsules `TorchLean.Session.selectedCapsule` can return
and which program `Session.apply` and `Session.applyRandom` run with them. `IO.Ref` reads
and writes are opaque, so the statements are equations between `IO` programs and facts about the
values a program can return (`MonadAttach.CanReturn`).

Three theorems tie a reference capsule to `Op.rule?`, the rule `Tape.apply` runs. With no profile
override, on the CPU, every capsule the session selects is the capsule of
`TorchLean.Backend.Reference.contract? op` (`Session.selectedCapsule_of_cpu`),
`Session.apply` records `op` with `Session.recordApply` (`Session.apply_of_cpu`), and
`Session.recordApply` runs `Tape.apply` on the CPU tape `t`, stores the tape it returns and
returns `t.size`, the id of the appended node (`Session.recordApply_eq_recordCPU`).
`TorchLean.Tape.apply_node` states that the node at `t.size` is the recorded node of the rule
`op.rule?` returns.

Main declarations:
- `Session.Config.planOp_sound`: a kernel the planner accepts comes from the effective profile.
- `Session.selectedCapsule_sound`: a selected capsule is the accepted kernel's capsule, with
  the operator's wire record and the session's device.
- `Session.selectedCapsule_of_cpu` and `Session.selectedCapsule_of_cuda`: the capsules of the
  maintained profiles. `Session.not_canReturn_selectedCapsule`: a profile override for another
  device selects no capsule.
- `Session.runsOnCUDA_eq_ok_false_iff` and `Session.runsOnCUDA_eq_ok_true_iff`: the two
  kinds of capsule `runsOnCUDA` accepts.
- `Session.recordSelected_of_reference_cpu`, `Session.recordSelected_of_libTorch_cuda` and
  `Session.recordSelected_of_runsOnCUDA_eq_error`: the program for each kind of capsule, with
  the `Session.checkImplementation_of_*` theorems for `checkImplementation`.
- `Session.recordApply_eq_recordCPU`: the CPU tape step.
- `Session.apply_of_cpu`, `Session.apply_of_cuda`, `Session.applyRandom_of_cpu` and
  `Session.applyRandom_of_cuda`: the programs of `apply` and `applyRandom` on the maintained
  profiles.
-/

public section

namespace TorchLean.Backend.KernelCapsule

/-- `c.validateEagerRequest op device = .ok ()` exactly when `c.op = op` and `c.device = device`. -/
theorem validateEagerRequest_eq_ok_iff {c : KernelCapsule} {op : Op.Wire}
    {device : Device} :
    c.validateEagerRequest op device = .ok () ↔ c.op = op ∧ c.device = device := by
  by_cases hop : c.op = op <;> by_cases hdevice : c.device = device <;>
    simp [validateEagerRequest, hop, hdevice, throw, throwThe, MonadExceptOf.throw, bind,
      Except.bind, pure, Except.pure]

end TorchLean.Backend.KernelCapsule

namespace TorchLean.Session.Config

/-- A kernel `k` with `config.planOp op gradEnabled = .ok k` comes from the effective profile
`profile`: `k` has the wire record of `op` and `profile.policy`, and its capsule is offered for `op`
by `profile.implementations`, admitted by `profile.availability` and admissible under
`profile.policy`. -/
theorem planOp_sound {config : Config} {ins outs : List Port} {op : Op ins outs}
    {gradEnabled : Bool} {k : Backend.AcceptedKernel}
    (h : config.planOp op gradEnabled = .ok k) :
    ∃ profile, config.effectiveBackendProfile gradEnabled = .ok profile ∧
      k.op = Op.encode ⟨ins, outs, op⟩ ∧ k.policy = profile.policy ∧
      Backend.Implementation.Offers profile.implementations op k.capsule ∧
      profile.availability.admitsCapsule k.capsule = true ∧
      k.capsule.admissible profile.policy = true := by
  unfold planOp at h
  cases hprofile : config.effectiveBackendProfile gradEnabled with
  | error e => simp [hprofile, bind, Except.bind] at h
  | ok profile =>
    refine ⟨profile, rfl, ?_⟩
    simp only [hprofile, bind, Except.bind] at h
    cases hplan : profile.planOps #[⟨ins, outs, op⟩] with
    | error e => simp [hplan] at h
    | ok plan =>
      simp only [hplan] at h
      cases hk : plan.kernels[0]? with
      | none => simp [hk] at h
      | some kernel =>
        simp only [hk] at h
        cases hacc : kernel.accept profile.policy with
        | error failures => simp [hacc] at h
        | ok accepted =>
          simp only [hacc, Except.ok.injEq] at h
          subst h
          obtain ⟨-, hop, hcapsule, hpolicy⟩ := Backend.PlannedKernel.accept_eq_ok_iff.mp hacc
          obtain ⟨-, hsound⟩ := Backend.Profile.planOps_sound hplan
          obtain ⟨⟨e, he, hkop, hoffers⟩, -, havailable, hadmissible⟩ :=
            hsound kernel (Array.mem_of_getElem? hk)
          have he' : e = ⟨ins, outs, op⟩ := by simpa using he
          subst he'
          rw [hop, hcapsule, hpolicy]
          exact ⟨hkop, rfl, hoffers, havailable, hadmissible⟩

/-- With no profile override, an effective profile has the maintained implementation tables. -/
private theorem implementations_of_backendProfile?_eq_none {config : Config}
    {gradEnabled : Bool} {profile : Backend.Profile}
    (hnone : config.backendProfile? = none)
    (h : config.effectiveBackendProfile gradEnabled = .ok profile) :
    profile.implementations = Backend.Registry.maintained := by
  unfold effectiveBackendProfile resolveBackendProfile at h
  rw [hnone] at h
  cases hdevice : Backend.Profile.maintainedForDevice? config.device with
  | none => simp [hdevice, bind, Except.bind, throw, throwThe, MonadExceptOf.throw] at h
  | some maintained =>
    simp only [hdevice, bind, Except.bind, pure, Except.pure, Except.ok.injEq] at h
    subst h
    have hm : maintained.implementations = Backend.Registry.maintained := by
      cases hd : config.device <;>
        simp only [hd, Backend.Profile.maintainedForDevice?, Option.some.injEq]
          at hdevice <;>
        (subst hdevice; rfl)
    split <;> exact hm

end TorchLean.Session.Config

namespace TorchLean.Session

variable {α : Type} [Storage α]

/-- A capsule `c` that `s.selectedCapsule op` can return is the capsule of a kernel `k` with
`s.options.planOp op s.gradEnabled = .ok k`, and `c` names the wire record of `op` and the
session's device. -/
theorem selectedCapsule_sound {s : Session α} {ins outs : List Port} {op : Op ins outs}
    {c : Backend.KernelCapsule} (h : MonadAttach.CanReturn (s.selectedCapsule op) c) :
    ∃ k, s.options.planOp op s.gradEnabled = .ok k ∧ k.capsule = c ∧
      c.op = Op.encode ⟨ins, outs, op⟩ ∧ c.device = s.options.device := by
  unfold selectedCapsule at h
  cases hplan : s.options.planOp op s.gradEnabled with
  | error msg =>
    simp only [hplan, EIO.throw_bind] at h
    exact absurd h EIO.not_canReturn_throw
  | ok planned =>
    simp only [hplan, pure_bind] at h
    refine ⟨planned, rfl, ?_⟩
    cases hvalid : planned.capsule.validateEagerRequest (Op.encode ⟨ins, outs, op⟩)
        s.options.device with
    | error msg =>
      simp only [hvalid, EIO.throw_bind] at h
      exact absurd h EIO.not_canReturn_throw
    | ok u =>
      cases u
      obtain ⟨hop, hdevice⟩ := Backend.KernelCapsule.validateEagerRequest_eq_ok_iff.mp hvalid
      simp only [hvalid] at h
      obtain ⟨selected, -, h⟩ := LawfulMonadAttach.canReturn_bind_imp' h
      have hc : planned.capsule = c := by
        split at h
        · exact LawfulMonadAttach.eq_of_canReturn_pure h
        · obtain ⟨_, -, h⟩ := LawfulMonadAttach.canReturn_bind_imp' h
          split at h
          · obtain ⟨_, -, h⟩ := LawfulMonadAttach.canReturn_bind_imp' h
            exact LawfulMonadAttach.eq_of_canReturn_pure h
          · exact LawfulMonadAttach.eq_of_canReturn_pure h
      subst hc
      exact ⟨rfl, hop, hdevice⟩

/-- With no profile override, on the CPU, a capsule `c` that `s.selectedCapsule op` can return is
offered for `op` by `Backend.Reference.implementation`: it is the capsule of
`Backend.Reference.contract? op`, of provider `reference` on device `cpu`. -/
theorem selectedCapsule_of_cpu {s : Session α} {ins outs : List Port} {op : Op ins outs}
    {c : Backend.KernelCapsule} (hprofile : s.options.backendProfile? = none)
    (hdevice : s.options.device = .cpu) (h : MonadAttach.CanReturn (s.selectedCapsule op) c) :
    Backend.Reference.implementation.capsule? op = some c ∧ c.provider = .reference ∧
      c.device = .cpu ∧
      ∃ contract, Backend.Reference.contract? op = some contract ∧
        c = contract.capsule .reference .cpu op := by
  obtain ⟨k, hplan, rfl, -, hcdevice⟩ := selectedCapsule_sound h
  obtain ⟨profile, hprof, -, -, ⟨impl, himpl, hoffer⟩, -, -⟩ := Config.planOp_sound hplan
  rw [Config.implementations_of_backendProfile?_eq_none hprofile hprof] at himpl
  simp only [Backend.Registry.maintained, List.mem_toArray, List.mem_cons, List.not_mem_nil,
    or_false] at himpl
  rcases himpl with rfl | rfl
  · rw [LibTorch.implementation.capsule_device op _ hoffer,
      LibTorch.implementation_identity.2.2, hdevice]
      at hcdevice
    cases hcdevice
  · obtain ⟨contract, hcontract, hcapsule⟩ := Option.map_eq_some_iff.mp hoffer
    exact ⟨hoffer, Backend.Reference.implementation.capsule_provider op _ hoffer,
      Backend.Reference.implementation.capsule_device op _ hoffer, contract, hcontract,
      hcapsule.symm⟩

/-- With no profile override, on CUDA, a capsule `c` that `s.selectedCapsule op` can return is
offered for `op` by `LibTorch.implementation`, of provider `libTorch` on device
`cuda`. -/
theorem selectedCapsule_of_cuda {s : Session α} {ins outs : List Port} {op : Op ins outs}
    {c : Backend.KernelCapsule} (hprofile : s.options.backendProfile? = none)
    (hdevice : s.options.device = .cuda) (h : MonadAttach.CanReturn (s.selectedCapsule op) c) :
    LibTorch.implementation.capsule? op = some c ∧ c.provider = .libTorch ∧
      c.device = .cuda := by
  obtain ⟨k, hplan, rfl, -, hcdevice⟩ := selectedCapsule_sound h
  obtain ⟨profile, hprof, -, -, ⟨impl, himpl, hoffer⟩, -, -⟩ := Config.planOp_sound hplan
  rw [Config.implementations_of_backendProfile?_eq_none hprofile hprof] at himpl
  simp only [Backend.Registry.maintained, List.mem_toArray, List.mem_cons, List.not_mem_nil,
    or_false] at himpl
  rcases himpl with rfl | rfl
  · refine ⟨hoffer, ?_, ?_⟩
    · rw [LibTorch.implementation.capsule_provider op _ hoffer]
      exact LibTorch.implementation_identity.2.1
    · rw [LibTorch.implementation.capsule_device op _ hoffer]
      exact LibTorch.implementation_identity.2.2
  · rw [Backend.Reference.implementation.capsule_device op _ hoffer, hdevice] at hcdevice
    cases hcdevice

/-- With a profile override `p` whose device is not the session's device, `s.selectedCapsule op`
returns no capsule. -/
theorem not_canReturn_selectedCapsule {s : Session α} {ins outs : List Port}
    {op : Op ins outs} {c : Backend.KernelCapsule} {p : Backend.Profile}
    (hprofile : s.options.backendProfile? = some p) (hdevice : p.policy.device ≠ s.options.device) :
    ¬ MonadAttach.CanReturn (s.selectedCapsule op) c := by
  intro h
  obtain ⟨k, hplan, -⟩ := selectedCapsule_sound h
  obtain ⟨profile, hprof, -⟩ := Config.planOp_sound hplan
  simp [Config.effectiveBackendProfile, Config.resolveBackendProfile, hprofile,
    hdevice, bind, Except.bind, throw, throwThe, MonadExceptOf.throw] at hprof

/-- `runsOnCUDA tag c = .ok false` exactly when `c` has provider `reference` and device `cpu`. -/
theorem runsOnCUDA_eq_ok_false_iff {tag : String} {c : Backend.KernelCapsule} :
    runsOnCUDA tag c = .ok false ↔ c.provider = .reference ∧ c.device = .cpu := by
  unfold runsOnCUDA
  split
  · next h =>
    simp only [Bool.and_eq_true, beq_iff_eq] at h
    simp [h, pure, Except.pure]
  · next h₁ =>
    simp only [Bool.and_eq_true, beq_iff_eq] at h₁
    split
    · next h₂ =>
      simp only [Bool.and_eq_true, beq_iff_eq] at h₂
      simp [h₂, pure, Except.pure]
    · exact iff_of_false (fun h => nomatch h) h₁

/-- `runsOnCUDA tag c = .ok true` exactly when `c` has provider `libTorch` and device `cuda`. -/
theorem runsOnCUDA_eq_ok_true_iff {tag : String} {c : Backend.KernelCapsule} :
    runsOnCUDA tag c = .ok true ↔ c.provider = .libTorch ∧ c.device = .cuda := by
  unfold runsOnCUDA
  split
  · next h =>
    simp only [Bool.and_eq_true, beq_iff_eq] at h
    simp [h, pure, Except.pure]
  · next h₁ =>
    split
    · next h₂ =>
      simp only [Bool.and_eq_true, beq_iff_eq] at h₂
      simp [h₂, pure, Except.pure]
    · next h₂ =>
      simp only [Bool.and_eq_true, beq_iff_eq] at h₂
      exact iff_of_false (fun h => nomatch h) h₂

/-- On a capsule of provider `reference` and device `cpu`,
`s.checkImplementation c op = pure ()`. -/
theorem checkImplementation_of_reference_cpu {s : Session α} {ins outs : List Port}
    {c : Backend.KernelCapsule} {op : Op ins outs} (hprovider : c.provider = .reference)
    (hdevice : c.device = .cpu) : s.checkImplementation c op = pure () := by
  unfold checkImplementation
  rw [runsOnCUDA_eq_ok_false_iff.mpr ⟨hprovider, hdevice⟩, IO.ofExcept_ok, pure_bind]
  rfl

/-- On a capsule of provider `libTorch` and device `cuda`,
`s.checkImplementation c op = discard (s.cudaEntry c op)`. -/
theorem checkImplementation_of_libTorch_cuda {s : Session α} {ins outs : List Port}
    {c : Backend.KernelCapsule} {op : Op ins outs} (hprovider : c.provider = .libTorch)
    (hdevice : c.device = .cuda) : s.checkImplementation c op = discard (s.cudaEntry c op) := by
  unfold checkImplementation
  rw [runsOnCUDA_eq_ok_true_iff.mpr ⟨hprovider, hdevice⟩, IO.ofExcept_ok, pure_bind]
  rfl

/-- When `runsOnCUDA` rejects the capsule with `e`, `s.checkImplementation c op` throws
`IO.userError e`. -/
theorem checkImplementation_of_runsOnCUDA_eq_error {s : Session α} {ins outs : List Port}
    {c : Backend.KernelCapsule} {op : Op ins outs} {e : String}
    (h : runsOnCUDA (Op.encode ⟨ins, outs, op⟩).tag c = .error e) :
    s.checkImplementation c op = throw (IO.userError e) := by
  unfold checkImplementation
  rw [h, IO.ofExcept_error, EIO.throw_bind]
  rfl

variable [Context α] [Op.Capabilities α]

/-- `s.recordApply name op args` is `s.recordCPU` of the step that sends a tape `t` to
`(t', .ok t.size)` when `Tape.apply t name op args = .ok (t', o)` and to `(t, .error e)` when it is
`.error e`. -/
theorem recordApply_eq_recordCPU {s : Session α} {sh : Shape} {ins rest : List Port}
    {name : String} {op : Op ins (.value sh :: rest)} {args : Tape.Args α ins} :
    s.recordApply name op args = s.recordCPU fun t =>
      match Tape.apply t name op args with
      | Except.ok (t', _) => (t', Except.ok t.size)
      | Except.error e => (t, Except.error e) := by
  unfold recordApply
  congr 1
  funext t
  cases h : Tape.apply t name op args with
  | error e => rfl
  | ok r =>
    obtain ⟨t', o⟩ := r
    have ho : o.1 = t.size := by
      cases hc : op.check with
      | error e => rw [Tape.apply_of_check_eq_error hc] at h; cases h
      | ok u =>
        cases u
        cases hs : op.rule? (α := α) with
        | none => rw [Tape.apply_of_rule?_eq_none hc hs] at h; cases h
        | some rule =>
          cases hxs : Tape.Args.read t ins args with
          | error e => rw [Tape.apply_of_read_eq_error hc hs hxs] at h; cases h
          | ok xs =>
            rw [Tape.apply_value h hs hxs]
            rfl
    change (t', Except.ok o.1) = (t', Except.ok t.size)
    rw [ho]

variable [TensorTransfer α]

/-- On a capsule of provider `reference` and device `cpu`, `s.recordSelected name c op args checks`
runs `checks`, records `op` on the CPU tape with `s.recordApply` and returns a reference to the
recorded node. -/
theorem recordSelected_of_reference_cpu {s : Session α} {ins : List Port} {sh : Shape}
    {name : String} {rest : List Port} {c : Backend.KernelCapsule}
    {op : Op ins (.value sh :: rest)} {args : Args α ins} {checks : IO Unit}
    (hprovider : c.provider = .reference) (hdevice : c.device = .cpu) :
    s.recordSelected name c op args checks = (do
      checks
      let id ← s.recordApply name op (Args.cpu ins args)
      pure { id, identity? := some (← s.refIdentity) } : IO (TensorRef α sh)) := by
  unfold recordSelected
  rw [runsOnCUDA_eq_ok_false_iff.mpr ⟨hprovider, hdevice⟩, IO.ofExcept_ok, pure_bind]
  rfl

/-- On a capsule of provider `libTorch` and device `cuda`,
`s.recordSelected name c op args checks` takes the CUDA entry `s.cudaEntry c op`, runs `checks`,
records the entry on the CUDA tape and returns a reference to the recorded node. -/
theorem recordSelected_of_libTorch_cuda {s : Session α} {ins : List Port} {sh : Shape}
    {name : String} {rest : List Port} {c : Backend.KernelCapsule}
    {op : Op ins (.value sh :: rest)} {args : Args α ins} {checks : IO Unit}
    (hprovider : c.provider = .libTorch) (hdevice : c.device = .cuda) :
    s.recordSelected name c op args checks = (do
      let kernel ← s.cudaEntry c op
      checks
      let parents ← Args.cuda ins args
      let id ← s.recordCUDAIO fun tape => do
        let (tape, result) ← kernel.record tape name (TensorTransfer.dtype (α := α)) parents
        pure (tape, result.map fun (id, _) => id)
      pure { id, identity? := some (← s.refIdentity) } : IO (TensorRef α sh)) := by
  unfold recordSelected
  rw [runsOnCUDA_eq_ok_true_iff.mpr ⟨hprovider, hdevice⟩, IO.ofExcept_ok, pure_bind]
  rfl

/-- When `runsOnCUDA` rejects the capsule with `e`, `s.recordSelected name c op args checks` throws
`IO.userError e`. -/
theorem recordSelected_of_runsOnCUDA_eq_error {s : Session α} {ins : List Port} {sh : Shape}
    {name : String} {rest : List Port} {c : Backend.KernelCapsule}
    {op : Op ins (.value sh :: rest)} {args : Args α ins} {checks : IO Unit} {e : String}
    (h : runsOnCUDA (Op.encode ⟨ins, .value sh :: rest, op⟩).tag c = .error e) :
    s.recordSelected name c op args checks = throw (IO.userError e) := by
  unfold recordSelected
  rw [h, IO.ofExcept_error, EIO.throw_bind]
  rfl

/-- With no profile override, on the CPU, `s.apply name op args` validates the references, selects
the capsule of `op`, records `op` on the CPU tape with `s.recordApply` and returns a reference to
the recorded node. -/
theorem apply_of_cpu {s : Session α} {ins : List Port} {sh : Shape} {name : String}
    {rest : List Port} {op : Op ins (.value sh :: rest)} {args : Args α ins}
    (hprofile : s.options.backendProfile? = none) (hdevice : s.options.device = .cpu) :
    s.apply name op args = (do
      s.validateRefIdentities (Args.identities ins args)
      let _ ← s.selectedCapsule op
      let id ← s.recordApply name op (Args.cpu ins args)
      pure { id, identity? := some (← s.refIdentity) } : IO (TensorRef α sh)) := by
  unfold apply
  congr 1
  funext u
  apply WeaklyLawfulMonadAttach.bind_congr
  intro c hc
  obtain ⟨-, hprovider, hcdevice, -⟩ := selectedCapsule_of_cpu hprofile hdevice hc
  rw [recordSelected_of_reference_cpu hprovider hcdevice, pure_bind]

/-- With no profile override, on CUDA, `s.apply name op args` validates the references, selects
the capsule of `op`, takes its CUDA entry, records the entry on the CUDA tape and returns a
reference to the recorded node. -/
theorem apply_of_cuda {s : Session α} {ins : List Port} {sh : Shape} {name : String}
    {rest : List Port} {op : Op ins (.value sh :: rest)} {args : Args α ins}
    (hprofile : s.options.backendProfile? = none) (hdevice : s.options.device = .cuda) :
    s.apply name op args = (do
      s.validateRefIdentities (Args.identities ins args)
      let capsule ← s.selectedCapsule op
      let kernel ← s.cudaEntry capsule op
      let parents ← Args.cuda ins args
      let id ← s.recordCUDAIO fun tape => do
        let (tape, result) ← kernel.record tape name (TensorTransfer.dtype (α := α)) parents
        pure (tape, result.map fun (id, _) => id)
      pure { id, identity? := some (← s.refIdentity) } : IO (TensorRef α sh)) := by
  unfold apply
  congr 1
  funext u
  apply WeaklyLawfulMonadAttach.bind_congr
  intro c hc
  obtain ⟨-, hprovider, hcdevice⟩ := selectedCapsule_of_cuda hprofile hdevice hc
  rw [recordSelected_of_libTorch_cuda hprovider hcdevice]
  simp only [pure_bind]

/-- With no profile override, on the CPU, `s.applyRandom name operator seed args` validates the
references, reads the draw counter `invocation`, selects the capsule of
`op := operator (Random.keyOf seed invocation)`, advances the counter, records `op` on the CPU
tape with `s.recordApply` and returns a reference to the recorded node. -/
theorem applyRandom_of_cpu {s : Session α} {ins : List Port} {sh : Shape} {name : String}
    {rest : List Port} {operator : UInt64 → Op ins (.value sh :: rest)} {seed : Nat}
    {args : Args α ins} (hprofile : s.options.backendProfile? = none)
    (hdevice : s.options.device = .cpu) :
    s.applyRandom name operator seed args = (do
      s.validateRefIdentities (Args.identities ins args)
      let invocation ← s.rngCounter.get
      let op := operator (Random.keyOf seed invocation)
      let _ ← s.selectedCapsule op
      s.takeRandomKey
      let id ← s.recordApply name op (Args.cpu ins args)
      pure { id, identity? := some (← s.refIdentity) } : IO (TensorRef α sh)) := by
  unfold applyRandom
  have hne : s.options.usesCUDA = false := by
    show (s.options.device == .cuda) = false
    rw [hdevice]; rfl
  simp only [hne, Bool.false_eq_true, ↓reduceIte]
  congr 1
  funext u
  congr 1
  funext invocation
  apply WeaklyLawfulMonadAttach.bind_congr
  intro c hc
  obtain ⟨-, hprovider, hcdevice, -⟩ := selectedCapsule_of_cpu hprofile hdevice hc
  rw [recordSelected_of_reference_cpu hprovider hcdevice]

/-- With no profile override, on CUDA, `s.applyRandom name operator seed args` validates the
references, reads the draw counter `invocation`, reads the CUDA tape, selects the capsule of
`op := operator (Random.keyOf seed invocation)`, takes its CUDA entry, advances the counter,
records the entry on the CUDA tape and returns a reference to the recorded node. -/
theorem applyRandom_of_cuda {s : Session α} {ins : List Port} {sh : Shape} {name : String}
    {rest : List Port} {operator : UInt64 → Op ins (.value sh :: rest)} {seed : Nat}
    {args : Args α ins} (hprofile : s.options.backendProfile? = none)
    (hdevice : s.options.device = .cuda) :
    s.applyRandom name operator seed args = (do
      s.validateRefIdentities (Args.identities ins args)
      let invocation ← s.rngCounter.get
      let op := operator (Random.keyOf seed invocation)
      let _ ← s.cudaTape
      let capsule ← s.selectedCapsule op
      let kernel ← s.cudaEntry capsule op
      s.takeRandomKey
      let parents ← Args.cuda ins args
      let id ← s.recordCUDAIO fun tape => do
        let (tape, result) ← kernel.record tape name (TensorTransfer.dtype (α := α)) parents
        pure (tape, result.map fun (id, _) => id)
      pure { id, identity? := some (← s.refIdentity) } : IO (TensorRef α sh)) := by
  unfold applyRandom
  have hcuda : s.options.usesCUDA = true := by
    show (s.options.device == .cuda) = true
    rw [hdevice]; rfl
  simp only [hcuda, ↓reduceIte]
  congr 1
  funext u
  congr 1
  funext invocation
  congr 1
  funext tape
  apply WeaklyLawfulMonadAttach.bind_congr
  intro c hc
  obtain ⟨-, hprovider, hcdevice⟩ := selectedCapsule_of_cuda hprofile hdevice hc
  rw [recordSelected_of_libTorch_cuda hprovider hcdevice]

end TorchLean.Session
