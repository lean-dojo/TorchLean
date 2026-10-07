/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Proofs.Backend.Dispatch
import Batteries.Lean.Except

/-!
# Maintained CUDA attention selection

The default profile selects a capsule declaring Lean-composed attention over LibTorch primitives.
Its declared VJP ownership matches the TorchLean tape policy; reduction order remains
implementation-defined.
This does not select or verify an ATen flash, efficient, cuDNN, or math implementation.
-/

@[expose] public section

namespace NN.Backend.BackendProfile

/-- The maintained CUDA registry and policy select the Lean-composed LibTorch attention capsule. -/
theorem checkedCuda_attention_choice :
    chooseCapsuleFor? checkedCuda.policy .attention
        (checkedCuda.availability.filterCapsules checkedCuda.registry) =
      some LibTorch.attention := by
  simp [chooseCapsuleFor?, checkedCuda, registry, Registry.flatten,
    Registry.maintainedModules, LibTorch.capsules, Availability.filterCapsules,
    Availability.admitsCapsule, Availability.cuda, LibTorch.attention,
    KernelCapsule.admissible, KernelCapsule.contractsAligned,
    KernelCapsule.allowedBy, KernelCapsule.matchesPreference, KernelCapsule.matchesDevice,
    KernelCapsule.matchesVJP, ContractClaim.matchesObligation, ContractDescriptor.guarded,
    ContractDescriptor.tested, AssurancePolicy.checked, AssurancePolicy.acceptsTrust,
    BackendOp.requiresVJP]

/-- A successful default attention plan retains exactly the selected Lean-composed capsule. -/
theorem checkedCuda_attention_plan
    {plan : KernelPlan}
    (hplan : checkedCuda.planOps #[.attention] = .ok plan) :
    plan.kernels =
      #[{ op := .attention, capsule := LibTorch.attention }] := by
  have hchoice :
      NN.Backend.planOp checkedCuda.policy
          (checkedCuda.availability.filterCapsules checkedCuda.registry)
          .attention =
        .ok { op := .attention, capsule := LibTorch.attention } := by
    simp [NN.Backend.planOp, checkedCuda_attention_choice, Pure.pure, Except.pure]
  unfold BackendProfile.planOps at hplan
  cases hvalid : Registry.validateModules checkedCuda.capsuleModules with
  | error err => simp [hvalid, Bind.bind, Except.bind] at hplan
  | ok valid =>
      have heq :
          ({ kernels :=
              #[{ op := .attention,
                  capsule := LibTorch.attention }] } : KernelPlan) = plan := by
        simpa [hvalid, planOpsAvailable, NN.Backend.planOps, hchoice,
          Bind.bind, Except.bind, Pure.pure, Except.pure] using hplan
      subst plan
      rfl

/--
The LibTorch attention capsule, which `checkedCuda_attention_choice` selects, declares a
Lean-composed local VJP compatible with the TorchLean tape request. Its numerical policy does not
promise a fixed reduction order.
-/
theorem checkedCuda_attention_contract :
    LibTorch.attention.provider = .libTorch ∧
      LibTorch.attention.vjpMode = .torchLeanTape ∧
      checkedCuda.policy.vjpMode = .torchLeanTape ∧
      LibTorch.attention.matchesVJP checkedCuda.policy = true ∧
      LibTorch.attention.numericalPolicy.reduction = .implementationDefined := by
  decide

end NN.Backend.BackendProfile
