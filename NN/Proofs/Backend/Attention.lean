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

The default profile selects the declared LibTorch attention bridge. Its local VJP is compatible
with the TorchLean tape policy, and its reduction order remains implementation-defined.
This does not select or verify an ATen flash, efficient, cuDNN, or math implementation.
-/

@[expose] public section

namespace NN.Backend.BackendProfile

/-- The maintained CUDA registry and policy select the direct LibTorch attention capsule. -/
theorem checkedCuda_attention_choice :
    chooseCapsuleFor? checkedCuda.policy .scaledDotProductAttention
        (checkedCuda.availability.filterCapsules checkedCuda.registry) =
      some Attention.libTorchDirectAttention := by
  simp [chooseCapsuleFor?, checkedCuda, registry, Registry.flatten,
    Registry.maintainedModules, Attention.capsules, Availability.filterCapsules,
    Availability.admitsCapsule, Availability.cuda, Attention.libTorchDirectAttention,
    Attention.torchLeanComposed, KernelCapsule.admissible, KernelCapsule.contractsAligned,
    KernelCapsule.allowedBy, KernelCapsule.matchesPreference, KernelCapsule.matchesDevice,
    KernelCapsule.matchesVJP, ContractClaim.matchesObligation, ContractDescriptor.guarded,
    ContractDescriptor.tested, AssurancePolicy.checked, AssurancePolicy.acceptsTrust,
    BackendOp.requiresVJP]

/-- A successful default attention plan retains exactly the selected direct LibTorch capsule. -/
theorem checkedCuda_attention_plan
    {plan : KernelPlan}
    (hplan : checkedCuda.planOps #[.scaledDotProductAttention] = .ok plan) :
    plan.kernels =
      #[{ op := .scaledDotProductAttention, capsule := Attention.libTorchDirectAttention }] := by
  have hchoice :
      NN.Backend.planOp checkedCuda.policy
          (checkedCuda.availability.filterCapsules checkedCuda.registry)
          .scaledDotProductAttention =
        .ok { op := .scaledDotProductAttention, capsule := Attention.libTorchDirectAttention } := by
    simp [NN.Backend.planOp, checkedCuda_attention_choice, Pure.pure, Except.pure]
  unfold BackendProfile.planOps at hplan
  cases hvalid : Registry.validateModules checkedCuda.capsuleModules with
  | error err => simp [hvalid, Bind.bind, Except.bind] at hplan
  | ok valid =>
      have heq :
          ({ kernels :=
              #[{ op := .scaledDotProductAttention,
                  capsule := Attention.libTorchDirectAttention }] } : KernelPlan) = plan := by
        simpa [hvalid, planOpsAvailable, NN.Backend.planOps, hchoice,
          Bind.bind, Except.bind, Pure.pure, Except.pure] using hplan
      subst plan
      rfl

/--
The direct LibTorch attention capsule, which `checkedCuda_attention_choice` selects, declares a
backend local VJP compatible with the TorchLean tape request. Its numerical policy does not
promise a fixed reduction order.
-/
theorem checkedCuda_attention_contract :
    Attention.libTorchDirectAttention.provider = .libTorch ∧
      Attention.libTorchDirectAttention.vjpMode = .backendVJP ∧
      checkedCuda.policy.vjpMode = .torchLeanTape ∧
      Attention.libTorchDirectAttention.matchesVJP checkedCuda.policy = true ∧
      Attention.libTorchDirectAttention.numericalPolicy.reduction = .implementationDefined := by
  decide

end NN.Backend.BackendProfile
