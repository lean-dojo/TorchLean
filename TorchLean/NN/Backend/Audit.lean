/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.Backend.Planner

/-!
# Backend Plan Audits

Inspection functions for contract-carrying kernel plans.

The planner chooses capsules. These functions read off what that choice means for trust
boundaries: how the shape, layout, value and VJP obligations of a selected capsule are supported
(`PlannedKernel.contracts`), and whether a selected capsule has the `trustedExternal`
classification (`KernelPlan.hasTrustedExternal`). The classifications describe declared evidence
and leave open whether Lean has proved the implementation, so a capsule classified `checked` may
run foreign code, as the LibTorch capsules do. Numerical certificates store kernel plans and
compare them on replay, which is why `KernelPlan` derives `DecidableEq`.
-/

@[expose] public section

namespace TorchLean
namespace Backend

namespace PlannedKernel

/-- The four contract descriptors of the selected capsule, paired with the field each one fills. -/
def contracts (k : PlannedKernel) : Array (Contract.Obligation × Contract.Descriptor) :=
  #[(.shape, k.capsule.shapeContract), (.layout, k.capsule.layoutContract),
    (.value, k.capsule.valueContract), (.vjp, k.capsule.vjpContract)]

end PlannedKernel

namespace KernelPlan

/-- Whether the capsule selected for any kernel of the plan has trust level `trustedExternal`. -/
def hasTrustedExternal (p : KernelPlan) : Bool :=
  p.kernels.any fun kernel => kernel.capsule.trustLevel == .trustedExternal

/-- Operator names whose selected capsules are trusted external. -/
def trustedExternalOps (p : KernelPlan) : Array String :=
  (p.kernels.filter fun kernel => kernel.capsule.trustLevel == .trustedExternal).map (·.op.tag)

end KernelPlan

end Backend
end TorchLean
