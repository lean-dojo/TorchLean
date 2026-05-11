/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Floats.IEEEExec.BridgeFP32Total
public import NN.MLTheory.Proofs.ReLU.Bridge.ReLUMlpBridge

/-!
# Shared IEEE32Exec helpers for approximation theorems

This module contains the backend-generic glue used by the executable universal-approximation
theorems. We keep it independent of any one approximation construction (hinge sums, shallow ReLU
MLPs, convolutional models, or transformer-style models) and record the common semantic operations:

- extracting the scalar output of a one-output network,
- evaluating a two-layer ReLU MLP over the executable `IEEE32Exec` scalar type,
- mapping executable IEEE binary32 values back to their real denotation, and
- interpreting executable linear-layer parameters as real parameters.

The mathematical role is the standard one in floating-point analysis: separate the exact real
network from the concrete IEEE-754 execution, then bridge the two by explicit rounding hypotheses
or verified rounding lemmas.  Useful references for this separation are IEEE Std 754-2019,
Goldberg's survey on floating-point arithmetic, and Higham's treatment of numerical error
analysis.  The ReLU approximation side is connected through `ReLUMlpBridge`, which supplies the
real-valued MLP semantics used by the universal-approximation files.
-/

@[expose] public section

namespace NN.MLTheory.Proofs.UniversalApproximation
namespace IEEE32ExecCore

open _root_.Spec
open NN.MLTheory.Proofs.ReLUMlpBridge
open TorchLean.Floats.IEEE754

noncomputable section

/--
Extract the only scalar from a one-output `IEEE32Exec` tensor.

Most approximation statements end with scalar-valued targets.  Keeping this as a named helper makes
the shape boundary explicit instead of hiding the `Fin 1` index proof at every call site.
-/
def extractScalarOutputIeee32exec (t : Tensor IEEE32Exec (.dim 1 .scalar)) : IEEE32Exec :=
  match t with
  | .dim f => Tensor.toScalar (f ⟨0, by decide⟩)

/--
Evaluate a two-layer ReLU MLP over executable IEEE binary32 semantics.

The corresponding real-valued evaluator is `mlpEvalNd` from `ReLUMlpBridge`.  Approximation
theorems compare `IEEE32Exec.toReal (mlpEvalNd xI)` against that real evaluator applied to
`tensorToReal xI`, thereby isolating the floating-point execution error from the approximation
and quantization errors.
-/
noncomputable def mlpEvalNdIeee32exec {n hidDim : Nat}
    (l1 : LinearSpec IEEE32Exec n hidDim) (l2 : LinearSpec IEEE32Exec hidDim 1)
    (x : Tensor IEEE32Exec (.dim n .scalar)) : IEEE32Exec :=
  extractScalarOutputIeee32exec (Examples.mlpForward l1 l2 x)

/--
Shape-preserving map over TorchLean specification tensors.

Lean's dependent tensor shape is part of the type, so the map is recursive over shapes rather than
implemented as a runtime loop. This keeps coercions such as `tensorToReal` definitionally
transparent in downstream proofs.
-/
noncomputable def tensorMap {α β : Type} (f : α → β) : {s : Shape} → Tensor α s → Tensor β s
  | .scalar, .scalar x => .scalar (f x)
  | .dim n s, .dim g => .dim (fun i : Fin n => tensorMap f (s := s) (g i))

/--
Interpret an executable IEEE tensor as the exact real tensor denoted by its entries.

This is not a cast to mathematical reals with ideal arithmetic; it is the semantic interpretation
of the concrete IEEE value after execution has already happened.
-/
noncomputable def tensorToReal {s : Shape} (t : Tensor IEEE32Exec s) : Tensor ℝ s :=
  tensorMap IEEE32Exec.toReal t

/--
Interpret executable linear-layer parameters as real-valued parameters entrywise.

This helper is used in three-term bounds:
real approximation error + parameter quantization error + concrete execution error.
-/
noncomputable def linearSpecToReal {inDim outDim : Nat}
    (m : LinearSpec IEEE32Exec inDim outDim) : LinearSpec ℝ inDim outDim :=
  { weights := tensorToReal m.weights
    bias := tensorToReal m.bias }

end
end IEEE32ExecCore
end NN.MLTheory.Proofs.UniversalApproximation
