/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Core.FloatInstances
public import Mathlib.Algebra.Order.Algebra
public import Mathlib.Analysis.SpecialFunctions.Pow.NNReal
public import Mathlib.Data.Sym.Sym2.Init
import Mathlib.Tactic.NormNum.GCD
import Mathlib.Tactic.Positivity.Finset
public import FloatLib.Floats.Formats.BinaryInterchange.Configured
public import FloatLib.Floats.Formats.BinaryInterchange.Conversion.Cast.Runtime
public import FloatLib.Floats.Formats.BinaryInterchange.Model.RealSemantics
public import FloatLib.Floats.Formats.BinaryInterchange.Model.ERealSemantics
public import FloatLib.Floats.Formats.IEEE754.Native
public import NN.Spec.Models.Mlp
-- Supplies the `ReLUMlpBridge` namespace opened below. `lake shake` cannot see an import that
-- only feeds an `open`, so it needs the annotation.
public import NN.MLTheory.Proofs.ReLU.Bridge.ReLUMlpBridge -- shake: keep

/-!
# Shared IEEE32Exec helpers for approximation theorems

This module contains the backend-generic glue used by the executable universal-approximation
theorems. We keep it independent of any one approximation construction (hinge sums, shallow ReLU
MLPs, convolutional models, or transformer-style models) and record the common semantic operations:

- extracting the scalar output of a one-output network,
- evaluating a two-layer ReLU MLP over the executable `ExecFloat.Binary 8 23` scalar type,
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

open FloatLib.Floats (ExecFloat)
open FloatLib.Floats.Formats.BinaryInterchange (Model FloatFormat)


namespace NN.MLTheory.Proofs.UniversalApproximation
namespace IEEE32ExecCore

open _root_.Spec _root_.TorchLean
open NN.MLTheory.Proofs.ReLUMlpBridge

noncomputable section

/--
Extract the only scalar from a one-output `ExecFloat.Binary 8 23` tensor.

Most approximation statements end with scalar-valued targets.  Keeping this as a named helper makes
the shape boundary explicit instead of hiding the `Fin 1` index proof at every call site.
-/
def extractScalarOutputIEEE32Exec (t : Tensor (ExecFloat.Binary 8 23) [1]) : ExecFloat.Binary 8 23
  :=
  t.getScalar ⟨0, by decide⟩

/--
Evaluate a two-layer ReLU MLP over executable IEEE binary32 semantics.

The corresponding real-valued evaluator is `mlpEval` from `ReLUMlpBridge`. Approximation
theorems compare `ExecFloat.Binary.toModel followed by Model.toReal (mlpEvalIEEE32Exec xI)` against
that real evaluator applied to
`tensorToReal xI`, thereby isolating the floating-point execution error from the approximation
and quantization errors.
-/
noncomputable def mlpEvalIEEE32Exec {n hidDim : Nat}
    (l1 : LinearSpec (ExecFloat.Binary 8 23) n hidDim) (l2 : LinearSpec (ExecFloat.Binary 8 23)
      hidDim 1)
    (x : Tensor (ExecFloat.Binary 8 23) [n]) : ExecFloat.Binary 8 23 :=
  extractScalarOutputIEEE32Exec (Examples.mlpForward l1 l2 x)

/--
Shape-preserving map over TorchLean specification tensors.

Lean's dependent tensor shape is part of the type, so the map is recursive over shapes rather than
implemented as a runtime loop. Coercions such as `tensorToReal` stay definitionally transparent in
downstream proofs.
-/
noncomputable def tensorMap {α β : Type} (f : α → β) {s : Shape}
    (tensor : Tensor α s) : Tensor β s :=
  Tensor.map f tensor

/--
Interpret an executable IEEE tensor as the exact real tensor denoted by its entries.

This is not a cast to mathematical reals with ideal arithmetic; it is the semantic interpretation
of the concrete IEEE value after execution has already happened.
-/
noncomputable def tensorToReal {s : Shape} (t : Tensor (ExecFloat.Binary 8 23) s) : Tensor ℝ s :=
  tensorMap (Model.toReal ∘ ExecFloat.Binary.toModel) t

/--
Interpret executable linear-layer parameters as real-valued parameters entrywise.

This helper is used in three-term bounds:
real approximation error + parameter quantization error + concrete execution error.
-/
noncomputable def linearSpecToReal {inDim outDim : Nat}
    (m : LinearSpec (ExecFloat.Binary 8 23) inDim outDim) : LinearSpec ℝ inDim outDim :=
  { weights := tensorToReal m.weights
    bias := tensorToReal m.bias }

end
end IEEE32ExecCore
end NN.MLTheory.Proofs.UniversalApproximation
