/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Tensor.ConfiguredBinary
-- Re-exported on purpose: the executable two-layer theorem evaluates
-- `UniversalApproximation.mlpEval` at configured binary values and at the reals.
public import TorchLean.NN.UniversalApproximation.TensorInput -- shake: keep

/-!
# Configured binary helpers for approximation theorems

Definitions shared by executable approximation theorems over a configured binary format: the real
denotation of executable tensors and the real interpretation of executable linear-layer parameters.
Neither depends on how the approximating network is built.

They follow the standard split of floating-point analysis: the exact real network and the concrete
execution are separate objects, related by rounding hypotheses or verified rounding lemmas. Useful
references for this separation are IEEE Std 754-2019, Goldberg's survey on floating-point
arithmetic, and Higham's treatment of numerical error analysis. The two-layer ReLU MLP is
`mlpEval`, one definition over any scalar context, which this module re-exports and
the theorems evaluate both at configured binary values and at the reals.
-/

@[expose] public section

open FloatLib.Floats (ExecFloat)
open FloatLib.Floats.Formats.BinaryInterchange


namespace TorchLean.UniversalApproximation

noncomputable section

variable {format : FloatFormat} {plan : Configured.StoragePlan format} {code : Type}
    [ExecFloat.ModelCodec plan (Model format) code]

local notation "Value" => ExecFloat (Configured.Family format code plan)

/--
Interpret a configured binary tensor entrywise through the total real denotation `Model.toReal`.

NaN and infinities map to `0`, so finiteness of the entries is a separate hypothesis.
-/
def tensorToReal {s : Shape} (t : Tensor Value s) : Tensor ℝ s :=
  Tensor.map (Model.toReal ∘ ExecFloat.Binary.toModel) t

/--
Interpret executable linear-layer parameters as real parameters, entrywise through `tensorToReal`.

The quantization and execution terms of the three-term bounds compare against the real MLP on
these parameters.
-/
def linearToReal {inDim outDim : Nat}
    (m : Linear Value inDim outDim) : Linear ℝ inDim outDim :=
  { weights := tensorToReal m.weights
    bias := tensorToReal m.bias }

end
end TorchLean.UniversalApproximation
