/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/
module

public import TorchLean.Verify.RuntimeApprox.Roundoff.Softmax
public import TorchLean.Verify.RuntimeApprox.Roundoff.CrossEntropy
public import FloatLib.Floats.Formats.BinaryInterchange.Configured
public import TorchLean.Tensor.Layers.Embedding
public import TorchLean.Verify.RuntimeApprox.Roundoff.Rules.Composite
public import TorchLean.Verify.RuntimeApprox.Roundoff.Rules.Movement
public import TorchLean.Verify.RuntimeApprox.Roundoff.Rules.Reduction

/-!
# Rounding error of cross-entropy of logits and of the embedding lookup

At finite logits `roundoff` reads each term of cross-entropy of logits through the
log-probabilities of its fiber (`CrossEntropy.log_probs_error`), and the embedding lookup is exact
at any index shape in every format, IEEE or not. The other tensor composites are analyzed by their
coverage theorems in `TorchLean.Coverage.TensorComposites`.
-/

public section

namespace TorchLean.Roundoff

open FloatLib.Floats FloatLib.Floats.Formats.BinaryInterchange

variable {format : FloatFormat} {plan : Configured.StoragePlan format} {code : Type}
  [ExecFloat.ModelCodec plan (Model format) code]

local notation "Value" => ExecFloat (Configured.Family format code plan)

/-! ## Composites -/

section

attribute [local roundoff_rule] softmax_error log_softmax_error CrossEntropy.log_probs_error

/-- Cross-entropy of logits with classes along any axis: each term reads the log-probabilities of
its fiber through `CrossEntropy.log_probs_error`, and the terms are summed in one flat fold. -/
roundoff theorem CrossEntropy.logits_error :=
  fun [Storage Value] {s : Shape} (dim : Fin s.rank) (x target : Tensor Value s)
    (_hformat : format.isIEEE = true) (_hfinite : ∀ c, ExecFloat.Binary.isFinite (x c) = true) =>
    ExecFloat.Binary.toReal (Tensor.crossEntropy .logits x target dim)

end

/-- The embedding lookup at indices of any shape reads one entry of the table and adds no error. -/
roundoff theorem embedding_error :=
  fun [Storage Value] {num dim : Nat} {t : Shape} (weight : Tensor Value [num, dim])
    (indices : Tensor (Fin num) t) (c : (t ++ [dim]).Coord) =>
    ExecFloat.Binary.toReal (Tensor.embedding weight indices c)

end TorchLean.Roundoff
