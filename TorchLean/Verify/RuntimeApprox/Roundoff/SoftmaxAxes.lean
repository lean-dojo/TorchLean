/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/
module

public import TorchLean.Verify.RuntimeApprox.Roundoff.Softmax
public import TorchLean.Verify.RuntimeApprox.Roundoff.Rules.Broadcast

/-!
# Softmax and log-softmax at symbolic lengths, rows and axes

`roundoff` analyzes computations over shapes that are variables. A pattern match on a length that
the read coordinate shows positive is read at the successor length, so the vector log-softmax and
its VJP are analyzed at a length `n`, read at `i : Fin n`. The last-axis specifications are read
through the row of the coordinate (`Tensor.rowwise_apply`), so the last-axis log-softmax is
analyzed at every shape.

Softmax, log-softmax and the log-softmax VJP along any axis are read as their vector kernels on the
fibers through the coordinate (`Tensor.alongAxis₂_apply`). With the vector theorems of
`TorchLean.Roundoff` registered as rules, each is analyzed at every shape and axis
as that vector theorem at the fiber, restated at the successor of the fiber's length.
-/

public section

namespace TorchLean.Roundoff

open FloatLib.Floats FloatLib.Floats.Formats.BinaryInterchange FloatLib.Numerics

variable {format : FloatFormat} {plan : Configured.StoragePlan format} {code : Type}
  [ExecFloat.ModelCodec plan (Model format) code]

local notation "Value" => ExecFloat (Configured.Family format code plan)

/-- The stable log-softmax of a vector of any length `n`, at `i : Fin n`: the match on the length
is read at `n - 1 + 1`. -/
roundoff mixed theorem log_softmax_error' :=
  fun [Storage Value] {n : Nat} (t : Tensor Value [n]) (_hformat : format.isIEEE = true)
    (_hfinite : ∀ i, ExecFloat.Binary.isFinite (t.getScalar i) = true) (i : Fin n) =>
    ExecFloat.Binary.toReal ((LogSoftmax.vec t).getScalar i)

/-- The log-softmax VJP of vectors of any length `n`, at `i : Fin n`. -/
roundoff mixed theorem log_softmax_backward_error' :=
  fun [Storage Value] {n : Nat} (y v : Tensor Value [n]) (_hformat : format.isIEEE = true)
    (_hy : ∀ i, ExecFloat.Binary.isFinite (y.getScalar i) = true)
    (_hv : ∀ i, ExecFloat.Binary.isFinite (v.getScalar i) = true) (i : Fin n) =>
    ExecFloat.Binary.toReal ((LogSoftmax.vecBackward y v).getScalar i)

/-- The last-axis log-softmax of a tensor of any shape, at any coordinate: zero on a scalar and
the log-softmax of the row through the coordinate otherwise. -/
roundoff mixed theorem last_axis_log_softmax_error :=
  fun [Storage Value] {s : Shape} (x : Tensor Value s) (_hformat : format.isIEEE = true)
    (_hfinite : ∀ c, ExecFloat.Binary.isFinite (x c) = true) (p : s.Coord) =>
    ExecFloat.Binary.toReal (LogSoftmax.innermost x p)

section

attribute [local roundoff_rule] softmax_error log_softmax_error
  log_softmax_backward_error

/-- Softmax along `axis` of a tensor of any shape: `softmax_error` at the fiber
through the coordinate. -/
roundoff mixed theorem axis_softmax_error :=
  fun [Storage Value] {s : Shape} (axis : Nat) [Shape.AxisInBounds axis s] (x : Tensor Value s)
    (_hformat : format.isIEEE = true) (_hfinite : ∀ c, ExecFloat.Binary.isFinite (x c) = true)
    (c : s.Coord) =>
    ExecFloat.Binary.toReal ((Tensor.softmax x ⟨axis, Shape.AxisInBounds.proof⟩) c)

/-- Log-softmax along `axis` of a tensor of any shape: `log_softmax_error` at the
fiber through the coordinate. -/
roundoff mixed theorem axis_log_softmax_error :=
  fun [Storage Value] {s : Shape} (axis : Nat) [Shape.AxisInBounds axis s] (x : Tensor Value s)
    (_hformat : format.isIEEE = true) (_hfinite : ∀ c, ExecFloat.Binary.isFinite (x c) = true)
    (c : s.Coord) =>
    ExecFloat.Binary.toReal (Tensor.logSoftmax x ⟨axis, Shape.AxisInBounds.proof⟩ c)

/-- The log-softmax VJP along `axis` of tensors of any shape:
`log_softmax_backward_error` at the fibers through the coordinate. -/
roundoff mixed theorem axis_log_softmax_backward_error :=
  fun [Storage Value] {s : Shape} (axis : Nat) [Shape.AxisInBounds axis s] (y v : Tensor Value s)
    (_hformat : format.isIEEE = true) (_hy : ∀ c, ExecFloat.Binary.isFinite (y c) = true)
    (_hv : ∀ c, ExecFloat.Binary.isFinite (v c) = true) (c : s.Coord) =>
    ExecFloat.Binary.toReal (Tensor.logSoftmaxBackward y v ⟨axis, Shape.AxisInBounds.proof⟩ c)

end

end TorchLean.Roundoff
