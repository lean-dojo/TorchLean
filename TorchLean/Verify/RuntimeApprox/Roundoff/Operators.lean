/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/
module

public import TorchLean.Verify.RuntimeApprox.Constants
public import TorchLean.Verify.RuntimeApprox.Roundoff.Coverage
public import TorchLean.Verify.RuntimeApprox.NF.CROWN
public import TorchLean.Verify.NumericalCertificate
public import TorchLean.Verify.Floats.IEEE754.Expressions
public import TorchLean.Verify.RuntimeApprox.NF.Attention
public import TorchLean.Verify.RuntimeApprox.NF.Convolution
public import TorchLean.Verify.RuntimeApprox.NF.EndToEnd
public import TorchLean.Verify.RuntimeApprox.NF.Normalization
public import TorchLean.Verify.RuntimeApprox.NF.Optimizers
public import TorchLean.Verify.RuntimeApprox.NF.ReductionOps
public import TorchLean.Verify.Floats.IEEE754.Reductions
public import TorchLean.Verify.RuntimeApprox.Roundoff.Rules.Autograd -- shake: keep
public import TorchLean.Verify.RuntimeApprox.Roundoff.SoftmaxShift
public import TorchLean.Verify.RuntimeApprox.TransformerBlock
public import TorchLean.NN.Tape.LawfulNode
public import TorchLean.NN.Softmax.Correctness
public import TorchLean.NN.TypedGraph.WithData
public import FloatLib.Floats.Formats.BinaryInterchange.Configured
public import TorchLean.Tensor.ConfiguredBinary
public import TorchLean.NN.VJPOp
public import TorchLean.Tensor.Elementwise
public import TorchLean.Tensor.Movement
public import TorchLean.Tensor.Layers.Dropout
public import TorchLean.Tensor.Layers.Embedding
public import TorchLean.Tensor.Layers.Linear
public import TorchLean.Tensor.Layers.Loss
public import TorchLean.Tensor.Layers.Loss.Axis
public import TorchLean.Tensor.Layers.Softmax
public import TorchLean.Tensor.Op.Table
public import TorchLean.Tensor.LinearAlgebra.Algorithms
public import TorchLean.Tensor.Layers.Normalization
public meta import TorchLean.Tensor.Einops.Elab.Einsum
public meta import TorchLean.Tensor.Einops.Elab.Pack
public meta import TorchLean.Tensor.Literal
public meta import TorchLean.Tensor.Einops.Elab.Transform
public meta import TorchLean.Tensor.Tactic.Einops.Report
import FloatLib.Numerics.Enclosure.Elementary.Proof
public import FloatLib.Numerics.Automation.Attributes
public import TorchLean.Tensor.Literal
public import TorchLean.Tensor.Tactic.Init
public import TorchLean.Verify.RuntimeApprox.Roundoff
public import TorchLean.Verify.RuntimeApprox.Roundoff.Rules.Elementwise
public import TorchLean.Verify.RuntimeApprox.Roundoff.Rules.Movement
public import TorchLean.Verify.RuntimeApprox.Roundoff.Rules.Tensor

/-!
# Rounding error of tensor operators and their pullbacks

`roundoff` proves these bounds from the operators' own definitions, for every IEEE format and at
symbolic shapes, with no separate numerical specification. They cover elementwise arithmetic,
scaling and division with their pullbacks, matrix products and their input gradients, reductions
along an axis, normalization, the linear layer and mean squared error. They also cover
padded windows, `log1p`, `expm1`, the quotient by the sigmoid's denominator `1 + exp y`, the
logarithm of a positive sum, the exponential and sigmoid pullbacks, and log-softmax with its
gradients. Over the rationals the dot product and the matrix pullback are exact.

In binary32 a tensor divided by itself at the subnormal entry `2^-140` decodes to one, where an
isolated reciprocal overflows. The last theorems evaluate the binary32 log-softmax gradient and
`expm1` at stated inputs. At the logits `[0, -20]` the stable VJP keeps a small gradient that the
conventional subtraction rounds to zero, within `10^-16` of the real gradient.
-/

public section

namespace TorchLean.Roundoff

open FloatLib.Floats
open FloatLib.Floats.Formats.BinaryInterchange
open Model.Transcendentals FloatLib.Numerics

variable {format : FloatFormat}

local notation "Value" => ExecFloat.Binary format.expWidth format.fracWidth
  format.encoding format.exponentBias format.expWidth_ge_two format.fracWidth_pos
  format.exponentBias_pos format.exponentBias_le_maxFinite

/-- Tensor coordinates retain arbitrary shapes and use the same scalar proof generator. -/
theorem tensor_product_error {shape : Shape} (x y : Tensor Value shape)
    (i : shape.Coord) (hformat : format.isIEEE = true)
    (hresult : ExecFloat.Binary.isFinite ((x * y) i) = true) :
    |ExecFloat.Binary.toReal ((x * y) i) -
        ExecFloat.Binary.toReal (x i) * ExecFloat.Binary.toReal (y i)| ≤
      Model.halfUlpAt format (ExecFloat.Binary.toReal (x i) * ExecFloat.Binary.toReal (y i)) := by
  roundoff

/-- Analyze the pullback of `TorchLean.Op.scale`, `Tensor.scale seed factor`, itself, with no
alternative backward program. -/
theorem scale_pullback_error {n : Nat} (seed : Tensor Value [n]) (factor : Value)
    (i : Fin n) (hformat : format.isIEEE = true)
    (hresult : ExecFloat.Binary.isFinite ((Tensor.scale seed factor).getScalar i) = true) :
    |ExecFloat.Binary.toReal ((Tensor.scale seed factor).getScalar i) -
        ExecFloat.Binary.toReal (seed.getScalar i) * ExecFloat.Binary.toReal factor| ≤
      Model.halfUlpAt format
        (ExecFloat.Binary.toReal (seed.getScalar i) * ExecFloat.Binary.toReal factor) := by
  roundoff

/-- The pullback of `TorchLean.Op.div` to its numerator, `seed / rhs`, incurs one division
rounding, with no separately rounded reciprocal and no extra real-denominator assumption when the
inputs are exact observations. -/
theorem division_pullback_error {shape : Shape} (rhs seed : Tensor Value shape)
    (i : shape.Coord) (hformat : format.isIEEE = true)
    (hrhs : ExecFloat.Binary.isFinite (rhs i) = true)
    (hresult : ExecFloat.Binary.isFinite ((seed / rhs) i) = true) :
    |ExecFloat.Binary.toReal ((seed / rhs) i) -
        ExecFloat.Binary.toReal (seed i) / ExecFloat.Binary.toReal (rhs i)| ≤
      Model.halfUlpAt format
        (ExecFloat.Binary.toReal (seed i) / ExecFloat.Binary.toReal (rhs i)) := by
  roundoff

/-- For a subnormal denominator, an isolated reciprocal overflows although the weighted gradient
is exactly one. This checks the numerator pullback of `TorchLean.Op.div`, including its tensor
operations. -/
theorem division_pullback_subnormal :
    let tiny : Binary32 := Rat.cast (1 / 2 ^ 140 : Rat)
    let value : Tensor Binary32 [1] := Tensor.full [1] tiny
    ExecFloat.Binary.toRat? ((value / value).getScalar 0) = some 1 := by
  dsimp only
  roundoff
  all_goals decide +kernel

/-- In an IEEE format, when every step is finite, the sum of the scale pullback
`Tensor.scale seed factor` is within `foldError` of the real fold, one rounding per multiplication
and per addition. The bound reads the pullback's own code, with no separately written backward
implementation. -/
theorem summed_scale_pullback_error [Storage Value] {shape : Shape}
    (seed : Tensor Value shape) (factor : Value)
    (hformat : format.isIEEE = true)
    (hvalid : AllSteps (fun a b : Value => a + b * factor)
      (fun a b => ExecFloat.Binary.isFinite (a + b * factor) = true ∧
        ExecFloat.Binary.isFinite (b * factor) = true) 0 seed) :
    |ExecFloat.Binary.toReal (Tensor.sum (Tensor.scale seed factor)) -
        seed.foldl (fun a b => a + ExecFloat.Binary.toReal b * ExecFloat.Binary.toReal factor)
          (ExecFloat.Binary.toReal (0 : Value))| ≤
      foldError (fun a b : Value => a + b * factor)
        (fun a b => a + ExecFloat.Binary.toReal b * ExecFloat.Binary.toReal factor)
        (fun a _ error b => error +
          Model.halfUlpAt format (ExecFloat.Binary.toReal b * ExecFloat.Binary.toReal factor) +
          Model.halfUlpAt format (ExecFloat.Binary.toReal a + ExecFloat.Binary.toReal (b * factor)))
        0 (ExecFloat.Binary.toReal (0 : Value)) 0 seed := by
  roundoff

/-- The library's dot product uses the same proof generator across arbitrary tensor shapes. -/
theorem exact_dot_error [Storage Rat] {shape : Shape} (x y : Tensor Rat shape) :
    |((Tensor.dot x y : Rat) : ℝ) -
        Tensor.dot (x.map (fun value : Rat => (value : ℝ)))
          (y.map (fun value : Rat => (value : ℝ)))| ≤ 0 := by
  roundoff

/-- The matrix pullback uses only addition, multiplication and zero, so over the rationals it is
exact, with no transcendental context or separate gradient specification. -/
theorem exact_matmul_pullback_error [Storage Rat] {m n p : Nat}
    (left : Tensor Rat [m, n]) (right : Tensor Rat [n, p]) (seed : Tensor Rat [m, p])
    (row : Fin m) (column : Fin n) :
    |((Tensor.get2 (Tensor.matmulBackward (batchA := []) (batchB := []) (batch := [])
          (Shape.CanBroadcastTo.refl []) (Shape.CanBroadcastTo.refl []) left right
        seed).1 row column : Rat) : ℝ) -
      Tensor.get2 (Tensor.matmulBackward (batchA := []) (batchB := []) (batch := [])
          (Shape.CanBroadcastTo.refl []) (Shape.CanBroadcastTo.refl [])
          (left.map fun value : Rat => (value : ℝ)) (right.map fun value : Rat => (value : ℝ))
          (seed.map fun value : Rat => (value : ℝ))).1 row column| ≤ 0 := by
  roundoff

section Configured

variable {plan : Configured.StoragePlan format} {code : Type}
  [ExecFloat.ModelCodec plan (Model format) code]

local notation "ConfiguredValue" => ExecFloat (Configured.Family format code plan)

/-- Exact max/abs selection retains the rounding in the tensor that supplied its entries. -/
roundoff theorem max_abs_square_error :=
  fun [Storage ConfiguredValue] {shape : Shape} (input : Tensor ConfiguredValue shape) =>
    ExecFloat.Binary.toReal (Tensor.maxAbs (Tensor.square input))

/-- RMS normalization over the last axis of a matrix with symbolic dimensions, for the configured
scalar. -/
roundoff theorem rms_norm_forward_error :=
  fun [Storage ConfiguredValue] {rows columns : Nat}
    (input : Tensor ConfiguredValue [rows, columns])
    (scale : Tensor ConfiguredValue [columns]) (epsilon : ConfiguredValue)
    (row : Fin rows) (column : Fin columns) =>
  ExecFloat.Binary.toReal
    (Tensor.get2 (Tensor.rmsNorm input [⟨1, Nat.one_lt_two⟩] scale epsilon) row column)

/-- Both vector normalization schedules retain their existing scaling and ordered reduction. -/
roundoff theorem normalized_vector_error :=
  fun [Storage ConfiguredValue] {width : Nat}
    (input : Tensor ConfiguredValue [width]) (epsilon : ConfiguredValue)
    (mean : Bool) (coordinate : Fin width) =>
    ExecFloat.Binary.toReal
      ((Tensor.normalizeL2 input epsilon mean).getScalar coordinate)

/-- Batch dimensions and broadcasting stay symbolic, and each dot product keeps its left fold. -/
roundoff theorem batched_matmul_forward_error :=
  fun [Storage ConfiguredValue] {leftBatch rightBatch batch : Shape} {m n p : Nat}
    (leftBroadcast : Shape.CanBroadcastTo leftBatch batch)
    (rightBroadcast : Shape.CanBroadcastTo rightBatch batch)
    (left : Tensor ConfiguredValue (leftBatch ++ [m, n]))
    (right : Tensor ConfiguredValue (rightBatch ++ [n, p]))
    (index : (batch ++ [m, p]).Coord) =>
    ExecFloat.Binary.toReal
      (Tensor.batchMatmul leftBroadcast rightBroadcast left right index)

/-- Both public input gradients retain an arbitrary batch rank when the batch shapes agree. -/
roundoff theorem matmul_right_gradient_error :=
  fun [Storage ConfiguredValue] {batch : Shape} {m n p : Nat}
    (left : Tensor ConfiguredValue (batch ++ [m, n]))
    (right : Tensor ConfiguredValue (batch ++ [n, p]))
    (upstream : Tensor ConfiguredValue (batch ++ [m, p]))
    (index : (batch ++ [n, p]).Coord) =>
    ExecFloat.Binary.toReal
      ((Tensor.matmulBackward (Shape.CanBroadcastTo.refl batch)
        (Shape.CanBroadcastTo.refl batch) left right upstream).2 index)

roundoff theorem matmul_left_gradient_error :=
  fun [Storage ConfiguredValue] {batch : Shape} {m n p : Nat}
    (left : Tensor ConfiguredValue (batch ++ [m, n]))
    (right : Tensor ConfiguredValue (batch ++ [n, p]))
    (upstream : Tensor ConfiguredValue (batch ++ [m, p]))
    (index : (batch ++ [m, n]).Coord) =>
    ExecFloat.Binary.toReal
      ((Tensor.matmulBackward (Shape.CanBroadcastTo.refl batch)
        (Shape.CanBroadcastTo.refl batch) left right upstream).1 index)

/-- A broadcast operand's existing pullback accumulates contributions from every batch. -/
roundoff theorem broadcast_matmul_pullback_error :=
  fun [Storage ConfiguredValue] {b m n p : Nat}
    (left : Tensor ConfiguredValue [b, m, n]) (right : Tensor ConfiguredValue [1, n, p])
    (upstream : Tensor ConfiguredValue [b, m, p]) (index : Shape.Coord [1, n, p]) =>
    ExecFloat.Binary.toReal
      ((Tensor.matmulBackward (Shape.CanBroadcastTo.refl [b])
        (Shape.CanBroadcastTo.dim_one Shape.CanBroadcastTo.scalar) left right
          upstream).2 index)

/-- Reducing a computed tensor includes its pointwise arithmetic, for any valid axis. -/
roundoff theorem axis_square_sum_error :=
  fun [Storage ConfiguredValue] {shape : Shape} (axis : Nat)
    [Shape.AxisInBounds axis shape] (input : Tensor ConfiguredValue shape)
    (index : Shape.Coord (shape.eraseIdx axis)) =>
    ExecFloat.Binary.toReal
      (Tensor.castShape (Tensor.reduce .sum (Tensor.square input)
        [⟨axis, Shape.AxisInBounds.proof⟩] false) (Shape.reduce_singleton _ _) index)

/-- The one-axis sum's pullback copies the upstream entries along the removed axis. It is the
arm's pullback of `Op.reduce` over `axis` (`TorchLean.Op.reduce_sum_singleton`). -/
roundoff theorem axis_sum_pullback_error :=
  fun [Storage ConfiguredValue] {shape : Shape} (axis : Nat)
    [Shape.HasNonemptyAxis axis shape] [Shape.WellFormed shape]
    (input : Tensor ConfiguredValue shape)
    (upstream : Tensor ConfiguredValue (shape.eraseIdx axis))
    (index : shape.Coord) =>
    ExecFloat.Binary.toReal
      (Tensor.broadcast upstream shape _ (shape.eraseIdx_broadcastDims axis) index)

/-- A mean over one axis retains the arithmetic that produced each entry and the one rounding of
its division by the exact extent, as `Tensor.mean` computes it. -/
roundoff theorem axis_square_mean_error :=
  fun [Storage ConfiguredValue] {shape : Shape} (axis : Nat)
    (nonempty : Shape.NonemptyAxis axis shape) (input : Tensor ConfiguredValue shape)
    (index : Shape.Coord (shape.eraseIdx axis)) =>
    ExecFloat.Binary.toReal
      (Tensor.map (Context.divNat · (shape.axisSize axis (h := nonempty.axisInBounds)))
        (Tensor.castShape (Tensor.reduce .sum (Tensor.square input)
          [⟨axis, nonempty.axisInBounds.proof⟩] false) (Shape.reduce_singleton _ _)) index)

/-- The layer's weights and bias are inputs, and their arithmetic uses the selected scalar. -/
roundoff theorem linear_forward_error :=
  fun [Storage ConfiguredValue] {inDim outDim : Nat}
    (layer : Linear ConfiguredValue inDim outDim)
    (input : Tensor ConfiguredValue [inDim]) (index : Fin outDim) =>
    ExecFloat.Binary.toReal ((Linear.forward layer input).getScalar index)

/-- Mean-loss analysis includes the one rounding of the division by the exact tensor-size count. -/
roundoff theorem mse_forward_error :=
  fun [Storage ConfiguredValue] {shape : Shape}
    (input target : Tensor ConfiguredValue shape) =>
    ExecFloat.Binary.toReal ((Tensor.scalar (Tensor.meanSquaredError input target)).item)

/-- A nonlinear pullback composes the existing forward and backward code. The interval law
accounts for the actual cosh result and the input error from squaring. -/
roundoff theorem sinh_square_gradient_error :=
  fun [Storage ConfiguredValue] {shape : Shape}
    (input upstream : Tensor ConfiguredValue shape) (index : shape.Coord) =>
    ExecFloat.Binary.toReal
      (((Tensor.full shape (2 : ConfiguredValue) * input) *
        (Tensor.map MathFunctions.cosh (Tensor.square input) * upstream) :
          Tensor ConfiguredValue shape) index)

/-- The padded window keeps the input bound at valid indices and the padding bound elsewhere, at
symbolic sizes and offsets. -/
theorem window_error [Storage ConfiguredValue] {n length : Nat}
    (input : Tensor ConfiguredValue [n]) (reference : Tensor ℝ [n])
    (pad : ConfiguredValue) (padReference inputError padError : ℝ)
    (offset : Nat) (index : Fin length)
    (hinput : ∀ i,
      |ExecFloat.Binary.toReal (input.getScalar i) - reference.getScalar i| ≤ inputError)
    (hpad : ¬ (offset + index.val < n) →
      |ExecFloat.Binary.toReal pad - padReference| ≤ padError) :
    |ExecFloat.Binary.toReal ((Tensor.window input length offset pad).getScalar index) -
        (Tensor.window reference length offset padReference).getScalar index| ≤
      if offset + index.val < n then inputError else padError := by
  roundoff

/-- Bound the actual matrix pullback against its real-valued implementation, including errors
in the incoming gradient and captured weights. Each product and addition retains its own rounding
term. Range assumptions concern the accumulation states actually visited. -/
theorem matmul_pullback_error [Storage ConfiguredValue] {m n p : Nat}
    (input : Tensor ConfiguredValue [m, n]) (seed : Tensor ConfiguredValue [m, p])
    (right : Tensor ConfiguredValue [n, p])
    (seedReference : Tensor ℝ [m, p]) (rightReference : Tensor ℝ [n, p])
    (seedError rightError : ℝ)
    (hseed : ∀ index : Shape.Coord [m, p],
      |ExecFloat.Binary.toReal (seed index) - seedReference index| ≤ seedError)
    (hright : ∀ index : Shape.Coord [n, p],
      |ExecFloat.Binary.toReal (right index) - rightReference index| ≤ rightError)
    (row : Fin m) (column : Fin n) (hformat : format.isIEEE = true)
    (hvalid : AllSteps
      (fun state (k : Shape.Coord [p]) =>
        state + Tensor.get2 seed row k.1 * Tensor.get2 right column k.1)
      (fun state k => ExecFloat.Binary.isFinite
          (state + Tensor.get2 seed row k.1 * Tensor.get2 right column k.1) = true ∧
        ExecFloat.Binary.isFinite (Tensor.get2 seed row k.1 * Tensor.get2 right column k.1) = true)
      0 (Tensor.Rep.coordinates [p])) :
    |ExecFloat.Binary.toReal
        (Tensor.get2 (Tensor.matmulBackward (batchA := []) (batchB := []) (batch := [])
          (Shape.CanBroadcastTo.refl []) (Shape.CanBroadcastTo.refl []) input right
          seed).1 row column) -
      Tensor.get2 (Tensor.matmulBackward (batchA := []) (batchB := []) (batch := [])
          (Shape.CanBroadcastTo.refl []) (Shape.CanBroadcastTo.refl [])
          (input.map ExecFloat.Binary.toReal) rightReference seedReference).1 row column| ≤
      foldError
        (fun state (k : Shape.Coord [p]) =>
          state + Tensor.get2 seed row k.1 * Tensor.get2 right column k.1)
        (fun state k => state + Tensor.get2 seedReference row k.1 *
          Tensor.get2 rightReference column k.1)
        (fun state _ error k => error +
          (|ExecFloat.Binary.toReal (Tensor.get2 right column k.1)| * seedError +
            |Tensor.get2 seedReference row k.1| * rightError +
            Model.halfUlpAt format (ExecFloat.Binary.toReal (Tensor.get2 seed row k.1) *
              ExecFloat.Binary.toReal (Tensor.get2 right column k.1))) +
          Model.halfUlpAt format (ExecFloat.Binary.toReal state +
            ExecFloat.Binary.toReal (Tensor.get2 seed row k.1 * Tensor.get2 right column k.1)))
        0 0 0 (Tensor.Rep.coordinates [p]) := by
  roundoff

/-- The pullback of negation after scaling, `Tensor.neg (Tensor.scale seed factor)`, composes the
two pullbacks, including an uncertain incoming cotangent. -/
theorem composed_pullback_error [Storage ConfiguredValue] {shape : Shape}
    (seed : Tensor ConfiguredValue shape) (factor : ConfiguredValue)
    (i : shape.Coord) (reference error : ℝ) (hformat : format.isIEEE = true)
    (hresult : ExecFloat.Binary.isFinite (seed i * factor) = true)
    (hinput : |ExecFloat.Binary.toReal (seed i) - reference| ≤ error) :
    |ExecFloat.Binary.toReal (Tensor.neg (Tensor.scale seed factor) i) -
        -(reference * ExecFloat.Binary.toReal factor)| ≤
      |ExecFloat.Binary.toReal factor| * error +
        Model.halfUlpAt format (ExecFloat.Binary.toReal (seed i) *
          ExecFloat.Binary.toReal factor) := by
  roundoff

/-- The negation pullback `Tensor.neg seed` introduces no rounding error, for any configured codec.
The input cotangent may already have an error bound. -/
theorem negation_pullback_error [Storage ConfiguredValue] {shape : Shape}
    (seed : Tensor ConfiguredValue shape) (reference error : shape.Coord → ℝ)
    (hfinite : ∀ i, ExecFloat.Binary.isFinite (seed i) = true)
    (hseed : ∀ i, |ExecFloat.Binary.toReal (seed i) - reference i| ≤ error i) :
    ∀ i, |ExecFloat.Binary.toReal (Tensor.neg seed i) - -reference i| ≤ error i := by
  roundoff

/-- Decode a full reduction without unrolling its tensor or changing the accumulation order.
Exact model decoding also covers non-IEEE encodings and arbitrary storage codecs. -/
theorem decoded_sum {shape : Shape} (tensor : Tensor ConfiguredValue shape) :
    ExecFloat.Binary.toModel (Tensor.sum tensor) =
      Tensor.foldl Model.add (ExecFloat.Binary.toModel (0 : ConfiguredValue))
        (tensor.map ExecFloat.Binary.toModel) := by
  roundoff

/-- On an IEEE descriptor `log1p` rounds `log (1 + x)` once, so its error is half an ulp. -/
theorem log1p_error (x : ConfiguredValue) (hformat : format.isIEEE = true)
    (hx : ExecFloat.Binary.isFinite x = true) (hdomain : -1 < ExecFloat.Binary.toReal x)
    (hrange : |Real.log (1 + ExecFloat.Binary.toReal x)| < Certified.overflowThreshold format) :
    |ExecFloat.Binary.toReal (Context.log1p x) - Real.log (1 + ExecFloat.Binary.toReal x)| ≤
      Model.halfUlpAt format (Real.log (1 + ExecFloat.Binary.toReal x)) := by
  roundoff

/-- On an IEEE descriptor `expm1` rounds `exp x - 1` once, so its error is half an ulp. -/
theorem expm1_error (x : ConfiguredValue) (hformat : format.isIEEE = true)
    (hx : ExecFloat.Binary.isFinite x = true)
    (hrange : Real.exp (ExecFloat.Binary.toReal x) - 1 < Certified.overflowThreshold format) :
    |ExecFloat.Binary.toReal (Context.expm1 x) - (Real.exp (ExecFloat.Binary.toReal x) - 1)| ≤
      Model.halfUlpAt format (Real.exp (ExecFloat.Binary.toReal x) - 1) := by
  roundoff

/-- On an IEEE format, for finite `x` and `y`, the computed quotient `x / (1 + exp y)` by the
sigmoid's denominator is finite and has a mixed error against `x / (1 + Real.exp y)` on the decoded
inputs. The conditions, on those inputs only, keep `exp y` below the overflow threshold and the
divisor and the quotient in range, and make the enclosure of the exact `1 + exp y` exclude zero,
which proves the computed divisor nonzero. -/
roundoff mixed theorem quotient_by_one_add_exp_error :=
  fun (x y : ConfiguredValue) (_hformat : format.isIEEE = true)
    (_hx : ExecFloat.Binary.isFinite x = true) (_hy : ExecFloat.Binary.isFinite y = true) =>
    ExecFloat.Binary.toReal (x / (1 + MathFunctions.exp y))

/-- On an IEEE format, for positive finite `y` and `z`, the correctly rounded logarithm of the
rounded sum `y + z` is finite and has a mixed error against `Real.log (y + z)` on the decoded
inputs: the sum's relative error `u` becomes the absolute error `-log (1 - u)`, and the logarithm
adds one rounding. The conditions keep the sum in range and the exact logarithm, widened by
`-log (1 - u)`, below the overflow threshold. -/
roundoff mixed theorem logarithm_of_positive_sum_error :=
  fun (y z : ConfiguredValue) (_hformat : format.isIEEE = true)
    (_hy : ExecFloat.Binary.isFinite y = true) (_hz : ExecFloat.Binary.isFinite z = true)
    (_hy₀ : 0 < ExecFloat.Binary.toReal y) (_hz₀ : 0 < ExecFloat.Binary.toReal z) =>
    ExecFloat.Binary.toReal (MathFunctions.log (y + z))

/-- Analyze the existing exponential pullback, including errors in its input and upstream
cotangent. It rounds `exp x * seed` once, so only the product's range is assumed, not that
`exp x` is finite. Rounding once makes its budget half an ulp at the exact product instead of
the two roundings of the exponential and the product, and also changes which inputs give a
finite result (`Context.expMul`). The selected codec, tensor shape and tensor storage remain
parameters. -/
theorem exponential_pullback_error [Storage ConfiguredValue] {shape : Shape}
    (input seed : Tensor ConfiguredValue shape) (index : shape.Coord)
    (reference seedReference inputError seedError : ℝ)
    (hformat : format.isIEEE = true)
    (hx : ExecFloat.Binary.isFinite (input index) = true)
    (hseed : ExecFloat.Binary.isFinite (seed index) = true)
    (hrange : |Real.exp (ExecFloat.Binary.toReal (input index)) *
      ExecFloat.Binary.toReal (seed index)| < Certified.overflowThreshold format)
    (hinput : |ExecFloat.Binary.toReal (input index) - reference| ≤ inputError)
    (hupstream : |ExecFloat.Binary.toReal (seed index) - seedReference| ≤ seedError) :
    |ExecFloat.Binary.toReal (Tensor.zipWith Context.expMul input seed index) -
        Real.exp reference * seedReference| ≤
      |ExecFloat.Binary.toReal (seed index)| *
          (Real.exp (max (ExecFloat.Binary.toReal (input index)) reference) * inputError) +
        |Real.exp reference| * seedError +
          Model.halfUlpAt format (Real.exp (ExecFloat.Binary.toReal (input index)) *
            ExecFloat.Binary.toReal (seed index)) := by
  roundoff

/-- The forward bound is generated from the implementation, including its normalization sum
and `log1p`. It remains available as a reusable rule while checking the composed operator. -/
roundoff theorem log_softmax_forward_error :=
  fun [storage : Storage ConfiguredValue] {n : Nat}
    (input : Tensor ConfiguredValue [n + 1]) (index : Fin (n + 1)) =>
    ExecFloat.Binary.toReal ((LogSoftmax.vec input).getScalar index)

attribute [local roundoff_rule] log_softmax_forward_error

/-- Infer the range conditions, real reference and error budget from the existing log-softmax
VJP, including both branches and their reductions. No separate floating-point VJP is supplied. -/
roundoff theorem log_softmax_pullback_error :=
  fun [storage : Storage ConfiguredValue] {n : Nat}
    (y seed : Tensor ConfiguredValue [n + 1]) (index : Fin (n + 1)) =>
    ExecFloat.Binary.toReal
      ((LogSoftmax.vecBackward y seed).getScalar index)

/-- Check the public autograd operator, composing forward error with the backward arithmetic.
The registered forward theorem is reused without substituting another implementation. -/
roundoff theorem log_softmax_gradient_error :=
  fun [storage : Storage ConfiguredValue] {n : Nat}
    (input seed : Tensor ConfiguredValue [n + 1]) (index : Fin (n + 1)) =>
    ExecFloat.Binary.toReal
      ((Tensor.logSoftmaxBackward (Tensor.logSoftmax input ⟨0, Nat.one_pos⟩) seed
        ⟨0, Nat.one_pos⟩).getScalar index)

/-- The public sigmoid pullback, generated from its implementation: the checked fast product,
the enclosure recovery and the half-exponential coefficient replay, each under the branch that
selects it. -/
roundoff theorem sigmoid_gradient_error :=
  fun [Storage ConfiguredValue] {shape : Shape}
    (input seed : Tensor ConfiguredValue shape) (index : shape.Coord)
    (_hformat : format.isIEEE = true) =>
    ExecFloat.Binary.toReal (Tensor.sigmoidBackward input seed index)

end Configured

/-- At the `binary32` logits `[0, -20]` and the one-hot upstream gradient `[1, 0]`, entry `0` of
the stable log-softmax VJP is `9282611 / 2 ^ 52`, and the conventional subtraction
`seed 0 - exp (y 0) * sum seed` rounds it to zero. The proof also discharges the premises of the
generated `log_softmax_pullback_error` at this input by kernel evaluation (a range condition
follows from the finiteness of the correctly rounded result), but the statement does not include
its bound. -/
theorem log_softmax_gradient_binary32 :
    let input : Tensor Binary32 [2] := Tensor.ofFn fun i => if i = 0 then 0 else -20
    let seed : Tensor Binary32 [2] := Tensor.ofFn fun i => if i = 0 then 1 else 0
    let y := LogSoftmax.vec input
    ExecFloat.Binary.toRat?
        ((LogSoftmax.vecBackward y seed).getScalar 0) =
      some (9282611 / 2 ^ 52) ∧
    ExecFloat.Binary.toRat?
        (seed.getScalar 0 - MathFunctions.exp (y.getScalar 0) * Tensor.sum seed) = some 0 := by
  intro input seed y
  have bound := log_softmax_pullback_error y seed 0 (by
    repeat' first | apply And.intro | intro
    all_goals first
      | decide +kernel
      | exact (ExecFloat.Binary.CorrectlyRounded.isFinite_exp_iff (by decide +kernel)
          (budget := {})).mp (by decide +kernel)
      | exact (ExecFloat.Binary.CorrectlyRounded.isFinite_expMinus1_iff (by decide +kernel)
          (budget := {})).mp (by decide +kernel))
  clear bound
  decide +kernel

/-- `Real.exp (-20)` lies between `206115361 / 10 ^ 17` and `206115366 / 10 ^ 17`, by FloatLib's
degree-ten enclosure. Only the two rational endpoint comparisons are evaluated, and the exponential
stays a real. -/
private theorem exp_neg_twenty_bounds :
    (206115361 / 10 ^ 17 : ℝ) ≤ Real.exp (-20) ∧
      Real.exp (-20) ≤ (206115366 / 10 ^ 17 : ℝ) := by
  have enclosed := Enclosure.contains_exp (-20 : Rat) 10
  have lower : (206115361 / 10 ^ 17 : Rat) ≤ (Enclosure.exp (-20) 10).lo := by
    decide +kernel
  have upper : (Enclosure.exp (-20) 10).hi ≤ (206115366 / 10 ^ 17 : Rat) := by
    decide +kernel
  have lowerReal : (206115361 / 10 ^ 17 : ℝ) ≤
      ((Enclosure.exp (-20) 10).lo : ℝ) := by
    simpa only [Rat.cast_div, Rat.cast_pow, Rat.cast_ofNat] using
      (Rat.cast_le (K := ℝ)).mpr lower
  have upperReal : ((Enclosure.exp (-20) 10).hi : ℝ) ≤
      (206115366 / 10 ^ 17 : ℝ) := by
    simpa only [Rat.cast_div, Rat.cast_pow, Rat.cast_ofNat] using
      (Rat.cast_le (K := ℝ)).mpr upper
  norm_num only [Rat.cast_neg, Rat.cast_ofNat] at enclosed
  exact ⟨lowerReal.trans enclosed.1, enclosed.2.trans upperReal⟩

/-- The reference is the existing real public operator's VJP, using the existing derivative
correctness connection. No second backward implementation is introduced. -/
theorem log_softmax_real_gradient :
    let input : Tensor ℝ [2] := Tensor.ofFn fun i => if i = 0 then 0 else -20
    let seed : Tensor ℝ [2] := Tensor.ofFn fun i => if i = 0 then 1 else 0
    (Tensor.logSoftmaxBackward (Tensor.logSoftmax input ⟨0, Nat.one_pos⟩) seed
      ⟨0, Nat.one_pos⟩).getScalar 0 =
      Real.exp (-20) / (1 + Real.exp (-20)) := by
  intro input seed
  have connected := congrArg (fun value : Vec 2 => value 0)
    (Tensor.getScalarE_logSoftmaxBackward input seed)
  have coordinate : (Tensor.logSoftmaxBackward (Tensor.logSoftmax input ⟨0, Nat.one_pos⟩) seed
      ⟨0, Nat.one_pos⟩).getScalar 0 =
      1 - 1 / (1 + Real.exp (-20)) := by
    simpa [Vec.logSoftmaxVJP,
      Vec.softmax, Vec.sumExp, Fin.sum_univ_two,
      input, seed] using connected
  rw [coordinate]
  have denominator : (1 + Real.exp (-20)) ≠ 0 := by positivity
  field_simp
  ring

/-- The public binary32 VJP preserves this small original-logit gradient: its absolute error is at
most 10^-16, its relative error is at most 10^-7, and it is strictly positive. A zero result
cannot satisfy these conclusions, even though it would pass a fixed 10^-7 absolute tolerance. -/
theorem log_softmax_accuracy_binary32 :
    let input : Tensor Binary32 [2] := Tensor.ofFn fun i => if i = 0 then 0 else -20
    let seed : Tensor Binary32 [2] := Tensor.ofFn fun i => if i = 0 then 1 else 0
    let actual := ExecFloat.Binary.toReal
      ((Tensor.logSoftmaxBackward (Tensor.logSoftmax input ⟨0, Nat.one_pos⟩) seed
        ⟨0, Nat.one_pos⟩).getScalar 0)
    let reference := Real.exp (-20) / (1 + Real.exp (-20))
    |actual - reference| ≤ (1 / 10 ^ 16 : ℝ) ∧
      |actual - reference| ≤ (1 / 10 ^ 7 : ℝ) * |reference| ∧ 0 < actual := by
  intro input seed actual reference
  have encoded : ExecFloat.Binary.toRat?
      ((Tensor.logSoftmaxBackward (Tensor.logSoftmax input ⟨0, Nat.one_pos⟩) seed
        ⟨0, Nat.one_pos⟩).getScalar 0) =
      some (9282611 / 2 ^ 52) := by
    change ExecFloat.Binary.toRat?
      ((LogSoftmax.vecBackward
        (LogSoftmax.vec input) seed).getScalar 0) = some _
    exact log_softmax_gradient_binary32.1
  have observed : actual = (9282611 / 2 ^ 52 : ℝ) := by
    have decoded := ExecFloat.Binary.toReal_of_toRat?_eq_some encoded
    norm_num only at decoded ⊢
    exact decoded
  have exponential := exp_neg_twenty_bounds
  have denominator : 0 < 1 + Real.exp (-20) := by positivity
  have referenceBounds :
      (206115361 / 10 ^ 17 : ℝ) / (1 + 206115361 / 10 ^ 17) ≤ reference ∧
        reference ≤ (206115366 / 10 ^ 17 : ℝ) / (1 + 206115366 / 10 ^ 17) := by
    constructor
    · apply (div_le_div_iff₀ (by norm_num) denominator).mpr
      nlinarith [exponential.1]
    · apply (div_le_div_iff₀ denominator (by norm_num)).mpr
      nlinarith [exponential.2]
  have lowerReference : (1 / 10 ^ 9 : ℝ) < reference := by
    exact lt_of_lt_of_le (by norm_num) referenceBounds.1
  have absoluteError : |actual - reference| ≤ (1 / 10 ^ 16 : ℝ) := by
    rw [observed]
    apply abs_le.mpr
    constructor
    · have upper := referenceBounds.2.trans
        (show (206115366 / 10 ^ 17 : ℝ) / (1 + 206115366 / 10 ^ 17) ≤
          (9282611 / 2 ^ 52 : ℝ) + 1 / 10 ^ 16 by norm_num)
      linarith
    · have lower :=
        (show (9282611 / 2 ^ 52 : ℝ) - 1 / 10 ^ 16 ≤
          (206115361 / 10 ^ 17 : ℝ) / (1 + 206115361 / 10 ^ 17) by norm_num).trans
          referenceBounds.1
      linarith
  have referencePositive : 0 < reference := lt_trans (by norm_num) lowerReference
  refine ⟨absoluteError, ?_, ?_⟩
  · rw [abs_of_pos referencePositive]
    exact absoluteError.trans (by linarith)
  · exact positive_of_error_lt absoluteError (by linarith)

/-- Instantiate the arbitrary-format theorem at a `Binary32` input. Kernel evaluation shows the
correctly rounded result finite, which places `exp x - 1` below the overflow threshold. -/
theorem expm1_rounding_error_binary32 :
    let x : Binary32 := Rat.cast (-1 / 2 ^ 25 : Rat)
    |ExecFloat.Binary.toReal (Context.expm1 x) -
        (Real.exp (ExecFloat.Binary.toReal x) - 1)| ≤
      Model.halfUlpAt FloatFormat.binary32 (Real.exp (ExecFloat.Binary.toReal x) - 1) := by
  intro x
  have hx : ExecFloat.Binary.isFinite x = true := by decide +kernel
  have hrange := (ExecFloat.Binary.CorrectlyRounded.isFinite_expMinus1_iff hx
    (budget := {})).mp (by decide +kernel)
  exact expm1_error x rfl hx hrange

end TorchLean.Roundoff
