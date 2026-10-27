/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Core.Random
public import NN.Tensor.Constructors
public import NN.Tensor.Internal.Elab.TensorLiteral

/-!
# Deterministic parameter initialization

Pure, reproducible initializers for `TorchLean.Tensor Float`. These definitions are used when
building model parameters before they enter a runtime session. Large runtime backends may provide
more specialized allocation paths, but they should implement the same initialization scheme.

The formulas follow the corresponding PyTorch initializers:

* `Scheme.xavierUniform` uses the Glorot bound `sqrt (6 / (fanIn + fanOut))`;
* `Scheme.kaimingUniform` uses the ReLU-oriented He bound `sqrt (6 / fanIn)`.

References:

* Glorot and Bengio, *Understanding the difficulty of training deep feedforward neural networks*,
  AISTATS 2010.
* He et al., *Delving Deep into Rectifiers*, ICCV 2015.
* PyTorch initialization reference: https://pytorch.org/docs/stable/nn.init.html
-/

@[expose] public section

namespace Runtime
namespace Autograd
namespace Torch
namespace Init

open Spec
open TorchLean TorchLean.Tensor

/-- A deterministic scheme for initializing a tensor of `Float` values. -/
inductive Scheme where
  | zeros
  | ones
  | uniform (lo hi : Float)
  | normal (mean std : Float)
  | xavierUniform (fanIn fanOut : Nat)
  | kaimingUniform (fanIn : Nat)
  deriving Repr

/-- Check the parameter domains of a declared initialization scheme. -/
def Scheme.validate : Scheme → Except String Unit
  | .zeros
  | .ones
  | .xavierUniform _ _
  | .kaimingUniform _ =>
      pure ()
  | .uniform lo hi => do
      unless lo.isFinite && hi.isFinite do
        throw "initialization: uniform bounds must be finite"
      unless lo ≤ hi do
        throw "initialization: uniform lower bound must not exceed the upper bound"
  | .normal mean std => do
      unless mean.isFinite && std.isFinite do
        throw "initialization: normal mean and standard deviation must be finite"
      unless 0.0 ≤ std do
        throw "initialization: normal standard deviation must be nonnegative"

namespace Internal

/-- Preserve the ordinary bound unless converting the fan count erased a nonzero root.

The denominator retains each scheme's original casting and addition order. On range loss,
FloatLib rounds the square root of the exact rational ratio without casting the count first. -/
def uniformLimit (fan : Nat) (denominator : Float) : Float :=
  if fan = 0 then 0
  else
    let ordinary := Float.sqrt (6.0 / denominator)
    if ordinary == 0 then
      FloatLib.Floats.ExecFloat.Binary.toFloat <|
        (FloatLib.Floats.ExecFloat.Binary.ofModel
          (FloatLib.Floats.Formats.BinaryInterchange.Model.AlgebraicRounding.sqrtRat
            FloatLib.Floats.Formats.BinaryInterchange.FloatFormat.binary64 false
            (6 / (fan : Rat))) : FloatLib.Floats.Binary64)
    else ordinary

end Internal

/--
Xavier/Glorot uniform bound.

The zero-fan case has no meaningful random interval. Returning zero keeps direct scheme use finite
and agrees with the empty-tensor behavior expected when a weight has no elements. A count that
overflows binary64 is handled before rounding the root, which may still be representable.
-/
def xavierUniformLimit (fanIn fanOut : Nat) : Float :=
  Internal.uniformLimit (fanIn + fanOut) (Float.ofNat fanIn + Float.ofNat fanOut)

/--
Kaiming/He uniform bound.

A zero fan-in uses the degenerate interval `[0, 0]`, avoiding infinities and `NaN` in host and
runtime initialization paths. Overflowing counts use the same exact-root recovery as Xavier.
-/
def kaimingUniformLimit (fanIn : Nat) : Float :=
  Internal.uniformLimit fanIn (Float.ofNat fanIn)

/-- Return sample `idx` from `sch`, using the counter-based stream determined by `seed`.

Uniform sampling keeps the ordinary affine formula unless the span of two finite bounds
overflows. Interpolating the endpoints separately then avoids materializing that span. -/
def sampleAt (sch : Scheme) (seed idx : Nat) : Float :=
  let key := Spec.Random.keyOf seed 0
  let denominator : Nat := (2 : Nat) ^ 32
  let unit := Spec.Random.sampleUnit (α := Float)
    (Spec.Random.sampleNat key idx denominator) denominator
  match sch with
  | .zeros => 0.0
  | .ones => 1.0
  | .uniform lo hi =>
      let width := hi - lo
      if width.isFinite || !(lo.isFinite && hi.isFinite) then lo + unit * width
      else (1.0 - unit) * lo + unit * hi
  | .normal mean std =>
      mean + std * Spec.Random.normalScalar key idx
  | .xavierUniform fanIn fanOut =>
      let limit := xavierUniformLimit fanIn fanOut
      (-limit) + unit * (2.0 * limit)
  | .kaimingUniform fanIn =>
      let limit := kaimingUniformLimit fanIn
      (-limit) + unit * (2.0 * limit)

/--
Initialize the row-major buffer directly, avoiding recursive subtensor construction.
Flat index `i` still receives `sampleAt sch seed i`, preserving the seed and sample order.
-/
def tensor (sch : Scheme) (seed : Nat := 0) : {s : Shape} → Tensor Float s
  | s => Tensor.generateFlat s (sampleAt sch seed)

/-- Initialize a matrix with the Xavier/Glorot uniform distribution and gain `1`. -/
def xavierUniform (outDim inDim : Nat) (seed : Nat := 0) :
    Tensor Float [outDim, inDim] :=
  tensor (s := .dim outDim (.dim inDim .scalar)) (.xavierUniform inDim outDim) seed

end Init
end Torch
end Autograd
end Runtime
