/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Tensor.FromFloat
public import TorchLean.Tensor.Random
public import TorchLean.Tensor.Constructors
public import TorchLean.Tensor.Literal

/-!
# Deterministic parameter initialization

Pure, reproducible initializers for TorchLean tensors. A scheme draws `Float` samples, and
`tensor` converts each sample once through the element type's `FromFloat` instance. These
definitions are used when building model parameters before they enter a runtime session. CUDA
buffers made by `TorchLean.Init.cudaBufferOf` hold these samples, each
rounded once to the buffer dtype.

The formulas follow the corresponding PyTorch initializers:

* `Scheme.xavierUniform` uses the Glorot bound `sqrt (6 / (fanIn + fanOut))`.
* `Scheme.kaimingUniform` uses the ReLU-oriented He bound `sqrt (6 / fanIn)`.

References:

* Glorot and Bengio, *Understanding the difficulty of training deep feedforward neural networks*,
  AISTATS 2010.
* He et al., *Delving Deep into Rectifiers*, ICCV 2015.
* PyTorch initialization reference: https://pytorch.org/docs/stable/nn.init.html
-/

@[expose] public section

namespace TorchLean
namespace Init

open TorchLean.Tensor

/-- A deterministic initialization scheme whose samples are `Float` values. -/
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

/-- The bound `sqrt (6 / denominator)` for a positive fan, and `0` for a zero fan.

The denominator keeps each scheme's own casting and addition order. When it overflows to
infinity, the ordinary root is zero, and FloatLib then rounds the square root of the exact
rational ratio `6 / fan` without casting the count first. -/
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

/--
Xavier/Glorot uniform bound `sqrt (6 / (fanIn + fanOut))`.

A zero total fan has no meaningful random interval and gives `0`, which keeps scheme samples
finite and matches a weight with no elements. A count that overflows binary64 is handled before
the root is rounded, and the root may be representable.
-/
def xavierUniformLimit (fanIn fanOut : Nat) : Float :=
  uniformLimit (fanIn + fanOut) (Float.ofNat fanIn + Float.ofNat fanOut)

/--
Kaiming/He uniform bound `sqrt (6 / fanIn)`.

A zero fan-in uses the degenerate interval `[0, 0]`, avoiding infinities and `NaN` in host and
runtime initialization paths. Overflowing counts use the same exact-root recovery as Xavier.
-/
def kaimingUniformLimit (fanIn : Nat) : Float :=
  uniformLimit fanIn (Float.ofNat fanIn)

/-- Return sample `idx` from `sch`, using the counter-based stream determined by `seed`.

Uniform sampling uses the affine formula `lo + u * (hi - lo)` on a unit draw `u` unless the span
of two finite bounds overflows. The endpoints are then interpolated separately as
`(1 - u) * lo + u * hi`, which never forms that span. -/
def sampleAt (sch : Scheme) (seed idx : Nat) : Float :=
  let key := Random.keyOf seed 0
  let denominator : Nat := (2 : Nat) ^ 32
  let unit := Random.sampleUnit (α := Float)
    (Random.sampleNat key idx denominator) denominator
  match sch with
  | .zeros => 0.0
  | .ones => 1.0
  | .uniform lo hi =>
      let width := hi - lo
      if width.isFinite || !(lo.isFinite && hi.isFinite) then lo + unit * width
      else (1.0 - unit) * lo + unit * hi
  | .normal mean std =>
      mean + std * Random.normalScalar key idx
  | .xavierUniform fanIn fanOut =>
      let limit := xavierUniformLimit fanIn fanOut
      (-limit) + unit * (2.0 * limit)
  | .kaimingUniform fanIn =>
      let limit := kaimingUniformLimit fanIn
      (-limit) + unit * (2.0 * limit)

/--
Build a tensor from `scheme` in row-major order: flat index `i` receives
`ofFloat (sampleAt scheme seed i)`. The element type and the shape are normally inferred
from the expected tensor type.
-/
def tensor {α : Type} [TorchLean.Storage α] [FromFloat α] {shape : Shape}
    (scheme : Scheme) (seed : Nat := 0) : Tensor α shape :=
  Tensor.generateFlat shape (fun i => ofFloat (sampleAt scheme seed i))

/--
Xavier/Glorot uniform initialization, with gain `1`, of a linear weight matrix.

Both widths are normally inferred from the expected `Tensor α [outputWidth, inputWidth]` type.
-/
def xavierUniform {α : Type} [TorchLean.Storage α] [FromFloat α]
    {outputWidth inputWidth : Nat} (seed : Nat := 0) :
    Tensor α [outputWidth, inputWidth] :=
  tensor (.xavierUniform inputWidth outputWidth) seed

/--
Kaiming/He uniform initialization of a linear weight matrix used before a ReLU.

Both widths are normally inferred from the expected `Tensor α [outputWidth, inputWidth]` type.
-/
def kaimingUniform {α : Type} [TorchLean.Storage α] [FromFloat α]
    {outputWidth inputWidth : Nat} (seed : Nat := 0) :
    Tensor α [outputWidth, inputWidth] :=
  tensor (.kaimingUniform inputWidth) seed

end Init
end TorchLean
