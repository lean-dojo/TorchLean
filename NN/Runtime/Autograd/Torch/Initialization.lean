/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Core.Random
public import NN.Tensor.Constructors
public import Mathlib.Algebra.Order.Field.Basic
import Mathlib.Tactic.NormNum.Inv
import Mathlib.Tactic.NormNum.Pow
import Mathlib.Tactic.Positivity.Finset
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

open Spec TorchLean
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

/-- Reject initializer parameters that would produce invalid floating-point samples. -/
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

/--
Xavier/Glorot uniform bound.

The zero-fan case has no meaningful random interval. Returning zero keeps direct scheme use finite
and agrees with the empty-tensor behavior expected when a weight has no elements.
-/
def xavierUniformLimit (fanIn fanOut : Nat) : Float :=
  if fanIn + fanOut = 0 then
    0.0
  else
    Float.sqrt (6.0 / (Float.ofNat fanIn + Float.ofNat fanOut))

/--
Kaiming/He uniform bound.

A zero fan-in uses the degenerate interval `[0, 0]`, avoiding infinities and `NaN` in host and
runtime initialization paths.
-/
def kaimingUniformLimit (fanIn : Nat) : Float :=
  if fanIn = 0 then
    0.0
  else
    Float.sqrt (6.0 / Float.ofNat fanIn)

/-- Return sample `idx` from `sch`, using the counter-based stream determined by `seed`. -/
def sampleAt (sch : Scheme) (seed idx : Nat) : Float :=
  let key := Spec.Random.keyOf seed 0
  let denominator : Nat := (2 : Nat) ^ 32
  let unit := Spec.Random.sampleUnit (α := Float)
    (Spec.Random.sampleNat key idx denominator) denominator
  match sch with
  | .zeros => 0.0
  | .ones => 1.0
  | .uniform lo hi =>
      lo + unit * (hi - lo)
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

/-- Initialize a matrix with the Kaiming/He uniform distribution for ReLU networks. -/
def kaimingUniform (outDim inDim : Nat) (seed : Nat := 0) :
    Tensor Float [outDim, inDim] :=
  tensor (s := .dim outDim (.dim inDim .scalar)) (.kaimingUniform inDim) seed

end Init
end Torch
end Autograd
end Runtime
