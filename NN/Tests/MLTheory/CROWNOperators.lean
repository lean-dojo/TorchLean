/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.MLTheory.CROWN.Operators.Arithmetic
public import NN.MLTheory.CROWN.Operators.Activations
public import NN.MLTheory.CROWN.Operators.BatchNorm
public import NN.MLTheory.CROWN.Operators.Trigonometric

/-!
# CROWN Operator Regressions

Concrete checks for interval and affine rules whose failures are easy to miss in larger verifier
tests. The soundness theorems remain the source of mathematical assurance; these checks protect the
executable formulas and their boundary cases.
-/

@[expose] public section

namespace NN.Tests.MLTheory.CROWNOperators

open NN.MLTheory.CROWN.Operators
open Spec TorchLean

def expectApprox (name : String) (actual expected : Float) (tol : Float := 1e-6) : IO Unit := do
  unless Float.abs (actual - expected) <= tol do
    throw <| IO.userError s!"{name}: expected {expected}, got {actual}"

def singletonTensor (x : Float) : Tensor Float [1] :=
  Tensor.ofFn fun _ => x

def singletonValue (x : Tensor Float [1]) : Float :=
  Tensor.getScalar x ⟨0, Nat.zero_lt_one⟩

def run : IO Unit := do
  let (_, _, slope, bias) := Arithmetic.affAbs (-1.0 : Float) 2.0
  expectApprox "abs crossing secant at lower endpoint" (slope * (-1.0) + bias) 1.0
  expectApprox "abs crossing secant at upper endpoint" (slope * 2.0 + bias) 2.0

  let below := Arithmetic.ibpClampScalar (-3.0 : Float) (-2.0) 0.0 1.0
  expectApprox "clamp interval below range lower" below.1 0.0
  expectApprox "clamp interval below range upper" below.2 0.0
  let above := Arithmetic.ibpClampScalar (2.0 : Float) 3.0 0.0 1.0
  expectApprox "clamp interval above range lower" above.1 1.0
  expectApprox "clamp interval above range upper" above.2 1.0
  let reversed := Arithmetic.ibpClampScalar (-3.0 : Float) 3.0 1.0 0.0
  expectApprox "clamp reversed bounds follows Spec.clampSpec lower" reversed.1 0.0
  expectApprox "clamp reversed bounds follows Spec.clampSpec upper" reversed.2 0.0

  let leaky := Activations.ibpLeakyReluScalar (-1.0 : Float) (-1.0) 2.0
  expectApprox "negative-slope leaky ReLU crossing lower" leaky.1 0.0
  expectApprox "negative-slope leaky ReLU crossing upper" leaky.2 2.0
  let (aLo, bLo, aHi, bHi) := Activations.affLeakyRelu (0.1 : Float) 0.0 0.0
  expectApprox "degenerate leaky ReLU lower slope" aLo 0.0
  expectApprox "degenerate leaky ReLU lower bias" bLo 0.0
  expectApprox "degenerate leaky ReLU upper slope" aHi 0.0
  expectApprox "degenerate leaky ReLU upper bias" bHi 0.0

  let bnParams : BatchNorm.BatchNormParams Float :=
    { dim := 1
      running_mean := singletonTensor 1.0
      running_var := singletonTensor (-4.0)
      gamma := singletonTensor 2.0
      beta := singletonTensor 3.0
      eps := 1.0 }
  expectApprox "BatchNorm CROWN scale uses spec variance totalization"
    (singletonValue (BatchNorm.computeScale bnParams)) 2.0
  expectApprox "BatchNorm CROWN offset uses spec variance totalization"
    (singletonValue (BatchNorm.computeOffset bnParams)) 1.0

  let product := Trigonometric.scalarIntervalMul (-1.0 : Float) 1.0 (-2.0) 3.0
  expectApprox "signed interval product lower" product.1 (-3.0)
  expectApprox "signed interval product upper" product.2 3.0

end NN.Tests.MLTheory.CROWNOperators
