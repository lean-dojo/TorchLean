/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Engine.Cuda.Ops.Core

/-!
# CUDA Tape Operations: Elementwise Nodes
-/

@[expose] public section

namespace Runtime
namespace Autograd
namespace Cuda

open Spec TorchLean
open TorchLean TorchLean.Tensor

namespace Tape

/-!
## Elementwise ops

The backward closures below return newly allocated gradient buffers. When a derivative uses
intermediate CUDA buffers, it releases those intermediates before returning the final gradient. The
returned buffers are owned by the tape/gradient accumulator; workspace buffers are owned locally.
-/

/-- Pointwise addition node for two tensors with the same shape. -/
def add {s : Shape} (t : Tape) (aId bId : Nat) : Result (Tape × Nat) :=
  binary (t := t) "add" aId bId s s s
    (forward := Buffer.add)
    (backward := fun _a _b dLdy =>
      let da := Buffer.copy dLdy
      let zeros := Buffer.zeros (Buffer.size dLdy)
      let dbRaw := Buffer.axpy zeros dLdy 1.0
      let db := Buffer.releaseThen zeros dbRaw
      (da, db))

/-- Pointwise subtraction node for two tensors with the same shape. -/
def sub {s : Shape} (t : Tape) (aId bId : Nat) : Result (Tape × Nat) :=
  binary (t := t) "sub" aId bId s s s
    (forward := Buffer.sub)
    (backward := fun _a _b dLdy => (Buffer.copy dLdy, Buffer.scale dLdy (-1.0)))

/-- Pointwise multiplication node for two tensors with the same shape. -/
def mul {s : Shape} (t : Tape) (aId bId : Nat) : Result (Tape × Nat) :=
  binary (t := t) "mul" aId bId s s s
    (forward := Buffer.mul)
    (backward := fun a b dLdy => (Buffer.mul dLdy b, Buffer.mul dLdy a))

/-- Multiply by a scalar constant. -/
def scale {s : Shape} (t : Tape) (xId : Nat) (c : Float) : Result (Tape × Nat) :=
  unary (t := t) "scale" xId s s
    (forward := fun x => Buffer.scale x c)
    (backward := fun _x dLdy => Buffer.scale dLdy c)

/-- Pointwise absolute-value node. -/
def abs {s : Shape} (t : Tape) (xId : Nat) : Result (Tape × Nat) :=
  unary (t := t) "abs" xId s s
    (forward := Buffer.abs)
    (backward := fun x dLdy => Buffer.absBwd x dLdy)

/-- Pointwise square-root node using the CUDA buffer derivative convention. -/
def sqrt {s : Shape} (t : Tape) (xId : Nat) : Result (Tape × Nat) :=
  unary (t := t) "sqrt" xId s s
    (forward := Buffer.sqrt)
    (backward := fun x dLdy => Buffer.sqrtBwd x dLdy)

/-- Clamp each element to `[lo, hi]`. -/
def clamp {s : Shape} (t : Tape) (xId : Nat) (lo hi : Float) : Result (Tape × Nat) :=
  unary (t := t) "clamp" xId s s
    (forward := fun x => Buffer.clamp x lo hi)
    (backward := fun x dLdy => Buffer.clampBwd x dLdy lo hi)

/-- Pointwise maximum node; the backward rule splits ties according to `Buffer.maxBwd`. -/
def max {s : Shape} (t : Tape) (aId bId : Nat) : Result (Tape × Nat) :=
  binary (t := t) "max" aId bId s s s
    (forward := Buffer.max)
    (backward := fun a b dLdy => Buffer.maxBwd a b dLdy)

/-- Pointwise minimum node; the backward rule splits ties according to `Buffer.minBwd`. -/
def min {s : Shape} (t : Tape) (aId bId : Nat) : Result (Tape × Nat) :=
  binary (t := t) "min" aId bId s s s
    (forward := Buffer.min)
    (backward := fun a b dLdy => Buffer.minBwd a b dLdy)

/-- Pointwise division node with the usual quotient-rule backward closure. -/
def div {s : Shape} (t : Tape) (aId bId : Nat) : Result (Tape × Nat) :=
  binary (t := t) "div" aId bId s s s
    (forward := Buffer.div)
    (backward := fun a b dLdy =>
      let da := Buffer.div dLdy b
      let b2 := Buffer.mul b b
      let aOverB2 := Buffer.div a b2
      let dLdyA := Buffer.mul dLdy aOverB2
      let dbRaw := Buffer.scale dLdyA (-1.0)
      let db := Buffer.releaseThen b2 <| Buffer.releaseThen aOverB2 <|
        Buffer.releaseThen dLdyA dbRaw
      (da, db))

/-- Pointwise ReLU node with zero derivative on the nonpositive branch. -/
def relu {s : Shape} (t : Tape) (xId : Nat) : Result (Tape × Nat) :=
  unary (t := t) "relu" xId s s
    (forward := Buffer.relu)
    (backward := fun x dLdy => Buffer.reluBwd x dLdy)

/-- Pointwise exponential node; backward recomputes `exp x` as local workspace. -/
def exp {s : Shape} (t : Tape) (xId : Nat) : Result (Tape × Nat) :=
  unary (t := t) "exp" xId s s
    (forward := Buffer.exp)
    (backward := fun x dLdy =>
      let ex := Buffer.exp x
      Buffer.releaseThen ex <| Buffer.mul dLdy ex)

/--
Elementwise sine with VJP `cos(x) * dLdy`.

The cosine buffer belongs to this backward call and is released after multiplication. The input
and upstream gradient remain owned by the tape.
-/
def sin {s : Shape} (t : Tape) (xId : Nat) : Result (Tape × Nat) :=
  unary (t := t) "sin" xId s s
    (forward := Buffer.sin)
    (backward := fun x dLdy =>
      let derivative := Buffer.cos x
      Buffer.releaseThen derivative <| Buffer.mul derivative dLdy)

/-- Elementwise cosine with VJP `-sin(x) * dLdy`, releasing both temporary buffers. -/
def cos {s : Shape} (t : Tape) (xId : Nat) : Result (Tape × Nat) :=
  unary (t := t) "cos" xId s s
    (forward := Buffer.cos)
    (backward := fun x dLdy =>
      let sine := Buffer.sin x
      let derivative := Buffer.scale sine (-1.0)
      Buffer.releaseThen sine <| Buffer.releaseThen derivative <| Buffer.mul derivative dLdy)

/-- Pointwise natural-log node; callers are responsible for the positive-domain convention. -/
def log {s : Shape} (t : Tape) (xId : Nat) : Result (Tape × Nat) :=
  unary (t := t) "log" xId s s
    (forward := Buffer.log)
    (backward := fun x dLdy =>
      let invX := Buffer.inv x
      Buffer.releaseThen invX <| Buffer.mul dLdy invX)

/-- Elementwise reciprocal `1/x`. -/
def inv {s : Shape} (t : Tape) (xId : Nat) : Result (Tape × Nat) :=
  unary (t := t) "inv" xId s s
    (forward := Buffer.inv)
    (backward := fun x dLdy =>
      let invx := Buffer.inv x
      let invx2 := Buffer.mul invx invx
      let prod := Buffer.mul dLdy invx2
      Buffer.releaseThen invx <| Buffer.releaseThen invx2 <|
        Buffer.releaseThen prod <| Buffer.scale prod (-1.0))

/--
Elementwise "safe log" that protects against `log(0)` by adding a small `ε` internally.

Spec semantics: `log(softplus(x) + ε)`.
-/
def safeLog {s : Shape} (t : Tape) (xId : Nat) (ε : Float) : Result (Tape × Nat) := do
  let n ← AnyBuffer.numelU32 s
  unary (t := t) "safe_log" xId s s
    (forward := fun x =>
      let epsBuf := Buffer.full n ε
      let sp := softplusBuf x n
      let denom := Buffer.add sp epsBuf
      let y := Buffer.log denom
      Buffer.releaseThen epsBuf <| Buffer.releaseThen sp <| Buffer.releaseThen denom y)
    (backward := fun x dLdy =>
      let epsBuf := Buffer.full n ε
      let sp := softplusBuf x n
      let denom := Buffer.add sp epsBuf
      let sig := Buffer.sigmoid x
      let dlog := Buffer.div sig denom
      Buffer.releaseThen epsBuf <| Buffer.releaseThen sp <| Buffer.releaseThen denom <|
        Buffer.releaseThen sig <| Buffer.releaseThen dlog <| Buffer.mul dLdy dlog)

/-- Direct sigmoid values with TorchLean's VJP `dLdy * (y * (1 - y))`. -/
def sigmoid {s : Shape} (t : Tape) (xId : Nat) : Result (Tape × Nat) := do
  let n ← AnyBuffer.numelU32 s
  unary (t := t) "sigmoid" xId s s
    (forward := Buffer.sigmoid)
    (backward := fun x dLdy =>
      let y := Buffer.sigmoid x
      let ones := Buffer.full n 1.0
      let oneMinusY := Buffer.sub ones y
      let dy := Buffer.mul y oneMinusY
      Buffer.releaseThen y <| Buffer.releaseThen ones <| Buffer.releaseThen oneMinusY <|
        Buffer.releaseThen dy <| Buffer.mul dLdy dy)

/-- Direct hyperbolic tangent values with TorchLean's VJP `dLdy * (1 - y * y)`. -/
def tanh {s : Shape} (t : Tape) (xId : Nat) : Result (Tape × Nat) := do
  let n ← AnyBuffer.numelU32 s
  unary (t := t) "tanh" xId s s
    (forward := Buffer.tanh)
    (backward := fun x dLdy =>
      let y := Buffer.tanh x
      let ones := Buffer.full n 1.0
      let y2 := Buffer.mul y y
      let dy := Buffer.sub ones y2
      Buffer.releaseThen y <| Buffer.releaseThen ones <| Buffer.releaseThen y2 <|
        Buffer.releaseThen dy <| Buffer.mul dLdy dy)

/--
Tanh-approximate GELU as one CUDA tape node.

The native kernels fuse only the pointwise numerical work. TorchLean still records the node and
owns its VJP rule through `Activation.geluDerivSpec`.
-/
def gelu {s : Shape} (t : Tape) (xId : Nat) : Result (Tape × Nat) :=
  unary (t := t) "gelu" xId s s
    (forward := Buffer.gelu)
    (backward := fun x dLdy => Buffer.geluBwd x dLdy)

/-- Pointwise softplus node with sigmoid derivative. -/
def softplus {s : Shape} (t : Tape) (xId : Nat) : Result (Tape × Nat) := do
  let n ← AnyBuffer.numelU32 s
  unary (t := t) "softplus" xId s s
    (forward := fun x => softplusBuf x n)
    (backward := fun x dLdy =>
      let dy := Buffer.sigmoid x
      Buffer.releaseThen dy <| Buffer.mul dLdy dy)
end Tape

end Cuda
end Autograd
end Runtime
