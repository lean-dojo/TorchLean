/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Engine.Cuda.Ops.Core
public import NN.Spec.Layers.RealFFT

/-!
# Native real-transform adjoints

Packed complex coordinates use the ordinary real dot product. If `w[k]` is one at DC/Nyquist
and two at interior frequencies, the adjoints are `R* g = n * irfft(g / w)` and
`I* g = (w / n) * rfft(g)`. Imaginary endpoint coordinates are zero in either adjoint.
The transforms are linear, so their JVPs apply the same forward transform to the tangent.
-/

@[expose] public section

namespace Runtime.Autograd.Cuda

open Spec

namespace Buffer

/-- Packed weights for a real-transform adjoint, repeated independently for each batch row. -/
def realFFTAdjointWeights (batch n : UInt32) (inverse : Bool) : Buffer :=
  ofFloatArray <| Id.run do
    let mut values := FloatArray.empty
    for _ in [:batch.toNat] do
      for frequency in [:n.toNat / 2 + 1] do
        let multiplicity := Float.ofNat (RealFFT.multiplicity n.toNat frequency)
        let weight := if inverse then multiplicity / Float.ofNat n.toNat
          else Float.ofNat n.toNat / multiplicity
        values := values.push weight
        values := values.push (if RealFFT.isEndpoint n.toNat frequency then 0 else weight)
    return values

/--
Adjoint of the unnormalized real transform for arbitrary packed cotangents.

The weights compensate for the conjugate pairs inserted by the normalized inverse. Both the
temporary weights and weighted spectrum are released after the inverse has consumed them.
-/
def rfft1dAdjoint (gradient : Buffer) (batch n : UInt32) : Buffer :=
  let weights := realFFTAdjointWeights batch n false
  let weighted := mul gradient weights
  let result := irfft1dPacked weighted batch n
  releaseThen weights <| releaseThen weighted result

/-- Adjoint of the normalized inverse, including exact zero imaginary endpoint gradients. -/
def irfft1dAdjoint (gradient : Buffer) (batch n : UInt32) : Buffer :=
  let transformed := rfft1dPacked gradient batch n
  let weights := realFFTAdjointWeights batch n true
  let result := mul transformed weights
  releaseThen transformed <| releaseThen weights result

end Buffer

namespace Tape

/-- Record a native packed real transform and its real-linear adjoint. -/
def rfft1d {batch n : Nat} (t : Tape) (xId : Nat) : Result (Tape × Nat) := do
  if n = 0 then throw "autograd: rfft1d: transform length must be positive"
  let batch32 ← AnyBuffer.natToU32Checked batch
  let n32 ← AnyBuffer.natToU32Checked n
  unary t "rfft1d" xId [batch, n] [batch, n / 2 + 1, 2]
    (fun x => Buffer.rfft1dPacked x batch32 n32)
    (fun _ gradient => Buffer.rfft1dAdjoint gradient batch32 n32)

/-- Record a normalized native inverse with its output length and packed-coordinate adjoint. -/
def irfft1d {batch n : Nat} (t : Tape) (xId : Nat) : Result (Tape × Nat) := do
  if n = 0 then throw "autograd: irfft1d: transform length must be positive"
  let batch32 ← AnyBuffer.natToU32Checked batch
  let n32 ← AnyBuffer.natToU32Checked n
  unary t "irfft1d" xId [batch, n / 2 + 1, 2] [batch, n]
    (fun x => Buffer.irfft1dPacked x batch32 n32)
    (fun _ gradient => Buffer.irfft1dAdjoint gradient batch32 n32)

end Tape
end Runtime.Autograd.Cuda
