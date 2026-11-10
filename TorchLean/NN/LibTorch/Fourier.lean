/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.LibTorch.Activation
public import TorchLean.Tensor.Layers.RealFFT

/-!
# Native real-transform adjoints

Packed complex coordinates use the ordinary real dot product. If `w[k]` is one at DC/Nyquist
and two at interior frequencies, the adjoints are `R* g = n * irfft(g / w)` and
`I* g = (w / n) * rfft(g)`. Imaginary endpoint coordinates are zero in either adjoint.
The transforms are linear, so their JVPs apply the same forward transform to the tangent.
-/

@[expose] public section

namespace TorchLean.LibTorch

namespace Buffer

/-- Packed weights for a real-transform adjoint, repeated independently for each batch row. -/
def realFFTAdjointWeights (batch n : UInt32) (inverse : Bool) (dtype : Dtype) : Buffer :=
  ofFloatArray (Id.run do
    let mut values := FloatArray.empty
    for _ in [:batch.toNat] do
      for frequency in [:n.toNat / 2 + 1] do
        let multiplicity := Float.ofNat (RealFFT.multiplicity n.toNat frequency)
        let weight := if inverse then multiplicity / Float.ofNat n.toNat
          else 1.0 / multiplicity
        values := values.push weight
        values := values.push (if RealFFT.isEndpoint n.toNat frequency then 0 else weight)
    return values) dtype

/--
Adjoint of the unnormalized real transform for arbitrary packed cotangents.

The weights compensate for the conjugate pairs inserted by the unnormalized inverse. Both the
temporary weights and weighted spectrum are released after the inverse has consumed them.
-/
def rfftAdjoint (gradient : Buffer) (batch n : UInt32) : Buffer :=
  let weights := realFFTAdjointWeights batch n false (Buffer.dtype gradient)
  let weighted := mul gradient weights
  let result := irfftPackedUnnormalized weighted batch n
  releaseThen weights <| releaseThen weighted result

/-- Adjoint of the normalized inverse, including exact zero imaginary endpoint gradients. -/
def irfftAdjoint (gradient : Buffer) (batch n : UInt32) : Buffer :=
  let transformed := rfftPacked gradient batch n
  let weights := realFFTAdjointWeights batch n true (Buffer.dtype gradient)
  let result := mul transformed weights
  releaseThen transformed <| releaseThen weights result

end Buffer

end TorchLean.LibTorch
