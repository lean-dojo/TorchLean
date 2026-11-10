/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.Layer.Recurrent
public import TorchLean.NN.Models.Mamba.Recurrence

/-!
# Mamba Layer

The trainable layer around the selective recurrence in `Mamba`. Every call starts from a zero
hidden state and an empty convolution history, so the layer keeps no sequence state between calls.
Streaming callers pass their own cache to `Mamba.run`.
-/

@[expose] public section

namespace TorchLean.nn.Layer

open TorchLean.Tensor
open Ops

/--
Selective Mamba-1 layer on a time-major sequence.

`hiddenWidth` is the output width. The convolution and recurrent path use
`innerWidth = options.expansion * hiddenWidth` channels, each with `options.stateWidth` diagonal
states. The kernel contains `options.kernelWidth` newest-first taps. Outputs depend only on the
current token and its prefix.

State order is `xProj, zProj, convKernel, convBias, dtProj, dtBias, logA, bProj, cProj, dSkip,
outProj`. Matrix orientations match `TorchLean.Mamba.SelectiveBlock`, and the positive Spec rate
tensor is `exp(logA)`. The time-step projection is dense, so importing a factorized reference
checkpoint requires multiplying its two time-step projection matrices.

Projection and kernel tensors use Xavier initialization, convolution bias starts at zero,
time-step bias gives an initial step of `0.01` when the projected feature is zero, the diagonal
rate magnitudes start at `1, ..., stateWidth`, and the skip coefficients start at one. The three
seed arguments control content/convolution, time-step/B/C, and gate/output initialization.

The recurrence is composed from generic differentiable operations, including the convolution, so
eager and typed graph execution differentiate the same program. The CUDA variable-coefficient
scan binding is not used here.
-/
def mamba (sequenceLength inputWidth hiddenWidth : Nat)
    (inputWeightSeed stateWeightSeed gateWeightSeed : Nat := 0)
    (options : Mamba.Options := {}) :
    Layer [sequenceLength, inputWidth] [sequenceLength, hiddenWidth] :=
  let innerWidth := options.expansion * hiddenWidth
  let stateWidth := options.stateWidth
  let kernelWidth := options.kernelWidth
  let x0 : Tensor Float [inputWidth, innerWidth] :=
    Init.tensor (.xavierUniform inputWidth innerWidth) inputWeightSeed
  let z0 : Tensor Float [inputWidth, innerWidth] :=
    Init.tensor (.xavierUniform inputWidth innerWidth) gateWeightSeed
  let kernel0 : Tensor Float [kernelWidth, innerWidth] :=
    Init.tensor (.xavierUniform kernelWidth kernelWidth) (inputWeightSeed + 1)
  let convBias0 := Tensor.zeros (α := Float) [innerWidth]
  let dt0 : Tensor Float [innerWidth, innerWidth] :=
    Init.tensor (.xavierUniform innerWidth innerWidth) stateWeightSeed
  let dtBiasValue := Float.log (Float.exp 0.01 - 1.0)
  let dtBias0 := Tensor.full [innerWidth] dtBiasValue
  let logA0 : Tensor Float [innerWidth, stateWidth] :=
    Tensor.dim fun _ => Tensor.dim fun n => Tensor.scalar (Float.log (Float.ofNat (n.val + 1)))
  let b0 : Tensor Float [innerWidth, stateWidth] :=
    Init.tensor (.xavierUniform innerWidth stateWidth) (stateWeightSeed + 1)
  let c0 : Tensor Float [innerWidth, stateWidth] :=
    Init.tensor (.xavierUniform innerWidth stateWidth) (stateWeightSeed + 2)
  let d0 := Tensor.ones (α := Float) [innerWidth]
  let out0 : Tensor Float [innerWidth, hiddenWidth] :=
    Init.tensor (.xavierUniform innerWidth hiddenWidth) (gateWeightSeed + 1)
  { kind := s!"Mamba({inputWidth}, {hiddenWidth}, state={stateWidth}, kernel={kernelWidth})"
    stateShapes :=
      [[inputWidth, innerWidth], [inputWidth, innerWidth], [kernelWidth, innerWidth],
       [innerWidth], [innerWidth, innerWidth], [innerWidth], [innerWidth, stateWidth],
       [innerWidth, stateWidth], [innerWidth, stateWidth], [innerWidth], [innerWidth, hiddenWidth]]
    initState := .cons x0 <| .cons z0 <| .cons kernel0 <| .cons convBias0 <|
      .cons dt0 <| .cons dtBias0 <| .cons logA0 <| .cons b0 <| .cons c0 <|
      .cons d0 <| .cons out0 .nil
    runtimeInit := some <|
      .cons (.scheme (.xavierUniform inputWidth innerWidth) inputWeightSeed) <|
      .cons (.scheme (.xavierUniform inputWidth innerWidth) gateWeightSeed) <|
      .cons (.scheme (.xavierUniform kernelWidth kernelWidth) (inputWeightSeed + 1)) <|
      .cons (.scheme .zeros) <|
      .cons (.scheme (.xavierUniform innerWidth innerWidth) stateWeightSeed) <|
      .cons (.flat (dtBias0.to FloatArray)) <|
      .cons (.flat (logA0.to FloatArray)) <|
      .cons (.scheme (.xavierUniform innerWidth stateWidth) (stateWeightSeed + 1)) <|
      .cons (.scheme (.xavierUniform innerWidth stateWidth) (stateWeightSeed + 2)) <|
      .cons (.scheme .ones) <|
      .cons (.scheme (.xavierUniform innerWidth hiddenWidth) (gateWeightSeed + 1)) .nil
    validateConfig := do
      validateRecurrent "Mamba" sequenceLength inputWidth hiddenWidth
      options.validate
    forward := fun _ {α} _ _ => fun {m} _ _ =>
      fun xProj zProj convKernel convBias dtProj dtBias logA bProj cProj dSkip outProj xs =>
        show m (Ref (m := m) (α := α) [sequenceLength, hiddenWidth]) from do
          let parameters : Mamba.Parameters (Ref (m := m) (α := α))
              inputWidth innerWidth stateWidth hiddenWidth kernelWidth :=
            ⟨xProj, zProj, convKernel, convBias, dtProj, dtBias, logA, bProj, cProj, dSkip, outProj⟩
          let initial ← Mamba.zeroState (m := m) (α := α)
          let rates ← Ops.apply (m := m) (α := α) .exp (logA, ())
          Prod.fst <$> scanRows (m := m) (α := α) initial fun t state => do
            let token ← Ops.apply (m := m) (α := α)
                (.indexAt (s := [sequenceLength, inputWidth]) ⟨0, Shape.AxisInBounds.proof⟩ t)
                (xs, ())
            Mamba.stepWithRates parameters rates state token }

end TorchLean.nn.Layer
