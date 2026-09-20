/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Model.Layers.Recurrent
public import NN.Runtime.Autograd.Model.Mamba

/-!
# Mamba Layer

The trainable layer around the selective recurrence in `Model.Mamba`. Every call starts with zero
hidden state and empty convolution history. Streaming callers can use `Mamba.runArray` with an
explicit cache instead; the ordinary layer has no mutable sequence state.
-/

@[expose] public section

namespace Runtime.Autograd.Model.Layers

open Spec TorchLean TorchLean.Tensor

/--
Selective Mamba-1 layer on a time-major sequence.

`hiddenWidth` is the output width. The convolution and recurrent path use
`innerWidth = options.expansion * hiddenWidth` channels, each with `options.stateWidth` diagonal
states. The kernel contains `options.kernelWidth` newest-first taps. Outputs depend only on the
current token and its prefix.

State order is `xProj, zProj, convKernel, convBias, dtProj, dtBias, logA, bProj, cProj, dSkip,
outProj`. Matrix orientations match `Models.SelectiveMambaBlockSpec`; the positive Spec rate
tensor is `exp(logA)`. This is the dense time-step parameterization: importing a factorized
reference checkpoint requires multiplying its two time-step projection matrices.

Projection and kernel tensors use Xavier initialization, convolution bias starts at zero,
time-step bias gives an initial step of `0.01` when the projected feature is zero, the diagonal
rate magnitudes start at `1, ..., stateWidth`, and the skip coefficients start at one. The three
seed arguments control content/convolution, time-step/B/C, and gate/output initialization.

The recurrence is composed from generic differentiable operations, including the convolution.
Eager and typed graph execution therefore differentiate the same program. The CUDA
variable-coefficient scan binding is not used here.
-/
def mamba (sequenceLength inputWidth hiddenWidth : Nat)
    (inputWeightSeed stateWeightSeed gateWeightSeed : Nat := 0)
    (options : Mamba.Options := {}) :
    Layer [sequenceLength, inputWidth] [sequenceLength, hiddenWidth] :=
  let innerWidth := options.expansion * hiddenWidth
  let stateWidth := options.stateWidth
  let kernelWidth := options.kernelWidth
  let x0 : Tensor Float [inputWidth, innerWidth] :=
    Torch.Init.tensor (.xavierUniform inputWidth innerWidth) inputWeightSeed
  let z0 : Tensor Float [inputWidth, innerWidth] :=
    Torch.Init.tensor (.xavierUniform inputWidth innerWidth) gateWeightSeed
  let kernel0 : Tensor Float [kernelWidth, innerWidth] :=
    Torch.Init.tensor (.xavierUniform kernelWidth kernelWidth) (inputWeightSeed + 1)
  let convBias0 := Tensor.zeros (α := Float) [innerWidth]
  let dt0 : Tensor Float [innerWidth, innerWidth] :=
    Torch.Init.tensor (.xavierUniform innerWidth innerWidth) stateWeightSeed
  let dtBiasValue := Float.log (Float.exp 0.01 - 1.0)
  let dtBias0 := Tensor.full [innerWidth] dtBiasValue
  let logA0 : Tensor Float [innerWidth, stateWidth] :=
    Tensor.dim fun _ => Tensor.dim fun n => Tensor.scalar (Float.log (Float.ofNat (n.val + 1)))
  let b0 : Tensor Float [innerWidth, stateWidth] :=
    Torch.Init.tensor (.xavierUniform innerWidth stateWidth) (stateWeightSeed + 1)
  let c0 : Tensor Float [innerWidth, stateWidth] :=
    Torch.Init.tensor (.xavierUniform innerWidth stateWidth) (stateWeightSeed + 2)
  let d0 := Tensor.ones (α := Float) [innerWidth]
  let out0 : Tensor Float [innerWidth, hiddenWidth] :=
    Torch.Init.tensor (.xavierUniform innerWidth hiddenWidth) (gateWeightSeed + 1)
  { kind := s!"Mamba({inputWidth}, {hiddenWidth}, state={stateWidth}, kernel={kernelWidth})"
    stateShapes :=
      [[inputWidth, innerWidth], [inputWidth, innerWidth], [kernelWidth, innerWidth],
       [innerWidth], [innerWidth, innerWidth], [innerWidth], [innerWidth, stateWidth],
       [innerWidth, stateWidth], [innerWidth, stateWidth], [innerWidth], [innerWidth, hiddenWidth]]
    initState := .cons x0 <| .cons z0 <| .cons kernel0 <| .cons convBias0 <|
      .cons dt0 <| .cons dtBias0 <| .cons logA0 <| .cons b0 <| .cons c0 <|
      .cons d0 <| .cons out0 .nil
    runtimeInit := some <|
      .cons (.xavierUniform inputWidth innerWidth inputWeightSeed) <|
      .cons (.xavierUniform inputWidth innerWidth gateWeightSeed) <|
      .cons (.xavierUniform kernelWidth kernelWidth (inputWeightSeed + 1)) <|
      .cons .zeros <|
      .cons (.xavierUniform innerWidth innerWidth stateWeightSeed) <|
      .cons (.flat (dtBias0.to FloatArray)) <|
      .cons (.flat (logA0.to FloatArray)) <|
      .cons (.xavierUniform innerWidth stateWidth (stateWeightSeed + 1)) <|
      .cons (.xavierUniform innerWidth stateWidth (stateWeightSeed + 2)) <|
      .cons .ones <|
      .cons (.xavierUniform innerWidth hiddenWidth (gateWeightSeed + 1)) .nil
    validateConfig := do
      Internal.validateRecurrentDimensions "Mamba" sequenceLength inputWidth hiddenWidth
      options.validate
    forward := fun _ {α} _ _ => fun {m} _ _ =>
      fun xProj zProj convKernel convBias dtProj dtBias logA bProj cProj dSkip outProj xs =>
        show m (Ref [sequenceLength, hiddenWidth]) from do
          let parameters : Mamba.Parameters (Ref (m := m) (α := α))
              inputWidth innerWidth stateWidth hiddenWidth kernelWidth :=
            ⟨xProj, zProj, convKernel, convBias, dtProj, dtBias, logA, bProj, cProj, dSkip, outProj⟩
          let initial ← Mamba.zeroState (m := m) (α := α)
          let rates ← exp (m := m) (α := α) logA
          let empty ← const (m := m) (α := α)
            (Tensor.full [sequenceLength, hiddenWidth] (0 : α))
          let (_, output) ← (List.finRange sequenceLength).foldlM (init := (initial, empty))
            fun (state, acc) t => do
              let token ← select (m := m) (α := α) 0 xs t
              let (next, value) ← Mamba.Internal.stepWithRates parameters rates state token
              let written ← Internal.writeLeading (m := m) (α := α) acc value t
              pure (next, written)
          pure output }

end Runtime.Autograd.Model.Layers
