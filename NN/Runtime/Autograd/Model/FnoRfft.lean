/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Model.Fno
public import NN.Runtime.Autograd.Model.Functional.Spectral

/-!
# One-dimensional real-FFT FNO blocks

The dense reference and ATen real-FFT paths share every parameter. Each block stores real and
imaginary spectral weights, a pointwise skip matrix, and a bias, in that order. The spectral
weights have shape `[modes, width, width]`; unlike the full-DFT model, they do not store separate
negative-frequency parameters.
-/

@[expose] public section

namespace Runtime.Autograd.Model.Layers.FNO

open Spec TorchLean

/--
One-sided spectral block followed by a pointwise skip connection and an activation.

The default ReLU and parameter order agree with `Cuda.Fno1dRfft`. Comparisons must still
load the same parameter tensors: independently seeded builders need not draw the same values.
Changing only `path` preserves the function and the checkpoint layout.
-/
def rfftBlock (grid width modes : Nat)
    (hgrid : 0 < grid) (hwidth : 0 < width) (hmodes : modes ≤ grid / 2 + 1)
    (path : F.SpectralPath := .automatic) (activation : Activation.Kind := .relu)
    (spectralRealSeed spectralImagSeed skipWeightSeed : Nat := 0) :
    Layer [grid, width] [grid, width] :=
  let spectralShape : Shape := [modes, width, width]
  let skipShape : Shape := [width, width]
  let realWeight := Torch.Init.tensor (s := spectralShape) (sch := .uniform (-0.04) 0.04)
    (seed := spectralRealSeed)
  let imagWeight := Torch.Init.tensor (s := spectralShape) (sch := .uniform (-0.04) 0.04)
    (seed := spectralImagSeed)
  let skipWeight := Torch.Init.tensor (s := skipShape) (sch := .uniform (-0.04) 0.04)
    (seed := skipWeightSeed)
  let bias := Tensor.zeros (α := Float) [width]
  { kind := "FNO1dRfftBlock"
    stateShapes := [spectralShape, spectralShape, skipShape, [width]]
    initState := .cons realWeight (.cons imagWeight (.cons skipWeight (.cons bias .nil)))
    runtimeInit := some <| .cons (.uniform (-0.04) 0.04 spectralRealSeed) <|
      .cons (.uniform (-0.04) 0.04 spectralImagSeed) <|
      .cons (.uniform (-0.04) 0.04 skipWeightSeed) <| .cons .zeros .nil
    requiresGrad := #[true, true, true, true]
    forward := fun _ {α} _ _ => fun {m} _ _ => fun wr wi skip bias x =>
      (show m (RefTy (m := m) (α := α) [grid, width]) from do
        let spectral ← F.spectralConv1dRfft hgrid hwidth hmodes x wr wi path
        let pointwise ← matmul (batchA := []) (batchB := []) (batch := []) x skip
        let expandedBias ← broadcastTo (s₂ := [grid, width]) Shape.BroadcastTo.proof bias
        let biased ← add pointwise expandedBias
        let result ← add spectral biased
        match activation with
        | .relu => relu result
        | .tanh => tanh result
        | .gelu => Torch.gelu result
        | .silu => Torch.silu result
        | .sigmoid => sigmoid result) }

end Runtime.Autograd.Model.Layers.FNO
