/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Tensor
public import NN.Spec.Layers.Normalization
public import NN.Spec.Core.Scalar
public import NN.Core.Numeric -- shake: keep
public import NN.Core.Numeric.Angle.Real -- shake: keep

/-!
# BugZoo: LayerNorm on a one-feature axis

LayerNorm has a sharp degenerate case: if the normalized axis has length one, then the mean is the
single input value and the variance is zero. The normalized value is therefore exactly zero, so the
affine output is the bias and the forward result is independent of both the input and the scale.

PyTorch analogy:

```python
torch.nn.functional.layer_norm(x, normalized_shape=(1,), weight=w, bias=b)
```

For every finite scalar `x`, the mathematical contract is:

$$
\begin{aligned}
\operatorname{mean}[x] &= x,\\
\operatorname{var}[x] &= 0,\\
\left(\frac{x-\operatorname{mean}}{\sqrt{\operatorname{var}+\varepsilon}}\right)
  \operatorname{weight}+\operatorname{bias} &= \operatorname{bias}.
\end{aligned}
$$

So reverse mode must report zero gradient for `weight` and zero input gradient. This file keeps the
contract small: the real-valued theorems record the algebra, and the concrete definitions below are
the public TorchLean spec terms used by the Python reproducer notes.

The algebraic statements use real arithmetic with totalized division and square root. They do not
assert finite native results for every epsilon or verify an external backward implementation.

Run `python3 scripts/verification/normalization_contract_probe.py --device cpu` to sweep input
magnitudes and compare PyTorch forward and backward residuals in float32 and float64.
`--device cuda` probes the GPU implementation when available; observed numbers depend on the
PyTorch build and hardware.
-/

@[expose] public section

namespace NN.Examples.BugZoo.LayerNormDegenerateAxis

open TorchLean
open TorchLean.Tensor

/-- A one-by-one matrix: a single batch element with a single feature. -/
abbrev OneMat (α : Type) [Storage α] := Tensor α [1, 1]
/-- A length-one vector, the shape LayerNorm's `γ` and `β` take here. -/
abbrev OneVec (α : Type) [Storage α] := Tensor α [1]

/-- Build a one-by-one matrix from a scalar. -/
def oneMat {α : Type} [Storage α] (x : α) : OneMat α :=
  [[x]]

/-- Build a length-one vector from a scalar. -/
def oneVec {α : Type} [Storage α] (x : α) : OneVec α :=
  [x]

/--
The scalar algebra behind one-feature LayerNorm: normalization contributes zero, so the affine
result is the bias.
-/
theorem one_feature_layernorm_scalar_contract (x gamma beta epsilon : ℝ) :
    (((x - x) / MathFunctions.sqrt (Max.max (0 + epsilon) 0)) * gamma + beta) = beta := by
  simp

/-- The scale/weight gradient is zero because the normalized one-feature value is zero. -/
theorem one_feature_layernorm_scale_grad_contract (x dy epsilon : ℝ) :
    dy * ((x - x) / MathFunctions.sqrt (Max.max (0 + epsilon) 0)) = 0 := by
  simp

/--
The input gradient is zero because the one-feature LayerNorm forward is constant in the input.
-/
theorem one_feature_layernorm_input_grad_contract (dy gamma invStd : ℝ) :
    invStd * ((dy * gamma) - (dy * gamma) - 0) = 0 := by
  ring

/-- TorchLean spec value for the public PyTorch repro: forward output. -/
def reproLayerNormForward : Float :=
  (Spec.layerNorm (α := Float) (seqLen := 1) (embedDim := 1)
      (oneMat (1000000.0 : Float))
      (oneVec (2.0 : Float))
      (oneVec (3.0 : Float))
      (by decide)
      (by decide)
      (0.00001 : Float))[((0 : Fin 1), (0 : Fin 1))]

/-- TorchLean spec value for the public PyTorch repro: gradient with respect to `weight`. -/
def reproLayerNormDWeight : Float :=
  (Spec.layerNormBackward (α := Float) (seqLen := 1) (embedDim := 1)
      (by decide)
      (by decide)
      (oneMat (1000000.0 : Float))
      (oneVec (2.0 : Float))
      (oneMat (1.0 : Float))
      (0.00001 : Float)).scaleGradient[0]

/-- TorchLean spec value for the public PyTorch repro: gradient with respect to input. -/
def reproLayerNormDX : Float :=
  (Spec.layerNormBackward (α := Float) (seqLen := 1) (embedDim := 1)
      (by decide)
      (by decide)
      (oneMat (1000000.0 : Float))
      (oneVec (2.0 : Float))
      (oneMat (1.0 : Float))
      (0.00001 : Float)).inputGradient[((0 : Fin 1), (0 : Fin 1))]

end NN.Examples.BugZoo.LayerNormDegenerateAxis
