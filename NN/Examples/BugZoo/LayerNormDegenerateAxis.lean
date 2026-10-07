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
contract small: the real-valued theorems record the algebra, and the concrete definitions below
instantiate the public TorchLean specification.

The algebraic statements use real arithmetic with totalized division and square root. They do not
assert finite native results for every epsilon or verify an external backward implementation.

-/

@[expose] public section

namespace NN.Examples.BugZoo.LayerNormDegenerateAxis

open TorchLean
open TorchLean.Tensor

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
def forward : Float :=
  (Spec.layerNorm (α := Float) (seqLen := 1) (embedDim := 1)
      [[1000000.0]]
      [2.0]
      [3.0]
      (by decide)
      (by decide)
      (0.00001 : Float))[((0 : Fin 1), (0 : Fin 1))]

/-- Both backward observations come from the same LayerNorm derivative specification. -/
def backward : Spec.NormalizationGradients Float [1, 1] [1] :=
  Spec.layerNormBackward (α := Float) (seqLen := 1) (embedDim := 1)
    (by decide)
    (by decide)
    [[1000000.0]]
    [2.0]
    [[1.0]]
    (0.00001 : Float)

end NN.Examples.BugZoo.LayerNormDegenerateAxis
