/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public meta import TorchLean.Tensor.Einops.Elab.Einsum
public meta import TorchLean.Tensor.Einops.Elab.Pack
public meta import TorchLean.Tensor.Literal
public meta import TorchLean.Tensor.Einops.Elab.Transform
public meta import TorchLean.Tensor.Tactic.Einops.Report
public import TorchLean.NN.Norm.LayerNorm.Adjoint
public import TorchLean.Tensor.Literal
public import TorchLean.Tensor.Layers.Normalization
public import TorchLean.Tensor.Numeric.InverseTrig.Real -- shake: keep
public import TorchLean.Tensor.Tactic.Init

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

Reverse mode must therefore report a zero gradient for `weight` and a zero input gradient. Over `ℝ`,
`scaleGrad_eq_zero` and `inputGrad_eq_zero` prove
this for `TorchLean.LayerNorm.backward` on a one-feature last axis, and
`affine_eq_bias` proves that every entry of `Tensor.layerNorm` is the bias.
`forward` and `backward` evaluate the same specification in `Float` at the input $10^6$.

The theorems use real arithmetic with totalized division and square root, so they hold for every
real epsilon. They do not assert finite native results or verify an external backward
implementation.
-/

@[expose] public section

namespace TorchLean.Examples.BugZoo.LayerNormAxis

open TorchLean.Tensor

/-- Over `ℝ`, LayerNorm over a one-feature last axis outputs the bias at every entry, for every
input, scale and epsilon. -/
theorem affine_eq_bias {seqLen : ℕ} (x : Tensor ℝ [seqLen, 1])
    (gamma beta : Tensor ℝ [1]) (epsilon : ℝ) (i : Fin seqLen) (j : Fin 1) :
    Tensor.get2 (Tensor.layerNorm x [⟨1, Nat.one_lt_two⟩] gamma beta epsilon) i j =
      getScalar beta j := by
  obtain rfl := Fin.fin_one_eq_zero j
  rw [Normalization.get2_layerNorm Nat.one_pos]
  simp

/-- Over `ℝ`, the LayerNorm backward rule on a one-feature last axis returns a zero scale
gradient, for every input, scale, upstream gradient and epsilon. -/
theorem scaleGrad_eq_zero {seqLen : ℕ} (hseq : 0 < seqLen)
    (x : Tensor ℝ [seqLen, 1]) (gamma : Tensor ℝ [1]) (dy : Tensor ℝ [seqLen, 1])
    (epsilon : ℝ) :
    (LayerNorm.backward hseq Nat.one_pos x gamma dy epsilon).scaleGradient = 0 := by
  rw [LayerNorm.backward_eq_backwardMat]
  refine Tensor.ext_vector fun j => ?_
  rw [LayerNorm.getScalar_scaleGradient]
  obtain rfl := Fin.fin_one_eq_zero j
  simp [Normalization.get2_lnCentered,
    Tensor.getScalar_eq_apply (0 : Tensor ℝ [1])]

/-- Over `ℝ`, the LayerNorm backward rule on a one-feature last axis returns a zero input
gradient, for every input, scale, upstream gradient and epsilon. -/
theorem inputGrad_eq_zero {seqLen : ℕ} (hseq : 0 < seqLen)
    (x : Tensor ℝ [seqLen, 1]) (gamma : Tensor ℝ [1]) (dy : Tensor ℝ [seqLen, 1])
    (epsilon : ℝ) :
    (LayerNorm.backward hseq Nat.one_pos x gamma dy epsilon).inputGradient = 0 := by
  rw [LayerNorm.backward_eq_backwardMat]
  refine Tensor.ext_get2 fun i j => ?_
  rw [LayerNorm.get2_inputGradient]
  obtain rfl := Fin.fin_one_eq_zero j
  simp [Normalization.get2_lnCentered,
    Tensor.get2_eq_apply (0 : Tensor ℝ [seqLen, 1])]

/-- The `Float` value of one-feature LayerNorm at input $10^6$, scale 2, bias 3 and epsilon
`1e-5`. -/
def forward : Float :=
  (Tensor.layerNorm (α := Float) ([[1000000.0]] : Tensor Float [1, 1]) [⟨1, Nat.one_lt_two⟩]
      ([2.0] : Tensor Float [1])
      ([3.0] : Tensor Float [1])
      (0.00001 : Float))[((0 : Fin 1), (0 : Fin 1))]

/-- The `Float` LayerNorm backward rule at the input and scale of `forward`, with upstream
gradient 1. -/
def backward : Normalization.Gradients Float [1, 1] [1] :=
  LayerNorm.backward (α := Float) (seqLen := 1) (embedDim := 1)
    (by decide)
    (by decide)
    [[1000000.0]]
    [2.0]
    [[1.0]]
    (0.00001 : Float)

end TorchLean.Examples.BugZoo.LayerNormAxis
