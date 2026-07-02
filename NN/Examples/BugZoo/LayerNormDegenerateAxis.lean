/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Layers.Normalization

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

```text
mean([x]) = x
var([x]) = 0
((x - mean) / sqrt(var + eps)) * weight + bias = bias
```

So reverse mode must report zero gradient for `weight` and zero input gradient. This file keeps the
contract small: the real-valued theorems record the algebra, and the concrete definitions below are
the public TorchLean spec terms used by the Python reproducer notes.
-/

@[expose] public section

namespace NN.Examples.BugZoo.LayerNormDegenerateAxis

open Spec
open Spec.Tensor

abbrev OneMat (α : Type) := Spec.Tensor α (.dim 1 (.dim 1 .scalar))
abbrev OneVec (α : Type) := Spec.Tensor α (.dim 1 .scalar)

def oneMat {α : Type} (x : α) : OneMat α :=
  Spec.Tensor.dim (fun _ => Spec.Tensor.dim (fun _ => Spec.Tensor.scalar x))

def oneVec {α : Type} (x : α) : OneVec α :=
  Spec.Tensor.dim (fun _ => Spec.Tensor.scalar x)

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

/-- The input gradient is zero because the one-feature LayerNorm forward is constant in the input. -/
theorem one_feature_layernorm_input_grad_contract (dy gamma invStd : ℝ) :
    invStd * ((dy * gamma) - (dy * gamma) - 0) = 0 := by
  ring

/-- TorchLean spec value for the public PyTorch repro: forward output. -/
def reproLayerNormForward : Float :=
  Spec.get2
    (Spec.layerNorm (α := Float) (seqLen := 1) (embedDim := 1)
      (oneMat (1000000.0 : Float))
      (oneVec (2.0 : Float))
      (oneVec (3.0 : Float))
      (by decide)
      (by decide)
      (0.00001 : Float))
    ⟨0, by decide⟩
    ⟨0, by decide⟩

/-- TorchLean spec value for the public PyTorch repro: gradient with respect to `weight`. -/
def reproLayerNormDWeight : Float :=
  Spec.Tensor.vecGet
    ((Spec.layerNormBackward (α := Float) (seqLen := 1) (embedDim := 1)
      (by decide)
      (by decide)
      (oneMat (1000000.0 : Float))
      (oneVec (2.0 : Float))
      (oneVec (3.0 : Float))
      (oneMat (1.0 : Float))
      (0.00001 : Float)).2.1)
    ⟨0, by decide⟩

/-- TorchLean spec value for the public PyTorch repro: gradient with respect to input. -/
def reproLayerNormDX : Float :=
  Spec.get2
    ((Spec.layerNormBackward (α := Float) (seqLen := 1) (embedDim := 1)
      (by decide)
      (by decide)
      (oneMat (1000000.0 : Float))
      (oneVec (2.0 : Float))
      (oneVec (3.0 : Float))
      (oneMat (1.0 : Float))
      (0.00001 : Float)).1)
    ⟨0, by decide⟩
    ⟨0, by decide⟩

end NN.Examples.BugZoo.LayerNormDegenerateAxis
