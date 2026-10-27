/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Autograd.Ops

/-!
# BugZoo: numerically stable losses and domain-sensitive ops

TensorFuzz found rare-input failures that ordinary test sets missed, including broken loss
functions that produced `NaN` and quantization/full-precision disagreements:

- Odena, Olsson, Andersen, and Goodfellow, “TensorFuzz: Debugging Neural Networks with
  Coverage-Guided Fuzzing”, ICML 2019.
  https://proceedings.mlr.press/v97/odena19a.html

The numerical-bugs study by Wang et al. gives a complementary source of real PyTorch/TensorFlow
failures: invalid domains for `log`, `sqrt`, division, `exp`, and related math APIs:

- Wang et al., “An Empirical Study on Numerical Bugs in Deep Learning Programs”, ASE NIER 2022.
  https://doi.org/10.1145/3551349.3559561

TorchLean cannot repair an arbitrary hand-written unstable loss after the fact. The design instead
gives stable primitives and domain-aware variants a named place in the spec. For example,
`crossEntropyLogitsSpec` is the logits API users should reach for: it is defined through
`logSoftmaxSpec` and keeps the original logits for weighted terms that overflow. This avoids
the fragile `softmax` followed by `log`, and zero-times-infinity from an overflowing logit
difference. Its real formula and both derivatives are proved in
`NN.Proofs.Autograd.FDeriv.SoftmaxSpec`; real equivalence is not bitwise floating-point equality.
Likewise, `safedivSpec`
and `safeDivOp` make epsilon-protected division explicit in the graph.

Bug-shaped PyTorch sketches:

```python
# Unstable: softmax can round to 0, then log(0) gives -inf and the loss can become NaN.
probs = torch.softmax(logits, dim=-1)
loss = -(target * torch.log(probs)).sum()

# Avoids probability underflow, but a log-probability can still overflow on extreme logits.
loss = -(target * torch.log_softmax(logits, dim=-1)).sum()
```

TorchLean equivalent:

```lean
Spec.crossEntropyLogitsSpec axis logits target
```

For division/domain bugs:

```python
# Risky when denom can be zero or denormal-sized.
y = x / denom
```

TorchLean makes the protected variant visible:

```lean
TorchLean.Tensor.safedivSpec x denom
```

The checked definition and theorem below expose the shifted denominator. Adding epsilon does not
prevent division by zero when `denom = -epsilon`, nor guarantee finite floating-point results.
Callers must establish the domain conditions for their inputs.
-/

@[expose] public section

open TorchLean

namespace NN.Examples.BugZoo.StableLoss

open TorchLean.Tensor

/--
Probability-space cross entropy clips the predicted probability before taking `log`.

Pass logits to `crossEntropyLogitsSpec`; use this form for probabilities.
-/
theorem crossEntropyProbabilities_clips_before_log {s : Spec.Shape} (axis : Nat)
    [Spec.Shape.AxisInBounds axis s]
    {α : Type} [Storage α] [Context α]
    (predicted target : Tensor α s) (epsilon : α) :
    Spec.crossEntropySpec axis predicted target epsilon =
      let clamp01 := fun x : α =>
        let x := if x > epsilon then x else epsilon
        if x < (1 : α) - epsilon then x else (1 : α) - epsilon
      let q := Tensor.map clamp01 predicted
      let logq := Tensor.logSpec q
      let total := Tensor.sumSpec (Tensor.mulSpec target logq)
      Spec.meanOverAxisSlices (s := s) axis (-total) := by
  simp [Spec.crossEntropySpec]

/-- Epsilon-protected division is a separate named tensor operation, not a hidden rewrite. -/
theorem safeDivSpec_unfold {s : Spec.Shape}
    {α : Type} [Storage α] [Context α] (x y : Tensor α s) :
    Tensor.safedivSpec x y =
      Tensor.map2Spec (fun a b => a / (b + Context.defaultEpsilon)) x y := by
  rfl

end NN.Examples.BugZoo.StableLoss
