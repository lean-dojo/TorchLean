/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Layers.Loss
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
  https://conf.researchr.org/details/ase-2022/ase-2022-nier-track/18/An-Empirical-Study-on-Numerical-Bugs-in-Deep-Learning-Programs

TorchLean cannot repair an arbitrary hand-written unstable loss after the fact. The design instead
gives stable primitives and domain-aware variants a named place in the spec. For example,
`crossEntropyLogitsSpec` is the logits API users should reach for: it is defined through
`logSoftmaxSpec`, rather than through a fragile `softmax` followed by `log`. Likewise, `safedivSpec`
and `safeDivOp` make epsilon-protected division explicit in the graph.

Bug-shaped PyTorch sketches:

```python
# Unstable: softmax can round to 0, then log(0) gives -inf and the loss can become NaN.
probs = torch.softmax(logits, dim=-1)
loss = -(target * torch.log(probs)).sum()

# Safer: PyTorch's cross_entropy/log_softmax path.
loss = -(target * torch.log_softmax(logits, dim=-1)).sum()
```

TorchLean equivalent:

```lean
Spec.crossEntropyLogitsSpec logits target
```

For division/domain bugs:

```python
# Risky when denom can be zero or denormal-sized.
y = x / denom
```

TorchLean makes the protected variant visible:

```lean
Spec.Tensor.safedivSpec x denom
```

This does not prove every custom loss finite automatically, but it gives the library-approved stable
path a checked definition and a theorem below.
-/

@[expose] public section

namespace NN.Examples.BugZoo.StableLoss

open Spec
open Spec.Tensor

/--
The logits cross-entropy spec is literally the log-softmax form.

This small theorem is useful as a public “contract card”: if a model uses
`crossEntropyLogitsSpec`, the intended decomposition is stable-logits first, then target weighting
and mean reduction. That is the TorchLean answer to the TensorFuzz-style broken-cross-entropy class
inside the verified fragment.
-/
theorem crossEntropyLogits_uses_logSoftmax {s : Spec.Shape}
    {α : Type} [Context α] [DecidableRel ((· > ·) : α → α → Prop)] [LE α]
    (logits target : Spec.Tensor α s) :
    Spec.crossEntropyLogitsSpec logits target =
      let logp := Activation.logSoftmaxSpec (α := α) (s := s) logits
      let total := Spec.Tensor.sumSpec (Spec.Tensor.mulSpec target logp)
      Spec.meanOver (s := s) (-total) := by
  rfl

/--
The logits-loss gradient spec is the familiar `softmax(logits) - target`, scaled by the number of
entries.

Verified AD can only prove the gradient for the loss we actually specify. This theorem makes the
specified training signal visible, so a future implementation can be checked against this contract
instead of an informal “cross entropy” name.
-/
theorem crossEntropyLogitsDeriv_is_softmax_minus_target {s : Spec.Shape}
    {α : Type} [Context α]
    (logits target : Spec.Tensor α s) :
    Spec.crossEntropyLogitsDerivSpec logits target =
      Spec.Tensor.scaleSpec
        (Spec.Tensor.subSpec (Activation.softmaxSpec (α := α) (s := s) logits) target)
        (1 / (Spec.Shape.size s : α)) := by
  rfl

/--
Probability-space cross entropy clips the predicted probability before taking `log`.

This is a different API from logits cross entropy. We keep both because the safe choice depends on
what the caller has: logits should use `crossEntropyLogitsSpec`; already-normalized probabilities
should use the clipped probability form below.
-/
theorem crossEntropyProbabilities_clips_before_log {s : Spec.Shape}
    {α : Type} [Context α] [DecidableRel ((· > ·) : α → α → Prop)] [LE α]
    (predicted target : Spec.Tensor α s) (epsilon : α) :
    Spec.crossEntropySpec predicted target epsilon =
      let clamp01 := fun x : α =>
        let x := if x > epsilon then x else epsilon
        if x < (1 : α) - epsilon then x else (1 : α) - epsilon
      let q := Spec.Tensor.mapSpec clamp01 predicted
      let logq := Spec.Tensor.logSpec q
      let total := Spec.Tensor.sumSpec (Spec.Tensor.mulSpec target logq)
      Spec.meanOver (s := s) (-total) := by
  rfl

/-- Epsilon-protected division is a separate named tensor operation, not a hidden rewrite. -/
theorem safeDivSpec_unfold {s : Spec.Shape}
    {α : Type} [Context α] (x y : Spec.Tensor α s) :
    Spec.Tensor.safedivSpec x y =
      Spec.Tensor.map2Spec (fun a b => a / (b + Numbers.epsilon)) x y := by
  rfl

end NN.Examples.BugZoo.StableLoss
