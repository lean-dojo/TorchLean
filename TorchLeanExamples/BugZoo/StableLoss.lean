/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.VJPOp
public import TorchLean.Tensor.Elementwise
public import TorchLean.Tensor.Movement
public import TorchLean.Tensor.Layers.Dropout
public import TorchLean.Tensor.Layers.Embedding
public import TorchLean.Tensor.Layers.Linear
public import TorchLean.Tensor.Layers.Loss
public import TorchLean.Tensor.Layers.Loss.Axis
public import TorchLean.Tensor.Layers.Softmax

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

TorchLean cannot repair an arbitrary hand-written unstable loss after the fact. What it does is
give stable primitives and domain-aware variants a named place in the spec. For logits, users
should reach for `crossEntropy .logits`. It uses the log-softmax kernel behind
`Tensor.logSoftmax`, and when an unweighted log-probability overflows it rebuilds the weighted
term from the original logits. That avoids the fragile `softmax` followed by `log`, and the
zero-times-infinity of an overflowing logit difference. Its real formula and both derivatives are
proved in `TorchLean.NN.Softmax.Correctness`. Those are identities over `ℝ`, and two
floating-point schedules of the same formula can differ in their bits. In the same way, `safeDiv`
(the registered operator `TorchLean.Op.safeDiv`) puts epsilon-protected division into the graph as
a node of its own.

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
Tensor.crossEntropy .logits logits target dim
```

For division/domain bugs:

```python
# Risky when denom can be zero or denormal-sized.
y = x / denom
```

TorchLean makes the protected variant visible:

```lean
TorchLean.Tensor.safeDiv x denom
```

`safeDiv_eq_zipWith` shows the shifted denominator. Adding epsilon does not prevent division by
zero when `denom = -epsilon` and does not guarantee finite floating-point results, so callers must
establish the domain conditions for their inputs.
-/

@[expose] public section

open FloatLib.Numerics (MathFunctions)

namespace TorchLean.Examples.BugZoo.StableLoss

open TorchLean.Tensor

/--
Probability-space cross entropy clamps each predicted probability into `[epsilon, 1 - epsilon]`
before the logarithm. At the upper clamp it computes `log (1 - epsilon)` as `log1p (-epsilon)`, so
the term, about `-epsilon`, survives even where `1 - epsilon` rounds to one.

Logits go to `crossEntropy .logits`, and this form is for probabilities.
-/
theorem crossEntropy_probabilities {s : Shape} (dim : Fin s.rank)
    {α : Type} [Storage α] [Context α]
    (predicted target : Tensor α s) (epsilon : α) :
    Tensor.crossEntropy (.probabilities epsilon) predicted target dim =
      let logClamped := fun x : α =>
        let x := if x > epsilon then x else epsilon
        if x < (1 : α) - epsilon then MathFunctions.log x else Context.log1p (-epsilon)
      let logq := Tensor.map logClamped predicted
      let total := Tensor.sum (target * logq)
      Tensor.meanOverAxisSlices (s := s) dim (-total) := by
  simp [Tensor.crossEntropy]

/--
`Tensor.safeDiv x y` is a named operation of its own: it divides each entry of `x` by the matching
entry of `y` plus `Context.defaultEpsilon`.
-/
theorem safeDiv_eq_zipWith {s : Shape}
    {α : Type} [Storage α] [Context α] (x y : Tensor α s) :
    Tensor.safeDiv x y =
      Tensor.zipWith (fun a b => a / (b + Context.defaultEpsilon)) x y := by
  rfl

end TorchLean.Examples.BugZoo.StableLoss
