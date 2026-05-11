/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Floats.NeuralFloat.Core
public import NN.Floats.NeuralFloat.Rounding

/-!
# Rounded scalar NN ops (`ℝ` + `neural_round`)

We collect *small* building blocks: common scalar functions used in neural networks,
defined in the “rounding-on-`ℝ`” style.

The pattern is always:

1. define the intended real-valued expression (e.g. `max x 0` for ReLU),
2. round it back to the target grid using `neural_round`.

This is the same modeling choice as `NF`:
we treat each primitive as a real computation followed by a rounding step.

If you need an error bound for any of these ops under round-to-nearest, you can usually get it
directly from the generic lemma `neural_error_bound_ulp` applied to the exact real expression.
-/

@[expose] public section


namespace TorchLean.Floats

namespace NNOps

variable {β : NeuralRadix} {fexp : ℤ → ℤ} [NeuralValidExp fexp]

/--
Round the result of a real-valued scalar function.

This is the basic adapter used throughout this file.
-/
@[inline] noncomputable def round1 (rnd : ℝ → ℤ) [NeuralValidRnd rnd] (g : ℝ → ℝ) (x : ℝ) : ℝ :=
  neuralRound (β := β) (fexp := fexp) rnd (g x)

/--
Round the result of a two-argument real-valued function.

This is handy for “activation-like” helpers that take a parameter (e.g. leaky ReLU slope).
-/
@[inline] noncomputable def round2 (rnd : ℝ → ℤ) [NeuralValidRnd rnd] (g : ℝ → ℝ → ℝ) (x y : ℝ) : ℝ
  :=
  neuralRound (β := β) (fexp := fexp) rnd (g x y)

/--
ReLU (rounded).

Mathematically, ReLU is `relu(x) = max(x, 0)`. Here we evaluate that in `ℝ` and then round.

This is close in spirit to what happens in practice: even if a framework fuses ops, the overall
effect is still “some real expression, then it gets rounded to the destination format”.
-/
noncomputable def neuralRelu (rnd : ℝ → ℤ) [NeuralValidRnd rnd] (x : ℝ) : ℝ :=
  round1 (β := β) (fexp := fexp) rnd (fun t => max t 0) x

/--
Leaky ReLU (rounded).

Definition (PyTorch `torch.nn.functional.leaky_relu`):

- if `x > 0` then `x`
- else `negative_slope * x`.
-/
noncomputable def neuralLeakyRelu (rnd : ℝ → ℤ) [NeuralValidRnd rnd] (negative_slope : ℝ) (x : ℝ)
  : ℝ :=
  round2 (β := β) (fexp := fexp) rnd (fun a t => if t > 0 then t else a * t) negative_slope x

/--
Sigmoid (rounded), with the usual stable piecewise definition.

We define the exact real function as:

- if `x ≥ 0`, use `1 / (1 + exp(-x))`
- else use `exp(x) / (1 + exp(x))`

The two branches are algebraically equal, but the piecewise form avoids overflow in `exp`.
-/
noncomputable def neuralSigmoid (rnd : ℝ → ℤ) [NeuralValidRnd rnd] (x : ℝ) : ℝ :=
  let stable_sigmoid := if x ≥ 0 then
    1 / (1 + Real.exp (-x))
  else
    Real.exp x / (1 + Real.exp x)
  neuralRound (β := β) (fexp := fexp) rnd stable_sigmoid

/--
`tanh` (rounded).

This is just “evaluate `Real.tanh` then round”. Any round-to-nearest error bound is inherited from
`neural_error_bound_ulp` applied at `Real.tanh x`.
-/
noncomputable def neuralTanh (rnd : ℝ → ℤ) [NeuralValidRnd rnd] (x : ℝ) : ℝ :=
  neuralRound (β := β) (fexp := fexp) rnd (Real.tanh x)

/--
SiLU / Swish (rounded).

PyTorch analogies: `torch.nn.functional.silu` / `torch.nn.SiLU`.

Definition: `silu(x) = x * sigmoid(x)`. We use the same stable sigmoid expression as
`neural_sigmoid`, then round the final result.
-/
noncomputable def neuralSilu (rnd : ℝ → ℤ) [NeuralValidRnd rnd] (x : ℝ) : ℝ :=
  let s := if x ≥ 0 then
    1 / (1 + Real.exp (-x))
  else
    Real.exp x / (1 + Real.exp x)
  neuralRound (β := β) (fexp := fexp) rnd (x * s)

/--
Softplus (rounded), with a stable piecewise definition.

PyTorch analogy: `torch.nn.functional.softplus`.

Naively, `softplus(x) = log(1 + exp(x))`. The piecewise form avoids catastrophic overflow in
`exp(x)` when `x` is large and positive:

- if `x > 0`, use `x + log(1 + exp(-x))`
- else use `log(1 + exp(x))`.
-/
noncomputable def neuralSoftplus (rnd : ℝ → ℤ) [NeuralValidRnd rnd] (x : ℝ) : ℝ :=
  let y :=
    if x > 0 then
      x + Real.log (1 + Real.exp (-x))
    else
      Real.log (1 + Real.exp x)
  neuralRound (β := β) (fexp := fexp) rnd y

end NNOps

end TorchLean.Floats
