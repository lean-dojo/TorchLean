import VersoManual
import NN.Proofs.Probability.DiffusionForward
import NN.Proofs.Gradients.Linear
import NN.Proofs.Gradients.Activation
import NN.Proofs.Autograd.FDeriv.Core
import NN.Tensor
import TorchLeanBlueprint.Bib
import TorchLeanBlueprint.Roles

open Verso.Genre Manual
open Verso.Genre.Manual.InlineLean

-- The declarations on this page live in four namespaces: the executable tensor specs in `Spec`, the
-- spec identities and calculus lemmas in `Proofs`, the measure-theoretic layer in
-- `NN.Proofs.Probability`, and the Fréchet-derivative layer in `Proofs.Autograd`. Opening all four
-- keeps the printed signatures close to what a caller would actually type, and keeps the displayed
-- code inside Verso's narrow column.
open TorchLean
open Spec
open Proofs
open MeasureTheory ProbabilityTheory
open NN.Proofs.Probability
open Proofs.Autograd
open scoped Autograd

-- A few signatures below are wider than this file's 100-column limit, so their `leanOutput` blocks
-- ask for `whitespace := lax` and are wrapped in the source. The rendered page still shows each
-- message exactly as Lean printed it.
open Lean.Elab.Tactic.GuardMsgs.WhitespaceMode (lax)

#doc (Manual) "Probability and Local Gradient Proofs" =>
%%%
tag := "probability-and-gradients"
%%%

Large-model proofs are rarely completed in one piece. A diffusion theorem needs a reusable statement
about affine images of Gaussian noise. An autograd theorem needs the derivative of each primitive.
A linear-layer proof needs to agree on whether weights are stored by input or output coordinate.
TorchLean keeps these local facts small enough to use independently.

Two examples show how these local results are used:

1. the forward noising kernel used by diffusion models;
2. scalar activation derivatives and linear-layer backward specifications.

The Gaussian result can be reused across diffusion schedules. The derivative results can be
composed across layers, provided each local differentiability condition holds.

# Vector-Jacobian Products

For a function with several outputs, a backward pass also needs an incoming cotangent specifying
how those outputs contribute to the surrounding calculation.

Let $`f:\mathbb R^n\to\mathbb R^m` be differentiable at $`x`, with derivative the linear map
$`Df(x)`. Forward mode evaluates $`Df(x)` on a direction: given $`v`, it returns
$`Df(x)\,v`, the *Jacobian-vector product*. Reverse mode evaluates the adjoint: given a cotangent
$`\delta\in\mathbb R^m`, it returns the unique vector $`\bar x` satisfying

$$`\langle Df(x)\,v,\ \delta\rangle
  =\langle v,\ \bar x\rangle
  \quad\text{for every }v,`

which in coordinates is $`\bar x=Df(x)^{\mathsf T}\delta`, the *vector-Jacobian product*. This is
the analytic derivative contract; algebraic tape theorems first prove adjointness of
the supplied JVP/VJP rules. The identity specifies how a change in the input affects the
cotangent-weighted output. For a scalar loss, $`m=1`; choosing
$`\delta=1` gives the gradient.

Two consequences shape everything below. First, a backward rule can be *stated* without mentioning
tapes, graphs, or memory: it is an identity between two vectors. Second, identifying that rule
with the adjoint of an actual derivative requires differentiability. A chosen linear rule can
have an adjoint at a kink even when it is not a derivative there. The analytic theorems below
therefore carry domain hypotheses for nonsmooth primitives.
The wider automatic-differentiation context, including why reverse mode is the right choice when
$`m\ll n`, is surveyed by Baydin and coauthors {Informal.citep baydin2018}[].

A concrete cotangent makes the transpose less mysterious. If a later scalar calculation is
$`L(y)=2y_0-y_1`, its incoming cotangent is $`(2,-1)`. The reverse rule must report how that
weighted combination changes when an input coordinate changes. Pairing with an arbitrary input
direction `v` tests all such changes at once. In a Euclidean space, equality of these pairings for
every `v` uniquely determines the returned vector. The tensor dot-product theorems use this
characterization to avoid constructing an entire Jacobian, which would store one entry for every
input/output coordinate pair.

# Forward Diffusion Kernels

Let `E` be a finite-dimensional real inner-product space and let `Z` have the standard Gaussian law
on `E`. Given scalars `a` and `b` and a clean state `x`, the forward noising step is

$$`X'=a x+bZ.`

In a DDPM schedule one usually takes

$$`a=\sqrt{\bar\alpha_t},
\qquad
b=\sqrt{1-\bar\alpha_t},`

so that

$$`x_t=\sqrt{\bar\alpha_t}\,x_0
      +\sqrt{1-\bar\alpha_t}\,\epsilon,`

which is the closed-form marginal of the forward chain in Ho, Jain, and Abbeel
{Informal.citep ddpm2020}[]. The schedule is not built into the probability theorem. The theorem
layer accepts arbitrary `a` and `b`; a diffusion model chooses them elsewhere, reusing the same
affine probability calculation across coefficient choices. DDIM
{Informal.citep ddim2021}[] has a distinct reverse-sampling construction; this forward Gaussian
theorem does not establish that sampler's correctness.

There is a closely related definition in the generative-model development:
`NN.MLTheory.Generative.Diffusion.forwardGaussian`. Both definitions scale a standard Gaussian and
translate it by a scaled clean state, but they live at different levels. `forwardNoising` works over
an arbitrary finite-dimensional real inner-product space and also supplies a Markov-kernel wrapper;
`forwardGaussian` is specialized to Euclidean space and records the Gaussian-closure theorem used by
the diffusion chapter. TorchLean does not currently contain a bridge theorem identifying the two
measures, so downstream proofs should use the declaration their hypotheses actually mention.

The definition
{src "NN/Proofs/Probability/DiffusionForward.lean"}[`forwardNoising`]
is a measure obtained by two pushforwards:

```lean (name := fwdNoising)
-- Inspect the structures needed to define a noising measure
-- on the state space.
#check @forwardNoising
```

```leanOutput fwdNoising (whitespace := lax)
@forwardNoising : {E : Type u_1} →
  [inst : NormedAddCommGroup E] →
    [inst_1 : InnerProductSpace ℝ E] → [FiniteDimensional ℝ E] →
      [inst : MeasurableSpace E] → ℝ → ℝ → E → Measure E
```

Its body applies the continuous linear scaling first and translates second:

```
-- Scale the standard Gaussian first, then translate it by
-- the clean signal.
def forwardNoising (a b : ℝ) (x : E) : Measure E :=
  ((stdGaussian E).map
      (b • (ContinuousLinearMap.id ℝ E))).map
    (fun y => a • x + y)
```

A pushforward describes a distribution by what happens to its samples: draw `z` from the original
measure, then apply the displayed map. Formally, the probability assigned to an output event is
the original measure of its preimage. The first `map` therefore changes the noise scale, and the
second moves its center. `Measure E` in the return type is more general than a probability measure;
the mass-one theorem below supplies the normalization fact for this construction. The definition
also admits `b = 0`, when the entire distribution is concentrated at `a • x`. Gaussian closure
here includes such degenerate distributions, so there is no nonzero-noise hypothesis to discharge.

The two stages match Mathlib's Gaussian-closure instances: a continuous linear image of a
Gaussian is Gaussian, and so is a translation. This lets `infer_instance` establish Gaussianity
from the definition. The following theorem combines the two stages into one affine pushforward:

```lean (name := fwdEqMap)
-- Replace the two pushforwards with the single affine noise
-- formula.
#check @forwardNoising_eq_map
```

```leanOutput fwdEqMap (whitespace := lax)
@forwardNoising_eq_map : ∀ {E : Type u_1}
  [inst : NormedAddCommGroup E] [inst_1 : InnerProductSpace ℝ E]
  [inst_2 : FiniteDimensional ℝ E] [inst_3 : MeasurableSpace E]
  [BorelSpace E] (a b : ℝ) (x : E),
  forwardNoising a b x =
    Measure.map (fun z => a • x + b • z) (stdGaussian E)
```

This is the one direct affine pushforward

$$`\operatorname{map}
  \bigl(z\mapsto ax+bz\bigr)
  \bigl(\mathcal N(0,I)\bigr),`

and it is tagged `@[simp]`, so downstream proofs almost never see the staged form. Because the
measure is Gaussian it is in particular a probability measure, which the file records explicitly:

```lean (name := fwdUniv)
-- Total mass one supplies normalization of the constructed
-- measure.
#check @forwardNoising_univ
```

```leanOutput fwdUniv (whitespace := lax)
@forwardNoising_univ : ∀ {E : Type u_1}
  [inst : NormedAddCommGroup E] [inst_1 : InnerProductSpace ℝ E]
  [inst_2 : FiniteDimensional ℝ E] [inst_3 : MeasurableSpace E]
  [BorelSpace E] (a b : ℝ) (x : E),
  (forwardNoising a b x) Set.univ = 1
```

A diffusion process needs composable transitions in addition to a measure for each starting point.
`forwardKernel a b : Kernel E E` packages the operation as a Markov kernel, built from
`Kernel.id ×ₖ Kernel.const E (stdGaussian E)` so that it fits Mathlib's kernel composition
machinery {Informal.citep mathlib2020}[]. The product kernel pairs the current state with fresh
Gaussian noise. The following theorem identifies its affine image with the measure already defined:

```lean (name := kernelApply)
-- Applying the kernel at a fixed state recovers the same
-- noising measure.
#check @forwardKernel_apply
```

```leanOutput kernelApply (whitespace := lax)
@forwardKernel_apply : ∀ {E : Type u_1}
  [inst : NormedAddCommGroup E] [inst_1 : InnerProductSpace ℝ E]
  [inst_2 : FiniteDimensional ℝ E] [inst_3 : MeasurableSpace E]
  [BorelSpace E] (a b : ℝ) (x : E),
  (forwardKernel a b) x = forwardNoising a b x
```

With that in hand, every fact proved about the measure transfers to the kernel by rewriting, which
is how `isGaussian_forwardKernel` is proved in two lines.

The distinction between a measure and a kernel is about dependence on the starting state. A
measure answers which noised outputs are possible from one fixed `x`. A kernel additionally
packages the measurability needed to vary `x` and integrate over a distribution of starting
states. Thus `forwardKernel_apply` is an equality of measures at each state, not a statement
about one sampled noise vector. A subsequent transition can use kernel composition, while a
moment calculation at a fixed clean input can rewrite directly to `forwardNoising`.

# Mean And Variance Of Forward Noising

`IsGaussian μ` says that `μ` belongs to the class of Gaussian measures. The definition of
`forwardNoising` fixes a particular distribution, but Gaussian closure
alone does not expose its mean or
covariance. A signal-to-noise argument needs the corresponding moment identities.

The first moment theorem identifies the mean:

```lean (name := meanThm)
-- The vector-valued integral identifies the mean of the
-- noisy state.
#check @integral_id_forwardNoising
```

```leanOutput meanThm (whitespace := lax)
@integral_id_forwardNoising : ∀ {E : Type u_1}
  [inst : NormedAddCommGroup E] [inst_1 : InnerProductSpace ℝ E]
  [inst_2 : FiniteDimensional ℝ E] [inst_3 : MeasurableSpace E]
  [BorelSpace E] (a b : ℝ) (x : E),
  ∫ (y : E), y ∂forwardNoising a b x = a • x
```

Rewriting with `forwardNoising_eq_map` turns the goal into an integral against `stdGaussian E`.
Splitting the
integral of the sum needs both halves integrable: the constant is integrable because the measure is
finite, and the identity is integrable under a Gaussian by Fernique's theorem, which Mathlib exposes
as `IsGaussian.integrable_id`. The noise term then vanishes because the standard Gaussian is
centred. The remaining constant term is the scaled clean state.

In the integral notation, `∂forwardNoising a b x` names the measure used for averaging; the
variable `y` is the noisy output being integrated. The result is a vector in `E`, as is `a • x`.
The symbol `•` denotes scalar multiplication, allowing the same theorem to apply to a scalar,
a coordinate vector, or another finite-dimensional real inner-product representation. The
integrability facts in the proof matter because the integral API is total: a syntactically valid
integral expression alone would not justify splitting a sum into two expectations.

The variance theorem describes spread along a continuous linear functional:

```lean (name := varThm)
-- Measure variance through an arbitrary continuous linear
-- readout.
#check @variance_dual_forwardNoising
```

```leanOutput varThm (whitespace := lax)
@variance_dual_forwardNoising : ∀ {E : Type u_1}
  [inst : NormedAddCommGroup E] [inst_1 : InnerProductSpace ℝ E]
  [inst_2 : FiniteDimensional ℝ E] [inst_3 : MeasurableSpace E]
  [BorelSpace E] (a b : ℝ) (x : E) (L : StrongDual ℝ E),
  Var[⇑L; forwardNoising a b x] = b ^ 2 * ‖L‖ ^ 2
```

`L : StrongDual ℝ E` is a continuous linear map from the state space to the reals.
Applying it to the noised state produces a constant plus `b` times a scalar Gaussian.
Translation leaves variance unchanged, and scaling multiplies it by the square of the scale.
Mathlib's `variance_dual_stdGaussian` supplies the variance of that scalar Gaussian.
In particular, instantiate `L` with the inner product against a unit vector to get variance
$`b^2` along that direction, with the right-hand side not depending on the direction at all. That
independence *is* the isotropy claim, stated without ever writing a matrix.

The right-hand side does not depend on `a` or the fixed clean state `x`: their contribution is
a translation. The noise scale `b` controls this conditional variance. If the clean state were
itself random, its variability would also contribute to an unconditional variance calculation.

For a non-unit direction, the norm factor records the scale of the measurement itself. Doubling
`L` doubles every measured deviation from its mean and multiplies the variance by four. Taking
`L = 0` gives zero variance. These cases explain why the theorem quantifies over all continuous
linear functionals instead of just coordinate projections. In a diffusion model, any fixed linear
readout of the noisy state can use the same theorem without first changing basis. The clean
signal enters its mean through `L (a • x)`, while its conditional noise variance is
$`b^2\|L\|^2`.

Both statements have kernel-level twins, `integral_id_forwardKernel` and
`variance_dual_forwardKernel`, proved by rewriting with `forwardKernel_apply` and applying the
measure-level theorem.

At `b = 0`, the affine noising map sends every Gaussian sample to `a • x`, so its pushforward
is concentrated at that one point. Its total mass is still one and every linear readout has
variance zero, exactly as these formulas predict. A zero noise scale is therefore a legitimate
specialization of the transition theorem. Nothing in the displayed type requires a positive
density or nonzero variance. At nonzero `b`, changing its sign preserves the variance because
the scale enters quadratically; the mean remains `a • x` in either case.

# Gaussian Noising Proofs

The following proofs are checked as the page builds. The section header
carries the same instance list the library file uses:

```lean
-- Rewrite the kernel to its measure before applying the
-- mass-one theorem.
section
variable {E : Type*}
  [NormedAddCommGroup E]
  [InnerProductSpace ℝ E]
  [FiniteDimensional ℝ E]
  [MeasurableSpace E]
  [BorelSpace E]

example (a b : ℝ) (x : E) :
    forwardKernel (E := E) a b x Set.univ = 1 := by
  rw [forwardKernel_apply]
  exact forwardNoising_univ a b x
```

Two moves: rewrite a kernel application as its noising measure, then use the probability-mass
theorem. Now the isotropy reading of the variance theorem, in the form a coordinate argument wants:

```lean
-- A unit measurement direction removes the norm factor from
-- the variance.
example (a b : ℝ) (x v : E) (hv : ‖v‖ = 1) :
    Var[innerSL ℝ v; forwardKernel (E := E) a b x]
      = b ^ 2 := by
  rw [variance_dual_forwardKernel]
  simp [hv]

end
```

The `simp` step is only discharging $`\|\operatorname{innerSL} v\|^2=\|v\|^2=1`, which Mathlib knows
because `innerSL` is a linear isometry onto its image. So the variance along any unit direction is
$`b^2`, and the choice of direction never appeared.

Now delete `[BorelSpace E]` from the variable block and ask for the mass-one fact again. The
measurable-space instance remains available, but Lean cannot connect it to the topology:

```lean +error (name := noBorel)
-- Keep the vector-space instances but deliberately omit the
-- Borel-space bridge.
section
variable {E : Type*}
  [NormedAddCommGroup E]
  [InnerProductSpace ℝ E]
  [FiniteDimensional ℝ E]
  [MeasurableSpace E]

example (a b : ℝ) (x : E) :
    forwardNoising (E := E) a b x Set.univ = 1 :=
  forwardNoising_univ a b x

end
```

```leanOutput noBorel (whitespace := lax)
failed to synthesize instance of type class
  BorelSpace E

Hint: Type class instance resolution failures can be inspected
with the `set_option trace.Meta.synthInstance true` command.
```

The error names a missing structure, not a failed numerical calculation. Lean knows how to add
and scale vectors from the algebraic instances, but the theorem's measurable pushforward also
uses the relation between open sets and measurable sets. The intentionally failing example keeps
all the algebra intact and removes only that relation. This is a useful way to read a typeclass
error: find the mathematical operation that needs the missing structure before trying to add
more imports or unfold definitions. Here the theorem's instance list already states the needed
assumption explicitly.

The Borel structure is what ties the measurable-space instance to the topology, and without it the
pushforward along a continuous map is not known to be measurable. Deleting
`[FiniteDimensional ℝ E]` instead removes `stdGaussian E`, since the standard Gaussian is built from
a finite orthonormal basis. These instances connect the topology, measurable structure, and
Gaussian construction used by the theorem.

This file does not sample noise, execute a diffusion network, or prove a reverse-time sampler.
Those are different objects. It proves the measure-theoretic forward transition that such
developments may cite.

# Gaussian Noising In PyTorch

The theorems above are exact statements about a measure. A sampling experiment gives a finite
statistical check of the same formula. Take
$`\bar\alpha_t=0.36`, so $`a=0.6` and $`b=0.8` in the real formula, and a clean state
$`x_0=(1,-2,0.5)`:

```
# Compare sampled coordinate moments with the conditional
# Gaussian formulas.
import torch, math
torch.manual_seed(0)
abar = 0.36
a, b = math.sqrt(abar), math.sqrt(1 - abar)
x0 = torch.tensor([1.0, -2.0, 0.5])
eps = torch.randn(2_000_000, 3)
xt = a * x0 + b * eps
print("empirical mean ", [round(v, 4) for v in xt.mean(0).tolist()])
print("a * x0         ", [round(v, 4) for v in (a * x0).tolist()])
print("empirical var  ", [round(v, 4) for v in xt.var(0).tolist()])
print(f"b^2 = {b*b:.2f}")
cov = torch.cov(xt.T)
off = cov - torch.diag(torch.diag(cov))
print(f"off-diagonal max |cov| = {off.abs().max().item():.5f}")
```

```
empirical mean  [0.5994, -1.1995, 0.3004]
a * x0          [0.6,    -1.2,     0.3]
empirical var   [0.6406,  0.6395,  0.6392]
b^2 = 0.64
off-diagonal max |cov| = 0.00081
```

Read the columns against the two theorems. The empirical mean approaches `a • x0`, which is
`integral_id_forwardNoising`. Each empirical variance approaches $`b^2=0.64` and the three agree
with each other, which is `variance_dual_forwardNoising` read along the three coordinate directions.
The largest off-diagonal covariance is $`8\times10^{-4}`, consistent with the same theorem read
along the diagonal directions $`(e_i\pm e_j)/\sqrt2`, whose equal variances imply zero
cross-covariance by expansion. Gaussianity then connects
uncorrelated coordinates to independence.

The displayed deviations include sampling error and floating arithmetic. A finite run cannot
establish the theorem's quantification over every continuous linear functional and every pair
`(a, b)`, or validate the random generator on its own. It can help detect a mismatch between the
intended model and the implementation. For example, predicting variance $`b` instead of $`b^2`
would predict `0.8`, inconsistent with these samples near `0.64`
{Informal.citep pytorch2019}[].

# The Linear Layer's Local Reverse Rule

TorchLean stores a linear layer's weight tensor with shape

$$`\texttt{[outDim, inDim]},`

matching PyTorch's `torch.nn.functional.linear` convention. A row of $`W` contains the weights
for one output coordinate; in row-major storage, the input index varies fastest. Using the same
layout simplifies checkpoint and gradient comparisons. Shape checking catches a transposed
rectangular matrix, but a square matrix still needs a check of its entries.

For one input vector,

$$`y_i=\sum_j W_{ij}x_j+b_i.`

If $`\delta_i` is the cotangent arriving from the rest of the computation, elementary
differentiation gives

$$`\frac{\partial L}{\partial x_j}
  =\sum_i W_{ij}\delta_i,\qquad
\frac{\partial L}{\partial W_{ij}}
  =\delta_i x_j,\qquad
\frac{\partial L}{\partial b_i}
  =\delta_i.`

In matrix notation:

$$`\bar x=W^\top\bar y,\qquad
\bar W=\bar y\,x^\top,\qquad
\bar b=\bar y.`

The file
{src "NN/Proofs/Gradients/Linear.lean"}[`NN.Proofs.Gradients.Linear`]
records these three equations using TorchLean tensors. Here is the weight rule:

```lean (name := wRule)
-- Weight cotangents have the same output-by-input
-- orientation as the weights.
#check @linearWeightsDerivSpec_eq_outerProductSpec
```

```leanOutput wRule (whitespace := lax)
@linearWeightsDerivSpec_eq_outerProductSpec :
  ∀ {inDim outDim : ℕ} (x : Tensor ℝ [inDim])
    (δ : Tensor ℝ [outDim]),
  linearWeightsDerivSpec x δ = outerProductSpec δ x
```

The input rule:

```lean (name := xRule)
-- Input cotangents sum contributions across the output rows
-- of the weights.
#check @linearInputDerivSpec_eq_vecMatMulSpec
```

```leanOutput xRule (whitespace := lax)
@linearInputDerivSpec_eq_vecMatMulSpec :
  ∀ {inDim outDim : ℕ} (layer : LinearSpec ℝ inDim outDim)
    (δ : Tensor ℝ [outDim]),
  linearInputDerivSpec layer.weights δ =
    vecMatMulSpec δ layer.weights
```

And the bias rule:

```lean (name := bRule)
-- Bias cotangents pass through unchanged; the other
-- arguments are unused.
#check @linearBiasDerivSpec_eq
```

```leanOutput bRule (whitespace := lax)
@linearBiasDerivSpec_eq : ∀ {inDim outDim : ℕ}
  (x : Tensor ℝ [inDim]) (δ : Tensor ℝ [outDim]),
  linearBiasDerivSpec Tensor.default δ x = δ
```

The bias rule has two arguments it never reads, a weight-gradient tensor and the input. They are
there so that all three rules have one call shape, which is what `linearBackwardSpec` relies on when
it produces the `LinearGradients` record. The theorem is stated at `Tensor.default` for the unused
slot to make the
point that the value cannot matter.

The tensor types also prevent transposing the outer product accidentally: $`\delta\otimes x` has
shape `[outDim, inDim]`, while $`x\otimes\delta` has shape `[inDim, outDim]`. For different input
and output dimensions, the swapped outer product has the wrong type.
The square example below still needs a value-level check.

The outer-product orientation also follows directly from a single weight perturbation. Changing
$`W_{ij}` by a small amount changes only output coordinate $`i`, with coefficient $`x_j`.
The incoming cotangent weights that output by $`\delta_i`, giving the entry
$`\delta_i x_j`. An input perturbation instead affects every output row, so the reverse rule
sums those weighted contributions over `i`. These are different index operations even though
both are matrix products. The three displayed signatures keep the input, weight, and bias
roles separate so later proofs can rewrite the appropriate part of the backward record.

# Linear-Layer Gradient Example

The specs are polymorphic over any scalar type with the right algebra, so the same definitions the
theorems are about can be executed at `Float`. Take

$$`W=\begin{pmatrix}1&2\\-1&3\end{pmatrix},
\quad x=\begin{pmatrix}4\\5\end{pmatrix},
\quad b=\delta=\begin{pmatrix}2\\-1\end{pmatrix}.`

```lean
-- Use a nonsymmetric square matrix so a mistaken transpose
-- changes the values.
def layerW : Tensor Float [2, 2] :=
  [[1, 2],
   [-1, 3]]

def layerX : Tensor Float [2] := [4, 5]
def layerB : Tensor Float [2] := [2, -1]
def cotangent : Tensor Float [2] := [2, -1]

-- The bias rule takes a weight-gradient argument it
-- never reads. This is what we pass for that slot,
-- to make the point that it cannot matter.
def unusedGrad : Tensor Float [2, 2] := Tensor.default
```

The forward pass first, so there is something for the cotangent to come back through:

```lean (name := fwdRun)
-- Evaluate the affine output before applying the chosen
-- incoming cotangent.
#eval Tensor.to
  (linearSpec { weights := layerW, bias := layerB } layerX)
  (Array Float)
```

```leanOutput fwdRun
#[16.000000, 10.000000]
```

By hand, $`1\cdot4+2\cdot5+2=16` and $`-1\cdot4+3\cdot5-1=10`. Now the three reverse rules:

```lean (name := revRun)
-- Display weight, input, and bias cotangents as flat arrays
-- for comparison.
#eval Tensor.to (linearWeightsDerivSpec layerX cotangent)
  (Array Float)
#eval Tensor.to (linearInputDerivSpec layerW cotangent)
  (Array Float)
#eval Tensor.to
  (linearBiasDerivSpec unusedGrad cotangent layerX)
  (Array Float)
```

```leanOutput revRun
#[8.000000, 10.000000, -4.000000, -5.000000]
```

```leanOutput revRun
#[3.000000, 1.000000]
```

```leanOutput revRun
#[2.000000, -1.000000]
```

The first array is $`\bar W=\delta x^\top` flattened row-major: $`(2\cdot4,\ 2\cdot5,\
-1\cdot4,\ -1\cdot5)`. The second is $`\bar x=W^\top\delta=(1\cdot2+(-1)(-1),\
2\cdot2+3\cdot(-1))=(3,1)`. The third is $`\delta` itself, unchanged, as the bias rule promises.

The products and sums here are small integers, exactly representable in binary32 and binary64.
This makes the run useful for checking index order: a discrepancy cannot be explained by rounding
in these operations.

# Linear-Layer Gradients In PyTorch

Same numbers, same layout, PyTorch's own autograd doing the differentiation:

```
# Backpropagate the same two-output cotangent through
# PyTorch’s linear layer.
W = torch.tensor([[1., 2.], [-1., 3.]], requires_grad=True)
b = torch.tensor([2., -1.], requires_grad=True)
x = torch.tensor([4., 5.], requires_grad=True)
y = torch.nn.functional.linear(x, W, b)
y.backward(torch.tensor([2., -1.]))
print("forward", y.tolist())
print("W.grad ", W.grad.tolist())
print("b.grad ", b.grad.tolist())
print("x.grad ", x.grad.tolist())
```

```
forward [16.0, 10.0]
W.grad  [[8.0, 10.0], [-4.0, -5.0]]
b.grad  [2.0, -1.0]
x.grad  [3.0, 1.0]
```

The entries agree, including the layout of `W.grad`. PyTorch reports the weight gradient in the
same `[outDim, inDim]` arrangement the
outer product produces, so a gradient computed on one side can be loaded on the other without a
transpose.

This example checks the two implementations' conventions on one input. A general equivalence
theorem would need to relate their definitions across all inputs, including their arithmetic.

# Batched Linear-Layer Gradients

The rules above are for one vector. Real training passes a batch, and the interesting part is that
the three gradients do not batch the same way: the input gradient gets one row per sample, while the
weight and bias gradients *accumulate* across samples. `linearDerivSpec` does this over any nonempty
leading shape by flattening the leading axes:

```lean (name := batchRun)
-- Accumulate shared parameter gradients while keeping one
-- input gradient per sample.
def batchX : Tensor Float [3, 2] :=
  [[4, 5], [1, 0], [0, 1]]

def batchD : Tensor Float [3, 2] :=
  [[2, -1], [1, 1], [0, 2]]

#eval
  let gradients :=
    linearDerivSpec (leading := [3]) (by decide)
      layerW batchX batchD
  (Tensor.to gradients.weightGradient (Array Float),
   Tensor.to gradients.biasGradient (Array Float),
   Tensor.to gradients.inputGradient (Array Float))
```

```leanOutput batchRun (whitespace := lax)
(#[9.000000, 10.000000, -3.000000, -3.000000],
 #[3.000000, 2.000000],
 #[3.000000, 1.000000, 0.000000, 5.000000, -2.000000, 6.000000])
```

Check the accumulation by hand. The three per-sample weight gradients are
$`\begin{pmatrix}8&10\\-4&-5\end{pmatrix}`, $`\begin{pmatrix}1&0\\1&0\end{pmatrix}`, and
$`\begin{pmatrix}0&0\\0&2\end{pmatrix}`; they sum to
$`\begin{pmatrix}9&10\\-3&-3\end{pmatrix}`, which is the first array. The bias gradient is
$`(2+1+0,\ -1+1+2)=(3,2)`. The input gradient keeps all three rows, $`(3,1)`, $`(0,5)`,
$`(-2,6)`, one $`W^\top\delta` per sample. PyTorch agrees on every entry:

```
# Clear the earlier parameter gradients before evaluating
# the batched example.
W.grad = None
b.grad = None
X = torch.tensor([[4., 5.], [1., 0.], [0., 1.]], requires_grad=True)
D = torch.tensor([[2., -1.], [1., 1.], [0., 2.]])
torch.nn.functional.linear(X, W, b).backward(D)
print("W.grad ", W.grad.tolist())
print("b.grad ", b.grad.tolist())
print("X.grad ", X.grad.tolist())
```

```
W.grad  [[9.0, 10.0], [-3.0, -3.0]]
b.grad  [3.0, 2.0]
X.grad  [[3.0, 1.0], [0.0, 5.0], [-2.0, 6.0]]
```

There is no division by the batch size in these formulas. The incoming tensor `batchD` already
specifies the derivative of the surrounding calculation with respect to each output. If that
calculation averages a loss across three samples, its cotangents include the factor $`1/3`, and
linearity then scales the accumulated parameter gradients accordingly. Adding an extra average
inside the layer would divide twice. The Python block resets `W.grad` and `b.grad` before the
batched call because those shared parameters were used in the preceding experiment; otherwise
PyTorch's accumulation would mix the two demonstrations.

`linearDerivSpec` currently has no correctness theorem. The unbatched identities above do not
establish that its flattening, accumulation, and output layout implement the batched derivative.
The proved batched
path in TorchLean goes through the tape-node layer instead, where matmul and reduction nodes carry
their own adjointness theorems and a batch is a shape rather than a special case; see
{src "NN/Proofs/Autograd/Coverage.lean"}[`NN.Proofs.Autograd.Coverage`] for what that layer covers.
If you want a theorem about the executable batched call above, it is open work.

# Definitional Linear-Layer Identities

All three linear identities are proved by `rfl`: `linearWeightsDerivSpec` is defined as the
outer product, so the theorem is a definitional unfolding that gives the equation a name and a
stable form for `simp`. Reading the coordinates makes the definitional character visible:

```lean
-- Reduce one weight-gradient entry to the corresponding
-- cotangent-input product.
example (x : Tensor ℝ [2]) (δ : Tensor ℝ [2])
    (i j : Fin 2) :
    get2 (linearWeightsDerivSpec x δ) i j =
      δ.getScalar i * x.getScalar j := by
  simp [linearWeightsDerivSpec]
```

Entry $`(i,j)` of the weight gradient is the product of two scalars, closed by `simp` from the
`get2_outerProductSpec` lemma alone.

These theorems give named equations for the backward-rule interface, useful for rewriting larger
expressions. Identifying those rules with the derivative of the forward tensor map requires a
separate `HasFDerivAt` argument. The equations alone also make no claim about a tape or CUDA kernel.

# Scalar Activation Calculus

Activation derivatives use a stronger style. The file
{src "NN/Proofs/Gradients/Activation.lean"}[`NN.Proofs.Gradients.Activation`]
proves Mathlib `HasDerivAt` statements for real scalar functions. For smooth sigmoid,

$$`\sigma(x)=\frac{1}{1+e^{-x}},
\qquad
\sigma'(x)=\sigma(x)\bigl(1-\sigma(x)\bigr),`

and the theorem is global, with no hypothesis at all:

```lean (name := sigThm)
-- The derivative proposition connects the sigmoid helper to
-- real calculus.
#check @sigmoid_deriv_correct
```

```leanOutput sigThm (whitespace := lax)
sigmoid_deriv_correct : ∀ (x : ℝ),
  HasDerivAt Activation.Math.sigmoidSpec
    (Activation.Math.sigmoidDerivSpec x) x
```

`HasDerivAt f d x` has three distinct arguments: the function being differentiated, its proposed
derivative value, and the point of evaluation. The proposition establishes the limiting
first-order behavior at that point, including differentiability. This is stronger than an equation
that merely unfolds the definition of a derivative helper. In the sigmoid signature,
`Activation.Math.sigmoidDerivSpec x` occupies the derivative-value slot, so the theorem connects
the executable formula's real interpretation to calculus. The universal `∀ x` means no input
point is omitted from this real-valued result.

The proof constructs derivatives of negation, exponential, addition, and inverse, then uses the
chain rule. The form $`\sigma(1-\sigma)` can reuse the forward sigmoid value without evaluating
$`e^{-x}/(1+e^{-x})^2` separately. The theorem's right-hand side uses this factored expression.

ReLU is different:

$$`\operatorname{ReLU}(x)=\max(x,0),\qquad
\operatorname{ReLU}'(x)=
\begin{cases}
0,&x<0,\\
1,&x>0.
\end{cases}`

ReLU has no ordinary derivative at zero. The following theorem supplies its derivative everywhere
else, with an explicit nonzero hypothesis:

```lean (name := reluThm)
-- The nonzero premise excludes the point where the ReLU
-- slopes disagree.
#check @relu_deriv_correct
```

```leanOutput reluThm (whitespace := lax)
relu_deriv_correct : ∀ (x : ℝ), x ≠ 0 →
  HasDerivAt Activation.Math.reluSpec
    (Activation.Math.reluDerivSpec x) x
```

The same kind of nonzero hypothesis appears for leaky ReLU and general-parameter ELU. Smooth
activations such as sigmoid, tanh, softplus, SiLU, and the chosen GELU formula have global
derivative theorems; guarded functions such as `safe_log` expose their domain parameter instead.

Two proofs, elaborated here. Away from the kink the ReLU theorem is usable and the derivative
computes to `1`:

```lean
-- Apply the local ReLU theorem at a positive point and
-- sigmoid’s theorem at zero.
example :
    HasDerivAt (Activation.Math.reluSpec : ℝ → ℝ) (1 : ℝ)
      (2 : ℝ) := by
  simpa [Activation.Math.reluDerivSpec] using
    relu_deriv_correct (2 : ℝ) (by norm_num)

example :
    HasDerivAt (Activation.Math.sigmoidSpec : ℝ → ℝ)
      (Activation.Math.sigmoidDerivSpec (0 : ℝ)) (0 : ℝ) :=
  sigmoid_deriv_correct 0
```

At the kink the same call cannot be made, and the error is exactly the hypothesis:

```lean +error (name := kinkFail)
-- At zero the ReLU theorem requires the impossible side
-- condition zero ≠ zero.
example :
    HasDerivAt (Activation.Math.reluSpec : ℝ → ℝ)
      (Activation.Math.reluDerivSpec (0 : ℝ)) (0 : ℝ) :=
  relu_deriv_correct (0 : ℝ) (by norm_num)
```

```leanOutput kinkFail
unsolved goals
⊢ False
```

`norm_num` reduced the side goal $`0\ne0` to `False` and then had nothing left to do. There is no
tactic that will close this, because the statement is not true.

# ReLU Backward Values At Zero

`reluDerivSpec` also returns a value at zero: its chosen backward value there is `0`.
That convention is outside the derivative theorem:

```lean (name := kinkRun)
-- Compare the chosen kink value with the smooth sigmoid
-- values at the origin.
#eval Activation.Math.reluDerivSpec (0.0 : Float)
#eval Activation.Math.sigmoidSpec (0.0 : Float)
#eval Activation.Math.sigmoidDerivSpec (0.0 : Float)
```

```leanOutput kinkRun
0.000000
```

```leanOutput kinkRun
0.500000
```

```leanOutput kinkRun
0.250000
```

PyTorch makes the same choice: `torch.relu` at `0.0` has gradient `0.0`, and
`torch.sigmoid(0.).backward()` gives `0.25`, matching the two sigmoid values above. So the
ReLU subgradient convention agrees. The forward value at signed zero has a separate behavior:

```lean (name := signedZero)
-- Taking a reciprocal exposes the sign of the returned
-- floating zero.
#eval Activation.Math.reluSpec (-0.0 : Float)
#eval (1.0 : Float) /
  Activation.Math.reluSpec (-0.0 : Float)
```

```leanOutput signedZero
0.000000
```

```leanOutput signedZero
inf
```

The second `#eval` is there because the first one is not conclusive: Lean prints negative zero as
`-0.000000`, but dividing is the way to be certain, and `inf` rather than `-inf` proves the result
is $`+0`. PyTorch returns $`-0.0` for `torch.relu(-0.)`, because it implements ReLU as a clamp that
passes the input through, while TorchLean's spec explicitly returns scalar zero when `x == 0`. That
equality
test also succeeds for `-0.0`, so the selected branch returns `+0.0`. Away from zero it uses
`max x 0`; the real identity `reluSpec_eq_max` recovers the usual maximum formula. Both are IEEE 754
legal; signed zero is a real part of the format, not an
accident {Informal.citep goldberg1991}[].

Both implementations select a zero gradient at the kink, but the sign of the forward zero can
affect a downstream division or another sign-sensitive operation. The `ℝ`-level
theorem cannot distinguish those signs, since $`\mathbb R` has one zero.

A runtime autograd system must specify its backward value at the kink. TorchLean chooses the
subgradient `0`, making the backward rule total. This convention does not supply an ordinary
`HasDerivAt` theorem at zero.

Other primitives have domain conditions too. A theorem using the usual positive domain of `log`
needs positivity;
one for reciprocal or division needs a nonzero denominator; square-root differentiation needs a
strictly positive point. Epsilon-protected `safe_log` and safe-division operations are different
functions, not licenses to drop those hypotheses from the raw functions. When composing a graph,
the local domain facts must be established at the actual saved forward values used by the reverse
rule.

# MLP Derivative Composition

For a network

$$`x\longmapsto W_2\,\sigma(W_1x+b_1)+b_2,`

a mathematical differentiation proof uses the affine derivative in each linear layer, the scalar
derivative of $`\sigma` lifted pointwise, the chain rule, and tensor-shape and adjoint bookkeeping.
TorchLean has all four, and the place to watch is what happens to the ReLU hypothesis when the
scalar theorem is lifted to a vector:

```lean (name := reluVecThm)
-- Pointwise differentiability requires every input
-- coordinate to avoid zero.
#check @hasFDerivAt_reluVec
```

```leanOutput reluVecThm (whitespace := lax)
@hasFDerivAt_reluVec : ∀ {n : ℕ} (x : Vec n),
  (∀ (i : Fin n), x.ofLp i ≠ 0) →
    HasFDerivAt reluVec (reluDerivCLM x) x
```

The scalar nonzero hypothesis becomes a condition on every coordinate. An $`n`-dimensional ReLU
map is nondifferentiable on the union of the $`n` coordinate hyperplanes; the displayed theorem
proves differentiability away from that union.

Composing the affine and activation derivatives gives a theorem about the full MLP backward rule:

```lean (name := mlpThm)
-- The MLP theorem tests the hidden pre-activations and
-- handles any cotangent.
#check @mlp_backward_eq_adjoint_fderiv
```

```leanOutput mlpThm (whitespace := lax)
@mlp_backward_eq_adjoint_fderiv : ∀ {inDim hidDim outDim : ℕ}
  (l1 : LinearSpec ℝ inDim hidDim)
  (l2 : LinearSpec ℝ hidDim outDim) (x : Tensor ℝ [inDim]),
  (∀ (i : Fin hidDim),
      have W1 := tensorToMatrix l1.weights;
      have b1 := getScalarE l1.bias;
      (affine W1 b1 (getScalarE x)).ofLp i ≠ 0) →
    ∀ (δ : Tensor ℝ [outDim]),
      getScalarE ((mlpOp l1 l2).backward x δ) =
        VJP[mlpVec l1 l2, getScalarE x] (getScalarE δ)
```

Read the conclusion against the definition from the opening section. On the left is what the
spec-level backward rule returns for a cotangent `δ`. On the right, `VJP[f, x]` is scoped notation
for the adjoint of the Fréchet derivative of `f` at `x`, so the right-hand side is the true analytic
vector-Jacobian product. The hypothesis is the lifted kink condition, now stated at the
pre-activation $`W_1x+b_1` rather than at the input, because that is where ReLU is actually
applied. This is the proof-layer analogue of the claim that `loss.backward()` computes the correct
VJP for a composed model {Informal.citep pytorch2019}[].

The `have W1 := ...; have b1 := ...;` expressions inside the printed hypothesis are local
abbreviations. They do not ask the caller to prove new propositions named `W1` or `b1`; they give
names to the vectorized weights and bias before stating that every pre-activation is nonzero.
Similarly, `getScalarE` translates the tensor result and cotangent into the Euclidean
representation used by `VJP`. Once the hypothesis holds, the final `∀ δ` covers every incoming
cotangent for this same forward input. Choosing different cotangents therefore needs no new
differentiability proof, while changing the input may require checking the pre-activations again.

Beyond the analytic layer, a runtime-autograd proof also has to show that the tape records the graph
correctly, that saved tensors and cotangent accumulation are right, and that the selected provider
implementation agrees with the operator semantics. The first two are proved in TorchLean for the
tape model: `backwardDenseAll_lowerGraphToTape_eq_backpropAllCtx` shows the sweep the eager trainer
actually executes equals the proved backpropagation on any lowered proof-carrying graph, and over
`ℝ` its companion `backwardDenseAll_lowerGraphToTape_adjoint_fderiv` equates that executed sweep
with the adjoint of the Fréchet derivative. The algebraic equality is generic over a commutative
semiring; the Fréchet-adjoint theorem
is specifically real-valued and carries differentiability hypotheses. The third obligation,
agreement of the `Float` and CUDA executions
with the operator semantics, remains an approximation and engineering concern rather than a theorem.

The probability theorem has an analogous place in diffusion. It supplies the exact forward
transition law with its two moments, while a model proof must still connect that law to a schedule,
a denoising objective, and an executable sampler.

# Probability And Autograd Proof Gaps

Collecting the gaps named above, in the order a reader would hit them:

* a bridge theorem identifying `forwardNoising` with
  `NN.MLTheory.Generative.Diffusion.forwardGaussian`, so the two developments can cite each other;
* the reverse-time transition and any statement about a denoising objective, which this file does
  not touch;
* a correctness theorem for `linearDerivSpec`, the batched call executed above;
* Fréchet-derivative statements for the tensor-typed linear map itself, as opposed to its
  vectorization through `getScalarE`;
* the runtime-layout lowering theorems that connect proved SSA graphs to the executable model
  wrappers, which {src "NN/Proofs/Autograd/Coverage.lean"}[`Coverage.lean`] tracks in detail;
* a numerical bridge for the particular executed `Float` or CUDA path. Existing rounded-model
  approximation theorems and runtime tests provide separate pieces, not automatic coverage.

The Gaussian construction is grounded in Mathlib's multivariate Gaussian and kernel libraries
{Informal.citep mathlib2020}[]. The reverse-mode formulas are standard matrix calculus. TorchLean's
declarations make these formulas available to later graph and runtime proofs, with explicit
assumptions on measurability, differentiability, and the point of evaluation.
