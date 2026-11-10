import VersoManual
import TorchLeanExamples.Transcendentals
import TorchLeanBlueprint.Bib
import TorchLeanBlueprint.Roles
import TorchLean.NN.LoRA
import TorchLean.NN.Autograd.Complex
import TorchLean.NN.Autograd.Differential
import TorchLean.NN.Autograd.Frontend
import TorchLean.NN.CLI.Training.Command
import TorchLean.NN.Data.Image
import TorchLean.NN.Data.Synthetic
import TorchLean.NN.Data.CausalLM
import TorchLean.NN.Models.CNN
import TorchLean.NN.Models.CausalTransformer.LanguageModel
import TorchLean.NN.Models.Diffusion.NoisePredictor
import TorchLean.NN.Models.FNO
import TorchLean.NN.Models.KAN
import TorchLean.NN.Models.Mamba
import TorchLean.NN.Models.PPO
import TorchLean.NN.Models.Recurrent
import TorchLean.NN.Models.ResNet
import TorchLean.NN.Models.MaskedPatchReconstructor
import TorchLean.NN.Models.UNet
import TorchLean.NN.Builder.MoE
import TorchLean.NN.Sequential.Training
import TorchLean.NN.RL.CLI
import TorchLean.NN.SelfSupervised
import TorchLean.NN.Text.GPT2BPE
import TorchLean.NN.Text.Generation
import TorchLean.NN.Text.Vocabulary
import TorchLean.NN.Trainer.Constructor
import TorchLean.NN.Trainer.Train
import TorchLean.NN.Op.Pointwise
import TorchLean.NN.Loss
import TorchLean.NN.Session.Random
import TorchLean.Verify.Trainer

open Verso.Genre Manual
open Verso.Genre.Manual.InlineLean
open TorchLean
open Lean.Elab.Tactic.GuardMsgs.WhitespaceMode (lax)
open FloatLib.Numerics (MathFunctions)

#doc (Manual) "Differentiable Scientific Models" =>
%%%
tag := "scientific-forward-models"
%%%

An exponential attenuation model gives us a gradient we can derive independently of autograd.
That makes it a useful starting point for an inverse problem: we can check the update direction
before asking an optimizer to fit observations. I'll start by varying one input, then let both
factors in the exponent be unknown to see why correct derivatives alone cannot identify them.

We evaluate the gradients, Jacobians, Hessians, and float counterexamples in named Lean blocks
during the build. Closed-form calculations and separate PyTorch snippets provide comparison points.

# Vegetation Attenuation Model

Consider a simplified vegetation-attenuation law: an exponential term represents transmission, and
an additive offset represents a surface contribution. This illustrative model fixes its
coefficients so that each derivative can be worked out by hand:

$$`f(x)=\tfrac12 e^{-2x}+\tfrac1{10}.`

Here $`x` is the attenuation variable. For nonnegative $`x`, increasing it reduces the exponential
contribution toward the offset $`\tfrac1{10}`. Both the input and the output are rank-zero tensors,
representing individual scalars:

```lean (name := sfmModel)
-- Keep the attenuation scale, exponential, and surface
-- offset explicit.
def sfmAttenuation : autograd.Function [] [] :=
  fun x => do
    let u ← Ops.apply .scale
        (x, .data (Ops.dataConst
          (Tensor.scalar (-2))), ())
    let e ← Ops.apply .exp (u, ())
    nn.functional.affine e (1 / 2) (1 / 10)
```

The three lines correspond one to one with the three stages of the formula:

$$`
u=-2x,\qquad
v=e^u,\qquad
f=\tfrac12 v+\tfrac1{10}.
`

By the chain rule, differentiating from the outside in,

$$`
f'(x)
=\tfrac12 e^{-2x}\cdot(-2)
=-e^{-2x},
`

so $`f'(0.5)=-e^{-1}\approx-0.367879`. The negative sign agrees with the attenuation interpretation.
The inner scaling contributes $`-2`, which cancels the outer factor of $`\tfrac12` in magnitude.
Autograd must propagate both factors through the three operations.

`autograd.grad` differentiates the program. Passing `value := true` returns the gradient and the
forward value from a single evaluation (the same convenience PyTorch spells
`torch.func.grad_and_value`):

```lean (name := sfmGrad)
-- Return the primal and its input gradient from the same
-- transform invocation.
#eval do
  let x : Tensor Float [] := Tensor.full [] 0.5
  let (g, y) ←
    autograd.grad (σ := []) (α := Float)
      sfmAttenuation x (value := true)
  IO.println s!"f(0.5)  = {y.item}"
  IO.println s!"f'(0.5) = {g.item}"
  IO.println s!"-e^-1   = {-Float.exp (-1.0)}"
```

```leanOutput sfmGrad (whitespace := lax)
f(0.5)  = 0.283940
f'(0.5) = -0.367879
-e^-1   = -0.367879
```

The last line is the closed form evaluated independently (in host Float, with no autograd
involved). It agrees to every printed digit.

Second derivatives come from the same program. For this model,

$$`
f''(x)=2e^{-2x},\qquad
f''(0.5)=2e^{-1}\approx0.735759,
`

and the forward-mode Jacobian should reproduce the first derivative we already have:

```lean (name := sfmSecond)
-- Compare forward differentiation and curvature with
-- independently evaluated formulas.
#eval do
  let x : Tensor Float [] := Tensor.full [] 0.5
  let j ←
    autograd.jacfwd (σ := []) (α := Float)
      sfmAttenuation x
  IO.println s!"jacfwd  = {j}"
  let h ←
    autograd.hessian (σ := []) (α := Float)
      sfmAttenuation x
  IO.println s!"hessian = {h}"
  IO.println s!"2e^-1   = {2.0 * Float.exp (-1.0)}"
```

```leanOutput sfmSecond (whitespace := lax)
jacfwd  = -0.367879
hessian = 0.735759
2e^-1   = 0.735759
```

At this input, forward and reverse mode agree at the printed precision, and the Hessian matches
$`2e^{-2x}`. For a scalar input and output, either Jacobian mode needs one directional pass (though
their overhead differs). More generally, forward mode takes one pass per input direction and
reverse mode one per output direction. A PDE residual may need derivatives with respect to only a
few coordinates, while a scalar training loss may depend on millions of parameters. Those
dimensions guide the choice of mode, as surveyed by {Informal.citet baydin2018}[].

In PyTorch the corresponding program uses float64, and its recorded transcript gives the same
displayed values:

```
# Use float64 to compare the same scalar formula and
# derivative queries.
import torch
from torch.func import grad_and_value, jacfwd, hessian

def f(x):
    return 0.5 * torch.exp(-2.0 * x) + 0.1

x = torch.tensor(0.5, dtype=torch.float64)
g, y = grad_and_value(f)(x)
print("f(0.5)  =", f"{y.item():.6f}")
print("f'(0.5) =", f"{g.item():.6f}")
print("jacfwd  =", f"{jacfwd(f)(x).item():.6f}")
print("hessian =", f"{hessian(f)(x).item():.6f}")
```

```
f(0.5)  = 0.283940
f'(0.5) = -0.367879
jacfwd  = -0.367879
hessian = 0.735759
```

What differs is how the programs expose their structure: `f` above is a Python function traced by
the differentiation transform, while `sfmAttenuation` has a type that says it maps a rank-zero
tensor to a rank-zero tensor and works for every supported scalar type.

Each of the three attenuation outputs describes a different aspect of the same local behavior. At
`0.5`, the prediction `0.283940` sits above the limiting offset `0.1`, and its negative slope says
that increasing attenuation reduces the prediction. Its positive second derivative says that this
negative slope becomes less steep as attenuation increases. That doesn't mean the prediction is
increasing. Read together, the forward value, slope, and curvature express a decreasing curve that
flattens toward its offset.

Here the exponential's argument is dimensionless. In a physical model, the units or normalization
of the attenuation variable and its coefficient must make that true (the scalar tensor type doesn't
track units). With its fixed coefficients, the example is a simplified forward law and not a
complete retrieval model. Connecting it to observations would also require the measurement model,
parameter ranges, and noise assumptions that the derivative calculation leaves open.

# Scalar Constants In Differentiable Programs

Decimal literal notation requires an instance that `autograd.Function` does not assume:

```lean +error (name := sfmLiteral)
-- A scalar-polymorphic function cannot assume
-- decimal-literal support.
example : autograd.Function [] [] :=
  fun x => TorchLean.Ops.apply .scale
      (x, .data (TorchLean.Ops.dataConst
        (Tensor.scalar 0.8)), ())
```

```leanOutput sfmLiteral (whitespace := lax)
failed to synthesize instance of type class
  OfScientific α✝

Hint: Type class instance resolution failures can be inspected
with the `set_option trace.Meta.synthInstance true` command.
```

The missing instance follows from the scalar-polymorphic type:

```lean (name := sfmFunctionType)
-- Inspect the scalar and operation interfaces available
-- inside the function body.
#print TorchLean.autograd.Function
```

```leanOutput sfmFunctionType (whitespace := lax)
@[reducible] def TorchLean.autograd.Function : Shape → Shape → Type 1 :=
fun σ τ =>
  {α : Type} →
    [inst : Storage α] →
      [inst_1 : Context α] →
        {m : Type → Type} →
          [Monad m] →
            [inst_3 : Ops m α] →
              Ops.Ref m α σ →
              m (Ops.Ref m α τ)
```

Inside the body, `α` is an abstract scalar type constrained only by `Storage` and `Context`.
There's no `OfScientific α`, because `0.8` isn't meaningful for every scalar a tensor can hold.
Integer constants and arithmetic expressions come from the standard `Zero`, `One`, `NatCast`,
`Div`, and `Neg` interfaces, and `Context.defaultEpsilon` supplies a backend-selected numerical
safeguard:

```lean (name := sfmNumbers)
-- These values instantiate the generic constants and
-- safeguard at host Float.
#eval ( ((1 / 2) : Float)
      , ((1 / 10) : Float)
      , (Context.defaultEpsilon : Float) )
```

```leanOutput sfmNumbers (whitespace := lax)
(0.500000, 0.100000, 0.000001)
```

The missing literal instance doesn't prohibit other fixed coefficients. Use ordinary integer
constants and scalar operations (four fifths, for example, can be written as `(2 * 2) / 5`). That
expression's value follows the chosen scalar arithmetic, so the real instance is exact while a
floating instance rounds. A learned coefficient whose gradient you need should be an input or
parameter instead.

You can instantiate the same `sfmAttenuation` at host `Float`, at `Float32`, or at `ℝ` for a
proof-facing interpretation. Sharing source keeps the formula aligned, but numerical refinement
needs separate evidence.

Host `Float` interprets those expressions and the default epsilon as
`(0.500000, 0.100000, 0.000001)`, and that tuple isn't a declaration that every backend represents
one tenth exactly or uses the same epsilon. In a scientific objective, changing epsilon changes a
regularized function, so retaining its selected value is part of describing the computation being
differentiated.

# Functional Primitives: `scale`, `shift`, And `affine`

The attenuation model uses the same `nn.functional` operations as a neural network's forward pass,
and we can compose them and differentiate the result with the same transforms.

`scale x c` is $`c\,x`, `shift x k` is $`x+k`, and `affine x c k` is $`c\,x+k`. Keeping `affine` as
a named helper makes the formula readable. Its implementation composes `scale` and `shift`, so it
isn't a fused primitive or a guarantee of one memory pass. Here's a short program that uses several
primitives at once:

```lean (name := sfmMixed)
-- The mean contributes a factor of one third to each input
-- derivative.
def sfmMixed : autograd.Function [3] [] :=
  fun xs => do
    let sq ← nn.functional.square xs
    let up ← nn.functional.shift sq 1
    let sc ← Ops.apply .scale
        (up, .data (Ops.dataConst
          (Tensor.scalar (1 / 2))), ())
    nn.functional.mean sc
```

Written out, the composite is $`g(x)=\frac1{3}\sum_i \tfrac12(x_i^2+1)`, so
$`\partial g/\partial x_i=\tfrac13 x_i`:

```lean (name := sfmMixedGrad)
-- The three coordinates should have gradients xᵢ / 3.
#eval do
  let xs : Tensor Float [3] := [1.0, 2.0, 3.0]
  let g ←
    autograd.grad (σ := [3]) (α := Float) sfmMixed xs
  IO.println s!"grad = {g}"
```

```leanOutput sfmMixedGrad (whitespace := lax)
grad = [0.333333, 0.666667, 1.000000]
```

`detach` preserves the forward value while cutting its derivative contribution. In
$`h(x)=x\cdot\operatorname{detach}(x)` the forward value is $`x^2`, but the second factor is a
constant as far as differentiation is concerned, so the transform returns
$`\operatorname{detach}(x)`, which is $`3` at $`x=3` instead of $`6`. That's a stop-gradient rule,
not the classical derivative of the forward function $`x^2`:

```lean (name := sfmDetach)
-- Only the first factor contributes a derivative path
-- through this product.
def sfmDetached : autograd.Function [] [] :=
  fun x => do
    let c ← Ops.apply .detach (x, ())
    nn.functional.mulB (s₁ := []) (s₂ := [])
      (t := []) x c

#eval do
  let x : Tensor Float [] := Tensor.full [] 3.0
  let g ←
    autograd.grad (σ := []) (α := Float)
      sfmDetached x
  IO.println s!"h(3) = 9, stop-gradient result = {g.item}"
```

```leanOutput sfmDetach (whitespace := lax)
h(3) = 9, stop-gradient result = 3.000000
```

Self-supervised objectives use this stop-gradient mechanism, and so do detached targets in
temporal-difference learning.

In the detach block, `h(3) = 9` is printed as literal explanatory text, while the gradient is what
the transform computes. Only one product-rule contribution remains, because the second factor is
detached.

# Inner Scaling And Derivatives

{src "TorchLeanExamples/Transcendentals.lean"}[`exponential`] computes
$`e^{\mathrm{scale}\,x}`. Changing `scale := -2` to `scale := -3` changes both the exponential's
argument and the factor contributed by the chain rule:

```lean (name := sfmThree)
-- Update the closed form when the inner coefficient
-- changes.
def sfmSteeper : autograd.Function [] [] :=
  Examples.Transcendentals.exponential
    (scale := -3)

#eval do
  let x : Tensor Float [] := Tensor.full [] 0.5
  let g ←
    autograd.grad (σ := []) (α := Float) sfmSteeper x
  IO.println s!"autograd       = {g.item}"
  IO.println s!"-3e^-1.5       = {-3.0 * Float.exp (-1.5)}"
  IO.println s!"stale -2e^-1   = {-2.0 * Float.exp (-1.0)}"
```

```leanOutput sfmThree (whitespace := lax)
autograd       = -0.669390
-3e^-1.5       = -0.669390
stale -2e^-1   = -0.735759
```

With the steeper scale, autograd returns $`-3e^{-1.5}`, matching the edited forward program. The
stale closed form ($`-2e^{-1}`) remains a distinct reference that autograd doesn't match.
Maintaining that calculation independently lets a test detect when the implementation and intended
formula diverge.

Changing the scale also illustrates why the size of a coefficient doesn't by itself predict the
size of a derivative. At `x = 0.5`, replacing `-2` with `-3` increases the chain-rule factor's
magnitude but decreases the exponential from `exp(-1)` to `exp(-1.5)`. Their product has magnitude
`0.669390`, smaller than the `0.735759` of the unedited control. Reading both effects keeps you
from taking a smaller gradient as evidence that the inner scale was ignored.

# Numerical Checks And Real-Valued Derivatives

Our derivative example above uses host `Float` at the input `0.5`, while over the reals the
derivative of `exp` is a theorem about the operator table's `exp` arm, at every shape:

```lean (name := sfmAdjoint)
-- The theorem covers every real input and cotangent of
-- every shape.
#check @TorchLean.Op.exp_backward_eq_adjoint
```

```leanOutput sfmAdjoint (whitespace := lax)
@Op.exp_backward_eq_adjoint : ∀ {s : Shape} (x δ : Tensor ℝ s),
  HasFDerivAt (fun y => ((Op.exp.rule?.get ⋯).forward (y, ())).1)
      (Op.slopeCLM fun c => Real.exp (Tensor.Rep.get x c)) x ∧
    ((Op.exp.rule?.get ⋯).backward (x, ()) (δ, ())).1 =
      (ContinuousLinearMap.adjoint
        (fderiv ℝ (fun y => ((Op.exp.rule?.get ⋯).forward (y, ())).1) x)) δ
```

In words, the theorem says two things: the map that multiplies each entry of a tangent by
`Real.exp` of the input entry is the Fréchet derivative of the arm's forward map at `x`, and the
arm's backward rule is the adjoint of that derivative for the Euclidean inner product of the
tensors. Here's its axiom audit:

```lean (name := sfmAxioms)
-- Inspect the theorem dependencies separately from
-- executable numerical behavior.
#print axioms TorchLean.Op.exp_backward_eq_adjoint
```

```leanOutput sfmAxioms (whitespace := lax)
'TorchLean.Op.exp_backward_eq_adjoint' depends on axioms: [propext,
 Classical.choice,
 Quot.sound]
```

The audit lists the three standard Lean axioms, with no `sorryAx` or TorchLean-specific axiom.

That theorem is about the real-valued proof-layer operation. Connecting it to host Float, CUDA, or
LibTorch requires the runtime refinement boundary (developed in {ref "runtime-approximation"}[the
runtime approximation chapter] and applied to the differentiation rules in
{ref "autograd-proofs"}[the autograd proofs chapter]).

The theorem uses shape `[1]` over `ℝ`, not the executable example's rank-zero `Float` interface.
Its cotangent `δ` is arbitrary, while `grad` uses only the unit seed.

# The Domain Of `log`

In the intended scientific reading, the logarithm is defined for positive arguments, and its
derivative

$$`\frac{d}{dx}\log x=\frac1x`

carries the same hypothesis. Floating-point hardware has no such hypothesis. It has NaN and
infinity:

```lean (name := sfmFloatLog)
-- Direct host arithmetic exposes exceptional values without
-- a domain guard.
#eval (Float.log (-1.0), Float.log 0.0)
```

```leanOutput sfmFloatLog (whitespace := lax)
(NaN, -inf)
```

Host floating-point operations and the differentiation transforms follow IEEE 754 on these inputs,
as PyTorch does, but total real-valued denotations handle them differently.

At the positive input $`2`, the transform returns the expected value and derivative:

```lean (name := sfmLogGood)
-- At a positive argument the value and the derivative
-- are those of the real logarithm.
def sfmLog : autograd.Function [] [] :=
  fun x => Ops.apply .log (x, ())

#eval do
  let x : Tensor Float [] := Tensor.full [] 2.0
  let (g, y) ←
    autograd.grad (σ := []) (α := Float) sfmLog x
      (value := true)
  IO.println s!"log 2 = {y.item}, grad = {g.item}"
```

```leanOutput sfmLogGood (whitespace := lax)
log 2 = 0.693147, grad = 0.500000
```

Outside the positive domain the transform raises no error, on any device: the forward value is
IEEE's NaN or $`-\infty`, and the pullback applies the registered $`1/x` formula:

```lean (name := sfmLogBad)
-- Value and gradient outside the real log domain: neither
-- raises an error.
#eval do
  for x0 in [-1.0, 0.0] do
    let x : Tensor Float [] := Tensor.full [] x0
    let (g, y) ←
      autograd.grad (σ := []) (α := Float) sfmLog x
        (value := true)
    IO.println s!"log {x0} = {y.item}, grad = {g.item}"
```

```leanOutput sfmLogBad (whitespace := lax)
log -1.000000 = NaN, grad = -1.000000
log 0.000000 = -inf, grad = inf
```

PyTorch applies the same registered $`1/x` backward formula on these inputs:

```
>>> # Inspect the registered backward formula at inputs
>>> # outside the real log domain.
>>> x = torch.tensor(-1.0, dtype=torch.float64, requires_grad=True)
>>> y = torch.log(x); torch.autograd.grad(y, x)
(tensor(-1., dtype=torch.float64),)
>>> x = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
>>> y = torch.log(x); torch.autograd.grad(y, x)
(tensor(inf, dtype=torch.float64),)
```

The value $`-1` is consistent with the registered formula, but it isn't a derivative of the real
logarithm on its positive domain. Updating parameters from it can hide an invalid forward loss (the
NaN value is the signal). The real-valued theorems about `log` keep the hypothesis $`x>0`, which
the runtime doesn't check.

A common attempted fix is to write $`\log(x+\epsilon)`. For arbitrary negative $`x` this doesn't
help, because adding a small positive $`\varepsilon` doesn't make a negative number positive. The
alternatives that actually work are:

- prove the precondition $`x>0` where the model guarantees it
- construct a positive quantity (for instance $`\operatorname{softplus}(x)+\varepsilon`) and
  document that you changed the model
- call a deliberately total operation and state its semantics.

TorchLean provides `safeLog` for models that need the third option. `safeLog` uses a positive real
argument when its epsilon is positive, and its specification is a definition, not a comment:

```lean (name := sfmSafeLogSpec)
-- Read the transformed argument: safeLog changes the model
-- even at positive inputs.
#print Activation.safeLog
```

```leanOutput sfmSafeLogSpec (whitespace := lax)
def TorchLean.Activation.safeLog : {α : Type} → [inst : Context α] → α →
  optParam α Context.defaultEpsilon → α :=
fun {α} [Context α] x ε =>
  MathFunctions.log (Activation.softplus x + ε)
```

So $`\operatorname{safelog}(x;\varepsilon)=\log(\operatorname{softplus}(x)+\varepsilon)`, whose
derivative is $`\operatorname{softplus}'(x)/(\operatorname{softplus}(x)+\varepsilon)`, and both are
defined for every real $`x` when $`\varepsilon>0`. An arbitrary `Context` or invalid epsilon
doesn't supply that mathematical guarantee. Here are the runtime operation and the specification
side by side, at the inputs where raw `log` returns NaN and $`-\infty`:

```lean (name := sfmSafeLog)
-- Compare the selected runtime rule with the
-- softplus-plus-epsilon specification.
def sfmSafeLog : autograd.Function [] [] :=
  fun x =>
    Ops.apply .safeLog
        (x, .data (Ops.dataConst
          (Tensor.scalar Context.defaultEpsilon)), ())

#eval do
  for x0 in [2.0, -1.0, 0.0] do
    let x : Tensor Float [] := Tensor.full [] x0
    let (g, y) ←
      autograd.grad (σ := []) (α := Float)
        sfmSafeLog x (value := true)
    let spec := Activation.safeLog x0
    IO.println s!"x = {x0}"
    IO.println s!"  runtime {y.item}, spec {spec}"
    IO.println s!"  grad {g.item}"
```

```leanOutput sfmSafeLog (whitespace := lax)
x = 2.000000
  runtime 0.754679, spec 0.754679
  grad 0.414117
x = -1.000000
  runtime -1.160713, spec -1.160713
  grad 0.858517
x = 0.000000
  runtime -0.366511, spec -0.366511
  grad 0.721346
```

The runtime and the specification agree to every printed digit, as does the PyTorch
transcription with the same $`\varepsilon=10^{-6}`:

```
# Match the transformed logarithm and epsilon used by the
# Lean example.
import torch, torch.nn.functional as F
eps = 1e-6
for v in [2.0, -1.0, 0.0]:
    x = torch.tensor(v, dtype=torch.float64, requires_grad=True)
    y = torch.log(F.softplus(x) + eps)
    g, = torch.autograd.grad(y, x)
    print(f"safelog({v}) = {y.item():.6f}  grad = {g.item():.6f}")
```

```
safelog(2.0) = 0.754679  grad = 0.414117
safelog(-1.0) = -1.160713  grad = 0.858517
safelog(0.0) = -0.366511  grad = 0.721346
```

The positive-input row also shows that `safeLog` changes the function even where raw logarithm is
defined: $`\operatorname{safelog}(2)=0.754679`, while $`\log 2=0.693147`. So its softplus
transformation and epsilon belong in the model definition and in any theorem about it.

One specification distinction is easy to miss here. Mathlib's real `log` and `sqrt` are total
functions: `Real.log 0 = 0`, `Real.log (-x) = Real.log x`, and `Real.sqrt x = 0` for `x ≤ 0`. The
rounded-real `NF` and `FP32` operations that {ref "runtime-approximation"}[the runtime
approximation chapter] builds inherit that totalization. They never produce NaN or infinity. So a
theorem proved about the rounded-real layer says nothing at all about the `NaN` printed above
unless it states the positivity hypothesis. Executable IEEE behavior lives in FloatLib binary32 and
in {ref "floats"}[the floats chapter], while real-valued theorems should carry the domain
hypotheses they mean.

An inverse-problem loop can test the objective with `Float.isFinite` and reject the proposed
parameter state instead of updating from an invalid objective.

But a finite objective is narrower than a useful one: a very large positive argument, upstream
overflow, or subsequent arithmetic may need additional checks. So the real statement that softplus
plus a positive epsilon is positive should be kept separate from finite-range and accuracy claims
about its floating-point implementation.

# Multiple Inputs

Scientific equations mix parameters with observations. Take the two-argument form of the same
decay,

$$`g(b,x)=e^{-bx},`

whose partial derivatives are $`\partial_b g=-x\,e^{-bx}` and $`\partial_x g=-b\,e^{-bx}`. Packing
both arguments into one input tensor means `indexAt` pulls out each coordinate along a given axis,
with the index bounded by the shape:

```lean (name := sfmPair)
-- Coordinate zero is b and coordinate one is x in this
-- input interface.
def sfmDecay : autograd.Function [2] [] :=
  fun input => do
    let b ← Ops.apply
        (.indexAt (s := [2]) ⟨0, by decide⟩ ⟨0, by decide⟩)
        (input, ())
    let x ← Ops.apply
        (.indexAt (s := [2]) ⟨0, by decide⟩ ⟨1, by decide⟩)
        (input, ())
    let bx ← nn.functional.mulB (s₁ := [])
      (s₂ := []) (t := []) b x
    let u ← Ops.apply .scale
        (bx, .data (Ops.dataConst
          (Tensor.scalar (-1))), ())
    Ops.apply .exp (u, ())

#eval do
  let v : Tensor Float [2] := [2.0, 0.5]
  let g ←
    autograd.grad (σ := [2]) (α := Float) sfmDecay v
  IO.println s!"grad (b, x) = {g}"
  let e := Float.exp (-1.0)
  IO.println s!"-x e^-bx    = {-0.5 * e}"
  IO.println s!"-b e^-bx    = {-2.0 * e}"
```

```leanOutput sfmPair (whitespace := lax)
grad (b, x) = [-0.183940, -0.735759]
-x e^-bx    = -0.183940
-b e^-bx    = -0.735759
```

PyTorch agrees, taking the gradient with respect to two leaf tensors instead of two coordinates:

```
>>> # Keep parameter and coordinate as separate leaves to
>>> # read both partial derivatives.
>>> b = torch.tensor(2.0, dtype=torch.float64, requires_grad=True)
>>> x = torch.tensor(0.5, dtype=torch.float64, requires_grad=True)
>>> torch.autograd.grad(torch.exp(-(b * x)), (b, x))
(tensor(-0.1839, dtype=torch.float64), tensor(-0.7358, dtype=torch.float64))
```

Both the `⟨0, by decide⟩` and `⟨1, by decide⟩` arguments have type `Fin` bounded by the axis size,
so the proof that the coordinate exists is discharged at elaboration time and an out-of-range read
isn't expressible. What the type doesn't fix is the meaning of the coordinates: shape `[2]` doesn't
say that position zero is the attenuation parameter and position one the observation. When the
arguments have different shapes or different physical meaning, a typed input pack is usually the
clearer choice (and an index selection (`Op.gather` along one axis, with indices from
`Tensor.selectIndices`) gathers several positions at once when a subset of features is wanted).
Either way, the feature order should be part of the input interface and its documentation.

Both partial derivatives are negative, but their meanings differ. The first measures a change in
the attenuation coefficient with the observation coordinate fixed, while the second changes the
coordinate with the coefficient fixed. At `[2, 0.5]` their magnitude ratio is four, because the
corresponding chain-rule factors are `0.5` and `2`. A gradient descent update would also need a
choice of parameter scaling and units before these two coordinates could be treated as comparable
step sizes. Their order is fixed by the packed tensor, while the model interpretation fixes which
coordinates are actually unknowns.

# Inverse Problems

Let $`x_i` denote known observation locations, $`y_i` the measured values, and $`\eta_i` the
measurement errors. A model with unknown parameters $`\theta` assumes

$$`y_i=f_\theta(x_i)+\eta_i,`

and estimates $`\theta` by least squares from $`N>0` observations:

$$`
L(\theta)=
\frac1N\sum_{i=1}^{N}
\left(f_\theta(x_i)-y_i\right)^2.
`

In that objective, the factor $`1/N` scales the gradient of every term. We can isolate the effect
of this reduction by averaging the attenuation values themselves (before introducing residuals or
unknown parameters), so the next block differentiates a mean output with respect to its three input
coordinates. Since $`\frac{\partial}{\partial x_i}\frac1N\sum_j f(x_j)=\frac1N f'(x_i)` and
$`f'(x)=-e^{-2x}`, each coordinate should get $`-e^{-2x_i}/3`:

```lean (name := sfmMean)
-- Isolate mean reduction before adding residual weights
-- from a fitting objective.
def sfmBatchMean : autograd.Function [3] [] :=
  fun xs => do
    let u ← Ops.apply .scale
        (xs, .data (Ops.dataConst
          (Tensor.scalar (-2))), ())
    let e ← Ops.apply .exp (u, ())
    let a ←
      nn.functional.affine e (1 / 2) (1 / 10)
    nn.functional.mean a

#eval do
  let xs : Tensor Float [3] := [0.0, 0.5, 1.0]
  let g ←
    autograd.grad (σ := [3]) (α := Float) sfmBatchMean xs
  IO.println s!"grad     = {g}"
  let third := 1.0 / 3.0
  let closed : Tensor Float [3] :=
    [-third, -third * Float.exp (-1.0),
      -third * Float.exp (-2.0)]
  IO.println s!"-e^-2x/3 = {closed}"
```

```leanOutput sfmMean (whitespace := lax)
grad     = [-0.333333, -0.122626, -0.045112]
-e^-2x/3 = [-0.333333, -0.122626, -0.045112]
```

The offset $`\tfrac1{10}` contributes nothing to any partial derivative, which is the arithmetic
statement that an additive bias in a forward model is invisible to the gradient with respect to
inputs. If that offset were the unknown, it would have to be a parameter.

To fit $`f_\theta`, the optimizer needs derivatives with respect to its parameter pack, as in
{ref "training-from-scratch"}[the training chapter]. The
{ref "backend-selection"}[backend planner chapter] explains how the selected runtime records its
implementation and evidence. Neither choice resolves an ambiguity in the model: if both $`b` and
$`x` are unknown and observations depend only on their product, different parameter pairs can fit
equally well.

Our mean block gives a mean-output derivative, not a least-squares parameter gradient. A
least-squares gradient also includes each residual and a factor of two, and residuals of different
signs can change its direction.

## Identifiability And A Flat Direction

Treat both inputs to `sfmDecay` as unknown parameters of a single observation. Its value depends
on them only through the product $`bx`, so every parameter pair on the hyperbola $`bx=1`
predicts the same value:

```lean (name := sfmFlat)
#eval do
  for (b, x) in
      [(2.0, 0.5), (1.0, 1.0), (4.0, 0.25)] do
    let v : Tensor Float [2] := [b, x]
    let (g, y) ←
      autograd.grad (σ := [2]) (α := Float) sfmDecay v
        (value := true)
    IO.println s!"b={b} x={x} g={y.item} grad={g}"
  -- The hyperbola through (2, 0.5) has tangent (b, -x)
  -- there, so this directional derivative must vanish.
  let v : Tensor Float [2] := [2.0, 0.5]
  let d ←
    autograd.grad (σ := [2]) (α := Float) sfmDecay v
  IO.println
    s!"grad . (2, -0.5) = {2.0 * d[0] - 0.5 * d[1]}"
```

```leanOutput sfmFlat (whitespace := lax)
b=2.000000 x=0.500000 g=0.367879 grad=[-0.183940, -0.735759]
b=1.000000 x=1.000000 g=0.367879 grad=[-0.367879, -0.367879]
b=4.000000 x=0.250000 g=0.367879 grad=[-0.091970, -1.471518]
grad . (2, -0.5) = 0.000000
```

For any fixed observed target, these three parameter vectors give the same residual and hence the
same least-squares loss. The last line is the local version of the same fact: the gradient is
$`-e^{-bx}(x,b)`, the tangent to the level curve at $`(b,x)` is $`(b,-x)`, and their inner product
is $`-e^{-bx}(xb-bx)`. That inner product is zero on paper, and in Float it's zero here as well,
because the two products cancel with matching finite power-of-two scalings in this example.

Additional information is needed. If a nonzero $`x` is known, one noiseless observation
$`y=e^{-bx}>0` already determines $`b=-\log(y)/x`. If both coordinates remain unknown, more
measurements must constrain them independently. In a larger model, the smallest eigenvalue of the
Gauss-Newton matrix $`J^\top J` can reveal directions to which the predictions are locally
insensitive (here $`J` is the Jacobian of the observations with respect to the unknown parameters).
For a single observation, that matrix is an outer product of one vector with itself, hence rank at
most one and singular for every $`(b,x)`. At $`(0,0)` its rank is zero. A floating-point spectrum
is evidence about a particular matrix, and a small eigenvalue alone doesn't prove
non-identifiability. When the model is simple enough to allow one, the proof is the algebraic
statement that the map $`(b,x)\mapsto f_{b,x}` is not injective, which the printed collision above
witnesses at one point.

Each row of the collision table has the same product `bx` and different individual coordinates.
Their gradients differ because the level curve has different tangents at those points, even though
the predicted observation remains `0.367879`. So a nonzero gradient doesn't rule out
non-identifiability: it can point across a level curve while giving no information along it.

The vanishing directional derivative is local. Moving a finite distance along the straight tangent
need not stay on the hyperbola, though moving along the hyperbola does. In this example the product
identity supplies the global ambiguity. A small numerical directional derivative in a larger model,
without such an identity, would supply only local evidence of weak sensitivity.

# PINN Residuals

For a PDE with differential operator $`\mathcal N`, written as

$$`\mathcal N[u]=0,`

a physics-informed neural network represents the solution by a neural field $`u_\theta(x,t)` and
minimizes the residual at sampled collocation points,

$$`
L_{\mathrm{PDE}}(\theta)
=
\frac1N\sum_i
\left|\mathcal N[u_\theta](x_i,t_i)\right|^2,
`

following {Informal.citet pinn2019}[]. Here $`N` is the number of collocation points. Computing
the objective and optimizing it require derivatives with respect to different variables:

- derivatives with respect to *input coordinates* build the PDE terms $`u_t`, $`u_x`, $`u_{xx}`
- derivatives with respect to *parameters* drive the optimizer.

The input-coordinate part is a function transform on a small map, which is what `jacfwd` and
`hessian` above compute. Here's a residual you can check by hand. The transport equation
$`u_t+2u_x=0` has the travelling-wave solution $`u(x,t)=e^{x-2t}`, so the real input derivatives
cancel exactly:

```lean (name := sfmPinn)
-- Read spatial and temporal coordinates in the same order
-- used by the residual.
def sfmField : autograd.Function [2] [] :=
  fun input => do
    let x ← Ops.apply
        (.indexAt (s := [2]) ⟨0, by decide⟩ ⟨0, by decide⟩)
        (input, ())
    let t ← Ops.apply
        (.indexAt (s := [2]) ⟨0, by decide⟩ ⟨1, by decide⟩)
        (input, ())
    let drift ← Ops.apply .scale
        (t, .data (Ops.dataConst
          (Tensor.scalar (-2))), ())
    let arg ← nn.functional.addB (s₁ := [])
      (s₂ := []) (t := []) x drift
    Ops.apply .exp (arg, ())

#eval do
  let p : Tensor Float [2] := [0.3, 0.1]
  let d ←
    autograd.jacfwd (σ := [2]) (α := Float) sfmField p
  IO.println s!"(u_x, u_t) = {d}"
```

```leanOutput sfmPinn (whitespace := lax)
(u_x, u_t) = [1.105171, -2.210342]
```

Here $`u_x=e^{0.1}=1.105171` and $`u_t=-2e^{0.1}=-2.210342`, so the residual $`u_t+2u_x` is zero,
and in this case exactly zero in Float as well because the two values differ by a power of two.
PyTorch's `torch.func.jacfwd` on the same function returns `[1.105171, -2.210342]` and a residual
of `0.000000`.

Zero residual is a statement about one equation, though, and it's easy to read it as a statement
about the field. That same $`u` isn't a solution of the diffusion equation $`u_t=u_{xx}`, and the
second input derivatives say by how much:

```lean (name := sfmHeat)
-- Use the spatial Hessian entry to test a different
-- differential equation.
#eval do
  let p : Tensor Float [2] := [0.3, 0.1]
  let h : Tensor Float [2, 2] ←
    autograd.hessian (σ := [2]) (α := Float) sfmField p
  let (d, y) ←
    autograd.grad (σ := [2]) (α := Float) sfmField p
      (value := true)
  let uxx := h[0][0]
  IO.println s!"u           = {y.item}"
  IO.println s!"hessian     = {h}"
  IO.println s!"u_t + 2 u_x = {d[1] + 2.0 * d[0]}"
  IO.println s!"u_t - u_xx  = {d[1] - uxx}"
```

```leanOutput sfmHeat (whitespace := lax)
u           = 1.105171
hessian     = [[1.105171, -2.210342],
 [-2.210342, 4.420684]]
u_t + 2 u_x = 0.000000
u_t - u_xx  = -3.315513
```

`hessian` returns a `Tensor Float [2, 2]` with rows $`(u_{xx},u_{xt})` and $`(u_{tx},u_{tt})`,
indexed by the two input coordinates. For this field those are $`u`, $`-2u`, $`-2u`, and $`4u`, the
powers of the drift coefficient appearing where differentiating $`e^{x-2t}` twice puts them. With a
zero transport residual and a diffusion residual of $`-3u`, the same field satisfies the transport
equation but fails the diffusion equation, so the residual must include the intended differential
operator. That Hessian costs one Hessian-vector product per input coordinate, which is why
input-coordinate derivatives stay cheap for a two-dimensional domain while parameter-space second
derivatives don't.

Its off-diagonal entries agree because both mixed derivatives of this smooth exponential equal
`-2u`. Selecting `[0][0]` extracts `u_xx`, and selecting another entry would define a different
residual even though all entries have the same scalar type. So the coordinate ordering established
when the input is unpacked must remain consistent when the differential operator is assembled.

A residual that vanishes at sampled points leaves the behavior between those points unestablished,
and a PDE solution also has to meet its initial and boundary conditions.
{ref "scientific-ml-verification"}[The scientific ML verification chapter] explains how a finite
residual artifact becomes a checked claim and what remains to be established over the domain.

There are no learned parameters in the field example. It checks the coordinate derivatives needed
to assemble a residual, using a closed-form field whose answers are known. Training a neural field
adds a separate derivative through that residual with respect to its parameter state. So the
displayed Hessian is a coordinate Hessian, not the Hessian of a training objective.

For that training step, `autograd.model.derivative` evaluates repeated input-directional
derivatives of a sequential model, and `autograd.model.derivativeVJP` pulls a residual cotangent
back to the model parameters. A direction tensor with rows `[dx, dx]` selects the second derivative
in direction `dx`, and rows `[dx, dt]` select a mixed derivative (each row has the input shape). We
implement them by nesting dual scalars around the existing evaluator and reverse pass, keeping the
directions and supplied cotangent constant.

Our runnable example trains a tanh MLP on $`u''=-2` over $`[-1,1]`, with zero values at both
boundaries, and it receives no solution labels:

```terminal
scripts/lake.sh exe torchlean pinn
```

In the PINN example, the loss is half the mean squared equation residual plus half the mean squared
boundary residual. For each equation term, the pullback seed is the residual divided by the
collocation count. We add the returned parameter gradients to the boundary gradients and pass them
to `nn.sgdStep`. In the recorded 1,500-step CPU run, this loss fell from `2.244093` to `0.006179`.
The exact solution $`1-x^2` appears only in the final comparison (it doesn't supply training data).
As before, these sampled residuals don't establish a uniform PDE guarantee.

# Floating-Point Transcendentals

Over the reals,

$$`e^{a+b}=e^ae^b.`

In floating point the two sides are different programs. Each `exp` call rounds, the product rounds
again, and the two roundings don't have to land on the same representable number:

```lean (name := sfmExpIdentity)
-- Scale the discrepancy for display without changing either
-- exponential program.
#eval
  ( (Float.exp (1.0 + 1.0)
      - Float.exp 1.0 * Float.exp 1.0) * 1e18
  , (Float.exp (0.7 + 1.3)
      - Float.exp 0.7 * Float.exp 1.3) * 1e18 )
```

```leanOutput sfmExpIdentity (whitespace := lax)
(888.178420, -888.178420)
```

Both differences are one unit in the last place of $`e^2\approx7.389` (scaled by $`10^{18}` so
they're visible in the output). Their opposite signs show that neither evaluation consistently
exceeds the other in these two examples. PyTorch reproduces the first one exactly
(`888.1784197001252` in float64), which is a useful parity observation, though it doesn't establish
that both dispatched to the same transcendental implementation. A theorem about the real identity
doesn't specify either floating-point evaluation order, so a proof obligation stated over $`ℝ`
needs a rounding or refinement step before it says anything about the numbers a run produced.

{Informal.citet goldberg1991}[] explains rounding and evaluation-order effects.
{Informal.citet flocq2011}[] separates formats from rounding operators in Coq, a
separation also used in TorchLean's float layer. {Informal.citet boldo2015}[] connects
rounded-arithmetic proofs to emitted code in CompCert. {ref "floating-point-literature"}[The
floating-point literature chapter] discusses how TorchLean's layer relates to each of them.

FloatLib's executable transcendental approximations are deterministic definitions, and
{ref "fp32-soundness"}[the FP32 soundness chapter] states what has been proved about them. What
hasn't been proved is a universal correctly-rounded contract. A passing derivative check at
$`\exp(-2x)` supports that input on that code path, but it doesn't establish accuracy for `exp`,
`log`, `sin`, or `cos` over all finite bit patterns.

# Parameterization, Reduction, And Arithmetic

Three variations isolate different parts of the model:

1. Replacing the $`-2` by a parameter and querying its gradient turns the forward model into the
   inverse problem above.
2. Widening `sfmBatchMean` from three observations to thirty reduces the contribution of each
   unchanged coordinate by the factor predicted by $`1/N`.
3. Evaluating the same finite input through FloatLib binary32 and host Float compares different
   arithmetic interpretations of the formula.

Change one choice at a time and update the independent derivative calculation with it, just as
the inner-scaling example changed both the exponential argument and its chain-rule factor.

Sources and further reading:

- {src "TorchLeanExamples/Transcendentals.lean"}[`Transcendentals.lean`], the scaled
  exponential used above
- {src "TorchLean/NN/Functional.lean"}[`TorchLean/NN/Functional.lean`], the
  `nn.functional` primitives
- {src "TorchLean/Tensor/Layers/Activation.lean"}[`Activation.lean`], the `softplus` and
  `safeLog` specifications
- {Informal.citet baydin2018}[] on automatic differentiation, and {Informal.citep pytorch2019}[]
  for the framework the comparisons above use
- {Informal.citet pinn2019}[] for the residual objective
- {Informal.citet goldberg1991}[], {Informal.citet flocq2011}[], and {Informal.citet boldo2015}[]
  on floating-point arithmetic and its formalizations.
