import VersoManual
import NN.API
import NN.Examples.Functional.Transcendentals
import TorchLeanBlueprint.Bib
import TorchLeanBlueprint.Roles

open Verso.Genre Manual
open Verso.Genre.Manual.InlineLean
open TorchLean
open Lean.Elab.Tactic.GuardMsgs.WhitespaceMode (lax)

#doc (Manual) "Differentiable Scientific Models" =>
%%%
tag := "scientific-forward-models"
%%%

An exponential attenuation model gives us a gradient we can derive independently of autograd.
That makes it a useful starting point for an inverse problem: we can check the update direction
before asking an optimizer to fit observations. I'll start by varying one input, then let both
factors in the exponent be unknown to see why correct derivatives alone cannot identify them.

The named Lean blocks evaluate the gradients, Jacobians, Hessians, and float counterexamples during
the build. Closed-form calculations and separate PyTorch snippets provide comparison points.

# Vegetation Attenuation Model

The model is the canopy term of a soil-moisture retrieval that
combines SMAP (Soil Moisture Active Passive) radiometry with NISAR (NASA-ISRO Synthetic Aperture
Radar) backscatter. Two-way transmittance through vegetation decays exponentially with
attenuation. We isolate that term and an additive surface offset, with fixed coefficients so
that each derivative can be calculated explicitly:

$$`f(x)=\tfrac12 e^{-2x}+\tfrac1{10}.`

Here $`x` is the attenuation variable. For nonnegative $`x`, increasing it reduces the
exponential contribution toward the offset $`\tfrac1{10}`. The rank-zero input and output tensors
represent individual scalars:

```lean (name := sfmModel)
-- Keep the attenuation scale, exponential, and surface
-- offset explicit.
def sfmAttenuation : autograd.Function [] [] :=
  fun x => do
    let u ← nn.functional.scale x (-2)
    let e ← nn.functional.exp u
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
forward value from a single evaluation, which is the same convenience PyTorch spells
`torch.func.grad_and_value`:

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

The last line is the closed form evaluated independently, in host Float, with no autograd involved.
It agrees to every printed digit.

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
$`2e^{-2x}`. For a scalar input and output, either Jacobian mode needs one directional pass,
though their overhead differs. More generally, forward mode takes one pass per input direction
and reverse mode one per output direction. A PDE residual may need derivatives with respect to
only a few coordinates; a scalar training loss may depend on millions of parameters. Those
dimensions guide the choice of mode, as surveyed by {Informal.citet baydin2018}[].

The PyTorch program is the same shape, and this is the transcript from the same machine
(float64; the displayed values also reproduce with PyTorch 2.11):

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

The four displayed values agree. The programs expose their structure differently: `f` above is a
Python function traced by the differentiation transform, while
`sfmAttenuation` has a type that says it maps a rank-zero tensor to a rank-zero tensor and works
for every supported scalar type.

The three attenuation outputs describe different aspects of the same local behavior. At `0.5`,
the prediction `0.283940` sits above the limiting offset `0.1`; its negative slope says that
increasing attenuation reduces the prediction. The positive second derivative says that this
negative slope becomes less steep as attenuation increases. It does not mean the prediction is
increasing. The forward value, slope, and curvature together express a decreasing curve that
flattens toward its offset.

Here the exponential's argument is dimensionless. In a physical model, the units or normalization
of the attenuation variable and its coefficient must make that true. The scalar tensor type does
not track units, and the example's fixed coefficients are a simplified forward law rather than a
complete retrieval model. Connecting it to observations would also require the measurement model,
parameter ranges, and noise assumptions that the derivative calculation leaves open.

# Scalar Constants In Differentiable Programs

Decimal literal notation requires an instance that `autograd.Function` does not assume:

```lean +error (name := sfmLiteral)
-- A scalar-polymorphic function cannot assume
-- decimal-literal support.
example : autograd.Function [] [] :=
  fun x => nn.functional.scale x 0.8
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
          [inst_2 : Monad m] →
            [inst_3 : Runtime.Autograd.Torch.Ops m α] →
              Runtime.ValueRef m α σ →
              m (Runtime.ValueRef m α τ)
```

Inside the body, `α` is an abstract scalar type constrained only by `Storage` and `Context`. There
is no `OfScientific α`, because `0.8` is not meaningful for every scalar a tensor can hold. The
standard `Zero`, `One`, `NatCast`, `Div`, and `Neg` interfaces provide integer constants and
arithmetic expressions. `Context.defaultEpsilon` supplies a backend-selected numerical safeguard:

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

The missing literal instance does not prohibit other fixed coefficients. Use ordinary integer
constants
and scalar operations: for example, four fifths can be written as
`(2 * 2) / 5`. Its value follows the chosen scalar arithmetic,
so the real instance is exact while a floating instance rounds. A learned coefficient should instead
be an input or parameter if its gradient is needed.

The same `sfmAttenuation` can be instantiated at host `Float`, at `Float32`, or at `ℝ` for a
proof-facing interpretation. Sharing source keeps the formula aligned; numerical refinement still
requires separate evidence.

The tuple `(0.500000, 0.100000, 0.000001)` is the host `Float` interpretation of those expressions
and the default epsilon. It is not a declaration that every backend represents one tenth exactly
or uses the same epsilon. In a scientific objective, changing epsilon changes a regularized
function, so retaining its selected value is part of describing the computation being
differentiated.

# Functional Primitives: `scale`, `shift`, And `affine`

The attenuation model uses the same `nn.functional` operations as a neural network's forward pass.
We can compose them and differentiate the result with the same transforms.

`scale x c` is $`c\,x`, `shift x k` is $`x+k`, and `affine x c k` is $`c\,x+k`. Keeping `affine` as
a named helper makes the formula readable. Its current implementation composes `scale` and `shift`;
it is not a fused primitive or a guarantee of one memory pass. A short program using several
primitives at once:

```lean (name := sfmMixed)
-- The mean contributes a factor of one third to each input
-- derivative.
def sfmMixed : autograd.Function [3] [] :=
  fun xs => do
    let sq ← nn.functional.square xs
    let up ← nn.functional.shift sq 1
    let sc ← nn.functional.scale up (1 / 2)
    nn.functional.mean sc
```

The composite is $`g(x)=\frac1{3}\sum_i \tfrac12(x_i^2+1)`, so
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

`detach` preserves the forward value while cutting its derivative contribution.
In $`h(x)=x\cdot\operatorname{detach}(x)` the forward value is
$`x^2`, but the second factor is a constant as far as differentiation is concerned, so
the transform returns $`\operatorname{detach}(x)`, which is $`3` at $`x=3` rather than $`6`.
This is a stop-gradient rule, not the classical derivative of the forward function $`x^2`:

```lean (name := sfmDetach)
-- Only the first factor contributes a derivative path
-- through this product.
def sfmDetached : autograd.Function [] [] :=
  fun x => do
    let c ← nn.functional.detach x
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

This mechanism is used in self-supervised objectives and detached targets in temporal-difference
learning.

In the mixed example, squaring contributes `2xᵢ`, the outer scale contributes `1/2`, and the mean
contributes `1/3`. Their product gives `xᵢ/3`, explaining the increasing coordinates of the printed
gradient. The added constant contributes to the forward mean but disappears from the input
derivative. A check that omitted the mean could return `[1, 2, 3]` with the right signs and still
be wrong by a factor of three.

In the detach block, `h(3) = 9` is printed as literal explanatory text; the transform computes the
gradient. Only one product-rule contribution remains because the second factor is detached.

# Positive And Negative Derivative Controls

{src "NN/Examples/Functional/Transcendentals.lean"}[`Transcendentals.lean`] checks the exponential
rule, an affine slope, and the negative factor in a composed exponential. The command
`lake exe torchlean transcendentals` runs these controls; the named block also runs them during
the guide build:

```lean (name := sfmControls)
-- Require both agreement with the formula and rejection of
-- the named wrong answers.
open NN.Examples.Functional.Transcendentals in
#eval checkAll
```

```leanOutput sfmControls (whitespace := lax)
[PASS] exp: grad = 1.648721 ≈ 1.648721
[PASS-NEG] exp≠1: grad = 1.648721 ≠ 1.000000 (test discriminates)
[PASS] affine(3x+1): grad = 3.000000 ≈ 3.000000
[PASS-NEG] affine≠1: grad = 3.000000 ≠ 1.000000 (test discriminates)
[PASS] exp(-2x): grad = -0.735759 ≈ -0.735759
[PASS-NEG] exp(-2x) sign: grad = -0.735759 ≠ 0.735759 (test discriminates)
[transcendentals] all positive + negative controls passed ✓
```

The three positive controls check $`\frac{d}{dx}e^x=e^x` at $`x=0.5`, a slope of three for
$`3x+1`, and $`\frac{d}{dx}e^{-2x}=-2e^{-2x}` at $`x=0.5`.

Each positive control is paired with a plausible wrong answer that must *not* match. A suite that
only checked $`\frac{d}{dx}e^x` at a point where
the answer happens to be close to one would also pass if autograd returned the constant one, and a
suite that compared $`|f'|` would pass with the sign flipped. The negative controls demonstrate
that each test can distinguish the defect it is intended to catch.

`PASS` uses an absolute tolerance of `1e-6` in the maintained control runner. `PASS-NEG` checks
that the named incorrect value lies outside that same tolerance. Neither label asserts bitwise
equality. For the exponential case, choosing `0.5` keeps the correct derivative far enough from
`1` for the test to distinguish a constant-one rule. For the composed exponential, the sign check
would catch a lost negative scaling even if the magnitude were correct. These controls are small
because their purpose is to localize those mistakes, not estimate accuracy across a domain.

# Inner Scaling And Derivatives

In the $`e^{-2x}` control, change the inner scaling from $`-2` to $`-3`. This changes both the
exponential's argument and the factor contributed by the chain rule:

```lean (name := sfmThree)
-- Update the closed form when the inner coefficient
-- changes.
def sfmSteeper : autograd.Function [] [] :=
  fun x => do
    let u ← nn.functional.scale x (-3)
    nn.functional.exp u

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

Autograd now returns $`-3e^{-1.5}`, matching the new forward program. The old closed form,
$`-2e^{-1}`, remains a distinct reference and no longer agrees. Maintaining that calculation
independently lets a test detect when the implementation and intended formula diverge.

The changed scaling also illustrates why the size of a coefficient does not alone predict the
size of a derivative. At `x = 0.5`, replacing `-2` with `-3` increases the chain-rule factor's
magnitude but decreases the exponential from `exp(-1)` to `exp(-1.5)`. Their product has magnitude
`0.669390`, smaller than the old `0.735759`. Reading both effects avoids interpreting a smaller
gradient as evidence that the inner scale was ignored.

# Numerical Checks And Real-Valued Derivatives

The controls use host `Float` on three formulas at the input `0.5`. The same file also gives a
proof-layer object for real-valued `exp`:

```lean (name := sfmProofSurface)
-- This bundle connects a real operation to its Fréchet
-- derivative.
open NN.Examples.Functional.Transcendentals in
#check @expProofSurface
```

```leanOutput sfmProofSurface (whitespace := lax)
expProofSurface : Proofs.Autograd.OpSpecFDerivCorrect 1 1
```

`OpSpecFDerivCorrect` packages the real-valued forward operation, its JVP candidate, its backward
candidate, and a proof that the JVP really is the Fréchet derivative. From that package TorchLean
derives the statement that the backward rule is the adjoint of the derivative:

```lean (name := sfmAdjoint)
-- The theorem covers every real input and cotangent of the
-- stated shape.
open NN.Examples.Functional.Transcendentals in
#check @expBackward_eq_adjoint_fderiv
```

```leanOutput sfmAdjoint (whitespace := lax)
expBackward_eq_adjoint_fderiv : ∀ (x δ : Tensor ℝ [1]),
  Proofs.Autograd.getScalarE (expProofSurface.correct.op.backward x δ) =
    (Proofs.Autograd.vjp expProofSurface.forwardVec
      (Proofs.Autograd.getScalarE x)) (Proofs.Autograd.getScalarE δ)
```

This quantifies over all real tensors `x` and cotangents `δ` of the stated shape. Its axiom
audit is:

```lean (name := sfmAxioms)
-- Inspect the theorem dependencies separately from
-- executable numerical behavior.
open NN.Examples.Functional.Transcendentals in
#print axioms expBackward_eq_adjoint_fderiv
```

```leanOutput sfmAxioms (whitespace := lax)
'NN.Examples.Functional.Transcendentals.expBackward_eq_adjoint_fderiv' depends on axioms: [propext,
 Classical.choice,
 Quot.sound]
```

The audit lists the three standard Lean axioms, with no `sorryAx` or TorchLean-specific axiom.

The theorem concerns the real-valued proof-layer operation. Connecting it to host Float, CUDA, or
LibTorch requires the runtime refinement boundary, which {ref "runtime-approximation"}[the runtime
approximation chapter] develops and {ref "autograd-proofs"}[the autograd proofs chapter] applies to
the differentiation rules.

The proof-surface type uses one-dimensional real tensors, whereas the executable attenuation
example used rank-zero `Float` tensors. Both represent one scalar quantity, but they are distinct
interfaces and the theorem explicitly names the former. Its cotangent `δ` is arbitrary: the
backward rule agrees with the derivative adjoint for every seed of the stated shape, not just the
unit seed used by `grad`.

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

Host floating-point operations, checked differentiation transforms, and total real-valued
denotations handle these inputs differently.

At the positive input $`2`, the checked transform returns the expected value and derivative:

```lean (name := sfmLogGood)
-- A positive argument satisfies the checked raw-log
-- precondition.
def sfmLog : autograd.Function [] [] :=
  fun x => nn.functional.log x

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

Public `IO` differentiation transforms check the raw-log domain before evaluating the forward
node, including gradient-only calls. The failure is catchable:

```lean (name := sfmLogBad)
-- Catch domain rejection before treating an invalid loss as
-- differentiable data.
#eval do
  for x0 in [-1.0, 0.0] do
    let x : Tensor Float [] := Tensor.full [] x0
    try
      let g ←
        autograd.grad (σ := []) (α := Float) sfmLog x
      IO.println s!"grad at {x0} = {g.item}"
    catch e =>
      IO.println s!"at {x0}: {e}"
```

```leanOutput sfmLogBad (whitespace := lax)
at -1.000000: autograd: log: input contains values <= 0 (or NaN);
use `safe_log` if you want epsilon protection
at 0.000000: autograd: log: input contains values <= 0 (or NaN);
use `safe_log` if you want epsilon protection
```

PyTorch instead applies the registered $`1/x` backward formula even on these invalid forward inputs:

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

The value $`-1` is consistent with the registered formula, but is not a derivative of the real
logarithm on its positive domain. Updating parameters from it can hide an invalid forward loss.
TorchLean's eager tape and checked public transforms reject these inputs. Pure internal graph
operations and mathematical real denotations retain their explicit domain assumptions; their
existence does not make every direct internal evaluation a checked runtime entrypoint.

A common attempted fix is to write $`\log(x+\epsilon)`. For arbitrary negative $`x` this does not
help, because adding a small positive $`\varepsilon` does not make a negative number positive. The
alternatives that actually work are:

- prove the precondition $`x>0` where the model guarantees it;
- construct a positive quantity, for instance $`\operatorname{softplus}(x)+\varepsilon`, and
  document that you changed the model;
- call a deliberately total operation and state its semantics.

TorchLean provides `safeLog` for models that need the third option.
`safeLog` uses a positive real argument when its epsilon is positive, and its specification is a
definition rather than a comment:

```lean (name := sfmSafeLogSpec)
-- Read the transformed argument; safeLog changes the model
-- even at positive inputs.
#print Activation.Math.safeLogSpec
```

```leanOutput sfmSafeLogSpec (whitespace := lax)
def Activation.Math.safeLogSpec : {α : Type} → [inst : Context α] → α →
  optParam α Context.defaultEpsilon → α :=
fun {α} [Context α] x ε =>
  MathFunctions.log (Activation.Math.softplusSpec x + ε)
```

So $`\operatorname{safelog}(x;\varepsilon)=\log(\operatorname{softplus}(x)+\varepsilon)`, whose
derivative is $`\operatorname{softplus}'(x)/(\operatorname{softplus}(x)+\varepsilon)`, and both are
defined for every real $`x` when $`\varepsilon>0`. An arbitrary `Context` or invalid epsilon
does not supply that mathematical guarantee. Running the runtime operation and the specification
side by side at
the input that broke raw `log`:

```lean (name := sfmSafeLog)
-- Compare the selected runtime rule with the
-- softplus-plus-epsilon specification.
def sfmSafeLog : autograd.Function [] [] :=
  fun x =>
    Runtime.Autograd.Torch.safeLog x
      (ε := Context.defaultEpsilon)

#eval do
  for x0 in [2.0, -1.0, 0.0] do
    let x : Tensor Float [] := Tensor.full [] x0
    let (g, y) ←
      autograd.grad (σ := []) (α := Float)
        sfmSafeLog x (value := true)
    let spec := Activation.Math.safeLogSpec x0
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

The positive-input row also shows that `safeLog` changes the function even where raw logarithm
is defined: $`\operatorname{safelog}(2)=0.754679`, while $`\log 2=0.693147`. Its softplus
transformation and epsilon therefore belong in the model definition and any theorem about it.

One specification distinction is easy to miss here. Mathlib's real `log` and `sqrt` are total
functions: `Real.log 0 = 0`, `Real.log (-x) = Real.log x`, and `Real.sqrt x = 0` for `x ≤ 0`.
The rounded-real `NF` and `FP32`
operations that {ref "runtime-approximation"}[the runtime approximation chapter] builds inherit
that totalization. They never produce NaN or infinity. So a theorem proved about the rounded-real
layer says nothing at all about the `NaN` printed above unless the positivity hypothesis is stated
explicitly. Executable IEEE behavior lives in FloatLib binary32 and in {ref "floats"}[the floats
chapter]; real-valued theorems should carry the domain hypotheses they mean.

At input `2`, raw log gives slope `0.5`; safe log gives `0.414117`. That difference is expected
from the softplus derivative in the numerator and the transformed argument in the denominator.
At `-1` and `0`, the finite safe-log outputs describe the extended model, while the raw-log
transform rejects the request before reporting a gradient. A catchable domain error gives an
inverse-problem loop a chance to reject a proposed parameter state instead of updating from an
invalid objective.

The error also mentions NaN because an unordered value cannot satisfy the positive-input guard.
Positive-domain validation is still narrower than a guarantee of a useful finite loss: a very
large positive argument, upstream overflow, or subsequent arithmetic may need additional checks.
The real statement that softplus plus a positive epsilon is positive should therefore be kept
separate from finite-range and accuracy claims about its floating-point implementation.

# Multiple Inputs

Scientific equations mix parameters with observations. Take the two-argument form of the same
decay,

$$`g(b,x)=e^{-bx},`

whose partial derivatives are $`\partial_b g=-x\,e^{-bx}` and $`\partial_x g=-b\,e^{-bx}`. Packing
both arguments into one input tensor means `select` pulls out each coordinate along an explicit
axis, with the index bounded by the shape:

```lean (name := sfmPair)
-- Coordinate zero is b and coordinate one is x in this
-- input interface.
def sfmDecay : autograd.Function [2] [] :=
  fun input => do
    let b ← Runtime.Autograd.Torch.select
      (s := [2]) 0 input ⟨0, by decide⟩
    let x ← Runtime.Autograd.Torch.select
      (s := [2]) 0 input ⟨1, by decide⟩
    let bx ← nn.functional.mulB (s₁ := [])
      (s₂ := []) (t := []) b x
    let u ← nn.functional.scale bx (-1)
    nn.functional.exp u

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

The `⟨0, by decide⟩` and `⟨1, by decide⟩` arguments have type `Fin`
bounded by the axis size, so the proof that the coordinate exists is discharged at elaboration
time and an out-of-range read is not expressible. What the type does not fix is the meaning of the
coordinates: shape `[2]` does not say that position zero is the attenuation parameter and position
one the observation. When the arguments have different shapes or different physical
meaning, a typed input pack is usually the clearer choice, and `indexSelect` gathers several
positions at once when a subset of features is wanted. Either way the feature order should be part
of the input interface and its documentation.

The two partial derivatives have different meanings even though both are negative. The first
measures a change in the attenuation coefficient with the observation coordinate fixed; the
second changes the coordinate with the coefficient fixed. At `[2, 0.5]`, their magnitude ratio is
four because the corresponding chain-rule factors are `0.5` and `2`. A gradient descent update
would also need a choice of parameter scaling and units before these two coordinates could be
treated as comparable step sizes. The packed tensor fixes their order, while the model
interpretation fixes which coordinates are actually unknowns.

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

The factor $`1/N` scales the gradient of every term in the objective. We can isolate the effect
of this reduction by averaging the attenuation values themselves, before introducing residuals or
unknown parameters. The next block therefore differentiates a mean output with respect to its
three input coordinates. Since
$`\frac{\partial}{\partial x_i}\frac1N\sum_j f(x_j)=\frac1N f'(x_i)` and
$`f'(x)=-e^{-2x}`, each coordinate should get $`-e^{-2x_i}/3`:

```lean (name := sfmMean)
-- Isolate mean reduction before adding residual weights
-- from a fitting objective.
def sfmBatchMean : autograd.Function [3] [] :=
  fun xs => do
    let u ← nn.functional.scale xs (-2)
    let e ← nn.functional.exp u
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

The mean-output example isolates reduction; it does not yet differentiate a least-squares fit.
For the displayed inputs, the exponential sensitivity decreases from `1` to `exp(-2)`, and the
mean scales each contribution by one third. A least-squares parameter gradient would additionally
multiply each prediction derivative by its residual and by `2`, then sum over observations.
Those residuals can have different signs, so the gradient of a fit need not resemble the three
negative entries printed here. Separating these two programs prevents a correct reduction check
from being mistaken for an end-to-end inverse-problem validation.

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
same least-squares loss. The
last line is the local version of the same fact: the gradient is $`-e^{-bx}(x,b)`, the tangent to
the level curve at $`(b,x)` is $`(b,-x)`, and their inner product is $`-e^{-bx}(xb-bx)`, which is
zero on paper and zero in Float here because the two products cancel before rounding can separate
them.

Additional information is needed. If a nonzero $`x` is known, one noiseless observation
$`y=e^{-bx}>0` already determines $`b=-\log(y)/x`. If both coordinates remain unknown, more
measurements must constrain them independently. In a larger model, the smallest eigenvalue of
the Gauss-Newton matrix $`J^\top J` can reveal directions to which the predictions are locally
insensitive. Here $`J` is the Jacobian of the observations with respect to the unknown parameters.
For a single observation,
that matrix is an outer product of one vector with itself, hence rank at most one and singular for
every
$`(b,x)`; at $`(0,0)` its rank is zero. A floating-point spectrum is evidence about a particular
matrix, and a small eigenvalue alone does not prove non-identifiability. The
proof, when the model is simple enough to allow one, is the algebraic statement that the map
$`(b,x)\mapsto f_{b,x}` is not injective, which is exactly what the printed collision above
witnesses at one point.

The collision table holds the product `bx` fixed while changing the individual coordinates.
Its gradients differ because the level curve has different tangents at those points, even though
the predicted observation remains `0.367879`. A nonzero gradient therefore does not rule out
non-identifiability: it can point across a level curve while giving no information along it.

The vanishing directional derivative is local. Moving a finite distance along the straight tangent
need not stay on the hyperbola; moving along the hyperbola does. The product identity supplies the
global ambiguity in this example. A small numerical directional derivative in a larger model,
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

- derivatives with respect to *input coordinates* build the PDE terms $`u_t`, $`u_x`, $`u_{xx}`;
- derivatives with respect to *parameters* drive the optimizer.

The input-coordinate part is a function transform on a small map, exactly what `jacfwd` and
`hessian` above compute. Here is a residual that can be checked by hand. The transport equation
$`u_t+2u_x=0` has the travelling-wave solution $`u(x,t)=e^{x-2t}`, so a correct pair of input
derivatives must cancel exactly:

```lean (name := sfmPinn)
-- Read spatial and temporal coordinates in the same order
-- used by the residual.
def sfmField : autograd.Function [2] [] :=
  fun input => do
    let x ← Runtime.Autograd.Torch.select
      (s := [2]) 0 input ⟨0, by decide⟩
    let t ← Runtime.Autograd.Torch.select
      (s := [2]) 0 input ⟨1, by decide⟩
    let drift ← nn.functional.scale t (-2)
    let arg ← nn.functional.addB (s₁ := [])
      (s₂ := []) (t := []) x drift
    nn.functional.exp arg

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

Zero residual is a statement about one equation, though, and it is easy to read it as a statement
about the field. The same $`u` is not a solution of the diffusion equation $`u_t=u_{xx}`, and the
second input derivatives say by how much:

```lean (name := sfmHeat)
-- Use the spatial Hessian entry to test a different
-- differential equation.
#eval do
  let p : Tensor Float [2] := [0.3, 0.1]
  let h ←
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

`hessian` returns a `Tensor Float [2, 2]` indexed by the two input coordinates. Its rows hold
$`(u_{xx},u_{xt})` and $`(u_{tx},u_{tt})`.
For this field those are $`u`, $`-2u`, $`-2u`, and $`4u`, the powers of the drift coefficient
appearing exactly where differentiating $`e^{x-2t}` twice puts them. The transport residual is zero
and the diffusion residual is $`-3u`. Thus the same field satisfies the transport equation but
fails the diffusion equation; the residual must include the intended differential operator.
The cost of that Hessian is one Hessian-vector product per input coordinate, which is
why input-coordinate derivatives stay cheap for a two-dimensional domain while parameter-space
second derivatives do not.

A residual that vanishes at sampled points leaves the behavior between those points unestablished;
a PDE solution also has to meet its initial and boundary conditions.
{ref "scientific-ml-verification"}[The scientific ML verification
chapter] explains how a finite residual artifact becomes a checked claim and what remains to be
established over the domain.

The field example contains no learned parameters. It checks the coordinate derivatives needed to
assemble a residual, using a closed-form field whose answers are known. Training a neural field
adds a separate derivative through that residual with respect to its parameter state. The displayed
Hessian is consequently a coordinate Hessian, not the Hessian of a training objective.

Its off-diagonal entries agree because both mixed derivatives of this smooth exponential equal
`-2u`. Selecting `[0][0]` extracts `u_xx`; selecting another entry would define a different residual
even though all entries have the same scalar type. The coordinate ordering established when the
input is unpacked must therefore remain consistent when the differential operator is assembled.

# Floating-Point Transcendentals

Over the reals,

$$`e^{a+b}=e^ae^b.`

In floating point the two sides are different programs. Each `exp` call rounds, the product rounds
again, and the two roundings do not have to land on the same representable number:

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

Both differences are one unit in the last place of $`e^2\approx7.389`, scaled by $`10^{18}` so they
are visible in the output. Their opposite signs show that neither evaluation consistently exceeds
the other in these two examples. PyTorch reproduces the first one exactly (`888.1784197001252`
in float64), which is
a useful parity observation, though it does not establish that both dispatched to the same
transcendental implementation. A theorem about the real identity does not specify either
floating-point evaluation order, so a
proof obligation stated over $`ℝ` needs an explicit rounding or refinement step before it says
anything about the numbers a run produced.

{Informal.citet goldberg1991}[] explains rounding and evaluation-order effects.
{Informal.citet flocq2011}[] separates formats from rounding operators in Coq, a
separation also used in TorchLean's float layer. {Informal.citet boldo2015}[] connects
rounded-arithmetic proofs to emitted code in CompCert. {ref "floating-point-literature"}[The
floating-point literature chapter] discusses how TorchLean's layer relates to each of them.

FloatLib's executable transcendental approximations are deterministic definitions, and
{ref "fp32-soundness"}[the FP32 soundness chapter] states what has been proved
about them. What has not been proved is a universal correctly-rounded contract. A passing
derivative check at $`\exp(-2x)` supports that input on that code path; it does not establish
accuracy for `exp`, `log`, `sin`, or `cos` over all finite bit patterns.

The display scale makes differences of about `8.88e-16` visible. It supplies no error bound beyond
these two evaluations.

# Parameterization, Reduction, And Arithmetic

Three variations isolate different parts of the model:

1. Replacing the $`-2` by a parameter and querying its gradient turns the forward
   model into the inverse problem above;
2. Widening `sfmBatchMean` from three observations to thirty reduces the contribution of each
   unchanged coordinate by the factor predicted by $`1/N`;
3. Evaluating the same finite input through FloatLib binary32 and host Float compares different
   arithmetic interpretations of the formula.

Change one choice at a time and update the independent derivative calculation with it, just as
the inner-scaling example changed both the exponential argument and its chain-rule factor.

Sources and further reading:

- {src "NN/Examples/Functional/Transcendentals.lean"}[`Transcendentals.lean`], the positive and
  negative controls run above;
- {src "NN/Runtime/Autograd/Model/Functional/Core.lean"}[`Functional/Core.lean`], the
  `nn.functional` primitives;
- {src "NN/Spec/Layers/Activation.lean"}[`Activation.lean`], the `softplus` and `safeLog`
  specifications;
- {Informal.citet baydin2018}[] on automatic differentiation, and {Informal.citep pytorch2019}[]
  for the framework the comparisons above use;
- {Informal.citet pinn2019}[] for the residual objective;
- {Informal.citet goldberg1991}[], {Informal.citet flocq2011}[], and
  {Informal.citet boldo2015}[] on floating-point arithmetic and its formalizations.
