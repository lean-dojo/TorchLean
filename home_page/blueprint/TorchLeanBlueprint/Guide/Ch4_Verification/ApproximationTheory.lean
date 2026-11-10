import VersoManual
import TorchLean.NN.UniversalApproximation.StoneWeierstrass
import TorchLean.NN.UniversalApproximation
import TorchLean.Verify.UniversalApproximation.Rounded
import TorchLean.Verify.UniversalApproximation.Binary
import TorchLean.Verify.UniversalApproximation.BinaryMLP.Error
import TorchLean.NN.UniversalApproximation.Rate
-- The worked mesh uses FloatLib scalar construction and bit observations directly.
import FloatLib
import TorchLeanBlueprint.Bib
import TorchLeanBlueprint.Roles

open Verso.Genre Manual
open Verso.Genre.Manual.InlineLean

-- Opening the approximation namespaces keeps displayed declarations within Verso's narrow code
-- column. The printed signatures still name everything in full.
open TorchLean.UniversalApproximation

-- Several of these signatures print wider than this file's 100-column limit, so their `leanOutput`
-- blocks ask for `whitespace := lax` and are wrapped in the source. The rendered page still shows
-- each message exactly as Lean printed it.
open Lean.Elab.Tactic.GuardMsgs.WhitespaceMode (lax)

open FloatLib.Floats (ExecFloat)
open FloatLib.Floats.Formats.BinaryInterchange (Model FloatFormat)

#doc (Manual) "Approximation Theory" =>
%%%
tag := "approximation-theory"
%%%

Suppose we want a ReLU network to represent a function `f` on an interval. An approximation theorem
may tell us that suitable weights *exist*. Once we choose weights, a verifier can try to enclose
that particular network's values on an input box, and to say what its binary32 execution does we
also need to account for rounding. Writing out the quantifiers shows why these are separate claims:

$$`\begin{aligned}
\text{representation:}\quad&
  \forall f\,\forall\varepsilon>0\,\exists\theta\,
  \sup_{x\in K}|N_\theta(x)-f(x)|<\varepsilon,\\
\text{verification:}\quad&
  \forall x\in B,\quad N_\theta(x)\in\mathcal A_\theta(B),\\
\text{execution:}\quad&
  \forall x\in B,\quad
  |N_{\theta,\mathrm{run}}(x)-N_{\theta,\mathbb R}(x)|\leq\delta(x).
\end{aligned}`

Only the first line is universal approximation. It doesn't certify a trained checkpoint, choose the
parameters for an optimizer, or prove that floating-point evaluation is close to the real network.
TorchLean keeps these statements near one another because they have to be composed, but it doesn't
identify them.

Where $`\exists\theta` sits is doing real work. We choose the target and tolerance first, then
obtain one parameter vector that works for every input in the domain. Moving the existential inside
the final input quantifier would permit a different network at every point, which would say very
little about representation. For a trained model, by contrast, the parameters have already been
fixed. So the useful question becomes whether those parameters satisfy an approximation bound, or
whether their outputs stay in an acceptable range. That's why a theorem about the network family
and a certificate for one checkpoint can both be useful without answering the same question.

# ReLU Hinge Approximation

Hinge notation makes the one-dimensional construction easier to understand. Let

$$`H(x)=b_0+\sum_{i=0}^{N-1} c_i\,\operatorname{ReLU}(x-t_i).`

Each term changes the slope at one knot $`t_i`. With knots on a sufficiently fine mesh and each
$`c_i` chosen from changes in the piecewise-linear slope, $`H` interpolates a Lipschitz target. And
the same expression is a one-hidden-layer network: the first linear layer computes $`x-t_i`, ReLU
applies the hinges, and the second linear layer forms their weighted sum.

For an exact example, take two hinges at the same knot $`t_0=0`, one reading $`x` and one reading
$`-x` (both with coefficient one). Their sum is the absolute value:

```lean
-- Split at zero, where the active hinge changes. Either
-- branch leaves one copy of the magnitude.
example (x : ℝ) : |x| = max x 0 + max (-x) 0 := by
  rcases le_total 0 x with h | h
  · rw [abs_of_nonneg h, max_eq_left h,
      max_eq_right (neg_nonpos.mpr h), add_zero]
  · rw [abs_of_nonpos h, max_eq_right h,
      max_eq_left (neg_nonneg.mpr h), zero_add]
```

Here `max y 0` is ReLU. So the right-hand side is a width-two ReLU network (first-layer weights
$`+1` and $`-1`, zero biases, and second-layer weights both $`1`). This example allows a hinge
facing each direction, while the interpolation formula above uses hinges facing right. Either way,
the proof splits at the knot: on one side of $`t_i` the hinge is inert, and on the other it
contributes its slope. The general theorem also uses the Lipschitz bound to choose a mesh and
control the interpolation error uniformly.

The theorem {src "TorchLean/NN/UniversalApproximation.lean"}[
`relu_Icc`] makes precisely that conversion. Its signature is:

```lean (name := uatIcc)
-- Expose the interval and Lipschitz premises before the
-- existential choice of network layers.
#check @relu_Icc
```

```leanOutput uatIcc (whitespace := lax)
@relu_Icc : ∀ {f : ℝ → ℝ} {a b L : ℝ},
  a < b →
    0 < L →
      (∀ x ∈ Set.Icc a b, ∀ y ∈ Set.Icc a b, |f x - f y| ≤ L * |x - y|) →
        ∀ ε > 0, ∃ hidDim l1 l2, ∀ x ∈ Set.Icc a b, |f x - mlpEvalScalar hidDim l1 l2 x| < ε
```

Here $`a<b` gives an interval with positive length. $`L>0` and `h_lip` provide a quantitative
continuity bound, and $`\varepsilon>0` is needed before a finite mesh can be chosen. From the
conclusion you get two TorchLean `Linear` values: the input-to-hidden layer and the
hidden-to-output layer used by `mlpEvalScalar`.

In the printed signature, `∀ x ∈ Set.Icc a b` abbreviates two arguments: an input `x` and a proof
that $`a\leq x\leq b`. Likewise, each arrow before the existential introduces a premise the caller
must supply. Braces around `{f : ℝ → ℝ}` mark an implicit parameter that Lean can infer from those
premises. Once `hidDim` is obtained, the types of `l1` and `l2` depend on that dimension, so their
hidden coordinates must agree. That dependency ties the analytic construction to a well-shaped
model. Strict positivity of `L` is harmless for a constant target: any positive upper bound is also
a Lipschitz bound for that target (even though its smallest Lipschitz constant is zero).

In the rate theorem, the width is

$$`N(L,a,b,\varepsilon)
  = \left\lceil\frac{2L(b-a)}{\varepsilon}\right\rceil+1.`

In Lean this is {src "TorchLean/NN/UniversalApproximation.lean"}[
`width`]. It's the witness in the qualitative theorem, and
`relu_Icc_rate` names the same hidden dimension in its statement.
Underneath, the arithmetic lemma proves

$$`\frac{2L(b-a)}{N}<\varepsilon.`

This bound is conservative. It establishes a construction with a stated size. It doesn't say that
the width is minimal, or that gradient descent will discover these hinge parameters.

# Approximation Rate And Network Width

Each `#check` below displays a declaration behind the width formula and is checked when the page
builds. The width takes four real arguments and returns a natural number:

```lean (name := widthSig)
-- The width is a natural count computed from four
-- real-valued specification parameters.
#check @width
```

```leanOutput widthSig
width : ℝ → ℝ → ℝ → ℝ → ℕ
```

By the arithmetic lemma, this choice of width makes the mesh error smaller than the requested
tolerance:

```lean (name := widthLt)
-- Inspect the strict mesh estimate obtained by adding one
-- after the ceiling.
#check @div_width_lt
```

```leanOutput widthLt (whitespace := lax)
@div_width_lt : ∀ {L a b ε : ℝ},
  0 < ε → 2 * L * (b - a) / ↑(width L a b ε) < ε
```

So the rate theorem has the same hypotheses as before, with the existential over `hidDim`
discharged by that formula:

```lean (name := uatRate)
-- The hidden dimension is fixed here, and only the two
-- layer parameter records remain existential.
#check @relu_Icc_rate
```

```leanOutput uatRate (whitespace := lax)
@relu_Icc_rate : ∀ {f : ℝ → ℝ} {a b L : ℝ},
  a < b →
    0 < L →
      (∀ x ∈ Set.Icc a b, ∀ y ∈ Set.Icc a b, |f x - f y| ≤ L * |x - y|) →
        ∀ ε > 0, ∃ l1 l2, ∀ x ∈ Set.Icc a b,
          |f x - mlpEvalScalar (width L a b ε) l1 l2 x| < ε
```

You can't evaluate the same definition with `#eval`:

```lean (name := widthEval) +error
-- This deliberately asks the compiler to execute a width
-- defined using mathematical reals.
#eval width 1 0 1 (1 / 10)
```

```leanOutput widthEval (whitespace := lax)
failed to compile definition, consider marking it as 'noncomputable' because it depends on
'width', which is 'noncomputable'
```

Because the width applies `Nat.ceil` to mathematical real numbers, it's a proof-level construction,
not a compiled numerical routine. That compiler error marks the boundary between an existence proof
over exact reals and an executable parameter-selection program. A runtime tool could compute the
same formula from rational inputs, but that would be a separate executable definition with a
refinement theorem.

But Lean can prove the value for specific inputs. For a `1`-Lipschitz target on $`[0,1]` at
$`\varepsilon=1/10`, the width is twenty-one hidden units:

```lean
-- Prove the real ceiling calculation for this tolerance
-- without compiling it as a numerical
-- routine.
example :
    width 1 0 1 (1 / 10) = 21 := by
  norm_num [width]
```

That is $`\lceil 2\cdot 1\cdot 1/(1/10)\rceil+1=21`, and `norm_num` proves the arithmetic
(including the ceiling). Halving $`\varepsilon` approximately doubles the width, so this
construction is linear in $`1/\varepsilon`.

Adding one after the ceiling makes the mesh estimate strict, even when the ratio is an integer. In
the signature, `↑` casts the positive natural width to a real denominator.

# Quadratic Approximation In Binary32

To construct concrete parameters, use four hinges on $`[0,1]` for the target $`f(x)=x^2`. The
following code computes the knots and chord slopes in `Float`, converts the knots and slope
increments to FloatLib binary32, and prints the resulting parameters. We evaluate their hinge sum
in binary32 arithmetic in the next example.

```lean (name := meshParams)
-- Each coefficient records a slope increment, so active
-- hinges reconstruct the chord slope.
def meshN : Nat := 4
def meshStep : Float := 1.0 / 4.0
def target (x : Float) : Float := x * x

def knot (i : Fin meshN) : (ExecFloat.Binary 8 23) :=
  (ExecFloat.Binary.ofFloat32 ∘ Float.toFloat32)
    (Nat.toFloat i.val * meshStep)

-- Slope of the chord of `target` over subinterval `i`.
def chordSlope (i : Nat) : Float :=
  (target (Nat.toFloat (i + 1) * meshStep)
    - target (Nat.toFloat i * meshStep)) / meshStep

def coeff (i : Fin meshN) : (ExecFloat.Binary 8 23) :=
  (ExecFloat.Binary.ofFloat32 ∘ Float.toFloat32)
    (if i.val = 0 then chordSlope 0
     else chordSlope i.val - chordSlope (i.val - 1))

#eval do
  let ts := (List.finRange meshN).map
    (knot · |> ExecFloat.Binary.toFloat32 |>.toFloat)
  let cs := (List.finRange meshN).map
    (coeff · |> ExecFloat.Binary.toFloat32 |>.toFloat)
  let ss := (List.range meshN).map chordSlope
  IO.println s!"knots  {ts}"
  IO.println s!"slopes {ss}"
  IO.println s!"coeffs {cs}"
```

```leanOutput meshParams (whitespace := lax)
knots  [0.000000, 0.250000, 0.500000, 0.750000]
slopes [0.250000, 0.750000, 1.250000, 1.750000]
coeffs [0.250000, 0.500000, 0.500000, 0.500000]
```

A hinge
$`c_i\operatorname{ReLU}(x-t_i)` is inert to the left of its knot and contributes slope $`c_i` to
the right, so between knots the sum has slope $`\sum_{t_i\leq x} c_i`. For that running total to
match the chord slopes $`s_0,\dots,s_{N-1}` printed above, the coefficients have to be the
*increments*: $`c_0=s_0` and $`c_i=s_i-s_{i-1}`. Slopes $`0.25,0.75,1.25,1.75` therefore give
coefficients $`0.25,0.5,0.5,0.5`. For example, after the second knot the first two hinges are
active: their coefficients add to the second chord's slope. This cumulative addition is why
`coeff` stores differences of slopes.

At $`x=3/8`, only the knots at zero and one quarter contribute. Their terms are $`(1/4)(3/8)=3/32`
and $`(1/2)(1/8)=1/16`, giving $`H(x)=5/32`. Since the target is $`9/64`, the difference is $`1/64`
(the `0.015625` in the table below). This also explains the tensor layout in the PyTorch
transcription below: each hidden coordinate stores one distance from a knot, and the output weight
on that coordinate stores a slope change. So the four coefficients aren't four independent chord
slopes. Copying the printed slope list into the output layer would change the function, even though
all layer dimensions would still match.

Evaluate the network on a grid twice as fine as its own mesh, so that both the knots and the points
between them appear:

```lean (name := meshRun)
-- Sample both knots and midpoints to distinguish
-- interpolation agreement from between-knot error.
#eval do
  for k in [0, 1, 2, 3, 4, 5, 6, 7, 8] do
    let x := Nat.toFloat k * 0.125
    let xe :=
      (ExecFloat.Binary.ofFloat32 ∘ Float.toFloat32) x
    let hv :=
      (ExecFloat.Binary.toFloat32
        (hingeFun knot coeff 0 xe)).toFloat
    IO.println s!"x={x}  f={target x}  H={hv}"
```

```leanOutput meshRun (whitespace := lax)
x=0.000000  f=0.000000  H=0.000000
x=0.125000  f=0.015625  H=0.031250
x=0.250000  f=0.062500  H=0.062500
x=0.375000  f=0.140625  H=0.156250
x=0.500000  f=0.250000  H=0.250000
x=0.625000  f=0.390625  H=0.406250
x=0.750000  f=0.562500  H=0.562500
x=0.875000  f=0.765625  H=0.781250
x=1.000000  f=1.000000  H=1.000000
```

At every knot the hinge sum agrees with $`x^2`, because its slopes match the chords. Between knots
the real chord interpolant lies above the curve, since the target is convex. In the displayed
calculation the overshoot is $`0.015625` at every midpoint: the midpoint error of a chord
interpolant of this quadratic is $`\tfrac18 f''h^2=\tfrac18\cdot 2\cdot(1/4)^2`.

Compare that with what the rate theorem certifies at this width. Since $`x^2` is $`2`-Lipschitz on
$`[0,1]`, the formula reaches four or fewer hidden units only once $`\varepsilon` is at least
$`4/3`:

```lean
-- Compare the prescribed widths for the same Lipschitz
-- bound at three tolerances.
example :
    width 2 0 1 (4 / 3) = 4 := by
  norm_num [width]

example :
    width 2 0 1 (1 / 10) = 41 := by
  norm_num [width]

example :
    width 2 0 1 (1 / 20) = 81 := by
  norm_num [width]
```

Here the rate theorem supplies a real network with error below $`4/3`, while the displayed binary32
mesh has sampled error $`0.015625`. To apply the theorem to these particular parameters, you have
to identify its real construction with this mesh and then bound conversion and execution error. And
the width calculation alone doesn't discharge those obligations. A sharper interpolation bound can
also use the second derivative of this smooth target (information absent from the Lipschitz
hypotheses). So the sampled error and the existence guarantee describe different claims.

## Sine Approximation

Convexity explains the one-sided interpolation error for $`x^2` and the increasing chord slopes,
but the hinge construction itself only needs the slopes and their differences. Apply it to eight
hinges on $`[0,1]` for $`f(x)=\sin(2\pi x)`:

```lean (name := meshWave)
-- Allow signed slope increments so the interpolant can
-- follow both halves of the wave.
def waveN : Nat := 8
def waveStep : Float := 1.0 / 8.0

def wave (x : Float) : Float :=
  Float.sin (2.0 * 3.141592653589793 * x)

def waveKnot (i : Fin waveN) : (ExecFloat.Binary 8 23) :=
  (ExecFloat.Binary.ofFloat32 ∘ Float.toFloat32)
    (Nat.toFloat i.val * waveStep)

def waveChord (i : Nat) : Float :=
  (wave (Nat.toFloat (i + 1) * waveStep)
    - wave (Nat.toFloat i * waveStep)) / waveStep

def waveCoeff (i : Fin waveN) : (ExecFloat.Binary 8 23) :=
  (ExecFloat.Binary.ofFloat32 ∘ Float.toFloat32)
    (if i.val = 0 then waveChord 0
     else waveChord i.val - waveChord (i.val - 1))

def waveHinge (x : Float) : Float :=
  (ExecFloat.Binary.toFloat32
    (hingeFun waveKnot waveCoeff 0
      ((ExecFloat.Binary.ofFloat32 ∘ Float.toFloat32)
        x))).toFloat

#eval do
  let cs :=
    (List.finRange waveN).map
      (waveCoeff · |> ExecFloat.Binary.toFloat32 |>.toFloat)
  IO.println s!"coeffs {cs}"
  -- Signed error at four midpoints, two per half.
  for k in [1, 3, 9, 11] do
    let x := Nat.toFloat k * 0.0625
    IO.println
      s!"x={x} f={wave x} H-f={waveHinge x - wave x}"
  let mut worst := 0.0
  for k in [0:17] do
    let x := Nat.toFloat k * 0.0625
    worst := max worst (waveHinge x - wave x).abs
  IO.println s!"worst on this grid = {worst}"
```
```leanOutput meshWave (whitespace := lax)
coeffs [5.656854, -3.313709, -4.686292,
  -3.313709, -0.000000, 3.313709, 4.686292,
  3.313709]
x=0.062500 f=0.382683 H-f=-0.029130
x=0.187500 f=0.923880 H-f=-0.070326
x=0.562500 f=-0.382683 H-f=0.029130
x=0.687500 f=-0.923880 H-f=0.070326
worst on this grid = 0.070326
```

Here the coefficients change sign because the chord slopes decrease as well as increase, so a hinge
can cancel the slope its predecessors added, not only steepen the curve. And the midpoint errors
have both signs: the interpolant sits below $`f` where $`f` is concave and above it where $`f` is
convex, so no one-sided statement of the kind that held for $`x^2` is available. In both cases the
Lipschitz error bound controls the absolute error.

The rate formula reacts to this target the way a worst-case bound has to. Since $`\sin(2\pi x)` is
$`2\pi`-Lipschitz, take $`L=7` as a rational upper bound and ask what eight hidden units certify:

```lean
-- Use the rational Lipschitz upper bound seven in the same
-- width formula.
example : width 7 0 1 2 = 8 := by
  norm_num [width]
```

So the formula assigns width eight to $`\varepsilon=2`. For this particular $`f` (its range is
$`[-1,1]`), that tolerance is loose: even the constant zero function has error at most one. On the
sampled grid, the displayed mesh's measured error is about $`0.07`. Because the rate theorem uses
only the supplied Lipschitz constant, it covers every $`7`-Lipschitz target with the same width. It
doesn't use the sine function's curvature. And the grid measurement doesn't establish a uniform
bound between samples.

Both signs in `H-f` are useful here. A positive number means the approximation lies above the
target, and taking an absolute value too early would hide the change in curvature between the two
halves of the wave. `worst` ends up as a maximum over seventeen evaluated inputs, whereas the
analytic error statement quantifies over an interval. To turn such a grid into a regional bound,
you'd also need to control how the error varies between adjacent inputs. In the Lipschitz
construction, that control comes from its hypotheses and mesh argument. Instead, the printed table
helps you inspect whether the chosen slope increments behave as intended.

## ReLU Approximation In PyTorch

We can place the same knots and slope increments in two `torch.nn` linear layers. In the first
layer the weights are all ones and the biases are $`-t_i`, and the second layer weights the
resulting hinges:

```
# Place each knot in the first-layer bias and each slope
# increment in the output weight.
import torch, torch.nn as nn

knots = torch.tensor([0.0, 0.25, 0.50, 0.75])
coeff = torch.tensor([[0.25, 0.5, 0.5, 0.5]])

net = nn.Sequential(nn.Linear(1, 4), nn.ReLU(), nn.Linear(4, 1))
with torch.no_grad():
    net[0].weight.copy_(torch.ones(4, 1))
    net[0].bias.copy_(-knots)
    net[2].weight.copy_(coeff)
    net[2].bias.zero_()

xs = torch.arange(0, 9, dtype=torch.float32).mul(0.125).unsqueeze(1)
with torch.no_grad():
    ys = net(xs).squeeze(1)
```

On CPU, the PyTorch transcription produces the nine `H` values printed above. Every input,
parameter, product, and partial sum on this grid is exactly representable in binary32. This check
covers these inputs on one backend, not every input or backend {Informal.citep pytorch2019}[].

Lean's table evaluates `hingeFun` at binary32 values, the evaluation used in the three-term
executable approximation theorem below. Applying that theorem to these parameters still requires
proofs of its approximation, quantization, and finiteness premises. A transcription into another
framework is a claim that has to be checked separately, which is what {ref "pytorch-roundtrip"}[the
roundtrip chapter] is for.

## Rounding Error

In the quadratic example, the hinge sum agrees with the target at the knots down to the last bit.
Both the knots and the chord slopes are multiples of $`1/4`, and on this small grid the products
and partial sums are also exactly representable in binary32. To expose differences hidden by
decimal formatting, the comparison below subtracts the bit patterns of the two positive outputs:

```lean (name := meshExact)
-- Compare stored encodings at interior dyadic knots, where
-- decimal output could hide a difference.
#eval do
  for k in [1, 2, 3] do
    let x := Nat.toFloat k * meshStep
    let xe :=
      (ExecFloat.Binary.ofFloat32 ∘ Float.toFloat32) x
    let hx := hingeFun knot coeff 0 xe
    let fx := (ExecFloat.Binary.ofFloat32 ∘ Float.toFloat32)
      (target x)
    let distance :=
      ExecFloat.Binary.toBits32 hx -
        ExecFloat.Binary.toBits32 fx
    IO.println s!"knot {k}: {distance} ulp"
```

```leanOutput meshExact
knot 1: 0 ulp
knot 2: 0 ulp
knot 3: 0 ulp
```

Move the same construction onto a mesh of three and the knots become multiples of $`1/3`, which
binary32 cannot represent:

```lean (name := meshThirds)
-- Convert the thirds-based construction to binary32 before
-- comparing its interior-knot outputs.
def thirds : Float := 1.0 / 3.0

def knot3 (i : Fin 3) : (ExecFloat.Binary 8 23) :=
  (ExecFloat.Binary.ofFloat32 ∘ Float.toFloat32)
    (Nat.toFloat i.val * thirds)

def chord3 (i : Nat) : Float :=
  (target (Nat.toFloat (i + 1) * thirds)
    - target (Nat.toFloat i * thirds)) / thirds

def coeff3 (i : Fin 3) : (ExecFloat.Binary 8 23) :=
  (ExecFloat.Binary.ofFloat32 ∘ Float.toFloat32)
    (if i.val = 0 then chord3 0
     else chord3 i.val - chord3 (i.val - 1))

#eval do
  for k in [1, 2] do
    let x := Nat.toFloat k * thirds
    let xe :=
      (ExecFloat.Binary.ofFloat32 ∘ Float.toFloat32) x
    let hx := hingeFun knot3 coeff3 0 xe
    let fx := (ExecFloat.Binary.ofFloat32 ∘ Float.toFloat32)
      (target x)
    let distance :=
      ExecFloat.Binary.toBits32 hx -
        ExecFloat.Binary.toBits32 fx
    IO.println s!"knot {k}: {distance} ulp"
```

```leanOutput meshThirds
knot 1: 1 ulp
knot 2: 1 ulp
```

At each interior knot the outputs are adjacent positive binary32 values, one ULP apart.
`Float`'s six-digit rendering shows `0.111111` for both outputs at the first knot and `0.444444`
for both at the second, hiding the discrepancy. This is an example of the distinction between a
stored value and its decimal display discussed by Goldberg {Informal.citep goldberg1991}[].

This comparison includes several rounding steps: conversion of the input and coefficients,
evaluation of the hinge sum, and conversion of the reference target value. The theorems below
separate parameter conversion from evaluation error, and they require finite executable
intermediates so that each arithmetic step can be related to a real-valued expression.

# From Real Parameters To A Floating-Point Format

Our worked examples use binary32, but the proof takes the format as a parameter. An executable
value also specifies its storage plan and a FloatLib codec relating stored values to the model. The
arithmetic refinement assumes `format.isIEEE = true`, so binary16, binary32, binary64, and
binary128 can use the same theorem. Once knots, coefficients, and the bias are stored in the chosen
format, the total error naturally splits into

$$`\begin{aligned}
|f(x)-H_{\mathrm{IEEE}}(x)|
\leq{}&
|f(x)-H_{\mathbb R}(x)|\\
&+|H_{\mathbb R}(x)-H_{\mathrm{embedded}}(x)|\\
&+|H_{\mathrm{embedded}}(x)-H_{\mathrm{IEEE}}(x)|.
\end{aligned}`

These are, respectively:

1. approximation error of the real hinge network,
2. parameter-quantization or reference error, and
3. rounded evaluation error.

You can see the distinction in
{src "TorchLean/Verify/UniversalApproximation/Binary.lean"}[the
three-term approximation theorem]. It takes real hinge parameters `tR` and `cR`, executable
parameters `t`, `c`, and `b0` in that format, and separate assumptions for the real approximation
and quantization terms. It also requires finiteness witnesses for the intermediate hinge sum and
output. Its conclusion adds `hingeFunErrorBound` to the two supplied tolerances.

To use this theorem, conversion and evaluation must satisfy its finiteness assumptions. Its
dyadic specialization starts from dyadic parameters and uses a half-ULP
rounding bound, but it still asks for finite evaluation witnesses.

The full signature exposes the obligations needed to connect the real approximant to executable
arithmetic:

```lean (name := ieeeThree)
-- Inspect the separate approximation,
-- parameter-quantization, and finite-execution obligations.
#check @relu_Icc_binary_three_term
```

```leanOutput ieeeThree (whitespace := lax)
@relu_Icc_binary_three_term : ∀ {format : FloatFormat}
  {plan : FloatLib.Floats.Formats.BinaryInterchange.Configured.StoragePlan format} {code : Type}
  [inst : ExecFloat.ModelCodec plan (Model format) code],
  format.isIEEE = true →
    ∀ {f : ℝ → ℝ} {a b : ℝ} {hidDim : ℕ} (tR cR : Fin hidDim → ℝ)
      (t c : Fin hidDim → ExecFloat (FloatLib.Floats.Formats.BinaryInterchange.Configured.Family
    format code plan))
      (b0 : ExecFloat (FloatLib.Floats.Formats.BinaryInterchange.Configured.Family
    format code plan)) (εApprox εQ : ℝ),
      (∀ (x : ExecFloat (FloatLib.Floats.Formats.BinaryInterchange.Configured.Family
    format code plan)),
          ExecFloat.Binary.isFinite x = true →
            (ExecFloat.Binary.toModel x).toReal ∈ Set.Icc a b →
              HingeSumFinite t c x 0 (List.finRange hidDim) ∧
                (hingeFun t c b0 x).isFinite = true) →
        (∀ (x : ExecFloat (FloatLib.Floats.Formats.BinaryInterchange.Configured.Family
    format code plan)),
            ExecFloat.Binary.isFinite x = true →
              (ExecFloat.Binary.toModel x).toReal ∈ Set.Icc a b →
                |f (ExecFloat.Binary.toModel x).toReal -
                      hingeFun tR cR (f a) (ExecFloat.Binary.toModel x).toReal| <
                  εApprox) →
          (∀ (x : ExecFloat (FloatLib.Floats.Formats.BinaryInterchange.Configured.Family
    format code plan)),
              ExecFloat.Binary.isFinite x = true →
                (ExecFloat.Binary.toModel x).toReal ∈ Set.Icc a b →
                  |hingeFun tR cR (f a) (ExecFloat.Binary.toModel x).toReal -
                        Rounded.hingeFunReal format (toRoundedVec t) (toRoundedVec c)
    (toRounded b0) (toRounded x)| ≤
                    εQ) →
            ∀ (x : ExecFloat (FloatLib.Floats.Formats.BinaryInterchange.Configured.Family
    format code plan)),
              ExecFloat.Binary.isFinite x = true →
                (ExecFloat.Binary.toModel x).toReal ∈ Set.Icc a b →
                  |f (ExecFloat.Binary.toModel x).toReal -
                        (hingeFun t c b0 x).toModel.toReal| <
                    εApprox + εQ +
                      Rounded.hingeFunErrorBound format (toRoundedVec t) (toRoundedVec c)
    (toRounded b0) (toRounded x)
```



Read the signature from the top. Parameter vectors come in pairs, `tR cR` over the reals and `t c`
over the configured format, because the theorem has to talk about both to relate them. Notice the
`HingeSumFinite` conjunct: it demands that *every intermediate* of the fold stay finite, not merely
the result. At each arithmetic step the proof uses finite real embeddings, so an overflowing
intermediate breaks that argument. That same conjunct rules out NaN and infinite parameters (a
condition a real-valued statement couldn't even express): a non-finite knot or coefficient makes
its subtraction or product non-finite, so `t` and `c` need no `isFinite` hypotheses of their own.
Next come the $`\varepsilon_{\mathrm{Approx}}` and $`\varepsilon_Q` budgets from the three-term
split, two hypotheses that are supplied, not derived. And the conclusion adds `hingeFunErrorBound`
(the rounding of the evaluation itself) to their sum.

For a particular checkpoint, each premise must be proved for its stored parameters and the stated
input domain. `instDecHingeSumFinite` makes the `HingeSumFinite` predicate decidable for a fixed
input and concrete parameters. After installing it with `haveI`, `decide` can prove a true concrete
instance, but a premise quantified over all inputs in an interval still needs an argument covering
that whole domain.

Notice that the final input is a stored value in the chosen format, and interval membership is
tested on its decoded real value `Model.toReal (ExecFloat.Binary.toModel x)`, so the target is
evaluated at the real number that stored input represents. If an application starts with an ideal
real measurement and rounds it before model evaluation, its input-conversion error is an additional
question. A Lipschitz bound on the target can help relate those two inputs, but that step isn't
hidden inside `εQ`: the quantization premise shown here compares two parameterizations at the same
embedded input. Keeping that input fixed makes the three terms correspond to three identifiable
functions.

A companion theorem for dyadic parameters replaces the supplied $`\varepsilon_Q` with a half-ULP
bound derived from dyadic parameters. It's the more usable form when reference parameters are given
as dyadics. It assumes knots in $`[a,b]` and finite rounded parameters, and it propagates the
per-parameter half-ULP errors through the hinge sensitivities (including the interval length
$`|b-a|`). Finite-evaluation witnesses remain necessary, because finite stored parameters can still
produce an overflowing intermediate.

The dyadic specialization exposes those witnesses in its signature:

```lean (name := ieeeHalfUlp)
-- Dyadic reference parameters supply the quantization
-- estimate, but finite evaluation is still required.
#check @relu_Icc_binary_dyadic
```

TorchLean also has an intermediate
{src "TorchLean/Verify/UniversalApproximation/Rounded.lean"}[
rounded-real theorem].
`Rounded.relu_Icc format` evaluates the hinge construction with
rounding at each operation and proves a pointwise bound of the form

$$`|f(x)-H_{\mathrm{rounded}}(x)|
  < \varepsilon+\operatorname{hingeFunErrorBound}(x).`

```lean (name := roundedUat)
-- This bound concerns the rounded-real carrier and reads
-- its result through the val field.
#check @Rounded.relu_Icc
```

```leanOutput roundedUat (whitespace := lax)
Rounded.relu_Icc : ∀ (format : FloatFormat)
  {f : ℝ → ℝ} {a b L : ℝ},
  a < b →
    0 < L →
      (∀ x ∈ Set.Icc a b, ∀ y ∈ Set.Icc a b, |f x - f y| ≤ L * |x - y|) →
        ∀ ε > 0,
          ∃ hidDim t c b0,
            ∀ x ∈ Set.Icc a b,
              |f x - (hingeFun t c b0 { val := x }).val| <
                ε + Rounded.hingeFunErrorBound format t c b0 { val := x }
```



Compare that conclusion with the real one printed earlier: the hypotheses are identical, the
existential produces hinge parameters in place of a `Linear` pair, and the bound has grown a
second summand. That summand bounds evaluation in this model, and it doesn't include a separate
conversion of arbitrary real parameters into finite storage.

`Rounded format` values carry arbitrary reals, and each primitive operation rounds its result to
the format's precision and underflow grid (with no upper exponent cutoff). This carrier is
FloatLib's `NF`, specialized by the format descriptor. Under the IEEE-format and
finite-intermediate hypotheses, the executable bridge connects it to stored values. Changing
`format` changes the grid and the error bounds without selecting a different proof file.

# Higher-Dimensional Domains

Our one-dimensional hinge proof is constructive and quantitative. For higher dimensions, the
development takes a different route. In
{src "TorchLean/NN/UniversalApproximation/StoneWeierstrass.lean"}[`StoneWeierstrass`],
`Tensor ℝ [n]` is identified with `Fin n → ℝ` by a homeomorphism. Coordinate functions generate a
subalgebra of continuous functions, and the proof establishes that this subalgebra separates
points: distinct tensors differ in at least one coordinate, and the corresponding coordinate
function distinguishes them. Mathlib's Stone-Weierstrass theorem {Informal.citep mathlib2020}[]
then supplies density on compact domains. TorchLean's own part establishes the homeomorphism and
checks the subalgebra hypotheses that theorem needs.

Here the approximating objects are coordinate-polynomial expressions. A subalgebra contains
constants and is closed under addition, multiplication, and scalar multiplication, so starting from
coordinate projections permits expressions such as a coordinate squared or a product of two
coordinates. Point separation says this class can distinguish any two different tensor inputs.
Compactness is what lets density be stated using uniform error over the whole domain. In the
source, the conclusion `∃ g : coordSubalg, ‖(g : C(K, ℝ)) - f‖ < ε` chooses one such continuous
function (`C(K, ℝ)` bundles continuity with the function, and its norm measures the largest
absolute error). A neural construction built on this result still has to represent or approximate
that polynomial expression.

That topological argument answers a broad representation question, but it doesn't produce a width
formula like the one from the one-dimensional Lipschitz construction. So the two proofs are
complementary:

- the hinge mesh exposes parameters and a rate on $`[a,b]`, and
- the coordinate-subalgebra proof handles compact multidimensional domains at a more abstract
  level.

# Approximation And Enclosure Guarantees

For the corresponding two-layer network, the scalar hinge construction also has a theorem. In the
{src "TorchLean/Verify/UniversalApproximation/BinaryMLP/Error.lean"}[
two-layer configured binary approximation API], `relu_mlp_binary_three_term` packages
representation, parameter-rounding, and executable-rounding error. Unlike the hinge theorem, this
one takes all three budgets (including the execution bound) as hypotheses and composes them by the
triangle inequality.

The theorem is checked as this page builds. Hovering over it shows its hypotheses:

```lean (name := endpoints)
-- The network theorem takes all three error budgets as
-- hypotheses.
#check @relu_mlp_binary_three_term
```

For the quantitative two-layer approximation theorem introduced earlier, the error budgets give:

$$`\text{target error}
\leq
\text{representation error}
+\text{parameter rounding error}
+\text{execution error}.`

Those three terms come from representation, parameter conversion, and executable arithmetic, and
writing the sum out lets an application decide where to spend its error budget. Separately, a CROWN
or interval certificate can supply a sound enclosure over an input region for the chosen network.
That enclosure is a regional property, not automatically another additive error term in this
inequality.

The constructions follow the classical universal-approximation tradition.
{Informal.citet cybenko1989}[] proved density for superpositions of a sigmoidal function, and
{Informal.citet hornik1991}[] removed the sigmoid-specific hypothesis for multilayer feedforward
networks. Both are density results: they say an approximant exists and are silent about its size.
The explicit ReLU construction obtains a width formula by choosing a mesh and computing its
slope increments.
