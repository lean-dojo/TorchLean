import VersoManual
import NN.MLTheory.Proofs.Approximation.FloatInterval.ConstantTarget
import NN.MLTheory.Proofs.Approximation.FloatInterval.ExactImageTheorem
import NN.MLTheory.Proofs.Approximation.FloatInterval.Semantics
import NN.MLTheory.Proofs.Approximation.Universal.StoneWeierstrass
import NN.MLTheory.Proofs.Approximation.Universal.UniversalApproximation
import NN.MLTheory.Proofs.Approximation.Universal.UniversalApproximationFP32
import NN.MLTheory.Proofs.Approximation.Universal.UniversalApproximationIEEE32Exec
import NN.MLTheory.Proofs.Approximation.Universal.UniversalApproximationIEEE32ExecTwoLayerMlp
import NN.MLTheory.Proofs.Approximation.Universal.UniversalApproximationRate
-- The worked mesh uses FloatLib scalar construction and bit observations directly.
import FloatLib
import TorchLeanBlueprint.Bib
import TorchLeanBlueprint.Roles

open Verso.Genre Manual
open Verso.Genre.Manual.InlineLean

-- The approximation results are spread over one shared namespace and four sub-namespaces: the
-- rounded-real and executable hinge theorems, the two-layer executable bridge, and the finite
-- interval domain. Opening them here keeps every displayed `#check` on one line, which is what
-- Verso's narrow code column allows. The printed signatures still name everything in full.
open NN.MLTheory.Proofs.UniversalApproximation
open NN.MLTheory.Proofs.UniversalApproximation.IEEE32ExecReLUApprox
open NN.MLTheory.Proofs.UniversalApproximation.IEEE32ExecTwoLayerMLP
open NN.MLTheory.Proofs.UniversalApproximation.FloatIntervalApprox
open NN.MLTheory.Proofs.UniversalApproximation.FloatIntervalApprox.ConstantTarget
open NN.MLTheory.Proofs.UniversalApproximation.FloatIntervalApprox.TwoLayerMLPExact

-- The retained approximation bridges share this theorem namespace.
open TorchLean.Floats.IEEE754

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

Suppose we want a ReLU network to represent a function `f` on an interval. An approximation
theorem may tell us that suitable weights *exist*. Once we choose weights, a verifier can try to
enclose that particular network's values on an input box. To say what its binary32 execution
does, we also need to account for rounding. The quantifiers show why these are separate claims:

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

Only the first line is universal approximation. It does not certify a trained checkpoint, choose
the parameters for an optimizer, or prove that floating-point evaluation is close to the real
network. TorchLean keeps these statements near one another because they eventually need to be
composed, but it does not identify them.

The position of $`\exists\theta` is doing real work. We choose the target and tolerance first,
then obtain one parameter vector that works for every input in the domain. Moving the existential
inside the final input quantifier would permit a different network at every point, which would
say very little about representation. For a trained model, by contrast, the parameters have
already been fixed. The useful question becomes whether those parameters satisfy an approximation
bound, or whether their outputs stay in an acceptable range. This is why a theorem about the
network family and a certificate for one checkpoint can both be useful without answering the
same question.

# ReLU Hinge Approximation

The one-dimensional construction is easier to understand in hinge notation. Let

$$`H(x)=b_0+\sum_{i=0}^{N-1} c_i\,\operatorname{ReLU}(x-t_i).`

Each term changes the slope at one knot $`t_i`. By choosing knots on a sufficiently fine mesh and
choosing $`c_i` from changes in the piecewise-linear slope, $`H` interpolates a Lipschitz target.
The same expression is a one-hidden-layer network: the first linear layer computes $`x-t_i`, ReLU
applies the hinges, and the second linear layer forms their weighted sum.

For an exact example, take two hinges at the same knot $`t_0=0`, one reading
$`x` and one reading $`-x`, both with coefficient one. Their sum is the absolute value:

```lean
-- Split at zero, where the active hinge changes; either
-- branch leaves one copy of the magnitude.
example (x : ℝ) : |x| = max x 0 + max (-x) 0 := by
  rcases le_total 0 x with h | h
  · rw [abs_of_nonneg h, max_eq_left h,
      max_eq_right (neg_nonpos.mpr h), add_zero]
  · rw [abs_of_nonpos h, max_eq_right h,
      max_eq_left (neg_nonneg.mpr h), zero_add]
```

Here `max y 0` is ReLU, so the right-hand side is a width-two ReLU network with first-layer weights
$`+1` and $`-1`, zero biases, and second-layer weights both $`1`. This example allows a hinge
facing each direction; the interpolation formula above uses hinges facing right. In either case,
the proof splits at the knot: on one side of $`t_i` the hinge is inert, on the other it contributes
its slope. The general theorem also uses the Lipschitz bound to choose a mesh and control the
interpolation error uniformly.

The theorem
{src "NN/MLTheory/Proofs/Approximation/Universal/UniversalApproximation.lean"}[
`relu_universal_approximation_Icc`] makes precisely that conversion. Its current signature is:

```lean (name := uatIcc)
-- Expose the interval and Lipschitz premises before the
-- existential choice of network layers.
#check @relu_universal_approximation_Icc
```

```leanOutput uatIcc (whitespace := lax)
@relu_universal_approximation_Icc : ∀ {f : ℝ → ℝ} {a b L : ℝ},
  a < b →
    0 < L →
      (∀ x ∈ Set.Icc a b, ∀ y ∈ Set.Icc a b, |f x - f y| ≤ L * |x - y|) →
        ∀ ε > 0, ∃ hidDim l1 l2, ∀ x ∈ Set.Icc a b, |f x - mlpEvalScalar hidDim l1 l2 x| < ε
```

Here $`a<b` gives an interval with positive length.
$`L>0` and `h_lip` provide a quantitative continuity bound. $`\varepsilon>0` is needed before a
finite mesh can be chosen. The conclusion supplies two TorchLean `LinearSpec` values: the
input-to-hidden layer and the hidden-to-output layer used by `mlpEvalScalar`.

The printed `∀ x ∈ Set.Icc a b` abbreviates two arguments: an input `x` and a proof that
$`a\leq x\leq b`. Likewise, each arrow before the existential introduces a premise the caller
must supply. The braces around `{f : ℝ → ℝ}` mark an implicit parameter that Lean can infer from
those premises. After obtaining `hidDim`, the types of `l1` and `l2` depend on that dimension;
their hidden coordinates must agree. This ties the analytic construction to a well-shaped model.
Strict positivity of `L` is harmless for a constant target: any positive upper bound is also a
Lipschitz bound for that target, even though its smallest Lipschitz constant is zero.

The rate theorem chooses the width

$$`N(L,a,b,\varepsilon)
  = \left\lceil\frac{2L(b-a)}{\varepsilon}\right\rceil+1.`

In Lean this is
{src "NN/MLTheory/Proofs/Approximation/Universal/UniversalApproximationRate.lean"}[
`reluApproximationWidth`], and `relu_universal_approximation_Icc_rate` uses exactly that hidden
dimension. The arithmetic lemma underneath it proves

$$`\frac{2L(b-a)}{N}<\varepsilon.`

This bound is conservative. It establishes a construction with a stated size; it does not claim
that the width is minimal, nor that gradient descent will discover these hinge parameters.

# Approximation Rate And Network Width

The following `#check` commands display the declarations behind the width formula and are checked
when the page builds. The width takes four real arguments and returns a natural number:

```lean (name := widthSig)
-- The width is a natural count computed from four
-- real-valued specification parameters.
#check @reluApproximationWidth
```

```leanOutput widthSig
reluApproximationWidth : ℝ → ℝ → ℝ → ℝ → ℕ
```

The arithmetic lemma shows that this choice of width makes the mesh error smaller than the
requested tolerance:

```lean (name := widthLt)
-- Inspect the strict mesh estimate obtained by adding one
-- after the ceiling.
#check @two_mul_mul_sub_div_relu_approximation_width_lt
```

```leanOutput widthLt (whitespace := lax)
@two_mul_mul_sub_div_relu_approximation_width_lt : ∀ {L a b ε : ℝ},
  0 < ε → 2 * L * (b - a) / ↑(reluApproximationWidth L a b ε) < ε
```

The rate theorem then has the same hypotheses as before, with the existential over `hidDim`
discharged by that formula:

```lean (name := uatRate)
-- The hidden dimension is now fixed; only the two layer
-- parameter records remain existential.
#check @relu_universal_approximation_Icc_rate
```

```leanOutput uatRate (whitespace := lax)
@relu_universal_approximation_Icc_rate : ∀ {f : ℝ → ℝ} {a b L : ℝ},
  a < b →
    0 < L →
      (∀ x ∈ Set.Icc a b, ∀ y ∈ Set.Icc a b, |f x - f y| ≤ L * |x - y|) →
        ∀ ε > 0, ∃ l1 l2, ∀ x ∈ Set.Icc a b,
          |f x - mlpEvalScalar (reluApproximationWidth L a b ε) l1 l2 x| < ε
```

The same definition cannot be evaluated with `#eval`:

```lean (name := widthEval) +error
-- This deliberately asks the compiler to execute a width
-- defined using mathematical reals.
#eval reluApproximationWidth 1 0 1 (1 / 10)
```

```leanOutput widthEval (whitespace := lax)
failed to compile definition, consider marking it as 'noncomputable' because it depends on
'reluApproximationWidth', which is 'noncomputable'
```

The width uses `Nat.ceil` on mathematical real numbers, so it is a proof-level construction rather
than a compiled numerical routine. That error marks the boundary between an existence proof over
exact reals and an executable parameter-selection program. A runtime tool could compute the same
formula from rational inputs, but that would be a separate executable definition with a refinement
theorem.

Lean can still prove the value for specific inputs. For a
`1`-Lipschitz target on $`[0,1]` at $`\varepsilon=1/10`, the width is twenty-one hidden units:

```lean
-- Prove the real ceiling calculation for this tolerance
-- without compiling it as a numerical
-- routine.
example :
    reluApproximationWidth 1 0 1 (1 / 10) = 21 := by
  norm_num [reluApproximationWidth]
```

That is $`\lceil 2\cdot 1\cdot 1/(1/10)\rceil+1=21`; `norm_num` proves the arithmetic, including the
ceiling. Halving $`\varepsilon` approximately doubles the width, so this construction is
linear in $`1/\varepsilon`.

The extra one after the ceiling turns a weak inequality into a strict one. If the ratio inside
the ceiling is already an integer, choosing that integer alone would only give the mesh estimate
at the requested tolerance; adding one leaves room for the final `< ε`. The proof also knows the
width is positive, so division by its real-valued cast is legitimate. In the displayed type,
`↑(reluApproximationWidth L a b ε)` is exactly that cast from a natural count to a real denominator.
`norm_num` establishes an equality about this definition inside Lean's logic. It does not require
an executable implementation of arbitrary real inputs, which is why the proof succeeds even
though the preceding compilation attempt fails.

# Quadratic Approximation In Binary32

To construct parameters explicitly, use four hinges on $`[0,1]` for the target $`f(x)=x^2`.
The following code computes the knots and chord slopes in `Float`, converts the knots and slope
increments to FloatLib binary32, and prints the resulting parameters. The next example evaluates
their
hinge sum in binary32 arithmetic.

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

At $`x=3/8`, only the knots at zero and one quarter contribute. Their terms are
$`(1/4)(3/8)=3/32` and $`(1/2)(1/8)=1/16`, giving $`H(x)=5/32`.
The target is $`9/64`, so the difference is $`1/64`, the `0.015625` seen in the table.
This also explains the tensor layout in the PyTorch transcription below: each hidden coordinate
stores one distance from a knot, and the output weight on that coordinate stores a slope change.
The four coefficients are not four independent chord slopes. Copying the printed slope list into
the output layer would change the function, even though all layer dimensions would still match.

Now evaluate the network on a grid twice as fine as its own mesh, so that both the knots and the
points between them appear:

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
        (hingeFunIeee knot coeff 0 xe)).toFloat
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

The hinge sum agrees with $`x^2` at every knot because its slopes match the chords. Between knots
the real chord interpolant lies above the curve, since the target is convex. In the displayed
calculation, the overshoot is
$`0.015625` at every midpoint: the midpoint error of a
chord interpolant of this quadratic is $`\tfrac18 f''h^2=\tfrac18\cdot 2\cdot(1/4)^2`.

Compare that with what the rate theorem certifies at this width. Since $`x^2` is $`2`-Lipschitz on
$`[0,1]`, the formula reaches four hidden units only once $`\varepsilon` is at least $`4/3`:

```lean
-- Compare the prescribed widths for the same Lipschitz
-- bound at three tolerances.
example :
    reluApproximationWidth 2 0 1 (4 / 3) = 4 := by
  norm_num [reluApproximationWidth]

example :
    reluApproximationWidth 2 0 1 (1 / 10) = 41 := by
  norm_num [reluApproximationWidth]

example :
    reluApproximationWidth 2 0 1 (1 / 20) = 81 := by
  norm_num [reluApproximationWidth]
```

The rate theorem supplies a real network with error below $`4/3`; the displayed binary32
mesh has sampled error $`0.015625`. Applying the theorem to these particular parameters requires
identifying its real construction with this mesh, then bounding conversion and execution error.
The width calculation alone does not discharge those obligations. A sharper interpolation bound
can also use the second derivative of this smooth target, information absent from the Lipschitz
hypotheses. The sampled error and the existence guarantee therefore describe different claims.

The last two examples make the size cost concrete: halving $`\varepsilon` changes the prescribed
width from 41 to 81. This is an upper bound on the width needed by the construction. Proving
minimality would also require a lower bound that rules out smaller networks.

## Sine Approximation

Convexity explains the one-sided interpolation error for $`x^2` and the increasing chord slopes.
The hinge construction itself only needs the slopes and their differences. Apply it to eight
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
    (hingeFunIeee waveKnot waveCoeff 0
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

The coefficients change sign because the chord
slopes now decrease as well as increase, so a hinge can cancel the slope its predecessors added
rather than only steepening the curve. And the midpoint errors have both signs: the interpolant sits
below $`f` where $`f` is concave and above it where $`f` is convex, so no one-sided statement of the
kind that held for $`x^2` is available. The Lipschitz error bound controls the absolute error in
both cases.

The rate formula reacts to this target the way a worst-case bound has to. Since
$`\sin(2\pi x)` is $`2\pi`-Lipschitz, take $`L=7` as a rational upper bound and ask what eight
hidden units certify:

```lean
-- Use the rational Lipschitz upper bound seven in the same
-- width formula.
example : reluApproximationWidth 7 0 1 2 = 8 := by
  norm_num [reluApproximationWidth]
```

The formula assigns width eight to $`\varepsilon=2`. For this particular $`f`, whose range is
$`[-1,1]`, that tolerance is loose: even the constant zero function has error at most one.
The measured error of the displayed mesh is about $`0.07` on the sampled grid. The rate theorem
uses only the supplied Lipschitz constant, so it covers every
$`7`-Lipschitz target with the same width. It does not use the sine function's curvature, and the
grid measurement does not establish a uniform bound between samples.

The two signs in `H-f` are useful here. A positive number means the approximation lies above the
target; taking an absolute value too early would hide the change in curvature between the two
halves of the wave. The final `worst` is a maximum over seventeen evaluated inputs, whereas the
analytic error statement quantifies over an interval. To turn such a grid into a regional bound,
one would also need to control how the error varies between adjacent inputs. The Lipschitz
construction supplies that control through its hypotheses and mesh argument. The printed table
instead helps inspect whether the chosen slope increments behave as intended.

## ReLU Approximation In PyTorch

We can place the same knots and slope increments in two `torch.nn` linear layers. The first
layer uses all-ones weights and biases $`-t_i`; the second weights the resulting hinges:

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

On the same grid, the PyTorch network produces the nine `H` values printed above. The inputs,
parameters, intermediate products, and partial sums in this small example are all exactly
representable in binary32, so neither evaluation accumulates rounding error
({Informal.citep pytorch2019}[]).

The Lean table evaluates `hingeFunIeee`, the function used in the three-term executable
approximation theorem below. Applying that theorem still requires proofs of its
approximation, quantization, and finiteness premises for these parameters. A
transcription into another framework is a claim that has to be checked separately, which is exactly
what {ref "pytorch-roundtrip"}[the roundtrip chapter] is for.

## Rounding Error

In the quadratic example, the hinge sum agrees with the target at the knots down to the last bit.
The knots and chord slopes are multiples of $`1/4`; on this small grid, the products and partial
sums are also exactly representable in binary32. The following comparison subtracts the bit
patterns of the two positive outputs, exposing differences hidden by decimal formatting:

```lean (name := meshExact)
-- Compare stored encodings at interior dyadic knots, where
-- decimal output could hide a difference.
#eval do
  for k in [1, 2, 3] do
    let x := Nat.toFloat k * meshStep
    let xe :=
      (ExecFloat.Binary.ofFloat32 ∘ Float.toFloat32) x
    let hx := hingeFunIeee knot coeff 0 xe
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
    let hx := hingeFunIeee knot3 coeff3 0 xe
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
stored value and its decimal display discussed by Goldberg ({Informal.citep goldberg1991}[]).

This comparison includes several rounding steps: conversion of the input and coefficients,
evaluation of the hinge sum, and conversion of the reference target value. The theorems below
separate parameter conversion from evaluation error. They also require finite executable
intermediates so that each arithmetic step can be related to a real-valued expression.

# From Real Parameters To Binary32

The real theorem is only the first leg of a finite-precision result. Once knots, coefficients, and
the bias are stored in binary32, the total error naturally splits into

$$`\begin{aligned}
|f(x)-H_{\mathrm{IEEE}}(x)|
\leq{}&
|f(x)-H_{\mathbb R}(x)|\\
&+|H_{\mathbb R}(x)-H_{\mathrm{embedded}}(x)|\\
&+|H_{\mathrm{embedded}}(x)-H_{\mathrm{IEEE}}(x)|.
\end{aligned}`

These are, respectively:

1. approximation error of the real hinge network;
2. parameter-quantization or reference error;
3. rounded evaluation error.

The distinction is visible in
{src "NN/MLTheory/Proofs/Approximation/Universal/UniversalApproximationIEEE32Exec.lean"}[
the three-term approximation theorem]. It takes real hinge parameters `tR` and `cR`,
executable FloatLib binary32 parameters `t`, `c`, and `b0`, and separate assumptions for the real
approximation and quantization terms. It also requires finiteness witnesses for the intermediate
hinge sum and output. Its conclusion adds `hingeFunErrorBound` to the two supplied tolerances.

To use this theorem, conversion and evaluation must satisfy its finiteness assumptions. Its
dyadic specialization starts from dyadic parameters and uses a half-ULP
rounding bound, but it still asks for finite evaluation witnesses.

The full signature exposes the obligations needed to connect the real approximant to executable
arithmetic:

```lean (name := ieeeThree)
-- Inspect the separate approximation,
-- parameter-quantization, and finite-execution obligations.
#check @reluApproximationIccIEEE32Exec_threeTerm
```

```leanOutput ieeeThree (whitespace := lax)
@reluApproximationIccIEEE32Exec_threeTerm : ∀ {f : ℝ → ℝ} {a b : ℝ} {hidDim : ℕ} (tR cR : Fin hidDim
  → ℝ)
  (t c :
    Fin hidDim →
      ExecFloat.Binary 8 23 FloatFormat.Encoding.ieee (FloatFormat.Encoding.ieee.defaultBias 8)
        embed._proof_1
        embed._proof_2 embed._proof_3 embed._proof_4)
  (b0 :
    ExecFloat.Binary 8 23 FloatFormat.Encoding.ieee (FloatFormat.Encoding.ieee.defaultBias 8)
      embed._proof_1
      embed._proof_2 embed._proof_3 embed._proof_4),
  (∀ (i : Fin hidDim), ExecFloat.Binary.isFinite (t i) = true) →
    (∀ (i : Fin hidDim), ExecFloat.Binary.isFinite (c i) = true) →
      ∀ (εApprox εQ : ℝ),
        (∀
            (x :
              ExecFloat.Binary 8 23 FloatFormat.Encoding.ieee (FloatFormat.Encoding.ieee.defaultBias
                8) embed._proof_1
                embed._proof_2 embed._proof_3 embed._proof_4),
            ExecFloat.Binary.isFinite x = true →
              (ExecFloat.Binary.toModel x).toReal ∈ Set.Icc a b →
                HingeSumFinite t c x 0 (List.finRange hidDim) ∧
                  ExecFloat.Binary.isFinite (hingeFunIeee t c b0 x) = true) →
          (∀
              (x :
                ExecFloat.Binary 8 23 FloatFormat.Encoding.ieee
                  (FloatFormat.Encoding.ieee.defaultBias 8) embed._proof_1
                  embed._proof_2 embed._proof_3 embed._proof_4),
              ExecFloat.Binary.isFinite x = true →
                (ExecFloat.Binary.toModel x).toReal ∈ Set.Icc a b →
                  |f (ExecFloat.Binary.toModel x).toReal -
                        hingeFun hidDim tR cR (f a) (ExecFloat.Binary.toModel x).toReal| <
                    εApprox) →
            (∀
                (x :
                  ExecFloat.Binary 8 23 FloatFormat.Encoding.ieee
                    (FloatFormat.Encoding.ieee.defaultBias 8)
                    embed._proof_1 embed._proof_2 embed._proof_3 embed._proof_4),
                ExecFloat.Binary.isFinite x = true →
                  (ExecFloat.Binary.toModel x).toReal ∈ Set.Icc a b →
                    |hingeFun hidDim tR cR (f a) (ExecFloat.Binary.toModel x).toReal -
                          hingeFunReal (embedVec t) (embedVec c) (embed b0) (embed x)| ≤
                      εQ) →
              ∀
                (x :
                  ExecFloat.Binary 8 23 FloatFormat.Encoding.ieee
                    (FloatFormat.Encoding.ieee.defaultBias 8)
                    embed._proof_1 embed._proof_2 embed._proof_3 embed._proof_4),
                ExecFloat.Binary.isFinite x = true →
                  (ExecFloat.Binary.toModel x).toReal ∈ Set.Icc a b →
                    |f (ExecFloat.Binary.toModel x).toReal -
                          (ExecFloat.Binary.toModel (hingeFunIeee t c b0 x)).toReal| <
                      εApprox + εQ + hingeFunErrorBound (embedVec t) (embedVec c) (embed b0) (embed
                        x)
```

Read it from the top. The two parameter vectors come in pairs, `tR cR` over the reals and `t c` over
FloatLib binary32, because the theorem has to talk about both to relate them. The two `isFinite`
hypotheses on `t` and `c` rule out NaN and infinite parameters, which is a condition a real-valued
statement could not even express. The `HingeSumFinite` conjunct is the one to notice: it demands
that *every intermediate* of the fold stay finite, not merely the result. The proof uses finite
real embeddings at each arithmetic step; an overflowing intermediate breaks that argument. The next
two hypotheses are the
$`\varepsilon_{\mathrm{Approx}}` and $`\varepsilon_Q` budgets from the three-term split, supplied
rather than derived. And the conclusion adds `hingeFunErrorBound` to their sum, which is the
rounding of the evaluation itself.

For a particular checkpoint, each premise must be proved for its stored parameters and the stated
input domain. `instDecHingeSumFinite` makes the `HingeSumFinite` predicate decidable for a fixed
input and concrete parameters. When it holds, `decide` can prove that instance. A premise
quantified over all inputs in an interval still needs an argument covering that whole domain.

Notice that the final input is a FloatLib binary32, and interval membership is tested on its decoded
real value `Model.toReal (ExecFloat.Binary.toModel x)`. The target is therefore evaluated at the
real number represented by that
stored input. If an application starts with an ideal real measurement and rounds it before model
evaluation, its input-conversion error is an additional question. A Lipschitz bound on the target
can help relate those two inputs, but that step is not hidden inside `εQ`: the quantization premise
shown here compares two parameterizations at the same embedded input. Keeping that input fixed
makes the three terms correspond to three identifiable functions.

A companion theorem for dyadic parameters replaces the supplied
$`\varepsilon_Q` with a half-ULP bound derived from dyadic parameters. It is the more usable form
when reference parameters are given as dyadics. The finite-evaluation witnesses remain necessary
because finite stored parameters can still produce an overflowing intermediate.

For example, the output accumulation can overflow even if every weight, bias, and input was
converted successfully. The witness follows the intermediate operations of the hinge evaluation,
which is why it appears separately from the parameter-conversion estimate. Checking only the
stored parameter bits would leave that part of the theorem's hypothesis unaddressed.
The dyadic specialization exposes those witnesses in its signature:

```lean (name := ieeeHalfUlp)
-- Dyadic reference parameters supply the quantization
-- estimate; finite evaluation is still
-- required.
#check @reluApproximationIccIEEE32Exec_dyadicHalfUlp
```

TorchLean also has an intermediate
{src "NN/MLTheory/Proofs/Approximation/Universal/UniversalApproximationFP32.lean"}[`FP32` theorem].
`relu_universal_approximation_Icc_fp32` evaluates the hinge construction in the clean finite
rounding model and proves a pointwise bound of the form

$$`|f(x)-H_{\mathrm{FP32}}(x)|
  < \varepsilon+\operatorname{hingeFunErrorBound}(x).`

```lean (name := fp32Uat)
-- This bound concerns the rounded-real carrier and reads
-- its result through the val field.
#check @relu_universal_approximation_Icc_fp32
```

```leanOutput fp32Uat (whitespace := lax)
@relu_universal_approximation_Icc_fp32 : ∀ {f : ℝ → ℝ} {a b L : ℝ},
  a < b →
    0 < L →
      (∀ x ∈ Set.Icc a b, ∀ y ∈ Set.Icc a b, |f x - f y| ≤ L * |x - y|) →
        ∀ ε > 0,
          ∃ hidDim t c b0,
            ∀ x ∈ Set.Icc a b,
              |f x - (hingeFunFp32 t c b0 { val := x }).val| <
                ε + hingeFunErrorBound t c b0 { val := x }
```

Compare that conclusion with the real one printed earlier. The hypotheses are identical, the
existential now produces hinge parameters rather than a `LinearSpec` pair, and the bound has grown a
second summand. That summand bounds evaluation in this model. It does not include a separate
conversion of
arbitrary real parameters into finite binary32 storage.

`FP32` is convenient for error analysis because values are represented by reals rounded at
binary32 precision with gradual underflow and no upper exponent cutoff. FloatLib binary32 is the
explicit bit-level model with overflow, signed zero, subnormals, infinities, and NaNs. A proof in
the former is not silently promoted to the latter.

# Exact Finite Interval Images

Approximation theory also appears in a finite semantic form. The module
{src "NN/MLTheory/Proofs/Approximation/FloatInterval/Semantics.lean"}[`FloatInterval.Semantics`]
defines intervals of FloatLib binary32 values and computes abstract operations by enumerating the
finite
concrete image and taking its hull. For addition, the central theorem is:

```lean (name := addSound)
-- The conclusion is enclosure of the executable sum for
-- every pair of interval members.
#check @OpsExact.add_sound
```

```leanOutput addSound (whitespace := lax)
OpsExact.add_sound : ∀ (A B : I) {x y : F},
  x ∈ A → y ∈ B → ExecFloat.add x y ∈ OpsExact.addSharp A B
```

For any members `x` of `A` and `y` of `B`, their executable sum belongs to `addSharp A B`.
No separate rounding hypothesis appears, because the
abstract operator was built by enumerating the concrete image rather than by an interval formula
that would need one.

The multiplication and ReLU theorems have the same shape. They are then composed through an affine
layer and a two-layer ReLU network. Hover the two signatures to read the hypotheses in full:

```lean (name := evalSound)
-- Follow scalar enclosure through an affine layer and then
-- the two-layer network.
#check @aff_sound
#check @eval_sound
```

The four hypotheses on `eval_sound` are the `isNaN ... = false` conditions on the two weight
matrices and two bias vectors. They are stated per entry rather than as one predicate on the
network, which is slightly more verbose to supply and considerably easier to discharge from a
concrete checkpoint.

The weight and bias hypotheses rule out NaN parameters. The conclusion is about the explicit
FloatLib binary32 evaluation, not an ideal real network. Because binary32 is finite, the exact
abstract
operators can in principle enumerate every concrete pair in an interval. This gives us a reference
semantics, but its cost grows with the number of represented values. Enumerating those pairs is
expensive even when the interval has a compact description.

There is also a difference between an exact concrete image and its interval hull. For a set of
executable results with gaps, the smallest containing interval can include values the operation
never returns. `add_sound` promises membership in that containing interval; it does not assert
that every member is attained. This is the right direction for verification: excluding an unsafe
value from the enclosure excludes it from the concrete image as well. It is less useful for proving
that a value is attainable. The constant-target result later in the chapter has a particularly
simple image because every allowed input returns the same finite constant.

Run the proof modules directly:

```terminal
# Check the scalar interval semantics and its finite-image
# theorem as separate modules.
lake env lean \
  NN/MLTheory/Proofs/Approximation/FloatInterval/Semantics.lean

lake env lean \
  NN/MLTheory/Proofs/Approximation/FloatInterval/ExactImageTheorem.lean
```

A successful run is silent and exits with status zero. If you want the same three signatures in a
scratch file rather than on this page, the imports and namespaces are:

```
-- Open the scalar and composed-network namespaces to
-- inspect their enclosure interfaces.
import NN.MLTheory.Proofs.Approximation.FloatInterval.Semantics

open NN.MLTheory.Proofs.UniversalApproximation.FloatIntervalApprox
open NN.MLTheory.Proofs.UniversalApproximation.FloatIntervalApprox.TwoLayerMLPExact

#check OpsExact.add_sound
#check aff_sound
#check eval_sound
```

Note that `add_sound` lives in `OpsExact` while `aff_sound` and `eval_sound` live in
`TwoLayerMLPExact`; the split follows the file's own layering of scalar operators under composed
network semantics.

An application of `aff_sound` must discharge each `isNaN ... = false` premise.
A point interval containing NaN cannot be treated as an ordinary ordered singleton. Omitting
one of these premises leaves that part of the enclosure argument unproved.

# Higher-Dimensional Domains

The one-dimensional hinge proof is constructive and quantitative. The higher-dimensional
development uses a different route. In
{src "NN/MLTheory/Proofs/Approximation/Universal/StoneWeierstrass.lean"}[`StoneWeierstrass`],
`Tensor ℝ [n]` is identified with `Fin n → ℝ` by a homeomorphism. Coordinate functions generate a
subalgebra of continuous functions, and the proof establishes that this subalgebra separates
points: distinct tensors differ in at least one coordinate, and the corresponding coordinate
function distinguishes them. Mathlib's Stone-Weierstrass theorem {Informal.citep mathlib2020}[]
then supplies density on compact domains. The TorchLean-specific work establishes the
homeomorphism and checks the subalgebra hypotheses needed by that theorem.

Here the approximating objects are coordinate-polynomial expressions. A subalgebra contains
constants and is closed under addition, multiplication, and scalar multiplication, so starting
from coordinate projections permits expressions such as a coordinate squared or a product of two
coordinates. Point separation says this class can distinguish any two different tensor inputs.
Compactness is what lets density be stated using uniform error over the whole domain. The source
conclusion `∃ g : coordSubalg, ‖(g : C(K, ℝ)) - f‖ < ε` chooses one such continuous function;
`C(K, ℝ)` bundles continuity with the function, and its norm measures the largest absolute error.
A later neural construction still has to represent or approximate that polynomial expression.

That topological argument answers a broad representation question, but it does not produce the
same explicit width formula as the one-dimensional Lipschitz construction. The two proofs are
complementary:

- the hinge mesh exposes parameters and a rate on $`[a,b]`;
- the coordinate-subalgebra proof handles compact multidimensional domains at a more abstract
  level.

# Approximation And Enclosure Guarantees

The scalar hinge construction also has a theorem for the corresponding two-layer network. The
{src "NN/MLTheory/Proofs/Approximation/Universal/UniversalApproximationIEEE32ExecTwoLayerMlp.lean"}[
two-layer binary32 approximation API] packages representation, parameter-rounding, and
executable-rounding error in `relu_twoLayerMlp_ieee32exec_threeTerm`. For finite interval semantics,
the
{src "NN/MLTheory/Proofs/Approximation/FloatInterval/ConstantTarget.lean"}[constant-target API]
proves `exactIntervalImage_constant`: a finite constant has the exact singleton interval image. The
latter is a useful base case for certificate composition, not a claim that an arbitrary nonconstant
network has an exact interval image.

Both are checked as this page builds; hover to read them:

```lean (name := endpoints)
-- Compare the three-budget network theorem with the exact
-- constant-image base case.
#check @relu_twoLayerMlp_ieee32exec_threeTerm
#check @exactIntervalImage_constant
```

After these layers are composed, one can make a statement with all errors visible:

$$`\text{target error}
\leq
\text{representation error}
+\text{parameter rounding error}
+\text{execution error}.`

A CROWN or interval certificate can then add a fourth component: a sound enclosure over an input
region for the chosen network. Each term comes from a different argument: model construction,
parameter conversion, runtime arithmetic, and regional verification. Writing the sum explicitly
lets an application decide where to spend its error budget.

The constructions follow the classical universal-approximation tradition.
{Informal.citet cybenko1989}[] proved density for superpositions of a sigmoidal function, and
{Informal.citet hornik1991}[] removed the sigmoid-specific hypothesis for multilayer feedforward
networks. Both are density results: they say an approximant exists and are silent about its size.
The explicit ReLU construction obtains a width formula by choosing a mesh and computing its
slope increments. The finite interval development instead encloses the outputs of a fixed
executable binary32 network by abstract interpretation.
