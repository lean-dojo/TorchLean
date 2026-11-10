import VersoManual
import TorchLean.Tensor.Conversion
import TorchLeanExamples.Verification.ODE.AST
import TorchLeanExamples.Verification.ODE.Verify
import TorchLeanExamples.Verification.PINN.Certificate
import TorchLeanExamples.Verification.PINN
import TorchLeanExamples.Verification.PINN.DatasetCheck
import TorchLeanExamples.Verification.PINN.PDEAST
import TorchLeanExamples.Verification.PINN.ResidualAffine
import TorchLeanExamples.Verification.Splines.PiecewisePolyCert
import TorchLeanBlueprint.Bib
import TorchLeanBlueprint.Roles

open Verso.Genre Manual
open Verso.Genre.Manual.InlineLean

-- The three scientific checkers live in three sibling namespaces under `TorchLean.Examples`, and
-- two of them define their own `Expr`, `Expr.eval`, and interval helpers. Opening all of them at
-- once would make `Expr` ambiguous, so we open the shared prefix and let the section names
-- disambiguate: `ODE.Expr` is the right-hand-side language, `PINN.Expr` is the residual language.
open TorchLean.Examples TorchLean
open PINN (referenceArch finiteDifferenceResidual)
open PINN (Prims)
open Splines Splines.Certificate

-- A few signatures below print wider than this file's 100-column limit, so their `leanOutput`
-- blocks ask for `whitespace := lax` and are wrapped in the source. The rendered page still shows
-- each message exactly as Lean printed it.
open Lean.Elab.Tactic.GuardMsgs.WhitespaceMode (lax)

#doc (Manual) "Scientific ML Verification" =>
%%%
tag := "scientific-ml-verification"
%%%

Scientific models need guarantees between the points where they are evaluated. A trained PINN
may look accurate on a plot while violating its PDE between sample points. A numerical ODE
trajectory may look smooth while accumulated error takes it outside the claimed corridor.
A spline fit may be excellent at the knots and deviate from the target inside one interval.
TorchLean's scientific
checkers examine finite certificates: proposed corridors, residual intervals, or polynomial pieces
whose conditions can be recomputed.

All three maintained paths share a producer-and-checker shape, but they check different things. For
candidate corridor networks, the ODE command tests interval subsolution, supersolution, ordering,
and initial-value conditions. On the PINN side, the bundled command replays residual and derivative
bounds for a fixed in-source graph and parameter set. The spline command checks that rational
polynomial pieces interpolate their declared knots exactly. It doesn't bound the polynomial between
those knots.

For classifier verification the artifact is often an input box and logit bounds, while for
scientific ML it may be a time corridor, a residual bound, a polynomial certificate, or a
derivative enclosure. In each case a producer proposes a finite object and Lean checks a smaller
predicate. Where a mathematical enclosure theorem exists, a separate soundness bridge must still
show that the executable predicate supplies its hypotheses.

# Scientific Verification Commands

You can list the registered verification tools with:

```terminal
# List the registered tools before selecting a scientific
# certificate checker.
scripts/lake.sh exe verify -- list
```

Three entries cover the scientific workflows described here (shown without the default artifact
paths that the real listing prints after each description):

```
pinn-cert [<path>]    -- PINN certificate recomputation check
spline-cert [<path>]  -- piecewise-polynomial certificate checker (optional Julia regen)
ode                   -- ODE enclosure verification (sub/super NN bounds)
```

Start with the bundled PINN artifact:

```terminal
# Replay the bundled PINN value and residual bounds at its
# declared sample boxes.
scripts/lake.sh exe verify -- pinn-cert
```

The certificate declares three sample points, so the command prints one block per point. Its
`residual_bounds_deriv` field stores intervals of approximately $`[-25.6479,25.6479]` at all three
points. An interval this wide doesn't establish a small PDE error. Later we trace its width through
the value and derivative calculations. TorchLean's checker reconstructs the bounds from the
in-source graph and compares them with the JSON using the fixed absolute tolerance
`certTol = 1e-5`. This replay doesn't say that a small residual implies closeness to the true PDE
solution: that requires a separate stability or a posteriori error theorem for the PDE.

This check has two numerical scales: the stored residual half-width (about $`25.6`) describes the
candidate enclosure being replayed, while the tolerance $`10^{-5}` says how closely two stored
endpoint numbers must match the recomputation. A small replay tolerance doesn't make the residual
small: it can faithfully reproduce a very wide interval. If a later theorem required
$`|R(x)|\le0.01`, a sound enclosure would have to fit inside $`[-0.01,0.01]`. Containing zero would
only mean that the interval doesn't exclude a zero residual, and it wouldn't bound every possible
residual near zero.

The spline sample is shorter:

```terminal
# Check exact rational interpolation of the bundled
# polynomial pieces.
scripts/lake.sh exe verify -- spline-cert
```

Acceptance means that the knot coordinates are strictly increasing, each piece names the matching
adjacent knots, every coefficient array has the declared length, and Horner evaluation at both
endpoints equals the declared knot values over exact rationals. If you change one coefficient and
rerun the checker, it should report an endpoint mismatch. In the spline section we build a
certificate in Lean and watch that message appear.

Two flags expose useful neighboring checks:

```terminal
# Request binary32 endpoint replay or regenerate the
# artifact before checking it.
scripts/lake.sh exe verify -- spline-cert --arithmetic ieee
scripts/lake.sh exe verify -- spline-cert --regen
```

`--arithmetic ieee` also requires every rational value to be exactly representable as finite
binary32, and it replays the endpoint equalities with FloatLib's binary32 arithmetic. Exact storage
of the inputs alone doesn't guarantee exact intermediate arithmetic, so the checker evaluates the
polynomial at both endpoints in that format and compares the results with the declared values.

`--regen` asks the Julia producer to write a fresh JSON document before Lean checks it. Neither flag
proves an interior range bound for a polynomial piece.

Without an equation or certificate, the ODE tool prints its two accepted modes (the certificate
placeholder is spelled out and the trailing option list omitted here):

```terminal
# With no equation or certificate, the ODE command displays
# its accepted modes.
scripts/lake.sh exe verify -- ode
```

```
Usage:
  scripts/lake.sh exe verify -- ode [--model=direct|torchlean]
    [--arithmetic=native|ieee] --cert=<certificate JSON>
  scripts/lake.sh exe verify -- ode [--model=direct|torchlean]
    [--arithmetic=native|ieee] --rhs="<expr>" --t0=<float>
    --t1=<float> --init=<float> --lower=<wL.json> --upper=<wU.json>
```

The bundled certificate is `TorchLeanExamples/Verification/ODE/sample_ode_cert.json`. Where the
checker gets its settings depends on the mode. In certificate mode the ODE expression, the time
segments, the initial interval, the two corridor networks, and the search settings all come from
the JSON. In inline mode they come from the command line, and only then do `--maxDepth`,
`--minWidth`, `--slack`, and `--verbose` mean anything. Passing one of those alongside `--cert` is
rejected rather than ignored:

```terminal
# Certificate mode rejects an inline search-setting
# override.
scripts/lake.sh exe verify -- ode \
  --cert=TorchLeanExamples/Verification/ODE/sample_ode_cert.json \
  --verbose=true
```

```terminal +output
error: unexpected arguments: [--verbose=true]
```

Rejecting the override keeps the search settings tied to the artifact. Increasing `slack` changes
the inequalities that pass. Changing `maxDepth` changes how far the checker can subdivide a failed
box. Recording those settings in the JSON makes the result reproducible without a separate list of
command-line overrides.

`--model` controls how the lower and upper corridor networks are evaluated: directly from the
imported graph, or after lowering through TorchLean. `--arithmetic` controls the arithmetic used by
the direct evaluator. When the command omits either switch, the certificate's declared setting is
used, while an inline verification defaults to direct and native. The TorchLean-lowered route
supports only `--arithmetic=native`, and pairing it with `ieee` is rejected rather than silently
changing the requested semantics.

Certificate times and initial endpoints must be finite and correctly ordered, and `minWidth` and
`slack` must be finite and nonnegative. Unknown backend names or wrong JSON field types are
rejected (they aren't replaced by defaults). During checking, a NaN or infinity in any interval
comparison counts as a failure, not as a successful unordered comparison.

A segment certificate is most useful when it records a cover of the intended time interval. Each
accepted segment then contributes a local obligation, and the coverage argument explains why no
time has been left unchecked. Refining a segment can tighten interval estimates by reducing the
range of each variable, but it adds no information about an omitted segment. Likewise, the initial
interval is data for all permitted starting values (not one sampled initial state). Keeping these
two roles separate helps explain why the checker reports the initial condition before its segment
results.

Expression parsing is part of that boundary. In both ODE and PINN expressions, exponentiation binds
more tightly than unary minus, so `-u^2` means `-(u^2)` (write `(-u)^2` for the other tree).
Natural powers use the ordinary identities $`u^0=1` and $`u^1=u`. Because the parser consumes the
whole input and rejects unknown identifiers or trailing tokens, a certificate can't be checked
against a silently shortened equation.

# Interval Arithmetic

Both the ODE and PINN commands evaluate expressions on intervals. A few small calculations show how
these bounds gain or lose precision. On the ODE side, the right-hand-side language and its interval
evaluator live in the {src "TorchLeanExamples/Verification/ODE/AST.lean"}[ODE AST module]:

```lean (name := odeAst)
-- Inspect the expression type, scalar-generic evaluator,
-- and command entrypoint.
#check ODE.Expr
#check @ODE.Expr.eval
#check @ODE.main
```

```leanOutput odeAst
TorchLean.Examples.ODE.Expr : Type
```

```leanOutput odeAst (whitespace := lax)
@ODE.Expr.eval : {α : Type} → [inst : Storage α] →
  [inst_1 : Context α] → [CROWN.BoundOps α] →
  [CROWN.NonlinearBoundOps α] →
  [FromFloat α] →
  ODE.Env α → ODE.Expr → Option (α × α)
```

```leanOutput odeAst
ODE.main : List String → IO Unit
```

`eval` is generic in the scalar `α`, so the same expression can be evaluated in host `Float` or in
FloatLib binary32, which is what makes `--arithmetic=native|ieee` a change of semantics rather than
a change of code. Its instances name the arithmetic: `BoundOps` rounds each sum, difference and
product outward, `NonlinearBoundOps` bounds division, `sin`, `cos`, `exp` and `log`, and `FromFloat`
converts constants. The decimal `0.1` has no binary64 value, so the parser keeps its binary64
roundings down and up as the constant, and `eval` converts them to `α` downward and upward. The
result is `Option (α × α)` because a rule can refuse: host `Float` has no `exp` or `log` bound, and
an infinite or NaN endpoint stops the evaluation. The caller must handle that failure before it can
compare interval endpoints.

So the printed type describes both the arithmetic and the possible failure. `ODE.Env α` supplies
intervals for time and state, while `ODE.Expr` describes which operations combine them. A result
`some (lo, hi)` means this evaluator produced two endpoints, and `none` means it didn't produce an
interval. There's no proof field in that pair. An enclosure theorem would need to connect the
endpoints to every real input represented by the environment. The directed constant conversion is
part of that connection: each constant's interval contains the real number it stands for, so the
right-hand side being bounded is the one the certificate names.

For multiplication, pairing lower endpoints and pairing upper endpoints isn't enough, because the
extreme values can come from any of four corner combinations:

```lean (name := cornerMul)
-- The minimum product uses endpoints from opposite sides of
-- the two intervals.
#eval CROWN.Graph.intervalMul
  (-1.0 : Float) 2.0 (-3.0) 1.0
```

```leanOutput cornerMul
(-6.000000, 3.000000)
```

In the output, the lower endpoint `-6` is $`2\cdot(-3)` (from the upper end of the first interval
and the lower end of the second). Neither $`(-1)\cdot(-3)=3` nor $`2\cdot 1=2` is the answer, which
is why {src "TorchLean/Verify/CROWN/FlatBox/Transfer.lean"}[`Graph.intervalMul`], the product rule
of both the ODE and the PINN evaluators, computes all four products, rounds each one outward and
takes their minimum and maximum. In host `Float` the printed endpoints are the binary64 neighbours
just outside $`-6` and $`3`.

Even exact interval operations can overestimate a compound expression. Take
the logistic right-hand side $`f(t,u)=u(1-u)`, written in the checker's own AST:

```lean
-- Keep both occurrences of the state variable visible in
-- the logistic expression.
def logistic : ODE.Expr :=
  .mul .u (.sub (.const 1.0 1.0) .u)

def unitBox : ODE.Env Float :=
  { t := (0.0, 1.0), u := (0.0, 0.5) }
```

```lean (name := logisticEval)
-- Evaluate the logistic expression on the selected state
-- interval.
#eval ODE.Expr.eval unitBox logistic
```

```leanOutput logisticEval
some (-0.000000, 0.500000)
```

On $`u\in[0,\tfrac12]` the true range of $`u(1-u)` is $`[0,\tfrac14]`, attained at the right
endpoint. But the evaluator returned $`[-2^{-1074},\tfrac12+2^{-52}]`, twice as wide. The printed
`-0.000000` is the smallest negative subnormal: host `Float` widens each result by one adjacent
binary64 value, and the exact corner product $`0` becomes $`-2^{-1074}`. The slack is there because
`.mul` sees two independent intervals. It has no way to know that the `u` on the left and the `u`
on the right are the same number, so it also allows $`u=\tfrac12` on the left while $`1-u=1` on the
right. Those two choices can't occur together. This loss of shared-variable information is called
the dependency problem.

The exact scalar range follows from the derivative $`1-2u` (nonnegative on $`[0,1/2]`), so the
logistic expression increases from zero to one quarter throughout this box. In contrast, the
interval product maximizes over the rectangle $`[0,1/2]\times[1/2,1]` (where the two coordinates
can vary independently), while the true values lie only on the line $`(u,1-u)` inside that
rectangle. This geometric view identifies the lost information: the arithmetic rules have enlarged
the set of possible pairs before taking its range. It also explains why extra floating precision
alone can't remove this particular factor-of-two overestimate.

Rewriting the expression can reduce that loss. Completing the square gives:

$$`u(1-u)=\tfrac14-\left(u-\tfrac12\right)^2.`

```lean
-- Shift the state before multiplication to reduce
-- dependency on this one-sided box.
def logisticShifted : ODE.Expr :=
  .sub (.const 0.25 0.25)
    (.mul (.sub .u (.const 0.5 0.5))
      (.sub .u (.const 0.5 0.5)))
```

```lean (name := shiftedEval)
-- Evaluate the equivalent real polynomial through a
-- different interval expression.
#eval ODE.Expr.eval unitBox logisticShifted
```

```leanOutput shiftedEval
some (-0.000000, 0.250000)
```

With the shift, the result matches the true range on the chosen box up to rounding, from about
$`-1.7\times10^{-16}` to $`\tfrac14+2^{-54}`. The AST multiplies two copies of the shifted variable
here too, but both copies range from minus one half to zero, and the four corner products give the
square range on that one-sided interval, widened by the outward rounding. Rewriting has improved
this calculation without giving the evaluator a general rule for recognizing repeated variables.

Both expressions agree over the reals, while their interval evaluations differ. A failed tube check
may therefore succeed after an algebraically justified rewrite of the right-hand side, but such a
rewrite also needs care under floating arithmetic, where algebraic identities need not preserve
evaluation results.

For the reciprocal of an interval straddling zero, no finite enclosure exists, so the evaluator
declines:

```lean (name := divEval)
-- A denominator interval crossing zero cannot produce a
-- finite reciprocal enclosure.
#eval ODE.Expr.eval
  ({ t := (0.0, 1.0), u := ((-1.0), 1.0) } : ODE.Env Float)
  (.div (.const 1.0 1.0) .u)
```

```leanOutput divEval
none
```

An extended-interval implementation could represent an unbounded range, but this checker requires
finite endpoints. Returning only the two endpoint quotients across a zero-crossing denominator
would miss the unbounded behavior. Returning `none` lets the caller report the failed expression.

The transcendental cases go through `NonlinearBoundOps`, and there the two arithmetics differ. Host
`Float` has no proved bound for its `sin` library call, so it answers with the global range
$`[-1,1]` even on a narrow interval:

```lean (name := sinNarrow)
-- Host Float bounds sine by its global range, even on a
-- narrow interval.
#eval ODE.Expr.eval
  ({ t := (0.0, 0.0), u := (0.0, 0.1) } : ODE.Env Float)
  (.sin .u)
```

```leanOutput sinNarrow
some (-1.000000, 1.000000)
```

FloatLib binary32 has an enclosure theorem for its sine bound,
{src "TorchLean/Verify/CROWN/BoundOps/Binary32.lean"}[`sinBounds_sound`], so the same
interval keeps its information:

```lean (name := sinBinary)
-- FloatLib binary32 bounds sine on the same interval.
#eval ODE.Expr.eval
  (α := FloatLib.Floats.ExecFloat.Binary 8 23)
  { t := (0, 0), u := (0, 0.1) } (.sin .u)
```

```leanOutput sinBinary
some (-2.0830732e-5, 0.09997918)
```

For that narrow interval the true range is $`[0,\sin 0.1]\approx[0,0.0998]` , and the printed
interval extends about $`2.1\times10^{-5}` below zero and $`1.5\times10^{-4}` above $`\sin(0.1)`. On
a wide interval the same bound degenerates to the global range:

```lean (name := sinWide)
-- A wide state interval reduces the sine result to its
-- global range.
#eval ODE.Expr.eval
  (α := FloatLib.Floats.ExecFloat.Binary 8 23)
  { t := (0, 0), u := (0, 6.3) } (.sin .u)
```

```leanOutput sinWide
some (-1, 1)
```

On that wide input the evaluator returns the global sine bound, which gives no additional
information from this particular input interval. The narrower example retains more information.
Subdividing time can help the ODE verifier in the same way, by applying its bound rules to smaller
boxes.

# ODE Enclosures

An ODE enclosure certificate is a finite description of a corridor around a trajectory. Checking it
doesn't depend on an external integrator's explanation of the run: the checker parses the ODE
expression, runs its interval-shaped endpoint calculation over each segment, and tests the
candidate tube's subsolution, supersolution, ordering, and initial-value conditions. That
calculation is an executable screening condition. A theorem about the true trajectory still needs
the outward-bound and real-analysis links described below.

On the mathematical side we have an ODE

$$`\dot x(t)=f(t,x(t)).`

A corridor certificate supplies lower and upper functions $`u_L(t),u_U(t)`. In Lean, the
real-analysis theorem works on a nonnegative time interval starting at zero. It assumes continuity
of the solution and both walls on the closed interval, right derivatives before the final time,
wall ordering throughout, initial inclusion, and

$$`u_L'(t)\le f(t,u_L(t)),\qquad f(t,u_U(t))\le u_U'(t).`

Its solution hypothesis is specifically the clamped dynamics
$`u'(t)=f(t,\max(u_L(t),\min(u_U(t),u(t))))`. The theorem proves that this solution remains between
the walls. Clamping then becomes a no-op, so the solution also solves the original ODE. Identifying
an independently supplied original-ODE solution requires suitable additional evidence (such as
uniqueness). A derivative-range inclusion for a box alone is not this comparison theorem.

The derivative inequalities express how a wall behaves at a possible crossing. At the upper wall,
the vector field points no faster upward than the wall itself moves, and at the lower wall it
points no faster downward. In the enclosure source, the proof applies a one-dimensional fencing
argument with a small strict perturbation of each wall, then removes that perturbation. Continuity
and the right-derivative assumptions make that argument available throughout the interval. They
can't be replaced by checking a few plotted slopes. Once enclosure is proved, evaluating the clamp
at the solution returns the solution itself, which is the step that recovers the original
differential equation along this trajectory.

The executable side is exposed through the
{src "TorchLeanExamples/Verification/ODE/Verify.lean"}[ODE checker API].
The core pieces are the expression AST, the interval evaluator, the segment certificate, and the
final checker result.

We can check the bundled sample's corridor by hand: it is the constant zero function on $`[0,1]`,
the right-hand side is the constant `0`, and both the lower and the upper corridor network are a
single linear layer with zero weight and zero bias. (The metadata says `tanh`, but this one-layer
architecture inserts no hidden activation.)

```terminal
# Check the constant zero corridor using the certificate’s
# declared arithmetic.
scripts/lake.sh exe verify -- ode \
  --cert=TorchLeanExamples/Verification/ODE/sample_ode_cert.json
```

Even for this constant example, the tube has to contain the initial interval, the lower corridor's
derivative has to stay below $`f`, the upper corridor's derivative has to stay above it, and the
two corridors have to stay ordered. Mathematically, all four conditions hold with zero slack. That
lets the next run isolate the effect of changing arithmetic.

Since the certificate declares `"arithmetic": "ieee"`, override that one field to compare the
executable checks:

```terminal
# Changing arithmetic exposes the strict initial endpoint
# comparison.
scripts/lake.sh exe verify -- ode --arithmetic=native \
  --cert=TorchLeanExamples/Verification/ODE/sample_ode_cert.json
```

Read the initial interval carefully. Here the check asks for the upper endpoint of the lower
corridor's value at $`t_0` to be at most `init.lo = 0`. A narrow enclosure straddling zero would
fail that comparison, even though the mathematical corridor is valid. Such a rejection would be a
false negative. Changing arithmetic can change these endpoint calculations, so compare the reported
endpoints before attributing a different result to the ODE itself.

For each wall, the initial comparison uses the conservative endpoint. To establish that every
possible lower-wall value is below the initial lower bound, its *upper* endpoint must be below that
bound, and dually the *lower* endpoint of the upper-wall enclosure must exceed the initial upper
bound. That's stronger than merely asking whether the wall intervals overlap the initial interval.
In the recorded native rejection, six-decimal printing hid the positive amount in the lower wall's
upper endpoint. So the failed comparison is compatible with the displayed zeros, and the diagnostic
needs to be read as an endpoint test, not as exact equality of the rendered text.

Even for a mathematically valid tube, the arithmetic setting affects acceptance. Every endpoint the
evaluator computes is rounded outward: FloatLib binary32 rounds each operation in the needed
direction, and host `Float` widens each correctly rounded sum, product and quotient by one adjacent
value and bounds `sin` and `cos` by $`[-1,1]`. Excess width can therefore cause false negatives. No
theorem about `runCertificate` states that the computed intervals contain the exact values, which
remains part of the missing connection between the checker and the real-valued theorem.

Time boxes are seeded from their endpoints, which avoids an extra center/radius rounding roundtrip.
Recursive subdivision uses an overflow-resistant midpoint and stops if no representable interior
midpoint remains (rejection diagnostics distinguish this limit from the configured maximum depth
and minimum width). Partitioning doesn't change the configured inequalities, and an initial-time
failure is a point check that later subdivision can't repair.

The `slack` setting does change the segment inequalities: `checkSub`, `checkSuper`, and
`checkOrder` each add it to the right-hand side of a comparison, with the sum rounded down. But the
real enclosure theorem has no such tolerance, so a positive-slack success needs an additional
argument recovering the exact inequalities required by the theorem. Of the initial checks, the two
containment checks use no slack, while the ordering test between the walls does.

The theorem side is the real mathematical statement. In the
{src "TorchLeanExamples/Verification/ODE/Enclosure.lean"}[ODE enclosure API], a corridor theorem
says, in plain language:

> A continuous solution of the clamped dynamics stays between continuous subsolution and
> supersolution walls under the stated right-derivative, initial-value, and ordering hypotheses.

The companion module
{src "TorchLeanExamples/Verification/ODE/EnclosureBackends.lean"}[ODE enclosure backends] proves
that a directly decoded trajectory of any configured binary format that is continuous on an
interval is constant there. Sampled values reach the real statement only through a real-valued
interpolant, whose hypotheses must be proved separately.

Lean has a local real enclosure theorem, and the executable checker computes the kinds of corridor
inequalities that theorem consumes. But no theorem says that a successful `runCertificate` call
supplies all of the real-analysis hypotheses of that enclosure theorem. Continuity, derivative
agreement, interval soundness, the treatment of slack, and any finite-to-real interpretation still
have to be connected, and broader neural ODE and integrator claims need their own enclosure
conditions and agreement evidence as well.

So the trusted boundary is:

```
external integrator/search -> proposed tube JSON
Lean parser/checker        -> interval side conditions for the tube
ODE theorem                -> statement about true trajectories, if theorem hypotheses match
runtime bridge             -> needed for a claim about a concrete finite-precision integrator
```

# PINN Certificates

A uniform PINN guarantee needs several pieces of evidence:

1. the imported parameters match the architecture
2. the PDE expression is the one being checked
3. the residual is bounded over the domain
4. boundary or dataset constraints are respected

TorchLean gives each piece a small object. The
{src "TorchLeanExamples/Verification/PINN/Architecture.lean"}[PINN architecture API] names
sequential network records and graph construction, and the
{src "TorchLeanExamples/Verification/PINN/PDEAST.lean"}[PDE expression API] names the PDE language.
A checkpoint's graph comes from the
{src "TorchLeanExamples/Verification/PINN/PyTorch/Model.lean"}[PyTorch model API], which builds it
from the checkpoint's loaded layers instead of letting a raw tensor dictionary float around
unchecked. The CROWN bounds on the network output `u(x)` are in the
{src "TorchLeanExamples/Verification/PINN/ResidualAffine.lean"}[residual affine API], and the box
splitters used by the command live in
{src "TorchLeanExamples/Verification/PINN/CLI.lean"}[the PINN CLI].

For the bundled network, the architecture record and graph builder expose its shape:

```lean (name := pinnArch)
-- Inspect the architecture, the graph builder, the
-- reference layers, and dataset options.
#check PINN.Architecture
#check @PINN.referenceModel
#check @PINN.referenceLayers
#check @PINN.DatasetOptions
```

```leanOutput pinnArch
TorchLean.Examples.PINN.Architecture : Type
```

```leanOutput pinnArch (whitespace := lax)
@PINN.referenceModel : {α : Type} → [inst : Storage α] →
  [Context α] → ℕ → Except String (IR.Graph × IR.Payload α)
```

```leanOutput pinnArch (whitespace := lax)
@PINN.referenceLayers : {α : Type} → [inst : Storage α] →
  [Context α] → ℕ → Array (SomeLinear α)
```

```leanOutput pinnArch
PINN.DatasetOptions : Type
```

`referenceLayers` is generic in the scalar type for the same reason `ODE.Expr.eval` is: the
identical demonstration weights have to be available as `Float` for the bundled replay and as
FloatLib binary32
when a check wants bit-level semantics. Asking the architecture for its layer dimensions gives the
`1 -> 16 -> 16 -> 1` network the certificate describes:

```lean (name := archDims)
-- Compare affine layer dimensions with the number of
-- nodes of the graph.
#eval (referenceArch 1).linearDims
#eval (PINN.referenceModel (α := Float) 1).map
  (·.1.nodes.size)
```

```leanOutput archDims
#[(1, 16), (16, 16), (16, 1)]
```

```leanOutput archDims
Except.ok 12
```

The graph has `12` nodes: after the input, each linear layer is three nodes (its weight and bias
constants and `Op.linear`), with a tanh node between consecutive layers, and its output is the last
`Op.linear` (node `11`). By design, the bundled `pinn-cert` path is smaller than the full target
sketched above. Its graph is that fixed tanh network `referenceModel 1`, with the deterministic
parameters `referenceLayers 1` (both defined in Lean), and the JSON supplies sample points, box
radii, a PDE expression, value intervals, and two kinds of residual interval. `Certificate.verify`
recomputes values and derivatives with the Float bound implementation, on the graph and its
tangent graphs (`Derivatives.ofGraph`), then compares the stored value and residual intervals
using `certTol`.
Individual derivative intervals are printed, but they aren't separate artifact fields checked for
equality. The command returns success or an error. It doesn't construct a proof object for a
uniform residual proposition.

The dimensions output describes three affine maps, while the node count describes their placement
in a graph that also holds their parameters and nonlinear operations. These are different counts.
One coordinate goes in, each hidden affine map produces sixteen coordinates, and the final map
produces the scalar field value. Computing a PDE residual requires derivatives with respect to that
input coordinate (not gradients with respect to the network's trainable weights). Here the layers'
parameters select the particular field being checked, and the PDE expression selects which of its
input derivatives enter the residual.

## Residual Interval Width

For a network with tanh hidden activations, the bundled certificate stores a derivative-based
residual enclosure of approximately $`[-25.6,25.6]`. Each hidden activation lies in $`[-1,1]`, and
the final affine layer has positive weights summing to $`2.8` (matching the certificate's `u`
bounds $`[-2.8,2.8]`). Starting from those value intervals lets us follow the width through two
different residual calculations. On this Float path, the tanh rule returns the whole range `[-1,1]`
regardless of its input interval, and its derivative rules also use global bounds. Shrinking this
input box alone therefore doesn't tighten these particular enclosures.

The certificate carries *two* residual fields. One of them, `residual_bounds`, comes from the
three-point second difference

$$`\frac{u(x-h)-2u(x)+u(x+h)}{h^2},`

with $`h=0.01`. Interval arithmetic evaluates that numerator as if the three values were
independent, so a $`\pm2.8` enclosure at each of the three points gives a numerator with half-width
$`4\times2.8=11.2`, and then the division by $`h^2=10^{-4}` multiplies the width by ten thousand:

```lean (name := fdResidual)
-- Independent value boxes lose the cancellation in the
-- second-difference numerator.
#eval finiteDifferenceResidual
  { lo := -2.8, hi := 2.8 }
  { lo := -2.8, hi := 2.8 }
  { lo := -2.8, hi := 2.8 } 0.01
```

```leanOutput fdResidual
{ lo := -112000.000000, hi := 112000.000000 }
```

To the printed digits and within `certTol`, the computed interval agrees with the bundled JSON's
`residual_bounds` (±112000.00000000017). Behind it is one line of arithmetic over the reals:

```lean
-- Check the half-width amplification using exact real
-- arithmetic.
example :
    (2.8 + 2.8 + 2.8 + 2.8) / ((1 : ℝ) / 100 * (1 / 100))
      = 112000 := by
  norm_num
```

For an exact smooth function, neighboring values are highly related: a Taylor expansion cancels the
constant and first-order terms in $`u(x-h)-2u(x)+u(x+h)`, leaving the second derivative times
$`h^2` plus a remainder. Independent intervals discard the relation needed for that cancellation.
Their numerator uncertainty need not shrink with `h`, so division amplifies it. That's a different
source of error from the Taylor remainder. Reducing the spacing addresses the remainder only after
enough regularity has been established, and it may make the independent-box estimate much worse. So
the two residual fields compare different computations, not two interchangeable estimates of the
same accuracy.

Propagating derivative bounds through the graph avoids this difference quotient. That's what the
`residual_bounds_deriv` field records and what the printed `u''` lines report. For the same network
whose value bounds are approximately $`\pm2.8`, the artifact stores approximately $`\pm25.6479` for
the second derivative, compared with $`\pm112000` for the difference quotient (a factor of about
four thousand). Because the bundled PDE string is `uxx`, the printed residual is the
derivative-based number. The signature of the finite-difference helper says where the two fields
differ:

```lean (name := fdSig)
-- The finite-difference helper receives three value
-- intervals and a spacing.
#check @finiteDifferenceResidual
```

```leanOutput fdSig (whitespace := lax)
finiteDifferenceResidual : FloatLib.Numerics.Interval Float → FloatLib.Numerics.Interval Float →
  FloatLib.Numerics.Interval Float → Float → FloatLib.Numerics.Interval Float
```

That helper consumes three *value* enclosures and a spacing. It receives no information about how
the values vary together along the network. Both residual fields are checked, making the loss from
independent stencil intervals visible alongside the direct derivative calculation. Relating the
difference quotient to the derivative would also require a truncation-error bound.

## The Residual Language

Residuals are written in a small expression language and evaluated on the primitive enclosures that
the graph propagation produced:

```lean (name := pdeSig)
-- Distinguish residual expression evaluation from the
-- graph’s value-bound helper.
#check PINN.Expr
#check @PINN.Expr.eval
#check @PINN.crownUBoundsNodewise
```

```leanOutput pdeSig
TorchLean.Examples.PINN.Expr : Type
```

```leanOutput pdeSig
PINN.Expr.eval : Prims → PINN.Expr → Option (Float × Float)
```

```leanOutput pdeSig (whitespace := lax)
PINN.crownUBoundsNodewise : IR.Graph →
  IR.Payload Float →
    CROWN.FlatBox Float → Array (Option (CROWN.FlatBox Float)) →
      Option (Float × Float)
```

`Prims` is a record of five optional intervals: the value, the two first partials, and the two
second partials. In the bundled one-dimensional checker, three of them are filled and the two
$`y`-axis entries stay empty, as written here:

```lean
-- Supply the one-dimensional artifact’s values and
-- derivatives, and leave Y absent.
def artifactPrims : Prims where
  u := some ((-2.8), 2.8)
  duX := some ((-0.0), 4.488)
  duY := none
  d2uX := some ((-25.6479), 25.6479)
  d2uY := none
```

In that record, the value and second-derivative bounds are the artifact's endpoints rounded to the
displayed decimals. Evaluating the PDE string `uxx` selects the supplied second-derivative
interval:

```lean (name := pdeX)
-- The uxx residual directly selects the available second X
-- derivative.
#eval PINN.Expr.eval artifactPrims (.d2u .x)
```

```leanOutput pdeX
some (-25.647900, 25.647900)
```

Ask the same primitives for a $`y`-derivative (a two-dimensional PDE string would need one):

```lean (name := pdeY)
-- Asking for a second Y derivative fails because no such
-- interval was supplied.
#eval PINN.Expr.eval artifactPrims (.d2u .y)
```

```leanOutput pdeY
none
```

Here `eval` returns `none` because the requested primitive is absent. In that case the checker
reports `PINN PDE '...' evaluation failed because required primitives are missing`. Supplying zero
instead would change the equation being checked.

Addition is endpointwise and rounded outward, so a Poisson-style residual $`u_{xx}+u` adds the
two enclosures:

```lean (name := pdeSum)
-- Add the value and second-derivative intervals to bound
-- the selected residual.
#eval PINN.Expr.eval artifactPrims (.add (.d2u .x) .u)
```

```leanOutput pdeSum
some (-28.447900, 28.447900)
```

Optional fields let one-dimensional and two-dimensional callers share the residual language.
An absent derivative is not a zero derivative: supplying zero would assert that the field has
no curvature in that direction.

Products use the four-corner rule of the ODE side, `Graph.intervalMul`, so the two factors of
`u * u` are independent intervals:

```lean (name := pdeSquare)
-- The two factors of u * u are independent intervals.
#eval PINN.Expr.eval artifactPrims (.mul .u .u)
```

```leanOutput pdeSquare
some (-7.840000, 7.840000)
```

On $`u\in[-2.8,2.8]` the square $`u^2` ranges over $`[0,7.84]`, but the product rule also pairs
$`u=-2.8` on the left with $`u=2.8` on the right, which gives the lower endpoint $`-7.84`. This is
the dependency problem of the logistic example. The parser reads `u^2` as the same product, so a
squared term keeps the lower endpoint $`-7.84` in this evaluator.

## Uniform Residual And Boundary Bounds

For a Burgers-style residual, the mathematical claim has the shape:

$$`R_\theta(t,x)
=
\partial_t u_\theta(t,x)
+u_\theta(t,x)\partial_x u_\theta(t,x)
-\nu\partial_{xx}u_\theta(t,x).`

The certificate target is a uniform bound over the domain:

$$`\forall (t,x)\in\Omega,\qquad |R_\theta(t,x)|\le\varepsilon.`

Boundary or data conditions have the same form:

$$`\forall z\in\partial\Omega,\qquad |u_\theta(z)-g(z)|\le\varepsilon_b.`

The
{src "TorchLeanExamples/Verification/PINN/Certificate.lean"}[PINN certificate API], the
{src "TorchLeanExamples/Verification/PINN/DatasetCheck.lean"}[dataset checker API], and the
{src "TorchLeanExamples/Verification/PINN/CLI.lean"}[PINN command API]
are the user-facing pieces for that path.

A PINN claim names the PDE residual, domain, boundary data, and parameters as well as the model
architecture, and TorchLean records these inputs across its PINN tools. `pinn-cli` explores one-
and two-dimensional residual boxes with IBP or CROWN-style methods, while `pinn-dataset-check`
performs pointwise interval containment checks and can load an optional PyTorch parameter file. By
default the dataset command is report-only: it prints `ok` and `bad` counts but exits successfully
even when misses are present. Use

```terminal
# Make dataset misses fail the command instead of appearing
# only in its report.
scripts/lake.sh exe verify -- pinn-dataset-check \
  --dataset=TorchLeanExamples/Verification/PINN/sample_dataset_1d.json --strict
```

when a nonzero `bad` count should fail an automated run (replace the bundled sample path with your
dataset). Even strict success is still a checker result until a soundness theorem connects the
selected bound path and imported parameters to a quantified PDE statement.

For the application, the reference point is the physics-informed neural network of
{Informal.citet pinn2019}[]. That paper motivates the residual objective. TorchLean's bundled
replay makes a narrower claim: for its fixed graph and parameters, the artifact's stored value and
two residual intervals agree (within the checker's comparison tolerance) with the Float quantities
recomputed at the declared boxes. That replay also prints the first- and second-derivative
intervals. The dataset and interactive commands have their own inputs and checks, and they
shouldn't be folded into the meaning of `pinn-cert`.

Boundary conditions answer a question that the residual cannot settle by itself. For the simple
equation $`u''=0`, every affine function has zero residual, so the residual alone cannot select
which affine function is the intended solution. Prescribed endpoint values can distinguish them.
For a general PDE, the implication from residual and boundary error to solution error depends on
the equation and its domain. This motivates keeping the boundary tolerance $`\varepsilon_b`
separate from the residual tolerance $`\varepsilon`: they enter different hypotheses of a
possible stability estimate, even when the same network and parameter file occur in both checks.

## PDE Residuals In PyTorch

A PINN training loop can compute a residual in PyTorch {Informal.citep pytorch2019}[] by
differentiating the network twice with respect to its input. With `create_graph=True`, the first
gradient retains the graph needed for the second differentiation
{Informal.citep baydin2018}[]:

```
# Retain the first derivative’s graph so the second input
# derivative can be computed.
xy = torch.rand(4096, 2, requires_grad=True)
u = model(xy)
g = torch.autograd.grad(u.sum(), xy, create_graph=True)[0]
uxx = torch.autograd.grad(g[:, 0].sum(), xy,
                          create_graph=True)[0][:, 0]
loss = ((uxx + u.squeeze()) ** 2).mean()
```

For a smooth model that processes each batch row independently, these sums compute the needed
per-point derivatives (a model that couples rows would require a different Jacobian calculation).
`loss` is a mean over 4096 sampled points, and replacing `.mean()` with `.max()` would give the
sample maximum, which is still a statement about those 4096 points and not about the domain.
Between them the residual is unconstrained. A network can be trained to drive the sampled residual
to $`10^{-6}` while violating the equation elsewhere. Nothing in the sampled loss alone rules this
out.

The intended box claim quantifies over every input in the box. Our Lean checker recomputes
candidate bounds. But the missing soundness bridge must justify that quantifier before these Float
bounds become a theorem about the residual. A sampled residual of $`10^{-6}` and a candidate
interval bound of $`25.6` could describe the same network: one measures selected inputs, while the
other may lose substantial information in propagation. Direct derivative bounds, subdivision, and
affine relaxations offer different ways to reduce that loss, but each selected path still needs a
soundness argument.

# Piecewise Polynomial and Spline Certificates

The spline path is concentrated in
{src "TorchLeanExamples/Verification/Splines/PiecewisePolyCert.lean"}[PiecewisePolyCert API], which
parses `piecewise_poly_v0` JSON, evaluates polynomial pieces by Horner's rule, checks exact
rational interpolation at adjacent knots, and also has a FloatLib binary32 exact conversion path.

A piecewise polynomial certificate names intervals $`I_i` and polynomial pieces

$$`p_i(x)=\sum_k a_{ik}(x-x_i)^k,\qquad x\in I_i.`

For a piece on $`I_i=[x_i,x_{i+1}]`, the checked equations are

$$`p_i(x_i)=y_i,
\qquad p_i(x_{i+1})=y_{i+1}.`

Behind that are two Lean objects, a generic Horner evaluation and a checker over exact rationals:

```lean (name := splineSigs)
-- Horner evaluation is scalar-generic, while the
-- certificate checker uses exact rationals.
#check @evalPolyHorner
#check @checkRat
```

```leanOutput splineSigs (whitespace := lax)
@evalPolyHorner : {α : Type} → [Zero α] → [Add α] → [Mul α] →
  Array α → α → α
```

```leanOutput splineSigs
checkRat : Certificate → IO Unit
```

`evalPolyHorner` asks only for `Zero`, `Add`, and `Mul`, which is what lets the same evaluation run
over `Rat` for exact checking and over FloatLib binary32 for the `--arithmetic ieee` pass. The
checker returns `IO Unit` and throws on the first mismatch. Successful direct evaluation prints
nothing, while a failed endpoint comparison names the piece and the two unequal values. Both direct
checkers validate the knot partition and piece metadata before evaluation (including at least two
knots and exactly one piece per adjacent pair), and constructing a certificate in Lean doesn't
bypass those checks.

Horner's rule reads the coefficient array from highest degree to lowest: for $`[a_0,a_1,a_2]`, it
forms $`(a_2t+a_1)t+a_0`, so the stored array is in ascending power order even though evaluation
folds from the right. A piece's local coordinate also matters: at its left endpoint it is zero, and
at the right endpoint it is `hi - lo`. Feeding the absolute right knot into this evaluator would
describe another polynomial whenever `lo` is nonzero, so the checker computes this translation
before comparing endpoint values, tying each piece's coefficients to its declared interval.

Matching endpoint values can support continuity when adjacent pieces share the same knot value, but
it doesn't also match their derivatives. For example, the bump below has derivative `1` at its left
endpoint and `-1` at its right endpoint, so joining it to a zero polynomial with the same endpoint
value produces a derivative jump. A certificate for a smooth surrogate would need to check the
appropriate derivative equalities at each join as well. This is especially relevant when that
surrogate is later differentiated to form an ODE or PDE residual: interpolation data alone doesn't
determine the residual at the joins.

An interior range theorem would instead need a statement such as

$$`\forall x\in I_i,\qquad p_i(x)\in[\ell_i,u_i],`

together with data or a proof sufficient to check it. That stronger condition isn't part of
`piecewise_poly_v0`. A passing certificate can illustrate the distinction. On the single interval
$`[0,1]` the coefficients $`[0,1,-1]` mean $`p(t)=t-t^2`, because pieces use the local coordinate
$`t=x-\mathrm{lo}`:

```lean
-- Ascending coefficients describe the bump t - t² in the
-- piece’s local coordinate.
def bumpCoeffs : Array ℚ := #[0, 1, -1]
```

```lean (name := bumpEnds)
-- Both endpoints vanish even though the polynomial is
-- nonzero inside the interval.
#eval evalPolyHorner bumpCoeffs (0 : ℚ)
#eval evalPolyHorner bumpCoeffs (1 : ℚ)
```

```leanOutput bumpEnds
0
```

```leanOutput bumpEnds
0
```

Both knot values are zero, so a certificate declaring knots at $`x=0,1` with values $`y=0,0` is
accepted. Calling the checker directly confirms the endpoint equalities without printing a message:

```lean (name := bumpCheck)
-- Construct the complete one-piece certificate and check
-- its endpoint equations.
def bumpCert : Certificate where
  degree := 2
  n := 2
  xs := TorchLean.Tensor.from #[(0 : ℚ), 1]
  ys := TorchLean.Tensor.from #[(0 : ℚ), 0]
  pieces := #[{ lo := 0, hi := 1, coeffs := bumpCoeffs }]

#eval checkRat bumpCert
```

At the midpoint, the same polynomial takes a nonzero value:

```lean (name := bumpMid)
-- Evaluate the same polynomial at the interior point
-- omitted by the endpoint checks.
#eval evalPolyHorner bumpCoeffs (1 / 2 : ℚ)
```

```leanOutput bumpMid
1 / 4
```

Every checked endpoint equation holds exactly, while the midpoint value differs from the knot
values by $`\tfrac14`. The certificate declares this polynomial. Nothing in it says the polynomial
is zero between the knots. An interior bound would be an additional proposition to check.

For this bump, completing the square gives $`t-t^2=1/4-(t-1/2)^2`. Together with $`0\le t\le1`,
that identity explains the entire interior range $`[0,1/4]` (additional mathematics beyond the two
checks at zero and one). Multiplying the bump by any positive rational would still leave both
endpoints zero while changing its midpoint height, so exact interpolation data alone can't recover
a bound on interior deviation from the zero line. A scientific surrogate using the piece between
knots needs the polynomial itself and a range argument, not only agreement with its training knots.

Changing the right knot value while retaining the polynomial produces an endpoint mismatch:

```lean
-- Change only the claimed knot values while retaining the
-- polynomial coefficients.
def tamperedYs : Array ℚ := #[0, 1]

def tamperedCert : Certificate :=
  { bumpCert with
    ys := TorchLean.Tensor.from tamperedYs }
```

```lean (name := tamperedCheck) +error
-- The checker reports the exact right-endpoint mismatch in
-- the tampered certificate.
#eval checkRat tamperedCert
```

```leanOutput tamperedCheck
endpoint mismatch at i=0: p(hi)=0 ≠ yNext=1
```

No tolerance appears anywhere in that comparison. Since the certificate is parsed into `Rat`,
evaluated in `Rat`, and compared with `==` on `Rat`, a passing check is an exact algebraic
identity, not a floating-point coincidence. On top of that, the `--arithmetic ieee` pass adds a
second, stricter question: every rational in the document must be exactly representable in
binary32, and the endpoint equalities must hold again under the executable reference semantics. A
certificate can pass the rational check and fail that one, and the failure is informative, because
either a stored value isn't exactly representable or rounded Horner evaluation fails an endpoint
equality even though all input constants are representable.

An external spline fitter can produce these same coefficients and knot data, and the exact rational
check then validates interpolation independently of the fitter's numerical procedure. It says
nothing further about approximation error, derivative continuity at knots, or interior ranges.

The representability test and the rounded evaluation test address different questions. A value such
as $`1/2` can be represented exactly in binary32, whereas $`1/3` can't. But exact input
coefficients don't imply exact products and sums throughout Horner's rule, since intermediate
results can require more significand bits or exceed the format's finite range. So the IEEE pass
checks both the conversion of the document and the endpoint computations under the reference
operations. On its own, the rational pass remains useful: it identifies the polynomial the artifact
describes without mixing that algebraic question with one execution format.
