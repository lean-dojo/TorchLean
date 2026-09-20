import VersoManual
import NN.Tensor.Conversion
import NN.Verification.ODE.Ast
import NN.Verification.ODE.Verify
import NN.Verification.PINN.Certificate
import NN.Verification.PINN.Core
import NN.Verification.PINN.DatasetCheck
import NN.Verification.PINN.PdeAst
import NN.Verification.PINN.ResidualAffine
import NN.Verification.Splines.PiecewisePolyCert
import TorchLeanBlueprint.Bib
import TorchLeanBlueprint.Roles

open Verso.Genre Manual
open Verso.Genre.Manual.InlineLean

-- The three scientific checkers live in three sibling namespaces under `NN.Verification`, and one
-- of them defines its own `Expr`, `eval`, and interval helpers. Opening all of them at once would
-- make `Expr` ambiguous, so we open the shared prefix and let the section names disambiguate:
-- `ODE.Expr` is the right-hand-side language, `PINN.PdeAst.Expr` is the residual language.
open NN.Verification
open NN.Verification.PINN (referenceArch finiteDifferenceResidual)
open NN.Verification.PINN.PdeAst (Prims)
open NN.Verification.Splines.PiecewisePolyCert

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

The three maintained paths share a producer-and-checker shape, but they check different things. The
ODE command tests interval subsolution, supersolution, ordering, and initial-value conditions for
candidate corridor networks. The bundled PINN command replays residual and derivative bounds for a
fixed in-source graph and parameter set. The spline command checks that rational polynomial pieces
interpolate their declared knots exactly; it does not yet bound the polynomial between those knots.

For classifier verification, the artifact is often an input box and logit bounds. For scientific ML,
the artifact may be a time corridor, a residual bound, a polynomial certificate, or a derivative
enclosure. In each case a producer proposes a finite object and Lean checks a smaller predicate.
Where a mathematical enclosure theorem exists, a separate soundness bridge must still show that
the executable predicate supplies its hypotheses.

# Scientific Verification Commands

The registered verification tools can be listed with:

```terminal
# List the registered tools before selecting a scientific
# certificate checker.
lake exe verify -- list
```

Three entries cover the scientific workflows described here, shown without the default artifact
paths that the real listing prints after each description:

```
pinn-cert [<path>]    -- PINN certificate recomputation check
spline-cert [<path>]  -- piecewise-polynomial certificate checker
ode                   -- ODE enclosure verification (sub/super NN bounds)
```

Start with the bundled PINN artifact:

```terminal
# Replay the bundled PINN value and residual bounds at its
# declared sample boxes.
lake exe verify -- pinn-cert
```

The certificate declares three sample points, so the command prints one block per point. This is
the whole output:

```terminal +output
Residual R(x) from PDE 'uxx': [-25.647900,25.647900]
u'(x-h)∈[-0.000000,4.488000], u'(x)∈[-0.000000,4.488000], u'(x+h)∈[-0.000000,4.488000]
u''(x-h)∈[-25.647900,25.647900], u''(x)∈[-25.647900,25.647900], u''(x+h)∈[-25.647900,25.647900]
Residual R(x) from PDE 'uxx': [-25.647900,25.647900]
u'(x-h)∈[-0.000000,4.488000], u'(x)∈[-0.000000,4.488000], u'(x+h)∈[-0.000000,4.488000]
u''(x-h)∈[-25.647900,25.647900], u''(x)∈[-25.647900,25.647900], u''(x+h)∈[-25.647900,25.647900]
Residual R(x) from PDE 'uxx': [-25.647900,25.647900]
u'(x-h)∈[-0.000000,4.488000], u'(x)∈[-0.000000,4.488000], u'(x+h)∈[-0.000000,4.488000]
u''(x-h)∈[-25.647900,25.647900], u''(x)∈[-25.647900,25.647900], u''(x+h)∈[-25.647900,25.647900]
PINN artifact replay matched Lean's recomputed residual bounds.
```

Every block is identical because the bound propagation gives the same intervals at these sample
boxes. A residual interval this wide does not establish a small PDE error; later we trace its width
through the value and derivative calculations. The checker reconstructs the bounds from the
in-source graph and compares them with the JSON using the fixed absolute tolerance
`certTol = 1e-5`. This replay does not say that a small residual implies closeness to the true
PDE solution; that requires a separate stability or a posteriori error theorem for the PDE.

There are two numerical scales in this report. The residual half-width, about $`25.6`, describes
the candidate enclosure being replayed. The tolerance $`10^{-5}` describes how closely two stored
endpoint numbers must match the recomputation. A small replay tolerance does not make the residual
small: it can faithfully reproduce a very wide interval. If a later theorem required
$`|R(x)|\le0.01`, a sound enclosure would have to fit inside $`[-0.01,0.01]`. Containing zero
would only mean that the interval does not exclude a zero residual; it would not bound every
possible residual near zero.

The spline sample is shorter:

```terminal
# Check exact rational interpolation of the bundled
# polynomial pieces.
lake exe verify -- spline-cert
```

and prints:

```terminal +output
Piecewise polynomial certificate verified.
```

Here "verified" means that the knot coordinates are strictly increasing, each piece names the
matching adjacent knots, every coefficient array has the declared length, and Horner evaluation at
both endpoints equals the declared knot values over exact rationals. Change one coefficient and
rerun the checker; an endpoint mismatch should be reported. Later in this chapter we build a
certificate in Lean and watch that message appear.

Two flags expose useful neighboring checks:

```terminal
# Request binary32 endpoint replay or regenerate the
# artifact before checking it.
lake exe verify -- spline-cert --arithmetic ieee
lake exe verify -- spline-cert --regen
```

`--arithmetic ieee` additionally requires every rational value to be exactly representable as finite
binary32 and replays the endpoint equalities with FloatLib's binary32 arithmetic, adding one
line:

The following transcript predates the FloatLib migration and retains its recorded scalar labels
and numerical results. Current `.ieee` execution uses FloatLib binary32.

```terminal +output
Piecewise polynomial certificate verified.
IEEE32Exec semantics check verified (exact representability + endpoint equalities).
```

`--regen` asks the Julia producer to write a fresh JSON document before Lean checks it. Neither flag
proves an interior range bound for a polynomial piece.

Without an equation or certificate, the ODE tool prints its two accepted modes (the certificate
placeholder is spelled out here for readability):

```terminal
# With no equation or certificate, the ODE command displays
# its accepted modes.
lake exe verify -- ode
```

```
Usage:
  lake exe verify -- ode [--model=direct|torchlean]
    [--arithmetic=native|ieee] --cert=<certificate JSON>
  lake exe verify -- ode [--model=direct|torchlean]
    [--arithmetic=native|ieee] --rhs="<expr>" --t0=<float>
    --t1=<float> --init=<float> --lower=<wL.json> --upper=<wU.json>
```

The bundled certificate is `NN/Examples/Verification/ODE/sample_ode_cert.json`.
The mode determines where the checker gets its settings. In certificate mode the ODE expression,
the time segments, the initial interval, the two
corridor networks, and the search settings all come from the JSON. In inline mode they come from the
command line, and only then do `--maxDepth`, `--minWidth`, `--slack`, and `--verbose` mean anything.
Passing one of those alongside `--cert` is rejected rather than ignored:

```terminal
# Certificate mode rejects an inline search-setting
# override.
lake exe verify -- ode \
  --cert=NN/Examples/Verification/ODE/sample_ode_cert.json \
  --verbose=true
```

```terminal +output
error: unexpected arguments: [--verbose=true]
```

This keeps the search settings tied to the artifact. Increasing `slack` changes the inequalities
that pass; changing `maxDepth` changes how far the checker can subdivide a failed box. Recording
those settings in the JSON makes the result reproducible without a separate list of command-line
overrides.

The `--model` choice controls how the lower and upper corridor networks are evaluated: directly from
the imported graph, or after lowering through TorchLean. The `--arithmetic` choice controls the
arithmetic used by the direct evaluator. When the command omits either switch, the certificate's
declared setting is used; an inline verification defaults to direct and native. Today the
TorchLean-lowered route supports only `--arithmetic=native`, and pairing it with `ieee` is rejected
rather than silently changing the requested semantics.

Certificate times and initial endpoints must be finite and correctly ordered; `minWidth` and
`slack` must be finite and nonnegative. Unknown backend names or wrong JSON field types are rejected
instead of being replaced by defaults. During checking, a NaN or infinity in any interval
comparison is a failure, not a successful unordered comparison.

A segment certificate is most useful when it records a cover of the intended time interval.
Each accepted segment then contributes a local obligation, and the coverage argument explains
why no time has been left unchecked. Refining a segment can make interval estimates tighter by
reducing the range of each variable, but it does not add information about an omitted segment.
Likewise, the initial interval is data for all permitted starting values, rather than one sampled
initial state. Keeping these two roles separate helps explain why the checker reports the
initial condition before its segment results.

Expression parsing is part of that boundary. In both ODE and PINN expressions, exponentiation
binds more tightly than unary minus, so `-u^2` means `-(u^2)`; write `(-u)^2` for the other tree.
Natural powers use the ordinary identities $`u^0=1` and $`u^1=u`. The parser consumes the whole
input and rejects unknown identifiers or trailing tokens, preventing a certificate from being
checked against a silently shortened equation.

# Interval Arithmetic

The ODE and PINN commands evaluate expressions on intervals. A few small calculations show how
these bounds gain or lose precision. The ODE side keeps its right-hand-side language and its
interval evaluator in the
{src "NN/Verification/ODE/Ast.lean"}[ODE AST module]:

```lean (name := odeAst)
-- Inspect the expression type, scalar-generic evaluator,
-- and command entrypoint.
#check ODE.Expr
#check @ODE.eval
#check @ODE.Verify.main
```

```leanOutput odeAst
NN.Verification.ODE.Expr : Type
```

```leanOutput odeAst (whitespace := lax)
@ODE.eval : {α : Type} → [TorchLean.Storage α] → [Context α] →
  (Float → α) → ODE.Env α → ODE.Expr → Option (α × α)
```

```leanOutput odeAst
ODE.Verify.main : List String → IO Unit
```

`eval` is generic in the scalar `α`, so the same
expression can be evaluated in host `Float` or in FloatLib binary32, which is what makes
`--arithmetic=native|ieee` a change of semantics rather than a change of code. It takes an explicit
`ofFloat` rather than assuming a coercion, because the literals in a certificate arrive as decimal
`Float` values and someone has to say how they enter the chosen carrier. And it returns
`Option (α × α)`: a selected interval rule can reject an expression or domain. The caller must
handle that failure before it can compare interval endpoints.

The printed type describes both the arithmetic and the possible failure. `ODE.Env α` supplies
intervals for time and state, while `ODE.Expr` describes which operations combine them. A result
`some (lo, hi)` means this evaluator produced two endpoints; `none` means it did not produce an
interval. There is no proof field in that pair. An enclosure theorem would need to connect the
endpoints to every real input represented by the environment. The explicit literal conversion
`Float → α` is part of that connection, since changing the interpretation of constants can
change the right-hand side being bounded.

For multiplication, pairing lower endpoints and pairing upper endpoints is insufficient: the
extreme values can come from any of four corner
combinations:

```lean (name := ivalMul)
-- The minimum product uses endpoints from opposite sides of
-- the two intervals.
#eval ODE.Ival.mul ((-1.0 : Float), (2.0 : Float))
  ((-3.0 : Float), (1.0 : Float))
```

```leanOutput ivalMul
(-6.000000, 3.000000)
```

The lower endpoint `-6` is $`2\cdot(-3)`, from the upper end of the first interval and the lower end
of the second. Neither $`(-1)\cdot(-3)=3` nor $`2\cdot 1=2` is the answer, which is why
{src "NN/Verification/ODE/Ast.lean"}[`Ival.mul`] computes all four products and takes their minimum
and maximum.

Even exact interval operations can overestimate a compound expression. Take
the logistic right-hand side $`f(t,u)=u(1-u)`, written in the checker's own AST:

```lean
-- Keep both occurrences of the state variable visible in
-- the logistic expression.
def logistic : ODE.Expr :=
  .mul .u (.sub (.const 1.0) .u)

def unitBox : ODE.Env Float :=
  { t := (0.0, 1.0), u := (0.0, 0.5) }
```

```lean (name := logisticEval)
-- Evaluate the logistic expression on the selected state
-- interval.
#eval ODE.eval id unitBox logistic
```

```leanOutput logisticEval
some (0.000000, 0.500000)
```

On $`u\in[0,\tfrac12]` the true range of $`u(1-u)` is $`[0,\tfrac14]`, attained at the right
endpoint. The evaluator returned $`[0,\tfrac12]` , twice as wide. For these exactly represented
arithmetic values, the returned interval contains the true range.
That observation is not a general outward-rounding theorem for the floating evaluator. The reason
for the slack is that `.mul` sees two
independent intervals. It has no way to know that the `u` on the left and the `u` on the right are
the same number, so it also allows $`u=\tfrac12` on the left while $`1-u=1` on the right. Those
two choices cannot occur together. This loss of shared-variable information is called the
dependency problem.

The exact scalar range follows from the derivative $`1-2u`, which is nonnegative on
$`[0,1/2]`. Thus the logistic expression increases from zero to one quarter throughout this
box. In contrast, the interval product maximizes over the rectangle
$`[0,1/2]\times[1/2,1]`, where the two coordinates can vary independently. The true values
lie only on the line $`(u,1-u)` inside that rectangle. This geometric view identifies the
lost information: the arithmetic rules have enlarged the set of possible pairs before taking
its range. It also explains why extra floating precision alone cannot remove this particular
factor-of-two overestimate.

Rewriting the expression can reduce that loss. Completing the square gives:

$$`u(1-u)=\tfrac14-\left(u-\tfrac12\right)^2.`

```lean
-- Shift the state before multiplication to reduce
-- dependency on this one-sided box.
def logisticShifted : ODE.Expr :=
  .sub (.const 0.25)
    (.mul (.sub .u (.const 0.5)) (.sub .u (.const 0.5)))
```

```lean (name := shiftedEval)
-- Evaluate the equivalent real polynomial through a
-- different interval expression.
#eval ODE.eval id unitBox logisticShifted
```

```leanOutput shiftedEval
some (0.000000, 0.250000)
```

This matches the true range on the chosen box. The AST still multiplies two copies of the shifted
variable, but both copies now range from minus one half to zero. The four corner products give
the exact square range on that one-sided interval. Rewriting has improved this calculation
without giving the evaluator a general rule for recognizing repeated variables.

The two expressions agree over the reals, while their interval evaluations differ. A failed tube
check may therefore succeed after an algebraically justified rewrite of the right-hand side.
Such a rewrite also needs care under floating arithmetic, where algebraic identities need not
preserve evaluation results.

For the reciprocal of an interval straddling zero, no finite enclosure exists, so the evaluator
declines:

```lean (name := divEval)
-- A denominator interval crossing zero cannot produce a
-- finite reciprocal enclosure.
#eval ODE.eval id
  ({ t := (0.0, 1.0), u := ((-1.0), 1.0) } : ODE.Env Float)
  (.div (.const 1.0) .u)
```

```leanOutput divEval
none
```

An extended-interval implementation could represent an unbounded range, but this checker requires
finite endpoints. Returning only the two endpoint quotients across a zero-crossing denominator
would miss the unbounded behavior. Returning `none` lets the caller report the failed expression.

The transcendental cases use analytic bounds rather than computing exact ranges. `sin` uses a
one-Lipschitz box around the interval midpoint, clamped to $`[-1,1]`. On a narrow interval this
gives:

```lean (name := sinNarrow)
-- A narrow state interval retains information in the
-- midpoint sine bound.
#eval ODE.eval id
  ({ t := (0.0, 0.0), u := (0.0, 0.1) } : ODE.Env Float)
  (.sin .u)
```

```leanOutput sinNarrow
some (-0.000021, 0.099979)
```

The true range is $`[0,\sin 0.1]\approx[0,0.0998]` , and the printed interval extends about
$`2.1\times10^{-5}` below zero and
$`1.5\times10^{-4}` above $`\sin(0.1)`. The Lipschitz rule is justified over the reals; these
rounded endpoint calculations still need an outward-error argument for a general soundness claim. On
a wide interval the same rule degenerates to the clamp:

```lean (name := sinWide)
-- A wide state interval reduces the sine result to its
-- global range.
#eval ODE.eval id
  ({ t := (0.0, 0.0), u := (0.0, 6.3) } : ODE.Env Float)
  (.sin .u)
```

```leanOutput sinWide
some (-1.000000, 1.000000)
```

This is the global sine bound; it gives no additional information from this particular input
interval. The narrower example retains more information. Subdividing time can help the ODE
verifier in the same way, by applying its bound rules to smaller boxes.

# ODE Enclosures

An ODE enclosure certificate is a finite description of a corridor around a trajectory. The checker
does not depend on an external integrator's explanation of the run. It parses the ODE expression,
runs its interval-shaped endpoint calculation over each segment, and tests the candidate tube's
subsolution, supersolution, ordering, and initial-value conditions. That calculation is an
executable screening condition; a theorem about the true trajectory still needs the outward-bound
and real-analysis links described below.

The mathematical object is an ODE

$$`\dot x(t)=f(t,x(t)).`

A corridor certificate supplies lower and upper functions $`u_L(t),u_U(t)`. The implemented
real-analysis theorem works on a nonnegative time interval starting at zero. It assumes continuity
of the solution and both walls on the closed interval, right derivatives before the final time,
wall ordering throughout, initial inclusion, and

$$`u_L'(t)\le f(t,u_L(t)),\qquad f(t,u_U(t))\le u_U'(t).`

Its solution hypothesis is specifically the clamped dynamics
$`u'(t)=f(t,\max(u_L(t),\min(u_U(t),u(t))))`. The theorem proves that this solution remains
between the walls; clamping then becomes a no-op, so it also solves the original ODE. Identifying
an independently supplied original-ODE solution requires suitable additional evidence, such as
uniqueness. A derivative-range inclusion for a box alone is not this comparison theorem.

The derivative inequalities express how a wall behaves at a possible crossing. At the upper wall,
the vector field points no faster upward than the wall itself moves; at the lower wall, it points
no faster downward. The proof in the enclosure source applies a one-dimensional fencing argument
with a small strict perturbation of each wall, then removes that perturbation. Continuity and the
right-derivative assumptions make that argument available throughout the interval. They cannot be
replaced by checking a few plotted slopes. Once enclosure is proved, evaluating the clamp at the
solution returns the solution itself, which is the step that recovers the original differential
equation along this trajectory.

The executable side is exposed through the
[ODE checker API](https://github.com/lean-dojo/TorchLean/blob/main/NN/Verification/ODE/Verify.lean).
The core pieces are the expression AST, the interval evaluator, the segment certificate, and the
final checker result.

The bundled sample lets us check the corridor by hand: it is the constant
zero function on $`[0,1]`, the right-hand side is the constant `0`, and both the lower and the upper
corridor network are a single tanh layer with zero weight and zero bias.

```terminal
# Check the constant zero corridor using the certificate’s
# declared arithmetic.
lake exe verify -- ode \
  --cert=NN/Examples/Verification/ODE/sample_ode_cert.json
```

```terminal +output
[ODE] initial OK at t0=0.000000
[ODE] certificate verified: all segments succeeded.
```

Even for this constant example, the tube has to contain the initial interval, the lower
corridor's derivative has to stay below $`f`, the upper corridor's derivative has to stay above it,
and the two corridors have to stay ordered. All four conditions hold with zero slack. This makes
it possible to isolate the effect of changing arithmetic in the next run.

The certificate declares `"arithmetic": "ieee"`. Override that one field and the same file is
rejected:

```terminal
# Changing arithmetic exposes the strict initial endpoint
# comparison.
lake exe verify -- ode --arithmetic=native \
  --cert=NN/Examples/Verification/ODE/sample_ode_cert.json
```

```terminal +output
[ODE] FAIL initial: uL(t0)∈(-0.000000, 0.000000) not ≤ init.lo=0.000000
error: [ODE] certificate verification failed.
```

The corridor really is the zero function, so the tube is valid and this rejection is a false
negative rather than an unsound acceptance. Read the printed interval carefully: the lower
corridor's value at $`t_0` was returned as a narrow box straddling zero, and the initial check
asks for its upper endpoint to be at most `init.lo = 0`. Under the coarser binary32 reference the
same quantity lands exactly on zero and the check passes. Under host double precision it does not.

The initial comparison uses the conservative endpoint for each wall. To establish that every
possible lower-wall value is below the initial lower bound, its *upper* endpoint must be below
that bound. Dually, the *lower* endpoint of the upper-wall enclosure must exceed the initial
upper bound. This is stronger than merely asking whether the wall intervals overlap the initial
interval. In the rejected example, six-decimal printing hides the positive amount in the lower
wall's upper endpoint. The failed comparison is therefore compatible with the displayed zeros;
the diagnostic needs to be read as an endpoint test, not as exact equality of the rendered text.

The arithmetic setting affects acceptance even for a mathematically valid tube. With sound
enclosures, excess width can cause false negatives; an endpoint rounded inward can instead miss
values that the check needs to cover. These examples do not establish outward rounding for every
computed endpoint. That remains part of the missing connection between the checker and the
real-valued theorem.

Time boxes are seeded from their endpoints directly, avoiding an extra center/radius rounding
roundtrip. Recursive subdivision uses an overflow-resistant midpoint and stops if no representable
interior midpoint remains. Rejection diagnostics distinguish this limit from the configured
maximum depth and minimum width. Partitioning does not change the configured inequalities, and
an initial-time failure is a point check that later subdivision cannot repair.

The `slack` setting does change the segment inequalities: `checkSub`, `checkSuper`, and
`checkOrder` each add it to the right-hand side of a comparison. The real enclosure theorem has
no such tolerance. A positive-slack success therefore needs an additional argument recovering
the exact inequalities required by the theorem. The initial containment checks use no slack.

The theorem side is the real mathematical statement. In the
{src "NN/Proofs/Verification/ODE/Enclosure.lean"}[ODE enclosure API], a corridor theorem says, in
plain language:

> A continuous solution of the clamped dynamics stays between continuous subsolution and
> supersolution walls under the stated right-derivative, initial-value, and ordering hypotheses.

The backend bridge in
{src "NN/Proofs/Verification/ODE/EnclosureBackends.lean"}[ODE enclosure backends] explains how
backend valued trajectories, including FP32 and FloatLib binary32 views, can be related back to
the real
statement through explicit interpretation maps.

Lean has a local real enclosure theorem, and the executable checker computes the kinds of corridor
inequalities that theorem consumes. There is not currently a theorem saying that a successful
`runCertificate` call supplies all of the real-analysis hypotheses of that enclosure theorem.
Continuity, derivative agreement, interval soundness, the treatment of slack, and any
finite-to-real interpretation still
have to be connected explicitly. Broader neural ODE and integrator claims need their own enclosure
conditions and agreement evidence as well.

The trusted boundary is therefore:

```
external integrator/search -> proposed tube JSON
Lean parser/checker        -> interval side conditions for the tube
ODE theorem                -> statement about true trajectories, if theorem hypotheses match
runtime bridge             -> needed for a claim about a concrete finite-precision integrator
```

# PINN Certificates

A uniform PINN guarantee needs several pieces of evidence:

1. the imported parameters match the architecture;
2. the PDE expression is the one being checked;
3. the residual is bounded over the domain;
4. boundary or dataset constraints are respected.

TorchLean gives each piece a small object. The
{src "NN/Verification/PINN/Architecture.lean"}[PINN architecture API] names sequential network
records and graph construction. The
{src "NN/Verification/PINN/PdeAst.lean"}[PDE expression API] names the PDE language. The
{src "NN/Verification/PINN/PyTorch/ParamStore.lean"}[PyTorch parameter store API] names imported
parameters instead of letting a raw tensor dictionary float around unchecked. The
{src "NN/Verification/PINN/ResidualAffine.lean"}[residual affine API] contains the bound helpers,
including McCormick style pieces and branch and bound support.

The architecture record and graph builder expose the bundled network's shape:

```lean (name := pinnArch)
-- Inspect the architecture, graph builder, parameter
-- carrier, and dataset options.
#check PINN.SequentialPINNArch
#check @PINN.buildReferenceGraph
#check @PINN.referenceParams
#check @PINN.DatasetCheck.Options
```

```leanOutput pinnArch
NN.Verification.PINN.SequentialPINNArch : Type
```

```leanOutput pinnArch
PINN.buildReferenceGraph : ℕ → NN.MLTheory.CROWN.Graph
```

```leanOutput pinnArch (whitespace := lax)
@PINN.referenceParams : {α : Type} → [inst : TorchLean.Storage α] →
  [inst_1 : Context α] → ℕ → NN.MLTheory.CROWN.Graph.ParamStore α
```

```leanOutput pinnArch
PINN.DatasetCheck.Options : Type
```

`referenceParams` is generic in the scalar type for the same reason `ODE.eval` is: the identical
demonstration weights have to be available as `Float` for the bundled replay and as FloatLib
binary32
when a check wants bit-level semantics. Asking the architecture for its layer dimensions gives the
`1 -> 16 -> 16 -> 1` network the certificate describes:

```lean (name := archDims)
-- Compare affine layer dimensions with the graph’s final
-- node identifier.
#eval (referenceArch 1).linearDims
#eval (referenceArch 1).outputNodeId
```

```leanOutput archDims
#[(1, 16), (16, 16), (16, 1)]
```

```leanOutput archDims
5
```

Node `5` is the output because the graph interleaves linear layers with activations, so three linear
layers and two tanh nodes give ids `1, 2, 3, 4, 5`. The bundled `pinn-cert` path is intentionally
smaller than the full target sketched above. Its graph is that fixed tanh network
`buildReferenceGraph 1` and its deterministic parameters are `referenceParams 1`, both defined in
Lean. The JSON supplies sample points, box radii, a PDE expression, value intervals, and two
kinds of
residual interval. `verifyCert` recomputes values and derivatives with the Float bound
implementation,
then compares the stored value and residual intervals using `certTol`. The individual derivative
intervals are printed; they are not separate artifact fields checked for equality. The command
returns success or an error; it does not construct a proof object for a uniform residual
proposition.

The dimensions output describes three affine maps, while the node id describes their placement
in a graph that also contains nonlinear operations. These are different counts. The input has
one coordinate, each hidden affine map produces sixteen coordinates, and the final map produces
the scalar field value. Computing a PDE residual requires derivatives with respect to that input
coordinate; it is not asking for gradients with respect to the network's trainable weights.
The parameter store selects the particular field being checked, and the PDE expression selects
which of its input derivatives enter the residual.

## Residual Interval Width

The transcript at the start of this chapter printed a residual enclosure of
$`[-25.6,25.6]` for a network with tanh hidden activations. Each hidden activation lies in
$`[-1,1]`;
the final affine layer has positive weights summing to $`2.8`, matching the certificate's `u`
bounds $`[-2.8,2.8]`. Starting from those value intervals lets us follow the width through two
different residual calculations.

The certificate carries *two* residual fields. The one named `residual_bounds` comes from the
three-point second difference

$$`\frac{u(x-h)-2u(x)+u(x+h)}{h^2},`

with $`h=0.01`. Interval arithmetic evaluates that numerator as if the three values were
independent, so a $`\pm2.8` enclosure at each of the three points gives a numerator with half-width
$`4\times2.8=11.2`, and then the division by $`h^2=10^{-4}` multiplies the width by ten thousand:

```lean (name := fdResidual)
-- Independent value boxes lose the cancellation in the
-- second-difference numerator.
#eval finiteDifferenceResidual
  { lower := -2.8, upper := 2.8 }
  { lower := -2.8, upper := 2.8 }
  { lower := -2.8, upper := 2.8 } 0.01
```

```leanOutput fdResidual
{ lower := -112000.000000, upper := 112000.000000 }
```

That is exactly the `residual_bounds` field stored in the bundled JSON, and the arithmetic behind it
is one line over the reals:

```lean
-- Check the half-width amplification using exact real
-- arithmetic.
example :
    (2.8 + 2.8 + 2.8 + 2.8) / ((1 : ℝ) / 100 * (1 / 100))
      = 112000 := by
  norm_num
```

Over the reals, those value intervals give the stated bound on the difference quotient, but say
little about the network's second derivative. Shrinking $`h` makes the interval width grow
quadratically, even when smoothness makes the finite-difference truncation error smaller.
The independent value boxes do not capture the cancellation needed for a sharp derivative
estimate.

For an exact smooth function, neighboring values are highly related. A Taylor expansion cancels
the constant and first-order terms in
$`u(x-h)-2u(x)+u(x+h)`, leaving the second derivative times $`h^2` plus a remainder. Independent
intervals discard the relation needed for that cancellation. Their numerator uncertainty need
not shrink with `h`, so division amplifies it. This is a different source of error from the
Taylor remainder. Reducing the spacing addresses the remainder only after enough regularity has
been established, and may make the independent-box estimate much worse. The two residual fields
therefore compare different computations, not two interchangeable estimates of the same accuracy.

Propagating derivative bounds through the graph avoids this difference quotient. That is what the
`residual_bounds_deriv` field records and what the printed `u''` lines report. For the same network
whose value bounds are $`\pm2.8`, propagation reports $`\pm25.6479` for the second derivative,
compared with $`\pm112000` for the difference quotient, a factor of about four thousand.
The printed residual is the derivative-based number because the bundled PDE
string is `uxx`. The signature of that helper says where the two fields differ:

```lean (name := fdSig)
-- The finite-difference helper receives three value
-- intervals and a spacing.
#check @finiteDifferenceResidual
```

```leanOutput fdSig (whitespace := lax)
finiteDifferenceResidual : PINN.FloatInterval → PINN.FloatInterval →
  PINN.FloatInterval → Float → PINN.FloatInterval
```

It consumes three *value* enclosures and a spacing. It receives no information about how the
values vary together along the network. Both residual fields are checked, making the loss from
independent stencil intervals visible alongside the direct derivative calculation. Relating the
difference quotient to the derivative would also require a truncation-error bound.

## The Residual Language

The residual itself is a small expression language, evaluated on the primitive enclosures the graph
propagation produced:

```lean (name := pdeSig)
-- Distinguish residual expression evaluation from the
-- graph’s value-bound helper.
#check PINN.PdeAst.Expr
#check @PINN.PdeAst.eval
#check @PINN.ResidualAffine.crownUBoundsForward
```

```leanOutput pdeSig
NN.Verification.PINN.PdeAst.Expr : Type
```

```leanOutput pdeSig
PINN.PdeAst.eval : Prims → PINN.PdeAst.Expr → Option (Float × Float)
```

```leanOutput pdeSig (whitespace := lax)
PINN.ResidualAffine.crownUBoundsForward : NN.MLTheory.CROWN.Graph →
  NN.MLTheory.CROWN.Graph.ParamStore Float →
  Array (Option (NN.MLTheory.CROWN.FlatBox Float)) → Option (Float × Float)
```

`Prims` is a record of five optional intervals: the value, the two first partials, and the two
second partials. The bundled one-dimensional checker fills three of them and leaves the two
$`y`-axis entries empty, exactly as written here:

```lean
-- Supply the one-dimensional artifact’s values and
-- derivatives; leave Y absent.
def artifactPrims : Prims where
  u := some ((-2.8), 2.8)
  duX := some ((-0.0), 4.488)
  duY := none
  d2uX := some ((-25.6479), 25.6479)
  d2uY := none
```

Those are the numbers from the bundled artifact. Evaluating the PDE string `uxx` against them
reproduces the residual the command printed:

```lean (name := pdeX)
-- The uxx residual directly selects the available second X
-- derivative.
#eval PINN.PdeAst.eval artifactPrims (.d2u .X)
```

```leanOutput pdeX
some (-25.647900, 25.647900)
```

Now ask the same primitives for a $`y`-derivative, which a two-dimensional PDE string would need:

```lean (name := pdeY)
-- Asking for a second Y derivative fails because no such
-- interval was supplied.
#eval PINN.PdeAst.eval artifactPrims (.d2u .Y)
```

```leanOutput pdeY
none
```

Here `eval` returns `none` because the requested primitive is absent. The checker reports
`PINN PDE '...' evaluation failed because required primitives are missing`. Supplying zero instead
would change the equation being checked.

Addition is endpointwise, so a Poisson-style residual $`u_{xx}+u` adds the two enclosures:

```lean (name := pdeSum)
-- Add the value and second-derivative intervals to bound
-- the selected residual.
#eval PINN.PdeAst.eval artifactPrims (.add (.d2u .X) .u)
```

```leanOutput pdeSum
some (-28.447900, 28.447900)
```

The `some` wrapper in the sum result records successful evaluation of every primitive needed by
that expression. For $`u_{xx}+u`, the lower endpoints add to $`-28.4479` and the upper endpoints
to $`28.4479`. Requesting $`u_{yy}` fails earlier because `d2uY` is absent. An absent derivative
is not a zero derivative: zero would be a substantive assertion that the field has no curvature
in that direction. This is why the optional fields belong in `Prims`. They let one-dimensional
and two-dimensional callers share a residual language without silently supplying missing
mathematical information.

The PINN and ODE evaluators use different multiplication rules on the same box:

```lean (name := twoMuls)
-- Compare the McCormick-plane collapse with the four-corner
-- product interval.
#eval PINN.PdeAst.ivalMul ((-1.0), 1.0) ((-1.0), 1.0)
#eval ODE.Ival.mul ((-1.0 : Float), (1.0 : Float))
  ((-1.0 : Float), (1.0 : Float))
```

```leanOutput twoMuls
(-3.000000, 3.000000)
```

```leanOutput twoMuls
(-1.000000, 1.000000)
```

Both intervals contain the true range on this exactly represented box; the four-product rule is
three times tighter. This calculation does not prove soundness for all floating inputs. The PINN
version selects a McCormick upper plane and lower plane, then bounds each plane over the box.
Affine bounds can preserve dependence on inputs when a CROWN-style pass carries them through
later layers. This particular `ivalMul`, however, immediately reduces them to two endpoints.
On $`[-1,1]^2`, the upper plane $`uv\le -u+v+1` agrees with the product at three corners, but
reaches `3` at the remaining corner, where the product is minus one.

The wider interval here is a consequence of that plane selection and collapse. The PDE evaluator
does not retain the affine form for later composition. Over exact reals, intersecting this result
with the four-product interval would preserve containment and improve the bound; a floating
implementation would also need to justify its rounding.

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
{src "NN/Verification/PINN/Certificate.lean"}[PINN certificate API], the
{src "NN/Verification/PINN/DatasetCheck.lean"}[dataset checker API], and the
[PINN command API](https://github.com/lean-dojo/TorchLean/blob/main/NN/Verification/PINN/CLI.lean)
are the user-facing pieces for that path.

A PINN claim names the PDE residual, domain, boundary data, and parameters as well as the model
architecture. TorchLean records these inputs across its PINN tools. `pinn-cli` explores one- and
two-dimensional residual boxes with IBP or CROWN-style methods, while `pinn-dataset-check`
performs pointwise interval containment checks and can load an optional PyTorch parameter file.
The dataset command is report-only by
default: it prints `ok` and `bad` counts but exits successfully even when misses are present. Use

```terminal
# Make dataset misses fail the command instead of appearing
# only in its report.
lake exe verify -- pinn-dataset-check --strict
```

when a nonzero `bad` count should fail an automated run. Even strict success is still a checker
result until a soundness theorem connects the selected bound path and imported parameters to a
quantified PDE statement.

The reference point for the application is the physics-informed neural network of
{Informal.citet pinn2019}[]. That paper motivates the residual objective. TorchLean's bundled replay
makes a narrower claim: for its fixed graph and parameters, the artifact's stored value and two
residual intervals agree, within the checker's comparison
tolerance, with the Float quantities recomputed at the declared boxes. The first- and
second-derivative intervals are printed during that replay. The dataset and
interactive commands have their own inputs and checks; they should not be folded into the meaning of
`pinn-cert`.

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
per-point derivatives. A model that couples rows would require a different Jacobian calculation.
`loss` is a mean over 4096 sampled points;
replacing `.mean()` with `.max()` would give the sample maximum, which is still a statement about
those 4096 points and not about the domain. Between them the residual is unconstrained, and a
network can be trained to drive the sampled residual to $`10^{-6}` while violating the equation
elsewhere. The sampled loss alone cannot rule this out.

The intended box claim quantifies over every input in the box. The current Lean checker
recomputes candidate bounds; the missing soundness bridge must justify that quantifier before
these Float bounds become a theorem about the residual.
A sampled residual of $`10^{-6}` and a candidate interval bound of $`25.6` could describe the same
network: one measures selected inputs, while the other may lose substantial information in
propagation. Direct derivative bounds, subdivision, and affine relaxations offer different ways
to reduce that loss. Each selected path still needs a soundness argument.

# Piecewise Polynomial and Spline Certificates

The spline path is concentrated in
{src "NN/Verification/Splines/PiecewisePolyCert.lean"}[NN.Verification.Splines.PiecewisePolyCert
API]. It parses `piecewise_poly_v0` JSON, evaluates polynomial pieces by Horner's rule, checks exact
rational interpolation at adjacent knots, and also has a FloatLib binary32 exact conversion path.

A piecewise polynomial certificate names intervals $`I_i` and polynomial pieces

$$`p_i(x)=\sum_k a_{ik}(x-x_i)^k,\qquad x\in I_i.`

For a piece on $`I_i=[x_i,x_{i+1}]`, the checked equations are

$$`p_i(x_i)=y_i,
\qquad p_i(x_{i+1})=y_{i+1}.`

The two Lean objects behind that are a generic Horner evaluation and a checker over exact
rationals:

```lean (name := splineSigs)
-- Horner evaluation is scalar-generic; the certificate
-- checker uses exact rationals.
#check @evalPolyHorner
#check @checkCertificateRat
```

```leanOutput splineSigs (whitespace := lax)
@evalPolyHorner : {α : Type} → [Zero α] → [Add α] → [Mul α] →
  Array α → α → α
```

```leanOutput splineSigs
checkCertificateRat : PiecewisePolyCertificate → IO Unit
```

`evalPolyHorner` asks only for `Zero`, `Add`, and `Mul`, which is what lets the same evaluation run
over `Rat` for exact checking and over FloatLib binary32 for the `--arithmetic ieee` pass. The
checker
returns `IO Unit` and throws on the first mismatch. Successful direct evaluation prints nothing;
a failed endpoint comparison names the piece and the two unequal values. The JSON parser also
requires at least two knots and exactly one piece per adjacent pair. Those count checks belong
to the parser: a direct call to `checkCertificateRat` checks the supplied pieces without separately
requiring their array to cover every adjacent pair.

Horner's rule reads the coefficient array from highest degree to lowest. For
$`[a_0,a_1,a_2]`, it forms $`(a_2t+a_1)t+a_0`, so the stored array is in ascending power order
even though evaluation folds from the right. The local coordinate also matters: at a piece's
left endpoint it is zero, and at the right endpoint it is `hi - lo`. Feeding the absolute right
knot into this evaluator would describe another polynomial whenever `lo` is nonzero. The checker
computes this translation explicitly before comparing endpoint values, tying each piece's
coefficients to its declared interval.

Matching endpoint values can support continuity when adjacent pieces share the same knot value,
but it does not also match their derivatives. For example, the bump below has derivative `1`
at its left endpoint and `-1` at its right endpoint. Joining it to a zero polynomial with the
same endpoint value produces a derivative jump. A certificate for a smooth surrogate would need
to check the appropriate derivative equalities at each join as well. This is especially relevant
when that surrogate is later differentiated to form an ODE or PDE residual: interpolation data
alone does not determine the residual at the joins.

An interior range theorem would instead need a statement such as

$$`\forall x\in I_i,\qquad p_i(x)\in[\ell_i,u_i],`

together with data or a proof sufficient to check it. That stronger condition is not part of
`piecewise_poly_v0` today. A passing certificate can illustrate the distinction. Pieces use the
local coordinate $`t=x-\mathrm{lo}`, so on the single interval
$`[0,1]` the coefficients $`[0,1,-1]` mean $`p(t)=t-t^2`:

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
def bumpCert : PiecewisePolyCertificate where
  degree := 2
  n := 2
  xs := TorchLean.Tensor.from #[(0 : ℚ), 1]
  ys := TorchLean.Tensor.from #[(0 : ℚ), 0]
  pieces := #[{ lo := 0, hi := 1, coeffs := bumpCoeffs }]

#eval checkCertificateRat bumpCert
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
values by $`\tfrac14`. The certificate declares this polynomial; it does not claim the polynomial
is zero between the knots. An interior bound would be an additional proposition to check.

For this bump, completing the square gives
$`t-t^2=1/4-(t-1/2)^2`. Together with $`0\le t\le1`, that identity explains the entire
interior range $`[0,1/4]`; it is additional mathematics beyond the two checks at zero and one.
Multiplying the bump by any positive rational would still leave both endpoints zero while
changing its midpoint height. Thus exact interpolation data alone cannot recover a bound on
interior deviation from the zero line. A scientific surrogate using the piece between knots
needs the polynomial itself and a range argument, not only agreement with its training knots.

Changing the right knot value while retaining the polynomial produces an endpoint mismatch:

```lean
-- Change only the claimed knot values while retaining the
-- polynomial coefficients.
def tamperedYs : Array ℚ := #[0, 1]

def tamperedCert : PiecewisePolyCertificate :=
  { bumpCert with
    ys := TorchLean.Tensor.from tamperedYs }
```

```lean (name := tamperedCheck) +error
-- The checker reports the exact right-endpoint mismatch in
-- the tampered certificate.
#eval checkCertificateRat tamperedCert
```

```leanOutput tamperedCheck
endpoint mismatch at i=0: p(hi)=0 ≠ yNext=1
```

No tolerance appears anywhere in that comparison. The certificate is parsed into `Rat`, evaluated in
`Rat`, and compared with `==` on `Rat`, so a passing check is an exact algebraic identity rather
than a floating-point coincidence. The `--arithmetic ieee` pass adds a second, stricter question on
top: every rational in the document must be exactly representable in binary32, and the endpoint
equalities must hold again under the executable reference semantics. A certificate can pass the
rational check and fail that one, and the failure is informative, because either a stored value is
not exactly representable or rounded Horner evaluation fails
an endpoint equality even though all input constants are representable.

An external spline fitter can produce these same coefficients and knot data. The exact rational
check then validates interpolation independently of the fitter's numerical procedure. It says
nothing further about approximation error, derivative continuity at knots, or interior ranges.

The representability test and the rounded evaluation test address different questions. A value
such as $`1/2` can be represented exactly in binary32, whereas $`1/3` cannot. But exact input
coefficients do not imply exact products and sums throughout Horner's rule. Intermediate results
can require more significand bits or exceed the format's finite range. The IEEE pass consequently
checks both the conversion of the document and the endpoint computations under the reference
operations. The rational pass remains useful on its own: it identifies the polynomial the
artifact describes without mixing that algebraic question with one execution format.

# Artifact Boundary Examples

The same scientific artifact can support different strengths of claim depending on what it exports.

- If a PINN JSON contains only sampled residuals, Lean can check those samples; it cannot infer a
  uniform residual bound over the domain.
- If a PINN certificate contains interval or affine residual bounds over domain boxes, Lean can
  recompute those box obligations. A uniform residual theorem additionally needs soundness of
  the bound computation and evidence covering the requested domain.
- If an ODE artifact contains a proposed trajectory but no interval enclosure condition, Lean can
  parse the trajectory but does not get an enclosure theorem.
- If a piecewise polynomial artifact contains rational coefficients in the current format, Lean can
  check exact knot interpolation. An interior range claim needs a richer schema and checker.

This is the same checked/proved/assumed distinction used for robustness certificates. The producer
may be a numerical solver; the theorem applies only to the artifact fields that Lean checked or to
producer hypotheses named in the statement.

# Checking Scientific Artifacts

Exporting a small artifact makes it possible to recompute its claims in Lean. Connecting the
accepted artifact to a theorem then requires a proof that these checks imply the theorem's
hypotheses. A plot can show how a corridor or residual bound behaves across the domain; the
checker and soundness theorem determine which conclusions follow from it.

The examples expose different sources of uncertainty. Rewriting $`u(1-u)` as
$`\tfrac14-(u-\tfrac12)^2` reduced interval dependency on the chosen box. Direct derivative
propagation avoided the width introduced by dividing independent value bounds by $`h^2`.
The spline midpoint value of $`\tfrac14` showed why exact knot interpolation does not determine
an interior range. Improving a bound and strengthening a certificate's claim are separate tasks;
both depend on identifying which information the current check uses.
