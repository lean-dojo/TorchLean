import VersoManual
import TorchLean.NN.Lipschitz.Network
import TorchLean.Verify.RuntimeApprox.NF.EndToEnd
-- The experiments narrow finite native inputs to binary32 and widen their results for display.
-- Import the scalar API directly, separately from the reduction proofs.
import FloatLib
-- The reduction-schedule section prints `SumResult.enclosure`, which lives with
-- the reduction-tree theorems rather than with the scalar operations.
import TorchLean.Verify.Floats.IEEE754.Reductions
import TorchLeanBlueprint.Bib
import TorchLeanBlueprint.Roles

open Verso.Genre Manual
open Verso.Genre.Manual.InlineLean
open FloatLib.Numerics
open FloatLib.Floats.Formats.Flocq
open TorchLean.Tensor
open TorchLean
-- Open the namespace containing the reduction theorems.
open IEEE754

-- Verso checks `leanOutput` blocks against the real compiler message. The graph
-- theorems below print wider than this file's 100-column limit, so those blocks ask
-- for `whitespace := lax`, which ignores where the expected text was wrapped. The
-- rendered page still shows the signature exactly as Lean printed it.
open Lean.Elab.Tactic.GuardMsgs.WhitespaceMode (lax)

open FloatLib.Floats (ExecFloat)
open FloatLib.Floats.Formats.BinaryInterchange (Model FloatFormat)

#doc (Manual) "Numerical Error" =>
%%%
tag := "floating-point-literature"
file := "Tracking-Numerical-Error-Through-A-Network"
%%%

A rounding error in the first linear layer becomes part of the input to ReLU and then to the second
layer. Later operations may preserve, amplify, or reduce it while introducing errors of their own.
So bounding each operation in isolation leaves out that dependence.

I'll follow the error through the same two-layer MLP used in the executable comparison below. Our
starting point is the local rounding analysis of {Informal.citet goldberg1991}[]. FloatLib supplies
those scalar facts, and TorchLean's covered tensor operations turn them into bounds on output
tensors. A graph theorem then composes the bounds in evaluation order.

Training extends the calculation. Reverse mode reads saved forward values, so their errors
affect the gradient. An optimizer then uses that gradient to change the parameters. With a
contract for each of these operations, we can follow the discrepancy through one complete
training step: forward values, gradients, and the next parameter state.

# Native And Reference MLP Execution

Run the native/reference comparison with:

```terminal
# Compare the executable binary32 model with the Float32
# path on the same MLP.
scripts/lake.sh exe torchlean float32_semantics
```

It evaluates

$$`y=W_2\operatorname{ReLU}(W_1x+b_1)+b_2`

twice, first with Lean's native `Float32` and then with the independent bit-level FloatLib binary32
reference (`ExecFloat.Binary 8 23`). Both runs evaluate the typed graph and then compute the
parameter and input VJPs through `autograd.model.vjp`.

Read the six named tensors in each report: the output, both weight gradients, both bias gradients,
and the input gradient. The final line reports their maximum absolute difference after converting
the entries to `Float`. Inspect that value: the command reports differences without requiring them
to be zero. For finite values a zero difference establishes more than agreement of the printed
decimals, but it doesn't distinguish signed zeros or inspect intermediate rounding steps.

Across the two runs the input and parameter pack stay fixed while the scalar implementation
changes. The approximation theorems below address quantified claims and state the assumptions
needed to connect rounded calculations to real-valued ones.

The gradient comparison exercises a different part of the computation from the forward comparison.
A scalar output seeded with one asks how that output changes with each parameter, so the printed
weight and bias entries must follow the model's parameter order. Matching only the final output
would miss, for example, a reverse rule that sends the right numbers to the wrong parameter tensor.
Inspecting every gradient tensor checks that correspondence on this example, while the
approximation theorems below explain how to make a statement with explicit input and arithmetic
hypotheses.

# The Approximation Relation

Let $`x` be an ideal real tensor, $`\widehat{x}` its rounded counterpart with the same shape, and
$`\varepsilon` a bound on their coordinatewise error. A map `toSpec` interprets each rounded
scalar as a real value. TorchLean writes the relation as

$$`\operatorname{ApproxTensor}(x,\widehat x,\varepsilon)
  \quad\Longleftrightarrow\quad
  \forall i,\;
  |\operatorname{toSpec}(\widehat x_i)-x_i|\leq\varepsilon`.

Here the index $`i` ranges over the tensor's coordinates, so the same bound applies at every entry.
For `NF`, the map is `NF.toReal`, which returns the stored real value (for another runtime type,
the map can be different).

Lean's signature says the same thing more tersely:

```lean (name := approxTensorSig)
-- Expose the scalar interpretation and the shared
-- coordinatewise error budget.
#check @ApproxTensor
```
```leanOutput approxTensorSig (whitespace := lax)
@ApproxTensor : {α : Type} → [inst : Storage α] →
  {s : Shape} → (α → ℝ) → Tensor ℝ s → Tensor α s → ℝ → Prop
```

The last argument is the error bound. Because it's a separate argument, graph propagation can
update the bound without rebuilding the tensor, and we can relate the same runtime value to several
ideal values, each with its own error. And the `toSpec` argument selects how to interpret the
runtime scalar.

A shared shape fixes which coordinates are compared. A budget of $`\varepsilon` permits that much
error at each coordinate, so summing many coordinates can require a larger output budget even when
every input satisfies the same relation. That's why a layer theorem needs an error transformer as
well as an input approximation hypothesis.

# Error Bounds For A Linear Layer

For one output coordinate, an exact affine layer computes

$$`y_j=\sum_{k=0}^{n-1}W_{jk}x_k+b_j`.

Here $`j` selects an output coordinate and $`n` is the input width. Suppose the runtime has
approximations $`\widehat W`, $`\widehat x`, and $`\widehat b` with coordinatewise errors
$`\varepsilon_W`, $`\varepsilon_x`, and $`\varepsilon_b`. In the following error calculation,
the hatted operands denote their decoded real values. Before accounting for arithmetic rounding,
one product satisfies

$$`
|\widehat W_{jk}\widehat x_k-W_{jk}x_k|
\leq
|W_{jk}|\,\varepsilon_x
+|x_k|\,\varepsilon_W
+\varepsilon_W\varepsilon_x.
`

To obtain the bound, expand the perturbed product and group the error in each operand. Suppressing
the coordinate indices gives

$$`
\widehat W\widehat x-Wx
=W(\widehat x-x)+x(\widehat W-W)
  +(\widehat W-W)(\widehat x-x).
`

In the expansion, the first term propagates input error through the exact weight and the second
propagates weight error through the exact input. And the third accounts for perturbing both
operands at once.

Arithmetic introduces further error. Write $`P_k` for the total error charged to product $`k` and
$`E_k` for the accumulated error after $`k` products, and let $`s_k` and $`p_k` be the ideal
partial sum and product. In the recurrence below, $`\rho_{\rm mul}(W_{jk},x_k)` must bound
multiplication rounding throughout the operand neighborhoods given by $`\varepsilon_W` and
$`\varepsilon_x`, and $`\rho_{\rm add}(s_k,p_k)` must likewise cover the partial sum and product
errors $`E_k` and $`P_k`. These error radii are implicit arguments of the notation. Evaluating
rounding error only at the ideal operands wouldn't suffice. With an exact zero accumulator, a
fixed-left dot product has the recurrence

$$`
\begin{aligned}
E_0 &= 0,\\
P_k &=
  |W_{jk}|\,\varepsilon_x
  +|x_k|\,\varepsilon_W
  +\varepsilon_W\varepsilon_x
  +\rho_{\rm mul}(W_{jk},x_k),\\
E_{k+1} &= E_k+P_k+\rho_{\rm add}(s_k,p_k).
\end{aligned}
`

Each step adds the previous sum error, its own product's error, and the rounding error of combining
them. After the final product, add $`\varepsilon_b` and the rounding of the bias addition. Local
rounding bounds must cover the operands reached by this execution, which is why the theorem needs
both a reduction order and magnitude information.

TorchLean's `NF` backend implements this style of propagation, with bounds computed from runtime
operands. In `NF.dotStep`, the magnitude terms use the decoded rounded operands plus their
error budgets, and the local rounding terms are half-ULP bounds at the runtime product and sum
before rounding. (`dotBound` also conservatively charges half an ULP for its initial zero
accumulator.) So the implementation follows the error decomposition above, but its bound expression
isn't identical. Matrix multiplication, convolution, reductions, and scalar arithmetic each provide
an error transformer and a theorem proving that transformer valid.

A dot product has two sources of uncertainty. Its represented weights and inputs may already differ
from the intended real data, and each multiplication and addition introduces another rounding step.
Even an exact multiplication of the represented operands would retain the first source, and
conversely, exactly represented input data don't make every intermediate product or sum
representable. Keeping the two contributions separate lets a caller improve the right part of a
failed bound: a more accurate parameter conversion and a different accumulation algorithm solve
different problems.

# Accumulated Error Bounds And Measurements

Our dot-product recurrence bounds absolute errors without exploiting cancellation between local
rounding errors, and it isn't automatically linear in the number of terms: local magnitudes and
rounding bounds may also grow. Under the classical relative-error model, with $`nu<1` and no
problematic underflow or overflow, a sequential sum satisfies

$$`|\widehat S-S|\leq\gamma_n\sum_i|x_i|,\qquad
\gamma_n=nu/(1-nu).`

Here $`u=2^{-24}` for binary32 (using $`n` instead of $`n-1` is a conservative convention). For
nonnegative terms with nonzero sum, the sequential-sum bound becomes a relative bound of
$`\gamma_n`. With mixed signs, divide by $`|S|` and keep the condition factor $`\sum_i|x_i|/|S|`.

Our harmonic-sum experiment compares binary32 against a binary64 reference, and its positive terms
keep the condition factor at one. Each term is first computed in binary64 and converted to
binary32, so the observed difference includes term conversion as well as accumulation. The printed
classical factor is a comparison scale, not a Lean proof about this complete experiment.

```lean (name := gammaGrow)
-- Compare accumulated harmonic sums with the growth of the
-- classical gamma factor.
-- The binary64 sum is a numerical reference, not an
-- exact-real oracle.
def harm32 (n : Nat) : (ExecFloat.Binary 8 23) :=
  (List.range n).foldl
    (fun acc k =>
      ExecFloat.add acc
        ((ExecFloat.Binary.ofFloat32 ∘ Float.toFloat32)
          (1.0 / Nat.toFloat (k + 1))))
    0

def harm64 (n : Nat) : Float :=
  (List.range n).foldl
    (fun acc k => acc + 1.0 / Nat.toFloat (k + 1)) 0.0

-- Unit roundoff for binary32, and Higham's growth factor.
def u32 : Float := 1.0 / 16777216.0

def gammaN (n : Nat) : Float :=
  Nat.toFloat n * u32 / (1.0 - Nat.toFloat n * u32)

#eval do
  for n in [4, 16, 64, 256, 1024, 4096] do
    let ideal := harm64 n
    let got :=
      (ExecFloat.Binary.toFloat32 (harm32 n)).toFloat
    let rel := ((got - ideal) / ideal).abs
    IO.println s!"n={n} observed={rel / u32} u  \
      bound={gammaN n / u32} u"
```
```leanOutput gammaGrow (whitespace := lax)
n=4 observed=1.280000 u  bound=4.000001 u
n=16 observed=0.164718 u  bound=16.000015 u
n=64 observed=2.872767 u  bound=64.000244 u
n=256 observed=2.237023 u  bound=256.003906 u
n=1024 observed=16.215394 u  bound=1024.062504 u
n=4096 observed=8.159338 u  bound=4097.000244 u
```

Both columns are in multiples of $`u`. The comparison factor is about $`n`, while the observed
error is much smaller at the larger lengths and isn't monotone. These deterministic data don't
establish a random-walk law or a $`\sqrt n` growth rate.

A worst-case bound need not predict typical error. It can also be sharpened by using more facts
about the inputs or intermediate values. Changing the program offers another route: compensated
summation, a pairwise reduction with logarithmic depth, or a wider accumulator. Each needs an error
contract matching the implemented arithmetic. The reduction theorem below makes its hypotheses
explicit. It doesn't certify the printed $`\gamma_n` column unconditionally.

# Sequential, Pairwise, And Compensated Summation

Keep the converted harmonic terms fixed and change only the accumulation algorithm. The naive
column uses the `harm32` fold above, while the pairwise column adds neighbours in rounds until one
value remains. Kahan summation keeps a correction for rounding lost at the previous addition and
uses it to adjust the next term. All three are compared with the same binary64 reference.

```lean (name := sumStrategies)
-- The same term the naive fold above added.
def harmTerm (k : Nat) : (ExecFloat.Binary 8 23) :=
  (ExecFloat.Binary.ofFloat32 ∘ Float.toFloat32)
    (1.0 / Nat.toFloat (k + 1))

-- Pairwise reduction: logarithmic depth instead of linear.
-- An odd element rides to the next round untouched.
def harmPairwise32 (n : Nat) :
    (ExecFloat.Binary 8 23) := Id.run do
  let mut level : Array (ExecFloat.Binary 8 23) :=
    (Array.range n).map harmTerm
  while level.size > 1 do
    let mut next : Array (ExecFloat.Binary 8 23) := #[]
    for i in [0 : level.size / 2] do
      next := next.push
        (ExecFloat.add level[2 * i]! level[2 * i + 1]!)
    if level.size % 2 == 1 then
      next := next.push level[level.size - 1]!
    level := next
  pure level[0]!

-- Kahan's signed correction estimates the addition error
-- as (next - total) - adjusted.
-- Subtract it from the next term.
def harmKahan32 (n : Nat) : (ExecFloat.Binary 8 23) :=
  ((List.range n).foldl
    (fun (state : (ExecFloat.Binary 8 23) ×
        (ExecFloat.Binary 8 23)) k =>
      let (total, drift) := state
      let adjusted := ExecFloat.sub (harmTerm k) drift
      let next := ExecFloat.add total adjusted
      let recovered := ExecFloat.sub next total
      (next, ExecFloat.sub recovered adjusted))
    (0, 0)).fst

#eval do
  for n in [64, 256, 1024, 4096] do
    let ideal := harm64 n
    let err := fun (got : (ExecFloat.Binary 8 23)) =>
      (((ExecFloat.Binary.toFloat32 got).toFloat - ideal)
        / ideal).abs / u32
    IO.println s!"n={n} naive={err (harm32 n)} \
      pairwise={err (harmPairwise32 n)} \
      kahan={err (harmKahan32 n)}"
```
```leanOutput sumStrategies (whitespace := lax)
n=64 naive=2.872767 pairwise=1.186388 kahan=0.499992
n=256 naive=2.237023 pairwise=0.930761 kahan=0.375501
n=1024 naive=16.215394 pairwise=1.300308 kahan=0.234945
n=4096 naive=8.159338 pairwise=2.763110 kahan=0.964368
```

Pairwise summation stays below three units of roundoff across a factor of sixty-four in length,
while the naive fold reaches sixteen. A depth-based worst-case analysis explains one advantage: in
a 4096-term pairwise reduction every input passes through twelve additions instead of up to 4096.
For the pairwise scheme, the classical analysis replaces $`\gamma_n` by
$`\gamma_{\lceil\log_2 n\rceil}`, which at $`n=4096` is about twelve units of roundoff instead of
four thousand. Kahan stays below one unit in these four runs. In the code, `recovered` subtracts
the old total from the new one to estimate how much of `adjusted` entered the accumulator, and the
difference is retained as `drift` and subtracted from the next term. This explains the role of the
correction without turning these measurements into a general error bound.

TorchLean's reduction theory represents schedules (such as the pairwise one) as FloatLib
`ReductionTree`s. `SumResult xs r` says that some schedule over some permutation of the inputs
produces `r` with finite leaves and finite intermediate results, and the enclosure theorem bounds
the error of whichever schedule that was:

```lean (name := sumTreeEnc)
-- Binary32 uses the tree's own representable operands.
#check @SumResult.enclosure
```

At binary32 the corollary uses $`u=2^{-24}` (the format's `Model.unitRoundoffAt`) and requires
finite inputs and finite intermediate sums. It reuses FloatLib's addition bound on the tree's own
representable operands (including subnormal sums), and it doesn't assume a relative rounding bound
for arbitrary real operands, which the usual unit roundoff can't supply near zero.

The factor `(1 + u) ^ t.nodeCount` is $`(1+u)^{n-1}` for $`n` leaves, since a tree with $`n` leaves
performs $`n-1` additions. It doesn't depend on tree depth, so it charges the same factor to a
balanced schedule and a sequential one. A sharper logarithmic-depth bound requires another theorem.
The library also doesn't provide a complete compensated-summation error contract. Its proved
Sterbenz subtraction lemma is a useful ingredient, but it doesn't by itself prove the Kahan program
above.

The witness tree records an allowed order. Its permutation condition preserves every supplied
summand with multiplicity, so it doesn't permit dropping a term.

`DotResult.enclosure` gives the same accumulation bound for a dot
product's already-rounded products. Comparing with the original real dot product also needs the
multiplication errors. Neither corollary identifies the schedule used by an external GPU kernel.

# Cancellation And Relative Error

The harmonic terms were positive, so their sum and sum of absolute values were equal. With mixed
signs, cancellation can make the final sum much smaller than the intermediate magnitudes. Add a
large number, a small one, and the negative of the large one in left-to-right order:

```lean (name := cancelSum)
-- Keep the summands fixed and change their order to expose
-- cancellation.
def cancelTerms : Array (ExecFloat.Binary 8 23) :=
  #[ (ExecFloat.Binary.ofFloat32 ∘ Float.toFloat32) 1.0e8
   , (ExecFloat.Binary.ofFloat32 ∘ Float.toFloat32) 1.0
   , (ExecFloat.Binary.ofFloat32 ∘ Float.toFloat32)
       (-1.0e8) ]

def foldLeft32 (xs : Array (ExecFloat.Binary 8 23)) :
    (ExecFloat.Binary 8 23) :=
  xs.foldl ExecFloat.add 0

-- Same three numbers, small term last.
def cancelReordered : Array (ExecFloat.Binary 8 23) :=
  #[ cancelTerms[0]!, cancelTerms[2]!, cancelTerms[1]! ]

#eval do
  let sumAbs :=
    cancelTerms.foldl
      (fun acc (x : (ExecFloat.Binary 8 23)) =>
        acc + (ExecFloat.Binary.toFloat32 x).toFloat.abs)
      0.0
  -- Growth factor for a three-leaf schedule: (1+u)^2 - 1.
  let growth := (1.0 + u32) * (1.0 + u32) - 1.0
  let naive :=
    (ExecFloat.Binary.toFloat32
      (foldLeft32 cancelTerms)).toFloat
  let reordered :=
    (ExecFloat.Binary.toFloat32
      (foldLeft32 cancelReordered)).toFloat
  IO.println s!"naive     = {naive}"
  IO.println s!"reordered = {reordered}"
  IO.println s!"exact     = 1.0"
  IO.println s!"sumAbs    = {sumAbs}"
  IO.println s!"bound     = {growth * sumAbs}"
```
```leanOutput cancelSum (whitespace := lax)
naive     = 0.000000
reordered = 1.000000
exact     = 1.0
sumAbs    = 200000001.000000
bound     = 23.841859
```

The naive order loses the answer completely. Binary32 keeps twenty four significand bits and the
two magnitudes are eight decimal orders apart, so `1e8 + 1` returns `1e8` unchanged and the final
subtraction returns zero. Relative to the exact answer that's a one hundred percent error. Move the
small term to the end and the same three numbers add up exactly.

The bound in `SumResult.enclosure` scales with `sumAbs` (the sum of absolute values). Here that
factor is about $`2\times 10^8` while the answer is $`1`, so the illustrative growth formula gives
an absolute bound of about twenty-four against an observed error of one. Turning an absolute bound
into relative accuracy requires the condition number $`\sum_i|x_i|\,/\,|\sum_i x_i|`, about
$`2\times 10^8` on this input.

So an absolute-error certificate can be sound while giving a weak relative bound on a nearly
cancelled result, and reporting `sumAbs` beside the bound makes that scale visible. Reordering also
matters: the same inputs produced either zero or one here. `SumResult` allows permutations of
the inputs, while applying a theorem about one particular schedule requires establishing that the
runtime used it.

Compensation attempts to preserve the unit contribution lost at the first addition. To claim a
stronger guarantee for that program, we'd need an argument about how its correction variable
behaves as well as a bound on each primitive operation.

# ReLU Lipschitz Bounds

ReLU is easier because it's 1-Lipschitz:

$$`
|\operatorname{ReLU}(u)-\operatorname{ReLU}(v)|
\leq |u-v|.
`

In Lean, with ReLU spelled as `max 0`:

```lean
-- A ReLU cannot increase the distance between two scalar
-- inputs: Mathlib's `abs_max_sub_max_le_abs` with `c = 0`.
example (u v : ℝ) :
    |max 0 u - max 0 v| ≤ |u - v| := by
  simpa only [max_comm (0 : ℝ)] using
    abs_max_sub_max_le_abs u v 0
```

If the runtime ReLU itself is exact for the declared scalar semantics, an incoming bound
$`\varepsilon` remains $`\varepsilon`. Things get interesting when $`u` and $`v` lie on opposite
sides of zero. There the derivative changes discontinuously, but the forward Lipschitz bound holds
for every pair of inputs: Mathlib's `abs_max_sub_max_le_abs` proves `|max a c - max b c| ≤ |a - b|`
for every `c`, and the example takes `c = 0`.

For tensors, TorchLean states the same fact in the $`\ell_2` metric, for every shape at once:

```lean (name := reluTensorSig)
-- The tensor theorem lifts the scalar estimate to the
-- stated tensor norm.
#check @l2Dist_relu_le
```
```leanOutput reluTensorSig (whitespace := lax)
@l2Dist_relu_le : ∀ {s : Shape} (x y : Tensor ℝ s), dist x.relu y.relu ≤ dist x y
```

This difference between forward and backward sensitivity matters. A small perturbation can leave
the ReLU output close while changing which VJP branch is selected. So backward approximation
carries branch hypotheses (or a bound that covers both possibilities) rather than blindly reusing
the forward proof.

# Range Conditions For Softmax And Normalization

For a nonlinear operation such as softmax, a global absolute-error rule is usually too weak.
TorchLean's stable softmax first subtracts the row maximum:

$$`
\operatorname{softmax}(z)_i
=
\frac{\exp(z_i-\max_j z_j)}
     {\sum_k\exp(z_k-\max_j z_j)}.
`

For a nonempty row of real logits, every shifted exponent input is nonpositive, the denominator is
at least one, and the output lies in $`[0,1]` (floating-point NaNs and infinities require separate
treatment). Those range facts control the local Lipschitz and rounding terms.

For layer normalization, let $`\mu` be the mean over the normalized coordinates and $`\sigma^2`
their variance. The positive stabilization constant $`\epsilon` is part of the operation (distinct
from the approximation-error bounds above), and the calculation proceeds through

$$`
x
\longmapsto \mu
\longmapsto x-\mu
\longmapsto (x-\mu)^2
\longmapsto \sigma^2+\epsilon
\longmapsto \sqrt{\sigma^2+\epsilon}
\longmapsto
\frac{x-\mu}{\sqrt{\sigma^2+\epsilon}}.
`

Subtracting the mean produces centered values, and squaring and averaging them produces the
variance. Adding $`\epsilon` gives the exact stabilized variance a positive lower bound. Rounding
errors must also be small enough to preserve that margin. `ApproxTensor.normalize` assumes a
lower bound $`\eta>0` for the exact stabilized variance, with premises `stabilizedError < η` and
`stdError < Real.sqrt η`. The first would keep the perturbed square-root input positive, but the
proof doesn't use it, because `Tensor.sqrt _ .clampNegative` clamps its input to zero. And the
second keeps the perturbed denominator away from zero. That theorem's error trace composes the
subtraction, stabilization, square-root, division, and affine-scaling bounds, using the supplied
error bounds for the input, mean, variance, and affine parameters.

These two margins refer to different stages and have different units. For the clamped square-root
proof, the standard-deviation error margin is the one needed to protect division. Positivity of
epsilon alone doesn't establish that margin. Both are mathematical premises about a noncomputable
trace, not comparisons performed by an executable checker.

# Composition Over A Graph

A proof-bearing `RevGraph` stores, for every node:

- the exact forward operation
- the rounded forward operation
- an error transformer for the forward result
- the exact and rounded VJPs
- an error transformer for the VJP
- proofs for both transformers

`RevGraph.eval_approx` follows the graph in topological order. If the input context satisfies its
declared bounds, the runtime output context satisfies the bounds computed by
`RevGraph.evalBounds`.

Our executable graph interpreter uses `GraphData`, and one theorem connects it to the same forward
result:

```lean (name := evalApproxSig)
-- Erase the local certificates and check which executable
-- forward graph they cover.
#check @NF.eval_approx_graphData
```
```leanOutput evalApproxSig (whitespace := lax)
@NF.eval_approx_graphData :
  ∀ {β : Radix} {fexp : ℤ → ℤ} {rnd : ℝ → ℤ}
    {Γ ss : List Shape} (g : RevGraph NF.toReal Γ ss)
    (xS : TensorPack ℝ Γ)
    (xR : TensorPack (NF β fexp rnd) Γ) (epsIn : EList Γ),
  ApproxCtx NF.toReal xS xR epsIn →
    ApproxCtx NF.toReal (g.evalReal xS)
      (g.toGraphData.eval xR ())
      (g.evalBounds epsIn xR)
```

`Γ` lists the input shapes and `ss` lists the shapes appended by the graph. In the hypothesis,
`ApproxCtx` relates the exact input pack `xS` to the rounded pack `xR` using `epsIn`. And the
conclusion relates the exact evaluation `g.evalReal xS` to the lowered `GraphData` evaluation, with
one bound per context entry supplied by `g.evalBounds epsIn xR`. Because the proof composes the
local contracts stored in the nodes of `g`, it holds for every `g`. No TorchLean model or
`TorchLean.IR.Graph` is lowered to a `RevGraph`, so the caller builds `g`.

This forward theorem concludes an approximation for the entire accumulated context: earlier inputs
and intermediate node values remain available alongside the final output. That's useful when the
next theorem concerns backpropagation, since a local VJP may read saved forward values, not only
the graph's output. In the signature, the shape lists keep those saved entries aligned with their
error budgets. An output-only comparison would leave the reverse induction without this
information.

# Backward Error Propagation

Reverse mode starts from a seed cotangent and applies local VJPs from outputs back to inputs and
parameters. {Informal.citet baydin2018}[] is the standard survey of the mechanism. For a
composition $`h(x)=g(f(x))`, let $`\bar h` be the seed cotangent at the output and $`J_f`, $`J_g`
the Jacobians of the two functions. Then the input cotangent is

$$`
\bar x
=J_f(x)^\mathsf{T}
  J_g(f(x))^\mathsf{T}\bar h.
`

Read the product from right to left: the VJP of $`g` maps the seed to a cotangent at the
intermediate value $`f(x)`, and then the VJP of $`f` maps that cotangent back to the input. The
rounded pass perturbs the saved forward values, the seed, each local VJP, and every gradient
accumulation. `RevGraph.backpropBounds` mirrors the runtime traversal and computes the resulting
error context. Our backward theorem adds a second `ApproxCtx` hypothesis for the seed:

```lean (name := backpropApproxSig)
-- The reverse theorem also carries seed errors and rounded
-- gradient accumulation.
#check @NF.backprop_approx_graphData
```
```leanOutput backpropApproxSig (whitespace := lax)
@NF.backprop_approx_graphData :
  ∀ {β : Radix} {fexp : ℤ → ℤ}
    [inst : ValidExp fexp] {rnd : ℝ → ℤ}
    [ValidRndToNearest rnd] {Γ ss : List Shape}
    (g : RevGraph NF.toReal Γ ss)
    (xS : TensorPack ℝ Γ)
    (xR : TensorPack (NF β fexp rnd) Γ) (epsIn : EList Γ)
    (seedS : TensorPack ℝ (Γ ++ ss))
    (seedR : TensorPack (NF β fexp rnd) (Γ ++ ss))
    (epsSeed : EList (Γ ++ ss)),
  ApproxCtx NF.toReal xS xR epsIn →
    ApproxCtx NF.toReal seedS seedR epsSeed →
      ApproxCtx NF.toReal (g.backpropReal xS seedS)
        (g.toGraphData.backpropCtx xR () seedR)
        (g.backpropBounds epsIn xR epsSeed seedR fun {Δ} =>
          NF.ctxAddBound)
```

So `GraphData.backpropCtx` stays inside the computed context whenever the inputs and seed satisfy
their initial approximation relations. The two instance arguments are the price of using the
nearest-rounding facts from the previous chapter: the exponent policy has to be a valid one and the
integer rounder has to round to nearest.

A cotangent seed is numerical input to the reverse computation. Scaling the seed scales the ideal
VJP, and errors in that seed also propagate through the local reverse rules. Fanout adds a second
issue: two paths can contribute to the same earlier variable. Their contributions must be summed,
and that sum rounds too. The explicit accumulation contract in the displayed theorem is what
accounts for this shared-variable case. Merely bounding each local VJP in isolation doesn't cover
it.

# Optimizer Arithmetic

For parameters $`\theta`, gradient $`g`, and learning rate $`\eta`, SGD computes

$$`\theta^+=\theta-\eta g`.

If $`\widehat\theta`, $`\widehat\eta`, and $`\widehat g` approximate their ideal values, the next
parameter bound combines the old parameter error, learning-rate error, gradient error, and local
multiplication/subtraction rounding.

Momentum adds a state recurrence. AdamW, as introduced by {Informal.citet adamw2019}[], adds first
and second moments, bias correction, square root, division, and decoupled weight decay. Rather than
hard-code each optimizer into the graph theorem, TorchLean uses `OptimizerContract`. A contract
supplies:

- ideal and rounded optimizer states
- an approximation relation for the state
- ideal and rounded update functions
- an error transformer for the new state and parameters
- a theorem that the transformer is sound

Our generic theorem takes one parameter gradient from reverse mode and passes it through an
optimizer satisfying that interface. Its statement carries the graph, optimizer contract, exact and
runtime states, and their error bounds into a conclusion about the updated state:

```lean
-- Check that an optimizer consumes the gradient bound
-- derived from this graph.
example :=
  @NF.update_approx_graphData
```

SGD, momentum SGD, and AdamW reuse that same graph theorem.

For an SGD step with exact learning rate, a gradient discrepancy of size $`\varepsilon_g`
contributes at most $`|\eta|\varepsilon_g` before the update's own rounding is counted. Read
practically, the optimizer layer translates gradient accuracy into parameter accuracy. It doesn't
say that the updated loss decreases. A descent argument needs properties of the loss and a suitable
step size, while a multi-step numerical argument must also relate the states and gradients supplied
at every later step.

# Runtime Optimizers And Proved Optimizer Contracts

TorchLean's executable training layer covers more than those three numerical contracts.
`TorchLean.Optimizer` implements SGD, momentum SGD, AdaGrad, RMSProp, Adam, AdamW,
Adadelta, projected SGD, and Muon over shape-indexed parameter lists, and `Trainer.Scheduler` adds
the learning-rate schedules. Trainer checkpoints save model state and reject a checkpoint whose
parameter layout doesn't match the model it's loaded into, while CUDA Adam and AdamW moments and
step counters use the separate `Checkpoint.Optimizer.save` and `Checkpoint.Optimizer.load`
interfaces.

We've proved NF numerical contracts for plain SGD, momentum SGD, and AdamW, and AdamW also carries
the positivity data needed for its square-root denominator. Runtime support for the other
optimizers is implemented and tested, but they don't inherit those numerical refinement theorems
automatically.

# Graph Numerical Certificates

On the executable side, the companion to these approximation theorems works over graphs of
registered operators (`TorchLean.IR.Graph`). The example constructs a two-layer MLP from ordinary IR
operations:

```
-- Each arrow below is an operation whose numerical rule
-- must be available.
input [1,2]
  -> matmul [2,3]
  -> add bias [1,3]
  -> ReLU
  -> matmul [3,1]
  -> add bias [1,1]
```

Running the command generates outward-rounded binary32 ranges for every node, binds them to the
selected backend profile, and replays a concrete FloatLib binary32 execution. The same file
certifies and replays a scalar graph, and `tamperedCheck` replaces an addition range with `[0,0]` ,
which certificate validation rejects. Backend-policy checks are part of the certificate machinery,
but these examples don't exercise every policy or failure mode.

Open {src "TorchLeanExamples/DeepDives/Floats/GraphNumericalCertificate.lean"}[
`GraphNumericalCertificate.lean`] and find `MLP.graph`, `MLP.sources`, `MLP.payload`,
`MLP.certificate`, and `MLP.replay`. Change one weight source interval so that it excludes the
payload value. `MLP.replay` then identifies the node whose value escaped the claimed enclosure.

These are artifact checks and concrete replay. Rejecting the changed interval checks that
validation uses the reconstructed ranges when deciding acceptance. An error-width argument has
another premise: the exact real execution must lie in those same ranges. Once both values are
enclosed, their separation is at most the interval width, so a narrow interval is useful for both
enclosure and error, but successful replay alone supplies only the rounded side of this argument.

# Training Error Traces

`traceTrainingStep` records the bounds for inspection. Here's its data layout, with empty arrays
standing in for the bounds a calculation would supply:

```lean
-- An empty report illustrates the data layout and supplies
-- no numerical evidence.
def emptyTrace : NF.TrainingStepTrace where
  optimizerName := "sgd"
  parameterIndex := 0
  forwardErrors := #[]
  backwardErrors := #[]
  gradientError := 0
  parameterError := 0
  optimizerStateErrors := #[]
  assumptions := #[]
```

Numeric fields in the trace store error bounds. `forwardErrors` and `backwardErrors` follow
typed-context order, including the input context and appended node results. `gradientError` and
`parameterError` summarize the parameter selected by `parameterIndex`. And the `assumptions` field
records the numbers the caller supplied, so you can read a displayed bound together with the
conditions it was computed from.

Nothing in the trace depends on the architecture. It identifies the parameter only by
`parameterIndex`, its position in the input context, and propagation needs only the typed graph and
that index. This is a proof-level reporting interface: its bounds are Lean reals, and an executable
dashboard needs a computable representation and a justified conversion. Constructing a
`TrainingStepTrace` record alone doesn't attach proofs to its arbitrary fields.

# Training-Step Bound Assumptions

Applying the combined theorem requires a graph of covered operators, initial approximation bounds,
an optimizer with a numerical contract, and arithmetic assumptions matching the execution.
For the MLP, this includes the order used by each dot product and the branch conditions needed by
the ReLU VJP. A native provider's connection to that arithmetic remains part of the backend
contract.

FloatLib keeps format and rounding as separate parameters in its generic results
{Informal.citep flocq2011}[]. To use one in this MLP, we select the parameters and establish its
hypotheses at the relevant operation, and the graph and optimizer contracts then carry that local
bound to the updated parameters.

# References

- Nicholas J. Higham,
  [*Accuracy and Stability of Numerical Algorithms*](https://doi.org/10.1137/1.9780898718027),
  second edition, for forward error, backward error, and the standard $`\gamma_n` style of
  accumulated rounding analysis.
- Jean-Michel Muller et al.,
  [*Handbook of Floating-Point Arithmetic*](https://doi.org/10.1007/978-3-319-76526-6),
  second edition, for ULPs, exactness, FMA, and reduction behavior.
