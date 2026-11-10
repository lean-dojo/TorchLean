import VersoManual
-- The executable examples use FloatLib directly. The imports below supply TorchLean's bounds.
import FloatLib
import TorchLean.Verify.Floats.FP32
import TorchLean.Verify.RuntimeApprox.NF.CROWN
import TorchLean.Verify.LibTorch.Contract
import TorchLeanBlueprint.Bib
import TorchLeanBlueprint.Roles

open Verso.Genre Manual
open Verso.Genre.Manual.InlineLean
open FloatLib.Numerics
open FloatLib.Floats.Formats.Flocq

-- The bounds quoted below are stated once for every `NF β fexp rnd` with round-to-nearest, in
-- these two namespaces. Opening them keeps the displayed `#check` lines short.
open TorchLean.Roundoff TorchLean.NF

-- TorchLean specializes rounded-real error bounds at binary32.
open TorchLean

-- The native arithmetic contract is where the "which provider actually ran this?" question is
-- stated as a hypothesis rather than assumed away, so the last section quotes it directly.
open LibTorch.Contract

-- Verso compares each `leanOutput` block against the real compiler message. Several signatures
-- below print wider than this file's 100-column limit, so those blocks ask for
-- `whitespace := lax`, which ignores where the expected text happens to be wrapped. The
-- rendered page still shows the message exactly as Lean printed it.
open Lean.Elab.Tactic.GuardMsgs.WhitespaceMode (lax)

open FloatLib.Floats (ExecFloat)
open FloatLib.Floats.ExecFloat.Binary (ofBits32 ofFloat32)
open FloatLib.Floats.Formats.BinaryInterchange (Model FloatFormat)

#doc (Manual) "Float32 Soundness" =>
%%%
tag := "fp32-soundness"
%%%

Suppose a real-valued verifier proves that the winning logit leads a competitor by `0.12`. On its
own, that isn't a statement about Float32 execution. If each rounded logit may move by `0.03`, the
winner may move down while the competitor moves up, leaving only

$$`0.12-0.03-0.03=0.06.`

The remaining margin is positive. With an original margin of `0.04`, however, the same two error
budgets would leave a negative lower bound and wouldn't establish that the winner survives. These
comparisons use exact real arithmetic:

```lean
-- Pay the possible error of both competing logits before
-- testing the remaining margin.
example : (0.12 : ℝ) - 0.03 - 0.03 = 0.06 := by
  norm_num

example : (0.04 : ℝ) - 0.03 - 0.03 < 0 := by
  norm_num
```

The second calculation doesn't show that the rounded classifier changes its answer. It shows that
the proposed error budget is too large to rule out a change.

To justify the `0.03`, we must choose a Float32 model, derive an operator-level error budget,
compose those bounds through the network, and account for the result in the verifier's final margin.

The direction of each inequality matters. To protect the winner, we need a lower bound on its
rounded logit and an upper bound on the competitor's rounded logit. An absolute error estimate
provides both directions, so it can serve either role. A negative remaining lower bound means that
this argument is inconclusive: it doesn't select an input that changes the prediction. This
distinction lets a verifier report an insufficient numerical margin without confusing it with a
counterexample to the network's property.

# Binary32 Representation

The error depends on the spacing of representable values. Choose `ExecFloat.Binary 8 23` for
binary32: eight exponent bits and twenty-three stored fraction bits. FloatLib can expose its
encoding as a `UInt32` and decode a finite value as a dyadic rational. In the records below,
`significand` is a nonnegative integer, and `exponent` gives its power-of-two scale. With
`negative := false`, the value is `significand * 2^exponent`, and a true `negative` flag negates
it.

```lean (name := fpsGrid)
-- Decode exact grid points at one and at both ends of the
-- finite binary32 range.
-- The word 0x3f800000 encodes one.
#eval Model.toDyadic? (ExecFloat.Binary.toModel
  (ofBits32 0x3f800000))
#eval Model.toDyadic? (ExecFloat.Binary.toModel
  (ExecFloat.Binary.nextUp (ofBits32 0x3f800000)))
#eval Model.toDyadic? (ExecFloat.Binary.toModel
  (ofBits32 1))
#eval Model.toDyadic? (ExecFloat.Binary.toModel
  (ofBits32 0x7f7fffff))
```

```leanOutput fpsGrid (whitespace := lax)
some { negative := false, significand := 8388608, exponent := -23 }
```

```leanOutput fpsGrid (whitespace := lax)
some { negative := false, significand := 8388609, exponent := -23 }
```

```leanOutput fpsGrid (whitespace := lax)
some { negative := false, significand := 1, exponent := -149 }
```

```leanOutput fpsGrid (whitespace := lax)
some { negative := false, significand := 16777215, exponent := 104 }
```

In the first record, one appears as $`2^{23}\cdot2^{-23}`, so the significand carries 24 bits and
the grid immediately above `1` has spacing $`2^{-23}`: the next float above `1` is
`8388609 * 2^-23`, exactly one step away. The largest finite float is $`(2^{24}-1)\cdot2^{104}`. At
the other end, the smallest positive float is $`1\cdot2^{-149}`, which Lean will print as an exact
rational if asked:

```lean (name := fpsMinSub)
-- The rational output exposes the minimum subnormal without
-- decimal rounding.
#eval ExecFloat.Binary.toRat? (ExecFloat.Binary.ofBits32 1)
```

```leanOutput fpsMinSub (whitespace := lax)
some (1 / 713623846352979940529142984724747568191373312)
```

The two exponents `-23` and `-149` are precisely the two numbers that define the proof model. Its
exponent function is `fltExp (-149) 24`, and evaluating that function at a large and a small
magnitude recovers the two regimes visible above:

```lean
-- The same exponent policy describes normal spacing and the
-- fixed subnormal floor.
example : Model.fexpOf FloatFormat.binary32 1 = -23 := by
  decide

example :
    Model.fexpOf FloatFormat.binary32 (-200) = -149 := by
  decide
```

`fltExp emin prec` is `fun e => max (e - prec) emin`. Here `e` is the magnitude exponent, and
the result gives the exponent of the grid spacing. In the normal range, subtracting `prec` lets
the spacing track the magnitude. Near zero, the lower limit fixes the spacing at
$`2^{\mathtt{emin}}`, allowing successively smaller subnormal significands on the same grid.

The table compares these exact values with familiar `float32` format metadata. Its
`smallest_subnormal` and `nmant` field names are NumPy's (the two libraries don't expose identical
`finfo` interfaces), and the printed decimal values approximate the same format limits.

:::table +header
*
  * quantity
  * Lean, from FloatLib binary32
  * PyTorch / NumPy `float32`
*
  * step above `1`
  * `8388609 * 2^-23`, so $`2^{-23}`
  * `finfo.eps = 1.1920929e-07`
*
  * smallest positive
  * `1 * 2^-149`
  * `smallest_subnormal = 1e-45`
*
  * largest finite
  * `16777215 * 2^104`
  * `finfo.max = 3.4028235e+38`
*
  * significand bits
  * `significand` reaches $`2^{24}-1`
  * `finfo.nmant = 23` plus the hidden bit
:::

The decoded significand isn't the stored fraction field. At one, the decoded integer includes the
leading bit that normal binary32 values leave implicit in their encoding. The subnormal record
instead has significand one and the fixed exponent `-149`, with no additional leading bit to
recover. This explains the apparently different sizes of the printed integers. All four records use
the same interpretation (integer times a power of two), so they can be compared exactly without
relying on how many decimal places a printer chooses.

# Decimal Rounding In Binary32

Whether the familiar binary64 example `0.1 + 0.2 != 0.3` holds depends on the precision. With the
binary32 operations shown here, the sum and the literal round to the same value:

```lean (name := fpsFolklore)
-- Check both equality and exact decoding for this
-- precision-dependent decimal example.
#eval
  (ofFloat32 0.1 : FloatLib.Floats.Binary32) +
    ofFloat32 0.2 == ofFloat32 0.3
#eval Model.toDyadic? (ExecFloat.Binary.toModel
  ((ofFloat32 0.1 : FloatLib.Floats.Binary32) +
    ofFloat32 0.2))
#eval Model.toDyadic? (ExecFloat.Binary.toModel
  (ofFloat32 0.3 : FloatLib.Floats.Binary32))
```

```leanOutput fpsFolklore (whitespace := lax)
true
```

```leanOutput fpsFolklore (whitespace := lax)
some { negative := false, significand := 10066330, exponent := -25 }
```

```leanOutput fpsFolklore (whitespace := lax)
some { negative := false, significand := 10066330, exponent := -25 }
```

PyTorch agrees, for the same reason:

```
>>> # Use binary32 explicitly; the familiar binary64 example
>>> # answers a different question.
>>> import torch
>>> t = lambda v: torch.tensor(v, dtype=torch.float32)
>>> bool(t(0.1) + t(0.2) == t(0.3))
True
```

Rounding `0.1` and `0.2` to 24 bits, adding exactly, and rounding once more happens to land on the
same grid point as rounding `0.3` itself. That coincidence depends on the chosen precision.

Changing precision therefore requires recomputing the relevant error budgets. Moving from `float32`
to `bfloat16` (which has eight significand bits) can reuse the generic format theory. But the
binary32-specific bridges, constants, and composed budgets need new specializations, and changing
one exponent function doesn't establish a new native backend bridge.

The shared grid point `10066330 * 2^-25` isn't the rational number three tenths. Here `ofFloat32`
imports Lean's native `Float32` arguments by their interchange words (the decimal literals have
already rounded to binary32). Then the addition and comparison use FloatLib's configured
operations. Keeping the conversion in view avoids turning a precision-specific observation into a
claim that decimal fractions have become exactly representable.

# Machine Epsilon And Unit Roundoff

The spacing at one and the unit roundoff are different constants, though both are sometimes called
"machine epsilon". Here's a computation that distinguishes them:

```lean (name := fpsEps)
-- Compare a half-spacing tie with an increment of one full
-- spacing.
#eval
  let one := ofBits32 0x3f800000
  one + ofBits32 0x33800000 == one
#eval
  let one := ofBits32 0x3f800000
  one + ofBits32 0x34000000 == one
```

```leanOutput fpsEps (whitespace := lax)
true
```

```leanOutput fpsEps (whitespace := lax)
false
```

The two bit patterns are $`2^{-24}` and $`2^{-23}`. Adding $`2^{-24}` to `1` gives back `1`,
because the exact sum sits exactly halfway between `1` and its neighbour and ties round to even.
Adding $`2^{-23}` moves to the next float. So $`2^{-23}` is the spacing, reported by NumPy as
`finfo(float32).eps`, while $`2^{-24}` is half the spacing immediately above `1`. Higham writes the
second one $`u` and calls it the unit roundoff (it's half the first). Both constants appear in
TorchLean, and the absolute error theorem uses half the local ULP, which also accounts for the
fixed spacing in the subnormal range.

The spacing at `1` is a theorem, not a measurement:

```lean
-- Express one as a radix power so the generic ULP theorem
-- applies directly.
theorem fpsUlpOne :
    Model.ulpAt FloatFormat.binary32 1 = 1 / 8388608 := by
  have h : (1 : ℝ) = bpow binaryRadix 0 := by
    simp [bpow]
  change ulp binaryRadix
    (Model.fexpOf FloatFormat.binary32) 1 = _
  rw [h, ulp_bpow]
  change bpow binaryRadix (-23) = _
  norm_num [bpow, Radix.toReal, binaryRadix]
```

The useful error theorem applies at an arbitrary exact result, including values that are not
representable. Instantiating it at $`1` alone would have zero actual rounding error:

```lean
-- Instantiate the nearest-rounding theorem at an arbitrary
-- exact real result.
theorem fpsHalfUlp (x : ℝ) :
    |Model.roundAt FloatFormat.binary32 x - x| ≤
      Model.ulpAt FloatFormat.binary32 x / 2 := by
  exact error_bound_ulp (β := binaryRadix)
    (fexp := Model.fexpOf FloatFormat.binary32)
    nearestEven x
```

Where a uniform relative-error formulation would fail, the absolute half-ULP formulation remains
useful. At the midpoint between zero and the least positive subnormal, rounding to zero loses the
entire nonzero value: the relative error is one. But the absolute error is still only half the
subnormal spacing. Thus `fpsHalfUlp` can cover that point with the same statement used near one,
while a relative estimate using the normal-range unit roundoff needs a hypothesis excluding this
underflow situation.

Rounding also breaks associativity. We can see why the evaluation order belongs in an error
bound by adding the same two increments in different orders:

```lean (name := fpsAssoc)
-- Only the parentheses change, and each addition still
-- rounds independently.
#eval
  let one := ofBits32 0x3f800000
  let halfUlp := ofBits32 0x33800000
  (one + halfUlp) + halfUlp == one
#eval
  let one := ofBits32 0x3f800000
  let halfUlp := ofBits32 0x33800000
  one + (halfUlp + halfUlp) == one
```

```leanOutput fpsAssoc (whitespace := lax)
true
```

```leanOutput fpsAssoc (whitespace := lax)
false
```

Left to right, each $`2^{-24}` is swallowed separately and the result is `1`. Grouped to the right,
the two halves combine exactly into $`2^{-23}` first and then survive. PyTorch reproduces both
lines:

```
>>> # Repeat both associations with the same binary32
>>> # midpoint increment.
>>> e = t(2.0) ** -24
>>> bool((t(1.0) + e) + e == t(1.0))
True
>>> bool(t(1.0) + (e + e) == t(1.0))
False
```

A sum of $`n` floats therefore has no single value, only a value per evaluation order. That's why
our dot-product bound is stated about a specific fold, not about "the sum", and why a claim about a
GPU kernel has to say something about reduction order before it can say anything about error.

# Rounded Reals And Executable Formats

`FP32` is TorchLean's rounded-real proof model:

```
-- The proof model fixes binary radix, gradual underflow,
-- and nearest-even rounding.
abbrev FP32 :=
  NF binaryRadix (Model.fexpOf FloatFormat.binary32) nearestEven
```

In this model, arithmetic is performed exactly over the reals, then rounded to the
binary32-precision grid with gradual underflow and nearest-even rounding. This follows the
rounded-real construction of {Informal.citep flocq2011}[]. An error such as
`round(x + y) - (x + y)` can then be bounded with real analysis. A bit-level model needs decoding
and an operation-agreement theorem to reach the same comparison.

`FP32` has no NaN, infinity, signed zero, exception flags, or upper exponent cutoff. It therefore
describes rounding error without describing every binary32 behavior. Connecting it to native code
requires a separate account of those cases, whose numerical consequences are discussed by
{Informal.citet goldberg1991}[].

Looking at the two ends of the format is a good way to see what "no upper cutoff" costs and what it
doesn't. At the bottom, the two models agree, because `FP32` keeps the same smallest step:

```lean
-- Specialize the spacing theorem at binary32.
example :
    Model.ulpAt FloatFormat.binary32 0 =
      bpow binaryRadix (-149) := by
  exact ulp_zero_FLT (β := binaryRadix)
    FloatFormat.binary32.minSubnormalExponent
    (FloatFormat.binary32.fracWidth + 1) (by positivity)
```

An exact magnitude at or below half of $`2^{-149}` rounds to zero under nearest-even rounding. Our
example sits exactly halfway between zero and the minimum positive subnormal, and both models
resolve the tie toward zero. This is ordinary gradual-underflow rounding:

```lean (name := fpsUnderflow)
-- The midpoint between zero and the least positive
-- subnormal rounds to even zero.
#eval ExecFloat.Binary.isZero (ofBits32 1 / 2)
```

```leanOutput fpsUnderflow (whitespace := lax)
true
```

At the upper end, adding the largest finite binary32 value to itself produces infinity. But the
rounded-real model has no upper cutoff, so it represents the corresponding rounded sum as a finite
real value:

```lean (name := fpsOverflow)
-- Finite operands can overflow, so input finiteness alone
-- is insufficient for a bridge.
#eval ExecFloat.Binary.isInf
  (ofBits32 0x7f7fffff + ofBits32 0x7f7fffff)
```

```leanOutput fpsOverflow (whitespace := lax)
true
```

Without an upper exponent cutoff, `FP32` arithmetic has a total, unbounded-exponent meaning.
Connecting it to a network execution still requires finite-path hypotheses. FloatLib binary32
represents all 32-bit encodings, including NaN, infinity, subnormals and signed zero. Use it when a
claim depends on those cases or exact result bits. For the two APIs, start at the [FP32 proof
semantics](https://github.com/lean-dojo/TorchLean/blob/main/TorchLean/Verify/Floats/FP32.lean) and
{ref "floats"}[the direct FloatLib representation and arithmetic].

Exact bridge theorems join the models for covered finite-path operations: basic arithmetic, FMA,
square root, order, and min/max. Their qualifications are operation-specific. The total division
bridge, for example, assumes a finite quotient, while the dyadic form states a nonzero-denominator
condition explicitly. A composite expression needs finite intermediate results, not merely finite
inputs. On IEEE formats, the configured `exp`, `log`, `atan`, `asin`, `acos` and `atan2` are
correctly rounded, while the approximate `tanh` still needs a separate accuracy or refinement
contract. A subnormal result is finite, but `0/0`, overflow to infinity, and square root of a
negative value aren't paths that the rounded-real theorem silently absorbs.

For a composite expression, finiteness must be available at the point where each bridge is applied.
Finite weights and inputs don't exclude an overflowing product inside a dot product, even if later
terms would cancel it in real arithmetic. Once infinity has appeared, a later subtraction need not
recover the finite real answer. Tracking intermediate ranges is therefore part of transferring the
composed theorem to binary32, and bounding only the network's ideal final output isn't enough.

# Correspondence With Flocq

The float layer follows Flocq's format vocabulary and statement structure
{Informal.citep flocq2011}[], while the corresponding proofs are Lean proofs over Mathlib's reals.
Its source docstrings record the following correspondences:

:::table +header
*
  * Flocq (Rocq)
  * FloatLib / TorchLean (Lean 4)
*
  * `Valid_exp`
  * `ValidExp`, whose one field is literally named `flocq_valid`
*
  * `FLT_exp emin prec`
  * `fltExp emin prec`; `Model.fexpOf FloatFormat.binary32 = fltExp (-149) 24`
*
  * `FLX_exp prec`
  * `flxExp prec`, the same family without a lower cutoff
*
  * `mag`
  * `magnitude`
*
  * `cexp`
  * `cexp`
*
  * `scaled_mantissa`
  * `scaledMantissa`
*
  * `generic_format`
  * `genericFormat`
*
  * `negligible_exp`
  * `negligibleExp`
*
  * `ulp`
  * `ulp`, with descriptor-based specialization `Model.ulpAt`
*
  * `round`
  * `round`, with nearest-even descriptor-based specialization `Model.roundAt`
*
  * `error_le_half_ulp`
  * `error_bound_ulp`
*
  * `error_le_ulp`
  * `round_abs_error_le_ulp`
*
  * `error_lt_ulp`
  * `round_abs_error_lt_ulp_of_inexact`
*
  * relative-error FLT lemmas
  * `relative_error_round_ulp`, `ulp_div_abs_le_FLT_normal`
*
  * `binary_float`, `Bplus` and friends
  * no analogue in `FP32` (FloatLib binary32 fills that role)
:::

Three design choices in that table explain the reusable interface.

First, the float layer follows Flocq's generic format instead of a standalone binary32 definition.
Separate format-specific developments would duplicate much of the grid and rounding theory. In the
generic construction a format is a function $`\mathtt{fexp}:\mathbb{Z}\to\mathbb{Z}` plus a
validity condition, so binary32 is a single definition and every bound about it is an instance of a
bound proved once.

The validity class records the conditions on the exponent function that the generic rounding proofs
need. Its `flocq_valid` field and names such as `negligibleExp` keep the source terminology, so you
can compare the two definitions side by side.

`round` takes the rounding mode as a separate argument, and `nearestEven` supplies that choice for
this specialization. Directed interval rounding can use the same format and ULP machinery. But the
half-ULP estimate depends on nearest rounding, while the grid alone only determines which values
are representable.

Flocq also contains a bounded-exponent `binary_float` type with the IEEE-754 special values and
proved operations over it. `FP32` has no counterpart (that job belongs to FloatLib binary32).
Beyond the paper {Informal.citep flocq2011}[], Boldo and Melquiond's *Computer Arithmetic and
Formal Proofs* (ISTE Press, 2017) gives a book-length account of Flocq's approach. TorchLean
imports the corresponding generic theory from `FloatLib.Floats.Formats.Flocq`, and its binary32
specialization and network error bounds build on those Lean definitions and proofs.

# Rounding Error For A Single Operation

Everything above the format is built from a single kind of fact: one arithmetic operation, one
rounding, half an ULP of the exact real result. Here it is for addition.

```lean
-- The bound covers only the rounding of the addition
-- itself.
example (a b : FP32) :
    |(a + b).val - (a.val + b.val)| ≤
      Model.halfUlpAt FloatFormat.binary32
        (a.val + b.val) :=
  Model.abs_roundAt_sub_le FloatFormat.binary32
    (a.val + b.val)
```

Read `a.val` as the exact real value of the float `a`, and `(a + b).val` as the real value of the
rounded sum. The claim is that they differ by at most `Model.halfUlpAt FloatFormat.binary32` of
the exact sum, which is `Model.ulpAt FloatFormat.binary32`
of it divided by two, as stated by `fpsHalfUlp`. Subtraction, multiplication, division, square
root, `exp`, `log`, the trigonometric and hyperbolic functions, and `abs` also round their exact
real result once, so `Model.abs_roundAt_sub_le` gives each of them the same bound.

The tolerance is `Model.halfUlpAt FloatFormat.binary32` evaluated at the exact real result of the
operation. It describes the rounding introduced by this step, while errors already present in its
operands must be accounted for separately when composing operations. A relative form is available
separately through `relative_error_round_ulp`, and it's the better tool when the question is about
significant digits, not about an absolute margin.

For the transcendental operations, the single-operation statement concerns the NF definition:
evaluate the specified real function and round its result once. It therefore has a local rounding
bound without choosing a polynomial approximation, table, or range-reduction algorithm. An
executable implementation of that function can introduce approximation error before its final
rounding, and the theorem above doesn't estimate that additional error. This is why the later tanh
network result is meaningful over `FP32` while its executable transcendental bridge remains a
separate obligation.

Even for addition, the local theorem can be conservative. If the exact sum is already a grid point,
its actual rounding error is zero, while the half-ULP allowance can remain positive. A composed
bound is allowed to overestimate in this way. Removing such a term requires an exactness argument
for that operation at those operands, not the observation that the reported budget looks larger
than the measured discrepancy.

# Linear Layer Error Bounds

A single output of a linear layer already combines several rounded operations:

$$`y_i=b_i+\sum_{j=0}^{n-1}W_{ij}x_j.`

Runtime weights, bias, and input may begin near their real counterparts. Each product introduces
another rounded result, the dot product accumulates those results in a declared order, and the bias
addition rounds once more. `linearErrorBudget` spells out the expression obtained by composing the
matrix-vector and final-addition bounds. `approxTensor_linear` proves that this expression bounds
every output coordinate.

The budget takes error bounds for weights, bias, and input, and the theorem requires approximation
evidence for each of those three quantities:

```lean (name := fpsLinBudget)
-- The budget depends on the rounded layer parameters and
-- rounded input.
#check @linearErrorBudget
```

```leanOutput fpsLinBudget (whitespace := lax)
@linearErrorBudget : {β : Radix} → {fexp : ℤ → ℤ} → [ValidExp fexp] → {rnd : ℝ → ℤ} →
  {inDim outDim : ℕ} → ℝ → ℝ → ℝ → Linear (NF β fexp rnd) inDim outDim →
  Tensor (NF β fexp rnd) [inDim] → ℝ
```

```lean (name := fpsLinThm)
-- Three approximation premises feed the bound for the
-- shaped affine output.
#check @approxTensor_linear
```

```leanOutput fpsLinThm (whitespace := lax)
@approxTensor_linear : ∀ {β : Radix} {fexp : ℤ → ℤ} [inst : ValidExp fexp] {rnd : ℝ → ℤ}
  [ValidRndToNearest rnd] {inDim outDim : ℕ}
  {WS : Linear ℝ inDim outDim} {xS : Tensor ℝ [inDim]}
  {WR : Linear (NF β fexp rnd) inDim outDim}
  {xR : Tensor (NF β fexp rnd) [inDim]} {epsW epsb epsx : ℝ},
  ApproxTensor NF.toReal WS.weights WR.weights epsW →
    ApproxTensor NF.toReal WS.bias WR.bias epsb →
      ApproxTensor NF.toReal xS xR epsx →
        ApproxTensor NF.toReal (WS.forward xS) (WR.forward xR)
          (linearErrorBudget epsW epsb epsx WR xR)
```

The real side uses `Linear ℝ inDim outDim`. The rounded side uses the same tensor
specification over `NF β fexp rnd`, the reals rounded by the rule `rnd` onto the grid of the
exponent function `fexp` in radix `β`. `FP32` is this type at binary32, so the theorem applies to
it as stated. `ApproxTensor` relates the two tensors componentwise after interpreting the rounded
values as reals. Thus the theorem is about a shaped layer, not an isolated scalar multiply.

Our dot-product budget follows the same index list as the runtime fold. At each index `k`, it adds
three contributions to the accumulated error:

- $`(|a_k|+\varepsilon_a)\varepsilon_b+(|b_k|+\varepsilon_b)\varepsilon_a`, a bound on existing
  operand errors, including products of the error budgets
- `ulp(a_k * b_k) / 2`, from rounding the product
- `ulp(acc + prod) / 2`, from rounding the accumulation

We start the fold at `ulp(0) / 2`, which by `ulp_zero FloatFormat.binary32` is $`2^{-150}` for
binary32 (the half-step at the very bottom of the grid). That's a conservative initial budget: the
zero accumulator itself is exact, so the term doesn't mean that initializing it incurs an error.
Then the bias addition contributes one more $`\varepsilon_x+\varepsilon_y+\mathtt{ulp}(x+y)/2`
term, and `linfNorm` collapses the resulting per-coordinate tensor of bounds into one number
for the layer. You'll find the source in
{src "TorchLean/Verify/RuntimeApprox/NF/Linalg.lean"}[`dotStep` and `dotBound`] and
{src "TorchLean/Verify/RuntimeApprox/NF/Elementwise.lean"}[`addBoundTensor`].

Two things follow from that shape. First, the bound depends on the actual runtime weights and
input, not only on their sizes, because `ulp` is evaluated at the intermediate values. A network
whose activations sit near `1` gets a much smaller budget than one whose activations sit near
`10^4`, and the expression says so. And the bound is tied to the left-to-right fold (the formal
echo of the non-associativity experiment above). Reassociate the sum and this isn't the right bound
for it.

That's also the concrete difference between a proved budget and a tuned tolerance. PyTorch users
will recognize the situation, because it's what `torch.testing.assert_close` is for
{Informal.citep pytorch2019}[] . Its dtype-based default tolerances are engineering defaults, not a
per-model theorem: nothing derives them from the operations a particular model performs, and
nothing tells you when a deeper network needs a looser value. A caller who has `linearErrorBudget`
instead can inspect it, compare it with a safety margin, and see which of the weights, the bias, or
the input width is responsible when the margin fails.

Input sensitivity and arithmetic rounding are visible separately in a linear layer. With exact,
fixed weights and a coordinatewise input error at most `epsx`, the perturbation in row `i` is
bounded by `epsx` times the sum of the absolute weights in that row. Weight uncertainty and rounded
arithmetic then add further terms. A small final dot product doesn't make these contributions
small: large positive and negative products can cancel in the value while their error bounds
accumulate. Following the intermediate magnitudes keeps that cancellation from being mistaken for
numerical accuracy.

# Composed Network Error Bounds

Our two-layer ReLU theorem feeds the first linear budget through the rounded ReLU rule, then uses
the result as the input budget for the second linear layer. The three-layer tanh theorem repeats
that pattern through two smooth activations, and neither theorem hides its final expression behind
an existential tolerance.

```lean (name := fpsMlpBudget)
-- The two-layer expression carries separate errors for both
-- parameter pairs and the input.
#check @reluTwoLayerMLPErrorBudget
```

```leanOutput fpsMlpBudget (whitespace := lax)
@reluTwoLayerMLPErrorBudget : {β : Radix} → {fexp : ℤ → ℤ} → [ValidExp fexp] →
  {rnd : ℝ → ℤ} → {d0 d1 d2 : ℕ} → ℝ → ℝ → ℝ → ℝ → ℝ → Linear (NF β fexp rnd) d0 d1 →
  Linear (NF β fexp rnd) d1 d2 → Tensor (NF β fexp rnd) [d0] → ℝ
```

Five real errors go into this budget: weights and bias of each layer, and the input. Printing the
composed correctness theorems in full would obscure the point (they're long), so the page
elaborates them without displaying the result. Their conclusion has the same shape as
`approxTensor_linear` above, with the network's forward expression in place of the single
`Linear.forward` call:

```lean
-- Inspect the composed network statements rather than
-- assuming closure for every architecture.
#check @approxTensor_reluTwoLayerMLP
#check @tanhMLP3ErrorBudget
#check @approxTensor_tanhMLP3
```

Read the argument lists and the shape of the composition is visible: the two-layer budget consumes
five errors, a weight and a bias error per layer and the input error. The three-layer tanh budget
takes seven in the same way, and the error of each hidden layer, passed through its tanh, is the
input error of the next. You can analyze these explicit real-valued expressions and bound them
further. They're noncomputable Lean definitions, though, so printing them as executable numerical
certificates needs a separate computable bound. ULP-dependent expressions also need not be
differentiable everywhere.

ReLU is 1-Lipschitz, so it doesn't amplify existing forward error. Even so, this NF implementation
wraps `max x 0` in `ofReal`, and its generic bound conservatively adds a half-ULP rounding term.
For representable inputs that rounding is exact, but the displayed composed budget doesn't remove
it automatically. And the tanh theorem accounts for activation rounding too.

These are architecture-shaped theorems for `Linear → ReLU → Linear` and
`Linear → tanh → Linear → tanh → Linear`. They demonstrate composition and cover the corresponding
examples, but they don't give an FP32 theorem to every model assembled from arbitrary operations.

In the three-layer statements, the seven error arguments correspond to six parameter tensors and
one input tensor, and the approximation theorem consumes evidence for each of them. Affine stages
consume input and parameter errors. Each tanh stage turns the error `e` of its pre-activation into
`min e 2 + ulp(tanh a) / 2` per coordinate: `tanh` is 1-Lipschitz, and no two of its values differ
by more than 2. So small input errors pass through tanh as they pass through ReLU, plus one
rounding term, and exact inputs cost only that rounding term.

These statements concern forward values. ReLU's Lipschitz property remains valid when an input
perturbation crosses zero, so the forward bound doesn't need a fixed activation pattern. A
derivative theorem asks a different question at zero and needs its own smoothness conditions or
selected-rule interpretation. Nothing in the composed forward budget supplies the missing analytic
derivative at that kink.

# IBP Bounds With Rounding Error

Suppose real IBP proves that an output lies in a box $`[\mathrm{lo},\mathrm{hi}]`, while the rounded
network theorem gives an error `epsOut` at a fixed input. Inflating every output coordinate by that
amount gives

$$`[\mathrm{lo}-\varepsilon_{\mathrm{out}},\;\mathrm{hi}+\varepsilon_{\mathrm{out}}].`

`reluTwoLayerMLP_ibp_sound` proves this construction pointwise for the two-layer ReLU MLP.
It combines the real IBP theorem with `approxTensor_reluTwoLayerMLP`, and its named budget is
`ibpReLUTwoLayerMLPErrorBudget`.

```lean (name := fpsIbpBudget)
-- The runtime input remains an argument of this pointwise
-- IBP error budget.
#check @ibpReLUTwoLayerMLPErrorBudget
```

```leanOutput fpsIbpBudget (whitespace := lax)
@ibpReLUTwoLayerMLPErrorBudget : {β : Radix} → {fexp : ℤ → ℤ} → [ValidExp fexp] →
  {rnd : ℝ → ℤ} → {inDim hidDim outDim : ℕ} →
  CROWN.TwoLayerMLP (NF β fexp rnd) inDim hidDim outDim →
  Tensor (NF β fexp rnd) [inDim] → ℝ → ℝ → ℝ → ℝ → ℝ → ℝ
```

```lean
-- Combine real enclosure at the input with the network’s
-- numerical approximation.
#check @reluTwoLayerMLP_ibp_sound
```

The tensor argument makes this budget pointwise in the runtime input. `CROWN.Box.inflate` uses one
amount across output coordinates, but it doesn't make that amount uniform over the input box. A
box-wide enclosure needs an upper bound that dominates the pointwise budget for every admissible
input. Our theorem establishes the pointwise result and leaves that further estimate to the
application.

The enclosure theorem's hypotheses relate four parameter tensors and the exact and rounded inputs.
Its box-membership premise concerns the exact input `xS`, and the rounded input `xR` is related to
it by `ex`, which is then propagated through the network. Consequently, requiring `xR` to lie in a
similarly printed box wouldn't replace the stated premise about `xS`. An application must identify
the intended real input and provide both its membership and approximation evidence.

Bounds on parameter magnitudes, activations, and intermediate sums can supply a uniform budget. A
maximum observed over sampled inputs can't replace that proof.

Both margin-transfer lemmas are small enough to read in full, and they're the exact formal content
of the `0.12` calculation we started with:

```lean (name := leMargin)
-- An upper-threshold proof must reserve room for upward
-- runtime error.
#check @le_of_real_le_sub_margin
```

```leanOutput leMargin (whitespace := lax)
@le_of_real_le_sub_margin : ∀ {y t z eps : ℝ}, y ≤ t - eps → |z - y| ≤ eps → z ≤ t
```

```lean (name := geMargin)
-- A lower-threshold proof must reserve room for downward
-- runtime error.
#check @ge_of_real_ge_add_margin
```

```leanOutput geMargin (whitespace := lax)
@ge_of_real_ge_add_margin : ∀ {y t z eps : ℝ}, y ≥ t + eps → |z - y| ≤ eps → z ≥ t
```

Here `z` is the real value of the rounded output, such as `NF.toReal yR` for a runtime value `yR`,
so both lemmas hold in every format. For a scalar threshold, `le_of_real_le_sub_margin` and
`ge_of_real_ge_add_margin` package the same arithmetic. For a classifier, apply it to both logits:
the true logit may fall by its budget and the competitor may rise by its budget. So the real margin
must pay both costs, just as in the `0.12` example at the start.

Both printed threshold lemmas have non-strict conclusions. They allow the rounded value to land
exactly on the threshold after using the entire error budget. That's sufficient for a property
stated with `≤` or `≥`. For a unique winning class, a surviving pairwise margin must instead be
strictly positive, unless a separate argument uses the classifier's tie convention. The final
comparison should therefore match the property being certified, including whether equality is
acceptable.

# Executable Refinement And Native Execution

So far, the result concerns the rounded-real `FP32` semantics. For a network whose primitives are
covered, a FloatLib binary32 claim can use the exact finite-path bridges and discharge their domain
and finiteness hypotheses. Those bridges cover basic arithmetic, FMA, square root, order, and
min/max behavior. For `exp` and `log`, the configured binary32 versions are FloatLib's correctly
rounded functions: `ExecFloat.Binary.CorrectlyRounded.toReal_exp_eq_roundAt` and
`toReal_log_eq_roundAt` state that a finite input in the domain whose exact result lies below the
overflow threshold rounds that result once. TorchLean's configured inverse trigonometric functions
round once in the same sense (`ConfiguredBinary.toReal_atan`, `toReal_asin`, `toReal_acos` and
`toReal_atan2`). Their real results are at most $`\pi` in magnitude, so a finite input in the
domain never overflows. `roundoff` has no numerical rule for them, so `#roundoff_coverage` records
their routes as unsupported. FloatLib's `tanh` is an approximation without such a theorem, so the
tanh MLP theorem above remains a result about the `FP32` proof model.

FloatLib also proves correspondence with Lean's logical native float model. Its native round-trip
theorem covers every value. Addition and subtraction correspondence require finite inputs, while
their results may overflow. Its square-root theorem covers every configured input through native
export, which canonicalizes NaNs. These are operation-specific statements about Lean's logical
definitions, with the hypotheses shown in {ref "floats"}[the floating-point chapter].

Agreement with a compiled provider requires a further account of the native program and
compilation path. Here the contract states the assumption and proves its consequences, which is
what the native arithmetic contract does. A native kernel returns a bit word, which enters as a
value of the format, and its agreement with one of FloatLib's operations is a named hypothesis.
For binary32 addition it unfolds as follows:

```lean (name := fpsNativeAgree)
-- Agreement is a proposition about one supplied native
-- function and one of FloatLib's operations.
example (native : FloatLib.Floats.Binary32 →
      FloatLib.Floats.Binary32 → FloatLib.Floats.Binary32 →
      FloatLib.Floats.Binary32) :
    Agreement .add native ↔
      ∀ x y z, EqUpToNaN (native x y z) (x + y) :=
  Iff.rfl
```

`Agreement operation native` is stated in every format of FloatLib's configured binary
family, binary32 and binary64 among them, for FloatLib's addition, subtraction, multiplication,
division, square root and fused multiply-add. The native function takes three operands, of which
an operation reads as many as its arity. The comparison uses a relation that permits differences
in NaN encodings:

```lean (name := fpsAgreeUpToNaN)
-- The comparison deliberately distinguishes equal values
-- from NaN classification.
#check @EqUpToNaN
```

```leanOutput fpsAgreeUpToNaN (whitespace := lax)
@EqUpToNaN : {format : FloatFormat} →
  {plan : FloatLib.Floats.Formats.BinaryInterchange.Configured.StoragePlan format} →
    {code : Type} →
      [ExecFloat.ModelCodec plan (Model format) code] →
        ExecFloat (FloatLib.Floats.Formats.BinaryInterchange.Configured.Family format code plan) →
          ExecFloat
            (FloatLib.Floats.Formats.BinaryInterchange.Configured.Family format code plan) →
            Prop
```

`EqUpToNaN native reference` holds when the two values are equal, which for values of one format
means equal bit words, or both are NaNs. The recorded parity results include different NaN
encodings for invalid operations, so strict bit equality would reject those providers. Allowing
that difference preserves the finite-result consequences: if the native result is finite, the NaN
alternative is impossible.

```lean (name := fpsAgreeFinite)
-- Finiteness eliminates the alternative in which both
-- values are merely NaNs.
#check @EqUpToNaN.eq_of_isFinite
```

```leanOutput fpsAgreeFinite (whitespace := lax)
@EqUpToNaN.eq_of_isFinite : ∀ {format : FloatFormat}
  {plan : FloatLib.Floats.Formats.BinaryInterchange.Configured.StoragePlan format} {code : Type}
  [inst : ExecFloat.ModelCodec plan (Model format) code]
  {native reference :
    ExecFloat (FloatLib.Floats.Formats.BinaryInterchange.Configured.Family format code plan)},
  EqUpToNaN native reference → ExecFloat.Binary.isFinite native = true → native = reference
```

So under the assumption, a finite native sum *is* FloatLib's sum, as a value and not only up to a
payload:

```lean (name := fpsNativeEq)
-- Provider agreement and a finite result identify
-- FloatLib's sum.
example (native : FloatLib.Floats.Binary32 →
      FloatLib.Floats.Binary32 → FloatLib.Floats.Binary32 →
      FloatLib.Floats.Binary32)
    (h : Agreement .add native)
    (x y z : FloatLib.Floats.Binary32)
    (hresult :
      ExecFloat.Binary.isFinite (native x y z) = true) :
    native x y z = x + y :=
  h.eq_of_isFinite x y z hresult
```

And the half-ULP bound follows for the native result in every IEEE format, with the same
finiteness side condition and the operation's `Domain`, which asks for a finite divisor in a
division and a finite operand that is zero or has sign bit `false` in a square root. For binary64
multiplication:

```lean (name := fpsNativeBound)
-- Transfer the reference rounding bound only after
-- that identification.
example (native : FloatLib.Floats.Binary64 →
      FloatLib.Floats.Binary64 → FloatLib.Floats.Binary64 →
      FloatLib.Floats.Binary64)
    (h : Agreement .mul native)
    (x y z : FloatLib.Floats.Binary64)
    (hresult :
      ExecFloat.Binary.isFinite (native x y z) = true) :
    |ExecFloat.Binary.toReal (native x y z) -
        ExecFloat.Binary.toReal x *
          ExecFloat.Binary.toReal y| ≤
      Model.halfUlpAt FloatFormat.binary64
        (ExecFloat.Binary.toReal x *
          ExecFloat.Binary.toReal y) :=
  h.abs_toReal_sub_le trivial hresult
```

`Agreement.abs_toReal_sub_le` covers the six operations at once: `toReal_reference_eq_roundAt`
rounds each exact result once, by FloatLib's `ExecFloat.Binary.toReal_*_eq_roundAt` theorems, and
`Model.abs_roundAt_sub_le` bounds that rounding.

`Agreement` remains an assumption about the provider's operations.
`scripts/checks/cuda.sh parity` exercises it in binary32 for addition, multiplication, division,
square root and fused multiply-add, but a finite sweep doesn't establish the universal claims, and
no sweep covers subtraction or binary64. Fast-math settings can violate them. FMA contraction can
also change which expression is evaluated, and reductions can use different schedules even when
each scalar operation satisfies the contract. Those expression-level choices need their own
agreement statements. For the deterministic CUDA sum, `LibTorch.reduceSumTree` fixes the schedule,
and `LibTorch.abs_toReal_sub_sum_le` bounds a finite native sum that agrees with it. Work on
verified compilation {Informal.citep boldo2015}[] addresses another part of this connection between
source arithmetic and machine execution.

The `float32_semantics` example described in the floating-point chapters is regression evidence for
one execution path, not that agreement theorem. A complete deployment claim therefore has a visible
chain:

```
-- The hypotheses must describe the same model, input
-- region, and execution path.
real property
  + FP32 approximation budget
  + finite IEEE bridge for each covered primitive
  + native-provider agreement
  = property of the selected execution path
```

Some applications stop earlier because their theorem is intentionally about the rounded-real model.
That's a legitimate claim as long as it says so. For the construction of the two numerical
representations, read *Floating-Point Semantics*. For the graph and checker side of the argument,
return to *Neural Network Verification*.

The finite-result implication retains the sign of zero, which conversion to a real number erases.


# Remaining Deployment Proof Obligations

Applying these results to a complete deployment still requires the following work.

These budgets are pointwise in the input. Turning `ibpReLUTwoLayerMLPErrorBudget` into a box-wide
constant needs an upper bound valid over the whole input box (it need not be the least upper
bound). We don't prove such a bound here.

Here the architecture-specific examples are `Linear → ReLU → Linear` and a three-layer tanh
network. For other models, the {ref "runtime-approximation"}[runtime approximation chapter] states
`RevGraph.backprop_approx` for a `RevGraph` whose nodes carry proved local error bounds. No model
is lowered to a `RevGraph`, so applying it to a model needs the graph, its local rules and a proof
that the graph computes the model.

Our tanh network theorem doesn't cover the executable `tanh` (one of FloatLib's approximation
kernels, with no error theorem), and the correctly rounded `exp` and `log` don't supply that
bridge.

Native agreement remains an assumption. The production parity sweep tests concrete inputs and
rejects finite-bit disagreements. Contraction and reduction schedules need additional
expression-level policies. Native conformance remains supported by tests and the declared
toolchain boundary.

# References

Binary32 is defined by the [IEEE 754-2019
standard](https://standards.ieee.org/standard/754-2019.html), and the
[Flocq](https://flocq.gitlabpages.inria.fr/) Rocq development supplies the generic-format
vocabulary compared with the Lean definitions above.

- Higham, *Accuracy and Stability of Numerical Algorithms* (2nd ed., SIAM, 2002), the standard
  numerical analysis reference for forward error and stability arguments of the kind TorchLean
  packages into margin-transfer lemmas, and the source of the unit roundoff notation used above.
- Boldo and Melquiond, *Computer Arithmetic and Formal Proofs: Verifying Floating-point Algorithms
  with the Coq System* (ISTE Press, 2017), the book-length account of the library this float layer
  is modelled on.
