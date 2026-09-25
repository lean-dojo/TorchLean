import VersoManual
-- Executable examples use FloatLib directly; the following imports supply TorchLean error bounds.
import FloatLib
import NN.Proofs.RuntimeApprox.FP32
import NN.Proofs.RuntimeApprox.FP32.Layers
import NN.Proofs.RuntimeApprox.FP32.MLP
import NN.Proofs.RuntimeApprox.FP32.CROWN
import NN.Runtime.Autograd.Engine.Cuda.Float32Contract
-- The parity example is the test behind the native-agreement assumption, and the last
-- section evaluates its API directly rather than paraphrasing what the CLI printed.
import NN.Tests.Floats.NativePrimitiveParity
import TorchLeanBlueprint.Bib
import TorchLeanBlueprint.Roles

open Verso.Genre Manual
open Verso.Genre.Manual.InlineLean
open FloatLib.Numerics
open FloatLib.Floats.Formats.Flocq

-- Every theorem quoted below lives in this one namespace, so opening it here keeps the
-- displayed `#check` lines short. Inside the namespace, `R` is the FP32 abbreviation and
-- `toSpec` is the map back to `ℝ`, which is why both names show up in the signatures.
open NN.Proofs.RuntimeApprox.FP32

-- TorchLean specializes rounded-real error bounds at binary32.
open TorchLean.Floats
open TorchLean.Floats.IEEE754

-- The CUDA float32 contract is where the "which provider actually ran this?" question is stated
-- as a hypothesis rather than assumed away, so the last section quotes it directly.
open Runtime.Autograd.Cuda.Float32Contract

-- `caseRows`, `summary`, `sweep` and `nanOracleReport` come from the parity example. They are
-- evaluated below, so the numbers on this page are produced by the same code the CLI runs.
open NN.Tests.Floats.NativePrimitiveParity

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

Suppose a real-valued verifier proves that the winning logit leads a competitor by `0.12`. That is
not yet a statement about Float32 execution. If each rounded logit may move by `0.03`, the winner
may move down while the competitor moves up, leaving only

$$`0.12-0.03-0.03=0.06.`

The remaining margin is positive. With an original margin of `0.04`, however, the same two error
budgets would leave a negative lower bound and would not establish that the winner survives.
These comparisons use exact real arithmetic:

```lean
-- Pay the possible error of both competing logits before
-- testing the remaining margin.
example : (0.12 : ℝ) - 0.03 - 0.03 = 0.06 := by
  norm_num

example : (0.04 : ℝ) - 0.03 - 0.03 < 0 := by
  norm_num
```

The second calculation does not show that the rounded classifier changes its answer. It shows
that the proposed error budget is too large to rule out a change.

To justify the `0.03`, we must choose a Float32 model, derive an operator-level error budget,
compose those bounds through the network, and account for the result in the verifier's final margin.

The direction of each inequality matters. To protect the winner, we need a lower bound on its
rounded logit and an upper bound on the competitor's rounded logit. An absolute error estimate
provides both directions, so it can serve either role. A negative remaining lower bound means
that this argument is inconclusive; it does not select an input that changes the prediction.
This distinction lets a verifier report an insufficient numerical margin without confusing it
with a counterexample to the network's property.

# Binary32 Representation

The error depends on the spacing of representable values. Choose `ExecFloat.Binary 8 23` for
binary32: eight exponent bits and twenty-three stored fraction bits. FloatLib can expose its
encoding as a `UInt32` and decode a finite value as a dyadic rational. In the records below,
`significand` is a nonnegative integer, and `exponent` gives its power-of-two scale. With
`negative := false`, the value is `significand * 2^exponent`; a true `negative` flag negates it.

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

The first record represents one as $`2^{23}\cdot2^{-23}`, so the significand carries 24 bits and the
grid immediately above `1` has spacing $`2^{-23}`: the next float above `1` is `8388609 * 2^-23`,
exactly one step
away. The largest finite float is $`(2^{24}-1)\cdot2^{104}`. The smallest positive float is
$`1\cdot2^{-149}`, and Lean will print that value as an exact rational if asked:

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
example : fexp32 1 = -23 := by
  simp [fexp32, fltExp]

example : fexp32 (-200) = -149 := by
  simp [fexp32, fltExp]
```

`fltExp emin prec` is `fun e => max (e - prec) emin`. Here `e` is the magnitude exponent, and
the result gives the exponent of the grid spacing. In the normal range, subtracting `prec` lets
the spacing track the magnitude. Near zero, the lower limit fixes the spacing at
$`2^{\mathtt{emin}}`, allowing successively smaller subnormal significands on the same grid.

The table compares these exact values with familiar `float32` format metadata. The
`smallest_subnormal` and `nmant` field names are NumPy's; the two libraries do not expose identical
`finfo` interfaces. The printed decimal values approximate the same format limits.

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

The decoded significand is not the stored fraction field. At one, the decoded integer includes
the leading bit that normal binary32 values leave implicit in their encoding. The subnormal
record instead has significand one and the fixed exponent `-149`; it has no additional leading
bit to recover. This explains the apparently different sizes of the printed integers. All four
records use the same interpretation, integer times a power of two, so they can be compared
exactly without relying on how many decimal places a printer chooses.

# Decimal Rounding In Binary32

The familiar binary64 example `0.1 + 0.2 != 0.3` depends on the precision. With the binary32
operations shown here, the sum and the literal round to the same value:

```lean (name := fpsFolklore)
-- Check both equality and exact decoding for this
-- precision-dependent decimal example.
#eval ofFloat32 0.1 + ofFloat32 0.2 == ofFloat32 0.3
#eval Model.toDyadic? (ExecFloat.Binary.toModel
  (ofFloat32 0.1 + ofFloat32 0.2))
#eval Model.toDyadic? (ExecFloat.Binary.toModel
  (ofFloat32 0.3))
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
same grid point as rounding `0.3` directly. This coincidence depends on the chosen precision.

Changing precision therefore requires recomputing the relevant error budgets. Moving from `float32`
to `bfloat16`, which has eight significand bits, can reuse the generic format theory.
Binary32-specific bridges, constants, and composed budgets in this chapter need
new specializations; changing one exponent function does not establish a new native backend bridge.

The two identical dyadic records explain more than the Boolean result alone. They show that the
sum and the converted literal occupy the same binary32 grid point, whose exact value is
`10066330 * 2^-25`. That point is not the rational number three tenths. Here `ofFloat32` imports
Lean's native `Float32` arguments by their interchange words; the decimal literals have already
rounded to binary32. The addition and comparison then use FloatLib's configured operations.
Keeping the conversion in view avoids turning a precision-specific observation into a claim
that decimal fractions have become exactly representable.

# Machine Epsilon And Unit Roundoff

The spacing at one and the unit roundoff are different constants, both sometimes called
"machine epsilon". The following computation distinguishes them:

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

The two bit patterns are $`2^{-24}` and $`2^{-23}`. Adding $`2^{-24}` to `1` gives back `1`, because
the exact sum sits exactly halfway between `1` and its neighbour and ties round to even. Adding
$`2^{-23}` moves to the next float. So $`2^{-23}` is the spacing, reported by NumPy as
`finfo(float32).eps`, while $`2^{-24}` is half the spacing immediately above `1`.
Higham writes the second one $`u` and calls it the unit roundoff; it is half the first. Both
constants appear in TorchLean. The absolute error theorem uses half the local ULP, which also
accounts for the fixed spacing in the subnormal range.

The spacing at `1` is a theorem, not a measurement:

```lean
-- Express one as a radix power so the generic ULP theorem
-- applies directly.
theorem fpsUlpOne : ulp32 1 = 1 / 8388608 := by
  have h : (1 : ℝ) = bpow binaryRadix 0 := by
    simp [bpow]
  rw [ulp32, h, ulp_bpow]
  norm_num [fexp32, fltExp, bpow,
    Radix.toReal, binaryRadix]
```

The useful error theorem applies at an arbitrary exact result, including values that are not
representable. Instantiating it at $`1` alone would have zero actual rounding error:

```lean
-- Instantiate the nearest-rounding theorem at an arbitrary
-- exact real result.
theorem fpsHalfUlp (x : ℝ) :
    |round32 x - x| ≤ ulp32 x / 2 := by
  exact error_bound_ulp (β := binaryRadix)
    (fexp := fexp32) rnd32 x
```

`8388608` is $`2^{23}` ; half that local spacing is $`2^{-24}` . That is the whole relationship
between the two constants, and it is the reason the
error terms in the layer budgets later on are all written as `ulp(...)/2` rather than as a fixed
number.

The absolute half-ULP formulation remains useful where a uniform relative-error formulation
would fail. At the midpoint between zero and the least positive subnormal, rounding to zero
loses the entire nonzero value: the relative error is one. The absolute error is still only
half the subnormal spacing. Thus `fpsHalfUlp` can cover that point with the same statement used
near one, while a relative estimate using the normal-range unit roundoff needs a hypothesis
excluding this underflow situation.

Rounding also breaks associativity. We can see why the evaluation order belongs in an error
bound by adding the same two increments in different orders:

```lean (name := fpsAssoc)
-- Only the parentheses change; each addition still rounds
-- independently.
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

A sum of $`n` floats therefore has no single value, only a value per evaluation order. This is why
the dot-product bound in this chapter is stated about a specific fold rather than about "the sum",
and why a claim about a GPU kernel has to say something about reduction order before it can say
anything about error.

# Rounded Reals And Executable Formats

`FP32` is TorchLean's rounded-real proof model:

```
-- The proof model fixes binary radix, gradual underflow,
-- and nearest-even rounding.
abbrev FP32 := NF binaryRadix fexp32 rnd32
```

Arithmetic is performed exactly over the reals, then rounded to the binary32-precision grid with
gradual underflow and nearest-even rounding. This follows the rounded-real construction of
{Informal.citep flocq2011}[]. An error such as `round(x + y) - (x + y)` can then be bounded with
real
analysis. A bit-level model needs decoding and an operation-agreement theorem to reach the same
comparison.

`FP32` has no NaN, infinity, signed zero, exception flags, or upper exponent cutoff. It therefore
describes rounding error without describing every binary32 behavior. Connecting it to native code
requires a separate account of those cases, whose numerical consequences are discussed by
{Informal.citet goldberg1991}[].

The two ends of the format are a good way to see what "no upper cutoff" costs and what it does not.
At the bottom, the two models agree, because `FP32` keeps the same smallest step:

```lean (name := fpsUlpZero)
-- Zero lies on the fixed subnormal grid, whose spacing
-- remains positive.
#check @ulp32_zero
```

```leanOutput fpsUlpZero (whitespace := lax)
ulp32_zero : ulp32 0 = bpow binaryRadix (-149)
```

An exact magnitude at or below half of $`2^{-149}` rounds to zero under nearest-even rounding.
The example is exactly halfway between zero and the minimum positive subnormal; the tie is resolved
toward zero in both models. This is ordinary gradual-underflow rounding:

```lean (name := fpsUnderflow)
-- The midpoint between zero and the least positive
-- subnormal rounds to even zero.
#eval ExecFloat.Binary.isZero (ofBits32 1 / 2)
```

```leanOutput fpsUnderflow (whitespace := lax)
true
```

At the upper end, adding the largest finite binary32 value to itself produces infinity. The
rounded-real model has no upper cutoff, so it represents the corresponding rounded sum as a finite
real value:

```lean (name := fpsOverflow)
-- Finite operands can overflow, so input finiteness alone
-- is insufficient for a bridge.
#eval ExecFloat.Binary.isInfinite
  (ofBits32 0x7f7fffff + ofBits32 0x7f7fffff)
```

```leanOutput fpsOverflow (whitespace := lax)
true
```

Without an upper exponent cutoff, `FP32` arithmetic has a total, unbounded-exponent meaning.
Connecting it to a network execution
still requires finite-path hypotheses. FloatLib binary32 represents all 32-bit encodings, including
NaN, infinity, subnormals and signed zero; use it when a claim depends on those cases or exact
result bits. The two APIs begin at the
[FP32 proof semantics](https://github.com/lean-dojo/TorchLean/blob/main/NN/Floats/FP32.lean) and
{ref "floats"}[the direct FloatLib representation and arithmetic].

Exact bridge theorems join the models for covered finite-path operations: basic arithmetic, FMA,
square root, order, and min/max. Their qualifications are operation-specific; for example, division
needs a nonzero denominator and a composite expression needs finite intermediate results rather
than merely finite inputs. Executable `log` and `tanh` still need a separate accuracy or refinement
contract. A subnormal result is finite; `0/0`, overflow to infinity, and square root of a negative
value are not paths that the rounded-real theorem silently absorbs.

For a composite expression, finiteness must be available at the point where each bridge is
applied. Finite weights and inputs do not exclude an overflowing product inside a dot product,
even if later terms would cancel it in real arithmetic. Once infinity has appeared, a later
subtraction need not recover the finite real answer. Tracking intermediate ranges is therefore
part of transferring the composed theorem to binary32. It is not enough to bound only the
network's ideal final output.

# Correspondence With Flocq

The float layer follows Flocq's format vocabulary and statement structure
{Informal.citep flocq2011}[]. The corresponding proofs are Lean proofs over Mathlib's reals. The
source docstrings record the following correspondences:

:::table +header
*
  * Flocq (Rocq)
  * TorchLean (Lean 4)
*
  * `Valid_exp`
  * `ValidExp`, whose one field is literally named `flocq_valid`
*
  * `FLT_exp emin prec`
  * `fltExp emin prec`, and `fexp32 = fltExp (-149) 24`
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
  * `ulp`, pinned to binary32 as `ulp32`
*
  * `round`
  * `round`, pinned to binary32 as `round32`
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
  * no analogue in `FP32`; FloatLib binary32 fills that role
:::

Three design choices in that table explain the reusable interface.

The first is following Flocq's generic format rather than defining binary32 directly. Separate
format-specific developments would duplicate much of the grid and rounding theory. In the generic
construction a format is a function
$`\mathtt{fexp}:\mathbb{Z}\to\mathbb{Z}` plus a validity condition, so binary32 is a single
definition and every bound about it is an instance of a bound proved once.

The validity class records the conditions on the exponent function that the generic rounding
proofs need. Its `flocq_valid` field and names such as `negligibleExp` retain the source
terminology, allowing the two definitions to be compared directly.

The rounding mode is a separate argument to `round`; `rnd32` selects nearest-even for this
specialization. Directed interval rounding can use the same format and ULP machinery. The
half-ULP estimate depends on nearest rounding, while the grid alone only determines which values
are representable.

Flocq also contains a bounded-exponent
`binary_float` type with the IEEE-754 special values and proved operations over it; `FP32` has no
counterpart, and that job belongs to FloatLib binary32. Beyond the paper
{Informal.citep flocq2011}[],
Boldo and Melquiond's *Computer Arithmetic and Formal Proofs* (ISTE Press, 2017) gives a book-length
account of Flocq's approach. TorchLean now imports the corresponding generic theory from
`FloatLib.Floats.Formats.Flocq`; its binary32 specialization and network error bounds build on those
Lean definitions and proofs.

The table is also a guide to which choices a specialization must supply. The exponent function
chooses the grid, its validity evidence makes the generic format theorems applicable, and the
rounding-mode evidence determines the error guarantee. Fixing binary radix and 24-bit precision
alone would not justify a half-ULP claim for a directed rounding mode. Conversely, changing
the mode does not require redefining which real values lie on the grid. This separation is
what allows the format facts to support both nearest-rounding analysis and interval endpoints.

# Rounding Error For A Single Operation

Everything above the format is built from a single kind of fact: one arithmetic operation, one
rounding, half an ULP of the exact real result. Here it is for addition.

```lean (name := fpsAtom)
-- This local theorem bounds only the rounding of the
-- addition itself.
#check @FP32.add_approxR
```

```leanOutput fpsAtom (whitespace := lax)
FP32.add_approxR : ∀ (a b : FP32),
  Proofs.RuntimeApprox.approxR (a.val + b.val) (a + b).val
    (Proofs.RuntimeApprox.ApproxTol.absOnly (eps32 (a.val + b.val)))
```

Read `a.val` as the exact real value of the float `a`, and `(a + b).val` as the real value of the
rounded sum. The claim is that they differ by at most `eps32` of the exact sum, which is `ulp32`
of it divided by two, as stated by `fpsHalfUlp`. The library has the same statement
for subtraction, multiplication, division, square root, `exp`, `log`, the trigonometric and
hyperbolic functions, and `abs`.

The tolerance is `eps32` evaluated at the exact real result of the operation. It describes the
rounding introduced by this step; errors already present in its operands must be accounted for
separately when composing operations. A relative form is available separately through
`relative_error_round_ulp`, and it is the better tool when the question is about significant digits
rather than about an absolute margin.

For the transcendental operations, the single-operation statement concerns the NF definition:
evaluate the specified real function and round its result once. It therefore has a local
rounding bound without choosing a polynomial approximation, table, or range-reduction algorithm.
An executable implementation of that function can introduce approximation error before its
final rounding, and the theorem above does not estimate that additional error. This is why the
later tanh network result is meaningful over `FP32` while its executable transcendental bridge
remains a separate obligation.

Even for addition, the local theorem can be conservative. If the exact sum is already a grid
point, its actual rounding error is zero, while the half-ULP allowance can remain positive.
A composed bound is allowed to overestimate in this way. Removing such a term requires an
exactness argument for that operation at those operands, rather than the observation that the
reported budget looks larger than the measured discrepancy.

# Linear Layer Error Bounds

A single output of a linear layer already combines several rounded operations:

$$`y_i=b_i+\sum_{j=0}^{n-1}W_{ij}x_j.`

The runtime weights, bias, and input may begin near their real counterparts. Each product
introduces another rounded result, the dot product accumulates those results in a declared order,
and the bias addition rounds once more. `linearErrorBudget` is the explicit expression obtained by
composing the matrix-vector and final-addition bounds. `approxTensor_linear_fp32` proves that this
expression bounds every output coordinate.

The budget takes error bounds for weights, bias, and input. The theorem requires approximation
evidence for each of those three quantities:

```lean (name := fpsLinBudget)
-- The budget depends on the rounded layer parameters and
-- rounded input.
#check @linearErrorBudget
```

```leanOutput fpsLinBudget (whitespace := lax)
@linearErrorBudget : {inDim outDim : ℕ} → ℝ → ℝ → ℝ →
  Spec.LinearSpec R inDim outDim → TorchLean.Tensor R [inDim] → ℝ
```

```lean (name := fpsLinThm)
-- Three approximation premises feed the bound for the
-- shaped affine output.
#check @approxTensor_linear_fp32
```

```leanOutput fpsLinThm (whitespace := lax)
@approxTensor_linear_fp32 : ∀ {inDim outDim : ℕ}
  {WS : Spec.LinearSpec ℝ inDim outDim} {xS : Spec.SpecTensor [inDim]}
  {WR : Spec.LinearSpec R inDim outDim} {xR : TorchLean.Tensor R [inDim]} {epsW epsb epsx : ℝ},
  Proofs.RuntimeApprox.approxTensor toSpec WS.weights WR.weights epsW →
    Proofs.RuntimeApprox.approxTensor toSpec WS.bias WR.bias epsb →
      Proofs.RuntimeApprox.approxTensor toSpec xS xR epsx →
        Proofs.RuntimeApprox.approxTensor toSpec (Spec.linearSpec WS xS) (Spec.linearSpec WR xR)
          (linearErrorBudget epsW epsb epsx WR xR)
```

The real side uses `LinearSpec ℝ inDim outDim`; the rounded side uses the same tensor
specification at scalar type `R`, the FP32 abbreviation in this proof namespace. `approxTensor`
relates the two tensors componentwise after interpreting the rounded values as reals. Thus the
theorem is about a shaped layer, not an isolated scalar multiply.

The dot-product budget follows the same index list as the runtime fold. At each index `k`, it
adds three contributions to the accumulated error:

- $`(|a_k|+\varepsilon_a)\varepsilon_b+(|b_k|+\varepsilon_b)\varepsilon_a` , a bound on existing
  operand errors,
  including products of the error budgets;
- `ulp(a_k * b_k) / 2`, from rounding the product;
- `ulp(acc + prod) / 2`, from rounding the accumulation.

The fold starts at `ulp(0) / 2`, which by `ulp32_zero` is $`2^{-150}` for binary32, the half-step at
the very bottom of the grid. This is a conservative initial budget: the zero accumulator itself
is exact, so the term does not claim that initializing it incurs an error. The bias addition then
contributes one more
$`\varepsilon_x+\varepsilon_y+\mathtt{ulp}(x+y)/2` term, and `linfNorm` collapses the resulting
per-coordinate tensor of bounds into one number for the layer. The source is
{src "NN/Proofs/RuntimeApprox/NF/Linalg.lean"}[`dotStep` and `dotBound`] and
{src "NN/Proofs/RuntimeApprox/NF/Ops/Elementwise/Core.lean"}[`addBoundTensor`].

Two things follow from that shape. The bound depends on the actual runtime weights and input, not
only on their sizes, because `ulp` is evaluated at the intermediate values; a network whose
activations sit near `1` gets a much smaller budget than one whose activations sit near `10^4`, and
the expression says so. And the bound is tied to the left-to-right fold, which is the formal echo
of the non-associativity experiment above. Reassociate the sum and this is no longer the right
bound for it.

This is also the concrete difference between a proved budget and a tuned tolerance. PyTorch users
will recognize the situation, because it is what `torch.testing.assert_close` is for
{Informal.citep pytorch2019}[] . Its dtype-based default tolerances are engineering defaults rather
than a per-model theorem: nothing derives them from the operations a particular model
performs, and nothing tells you when a deeper network needs a looser value. A caller who has
`linearErrorBudget` instead can inspect it, compare it with a safety margin, and see which of the
weights, the bias, or the input width is responsible when the margin fails.

Input sensitivity and arithmetic rounding are visible separately in a linear layer. With exact,
fixed weights and a coordinatewise input error at most `epsx`, the perturbation in row `i`
is bounded by `epsx` times the sum of the absolute weights in that row. Weight uncertainty
and rounded arithmetic then add further terms. A small final dot product does not make these
contributions small: large positive and negative products can cancel in the value while their
error bounds accumulate. Following the intermediate magnitudes keeps that cancellation from
being mistaken for numerical accuracy.

# Composed Network Error Bounds

The two-layer ReLU theorem feeds the first linear budget through the rounded ReLU rule, then uses
the result as the input budget for the second linear layer. The three-layer tanh theorem repeats the
same pattern through two smooth activations. Both expose their final expressions rather than hiding
them behind an existential tolerance.

```lean (name := fpsMlpBudget)
-- The two-layer expression carries separate errors for both
-- parameter pairs and the input.
#check @reluTwoLayerMlpErrorBudget
```

```leanOutput fpsMlpBudget (whitespace := lax)
@reluTwoLayerMlpErrorBudget : {d0 d1 d2 : ℕ} →
  ℝ → ℝ → ℝ → ℝ → ℝ → Spec.LinearSpec R d0 d1 →
  Spec.LinearSpec R d1 d2 → TorchLean.Tensor R [d0] → ℝ
```

Five real errors go in: weights and bias of each layer, and the input. The composed correctness
theorems are long enough that printing them in full would obscure the point, so the page elaborates
them without displaying the result. Their conclusion has exactly the shape of
`approxTensor_linear_fp32` above, with the network's forward expression in place of the single
`linearSpec` call:

```lean
-- Inspect the composed network statements rather than
-- assuming closure for every architecture.
#check @approxTensor_reluTwoLayerMlp_fp32
#check @tanhMlp3ErrorBudget
#check @approxTensor_tanhMlp3_fp32
```

Read the argument lists and the shape of the composition is visible: the two-layer budget consumes
five errors, the three-layer tanh budget consumes seven, and each additional layer contributes its
own weight and bias term. The explicit real-valued expressions can be analyzed and bounded further.
They are noncomputable
Lean definitions, so printing them as executable numerical certificates needs a separate computable
bound; ULP-dependent expressions also need not be differentiable everywhere.

ReLU is 1-Lipschitz, so it does not amplify existing forward error. This NF implementation still
wraps `max x 0` in `ofReal`, and its generic bound conservatively adds a half-ULP rounding term.
For representable inputs that rounding is exact, but the displayed composed budget does not
automatically remove it. The tanh theorem also accounts for activation rounding.

These are architecture-shaped theorems for `Linear → ReLU → Linear` and
`Linear → tanh → Linear → tanh → Linear`. They demonstrate composition and cover the
corresponding examples; they are not a claim that every model assembled from arbitrary operations
already has an FP32 theorem.

In the displayed three-layer signatures, the seven error arguments correspond to six parameter
tensors and one input tensor. The approximation theorem consumes evidence for each of them.
It first obtains a bound for the first affine output, feeds that into the tanh bound, and uses
the resulting activation error as the next layer's input error. The later bounds therefore
depend on earlier ones; adding independent per-layer tolerances after evaluating the network
would not express the same composition.

These statements concern forward values. ReLU's Lipschitz property remains valid when an input
perturbation crosses zero, so the forward bound does not need a fixed activation pattern.
A derivative theorem asks a different question at zero and needs its own smoothness conditions
or selected-rule interpretation. Nothing in the composed forward budget supplies the missing
analytic derivative at that kink.

# IBP Bounds With Rounding Error

Suppose real IBP proves that an output lies in a box $`[\mathrm{lo},\mathrm{hi}]`, while the rounded
network theorem gives an error `epsOut` at a fixed input. Inflating every output coordinate by that
amount gives

$$`[\mathrm{lo}-\varepsilon_{\mathrm{out}},\;\mathrm{hi}+\varepsilon_{\mathrm{out}}].`

`ibpBound_contains_reluTwoLayerMlp_fp32` proves this construction pointwise for the two-layer
ReLU MLP. It combines the real IBP theorem with `approxTensor_reluTwoLayerMlp_fp32`; its named
budget is `ibpReluTwoLayerErrorBudget`.

```lean (name := fpsIbpBudget)
-- The runtime input remains an argument of this pointwise
-- IBP error budget.
#check @ibpReluTwoLayerErrorBudget
```

```leanOutput fpsIbpBudget (whitespace := lax)
@ibpReluTwoLayerErrorBudget : {inDim hidDim outDim : ℕ} →
  NN.MLTheory.CROWN.TwoLayerMLP R inDim hidDim outDim →
  TorchLean.Tensor R [inDim] → ℝ → ℝ → ℝ → ℝ → ℝ → ℝ
```

```lean
-- Combine real enclosure at the input with the network’s
-- numerical approximation.
#check @ibpBound_contains_reluTwoLayerMlp_fp32
```

The tensor argument makes this budget pointwise in the runtime input. `inflateBoxUniform`
uses one amount across output coordinates; it does not make that amount uniform over the input
box. A box-wide enclosure needs an upper bound that dominates the pointwise budget for every
admissible input. The theorem here establishes the pointwise result and leaves that further
estimate to the application.

The enclosure theorem's hypotheses relate four parameter tensors and the exact and rounded
inputs. Its box-membership premise concerns the exact input `xS`. The rounded input `xR` is
related to it by `ex`, which is then propagated through the network. Consequently, requiring
`xR` to lie in a similarly printed box would not replace the stated premise about `xS`.
An application must identify the intended real input and provide both its membership and
approximation evidence.

To obtain one box for an entire input region, choose a constant that dominates the pointwise
error expression throughout that region. It may be conservative; finding the smallest such
constant is unnecessary. A maximum observed over sampled inputs is not that domination proof.
Bounds on parameter magnitudes, activations, and intermediate sums can instead supply the
inequalities needed to replace the input-dependent expression by a uniform one.

The two margin-transfer lemmas are small enough to read in full, and they are the exact formal
content of the `0.12` calculation this chapter opened with:

```lean (name := leMargin)
-- An upper-threshold proof must reserve room for upward
-- runtime error.
#check @fp32_le_of_real_le_sub_margin
```

```leanOutput leMargin (whitespace := lax)
@fp32_le_of_real_le_sub_margin : ∀ {y t : ℝ} {yR : R}
  {eps : ℝ}, y ≤ t - eps → |toSpec yR - y| ≤ eps →
  toSpec yR ≤ t
```

```lean (name := geMargin)
-- A lower-threshold proof must reserve room for downward
-- runtime error.
#check @fp32_ge_of_real_ge_add_margin
```

```leanOutput geMargin (whitespace := lax)
@fp32_ge_of_real_ge_add_margin : ∀ {y t : ℝ} {yR : R}
  {eps : ℝ}, y ≥ t + eps → |toSpec yR - y| ≤ eps →
  toSpec yR ≥ t
```

For a scalar threshold, `fp32_le_of_real_le_sub_margin` and
`fp32_ge_of_real_ge_add_margin` package the same arithmetic. For a classifier, apply it to both
logits: the true logit may fall by its budget and the competitor may rise by its budget. The real
margin must pay both costs, just as in the `0.12` example at the start.

Both printed threshold lemmas have non-strict conclusions. They allow the rounded value to land
exactly on the threshold after using the entire error budget. That is sufficient for a property
stated with `≤` or `≥`. For a unique winning class, a surviving pairwise margin must instead be
strictly positive, unless a separate argument uses the classifier's tie convention. The final
comparison should therefore match the property being certified, including whether equality
is acceptable.

# Executable Refinement And Native Execution

The result so far concerns the rounded-real `FP32` semantics. For a network whose primitives are
covered, a FloatLib binary32 claim can use the exact finite-path bridges and discharge their
domain and
finiteness hypotheses. The current bridges cover basic arithmetic, FMA, square root, order, and
min/max behavior; they do not yet give an accuracy or refinement theorem connecting executable
`log` or `tanh` to the rounded-real operations. In particular, the tanh MLP theorem above remains a
result about the `FP32` proof model until such a transcendental contract is supplied.

FloatLib also proves correspondence with Lean 4.34's logical native float model. Its native
round-trip theorem covers every value; addition and subtraction correspondence require finite
inputs, while their results may overflow. Its square-root theorem covers every configured input
through native export, which canonicalizes NaNs. These are operation-specific statements about
Lean's logical definitions, with the hypotheses shown in {ref "floats"}[the floating-point chapter].

Agreement with a compiled provider requires a further account of the native program and
compilation path. Here the contract states the assumption and proves its consequences, which is
what the CUDA
float32 contract does. Native results enter as raw 32-bit patterns, and the agreement between those
patterns and the reference model is a named hypothesis:

```lean (name := fpsNativeAgree)
-- Agreement is a proposition about supplied native
-- primitive implementations.
#check @NativePrimitiveAgreement
```

```leanOutput fpsNativeAgree (whitespace := lax)
NativePrimitiveAgreement : NativePrimitiveBits → Prop
```

The five fields, `add_bits`, `mul_bits`, `div_bits`, `fma_bits`, and `sqrt_bits`, require agreement
for the corresponding primitive. They use a relation that permits differences in NaN encodings:

```lean (name := fpsAgreeUpToNaN)
-- The comparison deliberately distinguishes finite bit
-- equality from NaN classification.
#check @AgreeUpToNaN
```

```leanOutput fpsAgreeUpToNaN (whitespace := lax)
AgreeUpToNaN : UInt32 → UInt32 → Prop
```

`AgreeUpToNaN native reference` holds when the bit patterns are equal or both encode NaN. The
recorded parity results below include different NaN encodings for invalid operations, so strict bit
equality would reject those providers. Allowing that difference preserves the finite-result
consequences: if the native result is finite, the NaN alternative is impossible.

```lean (name := fpsAgreeFinite)
-- Finiteness eliminates the alternative in which both
-- patterns merely encode NaN.
#check @AgreeUpToNaN.eq_of_isFinite
```

```leanOutput fpsAgreeFinite (whitespace := lax)
@AgreeUpToNaN.eq_of_isFinite : ∀ {native reference : UInt32},
  AgreeUpToNaN native reference →
    ExecFloat.Binary.isFinite (fromNativeBits native) = true → native = reference
```

So under the assumption, a finite native sum *is* the reference sum, as a value and not only up to a
payload:

```lean (name := fpsNativeEq)
-- Provider agreement and a finite result identify the
-- executable reference sum.
#check @native_add_eq_ieee32_of_isFinite
```

```leanOutput fpsNativeEq (whitespace := lax)
@native_add_eq_ieee32_of_isFinite : ∀ {native : NativePrimitiveBits},
  NativePrimitiveAgreement native →
    ∀ (x y : RefScalar),
      ExecFloat.Binary.isFinite (fromNativeBits (native.addBits x y)) = true →
        fromNativeBits (native.addBits x y) = ExecFloat.add x y
```

And the half-ULP bound follows for the native result, with the same finiteness side condition:

```lean (name := fpsNativeAdd)
-- Transfer the reference rounding bound only after proving
-- that identification.
#check @native_add_abs_error_of_isFinite
```

```leanOutput fpsNativeAdd (whitespace := lax)
@native_add_abs_error_of_isFinite : ∀ {native : NativePrimitiveBits},
  NativePrimitiveAgreement native →
    ∀ (x y : RefScalar),
      ExecFloat.Binary.isFinite (fromNativeBits (native.addBits x y)) = true →
        |(ExecFloat.Binary.toModel (fromNativeBits (native.addBits x y))).toReal -
            ((ExecFloat.Binary.toModel x).toReal + (ExecFloat.Binary.toModel y).toReal)| ≤
          eps32 ((ExecFloat.Binary.toModel x).toReal + (ExecFloat.Binary.toModel y).toReal)
```

`NativePrimitiveAgreement` remains an assumption about the provider's operations. Tests can
exercise instances of its five fields, but a finite sweep does not establish their universal
claims. Fast-math settings can violate them. FMA contraction can also change which expression is
evaluated, and reductions can use different schedules even when each scalar operation satisfies
the contract. Those expression-level choices need their own agreement statements. Work on verified
compilation {Informal.citep boldo2015}[] addresses another part of this connection
between source arithmetic and machine execution.

The `float32_semantics` example described in the floating-point chapters is regression evidence for
one execution path; it is not that agreement theorem. A complete deployment claim therefore has a
visible chain:

```
-- The hypotheses must describe the same model, input
-- region, and execution path.
real property
  + FP32 approximation budget
  + finite IEEE bridge for each covered primitive
  + native-provider agreement
  = property of the selected execution path
```

Some applications stop earlier because their theorem is intentionally about the rounded-real
model. That is a legitimate claim as long as it says so. For the construction of the two numerical
representations, read *Floating-Point Semantics*; for the graph and checker side of the argument,
return to *Neural Network Verification*.

The finite-result implication retains every finite bit, including the sign of zero; it is
stronger than equality after conversion to a real number. The only alternative allowed by
`AgreeUpToNaN` is that both results are NaNs. A finite result on one side and an infinity or
NaN on the other cannot satisfy the contract. This makes the finite bridge useful for numerical
bounds without asking those bounds to give a real meaning to exceptional values. The provider
agreement premise still has to apply to the primitive and configuration actually used.

# Native Parity Test Results

The parity examples test specific native operations against FloatLib binary32. The following results
show the evidence they provide and how to interpret disagreements.

The examples test two providers. The first is available inside Lean. `Float32` is a native binary32
type, backed by the host's floating-point unit through externs,
which makes it a provider in exactly the sense the contract means. The example
`native_float32_parity` compares it against FloatLib binary32, and its API is what the rest of this
section evaluates. The following block computes one curated case as the page builds:

```lean (name := fpsParityZeros)
-- Signed-zero bits expose distinctions that a real-valued
-- observation would erase.
#eval IO.println (caseRows
  { what := "signed zeros"
  , x := 0x80000000, y := 0x00000000 })
```

```leanOutput fpsParityZeros
  [signed zeros] 0x80000000 0x00000000
    add 0x00000000= mul 0x80000000= div 0x7fc00000= sqrt 0x80000000=
```

The first line is the two operands, `-0` and `+0`. The second is what `Float32` returned for each
primitive, with a trailing `=` when the reference model returned the same bits and a `!` when it did
not. Reading the row is the point of writing operands as bit patterns rather than as decimals:
`-0 + 0 = +0` and `-0 × 0 = -0` have different result signs, which the bits identify unambiguously,
and `-0 / +0` is an invalid operation, which is why a `NaN` shows up in the third column.

The square-root column applies the raw IEEE primitive to the first operand and preserves its
negative zero. This is a statement about that primitive's bits. A tensor API can define a
guarded square root with an explicit zero branch, in which case its semantics also include
the guard and selected result at the boundary. Comparing that API with the row requires
accounting for the branch before invoking the primitive contract. The bit-level example helps
identify which part of such a computation an agreement theorem would cover.

The fifteen curated cases were each chosen for a reason a reader can check: a tie that rounds down
and a tie that rounds up, a subnormal sum, an overflow to infinity, a cancellation that leaves one
ULP, the signed zeros above, the minimum normal halved into the subnormal range, the minimum
subnormal halved into nothing, one third as the canonical inexact quotient, and the invalid
operations `∞/∞` and `√(-1)`. Four primitives per case:

```lean (name := fpsParitySummary)
-- Count the four primitive comparisons for each curated
-- operand pair.
#eval IO.println (summary curatedCases)
```

```leanOutput fpsParitySummary
60 comparisons, all bit-exact
```

The sweep extends the selected examples with operands drawn from a `xorshift32` stream. Its seed
makes the sequence reproducible, so a disagreement can be investigated using the same operands:

```lean (name := fpsParitySweep)
-- A fixed generator seed makes any sampled disagreement
-- reproducible.
#eval IO.println (sweep 20000 1).render
```

```leanOutput fpsParitySweep
  add: 19839/19839 bit-exact
  mul: 19839/19839 bit-exact
  div: 19839/19839 bit-exact
  sqrt: 19839/19839 bit-exact
```

That is the interpreter, running inside the elaborator that built this page. The gap between
20,000 draws and 19,839 tested pairs consists of draws whose operands were `NaN`, which the sweep
skips.

This bit-exact sweep deliberately skips NaN operands. Lean's `Float32` cannot serve as an oracle for
`NaN` payloads at all,
because `Float32.Model.ofBits` is documented to canonicalize every `NaN` input into the canonical
one, while FloatLib binary32 keeps the pattern it was handed:

```lean (name := fpsParityNaN)
-- Inspect conversion before arithmetic when investigating a
-- NaN-payload mismatch.
#eval IO.println (nanOracleReport 0x7fac6de8)
```

```leanOutput fpsParityNaN
NaN 0x7fac6de8: Float32 keeps 0x7fc00000, configured binary32 keeps 0x7fac6de8
```

The disagreement occurs during conversion, before arithmetic. It therefore cannot establish a
mismatch in addition or division; a payload-sensitive arithmetic test must preserve the input bits.

The GPU check uses the same Lean reference stream and the production LibTorch backend.
`scripts/checks/cuda_float32_parity.sh` generates reference bits with `--emit-cases`, builds a C++
regression against the selected SDK and TorchLean library, and checks addition, multiplication,
division, square root, and fused multiply-add:

```terminal
# Use the same SDK and shared library as the CUDA build.
export TORCHLEAN_LIBTORCH_HOME=/path/to/libtorch
export TORCHLEAN_BACKEND_LIBRARY=/path/to/libtorchlean_libtorch.so
scripts/checks/cuda_float32_parity.sh --sweep 200000
```

Every finite result and signed zero must match the reference bits. When both results are NaNs,
`AgreeUpToNaN` permits different signs and payloads; the harness reports those encoding differences
separately. A zero result in place of a nonzero subnormal is a failure, as is a NaN in place of a
finite quotient. Ignoring NaN payloads does not relax those finite-value requirements.

The test reaches the operations TorchLean actually uses. Addition, multiplication, division, and
scalar AXPY pass through the production buffer interface. AXPY evaluates a fused multiply-add
through ATen `addcmul` with its scalar multiplier set to one. Square root has two checks: the raw
IEEE operation and `Buffer.sqrt`, whose selected activation returns zero for nonpositive inputs.
The latter is a different function and needs a different expected result.

Fusing a multiply and an add changes rounding. For binary32 inputs
$`x=1+2^{-23}`, $`y=1+2^{-23}`, and $`z=-(1+2^{-22})`, a fused operation retains $`2^{-46}`;
rounding the product first loses that residual. A second fixture distinguishes a true binary32
FMA from computing in binary64 and then rounding. These cases check the production AXPY path and
the staged Adam update, so passing a standalone arithmetic instruction would not be sufficient.

The regression also observes ATen calls with the caller's autograd mode both enabled and disabled.
TorchLean's native boundary must disable recording during the operation, return tensors without
LibTorch gradient history, and restore the caller's mode. TorchLean's own tape remains responsible
for recording dependencies and traversing local VJPs.

Passing this sweep is evidence about its inputs, the linked SDK, and the tested device. It does not
construct `NativePrimitiveAgreement` for every possible input. Retain the SDK version, source and
binary hashes, device, and precision settings with the result. A reduction also needs an
expression-level argument: primitive agreement does not determine the order in which its partial
sums are added.

# Remaining Deployment Proof Obligations

Applying these results to a complete deployment still requires the following work.

The budgets are pointwise in the input. Turning `ibpReluTwoLayerErrorBudget` into a box-wide
constant needs an upper bound valid over the whole input box; it need not be the least upper
bound. Such a bound is not proved here.

The covered architectures are two. `Linear → ReLU → Linear` and the three-layer tanh network have
composed theorems; other architectures need their own composition and any missing local rules.

The transcendental bridge is missing. `tanh` and `log` have rounded-real bounds and executable
implementations, and no theorem relating the two, so tanh networks stop at the proof model.

Native agreement remains an assumption. The production parity sweep tests concrete inputs and
rejects finite-bit disagreements. Contraction and reduction schedules need additional
expression-level policies. Native conformance remains supported by tests and the declared
toolchain boundary.

# References

The [IEEE 754-2019 standard](https://standards.ieee.org/standard/754-2019.html) defines binary32.
The [Flocq](https://flocq.gitlabpages.inria.fr/) Rocq development supplies the generic-format
vocabulary compared with the Lean definitions above.

- Higham, *Accuracy and Stability of Numerical Algorithms* (2nd ed., SIAM, 2002), the standard
  numerical analysis reference for forward error and stability arguments of the kind TorchLean
  packages into margin-transfer lemmas, and the source of the unit roundoff notation used above.
- Boldo and Melquiond, *Computer Arithmetic and Formal Proofs: Verifying Floating-point Algorithms
  with the Coq System* (ISTE Press, 2017), the book-length account of the library this float layer
  is modelled on.
