import VersoManual
import NN.Floats
import TorchLeanBlueprint.Bib
import TorchLeanBlueprint.Roles

open Verso.Genre Manual
open Verso.Genre.Manual.InlineLean
open TorchLean.Floats

-- `NN.Floats.Float32` re-exports `IEEE32Exec` as an abbreviation one namespace up,
-- so opening the outer and the inner namespace together makes the bare name
-- ambiguous. Hiding the inner one keeps every mention of it pointing at a single
-- constant. Readers writing their own files will hit the same thing.
open TorchLean.Floats.IEEE754 hiding IEEE32Exec
open TorchLean.Floats.IEEE754.Float32Bridge

-- Verso checks `leanOutput` blocks against the real compiler message. A few of the
-- records and signatures below print wider than this file's 100-column limit, so
-- those blocks ask for `whitespace := lax`, which ignores where the expected text
-- was wrapped. The rendered page still shows the message exactly as Lean printed it.
open Lean.Elab.Tactic.GuardMsgs.WhitespaceMode (lax)

#doc (Manual) "Floating-Point Semantics" =>
%%%
tag := "floats"
%%%

Neural-network papers write equations over real numbers, while hardware evaluates a sequence of
finite operations. Rounding, overflow, underflow, subnormals, infinities, and NaNs can therefore
change the value computed by a model. TorchLean gives these effects explicit mathematical and
executable meanings instead of treating them as details of the runtime.

# Standalone Numerical Library

The floating-point development is a reusable numerical library inside the TorchLean package. A
downstream Lean project can depend on TorchLean and import only:

```
-- The scalar arithmetic library can be imported without the
-- neural-network API.
import NN.Floats
open TorchLean.Floats
```

That import does not bring in tensors, model definitions, autograd, CUDA, certificate checkers, or
external numerical tools. It provides the generic format and rounding theory, the binary32-precision
rounded-real model, executable IEEE binary32 arithmetic, proved interval rounders, and scalar affine
quantization. Narrower imports such as `NN.Floats.NeuralFloat`, `NN.Floats.FP32`,
`NN.Floats.IEEEExec`, and `NN.Floats.Interval` are useful when a file needs only one layer.

Connections to the rest of TorchLean point in the other direction. `NN.Spec.Quantization` lifts the
scalar quantizer to shape-indexed tensors. `NN.Proofs.RuntimeApprox.FP32` connects rounded-real FP32
semantics to runtime-approximation proofs. Arb-based transcendental checks are optional and require
the explicit import `NN.Floats.Arb`. The numerical core therefore remains usable without adopting
TorchLean's model or runtime APIs.

The `lean` blocks in the rest of this chapter are elaborated while the page is built; their
`leanOutput` blocks are checked against Lean messages. Plain-text sketches illustrate the
interfaces. The page has already
opened `TorchLean.Floats`, `TorchLean.Floats.IEEE754`, and
`TorchLean.Floats.IEEE754.Float32Bridge`, so the examples use short names. One wrinkle is worth
copying into your own file: `NN.Floats.Float32` re-exports `IEEE32Exec` as an abbreviation one
namespace up, so opening the outer and inner namespaces together makes the bare name ambiguous.
Writing `open TorchLean.Floats.IEEE754 hiding IEEE32Exec` settles it.

# Nonassociative Binary32 Addition

Take the three exact real numbers

$$`a=2^{24},\qquad b=1,\qquad c=-2^{24}`.

Over the reals, both parenthesizations are equal:

$$`(a+b)+c=a+(b+c)=1`.

Binary32 has 24 bits of significand precision. Around $`2^{24}`, adjacent representable numbers are
two units apart. The exact value $`2^{24}+1` sits halfway between them, and ties-to-even rounds it
back to $`2^{24}`. Therefore

$$`
\operatorname{fl}(\operatorname{fl}(a+b)+c)=0,
\qquad
\operatorname{fl}(a+\operatorname{fl}(b+c))=1.
`

TorchLean runs this example from the bit patterns themselves. `IEEE32Exec` is the executable
binary32 model built later in the chapter; for now all that matters is that `ofBits` reads a raw
32-bit encoding and `add` is TorchLean's own addition algorithm rather than a call out to the host
CPU.

```lean
-- Use exact bit patterns so decimal conversion cannot
-- affect the associativity example.
def bigPower : IEEE32Exec :=     --  2^24
  IEEE32Exec.ofBits 0x4b800000

def unit : IEEE32Exec :=         --  1
  IEEE32Exec.ofBits 0x3f800000

def negBigPower : IEEE32Exec :=  -- -2^24
  IEEE32Exec.ofBits 0xcb800000
```

Adding left to right loses the $`1`:

```lean (name := leftAssoc)
-- Round the large positive partial sum before cancellation.
#eval ((bigPower.add unit).add negBigPower).bits
```
```leanOutput leftAssoc
0
```

Adding $`1` to $`-2^{24}` first keeps it, because that sum is exactly representable:

```lean (name := rightAssoc)
-- Cancel the negative large operand only after forming the
-- other partial sum.
#eval (bigPower.add (unit.add negBigPower)).bits
```
```leanOutput rightAssoc
1065353216
```

Those are encodings, not values: `0` is `0x00000000`, which is $`+0`, and `1065353216` is
`0x3f800000`, which is $`1`. Thus even finite binary32 addition is not associative, a central
lesson in {Informal.citet goldberg1991}[] and the reason a reduction has to declare its evaluation
order before anybody can prove an error bound for it.

The tie is what makes this particular case interesting, so it is worth watching it disappear.
Replace the $`1` by $`2` and the two groupings agree, because $`2^{24}+2` lands exactly on the
grid:

```lean
-- Two is one representable spacing above the large positive
-- operand.
def two : IEEE32Exec :=
  IEEE32Exec.ofBits 0x40000000
```
```lean (name := twoLeft)
-- Check whether the larger increment survives left
-- association.
#eval ((bigPower.add two).add negBigPower).bits
```
```leanOutput twoLeft
1073741824
```
```lean (name := twoRight)
-- Compare the same increment under right association.
#eval (bigPower.add (two.add negBigPower)).bits
```
```leanOutput twoRight
1073741824
```

Replace it by $`0.5` and they agree again, this time because both groupings lose the small addend
entirely:

```lean
-- Half a unit probes a different rounding tie from the
-- integer increments.
def oneHalf : IEEE32Exec :=
  IEEE32Exec.ofBits 0x3f000000
```
```lean (name := halfLeft)
-- The first addition decides whether this small
-- contribution survives.
#eval ((bigPower.add oneHalf).add negBigPower).bits
```
```leanOutput halfLeft
0
```
```lean (name := halfRight)
-- The negative partial sum has its own nearest-even tie to
-- resolve.
#eval (bigPower.add (oneHalf.add negBigPower)).bits
```
```leanOutput halfRight
0
```

Among these three increments, only $`b=1` changes the answer with grouping. Its positive-side
addition is a tie, while $`1-2^{24}` is exactly representable. The $`b=0.5` negative-side addition
is also a tie, but rounds to $`-2^{24}`.

The same calculation can now be examined at three levels. The spacing of the binary32 grid and
the nearest-even rule explain the rounded value. An executable model must also determine its bits
and handle signs, subnormals, infinities, and NaNs. A backend contract must then identify the
parenthesization or fused instruction used by the actual reduction.

TorchLean separates these levels so a proof about grid spacing need not also reason about device
dispatch:

```
-- The arrows identify mathematical dependencies between the
-- scalar models.
format and rounding theory
        ↓
rounded-real arithmetic
        ↓
binary32-precision rounded-real specialization
        ↓
executable IEEE bit patterns
        ↓
CPU, CUDA, and external providers
```

The diagram starts with the mathematical format, specializes it to binary32 precision, and then
connects that model to bit patterns and runtime providers. Each connection needs an agreement
statement; placing two representations next to each other does not establish one.

The integer printed after each evaluation is the stored word, not the represented numerical value.
For example, `1073741824` is the binary32 encoding of two. Comparing words makes the experiment
independent of how many decimal digits a pretty printer chooses. The examples also show why an
informal statement such as “small terms disappear next to large terms” is incomplete. Whether a
term survives depends on the spacing on that side of the large value and, at a midpoint, on the
tie rule. Changing one to two or to one half probes those separate decisions.

This matters when translating a mathematical formula into tensor code. Parentheses select which
partial result is rounded first. A compiler or parallel reducer that changes the association may
still evaluate the same real polynomial while producing a different bit pattern. The later
reduction contracts therefore describe an evaluation schedule, or quantify over an allowed family
of schedules, rather than treating associativity of real addition as a floating-point law.

# Flocq And `NeuralFloat`

TorchLean's generic floating-point theory was inspired by
[Flocq](https://flocq.gitlabpages.inria.fr/), the mature floating-point library developed in Rocq
(formerly Coq) by Sylvie Boldo, Guillaume Melquiond, and contributors
{Informal.citep flocq2011}[]. Flocq separates a floating-point *format* from a *rounding rule*.

A format tells us which real numbers are available. A rounding rule chooses one of those numbers.
The same binary32 format can round downward, upward, toward zero, or to nearest-even. Conversely,
the same nearest-even idea can be used with binary16, binary32, a fixed-point grid, or an
experimental low-precision format. Once those choices are separated, theorems about monotonicity,
exactness, neighboring values, and ULP error can be reused.

TorchLean follows that organization but is not a port of Flocq. The definitions and proofs are
native Lean. The library concentrates on the theory needed by tensor semantics, interval
enclosures, quantization, and runtime approximation. Flocq remains broader in areas such as its
effective-operation infrastructure and its collection of verified floating-point algorithms.
TorchLean adds a separate executable binary32 development and graph-level numerical machinery aimed
at machine learning.

The correspondence is useful when reading either library:

:::table +header
*
  * Mathematical role
  * Flocq vocabulary
  * TorchLean vocabulary
*
  * radix power $`\beta^e`
  * `bpow`
  * `neuralBpow`
*
  * mantissa/exponent value
  * `F2R`-style representation
  * `NeuralFloat`, `neuralToReal`
*
  * exponent policy
  * `fexp`
  * `fexp` with `NeuralValidExp`
*
  * representable grid
  * `generic_format`
  * `neuralGenericFormat`
*
  * fixed, unbounded, gradual formats
  * `FIX`, `FLX`, `FLT`
  * `FIXExp`, `FLXExp`, `FLTExp`
*
  * rounding to the grid
  * `round`
  * `neuralRound`
:::

The `neural` prefix is not claiming that floating-point arithmetic is unique to neural networks.
It marks TorchLean's generic floating layer and avoids colliding with Lean's host `Float`. The
theorems themselves are ordinary numerical analysis and can be used independently of a model.

# Mantissas And Exponents

Before imposing a precision, define a radix-$`\beta` value by an integer mantissa and an integer
exponent:

$$`\operatorname{value}_{\beta}(m,e)=m\beta^e`.

This is the structure `NeuralFloat β`, from
{src "NN/Floats/NeuralFloat/Core.lean"}[`NN/Floats/NeuralFloat/Core.lean`]:

```
-- A raw representation stores integers; format membership
-- is a separate predicate.
structure NeuralFloat (β : NeuralRadix) where
  mantissa : ℤ
  exponent : ℤ
```

For example:

```lean
-- Represent three quarters exactly as an integer times a
-- power of two.
def threeQuarters : NeuralFloat binaryRadix :=
  { mantissa := 3, exponent := -2 }
```

`neuralToReal` sends that pair to a real number, and the value is exactly what the arithmetic says
it is:

```lean
-- Decode the representation over the reals, without a
-- floating-point conversion.
example : neuralToReal threeQuarters = 3 / 4 := by
  norm_num [neuralToReal, neuralBpow, binaryRadix,
    NeuralRadix.toReal, threeQuarters]
```

There is intentionally no field saying “24 bits of precision.” The pair $`(3,-2)` and the pair
$`(6,-3)` denote the same real number, and that is a proof rather than a remark:

```lean
-- A second pair can denote the same real value without
-- being the same record.
def sixEighths : NeuralFloat binaryRadix :=
  { mantissa := 6, exponent := -3 }

example :
    neuralToReal sixEighths =
      neuralToReal threeQuarters := by
  norm_num [neuralToReal, neuralBpow, binaryRadix,
    NeuralRadix.toReal, threeQuarters, sixEighths]
```

A raw mantissa/exponent pair is therefore a representation, not yet a machine format. Keeping it
general lets later proofs normalize representations and reuse the same carrier at different
precisions.

A format is added as a predicate on the real value. Its parameter `fexp` chooses the exponent used
to test representability. Informally,

$$`\operatorname{neuralGenericFormat}(\beta,f_{\rm exp},x)`

means that after scaling $`x` by the exponent selected by `fexp`, the resulting mantissa is an
integer. The exponent policy is where precision and underflow enter.

The two representations of three quarters separate equality of data from equality of meaning.
Multiplying the mantissa by two and reducing the exponent by one leaves the represented real value
unchanged. A theorem about `neuralToReal` can use that equality without claiming that the two
records are structurally equal. This is why a raw mantissa/exponent pair alone is insufficient to
specify a format: the format must say which real values are representable at each magnitude, and
rounding must select one of those values.

# Formats As Exponent Policies

The central format parameter is an exponent function

$$`f_{\mathrm{exp}}:\mathbb Z\to\mathbb Z`.

For a nonzero real $`x`, its magnitude identifies the power of $`\beta` immediately above
$`|x|`. Applying `fexp` to that magnitude gives the canonical exponent at which the mantissa must
be integral. TorchLean's `neuralGenericFormat β fexp x` says precisely that $`x` lies on this grid.

Three standard policies explain most uses:

- `FIXExp emin` always returns `emin`. This is a fixed-point grid with constant spacing
  $`\beta^{e_{\min}}`.
- `FLXExp prec` returns `e - prec`. This models a precision of `prec` radix digits with no lower
  exponent bound.
- `FLTExp emin prec` returns `max (e - prec) emin`. This is the gradual-underflow format: normal
  values receive `prec` digits, while values near zero stay on the fixed subnormal grid
  $`\beta^{e_{\min}}`.

Precision must be positive. The raw integer formulas remain useful inside symbolic theorem
statements, where positivity is carried as a hypothesis. Code reading a format configuration uses
`NeuralFormatPrecision.ofNat?` or `NeuralFormatPrecision.ofInt?`; the checked value supplies valid
FLX, FLT, and FTZ exponent selectors. Thus zero and negative inputs are rejected instead of being
silently reinterpreted through an absolute value.

It helps to see this on a toy system. Take radix two, precision three, and minimum exponent $`-4`.
Between $`1` and $`2`, three-bit numbers are spaced by $`1/4`:

$$`1,\quad 1.25,\quad 1.5,\quad 1.75,\quad 2`.

Near zero, `FLTExp (-4) 3` stops decreasing the exponent, so the spacing becomes the constant
subnormal step $`2^{-4}=1/16`. `FLXExp 3` would continue creating smaller normal scales forever;
`FIXExp (-4)` would use the $`1/16` grid everywhere. These three policies are not unrelated format
implementations. They are three choices for the same exponent function interface.

These policies are ordinary integer functions. At magnitude $`1`, the gradual format chooses
exponent $`-2`, giving the $`1/4` spacing of the example above:

```lean (name := fltNormal)
-- The normal regime chooses spacing from magnitude minus
-- precision.
#eval FLTExp (-4) 3 1
```
```leanOutput fltNormal
-2
```

At magnitude $`-2` it has already reached the floor and returns `emin` rather than `-5`:

```lean (name := fltClamped)
-- Near zero the minimum exponent prevents the spacing from
-- shrinking further.
#eval FLTExp (-4) 3 (-2)
```
```leanOutput fltClamped
-4
```

`FLXExp` has no floor, so its exponent continues to decrease at smaller magnitudes:

```lean (name := flxUnbounded)
-- Removing the lower cutoff lets the spacing continue to
-- follow magnitude.
#eval FLXExp 3 (-2)
```
```leanOutput flxUnbounded
-5
```

and `FIXExp` ignores the magnitude altogether:

```lean (name := fixConstant)
-- Fixed-point spacing ignores the magnitude argument.
#eval FIXExp (-4) 1
```
```leanOutput fixConstant
-4
```

Binary32 uses radix two, precision 24, and the least subnormal exponent $`-149`. The corresponding
exponent policy is visible in the checked definition:

```lean (name := fexp32Print)
-- Inspect the actual precision and lower exponent used by
-- the FP32 specialization.
#print fexp32
```
```leanOutput fexp32Print
def TorchLean.Floats.fexp32 : ℤ → ℤ :=
FLTExp (-149) 24
```

The choice $`-149` is not the minimum *normal* exponent. Binary32 normal numbers begin at
$`2^{-126}`, but the 23 fraction bits extend the gradual-underflow grid down to $`2^{-149}`.
Encoding that fact in
`FLTExp` is what allows the same representability predicate to cover normal and subnormal finite
values. Three evaluations show the transition. Well inside the normal range the mantissa scale
tracks the magnitude:

```lean (name := fexp32Normal)
-- A large magnitude selects a coarser grid.
#eval fexp32 100
```
```leanOutput fexp32Normal
76
```

At magnitude $`-120` the value is still normal, so the exponent continues tracking:

```lean (name := fexp32Subnormal)
-- This smaller magnitude still lies above the spacing
-- floor.
#eval fexp32 (-120)
```
```leanOutput fexp32Subnormal
-144
```
At magnitude $`-149` the subnormal floor fixes the grid:

```lean (name := fexp32Floor)
-- At the lower end, the exponent policy returns the
-- subnormal spacing floor.
#eval fexp32 (-149)
```
```leanOutput fexp32Floor
-149
```

The benefit of the generic definition is visible in theorem statements. Monotonicity of rounding,
the half-ULP nearest-rounding bound, and fixed-grid exactness do not need separate proofs for every
precision. A binary32 theorem specializes the generic result by supplying `binaryRadix`, `fexp32`,
and nearest-even rounding.

The arguments to the exponent policy are magnitude exponents, not stored IEEE exponent fields.
For example, the output `76` from `fexp32 100` means that values at that magnitude use grid spacing
$`2^{76}`. The outputs near the lower end show when the maximum in `FLTExp` selects the fixed
floor instead of magnitude minus precision. No special-value encoding is involved in these
calculations. They describe a real-valued grid that can later be related to the finite encodings
of a bit-level format.

Precision and exponent validity also play different roles. A positive precision gives the intended
number of significant radix digits. The `NeuralValidExp` laws justify how the grid behaves when a
rounding step moves a value across a magnitude boundary. Supplying an arbitrary function of the
right type is enough to write an expression, but the generic rounding theorems require those laws.
The format constructors provide them for the supported policies.

# Rounding Rules

A format determines the available real values; a rounding rule selects one to replace an exact
input. The same binary format supports rounding toward negative infinity, toward positive
infinity, toward zero, and to nearest with ties to even.

Conceptually, TorchLean computes

$$`\operatorname{round}_{\beta,f,r}(x)
   = \beta^{e}\,r(x\beta^{-e}),
   \qquad e=f(\operatorname{mag}_{\beta}(x))`,

Here $`\beta` is the radix, $`f` is the exponent policy, and
$`\operatorname{mag}_{\beta}(x)` supplies the input's magnitude. Multiplying by $`\beta^{-e}`
expresses the input in units of the selected grid spacing. The function
$`r:\mathbb R\to\mathbb Z` rounds that scaled mantissa to an integer, and multiplication by
$`\beta^e` restores the scale. The type class
`NeuralValidRnd r` records the order properties required of that integer rounder.
`NeuralRoundingMode` packages the standard choices used by APIs.

Return to the toy three-bit format and round $`1.375`. Its canonical exponent in this binade is
$`-2`, so the scaled mantissa is

$$`1.375\cdot2^2=5.5`.

Rounding downward chooses mantissa $`5`, giving $`1.25`. Rounding upward chooses $`6`, giving
$`1.5`. Nearest-even also chooses $`6`, because the two candidates are equally distant and $`6` is
even. The format selected the scale $`2^{-2}`; the rounding mode selected the integer mantissa.

The separation gives us reusable theorems:

- rounding a representable value leaves it unchanged;
- every rounded value belongs to the format;
- directed rounding returns the greatest representable value below, or least one above, the input;
- nearest rounding has error at most half an ULP;
- rounding is monotone;
- an initial round-to-odd can prevent a later nearest-even double-rounding error.

The abstract definition is excellent for proofs: it says what rounding means without committing to
an implementation. `NN.Floats.Calc` supplies the representation-level middle layer. It brackets an
exact
value between representable neighbors, applies the rounding decision, and returns mantissa/exponent
data. Calculations starting from an arbitrary Lean real remain noncomputable; executable bit-level
arithmetic is supplied separately by `IEEE32Exec`. `FP32.round_eq_computed` proves that this
calculation agrees with the
abstract binary32 rounder.

The toy rounding calculation has one discrete step: rounding the scaled mantissa from
`5.5` to an integer. Everything before and after that step is exact real scaling. Downward rounding
chooses five, while upward and nearest-even rounding choose six, giving different points on the
same grid. This separation makes the half-ULP argument reusable: a half-unit error on the integer
grid becomes half a grid spacing after rescaling. Directed rounding uses a different integer
inequality but the same representation machinery.

# `NF`: Arithmetic In A Declared Format

The generic theorems above discuss individual real values and rounding functions. Neural-network
proofs need an object on which `+`, `*`, division, activations, and reductions can be written
without repeating all format parameters. That object is

```
-- The scalar type records the radix, exponent policy, and
-- integer rounding rule.
NF β fexp rnd
```

An `NF` value carries a real number, while its type fixes the radix, format, and rounding rule.
Primitive arithmetic computes the exact real operation and rounds the result back to the declared
grid. Schematically,

$$`\operatorname{NF.add}(a,b)
  = \operatorname{round}_{\beta,f,r}(a_{\mathbb R}+b_{\mathbb R})`.

Error proofs compare an ideal value with a rounded one. Storing the real projection directly lets
them express the difference as

$$`|\operatorname{NF.toReal}(\widehat x)-x|`

while the type fixes the format and rounding rule. `NF.ofReal` rounds a
real into the format. The low-level constructor is available for approximation relations, so
theorems that need a genuine grid value ask for `NF.IsRepresentable`.

A stored real value and a representable real value should not be confused when reading an `NF`
theorem. The type fixes the arithmetic operations, and those operations round their results, but
claims that rely on an operand already lying on the grid must have the corresponding evidence or
obtain it from how that operand was constructed. This distinction is useful for proofs that start
with arbitrary real inputs: the first conversion can incur error even if every later operation is
applied to representable values.

# `FP32`: The Binary32-Precision Rounded-Real Specialization

`FP32` is the specialization

```lean (name := fp32Print)
-- The abbreviation fixes those three choices for the proof
-- model.
#print FP32
```
```leanOutput fp32Print
@[reducible] def TorchLean.Floats.FP32 : Type :=
NF binaryRadix fexp32 rnd32
```

where `rnd32` is nearest-even. Note the `@[reducible]`: `FP32` is an abbreviation, not a wrapper
structure, so a proof can move between the two spellings without a coercion.

```lean
-- This equality is definitional: FP32 introduces no second
-- arithmetic construction.
example : FP32 = NF binaryRadix fexp32 rnd32 := rfl
```
It is the right model for a theorem whose intended reading is
"perform this real operation and round it at binary32 precision with gradual underflow." Its
exponent policy has no upper cutoff, so it is not the finite set of IEEE bit patterns. The aliases
`round32`, `ulp32`, and `eps32` expose the rounder, local spacing, and half-ULP scale directly over
`ℝ`.

That separation is deliberate. `FP32` omits:

- NaN and positive or negative infinity;
- signed zero and NaN payloads;
- overflow to infinity;
- IEEE exception flags.

That makes `FP32` the convenient layer for ordinary forward-error analysis. A theorem about a linear
layer can compare the exact dot product with a sequence of rounded operations without splitting
every line into finite, infinite, and NaN cases. When exceptional values matter, we move down one
level to `IEEE32Exec`.

# Domain Hypotheses For Total Operations

`NF` and `FP32` use Lean's total real operations internally. Real division by zero and square root
or logarithm outside their usual analytical domains therefore do not produce IEEE exceptions, and
the rounded-real format has no upper exponent bound that models overflow to infinity. These choices
keep algebraic definitions total; they do not turn invalid-domain calculations into faithful IEEE
executions.

Accordingly, an IEEE correspondence theorem states finiteness, nonzero-denominator, domain, and
no-overflow hypotheses where needed. At the bit level, `IEEE32Exec.toReal?` returns `none` for NaN
and infinity. The convenience projection `toReal` maps those encodings to zero and should be used
only under a finiteness hypothesis; interval work that includes infinities uses the extended-real
semantics instead.

A total real function always returns a real value, including at arguments where an IEEE operation
would signal an exception. Its derivative theorem still has to describe that chosen function.
For example, an algebraic inverse or a guarded logarithm cannot acquire the derivative of an
unguarded expression merely because both are written with familiar notation. Domain hypotheses
identify where the ordinary analytic formula applies; outside that region, the declared extension
or guard determines the function that must be analyzed.

# `IEEE32Exec`: Bits And Exceptional Behavior

`IEEE32Exec` stores a raw `UInt32` bit pattern. Its classifiers distinguish zero, subnormal, normal,
infinite, and NaN encodings. Core arithmetic is implemented in Lean using integer and dyadic
calculations, so it can be evaluated without delegating the operation to the host's floating-point
instruction.

At $`1`, one ULP is $`2^{-23}` and half an ULP is $`2^{-24}`. Here are three increments, a quarter
of a unit in the last place, half a unit, and a full unit, added to the `unit` defined at the top of
the chapter:

```lean
-- Choose increments below, at, and above the nearest-even
-- midpoint at one.
def quarterUlpAtOne : IEEE32Exec :=  -- 2^-25
  IEEE32Exec.ofBits 0x33000000

def halfUlpAtOne : IEEE32Exec :=     -- 2^-24
  IEEE32Exec.ofBits 0x33800000

def oneUlpAtOne : IEEE32Exec :=      -- 2^-23
  IEEE32Exec.ofBits 0x34000000
```

The first increment is below half an ULP, so nearest-even discards it:

```lean (name := quarterAdd)
-- A quarter ULP is too small to change the rounded sum.
#eval (unit.add quarterUlpAtOne).bits
```
```leanOutput quarterAdd
1065353216
```

The second sits exactly at the tie. Nearest-even picks the candidate with the even significand,
which is $`1` again:

```lean (name := halfAdd)
-- At half an ULP the tie rule selects the even significand.
#eval (unit.add halfUlpAtOne).bits
```
```leanOutput halfAdd
1065353216
```

Only the third advances the result to the next representable value, `0x3f800001`:

```lean (name := oneAdd)
-- A full ULP reaches the next representable value.
#eval (unit.add oneUlpAtOne).bits
```
```leanOutput oneAdd
1065353217
```

Those three evaluations are the bit-level version of the spacing picture developed above. The
executable `absorbs` predicate names the event in the first two: the accumulator is unchanged even
though the exact real increment is positive.

```lean (name := absorbQuarter)
-- Absorption asks whether the executable addition leaves
-- its first operand unchanged.
#eval unit.absorbs quarterUlpAtOne
```
```leanOutput absorbQuarter
true
```
```lean (name := absorbHalf)
-- The midpoint is also absorbed when the retained
-- significand is even.
#eval unit.absorbs halfUlpAtOne
```
```leanOutput absorbHalf
true
```
```lean (name := absorbOne)
-- The full spacing is large enough to escape absorption.
#eval unit.absorbs oneUlpAtOne
```
```leanOutput absorbOne
false
```

The definition compares the result of TorchLean's addition with its left operand:

```lean (name := absorbsPrint)
-- Inspect the exact Boolean observation made by the
-- absorption predicate.
#print IEEE32Exec.absorbs
```
```leanOutput absorbsPrint (whitespace := lax)
def TorchLean.Floats.IEEE754.IEEE32Exec.absorbs :
    IEEE754.IEEE32Exec → IEEE754.IEEE32Exec → Bool :=
fun a b => decide (a.add b = a)
```

In an accumulation, absorption means a new term leaves the accumulator unchanged. Repeating this
can lose many small contributions even though each addition is correctly rounded, as happened to
the unit increment in the opening example.

The value-only operations have status-bearing variants. `IEEEOutcome` pairs the result bits with an
`IEEEStatus` containing the invalid, divide-by-zero, overflow, underflow, and inexact flags
supported by the model. Underflow follows the documented tininess-after-rounding policy and is
raised only for an inexact tiny result.

For $`1` divided by $`+0`, the IEEE rule returns positive infinity and raises the divide-by-zero
flag:

```lean (name := divByZero)
-- A nonzero numerator divided by zero raises divide-by-zero
-- status.
#eval IEEE32Exec.divWithStatus
  IEEE32Exec.posOne IEEE32Exec.posZero
```
```leanOutput divByZero (whitespace := lax)
{ value := { bits := 2139095040 },
  status := { invalid := false, divideByZero := true,
              overflow := false, underflow := false,
              inexact := false } }
```

For $`0` divided by $`+0`, it returns the canonical NaN and raises `invalid`:

```lean (name := zeroByZero)
-- Zero divided by zero instead takes the invalid-operation
-- branch.
#eval IEEE32Exec.divWithStatus
  IEEE32Exec.posZero IEEE32Exec.posZero
```
```leanOutput zeroByZero (whitespace := lax)
{ value := { bits := 2143289344 },
  status := { invalid := true, divideByZero := false,
              overflow := false, underflow := false,
              inexact := false } }
```

`2139095040` is `0x7f800000`, positive infinity, and `2143289344` is `0x7fc00000`, the canonical
quiet NaN. Both statuses are derived from the same exact dyadic or rational intermediate used by the
arithmetic operation; no host floating-point instruction is called to guess the flag.

Transcendentals have a different status from basic arithmetic because IEEE 754 does not prescribe
one correctly rounded bit pattern for every elementary function. TorchLean provides deterministic
wrappers and approximation contracts; the runtime chapter explains how a concrete `libm`,
`libdevice`, or LibTorch implementation can be related to them.

Determinism fixes which answer an algorithm returns. A numerical accuracy claim additionally
needs a range-specific error or interval contract for that function. The executable transcendental
kernels have explicit special-value behavior, but the small Taylor bounds in the rules library do
not by themselves certify every executable `sin`, `cos`, or `tanh` input.

The three absorption results are observations about addition at one particular operand. They do
not define a universal smallest meaningful increment: at a larger magnitude, the local spacing
changes. The printed definition of `absorbs` is useful because it fixes exactly what was tested,
namely equality with the first operand after the executable addition. It neither estimates the
lost real contribution nor records an error bound for a longer expression.

The two division records distinguish the result from the reason for it. Dividing a nonzero value
by zero returns an infinity and sets divide-by-zero status; zero divided by zero returns NaN and
sets invalid status. Similarly, an underflow flag is not a synonym for “the result is small.”
Exact subnormal results and tiny inexact results can have different status. Keeping flags alongside
bits lets a theorem or diagnostic ask about that distinction instead of inferring it from a
decimal display.

# Lean `Float32`

Lean gives core `Float32` operations a kernel-visible semantics through `Float32.Model`. The model
defines bit conversion, classification, comparison, addition, subtraction, multiplication,
division, negation, absolute value, and square root. Compiled programs may replace these logical
definitions with native instructions through `@[extern]`; native conformance is recorded separately
for each runtime provider.

TorchLean compares `Float32.Model` with `IEEE32Exec`, its independent raw-bit implementation. The
classification theorem has no runtime assumption:

```lean
-- Relate the host type’s finiteness observation to its
-- logical bit model.
example (a : Float32) :
    Float32.isFinite a =
      IEEE32Exec.isFinite (toIEEE32Exec a) :=
  float32_isFinite_eq_ieee32 a
```

The theorem ranges over every canonical binary32 value. Addition has a corresponding arithmetic
theorem:

```lean
-- The addition bridge retains canonicalization, which
-- matters for NaN encodings.
example (a b : Float32) :
    toIEEE32Exec (a + b) =
      canonicalize
        (IEEE32Exec.add (toIEEE32Exec a)
          (toIEEE32Exec b)) :=
  toIEEE32Exec_add a b
```

The proof covers NaNs, signed infinities and zeros, subnormals, cancellation, normal rounding,
underflow, and overflow. Subtraction follows from addition and the sign-bit negation bridge.
Multiplication, division, square root, negation, and absolute value have analogous theorems.

The `canonicalize` on the right is part of the representation comparison. It selects the NaN
encoding used by `Float32.Model`; it does not round finite results a second time. For a finite
result, the equality therefore retains its exact bits, including a zero's sign. For a NaN
result, removing `canonicalize` would ask for a stronger payload-preservation property than this
theorem states. Inspecting classification before simplifying the equality keeps those two uses
of the bridge distinct.

The absorption example gives a concrete use of the bridge. Compute the same addition with
`Float32`, then inspect the result and its bit pattern:

```lean
-- Use the host binary32 type with an increment below half
-- the spacing at one.
def hostOne : Float32 := 1.0

def hostTiny : Float32 := 1.0 / 33554432.0  -- 2^-25
```
```lean (name := hostAdd)
-- The decimal display shows the rounded host result.
#eval hostOne + hostTiny
```
```leanOutput hostAdd
1.000000
```
```lean (name := hostAddBits)
-- Inspect its bits to remove any ambiguity introduced by
-- decimal formatting.
#eval (hostOne + hostTiny).toBits
```
```leanOutput hostAddBits
1065353216
```

and `toIEEE32Exec` carries that result to the bit pattern the executable model produced for the very
same addition:

```lean (name := hostBridge)
-- The logical conversion exposes the same result as an
-- IEEE32Exec bit pattern.
#eval (toIEEE32Exec (hostOne + hostTiny)).bits
```
```leanOutput hostBridge
1065353216
```

Exceptional values cross the bridge too. Division by zero on the host yields an infinity, and both
classifiers agree about it:

```lean (name := hostInf)
-- Finiteness is an explicit observation; real decoding
-- alone cannot establish it.
#eval Float32.isFinite (hostOne / 0)
```
```leanOutput hostInf
false
```
```lean (name := hostInfBits)
-- The infinity encoding explains why the preceding
-- finiteness check returned false.
#eval (hostOne / 0).toBits
```
```leanOutput hostInfBits
2139095040
```

`Float32.Model` is a canonical binary32 representation. It stores a `UInt32` together with a proof
that every NaN uses Lean's chosen canonical encoding. Addition, subtraction, multiplication,
division, negation, absolute value, square root, comparisons, and classification operate by
unpacking that value, computing in Lean's `UnpackedFloat` model, and repacking. Transcendental
functions such as `sin`, `exp`, and `log` remain opaque and need separate contracts.

`IEEE32Exec` accepts every raw binary32 bit pattern, retains deterministic NaN payload information,
reports IEEE exception status, and connects finite executions to the rounded-real and interval
developments. `Float32.Model` uses one canonical NaN representation. The bridge therefore
canonicalizes arithmetic results before comparing their bits.

The bridge has two parts:

```
-- These equalities describe the logical route through the
-- Float32 model.
Float32 operation
  = Float32.Model operation

Float32.Model operation
  = canonicalize (TorchLean IEEE32Exec operation)
                                             -- independent algorithm-equivalence obligation
```

TorchLean proves both equalities for the modeled core operations. The second compares two pure
Lean algorithms. Canonicalization accounts for their different NaN representations: Lean stores
one NaN bit pattern, while `IEEE32Exec` retains payload and sign bits. The compiled native
implementation needs a separate provider contract.

The resulting interfaces have three separate scopes:

:::table +header
*
  * Claim
  * Current status
*
  * A Lean `Float32` core operation denotes its `Float32.Model` operation.
  * Defined by Lean.
*
  * `Float32.Model` and `IEEE32Exec` classify canonical bit patterns identically.
  * Proved for finiteness, infinity, and NaN.
*
  * `Float32.Model` and `IEEE32Exec` compute the same canonical result.
  * Proved for comparison, addition, subtraction, multiplication, division, square root, negation,
    and absolute value.
*
  * Native CPU, CUDA, or library code computes the logical result.
  * A provider-specific backend contract, checked separately.
:::

The bridge makes `IEEE32Exec` error, interval, and reduction results available to logical
`Float32` proofs. Runtime execution uses the contract attached to its selected provider.

Real decoding deliberately forgets some observations available at the bit level. Both signed
zeros denote the real number zero, and a NaN has no real value to decode faithfully. Consequently,
a real-valued equality cannot by itself establish equality of zero signs or NaN payloads. The
host addition example prints both the displayed value and its bits to show which observation is
being compared. The canonicalization in the logical bridge is relevant precisely when raw NaN
encodings would otherwise distinguish the two sides.

This also explains the importance of the finiteness check before using a real bridge. A total
fallback used by a decoding function makes it convenient to state expressions over all bit
patterns, but the fallback is not a real interpretation of infinity or NaN. The theorem's finite
hypotheses ensure that its real arithmetic is being applied to genuine decoded values.

# Addition Across Numerical Models

The addition above has four useful interpretations.

First, the ideal real expression is

$$`z = 1 + 2^{-25}`.

Nothing is lost in $`\mathbb R`. Second, $`\operatorname{round}_{32}(z)=1`, justified by the
binary32 format and nearest-even rounding theory. Third, constructing `FP32` operands and adding
them applies that same `round32` policy to the exact sum. Fourth, `IEEE32Exec.add` runs the
bit-level algorithm and returns `0x3f800000`.

The bridge theorem supplies the nontrivial connection:

$$`\operatorname{toReal}
    (\operatorname{IEEE32Exec.add}(a,b))
  = \operatorname{round32}
    (\operatorname{toReal}(a)+\operatorname{toReal}(b))`,

under the theorem's finite-path and result hypotheses. The ULP bridge identifies the exponent
returned by executable `ulpExp?` with `ulp32` in the rounded-real model. The absorption theorem then
states that a successful finite executable absorption check implies the corresponding `round32`
addition leaves the left operand unchanged.

The representations line up as follows:

```
-- Each arrow requires the corresponding rounding or
-- execution agreement statement.
exact real expression
  -> generic format and rounding theorem
  -> FP32 finite specialization
  -> IEEE32Exec bit-level operation
  <-> Lean Float32.Model
  -> Lean CPU runtime, native CUDA, or external provider
```

The mathematical layers and both bit-level models are Lean definitions. The last arrow is supplied
by a native backend contract and its validation evidence.

# Exact Subtraction And Sterbenz's Lemma

Sterbenz's lemma says that subtraction can be exact even in floating-point arithmetic. If positive,
representable $`x` and $`y` are within a factor of two,

$$`\frac{y}{2}\leq x\leq 2y`,

then $`x-y` is representable in the same format. TorchLean first proves the fixed-grid and
unbounded-exponent results, then extends the argument to the gradual-underflow `FLT` format. The
extension matters near zero: a proof only about normal values would miss subtraction across the
normal/subnormal boundary.

`FP32.sub_exact_of_sterbenz` specializes the result to finite rounded-real binary32.
`IEEE32Exec.toReal_sub_eq_sub_of_sterbenz` goes further: finite bit patterns are decoded, proved
representable on the `fexp32` grid, passed through the rounded-real Sterbenz theorem, and related
back to executable subtraction. The theorem derives result finiteness from the operand hypotheses.

Restating the executable theorem is the clearest way to see what it costs to use. The hypotheses are
the Sterbenz condition plus finiteness, and the proof is the library theorem applied to them:

```lean
-- The factor-of-two condition makes this finite subtraction
-- exact by Sterbenz.
example (x y : IEEE32Exec)
    (hx : x.isFinite = true) (hy : y.isFinite = true)
    (hxpos : 0 < x.toReal) (hypos : 0 < y.toReal)
    (hxy : x.toReal ≤ 2 * y.toReal)
    (hyx : y.toReal ≤ 2 * x.toReal) :
    (x.sub y).toReal = x.toReal - y.toReal :=
  IEEE32Exec.toReal_sub_eq_sub_of_sterbenz x y hx hy
    hxpos hypos hxy hyx
```

Under those hypotheses, subtracting two bit patterns and decoding the result gives the exact real
difference. Running it on $`3-2`, using the
`two` from the opening calculation, shows the inexact flag staying clear:

```lean
-- Three and two satisfy the positive, nearby-operand
-- conditions.
def three : IEEE32Exec :=
  IEEE32Exec.ofBits 0x40400000
```
```lean (name := sterbenzEval)
-- Inspect both the exact difference and the absence of an
-- inexact flag.
#eval IEEE32Exec.subWithStatus three two
```
```leanOutput sterbenzEval (whitespace := lax)
{ value := { bits := 1065353216 },
  status := { invalid := false, divideByZero := false,
              overflow := false, underflow := false,
              inexact := false } }
```

The absorption bridge from the previous section has the same shape. It turns a decided `absorbs`
check on bit patterns into a statement about `round32` over the reals, which is the form an error
analysis wants:

```lean
-- Finite dyadic witnesses connect an observed absorbed
-- addition to real rounding.
example {a b : IEEE32Exec} {da db : IEEE32Exec.Dyadic}
    (ha : a.toDyadic? = some da)
    (hb : b.toDyadic? = some db)
    (hfin : (a.add b).isFinite = true)
    (habs : a.absorbs b = true) :
    round32 (a.toReal + b.toReal) = a.toReal :=
  IEEE32Exec.round32_add_eq_left_of_absorbs ha hb hfin habs
```

The proofs live in {src "NN/Floats/FP32/Sterbenz.lean"}[`NN/Floats/FP32/Sterbenz.lean`] and
{srcDir "NN/Floats/IEEEExec/Bridge/FP32"}[`NN/Floats/IEEEExec/Bridge/FP32`]. In each case, the
generic result supplies the rounding argument, the binary32 specialization fixes the format,
and the executable bridge connects the decoded bit patterns to that argument.

Sterbenz's conclusion concerns the subtraction of the represented operands. It can be exact even
when those operands were obtained by inaccurate earlier computations. If a preceding calculation
has already lost a small contribution, exact subtraction does not recover it. The theorem is
therefore especially useful inside an error analysis: it removes one prospective rounding term
when its positivity and factor-of-two hypotheses hold, while leaving the operands' existing
errors in place.

The displayed executable example makes that distinction observable. Three and two are represented
exactly, their ratio satisfies the hypothesis, and the difference is one with no inexact status.
The absorption bridge addresses a different situation: it starts from an executable equality and
uses finite dyadic witnesses to express that equality in the rounded-real model. Neither theorem
permits arbitrary reassociation of the surrounding computation.

# Tensors, Reductions, And Quantization

Pointwise tensor operations lift the scalar semantics coordinate by coordinate. A tensor theorem
over `NF` therefore inherits the declared rounding at each scalar operation.

Reductions add another choice: order. A left fold, balanced tree, warp reduction, atomic
accumulation, and library matrix multiplication may all use binary32 addition and still disagree.
Contraction adds the same issue for $`ab+c`: FMA rounds once, while separate multiplication and
addition round twice. The current capsule type records reduction order. It has no separate fields
for contraction, rounding mode, or subnormal handling, so an analysis that depends on those choices
must state them in its arithmetic assumptions.

PyTorch also dispatches among numerical policies {Informal.citep pytorch2019}[]. `torch.sum` over a
CUDA tensor need not use the reduction
tree the CPU kernel uses, `torch.matmul` may or may not contract with a fused multiply-add
depending on the dispatched backend, and setting `torch.backends.cuda.matmul.allow_tf32` changes the
precision of the multiply without changing the program text. The opening non-associativity example
shows why these choices affect a numerical claim. In TorchLean, a fixed-left certificate requires
the capsule to declare that reduction order; an implementation-defined reduction does not supply
the required agreement.

Affine quantization also uses the generic rounding layer. An `AffineQuantizer` has a positive scale
$`s`, zero point $`z`, and integer code bounds. Encoding divides by the scale, rounds to an integer,
shifts by the zero point, and clamps to those bounds. Decoding reverses the shift and scale:

$$`q(x)=\operatorname{clamp}
  \left(\operatorname{round}\left(\frac{x}{s}\right)+z\right)`,

$$`\widehat{x}(q)=s(q-z)`.

Here $`q(x)` is the stored integer code and $`\widehat{x}(q)` is its reconstructed real value.
Clamping explains why the reconstruction-error theorem needs a no-saturation hypothesis: outside
the code range, distance to the nearest grid point alone cannot bound the error.

The scalar definition and its arithmetic theorems live in `NN.Floats.Quantization`. The separate
`NN.Spec.Quantization` adapter applies the same equations at every coordinate of a shape-indexed
tensor. Together they prove code range, monotonicity, and in-range code round trips. The half-step
reconstruction bound additionally requires nearest rounding and inactive saturation. Later runtime
work can add packed int8 or int4
storage without changing these scalar theorems.

For quantization, the no-clipping hypothesis locates the part of the error controlled by nearest
rounding. Before clipping, an integer code is at most half a step from the scaled input. Multiplying
back by a positive scale gives the reconstruction bound. If that code lies outside the allowed
range, clipping can move it farther, so the same half-step conclusion no longer follows. The zero
point changes where the integer codes are centered; it does not remove the need to check the
range. This is a concrete example of a familiar numerical bound whose domain condition is part of
its meaning.

# Native Backend Contracts

Lean `Float32`, C and CUDA `float`, cuBLAS reductions, and LibTorch tensors can all appear in an
execution path. TorchLean proves that `Float32.Model` agrees with its independent executable
reference for classification, comparison, addition, subtraction, multiplication, division, square
root, negation, and absolute value. Native instructions are described by provider contracts.

For a native provider, a kernel capsule records:

- the operation and device;
- its provider and backward mode;
- shape and layout requirements;
- its reduction policy;
- the evidence attached to each shape, layout, value, and VJP contract.

The capsule makes reduction policy available to graph-level error analysis. Its `ContractEvidence`
type records a runtime guard, test suite, trusted boundary, or `notApplicable`; it has no theorem
constructor. The proved bridge for `Float32.Model` above concerns the logical arithmetic model.
It does not become a proof about a native provider merely because a capsule names that provider.

A backend policy describes which numerical choices an execution claims to make. Evidence about
primitive addition does not settle whether a dot product used separate multiplication and addition
or a fused multiply-add, because those expressions round at different points. Likewise, a fixed
left fold and a tree reduction call the same addition primitive in different orders. An application
must connect the selected policy to the expression covered by its theorem. The provider evidence
record helps retain that information, but constructing the record is not itself a proof that the
provider follows it on every execution.

# Numerical Models And Their Uses

Use the smallest layer that states the claim accurately:

:::table +header
*
  * Question
  * Representation
*
  * Which values lie on a radix/precision grid?
  * `NeuralFloat` formats
*
  * How does a declared format round exact real arithmetic?
  * `NF`
*
  * What is the binary32-precision rounded-real error?
  * `FP32`
*
  * What bits and IEEE exceptional cases result?
  * `IEEE32Exec`
*
  * What does a Lean `Float32` core operation mean in the logic?
  * `Float32.Model`
*
  * Does an interval enclose an operation?
  * directed `IEEE32Exec` or proved interval rounders
*
  * What did CPU, CUDA, or LibTorch execute?
  * runtime result plus an explicit bridge or boundary
:::

The table is also a useful debugging guide. If a theorem is cluttered with NaN cases while the
algorithm assumes a finite path, move up to `FP32`. If a proof needs the sign of zero or an
exception flag, move down to `IEEE32Exec`. If two GPU runs disagree, inspect reduction and
contraction policy before blaming the real-valued model.

# References

- Lean FRO,
  [`Float32.Model` source
  ](https://github.com/leanprover/lean4/blob/v4.33.0/src/lean/Init/Data/Float/Model/Float32.lean).
- The Lean Language Reference,
  [Floating-Point
  Numbers](https://lean-lang.org/doc/reference/latest/Basic-Types/Floating-Point-Numbers/).
- [Flocq in a Nutshell](https://flocq.gitlabpages.inria.fr/theos.html), an overview of formats,
  rounding, ULP results, double rounding, and effective operators.
- IEEE Computer Society,
  [IEEE Standard for Floating-Point Arithmetic,
  IEEE 754-2019](https://doi.org/10.1109/IEEESTD.2019.8766229).
- Jean-Michel Muller et al.,
  [*Handbook of Floating-Point Arithmetic*](https://doi.org/10.1007/978-3-319-76526-6),
  second edition.
- Nicholas J. Higham,
  [*Accuracy and Stability of Numerical Algorithms*](https://doi.org/10.1137/1.9780898718027),
  second edition.
- Pat H. Sterbenz, *Floating-Point Computation*, Prentice-Hall, 1974.
