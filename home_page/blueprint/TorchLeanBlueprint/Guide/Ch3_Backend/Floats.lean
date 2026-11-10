import VersoManual
import FloatLib
import FloatLib.Floats.Formats.IEEE754
import FloatLib.Floats.Formats.BinaryInterchange.Analysis.Sterbenz
import FloatLib.Floats
import FloatLib.Floats.Formats.BinaryInterchange.Configured.Transcendentals
import TorchLean.Tensor.Numeric
import FloatLib.Floats.Formats.BinaryInterchange
import FloatLib.Floats.Formats.Flocq.Theory.Rounding.Affine
import TorchLean.Verify.Floats.IEEE754
import TorchLeanBlueprint.Bib
import TorchLeanBlueprint.Roles

open Verso.Genre Manual
open Verso.Genre.Manual.InlineLean
open FloatLib.Numerics
open FloatLib.Floats (ExecFloat)
open FloatLib.Floats.Formats.Flocq
open FloatLib.Floats.Formats.BinaryInterchange (Model FloatFormat)
open TorchLean

-- Verso checks `leanOutput` blocks against the real compiler message. A few of the
-- records and signatures below print wider than this file's 100-column limit, so
-- those blocks ask for `whitespace := lax`, which ignores where the expected text
-- was wrapped. The rendered page still shows the message exactly as Lean printed it.
open Lean.Elab.Tactic.GuardMsgs.WhitespaceMode (lax)

#doc (Manual) "Floating-Point Semantics" =>
%%%
tag := "floats"
%%%

Adding three numbers is enough to expose a numerical choice in a model. In binary32, adding one
to a large value can lose the one before a later subtraction brings the answer back near zero.
Change the order, or give the intermediate result more precision, and that contribution may survive.
We can reproduce the calculation with FloatLib, inspect its exact encodings, and then explain the
result using the format's rounding rule.

# Choose A Format From FloatLib

I'll use binary32 for the first calculation. FloatLib's `ExecFloat.Binary` selects it with eight
exponent bits and 23 stored fraction bits, and the implicit leading bit gives normal values 24 bits
of significand precision:

```lean
abbrev ChapterBinary32 :=
  ExecFloat.Binary (exponentBits := 8) (fractionBits := 23)
```

Change those two parameters and you select a wider or custom format through the same arithmetic
interface. Whichever type you select can also be the element type of TorchLean's typed CPU tensors
and model graphs. LibTorch providers use native binary32 or binary64. Custom scalar functions can
also use configured binary arithmetic on CUDA, while unsupported carriers retain their precision on
CPU.

For the scalar examples, FloatLib is all we need, and TorchLean imports its definitions and proofs
directly. Its dependency declaration follows FloatLib's `main` branch (the manifest records the
exact revision a build used), so updating FloatLib requires updating that manifest and rebuilding.

From here on, `lean` blocks are elaborated while the page is built, and their `leanOutput` blocks
are checked against Lean messages. Plain-text sketches illustrate the interfaces. On this page we
open `FloatLib.Numerics`, `FloatLib.Floats.Formats.Flocq`, and `TorchLean`, and use
`ExecFloat` and `Model` for FloatLib's executable and reference APIs.

# Nonassociative Binary32 Addition

Take the three exact real numbers

$$`a=2^{24},\qquad b=1,\qquad c=-2^{24}`.

Over the reals, both parenthesizations are equal:

$$`(a+b)+c=a+(b+c)=1`.

Binary32 has 24 bits of significand precision, so around $`2^{24}` adjacent representable numbers
are two units apart. That puts the exact value $`2^{24}+1` halfway between them, and ties-to-even
rounds it back to $`2^{24}`. Therefore

$$`
\operatorname{fl}(\operatorname{fl}(a+b)+c)=0,
\qquad
\operatorname{fl}(a+\operatorname{fl}(b+c))=1.
`

We run this example from the bit patterns themselves. `ExecFloat.Binary.ofBits32` reads a raw
32-bit encoding, and ordinary `+` uses FloatLib's arithmetic. No native floating-point addition is
needed to compute the reference result.

```lean
-- Use exact bit patterns so decimal conversion cannot
-- affect the associativity example.
def bigPower : ChapterBinary32 :=     --  2^24
  ExecFloat.Binary.ofBits32 0x4b800000

def unit : ChapterBinary32 :=         --  1
  ExecFloat.Binary.ofBits32 0x3f800000

def negBigPower : ChapterBinary32 :=  -- -2^24
  ExecFloat.Binary.ofBits32 0xcb800000
```

Adding left to right loses the $`1`:

```lean (name := leftAssoc)
-- Round the large positive partial sum before cancellation.
#eval ExecFloat.Binary.toBits32
  ((bigPower + unit) + negBigPower)
```
```leanOutput leftAssoc
0
```

Adding $`1` to $`-2^{24}` first keeps it, because that sum is exactly representable:

```lean (name := rightAssoc)
-- Cancel the negative large operand only after forming the
-- other partial sum.
#eval ExecFloat.Binary.toBits32
  (bigPower + (unit + negBigPower))
```
```leanOutput rightAssoc
1065353216
```

The printed integers encode the results: `0` is `0x00000000`, which is $`+0`, and `1065353216` is
`0x3f800000`, which is $`1`. So even finite binary32 addition isn't associative, a central example
in {Informal.citet goldberg1991}[]. A reduction's error bound must account for the order in which
its additions run.

We can isolate the effect of the tie by changing the small addend.
Replace the $`1` by $`2` and the two groupings agree, because $`2^{24}+2` lands exactly on the
grid:

```lean
-- Two is one representable spacing above the large positive
-- operand.
def two : ChapterBinary32 :=
  ExecFloat.Binary.ofBits32 0x40000000
```
```lean (name := twoLeft)
-- Check whether the larger increment survives left
-- association.
#eval ExecFloat.Binary.toBits32
  ((bigPower + two) + negBigPower)
```
```leanOutput twoLeft
1073741824
```
```lean (name := twoRight)
-- Compare the same increment under right association.
#eval ExecFloat.Binary.toBits32
  (bigPower + (two + negBigPower))
```
```leanOutput twoRight
1073741824
```

Replace it by $`0.5` and they agree again, this time because both groupings lose the small addend
entirely:

```lean
-- Half a unit probes a different rounding tie from the
-- integer increments.
def oneHalf : ChapterBinary32 :=
  ExecFloat.Binary.ofBits32 0x3f000000
```
```lean (name := halfLeft)
-- The first addition decides whether this small
-- contribution survives.
#eval ExecFloat.Binary.toBits32
  ((bigPower + oneHalf) + negBigPower)
```
```leanOutput halfLeft
0
```
```lean (name := halfRight)
-- The negative partial sum has its own nearest-even tie to
-- resolve.
#eval ExecFloat.Binary.toBits32
  (bigPower + (oneHalf + negBigPower))
```
```leanOutput halfRight
0
```

Among these three increments, only $`b=1` changes the answer with grouping. Its positive-side
addition is a tie, while $`1-2^{24}` is exactly representable. The $`b=0.5` negative-side addition
is also a tie, but rounds to $`-2^{24}`.

We can examine the same calculation at three levels. The spacing of the binary32 grid and the
nearest-even rule explain the rounded value. An executable model must also determine its bits and
handle signs, subnormals, infinities, and NaNs. A backend contract must then identify the
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
statement: placing two representations next to each other doesn't establish one.

After each evaluation the printed integer is the stored word, not the represented numerical value.
For example, `1073741824` is the binary32 encoding of two. Comparing words makes the experiment
independent of how many decimal digits a pretty printer chooses. These examples also show why an
informal statement such as “small terms disappear next to large terms” is incomplete. Whether a
term survives depends on the spacing on that side of the large value and, at a midpoint, on the tie
rule. Changing one to two or to one half probes those separate decisions.

This matters when translating a mathematical formula into tensor code. Parentheses select which
partial result is rounded first. A compiler or parallel reducer that changes the association may
still evaluate the same real polynomial while producing a different bit pattern. So the later
reduction contracts describe an evaluation schedule (or quantify over an allowed family of
schedules) instead of treating associativity of real addition as a floating-point law.

# Formats And Rounding
%%%
tag := "TorchLean--Floating-Point-and-Native-Boundaries--Floating-Point-Semantics--\
        Flocq-And-FloatLib___s-Generic-Theory"
%%%

A format tells us which real numbers are available, and a rounding rule chooses one of those
numbers. The same binary32 format can round downward, upward, toward zero, or to nearest-even.
Conversely, the same nearest-even idea can be used with binary16, binary32, a fixed-point grid, or
an experimental low-precision format. Once those choices are separated, theorems about
monotonicity, exactness, neighboring values, and ULP error can be reused.

FloatLib's definitions and proofs are written in Lean. Its `Formats.Flocq` namespace contains the
generic theory, and its executable format families connect concrete encodings to reference
semantics. TorchLean adds tensor semantics, graph-level error propagation, and verification
interfaces. So a theorem about scalar rounding belongs to the imported numerical library, while a
theorem carrying that error through a backward graph belongs to TorchLean.

The namespace retains the name of
[Flocq](https://flocq.gitlabpages.inria.fr/), the Rocq (formerly Coq) library by Sylvie Boldo,
Guillaume Melquiond, and contributors {Informal.citep flocq2011}[]. For readers moving between
its literature and FloatLib's Lean source, the corresponding names are:

:::table +header
*
  * Mathematical role
  * Flocq vocabulary
  * FloatLib vocabulary
*
  * radix power $`\beta^e`
  * `bpow`
  * `bpow`
*
  * mantissa/exponent value
  * `F2R`-style representation
  * `FloatRep`, `toReal`
*
  * exponent policy
  * `fexp`
  * `fexp` with `ValidExp`
*
  * representable grid
  * `generic_format`
  * `genericFormat`
*
  * fixed, unbounded, gradual formats
  * `FIX`, `FLX`, `FLT`
  * `fixExp`, `flxExp`, `fltExp`
*
  * rounding to the grid
  * `round`
  * `round`
:::

FloatLib's names in the last column live in `FloatLib.Floats.Formats.Flocq`. That arithmetic theory
applies independently of a neural network: tensor and model proofs choose its radix, exponent
policy, and rounding rule as parameters.

# Mantissas And Exponents

Before imposing a precision, define a radix-$`\beta` value by an integer mantissa and an integer
exponent:

$$`\operatorname{value}_{\beta}(m,e)=m\beta^e`.

This is FloatLib's structure `FloatRep β`, defined in
`FloatLib/Floats/Formats/Flocq/Theory/Core.lean`:

```
-- A raw representation stores integers; format membership
-- is a separate predicate.
structure FloatRep (β : Radix) where
  mantissa : ℤ
  exponent : ℤ
```

For example:

```lean
-- Represent three quarters exactly as an integer times a
-- power of two.
def threeQuarters : FloatRep binaryRadix :=
  { mantissa := 3, exponent := -2 }
```

`toReal` sends that pair to a real number, and the value is exactly what the arithmetic says
it is:

```lean
-- Decode the representation over the reals, without a
-- floating-point conversion.
example : toReal threeQuarters = 3 / 4 := by
  norm_num [toReal, bpow, binaryRadix,
    Radix.toReal, threeQuarters]
```

There's intentionally no field saying “24 bits of precision.” Two different pairs, $`(3,-2)` and
$`(6,-3)`, denote the same real number, and Lean proves it:

```lean
-- A second pair can denote the same real value without
-- being the same record.
def sixEighths : FloatRep binaryRadix :=
  { mantissa := 6, exponent := -3 }

example :
    toReal sixEighths =
      toReal threeQuarters := by
  norm_num [toReal, bpow, binaryRadix,
    Radix.toReal, threeQuarters, sixEighths]
```

So a raw mantissa/exponent pair is a representation, and on its own it isn't a machine format.
Keeping it general lets later proofs normalize representations and reuse the same carrier at
different precisions.

A format is added as a predicate on the real value, and its parameter `fexp` chooses the exponent
at which representability is tested. Informally,

$$`\operatorname{genericFormat}(\beta,f_{\rm exp},x)`

means that after scaling $`x` by the exponent selected by `fexp`, the resulting mantissa is an
integer. Precision and underflow enter through the exponent policy.

Our two representations of three quarters separate equality of data from equality of meaning.
Multiplying the mantissa by two and reducing the exponent by one leaves the represented real value
unchanged. A theorem about `toReal` can use that equality without claiming that the two records are
structurally equal. That's why a raw mantissa/exponent pair alone isn't enough to specify a format:
the format must say which real values are representable at each magnitude, and rounding must select
one of those values.

# Formats As Exponent Policies

A format's central parameter is an exponent function

$$`f_{\mathrm{exp}}:\mathbb Z\to\mathbb Z`.

For a nonzero real $`x`, its magnitude identifies the power of $`\beta` immediately above
$`|x|`. Applying `fexp` to that magnitude gives the canonical exponent at which the mantissa must
be integral. FloatLib's `genericFormat β fexp x` says precisely that $`x` lies on this grid.

Three standard policies explain most uses:

- `fixExp emin` always returns `emin`. This is a fixed-point grid with constant spacing
  $`\beta^{e_{\min}}`.
- `flxExp prec` returns `e - prec`. This models a precision of `prec` radix digits with no lower
  exponent bound.
- `fltExp emin prec` returns `max (e - prec) emin`. This is the gradual-underflow format: normal
  values receive `prec` digits, while values near zero stay on the fixed subnormal grid
  $`\beta^{e_{\min}}`.

Precision must be positive. Inside symbolic theorem statements, where positivity is carried as a
hypothesis, the raw integer formulas remain useful. Code that reads a format configuration uses
`FormatPrecision.ofNat?` or `FormatPrecision.ofInt?`, and the checked value supplies valid FLX,
FLT, and FTZ exponent selectors. So zero and negative inputs are rejected instead of being silently
reinterpreted through an absolute value.

It helps to see this on a toy system with radix two, precision three, and minimum exponent $`-4`.
Between $`1` and $`2`, three-bit numbers are spaced by $`1/4`:

$$`1,\quad 1.25,\quad 1.5,\quad 1.75,\quad 2`.

Near zero, `fltExp (-4) 3` stops decreasing the exponent, so the spacing becomes the constant
subnormal step $`2^{-4}=1/16`. `flxExp 3` would continue creating smaller normal scales forever,
while `fixExp (-4)` would use the $`1/16` grid everywhere. These three policies aren't unrelated
format implementations. They're three choices for the same exponent function interface.

These policies are ordinary integer functions. At magnitude $`1`, the gradual format chooses
exponent $`-2`, giving the $`1/4` spacing of the example above:

```lean (name := fltNormal)
-- The normal regime chooses spacing from magnitude minus
-- precision.
#eval fltExp (-4) 3 1
```
```leanOutput fltNormal
-2
```

At magnitude $`-2` the gradual format has already reached the floor and returns `emin`, not `-5`:

```lean (name := fltClamped)
-- Near zero the minimum exponent prevents the spacing from
-- shrinking further.
#eval fltExp (-4) 3 (-2)
```
```leanOutput fltClamped
-4
```

`flxExp` has no floor, so its exponent continues to decrease at smaller magnitudes:

```lean (name := flxUnbounded)
-- Removing the lower cutoff lets the spacing continue to
-- follow magnitude.
#eval flxExp 3 (-2)
```
```leanOutput flxUnbounded
-5
```

and `fixExp` ignores the magnitude altogether:

```lean (name := fixConstant)
-- Fixed-point spacing ignores the magnitude argument.
#eval fixExp (-4) 1
```
```leanOutput fixConstant
-4
```

Binary32 uses radix two, precision 24, and the least subnormal exponent $`-149`. Its exponent
policy is visible in the checked definition:

```lean
-- The format descriptor selects the precision and
-- lower exponent.
example :
    Model.fexpOf FloatFormat.binary32 = fltExp (-149) 24 :=
  rfl
```

The choice $`-149` isn't the minimum *normal* exponent. Binary32 normal numbers begin at
$`2^{-126}`, but the 23 fraction bits extend the gradual-underflow grid down to $`2^{-149}`.
Encoding that fact in `fltExp` is what allows the same representability predicate to cover normal
and subnormal finite values. Three evaluations show the transition. Well inside the normal range
the mantissa scale tracks the magnitude:

```lean (name := fexp32Normal)
-- A large magnitude selects a coarser grid.
#eval Model.fexpOf FloatFormat.binary32 100
```
```leanOutput fexp32Normal
76
```

At magnitude $`-120` the value is still normal, so the exponent continues tracking:

```lean (name := fexp32Subnormal)
-- This smaller magnitude still lies above the spacing
-- floor.
#eval Model.fexpOf FloatFormat.binary32 (-120)
```
```leanOutput fexp32Subnormal
-144
```
At magnitude $`-149` the subnormal floor fixes the grid:

```lean (name := fexp32Floor)
-- At the lower end, the exponent policy returns the
-- subnormal spacing floor.
#eval Model.fexpOf FloatFormat.binary32 (-149)
```
```leanOutput fexp32Floor
-149
```

You can see the benefit of the generic definition in theorem statements. Monotonicity of rounding,
the half-ULP nearest-rounding bound, and fixed-grid exactness don't need separate proofs for every
precision. A binary32 theorem specializes the generic result by supplying `binaryRadix`,
`Model.fexpOf FloatFormat.binary32`, and nearest-even rounding.

Exponent-policy arguments are magnitude exponents, not stored IEEE exponent fields. For example,
the output `76` from `Model.fexpOf FloatFormat.binary32 100` means that values at that magnitude
use grid spacing $`2^{76}`. The outputs near the lower end show when the maximum in `fltExp`
selects the fixed floor instead of magnitude minus precision. These calculations involve no
special-value encoding. They describe a real-valued grid that can later be related to the finite
encodings of a bit-level format.

Precision and exponent validity also play different roles. A positive precision gives the intended
number of significant radix digits. The `ValidExp` laws justify how the grid behaves when a
rounding step moves a value across a magnitude boundary. Supplying an arbitrary function of the
right type is enough to write an expression, but the generic rounding theorems require those laws.
For the supported policies, the format constructors provide them.

# Rounding Rules

A format determines the available real values, and a rounding rule selects one to replace an exact
input. A single binary format supports rounding toward negative infinity, toward positive infinity,
toward zero, and to nearest with ties to even.

FloatLib expresses that rounding operation as

$$`\operatorname{round}_{\beta,f,r}(x)
   = \beta^{e}\,r(x\beta^{-e}),
   \qquad e=f(\operatorname{mag}_{\beta}(x))`,

In this formula, $`\beta` is the radix, $`f` is the exponent policy, and
$`\operatorname{mag}_{\beta}(x)` supplies the input's magnitude. Multiplying by $`\beta^{-e}`
expresses the input in units of the selected grid spacing. Then $`r:\mathbb R\to\mathbb Z` rounds
that scaled mantissa to an integer, and multiplication by $`\beta^e` restores the scale.
`ValidRnd r` is the type class that records the order properties required of that integer rounder,
and `RoundingMode` packages the standard choices used by APIs.

Return to the toy three-bit format and round $`1.375`. Its canonical exponent in this binade is
$`-2`, so the scaled mantissa is

$$`1.375\cdot2^2=5.5`.

Rounding downward chooses mantissa $`5`, giving $`1.25`. Rounding upward chooses $`6`, giving
$`1.5`. Nearest-even also chooses $`6`, because the two candidates are equally distant and $`6` is
even. So the format selected the scale $`2^{-2}`, and the rounding mode selected the integer
mantissa.

That separation gives us reusable theorems:

- rounding a representable value leaves it unchanged
- every rounded value belongs to the format
- directed rounding returns the greatest representable value below, or least one above, the input
- nearest rounding has error at most half an ULP
- rounding is monotone
- an initial round-to-odd can prevent a later nearest-even double-rounding error

Because the definition is abstract, a proof can reason about rounding without choosing an algorithm
for it. FloatLib's `Floats.Formats.Flocq.Calculation` modules connect that definition to
calculations on mantissas and exponents. `Location` and `Inbetween` describe a value's position
between neighbors. The rounding calculation uses this position to select a mantissa, then restores
its power-of-radix scale. `round_nearestEven_computed` proves that this calculation agrees with
abstract nearest-even rounding for the declared radix and exponent policy.

Calculations starting from an arbitrary Lean real remain noncomputable, while executable bit-level
arithmetic is supplied separately by configured `ExecFloat` values such as FloatLib binary32.

Our toy rounding calculation has one discrete step: rounding the scaled mantissa from `5.5` to an
integer. Everything before and after that step is exact real scaling. Downward rounding chooses
five, while upward and nearest-even rounding choose six, giving different points on the same grid.
That separation makes the half-ULP argument reusable: a half-unit error on the integer grid becomes
half a grid spacing after rescaling. Directed rounding uses a different integer inequality but the
same representation machinery.

# `NF`: Arithmetic In A Declared Format

Those generic theorems discuss individual real values and rounding functions. Neural-network proofs
need an object on which `+`, `*`, division, activations, and reductions can be written without
repeating all format parameters. That object is

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

Error proofs compare an ideal value with a rounded one, and storing the real projection lets them
express the difference as

$$`|\operatorname{NF.toReal}(\widehat x)-x|`

while the type fixes the format and rounding rule. `NF.ofReal` rounds a
real into the format. The low-level constructor is available for approximation relations, so
theorems that need a genuine grid value ask for `NF.IsRepresentable`.

When reading an `NF` theorem, don't confuse a stored real value with a representable real value.
The type fixes the arithmetic operations, and those operations round their results. But claims that
rely on an operand already lying on the grid must have the corresponding evidence or obtain it from
how that operand was constructed. That distinction is useful for proofs that start with arbitrary
real inputs: the first conversion can incur error even if every later operation is applied to
representable values.

# `FP32`: The Binary32-Precision Rounded-Real Specialization

`FP32` is the specialization

```lean
-- The abbreviation fixes the radix, format, and
-- tie-breaking rule.
example : FP32 =
    NF binaryRadix (Model.fexpOf FloatFormat.binary32)
      nearestEven := rfl
```

Here `nearestEven` breaks ties toward an even significand. Because the abbreviation is transparent,
a proof can move between the two spellings without a coercion.

That specialization is the right model for a theorem whose intended reading is "perform this real
operation and round it at binary32 precision with gradual underflow." Its exponent policy has no
upper cutoff, so it isn't the finite set of IEEE bit patterns. The generic functions
`Model.roundAt`, `Model.ulpAt`, and `Model.halfUlpAt` expose the rounder, local spacing, and
half-ULP scale over `ℝ`, and each takes the chosen `FloatFormat` as its first argument.

That separation is deliberate. `FP32` omits:

- NaN and positive or negative infinity
- signed zero and NaN payloads
- overflow to infinity
- IEEE exception flags

That makes `FP32` the convenient layer for ordinary forward-error analysis. A theorem about a linear
layer can compare the exact dot product with a sequence of rounded operations without splitting
every line into finite, infinite, and NaN cases. When exceptional values matter, we move down one
level to FloatLib binary32.

# Domain Hypotheses For Total Operations

`NF` and `FP32` use Lean's total real operations internally. So real division by zero and square
root or logarithm outside their usual analytical domains don't produce IEEE exceptions, and the
rounded-real format has no upper exponent bound that models overflow to infinity. These choices
keep algebraic definitions total, but they don't turn invalid-domain calculations into faithful
IEEE executions.

Accordingly, an IEEE correspondence theorem states finiteness, nonzero-denominator, domain, and
no-overflow hypotheses where needed. At the bit level, `ExecFloat.Binary.toRat?` returns `none` for
NaN and infinity. The model projection `Model.toReal` maps those encodings to zero and should be
used only under a finiteness hypothesis. Interval work that includes infinities uses the
extended-real semantics instead.

A total real function always returns a real value, including at arguments where an IEEE operation
would signal an exception. Its derivative theorem still has to describe that chosen function. For
example, an algebraic inverse or a guarded logarithm can't acquire the derivative of an unguarded
expression merely because both are written with familiar notation. Domain hypotheses identify where
the ordinary analytic formula applies, and outside that region the declared extension or guard
determines the function that must be analyzed.

# Choosing Executable Precision

The generic `NF` model supports proofs over real values. For an executable scalar, FloatLib exposes
`ExecFloat.Binary`, whose type parameters select the exponent width and stored fraction width.
Normal significands have one additional, implicit bit:

```lean (name := widerFormats)
-- Keep the same arithmetic interface and change the format.
abbrev ChapterBinary64 :=
  ExecFloat.Binary (exponentBits := 11) (fractionBits := 52)

abbrev ChapterBinary128 :=
  ExecFloat.Binary (exponentBits := 15)
    (fractionBits := 112)

abbrev ChapterCustom :=
  ExecFloat.Binary (exponentBits := 11) (fractionBits := 80)
```

The first two abbreviations give binary64's 53-bit precision and binary128's 113-bit precision.
Each type also carries format-validity proofs: at least two exponent bits, a positive fraction
width, and a valid positive bias. Our custom example keeps binary64's exponent width but stores 80
fraction bits, giving 81-bit precision for normal values. Other valid widths use the same
interface. Increasing the fraction width gives finer spacing, while increasing the exponent width
changes the range. Neither choice removes rounding or exceptional values, and wider arithmetic can
cost more time and memory.

Consider $`1+2^{-100}`. At one, binary64's next larger value is $`1+2^{-52}`, so adding $`2^{-100}`
rounds back to one. Binary128 can retain the increment. We construct $`2^{-100}` by dividing by the
exact integer $`2^{100}`, so no native `Float` occurs on the input path:

```lean
def narrowIncrement : ChapterBinary64 :=
  1 / 1267650600228229401496703205376

def wideIncrement : ChapterBinary128 :=
  1 / 1267650600228229401496703205376
```

The finite-result observer returns the exact rational represented by each computed encoding. Its
expected values below are rational arithmetic, independent of the floating-point implementation:

```lean (name := precisionWitness)
#eval
  (ExecFloat.Binary.toRat?
      ((1 : ChapterBinary64) + narrowIncrement) == some 1,
   ExecFloat.Binary.toRat?
      ((1 : ChapterBinary128) + wideIncrement) ==
     some (1 + 1 / (2 ^ 100 : Nat) : Rat))
```
```leanOutput precisionWitness
(true, true)
```

Constructing a binary64 value first and widening it afterwards would preserve the already-rounded
one. Direct literals and `ExecFloat.Binary.parse` (nearest-even by default) round exact decimal
or radix-two input into the selected destination format. Tensor inputs and model parameters need
the same care:
choosing a wide element type cannot recover information discarded by an earlier conversion.

Executable addition has a refinement theorem for all inputs at the selected type:

```lean
example (a b : ChapterBinary128) :
    a + b = ExecFloat.Spec.add a b :=
  ExecFloat.Proof.add_eq_spec a b
```

The refinement equality identifies the operation's reference semantics, including special values.
But a bound on distance from real addition also needs the relevant finite-path hypotheses and the
format's error theorem. A wider format also doesn't establish the accuracy of an elementary
function approximation. Configured transcendental functions are an explicit opt-in import,
`FloatLib.Floats.Formats.BinaryInterchange.Configured.Transcendentals`.

`TorchLean.NN.Sequential.Execution` connects configured scalars to typed CPU tensors and model
graphs. That path retains the selected type through parameters, forward values, and JVPs or VJPs.
`nn.sgdStep` also accepts a learning rate and state gradient in that type, allowing an explicit SGD
loop for models without buffer-update hooks. Frozen state remains unchanged, while a model that
needs running-statistics updates (such as BatchNorm) is rejected by this entrypoint. The tensor
chapter constructs typed state directly. A supervised loop can use the same state with
`trainer.openTyped (α := α) (initialState? := some state)`. That CPU session retains `α` through
samples, predictions, losses, and exact model-state checkpoints. Its finished result captures a
snapshot independent of later session updates. The {ref "training-from-scratch"}[training chapter]
checks one such update against an exact rational. Seeded initialization still begins with stored
`Float` values when no typed state is supplied, and optimizer and scheduler settings also begin as
`Float`. Use `nn.sgdStep` when the update coefficient itself needs the selected precision.

# Encodings And Exceptional Values

FloatLib's `ofBits32` and `toBits32` preserve the raw `UInt32` encoding of a binary32 value. Its
classifiers distinguish zero, subnormal, normal, infinite, and NaN values. Core arithmetic uses
integer and dyadic calculations, so it can be evaluated without delegating the operation to the
host's floating-point instruction. The format parameters fix the encoding, range, and spacing,
while the operation's rounding mode fixes how an exact intermediate lands on that grid.

At $`1`, one ULP is $`2^{-23}` and half an ULP is $`2^{-24}`. Here are three increments (a quarter
of a unit in the last place, half a unit, and a full unit) added to the `unit` defined at the top
of the chapter:

```lean
-- Choose increments below, at, and above the nearest-even
-- midpoint at one.
def quarterUlpAtOne : ChapterBinary32 :=  -- 2^-25
  ExecFloat.Binary.ofBits32 0x33000000

def halfUlpAtOne : ChapterBinary32 :=     -- 2^-24
  ExecFloat.Binary.ofBits32 0x33800000

def oneUlpAtOne : ChapterBinary32 :=      -- 2^-23
  ExecFloat.Binary.ofBits32 0x34000000
```

The first increment is below half an ULP, so nearest-even discards it:

```lean (name := quarterAdd)
-- A quarter ULP is too small to change the rounded sum.
#eval ExecFloat.Binary.toBits32 (unit + quarterUlpAtOne)
```
```leanOutput quarterAdd
1065353216
```

The second sits exactly at the tie. Nearest-even picks the candidate with the even significand,
which is $`1` again:

```lean (name := halfAdd)
-- At half an ULP the tie rule selects the even significand.
#eval ExecFloat.Binary.toBits32 (unit + halfUlpAtOne)
```
```leanOutput halfAdd
1065353216
```

Only the third advances the result to the next representable value, `0x3f800001`:

```lean (name := oneAdd)
-- A full ULP reaches the next representable value.
#eval ExecFloat.Binary.toBits32 (unit + oneUlpAtOne)
```
```leanOutput oneAdd
1065353217
```

In the first two results the bits match `unit`: the increment is absorbed even though it's
positive. The full-ULP increment changes the result. This depends on the operand's local spacing,
so it isn't a universal smallest meaningful increment.

In an accumulation, absorption means a new term leaves the accumulator unchanged. Repeating this
can lose many small contributions even though each addition is correctly rounded, as happened to
the unit increment in the opening example.

Value-only operations have status-bearing variants. FloatLib's configured
`ExecFloat.Binary.IEEEOutcome` contains the result `value`, its exception `status`, and a `ternary`
comparison with the exact result. That status records invalid, divide-by-zero, overflow, underflow,
and inexact flags. These calls default to nearest-even rounding. Underflow follows the documented
tininess-after-rounding policy and is raised only for an inexact tiny result. We project the
result's `UInt32` encoding for display:

For $`1` divided by $`+0`, the IEEE rule returns positive infinity and raises the divide-by-zero
flag:

```lean (name := divByZero)
-- A nonzero numerator divided by zero raises divide-by-zero
-- status.
#eval
  let result := ExecFloat.Binary.div
    (1 : ChapterBinary32) (0 : ChapterBinary32)
    (status := true)
  (ExecFloat.Binary.toBits32 result.value, result.status)
```
```leanOutput divByZero (whitespace := lax)
(2139095040, { divideByZero := true })
```

For $`0` divided by $`+0`, the rule returns the canonical NaN and raises `invalid`:

```lean (name := zeroByZero)
-- Zero divided by zero instead takes the invalid-operation
-- branch.
#eval
  let result := ExecFloat.Binary.div
    (0 : ChapterBinary32) (0 : ChapterBinary32)
    (status := true)
  (ExecFloat.Binary.toBits32 result.value, result.status)
```
```leanOutput zeroByZero (whitespace := lax)
(2143289344, { invalid := true })
```

`2139095040` is `0x7f800000` (positive infinity), and `2143289344` is `0x7fc00000` (the canonical
quiet NaN). Both statuses are derived from the same exact dyadic or rational intermediate used by
the arithmetic operation, and no host floating-point instruction is called to guess the flag.

Transcendentals have a different status from basic arithmetic because IEEE 754 doesn't prescribe
one correctly rounded bit pattern for every elementary function. FloatLib supplies explicit
approximation kernels, and for IEEE formats correctly rounded `exp`, `log`, `expMinus1`, `logPlus1`
and `expMul`, which TorchLean's configured `Context` uses. TorchLean adds correctly rounded `atan`,
`asin`, `acos` and `atan2` for IEEE formats
({src "TorchLean/Tensor/ConfiguredBinary/InverseTrigonometric.lean"}[the inverse
trigonometric functions]): exact comparisons with rational boundaries decide the rounding, with no
floating-point approximation, and the special values follow C99 Annex F. They form the
`InverseTrig` capability of every configured format. `InverseTrigonometric.toReal_atan` and its
companions prove the rounding at finite inputs in the domain. `roundoff` has no numerical rule for
these functions, so `#roundoff_coverage` records their routes as unsupported. The rounding theorems
cover these Lean definitions, not a native `libm`, `libdevice`, or LibTorch implementation.

Determinism fixes which answer an algorithm returns. A numerical accuracy claim also needs a
range-specific error or interval contract for that function. The executable transcendental kernels
have explicit special-value behavior, but the small Taylor bounds in the rules library don't by
themselves certify every executable `sin`, `cos`, or `tanh` input.

# Lean `Float32`

Lean gives core `Float32` operations a kernel-visible semantics through `Float32.Model`. The model
defines bit conversion, classification, comparison, addition, subtraction, multiplication,
division, negation, absolute value, and square root. Compiled programs may replace these logical
definitions with native instructions through `@[extern]`, and native conformance is recorded
separately for each runtime provider.

FloatLib supplies both the configured software arithmetic and its connections to Lean's model.
Use its native adapters directly. Importing a native value and exporting it again recovers the
same logical value:

```lean (name := nativeRoundTripProof)
example (a : Float32) :
    ExecFloat.Binary.toFloat32
      (ExecFloat.Binary.ofFloat32 a :
        ChapterBinary32) = a :=
  ExecFloat.Binary.toFloat32_ofFloat32 a
```

The round-trip theorem ranges over every native value: finite numbers, both signed zeros,
infinities, and Lean's canonical NaN. In the opposite direction, a configured word may carry a NaN
payload that Lean doesn't retain. Export followed by import canonicalizes that NaN but leaves
finite bits and zero signs unchanged.

Addition has a different domain from the round trip. Both operands must be finite:

```lean (name := nativeFiniteAddProof)
example (a b : Float32)
    (ha : a.isFinite = true) (hb : b.isFinite = true) :
    (ExecFloat.Binary.ofFloat32 (a + b) : ChapterBinary32) =
      ExecFloat.Binary.ofFloat32 a +
        ExecFloat.Binary.ofFloat32 b :=
  ExecFloat.Binary.ofFloat32_add_of_isFinite a b ha hb
```

In the addition theorem, the finite inputs include signed zeros and subnormals. The result has no
finiteness premise: cancellation, rounding, underflow, and overflow to infinity are all included.
`ExecFloat.Binary.ofFloat32_sub_of_isFinite` gives the corresponding subtraction theorem. Neither
statement asserts agreement for an addition or subtraction with a NaN or infinite input.

Square root does cover every configured binary32 word, with native export on both sides:

```lean (name := nativeSqrtProof)
example (x : ExecFloat.Binary 8 23) :
    ExecFloat.Binary.toFloat32 (ExecFloat.sqrt x) =
      (ExecFloat.Binary.toFloat32 x).sqrt :=
  ExecFloat.Binary.toFloat32_sqrt x
```

Negative inputs, signed zeros, infinities, and NaNs are included. Export selects Lean's canonical
NaN representation, so this equality doesn't preserve arbitrary NaN payloads. Canonicalization is a
representation change, not a second rounding of finite results. FloatLib exposes the same
round-trip, finite-add/sub, and square-root interfaces for binary64 through `ofFloat` and
`toFloat`.

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

and `ExecFloat.Binary.ofFloat32` imports that result as a configured value. Reading its word
removes the native display convention from the comparison:

```lean (name := hostBridge)
-- The logical conversion exposes the same result as a
-- configured binary32 bit pattern.
#eval ExecFloat.Binary.toBits32
  (ExecFloat.Binary.ofFloat32 (hostOne + hostTiny))
```
```leanOutput hostBridge
1065353216
```

Exceptional values also have logical native encodings. Division by zero in this example yields
an infinity:

```lean (name := hostInf)
-- Finiteness is an explicit observation: real decoding
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

FloatLib binary32 accepts every raw binary32 bit pattern, retains deterministic NaN payload
information,
reports IEEE exception status, and connects finite executions to the rounded-real and interval
developments. `Float32.Model` uses one canonical NaN representation. A comparison that can
produce NaN must account for that representation choice.

The comparison has two parts:

```
-- These equalities describe the logical route through the
-- Float32 model.
Float32 operation
  = Float32.Model operation

Float32.Model operation
  = configured FloatLib operation, through its model conversion
                                             -- operation-specific theorem and hypotheses
```

Lean defines the first equality. FloatLib proves operation-specific forms of the second between
pure Lean algorithms. Its square-root theorem accounts for NaNs through canonicalization, while its
native addition and subtraction theorems require finite inputs. A theorem refining configured
division to `Model.div` has a different scope: it doesn't identify arbitrary native `Float32`
division with that operation.

Integer conversion has its own policy. FloatLib's constructor proofs round arbitrary `Nat` and
`Int` inputs without first converting them through a narrower native format. In the other
direction, native signed-integer casts truncate toward zero and then saturate to the target range
(NaN maps to zero and infinities to the corresponding endpoint). Checked scalar conversion can
instead report exceptional or out-of-range inputs. Agreement with a checked conversion needs its
range hypotheses. None of these logical cast theorems asserts hardware exception flags.

Together, these interfaces expose several different claims:

:::table +header
*
  * Claim
  * Current status
*
  * A Lean `Float32` core operation denotes its `Float32.Model` operation.
  * Defined by Lean.
*
  * Importing and exporting a native `Float32` recovers it.
  * Proved for every native logical value.
*
  * Native and configured addition/subtraction agree.
  * Proved for finite operands (the result may overflow).
*
  * Configured square root commutes with native export.
  * Proved for every input word, with NaNs canonicalized at export.
*
  * Configured division refines its software model.
  * Proved for all configured operands and the selected rounding mode.
*
  * Native CPU, CUDA, or library code computes the logical result.
  * A provider-specific backend contract, checked separately.
:::

To transfer an error, interval, or reduction result to a native logical expression, first choose
the matching operation theorem and discharge its hypotheses. Runtime execution separately uses
the contract attached to its selected provider.

Real decoding deliberately forgets some observations available at the bit level. Both signed zeros
denote the real number zero, and a NaN has no real value to decode faithfully. So a real-valued
equality can't by itself establish equality of zero signs or NaN payloads. Our host addition
example prints both the displayed value and its bits to show which observation is being compared.
Canonicalization at native export is relevant precisely when raw NaN encodings would otherwise
distinguish the two sides.

That's also why the finiteness check matters before using a real bridge. A total fallback used by a
decoding function makes it convenient to state expressions over all bit patterns, but the fallback
isn't a real interpretation of infinity or NaN. The theorem's finite hypotheses ensure that its
real arithmetic is being applied to genuine decoded values.

# Addition Across Numerical Models

There are four useful interpretations of the addition above.

First, the ideal real expression is

$$`z = 1 + 2^{-25}`.

Nothing is lost in $`\mathbb R`. Second, $`\operatorname{round}_{32}(z)=1`, justified by the
binary32 format and nearest-even rounding theory. Third, constructing `FP32` operands and adding
them applies that same `Model.roundAt FloatFormat.binary32` policy to the exact sum. Fourth,
FloatLib addition runs the
bit-level algorithm and returns `0x3f800000`.

The bridge theorem supplies the nontrivial connection:

$$`\operatorname{toReal}
    (\operatorname{add}(a,b))
  = \operatorname{round}_{32}
    (\operatorname{toReal}(a)+\operatorname{toReal}(b))`,

under the theorem's finite-path and result hypotheses. The absorption theorem states that when a
finite executable sum equals its left operand, the corresponding
`Model.roundAt FloatFormat.binary32` addition leaves that operand unchanged.

Here's how the representations line up:

```
-- Each arrow requires the corresponding rounding or
-- execution agreement statement.
exact real expression
  -> generic format and rounding theorem
  -> FP32 finite specialization
  -> FloatLib configured bit-level operation
  <-> Lean Float32.Model
  -> Lean CPU runtime or LibTorch CUDA execution
```

In the diagram, the mathematical layers and both bit-level models are Lean definitions, while the
last arrow is supplied by a native backend contract and its validation evidence.

# Exact Subtraction And Sterbenz's Lemma

Sterbenz's lemma says that subtraction can be exact even in floating-point arithmetic. If positive,
representable $`x` and $`y` are within a factor of two,

$$`\frac{y}{2}\leq x\leq 2y`,

then $`x-y` is representable in the same format. FloatLib first proves the fixed-grid and
unbounded-exponent results, then extends the argument to the gradual-underflow `FLT` format, and
that extension matters near zero (a proof only about normal values would miss subtraction across
the normal/subnormal boundary).

FloatLib's `Model.roundAt_sub_eq_of_sterbenz fmt` applies to the rounded-real grid of any format.
`Model.toReal_sub_eq_of_sterbenz` goes further: for an IEEE format, bit patterns with positive
values are decoded, proved representable on that format's grid, passed through the rounded-real
Sterbenz theorem, and related back to executable subtraction. That executable theorem derives
result finiteness from the operand hypotheses.

Restating the executable theorem is the clearest way to see what it costs to use. Its hypotheses
are positivity and the factor-of-two bounds (a positive value is already finite, so no finiteness
hypothesis is needed). First, the proof unfolds the certified subtraction to its model operation.
Then FloatLib's Sterbenz theorem supplies the exact real equality, and `rfl` checks that the
binary32 descriptor is an IEEE format:

```lean (name := sterbenzProof)
-- The factor-of-two condition makes this finite subtraction
-- exact by Sterbenz.
open FloatLib.Floats.Formats.BinaryInterchange in
example (x y : ExecFloat.Binary 8 23)
    (hxpos : 0 < Model.toReal (ExecFloat.Binary.toModel x))
    (hypos : 0 < Model.toReal (ExecFloat.Binary.toModel y))
    (hxy : Model.toReal (ExecFloat.Binary.toModel x) ≤
      2 * Model.toReal (ExecFloat.Binary.toModel y))
    (hyx : Model.toReal (ExecFloat.Binary.toModel y) ≤
      2 * Model.toReal (ExecFloat.Binary.toModel x)) :
    Model.toReal (ExecFloat.Binary.toModel (x - y)) =
      Model.toReal (ExecFloat.Binary.toModel x) -
      Model.toReal (ExecFloat.Binary.toModel y) := by
  have hsub : ExecFloat.Binary.toModel (x - y) =
      Model.sub (ExecFloat.Binary.toModel x)
        (ExecFloat.Binary.toModel y) := by
    change ExecFloat.Binary.toModel (ExecFloat.sub x y) = _
    rw [ExecFloat.Proof.sub_eq_spec,
      Model.Proof.sub_eq_spec]
    exact ExecFloat.Binary.toModel_ofModel
      (Model.Spec.sub (ExecFloat.Binary.toModel x)
        (ExecFloat.Binary.toModel y))
  rw [hsub]
  exact Model.toReal_sub_eq_of_sterbenz
    (by rfl) hxpos hypos hxy hyx
```

Under those hypotheses, subtracting two bit patterns and decoding the result gives the exact real
difference. Running that subtraction on $`3-2` (with the `two` from the opening calculation) shows
the inexact flag staying clear:

```lean
-- Three and two satisfy the positive, nearby-operand
-- conditions.
def three : ChapterBinary32 :=
  ExecFloat.Binary.ofBits32 0x40400000
```
```lean (name := sterbenzEval)
-- Inspect both the exact difference and the absence of an
-- inexact flag.
#eval
  let result := ExecFloat.Binary.sub three two
    (status := true)
  (ExecFloat.Binary.toBits32 result.value, result.status)
```
```leanOutput sterbenzEval (whitespace := lax)
(1065353216, {})
```

Our absorption contract has the same shape. It turns the decided equality of two configured values
into a statement about `Model.roundAt FloatFormat.binary32` over the reals, which is the form an
error analysis wants:

```lean (name := absorptionProof)
-- A finite observed absorbed addition connects
-- directly to real rounding.
example {a b : ExecFloat.Binary 8 23}
    (hfin : ExecFloat.Binary.isFinite (a + b) = true)
    (habs : decide (a + b = a) = true) :
    Model.roundAt FloatFormat.binary32
      (Model.toReal (ExecFloat.Binary.toModel a) +
        Model.toReal (ExecFloat.Binary.toModel b)) =
      Model.toReal (ExecFloat.Binary.toModel a) :=
  (IEEE754.round_add_eq_left_of_absorbs
    hfin habs :)
```

FloatLib's `Formats.BinaryInterchange.Analysis.Sterbenz` supplies both the rounded-real and
executable model theorems above, and its executable proof uses FloatLib's subtraction specification
and codec round trip directly. We get the other links used here from the
{src "TorchLean/Verify/Floats/IEEE754.lean"}[finite arithmetic bridge], which also
states the absorption contract. In each case, the generic result supplies the rounding argument,
instantiating it at `ExecFloat.Binary 8 23` fixes the format, and the executable bridge connects
the decoded bit patterns to that argument.

Sterbenz's conclusion concerns subtracting the represented operands, and that subtraction can be
exact even when the operands came from inaccurate earlier computations. If a preceding calculation
has already lost a small contribution, exact subtraction doesn't recover it. So the theorem is
especially useful inside an error analysis: it removes one prospective rounding term when its
positivity and factor-of-two hypotheses hold, while leaving the operands' existing errors in place.

For the absorption contract, a finite executable sum already forces finite operands, so the
contract needs no separate operand-finiteness witnesses. Neither theorem permits reassociation of
the surrounding computation.

# Tensors, Reductions, And Quantization

Pointwise tensor operations lift the scalar semantics coordinate by coordinate, so a tensor theorem
over `NF` inherits the declared rounding at each scalar operation.

Reductions add another choice: order. A left fold, balanced tree, warp reduction, atomic
accumulation, and library matrix multiplication may all use binary32 addition and still disagree.
Contraction adds the same issue for $`ab+c`: FMA rounds once, while separate multiplication and
addition round twice. TorchLean's capsule type records reduction order but has no separate fields
for contraction, rounding mode, or subnormal handling, so an analysis that depends on those choices
must state them in its arithmetic assumptions.

PyTorch also dispatches among numerical policies {Informal.citep pytorch2019}[]. `torch.sum` over a
CUDA tensor need not use the reduction tree the CPU kernel uses, `torch.matmul` may or may not
contract with a fused multiply-add depending on the dispatched backend, and setting
`torch.backends.cuda.matmul.allow_tf32` changes the precision of the multiply without changing the
program text. Our opening non-associativity example shows why these choices affect a numerical
claim. In TorchLean, a fixed-left certificate requires the capsule to declare that reduction order.
An implementation-defined reduction doesn't supply the required agreement.

FloatLib also offers exact accumulation with one final rounding through `ExecFloat.Binary.sum` and
`ExecFloat.Binary.dot`, and those operations have a different semantics from a tree that rounds at
every addition or multiply-add. In the opening example, exact accumulation keeps the unit
contribution until the end, while a rounded tree may already have lost it. So TorchLean's rounded
reduction-tree statements remain necessary for the corresponding tensor and backend paths, and an
exact-accumulation theorem can't be substituted for their local rounding premises.

`TorchLean.Verify.Floats.IEEE754.Reductions` evaluates FloatLib `ReductionTree` schedules with the
executable operations of a configured IEEE format. Its `IEEE754.SumResult.enclosure` and
`IEEE754.DotResult.enclosure` rest on FloatLib's
`Model.ReductionTree.abs_toReal_eval_sub_sum_le_geometric`. That theorem applies
`abs_eval_sub_exact_le_geometric`, whose premise is local to the tree: each of its rounded
additions $`x+y` errs by at most $`u(|x|+|y|)`. Finite evaluation at every leaf and internal node
gives that premise with the format's unit roundoff, $`2^{-24}` for binary32, subnormal sums
included. No relative bound on rounding arbitrary reals is assumed. Binary32 has none with
$`u=2^{-24}`, because subnormal results need an absolute error term. The dot-product enclosure
bounds accumulation error relative to the exact sum of already-rounded products. Bounding the
earlier product errors is a separate step.

Affine quantization also uses the generic rounding layer. A `RealAffineQuantizer` has a positive
scale $`s`, zero point $`z`, and integer code bounds. Encoding divides by the scale, rounds to an
integer, shifts by the zero point, and clamps to those bounds. Decoding reverses the shift and
scale:

$$`q(x)=\operatorname{clamp}
  \left(\operatorname{round}\left(\frac{x}{s}\right)+z\right)`,

$$`\widehat{x}(q)=s(q-z)`.

Here $`q(x)` is the stored integer code and $`\widehat{x}(q)` is its reconstructed real value.
Clamping explains why the reconstruction-error theorem needs a no-saturation hypothesis: outside
the code range, distance to the nearest grid point alone cannot bound the error.

The scalar definition and its arithmetic theorems live in FloatLib. `Tensor.map` applies the same
equations at every coordinate of a shape-indexed tensor, and the facts next to `Tensor.forall_map`
carry the scalar theorems to every coordinate. Together they prove code range, monotonicity, and
in-range code round trips. For the half-step reconstruction bound we also need nearest rounding and
inactive saturation. These scalar theorems don't depend on how the runtime stores codes, so packed
int8 or int4 storage wouldn't change them.

A zero point shifts the code grid, but it doesn't remove the need to check for saturation.

# Proving A Real Bound

Sometimes we want a bound on a formula before choosing its floating-point implementation.
For example, suppose a coordinate lies between one and two and we raise it to a real power.
We can give FloatLib's `interval` tactic those assumptions and ask Lean to check an upper bound:

```lean
example (x y : ℝ)
    (hx : x ∈ Set.Icc 1 2)
    (hy : y ∈ Set.Icc (-2) 2) :
    x ^ y < 5 := by
  interval (depth := 0)
```

This proves the inequality for every pair in those ranges, not just sampled inputs, and the same
proof can be applied to tensor coordinates once we have their bounds. Real powers require a
strictly positive base interval. FloatLib also provides `Real.logb` and inverse hyperbolic
enclosures, each with its own domain checks.

To support another named function, `@[interval_extension]` registers an executable enclosure
together with its containment proof, and the tactic evaluates that enclosure again when it
subdivides an input box. We use the upstream API and don't keep a second registry in TorchLean.
These proofs concern real expressions, so relating them to a rounded tensor computation still
requires the arithmetic and backend contracts below.

# Native Backend Contracts

Lean `Float32`, C and CUDA `float`, cuBLAS reductions, and LibTorch tensors can all appear in an
execution path. FloatLib connects Lean's logical native models to configured software arithmetic
through the operation-specific theorems above. TorchLean's runtime contracts separately connect
provider results to configured binary32 error bounds. Native instructions require provider evidence.

For a native provider, a kernel capsule records:

- the registered operator (attributes included) and its device
- its provider and backward mode
- shape and layout requirements
- its reduction policy
- the evidence attached to each shape, layout, value, and VJP contract

A capsule makes reduction policy available to graph-level error analysis. Its `Contract.Evidence`
type records a shared definition, runtime guard, test suite, trusted boundary, or `notApplicable`
(the type has no theorem constructor). The bridge proved above for `Float32.Model` concerns the
logical arithmetic model, and it doesn't become a proof about a native provider merely because a
capsule names that provider.

A backend policy describes which numerical choices an execution claims to make. Evidence about
primitive addition doesn't settle whether a dot product used separate multiplication and addition
or a fused multiply-add, because those expressions round at different points. Likewise, a fixed
left fold and a tree reduction call the same addition primitive in different orders. An application
must connect the selected policy to the expression covered by its theorem. A provider evidence
record helps retain that information, but constructing the record isn't itself a proof that the
provider follows it on every execution.

# Numerical Models And Their Uses

Use the smallest layer that states the claim accurately:

:::table +header
*
  * To find out
  * Representation
*
  * Which values lie on a radix/precision grid
  * `FloatRep` formats
*
  * How a declared format rounds exact real arithmetic
  * `NF`
*
  * The binary32-precision rounded-real error
  * `FP32`
*
  * Which bits and IEEE exceptional cases result
  * `ExecFloat.Binary` with the chosen format
*
  * What a Lean `Float32` core operation means in the logic
  * `Float32.Model`
*
  * Whether an interval encloses an operation
  * FloatLib directed operations or proved interval rounders
*
  * What CPU, CUDA, or LibTorch executed
  * runtime result plus an explicit bridge or boundary
:::

When a numerical argument gets stuck, the table helps too. If a theorem is cluttered with NaN cases
while the algorithm assumes a finite path, move up to `FP32`. If a proof needs the sign of zero or
an exception flag, use the configured bit-level operations. If two GPU runs disagree, inspect
reduction and contraction policy before blaming the real-valued model.

To follow one of these calculations into the source, start with the operation it uses:

* `FloatLib.Floats.Formats.BinaryInterchange.Configured` defines executable formats, values,
  and rounding.
* `FloatLib.Floats.Formats.Flocq` contains the generic format, rounding, and error theory. Its
  `Calculation.Round` and `Calculation.Operations` modules supply the rounding calculations.
* `FloatLib.Floats.Formats.IEEE754` connects configured operations to Lean's native logical models.
* `FloatLib.Numerics.Enclosure.Interval.Runtime` supplies format-independent outward interval
  arithmetic, with containment proofs in `Interval.Proof` and real bounds in `Interval.Real`.
  `ExecFloat.Binary.Interval` adds configured binary operations for arbitrary formats, storage
  plans, and codecs. Decimal and posit endpoints use the corresponding `OutwardRounding` adapters.

TorchLean's numerical graph certificates use `FloatLib.Numerics.Interval (ExecFloat.Binary 8 23)`
directly, and their endpoint contract remains binary32. Choosing another format for a scalar
interval doesn't by itself supply a graph certificate for that format.

The tensor arguments live in TorchLean. `TorchLean.Verify.Floats.FP32` selects binary32's
gradual-underflow grid. The `NF` bounds carry rounding errors through tensor operations on any such
grid, binary32's included, up to whole networks in `TorchLean.Verify.RuntimeApprox.NF.MLP` and
`TorchLean.Verify.RuntimeApprox.NF.CROWN`. The reduction-tree modules account for accumulation
order. `Tensor.map` applies FloatLib's real-scale or rational quantizer at every coordinate.

# References

- Lean FRO,
  [`Float32.Model` source
  ](https://github.com/leanprover/lean4/blob/v4.35.0/src/Init/Data/Float/Model/Float32.lean).
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
