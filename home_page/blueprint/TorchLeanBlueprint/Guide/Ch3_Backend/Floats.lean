import VersoManual
import FloatLib
import FloatLib.Floats.Formats.IEEE754
import FloatLib.Floats.Formats.BinaryInterchange.Analysis.Sterbenz
import NN.Floats
import NN.Proofs.RuntimeApprox.IEEE32.Contracts
import TorchLeanBlueprint.Bib
import TorchLeanBlueprint.Roles

open Verso.Genre Manual
open Verso.Genre.Manual.InlineLean
open FloatLib.Numerics
open FloatLib.Floats (ExecFloat)
open FloatLib.Floats.Formats.Flocq
open FloatLib.Floats.Formats.BinaryInterchange (Model)
open TorchLean.Floats

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
exponent bits and 23 stored fraction bits. The implicit leading bit gives normal values 24 bits
of significand precision:

```lean
abbrev ChapterBinary32 :=
  ExecFloat.Binary (exponentBits := 8) (fractionBits := 23)
```

Changing those two parameters selects a wider or custom format through the same arithmetic
interface. The selected type can also be the element type of TorchLean's typed CPU tensors and
model graphs. CUDA providers currently use native binary32 or binary64.

The scalar examples only need FloatLib. TorchLean imports its definitions and proofs directly,
at revision `40301cd44f253a4ac6ccd34a0eb6c221e185e25c`.

The `lean` blocks in the rest of this chapter are elaborated while the page is built; their
`leanOutput` blocks are checked against Lean messages. Plain-text sketches illustrate the
interfaces. The page opens `FloatLib.Numerics`, `FloatLib.Floats.Formats.Flocq`, and
`TorchLean.Floats`, and uses `ExecFloat` and `Model` for FloatLib's executable and reference APIs.

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

We run this example from the bit patterns themselves. `ExecFloat.Binary.ofBits32` reads a raw
32-bit encoding; ordinary `+` uses FloatLib's arithmetic. No native floating-point addition is
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
`0x3f800000`, which is $`1`. Thus even finite binary32 addition is not associative, a central
example in {Informal.citet goldberg1991}[]. A reduction's error bound must account for the order
in which its additions run.

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

# Formats And Rounding
%%%
tag := "TorchLean--Floating-Point-and-Native-Boundaries--Floating-Point-Semantics--Flocq-And-FloatLib___s-Generic-Theory"
%%%

A format tells us which real numbers are available. A rounding rule chooses one of those numbers.
The same binary32 format can round downward, upward, toward zero, or to nearest-even. Conversely,
the same nearest-even idea can be used with binary16, binary32, a fixed-point grid, or an
experimental low-precision format. Once those choices are separated, theorems about monotonicity,
exactness, neighboring values, and ULP error can be reused.

FloatLib's definitions and proofs are written in Lean. Its `Formats.Flocq` namespace contains the
generic theory; its executable format families connect concrete encodings to reference semantics.
TorchLean adds tensor semantics, graph-level error propagation, and verification interfaces. A
theorem about scalar rounding therefore belongs to the imported numerical library, while a theorem
carrying that error through a backward graph belongs to TorchLean.

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

The names in the last column live in `FloatLib.Floats.Formats.Flocq`. The arithmetic theory applies
independently of a neural network; tensor and model proofs choose its radix, exponent policy, and
rounding rule as parameters.

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

There is intentionally no field saying “24 bits of precision.” The pair $`(3,-2)` and the pair
$`(6,-3)` denote the same real number, and that is a proof rather than a remark:

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

A raw mantissa/exponent pair is therefore a representation, not yet a machine format. Keeping it
general lets later proofs normalize representations and reuse the same carrier at different
precisions.

A format is added as a predicate on the real value. Its parameter `fexp` chooses the exponent used
to test representability. Informally,

$$`\operatorname{genericFormat}(\beta,f_{\rm exp},x)`

means that after scaling $`x` by the exponent selected by `fexp`, the resulting mantissa is an
integer. The exponent policy is where precision and underflow enter.

The two representations of three quarters separate equality of data from equality of meaning.
Multiplying the mantissa by two and reducing the exponent by one leaves the represented real value
unchanged. A theorem about `toReal` can use that equality without claiming that the two
records are structurally equal. This is why a raw mantissa/exponent pair alone is insufficient to
specify a format: the format must say which real values are representable at each magnitude, and
rounding must select one of those values.

# Formats As Exponent Policies

The central format parameter is an exponent function

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

Precision must be positive. The raw integer formulas remain useful inside symbolic theorem
statements, where positivity is carried as a hypothesis. Code reading a format configuration uses
`FormatPrecision.ofNat?` or `FormatPrecision.ofInt?`; the checked value supplies valid
FLX, FLT, and FTZ exponent selectors. Thus zero and negative inputs are rejected instead of being
silently reinterpreted through an absolute value.

It helps to see this on a toy system. Take radix two, precision three, and minimum exponent $`-4`.
Between $`1` and $`2`, three-bit numbers are spaced by $`1/4`:

$$`1,\quad 1.25,\quad 1.5,\quad 1.75,\quad 2`.

Near zero, `fltExp (-4) 3` stops decreasing the exponent, so the spacing becomes the constant
subnormal step $`2^{-4}=1/16`. `flxExp 3` would continue creating smaller normal scales forever;
`fixExp (-4)` would use the $`1/16` grid everywhere. These three policies are not unrelated format
implementations. They are three choices for the same exponent function interface.

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

At magnitude $`-2` it has already reached the floor and returns `emin` rather than `-5`:

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

Binary32 uses radix two, precision 24, and the least subnormal exponent $`-149`. The corresponding
exponent policy is visible in the checked definition:

```lean (name := fexp32Print)
-- Inspect the actual precision and lower exponent used by
-- the FP32 specialization.
#print fexp32
```
```leanOutput fexp32Print
def TorchLean.Floats.fexp32 : ℤ → ℤ :=
fltExp (-149) 24
```

The choice $`-149` is not the minimum *normal* exponent. Binary32 normal numbers begin at
$`2^{-126}`, but the 23 fraction bits extend the gradual-underflow grid down to $`2^{-149}`.
Encoding that fact in
`fltExp` is what allows the same representability predicate to cover normal and subnormal finite
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
$`2^{76}`. The outputs near the lower end show when the maximum in `fltExp` selects the fixed
floor instead of magnitude minus precision. No special-value encoding is involved in these
calculations. They describe a real-valued grid that can later be related to the finite encodings
of a bit-level format.

Precision and exponent validity also play different roles. A positive precision gives the intended
number of significant radix digits. The `ValidExp` laws justify how the grid behaves when a
rounding step moves a value across a magnitude boundary. Supplying an arbitrary function of the
right type is enough to write an expression, but the generic rounding theorems require those laws.
The format constructors provide them for the supported policies.

# Rounding Rules

A format determines the available real values; a rounding rule selects one to replace an exact
input. The same binary format supports rounding toward negative infinity, toward positive
infinity, toward zero, and to nearest with ties to even.

FloatLib expresses that rounding operation as

$$`\operatorname{round}_{\beta,f,r}(x)
   = \beta^{e}\,r(x\beta^{-e}),
   \qquad e=f(\operatorname{mag}_{\beta}(x))`,

Here $`\beta` is the radix, $`f` is the exponent policy, and
$`\operatorname{mag}_{\beta}(x)` supplies the input's magnitude. Multiplying by $`\beta^{-e}`
expresses the input in units of the selected grid spacing. The function
$`r:\mathbb R\to\mathbb Z` rounds that scaled mantissa to an integer, and multiplication by
$`\beta^e` restores the scale. The type class
`ValidRnd r` records the order properties required of that integer rounder.
`RoundingMode` packages the standard choices used by APIs.

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

The abstract definition lets a proof reason about rounding without choosing an algorithm for it.
FloatLib's `Floats.Formats.Flocq.Calculation` modules connect that definition to calculations on
mantissas and exponents. `Location` and `Inbetween` describe a value's position between
neighbors. The rounding calculation uses this position to select a mantissa, then restores its
power-of-radix scale. The theorem `round_nearestEven_computed` proves that this calculation agrees
with abstract nearest-even rounding for the declared radix and exponent policy.

Calculations starting from an arbitrary Lean real remain noncomputable; executable bit-level
arithmetic is supplied separately by configured `ExecFloat` values such as FloatLib binary32.

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

where `rnd32` is nearest-even. The `@[reducible]` annotation makes the abbreviation transparent,
so a proof can move between the two spellings without a coercion.

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
level to FloatLib binary32.

# Domain Hypotheses For Total Operations

`NF` and `FP32` use Lean's total real operations internally. Real division by zero and square root
or logarithm outside their usual analytical domains therefore do not produce IEEE exceptions, and
the rounded-real format has no upper exponent bound that models overflow to infinity. These choices
keep algebraic definitions total; they do not turn invalid-domain calculations into faithful IEEE
executions.

Accordingly, an IEEE correspondence theorem states finiteness, nonzero-denominator, domain, and
no-overflow hypotheses where needed. At the bit level, `ExecFloat.Binary.toRat?` returns `none`
for NaN
and infinity. The model projection `Model.toReal` maps those encodings to zero and should be used
only under a finiteness hypothesis; interval work that includes infinities uses the extended-real
semantics instead.

A total real function always returns a real value, including at arguments where an IEEE operation
would signal an exception. Its derivative theorem still has to describe that chosen function.
For example, an algebraic inverse or a guarded logarithm cannot acquire the derivative of an
unguarded expression merely because both are written with familiar notation. Domain hypotheses
identify where the ordinary analytic formula applies; outside that region, the declared extension
or guard determines the function that must be analyzed.

# Choosing Executable Precision

The generic `NF` model supports proofs over real values. For an executable scalar, FloatLib exposes
`ExecFloat.Binary`, whose type parameters select the exponent width and stored fraction width.
The normal significand has one additional, implicit bit:

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

These are binary64's 53-bit precision and binary128's 113-bit precision. The type also carries
format-validity proofs: at least two exponent bits, a positive fraction width, and a valid positive
bias. The custom example keeps binary64's exponent width but stores 80 fraction bits, giving
81-bit precision for normal values. Other valid widths use the same interface. Increasing the
fraction width gives finer spacing; increasing the exponent width changes the range. Neither
choice removes rounding or exceptional values, and wider arithmetic can cost more time and memory.

Consider $`1+2^{-100}`. At one, binary64's next larger value is $`1+2^{-52}`, so adding
$`2^{-100}` rounds back to one. Binary128 can retain the increment. We construct $`2^{-100}` by
dividing by the exact integer $`2^{100}`; no native `Float` occurs on the input path:

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

The executable addition has a refinement theorem for all inputs at the selected type:

```lean
example (a b : ChapterBinary128) :
    a + b = ExecFloat.Spec.add a b :=
  ExecFloat.Proof.add_eq_spec a b
```

This equality identifies the operation's reference semantics, including special values. A bound
on distance from real addition additionally needs the relevant finite-path hypotheses and the
format's error theorem. A wider format also does not establish the accuracy of an elementary
function approximation. Configured transcendental functions are an explicit opt-in import,
`FloatLib.Floats.Formats.BinaryInterchange.Configured.Transcendentals`.

TorchLean's `NN.API.Precision` connects configured scalars to typed CPU tensors and model graphs.
That path retains the selected type through parameters, forward values, and JVPs or VJPs.
`nn.sgdStep` also accepts a learning rate and state gradient in that type, allowing an explicit
SGD loop for models without buffer-update hooks. Frozen state remains unchanged; a model that
needs running-statistics updates, such as BatchNorm, is rejected by this entrypoint.
The tensor chapter constructs typed state directly. This preserves precision at initialization as
well as during execution; the supervised trainer's data and reporting interface passes through
native `Float`.

# Encodings And Exceptional Values

FloatLib's `ofBits32` and `toBits32` preserve the raw `UInt32` encoding of a binary32 value.
Its classifiers distinguish zero, subnormal, normal, infinite, and NaN values. Core arithmetic uses
integer and dyadic calculations, so it can be evaluated without delegating the operation to the
host's floating-point instruction. The format parameters fix the encoding, range, and spacing;
the operation's rounding mode fixes how an exact intermediate lands on that grid.

At $`1`, one ULP is $`2^{-23}` and half an ULP is $`2^{-24}`. Here are three increments, a quarter
of a unit in the last place, half a unit, and a full unit, added to the `unit` defined at the top of
the chapter:

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

Those three evaluations are the bit-level version of the spacing picture developed above. The
equality check detects absorption in the first two: the accumulator is unchanged even
though the exact real increment is positive.

```lean (name := absorbQuarter)
-- Absorption asks whether the executable addition leaves
-- its first operand unchanged.
#eval decide (unit + quarterUlpAtOne = unit)
```
```leanOutput absorbQuarter
true
```
```lean (name := absorbHalf)
-- The midpoint is also absorbed when the retained
-- significand is even.
#eval decide (unit + halfUlpAtOne = unit)
```
```leanOutput absorbHalf
true
```
```lean (name := absorbOne)
-- The full spacing is large enough to escape absorption.
#eval decide (unit + oneUlpAtOne = unit)
```
```leanOutput absorbOne
false
```

Each check compares the rounded sum with its left operand. `decide` turns equality of the
configured values into a Boolean, so the observed event is exact equality after rounding.

In an accumulation, absorption means a new term leaves the accumulator unchanged. Repeating this
can lose many small contributions even though each addition is correctly rounded, as happened to
the unit increment in the opening example.

The value-only operations have status-bearing variants. FloatLib's configured
`ExecFloat.Binary.IEEEOutcome` is a pair: the result value followed by an `IEEEStatus` containing
the invalid, divide-by-zero, overflow, underflow, and inexact flags. These calls take an explicit
rounding direction. Underflow follows the documented tininess-after-rounding policy and is raised
only for an inexact tiny result. We project the result's `UInt32` encoding for display:

For $`1` divided by $`+0`, the IEEE rule returns positive infinity and raises the divide-by-zero
flag:

```lean (name := divByZero)
-- A nonzero numerator divided by zero raises divide-by-zero
-- status.
#eval
  let (value, status) := ExecFloat.Binary.divWithStatus
    (1 : ChapterBinary32) (0 : ChapterBinary32) .nearestEven
  (ExecFloat.Binary.toBits32 value, status)
```
```leanOutput divByZero (whitespace := lax)
(2139095040,
 { invalid := false, divideByZero := true,
   overflow := false, underflow := false, inexact := false })
```

For $`0` divided by $`+0`, it returns the canonical NaN and raises `invalid`:

```lean (name := zeroByZero)
-- Zero divided by zero instead takes the invalid-operation
-- branch.
#eval
  let (value, status) := ExecFloat.Binary.divWithStatus
    (0 : ChapterBinary32) (0 : ChapterBinary32) .nearestEven
  (ExecFloat.Binary.toBits32 value, status)
```
```leanOutput zeroByZero (whitespace := lax)
(2143289344,
 { invalid := true, divideByZero := false,
   overflow := false, underflow := false, inexact := false })
```

`2139095040` is `0x7f800000`, positive infinity, and `2143289344` is `0x7fc00000`, the canonical
quiet NaN. Both statuses are derived from the same exact dyadic or rational intermediate used by the
arithmetic operation; no host floating-point instruction is called to guess the flag.

Transcendentals have a different status from basic arithmetic because IEEE 754 does not prescribe
one correctly rounded bit pattern for every elementary function. FloatLib supplies explicit
approximation kernels; the runtime chapter explains how a concrete `libm`,
`libdevice`, or LibTorch implementation can be related to them.

Determinism fixes which answer an algorithm returns. A numerical accuracy claim additionally
needs a range-specific error or interval contract for that function. The executable transcendental
kernels have explicit special-value behavior, but the small Taylor bounds in the rules library do
not by themselves certify every executable `sin`, `cos`, or `tanh` input.

The three absorption results are observations about addition at one particular operand. They do
not define a universal smallest meaningful increment: at a larger magnitude, the local spacing
changes. The displayed equality fixes exactly what was tested: the first operand is unchanged
after the executable addition. It neither estimates the
lost real contribution nor records an error bound for a longer expression.

The two division outcomes distinguish the result from the reason for it. Dividing a nonzero value
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

FloatLib supplies both the configured software arithmetic and its connections to Lean's model.
Use its native adapters directly. Importing a native value and exporting it again recovers the
same logical value:

```lean (name := nativeRoundTripProof)
example (a : Float32) :
    ExecFloat.Binary.toFloat32
      (ExecFloat.Binary.ofFloat32 a) = a :=
  ExecFloat.Binary.toFloat32_ofFloat32 a
```

This theorem ranges over every native value: finite numbers, both signed zeros, infinities, and
Lean's canonical NaN. In the opposite direction, a configured word may carry a NaN payload that
Lean does not retain. Export followed by import canonicalizes that NaN; it leaves finite bits and
zero signs unchanged.

Addition has a different domain from the round trip. Both operands must be finite:

```lean (name := nativeFiniteAddProof)
example (a b : Float32)
    (ha : a.isFinite = true) (hb : b.isFinite = true) :
    ExecFloat.Binary.ofFloat32 (a + b) =
      ExecFloat.Binary.ofFloat32 a +
        ExecFloat.Binary.ofFloat32 b :=
  ExecFloat.Binary.ofFloat32_add_of_isFinite a b ha hb
```

The finite inputs include signed zeros and subnormals. The result has no finiteness premise:
cancellation, rounding, underflow, and overflow to infinity are all included.
`ExecFloat.Binary.ofFloat32_sub_of_isFinite` gives the corresponding subtraction theorem.
Neither statement asserts agreement for an addition or subtraction with a NaN or infinite input.

Square root does cover every configured binary32 word, with native export on both sides:

```lean (name := nativeSqrtProof)
example (x : ExecFloat.Binary 8 23) :
    ExecFloat.Binary.toFloat32 (ExecFloat.sqrt x) =
      (ExecFloat.Binary.toFloat32 x).sqrt :=
  ExecFloat.Binary.toFloat32_sqrt x
```

Negative inputs, signed zeros, infinities, and NaNs are included. Export selects Lean's canonical
NaN representation, so this equality does not preserve arbitrary NaN payloads. Canonicalization is
a representation change, not a second rounding of finite results. FloatLib exposes the same
round-trip, finite-add/sub, and square-root interfaces for binary64 through `ofFloat` and `toFloat`.

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
pure Lean algorithms. Its square-root theorem accounts for NaNs through canonicalization; its
native addition and subtraction theorems require finite inputs. A theorem refining configured
division to `Model.divWithRounding` has a different scope: it does not identify arbitrary native
`Float32` division with that operation.

Integer conversion has its own policy. FloatLib's constructor proofs round arbitrary `Nat` and
`Int` inputs directly, without first converting them through a narrower native format. In the
other direction, native signed-integer casts truncate toward zero and then saturate to the target
range; NaN maps to zero and infinities to the corresponding endpoint. Checked scalar conversion
can instead report exceptional or out-of-range inputs. Agreement with a checked conversion needs
its range hypotheses. None of these logical cast theorems asserts hardware exception flags.

The resulting interfaces expose several different claims:

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
  * Proved for finite operands; the result may overflow.
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

Real decoding deliberately forgets some observations available at the bit level. Both signed
zeros denote the real number zero, and a NaN has no real value to decode faithfully. Consequently,
a real-valued equality cannot by itself establish equality of zero signs or NaN payloads. The
host addition example prints both the displayed value and its bits to show which observation is
being compared. Canonicalization at native export is relevant precisely when raw NaN
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
them applies that same `round32` policy to the exact sum. Fourth, FloatLib addition runs the
bit-level algorithm and returns `0x3f800000`.

The bridge theorem supplies the nontrivial connection:

$$`\operatorname{toReal}
    (\operatorname{add}(a,b))
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
  -> FloatLib configured bit-level operation
  <-> Lean Float32.Model
  -> Lean CPU runtime, native CUDA, or external provider
```

The mathematical layers and both bit-level models are Lean definitions. The last arrow is supplied
by a native backend contract and its validation evidence.

# Exact Subtraction And Sterbenz's Lemma

Sterbenz's lemma says that subtraction can be exact even in floating-point arithmetic. If positive,
representable $`x` and $`y` are within a factor of two,

$$`\frac{y}{2}\leq x\leq 2y`,

then $`x-y` is representable in the same format. FloatLib first proves the fixed-grid and
unbounded-exponent results, then extends the argument to the gradual-underflow `FLT` format. The
extension matters near zero: a proof only about normal values would miss subtraction across the
normal/subnormal boundary.

`FP32.sub_exact_of_sterbenz` specializes the result to finite rounded-real binary32.
FloatLib's `Model.toReal_sub_eq_of_sterbenz` goes further: for an IEEE format, finite bit patterns
are decoded, proved representable on that format's grid, passed through the rounded-real Sterbenz
theorem, and related back to executable subtraction. The theorem derives result finiteness from
the operand hypotheses.

Restating the executable theorem is the clearest way to see what it costs to use. The hypotheses are
positivity, the factor-of-two bounds, and finiteness. The first part of the proof unfolds the
certified subtraction to its model operation. FloatLib's Sterbenz theorem then supplies the exact
real equality; `rfl` checks that the binary32 descriptor is an IEEE format:

```lean (name := sterbenzProof)
-- The factor-of-two condition makes this finite subtraction
-- exact by Sterbenz.
open FloatLib.Floats.Formats.BinaryInterchange in
example (x y : ExecFloat.Binary 8 23)
    (hx : ExecFloat.Binary.isFinite x = true)
    (hy : ExecFloat.Binary.isFinite y = true)
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
    (by rfl) hx hy hxpos hypos hxy hyx
```

Under those hypotheses, subtracting two bit patterns and decoding the result gives the exact real
difference. Running it on $`3-2`, using the
`two` from the opening calculation, shows the inexact flag staying clear:

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
  let (value, status) :=
    ExecFloat.Binary.subWithStatus three two .nearestEven
  (ExecFloat.Binary.toBits32 value, status)
```
```leanOutput sterbenzEval (whitespace := lax)
(1065353216,
 { invalid := false, divideByZero := false,
   overflow := false, underflow := false, inexact := false })
```

The absorption contract has the same shape. It turns the decided equality of two configured
values into a statement about `round32` over the reals, which is the form an error analysis wants:

```lean (name := absorptionProof)
-- Finite dyadic witnesses connect an observed absorbed
-- addition to real rounding.
example {a b : ExecFloat.Binary 8 23}
    {da db : FloatLib.Numerics.Dyadic}
    (ha : Model.toDyadic?
      (ExecFloat.Binary.toModel a) = some da)
    (hb : Model.toDyadic?
      (ExecFloat.Binary.toModel b) = some db)
    (hfin : ExecFloat.Binary.isFinite (a + b) = true)
    (habs : decide (a + b = a) = true) :
    round32 (Model.toReal (ExecFloat.Binary.toModel a) +
      Model.toReal (ExecFloat.Binary.toModel b)) =
      Model.toReal (ExecFloat.Binary.toModel a) :=
  IEEE754.IEEE32Exec.round32_add_eq_left_of_absorbs
    ha hb hfin habs
```

The rounded-real specialization lives in
{src "NN/Floats/FP32/Sterbenz.lean"}[`NN/Floats/FP32/Sterbenz.lean`].
FloatLib's `Formats.BinaryInterchange.Analysis.Sterbenz` supplies the model theorem above.
The executable proof uses FloatLib's subtraction specification and codec round trip directly. The
{src "NN/Floats/IEEEExec/Bridge/Finite.lean"}[finite addition and multiplication bridge]
and {src "NN/Proofs/RuntimeApprox/IEEE32/Contracts.lean"}[absorption contract] supply the other
links used here. In each case, the generic result supplies the rounding argument, the binary32
specialization fixes the format, and the executable bridge connects the decoded bit patterns to
that argument.

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

FloatLib also offers exact accumulation with one final rounding through
`ExecFloat.Binary.sum` and `ExecFloat.Binary.dot`. Those operations have a different semantics from
a tree that rounds at every addition or multiply-add. In the opening example, exact accumulation
keeps the unit contribution until the end; a rounded tree may already have lost it. TorchLean's
rounded reduction-tree statements therefore remain necessary for the corresponding tensor and
backend paths. An exact-accumulation theorem cannot be substituted for their local rounding
premises.

The generic `SumTree` proof in `NN.Proofs.RuntimeApprox.Reductions.Tree` takes a
`RelativeLocalAddBound` for each rounded addition. Its binary32 specialization lives in
`NN.Proofs.RuntimeApprox.Reductions.IEEE32`. There, `sumTreeResult_enclosure` also requires finite
evaluation at every leaf and internal node. The dot-product counterpart, `dotTreeResult_enclosure`,
bounds accumulation error
relative to the exact sum of already-rounded products. Bounding the earlier product errors is
another step; the accumulation theorem does not discard them.

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
execution path. FloatLib connects Lean's logical native models to configured software arithmetic
through the operation-specific theorems above. TorchLean's runtime contracts separately connect
provider results to configured binary32 error bounds. Native instructions require provider evidence.

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
  * `FloatRep` formats
*
  * How does a declared format round exact real arithmetic?
  * `NF`
*
  * What is the binary32-precision rounded-real error?
  * `FP32`
*
  * What bits and IEEE exceptional cases result?
  * `ExecFloat.Binary` with the chosen format
*
  * What does a Lean `Float32` core operation mean in the logic?
  * `Float32.Model`
*
  * Does an interval enclose an operation?
  * FloatLib directed operations or proved interval rounders
*
  * What did CPU, CUDA, or LibTorch execute?
  * runtime result plus an explicit bridge or boundary
:::

The table also helps when a numerical argument gets stuck. If a theorem is cluttered with NaN cases while the
algorithm assumes a finite path, move up to `FP32`. If a proof needs the sign of zero or an
exception flag, use the configured bit-level operations. If two GPU runs disagree, inspect
reduction and
contraction policy before blaming the real-valued model.

To follow one of these calculations into the source, start with the operation it uses:

* `FloatLib.Floats.Formats.BinaryInterchange.Configured` defines executable formats, values,
  and rounding.
* `FloatLib.Floats.Formats.Flocq` contains the generic format, rounding, and error theory. Its
  `Calculation.Round` and `Calculation.Operations` modules supply the rounding calculations.
* `FloatLib.Floats.Formats.IEEE754` connects configured operations to Lean's native logical models.
* `FloatLib.Floats.Interval` supplies generic interval arithmetic.

The tensor arguments live in TorchLean. `NN.Floats.FP32` selects binary32's gradual-underflow
grid; `NN.Proofs.RuntimeApprox.FP32` carries its error bounds through tensor operations.
The reduction-tree modules account for accumulation order. `NN.Spec.Quantization` lifts the
real-scale quantizer to tensors, while `NN.Spec.Quantization.Rational` uses FloatLib's rational
quantizer. The external Arb adapter is available through `NN.Floats.Arb`.

# References

- Lean FRO,
  [`Float32.Model` source
  ](https://github.com/leanprover/lean4/blob/v4.34.0/src/Init/Data/Float/Model/Float32.lean).
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
