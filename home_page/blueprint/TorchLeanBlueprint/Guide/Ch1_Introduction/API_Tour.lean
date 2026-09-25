import VersoManual
import NN.API
import NN.API.Verification.Lowering
import TorchLeanBlueprint.Bib
import TorchLeanBlueprint.Roles

open Verso.Genre Manual
open Verso.Genre.Manual.InlineLean
open TorchLean
open Lean.Elab.Tactic.GuardMsgs.WhitespaceMode (lax)

open FloatLib.Floats (ExecFloat)
open FloatLib.Floats.Formats.BinaryInterchange (Model FloatFormat)

#doc (Manual) "API Tour" =>
%%%
tag := "api_tour"
file := "A-First-Walk-Through-The-API"
%%%

Application code uses the following import:

```
-- Application code needs the public API and its namespace
-- for the names below.
import NN.API

open TorchLean
```

The layers built with `nn` and trained with `Trainer` consume shape-indexed tensors. Those tensors
are also ordinary values we can construct and inspect directly, before running a model. This is
useful for checking how a reshape groups entries or which axis a matrix product sums over.

Named Lean blocks run while this chapter builds, and their outputs are checked against the
recorded messages. The Python and PyTorch transcripts
are separate comparisons, recorded with `torch 2.13.0+cu130`; the Lean build does not execute them.

# Tensors

A TorchLean tensor type has two parameters: the element type and the shape. Both are checked when
the file is elaborated, so `Tensor Float [2, 3]` is a different type from `Tensor Float [3, 2]`, and
both require `Float` elements rather than `Rat` elements.

Rank-zero tensors use the empty shape and an ordinary numeric literal:

```lean (name := atScalar)
-- A loss can be a tensor with no axes: this one stores the
-- scalar 0.5.
#eval (0.5 : Tensor Float [])
```

```leanOutput atScalar (whitespace := lax)
0.500000
```

A rank-zero tensor has an empty list of axes and one stored element. Representing a loss this
way lets it use the tensor interface, while its shape distinguishes it from a vector or matrix.

The element type selects the arithmetic. These four literals look almost identical and behave
differently:

```lean (name := atScalars)
-- Keep the four entries and shape fixed while changing
-- their scalar representation.
#eval ([0.1, 0.2, 0.3, 0.4] : Tensor Float [4])
#eval ([0.1, 0.2, 0.3, 0.4] : Tensor ℚ [4])
#eval ([1, 2, 3, 4] : Tensor Int [4])
#eval Tensor.cast
  ([0.1, 0.2, 0.3, 0.4] : Tensor Float [4]) Float32
```

```leanOutput atScalars (whitespace := lax)
[0.100000, 0.200000, 0.300000, 0.400000]
```

```leanOutput atScalars (whitespace := lax)
[(1 : Rat)/10, (1 : Rat)/5, (3 : Rat)/10, (2 : Rat)/5]
```

```leanOutput atScalars (whitespace := lax)
[1, 2, 3, 4]
```

```leanOutput atScalars (whitespace := lax)
[0.100000, 0.200000, 0.300000, 0.400000]
```

`Tensor ℚ` stores $`1/10` exactly, and its printer shows the fraction;
`Tensor Float` stores the binary64 number nearest to $`0.1`, and its printer rounds that back to six
decimals so it reads as `0.100000`. Rational arithmetic satisfies the field laws exactly;
floating-point rounding can break them, as the associativity example in the motivation chapter
shows. {Informal.citet goldberg1991}[] is the standard explanation of why.

Casting the Float tensor to Float32 keeps its four-entry shape but rounds its values to the
narrower format. A cast changes the numbers available to subsequent operations, even when the
printed decimals stay the same.

`Float` is Lean's binary64 host type and `Float32` is its binary32 type. For a format whose
precision you choose, use FloatLib directly. Its binary32 configuration has eight exponent bits
and twenty-three stored fraction bits. We can inspect its encoding without relying on a decimal
printer:

```lean (name := atBits)
-- Inspect the binary32 word for 0.1, then decode it for the
-- usual decimal display.
#eval ExecFloat.Binary.toBits32
  (0.1 : ExecFloat.Binary 8 23)
#eval (ExecFloat.Binary.toFloat32
  (0.1 : ExecFloat.Binary 8 23)).toFloat
```

```leanOutput atBits (whitespace := lax)
1036831949
```

```leanOutput atBits (whitespace := lax)
0.100000
```

Python's binary32 encoding provides an independent comparison for this literal:

```
>>> # Round the literal to binary32, then inspect its
>>> # unsigned word and hexadecimal encoding.
>>> import struct
>>> struct.unpack('<I', struct.pack('<f', 0.1))[0]
1036831949
>>> hex(1036831949)
'0x3dcccccd'
```

The sign bit is 0, the exponent is `0x7b`, and the fraction bits are `0x4ccccd`: the literal's
encoding matches FloatLib's binary32 result exactly. Comparing an operation would require checking
its output bits too.
{ref "floats"}[The floats chapter] explains how that reference is defined and
{ref "fp32-soundness"}[the FP32 soundness chapter] states what has been proved about it.

The same constructor with fifteen exponent bits and 112 fraction bits selects binary128.
Custom widths use that API too, subject to its width and bias constraints.
{ref "tensors-shapes"}[Tensors And Shapes] carries a binary128 value through a typed model's
parameters, output, and derivatives. That CPU path also supports `nn.sgdStep` in the selected
type. `trainer.openTyped` preserves the selected scalar through supervised samples, reports, and
model-state checkpoints on CPU. Supply typed initial state to retain the extra digits from the
start. The ordinary trainer's `.ieee` arithmetic setting still selects binary32; it is not a width
parameter.

There is also `Tensor ℝ`, used in specifications and proofs. Mathlib constructs real numbers from
Cauchy sequences; its general real arithmetic and order are noncomputable. Although particular
reals such as zero have finite decimal representations, this type is not an executable tensor
backend for numerical training.

Nested brackets show where every value lies along each axis, and the printer keeps that structure:

```lean (name := atCube)
-- Each pair of brackets adds an axis; indexing the first
-- axis selects a 2-by-2 slice.
def atCube : Tensor Float [2, 2, 2] :=
  [ [ [1, 2], [3, 4] ]
  , [ [5, 6], [7, 8] ] ]

#eval atCube
```

```leanOutput atCube (whitespace := lax)
[[[1.000000, 2.000000], [3.000000, 4.000000]], [[5.000000, 6.000000], [7.000000, 8.000000]]]
```

The expected shape also constrains tensor literals. Supplying too few entries produces:

```lean +error (name := atWrongLiteral)
-- The annotation requests three entries, so this two-entry
-- literal must be rejected.
example : Tensor Float [3] := [1.0, 2.0]
```

```leanOutput atWrongLiteral (whitespace := lax)
tensor literal has leading dimension 2, but the expected leading dimension is 3
```

Lean checks this literal against the declared shape. A PyTorch tensor constructor instead derives
its shape from the supplied data, so a mismatch with an intended dimension needs a later check or
an operation that requires that dimension.

For data that is computed rather than written down, `Tensor.from` builds a vector from an array
using the array's own length, and `reshape` reinterprets a flat buffer at another shape with the
same number of entries:

```lean (name := atFrom)
-- Fill a flat payload first, then group consecutive pairs
-- into matrix rows.
#eval (Tensor.from #[1.0, 2.0, 3.0, 4.0]).reshape [2, 2]
```

```leanOutput atFrom (whitespace := lax)
[[1.000000, 2.000000], [3.000000, 4.000000]]
```

Constructing the tensor from an `Array Float` packs its elements into `FloatArray` storage.
The reshape reuses that row-major payload. A loader first checks any dimensions read from a file,
then constructs the shaped value these operations consume.

# Arithmetic And Contractions

Two vectors and two matrices let us compare elementwise arithmetic with contractions. The
elementwise operations preserve their input shape:

```lean (name := atOps)
-- Reuse these vectors and matrices so the operations below
-- can be checked by hand.
def atU : Tensor Float [3] := [1.0, 2.0, 3.0]
def atV : Tensor Float [3] := [4.0, -5.0, 6.0]
def atM : Tensor Float [2, 3] :=
  [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
def atR : Tensor Float [3, 2] :=
  [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]

#eval Tensor.add atU atV
#eval Tensor.sub atU atV
#eval Tensor.mul atU atV
#eval Tensor.div atU atV
#eval Tensor.square atU
```

```leanOutput atOps (whitespace := lax)
[5.000000, -3.000000, 9.000000]
```

```leanOutput atOps (whitespace := lax)
[-3.000000, 7.000000, -3.000000]
```

```leanOutput atOps (whitespace := lax)
[4.000000, -10.000000, 18.000000]
```

```leanOutput atOps (whitespace := lax)
[0.250000, -0.400000, 0.500000]
```

```leanOutput atOps (whitespace := lax)
[1.000000, 4.000000, 9.000000]
```

The five outputs follow the five calls in order. Addition and subtraction pair entries with the
same index: the middle subtraction is $`2-(-5)=7`. Multiplication also stays entrywise, giving
$`[4,-10,18]`; it has not yet summed anything. Division gives $`[1/4,-2/5,1/2]`, and squaring
applies the same operation to each entry of the first vector. This distinction matters when
reading model code: multiplying two tensors of the same shape is not a request for a matrix
product or an inner product. Those operations name the axes that will be summed over.

A dot product reduces the elementwise product to one scalar. `Tensor.foldl` accumulates entries
in row-major order:

```lean (name := atDot)
-- Multiply corresponding entries, then accumulate the three
-- products from left to right.
def atDot (a b : Tensor Float [3]) : Float :=
  (Tensor.mul a b).foldl (fun acc x => acc + x) 0.0

#eval atDot atU atV
```

```leanOutput atDot (whitespace := lax)
12.000000
```

By hand: $`1\cdot4+2\cdot(-5)+3\cdot6=4-10+18=12`. This explicit fold shows the reduction order;
`Tensor.dotSpec` provides the specification operation, and the `einsum` example below offers a
contraction notation. This direct tensor fold does not record the autograd graph used by the
training runtime.

The matrix products are named after the ranks they combine:

```lean (name := atMat)
-- Compare matrix multiplication, row sums, column sums, and
-- the identity matrix.
#eval Tensor.matmul atM atR
#eval Tensor.matvec atM (Tensor.ones [3])
#eval Tensor.vecmat (Tensor.ones [2]) atM
#eval Tensor.identity (α := Float) 3
```

```leanOutput atMat (whitespace := lax)
[[4.000000, 5.000000], [10.000000, 11.000000]]
```

```leanOutput atMat (whitespace := lax)
[6.000000, 15.000000]
```

```leanOutput atMat (whitespace := lax)
[5.000000, 7.000000, 9.000000]
```

```leanOutput atMat (whitespace := lax)
[[1.000000, 0.000000, 0.000000], [0.000000, 1.000000, 0.000000], [0.000000, 0.000000, 1.000000]]
```

For the matrix product, the first row of `atM` meets the two columns of `atR`, giving
$`1+3=4` and $`2+3=5`. Repeating that calculation for the second row gives $`10` and $`11`.
The next two outputs use all-ones vectors to expose the direction of multiplication: `matvec`
adds across each row, whereas `vecmat` adds down each column. The final output is a separate
three-by-three identity matrix. Its diagonal ones preserve a compatible vector, and its off-diagonal
zeros contribute nothing to the corresponding sums.

`matmul` contracts the left matrix's columns with the right matrix's rows, requiring those
dimensions to agree. Since `atM` has shape `[2, 3]`, `Tensor.matmul atM atM` fails that check.

## Expressing Contractions With Index Names

These contractions differ in which axes they sum over and which they retain. `einsum` expresses
that choice using names for the axes: input indices absent from the output are summed, while
output indices describe the result shape. The notation is scoped syntax and needs one extra `open`:

```lean (name := atEin)
section
open TorchLean.Tensor

-- The dot product from a moment ago, as one contraction.
#eval einsum atU, atV "i, i ->"
-- Exactly `Tensor.matmul atM atR`.
#eval einsum atM, atR
  "row inner, inner col -> row col"
-- Row sums: drop an index and it is summed away.
#eval einsum atM "row col -> row"
-- Nothing summed, just a reordering.
#eval einsum atM "row col -> col row"
-- Indices that meet nowhere: the outer product.
#eval einsum atU, atV "left, right -> left right"

end
```

```leanOutput atEin (whitespace := lax)
12.000000
```

```leanOutput atEin (whitespace := lax)
[[4.000000, 5.000000], [10.000000, 11.000000]]
```

```leanOutput atEin (whitespace := lax)
[6.000000, 15.000000]
```

```leanOutput atEin (whitespace := lax)
[[1.000000, 4.000000], [2.000000, 5.000000],
 [3.000000, 6.000000]]
```

```leanOutput atEin (whitespace := lax)
[[4.000000, -5.000000, 6.000000],
 [8.000000, -10.000000, 12.000000],
 [12.000000, -15.000000, 18.000000]]
```

The first three results reproduce the dot product, matrix product, and matrix-vector product.
In `"row inner, inner col -> row col"`, the shared `inner` index aligns the contracted axes,
then disappears through summation. An index shared across inputs contracts only when omitted
from the output. In the final example both input indices remain, giving an outer product.

TorchLean uses words for subscripts, following the style of einops. PyTorch's corresponding
patterns use single letters; the contraction rule is the same:

```
# The same index strings contract entries, multiply
# matrices, and exchange matrix axes.
print(torch.einsum("i,i->", u, v))
print(torch.einsum("ij,jk->ik", M, R))
print(torch.einsum("ij->ji", M).shape)
```

```
tensor(12.)
tensor([[ 4.,  5.],
        [10., 11.]])
torch.Size([3, 2])
```

When the contracted axes disagree, PyTorch raises
`RuntimeError: einsum(): subscript j has size 2 for operand 1 which does not broadcast
with previously seen size 3`
during the call. TorchLean
checks the corresponding dimensions while elaborating the expression:

```lean +error (name := atEinBad)
-- Reusing M makes the shared axis length three on the left
-- but two on the right.
section
open TorchLean.Tensor

example :=
  einsum atM, atM
    "row inner, inner col -> row col"

end
```

```leanOutput atEinBad (whitespace := lax)
einsum shape error at columns 0-31:
axis 'inner' has incompatible lengths 2 and 3; one must be singleton
row inner, inner col -> row col
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
```

The carets point into the string literal, because the elaborator parsed the pattern, matched each
subscript against the literal shapes `[2, 3]` and `[2, 3]`, and found `inner` bound to 3 on the
left and 2 on the right. These literal dimensions let the elaborator decide the mismatch directly.
A valid pattern determines the result shape. Related notation includes
`rearrange`, `expand`, `reduce`, `pack`, and `unpack`,
and {ref "tensors-shapes"}[the tensor chapter] works through them.

We can also reduce these entries to a scalar loss, apply ReLU, or select a row:

```lean (name := atReduce)
-- Read these results in order: mean, mean squared error,
-- ReLU, entry, row, entry.
#eval Tensor.mean atU
#eval Tensor.meanSquaredError atU atV
#eval Tensor.relu ([-1.0, 0.0, 2.0] : Tensor Float [3])
#eval atU[0]
#eval atM[1]
#eval atM[((1 : Fin 2), (2 : Fin 3))]
```

```leanOutput atReduce (whitespace := lax)
2.000000
```

```leanOutput atReduce (whitespace := lax)
22.333333
```

```leanOutput atReduce (whitespace := lax)
[0.000000, 0.000000, 2.000000]
```

```leanOutput atReduce (whitespace := lax)
1.000000
```

```leanOutput atReduce (whitespace := lax)
[4.000000, 5.000000, 6.000000]
```

```leanOutput atReduce (whitespace := lax)
6.000000
```

Mean squared error squares each difference before taking the mean:
$`((-3)^2+7^2+(-3)^2)/3=67/3`, or
$`\frac13\left(9+49+9\right)=\frac{67}{3}\approx22.333333` with the final decimal rounded
for display.
ReLU keeps the positive entry and replaces the negative one by zero; the zero stays zero.

The indexing calls distinguish a row from an entry. Selecting row one leaves a length-three
tensor; selecting row one and column two reaches the scalar six. Indices are zero-based.
A pair of `Fin` values supplies both coordinates and their bounds, though numerals at `Fin` types
can wrap modulo the bound, as the
{ref "lean_ecosystem"}[Lean language chapter] explains.

The corresponding PyTorch operations give:

```
# Use binary64 tensors to compare these arithmetic
# operations with Lean Float values.
import torch
u = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
v = torch.tensor([4.0, -5.0, 6.0], dtype=torch.float64)
M = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float64)
R = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=torch.float64)
print("dot     ", torch.dot(u, v).item())
print("u * v   ", (u * v).tolist())
print("M @ R   ", (M @ R).tolist())
print("M @ ones", (M @ torch.ones(3, dtype=torch.float64)).tolist())
print("ones @ M", (torch.ones(2, dtype=torch.float64) @ M).tolist())
print("mean u  ", u.mean().item())
print("mse     ", torch.nn.functional.mse_loss(u, v).item())
print("M[1,2]  ", M[1, 2].item())
```

```
dot      12.0
u * v    [4.0, -10.0, 18.0]
M @ R    [[4.0, 5.0], [10.0, 11.0]]
M @ ones [6.0, 15.0]
ones @ M [5.0, 7.0, 9.0]
mean u   2.0
mse      22.333333333333332
M[1,2]   6.0
```

PyTorch prints the mean squared error as `22.333333333333332`, while TorchLean rounds its
display to six decimals. The displays are consistent, but six-decimal agreement alone cannot
establish equality of stored values. Neither display is the exact rational result $`67/3`.
{ref "runtime-approximation"}[The runtime approximation chapter] distinguishes the real-valued
specification, the rounded computation, and its representation.

# Flattening, Slicing, Stacking, And Joining

Before applying arithmetic, a model often needs to arrange its axes: flatten an image for a dense
layer, preserve a batch axis, append to a cache, or select a sequence window. These operations
compute their result shapes from their arguments. A batch of two `2 by 3` blocks shows how
flattening and slicing affect different axes:

```lean (name := atPlumb)
-- Keep the two samples separate while flattening or
-- shortening their inner axes.
def atBatch : Tensor Float [2, 2, 3] :=
  [[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
   [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]]

#eval Tensor.flattenAfter [2] atBatch
#eval Tensor.take atBatch 1 1
```
```leanOutput atPlumb (whitespace := lax)
[[1.000000, 2.000000, 3.000000, 4.000000, 5.000000, 6.000000],
 [7.000000, 8.000000, 9.000000, 10.000000, 11.000000, 12.000000]]
```
```leanOutput atPlumb (whitespace := lax)
[[[1.000000, 2.000000, 3.000000]], [[7.000000, 8.000000, 9.000000]]]
```

`flattenAfter [2]` says which leading shape to hold fixed and flattens everything past it, so the
result type is `Tensor Float [2, 6]` and the `6` is $`2\times3` computed by Lean. `take` keeps a
prefix of one axis; `take atBatch 1 1` keeps the first row of each block and leaves the batch axis
alone, giving `Tensor Float [2, 1, 3]`. `flattenAfter` names the preserved leading shape; `take`
receives an explicit axis and count. Their shape contracts are discussed in
{ref "torchlean_vs_pytorch"}[the comparison chapter].

Going the other way, build an axis instead of removing one:

```lean (name := atPlumbBuild)
-- Build rows from coordinates, join existing rows, then
-- repeat a whole vector.
#eval Tensor.stackLeading fun (i : Fin 3) =>
  Tensor.ofFn fun (j : Fin 2) => (i.val * 2 + j.val).toFloat
#eval Tensor.concat ([[1.0, 2.0]] : Tensor Float [1, 2])
  ([[3.0, 4.0], [5.0, 6.0]] : Tensor Float [2, 2])
#eval Tensor.repeatLeading 2 ([1.0, 2.0] : Tensor Float [2])
```
```leanOutput atPlumbBuild (whitespace := lax)
[[0.000000, 1.000000], [2.000000, 3.000000], [4.000000, 5.000000]]
```
```leanOutput atPlumbBuild
[[1.000000, 2.000000], [3.000000, 4.000000], [5.000000, 6.000000]]
```
```leanOutput atPlumbBuild
[[1.000000, 2.000000], [1.000000, 2.000000]]
```

The first construction uses indices as data: row `i` starts at `2*i`, so the result runs from
zero through five. Concatenation places the existing one-row tensor before the two-row tensor.
Repetition gives two copies of the same vector. Each construction fixes which entries belong to
each leading slice; a reshape cannot recover the intended grouping from a differently ordered
payload.

`stackLeading` takes a function from an index to a tensor. Its return type fixes the shape of
each selected tensor, so every entry in the stack must have that shape. `concat` joins on the
leading axis and adds the counts, `1 + 2 = 3`, in the result type; this is the operation a key-value
cache append is, and {ref "modern-models"}[Modern Models] uses it for exactly that.
`repeatLeading` is broadcasting made explicit, the counterpart of `expand` in PyTorch, except that
it allocates and says so instead of producing a stride-zero view.

## Slice Bounds As Proof Obligations

`take` asks for a proof that the requested count fits. Its default tactic handles concrete
dimensions, but cannot prove that five rows fit in an axis of length two:

```lean +error (name := atTakeTooMany)
-- Axis 1 has length two; a request to retain five entries
-- cannot supply hCount.
def atTooMany : Tensor Float [2, 5, 3] :=
  Tensor.take atBatch 1 5
```
```leanOutput atTakeTooMany
could not synthesize default value for parameter 'hCount' using tactics
```
```leanOutput atTakeTooMany
unsolved goals
⊢ False
```

The goal reduces to `False` because the requested count exceeds the dimension. PyTorch slicing
clips this request to the available two rows; TorchLean's `take` requires the specified count to
fit. With a symbolic shape, the caller may need to supply a proof of the bound, as discussed in
{ref "why_functional"}[the functional programming chapter] and used by the CIFAR crop in
{src "NN/Examples/Models/Common/RealData.lean"}[`RealData.lean`].

## Scalar Access And Updates

Element access comes in two flavors. Indexing with `Fin` values, as in the section above, cannot go
out of range because the index type already rules it out. Sometimes the coordinates come from
somewhere else, a parsed file or a loop bound the type checker cannot see, and then the checked
variants return an `Option`:

```lean (name := atScalarAccess)
-- Dynamic coordinates return an Option, including when an
-- update is out of bounds.
#eval atBatch.at? #[1, 0, 2]
#eval atBatch.at? #[1, 0, 3]
#eval (atBatch.set? #[0, 0, 0] 99.0).map (fun t => t[0])
```
```leanOutput atScalarAccess
some 9.000000
```
```leanOutput atScalarAccess
none
```
```leanOutput atScalarAccess (whitespace := lax)
some [[99.000000, 2.000000, 3.000000], [4.000000, 5.000000, 6.000000]]
```

The coordinate array contains one index per axis. In `[1, 0, 2]`, the last index fits the final
axis of length three; replacing it by three produces `none`. `set?` checks coordinates the same
way and returns the updated tensor inside `some`. Mapping over that result displays its first slice.
The original `atBatch` still has its old first entry.

PyTorch's `x[0, 0, 0] = 99.0` updates the tensor in place, so other holders see the change.
There is also `modify?` for the read-transform-write case;
{ref "running-example"}[the running example] uses it to nudge a single weight for a finite
difference check.

# Tensor Initialization

Initialization also reads the expected type, so callers do not repeat dimensions that the result
type already fixes:

```lean (name := atInitZeros)
-- A deterministic scheme needs only the expected shape to
-- allocate six zeros.
#eval (Init.tensor .zeros : Tensor Float [2, 3])
```

```leanOutput atInitZeros (whitespace := lax)
[[0.000000, 0.000000, 0.000000], [0.000000, 0.000000, 0.000000]]
```

The seeded initializers infer both matrix dimensions from the type and take an explicit seed.
Repeating the initializer with the same arguments reproduces its result:

```lean (name := atInitSeeded)
-- The matrix shape supplies fan-out four and fan-in three;
-- the seed fixes the draw.
#eval (Init.xavierUniform (seed := 2026) :
  Tensor Float [4, 3])
```

```leanOutput atInitSeeded (whitespace := lax)
[[-0.666660, -0.407379, 0.171516],
 [0.589004, -0.860998, -0.009056],
 [0.637224, 0.240239, 0.222714],
 [-0.674830, -0.151913, -0.376110]]
```

`Init.kaimingUniform` is the same call with the ReLU-appropriate gain, and both work for every
executable scalar with a `Runtime.FromFloat` instance. For constant tensors at any scalar type,
`Tensor.full`, `Tensor.zeros`, and `Tensor.ones` are the canonical constructors.

An unrelated draw from a global generator cannot change this initializer's result. Reproducing
a complete training run still requires controlling data order, runtime arithmetic, and other
execution choices as well as initialization.

# Model Construction

A model is described by a builder, and the builder is turned into a model by supplying a seed:

```lean (name := atModel)
-- Build the 2-to-8-to-1 architecture once, then inspect its
-- parameter layout.
def atSmall : nn.Builder (nn.Sequential [2] [1]) :=
  nn.Sequential![
    nn.linear 2 8,
    nn.relu,
    nn.linear 8 1
  ]

def atBuilt := nn.build 2026 atSmall

#eval nn.printSummary atBuilt
```

```leanOutput atModel (whitespace := lax)
Sequential: [2] -> [1], layers=3, params=33, state=33
  [0] Linear(2, 8): [2] -> [8] params=24, state=24 [[8, 2], [8]]
  [1] ReLU: [8] -> [8] params=0, state=0 []
  [2] Linear(8, 1): [8] -> [1] params=9, state=9 [[1, 8], [1]]
```

The summary accounts for every scalar parameter:

$$`8\cdot2+8+1\cdot8+1=33.`

It also prints the ordered payload shapes, which are available programmatically:

```lean (name := atShapes)
-- Checkpoint tensors must appear in this weight, bias,
-- weight, bias order.
#eval nn.stateShapes atBuilt
```

```leanOutput atShapes (whitespace := lax)
[[8, 2], [8], [1, 8], [1]]
```

A checkpoint adapter, a lowering pass, or an importer of foreign weights has to supply tensors in
exactly this order and with exactly these shapes, and because the list is a value rather than a
comment, the check is mechanical.

Widening the hidden layer from eight to twelve changes both weight matrices and the first bias.
The resulting count is

$$`12\cdot2+12+1\cdot12+1=49,`

which the summary reports:

```lean (name := atWider)
-- Changing the hidden width changes both affine parameter
-- groups, but keeps [2] -> [1].
def atWider : nn.Builder (nn.Sequential [2] [1]) :=
  nn.Sequential![
    nn.linear 2 12,
    nn.relu,
    nn.linear 12 1
  ]

#eval nn.printSummary (nn.build 2026 atWider)
```

```leanOutput atWider (whitespace := lax)
Sequential: [2] -> [1], layers=3, params=49, state=49
  [0] Linear(2, 12): [2] -> [12] params=36, state=36 [[12, 2], [12]]
  [1] ReLU: [12] -> [12] params=0, state=0 []
  [2] Linear(12, 1): [12] -> [1] params=13, state=13 [[1, 12], [1]]
```

The change from thirty-three to forty-nine parameters can be reconstructed from the shapes:
the first layer now stores $`12\cdot2+12=36` numbers, and the second stores $`1\cdot12+1=13`.
ReLU stores no weights or biases. The four tensor groups still play the same roles, but their
extents have changed, so a checkpoint for the eight-unit network cannot simply be attached to this
one. Matching the public input and output shapes is only the first part of matching model state.

Returning to the eight-unit model, PyTorch reports the same total and parameter layout:

```
>>> # Inspect the original eight-unit network: four
>>> # parameter tensors containing 33 numbers.
>>> model = nn.Sequential(nn.Linear(2, 8), nn.ReLU(), nn.Linear(8, 1))
>>> print(model)
Sequential(
  (0): Linear(in_features=2, out_features=8, bias=True)
  (1): ReLU()
  (2): Linear(in_features=8, out_features=1, bias=True)
)
>>> sum(p.numel() for p in model.parameters())
33
>>> [(n, tuple(p.shape)) for n, p in model.named_parameters()]
[('0.weight', (8, 2)), ('0.bias', (8,)),
 ('2.weight', (1, 8)), ('2.bias', (1,))]
```

`nn.printSummary` is the counterpart of `print(model)` together with `torchinfo.summary`. The
difference is where the shapes come from: PyTorch reports the shapes of allocated tensors, while
`[2] -> [1]` is part of `atBuilt`'s type, and the payload shape list is computed from its layer
definitions. A layer sequence whose shapes do not compose is a type error, not a run-time exception.

# Datasets

A supervised dataset pairs one input tensor with one target tensor, and the leading axis counts
examples:

```lean (name := atData)
-- Pair each input row with a length-one target; the leading
-- axis indexes samples.
def atXs : Tensor Float [4, 2] :=
  [ [0.0, 0.0]
  , [0.0, 1.0]
  , [1.0, 0.0]
  , [1.0, 1.0] ]

def atYs : Tensor Float [4, 1] :=
  [[0.2], [1.0], [1.0], [1.8]]

def atDataset : Trainer.Dataset [2] [1] :=
  Data.fromTensors atXs atYs

#check atDataset
```

```leanOutput atData (whitespace := lax)
atDataset : Trainer.Dataset [2] [1]
```

The dataset type records the shape of *one* input and *one* target. `Data.fromTensors` checks that
both tensors share the leading dimension and strips it, so four rows become four samples of shape
`[2]` and `[1]`.

Flattening the targets to `[4]` removes the output-feature axis. It would provide scalar targets
where this dataset requires shape `[1]`:

```lean +error (name := atFlatTargets)
-- Removing the target axis changes each sample from shape
-- [1] to scalar shape [].
def atFlat : Tensor Float [4] := [0.2, 1.0, 1.0, 1.8]

example : Trainer.Dataset [2] [1] :=
  Data.fromTensors atXs atFlat
```

```leanOutput atFlatTargets (whitespace := lax)
Application type mismatch: The argument
  atFlat
has type
  Tensor Float [4]
but is expected to have type
  Tensor Float (Shape.prependDim [1] 4)
in the application
  Data.fromTensors atXs atFlat
```

`Shape.prependDim [1] 4` is the elaborator saying what it wanted: four copies of the target shape
`[1]`. Retaining that axis also avoids the `[batch]` versus `[batch, 1]` broadcasting mismatch
in the mean squared error example in {ref "overview"}[the overview].

The same dataset interface accepts loaded data. `Data` exposes CSV, NPY, image, token-stream,
batching, shuffling, and checkpoint helpers, developed in
{ref "datasets-loaders"}[the datasets and loaders chapter]. The shape conversion always happens at a
named boundary: runtime dimensions are validated once, before values enter a statically shaped
dataset.

# Trainer Configuration

`Trainer.new` pairs the builder with an objective, an optimizer, and a seed:

```lean (name := atTrainer)
-- The seed initializes the model; predict below observes it
-- before any optimizer step.
def atTrainer : Trainer [2] [1] :=
  Trainer.new atSmall
    { objective := .mse
      optimizer := optim.adam { learningRate := 0.03 }
      seed := 2026 }

#eval do
  let before ← atTrainer.predict ([0.25, -0.75] :
    Tensor Float [2])
  IO.println s!"prediction before training = {before}"
```

```leanOutput atTrainer (whitespace := lax)
prediction before training = [-0.088261]
```

That number came from the seeded initial weights, with no training at all. The default execution
profile is checked CPU with the eager runtime and native `Float32`; a configuration may instead
select typed graph execution or the bit-level FloatLib binary32 reference, which
{ref "execution-modes"}[the execution modes chapter] covers. Device and provider selection are
runtime concerns and do not change the model's input and output shapes.

Training options belong to the call rather than to the trainer, because they describe one run:

```lean (name := atOptions)
-- Four samples contribute to each of 200 updates; logEvery
-- zero suppresses step logs.
def atOptions : Trainer.TrainOptions :=
  { steps := 200
    samplesPerStep := 4
    logEvery := 0 }

#check atOptions
```

```leanOutput atOptions (whitespace := lax)
atOptions : Trainer.TrainOptions
```

With four rows and four samples per step, each update uses the whole dataset. The update loss and
reported mean loss therefore cover the same examples here; a smaller `samplesPerStep` would change
that. `steps` counts optimizer updates, not epochs. Each update differentiates the four rows at the
current parameters, averages their gradients, and applies Adam once, following
{Informal.citet adamw2019}[] for the decoupled variant and the original Adam update otherwise.

# Training Results

This named block executes the loop during the guide build and checks the recorded output:

```lean (name := atTrain)
-- Train on the four stored rows and inspect the returned
-- trainer at a separate point.
#eval do
  let point : Tensor Float [2] := [0.25, -0.75]
  let trained ← atTrainer.train atDataset atOptions
  trained.printSummary
  let after ← trained.predict point
  IO.println s!"prediction after training  = {after}"
```

```leanOutput atTrain (whitespace := lax)
dataset size = 4
mean_loss(before training) = 1.507706
mean_loss(after training) = 0.000020
steps=200 arithmetic=native scalar=Float32 loss=1.507706 -> 0.000020
prediction after training  = [0.085863]
```

`trainer.train` returns a `Trainer.Result` and does not mutate the `atTrainer` definition; TorchLean
values are immutable, so the trained parameters live in the result. From there, `trained.state`
reads the parameters back as `Float` tensors, `trained.save path` writes a checkpoint, and
`Trainer.load atTrainer path atDataset` rebuilds a result from that file. The summary line also
records which arithmetic ran: `native` `Float32` here.

The same run in PyTorch, same architecture, same data, same optimizer settings:

```
# This is a separate seeded training run; the two libraries
# do not share initial weights.
torch.manual_seed(2026)
model = nn.Sequential(nn.Linear(2, 8), nn.ReLU(), nn.Linear(8, 1))
opt = torch.optim.Adam(model.parameters(), lr=0.03)
loss_fn = nn.MSELoss()
for _ in range(200):
    opt.zero_grad()
    loss = loss_fn(model(xs), ys)
    loss.backward()
    opt.step()
```

```terminal +output
mean_loss(before) = 0.901358
mean_loss(after)  = 0.000012
prediction: [-0.3288894295692444]
```

The runs start with different weights because the libraries use different generators. Both fit
the four training rows with thirty-three parameters over two hundred Adam steps, reaching a
training loss around $`10^{-5}`. That agreement says little about inputs outside the dataset.

Now look at the prediction at $`(0.25,-0.75)`. TorchLean says $`0.0859` and PyTorch says
$`-0.3289`. The four labels alone do not define a target away from their input points. If the
intended target is the affine function $`0.2+0.8(x_1+x_2)`, it gives $`-0.2` here, and both
predictions differ from it despite their small training losses. The coordinate $`-0.75` also lies
outside the training input range. The observations constrain the models at four points without
determining their behavior elsewhere. {ref "verification"}[The verification chapters] address
properties over specified regions: a loss of $`2\times10^{-5}` on this dataset does not establish
such a property.

# Verification Lowering

To compute bounds over a region, lower a model and a chosen parameter payload into the graph IR.
The following example uses `atBuilt` and its initial state, so it analyzes the model before
training. A claim about the trained result would instead need its trained payload:

```lean (name := atLower)
-- Lower the initial state explicitly; this payload has not
-- been replaced by trained weights.
def atLowered :
    Except String (Verification.LoweredIR Float) :=
  Verification.lowerForwardToIR atBuilt
    (nn.initialState atBuilt)

#eval match atLowered with
  | .ok _ => "lowered to graph IR"
  | .error message => s!"error: {message}"
```

```leanOutput atLower (whitespace := lax)
"lowered to graph IR"
```

`atBuilt` supplies the architecture and `nn.initialState atBuilt` supplies fresh initialized
weights, distinct from the trained state above.
The verifier still needs a region and output condition for this particular payload. Lowering
returns an error for an unsupported operation or malformed input; on success, the artifact
identifies the node where the input region belongs:

```lean (name := atLowerInput)
-- A successful lowering still carries an input node whose
-- shape the verifier must read.
#eval do
  match atLowered with
  | .error message => IO.println s!"error: {message}"
  | .ok lowered =>
    let shape := Verification.inputShape? lowered
    IO.println s!"verifier input shape = {shape}"
```

```leanOutput atLowerInput (whitespace := lax)
verifier input shape = ok: [2]
```

From there, interval bound propagation places a box at that node and pushes intervals to the output.
The executable tools have their own runner:

```terminal
# List the available workflows, then run the small
# model-to-IBP example.
lake exe verify -- list
lake exe verify -- torchlean-ibp
```

The objects that matter at this level are `NN.IR.Graph`, the parameter payload, the input region,
and the propagation state. {ref "graphs-and-ir"}[The graphs and IR chapter] and
{ref "verification"}[the verification chapter] develop them carefully, and
{ref "certificates"}[the certificates chapter] explains what a produced artifact does and does not
certify.

Note the extra import this section needed:

```
-- Lowering is an additional interface on top of the
-- application API.
import NN.API.Verification.Lowering
```

`NN.API` exposes the application interfaces; the additional import supplies verification lowering.
The broader `import NN` is available for files that need models, proofs, floating-point semantics,
backend contracts, and verification together.

# Autograd

Training obtains parameter gradients through reverse-mode automatic differentiation. The
`autograd` interface also exposes derivatives directly. For a scalar square, both the input
and output tensors have shape `[]`:

```lean (name := atGrad)
-- Seed the scalar output cotangent with one to obtain the
-- derivative of square at three.
def atSquare : autograd.Function [] [] :=
  fun x => nn.functional.square x

#eval do
  let x : Tensor Float [] := Tensor.full [] 3.0
  let g ←
    autograd.grad (σ := []) (α := Float) atSquare x
  IO.println s!"d/dx x^2 at 3 = {g.item}"
```

```leanOutput atGrad (whitespace := lax)
d/dx x^2 at 3 = 6.000000
```

The input and output shapes are both `[]`, so there is one input number and one output number.
For a vector output, a reverse pass would also need to say which combination of output changes
matters. The scalar convention here chooses weight one, making the returned cotangent simply
$`2x=6`. The function passed to `autograd.grad` uses runtime references through `nn.functional`;
the concrete input tensor supplies their values when the function is executed. That separation lets
the runtime record the operations needed for the backward pass.

The autograd quickstart extends this example to VJPs, Jacobian rows and columns, and a
Hessian-vector product:

```terminal
# Run the scalar-gradient and detach examples through the
# public command-line entry point.
lake exe torchlean quickstart_autograd
```

It also prints a loss and the same loss after `detach`: their forward values agree, but the
detached gradient is zero. {ref "autograd-walkthrough"}[The autograd
walkthrough] derives that behavior, {ref "scientific-forward-models"}[the scientific models chapter]
uses these transforms on PDE residuals, and {ref "autograd-proofs"}[the autograd proofs chapter]
states which derivative rules are theorems.

# Command Help And Further Reading

Command-specific help lists the runtime options for any example:

```terminal
# Inspect the MLP flags first, then the list of runnable
# examples.
lake exe torchlean quickstart_mlp --help
lake exe torchlean --help
```

The top-level help lists the runnable model families and the common `--device`, `--arithmetic`,
`--execution`, `--seed`, and `--show-backend` flags.

For exact names, the generated API page is the declaration index, and in a Lean file `#check` shows
the elaborated type of any name while editor hover reveals its documentation and source.

The state transitions and shape contracts used here are developed in
{ref "why_functional"}[the functional programming chapter] and {ref "lean_ecosystem"}[the Lean
language chapter]. For comparisons of autograd, optimizer conventions, graph import, and numerical
behavior, see
{ref "torchlean_vs_pytorch"}[the PyTorch comparison chapter], with {Informal.citep pytorch2019}[] as
the reference for the framework itself and {Informal.citep mathlib2020}[] for the mathematical
library the proof layer builds on.
