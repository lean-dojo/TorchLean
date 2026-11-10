import VersoManual
import TorchLean.NN.LoRA
import TorchLean.NN.Autograd.Function
import TorchLean.NN.Autograd.Frontend
import TorchLean.NN.Autograd.Differential
import TorchLean.NN.Autograd.Complex
import TorchLean.NN.CLI.Training.Command
import TorchLean.NN.Data.Synthetic
import TorchLean.NN.Data.CausalLM
import TorchLean.NN.Models.KAN
import TorchLean.NN.Models.CNN
import TorchLean.NN.Models.ResNet
import TorchLean.NN.Models.UNet
import TorchLean.NN.Models.Recurrent
import TorchLean.NN.Models.CausalTransformer.LanguageModel
import TorchLean.NN.Models.Mamba
import TorchLean.NN.Models.MaskedPatchReconstructor
import TorchLean.NN.Models.Diffusion.NoisePredictor
import TorchLean.NN.Models.FNO
import TorchLean.NN.Models.PPO
import TorchLean.NN.Builder.MoE
import TorchLean.NN.Sequential.Training
import TorchLean.NN.RL.CLI
import TorchLean.NN.RL.PPO.ActorCritic
import TorchLean.NN.SelfSupervised
import TorchLean.NN.Text.Vocabulary
import TorchLean.NN.Text.GPT2BPE
import TorchLean.NN.Text.Generation
import TorchLean.NN.Trainer.Constructor
import TorchLean.NN.Trainer.Train
import TorchLean.NN.Data.Image
import TorchLean.Verify.Lowering
import TorchLeanBlueprint.Bib
import TorchLeanBlueprint.Roles
import TorchLean.NN.Loss
import TorchLean.NN.Session.Random
import TorchLean.Verify.Trainer

open Verso.Genre Manual
open Verso.Genre.Manual.InlineLean
open TorchLean Trainer.Objective
open Lean.Elab.Tactic.GuardMsgs.WhitespaceMode (lax)

open FloatLib.Floats (ExecFloat)
open FloatLib.Floats.Formats.BinaryInterchange (Model FloatFormat)

#doc (Manual) "API Tour" =>
%%%
tag := "api_tour"
file := "A-First-Walk-Through-The-API"
%%%

Application code uses the following imports:

```
-- Application code imports the modules it uses and opens
-- their namespace for the names below.
import TorchLean.NN.Autograd.Function
import TorchLean.NN.Data.Dataset
import TorchLean.NN.Trainer.Constructor
import TorchLean.NN.Trainer.Train

open TorchLean
```

Layers built with `nn` and trained with `Trainer` consume shape-indexed tensors. Those tensors are
also ordinary values, and we can construct and inspect them before running a model, which is useful
for checking how a reshape groups entries or which axis a matrix product sums over.

Named Lean blocks run while this chapter builds, and their outputs are checked against the recorded
messages, but the Python and PyTorch transcripts are separate comparisons (recorded with
`torch 2.13.0+cu130`) that the Lean build doesn't execute.

# Tensors

A TorchLean tensor type has two parameters: the element type and the shape. Lean checks both when
it elaborates the file. So `Tensor Float [2, 3]` is a different type from `Tensor Float [3, 2]`,
and both require `Float` elements, not `Rat` elements.

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

Arithmetic follows the element type. These four literals look almost identical but behave
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

`Tensor ℚ` stores $`1/10` exactly, and its printer shows the fraction. `Tensor Float` stores the
binary64 number nearest to $`0.1`, and its printer rounds that back to six decimals so it reads as
`0.100000`. Rational arithmetic satisfies the field laws exactly, but floating-point rounding can
break them, as the associativity example in the motivation chapter shows.
{Informal.citet goldberg1991}[] is the standard explanation of why.

Casting the Float tensor to Float32 keeps its four-entry shape but rounds its values to the
narrower format. A cast changes the numbers available to subsequent operations, even when the
printed decimals stay the same.

`Float` is Lean's binary64 host type and `Float32` is its binary32 type. For a format whose
precision you choose, use FloatLib. Its binary32 configuration has eight exponent bits and
twenty-three stored fraction bits, and we can inspect its encoding without relying on a decimal
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

In that word, the sign bit is 0, the exponent is `0x7b`, and the fraction bits are `0x4ccccd`: the
literal's encoding matches FloatLib's binary32 result exactly. Comparing an operation would require
checking its output bits too. {ref "floats"}[The floats chapter] explains how that reference is
defined, and {ref "fp32-soundness"}[the FP32 soundness chapter] states what has been proved about
it.

With fifteen exponent bits and 112 fraction bits, the same constructor selects binary128, and
custom widths use that API too (subject to its width and bias constraints).
{ref "tensors-shapes"}[Tensors And Shapes] carries a binary128 value through a typed model's
parameters, output, and derivatives, and that CPU path also supports `nn.sgdStep` in the selected
type. On CPU, `trainer.openTyped` preserves the selected scalar through supervised samples,
reports, and model-state checkpoints (supply typed initial state to retain the extra digits from
the start). But the ordinary trainer's `ieee` arithmetic setting selects binary32, and it isn't a
width parameter.

There's also `Tensor ℝ`, used in specifications and proofs. Mathlib constructs real numbers from
Cauchy sequences, and its general real arithmetic and order are noncomputable. Particular reals
such as zero have finite decimal representations, but this type isn't an executable tensor backend
for numerical training.

Nested brackets show where every value lies along each axis, and the printer keeps that structure:

```lean (name := atCube)
-- Each pair of brackets adds an axis, and indexing the
-- first axis selects a 2-by-2 slice.
def atCube : Tensor Float [2, 2, 2] :=
  [ [ [1, 2], [3, 4] ]
  , [ [5, 6], [7, 8] ] ]

#eval atCube
```

```leanOutput atCube (whitespace := lax)
[[[1.000000, 2.000000], [3.000000, 4.000000]], [[5.000000, 6.000000], [7.000000, 8.000000]]]
```

Tensor literals have to fit the expected shape too. Supplying too few entries produces:

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

Constructing the tensor from an `Array Float` packs its elements into `FloatArray` storage, and the
reshape reuses that row-major payload. A loader first checks any dimensions read from a file, then
constructs the shaped value these operations consume.

# Arithmetic And Contractions

Two vectors and two matrices let us compare elementwise arithmetic with contractions. Elementwise
operations preserve their input shape:

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
$`[4,-10,18]` without summing anything. Division gives $`[1/4,-2/5,1/2]`, and squaring applies the
same operation to each entry of the first vector. That distinction matters when you read model
code: multiplying two tensors of the same shape isn't a request for a matrix product or an inner
product (those operations name the axes that will be summed over).

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

By hand: $`1\cdot4+2\cdot(-5)+3\cdot6=4-10+18=12`. Writing the fold out shows the reduction order,
while `Tensor.dot` provides the specification operation and the `einsum` example below offers a
contraction notation. This tensor fold doesn't record the autograd graph used by the training
runtime.

Matrix products are named after the ranks they combine:

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

For the matrix product, each row of `atM` meets the two columns of `atR`: the first row gives
$`1+3=4` and $`2+3=5`, and the second gives $`10` and $`11`. Then two all-ones vectors expose the
direction of multiplication: `matvec` adds across each row, whereas `vecmat` adds down each column.
Last comes the three-by-three identity matrix.

`matmul` contracts the left matrix's columns with the right matrix's rows, requiring those
dimensions to agree. Since `atM` has shape `[2, 3]`, `Tensor.matmul atM atM` fails that check.

## Expressing Contractions With Index Names

These contractions differ in which axes they sum over and which they keep, and `einsum` expresses
that choice with names for the axes: input indices absent from the output are summed, while output
indices describe the result shape. This notation is scoped syntax, so it needs one extra `open`:

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
patterns use single letters, but the contraction rule is the same:

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

Those carets point into the string literal, because the elaborator parsed the pattern, matched each
subscript against the literal shapes `[2, 3]` and `[2, 3]`, and found `inner` bound to 3 on the
left and 2 on the right. These literal dimensions let the elaborator decide the mismatch. A valid
pattern determines the result shape. Related notation includes `rearrange`, `expand`, `reduce`,
`pack`, and `unpack`, and {ref "tensors-shapes"}[the tensor chapter] works through them.

We can also reduce these entries to a scalar loss, apply ReLU, or select a row:

```lean (name := atReduce)
-- Read these results in order: mean, mean squared error,
-- ReLU, entry, row, entry.
#eval Tensor.mean atU [⟨0, by decide⟩] false
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
$`((-3)^2+7^2+(-3)^2)/3=67/3\approx22.333333`, rounded for display. ReLU keeps the positive entry
and replaces the negative one by zero (the zero stays zero).

Indexing distinguishes a row from an entry. Selecting row one leaves a length-three tensor, while
selecting row one and column two reaches the scalar six (indices are zero-based). A pair of `Fin`
values supplies both coordinates and their bounds, though numerals at `Fin` types can wrap modulo
the bound, as the {ref "lean_ecosystem"}[Lean language chapter] explains.

In PyTorch, the corresponding operations give:

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

PyTorch prints the mean squared error as `22.333333333333332`, while TorchLean rounds its display
to six decimals. Those displays are consistent. But six-decimal agreement alone can't establish
that the stored values are equal, and neither display is the exact rational result $`67/3`.
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
result type is `Tensor Float [2, 6]` (Lean computes the `6` as $`2\times3`). `take` keeps a prefix
of one axis: `take atBatch 1 1` keeps the first row of each block and leaves the batch axis alone,
giving `Tensor Float [2, 1, 3]`. `flattenAfter` names the preserved leading shape, while `take`
receives its axis and count as arguments. Their shape contracts are discussed in
{ref "torchlean_vs_pytorch"}[the comparison chapter].

Going the other way, build an axis instead of removing one:

```lean (name := atPlumbBuild)
-- Build rows from coordinates, join existing rows, then
-- repeat a whole vector.
#eval Tensor.dim fun (i : Fin 3) =>
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

In the first construction, the indices are data: row `i` starts at `2*i`, so the result runs from
zero through five. Concatenation places the existing one-row tensor before the two-row tensor, and
repetition gives two copies of the same vector. Each construction fixes which entries belong to
each leading slice, whereas a reshape can't recover the intended grouping from a differently
ordered payload.

`Tensor.dim` takes a function from an index to a tensor. Its return type fixes the shape of each
selected tensor, so every entry in the stack must have that shape. `concat` joins on the leading
axis and adds the counts (`1 + 2 = 3`) in the result type, and it can also join cache tensors when
their token axis is leading and their remaining shapes agree. `repeatLeading` is broadcasting made
explicit (the counterpart of `expand` in PyTorch), except that it allocates and says so instead of
producing a stride-zero view.

## Slice Bounds As Proof Obligations

`take` asks for a proof that the requested count fits. Its default tactic handles concrete
dimensions, but cannot prove that five rows fit in an axis of length two:

```lean +error (name := atTakeTooMany)
-- Axis 1 has length two, so a request to retain five
-- entries cannot supply hCount.
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

That goal reduces to `False` because the requested count exceeds the dimension. PyTorch slicing
clips this request to the available two rows, but TorchLean's `take` requires the specified count
to fit. With a symbolic shape, the caller may need to supply a proof of the bound, as discussed in
{ref "why_functional"}[the functional programming chapter] and used by the CIFAR crop in
{src "TorchLeanExamples/Models/RealData.lean"}[`RealData.lean`].

## Scalar Access And Updates

Element access comes in two flavors. Indexing with `Fin` values, as in the section above, can't go
out of range because the index type already rules it out. But sometimes the coordinates come from
somewhere else (a parsed file, or a loop bound the type checker can't see), and then the checked
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

A coordinate array holds one index per axis. In `[1, 0, 2]`, the last index fits the final axis of
length three, and replacing it by three produces `none`. `set?` checks coordinates the same way and
returns the updated tensor inside `some` (mapping over that result displays its first slice). And
`atBatch` itself keeps its original first entry.

PyTorch's `x[0, 0, 0] = 99.0` updates the tensor in place, so other holders see the change. For the
read-transform-write case there's also `modify?`, which {ref "running-example"}[the running
example] uses to nudge a single weight for a finite difference check.

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

Seeded initializers infer both matrix dimensions from the type and take the seed as an argument,
and repeating one with the same arguments reproduces its result:

```lean (name := atInitSeeded)
-- The matrix shape supplies fan-out four and fan-in
-- three, and the seed fixes the draw.
#eval (Init.xavierUniform (seed := 2026) :
  Tensor Float [4, 3])
```

```leanOutput atInitSeeded (whitespace := lax)
[[-0.666660, -0.407379, 0.171516],
 [0.589004, -0.860998, -0.009056],
 [0.637224, 0.240239, 0.222714],
 [-0.674830, -0.151913, -0.376110]]
```

`Init.kaimingUniform` infers its matrix dimensions the same way (it uses fan-in scaling for ReLU).
Both work for executable scalars with a `FromFloat` instance. For constant tensors,
`Tensor.full`, `Tensor.zeros`, and `Tensor.ones` are the canonical constructors.

An unrelated draw from a global generator can't change this initializer's result, but reproducing a
complete training run requires controlling data order, runtime arithmetic, and other execution
choices as well as initialization.

# Model Construction

We describe a model with a builder, and supplying a seed turns the builder into a model:

```lean (name := atModel)
-- Build the 2-to-8-to-1 architecture once, then inspect its
-- parameter layout.
def atSmall : nn.Builder (nn.Sequential [2] [1]) :=
  nn.Sequential![
    nn.linear 2 8,
    nn.op .relu,
    nn.linear 8 1
  ]

def atBuilt := nn.build 2026 atSmall

#eval nn.printSummary atBuilt
```

```leanOutput atModel (whitespace := lax)
Sequential: [2] -> [1], layers=3, params=33, state=33
  [0] Linear(2, 8): [2] -> [8] params=24, state=24 [[8, 2], [8]]
  [1] relu: [8] -> [8] params=0, state=0 []
  [2] Linear(8, 1): [8] -> [1] params=9, state=9 [[1, 8], [1]]
```

Every scalar parameter is accounted for:

$$`8\cdot2+8+1\cdot8+1=33.`

The summary also prints the ordered payload shapes, which are available programmatically:

```lean (name := atShapes)
-- Checkpoint tensors must appear in this weight, bias,
-- weight, bias order.
#eval nn.Sequential.stateShapes atBuilt
```

```leanOutput atShapes (whitespace := lax)
[[8, 2], [8], [1, 8], [1]]
```

A checkpoint adapter, a lowering pass, or an importer of foreign weights has to supply tensors in
exactly this order and with exactly these shapes. And because the list is a value, not a comment,
the check is mechanical.

Widening the hidden layer from eight to twelve changes both weight matrices and the first bias.
Counting again gives

$$`12\cdot2+12+1\cdot12+1=49,`

which the summary reports:

```lean (name := atWider)
-- Changing the hidden width changes both affine parameter
-- groups, but keeps [2] -> [1].
def atWider : nn.Builder (nn.Sequential [2] [1]) :=
  nn.Sequential![
    nn.linear 2 12,
    nn.op .relu,
    nn.linear 12 1
  ]

#eval nn.printSummary (nn.build 2026 atWider)
```

```leanOutput atWider (whitespace := lax)
Sequential: [2] -> [1], layers=3, params=49, state=49
  [0] Linear(2, 12): [2] -> [12] params=36, state=36 [[12, 2], [12]]
  [1] relu: [12] -> [12] params=0, state=0 []
  [2] Linear(12, 1): [12] -> [1] params=13, state=13 [[1, 12], [1]]
```

Both networks have four parameter tensors with the same roles but different extents, so the
eight-unit checkpoint doesn't fit the wider one. Matching `[2] → [1]` alone doesn't match model
state.

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

`nn.printSummary` is the counterpart of `print(model)` together with `torchinfo.summary`. What
differs is where the shapes come from: PyTorch reports the shapes of allocated tensors, while
`[2] -> [1]` is part of `atBuilt`'s type and the payload shape list is computed from its layer
definitions. So a layer sequence whose shapes don't compose is a type error, not a run-time
exception.

# Datasets

A supervised dataset pairs one input tensor with one target tensor, and the leading axis counts
examples:

```lean (name := atData)
-- Pair each input row with a length-one target. The
-- leading axis indexes samples.
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

A dataset's type records the shape of *one* input and *one* target. `Data.fromTensors` checks that
both tensors share the leading dimension and strips it, so four rows become four samples of shape
`[2]` and `[1]`.

Flattening the targets to `[4]` removes the output-feature axis, which would give scalar targets
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
  Tensor Float (Shape.dim 4 [1])
in the application
  Data.fromTensors atXs atFlat
```

`Shape.dim 4 [1]` is the elaborator saying what it wanted: four copies of the target shape
`[1]`. Retaining that axis also avoids the `[batch]` versus `[batch, 1]` broadcasting mismatch
in the mean squared error example in {ref "overview"}[the overview].

Loaded data uses the same dataset interface. `Data` exposes CSV, NPY, image, token-stream,
batching, shuffling, and checkpoint helpers (developed in {ref "datasets-loaders"}[the datasets and
loaders chapter]). Shape conversion always happens at a named boundary: runtime dimensions are
validated once, before values enter a statically shaped dataset.

# Trainer Configuration

`Trainer.create` pairs the builder with an objective, an optimizer, and a seed:

```lean (name := atTrainer)
-- The seed initializes the model, and predict below
-- observes it before any optimizer step.
def atTrainer : Trainer [2] [1] :=
  Trainer.create atSmall
    { objective := mse
      optimizer := optim.adam { lr := 0.03 }
      seed := 2026 }

#eval do
  let before ← atTrainer.predict ([0.25, -0.75] :
    Tensor Float [2])
  IO.println s!"prediction before training = {before}"
```

```leanOutput atTrainer (whitespace := lax)
prediction before training = [-0.047640]
```

That number came from the seeded initial weights, with no training at all. The default profile is
checked CPU with native `Float32`, but a configuration may instead select CUDA or the bit-level
FloatLib binary32 reference (see {ref "run-configuration"}[Choosing How A Model Runs]). Device and
provider selection are runtime concerns and don't change the model's input and output shapes.

Training options belong to the call, not to the trainer, because they describe one run:

```lean (name := atOptions)
-- Four samples contribute to each of 200 updates, and
-- logEvery zero suppresses step logs.
def atOptions : Trainer.TrainOptions :=
  { steps := 200
    samplesPerStep := 4
    logEvery := 0 }

#check atOptions
```

```leanOutput atOptions (whitespace := lax)
atOptions : Trainer.TrainOptions
```

With four rows and four samples per step, each update uses the whole dataset, so the update loss
and the reported mean loss cover the same examples here (a smaller `samplesPerStep` would change
that). `steps` counts optimizer updates, not epochs. Each update differentiates the four rows at
the current parameters, averages their gradients, and applies Adam once, following
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
mean_loss(before training) = 1.813294
mean_loss(after training) = 0.000142
steps=200 arithmetic=native scalar=Float32 loss=1.813294 -> 0.000142
prediction after training  = [0.132935]
```

`trainer.train` returns a `Trainer.Result` and doesn't mutate the `atTrainer` definition (TorchLean
values are immutable), so the trained parameters live in the result. From there, `trained.state`
reads the parameters back as `Float` tensors, `trained.save path` writes a checkpoint, and
`Trainer.load atTrainer path atDataset` rebuilds a result from that file. And the summary line
records which arithmetic ran: `native` `Float32` here.

Here's the same run in PyTorch (same architecture, same data, same optimizer settings):

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

Each library uses its own generator, so the runs start with different weights. Both fit the four
training rows with thirty-three parameters over two hundred Adam steps, reaching training losses of
$`1.4\times10^{-4}` in TorchLean and $`1.2\times10^{-5}` in PyTorch, but that agreement says little
about inputs outside the dataset.

Look at the prediction at $`(0.25,-0.75)`. TorchLean says $`0.1329` and PyTorch says $`-0.3289`.
Those four labels alone don't define a target away from their input points. If the intended target
is the affine function $`0.2+0.8(x_1+x_2)`, it gives $`-0.2` here, and both predictions differ from
it despite their small training losses. And the coordinate $`-0.75` lies outside the training input
range too. So the observations constrain the models at four points without determining their
behavior elsewhere. {ref "verification"}[The verification chapters] address properties over
specified regions: a loss of $`1.4\times10^{-4}` on this dataset doesn't establish such a property.

# Verification Lowering

To compute bounds over a region, lower a model and a chosen parameter payload into the graph IR.
Here we use `atBuilt` and its initial state, so the example analyzes the model before training. A
claim about the trained result would need its trained payload instead:

```lean (name := atLower)
-- Lower the initial state explicitly. This payload has
-- not been replaced by trained weights.
def atLowered :
    Except String (IR.LoweredModel Float) :=
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
weights (distinct from the trained state above). For this particular payload, the verifier also
needs a region and an output condition. Lowering returns an error for an unsupported operation or
malformed input, and on success the artifact identifies the node where the input region belongs:

```lean (name := atLowerInput)
-- A successful lowering still carries an input node whose
-- shape the verifier must read.
#eval do
  match atLowered with
  | .error message => IO.println s!"error: {message}"
  | .ok lowered =>
    let shape : Except String Shape :=
      match lowered.graph.nodes[lowered.input.node]? with
      | some (.input _ p) => .ok p.shape
      | _ => .error "no graph input"
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
scripts/lake.sh exe verify -- list
scripts/lake.sh exe verify -- torchlean-ibp
```

At this level, the objects that matter are `IR.Graph`, the parameter payload, the input
region, and the propagation state. Both {ref "graphs-and-ir"}[the graphs and IR chapter] and
{ref "verification"}[the verification chapter] develop them carefully, and {ref "certificates"}[the
certificates chapter] explains what a produced artifact does and doesn't certify.

Note the extra import this section needed:

```
-- Lowering is an additional interface on top of the
-- application API.
import TorchLean.Verify.Lowering
```

The application imports at the start of the chapter come from `TorchLean.NN`, which never imports
`TorchLean.Verify`, so a file that lowers a model for verification adds this import.

# Autograd

Training obtains parameter gradients through reverse-mode automatic differentiation, and you can
also ask the `autograd` interface for derivatives yourself. For a scalar square, both the input and
output tensors have shape `[]`:

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

With the input and output shapes both `[]`, there's one input number and one output number. For a
vector output, a reverse pass would also need to say which combination of output changes matters,
but here the scalar convention chooses weight one, making the returned cotangent $`2x=6`. The
function passed to `autograd.grad` uses runtime references through `nn.functional`, and the
concrete input tensor supplies their values when the function is executed. That separation lets the
runtime record the operations needed for the backward pass.

The quickstart covers scalar input gradients and model-state gradients. For Jacobians, Hessians,
directional derivatives, and gradient stopping, run:

```terminal
# Inspect higher-order derivatives and a detached model loss.
scripts/lake.sh exe torchlean autograd_transforms
```

In that command's output, the state-gradient row is zero because the loss detaches the model output
from the reverse pass. {ref "autograd-walkthrough"}[The autograd walkthrough] derives that
behavior, {ref "scientific-forward-models"}[the scientific models chapter] uses these transforms on
PDE residuals, and {ref "autograd-proofs"}[the autograd proofs chapter] states which derivative
rules are theorems.

# Command Help And Further Reading

Command-specific help lists the runtime options for any example:

```terminal
# Inspect the MLP flags first, then the list of runnable
# examples.
scripts/lake.sh exe torchlean quickstart_mlp --help
scripts/lake.sh exe torchlean --help
```

Top-level help lists the runnable model families and the common `--device`, `--arithmetic`,
`--seed`, and `--show-backend` flags.

For exact names, the generated API page is the declaration index, and in a Lean file `#check` shows
the elaborated type of any name while editor hover reveals its documentation and source.

We develop the state transitions and shape contracts used here in {ref "why_functional"}[the
functional programming chapter] and {ref "lean_ecosystem"}[the Lean language chapter]. For
comparisons of autograd, optimizer conventions, graph import, and numerical behavior, see
{ref "torchlean_vs_pytorch"}[the PyTorch comparison chapter], with {Informal.citep pytorch2019}[]
as the reference for the framework itself and {Informal.citep mathlib2020}[] for the mathematical
library the proof layer builds on.
