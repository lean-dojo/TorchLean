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
import FloatLib
import TorchLean.Tensor.Algebra
import TorchLean.Tensor.Algebra.Real
import TorchLean.Tensor.LinearAlgebra.Norm
import TorchLean.Tensor.Factorization.Orthonormal
import TorchLeanBlueprint.Bib
import TorchLeanBlueprint.Roles
import TorchLean.NN.Loss
import TorchLean.NN.Session.Random
import TorchLean.Verify.Trainer

open Verso.Genre Manual
open Verso.Genre.Manual.InlineLean
open TorchLean
open Shape (size)
open FloatLib.Floats (ExecFloat)

-- Verso compares `leanOutput` blocks against the real compiler message. Two of the
-- signatures below are wider than this file's 100-column limit, so those blocks ask
-- for `whitespace := lax`, which ignores where the expected text was wrapped. The
-- rendered page still shows the message exactly as Lean printed it.
open Lean.Elab.Tactic.GuardMsgs.WhitespaceMode (lax)

#doc (Manual) "Tensors" =>
%%%
tag := "tensors-shapes"
file := "Tensors-That-Remember-Their-Shapes"
%%%

Most tensor libraries carry a shape beside a buffer and check compatibility when an operation runs,
and PyTorch is the reference point for that design {Informal.citep pytorch2019}[]. TorchLean also
carries the shape in the Lean type, so a function that accepts a length-four vector can't
accidentally receive a $`2\times 2` matrix, even though both contain four scalar values.

For a classifier's output, `Tensor α [batch, classes]` puts examples on the first axis and scores
on the second. Our program supplies the names `batch` and `classes`, and the tensor type enforces
their sizes and order. A later layer must accept that shape, or we have to reshape, reduce, or
rearrange it ourselves.

Every marked Lean code block on this page is elaborated when the page is built, and the
`leanOutput` blocks check the compiler results they display. Shell commands and unmarked
reproduction snippets are separate instructions.

# Tensor Values And Shapes

From the repository root, run:

```terminal
# Run the same tensor constructions across the executable
# scalar types shown below.
scripts/lake.sh exe torchlean quickstart_tensors
```

Here's the recorded output:

```terminal +output
== Quickstart: tensor basics ==
Rank-zero tensor: 0.500000
Float tensor:    [0.100000, 0.200000, 0.300000, 0.400000]
Rational tensor: [(1 : Rat)/10, (1 : Rat)/5, (3 : Rat)/10, (2 : Rat)/5]
Integer tensor:  [1, 2, 3, 4]
Float32 cast:    [0.100000, 0.200000, 0.300000, 0.400000]
Rank-3 tensor:   [[[1.000000, 2.000000], [3.000000, 4.000000]],
                  [[5.000000, 6.000000], [7.000000, 8.000000]]]
Reshaped matrix: [[0.100000, 0.200000], [0.300000, 0.400000]]
```

The first line is a rank-zero tensor (written `Tensor Float [] := 0.5`). Four length-four tensors
then show that the same shape can carry different scalar meanings. `ℚ` displays exact fractions,
`Float` is Lean's host binary64 type, and `Float32` uses Lean's native binary32 operations. Last
come a rank-three literal and a shape-checked tensor reshape.

That output comes from the complete program in
{src "TorchLeanExamples/Quickstart/Tensors.lean"}[`Tensors.lean`], and the definitions
that follow develop the same ideas one operation at a time.

Everything from here on assumes these three lines at the top of the file:

```
-- The builder and execution modules also supply the
-- tensor constructors and public shape notation.
import TorchLean.NN.Builder
import TorchLean.NN.Sequential.Execution
open TorchLean
```

These imports cover the application examples. Theorem examples later in the chapter also require
`TorchLean.Tensor.Algebra` and `TorchLean.Tensor.Reduce.Real`, and the explicit float
example uses `FloatLib`. (We write `ExecFloat` for `FloatLib.Floats.ExecFloat`.)

# Tensor Type

A tensor's type answers two independent questions: what one entry is, and how the entries are
indexed. In `Tensor Float [4, 2]`, `Float` answers the first and `[4, 2]` answers the second.
Changing the scalar type can change rounding and the available arithmetic without changing any
coordinate, and changing the shape can change which coordinates are meaningful even when every
stored scalar stays the same. Keeping the two questions apart makes later conversions, reshapes,
and model signatures easier to read.

The canonical type is:

$$`\operatorname{TorchLean.Tensor}\;\alpha\;s`.

Application code writes a tensor type with concrete dimensions as `Tensor α [dims...]`, where the
first parameter is the element type and the second is the dimension list. All four of these are
well-formed types, and Lean accepts them without any further annotation:

```lean
-- Keep the coordinates fixed while changing the scalar
-- semantics of each entry.
-- Lean `Float` values.
example : Type := Tensor Float [4, 2]

-- Mathematical real numbers, for proofs.
example : Type := Tensor ℝ [4, 2]

-- Explicit binary32 bit-level execution.
example : Type := Tensor (ExecFloat.Binary 8 23) [4, 2]

-- A rank-zero `Float` tensor.
example : Type := Tensor Float []
```

Shape-polymorphic code may use a variable for the complete shape, as in `Tensor α s`. It's the same
tensor type, with `s` standing in for the written-out dimension list.

Public tensor annotations use list syntax, and shape operations such as concatenation, axis lookup,
and size calculation preserve that representation in application code.

`[4, 2]` says only that the tensor has four rows and two columns, and it doesn't carry or infer a
dtype. Lean takes the element type from `Tensor`'s first argument, or infers it from the value on
the right-hand side when the annotation is omitted.

Choosing the trainer's arithmetic selects the executable element type that the model is
instantiated at, together with its arithmetic semantics. For example, `arithmetic := .native`
instantiates the model at Lean's native `Float32`, but it doesn't reinterpret an already
constructed `Tensor Float [4, 2]`. Device storage is a further backend contract: TorchLean's CUDA
path stores native `Float32` model values as contiguous binary32 data. Keeping shape, arithmetic
semantics, and device representation distinct prevents a shape annotation from silently deciding
numerical behavior.

Shape literals are written from the outermost dimension to the innermost one:

```lean
-- Dimensions run from the outermost axis inward. The shape
-- [] has no axes, not zero entries.
def rankZeroShape : Shape := []
def vectorShape : Shape := [4]
def matrixShape : Shape := [3, 2]
def rankFourShape : Shape := [8, 3, 32, 32]
```

`[]` is the public rank-zero shape. Shapes print in this same bracket notation, so editor output,
diagnostics, and `repr` never require the inductive constructor spelling.

A rank-zero tensor also uses ordinary numeric syntax:

```lean
-- A rank-zero tensor stores one value, with the literal
-- interpreted at its annotated scalar
-- type.
def threshold : Tensor Float [] := 0.5
def exactCount : Tensor Rat [] := 3
```

An empty dimension list has one coordinate: choosing no axis positions leaves the scalar itself.
That's why `[]` is the shape of a scalar tensor. It differs from `[0]`, a vector with no entries,
and the distinction matters for a loss function, whose output is one scalar even when the input has
many axes. Both evaluations below print that sole value, and the brackets disappear because there's
no vector or matrix axis to display.

```lean (name := thresholdEval)
-- The scalar tensor prints its one Float value without
-- vector brackets.
#eval threshold
```
```leanOutput thresholdEval
0.500000
```

```lean (name := exactCountEval)
-- This scalar uses rational arithmetic, so its integer
-- value needs no floating-point
-- conversion.
#eval exactCount
```
```leanOutput exactCountEval
3
```

`rankFourShape` is a shape value that describes an ordinary rank-four tensor. Axis meanings come
from the operation or model: `[batch, channel, height, width]` and
`[batch, height, width, channel]` are two layout conventions over the same tensor type
{Informal.citep resnet2016}[]. That same tensor core can represent language tokens
{Informal.citep transformer2017}[], PDE grids {Informal.citep fno2021}[], volumetric data, batched
matrices, or an unusual scientific coordinate system.

# Shape-Certified Storage

A tensor owns one row-major buffer. Its erased proof field certifies that the physical buffer
length equals the number of scalar coordinates in its static shape. `Storage α` selects that
physical buffer: `Float` uses `FloatArray`, `UInt8` uses `ByteArray`, and arbitrary scalar types
use the default `Array α` storage unless a custom instance selects another representation.
`FloatArray` and `ByteArray` pack unboxed entries, while generic arrays may hold boxed values. So a
flat representation doesn't give every scalar type the same memory layout or execution cost.

A tensor also acts as a total function from an in-bounds coordinate to a scalar. Statements and
proofs use this observation view, which doesn't allocate a second recursive tensor at runtime. So
the type gives us three facts:

- every axis has the length written in the shape
- every legal index carries its own bounds proof
- definitions and proofs can reason extensionally over all valid coordinates.

For example, `Tensor.map` changes the element type while preserving every dimension:

```lean (name := mapSignature)
-- The source and target scalar types may differ, but map
-- retains the same shape s.
#check Tensor.map
```
```leanOutput mapSignature (whitespace := lax)
TorchLean.Tensor.map.{u, v} {α : Type u} [Storage α] {β : Type v} [Storage β]
  {shape : Shape} (f : α → β) (tensor : Tensor α shape) :
  Tensor β shape
```

Read that signature from the right: `shape` is bound once and appears in both the argument and the
result. So the result shape is fixed before any values are evaluated, and no runtime check can
disagree with it. `Tensor.map` traverses the row-major entries once and preserves the certified
shape.

With two `Storage` arguments, input and output can use different physical representations. At each
coordinate, `map` reads an `α`, applies `f`, and stores a `β`, so a numeric conversion can change
the storage without changing which coordinate an entry occupies.

# Tensor Literals And Shape Checking

Ordinary bracket syntax constructs a tensor whenever the expected type is a positive-rank `Tensor`,
and the elaborator checks every visible dimension:

```lean
-- Corrections line up coordinate by coordinate with the
-- scores, and no axis is broadcast.
def scores : Tensor Float [2, 2] :=
  [[1.0, 2.0], [3.0, 4.0]]

def corrections : Tensor Float [2, 2] :=
  [[0.2, -0.1], [0.0, 0.3]]

def adjusted : Tensor Float [2, 2] :=
  scores + corrections
```

Here the correction at row zero, column one belongs to the score at row zero, column one, and the
resulting entry `2.0 + (-0.1)` is printed below as `1.900000`. Nothing searches for an axis along
which to broadcast the correction, because both tensors already have shape `[2, 2]`. A
matrix-shaped correction spells out the correspondence even if a later version of the program
changes the values or reads them from a file.

Lean's `#eval` command uses the tensor's custom shape-aware `Repr` instance, so you can inspect a
tensor expression with it:

```lean (name := scoresEval)
-- The nested output follows the two rows in the shape
-- annotation.
#eval scores
```
```leanOutput scoresEval
[[1.000000, 2.000000], [3.000000, 4.000000]]
```

```lean (name := adjustedEval)
-- Each displayed entry is the score plus the correction at
-- the same coordinate.
#eval adjusted
```
```leanOutput adjustedEval
[[1.200000, 1.900000], [3.000000, 4.300000]]
```

Neither evaluation involves a container conversion, and `IO.println (reprStr adjusted)` uses the
same rendering in an `IO` program. Use the single typed boundary `Tensor.to value TargetType` only
for serialization, interoperability, and other APIs that ask for a different container:

```lean (name := adjustedArray)
-- Converting to an array exposes row-major order and leaves
-- out the shape annotation.
#eval Tensor.to adjusted (Array Float)
```
```leanOutput adjustedArray
#[1.200000, 1.900000, 3.000000, 4.300000]
```

The flattened array exposes the literal's row-major traversal order: the last index changes fastest,
so the flat order here is `1.2, 1.9, 3.0, 4.3`. `Tensor.to` also accepts `List Float` and
`Vector Float (Shape.size [2, 2])`, and since a vector's length is part of its type, a vector target
states the tensor's certified element count. Packed targets are available when their element type
matches (`FloatArray` for `Float` and `ByteArray` for `UInt8`). Because the target type is an
ordinary Lean argument, you choose the container the receiving API expects.

Rank-zero tensors are the one case that doesn't need brackets:

```lean
-- Loss values can remain tensors even after every axis has
-- been reduced.
def loss : Tensor Float [] := 1.25
def exactLoss : Tensor Rat [] := 5
```

In both definitions, the expected tensor type directs the ordinary numeric literal into the
certified one-element buffer.

Rectangularity and shape compatibility are separate requirements. Each of the next two examples
violates one of them, and Lean rejects both while elaborating the file:

```lean +error (name := ragged)
-- The second row supplies only one entry where the declared
-- shape requires two.
example : Tensor Float [2, 2] :=
  [[1.0, 2.0], [3.0]]
```
```leanOutput ragged
tensor literal has leading dimension 1, but the expected leading dimension is 2
```

In the first example the literal isn't rectangular. Descending into the second row, the elaborator
finds one entry where the shape promises two and reports the dimension it was checking instead of a
generic "bad literal" message.

```lean +error (name := wrongAnnotation)
-- Four entries do not make this rank-two literal a vector:
-- its nesting must also match.
example : Tensor Float [4] :=
  [[1.0, 2.0], [3.0, 4.0]]
```
```leanOutput wrongAnnotation
tensor literal has leading dimension 2, but the expected leading dimension is 4
```

The second literal has four values but the wrong structure: its outermost dimension has length two,
while `[4]` requires a single axis of length four. Equal element counts permit an explicit reshape,
but they don't make a matrix and a vector interchangeable.

When the element type is ambiguous, annotate it:

```lean
-- The annotation fixes rational entries and the two-by-two
-- coordinate layout together.
def exactGrid : Tensor Rat [2, 2] :=
  ([[1, 2], [3, 4]] : Tensor Rat [2, 2])
```

One-dimensional literals use the same notation:

```lean
-- A single pair of brackets supplies the four coordinates
-- of this vector.
def ramp : Tensor Float [4] := [0.0, 1.0, 2.0, 3.0]
```

Literals are for small, hand-written data. Constant tensors come from `Tensor.zeros`,
`Tensor.ones`, and `Tensor.full`. Values that already exist in memory come from `Tensor.from`,
which reads the source's intrinsic shape and so needs no failure branch. And `tensor.reshape`
reinterprets an existing buffer at another shape of the same size. Loaders that receive a claimed
shape from a file or a network payload are the one place where a check is unavoidable, and they get
their own section below.

# Arithmetic On Tensors

A weighted score connects pointwise arithmetic to reduction. Multiplication computes the
contribution of each feature, and summation combines those contributions into a scalar:

```lean
-- Use mixed signs so the reduction visibly combines
-- positive and negative products.
def weights : Tensor Float [3] := [0.5, -1.5, 2.0]
def features : Tensor Float [3] := [4.0, 2.0, 1.0]
```

`*` multiplies matching entries and keeps the shape. It isn't a matrix product:

```lean (name := pairwiseProducts)
-- Multiplication keeps the three separate products,
-- including the negative middle entry.
#eval weights * features
```
```leanOutput pairwiseProducts
[2.000000, -3.000000, 2.000000]
```

`+`, `-`, and `/` behave the same way, and `Tensor.add`, `Tensor.sub`, `Tensor.mul`, and
`Tensor.div` are the named spellings of the same four operations.

Adding those three products is a reduction: it consumes every entry and returns one scalar.

```lean (name := reducedProducts)
-- Summing the products reduces the three entries to a
-- scalar.
#eval Tensor.sum (weights * features)
```
```leanOutput reducedProducts
1.000000
```

Multiply and then sum is the dot product, and TorchLean defines it once under that name:

```lean (name := dotProduct)
-- dot performs that same product-and-sum computation in
-- one operation.
#eval weights.dot features
```
```leanOutput dotProduct
1.000000
```

```lean (name := dotSignature)
-- Both operands share s and α, and the result is an α
-- rather than a Tensor α [].
#check @Tensor.dot
```
```leanOutput dotSignature (whitespace := lax)
@Tensor.dot : {α : Type} → [inst : Storage α] →
  [Add α] → [Mul α] → [Zero α] →
    {s : Shape} → Tensor α s → Tensor α s → α
```

Two details in that signature are deliberate. Because `s` is an arbitrary shape and not a single
axis, the definition is "sum of elementwise products over all coordinates" and it applies to
matrices and higher-rank tensors as written. And the definition really is `Tensor.sum (a * b)`, one
line, which is what the executable path ran to produce `1.000000` above . The theorems in
{src "TorchLean/Tensor/Algebra.lean"}[`TorchLean/Tensor/Algebra.lean`] reach it
through the bridge lemma `Tensor.dot_eq_rep_dot`. Runtime dot and contraction kernels have
separate implementations whose relation to this reference must be established by their contracts.

Here are the same three steps in PyTorch {Informal.citep pytorch2019}[]:

```
# The Python expressions separate elementwise products from
# their scalar reduction.
weights = torch.tensor([0.5, -1.5, 2.0])
features = torch.tensor([4.0, 2.0, 1.0])

weights * features          # tensor([ 2., -3.,  2.])
(weights * features).sum()  # tensor(1.)
torch.dot(weights, features) # tensor(1.)
```

`torch.dot` raises at run time if either argument isn't one-dimensional, while `Tensor.dot`
accepts any shape and rejects a mismatch between the two arguments during elaboration. TorchLean's
definition also fixes the reduction order: `Tensor.sum` folds left to right over the row-major
coordinates. Floating-point addition isn't associative {Informal.citep goldberg1991}[], so a
different fold order is a different function, and pinning it down is what lets the floating-point
chapter state error bounds about this exact sum.

Matrix products are separate operations because they contract an axis instead of matching one:

```lean
-- The direction compares the first and last columns, and
-- the basis forms two column combinations.
def frame : Tensor Float [2, 3] :=
  [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
def direction : Tensor Float [3] := [1.0, 0.0, -1.0]
def basis : Tensor Float [3, 2] :=
  [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
def rowWeights : Tensor Float [2] := [1.0, 1.0]
```

```lean (name := matvecEval)
-- Each row is paired with the three-entry direction,
-- leaving one value per row.
#eval Tensor.matvec frame direction
```
```leanOutput matvecEval
[-2.000000, -2.000000]
```

```lean (name := matmulEval)
-- The shared width three is contracted, leaving two rows
-- and two output columns.
#eval Tensor.matmul frame basis
```
```leanOutput matmulEval
[[4.000000, 5.000000], [10.000000, 11.000000]]
```

For the first result, `direction` subtracts each row's final entry from its first: `1 - 3` and
`4 - 6` both give `-2`. In the matrix product, the columns of `basis` instead select two
combinations of each row. Its first column adds the first and third entries, giving `4` and `10`,
and its second adds the second and third, giving `5` and `11`. So the surviving row axis keeps
length two, while the output's column axis has length two because `basis` has two columns.

```lean (name := vecmatEval)
-- The two row weights combine the matrix rows into one
-- three-entry vector.
#eval Tensor.vecmat rowWeights frame
```
```leanOutput vecmatEval
[5.000000, 7.000000, 9.000000]
```

With both entries equal to one, `rowWeights` adds the rows of `frame`. Each output position is one
column sum, so the three entries are `1 + 4`, `2 + 5`, and `3 + 6`. That's a different contraction
from `matvec`: the length-two input weights rows, while `direction` weights the three columns.

`Tensor.matmul` has type `Tensor α [m, n] → Tensor α [n, p] → Tensor α [m, p]`, so the shared inner
extent is a single variable `n` and a mismatch is a type error, not a runtime exception.
`Tensor.matvec` and `Tensor.vecmat` are the two one-sided cases, while PyTorch reaches all three
through `torch.matmul` and decides which one it means from the ranks it sees at run time.

Whole-tensor reductions ignore the axis structure:

```lean (name := frameSum)
-- sum reduces all six entries, rather than preserving
-- either matrix axis.
#eval Tensor.sum frame
```
```leanOutput frameSum
21.000000
```

```lean (name := frameMean)
-- mean over both axes divides the sum of all six entries by
-- the matrix element count.
#eval Tensor.mean frame
  [⟨0, by decide⟩, ⟨1, by decide⟩] false
```
```leanOutput frameMean
3.500000
```

The sum visits all six entries, giving `1 + 2 + 3 + 4 + 5 + 6 = 21`, and the mean divides that
total by six. `Tensor.mean` takes the axes it averages as a list of `Fin 2` values, so each axis is
in range by construction, and `false` drops them from the result shape (PyTorch's `keepdim`).
TorchLean's general tensor operators name their axes the same way. For an empty tensor, the public
mean uses denominator one and returns zero under the usual scalar operations. That's a totalized
convention, not a statistical mean of samples. Neither result retains a row or column axis, so if
the next calculation needs one number per row, reducing the entire tensor loses the grouping it
needs. A named-axis reduction, shown further down, keeps the row axis and removes only the column
axis.

## No Implicit Broadcasting

NumPy and PyTorch can broadcast a `[3]` vector across the rows of a `[2, 3]` matrix. But
TorchLean's pointwise addition requires matching shapes, so this expression is rejected:

```lean +error (name := noBroadcast)
-- A vector and a matrix have different shapes even when
-- their trailing widths agree.
example := frame + direction
```
```leanOutput noBroadcast (whitespace := lax)
failed to synthesize instance of type class
  HAdd (Tensor Float [2, 3]) (Tensor Float [3])
    ?m.4

Hint: Type class instance resolution failures can be inspected
with the `set_option trace.Meta.synthInstance true` command.
```

To add the vector to every row, spell out the replication first:
`expand direction "column -> row column" with row := 2` produces a `[2, 3]` tensor that `+`
accepts. That pattern says which axis is retained and which is introduced, and the shape-pattern
section below develops the notation for replication, rearrangement, and contraction.

## Dot-Product Theorems

Theorem statements pair tensors with `Tensor.Rep.dot`, the exact finite sum
`∑ i, a i * b i` over all coordinates. Over the reals, symmetry of `Tensor.dot` follows from
the backend-generic algebra layer:

```lean
-- Move from the executable fold to the exact pairing,
-- then use symmetry of the pairing.
example (a b : Tensor ℝ [3]) :
    a.dot b = b.dot a := by
  rw [Tensor.dot_eq_rep_dot, Tensor.dot_eq_rep_dot,
    Tensor.dot_comm]
```

`Tensor.dot a b` is `Tensor.sum (a * b)`, a zero-seeded left fold in row-major order, and
`Tensor.dot_eq_rep_dot` rewrites it to the pairing over any `AddCommMonoid` with a multiplication.
`Tensor.dot_comm` closes the goal. It is stated over any `CommSemiring` from Mathlib
{Informal.citep mathlib2020}[], so the same lemma serves `ℝ`, `ℚ`, and other lawful commutative
semirings. Rounded floating-point arithmetic is not associative and does not satisfy those
hypotheses.

The two forms support different arguments. The fold fixes the execution order used in
floating-point analysis, and the finite sum is what Mathlib's `Finset` lemmas and inductive proofs
work with. We prove their agreement once, so later proofs can use either form. For induction on
the leading axis, `Tensor.dot_eq_sum_unstack` splits the pairing into the sum of the
pairings of the slices.

In the same file, the neighbouring lemmas are the ones an application proof usually needs:
`dot_scale_left` and `dot_scale_right` for scalar multiples, `dot_add_left` and
`dot_add_right` for linearity, `dot_full_zero_left` and `dot_full_zero_right` for the zero
tensor, `dot_vec_eq_sum` to reach an ordinary finite sum, and `dot_contract_matVec` for the
adjoint identity that the backward pass of a linear layer relies on.

# Total Indexing

Specification-level indices use `Fin`, so every index includes a proof that it lies inside its
axis. Indexing therefore doesn't return `Option` or throw an out-of-range exception, and its result
type also records which axis was removed. Runtime APIs that receive an unchecked natural-number
index validate it at the boundary before constructing the corresponding typed operation.

```lean
-- Each Fin index carries the proof that it lies within its
-- own axis.
def counts : Tensor Nat [2, 2] :=
  [[1, 2], [3, 4]]

def rowIndex : Fin 2 := ⟨1, by decide⟩
def columnIndex : Fin 2 := ⟨0, by decide⟩
```

```lean (name := countsIndex)
-- Row one and column zero select the first entry of the
-- second row.
#eval counts[rowIndex][columnIndex]
```
```leanOutput countsIndex
3
```

In each index, `by decide` is the bounds proof, discharged by computation because the bound is a
literal. Shape-polymorphic code puts a hypothesis from the surrounding theorem in that position
instead.

The returned `3` comes from row one, column zero (the second row is `[3, 4]`, and its first entry
is selected next). After the first indexing operation the type is a length-two row tensor, and
after the second it's the scalar element type `Nat`. Because the bounds are attached to the indices
before either selection occurs, a function can accept an arbitrary `Fin 2` without rechecking it,
even when the function can't know which of the two valid positions it will receive.

A complete typed coordinate can be passed to `tensor.at` for a direct scalar read, whereas chained
indexing first constructs the selected row tensor and so can incur the cost of an intermediate
slice. Both immutable operations, `tensor.set coordinate value` and `tensor.modify coordinate f`,
preserve the static tensor type. Packed `Float` and `UInt8` tensors use native copy-on-write
updates (which can reuse a uniquely owned buffer), while other element types use the general
array-backed implementation. Semantic laws for both operations are available to the simplifier,
including `(tensor.set coordinate value).at coordinate = value`.

Unchecked coordinate arrays cross a validation boundary: `tensor.at? #[...]`,
`tensor.set? #[...] value`, and `tensor.modify? #[...] f` return `none` when the rank or a bound is
wrong.

Total indexing is useful in proofs: a theorem about a tensor with one or more dimensions introduces
an arbitrary valid index and applies its induction hypothesis to the smaller tensor selected there.

# Trusted Memory And External Data

An in-memory array already has an intrinsic length, so converting it is total:

```lean
-- The returned tensor records the actual array length in
-- its dependent result type.
def asVector (xs : Array Float) : Tensor Float [xs.size] :=
  Tensor.from xs
```

Here the shape in the result type mentions `xs.size`, which is why no failure branch is needed: the
length is read off the value that was handed over, not promised separately.

That result type is useful even when `xs.size` isn't known until execution. “In the type” doesn't
mean “written as a numeral in the source”, since Lean can express a result whose dimension depends
on an argument's value. A caller that accepts any vector length can use `asVector`, while a caller
that specifically requires four entries needs the extra comparison in `loadTensor4`. Both
constructors take the same array and answer different questions about it.

A file or network payload also claims a shape, and that claim must be checked at the external
boundary:

```lean
-- Only the successful length comparison provides the
-- equality needed for the fixed shape.
def loadTensor4 (xs : Array Float) :
    Except String (Tensor Float [4]) :=
  if h : xs.size = 4 then
    .ok ((Tensor.from xs).reshape [4]
      (by simp [Shape.size, h]))
  else
    .error s!"expected 4 values, found {xs.size}"
```

An input with four entries reaches the successful branch, while a shorter input returns the size
mismatch without constructing a `Tensor Float [4]`:

```lean (name := loadGood)
-- Four supplied values pass the length check and produce a
-- typed tensor inside Except.ok.
#eval loadTensor4 #[1.5, 2.5, 3.5, 4.5]
```
```leanOutput loadGood
Except.ok [1.500000, 2.500000, 3.500000, 4.500000]
```

```lean (name := loadBad)
-- The failing branch reports the observed length without
-- constructing a Tensor Float [4].
#eval loadTensor4 #[1.5]
```
```leanOutput loadBad
Except.error "expected 4 values, found 1"
```

The proof argument to `reshape` connects the runtime check to the static result type.
`Tensor.from xs` has shape `[xs.size]`, so reshaping it to `[4]` requires
`Shape.size [xs.size] = Shape.size [4]`, and `simp [Shape.size, h]` supplies it from the branch
hypothesis `h : xs.size = 4`. Lean solves this obligation by itself when both shapes are literals,
but here one of them isn't, so the proof is written out. That single runtime comparison establishes
the length, and the proof lets downstream code use that fact through the tensor's type.

Only the parser or loader needs the failure branch. Once it has established the scalar count, the
ordinary `Tensor.from` and zero-copy `reshape` operations finish construction.

This is a recurring TorchLean pattern:

```
-- Validation is the step that turns a runtime payload into
-- a value with a checked type.
untyped external payload
  -> parser
  -> runtime validation
  -> typed Lean object
  -> theorem or model API
```

Validation proves something about the accepted payload. It doesn't prove that every future file is
valid, and it doesn't certify the code that produced the file.

# Scalar Types And Arithmetic Semantics

These tensors have the same shape and different semantics:

:::table +header
*
  * Tensor element
  * Meaning
*
  * `Float`
  * executable host floating point
*
  * `Rat` or `ℚ`
  * executable exact rationals
*
  * `Real` or `ℝ`
  * proof-level exact reals
*
  * `ExecFloat.Binary 8 23`
  * FloatLib's configured executable binary32
*
  * `FloatLib.Floats.ExecFloat.Binary e f`
  * executable configured binary arithmetic with `e` exponent bits and `f` fraction bits
*
  * `TorchLean.FP32`
  * rounded-real binary32-precision proof model, with no upper exponent bound or IEEE special
    values
*
  * `TorchLean.Complex (ExecFloat.Binary 8 23)`
  * executable complex scalar with binary32 real and imaginary components
:::

A run's executable arithmetic comes from the trainer's `arithmetic` field. Proofs instantiate
tensors over `ℝ` or `Floats.FP32` directly, and those noncomputable types don't appear as
command-line runtime choices. The generic runtime can execute complex scalar programs, but the
supervised trainer only offers its two real modes. Explicit complex training uses
`autograd.complex.grad` for a real loss, with both components retained in the state, predictions,
and checkpoints. FloatLib formats use the same tensor construction API. Select binary32 and
construct the decimals directly in that format:

```lean
-- Each decimal is rounded into binary32 directly.
def binary32 : Tensor (ExecFloat.Binary 8 23) [3] :=
  [0.1, 0.2, 0.3]
```

```lean (name := binary32Eval)
-- Decode the encodings explicitly. The scalar's internal
-- representation is not the display contract.
#eval binary32.data.map ExecFloat.Binary.toBits32
```
```leanOutput binary32Eval
#[1036831949, 1045220557, 1050253722]
```

FloatLib exposes the stored bit patterns, which makes the rounding in the conversion visible. The
first entry, `1036831949`, is `0x3DCCCCCD`, and the real number it denotes is
$`0.100000001490116119384765625`, not $`0.1`. That gap is how the floating-point chapter introduces
rounding error.

## Choosing Precision For A Tensor And Model

An element type can also select a wider software format. For example, binary128 uses 15 exponent
bits and 112 stored fraction bits, so a normal value has 113 significant bits (including the
implicit leading bit). Shapes and precision answer different questions: `[1]` says there's one
coordinate, and `Binary128` says how arithmetic on that coordinate rounds.

```lean
abbrev TensorBinary128 :=
  FloatLib.Floats.ExecFloat.Binary
    (exponentBits := 15) (fractionBits := 112)

def wideAffine : nn.Sequential [1] [1] :=
  nn.build 0 (nn.linear 1 1)
```

Give the weight and bias the same value $`a=1+2^{-100}`, which makes the model $`x\mapsto ax+a`. At
$`x=2` its output is $`3a` and its derivative with respect to the input is $`a`. All these values
fit in binary128. Binary64 would round the chosen parameter to 1 before the model started, so this
example also checks the input boundary.

The typed graph API accepts a state whose scalar type we choose directly. The model determines
its parameter shapes. Constructing the state with `.fill a` puts `a` in every entry of the weight
and the bias. It does not call an initializer that first generates native `Float` values.

```lean (name := wideChecks)
def wideForwardAndDerivative : IO (Bool × Bool × Bool) := do
  let exact : Rat := 1 + 1 / (2 ^ 100 : Nat)
  let a : TensorBinary128 := Rat.cast exact
  let input : Tensor TensorBinary128 [1] :=
    Tensor.full [1] 2
  let state :
      TensorPack TensorBinary128
        (nn.Sequential.stateShapes wideAffine) :=
    .fill a
  let graph ←
    nn.Sequential.lowerToTypedGraph wideAffine
      (α := TensorBinary128)
  let output := nn.TypedGraphModel.forward graph state input
  -- Zero parameter tangent, unit input tangent:
  -- differentiate with respect to x.
  let inputJvp :=
    nn.TypedGraphModel.jvp graph state .zero
      input (Tensor.full [1] 1)
  let (_, inputVjp) :=
    nn.TypedGraphModel.vjp graph state input
      (Tensor.full [1] 1)
  let decode : TensorBinary128 → Option Rat :=
    FloatLib.Floats.ExecFloat.Binary.toRat?
  pure (
    decode (Tensor.getScalar output ⟨0, by decide⟩) ==
      some (3 * exact),
    decode (Tensor.getScalar inputJvp ⟨0, by decide⟩) ==
      some exact,
    decode (Tensor.getScalar inputVjp ⟨0, by decide⟩) ==
      some exact)

#eval wideForwardAndDerivative
```

```leanOutput wideChecks
(true, true, true)
```

The three comparisons check the forward output, input JVP, and input VJP against the elementary
calculation above. Decoding finite results as exact rationals avoids a final conversion to `Float`
that could hide the extra precision. These checks test this model and input, while a derivative
theorem for a family of models is a separate statement.

That state and VJP can also support a training step. Keep the sample $`x=2`, choose target zero,
and use half squared error, $`L(w,b)=(2w+b)^2/2`. At $`w=b=a`, the prediction is $`3a`, so the
parameter gradients are $`6a` and $`3a`. With learning rate $`1/8`, one SGD update gives $`w'=a/4`,
$`b'=5a/8`, and prediction $`9a/8`. Those dyadic values fit in binary128 as well.

Pass the prediction residual to `nn.TypedGraphModel.vjp` to obtain the state gradient, then call
`nn.sgdStep wideAffine lr state gradient`. That call returns an
`Except String (TensorPack TensorBinary128 (nn.Sequential.stateShapes wideAffine))`, so you either
handle a model-validation error or continue with the updated immutable state. Because the learning
rate is a `TensorBinary128`, this update need not pass through a native `Float`. The maintained
typed training {src "TorchLeanExamples/Quickstart/TypedTraining.lean"}[code] implements this loop
for half squared error. The update preserves frozen state entries and rejects models with
buffer-update hooks (including BatchNorm, whose running statistics need a separate update).

This workflow runs through typed CPU operations. The chapter's imports bring in the configured
scalar instances, but `nn.sgdStep` needs `TorchLean.NN.Sequential.Training`. A supervised loop can
import `TorchLean.NN.Trainer.Session` and open
`trainer.openTyped (α := TensorBinary128) (initialState? := some state)` to preserve typed samples,
losses, predictions, and exact model-state checkpoints on CPU. The
{ref "training-from-scratch"}[training chapter] checks that session interface on an affine model.
Construct inputs and initial state in the chosen type: widening a stored `Float` cannot recover
discarded digits, and seeded initialization still starts from `Float` when no state is supplied. A
wider element type also does not supply a CUDA provider. The {ref "floats"}[floating-point chapter]
explains exact input conversion, format limits, and which arithmetic operations have refinement or
error theorems.

For small formats, check constants as well as inputs. Layer normalization's default tolerance
rounds the exact rational $`10^{-5}` into the selected scalar. With three exponent bits and two
fraction bits it becomes zero, so a constant row can encounter a zero denominator. `nn.layerNorm`
accepts an explicit rational `eps` before graph lowering. For this particular format, $`1/16` is a
representable positive choice (its suitability for a model is a separate numerical decision). Shape
checking and format validity don't prove that a tolerance survives conversion or that the resulting
computation stays finite.

# Mixed Element Types

`Tensor α s` is homogeneous: it can't place an `Int`, FP16 value, and FP32 value in different
entries. But different tensors may have different element types, and pointwise `+`, `-`, `*`, `/`,
`Tensor.add`, `Tensor.sub`, `Tensor.mul`, and `Tensor.div` automatically choose the registered
common type:

```lean
-- The mixed-scalar addition uses the supported
-- UInt8-to-Float operation at each coordinate.
def bytes : Tensor UInt8 [3] := [1, 2, 3]
def offsets : Tensor Float [3] := [0.5, 1.5, 2.5]
def shifted : Tensor Float [3] := bytes + offsets
```

```lean (name := shiftedEval)
-- Each byte value receives the Float offset at the same
-- vector position.
#eval shifted
```
```leanOutput shiftedEval
[1.500000, 3.500000, 5.500000]
```

The result is homogeneous `Float`, with promotion and arithmetic happening together while the
output buffer is built. TorchLean's built-in promotion order is
`UInt8 < Nat < Int < Rat < Float32 < Float`. A custom scalar participates by registering
`ElementCast` and `ElementPromotion` instances. Use `tensor.cast TargetScalar` when conversion is
the operation you intend and not a consequence of binary arithmetic.

That promotion chain covers the registered real numeric scalar families. Proof-oriented or
domain-specific scalars (such as `Real`, configured FloatLib binary formats, and `Complex α`)
remain valid homogeneous tensor elements, but mixed arithmetic for a new pair requires an explicit
promotion instance so TorchLean never invents numerical semantics.

Model programs select one numeric `α` for learned parameters, activations, and outputs, and scalar
polymorphism means the same definition can be interpreted again at another `α`. Integer indices and
boolean masks use dedicated operation interfaces so they aren't silently treated as differentiable
numeric tensors.

# Checked Shape Patterns

The complete shape-pattern API works on the same ordinary `Tensor` type, and the chapter's imports
include all of it. The pattern keywords are scoped syntax activated by `open TorchLean.Tensor`. The
`expand` form is the einops `repeat` operation under a name that does not collide with Lean's
`repeat`:

```lean
-- Axis names describe rearrangement and contraction, and
-- their lengths come from the tensor types.
open TorchLean.Tensor

def matrix : Tensor Float [2, 3] :=
  [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]

def right : Tensor Float [3, 4] :=
  [[1.0, 0.0, 0.0, 1.0],
   [0.0, 1.0, 0.0, 1.0],
   [0.0, 0.0, 1.0, 1.0]]

def transposed :=
  rearrange matrix "row column -> column row"

def tiled :=
  expand matrix "row column -> batch row column"
    with batch := 3

def rowSums :=
  reduce matrix "row column -> row" by sum

def product :=
  einsum matrix, right
    "row contracted, contracted column -> row column"

def packed := pack rowSums, matrix "*"

def restored := unpack packed "*"

def restoredRowSums : Tensor Float [2] :=
  restored ⟨0, by decide⟩

def restoredMatrix : Tensor Float [2, 3] :=
  restored ⟨1, by decide⟩

def dimensions : List (String × Nat) :=
  parse_shape matrix "row column"
```

Lean checks the literal pattern and every static input or output shape while it elaborates the
definition {Informal.citep lean4}[]. Output names select and order dimensions from the inputs, so
the inferred result type records the output shape and a conflicting annotation is rejected before
execution. None of the shape-pattern operations above needs a result annotation, because Lean
infers their tensor shapes, as these checks show:

```lean (name := transposedType)
-- Swapping the named axes changes [2, 3] into [3, 2].
#check transposed
```
```leanOutput transposedType
transposed : Tensor Float [3, 2]
```

```lean (name := productType)
-- The contracted axis disappears, while the row and
-- output-column axes remain.
#check product
```
```leanOutput productType
product : Tensor Float [2, 4]
```

For the definitions above, Lean infers:

:::table +header
*
  * Definition
  * Inferred type
  * Shape calculation
*
  * `transposed`
  * `Tensor Float [3, 2]`
  * `column row` selects the input dimensions in the order $`3,2`
*
  * `tiled`
  * `Tensor Float [3, 2, 3]`
  * the supplied `batch := 3` is inserted before `row column`
*
  * `rowSums`
  * `Tensor Float [2]`
  * reducing `column` removes its dimension and retains `row`
*
  * `product`
  * `Tensor Float [2, 4]`
  * the shared `contracted` dimension is checked as three and removed, and `row column` contributes
    $`2,4`
*
  * `packed.tensor`
  * `Tensor Float [8]`
  * the star flattens and concatenates two plus six scalar entries
*
  * `packed.shapes`
  * `List (List Nat)`
  * runtime metadata records the star portions `[2]` and `[2, 3]`
*
  * `restored 0`
  * `Tensor Float [2]`
  * `unpack` recovers the first statically indexed component
*
  * `restored 1`
  * `Tensor Float [2, 3]`
  * `unpack` recovers the second statically indexed component
*
  * `dimensions`
  * `List (String × Nat)`
  * the value is `[("row", 2), ("column", 3)]`
:::

Here are the transposed tensor and the parsed dimension names:

```lean (name := transposedEval)
-- Transpose changes coordinate order, so the first
-- displayed row collects a former column.
#eval transposed
```
```leanOutput transposedEval
[[1.000000, 4.000000], [2.000000, 5.000000], [3.000000, 6.000000]]
```

```lean (name := dimensionsEval)
-- The named dimensions are read from the existing shape
-- rather than from tensor values.
#eval dimensions
```
```leanOutput dimensionsEval
[("row", 2), ("column", 3)]
```

Each pattern macro binds pattern names to the input dimensions and synthesizes the result type
during elaboration. The definitions above omit result annotations, and the checked `#check` blocks
show the inferred transpose and contraction shapes. Multi-input `einsum` and `pack` automatically
promote registered element types just as ordinary arithmetic does. `pack` retains the exact star
shapes needed by `unpack`, so the round trip doesn't repeat dimensions or expose an anonymous
tensor-metadata pair. `parse_shape` returns ordinary named dimension data (here
`[("row", 2), ("column", 3)]`), not a tensor.

The transpose output is a useful check on the pattern's meaning: `[1, 4]` is the original first
column, printed as a row. Merely reshaping the same six flat values to `[3, 2]` would group them as
`[1, 2]`, `[3, 4]`, `[5, 6]` (a different operation), and the type `[3, 2]` alone can't distinguish
those results. So the pattern supplies the coordinate correspondence, while the type checks that
its dimensions fit.

Packing preserves a different piece of information. Here the flat payload has eight values, but
eight alone doesn't say where the length-two component ends or how to reconstruct the matrix. The
recorded star shapes retain that partition. Choosing component zero or one in `restored` then
recovers its respective type, so downstream code can consume a vector and a matrix without guessing
their original boundaries from a single flattened buffer.

# Tensors And Arrays

TorchLean uses `Tensor` for numerical data. A length-$`n` value has type `Tensor α [n]`, including
when `n` is computed at runtime, and since the shape already records its length, the value doesn't
need a separate `Vector` wrapper. Arrays remain useful for JSON decoding and collections of
heterogeneous records, while numerical algorithms consume tensors after decoding.

Lists in types such as `Tensor α [batch, width]` are shape syntax. They also index dependent
parameter packs whose tensors have different shapes.

`Tensor.window` takes a one-dimensional tensor and constructs a fixed-length slice, filling
positions beyond its end with the supplied padding value:

```lean
-- Both windows start at offset one, and the shorter source
-- needs the explicit padding value.
def context : Tensor Nat [4] :=
  Tensor.window ([10, 20, 30, 40, 50] : Tensor Nat [5])
    4 1 0

def padded : Tensor Nat [4] :=
  Tensor.window ([10, 20] : Tensor Nat [2]) 4 1 0
```

```lean (name := contextEval)
-- This source has four remaining entries, so the window
-- needs no padding.
#eval context
```
```leanOutput contextEval
[20, 30, 40, 50]
```

```lean (name := paddedEval)
-- Only 20 remains after the offset, so the other three
-- positions receive zero.
#eval padded
```
```leanOutput paddedEval
[20, 0, 0, 0]
```

In the type, the result length is the `length` argument. `offset` selects the first input
coordinate, and `pad` fills positions beyond the input. The second call shows why the operation is
total: a window that runs off the end of a short tensor keeps the promised length, and the pad
value is visible in the source and isn't chosen by the implementation. Window construction works
for every registered storage, including packed floats and bytes, exact rationals, executable IEEE
references, and boxed complex values.

At offset one, the long input supplies positions one through four. The short input supplies only
position one, and the other three requested positions become padding. So `[20, 0, 0, 0]` describes
an explicit data transformation, not evidence that the source contained four values. For a sequence
model, the program must still decide whether padding participates in attention or the loss. A fixed
output shape makes the window easy to consume, while the padding policy determines what its values
mean.

# Runtime Tensor Storage

On the CPU, the tensor is also the native representation. Users write and receive `Tensor α shape`
even though a CPU eager node internally stores `SomeTensor α` (which pairs one runtime shape with a
`Tensor α shape`). Erasing the shape this way lets one autograd tape contain activations and
gradients of different shapes in a single array. Recovering a typed tensor requires checking the
stored shape before exposing the value.

`TensorPack α shapes` solves a different problem. It's a statically heterogeneous tuple whose type
keeps the full list of member shapes, and model parameters, gradients, and typed graph contexts use
it when a fixed collection contains tensors of different shapes. It doesn't erase shapes, and it
isn't an alternative input or output tensor type.

For example, the two weights and two biases of a small MLP have four different positions in a known
state layout, and a `TensorPack` can describe that whole layout statically. An eager tape, however,
grows as operations execute and may contain a matrix, a vector, and a scalar loss. `SomeTensor`
lets those entries share an array while retaining each entry's runtime shape. Both wrappers contain
ordinary tensors. What differs is how much of the collection's structure the surrounding program
knows before execution.

A model program does not pass those values around directly while it records a computation. It uses
`Ops.Ref (m := m) (α := α) shape`, a shape-indexed handle owned by one session.
Reading the handle returns the corresponding tensor value. It has no meaning in another session and
is not a second tensor datatype. Session-backed runtime handles carry an owner token and recording
generation. Operations reject handles from another session and handles retained across `resetTape`.

TorchLean's maintained CUDA execution path uses a separate physical representation. A `SomeBuffer`
contains a runtime shape and a native device buffer in contiguous row-major order. Upload and
download functions connect it to `Tensor`, and CUDA tape operations validate the stored shape and
buffer size before dispatch. Proofs about graph evaluation and explicit kernel contracts cover the
operations for which TorchLean has established a bridge. We keep the native CUDA compiler, driver,
and hardware as named external boundaries and don't treat them as consequences of the tensor
definition.

# Linear Layers And Prefix Dimensions

PyTorch's `Linear(in_features, out_features)` acts on the last axis {Informal.citep pytorch2019}[].
TorchLean follows that useful convention while checking the complete map, and one layer constructor
(with a prefix argument when needed) supports:

```
-- The last axis changes width while every leading batch or
-- sequence axis is retained.
[2]              -> [8]
[batch, 2]       -> [batch, 8]
[batch, time, 2] -> [batch, time, 8]
```

Use `nn.linear 2 8`, `nn.linear 2 8 [batch]`, or `nn.linear 2 8 [batch, time]` respectively. The
prefix defaults to `[]` and isn't inferred backwards from a surrounding model annotation. Any
leading shape can serve as the prefix (it need not mean “batch” or “time”). So one linear
definition works for single examples, minibatches, sequences, and higher-rank collections.

At the value level, a linear map behaves the same way, acting on the final input axis and
preserving every leading axis:

```lean
-- One five-by-three weight matrix projects every trailing
-- three-entry activation vector.
def activations : Tensor Float [2, 4, 3] :=
  Tensor.zeros [2, 4, 3]
def projection : Tensor Float [5, 3] :=
  Tensor.zeros [5, 3]
def offset : Tensor Float [5] := Tensor.zeros [5]

def projected := (activations.linear projection offset).relu
```

```lean (name := projectedType)
-- The inferred result preserves [2, 4] and replaces the
-- final width three with five.
#check projected
```
```leanOutput projectedType
projected : Tensor Float (Shape.replaceLast [2, 4, 3] 5)
```

Lean prints the shape computation `Shape.replaceLast [2, 4, 3] 5` without reducing it to
`[2, 4, 5]`. These expressions are definitionally equal: evaluating the shape function gives the
literal shape.

There are eight length-three vectors in `activations` (two choices on the first axis and four on
the second). Each meets the same five rows of `projection`, producing five output features, and
each receives the same length-five bias. Nothing in this operation mixes the eight leading
positions, and ReLU then changes individual output values while preserving all three axes. Zeros
keep the values incidental here, since the `#check` result is about the shape of that whole
composition. An annotation can therefore use the shorter spelling without writing a proof:

```lean
-- The inferred replaceLast shape reduces to the literal [2,
-- 4, 5] without a manual cast.
def projectedAnnotated : Tensor Float [2, 4, 5] :=
  projected
```

Here the input's final extent fixes the input width as three, the weight fixes the output width as
five, and `Tensor.linear` replaces only that final extent. A mismatched weight is rejected during
elaboration, and the message names the exact obligation that failed:

```lean +error (name := widthMismatch)
-- Width four in the weight cannot contract with width three
-- in the activations.
def wrongWeight : Tensor Float [5, 4] :=
  Tensor.zeros [5, 4]

example := activations.linear wrongWeight offset
```
```leanOutput widthMismatch (whitespace := lax)
failed to synthesize instance of type class
  Shape.EndsWith 4 [2, 4, 3]

Hint: Type class instance resolution failures can be inspected
with the `set_option trace.Meta.synthInstance true` command.
```

`Shape.EndsWith 4 [2, 4, 3]` identifies the mismatch: the weight requires a final input width of
four, while the activations have width three.

That check also applies one level up. Change the second linear layer in the quickstart MLP from
`nn.linear 8 1` to `nn.linear 7 1` without changing the preceding layer, and the sequential model
doesn't compose: one layer produces a last dimension of eight while the next requires seven. You
get the error when the model is defined, before initialization, data loading, or training.

# Reshaping

A reshape from `[2,3]` to `[6]` is permitted because both shapes contain six values, but it has to
be written as an operation because the indexing interpretation changes. Conversely, reshaping
`[2,3]` to `[2,4]` is impossible because the element counts differ.

At the representation boundary, the layout convention matters. In row-major order:

$$`\operatorname{flatIndex}(i,j)=3i+j`

for a $`2\times 3` matrix (a column-major native library would use a different equation). Shape
equality doesn't prove layout agreement, so backend capsules record layout requirements separately.

# Proof Observation And Native Storage

Proofs and CPU execution use the same tensor value. A proof observes the tensor through total
coordinate lookup, while generated code traverses its contiguous storage. There's no
materialization step between a proof tensor and a host tensor.

For trusted memory, construction and reshape are separate:

```lean
-- The flat array supplies six entries, and reshape groups
-- consecutive triples into matrix rows.
def raw : Array Float :=
  #[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]

def flat := Tensor.from raw

def grid : Tensor Float [2, 3] := flat.reshape [2, 3]
```

```lean (name := gridEval)
-- The matrix view recovers two rows from the unchanged flat
-- scalar sequence.
#eval grid
```
```leanOutput gridEval
[[1.000000, 2.000000, 3.000000], [4.000000, 5.000000, 6.000000]]
```

```lean (name := gridArray)
-- Converting back to Array returns the same six values in
-- the original order.
#eval Tensor.to grid (Array Float)
```
```leanOutput gridArray
#[1.000000, 2.000000, 3.000000, 4.000000, 5.000000, 6.000000]
```

The round trip returns the original six values in the original order. For `Float`,
`Tensor.from raw` packs the boxed array into a `FloatArray`, and reshape reuses that buffer without
copying. Converting back with `Tensor.to grid (Array Float)` materializes a boxed array, while
requesting `FloatArray` instead exposes the packed buffer directly. CUDA remains a separate device
representation connected by checked upload and download operations.

`grid` provides matrix coordinates, but its array conversion is a flat sequence. An array consumer
that wants to reconstruct those coordinates needs the dimensions separately.

# Cost And Representation Notes

Shape indices remove ambiguity, but they don't erase the cost of an operation. Here's the useful
cost model:

:::table +header
*
  * Operation
  * Proof observation
  * Native execution
*
  * read a full coordinate with `tensor.at`
  * read one valid coordinate
  * compute a row-major offset, then read the buffer
*
  * map
  * visit every scalar and preserve the shape
  * one pass over contiguous storage
*
  * reshape
  * prove equal element counts and reinterpret index structure
  * retain the same certified storage buffer
*
  * matrix multiplication
  * the mathematical sum declared by the spec
  * plain tensor operations use their CPU implementation, while runtime graph operations may
    dispatch to a selected CUDA kernel, cuBLAS call, or other provider
:::

Big-O notation alone can't settle whether two providers agree. Two matrix multiplications may both
take $`O(mnk)` arithmetic operations while accumulating in different orders and returning different
Float32 bits, which is why the graph and backend chapters keep the operation, provider, and
numerical contract separate.

# Comparing Tensor Performance

TorchLean's checked representation doesn't make every scalar family equally fast, so compare the
same operation, scalar type, input shape, and ownership conditions. Allocate inputs before timing,
warm up each implementation, and check numerical results before you compare elapsed times. Record
thread settings as well (a sequential baseline and a task-parallel contraction measure different
execution policies).

Measure end-to-end training separately. Graph construction, parameter initialization, autograd,
and optimizer state can dominate even when individual tensor kernels are fast.

# Tensor Inspection In The Lean Infoview

Open
{src "TorchLeanExamples/DeepDives/TensorOperations.lean"}[
`TorchLeanExamples/DeepDives/TensorOperations.lean`] in VS Code with the Lean extension. Place the
cursor on:

```
-- The value view shows coordinates; the statistics view
-- summarizes the same matrix.
#tensor_view matrix
#tensor_stats_view matrix
```

The first widget renders a tensor and the second summarizes its values. Move the cursor to
`#tensor_view firstRow` to see the shape change after indexing.

# Shape Guarantees

If a model accepts `Tensor α input`, Lean checks that every statically represented layer
composes and that the final result has the declared output shape. It can also check that a parsed
runtime payload has the length promised by its dimensions. That is not a small property: automated
testing of production tensor frameworks keeps finding shape and type inconsistencies that survive
review {Informal.citep nnsmith2023}[].

Shape safety does not, by itself, prove:

- that external memory uses the expected row-major layout
- that two axes have the intended domain meaning
- that a CUDA kernel wrote within bounds
- that an arithmetic operation is numerically correct
- that training converges.

# Model Construction

Layers and model architectures are built from these shape maps (the next chapter's subject), and
the most useful sources to keep nearby are:

- {src "TorchLean/Tensor/README.md"}[TorchLean/Tensor/README.md] for the public tensor surface and
  its focused conversion, construction, printing, and operation modules
- {src "TorchLean/Tensor/Rep.lean"}[TorchLean/Tensor/Rep.lean] for the certified contiguous
  representation
- {src "TorchLean/Tensor/ShapeChange/Flatten.lean"}[TorchLean/Tensor/ShapeChange/Flatten.lean] for
  row-major flattening
- {src "TorchLean/Tensor/Algebra/Real.lean"}[TorchLean/Tensor/Algebra/Real.lean] for algebraic laws
- {src "TorchLean/Tensor/Algebra.lean"}[TorchLean/Tensor/Algebra.lean] for the
  dot-product and matrix-product algebra used above.
