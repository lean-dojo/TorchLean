import VersoManual
import NN.API
import FloatLib
import NN.Proofs.Tensor.Algebra
import NN.Proofs.Tensor.Basic
import TorchLeanBlueprint.Bib
import TorchLeanBlueprint.Roles

open Verso.Genre Manual
open Verso.Genre.Manual.InlineLean
open TorchLean
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

Most tensor libraries carry a shape beside a buffer and check compatibility when an operation
runs; PyTorch is the reference point for that design {Informal.citep pytorch2019}[]. TorchLean also
carries the shape in the Lean type. A function that accepts a length-four vector cannot
accidentally receive a $`2\times 2` matrix, even though both contain four scalar values.

For a classifier's output, `Tensor α [batch, classes]` puts examples on the first axis and scores
on the second. The names `batch` and `classes` come from our program; the tensor type enforces their
sizes and order. A later layer must accept that shape, or we must explicitly reshape, reduce, or
rearrange it.

The marked Lean code blocks are elaborated while this page is built, and `leanOutput` blocks
check the displayed compiler results. Shell commands and unmarked reproduction snippets are
separate instructions.

# Tensor Values And Shapes

From the repository root, run:

```terminal
# Run the same tensor constructions across the executable
# scalar types shown below.
lake exe torchlean quickstart_tensors
```

The output is:

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

The first line is a rank-zero tensor written directly as `Tensor Float [] := 0.5`. The three
length-four tensors show that the same shape can carry different scalar meanings. `ℚ` displays
exact fractions, `Float` is Lean's host binary64 type, and `Float32` uses Lean's native binary32
operations. The final two lines show a rank-three literal and a shape-checked tensor reshape.

The complete program is in
{src "NN/Examples/Quickstart/TensorBasics.lean"}[`TensorBasics.lean`],
and the following definitions develop the same ideas one operation at a time.

Everything from here on assumes these two lines at the top of the file:

```
-- The application import supplies the tensor constructors
-- and public shape notation.
import NN.API
open TorchLean
```

These imports cover the application examples. The theorem examples later in the chapter also
require `NN.Proofs.Tensor.Algebra` and `NN.Proofs.Tensor.Basic`; the explicit float example uses
`FloatLib`. We write `ExecFloat` for `FloatLib.Floats.ExecFloat`.

# Tensor Type

There are two independent questions to ask of a tensor: what is one entry, and how are entries
indexed? In `Tensor Float [4, 2]`, `Float` answers the first and `[4, 2]` answers the second.
Changing the scalar type can change rounding and available arithmetic without changing any
coordinate. Changing the shape can change which coordinates are meaningful even when every
stored scalar remains the same. Keeping these questions separate makes later conversions,
reshapes, and model signatures easier to read.

The canonical type is:

$$`\operatorname{TorchLean.Tensor}\;\alpha\;s`.

The application-facing spelling for concrete dimensions is `Tensor α [dims...]`. The first
parameter is the element type; the second is the dimension list. All four of these are well-formed
types, and Lean accepts them without any further annotation:

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

Shape-polymorphic code may use a variable for the complete shape, as in `Tensor α s`. This is the
same tensor type, with `s` standing for the dimension list rather than writing its entries out.

Public tensor annotations use list syntax. Shape operations such as concatenation, axis lookup, and
size calculation preserve that representation in application code.

`[4, 2]` says only that the tensor has four rows and two columns. It does not carry or infer a
dtype. Lean obtains the element type from `Tensor`'s first argument, or infers it from the value on
the right-hand side when the annotation is omitted.

The trainer's arithmetic choice selects the instantiated executable element type and its arithmetic
semantics. For example, `arithmetic := .native` instantiates the model at Lean 4.34's native
`Float32`; it does not
reinterpret an already constructed `Tensor Float [4, 2]`. Device storage is a further backend
contract: TorchLean's CUDA path stores native `Float32` model values as contiguous binary32 data.
Keeping shape, arithmetic semantics, and device representation distinct prevents a shape annotation
from silently deciding numerical behavior.

Shape literals are written from the outermost dimension to the innermost one:

```lean
-- Dimensions run from the outermost axis inward; [] has no
-- axes, not zero entries.
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
That is why `[]` is the shape of a scalar tensor. It differs from `[0]`, a vector with no entries.
The distinction matters for a loss function, whose output is one scalar even when the input has
many axes. The following two evaluations print that sole value; the brackets disappear because
there is no vector or matrix axis to display.

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

The two printed values already say something about element types. `Float` renders through Lean's
host binary64 formatting, so `0.5` comes back as `0.500000`; `Rat` renders as an exact integer
because three is exactly representable and no decimal expansion is involved.

There is no separate public scalar-tensor format. `[]` says that the tensor has no axes, while the
element type still determines whether its single value is native floating point, exact rational
arithmetic, an executable IEEE reference, or a proof-level number.

`rankFourShape` is a shape value that describes an ordinary rank-four tensor. Axis meanings come
from the
operation or model: `[batch, channel, height, width]` and `[batch, height, width, channel]` are two
layout conventions over the same tensor type {Informal.citep resnet2016}[]. The same tensor core can
represent language tokens {Informal.citep transformer2017}[], PDE grids
{Informal.citep fno2021}[], volumetric data, batched matrices, or an unusual scientific coordinate
system.

# Certified Contiguous Representation

A tensor owns one contiguous row-major buffer. Its erased proof field certifies that the physical
buffer length is exactly the number of scalar coordinates in its static shape. `Storage α` selects
that physical buffer: `Float` uses `FloatArray`, `UInt8` uses `ByteArray`, and arbitrary scalar
types use the default `Array α` storage unless a custom instance selects another representation.

The tensor also acts as a total function from an in-bounds coordinate to a scalar. This observation
view is what statements and proofs use; it does not allocate a second recursive tensor at runtime.
The type therefore gives us three facts:

- every axis has the length written in the shape;
- every legal index carries its own bounds proof;
- definitions and proofs can reason extensionally over all valid coordinates.

For example, `Tensor.map` changes the element type while preserving every dimension:

```lean (name := mapSignature)
-- The source and target scalar types may differ, but map
-- retains the same shape s.
#check Tensor.map
```
```leanOutput mapSignature (whitespace := lax)
TorchLean.Tensor.map {α β : Type} [Storage α] [Storage β]
  {shape : Shape} (f : α → β) (tensor : Tensor α shape) :
  Tensor β shape
```

Read that signature from the right: `shape` is bound once and appears in both the argument and the
result. The result shape is therefore fixed before any values are evaluated, and no runtime check
can disagree with it. `Tensor.map` traverses the contiguous buffer once and preserves the certified
shape.

The two `Storage` arguments allow different physical representations on input and output. At each
coordinate, `map` reads an `α`, applies `f`, and stores a `β`. A numeric conversion can therefore
change the storage without changing which coordinate an entry occupies.

# Tensor Literals And Shape Checking

Ordinary bracket syntax constructs a tensor whenever the expected type is a positive-rank
`Tensor`. The elaborator checks every visible dimension:

```lean
-- Corrections line up coordinate by coordinate with the
-- scores; no axis is broadcast.
def scores : Tensor Float [2, 2] :=
  [[1.0, 2.0], [3.0, 4.0]]

def corrections : Tensor Float [2, 2] :=
  [[0.2, -0.1], [0.0, 0.3]]

def adjusted : Tensor Float [2, 2] :=
  scores + corrections
```

Here the correction at row zero, column one belongs to the score at row zero, column one.
The resulting entry is `2.0 + (-0.1)`, printed below as `1.900000`. There is no search for
an axis along which to broadcast the correction: both tensors already have shape `[2, 2]`.
Writing a matrix-shaped correction makes the correspondence explicit even if a later version of
the program changes the values or reads them from a file.

Lean's `#eval` command uses the tensor's custom shape-aware `Repr` instance, so a tensor expression
can be inspected directly:

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

No container conversion is involved. `IO.println (reprStr adjusted)` uses the same rendering in an
`IO` program. Use the single typed boundary `Tensor.to value TargetType` only for serialization,
interoperability, and other APIs that explicitly request a different container:

```lean (name := adjustedArray)
-- Converting to an array exposes row-major order and leaves
-- out the shape annotation.
#eval Tensor.to adjusted (Array Float)
```
```leanOutput adjustedArray
#[1.200000, 1.900000, 3.000000, 4.300000]
```

The flattened array is the clearest evidence that the literal is stored in row-major order: the
last index changes fastest, so the flat order here is `1.2, 1.9, 3.0, 4.3`. `Tensor.to` also
accepts `List Float` and `Vector Float (Spec.Shape.size [2, 2])`; a vector's length is part of its
type, so the target states the tensor's certified element count. Packed targets are available when
their element type matches: `FloatArray` for `Float` and `ByteArray` for `UInt8`. The target type is
an ordinary Lean argument: choose the container the receiving API expects.

Rank-zero tensors are the one case that does not need brackets:

```lean
-- Loss values can remain tensors even after every axis has
-- been reduced.
def loss : Tensor Float [] := 1.25
def exactLoss : Tensor Rat [] := 5
```

The expected tensor type directs the ordinary numeric literal into the certified one-element
buffer.

Rectangularity and shape compatibility are separate requirements. The next two examples violate
one requirement each, and Lean rejects both while elaborating the file:

```lean +error (name := ragged)
-- The second row supplies only one entry where the declared
-- shape requires two.
example : Tensor Float [2, 2] :=
  [[1.0, 2.0], [3.0]]
```
```leanOutput ragged
tensor literal has leading dimension 1, but the expected leading dimension is 2
```

The literal is not rectangular. The elaborator descends into the second row, finds one entry where
the shape promises two, and reports the dimension it was checking rather than a generic "bad
literal" message.

```lean +error (name := wrongAnnotation)
-- Four entries do not make this rank-two literal a vector;
-- its nesting must also match.
example : Tensor Float [4] :=
  [[1.0, 2.0], [3.0, 4.0]]
```
```leanOutput wrongAnnotation
tensor literal has leading dimension 2, but the expected leading dimension is 4
```

The second literal has four values but the wrong structure: its outermost dimension has length
two, while `[4]` requires a single axis of length four. Equal element counts permit an explicit
reshape; they do not make a matrix and a vector interchangeable.

When the element type is ambiguous, make it explicit:

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
`Tensor.ones`, and `Tensor.full`; values that already exist in memory come from `Tensor.from`,
which reads the source's intrinsic shape and therefore needs no failure branch; and
`tensor.reshape` reinterprets an existing buffer at another shape of the same size. Loaders that
receive a claimed shape from a file or a network payload are the one place where a check is
unavoidable, and they get their own section below.

# Arithmetic On Tensors

A weighted score connects pointwise arithmetic to reduction. Multiplication computes the
contribution of each feature; summation combines those contributions into a scalar:

```lean
-- Use mixed signs so the reduction visibly combines
-- positive and negative products.
def weights : Tensor Float [3] := [0.5, -1.5, 2.0]
def features : Tensor Float [3] := [4.0, 2.0, 1.0]
```

`*` multiplies matching entries and keeps the shape. It is not a matrix product:

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
-- dotSpec performs that same product-and-sum computation in
-- one operation.
#eval weights.dotSpec features
```
```leanOutput dotProduct
1.000000
```

```lean (name := dotSignature)
-- Both operands share s and α; the result is an α rather
-- than a Tensor α [].
#check @Tensor.dotSpec
```
```leanOutput dotSignature (whitespace := lax)
@Tensor.dotSpec : {α : Type} → [inst : Storage α] →
  [Context α] → {s : Shape} → Tensor α s → Tensor α s → α
```

Two details in that signature are deliberate. The shape `s` is arbitrary rather than a single axis,
so the definition is "sum of elementwise products over all coordinates" and it applies to matrices
and higher-rank tensors as written. And the definition really is `sumSpec (mulSpec a b)`: one line,
used both by the executable path that produced `1.000000` above and by the theorems in
{src "NN/Proofs/Tensor/Algebra.lean"}[`NN/Proofs/Tensor/Algebra.lean`]. The `Spec` suffix marks a
definition that specifications and proofs quote directly. It does not mean "not executable", and
runtime dot and contraction kernels have separate implementations whose relation to this
reference must be established by their contracts.

The three lines that follow are the PyTorch spelling of the same three steps
{Informal.citep pytorch2019}[]:

```
# The Python expressions separate elementwise products from
# their scalar reduction.
weights = torch.tensor([0.5, -1.5, 2.0])
features = torch.tensor([4.0, 2.0, 1.0])

weights * features          # tensor([ 2., -3.,  2.])
(weights * features).sum()  # tensor(1.)
torch.dot(weights, features) # tensor(1.)
```

`torch.dot` raises at run time if either argument is not
one-dimensional, while `Tensor.dotSpec` accepts any shape and rejects a mismatch between the two
arguments during elaboration. The reduction order is also part of TorchLean's definition:
`Tensor.sum` folds left to right over the row-major coordinates. Floating-point addition is not
associative {Informal.citep goldberg1991}[], so a different fold order is a different function, and
pinning it down is what lets the floating-point chapter state error bounds about this exact sum.

Matrix products are separate operations because they contract an axis instead of matching one:

```lean
-- The direction compares the first and last columns; the
-- basis forms two column combinations.
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

The first result uses `direction` to subtract each row's final entry from its first:
`1 - 3` and `4 - 6` both give `-2`. In the matrix product, the columns of `basis` instead select
two combinations of each row. Its first column adds the first and third entries, giving `4` and
`10`; its second adds the second and third, giving `5` and `11`. The surviving row axis therefore
still has length two, while the new column axis has length two because `basis` has two columns.

```lean (name := vecmatEval)
-- The two row weights combine the matrix rows into one
-- three-entry vector.
#eval Tensor.vecmat rowWeights frame
```
```leanOutput vecmatEval
[5.000000, 7.000000, 9.000000]
```

The two ones in `rowWeights` add the rows of `frame`. Each output position is one column sum,
so the three entries are `1 + 4`, `2 + 5`, and `3 + 6`. This is a different contraction from
`matvec`: the length-two input weights rows, while `direction` weights the three columns.

`Tensor.matmul` has type `Tensor α [m, n] → Tensor α [n, p] → Tensor α [m, p]`, so the shared inner
extent is a single variable `n` and a mismatch is a type error rather than a runtime exception.
`Tensor.matvec` and `Tensor.vecmat` are the two one-sided cases; PyTorch reaches all three through
`torch.matmul` and decides which one it means from the ranks it sees at run time.

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
-- mean divides the sum of all six entries by the matrix
-- element count.
#eval Tensor.mean frame
```
```leanOutput frameMean
3.500000
```

The sum visits all six entries, giving `1 + 2 + 3 + 4 + 5 + 6 = 21`; the mean divides that
total by six. Neither result retains a row or column axis. If the next calculation needs one
number per row, reducing the entire tensor loses the grouping it needs. The named-axis reduction
below keeps the row axis and removes only the column axis.

## No Implicit Broadcasting

NumPy and PyTorch can broadcast a `[3]` vector across the rows of a `[2, 3]` matrix.
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

To add the vector to every row, first make the replication explicit:
`expand direction "column -> row column" with row := 2` produces a `[2, 3]` tensor that `+`
accepts. The pattern specifies which axis is retained and which is introduced. The shape-pattern
section below develops this notation for replication, rearrangement, and contraction.

## Dot-Product Theorems

The same dot-product definition also appears in theorem statements. Over the reals, symmetry
follows from the backend-generic algebra layer:

```lean
-- Move from the reference dot definition to the algebraic
-- dot theorem over real tensors.
example (a b : Tensor ℝ [3]) :
    a.dotSpec b = b.dotSpec a := by
  show Spec.dot a b = Spec.dot b a
  rw [Spec.dot_eq_tensorAlgebra_dot,
    Spec.dot_eq_tensorAlgebra_dot,
    Proofs.TensorAlgebra.dot_comm]
```

The proof connects the executable definition to an algebraic representation.
`show` replaces the goal with a definitionally equal one: `Spec.dot` over `ℝ` and
`Tensor.dotSpec` over `ℝ` are both `sumSpec (mulSpec a b)`, so Lean
accepts the rewording without a lemma. `Spec.dot_eq_tensorAlgebra_dot` then moves from the
sum-of-products form to `Proofs.TensorAlgebra.dot`, which recurses on the shape and is therefore
suited to induction on its dimensions. `Proofs.TensorAlgebra.dot_comm` closes the goal; it is stated
over any `CommSemiring` from Mathlib {Informal.citep mathlib2020}[], so the same lemma serves `ℝ`,
`ℚ`, and other lawful commutative semirings. Rounded floating-point arithmetic is not associative
and does not satisfy those hypotheses.

These two representations support different arguments. The fold form fixes the execution order
used in floating-point analysis; the recursive form exposes the structure used in inductive
proofs. The bridge lemma proves their agreement once so later proofs can use either form.

The neighbouring lemmas in the same file are the ones an application proof usually needs:
`dot_scale_left` and `dot_scale_right` for scalar multiples, `dot_add_left` and `dot_add_right` for
linearity, `dot_full_zero_right` for the zero tensor, `dot_vec_eq_sum` to reach an ordinary finite
sum, and `dot_mat_linear_adjoint` for the adjoint identity that the backward pass of a linear layer
relies on.

# Total Indexing

Specification-level indices use `Fin`, so every index includes a proof that it lies inside its
axis. Indexing therefore does not return `Option` or throw an out-of-range exception. The result
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

The `by decide` in each index is the bounds proof, discharged by computation because the bound is a
literal. In shape-polymorphic code the same position holds a hypothesis from the surrounding
theorem instead.

The returned `3` comes from row one, column zero: the second row is `[3, 4]`, and its first entry
is selected next. After the first indexing operation the type is a length-two row tensor;
after the second, it is the scalar element type `Nat`. The bounds are attached to the indices
before either selection occurs. This lets a function accept an arbitrary `Fin 2` without
rechecking it, even when the function cannot know which of the two valid positions it will receive.

A complete typed coordinate can be passed to `tensor.at` for a direct scalar read. Chained indexing
first constructs the selected row tensor, so it can incur the cost of an intermediate slice.
The immutable `tensor.set coordinate value` and `tensor.modify coordinate f` operations preserve
the static tensor type. Packed `Float` and `UInt8` tensors use native copy-on-write updates, which
can reuse a uniquely owned buffer; other element types use the general array-backed implementation.
Their semantic laws are available to the simplifier, including
`(tensor.set coordinate value).at coordinate = value`.

Unchecked coordinate arrays cross an explicit validation boundary:
`tensor.at? #[...]`, `tensor.set? #[...] value`, and `tensor.modify? #[...] f` return `none`
when the rank or a bound is wrong.

This is useful in proofs: a theorem about a tensor with one or more dimensions introduces an
arbitrary valid index and applies its induction hypothesis to the smaller tensor selected there.

# Trusted Memory And External Data

An in-memory array already has an intrinsic length, so converting it is total:

```lean
-- The returned tensor records the actual array length in
-- its dependent result type.
def asVector (xs : Array Float) : Tensor Float [xs.size] :=
  Tensor.from xs
```

The shape in the result type mentions `xs.size`, which is why no failure branch is needed: the
length is read off the value that was handed over rather than promised separately.

This result type is useful even when `xs.size` is not known until execution. “In the type” does
not mean “written as a numeral in the source”: Lean can express a result whose dimension depends
on an argument's value. A caller that accepts any vector length can use `asVector` directly.
A caller that specifically requires four entries needs the extra comparison in `loadTensor4`.
The two constructors answer different questions about the same array.

A file or network payload additionally claims a shape. That claim must be checked at the external
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

An input with four entries reaches the successful branch; a shorter input returns the size
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
`Tensor.from xs` has shape `[xs.size]`, so
reshaping it to `[4]` requires `Shape.size [xs.size] = Shape.size [4]`, and `simp [Shape.size, h]`
supplies it from the branch hypothesis `h : xs.size = 4`. Lean solves this obligation by itself
when both shapes are literals; here one of them is not, so the proof is written out. That single
runtime comparison establishes the length, and the proof lets downstream code use that fact
through the tensor's type.

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

The check proves something about the accepted payload. It does not prove that every future file is
valid, and it does not certify the code that produced the file.

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
  * `TorchLean.Floats.FP32`
  * rounded-real binary32-precision proof model; no upper exponent bound or IEEE special values
*
  * `TorchLean.Complex (ExecFloat.Binary 8 23)`
  * executable complex scalar with binary32 real and imaginary components
:::

The trainer's `arithmetic` field selects executable arithmetic for the run. Proofs instantiate
tensors over `ℝ` or `Floats.FP32` directly; those noncomputable types do not appear as command-line
runtime choices. The generic runtime can execute complex scalar programs, but the supervised trainer
only offers its two real modes. Explicit complex training uses `autograd.complex.grad` for a real
loss, with both components retained in the state, predictions, and checkpoints. FloatLib formats
use the same
tensor construction API. Select binary32 and construct the decimals directly in that format:

```lean
-- Each decimal is rounded into binary32 directly.
def binary32 : Tensor (ExecFloat.Binary 8 23) [3] :=
  [0.1, 0.2, 0.3]
```

```lean (name := binary32Eval)
-- Decode the encodings explicitly; the scalar's internal
-- representation is not the display contract.
#eval binary32.data.map ExecFloat.Binary.toBits32
```
```leanOutput binary32Eval
#[1036831949, 1045220557, 1050253722]
```

FloatLib exposes the stored bit patterns, making the rounding in the conversion visible.
The first entry,
`1036831949`, is `0x3DCCCCCD`, and the real number it denotes is $`0.100000001490116119384765625`,
not $`0.1`. The floating-point chapter uses this gap to introduce rounding error.

## Choosing Precision For A Tensor And Model

The element type can also select a wider software format. For example, binary128 uses 15 exponent
bits and 112 stored fraction bits. A normal value has 113 significant bits, including the implicit
leading bit. Shapes and precision answer different questions: `[1]` says there is one coordinate;
`Binary128` says how arithmetic on that coordinate rounds.

```lean
abbrev TensorBinary128 :=
  FloatLib.Floats.ExecFloat.Binary
    (exponentBits := 15) (fractionBits := 112)

def wideAffine : nn.Sequential [1] [1] :=
  nn.build 0 (nn.linear 1 1)
```

Give the weight and bias the same value $`a=1+2^{-100}`. This makes the model
$`x\mapsto ax+a`. At $`x=2` its output is $`3a` and its derivative with respect to the input is
$`a`. All these values fit in binary128. Binary64 would round the chosen parameter to 1 before
the model started, so this example also checks the input boundary.

The typed graph API accepts a state whose scalar type we choose directly. The model determines
its parameter shapes. Constructing `nn.State.full a` fills both the weight and bias with `a`;
it does not call an initializer that first generates native `Float` values.

```lean (name := wideChecks)
def wideForwardAndDerivative : IO (Bool × Bool × Bool) := do
  let exact : Rat := 1 + 1 / (2 ^ 100 : Nat)
  let a : TensorBinary128 := Rat.cast exact
  let input : Tensor TensorBinary128 [1] :=
    Tensor.full [1] 2
  let state :
      nn.State TensorBinary128
        (nn.stateShapes wideAffine) :=
    nn.State.full a
  let graph ←
    nn.lowerToTypedGraph wideAffine (α := TensorBinary128)
  let output := nn.TypedGraphModel.forward graph state input
  -- Zero parameter tangent, unit input tangent:
  -- differentiate with respect to x.
  let inputJvp :=
    nn.TypedGraphModel.jvp graph state nn.State.zeros
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
calculation above. They decode finite results as exact rationals, avoiding a final conversion to
`Float` that could hide the extra precision. They test this model and input; a derivative theorem
for a family of models is a separate statement.

The same state and VJP can support a training step. Keep the sample $`x=2`, choose target zero,
and use half squared error, $`L(w,b)=(2w+b)^2/2`. At $`w=b=a`, the prediction is $`3a`, so the
parameter gradients are $`6a` and $`3a`. With learning rate $`1/8`, one SGD update gives
$`w'=a/4`, $`b'=5a/8`, and prediction $`9a/8`. These dyadic values still fit in binary128.

Pass the prediction residual to `nn.TypedGraphModel.vjp` to obtain the state gradient, then call
`nn.sgdStep wideAffine learningRate state gradient`. The result has type
`Except String (nn.State TensorBinary128 (nn.stateShapes wideAffine))`: handle a model-validation
error or continue with the new immutable state. The learning rate is a `TensorBinary128`, so this
update need not pass through a native `Float`. The maintained typed training
[code](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Quickstart/TypedTraining.lean)
implements this loop for half squared error. The update preserves frozen state entries and
rejects models with buffer-update hooks, including BatchNorm, whose running statistics need a
separate update.

This workflow runs through typed CPU operations. The `NN.API.Precision` import, included in
`NN.API`, supplies the configured scalar instances. A wider element type does not supply a CUDA
kernel or widen the supervised trainer's `Float` datasets, initializer, checkpoints, and reports.
For a computation that needs the extra digits throughout, construct its inputs and state in the
chosen type and keep that type through the result boundary. The
{ref "floats"}[floating-point chapter] explains exact input conversion, format limits, and which
arithmetic operations have refinement or error theorems.

For small formats, check constants as well as inputs. Layer normalization's default tolerance
rounds the exact rational $`10^{-5}` into the selected scalar. With three exponent bits and two
fraction bits, it becomes zero; a constant row can then encounter a zero denominator.
`nn.layerNorm` accepts an explicit rational `eps` before graph lowering. For this particular
format, $`1/16` is a representable positive choice. Its suitability for a model is a separate
numerical decision. Shape checking and format validity do not prove that a tolerance survives
conversion or that the resulting computation stays finite.

# Mixed Element Types

`Tensor α s` is homogeneous: it cannot place an `Int`, FP16 value, and FP32 value in different
entries. Different tensors may still have different element types. Pointwise `+`, `-`, `*`, `/`,
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

The result is homogeneous `Float`; promotion and arithmetic happen together while the output
buffer is built. TorchLean's built-in promotion order is
`UInt8 < Nat < Int < Rat < Float32 < Float`. A custom scalar participates by registering
`ElementCast` and `ElementPromotion` instances. Use `tensor.cast TargetScalar` when conversion is
the operation you intend rather than a consequence of binary arithmetic.

This promotion chain covers the registered real numeric scalar families. Proof-oriented or
domain-specific scalars such as `Real`, configured FloatLib binary formats, and `Complex α` remain
valid homogeneous tensor
elements, but mixed arithmetic for a new pair requires an explicit promotion instance so TorchLean
never invents numerical semantics.

Current model programs and `NN.IR.Semantics` select one numeric `α` for learned parameters,
activations, and outputs. Scalar polymorphism means the same definition can be interpreted again
at another `α`. Integer indices and boolean masks use dedicated operation interfaces so they are
not silently treated as differentiable numeric tensors.

# Checked Shape Patterns

Importing `NN.Tensor` or `NN` exposes the complete shape-pattern API on the same ordinary
`Tensor` type. The pattern keywords are scoped syntax activated by `open TorchLean.Tensor`; the
`expand` form is the einops `repeat` operation under a name that does not collide with Lean's
`repeat`:

```lean
-- Axis names describe rearrangement and contraction; their
-- lengths come from the tensor types.
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

The literal pattern and every static input or output shape are checked while Lean elaborates the
definition {Informal.citep lean4}[]. The output names select and order dimensions from the inputs,
so the inferred result type records the output shape and a conflicting annotation is rejected before
execution. The shape-pattern operations above do not need result annotations: Lean infers their
tensor shapes, as these checks show:

```lean (name := transposedType)
-- Swapping the named axes changes [2, 3] into [3, 2].
#check transposed
```
```leanOutput transposedType
transposed : TorchLean.Tensor Float [3, 2]
```

```lean (name := productType)
-- The contracted axis disappears; the row and output-column
-- axes remain.
#check product
```
```leanOutput productType
product : TorchLean.Tensor Float [2, 4]
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
  * the shared `contracted` dimension is checked as three and removed; `row column` contributes
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

The macros read `matrix : Tensor Float [2, 3]` and
`right : Tensor Float [3, 4]` while elaborating the file, bind pattern names to those dimensions,
and synthesize the result type. The implementation receives a fully checked operation after
elaboration. The regression suite deliberately defines all eight results without result
annotations and then checks them against the types in this table.
Multi-input `einsum` and `pack` automatically promote registered element types just as ordinary
arithmetic does. `pack` retains the exact star shapes needed by `unpack`, so the round trip
does not repeat dimensions or expose an anonymous tensor-metadata pair. `parse_shape` returns
ordinary named dimension data, here
`[("row", 2), ("column", 3)]`, rather than a new tensor.

The transpose output is a useful check on the pattern's meaning: `[1, 4]` is the original first
column, now printed as a row. Merely reshaping the same six flat values to `[3, 2]` would group
them as `[1, 2]`, `[3, 4]`, `[5, 6]`, which is a different operation. The type `[3, 2]` alone
cannot distinguish those results. The pattern supplies the coordinate correspondence, while the
type checks that its dimensions fit.

Packing preserves a different piece of information. The flat payload has eight values, but eight
alone does not say where the length-two component ends or how to reconstruct the matrix. The
recorded star shapes retain that partition. Choosing component zero or one in `restored` then
recovers its respective type, so downstream code can consume a vector and a matrix without
guessing their original boundaries from a single flattened buffer.

# Tensors And Arrays

TorchLean uses `Tensor` for numerical data. A length-$`n` value has type `Tensor α [n]`,
including when `n` is computed at runtime. The shape already records its length, so it does not
need a separate `Vector` wrapper. Arrays remain useful for JSON decoding and collections of
heterogeneous records; numerical algorithms consume tensors after decoding.

Lists in types such as `Tensor α [batch, width]` are shape syntax. They also index dependent
parameter packs whose tensors have different shapes.

`Tensor.window` takes a one-dimensional tensor and constructs a fixed-length slice, filling
positions beyond its end with the supplied padding value:

```lean
-- Both windows start at offset one; the shorter source
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
-- Only 20 remains after the offset; the other three
-- positions receive zero.
#eval padded
```
```leanOutput paddedEval
[20, 0, 0, 0]
```

The result length is the `length` argument in the type. `offset` selects the first input
coordinate, and `pad` fills positions beyond the input. The second call explains why the
operation is total: a window that runs off the end of a short tensor still has the promised
length, and the pad
value is visible in the source rather than chosen by the implementation. The constructor works for
every registered storage, including packed floats and bytes, exact rationals, executable IEEE
references, and boxed complex values.

At offset one, the long input supplies positions one through four. The short input supplies only
position one; the other three requested positions become padding. Thus `[20, 0, 0, 0]` describes
an explicit data transformation, rather than evidence that the source contained four values.
For a sequence model, the program must still decide whether padding participates in attention or
the loss. A fixed output shape makes the window easy to consume, while the padding policy
determines what its values mean.

# Runtime Tensor Storage

The tensor is also the native CPU representation. Users still write and receive
`Tensor α shape`. A CPU eager node internally stores `SomeTensor α`, which pairs one runtime shape
with a `Tensor α shape`. This shape erasure lets one autograd tape contain activations and gradients
of different shapes in a single array. Recovering a typed tensor requires checking the stored
shape before exposing the value.

`TensorPack α shapes` solves a different problem. It is a statically heterogeneous tuple: the full
list of member shapes remains in its type. Model parameters, gradients, and typed graph contexts use
it when a fixed collection contains tensors of different shapes. It does not erase shapes, and it
is not an alternative input or output tensor type.

For example, the two weights and two biases of a small MLP have four different positions in a
known state layout. A `TensorPack` can describe that whole layout statically. An eager tape,
however, grows as operations execute and may contain a matrix, a vector, and a scalar loss.
`SomeTensor` lets those entries share an array while retaining each entry's runtime shape.
Both wrappers contain ordinary tensors; the difference is how much of the collection's structure
the surrounding program knows before execution.

A model program does not pass those values around directly while it records a computation. It uses
`Runtime.ValueRef m α shape`, a shape-indexed handle owned by one session. Reading the handle
returns the corresponding tensor value. It has no meaning in another session and is not a second
tensor datatype. Session-backed runtime handles carry an owner token and recording generation;
operations reject handles from another session and handles retained across `resetTape`.

CUDA execution is the one place where the physical representation must differ. An `AnyBuffer`
contains a runtime shape and a native device buffer in contiguous row-major order. Upload and
download functions connect it to `Tensor`; CUDA tape operations validate the stored shape and buffer
size before dispatch. Proofs about graph evaluation and explicit kernel contracts cover the
operations for which TorchLean has established a bridge. The native CUDA compiler, driver, and
hardware remain named external boundaries rather than being treated as consequences of the tensor
definition.

# Linear Layers And Prefix Dimensions

PyTorch's `Linear(in_features, out_features)` acts on the last axis {Informal.citep pytorch2019}[].
TorchLean follows that useful convention while checking the complete map. The same layer
constructor, with an explicit prefix argument when needed, supports:

```
-- The last axis changes width while every leading batch or
-- sequence axis is retained.
[2]              -> [8]
[batch, 2]       -> [batch, 8]
[batch, time, 2] -> [batch, time, 8]
```

Use `nn.linear 2 8`, `nn.linear 2 8 [batch]`, or `nn.linear 2 8 [batch, time]` respectively.
The prefix defaults to `[]`; it is not inferred backwards from a surrounding model annotation.
Any leading shape can serve as the prefix; it need not mean “batch” or “time.” One linear
definition therefore works for single examples, minibatches, sequences, and higher-rank
collections.

The value-level operation behaves the same way. A linear map acts on the final input axis and
preserves every leading axis:

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

There are eight length-three vectors in `activations`: two choices on the first axis and four
on the second. Each meets the same five rows of `projection`, producing five output features,
and each receives the same length-five bias. Nothing in this operation mixes the eight leading
positions. ReLU then changes individual output values while preserving all three axes. This
example uses zeros to keep the values incidental; the `#check` result is about the shape of that
whole composition. An annotation can therefore use the shorter spelling without an explicit proof:

```lean
-- The inferred replaceLast shape reduces to the literal [2,
-- 4, 5] without a manual cast.
def projectedAnnotated : Tensor Float [2, 4, 5] :=
  projected
```

The input's final extent fixes the input width as three; the weight fixes the output width as five;
and `Tensor.linear` replaces only that final extent. A mismatched weight is rejected during
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

The same check applies one level up. Change the second linear layer in the quickstart MLP from
`nn.linear 8 1` to `nn.linear 7 1` without changing the preceding layer, and the sequential model
no longer composes: one layer produces a last dimension of eight while the next requires seven.
The error appears when the model is defined, before initialization, data loading, or training.

# Reshaping

A reshape from `[2,3]` to `[6]` is permitted because both shapes contain six values. It still needs
an explicit operation because the indexing interpretation changes. Conversely, reshaping `[2,3]`
to `[2,4]` is impossible because the element counts differ.

The layout convention matters at the representation boundary. In row-major order:

$$`\operatorname{flatIndex}(i,j)=3i+j`

for a $`2\times 3` matrix. A column-major native library would use a different equation. Shape
equality does not prove layout agreement, so backend capsules record layout requirements separately.

# Proof Observation And Native Storage

Proofs and CPU execution use the same tensor value. A proof observes the tensor through total
coordinate lookup; generated code traverses its contiguous storage. There is no materialization
step between a proof tensor and a host tensor.

For trusted memory, construction and reshape are separate:

```lean
-- The flat array supplies six entries; reshape groups
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

The round trip returns the original six values in the original order. For `Float`, `Tensor.from
raw` packs the boxed array into a `FloatArray`, and reshape reuses that buffer without copying.
Converting back with `Tensor.to grid (Array Float)` materializes a boxed array; requesting
`FloatArray` instead exposes the packed buffer directly. CUDA remains a separate device
representation connected by checked upload and download operations.

`grid` provides matrix coordinates; converting it to an array provides a flat sequence. An array
consumer needs the dimensions separately if it is to reconstruct those coordinates.

# Cost And Representation Notes

The shape indices remove ambiguity, but they do not erase the cost of an operation. The useful
cost model is:

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
  * plain tensor operations use their CPU implementation; runtime graph operations may dispatch
    to a selected CUDA kernel, cuBLAS call, or other provider
:::

Big-O notation alone cannot settle provider agreement. Two matrix multiplications may both take
$`O(mnk)` arithmetic operations while accumulating in different orders and returning different
Float32 bits. The graph and backend chapters keep the operation, provider, and numerical contract
separate for this reason.

# Comparing Tensor Performance

The checked representation does not make every scalar family equally fast. Compare the same
operation, scalar type, input shape, and ownership conditions. Allocate inputs before timing,
warm up each implementation, and check numerical results before comparing elapsed times.
Record thread settings as well: a sequential baseline and a task-parallel contraction measure
different execution policies.

Measure end-to-end training separately. Graph construction, parameter initialization, autograd,
and optimizer state can dominate even when individual tensor kernels are fast.

# Common Tensor Declarations

The examples above use these declarations for construction, coordinate access, and arithmetic:

:::table +header
*
  * Declaration
  * Use
*
  * `[d₀, ..., dₙ]`
  * build a shape known while Lean elaborates the file
*
  * `[[...], [...]]`
  * construct a rectangular tensor literal checked against its expected shape
*
  * `0.5 : Tensor Float []`
  * construct a rank-zero tensor directly from an ordinary numeric literal
*
  * `Tensor.full shape value`, `Tensor.zeros shape`, `Tensor.ones shape`
  * build a constant tensor; the shape is usually inferred from the expected type
*
  * `Tensor.from source`
  * convert an in-memory array, list, vector, or packed buffer using its intrinsic shape
*
  * `tensor.reshape shape`
  * reinterpret an equal-size row-major buffer without moving scalar data; symbolic sizes may need a
    proof
*
  * `tensor.at coordinate`
  * read one scalar using a statically valid complete coordinate
*
  * `tensor.set coordinate value`
  * immutably replace one scalar while preserving shape and storage selection
*
  * `Tensor.sum tensor`, `Tensor.mean tensor`
  * reduce every entry to one scalar in row-major fold order
*
  * `left.dotSpec right`
  * sum the elementwise products of two equally shaped tensors
*
  * `Tensor.matmul left right`, `Tensor.matvec matrix vector`, `Tensor.vecmat vector matrix`
  * contract the shared extent of a matrix product
*
  * `tensor.linear weight bias`
  * apply a linear map to the final axis while preserving all leading axes
*
  * `tensor.relu`, `tensor.sigmoid`, `tensor.tanh`
  * apply a public pointwise activation without exposing its proof specification
*
  * `Tensor.qr tensor`
  * compute named reduced QR factors; an `m × n` input gives
    `result.q : Tensor α [m, min m n]` and `result.r : Tensor α [min m n, n]`
*
  * `Tensor.cholesky tensor`
  * compute the lower-triangular Cholesky factor candidate
*
  * `tensor.cast TargetElement`
  * convert element types using a registered `ElementCast`
*
  * `Tensor.castShape`
  * transport a tensor along a proved equality of shapes
*
  * `tensor.to TargetType`
  * convert to a requested in-memory representation such as `Array α`
:::

The generated API reference gives the complete signatures. This table is the smaller working set
used by the examples in this guide.

# Tensor Inspection In The Lean Infoview

Open
{src "NN/Examples/DeepDives/TensorOperations.lean"}[`NN/Examples/DeepDives/TensorOperations.lean`]
in VS Code with the Lean extension. Place the cursor on:

```
-- The value view shows coordinates; the statistics view
-- summarizes the same matrix.
#tensor_view matrix
#tensor_stats_view matrix
```

The first widget renders a tensor; the second summarizes its values. Move the cursor to
`#tensor_view firstRow` to see the shape change after indexing.

# Shape Guarantees

If a model accepts `Tensor α input`, Lean checks that every statically represented layer
composes and that the final result has the declared output shape. It can also check that a parsed
runtime payload has the length promised by its dimensions. That is not a small property: automated
testing of production tensor frameworks keeps finding shape and type inconsistencies that survive
review {Informal.citep nnsmith2023}[].

Shape safety does not, by itself, prove:

- that external memory uses the expected row-major layout;
- that two axes have the intended domain meaning;
- that a CUDA kernel wrote within bounds;
- that an arithmetic operation is numerically correct;
- that training converges.

# Model Construction

The next chapter turns shape maps into layers and model architectures. The most useful sources to
keep nearby are:

- [NN/Tensor.lean](https://github.com/lean-dojo/TorchLean/blob/main/NN/Tensor.lean) for the public
  tensor surface and its focused conversion, construction, printing, and operation modules;
- {src "NN/Tensor/Internal/Representation/Basic/Core.lean"}[
  NN/Tensor/Internal/Representation/Basic/Core.lean] for the certified contiguous representation;
- {src "NN/Proofs/Tensor/Basic.lean"}[NN/Proofs/Tensor/Basic.lean]
  for flattening, unflattening, and algebraic laws;
- {src "NN/Proofs/Tensor/Algebra.lean"}[NN/Proofs/Tensor/Algebra.lean]
  for the dot-product and matrix-product algebra used above.
