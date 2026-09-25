import VersoManual
import NN.API
import TorchLeanBlueprint.Bib
import TorchLeanBlueprint.Roles

open Verso.Genre Manual
open Verso.Genre.Manual.InlineLean
open TorchLean
open Lean.Elab.Tactic.GuardMsgs.WhitespaceMode (lax)

#doc (Manual) "Lean Basics" =>
%%%
tag := "lean_ecosystem"
file := "Enough-Lean-To-Begin"
%%%

TorchLean is ordinary Lean code. Models are definitions, shapes are values that appear in types,
training entry points are `IO` programs, and proofs are terms checked by Lean's kernel
{Informal.citep lean4}[]. The same language describes the model, its execution, and statements
about its behavior.

Much of a TorchLean program can be read from its types. A layer's type tells us which dimensions
it accepts; a trainer's prediction method tells us that execution takes place in `IO`. We can ask
Lean to print those types before running the model.

Named Lean blocks are elaborated when this guide builds, and their output blocks are checked
against Lean's messages, with the declared whitespace policy. Shell and Python examples are
separate transcripts.

# Running Lean Files

Create a file named `Primer.lean` at the TorchLean repository root and run it:

```terminal
# Check the saved primer file in the repository environment.
cd TorchLean
lake env lean Primer.lean
```

`lake env` runs the command in the project's environment, including the search paths for its Lean
libraries. The checkout pins its toolchain in `lean-toolchain`. Use this command from the repository
root so that `NN.API` resolves against the project and its dependencies.

When every command in the file succeeds, Lean prints nothing except the output you asked for with
`#check`, `#eval`, and friends, and exits with status zero. With the imports and namespace opening
shown in the next section, a file ending in

```
-- Name a length-two Float tensor, then evaluate its
-- entries.
def point : Tensor Float [2] := [0.25, -0.75]
#eval point
```

prints exactly one line:

```
[0.250000, -0.750000]
```

Repository Lake targets enable warnings-as-errors. A direct `lake env lean Primer.lean` invocation
sets the search environment but does not automatically apply every target-specific compiler option;
inspect any warnings as well as errors.

# Imports And Namespaces

Begin the file with:

```
-- Import the application definitions and make their short
-- names available.
import NN.API

open TorchLean
```

`import NN.API` loads the application interface: tensors, model builders, datasets, optimizers,
and the high-level trainer. Proofs and backend internals have additional imports, introduced where
the examples need them.

Most application-facing names live in the `TorchLean` namespace. `open TorchLean` lets us write
`Tensor`, `nn.linear`, and `Trainer.new` instead of prefixing each one with `TorchLean.`.

Lower-case `nn` and `optim` are namespaces. In Python, `import torch.nn as nn`
binds a module object to a variable, and `nn.Linear` is an attribute lookup performed while the
program runs {Informal.citep pytorch2019}[]. Here, `nn.linear` is resolved by the elaborator
before execution, and there is no
`nn` value you could pass to a function or reassign.

Lean can show you the type of any name it knows:

```lean (name := langTensorType)
-- Inspect the tensor type constructor, including its
-- required storage instance.
#check Tensor
```

```leanOutput langTensorType (whitespace := lax)
TorchLean.Tensor (α : Type) (shape : Shape) [Storage α] : Type
```

So a tensor type takes two arguments, an element type and a shape, plus one instance argument
describing how the elements are stored. Now ask about a layer:

```lean (name := langLinearType)
-- Ask which input and output shapes follow from the two
-- layer widths.
#check nn.linear 2 4
```

```leanOutput langLinearType (whitespace := lax)
nn.linear 2 4 : nn.Builder
  (nn.Sequential (Shape.appendDim [] 2) (Shape.appendDim [] 4))
```

The widths `2` and `4` appear in the type. A `2 → 4` linear layer and a `4 → 2` linear layer
therefore have different types, allowing subsequent operations to require the appropriate widths.

Lean prints the input shape as `Shape.appendDim [] 2`: the layer's definition constructs it by
appending an innermost axis to the empty shape. `appendDim` is marked `@[reducible]`, so Lean can
unfold it during type checking and accept `[2]` in its place:

```lean (name := langSameShape)
-- The computed appendDim shapes reduce to the literal
-- shapes in this annotation.
example : nn.Builder (nn.Sequential [2] [4]) :=
  nn.linear 2 4
```

Printed types can retain expressions such as `appendDim` rather than reducing every dimension.
When two shapes look different, check whether their expressions compute the same list.

One more, this time a definition with implicit and instance arguments and a default value:

```lean (name := langTrainerType)
-- Expose the inferred model type, shape indices, conversion
-- instance, and default config.
#check Trainer.new
```

```leanOutput langTrainerType (whitespace := lax)
TorchLean.Trainer.new.{u} {Model : Type u} {σ τ : Shape}
  [Trainer.ToModel Model σ τ] (model : Model)
  (config : Trainer.Config σ τ := { }) : Trainer σ τ
```

Start with the explicit argument `(model : Model)` and the result `Trainer σ τ`. Lean infers the
curly-braced arguments, including the shapes, and searches for the square-bracketed `ToModel`
instance that connects this model representation to the trainer. The printed `.{u}` is a universe
parameter describing how general `Model`'s type may be.

The last argument has a default: `:= { }` fills the configuration with its default field values.
Thus `Trainer.new model` is a complete call, with an objective and runtime settings even though we
have not written them out.

# Definitions

`def` gives a name to a value or a function:

```lean (name := langSquare)
-- A Nat argument and Nat result make this ordinary integer
-- computation explicit.
def square (n : Nat) : Nat :=
  n * n

#eval square 12
```

```leanOutput langSquare
144
```

The first `Nat` is the argument type and the second is the result type. Lean can infer many result
types. An explicit annotation at an API boundary also checks the implementation against the intended
interface, locating a mismatch at the definition.

Local values are introduced with `let`:

```lean (name := langCentered)
-- Compute the displacement once and use that local value
-- twice.
def centeredSquare (x center : Int) : Int :=
  let displacement := x - center
  displacement * displacement

#eval centeredSquare 7 3
```

```leanOutput langCentered
16
```

The `let` binds `displacement` to `x - center` for the rest of this expression. Later uses refer
to that value, so we can substitute its definition when reasoning about the result.

# Tensors

The type of a TorchLean tensor has two arguments:

```
-- Read alpha as the element type and s as the list of axis
-- lengths.
Tensor α s
```

`α` is the element type and `s` is the shape; a storage instance supplies their representation.
Because the type depends on the *value* `s`, it is a dependent type. This lets an operation express
relationships between dimensions in its argument and result types.

```lean (name := langPoint)
-- Check the type first, then evaluate the value: these
-- commands answer different questions.
def point : Tensor Float [2] := [0.25, -0.75]

#check point
#eval point
```

```leanOutput langPoint
point : Tensor Float [2]
```

```leanOutput langPoint
[0.250000, -0.750000]
```

`#check` elaborates an expression and reports its type; `#eval` executes it and displays the result.
The first line tells us that `point` has two Float entries, and the second tells us which entries
they are. The distinction also matters for effectful expressions: checking a prediction method's
type leaves the model unexecuted.

The bracket literal is not a `List Float` that happens to be used as a tensor. A custom elaborator
reads it against the expected type, checks each axis, and builds the tensor's buffer directly, which
is also why its error messages talk about dimensions rather than about lists.

The shape index is itself a Lean value:

```lean (name := langShapeType)
-- A shape is data too; here it is a list of natural-number
-- axis lengths.
#check ([2, 3] : Shape)
```

```leanOutput langShapeType
[2, 3] : List ℕ
```

A shape is a list of natural numbers, outermost axis first. Rank and element count are computed
from that list:

```lean (name := langShapeOps)
-- Count axes, multiply their lengths, and compare with the
-- one-element scalar shape.
#eval Shape.rank [2, 3]
#eval Shape.size [2, 3]
#eval Shape.size []
```

```leanOutput langShapeOps
2
```

```leanOutput langShapeOps
6
```

```leanOutput langShapeOps
1
```

The rank of `[2, 3]` is two because there are two axes; its size is six because there are two rows
with three entries each. The empty shape has rank zero but size one, following the empty-product
convention. This is how a scalar fits into the same tensor interface as a matrix. An axis of
length zero would describe a different object with no entries. When a signature uses `[]` for a
loss, it means one scalar value, not an empty batch or a missing result.

Both TorchLean and PyTorch track shape information. TorchLean's tensor type makes it available
when checking the surrounding program:

:::table +header
*
  * question
  * TorchLean
  * PyTorch
*
  * where the shape lives
  * in the type, `Tensor Float [2, 3]`
  * in the object, `t.shape`
*
  * when a shape is known
  * statically indexed; dynamic dimensions need boundary checks
  * runtime metadata, also usable during graph capture
*
  * what a wrong shape produces
  * an elaboration error
  * a runtime exception, if that line runs
*
  * shape of a scalar
  * `[]`
  * `torch.Size([])`
:::

# The Basic Operations

With shapes in their types, tensor operations can require equal dimensions or describe how an
output shape is computed. Two vectors illustrate the equal-shape case:

```lean (name := langElementwise)
-- Keep the two coordinates paired when adding and
-- multiplying these vectors.
def other : Tensor Float [2] := [4.0, 8.0]

#eval point + other
#eval Tensor.mul point other
```

```leanOutput langElementwise
[4.250000, 7.250000]
```

```leanOutput langElementwise
[1.000000, -6.000000]
```

`+` works because tensors of a fixed shape form an additive structure, and `Tensor.mul` is the
elementwise product, PyTorch's `*`. Both require *equal* shapes; neither one broadcasts silently.

A dot product is a product followed by a sum, and you can spell it either way:

```lean (name := langDot)
-- Compare an explicit multiply-and-sum expression with the
-- named dot-product operation.
#eval Tensor.sum (Tensor.mul point other)
#eval Tensor.dotSpec point other
```

```leanOutput langDot
-5.000000
```

```leanOutput langDot
-5.000000
```

The two expressions have the same specification: `dotSpec` is defined as
`sumSpec (mulSpec a b)` in
{src "NN/Spec/Core/TensorReductionShape/Reductions.lean"}[Spec/Core/.../Reductions.lean].
The expanded form shows which products are summed; the named operation gives proofs a reusable
definition to refer to.

Reductions collapse a tensor to a scalar:

```lean (name := langReductions)
-- Compare sum, mean, and mean squared distance from the
-- zero vector.
def origin : Tensor Float [2] := [0.0, 0.0]

#eval Tensor.sum point
#eval Tensor.mean point
#eval Tensor.meanSquaredError point origin
```

```leanOutput langReductions
-0.500000
```

```leanOutput langReductions
-0.250000
```

```leanOutput langReductions
0.312500
```

The sum is `0.25 - 0.75 = -0.5`, and dividing by two gives the mean `-0.25`. For mean squared
error, we square the deviations first: `0.0625` and `0.5625` average to `0.3125`. Squaring before
reducing prevents the positive and negative deviations from cancelling.

`Tensor.matmul` uses the same type-level dimension for the left matrix's columns and the right
matrix's rows. The arguments must agree on that dimension:

```lean (name := langMatmul)
-- The shared extent three is summed out, leaving a
-- two-by-two result.
def left : Tensor Float [2, 3] :=
  [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]

def rightMatrix : Tensor Float [3, 2] :=
  [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]

#eval Tensor.matmul left rightMatrix
```

```leanOutput langMatmul
[[4.000000, 5.000000], [10.000000, 11.000000]]
```

Matrix times vector and vector times matrix are separate operations rather than one overloaded
product with a promotion rule:

```lean (name := langMatvec)
-- Ones expose row sums on the right and column sums on the
-- left.
def ones : Tensor Float [3] := [1.0, 1.0, 1.0]

#eval Tensor.matvec left ones
#eval Tensor.vecmat ([1.0, 1.0] : Tensor Float [2]) left
```

```leanOutput langMatvec
[6.000000, 15.000000]
```

```leanOutput langMatvec
[5.000000, 7.000000, 9.000000]
```

In PyTorch, `a @ b` covers vector times vector, matrix times vector, matrix times matrix, and
batched products, selecting the operation from the ranks. TorchLean uses separate names with
checked rank contracts, so `matvec` states the intended operation at the call site. Dependent typing
also permits overloaded operations when their promotion rules are specified.

ReLU preserves the shape, while `reshape` changes its axes without changing the element count:

```lean (name := langMisc)
-- ReLU changes entries; reshape changes their grouping
-- while retaining flat order.
#eval Tensor.relu ([-1.0, 2.0] : Tensor Float [2])
#eval Tensor.reshape left (target := [3, 2])
```

```leanOutput langMisc
[0.000000, 2.000000]
```

```leanOutput langMisc
[[1.000000, 2.000000], [3.000000, 4.000000], [5.000000, 6.000000]]
```

The reshaped matrix keeps the flat order `1, 2, 3, 4, 5, 6`, grouping it into three rows of two.
A transpose would exchange row and column coordinates and rearrange that order. Both produce
shape `[3, 2]` from `[2, 3]`, so their shapes alone cannot distinguish them. ReLU changes the
negative entries while preserving both shape and coordinate layout.

`reshape` takes its target shape as a named argument and checks that the two shapes have the same
size while elaborating, so there is no runtime reshape failure to handle and no `-1` to infer.

To add a vector to every row of a matrix, first repeat it along the leading axis. The addition
then receives two tensors with the same shape:

```lean (name := langBroadcast)
-- Introduce the repeated row explicitly before adding it to
-- the matrix.
def rowVector : Tensor Float [3] :=
  [10.0, 20.0, 30.0]

#eval Tensor.repeatLeading 2 rowVector
#eval
  Tensor.add left (Tensor.repeatLeading 2 rowVector)
```

```leanOutput langBroadcast
[[10.000000, 20.000000, 30.000000], [10.000000, 20.000000, 30.000000]]
```

```leanOutput langBroadcast
[[11.000000, 22.000000, 33.000000], [14.000000, 25.000000, 36.000000]]
```

In PyTorch the same result is `left + row`, using broadcasting. Broadcasting can also combine
`torch.ones(2, 1) * torch.ones(1, 3)` into a `2 by 3` matrix. That outer product is useful when
intended, but it would conceal a mistaken axis in an intended elementwise product. The explicit
repetition in TorchLean identifies which axis is being added.

The corresponding PyTorch expressions give the following results:

```
>>> # Compare the scalar dot product, matrix contraction,
>>> # and a broadcast result's shape.
>>> import torch
>>> a = torch.tensor([0.25, -0.75]); b = torch.tensor([4.0, 8.0])
>>> torch.dot(a, b).item()
-5.0
>>> (a * b).sum().item()
-5.0
>>> left = torch.tensor([[1., 2., 3.], [4., 5., 6.]])
>>> right = torch.tensor([[1., 0.], [0., 1.], [1., 1.]])
>>> left @ right
tensor([[ 4.,  5.],
        [10., 11.]])
>>> (torch.ones(2, 3) + torch.ones(3)).shape
torch.Size([2, 3])
```

:::table +header
*
  * operation
  * TorchLean
  * PyTorch
*
  * elementwise sum
  * `x + y`
  * `x + y`
*
  * elementwise product
  * `Tensor.mul x y`
  * `x * y`
*
  * dot product
  * `Tensor.sum (Tensor.mul x y)`
  * `torch.dot(x, y)`
*
  * matrix product
  * `Tensor.matmul a b`
  * `a @ b`
*
  * matrix times vector
  * `Tensor.matvec a v`
  * `a @ v`
*
  * sum, mean
  * `Tensor.sum t`, `Tensor.mean t`
  * `t.sum()`, `t.mean()`
*
  * rectified linear unit
  * `Tensor.relu t`
  * `torch.relu(t)`
*
  * reshape
  * `Tensor.reshape t (target := [3, 2])`
  * `t.reshape(3, 2)`
*
  * broadcast a row
  * `Tensor.repeatLeading 2 r`
  * implicit in `m + r`
:::

# Tensor Indexing And Updates

A coordinate into a shape has its own type, `Shape.Coord`, built from the shape:

```lean (name := langCoord)
-- A complete coordinate reaches one scalar, with one
-- bounded index per axis.
#check (Tensor.at point)
#eval Tensor.at point (0, ())
#eval Tensor.at left (1, (2, ()))
```

```leanOutput langCoord
point.at : Spec.Shape.Coord [2] → Float
```

```leanOutput langCoord
0.250000
```

```leanOutput langCoord
6.000000
```

A coordinate for `[2]` is a `Fin 2` paired with the unit value, and a coordinate for `[2, 3]` is a
`Fin 2` paired with a coordinate for `[3]`, which is why the second read is written `(1, (2, ()))`.
The coordinate type unfolds to this nested pair:

```lean (name := langCoordUnfold)
-- Unfold the coordinate type to see the bound on the vector
-- index.
example : Spec.Shape.Coord [2] = (Fin 2 × PUnit) :=
  rfl
```

`Fin n` pairs a natural number with evidence that it is below `n`. The `PUnit` terminates the
coordinate tuple; a matrix coordinate has a bounded row and column index before that terminator.
Here `rfl` unfolds the coordinate type. When we subsequently look up an entry, the bounds travel
with its indices and need no further check.

A numeral at a `Fin` type requires care, however:

```lean (name := langWrap)
-- A numeral in Fin 2 wraps modulo two; this is not a
-- checked dynamic lookup of index five.
#eval Tensor.at point (5, ())
```

```leanOutput langWrap
-0.750000
```

There is no entry `5` in a length-two vector. What happened is that the numeral `5` was elaborated
at type `Fin 2`, and Lean's `OfNat` instance for `Fin` reduces numerals modulo the bound, so `5`
became `1` and we read the last entry. The access is in range, but it need not be the coordinate
the programmer intended.

A `Fin 2` value carries evidence that its stored natural number is below two; it does not record
whether that number came from a wrapped numeral. For coordinates supplied as runtime naturals,
the checked accessor preserves an out-of-range result as a failure:

```lean (name := langAtQuestion)
-- Dynamic Nat coordinates either find an entry or return
-- none.
#eval Tensor.at? left #[1, 2]
#eval Tensor.at? left #[5, 0]
```

```leanOutput langAtQuestion
some 6.000000
```

```leanOutput langAtQuestion
none
```

The first array names a valid row and column and returns `some 6`. Row five in the second array
exceeds the matrix's two rows, so it returns `none`. Use this interface for coordinates read from
a file or supplied by a user when an out-of-range coordinate should fail.

PyTorch raises `IndexError: index 5 is out of bounds for dimension 0 with size 2` for the same
index. The checked Lean accessor represents the outcome with `Option Float`: `some` contains a
value and `none` indicates an invalid coordinate. A caller must account for both to obtain a
`Float`.

# Shape Errors

The following examples exercise three shape checks. Each named block is required to fail during
the guide build, and its diagnostic is checked against the output shown.

The first is a literal whose length does not match its shape:

```lean +error -keep (name := langBadLiteral)
-- Three supplied entries cannot inhabit the expected
-- length-two tensor.
def tooLong : Tensor Float [2] := [1.0, 2.0, 3.0]
```

```leanOutput langBadLiteral
tensor literal has leading dimension 3, but the expected leading dimension is 2
```

The tensor literal elaborator checks the dimensions against the expected tensor type and reports
the mismatch directly. Its implementation is in
{src "NN/Tensor/Internal/Elab/TensorLiteral.lean"}[Internal/Elab/TensorLiteral.lean].

The second is two tensors that must agree and do not. The helper takes one shape variable for both
arguments, which is how you say "the same shape" in a type:

```lean (name := langRequireSame)
-- Reuse one shape variable to constrain both arguments,
-- without inspecting their values.
def requireSameShape {s : Shape}
    (_left _right : Tensor Float s) : Unit :=
  ()
```

```lean +error -keep (name := langBadSame)
-- The first argument fixes shape [2], so the length-three
-- second argument is rejected.
def mismatch := requireSameShape point rowVector
```

```leanOutput langBadSame (whitespace := lax)
Application type mismatch: The argument
  rowVector
has type
  Tensor Float [3]
but is expected to have type
  Tensor Float [2]
in the application
  requireSameShape point rowVector
```

The same requirement extends to composed layers. This model connects matching widths:

```lean (name := langModel)
-- Compose matching internal shapes and inspect the
-- initialized model type.
def primerModel : nn.Builder (nn.Sequential [2] [1]) :=
  nn.Sequential![
    nn.linear 2 8,
    nn.relu,
    nn.linear 8 1
  ]

#check nn.build 2026 primerModel
```

```leanOutput langModel (whitespace := lax)
nn.build 2026 primerModel : nn.Sequential [2] [1]
```

The first linear layer maps a length-two vector to length eight, `nn.relu` preserves that shape, and
the second consumes length eight and returns length one. Now change one number, `8` to `7`, in the
final layer:

```lean +error -keep (name := langBadWidth)
-- The second linear layer cannot consume the eight entries
-- produced before it.
def badModel : nn.Builder (nn.Sequential [2] [1]) :=
  nn.Sequential![
    nn.linear 2 8,
    nn.relu,
    nn.linear 7 1
  ]
```

```leanOutput langBadWidth (whitespace := lax)
nn.Sequential!: layer 2 expects input shape [7], but layer 1 outputs [8].
Change layer 2's input shape or insert a layer that converts [8] to [7].
```

The diagnostic identifies incompatible shapes at a composition boundary: layer 1 outputs `[8]`,
while layer 2 expects `[7]`. Here `nn.relu` preserves its input shape, so the final layer's
width `7` also constrains the activation's shape. To change widths here, the model needs an
operation whose type connects them.

## Shape Errors In PyTorch

PyTorch checks the same mismatch when the model receives an input:

```
>>> # Construction succeeds; the first forward call exposes
>>> # the incompatible hidden widths.
>>> import torch, torch.nn as nn
>>> net = nn.Sequential(nn.Linear(2, 8), nn.ReLU(), nn.Linear(7, 1))
>>> net
Sequential(
  (0): Linear(in_features=2, out_features=8, bias=True)
  (1): ReLU()
  (2): Linear(in_features=7, out_features=1, bias=True)
)
>>> net(torch.zeros(1, 2))
RuntimeError: mat1 and mat2 shapes cannot be multiplied (1x8 and 7x1)
```

Construction allocates the parameters without executing the composition, so the error appears
at the first forward pass. The timing matters particularly when an invalid operation occurs only
in a conditional branch:

```
>>> # Exercise both branches: only the matrix product needs
>>> # the incompatible extent seven.
>>> def scale(x, flag):
...     if flag:
...         return x @ torch.zeros(7, 1)
...     return x * 2.0
...
>>> scale(torch.zeros(1, 2), False).shape
torch.Size([1, 2])
>>> scale(torch.zeros(1, 2), True)
RuntimeError: mat1 and mat2 shapes cannot be multiplied (1x2 and 7x1)
```

The first call succeeds because it uses `flag=False`. A test covering only that branch does not
exercise the invalid operation.

Lean elaborates both branches of an `if`. In a version using shape-indexed tensor operations,
each branch must satisfy its operations' shape contracts and return the required type. This
checks those contracts even for an unexecuted branch. It does not establish other properties of
the branch, such as the intended values or the correctness of a native implementation.

# Structures

A `structure` groups named fields:

```lean (name := langAffine)
-- Store an affine function's two coefficients together and
-- evaluate it at one input.
structure Affine where
  weight : Float
  bias : Float
deriving Repr

def Affine.forward (p : Affine) (x : Float) : Float :=
  p.weight * x + p.bias

#eval Affine.forward { weight := 1.5, bias := -0.5 } 2.0
```

```leanOutput langAffine
2.500000
```

`deriving Repr` asks Lean to generate the printing code, which is why `#eval` on an `Affine` value
would show its fields. Note that `Affine.forward` is called as `p.forward x` as well: any function
whose first explicit argument has type `Affine` can be written with dot notation, and TorchLean's
whole session and trainer API is built on that convention.

The same record syntax lets us configure a trainer:

```
-- Set the loss, Adam learning rate, and initialization seed
-- in a configuration record.
{ objective := .mse
  optimizer := optim.adam { learningRate := 0.03 }
  seed := 2026 }
```

Lean recovers this literal's type from the expected type at the call site. That also tells it
which type `.mse` belongs to.

# Inductive Types

An `inductive` type enumerates the possible forms of a value:

```lean (name := langReduction)
-- Give both reduction choices a printable name through an
-- exhaustive match.
inductive Reduction where
  | mean
  | sum

def reductionName : Reduction → String
  | .mean => "mean"
  | .sum => "sum"

#eval reductionName .mean
```

```leanOutput langReduction
"mean"
```

The definition of `reductionName` matches one branch to each constructor. Omitting a case leaves
the function undefined for that alternative, which Lean reports:

```lean +error -keep (name := langMissing)
-- Leaving out sum makes this definition incomplete even
-- before any call is executed.
def partialName : Reduction → String
  | .mean => "mean"
```

```leanOutput langMissing (whitespace := lax)
Missing cases:
Reduction.sum
```

TorchLean's execution modes and graph operations are inductive types too. Adding a constructor
exposes matches that listed only the old cases. A wildcard branch remains exhaustive, so its
fallback behavior needs review when the alternatives change.

Exhaustiveness checks that every case has a branch. Whether an unsupported combination returns
an error depends on the branch's implementation; the type checker does not choose that policy.

# Implicit Arguments And Instances

We already used an implicit argument above. In

```lean (name := langImplicitAgain)
-- The caller can omit s, but both tensors must still
-- instantiate the same shape.
example {s : Shape}
    (_left _right : Tensor Float s) : Unit :=
  ()
```

the shape `s` in curly braces is written once and used twice. Callers can usually omit it: Lean
infers it from the first tensor and requires the second to match. Reusing a shape variable
expresses equality of dimensions without asking callers to pass and check separate integers.

Square brackets introduce instance arguments, resolved by search rather than by inference:

```lean (name := langClasses)
-- Request a way to print alpha; the Nat call supplies its
-- instance automatically.
def showTwice {α : Type} [ToString α] (x : α) : String :=
  toString x ++ ", " ++ toString x

#eval showTwice 7
```

```leanOutput langClasses
"7, 7"
```

Lean infers `Nat` from the argument seven, then finds a `ToString Nat` implementation. The
`Storage α` argument in the chapter's first `#check` works similarly: it supplies the operations
for storing tensor entries. TorchLean also requests arithmetic, decidable shape equality, and
printing through instances.

This lets a definition serve `Float` tensors at runtime and real-valued tensors in proofs.
The operations alone do not supply algebraic laws. A theorem using associativity must obtain that
law from its scalar type or hypotheses; replacing reals with floats can preserve the program's
structure while changing which theorems hold.

# Scalar Types

A declaration such as

```lean (name := langPreserve)
-- Returning the input preserves both its scalar type and
-- its two-entry shape.
def preserveShape {α : Type} (x : Tensor α [2]) := x
```

can be instantiated with `Float`, with `Rat`, or with FloatLib binary32. Within any one
instantiation, the input and the output use the same `α`. Nothing in the type lets a run start in
one element type and finish in another, and {ref "tensors-shapes"}[Tensors And Shapes] develops what
that means for a whole training run.

# Programs, Propositions, And Proofs

Lean uses the same language for programs and claims about programs. Their types distinguish
computing a result, stating a proposition, and proving it.

A definition that returns data computes:

```lean (name := langData)
-- Compute the number of output entries as an ordinary
-- natural number.
def outputWidth : Nat :=
  Shape.size [1]

#eval outputWidth
```

```leanOutput langData
1
```

A definition that returns `Prop` states a claim, and states it without proving anything:

```lean (name := langProp)
-- Name the claim about outputWidth without yet supplying
-- evidence for it.
def HasOneOutput : Prop :=
  outputWidth = 1
```

`HasOneOutput` names the proposition. To establish it, we supply a term of that type:

```lean (name := langTheorem)
-- Reduction of outputWidth to one supplies a proof of the
-- named proposition.
theorem hasOneOutput : HasOneOutput := by
  rfl
```

`rfl` closes the goal because both sides reduce to the same natural number. When Lean accepts that
declaration, the kernel has checked a proof of exactly `outputWidth = 1`, and you can ask what the
proof relied on:

```lean (name := langAxioms)
-- Inspect the foundations used by this particular proof,
-- including its dependencies.
#print axioms hasOneOutput
```

```leanOutput langAxioms
'hasOneOutput' does not depend on any axioms
```

`#print axioms` lists the axioms on which a theorem depends, including dependencies reached
through other declarations. A proof using `sorry` depends on a placeholder axiom that appears in
this list. Inspecting it is one part of auditing the result.

Mathematical results can also depend on the standard logical foundations:

```lean (name := langAxiomsMathlib)
-- Compare a library theorem whose dependency chain uses
-- standard logical axioms.
#print axioms Tensor.mean_eq_meanSpec
```

```leanOutput langAxiomsMathlib (whitespace := lax)
'TorchLean.Tensor.mean_eq_meanSpec' depends on axioms:
[propext, Classical.choice, Quot.sound]
```

The command follows theorem dependencies down to their axioms. Here `propext` identifies logically
equivalent propositions, `Classical.choice` supplies choices from existence, and `Quot.sound`
supports quotient constructions. These are standard foundations used throughout Mathlib
{Informal.citep mathlib2020}[]; individual theorems may depend on fewer of them.

To audit this result, read its conclusion and hypotheses alongside that list, checking for
unexpected axioms or unfinished proofs. The report concerns the named theorem, so it gives no
guarantee about a compiler or foreign kernel outside that theorem's statement.

The displayed theorem connects a public operation to its specification:

```lean (name := langBridge)
-- This public mean operation unfolds to its spec on the
-- same tensor.
example : Tensor.mean point = Tensor.meanSpec point :=
  rfl
```

Unfolding the public `Tensor.mean` wrapper gives `Tensor.meanSpec`, so `rfl` closes the equality
without a tolerance or a numerical evaluation. The program and proof use the same definition.
Relating that computation to an ideal real mean, or to a native reduction with another evaluation
order, requires a further theorem. {ref "spec-layer"}[The Specification Layer] explains which
connections follow by unfolding and which need additional proofs.

The same separation appears throughout verification. An interval pass computes a candidate output
box: that is a program, and running it produces data. A predicate states that the box contains every
semantic output: that is a proposition. A soundness theorem proves the predicate under explicit
assumptions: that is a proof. Returning a candidate box therefore does not by itself establish
containment. The theorem and its hypotheses supply that connection.

# Effects And IO

A pure function returns its value directly:

```lean (name := langIncrement)
-- Pure computation returns a number and performs no
-- printing.
def increment (n : Nat) : Nat :=
  n + 1
```

Printing is an effect, so the result type says so:

```lean (name := langPrint)
-- Sequence a print action using the pure result as part of
-- its message.
def printIncrement (n : Nat) : IO Unit := do
  IO.println s!"next = {increment n}"

#eval printIncrement 4
```

```leanOutput langPrint
next = 5
```

`IO Unit` describes an action whose return value carries no useful data. The text `next = 5`
is printed as an effect; `printIncrement` does not return that string. A training script likewise
prints losses separately from returning the trained model that a later prediction call needs.

The `do` block sequences effectful actions, and `s!"..."` is string interpolation. Training
sessions, checkpoint access, subprocess calls, and native runtime entry points all live in `IO`;
pure shape calculations and mathematical specifications return values directly. External data
needed by a pure specification must be supplied as an argument, where it can also appear in the
statement of a theorem.

In an executable module, an entry point such as `main : List String → IO UInt32` receives the
command-line arguments and performs the run, which is how `lake exe torchlean` reaches the examples.

# Immutability And Storage Reuse

Tensor updates in this chapter return values while preserving any earlier values still in use.
That semantic guarantee does not require copying the entire buffer on every update.

Unique ownership can avoid a full payload copy for operations such as scalar updates. Allocation,
index calculation, and reference-counting overhead still depend on the operation. Lean's compiler
can update uniquely referenced objects in place, as described in
*Counting Immutable Beans* {Informal.citep immutablebeans2019}[]. The following loops expose two
ownership patterns by repeatedly updating entry zero. The first keeps no
reference to the previous value, so its tensor is uniquely owned; the second holds `original` alive
across every update, so it is not:

```lean (name := langReuse)
-- Compare replacing the current tensor with retaining an
-- original used by every update.
def unsharedUpdates (steps : Nat) : IO Float := do
  let mut t : Tensor Float [65536] :=
    Tensor.full [65536] 0.0
  for i in [0:steps] do
    t := t.set (0, ()) (Float.ofNat i)
  pure (Tensor.at t (0, ()))

def sharedUpdates (steps : Nat) : IO Float := do
  let original : Tensor Float [65536] :=
    Tensor.full [65536] 0.0
  let mut last := 0.0
  for i in [0:steps] do
    let next := original.set (0, ()) (Float.ofNat i)
    last := Tensor.at next (0, ())
  pure last
```

Both functions return zero for zero updates and the last assigned index as a Float otherwise.
Their return values hide the storage difference: one loop replaces its current tensor, while the
other repeatedly updates from an original it must preserve.

To measure that difference, time both with `IO.monoNanosNow`, repeat with a smaller tensor, and
report the execution mode and update count. `#eval` and a compiled executable can have different
costs. These loops provide an experiment, not a timing result; changing the tensor size helps reveal
whether a measured cost grows with the amount of data that must be copied.

A retained value also matters to differentiation: the backward computation may need a value from
the forward pass. PyTorch tracks mutation constraints and rejects, for example, this update:

```
>>> # Updating a view of this leaf tensor is rejected while
>>> # gradient tracking is enabled.
>>> w = torch.zeros(4, requires_grad=True)
>>> w[0] = 1.0
RuntimeError: a view of a leaf Variable that requires grad is being used in an
in-place operation.
```

PyTorch rejects this tracked in-place update on a view of a leaf requiring gradients, even before
a forward tape has recorded it. With TorchLean's immutable tensor interface, a reference retained
by the tape keeps the old value available. Storage reuse must respect that sharing, so an update
cannot overwrite a buffer while another reference still needs its contents.

# Reading Types

Long TorchLean types get easier when you read from the result backwards. Take one from earlier in
this chapter:

```
-- Read the result from the inside out: a [2]-to-[4]
-- sequence produced by a seeded builder.
nn.linear 2 4 :
  nn.Builder (nn.Sequential [2] [4])
```

Read it as three statements:

1. the result, eventually, is an `nn.Sequential`;
2. that sequence maps a length-two vector to a length-four vector;
3. producing it lives in `nn.Builder`, because initialization consumes deterministic seeds and the
   builder is what threads them.

Similarly, `Trainer.new model : Trainer σ τ` says that construction returns a trainer whose model
input shape is `σ` and whose output shape is `τ`, and every later method call preserves those two
indices. Those indices connect the dataset and prediction methods to the model's interface.

# Elaboration Errors

Elaboration messages can be noisy, because they expose generated names, metavariables, and qualified
internal constants, as the model composition error above shows. A reliable reading order:

- find *has type*: that is the value you actually supplied;
- find *but is expected to have type*: that is the interface it has to satisfy;
- compare the two from the outside in, and stop at the first shape, element type, or namespace where
  they differ;
- ignore everything with a `✝` or a `?m.` in it until the end, since those are generated names and
  unsolved holes rather than things you wrote.

For model composition, work left to right and compare each layer's output with the next layer's
input. For tensors, compare the element type as well as the dimension list, since `Tensor Float [2]`
and `Tensor Rat [2]` fail differently than two mismatched shapes do. For training, check that the
dataset target shape agrees with the model output shape and with the objective.

When an expression is too large to read, cut it up. Give a subexpression a name with `def`, ask Lean
for its type with `#check`, and compose the pieces one at a time. Hover information in the editor is
the same service in a faster interface.

# Further Reading

- [Functional Programming in Lean](https://lean-lang.org/functional_programming_in_lean/) is an
  introduction to Lean as a programming language, including `do` notation and `IO`.
- The [Lean language reference](https://lean-lang.org/doc/reference/latest/) gives precise syntax
  and semantics when you need the exact rule rather than the idea.
- The language paper {Informal.citet lean4}[] explains the design, including why the elaborator is
  extensible enough for `nn.Sequential!` and the tensor literal to exist.
- [Mathematics in Lean](https://leanprover-community.github.io/mathematics_in_lean/) is the
  standard introduction to writing proofs with Mathlib {Informal.citep mathlib2020}[].
- The reference-counting paper {Informal.citet immutablebeans2019}[] is the background for the
   ownership examples above.

Sources:

- {src "NN/Tensor/Operations.lean"}[NN/Tensor/Operations.lean] for the elementwise, matrix, and
  indexing operations used here;
- {src "NN/Tensor/Reductions.lean"}[NN/Tensor/Reductions.lean] for `sum`, `mean`, and the
  mean squared error;
- {src "NN/Spec/Core/Shape.lean"}[NN/Spec/Core/Shape.lean] for shapes, coordinates, and `appendDim`;
- {src "NN/Tensor/Internal/Elab/TensorLiteral.lean"}[Internal/Elab/TensorLiteral.lean] for the
  tensor literal elaborator and its error messages;
- {src "NN/API.lean"}[NN/API.lean] for what `import NN.API` brings into scope.
