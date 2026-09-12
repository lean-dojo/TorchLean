import VersoManual
import NN.API
import NN.Spec.Models.Knn
import TorchLeanBlueprint.Bib
import TorchLeanBlueprint.Roles

open Verso.Genre Manual
open Verso.Genre.Manual.InlineLean
open TorchLean
open Lean.Elab.Tactic.GuardMsgs.WhitespaceMode (lax)

#doc (Manual) "TorchLean API" =>
%%%
tag := "torchlean-api"
file := "The-TorchLean-API"
%%%

Most user programs need two lines:

```
-- Import the application interface, then open the namespace
-- used by its public tensor and model
-- names.
import NN.API
open TorchLean
```

`NN.API` is the maintained import for application code. It brings in tensors, models, data,
training, optimizers, prediction, and explicit differentiation. It does not pull every proof,
verifier, floating-point implementation, widget, and backend-internal module into a small training
file.

Keeping those dependencies separate limits what a training script needs to elaborate and makes
its purpose easier to read from the imports. A file that also proves a property or inspects a
backend can add the relevant module explicitly. Lean's module system determines which
declarations are available in a file {Informal.citep lean4}[]; the import list also documents
which parts of the library the file depends on.

The import chooses which declarations are in scope; it does not choose the arithmetic or execution
mode of a later training run. Those choices remain values in the trainer configuration. Separating
these decisions is useful when reading a short script: its imports tell us which interfaces it uses,
its model annotation tells us what tensors it accepts, and its configuration tells us how the
executable computation will run.

We will build one program using only that API, run it, compare it against the same program in
PyTorch {Informal.citep pytorch2019}[], and then explain when a narrower or lower import is
appropriate. Named Lean blocks are build-checked; shell and PyTorch transcripts document separate
runs.

# The Namespaces

`Tensor` represents values, `nn` defines models, `Data` supplies examples, and `Trainer` carries
the model through optimization and prediction. The other namespaces expose configuration,
derivatives, and mathematical interfaces:

:::table +header
*
  * Namespace
  * Responsibility
*
  * `Tensor`, `Shape`
  * shape-indexed values and constructors
*
  * `nn`
  * layers, blocks, model families, functional operations
*
  * `Data`
  * datasets, loaders, batching, text and checkpoint helpers
*
  * `Trainer`
  * configuration, training, reports, prediction, manual loops
*
  * `optim`
  * optimizer configuration
*
  * `autograd`
  * function and model derivatives
*
  * `Runtime`
  * arithmetic semantics, execution mode, device, and backend-contract selection
*
  * `Verification`
  * model-to-IR lowering and IBP/CROWN helpers
*
  * `Spec`
  * mathematical definitions for classical and statistical models
:::

The lowercase `nn.linear`, `nn.relu`, and `optim.adam` names are the canonical spellings.
Internal implementation namespaces may be longer because they distinguish specification, runtime,
and proof layers. Import those layers when the application needs their declarations; the public
names above cover the model and training operations shown here.

# Tensor Operations

Tensor operations establish the shape conventions used by the model API. The vectors and matrices
below have explicit shapes, which Lean reports as part of their types:

```lean (name := apiVecs)
-- The matrix annotations retain row and column dimensions
-- beyond the flat entry count.
/-- Two vectors and two matrices to work with. -/
def apiV1 : Tensor Float [3] := [1.0, 2.0, 3.0]

def apiV2 : Tensor Float [3] := [4.0, 5.0, 6.0]

def apiM1 : Tensor Float [2, 3] :=
  [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]

def apiM2 : Tensor Float [3, 2] :=
  [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]

#check apiM1
```
```leanOutput apiVecs (whitespace := lax)
apiM1 : Tensor Float [2, 3]
```

The list literals are notation for tensor values, checked against the declared shape. A row of the
wrong length is a compile error rather than a surprise at the first matrix multiply.

The reported type of `apiM1` retains both dimensions, `[2, 3]`, rather than merely recording six
stored values. The first index selects one of two rows and the second selects one of three entries
in that row. A transpose and a reshape may preserve the total number of entries while assigning them
different coordinates. The matrix products below depend on that coordinate structure, so the shape
annotation carries information that a flat array's length would lose.

## Dot Product

The dot product is a sum of products, and TorchLean spells it that way:

```lean (name := apiDot)
-- Pair corresponding vector entries and sum the three
-- products.
#eval Tensor.dotSpec apiV1 apiV2
```
```leanOutput apiDot (whitespace := lax)
32.000000
```

$`1\cdot4+2\cdot5+3\cdot6=32`, and `torch.dot` agrees:

```
dot        : 32.0
```

`dotSpec` is the scalar-polymorphic reference definition used by proofs. It can also execute at
`Float`, as above. Other plain tensor operations, including `matmul`, are scalar-polymorphic too;
the absence of a `Spec` suffix does not by itself imply native dispatch or a different trust
boundary.
Use the runtime backend configuration when provider selection matters.

Its signature says the same thing:

```lean (name := apiDotSig)
-- The shared s permits any equal operand shapes, and α is
-- the scalar result type.
#check @Tensor.dotSpec
```
```leanOutput apiDotSig (whitespace := lax)
@Tensor.dotSpec : {α : Type} →
  [inst : Storage α] →
    [Context α] →
      {s : Shape} → Tensor α s → Tensor α s → α
```

Both arguments have the same shape `s`, so a dot product between a length three and a length four
vector does not typecheck. The result is a scalar of the element type, not a rank-zero tensor, which
is the choice that lets it be used directly in arithmetic.

The implicit `s` is not restricted to a vector shape: both operands may be any same-shaped tensors,
and all corresponding products contribute to the returned scalar. `Storage α` supplies their
representation, while `Context α` supplies the scalar operations used by this definition. Neither
argument changes the shape relation. In this example Lean infers `α := Float` and `s := [3]` from
the operands, so the call needs no explicit type or dimension arguments.

## Elementwise Arithmetic And Reductions

The dot product combines two operations with different result shapes. Elementwise multiplication
retains all three coordinates; a reduction combines them into a scalar. The following evaluations
show the products, then the sum and mean of the original vector:

```lean (name := apiMul)
-- Keep one product for each coordinate instead of reducing
-- the vector.
#eval Tensor.mulSpec apiV1 apiV2
```
```leanOutput apiMul (whitespace := lax)
[4.000000, 10.000000, 18.000000]
```

```lean (name := apiSum)
-- Sum the original first vector, not the product vector
-- computed above.
#eval Tensor.sumSpec apiV1
```
```leanOutput apiSum (whitespace := lax)
6.000000
```

```lean (name := apiMean)
-- Average the three entries of apiV1 using their total
-- element count.
#eval Tensor.meanSpec apiV1
```
```leanOutput apiMean (whitespace := lax)
2.000000
```

PyTorch, same three lines:

```
elementwise: [4.0, 10.0, 18.0]
sum, mean  : 6.0 2.0
```

Elementwise operations require identical shapes. To combine a reduced vector with a matrix,
the program must explicitly expand it along the intended axis.

Read the three outputs separately: `[4, 10, 18]` contains the coordinatewise products of the two
vectors, while `6` and `2` reduce `apiV1`, whose entries are `1, 2, 3`. Summing the product vector
would give the earlier dot product, `32`. This distinction is easy to miss when reductions appear
next to elementwise operations; following which value each call consumes prevents confusing a mean
of inputs with a mean of their products. The
{ref "bugzoo-catalog"}[BugZoo chapter] examines bugs caused by choosing that axis incorrectly.

## Matrix Products

Three products, distinguished by what is a matrix and what is a vector:

```lean (name := apiMatmul)
-- Contract the left columns with the right rows to produce
-- a two-by-two matrix.
#eval Tensor.matmul apiM1 apiM2
```
```leanOutput apiMatmul (whitespace := lax)
[[22.000000, 28.000000], [49.000000, 64.000000]]
```

```lean (name := apiMatvec)
-- A right-hand vector of ones sums each matrix row.
#eval Tensor.matvec apiM1 [1.0, 1.0, 1.0]
```
```leanOutput apiMatvec (whitespace := lax)
[6.000000, 15.000000]
```

```lean (name := apiVecmat)
-- A left-hand vector of ones combines the two rows
-- coordinatewise.
#eval Tensor.vecmat (α := Float) [1.0, 1.0] apiM1
```
```leanOutput apiVecmat (whitespace := lax)
[5.000000, 7.000000, 9.000000]
```

Against PyTorch:

```
matmul     : [[22.0, 28.0], [49.0, 64.0]]
matvec     : [6.0, 15.0]
vecmat     : [5.0, 7.0, 9.0]
```

The type of `matmul` states which dimension is contracted and which dimensions remain:

```lean (name := apiMatmulSig)
-- Only the shared dimension n is contracted; m and p become
-- the result dimensions.
#check @Tensor.matmul
```
```leanOutput apiMatmulSig (whitespace := lax)
@Tensor.matmul : {α : Type} →
  [inst : Storage α] →
    [Add α] →
      [Mul α] →
        [Zero α] →
          {m n p : ℕ} →
            Tensor α [m, n] →
              Tensor α [n, p] → Tensor α [m, p]
```

The shared `n` is the contraction. A dimension typo is reported where the product is written, not
where the resulting tensor is first used, and the instance list says exactly what the element type
must support: addition, multiplication, and a zero. No ordering, no division, no transcendental
functions. That is why the same operation works over the rationals in the certificate examples.

For `apiM1.matmul apiM2`, the result has two rows and two columns. Its top-left entry, `22`, is
`1 * 1 + 2 * 3 + 3 * 5`: one row of the left matrix paired with one column of the right matrix.
The two vector products make the direction equally visible. Multiplying by a three-entry vector of
ones sums each row to `[6, 15]`; a two-entry vector of ones on the left sums the rows together to
`[5, 7, 9]`. The surviving dimension tells us which sums we asked for.

Use the rearrangement notation for a bare transposed matrix:

```lean (name := apiTranspose)
-- Exchange the row and column axes, including the
-- corresponding coordinate permutation.
open TorchLean.Tensor in
#eval rearrange apiM1 "row column -> column row"
```
```leanOutput apiTranspose (whitespace := lax)
[[1.000000, 4.000000], [2.000000, 5.000000],
 [3.000000, 6.000000]]
```

Inside differentiable programs, use the corresponding `nn.functional` operations so the graph
records the rearrangement and its derivative.

## Reshape, Activations, Concatenation

These operations change different aspects of the same vector. Reshape changes its indexing
structure while retaining the values; subtracting two and applying ReLU changes the values while
retaining the length; concatenation appends a second copy and changes the length:

```lean (name := apiReshape)
-- Regroup the same three entries as a column with one entry
-- per row.
#eval apiV1.reshape [3, 1]
```
```leanOutput apiReshape (whitespace := lax)
[[1.000000], [2.000000], [3.000000]]
```

```lean (name := apiRelu)
-- Subtract two from every coordinate before ReLU clamps
-- negative results to zero.
#eval Tensor.relu
  (Tensor.subSpec apiV1 (Tensor.full [3] 2.0))
```
```leanOutput apiRelu (whitespace := lax)
[0.000000, 0.000000, 1.000000]
```

```lean (name := apiConcat)
-- Append the second vector after the first, preserving the
-- order within both copies.
#eval Tensor.concat apiV1 apiV1
```
```leanOutput apiConcat (whitespace := lax)
[1.000000, 2.000000, 3.000000, 1.000000, 2.000000, 3.000000]
```

PyTorch:

```
reshape    : [[1.0], [2.0], [3.0]]
relu(v1-2) : [0.0, 0.0, 1.0]
cat        : [1.0, 2.0, 3.0, 1.0, 2.0, 3.0]
```

`reshape` takes the target shape and checks that the element count matches, so a reshape that would
drop or invent elements is rejected at elaboration. `concat` returns shape `[6]` here, computed from
the inputs rather than supplied by the caller.

The displayed reshape is a column containing `1`, `2`, and `3`; it has not transposed a preexisting
matrix or changed those values. The activation example instead computes `[-1, 0, 1]` before ReLU
clamps the first entry, producing `[0, 0, 1]`. Concatenation preserves both copies in order. These
three results illustrate separate questions to ask of an operation: which coordinates exist, which
values occupy them, and whether any values are duplicated or discarded.

## Linear Layer Operations

Everything above composes into the operation a dense layer performs, and it is worth writing once
without any model machinery:

```lean (name := apiLinear)
-- One weight and bias pair is reused for each of these four
-- input rows.
/-- Four three-dimensional rows. -/
def apiRows : Tensor Float [4, 3] :=
  [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0],
   [0.0, 0.0, 1.0], [1.0, 1.0, 1.0]]

/-- A projection onto the first two coordinates. -/
def apiWeight : Tensor Float [2, 3] :=
  [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]

def apiBias : Tensor Float [2] := [0.5, -0.5]

#eval Tensor.linear apiRows apiWeight apiBias
```
```leanOutput apiLinear (whitespace := lax)
[[1.500000, -0.500000], [0.500000, 0.500000],
 [0.500000, -0.500000], [1.500000, 0.500000]]
```

That is $`xW^{\top}+b` applied to every row, and `torch.nn.functional.linear` returns the same
tensor:

```
linear: [[1.5, -0.5], [0.5, 0.5], [0.5, -0.5], [1.5, 0.5]]
```

The two weight rows select the first and second input coordinates. Adding the bias then shifts
every selected pair by `[0.5, -0.5]`. That gives `[1.5, -0.5]` for the first basis vector and
`[0.5, -0.5]` for the third, whose selected coordinates are both zero. The final input row has both
selected coordinates equal to one, producing `[1.5, 0.5]`. All four rows use one weight matrix and
one bias; batching does not allocate four independent affine maps.

The input above is a batch of four rows. The signature extends the same operation to any leading
dimensions:

```lean (name := apiLinearSig)
-- EndsWith identifies the trailing feature width while
-- preserving every leading dimension.
#check @Tensor.linear
```
```leanOutput apiLinearSig (whitespace := lax)
@Tensor.linear : {α : Type} →
  [inst : Storage α] →
    [Add α] →
      [Mul α] →
        [Zero α] →
          {shape : Shape} →
            {inputWidth outputWidth : ℕ} →
              [endsWith : Shape.EndsWith inputWidth shape] →
                Tensor α shape →
                  Tensor α [outputWidth, inputWidth] →
                    Tensor α [outputWidth] →
                      Tensor α (shape.replaceLast outputWidth)
```

The input may have any shape ending in `inputWidth`, and the output replaces that last dimension.
So the same declaration handles an unbatched vector, a batch of rows, and a batch of sequences of
rows, with the leading dimensions carried along by the `EndsWith` instance rather than by a runtime
reshape. This is the pattern to look for throughout the API: batching is a shape fact, resolved by
instance search, not a separate code path.

# Naming And Dot Syntax

Dot syntax is used when the value before the dot is the natural subject of the operation:
`tensor.reshape`, `module.run`, `trainer.train`, `trained.predict`, and
`result.printSummary`. The reshape above was written `apiV1.reshape [3, 1]` for exactly that
reason. A leading dot is used for a choice whose expected type is already known, such as `.native`,
`.eager`, or `.meanSquaredError`; the type checker knows which enumeration is meant, so repeating
its name would be noise.

Ordinary application code does not construct shapes through recursive representation constructors;
it writes `[]`, `[width]`, or `[batch, width]`. It also does not inspect public results through
positional tuple projections. Multi-value transforms use tuple destructuring, while training
progress reads as `.loss.before` and `.loss.after`. Runtime representations such as `TensorPack`
and `ObjectiveDefinition` stay behind the maintained API.

These conventions keep operations such as prediction and fields such as reported loss visible
at the call site without requiring callers to unpack the runtime representation.

# A Complete Training Program

The XOR problem supplies four input rows and a scalar target for each. The program below trains a
two-layer network with mean-squared error and Adam, then evaluates it at a point outside those
four rows. Save the declarations and `main` in `Scratch.lean` at the repository root, with the
import and `open` shown above.

The model, data, and trainer are ordinary declarations, so we can elaborate them right here and read
their types before running anything:

```lean (name := apiModel)
-- Keep the four XOR row pairs separate; the training loop
-- will group their gradients.
def apiModel :
    nn.Builder (nn.Sequential [2] [1]) :=
  nn.Sequential![
    nn.linear 2 8,
    nn.relu,
    nn.linear 8 1
  ]

def apiXs : Tensor Float [4, 2] :=
  [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]

def apiYs : Tensor Float [4, 1] :=
  [[0.0], [1.0], [1.0], [0.0]]

def apiData := Data.fromTensors apiXs apiYs

def apiTrainer :=
  Trainer.new apiModel
    { objective := .meanSquaredError
      optimizer := optim.adam { learningRate := 0.03 }
      arithmetic := .native
      execution := .eager
      seed := 2026 }

#check apiData
```
```leanOutput apiModel (whitespace := lax)
apiData : Trainer.Dataset [2] [1]
```

As with PyTorch's `TensorDataset`, TorchLean infers `Trainer.Dataset [2] [1]` because `apiXs` has
batched shape `[4, 2]` and `apiYs` has batched shape `[4, 1]`. The two shape parameters describe one
input sample and one target sample; the leading batch dimension is removed by the dataset
constructor, and its size only has to agree between the two arguments.

Materializing this dataset will therefore yield four paired items, not one item containing all
four rows. The model still consumes a `[2]` input and returns a `[1]` prediction. In the `main`
below, `samplesPerStep := 4` gathers the four item gradients at one parameter point before Adam
updates it. The grouping belongs to the training loop; it has not changed the model's shape into
`[4, 2] → [4, 1]`. The constructor signature shows where the shared row count is required:

```lean (name := apiFromTensors)
-- The shared leading n is removed from the shape of each
-- paired dataset item.
#check @Data.fromTensors
```
```leanOutput apiFromTensors (whitespace := lax)
@Data.fromTensors : {n : ℕ} →
  {σ τ : Shape} →
    Tensor Float (σ.prependDim n) →
      Tensor Float (τ.prependDim n) → Trainer.Dataset σ τ
```

The shared `n` is why a dataset with four inputs and three targets cannot be built. Add a `main`
that trains and predicts:

```
-- Average all four XOR sample gradients before each Adam
-- update.
def main : IO Unit := do
  let trained ← apiTrainer.train apiData
    { steps := 20
      samplesPerStep := 4
      logEvery := 5 }
  trained.printSummary

  let heldout : Tensor Float [2] :=
    [0.25, -0.75]
  let yhat ← trained.predict heldout
  IO.println s!"prediction={reprStr yhat}"
```

Lean's structure-instance notation is whitespace sensitive. In the `Trainer.new` configuration,
keep the first field aligned with the remaining fields as shown; misaligning `objective :=` can
produce `unexpected identifier; expected '}'` several lines later.

Check the definitions without starting the training run:

```terminal
# Elaborate the declarations and main without invoking main.
lake env lean Scratch.lean
```

Lean stays silent here apart from diagnostics. To execute the `main` in the file:

```terminal
# Run main after elaboration to execute the requested
# updates and held-out prediction.
lake env lean --run Scratch.lean
```

The run is deterministic at `seed := 2026`, and it ends with:

```terminal +output
dataset size = 4
mean_loss(before training) = 0.731368
step 0: loss=0.731368
step 5: loss=0.255428
step 10: loss=0.257351
step 15: loss=0.189020
mean_loss(after training) = 0.165308
steps=20 arithmetic=native scalar=Float32 loss=0.731368 -> 0.165308
prediction=[0.087481]
```

After twenty steps, the loss has fallen substantially, but the network has not fit the four XOR
targets. The logged loss is also not monotone: step 10 is worse than step 5. The prediction
`0.087481` at $`(0.25, -0.75)` needs a separate interpretation, since XOR supplies no target for
that point. The summary line
also records what the numbers mean: `arithmetic=native scalar=Float32`, so this is Lean's native
binary32 arithmetic, not the host binary64 in which the dataset literals were written.

Because each update in this program accumulates all four XOR rows, its step loss is the mean over
those rows at the current parameters. This differs from a single-sample training log, where adjacent
steps may report different samples. Even a full-dataset loss need not decrease at every Adam update:
the update uses its stored moments and configured rate. The lower final mean records progress over
the run, while the intervening entries show the particular trajectory.

Change `steps := 400` and `logEvery := 100`, and the same program fits the four training rows
to the displayed precision:

```terminal +output
dataset size = 4
mean_loss(before training) = 0.731368
step 0: loss=0.731368
step 100: loss=0.000058
step 200: loss=0.000000
step 300: loss=0.000000
mean_loss(after training) = 0.000000
steps=400 arithmetic=native scalar=Float32 loss=0.731368 -> 0.000000
prediction=[0.191567]
```

The corresponding architecture and optimizer in PyTorch,
`Sequential(Linear(2,8), ReLU(), Linear(8,1))` with `MSELoss` and
`Adam(lr=0.03)` for 400 full-batch steps:

```terminal +output
step 0: loss=0.402008
step 100: loss=0.000006
step 200: loss=0.000000
step 300: loss=0.000000
final loss=0.000000
prediction at [0.25,-0.75]: [0.9572102427482605]
```

Both frameworks drive the training loss below the displayed six-decimal resolution on the four rows.
They give different predictions away from the training
data: `0.191567` against `0.957210` at the same unseen point. The training objective does not
choose between those predictions. The initial
parameters differ (the initial losses, `0.731368` against `0.402008`, already say so), the four
training rows do not determine a unique function away from those points, and the optimization paths
can lead to different interpolants. A small final training loss alone therefore does not establish
agreement on new inputs. The {ref "certificates"}[verification chapters] develop a different kind
of evidence: statements that hold over whole input regions under explicit hypotheses.

The `--run` path is convenient for a one-file experiment. A maintained command should still receive
a Lake executable target so it can be built and invoked by name.

# Type Inspection In VS Code

Place the cursor on `apiModel`. The infoview shows the builder type, which is also what `#check`
prints:

```lean (name := apiModelType)
-- The boundary shapes belong to the builder even before its
-- seed has been supplied.
#check apiModel
```
```leanOutput apiModelType (whitespace := lax)
apiModel : nn.Builder (nn.Sequential [2] [1])
```

The `[2]` and `[1]` are the shapes of one input and one output sample. The layer list is checked
against the annotation on `apiModel`. Changing the final layer from `nn.linear 8 1` to
`nn.linear 7 1` breaks the boundary between the hidden representation and the output layer:

```lean +error (name := apiTypo)
-- The hidden layer produces eight entries, so a final layer
-- expecting seven cannot compose.
def apiTypo :
    nn.Builder (nn.Sequential [2] [1]) :=
  nn.Sequential![
    nn.linear 2 8,
    nn.relu,
    nn.linear 7 1
  ]
```
```leanOutput apiTypo (whitespace := lax)
Application type mismatch: The argument
  bc✝
has type
  nn.Sequential (Shape.appendDim [] 7) (Shape.appendDim [] 1)
but is expected to have type
  ?m.67 a✝ __r✝¹ bc✝ __r✝ (Shape.appendDim [] 8) [1]
in the application
  nn.compose a✝ bc✝
```

The error identifies the failed composition: the final layer expects width `7`, while the
preceding layer produces width `8`. Lean finds this mismatch from the model definition alone,
before a forward pass, data loading, or parameter initialization.

The layer constructor's signature explains the shapes in the error:

```lean (name := apiLinearBuilder)
-- The default empty batch shape makes the short constructor
-- act on a single feature vector.
#check @nn.linear
```
```leanOutput apiLinearBuilder (whitespace := lax)
@nn.linear : (inputWidth outputWidth : ℕ) →
  (batchShape : optParam Shape []) →
    optParam nn.Linear.Config { } →
      nn.Builder
        (nn.Sequential (batchShape.appendDim inputWidth)
          (batchShape.appendDim outputWidth))
```

Both the batch shape and the layer configuration are optional parameters with defaults, which is why
`nn.linear 2 8` is enough in the common case and why the pretty-printed type in the error mentions
`Shape.appendDim [] 7` rather than a bare `[7]`.

Place the cursor on `trained.predict`. Its input and output are trainer-facing host-`Float` tensors
with the model's checked shapes. The retained runner handles conversion to the arithmetic semantics
selected by the training configuration.

# Builder, Trainer, And Training Result

These three values have different lifetimes, and keeping them apart is what makes the same
architecture reusable:

```lean (name := apiTrainerNew)
-- ToModel converts a supported model representation into
-- the trainer with these boundary
-- shapes.
#check @Trainer.new
```
```leanOutput apiTrainerNew (whitespace := lax)
@Trainer.new : {Model : Type u_1} →
  {σ τ : Shape} →
    [Trainer.ToModel Model σ τ] →
      Model → optParam (Trainer.Config σ τ) { } → Trainer σ τ
```

A `nn.Builder (nn.Sequential input output)` describes architecture and seeded initialization.
`Trainer.new model config` materializes the builder at the selected seed and attaches objective,
optimizer, and runtime choices; it has not consumed data. `trainer.train data options` executes
updates and returns a `Trainer.Result` with the final parameters and prediction operations.

The `Trainer.ToModel` instance is the reason `Trainer.new` accepts a builder, an already-built
model definition, or anything else that can present itself as a model with those shapes, without a
conversion
call at the use site. Splitting the three stages permits the same architecture to be initialized
with several seeds, trained on several datasets, or interpreted by another runtime without
redefining its layers. It also means a failed run is diagnosable: the builder either typechecks
or it does not, independently of whether any data was available.

The signature's model type `M` is abstract because the constructor asks the `ToModel M input output`
instance for a compatible model definition. The resulting `Trainer input output` retains those
boundary shapes but hides the original construction syntax. A caller can consequently share a
training routine between a sequential builder and another supported model representation. This
abstraction changes how the model is supplied; it does not relax the requirement that its input and
output agree with the data and objective.

# Persistent And Per-Call Configuration

Persistent choices can be expressed as `Trainer.RunConfig`:

```
-- Keep optimizer and arithmetic fixed while selecting
-- execution mode or device profile.
def eagerCpu : Trainer.RunConfig :=
  { optimizer := optim.adam { learningRate := 0.03 }
    arithmetic := .native
    execution := .eager }

def typedGraphCpu : Trainer.RunConfig :=
  { eagerCpu with
    execution := .typedGraph
    device := .cpu }

def configuredTrainer :=
  Trainer.new model
    (Trainer.RunConfig.forObjective
      typedGraphCpu .meanSquaredError
      (seed := 2026))
```

`RunConfig` contains the optimizer, arithmetic semantics, execution mode, and backend profile. See
{ref "backend-selection"}[Backend Selection] for the
profile's provider, VJP, and evidence rules.

Optimizers are opaque values built from a record of hyperparameters. Their descriptions display
the constructor and configuration:

```lean (name := apiDescribe)
-- The text includes the Adam defaults that were not
-- repeated in the constructor.
#eval (optim.adam { learningRate := 0.03 }).describe
```
```leanOutput apiDescribe (whitespace := lax)
"optim.adam { learningRate := 0.030000,
  beta1 := 0.900000, beta2 := 0.999000,
  epsilon := 1.000000e-8 }"
```

The moment coefficients match PyTorch's Adam defaults, so a script ported from
`torch.optim.Adam(params, lr=0.03)` need not restate them. The stabilizer prints in exponent form
so its nonzero value remains visible; setting epsilon to zero would change the update rule.
The description is formatted text. Read the field defaults from `optim.Adam.Config` when the
exact values matter.

The available optimizers are `optim.sgd` (with optional heavy-ball momentum
{Informal.citep polyak1964}[]), `optim.adaGrad`, `optim.rmsProp`, `optim.adam`, `optim.adamW`
(decoupled weight decay {Informal.citep adamw2019}[]), and `optim.adaDelta`. They are configuration
records at this layer; the update rules themselves live in the runtime and are stated as
specifications in the training chapter.

Executable frontends can convert already-parsed common runtime flags with
`Trainer.RunConfig.fromRuntime`. Command-line applications use
`TorchLean.CLI.Trainer.parseCommandLine` to parse these flags into a training runtime configuration.

Per-call `TrainOptions` controls step count, sample grouping, logging cadence, and artifact fields.
`loadCheckpoint?` and `saveCheckpoint?` load or save model state around ordinary training, for
mean-squared error, cross entropy, and custom losses. They do not snapshot optimizer moments or
loader state. Alternating two-model training rejects these single-path options because one path
cannot represent both model states. `Trainer.RunConfig.withRuntime` overwrites the runtime dials of
a configuration from a
`Runtime.Config`, which is how a study varies execution mode or device without touching the
optimizer or the objective.

# Runtime Arithmetic

`Runtime.Arithmetic` is not a storage dtype. It chooses the arithmetic for an executable run.
`.native` selects Lean's native binary32 `Float32`; `.ieee` selects TorchLean's independent
raw-bit `IEEE32Exec` implementation; `.complex` selects complex numbers over that reference
binary32 implementation. The choice changes the meaning of arithmetic operations rather than
attaching a label to an untyped buffer. This is why the scratch run above reported
`arithmetic=native scalar=Float32` in its summary line: the run states its own semantics.

Trainer-facing datasets are commonly authored as `Tensor Float ...`, where `Float` is Lean's host
binary64 type. The dataset builder converts those values once the trainer selects its scalar
semantics. This is an input boundary, not a claim that training itself uses binary64.

Proofs choose `ℝ` or `TorchLean.Floats.FP32` directly. They are not runtime modes because
they are noncomputable. The high-level trainer accepts the real runtime modes. Genuine complex
training needs a real-valued loss, conjugate-aware reverse mode, and complex prediction results, so
the generic runtime's `.complex` mode is not presented as a supervised trainer option.

For theorem work, instantiate specification tensors directly over `ℝ` or `FP32`. For a runtime run,
choose executable arithmetic and record the provider boundary.

The direct tensor calculations earlier in this chapter use the `Float` written in their
annotations. They are not silently redirected through a trainer's `.native` setting. The same is
true of the explicit derivative example below, whose input is a `Tensor Float [3]`. The conversion
to native `Float32` belongs to the high-level training boundary. Keeping that boundary visible
avoids
attributing a binary64 calculation to binary32 merely because both print six decimal places.

# Runtime-Polymorphic Data

`Trainer.Dataset input target` knows how to materialize samples after the trainer selects
a scalar:

```
-- These constructors differ in data source and grouping
-- while returning the same dataset
-- interface.
Data.fromTensors
Data.fromSamples
Data.fromSupervisedSource
Data.fromCsv
Data.batch
```

The model and dataset must agree on their input and target dimension lists. A file loader checks
runtime dimensions while materializing the typed samples.

A true tensor minibatch changes shapes to `[batch,...]`. `Data.batch` stores each fixed-size
minibatch as one dataset item, so `TrainOptions.samplesPerStep := 1` executes one vectorized
forward/backward pass per update. On an unbatched dataset, the same option counts individual
samples. Values above one always mean gradient accumulation across several dataset items. The data
chapter develops both forms.

# Explicit Differentiation

Derivatives are taken of a program, and the program is a differentiable tensor function rather than
an arbitrary Lean function:

```lean (name := apiEnergy)
-- square retains three coordinates; mean produces the
-- scalar tensor required by grad.
/-- Mean of squares, as a differentiable tensor program. -/
def apiEnergy : autograd.Function [3] [] := fun x => do
  let squares ← nn.functional.square x
  nn.functional.mean squares

def apiVec : Tensor Float [3] := [1.0, 2.0, 3.0]

#eval autograd.grad apiEnergy apiVec (value := true)
```
```leanOutput apiEnergy (whitespace := lax)
([0.666667, 1.333333, 2.000000], 4.666667)
```

That is $`\tfrac{1}{3}\sum_i x_i^2` at $`(1,2,3)`, whose value is $`14/3 = 4.6\overline{6}` and
whose gradient is $`\tfrac{2}{3}x = (0.6\overline{6}, 1.3\overline{3}, 2)`. Passing
`value := true` returns the pair so the forward pass is not recomputed.

The tuple is ordered as gradient first, value second. Its first component has shape `[3]`, with
one derivative for each input coordinate; its second has shape `[]`, because the mean has reduced
all three squared entries to one scalar tensor. In particular, the final gradient coordinate is
exactly `2` in the displayed result because that input was `3`. The other two coordinates and the
loss require rounding when represented as `Float`. PyTorch's `backward` on
the same expression:

```
value: 4.666666507720947 grad: [0.6666666865348816, 1.3333333730697632, 2.0]
```

Same numbers, to the last displayed digit of TorchLean's six-decimal formatting. The shipped
quickstart prints the same result from a compiled binary rather than from elaboration:

```terminal +output
== Differentiate a tensor function ==
mean(x^2) = 4.666667
d/dx       = [0.666667, 1.333333, 2.000000]
```

The other transforms follow the same functional interface, which will look familiar from JAX
{Informal.citep jax2018}[]: a transform consumes a function and returns a function's results, rather
than mutating tensors. `jacrev` builds the Jacobian by reverse mode, one row per output:

```lean (name := apiJacrev)
-- A scalar-output Jacobian has only the input axes and
-- therefore agrees with the gradient.
#eval autograd.jacrev apiEnergy apiVec
```
```leanOutput apiJacrev (whitespace := lax)
[0.666667, 1.333333, 2.000000]
```

A scalar output has no axes, so this Jacobian has the input shape and equals the gradient above.

In general the Jacobian shape is the output shape followed by the input shape. Here the empty
output shape adds no dimensions, leaving `[3]`. The Hessian differentiates that three-coordinate
gradient, so it has shape `[3, 3]`: one coordinate identifies a gradient component and the other the
input coordinate being varied. Off-diagonal zeros mean that changing one input coordinate does not
change the gradient component of another for this particular separable function.
`hessian` composes forward over reverse to get second derivatives:

```lean (name := apiHessian)
-- Differentiate each gradient component to obtain a
-- three-by-three matrix.
#eval autograd.hessian apiEnergy apiVec
```
```leanOutput apiHessian (whitespace := lax)
[[0.666667, 0.000000, 0.000000],
  [0.000000, 0.666667, 0.000000],
  [0.000000, 0.000000, 0.666667]]
```

$`\tfrac{2}{3}I`, as the second derivative of a mean of squares must be, with exact zeros off the
diagonal rather than small noise. `torch.func.hessian` on the same function returns
`[[0.6666666865348816, 0.0, 0.0], [0.0, 0.6666666865348816, 0.0], [0.0, 0.0, 0.6666666865348816]]`,
the same matrix. Composing forward and reverse mode this way, rather than differentiating twice in
reverse, is the standard choice for a small input dimension {Informal.citep baydin2018}[].

The full set is `autograd.grad`, `autograd.vjp`, `autograd.jacfwd`, `autograd.jacrev`, and
`autograd.hessian` for tensor functions, and `autograd.model.grad`, `autograd.model.vjp`,
`autograd.model.jvp`, and `autograd.model.hvp` for a checked model.

Derivatives are returned as values. Parameter derivatives have the same dependent tensor-pack
structure as the parameters; there is no mutable `.grad` field on `Tensor` values. This is a
consequence of the host language rather than a stylistic preference: Lean values are immutable, and
its compiler recovers in-place updates through reference counting instead
{Informal.citep immutablebeans2019}[] . Each query returns a fresh gradient result without
accumulating into a persistent tensor field.
Callers can still accidentally reuse an old returned value or explicitly combine the wrong packs.
There is
no `zero_grad` in this API because there is nothing to zero. A model `vjp` returns a state gradient
and input gradient from one reverse pass; callers destructure the pair.

The model quickstart shows what a parameter gradient looks like when the parameters are a pack of
shaped tensors rather than a single vector:

```terminal +output
== Differentiate a model loss ==
loss     = 0.769887
gradient = [[1, 2]: [[-0.877432, 1.754865]], [1]: [-1.754865]]
```

Each entry is labeled by its shape, `[1, 2]` for the weight and `[1]` for the bias, so the printed
gradient has the same structure as the parameters it differentiates.

The two numbers in the weight entry are derivatives with respect to the two separate weights;
the bias entry has only one coordinate. The scalar loss printed above the pack is the objective
value at the same model state used to obtain those derivatives. This is enough to feed an optimizer
or compare a derivative calculation, but it is not itself an update: a gradient result leaves the
choice of learning rate, optimizer memory, and parameter replacement to its caller.

# Functional Tensor Operations

Use `nn.functional` when constructing a differentiable tensor program, as `apiEnergy` did above. An
arbitrary Lean function over `TorchLean.Tensor` is useful for specifications, and the `Tensor`
operations from the first section of this chapter are exactly that, but a plain function does not
carry runtime graph and derivative behavior. `nn.functional.square` and `Tensor.square` compute the
same mathematics; only the first one can be differentiated by the transforms above.

The distinction is the usual one for an embedded differentiable language: operations must register
the semantics that execution and automatic differentiation need. Tracing frameworks reach the same
place from the other direction, by recording a graph while ordinary code runs
{Informal.citep fx2022}[] . Here the graph is what the `autograd.Function` monad builds, so
typechecking establishes the operation interface and shapes. A selected transform can still
reject operations whose derivative lowering it does not support.

# Classical Models

Statistical and classical ML models do not need a neural layer stack. Their definitions live in
`Spec`, alongside the tensor mathematics they use. Keeping them in the library does not require
pretending they are neural networks; what they share with the neural code is the shape-indexed
tensor foundation and explicit data, not one forced architecture abstraction.

These models are not re-exported through `NN.API`, so a file that uses them adds one focused import.
Here is a complete executable k-nearest-neighbor classifier. The model stores its labeled samples;
evaluation computes distances and applies a deterministic majority vote, including deterministic
tie-breaking by neighbor order.

```lean (name := apiKnn)
-- Three neighbors vote using the four stored labeled
-- points; there is no optimizer step.
/-- Four labeled points, classified by three neighbors. -/
def apiPoint (x y : Float) : Tensor Float [2] := [x, y]

def apiKnn : Spec.KNN Float String 2 :=
  Spec.KNN.fromData Float String 2 3 #[
    (apiPoint 0.0 0.0, "blue"),
    (apiPoint 0.0 1.0, "blue"),
    (apiPoint 3.0 3.0, "orange"),
    (apiPoint 3.0 4.0, "orange")]

open Spec in
#eval classify Float String 2 apiKnn (apiPoint 0.2 0.1)
```
```leanOutput apiKnn (whitespace := lax)
"blue"
```

The dataset is an `Array` of pairs, and the `3` after the dimension is $`k`. A point near the far
cluster classifies the other way:

```lean (name := apiKnnOrange)
-- This query lies near the two orange points rather than
-- the two blue points.
open Spec in
#eval classify Float String 2 apiKnn (apiPoint 2.9 3.5)
```
```leanOutput apiKnnOrange (whitespace := lax)
"orange"
```

With $`k = 3` and only four stored points, every query mixes clusters, and the confidence is the
vote share rather than a probability:

```lean (name := apiKnnConf)
-- Return the winning label together with its fraction of
-- the three selected neighbor votes.
open Spec in
#eval classifyWithConfidence Float String 2
  apiKnn (apiPoint 0.2 0.1)
```
```leanOutput apiKnnConf (whitespace := lax)
("blue", 0.666667)
```

Two of the three nearest neighbors of $`(0.2, 0.1)` are blue, so the ideal vote share is
$`2/3` ; the returned `Float` is rounded. It is not calibrated and does not become one by being
computed in Lean.

This classifier has no seed-consuming builder or optimizer step. Its state is the four stored
labeled points, and a query selects neighbors from that state before voting. With three neighbors,
a cluster containing only two stored points can supply at most two votes even for a query very close
to it. The returned `0.666667` therefore describes the composition of the selected neighbor set;
it is not a fitted estimate of the chance that the query's label is blue.

The type says what the classifier needs from its scalars and labels:

```lean (name := apiClassifySig)
-- Ordering ranks distances, label equality tallies votes,
-- and Inhabited handles an empty vote.
open Spec in
#check @classify
```
```leanOutput apiClassifySig (whitespace := lax)
classify : (α β : Type) →
  (n : ℕ) →
    [inst : Storage α] →
      [inst_1 : Context α] →
        [DecidableRel fun x1 x2 => x1 > x2] →
          [BEq β] →
            [Hashable β] →
              [Inhabited β] →
                KNN α β n → Tensor α [n] → β
```

The decidable ordering lets the classifier rank distances, and `BEq` and `Hashable` let it tally
labels. `Inhabited` supplies a default label when there are no neighbors to vote. Callers that
require a vote-supported prediction must therefore ensure that neighbors exist. Other families,
including the GMM and HMM likelihood operations below, expose invalid inputs through optional
results.

The exported families deliberately expose different amounts of fitting and inference machinery:

:::table +header
*
  * Family
  * Available operations
  * Current boundary
*
  * kNN
  * nearest neighbors, classification, regression, confidence, and batch mapping
  * lazy stored-data model; no learned index or metric
*
  * random forest
  * symbolic-tree aggregation plus numeric regression fitting and Gini classification-tree fitting
  * deterministic reference fitting; rotated resamples replace randomized bootstrapping
*
  * naive Bayes
  * multinomial string-feature counting, log scores, prediction, and negative log likelihood
  * specialized to bags of `String` features and labels; fitting is counting, not gradient training
*
  * SVM
  * linear decisions, hinge objective, VJP, gradient-descent fit, prediction, and kernel functions
  * the fitter is a linear primal baseline; exported kernels do not constitute a kernel-SVM solver
*
  * GMM
  * component log densities, responsibilities, VJP, log likelihood, initialization, and EM
  * evaluation is optional and rejects invalid weights or non-positive-definite covariances
*
  * PCA
  * projection, inverse, VJP, reconstruction statistics, and a leading-component fit
  * fitting approximates one component with fixed power iteration; it is not a full SVD-based PCA
    fit
*
  * linear regression
  * scalar and batched forward/VJP, one gradient step, metrics, and regularized loss variants
  * exposes mathematical update primitives rather than a separate multi-step estimator
*
  * logistic regression
  * deterministic gradient-descent fit, probabilities, and thresholded predictions
  * unregularized binary baseline with explicit sigmoid evaluation, not an optimized solver
*
  * gradient-boosted trees
  * regression ensemble evaluation and boosting steps plus standalone Gini classification-tree
    fitting
  * bounded, deterministic tree routines; no exported boosted classification ensemble
*
  * HMM
  * scaled and unscaled forward passes, batching, likelihood, initialization, and Baum-Welch updates
  * finite discrete observations; probability normalization is an input invariant, not a type
    invariant
:::

These definitions execute directly when their scalar `Context` is executable, as `Float` is in the
example. They are pure tensor/reference algorithms rather than automatic `Trainer.new`, typed
graph, LibTorch, or CUDA routes. Shape indices rule out dimensional mismatches, but they do not by
themselves prove statistical assumptions, optimizer convergence, or a family-wide correctness
claim; consult each declaration's hypotheses and result type, especially `Option`-returning GMM and
HMM likelihood operations.

# Additional Imports

Use:

```
-- Use the broader umbrella when a file intentionally
-- combines several library layers.
import NN
```

when a file needs several lower layers, such as model code plus proof declarations and
backend inspection.

Focused subsystem imports include:

```
-- Choose the subsystem import that supplies the lower-level
-- declarations the file actually
-- uses.
import NN.Spec
import NN.Runtime
import NN.Floats
import NN.Verification
import NN.GraphSpec
```

Prefer the narrowest stable import that expresses the file's responsibility. The kNN section above
is the concrete case: `NN.API` does not carry the classical models, so the example adds
`import NN.Spec.Models.Knn` and nothing else. A numerical theorem should not import the entire
executable example catalog merely for convenience, and a training script should not depend on an
internal tape constructor.

# API Objects And Their Semantics

These objects may all refer to the same architecture:

:::table +header
*
  * Object
  * What it says
*
  * model declaration
  * layer structure and shapes
*
  * trainer run
  * one runtime configuration executed
*
  * `NN.IR.Graph`
  * explicit operation data
*
  * backend audit
  * provider and evidence choices
*
  * theorem
  * exactly one Lean proposition under hypotheses
*
  * certificate
  * accepted external claim plus checker theorem
:::

The distinction matters when reading a claim about a model. A trainer run reports what happened
once, under one arithmetic and one device. A theorem reports what must happen under its hypotheses,
on every input they allow. The XOR transcripts earlier in this chapter are the first kind. The
TorchLean API makes the common path concise without collapsing these meanings.

# Declaration Search

Use the generated API search:

```
/docs/search.html
```

On the published project site the prefix is `/TorchLean/docs/search.html`; a root-mounted local
preview uses `/docs/search.html` . The website navigation supplies the configured prefix. For source
search:

```terminal
# Search the source declarations for derivative definitions
# and soundness theorems.
rg -n "def diff|theorem .*sound" NN
```

The API reference answers “what is the exact declaration?” The surrounding chapters explain why
and when to use it. Source remains authoritative when a lower-level contract matters. If a name in
this book and a name in the source disagree, the source is right and the book has a bug; the
named Lean blocks are elaborated against the library. Prose and unmarked reproduction snippets
still need review when APIs change.

# CLI Examples And Related Chapters

Run:

```terminal
# These commands exercise public tensor, derivative, and
# training interfaces from compiled
# examples.
lake exe torchlean --help
lake exe torchlean quickstart_tensors
lake exe torchlean quickstart_autograd
lake exe torchlean quickstart_mlp --steps 20
```

The tensor quickstart uses one interface with four scalar types. The output lets us compare their
values without changing the tensor representation:

```terminal +output
== Quickstart: tensor basics ==
Rank-zero tensor: 0.500000
Float tensor:    [0.100000, 0.200000, 0.300000, 0.400000]
Rational tensor: [(1 : Rat)/10, (1 : Rat)/5, (3 : Rat)/10, (2 : Rat)/5]
Integer tensor:  [1, 2, 3, 4]
Float32 cast:    [0.100000, 0.200000, 0.300000, 0.400000]
```

The rational and floating-point rows distinguish stored values from decimal formatting.
`0.1` is not representable in binary floating point, so
the `Float` row prints a rounded value while the `Rat` row holds exactly $`1/10`. The certificate
chapters run their geometry over `Rat` for precisely this reason, and it is the same `Tensor` type
in both rows.

These four commands cover the tensor, derivative, model, dataset, trainer, and prediction
interfaces without requiring backend or proof internals.

Lean's
[source-file and module
reference](https://lean-lang.org/doc/reference/latest/Source-Files-and-Modules/) explains how these
imports determine the environment in which a file elaborates. With the common path in one place, the
next chapter keeps this same model and changes only how it executes.
