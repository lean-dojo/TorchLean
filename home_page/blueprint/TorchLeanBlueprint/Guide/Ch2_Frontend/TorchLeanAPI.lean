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
import TorchLean.NN.Models.KNN
import TorchLeanBlueprint.Bib
import TorchLeanBlueprint.Roles
import TorchLean.NN.Loss
import TorchLean.NN.Session.Random
import TorchLean.Verify.Trainer

open Verso.Genre Manual
open Verso.Genre.Manual.InlineLean
open TorchLean Arithmetic Trainer.Objective
open Lean.Elab.Tactic.GuardMsgs.WhitespaceMode (lax)

#doc (Manual) "TorchLean API" =>
%%%
tag := "torchlean-api"
file := "The-TorchLean-API"
%%%

The chapter's examples start with these five lines:

```
import TorchLean.NN.Autograd.Function
import TorchLean.NN.Data.Dataset
import TorchLean.NN.Trainer.Constructor
import TorchLean.NN.Trainer.Train
open TorchLean
```

These imports supply the tensors, model builders, and training operations used below. A file that
also proves a property or inspects a backend can add the relevant module to its imports.

Keeping those dependencies separate limits what a training script needs to elaborate. Lean's module
system determines which declarations are available in a file {Informal.citep lean4}[], and the
import list records those dependencies.

Arithmetic and device are separate choices in the trainer configuration. The imports leave them
open: we can keep a model's layers and shapes fixed while changing how a training run executes.

Our XOR example below uses the same architecture and optimizer in TorchLean and PyTorch
{Informal.citep pytorch2019}[]. Both fit the four training rows, yet predict differently away from
them. First come the calculations that establish the tensor and shape conventions needed to read
that comparison. Named Lean blocks are build-checked, while shell and PyTorch transcripts document
separate runs.

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
  * `Arithmetic`, `Backend`
  * arithmetic semantics, device, and backend-contract selection
*
  * `Verification`
  * model-to-IR lowering and IBP/CROWN helpers
*
  * one per model, such as `KNN` or `PCA`
  * mathematical definitions for classical and statistical models
:::

`nn.linear`, `nn.op`, and `optim.adam`, all lowercase, are the canonical spellings. Specification,
runtime, and proof declarations all sit under the `TorchLean` root with short names. Import the
module that holds a declaration when the application needs it (the public names above cover the
model and training operations shown here).

# Tensor Operations

Tensor operations establish the shape conventions used by the model API. Each vector and matrix
below declares its shape, which Lean reports as part of its type:

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

Those list literals are notation for tensor values, checked against the declared shape. A row of
the wrong length is a compile error, not a surprise at the first matrix multiply.

`[2, 3]` records more than six stored values: it fixes their row and column coordinates.
A transpose and a reshape can preserve the entry count while assigning different coordinates.

## Dot Product

A dot product is a sum of products, and TorchLean spells it that way:

```lean (name := apiDot)
-- Pair corresponding vector entries and sum the three
-- products.
#eval Tensor.dot apiV1 apiV2
```
```leanOutput apiDot (whitespace := lax)
32.000000
```

$`1\cdot4+2\cdot5+3\cdot6=32`, and `torch.dot` agrees:

```
dot        : 32.0
```

`Tensor.dot` is the scalar-polymorphic reference definition used by proofs. It can also execute at
`Float`, as above. Other plain tensor operations, including `matmul`, are scalar-polymorphic too,
and calling one doesn't by itself imply native dispatch or a different trust boundary.
Use the runtime backend configuration when provider selection matters.

Its signature says the same thing:

```lean (name := apiDotSig)
-- The shared s permits any equal operand shapes, and α is
-- the scalar result type.
#check @Tensor.dot
```
```leanOutput apiDotSig (whitespace := lax)
@Tensor.dot : {α : Type} →
  [inst : Storage α] →
    [Add α] → [Mul α] → [Zero α] →
      {s : Shape} → Tensor α s → Tensor α s → α
```

Both arguments have the same shape `s`, so a dot product between a length three and a length four
vector doesn't typecheck. Its result is a scalar of the element type, not a rank-zero tensor, and
that choice is what lets it be used directly in arithmetic.

Nothing restricts the implicit `s` to a vector shape: both operands may be any same-shaped tensors,
and all corresponding products contribute to the returned scalar. `Storage α` supplies their
representation, while `Add α`, `Mul α`, and `Zero α` supply addition, multiplication, and the
initial zero for the sum. None of these instances changes the shape relation. In this example Lean
infers `α := Float` and `s := [3]` from the operands, so the call needs no explicit type or
dimension arguments.

## Elementwise Arithmetic And Reductions

A dot product combines two operations with different result shapes. Elementwise multiplication
retains all three coordinates, while a reduction combines them into a scalar. Here are the
products, then the sum and mean of the original vector:

```lean (name := apiMul)
-- Keep one product for each coordinate instead of reducing
-- the vector.
#eval (apiV1 * apiV2)
```
```leanOutput apiMul (whitespace := lax)
[4.000000, 10.000000, 18.000000]
```

```lean (name := apiSum)
-- Sum the original first vector, not the product vector
-- computed above.
#eval Tensor.sum apiV1
```
```leanOutput apiSum (whitespace := lax)
6.000000
```

```lean (name := apiMean)
-- Average the three entries of apiV1 using their total
-- element count.
#eval Tensor.mean apiV1 [⟨0, by decide⟩] false
```
```leanOutput apiMean (whitespace := lax)
2.000000
```

`Tensor.mean` names the axes it averages by `Fin`: `⟨0, by decide⟩` is axis zero of the rank-one
shape, in range by construction, and `false` drops that axis from the result (PyTorch's
`keepdim`).

PyTorch, same three lines:

```
elementwise: [4.0, 10.0, 18.0]
sum, mean  : 6.0 2.0
```

Elementwise operations require identical shapes. To combine a reduced vector with a matrix,
the program must explicitly expand it along the intended axis.

Summing `[4, 10, 18]` would instead give the dot product, `32`. Keep track of which tensor each
reduction consumes (the {ref "bugzoo-catalog"}[BugZoo chapter] examines bugs caused by choosing
that axis incorrectly).

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

From the type of `matmul` you can read which dimension is contracted and which dimensions remain:

```lean (name := apiMatmulSig)
-- Only the shared dimension n is contracted, so m and p
-- become the result dimensions.
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

That shared `n` is the contraction. A dimension typo is reported where the product is written, not
where the resulting tensor is first used, and the instance list says exactly what the element type
must support: addition, multiplication, and a zero. No ordering, no division, no transcendental
functions. That's why the same operation works over the rationals in the certificate examples.

For `apiM1.matmul apiM2`, the result has two rows and two columns. Its top-left entry, `22`, is
`1 * 1 + 2 * 3 + 3 * 5`: one row of the left matrix paired with one column of the right matrix.
Both vector products make the direction equally visible. Multiplying by a three-entry vector of
ones sums each row to `[6, 15]`, while a two-entry vector of ones on the left sums the rows
together to `[5, 7, 9]`. Whichever dimension survives tells us which sums we asked for.

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
structure while retaining the values. Subtracting two and applying ReLU changes the values while
retaining the length, and concatenation appends a second copy and changes the length:

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
  (apiV1 - Tensor.full [3] 2.0)
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
drop or invent elements is rejected at elaboration. `concat` returns shape `[6]` here, computed
from the inputs, not supplied by the caller.

Reshaping gives a column containing `1`, `2`, and `3`, without transposing a preexisting matrix or
changing those values. By contrast, the activation example computes `[-1, 0, 1]` before ReLU clamps
the first entry, producing `[0, 0, 1]`. Concatenation preserves both copies in order. These three
results illustrate separate questions to ask of an operation: which coordinates exist, which values
occupy them, and whether any values are duplicated or discarded.

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

Here the two weight rows select the first and second input coordinates. Adding the bias then shifts
every selected pair by `[0.5, -0.5]`. That gives `[1.5, -0.5]` for the first basis vector and
`[0.5, -0.5]` for the third, whose selected coordinates are both zero. And the final input row has
both selected coordinates equal to one, producing `[1.5, 0.5]`. All four rows use one weight matrix
and one bias: batching doesn't allocate four independent affine maps.

That input is a batch of four rows, but the signature extends the same operation to any leading
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

Any shape ending in `inputWidth` is accepted as input, and the output replaces that last dimension.
So the same declaration handles an unbatched vector, a batch of rows, and a batch of sequences of
rows, with the leading dimensions carried along by the `EndsWith` instance, not by a runtime
reshape. Instance search checks the trailing feature width while retaining the prefix shape
(runtime implementations may still specialize how they execute batched operations).

# Naming And Dot Syntax

Dot syntax is used when the value before the dot is the natural subject of the operation:
`tensor.reshape`, `module.forward`, `trainer.train`, `trained.predict`, and `result.printSummary`.
That's why the reshape above is written `apiV1.reshape [3, 1]`. A choice such as a device or a loss
is a constructor of its own type. Where Lean expects that type, a leading dot names it, as in
`device := .cpu`, and opening the type's namespace allows the plain name: devices `cpu` and `cuda`
in `Backend.Device`, arithmetic `native`, `ieee` and `complex` in `Arithmetic`, losses `mse` and
`oneHotCrossEntropy` in `Trainer.Objective`, activations such as `relu` in `Activation.Kind`, the
verifier's `ibp` and `crown` in `Verification.Algorithm`, and its `outputBounds` and `topLabel` in
`Verification.Property`. This chapter opens `Arithmetic` and `Trainer.Objective`. The verifier's
choices are written with a leading dot, as in `(algorithm := .ibp)` and
`(property := .topLabel 0)`. A choice whose word is generic is written with its type instead, such
as `nn.Mode.eval` or `Loss.Reduction.sum`, so that it can't capture a variable of the same name.

Ordinary application code doesn't construct shapes through recursive representation constructors: it
writes `[]`, `[width]`, or `[batch, width]`. Nor does it inspect public results through
positional tuple projections. Multi-value transforms use tuple destructuring, while training
progress reads as `result.loss.before` and `result.loss.after`. Most training code uses the model's
typed state and trainer interfaces. A model's state is a `TensorPack α shapes`, the collection of
tensors with different shapes that lower-level graph code also uses.

These conventions keep operations such as prediction and fields such as reported loss visible
at the call site without requiring callers to unpack the runtime representation.

# A Complete Training Program

XOR supplies four input rows and a scalar target for each. Our program trains a two-layer network
with mean-squared error and Adam, then evaluates it at a point outside those four rows. Save the
declarations and `main` in `Scratch.lean` at the repository root (with the imports and `open` shown
above).

Since the model, data, and trainer are ordinary declarations, we can elaborate them right here and
read their types before running anything:

```lean (name := apiModel)
-- Keep the four XOR row pairs separate. The training loop
-- will group their gradients.
def apiModel :
    nn.Builder (nn.Sequential [2] [1]) :=
  nn.Sequential![
    nn.linear 2 8,
    nn.op .relu,
    nn.linear 8 1
  ]

def apiXs : Tensor Float [4, 2] :=
  [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]

def apiYs : Tensor Float [4, 1] :=
  [[0.0], [1.0], [1.0], [0.0]]

def apiData := Data.fromTensors apiXs apiYs

def apiTrainer :=
  Trainer.create apiModel
    { objective := mse
      optimizer := optim.adam { lr := 0.03 }
      arithmetic := native
      seed := 2026 }

#check apiData
```
```leanOutput apiModel (whitespace := lax)
apiData : Trainer.Dataset [2] [1]
```

As with PyTorch's `TensorDataset`, TorchLean infers `Trainer.Dataset [2] [1]` because `apiXs` has
batched shape `[4, 2]` and `apiYs` has batched shape `[4, 1]`. Those two shape parameters describe
one input sample and one target sample: the dataset constructor removes the leading batch dimension
(its size only has to agree between the two arguments).

Materializing this dataset will therefore yield four paired items, not one item containing all four
rows. And the model still consumes a `[2]` input and returns a `[1]` prediction. In the `main`
below, `samplesPerStep := 4` gathers the four item gradients at one parameter point before Adam
updates it, and that grouping belongs to the training loop: it hasn't changed the model's shape
into `[4, 2] → [4, 1]`. You can see in the constructor signature where the shared row count is
required:

```lean (name := apiFromTensors)
-- The shared leading n is removed from the shape of each
-- paired dataset item.
#check @Data.fromTensors
```
```leanOutput apiFromTensors (whitespace := lax)
@Data.fromTensors : {n : ℕ} →
  {σ τ : Shape} →
    Tensor Float (Shape.dim n σ) →
      Tensor Float (Shape.dim n τ) → Trainer.Dataset σ τ
```

That shared `n` is why a dataset with four inputs and three targets can't be built. Add a `main`
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

Lean's structure-instance notation is whitespace sensitive. In the `Trainer.create` configuration,
keep the first field aligned with the remaining fields as shown, because misaligning `objective :=`
can produce `unexpected identifier; expected '}'` several lines later.

Check the definitions without starting the training run:

```terminal
# Elaborate the declarations and main without invoking main.
scripts/lake.sh env lean Scratch.lean
```

Lean stays silent here apart from diagnostics. To execute the `main` in the file:

```terminal
# Run main after elaboration to execute the requested
# updates and held-out prediction.
scripts/lake.sh env lean --run Scratch.lean
```

With `seed := 2026`, the recorded CPU run ended with:

```terminal +output
dataset size = 4
mean_loss(before training) = 0.865279
step 5: loss=0.359841
step 10: loss=0.279992
step 15: loss=0.200866
step 20: loss=0.172833
mean_loss(after training) = 0.173368
steps=20 arithmetic=native scalar=Float32 loss=0.865279 -> 0.173368
prediction=[0.277499]
```

After twenty steps, the loss has fallen substantially, but the network hasn't fit the four XOR
targets. Nor is the loss monotone: the last update raises it from `0.172833`, logged at step 20, to
the final mean `0.173368`. XOR supplies no target for $`(0.25, -0.75)`, so the prediction there,
`0.277499`, needs a separate interpretation. And the summary line records what the numbers mean:
`arithmetic=native scalar=Float32`, so this is Lean's native binary32 arithmetic, not the host
binary64 in which the dataset literals were written.

Because each update in this program accumulates all four XOR rows, its step loss is the mean over
those rows at the current parameters. This differs from a single-sample training log, where adjacent
steps may report different samples. Even a full-dataset loss need not decrease at every Adam update:
the update uses its stored moments and configured rate. The lower final mean records progress over
the run, while the intervening entries show the particular trajectory.

Change `steps := 400` and `logEvery := 100`, and the same program settles at a mean loss of
`0.166667` without fitting the four training rows:

```terminal +output
dataset size = 4
mean_loss(before training) = 0.865279
step 100: loss=0.166668
step 200: loss=0.166667
step 300: loss=0.166667
step 400: loss=0.166667
mean_loss(after training) = 0.166667
steps=400 arithmetic=native scalar=Float32 loss=0.865279 -> 0.166667
prediction=[0.285693]
```

In PyTorch, the corresponding architecture and optimizer,
`Sequential(Linear(2,8), ReLU(), Linear(8,1))` with `MSELoss` and `Adam(lr=0.03)` for 400
full-batch steps:

```terminal +output
step 0: loss=0.402008
step 100: loss=0.000006
step 200: loss=0.000000
step 300: loss=0.000000
final loss=0.000000
prediction at [0.25,-0.75]: [0.9572102427482605]
```

PyTorch drives the training loss below the displayed six-decimal resolution on the four rows, so
this architecture can fit them. The TorchLean run from seed 2026 instead logs `0.166667` at steps
200, 300 and 400. The initial parameters differ, as the initial losses `0.865279` and `0.402008`
show, and from its initial parameters the TorchLean run reaches a plateau rather than a fit. Away
from the training data the two runs also give different predictions: `0.285693` against `0.957210`
at the same unseen point. Two runs that both fit the four rows can disagree there as well, because
the training rows don't determine a unique function away from those points. So a small final
training loss alone doesn't establish agreement on unseen inputs. A different kind of evidence,
developed in the {ref "certificates"}[verification chapters], consists of statements that hold over
whole input regions under explicit hypotheses.

The `--run` path is convenient for a one-file experiment, but a maintained command should still
receive a Lake executable target so it can be built and invoked by name.

# Type Inspection In VS Code

Place the cursor on `apiModel`. Lean's infoview shows the builder type, which is also what `#check`
prints:

```lean (name := apiModelType)
-- The boundary shapes belong to the builder even before its
-- seed has been supplied.
#check apiModel
```
```leanOutput apiModelType (whitespace := lax)
apiModel : nn.Builder (nn.Sequential [2] [1])
```

Here `[2]` and `[1]` are the shapes of one input and one output sample, and the layer list is
checked against the annotation on `apiModel`. Changing the final layer from `nn.linear 8 1` to
`nn.linear 7 1` breaks the boundary between the hidden representation and the output layer:

```lean +error -keep (name := apiTypo)
-- The hidden layer produces eight entries, so a final layer
-- expecting seven cannot compose.
def apiTypo :
    nn.Builder (nn.Sequential [2] [1]) :=
  nn.Sequential![
    nn.linear 2 8,
    nn.op .relu,
    nn.linear 7 1
  ]
```
```leanOutput apiTypo (whitespace := lax)
nn.Sequential!: layer 2 expects input shape [7], but layer 1 outputs [8].
Change layer 2's input shape or insert a layer that converts [8] to [7].
```

The final layer fixes ReLU's shape at `[7]`, which can't follow the first layer's `[8]` output. So
the error points to layer 2, the ReLU. Lean finds this mismatch from the model definition alone,
before a forward pass, data loading, or parameter initialization.

To see where the shapes in the error come from, look at the layer constructor's signature:

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
        (nn.Sequential (batchShape ++ [inputWidth])
          (batchShape ++ [outputWidth]))
```

Both the batch shape and the layer configuration are optional parameters with defaults, which is why
`nn.linear 2 8` is enough in the common case and why an unbatched layer has boundary shapes
`[inputWidth]` and `[outputWidth]`.

Place the cursor on `trained.predict`. Its input and output are trainer-facing host-`Float` tensors
with the model's checked shapes. Conversion to the arithmetic semantics selected by the training
configuration happens in the retained runner.

# Builder, Trainer, And Training Result

These three values have different lifetimes, and keeping them apart is what makes the same
architecture reusable:

```lean (name := apiTrainerNew)
-- ToModel converts a supported model representation into
-- the trainer with these boundary
-- shapes.
#check @Trainer.create
```
```leanOutput apiTrainerNew (whitespace := lax)
@Trainer.create : {Model : Type u_1} →
  {σ τ : Shape} →
    [Trainer.ToModel Model σ τ] →
      Model → optParam (Trainer.Config σ τ) { } → Trainer σ τ
```

A `nn.Builder (nn.Sequential input output)` describes architecture and seeded initialization.
`Trainer.create model config` materializes the builder at the selected seed and attaches objective,
optimizer, and runtime choices, but it hasn't consumed data. `trainer.train data options` executes
updates and returns a `Trainer.Result` with the final parameters and prediction operations.

Thanks to the `Trainer.ToModel` instance, `Trainer.create` accepts a builder, an already-built model
definition, or anything else that can present itself as a model with those shapes, without a
conversion call at the use site. Splitting the three stages permits the same architecture to be
initialized with several seeds, trained on several datasets, or interpreted by another runtime
without redefining its layers. It also means a failed run is diagnosable: the builder either
typechecks or it doesn't, independently of whether any data was available.

# Persistent And Per-Call Configuration

Persistent choices can be expressed as `Trainer.RunConfig`:

```
-- Keep the optimizer fixed while selecting arithmetic or
-- the device profile.
def nativeCpu : Trainer.RunConfig :=
  { optimizer := optim.adam { lr := 0.03 }
    arithmetic := native }

def ieeeCpu : Trainer.RunConfig :=
  { nativeCpu with
    arithmetic := ieee
    device := .cpu }

def configuredTrainer :=
  Trainer.create model { ieeeCpu with objective := mse, seed := 2026 }
```

`RunConfig` adds the optimizer and arithmetic semantics to the `Session.Config` fields: device,
seed, and backend profile. See
{ref "backend-selection"}[Backend Selection] for the
profile's provider, VJP, and evidence rules.

Optimizers are opaque values built from a record of hyperparameters. Their descriptions display
the constructor and configuration:

```lean (name := apiDescribe)
-- The text includes the Adam defaults that were not
-- repeated in the constructor.
#eval (optim.adam { lr := 0.03 }).describe
```
```leanOutput apiDescribe (whitespace := lax)
"optim.adam { lr := 2.9999999999999999e-2,
  beta1 := 9.0000000000000002e-1,
  beta2 := 9.9900000000000000e-1,
  epsilon := 1.0000000000000000e-8 }"
```

Those moment coefficients match PyTorch's Adam defaults, so a script ported from
`torch.optim.Adam(params, lr=0.03)` need not restate them. FloatLib supplies scientific decimal
text for the stored binary64 values. Before returning that text, the formatter checks that Lean's
literal decoder reconstructs the same bits (if the check fails, it emits a `Float.ofBits`
expression). That check protects small stabilizers and coefficients near one when their
descriptions are copied.

You can choose among `optim.sgd` (with optional heavy-ball momentum {Informal.citep polyak1964}[]),
`optim.adaGrad`, `optim.rmsProp`, `optim.adam`, `optim.adamW` (decoupled weight decay
{Informal.citep adamw2019}[]), and `optim.adaDelta`. At this layer they're configuration records,
while the update rules themselves live in the runtime and are stated as specifications in the
training chapter.

Executable frontends start from already-parsed common runtime flags with
`{ runtime with optimizer := ... }`. Command-line applications use
`Trainer.RunConfig.parseCommandLine` to parse these flags into a training runtime configuration.

Per-call `TrainOptions` controls step count, sample grouping, logging cadence, and artifact fields.
`loadCheckpoint?` and `saveCheckpoint?` load or save model state around ordinary training, for
mean-squared error, cross entropy, and custom losses. They do not snapshot optimizer moments or
loader state. Alternating two-model training rejects these single-path options because one path
cannot represent both model states. `{ run with toConfig := runtime }` replaces the runtime fields
of a configuration from a `Session.Config`, which is how a study varies the device or device grid
without touching the optimizer or the objective.

# Runtime Arithmetic

`Arithmetic` isn't a storage dtype. It chooses the arithmetic for an executable run.
`native` selects Lean's native binary32 `Float32`, `ieee` selects FloatLib binary32, and `complex`
selects complex numbers over FloatLib binary32. That choice changes the meaning of arithmetic
operations instead of attaching a label to an untyped buffer. This is why the scratch run above
reported `arithmetic=native scalar=Float32` in its summary line: the run states its own semantics.

Trainer-facing datasets are commonly authored as `Tensor Float ...`, where `Float` is Lean's host
binary64 type. Each dataset materializes its samples in the scalar selected by the trainer. This is
an input boundary, and it doesn't imply that training itself uses binary64.

For a different precision, choose a FloatLib binary format directly in typed tensors, state and
graphs. On the CPU path, `nn.sgdStep` or a supervised `trainer.openTyped (α := α)` session supports
binary128 and custom valid binary widths. Typed sessions preserve samples, state, losses,
predictions, and checkpoint encodings. Supply typed `initialState?` when the parameters need digits
beyond the model's stored seeded `Float` values. In the ordinary trainer, the `ieee` setting
selects binary32, not an arbitrary width.

Proofs choose `ℝ` or `TorchLean.FP32` directly (neither is a runtime mode, because both are
noncomputable). The high-level trainer accepts the real runtime modes. For a real loss on complex
parameters, use `autograd.complex.grad` and `nn.sgdStep` with explicit complex state. That gradient
contains both real-coordinate derivatives, and predictions and `Checkpoint.State` preserve both
components. The reference gradient algorithm takes two forward passes per complex parameter entry
(requesting the objective value adds one ordinary evaluation). It doesn't turn the trainer's
real-data interface into a complex-data interface.

Try `scripts/lake.sh exe torchlean complex_regression` for a complete complex binary32 example.
It fits a weight and bias against a real squared-residual loss and prints a held-out prediction.

Each direct tensor calculation above uses the `Float` written in its annotations. None of them is
silently redirected through a trainer's `native` setting. The same is true of the explicit
derivative example below (its input is a `Tensor Float [3]`). Conversion to native `Float32`
belongs to the high-level training boundary. Keeping that boundary visible avoids attributing a
binary64 calculation to binary32 merely because both print six decimal places.

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
Data.fromCSV
Data.batch
```

A model and its dataset must agree on their input and target dimension lists. A file loader checks
runtime dimensions while materializing the typed samples.

A true tensor minibatch changes shapes to `[batch,...]`. `Data.batch` stores each fixed-size
minibatch as one dataset item, so `TrainOptions.samplesPerStep := 1` executes one vectorized
forward/backward pass per update. On an unbatched dataset, the same option counts individual
samples. Values above one always mean gradient accumulation across several dataset items. Both
forms are developed in the data chapter.

# Explicit Differentiation

Derivatives are taken of a program, and that program is a differentiable tensor function, not an
arbitrary Lean function:

```lean (name := apiEnergy)
-- square retains three coordinates, and mean produces the
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

The tuple is ordered as gradient first, value second. Its first component has shape `[3]`, with one
derivative for each input coordinate, and its second has shape `[]`, because the mean has reduced
all three squared entries to one scalar tensor. In particular, the final gradient coordinate is
exactly `2` in the displayed result because that input was `3`. But the other two coordinates and
the loss require rounding when represented as `Float`. PyTorch's `backward` on the same expression:

```
value: 4.666666507720947 grad: [0.6666666865348816, 1.3333333730697632, 2.0]
```

Same numbers, to the last displayed digit of TorchLean's six-decimal formatting. And the shipped
quickstart prints the same result from a compiled binary, not from elaboration:

```terminal +output
== Differentiate a tensor function ==
mean(x^2) = 4.666667
d/dx       = [0.666667, 1.333333, 2.000000]
```

All the other transforms follow the same functional interface (familiar from JAX
{Informal.citep jax2018}[]): a transform consumes a function and returns a function's results
instead of mutating tensors. `jacrev` builds the Jacobian by reverse mode, one row per output:

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
diagonal, not small noise. `torch.func.hessian` on the same function returns
`[[0.6666666865348816, 0.0, 0.0], [0.0, 0.6666666865348816, 0.0], [0.0, 0.0, 0.6666666865348816]]`,
the same matrix at the displayed precision. Composing forward and reverse mode this way (instead of
differentiating twice in reverse) is the standard choice for a small input dimension
{Informal.citep baydin2018}[].

The full set is `autograd.grad`, `autograd.vjp`, `autograd.jacfwd`, `autograd.jacrev`, and
`autograd.hessian` for tensor functions, and `autograd.model.grad`, `autograd.model.vjp`,
`autograd.model.jvp`, and `autograd.model.hvp` for a checked model.

Derivatives are returned as values. Parameter derivatives have the same dependent tensor-pack
structure as the parameters, and there's no mutable `.grad` field on `Tensor` values. That's a
consequence of the host language, not a stylistic preference: Lean values are immutable, and its
compiler recovers in-place updates through reference counting instead
{Informal.citep immutablebeans2019}[] . Each query returns a fresh gradient result without
accumulating into a persistent tensor field. Callers can still accidentally reuse an old returned
value or explicitly combine the wrong packs. There's no `zero_grad` in this API because there's
nothing to zero. A model `vjp` returns a state gradient and input gradient from one reverse pass,
and callers destructure the pair.

In the model quickstart you can see what a parameter gradient looks like when the parameters are a
pack of shaped tensors, not a single vector:

```terminal +output
== Differentiate a model loss ==
loss     = 0.769887
gradient = [[1, 2]: [[-0.877432, 1.754865]], [1]: [-1.754865]]
```

Each entry is labeled by its shape (`[1, 2]` for the weight and `[1]` for the bias), so the printed
gradient has the same structure as the parameters it differentiates.

In the weight entry, the two numbers are derivatives with respect to the two separate weights,
while the bias entry has only one coordinate. Printed above the pack, the scalar loss is the
objective value at the same model state at which those derivatives were taken. That's enough to
feed an optimizer or compare a derivative calculation, but it isn't itself an update: a gradient
result leaves the choice of learning rate, optimizer memory, and parameter replacement to its
caller.

# Functional Tensor Operations

Use `nn.functional` when constructing a differentiable tensor program, as `apiEnergy` did above. An
arbitrary Lean function over `TorchLean.Tensor` is useful for specifications, and the `Tensor`
operations from the first section of this chapter are exactly that, but a plain function doesn't
carry runtime graph and derivative behavior. `nn.functional.square` and `Tensor.square` compute the
same mathematics, while only the first one can be differentiated by the transforms above.

This is the usual distinction for an embedded differentiable language: operations must register the
semantics that execution and automatic differentiation need. Tracing frameworks reach the same
place from the other direction, by recording a graph while ordinary code runs
{Informal.citep fx2022}[] . Here the graph is what the `autograd.Function` monad builds, so
typechecking establishes the operation interface and shapes, though a selected transform can still
reject operations whose derivative lowering it doesn't support.

# Classical Models

TorchLean's k-nearest-neighbor classifier stores labeled points and votes among nearby ones (there
are no layers to initialize). Its definition lives in `TorchLean/NN/Models`, alongside other
classical models that use the same shape-indexed tensors.

The chapter's opening imports don't bring in these models, so a file that uses them adds one
focused import. Here is a complete executable k-nearest-neighbor classifier: the model stores its
labeled samples, while evaluation computes distances and applies a deterministic majority vote
(ties are broken deterministically by neighbor order).

```lean (name := apiKnn)
-- Three neighbors vote using the four stored labeled
-- points. There is no optimizer step.
/-- Four labeled points, classified by three neighbors. -/
def apiPoint (x y : Float) : Tensor Float [2] := [x, y]

def apiKnn : KNN Float String 2 :=
  { k := 3
    dataset := #[
      (apiPoint 0.0 0.0, "blue"),
      (apiPoint 0.0 1.0, "blue"),
      (apiPoint 3.0 3.0, "orange"),
      (apiPoint 3.0 4.0, "orange")] }

open KNN in
#eval classify Float String 2 apiKnn (apiPoint 0.2 0.1)
```
```leanOutput apiKnn (whitespace := lax)
"blue"
```

Its dataset is an `Array` of labeled tensors, and `k := 3` selects three neighbors. A point near
the far cluster classifies the other way:

```lean (name := apiKnnOrange)
-- This query lies near the two orange points rather than
-- the two blue points.
open KNN in
#eval classify Float String 2 apiKnn (apiPoint 2.9 3.5)
```
```leanOutput apiKnnOrange (whitespace := lax)
"orange"
```

With $`k = 3` and only four stored points, every query mixes clusters, and the confidence is the
vote share, not a probability:

```lean (name := apiKnnConf)
-- Return the winning label together with its fraction of
-- the three selected neighbor votes.
open KNN in
#eval classifyWithConfidence Float String 2
  apiKnn (apiPoint 0.2 0.1)
```
```leanOutput apiKnnConf (whitespace := lax)
("blue", 0.666667)
```

Two of the three nearest neighbors of $`(0.2, 0.1)` are blue, which makes the ideal vote share
$`2/3`, and the returned `Float` is rounded. This vote share is not a calibrated probability.

What the classifier needs from its scalars and labels is written in its type:

```lean (name := apiClassifySig)
-- Ordering ranks distances, label equality tallies votes,
-- and Inhabited handles an empty vote.
open KNN in
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

With the decidable ordering the classifier can rank distances, and with `BEq` and `Hashable` it can
tally labels. `Inhabited` supplies a default label when there are no neighbors to vote. Callers
that require a vote-supported prediction must therefore ensure that neighbors exist. Other
families, including the GMM and HMM likelihood operations below, expose invalid inputs through
optional results.

We deliberately expose different amounts of fitting and inference machinery across the exported
families:

:::table +header
*
  * Family
  * Available operations
  * Current boundary
*
  * kNN
  * nearest neighbors, classification, regression, and confidence
  * lazy stored-data model, with no learned index or metric
*
  * random forest
  * symbolic-tree aggregation plus numeric regression fitting and Gini classification-tree fitting
  * seeded sampling with replacement for rows and optional feature subsampling per split
*
  * naive Bayes
  * multinomial string-feature counting, log scores, prediction, and negative log likelihood
  * specialized to bags of `String` features and labels, and fitting is counting, not gradient
    training
*
  * SVM
  * linear decisions, hinge objective, VJP, gradient-descent fit, prediction, and kernel functions
  * the fitter is a linear primal baseline, and exported kernels do not constitute a kernel-SVM
    solver
*
  * GMM
  * component log densities, responsibilities, VJP, log likelihood, initialization, and EM
  * optional evaluation checks weights and a scalar-arithmetic covariance criterion, but a rounded
    check is not a proof of real positive definiteness
*
  * PCA
  * projection, inverse, VJP, reconstruction statistics, and a leading-component fit
  * fitting approximates one component with deterministic multistart power iteration and is not a
    full SVD-based PCA fit
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
  * bounded, deterministic tree routines, with no exported boosted classification ensemble
*
  * HMM
  * scaled and unscaled forward passes, batching, likelihood, initialization, and Baum-Welch
    updates
  * finite discrete observations (probability normalization is an input invariant, not a type
    invariant)
:::

These definitions execute directly when their scalar `Context` is executable (as `Float` is in the
example). They're pure tensor/reference algorithms, not automatic `Trainer.create`, typed graph,
LibTorch, or CUDA routes. Shape indices rule out dimensional mismatches, but they don't by
themselves prove statistical assumptions, optimizer convergence, or a family-wide correctness
claim. Consult each declaration's hypotheses and result type, especially for the `Option`-returning
GMM and HMM likelihood operations.

# Additional Imports

No single import covers a whole library. A file that combines several libraries, such as model code
with tensor proofs and verification, imports a module from each:

```
-- Model code, tensor proofs and verification
-- lowering come from three libraries.
import TorchLean.NN.Trainer.Constructor
import TorchLean.Tensor.Algebra
import TorchLean.Verify.Lowering
```

Prefer the narrowest stable import that expresses the file's responsibility. For a concrete case,
take the kNN section above: the chapter's opening imports don't carry the classical models, so the
example adds `import TorchLean.NN.Models.KNN` and nothing else. A numerical theorem should not
import the entire executable example catalog merely for convenience, and a training script should
not depend on the internal tape API.

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
  * `IR.Graph`
  * explicit operation data
*
  * backend audit
  * provider and evidence choices
*
  * theorem
  * exactly one Lean proposition under hypotheses
*
  * certificate
  * artifact accepted under a checker's conditions (any soundness theorem has its own hypotheses)
:::

Our XOR transcripts record two particular training runs. A theorem about either resulting model
would need to state a property of its predictions and the inputs on which that property holds.
Their small training losses alone don't supply that statement.

# Declaration Search

Use the generated API search:

```
/docs/search.html
```

On the published project site the prefix is `/TorchLean/docs/search.html`, while a root-mounted
local preview uses `/docs/search.html` . Website navigation supplies the configured prefix. For
source search:

```terminal
# Search the source declarations for derivative definitions
# and soundness theorems.
rg -n "def diff|theorem .*sound" TorchLean
```

The API reference gives the exact declaration, and the surrounding chapters explain why and when to
use it. Source remains authoritative when a lower-level contract matters. If a name in this book
and a name in the source disagree, the source is right and the book has a bug. Named Lean blocks
are elaborated against the library, but prose and unmarked reproduction snippets still need review
when APIs change.

# CLI Examples And Related Chapters

Run:

```terminal
# These commands exercise public tensor, derivative, and
# training interfaces from compiled
# examples.
scripts/lake.sh exe torchlean --help
scripts/lake.sh exe torchlean quickstart_tensors
scripts/lake.sh exe torchlean quickstart_autograd
scripts/lake.sh exe torchlean quickstart_mlp --steps 20
```

The tensor quickstart uses one interface with four scalar types, and its output lets us compare
their values without changing the tensor representation:

```terminal +output
== Quickstart: tensor basics ==
Rank-zero tensor: 0.500000
Float tensor:    [0.100000, 0.200000, 0.300000, 0.400000]
Rational tensor: [(1 : Rat)/10, (1 : Rat)/5, (3 : Rat)/10, (2 : Rat)/5]
Integer tensor:  [1, 2, 3, 4]
Float32 cast:    [0.100000, 0.200000, 0.300000, 0.400000]
```

Side by side, the rational and floating-point rows distinguish stored values from decimal
formatting. `0.1` isn't representable in binary floating point, so the `Float` row prints a rounded
value while the `Rat` row holds exactly $`1/10`. For precisely this reason, the certificate
chapters run their geometry over `Rat`, and it's the same `Tensor` type in both rows.

Lean's
[source-file and module
reference](https://lean-lang.org/doc/reference/latest/Source-Files-and-Modules/) explains how these
imports determine the environment in which a file elaborates. The runtime chapters follow the
model from its trainer configuration to the operations that execute it.
