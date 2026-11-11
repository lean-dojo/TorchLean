import VersoManual
import TorchLean.NN.Kernel.Function
import TorchLean.NN.Kernel.Graph
import TorchLean.NN.Sequential.Training
import TorchLean.NN.Trainer.Session
import TorchLean.NN.Autograd.Function
import TorchLean.NN.Autograd.Differential
import TorchLean.NN.Autograd.Complex
import TorchLean.NN.Tactic.Autograd
import TorchLeanBlueprint.Roles
import TorchLean.NN.Session.Random
import TorchLean.CUDA.Tree.Elab
import TorchLean.CUDA.Tree.Run

open Verso.Genre Manual
open Verso.Genre.Manual.InlineLean
open TorchLean Backend.Device
open Lean.Elab.Tactic.GuardMsgs.WhitespaceMode (lax)

#doc (Manual) "Custom Tensor Computations" =>
%%%
tag := "custom-computations"
file := "Custom-Tensor-Computations"
%%%

```lean -show
namespace Tutorial.CustomComputations
```

Let's write a function and run it on a tensor. We'll start in Lean's Infoview with a square, then
try parameters, row sums and repeated updates, and we'll also ask for a gradient without rewriting
the function.

Sometimes the calculation we want isn't a named LibTorch operation: we might want to change an
activation, combine several elementwise steps, or read a small neighborhood of a tensor. We should
be able to write that calculation in Lean without maintaining a second implementation in a CUDA
file.

```lean (name := customSquare)
def square := fun (x : Float32) => x * x

def input : Tensor Float32 [3] :=
  Tensor.ofFn fun i => Float32.ofNat (i.val + 1)

#eval square.run input
```

```leanOutput customSquare (whitespace := lax)
[1.000000, 4.000000, 9.000000]
```

CPU is the default. Our result is a `Tensor Float32 [3]` like the input, and `#eval` prints it as
is, so we don't need to convert our data to a list or unpack a native buffer.

# Choosing a device

Pass the device when you run the function:

```lean
def onCpu : IO (Tensor Float32 [3]) :=
  square.run input (device := cpu)

def onGpu : IO (Tensor Float32 [3]) :=
  square.run input (device := cuda)
```

These definitions are checked when we build the guide, but we evaluate only the CPU example here,
because running `onGpu` needs a CUDA-enabled LibTorch build and a visible GPU, as described in
{ref "gpu-and-cuda"}[GPU and CUDA]. A CPU-only scalar type or format uses CPU with a diagnostic,
without converting its values to a narrower type, but an unavailable GPU (or an unsupported
function for a GPU-capable type) produces an IO error.

CPU applies the original Lean function to each entry. For GPU, the frontend recognizes supported
scalar arithmetic, comparisons, local bindings and conditionals during elaboration, and it builds a
typed expression with a proof that its evaluation agrees with the function we wrote. At runtime the
native bridge renders CUDA, compiles it with NVRTC (NVIDIA's runtime CUDA compiler) and executes
it.

On GPU, ordinary model operations such as matrix multiplication keep using LibTorch's ATen
primitives, while the portable CPU path evaluates the corresponding Lean runtime operations. This
API lets us supply our own scalar calculation without writing a separate CUDA source file. For
differentiation, we can record supported arithmetic at the call site (as in the next example), but
arbitrary indexed custom bodies don't acquire backward rules from source equivalence.

# Keeping the calculation on the tape

Let's keep the same square definition and ask for its gradient:

```lean (name := customAutograd)
#eval do
  let result ← square.run input (grad := true)
  IO.println result.value
  IO.println (← result.backward)
```

```leanOutput customAutograd (whitespace := lax)
[1.000000, 4.000000, 9.000000]
[2.000000, 4.000000, 6.000000]
```

We can pass the whole tensor to a function too. Here we sum its squared entries:

```lean (name := customEnergy)
def energy (x : Tensor Float32 [3]) : Float32 :=
  Tensor.sum (x * x)

#eval do
  let result ← energy.run input (grad := true)
  IO.println result.value
  IO.println (← result.backward)
```

```leanOutput customEnergy (whitespace := lax)
14.000000
[2.000000, 4.000000, 6.000000]
```

Concrete element types supply storage automatically. Generic definitions state their element
requirements, but calls infer them from the input:

```lean (name := customTensorSquare)
def tensorSquare {α : Type} [Storage α] [Mul α]
    {s : Shape} (x : Tensor α s) : Tensor α s := x * x

#eval tensorSquare.run input
```

```leanOutput customTensorSquare (whitespace := lax)
[1.000000, 4.000000, 9.000000]
```

Recording uses the existing arithmetic, tensor sums, square, ReLU, sigmoid, tanh and rank-two
matrix products, along with their VJP rules (a VJP takes an output gradient and propagates it back
to the inputs). For each recording the frontend adds a kernel-checked theorem: the recorded
program, run by the pure interpreter, computes our function. Each operation is recorded only when
it computes what the recorded operator computes in the element type's context, instance included,
so an `x * y` whose `Mul` instance means something else is an error rather than a wrong gradient.
The theorem covers the pure interpreter, not the eager tape or a GPU kernel. There's no separate
custom-gradient tape. Captured tensors are constants, and backward differentiates the supplied
input. Division and direct negation use that same interface. Scalar input-dependent conditionals
record only the chosen branch, while fixed-parameter scalar recurrences record each step (and the
count must be independent of differentiable inputs).

Backward consumes the recording, even on failure. Its default seed is all ones, which gives the
derivative of the sum of output entries. Pass `result.backward seed` to supply different output
gradients, or call `result.close` when backward isn't needed. A function that can't be recorded is
an elaboration error when we write `grad := true`, and an IO error when `grad` is known only at run
time. CPU forward-only calls evaluate the original function.

With recording, the element type can be a parameter too. On GPU, the tape retains binary32 for
`Float32`, binary64 for `Float`, and complete words for configured binary arithmetic (including
binary128). Configured recording supports addition, subtraction, multiplication, division, negation
and the structural operations used by scalar traversal, while other native model operations and
native optimizer checkpoints remain limited to their native dtypes. An unsupported operation asks
us to use CPU.

We can also check the underlying real derivative with the existing proof tactic:

```lean
example (x : ℝ) :
    HasDerivAt (fun x : ℝ => x * x) (2 * x) x := by
  autograd
```

This is a mathematical derivative statement, not a claim that finite-precision arithmetic is a
smooth real function or that foreign GPU instructions have been verified by Lean.

# A conditional activation

Let's change the calculation. We'll discard negative entries and square the others:

```lean (name := customPositiveSquare)
def positiveSquare := fun (x : Float32) =>
  if x < 0 then 0 else x * x

def signedInput : Tensor Float32 [3] := [-2, 1, 3]

#eval positiveSquare.run signedInput
```

```leanOutput customPositiveSquare (whitespace := lax)
[0.000000, 1.000000, 9.000000]
```

Here the conditional is part of our function, not a separate tensor operation we have to name, and
the GPU frontend supports this form too:

```lean
def activationOnGpu : IO (Tensor Float32 [3]) :=
  positiveSquare.run signedInput (device := cuda)
```

Only the chosen branch is evaluated, which matters when the other branch contains a division or
another operation we don't want to execute for that input.

We can ask for the activation's gradient without rewriting the conditional:

```lean (name := customBranchGradient)
#eval do
  let result ← positiveSquare.run signedInput (grad := true)
  IO.println result.value
  IO.println (← result.backward)

def activationGradientOnGpu :=
  positiveSquare.run signedInput
    (device := cuda) (grad := true)
```

```leanOutput customBranchGradient (whitespace := lax)
[0.000000, 1.000000, 9.000000]
[0.000000, 2.000000, 6.000000]
```

On GPU, scalar branch decisions synchronize the current value back to Lean, while the chosen
branch's arithmetic and backward operations stay on GPU. Recording a branch this way is more
expensive than the forward-only compiled conditional, especially for a large tensor. The gradient
follows the selected path, so at a branch boundary it need not be a mathematical derivative.

# Parameters and two inputs

Let's pass the activation slope as a parameter instead of giving every slope a different name:

```lean (name := customParameter)
def activate (slope : Float32) (xs : Tensor Float32 [3])
    (device : Backend.Device := cpu) :
    IO (Tensor Float32 [3]) :=
  (fun x => if x < 0 then slope * x else x).run xs
    (device := device)

#eval activate 0.25 signedInput
```

```leanOutput customParameter (whitespace := lax)
[-0.500000, 1.000000, 3.000000]
```

The slope is supplied at runtime, and the same call accepts `device := cuda`. Captured scalars,
unsigned indices and Boolean switches are shared by every output entry: we evaluate them on CPU and
embed them in the GPU source. Changing one can require recompiling with NVRTC (identical source
reuses the native cache), so for frequently changing parameters an indexed program can read values
from input tensors instead.

For a function of two scalar values, `zip` combines corresponding entries of two tensors:

```lean (name := customZip)
def blend (weight : Float32)
    (left right : Tensor Float32 [3])
    (device : Backend.Device := cpu) :
    IO (Tensor Float32 [3]) :=
  (fun x y => weight * x + (1 - weight) * y).zip
    left right (device := device)

#eval blend 0.25 input signedInput
```

```leanOutput customZip (whitespace := lax)
[-1.250000, 1.250000, 3.000000]
```

The inputs and result share one shape, so a mismatch is a type error (there's no silent
broadcasting or truncation). CPU accepts the same scalar formats as `run`, and GPU also accepts
configured binary arithmetic, with automatic native/software selection. Indexed programs below let
us read different positions and tensors with different shapes.

# Precision

Our annotation `Float32` selects binary32, and changing it to `Float` selects binary64 (the GPU
frontend supports both). Configured FloatLib binary types use compatible native operations where
available, and otherwise we represent each number with integer words and do the arithmetic on the
GPU, without narrowing wide values through binary64. Standard IEEE binary32/binary64 use native
arithmetic, and binary16/bfloat16 addition, subtraction and multiplication use native instructions
on compatible devices, while half/bfloat16 division and custom formats retain software arithmetic
(the full format, including its bias and encoding, determines the choice). NaN payload rules and
separate rounding of products and sums are preserved (these are foreign execution checks, not Lean
proofs). The software backend supports up to 30 exponent bits and 4096 fraction bits, and the
descriptor selects IEEE, finite-with-NaN, unsigned-zero or fully finite conventions. Decimal and
posit formats remain CPU-only.

Let's try the same operation with a wider binary format:

```lean (name := customWide)
abbrev Wide := FloatLib.Floats.ExecFloat.Binary 15 112

def squareWide := fun (x : Wide) => x * x
def wideInput : Tensor Wide [2] := [3, 5]

def wideOnGpu : IO (Tensor Wide [2]) :=
  squareWide.run wideInput (device := cuda)

#eval do
  let output ← squareWide.run wideInput
  pure (FloatLib.Floats.ExecFloat.Binary.toRat? output[0],
    FloatLib.Floats.ExecFloat.Binary.toRat? output[1])
```

```leanOutput customWide (whitespace := lax)
(some 9, some 25)
```

The two format parameters choose 15 exponent bits and 112 fraction bits. We inspect the finite
results as exact rationals instead of converting them through a narrower native float (the tensor
itself stores `Wide` values). Our editor output above comes from a CPU run, and `wideOnGpu` runs
software binary128 arithmetic on CUDA. Binary16, bfloat16, FP8 and custom widths use the same API,
with native operations selected only where compatible. This precision support applies to custom
computations, while the LibTorch model runner retains native binary32 or binary64 buffers.

Recording uses the same API and retains those complete words for gradients:

```lean (name := customWideGradient)
def wideGradientOnGpu :=
  squareWide.run wideInput (device := cuda) (grad := true)

#eval do
  let result ← squareWide.run wideInput (grad := true)
  let gradient ← result.backward
  pure (FloatLib.Floats.ExecFloat.Binary.toRat? gradient[0],
    FloatLib.Floats.ExecFloat.Binary.toRat? gradient[1])
```

```leanOutput customWideGradient (whitespace := lax)
(some 6, some 10)
```

# Reading several tensors

For indexed calculations, we supply a `Program` and put our input tensors in a `TensorPack`,
keeping each one's shape. Here we read the same flat index from two tensors and add the
entries:

```lean (name := customIndexed)
open Kernel

def addInputs : Kernel.Program Float32 :=
  Program.of (fun (read : Reader Float32) (i : UInt64) => do
    let x ← read 0 i
    let y ← read 1 i
    pure (x + y))

def inputs : TensorPack Float32 [[3], [3]] :=
  .pair input input

#eval addInputs.run inputs [3]
```

```leanOutput customIndexed (whitespace := lax)
[2.000000, 4.000000, 6.000000]
```

In each read, the operand number identifies an input tensor, and the index addresses its entries in
row-major order (the last axis changes fastest). Reads check both bounds and return an error if
either is invalid. We asked for the output shape `[3]`, and passing `device := cuda` uses the same
program on GPU. Bounded folds are supported too, and they preserve the written sequential
accumulation order.

# Summing rows

An elementwise function only sees its own entry, so to sum a row we need to read several entries.
We'll use a two-row, three-column tensor:

```lean (name := customRowSum)
def matrix : Tensor Float32 [2, 3] := [[1, 2, 3], [4, 5, 6]]

def rowSum (width : UInt64) : Kernel.Program Float32 :=
  Program.of (fun (read : Reader Float32) (row : UInt64) =>
    iterate (fun column acc => do
      let x ← read 0 (row * width + column)
      pure (acc + x)) width.toNat 0 0)

#eval (rowSum 3).run (.singleton matrix) [2]
```

```leanOutput customRowSum (whitespace := lax)
[6.000000, 15.000000]
```

`row * width + column` locates an entry in the flattened input. `iterate` takes a step function,
the number of steps, the starting index and the initial accumulator. Here it visits columns 0, 1
and 2, starting with a sum of zero, and the output shape `[2]` gives us one result per row. We pass
the width once and use it for both the stride and the number of steps.

```lean
def rowsOnGpu : IO (Tensor Float32 [2]) :=
  (rowSum 3).run (.singleton matrix) [2]
    (device := cuda)
```

Different output rows can run independently, but additions within a row retain the order we wrote.
Floating-point addition isn't associative, so replacing that loop with a different reduction tree
could change its result. This API doesn't make that replacement.

This example explains indexed reads and folds. For ordinary tensor reductions we'd normally use the
existing tensor API, and the model runtime can then use its CPU implementation or LibTorch on GPU.

# Repeated updates

A bounded loop can also evolve each tensor entry independently. We'll use the logistic update
`x ↦ rate * x * (1 - x)` and pass both the rate and number of steps:

```lean (name := customEvolution)
def evolve (rate : Float32) (steps : UInt64) :
    Kernel.Program Float32 :=
  Program.of (fun (read : Reader Float32)
      (i : UInt64) => do
    let initial ← read 0 i
    iterate (fun _ x => pure (rate * x * (1 - x)))
      steps.toNat 0 initial)

def starts : Tensor Float32 [2] := [0.5, 0.25]

#eval (evolve 2 2).run (.singleton starts) [2]
```

```leanOutput customEvolution (whitespace := lax)
[0.500000, 0.468750]
```

On GPU, output entries run independently, while the updates for each entry remain sequential. With
zero steps we recover the starting value. We can also write a numerical recurrence with Lean's
ordinary recursive syntax.

# Recursive source

Let's multiply by an input once per step, keeping the scalar type as a parameter:

```lean (name := customRecursivePower)
def power {α : Type} [One α] [Mul α] (x : α) : Nat → α
  | 0 => 1
  | n + 1 => power x n * x

def powers (depth : UInt64) (input : Tensor Float32 [3])
    (device : Backend.Device := cpu) :
    IO (Tensor Float32 [3]) :=
  (fun x => power x depth.toNat).run input
    (device := device)

#eval powers 3 input
```

```leanOutput customRecursivePower (whitespace := lax)
[1.000000, 8.000000, 27.000000]
```

GPU uses the same definition:

```lean
def powersOnGpu : IO (Tensor Float32 [3]) :=
  powers 3 input (device := cuda)

def widePowersOnGpu : IO (Tensor Wide [2]) :=
  (fun (x : Wide) => power x 3).run wideInput
    (device := cuda)
```

The frontend uses Lean's generated zero and successor equations to derive a loop and prove
agreement with the recursive function, so increasing the depth doesn't unroll a larger program.
Each output entry has its own sequential recurrence. For the depth we can use a literal below
`2^64`, a captured `UInt64.toNat`, or an unsigned calculation involving the output index. As
before, the scalar annotation selects precision, including configured binary arithmetic.

We can record that recurrence on the existing tape:

```lean (name := customRecursiveGradient)
#eval do
  let cube := fun (x : Float32) => power x 3
  let result ← cube.run input (grad := true)
  IO.println result.value
  IO.println (← result.backward)
```

```leanOutput customRecursiveGradient (whitespace := lax)
[1.000000, 8.000000, 27.000000]
[3.000000, 12.000000, 27.000000]
```

Each step adds ordinary tape operations. Backward visits them in reverse, keeping the scalar format
and accumulation schedule. Recording retains intermediate values and launches operations per step
(it doesn't fuse the whole recurrence into the forward-only GPU loop). The recurrence's count must
be independent of the input we're differentiating. Shape-polymorphic tensor recurrences can use the
same recorder too, retaining the input's shape and scalar type through the forward values and
gradients.

We can write the row sum recursively too:

```lean (name := customRecursiveRead)
def sumPrefix (read : Reader Float32) (start : UInt64) :
    Nat → Except Error Float32
  | 0 => pure 0
  | n + 1 => do
      let previous ← sumPrefix read start n
      let x ← read 0 (start + n.toUInt64)
      pure (previous + x)

def recursiveRows (width : UInt64) :
    Kernel.Program Float32 :=
  Program.of (fun (read : Reader Float32) (row : UInt64) =>
    sumPrefix read (row * width) width.toNat)

#eval (recursiveRows 3).run
  (.singleton matrix) [2]
```

```leanOutput customRecursiveRead (whitespace := lax)
[6.000000, 15.000000]
```

```lean
def recursiveRowsOnGpu : IO (Tensor Float32 [2]) :=
  (recursiveRows 3).run (.singleton matrix) [2]
    (device := cuda)
```

At depth zero there are no reads, and a failed read stops the recurrence just as it does in the
original definition. `iterate_eq_recurrence` proves this bridge without scalar algebraic laws (the
multiplication and addition schedules aren't reassociated).

This path requires fixed parameters and calls on the predecessor of the final natural-number
argument, and although supported step bodies can contain branches, reads and nested recurrences, it
isn't general tree recursion or recursion over dynamically allocated structures (the next section
folds trees). `Program.tangent` and `Program.gradient` derive the tangent and gradient programs of
an indexed `Program`, which run like the program itself (the scalar recorder above is a separate
route through the existing tape). Unsupported GPU source is rejected, not silently run on CPU. For
nested recurrences, pass the inner bounds as `UInt64` parameters too, because hard-coded inner
counts can exceed Lean's normal proof-elaboration budget.

# Recursion over trees

The recurrences above follow a natural number. For a recursive function over an inductive type of
our own, we derive `Sig.Inductive` for the type and pass the function to `Fold.of`:

```lean (name := customTreeSum)
inductive BTree where
  | leaf
  | node (l : BTree) (x : Float) (r : BTree)
  deriving Sig.Inductive

def BTree.sum : BTree → Float
  | .leaf => 0
  | .node l x r => l.sum + x + r.sum

def sumFold := Fold.of BTree.sum

def BTree.arena (t : BTree) : Arena Float :=
  Arena.encode (σ := BTree.sig) (T := fun _ => BTree)
    (s := 0) t

def small : BTree :=
  .node (.node .leaf 1 .leaf) 2 (.node .leaf 3 .leaf)

#eval sumFold.eval (Reader.ofArrays #[]) small.arena
```

```leanOutput customTreeSum (whitespace := lax)
Except.ok 6.000000
```

`Arena.encode` stores the tree by levels, with the children of level `r` on level `r + 1`. The fold
computes the levels from the deepest up with one row kernel, and Lean's kernel checks the
certificate that `Fold.of` builds. `Fold.eval_encode` then gives the Lean function on every tree
with fewer than `2^64` nodes:

```lean (name := customTreeProof)
theorem sumFold_eval (t : BTree)
    (h : Sig.Inductive.count (σ := BTree.sig)
      (T := fun _ => BTree) (s := 0) t < UInt64.size) :
    sumFold.eval (Reader.ofArrays #[]) t.arena =
      .ok t.sum :=
  sumFold.eval_encode _ t h

#print axioms sumFold_eval
```

```leanOutput customTreeProof (whitespace := lax)
'Tutorial.CustomComputations.sumFold_eval' depends on axioms:
  [propext, Classical.choice, Quot.sound]
```

`Arena.upload` puts the arena in device memory, and `Fold.run` launches the row kernel once per
level there:

```lean
def sumOnGpu : IO (Array (Except Error Float)) := do
  let arena ← Arena.upload BTree.sig small.arena
  sumFold.run arena #[]
```

`Fold.of` accepts functions of the forms `T → α`, `T → Except Error α` and
`Reader α → T → Except Error α`, for a type of one sort whose scalars are `Float`, `Float32` or a
configured binary format. It rejects a function it can't compile with a message that names the term.
Hylomorphisms expand the calls of a recursion that doesn't follow a value, such as Fibonacci or a
sum by halves, level by level on the GPU. Unfolds build the arena of a tree in device memory, where
folds run over it. The proofs cover the level schedule and the CUDA text of a row kernel for the
native formats, and NVRTC and the device run stay outside them.

# Checking the calculation

Lean is useful here because our function and theorems about it live in the same language. We can
use [mathlib](https://leanprover-community.github.io/mathlib-overview.html) for mathematical
results instead of rebuilding calculus and linear algebra for each program, and Lean's
[tactics](https://lean-lang.org/doc/reference/latest/Tactic-Proofs/Tactic-Reference/) construct
proofs that its kernel checks. FloatLib adds executable arithmetic and theorems about its numerical
behavior. Those are concrete reasons to build this inside Lean, not a claim that every Lean program
is faster or that every calculation already has a proof.

We take inspiration from functional array languages such as
[Accelerate](https://www.acceleratehs.org/), [Futhark](https://futhark-lang.org/) and
[Dex](https://arxiv.org/abs/2104.05372), as well as
[Bend](https://github.com/bendlang/bend). Our scope is narrower than a general GPU language:
we compile these tensor calculations, not arbitrary Lean programs.

Bend also supports laws and checked proofs, but its [implementation
notes](https://github.com/bendlang/bend/blob/main/WONTFIX.txt) describe F32 operations as axioms,
without bit-level definitions. FloatLib gives us a different starting point for reasoning about
rounded arithmetic, though on GPU we have to state the contracts connecting our arithmetic model to
native execution.

`Program.eval_eq_reference` proves that tensor evaluation agrees with the source calculation,
including failed reads, and the lowering theorems preserve branches, indexed reads and sequential
folds. Our emitted-source interpretation theorem covers native FP32/FP64 under contracts for scalar
arithmetic and input buffers. Configured-format CUDA arithmetic is checked against FloatLib by the
runtime regressions, not kernel-verified. These proofs don't verify NVRTC, the GPU or foreign
memory accesses.

Canonical IR can store these programs as custom nodes alongside registered operators. A custom node
is the operator `Op.custom` with the program's name and signature. The graph's payload holds the
program, and `Op.customRule` builds the node's rule from it, with a supplied pullback that claims
no derivative. In the IR, the shape checker validates the input and output signature. The resident
forward runner (`IR.Graph.runBuffers`, for graphs of registered operators) runs each operator
through its entry in `LibTorch.Kernel.ofOp` and each custom program through NVRTC, and requires
binary32 input buffers, `Float32` payloads and native binary32 custom programs, even though
standalone custom computations and the autograd tape support other precisions. Source equivalence
alone doesn't supply a real interval enclosure or a derivative, so IBP and CROWN reject unsupported
custom transfers, and PyTorch export rejects arbitrary custom bodies.

We document the API and its proof boundaries in
{src "TorchLean/NN/Kernel/Function.lean"}[`Function.lean`],
{src "TorchLean/NN/Kernel/Tensor.lean"}[`Tensor.lean`] and
{src "TorchLean/CUDA/Kernel/Emit/Source.lean"}[`Emit/Source.lean`].

```lean -show
end Tutorial.CustomComputations
```
