import VersoManual
import NN.Kernel
import NN.API.Precision
import NN.API.Autograd
import NN.Tactic.Autograd
import TorchLeanBlueprint.Roles

open Verso.Genre Manual
open Verso.Genre.Manual.InlineLean
open TorchLean
open Lean.Elab.Tactic.GuardMsgs.WhitespaceMode (lax)

#doc (Manual) "Custom Tensor Computations" =>
%%%
tag := "custom-computations"
file := "Custom-Tensor-Computations"
%%%

```lean -show
namespace Tutorial.CustomComputations
```

Let's write a function and run it on a tensor. We'll start in Lean's Infoview with a square,
then try parameters, row sums and repeated updates. We'll also ask for a gradient without
rewriting the function.

Sometimes the calculation we want isn't a named LibTorch operation. We might want to change
an activation, combine several elementwise steps,
or read a small neighborhood of a tensor. We should be able to write that calculation in Lean
without maintaining a second implementation in a CUDA file.

```lean (name := customSquare)
def square := fun (x : Float32) => x * x

def input : Tensor Float32 [3] :=
  Tensor.ofFn fun i => Float32.ofNat (i.val + 1)

#eval square.run input
```

```leanOutput customSquare (whitespace := lax)
[1.000000, 4.000000, 9.000000]
```

CPU is the default. The result is still a `Tensor Float32 [3]`, and `#eval` prints it directly.
We don't need to convert our data to a list or unpack a native buffer.

# Choosing a device

Pass the device when you run the function:

```lean
def onCpu : IO (Tensor Float32 [3]) :=
  square.run input (device := cpu)

def onGpu : IO (Tensor Float32 [3]) :=
  square.run input (device := gpu)
```

These definitions are checked when we build the guide. We evaluate only the CPU example here;
running `onGpu` needs a CUDA-enabled LibTorch build and a visible GPU, as described in
{ref "gpu-and-cuda"}[GPU and CUDA]. A CPU-only scalar type or format uses CPU with a diagnostic,
without converting its values to a narrower type. An unavailable GPU or unsupported function
for a GPU-capable type still produces an IO error.

CPU applies the original Lean function to each entry. For GPU, the frontend recognizes supported
scalar arithmetic, comparisons, local bindings and conditionals during elaboration. It builds a
typed expression with a proof that its evaluation agrees with the function we wrote. At runtime,
the native bridge renders CUDA, compiles it with NVRTC (NVIDIA's runtime CUDA compiler) and
executes it.

On GPU, ordinary model operations such as matrix multiplication still use LibTorch's ATen
primitives.
The portable CPU path evaluates the corresponding Lean runtime operations.
This API lets us supply our own scalar calculation without writing a separate CUDA source file.
For differentiation, we can record supported arithmetic at the call site, as in the
next example. Arbitrary indexed custom bodies don't acquire backward rules from source equivalence.

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
requirements, and calls still infer them from the input:

```lean (name := customTensorSquare)
def tensorSquare {α : Type} [Storage α] [Mul α]
    {s : Shape} (x : Tensor α s) : Tensor α s := x * x

#eval tensorSquare.run input
```

```leanOutput customTensorSquare (whitespace := lax)
[1.000000, 4.000000, 9.000000]
```

Recording uses the existing arithmetic, tensor sums, square, ReLU, sigmoid, tanh and rank-two
matrix products, along with their VJP rules. A VJP takes an output gradient and propagates it
back to the inputs. The frontend checks that its reconstructed arithmetic expression equals our
function, without reassociating operations. There is no separate custom-gradient tape.
Captured tensors are constants; backward differentiates the supplied input.
Division and direct negation use that same interface. Scalar input-dependent conditionals
record only the chosen branch; fixed-parameter scalar recurrences record each step. The count
must be independent of differentiable inputs.

Backward consumes the recording, even on failure. Its default seed is all ones, which gives the
derivative of the sum of output entries. Pass `result.backward seed` to supply different output
gradients, or
call `result.close` when backward isn't needed. Unsupported recordings return an error; CPU
forward-only calls still evaluate the original function.

The element type can still be a parameter. The GPU tape retains binary32 for `Float32`, binary64
for `Float`, and complete words for configured binary arithmetic, including binary128. Configured
recording supports addition, subtraction, multiplication, division, negation and the structural
operations used by scalar traversal. Other native model operations and native optimizer
checkpoints remain limited to their native dtypes; an unsupported operation asks us to use CPU.

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

The conditional is part of our function, not a separate tensor operation we have to name.
The GPU frontend supports this form too:

```lean
def activationOnGpu : IO (Tensor Float32 [3]) :=
  positiveSquare.run signedInput (device := gpu)
```

Only the chosen branch is evaluated. This matters when the other branch contains a division
or another operation we don't want to execute for that input.

We can ask for the activation's gradient without rewriting the conditional:

```lean (name := customBranchGradient)
#eval do
  let result ← positiveSquare.run signedInput (grad := true)
  IO.println result.value
  IO.println (← result.backward)

def activationGradientOnGpu :=
  positiveSquare.run signedInput
    (device := gpu) (grad := true)
```

```leanOutput customBranchGradient (whitespace := lax)
[0.000000, 1.000000, 9.000000]
[0.000000, 2.000000, 6.000000]
```

On GPU, scalar branch decisions synchronize the current value back to Lean. The chosen branch's
arithmetic and backward operations stay on GPU. This is more expensive than the forward-only
compiled conditional, especially for a large tensor. The gradient follows the selected path;
at a branch boundary it need not be a mathematical derivative.

# Parameters and two inputs

Let's pass the activation slope as a parameter, rather than give every slope a different name:

```lean (name := customParameter)
def activate (slope : Float32) (xs : Tensor Float32 [3])
    (device : NN.Backend.Device := cpu) :
    IO (Tensor Float32 [3]) :=
  (fun x => if x < 0 then slope * x else x).run xs
    (device := device)

#eval activate 0.25 signedInput
```

```leanOutput customParameter (whitespace := lax)
[-0.500000, 1.000000, 3.000000]
```

The slope is supplied at runtime, and the same call accepts `device := gpu`. Captured scalars,
unsigned indices and Boolean switches are shared by every output entry. We evaluate these shared
values on CPU and embed them in the GPU source. Changing one can require a new NVRTC compilation;
identical source reuses the native cache. For frequently changing parameters, an indexed program
can instead read values from input tensors.

For a function of two scalar values, `zip` combines corresponding entries of two tensors:

```lean (name := customZip)
def blend (weight : Float32)
    (left right : Tensor Float32 [3])
    (device : NN.Backend.Device := cpu) :
    IO (Tensor Float32 [3]) :=
  (fun x y => weight * x + (1 - weight) * y).zip
    left right (device := device)

#eval blend 0.25 input signedInput
```

```leanOutput customZip (whitespace := lax)
[-1.250000, 1.250000, 3.000000]
```

The inputs and result share one shape. A mismatch is a type error, with no silent broadcasting
or truncation. CPU accepts the same scalar formats as `run`; GPU also accepts configured
binary arithmetic with automatic native/software selection.
Indexed programs below let us read different positions and tensors with different shapes.

# Precision

Our annotation `Float32` selects binary32. Changing it to `Float` selects binary64; both are
supported by the GPU frontend. Configured FloatLib binary types use compatible native operations
where available. Otherwise, we represent each number with integer words and do the arithmetic
on the GPU, without narrowing wide values through binary64. Standard IEEE binary32/binary64 use
native arithmetic;
binary16/bfloat16 addition, subtraction and multiplication use native instructions on compatible
devices. Half/bfloat16 division and custom formats retain software arithmetic. The full format,
including its bias and encoding, determines the choice. NaN payload rules and separate rounding
of products and sums are preserved. These are foreign execution checks, not new Lean proofs.
The software backend supports up to 30 exponent bits and 4096 fraction bits.
The descriptor selects IEEE, finite-with-NaN, unsigned-zero or fully finite conventions.
Decimal and posit formats remain CPU-only.

Let's try the same operation with a wider binary format:

```lean (name := customWide)
abbrev Wide := FloatLib.Floats.ExecFloat.Binary 15 112

def squareWide := fun (x : Wide) => x * x
def wideInput : Tensor Wide [2] := [3, 5]

def wideOnGpu : IO (Tensor Wide [2]) :=
  squareWide.run wideInput (device := gpu)

#eval do
  let output ← squareWide.run wideInput
  pure (FloatLib.Floats.ExecFloat.Binary.toRat? output[0],
    FloatLib.Floats.ExecFloat.Binary.toRat? output[1])
```

```leanOutput customWide (whitespace := lax)
(some 9, some 25)
```

The two format parameters choose 15 exponent bits and 112 fraction bits. We inspect the finite
results as exact rationals, rather than convert them through a narrower native float. The tensor
still stores `Wide` values. The editor output above runs on CPU; `wideOnGpu` runs software
binary128 arithmetic on CUDA. Binary16, bfloat16, FP8 and custom widths use the same API, with
native operations selected only where compatible. This precision support applies to custom
computations; the LibTorch model runner retains native binary32 or binary64 buffers.

Recording uses the same API and retains those complete words for gradients:

```lean (name := customWideGradient)
def wideGradientOnGpu :=
  squareWide.run wideInput (device := gpu) (grad := true)

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

For indexed calculations, we supply a `Program` and put our input tensors in `Arguments`,
keeping each one's shape. Here we read the same flat index from two tensors and add the
entries:

```lean (name := customIndexed)
open NN.Kernel
open scoped NN.Kernel

def addInputs : Program Float32 :=
  Program.of (fun (read : Reader Float32) (i : UInt64) => do
    let x ← read 0 i
    let y ← read 1 i
    pure (x + y))

def inputs : Arguments Float32 [[3], [3]] :=
  (Arguments.empty.push input).push input

#eval addInputs.run inputs [3]
```

```leanOutput customIndexed (whitespace := lax)
[2.000000, 4.000000, 6.000000]
```

The operand number identifies an input tensor. The index addresses its entries in row-major
order, with the last axis changing fastest. Reads check both bounds and return an error if either
is invalid. The requested output shape
is `[3]`; passing `device := gpu` uses the same program on GPU. Bounded folds are also supported,
with the written sequential accumulation order preserved.

# Summing rows

An elementwise function only sees its own entry. To sum a row, we need to read several entries.
We'll use a two-row, three-column tensor:

```lean (name := customRowSum)
def matrix : Tensor Float32 [2, 3] := [[1, 2, 3], [4, 5, 6]]

def rowSum (width : UInt64) : Program Float32 :=
  Program.of (fun (read : Reader Float32) (row : UInt64) =>
    iterate (fun column acc => do
      let x ← read 0 (row * width + column)
      pure (acc + x)) width.toNat 0 0)

#eval (rowSum 3).run (Arguments.empty.push matrix) [2]
```

```leanOutput customRowSum (whitespace := lax)
[6.000000, 15.000000]
```

`row * width + column` locates an entry in the flattened input. `iterate` takes a step function,
the number of steps, the starting index, and the initial accumulator. Here it visits columns
0, 1 and 2, starting with a sum of zero. The output shape `[2]` gives us one result per row.
We pass the width once, using it for both the stride and the number of steps.

```lean
def rowsOnGpu : IO (Tensor Float32 [2]) :=
  (rowSum 3).run (Arguments.empty.push matrix) [2]
    (device := gpu)
```

Different output rows can run independently; additions within a row retain the order we wrote.
Floating-point addition is not associative, so replacing that loop with a different reduction
tree could change its result. This API does not make that replacement.

This example explains indexed reads and folds. For ordinary tensor reductions, we'd normally
use the existing tensor API. The model runtime can then use its CPU implementation or LibTorch
on GPU.

# Repeated updates

A bounded loop can also evolve each tensor entry independently. We'll use the logistic update
`x ↦ rate * x * (1 - x)` and pass both the rate and number of steps:

```lean (name := customEvolution)
def evolve (rate : Float32) (steps : UInt64) :
    Program Float32 :=
  Program.of (fun (read : Reader Float32)
      (i : UInt64) => do
    let initial ← read 0 i
    iterate (fun _ x => pure (rate * x * (1 - x)))
      steps.toNat 0 initial)

def starts : Tensor Float32 [2] := [0.5, 0.25]

#eval (evolve 2 2).run (Arguments.empty.push starts) [2]
```

```leanOutput customEvolution (whitespace := lax)
[0.500000, 0.468750]
```

On GPU, output entries run independently, while the updates for each entry remain sequential.
With zero steps we recover the starting value. We can also write a numerical recurrence directly
with Lean's ordinary recursive syntax.

# Recursive source

Let's multiply by an input once per step. We keep the scalar type as a parameter:

```lean (name := customRecursivePower)
def power {α : Type} [One α] [Mul α] (x : α) : Nat → α
  | 0 => 1
  | n + 1 => power x n * x

def powers (depth : UInt64) (input : Tensor Float32 [3])
    (device : NN.Backend.Device := cpu) :
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
  powers 3 input (device := gpu)

def widePowersOnGpu : IO (Tensor Wide [2]) :=
  (fun (x : Wide) => power x 3).run wideInput
    (device := gpu)
```

The frontend uses Lean's generated zero and successor equations to derive a loop and prove
agreement with the recursive function. Increasing the depth does not unroll a larger program.
Each output entry has its own sequential recurrence. The depth can be a literal below `2^64`,
a captured `UInt64.toNat`, or an unsigned calculation involving the output index.
The scalar annotation still selects precision, including configured binary arithmetic.

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

Each step adds ordinary tape operations. Backward visits them in reverse, keeping the scalar
format and accumulation schedule. Recording retains intermediate values and launches operations
per step; it doesn't fuse the whole recurrence into the forward-only GPU loop. Its count must
be independent of the input we're differentiating.
Shape-polymorphic tensor recurrences can use the same recorder too, retaining the input's shape
and scalar type through the forward values and gradients.

We can write the row sum recursively too:

```lean (name := customRecursiveRead)
def sumPrefix (read : Reader Float32) (start : UInt64) :
    Nat → Except Error Float32
  | 0 => pure 0
  | n + 1 => do
      let previous ← sumPrefix read start n
      let x ← read 0 (start + n.toUInt64)
      pure (previous + x)

def recursiveRows (width : UInt64) : Program Float32 :=
  Program.of (fun (read : Reader Float32) (row : UInt64) =>
    sumPrefix read (row * width) width.toNat)

#eval (recursiveRows 3).run
  (Arguments.empty.push matrix) [2]
```

```leanOutput customRecursiveRead (whitespace := lax)
[6.000000, 15.000000]
```

```lean
def recursiveRowsOnGpu : IO (Tensor Float32 [2]) :=
  (recursiveRows 3).run (Arguments.empty.push matrix) [2]
    (device := gpu)
```

At depth zero there are no reads. A failed read stops the recurrence, just as it does in the
original definition. `iterate_eq_recurrence` proves this bridge without scalar algebraic laws.
The multiplication and addition schedules are not reassociated.

This path requires fixed parameters and calls on the predecessor of the final natural-number
argument. Supported step bodies can contain branches, reads and nested recurrences. It is not
general tree recursion or recursion over dynamically allocated structures. Indexed `Program`
bodies do not acquire automatic backward rules; the scalar recorder above is a separate route
through the existing tape. Unsupported GPU source is rejected, not silently run on CPU.
For nested recurrences, pass the inner bounds as `UInt64` parameters too. Hard-coded inner counts
can still exceed Lean's normal proof-elaboration budget.

# Checking the calculation

Lean is useful here because our function and theorems about it live in the same language.
We can use [mathlib](https://leanprover-community.github.io/mathlib-overview.html) for mathematical
results rather than rebuild calculus and linear algebra for each program. Lean's
[tactics](https://lean-lang.org/doc/reference/latest/Tactic-Proofs/Tactic-Reference/) construct
proofs which its kernel checks. FloatLib adds executable arithmetic and theorems about its
numerical behavior. Those are concrete reasons to build this inside Lean, not a claim that every
Lean program is faster or that every calculation already has a proof.

We take inspiration from functional array languages such as
[Accelerate](https://www.acceleratehs.org/), [Futhark](https://futhark-lang.org/) and
[Dex](https://arxiv.org/abs/2104.05372), as well as
[Bend](https://github.com/bendlang/bend). Our scope is narrower than a general GPU language:
we compile these tensor calculations, not arbitrary Lean programs.

Bend also supports laws and checked proofs. Its current
[implementation notes](https://github.com/bendlang/bend/blob/main/WONTFIX.txt) describe F32
operations as axioms, with bit-level definitions planned. FloatLib gives us a different starting
point for reasoning about rounded arithmetic. On GPU, though, we still have to state the contracts
connecting our arithmetic model to native execution.

`Program.eval_eq_reference` proves that tensor evaluation agrees with the source calculation,
including failed reads. The lowering theorems preserve branches, indexed reads and sequential
folds. The emitted-source interpretation theorem currently covers native FP32/FP64 under explicit
contracts for scalar arithmetic and input buffers. Configured-format CUDA arithmetic is checked
against FloatLib by the runtime regressions, not kernel-verified. These proofs do not verify
NVRTC, the GPU or foreign memory accesses.

Canonical IR can store these programs as custom nodes alongside supported LibTorch operations.
Its shape checker validates the input and output signature. The resident `Graph.runBuffers`
forward runner currently requires binary32 inputs and payloads, even though standalone custom
computations and the autograd tape support other precisions. Source equivalence alone does not
supply a real interval enclosure or a derivative, so IBP and CROWN reject unsupported custom
transfers, and PyTorch export rejects arbitrary custom bodies.

The API and its proof boundaries are documented in
{src "NN/Kernel/Function.lean"}[`Function.lean`],
{src "NN/Kernel/Tensor.lean"}[`Tensor.lean`] and
{src "NN/Kernel/Cuda/Source.lean"}[`Cuda/Source.lean`].

```lean -show
end Tutorial.CustomComputations
```
