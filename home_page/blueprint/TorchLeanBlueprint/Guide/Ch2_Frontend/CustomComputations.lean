import VersoManual
import NN.Kernel
import NN.API.Precision
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

Let's write a scalar function and apply it to a tensor. We'll start on CPU, where we can run the
example in Lean's Infoview, then use the same function on GPU.

Why add this when we already have LibTorch? Sometimes the calculation we want isn't a named
library operation. We might want to change an activation, combine several elementwise steps,
or read a small neighborhood of a tensor. We should be able to write that calculation in Lean
without maintaining a second implementation in a CUDA file.

[Bend](https://github.com/bendlang/bend) makes a broader version of this idea appealing: write
functions, split independent work explicitly, and run it on CPU or GPU. Here we'll explore the
part we can use in TorchLean today: tensor calculations whose source we can also reason about
in Lean. This is not a compiler for arbitrary Lean programs on GPU.

Python with PyTorch already gives us tensor expressions, and
[custom C++/CUDA operators](https://docs.pytorch.org/tutorials/advanced/cpp_custom_ops.html)
when we need a native extension. We reuse that infrastructure for standard operations.
Here we're adding a way to keep a small custom calculation in the same language as its
specification.
[Rust ownership](https://doc.rust-lang.org/book/ch04-01-what-is-ownership.html)
addresses memory management through compile-time rules. A numerical theorem asks a different
question: what does this calculation mean, and which mathematical property does it satisfy?
Using Lean lets us ask both programming and mathematical questions without translating the
calculation into another proof language.

There are useful precedents beyond Bend. [Accelerate](https://www.acceleratehs.org/) embeds
parallel array computations in Haskell. [Futhark](https://futhark-lang.org/) compiles a functional
array language to CPU and GPU code. [Dex](https://arxiv.org/abs/2104.05372) explores explicit
indexing and typed index sets for numerical programming. Our indexed example below shares that
interest in writing down which entries contribute to each output. What we add inside Lean is a
checked correspondence between the accepted source calculation and its lowered representation,
alongside the mathematical libraries we already use for models.

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
{ref "gpu-and-cuda"}[GPU and CUDA]. An unavailable GPU or unsupported function produces an IO
error. The runtime does not move the computation to CPU without asking.

CPU applies the original Lean function to each entry. For GPU, the frontend recognizes supported
scalar arithmetic, comparisons, local bindings and conditionals during elaboration. It builds a
typed expression with a proof that its evaluation agrees with the function we wrote. At runtime,
the native bridge renders CUDA, compiles it with NVRTC (NVIDIA's runtime CUDA compiler) and
executes it.

On GPU, ordinary model operations such as matrix multiplication still use LibTorch's ATen
primitives.
The portable CPU path evaluates the corresponding Lean runtime operations.
This API lets us supply our own scalar calculation without writing a separate CUDA source file.
It does not generate a backward rule; custom differentiation needs its own proved rule and runtime
implementation.

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

# Precision

Our annotation `Float32` selects binary32. Changing it to `Float` selects binary64; both are
supported by the GPU frontend. CPU can also evaluate functions over FloatLib's configured formats
when the scalar type has TorchLean storage support. Arbitrary binary, decimal and posit arithmetic
is not yet compiled to GPU. Converting a value to binary64 would change its arithmetic, so the
runtime does not use that as an arbitrary-precision implementation.

Let's try the same operation on CPU with a wider binary format:

```lean (name := customWide)
abbrev Wide := FloatLib.Floats.ExecFloat.Binary 15 112

def squareWide := fun (x : Wide) => x * x
def wideInput : Tensor Wide [2] := [3, 5]

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
still stores `Wide` values. This is CPU execution of FloatLib arithmetic, not a binary128 GPU
kernel.

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

def rowSum : Program Float32 :=
  Program.of (fun (read : Reader Float32) (row : UInt64) =>
    iterate (fun column acc => do
      let x ← read 0 (row * 3 + column)
      pure (acc + x)) 3 0 0)

#eval rowSum.run (Arguments.empty.push matrix) [2]
```

```leanOutput customRowSum (whitespace := lax)
[6.000000, 15.000000]
```

`row * 3 + column` locates an entry in the flattened input. `iterate` takes a step function,
the number of steps, the starting index, and the initial accumulator. Here it visits columns
0, 1 and 2, starting with a sum of zero. The output shape `[2]` gives us one result per row.
For another width we'd change the row stride and the number of steps together.

```lean
def rowsOnGpu : IO (Tensor Float32 [2]) :=
  rowSum.run (Arguments.empty.push matrix) [2]
    (device := gpu)
```

Different output rows can run independently; additions within a row retain the order we wrote.
Floating-point addition is not associative, so replacing that loop with a different reduction
tree could change its result. This API does not make that replacement.

This example explains indexed reads and folds. For ordinary tensor reductions, we'd normally
use the existing tensor API. The model runtime can then use its CPU implementation or LibTorch
on GPU.

# Checking the calculation

Lean is useful here because our function and theorems about it live in the same language.
We can use [mathlib](https://leanprover-community.github.io/mathlib-overview.html) for mathematical
results rather than rebuild calculus and linear algebra for each program. Lean's
[tactics](https://lean-lang.org/doc/reference/latest/Tactic-Proofs/Tactic-Reference/) construct
proofs which its kernel checks. FloatLib adds executable arithmetic and theorems about its
numerical behavior. Those are concrete reasons to build this inside Lean, not a claim that every
Lean program is faster or that every calculation already has a proof.

Bend also supports laws and checked proofs. Its current
[implementation notes](https://github.com/bendlang/bend/blob/main/WONTFIX.txt) describe F32
operations as axioms, with bit-level definitions planned. FloatLib gives us a different starting
point for reasoning about rounded arithmetic. On GPU, though, we still have to state the contracts
connecting our arithmetic model to native execution.

`Program.eval_eq_reference` proves that tensor evaluation agrees with the source calculation,
including failed reads. The lowering theorems preserve branches, indexed reads and sequential
folds. The emitted-source theorem additionally uses explicit contracts for scalar arithmetic and
input buffers. These proofs do not verify NVRTC, the GPU or foreign memory accesses.

Canonical IR can store these programs as custom nodes alongside supported LibTorch operations.
Its shape checker validates the input and output signature. Source equivalence alone does not
supply a real interval enclosure or a derivative, so IBP and CROWN reject unsupported custom
transfers, and PyTorch export rejects arbitrary custom bodies.

The API and its proof boundaries are documented in
{src "NN/Kernel/Function.lean"}[`Function.lean`],
{src "NN/Kernel/Tensor.lean"}[`Tensor.lean`] and
{src "NN/Kernel/Cuda/Source.lean"}[`Cuda/Source.lean`].

```lean -show
end Tutorial.CustomComputations
```
