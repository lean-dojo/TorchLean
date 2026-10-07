---
title: Custom Tensor Computations
---

Let's apply a function we wrote in Lean to a tensor, then choose where to run it. We'll start
in the editor.

We already use LibTorch for standard GPU model operations. But when we want a different activation
or a small indexed calculation, it's useful to write the function in Lean and keep working with
tensors. We don't have to maintain a separate CUDA implementation for the supported calculations
shown here.

[Bend](https://github.com/bendlang/bend) is an interesting reference for this: it lets programmers
write functions and explicitly split work for parallel CPU/GPU execution. TorchLean's frontend
has a narrower scope. Here we'll use scalar functions, indexed reads and bounded folds—not
arbitrary recursive Lean programs on GPU.

Python with PyTorch already supports tensor expressions and
[custom C++/CUDA operators](https://docs.pytorch.org/tutorials/advanced/cpp_custom_ops.html).
We reuse its native infrastructure rather than replace it. Languages such as
[Rust](https://doc.rust-lang.org/book/ch04-01-what-is-ownership.html) give us compile-time ownership
rules for memory management. Here we're interested in another question too: can we state and
prove a mathematical property of the calculation we just wrote? Lean lets us work on the program
and that question together.

[Accelerate](https://www.acceleratehs.org/), [Futhark](https://futhark-lang.org/), and
[Dex](https://arxiv.org/abs/2104.05372) are useful references too: they explore functional array
programming and compilation for parallel numerical work. Dex's explicit indexing is especially
relevant to the row-sum example below. Here we keep the accepted source calculation connected to
its lowered representation by a Lean proof, alongside the libraries we use to reason about models.

## Start with a square

```lean
import NN.Kernel
import NN.API.Precision
open TorchLean

def square := fun (x : Float32) => x * x

def input : Tensor Float32 [3] :=
  Tensor.ofFn fun i => Float32.ofNat (i.val + 1)

#eval square.run input
-- [1.000000, 4.000000, 9.000000]
```

The result keeps the input shape. `#eval` prints the tensor directly, so there's no list conversion
or buffer handling in our example. CPU is the default.

## Run on GPU

We select the device at the call site:

`open TorchLean` makes `cpu` and `gpu` available by name; no leading dot is needed.

```lean
def onCpu : IO (Tensor Float32 [3]) :=
  square.run input (device := cpu)

def onGpu : IO (Tensor Float32 [3]) :=
  square.run input (device := gpu)
```

Run `onGpu` from a CUDA-enabled build with a visible GPU. The
[GPU guide]({{ '/blueprint/Floating-Point-and-Native-Boundaries/From-A-Tensor-Operation-To-A-GPU-Kernel/' | relative_url }})
explains SDK setup. Unsupported source and unavailable devices produce IO errors; the runtime
doesn't silently substitute CPU execution.

CPU evaluates our original function. GPU uses a frontend that recognizes scalar arithmetic,
comparisons, local bindings and conditionals, proves agreement with a typed expression, and
generates CUDA. The native bridge compiles that source with NVRTC, NVIDIA's runtime CUDA compiler.
Model operations such as
matrix multiplication still call LibTorch's ATen primitives.

## Change the activation

Let's keep only the nonnegative entries and square them:

```lean
def positiveSquare := fun (x : Float32) =>
  if x < 0 then 0 else x * x

def signedInput : Tensor Float32 [3] := [-2, 1, 3]

#eval positiveSquare.run signedInput
-- [0.000000, 1.000000, 9.000000]

def activationOnGpu : IO (Tensor Float32 [3]) :=
  positiveSquare.run signedInput (device := gpu)
```

The conditional belongs to the function we wrote. Only its chosen branch is evaluated;
the GPU frontend preserves that choice too. The result still has shape `[3]`.

## Read several tensors

An indexed `Program` lets an output read entries from more than one tensor. We put the tensors
in `Arguments`, keeping each one's shape:

```lean
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
-- [2.000000, 4.000000, 6.000000]
```

Here `0` and `1` identify our input tensors, and `i` counts through their entries in row-major
order, with the last axis changing fastest. Reads check bounds.
The output shape is `[3]`; adding `device := gpu` selects GPU execution for the same program.
Bounded folds preserve the sequential accumulation order we write.

## Sum each row

Now each output needs several input entries. We'll sum the rows of a matrix:

```lean
def matrix : Tensor Float32 [2, 3] := [[1, 2, 3], [4, 5, 6]]

def rowSum : Program Float32 :=
  Program.of (fun (read : Reader Float32) (row : UInt64) =>
    iterate (fun column acc => do
      let x ← read 0 (row * 3 + column)
      pure (acc + x)) 3 0 0)

#eval rowSum.run (Arguments.empty.push matrix) [2]
-- [6.000000, 15.000000]

def rowsOnGpu : IO (Tensor Float32 [2]) :=
  rowSum.run (Arguments.empty.push matrix) [2] (device := gpu)
```

`row * 3 + column` locates an entry in our three-column input. `iterate` runs three steps,
starting at column zero with an accumulator of zero. The output shape `[2]` asks for one sum
per row. If we change the width, we need to change both the stride and the step count.

Output rows can run independently. Within each row, additions keep the order we wrote:
floating-point addition is not associative, so a different reduction tree could change the
answer. This is a small example of indexed programming; for an ordinary reduction we'd usually
use the existing tensor API.

## Precision and proofs

Why do this in Lean? The calculation and its mathematical specification can share definitions.
We can reuse [mathlib](https://leanprover-community.github.io/mathlib-overview.html) rather than
rebuild the mathematics behind each model. Lean's
[tactics](https://lean-lang.org/doc/reference/latest/Tactic-Proofs/Tactic-Reference/) help construct
proofs, which its kernel checks. FloatLib supplies executable arithmetic and numerical theorems.
We don't have to change languages when a working example leads to a mathematical question.

Bend also has laws and checked proofs. Its current
[implementation notes](https://github.com/bendlang/bend/blob/main/WONTFIX.txt) describe native F32
operations as axioms, with bit-level definitions planned. FloatLib gives us existing infrastructure
for reasoning about rounded arithmetic. That doesn't establish a speed advantage over Bend,
and it doesn't automatically verify our native GPU execution.

GPU supports `Float32` and `Float`, with binary32 and binary64 arithmetic respectively. CPU also
supports ordinary functions over FloatLib formats with TorchLean storage instances. Arbitrary
binary, decimal and posit arithmetic is not currently compiled to GPU.

We can still use a wider FloatLib format on CPU:

```lean
abbrev Wide := FloatLib.Floats.ExecFloat.Binary 15 112
def squareWide := fun (x : Wide) => x * x
def wideInput : Tensor Wide [2] := [3, 5]

#eval do
  let output ← squareWide.run wideInput
  pure (FloatLib.Floats.ExecFloat.Binary.toRat? output[0],
    FloatLib.Floats.ExecFloat.Binary.toRat? output[1])
-- (some 9, some 25)
```

The parameters choose 15 exponent bits and 112 fraction bits. `toRat?` exposes each finite
result as an exact rational, so inspecting it doesn't narrow it to a native float.

The compiler's Lean proofs cover source evaluation, structured lowering and emitted-source
semantics under explicit arithmetic and input-buffer contracts. NVRTC, foreign memory and actual
GPU execution remain outside those proofs. Custom source equivalence also doesn't provide an
autograd rule or an interval enclosure.

The [blueprint chapter]({{ '/blueprint/Runtime___-Autograd___-and-Interop/Custom-Tensor-Computations/' | relative_url }})
checks the examples above during the guide build. For the API, see
[`Function.run`]({{ '/docs/NN/Kernel/Function.html#Function.run' | relative_url }}) and
[`Program.run`]({{ '/docs/NN/Kernel/Tensor.html#NN.Kernel.Program.run' | relative_url }}).
