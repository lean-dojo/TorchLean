---
title: Custom Tensor Computations
---

Let's write a function in Lean and run it on a tensor. We'll start with a square in the editor,
then add parameters, sum rows and repeat an update. We'll also ask for a gradient without
rewriting the function.

LibTorch already handles our standard GPU model operations. For a different activation or a
small custom calculation, though, we'd like to write it in Lean without maintaining a second
implementation in CUDA. The examples below show which calculations we can run this way.

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
explains SDK setup. A CPU-only scalar type or format uses CPU with a diagnostic, keeping its
precision. Unsupported source for a GPU-capable type and unavailable devices still produce IO errors.

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

## Pass parameters and combine tensors

We don't want a new function name for every activation slope. We can pass the slope when we run
the calculation:

```lean
def activate (slope : Float32) (xs : Tensor Float32 [3])
    (device : NN.Backend.Device := cpu) :
    IO (Tensor Float32 [3]) :=
  (fun x => if x < 0 then slope * x else x).run xs
    (device := device)

#eval activate 0.25 signedInput
-- [-0.500000, 1.000000, 3.000000]
```

`slope` is a runtime parameter. Passing `device := gpu` uses it in the generated calculation.
Captured scalars, unsigned indices and Boolean switches are shared by all entries. We evaluate
these shared values on CPU and embed them in the GPU source. Changing one can require a new
compilation; repeated calls with identical source reuse the native cache. For a parameter that
changes every training step, we can instead read it from an input tensor in an indexed program.

For two tensors, we use `zip`. Let's blend their corresponding entries:

```lean
def blend (weight : Float32)
    (left right : Tensor Float32 [3])
    (device : NN.Backend.Device := cpu) :
    IO (Tensor Float32 [3]) :=
  (fun x y => weight * x + (1 - weight) * y).zip
    left right (device := device)

#eval blend 0.25 input signedInput
-- [-1.250000, 1.250000, 3.000000]
```

Both inputs must have the same shape, and the result keeps it. We don't silently broadcast or
truncate either tensor. A scalar `fun x y => ...` reads corresponding entries; an indexed
`Program` below can read different positions and different input shapes. CPU supports the same
FloatLib scalar formats as `run`; GPU also supports configured binary arithmetic.

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

def rowSum (width : UInt64) : Program Float32 :=
  Program.of (fun (read : Reader Float32) (row : UInt64) =>
    iterate (fun column acc => do
      let x ← read 0 (row * width + column)
      pure (acc + x)) width.toNat 0 0)

#eval (rowSum 3).run (Arguments.empty.push matrix) [2]
-- [6.000000, 15.000000]

def rowsOnGpu : IO (Tensor Float32 [2]) :=
  (rowSum 3).run (Arguments.empty.push matrix) [2] (device := gpu)
```

`width` sets both the row stride and the step count. For this matrix we pass three; another
width uses the same definition. `iterate` starts at column zero with an accumulator of zero.
The output shape `[2]` asks for one sum per row. A zero width returns the initial zero without
reading the input.

Output rows can run independently. Within each row, additions keep the order we wrote:
floating-point addition is not associative, so a different reduction tree could change the
answer. This is a small example of indexed programming; for an ordinary reduction we'd usually
use the existing tensor API.

## Repeat a numerical update

Loops also let us simulate a small dynamical system. For each starting value, we'll repeat
the logistic update `x ↦ rate * x * (1 - x)`:

```lean
def evolve (rate : Float32) (steps : UInt64) :
    Program Float32 :=
  Program.of (fun (read : Reader Float32)
      (i : UInt64) => do
    let initial ← read 0 i
    iterate (fun _ x => pure (rate * x * (1 - x)))
      steps.toNat 0 initial)

def starts : Tensor Float32 [2] := [0.5, 0.25]

#eval (evolve 2 2).run (Arguments.empty.push starts) [2]
-- [0.500000, 0.468750]
```

Each output evolves independently on GPU; the updates for that output run in sequence. Both the
rate and the number of steps are parameters. With zero steps, we get the starting value back.
We can also write these kinds of updates as ordinary recursive functions, rather than call a
loop helper ourselves.

## Write a recursive function

Let's multiply by `x` once per step. We can keep the scalar type general:

```lean
def power {α : Type} [One α] [Mul α] (x : α) : Nat → α
  | 0 => 1
  | n + 1 => power x n * x

def powers (depth : UInt64) (input : Tensor Float32 [3])
    (device : NN.Backend.Device := cpu) : IO (Tensor Float32 [3]) :=
  (fun x => power x depth.toNat).run input (device := device)

#eval powers 3 input
-- [1.000000, 8.000000, 27.000000]
```

Adding `device := gpu` uses that same definition. The frontend reads Lean's zero and successor
equations, derives an ordered loop, and checks a proof that the loop agrees with `power`. It
doesn't build a bigger program for each additional step. Output entries can run independently;
the multiplications within one entry remain sequential.

The depth can be a literal below `2^64` or an unsigned runtime value converted with `.toNat`.
An indexed program can also choose it from the output index, giving different entries different
depths. The same `power` definition accepts FloatLib binary types; it isn't tied to FP32.

Recursive tensor reads use the same idea. Here is the row sum written recursively:

```lean
def sumPrefix (read : Reader Float32) (start : UInt64) : Nat → Except Error Float32
  | 0 => pure 0
  | n + 1 => do
      let previous ← sumPrefix read start n
      let x ← read 0 (start + n.toUInt64)
      pure (previous + x)

def recursiveRows (width : UInt64) : Program Float32 :=
  Program.of (fun (read : Reader Float32) (row : UInt64) =>
    sumPrefix read (row * width) width.toNat)

#eval (recursiveRows 3).run (Arguments.empty.push matrix) [2]
-- [6.000000, 15.000000]
```

At depth zero we make no reads. A failed read stops the recurrence; later steps cannot mask it.
The lowering proof covers those cases too. We retain the left-to-right addition order, which
matters for floating point.

Supported recursion keeps its parameters fixed and calls itself on the predecessor of its final
natural-number argument. The step can use supported branches, reads and nested recurrences.
For nested recurrences, pass the inner bounds as `UInt64` parameters as well; hard-coded inner
counts can still make proof elaboration exceed Lean's normal budget.
Tree recursion, changing recursive parameters and dynamically allocated recursive structures
are not accepted by this path. CPU Lean remains available for those programs; GPU calls report
unsupported source instead of silently moving them to CPU.

## Precision and proofs

In Lean, the calculation and its mathematical specification can share definitions.
We can reuse [mathlib](https://leanprover-community.github.io/mathlib-overview.html) rather than
rebuild the mathematics behind each model. Lean's
[tactics](https://lean-lang.org/doc/reference/latest/Tactic-Proofs/Tactic-Reference/) help construct
proofs, which its kernel checks. FloatLib supplies executable arithmetic and numerical theorems.
We don't have to change languages when a working example leads to a mathematical question.

We take inspiration from functional array languages such as
[Accelerate](https://www.acceleratehs.org/), [Futhark](https://futhark-lang.org/) and
[Dex](https://arxiv.org/abs/2104.05372), as well as
[Bend](https://github.com/bendlang/bend). Our scope is narrower than a general GPU language:
we compile the tensor calculations shown here, not arbitrary Lean programs.

Bend also has laws and checked proofs. Its current
[implementation notes](https://github.com/bendlang/bend/blob/main/WONTFIX.txt) describe native F32
operations as axioms, with bit-level definitions planned. FloatLib gives us existing infrastructure
for reasoning about rounded arithmetic. That doesn't establish a speed advantage over Bend,
and it doesn't automatically verify our native GPU execution.

`Float32` and `Float` use native binary32 and binary64 GPU arithmetic. Configured FloatLib
binary types now select compatible hardware operations automatically. For wider and custom formats,
we represent each number with integer words and do the arithmetic on the GPU. That lets us retain
digits that would disappear in FP64. Uploads and downloads carry the complete bit patterns,
not approximate native-float values.

Let's use binary128 with the same API. The parameters below choose 15 exponent bits and
112 fraction bits. `toRat?` gives each finite result as an exact rational, so inspecting it
doesn't lose the extra digits:

```lean
abbrev Wide := FloatLib.Floats.ExecFloat.Binary 15 112
def squareWide := fun (x : Wide) => x * x
def wideInput : Tensor Wide [2] := [3, 5]

def wideOnGpu : IO (Tensor Wide [2]) :=
  squareWide.run wideInput (device := gpu)

#eval do
  let output ← squareWide.run wideInput
  pure (FloatLib.Floats.ExecFloat.Binary.toRat? output[0],
    FloatLib.Floats.ExecFloat.Binary.toRat? output[1])
-- (some 9, some 25)
```

Binary16 (`ExecFloat.Binary 5 10`) and bfloat16 (`ExecFloat.Binary 8 7`) use native addition,
subtraction and multiplication on compatible GPUs. Configured IEEE binary32 and binary64 also
use native division. Half/bfloat16 division stays in software, as do FP8, custom biases and
non-IEEE encodings. The choice is made from the complete format and operation; the API doesn't change.
NaN handling preserves the configured payload rules, and separate products and sums stay separate.
Older GPUs retain software arithmetic when the needed instructions aren't available.

The configured backend supports up to 30 exponent bits and 4096 fraction bits,
with nearest-even rounding and gradual underflow. The descriptor also
selects IEEE, finite-with-NaN, unsigned-zero or fully finite conventions; exceptional values and
overflow follow that choice. Native selection is not a guarantee of a particular throughput.
Decimal and posit formats use the CPU fallback. LibTorch model buffers retain their native
binary32 or binary64 dtype; arbitrary configured precision uses the custom computation path.

The compiler's Lean proofs cover source evaluation and structured lowering. The emitted-source
interpretation theorem currently covers native FP32/FP64 under explicit arithmetic and input-buffer
contracts. Configured-format CUDA arithmetic is checked against FloatLib in the runtime regressions;
it is not kernel-verified. NVRTC, foreign memory and actual
GPU execution remain outside those proofs. Source equivalence alone doesn't provide an interval
enclosure or a backward rule for an arbitrary indexed body.

## Keep the calculation on the autograd tape

When we want the gradient, we keep the function definition and turn on recording at the call site:

```lean
import NN.Kernel
open TorchLean

def square {α : Type} [Mul α] (x : α) : α := x * x

def trainingInput : Tensor Float32 [3] := [1, 2, 3]
#eval do
  let result ← square.run trainingInput (grad := true)
  IO.println result.value
  IO.println (← result.backward)
-- [1.000000, 4.000000, 9.000000]
-- [2.000000, 4.000000, 6.000000]
```

We can also pass the whole tensor to our function. Here the output is a scalar, so the forward
result has shape `[]`. We don't have to specify storage for `Float32`:

```lean
def energy (x : Tensor Float32 [3]) : Float32 := Tensor.sum (x * x)

#eval do
  let result ← energy.run trainingInput (grad := true)
  IO.println result.value
  IO.println (← result.backward)
-- 14.000000
-- [2.000000, 4.000000, 6.000000]
```

For a generic tensor definition, we state what the element type supplies:

```lean
def tensorSquare {α : Type} [Storage α] [Mul α]
    {s : Shape} (x : Tensor α s) : Tensor α s := x * x

#eval tensorSquare.run trainingInput
-- [1.000000, 4.000000, 9.000000]
```

`[Storage α]` says how a tensor stores this unknown type; it isn't a device setting. Concrete
types already provide it, so calls infer it. The same generic square works with binary128 on CPU.

Shared `let` bindings stay shared on the existing tape. Recorded arithmetic, tensor sums, square,
ReLU, sigmoid, tanh and rank-two matrix products use the existing local VJP rules. Captured tensors are
constants, not extra trainable inputs. The frontend checks that its reconstructed arithmetic
expression equals our original function, without reassociating operations. Unsupported recordings return an error when requested;
CPU forward-only calls can still evaluate the original function.
Division, direct negation, scalar conditionals and fixed-parameter scalar recurrences can also
be recorded. Let's differentiate the activation we wrote earlier:

```lean
#eval do
  let result ← positiveSquare.run signedInput (grad := true)
  IO.println result.value
  IO.println (← result.backward)
-- [0.000000, 1.000000, 9.000000]
-- [0.000000, 2.000000, 6.000000]

def activationGradientOnGpu :=
  positiveSquare.run signedInput (device := gpu) (grad := true)
```

We record only the chosen branch, so an unused branch can't introduce an invalid division or
gradient. On GPU, choosing a scalar branch synchronizes its value back to Lean. The arithmetic
and backward operations for that branch stay on GPU. The returned gradient follows this path;
it isn't a promise of differentiability at every branch boundary.

The recursive `power` works with recording too:

```lean
#eval do
  let result ← (fun (x : Float32) => power x 3).run trainingInput (grad := true)
  IO.println result.value
  IO.println (← result.backward)
-- [1.000000, 8.000000, 27.000000]
-- [3.000000, 12.000000, 27.000000]
```

Here each recursive step adds ordinary operations to the tape. Backward visits those operations
in reverse. With recording, we retain intermediate values and launch operations per step; the
forward-only GPU call can instead use one compiled loop. The step count must be independent of
the input we're differentiating.
The recorder also accepts shape-polymorphic tensor recurrences built from the same operations;
their values and gradients keep the tensor's shape and scalar type.

Backward consumes the tape, even when it fails. For tensor outputs its default seed is all ones,
meaning the derivative of their sum. Pass `result.backward seed` to supply different output gradients,
or `result.close` to release a recording without backward. These checks don't verify foreign GPU
instructions or make rounded arithmetic a smooth real function; analytic correctness still uses
the existing derivative laws.

The GPU tape retains the input's number format: `Float32` records in binary32, `Float` in binary64,
and configured binary types retain their complete words. We can ask for a binary128 gradient too:

```lean
def wideGradientOnGpu := squareWide.run wideInput (device := gpu) (grad := true)
```

Configured GPU recording supports addition, subtraction, multiplication, division, negation and
the reshapes, slices and concatenations used by scalar traversal. The same tape accumulates
gradients in that format. This does not enable the rest of LibTorch's native operator catalogue
or native optimizer checkpoints for binary128. An unsupported operation asks us to use CPU.
Decimal, posit,
integer and other CPU-only forward types likewise use a reported CPU fallback when `gpu` is requested.
An unavailable GPU, a failed compilation or an invalid read is still an error, not a fallback.
Boolean or integer computations do not acquire an analytic derivative simply because they run on CPU.

The separate canonical `Graph.runBuffers` forward runner still requires binary32 inputs and
payloads. Its format restriction does not apply to the standalone `.run` examples above.

The [blueprint chapter]({{ '/blueprint/Runtime___-Autograd___-and-Interop/Custom-Tensor-Computations/' | relative_url }})
checks the examples above during the guide build. For the API, see
[`Function.run`]({{ '/docs/NN/Kernel/Function.html#Function.run' | relative_url }}) and
[`Program.run`]({{ '/docs/NN/Kernel/Tensor.html#NN.Kernel.Program.run' | relative_url }}).
