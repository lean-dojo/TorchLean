---
title: Custom Tensor Computations
---

Let's write a function in Lean and run it on a tensor. We'll start with a square in the editor,
then add parameters, sum rows and repeat an update. We'll also ask for a gradient without
rewriting the function. LibTorch already handles our standard GPU model operations. For a different
activation or a small custom calculation, we'd like to write it in Lean without maintaining a second
implementation in CUDA, and the examples below show which calculations we can run this way.

## Start with a square

```lean
import TorchLean.NN.Kernel.Function
import TorchLean.Tensor.Literal
open TorchLean

def square := fun (x : Float32) => x * x

def input : Tensor Float32 [3] :=
  Tensor.ofFn fun i => Float32.ofNat (i.val + 1)

#eval square.run input
-- [1.000000, 4.000000, 9.000000]
```

The result keeps the input shape, and `#eval` prints the tensor itself, so our example needs no
list conversion or buffer handling. CPU is the default.

## Run on GPU

We select the device at the call site. Its type is `Backend.Device`, and we write its constructors
with a leading dot:

```lean
def onCpu : IO (Tensor Float32 [3]) :=
  square.run input (device := .cpu)

def onGpu : IO (Tensor Float32 [3]) :=
  square.run input (device := .cuda)
```

Run `onGpu` from a CUDA-enabled build with a visible GPU (the [GPU
guide]({{ '/blueprint/Floating-Point-and-Native-Boundaries/From-A-Tensor-Operation-To-A-GPU-Kernel/' | relative_url }})
explains SDK setup). A CPU-only scalar type or format uses CPU with a diagnostic, keeping its
precision, but unsupported source for a GPU-capable type and unavailable devices produce IO errors.

CPU evaluates our original function. For GPU, a frontend recognizes scalar arithmetic, comparisons,
local bindings and conditionals, proves agreement with a typed expression, and generates CUDA,
which the native bridge compiles with NVRTC (NVIDIA's runtime CUDA compiler). Model operations such
as matrix multiplication call LibTorch's ATen primitives instead.

## Change the activation

Let's keep only the nonnegative entries and square them:

```lean
def positiveSquare := fun (x : Float32) =>
  if x < 0 then 0 else x * x

def signedInput : Tensor Float32 [3] := [-2, 1, 3]

#eval positiveSquare.run signedInput
-- [0.000000, 1.000000, 9.000000]

def activationOnGpu : IO (Tensor Float32 [3]) :=
  positiveSquare.run signedInput (device := .cuda)
```

That conditional belongs to the function we wrote. Only its chosen branch is evaluated, and the GPU
frontend preserves that choice too. Again, the result has shape `[3]`.

## Pass parameters and combine tensors

We don't want a new function name for every activation slope, so let's pass the slope when we run
the calculation:

```lean
def activate (slope : Float32) (xs : Tensor Float32 [3])
    (device : TorchLean.Backend.Device := .cpu) :
    IO (Tensor Float32 [3]) :=
  (fun x => if x < 0 then slope * x else x).run xs
    (device := device)

#eval activate 0.25 signedInput
-- [-0.500000, 1.000000, 3.000000]
```

`slope` is a runtime parameter, and passing `device := .cuda` uses it in the generated calculation.
Captured scalars, unsigned indices and Boolean switches are shared by all entries: we evaluate
these shared values on CPU and embed them in the GPU source, so changing one can require a new
compilation (though repeated calls with identical source reuse the native cache). For a parameter
that changes every training step, we can instead read it from an input tensor in an indexed
program.

For two tensors, we use `zip`. Let's blend their corresponding entries:

```lean
def blend (weight : Float32)
    (left right : Tensor Float32 [3])
    (device : TorchLean.Backend.Device := .cpu) :
    IO (Tensor Float32 [3]) :=
  (fun x y => weight * x + (1 - weight) * y).zip
    left right (device := device)

#eval blend 0.25 input signedInput
-- [-1.250000, 1.250000, 3.000000]
```

Both inputs must have the same shape, and the result keeps it (we don't silently broadcast or
truncate either tensor). A scalar `fun x y => ...` reads corresponding entries, while an indexed
`Program` below can read different positions and different input shapes. CPU supports the same
FloatLib scalar formats as `run`, and GPU also supports configured binary arithmetic.

## Read several tensors

An indexed `Program` lets an output read entries from more than one tensor. We put the tensors
in a `TensorPack`, keeping each one's shape:

```lean
open Kernel
open scoped Kernel

def addInputs : Kernel.Program Float32 :=
  Program.of (fun (read : Reader Float32) (i : UInt64) => do
    let x ← read 0 i
    let y ← read 1 i
    pure (x + y))

def inputs : TensorPack Float32 [[3], [3]] :=
  .pair input input

#eval addInputs.run inputs [3]
-- [2.000000, 4.000000, 6.000000]
```

Here `0` and `1` identify our input tensors, and `i` counts through their entries in row-major
order, with the last axis changing fastest. Reads check bounds. We ask for output shape `[3]`, and
adding `device := .cuda` selects GPU execution for the same program. Bounded folds preserve the
sequential accumulation order we write.

## Sum each row

This time each output needs several input entries, because we'll sum the rows of a matrix:

```lean
def matrix : Tensor Float32 [2, 3] := [[1, 2, 3], [4, 5, 6]]

def rowSum (width : UInt64) : Kernel.Program Float32 :=
  Program.of (fun (read : Reader Float32) (row : UInt64) =>
    iterate (fun column acc => do
      let x ← read 0 (row * width + column)
      pure (acc + x)) width.toNat 0 0)

#eval (rowSum 3).run (.singleton matrix) [2]
-- [6.000000, 15.000000]

def rowsOnGpu : IO (Tensor Float32 [2]) :=
  (rowSum 3).run (.singleton matrix) [2] (device := .cuda)
```

`width` sets both the row stride and the step count. For this matrix we pass three, but another
width uses the same definition. `iterate` starts at column zero with an accumulator of zero, and
the output shape `[2]` asks for one sum per row. A zero width returns that initial zero without
reading the input.

Output rows can run independently, but within each row the additions keep the order we wrote:
floating-point addition is not associative, so a different reduction tree could change the answer.
This is a small example of indexed programming, and for an ordinary reduction we'd usually use the
existing tensor API.

## Repeat a numerical update

Loops also let us simulate a small dynamical system. For each starting value, we'll repeat
the logistic update `x ↦ rate * x * (1 - x)`:

```lean
def evolve (rate : Float32) (steps : UInt64) :
    Kernel.Program Float32 :=
  Program.of (fun (read : Reader Float32)
      (i : UInt64) => do
    let initial ← read 0 i
    iterate (fun _ x => pure (rate * x * (1 - x)))
      steps.toNat 0 initial)

def starts : Tensor Float32 [2] := [0.5, 0.25]

#eval (evolve 2 2).run (.singleton starts) [2]
-- [0.500000, 0.468750]
```

Each output evolves independently on GPU, while the updates for that output run in sequence. Both
the rate and the number of steps are parameters (with zero steps, we get the starting value back).
We can also write these kinds of updates as ordinary recursive functions instead of calling a loop
helper ourselves.

## Write a recursive function

Let's multiply by `x` once per step. We can keep the scalar type general:

```lean
def power {α : Type} [One α] [Mul α] (x : α) : Nat → α
  | 0 => 1
  | n + 1 => power x n * x

def powers (depth : UInt64) (input : Tensor Float32 [3])
    (device : TorchLean.Backend.Device := .cpu) : IO (Tensor Float32 [3]) :=
  (fun x => power x depth.toNat).run input (device := device)

#eval powers 3 input
-- [1.000000, 8.000000, 27.000000]
```

Adding `device := .cuda` uses that same definition: the frontend reads Lean's zero and successor
equations, derives an ordered loop, and checks a proof that the loop agrees with `power`. It
doesn't build a bigger program for each additional step. Output entries can run independently, but
the multiplications within one entry remain sequential.

For the depth we can use a literal below `2^64` or an unsigned runtime value converted with
`.toNat`. An indexed program can also choose it from the output index, giving different entries
different depths. That same `power` definition accepts FloatLib binary types, because it isn't tied
to FP32.

Recursive tensor reads use the same idea. Here is the row sum written recursively:

```lean
def sumPrefix (read : Reader Float32) (start : UInt64) : Nat → Except Error Float32
  | 0 => pure 0
  | n + 1 => do
      let previous ← sumPrefix read start n
      let x ← read 0 (start + n.toUInt64)
      pure (previous + x)

def recursiveRows (width : UInt64) : Kernel.Program Float32 :=
  Program.of (fun (read : Reader Float32) (row : UInt64) =>
    sumPrefix read (row * width) width.toNat)

#eval (recursiveRows 3).run (.singleton matrix) [2]
-- [6.000000, 15.000000]
```

At depth zero we make no reads. A failed read stops the recurrence (later steps can't mask it), and
the lowering proof covers those cases too. We retain the left-to-right addition order, which
matters for floating point.

Supported recursion keeps its parameters fixed and calls itself on the predecessor of its final
natural-number argument. Its step can use supported branches, reads and nested recurrences. For
nested recurrences, pass the inner bounds as `UInt64` parameters as well, because hard-coded inner
counts can make proof elaboration exceed Lean's normal budget. This path doesn't accept tree
recursion, changing recursive parameters or dynamically allocated recursive structures, and GPU
calls report unsupported source instead of silently moving them to CPU. For recursion over a tree,
we write a fold, as in the next section.

## Recurse over a tree

For a recursive function over an inductive type of our own, we derive `Sig.Inductive` for the type
and pass the function to `Fold.of`:

```lean
import TorchLean.CUDA.Tree.Elab
import TorchLean.CUDA.Tree.Run
open TorchLean.Kernel

inductive BTree where
  | leaf
  | node (l : BTree) (x : Float) (r : BTree)
  deriving Sig.Inductive

def BTree.sum : BTree → Float
  | .leaf => 0
  | .node l x r => l.sum + x + r.sum

def sumFold := Fold.of BTree.sum

def BTree.arena (t : BTree) : Arena Float :=
  Arena.encode (σ := BTree.sig) (T := fun _ => BTree) (s := 0) t

def small : BTree := .node (.node .leaf 1 .leaf) 2 (.node .leaf 3 .leaf)

#eval sumFold.eval (Reader.ofArrays #[]) small.arena
-- Except.ok 6.000000
```

`Arena.encode` stores the tree by levels, with the children of level `r` on level `r + 1`. The fold
computes the levels from the deepest up with one row kernel, and Lean's kernel checks the
certificate that `Fold.of` builds. `Fold.eval_encode` then gives the Lean function on every tree
with fewer than `2^64` nodes, and `Fold.run` launches the row kernel once per level over an arena
in device memory:

```lean
theorem sumFold_eval (t : BTree)
    (h : Sig.Inductive.count (σ := BTree.sig) (T := fun _ => BTree) (s := 0) t < UInt64.size) :
    sumFold.eval (Reader.ofArrays #[]) t.arena = .ok t.sum :=
  sumFold.eval_encode _ t h

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
native formats. NVRTC, the launches, the bytes and the agreement of the device arithmetic with
Lean's are trusted, and the
[tree guide](https://github.com/lean-dojo/TorchLean/blob/main/TorchLean/CUDA/Tree/README.md) lists
the theorems.

We measured the examples of `TorchLeanExamples/Tree` with `tree_bench` on one A100, in
nanoseconds per row at 2^22 inner nodes (8,388,609 rows), on a shared machine with a load average
of 47 to 62 on its 96 cores. Each GPU number is the second of two runs, as the first compiles the
kernels:

| | Lean function | host schedule | GPU |
| --- | ---: | ---: | ---: |
| fold: sum of a tree | 27.8 | 674.4 | 0.1 |
| fold: a term of 2^22 operations | 38.0 | 704.9 | 0.2 |
| hylomorphism: sum by halves | 36.8 | 3,924.2 | 130.2 |
| hylomorphism: Fibonacci 33 | 13.0 | 2,318.8 | 1.4 |
| hylomorphism: midpoint rule | 66.1 | 4,955.1 | 1.3 |
| unfold: balanced tree of 2^22 entries | 39.3 | 4,451.4 | 62.9 |
| unfold: binomial tree of order 23 | 91.4 | 3,565.9 | 1.4 |

The GPU column of a fold runs over an arena already in device memory, and the upload costs 880 ns
per row. The sum by halves and the balanced tree pack their input on the host at every run, which
makes them slower on the GPU than the Lean function. A sum over an arena that an unfold leaves in
device memory takes 0.1 ns per row. Kernels also compute FloatLib's correctly rounded functions by
checked tables in formats of at most 20 bits. Over 2^20 values, `Table.Of.runDevice` computes `exp`
in binary16 at 221 ns per value, with the encoding, the transfers and the decoding included, and
FloatLib's `exp` takes 103 µs per value on the host in the same run.

## Precision and proofs

In Lean, the calculation and its mathematical specification can share definitions, and we can reuse
[mathlib](https://leanprover-community.github.io/mathlib-overview.html) instead of rebuilding the
mathematics behind each model. Lean's
[tactics](https://lean-lang.org/doc/reference/latest/Tactic-Proofs/Tactic-Reference/) help
construct proofs (which its kernel checks), and FloatLib supplies executable arithmetic and
numerical theorems. So we don't have to change languages when a working example leads to a
mathematical question.

We take inspiration from functional array languages such as
[Accelerate](https://www.acceleratehs.org/), [Futhark](https://futhark-lang.org/) and
[Dex](https://arxiv.org/abs/2104.05372), as well as
[Bend](https://github.com/bendlang/bend). Our scope is narrower than a general GPU language:
we compile the tensor calculations shown here, not arbitrary Lean programs.

Bend also has laws and checked proofs. Its [implementation
notes](https://github.com/bendlang/bend/blob/main/WONTFIX.txt) describe native F32 operations as
axioms. FloatLib gives us existing infrastructure for reasoning about rounded arithmetic. That
doesn't establish a speed advantage over Bend, and it doesn't automatically verify our native GPU
execution.

`Float32` and `Float` use native binary32 and binary64 GPU arithmetic, and configured FloatLib
binary types select compatible hardware operations automatically. For wider and custom formats, we
represent each number with integer words and do the arithmetic on the GPU, which lets us retain
digits that would disappear in FP64. Uploads and downloads carry the complete bit patterns, not
approximate native-float values.

Let's use binary128 with the same API (the parameters below choose 15 exponent bits and 112
fraction bits). `toRat?` gives each finite result as an exact rational, so inspecting it doesn't
lose the extra digits:

```lean
abbrev Wide := FloatLib.Floats.ExecFloat.Binary 15 112
def squareWide := fun (x : Wide) => x * x
def wideInput : Tensor Wide [2] := [3, 5]

def wideOnGpu : IO (Tensor Wide [2]) :=
  squareWide.run wideInput (device := .cuda)

#eval do
  let output ← squareWide.run wideInput
  pure (FloatLib.Floats.ExecFloat.Binary.toRat? output[0],
    FloatLib.Floats.ExecFloat.Binary.toRat? output[1])
-- (some 9, some 25)
```

Binary16 (`ExecFloat.Binary 5 10`) and bfloat16 (`ExecFloat.Binary 8 7`) use native addition,
subtraction and multiplication on compatible GPUs, and configured IEEE binary32 and binary64 also
use native division. Half/bfloat16 division stays in software, as do FP8, custom biases and
non-IEEE encodings. We make that choice from the complete format and operation, and the API doesn't
change. NaN handling preserves the configured payload rules, and separate products and sums stay
separate. Older GPUs retain software arithmetic when the needed instructions aren't available.

For configured formats, the backend supports up to 30 exponent bits and 4096 fraction bits, with
nearest-even rounding and gradual underflow. The descriptor also selects IEEE, finite-with-NaN,
unsigned-zero or fully finite conventions, and exceptional values and overflow follow that choice.
Native selection doesn't guarantee a particular throughput. Decimal and posit formats use the CPU
fallback. LibTorch model buffers retain their native binary32 or binary64 dtype, while arbitrary
configured precision uses the custom computation path.

The compiler's Lean proofs cover source evaluation and structured lowering, and the emitted-source
interpretation theorem covers native FP32/FP64 under arithmetic and input-buffer contracts.
Configured-format CUDA arithmetic is checked against FloatLib in the runtime regressions, but it
isn't kernel-verified. NVRTC, foreign memory and actual GPU execution remain outside those proofs.
Source equivalence alone doesn't provide an interval enclosure for an arbitrary indexed body. For
derivatives, `Program.tangent` and `Program.gradient` derive tangent and gradient programs from the
body, which run like the program itself. `Expr.gradient_eq_adjoint` proves over the reals that the
gradient computes the adjoint of the derivative where the evaluation is regular: no division it
reaches has a zero divisor, and no comparison has equal operands unless both are constants.

## Keep the calculation on the autograd tape

When we want the gradient, we keep the function definition and turn on recording at the call site:

```lean
import TorchLean.NN.Kernel.Function
import TorchLean.Tensor.Literal
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

`[Storage α]` says how a tensor stores this unknown type (it isn't a device setting). Concrete
types already provide it, so calls infer it, and the same generic square works with binary128 on
CPU.

Shared `let` bindings stay shared on the existing tape, and recorded arithmetic, tensor sums,
square, ReLU, sigmoid, tanh and rank-two matrix products use the existing local VJP rules. Captured
tensors are constants, not extra trainable inputs. For each recording the frontend adds a
kernel-checked theorem that the recorded program, run by the pure interpreter, computes our
original function, without reassociating operations. An operation whose instance computes
something other than the recorded operator is an error. Requesting an unsupported recording with
`grad := true` is an elaboration error, but CPU forward-only calls can still evaluate the original
function. Division, direct negation, scalar conditionals and fixed-parameter scalar recurrences can
also be recorded, so let's differentiate the activation we wrote earlier:

```lean
#eval do
  let result ← positiveSquare.run signedInput (grad := true)
  IO.println result.value
  IO.println (← result.backward)
-- [0.000000, 1.000000, 9.000000]
-- [0.000000, 2.000000, 6.000000]

def activationGradientOnGpu :=
  positiveSquare.run signedInput (device := .cuda) (grad := true)
```

We record only the chosen branch, so an unused branch can't introduce an invalid division or
gradient. On GPU, choosing a scalar branch synchronizes its value back to Lean, while the
arithmetic and backward operations for that branch stay on GPU. The returned gradient follows this
path and doesn't promise differentiability at every branch boundary.

Recording works for the recursive `power` too:

```lean
#eval do
  let result ← (fun (x : Float32) => power x 3).run trainingInput (grad := true)
  IO.println result.value
  IO.println (← result.backward)
-- [1.000000, 8.000000, 27.000000]
-- [3.000000, 12.000000, 27.000000]
```

Here each recursive step adds ordinary operations to the tape, and backward visits those operations
in reverse. With recording, we retain intermediate values and launch operations per step, whereas
the forward-only GPU call can use one compiled loop. The step count must be independent of the
input we're differentiating. The recorder also accepts shape-polymorphic tensor recurrences built
from the same operations, and their values and gradients keep the tensor's shape and scalar type.

Backward consumes the tape, even when it fails. For tensor outputs its default seed is all ones
(meaning the derivative of their sum). Pass `result.backward seed` to supply different output
gradients, or `result.close` to release a recording without backward. These checks don't verify
foreign GPU instructions or make rounded arithmetic a smooth real function. Analytic correctness
uses the existing derivative laws.

On GPU, the tape retains the input's number format: `Float32` records in binary32, `Float` in
binary64, and configured binary types retain their complete words. We can ask for a binary128
gradient too:

```lean
def wideGradientOnGpu := squareWide.run wideInput (device := .cuda) (grad := true)
```

Configured GPU recording supports addition, subtraction, multiplication, division, negation and the
reshapes, slices and concatenations used by scalar traversal, and the same tape accumulates
gradients in that format. This doesn't enable the rest of LibTorch's native operator catalogue or
native optimizer checkpoints for binary128, and an unsupported operation asks us to use CPU.
Decimal, posit, integer and other CPU-only forward types likewise use a reported CPU fallback when
`.cuda` is requested, but an unavailable GPU, a failed compilation or an invalid read is an error,
not a fallback. Running on CPU doesn't give Boolean or integer computations an analytic derivative.

For graphs with custom kernels, the resident CUDA runner (`TorchLean.IR.Graph.runBuffers`) runs each
operator through its entry in `LibTorch.Kernel.ofOp` and requires binary32 input buffers, `Float32`
payloads and native binary32 custom programs. That format restriction doesn't apply to the
standalone `.run` examples above.

During the guide build, the [blueprint
chapter]({{ '/blueprint/Runtime___-Autograd___-and-Interop/Custom-Tensor-Computations/' | relative_url }})
checks the examples above. For the API, see
[`Function.run`]({{ '/docs/TorchLean/NN/Kernel/Function.html#Function.run' | relative_url }}) and
[`Program.run`]({{ '/docs/TorchLean/NN/Kernel/Tensor.html#TorchLean.Kernel.Program.run' | relative_url }}).
