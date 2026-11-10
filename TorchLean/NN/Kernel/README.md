# Custom tensor computations

This API applies custom scalar calculations to the entries of a tensor. Write the calculation once
as an ordinary Lean function and choose the device when you run it: the CPU evaluates the Lean
function, and the GPU runs CUDA generated from it. Supported whole-tensor functions run on the GPU
through LibTorch. The CUDA compiler lives in [`TorchLean/CUDA/Kernel`](../../CUDA/Kernel/README.md).

- `Function.lean`: ordinary Lean functions, with `f.run`, `f.zip` and recording by `grad := true`.
- `Tensor.lean`: explicit programs on `TensorPack` inputs, with `Program.eval` and `Program.run`.
- `Graph.lean`: IR graphs on resident CUDA buffers, with `IR.Graph.runBuffers`.

```lean
import TorchLean.NN.Kernel.Function
open TorchLean

def square := fun (x : Float32) => x * x
def input : Tensor Float32 [3] := Tensor.ofFn fun i => Float32.ofNat (i.val + 1)

def onCpu : IO (Tensor Float32 [3]) := square.run input (device := .cpu)
def onGpu : IO (Tensor Float32 [3]) := square.run input (device := .cuda)

#eval square.run input (device := .cpu)
-- [1.000000, 4.000000, 9.000000]
```

Both results keep the input shape, and the CPU is the default device. Tensors print with `#eval`
or `IO.println`, with no conversion to a list or array. For the GPU, the frontend recognizes
supported source during elaboration and generates CUDA, which NVRTC compiles at run time. A scalar
type or format without a GPU encoding runs on the CPU with a diagnostic and keeps its precision.
An unsupported function for a GPU-capable type, an unavailable GPU and an unsupported device are
IO errors. GPU execution needs a CUDA-enabled LibTorch build and a visible device.

The [website example](../../../home_page/examples/custom-computations/index.md) starts from this
square, then adds parameters, several inputs, row sums, repeated updates, recursion, binary128
arithmetic and gradients. Lean checks its CPU outputs and device-selecting definitions in the
blueprint's [Custom Tensor Computations][guide] chapter.

[guide]: ../../../home_page/blueprint/TorchLeanBlueprint/Guide/Ch2_Frontend/CustomComputations.lean

## Recording a calculation for differentiation

We can record an ordinary function without changing its definition:

```lean
import TorchLean.NN.Kernel.Function
open TorchLean

def square {α : Type} [Mul α] (x : α) : α := x * x

def input : Tensor Float32 [3] := [1, 2, 3]
#eval do
  let result ← square.run input (grad := true)
  IO.println result.value
  IO.println (← result.backward)
-- [1.000000, 4.000000, 9.000000]
-- [2.000000, 4.000000, 6.000000]
```

Whole-tensor functions use the same call.

```lean
def energy (x : Tensor Float32 [3]) : Float32 :=
  Tensor.sum (x * x)

#eval do
  let result ← energy.run input (grad := true)
  IO.println result.value
  IO.println (← result.backward)
-- 14.000000
-- [2.000000, 4.000000, 6.000000]
```

Recording uses the eager autograd tape and its local VJP rules, with no second differentiation
engine. It supports arithmetic with division and direct negation, shared `let` bindings, tensor
sums, square, ReLU, sigmoid, tanh and rank-two matrix products. A scalar conditional that depends
on the input records only the chosen branch. A recurrence with fixed parameters and a final
natural-number count records each step, and the count must not depend on differentiable inputs.
Captured tensors are constants, and backward differentiates the supplied input. Lean's kernel
checks that a recipe rebuilt from the source's own functions equals the original function, with
no reassociation. Analytic correctness rests on the local derivative laws, and no theorem covers
the GPU instructions.

Backward consumes the recording, also when it fails. Its default cotangent is all ones, so it
differentiates the sum of the output entries. Pass a tensor to `result.backward seed` for another
cotangent, and call `result.close` when backward is not needed. Closing twice is harmless, and
backward after consumption or close is an error.

The element type stays a parameter. The GPU tape keeps native binary32 and binary64 values and
complete configured binary words, binary128 included. Configured recording supports addition,
subtraction, multiplication, division, negation, and the reshape, slice and concatenation
operations of scalar traversal (slices and concatenation along the first axis). Other native
model operations and native optimizer checkpoints do not accept encoded formats, and an
unsupported operation reports an error that asks for CPU execution. Scalar branch decisions
synchronize to Lean, while their arithmetic and backward operations stay on the GPU. A recorded
recurrence launches ordinary tape operations at each step and keeps the intermediates for
backward, where the forward-only path runs one fused loop. Branch derivatives follow the selected
path, and at a branch boundary they need not be mathematical derivatives.

CPU-only carriers run on the CPU with a diagnostic, and an unavailable GPU is an error. Without
`grad := true`, `.run` returns only the forward tensor. A function with no supported recording
runs on the CPU, and asking for its gradient is an error. Arbitrary indexed bodies and `.zip` get
no automatic backward rules from their source-equivalence proofs. For an indexed body,
`Program.gradient` builds the gradient as another kernel, proved over the reals, and the tape does
not record it ([derivative kernels](../../CUDA/Kernel/README.md#derivatives)).

## Recursive functions

Use ordinary Lean recursive syntax. The scalar type stays a parameter:

```lean
def power {α : Type} [One α] [Mul α] (x : α) : Nat → α
  | 0 => 1
  | n + 1 => power x n * x

def powers (depth : UInt64) (input : Tensor Float32 [3])
    (device : TorchLean.Backend.Device := .cpu) : IO (Tensor Float32 [3]) :=
  (fun x => power x depth.toNat).run input (device := device)
```

The CPU runs the original function. On the GPU, each tensor entry gets one ordered loop, which
`iterate_eq_recurrence` connects to the recursive equations. The same `power` works with
configured binary types through the configured arithmetic backend. Use a literal depth below
`2^64` or a `UInt64` converted with `UInt64.toNat`, so that an unbounded runtime `Nat` is never
narrowed silently. For nested recurrences, pass the inner bounds as `UInt64` parameters too.
Hard-coded inner counts can exceed Lean's normal elaboration budget even when their loop form is
recognized.

## Indexed calculations

For a binary scalar function, `f.zip left right (device := .cuda)` combines two tensors of one
shape without manual readers. For example,
`(fun (x y : Float32) => 0.25 * x + 0.75 * y).zip left right` blends two tensors on the CPU, and
adding the device argument selects generated CUDA. Both inputs and the result share one shape, so
Lean rejects mismatched shapes.

When an output reads several inputs or accumulates, write a `Program`:

```lean
import TorchLean.NN.Kernel.Function
open TorchLean.Kernel TorchLean
open scoped TorchLean.Kernel

def squared : Program Float32 := Program.of (fun (read : Reader Float32) (i : UInt64) => do
  let x ← read 0 i
  pure (x * x))

def input : Tensor Float32 [3] := Tensor.ofFn fun i => Float32.ofNat (i.val + 1)
def inputs : TensorPack Float32 [[3]] := .singleton input
def onCpu : IO (Tensor Float32 [3]) := squared.run inputs [3] (device := .cpu)
def onGpu : IO (Tensor Float32 [3]) := squared.run inputs [3] (device := .cuda)
```

Readers address operands by literal input number and row-major `UInt64` index, and an
input-number error is distinct from an address error. Empty output tensors make no reads.
`iterate` folds start at zero, and their counts are literal naturals below `2^64` or unsigned
values converted with `.toNat`. Accumulation keeps its order and does not assume that
floating-point addition is associative. Unsigned arithmetic keeps Lean's wraparound and
division-by-zero conventions.

`Program.eval` is the pure checked evaluator, and `Program.eval_eq_reference` proves that it
agrees with the source calculation for every set of tensor arguments and output shape, failures
included. Host arrays and resident buffers stay inside the native bridge, and callers use `.run`
with tensors. The resident CUDA graph runner `IR.Graph.runBuffers` runs graphs of registered
operators on binary32 device buffers: its input buffers must be binary32, its payloads are
`Float32` and its custom programs native binary32. It takes frozen, index, Boolean and integer
inputs as host tensors.

## Graph integration

In canonical IR a custom operation is the operator `Op.custom name inputs output`, whose ports are
its signature: a value port of each input shape and one of the output shape. The payload holds its
checked program at the node's id (`Payload.custom?`), and `Graph.check` checks each reference
against those ports. The resident runner in `Graph.lean` (`IR.Graph.runBuffers`) runs a custom
operation by its checked program, compiled by NVRTC, and every other operator by its CUDA entry in
`LibTorch.Kernel.ofOp`, with index, Boolean and integer values passed between nodes. The graph, the
payloads and the entries it needs are checked before execution, and an entry's own checks run when
it records, after earlier nodes have run. Unsupported operations are rejected. The runner is forward
only: each entry records on a temporary tape that is dropped, and no backward pass runs. Its softmax
is the tape's formula (the row maximum subtracted, the exponentials summed and divided), which need
not agree with ATen's softmax bit for bit, and no theorem covers the native execution. Forward-only
lowering and PyTorch export do not export arbitrary custom bodies.

The CUDA entries for GELU, SiLU, softplus, `exp`, `log`, `sin`, `cos`, sigmoid and `tanh` compute
through ATen with the CUDA vendor math library, GELU by its staged formula with vendor
transcendental operations. Their results need not equal the Lean reference bit for bit, the CPU
reference is the specification, and their capsule reports carry this qualification. In a
15-input binary32 GELU sample, 12 outputs agree with the reference and 3 differ by one ulp.
`remainder`, `fmax` and `fmin` have no registered CUDA entry, so the CUDA tape refuses them.

Source correspondence for a custom body is neither a real-enclosure theorem nor a derivative
theorem. IBP and CROWN derive no bounds from a custom body on the strength of its
source-evaluation proof or a supplied box alone. A custom gradient needs a separately proved rule
and its runtime integration.
