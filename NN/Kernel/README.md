# Custom tensor computations

Write the calculation once, then choose a device when you run it. Supported GPU tensor operations
already use LibTorch; CPU execution uses Lean. This API applies custom scalar calculations to
tensor entries.

```lean
import NN.Kernel
open TorchLean

def square := fun (x : Float32) => x * x
def input : Tensor Float32 [3] := Tensor.ofFn fun i => Float32.ofNat (i.val + 1)

def onCpu : IO (Tensor Float32 [3]) := square.run input (device := cpu)
def onGpu : IO (Tensor Float32 [3]) := square.run input (device := gpu)

#eval square.run input (device := cpu)
-- [1.000000, 4.000000, 9.000000]
```

Both results retain the input shape. CPU is the default and evaluates the original Lean function.
Tensors already print directly with `#eval` or `IO.println`; no list or array conversion is needed.
GPU recognizes supported source during elaboration, generates CUDA and executes it with NVRTC.
A CPU-only scalar type or format runs on CPU with a diagnostic, preserving its precision.
An unsupported function for a GPU-capable type, unavailable GPU or unsupported device still produces
an IO error. GPU execution needs a CUDA-enabled LibTorch build and a visible device.

The [website example](../../home_page/examples/custom-computations/index.md) walks through this
API. Its CPU outputs and device-selecting definitions are compiler-checked in the blueprint's
[Custom Tensor Computations](../../home_page/blueprint/TorchLeanBlueprint/Guide/Ch2_Frontend/CustomComputations.lean)
chapter.

## Precision and supported source

CPU accepts ordinary scalar functions using TorchLean's storage instances, including FloatLib
arithmetic. GPU uses hardware binary32/binary64 for `Float32`/`Float`. Configured binary
`ExecFloat.Binary` types select compatible native operations automatically; binary128 and custom
formats use integer-limb arithmetic. Complete words are uploaded and downloaded without
passing through native floats. The scalar annotation selects precision; tensor shapes stay the same.

The configured backend uses nearest-even rounding and gradual underflow. Selection checks the
exponent and fraction widths, bias, encoding, operation and GPU architecture. Standard IEEE binary32
and binary64 use native addition, subtraction, multiplication and division. Binary16 uses native
addition, subtraction and multiplication on SM 5.3 or newer; bfloat16 uses native arithmetic on
SM 8.0 or newer. Older devices keep the software path. Binary16/bfloat16 division stays in software
to avoid relying on widened division followed by another rounding. NaN payloads and invalid operations
follow the configured rules rather than a device-specific NaN convention. Separate multiplications
and additions are not fused. These choices are internal; users still select only the scalar and device.

The descriptor selects
IEEE, finite-with-NaN, unsigned-zero or fully finite conventions, including exceptional values and
overflow behavior. It supports up to 30 exponent bits and 4096 fraction bits.
Decimal and posit formats use the CPU fallback. This is custom computation support,
not an extension of native LibTorch arithmetic to arbitrary precisions. The foreign arithmetic
is checked against FloatLib but is not kernel-verified; native selection is not a throughput guarantee.

The instruction choices follow NVIDIA's
[PTX arithmetic definitions](https://docs.nvidia.com/cuda/parallel-thread-execution/).
On Ampere, bfloat16 addition and multiplication use the same one-rounding FMA identities as CUDA's
bfloat16 intrinsics; this does not fuse consecutive operations from the Lean program.

The GPU frontend supports scalar arithmetic, comparisons, local bindings and lazy conditionals.
Functions may capture scalar parameters, `UInt64` loop bounds and Boolean switches supplied at
runtime. Values independent of the output index and loop variables are evaluated on CPU and
embedded in the generated source. Changing them can require a new NVRTC compilation; the cache
reuses identical source. For frequently changing values, indexed input tensors avoid this
specialization.

The explicit program interface also supports checked indexed reads and bounded sequential folds.
Named scalar recurrences with a final natural-number argument can use the same loop backend.
They keep their parameters fixed and call themselves on the predecessor. The frontend uses Lean's
generated equation theorems and proves loop equivalence, including read failures. Nested supported
recurrences and helper functions are allowed. Depths can depend on the output index, not just host
parameters; the generated body is not unrolled once per recursive step.

It does not compile general tree recursion, changing recursive state, dynamically allocated
recursive structures, transcendental GPU primitives or automatic gradients for arbitrary indexed
programs. Recognition has a finite
traversal budget. If a recognized expression's equivalence
cannot be proved, it is rejected rather than accepted through an axiom.

## Recording a calculation for differentiation

We can record an ordinary function without changing its definition:

```lean
import NN.Kernel
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

Whole-tensor functions use the same call. For example:

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

This reuses the current tape and local VJP rules, not a second autodiff engine. Arithmetic,
shared `let` bindings, tensor sums, square, ReLU, sigmoid, tanh and rank-two matrix products are
supported, including division and direct negation. Scalar input-dependent conditionals record
only the chosen branch. Fixed-parameter recurrences with a final natural-number count record
each step; the count must be independent of differentiable inputs.
Captured tensors are constants; backward differentiates the supplied input. The frontend checks
that its reconstructed arithmetic expression equals the source, without reassociation. Analytic correctness
uses existing derivative laws; this is not a theorem about foreign GPU instructions.

Backward consumes the recording, including when it fails. Its default cotangent is all ones,
so it differentiates the sum of output entries. Pass a tensor to `result.backward seed` for a
different cotangent. Call `result.close` if backward isn't needed. Closing twice is harmless;
backward after consumption or close is an error.

The element type remains a parameter. The GPU tape retains native binary32/binary64 and complete
configured binary words, including binary128. Configured recording supports addition, subtraction,
multiplication, division, negation and the reshape/slice/concatenation operations used by scalar
traversal. Other native model operations and native optimizer checkpoints are not enabled for
encoded formats; unsupported operations report an error asking for CPU execution.
Scalar branch decisions synchronize to Lean; their arithmetic and backward operations remain on
GPU. Recorded recurrences launch ordinary tape operations per step and retain intermediates for
backward, rather than use the forward-only fused loop. Branch derivatives follow the selected
path; at a boundary they need not be mathematical derivatives.
CPU-only carriers still choose CPU with a diagnostic. An unavailable GPU remains an error.
Without `grad := true`, `.run` returns
only the forward tensor. A function without supported recording still runs on CPU, but asking
for its gradient returns an error. Arbitrary indexed bodies and `.zip` do not acquire automatic
backward rules from their source-equivalence proofs.

## Recursive functions

Use ordinary Lean recursive syntax. The scalar type stays a parameter:

```lean
def power {α : Type} [One α] [Mul α] (x : α) : Nat → α
  | 0 => 1
  | n + 1 => power x n * x

def powers (depth : UInt64) (input : Tensor Float32 [3])
    (device : NN.Backend.Device := cpu) : IO (Tensor Float32 [3]) :=
  (fun x => power x depth.toNat).run input (device := device)
```

CPU runs the original function. GPU uses one ordered loop per tensor entry, connected to those
recursive equations by `iterate_eq_recurrence`. The same `power` works with configured binary
types through the configured arithmetic backend. Choose a literal depth below `2^64`, or pass
`UInt64.toNat`; this does not silently narrow an unbounded runtime `Nat`.
For nested recurrences, pass the inner bounds as `UInt64` parameters too. Hard-coded inner counts
can still exceed Lean's normal elaboration budget, even when their loop form is recognized.

## Indexed calculations

For a binary scalar function, `f.zip left right (device := gpu)` combines equally shaped tensors
without manual readers. For example, `(fun (x y : Float32) => 0.25 * x + 0.75 * y).zip left right`
blends two tensors on CPU; adding the device argument selects generated CUDA. Both inputs and the
result share one shape, so mismatched shapes are rejected by Lean.

When an output reads several inputs or performs an accumulation, use an explicit `Program`:

```lean
import NN.Kernel
open NN.Kernel TorchLean
open scoped NN.Kernel

def squared : Program Float32 := Program.of (fun (read : Reader Float32) (i : UInt64) => do
  let x ← read 0 i
  pure (x * x))

def input : Tensor Float32 [3] := Tensor.ofFn fun i => Float32.ofNat (i.val + 1)
def inputs : Arguments Float32 [[3]] := Arguments.empty.push input
def onCpu : IO (Tensor Float32 [3]) := squared.run inputs [3] (device := cpu)
def onGpu : IO (Tensor Float32 [3]) := squared.run inputs [3] (device := gpu)
```

Readers address operands by literal input number and row-major `UInt64` index. Input-number and
address errors are distinct. Empty output tensors make no reads. `iterate` folds start at zero;
counts are literal naturals below `2^64` or unsigned values converted with `.toNat`. Accumulation
order is preserved, without assuming floating-point addition is associative. Unsigned arithmetic
retains Lean's wraparound and division-by-zero conventions.

`Program.eval` is the pure checked evaluator. `Program.eval_eq_reference` proves agreement with
the source calculation for every set of tensor arguments and output shape, including failures.
Host arrays and resident buffers stay inside the native bridge; callers use `.run` with tensors.
The autograd tape retains native binary32/binary64 or complete configured words. The separate canonical
`Graph.runBuffers` forward runner still requires binary32 inputs and payloads.

## What is proved

`Program.correct` connects recognized source to typed expression evaluation.
`Target.eval_lower` proves preservation through structured lowering, including failed reads,
lazy branches and sequential accumulation. Named-variable allocation and emitted statement
semantics are checked in `Cuda/Correctness.lean`, `Cuda/Statements.lean` and `Cuda/Source.lean`.
The native-dialect source theorem covers the signature, output guard and final write under explicit
scalar arithmetic and input-buffer contracts. It uses only Lean's standard axioms. Configured
formats reuse that renderer, but their bundled CUDA arithmetic is not kernel-verified.

These theorems do not verify NVIDIA's compiler, physical memory behavior, GPU execution or LibTorch.
Native error recording and concurrent execution remain external trust boundaries. The runtime
keeps borrowed inputs alive, checks bounds errors before exposing output and caches a bounded
number of compiled modules per process.

## Graph integration

Canonical IR stores a custom-operation signature and checked body in the payload. Shape inference
checks that signature. The resident runner in `Graph.lean` can combine custom bodies with its
supported LibTorch operations and validates every node before execution. Unsupported operations
are rejected. Forward-only lowering and PyTorch export do not export arbitrary custom bodies.

Custom source correspondence is not a real-enclosure or derivative theorem. IBP and CROWN cannot
derive bounds from a custom body solely because it has a source-evaluation proof or supplied box.
Custom gradients require a separate proved rule and runtime integration.

## Design references

The existing certified einsum compiler is TorchLean's precedent for producing a computation and
a proof of agreement with reference semantics. Related embedded/functional compiler designs include
[Accelerate](https://www.acceleratehs.org/documentation/users-guide/language.html),
[Futhark](https://futhark-lang.org/publications/pldi17.pdf) and
[Dex](https://arxiv.org/abs/2104.05372). GPU compilation uses
[NVIDIA NVRTC](https://docs.nvidia.com/cuda/nvrtc/index.html).
