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
An unsupported function, unavailable GPU or unsupported device produces an IO error; there is no
automatic CPU fallback. GPU execution needs a CUDA-enabled LibTorch build and a visible device.

The [website example](../../home_page/examples/custom-computations/index.md) walks through this
API. Its CPU outputs and device-selecting definitions are compiler-checked in the blueprint's
[Custom Tensor Computations](../../home_page/blueprint/TorchLeanBlueprint/Guide/Ch2_Frontend/CustomComputations.lean)
chapter.

## Precision and supported source

CPU accepts ordinary scalar functions using TorchLean's storage instances, including FloatLib
arithmetic. GPU currently accepts `Float32` and `Float`, computing in binary32 and binary64
respectively. The scalar annotation selects precision; it does not change tensor shapes.
FloatLib's arbitrary binary, decimal and posit formats are not automatically compiled to CUDA.
That requires a software-arithmetic backend, not a conversion to binary64.

The GPU frontend supports scalar arithmetic, comparisons, local bindings and lazy conditionals.
The explicit program interface also supports checked indexed reads and bounded sequential folds.
It does not compile arbitrary recursion, supply transcendental GPU primitives or generate custom
gradients. Recognition has a finite traversal budget. If a recognized expression's equivalence
cannot be proved, it is rejected rather than accepted through an axiom.

## Indexed calculations

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
The resident graph ABI currently stores binary32.

## What is proved

`Program.correct` connects recognized source to typed expression evaluation.
`Target.eval_lower` proves preservation through structured lowering, including failed reads,
lazy branches and sequential accumulation. Named-variable allocation and emitted statement
semantics are checked in `Cuda/Correctness.lean`, `Cuda/Statements.lean` and `Cuda/Source.lean`.
The source theorem covers the signature, output guard and final write under explicit scalar
arithmetic and input-buffer contracts. It uses only Lean's standard axioms.

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
