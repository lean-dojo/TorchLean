# CUDA compiler for custom computations

This directory holds the certified compiler behind custom tensor computations. `Elab.lean`
recognizes supported Lean scalar functions during elaboration, `Program.lean` keeps each function
with its typed `Expr.lean` expression and a proof that both agree, and `Target.lean`, `Emit.lean`
and `Emit/` lower that expression to CUDA source. `Launch.lean` compiles the source with NVRTC
and launches it, and `Scalar.lean` holds the exact scalar encodings. `Tangent.lean`,
`Gradient.lean` and `Map.lean` derive and prove the derivative programs of the section
[Derivatives](#derivatives), and `Table.lean` computes unary functions inside kernels by checked
table searches ([Tables](#tables)). Recursion over trees is in [`Tree/`](../Tree/README.md).

`f.run`, `f.zip` and `Program.run` live in [`TorchLean/NN/Kernel`](../../NN/Kernel/README.md).

## Precision and supported source

The CPU accepts ordinary scalar functions over TorchLean's storage instances, FloatLib arithmetic
included. On the GPU, `Float32` and `Float` use hardware binary32 and binary64. Configured binary
`ExecFloat.Binary` types select compatible native operations automatically, and binary128 and
custom formats use integer-limb arithmetic. Complete words are uploaded and downloaded without
passing through native floats. The scalar annotation selects the precision, and tensor shapes stay
the same.

The configured backend rounds to nearest even and underflows gradually. Selection checks the
exponent and fraction widths, the bias, the encoding, the operation and the GPU architecture.
Standard IEEE binary32 and binary64 use native addition, subtraction, multiplication and division.
Binary16 uses native addition, subtraction and multiplication on SM 5.3 and above, and bfloat16 on
SM 8.0 and above. Devices below those versions use the software path. Binary16 and bfloat16
division stays in software, because a widened division followed by narrowing can round twice. NaN
payloads and invalid operations follow the configured rules whatever the device's NaN convention.
Separate multiplications and additions are never fused. The choice of path is internal: callers
select only the scalar type and the device.

The format descriptor selects the IEEE, finite-with-NaN, unsigned-zero or fully finite
convention, which fixes the exceptional values and the overflow behavior. GPU arithmetic supports
up to 30 exponent bits and 4096 fraction bits. Wider formats, and decimal and posit formats, run on
the CPU fallback. Configured formats apply only to custom computations, and native LibTorch
arithmetic keeps its own dtypes. The device arithmetic is tested against FloatLib and has no Lean
proof, and choosing a native instruction says nothing about throughput.

The instruction choices follow NVIDIA's
[PTX arithmetic definitions](https://docs.nvidia.com/cuda/parallel-thread-execution/). On SM 8.x
devices, bfloat16 addition and multiplication use the FMA identities of CUDA's bfloat16
intrinsics, `a * 1 + b` and `a * b + (-0)`, each with a single rounding. Consecutive operations of
the Lean program stay separate.

The GPU frontend supports scalar arithmetic, comparisons, local bindings and lazy conditionals.
Functions may capture scalar parameters, `UInt64` loop bounds and Boolean switches supplied at run
time. Values that depend on neither the output index nor a loop variable are evaluated on the CPU
and embedded in the generated source. Changing them can force another NVRTC compilation, and the
cache reuses identical source. Pass values that change often as indexed input tensors to avoid
this specialization.

`Program` also supports checked indexed reads and bounded sequential folds. Named scalar
recurrences whose last argument is a natural number can use the same loop backend if they keep
their other parameters fixed and call themselves on the predecessor. The frontend uses Lean's
generated equation theorems to prove the loop equivalent to the recurrence, read failures
included. Nested supported recurrences and helper functions are allowed. Depths can depend on the
output index as well as on host parameters, and the generated body is a loop, with no unrolling
per recursive step.

`Program.of` does not compile general tree recursion, changing recursive state or dynamically
allocated recursive structures. For a recursive function over an inductive type, `Fold.of` in
[`Tree/`](../Tree/README.md) builds a fold that runs level by level on the GPU. Kernel expressions
have no transcendental primitives, and tables give them FloatLib's correctly rounded functions in
formats of at most 20 bits ([Tables](#tables)). Recognition has a finite traversal budget. A
recognized expression whose equivalence the frontend cannot prove is rejected, and no axiom stands
in for the proof.

GPU compilation also checks that the arithmetic in this proof is the arithmetic of TorchLean's
selected scalar encoding, so a local redefinition of `+` cannot turn silently into CUDA addition.
The frontend rejects overridden operations and variable scalar dictionaries, which CPU functions
may use. For GPU execution `Program.of` accepts only TorchLean's own three `Scalar` instances. A
`Program` built by hand around another `Scalar` dictionary is an extension trust boundary, and no
proof covers its device implementation.

## Derivatives

`Program.tangent` (`Tangent.lean`) and `Program.gradient` (`Gradient.lean`) differentiate the
typed expression of a program, so a custom operation gets its derivatives without a hand-written
backward pass. For a program `f` that reads `k` inputs, `f.tangent k` reads the inputs as
operands `0` to `k - 1` and their tangents as operands `k` to `2k - 1`, and computes the
derivative of every output in that direction. `f.gradient k j n` reads the cotangent of the `n`
outputs as operand `k` and computes the gradient in input `j`. Both are programs, so `run`
executes them on the CPU and, through the same emitter, on the GPU:

```lean
def quotient : Kernel.Program Float32 :=
  Program.of (fun (read : Kernel.Reader Float32) (i : UInt64) => (do
    let a ← read 0 i
    let b ← read 1 i
    pure (a * b / (a * a + 1)) : Except Kernel.Error Float32))

def gradientInA (a b cot : Tensor Float32 [16]) : IO (Tensor Float32 [16]) :=
  (quotient.gradient 2 0 16).run (.cons a (.cons b (.cons cot .nil))) [16]
```

At `a = -2`, `b = 2` and a cotangent of one, the entry of `gradientInA` is `-0.24000001`, and the
real gradient is `b (1 - a²) / (a² + 1)² = -0.24`.

The tangent follows the expression. A `let` binds the tangent next to its value, a conditional
keeps the primal predicate, a product or a quotient uses its rule in kernel arithmetic, and a
fold carries a tangent accumulator. The tangent of a product copies both factors, so a chain of
`p` products, such as `x₁ x₂ ⋯ x_p` written left to right, has a tangent of `(p² + 5p) / 2`
arithmetic operations. Entry `a` of a gradient runs the tangent of every output with the tangent
of input `j` set to one at `a` and zero elsewhere. A gradient of `m` entries costs `m n` tangent
evaluations, and no tape is kept. A fold whose tangent reads its accumulator, such as a running
product or a body that binds a value of the accumulator with `let`, recomputes the accumulator at
each step, which makes its tangent quadratic in the count.

In a binary format the zero tangents of the other entries do not make their terms vanish: a
partial derivative that is infinite or NaN times a zero tangent is NaN. So one such partial
derivative anywhere makes every entry of the gradient NaN. For `x * x` with one infinite entry
the tangent is right at every entry, and the gradient is NaN at every other entry. The same holds
for `1 / x` with one zero entry. The theorems below are about the reals, where every such term is
zero.

Over the reals, `Expr.hasFDerivAt_evaluate_tangent` in `Tangent/FDeriv.lean` proves that the
tangent computes the Fréchet derivative of the evaluation at a point where the evaluation
succeeds nearby and is regular: no scalar division has a zero divisor, and no scalar comparison
has equal operands unless both are constants. `Expr.gradient_eq_adjoint` in `Gradient/FDeriv.lean`
proves that the gradient computes the adjoint of that derivative at the cotangent.
`Expr.map_gradient_eq_adjoint` and its variants in `Map.lean` prove the same of a program over
`Float32` or binary32 read as reals. `TorchLeanTest/Verification/KernelMap.lean` proves with them
that, read as reals, the gradient of `quotient` computes `b (1 - a²) / (a² + 1)²` times the
cotangent in input zero and `a / (a² + 1)` times it in input one, at every point. A `Float32` run
evaluates the same expressions with rounding, and no theorem bounds its error.
`TorchLeanTest/CUDA/KernelGradient.lean` checks the derivatives of four programs on the CPU and
compares their GPU runs with the CPU bits.

## Tables

`Table.lean` lets a kernel compute a unary function of a scalar type of at most `Table.maxWidth`
bits by a binary search over keys, unrolled into loads and comparisons. `Table.make A f` evaluates
`f` at every word of the type and returns a table only when the table gives `f` at every word, bit
for bit (`Table.eval_eq_of_check`). A `Table.Of A f` carries that proof, and `Table.Of.eval_apply`
shows that the table, with its keys and values read from two inputs of the kernel, computes `f`
inside any kernel expression. In an IEEE format a table of FloatLib's `exp`, `expMinus1`, `log` or
`logPlus1` gives the exact result rounded once to nearest at a finite argument in the domain whose
result is below the overflow threshold (`Table.Of.toReal_exp`, `Table.Of.toReal_expMinus1`,
`Table.Of.toReal_log`, `Table.Of.toReal_logPlus1`). The tests build these tables in e4m3, e5m2,
e4m3fn, e4m3fnuz, a 12-bit format, binary16 and bfloat16. `Table.maxWidth` is 20, so binary32 and
binary64 get no table. Functions whose results at the NaNs and zeros take none of the forms of
`Table.Special`, such as the absolute value, get no table either. `Table.Of.runDevice` runs the
kernel of a table over an array on the GPU. That run is trusted like the other launches, and the
tests compare it with the function at every word.

## What is proved

`Program.correct` connects recognized source to typed expression evaluation. `Target.eval_lower`
proves that structured lowering preserves it, failed reads, lazy branches and sequential
accumulation included. The allocation of named variables and the semantics of the emitted
statements are checked in `Emit/Correctness.lean`, `Emit/Statement.lean` and `Emit/Source.lean`.
The native-dialect source theorem `denotes_emit` covers the signature, the output guard and the
final write, for any scalar arithmetic and for input buffers whose reads are bounds-checked loads.
It uses only Lean's standard axioms. Configured formats reuse that renderer, and their bundled
CUDA arithmetic has no Lean proof.

The correctness theorems say nothing about NVIDIA's compiler, physical memory, GPU execution or
LibTorch. Native error recording and concurrent execution are external trust boundaries. The
runtime keeps borrowed inputs alive, reads the bounds record before returning an output and caches
at most sixteen compiled modules per process.

## Design references

TorchLean's einsum elaborator (`TorchLean/Tensor/Einops/Elab/Einsum.lean`) likewise pairs a
specialized computation with a proof that it agrees with the general executor. Related embedded
and functional compiler designs include
[Accelerate](https://www.acceleratehs.org/documentation/users-guide/language.html),
[Futhark](https://futhark-lang.org/publications/pldi17.pdf) and
[Dex](https://arxiv.org/abs/2104.05372). GPU compilation uses
[NVIDIA NVRTC](https://docs.nvidia.com/cuda/nvrtc/index.html).
