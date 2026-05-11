import VersoManual
import VersoBlueprint

open Verso.Genre Manual

#doc (Manual) "Floating Point Semantics" =>
%%%
tag := "floats"
%%%

In TorchLean, "Float32" is not one object. It can mean an idealized rounded-real model used in
proofs, an executable IEEE-754 bit model inside Lean, a runtime `Float32` value, or a CUDA `float`
produced by a native kernel. Those meanings are related, but they are not interchangeable.

The purpose of this page is to name the numerical objects and the bridges between them.

# The Central Example

The easiest example is a dot product. Suppose a model computes:

$$`\sum_i w_i x_i`

Over the reals, this sum has one mathematical value. Over float32, the value also depends on
rounding after each operation, whether multiplication and addition are fused, and the order in which
the sum is evaluated. On a GPU, a parallel reduction may use a different tree from a CPU loop.

A useful theorem has to say which arithmetic it is about. It should state:

- which semantic scalar is used;
- which reduction or operation order is used;
- which finite/no-overflow side conditions are required;
- and which theorem, checker, or runtime agreement connects the executed value to the proof value.

That is the whole numeric discipline of this chapter.

# Four Numeric Layers

TorchLean separates four numerical layers:

- *Real specification*: `ℝ` tensors and spec functions. This is the ideal mathematical model:
  networks, losses, and verifier inequalities before rounding.
- *Rounded real proof model*: `FP32` / `NF`. This model performs real operations and rounds to the
  binary32 grid, which is the right setting for compositional error proofs.
- *Executable IEEE-style bits*: `IEEE32Exec`. This model stores raw `UInt32` binary32 values and
  includes signed zeros, infinities, NaNs, comparisons, and special rules.
- *Native/runtime execution*: Lean `Float32`, CUDA, cuBLAS, cuFFT, and external tools. These are
  fast producers of values whose agreement with Lean semantics is stated through runtime agreement,
  tests, certificates, or assumptions.

The first three layers are Lean definitions. The fourth layer is the implementation path, so claims
about it require an agreement statement with the Lean-side model. That is how TorchLean can run real
examples while still saying exactly which numerical object a theorem concerns.

# Why Both `FP32` And `IEEE32Exec` Exist

`FP32` and `IEEE32Exec` answer different questions.

`FP32` is the proof model. It is a rounded-real scalar: compute the real operation, round to the
binary32 format, and keep an explicit error bound. This is the right level for layerwise forward
error, verifier-margin transfer, and paper statements such as "the rounded execution stays within
ε of the real specification."

`IEEE32Exec` is the executable bit model. It stores raw binary32 bits and implements IEEE-style
behavior for the core operations, including signed zero, infinities, NaNs, comparisons, and
special-value propagation. This is the right level for widgets, demos, edge-case tests, and checking
what a binary32-shaped computation actually does.

The bridge theorems connect the two on the finite path. That means the inputs and result decode to
ordinary finite real values, and the operation-specific side conditions are satisfied. When a
computation leaves that finite path, the total theorems use `Option`: finite values become
`some r`, while NaN and infinity become `none`.

# API Map

These imports are the main landmarks:

```
import NN.Floats.Float32
import NN.Floats.FP32
import NN.Floats.IEEEExec
import NN.Floats.IEEEExec.BridgeFP32
import NN.Floats.IEEEExec.BridgeFP32Total
import NN.Floats.IEEEExec.BridgeInitFloat32

open TorchLean.Floats
open TorchLean.Floats.IEEE754

-- User-facing selector.
#check TorchLean.Floats.Float32Mode
#check TorchLean.Floats.F32

-- Proof model: finite rounded-real binary32 arithmetic.
#check TorchLean.Floats.FP32
#check TorchLean.Floats.FP32.toReal

-- Executable bit-level model: raw UInt32 binary32 values.
#check IEEE32Exec
#check IEEE32Exec.ofBits
#check IEEE32Exec.toBits
#check IEEE32Exec.toReal
#check IEEE32Exec.toReal?

-- Bridges from executable bits to proof semantics.
#check IEEE32Exec.toReal_add_eq_fp32Round
#check IEEE32Exec.toReal?_add_eq_ite

-- Runtime Float32 remains a named assumption boundary.
#check Float32Bridge.RuntimeFloat32MatchesIEEE32Exec
```

Read these names as the dependency graph. `F32` selects the scalar. `FP32` is the proof-side scalar.
`IEEE32Exec` is the executable scalar. `BridgeFP32` and `BridgeFP32Total` say how executable bits
become proof-side rounded reals. `BridgeInitFloat32` names what remains if a claim uses Lean's
runtime `Float32`.

# A Single Addition At Four Levels

The same expression, `a + b`, has four different readings:

- `a + b : ℝ` is exact mathematical addition.
- `FP32.add a b` is exact real addition rounded to the binary32 grid.
- `IEEE32Exec.add x y` is executable binary32 bit arithmetic with IEEE-style special cases.
- a runtime or CUDA add is a native implementation whose agreement is tested, checked, or assumed
  against a Lean-side contract.

The finite bridge theorem says the `IEEE32Exec` result agrees with the `FP32` rounded-real result
when the finite hypotheses hold:

```
import NN.Floats.IEEEExec.BridgeFP32
import NN.Floats.IEEEExec.BridgeFP32Total

open TorchLean.Floats.IEEE754

-- Finite bridge.
#check IEEE32Exec.toReal_add_eq_fp32Round

-- Total bridge.
#check IEEE32Exec.toReal?_add_eq_ite
```

That is the pattern for the rest of the chapter. Do not collapse the layers; connect them with a
named theorem, checker, or runtime agreement.

# Reductions Need An Order

Real addition is associative. Floating-point addition is not. Therefore a theorem about a float32
sum, dot product, convolution accumulation, attention score, or CUDA reduction needs an order
contract.

TorchLean states reduction facts around a finite evaluation tree rather than pretending every
parallel schedule is the same:

The [reductions API](https://github.com/lean-dojo/TorchLean/blob/main/NN/Floats/IEEEExec/Reductions.lean) contains the long-form theorem
names for sum trees and dot products. The important reading rule is shorter than the names:
soundness is stated for a specified finite evaluation tree, not for every possible reordering.

This is the numerical reason CUDA reductions are discussed in the native-boundary chapter. A fast
kernel may be a perfectly good training kernel, but a proof-quality statement needs a fixed tree or
an explicit reduction specification.

# Finite Path, Precisely

The finite path is the part of IEEE-style execution where the result can be read as an ordinary real
number.

Examples:

- a normal binary32 value is finite;
- a subnormal binary32 value is still finite, although with reduced precision;
- `+0.0` and `-0.0` are finite bit patterns, but proofs may need to know which operation produced
  them;
- overflow to `+Inf` or `-Inf` leaves the finite path;
- `0/0`, `Inf - Inf`, and `sqrt` of a negative finite value produce NaN paths, not finite paths;
- a reduction is finite only relative to a specified evaluation tree whose intermediate results
  remain finite.

The bridge files reflect this split:

- [BridgeFP32 API](https://github.com/lean-dojo/TorchLean/blob/main/NN/Floats/IEEEExec/BridgeFP32.lean) proves finite refinements such as
  `toReal_add_eq_fp32Round`, `toReal_mul_eq_fp32Round`, `toReal_div_eq_fp32Round`,
  `toReal_fma_eq_fp32Round`, and `toReal_sqrt_eq_fp32Round`.
- [BridgeFP32Total API](https://github.com/lean-dojo/TorchLean/blob/main/NN/Floats/IEEEExec/BridgeFP32Total.lean) packages total statements
  through `toReal?`, with theorem names such as `toReal?_add_eq_ite`,
  `toReal?_mul_eq_ite`, `toReal?_div_eq_ite`, and `toReal?_sqrt_eq_ite`.
- [SpecialRules API](https://github.com/lean-dojo/TorchLean/blob/main/NN/Floats/IEEEExec/SpecialRules.lean) records NaN, infinity,
  signed-zero, and special-case behavior for the executable kernel itself.

The informal theorem shape is:

- execute the operation in `IEEE32Exec`;
- decode the result with `toReal`;
- obtain the same real value as applying the corresponding real operation to the decoded inputs and
  rounding that result with `FP32.fp32Round`.

This reading is valid under the stated finite and operation specific hypotheses.

# Transcendentals And Library Boundaries

The proof-side `FP32` model defines transcendentals by applying the corresponding real function and
then rounding. Theorems such as `exp_abs_error`, `tanh_abs_error`, and interval membership lemmas
are therefore statements relative to Lean's real functions.

The executable and native stacks are different. A CPU `libm`, CUDA `libdevice`, or vendor library
may provide an approximation with documented accuracy. TorchLean therefore keeps transcendental
claims explicit:

- use `FP32` / `NF` for proof-side rounded-real statements;
- use `IEEE32Exec` for deterministic executable behavior where the operation is defined in Lean;
- use a runtime or CUDA contract when an external library computes the value.

This is the same lesson emphasized by the floating-point verification literature: compiler,
library, and hardware choices are part of the semantics unless they are isolated behind a proof or
contract.

# Runtime And CUDA Boundaries

Lean's runtime `Float32` operations are external runtime calls. Lean documents `Float` operations as
IEEE-style opaque operations that do not reduce in the kernel and compile to C operators; the same
general concern applies here. Runtime operations can be used, tested, and connected to Lean-side
semantics through explicit agreement assumptions.

TorchLean names that assumption surface:

```
import NN.Floats.IEEEExec.BridgeInitFloat32

open TorchLean.Floats.IEEE754
open TorchLean.Floats.IEEE754.Float32Bridge

#check RuntimeFloat32MatchesIEEE32Exec
#check RuntimeFloat32MatchesIEEE32Exec.toIEEE32Exec_add
#check RuntimeFloat32MatchesIEEE32Exec.toIEEE32Exec_mul
#check RuntimeFloat32MatchesIEEE32Exec.toIEEE32Exec_sqrt
```

The class says, operation by operation, that runtime float32 bits match the `IEEE32Exec` reference
bits. Once that bridge is supplied, downstream proofs can reuse the `IEEE32Exec` and `FP32`
theorems. Without it, a runtime result remains implementation evidence rather than a Lean-side
scalar statement.

CUDA uses the same idea at the native boundary:

```
import NN.Runtime.Autograd.Engine.Cuda.Float32Contract
import NN.Runtime.Autograd.Engine.Cuda.KernelSpec

open Runtime.Autograd.Cuda.Float32Contract

#check NativePrimitiveAgreement
#check native_add_abs_error_of_isFinite
#check native_mul_abs_error_of_isFinite
#check native_div_abs_error_of_isFinite
#check native_fma_abs_error_of_isFinite
#check native_sqrt_abs_error_of_isFinite
```

For the engineering details, read *GPU and CUDA Boundaries*. This page only fixes the scalar
meaning: native arithmetic is connected to proofs through explicit bit-agreement contracts, fixed
reduction specifications, and finite-path hypotheses.

# Concrete Bit Patterns

`IEEE32Exec` is useful because special values are concrete:

```
import NN.Floats.IEEEExec.Exec32

open TorchLean.Floats.IEEE754

def plusZero : IEEE32Exec := IEEE32Exec.ofBits (0x00000000 : UInt32)
def minusZero : IEEE32Exec := IEEE32Exec.ofBits (0x80000000 : UInt32)
def plusInf : IEEE32Exec := IEEE32Exec.ofBits (0x7f800000 : UInt32)
def qNaN : IEEE32Exec := IEEE32Exec.ofBits (0x7fc00000 : UInt32)

-- Round a Float64 literal to binary32 deterministically in Lean.
def third32 : IEEE32Exec := IEEE32Exec.ofFloat (Float.ofBits 0x3fd5555555555555)
```

For a visual inspection path, open the widget examples:
[Widgets API](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Advanced/Widgets.lean), especially `#float32_view` and
`#float32_round_view`.

# What The Claims Mean

Checked by Lean:

- `FP32` rounded-real definitions and local error lemmas;
- `IEEE32Exec` bit-level definitions for the supported core operations;
- finite bridge theorems from `IEEE32Exec` to `FP32`;
- total `toReal?` theorems that expose NaN/Inf paths;
- interval-facing and runtime-approximation lemmas that cite these scalar models.

Runtime agreement paths:

- Lean runtime `Float32` agreement with `IEEE32Exec`;
- CUDA primitive and kernel agreement with the Lean contracts;
- cuBLAS, cuFFT, libdevice, compiler, driver, and GPU behavior;
- correctly-rounded claims for external transcendental libraries;
- alternative rounding modes and exception flags not modeled by the current theorem path.

That split is the claim. Native execution is part of the workflow, and TorchLean makes the agreement
with Lean-side semantics small, named, and testable.

# How To Choose The Right Layer

Use this rule:

- Use `ℝ` for ideal networks, losses, and verifier statements.
- Use `FP32` / `NF` for compositional finite-precision error bounds.
- Use `IEEE32Exec` for bit inspection, NaN/Inf behavior, signed zeros, and executable binary32
  examples.
- Use `RuntimeFloat32MatchesIEEE32Exec` when runtime float results must be transported into the
  executable reference semantics.
- Use `Float32Contract` / `KernelSpec` when native CUDA kernels must be connected to the proof
  layer.

The rule is simple: choose the numerical object for the claim, then cite the bridge theorem,
checker, or runtime agreement that connects it to the layer below.

# References

- IEEE Std 754-2019, *Standard for Floating-Point Arithmetic*:
  [IEEE 754-2019](https://standards.ieee.org/standard/754-2019/)
- ISO/IEC/IEEE 60559:2020, the international floating-point arithmetic standard:
  [ISO/IEC/IEEE 60559:2020](https://standards.ieee.org/standard/60559-2020.html)
- David Goldberg, *What Every Computer Scientist Should Know About Floating-Point Arithmetic*:
  [Oracle-hosted reprint](https://docs.oracle.com/cd/E19957-01/806-3568/ncg_goldberg.html)
- Nicholas Higham, *Accuracy and Stability of Numerical Algorithms*:
  [SIAM book page](https://epubs.siam.org/doi/book/10.1137/1.9780898718027)
- Flocq, the Coq floating-point formalization:
  [Flocq documentation](https://flocq.gitlabpages.inria.fr/)
- FloatSpec, a Lean 4 floating-point formalization inspired by Flocq:
  [FloatSpec package](https://reservoir.lean-lang.org/%40Beneficial-AI-Foundation/FloatSpec)
- Gappa, for certified floating-point bounds and proof generation:
  [Gappa paper](https://arxiv.org/abs/0801.0523)
- CompCert, for verified compilation including machine floating-point models:
  [CompCert commented development](https://compcert.org/doc/)
- NVIDIA, *Floating Point and IEEE 754*:
  [NVIDIA floating-point guide](https://docs.nvidia.com/cuda/pdf/Floating_Point_on_NVIDIA_GPU.pdf)
