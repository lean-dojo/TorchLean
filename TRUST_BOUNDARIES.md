# TorchLean Trust Boundaries

TorchLean uses Lean to state and check mathematical claims about neural-network artifacts. Some
parts of the system are inside Lean's proof kernel; others are executable tools, native runtimes, or
external producers whose outputs may be checked by Lean.

This file is the project's trust inventory. It records the assumptions that matter for correctness
claims: Lean axioms, Prop-valued contracts, CUDA and FFI code, external numeric oracles, PyTorch
import/export scripts, Julia/Python producers, and artifact-checking conventions.

## Reading This File

| Layer | Example | How to read it |
| --- | --- | --- |
| Lean theorem | graph semantics, autograd correctness | checked by Lean |
| Executable checker | certificate parser, shape checker | checked by code/tests |
| Prop-valued contract | runtime Float32 agreement | assumption supplied by caller |
| FFI/native runtime | CUDA kernels, cuBLAS, cuFFT | external implementation path |
| External producer | Python, Julia, Arb, alpha-beta-CROWN | produces artifacts Lean may check |

## Lean Axioms

- `NN/MLTheory/CROWN/Lyapunov/Oracle.lean`: `crown_oracle` assumes an external CROWN
  checker has produced a `CrownOracleWitness lyap cert`; given that witness, the certificate
  soundly bounds `V` and `Vdot` over the stated region.
- `NN/Runtime/Autograd/Engine/Cuda/Trusted.lean`: `instNonemptyBuffer` assumes the external
  CUDA/stub runtime can construct values of the opaque `Cuda.Buffer` type.
- `scripts/checks/repo_lint.py` allowlists these exact axiom names. New axioms must be added here
  deliberately and documented in this file.

You can inspect theorem dependencies inside Lean with:

```lean
#print axioms Runtime.Autograd.Cuda.instNonemptyBuffer
#print axioms NN.MLTheory.CROWN.Lyapunov.crown_oracle
```

For an audit from the shell:

```bash
rg -n "^(noncomputable\\s+)?opaque |^axiom " NN -g'*.lean'
```

## Prop-Valued Contracts

Some declarations are `class ... : Prop` or `structure ... : Prop` rather than axioms. These are
not kernel assumptions by themselves: a theorem using one is conditional on the caller supplying the
fields. We still treat them as part of the trust model, because public theorem names and docs should
make those assumptions visible.

Important examples include:

- `TorchLean.Floats.IEEE754.Float32Bridge.RuntimeFloat32MatchesIEEE32Exec`, the runtime contract
  that Lean `Float32` primitives match the executable IEEE32 model at the bit level.
- `NN.MLTheory.CROWN.CrownTransferSound`, the transfer-rule soundness assumption used by
  graph-CROWN certificate theorems for backend/oracle-dependent relaxations.
- `NN.MLTheory.Proofs.Approximation.FloatInterval.OpsExact.Sound`, the local operation level exact
  interval soundness contract for finite IEEE32 interval arithmetic.

## Opaque Non-FFI Declarations

- `NN.MLTheory.CROWN.Lyapunov.CrownOracleWitness` is an abstract witness type for the external
  Lyapunov oracle.
- `NN.MLTheory.CROWN.betaAt` is an executable wrapper around a length-checked beta-phase array
  lookup. It keeps the checker executable without exposing brittle `Array.get!` internals to every
  proof.

## CUDA Runtime

- Files under `csrc/cuda/` are trusted FFI code. Lean checks shape metadata around calls, but kernel
  memory safety, launch behavior, and numerical behavior are outside Lean's proof kernel.
- `csrc/cuda/tensor/torchlean_cuda_tensor.cu` stores CUDA buffers as float32 and converts Lean `Float`
  values to/from float32 at the buffer boundary.
- CUDA buffer finalizers free device memory through `cudaFree`. This is safe for TorchLean's current
  default-stream runtime, where launches and host copies are ordered through the default stream. If
  future backends introduce user streams or asynchronous graph replay, finalizer/free ordering must
  be revisited explicitly.
- GPU matmul supports two explicit precision paths:
  - FP32: `NN/Runtime/Autograd/Engine/Cuda/Kernels.lean` uses `Cuda.Buffer.bmm`, backed by
    `cublasSgemmStridedBatched` in `csrc/cuda/kernels/torchlean_cuda_kernels.cu`.
  - FP64: `NN/Runtime/Autograd/Engine/Cuda/DGemm.lean` uses `torchleanDgemmCuda`, backed by
    `cublasDgemm` in `csrc/cuda/blas/torchlean_dgemm_cuda.cu`.
- The fast-kernel Float dispatcher makes this choice explicit via `GpuMatmulPrecision`.
- Several CUDA backward/reduction paths use `atomicAdd`. These are mathematically standard for
  accumulation but are not bit-deterministic across schedules because float32 addition is not
  associative.
- TorchLean provides an opt-in deterministic reductions mode that replaces the `atomicAdd`-based
  accumulation paths with fixed-order algorithms (slower, but bit-stable across runs on the same
  GPU). You can enable it either:
  - from Lean (recommended): `let _ := Runtime.Autograd.Cuda.Buffer.setDeterministicReductionsChecked true`
  - via env var: `TORCHLEAN_CUDA_DETERMINISTIC_REDUCTIONS=1` (or compatibility alias `TORCHLEAN_DETERMINISTIC_REDUCTIONS=1`)
  Coverage includes:
  - reductions: `Buffer.reduceSum`, `Buffer.reduceMean`, `reduceFromBroadcastTo`, `reduceSumAxis`
  - gather/scatter backprop: `scatterAdd`, `scatterAddRows`
  - pooling backward: `max_pool*`, `avg_pool*`, `smooth_max_pool*` (2D and N-D entrypoints)
  Does not cover:
  - nondeterminism from RNG (use seeded RNG ops, or manage seeds/counters explicitly)
  - numerically different results across GPU architectures, CUDA toolkit versions, or driver versions
  - kernels that are not on the deterministic-reductions allowlist (only the atomic-accumulation paths above)
- CUDA max-pooling follows the TorchLean spec, which models PyTorch-style negative-infinity padding
  by ignoring padded cells outside the domain when selecting the max. Backward tie-breaking is
  TorchLean-spec row-major deterministic when deterministic reductions are enabled, while external
  runtimes may choose different tie-breaking policies.
- FlashAttention has a fused-operator denotation for proofs in
  `NN/Spec/Layers/FlashAttention.lean`: over the spec semantics it denotes the same masked scaled
  dot-product attention as the standard `QKᵀ -> mask -> softmax -> PV` graph. The CUDA eager
  multi-head attention path can use native fused runtime kernels exposed through
  `NN/Runtime/Autograd/Engine/Cuda/Kernels.lean` and implemented in
  `csrc/cuda/kernels/torchlean_cuda_kernels.cu`. Those kernels favor clarity and correctness: fused
  forward/VJP kernels over already-split heads, not a production clone of Dao-AILab's tiled
  implementation. The Lean equalities are definitional denotation checks, not proofs of a tiled
  online-softmax recurrence. CUDA memory behavior and float32 arithmetic remain an FFI trust
  boundary; TorchLean regression-tests them against the composed attention path, but does not claim
  to verify CUDA machine code. References: FlashAttention (arXiv:2205.14135), FlashAttention-2
  (arXiv:2307.08691), FlashAttention-3 (arXiv:2407.08608), and the Dao-AILab `flash-attention`
  implementation.
- Attention masks use TorchLean's finite big-negative mask-fill convention (`-1000`) in the spec and
  CUDA fused kernels, rather than PyTorch SDPA's conceptual `-inf` masking. In float32 this normally
  underflows masked probabilities to zero after max-subtraction, but it is a documented semantic
  convention rather than a theorem of bit-identical PyTorch masking.
- Kernel launch synchronization is an implementation detail of the native runtime. Tensor/view
  kernels usually rely on default-stream ordering and later host copies to synchronize; conv/pool
  kernels explicitly synchronize after exported operations for clearer error attribution around
  heavier kernels. Both policies are outside Lean's kernel and should not be used as proof evidence.

## Executable Floating Point

- `NN/Floats/IEEEExec/` proves and implements a deterministic IEEE-style executable model for many
  core operations.
- Transcendental functions such as `exp`, `log`, and `tanh` are deterministic approximations unless
  a file states a stronger theorem for a specific operation.

## External Numeric Oracles

- CROWN/Lyapunov certificate generation is an external evidence producer when used through the
  oracle-backed workflow. Lean isolates that assumption behind `crown_oracle`; it does not prove that
  the external CROWN run was complete or correctly implemented.
- The Arb / `python-flint` integration under `NN/Floats/Arb/` is an external subprocess backend. It
  is useful for producing high-quality interval evidence, but an Arb response is still an oracle
  result unless the relevant certificate is independently checked in Lean.
- PyTorch import/export scripts and training helpers are external producers of weights, examples,
  or JSON artifacts. TorchLean can parse and replay those artifacts, but PyTorch training itself is
  not part of Lean's trusted kernel.
- The optional Julia wrapper `NN/Runtime/External/Julia.lean` follows the same pattern. It resolves
  `TORCHLEAN_JULIA` when set, otherwise falls back to `julia` on `PATH`, and compiles even when Julia
  is not installed. It is intended for “untrusted producer, Lean checker” workflows such as the
  piecewise-polynomial spline certificate workflow (producer scripts under `scripts/verification/splines/`,
  bundled fixtures under `NN/Examples/Verification/Splines/`).
- A Julia-produced spline or PINN artifact is trusted only after a Lean checker validates the small
  certificate data it needs: for example cell domains, polynomial coefficients, interval bounds, and
  claimed residual inequalities. Lean does not trust Julia's fitting process, optimizer, GPU use, or
  floating-point arithmetic merely because the subprocess returned successfully.
