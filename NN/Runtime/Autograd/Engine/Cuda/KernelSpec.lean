/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Engine.Cuda.Float32Contract

/-!
# Pure specifications for CUDA float32 kernels

This file is the proof-facing companion to `NN.Runtime.Autograd.Engine.Cuda.*`.

The native CUDA backend is an FFI boundary, so Lean cannot prove facts about the compiled `.cu`
binary directly. What we can do well is factor the interface into three layers:

1. **Pure Lean kernel specs** in this file: row-major indexing, elementwise maps, fixed-order
   reductions, gather/scatter, and batched matmul are ordinary Lean functions over finite indices.
2. **Scalar float32 facts** from `Float32Contract`: if native result bits match `IEEE32Exec`, then
   the existing `IEEE32Exec → FP32-on-ℝ` theorems apply.
3. **Native validation / trust boundary**: CUDA C, libdevice, cuBLAS, compiler flags, GPU hardware,
   and driver behavior are validated by tests and documented assumptions, not proved by Lean.

This split is deliberate. It lets us prove the algorithm/indexing contracts that TorchLean owns,
without claiming that the Lean kernel can inspect NVIDIA's compiler, runtime, or device ISA.

External references for the assumptions named here:

- IEEE 754-2019 defines binary32 arithmetic and special values:
  https://standards.ieee.org/ieee/754/6210/
- NVIDIA CUDA C Programming Guide documents the CUDA execution/memory model:
  https://docs.nvidia.com/cuda/cuda-c-programming-guide/
- cuBLAS documents GEMM's column-major API contract; TorchLean's CUDA BMM uses a row-major
  interpretation around that API:
  https://docs.nvidia.com/cuda/cublas/
- PyTorch's tensor docs are a useful user-facing analogue for row-major/strided tensor operations:
  https://pytorch.org/docs/stable/tensors.html
-/

@[expose] public section

namespace Runtime
namespace Autograd
namespace Cuda
namespace KernelSpec

open TorchLean.Floats
open TorchLean.Floats.IEEE754
open Float32Contract

noncomputable section

/-! ## Flat row-major buffers -/

/--
A pure Lean view of a contiguous float32 buffer of length `n`.

Native `Cuda.Buffer` is opaque and mutable outside Lean. `FlatBuffer n` is the spec counterpart:
every valid linear index has a reference scalar value.
-/
abbrev FlatBuffer (n : Nat) := Fin n → RefScalar

/-- A native-result buffer represented only by raw binary32 bits. -/
abbrev NativeBitsBuffer (n : Nat) := Fin n → UInt32

/-- Extensionality for the thin `IEEE32Exec` wrapper, phrased through native bits. -/
private theorem ref_ext {x y : RefScalar} (h : toNativeBits x = toNativeBits y) : x = y := by
  cases x
  cases y
  cases h
  rfl

/-- Reinterpret a native bit buffer as reference `IEEE32Exec` values. -/
def fromNativeBitsBuffer {n : Nat} (xs : NativeBitsBuffer n) : FlatBuffer n :=
  fun i => fromNativeBits (xs i)

/--
Total lookup for flat-buffer specs that decode indices from arithmetic.

Well-formed kernel preconditions should make the in-bounds branch fire. The fallback keeps the spec
total in Lean, and later layout proofs can discharge the bounds separately.
-/
def getD {n : Nat} (x : FlatBuffer n) (i : Nat) : RefScalar :=
  if h : i < n then x ⟨i, h⟩ else IEEE32Exec.posZero

/-- Extract reference bits pointwise. -/
def toNativeBitsBuffer {n : Nat} (xs : FlatBuffer n) : NativeBitsBuffer n :=
  fun i => toNativeBits (xs i)

@[simp] theorem fromNativeBitsBuffer_toNativeBitsBuffer {n : Nat} (xs : FlatBuffer n) :
    fromNativeBitsBuffer (toNativeBitsBuffer xs) = xs := by
  funext i
  simp [fromNativeBitsBuffer, toNativeBitsBuffer]

@[simp] theorem toNativeBitsBuffer_fromNativeBitsBuffer {n : Nat} (xs : NativeBitsBuffer n) :
    toNativeBitsBuffer (fromNativeBitsBuffer xs) = xs := by
  funext i
  simp [fromNativeBitsBuffer, toNativeBitsBuffer]

/-! ## Elementwise kernels -/

/-- Pure spec for a unary elementwise CUDA kernel. -/
def mapSpec {n : Nat} (f : RefScalar → RefScalar) (x : FlatBuffer n) : FlatBuffer n :=
  fun i => f (x i)

/-- Pure spec for a binary elementwise CUDA kernel. -/
def map2Spec {n : Nat} (f : RefScalar → RefScalar → RefScalar)
    (x y : FlatBuffer n) : FlatBuffer n :=
  fun i => f (x i) (y i)

/-- Elementwise addition reference spec. -/
def addSpec {n : Nat} : FlatBuffer n → FlatBuffer n → FlatBuffer n :=
  map2Spec IEEE32Exec.add

/-- Elementwise multiplication reference spec. -/
def mulSpec {n : Nat} : FlatBuffer n → FlatBuffer n → FlatBuffer n :=
  map2Spec IEEE32Exec.mul

/-- Elementwise division reference spec. -/
def divSpec {n : Nat} : FlatBuffer n → FlatBuffer n → FlatBuffer n :=
  map2Spec IEEE32Exec.div

/-- Elementwise square-root reference spec. -/
def sqrtSpec {n : Nat} : FlatBuffer n → FlatBuffer n :=
  mapSpec IEEE32Exec.sqrt

/--
If every native result bit agrees with reference addition, the whole native elementwise-add buffer
agrees extensionally with `addSpec`.
-/
theorem fromNativeBitsBuffer_eq_addSpec_of_bits
    {n : Nat} {bits : NativeBitsBuffer n} {x y : FlatBuffer n}
    (hbits : ∀ i, bits i = toNativeBits (IEEE32Exec.add (x i) (y i))) :
    fromNativeBitsBuffer bits = addSpec x y := by
  funext i
  apply ref_ext
  simp [fromNativeBitsBuffer, addSpec, map2Spec, hbits i]

/--
Elementwise multiplication version of `fromNativeBitsBuffer_eq_addSpec_of_bits`.
-/
theorem fromNativeBitsBuffer_eq_mulSpec_of_bits
    {n : Nat} {bits : NativeBitsBuffer n} {x y : FlatBuffer n}
    (hbits : ∀ i, bits i = toNativeBits (IEEE32Exec.mul (x i) (y i))) :
    fromNativeBitsBuffer bits = mulSpec x y := by
  funext i
  apply ref_ext
  simp [fromNativeBitsBuffer, mulSpec, map2Spec, hbits i]

/--
Elementwise division version of `fromNativeBitsBuffer_eq_addSpec_of_bits`.
-/
theorem fromNativeBitsBuffer_eq_divSpec_of_bits
    {n : Nat} {bits : NativeBitsBuffer n} {x y : FlatBuffer n}
    (hbits : ∀ i, bits i = toNativeBits (IEEE32Exec.div (x i) (y i))) :
    fromNativeBitsBuffer bits = divSpec x y := by
  funext i
  apply ref_ext
  simp [fromNativeBitsBuffer, divSpec, map2Spec, hbits i]

/--
Elementwise square-root version of `fromNativeBitsBuffer_eq_addSpec_of_bits`.
-/
theorem fromNativeBitsBuffer_eq_sqrtSpec_of_bits
    {n : Nat} {bits : NativeBitsBuffer n} {x : FlatBuffer n}
    (hbits : ∀ i, bits i = toNativeBits (IEEE32Exec.sqrt (x i))) :
    fromNativeBitsBuffer bits = sqrtSpec x := by
  funext i
  apply ref_ext
  simp [fromNativeBitsBuffer, sqrtSpec, mapSpec, hbits i]

/--
Pointwise real-error bound inherited by a native elementwise-add buffer after bit agreement.
-/
theorem native_add_pointwise_abs_error_of_bits
    {n : Nat} {bits : NativeBitsBuffer n} {x y : FlatBuffer n}
    (hbits : ∀ i, bits i = toNativeBits (IEEE32Exec.add (x i) (y i)))
    (i : Fin n)
    (hfin : IEEE32Exec.isFinite (fromNativeBits (bits i)) = true) :
    _root_.abs
        (IEEE32Exec.toReal (fromNativeBits (bits i)) -
          (IEEE32Exec.toReal (x i) + IEEE32Exec.toReal (y i))) ≤
      eps₃₂ (IEEE32Exec.toReal (x i) + IEEE32Exec.toReal (y i)) := by
  have hx : fromNativeBits (bits i) = IEEE32Exec.add (x i) (y i) := by
    apply ref_ext
    simp [hbits i]
  rw [hx] at hfin ⊢
  exact IEEE32Exec.toReal_add_abs_error_of_isFinite (x i) (y i) hfin

/--
Pointwise real-error bound inherited by a native elementwise-multiply buffer after bit agreement.
-/
theorem native_mul_pointwise_abs_error_of_bits
    {n : Nat} {bits : NativeBitsBuffer n} {x y : FlatBuffer n}
    (hbits : ∀ i, bits i = toNativeBits (IEEE32Exec.mul (x i) (y i)))
    (i : Fin n)
    (hfin : IEEE32Exec.isFinite (fromNativeBits (bits i)) = true) :
    _root_.abs
        (IEEE32Exec.toReal (fromNativeBits (bits i)) -
          (IEEE32Exec.toReal (x i) * IEEE32Exec.toReal (y i))) ≤
      eps₃₂ (IEEE32Exec.toReal (x i) * IEEE32Exec.toReal (y i)) := by
  have hx : fromNativeBits (bits i) = IEEE32Exec.mul (x i) (y i) := by
    apply ref_ext
    simp [hbits i]
  rw [hx] at hfin ⊢
  exact IEEE32Exec.toReal_mul_abs_error_of_isFinite (x i) (y i) hfin

/--
Pointwise real-error bound inherited by a native elementwise-division buffer after bit agreement.
-/
theorem native_div_pointwise_abs_error_of_bits
    {n : Nat} {bits : NativeBitsBuffer n} {x y : FlatBuffer n}
    (hbits : ∀ i, bits i = toNativeBits (IEEE32Exec.div (x i) (y i)))
    (i : Fin n)
    (hfin : IEEE32Exec.isFinite (fromNativeBits (bits i)) = true) :
    _root_.abs
        (IEEE32Exec.toReal (fromNativeBits (bits i)) -
          (IEEE32Exec.toReal (x i) / IEEE32Exec.toReal (y i))) ≤
      eps₃₂ (IEEE32Exec.toReal (x i) / IEEE32Exec.toReal (y i)) := by
  have hx : fromNativeBits (bits i) = IEEE32Exec.div (x i) (y i) := by
    apply ref_ext
    simp [hbits i]
  rw [hx] at hfin ⊢
  exact IEEE32Exec.toReal_div_abs_error_of_isFinite (x i) (y i) hfin

/--
Pointwise real-error bound inherited by a native elementwise-square-root buffer after bit agreement.
-/
theorem native_sqrt_pointwise_abs_error_of_bits
    {n : Nat} {bits : NativeBitsBuffer n} {x : FlatBuffer n}
    (hbits : ∀ i, bits i = toNativeBits (IEEE32Exec.sqrt (x i)))
    (i : Fin n)
    (hfin : IEEE32Exec.isFinite (fromNativeBits (bits i)) = true) :
    _root_.abs
        (IEEE32Exec.toReal (fromNativeBits (bits i)) -
          _root_.Real.sqrt (IEEE32Exec.toReal (x i))) ≤
      eps₃₂ (_root_.Real.sqrt (IEEE32Exec.toReal (x i))) := by
  have hx : fromNativeBits (bits i) = IEEE32Exec.sqrt (x i) := by
    apply ref_ext
    simp [hbits i]
  rw [hx] at hfin ⊢
  exact IEEE32Exec.toReal_sqrt_abs_error_of_isFinite (x i) hfin

/-! ## Fixed-order reductions -/

/--
Sequential left-fold reduction over a flat buffer.

This is a *deterministic algorithmic spec*, not a claim about CUDA atomics. Native atomic reductions
only refine this spec under an additional ordering/agreement assumption. TorchLean's deterministic
reduction mode is intended to make that assumption true for tested reduction paths.
-/
def reduceSumLeftSpec {n : Nat} (x : FlatBuffer n) : RefScalar :=
  (List.finRange n).foldl (fun acc i => IEEE32Exec.add acc (x i)) IEEE32Exec.posZero

/--
Explicit assumption package for a native reduction implementation.

Use this when a native CUDA reduction has been configured or validated to use the same fixed order as
`reduceSumLeftSpec`. Non-deterministic `atomicAdd` reductions should not claim this contract unless
the runtime mode or kernel implementation fixes the accumulation order.
-/
structure NativeReduceAgreement {n : Nat} (nativeBits : UInt32) (x : FlatBuffer n) : Prop where
  bits_eq_left_fold : nativeBits = toNativeBits (reduceSumLeftSpec x)

/-- Native fixed-order reduction inherits the `reduceSumLeftSpec` reference value. -/
theorem native_reduce_eq_leftSpec
    {n : Nat} {nativeBits : UInt32} {x : FlatBuffer n}
    (h : NativeReduceAgreement nativeBits x) :
  fromNativeBits nativeBits = reduceSumLeftSpec x := by
  apply ref_ext
  simp [h.bits_eq_left_fold]

/-! ## Gather/scatter indexing -/

/-- Gather `k` elements from a length-`n` vector using proof-carrying indices. -/
def gatherVecSpec {n k : Nat} (x : FlatBuffer n) (idx : Fin k → Fin n) : FlatBuffer k :=
  fun j => x (idx j)

/--
Scatter-add a length-`k` value buffer into a length-`n` input buffer.

Repeated indices are accumulated in increasing source-index order. This mirrors the mathematical
contract of scatter-add; a native parallel implementation must separately justify or validate its
accumulation order when bitwise reproducibility matters.
-/
def scatterAddSpec {n k : Nat} (x : FlatBuffer n) (values : FlatBuffer k)
    (idx : Fin k → Fin n) : FlatBuffer n :=
  fun i =>
    (List.finRange k).foldl
      (fun acc j => if idx j = i then IEEE32Exec.add acc (values j) else acc)
      (x i)

/-- A gather followed by scatter-add to zeros accumulates each selected source position. -/
def gatherThenScatterToZeroSpec {n k : Nat} (x : FlatBuffer n) (idx : Fin k → Fin n) :
    FlatBuffer n :=
  scatterAddSpec (fun _ => IEEE32Exec.posZero) (gatherVecSpec x idx) idx

/-! ## Batched row-major matrix multiplication -/

/-- Linear row-major index for `A[b, i, k]` with shape `(batch, m, n)`. -/
def bmmAIndex (m n : Nat) (b i k : Nat) : Nat :=
  (b * m + i) * n + k

/-- Linear row-major index for `B[b, k, j]` with shape `(batch, n, p)`. -/
def bmmBIndex (n p : Nat) (b k j : Nat) : Nat :=
  (b * n + k) * p + j

/-- Linear row-major index for `C[b, i, j]` with shape `(batch, m, p)`. -/
def bmmCIndex (m p : Nat) (b i j : Nat) : Nat :=
  (b * m + i) * p + j

/-- Decode a flat row-major output index for shape `(batch, m, p)`. -/
def bmmDecodeC (m p : Nat) (q : Nat) : Nat × Nat × Nat :=
  let rowSize := m * p
  let b := if rowSize = 0 then 0 else q / rowSize
  let r := if rowSize = 0 then 0 else q % rowSize
  let i := if p = 0 then 0 else r / p
  let j := if p = 0 then 0 else r % p
  (b, i, j)

/--
Pure row-major batched matrix multiplication spec.

For each output element `C[b,i,j]`, this folds over `k = 0..n-1` using `IEEE32Exec.mul` followed by
`IEEE32Exec.add`. This is intentionally a *specific* accumulation order. cuBLAS may use a different
tree/FMA strategy, so bit-for-bit agreement with this spec is an explicit native contract, not a
free theorem.
-/
def bmmSpec (batch m n p : Nat)
    (A : FlatBuffer (batch * m * n)) (B : FlatBuffer (batch * n * p)) :
    FlatBuffer (batch * m * p) :=
  fun q =>
    let (b, i, j) := bmmDecodeC m p q.val
    (List.finRange n).foldl
      (fun acc k =>
        let a := getD A (bmmAIndex m n b i k.val)
        let bVal := getD B (bmmBIndex n p b k.val j)
        IEEE32Exec.add acc (IEEE32Exec.mul a bVal))
      IEEE32Exec.posZero

/--
Agreement assumption for a native BMM implementation.

The scalar result bits must match `bmmSpec` at every output element. This is intentionally stronger
than "numerically close": it is the bitwise contract needed to reuse exact `IEEE32Exec` proofs.

For cuBLAS-backed kernels this assumption includes:
- row-major TorchLean buffers are interpreted consistently around cuBLAS's column-major GEMM API;
- the accumulation tree/FMA behavior is compatible with the selected reference spec, or the spec is
  adjusted to the documented cuBLAS/toolchain behavior;
- input and output strides match `(batch,m,n)`, `(batch,n,p)`, and `(batch,m,p)` row-major layout.
-/
structure NativeBmmAgreement
    {batch m n p : Nat}
    (nativeBits : NativeBitsBuffer (batch * m * p))
    (A : FlatBuffer (batch * m * n)) (B : FlatBuffer (batch * n * p)) : Prop where
  bits_eq_bmmSpec : ∀ q, nativeBits q = toNativeBits (bmmSpec batch m n p A B q)

/-- Native BMM inherits the pure row-major `bmmSpec` when the bitwise agreement contract holds. -/
theorem native_bmm_eq_spec
    {batch m n p : Nat}
    {nativeBits : NativeBitsBuffer (batch * m * p)}
    {A : FlatBuffer (batch * m * n)} {B : FlatBuffer (batch * n * p)}
    (h : NativeBmmAgreement nativeBits A B) :
    fromNativeBitsBuffer nativeBits = bmmSpec batch m n p A B := by
  funext q
  apply ref_ext
  simp [fromNativeBitsBuffer, h.bits_eq_bmmSpec q]

end

end KernelSpec
end Cuda
end Autograd
end Runtime
