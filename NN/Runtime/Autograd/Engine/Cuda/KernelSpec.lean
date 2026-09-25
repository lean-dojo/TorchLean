/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Engine.Cuda.Float32Contract

/-!
# Pure specifications for CUDA float32 kernels

This file is the proof layer companion to `NN.Runtime.Autograd.Engine.Cuda.*`.

The LibTorch CUDA backend is an FFI boundary. These theorems describe explicit Lean functions;
applying them to a compiled native operation requires a refinement contract. The interface has
three layers:

1. **Pure Lean kernel specs** in this file: row-major indexing, elementwise maps, fixed-order
   reductions, gather/scatter, and batched matmul are ordinary Lean functions over finite indices.
2. **Scalar float32 facts** from `Float32Contract`: if native result bits match `ExecFloat.Binary 8
23`, then
   the existing `configured binary32 → FP32-on-ℝ` theorems apply.
3. **Native validation / trust boundary**: LibTorch, its CUDA dependencies, compiler flags,
   GPU hardware,
   and driver behavior are validated by tests and documented assumptions, not proved by Lean.

This split is deliberate. It lets us prove the algorithm/indexing contracts that TorchLean owns,
without claiming that the Lean kernel can inspect NVIDIA's compiler, runtime, or device ISA.

External references for the assumptions named here:

- IEEE 754-2019 defines binary32 arithmetic and special values:
  https://standards.ieee.org/ieee/754/6210/
- NVIDIA CUDA C Programming Guide documents the CUDA execution/memory model:
  https://docs.nvidia.com/cuda/cuda-c-programming-guide/
- ATen supplies batched matrix multiplication and selects its native implementation:
  https://docs.pytorch.org/docs/stable/generated/torch.bmm.html
- PyTorch's tensor docs are a useful user-facing analogue for row-major/strided tensor operations:
  https://pytorch.org/docs/stable/tensors.html
-/

@[expose] public section

open FloatLib.Floats (ExecFloat)
open FloatLib.Floats.ExecFloat (Binary)
open FloatLib.Floats.ExecFloat.Binary (toModel)
open FloatLib.Floats.Formats.BinaryInterchange (Model FloatFormat)

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

/-!
`ref_ext` is the `RefScalar` extensionality lemma from `Float32Contract`, which point 2 of the
module docstring above already names as this file's source of scalar float32 facts. It had a
private copy here with the same statement and the same proof, so the copy is gone and the lemma
is opened instead.
-/
open Float32Contract (ref_ext)
/-- Reinterpret a native bit buffer as reference `ExecFloat.Binary 8 23` values. -/
def fromNativeBitsBuffer {n : Nat} (xs : NativeBitsBuffer n) : FlatBuffer n :=
  fun i => fromNativeBits (xs i)

/--
Total lookup for flat-buffer specs that decode indices from arithmetic.

Well-formed kernel preconditions should make the in-bounds branch fire. The fallback keeps the spec
total in Lean, and later layout proofs can discharge the bounds separately.
-/
def getD {n : Nat} (x : FlatBuffer n) (i : Nat) : RefScalar :=
  if h : i < n then x ⟨i, h⟩ else (Binary.zero false : Binary 8 23)

/-- Extract reference bits pointwise. -/
def toNativeBitsBuffer {n : Nat} (xs : FlatBuffer n) : NativeBitsBuffer n :=
  fun i => toNativeBits (xs i)

/-- A buffer of reference scalars survives a round trip through its bits. -/
@[simp] theorem fromNativeBitsBuffer_toNativeBitsBuffer {n : Nat} (xs : FlatBuffer n) :
    fromNativeBitsBuffer (toNativeBitsBuffer xs) = xs := by
  funext i
  simp [fromNativeBitsBuffer, toNativeBitsBuffer]

/-- And a buffer of bits survives a round trip through reference scalars, so kernel specs may be
stated on whichever side reads better and transported to the other. -/
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
  map2Spec ExecFloat.add

/-- Elementwise multiplication reference spec. -/
def mulSpec {n : Nat} : FlatBuffer n → FlatBuffer n → FlatBuffer n :=
  map2Spec ExecFloat.mul

/-- Elementwise division reference spec. -/
def divSpec {n : Nat} : FlatBuffer n → FlatBuffer n → FlatBuffer n :=
  map2Spec ExecFloat.div

/-- Elementwise square-root reference spec. -/
def sqrtSpec {n : Nat} : FlatBuffer n → FlatBuffer n :=
  mapSpec (Binary.sqrt (rounding := .nearestEven))

/--
If every native result bit agrees with reference addition, the whole native elementwise-add buffer
agrees extensionally with `addSpec`.
-/
theorem fromNativeBitsBuffer_eq_addSpec_of_bits
    {n : Nat} {bits : NativeBitsBuffer n} {x y : FlatBuffer n}
    (hbits : ∀ i, bits i = toNativeBits (ExecFloat.add (x i) (y i))) :
    fromNativeBitsBuffer bits = addSpec x y := by
  funext i
  apply ref_ext
  simp [fromNativeBitsBuffer, addSpec, map2Spec, hbits i]

/--
Elementwise multiplication version of `fromNativeBitsBuffer_eq_addSpec_of_bits`.
-/
theorem fromNativeBitsBuffer_eq_mulSpec_of_bits
    {n : Nat} {bits : NativeBitsBuffer n} {x y : FlatBuffer n}
    (hbits : ∀ i, bits i = toNativeBits (ExecFloat.mul (x i) (y i))) :
    fromNativeBitsBuffer bits = mulSpec x y := by
  funext i
  apply ref_ext
  simp [fromNativeBitsBuffer, mulSpec, map2Spec, hbits i]

/--
Elementwise division version of `fromNativeBitsBuffer_eq_addSpec_of_bits`.
-/
theorem fromNativeBitsBuffer_eq_divSpec_of_bits
    {n : Nat} {bits : NativeBitsBuffer n} {x y : FlatBuffer n}
    (hbits : ∀ i, bits i = toNativeBits (ExecFloat.div (x i) (y i))) :
    fromNativeBitsBuffer bits = divSpec x y := by
  funext i
  apply ref_ext
  simp [fromNativeBitsBuffer, divSpec, map2Spec, hbits i]

/--
Elementwise square-root version of `fromNativeBitsBuffer_eq_addSpec_of_bits`.
-/
theorem fromNativeBitsBuffer_eq_sqrtSpec_of_bits
    {n : Nat} {bits : NativeBitsBuffer n} {x : FlatBuffer n}
    (hbits : ∀ i, bits i = toNativeBits ((Binary.sqrt (rounding := .nearestEven)) (x i))) :
    fromNativeBitsBuffer bits = sqrtSpec x := by
  funext i
  apply ref_ext
  simp [fromNativeBitsBuffer, sqrtSpec, mapSpec, hbits i]

/-- Reference operations as native bits, for reuse of the scalar agreement contract. -/
private def referenceBits : NativePrimitiveBits where
  addBits x y := toNativeBits (ExecFloat.add x y)
  mulBits x y := toNativeBits (ExecFloat.mul x y)
  divBits x y := toNativeBits (ExecFloat.div x y)
  fmaBits x y z := toNativeBits ((Binary.fma (rounding := .nearestEven)) x y z)
  sqrtBits x := toNativeBits ((Binary.sqrt (rounding := .nearestEven)) x)

private theorem referenceAgreement : NativePrimitiveAgreement referenceBits where
  add_bits _ _ := Or.inl rfl
  mul_bits _ _ := Or.inl rfl
  div_bits _ _ := Or.inl rfl
  fma_bits _ _ _ := Or.inl rfl
  sqrt_bits _ := Or.inl rfl

/-- Pointwise real-error bound inherited by an elementwise-add buffer after bit agreement. -/
theorem native_add_pointwise_abs_error_of_bits
    {n : Nat} {bits : NativeBitsBuffer n} {x y : FlatBuffer n}
    (hbits : ∀ i, bits i = toNativeBits (ExecFloat.add (x i) (y i)))
    (i : Fin n)
    (hfin : Binary.isFinite (fromNativeBits (bits i)) = true) :
    abs
        ((toModel (fromNativeBits (bits i))).toReal -
          ((toModel (x i)).toReal + (toModel (y i)).toReal)) ≤
      eps32 ((toModel (x i)).toReal + (toModel (y i)).toReal) := by
  have hx : fromNativeBits (bits i) = ExecFloat.add (x i) (y i) := by
    apply ref_ext
    simp [hbits i]
  rw [hx] at hfin ⊢
  simpa only [referenceBits, fromNativeBits_toNativeBits] using
    native_add_abs_error_of_isFinite referenceAgreement (x i) (y i)
      (by simpa only [referenceBits, fromNativeBits_toNativeBits] using hfin)

/--
Pointwise real-error bound inherited by a native elementwise-multiply buffer after bit agreement.
-/
theorem native_mul_pointwise_abs_error_of_bits
    {n : Nat} {bits : NativeBitsBuffer n} {x y : FlatBuffer n}
    (hbits : ∀ i, bits i = toNativeBits (ExecFloat.mul (x i) (y i)))
    (i : Fin n)
    (hfin : Binary.isFinite (fromNativeBits (bits i)) = true) :
    abs
        ((toModel (fromNativeBits (bits i))).toReal -
          ((toModel (x i)).toReal * (toModel (y i)).toReal)) ≤
      eps32 ((toModel (x i)).toReal * (toModel (y i)).toReal) := by
  have hx : fromNativeBits (bits i) = ExecFloat.mul (x i) (y i) := by
    apply ref_ext
    simp [hbits i]
  rw [hx] at hfin ⊢
  simpa only [referenceBits, fromNativeBits_toNativeBits] using
    native_mul_abs_error_of_isFinite referenceAgreement (x i) (y i)
      (by simpa only [referenceBits, fromNativeBits_toNativeBits] using hfin)

/--
Pointwise real-error bound inherited by a native elementwise-division buffer after bit agreement.
-/
theorem native_div_pointwise_abs_error_of_bits
    {n : Nat} {bits : NativeBitsBuffer n} {x y : FlatBuffer n}
    (hbits : ∀ i, bits i = toNativeBits (ExecFloat.div (x i) (y i)))
    (i : Fin n)
    (hfin : Binary.isFinite (fromNativeBits (bits i)) = true) :
    abs
        ((toModel (fromNativeBits (bits i))).toReal -
          ((toModel (x i)).toReal / (toModel (y i)).toReal)) ≤
      eps32 ((toModel (x i)).toReal / (toModel (y i)).toReal) := by
  have hx : fromNativeBits (bits i) = ExecFloat.div (x i) (y i) := by
    apply ref_ext
    simp [hbits i]
  rw [hx] at hfin ⊢
  simpa only [referenceBits, fromNativeBits_toNativeBits] using
    native_div_abs_error_of_isFinite referenceAgreement (x i) (y i)
      (by simpa only [referenceBits, fromNativeBits_toNativeBits] using hfin)

/--
Pointwise real-error bound inherited by a native elementwise-square-root buffer after bit agreement.
-/
theorem native_sqrt_pointwise_abs_error_of_bits
    {n : Nat} {bits : NativeBitsBuffer n} {x : FlatBuffer n}
    (hbits : ∀ i, bits i = toNativeBits ((Binary.sqrt (rounding := .nearestEven)) (x i)))
    (i : Fin n)
    (hfin : Binary.isFinite (fromNativeBits (bits i)) = true) :
    abs
        ((toModel (fromNativeBits (bits i))).toReal -
          Real.sqrt ((toModel (x i)).toReal)) ≤
      eps32 (Real.sqrt ((toModel (x i)).toReal)) := by
  have hx : fromNativeBits (bits i) = (Binary.sqrt (rounding := .nearestEven)) (x i) := by
    apply ref_ext
    simp [hbits i]
  rw [hx] at hfin ⊢
  simpa only [referenceBits, fromNativeBits_toNativeBits] using
    native_sqrt_abs_error_of_isFinite referenceAgreement (x i)
      (by simpa only [referenceBits, fromNativeBits_toNativeBits] using hfin)

/-! ## Fixed-order reductions -/

/--
Sequential left-fold reduction over a flat buffer.

This is a *deterministic algorithmic spec*, not a claim about CUDA atomics. Native atomic reductions
only refine this spec under an additional ordering/agreement assumption. TorchLean's deterministic
reduction mode is intended to make that assumption true for tested reduction paths.
-/
def reduceSumLeftSpec {n : Nat} (x : FlatBuffer n) : RefScalar :=
  (List.finRange n).foldl (fun acc i => ExecFloat.add acc (x i)) (Binary.zero false : Binary 8 23)

/--
Explicit assumption package for a native reduction implementation.

Use this when a native CUDA reduction has been configured or validated to use the same fixed order
as `reduceSumLeftSpec`. Non-deterministic `atomicAdd` reductions should not claim this contract
unless the runtime mode or kernel implementation fixes the accumulation order.
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
      (fun acc j => if idx j = i then ExecFloat.add acc (values j) else acc)
      (x i)

/-! ## Batched row-major matrix multiplication -/

/-- Linear row-major index for `A[b, i, k]` with shape `(batch, m, n)`. -/
def bmmAIndex (m n : Nat) (b i k : Nat) : Nat :=
  (b * m + i) * n + k

/-- Linear row-major index for `B[b, k, j]` with shape `(batch, n, p)`. -/
def bmmBIndex (n p : Nat) (b k j : Nat) : Nat :=
  (b * n + k) * p + j

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

For each output element `C[b,i,j]`, this folds over `k = 0..n-1` using `ExecFloat.mul` followed by
`ExecFloat.add`. This fixes a *specific* accumulation order. LibTorch may use a different
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
        ExecFloat.add acc (ExecFloat.mul a bVal))
      (Binary.zero false : Binary 8 23)

/--
Agreement assumption for a native BMM implementation.

The scalar result bits must match `bmmSpec` at every output element. This is stronger
than "numerically close": it is the bitwise contract needed to reuse exact `ExecFloat.Binary 8 23`
proofs.

For a LibTorch implementation this assumption includes:
- row-major TorchLean buffers are interpreted with the intended tensor dimensions;
- the accumulation tree/FMA behavior agrees with the selected reference spec;
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
