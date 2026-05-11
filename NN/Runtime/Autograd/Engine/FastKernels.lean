/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Engine.Core
public import NN.Runtime.Autograd.Engine.Cuda.Convert
public import NN.Runtime.Autograd.Engine.Cuda.DGemm
public import NN.Runtime.Autograd.Engine.Cuda.Kernels

/-!
# FastKernels

Fast (runtime-only) kernels for the eager autograd tape.

This file is not used by the proof-linked compilation path. Instead, it provides drop-in
alternative `Tape.*` constructors for a few hot ops that are performance bottlenecks when
evaluated via the spec-layer definitions (which are written for proofs/clarity).

Current scope:
- `linear_fast` for 2D weights × 1D vector + bias (implemented as one `matmul` + bias add, so GPU
  mode routes the multiply through `FastMatmul`)
- `mse_loss_fast` for vector shapes (`.dim n .scalar`)

These are enabled from the PyTorch-style front-end (`NN/Runtime/Autograd/Torch/Core.lean`) via an
opt-in flag `Torch.Options.fastKernels`.
-/

@[expose] public section


namespace Runtime
namespace Autograd

open Spec
open Tensor

namespace FastKernels

/--
Precision selector for GPU-backed fast matmul over Lean `Float` tensors.

- `.fp32` routes through `Cuda.Buffer` and cuBLAS SGEMM, matching the precision used by the eager
  CUDA tensor-buffer stack.
- `.fp64` routes through the host `FloatArray` DGEMM bridge and cuBLAS DGEMM, preserving Lean
  `Float` precision for matmul-only research paths.
-/
inductive GpuMatmulPrecision where
  | fp32
  | fp64
deriving Repr, DecidableEq

/--
 Convert a length-`n` vector tensor to an `Array α`.

 This is a small runtime helper used by the fast kernels to run tight loops (`for i in [0:n]`).
 It is safe because the tensor shape carries `n`, and the result is constructed via `Array.ofFn`.
 -/
def vecToArray {α : Type} {n : Nat} :
    Tensor α (.dim n .scalar) → Array α
  | .dim f =>
      Array.ofFn (fun i : Fin n =>
        match f i with
        | .scalar x => x)

/--
 Convert an `(m×n)` matrix tensor into an array-of-rows representation.

 This is purely a representation change to make runtime loops faster/easier to write.
 -/
def matToRows {α : Type} {m n : Nat} :
    Tensor α (.dim m (.dim n .scalar)) → Array (Array α)
  | .dim rows =>
      Array.ofFn (fun i : Fin m =>
        match rows i with
        | .dim cols =>
            Array.ofFn (fun j : Fin n =>
              match cols j with
              | .scalar a => a))

/--
 Dot product of two length-`n` arrays.

 The fast kernels build arrays using `Array.ofFn`, so both inputs have the right size and `get!`
 is safe in this context.
 -/
def dot {α : Type} [Inhabited α] [Add α] [Mul α] [Zero α] (n : Nat) (a b : Array α) : α :=
  Id.run do
    let mut acc : α := 0
    for j in [0:n] do
      -- `a`/`b` come from `Array.ofFn`, so they have the right size; `get!` is safe here.
      acc := acc + a[j]! * b[j]!
    return acc

/--
`linearForward` computes the affine layer forward pass:

`y = W x + b`

for a 2D weight matrix `W : (outDim × inDim)` and 1D vectors `x : inDim`, `b : outDim`.

This is a runtime-only kernel intended for performance; compare the spec-layer definition, which is
optimized for proofs/clarity.
-/
def linearForward {α : Type} [Inhabited α] [Add α] [Mul α] [Zero α]
    {inDim outDim : Nat}
    (W : Tensor α (.dim outDim (.dim inDim .scalar)))
    (b : Tensor α (.dim outDim .scalar))
    (x : Tensor α (.dim inDim .scalar)) :
    Tensor α (.dim outDim .scalar) :=
  let wRows := matToRows (m := outDim) (n := inDim) W
  let bArr := vecToArray (n := outDim) b
  let xArr := vecToArray (n := inDim) x
  let yArr : Array α :=
    Array.ofFn (fun i : Fin outDim =>
      dot (α := α) inDim (wRows[i]!) xArr + bArr[i]!)
  Tensor.dim (fun i : Fin outDim => Tensor.scalar (yArr[i]!))

/--
`linearBackward` returns the standard affine-layer gradients.

If `y = W x + b` and the upstream gradient is `g = dL/dy`, then:

- `dL/dW = g ⊗ x` (outer product), i.e. `dW[i,k] = g[i] * x[k]`
- `dL/db = g`
- `dL/dx = Wᵀ g`, i.e. `dx[k] = Σ_i W[i,k] * g[i]`

These formulas match the usual backprop rule used by PyTorch autograd for `torch.nn.Linear`.
See PyTorch "Autograd mechanics": https://pytorch.org/docs/stable/notes/autograd.html
-/
def linearBackward {α : Type} [Inhabited α] [Add α] [Mul α] [Zero α]
    {inDim outDim : Nat}
    (W : Tensor α (.dim outDim (.dim inDim .scalar)))
    (x : Tensor α (.dim inDim .scalar))
    (dLdy : Tensor α (.dim outDim .scalar)) :
    (Tensor α (.dim outDim (.dim inDim .scalar)) × Tensor α (.dim outDim .scalar) × Tensor α (.dim
      inDim .scalar)) :=
  let wRows := matToRows (m := outDim) (n := inDim) W
  let xArr := vecToArray (n := inDim) x
  let gArr := vecToArray (n := outDim) dLdy
  let dWRows : Array (Array α) :=
    Array.ofFn (fun i : Fin outDim =>
      Array.ofFn (fun k : Fin inDim =>
        gArr[i]! * xArr[k]!))
  let dbArr : Array α := gArr
  let dxArr : Array α :=
    Array.ofFn (fun k : Fin inDim =>
      Id.run do
        let mut acc : α := 0
        for i in [0:outDim] do
          acc := acc + (wRows[i]!)[k]! * gArr[i]!
        return acc)
  let dW : Tensor α (.dim outDim (.dim inDim .scalar)) :=
    Tensor.dim (fun i : Fin outDim =>
      Tensor.dim (fun k : Fin inDim =>
        Tensor.scalar (dWRows[i]!)[k]!))
  let db : Tensor α (.dim outDim .scalar) :=
    Tensor.dim (fun i : Fin outDim => Tensor.scalar (dbArr[i]!))
  let dx : Tensor α (.dim inDim .scalar) :=
    Tensor.dim (fun k : Fin inDim => Tensor.scalar (dxArr[k]!))
  (dW, db, dx)

/-- View a length-`n` vector as an `(n × 1)` matrix (one column, row-major). -/
def vecAsColMat {α : Type} {n : Nat} (x : Tensor α (.dim n .scalar)) :
    Tensor α (.dim n (.dim 1 .scalar)) :=
  match x with
  | .dim f =>
      Tensor.dim (fun i : Fin n =>
        match f i with
        | .scalar a => Tensor.dim (fun _ : Fin 1 => Tensor.scalar a))

/-- View a length-`n` vector as a `(1 × n)` row matrix. -/
def vecAsRowMat {α : Type} {n : Nat} (x : Tensor α (.dim n .scalar)) :
    Tensor α (.dim 1 (.dim n .scalar)) :=
  match x with
  | .dim f =>
      Tensor.dim (fun _ : Fin 1 =>
        Tensor.dim (fun i : Fin n =>
          match f i with
          | .scalar a => Tensor.scalar a))

/-- Drop the trivial second dimension `(n × 1) → (n)`. -/
def colMatAsVec {α : Type} {n : Nat} (x : Tensor α (.dim n (.dim 1 .scalar))) :
    Tensor α (.dim n .scalar) :=
  match x with
  | .dim rows =>
      Tensor.dim (fun i : Fin n =>
        match rows i with
        | .dim col =>
            match col ⟨0, Nat.zero_lt_one⟩ with
            | .scalar a => Tensor.scalar a)

/--
Vector mean-squared error (MSE) with "mean" reduction:

`mse(yhat, target) = (Σ_i (yhat_i - target_i)^2) / n`.

This corresponds to `torch.nn.functional.mse_loss(..., reduction="mean")` for a 1D tensor.
Ref: https://pytorch.org/docs/stable/generated/torch.nn.functional.mse_loss.html
-/
def mseSpecVec {α : Type} [Inhabited α] [Add α] [Sub α] [Mul α] [Div α] [Zero α] [Coe Nat α]
    {n : Nat} (yhat target : Tensor α (.dim n .scalar)) : α :=
  let yArr := vecToArray (n := n) yhat
  let tArr := vecToArray (n := n) target
  let sumSq : α :=
    Id.run do
      let mut acc : α := 0
      for i in [0:n] do
        let d := yArr[i]! - tArr[i]!
        acc := acc + d * d
      return acc
  sumSq / (n : α)

/--
Gradient of `mseSpecVec` with respect to `yhat`.

If `mse = (Σ_i (yhat_i - target_i)^2) / n`, then:

`∂mse/∂yhat_i = 2 * (yhat_i - target_i) / n`.
-/
def mseDerivVec {α : Type} [Inhabited α] [Add α] [Sub α] [Mul α] [Div α] [Zero α] [One α] [Coe Nat
  α]
    {n : Nat} (yhat target : Tensor α (.dim n .scalar)) : Tensor α (.dim n .scalar) :=
  let yArr := vecToArray (n := n) yhat
  let tArr := vecToArray (n := n) target
  let two : α := (1 : α) + 1
  let c : α := two / (n : α)
  let gArr : Array α :=
    Array.ofFn (fun i : Fin n => (yArr[i]! - tArr[i]!) * c)
  Tensor.dim (fun i : Fin n => Tensor.scalar (gArr[i]!))

/--
Fast (runtime-only) 2D matmul kernel.

This is a tight-loop kernel (array-of-rows representation) intended to avoid the overhead of the
spec-layer definitions when running eager autograd.
-/
def matmulForward {α : Type} [Context α]
    {m n p : Nat}
    (a : Tensor α (.dim m (.dim n .scalar)))
    (b : Tensor α (.dim n (.dim p .scalar))) :
    Tensor α (.dim m (.dim p .scalar)) :=
  let matmulLean (a : Tensor α (.dim m (.dim n .scalar))) (b : Tensor α (.dim n (.dim p .scalar))) :
      Tensor α (.dim m (.dim p .scalar)) :=
    let aArr := matToRows (α := α) (m := m) (n := n) a
    let bArr := matToRows (α := α) (m := n) (n := p) b
    let cArr : Array (Array α) :=
      Array.ofFn (fun i : Fin m =>
        Array.ofFn (fun k : Fin p =>
          Id.run do
            let mut acc : α := 0
            for j in [0:n] do
              acc := acc + (aArr[i]!)[j]! * (bArr[j]!)[k]!
            return acc))
    Tensor.dim (fun i : Fin m =>
      Tensor.dim (fun k : Fin p =>
        Tensor.scalar (cArr[i]!)[k]!))
  matmulLean a b

/--
Fast (runtime-only) matmul dispatch.

This is used by the eager autograd tape fast-kernel variant `Tape.matmul_fast`. The `useGpu` flag is
treated as a hint: the default instance ignores it, while the `Float` instance can route to CUDA.
-/
class FastMatmul (α : Type) where
  matmul2dFast :
      {m n p : Nat} →
      (useGpu : Bool) →
      (gpuPrecision : GpuMatmulPrecision) →
      Tensor α (.dim m (.dim n .scalar)) →
      Tensor α (.dim n (.dim p .scalar)) →
      Tensor α (.dim m (.dim p .scalar))

/-- Default `FastMatmul`: always use the CPU fast-kernel implementation. -/
instance (priority := 10) {α : Type} [Context α] : FastMatmul α where
  matmul2dFast := fun {m n p} _useGpu _gpuPrecision a b =>
    matmulForward (α := α) (m := m) (n := n) (p := p) a b

namespace Cuda

/-- 2D matmul forward via cuBLAS DGEMM (`torchlean_dgemm_cuda` / `Cuda.torchleanDgemmCuda`). -/
def matmulForwardcuBLAS64 {m n p : Nat}
    (a : Tensor Float (.dim m (.dim n .scalar)))
    (b : Tensor Float (.dim n (.dim p .scalar))) :
    Tensor Float (.dim m (.dim p .scalar)) :=
  let aRows := matToRows a
  let bRows := matToRows b
  let flatA : FloatArray :=
    Id.run do
      let mut out : Array Float := Array.mkEmpty (m * n)
      for row in aRows do
        for x in row do
          out := out.push x
      return FloatArray.mk out
  let flatB : FloatArray :=
    Id.run do
      let mut out : Array Float := Array.mkEmpty (n * p)
      for row in bRows do
        for x in row do
          out := out.push x
      return FloatArray.mk out
  let flatC := Runtime.Autograd.Cuda.torchleanDgemmCuda flatA flatB
    (UInt32.ofNat m) (UInt32.ofNat n) (UInt32.ofNat p)
  Tensor.dim (fun i : Fin m =>
    Tensor.dim (fun j : Fin p =>
      Tensor.scalar (flatC.get! (i.val * p + j.val))))

/--
2D matmul forward via the float32 CUDA buffer stack.

This path uploads Lean `Float` values to `Cuda.Buffer` (rounding to float32), calls the existing
`Buffer.bmm` SGEMM implementation with `batch = 1`, then downloads the float32 result back to Lean
`Float`.
-/
def matmulForwardcuBLAS32 {m n p : Nat}
    (a : Tensor Float (.dim m (.dim n .scalar)))
    (b : Tensor Float (.dim n (.dim p .scalar))) :
    Tensor Float (.dim m (.dim p .scalar)) :=
  let aBuf := Runtime.Autograd.Cuda.Buffer.ofFloatArray
    (Runtime.Autograd.Cuda.Convert.flattenFloat (s := .dim m (.dim n .scalar)) a)
  let bBuf := Runtime.Autograd.Cuda.Buffer.ofFloatArray
    (Runtime.Autograd.Cuda.Convert.flattenFloat (s := .dim n (.dim p .scalar)) b)
  let cBuf := Runtime.Autograd.Cuda.Buffer.bmm aBuf bBuf
    (UInt32.ofNat 1) (UInt32.ofNat m) (UInt32.ofNat n) (UInt32.ofNat p)
  Runtime.Autograd.Cuda.Convert.unflattenFloatUnsafe
    (s := .dim m (.dim p .scalar))
    (Runtime.Autograd.Cuda.Buffer.toFloatArray cBuf)

/-- Default GPU matmul alias: the FP64 cuBLAS path for Lean `Float` tensors. -/
abbrev matmulForwardcuBLAS := @matmulForwardcuBLAS64

/-- Dispatch to the requested GPU matmul precision. -/
def matmulForwardcuBLASWith (precision : GpuMatmulPrecision) {m n p : Nat}
    (a : Tensor Float (.dim m (.dim n .scalar)))
    (b : Tensor Float (.dim n (.dim p .scalar))) :
    Tensor Float (.dim m (.dim p .scalar)) :=
  match precision with
  | .fp32 => matmulForwardcuBLAS32 (m := m) (n := n) (p := p) a b
  | .fp64 => matmulForwardcuBLAS64 (m := m) (n := n) (p := p) a b

end Cuda

/-- `Float` instance: use an explicit cuBLAS FP32/FP64 path when `useGpu=true`. -/
instance (priority := 1000) : FastMatmul Float where
  matmul2dFast := fun {m n p} useGpu gpuPrecision a b =>
    if useGpu then
      Cuda.matmulForwardcuBLASWith gpuPrecision (m := m) (n := n) (p := p) a b
    else
      matmulForward (α := Float) (m := m) (n := n) (p := p) a b

/--
Fast affine forward `y = W x + b` using a single 2D matmul (`W` is `(outDim × inDim)`, `x` as
`(inDim × 1)`), then `add_spec` for the bias.

When `useGpu` is true and `α = Float`, `FastMatmul` routes the multiply to the requested cuBLAS
precision (if built with `-K cuda=true`); otherwise the same path uses the CPU tight-loop matmul.
-/
def linearForwardFast {α : Type} [Add α] [FastMatmul α] {inDim outDim : Nat} (useGpu : Bool)
    (gpuPrecision : GpuMatmulPrecision := .fp32)
    (W : Tensor α (.dim outDim (.dim inDim .scalar)))
    (b : Tensor α (.dim outDim .scalar))
    (x : Tensor α (.dim inDim .scalar)) :
    Tensor α (.dim outDim .scalar) :=
  let wxCol :=
    FastMatmul.matmul2dFast (α := α) (m := outDim) (n := inDim) (p := 1) useGpu gpuPrecision W
      (vecAsColMat x)
  let wx := colMatAsVec wxCol
  addSpec (α := α) (s := .dim outDim .scalar) wx b

/--
Fast affine backward using three matmul-shaped operations (two GEMMs + bias copy).

Matches the usual `Linear` autograd: `dW = g xᵀ`, `db = g`, `dx = Wᵀ g`.
-/
def linearBackwardFast {α : Type} [Add α] [Mul α] [Zero α] [FastMatmul α] {inDim outDim : Nat}
    (useGpu : Bool)
    (gpuPrecision : GpuMatmulPrecision := .fp32)
    (W : Tensor α (.dim outDim (.dim inDim .scalar)))
    (x : Tensor α (.dim inDim .scalar))
    (dLdy : Tensor α (.dim outDim .scalar)) :
    (Tensor α (.dim outDim (.dim inDim .scalar)) × Tensor α (.dim outDim .scalar) × Tensor α (.dim
      inDim .scalar)) :=
  let dW :=
    FastMatmul.matmul2dFast (α := α) (m := outDim) (n := 1) (p := inDim) useGpu gpuPrecision
      (vecAsColMat dLdy) (vecAsRowMat x)
  let db := dLdy
  let WT := matrixTransposeSpec (α := α) (m := outDim) (n := inDim) W
  let dxCol :=
    FastMatmul.matmul2dFast (α := α) (m := inDim) (n := outDim) (p := 1) useGpu gpuPrecision WT
      (vecAsColMat dLdy)
  let dx := colMatAsVec dxCol
  (dW, db, dx)

/--
Fast (runtime-only) matmul backward rule computed via two matmuls and transposes.

If `y = a*b` and `g = dL/dy`, then:
- `dL/da = g * bᵀ`
- `dL/db = aᵀ * g`
-/
def matmulBackward {α : Type} [Context α] [FastMatmul α]
    {m n p : Nat} (useGpu : Bool)
    (gpuPrecision : GpuMatmulPrecision := .fp32)
    (a : Tensor α (.dim m (.dim n .scalar)))
    (b : Tensor α (.dim n (.dim p .scalar)))
    (dLdy : Tensor α (.dim m (.dim p .scalar))) :
    (Tensor α (.dim m (.dim n .scalar)) × Tensor α (.dim n (.dim p .scalar))) :=
  let bt := matrixTransposeSpec (α := α) (m := n) (n := p) b
  let aT := matrixTransposeSpec (α := α) (m := m) (n := n) a
  let dA := FastMatmul.matmul2dFast (α := α) (m := m) (n := p) (p := n) useGpu gpuPrecision dLdy bt
  let dB := FastMatmul.matmul2dFast (α := α) (m := n) (n := m) (p := p) useGpu gpuPrecision aT
    dLdy
  (dA, dB)

end FastKernels

namespace Tape

/--
 Fast runtime variant of `Tape.linear`.

 This is intended as a drop-in replacement for the affine layer `y = W x + b` when the shapes are
 exactly `W : (outDim×inDim)`, `b : outDim`, `x : inDim`. It implements the multiply via
 `FastMatmul` (one GEMM-shaped matmul + bias), so `Torch.Options.useGpu` can route `Float` matmuls to
 cuBLAS when enabled.

 PyTorch comparison: `torch.nn.Linear` / `torch.nn.functional.linear`.
 Reference: https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
 -/
def linearFast {α : Type} [Add α] [Mul α] [Zero α] [DecidableEq Shape] [FastKernels.FastMatmul α]
  {inDim outDim : Nat}
  (useGpu : Bool) (gpuPrecision : FastKernels.GpuMatmulPrecision := .fp32)
  (t : Tape α) (wId bId xId : Nat) : Result (Tape α × Nat) := do
  let W ← requireValue (α := α) (t := t) (s := .dim outDim (.dim inDim .scalar)) wId
  let b ← requireValue (α := α) (t := t) (s := .dim outDim .scalar) bId
  let x ← requireValue (α := α) (t := t) (s := .dim inDim .scalar) xId
  let y := FastKernels.linearForwardFast (α := α) (inDim := inDim) (outDim := outDim) useGpu
    gpuPrecision W b x
  let node : Node α :=
    { name := some "linear_fast"
      value := AnyTensor.mk y
      requires_grad := true
      parents := [wId, bId, xId]
      backward := fun dLdyAny => do
        let dLdy ← requireGrad (α := α) (τ := .dim outDim .scalar) dLdyAny
        let (dW, db, dx) := FastKernels.linearBackwardFast (α := α) (inDim := inDim)
          (outDim := outDim) useGpu gpuPrecision W x dLdy
        pure [(wId, AnyTensor.mk dW), (bId, AnyTensor.mk db), (xId, AnyTensor.mk dx)]
    }
  pure (t.addNode node)

/--
Fast runtime variant of `Tape.matmul` for 2D tensors.

This is a runtime-only optimization: it uses the eager fast-kernel implementation
`FastKernels.matmulForward` rather than the spec-layer definition.
-/
def matmulFast {α : Type} [Context α] [DecidableRel ((· > ·) : α → α → Prop)] [DecidableEq Shape]
  [FastKernels.FastMatmul α] {m n p : Nat} (useGpu : Bool)
  (gpuPrecision : FastKernels.GpuMatmulPrecision := .fp32) (t : Tape α) (aId bId : Nat) :
    Result (Tape α × Nat) := do
  let a ← requireValue (α := α) (t := t) (s := .dim m (.dim n .scalar)) aId
  let b ← requireValue (α := α) (t := t) (s := .dim n (.dim p .scalar)) bId
  let y := FastKernels.FastMatmul.matmul2dFast (α := α) (m := m) (n := n) (p := p) useGpu
    gpuPrecision a b
  let node : Node α :=
    { name := some "matmul_fast"
      value := AnyTensor.mk y
      requires_grad := true
      parents := [aId, bId]
      backward := fun dLdyAny => do
        let dLdy ← requireGrad (α := α) (τ := .dim m (.dim p .scalar)) dLdyAny
        let (dA, dB) :=
          FastKernels.matmulBackward (α := α) (m := m) (n := n) (p := p) useGpu gpuPrecision a b
            dLdy
        pure [(aId, AnyTensor.mk dA), (bId, AnyTensor.mk dB)]
    }
  pure (t.addNode node)

/--
 Fast runtime variant of `Tape.mse_loss` for vector shapes.

 - If `s = .dim n .scalar`, we use `FastKernels.mseSpecVec`/`FastKernels.mseDerivVec`.
 - If `s = .scalar` or a non-vector tensor, we fall back to the generic `Tape.mse_loss`.

 PyTorch comparison: `torch.nn.functional.mse_loss(..., reduction=\"mean\")`.
 -/
def mseLossFast {α : Type}
  [Inhabited α] [Add α] [Sub α] [Mul α] [Div α] [Zero α] [One α] [Coe Nat α] [DecidableEq Shape]
  {s : Shape} (t : Tape α) (yhatId targetId : Nat) : Result (Tape α × Nat) := by
  match s with
  | .scalar =>
      -- Scalars are already cheap; reuse the generic node.
      exact mseLoss (α := α) (t := t) (s := Shape.scalar) yhatId targetId
  | .dim n .scalar =>
      let go : Result (Tape α × Nat) := do
        let yhat ← requireValue (α := α) (t := t) (s := .dim n .scalar) yhatId
        let target ← requireValue (α := α) (t := t) (s := .dim n .scalar) targetId
        let lossVal : α := FastKernels.mseSpecVec (α := α) (n := n) yhat target
        let y : Tensor α Shape.scalar := Tensor.scalar lossVal
        let node : Node α :=
          { name := some "mse_loss_fast"
            value := AnyTensor.mk y
            requires_grad := true
            parents := [yhatId, targetId]
            backward := fun dLdyAny => do
              let dLdy ← requireGrad (α := α) (τ := Shape.scalar) dLdyAny
              let g : α := Tensor.toScalar dLdy
              let dYhat0 := FastKernels.mseDerivVec (α := α) (n := n) yhat target
              let dYhat : Tensor α (.dim n .scalar) := scaleSpec (α := α) (s := .dim n .scalar)
                dYhat0 g
              let dTarget : Tensor α (.dim n .scalar) := subSpec (fill (0 : α) (.dim n .scalar))
                dYhat
              pure [(yhatId, AnyTensor.mk dYhat), (targetId, AnyTensor.mk dTarget)]
          }
        pure (t.addNode node)
      exact go
  | .dim _ _ =>
      -- Fallback for non-vector shapes.
      exact mseLoss (α := α) (t := t) (s := s) yhatId targetId

end Tape

end Autograd
end Runtime
