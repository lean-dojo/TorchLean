/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.TorchLean.NN

import Mathlib.Algebra.Order.Algebra

/-!
# FFT (1D) building blocks

TorchLean’s layer/model definitions are scalar-polymorphic: a model runs over whatever scalar type
`α` you instantiate it with (e.g. `Float`, `IEEE32Exec`, `ℝ`, etc.).  A “real FFT” would normally
*change* the scalar type (real → complex), but TorchLean’s `LayerDef` does not support changing the
scalar type mid-model.

So this module provides **complex-domain** transforms: `fft` and `ifft` as layers that assume the
chosen scalar type `α` already behaves like a complex field (for example `TorchLean.Complex
IEEE32Exec`, selected via `--dtype=complex`).

Implementation note: we define `fft`/`ifft` as multiplication by explicit DFT matrices (so they are
purely built from existing ops like `const` and `matmul`).  This is correctness-first and keeps the
transform differentiable under the existing autograd rules.  It is not optimized for large `n`.

Numerics note:
- Over mathlib’s `ℂ`, the corresponding DFT/IDFT inversion facts are proved in
  `NN.Proofs.Analysis.Fft` (and the bridge to these `twiddle`/matrix definitions is in
  `NN.Proofs.Analysis.FftBridge`).
- For executable `IEEE32Exec`, `sin`/`cos` are implemented deterministically in Lean (see
  `NN.Floats.IEEEExec.Exec32`). This makes FFT execution reproducible across platforms. Proving
  tight end-to-end *accuracy* bounds for FFT still requires a separate analysis layer (or an
  interval/oracle backend) to relate those executable trigonometric approximations to real `sin/cos`.
-/

@[expose] public section

namespace Runtime
namespace Autograd
namespace TorchLean

open Spec
open Tensor

namespace NN

namespace FFT1D

/-- Shape abbreviation for a length-`n` vector. -/
abbrev vec (n : Nat) : Shape := .dim n .scalar

/-- Shape abbreviation for an `m×n` matrix. -/
abbrev mat (m n : Nat) : Shape := .dim m (.dim n .scalar)

/-!
We build twiddle factors using only the `Context` interface:
`I := sqrt(-1)` and `e^{-iθ} = cos θ - I * sin θ`.

This is intended to be instantiated with TorchLean’s own complex scalar
`TorchLean.Complex β` (for some base scalar `β`).  For real-only scalar backends, the formulas are
not meaningful.
-/

/-- The imaginary unit, represented as `sqrt(-1)` in the ambient scalar type. -/
def I {α : Type} [Context α] : α :=
  MathFunctions.sqrt Numbers.neg_one

/-- Twiddle factor `exp(-2π i * j*k / n)` written as `cos θ - i sin θ`. -/
def twiddle {α : Type} [Context α] (n : Nat) (j k : Nat) : α :=
  let twoPi : α := Numbers.two * MathFunctions.pi
  let ang : α := twoPi * (j : α) * (k : α) / (n : α)
  MathFunctions.cos ang - I (α := α) * MathFunctions.sin ang

/-- Twiddle factor `exp(+2π i * j*k / n)` written as `cos θ + i sin θ`. -/
def twiddleInv {α : Type} [Context α] (n : Nat) (j k : Nat) : α :=
  let twoPi : α := Numbers.two * MathFunctions.pi
  let ang : α := twoPi * (j : α) * (k : α) / (n : α)
  MathFunctions.cos ang + I (α := α) * MathFunctions.sin ang

/-- DFT matrix `F : n×n` with entries `F[k,j] = exp(-2π i j*k / n)`. -/
def dftMatrix {α : Type} [Context α] (n : Nat) : Tensor α (mat n n) :=
  Tensor.dim (fun k =>
    Tensor.dim (fun j =>
      Tensor.scalar (twiddle (α := α) (n := n) (j := j.val) (k := k.val))))

/-- Inverse DFT matrix `F⁻¹ : n×n` with entries `F⁻¹[j,k] = exp(+2π i j*k / n) / n`. -/
def idftMatrix {α : Type} [Context α] (n : Nat) : Tensor α (mat n n) :=
  Tensor.dim (fun j =>
    Tensor.dim (fun k =>
      Tensor.scalar (twiddleInv (α := α) (n := n) (j := j.val) (k := k.val) / (n : α))))

/--
FFT along the outermost axis of a tensor.

This applies the DFT to the leading dimension `n` of a shape `dim n rest` by:
1. reshaping to a matrix `n × (numel rest)`,
2. left-multiplying by the `n×n` DFT matrix, then
3. reshaping back.

This is the most generally useful primitive for building N-D FFTs: you can permute an axis to the
front, call `fftDim0`, then permute back.
-/
def fftDim0 (n : Nat) (rest : Shape) :
    LayerDef (.dim n rest) (.dim n rest) :=
  let sIn : Shape := .dim n rest
  let cols : Nat := Shape.size rest
  let sMat : Shape := mat n cols
  have hSz : Shape.size sIn = Shape.size sMat := by
    simp [Shape.size, sIn, sMat, cols]
  { paramShapes := []
    initParams := .nil
    forward := fun _ {α} _ _ =>
      fun {m} _ _ =>
        fun x =>
          (show m (RefTy (m := m) (α := α) sIn) from do
            let xMat ← TorchLean.reshape (m := m) (α := α) (s₁ := sIn) (s₂ := sMat) x hSz
            let f : Tensor α (mat n n) := dftMatrix (α := α) n
            let fR ← TorchLean.const (m := m) (α := α) (s := mat n n) f
            let yMat ←
              TorchLean.matmul (m := m) (α := α)
                (mDim := n) (nDim := n) (pDim := cols) fR xMat
            TorchLean.reshape (m := m) (α := α) (s₁ := sMat) (s₂ := sIn) yMat hSz.symm)
  }

/--
Inverse FFT along the outermost axis of a tensor (uses the inverse DFT matrix).

See `fftDim0` for the implementation strategy.
-/
def ifftDim0 (n : Nat) (rest : Shape) :
    LayerDef (.dim n rest) (.dim n rest) :=
  let sIn : Shape := .dim n rest
  let cols : Nat := Shape.size rest
  let sMat : Shape := mat n cols
  have hSz : Shape.size sIn = Shape.size sMat := by
    simp [Shape.size, sIn, sMat, cols]
  { paramShapes := []
    initParams := .nil
    forward := fun _ {α} _ _ =>
      fun {m} _ _ =>
        fun x =>
          (show m (RefTy (m := m) (α := α) sIn) from do
            let xMat ← TorchLean.reshape (m := m) (α := α) (s₁ := sIn) (s₂ := sMat) x hSz
            let f : Tensor α (mat n n) := idftMatrix (α := α) n
            let fR ← TorchLean.const (m := m) (α := α) (s := mat n n) f
            let yMat ←
              TorchLean.matmul (m := m) (α := α)
                (mDim := n) (nDim := n) (pDim := cols) fR xMat
            TorchLean.reshape (m := m) (α := α) (s₁ := sMat) (s₂ := sIn) yMat hSz.symm)
  }

/-- FFT on matrices: apply the DFT along the leading dimension (`n×width`). -/
abbrev fftMat (n width : Nat) : LayerDef (mat n width) (mat n width) :=
  fftDim0 (n := n) (rest := .dim width .scalar)

/-- Inverse FFT on matrices: apply the inverse DFT along the leading dimension (`n×width`). -/
abbrev ifftMat (n width : Nat) : LayerDef (mat n width) (mat n width) :=
  ifftDim0 (n := n) (rest := .dim width .scalar)

/-- FFT on vectors. -/
abbrev fftVec (n : Nat) : LayerDef (vec n) (vec n) :=
  fftDim0 (n := n) (rest := .scalar)

/-- Inverse FFT on vectors. -/
abbrev ifftVec (n : Nat) : LayerDef (vec n) (vec n) :=
  ifftDim0 (n := n) (rest := .scalar)

namespace Internal

/-- Apply a sequence of `swapAdjacentAtDepth` operations (shape-indexed permutation primitive). -/
def permuteBySwaps {α : Type} [Context α] [DecidableEq Shape]
    {m : Type → Type} [Monad m] [Ops (m := m) (α := α)]
    (x : Σ s : Shape, RefTy (m := m) (α := α) s) :
    (swaps : List Nat) → m (Σ s' : Shape, RefTy (m := m) (α := α) s')
  | [] => pure x
  | d :: ds => do
      let y ← TorchLean.swapAdjacentAtDepth (m := m) (α := α) (s := x.fst) d x.snd
      permuteBySwaps (α := α) (m := m) ⟨x.fst.swapAdjacentAtDepth d, y⟩ ds

end Internal

/-!
FFT along an axis at a given depth (0-based from the outermost).

This is implemented by swapping the target axis outward (one adjacent swap per step) until it
reaches depth `0`, applying `fftDim0`, then swapping back.

If `depth ≥ Shape.rank s`, this layer is the identity.
-/
def fftAtDepth : {s : Shape} → Nat → LayerDef s s
  | s, depth =>
    { paramShapes := []
      initParams := .nil
      forward := fun mode {α} _ _ =>
        fun {m} _ _ =>
          fun x =>
            (show m (RefTy (m := m) (α := α) s) from do
              if depth ≥ Shape.rank s then
                pure x
              else
                let swapsToFront : List Nat := (List.range depth).reverse
                let swapsBack : List Nat := List.range depth
                let xFront ← Internal.permuteBySwaps (α := α) (m := m) ⟨s, x⟩ swapsToFront
                match xFront with
                | ⟨.scalar, x0⟩ =>
                    -- Unreachable (rank is preserved by swaps and we checked `depth < rank s`), but
                    -- keep the fallback total.
                    let yBack ← Internal.permuteBySwaps (α := α) (m := m) ⟨.scalar, x0⟩ swapsBack
                    if h : yBack.fst = s then
                      pure (h ▸ yBack.snd)
                    else
                      pure x
                | ⟨.dim nDim rest, x0⟩ =>
                    let y0 ← (fftDim0 (n := nDim) (rest := rest)).forward mode (α := α) (m := m) x0
                    let yFront : Σ s' : Shape, RefTy (m := m) (α := α) s' :=
                      ⟨.dim nDim rest, y0⟩
                    let yBack ← Internal.permuteBySwaps (α := α) (m := m) yFront swapsBack
                    if h : yBack.fst = s then
                      pure (h ▸ yBack.snd)
                    else
                      pure x)
    }

/-- Inverse FFT along an axis at a given depth (see `fftAtDepth`). -/
def ifftAtDepth : {s : Shape} → Nat → LayerDef s s
  | s, depth =>
    { paramShapes := []
      initParams := .nil
      forward := fun mode {α} _ _ =>
        fun {m} _ _ =>
          fun x =>
            (show m (RefTy (m := m) (α := α) s) from do
              if depth ≥ Shape.rank s then
                pure x
              else
                let swapsToFront : List Nat := (List.range depth).reverse
                let swapsBack : List Nat := List.range depth
                let xFront ← Internal.permuteBySwaps (α := α) (m := m) ⟨s, x⟩ swapsToFront
                match xFront with
                | ⟨.scalar, x0⟩ =>
                    let yBack ← Internal.permuteBySwaps (α := α) (m := m) ⟨.scalar, x0⟩ swapsBack
                    if h : yBack.fst = s then
                      pure (h ▸ yBack.snd)
                    else
                      pure x
                | ⟨.dim nDim rest, x0⟩ =>
                    let y0 ← (ifftDim0 (n := nDim) (rest := rest)).forward mode (α := α) (m := m) x0
                    let yFront : Σ s' : Shape, RefTy (m := m) (α := α) s' :=
                      ⟨.dim nDim rest, y0⟩
                    let yBack ← Internal.permuteBySwaps (α := α) (m := m) yFront swapsBack
                    if h : yBack.fst = s then
                      pure (h ▸ yBack.snd)
                    else
                      pure x)
    }

end FFT1D

end NN
end TorchLean
end Autograd
end Runtime
