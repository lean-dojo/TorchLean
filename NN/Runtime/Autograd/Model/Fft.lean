/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Model.Layers.Core
import Mathlib.Algebra.Order.Algebra

/-!
# Fourier transforms along a selected axis

TorchLean’s layer/model definitions are scalar-polymorphic: a model runs over whatever scalar type
$\alpha$ you instantiate it with (for example `Float`, `IEEE32Exec`, or $\mathbb{R}$). A “real FFT”
would normally *change* the scalar type (real $\to$ complex), but TorchLean’s `Layer` does not
support changing the scalar type mid-model.

So this module provides **complex-domain** transforms: `fft` and `ifft` as layers that assume the
$\alpha$ already behaves like a complex field (for example `TorchLean.Complex
IEEE32Exec`, selected via `--arithmetic=complex`).

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
  interval/oracle backend) to relate those executable trigonometric approximations to real
  `sin/cos`.
-/

@[expose] public section

namespace Runtime
namespace Autograd
namespace Model

open Spec TorchLean
open TorchLean TorchLean.Tensor

namespace Layers

namespace FFT

/-!
We build twiddle factors using only the `Context` interface:
$I=\sqrt{-1}$ and $e^{-i\theta}=\cos\theta-I\sin\theta$.

This is intended to be instantiated with TorchLean’s own complex scalar
`TorchLean.Complex β` (for some base scalar $\beta$). For real-only scalar backends, the formulas
are not meaningful.
-/

/-- The imaginary unit, represented as `sqrt(-1)` in the ambient scalar type. -/
def I {α : Type} [TorchLean.Storage α] [Context α] : α :=
  MathFunctions.sqrt (-1)

/-- Twiddle factor $e^{-2\pi i jk/n}$ written as $\cos\theta-i\sin\theta$. -/
def twiddle {α : Type} [TorchLean.Storage α] [Context α] (n : Nat) (j k : Nat) : α :=
  let twoPi : α := 2 * MathFunctions.pi
  let ang : α := twoPi * (j : α) * (k : α) / (n : α)
  MathFunctions.cos ang - I (α := α) * MathFunctions.sin ang

/-- Twiddle factor $e^{2\pi i jk/n}$ written as $\cos\theta+i\sin\theta$. -/
def twiddleInv {α : Type} [TorchLean.Storage α] [Context α] (n : Nat) (j k : Nat) : α :=
  let twoPi : α := 2 * MathFunctions.pi
  let ang : α := twoPi * (j : α) * (k : α) / (n : α)
  MathFunctions.cos ang + I (α := α) * MathFunctions.sin ang

/-- DFT matrix $F\in\mathbb{C}^{n\times n}$ with entries $F_{k,j}=e^{-2\pi i jk/n}$. -/
def dftMatrix {α : Type} [TorchLean.Storage α] [Context α] (n : Nat) :
    Tensor α [n, n] :=
  Tensor.dim (fun k =>
    Tensor.dim (fun j =>
      Tensor.scalar (twiddle (α := α) (n := n) (j := j.val) (k := k.val))))

/-- Inverse DFT matrix $F^{-1}\in\mathbb{C}^{n\times n}$ with entries
$(F^{-1})_{j,k}=e^{2\pi i jk/n}/n$. -/
def idftMatrix {α : Type} [TorchLean.Storage α] [Context α] (n : Nat) :
    Tensor α [n, n] :=
  Tensor.dim (fun j =>
    Tensor.dim (fun k =>
      Tensor.scalar (twiddleInv (α := α) (n := n) (j := j.val) (k := k.val) / (n : α))))

namespace Internal

/--
Implementation of FFT along the outermost axis of a tensor.

This applies the DFT to the leading dimension `n` of a shape `dim n rest` by:
1. reshaping to a matrix `n × (numel rest)`,
2. left-multiplying by the `n×n` DFT matrix, then
3. reshaping back.

The public `fftAtDepth` operation moves an arbitrary axis here and restores the original axis order.
-/
def fftLeadingAxis (n : Nat) (rest : Shape) :
    Layer (rest.prependDim n) (rest.prependDim n) :=
  let sIn : Shape := rest.prependDim n
  let cols : Nat := Spec.Shape.size rest
  let sMat : Shape := [n, cols]
  have hSz : Spec.Shape.size sIn = Spec.Shape.size sMat := by
    simp [Spec.Shape.size, sIn, sMat, cols]
  { stateShapes := []
    initState := .nil
    forward := fun _ {α} _ _ =>
      fun {m} _ _ =>
        fun x =>
          (show m (RefTy (m := m) (α := α) sIn) from do
            let xMat ← Runtime.Autograd.Model.reshape (m := m) (α := α)
              (s₁ := sIn) (s₂ := sMat) x hSz
            let f : Tensor α [n, n] := dftMatrix (α := α) n
            let fR ← Runtime.Autograd.Model.const (m := m) (α := α) (s := [n, n]) f
            let yMat ←
              Runtime.Autograd.Model.matmul (m := m) (α := α)
                (batchA := .scalar) (batchB := .scalar) (batch := .scalar)
                (mDim := n) (nDim := n) (pDim := cols) fR xMat
            Runtime.Autograd.Model.reshape (m := m) (α := α) (s₁ := sMat) (s₂ := sIn) yMat hSz.symm)
  }

/-- Apply a sequence of `swapAdjacentAtDepth` operations (shape-indexed permutation primitive). -/
def permuteBySwaps {α : Type} [TorchLean.Storage α] [Context α]
    {m : Type → Type} [Monad m] [Ops (m := m) (α := α)]
    (x : Σ s : Shape, RefTy (m := m) (α := α) s) :
    (swaps : List Nat) → m (Σ s' : Shape, RefTy (m := m) (α := α) s')
  | .nil => pure x
  | .cons d ds => do
      let y ← Runtime.Autograd.Model.swapAdjacentAtDepth (m := m) (α := α) (s := x.fst) d x.snd
      permuteBySwaps (α := α) (m := m) ⟨x.fst.swapAdjacentAtDepth d, y⟩ ds

end Internal

/-!
FFT along an axis at a given depth (0-based from the outermost).

This is implemented by swapping the target axis outward (one adjacent swap per step) until it
reaches depth `0`, applying the outer-axis implementation, then swapping back.

If $\mathtt{depth}\ge\operatorname{rank}(s)$, this layer is the identity.
-/
def fftAtDepth : {s : Shape} → Nat → Layer s s
  | s, depth =>
    { stateShapes := []
      initState := .nil
      forward := fun mode {α} _ _ =>
        fun {m} _ _ =>
          fun x =>
            (show m (RefTy (m := m) (α := α) s) from do
              if depth ≥ Spec.Shape.rank s then
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
                    let y0 ← (Internal.fftLeadingAxis (n := nDim) (rest := rest)).forward mode
                      (α := α) (m := m) x0
                    let yFront : Σ s' : Shape, RefTy (m := m) (α := α) s' :=
                      ⟨.dim nDim rest, y0⟩
                    let yBack ← Internal.permuteBySwaps (α := α) (m := m) yFront swapsBack
                    if h : yBack.fst = s then
                      pure (h ▸ yBack.snd)
                    else
                      pure x)
    }

end FFT

end Layers
end Model
end Autograd
end Runtime
