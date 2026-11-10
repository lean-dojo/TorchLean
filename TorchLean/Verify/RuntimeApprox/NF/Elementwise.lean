/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Verify.RuntimeApprox.NF.Scalar

/-!
# NF Elementwise Bounds: Core Arithmetic Budgets

Per-entry error budgets for the elementwise `NF` operations, each a spec tensor computed from the
runtime input. A tensor bound takes the `linfNorm` of its budget tensor as the output epsilon.
-/

@[expose] public section

namespace TorchLean
namespace NF

open TorchLean.Tensor
open Robustness

noncomputable section

open FloatLib FloatLib.Numerics FloatLib.Floats.Formats
open Flocq

variable {β : Radix} {fexp : ℤ → ℤ} [ValidExp fexp]
variable {rnd : ℝ → ℤ} [ValidRndToNearest rnd]

local notation "R" => NF β fexp rnd

-- Elementwise bounds lifted to tensors via `linfNorm`.

/--
Per-entry bound tensor for addition.

With `a` and `b` the real values of corresponding entries of `xR` and `yR`, each entry of
`addBoundTensor epsx epsy xR yR` is `epsx + epsy + ulp (a + b) / 2`. Its `linfNorm` is the
output epsilon of `ApproxTensor.add`.
-/
def addBoundTensor {s : Shape} (epsx epsy : ℝ) (xR yR : Tensor R s) : Tensor ℝ s :=
  zipWith
    (fun a b =>
      epsx + epsy + ulp β fexp (a + b) / 2)
    (Tensor.toReal (α := R) (toSpec := NF.toReal) xR)
    (Tensor.toReal (α := R) (toSpec := NF.toReal) yR)

/--
Per-entry bound tensor for subtraction.

Each entry is `epsx + epsy + ulp (a - b) / 2` at the real values `a` and `b` of corresponding
entries of `xR` and `yR`, and its `linfNorm` is the output epsilon of `ApproxTensor.sub`.
-/
def subBoundTensor {s : Shape} (epsx epsy : ℝ) (xR yR : Tensor R s) : Tensor ℝ s :=
  zipWith
    (fun a b =>
      epsx + epsy + ulp β fexp (a - b) / 2)
    (Tensor.toReal (α := R) (toSpec := NF.toReal) xR)
    (Tensor.toReal (α := R) (toSpec := NF.toReal) yR)

/--
Per-entry bound tensor for multiplication.

Each entry is the `approx_mul` budget at the real values `a` and `b` of corresponding entries,
`(|a| + epsx) * epsy + (|b| + epsy) * epsx` for the input errors and their cross terms plus
`ulp (a * b) / 2` for the rounding.
-/
def mulBoundTensor {s : Shape} (epsx epsy : ℝ) (xR yR : Tensor R s) : Tensor ℝ s :=
  zipWith
    (fun a b =>
      (|a| + epsx) * epsy + (|b| + epsy) * epsx + ulp β fexp (a * b) / 2)
    (Tensor.toReal (α := R) (toSpec := NF.toReal) xR)
    (Tensor.toReal (α := R) (toSpec := NF.toReal) yR)

/--
Per-entry bound tensor for scaling by a runtime constant `c`.

Each entry is `|NF.toReal c| * eps + ulp (a * NF.toReal c) / 2`, the budget of `approx_scale`
for `xR * c` when `xR` is within `eps` of its spec and `c` counts as exact at its value
`NF.toReal c`.
-/
def scaleBoundTensor {s : Shape} (eps : ℝ) (c : R) (xR : Tensor R s) : Tensor ℝ s :=
  Tensor.map
    (fun a =>
      abs (NF.toReal c) * eps +
        ulp β fexp (a * NF.toReal c)
          / 2)
    (Tensor.toReal (α := R) (toSpec := NF.toReal) xR)

/-- Per-entry budget for scaling when the coefficient `c` is itself within `epsC` of its spec.

Each entry is `(|a| + eps) * epsC + (|NF.toReal c| + epsC) * eps + ulp (a * NF.toReal c) / 2`,
and `scaleBoundTensor` is the case `epsC = 0`. Attention needs the general form for its scale
`1 / sqrt(d)`, whose runtime value is already rounded when it is constructed.
-/
def scaleApproxBoundTensor {s : Shape} (eps epsC : ℝ) (c : R)
    (xR : Tensor R s) : Tensor ℝ s :=
  Tensor.map
    (fun a =>
      (|a| + eps) * epsC +
        (abs (NF.toReal c) + epsC) * eps +
        ulp β fexp
          (a * NF.toReal c) / 2)
    (Tensor.toReal (α := R) (toSpec := NF.toReal) xR)

/-- Per-entry negation budget `eps + ulp (-a) / 2` at the real value `a` of each entry. -/
def negBoundTensor {s : Shape} (eps : ℝ) (xR : Tensor R s) : Tensor ℝ s :=
  Tensor.map
    (fun a => eps + ulp β fexp (-a) / 2)
    (Tensor.toReal (α := R) (toSpec := NF.toReal) xR)

/-- Per-entry absolute-value budget `eps + ulp |a| / 2` at the real value `a` of each entry. -/
def absBoundTensor {s : Shape} (eps : ℝ) (xR : Tensor R s) : Tensor ℝ s :=
  Tensor.map
    (fun a => eps + ulp β fexp (abs a) / 2)
    (Tensor.toReal (α := R) (toSpec := NF.toReal) xR)

/--
Per-entry bound tensor for exponentiation (`exp`).

Each entry is `expErrorBound a eps`, the budget of `approx_exp`: the mean value bound on the
real `exp` plus half an ULP for the rounding.
-/
def expBoundTensor {s : Shape} (eps : ℝ) (xR : Tensor R s) : Tensor ℝ s :=
  Tensor.map (fun a => expErrorBound (β := β) (fexp := fexp) a eps)
    (Tensor.toReal (α := R) (toSpec := NF.toReal) xR)

/-- Per-entry budget `eps / √η + ulp (√(max a 0)) / 2` for the square root, valid when each exact
input is at least `η`. -/
def sqrtPosBoundTensor {s : Shape} (η eps : ℝ) (xR : Tensor R s) : Tensor ℝ s :=
  Tensor.map
    (fun a => eps / Real.sqrt η + ulp β fexp (Real.sqrt (max a 0)) / 2)
    (Tensor.toReal (α := R) (toSpec := NF.toReal) xR)

/-- Per-entry budget `√eps + ulp (√(max a 0)) / 2` of the clamped square root, which needs no
positive lower bound and so holds at zero too. -/
def sqrtBoundTensor {s : Shape} (eps : ℝ) (xR : Tensor R s) : Tensor ℝ s :=
  Tensor.map
    (fun a => Real.sqrt eps + ulp β fexp (Real.sqrt (max a 0)) / 2)
    (Tensor.toReal (α := R) (toSpec := NF.toReal) xR)

/--
Per-entry bound tensor for hyperbolic tangent (`tanh`) with input error `eps`.

Each entry is the bound `min eps 2 + ulp β fexp (Real.tanh a) / 2` of `approx_tanh`, with `a` the
real value of the matching entry of `xR`.
-/
def tanhBoundTensor {s : Shape} (eps : ℝ) (xR : Tensor R s) : Tensor ℝ s :=
  Tensor.map
    (fun a => min eps 2 + ulp β fexp (Real.tanh a) / 2)
    (Tensor.toReal (α := R) (toSpec := NF.toReal) xR)

-- Safe log (clamped) bound.

/--
Per-entry bound tensor for `safeLog`.

Each entry of `safeLogBoundTensor ε eps xR` is `(1/ε) * eps + ulp (safeLog ε a) / 2`, the Lipschitz
propagation term plus half an ULP for the rounding, and `ApproxTensor.safeLog` takes its
`linfNorm` as output epsilon.
-/
def safeLogBoundTensor {s : Shape} (ε eps : ℝ) (xR : Tensor R s) : Tensor ℝ s :=
  Tensor.map
    (fun a =>
      (1 / ε) * eps +
        ulp β fexp (safeLog (ε := ε) a) / 2)
    (Tensor.toReal (α := R) (toSpec := NF.toReal) xR)

/--
`ApproxTensor` bound for the clamped log `safeLog ε` at arbitrary tensor shapes, where `0 < ε`.

The scalar bound `approx_safeLog` lifts through `ApproxTensor.map`, and
the output epsilon is the `linfNorm` of `safeLogBoundTensor`.
-/
theorem _root_.TorchLean.ApproxTensor.safeLog {s : Shape} (ε : ℝ) (hε : 0 < ε) :
    ∀ {xS : Tensor ℝ s} {xR : Tensor R s} {eps : ℝ},
      ApproxTensor (α := R) (toSpec := NF.toReal) xS xR eps →
        ApproxTensor (α := R) (toSpec := NF.toReal)
          (Tensor.map (shape := s) (safeLog (ε := ε)) xS)
          (Tensor.map (shape := s) (safeLogR (β := β) (fexp := fexp) (rnd := rnd) ε) xR)
          (linfNorm (safeLogBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
            (s := s) ε eps xR)) := by
  intro xS xR eps hx
  have h :=
    ApproxTensor.map
      (α := R) (toSpec := NF.toReal)
      (s := s) (fS := NF.safeLog (ε := ε))
      (fR := safeLogR (β := β) (fexp := fexp) (rnd := rnd) ε)
      (bnd := fun a inputError =>
        (1 / ε) * inputError + ulp β fexp (NF.safeLog (ε := ε) a) / 2)
      (xS := xS) (xR := xR) (eps := eps) hx (by
        intro x xR hx
        exact approx_safeLog (β := β) (fexp := fexp) (rnd := rnd) hε hx)
  simpa [safeLogBoundTensor] using h

end

end NF
end TorchLean
