/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Verify.RuntimeApprox.NF.Elementwise.Binary
public import TorchLean.Verify.RuntimeApprox.NF.Elementwise.SafeDivSigmoid
public import TorchLean.Verify.RuntimeApprox.NF.Elementwise.Unary
public import TorchLean.Verify.RuntimeApprox.NF.Sum

/-!
# NF Forward Graph Nodes

`FwdNode` constructors for the rounded-real NF model, each packaging one operation with its error
bound for graph composition. The node functions are noncomputable models, and each soundness proof
relates a node's exact function to its rounded one. LibTorch execution needs a separate guarantee.
-/

@[expose] public section

namespace TorchLean
namespace NF

open TorchLean.Tensor
open Robustness

noncomputable section

section

open FloatLib FloatLib.Numerics FloatLib.Floats.Formats
open Flocq
open TensorPack (getIdx)

variable {β : Radix} {fexp : ℤ → ℤ} [ValidExp fexp]
variable {rnd : ℝ → ℤ} [ValidRndToNearest rnd]

local notation "R" => NF β fexp rnd

-- ---------------------------------------------------------------------------
-- `FwdNode` constructors for `NF` ops (for building SSA/DAG forward bounds)
-- ---------------------------------------------------------------------------

/-- `FwdNode` for elementwise multiplication (wraps `ApproxTensor.mul`). -/
def mulNode {Γ : List Shape} {s : Shape} (a b : Idx Γ s) :
    FwdNode (α := R) (toSpec := NF.toReal) Γ s := by
  refine
    { forwardReal := fun ctx =>
        getIdx (α := ℝ) ctx a * getIdx (α := ℝ) ctx b
    , forwardRuntime := fun ctx =>
        getIdx (α := R) ctx a * getIdx (α := R) ctx b
    , bound := fun eps ctx =>
        linfNorm (mulBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
          (s := s)
          (EList.getIdx (Γ := Γ) (s := s) eps a)
          (EList.getIdx (Γ := Γ) (s := s) eps b)
          (getIdx (α := R) ctx a) (getIdx (α := R) ctx b))
    , sound := ?_ }
  intro xS xR eps hctx
  have ha := ApproxCtx.getIdx (α := R) (toSpec := NF.toReal) hctx
    a
  have hb := ApproxCtx.getIdx (α := R) (toSpec := NF.toReal) hctx
    b
  simpa using
    (ApproxTensor.mul (β := β) (fexp := fexp) (rnd := rnd)
      (s := s) (xS := getIdx (α := ℝ) xS a) (yS := getIdx (α := ℝ) xS b)
      (xR := getIdx (α := R) xR a) (yR := getIdx (α := R) xR b)
      (epsx := EList.getIdx (Γ := Γ) (s := s) eps a) (epsy := EList.getIdx (Γ := Γ) (s := s) eps b)
      ha hb)

/--
`FwdNode` for clamped division `safeDiv`.

Requires a proof `hε : 0 < ε` and uses `ApproxTensor.safeDiv` to obtain an unconditional bound.
-/
def safeDivNode {Γ : List Shape} {s : Shape} (a b : Idx Γ s) (ε : ℝ) (hε : 0 < ε) :
    FwdNode (α := R) (toSpec := NF.toReal) Γ s := by
  refine
    { forwardReal := fun ctx =>
        zipWith (s := s) (safeDiv (ε := ε))
          (getIdx (α := ℝ) ctx a) (getIdx (α := ℝ) ctx b)
    , forwardRuntime := fun ctx =>
        zipWith (s := s) (safeDivR (β := β) (fexp := fexp) (rnd := rnd) ε)
          (getIdx (α := R) ctx a) (getIdx (α := R) ctx b)
    , bound := fun eps ctx =>
        linfNorm (safeDivBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
          (s := s)
          ε
          (EList.getIdx (Γ := Γ) (s := s) eps a)
          (EList.getIdx (Γ := Γ) (s := s) eps b)
          (getIdx (α := R) ctx a) (getIdx (α := R) ctx b))
    , sound := ?_ }
  intro xS xR eps hctx
  have ha := ApproxCtx.getIdx (α := R) (toSpec := NF.toReal) hctx
    a
  have hb := ApproxCtx.getIdx (α := R) (toSpec := NF.toReal) hctx
    b
  simpa using
    (ApproxTensor.safeDiv (β := β) (fexp := fexp) (rnd := rnd) (s := s) (ε := ε) hε
      (xS := getIdx (α := ℝ) xS a) (yS := getIdx (α := ℝ) xS b)
      (xR := getIdx (α := R) xR a) (yR := getIdx (α := R) xR b)
      (epsx := EList.getIdx (Γ := Γ) (s := s) eps a) (epsy := EList.getIdx (Γ := Γ) (s := s) eps b)
      ha hb)

/-- `FwdNode` for elementwise exponentiation (wraps `ApproxTensor.exp`). -/
def expNode {Γ : List Shape} {s : Shape} (a : Idx Γ s) :
    FwdNode (α := R) (toSpec := NF.toReal) Γ s := by
  refine
    { forwardReal := fun ctx =>
        Tensor.exp (getIdx (α := ℝ) ctx a)
    , forwardRuntime := fun ctx =>
        Tensor.exp (getIdx (α := R) ctx a)
    , bound := fun eps ctx =>
        linfNorm (expBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
          (s := s) (EList.getIdx (Γ := Γ) (s := s) eps a) (getIdx (α := R) ctx a))
    , sound := ?_ }
  intro xS xR eps hctx
  have ha := ApproxCtx.getIdx (α := R) (toSpec := NF.toReal) hctx
    a
  simpa using
    (ApproxTensor.exp (β := β) (fexp := fexp) (rnd := rnd)
      (s := s) (xS := getIdx (α := ℝ) xS a) (xR := getIdx (α := R) xR a)
      (eps := EList.getIdx (Γ := Γ) (s := s) eps a) ha)

/-- `FwdNode` for elementwise ReLU (`max · 0`, wraps `ApproxTensor.relu`). -/
def reluNode {Γ : List Shape} {s : Shape} (a : Idx Γ s) :
    FwdNode (α := R) (toSpec := NF.toReal) Γ s := by
  refine
    { forwardReal := fun ctx =>
        Tensor.map (fun x => max x 0) (getIdx (α := ℝ) ctx a)
    , forwardRuntime := fun ctx =>
        Tensor.map (reluR (β := β) (fexp := fexp) (rnd := rnd)) (getIdx (α := R) ctx a)
    , bound := fun eps ctx =>
        linfNorm (reluBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
          (s := s) (EList.getIdx (Γ := Γ) (s := s) eps a) (getIdx (α := R) ctx a))
    , sound := ?_ }
  intro xS xR eps hctx
  have ha := ApproxCtx.getIdx (α := R) (toSpec := NF.toReal) hctx
    a
  simpa using
    (ApproxTensor.relu (β := β) (fexp := fexp) (rnd := rnd)
      (s := s) (xS := getIdx (α := ℝ) xS a) (xR := getIdx (α := R) xR a)
      (eps := EList.getIdx (Γ := Γ) (s := s) eps a) ha)

/-- `FwdNode` for elementwise `Context.sigmoid` (wraps `ApproxTensor.sigmoid`). -/
def sigmoidNode {Γ : List Shape} {s : Shape} (a : Idx Γ s) :
    FwdNode (α := R) (toSpec := NF.toReal) Γ s := by
  refine
    { forwardReal := fun ctx =>
        Tensor.map (Context.sigmoid (α := ℝ)) (getIdx (α := ℝ) ctx a)
    , forwardRuntime := fun ctx =>
        Tensor.map (shape := s) (Context.sigmoid (α := R)) (getIdx (α := R) ctx a)
    , bound := fun eps ctx =>
        linfNorm (sigmoidBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
          (s := s) (EList.getIdx (Γ := Γ) (s := s) eps a) (getIdx (α := R) ctx a))
    , sound := ?_ }
  intro xS xR eps hctx
  have ha := ApproxCtx.getIdx (α := R) (toSpec := NF.toReal) hctx
    a
  simpa using
    (ApproxTensor.sigmoid (β := β) (fexp := fexp) (rnd := rnd) (s := s)
      (xS := getIdx (α := ℝ) xS a) (xR := getIdx (α := R) xR a)
      (eps := EList.getIdx (Γ := Γ) (s := s) eps a) ha)

/--
`FwdNode` for sum reduction (`Tensor.sum`).

The node reduces a tensor to a scalar, and `ApproxTensor.sum` supplies its accumulated budget
`sumBound`.
-/
def sumNode {Γ : List Shape} {s : Shape} (a : Idx Γ s) :
    FwdNode (α := R) (toSpec := NF.toReal) Γ Shape.scalar := by
  refine
    { forwardReal := fun ctx =>
        Tensor.scalar (Tensor.sum (α := ℝ) (s := s) (getIdx (α := ℝ) ctx a))
    , forwardRuntime := fun ctx =>
        Tensor.scalar (Tensor.sum (α := R) (s := s) (getIdx (α := R) ctx a))
    , bound := fun eps ctx =>
        sumBound (β := β) (fexp := fexp) (rnd := rnd) (s := s)
          (EList.getIdx (Γ := Γ) (s := s) eps a) (getIdx (α := R) ctx a)
    , sound := ?_ }
  intro xS xR eps hctx
  have ha := ApproxCtx.getIdx (α := R) (toSpec := NF.toReal) hctx
    a
  simpa using
    (ApproxTensor.sum (β := β) (fexp := fexp) (rnd := rnd) (s := s)
      (xS := getIdx (α := ℝ) xS a) (xR := getIdx (α := R) xR a)
      (eps := EList.getIdx (Γ := Γ) (s := s) eps a) ha)

end

end

end NF
end TorchLean
