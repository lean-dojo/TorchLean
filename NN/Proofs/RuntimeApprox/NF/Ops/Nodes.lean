/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Proofs.RuntimeApprox.NF.Ops.Elementwise.Binary
public import NN.Proofs.RuntimeApprox.NF.Ops.Elementwise.Softmax
public import NN.Proofs.RuntimeApprox.NF.Ops.Elementwise.Unary
public import NN.Proofs.RuntimeApprox.NF.Ops.Sum

/-!
# NF Forward Graph Nodes

`FwdNode` constructors for the NF backend.  These package the operation, runtime implementation,
bound computation, and soundness theorem so larger SSA/DAG graphs can compose the primitive proofs.
-/

@[expose] public section

namespace Proofs
namespace RuntimeApprox

open Spec TorchLean
open TorchLean TorchLean.Tensor
open NN.MLTheory.Robustness.Spec

noncomputable section

namespace NFBackend

open TorchLean.Floats
open Proofs.RuntimeRoundingApprox

variable {β : NeuralRadix} {fexp : ℤ → ℤ} [NeuralValidExp fexp]
variable {rnd : ℝ → ℤ} [NeuralValidRndToNearest rnd]

local notation "R" => TorchLean.Floats.NF β fexp rnd

-- ---------------------------------------------------------------------------
-- `FwdNode` constructors for `NF` ops (for building SSA/DAG forward bounds)
-- ---------------------------------------------------------------------------

/--
`FwdNode` for elementwise addition.

This packages `approxTensor_add_spec` so addition can be used inside larger verified `FwdGraph`s.
-/
def addNode {Γ : List Shape} {s : Shape} (a b : Idx Γ s) :
    FwdNode (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) Γ s :=
by
  classical
  refine
    { forwardSpec := fun ctx =>
        addSpec (getIdx (α := SpecScalar) ctx a) (getIdx (α := SpecScalar) ctx b)
    , forwardRuntime := fun ctx =>
        addSpec (getIdx (α := R) ctx a) (getIdx (α := R) ctx b)
    , bound := fun eps ctx =>
        linfNorm (addBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
          (s := s)
          (getIdxEps (Γ := Γ) (s := s) eps a)
          (getIdxEps (Γ := Γ) (s := s) eps b)
          (getIdx (α := R) ctx a) (getIdx (α := R) ctx b))
    , sound := ?_ }
  intro xS xR eps hctx
  have ha := approxCtx_getIdx (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) hctx
    a
  have hb := approxCtx_getIdx (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) hctx
    b
  simpa using
    (approxTensor_add_spec (β := β) (fexp := fexp) (rnd := rnd)
      (s := s) (xS := getIdx (α := SpecScalar) xS a) (yS := getIdx (α := SpecScalar) xS b)
      (xR := getIdx (α := R) xR a) (yR := getIdx (α := R) xR b)
      (epsx := getIdxEps (Γ := Γ) (s := s) eps a) (epsy := getIdxEps (Γ := Γ) (s := s) eps b)
      ha hb)

/-- `FwdNode` for elementwise subtraction (wraps `approxTensor_sub_spec`). -/
def subNode {Γ : List Shape} {s : Shape} (a b : Idx Γ s) :
    FwdNode (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) Γ s :=
by
  classical
  refine
    { forwardSpec := fun ctx =>
        subSpec (getIdx (α := SpecScalar) ctx a) (getIdx (α := SpecScalar) ctx b)
    , forwardRuntime := fun ctx =>
        subSpec (getIdx (α := R) ctx a) (getIdx (α := R) ctx b)
    , bound := fun eps ctx =>
        linfNorm (subBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
          (s := s)
          (getIdxEps (Γ := Γ) (s := s) eps a)
          (getIdxEps (Γ := Γ) (s := s) eps b)
          (getIdx (α := R) ctx a) (getIdx (α := R) ctx b))
    , sound := ?_ }
  intro xS xR eps hctx
  have ha := approxCtx_getIdx (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) hctx
    a
  have hb := approxCtx_getIdx (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) hctx
    b
  simpa using
    (approxTensor_sub_spec (β := β) (fexp := fexp) (rnd := rnd)
      (s := s) (xS := getIdx (α := SpecScalar) xS a) (yS := getIdx (α := SpecScalar) xS b)
      (xR := getIdx (α := R) xR a) (yR := getIdx (α := R) xR b)
      (epsx := getIdxEps (Γ := Γ) (s := s) eps a) (epsy := getIdxEps (Γ := Γ) (s := s) eps b)
      ha hb)

/-- `FwdNode` for elementwise multiplication (wraps `approxTensor_mul_spec`). -/
def mulNode {Γ : List Shape} {s : Shape} (a b : Idx Γ s) :
    FwdNode (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) Γ s :=
by
  classical
  refine
    { forwardSpec := fun ctx =>
        mulSpec (getIdx (α := SpecScalar) ctx a) (getIdx (α := SpecScalar) ctx b)
    , forwardRuntime := fun ctx =>
        mulSpec (getIdx (α := R) ctx a) (getIdx (α := R) ctx b)
    , bound := fun eps ctx =>
        linfNorm (mulBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
          (s := s)
          (getIdxEps (Γ := Γ) (s := s) eps a)
          (getIdxEps (Γ := Γ) (s := s) eps b)
          (getIdx (α := R) ctx a) (getIdx (α := R) ctx b))
    , sound := ?_ }
  intro xS xR eps hctx
  have ha := approxCtx_getIdx (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) hctx
    a
  have hb := approxCtx_getIdx (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) hctx
    b
  simpa using
    (approxTensor_mul_spec (β := β) (fexp := fexp) (rnd := rnd)
      (s := s) (xS := getIdx (α := SpecScalar) xS a) (yS := getIdx (α := SpecScalar) xS b)
      (xR := getIdx (α := R) xR a) (yR := getIdx (α := R) xR b)
      (epsx := getIdxEps (Γ := Γ) (s := s) eps a) (epsy := getIdxEps (Γ := Γ) (s := s) eps b)
      ha hb)

/--
`FwdNode` for clamped division `safeDiv`.

Requires a proof `hε : 0 < ε` and uses `approxTensor_safeDiv_spec` to obtain an unconditional bound.
-/
def safeDivNode {Γ : List Shape} {s : Shape} (a b : Idx Γ s) (ε : ℝ) (hε : 0 < ε) :
    FwdNode (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) Γ s :=
by
  classical
  refine
    { forwardSpec := fun ctx =>
        map2Spec (s := s) (safeDiv (ε := ε))
          (getIdx (α := SpecScalar) ctx a) (getIdx (α := SpecScalar) ctx b)
    , forwardRuntime := fun ctx =>
        map2Spec (s := s) (safeDivR (β := β) (fexp := fexp) (rnd := rnd) ε)
          (getIdx (α := R) ctx a) (getIdx (α := R) ctx b)
    , bound := fun eps ctx =>
        linfNorm (safeDivBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
          (s := s)
          ε
          (getIdxEps (Γ := Γ) (s := s) eps a)
          (getIdxEps (Γ := Γ) (s := s) eps b)
          (getIdx (α := R) ctx a) (getIdx (α := R) ctx b))
    , sound := ?_ }
  intro xS xR eps hctx
  have ha := approxCtx_getIdx (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) hctx
    a
  have hb := approxCtx_getIdx (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) hctx
    b
  simpa using
    (approxTensor_safeDiv_spec (β := β) (fexp := fexp) (rnd := rnd) (s := s) (ε := ε) hε
      (xS := getIdx (α := SpecScalar) xS a) (yS := getIdx (α := SpecScalar) xS b)
      (xR := getIdx (α := R) xR a) (yR := getIdx (α := R) xR b)
      (epsx := getIdxEps (Γ := Γ) (s := s) eps a) (epsy := getIdxEps (Γ := Γ) (s := s) eps b)
      ha hb)

/--
`FwdNode` for scaling by a runtime constant `c`.

Wraps `approxTensor_scale_spec`.
-/
def scaleNode {Γ : List Shape} {s : Shape} (a : Idx Γ s) (c : R) :
    FwdNode (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) Γ s :=
by
  classical
  refine
    { forwardSpec := fun ctx =>
        scaleSpec (α := SpecScalar) (s := s) (getIdx (α := SpecScalar) ctx a)
          (toSpec (β := β) (fexp := fexp) (rnd := rnd) c)
    , forwardRuntime := fun ctx =>
        scaleSpec (α := R) (s := s) (getIdx (α := R) ctx a) c
    , bound := fun eps ctx =>
        linfNorm (scaleBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
          (s := s) (getIdxEps (Γ := Γ) (s := s) eps a) c (getIdx (α := R) ctx a))
    , sound := ?_ }
  intro xS xR eps hctx
  have ha := approxCtx_getIdx (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) hctx
    a
  simpa using
    (approxTensor_scale_spec (β := β) (fexp := fexp) (rnd := rnd) (c := c)
      (s := s) (xS := getIdx (α := SpecScalar) xS a) (xR := getIdx (α := R) xR a)
      (eps := getIdxEps (Γ := Γ) (s := s) eps a) ha)

/-- `FwdNode` for elementwise negation (wraps `approxTensor_neg_spec`). -/
def negNode {Γ : List Shape} {s : Shape} (a : Idx Γ s) :
    FwdNode (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) Γ s :=
by
  classical
  refine
    { forwardSpec := fun ctx =>
        negSpec (getIdx (α := SpecScalar) ctx a)
    , forwardRuntime := fun ctx =>
        negSpec (getIdx (α := R) ctx a)
    , bound := fun eps ctx =>
        linfNorm (negBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
          (s := s) (getIdxEps (Γ := Γ) (s := s) eps a) (getIdx (α := R) ctx a))
    , sound := ?_ }
  intro xS xR eps hctx
  have ha := approxCtx_getIdx (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) hctx
    a
  simpa using
    (approxTensor_neg_spec (β := β) (fexp := fexp) (rnd := rnd)
      (s := s) (xS := getIdx (α := SpecScalar) xS a) (xR := getIdx (α := R) xR a)
      (eps := getIdxEps (Γ := Γ) (s := s) eps a) ha)

/-- `FwdNode` for elementwise absolute value (wraps `approxTensor_abs_spec`). -/
def absNode {Γ : List Shape} {s : Shape} (a : Idx Γ s) :
    FwdNode (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) Γ s :=
by
  classical
  refine
    { forwardSpec := fun ctx =>
        absSpec (getIdx (α := SpecScalar) ctx a)
    , forwardRuntime := fun ctx =>
        absSpec (getIdx (α := R) ctx a)
    , bound := fun eps ctx =>
        linfNorm (absBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
          (s := s) (getIdxEps (Γ := Γ) (s := s) eps a) (getIdx (α := R) ctx a))
    , sound := ?_ }
  intro xS xR eps hctx
  have ha := approxCtx_getIdx (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) hctx
    a
  simpa using
    (approxTensor_abs_spec (β := β) (fexp := fexp) (rnd := rnd)
      (s := s) (xS := getIdx (α := SpecScalar) xS a) (xR := getIdx (α := R) xR a)
      (eps := getIdxEps (Γ := Γ) (s := s) eps a) ha)

/-- `FwdNode` for elementwise exponentiation (wraps `approxTensor_exp_spec`). -/
def expNode {Γ : List Shape} {s : Shape} (a : Idx Γ s) :
    FwdNode (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) Γ s :=
by
  classical
  refine
    { forwardSpec := fun ctx =>
        expSpec (getIdx (α := SpecScalar) ctx a)
    , forwardRuntime := fun ctx =>
        expSpec (getIdx (α := R) ctx a)
    , bound := fun eps ctx =>
        linfNorm (expBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
          (s := s) (getIdxEps (Γ := Γ) (s := s) eps a) (getIdx (α := R) ctx a))
    , sound := ?_ }
  intro xS xR eps hctx
  have ha := approxCtx_getIdx (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) hctx
    a
  simpa using
    (approxTensor_exp_spec (β := β) (fexp := fexp) (rnd := rnd)
      (s := s) (xS := getIdx (α := SpecScalar) xS a) (xR := getIdx (α := R) xR a)
      (eps := getIdxEps (Γ := Γ) (s := s) eps a) ha)

/-- `FwdNode` for elementwise softplus (wraps `approxTensor_softplus_spec`). -/
def softplusNode {Γ : List Shape} {s : Shape} (a : Idx Γ s) :
    FwdNode (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) Γ s :=
by
  classical
  refine
    { forwardSpec := fun ctx =>
        mapSpec (s := s) (Activation.Math.softplusSpec (α := ℝ)) (getIdx (α := SpecScalar) ctx a)
    , forwardRuntime := fun ctx =>
        mapSpec (s := s) (softplusR (β := β) (fexp := fexp) (rnd := rnd)) (getIdx (α := R) ctx a)
    , bound := fun eps ctx =>
        linfNorm (softplusBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
          (s := s) (getIdxEps (Γ := Γ) (s := s) eps a) (getIdx (α := R) ctx a))
    , sound := ?_ }
  intro xS xR eps hctx
  have ha := approxCtx_getIdx (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) hctx
    a
  simpa using
    (approxTensor_softplus_spec (β := β) (fexp := fexp) (rnd := rnd)
      (s := s) (xS := getIdx (α := SpecScalar) xS a) (xR := getIdx (α := R) xR a)
      (eps := getIdxEps (Γ := Γ) (s := s) eps a) ha)

/--
`FwdNode` for clamped log `safeLog`.

Requires a proof `hε : 0 < ε` and wraps `approxTensor_safeLog_spec`.
-/
def safeLogNode {Γ : List Shape} {s : Shape} (a : Idx Γ s) (ε : ℝ) (hε : 0 < ε) :
    FwdNode (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) Γ s :=
by
  classical
  refine
    { forwardSpec := fun ctx =>
        mapSpec (s := s) (safeLog (ε := ε)) (getIdx (α := SpecScalar) ctx a)
    , forwardRuntime := fun ctx =>
        mapSpec (s := s) (safeLogR (β := β) (fexp := fexp) (rnd := rnd) ε) (getIdx (α := R) ctx a)
    , bound := fun eps ctx =>
        linfNorm (safeLogBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
          (s := s) ε (getIdxEps (Γ := Γ) (s := s) eps a) (getIdx (α := R) ctx a))
    , sound := ?_ }
  intro xS xR eps hctx
  have ha := approxCtx_getIdx (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) hctx
    a
  simpa using
    (approxTensor_safeLog_spec (β := β) (fexp := fexp) (rnd := rnd) (s := s) (ε := ε) hε
      (xS := getIdx (α := SpecScalar) xS a) (xR := getIdx (α := R) xR a)
      (eps := getIdxEps (Γ := Γ) (s := s) eps a) ha)

/--
`FwdNode` for the smooth `safeLog` activation.

Requires `hε : 0 < ε` and wraps `approxTensor_safe_log_spec`.
-/
def safeLogSoftplusNode {Γ : List Shape} {s : Shape} (a : Idx Γ s) (ε : ℝ) (hε : 0 < ε) :
    FwdNode (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) Γ s :=
by
  classical
  refine
    { forwardSpec := fun ctx =>
        mapSpec (s := s) (fun x => Activation.Math.safeLogSpec (α := ℝ) x ε) (getIdx (α :=
          SpecScalar) ctx a)
    , forwardRuntime := fun ctx =>
        mapSpec (s := s) (safeLogSoftplusR (β := β) (fexp := fexp) (rnd := rnd) ε)
          (getIdx (α := R) ctx a)
    , bound := fun eps ctx =>
        linfNorm (safeLogSoftplusBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
          (s := s) ε (getIdxEps (Γ := Γ) (s := s) eps a) (getIdx (α := R) ctx a))
    , sound := ?_ }
  intro xS xR eps hctx
  have ha := approxCtx_getIdx (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) hctx
    a
  simpa using
    (approxTensor_safe_log_spec (β := β) (fexp := fexp) (rnd := rnd) (s := s) (ε := ε) hε
      (xS := getIdx (α := SpecScalar) xS a) (xR := getIdx (α := R) xR a)
      (eps := getIdxEps (Γ := Γ) (s := s) eps a) ha)

/-- `FwdNode` for elementwise `tanh` (wraps `approxTensor_tanh_spec`). -/
def tanhNode {Γ : List Shape} {s : Shape} (a : Idx Γ s) :
    FwdNode (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) Γ s :=
by
  classical
  refine
    { forwardSpec := fun ctx =>
        mapSpec (s := s) MathFunctions.tanh (getIdx (α := SpecScalar) ctx a)
    , forwardRuntime := fun ctx =>
        mapSpec (s := s) MathFunctions.tanh (getIdx (α := R) ctx a)
    , bound := fun eps ctx =>
        linfNorm (tanhBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
          (s := s) (getIdxEps (Γ := Γ) (s := s) eps a) (getIdx (α := R) ctx a))
    , sound := ?_ }
  intro xS xR eps hctx
  have ha := approxCtx_getIdx (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) hctx
    a
  simpa using
    (approxTensor_tanh_spec (β := β) (fexp := fexp) (rnd := rnd)
      (s := s) (xS := getIdx (α := SpecScalar) xS a) (xR := getIdx (α := R) xR a)
      (eps := getIdxEps (Γ := Γ) (s := s) eps a) ha)

/-- `FwdNode` for elementwise sigmoid (wraps `approxTensor_sigmoid_spec`). -/
def sigmoidNode {Γ : List Shape} {s : Shape} (a : Idx Γ s) :
    FwdNode (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) Γ s :=
by
  classical
  refine
    { forwardSpec := fun ctx =>
        mapSpec (s := s) (Activation.Math.sigmoidSpec (α := ℝ)) (getIdx (α := SpecScalar) ctx a)
    , forwardRuntime := fun ctx =>
        mapSpec (s := s) (Activation.Math.sigmoidSpec (α := R)) (getIdx (α := R) ctx a)
    , bound := fun eps ctx =>
        linfNorm (sigmoidBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
          (s := s) (getIdxEps (Γ := Γ) (s := s) eps a) (getIdx (α := R) ctx a))
    , sound := ?_ }
  intro xS xR eps hctx
  have ha := approxCtx_getIdx (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) hctx
    a
  simpa using
    (approxTensor_sigmoid_spec (β := β) (fexp := fexp) (rnd := rnd) (s := s)
      (xS := getIdx (α := SpecScalar) xS a) (xR := getIdx (α := R) xR a)
      (eps := getIdxEps (Γ := Γ) (s := s) eps a) ha)

/-- `FwdNode` for elementwise ReLU (`max · 0`, wraps `approxTensor_relu_spec`). -/
def reluNode {Γ : List Shape} {s : Shape} (a : Idx Γ s) :
    FwdNode (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) Γ s :=
by
  classical
  refine
    { forwardSpec := fun ctx =>
        mapSpec (fun x => max x 0) (getIdx (α := SpecScalar) ctx a)
    , forwardRuntime := fun ctx =>
        mapSpec (reluR (β := β) (fexp := fexp) (rnd := rnd)) (getIdx (α := R) ctx a)
    , bound := fun eps ctx =>
        linfNorm (reluBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
          (s := s) (getIdxEps (Γ := Γ) (s := s) eps a) (getIdx (α := R) ctx a))
    , sound := ?_ }
  intro xS xR eps hctx
  have ha := approxCtx_getIdx (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) hctx
    a
  simpa using
    (approxTensor_relu_spec (β := β) (fexp := fexp) (rnd := rnd)
      (s := s) (xS := getIdx (α := SpecScalar) xS a) (xR := getIdx (α := R) xR a)
      (eps := getIdxEps (Γ := Γ) (s := s) eps a) ha)

/-- `FwdNode` for the scalar logistic-form `softmax` node (wraps `approxTensor_softmax_spec`). -/
def softmaxNode {Γ : List Shape} {s : Shape} (a : Idx Γ s) :
    FwdNode (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) Γ s :=
by
  classical
  refine
    { forwardSpec := fun ctx =>
        mapSpec (s := s) (Activation.Math.logisticSpec (α := ℝ)) (getIdx (α := SpecScalar) ctx a)
    , forwardRuntime := fun ctx =>
        mapSpec (s := s) (Activation.Math.logisticSpec (α := R)) (getIdx (α := R) ctx a)
    , bound := fun eps ctx =>
        linfNorm (softmaxBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
          (s := s) (getIdxEps (Γ := Γ) (s := s) eps a) (getIdx (α := R) ctx a))
    , sound := ?_ }
  intro xS xR eps hctx
  have ha := approxCtx_getIdx (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) hctx
    a
  simpa using
    (approxTensor_softmax_spec (β := β) (fexp := fexp) (rnd := rnd) (s := s)
      (xS := getIdx (α := SpecScalar) xS a) (xR := getIdx (α := R) xR a)
      (eps := getIdxEps (Γ := Γ) (s := s) eps a) ha)

/--
`FwdNode` for sum reduction (`sumSpec`).

This reduces a tensor to a scalar and uses `approxTensor_sum_spec` with the accumulated `sumBound`.
-/
def sumNode {Γ : List Shape} {s : Shape} (a : Idx Γ s) :
    FwdNode (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) Γ Shape.scalar :=
by
  classical
  refine
    { forwardSpec := fun ctx =>
        Tensor.scalar (sumSpec (α := ℝ) (s := s) (getIdx (α := SpecScalar) ctx a))
    , forwardRuntime := fun ctx =>
        Tensor.scalar (sumSpec (α := R) (s := s) (getIdx (α := R) ctx a))
    , bound := fun eps ctx =>
        sumBound (β := β) (fexp := fexp) (rnd := rnd) (s := s)
          (getIdxEps (Γ := Γ) (s := s) eps a) (getIdx (α := R) ctx a)
    , sound := ?_ }
  intro xS xR eps hctx
  have ha := approxCtx_getIdx (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) hctx
    a
  simpa using
    (approxTensor_sum_spec (β := β) (fexp := fexp) (rnd := rnd) (s := s)
      (xS := getIdx (α := SpecScalar) xS a) (xR := getIdx (α := R) xR a)
      (eps := getIdxEps (Γ := Γ) (s := s) eps a) ha)


end NFBackend

end

end RuntimeApprox
end Proofs
