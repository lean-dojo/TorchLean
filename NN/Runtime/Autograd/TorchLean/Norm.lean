/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.TorchLean.Functional

import Mathlib.Algebra.Order.Algebra

/-!
# Norm

TorchLean normalization helpers.

These are backend-generic (eager tape + compiled GraphM), built out of the small `Ops` surface.

PyTorch analogy: these correspond to the functional/core math behind `InstanceNorm2d`, `GroupNorm`,
`BatchNorm2d`, and common transformer norms (RMSNorm-style), but exposed as plain functions so they
can run eagerly or be compiled into the verifier IR.

Conventions:
- CNN tensors use PyTorch's `N×C×H×W` via `NN.Tensor.Shape.NCHW`.
- We keep epsilon explicit where it matters for stability.

Note: we do **not** implement PyTorch-style running-stat updates as a backend-generic op.
You can still compute batch statistics (mean/var) and update running buffers in the imperative
session layer.
-/

@[expose] public section


namespace Runtime
namespace Autograd
namespace TorchLean

open Spec
open Tensor

namespace Norm

/-- Proof helper: if `a > 0` and `b > 0`, then `a * b > 0`. -/
def natMulPos (a b : Nat) (ha : a > 0) (hb : b > 0) : a * b > 0 :=
  Nat.mul_pos ha hb

/--
Size lemma for flattening spatial dimensions in `N×C×H×W`.

This is used to justify `reshape` from `N×C×H×W` to `N×C×(H*W)` (a common trick in normalization).
-/
def reshapeNCHWToNCHWFlatSize {n c h w : Nat} :
    Shape.size (NN.Tensor.Shape.NCHW n c h w) = Shape.size (.dim n (.dim c (.dim (h * w) .scalar)))
      := by
  simp [Shape.size]

/-- Inverse of `reshapeNCHW_to_NC_HW` (same equality, reversed). -/
def reshapeNCHWFlatToNCHWSize {n c h w : Nat} :
    Shape.size (.dim n (.dim c (.dim (h * w) .scalar))) = Shape.size (NN.Tensor.Shape.NCHW n c h w)
      := by
  simpa using (reshapeNCHWToNCHWFlatSize (n := n) (c := c) (h := h) (w := w)).symm

/-- Size lemma for flattening spatial dimensions in `C×H×W` to `C×(H*W)`. -/
def reshapeCHWToCHWFlatSize {c h w : Nat} :
    Shape.size (NN.Tensor.Shape.CHW c h w) = Shape.size (.dim c (.dim (h * w) .scalar)) := by
  simp [Shape.size]

/-- Inverse of `reshapeCHW_to_C_HW` (same equality, reversed). -/
def reshapeCHWFlatToCHWSize {c h w : Nat} :
    Shape.size (.dim c (.dim (h * w) .scalar)) = Shape.size (NN.Tensor.Shape.CHW c h w) := by
  simpa using (reshapeCHWToCHWFlatSize (c := c) (h := h) (w := w)).symm

/--
Broadcast proof used to apply per-channel parameters over `N×C×HW`.

This is the shape-level counterpart of broadcasting a `(C)` vector over `(N×C×HW)` in PyTorch.
-/
def broadcastVecToNCHW (n c hw : Nat) :
    Shape.CanBroadcastTo (.dim c .scalar) (.dim n (.dim c (.dim hw .scalar))) := by
  -- `c` matches, and the scalar inner broadcasts to `hw`; then add leading `n`.
  apply Shape.CanBroadcastTo.expand_dims
  apply Shape.CanBroadcastTo.dim_eq
  exact Shape.CanBroadcastTo.scalar_to_any (.dim hw .scalar)

/-- Broadcast proof: `(N×C)` can broadcast to `(N×C×HW)` by expanding the trailing dimension. -/
def broadcastNCToNCHW (n c hw : Nat) :
    Shape.CanBroadcastTo (.dim n (.dim c .scalar)) (.dim n (.dim c (.dim hw .scalar))) := by
  apply Shape.CanBroadcastTo.dim_eq
  apply Shape.CanBroadcastTo.dim_eq
  exact Shape.CanBroadcastTo.scalar_to_any (.dim hw .scalar)

/--
Broadcast proof: `(C)` broadcasts to `(N×C×HW)`.

This is just `broadcastVec_to_NC_HW` with a clearer name at use sites.
-/
def broadcastCToNCHW (n c hw : Nat) :
    Shape.CanBroadcastTo (.dim c .scalar) (.dim n (.dim c (.dim hw .scalar))) :=
  broadcastVecToNCHW (n := n) (c := c) (hw := hw)

/-- Broadcast proof: `(C)` broadcasts to `(C×HW)`. -/
def broadcastCToCHW (c hw : Nat) :
    Shape.CanBroadcastTo (.dim c .scalar) (.dim c (.dim hw .scalar)) := by
  apply Shape.CanBroadcastTo.dim_eq
  exact Shape.CanBroadcastTo.scalar_to_any (.dim hw .scalar)

/-- Broadcast proof: `(embedDim)` broadcasts to `(seqLen×embedDim)` (used by transformer-style
  norms). -/
def broadcastCToSeqEmbed (seqLen embedDim : Nat) :
    Shape.CanBroadcastTo (.dim embedDim .scalar) (.dim seqLen (.dim embedDim .scalar)) := by
  apply Shape.CanBroadcastTo.expand_dims
  apply Shape.CanBroadcastTo.dim_eq
  exact Shape.CanBroadcastTo.scalar_to_any .scalar

/-- RMSNorm on `seqLen×embedDim` (normalize across the last axis, scale by `gamma`). -/
def rmsNormLast {α : Type} [Context α] [DecidableEq Shape]
    {m : Type → Type} [Monad m] [Ops (m := m) (α := α)]
    {seqLen embedDim : Nat} (h_seq_pos : seqLen > 0) (h_embed_pos : embedDim > 0)
    (x : RefTy (m := m) (α := α) (.dim seqLen (.dim embedDim .scalar)))
    (gamma : RefTy (m := m) (α := α) (.dim embedDim .scalar))
    (ε : α := Numbers.epsilon) :
    m (RefTy (m := m) (α := α) (.dim seqLen (.dim embedDim .scalar))) := do
  let s : Shape := .dim seqLen (.dim embedDim .scalar)
  let _ : Shape.WellFormed s := ⟨⟨h_seq_pos, ⟨h_embed_pos, trivial⟩⟩⟩
  -- mean(x^2) over last axis
  let sq ← F.square (m := m) (α := α) (s := s) x
  let axis : Nat := Shape.rank s - 1
  have hrank : Shape.rank s > 0 := by simp [s, Shape.rank]
  let _ : Shape.valid_axis_inst axis s := Shape.validAxisLastAuto hrank
  let meanSq ← reduceMean (m := m) (α := α) (s := s) axis sq
  let meanSqShape : Shape := shapeAfterSum s axis
  let zero ← const (m := m) (α := α) (s := meanSqShape) (Spec.fill (0 : α) meanSqShape)
  let meanSqClamped ← max (m := m) (α := α) (s := meanSqShape) meanSq zero
  let epsT ← const (m := m) (α := α) (s := meanSqShape) (Spec.fill ε meanSqShape)
  let denom ← sqrt (m := m) (α := α) (s := meanSqShape) (← add (m := m) (α := α) (s := meanSqShape)
    meanSqClamped epsT)
  let invDenom ← inv (m := m) (α := α) (s := meanSqShape) denom
  let cbBack := shapeAfterSumBroadcastBack (s := s) axis (by infer_instance) (by infer_instance)
  let invDenomB ← broadcastTo (m := m) (α := α) (s₁ := meanSqShape) (s₂ := s) cbBack invDenom
  let normalized ← mul (m := m) (α := α) (s := s) x invDenomB
  let gammaB ← broadcastTo (m := m) (α := α) (s₁ := .dim embedDim .scalar) (s₂ := s)
    (broadcastCToSeqEmbed seqLen embedDim) gamma
  mul (m := m) (α := α) (s := s) normalized gammaB

/--
Batched RMSNorm on `batch×seqLen×embedDim` tensors.

This is implemented by mapping the single-sample RMSNorm (`rms_norm_last`) over the leading batch
axis, so it is total even for `batch = 0`.
-/
def rmsNormLastBatched {α : Type} [Context α] [DecidableEq Shape]
    {m : Type → Type} [Monad m] [Ops (m := m) (α := α)]
    {batch seqLen embedDim : Nat} (h_seq_pos : seqLen > 0) (h_embed_pos : embedDim > 0)
    (x : RefTy (m := m) (α := α) (.dim batch (.dim seqLen (.dim embedDim .scalar))))
    (gamma : RefTy (m := m) (α := α) (.dim embedDim .scalar))
    (ε : α := Numbers.epsilon) :
    m (RefTy (m := m) (α := α) (.dim batch (.dim seqLen (.dim embedDim .scalar)))) :=
  Private.mapBatch0 (m := m) (α := α)
    (batch := batch) (s := .dim seqLen (.dim embedDim .scalar))
    (t := .dim seqLen (.dim embedDim .scalar))
    x
    (fun x1 =>
      rmsNormLast (m := m) (α := α)
        (seqLen := seqLen) (embedDim := embedDim) h_seq_pos h_embed_pos x1 gamma (ε := ε))

/-- InstanceNorm2d on `N×C×H×W` (per-sample, per-channel stats over `H×W`, affine). -/
def instanceNorm2dNchw {α : Type} [Context α] [DecidableEq Shape]
    {m : Type → Type} [Monad m] [Ops (m := m) (α := α)]
    {n c h w : Nat}
    (h_n_pos : n > 0) (h_c_pos : c > 0) (h_h_pos : h > 0) (h_w_pos : w > 0)
    (x : RefTy (m := m) (α := α) (NN.Tensor.Shape.NCHW n c h w))
    (gamma beta : RefTy (m := m) (α := α) (.dim c .scalar))
    (ε : α := Numbers.epsilon) :
    m (RefTy (m := m) (α := α) (NN.Tensor.Shape.NCHW n c h w)) := do
  let hw : Nat := h * w
  let sNCHW : Shape := NN.Tensor.Shape.NCHW n c h w
  let sNCHWFlat : Shape := .dim n (.dim c (.dim hw .scalar))
  let _ : Shape.WellFormed sNCHW := ⟨⟨h_n_pos, ⟨h_c_pos, ⟨h_h_pos, ⟨h_w_pos, trivial⟩⟩⟩⟩⟩
  let _ : Shape.WellFormed sNCHWFlat :=
    ⟨⟨h_n_pos, ⟨h_c_pos, ⟨natMulPos h w h_h_pos h_w_pos, trivial⟩⟩⟩⟩
  let xFlat ← reshape (m := m) (α := α) (s₁ := sNCHW) (s₂ := sNCHWFlat) x (reshapeNCHWToNCHWFlatSize (n
    := n) (c := c) (h := h) (w := w))
  -- mean over last axis (HW)
  let axisHW : Nat := Shape.rank sNCHWFlat - 1
  have hrank : Shape.rank sNCHWFlat > 0 := by simp [sNCHWFlat, Shape.rank]
  let _ : Shape.valid_axis_inst axisHW sNCHWFlat := Shape.validAxisLastAuto hrank
  let mean ← reduceMean (m := m) (α := α) (s := sNCHWFlat) axisHW xFlat
  let meanShape : Shape := shapeAfterSum sNCHWFlat axisHW  -- `N×C`
  let cbBack := shapeAfterSumBroadcastBack (s := sNCHWFlat) axisHW (by infer_instance) (by
    infer_instance)
  let meanB ← broadcastTo (m := m) (α := α) (s₁ := meanShape) (s₂ := sNCHWFlat) cbBack mean
  let centered ← sub (m := m) (α := α) (s := sNCHWFlat) xFlat meanB
  let sq ← F.square (m := m) (α := α) (s := sNCHWFlat) centered
  let var ← reduceMean (m := m) (α := α) (s := sNCHWFlat) axisHW sq
  let zero ← const (m := m) (α := α) (s := meanShape) (Spec.fill (0 : α) meanShape)
  let varClamped ← max (m := m) (α := α) (s := meanShape) var zero
  let epsT ← const (m := m) (α := α) (s := meanShape) (Spec.fill ε meanShape)
  let denom ← sqrt (m := m) (α := α) (s := meanShape) (← add (m := m) (α := α) (s := meanShape)
    varClamped epsT)
  let invDenom ← inv (m := m) (α := α) (s := meanShape) denom
  let invDenomB ← broadcastTo (m := m) (α := α) (s₁ := meanShape) (s₂ := sNCHWFlat) cbBack invDenom
  let normalized ← mul (m := m) (α := α) (s := sNCHWFlat) centered invDenomB
  let gammaB ← broadcastTo (m := m) (α := α) (s₁ := .dim c .scalar) (s₂ := sNCHWFlat)
    (broadcastCToNCHW (n := n) (c := c) (hw := hw)) gamma
  let betaB ← broadcastTo (m := m) (α := α) (s₁ := .dim c .scalar) (s₂ := sNCHWFlat)
    (broadcastCToNCHW (n := n) (c := c) (hw := hw)) beta
  let yFlat ← add (m := m) (α := α) (s := sNCHWFlat)
    (← mul (m := m) (α := α) (s := sNCHWFlat) normalized gammaB) betaB
  reshape (m := m) (α := α) (s₁ := sNCHWFlat) (s₂ := sNCHW) yFlat (reshapeNCHWFlatToNCHWSize (n := n) (c
    := c) (h := h) (w := w))

/--
GroupNorm on `N×C×H×W` with `groups`, affine.

This is purely functional (no running stats); it matches PyTorch `GroupNorm`.
-/
def groupNorm2dNchw {α : Type} [Context α] [DecidableEq Shape]
    {m : Type → Type} [Monad m] [Ops (m := m) (α := α)]
    {n c h w groups : Nat}
    (h_n_pos : n > 0) (h_c_pos : c > 0) (h_h_pos : h > 0) (h_w_pos : w > 0) (h_g_pos : groups > 0)
    (h_ge : c ≥ groups) (h_div : c % groups = 0)
    (x : RefTy (m := m) (α := α) (NN.Tensor.Shape.NCHW n c h w))
    (gamma beta : RefTy (m := m) (α := α) (.dim c .scalar))
    (ε : α := Numbers.epsilon) :
    m (RefTy (m := m) (α := α) (NN.Tensor.Shape.NCHW n c h w)) := do
  let channelsPerGroup : Nat := c / groups
  let sNCHW : Shape := NN.Tensor.Shape.NCHW n c h w
  let sG : Shape := .dim n (.dim groups (.dim channelsPerGroup (.dim h (.dim w .scalar))))
  let chw : Nat := channelsPerGroup * h * w
  let sGF : Shape := .dim n (.dim groups (.dim chw .scalar))
  let _ : Shape.WellFormed sNCHW := ⟨⟨h_n_pos, ⟨h_c_pos, ⟨h_h_pos, ⟨h_w_pos, trivial⟩⟩⟩⟩⟩
  let _ : Shape.WellFormed sG := by
    refine ⟨⟨h_n_pos, ?_⟩⟩
    refine ⟨h_g_pos, ?_⟩
    have h_cpg_pos : channelsPerGroup > 0 := by
      unfold channelsPerGroup
      apply Nat.div_pos
      exact h_ge
      exact h_g_pos
    exact ⟨h_cpg_pos, ⟨h_h_pos, ⟨h_w_pos, trivial⟩⟩⟩
  have hSize : Shape.size sNCHW = Shape.size sG := by
    -- `c = groups * (c/groups)` since `c % groups = 0`
    have hc : c = groups * channelsPerGroup := by
      have := (Nat.mod_add_div c groups).symm
      -- `c = c % groups + groups * (c / groups)`; with `h_div`, reduces to `groups * (c/groups)`
      -- rewrite and simplify.
      simp [h_div] at this
      -- `this` is `c = groups * (c / groups)`, but in a slightly different form; normalize.
      simpa [channelsPerGroup, Nat.add_comm, Nat.add_left_comm, Nat.add_assoc, Nat.zero_add] using
        this
    -- now compare sizes
    simp [sNCHW, sG, Shape.size, hc, Nat.mul_left_comm, Nat.mul_comm]
  have h_chw_pos : chw > 0 := by
    -- channelsPerGroup, h, w are positive; multiplication preserves positivity.
    have h_cpg_pos : channelsPerGroup > 0 := by
      unfold channelsPerGroup
      apply Nat.div_pos
      exact h_ge
      exact h_g_pos
    have : channelsPerGroup * h > 0 := Nat.mul_pos h_cpg_pos h_h_pos
    exact Nat.mul_pos this h_w_pos
  let _ : Shape.WellFormed sGF := ⟨⟨h_n_pos, ⟨h_g_pos, ⟨h_chw_pos, trivial⟩⟩⟩⟩
  have hSizeG_F : Shape.size sG = Shape.size sGF := by
    simp [sG, sGF, Shape.size, chw, Nat.mul_left_comm, Nat.mul_comm]

  let xG ← reshape (m := m) (α := α) (s₁ := sNCHW) (s₂ := sG) x hSize
  let xGF ← reshape (m := m) (α := α) (s₁ := sG) (s₂ := sGF) xG hSizeG_F
  -- mean/var over last axis (`channelsPerGroup * h * w`)
  let axis : Nat := Shape.rank sGF - 1
  have hrank : Shape.rank sGF > 0 := by simp [sGF, Shape.rank]
  let _ : Shape.valid_axis_inst axis sGF := Shape.validAxisLastAuto hrank
  let mean ← reduceMean (m := m) (α := α) (s := sGF) axis xGF
  let meanShape : Shape := shapeAfterSum sGF axis -- `N×groups`
  let cbMean : Shape.CanBroadcastTo meanShape sGF := by
    apply Shape.CanBroadcastTo.dim_eq
    apply Shape.CanBroadcastTo.dim_eq
    exact Shape.CanBroadcastTo.scalar_to_any (.dim chw .scalar)
  let meanB ← broadcastTo (m := m) (α := α) (s₁ := meanShape) (s₂ := sGF) cbMean mean
  let centered ← sub (m := m) (α := α) (s := sGF) xGF meanB
  let sq ← F.square (m := m) (α := α) (s := sGF) centered
  let var ← reduceMean (m := m) (α := α) (s := sGF) axis sq
  let zeroVar ← const (m := m) (α := α) (s := meanShape) (Spec.fill (0 : α) meanShape)
  let varClamped ← max (m := m) (α := α) (s := meanShape) var zeroVar
  let epsT ← const (m := m) (α := α) (s := meanShape) (Spec.fill ε meanShape)
  let denom ← sqrt (m := m) (α := α) (s := meanShape)
    (← add (m := m) (α := α) (s := meanShape) varClamped epsT)
  let invDenom ← inv (m := m) (α := α) (s := meanShape) denom
  let invDenomB ← broadcastTo (m := m) (α := α) (s₁ := meanShape) (s₂ := sGF) cbMean invDenom
  let normalized ← mul (m := m) (α := α) (s := sGF) centered invDenomB
  let yG ← reshape (m := m) (α := α) (s₁ := sGF) (s₂ := sG) normalized hSizeG_F.symm
  let yNCHW ← reshape (m := m) (α := α) (s₁ := sG) (s₂ := sNCHW) yG hSize.symm
  -- Apply per-channel affine in `N×C×(H*W)` form.
  let hw : Nat := h * w
  let sNCHWFlat : Shape := .dim n (.dim c (.dim hw .scalar))
  let _ : Shape.WellFormed sNCHWFlat :=
    ⟨⟨h_n_pos, ⟨h_c_pos, ⟨natMulPos h w h_h_pos h_w_pos, trivial⟩⟩⟩⟩
  let yFlat ← reshape (m := m) (α := α) (s₁ := sNCHW) (s₂ := sNCHWFlat) yNCHW
    (reshapeNCHWToNCHWFlatSize (n := n) (c := c) (h := h) (w := w))
  let cbC : Shape.CanBroadcastTo (.dim c .scalar) sNCHWFlat :=
    broadcastCToNCHW (n := n) (c := c) (hw := hw)
  let gammaB ← broadcastTo (m := m) (α := α) (s₁ := .dim c .scalar) (s₂ := sNCHWFlat) cbC gamma
  let betaB ← broadcastTo (m := m) (α := α) (s₁ := .dim c .scalar) (s₂ := sNCHWFlat) cbC beta
  let yFlat' ← add (m := m) (α := α) (s := sNCHWFlat)
    (← mul (m := m) (α := α) (s := sNCHWFlat) yFlat gammaB) betaB
  reshape (m := m) (α := α) (s₁ := sNCHWFlat) (s₂ := sNCHW) yFlat'
    (reshapeNCHWFlatToNCHWSize (n := n) (c := c) (h := h) (w := w))

/-- BatchNorm2d (training) returning the batch mean/var (per-channel) as additional refs. -/
def batchNorm2dNchwTrainStats {α : Type} [Context α] [DecidableEq Shape]
    {m : Type → Type} [Monad m] [Ops (m := m) (α := α)]
    {n c h w : Nat}
    (h_n_pos : n > 0) (h_c_pos : c > 0) (h_h_pos : h > 0) (h_w_pos : w > 0)
    (x : RefTy (m := m) (α := α) (NN.Tensor.Shape.NCHW n c h w))
    (gamma beta : RefTy (m := m) (α := α) (.dim c .scalar))
    (ε : α := Numbers.epsilon) :
    m (RefTy (m := m) (α := α) (NN.Tensor.Shape.NCHW n c h w) ×
       RefTy (m := m) (α := α) (.dim c .scalar) ×
       RefTy (m := m) (α := α) (.dim c .scalar)) := do
  let hw : Nat := h * w
  let sNCHW : Shape := NN.Tensor.Shape.NCHW n c h w
  let sNCHWFlat : Shape := .dim n (.dim c (.dim hw .scalar))
  let _ : Shape.WellFormed sNCHW := ⟨⟨h_n_pos, ⟨h_c_pos, ⟨h_h_pos, ⟨h_w_pos, trivial⟩⟩⟩⟩⟩
  let _ : Shape.WellFormed sNCHWFlat :=
    ⟨⟨h_n_pos, ⟨h_c_pos, ⟨natMulPos h w h_h_pos h_w_pos, trivial⟩⟩⟩⟩
  let xFlat ← reshape (m := m) (α := α) (s₁ := sNCHW) (s₂ := sNCHWFlat) x (reshapeNCHWToNCHWFlatSize (n
    := n) (c := c) (h := h) (w := w))
  -- mean over HW then mean over batch
  let axisHW : Nat := Shape.rank sNCHWFlat - 1
  have hrank : Shape.rank sNCHWFlat > 0 := by simp [sNCHWFlat, Shape.rank]
  let _ : Shape.valid_axis_inst axisHW sNCHWFlat := Shape.validAxisLastAuto hrank
  let meanHW ← reduceMean (m := m) (α := α) (s := sNCHWFlat) axisHW xFlat -- `N×C`
  let sNC : Shape := shapeAfterSum sNCHWFlat axisHW
  have hsNC : sNC = .dim n (.dim c .scalar) := by
    simp [sNC, sNCHWFlat, axisHW, Shape.rank, shapeAfterSum]
  let _ : Shape.WellFormed sNC := by
    simpa [hsNC] using (show Shape.WellFormed (.dim n (.dim c .scalar)) from ⟨⟨h_n_pos, ⟨h_c_pos,
      trivial⟩⟩⟩)
  let axisN : Nat := 0
  let _ : Shape.valid_axis_inst axisN sNC := by
    simpa [hsNC] using (Shape.validAxisInstZeroAlt2 (n := n) (s := .dim c .scalar) h_n_pos)
  let mean ← reduceMean (m := m) (α := α) (s := sNC) axisN meanHW -- `C`
  -- broadcast mean back to `N×C×HW`
  let cbMean : Shape.CanBroadcastTo (.dim c .scalar) sNCHWFlat :=
    broadcastCToNCHW (n := n) (c := c) (hw := hw)
  let meanB ← broadcastTo (m := m) (α := α) (s₁ := .dim c .scalar) (s₂ := sNCHWFlat) cbMean mean
  let centered ← sub (m := m) (α := α) (s := sNCHWFlat) xFlat meanB
  let sq ← F.square (m := m) (α := α) (s := sNCHWFlat) centered
  let varHW ← reduceMean (m := m) (α := α) (s := sNCHWFlat) axisHW sq -- `N×C`
  let var ← reduceMean (m := m) (α := α) (s := sNC) axisN varHW -- `C`
  let zero ← const (m := m) (α := α) (s := .dim c .scalar) (Spec.fill (0 : α) (.dim c .scalar))
  let varClamped ← max (m := m) (α := α) (s := .dim c .scalar) var zero
  let epsT ← const (m := m) (α := α) (s := .dim c .scalar) (Spec.fill ε (.dim c .scalar))
  let denom ← sqrt (m := m) (α := α) (s := .dim c .scalar) (← add (m := m) (α := α) (s := .dim c
    .scalar) varClamped epsT)
  let invDenom ← inv (m := m) (α := α) (s := .dim c .scalar) denom
  let invDenomB ← broadcastTo (m := m) (α := α) (s₁ := .dim c .scalar) (s₂ := sNCHWFlat) cbMean
    invDenom
  let normalized ← mul (m := m) (α := α) (s := sNCHWFlat) centered invDenomB
  let gammaB ← broadcastTo (m := m) (α := α) (s₁ := .dim c .scalar) (s₂ := sNCHWFlat) cbMean gamma
  let betaB ← broadcastTo (m := m) (α := α) (s₁ := .dim c .scalar) (s₂ := sNCHWFlat) cbMean beta
  let yFlat ← add (m := m) (α := α) (s := sNCHWFlat) (← mul (m := m) (α := α) (s := sNCHWFlat)
    normalized gammaB) betaB
  let y ← reshape (m := m) (α := α) (s₁ := sNCHWFlat) (s₂ := sNCHW) yFlat (reshapeNCHWFlatToNCHWSize (n
    := n) (c := c) (h := h) (w := w))
  pure (y, mean, varClamped)

/--
BatchNorm2d (training-style) on `N×C×H×W` (stats over `N×H×W`, affine).

This does *not* update running statistics; use the returned batch stats (see
  `batch_norm2d_nchw_train_stats`)
to update buffers in an imperative training loop if desired.
-/
def batchNorm2dNchwTrain {α : Type} [Context α] [DecidableEq Shape]
    {m : Type → Type} [Monad m] [Ops (m := m) (α := α)]
    {n c h w : Nat}
    (h_n_pos : n > 0) (h_c_pos : c > 0) (h_h_pos : h > 0) (h_w_pos : w > 0)
    (x : RefTy (m := m) (α := α) (NN.Tensor.Shape.NCHW n c h w))
    (gamma beta : RefTy (m := m) (α := α) (.dim c .scalar))
    (ε : α := Numbers.epsilon) :
    m (RefTy (m := m) (α := α) (NN.Tensor.Shape.NCHW n c h w)) := do
  let (y, _mean, _var) ← batchNorm2dNchwTrainStats
    (α := α) (m := m) (n := n) (c := c) (h := h) (w := w)
    h_n_pos h_c_pos h_h_pos h_w_pos x gamma beta (ε := ε)
  pure y

/--
Update running BatchNorm statistics (EMA):

`running := (1 - momentum) * running + momentum * batch`.

This is a small helper to match the common PyTorch-style training loop.
It does not attempt to model PyTorch's biased/unbiased variance conventions; pass whatever
`batchVar` you intend to store (e.g. the same `var` you used for normalization, or an unbiased
adjusted variant computed externally).
-/
def batchNormRunningUpdate {α : Type} [Context α] [DecidableEq Shape]
    {m : Type → Type} [Monad m] [Ops (m := m) (α := α)]
    {c : Nat} (h_c_pos : c > 0)
    (runningMean runningVar batchMean batchVar : RefTy (m := m) (α := α) (.dim c .scalar))
    (momentum : α) :
    m (RefTy (m := m) (α := α) (.dim c .scalar) × RefTy (m := m) (α := α) (.dim c .scalar)) := do
  let sC : Shape := .dim c .scalar
  let _ : Shape.WellFormed sC := ⟨⟨h_c_pos, trivial⟩⟩
  let momT ← const (m := m) (α := α) (s := sC) (Spec.fill momentum sC)
  let oneMinusMomT ← const (m := m) (α := α) (s := sC) (Spec.fill ((1 : α) - momentum) sC)
  let mean' ← add (m := m) (α := α) (s := sC)
    (← mul (m := m) (α := α) (s := sC) runningMean oneMinusMomT)
    (← mul (m := m) (α := α) (s := sC) batchMean momT)
  let var' ← add (m := m) (α := α) (s := sC)
    (← mul (m := m) (α := α) (s := sC) runningVar oneMinusMomT)
    (← mul (m := m) (α := α) (s := sC) batchVar momT)
  pure (mean', var')

/-- BatchNorm2d evaluation: use provided per-channel mean/var (e.g. running stats). -/
def batchNorm2dNchwEval {α : Type} [Context α] [DecidableEq Shape]
    {m : Type → Type} [Monad m] [Ops (m := m) (α := α)]
    {n c h w : Nat}
    (h_n_pos : n > 0) (h_c_pos : c > 0) (h_h_pos : h > 0) (h_w_pos : w > 0)
    (x : RefTy (m := m) (α := α) (NN.Tensor.Shape.NCHW n c h w))
    (gamma beta mean var : RefTy (m := m) (α := α) (.dim c .scalar))
    (ε : α := Numbers.epsilon) :
    m (RefTy (m := m) (α := α) (NN.Tensor.Shape.NCHW n c h w)) := do
  let hw : Nat := h * w
  let sNCHW : Shape := NN.Tensor.Shape.NCHW n c h w
  let sNCHWFlat : Shape := .dim n (.dim c (.dim hw .scalar))
  let _ : Shape.WellFormed sNCHW := ⟨⟨h_n_pos, ⟨h_c_pos, ⟨h_h_pos, ⟨h_w_pos, trivial⟩⟩⟩⟩⟩
  let _ : Shape.WellFormed sNCHWFlat :=
    ⟨⟨h_n_pos, ⟨h_c_pos, ⟨natMulPos h w h_h_pos h_w_pos, trivial⟩⟩⟩⟩
  let xFlat ← reshape (m := m) (α := α) (s₁ := sNCHW) (s₂ := sNCHWFlat) x (reshapeNCHWToNCHWFlatSize (n
    := n) (c := c) (h := h) (w := w))
  let cbC : Shape.CanBroadcastTo (.dim c .scalar) sNCHWFlat :=
    broadcastCToNCHW (n := n) (c := c) (hw := hw)
  let meanB ← broadcastTo (m := m) (α := α) (s₁ := .dim c .scalar) (s₂ := sNCHWFlat) cbC mean
  let centered ← sub (m := m) (α := α) (s := sNCHWFlat) xFlat meanB
  let zero ← const (m := m) (α := α) (s := .dim c .scalar) (Spec.fill (0 : α) (.dim c .scalar))
  let varClamped ← max (m := m) (α := α) (s := .dim c .scalar) var zero
  let epsT ← const (m := m) (α := α) (s := .dim c .scalar) (Spec.fill ε (.dim c .scalar))
  let denom ← sqrt (m := m) (α := α) (s := .dim c .scalar) (← add (m := m) (α := α) (s := .dim c
    .scalar) varClamped epsT)
  let invDenom ← inv (m := m) (α := α) (s := .dim c .scalar) denom
  let invDenomB ← broadcastTo (m := m) (α := α) (s₁ := .dim c .scalar) (s₂ := sNCHWFlat) cbC
    invDenom
  let normalized ← mul (m := m) (α := α) (s := sNCHWFlat) centered invDenomB
  let gammaB ← broadcastTo (m := m) (α := α) (s₁ := .dim c .scalar) (s₂ := sNCHWFlat) cbC gamma
  let betaB ← broadcastTo (m := m) (α := α) (s₁ := .dim c .scalar) (s₂ := sNCHWFlat) cbC beta
  let yFlat ← add (m := m) (α := α) (s := sNCHWFlat) (← mul (m := m) (α := α) (s := sNCHWFlat)
    normalized gammaB) betaB
  reshape (m := m) (α := α) (s₁ := sNCHWFlat) (s₂ := sNCHW) yFlat (reshapeNCHWFlatToNCHWSize (n := n) (c
    := c) (h := h) (w := w))

/-- BatchNorm2d evaluation on `C×H×W`: use provided per-channel mean/var (e.g. running stats). -/
def batchNorm2dChwEval {α : Type} [Context α] [DecidableEq Shape]
    {m : Type → Type} [Monad m] [Ops (m := m) (α := α)]
    {c h w : Nat}
    (h_c_pos : c > 0) (h_h_pos : h > 0) (h_w_pos : w > 0)
    (x : RefTy (m := m) (α := α) (NN.Tensor.Shape.CHW c h w))
    (gamma beta mean var : RefTy (m := m) (α := α) (.dim c .scalar))
    (ε : α := Numbers.epsilon) :
    m (RefTy (m := m) (α := α) (NN.Tensor.Shape.CHW c h w)) := do
  let hw : Nat := h * w
  let sCHW : Shape := NN.Tensor.Shape.CHW c h w
  let sC_HW : Shape := .dim c (.dim hw .scalar)
  let _ : Shape.WellFormed sCHW := ⟨⟨h_c_pos, ⟨h_h_pos, ⟨h_w_pos, trivial⟩⟩⟩⟩
  let _ : Shape.WellFormed sC_HW := ⟨⟨h_c_pos, ⟨natMulPos h w h_h_pos h_w_pos, trivial⟩⟩⟩
  let xFlat ← reshape (m := m) (α := α) (s₁ := sCHW) (s₂ := sC_HW) x (reshapeCHWToCHWFlatSize (c := c) (h
    := h) (w := w))
  let cbC : Shape.CanBroadcastTo (.dim c .scalar) sC_HW :=
    broadcastCToCHW (c := c) (hw := hw)
  let meanB ← broadcastTo (m := m) (α := α) (s₁ := .dim c .scalar) (s₂ := sC_HW) cbC mean
  let centered ← sub (m := m) (α := α) (s := sC_HW) xFlat meanB
  let zero ← const (m := m) (α := α) (s := .dim c .scalar) (Spec.fill (0 : α) (.dim c .scalar))
  let varClamped ← max (m := m) (α := α) (s := .dim c .scalar) var zero
  let epsT ← const (m := m) (α := α) (s := .dim c .scalar) (Spec.fill ε (.dim c .scalar))
  let denom ← sqrt (m := m) (α := α) (s := .dim c .scalar)
    (← add (m := m) (α := α) (s := .dim c .scalar) varClamped epsT)
  let invDenom ← inv (m := m) (α := α) (s := .dim c .scalar) denom
  let invDenomB ← broadcastTo (m := m) (α := α) (s₁ := .dim c .scalar) (s₂ := sC_HW) cbC invDenom
  let normalized ← mul (m := m) (α := α) (s := sC_HW) centered invDenomB
  let gammaB ← broadcastTo (m := m) (α := α) (s₁ := .dim c .scalar) (s₂ := sC_HW) cbC gamma
  let betaB ← broadcastTo (m := m) (α := α) (s₁ := .dim c .scalar) (s₂ := sC_HW) cbC beta
  let yFlat ← add (m := m) (α := α) (s := sC_HW)
    (← mul (m := m) (α := α) (s := sC_HW) normalized gammaB) betaB
  reshape (m := m) (α := α) (s₁ := sC_HW) (s₂ := sCHW) yFlat (reshapeCHWFlatToCHWSize (c := c) (h := h)
    (w := w))

end Norm

end TorchLean
end Autograd
end Runtime
