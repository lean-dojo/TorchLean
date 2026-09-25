/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.MLTheory.CROWN.Proofs.DirectedIBPNormalizationTensor
public import NN.MLTheory.CROWN.Proofs.LayerNormEnclosure
import all NN.MLTheory.CROWN.Graph.Engine.Base

/-!
# Rounded evaluation-mode BatchNorm

The real operation broadcasts stored channel statistics across the spatial shape and applies
`Spec.batchNormInference` independently to every leading sample. The flat-index lemmas below
connect that operation to the channel selected by the executable transfer.
-/

@[expose] public section

namespace NN.MLTheory.CROWN.Graph.DirectedBackward

open Spec TorchLean TorchLean.Tensor NN.IR
open NN.MLTheory.CROWN.Graph.Internal
open _root_.Proofs.Autograd.Norm
open BoundOps

noncomputable section

/-- The channel coordinate after an arbitrary leading batch shape. -/
def normalizationChannel (leading : Shape) {channels : Nat} {spatial : Shape} :
    (leading.concat (.dim channels spatial)).Coord → Fin channels :=
  match leading with
  | .scalar => fun c => c.1
  | .dim _ rest => fun c => normalizationChannel rest c.2

private theorem normalization_linearize_cons {n : Nat} {s : Shape}
    (i : Fin n) (c : s.Coord) :
    (Shape.Coord.linearize (shape := .dim n s) (i, c)).val =
      (Shape.Coord.linearize c).val + s.size * i.val := by
  simpa only [Shape.Coord.linearize, Shape.internalSize_eq] using
    TorchLean.Tensor.Internal.Coord.linearize_cons_val i c

private theorem normalization_channel_stride (leading spatial : Shape) (channels : Nat) :
    ((leading.concat (.dim channels spatial)).toList.drop (leading.rank + 1)).prod =
      spatial.size := by
  induction leading with
  | scalar => simp [Shape.concat, Shape.rank, Shape.toList, Shape.size_eq_prod]
  | dim n leading ih =>
      simpa only [Shape.concat, Shape.rank, Shape.toList, List.drop_succ_cons] using ih

private theorem normalization_channel_div_mod (leading : Shape) {channels : Nat}
    {spatial : Shape} (hspatial : 0 < spatial.size)
    (c : (leading.concat (.dim channels spatial)).Coord) :
    (Shape.Coord.linearize c).val / spatial.size % channels =
      (normalizationChannel leading c).val := by
  induction leading with
  | scalar =>
      change (Shape.Coord.linearize (shape := .dim channels spatial) (c.1, c.2)).val /
        spatial.size % channels = c.1.val
      rw [normalization_linearize_cons, Nat.add_mul_div_left _ _ hspatial,
        Nat.div_eq_of_lt (Shape.Coord.linearize c.2).isLt, Nat.zero_add,
        Nat.mod_eq_of_lt c.1.isLt]
  | dim n leading ih =>
      change (Shape.Coord.linearize
        (shape := .dim n (leading.concat (.dim channels spatial))) (c.1, c.2)).val /
          spatial.size % channels =
        (normalizationChannel leading c.2).val
      rw [normalization_linearize_cons]
      have hsize :
          (leading.concat (.dim channels spatial)).size * c.1.val =
            spatial.size * (channels * (leading.size * c.1.val)) := by
        simp only [Shape.size_concat, Shape.size]
        ring
      rw [hsize, Nat.add_mul_div_left _ _ hspatial, Nat.add_mul_mod_self_left]
      exact ih c.2

/-- The executable division/modulo channel index is the actual tensor channel coordinate. -/
theorem axisCoordinateOfFlat_normalizationChannel (leading spatial : Shape) (channels : Nat)
    (c : (leading.concat (.dim channels spatial)).Coord) :
    axisCoordinateOfFlat (leading.concat (.dim channels spatial)) leading.rank
        (Shape.Coord.linearize c).val % channels =
      (normalizationChannel leading c).val := by
  have hsize : 0 < (leading.concat (.dim channels spatial)).size :=
    Nat.zero_lt_of_lt (Shape.Coord.linearize c).isLt
  have hspatial : 0 < spatial.size := by
    by_contra h
    have hz : spatial.size = 0 := by omega
    simp [Shape.size_concat, Shape.size, hz] at hsize
  simp only [axisCoordinateOfFlat, normalization_channel_stride, hspatial.ne', ↓reduceIte]
  exact normalization_channel_div_mod leading hspatial c

/-- A present channel axis decomposes its shape into leading, channel, and spatial factors. -/
theorem normalization_shape_decomposition {s : Shape} {axis channels : Nat}
    (h : s.toList[axis]? = some channels) :
    ∃ leading spatial : Shape, axis = leading.rank ∧
      s = leading.concat (.dim channels spatial) := by
  induction s generalizing axis with
  | scalar => simp [Shape.toList] at h
  | dim n s ih =>
      cases axis with
      | zero =>
          have hn : n = channels := by simpa [Shape.toList] using h
          subst n
          exact ⟨.scalar, s, rfl, rfl⟩
      | succ axis =>
          have ht : s.toList[axis]? = some channels := by
            simpa only [Shape.toList, List.getElem?_cons_succ] using h
          obtain ⟨leading, spatial, haxis, hshape⟩ := ih ht
          exact ⟨.dim n leading, spatial, by simp [Shape.rank, haxis], by
            simp only [Shape.concat, hshape]⟩

private theorem normalization_broadcastChannel_apply {channels : Nat} {spatial : Shape}
    (v : Tensor ℝ [channels]) (i : Fin channels) (c : spatial.Coord) :
    Spec.broadcastChannel spatial v (i, c) = v.getScalar i := by
  have hspatial : 0 < spatial.size := Nat.zero_lt_of_lt (Shape.Coord.linearize c).isLt
  unfold Spec.broadcastChannel Tensor.reshapeSpec
  rw [TorchLean.Tensor.Internal.Rep.reshape_apply_coordEquiv]
  let hsize : Shape.size [channels, spatial.size] =
      Shape.size (.dim channels spatial) := by simp [Shape.size]
  let rc : Shape.Coord [channels, spatial.size] :=
    TorchLean.Tensor.Internal.Rep.reshapeCoordEquiv
    (Shape.internalSize_congr hsize) (i, c)
  have hlinear : (Shape.Coord.linearize rc).val =
      (Shape.Coord.linearize (shape := .dim channels spatial) (i, c)).val := by
    exact Tensor.reshapeCoordEquiv_linearize_val (Shape.internalSize_congr hsize) (i, c)
  have hchannel : rc.1 = i := by
    apply Fin.ext
    have hd := congrArg (fun k => k / spatial.size) hlinear
    change (Shape.Coord.linearize (shape := [channels, spatial.size])
        (rc.1, rc.2)).val / spatial.size =
      (Shape.Coord.linearize (shape := .dim channels spatial) (i, c)).val /
        spatial.size at hd
    have hrest : (Shape.Coord.linearize rc.2).val < spatial.size := by
      simpa only [Shape.size, Nat.mul_one] using (Shape.Coord.linearize rc.2).isLt
    rw [normalization_linearize_cons rc.1 rc.2, normalization_linearize_cons i c] at hd
    simpa only [Shape.size, Nat.mul_one,
      Nat.add_mul_div_left _ _ hspatial, Nat.div_eq_of_lt hrest,
      Nat.div_eq_of_lt (Shape.Coord.linearize c).isLt, Nat.zero_add] using hd
  have h := get2_broadcastAfterSum_one (n := spatial.size) v rc.1 rc.2.1
  have hcoord : (rc.1, rc.2.1, PUnit.unit) = rc :=
    Prod.ext rfl (Prod.ext rfl (Subsingleton.elim _ _))
  rw [Spec.get2_eq_apply, hcoord, hchannel] at h
  exact h

private theorem normalization_sqrt_max_zero (x : ℝ) : Real.sqrt (max x 0) = Real.sqrt x := by
  by_cases hx : 0 ≤ x
  · rw [max_eq_left hx]
  · rw [max_eq_right (le_of_not_ge hx), Real.sqrt_zero,
      Real.sqrt_eq_zero_of_nonpos (le_of_not_ge hx)]

/-- The coordinate formula for the actual inference specification, including variance clamping. -/
theorem batchNormInference_normalization_apply {channels : Nat} {spatial : Shape}
    (x : Tensor ℝ (.dim channels spatial)) (mean variance gamma beta : Tensor ℝ [channels])
    (epsilon : ℝ) (i : Fin channels) (c : spatial.Coord) :
    Spec.batchNormInference x mean variance gamma beta epsilon (i, c) =
      ((x (i, c) - mean.getScalar i) /
        Real.sqrt (max (variance.getScalar i) 0 + epsilon)) *
          gamma.getScalar i + beta.getScalar i := by
  simp only [Spec.batchNormInference, Tensor.addSpec, Tensor.subSpec, Tensor.mulSpec,
    Tensor.divSpec, Tensor.sqrtSpec, Shape.concat, Tensor.map2Spec_apply,
    Tensor.mapSpec, Tensor.map, TorchLean.Tensor.Internal.Rep.map_apply,
    Tensor.full_apply]
  change ((x (i, c) - Spec.broadcastChannel spatial mean (i, c)) /
    Real.sqrt (max
      (Spec.broadcastChannel spatial (Tensor.maxSpec variance (Tensor.full [channels] 0))
        (i, c) + epsilon) 0)) *
      Spec.broadcastChannel spatial gamma (i, c) +
      Spec.broadcastChannel spatial beta (i, c) = _
  rw [normalization_broadcastChannel_apply mean i c,
    normalization_broadcastChannel_apply gamma i c,
    normalization_broadcastChannel_apply beta i c,
    normalization_broadcastChannel_apply (Tensor.maxSpec variance (Tensor.full [channels] 0)) i c,
    getScalar_maxSpec, Tensor.getScalar_full]
  rw [normalization_sqrt_max_zero]

variable {α : Type} [Storage α] [Context α] [BoundOps α] [LawfulBoundOps α]

local notation "value" => LawfulBoundOps.toReal (α := α)

/-- Read the actual fixed running statistics and affine scalars for one channel. -/
def batchNormRealScalar (config : NN.IR.BatchNormEvalParams α) (channel : Fin config.c)
    (x : ℝ) : ℝ :=
  ((x - value (config.mean.getScalar channel)) /
    Real.sqrt (max (value (config.var.getScalar channel)) 0 + value config.eps)) *
      value (config.gamma.getScalar channel) + value (config.beta.getScalar channel)

/-- The real, channel-wise Spec operation applied across the same leading axes as IR evaluation. -/
def batchNormRealValue (leading spatial : Shape) (config : NN.IR.BatchNormEvalParams α)
    (f : Nat → ℝ) : Tensor ℝ (leading.concat (.dim config.c spatial)) :=
  Tensor.mapLeading leading
    (fun sample => Spec.batchNormInference sample
      (Tensor.ofFn fun i => value (config.mean.getScalar i))
      (Tensor.ofFn fun i => value (config.var.getScalar i))
      (Tensor.ofFn fun i => value (config.gamma.getScalar i))
      (Tensor.ofFn fun i => value (config.beta.getScalar i))
      (value config.eps))
    (tensorOfFlatValues (leading.concat (.dim config.c spatial)) f)

/-- The mathematical BatchNorm node equation does not mention interval boxes. -/
def BatchNormRealEquation (s : Shape) (axis : Nat) (config : NN.IR.BatchNormEvalParams α)
    (f g : Nat → ℝ) : Prop :=
  ∀ leading spatial : Shape, axis = leading.rank →
    s = leading.concat (.dim config.c spatial) →
      ∀ c : (leading.concat (.dim config.c spatial)).Coord,
        g (Shape.Coord.linearize c).val = batchNormRealValue leading spatial config f c

private theorem batchNorm_mapLeading_apply (leading : Shape) {spatial : Shape}
    (config : NN.IR.BatchNormEvalParams α)
    (x : Tensor ℝ (leading.concat (.dim config.c spatial)))
    (c : (leading.concat (.dim config.c spatial)).Coord) :
    Tensor.mapLeading leading
        (fun sample => Spec.batchNormInference sample
          (Tensor.ofFn fun i => value (config.mean.getScalar i))
          (Tensor.ofFn fun i => value (config.var.getScalar i))
          (Tensor.ofFn fun i => value (config.gamma.getScalar i))
          (Tensor.ofFn fun i => value (config.beta.getScalar i))
          (value config.eps)) x c =
      batchNormRealScalar config (normalizationChannel leading c) (x c) := by
  induction leading with
  | scalar =>
      simpa only [Tensor.mapLeading, normalizationChannel, batchNormRealScalar,
        Tensor.getScalar_ofFn] using
        batchNormInference_normalization_apply x
          (Tensor.ofFn fun i => value (config.mean.getScalar i))
          (Tensor.ofFn fun i => value (config.var.getScalar i))
          (Tensor.ofFn fun i => value (config.gamma.getScalar i))
          (Tensor.ofFn fun i => value (config.beta.getScalar i))
          (value config.eps) c.1 c.2
  | dim n leading ih =>
      simp only [Tensor.mapLeading, Tensor.dim, TorchLean.Tensor.Internal.Rep.stack_apply]
      simpa only [normalizationChannel, Tensor.unstack,
        TorchLean.Tensor.Internal.Rep.unstack_apply] using ih (x.unstack c.1) c.2

/-- The Spec equation yields the scalar expression at the engine's selected flat channel. -/
theorem BatchNormRealEquation.apply {s : Shape} {axis : Nat}
    {config : NN.IR.BatchNormEvalParams α} {f g : Nat → ℝ}
    (heq : BatchNormRealEquation s axis config f g)
    (hshape : s.toList[axis]? = some config.c) (k : Fin s.size)
    (hc : axisCoordinateOfFlat s axis k.val % config.c < config.c) :
    g k.val = batchNormRealScalar config ⟨axisCoordinateOfFlat s axis k.val % config.c, hc⟩
      (f k.val) := by
  obtain ⟨leading, spatial, haxis, hshape'⟩ := normalization_shape_decomposition hshape
  subst axis s
  let c := Shape.Coord.unlinearize k
  have hchannel : (⟨axisCoordinateOfFlat (leading.concat (.dim config.c spatial)) leading.rank
      k.val % config.c, hc⟩ : Fin config.c) = normalizationChannel leading c := by
    apply Fin.ext
    simpa only [c, Shape.Coord.linearize_unlinearize] using
      axisCoordinateOfFlat_normalizationChannel leading spatial config.c c
  have h := heq leading spatial rfl rfl c
  simpa only [batchNormRealValue, batchNorm_mapLeading_apply, tensorOfFlatValues_apply,
    c, Shape.Coord.linearize_unlinearize, hchannel] using h

variable [NonlinearBoundOps α] [LawfulNonlinearBoundOps α]

/-- Every successful evaluation-mode BatchNorm transfer encloses the real Spec operation with
the stored running statistics, scale, bias, and epsilon on every accepted channel axis. -/
theorem ibpBatchNormEval?_encloses {s : Shape} {axis : Nat}
    {config : NN.IR.BatchNormEvalParams α} {input output : FlatBox α} {f g : Nat → ℝ}
    (hx : RowEncloses input s.size f)
    (heq : BatchNormRealEquation s axis config f g)
    (hout : ibpBatchNormEval? s axis config input = some output) :
    RowEncloses output s.size g := by
  obtain ⟨d, lo, hi⟩ := input
  have hd := hx.1
  change d = s.size at hd
  subst d
  rw [rowEncloses_iff] at hx
  unfold ibpBatchNormEval? at hout
  obtain ⟨channels, hchannels, hout⟩ := Option.bind_eq_some_iff.mp hout
  split at hout
  · contradiction
  · rename_i hguard
    have hcpos : 0 < config.c := by
      by_contra h
      have hz : config.c = 0 := by omega
      simp [hz] at hguard
    have hcc : channels = config.c := by
      by_contra h
      simp [h] at hguard
    subst channels
    obtain ⟨_, _, hout⟩ := Option.bind_eq_some_iff.mp hout
    obtain ⟨parameters, hparameters, hout⟩ := Option.bind_eq_some_iff.mp hout
    have hp (ci : Fin config.c) :
        (parameters ci).1 = config.mean.getScalar ci ∧
        (parameters ci).2.1 = config.gamma.getScalar ci ∧
        (parameters ci).2.2.1 = config.beta.getScalar ci ∧
        value (parameters ci).2.2.2.1 ≤
            Real.sqrt (max (value (config.var.getScalar ci)) 0 + value config.eps) ∧
          Real.sqrt (max (value (config.var.getScalar ci)) 0 + value config.eps) ≤
            value (parameters ci).2.2.2.2 := by
      have hc := traverseFin_eq_some_iff.mp hparameters ci
      obtain ⟨_, _, hc⟩ := Option.bind_eq_some_iff.mp hc
      obtain ⟨_, _, hc⟩ := Option.bind_eq_some_iff.mp hc
      obtain ⟨_, _, hc⟩ := Option.bind_eq_some_iff.mp hc
      obtain ⟨_, _, hc⟩ := Option.bind_eq_some_iff.mp hc
      obtain ⟨stabilized, hstabilized, hc⟩ := Option.bind_eq_some_iff.mp hc
      have hstabilizedEq := LayerNormDirected.checkedFiniteBounds?_eq_of_eq_some hstabilized
      subst stabilized
      have hmax : value (max (config.var.getScalar ci) 0) =
          max (value (config.var.getScalar ci)) 0 := by
        rw [LawfulBoundOps.toReal_max, LawfulBoundOps.toReal_zero]
      have hshift := LayerNormDirected.shift_encloses (bias := config.eps)
        (le_of_eq hmax) (le_of_eq hmax.symm)
      obtain ⟨⟨denominatorLo, denominatorHi⟩, hdenominator, hc⟩ :=
        Option.bind_eq_some_iff.mp hc
      have hden := LawfulNonlinearBoundOps.sqrtBounds_enclosure
        (LayerNormDirected.checkedFiniteBounds?_bind_eq_some hdenominator)
        hshift.1 hshift.2
      dsimp only at hc
      split at hc
      · contradiction
      · have hcEq := Option.some.inj hc
        rw [← hcEq]
        exact ⟨rfl, rfl, rfl, hden⟩
    obtain ⟨bounds, hbounds, hout⟩ := Option.bind_eq_some_iff.mp hout
    have houtEq := Option.some.inj hout
    cases houtEq
    rw [rowEncloses_iff]
    intro i
    have hci : axisCoordinateOfFlat s axis i.val % config.c < config.c :=
      Nat.mod_lt _ hcpos
    let ci : Fin config.c := ⟨axisCoordinateOfFlat s axis i.val % config.c, hci⟩
    have hpoint := traverseFin_eq_some_iff.mp hbounds i
    simp only [dite_eq_left hci] at hpoint
    have hparam := hp ci
    rcases hpc : parameters ci with ⟨mean, scale, bias, denominatorLo, denominatorHi⟩
    simp only [hpc] at hparam
    change _ = some (bounds i) at hpoint
    rw [show parameters ⟨axisCoordinateOfFlat s axis i.val % config.c, hci⟩ =
      (mean, scale, bias, denominatorLo, denominatorHi) from hpc] at hpoint
    change (do
      let _ ← checkedFiniteBounds? (lo.getScalar i, hi.getScalar i)
      let (centeredLo, centeredHi) ← checkedFiniteBounds?
        (subDown (lo.getScalar i) mean, subUp (hi.getScalar i) mean)
      let (normalizedLo, normalizedHi) ←
        NonlinearBoundOps.divBounds centeredLo centeredHi denominatorLo denominatorHi >>=
          checkedFiniteBounds?
      let (scaledLo, scaledHi) ← checkedFiniteBounds?
        (intervalMul normalizedLo normalizedHi scale scale)
      checkedFiniteBounds? (addDown scaledLo bias, addUp scaledHi bias)) = some (bounds i)
      at hpoint
    obtain ⟨hmean, hscale, hbias, hden⟩ := hparam
    obtain ⟨_, _, hpoint⟩ := Option.bind_eq_some_iff.mp hpoint
    obtain ⟨centered, hcentered, hpoint⟩ := Option.bind_eq_some_iff.mp hpoint
    have hcenteredEq := LayerNormDirected.checkedFiniteBounds?_eq_of_eq_some hcentered
    subst centered
    have hcenter := LayerNormDirected.sub_encloses (hx i)
      (show value mean ≤ value mean ∧ value mean ≤ value mean from ⟨le_rfl, le_rfl⟩)
    obtain ⟨⟨normalizedLo, normalizedHi⟩, hnormalized, hpoint⟩ :=
      Option.bind_eq_some_iff.mp hpoint
    have hnormalizedBounds := LawfulNonlinearBoundOps.divBounds_enclosure
      (LayerNormDirected.checkedFiniteBounds?_bind_eq_some hnormalized)
      hcenter.1 hcenter.2 hden.1 hden.2
    have hscaled := NN.MLTheory.CROWN.IntervalLemmas.intervalMul_encloses
      hnormalizedBounds.1 hnormalizedBounds.2
      (le_refl (value scale)) (le_refl (value scale))
    obtain ⟨scaled, hscaledCheck, hpoint⟩ := Option.bind_eq_some_iff.mp hpoint
    have hscaledEq := LayerNormDirected.checkedFiniteBounds?_eq_of_eq_some hscaledCheck
    subst scaled
    have hshift := LayerNormDirected.shift_encloses (bias := bias) hscaled.1 hscaled.2
    have hfinal := LayerNormDirected.checkedFiniteBounds?_eq_of_eq_some hpoint
    rw [heq.apply hchannels i hci]
    simpa only [Tensor.getScalar_ofFn, ← hfinal, batchNormRealScalar, hmean, hscale, hbias, ci,
      intervalMul] using hshift

end

end NN.MLTheory.CROWN.Graph.DirectedBackward
