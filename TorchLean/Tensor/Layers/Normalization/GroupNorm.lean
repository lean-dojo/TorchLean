/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Tensor.Layers.Normalization

/-!
# Group normalization with channels on axis 1

`Tensor.groupNorm x dim groups gamma beta hGroups epsilon` in
`TorchLean.Tensor.Layers.Normalization` takes its channel axis `dim` anywhere in the
shape. `reference` below is group normalization written for inputs of shape
`[batch, channels] ++ spatial` only: it views the input as
`[batch, groups, channels / groups * size spatial]`, normalizes over the last axis of that view,
and applies `gamma` and `beta` per channel through a `[batch, channels, size spatial]` view.

`reference` follows `torch.nn.GroupNorm` (PyTorch 2.8 docs) by hand: the input is `(N, C, *)`, the
group count has to divide the channels, and each group is a set of eq. (7) in Wu and He, "Group
Normalization" (2018, arXiv:1803.08494).

`groupNorm_one_eq_reference` proves that `Tensor.groupNorm` with `dim = 1` equals `reference` for
every input, scale, shift, group count and `epsilon` that `reference` accepts (`groups > 0`,
`channels ≥ groups`, `channels % groups = 0`, a well-formed input shape), over any scalar type
with `Storage` and `Context`. Models use `Tensor.groupNorm`. `reference` is the right-hand side
of this theorem, and `normalizeGroups` is a step of its proof.
-/

@[expose] public section

open TorchLean.Tensor TorchLean.Shape
open Coord (linearize unlinearize unlinearize_linearize)
open TorchLean.Shape (rank size size_cons size_nil)

namespace TorchLean.GroupNorm

variable {α : Type} [Storage α] [Context α]

/-- Group normalization of an input of shape `[batch, channels] ++ spatial`, with the channels
on axis 1 split into `groups` contiguous groups of `channels / groups`. -/
def reference
    {batch channels groups : Nat} {spatial : Shape}
    (x : Tensor α (batch :: channels :: spatial))
    (gamma beta : Tensor α [channels])
    (hGroups : groups > 0 := by norm_num)
    (hGroupsLe : channels ≥ groups)
    (hDiv : channels % groups = 0)
    (epsilon : α := normalizationEpsilon)
    [Shape.WellFormed (batch :: channels :: spatial)] :
    Tensor α (batch :: channels :: spatial) :=
  let channelsPerGroup := channels / groups
  let spatialSize := Shape.size spatial
  let groupSize := channelsPerGroup * spatialSize
  let inputShape : Shape := batch :: channels :: spatial
  let groupedShape : Shape := [batch, groups, groupSize]
  let flatShape : Shape := [batch, channels, spatialSize]
  have hInput := Shape.WellFormed.proof (s := inputShape)
  have hBatch : 0 < batch := hInput.1
  have hChannels : 0 < channels := hInput.2.1
  have hSpatial : 0 < spatialSize := by
    simpa [spatialSize] using Shape.size_pos_of_positive hInput.2.2
  have hChannelsPerGroup : 0 < channelsPerGroup :=
    Nat.div_pos hGroupsLe hGroups
  have hGroupSize : 0 < groupSize :=
    Nat.mul_pos hChannelsPerGroup hSpatial
  have hChannelsEq : channels = groups * channelsPerGroup := by
    simpa [channelsPerGroup, hDiv] using (Nat.mod_add_div channels groups).symm
  letI : Shape.WellFormed groupedShape :=
    ⟨⟨hBatch, ⟨hGroups, ⟨hGroupSize, trivial⟩⟩⟩⟩
  letI : Shape.WellFormed flatShape :=
    ⟨⟨hBatch, ⟨hChannels, ⟨hSpatial, trivial⟩⟩⟩⟩
  have hGroupedSize : Shape.size inputShape = Shape.size groupedShape := by
    simp only [inputShape, groupedShape, Shape.size_nil, Shape.size_cons]
    rw [hChannelsEq]
    simp [groupSize, spatialSize, Nat.mul_assoc]
  have hFlatSize : Shape.size inputShape = Shape.size flatShape := by
    simp [inputShape, flatShape, spatialSize]
  let grouped : Tensor α groupedShape := Tensor.reshape x _ hGroupedSize
  let axis := Shape.rank groupedShape - 1
  let hAxis : Shape.HasNonemptyAxis axis groupedShape :=
    Shape.hasNonemptyAxis_of_lt_rank (by simp [axis, groupedShape])
  let rowAxis : Fin groupedShape.rank := ⟨axis, hAxis.proof.axisInBounds.proof⟩
  let mean := Tensor.map (Context.divNat · groupSize)
    (castShape (Tensor.reduce .sum grouped [rowAxis] false) (Shape.reduce_singleton _ _))
  let meanBroadcast := Tensor.broadcast mean groupedShape _
      (groupedShape.eraseIdx_broadcastDims axis)
  let centered := grouped - meanBroadcast
  let regularizer := Tensor.full (groupedShape.eraseIdx axis) epsilon
  let largest : Tensor α (groupedShape.eraseIdx axis) := castShape
    (Rep.ofFn fun o =>
      maxAbs (ofFn fun i : Fin (Shape.size (groupedShape.selectAxes [rowAxis])) =>
        centered (groupedShape.reduceSource [rowAxis] false o (Coord.unlinearize i))))
    (Shape.reduce_singleton groupedShape rowAxis)
  let scale := Tensor.map Context.stopGradient
    (Tensor.maximum largest (Tensor.sqrt (Tensor.abs regularizer) .clampNegative))
  let scaled := centered /
    Tensor.broadcast scale groupedShape _ (groupedShape.eraseIdx_broadcastDims axis)
  let meanSquare := Tensor.map (Context.divNat · groupSize)
    (castShape (Tensor.reduce .sum (scaled * scaled) [rowAxis] false) (Shape.reduce_singleton _ _))
  let denominator :=
    Tensor.sqrt (meanSquare + ((regularizer / scale) / scale)) .clampNegative
  let normalized :=
    scaled /
        Tensor.broadcast denominator groupedShape _ (groupedShape.eraseIdx_broadcastDims axis)
  let normalizedInput : Tensor α inputShape :=
    Tensor.reshape normalized _ hGroupedSize.symm
  let normalizedFlat : Tensor α flatShape :=
    Tensor.reshape normalizedInput _ hFlatSize
  let channelSpatialShape : Shape := [channels, spatialSize]
  letI : Shape.WellFormed channelSpatialShape :=
    ⟨⟨hChannels, ⟨hSpatial, trivial⟩⟩⟩
  let gammaSpatial : Tensor α channelSpatialShape :=
    Tensor.broadcast gamma channelSpatialShape _ (channelSpatialShape.eraseIdx_broadcastDims 1)
  let betaSpatial : Tensor α channelSpatialShape :=
    Tensor.broadcast beta channelSpatialShape _ (channelSpatialShape.eraseIdx_broadcastDims 1)
  let gammaFlat : Tensor α flatShape :=
    Tensor.broadcast gammaSpatial flatShape _ (flatShape.eraseIdx_broadcastDims 0)
  let betaFlat : Tensor α flatShape :=
    Tensor.broadcast betaSpatial flatShape _ (flatShape.eraseIdx_broadcastDims 0)
  let outputFlat := (normalizedFlat * gammaFlat) + betaFlat
  Tensor.reshape outputFlat _ hFlatSize.symm

/-- The normalization step of `Tensor.groupNorm` on its `[B, g, n]` view: each entry less the
mean of its row along the last axis, divided by the detached scale
`maximum (maxAbs centered row) (sqrt |epsilon|)`, then by `sqrt (m + epsilon / scale / scale)` for
the mean square `m` of the scaled row. -/
def normalizeGroups {B g n : Nat} (input : Tensor α [B, g, n]) (epsilon : α) :
    Tensor α [B, g, n] :=
  let grouped : Shape := [B, g, n]
  let last : Fin grouped.rank := ⟨2, by simp [grouped]⟩
  let spread := fun t : Tensor α (grouped.eraseAxes [last]) =>
    Tensor.broadcast t grouped (grouped.keptAxis [last]) (grouped.broadcastDims_eraseAxes [last])
  let mean := Tensor.map (Context.divNat · n) (Tensor.reduce .sum input [last] false)
  let centered := input - spread mean
  let regularizer := Tensor.full (grouped.eraseAxes [last]) epsilon
  let largest : Tensor α (grouped.eraseAxes [last]) := Rep.ofFn fun o =>
    maxAbs (ofFn fun i : Fin (Shape.size (grouped.selectAxes [last])) =>
      centered (grouped.reduceSource [last] false o (Coord.unlinearize i)))
  let scale := Tensor.map Context.stopGradient (Tensor.maximum largest
    (Tensor.sqrt (Tensor.abs regularizer) .clampNegative))
  let scaled := centered / spread scale
  let meanSquared : Tensor α (grouped.eraseAxes [last]) := Tensor.map (Context.divNat · n)
    (Tensor.reduce .sum (scaled * scaled) [last] false)
  let std := Tensor.sqrt (meanSquared + ((regularizer / scale) / scale)) .clampNegative
  scaled / spread std

/-- `Tensor.groupNorm` is `normalizeGroups` on the `[B, groups, n]` view of `x`, viewed back as
`s`, times `gamma` and plus `beta` broadcast along `dim`, for `B` the size of the axes before
`dim` and `n` the group size. -/
theorem groupNorm_eq {s : Shape} (x : Tensor α s) (dim : Fin s.rank) (groups : Nat)
    (gamma beta : Tensor α [s.extent dim]) (hGroups : groups ∣ s.extent dim) (epsilon : α)
    {B n : Nat} (hB : Shape.size (s.take dim.val) = B)
    (hn : s.extent dim / groups * Shape.size (s.drop (dim.val + 1)) = n)
    (h : s.size = Shape.size [B, groups, n]) :
    Tensor.groupNorm x dim groups gamma beta hGroups epsilon =
      Tensor.reshape (normalizeGroups (Tensor.reshape x _ h) epsilon) s h.symm *
          Tensor.broadcast gamma s _ (s.broadcastDims_axis dim) +
        Tensor.broadcast beta s _ (s.broadcastDims_axis dim) := by
  subst hB hn
  rfl

/-- Reshaping commutes with `a * b + c`. -/
theorem reshape_mul_add {s t : Shape} (a b c : Tensor α s) (h : s.size = t.size) :
    Tensor.reshape (a * b + c) t h =
      Tensor.reshape a t h * Tensor.reshape b t h + Tensor.reshape c t h := by
  refine Rep.ext fun i => ?_
  simp

/-- The row-major index of `(b, ch, r)` in `batch :: channels :: spatial` is that of
`(b, ch, linearize r)` in `[batch, channels, size spatial]`. -/
theorem linearize_flat {batch channels : Nat} {spatial : Shape} (b : Fin batch)
    (ch : Fin channels)
    (r : Shape.Coord spatial) (hk : (Coord.linearize r).val < Shape.size spatial) :
    (Coord.linearize (s := batch :: channels :: spatial) (b, ch, r)).val =
      (Coord.linearize (s := [batch, channels, Shape.size spatial])
        (b, ch, ⟨(Coord.linearize r).val, hk⟩, ())).val := by
  have e1 := Coord.linearize_cons_val (s := channels :: spatial) b (ch, r)
  have e2 := Coord.linearize_cons_val (s := spatial) ch r
  have e3 := Coord.linearize_cons_val (s := [channels, Shape.size spatial]) b
    (ch, ⟨(Coord.linearize r).val, hk⟩, ())
  have e4 := Coord.linearize_cons_val (s := [Shape.size spatial]) ch
    (⟨(Coord.linearize r).val, hk⟩, ())
  have e5 := Coord.linearize_cons_val (s := []) (⟨(Coord.linearize r).val, hk⟩ :
    Fin (Shape.size spatial)) ()
  have e6 : (Coord.linearize (s := []) ()).val = 0 := by
    have := (Coord.linearize (s := []) ()).isLt
    simp only [Shape.size, List.prod_nil] at this
    omega
  rw [e1, e2, e3, e4, e5, e6]
  simp only [Shape.size_nil, Shape.size_cons]
  ring

omit [Context α] in
/-- `gamma` broadcast along axis 1 of `[batch, channels, size spatial]`, viewed as
`[batch, channels] ++ spatial`, is `gamma` broadcast along axis 1 of that shape. -/
theorem reshape_broadcast_channel {batch channels : Nat} {spatial : Shape}
    (gamma : Tensor α [channels]) (h1 : 1 < Shape.rank (batch :: channels :: spatial))
    (h : Shape.size [batch, channels, Shape.size spatial] =
      Shape.size (batch :: channels :: spatial)) :
    Tensor.reshape
        (Tensor.broadcast
          (Tensor.broadcast gamma [channels, Shape.size spatial] _
            (Shape.eraseIdx_broadcastDims [channels, Shape.size spatial] 1))
          [batch, channels, Shape.size spatial] _
          (Shape.eraseIdx_broadcastDims [batch, channels, Shape.size spatial] 0))
        _ h =
      Tensor.broadcast gamma (batch :: channels :: spatial) _
        (Shape.broadcastDims_axis (batch :: channels :: spatial) ⟨1, h1⟩) := by
  refine Rep.ext fun c => ?_
  rw [Tensor.reshape_apply]
  erw [Tensor.broadcast_apply, Tensor.broadcast_apply]
  refine Eq.trans ?_ (Tensor.broadcast_apply _ _ _ _ c).symm
  obtain ⟨b, ch, r⟩ := c
  have hk : (Coord.linearize r).val < Shape.size spatial := (Coord.linearize r).isLt
  let c' : Shape.Coord [batch, channels, Shape.size spatial] :=
    (b, ch, ⟨(Coord.linearize r).val, hk⟩, ())
  have hc : Coord.unlinearize (Fin.cast h.symm
        (Coord.linearize (s := batch :: channels :: spatial) (b, ch, r))) =
      c' := by
    have hl : Fin.cast h.symm
        (Coord.linearize (s := batch :: channels :: spatial) (b, ch, r)) =
        Coord.linearize c' := by
      exact Fin.ext (linearize_flat b ch r hk)
    rw [hl, Coord.unlinearize_linearize]
  rw [hc]
  refine congrArg (Rep.get gamma) ?_
  apply Shape.Coord.ext_val
  intro j
  obtain ⟨j, hj⟩ := j
  obtain rfl : j = 0 := by simp at hj; omega
  simp only [Shape.Coord.get_ofFun]
  by_cases hA : Shape.extent [channels] ⟨0, hj⟩ = 1
  · rw [dite_eq_left hA, dite_eq_left hA]
  · rw [dite_eq_right hA, dite_eq_right hA]
    simp only [Fin.val_cast]
    rw [dite_eq_right]
    · rfl
    · simpa [Shape.extent] using hA

/-- `reference` unfolded: the normalization step on the grouped view, viewed as
`[batch, channels, size spatial]`, times and plus the broadcast `gamma` and `beta`, viewed back. -/
theorem reference_eq {batch channels groups : Nat} {spatial : Shape}
    (x : Tensor α (batch :: channels :: spatial)) (gamma beta : Tensor α [channels])
    (hGroups : groups > 0) (hGroupsLe : channels ≥ groups) (hDiv : channels % groups = 0)
    (epsilon : α) [Shape.WellFormed (batch :: channels :: spatial)]
    (hG : Shape.size (batch :: channels :: spatial) =
      Shape.size [batch, groups, channels / groups * Shape.size spatial])
    (hF : Shape.size (batch :: channels :: spatial) =
      Shape.size [batch, channels, Shape.size spatial]) :
    reference x gamma beta hGroups hGroupsLe hDiv epsilon =
      Tensor.reshape
        (Tensor.reshape (Tensor.reshape (normalizeGroups (Tensor.reshape x _ hG) epsilon) _ hG.symm)
            _ hF *
          Tensor.broadcast (Tensor.broadcast gamma [channels, Shape.size spatial] _
              (Shape.eraseIdx_broadcastDims [channels, Shape.size spatial] 1))
            [batch, channels, Shape.size spatial] _
            (Shape.eraseIdx_broadcastDims [batch, channels, Shape.size spatial] 0) +
          Tensor.broadcast (Tensor.broadcast beta [channels, Shape.size spatial] _
              (Shape.eraseIdx_broadcastDims [channels, Shape.size spatial] 1))
            [batch, channels, Shape.size spatial] _
            (Shape.eraseIdx_broadcastDims [batch, channels, Shape.size spatial] 0))
        _ hF.symm := by
  unfold reference normalizeGroups
  rfl

/-- `groupNorm_one_eq_reference` for any proof that axis 1 is in range. -/
theorem groupNorm_one_eq_reference' {batch channels groups : Nat} {spatial : Shape}
    (x : Tensor α (batch :: channels :: spatial)) (gamma beta : Tensor α [channels])
    (hGroups : groups > 0) (hGroupsLe : channels ≥ groups) (hDiv : channels % groups = 0)
    (epsilon : α) [Shape.WellFormed (batch :: channels :: spatial)]
    (h1 : 1 < Shape.rank (batch :: channels :: spatial)) :
    Tensor.groupNorm x ⟨1, h1⟩ groups gamma beta (Nat.dvd_of_mod_eq_zero hDiv) epsilon =
      reference x gamma beta hGroups hGroupsLe hDiv epsilon := by
  have hG : Shape.size (batch :: channels :: spatial) =
      Shape.size [batch, groups, channels / groups * Shape.size spatial] := by
    have h := Shape.size_eq_take_extent_drop (batch :: channels :: spatial) ⟨1, h1⟩
    conv_lhs => rw [h]
    show batch * 1 * (channels * Shape.size spatial) = _
    conv_lhs => rw [← Nat.div_mul_cancel (Nat.dvd_of_mod_eq_zero hDiv)]
    simp only [Shape.size_nil, Shape.size_cons, Nat.mul_one]
    rw [Nat.mul_comm (channels / groups) groups, Nat.mul_assoc]
  have hF : Shape.size (batch :: channels :: spatial) =
      Shape.size [batch, channels, Shape.size spatial] := by
    simp only [Shape.size_nil, Shape.size_cons, Nat.mul_one]
  rw [groupNorm_eq x ⟨1, h1⟩ groups gamma beta _ epsilon (B := batch)
      (n := channels / groups * Shape.size spatial) (Nat.mul_one batch) rfl hG,
    reference_eq x gamma beta hGroups hGroupsLe hDiv epsilon hG hF, reshape_mul_add,
    reshape_broadcast_channel gamma h1, reshape_broadcast_channel beta h1]
  rfl

/-- `Tensor.groupNorm` with the channels on axis 1 equals `reference` on every input of shape
`[batch, channels] ++ spatial`, under exactly `reference`'s hypotheses. -/
theorem groupNorm_one_eq_reference {batch channels groups : Nat} {spatial : Shape}
    (x : Tensor α (batch :: channels :: spatial)) (gamma beta : Tensor α [channels])
    (hGroups : groups > 0) (hGroupsLe : channels ≥ groups) (hDiv : channels % groups = 0)
    (epsilon : α) [Shape.WellFormed (batch :: channels :: spatial)] :
    Tensor.groupNorm x ⟨1, by simp⟩ groups gamma beta (Nat.dvd_of_mod_eq_zero hDiv)
        epsilon =
      reference x gamma beta hGroups hGroupsLe hDiv epsilon :=
  groupNorm_one_eq_reference' x gamma beta hGroups hGroupsLe hDiv epsilon _

end TorchLean.GroupNorm
