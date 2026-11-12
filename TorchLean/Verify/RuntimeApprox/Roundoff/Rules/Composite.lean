/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/
module

public import TorchLean.Tensor.Statistics
public import TorchLean.Tensor.Layers.Attention
public import TorchLean.Tensor.Layers.Loss.Axis
public import TorchLean.Tensor.Layers.MoE
public import TorchLean.Tensor.Layers.Normalization
public import TorchLean.Tensor.Layers.Softmax
public meta import TorchLean.Verify.RuntimeApprox.Roundoff.Rules
public import FloatLib.Floats.Formats.BinaryInterchange.Automation.Roundoff.Extension
public import TorchLean.Verify.RuntimeApprox.Roundoff.Rules
public import TorchLean.Verify.RuntimeApprox.Roundoff.Rules.Elementwise
import FloatLib.Floats.Formats.BinaryInterchange.Analysis.Rational
import TorchLean.Tensor.List

/-!
# Composites along any axes for `roundoff`

This module registers the read lemmas of the general composites. Softmax (with and without a mask),
log-softmax along any axis, their JVP and VJPs, the log-probabilities of cross-entropy and the top-k
routing mask are read as their vector kernels on the fiber through the coordinate, with
`roundoff_read`, so they unfold where the shape is literal. Log-sum-exp, the cosine loss and the
variance over any axes are read with `roundoff_rule` as their computations on the fiber of the
output coordinate.

Layer and batch normalization unfold into broadcasts, sums and variances over their axes. Their
detached scale `Tensor.normalizationScale` has rules of its own: the largest magnitude of finite
entries is exact, so the scale is off by at most the largest error among its centered entries and
`sqrt |regularizer|` (`abs_toReal_normalizationScale_sub_le`, and
`mixedError_toReal_normalizationScale` in mixed analysis). We read a statistic over the axes that is
broadcast back along the kept axes at its reduced shape, where the read law of its producer applies
(`TorchLean.Tensor.broadcast_eraseAxes_apply`, which `reducedStatistics` tries before the general
broadcast read). At a point of the fiber of an output coordinate `o`, it is read at `o`
(`TorchLean.Shape.ofFun_get_reduceSource_keptAxis`). Inside a sum along one axis, a statistic
broadcast back along that axis is read at `(e (e.symm (i, c))).2` for
`e = Shape.Coord.axisEquiv axis`, as in `Normalization.normalizedRows` (the JVP and backward kernels
of layer and batch normalization). We register Mathlib's `Equiv.apply_symm_apply`, so the statistic
is read at `c`. The scale of `Normalization.normalizedRows` is cast to the shape of the rows, and we
read it at the coordinate of the reduced shape with the same indices
(`TorchLean.Tensor.castShape_normalizationScale_apply`, which `reducedStatistics` tries before the
general cast read), where the scale rules apply. Group normalization is rewritten by
`TorchLean.Tensor.groupNorm_def`, which types the statistics of its grouped view at the reduced
shape, so the same reads apply. The largest magnitude in its scale is exact on finite entries
(`toReal_maxAbs`).
-/

public section

attribute [roundoff_read]
  TorchLean.Tensor.softmax_apply
  TorchLean.Tensor.softmax_mask_apply
  TorchLean.Tensor.softmax_online_apply
  TorchLean.MoE.topKMask_apply
  TorchLean.Tensor.softmaxBackward_apply
  TorchLean.Tensor.maskedSoftmaxBackward_apply
  TorchLean.Tensor.logSoftmax_apply
  TorchLean.Tensor.logSoftmaxJVP_apply
  TorchLean.Tensor.logSoftmaxBackward_apply
  TorchLean.CrossEntropy.logProbs_apply

attribute [roundoff_rule]
  TorchLean.Tensor.logsumexp_apply
  TorchLean.Tensor.cosineLoss_apply
  TorchLean.Tensor.var_apply

namespace TorchLean.Shape

/-- The indices of `s.reduceSource dims false o r` along the axes kept by the reduction, as a
coordinate of the reduced shape, are `o`. -/
theorem ofFun_get_reduceSource_keptAxis {s : Shape} (dims : List (Fin s.rank))
    (o : (s.reduce dims false).Coord) (r : (s.selectAxes dims).Coord) :
    (Coord.ofFun (s := s.reduce dims false) fun i =>
      Fin.cast (s.extent_keptAxis dims i).symm
        (Coord.get (s.reduceSource dims false o r) (s.keptAxis dims i))) = o := by
  have key : (Coord.ofFun (s := s.eraseAxes dims) fun i =>
      Fin.cast (s.extent_keptAxis dims i).symm
        (Coord.get (s.reduceSource dims false o r) (s.keptAxis dims i))) = o := by
    apply Coord.ext_index
    intro k hk
    have hlen : k < (s.otherAxes dims).length := by
      simpa [eraseAxes, rank_eq_length] using hk
    have hnot : (s.otherAxes dims)[k] ∉ dims := mem_otherAxes.mp (List.getElem_mem hlen)
    rw [Coord.index_ofFun, dite_eq_left hk, Fin.val_cast, Coord.val_get, index_reduceSource,
      dite_eq_left (Fin.isLt _)]
    simp only [keptAxis, Fin.eta, hnot, ↓reduceIte, keptIndex, Bool.cond_false,
      (s.nodup_otherAxes dims).idxOf_getElem k hlen]
    rfl
  exact key

end TorchLean.Shape

namespace TorchLean.Tensor

variable {α : Type} [Storage α] {s : Shape}

/-- A tensor `t` of the shape `s.reduce dims false` broadcast back to `s` along the kept axes reads
at `c` the entry of `t`, at that shape, at the indices of `c` along those axes. -/
theorem broadcast_eraseAxes_apply (dims : List (Fin s.rank)) (t : Tensor α (s.reduce dims false))
    (c : s.Coord) :
    broadcast (s := s.eraseAxes dims) t s (s.keptAxis dims) (s.broadcastDims_eraseAxes dims) c =
      Rep.get (s := s.reduce dims false) t
        (Shape.Coord.ofFun (s := s.reduce dims false) fun i =>
          Fin.cast (s.extent_keptAxis dims i).symm (Shape.Coord.get c (s.keptAxis dims i))) :=
  broadcast_apply_of_extent_eq t (s.keptAxis dims) (s.broadcastDims_eraseAxes dims)
    (s.extent_keptAxis dims) c

/-- A normalization scale cast to the shape `t` reads at `c` the scale at the coordinate of the
reduced shape with the indices of `c`. -/
theorem castShape_normalizationScale_apply [Context α] (centered : Tensor α s)
    (dims : List (Fin s.rank)) (regularizer : Tensor α (s.reduce dims false)) {t : Shape}
    (q : s.reduce dims false = t) (c : t.Coord) :
    castShape (normalizationScale centered dims regularizer) q c =
      normalizationScale centered dims regularizer (Shape.Coord.reindexEq q.symm c) :=
  castShape_apply_reindex _ q c

/-- `groupNorm` is its definition with the statistics of the grouped view typed at the reduced
shape `grouped.reduce [last] false`, where the reads of reductions and of broadcasts back apply. -/
theorem groupNorm_def [Context α] (x : Tensor α s) (dim : Fin s.rank) (groups : Nat)
    (gamma beta : Tensor α [s.extent dim]) (hGroups : groups ∣ s.extent dim) (epsilon : α) :
    groupNorm x dim groups gamma beta hGroups epsilon =
      let groupSize := s.extent dim / groups * Shape.size (s.drop (dim.val + 1))
      let grouped : Shape := [Shape.size (s.take dim.val), groups, groupSize]
      have hSize : s.size = grouped.size := by
        rw [s.size_eq_take_extent_drop dim]
        conv_lhs => rw [← Nat.div_mul_cancel hGroups]
        simp only [grouped, groupSize, Shape.size_nil, Shape.size_cons, Nat.mul_one]
        rw [Nat.mul_comm (s.extent dim / groups) groups, Nat.mul_assoc]
      let last : Fin grouped.rank := ⟨2, by simp [grouped]⟩
      let spread := fun t : Tensor α (grouped.eraseAxes [last]) =>
        broadcast t grouped (grouped.keptAxis [last]) (grouped.broadcastDims_eraseAxes [last])
      let input := reshape x grouped hSize
      let mean := map (Context.divNat · groupSize) (reduce .sum input [last] false)
      let centered := input - spread mean
      let regularizer := full (grouped.reduce [last] false) epsilon
      let largest : Tensor α (grouped.reduce [last] false) := Rep.ofFn fun o =>
        maxAbs (ofFn fun i : Fin (Shape.size (grouped.selectAxes [last])) =>
          centered (grouped.reduceSource [last] false o (Shape.Coord.unlinearize i)))
      let scale := map Context.stopGradient (Tensor.maximum largest (Tensor.sqrt (abs regularizer)
          .clampNegative))
      let scaled := centered / spread scale
      let meanSquared : Tensor α (grouped.reduce [last] false) :=
        map (Context.divNat · groupSize) (reduce .sum (scaled * scaled) [last] false)
      let std := Tensor.sqrt (meanSquared + ((regularizer / scale) / scale)) .clampNegative
      let channel := fun t : Tensor α [s.extent dim] => broadcast t s _ (s.broadcastDims_axis dim)
      (reshape (scaled / spread std) s hSize.symm * channel gamma) + channel beta :=
  rfl

end TorchLean.Tensor

attribute [roundoff_rule] TorchLean.Shape.ofFun_get_reduceSource_keptAxis Equiv.apply_symm_apply

namespace TorchLean.Roundoff

open FloatLib.Floats FloatLib.Floats.Formats.BinaryInterchange FloatLib.Numerics TorchLean
open ExecFloat.Binary (toModel toReal isFinite)

/-- Two left folds of `max` from starts within `c` of each other stay within the left fold of `max`
over the entrywise distance bounds `e`, from `c`. -/
private theorem abs_foldl_max_sub_le {ι : Type} (u v e : ι → ℝ) (h : ∀ i, |u i - v i| ≤ e i) :
    ∀ (l : List ι) (a b c : ℝ), |a - b| ≤ c →
      |l.foldl (fun m i => max m (u i)) a - l.foldl (fun m i => max m (v i)) b| ≤
        l.foldl (fun m i => max m (e i)) c
  | [], _, _, _, hc => hc
  | i :: l, a, b, c, hc => abs_foldl_max_sub_le u v e h l (max a (u i)) (max b (v i))
      (max c (e i)) ((abs_max_sub_max_le_max a (u i) b (v i)).trans (max_le_max hc (h i)))

/-- Left folds of `max` over entries with mixed errors against nonnegative exact entries, from a
start with a mixed error against a nonnegative exact start, have as mixed error the left folds of
`max` over the relative and the absolute errors. -/
private theorem mixedError_foldl_max {ι : Type} {u v r a : ι → ℝ}
    (h : ∀ i, MixedError (u i) (v i) (r i) (a i)) (hv : ∀ i, 0 ≤ v i) :
    ∀ (l : List ι) (x x₀ ρ η : ℝ), MixedError x x₀ ρ η → 0 ≤ x₀ →
      MixedError (l.foldl (fun m i => max m (u i)) x) (l.foldl (fun m i => max m (v i)) x₀)
        (l.foldl (fun m i => max m (r i)) ρ) (l.foldl (fun m i => max m (a i)) η)
  | [], _, _, _, _, hx, _ => hx
  | i :: l, _, _, _, _, hx, hx₀ => mixedError_foldl_max h hv l _ _ _ _
      (hx.max_of_nonneg (h i) hx₀ (hv i)) (le_max_of_le_left hx₀)

/-- A left fold of `max` over nonnegative reals that starts at one of its own entries equals the
fold from `0`. -/
private theorem foldl_max_start {ι : Type} (l : List ι) (g : ι → ℝ) (hg : ∀ i, 0 ≤ g i)
    {j : ι} (hj : j ∈ l) :
    l.foldl (fun m i => max m (g i)) (g j) = l.foldl (fun m i => max m (g i)) 0 := by
  apply le_antisymm
  · exact List.foldl_max_le_of_le l g (List.le_foldl_max_of_mem l g hj)
      fun i hi => List.le_foldl_max_of_mem l g hi
  · exact List.foldl_max_le_of_le l g ((hg j).trans (List.le_foldl_max_init l g (g j)))
      fun i hi => List.le_foldl_max_of_mem l g hi

section Scale

variable {format : FloatFormat} {plan : Configured.StoragePlan format} {code : Type}
  [ExecFloat.ModelCodec plan (Model format) code]

local notation "Value" => ExecFloat (Configured.Family format code plan)

/-- With finite centered entries on the fiber of `o` and a finite `sqrt |regularizer|` there, in a
format with signed zeros, the normalization scale at `o` is finite and its real value is the
maximum of the left fold of `max` over the absolute real entries from `0` and the real root. -/
private theorem toReal_normalizationScale [Storage Value] {s : Shape}
    (centered : Tensor Value s) (dims : List (Fin s.rank))
    (regularizer : Tensor Value (s.reduce dims false)) (o : (s.reduce dims false).Coord)
    (signedZero : format.supportsSignedZero = true)
    (finite : ∀ i, isFinite
      (centered (s.reduceSource dims false o (Shape.Coord.unlinearize i))) = true)
    (rootFinite : isFinite (Tensor.sqrt (Tensor.abs regularizer) .clampNegative o) = true) :
    isFinite (Tensor.normalizationScale centered dims regularizer o) = true ∧
      toReal (Tensor.normalizationScale centered dims regularizer o) =
        max ((List.finRange (Shape.size (s.selectAxes dims))).foldl
            (fun largest i => max largest
              |toReal (centered (s.reduceSource dims false o
                (Shape.Coord.unlinearize i)))|) 0)
          (toReal (Tensor.sqrt (Tensor.abs regularizer) .clampNegative o)) := by
  have absFinite : ∀ i, isFinite (Tensor.abs centered
      (s.reduceSource dims false o (Shape.Coord.unlinearize i))) = true := by
    intro i
    simp only [Tensor.abs, Tensor.map_apply]
    exact ExecFloat.Binary.isFinite_abs (finite i)
  have absValue : ∀ i, toReal (Tensor.abs centered
      (s.reduceSource dims false o (Shape.Coord.unlinearize i))) =
        |toReal (centered (s.reduceSource dims false o
          (Shape.Coord.unlinearize i)))| := by
    intro i
    simp only [Tensor.abs, Tensor.map_apply]
    change Model.toReal (ExecFloat.Binary.toModel (ExecFloat.Binary.abs _)) =
      |Model.toReal (ExecFloat.Binary.toModel _)|
    rw [ExecFloat.Binary.toModel_abs]
    exact Model.toReal_abs_of_isFinite _ signedZero (finite i)
  have combine : ∀ largest : Value, isFinite largest = true →
      toReal largest = (List.finRange (Shape.size (s.selectAxes dims))).foldl
        (fun largest i => max largest
          |toReal (centered (s.reduceSource dims false o
            (Shape.Coord.unlinearize i)))|) 0 →
      isFinite (Context.maximum largest
          (Tensor.sqrt (Tensor.abs regularizer) .clampNegative o)) = true ∧
        toReal (Context.maximum largest
            (Tensor.sqrt (Tensor.abs regularizer) .clampNegative o)) =
          max ((List.finRange (Shape.size (s.selectAxes dims))).foldl
              (fun largest i => max largest
                |toReal (centered (s.reduceSource dims false o
                  (Shape.Coord.unlinearize i)))|) 0)
            (toReal (Tensor.sqrt (Tensor.abs regularizer) .clampNegative o)) :=
    fun largest largestFinite largestValue =>
      ⟨isFinite_maximum largestFinite rootFinite,
        (toReal_maximum largestFinite rootFinite).trans (by rw [largestValue])⟩
  have hstop : ∀ v : Value, Context.stopGradient v = v := fun _ => rfl
  by_cases hP : ∀ d ∈ dims, 0 < s.extent d
  · simp only [Tensor.normalizationScale, dite_eq_left hP, Tensor.detach_apply,
      Tensor.maximum_apply, hstop]
    have hfold := toReal_foldl_maximum
      (fun i => Tensor.abs centered
        (s.reduceSource dims false o (Shape.Coord.unlinearize i)))
      (List.finRange _) _ (absFinite ⟨0, Shape.size_selectAxes_pos hP⟩)
      fun i _ => absFinite i
    refine combine _ ?_ ?_
    · rw [Tensor.reduce_max_apply]
      exact hfold.1
    · rw [Tensor.reduce_max_apply]
      refine hfold.2.trans ?_
      simp only [absValue]
      exact foldl_max_start _
        (fun i => |toReal (centered (s.reduceSource dims false o
          (Shape.Coord.unlinearize i)))|)
        (fun _ => abs_nonneg _) (List.mem_finRange _)
  · simp only [Tensor.normalizationScale, dite_eq_right hP, Tensor.detach_apply,
      Tensor.maximum_apply, hstop]
    have hsize : Shape.size (s.selectAxes dims) = 0 := by
      obtain ⟨d, hd, hlt⟩ : ∃ d ∈ dims, ¬ 0 < s.extent d :=
        Classical.byContradiction fun hnone =>
          hP fun d hd => Classical.byContradiction fun hlt => hnone ⟨d, hd, hlt⟩
      rw [Shape.size_eq_prod]
      apply List.prod_eq_zero
      simp only [Shape.selectAxes, List.mem_map, Shape.mem_sortedAxes]
      exact ⟨d, hd, by omega⟩
    have hempty : (List.finRange (Shape.size (s.selectAxes dims))).foldl
        (fun largest i => max largest
          |toReal (centered (s.reduceSource dims false o
            (Shape.Coord.unlinearize i)))|) (0 : ℝ) = 0 :=
      le_antisymm (List.foldl_max_le_of_le _ _ le_rfl fun i _ => absurd i.isLt (by omega))
        (List.le_foldl_max_init _ _ 0)
    refine combine _ ?_ ?_
    · rw [Tensor.full_apply]
      exact ExecFloat.Binary.isFinite_zero
    · rw [Tensor.full_apply, hempty]
      exact ExecFloat.Binary.toReal_zero

/-- A normalization scale over finite centered entries and a finite `sqrt |regularizer|`, in a
format with signed zeros, is finite. -/
theorem isFinite_normalizationScale [Storage Value] {s : Shape}
    {centered : Tensor Value s} {dims : List (Fin s.rank)}
    {regularizer : Tensor Value (s.reduce dims false)} {o : (s.reduce dims false).Coord}
    (signedZero : format.supportsSignedZero = true)
    (finite : ∀ i, isFinite
      (centered (s.reduceSource dims false o (Shape.Coord.unlinearize i))) = true)
    (rootFinite : isFinite (Tensor.sqrt (Tensor.abs regularizer) .clampNegative o) = true) :
    isFinite (Tensor.normalizationScale centered dims regularizer o) = true :=
  (toReal_normalizationScale centered dims regularizer o signedZero finite rootFinite).1

/-- If each finite centered entry on the fiber of `o` is within `error i` of `reference i` and the
finite `sqrt |regularizer|` at `o` is within `rootError` of `root`, then in a format with signed
zeros the normalization scale at `o` is within `max (foldl max error 0) rootError` of
`max (foldl max |reference| 0) root`. -/
theorem abs_toReal_normalizationScale_sub_le [Storage Value] {s : Shape}
    {centered : Tensor Value s} {dims : List (Fin s.rank)}
    {regularizer : Tensor Value (s.reduce dims false)} {o : (s.reduce dims false).Coord}
    {reference error : Fin (Shape.size (s.selectAxes dims)) → ℝ}
    {root rootError : ℝ} (signedZero : format.supportsSignedZero = true)
    (bound : ∀ i, |toReal (centered (s.reduceSource dims false o
      (Shape.Coord.unlinearize i))) - reference i| ≤ error i)
    (finite : ∀ i, isFinite
      (centered (s.reduceSource dims false o (Shape.Coord.unlinearize i))) = true)
    (rootBound : |toReal (Tensor.sqrt (Tensor.abs regularizer) .clampNegative o) - root| ≤
      rootError)
    (rootFinite : isFinite (Tensor.sqrt (Tensor.abs regularizer) .clampNegative o) = true) :
    |toReal (Tensor.normalizationScale centered dims regularizer o) -
        max ((List.finRange (Shape.size (s.selectAxes dims))).foldl
          (fun largest i => max largest |reference i|) 0) root| ≤
      max ((List.finRange (Shape.size (s.selectAxes dims))).foldl
        (fun largest i => max largest (error i)) 0) rootError := by
  rw [(toReal_normalizationScale centered dims regularizer o signedZero finite rootFinite).2]
  exact (abs_max_sub_max_le_max _ _ _ _).trans (max_le_max
    (abs_foldl_max_sub_le
      (fun i => |toReal (centered (s.reduceSource dims false o
        (Shape.Coord.unlinearize i)))|)
      (fun i => |reference i|) error
      (fun i => (abs_abs_sub_abs_le_abs_sub _ _).trans (bound i)) _ 0 0 0 (by simp)) rootBound)

/-- If each finite centered entry on the fiber of `o` has mixed error `relative i`, `absolute i`
against `reference i` and the finite `sqrt |regularizer|` at `o` has mixed error `rootRelative`,
`rootAbsolute` against `root ≥ 0`, then in a format with signed zeros the normalization scale at
`o` has mixed error `max (foldl max relative 0) rootRelative`, `max (foldl max absolute 0)
rootAbsolute` against `max (foldl max |reference| 0) root`. -/
theorem mixedError_toReal_normalizationScale [Storage Value] {s : Shape}
    {centered : Tensor Value s} {dims : List (Fin s.rank)}
    {regularizer : Tensor Value (s.reduce dims false)} {o : (s.reduce dims false).Coord}
    {reference relative absolute : Fin (Shape.size (s.selectAxes dims)) → ℝ}
    {root rootRelative rootAbsolute : ℝ} (signedZero : format.supportsSignedZero = true)
    (bound : ∀ i, MixedError (toReal (centered (s.reduceSource dims false o
      (Shape.Coord.unlinearize i)))) (reference i) (relative i) (absolute i))
    (finite : ∀ i, isFinite
      (centered (s.reduceSource dims false o (Shape.Coord.unlinearize i))) = true)
    (rootBound : MixedError (toReal (Tensor.sqrt (Tensor.abs regularizer) .clampNegative o)) root
      rootRelative rootAbsolute)
    (rootFinite : isFinite (Tensor.sqrt (Tensor.abs regularizer) .clampNegative o) = true)
    (rootNonneg : 0 ≤ root) :
    MixedError (toReal (Tensor.normalizationScale centered dims regularizer o))
      (max ((List.finRange (Shape.size (s.selectAxes dims))).foldl
        (fun largest i => max largest |reference i|) 0) root)
      (max ((List.finRange (Shape.size (s.selectAxes dims))).foldl
        (fun largest i => max largest (relative i)) 0) rootRelative)
      (max ((List.finRange (Shape.size (s.selectAxes dims))).foldl
        (fun largest i => max largest (absolute i)) 0) rootAbsolute) := by
  rw [(toReal_normalizationScale centered dims regularizer o signedZero finite rootFinite).2]
  exact (mixedError_foldl_max (fun i => (bound i).abs) (fun i => abs_nonneg _) _ 0 0 0 0
    (MixedError.refl 0) le_rfl).max_of_nonneg rootBound (List.le_foldl_max_init _ _ 0) rootNonneg

end Scale

attribute [roundoff_rule]
  isFinite_normalizationScale
  abs_toReal_normalizationScale_sub_le
  mixedError_toReal_normalizationScale

end TorchLean.Roundoff

public meta section

namespace TorchLean.Roundoff

open FloatLib.Floats.Formats.BinaryInterchange.Roundoff

/-- The program rewrites that keep statistics at their reduced shape: the read of a broadcast back
along the kept axes (`TorchLean.Tensor.broadcast_eraseAxes_apply`) ahead of the general broadcast
read, the read of a cast normalization scale (`TorchLean.Tensor.castShape_normalizationScale_apply`)
ahead of the general cast read, and `TorchLean.Tensor.groupNorm_def`. -/
def reducedStatistics : Extension where
  programRewrites := #[(``TorchLean.Tensor.broadcast_eraseAxes_apply, eval_prio high),
    (``TorchLean.Tensor.castShape_normalizationScale_apply, eval_prio high),
    (``TorchLean.Tensor.groupNorm_def, eval_prio high)]

attribute [roundoff_extension] reducedStatistics

end TorchLean.Roundoff
