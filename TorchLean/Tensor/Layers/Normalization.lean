/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Tensor.Statistics
public import TorchLean.Tensor.Layers.Normalization.BatchNorm
public import TorchLean.Tensor.Layers.Normalization.LayerNorm
public import TorchLean.Tensor.Layers.SpecialForms

/-!
# Layer, RMS, batch and group normalization over any axes

`layerNorm x dims gamma beta epsilon` normalizes every fiber of `x` over the axes `dims`, at any
rank. Each fiber has its mean subtracted, and the centered fiber is divided by
`scale = maximum (max |centered|) (sqrt |epsilon|)` (`Tensor.normalizationScale`, gradient
stopped). `var` takes the variance `v` of the scaled fiber, centering it a second time, and each
scaled entry is divided by `sqrt (max (v + epsilon / scale / scale) 0)` before `gamma` scales and
`beta` shifts it. No unscaled square is summed, so a fiber whose sum of squared deviations
overflows the format, but whose variance does not, normalizes without overflow: in binary16, four
entries at `±128` give `±1`. The mean is the unscaled running sum of the entries divided by their
count, so a fiber where that sum overflows, such as `[32544, 32992]` in binary16, gives NaN at every
entry. In real arithmetic, with a nonnegative `epsilon`, each entry is the centered entry over
`sqrt (variance + epsilon)`, as in PyTorch's `F.layer_norm`. `gamma` and `beta` are indexed by the
axes `dims` and broadcast along the others.

`rmsNorm x dims gamma epsilon` divides every fiber of `x` over `dims` by
`scale = maximum (maxAbs fiber) (sqrt |epsilon|)`, with the gradient stopped, then by
`sqrt (max (m + epsilon / scale / scale) 0)` for the mean `m` of the squared scaled fiber, and
multiplies by `gamma` along `dims`. Dividing by `scale` first avoids summing unscaled squares. In
real arithmetic, with a nonnegative `epsilon` and a positive scale, the result is
`x / sqrt (mean x^2 + epsilon) * gamma`, as in PyTorch's `F.rms_norm` with `normalized_shape` the
trailing axes.

`batchNorm x dim gamma beta stats epsilon` normalizes with one mean and variance per index along
the channel axis `dim`. With `.batch` they are the mean and the population variance over every
other axis. That variance is the mean of the squared centered entries after their division by
`Tensor.normalizationScale`, so that, as in `groupNorm`, no unscaled square is summed. With
`.running mean var` they are supplied, as at inference.

`groupNorm x dim groups gamma beta hGroups epsilon` splits the channels along `dim` into `groups`
contiguous groups. For every index of the axes before `dim`, each group's entries over its
channels and every axis after `dim` are centered by their mean. They are then divided by
`scale = maximum (maxAbs centered) (sqrt |epsilon|)`, with the gradient stopped, and by
`sqrt (max (m + epsilon / scale / scale) 0)` for the mean `m` of the squared scaled group, before
`gamma` scales and `beta` shifts each channel. No unscaled square is summed: in binary16 the
squared deviations of four entries at `±128` sum to `65536`, past the largest finite value, yet
the group gives `±1`. The mean is the unscaled running sum of the group's entries divided by their
count, so a group where that sum overflows gives NaN at every entry of the group. In real
arithmetic, with a nonnegative `epsilon` and a positive scale, each entry is the centered entry
over `sqrt (variance + epsilon)`, as in PyTorch's `GroupNorm` with `dim = 1` (Wu and He,
"Group Normalization", 2018, arXiv:1803.08494, Sec. 3.1).

`TorchLean.BatchNorm.runningBackward` is the reverse rule of `batchNorm` with running statistics,
which it holds fixed together with epsilon.

`layerNorm` follows Ba, Kiros and Hinton, "Layer Normalization" (2016, arXiv:1607.06450), with
the statistics of eq. (3) in Sec. 3 and the gain and bias of eq. (4) in Sec. 3.1. `rmsNorm` is
eq. (4) of Zhang and Sennrich, "Root Mean Square Layer Normalization" (NeurIPS 2019,
arXiv:1910.07467), Sec. 4. We typed both in by hand. None of these equations has an epsilon, and
ours sits where torch's does. `batchNorm` with `.batch` is Algorithm 1 of Ioffe and Szegedy,
"Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift"
(2015, arXiv:1502.03167), and `groupNorm` is Wu and He's eqs. (1), (2) and (6) over the sets of
their eq. (7). Every variance divides by the count, the biased estimate that the forward passes of
`torch.nn.LayerNorm`, `torch.nn.BatchNorm2d` and `torch.nn.GroupNorm` (PyTorch 2.8 docs) use too.
The default `normalizationEpsilon` is `1e-5`, torch's `eps` for those three. `torch.nn.RMSNorm`
defaults to `torch.finfo(x.dtype).eps`, `2^-23` in binary32, so a caller matching it passes that.
With `.running mean var` the result is step 11 of Ioffe and Szegedy's Algorithm 2 over `ℝ`, and
`batchNorm` never updates the statistics (torch's training mode does, at `momentum=0.1` by
default). A negative `var` is clamped to `0`, where torch takes the root of `var + eps` as it
stands and gives NaN at `var = -1`.

In `scripts/checks/test_ir_capture.py`, the `PyTorchCaptureTests` test
`test_normalizations_execute_with_their_payload` runs `Op.layerNorm` (`layerNorm` over the last
axis) and `Op.batchNormRunning` against PyTorch. On the layer norm rows `[1, 1, 1, 5]`,
`[-3, -3, -3, 1]` and `[5, 5, 5, 5]`, and batch norms of integers from `-4` to `4` whose running
variances plus eps are 1, 4 and 16, the outputs agree bit for bit in binary64, binary32, binary16
and bfloat16. Random binary32 data agrees within `1e-5` relative and `1e-6` absolute. None of the
PyTorch tests we found reaches `groupNorm`, `rmsNorm` or `.batch`. The Python importer builds group
norm, and batch norm over batch statistics, from moments.

* `Shape.keptAxis` and `Shape.listedAxis` place the axes of the reduced and the selected shape
  among the axes of `s`, and `Shape.broadcastDims_eraseAxes`, `Shape.broadcastDims_selectAxes`
  and `Shape.broadcastDims_axis` broadcast those shapes and a channel vector back to `s`.
* `var_eq_mean_centered_mul_self` writes the variance over `dims` as the mean of the squared
  deviations from the mean, so `layerNorm` is a composition of means, broadcasts and elementwise
  operations.
* Over the axes from `axis` on, `layerNorm` is layer normalization along the rows of the matrix
  view of `x`, with the same operations in the same order (`layerNorm_axesFrom_eq_reshape`).
-/

@[expose] public section

open TorchLean.Normalization
open FloatLib.Numerics (MathFunctions)
open TorchLean.Shape.Coord (linearize unlinearize unlinearize_linearize)
open TorchLean.Shape (rank rank_eq_length size size_cons)
open TorchLean.Shape (size_eq_prod size_nil)
open TorchLean.Shape.Coord (index_appendEquiv_snd)

namespace TorchLean.Shape

/-- The axis of `s` that is axis `i` of `s.eraseAxes dims`. -/
def keptAxis (s : Shape) (dims : List (Fin s.rank)) (i : Fin (s.eraseAxes dims).rank) :
    Fin s.rank :=
  (s.otherAxes dims)[i.val]'(by simpa [eraseAxes, Shape.rank_eq_length] using i.isLt)

/-- The axis of `s` that is axis `i` of `s.selectAxes dims`. -/
def listedAxis (s : Shape) (dims : List (Fin s.rank)) (i : Fin (s.selectAxes dims).rank) :
    Fin s.rank :=
  (s.sortedAxes dims)[i.val]'(by simpa [selectAxes, Shape.rank_eq_length] using i.isLt)

/-- Axis `i` of `s.eraseAxes dims` has the extent of the axis of `s` it keeps. -/
theorem extent_keptAxis (s : Shape) (dims : List (Fin s.rank))
    (i : Fin (s.eraseAxes dims).rank) :
    (s.eraseAxes dims).extent i = s.extent (s.keptAxis dims i) := by
  simp [extent, eraseAxes, keptAxis]

/-- Axis `i` of `s.selectAxes dims` has the extent of the axis of `s` it lists. -/
theorem extent_listedAxis (s : Shape) (dims : List (Fin s.rank))
    (i : Fin (s.selectAxes dims).rank) :
    (s.selectAxes dims).extent i = s.extent (s.listedAxis dims i) := by
  simp [extent, selectAxes, listedAxis]

/-- A reduction leaves a shape that broadcasts back along the axes it keeps. -/
theorem broadcastDims_eraseAxes (s : Shape) (dims : List (Fin s.rank)) :
    BroadcastDims (s.eraseAxes dims) s (s.keptAxis dims) where
  strictMono _ _ hij :=
    List.pairwise_iff_getElem.mp (s.pairwise_otherAxes dims) _ _ _ _ hij
  extent_eq i := Or.inr (s.extent_keptAxis dims i)

/-- Selecting the axes `dims` of `s` gives a shape that broadcasts back to `s` along them. -/
theorem broadcastDims_selectAxes (s : Shape) (dims : List (Fin s.rank)) :
    BroadcastDims (s.selectAxes dims) s (s.listedAxis dims) where
  strictMono _ _ hij :=
    List.pairwise_iff_getElem.mp (s.pairwise_sortedAxes dims) _ _ _ _ hij
  extent_eq i := Or.inr (s.extent_listedAxis dims i)

/-- Complementing `[d]` twice among the axes of `s` gives `[d]`. -/
theorem otherAxes_otherAxes_singleton (s : Shape) (d : Fin s.rank) :
    s.otherAxes (s.otherAxes [d]) = [d] := by
  have h : s.otherAxes (s.otherAxes [d]) = s.sortedAxes [d] := by
    unfold otherAxes sortedAxes
    apply List.filter_congr
    intro x _
    simp
  rw [h, sortedAxes_singleton]

/-- Reducing over every axis but `d` leaves the extent of `d`. -/
theorem eraseAxes_otherAxes_singleton (s : Shape) (d : Fin s.rank) :
    s.eraseAxes (s.otherAxes [d]) = [s.extent d] := by
  rw [eraseAxes, otherAxes_otherAxes_singleton]
  rfl

/-- Every axis of `n :: s` but the first lists the extents of `s`. -/
theorem selectAxes_otherAxes_zero (n : Nat) (s : Shape) :
    Shape.selectAxes (n :: s) (Shape.otherAxes (n :: s) [⟨0, Nat.succ_pos _⟩]) = s := by
  rw [selectAxes, sortedAxes_otherAxes]
  apply List.ext_getElem
  · rw [List.length_map, length_otherAxes_singleton, Shape.rank_eq_length]
    rfl
  · intro k h₁ h₂
    rw [List.getElem_map]
    have hk : k < (Shape.otherAxes (n :: s) [⟨0, Nat.succ_pos _⟩]).length := by
      simpa only [List.length_map] using h₁
    have := val_getElem_otherAxes_singleton (n :: s) ⟨0, Nat.succ_pos _⟩ hk
    simp only [Nat.not_lt_zero, ite_false] at this
    simp only [extent, this]
    rfl

/-- The axis after the leading axes of `leading ++ [width]` has extent `width`. -/
theorem extent_append_singleton_rank (leading : Shape) (width : Nat) :
    Shape.extent (leading ++ [width]) ⟨leading.rank, by simp⟩ = width := by
  induction leading with
  | scalar => rfl
  | dim n rest ih => exact ih

/-- The entries of `s` split at axis `d`: the axes before it, the axis, and the axes after it. -/
theorem size_eq_take_extent_drop (s : Shape) (d : Fin s.rank) :
    s.size = Shape.size (s.take d.val) * (s.extent d * Shape.size (s.drop (d.val + 1))) := by
  have hd : d.val < s.length := d.isLt.trans_eq s.rank_eq_length
  simp only [Shape.size, extent]
  rw [← List.prod_cons, ← List.drop_eq_getElem_cons hd, List.prod_take_mul_prod_drop]

end TorchLean.Shape

namespace TorchLean

open Shape (Coord)

variable {α : Type} [Storage α] [Context α] {s : Shape}

/-- Layer normalization of `x` over the axes `dims`: each entry less the mean of its fiber, divided
by `scale = normalizationScale` of the centered fiber (gradient stopped), then by
`sqrt (max (v + epsilon / scale / scale) 0)` for the variance `v` of the scaled fiber, then scaled
by `gamma` and shifted by `beta` along `dims`. The mean sums the raw fiber, so a finite fiber
gives NaN when that sum or an entry less the mean overflows, as binary64 `[2^1023, 2^1023]` does. -/
def Tensor.layerNorm (x : Tensor α s) (dims : List (Fin s.rank))
    (gamma beta : Tensor α (s.selectAxes dims)) (epsilon : α := TorchLean.normalizationEpsilon) :
    Tensor α s :=
  let n := Shape.size (s.selectAxes dims)
  let spread := fun t : Tensor α (s.eraseAxes dims) =>
    broadcast t s (s.keptAxis dims) (s.broadcastDims_eraseAxes dims)
  let mean := map (Context.divNat · (if n = 0 then 1 else n)) (reduce .sum x dims false)
  let centered := x - spread mean
  let regularizer := full (s.reduce dims false) epsilon
  let scale := normalizationScale centered dims regularizer
  let scaled := centered / spread scale
  Normalization.normalize scaled (var scaled dims false) ((regularizer / scale) / scale) gamma beta
    (s.broadcastDims_eraseAxes dims) (s.broadcastDims_selectAxes dims)
    (s.broadcastDims_selectAxes dims)

/-- RMS normalization of `x` over the axes `dims`: each fiber is divided by
`scale = maximum (maxAbs fiber) (sqrt |epsilon|)` (IEEE maximum, gradient stopped), then by
`sqrt (m + epsilon / scale / scale)` for the mean `m` of the squared scaled fiber, then multiplied
by `gamma` along `dims`. A large fiber can still overflow in a narrow format, and nonfinite entries
have no real normalization guarantee. -/
def Tensor.rmsNorm (x : Tensor α s) (dims : List (Fin s.rank))
    (gamma : Tensor α (s.selectAxes dims))
    (epsilon : α := TorchLean.normalizationEpsilon) : Tensor α s :=
  let n := Shape.size (s.selectAxes dims)
  let spread := fun t : Tensor α (s.eraseAxes dims) =>
    broadcast t s (s.keptAxis dims) (s.broadcastDims_eraseAxes dims)
  let regularizer := full (s.eraseAxes dims) epsilon
  let largest : Tensor α (s.eraseAxes dims) := Rep.ofFn fun o =>
    maxAbs (ofFn fun i : Fin n => x (s.reduceSource dims false o (Coord.unlinearize i)))
  let scale := map Context.stopGradient (Tensor.maximum largest (Tensor.sqrt (abs regularizer)
      .clampNegative))
  let scaled := x / spread scale
  let meanSquared : Tensor α (s.eraseAxes dims) := map (Context.divNat · (if n = 0 then 1 else n))
    (reduce .sum (square scaled) dims false)
  let denominator := Tensor.sqrt (meanSquared + ((regularizer / scale) / scale)) .clampNegative
  (scaled / spread denominator) *
    broadcast gamma s (s.listedAxis dims) (s.broadcastDims_selectAxes dims)

/-- The statistics that `batchNorm` normalizes with. -/
inductive BatchNorm.Stats (α : Type) [Storage α] (c : Nat)
  /-- The mean and variance of the input over every axis but the channel axis. -/
  | batch
  /-- Supplied running mean and variance per channel. -/
  | running (mean var : Tensor α [c])

/-- Batch normalization of `x` with channels along `dim`, scaled by `gamma` and shifted by `beta`
per channel. With `.batch` each entry less its channel's mean over every other axis is divided by
`scale = normalizationScale` of the centered channel (gradient stopped) and by
`sqrt (max (m + epsilon / scale / scale) 0)` for the mean `m` of its squared scaled entries. The
channel mean sums the raw channel, so a finite channel gives NaN when that sum or an entry less the
mean overflows. With `.running mean var` the entry less `mean` is divided by
`sqrt (max (maximum var 0 + epsilon) 0)`. -/
def Tensor.batchNorm (x : Tensor α s) (dim : Fin s.rank) (gamma beta : Tensor α [s.extent dim])
    (stats : BatchNorm.Stats α (s.extent dim) := .batch)
    (epsilon : α := TorchLean.normalizationEpsilon) : Tensor α s :=
  match stats with
  | .batch =>
      let others := s.otherAxes [dim]
      let n := Shape.size (s.selectAxes others)
      let spreadStats := fun t : Tensor α (s.eraseAxes others) =>
        broadcast t s (s.keptAxis others) (s.broadcastDims_eraseAxes others)
      let mean := map (Context.divNat · (if n = 0 then 1 else n)) (reduce .sum x others false)
      let centered := x - spreadStats mean
      let regularizer := full (s.reduce others false) epsilon
      let scale := normalizationScale centered others regularizer
      let scaled := centered / spreadStats scale
      let meanSquare : Tensor α (s.reduce others false) :=
        map (Context.divNat · (if n = 0 then 1 else n)) (reduce .sum (scaled * scaled) others false)
      Normalization.normalize scaled meanSquare ((regularizer / scale) / scale) gamma beta
        (s.broadcastDims_eraseAxes others) (s.broadcastDims_axis dim) (s.broadcastDims_axis dim)
  | .running mean var =>
      Normalization.normalize (x - broadcast mean s _ (s.broadcastDims_axis dim))
        (Tensor.maximum var (full _ 0)) (full _ epsilon) gamma beta (s.broadcastDims_axis dim)
        (s.broadcastDims_axis dim) (s.broadcastDims_axis dim)

/-- Group normalization of `x` with channels along `dim`, split into `groups` contiguous groups of
`s.extent dim / groups` channels. `x` is viewed as `[size before, groups, group size]` for the axes
before `dim` and the group's channels times the axes after `dim`. Each centered group `c` (its
entries less their mean) is divided by `scale = maximum (maxAbs c) (sqrt |epsilon|)` (IEEE maximum,
gradient stopped), then by `sqrt (max (m + epsilon / scale / scale) 0)` for the mean `m` of the
squared scaled group, then scaled by `gamma` and shifted by `beta` per channel. The mean sums the
raw group, so a finite group gives NaN when that sum or an entry less the mean overflows. With no
axis before `dim` there is one batch item, and with none after it, one position. A group has no
entries only when `x` has none. -/
def Tensor.groupNorm (x : Tensor α s) (dim : Fin s.rank) (groups : Nat)
    (gamma beta : Tensor α [s.extent dim]) (hGroups : groups ∣ s.extent dim)
    (epsilon : α := TorchLean.normalizationEpsilon) : Tensor α s :=
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
  let regularizer := full (grouped.eraseAxes [last]) epsilon
  let largest : Tensor α (grouped.eraseAxes [last]) := Rep.ofFn fun o =>
    maxAbs (ofFn fun i : Fin (Shape.size (grouped.selectAxes [last])) =>
      centered (grouped.reduceSource [last] false o (Coord.unlinearize i)))
  let scale := map Context.stopGradient (Tensor.maximum largest (Tensor.sqrt (abs regularizer)
      .clampNegative))
  let scaled := centered / spread scale
  let meanSquared : Tensor α (grouped.eraseAxes [last]) :=
    map (Context.divNat · groupSize) (reduce .sum (scaled * scaled) [last] false)
  let std := Tensor.sqrt (meanSquared + ((regularizer / scale) / scale)) .clampNegative
  let channel := fun t : Tensor α [s.extent dim] => broadcast t s _ (s.broadcastDims_axis dim)
  (reshape (scaled / spread std) s hSize.symm * channel gamma) + channel beta

/-! ## Rows of a reshaped tensor -/

/-- The index of a vector coordinate has the value of its flat index. -/
theorem Shape.Coord.val_unlinearize_vector {m : Nat} (k : Fin (Shape.size [m])) :
    (Coord.unlinearize (s := [m]) k).1.val = k.val := by
  have h := congrArg Fin.val (Coord.linearize_unlinearize (s := [m]) k)
  rw [← Prod.mk.eta (p := Coord.unlinearize (s := [m]) k),
    Coord.linearize_cons_val] at h
  have h0 := (Coord.linearize (Coord.unlinearize (s := [m]) k).2).isLt
  simp only [Shape.size_nil, Nat.lt_one_iff] at h0
  simp only [h0, Shape.size_nil, Nat.one_mul, Nat.zero_add] at h
  exact h

/-- Two folds over the indices below equal bounds agree when their steps agree at indices with
equal values. -/
theorem Tensor.foldl_finRange_cast {β : Type} {m n : Nat} (h : m = n) (f : β → Fin m → β)
    (g : β → Fin n → β) (hfg : ∀ a i, f a i = g a (Fin.cast h i)) (b : β) :
    (List.finRange m).foldl f b = (List.finRange n).foldl g b := by
  subst h
  have : f = g := funext fun a => funext fun i => hfg a i
  rw [this]

/-- Unlinearizing in equal shapes gives coordinates with the same indices. -/
theorem Shape.Coord.index_unlinearize_congr {t u : Shape} (h : t = u) (k : Fin (Shape.size t))
    (j : Nat) :
    Shape.Coord.index (Coord.unlinearize k) j =
      Shape.Coord.index (Coord.unlinearize (s := u)
        (Fin.cast (congrArg Shape.size h) k)) j := by
  subst h
  rfl

/-- The sum of a function of the entries of row `c` of `x` reshaped to `[n, size sp]` is the
row-major sum over every axis of `x` but the first, at index `c` of the first. -/
theorem Tensor.sum_reshape_row {n : Nat} {sp : Shape} (x : Tensor α (n :: sp))
    (h : Shape.size (n :: sp) = Shape.size [n, Shape.size sp]) (c : Fin n)
    (o : (Shape.reduce (n :: sp) (Shape.otherAxes (n :: sp) [⟨0, Nat.succ_pos _⟩]) false).Coord)
    (ho : Shape.Coord.index o 0 = c.val) (f : α → α) :
    Tensor.sum (ofFn fun i : Fin (Shape.size sp) => f (Tensor.reshape x _ h (c, (i, ())))) =
      (List.finRange (Shape.size
          (Shape.selectAxes (n :: sp) (Shape.otherAxes (n :: sp) [⟨0, Nat.succ_pos _⟩])))).foldl
        (fun acc k => acc + f (x (Shape.reduceSource (n :: sp) _ false o
          (Coord.unlinearize k)))) 0 := by
  have hsel := Shape.selectAxes_otherAxes_zero n sp
  have hsize : Shape.size [Shape.size sp] = Shape.size
      (Shape.selectAxes (n :: sp) (Shape.otherAxes (n :: sp) [⟨0, Nat.succ_pos _⟩])) := by
    rw [hsel, Shape.size_cons, Shape.size_nil, Nat.mul_one]
  rw [Tensor.sum_eq_foldl, Rep.foldl_eq_fin_foldl,
      Fin.foldl_eq_finRange_foldl]
  simp only [Rep.getFlat_eq_get, ofFn, Rep.get_ofFn,
    reshape_apply]
  refine foldl_finRange_cast hsize _ _ (fun a k => ?_) 0
  have hk : k.val < Shape.size sp := by
    have hlt := k.isLt
    have e : Shape.size [Shape.size sp] = Shape.size sp := by
      rw [Shape.size_cons, Shape.size_nil, Nat.mul_one]
    omega
  have hleft : Coord.unlinearize (Fin.cast h.symm (Coord.linearize
      ((c, (Coord.unlinearize k).1, ()) : Shape.Coord [n, Shape.size sp]))) =
      ((c, Coord.unlinearize ⟨k.val, hk⟩) : Shape.Coord (n :: sp)) := by
    rw [← @Coord.unlinearize_linearize (n :: sp) (c, Coord.unlinearize ⟨k.val, hk⟩)]
    congr 1
    apply Fin.ext
    have h0 := (Coord.linearize (s := []) ()).isLt
    simp only [Shape.size_nil, Nat.lt_one_iff] at h0
    have h1 := Coord.linearize_cons_val (s := [Shape.size sp]) c
      ((Coord.unlinearize k).1, ())
    have h2 := Coord.linearize_cons_val (s := []) (Coord.unlinearize k).1 ()
    have h3 := Coord.linearize_cons_val (s := sp) c
      (Coord.unlinearize ⟨k.val, hk⟩)
    have h4 := Coord.val_unlinearize_vector k
    have e1 : Shape.size [Shape.size sp] = Shape.size sp := by
      rw [Shape.size_cons, Shape.size_nil, Nat.mul_one]
    have e2 : Shape.size [] = 1 := rfl
    have hX : Shape.size [Shape.size sp] * c.val = Shape.size sp * c.val := by
      rw [e1]
    have hY : Shape.size [] * (Coord.unlinearize k).1.val =
        (Coord.unlinearize k).1.val := by
      rw [e2, Nat.one_mul]
    simp only [Fin.val_cast]
    rw [Coord.linearize_unlinearize] at h3
    erw [h1, h2, h3]
    rw [h0, hY, hX, h4, Nat.zero_add]
  have hright : Shape.reduceSource (n :: sp) (Shape.otherAxes (n :: sp) [⟨0, Nat.succ_pos _⟩])
      false o (Coord.unlinearize (Fin.cast hsize k)) =
      ((c, Coord.unlinearize ⟨k.val, hk⟩) : Shape.Coord (n :: sp)) := by
    apply Shape.Coord.ext_index
    intro j hj
    rw [Shape.index_reduceSource, dite_eq_left hj]
    cases j with
    | zero =>
        rw [ite_eq_right (by simp only [Shape.mem_otherAxes, List.mem_singleton, not_not]),
          Shape.keptIndex, Bool.cond_false,
          Shape.otherAxes_otherAxes_singleton, List.idxOf_cons_self, ho,
          Shape.Coord.index_zero]
    | succ j =>
        have hne : (⟨j + 1, hj⟩ : Fin (Shape.rank (n :: sp))) ≠ ⟨0, Nat.succ_pos _⟩ := by
          simp [Fin.ext_iff]
        rw [ite_eq_left (Shape.mem_otherAxes.mpr (by simpa only [List.mem_singleton] using hne)),
          Shape.sortedAxes_otherAxes,
          Shape.idxOf_otherAxes_singleton _ hne, Shape.Coord.index_cons_succ,
          Coord.index_unlinearize_congr hsel]
        simp only [Nat.add_sub_cancel]
        rfl
  rw [hleft, hright]

end TorchLean

/-! ## Layer normalization over a suffix as rows of a matrix -/

namespace TorchLean.Shape

/-- The axes of `s` from `axis` on, in increasing order: the normalized axes of PyTorch's
`layer_norm` with `normalized_shape = s[axis:]`. -/
def axesFrom (s : Shape) (axis : Nat) : List (Fin s.rank) :=
  (List.finRange s.rank).filter (axis ≤ ·.val)

/-- An axis is among the axes from `axis` on exactly when it is at least `axis`. -/
@[simp] theorem mem_axesFrom {s : Shape} {axis : Nat} {d : Fin s.rank} :
    d ∈ s.axesFrom axis ↔ axis ≤ d.val := by
  simp [axesFrom]

/-- The naturals below `r` that are at least `axis` are `axis, ..., r - 1`. -/
private theorem filter_range_le {axis r : Nat} (h : axis ≤ r) :
    (List.range r).filter (fun k => decide (axis ≤ k)) = List.range' axis (r - axis) := by
  have hsplit : List.range r = List.range' 0 axis ++ List.range' axis (r - axis) := by
    have happ := List.range'_append_1 (s := 0) (m := axis) (n := r - axis)
    rw [Nat.zero_add, Nat.add_sub_cancel' h] at happ
    rw [List.range_eq_range', happ]
  rw [hsplit, List.filter_append, List.filter_eq_nil_iff.mpr, List.filter_eq_self.mpr,
    List.nil_append]
  · intro k hk
    simp only [List.mem_range'_1] at hk
    simpa using hk.1
  · intro k hk
    simp only [List.mem_range'_1] at hk
    simp only [decide_eq_true_eq]
    omega

/-- The naturals below `r` that are below `axis` are the naturals below `axis`. -/
private theorem filter_range_lt {axis r : Nat} (h : axis ≤ r) :
    (List.range r).filter (fun k => !decide (axis ≤ k)) = List.range axis := by
  have hsplit : List.range r = List.range' 0 axis ++ List.range' axis (r - axis) := by
    have happ := List.range'_append_1 (s := 0) (m := axis) (n := r - axis)
    rw [Nat.zero_add, Nat.add_sub_cancel' h] at happ
    rw [List.range_eq_range', happ]
  rw [hsplit, List.filter_append, List.filter_eq_self.mpr, List.filter_eq_nil_iff.mpr,
    List.append_nil, List.range_eq_range']
  · intro k hk
    simp only [List.mem_range'_1] at hk
    simp only [Bool.not_eq_true', decide_eq_false_iff_not, Nat.not_le]
    omega
  · intro k hk
    simp only [List.mem_range'_1] at hk
    simp only [Bool.not_eq_eq_eq_not, Bool.not_true, decide_eq_false_iff_not, Nat.not_le]
    omega

/-- The axes from `axis` on are `axis, axis + 1, ...`. -/
theorem map_val_axesFrom {s : Shape} {axis : Nat} (h : axis ≤ s.rank) :
    (s.axesFrom axis).map Fin.val = List.range' axis (s.rank - axis) := by
  show List.map Fin.val (List.filter ((fun k => decide (axis ≤ k)) ∘ Fin.val) _) = _
  rw [← List.filter_map, List.map_coe_finRange_eq_range, filter_range_le h]

/-- The axes other than those from `axis` on are `0, 1, ..., axis - 1`, in order. -/
theorem map_val_otherAxes_axesFrom {s : Shape} {axis : Nat} (h : axis ≤ s.rank) :
    (s.otherAxes (s.axesFrom axis)).map Fin.val = List.range axis := by
  have hp : (fun d : Fin s.rank => decide (d ∉ s.axesFrom axis)) =
      (fun k => !decide (axis ≤ k)) ∘ Fin.val := by
    funext d
    simp only [mem_axesFrom, Function.comp_apply, decide_not]
  rw [otherAxes, hp, ← List.filter_map, List.map_coe_finRange_eq_range, filter_range_lt h]


/-- There are `s.rank - axis` axes from `axis` on. -/
theorem length_axesFrom {s : Shape} {axis : Nat} (h : axis ≤ s.rank) :
    (s.axesFrom axis).length = s.rank - axis := by
  rw [← List.length_map (f := Fin.val), map_val_axesFrom h, List.length_range']

/-- Position `k` of the axes from `axis` on is axis `axis + k`. -/
theorem val_getElem_axesFrom {s : Shape} {axis : Nat} (h : axis ≤ s.rank) {k : Nat}
    (hk : k < (s.axesFrom axis).length) : ((s.axesFrom axis)[k]).val = axis + k := by
  have hk' : k < s.rank - axis := length_axesFrom h ▸ hk
  have e := congrArg (fun l : List Nat => l[k]?) (map_val_axesFrom h)
  simpa [List.getElem?_eq_getElem hk, hk'] using e

/-- Position `k` of the axes before `axis` is axis `k`. -/
theorem val_getElem_otherAxes_axesFrom {s : Shape} {axis : Nat} (h : axis ≤ s.rank) {k : Nat}
    (hk : k < (s.otherAxes (s.axesFrom axis)).length) :
    ((s.otherAxes (s.axesFrom axis))[k]).val = k := by
  have hk' : k < axis := by
    simpa [← List.length_map (f := Fin.val), map_val_otherAxes_axesFrom h] using hk
  have e := congrArg (fun l : List Nat => l[k]?) (map_val_otherAxes_axesFrom h)
  simpa [List.getElem?_eq_getElem hk, hk'] using e

/-- Sorting leaves the axes from `axis` on unchanged. -/
@[simp] theorem sortedAxes_axesFrom (s : Shape) (axis : Nat) :
    s.sortedAxes (s.axesFrom axis) = s.axesFrom axis := by
  unfold sortedAxes axesFrom
  apply List.filter_congr
  intro d _
  simp

/-- The axes from `axis` on select the suffix `s.drop axis`. -/
theorem selectAxes_axesFrom {s : Shape} {axis : Nat} (h : axis ≤ s.rank) :
    s.selectAxes (s.axesFrom axis) = s.drop axis := by
  apply List.ext_getElem
  · rw [selectAxes, sortedAxes_axesFrom, List.length_map, length_axesFrom h, List.length_drop,
      Shape.rank_eq_length]
  · intro k h₁ h₂
    have hk : k < (s.axesFrom axis).length := by simpa [selectAxes] using h₁
    simp only [selectAxes, sortedAxes_axesFrom, List.getElem_map, List.getElem_drop, extent,
      val_getElem_axesFrom h hk]

/-- Erasing the axes from `axis` on leaves the prefix `s.take axis`. -/
theorem eraseAxes_axesFrom {s : Shape} {axis : Nat} (h : axis ≤ s.rank) :
    s.eraseAxes (s.axesFrom axis) = s.take axis := by
  apply List.ext_getElem
  · rw [eraseAxes, List.length_map, ← List.length_map (f := Fin.val),
      map_val_otherAxes_axesFrom h, List.length_range, List.length_take, ← Shape.rank_eq_length]
    omega
  · intro k h₁ h₂
    have hk : k < (s.otherAxes (s.axesFrom axis)).length := by simpa [eraseAxes] using h₁
    simp only [eraseAxes, List.getElem_map, List.getElem_take, extent,
      val_getElem_otherAxes_axesFrom h hk]

/-- A shape has as many entries as its matrix view `[(s.take axis).prod, (s.drop axis).prod]`. -/
theorem size_eq_size_take_drop (s : Shape) (axis : Nat) :
    s.size = Shape.size [(s.take axis).prod, (s.drop axis).prod] := by
  rw [Shape.size_eq_prod, Shape.size_eq_prod]
  simp only [List.prod_cons, List.prod_nil, Nat.mul_one]
  rw [← List.prod_append, List.take_append_drop]

/-- The axes from `axis` on select as many entries as one row of that matrix view. -/
theorem size_selectAxes_axesFrom {s : Shape} {axis : Nat} (h : axis ≤ s.rank) :
    (s.selectAxes (s.axesFrom axis)).size = Shape.size [(s.drop axis).prod] := by
  rw [selectAxes_axesFrom h, Shape.size_eq_prod, Shape.size_eq_prod]
  simp

/-- The last axis of `lead ++ [n]` is the one axis from `lead.length` on. -/
theorem eq_axesFrom_last {lead : Shape} {n : Nat} {ds : List (Fin (lead ++ [n]).rank)}
    (hds : ds.map Fin.val = [lead.length]) : ds = (lead ++ [n]).axesFrom lead.length := by
  apply List.map_injective_iff.mpr Fin.val_injective
  rw [hds, map_val_axesFrom (by simp [Shape.rank_eq_length])]
  simp [Shape.rank_eq_length]

/-- The last axis of a shape `lead ++ [n]` selects `[n]`. -/
theorem selectAxes_last {s lead : Shape} {n : Nat} (hs : s = lead ++ [n])
    {ds : List (Fin s.rank)} (hds : ds.map Fin.val = [lead.length]) : s.selectAxes ds = [n] := by
  subst hs
  rw [eq_axesFrom_last hds, selectAxes_axesFrom (by simp [Shape.rank_eq_length])]
  simp

end TorchLean.Shape
namespace TorchLean.Tensor

open Shape.Coord (index_appendEquiv_fst)

variable {α : Type} [Storage α] {s : Shape}

/-- The coordinate of the reduced shape that `c` lies over. -/
private def keptCoord (dims : List (Fin s.rank)) (c : s.Coord) : (s.reduce dims false).Coord :=
  Shape.Coord.ofFun (s := s.eraseAxes dims) fun i => Fin.cast (s.extent_keptAxis dims i).symm
    (Shape.Coord.get c (s.keptAxis dims i))

/-- The coordinate of the selected shape that `c` lies over. -/
private def listedCoord (dims : List (Fin s.rank)) (c : s.Coord) : (s.selectAxes dims).Coord :=
  Shape.Coord.ofFun fun i => Fin.cast (s.extent_listedAxis dims i).symm
    (Shape.Coord.get c (s.listedAxis dims i))

/-- The indices of `keptCoord dims c` are those of `c` along the axes it keeps. -/
private theorem index_keptCoord (dims : List (Fin s.rank)) (c : s.Coord) (k : Nat) :
    Shape.Coord.index (keptCoord dims c) k =
      if h : k < (s.eraseAxes dims).rank then
        Shape.Coord.index c (s.keptAxis dims ⟨k, h⟩).val
      else 0 := by
  show Shape.Coord.index (s := s.eraseAxes dims) (Shape.Coord.ofFun _) k = _
  rw [Shape.Coord.index_ofFun]
  split_ifs
  · simp [Shape.Coord.val_get]
  · rfl

/-- The indices of `listedCoord dims c` are those of `c` along the listed axes. -/
private theorem index_listedCoord (dims : List (Fin s.rank)) (c : s.Coord) (k : Nat) :
    Shape.Coord.index (listedCoord dims c) k =
      if h : k < (s.selectAxes dims).rank then
        Shape.Coord.index c (s.listedAxis dims ⟨k, h⟩).val
      else 0 := by
  rw [listedCoord, Shape.Coord.index_ofFun]
  split_ifs
  · simp [Shape.Coord.val_get]
  · rfl

/-- A coordinate read by a reduction lies over the output coordinate it is read for. -/
private theorem keptCoord_reduceSource (dims : List (Fin s.rank))
    (o : (s.reduce dims false).Coord) (r : (s.selectAxes dims).Coord) :
    keptCoord dims (s.reduceSource dims false o r) = o := by
  apply Shape.Coord.ext_index
  intro k hk
  have hk' : k < (s.eraseAxes dims).rank := hk
  have hlen : k < (s.otherAxes dims).length := by
    simpa [Shape.eraseAxes, Shape.rank_eq_length] using hk'
  have hnot : (s.otherAxes dims)[k] ∉ dims :=
    Shape.mem_otherAxes.mp (List.getElem_mem hlen)
  rw [index_keptCoord, dite_eq_left hk', Shape.index_reduceSource, dite_eq_left (Fin.isLt _)]
  simp only [Shape.keptAxis, Fin.eta, hnot, ↓reduceIte, Shape.keptIndex, Bool.cond_false,
    (s.nodup_otherAxes dims).idxOf_getElem k hlen]

/-- A coordinate read by a reduction lies over the reduced coordinate it is read at. -/
private theorem listedCoord_reduceSource (dims : List (Fin s.rank))
    (o : (s.reduce dims false).Coord) (r : (s.selectAxes dims).Coord) :
    listedCoord dims (s.reduceSource dims false o r) = r := by
  apply Shape.Coord.ext_index
  intro k hk
  have hlen : k < (s.sortedAxes dims).length := by
    simpa [Shape.selectAxes, Shape.rank_eq_length] using hk
  have hmem : (s.sortedAxes dims)[k] ∈ dims :=
    Shape.mem_sortedAxes.mp (List.getElem_mem hlen)
  have hnd : (s.sortedAxes dims).Nodup := (List.nodup_finRange s.rank).filter _
  rw [index_listedCoord, dite_eq_left hk, Shape.index_reduceSource, dite_eq_left (Fin.isLt _)]
  simp only [Shape.listedAxis, Fin.eta, hmem, ↓reduceIte, hnd.idxOf_getElem k hlen]


/-- A coordinate is read by a reduction at the output and reduced coordinates it lies over. -/
private theorem reduceSource_keptCoord_listedCoord (dims : List (Fin s.rank)) (c : s.Coord) :
    s.reduceSource dims false (keptCoord dims c) (listedCoord dims c) = c := by
  apply Shape.Coord.ext_index
  intro k hk
  rw [Shape.index_reduceSource, dite_eq_left hk]
  split_ifs with hd
  · have hmem : (⟨k, hk⟩ : Fin s.rank) ∈ s.sortedAxes dims := Shape.mem_sortedAxes.mpr hd
    have hlt : (s.sortedAxes dims).idxOf ⟨k, hk⟩ < (s.selectAxes dims).rank := by
      simpa [Shape.selectAxes, Shape.rank_eq_length] using List.idxOf_lt_length_iff.mpr hmem
    rw [index_listedCoord, dite_eq_left hlt]
    simp only [Shape.listedAxis, List.getElem_idxOf]
  · have hmem : (⟨k, hk⟩ : Fin s.rank) ∈ s.otherAxes dims := Shape.mem_otherAxes.mpr hd
    have hlt : (s.otherAxes dims).idxOf ⟨k, hk⟩ < (s.eraseAxes dims).rank := by
      simpa [Shape.eraseAxes, Shape.rank_eq_length] using List.idxOf_lt_length_iff.mpr hmem
    simp only [Shape.keptIndex, Bool.cond_false]
    rw [index_keptCoord, dite_eq_left hlt]
    simp only [Shape.keptAxis, List.getElem_idxOf]

/-- An entry of a sum over `dims` is the sum of the entries lying over it. -/
private theorem reduce_sum_apply_eq_sum_ite [AddCommMonoid α] (w : Tensor α s)
    (dims : List (Fin s.rank)) (o : (s.reduce dims false).Coord) :
    (Tensor.reduce .sum w dims false) o = ∑ c, if keptCoord dims c = o then w c else 0 := by
  have hfold : ∀ {n : Nat} (f : Fin n → α),
      (List.finRange n).foldl (fun acc i => acc + f i) 0 = ∑ i, f i := fun f => by
    rw [Fin.sum_univ_def, List.sum_eq_foldl, List.foldl_map]
  rw [reduce_sum_apply, hfold]
  refine ((Shape.Coord.equivFin _).symm.sum_comp
    (fun r => w (s.reduceSource dims false o r))).trans ?_
  rw [← Finset.sum_filter]
  refine Finset.sum_nbij' (fun r => s.reduceSource dims false o r) (listedCoord dims)
    ?_ ?_ ?_ ?_ ?_
  · intro r _
    exact Finset.mem_filter.mpr ⟨Finset.mem_univ _, keptCoord_reduceSource dims o r⟩
  · intro c _
    exact Finset.mem_univ _
  · intro r _
    exact listedCoord_reduceSource dims o r
  · intro c hc
    rw [← (Finset.mem_filter.mp hc).2]
    exact reduceSource_keptCoord_listedCoord dims c
  · intro r _
    rfl

/-- Every axis but `dim` leaves the single axis `dim`. -/
private theorem keptAxis_otherAxes_singleton (dim : Fin s.rank)
    (i : Fin (s.eraseAxes (s.otherAxes [dim])).rank) :
    s.keptAxis (s.otherAxes [dim]) i = dim := by
  rw [Shape.keptAxis, List.getElem_of_eq (Shape.otherAxes_otherAxes_singleton s dim)]
  simp

/-- In a sum over every axis but `dim`, a coordinate lies over an output coordinate exactly when
their indices along `dim` agree. -/
private theorem keptCoord_otherAxes_eq_iff (dim : Fin s.rank)
    (h : s.reduce (s.otherAxes [dim]) false = [s.extent dim]) (c : s.Coord)
    (o : Shape.Coord [s.extent dim]) :
    keptCoord (s.otherAxes [dim]) c = _root_.cast (congrArg Shape.Coord h.symm) o ↔
      Shape.Coord.get c dim = o.1 := by
  have hrank : (s.eraseAxes (s.otherAxes [dim])).rank = 1 := by
    rw [Shape.eraseAxes_otherAxes_singleton]
    rfl
  have h0 : 0 < (s.eraseAxes (s.otherAxes [dim])).rank := by omega
  have hidx : Shape.Coord.index (keptCoord (s.otherAxes [dim]) c) 0 =
      (Shape.Coord.get c dim).val := by
    rw [index_keptCoord, dite_eq_left h0, keptAxis_otherAxes_singleton, Shape.Coord.val_get]
  constructor
  · intro e
    apply Fin.ext
    have e := congrArg (fun z => Shape.Coord.index z 0) e
    simp only [hidx, Shape.Coord.index_cast h.symm] at e
    exact e
  · intro e
    apply Shape.Coord.ext_index
    intro k hk
    have hk0 : k = 0 := by
      change k < (s.eraseAxes (s.otherAxes [dim])).rank at hk
      omega
    subst hk0
    rw [hidx, Shape.Coord.index_cast h.symm, e]
    rfl

/-- An entry of the sum over every axis but `dim` is the sum of the entries whose index along
`dim` is that entry's index. -/
theorem reduce_sum_otherAxes_apply [AddCommMonoid α] (dim : Fin s.rank)
    (w : Tensor α s) (o : Shape.Coord [s.extent dim]) :
    castShape (Tensor.reduce .sum w (s.otherAxes [dim]) false)
        (Shape.eraseAxes_otherAxes_singleton s dim) o =
      ∑ c, if Shape.Coord.get c dim = o.1 then w c else 0 := by
  have h : s.reduce (s.otherAxes [dim]) false = [s.extent dim] :=
    Shape.eraseAxes_otherAxes_singleton s dim
  show castShape (Tensor.reduce .sum w (s.otherAxes [dim]) false) h o = _
  rw [castShape_apply, reduce_sum_apply_eq_sum_ite]
  exact Finset.sum_congr rfl fun c _ => if_congr (keptCoord_otherAxes_eq_iff dim h c o) rfl rfl

/-- Broadcasting a reduced tensor back reads it at the coordinate `keptCoord dims c` that `c` lies
over. -/
private theorem broadcast_keptAxis_apply (dims : List (Fin s.rank))
    (y : Tensor α (s.reduce dims false)) (c : s.Coord) :
    broadcast (s := s.eraseAxes dims) y s (s.keptAxis dims) (s.broadcastDims_eraseAxes dims) c =
      y (keptCoord dims c) :=
  broadcast_apply_of_extent_eq (s := s.eraseAxes dims) y _ _ (s.extent_keptAxis dims) c

/-- Broadcasting a tensor on the listed axes reads it at the listed coordinate. -/
private theorem broadcast_listedAxis_apply (dims : List (Fin s.rank))
    (y : Tensor α (s.selectAxes dims)) (c : s.Coord) :
    broadcast y s (s.listedAxis dims) (s.broadcastDims_selectAxes dims) c =
      y (listedCoord dims c) :=
  broadcast_apply_of_extent_eq y _ _ (s.extent_listedAxis dims) c


/-- Coordinates of equal shapes with the same indices have the same row-major index. -/
private theorem linearize_val_congr {t u : Shape} (h : t = u) {c : t.Coord} {c' : u.Coord}
    (hc : ∀ k, Shape.Coord.index c k = Shape.Coord.index c' k) :
    (Shape.Coord.linearize c).val = (Shape.Coord.linearize c').val := by
  subst h
  rw [Shape.Coord.ext_index fun k _ => hc k]

/-- Listed axis `i` of the axes from `axis` on is axis `axis + i`. -/
private theorem val_listedAxis_axesFrom {axis : Nat} (h : axis ≤ s.rank)
    (i : Fin (s.selectAxes (s.axesFrom axis)).rank) :
    (s.listedAxis (s.axesFrom axis) i).val = axis + i.val := by
  have hi : i.val < (s.axesFrom axis).length := by
    have := i.isLt
    simpa [Shape.selectAxes, Shape.rank_eq_length] using this
  rw [Shape.listedAxis, List.getElem_of_eq (Shape.sortedAxes_axesFrom s axis)]
  exact Shape.val_getElem_axesFrom h hi

/-- Kept axis `i` of the axes from `axis` on is axis `i`. -/
private theorem val_keptAxis_axesFrom {axis : Nat} (h : axis ≤ s.rank)
    (i : Fin (s.eraseAxes (s.axesFrom axis)).rank) :
    (s.keptAxis (s.axesFrom axis) i).val = i.val := by
  have hi : i.val < (s.otherAxes (s.axesFrom axis)).length := by
    have := i.isLt
    simpa [Shape.eraseAxes, Shape.rank_eq_length] using this
  exact Shape.val_getElem_otherAxes_axesFrom h hi

/-- The row-major index of `c` splits into the index of its normalized suffix and that of the
prefix it lies over. -/
private theorem linearize_val_split {axis : Nat} (h : axis ≤ s.rank) (c : s.Coord) :
    (Shape.Coord.linearize c).val =
      (Shape.Coord.linearize (listedCoord (s.axesFrom axis) c)).val +
        Shape.size (s.selectAxes (s.axesFrom axis)) *
          (Shape.Coord.linearize (keptCoord (s.axesFrom axis) c)).val := by
  have hs : s.take axis ++ s.drop axis = s := List.take_append_drop axis s
  have hrank : s.rank = s.length := Shape.rank_eq_length s
  have htake : (s.take axis).length = axis := by
    rw [List.length_take]
    omega
  set c' := Shape.Coord.reindexEq hs.symm c with hc'
  have h1 : (Shape.Coord.linearize c).val = (Shape.Coord.linearize c').val :=
    linearize_val_congr hs.symm fun k => (Shape.Coord.index_reindexEq _ _ _).symm
  have h2 : c' = (Shape.Coord.appendEquiv (s.take axis) (s.drop axis)).symm
      (Shape.Coord.appendEquiv (s.take axis) (s.drop axis) c') :=
    (Equiv.symm_apply_apply _ _).symm
  have hq : (Shape.Coord.linearize (listedCoord (s.axesFrom axis) c)).val =
      (Shape.Coord.linearize
        (Shape.Coord.appendEquiv (s.take axis) (s.drop axis) c').2).val := by
    refine linearize_val_congr (Shape.selectAxes_axesFrom h) fun k => ?_
    rw [index_listedCoord, index_appendEquiv_snd, hc', Shape.Coord.index_reindexEq,
      htake]
    split_ifs with hk
    · rw [val_listedAxis_axesFrom h, Nat.add_comm]
    · have hr : (s.selectAxes (s.axesFrom axis)).rank = s.rank - axis := by
        rw [Shape.selectAxes_axesFrom h, Shape.rank_eq_length, List.length_drop, hrank]
      rw [Shape.Coord.index_of_rank_le]
      omega
  have hp : (Shape.Coord.linearize (keptCoord (s.axesFrom axis) c)).val =
      (Shape.Coord.linearize
        (Shape.Coord.appendEquiv (s.take axis) (s.drop axis) c').1).val := by
    refine linearize_val_congr (t := s.reduce (s.axesFrom axis) false)
      (Shape.eraseAxes_axesFrom h) fun k => ?_
    rw [index_keptCoord]
    split_ifs with hk
    · rw [val_keptAxis_axesFrom h]
      have hk' : k < (s.take axis).length := by
        have hr : (s.eraseAxes (s.axesFrom axis)).rank = (s.take axis).length := by
          rw [Shape.eraseAxes_axesFrom h, Shape.rank_eq_length]
        omega
      rw [index_appendEquiv_fst _ _ _ _ hk', hc', Shape.Coord.index_reindexEq]
    · have hr : (s.eraseAxes (s.axesFrom axis)).rank = (s.take axis).length := by
        rw [Shape.eraseAxes_axesFrom h, Shape.rank_eq_length]
      rw [Shape.Coord.index_of_rank_le]
      rw [Shape.rank_eq_length]
      omega
  rw [h1, h2, Shape.Coord.linearize_appendEquiv_symm_val, hq, hp,
    Shape.selectAxes_axesFrom h]

/-- A reshape reads the coordinate with the same row-major index. -/
private theorem reshape_apply_of_val {t : Shape} (y : Tensor α s) (h : s.size = t.size)
    (i : t.Coord) (c : s.Coord)
    (hv : (Shape.Coord.linearize i).val = (Shape.Coord.linearize c).val) :
    reshape y t h i = y c := by
  rw [reshape_apply]
  congr 1
  apply Shape.Coord.linearize_injective
  rw [Shape.Coord.linearize_unlinearize]
  exact Fin.ext (by simp [hv])

/-- The axes of a matrix from axis one on are its column axis. -/
private theorem axesFrom_pair (m n : Nat) :
    Shape.axesFrom [m, n] 1 = [⟨1, Nat.one_lt_two⟩] :=
  rfl

/-- Reshaping a cast tensor is reshaping the tensor. -/
theorem reshape_castShape {source target t : Shape} (g : Tensor α source) (h : source = target)
    (hs : target.size = t.size) : reshape (castShape g h) t hs = reshape g t (h ▸ hs) := by
  cases h
  rfl

variable [Context α]

/-- Over `dims`, the variance is the mean of the squared deviations from the mean, the mean
broadcast back along the kept axes: the same sums and divisions in the same order. -/
theorem var_eq_mean_centered_mul_self (x : Tensor α s) (dims : List (Fin s.rank)) :
    var x dims false =
      mean ((x - broadcast (s := s.eraseAxes dims) (mean x dims false) s (s.keptAxis dims)
          (s.broadcastDims_eraseAxes dims)) *
        (x - broadcast (s := s.eraseAxes dims) (mean x dims false) s (s.keptAxis dims)
          (s.broadcastDims_eraseAxes dims))) dims false := by
  apply Rep.ext
  intro o
  simp only [var_apply, mean, Tensor.map_apply, reduce_sum_apply, mul_def, sub_def,
    zipWith_apply, broadcast_keptAxis_apply, keptCoord_reduceSource]
  rfl

/-- An entry of a detached tensor is the entry with its differentiation metadata cleared. -/
theorem detach_apply (x : Tensor α s) (c : s.Coord) : detach x c = Context.stopGradient (x c) := by
  unfold detach Context.stopGradient
  rcases (Context.stopGradient? (α := α)) with _ | clear
  · rfl
  · simp only [Tensor.map_apply]

/-- One entry of a layer normalization: `xc` less the mean of `value`, divided by the scale of the
centered fiber and by the root of the scaled fiber's variance plus the scaled `epsilon`, then
scaled by `g` and shifted by `b`. -/
private def normalizedEntry (n : Nat) (value : Fin n → α) (xc g b epsilon : α) : α :=
  let mean := Context.divNat ((List.finRange n).foldl (fun acc i => acc + value i) 0)
    (if n = 0 then 1 else n)
  let largest := if h : 0 < n then
      (List.finRange n).foldl
        (fun acc i => Context.maximum acc (MathFunctions.abs (value i - mean)))
        (MathFunctions.abs (value ⟨0, h⟩ - mean))
    else 0
  let scale := Context.stopGradient
    (Context.maximum largest (MathFunctions.sqrt (Max.max (MathFunctions.abs epsilon) 0)))
  let scaledMean := Context.divNat
    ((List.finRange n).foldl (fun acc i => acc + (value i - mean) / scale) 0)
    (if n = 0 then 1 else n)
  let variance := Context.divNat ((List.finRange n).foldl (fun acc i =>
      let difference := (value i - mean) / scale - scaledMean
      acc + difference * difference) 0)
    (if n = 0 then 1 else n - 0)
  (xc - mean) / scale / MathFunctions.sqrt (Max.max (variance + epsilon / scale / scale) 0) * g + b

/-- An entry of `layerNorm` is `normalizedEntry` of its fiber, entry and parameters. -/
private theorem layerNorm_apply_entry (x : Tensor α s) (dims : List (Fin s.rank))
    (gamma beta : Tensor α (s.selectAxes dims)) (epsilon : α) (c : s.Coord) :
    layerNorm x dims gamma beta epsilon c =
      normalizedEntry (Shape.size (s.selectAxes dims))
        (fun i => x (s.reduceSource dims false (keptCoord dims c) (Shape.Coord.unlinearize i)))
        (x c) (gamma (listedCoord dims c)) (beta (listedCoord dims c)) epsilon := by
  by_cases hpos : ∀ d ∈ dims, 0 < s.extent d
  · -- `Normalization.normalize` broadcasts the root from the statistics' shape
    -- `s.reduce dims false`.
    have hstd (y : Tensor α (s.reduce dims false)) :
        broadcast (s := s.reduce dims false) y s (s.keptAxis dims)
          (s.broadcastDims_eraseAxes dims) c = y (keptCoord dims c) :=
      broadcast_keptAxis_apply dims y c
    -- Rewriting under `Normalization.normalize` before unfolding it makes simp declare its
    -- congruence lemma in this module, so `#roundoff_coverage` in the importing modules does not
    -- declare it.
    simp only [layerNorm, normalizationScale, dite_eq_left hpos]
    simp only [Normalization.normalize, add_def, mul_def, div_def, sub_def, Tensor.maximum,
      Tensor.sqrt, zipWith_apply]
    rw [broadcast_listedAxis_apply, broadcast_listedAxis_apply]
    simp only [hstd, broadcast_keptAxis_apply, Tensor.map_apply, zipWith_apply,
      full_apply, var_apply, reduce_sum_apply, reduce_max_apply, detach_apply, Tensor.abs,
      keptCoord_reduceSource, normalizedEntry, dite_eq_left (Shape.size_selectAxes_pos hpos)]
  · exfalso
    push Not at hpos
    obtain ⟨d, -, hd⟩ := hpos
    exact absurd (Shape.Coord.get c d).isLt (by omega)

omit [Storage α] in
/-- `normalizedEntry` depends only on the values it reads. -/
private theorem normalizedEntry_congr {m n : Nat} (h : m = n) {v : Fin m → α} {w : Fin n → α}
    (hv : ∀ i, v i = w (Fin.cast h i)) {xc xc' g g' b b' : α} (hxc : xc = xc') (hg : g = g')
    (hb : b = b') (epsilon : α) :
    normalizedEntry m v xc g b epsilon = normalizedEntry n w xc' g' b' epsilon := by
  subst h hxc hg hb
  have : v = w := funext fun i => (hv i).trans (by rw [Fin.cast_eq_self])
  rw [this]

/-- Layer normalization over the axes from `axis` on is layer normalization along the rows of the
matrix view `[(s.take axis).prod, (s.drop axis).prod]` of `x`, reshaped back to `s`. Both compute
the same operations in the same order. -/
theorem layerNorm_axesFrom_eq_reshape (x : Tensor α s) {axis : Nat} (haxis : axis ≤ s.rank)
    (gamma beta : Tensor α (s.selectAxes (s.axesFrom axis))) (epsilon : α)
    (hx : s.size = Shape.size [(s.take axis).prod, (s.drop axis).prod])
    (hg : Shape.size (s.selectAxes (s.axesFrom axis)) = Shape.size [(s.drop axis).prod]) :
    layerNorm x (s.axesFrom axis) gamma beta epsilon =
      reshape (layerNorm (reshape x _ hx) [⟨1, Nat.one_lt_two⟩]
          (reshape gamma [(s.drop axis).prod] hg) (reshape beta [(s.drop axis).prod] hg)
          epsilon)
        s hx.symm := by
  apply Rep.ext
  intro c
  rw [layerNorm_apply_entry, reshape_apply]
  refine Eq.trans ?_ (layerNorm_apply_entry (s := [(s.take axis).prod, (s.drop axis).prod]) _
    [⟨1, Nat.one_lt_two⟩] _ _ epsilon _).symm
  generalize hc2 : Shape.Coord.unlinearize
    (Fin.cast hx (Shape.Coord.linearize c)) = c2
  have hK2 : (Shape.Coord.linearize c2).val = (Shape.Coord.linearize c).val := by
    rw [← hc2, Shape.Coord.linearize_unlinearize]
    rfl
  have hn : Shape.size (s.selectAxes (s.axesFrom axis)) =
      Shape.size (Shape.selectAxes [(s.take axis).prod, (s.drop axis).prod]
        [⟨1, Nat.one_lt_two⟩]) := by
    rw [Shape.selectAxes_axesFrom haxis, Shape.selectAxes_singleton]
    simp [Shape.size, Shape.extent]
  have F2 : ∀ c' : Shape.Coord [(s.take axis).prod, (s.drop axis).prod],
      (Shape.Coord.linearize c').val =
        (Shape.Coord.linearize (listedCoord (s := [(s.take axis).prod, (s.drop axis).prod])
          [⟨1, Nat.one_lt_two⟩] c')).val +
          Shape.size (Shape.selectAxes [(s.take axis).prod, (s.drop axis).prod]
            [⟨1, Nat.one_lt_two⟩]) *
            (Shape.Coord.linearize (keptCoord (s := [(s.take axis).prod, (s.drop axis).prod])
              [⟨1, Nat.one_lt_two⟩] c')).val := by
    intro c'
    have := linearize_val_split (s := [(s.take axis).prod, (s.drop axis).prod]) (axis := 1)
      (by simp [Shape.rank_eq_length]) c'
    rwa [axesFrom_pair] at this
  have hlt1 := (Shape.Coord.linearize (listedCoord (s.axesFrom axis) c)).isLt
  obtain ⟨hk1, hl1⟩ :=
    (Nat.div_mod_unique (Nat.zero_lt_of_lt hlt1)).2 ⟨(linearize_val_split haxis c).symm, hlt1⟩
  have F2c := F2 c2
  rw [hK2] at F2c
  obtain ⟨hk2, hl2⟩ := (Nat.div_mod_unique (hn ▸ Nat.zero_lt_of_lt hlt1)).2
    ⟨F2c.symm, (Shape.Coord.linearize _).isLt⟩
  refine normalizedEntry_congr hn (fun i => ?_) (reshape_apply_of_val x hx c2 c hK2).symm
    (reshape_apply_of_val gamma hg _ _ ?_).symm (reshape_apply_of_val beta hg _ _ ?_).symm epsilon
  · symm
    apply reshape_apply_of_val
    have e1 := linearize_val_split haxis
      (s.reduceSource (s.axesFrom axis) false (keptCoord (s.axesFrom axis) c)
        (Shape.Coord.unlinearize i))
    rw [listedCoord_reduceSource, keptCoord_reduceSource, Shape.Coord.linearize_unlinearize]
      at e1
    have e2 := F2 (Shape.reduceSource [(s.take axis).prod, (s.drop axis).prod]
      [⟨1, Nat.one_lt_two⟩] false
      (keptCoord (s := [(s.take axis).prod, (s.drop axis).prod]) [⟨1, Nat.one_lt_two⟩] c2)
      (Shape.Coord.unlinearize (Fin.cast hn i)))
    rw [listedCoord_reduceSource, keptCoord_reduceSource, Shape.Coord.linearize_unlinearize]
      at e2
    rw [e1, e2, ← hk1, ← hk2, Fin.val_cast, ← hn]
  · exact hl2.symm.trans (hn ▸ hl1)
  · exact hl2.symm.trans (hn ▸ hl1)

/-- `layerNorm_axesFrom_eq_reshape` with the affine parameters given on the suffix shape
`s.drop axis`, as a payload stores them. -/
theorem layerNorm_axesFrom_eq_reshape' (x : Tensor α s) {axis : Nat}
    (haxis : axis ≤ s.rank) (gamma beta : Tensor α (s.drop axis)) (epsilon : α)
    (hx : s.size = Shape.size [(s.take axis).prod, (s.drop axis).prod])
    (hg : Shape.size (s.drop axis) = Shape.size [(s.drop axis).prod]) :
    layerNorm x (s.axesFrom axis) (castShape gamma (Shape.selectAxes_axesFrom haxis).symm)
        (castShape beta (Shape.selectAxes_axesFrom haxis).symm) epsilon =
      reshape (layerNorm (reshape x _ hx) [⟨1, Nat.one_lt_two⟩]
          (reshape gamma [(s.drop axis).prod] hg) (reshape beta [(s.drop axis).prod] hg)
          epsilon)
        s hx.symm := by
  rw [layerNorm_axesFrom_eq_reshape x haxis _ _ epsilon hx (Shape.size_selectAxes_axesFrom haxis),
    reshape_castShape, reshape_castShape]

/-- Layer normalization over the last axis of `x : Tensor α s`, `s = lead ++ [n]`, is layer
normalization along the rows of its matrix view of `m = lead.size` rows, reshaped back to `s`,
with the scale and bias given on `[n]`. Both sides compute the same operations in the same order.
-/
theorem layerNorm_last_eq_reshape {lead : Shape} {m n : Nat} (hm : m = lead.size)
    (hs : s = lead ++ [n]) (x : Tensor α s) {ds : List (Fin s.rank)}
    (hds : ds.map Fin.val = [lead.length]) (gamma beta : Tensor α [n])
    (hg : [n] = s.selectAxes ds) (epsilon : α) :
    layerNorm x ds (castShape gamma hg) (castShape beta hg) epsilon =
      reshape (layerNorm (reshape x [m, n] (Shape.size_eq_of_append hm hs))
          [⟨1, Nat.one_lt_two⟩] gamma beta epsilon) s (Shape.size_eq_of_append hm hs).symm := by
  subst hs
  have haxis : lead.length ≤ (lead ++ [n]).rank := by
    simp [Shape.rank_eq_length]
  have hdrop : (lead ++ [n]).drop lead.length = [n] := by
    simp
  obtain rfl := Shape.eq_axesFrom_last hds
  have hcast : ∀ t : Tensor α [n], castShape t hg =
      castShape (castShape t hdrop.symm) (Shape.selectAxes_axesFrom haxis).symm := by
    intro t
    rw [castShape_trans]
  have key : ∀ {a b : Nat} (ha : a = m) (hb : b = n)
      (hx : (lead ++ [n]).size = Shape.size [a, b]) (hg' : Shape.size [n] = Shape.size [b]),
      reshape (layerNorm (reshape x [a, b] hx) [⟨1, Nat.one_lt_two⟩] (reshape gamma [b] hg')
          (reshape beta [b] hg') epsilon) _ hx.symm =
        reshape (layerNorm (reshape x [m, n] (Shape.size_eq_of_append hm rfl))
          [⟨1, Nat.one_lt_two⟩] gamma beta epsilon) _ (Shape.size_eq_of_append hm rfl).symm := by
    intro a b ha hb hx hg'
    subst ha hb
    rfl
  rw [hcast gamma, hcast beta, layerNorm_axesFrom_eq_reshape' x haxis _ _ epsilon
    (Shape.size_eq_size_take_drop _ _) (by rw [hdrop]; simp), reshape_castShape,
    reshape_castShape]
  refine key ?_ ?_ _ _
  · rw [hm, Shape.size_eq_prod]
    simp
  · rw [hdrop]
    simp

end TorchLean.Tensor

/-! ## Batch normalization with running statistics along any axis -/

namespace TorchLean.Shape

/-- A nonscalar shape has a first axis. -/
theorem zero_lt_rank_dim (n : Nat) (s : Shape) : 0 < (Shape.dim n s).rank := by
  simp [Shape.rank_eq_length]

/-- After a leading shape, the next axis has the extent placed there. -/
theorem extent_concat_dim (leading spatial : Shape) (n : Nat) :
    (leading ++ (.dim n spatial)).extent ⟨leading.rank, rank_lt_rank_append leading spatial n⟩ =
      n := by
  simp only [extent, Shape.rank_eq_length]
  rw [List.getElem_append_right (Nat.le_refl _)]
  simp

/-- A vector along the axis after a leading shape has the length placed there. -/
theorem singleton_eq_extent_concat_dim (leading spatial : Shape) (n : Nat) :
    [n] = [(leading ++ (.dim n spatial)).extent
      ⟨leading.rank, rank_lt_rank_append leading spatial n⟩] :=
  congrArg ([·]) (extent_concat_dim leading spatial n).symm

end TorchLean.Shape

namespace TorchLean.Tensor

variable {α : Type} [Storage α]

/-- Reading a vector cast along an equality of lengths reads it at the cast index. -/
theorem getScalar_castShape {m n : Nat} (t : Tensor α [m]) (h : [m] = [n]) (k : Fin n) :
    getScalar (castShape t h) k =
      t.getScalar (Fin.cast (List.head_eq_of_cons_eq h).symm k) := by
  have hmn : m = n := List.head_eq_of_cons_eq h
  subst hmn
  rw [castShape_self]
  rfl

variable {s : Shape}

/-- A vector along axis `d` broadcast to `s` reads its entry at the index along `d`. -/
theorem broadcast_axis_apply (d : Fin s.rank) (t : Tensor α [s.extent d]) (c : s.Coord) :
    broadcast t s (fun _ => d) (s.broadcastDims_axis d) c = t.getScalar (Shape.Coord.get c d) := by
  rw [broadcast_apply, getScalar_eq_apply]
  congr 1
  apply Shape.Coord.ext_val
  intro i
  match i with
  | ⟨0, _⟩ =>
      rw [Shape.Coord.get_ofFun]
      split_ifs with hi
      · have := (Shape.Coord.get c d).isLt
        change 0 = (Shape.Coord.get c d).val
        change s.extent d = 1 at hi
        omega
      · rfl

variable [Context α]

/-- An entry of batch normalization with running statistics normalizes the entry with the
statistics and affine parameters at its index along `dim`. -/
theorem batchNorm_running_apply (x : Tensor α s) (dim : Fin s.rank)
    (gamma beta mean var : Tensor α [s.extent dim]) (epsilon : α) (c : s.Coord) :
    batchNorm x dim gamma beta (.running mean var) epsilon c =
      (x c - mean.getScalar (Shape.Coord.get c dim)) /
          MathFunctions.sqrt
            (Max.max (Context.maximum (var.getScalar (Shape.Coord.get c dim)) 0 + epsilon) 0) *
        gamma.getScalar (Shape.Coord.get c dim) + beta.getScalar (Shape.Coord.get c dim) := by
  simp only [batchNorm, Normalization.normalize, add_def, mul_def, div_def, sub_def, Tensor.sqrt,
    Tensor.maximum, zipWith_apply, Tensor.map_apply, full_apply,
    broadcast_axis_apply, getScalar_eq_apply]

/-- `batchNorm_running_apply` along the outermost axis, with the parameters indexed by its
extent. -/
theorem batchNorm_running_zero_apply {n : Nat} {t : Shape} (x : Tensor α (.dim n t))
    (h : 0 < (Shape.dim n t).rank) (gamma beta mean var : Tensor α [n]) (epsilon : α)
    (c : (Shape.dim n t).Coord) :
    batchNorm x ⟨0, h⟩ gamma beta (.running mean var) epsilon c =
      (x c - mean.getScalar c.1) /
          MathFunctions.sqrt (Max.max (Context.maximum (var.getScalar c.1) 0 + epsilon) 0) *
        gamma.getScalar c.1 + beta.getScalar c.1 :=
  batchNorm_running_apply x ⟨0, h⟩ gamma beta mean var epsilon c

/-- Batch normalization with running statistics, applied by `mapLeading` along the first axis of
every slice, is batch normalization of the whole tensor along the axis after the leading ones. -/
theorem mapLeading_batchNorm_running (leading spatial : Shape) {channels : Nat}
    (gamma beta mean var : Tensor α [channels]) (epsilon : α)
    (x : Tensor α (leading ++ (.dim channels spatial))) :
    mapLeading leading (fun sample : Tensor α (.dim channels spatial) =>
        batchNorm sample ⟨0, Shape.zero_lt_rank_dim channels spatial⟩ gamma beta
          (.running mean var) epsilon) x =
      batchNorm x ⟨leading.rank, Shape.rank_lt_rank_append leading spatial channels⟩
        (castShape gamma (Shape.singleton_eq_extent_concat_dim leading spatial channels))
        (castShape beta (Shape.singleton_eq_extent_concat_dim leading spatial channels))
        (.running (castShape mean (Shape.singleton_eq_extent_concat_dim leading spatial channels))
          (castShape var (Shape.singleton_eq_extent_concat_dim leading spatial channels)))
        epsilon := by
  apply Rep.ext
  intro c
  rw [mapLeading_apply]
  dsimp only
  rw [batchNorm_running_zero_apply, batchNorm_running_apply, getScalar_castShape,
    getScalar_castShape, getScalar_castShape, getScalar_castShape, batchSlice_apply,
    Equiv.symm_apply_apply]
  have hindex : (Shape.Coord.appendEquiv leading (Shape.dim channels spatial)
        c).2.1 =
      Fin.cast (Shape.extent_concat_dim leading spatial channels)
        (Shape.Coord.get c ⟨leading.rank, Shape.rank_lt_rank_append leading spatial channels⟩) := by
    apply Fin.ext
    rw [Fin.val_cast, Shape.Coord.val_get, ← Shape.Coord.index_zero, index_appendEquiv_snd,
        Nat.zero_add]
    simp only [Shape.rank_eq_length]
  rw [hindex]

end TorchLean.Tensor

/-! ## The reverse rule of batch normalization with running statistics -/

namespace TorchLean

open TorchLean.Tensor

variable {α : Type} [Storage α] [Context α] {s : Shape}

/-- Reverse rule of `Tensor.batchNorm x dim gamma beta (.running mean var) epsilon`, with the
running mean, the running variance and epsilon held fixed. Let `g` be the output gradient and
`spread` the broadcast of a channel vector along `dim`. `std` is the forward's denominator
`sqrt (max (spread (maximum var 0) + epsilon) 0)`, both clamps kept: every input of it is fixed,
so it contributes no derivative factor. `(g * spread gamma) / std` is the input gradient, the
product rule's pullback followed by the quotient rule's, as `Op.mul` and `Op.div` compose. Over
every axis but `dim`, the scale gradient sums `g * ((x - spread mean) / std)` and the bias gradient
sums `g`. -/
def BatchNorm.runningBackward (x : Tensor α s) (dim : Fin s.rank)
    (gamma mean var : Tensor α [s.extent dim]) (epsilon : α) (gradOutput : Tensor α s) :
    Gradients α s [s.extent dim] :=
  let spread := fun t : Tensor α [s.extent dim] => broadcast t s _ (s.broadcastDims_axis dim)
  let others := s.otherAxes [dim]
  let std := Tensor.sqrt (spread (Tensor.maximum var (full _ 0)) + full s epsilon)
    .clampNegative
  { inputGradient := (gradOutput * spread gamma) / std
    scaleGradient := castShape (reduce .sum (gradOutput * ((x - spread mean) / std)) others false)
      (Shape.eraseAxes_otherAxes_singleton s dim)
    biasGradient := castShape (reduce .sum gradOutput others false)
      (Shape.eraseAxes_otherAxes_singleton s dim) }

/-- The forward of batch normalization with running statistics, in the terms of
`BatchNorm.runningBackward`: the centered input divided by its denominator, times the broadcast
scale, plus the broadcast bias. -/
theorem Tensor.batchNorm_running_eq (x : Tensor α s) (dim : Fin s.rank)
    (gamma beta mean var : Tensor α [s.extent dim]) (epsilon : α) :
    Tensor.batchNorm x dim gamma beta (.running mean var) epsilon =
      ((x - broadcast mean s _ (s.broadcastDims_axis dim)) /
          Tensor.sqrt (broadcast (Tensor.maximum var (full _ 0)) s _
            (s.broadcastDims_axis dim) + full s epsilon) .clampNegative) *
        broadcast gamma s _ (s.broadcastDims_axis dim) +
      broadcast beta s _ (s.broadcastDims_axis dim) :=
  normalize_full _ _ _ _ _ _ _ _

end TorchLean
