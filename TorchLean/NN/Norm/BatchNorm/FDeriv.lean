/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Tensor.Layers.Normalization
public import TorchLean.NN.Norm.MatrixEntries
public import TorchLean.NN.Norm.RowNorm

/-!
# Fréchet derivative of BatchNorm

On a channel-first tensor, `Tensor.batchNorm x 0 gamma beta .batch ε` flattens the spatial axes
`sSpatial` to a `[channels, sSpatial.size]` matrix, normalizes every row of that matrix with the row
statistics and applies the per-channel affine parameters (`batchNorm_eq_explicit`). Each centered
row is divided by its scale `channelScale` before it is squared, and over the reals the scale
cancels (`get2_flat_batchNorm`) because it bounds every centered entry and `sqrt |ε|`
(`le_getScalar_channelScale`). For `0 < ε` the map `(x, gamma, beta) ↦ batchNorm` is Fréchet
differentiable on the flattened vectors, and its derivative is `TorchLean.BatchNorm.jvp`.

The calculus comes from `RowNorm.hasFDerivAt_nrm`, the derivative of one normalized row entry.
The rest is bookkeeping: reshaping commutes with pointwise operations, the clamps in the spec are
inactive because the variance is a mean of squares, and every entry of the JVP is the closed-form
row differential.
-/

@[expose] public section

open FloatLib.Numerics (MathFunctions)
open TorchLean.Shape (size size_cons)

namespace TorchLean
namespace BatchNorm

open Tensor
open Normalization RowNorm Matmul

open scoped BigOperators

noncomputable section

/-! ## Reshaping -/

/-- Reshaping commutes with pointwise binary operations. -/
theorem _root_.TorchLean.Tensor.reshape_zipWith {s₁ s₂ : Shape} (f : ℝ → ℝ → ℝ) (a b : Tensor ℝ s₁)
    (h : s₁.size = s₂.size) :
    Tensor.reshape (zipWith f a b) _ h = zipWith f (Tensor.reshape a _ h)
        (Tensor.reshape b _ h) := by
  apply Tensor.Rep.ext
  intro c
  simp [Tensor.reshape]

/-- Reshaping commutes with tensor addition. -/
theorem _root_.TorchLean.Tensor.reshape_add {s₁ s₂ : Shape} (a b : Tensor ℝ s₁)
    (h : s₁.size = s₂.size) :
    Tensor.reshape (a + b) _ h = Tensor.reshape a _ h + Tensor.reshape b _ h :=
  reshape_zipWith _ a b h

/-- Reshaping commutes with tensor subtraction. -/
theorem _root_.TorchLean.Tensor.reshape_sub {s₁ s₂ : Shape} (a b : Tensor ℝ s₁)
    (h : s₁.size = s₂.size) :
    Tensor.reshape (a - b) _ h = Tensor.reshape a _ h - Tensor.reshape b _ h :=
  reshape_zipWith _ a b h

/-- Reshaping commutes with pointwise multiplication. -/
theorem _root_.TorchLean.Tensor.reshape_mul {s₁ s₂ : Shape} (a b : Tensor ℝ s₁)
    (h : s₁.size = s₂.size) :
    Tensor.reshape (a * b) _ h = Tensor.reshape a _ h * Tensor.reshape b _ h :=
  reshape_zipWith _ a b h

/-- Reshaping commutes with pointwise division. -/
theorem _root_.TorchLean.Tensor.reshape_div {s₁ s₂ : Shape} (a b : Tensor ℝ s₁)
    (h : s₁.size = s₂.size) :
    Tensor.reshape (a / b) _ h = Tensor.reshape a _ h / Tensor.reshape b _ h :=
  reshape_zipWith _ a b h

/-! ## The flattened matrix form -/

variable {channels : Nat} {sSpatial : Shape}

/-- Flattening the spatial axes preserves the size. -/
theorem size_flat (channels : Nat) (sSpatial : Shape) :
    Shape.size (Shape.dim channels sSpatial) =
      Shape.size (Shape.dim channels (.dim sSpatial.size .scalar)) := by
  simp

/-- The spatial size is positive when the channel-first shape is well formed. -/
theorem size_pos_of_wellFormed_dim (channels : Nat) [Shape.WellFormed (.dim channels sSpatial)] :
    0 < sSpatial.size :=
  Shape.size_pos_of_positive (Shape.WellFormed.proof (s := .dim channels sSpatial)).2

/-- The input flattened to a `[channels, sSpatial.size]` matrix. -/
def flat (x : Tensor ℝ (.dim channels sSpatial)) : Tensor ℝ [channels, sSpatial.size] :=
  Tensor.reshape x _ (size_flat channels sSpatial)

/-- Erasing the spatial axes of a channel-first shape leaves the channel axis. -/
theorem eraseAxes_otherAxes_zero (channels : Nat) (sSpatial : Shape) :
    Shape.eraseAxes (channels :: sSpatial)
      (Shape.otherAxes (channels :: sSpatial) [⟨0, Nat.succ_pos _⟩]) = [channels] :=
  Shape.eraseAxes_otherAxes_singleton _ _

/-- The row means of the flattened input are its means over the spatial axes. -/
theorem lnMean_flat [Shape.WellFormed (.dim channels sSpatial)]
    (t : Tensor ℝ (.dim channels sSpatial)) :
    lnMean (flat t) =
      Tensor.castShape (Tensor.map (Context.divNat · (if Shape.size
          (Shape.selectAxes (channels :: sSpatial) (Shape.otherAxes (channels :: sSpatial)
            [⟨0, Nat.succ_pos _⟩])) = 0 then 1 else Shape.size
              (Shape.selectAxes (channels :: sSpatial) (Shape.otherAxes (channels :: sSpatial)
                [⟨0, Nat.succ_pos _⟩]))))
        (Tensor.reduce .sum t (Shape.otherAxes (channels :: sSpatial) [⟨0, Nat.succ_pos _⟩])
          false)) (eraseAxes_otherAxes_zero channels sSpatial) := by
  have hpos : 0 < Shape.size sSpatial := size_pos_of_wellFormed_dim channels
  have hcount : ((if Shape.size (Shape.selectAxes (channels :: sSpatial)
      (Shape.otherAxes (channels :: sSpatial) [⟨0, Nat.succ_pos _⟩])) = 0 then 1
      else Shape.size (Shape.selectAxes (channels :: sSpatial)
        (Shape.otherAxes (channels :: sSpatial) [⟨0, Nat.succ_pos _⟩])) : Nat)) =
      Shape.size sSpatial := by
    rw [Shape.selectAxes_otherAxes_zero, ite_eq_right (by omega)]
  apply Tensor.Rep.ext
  intro o
  erw [Tensor.castShape_apply]
  simp only [lnMean, flat, Tensor.map_apply, hcount]
  congr 1
  erw [castShape_reduce_sum_singleton_mk_apply _ 1]
  rw [← Tensor.sum_eq_foldl]
  exact (Tensor.sum_reshape_row t (size_flat channels sSpatial) o.1
    (_root_.cast (congrArg Shape.Coord (eraseAxes_otherAxes_zero channels sSpatial).symm) o)
    ((Shape.Coord.index_cast (eraseAxes_otherAxes_zero channels sSpatial).symm o 0).trans rfl)
      fun v => v).trans (Tensor.reduce_sum_apply t _ false _ _).symm

/-- Broadcasting a channel vector along the leading axis is the broadcast of the spatial
statistics it is cast from. -/
theorem broadcast_castShape_spatial (t : Tensor ℝ (Shape.reduce (channels :: sSpatial)
    (Shape.otherAxes (channels :: sSpatial) [⟨0, Nat.succ_pos _⟩]) false)) :
    Tensor.broadcast (Tensor.castShape t (eraseAxes_otherAxes_zero channels sSpatial))
        (channels :: sSpatial) (fun _ => ⟨0, Nat.succ_pos _⟩) (Shape.broadcastDims_head _ _) =
      t.broadcast (channels :: sSpatial) (Shape.keptAxis _ _)
        (Shape.broadcastDims_eraseAxes _ _) := by
  have hkept (h : 0 < (Shape.eraseAxes (channels :: sSpatial)
      (Shape.otherAxes (channels :: sSpatial) [⟨0, Nat.succ_pos _⟩])).rank) :
      (Shape.keptAxis (channels :: sSpatial)
        (Shape.otherAxes (channels :: sSpatial) [⟨0, Nat.succ_pos _⟩]) ⟨0, h⟩).val = 0 := by
    simp only [Shape.keptAxis]
    rw [List.getElem_of_eq (Shape.otherAxes_otherAxes_singleton _ _)]
    rfl
  apply Tensor.Rep.ext
  intro c
  rw [Tensor.broadcast_apply_of_extent_eq _ _ _ (fun i => by match i with | ⟨0, _⟩ => rfl)]
  erw [Tensor.broadcast_apply_of_extent_eq _ _ _ (Shape.extent_keptAxis _ _),
    Tensor.castShape_apply]
  congr 1
  apply Shape.Coord.ext_index
  intro k hk
  have hk0 : k = 0 := by
    have h1 := hk
    simp only [Shape.reduce, Bool.cond_false, eraseAxes_otherAxes_zero] at h1
    have h2 : k < 1 := h1
    omega
  subst hk0
  rw [Shape.Coord.index_cast, Shape.Coord.index_ofFun]
  erw [Shape.Coord.index_ofFun]
  rw [dite_eq_left (by simp), dite_eq_left hk]
  simp only [Fin.val_cast, Shape.Coord.val_get]
  · exact (congrArg (Shape.Coord.index c) (hkept hk)).symm
  · exact (eraseAxes_otherAxes_zero channels sSpatial).symm

/-- Casting the shape commutes with the regularized square root of a normalization. -/
theorem castShape_sqrt_add_div {u v : Shape} (h : u = v) (A SC : Tensor ℝ u) (ε : ℝ) :
    Tensor.castShape (Tensor.sqrt (A + ((Tensor.full u ε / SC) / SC)) .clampNegative) h =
      Tensor.sqrt (Tensor.castShape A h + ((Tensor.full v ε / Tensor.castShape SC h) /
        Tensor.castShape SC h)) .clampNegative := by
  subst h
  rfl

/-- Every bound on the magnitudes that the spatial reduction reads for row `c` also bounds each
entry of row `c` of the flattened tensor. -/
theorem abs_get2_flat_le (t : Tensor ℝ (.dim channels sSpatial)) (c : Fin channels)
    (o : (Shape.reduce (channels :: sSpatial)
      (Shape.otherAxes (channels :: sSpatial) [⟨0, Nat.succ_pos _⟩]) false).Coord)
    (ho : Shape.Coord.index o 0 = c.val) {L : ℝ}
    (hL : ∀ k, |t (Shape.reduceSource (channels :: sSpatial) _ false o
      (Shape.Coord.unlinearize k))| ≤ L)
    (p : Fin sSpatial.size) : |Tensor.get2 (flat t) c p| ≤ L := by
  have hsum := Tensor.sum_reshape_row t (size_flat channels sSpatial) c o ho
    (fun v => max (|v| - L) 0)
  have hzero : ∀ (l : List (Fin (Shape.size (Shape.selectAxes
      (channels :: sSpatial) (Shape.otherAxes (channels :: sSpatial) [⟨0, Nat.succ_pos _⟩])))))
      (a : ℝ), l.foldl (fun acc k => acc + max (|t (Shape.reduceSource (channels :: sSpatial) _
        false o (Shape.Coord.unlinearize k))| - L) 0) a = a := by
    intro l
    induction l with
    | nil => intro a; rfl
    | cons k l ih =>
      intro a
      rw [List.foldl_cons, max_eq_right (sub_nonpos.mpr (hL k)), add_zero, ih]
  rw [hzero, sum_vec] at hsum
  simp only [getScalar_ofFn] at hsum
  have h0 := (Finset.sum_eq_zero_iff_of_nonneg fun i _ => le_max_right _ _).mp hsum p
    (Finset.mem_univ _)
  rw [Tensor.get2_eq_apply, flat]
  exact sub_nonpos.mp (max_eq_right_iff.mp h0)

/-- The scale of a tensor over the spatial axes bounds the magnitudes of each row of the flattened
tensor and `sqrt |ε|`. -/
theorem le_normalizationScale_spatial [Shape.WellFormed (.dim channels sSpatial)]
    (t : Tensor ℝ (.dim channels sSpatial)) (ε : ℝ) (c : Fin channels) :
    (∀ p, |Tensor.get2 (flat t) c p| ≤ getScalar (Tensor.castShape (Tensor.normalizationScale t
        (Shape.otherAxes (channels :: sSpatial) [⟨0, Nat.succ_pos _⟩]) (Tensor.full _ ε))
        (eraseAxes_otherAxes_zero channels sSpatial)) c) ∧
      Real.sqrt |ε| ≤ getScalar (Tensor.castShape (Tensor.normalizationScale t
        (Shape.otherAxes (channels :: sSpatial) [⟨0, Nat.succ_pos _⟩]) (Tensor.full _ ε))
        (eraseAxes_otherAxes_zero channels sSpatial)) c := by
  have hext : ∀ d ∈ Shape.otherAxes (channels :: sSpatial) [⟨0, Nat.succ_pos _⟩],
      0 < Shape.extent (channels :: sSpatial) d := by
    intro d _
    refine Nat.pos_of_ne_zero fun h0 => ?_
    have hmem : Shape.extent (channels :: sSpatial) d ∈ (channels :: sSpatial) :=
      List.getElem_mem _
    have hsize :=
      Shape.size_pos_of_positive (Shape.WellFormed.proof (s := .dim channels sSpatial))
    exact absurd (Shape.size_eq_zero_iff.mpr (h0 ▸ hmem)) hsize.ne'
  have habs : ∀ {s : Shape} (u : Tensor ℝ s) (o : s.Coord), Tensor.abs u o = |u o| :=
    fun u o => (Tensor.map_apply MathFunctions.abs u o).trans
      (LawfulContext.abs_eq (u o))
  have hstop : ∀ v : ℝ, Context.stopGradient v = v := fun _ => rfl
  obtain ⟨o, ho, hval⟩ : ∃ o : (Shape.reduce (channels :: sSpatial)
      (Shape.otherAxes (channels :: sSpatial) [⟨0, Nat.succ_pos _⟩]) false).Coord,
      Shape.Coord.index o 0 = c.val ∧
        getScalar (Tensor.castShape (Tensor.normalizationScale t
          (Shape.otherAxes (channels :: sSpatial) [⟨0, Nat.succ_pos _⟩]) (Tensor.full _ ε))
          (eraseAxes_otherAxes_zero channels sSpatial)) c =
        Tensor.normalizationScale t (Shape.otherAxes (channels :: sSpatial)
          [⟨0, Nat.succ_pos _⟩]) (Tensor.full _ ε) o :=
    ⟨_, (Shape.Coord.index_cast (eraseAxes_otherAxes_zero channels sSpatial).symm
        ((c, ()) : Shape.Coord [channels]) 0).trans rfl, by
      rw [Tensor.getScalar_eq_apply]
      exact castShape_apply _ _ _⟩
  have hreg : Tensor.sqrt (Tensor.abs (Tensor.full (Shape.reduce (channels :: sSpatial)
      (Shape.otherAxes (channels :: sSpatial) [⟨0, Nat.succ_pos _⟩]) false) ε)) .clampNegative o =
      Real.sqrt |ε| := by
    rw [sqrt_clampNegative_apply, habs, full_apply]
    exact congrArg Real.sqrt (max_eq_left (abs_nonneg ε))
  rw [hval]
  simp only [Tensor.normalizationScale]
  rw [detach_apply, hstop, maximum_apply, dite_eq_left hext, reduce_max_apply]
  simp only [Context.maximum_real]
  refine ⟨fun p => le_max_of_le_left (abs_get2_flat_le t c o ho (fun k =>
    (habs t _).symm.trans_le (List.le_foldl_max_of_mem _ (fun k => Tensor.abs t
      (Shape.reduceSource _ _ false o (Shape.Coord.unlinearize k)))
        (List.mem_finRange k))) p), le_max_of_le_right hreg.ge⟩

/-- The scale `Tensor.normalizationScale` of each centered channel of `x`, as a channel vector. -/
def channelScale (x : Tensor ℝ (.dim channels sSpatial)) (ε : ℝ) : Tensor ℝ [channels] :=
  Tensor.castShape (Tensor.normalizationScale
      (x - Tensor.reshape (rowBroadcast (n := sSpatial.size) (lnMean (flat x))) _
        ((size_flat channels sSpatial).symm))
      (Shape.otherAxes (channels :: sSpatial) [⟨0, Nat.succ_pos _⟩]) (Tensor.full _ ε))
    (eraseAxes_otherAxes_zero channels sSpatial)

/-- The scale of each channel bounds the magnitudes of the centered row of the flattened input
and `sqrt |ε|`. -/
theorem le_getScalar_channelScale [Shape.WellFormed (.dim channels sSpatial)]
    (x : Tensor ℝ (.dim channels sSpatial)) (ε : ℝ) (c : Fin channels) :
    (∀ p, |Tensor.get2 (flat x) c p - 𝔼 q, Tensor.get2 (flat x) c q| ≤
        getScalar (channelScale x ε) c) ∧
      Real.sqrt |ε| ≤ getScalar (channelScale x ε) c := by
  have hflat : flat (x - Tensor.reshape (rowBroadcast (n := sSpatial.size)
      (lnMean (flat x))) _ ((size_flat channels sSpatial).symm)) = lnCentered (flat x) := by
    simp only [flat, reshape_sub, reshape_reshape_self]
    rfl
  obtain ⟨hc, hε⟩ := le_normalizationScale_spatial (x - Tensor.reshape (rowBroadcast
    (n := sSpatial.size) (lnMean (flat x))) _ ((size_flat channels sSpatial).symm)) ε c
  refine ⟨fun p => ?_, hε⟩
  have h := hc p
  rwa [hflat, get2_lnCentered] at h

/-- Batch normalization along the leading axis with its reshapes spelled out: the statistics are
row statistics of the flattened input, and each centered row is divided by its scale
`channelScale` before it is squared. -/
def explicit (x : Tensor ℝ (.dim channels sSpatial)) (gamma beta : Tensor ℝ [channels])
    (ε : ℝ) : Tensor ℝ (.dim channels sSpatial) :=
  let spread := fun v : Tensor ℝ [channels] =>
    Tensor.reshape (rowBroadcast (n := sSpatial.size) v) _ ((size_flat channels sSpatial).symm)
  let scale := channelScale x ε
  let scaled := (x - spread (lnMean (flat x))) / spread scale
  let std := Tensor.sqrt (lnMean (flat (scaled * scaled)) +
    ((Tensor.full [channels] ε / scale) / scale)) .clampNegative
  ((scaled / spread std) * spread gamma) + spread beta

/-- Batch normalization along the leading axis with batch statistics is `explicit`: the
means and the scale over the other axes are taken row by row in the flattened matrix. -/
theorem batchNorm_eq_explicit [hwf : Shape.WellFormed (.dim channels sSpatial)]
    (x : Tensor ℝ (.dim channels sSpatial)) (gamma beta : Tensor ℝ [channels]) (ε : ℝ) :
    Tensor.batchNorm x ⟨0, Nat.succ_pos _⟩ gamma beta .batch ε =
      explicit x gamma beta ε := by
  symm
  simp only [explicit, channelScale]
  simp only [lnMean_flat]
  simp only [← castShape_sqrt_add_div (u := Shape.reduce (channels :: sSpatial)
    (Shape.otherAxes (channels :: sSpatial) [⟨0, Nat.succ_pos _⟩]) false)
    (eraseAxes_otherAxes_zero channels sSpatial)]
  simp only [rowBroadcast]
  simp only [Tensor.reshape_broadcast_eq_broadcast]
  simp only [broadcast_castShape_spatial]
  rfl

/-- Entries of batch normalization along the leading axis in the flattened matrix: each row is
centered, divided by the clamped standard deviation, scaled and shifted by the channel
parameters. -/
theorem get2_flat_batchNorm [Shape.WellFormed (.dim channels sSpatial)]
    (x : Tensor ℝ (.dim channels sSpatial)) (gamma beta : Tensor ℝ [channels]) (ε : ℝ)
    (c : Fin channels) (p : Fin sSpatial.size) :
    Tensor.get2 (Tensor.reshape (Tensor.batchNorm x ⟨0, Nat.succ_pos _⟩ gamma beta .batch ε) _
        (size_flat channels sSpatial)) c p =
      (Tensor.get2 (flat x) c p - 𝔼 q, Tensor.get2 (flat x) c q) /
          Real.sqrt (max (max (TorchLean.var (Tensor.get2 (flat x) c)) 0 + ε) 0) *
            getScalar gamma c +
        getScalar beta c := by
  obtain ⟨hc, hε⟩ := le_getScalar_channelScale x ε c
  have hflat : Tensor.reshape (explicit x gamma beta ε) _ (size_flat channels sSpatial) =
      (rowsGiven (flat x) (channelScale x ε) ε false).1 * rowBroadcast gamma +
        rowBroadcast beta := by
    simp only [explicit, flat, reshape_add, reshape_mul, reshape_div, reshape_sub,
      reshape_reshape_self]
    rfl
  rw [batchNorm_eq_explicit, hflat, get2_add, get2_mul,
    get2_rowsGiven_fst _ _ ε false c hc hε, get2_rowBroadcast, get2_rowBroadcast,
    max_eq_left (var_nonneg (Tensor.get2 (flat x) c))]

/-- `jvp` is the normalized-data JVP of the flattened tensors, reshaped back to the input
shape. -/
theorem jvp_eq_reshape_normalizedJVP
    (x dx : Tensor ℝ (.dim channels sSpatial)) (gamma dgamma beta dbeta : Tensor ℝ [channels])
    (ε : ℝ) :
    jvp x dx gamma dgamma beta dbeta ε =
      Tensor.reshape (BatchNorm.normalizedJVP (flat dx)
          (Normalization.normalizedRows (flat x) ε false).1
          (Normalization.normalizedRows (flat x) ε false).2 gamma dgamma dbeta) _
        ((size_flat channels sSpatial).symm) :=
  rfl

/-- Entries of the normalized-data JVP. -/
theorem get2_normalizedJVP {C P : Nat} (t xHat : Tensor ℝ [C, P])
    (invStd gamma dgamma dbeta : Tensor ℝ [C]) (c : Fin C) (p : Fin P) :
    Tensor.get2 (BatchNorm.normalizedJVP t xHat invStd gamma dgamma dbeta) c p =
      getScalar invStd c *
          (Tensor.get2 t c p - (∑ k : Fin P, Tensor.get2 t c k) / P -
            Tensor.get2 xHat c p * ((∑ k : Fin P, Tensor.get2 t c k * Tensor.get2 xHat c k) / P)) *
          getScalar gamma c +
        Tensor.get2 xHat c p * getScalar dgamma c + getScalar dbeta c := by
  have hb (v : Tensor ℝ ([C, P].eraseIdx 1)) (c : Fin C) (p : Fin P) :
      Tensor.get2 (Tensor.broadcast v [C, P] _ (Shape.eraseIdx_broadcastDims [C, P] 1)) c p =
        getScalar (n := C) v c :=
    get2_rowBroadcast v c p
  have hb' (v : Tensor ℝ [C]) (c : Fin C) (p : Fin P) :
      Tensor.get2 (Tensor.broadcast (s := [C, P].eraseIdx 1) v [C, P] _
          (Shape.eraseIdx_broadcastDims [C, P] 1)) c p = getScalar v c :=
    get2_rowBroadcast v c p
  have hm (x : Tensor ℝ [C, P]) (c : Fin C) :
      getScalar (n := C) (Tensor.map (shape := [C, P].eraseIdx 1) (Context.divNat · P)
          (castShape (target := [C, P].eraseIdx 1) (reduce .sum x [⟨1, Nat.one_lt_two⟩] false)
            (Shape.reduce_singleton _ _))) c =
        (∑ k : Fin P, Tensor.get2 x c k) / P :=
    getScalar_rowMean x c
  simp only [BatchNorm.normalizedJVP, get2_add, get2_mul, get2_sub, hb, hb', hm]

/-! ## Vector form -/

/-- Domain of the vectorized BatchNorm: flattened input and the two channel parameters. -/
abbrev Domain (channels : Nat) (sSpatial : Shape) : Type :=
  Vec (Shape.size (Shape.dim channels sSpatial)) × Vec (Shape.size [channels]) ×
    Vec (Shape.size [channels])

/-- Batch normalization along the leading axis on flattened vectors. -/
def forwardVec [Shape.WellFormed (.dim channels sSpatial)] (ε : ℝ) (q : Domain channels sSpatial) :
    Vec (Shape.size (Shape.dim channels sSpatial)) :=
  toVec (Tensor.batchNorm (ofVec (s := .dim channels sSpatial) q.1) ⟨0, Nat.succ_pos _⟩
    (ofVec (s := .dim channels .scalar) q.2.1)
    (ofVec (s := .dim channels .scalar) q.2.2) .batch ε)

/-- Row (channel) of a flat input index in the `[channels, sSpatial.size]` matrix. -/
def rowIdx (k : Fin (Shape.size (Shape.dim channels sSpatial))) : Fin channels :=
  rowOf (Fin.cast (size_flat channels sSpatial) k)

/-- Column (position) of a flat input index in the `[channels, sSpatial.size]` matrix. -/
def colIdx (k : Fin (Shape.size (Shape.dim channels sSpatial))) : Fin sSpatial.size :=
  colOf (Fin.cast (size_flat channels sSpatial) k)

/-- Channel index of a coordinate of the input, as an index into a channel vector. -/
def chanIdx (k : Fin (Shape.size (Shape.dim channels sSpatial))) :
    Fin (Shape.size [channels]) :=
  Fin.cast (Shape.size_vector channels).symm (rowIdx k)

/-- The input as a flattened `[channels, sSpatial.size]` matrix vector. -/
def matVec (v : Vec (Shape.size (Shape.dim channels sSpatial))) :
    Vec (Shape.size [channels, sSpatial.size]) :=
  LinearIsometryEquiv.piLpCongrLeft 2 ℝ ℝ (finCongr (size_flat channels sSpatial)) v

/-- Every input coordinate is `idxMN` of its row and column, up to the size cast. -/
theorem cast_eq_idxMN (k : Fin (Shape.size (Shape.dim channels sSpatial))) :
    Fin.cast (size_flat channels sSpatial) k =
      idxMN (m := channels) (n := sSpatial.size) (rowIdx k) (colIdx k) :=
  (idxMN_rowOf_colOf _).symm

/-- Entries of a flattened tensor are coordinates of `matVec` of its vectorization. -/
theorem get2_flat (t : Tensor ℝ (.dim channels sSpatial)) (c : Fin channels)
    (p : Fin sSpatial.size) :
    Tensor.get2 (flat t) c p =
      matVec (toVec t) (idxMN (m := channels) (n := sSpatial.size) c p) := by
  rw [← Matmul.toVec_idxMN, flat, toVec_reshape_apply, matVec,
    LinearIsometryEquiv.piLpCongrLeft_apply, Equiv.piCongrLeft'_apply, finCongr_symm_apply]

/-- Entries of the flattened input tensor are coordinates of `matVec`. -/
theorem get2_flat_ofVec (v : Vec (Shape.size (Shape.dim channels sSpatial)))
    (c : Fin channels) (p : Fin sSpatial.size) :
    Tensor.get2 (flat (ofVec (s := .dim channels sSpatial) v)) c p =
      matVec v (idxMN (m := channels) (n := sSpatial.size) c p) := by
  rw [get2_flat, toVec_ofVec]

/-- The mean of row `c` of a flattened tensor is the mean of row `c` of `matVec` of its
vectorization. -/
theorem rowMeanE_flat (t : Tensor ℝ (.dim channels sSpatial)) (c : Fin channels) :
    𝔼 p, Tensor.get2 (flat t) c p =
      𝔼 p, matVec (toVec t) (idxMN (m := channels) (n := sSpatial.size) c p) :=
  Finset.expect_congr rfl fun p _ => get2_flat t c p

/-- The variance of row `c` of a flattened tensor is the variance of row `c` of `matVec` of its
vectorization. -/
theorem rowVarE_flat (t : Tensor ℝ (.dim channels sSpatial)) (c : Fin channels) :
    TorchLean.var (Tensor.get2 (flat t) c) =
      TorchLean.var fun p =>
        matVec (toVec t) (idxMN (m := channels) (n := sSpatial.size) c p) :=
  congrArg TorchLean.var (funext (get2_flat t c))

/-- Entries of a channel vector are coordinates of its vectorization. -/
theorem getScalar_chan (g : Tensor ℝ [channels]) (c : Fin channels) :
    getScalar g c = toVec g (Fin.cast (Shape.size_vector channels).symm c) := by
  rw [toVec_vec]
  rfl

/-- Closed form of the vectorized BatchNorm: each entry is its normalized row entry `nrm` times the
channel's `gamma`, plus the channel's `beta`. It agrees with `forwardVec` for positive `ε`
(`forwardVec_eq_closedForm`). -/
def closedForm (ε : ℝ) (q : Domain channels sSpatial) :
    Vec (Shape.size (Shape.dim channels sSpatial)) :=
  WithLp.toLp 2 fun k =>
    nrm (matVec q.1) ε (rowIdx k) (colIdx k) * q.2.1 (chanIdx k) + q.2.2 (chanIdx k)

/-- For positive `ε` the clamps in batch normalization are inactive and the vectorized map is its
closed form. -/
theorem forwardVec_eq_closedForm [Shape.WellFormed (.dim channels sSpatial)] {ε : ℝ} (hε : 0 < ε) :
    forwardVec (channels := channels) (sSpatial := sSpatial) ε = closedForm ε := by
  funext q
  apply PiLp.ext
  intro k
  have hL : (forwardVec ε q).ofLp k =
      Tensor.get2
        (Tensor.reshape (Tensor.batchNorm (ofVec (s := .dim channels sSpatial) q.1)
            ⟨0, Nat.succ_pos _⟩ (ofVec (s := .dim channels .scalar) q.2.1)
            (ofVec (s := .dim channels .scalar) q.2.2) .batch ε) _
            (size_flat channels sSpatial))
        (rowIdx k) (colIdx k) := by
    rw [← Matmul.toVec_idxMN, ← cast_eq_idxMN, toVec_reshape_apply]
    rfl
  rw [hL, get2_flat_batchNorm, get2_flat_ofVec, rowMeanE_flat, rowVarE_flat,
    toVec_ofVec, getScalar_chan, getScalar_chan, toVec_ofVec,
    toVec_ofVec,
    max_eq_left (var_nonneg fun p =>
      matVec q.1 (idxMN (m := channels) (n := sSpatial.size) (rowIdx k) p)),
    max_eq_left (rowVar_add_pos hε (matVec q.1) (rowIdx k)).le]
  simp only [closedForm, PiLp.toLp_apply, nrm, centered, invStd, chanIdx, div_eq_mul_inv]

/-- Derivative of the closed form at a point with flattened input `X` and `gamma` coordinates `g`,
as a continuous linear map (`beta` does not enter it). -/
def derivative (X : Vec (Shape.size [channels, sSpatial.size]))
    (g : Vec (Shape.size [channels])) (ε : ℝ) :
    Domain channels sSpatial →L[ℝ] Vec (Shape.size (Shape.dim channels sSpatial)) := by
  classical
  let fLin : Domain channels sSpatial →ₗ[ℝ] Vec (Shape.size (Shape.dim channels sSpatial)) :=
    { toFun := fun d =>
        WithLp.toLp 2 fun k =>
          nrmJVPCLM X ε (matVec d.1) (idxMN (m := channels) (n := sSpatial.size)
              (rowIdx k) (colIdx k)) * g (chanIdx k) +
            nrm X ε (rowIdx k) (colIdx k) * d.2.1 (chanIdx k) + d.2.2 (chanIdx k)
      map_add' := by
        intro a b
        apply PiLp.ext
        intro k
        simp only [matVec, Prod.fst_add, Prod.snd_add, map_add, PiLp.add_apply]
        ring
      map_smul' := by
        intro r a
        apply PiLp.ext
        intro k
        simp only [matVec, Prod.smul_fst, Prod.smul_snd, map_smul,
          PiLp.smul_apply, smul_eq_mul, RingHom.id_apply]
        ring }
  exact { toLinearMap := fLin, cont := LinearMap.continuous_of_finiteDimensional fLin }

/-- Coordinates of the batch-norm differential, split into its three contributions.

Read left to right, the summands are the Jacobian of the normalization applied to the input
perturbation and scaled by `γ`, the normalized activation times the perturbation of `γ`, and the
perturbation of `β`. Each summand reads one component of the tangent, so the input, scale and
shift tangents contribute separately. -/
@[simp] theorem derivative_apply (X : Vec (Shape.size [channels, sSpatial.size]))
    (g : Vec (Shape.size [channels])) (ε : ℝ) (d : Domain channels sSpatial)
    (k : Fin (Shape.size (Shape.dim channels sSpatial))) :
    derivative X g ε d k =
      nrmJVPCLM X ε (matVec d.1) (idxMN (m := channels) (n := sSpatial.size)
          (rowIdx k) (colIdx k)) * g (chanIdx k) +
        nrm X ε (rowIdx k) (colIdx k) * d.2.1 (chanIdx k) + d.2.2 (chanIdx k) := by
  -- Here and below, simp leaves out `size_cons`. It would rewrite the sizes inside `Fin` types
  -- before the `WithLp.toLp` lemmas match.
  simp [derivative, -Shape.size_cons]

/-- For positive `ε` and at least one position, the closed form has Fréchet derivative `derivative`
at every point. -/
theorem hasFDerivAt_closedForm (hP : 0 < sSpatial.size) {ε : ℝ} (hε : 0 < ε)
    (q : Domain channels sSpatial) :
    HasFDerivAt (closedForm (channels := channels) (sSpatial := sSpatial) ε)
      (derivative (matVec q.1) q.2.1 ε) q := by
  rw [← hasFDerivWithinAt_univ, hasFDerivWithinAt_euclidean]
  intro k
  rw [hasFDerivWithinAt_univ]
  have hX : HasFDerivAt (fun d : Domain channels sSpatial => matVec d.1)
      (ContinuousLinearMap.comp
        (LinearIsometryEquiv.piLpCongrLeft 2 ℝ ℝ
          (finCongr (size_flat channels sSpatial))).toContinuousLinearEquiv.toContinuousLinearMap
        (ContinuousLinearMap.fst ℝ (Vec (Shape.size (Shape.dim channels sSpatial)))
          (Vec (Shape.size [channels]) × Vec (Shape.size [channels])))) q :=
    (LinearIsometryEquiv.piLpCongrLeft 2 ℝ ℝ
      (finCongr (size_flat channels sSpatial))).toContinuousLinearEquiv.toContinuousLinearMap
      |>.hasFDerivAt.comp q hasFDerivAt_fst
  have hnrm := (hasFDerivAt_nrm hε (matVec q.1) (rowIdx k) (colIdx k)).comp q hX
  have hg : HasFDerivAt (fun d : Domain channels sSpatial => d.2.1 (chanIdx k))
      ((EuclideanSpace.proj (𝕜 := ℝ) (chanIdx k) : Vec (Shape.size [channels]) →L[ℝ] ℝ).comp
        ((ContinuousLinearMap.fst ℝ (Vec (Shape.size [channels]))
          (Vec (Shape.size [channels]))).comp
          (ContinuousLinearMap.snd ℝ (Vec (Shape.size (Shape.dim channels sSpatial)))
            (Vec (Shape.size [channels]) × Vec (Shape.size [channels]))))) q :=
    (EuclideanSpace.proj (𝕜 := ℝ) (chanIdx k)).hasFDerivAt.comp q
      (hasFDerivAt_fst.comp q hasFDerivAt_snd)
  have hb : HasFDerivAt (fun d : Domain channels sSpatial => d.2.2 (chanIdx k))
      ((EuclideanSpace.proj (𝕜 := ℝ) (chanIdx k) : Vec (Shape.size [channels]) →L[ℝ] ℝ).comp
        ((ContinuousLinearMap.snd ℝ (Vec (Shape.size [channels]))
          (Vec (Shape.size [channels]))).comp
          (ContinuousLinearMap.snd ℝ (Vec (Shape.size (Shape.dim channels sSpatial)))
            (Vec (Shape.size [channels]) × Vec (Shape.size [channels]))))) q :=
    (EuclideanSpace.proj (𝕜 := ℝ) (chanIdx k)).hasFDerivAt.comp q
      (hasFDerivAt_snd.comp q hasFDerivAt_snd)
  have h := (hnrm.mul hg).add hb
  have hfun : (fun d : Domain channels sSpatial => closedForm ε d k) =
      fun d : Domain channels sSpatial =>
        nrm (matVec d.1) ε (rowIdx k) (colIdx k) * d.2.1 (chanIdx k) + d.2.2 (chanIdx k) := by
    funext d
    simp [closedForm, -Shape.size_cons]
  rw [hfun]
  refine h.congr_fderiv (ContinuousLinearMap.ext fun d => ?_)
  simp only [ContinuousLinearMap.comp_apply, _root_.add_apply, smul_apply, smul_eq_mul,
    PiLp.proj_apply,
    ContinuousLinearMap.coe_fst', ContinuousLinearMap.coe_snd', derivative_apply, nrmJVPCLM_apply,
    rowOf_idxMN, colOf_idxMN, nrmD_apply hP hε, ContinuousLinearEquiv.coe_coe,
    LinearIsometryEquiv.coe_toContinuousLinearEquiv, matVec, Function.comp_def]
  ring

/-! ## Main theorems -/

/-- For positive `ε`, batch normalization along the leading axis is Fréchet differentiable in
`(x, gamma, beta)`, with derivative `derivative`. -/
theorem hasFDerivAt_forwardVec [Shape.WellFormed (.dim channels sSpatial)] {ε : ℝ} (hε : 0 < ε)
    (x : Tensor ℝ (.dim channels sSpatial)) (gamma beta : Tensor ℝ [channels]) :
    HasFDerivAt (forwardVec (channels := channels) (sSpatial := sSpatial) ε)
      (derivative (matVec (toVec x)) (toVec gamma) ε)
      (toVec x, toVec gamma, toVec beta) := by
  rw [forwardVec_eq_closedForm hε]
  exact hasFDerivAt_closedForm (size_pos_of_wellFormed_dim channels) hε
    (toVec x, toVec gamma, toVec beta)

/-- Entries of `jvp` in the flattened matrix are the closed-form row
differential, for positive `ε`. -/
theorem get2_flat_jvp [Shape.WellFormed (.dim channels sSpatial)] {ε : ℝ} (hε : 0 < ε)
    (x dx : Tensor ℝ (.dim channels sSpatial)) (gamma dgamma beta dbeta : Tensor ℝ [channels])
    (c : Fin channels) (p : Fin sSpatial.size) :
    Tensor.get2
        (Tensor.reshape (jvp x dx gamma dgamma beta dbeta ε) _
            (size_flat channels sSpatial)) c p =
      nrmJVP (matVec (toVec x)) ε (matVec (toVec dx)) c p * getScalar gamma c +
        nrm (matVec (toVec x)) ε c p * getScalar dgamma c + getScalar dbeta c := by
  have hP : 0 < sSpatial.size := size_pos_of_wellFormed_dim channels
  have hVε := max_eq_left (rowVar_add_pos hε (matVec (toVec x)) c).le
  rw [jvp_eq_reshape_normalizedJVP]
  simp only [reshape_reshape_self, get2_normalizedJVP, normalizedRows_fst hP,
    normalizedRows_snd hP, get2_div, get2_rowBroadcast, get2_lnCentered, getScalar_div,
    getScalar_full, getScalar_lnStd, rowVarE_flat, get2_flat]
  rw [hVε]
  simp only [nrmJVP, nrm, centered, invStd, Fintype.expect_eq_sum_div_card, Fintype.card_fin,
    div_eq_mul_inv, one_mul]

/-- For positive `ε`, the derivative of BatchNorm applied to packed tangents is
`jvp`. -/
theorem derivative_eq_jvp [Shape.WellFormed (.dim channels sSpatial)] {ε : ℝ} (hε : 0 < ε)
    (x dx : Tensor ℝ (.dim channels sSpatial)) (gamma dgamma beta dbeta : Tensor ℝ [channels]) :
    derivative (matVec (toVec x)) (toVec gamma) ε
        (toVec dx, toVec dgamma, toVec dbeta) =
      toVec (jvp x dx gamma dgamma beta dbeta ε) := by
  apply PiLp.ext
  intro k
  have hRHS : (toVec (jvp x dx gamma dgamma beta dbeta ε)).ofLp k =
      Tensor.get2
        (Tensor.reshape (jvp x dx gamma dgamma beta dbeta ε) _
            (size_flat channels sSpatial)) (rowIdx k) (colIdx k) := by
    rw [← Matmul.toVec_idxMN, ← cast_eq_idxMN, toVec_reshape_apply]
    rfl
  rw [hRHS, get2_flat_jvp hε, derivative_apply, nrmJVPCLM_apply, rowOf_idxMN, colOf_idxMN,
    getScalar_chan, getScalar_chan, getScalar_chan]
  rfl

/-- The Fréchet derivative of batch normalization along the leading axis in `(x, gamma, beta)`
applied to a tangent triple is `jvp`, for positive `ε`. -/
theorem fderiv_forwardVec_eq_jvp [Shape.WellFormed (.dim channels sSpatial)] {ε : ℝ}
    (hε : 0 < ε) (x dx : Tensor ℝ (.dim channels sSpatial))
    (gamma dgamma beta dbeta : Tensor ℝ [channels]) :
    fderiv ℝ (forwardVec (channels := channels) (sSpatial := sSpatial) ε)
        (toVec x, toVec gamma, toVec beta)
        (toVec dx, toVec dgamma, toVec dbeta) =
      toVec (jvp x dx gamma dgamma beta dbeta ε) := by
  rw [(hasFDerivAt_forwardVec hε x gamma beta).fderiv, derivative_eq_jvp hε]

end

end BatchNorm
end TorchLean
