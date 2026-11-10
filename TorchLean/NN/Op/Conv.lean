/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.FDeriv.PullbackTensor
public import TorchLean.Tensor.Movement.BroadcastAdjoint
public import TorchLean.Tensor.Op.Cert
public import TorchLean.Tensor.Op.Table.Conv
public import TorchLean.Tensor.Window

/-!
# The derivatives of `Op.conv` and `Op.convTranspose`

We prove that the stored backward of `Op.conv` and of `Op.convTranspose` is the adjoint of the
derivative of the forward. We expand each forward, a window sum bilinear in the input and the kernel
plus the broadcast bias, and each stored gradient into sums over pairs of coordinates in an index
relation (`Tensor.convRel` and the others), and we show that the relations agree. This gives
`Tensor.inner_conv_eq_inner_convBackwardWeight`, `Tensor.inner_conv_eq_inner_convBackwardInput` and
their `convTranspose` counterparts. With `O` and `I` the kernel's output and input features and `C`
the input's features, we prove the input identity of `conv` where `0 < O / groups` or `O = 0`, the
kernel identity of `convTranspose` where `groups ≤ 1`, `0 < I / groups`, `I = 0` or `C = 0`, and
its input identity where `groups ≤ 1`, `0 < C / groups` with `C / groups = I / groups`, or
`C = 0`. These conditions are sufficient, not exact. The other identities need no hypothesis.

With `Tensor.convCLM` and `Tensor.convTransposeCLM`, the identities give the `fun_prop` rules
`HasPullbackAt.conv`, `HasPullbackAt.convTranspose`, `HasPullbackAt.conv_right` and
`HasPullbackAt.convTranspose_right`. The bias gradient is the reduction `reduceBroadcast` of the
bias broadcast (`Tensor.castShape_reduce_otherAxes_eq_reduceBroadcast`). The certificates cover the
kernel and bias of `conv` and the bias of `convTranspose` at every attribute, the kernel and bias of
`convTranspose` and all three inputs of either operator under the hypotheses above, and all three
inputs on the range that `Op.check` accepts.
-/

@[expose] public section

open scoped RealInnerProductSpace

namespace TorchLean

namespace Shape.Coord

/-- Two coordinates with the same index along every axis `e p` are equal. -/
theorem ext_positions {s : Shape} {n : Nat} (e : Fin n ≃ Fin s.rank) {c c' : s.Coord}
    (h : ∀ p, index c (e p).val = index c' (e p).val) : c = c' :=
  ext_index fun k hk => by
    have := h (e.symm ⟨k, hk⟩)
    rwa [Equiv.apply_symm_apply] at this

/-- A coordinate of `List.ofFn g` exists with any indices below `g`. -/
theorem exists_ofFn {n : Nat} (g : Fin n → Nat) (f : Fin n → Nat) (h : ∀ k, f k < g k) :
    ∃ c : Coord (List.ofFn g), ∀ k : Fin n, index c k.val = f k := by
  have hr : ∀ d : Fin (Shape.rank (List.ofFn g)), d.val < n := fun d => by
    simpa [Shape.rank_eq_length] using d.isLt
  refine ⟨ofFun fun d => ⟨f ⟨d.val, hr d⟩, ?_⟩, fun k => ?_⟩
  · simpa [Shape.extent, List.getElem_ofFn] using h ⟨d.val, hr d⟩
  · rw [index_ofFun, dite_eq_left (by simp [Shape.rank_eq_length])]

/-- A coordinate of the spatial positions `e` of a shape exists with the spatial indices of `c`. -/
private theorem exists_spatial {s : Shape} {r : Nat} (e : Fin (r + 2) ≃ Fin s.rank) (c : s.Coord) :
    ∃ m : Coord (List.ofFn fun i : Fin r => s.extent (e i.succ.succ)),
      ∀ k : Fin r, index m k.val = index c (e k.succ.succ).val :=
  exists_ofFn _ _ fun j => index_lt c (e j.succ.succ).isLt

end Shape.Coord

/-- A statement about every position of `Fin (r + 2)` splits into the two feature positions and
the spatial positions. -/
private theorem forall_fin_add_two {r : Nat} {P : Fin (r + 2) → Prop} :
    (∀ p, P p) ↔ P ⟨0, by omega⟩ ∧ P ⟨1, by omega⟩ ∧ ∀ j : Fin r, P ⟨j.val + 2, by omega⟩ := by
  refine ⟨fun h => ⟨h _, h _, fun j => h _⟩, fun ⟨h0, h1, h2⟩ p => ?_⟩
  obtain ⟨p, hp⟩ := p
  match p, hp with
  | 0, _ => exact h0
  | 1, _ => exact h1
  | j + 2, hj => exact h2 ⟨j, by omega⟩

namespace Tensor

variable {s t : Shape} {r : Nat}

/-- A sum of an indicator over a type on which the indicated set has at most one point is the
indicator of its nonemptiness. -/
private theorem sum_ite_exists {ι : Type*} [Fintype ι] (P : ι → Prop) [DecidablePred P]
    {_ : Decidable (∃ i, P i)} (a : ℝ) (h : ∀ i j, P i → P j → i = j) :
    ∑ i, (if P i then a else 0) = if ∃ i, P i then a else 0 := by
  convert Fintype.sum_ite_eq_ite_exists P h a

/-- A left fold adding a function of the coordinates of `W` in row-major order is the sum of the
function. -/
theorem foldl_add_unlinearize {W : Shape} (F : W.Coord → ℝ) :
    (List.finRange (Shape.size W)).foldl (fun acc i => acc + F (Shape.Coord.unlinearize i)) 0 =
      ∑ k : W.Coord, F k := by
  rw [List.finRange_foldl_add_eq_add_sum, zero_add]
  exact Equiv.sum_comp (Shape.Coord.equivFin W).symm F

/-- An entry of a real tensor is the sum of the entries at the coordinates with its indices along
every axis `e p`. -/
theorem apply_eq_sum_ite {n : Nat} (x : Tensor ℝ s) (e : Fin n ≃ Fin s.rank) (c₀ : s.Coord) :
    x c₀ = ∑ c, if (∀ p, Shape.Coord.index c (e p).val = Shape.Coord.index c₀ (e p).val) then
      x c else 0 := by
  rw [Finset.sum_eq_single c₀
    (fun c _ hc => ite_eq_right fun h => hc (Shape.Coord.ext_positions e h)) (by simp),
    ite_eq_left fun _ => rfl]

/-- `getOrZero` is the sum of the entries at the coordinates with the given indices. -/
theorem getOrZero_eq_sum {n : Nat} (x : Tensor ℝ s) (e : Fin n ≃ Fin s.rank) (index : Fin n → Nat) :
    getOrZero x e index = ∑ c, if (∀ p, Shape.Coord.index c (e p).val = index p) then
      x c else 0 := by
  unfold getOrZero
  split_ifs with h
  · rw [apply_eq_sum_ite x e]
    simp only [Shape.Coord.index_ofPositions]
  · refine (Finset.sum_eq_zero fun c _ => ite_eq_right fun hc => h fun p => ?_).symm
    rw [← hc p]
    exact Shape.Coord.index_lt c (e p).isLt

/-- A left fold adding `G` over the flat indices of `W` is the sum of `G` at the linearized
coordinates of `W`. -/
private theorem foldl_add_eq_sum_linearize {W : Shape} (G : Fin (Shape.size W) → ℝ) :
    (List.finRange (Shape.size W)).foldl (fun acc i => acc + G i) 0 =
      ∑ k : W.Coord, G (Shape.Coord.linearize k) := by
  rw [List.finRange_foldl_add_eq_add_sum, zero_add]
  exact (Equiv.sum_comp (Shape.Coord.equivFin W) G).symm

open Classical in
/-- A sum over the cells `q` of guarded products of the `getOrZero` reads of `u` at `I q` and of
`v` at `J q` is the sum over the pairs of coordinates in `R`, the pairs some cell reads, when the
two reads determine the cell. -/
private theorem sum_ite_getOrZero_mul {W : Shape} {n m : Nat} {u : Tensor ℝ s} {v : Tensor ℝ t}
    {e₁ : Fin n ≃ Fin s.rank} {e₂ : Fin m ≃ Fin t.rank} {P : W.Coord → Prop} [DecidablePred P]
    {I : W.Coord → Fin n → Nat} {J : W.Coord → Fin m → Nat} {R : s.Coord → t.Coord → Prop}
    (hR : ∀ a b, R a b ↔ ∃ q, (P q ∧ ∀ p, Shape.Coord.index a (e₁ p).val = I q p) ∧
      ∀ p, Shape.Coord.index b (e₂ p).val = J q p)
    (h : ∀ q q', (∀ p, I q p = I q' p) → (∀ p, J q p = J q' p) → q = q') :
    ∑ q, (if P q then getOrZero u e₁ (I q) * getOrZero v e₂ (J q) else 0) =
      ∑ a, ∑ b, if R a b then u a * v b else 0 := by
  have hq : ∀ q, (if P q then getOrZero u e₁ (I q) * getOrZero v e₂ (J q) else 0) =
      ∑ a, ∑ b, if (P q ∧ ∀ p, Shape.Coord.index a (e₁ p).val = I q p) ∧
        ∀ p, Shape.Coord.index b (e₂ p).val = J q p then u a * v b else 0 := fun q => by
    rw [getOrZero_eq_sum u, getOrZero_eq_sum v]
    by_cases hP : P q
    · simp only [hP, ↓reduceIte, true_and, Finset.sum_mul_sum, ite_zero_mul_ite_zero]
    · simp only [hP, ↓reduceIte, false_and, Finset.sum_const_zero]
  simp only [hq, hR]
  rw [Finset.sum_comm]
  refine Finset.sum_congr rfl fun a _ => ?_
  rw [Finset.sum_comm]
  exact Finset.sum_congr rfl fun b _ => sum_ite_exists _ _ fun q q' hq hq' =>
    h q q' (fun p => (hq.1.2 p).symm.trans (hq'.1.2 p)) fun p => (hq.2 p).symm.trans (hq'.2 p)

/-- `convInput` is the sum of the entries at the input coordinates the window cell reads. -/
theorem convInput_eq_sum (x : Tensor ℝ s) (dims : Shape.ConvDims s t r)
    (stride padLow padHigh inputDilation dilation : Fin r → Nat) (groups batchGroups : Nat)
    (o : (s.conv t dims stride padLow padHigh inputDilation dilation batchGroups).Coord)
    (k : (t.convWindow dims).Coord) :
    convInput x dims stride padLow padHigh inputDilation dilation groups batchGroups o k =
      ∑ c, if (∀ p, dims.inputAligned stride padLow padHigh inputDilation dilation batchGroups
          o k p = true) ∧ ∀ p, Shape.Coord.index c (dims.input p).val =
            dims.inputIndex stride padLow padHigh inputDilation dilation groups batchGroups o k p
        then x c else 0 := by
  unfold convInput
  split_ifs with h
  · rw [apply_eq_sum_ite x dims.input]
    simp only [Shape.Coord.index_ofPositions]
    exact Finset.sum_congr rfl fun c _ =>
      if_congr ⟨fun hc => ⟨fun p => (h p).1, hc⟩, fun hc => hc.2⟩ rfl rfl
  · refine (Finset.sum_eq_zero fun c _ => ite_eq_right fun hc => h fun p => ⟨hc.1 p, ?_⟩).symm
    rw [← hc.2 p]
    exact Shape.Coord.index_lt c (dims.input p).isLt

/-- The relation of `conv` without input dilation or batch groups: the output coordinate `o`
reads the input coordinate `c` against the kernel coordinate `w` through a window cell. -/
def convRel (dims : Shape.ConvDims s t r) (stride padLow padHigh dilation : Fin r → Nat)
    (groups : Nat) (o : (s.conv t dims stride padLow padHigh (fun _ => 1) dilation 1).Coord)
    (c : s.Coord) (w : t.Coord) : Prop :=
  ∃ k : (t.convWindow dims).Coord,
    ((∀ p, dims.inputAligned stride padLow padHigh (fun _ => 1) dilation 1 o k p = true) ∧
      ∀ p, Shape.Coord.index c (dims.input p).val =
        dims.inputIndex stride padLow padHigh (fun _ => 1) dilation groups 1 o k p) ∧
    ∀ p, Shape.Coord.index w (dims.kernel p).val =
      Shape.Coord.index (dims.kernelCoord (dims.outFeature o) k) (dims.kernel p).val

/-- The window of `conv` has rank `r + 1`. -/
theorem rank_convWindow (dims : Shape.ConvDims s t r) : (t.convWindow dims).rank = r + 1 := by
  simp [Shape.convWindow, Shape.rank_eq_length]

open Classical in
/-- An entry of `conv` is the sum over the input and kernel coordinates in `convRel`. -/
theorem conv_eq_sum (x : Tensor ℝ s) (w : Tensor ℝ t) (dims : Shape.ConvDims s t r)
    (stride padLow padHigh dilation : Fin r → Nat) (groups : Nat)
    (o : (s.conv t dims stride padLow padHigh (fun _ => 1) dilation 1).Coord) :
    conv x w dims stride padLow padHigh (fun _ => 1) dilation groups 1 o =
      ∑ c, ∑ w', if convRel dims stride padLow padHigh dilation groups o c w' then
        x c * w w' else 0 := by
  rw [conv_apply, foldl_add_unlinearize (fun k => convInput x dims stride padLow padHigh
    (fun _ => 1) dilation groups 1 o k * w (dims.kernelCoord (dims.outFeature o) k))]
  have hk : ∀ k, convInput x dims stride padLow padHigh (fun _ => 1) dilation groups 1 o k *
      w (dims.kernelCoord (dims.outFeature o) k) = ∑ c, ∑ w', if
        ((∀ p, dims.inputAligned stride padLow padHigh (fun _ => 1) dilation 1 o k p = true) ∧
          ∀ p, Shape.Coord.index c (dims.input p).val =
            dims.inputIndex stride padLow padHigh (fun _ => 1) dilation groups 1 o k p) ∧
        ∀ p, Shape.Coord.index w' (dims.kernel p).val =
          Shape.Coord.index (dims.kernelCoord (dims.outFeature o) k) (dims.kernel p).val
        then x c * w w' else 0 := fun k => by
    rw [convInput_eq_sum, apply_eq_sum_ite w dims.kernel (dims.kernelCoord _ k), Finset.sum_mul_sum]
    simp only [ite_zero_mul_ite_zero]
  simp only [hk]
  rw [Finset.sum_comm]
  refine Finset.sum_congr rfl fun c _ => ?_
  rw [Finset.sum_comm]
  refine Finset.sum_congr rfl fun w' _ => ?_
  refine sum_ite_exists _ _ fun k k' hk hk' => Shape.Coord.ext_index fun q hq => ?_
  rw [rank_convWindow] at hq
  have h1 := hk.2 ⟨q + 1, by omega⟩
  have h2 := hk'.2 ⟨q + 1, by omega⟩
  simp only [Shape.ConvDims.kernelCoord, Shape.Coord.index_ofPositions] at h1 h2
  omega

/-- The second successor of `j` is the position `j + 2`. -/
private theorem fin_succ_succ (j : Fin r) : j.succ.succ = (⟨j.val + 2, by omega⟩ : Fin (r + 2)) :=
  rfl

/-- The axes of the output of `conv` have the numbers of the input's. -/
@[simp] theorem _root_.TorchLean.Shape.ConvDims.val_output (dims : Shape.ConvDims s t r)
    (stride padLow padHigh inputDilation dilation : Fin r → Nat) (batchGroups : Nat)
    (p : Fin (r + 2)) :
    (dims.output stride padLow padHigh inputDilation dilation batchGroups p).val =
      (dims.input p).val :=
  rfl

open Shape.ConvDims (val_output)

/-- Three nested sums in reverse order. -/
private theorem sum_comm_three {A B C : Type*} [Fintype A] [Fintype B] [Fintype C]
    (F : A → B → C → ℝ) : ∑ a, ∑ b, ∑ c, F a b c = ∑ c, ∑ b, ∑ a, F a b c := by
  rw [Finset.sum_comm, Finset.sum_congr rfl fun b _ => Finset.sum_comm, Finset.sum_comm]

/-- The relation of the weight gradient of `conv`: at the kernel coordinate `w`, some cell of
the batch and output window reads the input coordinate `c` against the output coordinate `o`. -/
def convBackwardWeightRel (dims : Shape.ConvDims s t r)
    (stride padLow padHigh dilation : Fin r → Nat) (groups : Nat) (w : t.Coord) (c : s.Coord)
    (o : (s.conv t dims stride padLow padHigh (fun _ => 1) dilation 1).Coord) : Prop :=
  ∃ q : (s.convBackwardWeightWindow t dims stride padLow padHigh dilation).Coord,
    (dims.backwardWeightAbovePadLow stride padLow padHigh dilation w q = true ∧
      ∀ p, Shape.Coord.index c (dims.input p).val =
        dims.backwardWeightInputIndex stride padLow padHigh dilation groups w q p) ∧
    ∀ p, Shape.Coord.index o (dims.output stride padLow padHigh (fun _ => 1) dilation 1 p).val =
      dims.backwardWeightOutputIndex stride padLow padHigh dilation w q p

open Classical in
/-- An entry of the weight gradient of `conv` is the sum over the input and output coordinates in
`convBackwardWeightRel`. -/
theorem convBackwardWeight_eq_sum (x : Tensor ℝ s) (dims : Shape.ConvDims s t r)
    (stride padLow padHigh dilation : Fin r → Nat) (groups : Nat)
    (δ : Tensor ℝ (s.conv t dims stride padLow padHigh (fun _ => 1) dilation 1)) (w : t.Coord) :
    convBackwardWeight x dims stride padLow padHigh dilation groups δ w =
      ∑ c, ∑ o, if convBackwardWeightRel dims stride padLow padHigh dilation groups w c o then
        x c * δ o else 0 := by
  rw [convBackwardWeight_apply, foldl_add_eq_sum_linearize]
  simp only [Shape.Coord.unlinearize_linearize, ite_zero_mul]
  refine sum_ite_getOrZero_mul (fun _ _ => Iff.rfl) fun q q' _ h =>
    Shape.Coord.ext_index fun n hn => ?_
  have hn' : n < r + 1 := by simpa [Shape.convBackwardWeightWindow, Shape.rank_eq_length] using hn
  match n, hn' with
  | 0, _ =>
    have h1 := h ⟨0, by omega⟩
    simp only [Shape.ConvDims.backwardWeightOutputIndex] at h1
    show q.1.val = q'.1.val
    omega
  | j + 1, hj =>
    have h1 := h ⟨j + 2, by omega⟩
    simp only [Shape.ConvDims.backwardWeightOutputIndex] at h1
    omega

/-- The relations of `conv` and of its weight gradient agree. -/
theorem convRel_iff_convBackwardWeightRel (dims : Shape.ConvDims s t r)
    (stride padLow padHigh dilation : Fin r → Nat) (groups : Nat)
    (o : (s.conv t dims stride padLow padHigh (fun _ => 1) dilation 1).Coord) (c : s.Coord)
    (w : t.Coord) :
    convRel dims stride padLow padHigh dilation groups o c w ↔
      convBackwardWeightRel dims stride padLow padHigh dilation groups w c o := by
  have ho := fun p => Shape.index_conv_lt dims o p
  have ho1 : Shape.Coord.index o (dims.input ⟨1, by omega⟩).val <
      t.extent (dims.kernel ⟨0, by omega⟩) := ho ⟨1, by omega⟩
  constructor
  · rintro ⟨k, ⟨hal, hc⟩, hw⟩
    obtain ⟨q', hq'⟩ := Shape.Coord.exists_ofFn
      (fun i : Fin r => Shape.convExtent s t dims stride padLow padHigh (fun _ => 1) dilation 1
        i.succ.succ) (fun j => Shape.Coord.index o (dims.input j.succ.succ).val)
      (fun j => ho j.succ.succ)
    let q : (s.convBackwardWeightWindow t dims stride padLow padHigh dilation).Coord :=
      (⟨Shape.Coord.index o (dims.input ⟨0, by omega⟩).val, ho ⟨0, by omega⟩⟩, q')
    have hq : ∀ j : Fin r, Shape.Coord.index q (j.val + 1) =
        Shape.Coord.index o (dims.input ⟨j.val + 2, by omega⟩).val := fun j => hq' j
    obtain ⟨hw0, hw1⟩ := And.intro (hw ⟨0, by omega⟩) (hw ⟨1, by omega⟩)
    have hwj : ∀ j : Fin r, Shape.Coord.index w (dims.kernel ⟨j.val + 2, by omega⟩).val =
        Shape.Coord.index k (j.val + 1) := fun j => by
      have := hw ⟨j.val + 2, by omega⟩
      simpa only [Shape.ConvDims.kernelCoord, Shape.Coord.index_ofPositions] using this
    simp only [Shape.ConvDims.kernelCoord, Shape.Coord.index_ofPositions,
      Shape.ConvDims.outFeature] at hw0 hw1
    refine ⟨q, ⟨?_, ?_⟩, ?_⟩
    · simp only [Shape.ConvDims.backwardWeightAbovePadLow, decide_eq_true_eq, fin_succ_succ]
      intro j
      have := hal ⟨j.val + 2, by omega⟩
      simp only [Shape.ConvDims.inputAligned, Bool.and_eq_true, decide_eq_true_eq, Fin.eta] at this
      rw [hq j, hwj j]
      exact this.1
    · refine forall_fin_add_two.mpr ⟨?_, ?_, fun j => ?_⟩
      · simpa only [Shape.ConvDims.inputIndex, Shape.ConvDims.inputOffset, Nat.div_one,
          Nat.div_eq_of_lt ho1, zero_mul, zero_add, Shape.ConvDims.backwardWeightInputIndex]
          using hc ⟨0, by omega⟩
      · simpa only [Shape.ConvDims.inputIndex, Shape.ConvDims.inputOffset,
          Shape.ConvDims.backwardWeightInputIndex, hw0, hw1] using hc ⟨1, by omega⟩
      · simpa only [Shape.ConvDims.inputIndex, Shape.ConvDims.inputOffset, Nat.div_one, Fin.eta,
          Shape.ConvDims.backwardWeightInputIndex, hq j, hwj j] using hc ⟨j.val + 2, by omega⟩
    · refine forall_fin_add_two.mpr ⟨?_, ?_, fun j => ?_⟩
      · simp only [Shape.ConvDims.backwardWeightOutputIndex, val_output]
        rfl
      · simp only [Shape.ConvDims.backwardWeightOutputIndex, val_output, hw0]
      · simp only [Shape.ConvDims.backwardWeightOutputIndex, val_output, hq j]
  · rintro ⟨q, ⟨hapl, hc⟩, ho'⟩
    obtain ⟨k, hk⟩ : ∃ k : (t.convWindow dims).Coord, ∀ p : Fin (r + 1),
        Shape.Coord.index k p.val = Shape.Coord.index w (dims.kernel p.succ).val :=
      Shape.Coord.exists_ofFn _ _ fun p => Shape.Coord.index_lt w (dims.kernel p.succ).isLt
    have hk0 : Shape.Coord.index k 0 = Shape.Coord.index w (dims.kernel ⟨1, by omega⟩).val :=
      hk ⟨0, by omega⟩
    have hkj : ∀ j : Fin r, Shape.Coord.index k (j.val + 1) =
        Shape.Coord.index w (dims.kernel ⟨j.val + 2, by omega⟩).val := fun j =>
      hk ⟨j.val + 1, by omega⟩
    obtain ⟨ho0, ho1'⟩ := And.intro (ho' ⟨0, by omega⟩) (ho' ⟨1, by omega⟩)
    have hoj : ∀ j : Fin r, Shape.Coord.index o (dims.input ⟨j.val + 2, by omega⟩).val =
        Shape.Coord.index q (j.val + 1) := fun j => by
      have := ho' ⟨j.val + 2, by omega⟩
      simpa only [Shape.ConvDims.backwardWeightOutputIndex, val_output] using this
    simp only [Shape.ConvDims.backwardWeightOutputIndex, val_output] at ho0 ho1'
    simp only [Shape.ConvDims.backwardWeightAbovePadLow, decide_eq_true_eq, fin_succ_succ] at hapl
    refine ⟨k, ⟨?_, ?_⟩, ?_⟩
    · refine forall_fin_add_two.mpr ⟨rfl, rfl, fun j => ?_⟩
      simp only [Shape.ConvDims.inputAligned, Bool.and_eq_true, decide_eq_true_eq, Fin.eta,
        Nat.mod_one, and_true, hoj j, hkj j]
      exact hapl j
    · refine forall_fin_add_two.mpr ⟨?_, ?_, fun j => ?_⟩
      · simpa only [Shape.ConvDims.backwardWeightInputIndex, Shape.ConvDims.inputIndex,
          Shape.ConvDims.inputOffset, Nat.div_one, Nat.div_eq_of_lt ho1, zero_mul, zero_add, ho0]
          using hc ⟨0, by omega⟩
      · simpa only [Shape.ConvDims.backwardWeightInputIndex, Shape.ConvDims.inputIndex,
          Shape.ConvDims.inputOffset, ho1', hk0] using hc ⟨1, by omega⟩
      · simpa only [Shape.ConvDims.backwardWeightInputIndex, Fin.eta, Shape.ConvDims.inputIndex,
          Shape.ConvDims.inputOffset, Nat.div_one, hoj j, hkj j] using hc ⟨j.val + 2, by omega⟩
    · refine forall_fin_add_two.mpr ⟨?_, ?_, fun j => ?_⟩
      · simp only [Shape.ConvDims.kernelCoord, Shape.Coord.index_ofPositions,
          Shape.ConvDims.outFeature, ho1']
      · simp only [Shape.ConvDims.kernelCoord, Shape.Coord.index_ofPositions, hk0]
      · simp only [Shape.ConvDims.kernelCoord, Shape.Coord.index_ofPositions, hkj j]

/-- The weight gradient of `conv` is the transpose of `conv` in its kernel. -/
theorem inner_conv_eq_inner_convBackwardWeight (x : Tensor ℝ s) (w : Tensor ℝ t)
    (dims : Shape.ConvDims s t r) (stride padLow padHigh dilation : Fin r → Nat) (groups : Nat)
    (δ : Tensor ℝ (s.conv t dims stride padLow padHigh (fun _ => 1) dilation 1)) :
    ⟪conv x w dims stride padLow padHigh (fun _ => 1) dilation groups 1, δ⟫ =
      ⟪w, convBackwardWeight x dims stride padLow padHigh dilation groups δ⟫ := by
  rw [Tensor.inner_eq_sum, Tensor.inner_eq_sum]
  simp only [conv_eq_sum, convBackwardWeight_eq_sum, Finset.sum_mul, Finset.mul_sum]
  refine Eq.trans ?_ (sum_comm_three _).symm
  refine Finset.sum_congr rfl fun o _ => Finset.sum_congr rfl fun c _ =>
    Finset.sum_congr rfl fun w' _ => ?_
  rw [convRel_iff_convBackwardWeightRel]
  split_ifs <;> ring

/-- A nonzero sliding window output extent has a nonzero stride. -/
private theorem stride_ne_zero_of_lt {n k st d pl ph i : Nat}
    (h : i < Shape.slidingWindowOutDimDilated n k st d pl ph) : st ≠ 0 := by
  rintro rfl
  simp [Shape.slidingWindowOutDimDilated] at h

/-- The quotient of `q * n + b` by `n` is `q` for `b < n`. -/
private theorem div_eq_of_eq_mul_add {a q n b : Nat} (h : a = q * n + b) (hb : b < n) :
    a / n = q := by
  subst h
  rw [Nat.add_comm, Nat.add_mul_div_right _ _ (by omega), Nat.div_eq_of_lt hb, Nat.zero_add]

/-- The remainder of `q * n + b` modulo `n` is `b` for `b < n`. -/
private theorem mod_eq_of_eq_mul_add {a q n b : Nat} (h : a = q * n + b) (hb : b < n) :
    a % n = b := by
  subst h
  rw [Nat.add_comm, Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt hb]

/-- The relation of the input gradient of `conv`: at the input coordinate `c`, some cell of the
window of the group's output features and the kernel's spatial axes reads the output coordinate
`o` against the kernel coordinate `w`. -/
def convBackwardInputRel (dims : Shape.ConvDims s t r)
    (stride padLow padHigh dilation : Fin r → Nat) (groups : Nat) (c : s.Coord)
    (o : (s.conv t dims stride padLow padHigh (fun _ => 1) dilation 1).Coord) (w : t.Coord) :
    Prop :=
  ∃ m : (t.convBackwardInputWindow dims groups).Coord,
    (dims.backwardInputAligned stride padLow dilation groups c m = true ∧
      ∀ p, Shape.Coord.index o (dims.output stride padLow padHigh (fun _ => 1) dilation 1 p).val =
        dims.backwardInputOutputIndex stride padLow dilation groups c m p) ∧
    ∀ p, Shape.Coord.index w (dims.kernel p).val = dims.backwardInputKernelIndex groups c m p

open Classical in
/-- An entry of the input gradient of `conv` is the sum over the output and kernel coordinates in
`convBackwardInputRel`. -/
theorem convBackwardInput_eq_sum (w : Tensor ℝ t) (dims : Shape.ConvDims s t r)
    (stride padLow padHigh dilation : Fin r → Nat) (groups : Nat)
    (δ : Tensor ℝ (s.conv t dims stride padLow padHigh (fun _ => 1) dilation 1)) (c : s.Coord) :
    convBackwardInput w dims stride padLow padHigh dilation groups δ c =
      ∑ o, ∑ w', if convBackwardInputRel dims stride padLow padHigh dilation groups c o w' then
        δ o * w w' else 0 := by
  simp only [convBackwardInput_apply, ← add_ite, foldl_add_eq_sum_linearize,
    Shape.Coord.unlinearize_linearize]
  refine sum_ite_getOrZero_mul (fun _ _ => Iff.rfl) fun m m' _ h =>
    Shape.Coord.ext_index fun n hn => ?_
  have hn' : n < r + 1 := by simpa [Shape.convBackwardInputWindow, Shape.rank_eq_length] using hn
  match n, hn' with
  | 0, _ =>
    have h1 := h ⟨0, by omega⟩
    simp only [Shape.ConvDims.backwardInputKernelIndex] at h1
    show m.1.val = m'.1.val
    omega
  | j + 1, hj =>
    have h1 := h ⟨j + 2, by omega⟩
    simp only [Shape.ConvDims.backwardInputKernelIndex] at h1
    omega

/-- Where `groups` is positive and at most the kernel's output features, or the kernel has no
output features, the relations of `conv` and of its input gradient agree. -/
theorem convRel_iff_convBackwardInputRel (dims : Shape.ConvDims s t r)
    (stride padLow padHigh dilation : Fin r → Nat) (groups : Nat)
    (hO : 0 < t.extent (dims.kernel ⟨0, by omega⟩) / groups ∨
      t.extent (dims.kernel ⟨0, by omega⟩) = 0)
    (o : (s.conv t dims stride padLow padHigh (fun _ => 1) dilation 1).Coord) (c : s.Coord)
    (w : t.Coord) :
    convRel dims stride padLow padHigh dilation groups o c w ↔
      convBackwardInputRel dims stride padLow padHigh dilation groups c o w := by
  have ho := fun p => Shape.index_conv_lt dims o p
  have ho1 : Shape.Coord.index o (dims.input ⟨1, by omega⟩).val <
      t.extent (dims.kernel ⟨0, by omega⟩) := ho ⟨1, by omega⟩
  have hst : ∀ j : Fin r, stride j ≠ 0 := fun j => stride_ne_zero_of_lt (ho ⟨j.val + 2, by omega⟩)
  constructor
  · rintro ⟨k, ⟨hal, hc⟩, hw⟩
    have hO : 0 < t.extent (dims.kernel ⟨0, by omega⟩) / groups := hO.resolve_right (by omega)
    obtain ⟨m', hm'⟩ := Shape.Coord.exists_spatial dims.kernel w
    let m : (t.convBackwardInputWindow dims groups).Coord :=
      (⟨Shape.Coord.index o (dims.input ⟨1, by omega⟩).val %
        (t.extent (dims.kernel ⟨0, by omega⟩) / groups), Nat.mod_lt _ hO⟩, m')
    have hm : ∀ j : Fin r, Shape.Coord.index m (j.val + 1) =
        Shape.Coord.index w (dims.kernel ⟨j.val + 2, by omega⟩).val := fun j => hm' j
    obtain ⟨hw0, hw1⟩ := And.intro (hw ⟨0, by omega⟩) (hw ⟨1, by omega⟩)
    have hwj : ∀ j : Fin r, Shape.Coord.index w (dims.kernel ⟨j.val + 2, by omega⟩).val =
        Shape.Coord.index k (j.val + 1) := fun j => by
      have := hw ⟨j.val + 2, by omega⟩
      simpa only [Shape.ConvDims.kernelCoord, Shape.Coord.index_ofPositions] using this
    simp only [Shape.ConvDims.kernelCoord, Shape.Coord.index_ofPositions,
      Shape.ConvDims.outFeature] at hw0 hw1
    have hk0 : Shape.Coord.index k 0 < t.extent (dims.kernel ⟨1, by omega⟩) :=
      hw1 ▸ Shape.Coord.index_lt w (dims.kernel ⟨1, by omega⟩).isLt
    obtain ⟨hc0, hc1⟩ := And.intro (hc ⟨0, by omega⟩) (hc ⟨1, by omega⟩)
    have hcj : ∀ j : Fin r, Shape.Coord.index c (dims.input ⟨j.val + 2, by omega⟩).val =
        Shape.Coord.index o (dims.input ⟨j.val + 2, by omega⟩).val * stride j +
          Shape.Coord.index k (j.val + 1) * dilation j - padLow j := fun j => by
      simpa only [Shape.ConvDims.inputIndex, Shape.ConvDims.inputOffset, Nat.div_one,
        Fin.eta] using hc ⟨j.val + 2, by omega⟩
    have halj : ∀ j : Fin r, padLow j ≤
        Shape.Coord.index o (dims.input ⟨j.val + 2, by omega⟩).val * stride j +
          Shape.Coord.index k (j.val + 1) * dilation j := fun j => by
      have := hal ⟨j.val + 2, by omega⟩
      simp only [Shape.ConvDims.inputAligned, Bool.and_eq_true, decide_eq_true_eq, Fin.eta] at this
      exact this.1
    simp only [Shape.ConvDims.inputIndex, Shape.ConvDims.inputOffset, Nat.div_one,
      Nat.div_eq_of_lt ho1, zero_mul, zero_add] at hc0
    simp only [Shape.ConvDims.inputIndex, Shape.ConvDims.inputOffset] at hc1
    have hdiv := div_eq_of_eq_mul_add hc1 hk0
    have hmod := mod_eq_of_eq_mul_add hc1 hk0
    have hsp : ∀ j : Fin r, Shape.Coord.index c (dims.input ⟨j.val + 2, by omega⟩).val + padLow j -
          Shape.Coord.index w (dims.kernel ⟨j.val + 2, by omega⟩).val * dilation j =
            Shape.Coord.index o (dims.input ⟨j.val + 2, by omega⟩).val * stride j := fun j => by
      rw [hcj j, hwj j]
      have := halj j
      omega
    refine ⟨m, ⟨?_, ?_⟩, ?_⟩
    · simp only [Shape.ConvDims.backwardInputAligned, decide_eq_true_eq, fin_succ_succ]
      intro j
      rw [hm j, hsp j]
      refine ⟨hst j, ?_, Nat.mul_mod_left _ _⟩
      rw [hcj j, hwj j]
      have := halj j
      omega
    · refine forall_fin_add_two.mpr ⟨?_, ?_, fun j => ?_⟩
      · simpa only [Shape.ConvDims.backwardInputOutputIndex, val_output] using hc0.symm
      · simp only [Shape.ConvDims.backwardInputOutputIndex, val_output, hdiv]
        exact (Nat.div_add_mod' _ _).symm
      · simp only [Shape.ConvDims.backwardInputOutputIndex, val_output, Fin.eta, hm j, hsp j]
        exact (Nat.mul_div_cancel _ (Nat.pos_of_ne_zero (hst j))).symm
    · refine forall_fin_add_two.mpr ⟨?_, ?_, fun j => ?_⟩
      · simp only [Shape.ConvDims.backwardInputKernelIndex, hdiv, hw0]
        exact (Nat.div_add_mod' _ _).symm
      · simp only [Shape.ConvDims.backwardInputKernelIndex, hmod, hw1]
      · simp only [Shape.ConvDims.backwardInputKernelIndex, hm j]
  · rintro ⟨m, ⟨hal, ho'⟩, hw⟩
    obtain ⟨k, hk⟩ : ∃ k : (t.convWindow dims).Coord, ∀ p : Fin (r + 1),
        Shape.Coord.index k p.val = Shape.Coord.index w (dims.kernel p.succ).val :=
      Shape.Coord.exists_ofFn _ _ fun p => Shape.Coord.index_lt w (dims.kernel p.succ).isLt
    have hk0 : Shape.Coord.index k 0 = Shape.Coord.index w (dims.kernel ⟨1, by omega⟩).val :=
      hk ⟨0, by omega⟩
    have hkj : ∀ j : Fin r, Shape.Coord.index k (j.val + 1) =
        Shape.Coord.index w (dims.kernel ⟨j.val + 2, by omega⟩).val := fun j =>
      hk ⟨j.val + 1, by omega⟩
    obtain ⟨hw0, hw1⟩ := And.intro (hw ⟨0, by omega⟩) (hw ⟨1, by omega⟩)
    have hwj : ∀ j : Fin r, Shape.Coord.index w (dims.kernel ⟨j.val + 2, by omega⟩).val =
        Shape.Coord.index m (j.val + 1) := fun j => hw ⟨j.val + 2, by omega⟩
    simp only [Shape.ConvDims.backwardInputKernelIndex] at hw0 hw1
    obtain ⟨ho0, ho1'⟩ := And.intro (ho' ⟨0, by omega⟩) (ho' ⟨1, by omega⟩)
    simp only [Shape.ConvDims.backwardInputOutputIndex, val_output] at ho0 ho1'
    have hoj : ∀ j : Fin r, Shape.Coord.index o (dims.input ⟨j.val + 2, by omega⟩).val =
        (Shape.Coord.index c (dims.input ⟨j.val + 2, by omega⟩).val + padLow j -
          Shape.Coord.index m (j.val + 1) * dilation j) / stride j := fun j => by
      have := ho' ⟨j.val + 2, by omega⟩
      simpa only [Shape.ConvDims.backwardInputOutputIndex, val_output, Fin.eta] using this
    simp only [Shape.ConvDims.backwardInputAligned, decide_eq_true_eq, fin_succ_succ] at hal
    have hmul : ∀ j : Fin r, Shape.Coord.index o (dims.input ⟨j.val + 2, by omega⟩).val * stride j =
          Shape.Coord.index c (dims.input ⟨j.val + 2, by omega⟩).val + padLow j -
            Shape.Coord.index m (j.val + 1) * dilation j := fun j => by
      rw [hoj j]
      exact Nat.div_mul_cancel (Nat.dvd_of_mod_eq_zero (hal j).2.2)
    have hdiv := div_eq_of_eq_mul_add ho1' m.1.isLt
    refine ⟨k, ⟨?_, ?_⟩, ?_⟩
    · refine forall_fin_add_two.mpr ⟨rfl, rfl, fun j => ?_⟩
      simp only [Shape.ConvDims.inputAligned, Bool.and_eq_true, decide_eq_true_eq, Fin.eta,
        Nat.mod_one, and_true, hkj j, hwj j, hmul j]
      have := (hal j).2.1
      omega
    · refine forall_fin_add_two.mpr ⟨?_, ?_, fun j => ?_⟩
      · simp only [Shape.ConvDims.inputIndex, Shape.ConvDims.inputOffset, Nat.div_one,
          Nat.div_eq_of_lt ho1, zero_mul, zero_add, ho0]
      · simp only [Shape.ConvDims.inputIndex, Shape.ConvDims.inputOffset, hdiv, hk0, hw1]
        exact (Nat.div_add_mod' _ _).symm
      · simp only [Shape.ConvDims.inputIndex, Shape.ConvDims.inputOffset, Nat.div_one, Fin.eta,
          hkj j, hwj j, hmul j]
        have := (hal j).2.1
        omega
    · refine forall_fin_add_two.mpr ⟨?_, ?_, fun j => ?_⟩
      · simp only [Shape.ConvDims.kernelCoord, Shape.Coord.index_ofPositions,
          Shape.ConvDims.outFeature, ho1', hw0]
      · simp only [Shape.ConvDims.kernelCoord, Shape.Coord.index_ofPositions, hk0]
      · simp only [Shape.ConvDims.kernelCoord, Shape.Coord.index_ofPositions, hkj j]

/-- Where `groups` is positive and at most the kernel's output features, or the kernel has no
output features, the input gradient of `conv` is the transpose of `conv` in its input. -/
theorem inner_conv_eq_inner_convBackwardInput (x : Tensor ℝ s) (w : Tensor ℝ t)
    (dims : Shape.ConvDims s t r) (stride padLow padHigh dilation : Fin r → Nat) (groups : Nat)
    (hO : 0 < t.extent (dims.kernel ⟨0, by omega⟩) / groups ∨
      t.extent (dims.kernel ⟨0, by omega⟩) = 0)
    (δ : Tensor ℝ (s.conv t dims stride padLow padHigh (fun _ => 1) dilation 1)) :
    ⟪conv x w dims stride padLow padHigh (fun _ => 1) dilation groups 1, δ⟫ =
      ⟪x, convBackwardInput w dims stride padLow padHigh dilation groups δ⟫ := by
  rw [Tensor.inner_eq_sum, Tensor.inner_eq_sum]
  simp only [conv_eq_sum, convBackwardInput_eq_sum, Finset.sum_mul, Finset.mul_sum]
  refine Eq.trans Finset.sum_comm ?_
  refine Finset.sum_congr rfl fun c _ => Finset.sum_congr rfl fun o _ =>
    Finset.sum_congr rfl fun w' _ => ?_
  rw [convRel_iff_convBackwardInputRel dims stride padLow padHigh dilation groups hO]
  split_ifs <;> ring

/-- The axes of the output of `convTranspose` have the numbers of the input's. -/
@[simp] theorem _root_.TorchLean.Shape.ConvDims.val_transposeOutput (dims : Shape.ConvDims s t r)
    (stride padLow padHigh outputPadding dilation : Fin r → Nat) (groups : Nat) (p : Fin (r + 2)) :
    (dims.transposeOutput stride padLow padHigh outputPadding dilation groups p).val =
      (dims.input p).val :=
  rfl

open Shape.ConvDims (val_transposeOutput)

/-- A nonzero output extent of a transposed convolution has a nonzero stride. -/
private theorem stride_ne_zero_of_lt_convTransposeOutDim {n k st pl ph op d i : Nat}
    (h : i < Shape.convTransposeOutDim n k st pl ph op d) : st ≠ 0 := by
  rintro rfl
  simp [Shape.convTransposeOutDim] at h

/-- A conditional adding `b` to `a` adds the conditional `b` or zero to `a`. -/
private theorem ite_add_else_self {P : Prop} [Decidable P] (a b : ℝ) :
    (if P then a + b else a) = a + if P then b else 0 := by
  split_ifs <;> simp

/-- `convTransposeInput` is the sum of the entries at the input coordinates the window cell
reads. -/
theorem convTransposeInput_eq_sum (x : Tensor ℝ s) (dims : Shape.ConvDims s t r)
    (stride padLow padHigh outputPadding dilation : Fin r → Nat) (groups : Nat)
    (o : (s.convTranspose t dims stride padLow padHigh outputPadding dilation groups).Coord)
    (k : (s.convTransposeWindow t dims groups).Coord) :
    convTransposeInput x dims stride padLow padHigh outputPadding dilation groups o k =
      ∑ c, if (∀ p, convTransposeAligned dims stride padLow padHigh outputPadding dilation groups
          o k p = true) ∧ ∀ p, Shape.Coord.index c (dims.input p).val =
            convTransposeIndex dims stride padLow padHigh outputPadding dilation groups o k p
        then x c else 0 := by
  unfold convTransposeInput
  split_ifs with h
  · rw [apply_eq_sum_ite x dims.input]
    simp only [Shape.Coord.index_ofPositions]
    exact Finset.sum_congr rfl fun c _ =>
      if_congr ⟨fun hc => ⟨fun p => (h p).1, hc⟩, fun hc => hc.2⟩ rfl rfl
  · refine (Finset.sum_eq_zero fun c _ => ite_eq_right fun hc => h fun p => ⟨hc.1 p, ?_⟩).symm
    rw [← hc.2 p]
    exact Shape.Coord.index_lt c (dims.input p).isLt

/-- The relation of `convTranspose`: the output coordinate `o` reads the input coordinate `c`
against the kernel coordinate `w` through a window cell. -/
def convTransposeRel (dims : Shape.ConvDims s t r)
    (stride padLow padHigh outputPadding dilation : Fin r → Nat) (groups : Nat)
    (o : (s.convTranspose t dims stride padLow padHigh outputPadding dilation groups).Coord)
    (c : s.Coord) (w : t.Coord) : Prop :=
  ∃ k : (s.convTransposeWindow t dims groups).Coord,
    ((∀ p, convTransposeAligned dims stride padLow padHigh outputPadding dilation groups o k p =
        true) ∧
      ∀ p, Shape.Coord.index c (dims.input p).val =
        convTransposeIndex dims stride padLow padHigh outputPadding dilation groups o k p) ∧
    ∀ p, Shape.Coord.index w (dims.kernel p).val =
      Shape.Coord.index (convTransposeKernel dims o k) (dims.kernel p).val

/-- The window of `convTranspose` has rank `r + 1`. -/
theorem rank_convTransposeWindow (dims : Shape.ConvDims s t r) (groups : Nat) :
    (s.convTransposeWindow t dims groups).rank = r + 1 := by
  simp [Shape.convTransposeWindow, Shape.rank_eq_length]

open Classical in
/-- An entry of `convTranspose` is the sum over the input and kernel coordinates in
`convTransposeRel`. -/
theorem convTranspose_eq_sum (x : Tensor ℝ s) (w : Tensor ℝ t) (dims : Shape.ConvDims s t r)
    (stride padLow padHigh outputPadding dilation : Fin r → Nat) (groups : Nat)
    (o : (s.convTranspose t dims stride padLow padHigh outputPadding dilation groups).Coord) :
    convTranspose x w dims stride padLow padHigh outputPadding dilation groups o =
      ∑ c, ∑ w', if convTransposeRel dims stride padLow padHigh outputPadding dilation groups o c
        w' then x c * w w' else 0 := by
  rw [convTranspose_apply, foldl_add_unlinearize (fun k => convTransposeInput x dims stride
    padLow padHigh outputPadding dilation groups o k * w (convTransposeKernel dims o k))]
  have hk : ∀ k, convTransposeInput x dims stride padLow padHigh outputPadding dilation groups o
      k * w (convTransposeKernel dims o k) = ∑ c, ∑ w', if
        ((∀ p, convTransposeAligned dims stride padLow padHigh outputPadding dilation groups o k
            p = true) ∧
          ∀ p, Shape.Coord.index c (dims.input p).val =
            convTransposeIndex dims stride padLow padHigh outputPadding dilation groups o k p) ∧
        ∀ p, Shape.Coord.index w' (dims.kernel p).val =
          Shape.Coord.index (convTransposeKernel dims o k) (dims.kernel p).val
        then x c * w w' else 0 := fun k => by
    rw [convTransposeInput_eq_sum, apply_eq_sum_ite w dims.kernel (convTransposeKernel dims o k),
      Finset.sum_mul_sum]
    simp only [ite_zero_mul_ite_zero]
  simp only [hk]
  rw [Finset.sum_comm]
  refine Finset.sum_congr rfl fun c _ => ?_
  rw [Finset.sum_comm]
  refine Finset.sum_congr rfl fun w' _ => ?_
  refine sum_ite_exists _ _ fun k k' hk hk' => Shape.Coord.ext_index fun n hn => ?_
  rw [rank_convTransposeWindow] at hn
  match n, hn with
  | 0, _ =>
    have h1 := hk.2 ⟨1, by omega⟩
    have h2 := hk'.2 ⟨1, by omega⟩
    simp only [convTransposeKernel, Shape.Coord.index_ofPositions] at h1 h2
    show k.1.val = k'.1.val
    omega
  | j + 1, hj =>
    have h1 := hk.2 ⟨j + 2, by omega⟩
    have h2 := hk'.2 ⟨j + 2, by omega⟩
    simp only [convTransposeKernel, Shape.Coord.index_ofPositions] at h1 h2
    omega

/-- The relation of the input gradient of `convTranspose`: at the input coordinate `c`, some cell
of the window of the kernel's output features and spatial axes reads the kernel coordinate `w`
against the output coordinate `o`. -/
def convTransposeBackwardInputRel (dims : Shape.ConvDims s t r)
    (stride padLow padHigh outputPadding dilation : Fin r → Nat) (groups : Nat) (c : s.Coord)
    (w : t.Coord)
    (o : (s.convTranspose t dims stride padLow padHigh outputPadding dilation groups).Coord) :
    Prop :=
  ∃ m : (t.convTransposeBackwardInputWindow dims).Coord,
    (dims.transposeBackwardInputAbovePadLow stride padLow dilation c m = true ∧
      ∀ p, Shape.Coord.index w (dims.kernel p).val =
        dims.transposeBackwardInputKernelIndex groups c m p) ∧
    ∀ p, Shape.Coord.index o
        (dims.transposeOutput stride padLow padHigh outputPadding dilation groups p).val =
      dims.transposeBackwardInputOutputIndex stride padLow dilation groups c m p

open Classical in
/-- An entry of the input gradient of `convTranspose` is the sum over the kernel and output
coordinates in `convTransposeBackwardInputRel`. -/
theorem convTransposeBackwardInput_eq_sum (w : Tensor ℝ t) (dims : Shape.ConvDims s t r)
    (stride padLow padHigh outputPadding dilation : Fin r → Nat) (groups : Nat)
    (δ : Tensor ℝ (s.convTranspose t dims stride padLow padHigh outputPadding dilation groups))
    (c : s.Coord) :
    convTransposeBackwardInput w dims stride padLow padHigh outputPadding dilation groups δ c =
      ∑ w', ∑ o, if convTransposeBackwardInputRel dims stride padLow padHigh outputPadding
        dilation groups c w' o then w w' * δ o else 0 := by
  simp only [convTransposeBackwardInput_apply, ite_add_else_self, foldl_add_eq_sum_linearize,
    Shape.Coord.unlinearize_linearize]
  refine sum_ite_getOrZero_mul (fun _ _ => Iff.rfl) fun m m' h _ =>
    Shape.Coord.ext_index fun n hn => ?_
  have hn' : n < r + 1 := by
    simpa [Shape.convTransposeBackwardInputWindow, Shape.rank_eq_length] using hn
  match n, hn' with
  | 0, _ =>
    have h1 := h ⟨0, by omega⟩
    simp only [Shape.ConvDims.transposeBackwardInputKernelIndex] at h1
    show m.1.val = m'.1.val
    omega
  | j + 1, hj =>
    have h1 := h ⟨j + 2, by omega⟩
    simp only [Shape.ConvDims.transposeBackwardInputKernelIndex] at h1
    omega

/-- The relation of the weight gradient of `convTranspose`: at the kernel coordinate `w`, some
cell of the batch and input window reads the input coordinate `c` against the output coordinate
`o`. -/
def convTransposeBackwardWeightRel (dims : Shape.ConvDims s t r)
    (stride padLow padHigh outputPadding dilation : Fin r → Nat) (groups : Nat) (w : t.Coord)
    (c : s.Coord)
    (o : (s.convTranspose t dims stride padLow padHigh outputPadding dilation groups).Coord) :
    Prop :=
  ∃ q : (s.convTransposeBackwardWeightWindow dims).Coord,
    (dims.transposeBackwardWeightAbovePadLow stride padLow dilation w q = true ∧
      ∀ p, Shape.Coord.index c (dims.input p).val =
        dims.transposeBackwardWeightInputIndex groups w q p) ∧
    ∀ p, Shape.Coord.index o
        (dims.transposeOutput stride padLow padHigh outputPadding dilation groups p).val =
      dims.transposeBackwardWeightOutputIndex stride padLow dilation groups w q p

open Classical in
/-- An entry of the weight gradient of `convTranspose` is the sum over the input and output
coordinates in `convTransposeBackwardWeightRel`. -/
theorem convTransposeBackwardWeight_eq_sum (x : Tensor ℝ s) (dims : Shape.ConvDims s t r)
    (stride padLow padHigh outputPadding dilation : Fin r → Nat) (groups : Nat)
    (δ : Tensor ℝ (s.convTranspose t dims stride padLow padHigh outputPadding dilation groups))
    (w : t.Coord) :
    convTransposeBackwardWeight x dims stride padLow padHigh outputPadding dilation groups δ w =
      ∑ c, ∑ o, if convTransposeBackwardWeightRel dims stride padLow padHigh outputPadding
        dilation groups w c o then x c * δ o else 0 := by
  simp only [convTransposeBackwardWeight_apply, ite_add_else_self, foldl_add_eq_sum_linearize,
    Shape.Coord.unlinearize_linearize]
  refine sum_ite_getOrZero_mul (fun _ _ => Iff.rfl) fun q q' h _ =>
    Shape.Coord.ext_index fun n hn => ?_
  have hn' : n < r + 1 := by
    simpa [Shape.convTransposeBackwardWeightWindow, Shape.rank_eq_length] using hn
  match n, hn' with
  | 0, _ =>
    have h1 := h ⟨0, by omega⟩
    simp only [Shape.ConvDims.transposeBackwardWeightInputIndex] at h1
    show q.1.val = q'.1.val
    omega
  | j + 1, hj =>
    have h1 := h ⟨j + 2, by omega⟩
    simp only [Shape.ConvDims.transposeBackwardWeightInputIndex] at h1
    omega

/-- Where `groups` is at most one, or the input and the kernel have the same positive number of
input features per group, or the input has no features, the relations of `convTranspose` and of
its input gradient agree. -/
theorem convTransposeRel_iff_convTransposeBackwardInputRel (dims : Shape.ConvDims s t r)
    (stride padLow padHigh outputPadding dilation : Fin r → Nat) (groups : Nat)
    (hC : groups ≤ 1 ∨ (0 < s.extent (dims.input ⟨1, by omega⟩) / groups ∧
        s.extent (dims.input ⟨1, by omega⟩) / groups =
          t.extent (dims.kernel ⟨1, by omega⟩) / groups) ∨
      s.extent (dims.input ⟨1, by omega⟩) = 0)
    (o : (s.convTranspose t dims stride padLow padHigh outputPadding dilation groups).Coord)
    (c : s.Coord) (w : t.Coord) :
    convTransposeRel dims stride padLow padHigh outputPadding dilation groups o c w ↔
      convTransposeBackwardInputRel dims stride padLow padHigh outputPadding dilation groups c w
        o := by
  have ho := fun p => Shape.index_convTranspose_lt dims o p
  have ho1 : Shape.Coord.index o (dims.input ⟨1, by omega⟩).val <
      groups * t.extent (dims.kernel ⟨0, by omega⟩) := ho ⟨1, by omega⟩
  have hst : ∀ j : Fin r, stride j ≠ 0 := fun j =>
    stride_ne_zero_of_lt_convTransposeOutDim (ho ⟨j.val + 2, by omega⟩)
  have hcl : Shape.Coord.index c (dims.input ⟨1, by omega⟩).val <
      s.extent (dims.input ⟨1, by omega⟩) := Shape.Coord.index_lt c _
  have hg : groups = 1 ∨ (0 < s.extent (dims.input ⟨1, by omega⟩) / groups ∧
      s.extent (dims.input ⟨1, by omega⟩) / groups =
        t.extent (dims.kernel ⟨1, by omega⟩) / groups) := by
    rcases hC with h | h | h
    · refine Or.inl (Nat.le_antisymm h (Nat.pos_of_ne_zero fun h0 => ?_))
      simp [h0] at ho1
    · exact Or.inr h
    · exact absurd h (by omega)
  constructor
  · rintro ⟨k, ⟨hal, hc⟩, hw⟩
    have hO : 0 < t.extent (dims.kernel ⟨0, by omega⟩) := Nat.pos_of_lt_mul_left ho1
    obtain ⟨m', hm'⟩ := Shape.Coord.exists_spatial dims.kernel w
    let m : (t.convTransposeBackwardInputWindow dims).Coord :=
      (⟨Shape.Coord.index o (dims.input ⟨1, by omega⟩).val %
        t.extent (dims.kernel ⟨0, by omega⟩), Nat.mod_lt _ hO⟩, m')
    have hm : ∀ j : Fin r, Shape.Coord.index m (j.val + 1) =
        Shape.Coord.index w (dims.kernel ⟨j.val + 2, by omega⟩).val := fun j => hm' j
    obtain ⟨hw0, hw1⟩ := And.intro (hw ⟨0, by omega⟩) (hw ⟨1, by omega⟩)
    have hwj : ∀ j : Fin r, Shape.Coord.index w (dims.kernel ⟨j.val + 2, by omega⟩).val =
        Shape.Coord.index k (j.val + 1) := fun j => by
      simpa only [convTransposeKernel, Shape.Coord.index_ofPositions] using hw ⟨j.val + 2, by omega⟩
    simp only [convTransposeKernel, Shape.Coord.index_ofPositions] at hw0 hw1
    obtain ⟨hc0, hc1⟩ := And.intro (hc ⟨0, by omega⟩) (hc ⟨1, by omega⟩)
    simp only [convTransposeIndex, convTransposeOffset] at hc0 hc1
    have hk0 : k.1.val < s.extent (dims.input ⟨1, by omega⟩) / groups := by
      rcases hg with h1 | ⟨-, h⟩
      · rw [show s.extent (dims.input ⟨1, by omega⟩) / groups =
          s.extent (dims.input ⟨1, by omega⟩) by simp only [h1, Nat.div_one]]
        omega
      · rw [h]
        exact k.1.isLt
    have halj : ∀ j : Fin r, Shape.Coord.index k (j.val + 1) * dilation j ≤
        Shape.Coord.index o (dims.input ⟨j.val + 2, by omega⟩).val + padLow j ∧
        (Shape.Coord.index o (dims.input ⟨j.val + 2, by omega⟩).val + padLow j -
          Shape.Coord.index k (j.val + 1) * dilation j) % stride j = 0 := fun j => by
      have := hal ⟨j.val + 2, by omega⟩
      simp only [convTransposeAligned, Bool.and_eq_true, decide_eq_true_eq] at this
      simpa only [convTransposeOffset, Fin.eta] using this
    have hmul : ∀ j : Fin r, Shape.Coord.index c (dims.input ⟨j.val + 2, by omega⟩).val * stride j =
          Shape.Coord.index o (dims.input ⟨j.val + 2, by omega⟩).val + padLow j -
            Shape.Coord.index k (j.val + 1) * dilation j := fun j => by
      have := hc ⟨j.val + 2, by omega⟩
      simp only [convTransposeIndex, convTransposeOffset, Fin.eta] at this
      rw [this]
      exact Nat.div_mul_cancel (Nat.dvd_of_mod_eq_zero (halj j).2)
    have hdiv := div_eq_of_eq_mul_add hc1 hk0
    have hmod := mod_eq_of_eq_mul_add hc1 hk0
    refine ⟨m, ⟨?_, ?_⟩, ?_⟩
    · simp only [Shape.ConvDims.transposeBackwardInputAbovePadLow, decide_eq_true_eq, fin_succ_succ]
      intro j
      rw [hm j, hwj j, hmul j]
      have := (halj j).1
      omega
    · refine forall_fin_add_two.mpr ⟨?_, ?_, fun j => ?_⟩
      · simpa only [Shape.ConvDims.transposeBackwardInputKernelIndex] using hw0
      · simpa only [Shape.ConvDims.transposeBackwardInputKernelIndex, hdiv, hmod] using hw1
      · simp only [Shape.ConvDims.transposeBackwardInputKernelIndex, hm j]
    · refine forall_fin_add_two.mpr ⟨?_, ?_, fun j => ?_⟩
      · simp only [Shape.ConvDims.transposeBackwardInputOutputIndex, val_transposeOutput]
        exact hc0.symm
      · simp only [Shape.ConvDims.transposeBackwardInputOutputIndex, val_transposeOutput, hdiv]
        exact (Nat.div_add_mod' _ _).symm
      · simp only [Shape.ConvDims.transposeBackwardInputOutputIndex, val_transposeOutput,
          Fin.eta, hm j, hwj j, hmul j]
        have := (halj j).1
        omega
  · rintro ⟨m, ⟨hab, hw⟩, ho'⟩
    obtain ⟨k', hk'⟩ := Shape.Coord.exists_spatial dims.kernel w
    obtain ⟨hw0, hw1⟩ := And.intro (hw ⟨0, by omega⟩) (hw ⟨1, by omega⟩)
    simp only [Shape.ConvDims.transposeBackwardInputKernelIndex] at hw0 hw1
    have hk0 : Shape.Coord.index c (dims.input ⟨1, by omega⟩).val %
        (s.extent (dims.input ⟨1, by omega⟩) / groups) <
          t.extent (dims.kernel ⟨1, by omega⟩) / groups := by
      rcases hg with h1 | ⟨h0, h⟩
      · have : Shape.Coord.index w (dims.kernel ⟨1, by omega⟩).val <
            t.extent (dims.kernel ⟨1, by omega⟩) := Shape.Coord.index_lt w _
        rw [show t.extent (dims.kernel ⟨1, by omega⟩) / groups =
          t.extent (dims.kernel ⟨1, by omega⟩) by simp only [h1, Nat.div_one]] at hw1 ⊢
        omega
      · rw [← h]
        exact Nat.mod_lt _ h0
    let k : (s.convTransposeWindow t dims groups).Coord :=
      (⟨Shape.Coord.index c (dims.input ⟨1, by omega⟩).val %
        (s.extent (dims.input ⟨1, by omega⟩) / groups), hk0⟩, k')
    have hk : ∀ j : Fin r, Shape.Coord.index k (j.val + 1) =
        Shape.Coord.index w (dims.kernel ⟨j.val + 2, by omega⟩).val := fun j => hk' j
    have hwj : ∀ j : Fin r, Shape.Coord.index w (dims.kernel ⟨j.val + 2, by omega⟩).val =
        Shape.Coord.index m (j.val + 1) := fun j => hw ⟨j.val + 2, by omega⟩
    obtain ⟨ho0, ho1'⟩ := And.intro (ho' ⟨0, by omega⟩) (ho' ⟨1, by omega⟩)
    simp only [Shape.ConvDims.transposeBackwardInputOutputIndex, val_transposeOutput] at ho0 ho1'
    have hoj : ∀ j : Fin r, Shape.Coord.index o (dims.input ⟨j.val + 2, by omega⟩).val =
        Shape.Coord.index c (dims.input ⟨j.val + 2, by omega⟩).val * stride j +
          Shape.Coord.index m (j.val + 1) * dilation j - padLow j := fun j => by
      simpa only [Shape.ConvDims.transposeBackwardInputOutputIndex, val_transposeOutput,
        Fin.eta] using ho' ⟨j.val + 2, by omega⟩
    simp only [Shape.ConvDims.transposeBackwardInputAbovePadLow, decide_eq_true_eq,
      fin_succ_succ] at hab
    have hsp : ∀ j : Fin r, Shape.Coord.index o (dims.input ⟨j.val + 2, by omega⟩).val + padLow j -
          Shape.Coord.index m (j.val + 1) * dilation j =
            Shape.Coord.index c (dims.input ⟨j.val + 2, by omega⟩).val * stride j := fun j => by
      rw [hoj j]
      have := hab j
      omega
    have hdiv := div_eq_of_eq_mul_add ho1' m.1.isLt
    have hmod := mod_eq_of_eq_mul_add ho1' m.1.isLt
    refine ⟨k, ⟨?_, ?_⟩, ?_⟩
    · refine forall_fin_add_two.mpr ⟨rfl, rfl, fun j => ?_⟩
      simp only [convTransposeAligned, convTransposeOffset, Bool.and_eq_true, decide_eq_true_eq,
        Fin.eta, hk j, hwj j, hsp j, Nat.mul_mod_left, and_true]
      rw [hoj j]
      have := hab j
      omega
    · refine forall_fin_add_two.mpr ⟨?_, ?_, fun j => ?_⟩
      · simpa only [convTransposeIndex, convTransposeOffset] using ho0.symm
      · simpa only [convTransposeIndex, convTransposeOffset, hdiv] using (Nat.div_add_mod' _ _).symm
      · simp only [convTransposeIndex, convTransposeOffset, Fin.eta, hk j, hwj j, hsp j]
        exact (Nat.mul_div_cancel _ (Nat.pos_of_ne_zero (hst j))).symm
    · refine forall_fin_add_two.mpr ⟨?_, ?_, fun j => ?_⟩
      · simpa only [convTransposeKernel, Shape.Coord.index_ofPositions, hmod] using hw0
      · simpa only [convTransposeKernel, Shape.Coord.index_ofPositions, hdiv] using hw1
      · simp only [convTransposeKernel, Shape.Coord.index_ofPositions, hk j]

/-- Where `groups` is at most one or at most the kernel's input features, or the kernel has no
input features, or the input has no features, the relations of `convTranspose` and of its weight
gradient agree. -/
theorem convTransposeRel_iff_convTransposeBackwardWeightRel (dims : Shape.ConvDims s t r)
    (stride padLow padHigh outputPadding dilation : Fin r → Nat) (groups : Nat)
    (hI : groups ≤ 1 ∨ 0 < t.extent (dims.kernel ⟨1, by omega⟩) / groups ∨
      t.extent (dims.kernel ⟨1, by omega⟩) = 0 ∨ s.extent (dims.input ⟨1, by omega⟩) = 0)
    (o : (s.convTranspose t dims stride padLow padHigh outputPadding dilation groups).Coord)
    (c : s.Coord) (w : t.Coord) :
    convTransposeRel dims stride padLow padHigh outputPadding dilation groups o c w ↔
      convTransposeBackwardWeightRel dims stride padLow padHigh outputPadding dilation groups w c
        o := by
  have ho := fun p => Shape.index_convTranspose_lt dims o p
  have hst : ∀ j : Fin r, stride j ≠ 0 := fun j =>
    stride_ne_zero_of_lt_convTransposeOutDim (ho ⟨j.val + 2, by omega⟩)
  have hG : 0 < t.extent (dims.kernel ⟨1, by omega⟩) / groups := by
    have : Shape.Coord.index w (dims.kernel ⟨1, by omega⟩).val <
        t.extent (dims.kernel ⟨1, by omega⟩) := Shape.Coord.index_lt w _
    have ho1 : Shape.Coord.index o (dims.input ⟨1, by omega⟩).val <
        groups * t.extent (dims.kernel ⟨0, by omega⟩) := ho ⟨1, by omega⟩
    have : Shape.Coord.index c (dims.input ⟨1, by omega⟩).val <
        s.extent (dims.input ⟨1, by omega⟩) := Shape.Coord.index_lt c _
    rcases hI with h | h | h | h
    · rw [Nat.le_antisymm h (Nat.pos_of_ne_zero fun h0 => by simp [h0] at ho1), Nat.div_one]
      omega
    · exact h
    all_goals omega
  have hw0lt := Shape.Coord.index_lt w (dims.kernel ⟨0, by omega⟩).isLt
  constructor
  · rintro ⟨k, ⟨hal, hc⟩, hw⟩
    obtain ⟨q', hq'⟩ := Shape.Coord.exists_spatial dims.input c
    let q : (s.convTransposeBackwardWeightWindow dims).Coord :=
      (⟨Shape.Coord.index o (dims.input ⟨0, by omega⟩).val, ho ⟨0, by omega⟩⟩, q')
    have hq : ∀ j : Fin r, Shape.Coord.index q (j.val + 1) =
        Shape.Coord.index c (dims.input ⟨j.val + 2, by omega⟩).val := fun j => hq' j
    obtain ⟨hw0, hw1⟩ := And.intro (hw ⟨0, by omega⟩) (hw ⟨1, by omega⟩)
    have hwj : ∀ j : Fin r, Shape.Coord.index w (dims.kernel ⟨j.val + 2, by omega⟩).val =
        Shape.Coord.index k (j.val + 1) := fun j => by
      simpa only [convTransposeKernel, Shape.Coord.index_ofPositions] using hw ⟨j.val + 2, by omega⟩
    simp only [convTransposeKernel, Shape.Coord.index_ofPositions] at hw0 hw1
    obtain ⟨hc0, hc1⟩ := And.intro (hc ⟨0, by omega⟩) (hc ⟨1, by omega⟩)
    simp only [convTransposeIndex, convTransposeOffset] at hc0 hc1
    have halj : ∀ j : Fin r, Shape.Coord.index k (j.val + 1) * dilation j ≤
        Shape.Coord.index o (dims.input ⟨j.val + 2, by omega⟩).val + padLow j ∧
        (Shape.Coord.index o (dims.input ⟨j.val + 2, by omega⟩).val + padLow j -
          Shape.Coord.index k (j.val + 1) * dilation j) % stride j = 0 := fun j => by
      have := hal ⟨j.val + 2, by omega⟩
      simp only [convTransposeAligned, Bool.and_eq_true, decide_eq_true_eq] at this
      simpa only [convTransposeOffset, Fin.eta] using this
    have hmul : ∀ j : Fin r, Shape.Coord.index c (dims.input ⟨j.val + 2, by omega⟩).val * stride j =
          Shape.Coord.index o (dims.input ⟨j.val + 2, by omega⟩).val + padLow j -
            Shape.Coord.index k (j.val + 1) * dilation j := fun j => by
      have := hc ⟨j.val + 2, by omega⟩
      simp only [convTransposeIndex, convTransposeOffset, Fin.eta] at this
      rw [this]
      exact Nat.div_mul_cancel (Nat.dvd_of_mod_eq_zero (halj j).2)
    have hdiv := div_eq_of_eq_mul_add hw1 k.1.isLt
    have hmod := mod_eq_of_eq_mul_add hw1 k.1.isLt
    refine ⟨q, ⟨?_, ?_⟩, ?_⟩
    · simp only [Shape.ConvDims.transposeBackwardWeightAbovePadLow, decide_eq_true_eq,
        fin_succ_succ]
      intro j
      rw [hq j, hwj j, hmul j]
      have := (halj j).1
      omega
    · refine forall_fin_add_two.mpr ⟨?_, ?_, fun j => ?_⟩
      · simpa only [Shape.ConvDims.transposeBackwardWeightInputIndex] using hc0
      · simpa only [Shape.ConvDims.transposeBackwardWeightInputIndex, hdiv, hmod] using hc1
      · simp only [Shape.ConvDims.transposeBackwardWeightInputIndex, hq j]
    · refine forall_fin_add_two.mpr ⟨?_, ?_, fun j => ?_⟩
      · simp only [Shape.ConvDims.transposeBackwardWeightOutputIndex, val_transposeOutput]
        rfl
      · simp only [Shape.ConvDims.transposeBackwardWeightOutputIndex, val_transposeOutput, hdiv,
          hw0]
        exact (Nat.div_add_mod' _ _).symm
      · simp only [Shape.ConvDims.transposeBackwardWeightOutputIndex, val_transposeOutput,
          Fin.eta, hq j, hwj j, hmul j]
        have := (halj j).1
        omega
  · rintro ⟨q, ⟨hab, hc⟩, ho'⟩
    obtain ⟨k', hk'⟩ := Shape.Coord.exists_spatial dims.kernel w
    let k : (s.convTransposeWindow t dims groups).Coord :=
      (⟨Shape.Coord.index w (dims.kernel ⟨1, by omega⟩).val %
        (t.extent (dims.kernel ⟨1, by omega⟩) / groups), Nat.mod_lt _ hG⟩, k')
    have hk : ∀ j : Fin r, Shape.Coord.index k (j.val + 1) =
        Shape.Coord.index w (dims.kernel ⟨j.val + 2, by omega⟩).val := fun j => hk' j
    obtain ⟨hc0, hc1⟩ := And.intro (hc ⟨0, by omega⟩) (hc ⟨1, by omega⟩)
    have hcj : ∀ j : Fin r, Shape.Coord.index c (dims.input ⟨j.val + 2, by omega⟩).val =
        Shape.Coord.index q (j.val + 1) := fun j => hc ⟨j.val + 2, by omega⟩
    simp only [Shape.ConvDims.transposeBackwardWeightInputIndex] at hc0 hc1
    obtain ⟨ho0, ho1'⟩ := And.intro (ho' ⟨0, by omega⟩) (ho' ⟨1, by omega⟩)
    simp only [Shape.ConvDims.transposeBackwardWeightOutputIndex, val_transposeOutput] at ho0 ho1'
    have hoj : ∀ j : Fin r, Shape.Coord.index o (dims.input ⟨j.val + 2, by omega⟩).val =
        Shape.Coord.index q (j.val + 1) * stride j +
          Shape.Coord.index w (dims.kernel ⟨j.val + 2, by omega⟩).val * dilation j - padLow j :=
      fun j => by
        simpa only [Shape.ConvDims.transposeBackwardWeightOutputIndex, val_transposeOutput,
          Fin.eta] using ho' ⟨j.val + 2, by omega⟩
    simp only [Shape.ConvDims.transposeBackwardWeightAbovePadLow, decide_eq_true_eq,
      fin_succ_succ] at hab
    have hsp : ∀ j : Fin r, Shape.Coord.index o (dims.input ⟨j.val + 2, by omega⟩).val + padLow j -
          Shape.Coord.index w (dims.kernel ⟨j.val + 2, by omega⟩).val * dilation j =
            Shape.Coord.index q (j.val + 1) * stride j := fun j => by
      rw [hoj j]
      have := hab j
      omega
    have hdiv := div_eq_of_eq_mul_add ho1' hw0lt
    have hmod := mod_eq_of_eq_mul_add ho1' hw0lt
    refine ⟨k, ⟨?_, ?_⟩, ?_⟩
    · refine forall_fin_add_two.mpr ⟨rfl, rfl, fun j => ?_⟩
      simp only [convTransposeAligned, convTransposeOffset, Bool.and_eq_true, decide_eq_true_eq,
        Fin.eta, hk j, hsp j, Nat.mul_mod_left, and_true]
      rw [hoj j]
      have := hab j
      omega
    · refine forall_fin_add_two.mpr ⟨?_, ?_, fun j => ?_⟩
      · simpa only [convTransposeIndex, convTransposeOffset, ho0] using hc0
      · simpa only [convTransposeIndex, convTransposeOffset, hdiv] using hc1
      · simp only [convTransposeIndex, convTransposeOffset, Fin.eta, hk j, hsp j, hcj j]
        exact (Nat.mul_div_cancel _ (Nat.pos_of_ne_zero (hst j))).symm
    · refine forall_fin_add_two.mpr ⟨?_, ?_, fun j => ?_⟩
      · simp only [convTransposeKernel, Shape.Coord.index_ofPositions, hmod]
      · simp only [convTransposeKernel, Shape.Coord.index_ofPositions, hdiv]
        exact (Nat.div_add_mod' _ _).symm
      · simp only [convTransposeKernel, Shape.Coord.index_ofPositions, hk j]

/-- Where `groups` is at most one or at most the kernel's input features, or the kernel has no
input features, or the input has no features, the weight gradient of `convTranspose` is the
transpose of `convTranspose` in its kernel. -/
theorem inner_convTranspose_eq_inner_convTransposeBackwardWeight (x : Tensor ℝ s)
    (w : Tensor ℝ t) (dims : Shape.ConvDims s t r)
    (stride padLow padHigh outputPadding dilation : Fin r → Nat) (groups : Nat)
    (hI : groups ≤ 1 ∨ 0 < t.extent (dims.kernel ⟨1, by omega⟩) / groups ∨
      t.extent (dims.kernel ⟨1, by omega⟩) = 0 ∨ s.extent (dims.input ⟨1, by omega⟩) = 0)
    (δ : Tensor ℝ (s.convTranspose t dims stride padLow padHigh outputPadding dilation groups)) :
    ⟪convTranspose x w dims stride padLow padHigh outputPadding dilation groups, δ⟫ =
      ⟪w, convTransposeBackwardWeight x dims stride padLow padHigh outputPadding dilation groups
        δ⟫ := by
  rw [Tensor.inner_eq_sum, Tensor.inner_eq_sum]
  simp only [convTranspose_eq_sum, convTransposeBackwardWeight_eq_sum, Finset.sum_mul,
    Finset.mul_sum]
  refine Eq.trans ?_ (sum_comm_three _).symm
  refine Finset.sum_congr rfl fun o _ => Finset.sum_congr rfl fun c _ =>
    Finset.sum_congr rfl fun w' _ => ?_
  rw [convTransposeRel_iff_convTransposeBackwardWeightRel _ _ _ _ _ _ _ hI]
  split_ifs <;> ring

/-- Where `groups` is at most one, or the input and the kernel have the same positive number of
input features per group, or the input has no features, the input gradient of `convTranspose` is
the transpose of `convTranspose` in its input. -/
theorem inner_convTranspose_eq_inner_convTransposeBackwardInput (x : Tensor ℝ s)
    (w : Tensor ℝ t) (dims : Shape.ConvDims s t r)
    (stride padLow padHigh outputPadding dilation : Fin r → Nat) (groups : Nat)
    (hC : groups ≤ 1 ∨ (0 < s.extent (dims.input ⟨1, by omega⟩) / groups ∧
        s.extent (dims.input ⟨1, by omega⟩) / groups =
          t.extent (dims.kernel ⟨1, by omega⟩) / groups) ∨
      s.extent (dims.input ⟨1, by omega⟩) = 0)
    (δ : Tensor ℝ (s.convTranspose t dims stride padLow padHigh outputPadding dilation groups)) :
    ⟪convTranspose x w dims stride padLow padHigh outputPadding dilation groups, δ⟫ =
      ⟪x, convTransposeBackwardInput w dims stride padLow padHigh outputPadding dilation groups
        δ⟫ := by
  rw [Tensor.inner_eq_sum, Tensor.inner_eq_sum]
  simp only [convTranspose_eq_sum, convTransposeBackwardInput_eq_sum, Finset.sum_mul,
    Finset.mul_sum]
  refine Eq.trans Finset.sum_comm ?_
  refine Finset.sum_congr rfl fun c _ => Eq.trans Finset.sum_comm ?_
  refine Finset.sum_congr rfl fun w' _ => Finset.sum_congr rfl fun o _ => ?_
  rw [convTransposeRel_iff_convTransposeBackwardInputRel _ _ _ _ _ _ _ hC]
  split_ifs <;> ring

/-- Summing over every axis but `d` and casting to a vector of length `n` is the reduction
`reduceBroadcast` of the broadcast of such vectors along `d`. -/
theorem castShape_reduce_otherAxes_eq_reduceBroadcast {S : Shape} (d : Fin S.rank) {n : Nat}
    (h : S.reduce (S.otherAxes [d]) false = [n]) (hb : Shape.BroadcastDims [n] S fun _ => d)
    (δ : Tensor ℝ S) :
    castShape (Tensor.reduce .sum δ (S.otherAxes [d]) false) h =
      reduceBroadcast (fun _ => d) hb δ := by
  obtain rfl : n = S.extent d :=
    (List.cons.inj (h.symm.trans (Shape.eraseAxes_otherAxes_singleton S d))).1
  exact reduce_otherAxes_eq_reduceBroadcast d δ

/-- `conv` with unit input dilation and one batch group, as a continuous bilinear map of the
input and the kernel. -/
noncomputable def convCLM (dims : Shape.ConvDims s t r)
    (stride padLow padHigh dilation : Fin r → Nat) (groups : Nat) :
    Tensor ℝ s →L[ℝ] Tensor ℝ t →L[ℝ]
      Tensor ℝ (s.conv t dims stride padLow padHigh (fun _ => 1) dilation 1) :=
  LinearMap.toContinuousLinearMap <| LinearMap.toContinuousLinearMap.toLinearMap ∘ₗ LinearMap.mk₂ ℝ
      (fun x w => conv x w dims stride padLow padHigh (fun _ => 1) dilation groups 1)
      (fun x x' w => Rep.ext fun o => by
        simp only [conv_eq_sum, Tensor.add_apply, ← Finset.sum_add_distrib]
        exact Finset.sum_congr rfl fun _ _ => Finset.sum_congr rfl fun _ _ => by split_ifs <;> ring)
      (fun a x w => Rep.ext fun o => by
        simp only [conv_eq_sum, Rep.smul_apply, smul_eq_mul, Finset.mul_sum]
        exact Finset.sum_congr rfl fun _ _ => Finset.sum_congr rfl fun _ _ => by split_ifs <;> ring)
      (fun x w w' => Rep.ext fun o => by
        simp only [conv_eq_sum, Tensor.add_apply, ← Finset.sum_add_distrib]
        exact Finset.sum_congr rfl fun _ _ => Finset.sum_congr rfl fun _ _ => by split_ifs <;> ring)
      (fun a x w => Rep.ext fun o => by
        simp only [conv_eq_sum, Rep.smul_apply, smul_eq_mul, Finset.mul_sum]
        exact Finset.sum_congr rfl fun _ _ => Finset.sum_congr rfl fun _ _ => by split_ifs <;> ring)

/-- `convCLM` applied to an input and a kernel is their `conv`. -/
@[simp] theorem convCLM_apply (dims : Shape.ConvDims s t r)
    (stride padLow padHigh dilation : Fin r → Nat) (groups : Nat) (x : Tensor ℝ s)
    (w : Tensor ℝ t) :
    convCLM dims stride padLow padHigh dilation groups x w =
      conv x w dims stride padLow padHigh (fun _ => 1) dilation groups 1 :=
  rfl

/-- `convTranspose` as a continuous bilinear map of the input and the kernel. -/
noncomputable def convTransposeCLM (dims : Shape.ConvDims s t r)
    (stride padLow padHigh outputPadding dilation : Fin r → Nat) (groups : Nat) :
    Tensor ℝ s →L[ℝ] Tensor ℝ t →L[ℝ]
      Tensor ℝ (s.convTranspose t dims stride padLow padHigh outputPadding dilation groups) :=
  LinearMap.toContinuousLinearMap <| LinearMap.toContinuousLinearMap.toLinearMap ∘ₗ LinearMap.mk₂ ℝ
      (fun x w => convTranspose x w dims stride padLow padHigh outputPadding dilation groups)
      (fun x x' w => Rep.ext fun o => by
        simp only [convTranspose_eq_sum, Tensor.add_apply, ← Finset.sum_add_distrib]
        exact Finset.sum_congr rfl fun _ _ => Finset.sum_congr rfl fun _ _ => by split_ifs <;> ring)
      (fun a x w => Rep.ext fun o => by
        simp only [convTranspose_eq_sum, Rep.smul_apply, smul_eq_mul, Finset.mul_sum]
        exact Finset.sum_congr rfl fun _ _ => Finset.sum_congr rfl fun _ _ => by split_ifs <;> ring)
      (fun x w w' => Rep.ext fun o => by
        simp only [convTranspose_eq_sum, Tensor.add_apply, ← Finset.sum_add_distrib]
        exact Finset.sum_congr rfl fun _ _ => Finset.sum_congr rfl fun _ _ => by split_ifs <;> ring)
      (fun a x w => Rep.ext fun o => by
        simp only [convTranspose_eq_sum, Rep.smul_apply, smul_eq_mul, Finset.mul_sum]
        exact Finset.sum_congr rfl fun _ _ => Finset.sum_congr rfl fun _ _ => by split_ifs <;> ring)

/-- `convTransposeCLM` applied to an input and a kernel is their `convTranspose`. -/
@[simp] theorem convTransposeCLM_apply (dims : Shape.ConvDims s t r)
    (stride padLow padHigh outputPadding dilation : Fin r → Nat) (groups : Nat)
    (x : Tensor ℝ s) (w : Tensor ℝ t) :
    convTransposeCLM dims stride padLow padHigh outputPadding dilation groups x w =
      convTranspose x w dims stride padLow padHigh outputPadding dilation groups :=
  rfl

end Tensor

namespace HasPullbackAt

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E] {s t : Shape} {r : Nat}
  {x : E} {f : E → Tensor ℝ s} {g : E → Tensor ℝ t} {Df : E →L[ℝ] Tensor ℝ s}
  {Dg : E →L[ℝ] Tensor ℝ t} {bf : Tensor ℝ s → E} {bg : Tensor ℝ t → E}

/-- The rule for `conv` of a fixed input `u` and a kernel `g`, with unit input dilation and one
batch group, at every `groups`: the kernel pulls back through `convBackwardWeight`. -/
@[fun_prop]
theorem conv_right (u : Tensor ℝ s) (dims : Shape.ConvDims s t r)
    (stride padLow padHigh dilation : Fin r → Nat) (groups : Nat) (hg : HasPullbackAt g Dg bg x) :
    HasPullbackAt
      (fun y => Tensor.conv u (g y) dims stride padLow padHigh (fun _ => 1) dilation groups 1)
      ((Tensor.convCLM dims stride padLow padHigh dilation groups u).comp Dg)
      (fun δ => bg (Tensor.convBackwardWeight u dims stride padLow padHigh dilation groups δ))
      x := by
  simpa only [Tensor.convCLM_apply] using
    clm (Tensor.convCLM dims stride padLow padHigh dilation groups u)
      (Tensor.convBackwardWeight u dims stride padLow padHigh dilation groups)
      (fun v δ => Tensor.inner_conv_eq_inner_convBackwardWeight u v dims stride padLow padHigh
        dilation groups δ) hg

/-- The product rule for `conv` of an input `f` and a kernel `g`, with unit input dilation and one
batch group, where `groups` is positive and at most the kernel's output features or the kernel has
no output features: the input pulls back through `convBackwardInput` and the kernel through
`convBackwardWeight`. -/
@[fun_prop]
theorem conv (dims : Shape.ConvDims s t r) (stride padLow padHigh dilation : Fin r → Nat)
    (groups : Nat) (hO : 0 < t.extent (dims.kernel ⟨0, by omega⟩) / groups ∨
      t.extent (dims.kernel ⟨0, by omega⟩) = 0)
    (hf : HasPullbackAt f Df bf x) (hg : HasPullbackAt g Dg bg x) :
    HasPullbackAt
      (fun y => Tensor.conv (f y) (g y) dims stride padLow padHigh (fun _ => 1) dilation groups 1)
      (((Tensor.convCLM dims stride padLow padHigh dilation groups).flip (g x)).comp Df +
        (Tensor.convCLM dims stride padLow padHigh dilation groups (f x)).comp Dg)
      (fun δ => bf (Tensor.convBackwardInput (g x) dims stride padLow padHigh dilation groups δ) +
        bg (Tensor.convBackwardWeight (f x) dims stride padLow padHigh dilation groups δ)) x := by
  simpa only [Tensor.convCLM_apply] using
    bilin (Tensor.convCLM dims stride padLow padHigh dilation groups)
      (fun w δ => Tensor.convBackwardInput w dims stride padLow padHigh dilation groups δ)
      (fun u δ => Tensor.convBackwardWeight u dims stride padLow padHigh dilation groups δ)
      (fun u v δ => Tensor.inner_conv_eq_inner_convBackwardInput u v dims stride padLow padHigh
        dilation groups hO δ)
      (fun u v δ => Tensor.inner_conv_eq_inner_convBackwardWeight u v dims stride padLow padHigh
        dilation groups δ) hf hg

/-- The rule for `convTranspose` of a fixed input `u` and a kernel `g`, where `groups` is at most
one or at most the kernel's input features, or the kernel has no input features, or the input has
no features: the kernel pulls back through `convTransposeBackwardWeight`. -/
@[fun_prop]
theorem convTranspose_right (u : Tensor ℝ s) (dims : Shape.ConvDims s t r)
    (stride padLow padHigh outputPadding dilation : Fin r → Nat) (groups : Nat)
    (hI : groups ≤ 1 ∨ 0 < t.extent (dims.kernel ⟨1, by omega⟩) / groups ∨
      t.extent (dims.kernel ⟨1, by omega⟩) = 0 ∨ s.extent (dims.input ⟨1, by omega⟩) = 0)
    (hg : HasPullbackAt g Dg bg x) :
    HasPullbackAt
      (fun y => Tensor.convTranspose u (g y) dims stride padLow padHigh outputPadding dilation
        groups)
      ((Tensor.convTransposeCLM dims stride padLow padHigh outputPadding dilation groups u).comp Dg)
      (fun δ => bg (Tensor.convTransposeBackwardWeight u dims stride padLow padHigh outputPadding
        dilation groups δ)) x := by
  simpa only [Tensor.convTransposeCLM_apply] using
    clm (Tensor.convTransposeCLM dims stride padLow padHigh outputPadding dilation groups u)
      (Tensor.convTransposeBackwardWeight u dims stride padLow padHigh outputPadding dilation
        groups)
      (fun v δ => Tensor.inner_convTranspose_eq_inner_convTransposeBackwardWeight u v dims stride
        padLow padHigh outputPadding dilation groups hI δ) hg

/-- The product rule for `convTranspose` of an input `f` and a kernel `g`, where `hI` is the
hypothesis of `convTranspose_right` and `hC` says that `groups` is at most one, or the input and
the kernel have the same positive number of input features per group, or the input has no
features: the input pulls back through `convTransposeBackwardInput` and the kernel through
`convTransposeBackwardWeight`. -/
@[fun_prop]
theorem convTranspose (dims : Shape.ConvDims s t r)
    (stride padLow padHigh outputPadding dilation : Fin r → Nat) (groups : Nat)
    (hI : groups ≤ 1 ∨ 0 < t.extent (dims.kernel ⟨1, by omega⟩) / groups ∨
      t.extent (dims.kernel ⟨1, by omega⟩) = 0 ∨ s.extent (dims.input ⟨1, by omega⟩) = 0)
    (hC : groups ≤ 1 ∨ (0 < s.extent (dims.input ⟨1, by omega⟩) / groups ∧
        s.extent (dims.input ⟨1, by omega⟩) / groups =
          t.extent (dims.kernel ⟨1, by omega⟩) / groups) ∨
      s.extent (dims.input ⟨1, by omega⟩) = 0)
    (hf : HasPullbackAt f Df bf x) (hg : HasPullbackAt g Dg bg x) :
    HasPullbackAt
      (fun y => Tensor.convTranspose (f y) (g y) dims stride padLow padHigh outputPadding dilation
        groups)
      (((Tensor.convTransposeCLM dims stride padLow padHigh outputPadding dilation groups).flip
          (g x)).comp Df +
        (Tensor.convTransposeCLM dims stride padLow padHigh outputPadding dilation groups
          (f x)).comp Dg)
      (fun δ => bf (Tensor.convTransposeBackwardInput (g x) dims stride padLow padHigh
          outputPadding dilation groups δ) +
        bg (Tensor.convTransposeBackwardWeight (f x) dims stride padLow padHigh outputPadding
          dilation groups δ)) x := by
  simpa only [Tensor.convTransposeCLM_apply] using
    bilin (Tensor.convTransposeCLM dims stride padLow padHigh outputPadding dilation groups)
      (fun w δ => Tensor.convTransposeBackwardInput w dims stride padLow padHigh outputPadding
        dilation groups δ)
      (fun u δ => Tensor.convTransposeBackwardWeight u dims stride padLow padHigh outputPadding
        dilation groups δ)
      (fun u v δ => Tensor.inner_convTranspose_eq_inner_convTransposeBackwardInput u v dims
        stride padLow padHigh outputPadding dilation groups hC δ)
      (fun u v δ => Tensor.inner_convTranspose_eq_inner_convTransposeBackwardWeight u v dims
        stride padLow padHigh outputPadding dilation groups hI δ) hf hg

end HasPullbackAt

namespace Op

open Tensor

/-- At every attribute, the forward of `Op.conv` is differentiable in its kernel and bias jointly
on their L2 product, with the input fixed, and the kernel and bias components of its backward are
the adjoint of that derivative. -/
@[op_cert TorchLean.Op.conv derivative 0 1]
theorem conv_kernel_bias_backward_eq_adjoint {s k : Shape} {r : Nat}
    (dims : Shape.ConvDims s k r) (stride padLow padHigh dilation : Fin r → Nat) (groups : Nat)
    {lead x y : Shape} (hlead : s.extent (dims.input ⟨0, by omega⟩) = lead.size)
    (hx : x = s.expandAxis (dims.input ⟨0, by omega⟩) lead)
    (hy : y = (s.conv k dims stride padLow padHigh (fun _ => 1) dilation 1).expandAxis
      (dims.output stride padLow padHigh (fun _ => 1) dilation 1 ⟨0, by omega⟩) lead)
    (kernel : Tensor ℝ k) (bias : Tensor ℝ [k.extent (dims.kernel ⟨0, by omega⟩)])
    (input : Tensor ℝ x) (δ : Tensor ℝ y) :
    DifferentiableAt ℝ
      (fun p : WithLp 2 (Tensor ℝ k × Tensor ℝ [k.extent (dims.kernel ⟨0, by omega⟩)]) =>
        (((Op.conv dims stride padLow padHigh dilation groups hlead hx hy).rule? (α := ℝ)
          |>.get rfl).forward (p.fst, p.snd, input, ())).1)
      (WithLp.toLp 2 (kernel, bias)) ∧
    WithLp.toLp 2 ((((Op.conv dims stride padLow padHigh dilation groups hlead hx hy).rule? (α := ℝ)
          |>.get rfl).backward (kernel, bias, input, ()) (δ, ())).1,
        (((Op.conv dims stride padLow padHigh dilation groups hlead hx hy).rule? (α := ℝ)
          |>.get rfl).backward (kernel, bias, input, ()) (δ, ())).2.1) =
      (fderiv ℝ (fun p : WithLp 2 (Tensor ℝ k × Tensor ℝ [k.extent (dims.kernel ⟨0, by omega⟩)]) =>
          (((Op.conv dims stride padLow padHigh dilation groups hlead hx hy).rule? (α := ℝ)
            |>.get rfl).forward (p.fst, p.snd, input, ())).1)
        (WithLp.toLp 2 (kernel, bias))).adjoint δ := by
  simp only [Op.rule?_conv, Option.get_some]
  apply HasPullbackAt.differentiableAt_and
  · fun_prop
  · refine withLp_prod_ext ?_ ?_ <;> simp only [WithLp.add_fst, WithLp.add_snd, WithLp.toLp_fst,
      WithLp.toLp_snd, zero_add, add_zero]
    exact castShape_reduce_otherAxes_eq_reduceBroadcast _ _ _ _

/-- At every attribute, the forward of `Op.convTranspose` is differentiable in its bias, and the
bias component of its backward is the adjoint of that derivative. -/
@[op_cert TorchLean.Op.convTranspose derivative 1]
theorem convTranspose_bias_backward_eq_adjoint {s k : Shape} {r : Nat}
    (dims : Shape.ConvDims s k r) (stride padLow padHigh outputPadding dilation : Fin r → Nat)
    (groups : Nat) (kernel : Tensor ℝ k)
    (bias : Tensor ℝ [groups * k.extent (dims.kernel ⟨0, by omega⟩)]) (input : Tensor ℝ s)
    (δ : Tensor ℝ (s.convTranspose k dims stride padLow padHigh outputPadding dilation groups)) :
    DifferentiableAt ℝ (fun b : Tensor ℝ [groups * k.extent (dims.kernel ⟨0, by omega⟩)] =>
        (((Op.convTranspose dims stride padLow padHigh outputPadding dilation groups).rule?
          (α := ℝ) |>.get rfl).forward (kernel, b, input, ())).1) bias ∧
    (((Op.convTranspose dims stride padLow padHigh outputPadding dilation groups).rule?
        (α := ℝ) |>.get rfl).backward (kernel, bias, input, ()) (δ, ())).2.1 =
      (fderiv ℝ (fun b : Tensor ℝ [groups * k.extent (dims.kernel ⟨0, by omega⟩)] =>
        (((Op.convTranspose dims stride padLow padHigh outputPadding dilation groups).rule?
          (α := ℝ) |>.get rfl).forward (kernel, b, input, ())).1) bias).adjoint δ := by
  simp only [Op.rule?_convTranspose, Option.get_some]
  apply HasPullbackAt.differentiableAt_and
  · fun_prop
  · simpa only [zero_add] using castShape_reduce_otherAxes_eq_reduceBroadcast _ _ _ _

/-- Where `groups` is at most one or at most the kernel's input features, or the kernel has no
input features, or the input has no features, the forward of `Op.convTranspose` is differentiable
in its kernel and bias jointly on their L2 product, with the input fixed, and the kernel and bias
components of its backward are the adjoint of that derivative. The condition is sufficient, not
exact: outside it the forward is constant in the kernel, so the backward is the adjoint exactly
where its kernel component vanishes, for example at a zero input. -/
@[op_cert TorchLean.Op.convTranspose derivative 0 1]
theorem convTranspose_kernel_bias_backward_eq_adjoint {s k : Shape} {r : Nat}
    (dims : Shape.ConvDims s k r) (stride padLow padHigh outputPadding dilation : Fin r → Nat)
    (groups : Nat) (hI : groups ≤ 1 ∨ 0 < k.extent (dims.kernel ⟨1, by omega⟩) / groups ∨
      k.extent (dims.kernel ⟨1, by omega⟩) = 0 ∨ s.extent (dims.input ⟨1, by omega⟩) = 0)
    (kernel : Tensor ℝ k) (bias : Tensor ℝ [groups * k.extent (dims.kernel ⟨0, by omega⟩)])
    (input : Tensor ℝ s)
    (δ : Tensor ℝ (s.convTranspose k dims stride padLow padHigh outputPadding dilation groups)) :
    DifferentiableAt ℝ (fun p : WithLp 2 (Tensor ℝ k ×
          Tensor ℝ [groups * k.extent (dims.kernel ⟨0, by omega⟩)]) =>
        (((Op.convTranspose dims stride padLow padHigh outputPadding dilation groups).rule?
          (α := ℝ) |>.get rfl).forward (p.fst, p.snd, input, ())).1)
      (WithLp.toLp 2 (kernel, bias)) ∧
    WithLp.toLp 2
        ((((Op.convTranspose dims stride padLow padHigh outputPadding dilation groups).rule?
          (α := ℝ) |>.get rfl).backward (kernel, bias, input, ()) (δ, ())).1,
        (((Op.convTranspose dims stride padLow padHigh outputPadding dilation groups).rule?
          (α := ℝ) |>.get rfl).backward (kernel, bias, input, ()) (δ, ())).2.1) =
      (fderiv ℝ (fun p : WithLp 2 (Tensor ℝ k ×
          Tensor ℝ [groups * k.extent (dims.kernel ⟨0, by omega⟩)]) =>
        (((Op.convTranspose dims stride padLow padHigh outputPadding dilation groups).rule?
          (α := ℝ) |>.get rfl).forward (p.fst, p.snd, input, ())).1)
        (WithLp.toLp 2 (kernel, bias))).adjoint δ := by
  simp only [Op.rule?_convTranspose, Option.get_some]
  apply HasPullbackAt.differentiableAt_and
  · fun_prop (disch := assumption)
  · refine withLp_prod_ext ?_ ?_ <;> simp only [WithLp.add_fst, WithLp.add_snd, WithLp.toLp_fst,
      WithLp.toLp_snd, zero_add, add_zero]
    exact castShape_reduce_otherAxes_eq_reduceBroadcast _ _ _ _

/-- Where `groups` is positive and at most the kernel's output features, or the kernel has no
output features, the forward of `Op.conv` is differentiable in its kernel, bias and input on their
L2 product, and its backward is the adjoint of that derivative. The condition is sufficient, not
exact: outside it the input component of the backward is zero, so the backward is the adjoint
exactly where the input component of the adjoint at `δ` vanishes, for example at a zero kernel. -/
@[op_cert TorchLean.Op.conv derivative 0 1 2]
theorem conv_prod_backward_eq_adjoint_of_div_pos {s k : Shape} {r : Nat}
    (dims : Shape.ConvDims s k r) (stride padLow padHigh dilation : Fin r → Nat) (groups : Nat)
    {lead x y : Shape} (hlead : s.extent (dims.input ⟨0, by omega⟩) = lead.size)
    (hx : x = s.expandAxis (dims.input ⟨0, by omega⟩) lead)
    (hy : y = (s.conv k dims stride padLow padHigh (fun _ => 1) dilation 1).expandAxis
      (dims.output stride padLow padHigh (fun _ => 1) dilation 1 ⟨0, by omega⟩) lead)
    (hO : 0 < k.extent (dims.kernel ⟨0, by omega⟩) / groups ∨
      k.extent (dims.kernel ⟨0, by omega⟩) = 0)
    (kernel : Tensor ℝ k) (bias : Tensor ℝ [k.extent (dims.kernel ⟨0, by omega⟩)])
    (input : Tensor ℝ x) (δ : Tensor ℝ y) :
    DifferentiableAt ℝ (fun p : WithLp 2 (Tensor ℝ k ×
          WithLp 2 (Tensor ℝ [k.extent (dims.kernel ⟨0, by omega⟩)] × Tensor ℝ x)) =>
        (((Op.conv dims stride padLow padHigh dilation groups hlead hx hy).rule? (α := ℝ)
          |>.get rfl).forward (p.fst, p.snd.fst, p.snd.snd, ())).1)
      (WithLp.toLp 2 (kernel, WithLp.toLp 2 (bias, input))) ∧
    WithLp.toLp 2 ((((Op.conv dims stride padLow padHigh dilation groups hlead hx hy).rule? (α := ℝ)
          |>.get rfl).backward (kernel, bias, input, ()) (δ, ())).1,
        WithLp.toLp 2
          ((((Op.conv dims stride padLow padHigh dilation groups hlead hx hy).rule? (α := ℝ)
            |>.get rfl).backward (kernel, bias, input, ()) (δ, ())).2.1,
          (((Op.conv dims stride padLow padHigh dilation groups hlead hx hy).rule? (α := ℝ)
            |>.get rfl).backward (kernel, bias, input, ()) (δ, ())).2.2.1)) =
      (fderiv ℝ (fun p : WithLp 2 (Tensor ℝ k ×
          WithLp 2 (Tensor ℝ [k.extent (dims.kernel ⟨0, by omega⟩)] × Tensor ℝ x)) =>
        (((Op.conv dims stride padLow padHigh dilation groups hlead hx hy).rule? (α := ℝ)
          |>.get rfl).forward (p.fst, p.snd.fst, p.snd.snd, ())).1)
        (WithLp.toLp 2 (kernel, WithLp.toLp 2 (bias, input)))).adjoint δ := by
  simp only [Op.rule?_conv, Option.get_some]
  apply HasPullbackAt.differentiableAt_and
  · fun_prop (disch := assumption)
  · refine withLp_prod_ext ?_ (withLp_prod_ext ?_ ?_) <;> simp only [WithLp.add_fst,
      WithLp.add_snd, WithLp.toLp_fst, WithLp.toLp_snd, zero_add, add_zero]
    exact castShape_reduce_otherAxes_eq_reduceBroadcast _ _ _ _

/-- Where `groups` is at most one, or the input and the kernel have the same positive number of
input features per group, or the input has no features, the forward of `Op.convTranspose` is
differentiable in its kernel, bias and input on their L2 product, and its backward is the adjoint
of that derivative. The condition is sufficient, not exact: outside it the backward is also the
adjoint where the kernel and the input are both zero. -/
@[op_cert TorchLean.Op.convTranspose derivative 0 1 2]
theorem convTranspose_prod_backward_eq_adjoint_of_div_eq {s k : Shape} {r : Nat}
    (dims : Shape.ConvDims s k r) (stride padLow padHigh outputPadding dilation : Fin r → Nat)
    (groups : Nat) (hC : groups ≤ 1 ∨ (0 < s.extent (dims.input ⟨1, by omega⟩) / groups ∧
        s.extent (dims.input ⟨1, by omega⟩) / groups =
          k.extent (dims.kernel ⟨1, by omega⟩) / groups) ∨
      s.extent (dims.input ⟨1, by omega⟩) = 0)
    (kernel : Tensor ℝ k) (bias : Tensor ℝ [groups * k.extent (dims.kernel ⟨0, by omega⟩)])
    (input : Tensor ℝ s)
    (δ : Tensor ℝ (s.convTranspose k dims stride padLow padHigh outputPadding dilation groups)) :
    DifferentiableAt ℝ (fun p : WithLp 2 (Tensor ℝ k ×
          WithLp 2 (Tensor ℝ [groups * k.extent (dims.kernel ⟨0, by omega⟩)] × Tensor ℝ s)) =>
        (((Op.convTranspose dims stride padLow padHigh outputPadding dilation groups).rule?
          (α := ℝ) |>.get rfl).forward (p.fst, p.snd.fst, p.snd.snd, ())).1)
      (WithLp.toLp 2 (kernel, WithLp.toLp 2 (bias, input))) ∧
    WithLp.toLp 2
        ((((Op.convTranspose dims stride padLow padHigh outputPadding dilation groups).rule?
          (α := ℝ) |>.get rfl).backward (kernel, bias, input, ()) (δ, ())).1,
        WithLp.toLp 2
          ((((Op.convTranspose dims stride padLow padHigh outputPadding dilation groups).rule?
            (α := ℝ) |>.get rfl).backward (kernel, bias, input, ()) (δ, ())).2.1,
          (((Op.convTranspose dims stride padLow padHigh outputPadding dilation groups).rule?
            (α := ℝ) |>.get rfl).backward (kernel, bias, input, ()) (δ, ())).2.2.1)) =
      (fderiv ℝ (fun p : WithLp 2 (Tensor ℝ k ×
          WithLp 2 (Tensor ℝ [groups * k.extent (dims.kernel ⟨0, by omega⟩)] × Tensor ℝ s)) =>
        (((Op.convTranspose dims stride padLow padHigh outputPadding dilation groups).rule?
          (α := ℝ) |>.get rfl).forward (p.fst, p.snd.fst, p.snd.snd, ())).1)
        (WithLp.toLp 2 (kernel, WithLp.toLp 2 (bias, input)))).adjoint δ := by
  have hI : groups ≤ 1 ∨ 0 < k.extent (dims.kernel ⟨1, by omega⟩) / groups ∨
      k.extent (dims.kernel ⟨1, by omega⟩) = 0 ∨ s.extent (dims.input ⟨1, by omega⟩) = 0 := by
    rcases hC with h | ⟨h1, h2⟩ | h
    exacts [Or.inl h, Or.inr (Or.inl (h2 ▸ h1)), Or.inr (Or.inr (Or.inr h))]
  simp only [Op.rule?_convTranspose, Option.get_some]
  apply HasPullbackAt.differentiableAt_and
  · fun_prop (disch := assumption)
  · refine withLp_prod_ext ?_ (withLp_prod_ext ?_ ?_) <;> simp only [WithLp.add_fst,
      WithLp.add_snd, WithLp.toLp_fst, WithLp.toLp_snd, zero_add, add_zero]
    exact castShape_reduce_otherAxes_eq_reduceBroadcast _ _ _ _

/-- Where `groups` is positive and divides the kernel's output features, the forward of `Op.conv`
is differentiable in its kernel, bias and input on their L2 product, and its backward is the
adjoint of that derivative. These sufficient hypotheses are the range that `Op.check` accepts. -/
@[op_cert TorchLean.Op.conv derivative 0 1 2]
theorem conv_prod_backward_eq_adjoint {s k : Shape} {r : Nat} (dims : Shape.ConvDims s k r)
    (stride padLow padHigh dilation : Fin r → Nat) (groups : Nat) {lead x y : Shape}
    (hlead : s.extent (dims.input ⟨0, by omega⟩) = lead.size)
    (hx : x = s.expandAxis (dims.input ⟨0, by omega⟩) lead)
    (hy : y = (s.conv k dims stride padLow padHigh (fun _ => 1) dilation 1).expandAxis
      (dims.output stride padLow padHigh (fun _ => 1) dilation 1 ⟨0, by omega⟩) lead)
    (hgroups : 0 < groups) (hdvd : groups ∣ k.extent (dims.kernel ⟨0, by omega⟩))
    (kernel : Tensor ℝ k) (bias : Tensor ℝ [k.extent (dims.kernel ⟨0, by omega⟩)])
    (input : Tensor ℝ x) (δ : Tensor ℝ y) :
    DifferentiableAt ℝ (fun p : WithLp 2 (Tensor ℝ k ×
          WithLp 2 (Tensor ℝ [k.extent (dims.kernel ⟨0, by omega⟩)] × Tensor ℝ x)) =>
        (((Op.conv dims stride padLow padHigh dilation groups hlead hx hy).rule? (α := ℝ)
          |>.get rfl).forward (p.fst, p.snd.fst, p.snd.snd, ())).1)
      (WithLp.toLp 2 (kernel, WithLp.toLp 2 (bias, input))) ∧
    WithLp.toLp 2 ((((Op.conv dims stride padLow padHigh dilation groups hlead hx hy).rule? (α := ℝ)
          |>.get rfl).backward (kernel, bias, input, ()) (δ, ())).1,
        WithLp.toLp 2
          ((((Op.conv dims stride padLow padHigh dilation groups hlead hx hy).rule? (α := ℝ)
            |>.get rfl).backward (kernel, bias, input, ()) (δ, ())).2.1,
          (((Op.conv dims stride padLow padHigh dilation groups hlead hx hy).rule? (α := ℝ)
            |>.get rfl).backward (kernel, bias, input, ()) (δ, ())).2.2.1)) =
      (fderiv ℝ (fun p : WithLp 2 (Tensor ℝ k ×
          WithLp 2 (Tensor ℝ [k.extent (dims.kernel ⟨0, by omega⟩)] × Tensor ℝ x)) =>
        (((Op.conv dims stride padLow padHigh dilation groups hlead hx hy).rule? (α := ℝ)
          |>.get rfl).forward (p.fst, p.snd.fst, p.snd.snd, ())).1)
        (WithLp.toLp 2 (kernel, WithLp.toLp 2 (bias, input)))).adjoint δ :=
  conv_prod_backward_eq_adjoint_of_div_pos dims stride padLow padHigh dilation groups hlead hx hy
    ((Nat.eq_zero_or_pos (k.extent (dims.kernel ⟨0, by omega⟩))).symm.imp_left fun h =>
      Nat.div_pos (Nat.le_of_dvd h hdvd) hgroups) kernel bias input δ

/-- Where `groups` is positive and divides the input's features, which are the kernel's input
features, the forward of `Op.convTranspose` is differentiable in its kernel, bias and input on
their L2 product, and its backward is the adjoint of that derivative. These sufficient hypotheses
are the range that `Op.check` accepts. -/
@[op_cert TorchLean.Op.convTranspose derivative 0 1 2]
theorem convTranspose_prod_backward_eq_adjoint {s k : Shape} {r : Nat}
    (dims : Shape.ConvDims s k r) (stride padLow padHigh outputPadding dilation : Fin r → Nat)
    (groups : Nat) (hgroups : 0 < groups) (hdvd : groups ∣ s.extent (dims.input ⟨1, by omega⟩))
    (hI : k.extent (dims.kernel ⟨1, by omega⟩) = s.extent (dims.input ⟨1, by omega⟩))
    (kernel : Tensor ℝ k) (bias : Tensor ℝ [groups * k.extent (dims.kernel ⟨0, by omega⟩)])
    (input : Tensor ℝ s)
    (δ : Tensor ℝ (s.convTranspose k dims stride padLow padHigh outputPadding dilation groups)) :
    DifferentiableAt ℝ (fun p : WithLp 2 (Tensor ℝ k ×
          WithLp 2 (Tensor ℝ [groups * k.extent (dims.kernel ⟨0, by omega⟩)] × Tensor ℝ s)) =>
        (((Op.convTranspose dims stride padLow padHigh outputPadding dilation groups).rule?
          (α := ℝ) |>.get rfl).forward (p.fst, p.snd.fst, p.snd.snd, ())).1)
      (WithLp.toLp 2 (kernel, WithLp.toLp 2 (bias, input))) ∧
    WithLp.toLp 2
        ((((Op.convTranspose dims stride padLow padHigh outputPadding dilation groups).rule?
          (α := ℝ) |>.get rfl).backward (kernel, bias, input, ()) (δ, ())).1,
        WithLp.toLp 2
          ((((Op.convTranspose dims stride padLow padHigh outputPadding dilation groups).rule?
            (α := ℝ) |>.get rfl).backward (kernel, bias, input, ()) (δ, ())).2.1,
          (((Op.convTranspose dims stride padLow padHigh outputPadding dilation groups).rule?
            (α := ℝ) |>.get rfl).backward (kernel, bias, input, ()) (δ, ())).2.2.1)) =
      (fderiv ℝ (fun p : WithLp 2 (Tensor ℝ k ×
          WithLp 2 (Tensor ℝ [groups * k.extent (dims.kernel ⟨0, by omega⟩)] × Tensor ℝ s)) =>
        (((Op.convTranspose dims stride padLow padHigh outputPadding dilation groups).rule?
          (α := ℝ) |>.get rfl).forward (p.fst, p.snd.fst, p.snd.snd, ())).1)
        (WithLp.toLp 2 (kernel, WithLp.toLp 2 (bias, input)))).adjoint δ :=
  have hC0 := (Nat.eq_zero_or_pos (s.extent (dims.input ⟨1, by omega⟩))).symm
  convTranspose_prod_backward_eq_adjoint_of_div_eq dims stride padLow padHigh outputPadding
    dilation groups (Or.inr (hC0.imp_left fun h =>
      ⟨Nat.div_pos (Nat.le_of_dvd h hdvd) hgroups, by rw [hI]⟩)) kernel bias input δ

end Op

end TorchLean
