/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Verify.Certificate.Rational
public import TorchLean.Verify.Tensor
public import TorchLean.NN.Sequential
public import TorchLean.NN.Layer.Activation
public import TorchLean.NN.Layer.Recurrent
public import TorchLean.NN.Sequential.Composition
public import TorchLean.Tensor.Tactic.Verify
public import TorchLean.Verify.CROWN.FlatBox.Affine.Real
public import TorchLean.Verify.Monotonicity
public import TorchLean.Verify.CROWN.FlatBox.Real

/-!
# Exact CROWN output-query checks

`Query.check` recomputes the affine bounds of a dense/ReLU network over an input box in rational
arithmetic and checks strict or non-strict output inequalities against them. Its soundness theorem
`Query.check_sound` states the inequalities for the network's `nn.Sequential` model evaluated over
the reals by `nn.Sequential.eval`, at every point of the box. Every transfer and the final query
check are proved here, with no external soundness premise.

We wrote the propagation by hand from the forward mode of auto_LiRPA (Algorithm 1). It starts from
the identity forms (`Bounds.identity`), and a linear layer gets the affine row of their Table 6
(`Bounds.linear`, proved by `Bounds.linear_encloses`). Before a ReLU both forms are bounded over
the input box (`Bounds.interval`), and an unstable neuron gets the lower line `y = α z` of α-CROWN's
Eq. (5) with the triangle's chord above (`Bounds.relu`). The certificate supplies α and Lean never
tunes it. `Network.propagate` gives `none` when `checkAlpha` finds a slope outside `[0, 1]`, and
`Bounds.relu_encloses` rests on `Cert.alphaRelaxLowerScalar_sound` and `ReLU.relaxScalar_sound`.
The query's rows act as one more linear layer on the output forms before `checkUpper` bounds them
over the box. Every step is exact in `ℚ`, so nothing rounds.

References:
- auto_LiRPA: Xu et al., "Automatic Perturbation Analysis for Scalable Certified Robustness and
  Beyond", NeurIPS 2020, arXiv:2002.12920, Sec. 3.1 (Algorithm 1) and App. A.1 (Table 6).
- α-CROWN: Xu et al., "Fast and Complete: Enabling Complete Neural Network Verification with
  Rapid and Massively Parallel Incomplete Verifiers", ICLR 2021, arXiv:2011.13824, Sec. 2.3,
  Eq. (5).
-/

@[expose] public section

open scoped TorchLean.nn.Sequential

namespace TorchLean.Certificate.CROWNQuery

open Tensor Linear
open CROWN Graph
open Rational
open Cert ReLU
open scoped RationalAlgebraic

/-- Whether the upper endpoint of every row of `upper` over the box `input` is negative when
`strict` is set and nonpositive otherwise. -/
def checkUpper {n m : Nat} (strict : Bool) (upper : Linear ℚ n m)
    (input : Box ℚ [n]) : Bool :=
  let bounds := IBP.linear upper.weights input ⟨upper.bias, upper.bias⟩
  (List.finRange m).all fun i =>
    if strict then decide (bounds.hi.getScalar i < 0)
    else decide (bounds.hi.getScalar i ≤ 0)

/-- Interval bounds of an exact affine form over the input box enclose its real evaluation. -/
theorem affine_interval_encloses {n m : Nat} (a : Linear ℚ n m)
    (input : Box ℚ [n]) (x : Tensor ℝ [n])
    (hx : Directed.RowEncloses
      ⟨n, realTensor input.lo, realTensor input.hi⟩ n fun i => getAtOrZero x [i]) :
    ∀ i, (realTensor (IBP.linear a.weights input ⟨a.bias, a.bias⟩).lo).getScalar i ≤
      (Linear.forward (realAffine a) x).getScalar i ∧
      (Linear.forward (realAffine a) x).getScalar i ≤
        (realTensor (IBP.linear a.weights input ⟨a.bias, a.bias⟩).hi).getScalar i := by
  have h := IBP.linear_sound_real
    (realTensor a.weights) (realBox input) (Box.point (realTensor a.bias))
    x (realTensor a.bias) ((encloses_iff_getScalar _ _ x).mp hx) (Box.contains_point_self _)
  have hm := realBox_linear a.weights input ⟨a.bias, a.bias⟩
  have hlo := congrArg Box.lo hm
  have hhi := congrArg Box.hi hm
  simp only [realBox] at hlo hhi
  intro i
  have hi := h i
  change (IBP.linear (realTensor a.weights) (realBox input)
      (Box.point (realTensor a.bias))).lo.getScalar i ≤ _ ∧
    _ ≤ (IBP.linear (realTensor a.weights) (realBox input)
      (Box.point (realTensor a.bias))).hi.getScalar i at hi
  simp only [realBox, Box.point] at hi
  rw [← hlo, ← hhi] at hi
  exact hi

/-- Every accepted upper form is nonpositive throughout the real input box, and negative when
`strict` is set. -/
theorem checkUpper_sound {n m : Nat} (strict : Bool) (upper : Linear ℚ n m)
    (input : Box ℚ [n]) (h : checkUpper strict upper input = true)
    (x : Tensor ℝ [n])
    (hx : Directed.RowEncloses
      ⟨n, realTensor input.lo, realTensor input.hi⟩ n fun i => getAtOrZero x [i]) (i : Fin m) :
    if strict then (Linear.forward (realAffine upper) x).getScalar i < 0
    else (Linear.forward (realAffine upper) x).getScalar i ≤ 0 := by
  have hi := (affine_interval_encloses upper input x hx i).2
  simp only [realTensor_getScalar] at hi
  have hcheck := (List.all_eq_true.mp h) i (List.mem_finRange i)
  cases strict with
  | false =>
      simp only [Bool.false_eq_true, ↓reduceIte] at hcheck ⊢
      have hr : (((IBP.linear upper.weights input ⟨upper.bias, upper.bias⟩).hi.getScalar i : ℚ)
        : ℝ) ≤ 0 := by exact_mod_cast of_decide_eq_true hcheck
      exact hi.trans hr
  | true =>
      simp only [↓reduceIte] at hcheck ⊢
      have hr : (((IBP.linear upper.weights input ⟨upper.bias, upper.bias⟩).hi.getScalar i : ℚ)
        : ℝ) < 0 := by exact_mod_cast of_decide_eq_true hcheck
      exact hi.trans_lt hr

/-- Lower and upper rational affine bounds, with the input dimension `n` and the output dimension
`m` in the type. -/
structure Bounds (n m : Nat) where
  /-- The lower affine forms. -/
  lo : Linear ℚ n m
  /-- The upper affine forms. -/
  hi : Linear ℚ n m

/-- Forget the static dimensions to reuse the dimension-carrying CROWN transfer functions. -/
def Bounds.flat {n m : Nat} (b : Bounds n m) : FlatAffineBounds ℚ :=
  ⟨n, m, b.lo, b.hi⟩

/-- The bounds enclose `y` at the real input `x`: in every coordinate, the lower form at `x` is at
most `y` and the upper form at least `y`. -/
def Bounds.Encloses {n m : Nat} (b : Bounds n m) (x : Tensor ℝ [n])
    (y : Tensor ℝ [m]) : Prop :=
  ∀ i, (Linear.forward (realAffine b.lo) x).getScalar i ≤ y.getScalar i ∧
    y.getScalar i ≤ (Linear.forward (realAffine b.hi) x).getScalar i

/-- The identity affine bounds at the network input. -/
def Bounds.identity (n : Nat) : Bounds n n :=
  ⟨CROWN.Graph.affIdentity n, CROWN.Graph.affIdentity n⟩

/-- Push affine bounds through an exact linear layer with the sign-split CROWN transfer. -/
def Bounds.linear {n m k : Nat} (b : Bounds n m) (w : Tensor ℚ [k, m])
    (bias : Tensor ℚ [k]) : Bounds n k :=
  let out := CROWN.Graph.propagateLinearBounds w bias b.flat rfl
  ⟨out.loAff, out.hiAff⟩

/-- Bound each affine form over the original input box. -/
def Bounds.interval {n m : Nat} (b : Bounds n m) (input : Box ℚ [n]) : Box ℚ [m] :=
  ⟨(IBP.linear b.lo.weights input ⟨b.lo.bias, b.lo.bias⟩).lo,
   (IBP.linear b.hi.weights input ⟨b.hi.bias, b.hi.bias⟩).hi⟩

/-- Push affine bounds through ReLU using the α lower and triangle upper relaxations. -/
def Bounds.relu {n m : Nat} (b : Bounds n m) (input : Box ℚ [n])
    (alpha : Tensor ℚ [m]) : Bounds n m :=
  let pre := b.interval input
  ⟨ReLU.propagateAffine
      (CROWN.Cert.alphaRelaxLowerVec pre.lo pre.hi alpha) b.lo,
   ReLU.propagateAffine (ReLU.relaxVector pre.lo pre.hi) b.hi⟩

/-- The identity bounds enclose `x` at `x`. -/
theorem Bounds.identity_encloses {n : Nat} (x : Tensor ℝ [n]) :
    (Bounds.identity n).Encloses x x := by
  have h := boundsEvalAt_boundsIdentity x
  have hr := realAffineBounds_identity n
  simp only [realAffineBounds, CROWN.Graph.boundsIdentity] at hr
  injection hr with _ _ hloAff hhiAff
  simp only [boundsEvalAt, CROWN.Graph.boundsIdentity] at h
  injection h with _ hlo hhi
  intro i
  constructor
  · simpa [identity, ← hloAff] using le_of_eq (congrArg (fun t => t.getScalar i) hlo)
  · simpa [identity, ← hhiAff] using le_of_eq (congrArg (fun t => t.getScalar i) hhi).symm

/-- If `b` encloses `y` at `x`, then `b.linear w bias` encloses the product of `w` with `y`
plus `bias` at `x`, with `w` and `bias` read as reals. -/
theorem Bounds.linear_encloses {n m k : Nat} (b : Bounds n m)
    (w : Tensor ℚ [k, m]) (bias : Tensor ℚ [k]) (x : Tensor ℝ [n])
    (y : Tensor ℝ [m]) (h : b.Encloses x y) :
    (b.linear w bias).Encloses x
      (Tensor.contract (realTensor w) y _ _ [k] .matVec + realTensor bias) := by
  have hbox := (encloses_iff_getScalar _ _ _).mpr h
  have hs := encloses_linear_signSplit (realTensor w) (realTensor bias) _ _ y hbox
  have hs' := (encloses_iff_getScalar _ _ _).mp hs
  rw [← Linear.forward_linear_pair (IBP.matPos (realTensor w)) (IBP.matNeg (realTensor w))
    (realAffine b.lo) (realAffine b.hi) (realTensor bias) x] at hs'
  rw [← Linear.forward_linear_pair (IBP.matPos (realTensor w)) (IBP.matNeg (realTensor w))
    (realAffine b.hi) (realAffine b.lo) (realTensor bias) x] at hs'
  simp only [Encloses, linear, flat, CROWN.Graph.propagateLinearBounds,
    realAffine, realTensor_add, realTensor_matmul,
    realTensor_contract_matVec, realTensor_matPos, realTensor_matNeg]
  exact hs'

/-- For `x` in the box `input` and `b` enclosing `y` at `x`, every coordinate of `y` lies in
the box `b.interval input`, read as reals. -/
theorem Bounds.interval_encloses {n m : Nat} (b : Bounds n m) (input : Box ℚ [n])
    (x : Tensor ℝ [n]) (y : Tensor ℝ [m])
    (hx : Directed.RowEncloses
      ⟨n, realTensor input.lo, realTensor input.hi⟩ n fun i => getAtOrZero x [i])
    (hy : b.Encloses x y) :
    ∀ i, (realTensor (b.interval input).lo).getScalar i ≤ y.getScalar i ∧
      y.getScalar i ≤ (realTensor (b.interval input).hi).getScalar i := by
  intro i
  exact ⟨((affine_interval_encloses b.lo input x hx i).1).trans (hy i).1,
    (hy i).2 |>.trans (affine_interval_encloses b.hi input x hx i).2⟩

/-- Every proposed ReLU lower slope lies in `[0, 1]`, as the relaxation soundness requires. -/
def checkAlpha {m : Nat} (alpha : Tensor ℚ [m]) : Bool :=
  (List.finRange m).all fun i => decide (0 ≤ alpha.getScalar i ∧ alpha.getScalar i ≤ 1)

/-- When `checkAlpha alpha` holds, every entry of `alpha` lies in `[0, 1]`. -/
theorem checkAlpha_sound {m : Nat} (alpha : Tensor ℚ [m]) (h : checkAlpha alpha = true)
    (i : Fin m) : 0 ≤ (realTensor alpha).getScalar i ∧ (realTensor alpha).getScalar i ≤ 1 := by
  have hq := of_decide_eq_true (List.all_eq_true.mp h i (List.mem_finRange i))
  simp only [realTensor_getScalar]
  exact_mod_cast hq

/-- For `x` in the box `input`, `b` enclosing `y` at `x` and slopes `alpha` that pass
`checkAlpha`, the bounds `b.relu input alpha` enclose `Tensor.relu y` at `x`. -/
theorem Bounds.relu_encloses {n m : Nat} (b : Bounds n m) (input : Box ℚ [n])
    (alpha : Tensor ℚ [m]) (ha : checkAlpha alpha = true)
    (x : Tensor ℝ [n]) (y : Tensor ℝ [m])
    (hx : Directed.RowEncloses
      ⟨n, realTensor input.lo, realTensor input.hi⟩ n fun i => getAtOrZero x [i])
    (hy : b.Encloses x y) :
    (b.relu input alpha).Encloses x (Tensor.relu y) := by
  have hpre := b.interval_encloses input x y hx hy
  intro i
  have hai := checkAlpha_sound alpha ha i
  have hl := CROWN.Cert.alphaRelaxLowerScalar_sound
    (lowerBound := (hpre i).1) (upperBound := (hpre i).2)
    (alphaNonnegative := hai.1) (alphaAtMostOne := hai.2)
  have hu := CROWN.ReLU.relaxScalar_sound _ _ _ (hpre i).1
    (hpre i).2
  have sl := alphaRelaxLowerScalar_slope_nonneg
    ((realTensor (b.interval input).lo).getScalar i)
    ((realTensor (b.interval input).hi).getScalar i)
    ((realTensor alpha).getScalar i) hai.1
  have su := relaxScalar_slope_nonneg
    ((realTensor (b.interval input).lo).getScalar i)
    ((realTensor (b.interval input).hi).getScalar i)
  dsimp only [relu]
  simp only [realAffine_propagate, realRelax_lowerVec, realRelax_upperVec,
    Linear.forward_propagateAffine, getScalar_alphaRelaxLowerVec,
    getScalar_relaxVector, Tensor.relu, Tensor.getScalar_map]
  exact ⟨(add_le_add (mul_le_mul_of_nonneg_left (hy i).1 sl) le_rfl).trans hl,
    hu.trans (add_le_add (mul_le_mul_of_nonneg_left (hy i).2 su) le_rfl)⟩

/-- Dense/ReLU networks with a proposed lower-relaxation slope at each ReLU. -/
inductive Network : Nat → Nat → Type where
  /-- An exact dense layer. -/
  | linear {n m : Nat} (layer : Linear ℚ n m) : Network n m
  /-- A ReLU, with the proposed lower-relaxation slope `alpha` of each coordinate. -/
  | relu {n : Nat} (alpha : Tensor ℚ [n]) : Network n n
  /-- One network after another. -/
  | comp {n m k : Nat} : Network n m → Network m k → Network n k

/-- The network's linear and ReLU layers without the proposed slopes. The certificate's
`Monotonicity.Certificate.model`, evaluated on its `Monotonicity.Certificate.params`, is the
network's model. -/
def Network.toCertificate {n m : Nat} : Network n m → Monotonicity.Certificate [n] [m]
  | .linear layer => .linear layer
  | @Network.relu n _ => .relu [n]
  | .comp first rest => .comp first.toCertificate rest.toCertificate

/-- Exact rational execution agrees with the real-valued model after interpreting its output. -/
theorem Network.realTensor_forward {n m : Nat} (net : Network n m) (x : Tensor ℚ [n]) :
    realTensor (net.toCertificate.model.eval (net.toCertificate.params (α := ℚ)) x) =
      net.toCertificate.model.eval (net.toCertificate.params (α := ℝ)) (realTensor x) := by
  have cast_id {s : Shape} (t : Tensor ℚ s) : Tensor.map (fun q : ℚ => (q : ℚ)) t = t := by
    apply Tensor.Rep.ext
    intro coordinate
    simp only [Tensor.map_apply]
  induction net with
  | linear layer =>
      change realTensor (forward ⟨Tensor.map (fun q : ℚ => (q : ℚ)) layer.weights,
          Tensor.map (fun q : ℚ => (q : ℚ)) layer.bias⟩ x) =
        forward ⟨realTensor layer.weights, realTensor layer.bias⟩ (realTensor x)
      simp only [forward, cast_id, realTensor_add, realTensor_contract_matVec]
  | relu alpha => exact realTensor_relu x
  | comp first rest hf hr =>
      simp only [toCertificate, Monotonicity.Certificate.model, Monotonicity.Certificate.params,
        nn.Sequential.eval_comp]
      rw [hr, hf]

/-- Propagate the bounds `b` through the network with the CROWN transfers in exact rational
arithmetic, and `none` when a ReLU slope lies outside `[0, 1]`. -/
def Network.propagate {n m k : Nat} (net : Network m k) (b : Bounds n m)
    (input : Box ℚ [n]) : Option (Bounds n k) :=
  match net with
  | .linear layer => some (b.linear layer.weights layer.bias)
  | .relu alpha => if checkAlpha alpha then some (b.relu input alpha) else none
  | .comp first rest => do
      let mid ← first.propagate b input
      rest.propagate mid input

/-- If `b` encloses `y` at an input `x` of the box and propagation succeeds, the result encloses
the network's output on `y` at `x`. -/
theorem Network.propagate_sound {n m k : Nat} (net : Network m k) (b : Bounds n m)
    (input : Box ℚ [n]) (out : Bounds n k) (h : net.propagate b input = some out)
    (x : Tensor ℝ [n]) (y : Tensor ℝ [m])
    (hx : Directed.RowEncloses
      ⟨n, realTensor input.lo, realTensor input.hi⟩ n fun i => getAtOrZero x [i])
    (hy : b.Encloses x y) :
    out.Encloses x (net.toCertificate.model.eval net.toCertificate.params y) := by
  induction net with
  | linear layer =>
      cases Option.some.inj h
      exact b.linear_encloses layer.weights layer.bias x y hy
  | relu alpha =>
      simp only [propagate] at h
      split at h
      · rename_i ha
        cases Option.some.inj h
        exact b.relu_encloses input alpha ha x y hx hy
      · contradiction
  | comp first rest hf hr =>
      simp only [propagate, Option.bind_eq_bind] at h
      cases hm : first.propagate b input with
      | none => simp [hm] at h
      | some mid =>
          have ht : rest.propagate mid input = some out := by simpa [hm] using h
          simp only [toCertificate, Monotonicity.Certificate.model,
            Monotonicity.Certificate.params, nn.Sequential.eval_comp]
          exact hr mid out ht (first.toCertificate.model.eval first.toCertificate.params y)
            (hf b mid hm y hy)

/-- A safety query: a network, an input box and a conjunction of affine output inequalities, each
row of `weights * output + bias` negative when `strict` is set and nonpositive otherwise. -/
structure Query (n m : Nat) where
  /-- The network. -/
  network : Network n m
  /-- The input box. -/
  input : Box ℚ [n]
  /-- The number of output inequalities. -/
  numConstraints : Nat
  /-- The rows `weights * output + bias` of the output inequalities. -/
  inequalities : Linear ℚ m numConstraints
  /-- Whether each row must be negative, rather than nonpositive. -/
  strict : Bool

/-- Reject reversed boxes and empty input, output, or query dimensions. -/
def Query.wellFormed {n m : Nat} (q : Query n m) : Bool :=
  decide (0 < n ∧ 0 < m ∧ 0 < q.numConstraints) &&
    Tensor.boundsOrdered q.input.lo q.input.hi

/-- The executable safety check of a well-formed query: the bounds recomputed from the identity at
the input, through the network and the inequalities, must pass `checkUpper` over the box. No
externally supplied affine bounds are trusted. -/
def Query.check {n m : Nat} (q : Query n m) : Bool :=
  q.wellFormed && match q.network.propagate (Bounds.identity n) q.input with
  | none => false
  | some out => checkUpper q.strict
      (out.linear q.inequalities.weights q.inequalities.bias).hi q.input

/-- Every real input in the decoded box satisfies every decoded output inequality. -/
def Query.Safe {n m : Nat} (q : Query n m) : Prop :=
  ∀ x : Tensor ℝ [n],
    Directed.RowEncloses ⟨n, realTensor q.input.lo, realTensor q.input.hi⟩ n
      (fun i => getAtOrZero x [i]) →
    ∀ i : Fin q.numConstraints,
      let result := (Tensor.contract (realTensor q.inequalities.weights)
          (q.network.toCertificate.model.eval q.network.toCertificate.params x) _ _
          [q.numConstraints] .matVec) +
        realTensor q.inequalities.bias
      if q.strict then result.getScalar i < 0 else result.getScalar i ≤ 0

/-- An accepted query is safe, with no unproved transfer or coverage premise. -/
@[verify] theorem Query.check_sound {n m : Nat} (q : Query n m) (h : q.check = true) : q.Safe := by
  simp only [check, Bool.and_eq_true] at h
  obtain ⟨_, hc⟩ := h
  cases hp : q.network.propagate (Bounds.identity n) q.input with
  | none => simp [hp] at hc
  | some out =>
      have hg : checkUpper q.strict
          (out.linear q.inequalities.weights q.inequalities.bias).hi q.input = true := by
        simpa [hp] using hc
      intro x hx i
      have hn := q.network.propagate_sound (Bounds.identity n) q.input out hp x x hx
        (Bounds.identity_encloses x)
      have hq := out.linear_encloses q.inequalities.weights q.inequalities.bias x
        (q.network.toCertificate.model.eval q.network.toCertificate.params x) hn
      have hu := checkUpper_sound q.strict _ q.input hg x hx i
      have hi := (hq i).2
      cases hstrict : q.strict <;> simp only [hstrict, Bool.false_eq_true, ↓reduceIte] at hu ⊢
      · exact hi.trans hu
      · exact hi.trans_lt hu

end TorchLean.Certificate.CROWNQuery
