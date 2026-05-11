/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import Mathlib.Analysis.SpecialFunctions.Pow.Real
public import NN.Floats.NeuralFloat.Core
public import NN.MLTheory.Proofs.Approximation.FloatInterval.Semantics

/-!
# Exact Interval Images for Rounded Targets

Structured theorem statements for exact interval images of rounded floating-point targets,
specialized to `IEEE32Exec`.

This file defines correctly-rounded activation assumptions, separating-activation assumptions,
finite σ-networks, exact interval semantics, and the pipeline theorem:

`correctly rounded activation + separating construction + exact semantics construction`
implies exact interval images for every finite rounded target on `[-1,1]^d`.
-/

@[expose] public section


namespace NN.MLTheory.Proofs.UniversalApproximation

open TorchLean.Floats
open TorchLean.Floats.IEEE754

namespace FloatIntervalApprox

open IEEE32Exec

noncomputable section

/-!
## Separating activation condition, specialized to `IEEE32Exec`

Source: Hwang et al. (`arXiv:2506.16065`), Condition 1 and its correctly-rounded sufficient
conditions.
-/

namespace Condition1

open IEEE32Exec

/-- Alias for the executable float format used throughout this file (`IEEE32Exec`). -/
abbrev F : Type := IEEE32Exec

 /-! ### Float-format constants for IEEE binary32 -/

 -- IEEE-754 binary32 parameters used by the paper.
 /-- Mantissa bitwidth `M` for IEEE binary32 (excluding the hidden leading bit). -/
 def M : Nat := 23
 /-- Minimum normal exponent `emin` for IEEE binary32. -/
 def emin : Int := -126
 /-- Maximum normal exponent `emax` for IEEE binary32. -/
 def emax : Int := 127

-- Machine epsilon ε = 2^{-23} (as a real number).
noncomputable def ε : ℝ := neuralBpow binaryRadix (-23)

-- Smallest positive float ω = 2^{-149} (as a real number).
noncomputable def ω : ℝ := neuralBpow binaryRadix (-149)

noncomputable def pow2 (k : Int) : ℝ := neuralBpow binaryRadix k

/-! ### Basic helpers -/

def finite (x : F) : Prop := IEEE32Exec.isFinite x = true

noncomputable def rabs (x : F) : ℝ := |IEEE32Exec.toReal x|

noncomputable def between (a b x : F) : Prop :=
  let lo := IEEE32Exec.minimum a b
  let hi := IEEE32Exec.maximum a b
  lo ≤ x ∧ x ≤ hi

/-! ### Separating activation condition -/

structure Witness (σ : F → F) where
  /-- First finite input witnessing the zero anchor in the separating condition. -/
  c1 : F
  /-- Second finite input used to create a nonzero separated activation value. -/
  c2 : F
  /-- The first witness input is finite. -/
  c1_finite : finite c1
  /-- The second witness input is finite. -/
  c2_finite : finite c2
  /-- The activation sends the first witness to zero. -/
  sigma_c1_eq_zero : σ c1 = Numbers.zero
  /-- The activation value at the second witness is finite. -/
  sigma_c2_finite : finite (σ c2)
  /-- The second activation value has the magnitude required by the binary32 separation bound. -/
  sigma_c2_abs_mem :
    (rabs (σ c2) ∈ Set.Icc (ε / 2 + 2 * ε^2) (5 / 4 - 2 * ε))
  /-- At least one witness input is not too close to underflow. -/
  max_abs_c1_c2_ge :
    max (rabs c1) (rabs c2) ≥ pow2 (emin + 1)
  /-- Between `c1` and `c2`, activation values stay between the endpoint activation values. -/
  sigma_between_on_Icc :
    ∀ x : F, c1 ≤ x → x ≤ c2 → between (σ c1) (σ c2) (σ x)

  /-- Threshold witness for the local binary32 separation step. -/
  η : F
  /-- The threshold witness is finite. -/
  η_finite : finite η
  /-- The threshold witness lies in the binary32 magnitude window required by the theorem. -/
  η_abs_mem :
    (rabs η ∈ Set.Icc (pow2 (emin + 5)) (4 - 8 * ε))
  /-- The activation value at the threshold is finite. -/
  sigma_eta_finite : finite (σ η)
  /-- The activation value at the next float above the threshold is finite. -/
  sigma_etaPlus_finite : finite (σ (IEEE32Exec.nextUp η))
  /-- The threshold activation magnitude is bounded in the required binary32 window. -/
  sigma_eta_abs_mem :
    (rabs (σ η) ∈ Set.Icc (pow2 (emin + 5)) (pow2 (emax - 6) * rabs η))
  /-- The next-up activation magnitude is bounded in the required binary32 window. -/
  sigma_etaPlus_abs_mem :
    (rabs (σ (IEEE32Exec.nextUp η)) ∈ Set.Icc (pow2 (emin + 5)) (pow2 (emax - 6) * rabs η))
  /-- The activation separates values below and above the threshold interval. -/
  threshold_separates :
    ∀ x y : F, x ≤ η → η < IEEE32Exec.nextUp η → IEEE32Exec.nextUp η ≤ y →
      (σ x ≤ σ η ∧ σ η < σ (IEEE32Exec.nextUp η) ∧ σ (IEEE32Exec.nextUp η) ≤ σ y) ∨
      (σ x ≥ σ η ∧ σ η > σ (IEEE32Exec.nextUp η) ∧ σ (IEEE32Exec.nextUp η) ≥ σ y)

  /-- Real Lipschitz envelope around the threshold. -/
  lam : ℝ
  /-- The envelope constant is in the paper's binary32-safe range. -/
  lam_mem :
    (lam ∈ Set.Icc (0 : ℝ) (pow2 (emax - 7) * min (rabs (σ η)) (pow2 (Int.ofNat (M + 3)))))
  /-- Activation values obey the Lipschitz envelope on both sides of the threshold gap. -/
  lipschitz_around_threshold :
    ∀ x y : F, x ≤ η → η < IEEE32Exec.nextUp η → IEEE32Exec.nextUp η ≤ y →
      (|IEEE32Exec.toReal (σ x) - IEEE32Exec.toReal (σ η)| ≤ lam * |IEEE32Exec.toReal x -
        IEEE32Exec.toReal η|) ∧
      (|IEEE32Exec.toReal (σ y) - IEEE32Exec.toReal (σ (IEEE32Exec.nextUp η))| ≤
          lam * |IEEE32Exec.toReal y - IEEE32Exec.toReal (IEEE32Exec.nextUp η)|)

 /-- Separating activation condition for `σ`, packaged as existence of a `Witness`. -/
 def Holds (σ : F → F) : Prop := Nonempty (Witness σ)

/-! ### Correctly-rounded activations and real sufficient conditions -/

structure CorrectlyRounded (ρ : ℝ → ℝ) (σ : F → F) : Prop where
  /-- Finite executable inputs evaluate to finite executable outputs with the declared rounding law. -/
  finite_input_implies :
    ∀ x : F, finite x → finite (σ x) ∧ IEEE32Exec.toReal (σ x) = IEEE32Exec.fp32Round (ρ
      (IEEE32Exec.toReal x))

 /--
Real-valued sufficient conditions used to prove that a correctly-rounded activation satisfies the
separating activation condition.

This is a “real” analogue of `Witness` that talks about a target function `ρ : ℝ → ℝ`.
-/
 structure RealWitness (ρ : ℝ → ℝ) where
  /-- First binary32 input used by the real sufficient conditions. -/
  c1' : F
  /-- Second binary32 input used by the real sufficient conditions. -/
  c2' : F
  /-- The first input is finite. -/
  c1'_finite : finite c1'
  /-- The second input is finite. -/
  c2'_finite : finite c2'
  /-- The real activation is close to zero at the first witness. -/
  rho_c1'_small : |ρ (IEEE32Exec.toReal c1')| ≤ ω / 2
  /-- The real activation has the required magnitude at the second witness. -/
  rho_c2'_range : |ρ (IEEE32Exec.toReal c2')| ∈ Set.Icc (ε / 2 + 2 * ε^2) (5 / 4 - 2 * ε)
  /-- At least one real witness input has magnitude safely above underflow. -/
  max_abs_c1'_c2'_ge :
    max (rabs c1') (rabs c2') ≥ pow2 (emin + 1)
  /-- On the witness interval, `ρ` stays between its endpoint values. -/
  rho_between_on_Icc :
    ∀ x : ℝ,
      IEEE32Exec.toReal c1' ≤ x → x ≤ IEEE32Exec.toReal c2' →
        (min (ρ (IEEE32Exec.toReal c1')) (ρ (IEEE32Exec.toReal c2')) ≤ ρ x ∧
          ρ x ≤ max (ρ (IEEE32Exec.toReal c1')) (ρ (IEEE32Exec.toReal c2')))

  /-- Real-valued threshold location used by the sufficient conditions. -/
  δ : ℝ
  /-- The threshold lies in the specified central window. -/
  δ_abs_mem : |δ| ∈ Set.Icc (3 / 8 : ℝ) (7 / 8 : ℝ)
  /-- Threshold separation and local growth conditions for the real activation. -/
  rho_threshold_properties :
    (∀ x y : ℝ,
      x ≤ δ - 1 / 8 → δ + 1 / 8 ≤ y →
        (ρ x ≤ ρ (δ - 1 / 8) ∧ ρ (δ - 1 / 8) < ρ (δ + 1 / 8) ∧ ρ (δ + 1 / 8) ≤ ρ y) ∨
        (ρ x ≥ ρ (δ - 1 / 8) ∧ ρ (δ - 1 / 8) > ρ (δ + 1 / 8) ∧ ρ (δ + 1 / 8) ≥ ρ y))
    ∧ (∀ x y : ℝ,
      x ∈ Set.Icc (δ - 1 / 8) (δ + 1 / 8) →
      y ∈ Set.Icc (δ - 1 / 8) (δ + 1 / 8) →
        (|ρ x| ∈ Set.Icc (1 / 4 : ℝ) (1 : ℝ)) ∧ (|ρ x - ρ y| > (1 / 8 : ℝ) * |x - y|))

  /-- Lipschitz constant for the real activation. -/
  lam : ℝ
  /-- The Lipschitz constant is in the binary32-safe range required by the theorem. -/
  lam_mem : lam ∈ Set.Icc (0 : ℝ) ((1 / 5 : ℝ) * pow2 (emax - 9))
  /-- Global Lipschitz bound for the real activation. -/
  lipschitz : ∀ x y : ℝ, |ρ x - ρ y| ≤ lam * |x - y|

 /-- Existence of a `RealWitness` for `ρ`. -/
 def RealSufficientConditions (ρ : ℝ → ℝ) : Prop := Nonempty (RealWitness ρ)

 /--
Correct-rounding bridge: a real activation satisfying the sufficient conditions induces a rounded
floating-point activation satisfying the separating activation condition.
-/
 def correctRoundingSatisfiesSeparatingActivation (ρ : ℝ → ℝ) (σ : F → F) : Prop :=
  CorrectlyRounded ρ σ → RealSufficientConditions ρ → Holds σ

end Condition1

namespace PaperStatements

open I OpsExact ExactImage

/-! ## Finite σ-network model -/

namespace SigmaNet

/-- Parameters of an affine layer `din → dout` over `IEEE32Exec` scalars. -/
structure Affine (din dout : Nat) where
  /-- W. -/
  W : Fin dout → Fin din → F
  /-- b. -/
  b : Fin dout → F

/-- Apply an affine layer to an input vector using IEEE32Exec primitives. -/
def aff {din dout : Nat} (A : Affine din dout) (x : Fin din → F) : Fin dout → F :=
  fun i =>
    let s :=
      (List.finRange din).foldl
        (fun acc j => IEEE32Exec.add acc (IEEE32Exec.mul (A.W i j) (x j)))
        (Numbers.zero : F)
    IEEE32Exec.add s (A.b i)

/-- A feedforward σ-network: affine layers with σ between them, no σ after the final affine. -/
inductive Net : Nat → Nat → Type
  | last {din dout : Nat} (A : Affine din dout) : Net din dout
  | step {din mid dout : Nat} (A : Affine din mid) (n : Net mid dout) : Net din dout

/-- Evaluate a σ-network on a concrete input vector. -/
def eval (σ : F → F) : {din dout : Nat} → Net din dout → (Fin din → F) → (Fin dout → F)
  | _, _, Net.last A, x => aff A x
  | _, _, Net.step A n, x => eval σ n (fun i => σ (aff A x i))

/-- Specialized evaluator for scalar-output (`dout = 1`) networks. -/
def eval1 {din : Nat} (σ : F → F) (n : Net din 1) (x : Fin din → F) : F :=
  (eval σ n x) 0

/-! ## Interval semantics using `OpsExact` -/

def sigmaSharp (σ : F → F) (J : I) : I :=
  OpsExact.hull <| (OpsExact.γFinsetI J).image σ

/-- Interval semantics of an affine layer, using the exact interval ops from `OpsExact`. -/
def affSharp {din dout : Nat} (A : Affine din dout) (B : I.Box din) : I.Box dout :=
  fun i =>
    let terms : Fin din → I := fun j => OpsExact.mulSharp (I.range (A.W i j) (A.W i j)) (B j)
    let s := OpsExact.sumSharp terms
    OpsExact.addSharp s (I.range (A.b i) (A.b i))

/-- Interval semantics for a network: apply `affSharp` and then push the activation through
  `sigmaSharp`. -/
def evalSharp (σ : F → F) : {din dout : Nat} → Net din dout → I.Box din → I.Box dout
  | _, _, Net.last A, B => affSharp A B
  | _, _, Net.step A n, B => evalSharp σ n (fun i => sigmaSharp σ ((affSharp A B) i))

/-- Specialized interval evaluator for scalar-output (`dout = 1`) networks. -/
def evalSharp1 {din : Nat} (σ : F → F) (n : Net din 1) (B : I.Box din) : I :=
  (evalSharp σ n B) 0

end SigmaNet

/-! ## Bounded interval domains and direct-image hulls -/

def IntervalIn (a b : F) (J : I) : Prop :=
  ∀ ⦃x : F⦄, x ∈ J → a ≤ x ∧ x ≤ b

/-- A product box `B` lies in `I[a,b]^d` (componentwise). -/
def BoxIn {d : Nat} (a b : F) (B : I.Box d) : Prop :=
  ∀ i, IntervalIn a b (B i)

/-- The canonical cube domain `[-1,1]^d`. -/
def CubeBox {d : Nat} (B : I.Box d) : Prop :=
  BoxIn (a := Numbers.neg_one) (b := Numbers.one) B

/-- Ideal abstraction `h♯(B)`: interval hull of the direct image `h(γ(B))` (computed over the finite
  concretization). -/
def idealSharp {d : Nat} (h : (Fin d → F) → F) (B : I.Box d) : I :=
  OpsExact.hull <| (OpsExact.γFinsetBox B).image h

/-! ## Indicator functions and separability -/

namespace Indicators

/-- Indicator of a set `S ⊆ F^d`, returning `1`/`0` in `F`. -/
noncomputable def ι {d : Nat} (S : Set (Fin d → F)) : (Fin d → F) → F := by
  classical
  exact fun x => if x ∈ S then (Numbers.one : F) else (Numbers.zero : F)

/-- Strict-threshold indicator `ι_{>a}` on `F`. -/
noncomputable def ιGt (a : F) : F → F :=
  fun x => if a < x then (Numbers.one : F) else (Numbers.zero : F)

/-- Non-strict threshold indicator `ι_{≥a}` on `F`. -/
noncomputable def ιGe (a : F) : F → F :=
  fun x => if a ≤ x then (Numbers.one : F) else (Numbers.zero : F)

/-- Strict-threshold indicator `ι_{<a}` on `F`. -/
noncomputable def ιLt (a : F) : F → F :=
  fun x => if x < a then (Numbers.one : F) else (Numbers.zero : F)

/-- Non-strict threshold indicator `ι_{≤a}` on `F`. -/
noncomputable def ιLe (a : F) : F → F :=
  fun x => if x ≤ a then (Numbers.one : F) else (Numbers.zero : F)

/-- Scaled (by `K`) indicator: `K ⊗ ι`. -/
noncomputable def scale (K : F) (g : F → F) : F → F :=
  fun x => IEEE32Exec.mul K (g x)

end Indicators

namespace Separability

open SigmaNet Indicators

/-!
We model “there exists a σ-network implementing a scaled threshold-indicator exactly under interval
semantics on `I[a,b]`” using our `SigmaNet.Net` interval interpreter `evalSharp1`.
-/

def IntervalDomain (a b : F) : Set I :=
  {J | IntervalIn (a := a) (b := b) J}

 /-- “Ideal” abstraction `g♯` for a unary function `g : F → F` on an input interval `J`. -/
 def unaryIdealSharp (g : F → F) (J : I) : I :=
   OpsExact.hull <| (OpsExact.γFinsetI J).image g

 /--
Separability of `σ` on `I[a,b]` with threshold `η` and scale `K`.

Informally, this asserts existence of σ-networks whose **interval semantics** coincides with the
ideal abstraction of scaled threshold indicators (`ι_{≤z}`, `ι_{≥z}`, and `ι_{>η}`).
-/
 def SeparableOn (σ : F → F) (a b η K : F) : Prop :=
   (∀ z : F,
       (a ≤ z ∧ z ≤ b) →
      (∃ ϕle : SigmaNet.Net 1 1,
          ∀ J, J ∈ IntervalDomain (a := a) (b := b) → SigmaNet.evalSharp1 σ ϕle (fun _ => J) =
            unaryIdealSharp (scale K (ιLe z)) J) ∧
        (∃ ϕge : SigmaNet.Net 1 1,
          ∀ J, J ∈ IntervalDomain (a := a) (b := b) → SigmaNet.evalSharp1 σ ϕge (fun _ => J) =
            unaryIdealSharp (scale K (ιGe z)) J)) ∧
  (∃ ψη : SigmaNet.Net 1 1,
      ∀ J, J ∈ IntervalDomain (a := a) (b := b) → SigmaNet.evalSharp1 σ ψη (fun _ => J) =
        unaryIdealSharp (scale K (ιGt η)) J)

end Separability

/-! ## Finite min/max witnesses for ideal hulls -/

namespace IdealMinMax

open OpsExact

private theorem not_exists_isNaN_true_of_forall_isNaN_false (s : Finset F)
    (hn : ∀ z ∈ s, IEEE32Exec.isNaN z = false) :
    ¬∃ z ∈ s, IEEE32Exec.isNaN z = true := by
  intro hex
  rcases hex with ⟨z, hz, hzNaN⟩
  have hzFalse : IEEE32Exec.isNaN z = false := hn z hz
  have hzNaN' := hzNaN
  rw [hzFalse] at hzNaN'
  cases hzNaN'

 /-- `chooseMin` is below every element of the finset (under the no-NaN side condition). -/
theorem chooseMin_le_of_mem (s : Finset F) (hs : s.Nonempty)
    (hn : ∀ z ∈ s, IEEE32Exec.isNaN z = false) {y : F} (hy : y ∈ s) :
    OpsExact.chooseMin s hs ≤ y := by
  classical
  have hminNaN : IEEE32Exec.isNaN (OpsExact.chooseMin s hs) = false :=
    hn _ (OpsExact.chooseMin_spec s hs).1
  have hyNaN : IEEE32Exec.isNaN y = false := hn _ hy
  have hminE : OpsExact.toERealTotal (OpsExact.chooseMin s hs) ≤ OpsExact.toERealTotal y := by
    have hyImg : OpsExact.toERealTotal y ∈ s.image OpsExact.toERealTotal := Finset.mem_image_of_mem
      _ hy
    simpa [(OpsExact.chooseMin_spec s hs).2] using
      Finset.min'_le (s.image OpsExact.toERealTotal) (OpsExact.toERealTotal y) hyImg
  exact
    (OpsExact.le_iff_toERealTotal_le_of_isNaN_false (x := OpsExact.chooseMin s hs) (y := y) hminNaN
      hyNaN).2 hminE

 /-- `chooseMax` is above every element of the finset (under the no-NaN side condition). -/
theorem le_chooseMax_of_mem (s : Finset F) (hs : s.Nonempty)
    (hn : ∀ z ∈ s, IEEE32Exec.isNaN z = false) {y : F} (hy : y ∈ s) :
    y ≤ OpsExact.chooseMax s hs := by
  classical
  have hmaxNaN : IEEE32Exec.isNaN (OpsExact.chooseMax s hs) = false :=
    hn _ (OpsExact.chooseMax_spec s hs).1
  have hyNaN : IEEE32Exec.isNaN y = false := hn _ hy
  have hmaxE : OpsExact.toERealTotal y ≤ OpsExact.toERealTotal (OpsExact.chooseMax s hs) := by
    have hyImg : OpsExact.toERealTotal y ∈ s.image OpsExact.toERealTotal := Finset.mem_image_of_mem
      _ hy
    simpa [(OpsExact.chooseMax_spec s hs).2] using
      Finset.le_max' (s.image OpsExact.toERealTotal) (OpsExact.toERealTotal y) hyImg
  exact
    (OpsExact.le_iff_toERealTotal_le_of_isNaN_false (x := y) (y := OpsExact.chooseMax s hs) hyNaN
      hmaxNaN).2 hmaxE

 /--
Existence of min/max witnesses for `idealSharp` on a nonempty box domain.

This packages the interval hull characterization of `idealSharp` as an `Icc m M`.
-/
theorem exists_minmax_for_idealSharp {d : Nat} (h : (Fin d → F) → F) (B : I.Box d)
    (hB : (I.γ (d := d) B).Nonempty)
    (hnan : ∀ x, IEEE32Exec.isNaN (h x) = false) :
    ∃ m M,
      IsMinOn h (I.γ (d := d) B) m ∧
      IsMaxOn h (I.γ (d := d) B) M ∧
      I.γI (idealSharp (d := d) h B) = Icc m M := by
  classical
  -- Work with the concrete image finset.
  let s : Finset F := (OpsExact.γFinsetBox B).image h
  have hs_nonempty : s.Nonempty := by
    rcases hB with ⟨x0, hx0⟩
    have hx0' : x0 ∈ OpsExact.γFinsetBox B := by
      -- unfold `γFinsetBox` membership
      have : ∀ i, x0 i ∈ B i := hx0
      simpa [OpsExact.mem_γFinsetBox_iff] using this
    exact ⟨h x0, Finset.mem_image_of_mem _ hx0'⟩
  have hn : ∀ z ∈ s, IEEE32Exec.isNaN z = false := by
    intro z hz
    rcases Finset.mem_image.mp hz with ⟨x, hx, rfl⟩
    exact hnan x
  have hnoNaN : ¬∃ z ∈ s, IEEE32Exec.isNaN z = true :=
    not_exists_isNaN_true_of_forall_isNaN_false s hn
  -- Define min/max witnesses.
  let m : F := OpsExact.chooseMin s hs_nonempty
  let M : F := OpsExact.chooseMax s hs_nonempty
  refine ⟨m, M, ?_, ?_, ?_⟩
  · -- IsMinOn
    refine And.intro ?_ ?_
    · -- existence of arg attaining `m`
      have hm_mem : m ∈ s := (OpsExact.chooseMin_spec s hs_nonempty).1
      rcases Finset.mem_image.mp hm_mem with ⟨x, hx, hxEq⟩
      refine ⟨x, ?_, ?_⟩
      · -- x ∈ γ(B)
        intro i
        have : ∀ i, x i ∈ B i := (OpsExact.mem_γFinsetBox_iff (B := B) (x := x)).1 hx
        exact this i
      · simpa [m] using hxEq
    · -- lower endpoint branch
      intro y hy
      rcases hy with ⟨x, hxB, hxy⟩
      subst hxy
      have hx_fin : x ∈ OpsExact.γFinsetBox B := by
        -- `γFinsetBox` is the finset representation of `γ`.
        exact (OpsExact.mem_γFinsetBox_iff (B := B) (x := x)).2 hxB
      have hy_mem : h x ∈ s := Finset.mem_image_of_mem _ hx_fin
      simpa [m] using (chooseMin_le_of_mem s hs_nonempty hn hy_mem)
  · -- IsMaxOn
    refine And.intro ?_ ?_
    ·
      have hM_mem : M ∈ s := (OpsExact.chooseMax_spec s hs_nonempty).1
      rcases Finset.mem_image.mp hM_mem with ⟨x, hx, hxEq⟩
      refine ⟨x, ?_, ?_⟩
      · intro i
        have : ∀ i, x i ∈ B i := (OpsExact.mem_γFinsetBox_iff (B := B) (x := x)).1 hx
        exact this i
      · simpa [M] using hxEq
    ·
      intro y hy
      rcases hy with ⟨x, hxB, hxy⟩
      subst hxy
      have hx_fin : x ∈ OpsExact.γFinsetBox B := by
        exact (OpsExact.mem_γFinsetBox_iff (B := B) (x := x)).2 hxB
      have hy_mem : h x ∈ s := Finset.mem_image_of_mem _ hx_fin
      simpa [M] using (le_chooseMax_of_mem s hs_nonempty hn hy_mem)
  · -- concretization equality
    -- `idealSharp h B = hull s`.
    have hideal : idealSharp (d := d) h B = OpsExact.hull s := by
      rfl
    -- Under the no-NaN proof, `hull s` is `[m,M]`.
    have hhull : OpsExact.hull s = I.range m M := by
      unfold OpsExact.hull
      simp [hnoNaN, hs_nonempty, m, M]
    -- Finish.
    ext x
    -- Unfold the let-binding `s` so `hhull` can rewrite.
    -- `simp` reduces everything except the definitional `Set` membership on `F → Prop`.
    simp [idealSharp, s, hhull, I.γI]
    change (m ≤ x ∧ x ≤ M) ↔ (m ≤ x ∧ x ≤ M)
    rfl

end IdealMinMax

/-! ## Exact interval images from separating threshold networks -/

namespace ExactImageFromSeparability

open SigmaNet IdealMinMax

/-! ## Separability and exact-semantics premises -/

/-- Separating activation condition specialized to `IEEE32Exec`. -/
abbrev SeparatingActivation (σ : F → F) : Prop := Condition1.Holds σ

/-!
Premise: the separating activation condition yields exact threshold-indicator networks on
`[-1,1]`.
-/
def separatingActivationYieldsThresholdNetworksOnCube (σ : F → F) : Prop :=
  ∀ w : Condition1.Witness σ,
    Separability.SeparableOn σ (a := Numbers.neg_one) (b := Numbers.one) w.η (σ w.c2)

/--
Premise: separability yields a σ-network whose interval semantics equals the direct-image hull.
-/
def thresholdNetworksYieldExactIntervalSemantics (σ : F → F) : Prop :=
  ∀ {a b η K : F},
    Separability.SeparableOn σ (a := a) (b := b) η K →
      ∀ {d : Nat} (h : (Fin d → F) → F),
        (∀ x, IEEE32Exec.isNaN (h x) = false) →
        ∃ n : SigmaNet.Net d 1,
          ∀ B, BoxIn (d := d) (a := a) (b := b) B → SigmaNet.evalSharp1 σ n B = idealSharp (d := d)
            h B

/-- For any NaN-free rounded target `h`, there exists a σ-network with exact interval semantics. -/
def exactIntervalSemanticsUniversalOnCube (σ : F → F) : Prop :=
  ∀ {d : Nat} (h : (Fin d → F) → F),
    (∀ x, IEEE32Exec.isNaN (h x) = false) →
    ∃ n : SigmaNet.Net d 1,
      ∀ B, CubeBox (d := d) B → SigmaNet.evalSharp1 σ n B = idealSharp (d := d) h B

/-- Separating activations and threshold-network composition imply exact interval semantics. -/
theorem exactIntervalSemantics_universalOnCube_of_condition1_and_separableOn (σ : F → F) :
    SeparatingActivation σ →
      separatingActivationYieldsThresholdNetworksOnCube σ →
        thresholdNetworksYieldExactIntervalSemantics σ →
          exactIntervalSemanticsUniversalOnCube σ := by
  intro hcond hL2 hL3 d h hnan
  rcases hcond with ⟨w⟩
  have hsep : Separability.SeparableOn σ (a := Numbers.neg_one) (b := Numbers.one) w.η (σ w.c2) :=
    hL2 w
  rcases hL3 (a := Numbers.neg_one) (b := Numbers.one) (η := w.η) (K := σ w.c2) hsep (d := d) (h :=
    h) hnan with
    ⟨n, hn⟩
  refine ⟨n, ?_⟩
  intro B hB
  -- `CubeBox` is `BoxIn [-1,1]`.
  exact hn B hB

/--
Exact interval-image theorem for rounded targets: for every NaN-free rounded target `fHat`, there is
a σ-network whose interval semantics is exactly the min/max hull of `fHat '' γ(B)` on every cube
box.
-/
def roundedTargetExactIntervalImage (σ : F → F) : Prop :=
  ∀ {d : Nat} (fHat : (Fin d → F) → F),
    (∀ x, IEEE32Exec.isNaN (fHat x) = false) →
    ∃ n : SigmaNet.Net d 1,
      ∀ B, CubeBox (d := d) B → (I.γ (d := d) B).Nonempty →
        ∃ m M,
          IsMinOn fHat (I.γ (d := d) B) m ∧
          IsMaxOn fHat (I.γ (d := d) B) M ∧
          I.γI (SigmaNet.evalSharp1 σ n B) = Icc m M

/-- Derive exact interval images from exact interval semantics by choosing finite min/max witnesses. -/
theorem roundedTargetExactIntervalImage_of_exactIntervalSemantics (σ : F → F) :
    exactIntervalSemanticsUniversalOnCube σ → roundedTargetExactIntervalImage σ := by
  intro hL3 d fHat hfHat
  rcases hL3 (d := d) (h := fHat) hfHat with ⟨n, hn⟩
  refine ⟨n, ?_⟩
  intro B hB hne
  have hEq : SigmaNet.evalSharp1 σ n B = idealSharp (d := d) fHat B := hn B hB
  -- Choose min/max from the ideal hull.
  rcases IdealMinMax.exists_minmax_for_idealSharp (d := d) (h := fHat) (B := B) hne hfHat with
    ⟨m, M, hmin, hmax, hgamma⟩
  refine ⟨m, M, hmin, hmax, ?_⟩
  simpa [hEq] using hgamma

/-! ## Correct rounding to exact interval images -/

open Condition1

/--
Pipeline theorem: correctly-rounded real activations plus separating-threshold constructions imply
exact interval images for rounded targets.
-/
theorem roundedTargetExactIntervalImage_of_correctRounding
    (ρ : ℝ → ℝ) (σ : F → F)
    (hRound : Condition1.correctRoundingSatisfiesSeparatingActivation ρ σ)
    (hCR : Condition1.CorrectlyRounded ρ σ)
    (hReal : Condition1.RealSufficientConditions ρ)
    (hThresholds : separatingActivationYieldsThresholdNetworksOnCube σ)
    (hExactConstruction : thresholdNetworksYieldExactIntervalSemantics σ) :
    roundedTargetExactIntervalImage σ := by
  have hCond1 : SeparatingActivation σ := hRound hCR hReal
  have hExact : exactIntervalSemanticsUniversalOnCube σ :=
    exactIntervalSemantics_universalOnCube_of_condition1_and_separableOn (σ := σ) hCond1
      hThresholds hExactConstruction
  show roundedTargetExactIntervalImage σ
  exact roundedTargetExactIntervalImage_of_exactIntervalSemantics (σ := σ) hExact

end ExactImageFromSeparability

end PaperStatements

end

end FloatIntervalApprox

end NN.MLTheory.Proofs.UniversalApproximation
