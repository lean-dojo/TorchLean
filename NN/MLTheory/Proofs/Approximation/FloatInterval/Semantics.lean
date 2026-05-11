/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import Mathlib.Data.EReal.Basic
public import Mathlib.Data.Fin.Tuple.Basic
public import Mathlib.Data.Set.Image
public import NN.Floats.IEEEExec.BridgeERealTotal
public import NN.Floats.IEEEExec.Exec32

/-!
# Floating-Point Interval Semantics

Interval-domain semantics for `IEEE32Exec` neural networks.

This file formalizes the interval domain, concretization map, executable interval operators, and
the exact-interval-image property used by the floating-point interval-approximation theorem of
Hwang, Lee, Park, Park, and Saad, *Floating-Point Neural Networks Are Provably Robust Universal
Approximators* (`arXiv:2506.16065`).
-/

@[expose] public section


namespace NN.MLTheory.Proofs.UniversalApproximation

open TorchLean.Floats.IEEE754

namespace FloatIntervalApprox

open IEEE32Exec

noncomputable section

/-! ## Basic aliases -/

/-- Shorthand for the executable binary32 float type used in this development. -/
abbrev F : Type := IEEE32Exec

/-!
`IEEE32Exec` is stored as a `UInt32` bit-pattern, so the carrier is finite. We use this only to
obtain `Finset.univ` for paper-style “finite hull” definitions; nothing is computed.
-/

namespace DecidableInstances

open IEEE32Exec

instance : DecidableRel (fun x y : F => x ≤ y) := by
  classical
  intro x y
  infer_instance

end DecidableInstances

namespace FintypeInstances

noncomputable instance : Finite UInt32 := by
  classical
  refine Finite.of_injective (fun u : UInt32 => (⟨u.toNat, u.toNat_lt⟩ : Fin (2 ^ 32))) ?_
  intro a b hab
  have : a.toNat = b.toNat := by
    simpa using congrArg Fin.val hab
  exact (UInt32.toNat_inj).1 this

noncomputable instance : Fintype UInt32 := by
  classical
  exact Fintype.ofFinite UInt32

noncomputable instance : Finite IEEE32Exec := by
  classical
  refine Finite.of_injective IEEE32Exec.toBits ?_
  intro a b hab
  cases a
  cases b
  cases hab
  rfl

noncomputable instance : Fintype IEEE32Exec := by
  classical
  exact Fintype.ofFinite IEEE32Exec

end FintypeInstances

/-! ## Small helper lemmas about the `IEEE32Exec` order -/

namespace ExecLemmas

open IEEE32Exec

theorem compare_self_of_isNaN_false (x : F) (hx : IEEE32Exec.isNaN x = false) :
    IEEE32Exec.compare x x = some .eq := by
  classical
  cases hinf : IEEE32Exec.isInf x with
  | true =>
      -- `compare` short-circuits on infinities.
      simp [IEEE32Exec.compare, hx, hinf]
  | false =>
      -- For finite, non-NaN values, `toDyadic?` is definitional and comparison reduces to
      -- `cmpDyadic d d = .eq`.
      have hdy : ∃ d, IEEE32Exec.toDyadic? x = some d := by
        unfold IEEE32Exec.toDyadic?
        -- Reduce to the finite branch and pick the corresponding dyadic witness.
        simp [hx, hinf]
        by_cases he : IEEE32Exec.expField x = 0
        · by_cases hf : IEEE32Exec.fracField x = 0
          · refine ⟨{ sign := IEEE32Exec.signBit x, mant := 0, exp := 0 }, ?_⟩
            simp [he, hf]
          · refine ⟨{ sign := IEEE32Exec.signBit x, mant := (IEEE32Exec.fracField x).toNat, exp :=
            -149 }, ?_⟩
            simp [he, hf]
        · refine
            ⟨{ sign := IEEE32Exec.signBit x
              , mant := IEEE32Exec.pow2 23 + (IEEE32Exec.fracField x).toNat
              , exp := (Int.ofNat (IEEE32Exec.expField x).toNat) - 150 }, ?_⟩
          simp [he]
      rcases hdy with ⟨d, hd⟩
      simp [IEEE32Exec.compare, hx, hinf, hd, IEEE32Exec.cmpDyadic]

theorem le_self_of_isNaN_false (x : F) (hx : IEEE32Exec.isNaN x = false) : x ≤ x := by
  have hcmp : IEEE32Exec.compare x x = some .eq := compare_self_of_isNaN_false (x := x) hx
  change IEEE32Exec.le x x
  simp [IEEE32Exec.le, hcmp]

end ExecLemmas

/-! ## Interval domain `I` (Eq. 6) -/

inductive I where
  | top : I
  | range : F → F → I
  deriving Repr

namespace I

/-- Concretization `γ` for abstract intervals (Eq. 7). -/
def γI : I → Set F
  | top => Set.univ
  | range a b => fun x => a ≤ x ∧ x ≤ b

/-- Membership in an abstract interval, via the concretization `γI`. -/
instance : Membership F I where
  mem J x := x ∈ γI J

/-- Abstract boxes `B ∈ I^d`. -/
abbrev Box (d : Nat) : Type := Fin d → I

/-- Concretization `γ` for boxes (Eq. 7). -/
def γ {d : Nat} (B : Box d) : Set (Fin d → F) := fun x => ∀ i, x i ∈ B i

/-- A box is in `[-1,1]^d` (paper: “abstract boxes in `[-1,1]^d`”). -/
def InCube {d : Nat} (B : Box d) : Prop :=
  ∀ i, (Numbers.neg_one : F) ∈ B i ∧ (Numbers.one : F) ∈ B i

@[simp] theorem mem_top (x : F) : x ∈ (top : I) := by
  -- `γ(top) = univ`.
  trivial

@[simp] theorem mem_range_iff (x a b : F) : x ∈ (range a b : I) ↔ a ≤ x ∧ x ≤ b := by
  rfl

/-- Point interval `⟨x,x⟩`. -/
@[inline] def point (x : F) : I := range x x

theorem mem_point_of_isNaN_false (x : F) (hx : IEEE32Exec.isNaN x = false) : x ∈ point x := by
  dsimp [point]
  -- `x ∈ ⟨x,x⟩` reduces to `x ≤ x ∧ x ≤ x`.
  simp [mem_range_iff, ExecLemmas.le_self_of_isNaN_false (x := x) hx]

/-- Point box `⟨x,x⟩^d`. -/
@[inline] def pointBox {d : Nat} (x : Fin d → F) : Box d := fun i => point (x i)

theorem mem_pointBox_of_isNaN_false {d : Nat} (x : Fin d → F) (hx : ∀ i, IEEE32Exec.isNaN (x i) =
  false) :
    x ∈ γ (pointBox (d := d) x) := by
  intro i
  exact mem_point_of_isNaN_false (x := x i) (hx i)

end I

/-! ## Executable interval operators for `+`, `*`, and ReLU -/

namespace OpsExact

open I

/-- Minimum of two `IEEE32Exec` values (NaN-aware, via `IEEE32Exec.minimum`). -/
@[inline] def min2 (x y : F) : F := IEEE32Exec.minimum x y

/-- Maximum of two `IEEE32Exec` values (NaN-aware, via `IEEE32Exec.maximum`). -/
@[inline] def max2 (x y : F) : F := IEEE32Exec.maximum x y

/-- Minimum of four `IEEE32Exec` values, computed via nested `min2`. -/
@[inline] def min4 (a b c d : F) : F := min2 (min2 a b) (min2 c d)
/-- Maximum of four `IEEE32Exec` values, computed via nested `max2`. -/
@[inline] def max4 (a b c d : F) : F := max2 (max2 a b) (max2 c d)

/-- Return `true` iff any of the four arguments is `NaN`. -/
@[inline] def hasNaN4 (a b c d : F) : Bool :=
  IEEE32Exec.isNaN a || IEEE32Exec.isNaN b || IEEE32Exec.isNaN c || IEEE32Exec.isNaN d

/-- Corner-based interval addition for `IEEE32Exec.add`. -/
def addSharpCorners : I → I → I
  | I.top, _ => I.top
  | _, I.top => I.top
  | I.range a b, I.range c d =>
      let p00 := IEEE32Exec.add a c
      let p01 := IEEE32Exec.add a d
      let p10 := IEEE32Exec.add b c
      let p11 := IEEE32Exec.add b d
      if hasNaN4 p00 p01 p10 p11 then
        I.top
      else
        I.range (min4 p00 p01 p10 p11) (max4 p00 p01 p10 p11)

/-- Corner-based interval multiplication for `IEEE32Exec.mul`. -/
def mulSharpCorners : I → I → I
  | I.top, _ => I.top
  | _, I.top => I.top
  | I.range a b, I.range c d =>
      let p00 := IEEE32Exec.mul a c
      let p01 := IEEE32Exec.mul a d
      let p10 := IEEE32Exec.mul b c
      let p11 := IEEE32Exec.mul b d
      if hasNaN4 p00 p01 p10 p11 then
        I.top
  else
        I.range (min4 p00 p01 p10 p11) (max4 p00 p01 p10 p11)

/-- Executable ReLU for `IEEE32Exec`, defined via `IEEE32Exec.maximum`. -/
@[inline] def relu (x : F) : F := IEEE32Exec.maximum x (Numbers.zero : F)

/--
Exact `ReLU♯` for intervals, using monotonicity of ReLU:
for `⟨a,b⟩`, `ReLU([a,b]) = [ReLU(a), ReLU(b)]`.
-/
def reluSharpEndpoints : I → I
  | I.top => I.top
  | I.range a b =>
      let ra := relu a
      let rb := relu b
      if IEEE32Exec.isNaN ra || IEEE32Exec.isNaN rb then I.top else I.range ra rb

/-! ### Eq. (8): exact interval hull on finite sets -/

/-- Totalized extended-real interpretation (defaults to `0` only on NaN). -/
noncomputable def toERealTotal (x : F) : EReal :=
  if IEEE32Exec.isNaN x then
    (0 : EReal)
  else if IEEE32Exec.isInf x then
    (if IEEE32Exec.signBit x then (⊥ : EReal) else (⊤ : EReal))
  else
    (IEEE32Exec.toReal x : EReal)

theorem le_iff_toERealTotal_le_of_isNaN_false (x y : F)
    (hx : IEEE32Exec.isNaN x = false) (hy : IEEE32Exec.isNaN y = false) :
    x ≤ y ↔ toERealTotal x ≤ toERealTotal y := by
  classical
  -- Split on infinities; for the finite branch, use `BridgeFP32Total` compare↔`toReal` lemmas.
  cases hxInf : IEEE32Exec.isInf x with
  | true =>
      cases hyInf : IEEE32Exec.isInf y with
      | true =>
          cases hsx : IEEE32Exec.signBit x <;> cases hsy : IEEE32Exec.signBit y <;>
            (change IEEE32Exec.le x y ↔ _; simp [IEEE32Exec.le, IEEE32Exec.compare, toERealTotal,
              hx, hy, hxInf, hyInf, hsx, hsy])
      | false =>
          cases hsx : IEEE32Exec.signBit x <;>
            (change IEEE32Exec.le x y ↔ _; simp [IEEE32Exec.le, IEEE32Exec.compare, toERealTotal,
              hx, hy, hxInf, hyInf, hsx])
  | false =>
      cases hyInf : IEEE32Exec.isInf y with
      | true =>
          cases hsy : IEEE32Exec.signBit y <;>
            (change IEEE32Exec.le x y ↔ _; simp [IEEE32Exec.le, IEEE32Exec.compare, toERealTotal,
              hx, hy, hxInf, hyInf, hsy])
      | false =>
          have hxFin : IEEE32Exec.isFinite x = true :=
            IEEE32Exec.isFinite_eq_true_of_isNaN_eq_false_of_isInf_eq_false (x := x) hx hxInf
          have hyFin : IEEE32Exec.isFinite y = true :=
            IEEE32Exec.isFinite_eq_true_of_isNaN_eq_false_of_isInf_eq_false (x := y) hy hyInf
          have hlt : IEEE32Exec.compare x y = some .lt ↔ IEEE32Exec.toReal x < IEEE32Exec.toReal y
            :=
            IEEE32Exec.compare_eq_some_lt_iff_toReal_lt_of_isFinite (x := x) (y := y) hxFin hyFin
          have heq : IEEE32Exec.compare x y = some .eq ↔ IEEE32Exec.toReal x = IEEE32Exec.toReal y
            :=
            IEEE32Exec.compare_eq_some_eq_iff_toReal_eq_of_isFinite (x := x) (y := y) hxFin hyFin
          have hgt : IEEE32Exec.compare x y = some .gt ↔ IEEE32Exec.toReal y < IEEE32Exec.toReal x
            :=
            IEEE32Exec.compare_eq_some_gt_iff_toReal_gt_of_isFinite (x := x) (y := y) hxFin hyFin
          have hcmp_ne_none : IEEE32Exec.compare x y ≠ none := by
            intro hcmp
            unfold IEEE32Exec.compare at hcmp
            simp [hx, hy, hxInf, hyInf] at hcmp
            have hxDy : IEEE32Exec.toDyadic? x ≠ none := by
              intro hxDy
              have : IEEE32Exec.isFinite x = false :=
                IEEE32Exec.isFinite_eq_false_of_toDyadic?_eq_none (x := x) hxDy
              have h := this
              rw [hxFin] at h
              cases h
            have hyDy : IEEE32Exec.toDyadic? y ≠ none := by
              intro hyDy
              have : IEEE32Exec.isFinite y = false :=
                IEEE32Exec.isFinite_eq_false_of_toDyadic?_eq_none (x := y) hyDy
              have h := this
              rw [hyFin] at h
              cases h
            cases hxdy : IEEE32Exec.toDyadic? x with
            | none =>
                exact (hxDy hxdy).elim
            | some dx =>
                cases hydy : IEEE32Exec.toDyadic? y with
                | none =>
                    exact (hyDy hydy).elim
                | some dy =>
                    simp [hxdy, hydy] at hcmp
          constructor
          · intro hxy
            change IEEE32Exec.le x y at hxy
            cases hcmp : IEEE32Exec.compare x y with
            | none =>
                exact False.elim (hcmp_ne_none hcmp)
            | some o =>
                cases o with
                | lt =>
                    have hto : IEEE32Exec.toReal x < IEEE32Exec.toReal y := (hlt).1 (by simp [hcmp])
                    have hle : (IEEE32Exec.toReal x : EReal) ≤ (IEEE32Exec.toReal y : EReal) := by
                      simpa [EReal.coe_le_coe_iff] using le_of_lt hto
                    simpa [toERealTotal, hx, hy, hxInf, hyInf] using hle
                | eq =>
                    have hto : IEEE32Exec.toReal x = IEEE32Exec.toReal y := (heq).1 (by simp [hcmp])
                    have hle : (IEEE32Exec.toReal x : EReal) ≤ (IEEE32Exec.toReal y : EReal) := by
                      simpa [EReal.coe_le_coe_iff] using le_of_eq hto
                    simpa [toERealTotal, hx, hy, hxInf, hyInf] using hle
                | gt =>
                    have : False := by
                      simp [IEEE32Exec.le, hcmp] at hxy
                    exact this.elim
          · intro hxy
            change IEEE32Exec.le x y
            cases hcmp : IEEE32Exec.compare x y with
            | none =>
                exact False.elim (hcmp_ne_none hcmp)
            | some o =>
                cases o with
                | lt => simp [IEEE32Exec.le, hcmp]
                | eq => simp [IEEE32Exec.le, hcmp]
                | gt =>
                    have hto : IEEE32Exec.toReal y < IEEE32Exec.toReal x := (hgt).1 (by simp [hcmp])
                    have hnot : ¬ (IEEE32Exec.toReal x : EReal) ≤ (IEEE32Exec.toReal y : EReal) :=
                      by
                      simpa [EReal.coe_le_coe_iff] using (not_le_of_gt hto)
                    have hxy' : (IEEE32Exec.toReal x : EReal) ≤ (IEEE32Exec.toReal y : EReal) := by
                      simpa [toERealTotal, hx, hy, hxInf, hyInf] using hxy
                    exact (hnot hxy').elim

theorem exists_chooseMin (s : Finset F) (hs : s.Nonempty) :
    ∃ x, x ∈ s ∧ toERealTotal x = (s.image toERealTotal).min' (hs.image _) := by
  classical
  -- `min'` is an element of the image, hence has a preimage in `s`.
  have hmem : (s.image toERealTotal).min' (hs.image _) ∈ (s.image toERealTotal) :=
    Finset.min'_mem _ (hs.image _)
  rcases Finset.mem_image.mp hmem with ⟨x, hx, hxEq⟩
  exact ⟨x, hx, hxEq⟩

theorem exists_chooseMax (s : Finset F) (hs : s.Nonempty) :
    ∃ x, x ∈ s ∧ toERealTotal x = (s.image toERealTotal).max' (hs.image _) := by
  classical
  have hmem : (s.image toERealTotal).max' (hs.image _) ∈ (s.image toERealTotal) :=
    Finset.max'_mem _ (hs.image _)
  rcases Finset.mem_image.mp hmem with ⟨x, hx, hxEq⟩
  exact ⟨x, hx, hxEq⟩

noncomputable def chooseMin (s : Finset F) (hs : s.Nonempty) : F :=
  Classical.choose (exists_chooseMin s hs)

noncomputable def chooseMax (s : Finset F) (hs : s.Nonempty) : F :=
  Classical.choose (exists_chooseMax s hs)

theorem chooseMin_spec (s : Finset F) (hs : s.Nonempty) :
    chooseMin s hs ∈ s ∧ toERealTotal (chooseMin s hs) = (s.image toERealTotal).min' (hs.image _) :=
      by
  simpa [chooseMin] using (Classical.choose_spec (exists_chooseMin s hs))

theorem chooseMax_spec (s : Finset F) (hs : s.Nonempty) :
    chooseMax s hs ∈ s ∧ toERealTotal (chooseMax s hs) = (s.image toERealTotal).max' (hs.image _) :=
      by
  simpa [chooseMax] using (Classical.choose_spec (exists_chooseMax s hs))

/--
Interval hull for a finite set of floats:
- `⊤` if the set contains a NaN (paper: `⊥ ∈ S`), otherwise
- the interval `⟨min S, max S⟩`.
-/
noncomputable def hull (s : Finset F) : I :=
  if _hnan : ∃ x ∈ s, IEEE32Exec.isNaN x = true then
    I.top
  else if hs : s.Nonempty then
    I.range (chooseMin s hs) (chooseMax s hs)
  else
    I.top

theorem mem_hull_of_mem (s : Finset F) {x : F} (hx : x ∈ s) : x ∈ hull s := by
  classical
  unfold hull
  by_cases hnan : ∃ z ∈ s, IEEE32Exec.isNaN z = true
  · simp [hnan, I.mem_top]
  · have hs : s.Nonempty := ⟨x, hx⟩
    have hn : ∀ z, z ∈ s → IEEE32Exec.isNaN z = false := by
      intro z hz
      by_contra hz'
      have : IEEE32Exec.isNaN z = true := by simpa using hz'
      exact hnan ⟨z, hz, this⟩
    have hxNaN : IEEE32Exec.isNaN x = false := hn x hx
    have hminNaN : IEEE32Exec.isNaN (chooseMin s hs) = false := hn _ (chooseMin_spec s hs).1
    have hmaxNaN : IEEE32Exec.isNaN (chooseMax s hs) = false := hn _ (chooseMax_spec s hs).1
    -- bounds in `EReal` from min'/max' on the image
    have hminE :
        toERealTotal (chooseMin s hs) ≤ toERealTotal x := by
      -- rewrite via the `chooseMin` equality and `min'_le`.
      have hxImg : toERealTotal x ∈ s.image toERealTotal := Finset.mem_image_of_mem _ hx
      simpa [(chooseMin_spec s hs).2] using Finset.min'_le (s.image toERealTotal) (toERealTotal x)
        hxImg
    have hmaxE :
        toERealTotal x ≤ toERealTotal (chooseMax s hs) := by
      have hxImg : toERealTotal x ∈ s.image toERealTotal := Finset.mem_image_of_mem _ hx
      -- `le_max'` is the dual lemma for maxima.
      simpa [(chooseMax_spec s hs).2] using Finset.le_max' (s.image toERealTotal) (toERealTotal x)
        hxImg
    have hmin : chooseMin s hs ≤ x :=
      (le_iff_toERealTotal_le_of_isNaN_false (x := chooseMin s hs) (y := x) hminNaN hxNaN).2 hminE
    have hmax : x ≤ chooseMax s hs :=
      (le_iff_toERealTotal_le_of_isNaN_false (x := x) (y := chooseMax s hs) hxNaN hmaxNaN).2 hmaxE
    simp [hnan, hs, I.mem_range_iff, hmin, hmax]

/-! ### Interval ops `⊕♯/⊗♯/σ♯` instantiated from `hull` -/

noncomputable def γFinsetI : I → Finset F
  | I.top => Finset.univ
  | I.range a b => by
      classical
      exact Finset.univ.filter (fun x => x ∈ (I.range a b : I))

noncomputable def γFinsetBox {d : Nat} (B : I.Box d) : Finset (Fin d → F) := by
  classical
  exact Finset.univ.filter (fun x => ∀ i, x i ∈ B i)

theorem mem_γFinsetBox_iff {d : Nat} (B : I.Box d) (x : Fin d → F) :
    x ∈ γFinsetBox B ↔ ∀ i, x i ∈ B i := by
  classical
  simp [γFinsetBox]

/-- Build a 2D box from two intervals (coordinate `0` is `A`, coordinate `1` is `B`). -/
@[inline] def box2 (A B : I) : I.Box 2 :=
  fun
    | 0 => A
    | 1 => B

/-- Pair two scalars into a `Fin 2 → F` vector, indexed by `0` and `1`. -/
@[inline] private def pair (x y : F) : Fin 2 → F :=
  fun
    | 0 => x
    | 1 => y

noncomputable def addSharp : I → I → I
  | I.top, _ => I.top
  | _, I.top => I.top
  | A, B =>
      hull <| (γFinsetBox (box2 A B)).image (fun p => IEEE32Exec.add (p 0) (p 1))

noncomputable def mulSharp : I → I → I
  | I.top, _ => I.top
  | _, I.top => I.top
  | A, B =>
      hull <| (γFinsetBox (box2 A B)).image (fun p => IEEE32Exec.mul (p 0) (p 1))

noncomputable def reluSharp : I → I
  | I.top => I.top
  | A =>
      hull <| (γFinsetI A).image relu

/-- Interval summation `◦∑♯` from the paper: fold with `addSharp`. -/
def sumSharp {n : Nat} (ts : Fin n → I) : I :=
  (List.finRange n).foldl (fun acc i => addSharp acc (ts i)) (I.range (Numbers.zero : F)
    (Numbers.zero : F))

/-!
`OpsExact` implements the finite interval semantics used for exact interval-image statements.

Proving its soundness for `IEEE32Exec` (+/*/ReLU) is substantial work. This file isolates
that work behind an explicit interface, so higher-level semantic proofs can be completed
once op-level soundness lemmas are available.
-/

class Sound : Prop where
  add_sound :
    ∀ {A B : I} {x y : F}, x ∈ A → y ∈ B → IEEE32Exec.add x y ∈ addSharp A B
  mul_sound :
    ∀ {A B : I} {x y : F}, x ∈ A → y ∈ B → IEEE32Exec.mul x y ∈ mulSharp A B
  relu_sound :
    ∀ {A : I} {x : F}, x ∈ A → relu x ∈ reluSharp A

theorem add_sound (A B : I) :
    ∀ {x y : F}, x ∈ A → y ∈ B → IEEE32Exec.add x y ∈ addSharp A B := by
  intro x y hx hy
  cases A with
  | top =>
      simp [addSharp, I.mem_top]
  | range a b =>
      cases B with
      | top =>
          simp [addSharp, I.mem_top]
      | range c d =>
          have hp : pair x y ∈ γFinsetBox (box2 (I.range a b) (I.range c d)) := by
            refine (mem_γFinsetBox_iff (B := box2 (I.range a b) (I.range c d)) (x := pair x y)).2 ?_
            exact (Fin.forall_fin_two).2 ⟨by simpa [pair, box2] using hx, by simpa [pair, box2]
              using hy⟩
          have hmem :
              IEEE32Exec.add x y ∈
                (γFinsetBox (box2 (I.range a b) (I.range c d))).image (fun p => IEEE32Exec.add (p 0)
                  (p 1)) := by
            refine Finset.mem_image.mpr ?_
            refine ⟨pair x y, hp, ?_⟩
            simp [pair]
          simpa [addSharp] using (mem_hull_of_mem _ hmem)

theorem mul_sound (A B : I) :
    ∀ {x y : F}, x ∈ A → y ∈ B → IEEE32Exec.mul x y ∈ mulSharp A B := by
  intro x y hx hy
  cases A with
  | top =>
      simp [mulSharp, I.mem_top]
  | range a b =>
      cases B with
      | top =>
          simp [mulSharp, I.mem_top]
      | range c d =>
          have hp : pair x y ∈ γFinsetBox (box2 (I.range a b) (I.range c d)) := by
            refine (mem_γFinsetBox_iff (B := box2 (I.range a b) (I.range c d)) (x := pair x y)).2 ?_
            exact (Fin.forall_fin_two).2 ⟨by simpa [pair, box2] using hx, by simpa [pair, box2]
              using hy⟩
          have hmem :
              IEEE32Exec.mul x y ∈
                (γFinsetBox (box2 (I.range a b) (I.range c d))).image (fun p => IEEE32Exec.mul (p 0)
                  (p 1)) := by
            refine Finset.mem_image.mpr ?_
            refine ⟨pair x y, hp, ?_⟩
            simp [pair]
          simpa [mulSharp] using (mem_hull_of_mem _ hmem)

theorem relu_sound (A : I) :
    ∀ {x : F}, x ∈ A → relu x ∈ reluSharp A := by
  intro x hx
  cases A with
  | top =>
      simp [reluSharp, I.mem_top]
  | range a b =>
      have hx' : x ∈ γFinsetI (I.range a b) := by
        simpa [γFinsetI, hx]
      have hmem : relu x ∈ (γFinsetI (I.range a b)).image relu :=
        Finset.mem_image_of_mem _ hx'
      simpa [reluSharp] using (mem_hull_of_mem _ hmem)

noncomputable instance : Sound :=
  ⟨by
      intro A B x y hx hy
      exact add_sound (A := A) (B := B) hx hy
    , by
      intro A B x y hx hy
      exact mul_sound (A := A) (B := B) hx hy
    , by
      intro A x hx
      exact relu_sound (A := A) (x := x) hx⟩

theorem sumSharp_sound [Sound] {n : Nat} (ts : Fin n → I) (t : Fin n → F)
    (ht : ∀ i, t i ∈ ts i) :
    (List.finRange n).foldl (fun acc i => IEEE32Exec.add acc (t i)) (Numbers.zero : F) ∈ sumSharp ts
      := by
  -- Prove the stronger list-induction form, then instantiate with `List.finRange n`.
  have hz : IEEE32Exec.isNaN (Numbers.zero : F) = false := by
    -- For `IEEE32Exec`, `Numbers.zero` is definitionally `posZero = ofBits 0`.
    change IEEE32Exec.isNaN (IEEE32Exec.posZero : F) = false
    simp [IEEE32Exec.isNaN, IEEE32Exec.posZero, IEEE32Exec.ofBits, IEEE32Exec.expField,
      IEEE32Exec.fracField, IEEE32Exec.expAllOnes]
  have h0 : (Numbers.zero : F) ∈ (I.range (Numbers.zero : F) (Numbers.zero : F)) := by
    -- Avoid rewriting via `point` to keep simp from collapsing `∧` goals.
    exact And.intro (ExecLemmas.le_self_of_isNaN_false (x := (Numbers.zero : F)) hz)
      (ExecLemmas.le_self_of_isNaN_false (x := (Numbers.zero : F)) hz)
  have hList :
      ∀ (l : List (Fin n)) (accI : I) (accV : F),
        accV ∈ accI →
          (l.foldl (fun acc i => IEEE32Exec.add acc (t i)) accV) ∈
            (l.foldl (fun acc i => addSharp acc (ts i)) accI) := by
    intro l
    induction l with
    | nil =>
        intro accI accV hacc
        simpa using hacc
    | cons i l ih =>
        intro accI accV hacc
        have hi : t i ∈ ts i := ht i
        have hstep : IEEE32Exec.add accV (t i) ∈ addSharp accI (ts i) :=
          Sound.add_sound (A := accI) (B := ts i) (x := accV) (y := t i) hacc hi
        simpa using
          (ih (accI := addSharp accI (ts i)) (accV := IEEE32Exec.add accV (t i)) hstep)
  -- Finish by unfolding `sumSharp` and using the list induction lemma.
  simpa [sumSharp] using hList (List.finRange n) (I.range (Numbers.zero : F) (Numbers.zero : F))
    (Numbers.zero : F) h0

end OpsExact

/-! ## Exact interval-image property for rounded targets -/

namespace ExactImage

open I

/-- Float interval set `{x | a ≤ x ∧ x ≤ b}` (avoids needing `Preorder`). -/
def Icc (a b : F) : Set F := fun x => a ≤ x ∧ x ≤ b

/-- `m` is a minimum of `g` on the set `S`, stated without choosing a canonical `min`. -/
def IsMinOn {X : Type} (g : X → F) (S : Set X) (m : F) : Prop :=
  (∃ x, x ∈ S ∧ g x = m) ∧ ∀ y, (∃ x, x ∈ S ∧ g x = y) → m ≤ y

/-- `M` is a maximum of `g` on the set `S`, stated without choosing a canonical `max`. -/
def IsMaxOn {X : Type} (g : X → F) (S : Set X) (M : F) : Prop :=
  (∃ x, x ∈ S ∧ g x = M) ∧ ∀ y, (∃ x, x ∈ S ∧ g x = y) → y ≤ M

/--
For each box `B`, the abstract output interval is exactly the interval hull of the rounded target's
direct image on `γ(B)`, expressed via existential min/max witnesses.
-/
def ExactIntervalImage {d : Nat} (g : (Fin d → F) → F) (_ν : (Fin d → F) → F)
    (nuInt : I.Box d → I) : Prop :=
  ∀ B, (I.γ (d := d) B).Nonempty →
    ∃ m M,
      IsMinOn g (I.γ (d := d) B) m ∧
      IsMaxOn g (I.γ (d := d) B) M ∧
      I.γI (nuInt B) = Icc m M

theorem exactIntervalImage_constant {d : Nat} (c : F) (hc : isNaN c = false) :
    ExactIntervalImage (d := d) (g := fun _ => c) (_ν := fun _ => c)
      (nuInt := fun _ => I.range c c) := by
  intro B hne
  refine ⟨c, c, ?_, ?_, ?_⟩
  · -- min witness
    rcases hne with ⟨x0, hx0⟩
    refine And.intro ?_ ?_
    · exact ⟨x0, hx0, rfl⟩
    · intro y hy
      rcases hy with ⟨x, hx, hgy⟩
      subst hgy
      simpa using ExecLemmas.le_self_of_isNaN_false (x := c) hc
  · -- max witness
    rcases hne with ⟨x0, hx0⟩
    refine And.intro ?_ ?_
    · exact ⟨x0, hx0, rfl⟩
    · intro y hy
      rcases hy with ⟨x, hx, hgy⟩
      subst hgy
      simpa using ExecLemmas.le_self_of_isNaN_false (x := c) hc
  · -- concretization equality
    ext x
    dsimp [I.γI, Icc]
    rfl

end ExactImage

/-! ## Two-layer interval evaluator using `OpsExact` -/

namespace MLP2Exact

open I OpsExact

/-- Parameters of a 2-layer MLP of shape `d → h → 1` for the exact interval semantics (`OpsExact`).
  -/
structure Net (d h : Nat) where
  /-- Weight matrix for layer 1. -/
  W1 : Fin h → Fin d → F
  /-- Bias for layer 1. -/
  b1 : Fin h → F
  /-- Weight matrix for layer 2. -/
  W2 : Fin 1 → Fin h → F
  /-- Bias for layer 2. -/
  b2 : Fin 1 → F

/-- Apply an affine layer to an input vector using IEEE32Exec arithmetic. -/
def aff {d m : Nat} (W : Fin m → Fin d → F) (b : Fin m → F) (x : Fin d → F) : Fin m → F :=
  fun i =>
    let s :=
      (List.finRange d).foldl
        (fun acc j => IEEE32Exec.add acc (IEEE32Exec.mul (W i j) (x j)))
        (Numbers.zero : F)
    IEEE32Exec.add s (b i)

/-- Evaluate a 2-layer ReLU MLP on a concrete input, using the exact op wrappers (`OpsExact.relu`).
  -/
def eval {d h : Nat} (net : Net d h) (x : Fin d → F) : F :=
  let z1 : Fin h → F := aff net.W1 net.b1 x
  let a1 : Fin h → F := fun i => OpsExact.relu (z1 i)
  let z2 : Fin 1 → F := aff net.W2 net.b2 a1
  z2 0

/-- Interval affine transform `aff♯` using corner multiplication and interval summation. -/
def affSharp {d m : Nat} (W : Fin m → Fin d → F) (b : Fin m → F) (B : I.Box d) : I.Box m :=
  fun i =>
    let terms : Fin d → I := fun j => OpsExact.mulSharp (I.range (W i j) (W i j)) (B j)
    let s := OpsExact.sumSharp terms
    OpsExact.addSharp s (I.range (b i) (b i))

/-- Interval semantics `ν♯` for 2-layer ReLU MLPs. -/
def evalSharp {d h : Nat} (net : Net d h) (B : I.Box d) : I :=
  let z1 : I.Box h := affSharp net.W1 net.b1 B
  let a1 : I.Box h := fun i => OpsExact.reluSharp (z1 i)
  let z2 : I.Box 1 := affSharp net.W2 net.b2 a1
  z2 0

theorem aff_sound [OpsExact.Sound] {d m : Nat}
    (W : Fin m → Fin d → F) (b : Fin m → F) (B : I.Box d)
    (hW : ∀ i j, IEEE32Exec.isNaN (W i j) = false)
    (hb : ∀ i, IEEE32Exec.isNaN (b i) = false) :
    ∀ {x : Fin d → F}, x ∈ I.γ B → aff W b x ∈ I.γ (affSharp W b B) := by
  intro x hx i
  -- Soundness of each multiplicative term.
  have hterms : ∀ j, IEEE32Exec.mul (W i j) (x j) ∈ OpsExact.mulSharp (I.range (W i j) (W i j)) (B
    j) := by
    intro j
    have hWij : (W i j) ∈ I.point (W i j) := I.mem_point_of_isNaN_false (x := W i j) (hW i j)
    have hxj : x j ∈ B j := hx j
    -- Coerce `hWij` from `point` to `range`.
    have hWij' : (W i j) ∈ (I.range (W i j) (W i j)) := by simpa [I.point] using hWij
    exact OpsExact.Sound.mul_sound (A := I.range (W i j) (W i j)) (B := B j) hWij' hxj
  -- Sum soundness via `◦∑♯`.
  have hsum :
      (List.finRange d).foldl (fun acc j => IEEE32Exec.add acc (IEEE32Exec.mul (W i j) (x j)))
        (Numbers.zero : F)
        ∈ OpsExact.sumSharp (fun j => OpsExact.mulSharp (I.range (W i j) (W i j)) (B j)) := by
    -- Apply `sumSharp_sound` with the instantiated term intervals and values.
    simpa using
      (OpsExact.sumSharp_sound (ts := fun j => OpsExact.mulSharp (I.range (W i j) (W i j)) (B j))
        (t := fun j => IEEE32Exec.mul (W i j) (x j)) hterms)
  -- Add the bias (a point interval).
  have hbi : (b i) ∈ (I.range (b i) (b i)) := by
    simpa [I.point] using (I.mem_point_of_isNaN_false (x := b i) (hb i))
  have hfinal :
      IEEE32Exec.add
          ((List.finRange d).foldl (fun acc j => IEEE32Exec.add acc (IEEE32Exec.mul (W i j) (x j)))
            (Numbers.zero : F))
          (b i)
        ∈ OpsExact.addSharp
            (OpsExact.sumSharp (fun j => OpsExact.mulSharp (I.range (W i j) (W i j)) (B j)))
            (I.range (b i) (b i)) :=
    OpsExact.Sound.add_sound (A := OpsExact.sumSharp (fun j => OpsExact.mulSharp (I.range (W i j) (W
      i j)) (B j)))
      (B := I.range (b i) (b i)) hsum hbi
  simpa [aff, affSharp] using hfinal

theorem eval_sound [OpsExact.Sound] {d h : Nat} (net : Net d h) (B : I.Box d)
    (hW1 : ∀ i j, IEEE32Exec.isNaN (net.W1 i j) = false)
    (hb1 : ∀ i, IEEE32Exec.isNaN (net.b1 i) = false)
    (hW2 : ∀ i j, IEEE32Exec.isNaN (net.W2 i j) = false)
    (hb2 : ∀ i, IEEE32Exec.isNaN (net.b2 i) = false) :
    ∀ {x : Fin d → F}, x ∈ I.γ B → eval net x ∈ evalSharp net B := by
  intro x hx
  -- First affine layer.
  have hz1 : aff net.W1 net.b1 x ∈ I.γ (affSharp net.W1 net.b1 B) :=
    aff_sound (W := net.W1) (b := net.b1) (B := B) hW1 hb1 hx
  -- ReLU layer.
  have ha1 : (fun i => OpsExact.relu ((aff net.W1 net.b1 x) i)) ∈
      I.γ (fun i => OpsExact.reluSharp ((affSharp net.W1 net.b1 B) i)) := by
    intro i
    exact OpsExact.Sound.relu_sound (A := (affSharp net.W1 net.b1 B) i) (x := (aff net.W1 net.b1 x)
      i) (hz1 i)
  -- Second affine layer (input box is the ReLU-abstracted hidden activations).
  have hz2 : aff net.W2 net.b2 (fun i => OpsExact.relu ((aff net.W1 net.b1 x) i)) ∈
      I.γ (affSharp net.W2 net.b2 (fun i => OpsExact.reluSharp ((affSharp net.W1 net.b1 B) i))) :=
    aff_sound (W := net.W2) (b := net.b2)
      (B := fun i => OpsExact.reluSharp ((affSharp net.W1 net.b1 B) i)) hW2 hb2 ha1
  -- Output is the single coordinate `0`.
  simpa [eval, evalSharp] using (hz2 0)

theorem interval_semantics_sound [OpsExact.Sound] {d h : Nat} (net : Net d h) (B : I.Box d)
    (hW1 : ∀ i j, IEEE32Exec.isNaN (net.W1 i j) = false)
    (hb1 : ∀ i, IEEE32Exec.isNaN (net.b1 i) = false)
    (hW2 : ∀ i j, IEEE32Exec.isNaN (net.W2 i j) = false)
    (hb2 : ∀ i, IEEE32Exec.isNaN (net.b2 i) = false) :
    Set.image (eval net) (I.γ B) ⊆ I.γI (evalSharp net B) := by
  intro y hy
  rcases hy with ⟨x, hx, rfl⟩
  exact eval_sound (net := net) (B := B) hW1 hb1 hW2 hb2 hx

theorem eval_sound_pointBox [OpsExact.Sound] {d h : Nat} (net : Net d h) (x : Fin d → F)
    (hx : ∀ i, IEEE32Exec.isNaN (x i) = false)
    (hW1 : ∀ i j, IEEE32Exec.isNaN (net.W1 i j) = false)
    (hb1 : ∀ i, IEEE32Exec.isNaN (net.b1 i) = false)
    (hW2 : ∀ i j, IEEE32Exec.isNaN (net.W2 i j) = false)
    (hb2 : ∀ i, IEEE32Exec.isNaN (net.b2 i) = false) :
    eval net x ∈ evalSharp net (I.pointBox (d := d) x) := by
  have hxBox : x ∈ I.γ (I.pointBox (d := d) x) := I.mem_pointBox_of_isNaN_false (x := x) hx
  exact eval_sound (net := net) (B := I.pointBox (d := d) x) hW1 hb1 hW2 hb2 hxBox

end MLP2Exact

end

end FloatIntervalApprox

end NN.MLTheory.Proofs.UniversalApproximation
