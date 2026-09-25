/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import FloatLib.Floats.Formats.BinaryInterchange.Configured
public import FloatLib.Floats.Formats.IEEE754.Native

/-!
# Floating-Point Interval Semantics

Interval-domain semantics for `ExecFloat.Binary 8 23` neural networks.

This file formalizes the interval domain, concretization map, executable interval operators, and
the exact-interval-image property used by the floating-point interval-approximation theorem of
Hwang, Lee, Park, Park, and Saad, *Floating-Point Neural Networks Are Provably Robust Universal
Approximators* (`arXiv:2506.16065`).
-/

@[expose] public section

open FloatLib.Floats (ExecFloat)
open FloatLib.Floats.ExecFloat.Binary (isInfinite isNaN signBit toModel)
open FloatLib.Floats.Formats.BinaryInterchange (Model FloatFormat)


namespace NN.MLTheory.Proofs.UniversalApproximation

open FloatLib.Floats.Formats.BinaryInterchange

namespace FloatIntervalApprox

noncomputable section

/-! ## Basic aliases -/

/-- Shorthand for the executable binary32 float type used in this development. -/
abbrev F : Type := (ExecFloat.Binary 8 23)

/-!
`ExecFloat.Binary 8 23` is stored as a `UInt32` bit-pattern, so the carrier is finite. We use this
only to
obtain `Finset.univ` for paper-style “finite hull” definitions; nothing is computed.
-/

namespace DecidableInstances

instance : DecidableRel (fun x y : F => x ≤ y) := by
  classical
  intro x y
  infer_instance

end DecidableInstances

namespace FintypeInstances

/-- `UInt32` is finite, via the injection into `Fin (2 ^ 32)`. -/
noncomputable instance : Finite UInt32 := by
  classical
  refine Finite.of_injective (fun u : UInt32 => (⟨u.toNat, u.toNat_lt⟩ : Fin (2 ^ 32))) ?_
  intro a b hab
  have : a.toNat = b.toNat := by
    simpa using congrArg Fin.val hab
  exact (UInt32.toNat_inj).1 this

/-- Hence a `Fintype`, classically. -/
noncomputable instance : Fintype UInt32 := by
  classical
  exact Fintype.ofFinite UInt32

/-- Binary32 is finite, since it is determined by its bits. -/
noncomputable instance : Finite (ExecFloat.Binary 8 23) := by
  classical
  refine Finite.of_injective ExecFloat.Binary.toBits32 ?_
  intro a b hab
  have h := congrArg ExecFloat.Binary.ofBits32 hab
  simpa only [ExecFloat.Binary.ofBits32_toBits32] using h

/-- Hence a `Fintype`. This is what makes the interval hull below a `Finset` construction and the
abstract operators exact rather than approximate: the whole float type can be enumerated. -/
noncomputable instance : Fintype (ExecFloat.Binary 8 23) := by
  classical
  exact Fintype.ofFinite (ExecFloat.Binary 8 23)

end FintypeInstances

/-! ## Small helper lemmas about the `ExecFloat.Binary 8 23` order -/

namespace ExecLemmas

/-- Comparing a non-NaN float with itself yields `eq`, infinities included. -/
theorem compare_self_of_isNaN_false (x : F) (hx : isNaN x = false) :
    ExecFloat.compare x x = some .eq := by
  change Model.compare (toModel x) (toModel x) = some .eq
  exact Model.compare_self_of_isNaN_eq_false (toModel x) hx

/-- Reflexivity of `≤` away from NaN, which is as much as IEEE 754 order gives. -/
theorem le_self_of_isNaN_false (x : F) (hx : isNaN x = false) : x ≤ x := by
  apply FloatLib.Floats.ExecFloat.Binary.le_iff_le_toModel.mpr
  exact (Model.Interval.le_iff_toEReal_le_of_isNaN_eq_false
    (toModel x) (toModel x) hx hx).mpr le_rfl

end ExecLemmas

/-! ## Interval domain `I` (Eq. 6) -/

/-- The interval abstract domain of Eq. 6: either the top element or a pair of float endpoints. -/
inductive I where
  /-- No information: every float is possible. -/
  | top : I
  /-- The floats between the two endpoints, inclusive. -/
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
  ∀ i, ((-1) : F) ∈ B i ∧ (1 : F) ∈ B i

/-- Everything is in `⊤`, NaN included; that is what makes `⊤` the sound fallback. -/
@[simp] theorem mem_top (x : F) : x ∈ (top : I) := by
  -- `γ(top) = univ`.
  trivial

/-- Membership in a range interval unfolds to the two float comparisons. -/
@[simp] theorem mem_range_iff (x a b : F) : x ∈ (range a b : I) ↔ a ≤ x ∧ x ≤ b := by
  rfl

/-- Point interval `⟨x,x⟩`. -/
@[inline] def point (x : F) : I := range x x

theorem mem_point_of_isNaN_false (x : F) (hx : isNaN x = false) : x ∈ point x := by
  dsimp [point]
  -- `x ∈ ⟨x,x⟩` reduces to `x ≤ x ∧ x ≤ x`.
  simp [mem_range_iff, ExecLemmas.le_self_of_isNaN_false (x := x) hx]

/-- Point box `⟨x,x⟩^d`. -/
@[inline] def pointBox {d : Nat} (x : Fin d → F) : Box d := fun i => point (x i)

theorem mem_pointBox_of_isNaN_false {d : Nat} (x : Fin d → F) (hx : ∀ i, isNaN (x i) =
  false) :
    x ∈ γ (pointBox (d := d) x) := by
  intro i
  exact mem_point_of_isNaN_false (x := x i) (hx i)

end I

/-! ## Executable interval operators for `+`, `*`, and ReLU -/

namespace OpsExact

open I

/-- Minimum of two `ExecFloat.Binary 8 23` values (NaN-aware, via `min`). -/
@[inline] def min2 (x y : F) : F := min x y

/-- Maximum of two `ExecFloat.Binary 8 23` values (NaN-aware, via `max`). -/
@[inline] def max2 (x y : F) : F := max x y

/-- Minimum of four `ExecFloat.Binary 8 23` values, computed via nested `min2`. -/
@[inline] def minOfFour (a b c d : F) : F := min2 (min2 a b) (min2 c d)
/-- Maximum of four `ExecFloat.Binary 8 23` values, computed via nested `max2`. -/
@[inline] def maxOfFour (a b c d : F) : F := max2 (max2 a b) (max2 c d)

/-- Return `true` iff any of the four arguments is `NaN`. -/
@[inline] def hasNaNAmongFour (a b c d : F) : Bool :=
  isNaN a || isNaN b || isNaN c || isNaN d

/-- Corner-based interval addition for `ExecFloat.add`. -/
def addSharpCorners : I → I → I
  | I.top, _ => I.top
  | _, I.top => I.top
  | I.range a b, I.range c d =>
      let p00 := ExecFloat.add a c
      let p01 := ExecFloat.add a d
      let p10 := ExecFloat.add b c
      let p11 := ExecFloat.add b d
      if hasNaNAmongFour p00 p01 p10 p11 then
        I.top
      else
        I.range (minOfFour p00 p01 p10 p11) (maxOfFour p00 p01 p10 p11)

/-- Corner-based interval multiplication for `ExecFloat.mul`. -/
def mulSharpCorners : I → I → I
  | I.top, _ => I.top
  | _, I.top => I.top
  | I.range a b, I.range c d =>
      let p00 := ExecFloat.mul a c
      let p01 := ExecFloat.mul a d
      let p10 := ExecFloat.mul b c
      let p11 := ExecFloat.mul b d
      if hasNaNAmongFour p00 p01 p10 p11 then
        I.top
  else
        I.range (minOfFour p00 p01 p10 p11) (maxOfFour p00 p01 p10 p11)

/-- Executable ReLU for `ExecFloat.Binary 8 23`, defined via `max`. -/
@[inline] def relu (x : F) : F := max x (0 : F)

/--
Exact `ReLU♯` for intervals, using monotonicity of ReLU:
for `⟨a,b⟩`, `ReLU([a,b]) = [ReLU(a), ReLU(b)]`.
-/
def reluSharpEndpoints : I → I
  | I.top => I.top
  | I.range a b =>
      let ra := relu a
      let rb := relu b
      if isNaN ra || isNaN rb then I.top else I.range ra rb

/-! ### Eq. (8): exact interval hull on finite sets -/

/-- Totalized extended-real interpretation (defaults to `0` only on NaN). -/
noncomputable def toERealTotal (x : F) : EReal :=
  if isNaN x then
    (0 : EReal)
  else if isInfinite x then
    (if signBit x then (⊥ : EReal) else (⊤ : EReal))
  else
    ((toModel x).toReal : EReal)

private theorem toERealTotal_eq_model (x : F) :
    toERealTotal x = Model.toEReal (toModel x) := by
  exact (Model.toEReal_eq_ite (toModel x)).symm

/-- Away from NaN, float comparison agrees with the order on `EReal` under the total embedding.

This is the lemma that buys the whole development a linear order to take minima and maxima in:
`ExecFloat.Binary 8 23` itself has no `LinearOrder`, but its non-NaN part embeds into one. -/
theorem le_iff_toERealTotal_le_of_isNaN_false (x y : F)
    (hx : isNaN x = false) (hy : isNaN y = false) :
    x ≤ y ↔ toERealTotal x ≤ toERealTotal y := by
  rw [toERealTotal_eq_model, toERealTotal_eq_model,
    FloatLib.Floats.ExecFloat.Binary.le_iff_le_toModel]
  exact Model.Interval.le_iff_toEReal_le_of_isNaN_eq_false
    (toModel x) (toModel y) hx hy

/-- A nonempty finite set of floats has an element attaining the minimum of its `EReal` image. -/
theorem exists_chooseMin (s : Finset F) (hs : s.Nonempty) :
    ∃ x, x ∈ s ∧ toERealTotal x = (s.image toERealTotal).min' (hs.image _) := by
  classical
  -- `min'` is an element of the image, hence has a preimage in `s`.
  have hmem : (s.image toERealTotal).min' (hs.image _) ∈ (s.image toERealTotal) :=
    Finset.min'_mem _ (hs.image _)
  rcases Finset.mem_image.mp hmem with ⟨x, hx, hxEq⟩
  exact ⟨x, hx, hxEq⟩

/-- Dually, an element attaining the maximum. -/
theorem exists_chooseMax (s : Finset F) (hs : s.Nonempty) :
    ∃ x, x ∈ s ∧ toERealTotal x = (s.image toERealTotal).max' (hs.image _) := by
  classical
  have hmem : (s.image toERealTotal).max' (hs.image _) ∈ (s.image toERealTotal) :=
    Finset.max'_mem _ (hs.image _)
  rcases Finset.mem_image.mp hmem with ⟨x, hx, hxEq⟩
  exact ⟨x, hx, hxEq⟩

/-- A minimizing element of `s`, chosen classically.

Choice rather than computation because several distinct floats can share one `EReal` value (the two
zeros), so "the" minimum is not well defined as a float; every use below only needs some minimizer.
-/
noncomputable def chooseMin (s : Finset F) (hs : s.Nonempty) : F :=
  Classical.choose (exists_chooseMin s hs)

/-- A maximizing element of `s`, chosen classically. -/
noncomputable def chooseMax (s : Finset F) (hs : s.Nonempty) : F :=
  Classical.choose (exists_chooseMax s hs)

/-- `chooseMin` is in the set and attains the minimum of the image. -/
theorem chooseMin_spec (s : Finset F) (hs : s.Nonempty) :
    chooseMin s hs ∈ s ∧ toERealTotal (chooseMin s hs) = (s.image toERealTotal).min' (hs.image _) :=
      by
  simpa [chooseMin] using (Classical.choose_spec (exists_chooseMin s hs))

/-- `chooseMax` is in the set and attains the maximum of the image. -/
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
  if _hnan : ∃ x ∈ s, isNaN x = true then
    I.top
  else if hs : s.Nonempty then
    I.range (chooseMin s hs) (chooseMax s hs)
  else
    I.top

/-- The hull contains every element of the set it was built from. All three soundness proofs below
reduce to this one fact. -/
theorem mem_hull_of_mem (s : Finset F) {x : F} (hx : x ∈ s) : x ∈ hull s := by
  classical
  unfold hull
  by_cases hnan : ∃ z ∈ s, isNaN z = true
  · simp [hnan, I.mem_top]
  · have hs : s.Nonempty := ⟨x, hx⟩
    have hn : ∀ z, z ∈ s → isNaN z = false := by
      intro z hz
      by_contra hz'
      have : isNaN z = true := by simpa using hz'
      exact hnan ⟨z, hz, this⟩
    have hxNaN : isNaN x = false := hn x hx
    have hminNaN : isNaN (chooseMin s hs) = false := hn _ (chooseMin_spec s hs).1
    have hmaxNaN : isNaN (chooseMax s hs) = false := hn _ (chooseMax_spec s hs).1
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

/-- Concretization of an interval as a `Finset`, by filtering the (finite) float type. -/
noncomputable def γFinsetI : I → Finset F
  | I.top => Finset.univ
  | I.range a b => by
      classical
      exact Finset.univ.filter (fun x => x ∈ (I.range a b : I))

/-- Concretization of a box as a `Finset` of points. Finite because `F` is, which is what lets the
abstract operators be defined as images of concrete ones rather than by endpoint formulas. -/
noncomputable def γFinsetBox {d : Nat} (B : I.Box d) : Finset (Fin d → F) := by
  classical
  exact Finset.univ.filter (fun x => ∀ i, x i ∈ B i)

/-- Membership in the `Finset` concretization is coordinatewise interval membership. -/
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

/-- Abstract addition `+♯`: the hull of float addition over the whole input box.

Defined as an image rather than by adding endpoints, because float addition is not monotone in the
presence of NaN and signed zeros, so an endpoint formula would not be exact. `⊤` absorbs. -/
noncomputable def addSharp : I → I → I
  | I.top, _ => I.top
  | _, I.top => I.top
  | A, B =>
      hull <| (γFinsetBox (box2 A B)).image (fun p => ExecFloat.add (p 0) (p 1))

/-- Abstract multiplication `*♯`, again as the hull of the concrete image over the box. -/
noncomputable def mulSharp : I → I → I
  | I.top, _ => I.top
  | _, I.top => I.top
  | A, B =>
      hull <| (γFinsetBox (box2 A B)).image (fun p => ExecFloat.mul (p 0) (p 1))

/-- Abstract ReLU, the hull of the concrete image. -/
noncomputable def reluSharp : I → I
  | I.top => I.top
  | A =>
      hull <| (γFinsetI A).image relu

/-- Interval summation `◦∑♯` from the paper: fold with `addSharp`. -/
def sumSharp {n : Nat} (ts : Fin n → I) : I :=
  (List.finRange n).foldl (fun acc i => addSharp acc (ts i)) (I.range (0 : F)
    (0 : F))

/-!
`OpsExact` implements the finite interval semantics used for exact interval-image statements.

The `Sound` class isolates the operation-level obligations used by higher-level semantic proofs.
The addition, multiplication, and ReLU obligations are proved below, followed by the canonical
`ExecFloat.Binary 8 23` instance.
-/

/-- The per-operation soundness obligations the interval semantics rests on: every abstract
operation must contain the concrete result of any pair of members of its arguments. -/
class Sound : Prop where
  /-- Abstract addition contains every concrete sum of members. -/
  add_sound :
    ∀ {A B : I} {x y : F}, x ∈ A → y ∈ B → ExecFloat.add x y ∈ addSharp A B
  /-- Abstract multiplication contains every concrete product of members. -/
  mul_sound :
    ∀ {A B : I} {x y : F}, x ∈ A → y ∈ B → ExecFloat.mul x y ∈ mulSharp A B
  /-- Abstract ReLU contains the concrete ReLU of every member. -/
  relu_sound :
    ∀ {A : I} {x : F}, x ∈ A → relu x ∈ reluSharp A

/-- Soundness of abstract addition: concrete sums of members stay in the abstract sum. -/
theorem add_sound (A B : I) :
    ∀ {x y : F}, x ∈ A → y ∈ B → ExecFloat.add x y ∈ addSharp A B := by
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
              ExecFloat.add x y ∈
                (γFinsetBox (box2 (I.range a b) (I.range c d))).image (fun p => ExecFloat.add (p 0)
                  (p 1)) := by
            refine Finset.mem_image.mpr ?_
            refine ⟨pair x y, hp, ?_⟩
            simp [pair]
          simpa [addSharp] using (mem_hull_of_mem _ hmem)

/-- Soundness of abstract multiplication. -/
theorem mul_sound (A B : I) :
    ∀ {x y : F}, x ∈ A → y ∈ B → ExecFloat.mul x y ∈ mulSharp A B := by
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
              ExecFloat.mul x y ∈
                (γFinsetBox (box2 (I.range a b) (I.range c d))).image (fun p => ExecFloat.mul (p 0)
                  (p 1)) := by
            refine Finset.mem_image.mpr ?_
            refine ⟨pair x y, hp, ?_⟩
            simp [pair]
          simpa [mulSharp] using (mem_hull_of_mem _ hmem)

/-- Soundness of abstract ReLU. -/
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

/-- Soundness of the interval sum `◦∑♯`: folding concrete additions stays inside the folded
intervals.

Proved by induction on the list rather than on `Fin n`, so that the accumulator interval can vary;
the base case needs `0 ∈ ⟨0, 0⟩`, which is where the non-NaN side condition on zero comes in. -/
theorem sumSharp_sound [Sound] {n : Nat} (ts : Fin n → I) (t : Fin n → F)
    (ht : ∀ i, t i ∈ ts i) :
    (List.finRange n).foldl (fun acc i => ExecFloat.add acc (t i)) (0 : F) ∈ sumSharp ts
      := by
  -- Prove the stronger list-induction form, then instantiate with `List.finRange n`.
  have hz : isNaN (0 : F) = false := by decide
  have h0 : (0 : F) ∈ (I.range (0 : F) (0 : F)) := by
    -- Avoid rewriting via `point` to keep simp from collapsing `∧` goals.
    exact And.intro (ExecLemmas.le_self_of_isNaN_false (x := (0 : F)) hz)
      (ExecLemmas.le_self_of_isNaN_false (x := (0 : F)) hz)
  have hList :
      ∀ (l : List (Fin n)) (accI : I) (accV : F),
        accV ∈ accI →
          (l.foldl (fun acc i => ExecFloat.add acc (t i)) accV) ∈
            (l.foldl (fun acc i => addSharp acc (ts i)) accI) := by
    intro l
    induction l with
    | nil =>
        intro accI accV hacc
        simpa using hacc
    | cons i l ih =>
        intro accI accV hacc
        have hi : t i ∈ ts i := ht i
        have hstep : ExecFloat.add accV (t i) ∈ addSharp accI (ts i) :=
          Sound.add_sound (A := accI) (B := ts i) (x := accV) (y := t i) hacc hi
        simpa using
          (ih (accI := addSharp accI (ts i)) (accV := ExecFloat.add accV (t i)) hstep)
  -- Finish by unfolding `sumSharp` and using the list induction lemma.
  simpa [sumSharp] using hList (List.finRange n) (I.range (0 : F) (0 : F))
    (0 : F) h0

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

/-- A constant target has an exact interval image: the point interval `⟨c, c⟩`.

The easiest instance of the exactness property, and the one the constant-target construction of the
paper needs; the nonemptiness hypothesis is what supplies the min and max witnesses. -/
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

namespace TwoLayerMLPExact

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
        (fun acc j => ExecFloat.add acc (ExecFloat.mul (W i j) (x j)))
        (0 : F)
    ExecFloat.add s (b i)

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

/-- The abstract affine transform is sound, given that no weight or bias is NaN.

Weights enter as point intervals, so the proof is `mulSharp` soundness coordinatewise, then
`sumSharp` soundness, then one `addSharp` for the bias, in exactly the order `aff` computes. -/
theorem aff_sound [OpsExact.Sound] {d m : Nat}
    (W : Fin m → Fin d → F) (b : Fin m → F) (B : I.Box d)
    (hW : ∀ i j, isNaN (W i j) = false)
    (hb : ∀ i, isNaN (b i) = false) :
    ∀ {x : Fin d → F}, x ∈ I.γ B → aff W b x ∈ I.γ (affSharp W b B) := by
  intro x hx i
  -- Soundness of each multiplicative term.
  have hterms : ∀ j, ExecFloat.mul (W i j) (x j) ∈ OpsExact.mulSharp (I.range (W i j) (W i j)) (B
    j) := by
    intro j
    have hWij : (W i j) ∈ I.point (W i j) := I.mem_point_of_isNaN_false (x := W i j) (hW i j)
    have hxj : x j ∈ B j := hx j
    -- Coerce `hWij` from `point` to `range`.
    have hWij' : (W i j) ∈ (I.range (W i j) (W i j)) := by simpa [I.point] using hWij
    exact OpsExact.Sound.mul_sound (A := I.range (W i j) (W i j)) (B := B j) hWij' hxj
  -- Sum soundness via `◦∑♯`.
  have hsum :
      (List.finRange d).foldl (fun acc j => ExecFloat.add acc (ExecFloat.mul (W i j) (x j)))
        (0 : F)
        ∈ OpsExact.sumSharp (fun j => OpsExact.mulSharp (I.range (W i j) (W i j)) (B j)) := by
    -- Apply `sumSharp_sound` with the instantiated term intervals and values.
    simpa using
      (OpsExact.sumSharp_sound (ts := fun j => OpsExact.mulSharp (I.range (W i j) (W i j)) (B j))
        (t := fun j => ExecFloat.mul (W i j) (x j)) hterms)
  -- Add the bias (a point interval).
  have hbi : (b i) ∈ (I.range (b i) (b i)) := by
    simpa [I.point] using (I.mem_point_of_isNaN_false (x := b i) (hb i))
  have hfinal :
      ExecFloat.add
          ((List.finRange d).foldl (fun acc j => ExecFloat.add acc (ExecFloat.mul (W i j) (x j)))
            (0 : F))
          (b i)
        ∈ OpsExact.addSharp
            (OpsExact.sumSharp (fun j => OpsExact.mulSharp (I.range (W i j) (W i j)) (B j)))
            (I.range (b i) (b i)) :=
    OpsExact.Sound.add_sound (A := OpsExact.sumSharp (fun j => OpsExact.mulSharp (I.range (W i j) (W
      i j)) (B j)))
      (B := I.range (b i) (b i)) hsum hbi
  simpa [aff, affSharp] using hfinal

/-- The abstract semantics `ν♯` of a two-layer ReLU network overapproximates the concrete one. -/
theorem eval_sound [OpsExact.Sound] {d h : Nat} (net : Net d h) (B : I.Box d)
    (hW1 : ∀ i j, isNaN (net.W1 i j) = false)
    (hb1 : ∀ i, isNaN (net.b1 i) = false)
    (hW2 : ∀ i j, isNaN (net.W2 i j) = false)
    (hb2 : ∀ i, isNaN (net.b2 i) = false) :
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

/-- Set-level restatement: the image of the concretization is contained in the abstract output.

This is the form the approximation theorem cites, since it speaks about images of sets rather than
about individual points. -/
theorem interval_semantics_sound [OpsExact.Sound] {d h : Nat} (net : Net d h) (B : I.Box d)
    (hW1 : ∀ i j, isNaN (net.W1 i j) = false)
    (hb1 : ∀ i, isNaN (net.b1 i) = false)
    (hW2 : ∀ i j, isNaN (net.W2 i j) = false)
    (hb2 : ∀ i, isNaN (net.b2 i) = false) :
    Set.image (eval net) (I.γ B) ⊆ I.γI (evalSharp net B) := by
  intro y hy
  rcases hy with ⟨x, hx, rfl⟩
  exact eval_sound (net := net) (B := B) hW1 hb1 hW2 hb2 hx

/-- Specialization to a point box: the abstract semantics contains the concrete value at `x`.

Worth stating separately because it says the abstraction has no false negatives at single inputs,
which is what a verifier reports back to a user. -/
theorem eval_sound_pointBox [OpsExact.Sound] {d h : Nat} (net : Net d h) (x : Fin d → F)
    (hx : ∀ i, isNaN (x i) = false)
    (hW1 : ∀ i j, isNaN (net.W1 i j) = false)
    (hb1 : ∀ i, isNaN (net.b1 i) = false)
    (hW2 : ∀ i j, isNaN (net.W2 i j) = false)
    (hb2 : ∀ i, isNaN (net.b2 i) = false) :
    eval net x ∈ evalSharp net (I.pointBox (d := d) x) := by
  have hxBox : x ∈ I.γ (I.pointBox (d := d) x) := I.mem_pointBox_of_isNaN_false (x := x) hx
  exact eval_sound (net := net) (B := I.pointBox (d := d) x) hW1 hb1 hW2 hb2 hxBox

end TwoLayerMLPExact

end

end FloatIntervalApprox

end NN.MLTheory.Proofs.UniversalApproximation
