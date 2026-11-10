/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import Mathlib.Data.FinEnum
public import FloatLib.Floats.Formats.BinaryInterchange.Configured
public import FloatLib.Floats.Formats.IEEE754.Native
public import TorchLean.Verify.UniversalApproximation.Binary

/-!
# Interval semantics of binary32 ReLU networks

We define the interval domain `I` over `Binary32`: `I.range a b` holds the floats between its
endpoints and `I.top` holds every float, NaN included. Since `Binary32` is finite, the abstract
addition, multiplication and ReLU are interval hulls of the exact images of the float operations
over a box, and we prove each of them sound.

The main theorem is `TwoLayerMLPExact.interval_semantics_sound`: when no weight or bias of a
two-layer ReLU network is NaN, the image of a box under the binary32 evaluation lies in the abstract
output. Only `I.top` contains NaN, so NaN outputs are covered as well.
`ExactIntervalImage` says when an abstract evaluator computes exact interval images.
The setting is that of Hwang, Lee, Park, Park, and Saad, *Floating-Point Neural Networks Are
Provably Robust Universal Approximators* (`arXiv:2506.16065`).
-/

@[expose] public section

open FloatLib.Floats (ExecFloat Binary32)
open FloatLib.Floats.ExecFloat.Binary (isNaN toModel)
open FloatLib.Floats.Formats.BinaryInterchange (Model)

namespace TorchLean.UniversalApproximation

namespace FloatIntervalApprox

open Activation (relu)

noncomputable section

/-! ## Finiteness of `Binary32` -/

/-- `Binary32` is finite, since a value is determined by its 32-bit interchange word. -/
instance : Finite Binary32 :=
  Finite.of_injective ExecFloat.Binary.toBits32 fun a b hab => by
    simpa only [ExecFloat.Binary.ofBits32_toBits32] using congrArg ExecFloat.Binary.ofBits32 hab

/-- Enumerating `Binary32` makes the interval hulls below finite constructions, so the abstract
operations are exact. -/
instance : Fintype Binary32 := Fintype.ofFinite Binary32

/-! ## The interval domain `I` -/

/-- The interval domain over `Binary32`: the top element or a pair of float endpoints. -/
inductive I where
  /-- No information: every float is possible, NaN included. -/
  | top : I
  /-- The floats between the two endpoints, inclusive. -/
  | range : Binary32 → Binary32 → I
  deriving Repr

namespace I

/-- Concretization of an abstract interval: `top` is every float and `range a b` is the set of
floats `x` with `a ≤ x` and `x ≤ b`. -/
def γI : I → Set Binary32
  | top => Set.univ
  | range a b => fun x => a ≤ x ∧ x ≤ b

/-- Membership in an abstract interval is membership in its concretization `γI`. -/
instance : Membership Binary32 I where
  mem J x := x ∈ γI J

/-- Concretization of a box of abstract intervals, coordinatewise. -/
def γ {d : Nat} (B : Fin d → I) : Set (Fin d → Binary32) := fun x => ∀ i, x i ∈ B i

/-- Every float lies in `top`, NaN included. -/
@[simp] theorem mem_top (x : Binary32) : x ∈ (top : I) := by
  trivial

/-- Membership in `range a b` is the pair of float comparisons `a ≤ x` and `x ≤ b`. -/
@[simp] theorem mem_range_iff (x a b : Binary32) : x ∈ (range a b : I) ↔ a ≤ x ∧ x ≤ b :=
  Iff.rfl

/-- The point interval `range x x`. -/
@[inline] def point (x : Binary32) : I := range x x

/-- A float that is not NaN lies in its point interval. -/
theorem mem_point_of_isNaN_false (x : Binary32) (hx : isNaN x = false) : x ∈ point x := by
  have h : x ≤ x := (ExecFloat.Binary.le_self_iff x).mpr hx
  exact (mem_range_iff x x x).mpr ⟨h, h⟩

/-- The point box of a vector: `point` in every coordinate. -/
@[inline] def pointBox {d : Nat} (x : Fin d → Binary32) : Fin d → I := fun i => point (x i)

/-- A vector with no NaN coordinate lies in its point box. -/
theorem mem_pointBox_of_isNaN_false {d : Nat} (x : Fin d → Binary32)
    (hx : ∀ i, isNaN (x i) = false) : x ∈ γ (pointBox x) := by
  intro i
  exact mem_point_of_isNaN_false (x i) (hx i)

end I

/-! ## Exact abstract operations -/

section

open I

/-- Away from NaN, the float order agrees with the order of the extended-real values. -/
theorem le_iff_toEReal_le_of_isNaN_eq_false (x y : Binary32)
    (hx : isNaN x = false) (hy : isNaN y = false) :
    x ≤ y ↔ Model.toEReal (toModel x) ≤ Model.toEReal (toModel y) := by
  rw [ExecFloat.Binary.le_iff_le_toModel]
  exact Model.Interval.le_iff_toEReal_le_of_isNaN_eq_false (toModel x) (toModel y) hx hy

/-- A nonempty finite set of floats has a member whose extended-real value is the least one. -/
theorem exists_chooseMin (s : Finset Binary32) (hs : s.Nonempty) :
    ∃ x, x ∈ s ∧ Model.toEReal (toModel x) =
      (s.image fun y => Model.toEReal (toModel y)).min' (hs.image _) :=
  Finset.mem_image.mp (Finset.min'_mem _ _)

/-- A nonempty finite set of floats has a member whose extended-real value is the greatest one. -/
theorem exists_chooseMax (s : Finset Binary32) (hs : s.Nonempty) :
    ∃ x, x ∈ s ∧ Model.toEReal (toModel x) =
      (s.image fun y => Model.toEReal (toModel y)).max' (hs.image _) :=
  Finset.mem_image.mp (Finset.max'_mem _ _)

/-- A member of `s` with the least extended-real value. We choose one classically, since distinct
floats such as the two zeros share a value. -/
def chooseMin (s : Finset Binary32) (hs : s.Nonempty) : Binary32 :=
  Classical.choose (exists_chooseMin s hs)

/-- A member of `s` with the greatest extended-real value, chosen classically. -/
def chooseMax (s : Finset Binary32) (hs : s.Nonempty) : Binary32 :=
  Classical.choose (exists_chooseMax s hs)

/-- `chooseMin s hs` lies in `s` and has the least extended-real value of `s`. -/
theorem chooseMin_spec (s : Finset Binary32) (hs : s.Nonempty) :
    chooseMin s hs ∈ s ∧ Model.toEReal (toModel (chooseMin s hs)) =
      (s.image fun y => Model.toEReal (toModel y)).min' (hs.image _) :=
  Classical.choose_spec (exists_chooseMin s hs)

/-- `chooseMax s hs` lies in `s` and has the greatest extended-real value of `s`. -/
theorem chooseMax_spec (s : Finset Binary32) (hs : s.Nonempty) :
    chooseMax s hs ∈ s ∧ Model.toEReal (toModel (chooseMax s hs)) =
      (s.image fun y => Model.toEReal (toModel y)).max' (hs.image _) :=
  Classical.choose_spec (exists_chooseMax s hs)

/-- When no member of `s` is NaN, `chooseMin s hs ≤ y` for every member `y` of `s`. -/
theorem chooseMin_le_of_mem (s : Finset Binary32) (hs : s.Nonempty)
    (hn : ∀ z ∈ s, isNaN z = false) {y : Binary32} (hy : y ∈ s) : chooseMin s hs ≤ y := by
  refine (le_iff_toEReal_le_of_isNaN_eq_false _ _ (hn _ (chooseMin_spec s hs).1)
    (hn y hy)).2 ?_
  rw [(chooseMin_spec s hs).2]
  exact Finset.min'_le _ _ (Finset.mem_image_of_mem (fun y => Model.toEReal (toModel y)) hy)

/-- When no member of `s` is NaN, `y ≤ chooseMax s hs` for every member `y` of `s`. -/
theorem le_chooseMax_of_mem (s : Finset Binary32) (hs : s.Nonempty)
    (hn : ∀ z ∈ s, isNaN z = false) {y : Binary32} (hy : y ∈ s) : y ≤ chooseMax s hs := by
  refine (le_iff_toEReal_le_of_isNaN_eq_false _ _ (hn y hy)
    (hn _ (chooseMax_spec s hs).1)).2 ?_
  rw [(chooseMax_spec s hs).2]
  exact Finset.le_max' _ _ (Finset.mem_image_of_mem (fun y => Model.toEReal (toModel y)) hy)

/-- The interval hull of a finite set of floats: `top` when the set contains a NaN or is empty,
otherwise the range from a least to a greatest member. -/
def hull (s : Finset Binary32) : I :=
  if _hnan : ∃ x ∈ s, isNaN x = true then
    I.top
  else if hs : s.Nonempty then
    I.range (chooseMin s hs) (chooseMax s hs)
  else
    I.top

/-- The hull of a finite set of floats contains every member of the set. -/
theorem mem_hull_of_mem (s : Finset Binary32) {x : Binary32} (hx : x ∈ s) : x ∈ hull s := by
  unfold hull
  by_cases hnan : ∃ z ∈ s, isNaN z = true
  · simp [hnan]
  · have hn : ∀ z ∈ s, isNaN z = false := by
      intro z hz
      by_contra hz'
      exact hnan ⟨z, hz, by simpa using hz'⟩
    have hs : s.Nonempty := ⟨x, hx⟩
    have hmin := chooseMin_le_of_mem s hs hn hx
    have hmax := le_chooseMax_of_mem s hs hn hx
    simp [hnan, hs, hmin, hmax]

/-- The concretization of an abstract interval as a `Finset`, filtering the finite float type. -/
def γFinsetI : I → Finset Binary32
  | I.top => Finset.univ
  | I.range a b => by
      classical
      exact Finset.univ.filter (fun x => x ∈ (I.range a b : I))

/-- The concretization of a box as a `Finset` of vectors. -/
def γFinsetBox {d : Nat} (B : Fin d → I) : Finset (Fin d → Binary32) := by
  classical
  exact Finset.univ.filter (fun x => ∀ i, x i ∈ B i)

/-- Membership in `γFinsetBox B` is coordinatewise membership in `B`. -/
theorem mem_γFinsetBox_iff {d : Nat} (B : Fin d → I) (x : Fin d → Binary32) :
    x ∈ γFinsetBox B ↔ ∀ i, x i ∈ B i := by
  classical
  simp [γFinsetBox]

/-- Abstract addition: the hull of the float sums over the box `![A, B]`, with `top` absorbing. We
take the image rather than adding endpoints since NaN and signed zeros make float addition fail to
be monotone. -/
def addSharp : I → I → I
  | I.top, _ => I.top
  | _, I.top => I.top
  | A, B => hull <| (γFinsetBox ![A, B]).image fun p => ExecFloat.add (p 0) (p 1)

/-- Abstract multiplication: the hull of the float products over the box `![A, B]`, with `top`
absorbing. -/
def mulSharp : I → I → I
  | I.top, _ => I.top
  | _, I.top => I.top
  | A, B => hull <| (γFinsetBox ![A, B]).image fun p => ExecFloat.mul (p 0) (p 1)

/-- Abstract ReLU: the hull of the image of `relu`, with `top` absorbing. -/
def reluSharp : I → I
  | I.top => I.top
  | A => hull <| (γFinsetI A).image relu

/-- Abstract summation: a left fold of `addSharp` from the point interval at `0`. -/
def sumSharp {n : Nat} (ts : Fin n → I) : I :=
  (List.finRange n).foldl (fun acc i => addSharp acc (ts i)) (I.range 0 0)

/-- The float sum of members `x ∈ A` and `y ∈ B` lies in `addSharp A B`. -/
theorem add_sound (A B : I) :
    ∀ {x y : Binary32}, x ∈ A → y ∈ B → ExecFloat.add x y ∈ addSharp A B := by
  intro x y hx hy
  cases A with
  | top => simp [addSharp]
  | range a b =>
      cases B with
      | top => simp [addSharp]
      | range c d =>
          have hp : ![x, y] ∈ γFinsetBox ![I.range a b, I.range c d] :=
            (mem_γFinsetBox_iff _ _).2 (Fin.forall_fin_two.2 ⟨hx, hy⟩)
          have hmem : ExecFloat.add x y ∈
              (γFinsetBox ![I.range a b, I.range c d]).image
                fun p => ExecFloat.add (p 0) (p 1) :=
            Finset.mem_image.mpr ⟨![x, y], hp, rfl⟩
          exact mem_hull_of_mem _ hmem

/-- The float product of members `x ∈ A` and `y ∈ B` lies in `mulSharp A B`. -/
theorem mul_sound (A B : I) :
    ∀ {x y : Binary32}, x ∈ A → y ∈ B → ExecFloat.mul x y ∈ mulSharp A B := by
  intro x y hx hy
  cases A with
  | top => simp [mulSharp]
  | range a b =>
      cases B with
      | top => simp [mulSharp]
      | range c d =>
          have hp : ![x, y] ∈ γFinsetBox ![I.range a b, I.range c d] :=
            (mem_γFinsetBox_iff _ _).2 (Fin.forall_fin_two.2 ⟨hx, hy⟩)
          have hmem : ExecFloat.mul x y ∈
              (γFinsetBox ![I.range a b, I.range c d]).image
                fun p => ExecFloat.mul (p 0) (p 1) :=
            Finset.mem_image.mpr ⟨![x, y], hp, rfl⟩
          exact mem_hull_of_mem _ hmem

/-- The ReLU of a member `x ∈ A` lies in `reluSharp A`. -/
theorem relu_sound (A : I) : ∀ {x : Binary32}, x ∈ A → relu x ∈ reluSharp A := by
  intro x hx
  cases A with
  | top => simp [reluSharp]
  | range a b =>
      have hx' : x ∈ γFinsetI (I.range a b) := by
        simpa [γFinsetI] using hx
      exact mem_hull_of_mem _ (Finset.mem_image_of_mem relu hx')

/-- When `t i ∈ ts i` for every `i`, the left fold of the float sums of `t` from `0` lies in
`sumSharp ts`. The fold starts in `I.range 0 0`, which contains `0` since `0` is not NaN. -/
theorem sumSharp_sound {n : Nat} (ts : Fin n → I) (t : Fin n → Binary32)
    (ht : ∀ i, t i ∈ ts i) :
    (List.finRange n).foldl (fun acc i => ExecFloat.add acc (t i)) (0 : Binary32) ∈
      sumSharp ts := by
  have hList : ∀ (l : List (Fin n)) (accI : I) (accV : Binary32), accV ∈ accI →
      l.foldl (fun acc i => ExecFloat.add acc (t i)) accV ∈
        l.foldl (fun acc i => addSharp acc (ts i)) accI := by
    intro l
    induction l with
    | nil =>
        intro accI accV hacc
        simpa using hacc
    | cons i l ih =>
        intro accI accV hacc
        simpa using ih (addSharp accI (ts i)) (ExecFloat.add accV (t i))
          (add_sound accI (ts i) hacc (ht i))
  exact hList (List.finRange n) (I.range 0 0) 0 (I.mem_point_of_isNaN_false 0 (by decide))

end

/-! ## Exact interval images -/

/-- The abstract evaluator `nuInt` computes exact interval images of `g`: on every box `B` with a
member, the floats in `nuInt B` are those between a least and a greatest value of `g` on `B`. -/
def ExactIntervalImage {d : Nat} {J : Type} [Membership Binary32 J]
    (g : (Fin d → Binary32) → Binary32) (nuInt : (Fin d → J) → J) : Prop :=
  ∀ B : Fin d → J, {x : Fin d → Binary32 | ∀ i, x i ∈ B i}.Nonempty →
    ∃ m M, IsLeast (g '' {x | ∀ i, x i ∈ B i}) m ∧ IsGreatest (g '' {x | ∀ i, x i ∈ B i}) M ∧
      {x : Binary32 | x ∈ nuInt B} = {x | m ≤ x ∧ x ≤ M}

/-- A constant `c` that is not NaN has exact interval images under the constant evaluator
`I.range c c`. -/
theorem exactIntervalImage_constant {d : Nat} (c : Binary32) (hc : isNaN c = false) :
    ExactIntervalImage (d := d) (fun _ => c) (fun _ => I.range c c) := by
  intro B hne
  have hcc : c ≤ c := (ExecFloat.Binary.le_self_iff c).mpr hc
  obtain ⟨x0, hx0⟩ := hne
  refine ⟨c, c, ⟨⟨x0, hx0, rfl⟩, ?_⟩, ⟨⟨x0, hx0, rfl⟩, ?_⟩, rfl⟩
  · rintro _ ⟨_, _, rfl⟩
    exact hcc
  · rintro _ ⟨_, _, rfl⟩
    exact hcc

/-- The ideal abstraction of `h` on a box `B`: the hull of the image of `h` over the finite
concretization `γFinsetBox B`. -/
def idealSharp {d : Nat} (h : (Fin d → Binary32) → Binary32) (B : Fin d → I) : I :=
  hull ((γFinsetBox B).image h)

/-- When `h` never returns NaN and `I.γ B` is nonempty, `idealSharp h B` concretizes to the floats
between a least and a greatest value of `h` on `I.γ B`. -/
theorem exists_minmax_for_idealSharp {d : Nat} (h : (Fin d → Binary32) → Binary32)
    (B : Fin d → I) (hB : (I.γ B).Nonempty) (hnan : ∀ x, isNaN (h x) = false) :
    ∃ m M, IsLeast (h '' I.γ B) m ∧ IsGreatest (h '' I.γ B) M ∧
      I.γI (idealSharp h B) = {x | m ≤ x ∧ x ≤ M} := by
  unfold idealSharp
  have hmem : ∀ z, z ∈ (γFinsetBox B).image h ↔ z ∈ h '' I.γ B := by
    intro z
    simp only [Finset.mem_image, Set.mem_image, mem_γFinsetBox_iff]
    rfl
  generalize (γFinsetBox B).image h = s at hmem ⊢
  have hs : s.Nonempty := by
    obtain ⟨x0, hx0⟩ := hB
    exact ⟨h x0, (hmem _).2 ⟨x0, hx0, rfl⟩⟩
  have hn : ∀ z ∈ s, isNaN z = false := by
    intro z hz
    obtain ⟨x, -, rfl⟩ := (hmem z).1 hz
    exact hnan x
  have hnoNaN : ¬∃ z ∈ s, isNaN z = true := by
    rintro ⟨z, hz, hz'⟩
    rw [hn z hz] at hz'
    cases hz'
  refine ⟨chooseMin s hs, chooseMax s hs,
    ⟨(hmem _).1 (chooseMin_spec s hs).1,
      fun y hy => chooseMin_le_of_mem s hs hn ((hmem y).2 hy)⟩,
    ⟨(hmem _).1 (chooseMax_spec s hs).1,
      fun y hy => le_chooseMax_of_mem s hs hn ((hmem y).2 hy)⟩, ?_⟩
  rw [hull, dite_eq_right hnoNaN, dite_eq_left hs]
  rfl

/-! ## Two-layer ReLU networks -/

namespace TwoLayerMLPExact

/-- Parameters of a two-layer network of shape `d → h → 1` over `Binary32`. -/
structure Net (d h : Nat) where
  /-- Weight matrix of the first layer. -/
  W1 : Fin h → Fin d → Binary32
  /-- Bias of the first layer. -/
  b1 : Fin h → Binary32
  /-- Weight matrix of the second layer. -/
  W2 : Fin 1 → Fin h → Binary32
  /-- Bias of the second layer. -/
  b2 : Fin 1 → Binary32

/-- An affine layer in binary32 arithmetic: a left fold of the products `W i j * x j` from `0`,
then the bias `b i`. -/
def aff {d m : Nat} (W : Fin m → Fin d → Binary32) (b : Fin m → Binary32)
    (x : Fin d → Binary32) : Fin m → Binary32 :=
  fun i =>
    let s :=
      (List.finRange d).foldl
        (fun acc j => ExecFloat.add acc (ExecFloat.mul (W i j) (x j))) (0 : Binary32)
    ExecFloat.add s (b i)

/-- The binary32 evaluation of a two-layer ReLU network. -/
def eval {d h : Nat} (net : Net d h) (x : Fin d → Binary32) : Binary32 :=
  let z1 : Fin h → Binary32 := aff net.W1 net.b1 x
  let a1 : Fin h → Binary32 := fun i => relu (z1 i)
  let z2 : Fin 1 → Binary32 := aff net.W2 net.b2 a1
  z2 0

/-- The abstract affine layer: `mulSharp` with point weights, then `sumSharp`, then `addSharp`
with the point bias. -/
def affSharp {d m : Nat} (W : Fin m → Fin d → Binary32) (b : Fin m → Binary32)
    (B : Fin d → I) : Fin m → I :=
  fun i =>
    let terms : Fin d → I := fun j => mulSharp (I.range (W i j) (W i j)) (B j)
    let s := sumSharp terms
    addSharp s (I.range (b i) (b i))

/-- The abstract evaluation of a two-layer ReLU network on a box. -/
def evalSharp {d h : Nat} (net : Net d h) (B : Fin d → I) : I :=
  let z1 : Fin h → I := affSharp net.W1 net.b1 B
  let a1 : Fin h → I := fun i => reluSharp (z1 i)
  let z2 : Fin 1 → I := affSharp net.W2 net.b2 a1
  z2 0

/-- When no weight or bias is NaN, `aff W b` maps every member of `I.γ B` into
`I.γ (affSharp W b B)`. -/
theorem aff_sound {d m : Nat} (W : Fin m → Fin d → Binary32) (b : Fin m → Binary32)
    (B : Fin d → I) (hW : ∀ i j, isNaN (W i j) = false) (hb : ∀ i, isNaN (b i) = false) :
    ∀ {x : Fin d → Binary32}, x ∈ I.γ B → aff W b x ∈ I.γ (affSharp W b B) := by
  intro x hx i
  have hterms : ∀ j, ExecFloat.mul (W i j) (x j) ∈ mulSharp (I.range (W i j) (W i j)) (B j) :=
    fun j => mul_sound _ _ (I.mem_point_of_isNaN_false (W i j) (hW i j)) (hx j)
  exact add_sound _ _ (sumSharp_sound _ _ hterms) (I.mem_point_of_isNaN_false (b i) (hb i))

/-- When no weight or bias is NaN, `eval net` maps every member of `I.γ B` into
`evalSharp net B`. -/
theorem eval_sound {d h : Nat} (net : Net d h) (B : Fin d → I)
    (hW1 : ∀ i j, isNaN (net.W1 i j) = false)
    (hb1 : ∀ i, isNaN (net.b1 i) = false)
    (hW2 : ∀ i j, isNaN (net.W2 i j) = false)
    (hb2 : ∀ i, isNaN (net.b2 i) = false) :
    ∀ {x : Fin d → Binary32}, x ∈ I.γ B → eval net x ∈ evalSharp net B := by
  intro x hx
  have hz1 := aff_sound net.W1 net.b1 B hW1 hb1 hx
  have ha1 : (fun i => relu (aff net.W1 net.b1 x i)) ∈
      I.γ (fun i => reluSharp (affSharp net.W1 net.b1 B i)) := by
    intro i
    exact relu_sound _ (hz1 i)
  exact aff_sound net.W2 net.b2 _ hW2 hb2 ha1 0

/-- When no weight or bias is NaN, the image of `I.γ B` under `eval net` lies in
`I.γI (evalSharp net B)`. Only `I.top` contains NaN, so this covers NaN outputs as well. -/
theorem interval_semantics_sound {d h : Nat} (net : Net d h) (B : Fin d → I)
    (hW1 : ∀ i j, isNaN (net.W1 i j) = false)
    (hb1 : ∀ i, isNaN (net.b1 i) = false)
    (hW2 : ∀ i j, isNaN (net.W2 i j) = false)
    (hb2 : ∀ i, isNaN (net.b2 i) = false) :
    Set.image (eval net) (I.γ B) ⊆ I.γI (evalSharp net B) := by
  rintro _ ⟨x, hx, rfl⟩
  exact eval_sound net B hW1 hb1 hW2 hb2 hx

/-- When no coordinate of `x`, weight or bias is NaN, `eval net x` lies in
`evalSharp net (I.pointBox x)`. -/
theorem eval_sound_pointBox {d h : Nat} (net : Net d h) (x : Fin d → Binary32)
    (hx : ∀ i, isNaN (x i) = false)
    (hW1 : ∀ i j, isNaN (net.W1 i j) = false)
    (hb1 : ∀ i, isNaN (net.b1 i) = false)
    (hW2 : ∀ i j, isNaN (net.W2 i j) = false)
    (hb2 : ∀ i, isNaN (net.b2 i) = false) :
    eval net x ∈ evalSharp net (I.pointBox x) :=
  eval_sound net (I.pointBox x) hW1 hb1 hW2 hb2 (I.mem_pointBox_of_isNaN_false x hx)

end TwoLayerMLPExact

end

end FloatIntervalApprox

end TorchLean.UniversalApproximation
