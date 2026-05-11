/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Core.Tensor.Constructors
public import NN.Spec.Core.Tensor.Linalg
public import NN.Spec.Core.TensorOps

/-!
# Tensor Algebra Proofs

Backend-generic algebraic tensor lemmas.

This file provides the semiring-level dot-product API and supporting fold/sum lemmas used by
autograd and tape soundness proofs. The point of this file is to avoid baking `ℝ`, topology, or
calculus assumptions into algebra that only needs addition, multiplication, and finite sums.

`NN.Proofs.Tensor.Basic` imports this file and adds the `ℝ`-specialized spec-facing toolkit
(norms, shape facts, Frobenius identities, and convenience lemmas for analysis proofs). Keeping this
file generic lets correctness theorems reuse the same algebra over exact rationals, real numbers, or
other semiring-like scalar models.

## PyTorch correspondence / citations

- Elementwise product + sum (“flattened dot” for same-shape tensors): `(a * b).sum()`.
  https://pytorch.org/docs/stable/generated/torch.sum.html
- 1D dot product: `torch.dot(a, b)` (for real tensors).
  https://pytorch.org/docs/stable/generated/torch.dot.html
-/

@[expose] public section


namespace Proofs
namespace TensorAlgebra

open Spec
open Tensor

noncomputable section

/-! ## Vector views -/

/--
View a 1D tensor of shape `(n,)` as a function `Fin n → α`.

Informally: this is the spec-level analogue of PyTorch indexing `t[i]` for a vector.
-/
def toVec {α : Type} {n : Nat} (t : Tensor α (.dim n .scalar)) : Fin n → α :=
  match t with
  | .dim f => fun i => match f i with | .scalar x => x

/--
Build a 1D tensor of shape `(n,)` from a function `Fin n → α`.

`ofVec` is the inverse direction of `toVec` (up to definitional equality).
-/
def ofVec {α : Type} {n : Nat} (v : Fin n → α) : Tensor α (.dim n .scalar) :=
  .dim (fun i => .scalar (v i))

@[simp] theorem toVec_ofVec {α : Type} {n : Nat} (v : Fin n → α) :
    toVec (ofVec v) = v := by
  funext i
  simp [toVec, ofVec]

theorem ofVec_toVec {α : Type} {n : Nat} (t : Tensor α (.dim n .scalar)) :
    ofVec (toVec t) = t := by
  cases t with
  | dim f =>
    apply congrArg Tensor.dim
    funext i
    cases h : f i with
    | scalar x =>
      simp [toVec, h]

/-! ## Recursive same-shape dot product -/

/--
Dot product for tensors of the **same shape**: multiply elementwise and sum over all scalar leaves.

For shape `(n,)` this matches `torch.dot` (over a commutative semiring, i.e. ignoring complex
conjugation). More generally it matches the common PyTorch idiom `(a * b).sum()` for same-shape
tensors.
-/
def dot {α : Type} [Zero α] [Add α] [Mul α] :
    ∀ {s : Shape}, Tensor α s → Tensor α s → α
  | .scalar, Tensor.scalar a, Tensor.scalar b => a * b
  | .dim n s, Tensor.dim f, Tensor.dim g =>
      (List.finRange n).foldl (fun acc i => acc + dot (s := s) (f i) (g i)) 0

/-! ## List-fold helpers -/

private lemma foldl_add_congr {α β : Type} [Add α] (l : List β) (f g : β → α) (a : α)
    (h : ∀ x, f x = g x) :
    l.foldl (fun s x => s + f x) a = l.foldl (fun s x => s + g x) a := by
  induction l generalizing a with
  | nil => simp
  | cons hd tl ih =>
    simp [List.foldl, h hd, ih]

private lemma foldl_add_const_zero {α β : Type} [AddMonoid α] (l : List β) (a : α) :
    l.foldl (fun s _ => s + (0 : α)) a = a := by
  induction l generalizing a with
  | nil => simp
  | cons _ tl ih =>
    simp [List.foldl, add_zero]

/--
Distribute a fold of `g1 x + g2 x` into the sum of two folds.

This is a general “sum splits over addition” lemma used to prove bilinearity-like properties of
`dot`.
-/
lemma foldl_add_distrib2 {α β : Type} [AddCommMonoid α] (l : List β) (g1 g2 : β → α) :
    ∀ a1 a2,
      l.foldl (fun s x => s + (g1 x + g2 x)) (a1 + a2) =
        l.foldl (fun s x => s + g1 x) a1 + l.foldl (fun s x => s + g2 x) a2 := by
  intro a1 a2
  induction l generalizing a1 a2 with
  | nil =>
    simp
  | cons hd tl ih =>
    simp [List.foldl]
    have hacc :
        (a1 + a2) + (g1 hd + g2 hd) = (a1 + g1 hd) + (a2 + g2 hd) := by
      abel
    simpa [hacc, add_assoc, add_left_comm, add_comm] using
      (ih (a1 := a1 + g1 hd) (a2 := a2 + g2 hd))

/-! ## From executable folds to proof-friendly sums -/

/--
Rewrite the spec-style fold over `List.finRange n` into a proof-friendly `Finset.univ.sum`.

Many spec definitions use `List.foldl` (it is definitional and convenient for computation), while
proofs often prefer `Finset.sum` so they can use standard big-operator lemmas.
-/
lemma finRange_foldl_add_eq_finset_sum {α : Type} [AddCommMonoid α] {n : Nat} (f : Fin n → α) :
    (List.finRange n).foldl (fun s i => s + f i) 0 = (Finset.univ : Finset (Fin n)).sum f := by
  classical
  have hL :
      (List.finRange n).foldl (fun s i => s + f i) 0 = ((List.finRange n).map f).sum := by
    have hmap :
        (List.finRange n).foldl (fun s i => s + f i) 0 =
          List.foldl (fun s x => s + x) 0 ((List.finRange n).map f) := by
      simpa using
        (List.foldl_map (f := f) (g := fun s x => s + x) (l := List.finRange n) (init := (0 :
          α))).symm
    have hfold :
        List.foldl (fun s x : α => s + x) 0 ((List.finRange n).map f) =
          ((List.finRange n).map f).sum := by
      simpa [List.sum] using
        (List.foldl_eq_foldr (f := fun s x : α => s + x) (a := (0 : α)) ((List.finRange n).map f))
    simpa [hmap] using hfold
  have hR :
      (Finset.univ : Finset (Fin n)).sum f = ((List.finRange n).map f).sum := by
    simp [Finset.sum, Finset.val_univ_fin, Multiset.map_coe, Multiset.sum_coe]
  exact hL.trans hR.symm

/-! ## Dot-product algebra -/

section

variable {α : Type} [CommSemiring α]

@[simp] theorem dot_scalar (a b : α) :
    dot (α := α) (s := Shape.scalar) (Tensor.scalar a) (Tensor.scalar b) = a * b := by
  simp [dot]

/-! ### Scaling -/

private lemma foldl_add_mul_right {β : Type} (l : List β) (g : β → α) (a k : α) :
    l.foldl (fun acc x => acc + g x * k) (a * k) = (l.foldl (fun acc x => acc + g x) a) * k := by
  induction l generalizing a with
  | nil =>
    simp
  | cons hd tl ih =>
    -- Rewrite the starting accumulator using distributivity, then apply IH.
    have hstart : a * k + g hd * k = (a + g hd) * k := by
      simp [add_mul]
    simpa [List.foldl, hstart] using (ih (a := a + g hd))

theorem dot_scale_right {s : Shape} (a b : Tensor α s) (k : α) :
    dot (α := α) a (scaleSpec (α := α) (s := s) b k) = dot (α := α) a b * k := by
  induction s with
  | scalar =>
    cases a; cases b
    simp [TensorAlgebra.dot, Tensor.scaleSpec, Tensor.mapSpec, mul_assoc]
  | dim n s ih =>
    cases a with
    | dim fa =>
      cases b with
      | dim fb =>
        have hterm :
            ∀ i : Fin n,
              dot (α := α) (fa i) (scaleSpec (α := α) (s := s) (fb i) k) =
                dot (α := α) (fa i) (fb i) * k := by
          intro i
          simpa using (ih (a := fa i) (b := fb i))
        have hcongr :
            (List.finRange n).foldl
                (fun acc i => acc + dot (α := α) (fa i) (scaleSpec (α := α) (s := s) (fb i) k)) 0 =
              (List.finRange n).foldl (fun acc i => acc + dot (α := α) (fa i) (fb i) * k) 0 := by
          simpa using
            (foldl_add_congr (l := List.finRange n)
              (f := fun i => dot (α := α) (fa i) (scaleSpec (α := α) (s := s) (fb i) k))
              (g := fun i => dot (α := α) (fa i) (fb i) * k)
              (a := (0 : α)) (h := hterm))
        have hmul :
            (List.finRange n).foldl (fun acc i => acc + dot (α := α) (fa i) (fb i) * k) 0 =
              (List.finRange n).foldl (fun acc i => acc + dot (α := α) (fa i) (fb i)) 0 * k := by
          -- Use the fold lemma with initial accumulator `0`.
          simpa [zero_mul] using
            (foldl_add_mul_right (α := α) (l := List.finRange n)
              (g := fun i => dot (α := α) (fa i) (fb i)) (a := (0 : α)) (k := k))
        calc
          dot (α := α) (Tensor.dim fa) (scaleSpec (α := α) (s := Shape.dim n s) (Tensor.dim fb) k)
              =
              (List.finRange n).foldl
                (fun acc i => acc + dot (α := α) (fa i) (scaleSpec (α := α) (s := s) (fb i) k)) 0
                  := by
                simp [TensorAlgebra.dot, Tensor.scaleSpec, Tensor.mapSpec]
          _ =
              (List.finRange n).foldl (fun acc i => acc + dot (α := α) (fa i) (fb i) * k) 0 :=
                hcongr
          _ =
              (List.finRange n).foldl (fun acc i => acc + dot (α := α) (fa i) (fb i)) 0 * k := hmul
          _ = dot (α := α) (Tensor.dim fa) (Tensor.dim fb) * k := by
                simp [TensorAlgebra.dot]

/-- Symmetry of `dot` (commutativity) over a commutative semiring. -/
theorem dot_comm {s : Shape} (a b : Tensor α s) :
    dot (α := α) a b = dot (α := α) b a := by
  induction s with
  | scalar =>
    cases a; cases b
    simp [dot, mul_comm]
  | dim n s ih =>
    cases a with
    | dim fa =>
      cases b with
      | dim fb =>
        have hterm : ∀ i : Fin n, dot (α := α) (fa i) (fb i) = dot (α := α) (fb i) (fa i) := by
          intro i
          simpa using (ih (a := fa i) (b := fb i))
        -- Use congruence over the fold.
        have hfold :
            (List.finRange n).foldl (fun acc i => acc + dot (α := α) (fa i) (fb i)) 0 =
              (List.finRange n).foldl (fun acc i => acc + dot (α := α) (fb i) (fa i)) 0 := by
          simpa using (foldl_add_congr (l := List.finRange n)
            (f := fun i => dot (α := α) (fa i) (fb i))
            (g := fun i => dot (α := α) (fb i) (fa i)) (a := (0 : α)) (h := hterm))
        simpa [dot] using hfold

/-- Left-scaling: `dot (k • a) b = k * dot a b`. -/
theorem dot_scale_left {s : Shape} (a b : Tensor α s) (k : α) :
    dot (α := α) (scaleSpec (α := α) (s := s) a k) b = dot (α := α) a b * k := by
  -- Reduce to `dot_scale_right` by commutativity of the dot product.
  calc
    dot (α := α) (scaleSpec (α := α) (s := s) a k) b
        = dot (α := α) b (scaleSpec (α := α) (s := s) a k) := by
            simpa using (dot_comm (α := α) (a := scaleSpec (α := α) (s := s) a k) (b := b))
    _ = dot (α := α) b a * k := dot_scale_right (α := α) (s := s) (a := b) (b := a) (k := k)
    _ = dot (α := α) a b * k := by
          simp [dot_comm (α := α)]

/-- Additivity in the left argument: `dot (a + b) c = dot a c + dot b c`. -/
theorem dot_add_left {s : Shape} (a b c : Tensor α s) :
    dot (α := α) (addSpec a b) c = dot (α := α) a c + dot (α := α) b c := by
  induction s with
  | scalar =>
    cases a; cases b; cases c
    simp [dot, addSpec, map2Spec, add_mul]
  | dim n s ih =>
    cases a with
    | dim fa =>
      cases b with
      | dim fb =>
        cases c with
        | dim fc =>
          -- Rewrite the dot at dimension level to a single list fold, then split it.
          have hterm :
              ∀ i : Fin n,
                dot (α := α) (addSpec (fa i) (fb i)) (fc i) =
                  dot (α := α) (fa i) (fc i) + dot (α := α) (fb i) (fc i) := by
            intro i
            simpa using (ih (a := fa i) (b := fb i) (c := fc i))
          have hfold :
              (List.finRange n).foldl (fun acc i => acc + dot (α := α) (addSpec (fa i) (fb i)) (fc
                i)) 0 =
                (List.finRange n).foldl (fun acc i => acc + dot (α := α) (fa i) (fc i)) 0 +
                  (List.finRange n).foldl (fun acc i => acc + dot (α := α) (fb i) (fc i)) 0 := by
            -- First rewrite each term using `hterm`, then apply `foldl_add_distrib2`.
            have hcongr :
                (List.finRange n).foldl (fun acc i => acc + dot (α := α) (addSpec (fa i) (fb i))
                  (fc i)) 0 =
                  (List.finRange n).foldl
                    (fun acc i => acc + (dot (α := α) (fa i) (fc i) + dot (α := α) (fb i) (fc i))) 0
                      := by
              simpa using (foldl_add_congr (l := List.finRange n)
                (f := fun i => dot (α := α) (addSpec (fa i) (fb i)) (fc i))
                (g := fun i => dot (α := α) (fa i) (fc i) + dot (α := α) (fb i) (fc i))
                (a := (0 : α)) (h := hterm))
            -- Now split the fold into two folds.
            have hsplit :=
              foldl_add_distrib2 (l := List.finRange n)
                (g1 := fun i => dot (α := α) (fa i) (fc i))
                (g2 := fun i => dot (α := α) (fb i) (fc i)) (a1 := (0 : α)) (a2 := (0 : α))
            simpa [hcongr, add_assoc, add_left_comm, add_comm] using hsplit
          simpa [dot, addSpec, map2Spec] using hfold

/-- Additivity in the right argument: `dot a (b + c) = dot a b + dot a c`. -/
theorem dot_add_right {s : Shape} (a b c : Tensor α s) :
    dot (α := α) a (addSpec b c) = dot (α := α) a b + dot (α := α) a c := by
  -- Derive from `dot_add_left` using commutativity of dot.
  calc
    dot (α := α) a (addSpec b c)
        = dot (α := α) (addSpec b c) a := by
            simpa using (dot_comm (α := α) (a := a) (b := addSpec b c))
    _ = dot (α := α) b a + dot (α := α) c a := by
          simpa using (dot_add_left (α := α) (a := b) (b := c) (c := a))
    _ = dot (α := α) a b + dot (α := α) a c := by
          simp [dot_comm (α := α)]

/-- `dot a 0 = 0` (where `0` is the all-zeros tensor via `fill`). -/
theorem dot_fill_zero_right {s : Shape} (a : Tensor α s) :
    dot (α := α) a (fill (0 : α) s) = 0 := by
  induction s with
  | scalar =>
    cases a
    simp [fill, mul_zero]
  | dim n s ih =>
    cases a with
    | dim fa =>
      have hterm : ∀ i : Fin n, dot (α := α) (fa i) (fill (0 : α) s) = 0 := by
        intro i
        simpa using (ih (a := fa i))
      have hcongr :
          (List.finRange n).foldl (fun acc i => acc + dot (α := α) (fa i) (fill (0 : α) s)) 0 =
            (List.finRange n).foldl (fun acc (_i : Fin n) => acc + (0 : α)) 0 := by
        simpa using (foldl_add_congr (l := List.finRange n)
          (f := fun i => dot (α := α) (fa i) (fill (0 : α) s))
          (g := fun _i => (0 : α)) (a := (0 : α)) (h := hterm))
      calc
        dot (α := α) (Tensor.dim fa) (fill (0 : α) (Shape.dim n s))
            = (List.finRange n).foldl (fun acc i => acc + dot (α := α) (fa i) (fill (0 : α) s)) 0 :=
              by
                simp [dot, fill]
        _ = (List.finRange n).foldl (fun acc (_i : Fin n) => acc + (0 : α)) 0 := hcongr
        _ = 0 := by
              simp

-- Dot product of vectors as a `Finset` sum over coordinates.
/--
For vectors `(n,)`, `dot` can be expressed as an explicit `Finset.univ.sum`.

This is the bridge between the recursive `dot` definition and “PyTorch-looking” summation formulas.
-/
theorem dot_vec_eq_sum {n : Nat} (a b : Tensor α (.dim n .scalar)) :
    dot (α := α) a b = ∑ i : Fin n, (toVec a i) * (toVec b i) := by
  classical
  cases a with
  | dim fa =>
    cases b with
    | dim fb =>
      have hfold :
          (List.finRange n).foldl (fun s i => s + dot (α := α) (fa i) (fb i)) 0 =
            (∑ i : Fin n, dot (α := α) (fa i) (fb i)) :=
        finRange_foldl_add_eq_finset_sum (f := fun i : Fin n => dot (α := α) (fa i) (fb i))
      have hterm :
          (∑ i : Fin n, dot (α := α) (fa i) (fb i)) =
            (∑ i : Fin n, (toVec (Tensor.dim fa) i) * (toVec (Tensor.dim fb) i)) := by
        refine Finset.sum_congr rfl ?_
        intro i _
        cases hfa : fa i with
        | scalar ai =>
          cases hfb : fb i with
          | scalar bi =>
            simp [TensorAlgebra.dot, toVec, hfa, hfb]
      calc
        dot (α := α) (Tensor.dim fa) (Tensor.dim fb)
            = (List.finRange n).foldl (fun s i => s + dot (α := α) (fa i) (fb i)) 0 := by
                simp [TensorAlgebra.dot]
        _ = ∑ i : Fin n, dot (α := α) (fa i) (fb i) := hfold
        _ = ∑ i : Fin n, (toVec (Tensor.dim fa) i) * (toVec (Tensor.dim fb) i) := hterm

/-! ## Matrix/vector adjointness: `⟪y, W x⟫ = ⟪y W, x⟫` -/

omit [CommSemiring α] in
lemma get2_eq {m n : Nat} (A : Tensor α (.dim m (.dim n .scalar))) (i : Fin m) (j : Fin n) :
    get2 A i j =
      match Spec.get A i with
      | Tensor.dim row =>
          match row j with
          | Tensor.scalar v => v := by
  unfold get2
  rfl

omit [CommSemiring α] in
lemma get_eq {m : Nat} {s : Shape} (t : Tensor α (.dim m s)) (i : Fin m) :
    Spec.get t i = match t with
      | Tensor.dim f => f i := by
  unfold Spec.get
  rfl

-- Helper: convert the scalar-tensor fold used in `mat_vec_mul_spec` into a fold on scalars.
lemma foldl_matvec_scalar {n : Nat} (l : List (Fin n)) (a : α)
  (cols vals : Fin n → Tensor α Shape.scalar) :
  l.foldl
      (fun (acc : Tensor α Shape.scalar) (k : Fin n) =>
        Spec.matVecMulSpec.match_1
          (motive := fun _ _ _ => Tensor α Shape.scalar)
          acc (cols k) (vals k)
          (fun s ak vk => Tensor.scalar (s + ak * vk)))
      (Tensor.scalar a)
    =
    Tensor.scalar
      (l.foldl
        (fun (s : α) (k : Fin n) =>
          Spec.matMulSpec.match_1 (motive := fun _ _ => α) (cols k) (vals k)
            (fun ak vk => s + ak * vk))
        a) := by
  induction l generalizing a with
  | nil =>
    simp [List.foldl]
  | cons hd tl ih =>
    cases hcol : cols hd with
    | scalar ak =>
      cases hval : vals hd with
      | scalar vk =>
        simp [List.foldl, hcol, hval, ih]

/--
Matrix-vector multiply coordinate expansion.

This is a spec-to-proof bridge: it turns the `List.finRange` fold in `mat_vec_mul_spec` into a
`Finset.univ.sum` formula, matching the textbook / PyTorch view of matvec as a dot product.
-/
lemma toVec_mat_vec_mul_spec {m n : Nat}
  (A : Tensor α (.dim m (.dim n .scalar)))
  (v : Tensor α (.dim n .scalar)) (i : Fin m) :
  toVec (matVecMulSpec A v) i = ∑ k : Fin n, (get2 A i k) * (toVec v k) := by
  classical
  cases A with
  | dim rowsA =>
    cases v with
    | dim valuesV =>
      cases hrow : rowsA i with
      | dim colsA =>
        -- Unfold `toVec (mat_vec_mul_spec A v) i`:
        -- this produces a `match` that extracts the scalar from the fold accumulator.
        simp [matVecMulSpec, toVec, hrow]
        -- Convert the scalar-tensor fold into a scalar fold on `α`, then rewrite the outer `match`
        -- extraction.
        have hextract := by
          have hfold :=
            foldl_matvec_scalar (l := List.finRange n) (a := (0 : α)) (cols := colsA) (vals :=
              valuesV)
          -- Apply the scalar-extraction function `Tensor.scalar x ↦ x` to both sides.
          simpa using congrArg (fun t => match t with | Tensor.scalar x => x) hfold
        rw [hextract]
        -- Put the fold into canonical `s + f k` form, then convert to a `Finset.univ.sum`.
        let f : Fin n → α := fun k =>
          Spec.matMulSpec.match_1 (motive := fun _ _ => α) (colsA k) (valuesV k) (fun ak vk => ak
            * vk)
        have hfun :
            (fun (s : α) (k : Fin n) =>
                Spec.matMulSpec.match_1 (motive := fun _ _ => α) (colsA k) (valuesV k)
                  (fun ak vk => s + ak * vk))
              =
              (fun s k => s + f k) := by
          funext s k
          cases hcol : colsA k with
          | scalar ak =>
            cases hv : valuesV k with
            | scalar vk =>
              simp [f, hcol, hv]
        rw [hfun]
        have hsum : (List.finRange n).foldl (fun s k => s + f k) 0 = ∑ k : Fin n, f k := by
          simpa using finRange_foldl_add_eq_finset_sum (f := f)
        rw [hsum]
        have hf :
            ∀ k : Fin n, f k = get2 (Tensor.dim rowsA) i k * toVec (Tensor.dim valuesV) k := by
          intro k
          cases hcol : colsA k with
          | scalar ak =>
            cases hv : valuesV k with
            | scalar vk =>
              simp [f, get2_eq, get_eq, toVec, hrow, hcol, hv]
        refine (Finset.sum_congr rfl ?_)
        intro k _
        simpa using hf k

/--
Vector-matrix multiply coordinate expansion.

This is the right-adjoint analogue of `toVec_mat_vec_mul_spec`.
-/
lemma toVec_vec_mat_mul_spec {m n : Nat}
  (v : Tensor α (.dim m .scalar))
  (A : Tensor α (.dim m (.dim n .scalar))) (j : Fin n) :
  toVec (vecMatMulSpec v A) j = ∑ i : Fin m, (toVec v i) * (get2 A i j) := by
  classical
  cases v with
  | dim valuesV =>
    cases A with
    | dim rowsA =>
      simp [vecMatMulSpec, toVec, get2_eq, get_eq]
      let f : Fin m → α := fun i =>
        vecMatMulSpec.match_3
          (motive := fun _ _ => α)
          (valuesV i) (rowsA i)
          (fun vi colsA =>
            vecMatMulSpec.match_1
              (motive := fun _ => α)
              (colsA j)
              (fun aij => vi * aij))
      have hfun :
          (fun (s : α) (i : Fin m) =>
              vecMatMulSpec.match_3
                (motive := fun _ _ => α)
                (valuesV i) (rowsA i)
                (fun vi colsA =>
                  vecMatMulSpec.match_1
                    (motive := fun _ => α)
                    (colsA j)
                    (fun aij => s + vi * aij)))
            =
            (fun s i => s + f i) := by
        funext s i
        cases hv : valuesV i with
        | scalar vi =>
          cases hrow : rowsA i with
          | dim colsA =>
            cases hcol : colsA j with
            | scalar aij =>
              simp [f, hv, hrow, hcol]
      have hsum : (List.finRange m).foldl (fun s i => s + f i) 0 = ∑ i : Fin m, f i :=
        finRange_foldl_add_eq_finset_sum (f := f)
      rw [hfun, hsum]
      refine Finset.sum_congr rfl ?_
      intro i _
      cases hv : valuesV i with
      | scalar vi =>
        cases hrow : rowsA i with
        | dim colsA =>
          cases hcol : colsA j with
          | scalar aij =>
            simp [f, hv, hrow, hcol]

/--
Adjointness identity for matvec under `dot`.

Informally: `⟨dLdy, W dx⟩ = ⟨dLdy W, dx⟩`, i.e. the transpose-adjoint relationship between
matrix-vector and vector-matrix multiplication.

In PyTorch this corresponds to the familiar identity `(dLdyᵀ @ (W @ dx)) = ((dLdyᵀ @ W) @ dx)`,
written with explicit sums.
-/
theorem dot_mat_linear_adjoint
  {inDim outDim : Nat}
  (W : Tensor α (.dim outDim (.dim inDim .scalar)))
  (dLdy : Tensor α (.dim outDim .scalar))
  (dx : Tensor α (.dim inDim .scalar)) :
  dot (α := α) dLdy (matVecMulSpec W dx)
  = dot (α := α) (vecMatMulSpec dLdy W) dx := by
  classical
  calc
    dot (α := α) dLdy (matVecMulSpec W dx)
        = ∑ i : Fin outDim, (toVec dLdy i) * (toVec (matVecMulSpec W dx) i) := by
            simpa using (dot_vec_eq_sum (α := α) (a := dLdy) (b := matVecMulSpec W dx))
    _ = ∑ i : Fin outDim, (toVec dLdy i) * (∑ k : Fin inDim, (get2 W i k) * (toVec dx k)) := by
            refine Finset.sum_congr rfl ?_
            intro i _
            simp [toVec_mat_vec_mul_spec (α := α) (A := W) (v := dx) (i := i)]
    _ = ∑ i : Fin outDim, ∑ k : Fin inDim,
          (toVec dLdy i) * ((get2 W i k) * (toVec dx k)) := by
            refine Finset.sum_congr rfl ?_
            intro i _
            simpa using
              (Finset.mul_sum (s := (Finset.univ : Finset (Fin inDim)))
                (f := fun k : Fin inDim => (get2 W i k) * (toVec dx k))
                (a := toVec dLdy i))
    _ = ∑ k : Fin inDim, ∑ i : Fin outDim,
          (toVec dLdy i) * ((get2 W i k) * (toVec dx k)) := by
            simpa using
              (Finset.sum_comm
                (s := (Finset.univ : Finset (Fin outDim)))
                (t := (Finset.univ : Finset (Fin inDim)))
                (f := fun i k => (toVec dLdy i) * ((get2 W i k) * (toVec dx k))))
    _ = ∑ k : Fin inDim,
          (∑ i : Fin outDim, (toVec dLdy i) * (get2 W i k)) * (toVec dx k) := by
            refine Finset.sum_congr rfl ?_
            intro k _
            calc
              (∑ i : Fin outDim, (toVec dLdy i) * ((get2 W i k) * (toVec dx k)))
                  = ∑ i : Fin outDim, ((toVec dLdy i) * (get2 W i k)) * (toVec dx k) := by
                      refine Finset.sum_congr rfl ?_
                      intro i _
                      simp [mul_assoc]
              _ = (∑ i : Fin outDim, (toVec dLdy i) * (get2 W i k)) * (toVec dx k) := by
                      symm
                      simpa using
                        (Finset.sum_mul (s := (Finset.univ : Finset (Fin outDim)))
                          (f := fun i : Fin outDim => (toVec dLdy i) * (get2 W i k))
                          (a := toVec dx k))
    _ = ∑ k : Fin inDim, (toVec (vecMatMulSpec dLdy W) k) * (toVec dx k) := by
            refine Finset.sum_congr rfl ?_
            intro k _
            simp [toVec_vec_mat_mul_spec (α := α) (v := dLdy) (A := W) (j := k), mul_comm]
    _ = dot (α := α) (vecMatMulSpec dLdy W) dx := by
            symm
            simpa using (dot_vec_eq_sum (α := α) (a := vecMatMulSpec dLdy W) (b := dx))

end

end

end TensorAlgebra
end Proofs
