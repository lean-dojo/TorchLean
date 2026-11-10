/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import Mathlib.Data.Multiset.Fintype
public import TorchLean.NN.UniversalApproximation.StoneWeierstrass
public import TorchLean.NN.UniversalApproximation.Product
public import TorchLean.Tensor.Contract

/-!
# ReLU approximation on compact sets (nD)

For a compact set `K` of vectors in `Tensor ℝ [n]`, `relu_compact` proves
that every continuous `f : C(K, ℝ)` is a uniform limit on `K` of single-hidden-layer ReLU MLPs, in
the sense of `Approx`. Stone–Weierstrass makes the coordinate polynomials dense in `C(K, ℝ)`.
Approximability is closed under sums (hidden layers placed side by side) and scalar multiples (a
scaled output layer), so products of coordinates are the case to handle. Polarization writes a
product of `d` coordinates as a combination of `d`-th powers of linear forms, and each such power
is a ridge function, approximated by composing the 1D ReLU approximation of `u ↦ u^d` with the
linear form.

Other results:
- `polarization_prod`: the signed sum of `d`-th powers over `{±1}^d` isolates a product of `d`
  reals.
- `approx_of_mem_coordSubalg`: every coordinate polynomial (element of `coordSubalg`) on a
  compact set `K` is uniformly approximable by a 2-layer ReLU MLP.
- `relu_mul_coord_box`: for `M > 0`, the coordinate product `x_i * x_j`
  is uniformly approximable on the box `[-M, M]^n`.

Dependencies:
- `TorchLean.NN.UniversalApproximation` (constructive 1D ReLU
  approximation).
- `TorchLean.NN.UniversalApproximation.StoneWeierstrass` (Stone–Weierstrass density
  of coordinate polynomials on compact sets of tensor vectors).
- `TorchLean.NN.UniversalApproximation.Product` (the product of two inputs on a 2D box,
  used by `relu_mul_coord_box`).
- `TorchLean.NN.UniversalApproximation.TensorInput` (lifting 1D MLP constructions to
  `Tensor ℝ [n]`).
-/

@[expose] public section


namespace TorchLean.UniversalApproximation


/-! ## Network algebra shared by the closure properties -/

/-- Appending the hidden layers of two networks and adding their scalar outputs evaluates to the
sum of the two networks. -/
theorem mlpEval_append_add {n m k : Nat}
    (l1f : Linear ℝ n m) (l1g : Linear ℝ n k)
    (l2f : Linear ℝ m 1) (l2g : Linear ℝ k 1) (x : Tensor ℝ [n]) :
    mlpEval (n := n) (hidDim := m + k) (appendLinear (inDim := n) l1f l1g)
        (combineOutput (m := m) (n := k) (α := (1 : ℝ)) (β := (1 : ℝ)) (γ := 0) l2f l2g) x =
      mlpEval (n := n) (hidDim := m) l1f l2f x + mlpEval (n := n) (hidDim := k) l1g l2g x := by
  simpa using
    mlpEval_appendLinear (inDim := n) (m := m) (n := k) (l1a := l1f) (l1b := l1g)
      (l2a := l2f) (l2b := l2g) (α := (1 : ℝ)) (β := (1 : ℝ)) (γ := 0) (x := x)

/-- Scale the weights and bias of a scalar output layer by `c`. -/
noncomputable def scaleOutput {m : Nat} (c : ℝ) (l2 : Linear ℝ m 1) : Linear ℝ m 1 :=
  { weights := Tensor.matrix (m := 1) (n := m) (fun _ j => c * Tensor.get2 l2.weights 0 j)
    bias := Tensor.ofFn (n := 1) (fun _ => c * extractScalarOutput l2.bias) }

/-- The network output is affine in the output layer, so scaling that layer scales the output. -/
theorem mlpEval_scaleOutput {n m : Nat} (c : ℝ) (l1 : Linear ℝ n m) (l2 : Linear ℝ m 1)
    (x : Tensor ℝ [n]) :
    mlpEval (n := n) (hidDim := m) l1 (scaleOutput c l2) x =
      c * mlpEval (n := n) (hidDim := m) l1 l2 x := by
  classical
  rw [mlpEval_eq_bias_sum (l1 := l1) (l2 := scaleOutput c l2) (x := x),
    mlpEval_eq_bias_sum (l1 := l1) (l2 := l2) (x := x)]
  simp [scaleOutput, Tensor.matrix, Tensor.get2, extractScalarOutput, Tensor.ofFn,
    mul_add, Finset.mul_sum, mul_assoc]

/-! ## Uniform approximation of continuous maps on `K`

The predicate is stated for `C(K, ℝ)`, the space where Stone–Weierstrass gives density of the
coordinate polynomials. -/

/-- `Approx K f`: for every `ε > 0` there is a single-hidden-layer ReLU MLP whose output
`mlpEval l1 l2 x.1` is within `ε` of `f x` at every point `x` of `K`. -/
def Approx {n : Nat} (K : Set (Tensor ℝ [n])) (f : C(K, ℝ)) : Prop :=
  ∀ ε > 0, ∃ (hidDim : ℕ) (l1 : Linear ℝ n hidDim) (l2 : Linear ℝ hidDim 1),
    ∀ x : K, |f x - mlpEval (n := n) (hidDim := hidDim) l1 l2 x.1| < ε

namespace Approx

/-! ## Closure properties -/

/-- The zero continuous function is uniformly approximable on `K`. -/
theorem zero {n : Nat} (K : Set (Tensor ℝ [n])) :
    Approx (n := n) K (0 : C(K, ℝ)) := by
  intro ε hε
  refine ⟨2, affineIdLayer1 (n := n) (w := fun _ => (0 : ℝ)) (b := 0), affineIdLayer2, ?_⟩
  intro x
  have : mlpEval (n := n) (hidDim := 2)
        (affineIdLayer1 (n := n) (w := fun _ => (0 : ℝ)) (b := 0)) affineIdLayer2 x.1 = 0 := by
    simp [mlpEval_affineIdLayer, dotProduct]
  simpa [this] using hε

/-- If `f` and `g` are uniformly approximable on `K`, then so is `f + g`. -/
theorem add {n : Nat} {K : Set (Tensor ℝ [n])} {f g : C(K, ℝ)}
    (hf : Approx (n := n) K f) (hg : Approx (n := n) K g) :
    Approx (n := n) K (f + g) := by
  intro ε hε
  have hε2 : 0 < ε / 2 := half_pos hε
  rcases hf (ε / 2) hε2 with ⟨m, l1f, l2f, hf'⟩
  rcases hg (ε / 2) hε2 with ⟨k, l1g, l2g, hg'⟩
  refine ⟨m + k, appendLinear (inDim := n) l1f l1g,
    combineOutput (m := m) (n := k) (α := (1 : ℝ)) (β := (1 : ℝ)) (γ := 0) l2f l2g,
    fun x => ?_⟩
  calc |(f + g) x - mlpEval (n := n) (hidDim := m + k) (appendLinear (inDim := n) l1f l1g)
          (combineOutput (m := m) (n := k) (α := (1 : ℝ)) (β := (1 : ℝ)) (γ := 0) l2f l2g) x.1|
      = |(f x - mlpEval (n := n) (hidDim := m) l1f l2f x.1)
          + (g x - mlpEval (n := n) (hidDim := k) l1g l2g x.1)| := by
        rw [mlpEval_append_add, ContinuousMap.add_apply]
        congr 1
        ring
    _ ≤ |f x - mlpEval (n := n) (hidDim := m) l1f l2f x.1|
          + |g x - mlpEval (n := n) (hidDim := k) l1g l2g x.1| := abs_add_le _ _
    _ < ε := by linarith [hf' x, hg' x]

/-- If `f` is uniformly approximable on `K`, then so is the scalar multiple `c • f`. -/
theorem smul {n : Nat} {K : Set (Tensor ℝ [n])} (c : ℝ) {f : C(K, ℝ)}
    (hf : Approx (n := n) K f) :
    Approx (n := n) K (c • f) := by
  by_cases hc : c = 0
  · subst hc
    simpa using (zero (n := n) K)
  intro ε hε
  have hcabs : 0 < |c| := abs_pos.2 hc
  rcases hf (ε / |c|) (div_pos hε hcabs) with ⟨m, l1, l2, hf'⟩
  refine ⟨m, l1, scaleOutput c l2, fun x => ?_⟩
  have hcancel : |c| * (ε / |c|) = ε := by field_simp [hc, abs_ne_zero.2 hc]
  have : |c * f x - mlpEval (n := n) (hidDim := m) l1 (scaleOutput c l2) x.1| < ε :=
    calc |c * f x - mlpEval (n := n) (hidDim := m) l1 (scaleOutput c l2) x.1|
        = |c| * |f x - mlpEval (n := n) (hidDim := m) l1 l2 x.1| := by
          rw [mlpEval_scaleOutput, ← mul_sub, abs_mul]
      _ < |c| * (ε / |c|) := mul_lt_mul_of_pos_left (hf' x) hcabs
      _ = ε := hcancel
  simpa using this

/-- Finite sums preserve `Approx` (Finset-indexed). -/
theorem sum_finset {n : Nat} {K : Set (Tensor ℝ [n])}
    {ι : Type} (s : Finset ι) (f : ι → C(K, ℝ))
    (hf : ∀ i ∈ s, Approx (n := n) K (f i)) :
    Approx (n := n) K (∑ i ∈ s, f i) := by
  classical
  -- Induct on the finset while threading the hypothesis `hf`.
  revert hf
  induction s using Finset.induction_on with
  | empty =>
      intro _hf
      simpa using (zero (n := n) K)
  | @insert a s ha ih =>
      intro hf'
      have ha' : Approx (n := n) K (f a) := hf' a (by simp [ha])
      have hs' : ∀ i ∈ s, Approx (n := n) K (f i) := by
        intro i hi
        exact hf' i (by simp [hi])
      have ih' : Approx (n := n) K (∑ i ∈ s, f i) := ih hs'
      simpa [Finset.sum_insert ha, add_comm, add_left_comm, add_assoc] using
        (add (n := n) (K := K) (f := f a) (g := ∑ i ∈ s, f i) ha' ih')

/-- Finite sums preserve `Approx` (Fintype-indexed). -/
theorem sum_fintype {n : Nat} {K : Set (Tensor ℝ [n])}
    {ι : Type} [Fintype ι] (f : ι → C(K, ℝ)) (hf : ∀ i : ι, Approx (n := n) K (f i)) :
    Approx (n := n) K (∑ i : ι, f i) := by
  classical
  -- `∑ i, f i` is a `Finset.univ` sum.
  simpa using
    (sum_finset (n := n) (K := K) (s := (Finset.univ : Finset ι)) (f := f)
      (by intro i hi; simpa using hf i))

end Approx

-- ---------------------------------------------------------------------------
-- Stone–Weierstrass coordinate algebra = multivariate-polynomial evaluation range
-- ---------------------------------------------------------------------------

/--
The Stone–Weierstrass coordinate subalgebra is the range of `MvPolynomial.aeval coord`.

Every element of `coordSubalg` is therefore a real polynomial evaluated at the coordinate maps,
which is how `approx_of_mem_coordSubalg` reduces to monomials.
-/
theorem coordSubalg_eq_range_aeval {n : Nat} (K : Set (Tensor ℝ [n])) :
    coordSubalg (K := K) =
      (MvPolynomial.aeval (coord (K := K))).range := by
  simpa [coordSubalg] using
    (Algebra.adjoin_range_eq_range_aeval (R := ℝ) (f := coord (K := K)))

-- ---------------------------------------------------------------------------
-- Polarization identity for products (sum over {±1}^d picks out the full product)
-- ---------------------------------------------------------------------------

section Polarization

open scoped BigOperators

/-- Sign associated to a Boolean: `true ↦ +1`, `false ↦ -1`. -/
noncomputable def sgn (b : Bool) : ℝ := if b then (1 : ℝ) else (-1 : ℝ)

/-- Product of signs for an assignment `ε : Fin d → Bool`. -/
noncomputable def signedProd {d : Nat} (ε : Fin d → Bool) : ℝ :=
  ∏ i : Fin d, sgn (ε i)

/-- Signed linear form `∑ i, sgn (ε i) * u i`. -/
noncomputable def signedSum {d : Nat} (ε : Fin d → Bool) (u : Fin d → ℝ) : ℝ :=
  ∑ i : Fin d, sgn (ε i) * u i

/-- Closed form for `∑ b : Bool, (sgn b)^k`. -/
theorem sum_bool_sgn_pow (k : ℕ) : (∑ b : Bool, (sgn b) ^ k) = (1 : ℝ) + (-1 : ℝ) ^ k := by
  classical
  -- `Fintype.sum_bool` expands the sum over the two Bool values.
  simp [sgn]

/-- For even exponents, `∑ b : Bool, (sgn b)^k = 2`. -/
theorem sum_bool_sgn_pow_even (k : ℕ) (hk : Even k) : (∑ b : Bool, (sgn b) ^ k) = (2 : ℝ) := by
  rw [sum_bool_sgn_pow, hk.neg_one_pow]
  norm_num

/-- For odd exponents, `∑ b : Bool, (sgn b)^k = 0`. -/
theorem sum_bool_sgn_pow_odd (k : ℕ) (hk : Odd k) : (∑ b : Bool, (sgn b) ^ k) = (0 : ℝ) := by
  simpa [sum_bool_sgn_pow, hk.neg_one_pow] using (sum_bool_sgn_pow (k := k))

/-- The cardinality of the fiber `{ i | p i = j }` as a natural number. -/
noncomputable def fiberCount {d : Nat} (p : Fin d → Fin d) (j : Fin d) : ℕ :=
  (Finset.univ.filter (fun i : Fin d => p i = j)).card

/--
`∏ i, sgn (ε (p i))` is the product over `j` of `sgn (ε j) ^ fiberCount p j`, with the factors
grouped by the fibers of `p`.
-/
theorem prod_sgn_comp {d : Nat} (p : Fin d → Fin d) (ε : Fin d → Bool) :
    (Finset.univ.prod fun i : Fin d => sgn (ε (p i)))
      =
    Finset.univ.prod fun j : Fin d => (sgn (ε j)) ^ (fiberCount (d := d) p j) := by
  classical
  -- `Finset.prod_fiberwise'` groups the product over `i` by the value of `p i`:
  --   `∏ j, ∏ i ∈ univ with p i = j, (sgn (ε j)) = ∏ i, (sgn (ε (p i)))`.
  have hfib :
      (Finset.univ.prod fun j : Fin d =>
          ∏ i ∈ (Finset.univ : Finset (Fin d)) with p i = j, sgn (ε j))
        =
      (Finset.univ.prod fun i : Fin d => sgn (ε (p i))) := by
    simpa using
      (Finset.prod_fiberwise' (s := (Finset.univ : Finset (Fin d))) (g := p)
        (f := fun j : Fin d => sgn (ε j)))
  have hpow :
      (Finset.univ.prod fun j : Fin d =>
          ∏ i ∈ (Finset.univ : Finset (Fin d)) with p i = j, sgn (ε j))
        =
      (Finset.univ.prod fun j : Fin d => (sgn (ε j)) ^ (fiberCount (d := d) p j)) := by
    -- Each inner product is a constant product, hence a power by the fiber card.
    simp [fiberCount, Finset.prod_const]
  -- Combine.
  simpa [hpow, fiberCount] using hfib.symm

/--
Sign-cancellation coefficient of `p : Fin d → Fin d`: the sum over sign assignments `ε` of
`∏ i, sgn (ε i)` times `∏ i, sgn (ε (p i))`.

Expanding the polarization sum and swapping the two sums makes it the coefficient of
`∏ i, u (p i)`. It is `2^d` for bijective `p` and `0` otherwise
(`signCoeff_eq_ite_bijective`).
-/
noncomputable def signCoeff {d : Nat} (p : Fin d → Fin d) : ℝ :=
  ∑ ε : (Fin d → Bool),
    (Finset.univ.prod fun i : Fin d => sgn (ε i)) *
      (Finset.univ.prod fun i : Fin d => sgn (ε (p i)))

/-- Product-of-sums form: `signCoeff p` is `∏ j, ∑ b : Bool, (sgn b) ^ (fiberCount p j + 1)`. -/
theorem signCoeff_eq_prod_sum_pow {d : Nat} (p : Fin d → Fin d) :
    signCoeff (d := d) p
      =
    ∏ j : Fin d, (∑ b : Bool, (sgn b) ^ (fiberCount (d := d) p j + 1)) := by
  classical
  -- Rewrite the second product using fiber counts, then factor the sum over `ε` as a product over
  -- coordinates.
  have hrewrite :
      signCoeff (d := d) p
        =
      ∑ ε : (Fin d → Bool),
        (∏ j : Fin d, (sgn (ε j)) ^ (fiberCount (d := d) p j + 1)) := by
    -- expand `signCoeff`, rewrite the composed product, then combine powers.
    classical
    unfold signCoeff
    refine Finset.sum_congr rfl ?_
    intro ε hε
    have hcomp :
        (Finset.univ.prod fun i : Fin d => sgn (ε (p i)))
          =
        (Finset.univ.prod fun j : Fin d => (sgn (ε j)) ^ (fiberCount (d := d) p j)) := by
      simpa using (prod_sgn_comp (d := d) p ε)
    -- The extra factor `∏ j, sgn (ε j)` raises each exponent by one (`a * a^k = a^(k+1)`).
    simp [hcomp, fiberCount, Finset.prod_mul_distrib, pow_succ, mul_comm]
  -- `Fintype.prod_sum` (`∏ i, ∑ j, f i j = ∑ x, ∏ i, f i (x i)`), read right to left with
  -- `ι = Fin d` and `κ i = Bool`, factors the sum over all `ε : Fin d → Bool` into a product of
  -- sums.
  have hfactor :
      (∑ ε : (Fin d → Bool),
          (∏ j : Fin d, (sgn (ε j)) ^ (fiberCount (d := d) p j + 1)))
        =
      (∏ j : Fin d, (∑ b : Bool, (sgn b) ^ (fiberCount (d := d) p j + 1))) := by
    simpa using
      (Fintype.prod_sum (ι := Fin d) (κ := fun _ : Fin d => Bool)
        (f := fun j b => (sgn b) ^ (fiberCount (d := d) p j + 1))).symm
  -- Put it together.
  simp [hrewrite, hfactor]

/--
`signCoeff p` is `2^d` when every fiber of `p` has odd size, and `0` otherwise.
-/
theorem signCoeff_eq_ite_odd {d : Nat} (p : Fin d → Fin d) :
    signCoeff (d := d) p =
      if (∀ j : Fin d, Odd (fiberCount (d := d) p j)) then (2 : ℝ) ^ d else 0 := by
  classical
  -- Use the product-of-sums form.
  rw [signCoeff_eq_prod_sum_pow (d := d) p]
  by_cases hall : ∀ j : Fin d, Odd (fiberCount (d := d) p j)
  · -- Each factor is `2` since `fiberCount j + 1` is even.
    -- Reduce the goal `... = if ... then ... else ...` using `hall`.
    simp [hall]
    have hfac : ∀ j : Fin d,
        (∑ b : Bool, (sgn b) ^ (fiberCount (d := d) p j + 1)) = (2 : ℝ) := fun j =>
      sum_bool_sgn_pow_even (k := fiberCount (d := d) p j + 1) (hall j).add_one
    -- Rewrite the product using `hfac`, then compute the product of the constant `2`.
    have hprod :
        (Finset.univ.prod fun j : Fin d =>
            (sgn true ^ (fiberCount (d := d) p j + 1) + sgn false ^ (fiberCount (d := d) p j + 1)))
          =
        (Finset.univ.prod fun _j : Fin d => (2 : ℝ)) := by
      classical
      refine Finset.prod_congr rfl ?_
      intro j hj
      simpa [Fintype.sum_bool, add_comm, add_left_comm, add_assoc] using hfac j
    -- `∏ j, 2 = 2^d`.
    have hconst : (Finset.univ.prod fun _j : Fin d => (2 : ℝ)) = (2 : ℝ) ^ d := by
      simp [Finset.prod_const]
    -- Unfold the `Fintype` product to a `Finset.univ.prod` and finish.
    change (Finset.univ.prod fun j : Fin d =>
        (sgn true ^ (fiberCount (d := d) p j + 1) + sgn false ^ (fiberCount (d := d) p j + 1))) = (2
          : ℝ) ^ d
    simp [hprod, hconst]
  · -- Some fiber count is even, hence one factor is `0`, so the whole product is `0`.
    -- Reduce the goal `... = if ... then ... else ...` using `hall`.
    simp [hall]
    obtain ⟨j0, hj0⟩ := not_forall.1 hall
    have hfactor0 :
        (∑ b : Bool, (sgn b) ^ (fiberCount (d := d) p j0 + 1)) = (0 : ℝ) :=
      sum_bool_sgn_pow_odd (k := fiberCount (d := d) p j0 + 1)
        (Nat.not_odd_iff_even.1 hj0).add_one
    -- The product over `univ` is zero if any factor is zero.
    have : (Finset.univ.prod fun j : Fin d =>
          (sgn true ^ (fiberCount (d := d) p j + 1) + sgn false ^ (fiberCount (d := d) p j + 1))) =
            (0 : ℝ) := by
      classical
      apply Finset.prod_eq_zero (Finset.mem_univ j0)
      simpa [Fintype.sum_bool, add_comm, add_left_comm, add_assoc] using hfactor0
    -- Restate the `Fintype` product as a `Finset` product.
    change (Finset.univ.prod fun j : Fin d =>
        (sgn true ^ (fiberCount (d := d) p j + 1) + sgn false ^ (fiberCount (d := d) p j + 1))) = 0
    simpa using this

/--
For a function `p : Fin d → Fin d`, all fiber cardinalities are odd iff `p` is bijective.

Since `Fin d` is finite of size `d`, odd fibers force every fiber to have size `1`.
-/
theorem forall_odd_fiberCount_iff {d : Nat} (p : Fin d → Fin d) :
    (∀ j : Fin d, Odd (fiberCount (d := d) p j)) ↔ Function.Bijective p := by
  classical
  constructor
  · intro hall
    -- The fibers partition `Fin d`, so the `d` fiber sizes sum to `d`. Each is odd, hence at
    -- least `1`, which forces every fiber to be a singleton.
    have hsum : ∑ j : Fin d, fiberCount (d := d) p j = d := by
      simpa [fiberCount] using
        (Finset.card_eq_sum_card_fiberwise (f := p) (s := (Finset.univ : Finset (Fin d)))
          (t := (Finset.univ : Finset (Fin d))) (fun _ _ => Finset.mem_univ _)).symm
    have hpos : ∀ j ∈ (Finset.univ : Finset (Fin d)), 1 ≤ fiberCount (d := d) p j :=
      fun j _ => Nat.succ_le_of_lt (hall j).pos
    have hone : ∀ j : Fin d, fiberCount (d := d) p j = 1 := fun j =>
      ((Finset.sum_eq_sum_iff_of_le hpos).1 (by simp [hsum]) j (Finset.mem_univ j)).symm
    refine (Function.bijective_iff_existsUnique p).2 fun j => ?_
    obtain ⟨i, hi⟩ := Finset.card_eq_one.1 (hone j)
    obtain ⟨hi_mem, huniq⟩ := Finset.eq_singleton_iff_unique_mem.1 hi
    exact ⟨i, by simpa using hi_mem, fun k hk => huniq k (by simpa using hk)⟩
  · intro hb j
    rcases hb.2 j with ⟨i, rfl⟩
    have hset : (Finset.univ.filter fun k : Fin d => p k = p i) = ({i} : Finset (Fin d)) := by
      ext k
      simp [hb.1.eq_iff]
    have h1 : fiberCount (d := d) p (p i) = 1 := by
      simp [fiberCount, hset]
    simp [h1]

/-- `signCoeff p` is `2^d` when `p` is bijective, and `0` otherwise. -/
theorem signCoeff_eq_ite_bijective {d : Nat} (p : Fin d → Fin d) :
    signCoeff (d := d) p = if Function.Bijective p then (2 : ℝ) ^ d else 0 := by
  classical
  have h := signCoeff_eq_ite_odd (d := d) p
  by_cases hb : Function.Bijective p
  · -- reduce the goal with `hb`, then use the all-odd characterization.
    rw [ite_eq_left hb]
    have hall : ∀ j : Fin d, Odd (fiberCount (d := d) p j) :=
      (forall_odd_fiberCount_iff (d := d) p).2 hb
    simpa [hall] using h
  · rw [ite_eq_right hb]
    have hall : ¬ (∀ j : Fin d, Odd (fiberCount (d := d) p j)) := by
      intro hall
      exact hb ((forall_odd_fiberCount_iff (d := d) p).1 hall)
    simpa [hall] using h

/-! ## Polarization identity -/

/--
Polarization identity for products: summing `signedProd ε * (signedSum ε u) ^ d` over all sign
assignments `ε` gives `2^d * Nat.factorial d * ∏ i, u i`.

After expansion, the monomial `∏ i, u (p i)` carries the weight `signCoeff p`, which vanishes
unless `p` is a permutation, and each of the `Nat.factorial d` permutations contributes
`2^d * ∏ i, u i`.
-/
theorem polarization_prod {d : Nat} (u : Fin d → ℝ) :
    (∑ ε : (Fin d → Bool), (signedProd (d := d) ε) * (signedSum (d := d) ε u) ^ d)
      =
    (2 : ℝ) ^ d * (Nat.factorial d) * (∏ i : Fin d, u i) := by
  classical
  -- Expand the powers, swap the two sums, evaluate `signCoeff`, and count bijections.
  have hpow :
      ∀ ε : (Fin d → Bool),
        (signedSum (d := d) ε u) ^ d
          =
        ∑ p : (Fin d → Fin d), ∏ i : Fin d, (sgn (ε (p i)) * u (p i)) := by
    intro ε
    simpa [signedSum, mul_assoc, mul_left_comm, mul_comm] using
      (Fintype.sum_pow (ι := Fin d) (f := fun i : Fin d => sgn (ε i) * u i) d)

  have hswap :
      (∑ ε : (Fin d → Bool), (signedProd (d := d) ε) * (signedSum (d := d) ε u) ^ d)
        =
      ∑ p : (Fin d → Fin d),
        (∏ i : Fin d, u (p i)) * signCoeff (d := d) p := by
    -- Pure algebraic rearrangement. `signedSum` is left out of the simp set so that `hpow` can
    -- rewrite its powers.
    simp [hpow, signedProd, signCoeff, Finset.mul_sum,
      Finset.prod_mul_distrib, mul_left_comm, mul_comm]
    exact Finset.sum_comm

  rw [hswap]

  have hrewrite :
      (∑ p : (Fin d → Fin d), (∏ i : Fin d, u (p i)) * signCoeff (d := d) p)
        =
      ∑ p : (Fin d → Fin d),
        if Function.Bijective p then (2 : ℝ) ^ d * (∏ i : Fin d, u i) else 0 := by
    classical
    refine Fintype.sum_congr
      (fun p : (Fin d → Fin d) => (∏ i : Fin d, u (p i)) * signCoeff (d := d) p)
      (fun p : (Fin d → Fin d) =>
        if Function.Bijective p then (2 : ℝ) ^ d * (∏ i : Fin d, u i) else 0) (fun p => ?_)
    by_cases hb : Function.Bijective p
    · -- bijective case: both sides collapse to the same constant.
      have hprod : (∏ i : Fin d, u (p i)) = ∏ i : Fin d, u i := by
        simpa using (Function.Bijective.prod_comp (e := p) hb (g := u))
      -- left side
      rw [signCoeff_eq_ite_bijective (d := d) p, ite_eq_left hb]
      -- right side
      rw [ite_eq_left hb]
      -- rewrite the `u`-product and commute.
      simp [hprod, mul_comm]
    · -- non-bijective: `signCoeff = 0` and the RHS is `0`.
      rw [signCoeff_eq_ite_bijective (d := d) p, ite_eq_right hb]
      rw [ite_eq_right hb]
      simp

  rw [hrewrite]

  -- Count bijections: `|{p : Fin d → Fin d // Bijective p}| = d!`.
  have hcard_bij :
      Fintype.card {p : (Fin d → Fin d) // Function.Bijective p} = Nat.factorial d := by
    classical
    let e : ({p : (Fin d → Fin d) // Function.Bijective p} ≃ Equiv.Perm (Fin d)) :=
      { toFun := fun p => Equiv.ofBijective p.1 p.2
        invFun := fun σ => ⟨σ, σ.bijective⟩
        left_inv := by
          intro p
          ext i
          rfl
        right_inv := by
          intro σ
          ext i
          rfl }
    simpa using (Fintype.card_congr e).trans (by simpa using (Fintype.card_perm (α := Fin d)))

  have hind :
      (∑ p : (Fin d → Fin d), if Function.Bijective p then (1 : ℝ) else 0)
        = (Nat.factorial d : ℝ) := by
    classical
    have hsum :
        (∑ p : (Fin d → Fin d), if Function.Bijective p then (1 : ℝ) else 0)
          =
        ((Finset.univ.filter fun p : (Fin d → Fin d) => Function.Bijective p).card : ℝ) := by
      simp
    have hfilter :
        (Finset.univ.filter fun p : (Fin d → Fin d) => Function.Bijective p).card = Nat.factorial d
          := by
      rw [← Fintype.card_subtype]
      exact hcard_bij
    -- conclude by casting the card equality
    calc
      (∑ p : (Fin d → Fin d), if Function.Bijective p then (1 : ℝ) else 0)
          = ((Finset.univ.filter fun p : (Fin d → Fin d) => Function.Bijective p).card : ℝ) := hsum
      _ = (Nat.factorial d : ℝ) := by
          exact_mod_cast hfilter

  -- Factor out the constant and finish.
  have hfact :
      (∑ p : (Fin d → Fin d),
        if Function.Bijective p then (2 : ℝ) ^ d * (∏ i : Fin d, u i) else 0)
        =
      ((2 : ℝ) ^ d * (∏ i : Fin d, u i)) *
        (∑ p : (Fin d → Fin d), if Function.Bijective p then (1 : ℝ) else 0) := by
    classical
    have :
        (∑ p : (Fin d → Fin d),
          if Function.Bijective p then (2 : ℝ) ^ d * (∏ i : Fin d, u i) else 0)
          =
        ∑ p : (Fin d → Fin d),
          ((2 : ℝ) ^ d * (∏ i : Fin d, u i)) * (if Function.Bijective p then (1 : ℝ) else 0) := by
      refine Fintype.sum_congr
        (fun p : (Fin d → Fin d) =>
          if Function.Bijective p then (2 : ℝ) ^ d * (∏ i : Fin d, u i) else 0)
        (fun p : (Fin d → Fin d) =>
          ((2 : ℝ) ^ d * (∏ i : Fin d, u i)) * (if Function.Bijective p then (1 : ℝ) else 0))
        (fun p => ?_)
      by_cases hb : Function.Bijective p <;> simp
    rw [this]
    simpa using
      (Finset.mul_sum (a := ((2 : ℝ) ^ d * (∏ i : Fin d, u i)))
        (f := fun p : (Fin d → Fin d) => (if Function.Bijective p then (1 : ℝ) else 0))
        (s := (Finset.univ : Finset (Fin d → Fin d)))).symm

  calc
      (∑ p : (Fin d → Fin d),
          if Function.Bijective p then (2 : ℝ) ^ d * (∏ i : Fin d, u i) else 0)
          =
        ((2 : ℝ) ^ d * (∏ i : Fin d, u i)) *
          (∑ p : (Fin d → Fin d), if Function.Bijective p then (1 : ℝ) else 0) := hfact
    _ = (2 : ℝ) ^ d * (Nat.factorial d) * (∏ i : Fin d, u i) := by
      -- Normalize by rewriting first, then use commutativity and associativity.
      rw [hind]
      ring_nf

end Polarization

/-! ## Compact domains: boxes and linear forms -/

/--
Coordinate multiplication on the box `[-M,M]^n`: for `M > 0` and indices `i j : Fin n`, the map
`x ↦ x_i * x_j` is uniformly approximable on `boxN n M` by a single-hidden-layer ReLU MLP.

The network is the two-input product network of `relu_mul_box` with a
first layer that reads coordinates `i` and `j`.
-/
theorem relu_mul_coord_box
    {n : Nat} {M : ℝ} (hM : 0 < M) (i j : Fin n) :
    ∀ ε > 0, ∃ (hidDim : ℕ) (l1 : Linear ℝ n hidDim) (l2 : Linear ℝ hidDim 1),
      ∀ x ∈ boxN n M,
        |(Tensor.getScalar x i * Tensor.getScalar x j) -
          mlpEval (n := n) (hidDim := hidDim) l1 l2 x| < ε := by
  classical
  intro ε hε
  obtain ⟨hidDim, l1, l2, happ⟩ := relu_mul_box hM ε hε
  -- Feed coordinates `i,j` to the existing two-input network. The weights add when `i = j`.
  let l1Coord : Linear ℝ n hidDim :=
    { weights := Tensor.matrix fun r k =>
        Tensor.get2 l1.weights r 0 * (Pi.single i 1 : Fin n → ℝ) k +
          Tensor.get2 l1.weights r 1 * (Pi.single j 1 : Fin n → ℝ) k
      bias := l1.bias }
  refine ⟨hidDim, l1Coord, l2, ?_⟩
  intro x hx
  let y : Tensor ℝ [2] := Tensor.ofFn fun k =>
    if k = 0 then Tensor.getScalar x i else Tensor.getScalar x j
  have hy : y ∈ boxN 2 M := by
    change ∀ k : Fin 2, Tensor.getScalar y k ∈ Set.Icc (-M) M
    simpa [Fin.forall_fin_two, y, Tensor.ofFn] using And.intro (hx i) (hx j)
  have hlinear : Linear.forward l1Coord x = Linear.forward l1 y := by
    apply Tensor.ext_vector
    intro r
    simp only [Linear.forward, Tensor.getScalar_add,
      Tensor.getScalar_contract_matVec_eq_sum]
    simp [l1Coord, y, Tensor.matrix, Tensor.ofFn, Tensor.get2, Pi.single_apply,
      add_mul, Finset.sum_add_distrib, Fin.sum_univ_two]
  have heval : mlpEval l1Coord l2 x = mlpEval l1 l2 y := by
    simp only [mlpEval, MLP.forward_def, hlinear]
  rw [heval]
  simpa [mulFun, y, Tensor.ofFn] using happ y hy

-- ---------------------------------------------------------------------------
-- Next building block: approximating `u ↦ u^d` on bounded intervals
-- ---------------------------------------------------------------------------

/--
Uniform approximation of the power function `x ↦ x^d` on `[-R,R]`, for `R > 0`, by a 1D ReLU MLP.

The proof applies `relu_Icc` with the Lipschitz constant
`max (d * R ^ (d - 1)) 1`, which `abs_pow_sub_pow_le` gives on `[-R,R]`.
-/
theorem relu_pow_Icc {R : ℝ} (hR : 0 < R) (d : ℕ) :
    ∀ ε > 0, ∃ (hidDim : ℕ) (l1 : Linear ℝ 1 hidDim) (l2 : Linear ℝ hidDim 1),
      ∀ x ∈ Set.Icc (-R) R, |x ^ d - mlpEvalScalar hidDim l1 l2 x| < ε := by
  -- The 1D theorem needs a positive Lipschitz constant, so enlarge `d * R ^ (d - 1)` to at
  -- least `1`.
  have hLip : ∀ x ∈ Set.Icc (-R) R, ∀ y ∈ Set.Icc (-R) R,
      |x ^ d - y ^ d| ≤ max (d * R ^ (d - 1)) 1 * |x - y| := by
    intro x hx y hy
    have hpow : max |x| |y| ^ (d - 1) ≤ R ^ (d - 1) :=
      pow_le_pow_left₀ ((abs_nonneg x).trans (le_max_left _ _))
        (max_le (abs_le.2 ⟨hx.1, hx.2⟩) (abs_le.2 ⟨hy.1, hy.2⟩)) (d - 1)
    calc |x ^ d - y ^ d| ≤ |x - y| * d * max |x| |y| ^ (d - 1) := abs_pow_sub_pow_le x y d
      _ ≤ |x - y| * d * R ^ (d - 1) := mul_le_mul_of_nonneg_left hpow (by positivity)
      _ = (d * R ^ (d - 1)) * |x - y| := by ring
      _ ≤ max (d * R ^ (d - 1)) 1 * |x - y| :=
        mul_le_mul_of_nonneg_right (le_max_left _ _) (abs_nonneg _)
  exact relu_Icc (f := fun x : ℝ => x ^ d) (a := -R) (b := R)
    (L := max (d * R ^ (d - 1)) 1) (by linarith) (lt_of_lt_of_le zero_lt_one (le_max_right _ _))
    hLip

-- ---------------------------------------------------------------------------
-- Stone–Weierstrass → ReLU bridge (compact-set approximation)
-- ---------------------------------------------------------------------------

section ReLUStoneWeierstrassBridge

open scoped BigOperators
open ContinuousMap

variable {n : Nat} (K : Set (Tensor ℝ [n])) [CompactSpace K]

omit [CompactSpace K] in
/-- The linear form `x ↦ w ⋅ x` as a continuous map on the set `K`. -/
noncomputable def linFormC (w : Fin n → ℝ) : C(K, ℝ) :=
  ∑ i : Fin n, w i • coord (K := K) i

omit [CompactSpace K] in
/-- Evaluate `linFormC` as the dot product `w ⋅ x` on the underlying tensor vector. -/
theorem linFormC_apply (w : Fin n → ℝ) (x : K) :
    linFormC K w x = w ⬝ᵥ x.1.getScalar := by
  classical
  simp only [linFormC, coord, dotProduct, ContinuousMap.sum_apply,
    ContinuousMap.smul_apply, smul_eq_mul]
  apply Finset.sum_congr rfl
  intro i _
  change w i * Tensor.vectorEquiv n x.1 i = w i * Tensor.getScalar x.1 i
  rw [Tensor.vectorEquiv_apply]

/-- On compact `K`, the ridge function `x ↦ (w ⋅ x)^d` is uniformly approximable by a 2-layer ReLU
MLP: the 1D power approximation on `[-R,R]`, with `R = max 1 ‖linFormC K w‖`, lifted through `w`. -/
theorem approx_pow_linFormC (w : Fin n → ℝ) (d : ℕ) :
    Approx (n := n) K ((linFormC K w) ^ d) := by
  intro ε hε
  let R : ℝ := max 1 ‖linFormC K w‖
  have hR : 0 < R := lt_of_lt_of_le zero_lt_one (le_max_left 1 ‖linFormC K w‖)
  rcases relu_pow_Icc (R := R) hR d ε hε with ⟨hidDim, l1, l2, hpow⟩
  let l1' : Linear ℝ n hidDim := liftScalarLayer1 (n := n) l1 w 0
  refine ⟨hidDim, l1', l2, ?_⟩
  intro x
  have hxR : linFormC K w x ∈ Set.Icc (-R) R := by
    have habs : |linFormC K w x| ≤ ‖linFormC K w‖ := by
      simpa using (ContinuousMap.norm_coe_le_norm (linFormC K w) x)
    have habsR : |linFormC K w x| ≤ R := le_trans habs (le_max_right 1 ‖linFormC K w‖)
    exact (abs_le).1 habsR
  have hpowx :
      |(linFormC K w x) ^ d - mlpEvalScalar hidDim l1 l2 (linFormC K w x)| < ε :=
    hpow (linFormC K w x) hxR
  have hlift :
      mlpEval (n := n) (hidDim := hidDim) l1' l2 x.1
        =
      mlpEvalScalar hidDim l1 l2 (w ⬝ᵥ x.1.getScalar) := by
    simpa [l1'] using
      (mlpEval_liftScalarLayer (n := n) (hidDim := hidDim) l1 l2 w 0 x.1)
  have hdot : w ⬝ᵥ x.1.getScalar = linFormC K w x :=
    (linFormC_apply (K := K) (w := w) (x := x)).symm
  -- rewrite the approximator in terms of the lifted network.
  simpa [hlift, hdot] using hpowx

end ReLUStoneWeierstrassBridge

section ReLUStoneWeierstrassBridgeProducts

open scoped BigOperators
open ContinuousMap

variable {n : Nat} (K : Set (Tensor ℝ [n])) [CompactSpace K]

/-- Weight vector encoding a signed sum of selected coordinates `∑ i, sgn(ε i) * x_{idx i}`. -/
noncomputable def wSigned {d : Nat} (idx : Fin d → Fin n) (ε : Fin d → Bool) : Fin n → ℝ :=
  fun j : Fin n => ∑ i : Fin d, sgn (ε i) * (Pi.single (idx i) 1 : Fin n → ℝ) j

/-- `wSigned idx ε ⬝ᵥ x.getScalar` computes the signed sum of the selected coordinates of `x`. -/
theorem dot_wSigned_eq_signedSum {d : Nat} (idx : Fin d → Fin n) (ε : Fin d → Bool)
    (x : Tensor ℝ [n]) :
    wSigned (n := n) idx ε ⬝ᵥ x.getScalar =
      signedSum (d := d) ε (fun i : Fin d => Tensor.getScalar x (idx i)) := by
  classical
  -- Swap the two sums, then each basis vector picks out one coordinate.
  simp only [wSigned, signedSum, dotProduct, Finset.sum_mul, mul_assoc]
  rw [Finset.sum_comm]
  refine Fintype.sum_congr _ _ (fun i => ?_)
  rw [← Finset.mul_sum]
  exact congrArg (sgn (ε i) * ·) (single_one_dotProduct (idx i) x.getScalar)

/--
Uniform approximation of a coordinate-product monomial on a compact set: for `idx : Fin d → Fin n`,
the map `x ↦ ∏ i, x_{idx i}` (a product of coordinate maps on `K`) is uniformly approximable by a
2-layer ReLU MLP.

By `polarization_prod` the product is `1 / (2^d * Nat.factorial d)` times a signed sum of `d`-th
powers of linear forms, and each power is approximable by `approx_pow_linFormC`.
-/
theorem approx_coordProd_fin {d : Nat} (idx : Fin d → Fin n) :
    Approx (n := n) K (∏ i : Fin d, coord (K := K) (idx i)) := by
  classical
  -- Rewrite the product using the polarization identity, then approximate that RHS.
  let C : ℝ := (2 : ℝ) ^ d * (Nat.factorial d)
  have hCpos : 0 < C := by
    have hpow : 0 < (2 : ℝ) ^ d := by
      exact pow_pos (by norm_num : (0 : ℝ) < 2) d
    have hfac : 0 < (Nat.factorial d : ℝ) := by
      exact_mod_cast Nat.factorial_pos d
    simpa [C, mul_assoc] using mul_pos hpow hfac
  have hCne : C ≠ 0 := ne_of_gt hCpos

  -- Define the polarization RHS as a continuous map.
  let term : (Fin d → Bool) → C(K, ℝ) :=
    fun ε =>
      (signedProd (d := d) ε) • ((linFormC K (wSigned (n := n) idx ε)) ^ d)
  let rhs : C(K, ℝ) := (1 / C) • (∑ ε : (Fin d → Bool), term ε)

  have hrhs_eq :
      rhs = (∏ i : Fin d, coord (K := K) (idx i)) := by
    ext x
    -- reduce to a pointwise real identity and apply `polarization_prod`.
    have hpol := polarization_prod (d := d)
      (u := fun i : Fin d => Tensor.getScalar x.1 (idx i))
    -- rewrite the RHS evaluation into the polarization sum
    have hterm_eval :
        (∑ ε : (Fin d → Bool), term ε) x
          =
        ∑ ε : (Fin d → Bool),
          (signedProd (d := d) ε) *
            (signedSum (d := d) ε
              (fun i : Fin d => Tensor.getScalar x.1 (idx i))) ^ d := by
      -- evaluate `term` and rewrite the lifted linear form as the signed sum.
      simp [term, linFormC_apply (K := K), dot_wSigned_eq_signedSum (n := n) (idx := idx)]
    -- Cancel the constant `C` using `hpol`.
    have : (rhs x) = (∏ i : Fin d, coord (K := K) (idx i) x) := by
      -- unfold the scalings, use `hpol`, and simplify.
      have hpolC :
          (∑ ε : (Fin d → Bool),
              (signedProd (d := d) ε) *
                (signedSum (d := d) ε
                  (fun i : Fin d => Tensor.getScalar x.1 (idx i))) ^ d)
            = C * (∏ i : Fin d, Tensor.getScalar x.1 (idx i)) := by
        simpa [C, mul_assoc, mul_left_comm, mul_comm] using hpol
      -- compute the coordinate product pointwise
      have hprod :
          (∏ i : Fin d, coord (K := K) (idx i) x) =
            ∏ i : Fin d, Tensor.getScalar x.1 (idx i) := by
        apply Finset.prod_congr rfl
        intro i _
        exact Tensor.vectorEquiv_apply x.1 (idx i)
      -- Simplify `rhs x`, keeping `C` folded so that `simp` can cancel it with `hCne`.
      simp [rhs, ContinuousMap.smul_apply, hterm_eval, hprod, hpolC, hCne]
    simpa using this

  -- `rhs` is approximable by closure under sum/scalar-mul, and equality transfers it to the
  -- product.
  have happ_rhs : Approx (n := n) K rhs := by
    -- First approximate each `term ε`.
    have happ_term : ∀ ε : (Fin d → Bool), Approx (n := n) K (term ε) := by
      intro ε
      -- Approximate the `d`-th power of the linear form, then scale by `signedProd ε`.
      have hp : Approx (n := n) K ((linFormC K (wSigned (n := n) idx ε)) ^ d) :=
        approx_pow_linFormC (K := K) (w := wSigned (n := n) idx ε) d
      simpa [term] using (Approx.smul (n := n) (K := K) (c := signedProd (d := d) ε) (f := _) hp)
    -- sum the terms, then scale by `1/C`.
    have happ_sum : Approx (n := n) K (∑ ε : (Fin d → Bool), term ε) :=
      Approx.sum_fintype (n := n) (K := K) (f := term) happ_term
    simpa [rhs] using (Approx.smul (n := n) (K := K) (c := (1 / C)) (f := _) happ_sum)

  -- finish
  simpa [hrhs_eq] using happ_rhs

/--
Uniform approximation of a coordinate product over any finite index type `ι`, by reindexing
`approx_coordProd_fin` along `Fintype.equivFin ι`.
-/
theorem approx_coordProd {ι : Type} [Fintype ι] (idx : ι → Fin n) :
    Approx (n := n) K (∏ i : ι, coord (K := K) (idx i)) := by
  classical
  let d : Nat := Fintype.card ι
  let e : ι ≃ Fin d := Fintype.equivFin ι
  have happ :
      Approx (n := n) K (∏ j : Fin d, coord (K := K) (idx (e.symm j))) := by
    simpa using (approx_coordProd_fin (K := K) (n := n) (d := d) (idx := fun j => idx (e.symm j)))
  have hprod :
      (∏ i : ι, coord (K := K) (idx i))
        =
      (∏ j : Fin d, coord (K := K) (idx (e.symm j))) := by
    -- `Fintype.prod_equiv` is the cleanest way to reindex the product.
    refine Fintype.prod_equiv e
      (fun i : ι => coord (K := K) (idx i))
      (fun j : Fin d => coord (K := K) (idx (e.symm j))) ?_
    intro i
    simp
  simpa [hprod] using happ

end ReLUStoneWeierstrassBridgeProducts

-- ---------------------------------------------------------------------------
-- Polynomials in coordinates are approximable by 2-layer ReLU MLPs (compact-set, nD)
-- ---------------------------------------------------------------------------

section ReLUStoneWeierstrassBridgePolynomials

open scoped BigOperators
open ContinuousMap

variable {n : Nat} (K : Set (Tensor ℝ [n])) [CompactSpace K]

/--
The exponent-vector product `d.prod (fun a n => (g a)^n)` multiplies `g a` once for each
occurrence of `a` in `d.toMultiset`.
-/
theorem prod_pow_eq_prod_toMultiset {α β : Type} [DecidableEq α] [CommMonoid β]
    (d : α →₀ ℕ) (g : α → β) :
    (d.prod fun a n => (g a) ^ n) = ∏ x : d.toMultiset, g (x : α) := by
  classical
  -- Push `g` through `toMultiset` as a `mapDomain`, then read the product off exponentwise.
  rw [Finset.prod_eq_multiset_prod, Multiset.map_univ, Finsupp.toMultiset_map,
    Finsupp.prod_toMultiset,
    Finsupp.prod_mapDomain_index (fun _ => pow_zero _) (fun _ _ _ => pow_add _ _ _)]

/-- Uniform approximation for an evaluated coordinate monomial `aeval (monomial d r)` on `K`. -/
theorem approx_aeval_coord_monomial (d : (Fin n) →₀ ℕ) (r : ℝ) :
    Approx (n := n) K
      (MvPolynomial.aeval (coord (K := K)) (MvPolynomial.monomial d r)) := by
  classical
  -- Approximate the repeated-coordinate product coming from `d.toMultiset`.
  have hprod :
      Approx (n := n) K
        (∏ x : d.toMultiset, coord (K := K) (x : Fin n)) :=
    approx_coordProd (K := K) (n := n) (idx := fun x : d.toMultiset => (x : Fin n))
  -- Rewrite the monomial evaluation into a scalar multiple of the repeated-coordinate product.
  have hrewrite :
      (MvPolynomial.aeval (coord (K := K)) (MvPolynomial.monomial d r))
        =
      r • (∏ x : d.toMultiset, coord (K := K) (x : Fin n)) := by
    -- `aeval_monomial` gives `algebraMap r * d.prod (...)`. Rewrite the `Finsupp.prod` as a
    -- product over `d.toMultiset`, then turn the `algebraMap` factor into a scalar action.
    have hpow :
        d.prod (fun i k => (coord (K := K) i) ^ k)
          =
        ∏ x : d.toMultiset, coord (K := K) (x : Fin n) := by
      simpa using
        (prod_pow_eq_prod_toMultiset (d := d)
          (g := fun i : Fin n => coord (K := K) i))
    calc
      MvPolynomial.aeval (coord (K := K)) (MvPolynomial.monomial d r)
          =
        (algebraMap ℝ C(K, ℝ)) r *
          d.prod (fun i k => (coord (K := K) i) ^ k) := by
            simpa using (MvPolynomial.aeval_monomial (g := coord (K := K)) (d := d)
              (r := r))
      _ =
        (algebraMap ℝ C(K, ℝ)) r *
          (∏ x : d.toMultiset, coord (K := K) (x : Fin n)) := by
            simp [hpow]
      _ = r • (∏ x : d.toMultiset, coord (K := K) (x : Fin n)) := by
            simp [Algebra.smul_def]
  -- Finish via closure under scalar multiplication.
  simpa [hrewrite] using (Approx.smul (n := n) (K := K) (c := r) (f := _) hprod)

/-- Uniform approximation of a coordinate polynomial `aeval coord p` on a compact set `K`. -/
theorem approx_aeval_coord (p : MvPolynomial (Fin n) ℝ) :
    Approx (n := n) K (MvPolynomial.aeval (coord (K := K)) p) := by
  classical
  -- Expand `p = ∑ d ∈ p.support, monomial d (p.coeff d)` (`p.as_sum`) and approximate it
  -- monomial by monomial.
  have hdecomp :
      MvPolynomial.aeval (coord (K := K)) p
        =
      ∑ d ∈ p.support,
        MvPolynomial.aeval (coord (K := K)) (MvPolynomial.monomial d (p.coeff d))
          := by
    -- Rewrite `p` using `p.as_sum`, then push `aeval` through the finite sum.
    calc
      MvPolynomial.aeval (coord (K := K)) p
          =
        MvPolynomial.aeval (coord (K := K))
          (∑ d ∈ p.support, MvPolynomial.monomial d (p.coeff d)) := by
            simp
      _ =
        ∑ d ∈ p.support,
          MvPolynomial.aeval (coord (K := K)) (MvPolynomial.monomial d (p.coeff d))
            := by
            simpa using (MvPolynomial.aeval_sum (f := coord (K := K))
              (s := p.support) (φ := fun d => MvPolynomial.monomial d (p.coeff d)))
  -- Approximate each monomial term, then sum.
  have hterm :
      ∀ d ∈ p.support,
        Approx (n := n) K
          (MvPolynomial.aeval (coord (K := K)) (MvPolynomial.monomial d (p.coeff
            d))) := by
    intro d hd
    simpa using approx_aeval_coord_monomial (K := K) (n := n) d (p.coeff d)
  -- Use `Approx.sum_finset` on the finite support.
  have hsum :
      Approx (n := n) K
        (∑ d ∈ p.support,
          MvPolynomial.aeval (coord (K := K)) (MvPolynomial.monomial d (p.coeff
            d))) :=
    Approx.sum_finset (n := n) (K := K) (s := p.support)
      (f := fun d => MvPolynomial.aeval (coord (K := K)) (MvPolynomial.monomial d
        (p.coeff d))) hterm
  simpa [hdecomp] using hsum

end ReLUStoneWeierstrassBridgePolynomials

-- ---------------------------------------------------------------------------
-- Full compact-set nD approximation for 2-layer ReLU MLPs (via Stone–Weierstrass)
-- ---------------------------------------------------------------------------

section ReLUStoneWeierstrassBridgeFull

open scoped BigOperators
open ContinuousMap

variable {n : Nat} (K : Set (Tensor ℝ [n])) [CompactSpace K]

-- ---------------------------------------------------------------------------
-- Bridge theorem: coordinate Stone–Weierstrass subalgebra ⊆ ReLU uniform closure (as `Approx`)
-- ---------------------------------------------------------------------------

/--
Bridge lemma: on compact `K`, every element of the Stone–Weierstrass coordinate subalgebra
satisfies `Approx`.

By `coordSubalg_eq_range_aeval` such an element is `aeval coord p` for a real polynomial `p`, and
`approx_aeval_coord` approximates that.
-/
theorem approx_of_mem_coordSubalg {g : C(K, ℝ)}
    (hg : g ∈ coordSubalg (K := K)) :
    Approx (n := n) K g := by
  rw [coordSubalg_eq_range_aeval (K := K)] at hg
  obtain ⟨p, rfl⟩ := hg
  exact approx_aeval_coord (K := K) (n := n) p

/--
ReLU universal approximation on compact sets (nD).

For compact `K` and any continuous `f : C(K,ℝ)`, `f` is uniformly approximable on `K` by a
single-hidden-layer ReLU MLP, in the `Approx` sense.
-/
theorem relu_compact (f : C(K, ℝ)) :
    Approx (n := n) K f := by
  intro ε hε
  have hε2 : 0 < ε / 2 := half_pos hε
  -- Stone–Weierstrass gives a coordinate polynomial `g` within `ε / 2` of `f` in sup norm, and
  -- the bridge lemma gives a ReLU network within `ε / 2` of `g`. The two errors add pointwise.
  rcases exists_coordSubalg_near (K := K) f (ε / 2) hε2 with ⟨g, hg⟩
  rcases approx_of_mem_coordSubalg (K := K) g.property (ε / 2) hε2 with ⟨hidDim, l1, l2, hnet⟩
  refine ⟨hidDim, l1, l2, fun x => ?_⟩
  have hfg : |f x - (g : C(K, ℝ)) x| < ε / 2 := by
    have hle : |(g : C(K, ℝ)) x - f x| ≤ ‖(g : C(K, ℝ)) - f‖ := by
      simpa using ContinuousMap.norm_coe_le_norm ((g : C(K, ℝ)) - f) x
    rw [abs_sub_comm] at hle
    exact lt_of_le_of_lt hle hg
  calc |f x - mlpEval (n := n) (hidDim := hidDim) l1 l2 x.1|
      ≤ |f x - (g : C(K, ℝ)) x|
          + |(g : C(K, ℝ)) x - mlpEval (n := n) (hidDim := hidDim) l1 l2 x.1| :=
        abs_sub_le _ _ _
    _ < ε := by linarith [hnet x]

end ReLUStoneWeierstrassBridgeFull

end TorchLean.UniversalApproximation
