/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import Mathlib.Algebra.BigOperators.GroupWithZero.Action
public import Mathlib.Algebra.BigOperators.Ring.Finset
public import Mathlib.Algebra.BigOperators.Ring.List
public import Mathlib.Algebra.BigOperators.Ring.Multiset
public import Mathlib.Algebra.BigOperators.Ring.Nat
public import Mathlib.Algebra.Module.TransferInstance
public import Mathlib.Data.Fin.Basic
public import Mathlib.Data.List.FinRange
public import Mathlib.Data.Real.Basic
public import NN.Proofs.Tensor.Algebra
public import NN.Spec.Core.Context
public import NN.Spec.Core.Shape
public import NN.Spec.Core.Tensor
public import NN.Spec.Core.TensorOps
public import NN.Spec.Core.TensorReductionShape

/-!
# Real Tensor Proof Toolkit

This file is the `ℝ`-specialized proof-facing companion to the spec tensor layer.

The tensor proof folder has two layers:

- `NN.Proofs.Tensor.Algebra` is backend-generic and proves semiring facts about recursive tensor
  dot products and executable folds.
- this file works in `Spec` over `ℝ`, where calculus, norms, Frobenius products, and model-analysis
  lemmas live.

The statements are intentionally *PyTorch-shaped* where that helps readers:

- `flattenR` / `unflattenR` give a `Fin (Shape.size s) → ℝ` view of `Tensor ℝ s`.
- lemmas relate `toVec` views to `add_spec`, `scale_spec`, etc.

We re-export selected generic helpers from `NN.Proofs.Tensor.Algebra` into the `Spec.*` namespace so
downstream proof files can use one consistent tensor vocabulary (`Spec.toVec`, `Spec.ofVec`,
`Spec.finRange_foldl_add_eq_finset_sum`) through shared fold and vector lemmas.

## PyTorch correspondence / citations

- Flatten / reshape: `torch.flatten`, `torch.reshape`, and `Tensor.view`.
  https://pytorch.org/docs/stable/generated/torch.flatten.html
  https://pytorch.org/docs/stable/generated/torch.reshape.html
  https://pytorch.org/docs/stable/generated/torch.Tensor.view.html
- “numel”: `tensor.numel()` corresponds to `Shape.size`.
  https://pytorch.org/docs/stable/generated/torch.Tensor.numel.html
-/

@[expose] public section


namespace Spec

open Tensor
open scoped BigOperators

-- Re-export generic helpers (defined once in `Proofs.TensorAlgebra`) into `Spec.*`.
export Proofs.TensorAlgebra (toVec ofVec toVec_ofVec ofVec_toVec)
export Proofs.TensorAlgebra (finRange_foldl_add_eq_finset_sum foldl_add_distrib2
  foldl_matvec_scalar)

/-! ## Algebraic instances for small tensor shapes -/

/-- Additive commutative monoid structure on scalar-shaped real tensors, transported by equivalence.
-/
instance : AddCommMonoid (Tensor ℝ .scalar) :=
  Equiv.addCommMonoid (Tensor.scalarEquiv ℝ)

/-- Additive commutative monoid structure on 1D real tensors (transported via an equiv). -/
instance {n : Nat} : AddCommMonoid (Tensor ℝ (.dim n .scalar)) :=
  Equiv.addCommMonoid (Tensor.dimScalarEquiv n)

/-- Scalar tensors inherit an `ℝ`-module structure when their entries do, transported by equivalence.
-/
instance {α : Type} [AddCommMonoid α] [Module ℝ α] : Module ℝ (Tensor α .scalar) :=
  Equiv.module ℝ (Tensor.scalarEquiv α)

/-- `Tensor α (dim n scalar)` inherits an `ℝ`-module structure when `α` is an `ℝ`-module (via an
  equiv). -/
instance {α : Type} [AddCommMonoid α] [Module ℝ α] {n : Nat} : Module ℝ (Tensor α (.dim n .scalar))
  :=
  Equiv.module ℝ (Tensor.dimScalarEquiv n)

/-- Noncomputable `ℝ`-module instance on scalar real tensors (for calculus proofs). -/
noncomputable instance : Module ℝ (Tensor ℝ .scalar) :=
  Equiv.module ℝ (Tensor.scalarEquiv ℝ)

/-- Noncomputable `ℝ`-module instance on 1D real tensors (for calculus proofs). -/
noncomputable instance {n : Nat} : Module ℝ (Tensor ℝ (.dim n .scalar)) :=
  Equiv.module ℝ (Tensor.dimScalarEquiv n)

/-! ## 1D helpers -/

/-- `toVec` distributes over pointwise addition (`add_spec`). -/
lemma toVec_add_spec {n : Nat} (x y : Tensor ℝ (.dim n .scalar)) :
    toVec (addSpec x y) = fun i => toVec x i + toVec y i := by
  cases x with
  | dim vx =>
    cases y with
    | dim vy =>
      funext i
      cases hx : vx i
      cases hy : vy i
      simp [toVec, addSpec, map2Spec, hx, hy]

/-- `toVec` distributes over pointwise scaling (`scale_spec`). -/
lemma toVec_scale_spec {n : Nat} (x : Tensor ℝ (.dim n .scalar)) (c : ℝ) :
    toVec (scaleSpec x c) = fun i => toVec x i * c := by
  cases x with
  | dim vx =>
    funext i
    cases hx : vx i
    simp [toVec, scaleSpec, mapSpec, hx]

/--
Flatten a tensor of shape `s` into a 1D view `Fin (Shape.size s) → ℝ`.

This is the proof-facing counterpart of `Spec.Tensor.flatten_spec` specialized to `ℝ`. In PyTorch
terms it is the functional analogue of flattening a tensor and then indexing it linearly
(`torch.flatten`, `tensor.view(-1)`). See the spec file `NN/Spec/Core/TensorReductionShape.lean`
for the definitional flatten/unflatten interface.

Citations:
https://pytorch.org/docs/stable/generated/torch.flatten.html
https://pytorch.org/docs/stable/generated/torch.Tensor.view.html
-/
def flattenR {s : Shape} (x : Tensor ℝ s) : Fin (Shape.size s) → ℝ :=
  toVec (flattenSpec (α:=ℝ) x)

/--
Unflatten a 1D view `Fin (Shape.size s) → ℝ` back into a tensor of shape `s`.

This is the proof-facing counterpart of `Spec.Tensor.unflatten_spec` specialized to `ℝ`, and is
intended to round-trip with `flattenR` under the spec lemmas in
`NN/Spec/Core/TensorReductionShape.lean`.
-/
def unflattenR {s : Shape} (v : Fin (Shape.size s) → ℝ) : Tensor ℝ s :=
  unflattenSpec (α:=ℝ) s (ofVec v)

/-! ## Pointwise tensor algebra -/

/-- Elementwise multiplication is associative (`mul_spec` is pointwise `(*)`). -/
theorem mul_spec_assoc {s : Shape} (a b c : Tensor ℝ s) :
  mulSpec a (mulSpec b c) = mulSpec (mulSpec a b) c := by
  induction s with
  | scalar =>
      cases a
      cases b
      cases c
      simp [mulSpec, map2Spec, mul_assoc]
  | dim n s ih =>
      cases a with
      | dim fa =>
        cases b with
        | dim fb =>
          cases c with
          | dim fc =>
            simp [mulSpec, map2Spec]
            funext i
            simpa using ih (a := fa i) (b := fb i) (c := fc i)

/-- Elementwise multiplication is commutative (`mul_spec` is pointwise `(*)`). -/
theorem mul_spec_comm {s : Shape}
  (a b : Tensor ℝ s) : mulSpec a b = mulSpec b a := by
  induction s with
  | scalar =>
      cases a
      cases b
      simp [mulSpec, map2Spec, mul_comm]
  | dim n s ih =>
      cases a with
      | dim fa =>
        cases b with
        | dim fb =>
          simp [mulSpec, map2Spec]
          funext i
          simpa using ih (a := fa i) (b := fb i)

/-- Elementwise addition is commutative (`add_spec` is pointwise `(+)`). -/
theorem add_spec_comm {s : Shape}
  (a b : Tensor ℝ s) : addSpec a b = addSpec b a := by
  induction s with
  | scalar =>
      cases a
      cases b
      simp [addSpec, map2Spec, add_comm]
  | dim n s ih =>
      cases a with
      | dim fa =>
        cases b with
        | dim fb =>
          simp [addSpec, map2Spec]
          funext i
          simpa using ih (a := fa i) (b := fb i)

/-- Elementwise multiplication of two all-zero tensors is the all-zero tensor. -/
theorem mul_spec_fill_zero {s : Shape} :
    mulSpec (fill (0 : ℝ) s) (fill (0 : ℝ) s) = fill (0 : ℝ) s := by
  induction s with
  | scalar =>
    simp [mulSpec, map2Spec, fill]
  | dim n s ih =>
    simp [mulSpec, map2Spec, fill]
    funext i
    simpa using ih

/-! ## Real dot product and fold bridges -/

/--
Real dot product for same-shape tensors, defined as the sum of elementwise products.

This matches the common PyTorch idiom `(a * b).sum()` for same-shape tensors.
Citations:
https://pytorch.org/docs/stable/generated/torch.sum.html

This is the `Spec`-namespace dot product used by real-analysis proofs. The backend-generic
recursive dot product is `Proofs.TensorAlgebra.dot`.
-/
noncomputable def dot {s : Shape} (a b : Tensor ℝ s) : ℝ :=
  sumSpec (mulSpec a b)

/--
One-step unfolding of the internal tail-recursive helper `tensor_foldl_spec.go` when the loop
condition holds (`k < n`).

This lemma is a proof tool: it lets proofs *peel one loop step* without using `unfold` directly.
-/
lemma tensor_foldl_spec_go_of_lt {α β : Type} (f : β → α → β)
    {n : Nat} {s : Shape} (values : Fin n → Tensor α s) {k : Nat} (acc : β) (hk : k < n) :
    tensorFoldlSpec.go f n s values k acc =
      tensorFoldlSpec.go f n s values (k + 1) (tensorFoldlSpec f acc (values ⟨k, hk⟩)) := by
  -- Use the definitional equation once, but prevent `simp` from unfolding `go` recursively.
  rw [tensorFoldlSpec.go.eq_1]
  simp [hk]

/--
One-step unfolding of the internal tail-recursive helper `tensor_foldl_spec.go` when the loop
condition fails (`¬ k < n`), i.e. the loop terminates and returns the accumulator.
-/
lemma tensor_foldl_spec_go_of_not_lt {α β : Type} (f : β → α → β)
    {n : Nat} {s : Shape} (values : Fin n → Tensor α s) {k : Nat} (acc : β) (hk : ¬ k < n) :
    tensorFoldlSpec.go f n s values k acc = acc := by
  rw [tensorFoldlSpec.go.eq_1]
  simp [hk]

/--
Accumulator lemma for `tensor_foldl_spec` specialized to addition.

Informally: folding with `(+)` over a tensor adds `sum_spec t` to the initial accumulator.
This is frequently used to move between “fold-style” specs and “sum-style” algebra.
-/
lemma tensor_foldl_spec_add_init {s : Shape} (acc : ℝ) (t : Tensor ℝ s) :
    tensorFoldlSpec (· + ·) acc t = acc + sumSpec t := by
  induction s generalizing acc with
  | scalar =>
    cases t with
    | scalar x =>
      simp [tensorFoldlSpec, sumSpec]
  | dim n s ih =>
    cases t with
    | dim values =>
      -- `tensor_foldl_spec` uses an internal tail-recursive `go`; prove the accumulator lemma for
      -- it.
      -- We show: `go k acc = acc + go k 0` by induction on `n - k`.
      have go_add : ∀ k acc, k ≤ n →
          tensorFoldlSpec.go (· + ·) n s values k acc =
            acc + tensorFoldlSpec.go (· + ·) n s values k 0 := by
        intro k acc hk
        induction hn : n - k generalizing k acc with
        | zero =>
          have hk' : k = n := by
            exact Nat.le_antisymm hk (Nat.sub_eq_zero_iff_le.mp hn)
          subst k
          -- Since `k = n`, the loop condition is false and `go` returns the accumulator.
          have hgo_acc :
              tensorFoldlSpec.go (· + ·) n s values n acc = acc := by
            simpa using
              (tensor_foldl_spec_go_of_not_lt (f := (· + ·)) (values := values) (k := n) (acc := acc)
                (by simp))
          have hgo_0 :
              tensorFoldlSpec.go (· + ·) n s values n (0 : ℝ) = (0 : ℝ) := by
            simpa using
              (tensor_foldl_spec_go_of_not_lt (f := (· + ·)) (values := values) (k := n)
                (acc := (0 : ℝ)) (by simp))
          simp [hgo_acc, hgo_0]
        | succ m ih_go =>
          have hlt : k < n := by
            exact Nat.sub_pos_iff_lt.mp (by simp [hn])
          have hk1 : k + 1 ≤ n := Nat.succ_le_of_lt hlt
          -- Peel one `go` step at index `k` on both sides.
          rw [tensor_foldl_spec_go_of_lt (f := (· + ·)) (values := values) (k := k) (acc := acc) hlt]
          rw [tensor_foldl_spec_go_of_lt (f := (· + ·)) (values := values) (k := k)
            (acc := (0 : ℝ)) hlt]
          -- Apply the induction hypothesis for `k+1` at the updated accumulators.
          have h_next : n - (k + 1) = m := by
            simp [Nat.sub_succ, hn]
          -- IH on the sub-tensor: `fold acc = acc + sum`
          have h_step :
              tensorFoldlSpec (· + ·) acc (values ⟨k, hlt⟩) =
                acc + sumSpec (values ⟨k, hlt⟩) := ih acc (values ⟨k, hlt⟩)
          have h_step0 :
              tensorFoldlSpec (· + ·) 0 (values ⟨k, hlt⟩) =
                0 + sumSpec (values ⟨k, hlt⟩) := ih 0 (values ⟨k, hlt⟩)
          -- Use IH on `go` for the (k+1)-suffix.
          have hgo_acc :
              tensorFoldlSpec.go (· + ·) n s values (k + 1)
                  (tensorFoldlSpec (· + ·) acc (values ⟨k, hlt⟩))
                =
              tensorFoldlSpec (· + ·) acc (values ⟨k, hlt⟩)
                + tensorFoldlSpec.go (· + ·) n s values (k + 1) 0 := by
            -- `ih_go` is stated for the successor case `m = n - k - 1`
            -- so rewrite `n - (k+1)` to `m`.
            have := ih_go (k := k + 1) (acc := tensorFoldlSpec (· + ·) acc (values ⟨k, hlt⟩)) hk1
            simpa [h_next] using this
          have hgo_0 :
              tensorFoldlSpec.go (· + ·) n s values (k + 1)
                  (tensorFoldlSpec (· + ·) 0 (values ⟨k, hlt⟩))
                =
              tensorFoldlSpec (· + ·) 0 (values ⟨k, hlt⟩)
                + tensorFoldlSpec.go (· + ·) n s values (k + 1) 0 := by
            have := ih_go (k := k + 1) (acc := tensorFoldlSpec (· + ·) 0 (values ⟨k, hlt⟩)) hk1
            simpa [h_next] using this
          -- Put everything together.
          -- Goal is:
          --   go (k+1) (fold acc t_k) = acc + go (k+1) (fold 0 t_k)
          -- Rewrite both sides via the two `hgo_*` lemmas, then use `h_step`/`h_step0`.
          -- Note: `0 + x = x`, and rearrange via associativity/commutativity.
          calc
            tensorFoldlSpec.go (· + ·) n s values (k + 1)
                (tensorFoldlSpec (· + ·) acc (values ⟨k, hlt⟩))
                =
              tensorFoldlSpec (· + ·) acc (values ⟨k, hlt⟩)
                + tensorFoldlSpec.go (· + ·) n s values (k + 1) 0 := hgo_acc
            _ =
              (acc + sumSpec (values ⟨k, hlt⟩))
                + tensorFoldlSpec.go (· + ·) n s values (k + 1) 0 := by
                  simp [h_step]
            _ =
              acc +
                ((0 + sumSpec (values ⟨k, hlt⟩))
                  + tensorFoldlSpec.go (· + ·) n s values (k + 1) 0) := by
                  ring
            _ =
              acc +
                (tensorFoldlSpec (· + ·) 0 (values ⟨k, hlt⟩)
                  + tensorFoldlSpec.go (· + ·) n s values (k + 1) 0) := by
                  simp [h_step0]
            _ =
              acc +
                tensorFoldlSpec.go (· + ·) n s values (k + 1)
                  (tensorFoldlSpec (· + ·) 0 (values ⟨k, hlt⟩)) := by
                  -- use `hgo_0` backwards and rearrange
                  simp [hgo_0]
      -- Use `go_add` at k=0
      have h0 := go_add (k := 0) (acc := acc) (by exact Nat.zero_le n)
      -- finish: unfold `tensor_foldl_spec`/`sum_spec` for the outer dimension
      simpa [tensorFoldlSpec, sumSpec] using h0


-- Rewriting lemma under dot using associativity/commutativity
/-- Reassociate a `dot` over a pointwise product, using commutativity/associativity of `mul_spec`.
  -/
theorem dot_mul_reassoc {s : Shape}
  (dLdy m dx : Tensor ℝ s) :
  dot dLdy (mulSpec m dx) = dot (mulSpec m dLdy) dx := by
  have hAssoc := mul_spec_assoc (a := dLdy) (b := m) (c := dx)
  have hComm := mul_spec_comm (a := dLdy) (b := m)
  -- `mul_spec dLdy (mul_spec m dx) = mul_spec (mul_spec dLdy m) dx`
  -- and `mul_spec (mul_spec dLdy m) dx = mul_spec (mul_spec m dLdy) dx`.
  simp [dot, hAssoc, hComm]

/-- Unfolding lemma for `get2` (2D tensor indexing). -/
lemma get2_eq {α : Type} {m n : Nat} (A : Tensor α (.dim m (.dim n .scalar))) (i : Fin m) (j : Fin
  n) :
  get2 A i j =
    match get A i with
    | Tensor.dim row => match row j with
      | Tensor.scalar v => v := by
  -- unfold get2 definition explicitly so both sides match
  unfold get2
  rfl


/-- Unfolding lemma for `get` on a `Tensor.dim` value. -/
lemma get_eq {α : Type} {n s} (t : Tensor α (.dim n s)) (i : Fin n) :
  get t i = match t with
  | Tensor.dim f => f i := by
  unfold get
  rfl

/--
Coordinate formula for `mat_vec_mul_spec`, converted from the spec's `List.finRange` fold to a
`Finset.univ.sum`.

This is the “PyTorch-looking” statement of matvec: each output entry is a dot product of the
corresponding row with the input vector.
-/
lemma toVec_mat_vec_mul_spec {m n : Nat}
  (A : Tensor ℝ (.dim m (.dim n .scalar)))
  (v : Tensor ℝ (.dim n .scalar)) (i : Fin m) :
  toVec (matVecMulSpec A v) i = ∑ k : Fin n, (get2 A i k) * (toVec v k) := by
  -- Reuse the backend-generic lemma from `NN/Proofs/Tensor/Algebra.lean` (instantiated at `ℝ`).
  simpa using
    (Proofs.TensorAlgebra.toVec_mat_vec_mul_spec (α := ℝ) (A := A) (v := v) (i := i))

/--
Coordinate formula for `mat_mul_spec` (matrix-matrix multiplication).

This is the standard triple-sum identity: `(A @ B)[i,j] = ∑ k, A[i,k] * B[k,j]`, matching the
textbook/PyTorch view of matrix multiplication.

Citations:
https://pytorch.org/docs/stable/generated/torch.matmul.html
-/
lemma get2_mat_mul_spec {m n p : Nat}
  (A : Tensor ℝ (.dim m (.dim n .scalar)))
  (B : Tensor ℝ (.dim n (.dim p .scalar))) (i : Fin m) (j : Fin p) :
  get2 (matMulSpec A B) i j = ∑ k : Fin n, (get2 A i k) * (get2 B k j) := by
  classical
  cases A with
  | dim rowsA =>
    cases B with
    | dim rowsB =>
      -- Unfold the matrix multiplication at `(i,j)` and convert the `finRange` fold to a
      -- `Finset.sum`.
      simp [matMulSpec, get2_eq, get_eq]
      -- Put the fold into the canonical `s + f k` form to apply `finRange_foldl_add_eq_finset_sum`.
      let f : Fin n → ℝ := fun k =>
        matMulSpec.match_3 (motive := fun _ _ => ℝ) (rowsA i) (rowsB k) (fun colsA colsB =>
          matMulSpec.match_1 (motive := fun _ _ => ℝ) (colsA k) (colsB j) (fun a b => a * b))
      have hfun :
          (fun (s : ℝ) (k : Fin n) =>
              matMulSpec.match_3 (motive := fun _ _ => ℝ) (rowsA i) (rowsB k) (fun colsA colsB =>
                matMulSpec.match_1 (motive := fun _ _ => ℝ) (colsA k) (colsB j) (fun a b => s + a
                  * b)))
            =
            (fun s k => s + f k) := by
        funext s k
        cases hrowA : rowsA i with
        | dim colsA =>
          cases hrowB : rowsB k with
          | dim colsB =>
            cases hA : colsA k with
            | scalar a =>
              cases hB : colsB j with
              | scalar b =>
                simp [f, hrowA, hrowB, hA, hB]
      have hsum : (List.finRange n).foldl (fun s k => s + f k) 0 = ∑ k : Fin n, f k :=
        finRange_foldl_add_eq_finset_sum (f := f)
      rw [hfun, hsum]
      refine Finset.sum_congr rfl ?_
      intro k _
      cases hrowA : rowsA i with
      | dim colsA =>
        cases hrowB : rowsB k with
        | dim colsB =>
          cases hA : colsA k with
          | scalar a =>
            cases hB : colsB j with
            | scalar b =>
              simp [f, hrowA, hrowB, hA, hB]

/--
Sum over the outer dimension unfolds into a `Finset.univ` sum of inner `sum_spec`.

This is the tensor analogue of `torch.sum` reducing over a leading dimension.
-/
lemma sum_spec_dim {n : Nat} {s : Shape} (t : Tensor ℝ (.dim n s)) :
  sumSpec t = ∑ i : Fin n, sumSpec (get t i) := by
  classical
  cases t with
  | dim values =>
      let f : Fin n → ℝ := fun i => sumSpec (values i)
      have go_eq :
          ∀ k acc, k ≤ n →
            tensorFoldlSpec.go (· + ·) n s values k acc =
              acc + (Finset.univ.filter (fun i : Fin n => k ≤ i.val)).sum f := by
        intro k acc hk
        induction hn : n - k generalizing k acc with
        | zero =>
            have hk' : k = n := by
              exact Nat.le_antisymm hk (Nat.sub_eq_zero_iff_le.mp hn)
            subst k
            have hgo :
                tensorFoldlSpec.go (· + ·) n s values n acc = acc := by
              simpa using
                (tensor_foldl_spec_go_of_not_lt (f := (· + ·)) (values := values) (k := n) (acc := acc)
                  (by simp))
            simp [hgo]
            have hfilter : (Finset.univ.filter (fun i : Fin n => n ≤ i.val)) = (∅ : Finset (Fin n))
              := by
              ext i
              simp [Nat.not_le_of_lt i.isLt]
            simp [hfilter]
        | succ m ih =>
            have hlt : k < n := by
              exact Nat.sub_pos_iff_lt.mp (by simp [hn])
            have hk1 : k + 1 ≤ n := Nat.succ_le_of_lt hlt
            rw [tensor_foldl_spec_go_of_lt (f := (· + ·)) (values := values) (k := k) (acc := acc) hlt]
            have hstep :
                tensorFoldlSpec (· + ·) acc (values ⟨k, hlt⟩) = acc + f ⟨k, hlt⟩ := by
              simpa [f] using
                (tensor_foldl_spec_add_init (s := s) (acc := acc) (t := values ⟨k, hlt⟩))
            have h_next : n - (k + 1) = m := by
              simp [Nat.sub_succ, hn]
            have ih' :=
              (ih (k := k + 1) (acc := acc + f ⟨k, hlt⟩) hk1)
            have ih'' :
                tensorFoldlSpec.go (· + ·) n s values (k + 1) (acc + f ⟨k, hlt⟩) =
                  (acc + f ⟨k, hlt⟩) +
                    (Finset.univ.filter (fun i : Fin n => k + 1 ≤ i.val)).sum f := by
              simpa [h_next] using ih'
            let Sk : Finset (Fin n) := Finset.univ.filter (fun i : Fin n => k ≤ i.val)
            let Sk1 : Finset (Fin n) := Finset.univ.filter (fun i : Fin n => k + 1 ≤ i.val)
            have hSk : Sk = insert (⟨k, hlt⟩ : Fin n) Sk1 := by
              ext i
              constructor
              · intro hiSk
                have hle : k ≤ i.val := by
                  simpa [Sk] using hiSk
                have hcase : k = i.val ∨ k < i.val := Nat.eq_or_lt_of_le hle
                refine (Finset.mem_insert).2 ?_
                cases hcase with
                | inl hEq =>
                    left
                    apply Fin.ext
                    exact hEq.symm
                | inr hLt =>
                    right
                    have hk1' : k + 1 ≤ i.val := Nat.succ_le_of_lt hLt
                    simpa [Sk1] using hk1'
              · intro hiIns
                have hi' : i = (⟨k, hlt⟩ : Fin n) ∨ i ∈ Sk1 := (Finset.mem_insert).1 hiIns
                cases hi' with
                | inl hEq =>
                    subst hEq
                    simp [Sk]
                  | inr hiSk1 =>
                      have hk1' : k + 1 ≤ i.val := by
                        simpa [Sk1] using hiSk1
                      have hle : k ≤ i.val := Nat.le_trans (Nat.le_succ k) hk1'
                      simpa [Sk] using hle
            have hk_not_mem1 : (⟨k, hlt⟩ : Fin n) ∉ Sk1 := by
              simp [Sk1]
            have hSk_sum : Sk.sum f = f ⟨k, hlt⟩ + Sk1.sum f := by
              have :
                  (insert (⟨k, hlt⟩ : Fin n) Sk1).sum f = f ⟨k, hlt⟩ + Sk1.sum f := by
                simpa using
                  (Finset.sum_insert (s := Sk1) (a := (⟨k, hlt⟩ : Fin n)) (f := f) hk_not_mem1)
              simpa [hSk] using this
            calc
              tensorFoldlSpec.go (· + ·) n s values (k + 1)
                  (tensorFoldlSpec (· + ·) acc (values ⟨k, hlt⟩))
                  =
                tensorFoldlSpec.go (· + ·) n s values (k + 1) (acc + f ⟨k, hlt⟩) := by
                  simp [hstep]
              _ = (acc + f ⟨k, hlt⟩) + Sk1.sum f := ih''
              _ = acc + (f ⟨k, hlt⟩ + Sk1.sum f) := by ring
              _ = acc + Sk.sum f := by
                    simp [hSk_sum, Sk]
      have hfilter0 :
          (Finset.univ.filter (fun i : Fin n => (0 : Nat) ≤ i.val)) = (Finset.univ : Finset (Fin n))
            := by
        ext i
        simp
      have h0 := go_eq (k := 0) (acc := (0 : ℝ)) (Nat.zero_le n)
      -- turn `Finset.univ.sum f` into the claimed RHS using `get_eq`
      have hget : ∀ i : Fin n, get (Tensor.dim values) i = values i := by
        intro i
        simp [get_eq]
      calc
        sumSpec (Tensor.dim values)
            = tensorFoldlSpec.go (· + ·) n s values 0 0 := by
                simp [sumSpec, tensorFoldlSpec]
        _ = (0 : ℝ) + (Finset.univ.filter (fun i : Fin n => (0 : Nat) ≤ i.val)).sum f := by
              simpa using h0
        _ = ∑ i : Fin n, sumSpec (get (Tensor.dim values) i) := by
              simp [f, hget]

-- Sum of a vector as a `Finset` sum of its entries.
/-- `sum_spec` on a 1D tensor equals the `Finset` sum of its coordinates (`toVec`). -/
lemma sum_spec_vec {n : Nat} (v : Tensor ℝ (.dim n .scalar)) :
  sumSpec v = ∑ i : Fin n, toVec v i := by
  classical
  cases v with
  | dim values =>
      -- `sum_spec_dim` reduces a vector sum to a sum of scalar `sum_spec`.
      have h :=
        (sum_spec_dim (t := (Tensor.dim values : Tensor ℝ (.dim n .scalar))) (s := .scalar))
      -- Turn each scalar `sum_spec` into the corresponding coordinate.
      refine h.trans ?_
      refine Finset.sum_congr rfl ?_
      intro i _
      cases hval : values i with
      | scalar x =>
          simp [get_eq, toVec, sumSpec, tensorFoldlSpec, hval]

-- Pointwise product of vectors under `toVec`.
/-- `toVec` of `mul_spec` is pointwise multiplication of coordinate functions. -/
lemma toVec_mul_spec {n : Nat} (a b : Tensor ℝ (.dim n .scalar)) (i : Fin n) :
  toVec (mulSpec a b) i = toVec a i * toVec b i := by
  cases a with
  | dim fa =>
    cases b with
    | dim fb =>
      cases ha : fa i with
      | scalar x =>
        cases hb : fb i with
        | scalar y =>
          simp [mulSpec, map2Spec, toVec, ha, hb]

-- Dot product of vectors as a `Finset` sum over coordinates.
/-- Dot product of vectors is the coordinate-wise sum `∑ i, a[i] * b[i]`. -/
lemma dot_vec_eq_sum {n : Nat} (a b : Tensor ℝ (.dim n .scalar)) :
  dot a b = ∑ i : Fin n, toVec a i * toVec b i := by
  simp [dot]
  have hsum := sum_spec_vec (v := mulSpec a b)
  calc
    sumSpec (mulSpec a b) = ∑ i : Fin n, toVec (mulSpec a b) i := by
      simpa using hsum
    _ = ∑ i : Fin n, toVec a i * toVec b i := by
      refine Finset.sum_congr rfl ?_
      intro i _
      simpa using (toVec_mul_spec (a := a) (b := b) (i := i))

-- Converting the spec-level `List.finRange` fold for `vec_mat_mul_spec` into a `Finset.univ` sum.
/-- Coordinate formula for `vec_mat_mul_spec` as a `Finset` sum: `(v @ A)[j] = ∑ i, v[i] * A[i,j]`.
  -/
lemma toVec_vec_mat_mul_spec {m n : Nat}
  (v : Tensor ℝ (.dim m .scalar))
  (A : Tensor ℝ (.dim m (.dim n .scalar))) (j : Fin n) :
  toVec (vecMatMulSpec v A) j = ∑ i : Fin m, (toVec v i) * (get2 A i j) := by
  classical
  cases v with
  | dim valuesV =>
    cases A with
    | dim rowsA =>
      simp [vecMatMulSpec, toVec, get2_eq, get_eq]
      let f : Fin m → ℝ := fun i =>
        vecMatMulSpec.match_3
          (motive := fun _ _ => ℝ)
          (valuesV i) (rowsA i)
          (fun vi colsA =>
            vecMatMulSpec.match_1
              (motive := fun _ => ℝ)
              (colsA j)
              (fun aij => vi * aij))

      have hfun :
          (fun (s : ℝ) (i : Fin m) =>
              vecMatMulSpec.match_3
                (motive := fun _ _ => ℝ)
                (valuesV i) (rowsA i)
                (fun vi colsA =>
                  vecMatMulSpec.match_1
                    (motive := fun _ => ℝ)
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
Adjointness of matrix-vector and vector-matrix multiplication under the `dot` product:
`⟪y, W x⟫ = ⟪y W, x⟫` (a.k.a. `⟪y, W x⟫ = ⟪Wᵀ y, x⟫` depending on conventions).

This is the algebraic heart of the linear-layer gradient rule.
-/
theorem dot_mat_linear_adjoint
  {inDim outDim : Nat}
  (W : Tensor ℝ (.dim outDim (.dim inDim .scalar)))
  (dLdy : Tensor ℝ (.dim outDim .scalar))
  (dx : Tensor ℝ (.dim inDim .scalar)) :
  dot dLdy (matVecMulSpec W dx)
  = dot (vecMatMulSpec dLdy W) dx := by
  -- This is the key adjoint property: ⟨y, Wx⟩ = ⟨W^T y, x⟩
  -- Essential for gradient computation in neural networks
  classical
  calc
    dot dLdy (matVecMulSpec W dx)
        = ∑ i : Fin outDim, (toVec dLdy i) * (toVec (matVecMulSpec W dx) i) := by
            simpa using (dot_vec_eq_sum (a := dLdy) (b := matVecMulSpec W dx))
    _ = ∑ i : Fin outDim, (toVec dLdy i) * (∑ k : Fin inDim, (get2 W i k) * (toVec dx k)) := by
            refine Finset.sum_congr rfl ?_
            intro i _
            simp [toVec_mat_vec_mul_spec]
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
            simp [toVec_vec_mat_mul_spec]
    _ = dot (vecMatMulSpec dLdy W) dx := by
            symm
            simpa using (dot_vec_eq_sum (a := vecMatMulSpec dLdy W) (b := dx))

/--
`shapeOf` recovers the shape already tracked in the tensor type.

This is a small bridge for proofs that move between value-level shape computations and type-indexed
tensor operations.
-/
theorem shapeOf_eq_shape {α : Type} {s : Shape} (t : Tensor α s) :
  shapeOf t = s := by
  induction s with
  | scalar =>
    match t with
    | Tensor.scalar _ => rfl
  | dim n s ih =>
    match t with
    | Tensor.dim f =>
      match n with
      | 0 => rfl
      | Nat.succ n' =>
        -- apply induction hypothesis to first element
        have h := ih (f ⟨0, Nat.zero_lt_succ n'⟩)
        simpa [shapeOf, h]


/-- Indexing the outer dimension of a tensor exposes a subtensor with the declared inner shape. -/
theorem get_preserves_inner_shape {n : Nat} {s : Shape}
  (t : Tensor ℝ (.dim n s)) (i : Fin n) :
  shapeOf (get t i) = s := by
  cases t with
  | dim f =>
    simp only [get]
    exact shapeOf_eq_shape (f i)

/-! ## Map and elementwise operation laws -/

/-- Functor identity law for `map_spec`: mapping `id` is a no-op. -/
theorem map_spec_id {s : Shape} (t : Tensor ℝ s) :
  mapSpec id t = t := by
  induction s with
  | scalar => cases t; rfl
  | dim n s ih =>
    cases t with | dim f =>
    simp [mapSpec]
    funext i
    exact ih (f i)

/-- Functor law for `map_spec`: mapping `g` then `f` equals mapping `f ∘ g`. -/
theorem map_spec_comp {s : Shape} (f g : ℝ → ℝ) (t : Tensor ℝ s) :
  mapSpec f (mapSpec g t) = mapSpec (f ∘ g) t := by
  induction s with
  | scalar => cases t; rfl
  | dim n s ih =>
    cases t with | dim h =>
    simp [mapSpec]
    funext i
    exact ih (h i)

/-- A scalar additivity law lifts pointwise through `map_spec` and `add_spec`. -/
theorem map_spec_add_distrib {s : Shape} (f : ℝ → ℝ) (a b : Tensor ℝ s)
  (h : ∀ x y, f (x + y) = f x + f y) :
  mapSpec f (addSpec a b) = addSpec (mapSpec f a) (mapSpec f b) := by
  induction s with
  | scalar =>
    cases a; cases b
    simp [mapSpec, addSpec, map2Spec]
    exact h _ _
  | dim n s ih =>
    cases a; cases b; rename_i fa fb
    simp [mapSpec, addSpec, map2Spec]
    funext i
    exact ih (fa i) (fb i)

/-- Commutativity transfer: if `f` is commutative, then `map2_spec f` is commutative on tensors. -/
theorem map2_spec_comm {s : Shape} (f : ℝ → ℝ → ℝ) (a b : Tensor ℝ s)
  (h : ∀ x y, f x y = f y x) :
  map2Spec f a b = map2Spec f b a := by
  induction s with
  | scalar => cases a; cases b; simp [map2Spec]; exact h _ _
  | dim n s ih =>
    cases a; cases b; rename_i fa fb
    simp [map2Spec]
    funext i
    exact ih (fa i) (fb i)

/-! ## Matrix and vector algebra -/

/-- Associativity of matrix-vector multiplication: `A (B x) = (A B) x`. -/
theorem mat_vec_assoc {m n p : Nat}
  (A : Tensor ℝ (.dim m (.dim n .scalar)))
  (B : Tensor ℝ (.dim n (.dim p .scalar)))
  (x : Tensor ℝ (.dim p .scalar)) :
  matVecMulSpec A (matVecMulSpec B x) =
  matVecMulSpec (matMulSpec A B) x := by
  classical
  have hto :
      toVec (matVecMulSpec A (matVecMulSpec B x)) =
        toVec (matVecMulSpec (matMulSpec A B) x) := by
    funext i
    have hBx : ∀ k : Fin n,
        toVec (matVecMulSpec B x) k = ∑ j : Fin p, (get2 B k j) * (toVec x j) := by
      intro k
      simpa using (toVec_mat_vec_mul_spec (A := B) (v := x) (i := k))

    -- Expand both sides into finite sums and use a Fubini-style swap.
    have h_expand :
        (∑ k : Fin n, (get2 A i k) * (∑ j : Fin p, (get2 B k j) * (toVec x j))) =
          (∑ j : Fin p, (∑ k : Fin n, (get2 A i k) * (get2 B k j)) * (toVec x j)) := by
      -- This is a finite-dimensional distributivity/commutation identity.
      -- We follow the standard pattern: expand, swap sums, factor.
      classical
      -- Expand `get2 A i k * (∑ j, ...)` into a double sum.
      have h1 :
          (∑ k : Fin n, (get2 A i k) * (∑ j : Fin p, (get2 B k j) * (toVec x j))) =
            (∑ k : Fin n, ∑ j : Fin p, (get2 A i k) * ((get2 B k j) * (toVec x j))) := by
        simp [Finset.mul_sum]
      -- Swap the order of summation.
      have h2 :
          (∑ k : Fin n, ∑ j : Fin p, (get2 A i k) * ((get2 B k j) * (toVec x j))) =
            (∑ j : Fin p, ∑ k : Fin n, (get2 A i k) * ((get2 B k j) * (toVec x j))) := by
        simpa using
          (Finset.sum_comm (s := (Finset.univ : Finset (Fin n))) (t := (Finset.univ : Finset (Fin
            p)))
            (f := fun k j => (get2 A i k) * ((get2 B k j) * (toVec x j))))
      -- Factor `(toVec x j)` out of the inner sum.
      have h3 :
          (∑ j : Fin p, ∑ k : Fin n, (get2 A i k) * ((get2 B k j) * (toVec x j))) =
            (∑ j : Fin p, (∑ k : Fin n, (get2 A i k) * (get2 B k j)) * (toVec x j)) := by
        refine Finset.sum_congr rfl ?_
        intro j _
        have h_reassoc :
            (∑ k : Fin n, (get2 A i k) * ((get2 B k j) * (toVec x j))) =
              (∑ k : Fin n, ((get2 A i k) * (get2 B k j)) * (toVec x j)) := by
          refine Finset.sum_congr rfl ?_
          intro k _
          simpa using (mul_assoc (get2 A i k) (get2 B k j) (toVec x j)).symm
        have h_pull :
            (∑ k : Fin n, ((get2 A i k) * (get2 B k j)) * (toVec x j)) =
              (∑ k : Fin n, (get2 A i k) * (get2 B k j)) * (toVec x j) := by
          simp [Finset.sum_mul]
        exact h_reassoc.trans h_pull

      exact h1.trans (h2.trans h3)

    -- Turn the vector components into the needed sum forms, then apply `h_expand`.
    have lhs :
        toVec (matVecMulSpec A (matVecMulSpec B x)) i =
          ∑ k : Fin n, (get2 A i k) * (∑ j : Fin p, (get2 B k j) * (toVec x j)) := by
      -- start from `toVec_mat_vec_mul_spec` and rewrite each inner component via `hBx`
      have hA :
          toVec (matVecMulSpec A (matVecMulSpec B x)) i =
            ∑ k : Fin n, (get2 A i k) * (toVec (matVecMulSpec B x) k) := by
        simpa using (toVec_mat_vec_mul_spec (A := A) (v := matVecMulSpec B x) (i := i))
      -- rewrite `toVec (mat_vec_mul_spec B x) k`
      classical
      refine hA.trans ?_
      refine Finset.sum_congr rfl ?_
      intro k _
      simp [hBx k]

    have rhs :
        toVec (matVecMulSpec (matMulSpec A B) x) i =
          ∑ j : Fin p, (∑ k : Fin n, (get2 A i k) * (get2 B k j)) * (toVec x j) := by
      -- rewrite the matrix multiplication entry via `get2_mat_mul_spec`
      have hR :
          toVec (matVecMulSpec (matMulSpec A B) x) i =
            ∑ j : Fin p, (get2 (matMulSpec A B) i j) * (toVec x j) := by
        simpa using (toVec_mat_vec_mul_spec (A := matMulSpec A B) (v := x) (i := i))
      -- now rewrite `get2 (mat_mul_spec A B) i j`
      classical
      refine hR.trans ?_
      refine Finset.sum_congr rfl ?_
      intro j _
      simp [get2_mat_mul_spec]

    -- Combine.
    simpa [lhs, rhs] using h_expand

  -- Lift pointwise equality back to tensors via `ofVec`.
  have h := congrArg ofVec hto
  -- `ofVec (toVec t) = t` for vectors.
  simpa [ofVec_toVec] using h

/-- Matrix transpose is an involution. -/
theorem matrix_transpose_involution {m n : Nat}
  (A : Tensor ℝ (.dim m (.dim n .scalar))) :
  matrixTransposeSpec (matrixTransposeSpec A) = A := by
  cases A with
  | dim rows =>
    -- reduce to function extensionality on the underlying `Fin`-indexed structure
    apply congrArg Tensor.dim
    funext i
    -- each row is itself a `.dim`
    cases hrow : rows i with
    | dim cols =>
      -- show the transposed-transposed row equals the original row
      apply congrArg Tensor.dim
      funext j
      cases hcol : cols j with
      | scalar v =>
        -- everything is definitional once we unfold `matrix_transpose_spec`
        simp [hrow, hcol]

/-- Coordinate rule for `matrix_transpose_spec`: `(Aᵀ)[i,j] = A[j,i]`. -/
lemma get2_matrix_transpose_spec {m n : Nat}
  (A : Tensor ℝ (.dim m (.dim n .scalar))) (i : Fin n) (j : Fin m) :
  get2 (matrixTransposeSpec A) i j = get2 A j i := by
  cases A with
  | dim rows =>
    -- Unfold transpose + `get2` and reduce the extra scalar match.
    simp [Tensor.matrixTransposeSpec, get2_eq, get_eq]
    cases hrow : rows j with
    | dim cols =>
      cases hcol : cols i with
      | scalar value =>
        simp [hcol]

/-- Matrix extensionality: matrices are equal when all their entries are equal. -/
lemma matrix_ext {m n : Nat} {A B : Tensor ℝ (.dim m (.dim n .scalar))} :
  (∀ i : Fin m, ∀ j : Fin n, get2 A i j = get2 B i j) → A = B := by
  intro h
  cases A with
  | dim rowsA =>
    cases B with
    | dim rowsB =>
      apply congrArg Tensor.dim
      funext i

      -- Prove row equality via `toVec` and then lift back with `ofVec`.
      have hto : toVec (rowsA i) = toVec (rowsB i) := by
        funext j
        cases hrowA : rowsA i with
        | dim colsA =>
          cases hrowB : rowsB i with
          | dim colsB =>
            cases hcolA : colsA j with
            | scalar a =>
              cases hcolB : colsB j with
              | scalar b =>
                have hij : get2 (Tensor.dim rowsA) i j = get2 (Tensor.dim rowsB) i j := h i j
                have hab : a = b := by
                  simpa [get2_eq, get_eq, hrowA, hrowB, hcolA, hcolB] using hij
                simp [toVec, hcolA, hcolB, hab]

      have hrow := congrArg ofVec hto
      simpa [ofVec_toVec] using hrow

/-- Transpose of a product: `(A ⬝ B)ᵀ = Bᵀ ⬝ Aᵀ`. -/
theorem matrix_transpose_mul {m n p : Nat}
  (A : Tensor ℝ (.dim m (.dim n .scalar)))
  (B : Tensor ℝ (.dim n (.dim p .scalar))) :
  matrixTransposeSpec (matMulSpec A B) =
  matMulSpec (matrixTransposeSpec B) (matrixTransposeSpec A) := by
  classical
  -- Prove equality by `get2`-extensionality on matrix entries.
  apply matrix_ext
  intro j i
  -- Compare the `(j,i)` entry of both sides.
  calc
    get2 (matrixTransposeSpec (matMulSpec A B)) j i
        = get2 (matMulSpec A B) i j := by
            simpa using (get2_matrix_transpose_spec (A := matMulSpec A B) (i := j) (j := i))
    _ = ∑ k : Fin n, (get2 A i k) * (get2 B k j) := by
          simpa using (get2_mat_mul_spec (A := A) (B := B) (i := i) (j := j))
    _ = ∑ k : Fin n, (get2 B k j) * (get2 A i k) := by
          refine Finset.sum_congr rfl ?_
          intro k _
          simp [mul_comm]
    _ = ∑ k : Fin n, (get2 (matrixTransposeSpec B) j k) * (get2 (matrixTransposeSpec A) k i) :=
      by
          refine Finset.sum_congr rfl ?_
          intro k _
          simp [get2_matrix_transpose_spec]
    _ = get2 (matMulSpec (matrixTransposeSpec B) (matrixTransposeSpec A)) j i := by
          symm
          simpa using
            (get2_mat_mul_spec (A := matrixTransposeSpec B) (B := matrixTransposeSpec A) (i :=
              j) (j := i))

-- ---------------------------------------------------------------------------
-- Frobenius dot / matmul adjointness
-- ---------------------------------------------------------------------------

/-- Expand the matrix dot-product as a double sum over entries (Frobenius inner product). -/
lemma dot_mat_eq_sum {m n : Nat}
  (A B : Tensor ℝ (.dim m (.dim n .scalar))) :
  dot A B = ∑ i : Fin m, ∑ j : Fin n, (get2 A i j) * (get2 B i j) := by
  classical
  cases A with
  | dim rowsA =>
    cases B with
    | dim rowsB =>
      -- Unfold `dot` to `sum_spec (mul_spec ...)` and sum over the outer dimension.
      have hout :
          dot (Tensor.dim rowsA) (Tensor.dim rowsB)
            =
          ∑ i : Fin m, sumSpec (mulSpec (rowsA i) (rowsB i)) := by
        -- `mul_spec` is rowwise, and `sum_spec_dim` unfolds the outer fold.
        simpa [dot, mulSpec, map2Spec, get_eq] using
          (sum_spec_dim (t := mulSpec (Tensor.dim rowsA) (Tensor.dim rowsB)))
      -- Unfold each row sum as a coordinate sum.
      calc
        dot (Tensor.dim rowsA) (Tensor.dim rowsB)
            = ∑ i : Fin m, sumSpec (mulSpec (rowsA i) (rowsB i)) := hout
        _ = ∑ i : Fin m, ∑ j : Fin n,
              (get2 (Tensor.dim rowsA) i j) * (get2 (Tensor.dim rowsB) i j) := by
              refine Finset.sum_congr rfl ?_
              intro i _
              cases hA : rowsA i with
              | dim colsA =>
                cases hB : rowsB i with
                | dim colsB =>
                  -- Rowwise: reduce to the vector lemma `sum_spec_vec`.
                  have hsum :
                      sumSpec (mulSpec (Tensor.dim colsA) (Tensor.dim colsB))
                        =
                      ∑ j : Fin n, toVec (mulSpec (Tensor.dim colsA) (Tensor.dim colsB)) j := by
                      simpa using (sum_spec_vec (v := mulSpec (Tensor.dim colsA) (Tensor.dim
                        colsB)))
                  -- Rewrite via `sum_spec_vec`, then compare summands coordinatewise.
                  rw [hsum]
                  refine Finset.sum_congr rfl ?_
                  intro j _
                  -- Everything is definitional on scalar entries.
                  cases hcolA : colsA j with
                  | scalar a =>
                    cases hcolB : colsB j with
                    | scalar b =>
                      simp [toVec, mulSpec, map2Spec, get2_eq, get_eq, hA, hB, hcolA, hcolB]

/-- Right-adjointness of matrix multiplication under the Frobenius dot-product.

Informally: `⟪A ⬝ B, C⟫ = ⟪A, C ⬝ Bᵀ⟫`.
-/
theorem dot_mat_mul_right_adjoint
  {m n p : Nat}
  (A : Tensor ℝ (.dim m (.dim n .scalar)))
  (B : Tensor ℝ (.dim n (.dim p .scalar)))
  (C : Tensor ℝ (.dim m (.dim p .scalar))) :
  dot (matMulSpec A B) C = dot A (matMulSpec C (matrixTransposeSpec B)) := by
  classical
  -- Expand both sides into entry sums; then it's just rearranging a finite triple sum.
  -- LHS: ∑ i ∑ j (∑ k Aik*Bkj) * Cij
  -- RHS: ∑ i ∑ k Aik * (∑ j Cij*Bkj)
  rw [dot_mat_eq_sum (A := matMulSpec A B) (B := C)]
  rw [dot_mat_eq_sum (A := A) (B := matMulSpec C (matrixTransposeSpec B))]
  -- Rewrite matrix products and transpose entries.
  simp [get2_mat_mul_spec, get2_matrix_transpose_spec, Finset.mul_sum, Finset.sum_mul]
  -- The goal is now exactly `Finset.sum_comm` on the two inner indices.
  refine Finset.sum_congr rfl ?_
  intro i _
  simpa [mul_assoc, mul_left_comm, mul_comm] using
    (Finset.sum_comm (s := (Finset.univ : Finset (Fin p))) (t := (Finset.univ : Finset (Fin n)))
      (f := fun j k => get2 A i k * (get2 C i j * get2 B k j)))

/-- Transpose invariance of the Frobenius dot-product: `⟪Aᵀ, Bᵀ⟫ = ⟪A, B⟫`. -/
lemma dot_mat_transpose {m n : Nat}
  (A B : Tensor ℝ (.dim m (.dim n .scalar))) :
  dot (matrixTransposeSpec A) (matrixTransposeSpec B) = dot A B := by
  classical
  -- Expand both sides and use `get2_matrix_transpose_spec`.
  rw [dot_mat_eq_sum (A := matrixTransposeSpec A) (B := matrixTransposeSpec B)]
  rw [dot_mat_eq_sum (A := A) (B := B)]
  -- LHS is `∑ i:Fin n, ∑ j:Fin m, A_{j,i} * B_{j,i}`; swap sums to match RHS.
  simpa [get2_matrix_transpose_spec, mul_assoc, mul_left_comm, mul_comm] using
    (Finset.sum_comm
      (s := (Finset.univ : Finset (Fin n)))
      (t := (Finset.univ : Finset (Fin m)))
      (f := fun i j => get2 A j i * get2 B j i))

/-- Left-adjointness of matrix multiplication under the Frobenius dot-product.

Informally: `⟪A ⬝ B, C⟫ = ⟪B, Aᵀ ⬝ C⟫`.
-/
theorem dot_mat_mul_left_adjoint
  {m n p : Nat}
  (A : Tensor ℝ (.dim m (.dim n .scalar)))
  (B : Tensor ℝ (.dim n (.dim p .scalar)))
  (C : Tensor ℝ (.dim m (.dim p .scalar))) :
  dot (matMulSpec A B) C = dot B (matMulSpec (matrixTransposeSpec A) C) := by
  classical
  -- Reduce to the right-adjoint lemma via transpose.
  -- ⟪A·B, C⟫ = ⟪(A·B)ᵀ, Cᵀ⟫ = ⟪Bᵀ·Aᵀ, Cᵀ⟫ = ⟪B, (Cᵀ·A)ᵀ⟫ = ⟪B, Aᵀ·C⟫.
  have htrans :=
    (dot_mat_transpose (m := m) (n := p) (A := matMulSpec A B) (B := C)).symm
  -- rewrite `(A·B)ᵀ`
  have hmulT :
      matrixTransposeSpec (matMulSpec A B) =
        matMulSpec (matrixTransposeSpec B) (matrixTransposeSpec A) :=
    matrix_transpose_mul (A := A) (B := B)
  -- apply the right-adjoint lemma to `Bᵀ·Aᵀ` against `Cᵀ`
  have hadj :
      dot (matMulSpec (matrixTransposeSpec B) (matrixTransposeSpec A)) (matrixTransposeSpec
        C)
        =
      dot (matrixTransposeSpec B)
        (matMulSpec (matrixTransposeSpec C) (matrixTransposeSpec (matrixTransposeSpec A)))
          := by
    simpa using
      (dot_mat_mul_right_adjoint (A := matrixTransposeSpec B) (B := matrixTransposeSpec A)
        (C := matrixTransposeSpec C))
  -- simplify involutions and transpose the last dot back
  have hAinv : matrixTransposeSpec (matrixTransposeSpec A) = A :=
    matrix_transpose_involution (A := A)
  have hCinv : matrixTransposeSpec (matrixTransposeSpec C) = C :=
    matrix_transpose_involution (A := C)
  -- `dot (Bᵀ) D = dot B (Dᵀ)` for matching shapes.
  have hdot_swap :
      dot (matrixTransposeSpec B) (matMulSpec (matrixTransposeSpec C) A)
        =
      dot B (matrixTransposeSpec (matMulSpec (matrixTransposeSpec C) A)) := by
    -- Apply `dot_mat_transpose` to `B` and `((Cᵀ·A)ᵀ)`.
    have := dot_mat_transpose (m := n) (n := p)
      (A := B) (B := matrixTransposeSpec (matMulSpec (matrixTransposeSpec C) A))
    -- Rewrite involutions.
    simpa [matrix_transpose_involution, hCinv] using this
  -- Finish by rewriting `transpose (Cᵀ·A) = Aᵀ·C`.
  calc
    dot (matMulSpec A B) C
        = dot (matrixTransposeSpec (matMulSpec A B)) (matrixTransposeSpec C) := htrans
    _ = dot (matMulSpec (matrixTransposeSpec B) (matrixTransposeSpec A))
      (matrixTransposeSpec C) := by
          simp [hmulT]
    _ = dot (matrixTransposeSpec B) (matMulSpec (matrixTransposeSpec C) A) := by
          simpa [hAinv] using hadj
    _ = dot B (matrixTransposeSpec (matMulSpec (matrixTransposeSpec C) A)) := hdot_swap
    _ = dot B (matMulSpec (matrixTransposeSpec A) C) := by
          -- `transpose (Cᵀ·A) = Aᵀ·C`
          simp [matrix_transpose_mul, hCinv]

/--
Outer product properties.
Essential for proving weight gradient correctness.
-/
theorem outer_product_transpose {m n : Nat}
  (a : Tensor ℝ (.dim m .scalar))
  (b : Tensor ℝ (.dim n .scalar)) :
  matrixTransposeSpec (outerProductSpec a b) = outerProductSpec b a := by
  cases a with | dim fa =>
  cases b with | dim fb =>
  simp only [outerProductSpec, matrixTransposeSpec]
  -- extensionality on the outer/inner indices
  apply congrArg Tensor.dim
  funext i
  apply congrArg Tensor.dim
  funext j
  cases fa j with
  | scalar x =>
    cases fb i with
    | scalar y =>
      simp [mul_comm]

/-! ## Reductions and aggregation -/

/-- Sum distributes over elementwise addition. -/
theorem sum_spec_add_distrib {s : Shape} (a b : Tensor ℝ s) :
  sumSpec (addSpec a b) = sumSpec a + sumSpec b := by
  unfold sumSpec addSpec
  induction s with
  | scalar =>
    cases a; cases b
    simp [tensorFoldlSpec, map2Spec]
  | dim n s ih =>
    cases a with | dim fa =>
    cases b with | dim fb =>
    simp [tensorFoldlSpec, map2Spec]
    -- We need to show that folding over the sum equals sum of the folds
    -- This uses properties of fold with addition
    -- tensor_foldl_spec.go (· + ·) n s (fun i => add_spec (fa i) (fb i)) 0 0 =
    -- tensor_foldl_spec.go (· + ·) n s fa 0 0 + tensor_foldl_spec.go (· + ·) n s fb 0 0
    have h : ∀ k acc1 acc2 acc3, k ≤ n → acc3 = acc1 + acc2 →
      tensorFoldlSpec.go (· + ·) n s (fun i => addSpec (fa i) (fb i)) k acc3 =
      tensorFoldlSpec.go (· + ·) n s fa k acc1 + tensorFoldlSpec.go (· + ·) n s fb k acc2 := by
      intro k acc1 acc2 acc3 hk hacc
      induction hn : n - k generalizing k acc1 acc2 acc3 with
      | zero =>
        have k_eq_n : k = n := by
          exact Nat.le_antisymm hk (Nat.sub_eq_zero_iff_le.mp hn)
        subst k
        simp [tensor_foldl_spec_go_of_not_lt, hacc]
      | succ m ih_fold =>
        have hlt : k < n := by
          exact Nat.sub_pos_iff_lt.mp (by simp [hn])
        -- Peel one loop step at index `k` for each `go`.
        rw [tensor_foldl_spec_go_of_lt (f := (· + ·)) (values := fun i => addSpec (fa i) (fb i))
          (k := k) (acc := acc3) hlt]
        rw [tensor_foldl_spec_go_of_lt (f := (· + ·)) (values := fa) (k := k) (acc := acc1) hlt]
        rw [tensor_foldl_spec_go_of_lt (f := (· + ·)) (values := fb) (k := k) (acc := acc2) hlt]
        have h_next : n - (k + 1) = m := by
          simp [Nat.sub_succ, hn]
        -- Apply IH with updated accumulators
        let new_acc1 := tensorFoldlSpec (· + ·) acc1 (fa ⟨k, hlt⟩)
        let new_acc2 := tensorFoldlSpec (· + ·) acc2 (fb ⟨k, hlt⟩)
        let new_acc3 := tensorFoldlSpec (· + ·) acc3 (addSpec (fa ⟨k, hlt⟩) (fb ⟨k, hlt⟩))
        -- Need to show new_acc3 = new_acc1 + new_acc2
        have acc_eq : new_acc3 = new_acc1 + new_acc2 := by
          simp only [new_acc1, new_acc2, new_acc3]
          -- tensor_foldl_spec with addition just adds sum_spec to accumulator
          have h1 : tensorFoldlSpec (· + ·) acc3 (addSpec (fa ⟨k, hlt⟩) (fb ⟨k, hlt⟩)) =
                    acc3 + sumSpec (addSpec (fa ⟨k, hlt⟩) (fb ⟨k, hlt⟩)) := by
            simpa using (tensor_foldl_spec_add_init (s := s) (acc := acc3)
              (t := addSpec (fa ⟨k, hlt⟩) (fb ⟨k, hlt⟩)))
          have h2 : tensorFoldlSpec (· + ·) acc1 (fa ⟨k, hlt⟩) = acc1 + sumSpec (fa ⟨k, hlt⟩) :=
            by
            simpa using (tensor_foldl_spec_add_init (s := s) (acc := acc1) (t := fa ⟨k, hlt⟩))
          have h3 : tensorFoldlSpec (· + ·) acc2 (fb ⟨k, hlt⟩) = acc2 + sumSpec (fb ⟨k, hlt⟩) :=
            by
            simpa using (tensor_foldl_spec_add_init (s := s) (acc := acc2) (t := fb ⟨k, hlt⟩))
          rw [h1, h2, h3]
          -- Now we have: acc3 + sum_spec (add_spec (fa ⟨k, hlt⟩) (fb ⟨k, hlt⟩)) =
          --              acc1 + sum_spec (fa ⟨k, hlt⟩) + (acc2 + sum_spec (fb ⟨k, hlt⟩))
          -- We can use ih: sum_spec (add_spec a b) = sum_spec a + sum_spec b
          -- First show that sum_spec (add_spec (fa ⟨k, hlt⟩) (fb ⟨k, hlt⟩)) = sum_spec (fa ⟨k,
          -- hlt⟩) + sum_spec (fb ⟨k, hlt⟩)
          have sum_add : sumSpec (addSpec (fa ⟨k, hlt⟩) (fb ⟨k, hlt⟩)) = sumSpec (fa ⟨k, hlt⟩) +
            sumSpec (fb ⟨k, hlt⟩) := by
            -- Apply the induction hypothesis
            unfold sumSpec
            unfold addSpec
            exact ih (fa ⟨k, hlt⟩) (fb ⟨k, hlt⟩)
          rw [sum_add, hacc]
          ring
        exact ih_fold (k + 1) new_acc1 new_acc2 new_acc3 (Nat.succ_le_of_lt hlt) acc_eq h_next
    exact h 0 0 0 0 (Nat.zero_le n) (by ring : (0 : ℝ) = 0 + 0)

/--
Dot product properties.
Essential for gradient computations and neural network training.
-/
theorem dot_comm {s : Shape} (a b : Tensor ℝ s) :
  dot a b = dot b a := by
  simp [dot, mul_spec_comm]

/-- Dot-product distributes over addition in the left argument. -/
theorem dot_add_left {s : Shape} (a b c : Tensor ℝ s) :
  dot (addSpec a b) c = dot a c + dot b c := by
  simp [dot]
  -- Reduce to distributivity of `mul_spec` and `sum_spec_add_distrib`.
  have hmul : mulSpec (addSpec a b) c = addSpec (mulSpec a c) (mulSpec b c) := by
    -- Structural recursion on `s` and use distributivity of ℝ.
    induction s with
    | scalar =>
      cases a; cases b; cases c
      simp [mulSpec, addSpec, map2Spec]
      ring
    | dim n s ih =>
      cases a with | dim fa =>
      cases b with | dim fb =>
      cases c with | dim fc =>
      simp [mulSpec, addSpec, map2Spec]
      funext i
      exact ih (fa i) (fb i) (fc i)
  rw [hmul]
  simpa using sum_spec_add_distrib (a := mulSpec a c) (b := mulSpec b c)

/-- Scaling a tensor scales its dot-product: `dot (scale_spec a k) b = k * dot a b`. -/
theorem dot_scale_left {s : Shape} (a b : Tensor ℝ s) (k : ℝ) :
  dot (scaleSpec a k) b = k * dot a b := by
  -- Induction on the tensor shape.
  induction s with
  | scalar =>
    cases a with
    | scalar x =>
      cases b with
      | scalar y =>
        simp [dot, scaleSpec, sumSpec, tensorFoldlSpec, mulSpec, mapSpec, map2Spec]
        ring
  | dim n s ih =>
    cases a with
    | dim fa =>
      cases b with
      | dim fb =>
        -- Reduce both sides to scaling of the outer fold.
        let scaled : Fin n → Tensor ℝ s := fun i => mulSpec (scaleSpec (fa i) k) (fb i)
        let unscaled : Fin n → Tensor ℝ s := fun i => mulSpec (fa i) (fb i)

        have component : ∀ i : Fin n, sumSpec (scaled i) = k * sumSpec (unscaled i) := by
          intro i
          simpa [dot, scaled, unscaled] using ih (a := fa i) (b := fb i)

        have go_scale : ∀ j acc,
            tensorFoldlSpec.go (· + ·) n s scaled j (k * acc) =
              k * tensorFoldlSpec.go (· + ·) n s unscaled j acc := by
          intro j acc
          induction hn : n - j generalizing j acc with
          | zero =>
            have hnot : ¬ j < n := by
              exact Nat.not_lt.mpr (Nat.sub_eq_zero_iff_le.mp hn)
            -- Both `go` loops terminate and return the accumulator.
            simp
              [tensor_foldl_spec_go_of_not_lt (f := (· + ·)) (values := scaled) (k := j) (acc := k * acc) hnot,
                tensor_foldl_spec_go_of_not_lt (f := (· + ·)) (values := unscaled) (k := j) (acc := acc) hnot]
          | succ m ihj =>
            have hlt : j < n := by
              exact Nat.sub_pos_iff_lt.mp (by simp [hn])
            rw [tensor_foldl_spec_go_of_lt (f := (· + ·)) (values := scaled) (k := j) (acc := k * acc) hlt]
            rw [tensor_foldl_spec_go_of_lt (f := (· + ·)) (values := unscaled) (k := j) (acc := acc) hlt]
            have h_next : n - (j + 1) = m := by
              simp [Nat.sub_succ, hn]
            -- Show that the accumulator update scales correctly.
            have step :
                tensorFoldlSpec (· + ·) (k * acc) (scaled ⟨j, hlt⟩) =
                  k * tensorFoldlSpec (· + ·) acc (unscaled ⟨j, hlt⟩) := by
              have h_scaled :
                  tensorFoldlSpec (· + ·) (k * acc) (scaled ⟨j, hlt⟩) =
                    (k * acc) + sumSpec (scaled ⟨j, hlt⟩) := by
                simpa using
                  (tensor_foldl_spec_add_init (s := s) (acc := k * acc) (t := scaled ⟨j, hlt⟩))
              have h_unscaled :
                  tensorFoldlSpec (· + ·) acc (unscaled ⟨j, hlt⟩) =
                    acc + sumSpec (unscaled ⟨j, hlt⟩) := by
                simpa using
                  (tensor_foldl_spec_add_init (s := s) (acc := acc) (t := unscaled ⟨j, hlt⟩))
              have h_comp : sumSpec (scaled ⟨j, hlt⟩) = k * sumSpec (unscaled ⟨j, hlt⟩) :=
                component ⟨j, hlt⟩
              calc
                tensorFoldlSpec (· + ·) (k * acc) (scaled ⟨j, hlt⟩)
                    = (k * acc) + sumSpec (scaled ⟨j, hlt⟩) := h_scaled
                _ = (k * acc) + (k * sumSpec (unscaled ⟨j, hlt⟩)) := by
                      simp [h_comp]
                _ = k * (acc + sumSpec (unscaled ⟨j, hlt⟩)) := by
                      ring
                _ = k * tensorFoldlSpec (· + ·) acc (unscaled ⟨j, hlt⟩) := by
                      simp [h_unscaled]
            -- Apply IH to the tail with the correctly-scaled accumulator.
            simpa [step] using
              (ihj (j := j + 1) (acc := tensorFoldlSpec (· + ·) acc (unscaled ⟨j, hlt⟩)) h_next)

        -- Finish by unfolding `dot`/`sum_spec` and reducing to `go_scale` at `j=0, acc=0`.
        simpa [dot, sumSpec, tensorFoldlSpec, mulSpec, scaleSpec, mapSpec, map2Spec, scaled,
          unscaled] using
          (go_scale (j := 0) (acc := 0))

/-!
## Flatten / unflatten round-trips

The round-trip lemmas for the spec definitions live with the definitions themselves:
`NN/Spec/Core/TensorReductionShape.lean` provides

- `Spec.Tensor.flatten_unflatten_inverse`
- `Spec.Tensor.unflatten_flatten_inverse`

so downstream proof files do not need to re-prove the index arithmetic here.
-/

/--
Size preservation under operations.
Essential for proving tensor operations maintain expected dimensions.
-/
theorem shape_size_add {s : Shape} (a b : Tensor ℝ s) :
  Shape.size (shapeOf (addSpec a b)) = Shape.size s := by
  rw [shapeOf_eq_shape]

/-- Size preservation for `mul_spec`: elementwise multiplication does not change shape size. -/
theorem shape_size_mul {s : Shape} (a b : Tensor ℝ s) :
  Shape.size (shapeOf (mulSpec a b)) = Shape.size s := by
  rw [shapeOf_eq_shape]

-- Error and approximation theorems
/-
Numerical stability: helper bounds for `safediv_spec`.

The actual user-facing statement in this file is `safediv_bound`; the helpers below exist only to
make that proof easy to maintain.
-/
-- Helper: a uniform bound for all entries of a tensor (used in `safediv_bound`).
/-- Predicate `tensorAllLE b t`: all entries of `t` are `≤ b`. -/
private def tensorAllLE (bound : ℝ) : ∀ {s : Shape}, Tensor ℝ s → Prop
  | .scalar, .scalar x => x ≤ bound
  | .dim _ _, .dim values => ∀ j, tensorAllLE bound (values j)

-- Maximum entry value over a tensor (using `0` as a safe initial accumulator).
/-- A maximum-like bound for tensor entries, computed structurally (used only for bounding lemmas).
  -/
private def tensorMax : ∀ {s : Shape}, Tensor ℝ s → ℝ
  | .scalar, .scalar x => x
  | .dim n _, .dim values =>
      ((List.finRange n).map (fun j => tensorMax (values j))).foldl max 0

/-- The fold accumulator `init` is always `≤ foldl max init l`. -/
private lemma le_foldl_max_init (init : ℝ) (l : List ℝ) : init ≤ l.foldl max init := by
  induction l generalizing init with
  | nil =>
    simp
  | cons a tl ih =>
    simp [List.foldl]
    exact le_trans (le_max_left init a) (ih (init := max init a))

/-- Any element of a list is `≤ foldl max init l`. -/
private lemma le_foldl_max_of_mem {init x : ℝ} {l : List ℝ} (hx : x ∈ l) : x ≤ l.foldl max init :=
  by
  induction l generalizing init with
  | nil =>
    cases hx
  | cons a tl ih =>
    simp [List.mem_cons] at hx
    -- foldl max init (a :: tl) = foldl max (max init a) tl
    simp [List.foldl]
    cases hx with
    | inl hxa =>
      subst x
      exact le_trans (le_max_right init a) (le_foldl_max_init (init := max init a) tl)
    | inr hx' =>
      exact ih (init := max init a) hx'

/-- Each slice maximum is bounded by the tensor maximum of a `.dim` tensor. -/
private lemma tensorMax_le_dim {n : Nat} {inner : Shape} (values : Fin n → Tensor ℝ inner) (j : Fin
  n) :
    tensorMax (values j) ≤ tensorMax (Tensor.dim values) := by
  -- `tensorMax (Tensor.dim values)` is a `foldl max` over the list of slice maxima.
  have hj : j ∈ List.finRange n := by simp
  have hmem :
      tensorMax (values j) ∈ (List.finRange n).map (fun k => tensorMax (values k)) :=
    List.mem_map_of_mem (f := fun k => tensorMax (values k)) hj
  -- Use the generic list bound lemma.
  simpa [tensorMax] using
    (le_foldl_max_of_mem (init := 0) (x := tensorMax (values j))
      (l := (List.finRange n).map (fun k => tensorMax (values k))) hmem)

/-- Monotonicity: if `tensorAllLE b₁ t` and `b₁ ≤ b₂`, then `tensorAllLE b₂ t`. -/
private lemma tensorAllLE_mono {bnd₁ bnd₂ : ℝ} :
    ∀ {s : Shape} (t : Tensor ℝ s), tensorAllLE bnd₁ t → bnd₁ ≤ bnd₂ → tensorAllLE bnd₂ t := by
  intro s t ht hle
  induction s with
  | scalar =>
    cases t with
    | scalar x =>
      simpa [tensorAllLE] using le_trans ht hle
  | dim n inner ih =>
    cases t with
    | dim values =>
      intro j
      exact ih (t := values j) (ht j)

/-- Every tensor is bounded by its computed maximum: `tensorAllLE (tensorMax t) t`. -/
private lemma tensorAllLE_tensorMax : ∀ {s : Shape} (t : Tensor ℝ s), tensorAllLE (tensorMax t) t :=
  by
  intro s t
  induction s with
  | scalar =>
    cases t with
    | scalar x =>
      simp [tensorAllLE, tensorMax]
  | dim n inner ih =>
    cases t with
    | dim values =>
      intro j
      have h_sub : tensorAllLE (tensorMax (values j)) (values j) := ih (t := values j)
      have h_le : tensorMax (values j) ≤ tensorMax (Tensor.dim values) :=
        tensorMax_le_dim values j
      exact tensorAllLE_mono (t := values j) h_sub h_le

/-- If all entries are `≤ bound`, then clamping by `min · bound` is the identity. -/
private lemma map_min_eq_self_of_tensorAllLE {bound : ℝ} :
    ∀ {s : Shape} (t : Tensor ℝ s), tensorAllLE bound t →
      mapSpec (fun x => min x bound) t = t := by
  intro s t ht
  induction s with
  | scalar =>
    cases t with
    | scalar x =>
      have hx : x ≤ bound := by simpa [tensorAllLE] using ht
      simp [mapSpec, hx]
  | dim n inner ih =>
    cases t with
    | dim values =>
      apply congrArg Tensor.dim
      funext j
      exact ih (t := values j) (ht j)

/-- Existence of a uniform bound for `safediv_spec`, expressed via an idempotent `min`-clamp. -/
theorem safediv_bound {s : Shape} (a b : Tensor ℝ s) :
  ∀ _i : Fin s.size, (Numbers.epsilon : ℝ) > 0 →
  ∃ bound, absSpec (safedivSpec a b) = mapSpec (fun x => min x bound) (absSpec (safedivSpec a
    b)) := by
  -- The `i` and positivity hypothesis are irrelevant: the tensor has finitely many entries, so we
  -- can take `bound` to be a maximum over all entries (then `min x bound = x` everywhere).
  intro _ _
  let t := absSpec (safedivSpec a b)
  refine ⟨tensorMax t, ?_⟩
  simpa [t] using
    (Eq.symm (map_min_eq_self_of_tensorAllLE (t := t) (tensorAllLE_tensorMax (t := t))))

/--
Tensor norm properties.
Essential for regularization and optimization proofs.
-/
noncomputable def tensorNormSquared {s : Shape} (t : Tensor ℝ s) : ℝ :=
  dot t t

/-- `tensor_norm_squared t` is nonnegative, since it is a sum of squares. -/
theorem tensor_norm_squared_nonneg {s : Shape} (t : Tensor ℝ s) :
  tensorNormSquared t ≥ 0 := by
  -- `tensor_norm_squared t = dot t t = sum_spec (mul_spec t t)` is a sum of squares.
  -- We prove non-negativity by structural induction on the shape.
  -- (The recursion follows the definition of `tensor_foldl_spec` used by `sum_spec`.)
  induction s with
  | scalar =>
    cases t with
    | scalar x =>
      -- dot (scalar x) (scalar x) = x * x
      simp [tensorNormSquared, dot, sumSpec, mulSpec, map2Spec, tensorFoldlSpec,
        mul_self_nonneg]
  | dim n s ih =>
    cases t with
    | dim values =>
      -- Square each sub-tensor and sum all entries via `tensor_foldl_spec.go`.
      let valuesSq : Fin n → Tensor ℝ s := fun i => mulSpec (values i) (values i)
      have term_nonneg : ∀ i : Fin n, 0 ≤ sumSpec (valuesSq i) := by
        intro i
        -- `sum_spec (valuesSq i) = tensor_norm_squared (values i)` by definition.
        simpa [valuesSq, tensorNormSquared, dot] using (ih (t := values i))

      -- Show the fold accumulator stays nonnegative.
      have go_nonneg :
          ∀ k acc, k ≤ n → 0 ≤ acc →
            0 ≤ tensorFoldlSpec.go (· + ·) n s valuesSq k acc := by
        intro k acc hk hacc
        -- Induct on the remaining length `n - k`.
        induction hn : n - k generalizing k acc with
        | zero =>
          have hk' : k = n := by
            have : n ≤ k := Nat.sub_eq_zero_iff_le.mp hn
            exact Nat.le_antisymm hk this
          subst k
          -- `k = n` so the loop stops immediately.
          have hgo :
              tensorFoldlSpec.go (· + ·) n s valuesSq n acc = acc := by
            simpa using
              (tensor_foldl_spec_go_of_not_lt (f := (· + ·)) (values := valuesSq) (k := n) (acc := acc)
                (by simp))
          simp [hgo, hacc]
        | succ m ih_go =>
          have hlt : k < n := by
            have : 0 < n - k := by simp [hn]
            exact Nat.sub_pos_iff_lt.mp this
          have hk1 : k + 1 ≤ n := Nat.succ_le_of_lt hlt
          -- Unfold one loop step at index `k`.
          rw [tensor_foldl_spec_go_of_lt (f := (· + ·)) (values := valuesSq) (k := k) (acc := acc) hlt]
          -- The next accumulator is `acc + sum_spec (valuesSq k)`.
          have hstep :
              tensorFoldlSpec (· + ·) acc (valuesSq ⟨k, hlt⟩) =
                acc + sumSpec (valuesSq ⟨k, hlt⟩) := by
            simpa using
              (tensor_foldl_spec_add_init (s := s) (acc := acc) (t := valuesSq ⟨k, hlt⟩))
          have hacc' : 0 ≤ tensorFoldlSpec (· + ·) acc (valuesSq ⟨k, hlt⟩) := by
            have hterm : 0 ≤ sumSpec (valuesSq ⟨k, hlt⟩) := term_nonneg ⟨k, hlt⟩
            simpa [hstep] using add_nonneg hacc hterm
          -- Reduce `n - (k+1)` to `m` and apply IH.
          have h_next : n - (k + 1) = m := by
            rw [Nat.sub_succ, hn]
            rfl
          have := ih_go (k := k + 1)
            (acc := tensorFoldlSpec (· + ·) acc (valuesSq ⟨k, hlt⟩)) hk1 hacc'
          simpa [h_next] using this

      -- Finish by unfolding `tensor_norm_squared` and applying `go_nonneg` at `k=0, acc=0`.
      simpa [tensorNormSquared, dot, sumSpec, mulSpec, map2Spec, tensorFoldlSpec, valuesSq]
        using
        (go_nonneg (k := 0) (acc := (0 : ℝ)) (by exact Nat.zero_le n) (by simp))

/--
Convenience orientation of `tensor_norm_squared_nonneg`.

Keep this untagged as a simp lemma: the canonical theorem above uses the existing `>=` spelling,
while downstream analysis proofs often need the `0 ≤ ...` spelling for `Real.sq_sqrt`.
-/
theorem tensor_norm_squared_nonneg2 {s : Shape} (t : Tensor ℝ s) :
  0 <= tensorNormSquared t := by
  simpa [ge_iff_le] using tensor_norm_squared_nonneg (t := t)

/-- `tensor_norm_squared t = 0` iff `t` is the all-zero tensor. -/
theorem tensor_norm_squared_zero_iff {s : Shape} (t : Tensor ℝ s) :
  tensorNormSquared t = 0 ↔ t = fill (0 : ℝ) s := by
  -- Both directions by induction on the shape.
  induction s with
  | scalar =>
    cases t with
    | scalar x =>
      -- `tensor_norm_squared (scalar x) = x*x`.
      simp [tensorNormSquared, dot, sumSpec, mulSpec, map2Spec, tensorFoldlSpec, fill]
  | dim n s ih =>
    cases t with
    | dim values =>
      let valuesSq : Fin n → Tensor ℝ s := fun i => mulSpec (values i) (values i)
      have term_nonneg : ∀ i : Fin n, 0 ≤ sumSpec (valuesSq i) := by
        intro i
        simpa [ge_iff_le, valuesSq, tensorNormSquared, dot] using
          tensor_norm_squared_nonneg (t := values i)

      -- Monotonicity of the `go` loop: accumulator is always ≤ final result.
      have go_ge :
          ∀ k acc, k ≤ n →
            acc ≤ tensorFoldlSpec.go (· + ·) n s valuesSq k acc := by
        intro k acc hk
        induction hn : n - k generalizing k acc with
        | zero =>
          have hk' : k = n := by
            have : n ≤ k := Nat.sub_eq_zero_iff_le.mp hn
            exact Nat.le_antisymm hk this
          subst k
          have hgo :
              tensorFoldlSpec.go (· + ·) n s valuesSq n acc = acc := by
            simpa using
              (tensor_foldl_spec_go_of_not_lt (f := (· + ·)) (values := valuesSq) (k := n) (acc := acc)
                (by simp))
          simp [hgo]
        | succ m ih_go =>
          have hlt : k < n := by
            have : 0 < n - k := by simp [hn]
            exact Nat.sub_pos_iff_lt.mp this
          have hk1 : k + 1 ≤ n := Nat.succ_le_of_lt hlt
          rw [tensor_foldl_spec_go_of_lt (f := (· + ·)) (values := valuesSq) (k := k) (acc := acc) hlt]
          -- `acc ≤ acc + term` and then apply IH to the remainder.
          have hstep :
              tensorFoldlSpec (· + ·) acc (valuesSq ⟨k, hlt⟩) =
                acc + sumSpec (valuesSq ⟨k, hlt⟩) := by
            simpa using
              (tensor_foldl_spec_add_init (s := s) (acc := acc) (t := valuesSq ⟨k, hlt⟩))
          have hacc_le :
              acc ≤ tensorFoldlSpec (· + ·) acc (valuesSq ⟨k, hlt⟩) := by
            have hterm : 0 ≤ sumSpec (valuesSq ⟨k, hlt⟩) := term_nonneg ⟨k, hlt⟩
            -- `acc ≤ acc + term`
            have : acc ≤ acc + sumSpec (valuesSq ⟨k, hlt⟩) :=
              le_add_of_nonneg_right (a := acc) hterm
            simpa [hstep] using this
          have h_next : n - (k + 1) = m := by
            rw [Nat.sub_succ, hn]
            rfl
          have hrest :=
            ih_go (k := k + 1)
              (acc := tensorFoldlSpec (· + ·) acc (valuesSq ⟨k, hlt⟩)) hk1
          -- `acc ≤ newAcc ≤ go (k+1) newAcc`.
          exact le_trans hacc_le (by simpa [h_next] using hrest)

      -- Main equivalence.
      constructor
      · -- → direction: if the sum of squares is 0, all components are 0.
        intro h0
        -- Unfold `tensor_norm_squared` to get a statement about `go`.
        have hgo0 :
            tensorFoldlSpec.go (· + ·) n s valuesSq 0 0 = 0 := by
          simpa [tensorNormSquared, dot, sumSpec, mulSpec, map2Spec, tensorFoldlSpec,
            valuesSq] using h0

        -- Prove each `values i` is zero by iterating the `go` loop.
        have go_all_zero :
            ∀ k acc, k ≤ n → 0 ≤ acc →
              tensorFoldlSpec.go (· + ·) n s valuesSq k acc = 0 →
                acc = 0 ∧ ∀ i : Fin n, i.val ≥ k → values i = fill (0 : ℝ) s := by
          intro k acc hk hacc hgo
          induction hn : n - k generalizing k acc with
          | zero =>
            have hk' : k = n := by
              have : n ≤ k := Nat.sub_eq_zero_iff_le.mp hn
              exact Nat.le_antisymm hk this
            subst k
            -- loop stops: go n acc = acc
            have hgo_stop : acc = 0 := by
              simpa
                [tensor_foldl_spec_go_of_not_lt (f := (· + ·)) (values := valuesSq) (k := n) (acc := acc)
                  (by simp)]
                using hgo
            exact ⟨hgo_stop, by intro i hi; exfalso; exact Nat.not_lt_of_ge hi i.isLt⟩
          | succ m ih_go =>
            have hlt : k < n := by
              have : 0 < n - k := by simp [hn]
              exact Nat.sub_pos_iff_lt.mp this
            have hk1 : k + 1 ≤ n := Nat.succ_le_of_lt hlt
            -- Peel one loop step from the hypothesis `hgo`.
            have hgo_step :
                tensorFoldlSpec.go (· + ·) n s valuesSq (k + 1)
                    (tensorFoldlSpec (· + ·) acc (valuesSq ⟨k, hlt⟩))
                  = 0 := by
              simpa
                [tensor_foldl_spec_go_of_lt (f := (· + ·)) (values := valuesSq) (k := k) (acc := acc) hlt]
                using hgo
            -- New accumulator after processing index `k`.
            let nextAcc := tensorFoldlSpec (· + ·) acc (valuesSq ⟨k, hlt⟩)
            have hstep :
                nextAcc = acc + sumSpec (valuesSq ⟨k, hlt⟩) := by
              -- expand `nextAcc` and use `tensor_foldl_spec_add_init`
              simp [nextAcc, tensor_foldl_spec_add_init (s := s) (acc := acc) (t := valuesSq ⟨k,
                hlt⟩)]
            -- `go (k+1) nextAcc = 0`
            have hgo' : tensorFoldlSpec.go (· + ·) n s valuesSq (k + 1) nextAcc = 0 := by
              simpa [nextAcc] using hgo_step
            -- `nextAcc ≤ go (k+1) nextAcc = 0`
            have hnext_le0 : nextAcc ≤ 0 := by
              have hge := go_ge (k := k + 1) (acc := nextAcc) hk1
              exact le_trans hge (le_of_eq hgo')
            -- `0 ≤ nextAcc`
            have hterm : 0 ≤ sumSpec (valuesSq ⟨k, hlt⟩) := term_nonneg ⟨k, hlt⟩
            have hnext_ge0 : 0 ≤ nextAcc := by
              -- nextAcc = acc + term
              have : 0 ≤ acc + sumSpec (valuesSq ⟨k, hlt⟩) := add_nonneg hacc hterm
              simpa [hstep] using this
            -- hence `nextAcc = 0`
            have hnext0 : nextAcc = 0 := le_antisymm hnext_le0 hnext_ge0
            -- From `nextAcc = acc + term`, deduce `acc = 0` and `term = 0`.
            have hacc0 : acc = 0 := by
              have : acc ≤ nextAcc := by
                -- acc ≤ acc + term
                have : acc ≤ acc + sumSpec (valuesSq ⟨k, hlt⟩) := le_add_of_nonneg_right hterm
                simpa [hstep] using this
              exact le_antisymm (le_trans this (le_of_eq hnext0)) hacc
            have hterm0 : sumSpec (valuesSq ⟨k, hlt⟩) = 0 := by
              -- rewrite `nextAcc = acc + term` and use `acc = 0`, `nextAcc = 0`
              have : acc + sumSpec (valuesSq ⟨k, hlt⟩) = 0 := by simpa [hstep] using hnext0
              simpa [hacc0] using this
            -- Now apply IH on the remainder with `acc = 0` and `nextAcc = 0`.
            have h_next : n - (k + 1) = m := by
              rw [Nat.sub_succ, hn]
              rfl
            have ih_res :=
              (ih_go (k := k + 1) (acc := nextAcc) hk1 (by simp [hnext0]) hgo') h_next
            rcases ih_res with ⟨_acc0, htail⟩
            -- Deduce `values ⟨k, hlt⟩ = fill 0` from `term = 0` and IH on the inner shape.
            have hhead : values ⟨k, hlt⟩ = fill (0 : ℝ) s := by
              -- `term = tensor_norm_squared (values k)`:
              have : tensorNormSquared (values ⟨k, hlt⟩) = 0 := by
                simpa [valuesSq, tensorNormSquared, dot] using hterm0
              exact (ih (t := values ⟨k, hlt⟩)).1 this
            refine ⟨hacc0, ?_⟩
            intro i hi
            -- Split on whether `i.val = k` or `i.val ≥ k+1`.
            have hcase : i.val = k ∨ i.val ≥ k + 1 := by
              have hk' : k = i.val ∨ k < i.val := Nat.eq_or_lt_of_le hi
              cases hk' with
              | inl hk_eq => exact Or.inl hk_eq.symm
              | inr hk_lt => exact Or.inr (Nat.succ_le_of_lt hk_lt)
            cases hcase with
            | inl hk' =>
              -- `i = ⟨k, _⟩`
              have : i = ⟨k, hlt⟩ := by
                apply Fin.ext
                exact hk'
              subst this
              exact hhead
            | inr hge =>
              exact htail i hge

        have hall := go_all_zero (k := 0) (acc := (0 : ℝ)) (by exact Nat.zero_le n) (by simp) hgo0
        rcases hall with ⟨_acc0, hvals⟩
        -- Rebuild the tensor from pointwise equalities.
        apply congrArg Tensor.dim
        funext i
        exact hvals i (by exact Nat.zero_le _)
      · -- ← direction: the zero tensor has zero norm-squared.
        intro ht0
        rw [ht0]
        -- `fill 0` has `tensor_norm_squared = 0` because it contains only zeros.
        let innerZero : Tensor ℝ s := fill (0 : ℝ) s

        -- Inner zero tensor has zero norm-squared by IH.
        have hnorm_inner : tensorNormSquared innerZero = 0 :=
          (ih (t := innerZero)).2 rfl

        -- And elementwise multiplication preserves `fill 0`.
        have hm_inner : mulSpec innerZero innerZero = innerZero := by
          simpa [innerZero] using (mul_spec_fill_zero (s := s))

        -- Hence `sum_spec innerZero = 0`.
        have hsum_inner : sumSpec innerZero = 0 := by
          simpa [tensorNormSquared, dot, hm_inner] using hnorm_inner

        -- Outer elementwise product is also `fill 0`.
        have hm_outer :
            mulSpec (fill (0 : ℝ) (Shape.dim n s)) (fill (0 : ℝ) (Shape.dim n s)) =
              fill (0 : ℝ) (Shape.dim n s) := by
          simpa using (mul_spec_fill_zero (s := Shape.dim n s))

        -- Show `sum_spec (fill 0 (dim n s)) = 0` by iterating the `go` loop.
        have go_zero :
            ∀ k, k ≤ n →
              tensorFoldlSpec.go (· + ·) n s (fun _ : Fin n => innerZero) k 0 = 0 := by
          intro k hk
          induction hn : n - k generalizing k with
          | zero =>
            have hk' : k = n := by
              have : n ≤ k := Nat.sub_eq_zero_iff_le.mp hn
              exact Nat.le_antisymm hk this
            subst k
            simpa using
              (tensor_foldl_spec_go_of_not_lt (f := (· + ·)) (values := fun _ : Fin n => innerZero)
                (k := n) (acc := (0 : ℝ)) (by simp))
          | succ m ih_go =>
            have hlt : k < n := by
              have : 0 < n - k := by simp [hn]
              exact Nat.sub_pos_iff_lt.mp this
            have hk1 : k + 1 ≤ n := Nat.succ_le_of_lt hlt
            rw [tensor_foldl_spec_go_of_lt (f := (· + ·)) (values := fun _ : Fin n => innerZero) (k := k)
              (acc := (0 : ℝ)) hlt]
            -- `tensor_foldl_spec (+) 0 innerZero = 0` since `sum_spec innerZero = 0`.
            have hstep0 : tensorFoldlSpec (· + ·) 0 innerZero = 0 := by
              simpa [hsum_inner] using
                (tensor_foldl_spec_add_init (s := s) (acc := (0 : ℝ)) (t := innerZero))
            -- Reduce to the tail and apply IH.
            have h_next : n - (k + 1) = m := by
              rw [Nat.sub_succ, hn]
              rfl
            have := ih_go (k := k + 1) hk1
            simpa [hstep0, h_next] using this

        have hsum_outer : sumSpec (fill (0 : ℝ) (Shape.dim n s)) = 0 := by
          -- unfold `sum_spec`/`tensor_foldl_spec` and apply `go_zero` at `k=0`.
          simpa [sumSpec, tensorFoldlSpec, fill, innerZero] using
            (go_zero (k := 0) (by exact Nat.zero_le n))

        -- Put it together: `tensor_norm_squared = dot = sum_spec (mul_spec _)`.
        simp [tensorNormSquared, dot, hm_outer, hsum_outer]

/-! ## Extensionality and structural algebra -/

/-- Tensor extensionality over a generic element type: equal `getSpec` views imply equal tensors. -/
theorem tensor_ext {α : Type} {s : Shape} {x y : Tensor α s} :
  (∀ idxs : List Nat, getSpec x idxs = getSpec y idxs) → x = y := by
  intro h
  -- do induction on the shape, generalizing x and y so ih can be applied to sub-tensors
  induction s with
  | scalar =>
    -- both x and y must be Tensor.scalar a, Tensor.scalar b
    cases x with
    | scalar a =>
      cases y with
      | scalar b =>
        -- use the index [] to get equality of stored scalars
        have : getSpec (Tensor.scalar a) [] = getSpec (Tensor.scalar b) [] := h []
        have hab : a = b := by
          simpa using this
        simp [hab]

  | dim n s ih =>
    -- x and y must be Tensor.dim fx and Tensor.dim fy
    cases x with
    | dim fx =>
      cases y with
      | dim fy =>
        -- if dimension is zero, both are dim 0 s and are equal by definition
        by_cases hn : n = 0
        · -- n = 0
          -- both constructors are `Tensor.dim` with first argument `Fin 0 → _`
          -- there is only one inhabitant of `Fin 0 → _` so fx = fy definitionally;
          -- we can finish by refl
          have : fx = fy := by
            apply funext
            intro i
            rw[hn] at i
            exact Fin.elim0 i  -- impossible, Fin 0 has no elements
          rw [this]

        · -- n = Nat.succ n'
          -- show pointwise equality of the component functions fx and fy
          have pointwise : ∀ i : Fin n, fx i = fy i := by
            intro i
            -- to apply ih, we must show get_spec (fx i) idxs = get_spec (fy i) idxs for all idxs
            apply ih
            intro idxs
            -- by the computation rule for get_spec on `Tensor.dim`, we can rewrite
            -- get_spec (fx i) idxs  = get_spec (Tensor.dim fx) (i.val :: idxs)
            -- and similarly for fy; then use the hypothesis `h`
            calc
              getSpec (fx i) idxs = getSpec (Tensor.dim fx) (i.val :: idxs) := by
                simp [i.isLt]
              _ = getSpec (Tensor.dim fy) (i.val :: idxs) := by
                apply h
              _ = getSpec (fy i) idxs := by
                simp [i.isLt]

          -- now lift pointwise equality of the functions to equality of functions
          have eq_funcs : fx = fy := by
            apply funext
            intro i
            exact pointwise i

          -- rewrite and finish
          rw [eq_funcs]
/-- Elementwise addition is associative (over ℝ tensors). -/
theorem add_spec_assoc {s : Shape}
  (a b c : Tensor ℝ s) :
  addSpec (addSpec a b) c = addSpec a (addSpec b c) := by
  -- Structural recursion on `s` and use associativity of ℝ addition
  induction s with
  | scalar =>
    cases a; cases b; cases c
    simp [addSpec, map2Spec]
    ring  -- Uses associativity of real addition
  | dim n s ih =>
    cases a with | dim fa =>
    cases b with | dim fb =>
    cases c with | dim fc =>
    simp [addSpec, map2Spec]
    funext i
    exact ih (fa i) (fb i) (fc i)

/-- Elementwise subtraction distributes over addition on the right. -/
theorem sub_spec_add_right {s : Shape}
  (a b c : Tensor ℝ s) :
  subSpec a (addSpec b c) = addSpec (subSpec a b) (negSpec c) := by
  -- Expand definitions using map2_spec and use ring properties
  induction s with
  | scalar =>
    cases a; cases b; cases c
    simp [subSpec, addSpec, negSpec, map2Spec, mapSpec]
    ring  -- Uses distributivity: a - (b + c) = (a - b) + (-c)
  | dim n s ih =>
    cases a with | dim fa =>
    cases b with | dim fb =>
    cases c with | dim fc =>
    simp [subSpec, addSpec, negSpec, map2Spec, mapSpec]
    funext i
    exact ih (fa i) (fb i) (fc i)

/-- Elementwise multiplication distributes over addition on the right. -/
theorem mul_spec_add_right {s : Shape}
  (a b c : Tensor ℝ s) :
  mulSpec a (addSpec b c) = addSpec (mulSpec a b) (mulSpec a c) := by
  -- Structural recursion on `s` and use distributivity of ℝ
  induction s with
  | scalar =>
    cases a; cases b; cases c
    simp [mulSpec, addSpec, map2Spec]
    ring  -- Uses distributivity of real multiplication
  | dim n s ih =>
    cases a with | dim fa =>
    cases b with | dim fb =>
    cases c with | dim fc =>
    simp [mulSpec, addSpec, map2Spec]
    funext i
    exact ih (fa i) (fb i) (fc i)

/-- Elementwise multiplication distributes over addition on the left. -/
theorem mul_spec_add_left {s : Shape}
  (a b c : Tensor ℝ s) :
  mulSpec (addSpec a b) c = addSpec (mulSpec a c) (mulSpec b c) := by
  -- Structural recursion on `s` and use distributivity of ℝ
  induction s with
  | scalar =>
    cases a; cases b; cases c
    simp [mulSpec, addSpec, map2Spec]
    ring  -- Uses distributivity of real multiplication
  | dim n s ih =>
    cases a with | dim fa =>
    cases b with | dim fb =>
    cases c with | dim fc =>
    simp [mulSpec, addSpec, map2Spec]
    funext i
    exact ih (fa i) (fb i) (fc i)

/-- Bias cancellation for tensor subtraction: `(a + c) - (b + c) = a - b`. -/
theorem sub_spec_bias_cancel {s : Shape} (a b c : Tensor ℝ s) :
  subSpec (addSpec a c) (addSpec b c) = subSpec a b := by
  -- Key lemma: (a + c) - (b + c) = a - b
  induction s with
  | scalar =>
    cases a; cases b; cases c
    simp [subSpec, addSpec, map2Spec]
  | dim n s ih =>
    cases a with | dim fa =>
    cases b with | dim fb =>
    cases c with | dim fc =>
    simp [subSpec, addSpec, map2Spec]
    funext i
    exact ih (fa i) (fb i) (fc i)

/-- Linearity of matrix-vector multiplication in the vector argument (addition). -/
theorem mat_vec_add {m n : Nat}
  (W : Tensor ℝ (.dim m (.dim n .scalar)))
  (x y : Tensor ℝ (.dim n .scalar)) :
  matVecMulSpec W (addSpec x y) =
  addSpec (matVecMulSpec W x) (matVecMulSpec W y) := by
  classical
  have hToVec :
      toVec (matVecMulSpec W (addSpec x y)) =
        toVec (addSpec (matVecMulSpec W x) (matVecMulSpec W y)) := by
    funext i
    -- Rewrite all mat-vec outputs as sums.
    rw [toVec_mat_vec_mul_spec (A := W) (v := addSpec x y) (i := i)]
    -- Expand the elementwise addition on the right (without unfolding `toVec` itself).
    simp [toVec_add_spec]
    rw [toVec_mat_vec_mul_spec (A := W) (v := x) (i := i)]
    rw [toVec_mat_vec_mul_spec (A := W) (v := y) (i := i)]
    -- Distribute `*` over `+` inside the sum and split the sum.
    simp [mul_add, Finset.sum_add_distrib]

  have hTensor :
      ofVec (toVec (matVecMulSpec W (addSpec x y))) =
        ofVec (toVec (addSpec (matVecMulSpec W x) (matVecMulSpec W y))) :=
    congrArg ofVec hToVec

  -- `ofVec ∘ toVec` is identity.
  simpa using
    (Eq.trans (ofVec_toVec (t := matVecMulSpec W (addSpec x y))).symm
      (Eq.trans hTensor (ofVec_toVec (t := addSpec (matVecMulSpec W x) (matVecMulSpec W
        y)))))

/-- Linearity of matrix-vector multiplication in the vector argument (scaling). -/
theorem mat_vec_scale {m n : Nat}
  (W : Tensor ℝ (.dim m (.dim n .scalar)))
  (x : Tensor ℝ (.dim n .scalar)) (c : ℝ) :
  matVecMulSpec W (scaleSpec x c) =
  scaleSpec (matVecMulSpec W x) c := by
  classical
  have hToVec :
      toVec (matVecMulSpec W (scaleSpec x c)) =
        toVec (scaleSpec (matVecMulSpec W x) c) := by
    funext i
    rw [toVec_mat_vec_mul_spec (A := W) (v := scaleSpec x c) (i := i)]
    -- `toVec (scale_spec _ c)` is pointwise scaling.
    simp [toVec_scale_spec]
    rw [toVec_mat_vec_mul_spec (A := W) (v := x) (i := i)]
    -- Pull out the scalar `c` from the sum.
    -- (Reassociate `*` so `Finset.sum_mul` applies.)
    have hassoc :
        (∑ k : Fin n, get2 W i k * (toVec x k * c)) =
          ∑ k : Fin n, (get2 W i k * toVec x k) * c := by
      refine Finset.sum_congr rfl ?_
      intro k _
      ring
    -- Now use `Finset.sum_mul` to factor `c` to the right.
    -- (`Finset.sum_mul` gives the reverse direction, so use symmetry.)
    simpa [hassoc, mul_assoc] using
      (Finset.sum_mul (s := (Finset.univ : Finset (Fin n)))
        (f := fun k : Fin n => get2 W i k * toVec x k) (a := c)).symm

  have hTensor :
      ofVec (toVec (matVecMulSpec W (scaleSpec x c))) =
        ofVec (toVec (scaleSpec (matVecMulSpec W x) c)) :=
    congrArg ofVec hToVec

  simpa using
    (Eq.trans (ofVec_toVec (t := matVecMulSpec W (scaleSpec x c))).symm
      (Eq.trans hTensor (ofVec_toVec (t := scaleSpec (matVecMulSpec W x) c))))

/-- Full linearity of matrix-vector multiplication in the vector argument. -/
theorem mat_vec_linear_combination {m n : Nat}
  (W : Tensor ℝ (.dim m (.dim n .scalar)))
  (x y : Tensor ℝ (.dim n .scalar)) (a b : ℝ) :
  matVecMulSpec W (addSpec (scaleSpec x a) (scaleSpec y b)) =
  addSpec (scaleSpec (matVecMulSpec W x) a)
           (scaleSpec (matVecMulSpec W y) b) := by
  -- Combine mat_vec_add and mat_vec_scale
  rw [mat_vec_add, mat_vec_scale, mat_vec_scale]

/--
Simplification lemmas for common patterns.
Useful for automated proof tactics.
-/
@[simp]
theorem get_dim_scalar {n : Nat} (f : Fin n → Tensor ℝ .scalar) (i : Fin n) :
  get (Tensor.dim f) i = f i := by rfl

/-- `toScalar` is the identity on scalar tensors. -/
@[simp]
theorem toScalar_scalar (x : ℝ) :
  toScalar (Tensor.scalar x) = x := by rfl

/-- Mapping `0 + ·` over an `Option` is the identity. -/
lemma option_zero_add (o : Option ℝ) : o.map (fun x => 0 + x) = o := by
  cases o
  · rfl
  · simp [zero_add]

-- add_spec with zero tensor on the left
/-- Left identity for `add_spec`: adding the all-zero tensor does nothing. -/
@[simp]
theorem add_spec_zero_left {s : Shape} : ∀ (t : Tensor ℝ s),
  addSpec (fill 0 s) t = t
| Tensor.scalar a => by simp [addSpec, map2Spec, fill, zero_add]
| Tensor.dim fx => by
    simp [addSpec, map2Spec, fill]
    apply funext
    intro i
    -- recursively call theorem on component fx i
    exact add_spec_zero_left (fx i)

-- add_spec with zero tensor on the right
/-- Right identity for `add_spec`: adding the all-zero tensor does nothing. -/
@[simp]
theorem add_spec_zero_right {s : Shape} : ∀ (t : Tensor ℝ s),
  addSpec t (fill (0 : ℝ) s) = t
  | Tensor.scalar a => by simp [addSpec, map2Spec, fill]
  | Tensor.dim fx => by
    simp [addSpec, map2Spec, fill]
    apply funext
    intro i
    -- recursively call theorem on component fx i
    exact add_spec_zero_right (fx i)

-- mul_spec with one tensor on the left
/-- Left identity for `mul_spec`: multiplying by the all-ones tensor does nothing. -/
@[simp]
theorem mul_spec_one_left {s : Shape} : ∀ (t : Tensor ℝ s),
  mulSpec (fill (1 : ℝ) s) t = t
| Tensor.scalar a => by
    -- scalar case: 1 * a = a
    simp [mulSpec, map2Spec, fill]
| Tensor.dim fx => by
    -- dim case: function extensionality over components
    simp [mulSpec, map2Spec, fill]
    apply funext
    intro i
    -- recursively apply theorem to component
    exact mul_spec_one_left (fx i)

-- mul_spec with one tensor on the right
/-- Right identity for `mul_spec`: multiplying by the all-ones tensor does nothing. -/
@[simp]
theorem mul_spec_one_right {s : Shape} : ∀ (t : Tensor ℝ s),
  mulSpec t (fill (1 : ℝ) s) = t
  | Tensor.scalar a => by
    -- scalar case: a * 1 = a
    simp [mulSpec, map2Spec, fill]
  | Tensor.dim fx => by
      -- dim case: function extensionality over components
      simp [mulSpec, map2Spec, fill]
      apply funext
      intro i
      -- recursively apply theorem to component
      exact mul_spec_one_right (fx i)

end Spec
