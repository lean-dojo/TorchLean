/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Proofs.Tensor.Basic
public import NN.Proofs.Utils.List
public import NN.Spec.Layers.Conv
public import NN.Spec.Layers.Utils

public import Mathlib.Algebra.BigOperators.Ring.Finset
public import Mathlib.Data.Fintype.BigOperators

/-!
# BackwardDot

`Conv2D` backward dot-level correctness (bridge lemma).

The runtime autograd engine (`NN.Runtime.Autograd.Engine`) computes Conv2D gradients via the
handwritten spec:

* `Spec.conv2d_backward_spec` (kernel/bias/input)

For the **analytic** spec-level theorem over `ℝ`, Conv2D is already covered via `fderiv`/adjoints in:

* `NN.Proofs.Autograd.Tape.Ops.Conv.FDeriv`

That file provides a proof-only Conv2D node whose VJP is `(fderiv forward)†`, so any DAG using it
is covered by the global theorem `Graph.backpropVec_eq_adjoint_fderiv`.

What remains here is the dot/adjointness bridge:

  dot (JVP_conv2d …) δ = dot dKernel gK + dot dBias gB + dot dInput gX

where `(gK, gB, gX) = Spec.conv2d_backward_spec … δ`.

The padding-related rewrites needed for the input-gradient proof are factored into:

* `NN/Spec/Layers/Utils.lean` (`get_at_or_zero_pad_multi_channel*` lemmas)
-/

@[expose] public section


namespace Proofs
namespace Autograd
namespace Conv2D

open Spec
open Tensor

open scoped BigOperators

noncomputable section

-- This file performs large but routine finite-sum rearrangements; give the kernel-bridge proofs
-- enough heartbeats to avoid brittle timeouts during `lake build`.
set_option maxHeartbeats 12000000

-- ---------------------------------------------------------------------------
-- Small helpers
-- ---------------------------------------------------------------------------

private lemma dot_add_right {s : Shape} (a b c : Tensor ℝ s) :
    dot a (addSpec b c) = dot a b + dot a c := by
  -- `dot` is symmetric; reduce to `dot_add_left`.
  calc
    dot a (addSpec b c) = dot (addSpec b c) a := by
      simpa using (dot_comm (a := a) (b := addSpec b c))
    _ = dot b a + dot c a := by
      simpa using (dot_add_left (a := b) (b := c) (c := a))
    _ = dot a b + dot a c := by
      simp [dot_comm]

private lemma dot_dim {n : Nat} {s : Shape} (a b : Tensor ℝ (.dim n s)) :
    dot a b = ∑ i : Fin n, dot (get a i) (get b i) := by
  classical
  cases a with
  | dim fa =>
    cases b with
    | dim fb =>
      unfold dot
      have hsum := (Spec.sum_spec_dim (t := mulSpec (Tensor.dim fa) (Tensor.dim fb)))
      -- `mul_spec` is pointwise on `.dim`, so each slice is `mul_spec (fa i) (fb i)`.
      simpa [dot, mulSpec, map2Spec, get_eq] using hsum

private lemma dot_scalar (x y : ℝ) :
    dot (Tensor.scalar x) (Tensor.scalar y) = x * y := by
  simp [dot, sumSpec, tensorFoldlSpec, mulSpec, map2Spec]

private lemma dot_vec_eq_sum_get
    {n : Nat} (a b : Tensor ℝ (.dim n .scalar)) :
    dot a b = ∑ i : Fin n, (getAtOrZero a [i.val]) * (getAtOrZero b [i.val]) := by
  classical
  cases a with
  | dim fa =>
    cases b with
    | dim fb =>
      -- Use `dot_dim` to reduce to a sum of scalar dots.
      rw [dot_dim (a := Tensor.dim fa) (b := Tensor.dim fb)]
      refine Finset.sum_congr rfl ?_
      intro i _
      cases ha : fa i with
      | scalar x =>
        cases hb : fb i with
        | scalar y =>
          -- Each summand is scalar multiplication.
          have hdot : dot (Tensor.scalar x) (Tensor.scalar y) = x * y := dot_scalar x y
          -- And `get_at_or_zero` reads that scalar at index `i`.
          have hreadA : getAtOrZero (Tensor.dim fa) [i.val] = x := by
            simp [i.isLt, ha]
          have hreadB : getAtOrZero (Tensor.dim fb) [i.val] = y := by
            simp [i.isLt, hb]
          simp [get_eq, ha, hb, hdot, hreadA, hreadB]

@[simp] private lemma get_at_or_zero_tensor_cast {s t : Shape} (h : s = t) (x : Tensor ℝ s)
    (idx : List Nat) :
    getAtOrZero (Tensor.castShape (t := x) h) idx = getAtOrZero x idx := by
  cases h
  rfl

private lemma get2_eq_get_at_or_zero
    {m n : Nat} (A : Tensor ℝ (.dim m (.dim n .scalar))) (i : Fin m) (j : Fin n) :
    get2 A i j = getAtOrZero A [i.val, j.val] := by
  cases A with
  | dim rows =>
    cases hrow : rows i with
    | dim cols =>
      cases hcell : cols j with
      | scalar v =>
        simp [Spec.get2, get_eq, i.isLt, j.isLt, hrow, hcell]

private lemma get_at_or_zero_get_channel
    {C H W : Nat} (t : Tensor ℝ (.dim C (.dim H (.dim W .scalar))))
    (c : Fin C) (i : Fin H) (j : Fin W) :
    getAtOrZero (get t c) [i.val, j.val] = getAtOrZero t [c.val, i.val, j.val] := by
  cases t with
  | dim fC =>
    simp [get_eq, c.isLt]

-- ---------------------------------------------------------------------------
-- Bias broadcast dot lemma
-- ---------------------------------------------------------------------------

/-- Broadcast a bias gradient `outC` across spatial axes into an `outC×outH×outW` tensor. -/
def biasBroadcast {outC outH outW : Nat} (db : Tensor ℝ (.dim outC .scalar)) :
    Tensor ℝ (.dim outC (.dim outH (.dim outW .scalar))) :=
  Tensor.dim (fun oc =>
    Tensor.dim (fun _i =>
      Tensor.dim (fun _j =>
        Tensor.scalar (getAtOrZero db [oc.val]))))

private lemma conv2d_bias_deriv_get
    {inC outC kH kW stride padding inH inW : Nat}
    {h1 : inC ≠ 0} {h2 : kH ≠ 0} {h3 : kW ≠ 0}
    (layer : Spec.Conv2DSpec inC outC kH kW stride padding ℝ h1 h2 h3)
    (input : Spec.MultiChannelImage inC inH inW ℝ)
    (δ : Spec.MultiChannelImage outC
      ((inH + 2 * padding - kH) / stride + 1)
      ((inW + 2 * padding - kW) / stride + 1) ℝ)
    (oc : Fin outC) :
    getAtOrZero (Spec.conv2dBiasDerivSpec (α := ℝ) (layer := layer) (input := input)
      (grad_output := δ)) [oc.val]
      =
    ∑ i : Fin ((inH + 2 * padding - kH) / stride + 1),
      ∑ j : Fin ((inW + 2 * padding - kW) / stride + 1),
        getAtOrZero δ [oc.val, i.val, j.val] := by
  classical
  unfold Spec.conv2dBiasDerivSpec
  -- Unfold the two nested `finRange` folds into `Finset.univ` sums.
  -- First, rewrite the inner fold (over `out_w`) as `acc + ∑ j`.
  simp only []
  -- After unfolding, the value is a scalar tensor, so `get_at_or_zero .. []` extracts the scalar.
  -- Convert the inner fold and then the outer fold.
  have houtW :
      ∀ (i : Fin ((inH + 2 * padding - kH) / stride + 1)) (acc : ℝ),
        (List.finRange ((inW + 2 * padding - kW) / stride + 1)).foldl
            (fun acc j => acc + getAtOrZero δ [oc.val, i.val, j.val]) acc
          =
        acc + ∑ j : Fin ((inW + 2 * padding - kW) / stride + 1), getAtOrZero δ [oc.val, i.val,
          j.val] := by
    intro i acc
    -- Use `foldl_add_init` then `Spec.finRange_foldl_add_eq_finset_sum`.
    have h1 :=
      List.foldl_add_init (l := List.finRange ((inW + 2 * padding - kW) / stride + 1))
        (f := fun j : Fin ((inW + 2 * padding - kW) / stride + 1) => getAtOrZero δ [oc.val,
          i.val, j.val])
        (acc := acc)
    -- Convert the `0`-initialized fold into a `Finset` sum.
    have h2 :
        (List.finRange ((inW + 2 * padding - kW) / stride + 1)).foldl
            (fun s j => s + getAtOrZero δ [oc.val, i.val, j.val]) 0
          =
        ∑ j : Fin ((inW + 2 * padding - kW) / stride + 1), getAtOrZero δ [oc.val, i.val, j.val]
          := by
      simpa using
        (Spec.finRange_foldl_add_eq_finset_sum (f := fun j : Fin ((inW + 2 * padding - kW) / stride
          + 1) =>
        getAtOrZero δ [oc.val, i.val, j.val]))
    simpa [h2] using h1

  -- Now handle the outer fold over `out_h`.
  have houtH :
      (List.finRange ((inH + 2 * padding - kH) / stride + 1)).foldl
          (fun acc (i : Fin ((inH + 2 * padding - kH) / stride + 1)) =>
            (List.finRange ((inW + 2 * padding - kW) / stride + 1)).foldl
                (fun acc j => acc + getAtOrZero δ [oc.val, i.val, j.val]) acc)
          0
        =
      ∑ i : Fin ((inH + 2 * padding - kH) / stride + 1),
        ∑ j : Fin ((inW + 2 * padding - kW) / stride + 1), getAtOrZero δ [oc.val, i.val, j.val]
          := by
    -- Rewrite the step function using `houtW`, then convert the resulting fold to a sum.
    -- First show the fold is equivalent to `foldl (fun acc i => acc + innerSum i) 0`.
    have hstep :
        (fun acc (i : Fin ((inH + 2 * padding - kH) / stride + 1)) =>
          (List.finRange ((inW + 2 * padding - kW) / stride + 1)).foldl
              (fun acc j => acc + getAtOrZero δ [oc.val, i.val, j.val]) acc)
        =
        (fun acc (i : Fin ((inH + 2 * padding - kH) / stride + 1)) =>
          acc + ∑ j : Fin ((inW + 2 * padding - kW) / stride + 1), getAtOrZero δ [oc.val, i.val,
            j.val]) := by
      funext acc i
      simpa using (houtW i acc)
    -- Convert.
    simp [hstep, Spec.finRange_foldl_add_eq_finset_sum]

  -- Finish by `simp`ing the definition back to the fold we rewrote (and normalizing away the
  -- `tensor_cast` inserted by `conv2d_bias_deriv_spec`).
  simpa [Spec.convBiasDerivSpec, Spec.Private.foldlIndices, Spec.convOutSpatial, Spec.convOutDim,
    Vector.get, Vector.toList, oc.isLt] using houtH

private lemma dot_biasBroadcast_eq_dot_bias_deriv
    {inC outC kH kW stride padding inH inW : Nat}
    {h1 : inC ≠ 0} {h2 : kH ≠ 0} {h3 : kW ≠ 0}
    (layer : Spec.Conv2DSpec inC outC kH kW stride padding ℝ h1 h2 h3)
    (input : Spec.MultiChannelImage inC inH inW ℝ)
    (db : Tensor ℝ (.dim outC .scalar))
    (δ : Spec.MultiChannelImage outC
      ((inH + 2 * padding - kH) / stride + 1)
      ((inW + 2 * padding - kW) / stride + 1) ℝ) :
    dot (biasBroadcast
          (outC := outC)
          (outH := ((inH + 2 * padding - kH) / stride + 1))
          (outW := ((inW + 2 * padding - kW) / stride + 1))
          db) δ
      =
    dot db (Spec.conv2dBiasDerivSpec (α := ℝ) (layer := layer) (input := input) (grad_output :=
      δ)) := by
  classical
  -- Expand the LHS dot on the 3D output as `∑ oc, dot (sliceA oc) (sliceδ oc)`,
  -- then expand each matrix dot with `dot_mat_eq_sum`.
  have hdot3 :
      dot (biasBroadcast
            (outC := outC)
            (outH := ((inH + 2 * padding - kH) / stride + 1))
            (outW := ((inW + 2 * padding - kW) / stride + 1))
            db) δ
        =
      ∑ oc : Fin outC,
        ∑ i : Fin ((inH + 2 * padding - kH) / stride + 1),
          ∑ j : Fin ((inW + 2 * padding - kW) / stride + 1),
            (getAtOrZero db [oc.val]) * (getAtOrZero δ [oc.val, i.val, j.val]) := by
    -- Outer dimension over channels.
    rw [dot_dim (a := biasBroadcast (outC := outC)
            (outH := ((inH + 2 * padding - kH) / stride + 1))
            (outW := ((inW + 2 * padding - kW) / stride + 1)) db) (b := δ)]
    refine Finset.sum_congr rfl ?_
    intro oc _
    -- Matrix dot on the spatial slice.
    have hmat :=
      Spec.dot_mat_eq_sum (m := ((inH + 2 * padding - kH) / stride + 1))
        (n := ((inW + 2 * padding - kW) / stride + 1))
        (A := get (biasBroadcast (outC := outC)
          (outH := ((inH + 2 * padding - kH) / stride + 1))
          (outW := ((inW + 2 * padding - kW) / stride + 1)) db) oc)
        (B := get δ oc)
    -- Rewrite `get2` to `get_at_or_zero`, and simplify the broadcast slice entry.
    -- `get2 (get (biasBroadcast db) oc) i j = db[oc]`.
    -- `get2 (get δ oc) i j = δ[oc,i,j]`.
    -- Rewrite the slice `get2` in terms of `get_at_or_zero`, then lift it back to the 3D index.
    -- `get δ oc` is the `oc`-th channel slice.
    have hδ :
        ∀ (i : Fin ((inH + 2 * padding - kH) / stride + 1)) (j : Fin ((inW + 2 * padding - kW) /
          stride + 1)),
          getAtOrZero (get δ oc) [i.val, j.val] = getAtOrZero δ [oc.val, i.val, j.val] := by
      intro i j
      simpa using (get_at_or_zero_get_channel (t := δ) (c := oc) (i := i) (j := j))
    -- Now rewrite each summand.
    refine (hmat.trans ?_ )
    -- Rewrite the double sum term-by-term.
    refine Finset.sum_congr rfl ?_
    intro i _
    refine Finset.sum_congr rfl ?_
    intro j _
    -- Convert `get2` to `get_at_or_zero` on both slices.
    have hA2 :
        get2 (get (biasBroadcast (outC := outC)
          (outH := ((inH + 2 * padding - kH) / stride + 1))
          (outW := ((inW + 2 * padding - kW) / stride + 1)) db) oc) i j
          =
        getAtOrZero db [oc.val] := by
      -- `biasBroadcast` is constant across the spatial indices.
      -- First rewrite `get2` to a `get_at_or_zero` on the matrix slice.
      have h := get2_eq_get_at_or_zero
        (A := get (biasBroadcast (outC := outC)
          (outH := ((inH + 2 * padding - kH) / stride + 1))
          (outW := ((inW + 2 * padding - kW) / stride + 1)) db) oc) i j
      -- Then simplify the broadcast slice read.
      -- (All indices are in bounds, so `get_at_or_zero` takes the `if_pos` branches.)
      simpa [biasBroadcast, get_eq, oc.isLt, i.isLt, j.isLt] using h
    have hB2 :
        get2 (get δ oc) i j = getAtOrZero δ [oc.val, i.val, j.val] := by
      have h := get2_eq_get_at_or_zero (A := get δ oc) i j
      -- Lift slice read back to the 3D index using `hδ`.
      simpa [hδ i j] using h
    -- Finish.
    simp [hA2, hB2]

  -- Expand the RHS dot on the bias vector using `dot_vec_eq_sum`.
  have hdot1 :
      dot db (Spec.conv2dBiasDerivSpec (α := ℝ) (layer := layer) (input := input) (grad_output
        := δ))
        =
      ∑ oc : Fin outC,
        (getAtOrZero db [oc.val]) *
          (getAtOrZero (Spec.conv2dBiasDerivSpec (α := ℝ) (layer := layer) (input := input)
            (grad_output := δ)) [oc.val]) := by
    simpa using
      (dot_vec_eq_sum_get (a := db)
        (b := Spec.conv2dBiasDerivSpec (α := ℝ) (layer := layer) (input := input) (grad_output
          := δ)))

  -- Rewrite each bias-deriv coordinate as the spatial sum, then compare with `hdot3`.
  calc
    dot (biasBroadcast (outC := outC)
          (outH := ((inH + 2 * padding - kH) / stride + 1))
          (outW := ((inW + 2 * padding - kW) / stride + 1)) db) δ
        =
      ∑ oc : Fin outC,
        ∑ i : Fin ((inH + 2 * padding - kH) / stride + 1),
          ∑ j : Fin ((inW + 2 * padding - kW) / stride + 1),
            (getAtOrZero db [oc.val]) * (getAtOrZero δ [oc.val, i.val, j.val]) := hdot3
    _ =
      ∑ oc : Fin outC,
        (getAtOrZero db [oc.val]) *
          (∑ i : Fin ((inH + 2 * padding - kH) / stride + 1),
            ∑ j : Fin ((inW + 2 * padding - kW) / stride + 1),
              getAtOrZero δ [oc.val, i.val, j.val]) := by
        simp [Finset.mul_sum]
    _ =
      ∑ oc : Fin outC,
        (getAtOrZero db [oc.val]) *
          (getAtOrZero (Spec.conv2dBiasDerivSpec (α := ℝ) (layer := layer) (input := input)
            (grad_output := δ)) [oc.val]) := by
        refine Finset.sum_congr rfl ?_
        intro oc _
        simp [conv2d_bias_deriv_get (layer := layer) (input := input) (δ := δ) (oc := oc)]
    _ = dot db (Spec.conv2dBiasDerivSpec (α := ℝ) (layer := layer) (input := input)
      (grad_output := δ)) := by
        simp [hdot1]

-- ---------------------------------------------------------------------------
-- Dot expansions for 3D/4D tensors in terms of `get_at_or_zero`
-- ---------------------------------------------------------------------------

private lemma dot3_eq_sum
    {C H W : Nat}
    (a b : Tensor ℝ (.dim C (.dim H (.dim W .scalar)))) :
    dot a b =
      ∑ c : Fin C, ∑ i : Fin H, ∑ j : Fin W,
        (getAtOrZero a [c.val, i.val, j.val]) * (getAtOrZero b [c.val, i.val, j.val]) := by
  classical
  -- Outer dimension.
  rw [dot_dim (a := a) (b := b)]
  refine Finset.sum_congr rfl ?_
  intro c _
  -- Matrix dot on the slice.
  have hmat := Spec.dot_mat_eq_sum (m := H) (n := W) (A := get a c) (B := get b c)
  -- Rewrite each `get2` as `get_at_or_zero`, and lift slice reads back to 3D.
  refine hmat.trans ?_
  refine Finset.sum_congr rfl ?_
  intro i _
  refine Finset.sum_congr rfl ?_
  intro j _
  have hA : get2 (get a c) i j = getAtOrZero a [c.val, i.val, j.val] := by
    -- `get2` -> `get_at_or_zero` on the slice, then lift to 3D.
    have h1 := get2_eq_get_at_or_zero (A := get a c) i j
    have h2 : getAtOrZero (get a c) [i.val, j.val] = getAtOrZero a [c.val, i.val, j.val] :=
      (get_at_or_zero_get_channel (t := a) (c := c) (i := i) (j := j))
    simpa [h2] using h1
  have hB : get2 (get b c) i j = getAtOrZero b [c.val, i.val, j.val] := by
    have h1 := get2_eq_get_at_or_zero (A := get b c) i j
    have h2 : getAtOrZero (get b c) [i.val, j.val] = getAtOrZero b [c.val, i.val, j.val] :=
      (get_at_or_zero_get_channel (t := b) (c := c) (i := i) (j := j))
    simpa [h2] using h1
  simp [hA, hB]

private lemma get_at_or_zero_get_outer3
    {OC IC KH KW : Nat}
    (k : Tensor ℝ (.dim OC (.dim IC (.dim KH (.dim KW .scalar)))))
    (oc : Fin OC) (ic : Fin IC) (di : Fin KH) (dj : Fin KW) :
    getAtOrZero (get k oc) [ic.val, di.val, dj.val] =
      getAtOrZero k [oc.val, ic.val, di.val, dj.val] := by
  cases k with
  | dim fOC =>
    simp [get_eq, oc.isLt]

private lemma dot4_eq_sum
    {OC IC KH KW : Nat}
    (a b : Tensor ℝ (.dim OC (.dim IC (.dim KH (.dim KW .scalar))))) :
    dot a b =
      ∑ oc : Fin OC, ∑ ic : Fin IC, ∑ di : Fin KH, ∑ dj : Fin KW,
        (getAtOrZero a [oc.val, ic.val, di.val, dj.val]) *
          (getAtOrZero b [oc.val, ic.val, di.val, dj.val]) := by
  classical
  -- Outer OC dimension.
  rw [dot_dim (a := a) (b := b)]
  refine Finset.sum_congr rfl ?_
  intro oc _
  -- Apply the 3D lemma to the slice.
  have hs := dot3_eq_sum (a := get a oc) (b := get b oc)
  -- Lift slice reads back to 4D.
  -- The RHS of `hs` is `∑ ic, ∑ di, ∑ dj, get_at_or_zero (get a oc) [...] * ...`.
  -- Rewrite those reads to 4D indices using `get_at_or_zero_get_outer3`.
  simpa [get_at_or_zero_get_outer3] using hs

-- ---------------------------------------------------------------------------
-- Fold-to-sum helper for the Conv2D specs
-- ---------------------------------------------------------------------------

private lemma finRange_foldl_add_acc {n : Nat} (f : Fin n → ℝ) (acc : ℝ) :
    (List.finRange n).foldl (fun s i => s + f i) acc = acc + ∑ i : Fin n, f i := by
  -- Use `foldl_add_init` to peel off the accumulator, then convert the `0`-fold to a sum.
  have h1 :=
    List.foldl_add_init (l := List.finRange n) (f := f) (acc := acc)
  have h2 : (List.finRange n).foldl (fun s i => s + f i) 0 = ∑ i : Fin n, f i := by
    simpa using Spec.finRange_foldl_add_eq_finset_sum (f := f)
  simpa [h2] using h1

private lemma sum_mul {ι : Type} [Fintype ι] (f : ι → ℝ) (a : ℝ) :
    (∑ i : ι, f i) * a = ∑ i : ι, f i * a := by
  classical
  simpa using (Finset.sum_mul (s := (Finset.univ : Finset ι)) (f := f) a)

private lemma mul_sum {ι : Type} [Fintype ι] (a : ℝ) (f : ι → ℝ) :
    a * (∑ i : ι, f i) = ∑ i : ι, a * f i := by
  classical
  simpa using (Finset.mul_sum (s := (Finset.univ : Finset ι)) (f := f) a)

private lemma sum_comm {α β : Type} [Fintype α] [Fintype β] (f : α → β → ℝ) :
    (∑ a : α, ∑ b : β, f a b) = ∑ b : β, ∑ a : α, f a b := by
  classical
  simpa using (Finset.sum_comm (s := (Finset.univ : Finset α)) (t := (Finset.univ : Finset β)) (f :=
    f))

-- ---------------------------------------------------------------------------
-- Conv2D padding helper (matches the spec's `if padding = 0 then cast else pad`)
-- ---------------------------------------------------------------------------

/-!
`paddedInput` matches the forward spec’s padding convention:
- if `padding = 0`, it is just a shape cast; and
- otherwise it is `pad_multi_channel`.

PyTorch analogue: `torch.nn.functional.pad` (or implicit padding inside `conv2d`).
https://pytorch.org/docs/stable/generated/torch.nn.functional.pad.html
https://pytorch.org/docs/stable/generated/torch.nn.functional.conv2d.html
-/

def paddedInput {inC inH inW padding : Nat}
    (input : Spec.MultiChannelImage inC inH inW ℝ) :
    Spec.MultiChannelImage inC (inH + 2 * padding) (inW + 2 * padding) ℝ :=
  if h4 : padding = 0 then
    tensorCast
      (Shape.dim inC (Shape.dim (inH + 2 * padding) (Shape.dim (inW + 2 * padding) .scalar)))
      (by simp; rw [h4])
      input
  else
    padMultiChannel input padding

lemma get_at_or_zero_paddedInput
    {inC inH inW padding : Nat}
    (img : Spec.MultiChannelImage inC inH inW ℝ) (c : Fin inC) (p q : Nat) :
    getAtOrZero (paddedInput (inC := inC) (inH := inH) (inW := inW) (padding := padding) img)
      [c.val, p, q]
      =
    (if _h : p < padding ∨ q < padding then (0 : ℝ) else getAtOrZero img [c.val, p - padding, q -
      padding]) := by
  classical
  by_cases h0 : padding = 0
  · subst h0
    -- No padding: the cast is definitional and the `if` condition is always false.
    simp [paddedInput]
  · -- Positive padding: reduce to `get_at_or_zero_pad_multi_channel`.
    simpa [paddedInput, h0] using
      (Spec.get_at_or_zero_pad_multi_channel (α := ℝ) (img := img) (c := c) (p := p) (q := q)
        (padding := padding))

private lemma mkInputIdx_match_eq_paddedInput
    {inC inH inW stride padding : Nat}
    (img : Spec.MultiChannelImage inC inH inW ℝ) (c : Fin inC)
    (oi di oj dj : Nat) :
    (match Private.mkInputIdx? [oi, oj] [di, dj] [stride, stride] [padding, padding] with
      | none => (0 : ℝ)
      | some inIdx => getAtOrZero img (c.val :: inIdx))
      =
    getAtOrZero (paddedInput (inC := inC) (inH := inH) (inW := inW) (padding := padding) img)
      [c.val, oi * stride + di, oj * stride + dj] := by
  classical
  -- Expand the RHS using the explicit padded-input read formula.
  rw [get_at_or_zero_paddedInput (img := img) (c := c) (p := oi * stride + di) (q := oj * stride + dj)]
  by_cases h0 : oi * stride + di < padding
  · simp [Private.mkInputIdx?, h0]
  · by_cases h1 : oj * stride + dj < padding
    · simp [Private.mkInputIdx?, h0, h1]
    · simp [Private.mkInputIdx?, h0, h1]

private lemma sum_shift_eq_paddedInput
    {inC inH inW padding : Nat}
    (x : Spec.MultiChannelImage inC inH inW ℝ) (ic : Fin inC) (p q : Nat) :
    (∑ i : Fin inH, ∑ j : Fin inW,
        if (p = i.val + padding ∧ q = j.val + padding) then getAtOrZero x [ic.val, i.val, j.val]
          else 0)
      =
    getAtOrZero (paddedInput (inC := inC) (inH := inH) (inW := inW) (padding := padding) x)
      [ic.val, p, q] := by
  classical
  -- Compare both sides to the explicit `paddedInput` read formula.
  rw [get_at_or_zero_paddedInput (img := x) (c := ic) (p := p) (q := q)]
  by_cases hp : p < padding ∨ q < padding
  · -- In the padded border: the conjunction can never hold, so the sum is `0`.
    cases hp with
    | inl hp' =>
        have hpOr : p < padding ∨ q < padding := Or.inl hp'
        have hne : ∀ i : Fin inH, p ≠ i.val + padding := by
          intro i
          have hle : padding ≤ i.val + padding := by
            simp [Nat.add_comm]
          exact Nat.ne_of_lt (lt_of_lt_of_le hp' hle)
        have hfalse : ∀ i : Fin inH, ∀ j : Fin inW, ¬(p = i.val + padding ∧ q = j.val + padding) :=
          by
          intro i j h
          exact hne i h.1
        simp [hpOr, hfalse]
    | inr hq' =>
        have hpOr : p < padding ∨ q < padding := Or.inr hq'
        have hne : ∀ j : Fin inW, q ≠ j.val + padding := by
          intro j
          have hle : padding ≤ j.val + padding := by
            simp [Nat.add_comm]
          exact Nat.ne_of_lt (lt_of_lt_of_le hq' hle)
        have hfalse : ∀ i : Fin inH, ∀ j : Fin inW, ¬(p = i.val + padding ∧ q = j.val + padding) :=
          by
          intro i j h
          exact hne j h.2
        simp [hpOr, hfalse]
  · -- Interior: reduce to a shifted read of `x` and collapse the sums using `Fintype.sum_ite_eq'`.
    have hpP : padding ≤ p := by
      have : ¬ p < padding := by
        intro hlt
        exact hp (Or.inl hlt)
      exact Nat.le_of_not_gt this
    have hqP : padding ≤ q := by
      have : ¬ q < padding := by
        intro hlt
        exact hp (Or.inr hlt)
      exact Nat.le_of_not_gt this
    let vp : Nat := p - padding
    let vq : Nat := q - padding
    by_cases hvp : vp < inH
    · by_cases hvq : vq < inW
      · let i0 : Fin inH := ⟨vp, hvp⟩
        let j0 : Fin inW := ⟨vq, hvq⟩
        have hip : p = i0.val + padding := by
          simp [i0, vp, Nat.sub_add_cancel hpP]
        have hjq : q = j0.val + padding := by
          simp [j0, vq, Nat.sub_add_cancel hqP]
        have hiff :
            ∀ (i : Fin inH) (j : Fin inW),
              (p = i.val + padding ∧ q = j.val + padding) ↔ (i = i0 ∧ j = j0) := by
          intro i j
          constructor
          · intro h
            have hiVal : vp = i.val := by
              have := congrArg (fun t => t - padding) h.1
              simpa [vp, Nat.add_sub_cancel] using this
            have hjVal : vq = j.val := by
              have := congrArg (fun t => t - padding) h.2
              simpa [vq, Nat.add_sub_cancel] using this
            have hi : i = i0 := by
              apply Fin.ext
              simpa [i0] using hiVal.symm
            have hj : j = j0 := by
              apply Fin.ext
              simpa [j0] using hjVal.symm
            exact ⟨hi, hj⟩
          · rintro ⟨rfl, rfl⟩
            exact ⟨hip, hjq⟩
        have hrewrite :
            (∑ i : Fin inH, ∑ j : Fin inW,
                if (p = i.val + padding ∧ q = j.val + padding) then getAtOrZero x [ic.val, i.val,
                  j.val] else 0)
              =
            (∑ i : Fin inH, ∑ j : Fin inW,
                if (i = i0 ∧ j = j0) then getAtOrZero x [ic.val, i0.val, j0.val] else 0) := by
          refine Fintype.sum_congr _ _ ?_
          intro i
          refine Fintype.sum_congr _ _ ?_
          intro j
          by_cases hij : i = i0 ∧ j = j0
          · have : p = i.val + padding ∧ q = j.val + padding := (hiff i j).2 hij
            rcases hij with ⟨rfl, rfl⟩
            simp [this]
          · have : ¬(p = i.val + padding ∧ q = j.val + padding) := by
              intro h
              exact hij ((hiff i j).1 h)
            simp [hij, this]
        have hsum :
            (∑ i : Fin inH, ∑ j : Fin inW,
                if (i = i0 ∧ j = j0) then getAtOrZero x [ic.val, i0.val, j0.val] else 0)
              =
            getAtOrZero x [ic.val, i0.val, j0.val] := by
          -- Split the conjunction and use `Fintype.sum_ite_eq'` twice.
          calc
            (∑ i : Fin inH, ∑ j : Fin inW,
                  if (i = i0 ∧ j = j0) then getAtOrZero x [ic.val, i0.val, j0.val] else 0)
                =
              ∑ i : Fin inH,
                if i = i0 then (∑ j : Fin inW, if j = j0 then getAtOrZero x [ic.val, i0.val,
                  j0.val] else 0) else 0 := by
                  refine Fintype.sum_congr _ _ ?_
                  intro i
                  by_cases hi : i = i0 <;> simp [hi]
            _ =
              ∑ i : Fin inH, if i = i0 then getAtOrZero x [ic.val, i0.val, j0.val] else 0 := by
                  refine Fintype.sum_congr _ _ ?_
                  intro i
                  by_cases hi : i = i0 <;> simp [hi]
            _ = getAtOrZero x [ic.val, i0.val, j0.val] := by
                  simp
        -- Finish this case.
        simp [hp, vp, vq, i0, j0, hrewrite, hsum]
      · -- `vq` out of bounds: both sides are `0`.
        have rhs0 : getAtOrZero x [ic.val, vp, vq] = 0 := by
          cases x with
          | dim fC =>
            cases hrow : fC ic with
            | dim fH =>
              cases hcell : fH ⟨vp, hvp⟩ with
              | dim fW =>
                simp [ic.isLt, hvp, hvq, vp, vq, hrow, hcell]
        have hvq' : inW ≤ vq := Nat.le_of_not_gt hvq
        have hfalse : ∀ i : Fin inH, ∀ j : Fin inW, ¬(p = i.val + padding ∧ q = j.val + padding) :=
          by
          intro i j h
          have : vq = j.val := by
            have := congrArg (fun t => t - padding) h.2
            simpa [vq, Nat.add_sub_cancel] using this
          exact (Nat.not_lt_of_ge hvq') (this ▸ j.isLt)
        simp [hp, rhs0, hfalse, vp, vq]
    · -- `vp` out of bounds: both sides are `0`.
      have rhs0 : getAtOrZero x [ic.val, vp, vq] = 0 := by
        cases x with
        | dim fC =>
          cases hrow : fC ic with
          | dim fH =>
            simp [ic.isLt, hvp, vp, vq, hrow]
      have hvp' : inH ≤ vp := Nat.le_of_not_gt hvp
      have hfalse : ∀ i : Fin inH, ∀ j : Fin inW, ¬(p = i.val + padding ∧ q = j.val + padding) := by
        intro i j h
        have : vp = i.val := by
          have := congrArg (fun t => t - padding) h.1
          simpa [vp, Nat.add_sub_cancel] using this
        exact (Nat.not_lt_of_ge hvp') (this ▸ i.isLt)
      simp [hp, rhs0, hfalse, vp, vq]

/-!
Output shape helpers (no dilation): these are the standard “convolution arithmetic” formulas.
They are kept as definitions so later statements can share the same expression.
-/

def outH (inH kH stride padding : Nat) : Nat :=
  (inH + 2 * padding - kH) / stride + 1

/-- Output width helper (no dilation). -/
def outW (inW kW stride padding : Nat) : Nat :=
  (inW + 2 * padding - kW) / stride + 1

-- ---------------------------------------------------------------------------
-- Kernel bridge: dot(conv(dKernel, x), δ) = dot(dKernel, ∂L/∂kernel)
-- ---------------------------------------------------------------------------

lemma conv2d_spec_noBias_get
    {inC outC kH kW stride padding inH inW : Nat}
    {h1 : inC ≠ 0} {h2 : kH ≠ 0} {h3 : kW ≠ 0}
    (dKernel : Tensor ℝ (.dim outC (.dim inC (.dim kH (.dim kW .scalar)))))
    (input : Spec.MultiChannelImage inC inH inW ℝ)
    (oc : Fin outC) (i : Fin (outH inH kH stride padding)) (j : Fin (outW inW kW stride padding)) :
    let layerK : Spec.Conv2DSpec inC outC kH kW stride padding ℝ h1 h2 h3 :=
      { kernel := dKernel, bias := fill (0 : ℝ) (.dim outC .scalar) }
    getAtOrZero (Spec.conv2dSpec (α := ℝ) (layer := layerK) input) [oc.val, i.val, j.val]
      =
    ∑ ic : Fin inC,
      ∑ di : Fin kH,
        ∑ dj : Fin kW,
          getAtOrZero dKernel [oc.val, ic.val, di.val, dj.val] *
            getAtOrZero (paddedInput (inC := inC) (inH := inH) (inW := inW) (padding := padding)
              input)
              [ic.val, i.val * stride + di.val, j.val * stride + dj.val] := by
  intro layerK
  classical
  unfold Spec.conv2dSpec
  -- Peel the requested output entry and convert the nested `finRange` folds into `Finset` sums.
  simp [layerK, fill, getAtOrZero, oc.isLt, finRange_foldl_add_acc]
  -- Discharge the bounds checks introduced by `getAtOrZero` (these follow from `i.isLt` / `j.isLt`).
  have hi : (i : Nat) ≤ (inH + 2 * padding - kH) / stride := by
    simpa [outH] using (Nat.lt_succ_iff.mp i.isLt)
  have hj : (j : Nat) ≤ (inW + 2 * padding - kW) / stride := by
    simpa [outW] using (Nat.lt_succ_iff.mp j.isLt)
  simp [hi, hj]
  -- Rewrite the `mkInputIdx?`-based read into the `paddedInput` helper, then commute the product
  -- so the summand matches the statement (`kernel * paddedInput`).
  refine Finset.sum_congr rfl ?_
  intro ic _
  refine Finset.sum_congr rfl ?_
  intro di _
  refine Finset.sum_congr rfl ?_
  intro dj _
  have hread :=
    mkInputIdx_match_eq_paddedInput (stride := stride) (padding := padding) (img := input) (c := ic)
      (oi := i.val) (di := di.val) (oj := j.val) (dj := dj.val)
  -- Expand the match using `hread`, then commute the scalar product so the summand matches the
  -- statement’s `kernel * paddedInput` order.
  calc
    (match Private.mkInputIdx? [i.val, j.val] [di.val, dj.val] [stride, stride] [padding, padding] with
        | none => 0
        | some inIdx => getAtOrZero input (ic.val :: inIdx)) *
        getAtOrZero dKernel [oc.val, ic.val, di.val, dj.val]
        =
      getAtOrZero
          (paddedInput (inC := inC) (inH := inH) (inW := inW) (padding := padding) input)
          [ic.val, i.val * stride + di.val, j.val * stride + dj.val] *
        getAtOrZero dKernel [oc.val, ic.val, di.val, dj.val] := by
      simp [hread]
    _ =
      getAtOrZero dKernel [oc.val, ic.val, di.val, dj.val] *
        getAtOrZero
          (paddedInput (inC := inC) (inH := inH) (inW := inW) (padding := padding) input)
          [ic.val, i.val * stride + di.val, j.val * stride + dj.val] := by
      simpa using
        (mul_comm
          (getAtOrZero
            (paddedInput (inC := inC) (inH := inH) (inW := inW) (padding := padding) input)
            [ic.val, i.val * stride + di.val, j.val * stride + dj.val])
          (getAtOrZero dKernel [oc.val, ic.val, di.val, dj.val]))

private lemma conv2d_kernel_deriv_get
    {inC outC kH kW stride padding inH inW : Nat}
    {h1 : inC ≠ 0} {h2 : kH ≠ 0} {h3 : kW ≠ 0}
    (layer : Spec.Conv2DSpec inC outC kH kW stride padding ℝ h1 h2 h3)
    (input : Spec.MultiChannelImage inC inH inW ℝ)
    (δ : Spec.MultiChannelImage outC (outH inH kH stride padding) (outW inW kW stride padding) ℝ)
    (oc : Fin outC) (ic : Fin inC) (di : Fin kH) (dj : Fin kW) :
    getAtOrZero (Spec.conv2dKernelDerivSpec (α := ℝ) (layer := layer) (input := input)
      (grad_output := δ))
        [oc.val, ic.val, di.val, dj.val]
      =
    ∑ i : Fin (outH inH kH stride padding),
      ∑ j : Fin (outW inW kW stride padding),
        getAtOrZero (paddedInput (inC := inC) (inH := inH) (inW := inW) (padding := padding)
          input)
          [ic.val, i.val * stride + di.val, j.val * stride + dj.val] *
          getAtOrZero δ [oc.val, i.val, j.val] := by
  classical
  unfold Spec.conv2dKernelDerivSpec
  simp (config := { maxSteps := 2000000 })
    [finRange_foldl_add_acc, outH, outW]
  apply Finset.sum_congr
  · rfl
  intro i _
  apply Finset.sum_congr
  · rfl
  intro j _
  have hread :=
    mkInputIdx_match_eq_paddedInput (stride := stride) (padding := padding) (img := input)
    (c := ic) (oi := i.val) (di := di.val) (oj := j.val) (dj := dj.val)
  exact congrArg (fun x => x * getAtOrZero δ [oc.val, i.val, j.val]) hread

private lemma conv2d_input_deriv_get
    {inC outC kH kW stride padding inH inW : Nat}
    {h1 : inC ≠ 0} {h2 : kH ≠ 0} {h3 : kW ≠ 0}
    (layer : Spec.Conv2DSpec inC outC kH kW stride padding ℝ h1 h2 h3)
    (input : Spec.MultiChannelImage inC inH inW ℝ)
    (δ : Spec.MultiChannelImage outC (outH inH kH stride padding) (outW inW kW stride padding) ℝ)
    (ic : Fin inC) (i : Fin inH) (j : Fin inW) :
    getAtOrZero (Spec.conv2dInputDerivSpec (α := ℝ) (layer := layer) (input := input)
      (grad_output := δ))
        [ic.val, i.val, j.val]
      =
    ∑ oc : Fin outC,
      ∑ oi : Fin (outH inH kH stride padding),
        ∑ oj : Fin (outW inW kW stride padding),
          ∑ di : Fin kH,
            ∑ dj : Fin kW,
              (if _h :
                  (oi.val * stride + di.val = i.val + padding) ∧
                  (oj.val * stride + dj.val = j.val + padding) then
                getAtOrZero δ [oc.val, oi.val, oj.val] *
                  getAtOrZero layer.kernel [oc.val, ic.val, di.val, dj.val]
              else
                0) := by
  classical
  unfold Spec.conv2dInputDerivSpec
  simp (config := { maxSteps := 2000000 })
    [outH, outW, ic.isLt, i.isLt, j.isLt, finRange_foldl_add_acc, add_comm, mul_comm]
  rfl

private lemma dot_conv2d_kernel
    {inC outC kH kW stride padding inH inW : Nat}
    {h1 : inC ≠ 0} {h2 : kH ≠ 0} {h3 : kW ≠ 0}
    (layer : Spec.Conv2DSpec inC outC kH kW stride padding ℝ h1 h2 h3)
    (input : Spec.MultiChannelImage inC inH inW ℝ)
    (dKernel : Tensor ℝ (.dim outC (.dim inC (.dim kH (.dim kW .scalar)))))
    (δ : Spec.MultiChannelImage outC (outH inH kH stride padding) (outW inW kW stride padding) ℝ) :
    let layerK : Spec.Conv2DSpec inC outC kH kW stride padding ℝ h1 h2 h3 :=
      { kernel := dKernel, bias := fill (0 : ℝ) (.dim outC .scalar) }
    dot (Spec.conv2dSpec (α := ℝ) (layer := layerK) input) δ
      =
    dot dKernel (Spec.conv2dKernelDerivSpec (α := ℝ) (layer := layer) (input := input)
      (grad_output := δ)) := by
  intro layerK
  classical
  -- Expand both dots into explicit sums and show they coincide.
  -- LHS: sum over output indices.
  rw [dot3_eq_sum (a := Spec.conv2dSpec (α := ℝ) (layer := layerK) input) (b := δ)]
  -- Make the binder sizes explicit using the local `outH/outW` abbreviations so we can rewrite with
  -- `hL`.
  change
      (∑ oc : Fin outC, ∑ i : Fin (outH inH kH stride padding), ∑ j : Fin (outW inW kW stride
        padding),
          getAtOrZero (Spec.conv2dSpec (α := ℝ) (layer := layerK) input) [oc.val, i.val, j.val]
            *
            getAtOrZero δ [oc.val, i.val, j.val])
        =
      dot dKernel (Spec.conv2dKernelDerivSpec (α := ℝ) (layer := layer) (input := input)
        (grad_output := δ))
  -- Rewrite the conv output entries via the no-bias formula.
  have hL :
      (∑ oc : Fin outC, ∑ i : Fin (outH inH kH stride padding), ∑ j : Fin (outW inW kW stride
        padding),
          getAtOrZero (Spec.conv2dSpec (α := ℝ) (layer := layerK) input) [oc.val, i.val, j.val]
            *
            getAtOrZero δ [oc.val, i.val, j.val])
        =
      (∑ oc : Fin outC, ∑ i : Fin (outH inH kH stride padding), ∑ j : Fin (outW inW kW stride
        padding),
          (∑ ic : Fin inC, ∑ di : Fin kH, ∑ dj : Fin kW,
              getAtOrZero dKernel [oc.val, ic.val, di.val, dj.val] *
                getAtOrZero (paddedInput (inC := inC) (inH := inH) (inW := inW) (padding :=
                  padding) input)
                  [ic.val, i.val * stride + di.val, j.val * stride + dj.val]) *
            getAtOrZero δ [oc.val, i.val, j.val]) := by
    refine Finset.sum_congr rfl ?_
    intro oc _
    refine Finset.sum_congr rfl ?_
    intro i _
    refine Finset.sum_congr rfl ?_
    intro j _
    -- Avoid `simp` here: it may cancel the common `* δ` factor and introduce a disjunction.
    have hEntry :=
      (by
        simpa [layerK] using
          (conv2d_spec_noBias_get (h1 := h1) (h2 := h2) (h3 := h3)
            (dKernel := dKernel) (input := input) (oc := oc) (i := i) (j := j)))
    -- Rewrite the conv entry, then close by reflexivity.
    simpa [mul_assoc] using congrArg (fun x => x * getAtOrZero δ [oc.val, i.val, j.val]) hEntry
  rw [hL]
  -- Push the output cotangent inside and reassociate the sums.
  -- This makes the expression match the kernel-gradient dot expansion.
  have hReorder :
      (∑ oc : Fin outC, ∑ i : Fin (outH inH kH stride padding), ∑ j : Fin (outW inW kW stride
        padding),
          (∑ ic : Fin inC, ∑ di : Fin kH, ∑ dj : Fin kW,
              getAtOrZero dKernel [oc.val, ic.val, di.val, dj.val] *
                getAtOrZero (paddedInput (inC := inC) (inH := inH) (inW := inW) (padding :=
                  padding) input)
                  [ic.val, i.val * stride + di.val, j.val * stride + dj.val]) *
            getAtOrZero δ [oc.val, i.val, j.val])
        =
      (∑ oc : Fin outC, ∑ ic : Fin inC, ∑ di : Fin kH, ∑ dj : Fin kW,
          getAtOrZero dKernel [oc.val, ic.val, di.val, dj.val] *
            (∑ i : Fin (outH inH kH stride padding), ∑ j : Fin (outW inW kW stride padding),
              getAtOrZero (paddedInput (inC := inC) (inH := inH) (inW := inW) (padding :=
                padding) input)
                [ic.val, i.val * stride + di.val, j.val * stride + dj.val] *
                getAtOrZero δ [oc.val, i.val, j.val])) := by
    classical
    -- Expand to a 6-index sum, commute sums, then factor the kernel entry out of the `(i,j)` sums.
    let G (oc : Fin outC) (i : Fin (outH inH kH stride padding)) (j : Fin (outW inW kW stride
      padding))
        (ic : Fin inC) (di : Fin kH) (dj : Fin kW) : ℝ :=
      (getAtOrZero dKernel [oc.val, ic.val, di.val, dj.val] *
          getAtOrZero (paddedInput (inC := inC) (inH := inH) (inW := inW) (padding := padding)
            input)
            [ic.val, i.val * stride + di.val, j.val * stride + dj.val]) *
        getAtOrZero δ [oc.val, i.val, j.val]

    have hMul3
        (oc : Fin outC) (i : Fin (outH inH kH stride padding)) (j : Fin (outW inW kW stride
          padding)) :
        (∑ ic : Fin inC, ∑ di : Fin kH, ∑ dj : Fin kW,
              getAtOrZero dKernel [oc.val, ic.val, di.val, dj.val] *
                getAtOrZero (paddedInput (inC := inC) (inH := inH) (inW := inW) (padding :=
                  padding) input)
                  [ic.val, i.val * stride + di.val, j.val * stride + dj.val]) *
            getAtOrZero δ [oc.val, i.val, j.val]
          =
        ∑ ic : Fin inC, ∑ di : Fin kH, ∑ dj : Fin kW, G oc i j ic di dj := by
      -- Distribute the final `* δ` into the nested sums.
      -- First over `ic`, then `di`, then `dj`.
      have hIc :
          (∑ ic : Fin inC,
                (∑ di : Fin kH, ∑ dj : Fin kW,
                      getAtOrZero dKernel [oc.val, ic.val, di.val, dj.val] *
                        getAtOrZero (paddedInput (inC := inC) (inH := inH) (inW := inW) (padding
                          := padding) input)
                          [ic.val, i.val * stride + di.val, j.val * stride + dj.val])) *
              getAtOrZero δ [oc.val, i.val, j.val]
            =
          ∑ ic : Fin inC,
              ((∑ di : Fin kH, ∑ dj : Fin kW,
                    getAtOrZero dKernel [oc.val, ic.val, di.val, dj.val] *
                      getAtOrZero (paddedInput (inC := inC) (inH := inH) (inW := inW) (padding :=
                        padding) input)
                        [ic.val, i.val * stride + di.val, j.val * stride + dj.val]) *
                getAtOrZero δ [oc.val, i.val, j.val]) := by
        -- `sum_mul` over `ic`
        simpa [mul_assoc] using
          (sum_mul (ι := Fin inC)
            (f := fun ic =>
              ∑ di : Fin kH, ∑ dj : Fin kW,
                getAtOrZero dKernel [oc.val, ic.val, di.val, dj.val] *
                  getAtOrZero (paddedInput (inC := inC) (inH := inH) (inW := inW) (padding :=
                    padding) input)
                    [ic.val, i.val * stride + di.val, j.val * stride + dj.val])
            (a := getAtOrZero δ [oc.val, i.val, j.val]))
      -- Now distribute inside each `ic`.
      calc
        (∑ ic : Fin inC, ∑ di : Fin kH, ∑ dj : Fin kW,
              getAtOrZero dKernel [oc.val, ic.val, di.val, dj.val] *
                getAtOrZero (paddedInput (inC := inC) (inH := inH) (inW := inW) (padding :=
                  padding) input)
                  [ic.val, i.val * stride + di.val, j.val * stride + dj.val]) *
            getAtOrZero δ [oc.val, i.val, j.val]
            =
          ∑ ic : Fin inC,
              ((∑ di : Fin kH, ∑ dj : Fin kW,
                    getAtOrZero dKernel [oc.val, ic.val, di.val, dj.val] *
                      getAtOrZero (paddedInput (inC := inC) (inH := inH) (inW := inW) (padding :=
                        padding) input)
                        [ic.val, i.val * stride + di.val, j.val * stride + dj.val]) *
                getAtOrZero δ [oc.val, i.val, j.val]) := hIc
        _ =
          ∑ ic : Fin inC, ∑ di : Fin kH,
              ((∑ dj : Fin kW,
                    getAtOrZero dKernel [oc.val, ic.val, di.val, dj.val] *
                      getAtOrZero (paddedInput (inC := inC) (inH := inH) (inW := inW) (padding :=
                        padding) input)
                        [ic.val, i.val * stride + di.val, j.val * stride + dj.val]) *
                getAtOrZero δ [oc.val, i.val, j.val]) := by
          refine Fintype.sum_congr _ _ ?_
          intro ic
          simpa [mul_assoc] using
            (sum_mul (ι := Fin kH)
              (f := fun di =>
                ∑ dj : Fin kW,
                  getAtOrZero dKernel [oc.val, ic.val, di.val, dj.val] *
                    getAtOrZero (paddedInput (inC := inC) (inH := inH) (inW := inW) (padding :=
                      padding) input)
                      [ic.val, i.val * stride + di.val, j.val * stride + dj.val])
              (a := getAtOrZero δ [oc.val, i.val, j.val]))
        _ =
          ∑ ic : Fin inC, ∑ di : Fin kH, ∑ dj : Fin kW,
              (getAtOrZero dKernel [oc.val, ic.val, di.val, dj.val] *
                  getAtOrZero (paddedInput (inC := inC) (inH := inH) (inW := inW) (padding :=
                    padding) input)
                    [ic.val, i.val * stride + di.val, j.val * stride + dj.val]) *
                getAtOrZero δ [oc.val, i.val, j.val] := by
          refine Fintype.sum_congr _ _ ?_
          intro ic
          refine Fintype.sum_congr _ _ ?_
          intro di
          simpa [mul_assoc] using
            (sum_mul (ι := Fin kW)
              (f := fun dj =>
                getAtOrZero dKernel [oc.val, ic.val, di.val, dj.val] *
                  getAtOrZero (paddedInput (inC := inC) (inH := inH) (inW := inW) (padding :=
                    padding) input)
                    [ic.val, i.val * stride + di.val, j.val * stride + dj.val])
              (a := getAtOrZero δ [oc.val, i.val, j.val]))
        _ = ∑ ic : Fin inC, ∑ di : Fin kH, ∑ dj : Fin kW, G oc i j ic di dj := by
          rfl

    -- LHS: rewrite using `hMul3`, then commute sums to match the RHS's ordering.
    have hLHS :
        (∑ oc : Fin outC, ∑ i : Fin (outH inH kH stride padding), ∑ j : Fin (outW inW kW stride
          padding),
              (∑ ic : Fin inC, ∑ di : Fin kH, ∑ dj : Fin kW,
                    getAtOrZero dKernel [oc.val, ic.val, di.val, dj.val] *
                      getAtOrZero (paddedInput (inC := inC) (inH := inH) (inW := inW) (padding :=
                        padding) input)
                        [ic.val, i.val * stride + di.val, j.val * stride + dj.val]) *
                getAtOrZero δ [oc.val, i.val, j.val])
          =
        (∑ oc : Fin outC, ∑ ic : Fin inC, ∑ di : Fin kH, ∑ dj : Fin kW,
              ∑ i : Fin (outH inH kH stride padding), ∑ j : Fin (outW inW kW stride padding), G oc i
                j ic di dj) := by
      refine Fintype.sum_congr _ _ ?_
      intro oc
      -- Apply `hMul3` pointwise under the `(i,j)` sums.
      have h1 :
          (∑ i : Fin (outH inH kH stride padding), ∑ j : Fin (outW inW kW stride padding),
                (∑ ic : Fin inC, ∑ di : Fin kH, ∑ dj : Fin kW,
                      getAtOrZero dKernel [oc.val, ic.val, di.val, dj.val] *
                        getAtOrZero (paddedInput (inC := inC) (inH := inH) (inW := inW) (padding
                          := padding) input)
                          [ic.val, i.val * stride + di.val, j.val * stride + dj.val]) *
                  getAtOrZero δ [oc.val, i.val, j.val])
            =
          (∑ i : Fin (outH inH kH stride padding), ∑ j : Fin (outW inW kW stride padding),
                ∑ ic : Fin inC, ∑ di : Fin kH, ∑ dj : Fin kW, G oc i j ic di dj) := by
        refine Fintype.sum_congr _ _ ?_
        intro i
        refine Fintype.sum_congr _ _ ?_
        intro j
        simpa using (hMul3 oc i j)
      -- Commute `(i,j)` past `(ic,di,dj)` using `sum_comm`.
      -- Step 1: swap `j` and `ic`.
      have h2 :
          (∑ i : Fin (outH inH kH stride padding), ∑ j : Fin (outW inW kW stride padding),
                ∑ ic : Fin inC, ∑ di : Fin kH, ∑ dj : Fin kW, G oc i j ic di dj)
            =
          (∑ i : Fin (outH inH kH stride padding), ∑ ic : Fin inC, ∑ j : Fin (outW inW kW stride
            padding),
                ∑ di : Fin kH, ∑ dj : Fin kW, G oc i j ic di dj) := by
        refine Fintype.sum_congr _ _ ?_
        intro i
        simpa using
          (sum_comm (α := Fin (outW inW kW stride padding)) (β := Fin inC)
            (f := fun j ic => ∑ di : Fin kH, ∑ dj : Fin kW, G oc i j ic di dj))
      -- Step 2: swap `i` and `ic`.
      have h3 :
          (∑ i : Fin (outH inH kH stride padding), ∑ ic : Fin inC, ∑ j : Fin (outW inW kW stride
            padding),
                ∑ di : Fin kH, ∑ dj : Fin kW, G oc i j ic di dj)
            =
          (∑ ic : Fin inC, ∑ i : Fin (outH inH kH stride padding), ∑ j : Fin (outW inW kW stride
            padding),
                ∑ di : Fin kH, ∑ dj : Fin kW, G oc i j ic di dj) := by
        simpa using
          (sum_comm (α := Fin (outH inH kH stride padding)) (β := Fin inC)
            (f := fun i ic => ∑ j : Fin (outW inW kW stride padding),
              ∑ di : Fin kH, ∑ dj : Fin kW, G oc i j ic di dj))
      -- Step 3: swap `j` with `di`, then `i` with `di`, and similarly for `dj`.
      have h4 :
          (∑ ic : Fin inC, ∑ i : Fin (outH inH kH stride padding), ∑ j : Fin (outW inW kW stride
            padding),
                ∑ di : Fin kH, ∑ dj : Fin kW, G oc i j ic di dj)
            =
          (∑ ic : Fin inC, ∑ di : Fin kH, ∑ dj : Fin kW,
                ∑ i : Fin (outH inH kH stride padding), ∑ j : Fin (outW inW kW stride padding), G oc
                  i j ic di dj) := by
        -- Move `di` out past `(i,j)`, then `dj`.
        -- We do this by a fixed sequence of adjacent swaps.
        calc
          (∑ ic : Fin inC, ∑ i : Fin (outH inH kH stride padding), ∑ j : Fin (outW inW kW stride
            padding),
                ∑ di : Fin kH, ∑ dj : Fin kW, G oc i j ic di dj)
              =
            (∑ ic : Fin inC, ∑ i : Fin (outH inH kH stride padding), ∑ di : Fin kH, ∑ j : Fin (outW
              inW kW stride padding),
                ∑ dj : Fin kW, G oc i j ic di dj) := by
              refine Fintype.sum_congr _ _ ?_
              intro ic
              refine Fintype.sum_congr _ _ ?_
              intro i
              -- swap `j` and `di`
              simpa using
                (sum_comm (α := Fin (outW inW kW stride padding)) (β := Fin kH)
                  (f := fun j di => ∑ dj : Fin kW, G oc i j ic di dj))
          _ =
            (∑ ic : Fin inC, ∑ di : Fin kH, ∑ i : Fin (outH inH kH stride padding), ∑ j : Fin (outW
              inW kW stride padding),
                ∑ dj : Fin kW, G oc i j ic di dj) := by
              refine Fintype.sum_congr _ _ ?_
              intro ic
              -- swap `i` and `di`
              simpa using
                (sum_comm (α := Fin (outH inH kH stride padding)) (β := Fin kH)
                  (f := fun i di => ∑ j : Fin (outW inW kW stride padding),
                    ∑ dj : Fin kW, G oc i j ic di dj))
          _ =
            (∑ ic : Fin inC, ∑ di : Fin kH, ∑ i : Fin (outH inH kH stride padding), ∑ dj : Fin kW, ∑
              j : Fin (outW inW kW stride padding),
                G oc i j ic di dj) := by
              refine Fintype.sum_congr _ _ ?_
              intro ic
              refine Fintype.sum_congr _ _ ?_
              intro di
              refine Fintype.sum_congr _ _ ?_
              intro i
              -- swap `j` and `dj`
              simpa using
                (sum_comm (α := Fin (outW inW kW stride padding)) (β := Fin kW)
                  (f := fun j dj => G oc i j ic di dj))
          _ =
            (∑ ic : Fin inC, ∑ di : Fin kH, ∑ dj : Fin kW, ∑ i : Fin (outH inH kH stride padding), ∑
              j : Fin (outW inW kW stride padding),
                G oc i j ic di dj) := by
              refine Fintype.sum_congr _ _ ?_
              intro ic
              refine Fintype.sum_congr _ _ ?_
              intro di
              -- swap `i` and `dj`
              simpa using
                (sum_comm (α := Fin (outH inH kH stride padding)) (β := Fin kW)
                  (f := fun i dj => ∑ j : Fin (outW inW kW stride padding), G oc i j ic di dj))
      -- Assemble.
      simp [h1, h2, h3, h4]

    -- RHS: factor kernel entry out of the `(i,j)` sums.
    have hRHS :
        (∑ oc : Fin outC, ∑ ic : Fin inC, ∑ di : Fin kH, ∑ dj : Fin kW,
              getAtOrZero dKernel [oc.val, ic.val, di.val, dj.val] *
                (∑ i : Fin (outH inH kH stride padding), ∑ j : Fin (outW inW kW stride padding),
                  getAtOrZero (paddedInput (inC := inC) (inH := inH) (inW := inW) (padding :=
                    padding) input)
                    [ic.val, i.val * stride + di.val, j.val * stride + dj.val] *
                    getAtOrZero δ [oc.val, i.val, j.val]))
          =
        (∑ oc : Fin outC, ∑ ic : Fin inC, ∑ di : Fin kH, ∑ dj : Fin kW,
              ∑ i : Fin (outH inH kH stride padding), ∑ j : Fin (outW inW kW stride padding), G oc i
                j ic di dj) := by
      refine Fintype.sum_congr _ _ ?_
      intro oc
      refine Fintype.sum_congr _ _ ?_
      intro ic
      refine Fintype.sum_congr _ _ ?_
      intro di
      refine Fintype.sum_congr _ _ ?_
      intro dj
      -- Expand the product into the `(i,j)` sums.
      -- First over `i`, then over `j`.
      calc
        getAtOrZero dKernel [oc.val, ic.val, di.val, dj.val] *
            (∑ i : Fin (outH inH kH stride padding), ∑ j : Fin (outW inW kW stride padding),
              getAtOrZero (paddedInput (inC := inC) (inH := inH) (inW := inW) (padding :=
                padding) input)
                [ic.val, i.val * stride + di.val, j.val * stride + dj.val] *
                getAtOrZero δ [oc.val, i.val, j.val])
            =
          ∑ i : Fin (outH inH kH stride padding),
              getAtOrZero dKernel [oc.val, ic.val, di.val, dj.val] *
                (∑ j : Fin (outW inW kW stride padding),
                  getAtOrZero (paddedInput (inC := inC) (inH := inH) (inW := inW) (padding :=
                    padding) input)
                    [ic.val, i.val * stride + di.val, j.val * stride + dj.val] *
                    getAtOrZero δ [oc.val, i.val, j.val]) := by
          simpa [mul_assoc] using
            (mul_sum (ι := Fin (outH inH kH stride padding))
              (a := getAtOrZero dKernel [oc.val, ic.val, di.val, dj.val])
              (f := fun i =>
                ∑ j : Fin (outW inW kW stride padding),
                  getAtOrZero (paddedInput (inC := inC) (inH := inH) (inW := inW) (padding :=
                    padding) input)
                    [ic.val, i.val * stride + di.val, j.val * stride + dj.val] *
                    getAtOrZero δ [oc.val, i.val, j.val]))
        _ =
          ∑ i : Fin (outH inH kH stride padding), ∑ j : Fin (outW inW kW stride padding),
              (getAtOrZero dKernel [oc.val, ic.val, di.val, dj.val] *
                  getAtOrZero (paddedInput (inC := inC) (inH := inH) (inW := inW) (padding :=
                    padding) input)
                    [ic.val, i.val * stride + di.val, j.val * stride + dj.val]) *
                getAtOrZero δ [oc.val, i.val, j.val] := by
          refine Fintype.sum_congr _ _ ?_
          intro i
          simpa [mul_assoc] using
            (mul_sum (ι := Fin (outW inW kW stride padding))
              (a := getAtOrZero dKernel [oc.val, ic.val, di.val, dj.val])
              (f := fun j =>
                getAtOrZero (paddedInput (inC := inC) (inH := inH) (inW := inW) (padding :=
                  padding) input)
                  [ic.val, i.val * stride + di.val, j.val * stride + dj.val] *
                  getAtOrZero δ [oc.val, i.val, j.val]))
        _ = ∑ i : Fin (outH inH kH stride padding), ∑ j : Fin (outW inW kW stride padding), G oc i j
          ic di dj := by
          rfl

    -- Conclude by a common expanded form.
    exact hLHS.trans hRHS.symm
  rw [hReorder]
  -- RHS: expand dot on the 4D kernel.
  rw [dot4_eq_sum (a := dKernel) (b := Spec.conv2dKernelDerivSpec (α := ℝ) (layer := layer)
    (input := input) (grad_output := δ))]
  -- Rewrite each gradient entry and conclude.
  refine Finset.sum_congr rfl ?_
  intro oc _
  refine Finset.sum_congr rfl ?_
  intro ic _
  refine Finset.sum_congr rfl ?_
  intro di _
  refine Finset.sum_congr rfl ?_
  intro dj _
  simp [conv2d_kernel_deriv_get (layer := layer) (input := input) (δ := δ) (oc := oc) (ic := ic) (di
    := di) (dj := dj),
    mul_comm]

private lemma dot_conv2d_input
    {inC outC kH kW stride padding inH inW : Nat}
    {h1 : inC ≠ 0} {h2 : kH ≠ 0} {h3 : kW ≠ 0}
    (layer : Spec.Conv2DSpec inC outC kH kW stride padding ℝ h1 h2 h3)
    (input : Spec.MultiChannelImage inC inH inW ℝ)
    (dInput : Spec.MultiChannelImage inC inH inW ℝ)
    (δ : Spec.MultiChannelImage outC (outH inH kH stride padding) (outW inW kW stride padding) ℝ) :
    let layer0 : Spec.Conv2DSpec inC outC kH kW stride padding ℝ h1 h2 h3 :=
      { kernel := layer.kernel, bias := fill (0 : ℝ) (.dim outC .scalar) }
    dot (Spec.conv2dSpec (α := ℝ) (layer := layer0) dInput) δ
      =
    dot dInput (Spec.conv2dInputDerivSpec (α := ℝ) (layer := layer) (input := input)
      (grad_output := δ)) := by
  intro layer0
  classical

  -- A canonical expanded sum for both sides.
  let S : ℝ :=
    ∑ oc : Fin outC,
      ∑ oi : Fin (outH inH kH stride padding),
        ∑ oj : Fin (outW inW kW stride padding),
          ∑ ic : Fin inC,
            ∑ di : Fin kH,
              ∑ dj : Fin kW,
                (getAtOrZero layer.kernel [oc.val, ic.val, di.val, dj.val]) *
                  (getAtOrZero
                    (paddedInput (inC := inC) (inH := inH) (inW := inW) (padding := padding) dInput)
                    [ic.val, oi.val * stride + di.val, oj.val * stride + dj.val]) *
                  (getAtOrZero δ [oc.val, oi.val, oj.val])

  have hLHS :
      dot (Spec.conv2dSpec (α := ℝ) (layer := layer0) dInput) δ = S := by
    -- Expand the dot and rewrite each conv entry via `conv2d_spec_noBias_get`.
    rw [dot3_eq_sum (a := Spec.conv2dSpec (α := ℝ) (layer := layer0) dInput) (b := δ)]
    -- Match the binder sizes explicitly.
    change
      (∑ oc : Fin outC, ∑ oi : Fin (outH inH kH stride padding), ∑ oj : Fin (outW inW kW stride
        padding),
          getAtOrZero (Spec.conv2dSpec (α := ℝ) (layer := layer0) dInput) [oc.val, oi.val,
            oj.val] *
            getAtOrZero δ [oc.val, oi.val, oj.val])
        = S
    -- Rewrite each conv entry and distribute the final `* δ` into the `(ic,di,dj)` sums.
    refine (by
      refine Fintype.sum_congr _ _ ?_
      intro oc
      refine Fintype.sum_congr _ _ ?_
      intro oi
      refine Fintype.sum_congr _ _ ?_
      intro oj
      have hEntry :=
        (by
          simpa [layer0] using
            (conv2d_spec_noBias_get (h1 := h1) (h2 := h2) (h3 := h3)
              (dKernel := layer.kernel) (input := dInput) (oc := oc) (i := oi) (j := oj)))
      have hEntry' :
          getAtOrZero (Spec.conv2dSpec (α := ℝ) (layer := layer0) dInput) [oc.val, oi.val,
            oj.val]
            =
          ∑ ic : Fin inC,
            ∑ di : Fin kH,
              ∑ dj : Fin kW,
                getAtOrZero layer.kernel [oc.val, ic.val, di.val, dj.val] *
                  getAtOrZero
                    (paddedInput (inC := inC) (inH := inH) (inW := inW) (padding := padding) dInput)
                    [ic.val, oi.val * stride + di.val, oj.val * stride + dj.val] := by
        simpa using hEntry
      -- Multiply by `δ[oc,oi,oj]` and distribute across the nested sums.
      have hMul :
          (getAtOrZero (Spec.conv2dSpec (α := ℝ) (layer := layer0) dInput) [oc.val, oi.val,
            oj.val] *
              getAtOrZero δ [oc.val, oi.val, oj.val])
            =
          ∑ ic : Fin inC,
            ∑ di : Fin kH,
              ∑ dj : Fin kW,
                (getAtOrZero layer.kernel [oc.val, ic.val, di.val, dj.val]) *
                  (getAtOrZero
                    (paddedInput (inC := inC) (inH := inH) (inW := inW) (padding := padding) dInput)
                    [ic.val, oi.val * stride + di.val, oj.val * stride + dj.val]) *
                  (getAtOrZero δ [oc.val, oi.val, oj.val]) := by
        -- Rewrite the conv entry, then distribute the product.
        calc
          (getAtOrZero (Spec.conv2dSpec (α := ℝ) (layer := layer0) dInput) [oc.val, oi.val,
            oj.val] *
              getAtOrZero δ [oc.val, oi.val, oj.val])
              =
            ((∑ ic : Fin inC, ∑ di : Fin kH, ∑ dj : Fin kW,
                getAtOrZero layer.kernel [oc.val, ic.val, di.val, dj.val] *
                  getAtOrZero
                    (paddedInput (inC := inC) (inH := inH) (inW := inW) (padding := padding) dInput)
                    [ic.val, oi.val * stride + di.val, oj.val * stride + dj.val]) *
              getAtOrZero δ [oc.val, oi.val, oj.val]) := by
                simp [hEntry']
          _ =
            ∑ ic : Fin inC,
              (∑ di : Fin kH, ∑ dj : Fin kW,
                  getAtOrZero layer.kernel [oc.val, ic.val, di.val, dj.val] *
                    getAtOrZero
                      (paddedInput (inC := inC) (inH := inH) (inW := inW) (padding := padding)
                        dInput)
                      [ic.val, oi.val * stride + di.val, oj.val * stride + dj.val]) *
                getAtOrZero δ [oc.val, oi.val, oj.val] := by
              -- Distribute over `ic`.
              simpa [mul_assoc] using
                (sum_mul (ι := Fin inC)
                  (f := fun ic =>
                    ∑ di : Fin kH, ∑ dj : Fin kW,
                      getAtOrZero layer.kernel [oc.val, ic.val, di.val, dj.val] *
                        getAtOrZero
                          (paddedInput (inC := inC) (inH := inH) (inW := inW) (padding := padding)
                            dInput)
                          [ic.val, oi.val * stride + di.val, oj.val * stride + dj.val])
                  (a := getAtOrZero δ [oc.val, oi.val, oj.val]))
          _ =
            ∑ ic : Fin inC, ∑ di : Fin kH,
              (∑ dj : Fin kW,
                  getAtOrZero layer.kernel [oc.val, ic.val, di.val, dj.val] *
                    getAtOrZero
                      (paddedInput (inC := inC) (inH := inH) (inW := inW) (padding := padding)
                        dInput)
                      [ic.val, oi.val * stride + di.val, oj.val * stride + dj.val]) *
                getAtOrZero δ [oc.val, oi.val, oj.val] := by
              refine Fintype.sum_congr _ _ ?_
              intro ic
              simpa [mul_assoc] using
                (sum_mul (ι := Fin kH)
                  (f := fun di =>
                    ∑ dj : Fin kW,
                      getAtOrZero layer.kernel [oc.val, ic.val, di.val, dj.val] *
                        getAtOrZero
                          (paddedInput (inC := inC) (inH := inH) (inW := inW) (padding := padding)
                            dInput)
                          [ic.val, oi.val * stride + di.val, oj.val * stride + dj.val])
                  (a := getAtOrZero δ [oc.val, oi.val, oj.val]))
          _ =
            ∑ ic : Fin inC, ∑ di : Fin kH, ∑ dj : Fin kW,
              (getAtOrZero layer.kernel [oc.val, ic.val, di.val, dj.val] *
                  getAtOrZero
                    (paddedInput (inC := inC) (inH := inH) (inW := inW) (padding := padding) dInput)
                    [ic.val, oi.val * stride + di.val, oj.val * stride + dj.val]) *
                getAtOrZero δ [oc.val, oi.val, oj.val] := by
              refine Fintype.sum_congr _ _ ?_
              intro ic
              refine Fintype.sum_congr _ _ ?_
              intro di
              simpa [mul_assoc] using
                (sum_mul (ι := Fin kW)
                  (f := fun dj =>
                    getAtOrZero layer.kernel [oc.val, ic.val, di.val, dj.val] *
                      getAtOrZero
                        (paddedInput (inC := inC) (inH := inH) (inW := inW) (padding := padding)
                          dInput)
                        [ic.val, oi.val * stride + di.val, oj.val * stride + dj.val])
                  (a := getAtOrZero δ [oc.val, oi.val, oj.val]))
          _ =
            ∑ ic : Fin inC, ∑ di : Fin kH, ∑ dj : Fin kW,
              (getAtOrZero layer.kernel [oc.val, ic.val, di.val, dj.val]) *
                (getAtOrZero
                  (paddedInput (inC := inC) (inH := inH) (inW := inW) (padding := padding) dInput)
                  [ic.val, oi.val * stride + di.val, oj.val * stride + dj.val]) *
                (getAtOrZero δ [oc.val, oi.val, oj.val]) := by
              simp [mul_assoc]
      simpa [S, mul_assoc] using hMul)

  have hRHS :
      dot dInput (Spec.conv2dInputDerivSpec (α := ℝ) (layer := layer) (input := input)
        (grad_output := δ)) = S := by
    -- Expand the dot.
    rw [dot3_eq_sum (a := dInput)
      (b := Spec.conv2dInputDerivSpec (α := ℝ) (layer := layer) (input := input) (grad_output :=
        δ))]
    -- Match binder sizes.
    change
      (∑ ic : Fin inC, ∑ i : Fin inH, ∑ j : Fin inW,
          getAtOrZero dInput [ic.val, i.val, j.val] *
            getAtOrZero
              (Spec.conv2dInputDerivSpec (α := ℝ) (layer := layer) (input := input) (grad_output
                := δ))
              [ic.val, i.val, j.val]) = S
    -- Rewrite each gradient entry using `conv2d_input_deriv_get`.
    have hRewrite :
        (∑ ic : Fin inC, ∑ i : Fin inH, ∑ j : Fin inW,
            getAtOrZero dInput [ic.val, i.val, j.val] *
              getAtOrZero
                (Spec.conv2dInputDerivSpec (α := ℝ) (layer := layer) (input := input)
                  (grad_output := δ))
                [ic.val, i.val, j.val])
          =
        (∑ ic : Fin inC, ∑ i : Fin inH, ∑ j : Fin inW,
            getAtOrZero dInput [ic.val, i.val, j.val] *
              (∑ oc : Fin outC,
                ∑ oi : Fin (outH inH kH stride padding),
                  ∑ oj : Fin (outW inW kW stride padding),
                    ∑ di : Fin kH,
                      ∑ dj : Fin kW,
                        if h :
                            (oi.val * stride + di.val = i.val + padding) ∧
                            (oj.val * stride + dj.val = j.val + padding) then
                          getAtOrZero δ [oc.val, oi.val, oj.val] *
                            getAtOrZero layer.kernel [oc.val, ic.val, di.val, dj.val]
                        else 0)) := by
      refine Fintype.sum_congr _ _ ?_
      intro ic
      refine Fintype.sum_congr _ _ ?_
      intro i
      refine Fintype.sum_congr _ _ ?_
      intro j
      simp [conv2d_input_deriv_get (layer := layer) (input := input) (δ := δ) (ic := ic) (i := i) (j
        := j)]
    -- Distribute the multiplication into the nested sums, then commute sums so that `(i,j)` are the
    -- innermost sums.
    -- Finally, collapse the `(i,j)` sum with `sum_shift_eq_paddedInput`.
    rw [hRewrite]
    -- Push `get_at_or_zero dInput[...]` into the sums.
    have hDist :
        (∑ ic : Fin inC, ∑ i : Fin inH, ∑ j : Fin inW,
            getAtOrZero dInput [ic.val, i.val, j.val] *
              (∑ oc : Fin outC,
                ∑ oi : Fin (outH inH kH stride padding),
                  ∑ oj : Fin (outW inW kW stride padding),
                    ∑ di : Fin kH,
                      ∑ dj : Fin kW,
                        if h :
                            (oi.val * stride + di.val = i.val + padding) ∧
                            (oj.val * stride + dj.val = j.val + padding) then
                          getAtOrZero δ [oc.val, oi.val, oj.val] *
                            getAtOrZero layer.kernel [oc.val, ic.val, di.val, dj.val]
                        else 0))
          =
        (∑ ic : Fin inC, ∑ i : Fin inH, ∑ j : Fin inW,
            ∑ oc : Fin outC,
              ∑ oi : Fin (outH inH kH stride padding),
                ∑ oj : Fin (outW inW kW stride padding),
                  ∑ di : Fin kH,
                    ∑ dj : Fin kW,
                      getAtOrZero dInput [ic.val, i.val, j.val] *
                        (if h :
                            (oi.val * stride + di.val = i.val + padding) ∧
                            (oj.val * stride + dj.val = j.val + padding) then
                          getAtOrZero δ [oc.val, oi.val, oj.val] *
                            getAtOrZero layer.kernel [oc.val, ic.val, di.val, dj.val]
                        else 0)) := by
      refine Fintype.sum_congr _ _ ?_
      intro ic
      refine Fintype.sum_congr _ _ ?_
      intro i
      refine Fintype.sum_congr _ _ ?_
      intro j
      -- Distribute across each nested sum using `mul_sum`, one level at a time.
      let a : ℝ := getAtOrZero dInput [ic.val, i.val, j.val]
      calc
        a *
            (∑ oc : Fin outC,
              ∑ oi : Fin (outH inH kH stride padding),
                ∑ oj : Fin (outW inW kW stride padding),
                  ∑ di : Fin kH,
                    ∑ dj : Fin kW,
                      if h :
                          (oi.val * stride + di.val = i.val + padding) ∧
                          (oj.val * stride + dj.val = j.val + padding) then
                        getAtOrZero δ [oc.val, oi.val, oj.val] *
                          getAtOrZero layer.kernel [oc.val, ic.val, di.val, dj.val]
                      else 0)
            =
          ∑ oc : Fin outC,
            a *
              (∑ oi : Fin (outH inH kH stride padding),
                ∑ oj : Fin (outW inW kW stride padding),
                  ∑ di : Fin kH,
                    ∑ dj : Fin kW,
                      if h :
                          (oi.val * stride + di.val = i.val + padding) ∧
                          (oj.val * stride + dj.val = j.val + padding) then
                        getAtOrZero δ [oc.val, oi.val, oj.val] *
                          getAtOrZero layer.kernel [oc.val, ic.val, di.val, dj.val]
                      else 0) := by
            simpa [a] using
              (mul_sum (ι := Fin outC) (a := a)
                (f := fun oc =>
                  ∑ oi : Fin (outH inH kH stride padding),
                    ∑ oj : Fin (outW inW kW stride padding),
                      ∑ di : Fin kH,
                        ∑ dj : Fin kW,
                          if h :
                              (oi.val * stride + di.val = i.val + padding) ∧
                              (oj.val * stride + dj.val = j.val + padding) then
                            getAtOrZero δ [oc.val, oi.val, oj.val] *
                              getAtOrZero layer.kernel [oc.val, ic.val, di.val, dj.val]
                          else 0))
        _ =
          ∑ oc : Fin outC,
            ∑ oi : Fin (outH inH kH stride padding),
              a *
                (∑ oj : Fin (outW inW kW stride padding),
                  ∑ di : Fin kH,
                    ∑ dj : Fin kW,
                      if h :
                          (oi.val * stride + di.val = i.val + padding) ∧
                          (oj.val * stride + dj.val = j.val + padding) then
                        getAtOrZero δ [oc.val, oi.val, oj.val] *
                          getAtOrZero layer.kernel [oc.val, ic.val, di.val, dj.val]
                      else 0) := by
            refine Fintype.sum_congr _ _ ?_
            intro oc
            simpa [a] using
              (mul_sum (ι := Fin (outH inH kH stride padding)) (a := a)
                (f := fun oi =>
                  ∑ oj : Fin (outW inW kW stride padding),
                    ∑ di : Fin kH,
                      ∑ dj : Fin kW,
                        if h :
                            (oi.val * stride + di.val = i.val + padding) ∧
                            (oj.val * stride + dj.val = j.val + padding) then
                          getAtOrZero δ [oc.val, oi.val, oj.val] *
                            getAtOrZero layer.kernel [oc.val, ic.val, di.val, dj.val]
                        else 0))
        _ =
          ∑ oc : Fin outC,
            ∑ oi : Fin (outH inH kH stride padding),
              ∑ oj : Fin (outW inW kW stride padding),
                a *
                  (∑ di : Fin kH,
                    ∑ dj : Fin kW,
                      if h :
                          (oi.val * stride + di.val = i.val + padding) ∧
                          (oj.val * stride + dj.val = j.val + padding) then
                        getAtOrZero δ [oc.val, oi.val, oj.val] *
                          getAtOrZero layer.kernel [oc.val, ic.val, di.val, dj.val]
                      else 0) := by
            refine Fintype.sum_congr _ _ ?_
            intro oc
            refine Fintype.sum_congr _ _ ?_
            intro oi
            simpa [a] using
              (mul_sum (ι := Fin (outW inW kW stride padding)) (a := a)
                (f := fun oj =>
                  ∑ di : Fin kH,
                    ∑ dj : Fin kW,
                      if h :
                          (oi.val * stride + di.val = i.val + padding) ∧
                          (oj.val * stride + dj.val = j.val + padding) then
                        getAtOrZero δ [oc.val, oi.val, oj.val] *
                          getAtOrZero layer.kernel [oc.val, ic.val, di.val, dj.val]
                      else 0))
        _ =
          ∑ oc : Fin outC,
            ∑ oi : Fin (outH inH kH stride padding),
              ∑ oj : Fin (outW inW kW stride padding),
                ∑ di : Fin kH,
                  a *
                    (∑ dj : Fin kW,
                      if h :
                          (oi.val * stride + di.val = i.val + padding) ∧
                          (oj.val * stride + dj.val = j.val + padding) then
                        getAtOrZero δ [oc.val, oi.val, oj.val] *
                          getAtOrZero layer.kernel [oc.val, ic.val, di.val, dj.val]
                      else 0) := by
            refine Fintype.sum_congr _ _ ?_
            intro oc
            refine Fintype.sum_congr _ _ ?_
            intro oi
            refine Fintype.sum_congr _ _ ?_
            intro oj
            simpa [a] using
              (mul_sum (ι := Fin kH) (a := a)
                (f := fun di =>
                  ∑ dj : Fin kW,
                    if h :
                        (oi.val * stride + di.val = i.val + padding) ∧
                        (oj.val * stride + dj.val = j.val + padding) then
                      getAtOrZero δ [oc.val, oi.val, oj.val] *
                        getAtOrZero layer.kernel [oc.val, ic.val, di.val, dj.val]
                    else 0))
        _ =
          ∑ oc : Fin outC,
            ∑ oi : Fin (outH inH kH stride padding),
              ∑ oj : Fin (outW inW kW stride padding),
                ∑ di : Fin kH,
                  ∑ dj : Fin kW,
                    a *
                      (if h :
                          (oi.val * stride + di.val = i.val + padding) ∧
                          (oj.val * stride + dj.val = j.val + padding) then
                        getAtOrZero δ [oc.val, oi.val, oj.val] *
                          getAtOrZero layer.kernel [oc.val, ic.val, di.val, dj.val]
                      else 0) := by
            refine Fintype.sum_congr _ _ ?_
            intro oc
            refine Fintype.sum_congr _ _ ?_
            intro oi
            refine Fintype.sum_congr _ _ ?_
            intro oj
            refine Fintype.sum_congr _ _ ?_
            intro di
            simpa [a] using
              (mul_sum (ι := Fin kW) (a := a)
                (f := fun dj =>
                  if h :
                      (oi.val * stride + di.val = i.val + padding) ∧
                      (oj.val * stride + dj.val = j.val + padding) then
                    getAtOrZero δ [oc.val, oi.val, oj.val] *
                      getAtOrZero layer.kernel [oc.val, ic.val, di.val, dj.val]
                  else 0))
        _ =
          (∑ oc : Fin outC,
              ∑ oi : Fin (outH inH kH stride padding),
                ∑ oj : Fin (outW inW kW stride padding),
                  ∑ di : Fin kH,
                    ∑ dj : Fin kW,
                      getAtOrZero dInput [ic.val, i.val, j.val] *
                        (if h :
                            (oi.val * stride + di.val = i.val + padding) ∧
                            (oj.val * stride + dj.val = j.val + padding) then
                          getAtOrZero δ [oc.val, oi.val, oj.val] *
                            getAtOrZero layer.kernel [oc.val, ic.val, di.val, dj.val]
                        else 0)) := by
            simp [a]
    rw [hDist]
    -- Commute sums to get the order `oc, oi, oj, ic, di, dj, i, j`.
    have hComm₁ :
        (∑ ic : Fin inC, ∑ i : Fin inH, ∑ j : Fin inW,
            ∑ oc : Fin outC,
              ∑ oi : Fin (outH inH kH stride padding),
                ∑ oj : Fin (outW inW kW stride padding),
                  ∑ di : Fin kH,
                    ∑ dj : Fin kW,
                      getAtOrZero dInput [ic.val, i.val, j.val] *
                        (if h :
                            (oi.val * stride + di.val = i.val + padding) ∧
                            (oj.val * stride + dj.val = j.val + padding) then
                          getAtOrZero δ [oc.val, oi.val, oj.val] *
                            getAtOrZero layer.kernel [oc.val, ic.val, di.val, dj.val]
                        else 0))
          =
        (∑ oc : Fin outC,
          ∑ oi : Fin (outH inH kH stride padding),
            ∑ oj : Fin (outW inW kW stride padding),
              ∑ ic : Fin inC,
                ∑ di : Fin kH,
                  ∑ dj : Fin kW,
                    ∑ i : Fin inH,
                      ∑ j : Fin inW,
                        getAtOrZero dInput [ic.val, i.val, j.val] *
                          (if h :
                              (oi.val * stride + di.val = i.val + padding) ∧
                              (oj.val * stride + dj.val = j.val + padding) then
                            getAtOrZero δ [oc.val, oi.val, oj.val] *
                              getAtOrZero layer.kernel [oc.val, ic.val, di.val, dj.val]
                          else 0)) := by
      -- Move `oc` outward past `j`, `i`, and `ic`, then move `oi/oj/di/dj` outward past `i/j`
      -- similarly.
      -- Do this by adjacent swaps using `sum_comm`.
      -- Swap `j` with `oc`.
      have h1 :
          (∑ ic : Fin inC, ∑ i : Fin inH, ∑ j : Fin inW, ∑ oc : Fin outC,
                ∑ oi : Fin (outH inH kH stride padding),
                  ∑ oj : Fin (outW inW kW stride padding),
                    ∑ di : Fin kH,
                      ∑ dj : Fin kW,
                        getAtOrZero dInput [ic.val, i.val, j.val] *
                          (if h :
                              (oi.val * stride + di.val = i.val + padding) ∧
                              (oj.val * stride + dj.val = j.val + padding) then
                            getAtOrZero δ [oc.val, oi.val, oj.val] *
                              getAtOrZero layer.kernel [oc.val, ic.val, di.val, dj.val]
                          else 0))
            =
          (∑ ic : Fin inC, ∑ i : Fin inH, ∑ oc : Fin outC, ∑ j : Fin inW,
                ∑ oi : Fin (outH inH kH stride padding),
                  ∑ oj : Fin (outW inW kW stride padding),
                    ∑ di : Fin kH,
                      ∑ dj : Fin kW,
                        getAtOrZero dInput [ic.val, i.val, j.val] *
                          (if h :
                              (oi.val * stride + di.val = i.val + padding) ∧
                              (oj.val * stride + dj.val = j.val + padding) then
                            getAtOrZero δ [oc.val, oi.val, oj.val] *
                              getAtOrZero layer.kernel [oc.val, ic.val, di.val, dj.val]
                          else 0)) := by
        refine Fintype.sum_congr _ _ ?_
        intro ic
        refine Fintype.sum_congr _ _ ?_
        intro i
        simpa using
          (sum_comm (α := Fin inW) (β := Fin outC)
            (f := fun j oc =>
              ∑ oi : Fin (outH inH kH stride padding),
                ∑ oj : Fin (outW inW kW stride padding),
                  ∑ di : Fin kH,
                    ∑ dj : Fin kW,
                      getAtOrZero dInput [ic.val, i.val, j.val] *
                        (if h :
                            (oi.val * stride + di.val = i.val + padding) ∧
                            (oj.val * stride + dj.val = j.val + padding) then
                          getAtOrZero δ [oc.val, oi.val, oj.val] *
                            getAtOrZero layer.kernel [oc.val, ic.val, di.val, dj.val]
                        else 0)))
      -- Swap `i` with `oc`.
      have h2 :
          (∑ ic : Fin inC, ∑ i : Fin inH, ∑ oc : Fin outC, ∑ j : Fin inW,
              ∑ oi : Fin (outH inH kH stride padding),
                ∑ oj : Fin (outW inW kW stride padding),
                  ∑ di : Fin kH,
                    ∑ dj : Fin kW,
                      getAtOrZero dInput [ic.val, i.val, j.val] *
                        (if h :
                            (oi.val * stride + di.val = i.val + padding) ∧
                            (oj.val * stride + dj.val = j.val + padding) then
                          getAtOrZero δ [oc.val, oi.val, oj.val] *
                            getAtOrZero layer.kernel [oc.val, ic.val, di.val, dj.val]
                        else 0))
            =
          (∑ ic : Fin inC, ∑ oc : Fin outC, ∑ i : Fin inH, ∑ j : Fin inW,
              ∑ oi : Fin (outH inH kH stride padding),
                ∑ oj : Fin (outW inW kW stride padding),
                  ∑ di : Fin kH,
                    ∑ dj : Fin kW,
                      getAtOrZero dInput [ic.val, i.val, j.val] *
                        (if h :
                            (oi.val * stride + di.val = i.val + padding) ∧
                            (oj.val * stride + dj.val = j.val + padding) then
                          getAtOrZero δ [oc.val, oi.val, oj.val] *
                            getAtOrZero layer.kernel [oc.val, ic.val, di.val, dj.val]
                        else 0)) := by
        refine Fintype.sum_congr _ _ ?_
        intro ic
        simpa using
          (sum_comm (α := Fin inH) (β := Fin outC)
            (f := fun i oc =>
              ∑ j : Fin inW,
                ∑ oi : Fin (outH inH kH stride padding),
                  ∑ oj : Fin (outW inW kW stride padding),
                    ∑ di : Fin kH,
                      ∑ dj : Fin kW,
                        getAtOrZero dInput [ic.val, i.val, j.val] *
                          (if h :
                              (oi.val * stride + di.val = i.val + padding) ∧
                              (oj.val * stride + dj.val = j.val + padding) then
                            getAtOrZero δ [oc.val, oi.val, oj.val] *
                              getAtOrZero layer.kernel [oc.val, ic.val, di.val, dj.val]
                          else 0)))
      -- Swap `ic` with `oc`.
      have h3 :
          (∑ ic : Fin inC, ∑ oc : Fin outC, ∑ i : Fin inH, ∑ j : Fin inW,
              ∑ oi : Fin (outH inH kH stride padding),
                ∑ oj : Fin (outW inW kW stride padding),
                  ∑ di : Fin kH,
                    ∑ dj : Fin kW,
                      getAtOrZero dInput [ic.val, i.val, j.val] *
                        (if h :
                            (oi.val * stride + di.val = i.val + padding) ∧
                            (oj.val * stride + dj.val = j.val + padding) then
                          getAtOrZero δ [oc.val, oi.val, oj.val] *
                            getAtOrZero layer.kernel [oc.val, ic.val, di.val, dj.val]
                        else 0))
            =
          (∑ oc : Fin outC, ∑ ic : Fin inC, ∑ i : Fin inH, ∑ j : Fin inW,
              ∑ oi : Fin (outH inH kH stride padding),
                ∑ oj : Fin (outW inW kW stride padding),
                  ∑ di : Fin kH,
                    ∑ dj : Fin kW,
                      getAtOrZero dInput [ic.val, i.val, j.val] *
                        (if h :
                            (oi.val * stride + di.val = i.val + padding) ∧
                            (oj.val * stride + dj.val = j.val + padding) then
                          getAtOrZero δ [oc.val, oi.val, oj.val] *
                            getAtOrZero layer.kernel [oc.val, ic.val, di.val, dj.val]
                        else 0)) := by
        simpa using
          (sum_comm (α := Fin inC) (β := Fin outC)
            (f := fun ic oc =>
              ∑ i : Fin inH,
                ∑ j : Fin inW,
                  ∑ oi : Fin (outH inH kH stride padding),
                    ∑ oj : Fin (outW inW kW stride padding),
                      ∑ di : Fin kH,
                        ∑ dj : Fin kW,
                          getAtOrZero dInput [ic.val, i.val, j.val] *
                            (if h :
                                (oi.val * stride + di.val = i.val + padding) ∧
                                (oj.val * stride + dj.val = j.val + padding) then
                              getAtOrZero δ [oc.val, oi.val, oj.val] *
                                getAtOrZero layer.kernel [oc.val, ic.val, di.val, dj.val]
                            else 0)))
      -- Now move `oi` outward past `i/j` (and similarly `oj/di/dj`), and reorder to
      -- `oc,oi,oj,ic,di,dj,i,j`.
      -- We do this by applying `sum_comm` repeatedly under `oc` and `ic`.
      have h4 :
          (∑ oc : Fin outC, ∑ ic : Fin inC, ∑ i : Fin inH, ∑ j : Fin inW,
                ∑ oi : Fin (outH inH kH stride padding),
                  ∑ oj : Fin (outW inW kW stride padding),
                    ∑ di : Fin kH,
                      ∑ dj : Fin kW,
                        getAtOrZero dInput [ic.val, i.val, j.val] *
                          (if h :
                              (oi.val * stride + di.val = i.val + padding) ∧
                              (oj.val * stride + dj.val = j.val + padding) then
                            getAtOrZero δ [oc.val, oi.val, oj.val] *
                              getAtOrZero layer.kernel [oc.val, ic.val, di.val, dj.val]
                          else 0))
            =
          (∑ oc : Fin outC,
            ∑ oi : Fin (outH inH kH stride padding),
              ∑ oj : Fin (outW inW kW stride padding),
                ∑ ic : Fin inC,
                  ∑ di : Fin kH,
                    ∑ dj : Fin kW,
                      ∑ i : Fin inH,
                        ∑ j : Fin inW,
                          getAtOrZero dInput [ic.val, i.val, j.val] *
                            (if h :
                                (oi.val * stride + di.val = i.val + padding) ∧
                                (oj.val * stride + dj.val = j.val + padding) then
                              getAtOrZero δ [oc.val, oi.val, oj.val] *
                                getAtOrZero layer.kernel [oc.val, ic.val, di.val, dj.val]
                            else 0)) := by
        refine Fintype.sum_congr _ _ ?_
        intro oc
        -- Swap `oi` outward past `j`, then `i`, then past `ic`.
        -- Step A: under `ic`, swap `j` with `oi` and `i` with `oi`.
        have hA :
            (∑ ic : Fin inC, ∑ i : Fin inH, ∑ j : Fin inW,
                  ∑ oi : Fin (outH inH kH stride padding),
                    ∑ oj : Fin (outW inW kW stride padding),
                      ∑ di : Fin kH,
                        ∑ dj : Fin kW,
                          getAtOrZero dInput [ic.val, i.val, j.val] *
                            (if h :
                                (oi.val * stride + di.val = i.val + padding) ∧
                                (oj.val * stride + dj.val = j.val + padding) then
                              getAtOrZero δ [oc.val, oi.val, oj.val] *
                                getAtOrZero layer.kernel [oc.val, ic.val, di.val, dj.val]
                            else 0))
              =
            (∑ ic : Fin inC,
              ∑ oi : Fin (outH inH kH stride padding),
                ∑ i : Fin inH,
                  ∑ j : Fin inW,
                    ∑ oj : Fin (outW inW kW stride padding),
                      ∑ di : Fin kH,
                        ∑ dj : Fin kW,
                          getAtOrZero dInput [ic.val, i.val, j.val] *
                            (if h :
                                (oi.val * stride + di.val = i.val + padding) ∧
                                (oj.val * stride + dj.val = j.val + padding) then
                              getAtOrZero δ [oc.val, oi.val, oj.val] *
                                getAtOrZero layer.kernel [oc.val, ic.val, di.val, dj.val]
                            else 0)) := by
          refine Fintype.sum_congr _ _ ?_
          intro ic
          -- swap `j` and `oi`, then `i` and `oi`
          have hAj :
              (∑ i : Fin inH, ∑ j : Fin inW, ∑ oi : Fin (outH inH kH stride padding),
                    ∑ oj : Fin (outW inW kW stride padding),
                      ∑ di : Fin kH,
                        ∑ dj : Fin kW,
                          getAtOrZero dInput [ic.val, i.val, j.val] *
                            (if h :
                                (oi.val * stride + di.val = i.val + padding) ∧
                                (oj.val * stride + dj.val = j.val + padding) then
                              getAtOrZero δ [oc.val, oi.val, oj.val] *
                                getAtOrZero layer.kernel [oc.val, ic.val, di.val, dj.val]
                            else 0))
                =
              (∑ i : Fin inH, ∑ oi : Fin (outH inH kH stride padding), ∑ j : Fin inW,
                    ∑ oj : Fin (outW inW kW stride padding),
                      ∑ di : Fin kH,
                        ∑ dj : Fin kW,
                          getAtOrZero dInput [ic.val, i.val, j.val] *
                            (if h :
                                (oi.val * stride + di.val = i.val + padding) ∧
                                (oj.val * stride + dj.val = j.val + padding) then
                              getAtOrZero δ [oc.val, oi.val, oj.val] *
                                getAtOrZero layer.kernel [oc.val, ic.val, di.val, dj.val]
                            else 0)) := by
            refine Fintype.sum_congr _ _ ?_
            intro i
            simpa using
              (sum_comm (α := Fin inW) (β := Fin (outH inH kH stride padding))
                (f := fun j oi =>
                  ∑ oj : Fin (outW inW kW stride padding),
                    ∑ di : Fin kH,
                      ∑ dj : Fin kW,
                        getAtOrZero dInput [ic.val, i.val, j.val] *
                          (if h :
                              (oi.val * stride + di.val = i.val + padding) ∧
                              (oj.val * stride + dj.val = j.val + padding) then
                            getAtOrZero δ [oc.val, oi.val, oj.val] *
                              getAtOrZero layer.kernel [oc.val, ic.val, di.val, dj.val]
                          else 0)))
          have hAi :
              (∑ i : Fin inH, ∑ oi : Fin (outH inH kH stride padding), ∑ j : Fin inW,
                    ∑ oj : Fin (outW inW kW stride padding),
                      ∑ di : Fin kH,
                        ∑ dj : Fin kW,
                          getAtOrZero dInput [ic.val, i.val, j.val] *
                            (if h :
                                (oi.val * stride + di.val = i.val + padding) ∧
                                (oj.val * stride + dj.val = j.val + padding) then
                              getAtOrZero δ [oc.val, oi.val, oj.val] *
                                getAtOrZero layer.kernel [oc.val, ic.val, di.val, dj.val]
                            else 0))
                =
              (∑ oi : Fin (outH inH kH stride padding), ∑ i : Fin inH, ∑ j : Fin inW,
                    ∑ oj : Fin (outW inW kW stride padding),
                      ∑ di : Fin kH,
                        ∑ dj : Fin kW,
                          getAtOrZero dInput [ic.val, i.val, j.val] *
                            (if h :
                                (oi.val * stride + di.val = i.val + padding) ∧
                                (oj.val * stride + dj.val = j.val + padding) then
                              getAtOrZero δ [oc.val, oi.val, oj.val] *
                                getAtOrZero layer.kernel [oc.val, ic.val, di.val, dj.val]
                            else 0)) := by
            simpa using
              (sum_comm (α := Fin inH) (β := Fin (outH inH kH stride padding))
                (f := fun i oi =>
                  ∑ j : Fin inW,
                    ∑ oj : Fin (outW inW kW stride padding),
                      ∑ di : Fin kH,
                        ∑ dj : Fin kW,
                          getAtOrZero dInput [ic.val, i.val, j.val] *
                            (if h :
                                (oi.val * stride + di.val = i.val + padding) ∧
                                (oj.val * stride + dj.val = j.val + padding) then
                              getAtOrZero δ [oc.val, oi.val, oj.val] *
                                getAtOrZero layer.kernel [oc.val, ic.val, di.val, dj.val]
                            else 0)))
          exact hAj.trans hAi
        -- Now swap `ic` and `oi`.
        have hB :
            (∑ ic : Fin inC, ∑ oi : Fin (outH inH kH stride padding), ∑ i : Fin inH, ∑ j : Fin inW,
                  ∑ oj : Fin (outW inW kW stride padding),
                    ∑ di : Fin kH,
                      ∑ dj : Fin kW,
                        getAtOrZero dInput [ic.val, i.val, j.val] *
                          (if h :
                              (oi.val * stride + di.val = i.val + padding) ∧
                              (oj.val * stride + dj.val = j.val + padding) then
                            getAtOrZero δ [oc.val, oi.val, oj.val] *
                              getAtOrZero layer.kernel [oc.val, ic.val, di.val, dj.val]
                          else 0))
              =
            (∑ oi : Fin (outH inH kH stride padding), ∑ ic : Fin inC, ∑ i : Fin inH, ∑ j : Fin inW,
                  ∑ oj : Fin (outW inW kW stride padding),
                    ∑ di : Fin kH,
                      ∑ dj : Fin kW,
                        getAtOrZero dInput [ic.val, i.val, j.val] *
                          (if h :
                              (oi.val * stride + di.val = i.val + padding) ∧
                              (oj.val * stride + dj.val = j.val + padding) then
                            getAtOrZero δ [oc.val, oi.val, oj.val] *
                              getAtOrZero layer.kernel [oc.val, ic.val, di.val, dj.val]
                          else 0)) := by
          simpa using
            (sum_comm (α := Fin inC) (β := Fin (outH inH kH stride padding))
              (f := fun ic oi =>
                ∑ i : Fin inH,
                  ∑ j : Fin inW,
                    ∑ oj : Fin (outW inW kW stride padding),
                      ∑ di : Fin kH,
                        ∑ dj : Fin kW,
                          getAtOrZero dInput [ic.val, i.val, j.val] *
                            (if h :
                                (oi.val * stride + di.val = i.val + padding) ∧
                                (oj.val * stride + dj.val = j.val + padding) then
                              getAtOrZero δ [oc.val, oi.val, oj.val] *
                                getAtOrZero layer.kernel [oc.val, ic.val, di.val, dj.val]
                            else 0)))
        -- Now move `oj` outside `ic,i,j` (so it sits after `oi`), then move `di/dj` outside `i/j`.
        have hC :
            (∑ oi : Fin (outH inH kH stride padding),
                ∑ ic : Fin inC,
                  ∑ i : Fin inH,
                    ∑ j : Fin inW,
                      ∑ oj : Fin (outW inW kW stride padding),
                        ∑ di : Fin kH,
                          ∑ dj : Fin kW,
                            getAtOrZero dInput [ic.val, i.val, j.val] *
                              (if h :
                                  (oi.val * stride + di.val = i.val + padding) ∧
                                  (oj.val * stride + dj.val = j.val + padding) then
                                getAtOrZero δ [oc.val, oi.val, oj.val] *
                                  getAtOrZero layer.kernel [oc.val, ic.val, di.val, dj.val]
                              else 0))
              =
            (∑ oi : Fin (outH inH kH stride padding),
                ∑ oj : Fin (outW inW kW stride padding),
                  ∑ ic : Fin inC,
                    ∑ i : Fin inH,
                      ∑ j : Fin inW,
                        ∑ di : Fin kH,
                          ∑ dj : Fin kW,
                            getAtOrZero dInput [ic.val, i.val, j.val] *
                              (if h :
                                  (oi.val * stride + di.val = i.val + padding) ∧
                                  (oj.val * stride + dj.val = j.val + padding) then
                                getAtOrZero δ [oc.val, oi.val, oj.val] *
                                  getAtOrZero layer.kernel [oc.val, ic.val, di.val, dj.val]
                              else 0)) := by
          refine Fintype.sum_congr _ _ ?_
          intro oi
          -- Swap `oj` outward past `j`, then `i`, then `ic`.
          have hC1 :
              (∑ ic : Fin inC, ∑ i : Fin inH, ∑ j : Fin inW,
                    ∑ oj : Fin (outW inW kW stride padding),
                      ∑ di : Fin kH,
                        ∑ dj : Fin kW,
                          getAtOrZero dInput [ic.val, i.val, j.val] *
                            (if h :
                                (oi.val * stride + di.val = i.val + padding) ∧
                                (oj.val * stride + dj.val = j.val + padding) then
                              getAtOrZero δ [oc.val, oi.val, oj.val] *
                                getAtOrZero layer.kernel [oc.val, ic.val, di.val, dj.val]
                            else 0))
                =
              (∑ ic : Fin inC, ∑ i : Fin inH, ∑ oj : Fin (outW inW kW stride padding), ∑ j : Fin
                inW,
                    ∑ di : Fin kH,
                      ∑ dj : Fin kW,
                        getAtOrZero dInput [ic.val, i.val, j.val] *
                          (if h :
                              (oi.val * stride + di.val = i.val + padding) ∧
                              (oj.val * stride + dj.val = j.val + padding) then
                            getAtOrZero δ [oc.val, oi.val, oj.val] *
                              getAtOrZero layer.kernel [oc.val, ic.val, di.val, dj.val]
                          else 0)) := by
            refine Fintype.sum_congr _ _ ?_
            intro ic
            refine Fintype.sum_congr _ _ ?_
            intro i
            simpa using
              (sum_comm (α := Fin inW) (β := Fin (outW inW kW stride padding))
                (f := fun j oj =>
                  ∑ di : Fin kH,
                    ∑ dj : Fin kW,
                      getAtOrZero dInput [ic.val, i.val, j.val] *
                        (if h :
                            (oi.val * stride + di.val = i.val + padding) ∧
                            (oj.val * stride + dj.val = j.val + padding) then
                          getAtOrZero δ [oc.val, oi.val, oj.val] *
                            getAtOrZero layer.kernel [oc.val, ic.val, di.val, dj.val]
                        else 0)))
          have hC2 :
              (∑ ic : Fin inC, ∑ i : Fin inH, ∑ oj : Fin (outW inW kW stride padding), ∑ j : Fin
                inW,
                    ∑ di : Fin kH,
                      ∑ dj : Fin kW,
                        getAtOrZero dInput [ic.val, i.val, j.val] *
                          (if h :
                              (oi.val * stride + di.val = i.val + padding) ∧
                              (oj.val * stride + dj.val = j.val + padding) then
                            getAtOrZero δ [oc.val, oi.val, oj.val] *
                              getAtOrZero layer.kernel [oc.val, ic.val, di.val, dj.val]
                          else 0))
                =
              (∑ ic : Fin inC, ∑ oj : Fin (outW inW kW stride padding), ∑ i : Fin inH, ∑ j : Fin
                inW,
                    ∑ di : Fin kH,
                      ∑ dj : Fin kW,
                        getAtOrZero dInput [ic.val, i.val, j.val] *
                          (if h :
                              (oi.val * stride + di.val = i.val + padding) ∧
                              (oj.val * stride + dj.val = j.val + padding) then
                            getAtOrZero δ [oc.val, oi.val, oj.val] *
                              getAtOrZero layer.kernel [oc.val, ic.val, di.val, dj.val]
                          else 0)) := by
            refine Fintype.sum_congr _ _ ?_
            intro ic
            simpa using
              (sum_comm (α := Fin inH) (β := Fin (outW inW kW stride padding))
                (f := fun i oj =>
                  ∑ j : Fin inW,
                    ∑ di : Fin kH,
                      ∑ dj : Fin kW,
                        getAtOrZero dInput [ic.val, i.val, j.val] *
                          (if h :
                              (oi.val * stride + di.val = i.val + padding) ∧
                              (oj.val * stride + dj.val = j.val + padding) then
                            getAtOrZero δ [oc.val, oi.val, oj.val] *
                              getAtOrZero layer.kernel [oc.val, ic.val, di.val, dj.val]
                          else 0)))
          have hC3 :
              (∑ ic : Fin inC, ∑ oj : Fin (outW inW kW stride padding), ∑ i : Fin inH, ∑ j : Fin
                inW,
                    ∑ di : Fin kH,
                      ∑ dj : Fin kW,
                        getAtOrZero dInput [ic.val, i.val, j.val] *
                          (if h :
                              (oi.val * stride + di.val = i.val + padding) ∧
                              (oj.val * stride + dj.val = j.val + padding) then
                            getAtOrZero δ [oc.val, oi.val, oj.val] *
                              getAtOrZero layer.kernel [oc.val, ic.val, di.val, dj.val]
                          else 0))
                =
              (∑ oj : Fin (outW inW kW stride padding), ∑ ic : Fin inC, ∑ i : Fin inH, ∑ j : Fin
                inW,
                    ∑ di : Fin kH,
                      ∑ dj : Fin kW,
                        getAtOrZero dInput [ic.val, i.val, j.val] *
                          (if h :
                              (oi.val * stride + di.val = i.val + padding) ∧
                              (oj.val * stride + dj.val = j.val + padding) then
                            getAtOrZero δ [oc.val, oi.val, oj.val] *
                              getAtOrZero layer.kernel [oc.val, ic.val, di.val, dj.val]
                          else 0)) := by
            simpa using
              (sum_comm (α := Fin inC) (β := Fin (outW inW kW stride padding))
                (f := fun ic oj =>
                  ∑ i : Fin inH,
                    ∑ j : Fin inW,
                      ∑ di : Fin kH,
                        ∑ dj : Fin kW,
                          getAtOrZero dInput [ic.val, i.val, j.val] *
                            (if h :
                                (oi.val * stride + di.val = i.val + padding) ∧
                                (oj.val * stride + dj.val = j.val + padding) then
                              getAtOrZero δ [oc.val, oi.val, oj.val] *
                                getAtOrZero layer.kernel [oc.val, ic.val, di.val, dj.val]
                            else 0)))
          exact hC1.trans (hC2.trans hC3)

        have hD :
            (∑ oi : Fin (outH inH kH stride padding),
                ∑ oj : Fin (outW inW kW stride padding),
                  ∑ ic : Fin inC,
                    ∑ i : Fin inH,
                      ∑ j : Fin inW,
                        ∑ di : Fin kH,
                          ∑ dj : Fin kW,
                            getAtOrZero dInput [ic.val, i.val, j.val] *
                              (if h :
                                  (oi.val * stride + di.val = i.val + padding) ∧
                                  (oj.val * stride + dj.val = j.val + padding) then
                                getAtOrZero δ [oc.val, oi.val, oj.val] *
                                  getAtOrZero layer.kernel [oc.val, ic.val, di.val, dj.val]
                              else 0))
              =
            (∑ oi : Fin (outH inH kH stride padding),
                ∑ oj : Fin (outW inW kW stride padding),
                  ∑ ic : Fin inC,
                    ∑ di : Fin kH,
                      ∑ dj : Fin kW,
                        ∑ i : Fin inH,
                          ∑ j : Fin inW,
                            getAtOrZero dInput [ic.val, i.val, j.val] *
                              (if h :
                                  (oi.val * stride + di.val = i.val + padding) ∧
                                  (oj.val * stride + dj.val = j.val + padding) then
                                getAtOrZero δ [oc.val, oi.val, oj.val] *
                                  getAtOrZero layer.kernel [oc.val, ic.val, di.val, dj.val]
                              else 0)) := by
          refine Fintype.sum_congr _ _ ?_
          intro oi
          refine Fintype.sum_congr _ _ ?_
          intro oj
          refine Fintype.sum_congr _ _ ?_
          intro ic
          -- Move `di` out past `(i,j)`, then move `dj` out past `(i,j)`.
          have hD1 :
              (∑ i : Fin inH, ∑ j : Fin inW, ∑ di : Fin kH, ∑ dj : Fin kW,
                    getAtOrZero dInput [ic.val, i.val, j.val] *
                      (if h :
                          (oi.val * stride + di.val = i.val + padding) ∧
                          (oj.val * stride + dj.val = j.val + padding) then
                        getAtOrZero δ [oc.val, oi.val, oj.val] *
                          getAtOrZero layer.kernel [oc.val, ic.val, di.val, dj.val]
                      else 0))
                =
              (∑ di : Fin kH, ∑ i : Fin inH, ∑ j : Fin inW, ∑ dj : Fin kW,
                    getAtOrZero dInput [ic.val, i.val, j.val] *
                      (if h :
                          (oi.val * stride + di.val = i.val + padding) ∧
                          (oj.val * stride + dj.val = j.val + padding) then
                        getAtOrZero δ [oc.val, oi.val, oj.val] *
                          getAtOrZero layer.kernel [oc.val, ic.val, di.val, dj.val]
                      else 0)) := by
            -- swap `j` and `di`, then `i` and `di`
            calc
              (∑ i : Fin inH, ∑ j : Fin inW, ∑ di : Fin kH, ∑ dj : Fin kW,
                    getAtOrZero dInput [ic.val, i.val, j.val] *
                      (if h :
                          (oi.val * stride + di.val = i.val + padding) ∧
                          (oj.val * stride + dj.val = j.val + padding) then
                        getAtOrZero δ [oc.val, oi.val, oj.val] *
                          getAtOrZero layer.kernel [oc.val, ic.val, di.val, dj.val]
                      else 0))
                  =
                (∑ i : Fin inH, ∑ di : Fin kH, ∑ j : Fin inW, ∑ dj : Fin kW,
                    getAtOrZero dInput [ic.val, i.val, j.val] *
                      (if h :
                          (oi.val * stride + di.val = i.val + padding) ∧
                          (oj.val * stride + dj.val = j.val + padding) then
                        getAtOrZero δ [oc.val, oi.val, oj.val] *
                          getAtOrZero layer.kernel [oc.val, ic.val, di.val, dj.val]
                      else 0)) := by
                  refine Fintype.sum_congr _ _ ?_
                  intro i
                  simpa using
                    (sum_comm (α := Fin inW) (β := Fin kH)
                      (f := fun j di =>
                        ∑ dj : Fin kW,
                          getAtOrZero dInput [ic.val, i.val, j.val] *
                            (if h :
                                (oi.val * stride + di.val = i.val + padding) ∧
                                (oj.val * stride + dj.val = j.val + padding) then
                              getAtOrZero δ [oc.val, oi.val, oj.val] *
                                getAtOrZero layer.kernel [oc.val, ic.val, di.val, dj.val]
                            else 0)))
              _ =
                (∑ di : Fin kH, ∑ i : Fin inH, ∑ j : Fin inW, ∑ dj : Fin kW,
                    getAtOrZero dInput [ic.val, i.val, j.val] *
                      (if h :
                          (oi.val * stride + di.val = i.val + padding) ∧
                          (oj.val * stride + dj.val = j.val + padding) then
                        getAtOrZero δ [oc.val, oi.val, oj.val] *
                          getAtOrZero layer.kernel [oc.val, ic.val, di.val, dj.val]
                      else 0)) := by
                  simpa using
                    (sum_comm (α := Fin inH) (β := Fin kH)
                      (f := fun i di =>
                        ∑ j : Fin inW, ∑ dj : Fin kW,
                          getAtOrZero dInput [ic.val, i.val, j.val] *
                            (if h :
                                (oi.val * stride + di.val = i.val + padding) ∧
                                (oj.val * stride + dj.val = j.val + padding) then
                              getAtOrZero δ [oc.val, oi.val, oj.val] *
                                getAtOrZero layer.kernel [oc.val, ic.val, di.val, dj.val]
                            else 0)))
          have hD2 :
              (∑ di : Fin kH, ∑ i : Fin inH, ∑ j : Fin inW, ∑ dj : Fin kW,
                    getAtOrZero dInput [ic.val, i.val, j.val] *
                      (if h :
                          (oi.val * stride + di.val = i.val + padding) ∧
                          (oj.val * stride + dj.val = j.val + padding) then
                        getAtOrZero δ [oc.val, oi.val, oj.val] *
                          getAtOrZero layer.kernel [oc.val, ic.val, di.val, dj.val]
                      else 0))
                =
              (∑ di : Fin kH, ∑ dj : Fin kW, ∑ i : Fin inH, ∑ j : Fin inW,
                    getAtOrZero dInput [ic.val, i.val, j.val] *
                      (if h :
                          (oi.val * stride + di.val = i.val + padding) ∧
                          (oj.val * stride + dj.val = j.val + padding) then
                        getAtOrZero δ [oc.val, oi.val, oj.val] *
                          getAtOrZero layer.kernel [oc.val, ic.val, di.val, dj.val]
                      else 0)) := by
            refine Fintype.sum_congr _ _ ?_
            intro di
            -- swap `j` and `dj`, then `i` and `dj`
            calc
              (∑ i : Fin inH, ∑ j : Fin inW, ∑ dj : Fin kW,
                    getAtOrZero dInput [ic.val, i.val, j.val] *
                      (if h :
                          (oi.val * stride + di.val = i.val + padding) ∧
                          (oj.val * stride + dj.val = j.val + padding) then
                        getAtOrZero δ [oc.val, oi.val, oj.val] *
                          getAtOrZero layer.kernel [oc.val, ic.val, di.val, dj.val]
                      else 0))
                  =
                (∑ i : Fin inH, ∑ dj : Fin kW, ∑ j : Fin inW,
                    getAtOrZero dInput [ic.val, i.val, j.val] *
                      (if h :
                          (oi.val * stride + di.val = i.val + padding) ∧
                          (oj.val * stride + dj.val = j.val + padding) then
                        getAtOrZero δ [oc.val, oi.val, oj.val] *
                          getAtOrZero layer.kernel [oc.val, ic.val, di.val, dj.val]
                      else 0)) := by
                  refine Fintype.sum_congr _ _ ?_
                  intro i
                  simpa using
                    (sum_comm (α := Fin inW) (β := Fin kW)
                      (f := fun j dj =>
                        getAtOrZero dInput [ic.val, i.val, j.val] *
                          (if h :
                              (oi.val * stride + di.val = i.val + padding) ∧
                              (oj.val * stride + dj.val = j.val + padding) then
                            getAtOrZero δ [oc.val, oi.val, oj.val] *
                              getAtOrZero layer.kernel [oc.val, ic.val, di.val, dj.val]
                          else 0)))
              _ =
                (∑ dj : Fin kW, ∑ i : Fin inH, ∑ j : Fin inW,
                    getAtOrZero dInput [ic.val, i.val, j.val] *
                      (if h :
                          (oi.val * stride + di.val = i.val + padding) ∧
                          (oj.val * stride + dj.val = j.val + padding) then
                        getAtOrZero δ [oc.val, oi.val, oj.val] *
                          getAtOrZero layer.kernel [oc.val, ic.val, di.val, dj.val]
                      else 0)) := by
                  simpa using
                    (sum_comm (α := Fin inH) (β := Fin kW)
                      (f := fun i dj =>
                        ∑ j : Fin inW,
                          getAtOrZero dInput [ic.val, i.val, j.val] *
                            (if h :
                                (oi.val * stride + di.val = i.val + padding) ∧
                                (oj.val * stride + dj.val = j.val + padding) then
                              getAtOrZero δ [oc.val, oi.val, oj.val] *
                                getAtOrZero layer.kernel [oc.val, ic.val, di.val, dj.val]
                            else 0)))
          exact hD1.trans hD2

        -- Assemble: first use `hA`/`hB` to move `oi`, then `hC` to move `oj`, then `hD` to move
        -- `di/dj`.
        calc
          (∑ ic : Fin inC, ∑ i : Fin inH, ∑ j : Fin inW,
                ∑ oi : Fin (outH inH kH stride padding),
                  ∑ oj : Fin (outW inW kW stride padding),
                    ∑ di : Fin kH,
                      ∑ dj : Fin kW,
                        getAtOrZero dInput [ic.val, i.val, j.val] *
                          (if h :
                              (oi.val * stride + di.val = i.val + padding) ∧
                              (oj.val * stride + dj.val = j.val + padding) then
                            getAtOrZero δ [oc.val, oi.val, oj.val] *
                              getAtOrZero layer.kernel [oc.val, ic.val, di.val, dj.val]
                          else 0))
              =
            (∑ oi : Fin (outH inH kH stride padding),
                ∑ ic : Fin inC,
                  ∑ i : Fin inH,
                    ∑ j : Fin inW,
                      ∑ oj : Fin (outW inW kW stride padding),
                        ∑ di : Fin kH,
                          ∑ dj : Fin kW,
                            getAtOrZero dInput [ic.val, i.val, j.val] *
                              (if h :
                                  (oi.val * stride + di.val = i.val + padding) ∧
                                  (oj.val * stride + dj.val = j.val + padding) then
                                getAtOrZero δ [oc.val, oi.val, oj.val] *
                                  getAtOrZero layer.kernel [oc.val, ic.val, di.val, dj.val]
                              else 0)) := by
                exact hA.trans hB
          _ =
            (∑ oi : Fin (outH inH kH stride padding),
                ∑ oj : Fin (outW inW kW stride padding),
                  ∑ ic : Fin inC,
                    ∑ i : Fin inH,
                      ∑ j : Fin inW,
                        ∑ di : Fin kH,
                          ∑ dj : Fin kW,
                            getAtOrZero dInput [ic.val, i.val, j.val] *
                              (if h :
                                  (oi.val * stride + di.val = i.val + padding) ∧
                                  (oj.val * stride + dj.val = j.val + padding) then
                                getAtOrZero δ [oc.val, oi.val, oj.val] *
                                  getAtOrZero layer.kernel [oc.val, ic.val, di.val, dj.val]
                              else 0)) := by
                exact hC
          _ =
            (∑ oi : Fin (outH inH kH stride padding),
                ∑ oj : Fin (outW inW kW stride padding),
                  ∑ ic : Fin inC,
                    ∑ di : Fin kH,
                      ∑ dj : Fin kW,
                        ∑ i : Fin inH,
                          ∑ j : Fin inW,
                            getAtOrZero dInput [ic.val, i.val, j.val] *
                              (if h :
                                  (oi.val * stride + di.val = i.val + padding) ∧
                                  (oj.val * stride + dj.val = j.val + padding) then
                                getAtOrZero δ [oc.val, oi.val, oj.val] *
                                  getAtOrZero layer.kernel [oc.val, ic.val, di.val, dj.val]
                              else 0)) := by
                exact hD
      -- Combine the pieces.
      exact h1.trans (h2.trans (h3.trans h4))
    rw [hComm₁]
    -- Evaluate the inner `(i,j)` sum using `sum_shift_eq_paddedInput`, then simplify to `S`.
    refine (by
      -- Work under the outer sums.
      refine Fintype.sum_congr _ _ ?_
      intro oc
      refine Fintype.sum_congr _ _ ?_
      intro oi
      refine Fintype.sum_congr _ _ ?_
      intro oj
      refine Fintype.sum_congr _ _ ?_
      intro ic
      refine Fintype.sum_congr _ _ ?_
      intro di
      refine Fintype.sum_congr _ _ ?_
      intro dj
      -- Define the constant factor chosen by the indicator.
      let c : ℝ :=
        getAtOrZero δ [oc.val, oi.val, oj.val] * getAtOrZero layer.kernel [oc.val, ic.val,
          di.val, dj.val]
      -- Show the inner sum collapses to a padded read times `c`.
      have hInner :
          (∑ i : Fin inH, ∑ j : Fin inW,
              getAtOrZero dInput [ic.val, i.val, j.val] *
                (if h :
                    (oi.val * stride + di.val = i.val + padding) ∧
                    (oj.val * stride + dj.val = j.val + padding) then
                  getAtOrZero δ [oc.val, oi.val, oj.val] *
                    getAtOrZero layer.kernel [oc.val, ic.val, di.val, dj.val]
                else 0))
            =
          getAtOrZero
              (paddedInput (inC := inC) (inH := inH) (inW := inW) (padding := padding) dInput)
              [ic.val, oi.val * stride + di.val, oj.val * stride + dj.val] * c := by
        -- Rewrite each summand to match `sum_shift_eq_paddedInput`, then factor out `c` and apply
        -- the shift lemma.
        have hSummand :
            (fun (i : Fin inH) (j : Fin inW) =>
                getAtOrZero dInput [ic.val, i.val, j.val] *
                  (if h :
                      (oi.val * stride + di.val = i.val + padding) ∧
                      (oj.val * stride + dj.val = j.val + padding) then
                    getAtOrZero δ [oc.val, oi.val, oj.val] *
                      getAtOrZero layer.kernel [oc.val, ic.val, di.val, dj.val]
                  else 0))
              =
            (fun i j =>
                (if (oi.val * stride + di.val = i.val + padding ∧ oj.val * stride + dj.val = j.val +
                  padding) then
                  getAtOrZero dInput [ic.val, i.val, j.val]
                else 0) * c) := by
          funext i j
          by_cases h :
              (oi.val * stride + di.val = i.val + padding) ∧
              (oj.val * stride + dj.val = j.val + padding)
          · simp [c, h]
          · simp [c, h]
        -- Apply the summand rewrite.
        simp []
        -- Factor `c` out of the inner sums and use `sum_shift_eq_paddedInput` to collapse the
        -- indicator sum.
        -- First collapse `j`, then collapse `i`.
        have hPullJ :
            (∑ i : Fin inH, ∑ j : Fin inW,
                (if (oi.val * stride + di.val = i.val + padding ∧ oj.val * stride + dj.val = j.val +
                  padding) then
                    getAtOrZero dInput [ic.val, i.val, j.val]
                  else 0) * c)
              =
            (∑ i : Fin inH, ∑ j : Fin inW,
                if (oi.val * stride + di.val = i.val + padding ∧ oj.val * stride + dj.val = j.val +
                  padding) then
                    getAtOrZero dInput [ic.val, i.val, j.val]
                  else 0) * c := by
          -- Pull `c` out of the `j` sum, then out of the `i` sum (reverse direction of `sum_mul`
          -- twice).
          have hj (i : Fin inH) :
              (∑ j : Fin inW,
                  (if (oi.val * stride + di.val = i.val + padding ∧ oj.val * stride + dj.val = j.val
                    + padding) then
                      getAtOrZero dInput [ic.val, i.val, j.val]
                    else 0) * c)
                =
              (∑ j : Fin inW,
                  if (oi.val * stride + di.val = i.val + padding ∧ oj.val * stride + dj.val = j.val
                    + padding) then
                      getAtOrZero dInput [ic.val, i.val, j.val]
                    else 0) * c := by
            simpa using
              (sum_mul (ι := Fin inW)
                (f := fun j =>
                  if (oi.val * stride + di.val = i.val + padding ∧ oj.val * stride + dj.val = j.val
                    + padding) then
                      getAtOrZero dInput [ic.val, i.val, j.val]
                    else 0)
                (a := c)).symm
          calc
            (∑ i : Fin inH, ∑ j : Fin inW,
                (if (oi.val * stride + di.val = i.val + padding ∧ oj.val * stride + dj.val = j.val +
                  padding) then
                    getAtOrZero dInput [ic.val, i.val, j.val]
                  else 0) * c)
                =
              (∑ i : Fin inH,
                  (∑ j : Fin inW,
                      if (oi.val * stride + di.val = i.val + padding ∧ oj.val * stride + dj.val =
                        j.val + padding) then
                          getAtOrZero dInput [ic.val, i.val, j.val]
                        else 0) * c) := by
                refine Fintype.sum_congr _ _ ?_
                intro i
                simpa using hj (i := i)
            _ =
              (∑ i : Fin inH, ∑ j : Fin inW,
                  if (oi.val * stride + di.val = i.val + padding ∧ oj.val * stride + dj.val = j.val
                    + padding) then
                      getAtOrZero dInput [ic.val, i.val, j.val]
                    else 0) * c := by
                simpa using
                  (sum_mul (ι := Fin inH)
                    (f := fun i =>
                      ∑ j : Fin inW,
                        if (oi.val * stride + di.val = i.val + padding ∧ oj.val * stride + dj.val =
                          j.val + padding) then
                            getAtOrZero dInput [ic.val, i.val, j.val]
                          else 0)
                    (a := c)).symm
        -- Now apply `sum_shift_eq_paddedInput` to the indicator sum (without `c`) and finish.
        have hShift :
            (∑ i : Fin inH, ∑ j : Fin inW,
                if (oi.val * stride + di.val = i.val + padding ∧ oj.val * stride + dj.val = j.val +
                  padding) then
                  getAtOrZero dInput [ic.val, i.val, j.val]
                else 0)
              =
            getAtOrZero
                (paddedInput (inC := inC) (inH := inH) (inW := inW) (padding := padding) dInput)
                [ic.val, oi.val * stride + di.val, oj.val * stride + dj.val] := by
          -- This is exactly `sum_shift_eq_paddedInput` with `p = oi*stride+di`, `q = oj*stride+dj`.
          simpa [and_left_comm, and_assoc, and_comm] using
            (sum_shift_eq_paddedInput (x := dInput) (ic := ic)
              (p := oi.val * stride + di.val) (q := oj.val * stride + dj.val))
        -- Combine: first rewrite `ite`-products into the form `(..) * c`, then use `hPullJ` and
        -- `hShift`.
        have hIteMul :
            (∑ i : Fin inH, ∑ j : Fin inW,
                if (oi.val * stride + di.val = i.val + padding ∧ oj.val * stride + dj.val = j.val +
                  padding) then
                  getAtOrZero dInput [ic.val, i.val, j.val] * c
                else 0)
              =
            (∑ i : Fin inH, ∑ j : Fin inW,
                (if (oi.val * stride + di.val = i.val + padding ∧ oj.val * stride + dj.val = j.val +
                  padding) then
                  getAtOrZero dInput [ic.val, i.val, j.val]
                else 0) * c) := by
          refine Fintype.sum_congr _ _ ?_
          intro i
          refine Fintype.sum_congr _ _ ?_
          intro j
          by_cases h :
              (oi.val * stride + di.val = i.val + padding ∧ oj.val * stride + dj.val = j.val +
                padding)
          · simp [h]
          · simp [h]
        calc
          (∑ i : Fin inH, ∑ j : Fin inW,
              if (oi.val * stride + di.val = i.val + padding ∧ oj.val * stride + dj.val = j.val +
                padding) then
                getAtOrZero dInput [ic.val, i.val, j.val] * c
              else 0)
              =
            (∑ i : Fin inH, ∑ j : Fin inW,
                (if (oi.val * stride + di.val = i.val + padding ∧ oj.val * stride + dj.val = j.val +
                  padding) then
                  getAtOrZero dInput [ic.val, i.val, j.val]
                else 0) * c) := hIteMul
          _ =
            (∑ i : Fin inH, ∑ j : Fin inW,
                if (oi.val * stride + di.val = i.val + padding ∧ oj.val * stride + dj.val = j.val +
                  padding) then
                  getAtOrZero dInput [ic.val, i.val, j.val]
                else 0) * c := hPullJ
          _ =
            getAtOrZero
                (paddedInput (inC := inC) (inH := inH) (inW := inW) (padding := padding) dInput)
                [ic.val, oi.val * stride + di.val, oj.val * stride + dj.val] * c := by
              simp [hShift]
      -- Use the collapsed inner sum and simplify to `S`.
      -- Replace `c` and reorder multiplications to match `S`.
      have hReorder :
          getAtOrZero
              (paddedInput (inC := inC) (inH := inH) (inW := inW) (padding := padding) dInput)
              [ic.val, oi.val * stride + di.val, oj.val * stride + dj.val] * c
            =
          getAtOrZero layer.kernel [oc.val, ic.val, di.val, dj.val] *
              getAtOrZero
                (paddedInput (inC := inC) (inH := inH) (inW := inW) (padding := padding) dInput)
                [ic.val, oi.val * stride + di.val, oj.val * stride + dj.val] *
              getAtOrZero δ [oc.val, oi.val, oj.val] := by
        -- Avoid `simp` with `mul_comm`: it rewrites `Nat` multiplication inside indices.
        dsimp [c]
        -- Let-bind the scalar factors to keep rewrites local.
        set x :=
          getAtOrZero
            (paddedInput (inC := inC) (inH := inH) (inW := inW) (padding := padding) dInput)
            [ic.val, oi.val * stride + di.val, oj.val * stride + dj.val]
        set d := getAtOrZero δ [oc.val, oi.val, oj.val]
        set k := getAtOrZero layer.kernel [oc.val, ic.val, di.val, dj.val]
        -- Goal becomes `x * (d * k) = k * x * d`.
        calc
          x * (d * k) = x * (k * d) := by
            rw [mul_comm d k]
          _ = (x * k) * d := by
            rw [mul_assoc x k d]
          _ = (k * x) * d := by
            rw [mul_comm x k]
          _ = k * x * d := by
            rw [mul_assoc k x d]
      -- Combine `hInner` with the scalar reordering, then unfold the `S` summand.
      have : (∑ i : Fin inH, ∑ j : Fin inW,
          getAtOrZero dInput [ic.val, i.val, j.val] *
            (if h :
                (oi.val * stride + di.val = i.val + padding) ∧
                (oj.val * stride + dj.val = j.val + padding) then
              getAtOrZero δ [oc.val, oi.val, oj.val] *
                getAtOrZero layer.kernel [oc.val, ic.val, di.val, dj.val]
            else 0))
          =
          getAtOrZero layer.kernel [oc.val, ic.val, di.val, dj.val] *
              getAtOrZero
                (paddedInput (inC := inC) (inH := inH) (inW := inW) (padding := padding) dInput)
                [ic.val, oi.val * stride + di.val, oj.val * stride + dj.val] *
              getAtOrZero δ [oc.val, oi.val, oj.val] := by
        exact hInner.trans hReorder
      simpa [S] using this
    )

  exact hLHS.trans hRHS.symm

/--
Main dot-level bridge theorem for Conv2D.

It states that the triple returned by `Spec.conv2d_backward_spec` is the adjoint (w.r.t. `Spec.dot`)
of the corresponding forward-mode directional derivatives with respect to `(kernel, bias, input)`.

This is the key lemma connecting the handwritten runtime backward to the analytic “VJP is adjoint
of `fderiv`” theorem in `NN.Proofs.Autograd.Tape.Ops.Conv.FDeriv`.
-/
theorem conv2d_backward_spec_dot
    {inC outC kH kW stride padding inH inW : Nat}
    {h1 : inC ≠ 0} {h2 : kH ≠ 0} {h3 : kW ≠ 0}
    (layer : Spec.Conv2DSpec inC outC kH kW stride padding ℝ h1 h2 h3)
    (input : Spec.MultiChannelImage inC inH inW ℝ)
    (δ : Spec.MultiChannelImage outC (outH inH kH stride padding) (outW inW kW stride padding) ℝ)
    (dKernel : Tensor ℝ (.dim outC (.dim inC (.dim kH (.dim kW .scalar)))))
    (dBias : Tensor ℝ (.dim outC .scalar))
    (dInput : Spec.MultiChannelImage inC inH inW ℝ) :
    let layerK : Spec.Conv2DSpec inC outC kH kW stride padding ℝ h1 h2 h3 :=
      { kernel := dKernel, bias := fill (0 : ℝ) (.dim outC .scalar) }
    let layer0 : Spec.Conv2DSpec inC outC kH kW stride padding ℝ h1 h2 h3 :=
      { kernel := layer.kernel, bias := fill (0 : ℝ) (.dim outC .scalar) }
    let jvp : Spec.MultiChannelImage outC (outH inH kH stride padding) (outW inW kW stride padding)
      ℝ :=
      addSpec
        (Spec.conv2dSpec (α := ℝ) (layer := layerK) input)
        (addSpec
          (biasBroadcast (outC := outC) (outH := outH inH kH stride padding) (outW := outW inW kW
            stride padding) dBias)
          (Spec.conv2dSpec (α := ℝ) (layer := layer0) dInput))
    let grads := Spec.conv2dBackwardSpec (α := ℝ) (layer := layer) (input := input) (grad_output
      := δ)
    dot jvp δ =
      dot dKernel grads.1 + dot dBias grads.2.1 + dot dInput grads.2.2 := by
  intro layerK layer0 jvp grads
  -- Expand the dot across the JVP sum and apply the three component dot-bridge lemmas.
  have hK : dot (Spec.conv2dSpec (α := ℝ) (layer := layerK) input) δ =
      dot dKernel (Spec.conv2dKernelDerivSpec (α := ℝ) (layer := layer) (input := input)
        (grad_output := δ)) := by
    simpa using (dot_conv2d_kernel (layer := layer) (input := input) (dKernel := dKernel) (δ := δ))
  have hB :
      dot (biasBroadcast (outC := outC) (outH := outH inH kH stride padding) (outW := outW inW kW
        stride padding) dBias) δ
        =
      dot dBias (Spec.conv2dBiasDerivSpec (α := ℝ) (layer := layer) (input := input)
        (grad_output := δ)) := by
    simpa using (dot_biasBroadcast_eq_dot_bias_deriv (layer := layer) (input := input) (db := dBias)
      (δ := δ))
  have hX :
      dot (Spec.conv2dSpec (α := ℝ) (layer := layer0) dInput) δ
        =
      dot dInput (Spec.conv2dInputDerivSpec (α := ℝ) (layer := layer) (input := input)
        (grad_output := δ)) := by
    simpa using (dot_conv2d_input (layer := layer) (input := input) (dInput := dInput) (δ := δ))
  -- Now finish by splitting the dot on `jvp` and rewriting `grads`.
  subst jvp
  -- Split dot over the nested additions, then rewrite using `hK`/`hB`/`hX` and unfold `grads`.
  have hsplit :
      dot
          (addSpec
            (Spec.conv2dSpec (α := ℝ) (layer := layerK) input)
            ((biasBroadcast (outC := outC) (outH := outH inH kH stride padding) (outW := outW inW kW
              stride padding) dBias).addSpec
              (Spec.conv2dSpec (α := ℝ) (layer := layer0) dInput)))
          δ
        =
      dot (Spec.conv2dSpec (α := ℝ) (layer := layerK) input) δ +
        dot (biasBroadcast (outC := outC) (outH := outH inH kH stride padding) (outW := outW inW kW
          stride padding) dBias) δ +
        dot (Spec.conv2dSpec (α := ℝ) (layer := layer0) dInput) δ := by
    -- Expand dot across the outer `+`, then across the inner `+`.
    calc
      dot
          (addSpec
            (Spec.conv2dSpec (α := ℝ) (layer := layerK) input)
            ((biasBroadcast (outC := outC) (outH := outH inH kH stride padding) (outW := outW inW kW
              stride padding) dBias).addSpec
              (Spec.conv2dSpec (α := ℝ) (layer := layer0) dInput)))
          δ
          =
      dot (Spec.conv2dSpec (α := ℝ) (layer := layerK) input) δ +
          dot
              ((biasBroadcast (outC := outC) (outH := outH inH kH stride padding) (outW := outW inW
                kW stride padding) dBias).addSpec
                (Spec.conv2dSpec (α := ℝ) (layer := layer0) dInput))
              δ := by
                -- Apply the dot-add lemma once (don’t rewrite the RHS further).
                simpa using
                  (dot_add_left
                    (a := Spec.conv2dSpec (α := ℝ) (layer := layerK) input)
                    (b :=
                      (biasBroadcast (outC := outC) (outH := outH inH kH stride padding)
                        (outW := outW inW kW stride padding) dBias).addSpec
                        (Spec.conv2dSpec (α := ℝ) (layer := layer0) dInput))
                    (c := δ))
      _ =
        dot (Spec.conv2dSpec (α := ℝ) (layer := layerK) input) δ +
          (dot (biasBroadcast (outC := outC) (outH := outH inH kH stride padding) (outW := outW inW
            kW stride padding) dBias) δ +
            dot (Spec.conv2dSpec (α := ℝ) (layer := layer0) dInput) δ) := by
              -- Use the dot-add lemma directly for the inner sum.
              simpa using
                (dot_add_left
                  (a := biasBroadcast (outC := outC) (outH := outH inH kH stride padding) (outW :=
                    outW inW kW stride padding) dBias)
                  (b := Spec.conv2dSpec (α := ℝ) (layer := layer0) dInput)
                  (c := δ))
      _ =
        dot (Spec.conv2dSpec (α := ℝ) (layer := layerK) input) δ +
          dot (biasBroadcast (outC := outC) (outH := outH inH kH stride padding) (outW := outW inW
            kW stride padding) dBias) δ +
          dot (Spec.conv2dSpec (α := ℝ) (layer := layer0) dInput) δ := by
            ring
  -- Rewrite the dot terms using the component bridges, then unfold `grads` and reassociate.
  calc
    dot
        (addSpec
          (Spec.conv2dSpec (α := ℝ) (layer := layerK) input)
          ((biasBroadcast (outC := outC) (outH := outH inH kH stride padding) (outW := outW inW kW
            stride padding) dBias).addSpec
            (Spec.conv2dSpec (α := ℝ) (layer := layer0) dInput)))
        δ
        =
      dot (Spec.conv2dSpec (α := ℝ) (layer := layerK) input) δ +
        dot (biasBroadcast (outC := outC) (outH := outH inH kH stride padding) (outW := outW inW kW
          stride padding) dBias) δ +
        dot (Spec.conv2dSpec (α := ℝ) (layer := layer0) dInput) δ := by
          simpa using hsplit
    _ =
      dot dKernel (Spec.conv2dKernelDerivSpec (α := ℝ) (layer := layer) (input := input)
        (grad_output := δ)) +
        dot dBias (Spec.conv2dBiasDerivSpec (α := ℝ) (layer := layer) (input := input)
          (grad_output := δ)) +
        dot dInput (Spec.conv2dInputDerivSpec (α := ℝ) (layer := layer) (input := input)
          (grad_output := δ)) := by
          -- rewrite each term using the corresponding dot bridge lemma
          simp [hK, hB, hX, add_assoc, add_comm]
      _ = dot dKernel grads.1 + dot dBias grads.2.1 + dot dInput grads.2.2 := by
            simp [grads, Spec.conv2dBackwardSpec]
end

end Conv2D
end Autograd
end Proofs
