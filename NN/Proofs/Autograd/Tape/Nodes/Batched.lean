/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Proofs.Autograd.Tape.Nodes.GraphComposition

/-!
# Batched

Additional `HasFDerivAt`-level nodes for **batched (3D) ops**.

These are useful for `MultiHeadAttention` graphs where the head dimension is explicit:
- batched matrix multiplication: `(h×m×n) × (h×n×p) → (h×m×p)`
- batched (row-wise) softmax: `h × (m×n) → h × (m×n)`

All results here are spec-level over `ℝ`.
-/

@[expose] public section


namespace Proofs
namespace Autograd

open Spec
open Tensor

noncomputable section

open scoped BigOperators

namespace TapeNodes

namespace Batched

-- ---------------------------------------------------------------------------
-- Head splitting/combining on flattened vectors
-- ---------------------------------------------------------------------------

/--
Split a flattened `h * n` vector into `h` “heads” of length `n`.

This is the vector-level analogue of reshaping `(..., h*n)` into `(..., h, n)`.
It is used to define batched operations as head-wise operations.
-/
def heads {h n : Nat} (x : Vec (h * n)) : Fin h → Vec n :=
  fun head => vecOfFun (n := n) fun j => x (finProdFinEquiv (head, j))

/-- Inverse of `heads`: concatenate head vectors back into one flattened vector. -/
def unheads {h n : Nat} (r : Fin h → Vec n) : Vec (h * n) :=
  vecOfFun (n := h * n) fun ip =>
    let p : Fin h × Fin n := (finProdFinEquiv : Fin h × Fin n ≃ Fin (h * n)).symm ip
    r p.1 p.2

/-- Continuous linear map version of `heads`. -/
def headsCLM {h n : Nat} : Vec (h * n) →L[ℝ] (Fin h → Vec n) := by
  classical
  let fLin : Vec (h * n) →ₗ[ℝ] (Fin h → Vec n) :=
    { toFun := heads (h := h) (n := n)
      map_add' := by
        intro x y
        funext head
        ext j
        simp [heads, Pi.add_apply]
      map_smul' := by
        intro a x
        funext head
        ext j
        simp [heads, Pi.smul_apply, smul_eq_mul] }
  refine ⟨fLin, ?_⟩
  exact LinearMap.continuous_of_finiteDimensional (f := fLin)

/-- Continuous linear map version of `unheads`. -/
def unheadsCLM {h n : Nat} : (Fin h → Vec n) →L[ℝ] Vec (h * n) := by
  classical
  let fLin : (Fin h → Vec n) →ₗ[ℝ] Vec (h * n) :=
    { toFun := unheads (h := h) (n := n)
      map_add' := by
        intro r₁ r₂
        ext ip
        simp [unheads, Pi.add_apply]
      map_smul' := by
        intro a r
        ext ip
        simp [unheads, Pi.smul_apply, smul_eq_mul] }
  refine ⟨fLin, ?_⟩
  exact LinearMap.continuous_of_finiteDimensional (f := fLin)

@[simp] lemma headsCLM_apply {h n : Nat} (x : Vec (h * n)) :
    headsCLM (h := h) (n := n) x = heads (h := h) (n := n) x := by
  rfl

@[simp] lemma unheadsCLM_apply {h n : Nat} (r : Fin h → Vec n) :
    unheadsCLM (h := h) (n := n) r = unheads (h := h) (n := n) r := by
  rfl

lemma size3_eq (h m n : Nat) :
    Shape.size (.dim h (.dim m (.dim n .scalar))) = h * (m * n) := by
  simp [Shape.size]

-- ---------------------------------------------------------------------------
-- Batched matmul (bilinear)
-- ---------------------------------------------------------------------------

/-- Flattened size of `h` many `m×n` matrices (row-major): `h * (m*n)`. -/
abbrev HMatSize (h m n : Nat) : Nat := h * Matmul.matSize m n

set_option maxHeartbeats 1000000 in
/-- Bilinear map for batched matmul, packaged as `A →L (B →L A ⬝ B)` in flattened form. -/
def matmulBilin {h m n p : Nat} :
    Vec (HMatSize h m n) →L[ℝ] Vec (HMatSize h n p) →L[ℝ] Vec (HMatSize h m p) := by
  classical
  let B : Vec (Matmul.matSize m n) →L[ℝ] Vec (Matmul.matSize n p) →L[ℝ] Vec (Matmul.matSize m p) :=
    Matmul.matmulBilin (m := m) (n := n) (p := p)
  let LA : Vec (HMatSize h m n) →L[ℝ] (Fin h → Vec (Matmul.matSize m n)) :=
    headsCLM (h := h) (n := Matmul.matSize m n)
  let LB : Vec (HMatSize h n p) →L[ℝ] (Fin h → Vec (Matmul.matSize n p)) :=
    headsCLM (h := h) (n := Matmul.matSize n p)
  let LC : (Fin h → Vec (Matmul.matSize m p)) →L[ℝ] Vec (HMatSize h m p) :=
    unheadsCLM (h := h) (n := Matmul.matSize m p)

  -- Headwise bilinear map `(Fin h → A) × (Fin h → B) → (Fin h → C)`.
  let BH : (Fin h → Vec (Matmul.matSize m n)) →L[ℝ]
      (Fin h → Vec (Matmul.matSize n p)) →L[ℝ]
        (Fin h → Vec (Matmul.matSize m p)) := by
    classical
    let fLin : (Fin h → Vec (Matmul.matSize m n)) →ₗ[ℝ]
        (Fin h → Vec (Matmul.matSize n p)) →L[ℝ] (Fin h → Vec (Matmul.matSize m p)) :=
      { toFun := fun aH =>
          ContinuousLinearMap.pi (R := ℝ) (fun head : Fin h =>
            ((B (aH head)).comp (ContinuousLinearMap.proj (R := ℝ) head)))
        map_add' := by
          intro a1 a2
          ext bH head ip
          simp [ContinuousLinearMap.pi_apply, ContinuousLinearMap.proj_apply,
            ContinuousLinearMap.comp_apply,
            B, ContinuousLinearMap.add_apply, Pi.add_apply]
        map_smul' := by
          intro r a1
          ext bH head ip
          simp [ContinuousLinearMap.pi_apply, ContinuousLinearMap.proj_apply,
            ContinuousLinearMap.comp_apply,
            B, ContinuousLinearMap.smul_apply, Pi.smul_apply, smul_eq_mul] }
    refine ⟨fLin, ?_⟩
    exact LinearMap.continuous_of_finiteDimensional (f := fLin)

  -- Flatten the headwise map back to the row-major vector representation.
  let fLin : Vec (HMatSize h m n) →ₗ[ℝ] Vec (HMatSize h n p) →L[ℝ] Vec (HMatSize h m p) :=
    { toFun := fun aFlat =>
        LC.comp (((BH (LA aFlat)).comp LB))
      map_add' := by
        intro a1 a2
        ext bFlat ip
        -- Decode `ip` into a head index and an intra-head matrix index, then use linearity of `B`.
        let q : Fin h × Fin (Matmul.matSize m p) :=
          (finProdFinEquiv : Fin h × Fin (Matmul.matSize m p) ≃ Fin (h * Matmul.matSize m p)).symm
            ip
        have hheads :
            heads (h := h) (n := Matmul.matSize m n) (a1 + a2) q.1 =
              heads (h := h) (n := Matmul.matSize m n) a1 q.1 +
                heads (h := h) (n := Matmul.matSize m n) a2 q.1 := by
          ext j
          simp [heads]
        have hB' :
            B (heads (h := h) (n := Matmul.matSize m n) (a1 + a2) q.1)
              =
            B (heads (h := h) (n := Matmul.matSize m n) a1 q.1) +
              B (heads (h := h) (n := Matmul.matSize m n) a2 q.1) := by
          rw [hheads]
          exact
            B.map_add
              (heads (h := h) (n := Matmul.matSize m n) a1 q.1)
              (heads (h := h) (n := Matmul.matSize m n) a2 q.1)
        have hBapp :=
          congrArg (fun F => F (heads (h := h) (n := Matmul.matSize n p) bFlat q.1)) hB'
        have hBcoord := congrArg (fun v => v q.2) hBapp
        -- Expand the `unheads` projection at `ip` and finish.
        simp [heads, q, ContinuousLinearMap.add_apply] at hBcoord
        exact hBcoord
      map_smul' := by
        intro r a1
        ext bFlat ip
        let q : Fin h × Fin (Matmul.matSize m p) :=
          (finProdFinEquiv : Fin h × Fin (Matmul.matSize m p) ≃ Fin (h * Matmul.matSize m p)).symm
            ip
        have hheads :
            heads (h := h) (n := Matmul.matSize m n) (r • a1) q.1 =
              r • heads (h := h) (n := Matmul.matSize m n) a1 q.1 := by
          ext j
          simp [heads, smul_eq_mul]
        have hB' :
            B (heads (h := h) (n := Matmul.matSize m n) (r • a1) q.1)
              =
            r • B (heads (h := h) (n := Matmul.matSize m n) a1 q.1) := by
          rw [hheads]
          exact B.map_smulₛₗ r (heads (h := h) (n := Matmul.matSize m n) a1 q.1)
        have hBapp :=
          congrArg (fun F => F (heads (h := h) (n := Matmul.matSize n p) bFlat q.1)) hB'
        have hBcoord := congrArg (fun v => v q.2) hBapp
        simp [heads, q, ContinuousLinearMap.smul_apply, smul_eq_mul] at hBcoord
        exact hBcoord }

  refine ⟨fLin, ?_⟩
  exact LinearMap.continuous_of_finiteDimensional (f := fLin)

/--
Batched matmul node (head-wise): `(h×m×n) × (h×n×p) → (h×m×p)`.

PyTorch analogue: `torch.matmul` with leading batch dimension `h`.
https://pytorch.org/docs/stable/generated/torch.matmul.html
-/
def matmul {Γ : List Shape} {h m n p : Nat}
    (A : Idx Γ (.dim h (.dim m (.dim n .scalar))))
    (B : Idx Γ (.dim h (.dim n (.dim p .scalar)))) :
    Node Γ (.dim h (.dim m (.dim p .scalar))) :=
  let sA : Shape := .dim h (.dim m (.dim n .scalar))
  let sB : Shape := .dim h (.dim n (.dim p .scalar))
  let sOut : Shape := .dim h (.dim m (.dim p .scalar))
  let Bmul : Vec (Shape.size sA) →L[ℝ] Vec (Shape.size sB) →L[ℝ] Vec (Shape.size sOut) :=
    matmulBilin (h := h) (m := m) (n := n) (p := p)
  let fA : CtxVec Γ → Vec (Shape.size sA) :=
    fun x => CtxVec.get (Γ := Γ) (s := sA) A x
  let fB : CtxVec Γ → Vec (Shape.size sB) :=
    fun x => CtxVec.get (Γ := Γ) (s := sB) B x
  let deriv0 : CtxVec Γ → (CtxVec Γ →L[ℝ] Vec (Shape.size sOut)) :=
    fun x =>
      (Bmul.precompR (CtxVec Γ) (fA x) (CtxVec.getCLM (Γ := Γ) (s := sB) B)) +
        (Bmul.precompL (CtxVec Γ) (CtxVec.getCLM (Γ := Γ) (s := sA) A) (fB x))
  Node.ofVec (Γ := Γ) (τ := sOut)
    (f := fun x => (Bmul (fA x)) (fB x))
    (jvp := fun x dx => (deriv0 x) dx)
    (vjp := fun x δ => (deriv0 x).adjoint δ)
    (correct_inner := by
      intro x dx δ
      simpa using (ContinuousLinearMap.adjoint_inner_right (A := deriv0 x) (x := dx) (y := δ)).symm)

/-- `NodeFDerivCorrect` for the batched matmul node. -/
def matmulFderiv {Γ : List Shape} {h m n p : Nat}
    (A : Idx Γ (.dim h (.dim m (.dim n .scalar))))
    (B : Idx Γ (.dim h (.dim n (.dim p .scalar)))) :
    NodeFDerivCorrect (matmul (Γ := Γ) (h := h) (m := m) (n := n) (p := p) A B) := by
  classical
  let sA : Shape := .dim h (.dim m (.dim n .scalar))
  let sB : Shape := .dim h (.dim n (.dim p .scalar))
  let sOut : Shape := .dim h (.dim m (.dim p .scalar))
  let Bmul : Vec (Shape.size sA) →L[ℝ] Vec (Shape.size sB) →L[ℝ] Vec (Shape.size sOut) :=
    matmulBilin (h := h) (m := m) (n := n) (p := p)
  let fA : CtxVec Γ → Vec (Shape.size sA) :=
    fun x => CtxVec.get (Γ := Γ) (s := sA) A x
  let fB : CtxVec Γ → Vec (Shape.size sB) :=
    fun x => CtxVec.get (Γ := Γ) (s := sB) B x

  refine
    { deriv := fun x =>
        (Bmul.precompR (CtxVec Γ) (fA x) (CtxVec.getCLM (Γ := Γ) (s := sB) B)) +
          (Bmul.precompL (CtxVec Γ) (CtxVec.getCLM (Γ := Γ) (s := sA) A) (fB x))
      hasFDerivAt := ?_
      jvp_eq := ?_ }
  · intro xV
    have hA :
        HasFDerivAt fA (CtxVec.getCLM (Γ := Γ) (s := sA) A) xV := by
      have h := (CtxVec.getCLM (Γ := Γ) (s := sA) A).hasFDerivAt (x := xV)
      have hfun : fA = fun x => (CtxVec.getCLM (Γ := Γ) (s := sA) A) x := by
        funext x; simp [fA, CtxVec.getCLM_apply]
      exact h.congr_of_eventuallyEq hfun.eventuallyEq
    have hB :
        HasFDerivAt fB (CtxVec.getCLM (Γ := Γ) (s := sB) B) xV := by
      have h := (CtxVec.getCLM (Γ := Γ) (s := sB) B).hasFDerivAt (x := xV)
      have hfun : fB = fun x => (CtxVec.getCLM (Γ := Γ) (s := sB) B) x := by
        funext x; simp [fB, CtxVec.getCLM_apply]
      exact h.congr_of_eventuallyEq hfun.eventuallyEq

    have hbilin :=
      ContinuousLinearMap.hasFDerivAt_of_bilinear (B := Bmul) (hf := hA) (hg := hB)
    have hEq :
        (Node.forwardVec (Γ := Γ) (τ := sOut)
            (matmul (Γ := Γ) (h := h) (m := m) (n := n) (p := p) A B))
          =
        (fun x : CtxVec Γ => (Bmul (fA x)) (fB x)) := by
      funext x
      simp [matmul, Node.forwardVec_ofVec, Bmul, fA, fB]
    exact hbilin.congr_of_eventuallyEq hEq.eventuallyEq
  · intro xV dxV
    ext ip
    -- First unfold the node JVP (LHS) into the explicit bilinear JVP formula.
    simp [matmul, Node.jvpVec_ofVec, Bmul, fA, fB]

-- ---------------------------------------------------------------------------
-- Batched row-wise softmax on matrices
-- ---------------------------------------------------------------------------

/--
Batched row-wise softmax node: apply `softmax_last` independently per head.

Shape: `h × (m×n) → h × (m×n)`, where each head contains an `m×n` matrix and softmax is along the
last axis (size `n`) within each row.

PyTorch analogue: `torch.nn.functional.softmax(x, dim=-1)` with a leading batch dimension.
https://pytorch.org/docs/stable/generated/torch.nn.functional.softmax.html
-/
def softmaxLast {Γ : List Shape} {h m n : Nat}
    (idx : Idx Γ (.dim h (.dim m (.dim n .scalar)))) :
    Node Γ (.dim h (.dim m (.dim n .scalar))) :=
  let s : Shape := .dim h (.dim m (.dim n .scalar))
  let hsz : Shape.size s = h * (m * n) := size3_eq (h := h) (m := m) (n := n)
  let forward0 : Vec (h * (m * n)) → Vec (h * (m * n)) :=
    fun x =>
      unheads (h := h) (n := m * n) (fun head =>
        SoftmaxLastAxis.forwardMN (m := m) (n := n) (heads (h := h) (n := m * n) x head))
  let deriv0 : Vec (h * (m * n)) → Vec (h * (m * n)) →L[ℝ] Vec (h * (m * n)) :=
    fun x =>
      let xH : Fin h → Vec (m * n) := heads (h := h) (n := m * n) x
      let DG : (Fin h → Vec (m * n)) →L[ℝ] (Fin h → Vec (m * n)) :=
        ContinuousLinearMap.pi (R := ℝ) (fun head : Fin h =>
          (SoftmaxLastAxis.derivMN (m := m) (n := n) (xH head)).comp
            (ContinuousLinearMap.proj (R := ℝ) head))
      (unheadsCLM (h := h) (n := m * n)).comp (DG.comp (headsCLM (h := h) (n := m * n)))
  let D : CtxVec Γ → (CtxVec Γ →L[ℝ] Vec (Shape.size s)) :=
    fun xV =>
      (Graph.castCLM (h := hsz.symm)).comp
        ((deriv0 (castVec hsz (CtxVec.get (Γ := Γ) (s := s) idx xV))).comp
          ((Graph.castCLM (h := hsz)).comp (CtxVec.getCLM (Γ := Γ) (s := s) idx)))

  Node.ofVec (Γ := Γ) (τ := s)
    (f := fun xV =>
      castVec hsz.symm (forward0 (castVec hsz (CtxVec.get (Γ := Γ) (s := s) idx xV))))
    (jvp := fun xV dxV => (D xV) dxV)
    (vjp := fun xV δV => (D xV).adjoint δV)
    (correct_inner := by
      intro xV dxV δV
      simpa using (ContinuousLinearMap.adjoint_inner_right (A := D xV) (x := dxV) (y := δV)).symm)

/-- `NodeFDerivCorrect` for `softmax_last` in the batched/head-wise setting. -/
def softmaxLastFderiv {Γ : List Shape} {h m n : Nat}
    (idx : Idx Γ (.dim h (.dim m (.dim n .scalar)))) :
    NodeFDerivCorrect (softmaxLast (Γ := Γ) (h := h) (m := m) (n := n) idx) := by
  classical
  let s : Shape := .dim h (.dim m (.dim n .scalar))
  let hsz : Shape.size s = h * (m * n) := size3_eq (h := h) (m := m) (n := n)
  let forward0 : Vec (h * (m * n)) → Vec (h * (m * n)) :=
    fun x =>
      unheads (h := h) (n := m * n) (fun head =>
        SoftmaxLastAxis.forwardMN (m := m) (n := n) (heads (h := h) (n := m * n) x head))
  let deriv0 : Vec (h * (m * n)) → Vec (h * (m * n)) →L[ℝ] Vec (h * (m * n)) :=
    fun x =>
      let xH : Fin h → Vec (m * n) := heads (h := h) (n := m * n) x
      let DG : (Fin h → Vec (m * n)) →L[ℝ] (Fin h → Vec (m * n)) :=
        ContinuousLinearMap.pi (R := ℝ) (fun head : Fin h =>
          (SoftmaxLastAxis.derivMN (m := m) (n := n) (xH head)).comp
            (ContinuousLinearMap.proj (R := ℝ) head))
      (unheadsCLM (h := h) (n := m * n)).comp (DG.comp (headsCLM (h := h) (n := m * n)))
  let D : CtxVec Γ → (CtxVec Γ →L[ℝ] Vec (Shape.size s)) :=
    fun xV =>
      (Graph.castCLM (h := hsz.symm)).comp
        ((deriv0 (castVec hsz (CtxVec.get (Γ := Γ) (s := s) idx xV))).comp
          ((Graph.castCLM (h := hsz)).comp (CtxVec.getCLM (Γ := Γ) (s := s) idx)))

  refine
    { deriv := fun xV => D xV
      hasFDerivAt := ?_
      jvp_eq := ?_ }
  · intro xV
    -- Middle map: apply the matrix-softmax independently to each head.
    let G : (Fin h → Vec (m * n)) → (Fin h → Vec (m * n)) :=
      fun r head => SoftmaxLastAxis.forwardMN (m := m) (n := n) (r head)
    let x0 : Vec (h * (m * n)) := castVec hsz (CtxVec.get (Γ := Γ) (s := s) idx xV)
    let r0 : Fin h → Vec (m * n) := heads (h := h) (n := m * n) x0
    let DG : (Fin h → Vec (m * n)) →L[ℝ] (Fin h → Vec (m * n)) :=
      ContinuousLinearMap.pi (R := ℝ) (fun head : Fin h =>
        (SoftmaxLastAxis.derivMN (m := m) (n := n) (r0 head)).comp (ContinuousLinearMap.proj (R :=
          ℝ) head))

    have hG : HasFDerivAt G DG r0 := by
      refine (hasFDerivAt_pi (𝕜 := ℝ)
        (φ := fun head : Fin h => fun r : (Fin h → Vec (m * n)) =>
          SoftmaxLastAxis.forwardMN (m := m) (n := n) (r head))
        (φ' := fun head : Fin h =>
          (SoftmaxLastAxis.derivMN (m := m) (n := n) (r0 head)).comp
            (ContinuousLinearMap.proj (R := ℝ) head))
        (x := r0)).2 ?_
      intro head
      have hsoft :
          HasFDerivAt (SoftmaxLastAxis.forwardMN (m := m) (n := n))
            (SoftmaxLastAxis.derivMN (m := m) (n := n) (r0 head)) (r0 head) :=
        SoftmaxLastAxis.hasFDerivAt_forwardMN (m := m) (n := n) (x := r0 head)
      have happly :
          HasFDerivAt (fun r : (Fin h → Vec (m * n)) => r head)
            (ContinuousLinearMap.proj (R := ℝ) head) r0 := by
        simpa using ((ContinuousLinearMap.proj (R := ℝ) head).hasFDerivAt (x := r0))
      exact hsoft.comp r0 happly

    -- Linear reshapes/casts around the middle map.
    have hget :
        HasFDerivAt (fun x : CtxVec Γ => castVec hsz (CtxVec.get (Γ := Γ) (s := s) idx x))
          ((Graph.castCLM (h := hsz)).comp (CtxVec.getCLM (Γ := Γ) (s := s) idx)) xV := by
      have hlin :=
        ((Graph.castCLM (h := hsz)).comp (CtxVec.getCLM (Γ := Γ) (s := s) idx)).hasFDerivAt (x :=
          xV)
      have hfun :
          (fun x : CtxVec Γ => castVec hsz (CtxVec.get (Γ := Γ) (s := s) idx x))
            =
          fun x : CtxVec Γ =>
            ((Graph.castCLM (h := hsz)).comp (CtxVec.getCLM (Γ := Γ) (s := s) idx)) x := by
        funext x
        simp [Graph.castCLM, ContinuousLinearMap.comp_apply, CtxVec.getCLM_apply]
      exact hlin.congr_of_eventuallyEq hfun.eventuallyEq

    have hheads :
        HasFDerivAt (headsCLM (h := h) (n := m * n)) (headsCLM (h := h) (n := m * n)) x0 := by
      simpa using ((headsCLM (h := h) (n := m * n)).hasFDerivAt (x := x0))
    have hunheads :
        HasFDerivAt (unheadsCLM (h := h) (n := m * n)) (unheadsCLM (h := h) (n := m * n)) (G r0) :=
          by
      simpa using ((unheadsCLM (h := h) (n := m * n)).hasFDerivAt (x := G r0))

    have hmid :
        HasFDerivAt (fun z : Vec (h * (m * n)) => unheadsCLM (h := h) (n := m * n) (G (headsCLM (h
          := h) (n := m * n) z)))
          (deriv0 x0) x0 := by
      -- chain `headsCLM` → `G` → `unheadsCLM`
      have hG' := hG.comp x0 hheads
      have hcomp := hunheads.comp x0 hG'
      simpa [deriv0, G, r0] using hcomp

    have hcastOut :
        HasFDerivAt (fun y : Vec (h * (m * n)) => castVec hsz.symm y)
          (Graph.castCLM (h := hsz.symm)) (forward0 x0) := by
      simpa [Graph.castCLM] using ((Graph.castCLM (h := hsz.symm)).hasFDerivAt (x := forward0 x0))

    have hforward0 :
        HasFDerivAt (fun x : CtxVec Γ => forward0 (castVec hsz (CtxVec.get (Γ := Γ) (s := s) idx
          x)))
          ((deriv0 x0).comp ((Graph.castCLM (h := hsz)).comp (CtxVec.getCLM (Γ := Γ) (s := s) idx)))
            xV := by
      exact hmid.comp xV hget

    have hcomp := hcastOut.comp xV hforward0
    -- rewrite to the node's `forwardVec`
    have hEq :
        (Node.forwardVec (Γ := Γ) (τ := s) (softmaxLast (Γ := Γ) (h := h) (m := m) (n := n) idx))
          =
        (fun x : CtxVec Γ =>
          castVec hsz.symm (forward0 (castVec hsz (CtxVec.get (Γ := Γ) (s := s) idx x)))) := by
      funext x
      simp [softmaxLast, forward0, Node.forwardVec_ofVec]
    exact hcomp.congr_of_eventuallyEq hEq.eventuallyEq
  · intro xV dxV
    ext ip
    -- First compute the node's JVP (the explicit headwise derivative formula).
    simp [softmaxLast, Node.jvpVec_ofVec, Graph.castCLM,
      ContinuousLinearMap.comp_apply, ContinuousLinearMap.proj_apply,
      CtxVec.getCLM_apply, headsCLM_apply, unheadsCLM_apply, heads, unheads]
    -- Then show the `D`-based JVP reduces to the same expression.
    have hD :
        (D xV) dxV ip =
          castVec hsz.symm
            (unheads (h := h) (n := m * n) (fun head =>
              (SoftmaxLastAxis.derivMN (m := m) (n := n)
                  (heads (h := h) (n := m * n) (castVec hsz
                    (CtxVec.get (Γ := Γ) (s := s) idx xV)) head))
                (heads (h := h) (n := m * n) (castVec hsz
                    (CtxVec.get (Γ := Γ) (s := s) idx dxV)) head))) ip := by
      simp [D, deriv0, Graph.castCLM, ContinuousLinearMap.comp_apply,
        ContinuousLinearMap.proj_apply, CtxVec.getCLM_apply, headsCLM_apply, unheadsCLM_apply,
          heads, unheads]
    exact hD.symm

end Batched

end TapeNodes

end
end Autograd
end Proofs
