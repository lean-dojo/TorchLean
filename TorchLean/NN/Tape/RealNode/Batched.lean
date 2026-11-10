/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.Tape.RealNode.Matrix

/-!
# Batched matrix multiplication

The tape node `matmul` multiplies `h` pairs of matrices, `(h×m×n) × (h×n×p) → (h×m×p)` over `ℝ`,
with its `RealNode.FDerivCorrect` certificate `matmulFDeriv`, for graphs that keep the head
dimension as a separate axis, such as multi-head attention. Its bilinear map `matmulBilin` splits
the flattened inputs into heads with `rowsCLM` and joins the headwise products with `unrowsCLM`.
We take the rowwise softmax of every head from the `Op.softmax` arm of `Tape.LawfulNode.ofArm`.
-/

@[expose] public section


open TorchLean.Shape (size size_cons)

-- In this file simp leaves out `size_cons`. It would rewrite the sizes inside the `Fin` and `Vec`
-- types before the head-splitting, `ofFn` and cast lemmas match. `size3_eq` names it.
attribute [-simp] TorchLean.Shape.size_cons

namespace TorchLean

open TorchLean.Tensor

noncomputable section

open scoped BigOperators

namespace Tape.RealNode

namespace Batched

/-- Flattened size of a rank-3 shape. -/
theorem size3_eq (h m n : Nat) :
    Shape.size (Shape.dim h (.dim m (.dim n .scalar))) = h * (m * n) := by
  simp [Shape.size_cons]

-- ---------------------------------------------------------------------------
-- Batched matmul (bilinear)
-- ---------------------------------------------------------------------------

/-- Bilinear map for batched matmul, packaged as `A →L (B →L A ⬝ B)` in flattened form. -/
def matmulBilin {h m n p : Nat} :
    Vec (h * Shape.size [m, n]) →L[ℝ] Vec (h * Shape.size [n, p]) →L[ℝ]
      Vec (h * Shape.size [m, p]) := by
  classical
  let B : Vec (Shape.size [m, n]) →L[ℝ] Vec (Shape.size [n, p]) →L[ℝ]
      Vec (Shape.size [m, p]) :=
    Matmul.bilin (m := m) (n := n) (p := p)
  let LA : Vec (h * Shape.size [m, n]) →L[ℝ] (Fin h → Vec (Shape.size [m, n])) :=
    rowsCLM (m := h) (n := Shape.size [m, n])
  let LB : Vec (h * Shape.size [n, p]) →L[ℝ] (Fin h → Vec (Shape.size [n, p])) :=
    rowsCLM (m := h) (n := Shape.size [n, p])
  let LC : (Fin h → Vec (Shape.size [m, p])) →L[ℝ] Vec (h * Shape.size [m, p]) :=
    unrowsCLM (m := h) (n := Shape.size [m, p])

  -- Headwise bilinear map `(Fin h → A) × (Fin h → B) → (Fin h → C)`.
  let BH : (Fin h → Vec (Shape.size [m, n])) →L[ℝ]
      (Fin h → Vec (Shape.size [n, p])) →L[ℝ]
        (Fin h → Vec (Shape.size [m, p])) := by
    classical
    let fLin : (Fin h → Vec (Shape.size [m, n])) →ₗ[ℝ]
        (Fin h → Vec (Shape.size [n, p])) →L[ℝ] (Fin h → Vec (Shape.size [m, p])) :=
      { toFun := fun aH =>
          ContinuousLinearMap.pi (R := ℝ) (fun head : Fin h =>
            ((B (aH head)).comp (ContinuousLinearMap.proj (R := ℝ) head)))
        map_add' := by
          intro a1 a2
          ext bH head ip
          simp [ContinuousLinearMap.pi_apply, ContinuousLinearMap.proj_apply,
            ContinuousLinearMap.comp_apply, B, Pi.add_apply]
        map_smul' := by
          intro r a1
          ext bH head ip
          simp [ContinuousLinearMap.pi_apply, ContinuousLinearMap.proj_apply,
            ContinuousLinearMap.comp_apply, B, Pi.smul_apply, smul_eq_mul] }
    refine ⟨fLin, ?_⟩
    exact LinearMap.continuous_of_finiteDimensional (f := fLin)

  -- Flatten the headwise map back to the row-major vector representation.
  let fLin : Vec (h * Shape.size [m, n]) →ₗ[ℝ]
      Vec (h * Shape.size [n, p]) →L[ℝ] Vec (h * Shape.size [m, p]) :=
    { toFun := fun aFlat =>
        LC.comp (((BH (LA aFlat)).comp LB))
      map_add' := by
        intro a1 a2
        ext bFlat ip
        -- Decode `ip` into a head index and an intra-head matrix index, then use linearity of `B`.
        let q : Fin h × Fin (Shape.size [m, p]) :=
          (finProdFinEquiv : Fin h × Fin (Shape.size [m, p]) ≃
            Fin (h * Shape.size [m, p])).symm
            ip
        have hheads :
            rows (m := h) (n := Shape.size [m, n]) (a1 + a2) q.1 =
              rows (m := h) (n := Shape.size [m, n]) a1 q.1 +
                rows (m := h) (n := Shape.size [m, n]) a2 q.1 := by
          ext j
          simp [rows]
        have hB' :
            B (rows (m := h) (n := Shape.size [m, n]) (a1 + a2) q.1)
              =
            B (rows (m := h) (n := Shape.size [m, n]) a1 q.1) +
              B (rows (m := h) (n := Shape.size [m, n]) a2 q.1) := by
          rw [hheads]
          exact
            B.map_add
              (rows (m := h) (n := Shape.size [m, n]) a1 q.1)
              (rows (m := h) (n := Shape.size [m, n]) a2 q.1)
        have hBapp :=
          congrArg (fun F => F (rows (m := h) (n := Shape.size [n, p]) bFlat q.1)) hB'
        have hBcoord := congrArg (fun v => v q.2) hBapp
        -- Expand the `unrows` projection at `ip` and finish.
        simp [rows, q] at hBcoord
        exact hBcoord
      map_smul' := by
        intro r a1
        ext bFlat ip
        let q : Fin h × Fin (Shape.size [m, p]) :=
          (finProdFinEquiv : Fin h × Fin (Shape.size [m, p]) ≃
            Fin (h * Shape.size [m, p])).symm
            ip
        have hheads :
            rows (m := h) (n := Shape.size [m, n]) (r • a1) q.1 =
              r • rows (m := h) (n := Shape.size [m, n]) a1 q.1 := by
          ext j
          simp [rows, smul_eq_mul]
        have hB' :
            B (rows (m := h) (n := Shape.size [m, n]) (r • a1) q.1)
              =
            r • B (rows (m := h) (n := Shape.size [m, n]) a1 q.1) := by
          rw [hheads]
          exact B.map_smulₛₗ r (rows (m := h) (n := Shape.size [m, n]) a1 q.1)
        have hBapp :=
          congrArg (fun F => F (rows (m := h) (n := Shape.size [n, p]) bFlat q.1)) hB'
        have hBcoord := congrArg (fun v => v q.2) hBapp
        simp [rows, q, smul_eq_mul] at hBcoord
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
    LawfulNode (α := ℝ) Unit Γ (.dim h (.dim m (.dim p .scalar))) :=
  let sA : Shape := .dim h (.dim m (.dim n .scalar))
  let sB : Shape := .dim h (.dim n (.dim p .scalar))
  let sOut : Shape := .dim h (.dim m (.dim p .scalar))
  let Bmul :
      Vec (Shape.size sA) →L[ℝ] Vec (Shape.size sB) →L[ℝ] Vec (Shape.size sOut) :=
    matmulBilin (h := h) (m := m) (n := n) (p := p)
  let fA : CtxVec Γ → Vec (Shape.size sA) :=
    fun x => CtxVec.get (Γ := Γ) (s := sA) A x
  let fB : CtxVec Γ → Vec (Shape.size sB) :=
    fun x => CtxVec.get (Γ := Γ) (s := sB) B x
  let deriv0 : CtxVec Γ → (CtxVec Γ →L[ℝ] Vec (Shape.size sOut)) :=
    fun x =>
      (Bmul.precompR (CtxVec Γ) (fA x) (CtxVec.getCLM (Γ := Γ) (s := sB) B)) +
        (Bmul.precompL (CtxVec Γ) (CtxVec.getCLM (Γ := Γ) (s := sA) A) (fB x))
  LawfulNode.ofFn (Γ := Γ) (τ := sOut)
    (f := fun x => (Bmul (fA x)) (fB x))
    (jvp := fun x dx => (deriv0 x) dx)
    (vjp := fun x δ => (deriv0 x).adjoint δ)
    (correct_inner := by
      intro x dx δ
      simpa using (ContinuousLinearMap.adjoint_inner_right (A := deriv0 x) (x := dx) (y := δ)).symm)

/-- `FDerivCorrect` for the batched matmul node. -/
def matmulFDeriv {Γ : List Shape} {h m n p : Nat}
    (A : Idx Γ (.dim h (.dim m (.dim n .scalar))))
    (B : Idx Γ (.dim h (.dim n (.dim p .scalar)))) :
    FDerivCorrect (matmul (Γ := Γ) (h := h) (m := m) (n := n) (p := p) A B).toNodeData () := by
  classical
  let sA : Shape := .dim h (.dim m (.dim n .scalar))
  let sB : Shape := .dim h (.dim n (.dim p .scalar))
  let sOut : Shape := .dim h (.dim m (.dim p .scalar))
  let Bmul :
      Vec (Shape.size sA) →L[ℝ] Vec (Shape.size sB) →L[ℝ] Vec (Shape.size sOut) :=
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
    have hA : HasFDerivAt fA (CtxVec.getCLM (Γ := Γ) (s := sA) A) xV :=
      CtxVec.hasFDerivAt_get (Γ := Γ) (s := sA) A xV
    have hB : HasFDerivAt fB (CtxVec.getCLM (Γ := Γ) (s := sB) B) xV :=
      CtxVec.hasFDerivAt_get (Γ := Γ) (s := sB) B xV
    have hbilin :=
      ContinuousLinearMap.hasFDerivAt_of_bilinear (B := Bmul) (hf := hA) (hg := hB)
    have hEq :
        (NodeData.forwardVec (Γ := Γ) (τ := sOut)
            (matmul (Γ := Γ) (h := h) (m := m) (n := n) (p := p) A B).toNodeData ())
          =
        (fun x : CtxVec Γ => (Bmul (fA x)) (fB x)) := by
      funext x
      simp [matmul, LawfulNode.forwardVec_ofFn, Bmul, fA, fB]
    exact hbilin.congr_of_eventuallyEq hEq.eventuallyEq
  · intro xV dxV
    ext ip
    -- Unfolding the node JVP into the bilinear JVP formula closes the goal.
    simp [matmul, LawfulNode.jvpVec_ofFn, Bmul, fA, fB]

end Batched

end Tape.RealNode

end
end TorchLean
