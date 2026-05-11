/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Proofs.Autograd.Tape.Nodes.Elementwise

/-!
# Arithmetic and stochastic tape nodes

Affine nodes, pointwise arithmetic, scaling, multiplication, squaring, and fixed-mask stochastic
operators. Dropout appears here only in its fixed-mask deterministic form, which is the semantics
that can be differentiated directly.
-/

@[expose] public section

namespace Proofs
namespace Autograd

open Spec
open Tensor

noncomputable section

open scoped BigOperators

namespace TapeNodes

/--
Fixed affine node over arbitrary tensor shapes.

This is the proof-level version of a linear layer after all leading dimensions have been flattened
into the vector representation used by `CtxVec`. It is intentionally shape-generic: a usual
unbatched `LinearSpec`, a position-wise Transformer FFN map over a whole sequence, or any other
fixed affine map can instantiate the same theorem by supplying the appropriate continuous linear
map `A` and bias vector `b`.
-/
def affine {Γ : List Shape} {sIn sOut : Shape}
    (idx : Idx Γ sIn) (A : Vec (Shape.size sIn) →L[ℝ] Vec (Shape.size sOut))
    (b : Vec (Shape.size sOut)) : Node Γ sOut :=
  Node.ofVec (Γ := Γ) (τ := sOut)
    (f := fun x => A (CtxVec.get (Γ := Γ) (s := sIn) idx x) + b)
    (jvp := fun _x dx => A (CtxVec.get (Γ := Γ) (s := sIn) idx dx))
    (vjp := fun _x δ => CtxVec.single (Γ := Γ) (s := sIn) idx (A.adjoint δ))
    (correct_inner := by
      intro _x dx δ
      rw [CtxVec.inner_get_single]
      simpa using
        (ContinuousLinearMap.adjoint_inner_right (A := A)
          (x := CtxVec.get (Γ := Γ) (s := sIn) idx dx) (y := δ)).symm)

/-- `NodeFDerivCorrect` for a fixed affine map over arbitrary tensor shapes. -/
def affineFderiv {Γ : List Shape} {sIn sOut : Shape}
    (idx : Idx Γ sIn) (A : Vec (Shape.size sIn) →L[ℝ] Vec (Shape.size sOut))
    (b : Vec (Shape.size sOut)) :
    NodeFDerivCorrect (affine (Γ := Γ) (sIn := sIn) (sOut := sOut) idx A b) :=
  { deriv := fun _x => A.comp (CtxVec.getCLM (Γ := Γ) (s := sIn) idx)
    hasFDerivAt := by
      intro xV
      have hlin :
          HasFDerivAt
            (fun x : CtxVec Γ => A (CtxVec.get (Γ := Γ) (s := sIn) idx x))
            (A.comp (CtxVec.getCLM (Γ := Γ) (s := sIn) idx)) xV := by
        have h :=
          ((A.comp (CtxVec.getCLM (Γ := Γ) (s := sIn) idx)).hasFDerivAt (x := xV))
        have hfun :
            (fun x : CtxVec Γ => A (CtxVec.get (Γ := Γ) (s := sIn) idx x)) =
              fun x => (A.comp (CtxVec.getCLM (Γ := Γ) (s := sIn) idx)) x := by
          funext x
          simp [CtxVec.getCLM_apply, ContinuousLinearMap.comp_apply]
        exact h.congr_of_eventuallyEq hfun.eventuallyEq
      simpa [affine, Node.forwardVec_ofVec, ContinuousLinearMap.comp_apply] using
        hlin.const_add b
    jvp_eq := by
      intro xV dxV
      simp [affine, Node.jvpVec_ofVec, ContinuousLinearMap.comp_apply, CtxVec.getCLM_apply] }

-- ---------------------------------------------------------------------------
-- Shape-generic binary ops on context entries
-- ---------------------------------------------------------------------------

/-- Add two same-shaped context entries. -/
def add {Γ : List Shape} {s : Shape} (a b : Idx Γ s) : Node Γ s :=
  Node.ofVec (Γ := Γ) (τ := s)
    (f := fun x => (CtxVec.get (Γ := Γ) (s := s) a x) + (CtxVec.get (Γ := Γ) (s := s) b x))
    (jvp := fun _x dx => (CtxVec.get (Γ := Γ) (s := s) a dx) + (CtxVec.get (Γ := Γ) (s := s) b dx))
    (vjp := fun _x δ => (CtxVec.single (Γ := Γ) (s := s) a δ) + (CtxVec.single (Γ := Γ) (s := s) b
      δ))
    (correct_inner := by
      intro _x dx δ
      classical
      simp [inner_add_left, inner_add_right, CtxVec.inner_get_single])

/-- `NodeFDerivCorrect` for `add` (derivative is pointwise addition of the two projections). -/
def addFderiv {Γ : List Shape} {s : Shape} (a b : Idx Γ s) :
    NodeFDerivCorrect (add (Γ := Γ) (s := s) a b) :=
{ deriv := fun _ => (CtxVec.getCLM (Γ := Γ) (s := s) a) + (CtxVec.getCLM (Γ := Γ) (s := s) b)
  hasFDerivAt := by
    intro xV
    -- the forward map is a continuous linear map
    have hfun :
        (fun x : CtxVec Γ =>
            CtxVec.get (Γ := Γ) (s := s) a x + CtxVec.get (Γ := Γ) (s := s) b x)
          =
        fun x : CtxVec Γ =>
            ((CtxVec.getCLM (Γ := Γ) (s := s) a) + (CtxVec.getCLM (Γ := Γ) (s := s) b)) x := by
      funext x
      simp [CtxVec.getCLM_apply]
    simpa [add, Node.forwardVec_ofVec, hfun] using
      (((CtxVec.getCLM (Γ := Γ) (s := s) a) + (CtxVec.getCLM (Γ := Γ) (s := s) b)).hasFDerivAt (x :=
        xV))
  jvp_eq := by
    intro xV dxV
    simp [add, Node.jvpVec_ofVec, ContinuousLinearMap.add_apply, CtxVec.getCLM_apply] }

/-- Subtract two same-shaped context entries. -/
def sub {Γ : List Shape} {s : Shape} (a b : Idx Γ s) : Node Γ s :=
  Node.ofVec (Γ := Γ) (τ := s)
    (f := fun x => (CtxVec.get (Γ := Γ) (s := s) a x) - (CtxVec.get (Γ := Γ) (s := s) b x))
    (jvp := fun _x dx => (CtxVec.get (Γ := Γ) (s := s) a dx) - (CtxVec.get (Γ := Γ) (s := s) b dx))
    (vjp := fun _x δ => (CtxVec.single (Γ := Γ) (s := s) a δ) - (CtxVec.single (Γ := Γ) (s := s) b
      δ))
    (correct_inner := by
      intro _x dx δ
      classical
      -- linearity + `CtxVec.inner_get_single`
      simp [sub_eq_add_neg, inner_add_left, inner_add_right, inner_neg_left, inner_neg_right,
        CtxVec.inner_get_single])

/-- `NodeFDerivCorrect` for `sub` (derivative is pointwise subtraction of projections). -/
def subFderiv {Γ : List Shape} {s : Shape} (a b : Idx Γ s) :
    NodeFDerivCorrect (sub (Γ := Γ) (s := s) a b) :=
{ deriv := fun _ => (CtxVec.getCLM (Γ := Γ) (s := s) a) - (CtxVec.getCLM (Γ := Γ) (s := s) b)
  hasFDerivAt := by
    intro xV
    have hfun :
        (fun x : CtxVec Γ =>
            CtxVec.get (Γ := Γ) (s := s) a x - CtxVec.get (Γ := Γ) (s := s) b x)
          =
        fun x : CtxVec Γ =>
            ((CtxVec.getCLM (Γ := Γ) (s := s) a) - (CtxVec.getCLM (Γ := Γ) (s := s) b)) x := by
      funext x
      simp [CtxVec.getCLM_apply]
    simpa [sub, Node.forwardVec_ofVec, hfun] using
      (((CtxVec.getCLM (Γ := Γ) (s := s) a) - (CtxVec.getCLM (Γ := Γ) (s := s) b)).hasFDerivAt (x :=
        xV))
  jvp_eq := by
    intro xV dxV
    simp [sub, Node.jvpVec_ofVec, ContinuousLinearMap.sub_apply, CtxVec.getCLM_apply] }

/-- Scale a context entry by a constant scalar. -/
def scale {Γ : List Shape} {s : Shape} (idx : Idx Γ s) (c : ℝ) : Node Γ s :=
  Node.ofVec (Γ := Γ) (τ := s)
    (f := fun x => c • (CtxVec.get (Γ := Γ) (s := s) idx x))
    (jvp := fun _x dx => c • (CtxVec.get (Γ := Γ) (s := s) idx dx))
    (vjp := fun _x δ => CtxVec.single (Γ := Γ) (s := s) idx (c • δ))
    (correct_inner := by
      intro _x dx δ
      classical
      -- `⟪c • dx, δ⟫ = ⟪dx, c • δ⟫` over `ℝ`
      simp [CtxVec.inner_get_single, inner_smul_left, inner_smul_right])

/-- `NodeFDerivCorrect` for `scale` (derivative is scaling of the projection CLM). -/
def scaleFderiv {Γ : List Shape} {s : Shape} (idx : Idx Γ s) (c : ℝ) :
    NodeFDerivCorrect (scale (Γ := Γ) (s := s) idx c) :=
{ deriv := fun _ => c • (CtxVec.getCLM (Γ := Γ) (s := s) idx)
  hasFDerivAt := by
    intro xV
    have hfun :
        (fun x : CtxVec Γ => c • CtxVec.get (Γ := Γ) (s := s) idx x)
          =
        fun x : CtxVec Γ => (c • (CtxVec.getCLM (Γ := Γ) (s := s) idx)) x := by
      funext x
      simp [CtxVec.getCLM_apply]
    simpa [scale, Node.forwardVec_ofVec, hfun] using
      ((c • (CtxVec.getCLM (Γ := Γ) (s := s) idx)).hasFDerivAt (x := xV))
  jvp_eq := by
    intro xV dxV
    simp [scale, Node.jvpVec_ofVec, CtxVec.getCLM_apply] }

/--
Apply a fixed elementwise multiplier.

This is the differentiable core of deterministic training-mode dropout: once a seed has sampled a
mask, the forward pass is just `x ↦ coeff ⊙ x` (with `coeff = mask / keepProb` for inverted
dropout). The randomness itself is not differentiated; it is represented by the fixed `coeff`.
-/
def fixedMaskScale {Γ : List Shape} {s : Shape} (idx : Idx Γ s) (coeff : Vec (Shape.size s)) :
    Node Γ s :=
  Node.ofVec (Γ := Γ) (τ := s)
    (f := fun x => vecOfFun (n := Shape.size s) fun i =>
      coeff i * CtxVec.get (Γ := Γ) (s := s) idx x i)
    (jvp := fun _x dx => vecOfFun (n := Shape.size s) fun i =>
      coeff i * CtxVec.get (Γ := Γ) (s := s) idx dx i)
    (vjp := fun _x δ => CtxVec.single (Γ := Γ) (s := s) idx
      (vecOfFun (n := Shape.size s) fun i => coeff i * δ i))
    (correct_inner := by
      intro _x dx δ
      classical
      rw [CtxVec.inner_get_single]
      simp [inner_eq_sum_mul, mul_assoc, mul_left_comm, mul_comm])

/--
`NodeFDerivCorrect` for a fixed-mask scaling node.

The derivative is the diagonal linear map induced by `coeff`. This is the proof-level contract for
seeded dropout after the mask has been generated: fixed seed and fixed keep probability determine
`coeff`; gradients flow only through the input activation.
-/
def fixedMaskScaleFderiv {Γ : List Shape} {s : Shape} (idx : Idx Γ s)
    (coeff : Vec (Shape.size s)) :
    NodeFDerivCorrect (fixedMaskScale (Γ := Γ) (s := s) idx coeff) :=
  let D : CtxVec Γ →L[ℝ] Vec (Shape.size s) := by
    classical
    let L : Vec (Shape.size s) →L[ℝ] Vec (Shape.size s) := by
      let fLin : Vec (Shape.size s) →ₗ[ℝ] Vec (Shape.size s) :=
        { toFun := fun v => vecOfFun (n := Shape.size s) fun i => coeff i * v i
          map_add' := by
            intro x y
            ext i
            simp [mul_add]
          map_smul' := by
            intro r x
            ext i
            simp [smul_eq_mul]
            ring }
      exact ⟨fLin, LinearMap.continuous_of_finiteDimensional (f := fLin)⟩
    exact L.comp (CtxVec.getCLM (Γ := Γ) (s := s) idx)
  { deriv := fun _ => D
    hasFDerivAt := by
      intro xV
      have hlin : HasFDerivAt (fun x : CtxVec Γ => D x) D xV :=
        D.hasFDerivAt
      have hfun :
          (fun x : CtxVec Γ =>
              Node.forwardVec (Γ := Γ) (τ := s)
                (fixedMaskScale (Γ := Γ) (s := s) idx coeff) x)
            =
          fun x : CtxVec Γ => D x := by
        funext x
        ext i
        simp [fixedMaskScale, Node.forwardVec_ofVec, D, CtxVec.getCLM_apply,
          ContinuousLinearMap.comp_apply]
      exact hlin.congr_of_eventuallyEq hfun.eventuallyEq
    jvp_eq := by
      intro xV dxV
      ext i
      simp [fixedMaskScale, Node.jvpVec_ofVec, D, CtxVec.getCLM_apply,
        ContinuousLinearMap.comp_apply] }

/-- Coefficients for inverted dropout from a fixed Boolean keep mask and scalar keep probability. -/
def invertedDropoutCoeff {s : Shape} (mask : Fin (Shape.size s) → Bool) (keepProb : ℝ) :
    Vec (Shape.size s) :=
  vecOfFun (n := Shape.size s) fun i => if mask i then keepProb⁻¹ else 0

/--
Fixed-mask inverted dropout node.

This is the theorem-facing form of training-mode dropout. A runtime seed may generate `mask`, but
the derivative theorem is stated after sampling: `mask` and `keepProb` are constants, and the node
is simply a fixed diagonal linear map on the activation.
-/
def fixedInvertedDropout {Γ : List Shape} {s : Shape} (idx : Idx Γ s)
    (mask : Fin (Shape.size s) → Bool) (keepProb : ℝ) : Node Γ s :=
  fixedMaskScale (Γ := Γ) (s := s) idx (invertedDropoutCoeff (s := s) mask keepProb)

/-- `NodeFDerivCorrect` for fixed-mask inverted dropout. -/
def fixedInvertedDropoutFderiv {Γ : List Shape} {s : Shape} (idx : Idx Γ s)
    (mask : Fin (Shape.size s) → Bool) (keepProb : ℝ) :
    NodeFDerivCorrect
      (fixedInvertedDropout (Γ := Γ) (s := s) idx mask keepProb) :=
  fixedMaskScaleFderiv (Γ := Γ) (s := s) idx
    (invertedDropoutCoeff (s := s) mask keepProb)

/-- Pointwise multiplication of two same-shaped context entries. -/
def mul {Γ : List Shape} {s : Shape} (a b : Idx Γ s) : Node Γ s :=
  let n : Nat := Shape.size s
  let hadamard : Vec n → Vec n → Vec n := fun u v => vecOfFun (n := n) fun i => u i * v i
  Node.ofVec (Γ := Γ) (τ := s)
    (f := fun x => hadamard (CtxVec.get (Γ := Γ) (s := s) a x) (CtxVec.get (Γ := Γ) (s := s) b x))
    (jvp := fun x dx =>
      hadamard (CtxVec.get (Γ := Γ) (s := s) a x) (CtxVec.get (Γ := Γ) (s := s) b dx) +
        hadamard (CtxVec.get (Γ := Γ) (s := s) b x) (CtxVec.get (Γ := Γ) (s := s) a dx))
    (vjp := fun x δ =>
      CtxVec.single (Γ := Γ) (s := s) a (hadamard δ (CtxVec.get (Γ := Γ) (s := s) b x)) +
        CtxVec.single (Γ := Γ) (s := s) b (hadamard δ (CtxVec.get (Γ := Γ) (s := s) a x)))
    (correct_inner := by
      intro x dx δ
      classical
      have hA :
          inner ℝ (hadamard (CtxVec.get (Γ := Γ) (s := s) a x) (CtxVec.get (Γ := Γ) (s := s) b dx))
            δ
            =
          inner ℝ (CtxVec.get (Γ := Γ) (s := s) b dx) (hadamard δ (CtxVec.get (Γ := Γ) (s := s) a
            x)) := by
        simp [hadamard, inner_eq_sum_mul, mul_assoc, mul_comm]
        rfl
      have hB :
          inner ℝ (hadamard (CtxVec.get (Γ := Γ) (s := s) b x) (CtxVec.get (Γ := Γ) (s := s) a dx))
            δ
            =
          inner ℝ (CtxVec.get (Γ := Γ) (s := s) a dx) (hadamard δ (CtxVec.get (Γ := Γ) (s := s) b
            x)) := by
        simp [hadamard, inner_eq_sum_mul, mul_left_comm, mul_comm]
        rfl
      calc
        inner ℝ
            (hadamard (CtxVec.get (Γ := Γ) (s := s) a x) (CtxVec.get (Γ := Γ) (s := s) b dx) +
              hadamard (CtxVec.get (Γ := Γ) (s := s) b x) (CtxVec.get (Γ := Γ) (s := s) a dx)) δ
            =
            inner ℝ (hadamard (CtxVec.get (Γ := Γ) (s := s) a x) (CtxVec.get (Γ := Γ) (s := s) b
              dx)) δ +
              inner ℝ (hadamard (CtxVec.get (Γ := Γ) (s := s) b x) (CtxVec.get (Γ := Γ) (s := s) a
                dx)) δ := by
              simp [inner_add_left]
        _ =
            inner ℝ (CtxVec.get (Γ := Γ) (s := s) b dx) (hadamard δ (CtxVec.get (Γ := Γ) (s := s) a
              x)) +
              inner ℝ (CtxVec.get (Γ := Γ) (s := s) a dx) (hadamard δ (CtxVec.get (Γ := Γ) (s := s)
                b x)) := by
              simp [hA, hB]
        _ =
            inner ℝ dx
                (CtxVec.single (Γ := Γ) (s := s) a (hadamard δ (CtxVec.get (Γ := Γ) (s := s) b x)))
                  +
              inner ℝ dx
                (CtxVec.single (Γ := Γ) (s := s) b (hadamard δ (CtxVec.get (Γ := Γ) (s := s) a x)))
                  := by
              simp [CtxVec.inner_get_single, add_comm]
        _ =
            inner ℝ dx
              (CtxVec.single (Γ := Γ) (s := s) a (hadamard δ (CtxVec.get (Γ := Γ) (s := s) b x)) +
                CtxVec.single (Γ := Γ) (s := s) b (hadamard δ (CtxVec.get (Γ := Γ) (s := s) a x)))
                  := by
              simp [inner_add_right])

/-- `NodeFDerivCorrect` for `mul` (Hadamard product), using the product rule coordinatewise. -/
def mulFderiv {Γ : List Shape} {s : Shape} (a b : Idx Γ s) :
    NodeFDerivCorrect (mul (Γ := Γ) (s := s) a b) :=
{ deriv := fun xV =>
    let n : Nat := Shape.size s
    (euclideanEquiv n).symm.toContinuousLinearMap.comp <|
      ContinuousLinearMap.pi (fun i : Fin n =>
        let a_i : CtxVec Γ →L[ℝ] ℝ :=
          (evalCLM (n := n) i).comp (CtxVec.getCLM (Γ := Γ) (s := s) a)
        let b_i : CtxVec Γ →L[ℝ] ℝ :=
          (evalCLM (n := n) i).comp (CtxVec.getCLM (Γ := Γ) (s := s) b)
        (CtxVec.get (Γ := Γ) (s := s) a xV i) • b_i + (CtxVec.get (Γ := Γ) (s := s) b xV i) • a_i)
  hasFDerivAt := by
    intro xV
    classical
    let n : Nat := Shape.size s
    let aFun : CtxVec Γ → Vec n := fun x => CtxVec.get (Γ := Γ) (s := s) a x
    let bFun : CtxVec Γ → Vec n := fun x => CtxVec.get (Γ := Γ) (s := s) b x
    -- prove coordinatewise and assemble with `hasFDerivAt_pi`
    have hcoord :
        ∀ i : Fin n,
          HasFDerivAt (fun x : CtxVec Γ => (aFun x i) * (bFun x i))
            ((aFun xV i) • ((evalCLM (n := n) i).comp (CtxVec.getCLM (Γ := Γ) (s := s) b)) +
              (bFun xV i) • ((evalCLM (n := n) i).comp (CtxVec.getCLM (Γ := Γ) (s := s) a))) xV :=
                by
      intro i
      let aCLM : CtxVec Γ →L[ℝ] ℝ :=
        (evalCLM (n := n) i).comp (CtxVec.getCLM (Γ := Γ) (s := s) a)
      let bCLM : CtxVec Γ →L[ℝ] ℝ :=
        (evalCLM (n := n) i).comp (CtxVec.getCLM (Γ := Γ) (s := s) b)
      -- Treat `aCLM`/`bCLM` as the derivatives of the coordinate projections `x ↦ get a x i`, `x ↦
      -- get b x i`.
      have ha0 : HasFDerivAt (fun x : CtxVec Γ => aCLM x) aCLM xV :=
        aCLM.hasFDerivAt (x := xV)
      have hb0 : HasFDerivAt (fun x : CtxVec Γ => bCLM x) bCLM xV :=
        bCLM.hasFDerivAt (x := xV)
      have ha : HasFDerivAt (fun x : CtxVec Γ => CtxVec.get (Γ := Γ) (s := s) a x i) aCLM xV := by
        have hEq :
            (fun x : CtxVec Γ => CtxVec.get (Γ := Γ) (s := s) a x i) =
              (fun x : CtxVec Γ => aCLM x) := by
          funext x
          -- Unfold `aCLM` to a coordinate projection of `CtxVec.getCLM a`.
          simp [aCLM, ContinuousLinearMap.comp_apply, evalCLM_apply]
          -- Relate `getCLM` to `get` under the coordinate projection.
          exact
            (congrArg (fun v : Vec n => v.ofLp i) (CtxVec.getCLM_apply (Γ := Γ) (s := s) a x)).symm
        exact ha0.congr_of_eventuallyEq hEq.eventuallyEq
      have hb : HasFDerivAt (fun x : CtxVec Γ => CtxVec.get (Γ := Γ) (s := s) b x i) bCLM xV := by
        have hEq :
            (fun x : CtxVec Γ => CtxVec.get (Γ := Γ) (s := s) b x i) =
              (fun x : CtxVec Γ => bCLM x) := by
          funext x
          simp [bCLM, ContinuousLinearMap.comp_apply, evalCLM_apply]
          exact
            (congrArg (fun v : Vec n => v.ofLp i) (CtxVec.getCLM_apply (Γ := Γ) (s := s) b x)).symm
        exact hb0.congr_of_eventuallyEq hEq.eventuallyEq
      -- Product rule in `ℝ`.
      have hmul :=
        (ha.mul hb)
      simpa [aCLM, bCLM, aFun, bFun] using hmul
    have hpi :
        HasFDerivAt
          (fun x : CtxVec Γ => fun i : Fin n => (aFun x i) * (bFun x i))
          (ContinuousLinearMap.pi (fun i : Fin n =>
            (aFun xV i) • ((evalCLM (n := n) i).comp (CtxVec.getCLM (Γ := Γ) (s := s) b)) +
              (bFun xV i) • ((evalCLM (n := n) i).comp (CtxVec.getCLM (Γ := Γ) (s := s) a))))
          xV := by
      -- `hasFDerivAt_pi` packages the coordinate statements
      refine (hasFDerivAt_pi (𝕜 := ℝ)
        (φ := fun i : Fin n => fun x : CtxVec Γ => (aFun x i) * (bFun x i))
        (φ' := fun i : Fin n =>
          (aFun xV i) • ((evalCLM (n := n) i).comp (CtxVec.getCLM (Γ := Γ) (s := s) b)) +
            (bFun xV i) • ((evalCLM (n := n) i).comp (CtxVec.getCLM (Γ := Γ) (s := s) a)))
        (x := xV)).2 ?_
      intro i
      simpa using hcoord i
    -- transport `hpi` (a `Fin n → ℝ` statement) through the `Vec n ≃ₗ[ℝ] Fin n → ℝ` equivalence
    have he' :
        HasFDerivAt (fun g : Fin n → ℝ => (euclideanEquiv n).symm g) ((euclideanEquiv
          n).symm.toContinuousLinearMap)
          (fun i : Fin n => (aFun xV i) * (bFun xV i)) :=
      (ContinuousLinearMap.hasFDerivAt (euclideanEquiv n).symm.toContinuousLinearMap)
    have hcomp := he'.comp xV hpi
    have hEq :
        (Node.forwardVec (Γ := Γ) (τ := s) (mul (Γ := Γ) (s := s) a b))
          =
        (fun x : CtxVec Γ => vecOfFun (n := n) fun i : Fin n => (aFun x i) * (bFun x i)) := by
      funext x
      ext i
      simp [mul, vecOfFun, aFun, bFun, Node.forwardVec_ofVec]
    exact hcomp.congr_of_eventuallyEq hEq.eventuallyEq
  jvp_eq := by
    intro xV dxV
    classical
    ext i
    -- Expand the JVP, then evaluate the `pi`-assembled derivative in coordinate `i`.
    simp [mul, Node.jvpVec_ofVec, vecOfFun, ContinuousLinearMap.comp_apply,
      ContinuousLinearMap.add_apply, ContinuousLinearMap.smul_apply, ContinuousLinearMap.comp_apply,
      evalCLM_apply, CtxVec.getCLM_apply]
}

/-- Runtime `square` node, implemented as `x ⊙ x`. -/
def square {Γ : List Shape} {s : Shape} (idx : Idx Γ s) : Node Γ s :=
  mul (Γ := Γ) (s := s) idx idx

/-- Global `NodeFDerivCorrect` for elementwise square. -/
def squareFderiv {Γ : List Shape} {s : Shape} (idx : Idx Γ s) :
    NodeFDerivCorrect (square (Γ := Γ) (s := s) idx) :=
  mulFderiv (Γ := Γ) (s := s) idx idx

end TapeNodes

end

end Autograd
end Proofs
