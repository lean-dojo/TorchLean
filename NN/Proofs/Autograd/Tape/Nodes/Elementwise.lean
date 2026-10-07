/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Proofs.Autograd.Tape.Nodes.Context
public import Mathlib.Analysis.Calculus.Deriv.Abs
public import NN.Proofs.Autograd.FDeriv.Elementwise

/-!
# Elementwise tape nodes

Reusable `NodeFDerivCorrect` wrappers for scalar functions lifted pointwise to tensors, including
common activations such as ReLU, sigmoid, tanh, SiLU, GELU, ELU, and safe differentiable variants.
-/

@[expose] public section

namespace Proofs
namespace Autograd

open Spec TorchLean
open TorchLean TorchLean.Tensor

noncomputable section

open scoped BigOperators

namespace TapeNodes

/-- `CtxVec.get` specialized to vector shapes. -/
def getVec {Γ : List Shape} {n : Nat} (idx : Idx Γ (.dim n .scalar)) (x : CtxVec Γ) : Vec n :=
  castVec (by simp [Spec.Shape.size] : Spec.Shape.size (.dim n .scalar) = n)
    (CtxVec.get (Γ := Γ) (s := .dim n .scalar) idx x)

/-- `CtxVec.getCLM` specialized to vector shapes `.dim n .scalar`. -/
def getVecCLM {Γ : List Shape} {n : Nat} (idx : Idx Γ (.dim n .scalar)) : CtxVec Γ →L[ℝ] Vec n :=
  (Graph.castCLM (h := (by simp [Spec.Shape.size] : Spec.Shape.size (.dim n .scalar) = n))).comp
    (CtxVec.getCLM (Γ := Γ) (s := .dim n .scalar) idx)

/-- The vector-shaped context lookup agrees with `getVec`, size cast included. -/
@[simp] theorem getVecCLM_apply {Γ : List Shape} {n : Nat} (idx : Idx Γ (.dim n .scalar))
    (x : CtxVec Γ) :
    getVecCLM (Γ := Γ) (n := n) idx x = getVec (Γ := Γ) (n := n) idx x := by
  simp [getVecCLM, getVec, CtxVec.getCLM_apply, Graph.castCLM]

/-- Inject a `Vec n` into a vectorized context at `idx` (fills other blocks with zeros). -/
def singleVec {Γ : List Shape} {n : Nat} (idx : Idx Γ (.dim n .scalar)) (v : Vec n) : CtxVec Γ :=
  CtxVec.single (Γ := Γ) (s := .dim n .scalar) idx
    (castVec (by simp [Spec.Shape.size] : Spec.Shape.size (.dim n .scalar) = n).symm v)

/-- Reading a context slot is adjoint to writing that slot: `⟪x, single idx v⟫ = ⟪get idx x, v⟫`.

This is the adjointness fact behind every elementwise node's VJP. A node reads one slot and writes
one slot, so its reverse pass is the transpose of its forward projection, and that is exactly what
this equation says. -/
@[simp] theorem inner_getVec_singleVec {Γ : List Shape} {n : Nat} (idx : Idx Γ (.dim n .scalar))
    (x : CtxVec Γ) (v : Vec n) :
    inner ℝ x (singleVec (Γ := Γ) (n := n) idx v) = inner ℝ (getVec (Γ := Γ) (n := n) idx x) v := by
  rw [singleVec, CtxVec.inner_get_single, ← inner_castVec_left]
  rfl

-- ---------------------------------------------------------------------------
-- Generic elementwise nodes on flattened tensors
-- ---------------------------------------------------------------------------

/-- Elementwise node: apply a scalar function pointwise on a context entry. -/
def elemwise {Γ : List Shape} {s : Shape} (idx : Idx Γ s) (f f' : ℝ → ℝ) : Node Γ s :=
  let n : Nat := Spec.Shape.size s
  Node.ofFn (Γ := Γ) (τ := s)
    (f := fun xV =>
      vecOfFun (n := n) (fun i : Fin n => f (CtxVec.get (Γ := Γ) (s := s) idx xV i)))
    (jvp := fun xV dxV =>
      vecOfFun (n := n) (fun i : Fin n =>
        (CtxVec.get (Γ := Γ) (s := s) idx dxV i) * f' (CtxVec.get (Γ := Γ) (s := s) idx xV i)))
    (vjp := fun xV δ =>
      CtxVec.single (Γ := Γ) (s := s) idx
        (vecOfFun (n := n) (fun i : Fin n => δ i * f' (CtxVec.get (Γ := Γ) (s := s) idx xV i))))
    (correct_inner := by
      intro xV dxV δ
      classical
      -- First use the context adjointness, then expand both inners.
      simp [CtxVec.inner_get_single, vecOfFun]
      simp [inner_eq_sum_mul, mul_assoc, mul_comm]
      rfl)

/-- Pointwise analytic correctness for `elemwise` nodes from a coordinatewise `HasDerivAt`
  hypothesis. -/
def elemwiseFderivAt {Γ : List Shape} {s : Shape} (idx : Idx Γ s) (f f' : ℝ → ℝ) (xV : CtxVec Γ)
    (hf : ∀ i : Fin (Spec.Shape.size s),
      HasDerivAt f (f' (CtxVec.get (Γ := Γ) (s := s) idx xV i)) (CtxVec.get (Γ := Γ) (s := s) idx xV
        i)) :
    NodeFDerivCorrectAt (elemwise (Γ := Γ) (s := s) idx f f') xV := by
  classical
  let n : Nat := Spec.Shape.size s
  refine
    { deriv :=
        (elemwiseDerivCLM (n := n) f' (CtxVec.get (Γ := Γ) (s := s) idx xV)).comp
          (CtxVec.getCLM (Γ := Γ) (s := s) idx)
      hasFDerivAt := ?_
      jvp_eq := ?_ }
  · have hget :
        HasFDerivAt (fun x : CtxVec Γ => CtxVec.get (Γ := Γ) (s := s) idx x)
          (CtxVec.getCLM (Γ := Γ) (s := s) idx) xV :=
      CtxVec.hasFDerivAt_get idx xV
    have helem :
        HasFDerivAt (elemwiseVec (n := n) f) (elemwiseDerivCLM (n := n) f' (CtxVec.get (Γ := Γ) (s
          := s) idx xV))
          (CtxVec.get (Γ := Γ) (s := s) idx xV) :=
      hasFDerivAt_elemwiseVec_at (n := n) (x := CtxVec.get (Γ := Γ) (s := s) idx xV) (f := f) (f' :=
        f') hf
    have hcomp := helem.comp xV hget
    have hforward :
        (elemwiseVec (n := n) f ∘ fun x : CtxVec Γ => CtxVec.get (Γ := Γ) (s := s) idx x)
          =
        (fun xV : CtxVec Γ =>
          vecOfFun fun i => f ((CtxVec.get (Γ := Γ) (s := s) idx xV).ofLp i)) := by
      funext x
      ext i
      simp [elemwiseVec, vecOfFun]
    rw [hforward] at hcomp
    simpa [elemwise, Node.forwardVec_ofFn, elemwiseVec, n, ContinuousLinearMap.comp_apply] using
      hcomp
  · intro dxV
    ext i
    simp [elemwise, Node.jvpVec_ofFn, elemwiseDerivCLM, ContinuousLinearMap.comp_apply, n,
      CtxVec.getCLM_apply, vecOfFun]

/-- Analytic correctness for `elemwise` nodes from a scalar `HasDerivAt` hypothesis valid
everywhere: the pointwise certificate `elemwiseFderivAt`, read at every basepoint. -/
def elemwiseFderiv {Γ : List Shape} {s : Shape} (idx : Idx Γ s) (f f' : ℝ → ℝ)
    (hf : ∀ z, HasDerivAt f (f' z) z) :
    NodeFDerivCorrect (elemwise (Γ := Γ) (s := s) idx f f') where
  deriv xV :=
    (elemwiseDerivCLM (n := Spec.Shape.size s) f' (CtxVec.get (Γ := Γ) (s := s) idx xV)).comp
      (CtxVec.getCLM (Γ := Γ) (s := s) idx)
  hasFDerivAt xV := (elemwiseFderivAt (Γ := Γ) (s := s) idx f f' xV fun _ => hf _).hasFDerivAt
  jvp_eq xV := (elemwiseFderivAt (Γ := Γ) (s := s) idx f f' xV fun _ => hf _).jvp_eq

/-- Runtime `relu` node (elementwise; nondifferentiable at zero). -/
def relu {Γ : List Shape} {s : Shape} (idx : Idx Γ s) : Node Γ s :=
  elemwise (Γ := Γ) (s := s) idx Activation.Math.reluSpec Activation.Math.reluDerivSpec

/-- Pointwise `NodeFDerivCorrectAt` for `relu` under the assumption that inputs are nonzero. -/
def reluFderivAt {Γ : List Shape} {s : Shape} (idx : Idx Γ s) (xV : CtxVec Γ)
    (hx : ∀ i : Fin (Spec.Shape.size s), CtxVec.get (Γ := Γ) (s := s) idx xV i ≠ 0) :
    NodeFDerivCorrectAt (relu (Γ := Γ) (s := s) idx) xV :=
  elemwiseFderivAt (Γ := Γ) (s := s) idx
    Activation.Math.reluSpec Activation.Math.reluDerivSpec xV
    (fun i => Proofs.relu_deriv_correct (x := CtxVec.get (Γ := Γ) (s := s) idx xV i) (h := hx i))

/-- Runtime `abs` node (elementwise; nondifferentiable at zero). -/
def abs {Γ : List Shape} {s : Shape} (idx : Idx Γ s) : Node Γ s :=
  elemwise (Γ := Γ) (s := s) idx (fun x : ℝ => |x|) (fun x => (SignType.sign x : ℝ))

/-- Pointwise `NodeFDerivCorrectAt` for `abs` under the assumption that inputs are nonzero. -/
def absFderivAt {Γ : List Shape} {s : Shape} (idx : Idx Γ s) (xV : CtxVec Γ)
    (hx : ∀ i : Fin (Spec.Shape.size s), CtxVec.get (Γ := Γ) (s := s) idx xV i ≠ 0) :
    NodeFDerivCorrectAt (abs (Γ := Γ) (s := s) idx) xV :=
  elemwiseFderivAt (Γ := Γ) (s := s) idx (fun x : ℝ => |x|) (fun x => (SignType.sign x : ℝ)) xV
    (fun i => by simpa using (hasDerivAt_abs (hx i)))

/-- Runtime `log` node (elementwise; differentiable only away from zero). -/
def log {Γ : List Shape} {s : Shape} (idx : Idx Γ s) : Node Γ s :=
  elemwise (Γ := Γ) (s := s) idx Real.log (fun x => x⁻¹)

/-- Pointwise `NodeFDerivCorrectAt` for `log` under the assumption that inputs are nonzero. -/
def logFderivAt {Γ : List Shape} {s : Shape} (idx : Idx Γ s) (xV : CtxVec Γ)
    (hx : ∀ i : Fin (Spec.Shape.size s), CtxVec.get (Γ := Γ) (s := s) idx xV i ≠ 0) :
    NodeFDerivCorrectAt (log (Γ := Γ) (s := s) idx) xV :=
  elemwiseFderivAt (Γ := Γ) (s := s) idx Real.log (fun x => x⁻¹) xV
    (fun i => Real.hasDerivAt_log (hx i))

/-- Elementwise inverse node (differentiable only away from zero). -/
def inv {Γ : List Shape} {s : Shape} (idx : Idx Γ s) : Node Γ s :=
  elemwise (Γ := Γ) (s := s) idx (fun x => x⁻¹) (fun x => -((x ^ 2)⁻¹))

/-- Pointwise `NodeFDerivCorrectAt` for `inv` under the assumption that inputs are nonzero. -/
def invFderivAt {Γ : List Shape} {s : Shape} (idx : Idx Γ s) (xV : CtxVec Γ)
    (hx : ∀ i : Fin (Spec.Shape.size s), CtxVec.get (Γ := Γ) (s := s) idx xV i ≠ 0) :
    NodeFDerivCorrectAt (inv (Γ := Γ) (s := s) idx) xV :=
  elemwiseFderivAt (Γ := Γ) (s := s) idx (fun x => x⁻¹) (fun x => -((x ^ 2)⁻¹)) xV
    (fun i => by simpa using (hasDerivAt_inv (hx i)))

/-- Derivative of the scalar function `y ↦ sqrt (max y 0)` at positive points. -/
theorem hasDerivAt_sqrt_clamp_of_pos {x : ℝ} (hx : 0 < x) :
    HasDerivAt (fun y : ℝ => Real.sqrt (max y 0)) (1 / (2 * Real.sqrt x)) x := by
  -- Near a positive `x` the clamp is inactive, so the function agrees with `Real.sqrt`.
  have heq :
      (fun y : ℝ => Real.sqrt (max y 0)) =ᶠ[nhds x] fun y : ℝ => Real.sqrt y := by
    filter_upwards [Ioi_mem_nhds hx] with y (hy : 0 < y)
    simp [max_eq_left hy.le]
  have hs : HasDerivAt (fun y : ℝ => Real.sqrt y) (1 / (2 * Real.sqrt x)) x := by
    -- `Real.hasDerivAt_sqrt` expects `x ≠ 0`.
    simpa using (Real.hasDerivAt_sqrt (ne_of_gt hx))
  -- transport across the local equality `max y 0 = y` near positive `x`
  exact hs.congr_of_eventuallyEq heq

/-- Elementwise "clamped sqrt": `sqrt (max x 0)` (differentiable on `x > 0`). -/
def sqrtClamp {Γ : List Shape} {s : Shape} (idx : Idx Γ s) : Node Γ s :=
  elemwise (Γ := Γ) (s := s) idx (fun x => Real.sqrt (max x 0)) (fun x => 1 / (2 * Real.sqrt x))

/-- Pointwise `NodeFDerivCorrectAt` for `sqrtClamp` under the assumption that inputs are strictly
  positive. -/
def sqrtClampFderivAt {Γ : List Shape} {s : Shape} (idx : Idx Γ s) (xV : CtxVec Γ)
    (hx : ∀ i : Fin (Spec.Shape.size s), 0 < CtxVec.get (Γ := Γ) (s := s) idx xV i) :
    NodeFDerivCorrectAt (sqrtClamp (Γ := Γ) (s := s) idx) xV :=
  elemwiseFderivAt (Γ := Γ) (s := s) idx
    (fun x => Real.sqrt (max x 0)) (fun x => 1 / (2 * Real.sqrt x)) xV
    (fun i => hasDerivAt_sqrt_clamp_of_pos (hx i))

/-- Runtime `sqrt` node (elementwise; nondifferentiable at zero). -/
def sqrt {Γ : List Shape} {s : Shape} (idx : Idx Γ s) : Node Γ s :=
  elemwise (Γ := Γ) (s := s) idx Real.sqrt (fun x => 1 / (2 * Real.sqrt x))

/-- Pointwise `NodeFDerivCorrectAt` for `sqrt` under the assumption that inputs are nonzero. -/
def sqrtFderivAt {Γ : List Shape} {s : Shape} (idx : Idx Γ s) (xV : CtxVec Γ)
    (hx : ∀ i : Fin (Spec.Shape.size s), CtxVec.get (Γ := Γ) (s := s) idx xV i ≠ 0) :
    NodeFDerivCorrectAt (sqrt (Γ := Γ) (s := s) idx) xV :=
  elemwiseFderivAt (Γ := Γ) (s := s) idx Real.sqrt (fun x => 1 / (2 * Real.sqrt x)) xV
    (fun i => by simpa using (Real.hasDerivAt_sqrt (hx i)))

/-- Runtime scalar logistic node, applied elementwise.

Softmax over a whole axis is a separate family of nodes in `Softmax.lean`; this node is the
one-dimensional logistic map used by scalar activations. -/
def logistic {Γ : List Shape} {s : Shape} (idx : Idx Γ s) : Node Γ s :=
  elemwise (Γ := Γ) (s := s) idx Activation.Math.logisticSpec Activation.Math.logisticDerivSpec

/-- Global `NodeFDerivCorrect` for `logistic` (uses the scalar derivative lemma). -/
def logisticFderiv {Γ : List Shape} {s : Shape} (idx : Idx Γ s) :
    NodeFDerivCorrect (logistic (Γ := Γ) (s := s) idx) :=
  elemwiseFderiv (Γ := Γ) (s := s) idx
    Activation.Math.logisticSpec Activation.Math.logisticDerivSpec
    (fun z => Proofs.logistic_deriv_correct (x := z))

/-- Runtime `sigmoid` node (elementwise). -/
def sigmoid {Γ : List Shape} {s : Shape} (idx : Idx Γ s) : Node Γ s :=
  elemwise (Γ := Γ) (s := s) idx Activation.Math.sigmoidSpec Activation.Math.sigmoidDerivSpec

/-- Global `NodeFDerivCorrect` for `sigmoid`. -/
def sigmoidFderiv {Γ : List Shape} {s : Shape} (idx : Idx Γ s) :
    NodeFDerivCorrect (sigmoid (Γ := Γ) (s := s) idx) :=
  elemwiseFderiv (Γ := Γ) (s := s) idx
    Activation.Math.sigmoidSpec Activation.Math.sigmoidDerivSpec
    (fun z => Proofs.sigmoid_deriv_correct (x := z))

/-- Runtime `tanh` node (elementwise). -/
def tanh {Γ : List Shape} {s : Shape} (idx : Idx Γ s) : Node Γ s :=
  elemwise (Γ := Γ) (s := s) idx Activation.Math.tanhSpec Activation.Math.tanhDerivSpec

/-- Global `NodeFDerivCorrect` for `tanh`. -/
def tanhFderiv {Γ : List Shape} {s : Shape} (idx : Idx Γ s) :
    NodeFDerivCorrect (tanh (Γ := Γ) (s := s) idx) :=
  elemwiseFderiv (Γ := Γ) (s := s) idx
    Activation.Math.tanhSpec Activation.Math.tanhDerivSpec
    (fun z => Proofs.tanh_deriv_correct (x := z))

/-- Runtime `softplus` node (elementwise, smooth ReLU surrogate). -/
def softplus {Γ : List Shape} {s : Shape} (idx : Idx Γ s) : Node Γ s :=
  elemwise (Γ := Γ) (s := s) idx Activation.Math.softplusSpec Activation.Math.softplusDerivSpec

/-- Global `NodeFDerivCorrect` for `softplus`. -/
def softplusFderiv {Γ : List Shape} {s : Shape} (idx : Idx Γ s) :
    NodeFDerivCorrect (softplus (Γ := Γ) (s := s) idx) :=
  elemwiseFderiv (Γ := Γ) (s := s) idx
    Activation.Math.softplusSpec Activation.Math.softplusDerivSpec
    (fun z => Proofs.softplus_deriv_correct (x := z))

/-- Runtime `silu` node (elementwise). -/
def silu {Γ : List Shape} {s : Shape} (idx : Idx Γ s) : Node Γ s :=
  elemwise (Γ := Γ) (s := s) idx Activation.Math.swishSpec Activation.Math.swishDerivSpec

/-- Global `NodeFDerivCorrect` for SiLU. -/
def siluFderiv {Γ : List Shape} {s : Shape} (idx : Idx Γ s) :
    NodeFDerivCorrect (silu (Γ := Γ) (s := s) idx) :=
  elemwiseFderiv (Γ := Γ) (s := s) idx
    Activation.Math.swishSpec Activation.Math.swishDerivSpec
    (fun z => Proofs.silu_deriv_correct (x := z))

/-- Runtime tanh-approximate `gelu` node (elementwise). -/
def gelu {Γ : List Shape} {s : Shape} (idx : Idx Γ s) : Node Γ s :=
  elemwise (Γ := Γ) (s := s) idx Activation.Math.geluSpec Activation.Math.geluDerivSpec

/-- Global `NodeFDerivCorrect` for tanh-approximate GELU. -/
def geluFderiv {Γ : List Shape} {s : Shape} (idx : Idx Γ s) :
    NodeFDerivCorrect (gelu (Γ := Γ) (s := s) idx) :=
  elemwiseFderiv (Γ := Γ) (s := s) idx
    Activation.Math.geluSpec Activation.Math.geluDerivSpec
    (fun z => Proofs.gelu_deriv_correct (x := z))

/-- Runtime `safeLog` node (elementwise, always-defined log surrogate). -/
def safeLog {Γ : List Shape} {s : Shape} (idx : Idx Γ s) (ε : ℝ) : Node Γ s :=
  elemwise (Γ := Γ) (s := s) idx
    (fun x => Activation.Math.safeLogSpec (α := ℝ) x ε)
    (fun x => Activation.Math.safeLogDerivSpec (α := ℝ) x ε)

/-- Global `NodeFDerivCorrect` for `safeLog` (requires `0 < ε`). -/
def safeLogFderiv {Γ : List Shape} {s : Shape} (idx : Idx Γ s) (ε : ℝ) (hε : 0 < ε) :
    NodeFDerivCorrect (safeLog (Γ := Γ) (s := s) idx ε) :=
  elemwiseFderiv (Γ := Γ) (s := s) idx
    (fun x => Activation.Math.safeLogSpec (α := ℝ) x ε)
    (fun x => Activation.Math.safeLogDerivSpec (α := ℝ) x ε)
    (fun z => Proofs.safe_log_deriv_correct (x := z) (ε := ε) hε)

/-- Runtime `smoothAbs` node (elementwise, smooth abs surrogate). -/
def smoothAbs {Γ : List Shape} {s : Shape} (idx : Idx Γ s) (ε : ℝ) : Node Γ s :=
  elemwise (Γ := Γ) (s := s) idx
    (fun x => Activation.Math.smoothAbsSpec (α := ℝ) x ε)
    (fun x => Activation.Math.smoothAbsDerivSpec (α := ℝ) x ε)

/-- Global `NodeFDerivCorrect` for `smoothAbs` (requires `0 < ε`). -/
def smoothAbsFderiv {Γ : List Shape} {s : Shape} (idx : Idx Γ s) (ε : ℝ) (hε : 0 < ε) :
    NodeFDerivCorrect (smoothAbs (Γ := Γ) (s := s) idx ε) :=
  elemwiseFderiv (Γ := Γ) (s := s) idx
    (fun x => Activation.Math.smoothAbsSpec (α := ℝ) x ε)
    (fun x => Activation.Math.smoothAbsDerivSpec (α := ℝ) x ε)
    (fun z => Proofs.smooth_abs_deriv_correct (x := z) (ε := ε) hε)

/-- Runtime `exp` node (elementwise). -/
def exp {Γ : List Shape} {s : Shape} (idx : Idx Γ s) : Node Γ s :=
  elemwise (Γ := Γ) (s := s) idx Real.exp Real.exp

/-- Global `NodeFDerivCorrect` for the elementwise exponential. -/
def expFderiv {Γ : List Shape} {s : Shape} (idx : Idx Γ s) :
    NodeFDerivCorrect (exp (Γ := Γ) (s := s) idx) :=
  elemwiseFderiv (Γ := Γ) (s := s) idx Real.exp Real.exp (fun z => Real.hasDerivAt_exp z)

/-- Runtime `sinh` node (elementwise). -/
def sinh {Γ : List Shape} {s : Shape} (idx : Idx Γ s) : Node Γ s :=
  elemwise (Γ := Γ) (s := s) idx Activation.Math.sinhSpec Activation.Math.sinhDerivSpec

/-- Global `NodeFDerivCorrect` for elementwise hyperbolic sine. -/
def sinhFderiv {Γ : List Shape} {s : Shape} (idx : Idx Γ s) :
    NodeFDerivCorrect (sinh (Γ := Γ) (s := s) idx) :=
  elemwiseFderiv (Γ := Γ) (s := s) idx
    Activation.Math.sinhSpec Activation.Math.sinhDerivSpec
    (fun z => Proofs.sinh_deriv_correct (x := z))

/-- Runtime `cosh` node (elementwise). -/
def cosh {Γ : List Shape} {s : Shape} (idx : Idx Γ s) : Node Γ s :=
  elemwise (Γ := Γ) (s := s) idx Activation.Math.coshSpec Activation.Math.coshDerivSpec

/-- Global `NodeFDerivCorrect` for elementwise hyperbolic cosine. -/
def coshFderiv {Γ : List Shape} {s : Shape} (idx : Idx Γ s) :
    NodeFDerivCorrect (cosh (Γ := Γ) (s := s) idx) :=
  elemwiseFderiv (Γ := Γ) (s := s) idx
    Activation.Math.coshSpec Activation.Math.coshDerivSpec
    (fun z => Proofs.cosh_deriv_correct (x := z))

/-- Runtime `elu` node (elementwise; nondifferentiable at zero unless `alpha = 1`). -/
def elu {Γ : List Shape} {s : Shape} (idx : Idx Γ s) (alpha : ℝ) : Node Γ s :=
  elemwise (Γ := Γ) (s := s) idx
    (fun x => Activation.Math.eluSpec x alpha)
    (fun x => Activation.Math.eluDerivSpec x alpha)

/--
Pointwise `NodeFDerivCorrectAt` for ELU under the usual no-coordinate-at-the-kink assumption.

For arbitrary `alpha`, ELU has left derivative `alpha` and right derivative `1` at zero. Keeping the
hypothesis here avoids baking PyTorch's subgradient convention into a mathematical derivative
theorem.
-/
def eluFderivAt {Γ : List Shape} {s : Shape} (idx : Idx Γ s) (alpha : ℝ) (xV : CtxVec Γ)
    (hx : ∀ i, CtxVec.get (Γ := Γ) (s := s) idx xV i ≠ 0) :
    NodeFDerivCorrectAt (elu (Γ := Γ) (s := s) idx alpha) xV :=
  elemwiseFderivAt (Γ := Γ) (s := s) idx
    (fun x => Activation.Math.eluSpec x alpha)
    (fun x => Activation.Math.eluDerivSpec x alpha)
    xV
    (fun i => Proofs.elu_deriv_correct
      (x := CtxVec.get (Γ := Γ) (s := s) idx xV i) (α := alpha) (h := hx i))

/-- Unary node applying an analytically-correct `OpSpec` at a context index. -/
def unaryOp {Γ : List Shape} {inDim outDim : Nat}
    (idx : Idx Γ (.dim inDim .scalar))
    (C : OpSpecFDerivCorrect inDim outDim) : Node Γ (.dim outDim .scalar) :=
  let hOut : Spec.Shape.size (.dim outDim .scalar) = outDim := by simp [Spec.Shape.size]
  Node.ofFn (Γ := Γ) (τ := .dim outDim .scalar)
    (f := fun ctxV => castVec hOut.symm (C.forwardVec ((getVecCLM (Γ := Γ) (n := inDim) idx) ctxV)))
    (jvp := fun ctxV dctxV =>
      castVec hOut.symm
        (getScalarE (C.correct.jvp (ofFnE ((getVecCLM (Γ := Γ) (n := inDim) idx) ctxV))
          (ofFnE ((getVecCLM (Γ := Γ) (n := inDim) idx) dctxV)))))
    (vjp := fun ctxV δV =>
      let δV' : Vec outDim := castVec hOut δV
      singleVec (Γ := Γ) (n := inDim) idx
        (getScalarE (C.correct.op.backward (ofFnE ((getVecCLM (Γ := Γ) (n := inDim) idx) ctxV))
          (ofFnE δV'))))
    (correct_inner := by
      intro ctxV dctxV δV
      rw [inner_castVec_left, inner_getVec_singleVec]
      have h := C.correct.correct
        (x := ofFnE (getVec (Γ := Γ) (n := inDim) idx ctxV))
        (dx := ofFnE (getVec (Γ := Γ) (n := inDim) idx dctxV))
        (δ := ofFnE (castVec hOut δV))
      simpa [dot_eq_inner_vec, getScalarE_ofFnE, getVecCLM_apply] using h)

/-- `NodeFDerivCorrect` for `unaryOp`. -/
def unaryOpFderiv {Γ : List Shape} {inDim outDim : Nat}
    (idx : Idx Γ (.dim inDim .scalar))
    (C : OpSpecFDerivCorrect inDim outDim) :
    NodeFDerivCorrect (unaryOp (Γ := Γ) (inDim := inDim) (outDim := outDim) idx C) :=
{ deriv := fun xV =>
    let hOut : Spec.Shape.size (.dim outDim .scalar) = outDim := by simp [Spec.Shape.size]
    (Graph.castCLM (h := hOut.symm)).comp
      ((C.deriv ((getVecCLM (Γ := Γ) (n := inDim) idx) xV)).comp (getVecCLM (Γ := Γ) (n := inDim)
        idx))
  hasFDerivAt := by
    intro xV
    let hOut : Spec.Shape.size (.dim outDim .scalar) = outDim := by simp [Spec.Shape.size]
    -- projection is linear
    have hproj :
        HasFDerivAt (fun xV : CtxVec Γ => (getVecCLM (Γ := Γ) (n := inDim) idx) xV)
          (getVecCLM (Γ := Γ) (n := inDim) idx) xV :=
      (getVecCLM (Γ := Γ) (n := inDim) idx).hasFDerivAt (x := xV)
    have hC :
        HasFDerivAt (fun x : Vec inDim => C.forwardVec x)
          (C.deriv ((getVecCLM (Γ := Γ) (n := inDim) idx) xV))
          ((getVecCLM (Γ := Γ) (n := inDim) idx) xV) := by
      simpa [OpSpecFDerivCorrect.forwardVec] using (C.hasFDerivAt ((getVecCLM (Γ := Γ) (n := inDim)
        idx) xV))
    have hcomp : HasFDerivAt (fun xV : CtxVec Γ => C.forwardVec ((getVecCLM (Γ := Γ) (n := inDim)
      idx) xV))
        ((C.deriv ((getVecCLM (Γ := Γ) (n := inDim) idx) xV)).comp (getVecCLM (Γ := Γ) (n := inDim)
          idx)) xV :=
      (hC.comp xV hproj)
    -- output cast is linear
    have hcast :
        HasFDerivAt (fun y : Vec outDim => castVec hOut.symm y) (Graph.castCLM (h := hOut.symm))
          (C.forwardVec ((getVecCLM (Γ := Γ) (n := inDim) idx) xV)) := by
      simpa [Graph.castCLM] using ((Graph.castCLM (h := hOut.symm)).hasFDerivAt (x := C.forwardVec
        ((getVecCLM (Γ := Γ) (n := inDim) idx) xV)))
    have hfinal := hcast.comp xV hcomp
    -- `unaryOp.forwardVec` is definitional to this composition (after unfolding casts).
    have htarget :
        (unaryOp (Γ := Γ) (inDim := inDim) (outDim := outDim) idx C).forwardVec =
          (fun ctxV : CtxVec Γ =>
            castVec hOut.symm (C.forwardVec ((getVecCLM (Γ := Γ) (n := inDim) idx) ctxV))) := by
      funext ctxV
      simp [unaryOp, Node.forwardVec_ofFn]
    rw [htarget]
    simpa [Function.comp_def] using hfinal
  jvp_eq := by
    intro xV dxV
    let hOut : Spec.Shape.size (.dim outDim .scalar) = outDim := by simp [Spec.Shape.size]
    have hjvp :=
      C.jvp_eq
        (xV := (getVecCLM (Γ := Γ) (n := inDim) idx) xV)
        (dxV := (getVecCLM (Γ := Γ) (n := inDim) idx) dxV)
    -- apply the output cast to both sides
    have hjvp' := congrArg (castVec hOut.symm) hjvp
    -- unfold `unaryOp` and simplify
    simpa [unaryOp, getVecCLM_apply, getVec, Graph.castCLM, hOut, ContinuousLinearMap.comp_apply]
      using hjvp' }

/-- Linear layer as a single tape node (fixed weights/bias in the `Spec.LinearSpec`). -/
def linear {Γ : List Shape} {inDim outDim : Nat}
    (x : Idx Γ (.dim inDim .scalar)) (m : Spec.LinearSpec ℝ inDim outDim) :
    Node Γ (.dim outDim .scalar) :=
  unaryOp (Γ := Γ) (inDim := inDim) (outDim := outDim) x (OpSpecFDerivCorrect.linear m)

/-- `NodeFDerivCorrect` for `linear`: the node derivative is the spec's `OpSpec` derivative. -/
def linearFderiv {Γ : List Shape} {inDim outDim : Nat}
    (x : Idx Γ (.dim inDim .scalar)) (m : Spec.LinearSpec ℝ inDim outDim) :
    NodeFDerivCorrect (linear (Γ := Γ) (inDim := inDim) (outDim := outDim) x m) :=
  unaryOpFderiv (Γ := Γ) (inDim := inDim) (outDim := outDim) x (OpSpecFDerivCorrect.linear m)

end TapeNodes

end

end Autograd
end Proofs
