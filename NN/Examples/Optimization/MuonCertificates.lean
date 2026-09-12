/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.MLTheory.Optimization.Muon
public import NN.Tensor

/-!
# What a Muon Step Certificate Says

A Muon update first forms a momentum buffer, then chooses a matrix direction `Q`, and finally
updates parameters by `P' = P - learningRate * Q`. An **exact step certificate** proves that the
chosen direction has orthonormal columns (`QᵀQ = I`) and is the direction actually used in that
update. It also records the new momentum state.

Start with `Concrete` below. Its one-column direction is `Q = (3/5, 4/5)ᵀ`, so the entire Gram
condition is `(3/5)² + (4/5)² = 1`. Lean proves this equation, rejects `(1, 1)ᵀ`, constructs a real
`Optim.Muon.ExactCertifiedStep`, and derives the updated parameters `(97/50, 73/25)ᵀ`.

This example uses exact real arithmetic and an already-normalized buffer. The identity backend
is sufficient for this one buffer; it does not orthogonalize arbitrary inputs. The later QR and
Newton–Schulz examples show which additional backend hypotheses a general application must supply.
None of these step certificates proves convergence, lower loss, speed, or agreement with CUDA.

Build with `lake build NN.Examples.Optimization`. This is a proof tutorial, with no CLI or training
run. Runtime configuration uses `TorchLean.optim.muon.optimizer`; theorem statements use
`Optim.Muon`.
-/

@[expose] public section

namespace NN.Examples.Optimization.MuonCertificates

open TorchLean

/-! ## A certificate with actual numbers -/

namespace Concrete

open TorchLean.Tensor Spec

/-- A two-row, one-column matrix containing `x` above `y`. -/
noncomputable def column (x y : ℝ) : Tensor ℝ [2, 1] :=
  Tensor.dim fun i => Tensor.ofFn fun _ => if i = 0 then x else y

/-- Reading the column exposes the two numbers used in the scalar calculations below. -/
@[simp] theorem get2_column (x y : ℝ) (i : Fin 2) (j : Fin 1) :
    get2 (column x y) i j = if i = 0 then x else y := by
  simp [column, get2, Spec.get]

/-- The 3–4–5 triangle supplies a unit column, `Q = (3/5, 4/5)ᵀ`. -/
noncomputable def direction : Tensor ℝ [2, 1] :=
  column (3 / 5) (4 / 5)

/-- The single entry of `QᵀQ` is `(3/5)² + (4/5)² = 1`. -/
theorem direction_has_exact_gram : Optim.Muon.HasExactColumnGram direction := by
  apply matrix_ext
  intro i j
  fin_cases i
  fin_cases j
  change get2 (matMulSpec (swapAdjacentAxes direction 0) direction) 0 0 = _
  rw [get2_mat_mul_spec]
  simp only [Fin.sum_univ_two, get2_matrix_transpose_spec]
  norm_num [direction, identityTensorSpec, get2, Spec.get, column]

/-- The column `(1, 1)ᵀ` fails the certificate: its squared length is `2`, not `1`. -/
theorem unnormalized_direction_rejected :
    ¬ Optim.Muon.HasExactColumnGram (column 1 1) := by
  intro h
  have hEntry := congrArg (fun matrix => get2 matrix 0 0) h
  change get2 (matMulSpec (swapAdjacentAxes (column 1 1) 0) (column 1 1)) 0 0 = _ at hEntry
  rw [get2_mat_mul_spec] at hEntry
  simp only [Fin.sum_univ_two, get2_matrix_transpose_spec] at hEntry
  norm_num [identityTensorSpec, get2, Spec.get, column] at hEntry

/-!
Now use `Q` as the gradient, start from parameters `(2, 3)ᵀ`, and set momentum to zero.
The fresh buffer is therefore `0 * oldBuffer + Q = Q`. Passing it through the identity backend
leaves the already-proved unit column unchanged. A non-unit gradient would require a different
backend or would fail the exact Gram obligation, as the preceding counterexample shows.
-/

/-- Parameters before the step. -/
noncomputable def parameters : Tensor ℝ [2, 1] :=
  column 2 3

/-- Learning rate `1/10`, zero momentum, and a backend valid for this already-unit direction. -/
noncomputable def state : Optim.Muon.State ℝ [2, 1] :=
  Optim.Muon.init (1 / 10) 0 Optim.Muon.identityOrthogonalizer parameters

/-- Zero momentum makes the fresh buffer exactly the supplied gradient. -/
theorem fresh_buffer_eq_direction :
    (Optim.Muon.update state parameters direction).optimizerState.momentumBuffer = direction := by
  apply matrix_ext
  intro i j
  simp [Optim.Muon.update, state, Optim.Muon.init, Optim.updateMomentumBuffer,
    addSpec, scaleSpec]

/--
A complete exact step certificate for these concrete inputs, with no backend hypothesis left open.
The first field proves the actual direction and its unit Gram matrix; the remaining fields tie
that direction to the next optimizer state and parameter update.
-/
theorem step_certified :
    Optim.Muon.ExactCertifiedStep state parameters direction direction := by
  constructor
  · constructor
    · change direction =
        (Optim.Muon.update state parameters direction).optimizerState.momentumBuffer
      exact fresh_buffer_eq_direction.symm
    · exact direction_has_exact_gram
  · rfl
  · change subSpec parameters
      (scaleSpec (Optim.Muon.update state parameters direction).optimizerState.momentumBuffer
        state.learningRate) = _
    rw [fresh_buffer_eq_direction]

/-- Consuming the certificate gives `(2, 3)ᵀ - (1/10) * (3/5, 4/5)ᵀ = (97/50, 73/25)ᵀ`. -/
theorem updated_parameters_eq :
    (Optim.Muon.update state parameters direction).parameters = column (97 / 50) (73 / 25) := by
  rw [step_certified.parameters_eq]
  apply matrix_ext
  intro i j
  fin_cases i <;> fin_cases j <;>
    norm_num [subSpec, scaleSpec, parameters, direction, state, Optim.Muon.init]

end Concrete

/-!
## Supplying a general orthogonalization backend

For multiple columns, `QᵀQ = I` says that each column has length one and distinct columns have
inner product zero. The QR example requires positive pivots. The approximate Newton–Schulz
example requires a proved entrywise residual bound on `QᵀQ - I`; choosing an iteration count alone
does not establish that bound.

These are conditional API examples. Unlike `Concrete.step_certified`, they leave the backend's
success obligation to the caller and then extract the useful fields of the resulting certificate.
-/

/--
Using the QR checked backend, a positive-pivot proof gives a certified step; from that step we can
recover both the exact Gram certificate for the direction and the parameter-update equation.
-/
theorem qr_update_step_direction_has_exact_gram {m n : Nat}
    (learningRate momentum : ℝ) (momentumBuffer parameters gradients : Tensor ℝ [m, n])
    (hpivots :
      Optim.Muon.HasPositiveQRPivots
        (Optim.Muon.update
          ({ learningRate := learningRate, momentum := momentum, momentumBuffer := momentumBuffer,
             orthogonalizer := Optim.Muon.qrOrthogonalizer (m := m) (n := n) } :
            Optim.Muon.State ℝ [m, n])
          parameters gradients).optimizerState.momentumBuffer) :
    ∃ direction : Tensor ℝ [m, n],
      Optim.Muon.HasExactColumnGram direction ∧
      (Optim.Muon.update
          ({ learningRate := learningRate, momentum := momentum, momentumBuffer := momentumBuffer,
             orthogonalizer := Optim.Muon.qrOrthogonalizer (m := m) (n := n) } :
            Optim.Muon.State ℝ [m, n])
          parameters gradients).parameters =
        Tensor.subSpec parameters
          (Tensor.scaleSpec direction learningRate) := by
  obtain ⟨direction, hstep⟩ :=
    Optim.Muon.update_has_exact_certified_step_qr
      learningRate momentum momentumBuffer parameters gradients hpivots
  exact ⟨direction,
    hstep.hasExactColumnGram,
    hstep.parameters_eq⟩

/--
Using the residual-checked Newton-Schulz backend, a residual proof gives a certified approximate
step; from that step we can recover the residual-bound certificate and the parameter equation.
-/
theorem newtonSchulz_update_step_direction_has_approx_gram
    {α : Type} [Storage α] [Context α] {m n : Nat} {eps : α}
    (coeffs : Optim.Muon.NewtonSchulzCoeffs α) (steps : Nat)
    (learningRate momentum : α) (momentumBuffer parameters gradients : Tensor α [m, n])
    (hresidual :
      Optim.Muon.ApproxOrthogonalizesBuffer eps
        (Optim.Muon.newtonSchulzOrthogonalizer (α := α) (m := m) (n := n) coeffs steps)
        (Optim.Muon.update
          ({ learningRate := learningRate, momentum := momentum, momentumBuffer := momentumBuffer,
             orthogonalizer :=
              Optim.Muon.newtonSchulzOrthogonalizer (α := α) (m := m) (n := n) coeffs steps } :
            Optim.Muon.State α [m, n])
          parameters gradients).optimizerState.momentumBuffer) :
    ∃ direction : Tensor α [m, n],
      Optim.Muon.HasApproxColumnGram eps direction ∧
      (Optim.Muon.update
          ({ learningRate := learningRate, momentum := momentum, momentumBuffer := momentumBuffer,
             orthogonalizer :=
              Optim.Muon.newtonSchulzOrthogonalizer (α := α) (m := m) (n := n) coeffs steps } :
            Optim.Muon.State α [m, n])
          parameters gradients).parameters =
        Tensor.subSpec parameters
          (Tensor.scaleSpec direction learningRate) := by
  obtain ⟨direction, hstep⟩ :=
    Optim.Muon.approxCertifiedStep_of_checkedBackend
      (backend := Optim.Muon.newtonSchulzResidualCheckedOrthogonalizer
        (α := α) (m := m) (n := n) coeffs steps eps)
      learningRate momentum momentumBuffer parameters gradients hresidual
  exact ⟨direction,
    hstep.hasApproxColumnGram,
    hstep.parameters_eq⟩

end NN.Examples.Optimization.MuonCertificates
