/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.Optimizer.Muon.QR
public meta import TorchLean.Tensor.Einops.Elab.Einsum
public meta import TorchLean.Tensor.Einops.Elab.Pack
public meta import TorchLean.Tensor.Literal
public meta import TorchLean.Tensor.Einops.Elab.Transform
public meta import TorchLean.Tensor.Tactic.Einops.Report
public import TorchLean.Tensor.Literal

/-!
# What a Muon Step Certificate Says

A Muon update first forms a momentum buffer, then chooses a matrix direction `Q`, and finally
updates parameters by `P' = P - lr * Q`. An **exact step certificate** proves that the
chosen direction has orthonormal columns (`QᵀQ = I`) and is the direction actually used in that
update. It also fixes the updated momentum buffer, the only part of the optimizer state that Muon
changes.

Start with `Concrete` below. Its one-column direction is `Q = (3/5, 4/5)ᵀ`, so the entire Gram
condition is `(3/5)² + (4/5)² = 1`. Lean proves this equation, rejects `(1, 1)ᵀ`, constructs a real
`TorchLean.Muon.ExactCertifiedStep`, and derives the updated parameters `(97/50, 73/25)ᵀ`.

The example uses exact real arithmetic and an already-normalized buffer, and for that buffer the
identity backend suffices because it returns its input unchanged. The later QR and Newton–Schulz
examples show which backend hypotheses a general application must supply. None of these step
certificates proves convergence, lower loss, speed, or agreement with CUDA.

Build with `scripts/lake.sh build TorchLeanExamples.DeepDives.Muon`. The file is a
proof tutorial, with no CLI or training run. Runtime configuration goes through
`TorchLean.optim.muon`, and the theorem statements use `TorchLean.Muon`.
-/

@[expose] public section

namespace TorchLean.Examples.Muon

/-! ## A certificate with actual numbers -/

namespace Concrete

open TorchLean.Tensor

/-- A two-row, one-column matrix containing `x` above `y`. -/
noncomputable def column (x y : ℝ) : Tensor ℝ [2, 1] :=
  Tensor.dim fun i => Tensor.ofFn fun _ => if i = 0 then x else y

/-- Reading the column exposes the two numbers used in the scalar calculations below. -/
@[simp] theorem get2_column (x y : ℝ) (i : Fin 2) (j : Fin 1) :
    get2 (column x y) i j = if i = 0 then x else y := by
  simp [column, get2]

/-- The 3–4–5 triangle supplies a unit column, `Q = (3/5, 4/5)ᵀ`. -/
noncomputable def direction : Tensor ℝ [2, 1] :=
  column (3 / 5) (4 / 5)

/-- The single entry of `QᵀQ` is `(3/5)² + (4/5)² = 1`. -/
theorem direction_has_exact_gram : Muon.HasExactColumnGram direction := by
  apply ext_get2
  intro i j
  fin_cases i
  fin_cases j
  change get2 (Tensor.matmul (Tensor.permute direction
      (Shape.adjacentSwap [2, 1] 0) [1, 2]) direction) 0 0 = _
  rw [get2_matmul_eq_sum]
  simp only [Fin.sum_univ_two, get2_transpose]
  norm_num [direction, Tensor.identity, get2, column]

/-- The column `(1, 1)ᵀ` fails the certificate because its squared length is `2`. -/
theorem unnormalized_not_exact_gram :
    ¬ Muon.HasExactColumnGram (column 1 1) := by
  intro h
  have hEntry := congrArg (fun matrix => get2 matrix 0 0) h
  change get2 (Tensor.matmul (Tensor.permute (column 1 1)
      (Shape.adjacentSwap [2, 1] 0) [1, 2]) (column 1 1)) 0 0 = _ at hEntry
  rw [get2_matmul_eq_sum] at hEntry
  simp only [Fin.sum_univ_two, get2_transpose] at hEntry
  norm_num [Tensor.identity, get2, column] at hEntry

/-!
Next, take `Q` as the gradient, start from parameters `(2, 3)ᵀ` and set momentum to zero, so the
fresh buffer is `0 * oldBuffer + Q = Q`. The identity backend passes this unit column through
unchanged. A non-unit gradient would need a different backend or would fail the exact Gram
obligation, as the counterexample above shows.
-/

/-- Parameters before the step. -/
noncomputable def parameters : Tensor ℝ [2, 1] :=
  column 2 3

/-- Learning rate `1/10`, zero momentum, and the identity backend, which suffices for this `Q`. -/
noncomputable def state : Muon.State ℝ [2, 1] :=
  Muon.init (1 / 10) 0 Muon.identityOrthogonalizer parameters

/-- Zero momentum makes the fresh buffer exactly the supplied gradient. -/
theorem fresh_buffer_eq_direction :
    (Muon.update state parameters direction).optimizerState.momentumBuffer = direction := by
  apply ext_get2
  intro i j
  simp [Muon.update, state, Muon.init, MomentumSGD.init,
    Optimizer.updateMomentumBuffer, add_def, Tensor.scale]

/--
A complete exact step certificate for these concrete inputs, with no backend hypothesis left open.
The first field identifies the direction and proves its unit Gram matrix, and the other two tie
that direction to the next optimizer state and to the parameter update.
-/
theorem step_certified :
    Muon.ExactCertifiedStep state parameters direction direction := by
  constructor
  · constructor
    · change direction =
        (Muon.update state parameters direction).optimizerState.momentumBuffer
      exact fresh_buffer_eq_direction.symm
    · exact direction_has_exact_gram
  · rfl
  · change (parameters -
      (Tensor.scale (Muon.update state parameters direction).optimizerState.momentumBuffer
        state.lr)) = _
    rw [fresh_buffer_eq_direction]

/-- Consuming the certificate gives `(2, 3)ᵀ - (1/10) * (3/5, 4/5)ᵀ = (97/50, 73/25)ᵀ`. -/
theorem updated_parameters_eq :
    (Muon.update state parameters direction).parameters = column (97 / 50) (73 / 25) := by
  rw [step_certified.parameters_eq]
  apply ext_get2
  intro i j
  fin_cases i <;> fin_cases j <;>
    norm_num [sub_def, Tensor.scale, parameters, direction, state, Muon.init,
      MomentumSGD.init]

end Concrete

/-!
## Supplying a general orthogonalization backend

For multiple columns, `QᵀQ = I` says that each column has length one and distinct columns have
inner product zero. The QR example requires positive pivots. The approximate Newton–Schulz
example requires a proved entrywise residual bound on `QᵀQ - I`, and a choice of iteration count
does not establish that bound by itself.

Both theorems below are conditional API examples. Where `Concrete.step_certified` discharges every
obligation, they leave the backend's success obligation to the caller and then extract the useful
fields of the resulting certificate.
-/

/--
With the QR checked backend, a proof of positive pivots gives a certified step, from which we
recover both the exact Gram certificate for the direction and the parameter-update equation.
-/
theorem qr_step_exact_gram {m n : Nat}
    (lr momentum : ℝ) (momentumBuffer parameters gradients : Tensor ℝ [m, n])
    (hpivots :
      Muon.QR.HasPositivePivots
        (Muon.update
          ({ lr := lr, momentum := momentum, momentumBuffer := momentumBuffer,
             orthogonalizer := Muon.QR.orthogonalizer (m := m) (n := n) } :
            Muon.State ℝ [m, n])
          parameters gradients).optimizerState.momentumBuffer) :
    ∃ direction : Tensor ℝ [m, n],
      Muon.HasExactColumnGram direction ∧
      (Muon.update
          ({ lr := lr, momentum := momentum, momentumBuffer := momentumBuffer,
             orthogonalizer := Muon.QR.orthogonalizer (m := m) (n := n) } :
            Muon.State ℝ [m, n])
          parameters gradients).parameters =
        parameters -
          Tensor.scale direction lr := by
  obtain ⟨direction, hstep⟩ :=
    Muon.QR.exists_exactCertifiedStep
      lr momentum momentumBuffer parameters gradients hpivots
  exact ⟨direction,
    hstep.hasExactColumnGram,
    hstep.parameters_eq⟩

/--
With the residual-checked Newton-Schulz backend, a residual proof gives a certified approximate
step, from which we recover the residual-bound certificate and the parameter equation.
-/
theorem newtonSchulz_step_approx_gram
    {α : Type} [Storage α] [Context α] {m n : Nat} {eps : α}
    (coeffs : Muon.NewtonSchulz.Coeffs α) (steps : Nat)
    (lr momentum : α) (momentumBuffer parameters gradients : Tensor α [m, n])
    (hresidual :
      Muon.ApproxOrthogonalizesBuffer eps
        (Muon.NewtonSchulz.orthogonalizer (α := α) (m := m) (n := n) coeffs steps)
        (Muon.update
          ({ lr := lr, momentum := momentum, momentumBuffer := momentumBuffer,
             orthogonalizer :=
              Muon.NewtonSchulz.orthogonalizer (α := α) (m := m) (n := n) coeffs steps } :
            Muon.State α [m, n])
          parameters gradients).optimizerState.momentumBuffer) :
    ∃ direction : Tensor α [m, n],
      Muon.HasApproxColumnGram eps direction ∧
      (Muon.update
          ({ lr := lr, momentum := momentum, momentumBuffer := momentumBuffer,
             orthogonalizer :=
              Muon.NewtonSchulz.orthogonalizer (α := α) (m := m) (n := n) coeffs steps } :
            Muon.State α [m, n])
          parameters gradients).parameters =
        parameters -
          Tensor.scale direction lr := by
  obtain ⟨direction, hstep⟩ :=
    Muon.approxCertifiedStep_of_success
      (backend := Muon.NewtonSchulz.residualChecked
        (α := α) (m := m) (n := n) coeffs steps eps)
      lr momentum momentumBuffer parameters gradients hresidual
  exact ⟨direction,
    hstep.hasApproxColumnGram,
    hstep.parameters_eq⟩

end TorchLean.Examples.Muon
