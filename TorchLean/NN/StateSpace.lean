/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.Dynamics
public import TorchLean.Tensor.Layers.StateSpace

/-!
# State-space dynamical systems

State-space sequence models are recurrent dynamical systems:

$$
h_{t+1}=A_t h_t+B_t x_t,\qquad
y_t=C_t h_t+D_t x_t.
$$

The full S4/Mamba family uses structured matrices and input-dependent ("selective") parameters.
The definitions here cover the diagonal (channelwise) version, which is:

- the recurrence of `Tensor.selectiveScan` along the step axis with shared coefficients,
- enough to express the formal affine-scan algebra,
- and a useful educational baseline for examples.

`DiagonalSSM.drivenSystem` views a model over `ℝ` as a `DrivenSystem` from
`TorchLean/NN/Dynamics.lean`, so `DrivenSystem.iterate` runs it on an input stream, and after
`DrivenSystem.freeze` Mathlib's iterate `step^[n]`, `Function.IsFixedPt` and
`Robustness.IsContractive` of the step apply.

We wrote `DiagonalSSM` by hand in discrete time. `A` is the transition itself, with no step size
and no discretization rule, where S4 builds its transition from a continuous one by the bilinear
method and Mamba by the zero-order hold. `readout` takes whichever state it's given, and
`TorchLean.S4.step` passes the updated one, so there `y_t` reads `h_{t+1}`. With `A i = 0.9` and
`B i = 1`, a unit impulse has decayed to `0.9 ^ 10 ≈ 0.35` ten steps later.

References:
- Albert Gu, Karan Goel, and Christopher Ré (2022), "Efficiently Modeling Long Sequences with
  Structured State Spaces", ICLR 2022, arXiv:2111.00396, eq. (3).
- Albert Gu and Tri Dao (2024), "Mamba: Linear-Time Sequence Modeling with Selective State
  Spaces", COLM 2024, arXiv:2312.00752, eq. (4).
-/

@[expose] public section


namespace TorchLean.Dynamics

open StateSpace

/--
A diagonal state-space model with elementwise input, state, and output channels.

It is the smallest useful SSM shape: it has the recurrent memory path of Mamba/S4 and is easy to
verify. Richer models can wrap this core with linear projections and gates.
-/
structure DiagonalSSM (α : Type) [Storage α] (dim : Nat) where
  /-- Elementwise recurrent multiplier. Stability usually requires $\lvert A_i\rvert<1$. -/
  A : Tensor α [dim]
  /-- Elementwise input-to-state gain. -/
  B : Tensor α [dim]
  /-- Elementwise state-to-output gain. -/
  C : Tensor α [dim]
  /-- Elementwise residual / skip gain. -/
  D : Tensor α [dim]

namespace DiagonalSSM

variable {α : Type} [Storage α] [Add α] [Mul α] {dim : Nat}

/-- One recurrent state update $h'=A\odot h+B\odot x$. -/
def step (m : DiagonalSSM α dim)
    (h x : Tensor α [dim]) : Tensor α [dim] :=
  (m.A * h) + (m.B * x)

/-- Each channel of a state update is the scalar affine update `A i * h i + B i * x i`. -/
@[simp] theorem step_apply (m : DiagonalSSM α dim) (h x : Tensor α [dim])
    (i : Shape.Coord [dim]) :
    m.step h x i = m.A i * h i + m.B i * x i := by
  simp [step]

/-- Readout $y=C\odot h+D\odot x$. -/
def readout (m : DiagonalSSM α dim)
    (h x : Tensor α [dim]) : Tensor α [dim] :=
  (m.C * h) + (m.D * x)

/-- Run a full input sequence and return all hidden states: `step` folded over `xs` from `h0`.
`TorchLean.Dynamics.DiagonalSSM.scan_eq_selectiveScan` proves it is `Tensor.selectiveScan`
along the step axis. -/
def scan (m : DiagonalSSM α dim)
    (h0 : Tensor α [dim])
    (xs : Array (Tensor α [dim])) :
    Array (Tensor α [dim]) :=
  (scanArray (fun h x => let next := m.step h x; (next, next)) h0 xs).2

/-- Scanning no input produces no hidden states. -/
@[simp] theorem scan_nil (m : DiagonalSSM α dim)
    (h0 : Tensor α [dim]) :
    m.scan h0 #[] = #[] := by
  rfl

/-- A diagonal SSM scan returns one hidden state for each input token. -/
@[simp] theorem scan_size (m : DiagonalSSM α dim)
    (h0 : Tensor α [dim])
    (xs : Array (Tensor α [dim])) :
    (m.scan h0 xs).size = xs.size :=
  scanArray_outputs_size _ h0 xs

end DiagonalSSM

/--
View a diagonal SSM over `ℝ` as a `DrivenSystem`, the bridge to the dynamics API.

`DrivenSystem` is stated over TorchLean's default `ℝ`, while the SSM record above is
polymorphic over scalar backends.
-/
def DiagonalSSM.drivenSystem {dim : Nat}
    (m : DiagonalSSM ℝ dim) : DrivenSystem (.dim dim .scalar) (.dim dim .scalar) where
  step := fun h x => m.step h x

/-- Viewing a diagonal SSM as a driven dynamical system does not change its step.

The wrapper exists so the dynamics theorems apply to the layer, and the proof is `rfl`: the wrapper
carries no extra content, so a bound proved about the system holds of the layer. -/
@[simp] theorem DiagonalSSM.drivenSystem_step {dim : Nat}
    (m : DiagonalSSM ℝ dim)
    (h x : Tensor ℝ [dim]) :
    (m.drivenSystem).step h x = m.step h x := by
  rfl

end TorchLean.Dynamics
