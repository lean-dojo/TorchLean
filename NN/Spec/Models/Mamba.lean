/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Core.Tensor.Linalg
public import NN.Spec.Dynamics.StateSpace
public import NN.Spec.Layers.Activation

/-!
# Mamba-style selective state-space blocks

Mamba replaces quadratic attention with a linear-time selective state-space recurrence.  In full
models, the token controls discretization and input/output state parameters.

This file exposes two layers:

- `MambaBlockSpec`: a compact theorem-friendly diagonal SSM block kept for scan laws and kernel
  validation.
- `SelectiveMambaBlockSpec`: a fuller Mamba-style block with input/gate projections, causal
  depthwise convolution, SiLU, token-dependent `Delta/B/C`, diagonal selective scan, gated output,
  and output projection.

The compact block is intentionally retained: it is the smallest reusable core for proving scan
algebra and for validating CUDA kernels.  The full block builds the paper-style Mamba dataflow on
top of the same affine-scan idea.

- recurrent selective scan (`h ← A ⊙ h + B ⊙ x_state`),
- a gated state readout,
- tokenwise input/output projections.

References:
- Gu, Dao. "Mamba: Linear-Time Sequence Modeling with Selective State Spaces", COLM 2024.
- Dao, Gu. "Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured
  State Space Duality" (Mamba-2), ICML 2024.
-/

@[expose] public section

namespace Models

open Spec
open Tensor
open NN.Spec.Dynamics

/-- Parameters for a compact diagonal Mamba-style block. -/
structure MambaBlockSpec (α : Type) (inputDim stateDim outputDim : Nat) where
  /-- Input projection into SSM state channels. -/
  inProj : Tensor α (.dim inputDim (.dim stateDim .scalar))
  /-- Gate projection. The gate is `sigmoid(x @ gateProj)`. -/
  gateProj : Tensor α (.dim inputDim (.dim stateDim .scalar))
  /-- Output projection from gated state channels. -/
  outProj : Tensor α (.dim stateDim (.dim outputDim .scalar))
  /-- Diagonal state-space core. -/
  ssm : DiagonalSSM α stateDim

namespace MambaBlockSpec

variable {α : Type} [Context α]
variable {inputDim stateDim outputDim : Nat}

/-- Input-to-state projection. -/
def projectInput (m : MambaBlockSpec α inputDim stateDim outputDim)
    (x : Tensor α (.dim inputDim .scalar)) : Tensor α (.dim stateDim .scalar) :=
  vecMatMulSpec x m.inProj

/-- Token-dependent sigmoid gate. -/
def gate (m : MambaBlockSpec α inputDim stateDim outputDim)
    (x : Tensor α (.dim inputDim .scalar)) : Tensor α (.dim stateDim .scalar) :=
  Tensor.mapSpec Activation.Math.sigmoidSpec (vecMatMulSpec x m.gateProj)

/-- One Mamba-style token step, returning `(new_state, output)`. -/
def step (m : MambaBlockSpec α inputDim stateDim outputDim)
    (h : Tensor α (.dim stateDim .scalar))
    (x : Tensor α (.dim inputDim .scalar)) :
    Tensor α (.dim stateDim .scalar) × Tensor α (.dim outputDim .scalar) :=
  let xState := m.projectInput x
  let h' := m.ssm.step h xState
  let yState := m.ssm.readout h' xState
  let gated := yState * m.gate x
  (h', vecMatMulSpec gated m.outProj)

/-- Run a list of tokens through the recurrent block. -/
def runList (m : MambaBlockSpec α inputDim stateDim outputDim)
    (h0 : Tensor α (.dim stateDim .scalar)) :
    List (Tensor α (.dim inputDim .scalar)) →
    Tensor α (.dim stateDim .scalar) × List (Tensor α (.dim outputDim .scalar))
  | [] => (h0, [])
  | x :: xs =>
      let (h1, y) := m.step h0 x
      let (hN, ys) := m.runList h1 xs
      (hN, y :: ys)

@[simp] theorem runList_nil (m : MambaBlockSpec α inputDim stateDim outputDim)
    (h0 : Tensor α (.dim stateDim .scalar)) :
    m.runList h0 [] = (h0, []) := by
  rfl

@[simp] theorem runList_cons (m : MambaBlockSpec α inputDim stateDim outputDim)
    (h0 : Tensor α (.dim stateDim .scalar))
    (x : Tensor α (.dim inputDim .scalar))
    (xs : List (Tensor α (.dim inputDim .scalar))) :
    m.runList h0 (x :: xs) =
      let (h1, y) := m.step h0 x
      let (hN, ys) := m.runList h1 xs
      (hN, y :: ys) := by
  rfl

/-- A Mamba recurrent pass emits one output token per input token. -/
theorem runList_outputs_length (m : MambaBlockSpec α inputDim stateDim outputDim)
    (h0 : Tensor α (.dim stateDim .scalar))
    (xs : List (Tensor α (.dim inputDim .scalar))) :
    (m.runList h0 xs).2.length = xs.length := by
  induction xs generalizing h0 with
  | nil =>
      simp
  | cons x rest ih =>
      simp [runList_cons, ih]

end MambaBlockSpec

/-- Parameters for a fuller Mamba-style selective SSM block.

Shape conventions:
- `inputDim`: token/input feature width,
- `innerDim`: expanded channel width used by Mamba's convolution and SSM path,
- `stateDim`: per-channel diagonal SSM state size,
- `outputDim`: output feature width,
- `convWidth`: causal depthwise-convolution width.

The recurrence state has shape `[innerDim, stateDim]`.  This mirrors the common implementation
view of Mamba where each expanded channel carries a small diagonal state vector.
-/
structure SelectiveMambaBlockSpec
    (α : Type) (inputDim innerDim stateDim outputDim convWidth : Nat) where
  /-- Content/input projection `x -> x_path`. -/
  xProj : Tensor α (.dim inputDim (.dim innerDim .scalar))
  /-- Gate projection `x -> z_path`. -/
  zProj : Tensor α (.dim inputDim (.dim innerDim .scalar))
  /-- Causal depthwise-convolution kernel, indexed by `(tap, channel)`. -/
  convKernel : Tensor α (.dim convWidth (.dim innerDim .scalar))
  /-- Causal depthwise-convolution bias. -/
  convBias : Tensor α (.dim innerDim .scalar)
  /-- Projection from activated convolution features to per-channel time steps `Delta`. -/
  dtProj : Tensor α (.dim innerDim (.dim innerDim .scalar))
  /-- Bias before the `softplus` time-step nonlinearity. -/
  dtBias : Tensor α (.dim innerDim .scalar)
  /-- Positive diagonal state rates `A[d,n]` used as `exp(-Delta[d] * A[d,n])`. -/
  A : Tensor α (.dim innerDim (.dim stateDim .scalar))
  /-- Token-dependent input-state projection `B_t = u_t @ bProj`. -/
  bProj : Tensor α (.dim innerDim (.dim stateDim .scalar))
  /-- Token-dependent state-output projection `C_t = u_t @ cProj`. -/
  cProj : Tensor α (.dim innerDim (.dim stateDim .scalar))
  /-- Per-channel residual/skip coefficient. -/
  dSkip : Tensor α (.dim innerDim .scalar)
  /-- Output projection from expanded channels to output features. -/
  outProj : Tensor α (.dim innerDim (.dim outputDim .scalar))

namespace SelectiveMambaBlockSpec

variable {α : Type} [Context α]
variable {inputDim innerDim stateDim outputDim convWidth : Nat}

/-- Content path projection. -/
def projectX (m : SelectiveMambaBlockSpec α inputDim innerDim stateDim outputDim convWidth)
    (x : Tensor α (.dim inputDim .scalar)) : Tensor α (.dim innerDim .scalar) :=
  vecMatMulSpec x m.xProj

/-- Gate path projection. -/
def projectZ (m : SelectiveMambaBlockSpec α inputDim innerDim stateDim outputDim convWidth)
    (x : Tensor α (.dim inputDim .scalar)) : Tensor α (.dim innerDim .scalar) :=
  vecMatMulSpec x m.zProj

/-- SiLU/Swish applied channelwise. -/
def siluVec (x : Tensor α (.dim innerDim .scalar)) : Tensor α (.dim innerDim .scalar) :=
  Tensor.mapSpec Activation.Math.swishSpec x

/--
Causal depthwise convolution from a newest-first history of projected tokens.

`history[0]` is the current projected token, `history[1]` is the previous token, etc.  Missing
history entries are treated as zero padding.
-/
def causalDepthwiseConv
    (m : SelectiveMambaBlockSpec α inputDim innerDim stateDim outputDim convWidth)
    (history : List (Tensor α (.dim innerDim .scalar))) :
    Tensor α (.dim innerDim .scalar) :=
  Tensor.dim (fun c : Fin innerDim =>
    Tensor.scalar <|
      (List.finRange convWidth).foldl
        (fun acc tap =>
          let zeroInner : Tensor α (.dim innerDim .scalar) :=
            Tensor.dim (fun _ => Tensor.scalar 0)
          let xTap : α := Tensor.vecGet (history.getD tap.val zeroInner) c
          acc + xTap * get2 m.convKernel tap c)
        (Tensor.vecGet m.convBias c))

/-- Token-dependent positive time steps `Delta = softplus(u @ dtProj + dtBias)`. -/
def delta
    (m : SelectiveMambaBlockSpec α inputDim innerDim stateDim outputDim convWidth)
    (u : Tensor α (.dim innerDim .scalar)) : Tensor α (.dim innerDim .scalar) :=
  Tensor.mapSpec Activation.Math.softplusSpec (vecMatMulSpec u m.dtProj + m.dtBias)

/-- Token-dependent input-state vector `B_t`. -/
def bToken
    (m : SelectiveMambaBlockSpec α inputDim innerDim stateDim outputDim convWidth)
    (u : Tensor α (.dim innerDim .scalar)) : Tensor α (.dim stateDim .scalar) :=
  vecMatMulSpec u m.bProj

/-- Token-dependent state-output vector `C_t`. -/
def cToken
    (m : SelectiveMambaBlockSpec α inputDim innerDim stateDim outputDim convWidth)
    (u : Tensor α (.dim innerDim .scalar)) : Tensor α (.dim stateDim .scalar) :=
  vecMatMulSpec u m.cProj

/--
One selective diagonal SSM update:

`h'[d,n] = exp(-Delta[d] * A[d,n]) * h[d,n] + (Delta[d] * B_t[n]) * u[d]`.
-/
def selectiveStateStep
    (m : SelectiveMambaBlockSpec α inputDim innerDim stateDim outputDim convWidth)
    (h : Tensor α (.dim innerDim (.dim stateDim .scalar)))
    (u : Tensor α (.dim innerDim .scalar)) :
    Tensor α (.dim innerDim (.dim stateDim .scalar)) :=
  let Δ := m.delta u
  let B := m.bToken u
  Tensor.dim (fun d : Fin innerDim =>
    Tensor.dim (fun n : Fin stateDim =>
      let deltaD := Tensor.vecGet Δ d
      let aBar := MathFunctions.exp (-(deltaD * get2 m.A d n))
      let bBar := deltaD * Tensor.vecGet B n
      Tensor.scalar (aBar * get2 h d n + bBar * Tensor.vecGet u d)))

/-- Read out expanded channels from the updated state using `C_t`, plus the Mamba skip path. -/
def stateReadout
    (m : SelectiveMambaBlockSpec α inputDim innerDim stateDim outputDim convWidth)
    (h : Tensor α (.dim innerDim (.dim stateDim .scalar)))
    (u : Tensor α (.dim innerDim .scalar)) :
    Tensor α (.dim innerDim .scalar) :=
  let C := m.cToken u
  Tensor.dim (fun d : Fin innerDim =>
    Tensor.scalar <|
      (List.finRange stateDim).foldl
        (fun acc n => acc + get2 h d n * Tensor.vecGet C n)
        (Tensor.vecGet m.dSkip d * Tensor.vecGet u d))

/--
One full Mamba token step from an already-updated convolution history.

The `history` argument is newest-first and must include the current projected content token.
-/
def stepWithHistory
    (m : SelectiveMambaBlockSpec α inputDim innerDim stateDim outputDim convWidth)
    (h : Tensor α (.dim innerDim (.dim stateDim .scalar)))
    (history : List (Tensor α (.dim innerDim .scalar)))
    (z : Tensor α (.dim innerDim .scalar)) :
    Tensor α (.dim innerDim (.dim stateDim .scalar)) × Tensor α (.dim outputDim .scalar) :=
  let u := siluVec (m.causalDepthwiseConv history)
  let h' := m.selectiveStateStep h u
  let y := m.stateReadout h' u
  let gated := y * siluVec z
  (h', vecMatMulSpec gated m.outProj)

/-- Internal recurrent runner carrying the causal convolution history. -/
def runListAux
    (m : SelectiveMambaBlockSpec α inputDim innerDim stateDim outputDim convWidth)
    (h0 : Tensor α (.dim innerDim (.dim stateDim .scalar)))
    (history : List (Tensor α (.dim innerDim .scalar))) :
    List (Tensor α (.dim inputDim .scalar)) →
    Tensor α (.dim innerDim (.dim stateDim .scalar)) × List (Tensor α (.dim outputDim .scalar))
  | [] => (h0, [])
  | x :: xs =>
      let xPath := m.projectX x
      let zPath := m.projectZ x
      let history' := xPath :: history
      let (h1, y) := m.stepWithHistory h0 history' zPath
      let (hN, ys) := m.runListAux h1 history' xs
      (hN, y :: ys)

/-- Run a sequence through the full selective Mamba block. -/
def runList
    (m : SelectiveMambaBlockSpec α inputDim innerDim stateDim outputDim convWidth)
    (h0 : Tensor α (.dim innerDim (.dim stateDim .scalar))) :
    List (Tensor α (.dim inputDim .scalar)) →
    Tensor α (.dim innerDim (.dim stateDim .scalar)) × List (Tensor α (.dim outputDim .scalar)) :=
  m.runListAux h0 []

@[simp] theorem runListAux_nil
    (m : SelectiveMambaBlockSpec α inputDim innerDim stateDim outputDim convWidth)
    (h0 : Tensor α (.dim innerDim (.dim stateDim .scalar)))
    (history : List (Tensor α (.dim innerDim .scalar))) :
    m.runListAux h0 history [] = (h0, []) := by
  rfl

@[simp] theorem runList_nil
    (m : SelectiveMambaBlockSpec α inputDim innerDim stateDim outputDim convWidth)
    (h0 : Tensor α (.dim innerDim (.dim stateDim .scalar))) :
    m.runList h0 [] = (h0, []) := by
  rfl

/-- The full Mamba recurrent pass emits one output token per input token. -/
theorem runListAux_outputs_length
    (m : SelectiveMambaBlockSpec α inputDim innerDim stateDim outputDim convWidth)
    (h0 : Tensor α (.dim innerDim (.dim stateDim .scalar)))
    (history : List (Tensor α (.dim innerDim .scalar)))
    (xs : List (Tensor α (.dim inputDim .scalar))) :
    (m.runListAux h0 history xs).2.length = xs.length := by
  induction xs generalizing h0 history with
  | nil =>
      simp
  | cons x rest ih =>
      simp [runListAux, ih]

/-- The public full Mamba runner emits one output token per input token. -/
theorem runList_outputs_length
    (m : SelectiveMambaBlockSpec α inputDim innerDim stateDim outputDim convWidth)
    (h0 : Tensor α (.dim innerDim (.dim stateDim .scalar)))
    (xs : List (Tensor α (.dim inputDim .scalar))) :
    (m.runList h0 xs).2.length = xs.length := by
  simpa [runList] using m.runListAux_outputs_length h0 [] xs

end SelectiveMambaBlockSpec

end Models
