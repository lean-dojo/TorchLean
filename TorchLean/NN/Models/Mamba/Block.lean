/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.StateSpace
public import TorchLean.Tensor.Layers.Activation
public import TorchLean.Tensor.Contract

/-!
# Mamba-style selective state-space blocks

Mamba replaces quadratic attention with a linear-time selective state-space recurrence.  In full
models, the token controls discretization and input/output state parameters.

The module has two blocks:

- `Mamba.CompactBlock`: a compact theorem-friendly diagonal SSM block for scan laws.
- `Mamba.SelectiveBlock`: a fuller Mamba-style block with input/gate projections, causal
  depthwise convolution, SiLU, token-dependent `Delta/B/C`, diagonal selective scan, gated output,
  and output projection.

The compact block is the smallest reusable core for proving scan algebra. It consists of:

- a recurrent selective scan (`h ← A ⊙ h + B ⊙ x_state`),
- a gated state readout,
- tokenwise input/output projections.

The full block builds the paper-style Mamba dataflow on top of the same affine-scan idea.

We wrote `SelectiveBlock` by hand from Algorithm 2 and Figure 3 of Gu and Dao (2024). `Delta` is
`softplus` of a bias plus a projection (Algorithm 2, line 4), `B_t` and `C_t` are the `Linear_N`
maps of their Section 3.2, and SiLU acts on both branches as in Figure 3. Our `A` holds the
positive rates `−A` of the paper. The transition `exp(-Delta * A)` is the zero-order hold of their
eq. (4), but the input term is the first-order `Delta * B`, the `dB` of the authors'
`mamba_simple.py`, where eq. (4) multiplies it by `(ΔA)⁻¹(exp(ΔA) − I)`. At `Delta = 0.01` that
factor is about `0.995` for rate `1` and `0.924` for rate `16`. The `dSkip * u` readout term is the
same code's `D * x`, which eq. (2b) leaves out. `dtProj` is Section 3.6's low-rank
`Linear_D(Linear_R(x))` taken at full rank.

## Implementation status

`nn.mamba` uses the trainable selective block in
`TorchLean/NN/Layer/Mamba.lean`. The recurrence in
`TorchLean/NN/Models/Mamba/Recurrence.lean` follows the
`Mamba.SelectiveBlock` dataflow through generic differentiable operations: causal depthwise
convolution and SiLU produce the feature used for softplus time steps and token-dependent `B`/`C`,
then the diagonal state update feeds the skip connection, SiLU gate, and output projection.
The runtime stores `logA`, so this Spec's rate tensor corresponds to `exp(logA)`. The runtime uses
the dense `dtProj` time-step projection and calls no fused variable-coefficient scan.

The layer has eleven trainable parameter tensors. Its expanded channel count is
`innerWidth = expansion * hiddenWidth`, where `hiddenWidth` is the output feature width.
Expansion, state width, and convolution width determine the saved tensor shapes. Each layer call
starts with zero recurrent state and empty convolution history, and
`TorchLean.Mamba.run` accepts and returns both a state tensor of shape
`[innerWidth, stateWidth]` and the newest-first projected-token history.

The causality (prefix-preservation) theorems in
`TorchLean/NN/StateSpace/Causality.lean` concern `Mamba.CompactBlock.runArray`,
`Mamba.SelectiveBlock.runArray`, and `Mamba.SelectiveBlock.runArrayWithHistory`, built on
the scan algebra in `TorchLean/NN/StateSpace/Scan.lean`. No theorem relates the generic
differentiable runtime to these Spec runners or proves its gradients, and no test compares either
block with PyTorch.

## Importing reference checkpoints

- `convKernel` is indexed `(tap, channel)` with taps newest-first. PyTorch's depthwise `conv1d`
  weight is `(channel, 1, tap)` with taps oldest-first, so import needs a transpose and a reversal.
- `dtProj` is one square `innerDim × innerDim` map. The reference model factors it through
  `dt_rank` (`x_proj` then `dt_proj`), and the two factors must be multiplied out before loading.
- `bProj` and `cProj` are separate `innerDim × stateDim` matrices, while the reference model
  packs `B`, `C`, and the `dt` input into one `x_proj` output.

References:
- Albert Gu and Tri Dao (2024), "Mamba: Linear-Time Sequence Modeling with Selective State
  Spaces", COLM 2024, arXiv:2312.00752, eqs. (2a), (2b) and (4), Algorithm 2, Sections 3.2 and
  3.6, Figure 3.
- `mamba_ssm/modules/mamba_simple.py` in github.com/state-spaces/mamba, the `Mamba.step` update.
- Tri Dao and Albert Gu (2024), "Transformers are SSMs: Generalized Models and Efficient
  Algorithms Through Structured State Space Duality" (Mamba-2), ICML 2024, arXiv:2405.21060.
-/

@[expose] public section

open FloatLib.Numerics (MathFunctions)

namespace TorchLean

open Tensor
open Dynamics

/-- Parameters for a compact diagonal Mamba-style block. -/
structure Mamba.CompactBlock (α : Type) [Storage α]
    (inputDim stateDim outputDim : Nat) where
  /-- Input projection into SSM state channels. -/
  inProj : Tensor α [inputDim, stateDim]
  /-- Gate projection. The gate is `sigmoid(x @ gateProj)`. -/
  gateProj : Tensor α [inputDim, stateDim]
  /-- Output projection from gated state channels. -/
  outProj : Tensor α [stateDim, outputDim]
  /-- Diagonal state-space core. -/
  ssm : DiagonalSSM α stateDim

namespace Mamba.CompactBlock

variable {α : Type} [Storage α] [Context α]
variable {inputDim stateDim outputDim : Nat}

/-- Token-dependent sigmoid gate. -/
def gate (m : CompactBlock α inputDim stateDim outputDim)
    (x : Tensor α [inputDim]) : Tensor α [stateDim] :=
  Tensor.map Context.sigmoid (Tensor.contract x m.gateProj _ _ [stateDim] .vecMat)

/-- One Mamba-style token step, returning `(new_state, output)`. -/
def step (m : CompactBlock α inputDim stateDim outputDim)
    (h : Tensor α [stateDim])
    (x : Tensor α [inputDim]) :
    Tensor α [stateDim] × Tensor α [outputDim] :=
  let xState := Tensor.contract x m.inProj _ _ [stateDim] .vecMat
  let h' := m.ssm.step h xState
  let yState := m.ssm.readout h' xState
  let gated := yState * m.gate x
  (h', Tensor.contract gated m.outProj _ _ [outputDim] .vecMat)

/-- Run an array of tokens through the recurrent block. -/
def runArray (m : CompactBlock α inputDim stateDim outputDim)
    (h0 : Tensor α [stateDim])
    (xs : Array (Tensor α [inputDim])) :
    Tensor α [stateDim] × Array (Tensor α [outputDim]) :=
  StateSpace.scanArray m.step h0 xs

/-- An empty token sequence leaves the hidden state untouched and emits nothing. -/
@[simp] theorem runArray_empty (m : CompactBlock α inputDim stateDim outputDim)
    (h0 : Tensor α [stateDim]) :
    m.runArray h0 #[] = (h0, #[]) := by
  rfl

/-- A Mamba recurrent pass emits one output token per input token. -/
@[simp] theorem runArray_outputs_size (m : CompactBlock α inputDim stateDim outputDim)
    (h0 : Tensor α [stateDim])
    (xs : Array (Tensor α [inputDim])) :
    (m.runArray h0 xs).2.size = xs.size := by
  exact StateSpace.scanArray_outputs_size m.step h0 xs

end Mamba.CompactBlock

/-- Parameters for a fuller Mamba-style selective SSM block.

Shape conventions:
- `inputDim`: token/input feature width,
- `innerDim`: expanded channel width used by Mamba's convolution and SSM path,
- `stateDim`: per-channel diagonal SSM state size,
- `outputDim`: output feature width,
- `convWidth`: causal depthwise-convolution width.

The recurrence state has shape `[innerDim, stateDim]`, as in the common implementation view of
Mamba where each expanded channel carries a small diagonal state vector.
-/
structure Mamba.SelectiveBlock
    (α : Type) [Storage α]
    (inputDim innerDim stateDim outputDim convWidth : Nat) where
  /-- Content/input projection `x -> x_path`. -/
  xProj : Tensor α [inputDim, innerDim]
  /-- Gate projection `x -> z_path`. -/
  zProj : Tensor α [inputDim, innerDim]
  /-- Causal depthwise-convolution kernel, indexed by `(tap, channel)` with tap `0` applied to the
  current token and tap `t` to the token `t` steps back. PyTorch's `conv1d` weight of shape
  `(innerDim, 1, convWidth)` stores taps oldest-first along its last axis, so importing a checkpoint
  requires transposing to `(tap, channel)` and reversing the tap axis. -/
  convKernel : Tensor α [convWidth, innerDim]
  /-- Causal depthwise-convolution bias. -/
  convBias : Tensor α [innerDim]
  /-- Projection from activated convolution features to per-channel time steps `Delta`.

  It is one square map, which the reference implementation factors through a low-rank bottleneck:
  `x_proj` (`innerDim -> dt_rank`) followed by `dt_proj` (`dt_rank -> innerDim`). The product of
  those two matrices can be loaded here, and the factorization itself is not modelled. -/
  dtProj : Tensor α [innerDim, innerDim]
  /-- Bias before the `softplus` time-step nonlinearity. -/
  dtBias : Tensor α [innerDim]
  /-- Positive diagonal state rates `A[d,n]` used as `exp(-Delta[d] * A[d,n])`. -/
  A : Tensor α [innerDim, stateDim]
  /-- Token-dependent input-state projection `B_t = u_t @ bProj`. -/
  bProj : Tensor α [innerDim, stateDim]
  /-- Token-dependent state-output projection `C_t = u_t @ cProj`. -/
  cProj : Tensor α [innerDim, stateDim]
  /-- Per-channel residual/skip coefficient. -/
  dSkip : Tensor α [innerDim]
  /-- Output projection from expanded channels to output features. -/
  outProj : Tensor α [innerDim, outputDim]

namespace Mamba.SelectiveBlock

variable {α : Type} [Storage α] [Context α]
variable {inputDim innerDim stateDim outputDim convWidth : Nat}

/--
Causal depthwise convolution from a newest-first history of projected tokens.

`history[0]` is the current projected token, `history[1]` is the previous token, etc.  Missing
history entries are treated as zero padding. Channel `c` of the result is
`convBias[c] + Σ_tap history[tap][c] * convKernel[tap, c]`, summed in tap order.
-/
def causalDepthwiseConv
    (m : SelectiveBlock α inputDim innerDim stateDim outputDim convWidth)
    (history : Array (Tensor α [innerDim])) :
    Tensor α [innerDim] :=
  Tensor.dim (fun c : Fin innerDim =>
    Tensor.scalar <|
      (List.finRange convWidth).foldl
        (fun acc tap =>
          let xTap : α :=
            match history[tap.val]? with
            | some token => token.getScalar c
            | none => 0
          acc + xTap * get2 m.convKernel tap c)
        (Tensor.getScalar m.convBias c))

/-- Token-dependent positive time steps `Delta = softplus(u @ dtProj + dtBias)`. -/
def delta
    (m : SelectiveBlock α inputDim innerDim stateDim outputDim convWidth)
    (u : Tensor α [innerDim]) : Tensor α [innerDim] :=
  Tensor.map Activation.softplus
      (Tensor.contract u m.dtProj _ _ [innerDim] .vecMat + m.dtBias)

/--
One selective diagonal SSM update:

`h'[d,n] = exp(-Delta[d] * A[d,n]) * h[d,n] + (Delta[d] * B_t[n]) * u[d]`.
-/
def selectiveStateStep
    (m : SelectiveBlock α inputDim innerDim stateDim outputDim convWidth)
    (h : Tensor α [innerDim, stateDim])
    (u : Tensor α [innerDim]) :
    Tensor α [innerDim, stateDim] :=
  let Δ := m.delta u
  let B := Tensor.contract u m.bProj _ _ [stateDim] .vecMat
  Tensor.dim (fun d : Fin innerDim =>
    Tensor.dim (fun n : Fin stateDim =>
      let deltaD := Tensor.getScalar Δ d
      let aBar := MathFunctions.exp (-(deltaD * get2 m.A d n))
      let bBar := deltaD * Tensor.getScalar B n
      Tensor.scalar (aBar * get2 h d n + bBar * Tensor.getScalar u d)))

/-- Read out expanded channels from the updated state using `C_t`, plus the Mamba skip path:
`y[d] = dSkip[d] * u[d] + Σ_n h[d,n] * C_t[n]`. -/
def stateReadout
    (m : SelectiveBlock α inputDim innerDim stateDim outputDim convWidth)
    (h : Tensor α [innerDim, stateDim])
    (u : Tensor α [innerDim]) :
    Tensor α [innerDim] :=
  let C := Tensor.contract u m.cProj _ _ [stateDim] .vecMat
  Tensor.dim (fun d : Fin innerDim =>
    Tensor.scalar <|
      (List.finRange stateDim).foldl
        (fun acc n => acc + get2 h d n * Tensor.getScalar C n)
        (Tensor.getScalar m.dSkip d * Tensor.getScalar u d))

/--
One full Mamba token step from an already-updated convolution history.

The `history` argument is newest-first and must include the current projected content token.
-/
def stepWithHistory
    (m : SelectiveBlock α inputDim innerDim stateDim outputDim convWidth)
    (h : Tensor α [innerDim, stateDim])
    (history : Array (Tensor α [innerDim]))
    (z : Tensor α [innerDim]) :
    Tensor α [innerDim, stateDim] × Tensor α [outputDim] :=
  let u := Tensor.map Activation.silu (m.causalDepthwiseConv history)
  let h' := m.selectiveStateStep h u
  let y := m.stateReadout h' u
  let gated := y * Tensor.map Activation.silu z
  (h', Tensor.contract gated m.outProj _ _ [outputDim] .vecMat)

/-- One recurrent step while carrying the newest-first convolution history.

The carried history is truncated to the `convWidth` most recent projected tokens. Entries further
back are never read by `causalDepthwiseConv`, so dropping them changes no output and keeps the
state size bounded over arbitrarily long sequences. -/
def stepWithConvolutionHistory
    (m : SelectiveBlock α inputDim innerDim stateDim outputDim convWidth)
    (state : Tensor α [innerDim, stateDim] ×
      Array (Tensor α [innerDim]))
    (x : Tensor α [inputDim]) :
    (Tensor α [innerDim, stateDim] ×
      Array (Tensor α [innerDim])) ×
      Tensor α [outputDim] :=
  let xPath := Tensor.contract x m.xProj _ _ [innerDim] .vecMat
  let zPath := Tensor.contract x m.zProj _ _ [innerDim] .vecMat
  let history := (#[xPath] ++ state.2).take convWidth
  let (nextHidden, output) := m.stepWithHistory state.1 history zPath
  ((nextHidden, history), output)

/-- Recurrent runner from an existing state and newest-first convolution history. -/
def runArrayWithHistory
    (m : SelectiveBlock α inputDim innerDim stateDim outputDim convWidth)
    (h0 : Tensor α [innerDim, stateDim])
    (history : Array (Tensor α [innerDim]))
    (xs : Array (Tensor α [inputDim])) :
    Tensor α [innerDim, stateDim] ×
      Array (Tensor α [outputDim]) :=
  let result := StateSpace.scanArray m.stepWithConvolutionHistory (h0, history) xs
  (result.1.1, result.2)

/-- Run a sequence through the full selective Mamba block. -/
def runArray
    (m : SelectiveBlock α inputDim innerDim stateDim outputDim convWidth)
    (h0 : Tensor α [innerDim, stateDim])
    (xs : Array (Tensor α [inputDim])) :
    Tensor α [innerDim, stateDim] ×
      Array (Tensor α [outputDim]) :=
  m.runArrayWithHistory h0 #[] xs

/-- With no tokens the convolution history is never consulted and the state is returned as is. -/
@[simp] theorem runArrayWithHistory_empty
    (m : SelectiveBlock α inputDim innerDim stateDim outputDim convWidth)
    (h0 : Tensor α [innerDim, stateDim])
    (history : Array (Tensor α [innerDim])) :
    m.runArrayWithHistory h0 history #[] = (h0, #[]) := by
  rfl

/-- An empty token sequence leaves the selective block's state unchanged and emits nothing. -/
@[simp] theorem runArray_empty
    (m : SelectiveBlock α inputDim innerDim stateDim outputDim convWidth)
    (h0 : Tensor α [innerDim, stateDim]) :
    m.runArray h0 #[] = (h0, #[]) := by
  rfl

/-- The full Mamba recurrent pass emits one output token per input token. -/
@[simp] theorem runArrayWithHistory_outputs_size
    (m : SelectiveBlock α inputDim innerDim stateDim outputDim convWidth)
    (h0 : Tensor α [innerDim, stateDim])
    (history : Array (Tensor α [innerDim]))
    (xs : Array (Tensor α [inputDim])) :
    (m.runArrayWithHistory h0 history xs).2.size = xs.size := by
  exact StateSpace.scanArray_outputs_size _ (h0, history) xs

/-- The public full Mamba runner emits one output token per input token. -/
@[simp] theorem runArray_outputs_size
    (m : SelectiveBlock α inputDim innerDim stateDim outputDim convWidth)
    (h0 : Tensor α [innerDim, stateDim])
    (xs : Array (Tensor α [inputDim])) :
    (m.runArray h0 xs).2.size = xs.size := by
  unfold runArray
  exact m.runArrayWithHistory_outputs_size h0 #[] xs

end Mamba.SelectiveBlock

end TorchLean
