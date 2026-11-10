/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.Program

/-!
# Selective Mamba

The Mamba-1 recurrence, expressed through the same differentiable operations used by other
TorchLean layers. A token determines its convolution feature, time step, and input/output state
vectors. The previous hidden state enters only the affine state update.

Parameter matrices use the orientation of `TorchLean.Mamba.SelectiveBlock`: a row vector multiplies
a matrix on the right. The convolution kernel is newest-first. The parameters store `logA`, and
`A = exp(logA)` is computed before the recurrence, so over the reals every continuous-time rate `-A`
is negative throughout training. Floating-point overflow and underflow follow the chosen backend's
arithmetic.

`step` and `run` carry both the hidden state and projected-token history. Keeping references
in that cache lets a continued computation retain gradients through earlier chunks. Callers that
want truncated backpropagation can detach those references themselves.

The layer runs token by token with the Spec's dense time-step projection. The authors' default
block uses a fused variable-coefficient scan and a low-rank time-step parameterization.

`stepWithRates` is `Mamba.SelectiveBlock.stepWithConvolutionHistory` transcribed by hand, so the
source notes in `TorchLean.NN.Models.Mamba.Block` hold here too, the first-order `delta * B` input
term included. Over the reals the two compute the same step. In floating point the readout can round
differently: the Spec folds `dSkip * u` in before the state terms, and here it's added after the
matmul. No theorem links the two, and no TorchLeanTest test runs this recurrence.
-/

@[expose] public section

namespace TorchLean.Mamba

open TorchLean.Tensor
open Ops

/-- Dimensions inside a selective Mamba layer, independently of its input and output widths. -/
structure Options where
  /-- Expanded channels per output feature: `innerWidth = expansion * outputWidth`. -/
  expansion : Nat := 2
  /-- Number of diagonal recurrent states carried by each expanded channel. -/
  stateWidth : Nat := 16
  /-- Number of newest-first taps in the causal depthwise convolution. -/
  kernelWidth : Nat := 4
deriving Repr, DecidableEq

/-- Reject empty internal axes before a layer allocates parameters or records operations. -/
def Options.validate (options : Options) : Except String Unit := do
  if options.expansion = 0 then
    throw "Mamba: expansion must be positive"
  if options.stateWidth = 0 then
    throw "Mamba: state width must be positive"
  if options.kernelWidth = 0 then
    throw "Mamba: kernel width must be positive"

/--
The eleven trainable tensors, in Spec matrix orientation.

The content, gate, B, C, and output projections have no bias. The convolution and time-step
projection each have a bias. `logA` parameterizes positive rate magnitudes, and `dSkip` multiplies
the activated convolution feature `u` in the readout term `dSkip * u`.
-/
structure Parameters (T : Shape → Type)
    (inputWidth innerWidth stateWidth outputWidth kernelWidth : Nat) where
  /-- Project raw tokens into the content path carried by the convolution cache. -/
  xProj : T [inputWidth, innerWidth]
  /-- Project raw tokens into the SiLU gate, independently of the convolution path. -/
  zProj : T [inputWidth, innerWidth]
  /-- Depthwise taps: row zero multiplies the current projected token. -/
  convKernel : T [kernelWidth, innerWidth]
  /-- Additive channel bias before the content SiLU. -/
  convBias : T [innerWidth]
  /-- Dense map from activated content to per-channel time steps before softplus. -/
  dtProj : T [innerWidth, innerWidth]
  /-- Time-step bias before softplus, stored separately from the dense projection. -/
  dtBias : T [innerWidth]
  /-- Logarithms of rate magnitudes, with transitions `exp(-delta * exp(logA))`. -/
  logA : T [innerWidth, stateWidth]
  /-- Produce a token's B vector, shared by the expanded channels. -/
  bProj : T [innerWidth, stateWidth]
  /-- Produce a token's C vector for reading out the updated state. -/
  cProj : T [innerWidth, stateWidth]
  /-- Per-channel coefficient of the direct `D * u` readout path. -/
  dSkip : T [innerWidth]
  /-- Project gated expanded channels back to the requested output width. -/
  outProj : T [innerWidth, outputWidth]

/--
State at a chunk boundary.

`history[0]` is the most recent projected content token, before convolution and SiLU. A step
prepends the current token's projection and keeps at most `kernelWidth` entries, and missing
entries mean zero padding. The hidden state has one `stateWidth` vector for each expanded channel.
-/
structure State (T : Shape → Type) (innerWidth stateWidth : Nat) where
  /-- Diagonal state after the last processed token. -/
  hidden : T [innerWidth, stateWidth]
  /-- Projected content tokens in newest-first order. -/
  history : Array (T [innerWidth])

variable {α : Type} [Storage α] [Context α]
variable {m : Type → Type} [Monad m] [Ops (m := m) (α := α)]
variable {inputWidth innerWidth stateWidth outputWidth kernelWidth : Nat}

/-- Right-multiply a vector by a Spec-oriented projection without introducing a bias parameter. -/
def project {input output : Nat}
    (x : Ref (m := m) (α := α) [input])
    (weight : Ref (m := m) (α := α) [input, output]) :
    m (Ref (m := m) (α := α) [output]) := do
  let row ← Ops.apply (m := m) (α := α)
      (.reshape (t := [1, input]) (by simp)) (x, ())
  let y : Ref (m := m) (α := α) [1, output] ← Ops.apply (m := m) (α := α)
      (.matmul (batchA := []) (batchB := []) (batch := []) false false (fun h => nomatch h)
        (fun h => nomatch h) Shape.BroadcastTo.proof Shape.BroadcastTo.proof) (row, weight, ())
  Ops.apply (m := m) (α := α)
      (.reshape (t := [output]) (by simp)) (y, ())

/--
One step with rate magnitudes already computed.

Sharing `exp(logA)` across a sequence avoids recording the same exponential for every token. Its
reference participates in every transition, so reverse accumulation sums the rate gradients of all
tokens.
-/
def stepWithRates
    (parameters : Parameters (Ref (m := m) (α := α))
      inputWidth innerWidth stateWidth outputWidth kernelWidth)
    (rates : Ref (m := m) (α := α) [innerWidth, stateWidth])
    (state : State (Ref (m := m) (α := α)) innerWidth stateWidth)
    (x : Ref (m := m) (α := α) [inputWidth]) :
    m (State (Ref (m := m) (α := α)) innerWidth stateWidth ×
      Ref (m := m) (α := α) [outputWidth]) := do
  let xPath ← project x parameters.xProj
  let zPath ← project x parameters.zProj
  let history := (#[xPath] ++ state.history).take kernelWidth
  let padding ← Ops.const (m := m) (α := α) (Tensor.full [innerWidth] (0 : α))
  let convolution ← (List.finRange kernelWidth).foldlM (init := parameters.convBias)
    fun acc tap => do
      let weight ← Ops.apply (m := m) (α := α)
          (.indexAt ⟨0, Shape.AxisInBounds.proof⟩ tap) (parameters.convKernel, ())
      let term ← Ops.apply (m := m) (α := α) .mul
          (weight, (history[tap.val]?.getD padding), ())
      Ops.apply (m := m) (α := α) .add (acc, term, ())
  let u ← Ops.apply (m := m) (α := α) .silu (convolution, ())
  let deltaLinear ← project u parameters.dtProj
  let deltaPre ← Ops.apply (m := m) (α := α) .add
      (deltaLinear, parameters.dtBias, ())
  let delta ← Ops.apply (m := m) (α := α) .softplus (deltaPre, ())
  let b ← project u parameters.bProj
  let c ← project u parameters.cProj
  -- Each channel has its own time step across its state coordinates, and all channels share the
  -- token's B and C vectors.
  let deltaState ← Ops.apply (m := m) (α := α)
      (.broadcast (shapeAfterSumKeepDim_canBroadcastTo [innerWidth, stateWidth] 1).dims
        (shapeAfterSumKeepDim_canBroadcastTo [innerWidth, stateWidth] 1).broadcastDims)
      ((← Ops.apply (m := m) (α := α)
          (.reshape (t := shapeAfterSumKeepDim [innerWidth, stateWidth] 1)
            (size_shapeAfterSumKeepDim [innerWidth, stateWidth] 1).symm) (delta, ())), ())
  let bState ← Ops.apply (m := m) (α := α)
      (.broadcast (shapeAfterSumKeepDim_canBroadcastTo [innerWidth, stateWidth] 0).dims
        (shapeAfterSumKeepDim_canBroadcastTo [innerWidth, stateWidth] 0).broadcastDims)
      ((← Ops.apply (m := m) (α := α)
          (.reshape (t := shapeAfterSumKeepDim [innerWidth, stateWidth] 0)
            (size_shapeAfterSumKeepDim [innerWidth, stateWidth] 0).symm) (b, ())), ())
  let uState ← Ops.apply (m := m) (α := α)
      (.broadcast (shapeAfterSumKeepDim_canBroadcastTo [innerWidth, stateWidth] 1).dims
        (shapeAfterSumKeepDim_canBroadcastTo [innerWidth, stateWidth] 1).broadcastDims)
      ((← Ops.apply (m := m) (α := α)
          (.reshape (t := shapeAfterSumKeepDim [innerWidth, stateWidth] 1)
            (size_shapeAfterSumKeepDim [innerWidth, stateWidth] 1).symm) (u, ())), ())
  let scaledRates ← Ops.apply (m := m) (α := α) .mul (deltaState, rates, ())
  let negativeRates ← Ops.apply (m := m) (α := α) .scale
      (scaledRates, .data
        (Ops.dataConst (m := m) (α := α) (Tensor.scalar (-1 : α))), ())
  let retention ← Ops.apply (m := m) (α := α) .exp (negativeRates, ())
  let retained ← Ops.apply (m := m) (α := α) .mul
      (retention, state.hidden, ())
  let deltaB ← Ops.apply (m := m) (α := α) .mul (deltaState, bState, ())
  let written ← Ops.apply (m := m) (α := α) .mul (deltaB, uState, ())
  let hidden ← Ops.apply (m := m) (α := α) .add (retained, written, ())
  -- Contract the updated state against C, then add D*u before the SiLU output gate.
  let cColumn ← Ops.apply (m := m) (α := α)
      (.reshape (t := [stateWidth, 1]) (by simp)) (c, ())
  let readoutColumn : Ref (m := m) (α := α) [innerWidth, 1] ←
    Ops.apply (m := m) (α := α)
      (.matmul (batchA := []) (batchB := []) (batch := []) false false (fun h => nomatch h)
        (fun h => nomatch h) Shape.BroadcastTo.proof Shape.BroadcastTo.proof) (hidden, cColumn, ())
  let readout ← Ops.apply (m := m) (α := α)
      (.reshape (t := [innerWidth]) (by simp)) (readoutColumn, ())
  let skip ← Ops.apply (m := m) (α := α) .mul (parameters.dSkip, u, ())
  let y ← Ops.apply (m := m) (α := α) .add (skip, readout, ())
  let gate ← Ops.apply (m := m) (α := α) .silu (zPath, ())
  let gated ← Ops.apply (m := m) (α := α) .mul (y, gate, ())
  let output ← project gated parameters.outProj
  pure (⟨hidden, history⟩, output)

/-- Start a fresh sequence with zero recurrent state and zero-padded convolution history. -/
def zeroState : m (State (Ref (m := m) (α := α)) innerWidth stateWidth) := do
  let hidden ← Ops.const (m := m) (α := α) (Tensor.full [innerWidth, stateWidth] (0 : α))
  pure ⟨hidden, #[]⟩

/--
Advance one token using the selective Mamba-1 equations.

With `u = SiLU(causalConv(x @ xProj))`, the update is
`h'[d,n] = exp(-softplus(u @ dtProj + dtBias)[d] * exp(logA[d,n])) * h[d,n]
          + (delta[d] * (u @ bProj)[n]) * u[d]`.
The readout contracts `h'` with `u @ cProj`, adds `dSkip * u`, gates by `SiLU(x @ zProj)`,
and applies `outProj`.
-/
def step
    (parameters : Parameters (Ref (m := m) (α := α))
      inputWidth innerWidth stateWidth outputWidth kernelWidth)
    (state : State (Ref (m := m) (α := α)) innerWidth stateWidth)
    (x : Ref (m := m) (α := α) [inputWidth]) :
    m (State (Ref (m := m) (α := α)) innerWidth stateWidth ×
      Ref (m := m) (α := α) [outputWidth]) := do
  let rates ← Ops.apply (m := m) (α := α) .exp (parameters.logA, ())
  stepWithRates parameters rates state x

/--
Run a chunk and return the complete cache needed by the next chunk, together with its outputs.

An empty chunk preserves both pieces of state. Chunk boundaries introduce no detach operation:
using the returned references in a later chunk keeps gradients through the earlier computation.
-/
def run
    (parameters : Parameters (Ref (m := m) (α := α))
      inputWidth innerWidth stateWidth outputWidth kernelWidth)
    (state : State (Ref (m := m) (α := α)) innerWidth stateWidth)
    (xs : Array (Ref (m := m) (α := α) [inputWidth])) :
    m (State (Ref (m := m) (α := α)) innerWidth stateWidth ×
      Array (Ref (m := m) (α := α) [outputWidth])) := do
  if xs.isEmpty then
    return (state, #[])
  let rates ← Ops.apply (m := m) (α := α) .exp (parameters.logA, ())
  xs.foldlM (init := (state, #[])) fun (previous, outputs) x => do
    let (next, output) ← stepWithRates parameters rates previous x
    pure (next, outputs.push output)

end TorchLean.Mamba
