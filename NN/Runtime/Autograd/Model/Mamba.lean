/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Model.Program

/-!
# Selective Mamba

The Mamba-1 recurrence, expressed through the same differentiable operations used by other
TorchLean layers. A token determines its convolution feature, time step, and input/output state
vectors. The previous hidden state enters only the affine state update.

Parameter matrices use the orientation of `Models.SelectiveMambaBlockSpec`: a row vector multiplies
a matrix on the right. The convolution kernel is newest-first. We store `logA` and compute
`A = exp(logA)` before the recurrence; over the reals this makes every continuous-time rate `-A`
negative throughout training. Floating-point overflow and underflow still follow the chosen
backend's arithmetic.

`step` and `runArray` carry both the hidden state and projected-token history. Keeping references
in that cache lets a continued computation retain gradients through earlier chunks. Callers that
want truncated backpropagation can detach those references explicitly.

This implementation follows the Spec's dense time-step projection. It does not use a fused
variable-coefficient scan or the low-rank time-step parameterization of the authors' default block.
-/

@[expose] public section

namespace Runtime.Autograd.Model.Mamba

open Spec TorchLean TorchLean.Tensor

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
the activated convolution feature directly.
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
  /-- Logarithms of rate magnitudes; transitions use `exp(-delta * exp(logA))`. -/
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
prepends the new projection and retains at most `kernelWidth` entries. Missing entries mean zero
padding; the hidden state has one `stateWidth` vector for each expanded channel.
-/
structure State (T : Shape → Type) (innerWidth stateWidth : Nat) where
  /-- Diagonal state after the most recently processed token. -/
  hidden : T [innerWidth, stateWidth]
  /-- Projected content tokens in newest-first order. -/
  history : Array (T [innerWidth])

variable {α : Type} [Storage α] [Context α]
variable {m : Type → Type} [Monad m] [Ops (m := m) (α := α)]
variable {inputWidth innerWidth stateWidth outputWidth kernelWidth : Nat}

namespace Internal

/-- Right-multiply a vector by a Spec-oriented projection without introducing a bias parameter. -/
def project {input output : Nat}
    (x : Ref (m := m) (α := α) [input])
    (weight : Ref (m := m) (α := α) [input, output]) :
    m (Ref (m := m) (α := α) [output]) := do
  let row ← reshape (m := m) (α := α) (s₂ := [1, input]) x (by simp [Shape.size])
  let y ← matmul (m := m) (α := α) (batchA := []) (batchB := []) (batch := []) row weight
  reshape (m := m) (α := α) (s₂ := [output]) y (by simp [Shape.size])

/--
One step with rate magnitudes already computed.

Sharing `exp(logA)` across a sequence avoids recording the same exponential for every token. Its
reference still participates in each transition, so reverse accumulation sums all rate gradients.
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
  let padding ← const (m := m) (α := α) (Tensor.full [innerWidth] (0 : α))
  let convolution ← (List.finRange kernelWidth).foldlM (init := parameters.convBias)
    fun acc tap => do
      let weight ← select (m := m) (α := α) 0 parameters.convKernel tap
      let term ← mul (m := m) (α := α) weight (history[tap.val]?.getD padding)
      add (m := m) (α := α) acc term
  let u ← silu (m := m) (α := α) convolution
  let deltaLinear ← project u parameters.dtProj
  let deltaPre ← add (m := m) (α := α) deltaLinear parameters.dtBias
  let delta ← softplus (m := m) (α := α) deltaPre
  let b ← project u parameters.bProj
  let c ← project u parameters.cProj
  -- Each channel has its own time step and shares the token's B/C vectors across state coordinates.
  let deltaState ← Torch.broadcastAfterSum (m := m) (α := α) [innerWidth, stateWidth] 1 delta
  let bState ← Torch.broadcastAfterSum (m := m) (α := α) [innerWidth, stateWidth] 0 b
  let uState ← Torch.broadcastAfterSum (m := m) (α := α) [innerWidth, stateWidth] 1 u
  let scaledRates ← mul (m := m) (α := α) deltaState rates
  let negativeRates ← scale (m := m) (α := α) scaledRates (-1 : α)
  let retention ← exp (m := m) (α := α) negativeRates
  let retained ← mul (m := m) (α := α) retention state.hidden
  let deltaB ← mul (m := m) (α := α) deltaState bState
  let written ← mul (m := m) (α := α) deltaB uState
  let hidden ← add (m := m) (α := α) retained written
  -- Contract the updated state against C, then add D*u before the SiLU output gate.
  let cColumn ← reshape (m := m) (α := α) (s₂ := [stateWidth, 1]) c
    (by simp [Shape.size])
  let readoutColumn ← matmul (m := m) (α := α)
    (batchA := []) (batchB := []) (batch := []) hidden cColumn
  let readout ← reshape (m := m) (α := α) (s₂ := [innerWidth]) readoutColumn
    (by simp [Shape.size])
  let skip ← mul (m := m) (α := α) parameters.dSkip u
  let y ← add (m := m) (α := α) skip readout
  let gate ← silu (m := m) (α := α) zPath
  let gated ← mul (m := m) (α := α) y gate
  let output ← project gated parameters.outProj
  pure (⟨hidden, history⟩, output)

end Internal

/-- Start a fresh sequence with zero recurrent state and zero-padded convolution history. -/
def zeroState : m (State (Ref (m := m) (α := α)) innerWidth stateWidth) := do
  let hidden ← const (m := m) (α := α) (Tensor.full [innerWidth, stateWidth] (0 : α))
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
  let rates ← exp (m := m) (α := α) parameters.logA
  Internal.stepWithRates parameters rates state x

/--
Run a chunk and return the complete cache needed by the next chunk, together with its outputs.

An empty chunk preserves both pieces of state. Chunk boundaries introduce no detach operation:
using the returned references in a later chunk keeps gradients through the earlier computation.
-/
def runArray
    (parameters : Parameters (Ref (m := m) (α := α))
      inputWidth innerWidth stateWidth outputWidth kernelWidth)
    (state : State (Ref (m := m) (α := α)) innerWidth stateWidth)
    (xs : Array (Ref (m := m) (α := α) [inputWidth])) :
    m (State (Ref (m := m) (α := α)) innerWidth stateWidth ×
      Array (Ref (m := m) (α := α) [outputWidth])) := do
  if xs.isEmpty then
    return (state, #[])
  let rates ← exp (m := m) (α := α) parameters.logA
  xs.foldlM (init := (state, #[])) fun (previous, outputs) x => do
    let (next, output) ← Internal.stepWithRates parameters rates previous x
    pure (next, outputs.push output)

end Runtime.Autograd.Model.Mamba
