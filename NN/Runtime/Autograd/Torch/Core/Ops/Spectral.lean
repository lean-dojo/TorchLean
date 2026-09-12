/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Torch.Core.Ops.Dispatch
public import NN.Runtime.Autograd.Engine.Cuda.Ops.Fourier
public import NN.Runtime.Autograd.Engine.Cuda.Ops.SelectiveScan
public import NN.Runtime.Autograd.Engine.Cuda.Ops.Linear

/-!
# Native hooks for Fourier transforms and diagonal scans

CPU execution declines these hooks before recording anything, allowing the generic differentiable
program to run on its ordinary operations. CUDA execution validates reference identities, selects
the corresponding backend capsule, and records a native node. This keeps provider selection and
cross-session checks on the same path as the other eager operations.
-/

@[expose] public section

namespace Runtime.Autograd.Torch.Internal.EagerSession

open Spec TorchLean

/-- Run a native tape constructor only for a CUDA session, preserving its reference identity. -/
def spectralNative? {α : Type} [Storage α] {shape : Shape} (s : EagerSession α)
    (op : NN.Backend.BackendOp) (refs : Array (Option RefIdentity))
    (record : Cuda.Tape → Result (Cuda.Tape × Nat)) : IO (Option (TensorRef α shape)) := do
  if Config.device s.options != .cuda then return none
  let cpu : IO (TensorRef α shape) :=
    throw <| IO.userError "torch: native spectral hook requires a CUDA session"
  let cuda := do
    let (tape, id) ← okOrThrow (record (← s.cudaTape.get))
    s.cudaTape.set tape
    return some { id := id }
  return some (← dispatchCudaOpt s op refs cpu cuda)

/-- Native real-transform hook; CPU interpreters use the generic matrix reference. -/
def rfft1dNative? {α : Type} [Storage α] (s : EagerSession α) {batch n : Nat}
    (x : TensorRef α [batch, n]) : IO (Option (TensorRef α [batch, n / 2 + 1, 2])) :=
  spectralNative? s .fftFno #[x.identity?] fun tape =>
    Cuda.Tape.rfft1d (batch := batch) (n := n) tape x.id

/-- Native normalized inverse hook with explicit output length. -/
def irfft1dNative? {α : Type} [Storage α] (s : EagerSession α) {batch n : Nat}
    (x : TensorRef α [batch, n / 2 + 1, 2]) : IO (Option (TensorRef α [batch, n])) :=
  spectralNative? s .fftFno #[x.identity?] fun tape =>
    Cuda.Tape.irfft1d (batch := batch) (n := n) tape x.id

/-- Native shared-coefficient scan hook with gradients for coefficients, input, and state. -/
def selectiveScanDiagNative? {α : Type} [Storage α] (s : EagerSession α) {seqLen state : Nat}
    (a b : TensorRef α [state]) (x : TensorRef α [seqLen, state])
    (initial : TensorRef α [state]) : IO (Option (TensorRef α [seqLen, state])) :=
  spectralNative? s .selectiveScan #[a.identity?, b.identity?, x.identity?, initial.identity?]
    fun tape => Cuda.Tape.selectiveScanDiag (seqLen := seqLen) (state := state)
      tape a.id b.id x.id initial.id

/-- Native token-dependent scan hook, including coefficient and continuation-state gradients. -/
def selectiveScanDiagVarNative? {α : Type} [Storage α] (s : EagerSession α) {seqLen state : Nat}
    (a b x : TensorRef α [seqLen, state]) (initial : TensorRef α [state]) :
    IO (Option (TensorRef α [seqLen, state])) :=
  spectralNative? s .selectiveScan #[a.identity?, b.identity?, x.identity?, initial.identity?]
    fun tape => Cuda.Tape.selectiveScanDiagVar (seqLen := seqLen) (state := state)
      tape a.id b.id x.id initial.id

/-- Native one-sided spectral convolution with the same parameter layout as its reference. -/
def spectralConv1dRfftNative? {α : Type} [Storage α] (s : EagerSession α)
    {grid width modes : Nat} (x : TensorRef α [grid, width])
    (realWeight imagWeight : TensorRef α [modes, width, width]) :
    IO (Option (TensorRef α [grid, width])) :=
  spectralNative? s .fftFno #[x.identity?, realWeight.identity?, imagWeight.identity?]
    fun tape => Cuda.Tape.Internal.spectralConv1dRfft (grid := grid) (width := width)
      (modes := modes) tape x.id realWeight.id imagWeight.id

end Runtime.Autograd.Torch.Internal.EagerSession
