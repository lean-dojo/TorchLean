/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Compiled.Core
public import NN.Runtime.Autograd.Compiled.GraphM
public import NN.Runtime.Autograd.Engine.FastKernels
public import NN.Runtime.Autograd.Engine.Cuda.Tape
public import NN.Runtime.Autograd.Engine.Cuda.Kernels
public import NN.Runtime.Autograd.Engine.Cuda.ConvPool
public import NN.Runtime.Autograd.Engine.Cuda.Ops
public import NN.Runtime.Autograd.Engine.Cuda.Shape
import Batteries.Data.Vector.Lemmas
import Init.Data.Vector.Lemmas
import Mathlib.Algebra.Order.Algebra

/-!
# Torch Runtime Types

Public handles and options for the Torch-style front-end. This file contains no eager
operation implementations; it defines the objects the other runtime modules share.
-/


@[expose] public section

namespace Runtime
namespace Autograd
namespace Torch

open Spec
open Tensor
open Proofs.Autograd.Algebra

/--
`TList` is the dependently-typed heterogeneous tensor list used by the proved IR.

We re-export it here under the `Torch` front-end namespace because many user-facing helpers
(trainer APIs, parameter packs, etc.) are naturally expressed as `TList`s.
-/
abbrev TList (α : Type) (ss : List Shape) := Proofs.Autograd.Algebra.TList α ss

/--
Execution backend for the Torch-style front-end.

- `.eager`: build and execute a runtime tape directly (imperative, PyTorch-like).
- `.compiled`: record typed IR and run a compiled tape (proof-friendly path, see
  `Torch.LinkedSession` / `TorchLean.Session`).

This is not a CUDA Graph selector. CUDA is controlled by `Options.device` on the
eager backend; CUDA Graph capture/replay will require a distinct persistent-buffer backend.
-/
inductive Backend where
  | eager
  | compiled
deriving Repr, DecidableEq

/--
Execution device selector (PyTorch comparison: `cpu` vs `cuda`).

Current scope:
- eager backend only: selects whether the hidden tape is CPU (`Runtime.Autograd.Tape`) or CUDA
  (`Runtime.Autograd.Cuda.Tape`),
- compiled backend is unchanged (proof semantics / typed IR are device-agnostic).
-/
inductive Device where
  | cpu
  | cuda
deriving Repr, DecidableEq

/--
Options controlling the behavior of the Torch-style front-end.

PyTorch comparison: these are approximately "session/global" settings (e.g. default `requires_grad`,
and runtime-only performance toggles).
-/
structure Options where
  /-- Execution backend selection. -/
  backend : Backend := .eager
  /-- Default `requires_grad` value for newly created parameters/inputs when a caller omits it. -/
  requiresGradByDefault : Bool := true
  /--
  Global deterministic seed for runtime randomness.

  TorchLean keeps the semantic core pure and seed-threaded (JAX-style), so this value is the
  runtime seed that user code can thread into:
  - model initialization (per-layer init keys),
  - dataset shuffles / sampling,
  - and session-level RNG state (dropout, etc.).

  PyTorch analogue: `torch.manual_seed(seed)`.
  -/
  seed : Nat := 0
  /--
  Enable runtime-only fast kernels for a few hot ops in the eager backend.

  This is an execution/performance flag; it is not used by the proof-linked compilation path.
  -/
  fastKernels : Bool := false
  /--
  GPU precision for fast-kernel matmul over Lean `Float` tensors.

  `.fp32` matches the eager CUDA buffer stack. `.fp64` selects the double-precision DGEMM path for
  matmul-only `Float` workloads that request double precision on GPU.
  -/
  fastGpuMatmulPrecision : Runtime.Autograd.FastKernels.GpuMatmulPrecision := .fp32
  /--
  Skip autograd during backward (inference-style evaluation).

  Used by `eval1NoGrad` and similar helpers that run forward without recording gradients.
  -/
  noGrad : Bool := false
  /--
  Eager execution on CUDA.

  When `true` and `backend = .eager`, the eager session uses the CUDA tape
  (`Runtime.Autograd.Cuda.Tape`) and Torch ops must route to CUDA implementations (no implicit CPU
  fallback).

  Compiled backend behavior is unchanged.
  -/
  useGpu : Bool := false
deriving Repr

/- Convenience API for PyTorch-style device selection. -/
namespace Options

/-- Read the device selector corresponding to `useGpu`. -/
def device (opts : Options) : Device :=
  if opts.useGpu then .cuda else .cpu

/-- Return a copy of the options that selects the requested execution device. -/
def toDevice (opts : Options) (d : Device) : Options :=
  { opts with useGpu := d == .cuda }

end Options

/--
Opaque handle to a tensor value in the current session/tape.

This is the TorchLean analogue of a PyTorch `Tensor` object whose "identity" is a node/leaf id in
  the
autograd tape. The phantom shape index `s` makes shape mismatches explicit at compile time.
-/
structure TensorRef (α : Type) (s : Shape) where
  /-- Node/leaf identifier in the owning session tape. -/
  id : Nat
deriving Repr

/--
Handle to a `Nat` stored in the session's non-differentiable environment.

This is used to model index-like inputs (class labels, gather indices, etc.) which should not
receive gradients.
-/
structure NatRef where
  /-- Index into the session's non-differentiable `Nat` environment. -/
  id : Nat
deriving Repr

/--
Handle to a contiguous block of `k` `Nat`s in the session's non-differentiable environment.
-/
structure NatVecRef (k : Nat) where
  /-- Start offset of the contiguous `k`-element block in the session's `Nat` environment. -/
  start : Nat
deriving Repr

/--
Trainable parameter: a mutable tensor value plus metadata.

PyTorch comparison: analogous to `torch.nn.Parameter`, except the parameter becomes part of the
autograd graph only when you `use` it in a particular session/tape.
-/
structure Param (α : Type) (s : Shape) where
  /-- Optional user-facing name for logging/debugging. -/
  name : Option String := none
  /-- Value at the current point. -/
  value : IO.Ref (Tensor α s)
  /--
  Optional CUDA-resident mirror of `value`.

  The eager CUDA trainer uses this as a persistent-parameter cache: repeated forward
  passes can reuse the device buffer instead of uploading the host tensor every step.  The host
  `value` remains the public source for CPU runs and exact/symbolic scalar instantiations.  When a
  caller explicitly reads parameters after CUDA training, the device mirror is copied back here.
  -/
  cudaValue : IO.Ref (Option Runtime.Autograd.Cuda.AnyBuffer)
  /--
  Whether `value` is known to match `cudaValue`.

  CUDA optimizer steps mark this `false` after updating only the device mirror.  Public parameter
  readback synchronizes and flips it back to `true`.
  -/
  hostCurrent : IO.Ref Bool
  /-- Whether this parameter receives accumulated gradients and optimizer updates. -/
  requiresGrad : Bool := true

/--
Type-erased parameter wrapper.

This exists so session code can store heterogeneous parameter shapes in a single `HashMap` keyed
by leaf id (used for SGD updates).
-/
structure AnyParam (α : Type) where
  /-- Runtime shape of the erased parameter. -/
  s : Shape
  /-- Whether the underlying parameter receives optimizer updates. -/
  requiresGrad : Bool
  /-- Read the current parameter value with its runtime shape. -/
  get : IO (Runtime.AnyTensor α)
  /-- Overwrite the current parameter value, checking shape at the call site. -/
  set : Runtime.AnyTensor α → IO Unit
  /-- Store a CUDA buffer mirror without forcing an immediate host download. -/
  setCuda : Runtime.Autograd.Cuda.AnyBuffer → IO Unit

namespace AnyParam

/--
Make the result of a native CUDA cleanup call observable in `IO`.

Some CUDA cleanup functions are exposed as pure opaque calls because they are also useful in pure
buffer-building expressions. In executable cleanup paths, we still want the native call to be
sequenced with the surrounding eager-session updates. Branching into `IO` on the returned flag gives
Lean a real dependency on the result without printing anything or changing behavior.
-/
def observeCudaCleanupFlag (released : UInt32) : IO Unit :=
  if released == 0 then
    IO.sleep 0
  else
    pure ()

/-- Release a cached CUDA mirror, if one exists.

CUDA buffers are external objects whose native finalizer tolerates repeated cleanup attempts, but
parameter updates know exactly when an old device mirror is no longer the current value. Releasing
that mirror here keeps eager CUDA sessions explicit about ownership.
-/
def releaseCachedCudaValue {α : Type} {s : Shape} (p : Param α s) : IO Unit := do
  match ← p.cudaValue.get with
  | none => pure ()
  | some any =>
      let released := Runtime.Autograd.Cuda.Buffer.release any.buf
      observeCudaCleanupFlag released

/--
Package a typed `Param α s` as an `AnyParam α`, checking shape on `set`.

This is the bridge that allows generic optimizers/update routines to operate over heterogeneous
parameter packs.
-/
def ofParam {α : Type} {s : Shape} (p : Param α s) : AnyParam α :=
  { s := s
    requiresGrad := p.requiresGrad
    get := do
      let v ← p.value.get
      pure (Runtime.Autograd.AnyTensor.mk v)
    set := fun v => do
      if h : v.s = s then
        releaseCachedCudaValue p
        p.value.set (Tensor.castShape v.t h)
        p.cudaValue.set none
        p.hostCurrent.set true
      else
        throw <| IO.userError
          s!"torch: param update shape mismatch (expected {Shape.pretty s}, got {Shape.pretty v.s})"
    setCuda := fun v => do
      if _h : v.s = s then
        releaseCachedCudaValue p
        p.cudaValue.set (some { s := s, buf := v.buf })
        p.hostCurrent.set false
      else
        throw <| IO.userError
          s!"torch: CUDA param update shape mismatch (expected {Shape.pretty s}, got {Shape.pretty v.s})"
          }

end AnyParam

/-- Convenience: throw `IO.userError` on a `.error` result. -/
abbrev okOrThrow {α : Type} : Runtime.Autograd.Result α → IO α :=
  Runtime.Autograd.okOrThrow
end Torch
end Autograd
end Runtime
