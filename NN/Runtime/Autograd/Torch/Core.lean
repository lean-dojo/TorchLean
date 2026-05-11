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
# Torch Core

PyTorch-style imperative front-end (eager runtime).

This wraps the eager runtime tape (`Runtime.Autograd.Tape`) behind an `IO.Ref` so user code can
look closer to PyTorch:
- create parameters and inputs as objects;
- call ops as methods (no explicit tape threading);
- run `backward` and optionally apply SGD updates.

This is purely a convenience layer; correctness/proof connections live elsewhere (e.g.
`Runtime.Autograd.Compiled` / `Proofs.Autograd.Algebra.Graph`).

References:
- PyTorch `torch.autograd` docs: https://pytorch.org/docs/stable/autograd.html
- PyTorch "Autograd mechanics": https://pytorch.org/docs/stable/notes/autograd.html
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

This is intentionally not a CUDA Graph selector. CUDA is controlled by `Options.device` on the
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

PyTorch comparison: these are roughly "session/global" settings (e.g. default `requires_grad`,
and runtime-only performance toggles).
-/
structure Options where
  /-- Execution backend selection. -/
  backend : Backend := .eager
  /-- Default `requires_grad` value for newly created parameters/inputs when a caller omits it. -/
  requiresGradByDefault : Bool := true
  /--
  Global deterministic seed for demo-style randomness.

  TorchLean keeps the semantic core pure and seed-threaded (JAX-style), so this is best understood
  as a *convenient default seed knob* that user code can thread into:
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
  matmul-only `Float` workloads that intentionally want double precision on GPU.
  -/
  fastGpuMatmulPrecision : Runtime.Autograd.FastKernels.GpuMatmulPrecision := .fp32
  /--
  Eager execution on CUDA.

  When `true` and `backend = .eager`, the eager session uses the CUDA tape
  (`Runtime.Autograd.Cuda.Tape`) and Torch ops must route to CUDA implementations (no implicit CPU
  fallback).

  Compiled backend behavior is unchanged.
  -/
  useGpu : Bool := false
  /--
  Strict CUDA mode (eager backend only).

  Note: in CUDA eager mode TorchLean does not fall back to CPU per-op; missing CUDA ops always
  throw. This flag is retained for API compatibility.
  -/
  strictCuda : Bool := false
deriving Repr

/- Convenience API for PyTorch-style device selection. -/
namespace Options

/-- Read the device selector corresponding to `useGpu`. -/
def device (opts : Options) : Device :=
  if opts.useGpu then .cuda else .cpu

/-- Set the device selector. -/
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

  The eager CUDA trainer uses this as a lightweight persistent-parameter cache: repeated forward
  passes can reuse the device buffer instead of uploading the host tensor every step.  The host
  `value` remains the public source for CPU/proof-oriented APIs and is synchronized on explicit
  readback.
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

/-- Release a cached CUDA mirror, if one exists.

CUDA buffers are external objects whose native finalizer tolerates repeated cleanup attempts, but
long training loops should not wait for GC pressure before returning replaced parameter mirrors to
the allocator.
-/
def releaseCachedCudaValue {α : Type} {s : Shape} (p : Param α s) : IO Unit := do
  match ← p.cudaValue.get with
  | none => pure ()
  | some any =>
      let _ := Runtime.Autograd.Cuda.Buffer.release any.buf
      pure ()

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

/-!
### Eager backend internals

The eager backend for the backend-generic `Ops` interface needs a small tape-backed session to
thread an `IO.Ref` to the runtime tape.

This is intentionally kept under `Torch.Internal.*`; the public session-style API is
`Runtime.Autograd.TorchLean.Session`.
-/

namespace Internal

/-!
### CUDA Bridge (Upload/Download)

The CUDA eager tape stores float32 device buffers (`Runtime.Autograd.Cuda.Buffer`) paired with a
runtime `Shape` (`Runtime.Autograd.Cuda.AnyBuffer`).

The Torch eager front-end still presents the spec-level `Tensor α s` API, so in CUDA mode we need:
- upload: `Tensor α s` -> `Cuda.AnyBuffer` (float32, contiguous, row-major)
- download: `Cuda.AnyBuffer` -> `Runtime.AnyTensor α`

The helper namespace gives CUDA bridge conversions stable call sites and a clear boundary.
-/
namespace CudaBridge

/-- Conversions required by the eager CUDA tape path. -/
class TensorConv (α : Type) where
  /-- Upload a spec tensor to a CUDA `AnyBuffer` (float32). -/
  toAnyBuffer : {s : Shape} → Tensor α s → IO Runtime.Autograd.Cuda.AnyBuffer
  /-- Download a CUDA `AnyBuffer` to a runtime `AnyTensor` (shape-erased). -/
  ofAnyBuffer : Runtime.Autograd.Cuda.AnyBuffer → IO (Runtime.AnyTensor α)
  /-- Convert a scalar constant to a host `Float` for CUDA kernels (e.g. `scale`, `axpy`). -/
  toFloat : α → IO Float

/-! #### Float implementation -/

/-- `Float` CUDA conversions: upload/download via row-major `FloatArray`. -/
instance (priority := 1000) : TensorConv Float where
  toAnyBuffer := fun {s} t => do
    let a := Runtime.Autograd.Cuda.Convert.flattenFloat (s := s) t
    pure { s := s, buf := Runtime.Autograd.Cuda.Buffer.ofFloatArray a }
  ofAnyBuffer := fun any => do
    let a := Runtime.Autograd.Cuda.Buffer.toFloatArray any.buf
    if a.size != Shape.size any.s then
      throw <| IO.userError
        s!"torch: cuda: bad buffer length (expected {Shape.size any.s}, got {a.size})"
    let t : Tensor Float any.s :=
      Runtime.Autograd.Cuda.Convert.unflattenFloatUnsafe (s := any.s) a
    pure { s := any.s, t := t }
  toFloat := fun x => pure x

/--
Generic CPU-preserving fallback for scalar types without a CUDA wire-format bridge.

Many TorchLean sessions are scalar-polymorphic on CPU, while the eager CUDA tape stores float32
buffers. The fallback keeps CPU execution available for proof-oriented scalar backends and fails
loudly if a CUDA-only conversion is actually requested. Add a higher-priority `TensorConv α`
instance for scalar types that have a deliberate float32 wire representation.
-/
instance (priority := 10) (α : Type) : TensorConv α where
  toAnyBuffer := fun {_s} _ =>
    throw <| IO.userError
      "torch: cuda: this scalar type has no CudaBridge.TensorConv upload; use Float for eager CUDA or run this scalar on CPU"
  ofAnyBuffer := fun _ =>
    throw <| IO.userError
      "torch: cuda: this scalar type has no CudaBridge.TensorConv download; use Float for eager CUDA or run this scalar on CPU"
  toFloat := fun _ =>
    throw <| IO.userError
      "torch: cuda: this scalar type has no CudaBridge.TensorConv scalar conversion; use Float for eager CUDA or run this scalar on CPU"

/-! #### Shape helpers for CUDA kernels -/

/-- Runtime dimension list as an `Array Nat` (outermost-first). -/
def dimsArray (s : Shape) : Array Nat :=
  Shape.toArray s

/-- `axisMap` as an array. -/
def axisMapArray {s₁ s₂ : Shape} (cb : Shape.CanBroadcastTo s₁ s₂) : Array Nat :=
  Runtime.Autograd.Cuda.Broadcast.axisMap cb

end CudaBridge

/--
Synchronize a CUDA-updated parameter back to its host tensor, if needed.

This is deliberately explicit.  Training hot paths keep parameters resident on device; public
readback APIs call this helper before exposing parameter tensors to the Lean side.
-/
def syncParamCudaToHost {α : Type} [CudaBridge.TensorConv α] {sh : Shape} [DecidableEq Shape]
    (p : Param α sh) : IO Unit := do
  let current ← p.hostCurrent.get
  if current then
    pure ()
  else
    match ← p.cudaValue.get with
    | none =>
        p.hostCurrent.set true
    | some any =>
        let anyHost ← CudaBridge.TensorConv.ofAnyBuffer (α := α) any
        if h : anyHost.s = sh then
          p.value.set (Tensor.castShape anyHost.t h)
          p.hostCurrent.set true
        else
          throw <| IO.userError <|
            s!"torch: CUDA param sync shape mismatch (expected {Shape.pretty sh}, got "
              ++ s!"{Shape.pretty anyHost.s})"

/-- Store/update the CUDA mirror of a parameter and mark the host tensor stale. -/
def setParamCudaValue {α : Type} {sh : Shape} (p : Param α sh)
    (any : Runtime.Autograd.Cuda.AnyBuffer) : IO Unit := do
  if _h : any.s = sh then
    AnyParam.releaseCachedCudaValue p
    p.cudaValue.set (some { s := sh, buf := any.buf })
    p.hostCurrent.set false
  else
    throw <| IO.userError <|
      s!"torch: CUDA param cache shape mismatch (expected {Shape.pretty sh}, got "
        ++ s!"{Shape.pretty any.s})"

/-- Overwrite a host parameter value and invalidate any stale CUDA mirror. -/
def setParamHostValue {α : Type} {sh : Shape} (p : Param α sh) (v : Tensor α sh) : IO Unit := do
  AnyParam.releaseCachedCudaValue p
  p.value.set v
  p.cudaValue.set none
  p.hostCurrent.set true

/--
Internal eager session: a mutable runtime tape plus side tables.

This is the state needed to offer a PyTorch-like API where "tensors" are opaque references and ops
mutate a hidden tape stored in an `IO.Ref`.

Notes:
- `tape` stores values and backward closures (`Runtime.Autograd.Tape`).
- `paramsByLeaf` remembers which tape leaf ids correspond to trainable parameters (for SGD).
- `nats` stores non-differentiable `Nat` inputs used for indexing-like operations.
-/
structure EagerSession (α : Type) where
  /-- Session options controlling backend/device/kernel behavior. -/
  opts : Options
  /-- CPU eager tape used when `opts.useGpu = false`. -/
  tape : IO.Ref (Runtime.Autograd.Tape α)
  /-- CUDA eager tape used when `opts.useGpu = true`. -/
  cudaTape : IO.Ref (Runtime.Autograd.Cuda.Tape)
  /-- Map from tape leaf ids to trainable parameter objects. -/
  paramsByLeaf : IO.Ref (Std.HashMap Nat (AnyParam α))
  /-- Non-differentiable integer inputs for dynamic indexing operations. -/
  nats : IO.Ref (Array Nat)

namespace EagerSession

/-- Allocate a fresh eager session with an empty tape and empty side tables. -/
def new {α : Type} (opts : Options := {}) : IO (EagerSession α) := do
  let tape ← IO.mkRef Runtime.Autograd.Tape.empty
  let cudaTape ← IO.mkRef Runtime.Autograd.Cuda.Tape.empty
  let paramsByLeaf ← IO.mkRef (Std.HashMap.emptyWithCapacity)
  let nats ← IO.mkRef #[]
  pure { opts := opts, tape := tape, cudaTape := cudaTape, paramsByLeaf := paramsByLeaf, nats := nats }

/-- Force-free a CUDA buffer allocation; the external finalizer is safe to call twice. -/
def releaseCudaBuffer (b : Runtime.Autograd.Cuda.Buffer) : IO Unit := do
  let released := Runtime.Autograd.Cuda.Buffer.release b
  if released == 0 then
    pure ()
  else
    pure ()

/-- Force-release a shape-erased CUDA buffer. -/
def releaseCudaAnyBuffer (b : Runtime.Autograd.Cuda.AnyBuffer) : IO Unit :=
  releaseCudaBuffer b.buf

/--
Release current CUDA tape values that are not persistent parameter mirrors.

Eager CUDA training creates many temporary buffers per step. Relying only on external-object
finalizers can produce high transient memory pressure in long runs, so reset/step paths call this
before discarding the current tape snapshot.
-/
def releaseCudaTapeNonParamValues {α : Type} (s : EagerSession α) : IO Unit := do
  let t ← s.cudaTape.get
  let params ← s.paramsByLeaf.get
  for i in [0:t.nodes.size] do
    if params.contains i then
      match t.nodes[i]? with
      | none => pure ()
      | some node =>
          for b in node.cleanup do
            releaseCudaBuffer b
    else
      match t.nodes[i]? with
      | none => pure ()
      | some node =>
          releaseCudaAnyBuffer node.value
          for b in node.cleanup do
            releaseCudaBuffer b

/--
Release CUDA tape values after an optimizer step.

Unlike `releaseCudaTapeNonParamValues`, this may release trainable parameter leaf buffers too. In a
CUDA optimizer step, trainable parameters have already been written to fresh persistent mirrors, so
the leaf buffers from the just-consumed tape are stale. Non-trainable parameter leaves still *are*
their persistent mirrors, so we keep those cached.
-/
def releaseCudaTapeAfterOptimizerStep {α : Type} (s : EagerSession α) : IO Unit := do
  let t ← s.cudaTape.get
  let params ← s.paramsByLeaf.get
  for i in [0:t.nodes.size] do
    match t.nodes[i]? with
    | none => pure ()
    | some node =>
        match params.get? i with
        | some p =>
            if p.requiresGrad then
              releaseCudaAnyBuffer node.value
            else
              pure ()
        | none =>
            releaseCudaAnyBuffer node.value
        for b in node.cleanup do
          releaseCudaBuffer b

/-- Release a dense CUDA gradient array after an optimizer has consumed it. -/
def releaseCudaAnyBufferArray (xs : Array Runtime.Autograd.Cuda.AnyBuffer) : IO Unit := do
  for x in xs do
    releaseCudaAnyBuffer x

/-- Ask the native allocator to return/free pages after a large CUDA eager step. -/
def collectCudaAllocator : IO Unit := do
  let released := Runtime.Autograd.Cuda.Buffer.collectAllocator true
  -- Branch on the opaque FFI result so the collector call is not optimized away.
  if released == 0 then
    pure ()
  else
    pure ()

/--
Reset the tape and side tables.

PyTorch comparison: like starting a fresh forward pass where the autograd graph is discarded.
-/
def resetTape {α : Type} (s : EagerSession α) : IO Unit := do
  if Options.device s.opts == .cuda then
    releaseCudaTapeNonParamValues s
  s.tape.set Runtime.Autograd.Tape.empty
  s.cudaTape.set Runtime.Autograd.Cuda.Tape.empty
  s.paramsByLeaf.set (Std.HashMap.emptyWithCapacity)
  s.nats.set #[]

/--
Create a mutable parameter object (not yet on the tape).

To record this parameter on the session tape, call `use`, which reads the parameter and records it
as a leaf.
-/
def param {α : Type} (s : EagerSession α) {sh : Shape}
  (init : Tensor α sh) (name : Option String := none) (requiresGrad : Option Bool := none) :
  IO (Param α sh) := do
  let r ← IO.mkRef init
  let cudaValue ← IO.mkRef (none : Option Runtime.Autograd.Cuda.AnyBuffer)
  let hostCurrent ← IO.mkRef true
  pure { name := name
         value := r
         cudaValue := cudaValue
         hostCurrent := hostCurrent
         requiresGrad := requiresGrad.getD s.opts.requiresGradByDefault }

/--
Read back the concrete tensor value stored at a `TensorRef`.

This is a dynamic check: we ensure the id exists on the tape and the stored shape matches `sh`.
-/
def getValue {α : Type} [CudaBridge.TensorConv α] (s : EagerSession α) {sh : Shape} [DecidableEq Shape]
  (x : TensorRef α sh) : IO (Tensor α sh) := do
  if Options.device s.opts == .cuda then
    let t ← s.cudaTape.get
    let any ← match t.getValue? x.id with
      | some v => pure v
      | none => throw <| IO.userError "torch: invalid tensor id (missing CUDA value)"
    let anyHost ← CudaBridge.TensorConv.ofAnyBuffer (α := α) any
    if h : anyHost.s = sh then
      pure (Tensor.castShape anyHost.t h)
    else
      throw <| IO.userError <|
        s!"torch: shape mismatch when reading value (expected {Shape.pretty sh}, got "
          ++ s!"{Shape.pretty anyHost.s})"
  else
    let t ← s.tape.get
    let any ← match t.getValue? x.id with
      | some v => pure v
      | none => throw <| IO.userError "torch: invalid tensor id (missing value)"
    if h : any.s = sh then
      pure (Tensor.castShape any.t h)
    else
      throw <| IO.userError <|
        s!"torch: shape mismatch when reading value (expected {Shape.pretty sh}, got "
          ++ s!"{Shape.pretty any.s})"

/--
Record a constant leaf (non-differentiable) on the tape.

PyTorch comparison: like constructing a tensor with `requires_grad=False`.
-/
def const {α : Type} [CudaBridge.TensorConv α] (s : EagerSession α) {sh : Shape} [DecidableEq Shape]
  (v : Tensor α sh) (name : Option String := none) : IO (TensorRef α sh) := do
  if Options.device s.opts == .cuda then
    let anyBuf ← CudaBridge.TensorConv.toAnyBuffer (α := α) (s := sh) v
    let t0 ← s.cudaTape.get
    let (t1, id) :=
      Runtime.Autograd.Cuda.Tape.leaf (t := t0) (value := anyBuf) (name := name)
        (requires_grad := false)
    s.cudaTape.set t1
    pure { id := id }
  else
    let t0 ← s.tape.get
    let (t1, id) := Runtime.Autograd.Tape.leaf (t := t0) (s := sh) (value := v)
      (name := name) (requires_grad := false)
    s.tape.set t1
    pure { id := id }

/--
Stop-gradient boundary.

Forward semantics: identity (`detach(x) = x`).
Backward semantics: no gradient flows to `x`.
-/
def detach {α : Type} [CudaBridge.TensorConv α] (s : EagerSession α) {sh : Shape} [DecidableEq Shape]
    (x : TensorRef α sh) (name : Option String := none) : IO (TensorRef α sh) := do
  if Options.device s.opts == .cuda then
    let t0 ← s.cudaTape.get
    let any ← match t0.getValue? x.id with
      | some v => pure v
      | none => throw <| IO.userError "torch: detach: invalid tensor id (missing CUDA value)"
    if _h : any.s = sh then
      let any' : Runtime.Autograd.Cuda.AnyBuffer := { s := sh, buf := any.buf }
      let node : Runtime.Autograd.Cuda.Node :=
        { name := name
          value := any'
          requires_grad := false
          parents := [x.id]
          backward := fun _ => .ok [] }
      let (t1, id) := Runtime.Autograd.Cuda.Tape.addNode t0 node
      s.cudaTape.set t1
      pure { id := id }
    else
      throw <| IO.userError <|
        s!"torch: detach: shape mismatch (expected {Shape.pretty sh}, got {Shape.pretty any.s})"
  else
    let xVal ← getValue (α := α) s (sh := sh) x
    let t0 ← s.tape.get
    let node : Runtime.Autograd.Node α :=
      { name := name
        value := Runtime.Autograd.AnyTensor.mk xVal
        requires_grad := false
        parents := [x.id]
        backward := fun _ => .ok [] }
    let (t1, id) := Runtime.Autograd.Tape.addNode t0 node
    s.tape.set t1
    pure { id := id }

/-- Deterministic `U[0,1)` tensor generator (seeded). -/
def randUniform {α : Type} [Context α] [CudaBridge.TensorConv α]
    (s : EagerSession α) {sh : Shape} [DecidableEq Shape]
    (seed : Nat) (name : Option String := none) : IO (TensorRef α sh) := do
  if Options.device s.opts == .cuda then
    let t0 ← s.cudaTape.get
    let counter := t0.size
    let key := Runtime.Autograd.TorchLean.Random.keyOf seed counter
    let n32 := UInt32.ofNat (Shape.size sh)
    let buf := Runtime.Autograd.Cuda.Buffer.randUniform n32 key
    let any : Runtime.Autograd.Cuda.AnyBuffer := { s := sh, buf := buf }
    let (t1, id) :=
      Runtime.Autograd.Cuda.Tape.leaf (t := t0) (value := any) (name := name) (requires_grad := false)
    s.cudaTape.set t1
    pure { id := id }
  else
    let t0 ← s.tape.get
    let counter := t0.size
    let key := Runtime.Autograd.TorchLean.Random.keyOf seed counter
    let v : Tensor α sh := Runtime.Autograd.TorchLean.Random.uniform (α := α) key (s := sh)
    const (α := α) s (sh := sh) v (name := name)

/-- Deterministic `{0,1}` mask generator (seeded) with a scalar keep-probability input. -/
def bernoulliMask {α : Type} [Context α] [CudaBridge.TensorConv α]
    (s : EagerSession α) {sh : Shape} [DecidableEq Shape]
    (keepProb : TensorRef α Shape.scalar) (seed : Nat) (name : Option String := none) :
    IO (TensorRef α sh) := do
  let kpT ← getValue (α := α) s (sh := Shape.scalar) keepProb
  let keepProbVal : α :=
    match kpT with
    | Tensor.scalar v => v
  if Options.device s.opts == .cuda then
    let t0 ← s.cudaTape.get
    let counter := t0.size
    let key := Runtime.Autograd.TorchLean.Random.keyOf seed counter
    let n32 := UInt32.ofNat (Shape.size sh)
    let keepF ← CudaBridge.TensorConv.toFloat (α := α) keepProbVal
    let buf := Runtime.Autograd.Cuda.Buffer.bernoulliMask n32 keepF key
    let any : Runtime.Autograd.Cuda.AnyBuffer := { s := sh, buf := buf }
    let (t1, id) :=
      Runtime.Autograd.Cuda.Tape.leaf (t := t0) (value := any) (name := name) (requires_grad := false)
    s.cudaTape.set t1
    pure { id := id }
  else
    let t0 ← s.tape.get
    let counter := t0.size
    let key := Runtime.Autograd.TorchLean.Random.keyOf seed counter
    let v : Tensor α sh := Runtime.Autograd.TorchLean.Random.mask (α := α) key keepProbVal (s := sh)
    const (α := α) s (sh := sh) v (name := name)

/--
Use a parameter in the tape by recording its current value as a leaf.

The returned `TensorRef` is the handle you pass to ops. The leaf id is stored in `paramsByLeaf` so
optimizer steps (e.g. SGD) can update parameters after `backward`.
PyTorch comparison: like using a `torch.nn.Parameter` in a forward pass (it becomes a leaf in the
autograd graph).
-/
def use {α : Type} [CudaBridge.TensorConv α] (s : EagerSession α) {sh : Shape} [DecidableEq Shape]
  (p : Param α sh) : IO (TensorRef α sh) := do
  let id ←
    if Options.device s.opts == .cuda then
      let anyBuf ←
        match ← p.cudaValue.get with
        | some any =>
            if _h : any.s = sh then
              pure ({ s := sh, buf := any.buf } : Runtime.Autograd.Cuda.AnyBuffer)
            else
              -- A well-formed `Param` has a cached CUDA buffer with the declared shape; if the
              -- cache is inconsistent, re-upload from the host value.
              let v ← p.value.get
              let uploaded ← CudaBridge.TensorConv.toAnyBuffer (α := α) (s := sh) v
              AnyParam.releaseCachedCudaValue p
              p.cudaValue.set (some uploaded)
              p.hostCurrent.set true
              pure uploaded
        | none =>
            let v ← p.value.get
            let uploaded ← CudaBridge.TensorConv.toAnyBuffer (α := α) (s := sh) v
            AnyParam.releaseCachedCudaValue p
            p.cudaValue.set (some uploaded)
            p.hostCurrent.set true
            pure uploaded
      let t0 ← s.cudaTape.get
      let (t1, id) :=
        Runtime.Autograd.Cuda.Tape.leaf (t := t0) (value := anyBuf) (name := p.name)
          (requires_grad := p.requiresGrad)
      s.cudaTape.set t1
      pure id
    else
      syncParamCudaToHost (α := α) (sh := sh) p
      let v ← p.value.get
      let t0 ← s.tape.get
      let (t1, id) :=
        Runtime.Autograd.Tape.leaf (t := t0) (s := sh)
          (value := v) (name := p.name) (requires_grad := p.requiresGrad)
      s.tape.set t1
      pure id
  s.paramsByLeaf.modify (fun m => m.insert id (AnyParam.ofParam p))
  pure { id := id }

/--
Record an external input tensor as a leaf on the tape.

PyTorch comparison: like introducing a tensor into the autograd graph with a chosen
`requires_grad` flag.
-/
def input {α : Type} [CudaBridge.TensorConv α] (s : EagerSession α) {sh : Shape} [DecidableEq Shape]
  (v : Tensor α sh) (name : Option String := none) (requiresGrad : Bool := false) :
  IO (TensorRef α sh) := do
  if Options.device s.opts == .cuda then
    let anyBuf ← CudaBridge.TensorConv.toAnyBuffer (α := α) (s := sh) v
    let t0 ← s.cudaTape.get
    let (t1, id) := Runtime.Autograd.Cuda.Tape.leaf (t := t0) (value := anyBuf)
      (name := name) (requires_grad := requiresGrad)
    s.cudaTape.set t1
    pure { id := id }
  else
    let t0 ← s.tape.get
    let (t1, id) := Runtime.Autograd.Tape.leaf (t := t0) (s := sh) (value := v)
      (name := name) (requires_grad := requiresGrad)
    s.tape.set t1
    pure { id := id }

/--
Record a non-differentiable `Nat` input in the session environment.

This supports ops that depend on indices/labels that should not receive gradients.
-/
def inputNat {α : Type} (s : EagerSession α) (v : Nat) : IO NatRef := do
  let xs ← s.nats.get
  let id := xs.size
  s.nats.set (xs.push v)
  pure { id := id }

/-- Read a previously recorded `NatRef`. -/
def getNat {α : Type} (s : EagerSession α) (r : NatRef) : IO Nat := do
  let xs ← s.nats.get
  if h : r.id < xs.size then
    pure <| xs[r.id]'h
  else
    throw <| IO.userError "torch: invalid nat id"

/-- Overwrite a previously recorded `NatRef`. -/
def setNat {α : Type} (s : EagerSession α) (r : NatRef) (v : Nat) : IO Unit := do
  let xs ← s.nats.get
  if h : r.id < xs.size then
    let i : Fin xs.size := ⟨r.id, h⟩
    s.nats.set (xs.set i v)
  else
    throw <| IO.userError "torch: invalid nat id"

/--
Convert a `Tensor Nat (.dim k .scalar)` to an `Array Nat`.

Used to stage `NatVecRef` values into the session environment.
-/
def natVecToArray {k : Nat} (v : Tensor Nat (.dim k .scalar)) : Array Nat :=
  match v with
  | .dim f =>
      Array.ofFn (fun i : Fin k =>
        match f i with
        | .scalar n => n)

/--
Record a non-differentiable vector of `Nat`s in the session environment.

Returns a `NatVecRef k` pointing to the stored block.
-/
def inputNatVec {α : Type} {k : Nat} (s : EagerSession α) (v : Tensor Nat (.dim k .scalar)) : IO
  (NatVecRef k) := do
  let old ← s.nats.get
  let start := old.size
  let xsNew := (natVecToArray (k := k) v).foldl (fun acc x => acc.push x) old
  s.nats.set xsNew
  pure { start := start }

/-- Read back the vector stored at `NatVecRef k`. -/
def getNatVec {α : Type} {k : Nat} (s : EagerSession α) (r : NatVecRef k) : IO (Tensor Nat (.dim k
  .scalar)) := do
  let xs ← s.nats.get
  if h : r.start + k ≤ xs.size then
    pure <|
      Tensor.dim (fun i =>
        have hi : r.start + i.val < r.start + k := Nat.add_lt_add_left i.is_lt r.start
        have hi' : r.start + i.val < xs.size := lt_of_lt_of_le hi h
        Tensor.scalar (xs[r.start + i.val]'hi'))
  else
    throw <| IO.userError "torch: invalid nat vec ref (out of bounds)"

/-- Overwrite the stored vector at `NatVecRef k`. -/
def setNatVec {α : Type} {k : Nat} (s : EagerSession α) (r : NatVecRef k) (v : Tensor Nat (.dim k
  .scalar)) : IO Unit := do
  let xs ← s.nats.get
  if h : r.start + k ≤ xs.size then
    let xs' :=
      (List.finRange k).foldl (fun acc (i : Fin k) =>
        have hi : r.start + i.val < xs.size := by
          have hlt : r.start + i.val < r.start + k := Nat.add_lt_add_left i.is_lt r.start
          exact lt_of_lt_of_le hlt h
        let vi : Nat :=
          match getAtSpec v i with
          | .scalar n => n
        acc.set! (r.start + i.val) vi
      ) xs
    s.nats.set xs'
  else
    throw <| IO.userError "torch: invalid nat vec ref (out of bounds)"

/-!
## Tensor ops (eager tape wrappers)

The following definitions are thin wrappers around `Runtime.Autograd.Tape.*` primitives. Each one:
- reads the current tape from `s.tape`,
- appends a new node/leaf via a `Tape.*` constructor,
- writes the updated tape back, and
- returns a fresh `TensorRef` pointing to the new node id.

PyTorch comparison: this is the standard eager autograd mechanism (a dynamic tape of ops).
-/

/--
Dispatch an eager op with optional CUDA support.

When `Options.device = .cuda`, any op whose CUDA implementation returns `none` will throw.

TorchLean's CUDA eager mode is intentionally "no per-op CPU fallback": either the op is
supported by CUDA, or it errors immediately.
-/
def dispatchCudaOpt {α β : Type} (s : EagerSession α) (opName : String)
    (cpu : IO β) (cuda : IO (Option β)) : IO β := do
  if Options.device s.opts == .cuda then
    let r? ← cuda
    match r? with
    | some r => pure r
    | none =>
        throw <| IO.userError s!"torch: cuda: `{opName}` is unsupported by the eager CUDA backend"
  else
    cpu

/-- Record elementwise addition `a + b`. PyTorch: `torch.add`. -/
def add {α : Type} (s : EagerSession α) [Add α] [DecidableEq Shape] {sh : Shape}
  (a b : TensorRef α sh) : IO (TensorRef α sh) := do
  let cpu := do
    let t0 ← s.tape.get
    let (t1, id) ← okOrThrow (Runtime.Autograd.Tape.add (t := t0) (s := sh) a.id b.id)
    s.tape.set t1
    pure { id := id }
  let cuda := do
    let t0 ← s.cudaTape.get
    let (t1, id) ← okOrThrow <|
      Runtime.Autograd.Cuda.Tape.add (t := t0) (s := sh) a.id b.id
    s.cudaTape.set t1
    pure (some { id := id })
  dispatchCudaOpt (α := α) s "add" cpu cuda

/-- Record elementwise subtraction `a - b`. PyTorch: `torch.sub`. -/
def sub {α : Type} (s : EagerSession α) [Sub α] [Zero α] [DecidableEq Shape] {sh : Shape}
  (a b : TensorRef α sh) : IO (TensorRef α sh) := do
  let cpu := do
    let t0 ← s.tape.get
    let (t1, id) ← okOrThrow (Runtime.Autograd.Tape.sub (t := t0) (s := sh) a.id b.id)
    s.tape.set t1
    pure { id := id }
  let cuda := do
    let t0 ← s.cudaTape.get
    let (t1, id) ← okOrThrow <|
      Runtime.Autograd.Cuda.Tape.sub (t := t0) (s := sh) a.id b.id
    s.cudaTape.set t1
    pure (some { id := id })
  dispatchCudaOpt (α := α) s "sub" cpu cuda

/-- Record elementwise multiplication `a * b`. PyTorch: `torch.mul`. -/
def mul {α : Type} (s : EagerSession α) [Mul α] [DecidableEq Shape] {sh : Shape}
  (a b : TensorRef α sh) : IO (TensorRef α sh) := do
  let cpu := do
    let t0 ← s.tape.get
    let (t1, id) ← okOrThrow (Runtime.Autograd.Tape.mul (t := t0) (s := sh) a.id b.id)
    s.tape.set t1
    pure { id := id }
  let cuda := do
    let t0 ← s.cudaTape.get
    let (t1, id) ← okOrThrow <|
      Runtime.Autograd.Cuda.Tape.mul (t := t0) (s := sh) a.id b.id
    s.cudaTape.set t1
    pure (some { id := id })
  dispatchCudaOpt (α := α) s "mul" cpu cuda

/-- Record scaling by a scalar constant. PyTorch: `x * c`. -/
def scale {α : Type} [CudaBridge.TensorConv α] (s : EagerSession α) [Mul α] [DecidableEq Shape]
  {sh : Shape}
  (x : TensorRef α sh) (c : α) : IO (TensorRef α sh) := do
  let cpu := do
    let t0 ← s.tape.get
    let (t1, id) ← okOrThrow (Runtime.Autograd.Tape.scale (t := t0) (s := sh) x.id c)
    s.tape.set t1
    pure { id := id }
  let cuda := do
    let cF ← CudaBridge.TensorConv.toFloat (α := α) c
    let t0 ← s.cudaTape.get
    let (t1, id) ← okOrThrow <|
      Runtime.Autograd.Cuda.Tape.scale (t := t0) (s := sh) x.id cF
    s.cudaTape.set t1
    pure (some { id := id })
  dispatchCudaOpt (α := α) s "scale" cpu cuda

/-- Record elementwise absolute value. PyTorch: `torch.abs`. -/
def abs {α : Type} (s : EagerSession α) [Context α] [DecidableRel ((· > ·) : α → α → Prop)]
  [DecidableEq Shape]
  {sh : Shape} (x : TensorRef α sh) : IO (TensorRef α sh) := do
  let cpu := do
    let t0 ← s.tape.get
    let (t1, id) ← okOrThrow (Runtime.Autograd.Tape.abs (t := t0) (s := sh) x.id)
    s.tape.set t1
    pure { id := id }
  let cuda := do
    let t0 ← s.cudaTape.get
    let (t1, id) ← okOrThrow <|
      Runtime.Autograd.Cuda.Tape.abs (t := t0) (s := sh) x.id
    s.cudaTape.set t1
    pure (some { id := id })
  dispatchCudaOpt (α := α) s "abs" cpu cuda

/-- Record elementwise square root. PyTorch: `torch.sqrt`. -/
def sqrt {α : Type} (s : EagerSession α) [Context α] [DecidableRel ((· > ·) : α → α → Prop)]
  [DecidableEq Shape]
  {sh : Shape} (x : TensorRef α sh) : IO (TensorRef α sh) := do
  let cpu := do
    let t0 ← s.tape.get
    let (t1, id) ← okOrThrow (Runtime.Autograd.Tape.sqrt (t := t0) (s := sh) x.id)
    s.tape.set t1
    pure { id := id }
  let cuda := do
    let t0 ← s.cudaTape.get
    let (t1, id) ← okOrThrow <|
      Runtime.Autograd.Cuda.Tape.sqrt (t := t0) (s := sh) x.id
    s.cudaTape.set t1
    pure (some { id := id })
  dispatchCudaOpt (α := α) s "sqrt" cpu cuda

/-- Record elementwise clamp to `[minVal,maxVal]`. PyTorch: `torch.clamp`. -/
def clamp {α : Type} [CudaBridge.TensorConv α] (s : EagerSession α) [Context α]
  [DecidableRel ((· > ·) : α → α → Prop)] [DecidableEq Shape]
  {sh : Shape} (x : TensorRef α sh) (minVal maxVal : α) : IO (TensorRef α sh) := do
  let cpu := do
    let t0 ← s.tape.get
    let (t1, id) ← okOrThrow (Runtime.Autograd.Tape.clamp (t := t0) (s := sh) x.id minVal maxVal)
    s.tape.set t1
    pure { id := id }
  let cuda := do
    let lo ← CudaBridge.TensorConv.toFloat (α := α) minVal
    let hi ← CudaBridge.TensorConv.toFloat (α := α) maxVal
    let t0 ← s.cudaTape.get
    let (t1, id) ← okOrThrow <|
      Runtime.Autograd.Cuda.Tape.clamp (t := t0) (s := sh) x.id lo hi
    s.cudaTape.set t1
    pure (some { id := id })
  dispatchCudaOpt (α := α) s "clamp" cpu cuda

/-- Record elementwise maximum. PyTorch: `torch.maximum`. -/
def max {α : Type} (s : EagerSession α) [Context α] [DecidableRel ((· > ·) : α → α → Prop)]
  [DecidableEq Shape]
  {sh : Shape} (a b : TensorRef α sh) : IO (TensorRef α sh) := do
  let cpu := do
    let t0 ← s.tape.get
    let (t1, id) ← okOrThrow (Runtime.Autograd.Tape.max (t := t0) (s := sh) a.id b.id)
    s.tape.set t1
    pure { id := id }
  let cuda := do
    let t0 ← s.cudaTape.get
    let (t1, id) ← okOrThrow <|
      Runtime.Autograd.Cuda.Tape.max (t := t0) (s := sh) a.id b.id
    s.cudaTape.set t1
    pure (some { id := id })
  dispatchCudaOpt (α := α) s "max" cpu cuda

/-- Record elementwise minimum. PyTorch: `torch.minimum`. -/
def min {α : Type} (s : EagerSession α) [Context α] [DecidableRel ((· > ·) : α → α → Prop)]
  [DecidableEq Shape]
  {sh : Shape} (a b : TensorRef α sh) : IO (TensorRef α sh) := do
  let cpu := do
    let t0 ← s.tape.get
    let (t1, id) ← okOrThrow (Runtime.Autograd.Tape.min (t := t0) (s := sh) a.id b.id)
    s.tape.set t1
    pure { id := id }
  let cuda := do
    let t0 ← s.cudaTape.get
    let (t1, id) ← okOrThrow <|
      Runtime.Autograd.Cuda.Tape.min (t := t0) (s := sh) a.id b.id
    s.cudaTape.set t1
    pure (some { id := id })
  dispatchCudaOpt (α := α) s "min" cpu cuda

/-- Record elementwise ReLU. PyTorch: `torch.relu` / `torch.nn.functional.relu`. -/
def relu {α : Type} (s : EagerSession α)
  [Mul α] [Zero α] [Max α] [One α] [LT α]
  [DecidableRel ((· > ·) : α → α → Prop)] [DecidableEq Shape]
  {sh : Shape} (x : TensorRef α sh) : IO (TensorRef α sh) := do
  let cpu := do
    let t0 ← s.tape.get
    let (t1, id) ← okOrThrow (Runtime.Autograd.Tape.relu (t := t0) (s := sh) x.id)
    s.tape.set t1
    pure { id := id }
  let cuda := do
    let t0 ← s.cudaTape.get
    let (t1, id) ← okOrThrow <|
      Runtime.Autograd.Cuda.Tape.relu (t := t0) (s := sh) x.id
    s.cudaTape.set t1
    pure (some { id := id })
  dispatchCudaOpt (α := α) s "relu" cpu cuda

/-- Record elementwise sigmoid. PyTorch: `torch.sigmoid`. -/
def sigmoid {α : Type} (s : EagerSession α) [Context α] [DecidableEq Shape]
  {sh : Shape} (x : TensorRef α sh) : IO (TensorRef α sh) := do
  let cpu := do
    let t0 ← s.tape.get
    let (t1, id) ← okOrThrow (Runtime.Autograd.Tape.sigmoid (t := t0) (s := sh) x.id)
    s.tape.set t1
    pure { id := id }
  let cuda := do
    let t0 ← s.cudaTape.get
    let (t1, id) ← okOrThrow (Runtime.Autograd.Cuda.Tape.sigmoid (t := t0) (s := sh) x.id)
    s.cudaTape.set t1
    pure (some { id := id })
  dispatchCudaOpt (α := α) s "sigmoid" cpu cuda

/-- Record elementwise tanh. PyTorch: `torch.tanh`. -/
def tanh {α : Type} (s : EagerSession α) [Context α] [DecidableEq Shape]
  {sh : Shape} (x : TensorRef α sh) : IO (TensorRef α sh) := do
  let cpu := do
    let t0 ← s.tape.get
    let (t1, id) ← okOrThrow (Runtime.Autograd.Tape.tanh (t := t0) (s := sh) x.id)
    s.tape.set t1
    pure { id := id }
  let cuda := do
    let t0 ← s.cudaTape.get
    let (t1, id) ← okOrThrow (Runtime.Autograd.Cuda.Tape.tanh (t := t0) (s := sh) x.id)
    s.cudaTape.set t1
    pure (some { id := id })
  dispatchCudaOpt (α := α) s "tanh" cpu cuda

/--
Record softmax (shape-preserving).

PyTorch comparison: `torch.softmax(x, dim=...)` (dimension convention is chosen by the underlying
  tape op).
-/
def softmax {α : Type} (s : EagerSession α) [Context α] [DecidableEq Shape]
  {sh : Shape} (x : TensorRef α sh) : IO (TensorRef α sh) := do
  let cpu := do
    let t0 ← s.tape.get
    let (t1, id) ← okOrThrow (Runtime.Autograd.Tape.softmax (t := t0) (s := sh) x.id)
    s.tape.set t1
    pure { id := id }
  let cuda := do
    let t0 ← s.cudaTape.get
    let (t1, id) ← okOrThrow (Runtime.Autograd.Cuda.Tape.softmax (t := t0) (s := sh) x.id)
    s.cudaTape.set t1
    pure (some { id := id })
  dispatchCudaOpt (α := α) s "softmax" cpu cuda

/--
Record stable log-softmax (shape-preserving, last-axis convention).

PyTorch comparison: `torch.nn.functional.log_softmax(x, dim=-1)`.
-/
def logSoftmax {α : Type} (s : EagerSession α) [Context α] [DecidableEq Shape]
  {sh : Shape} (x : TensorRef α sh) : IO (TensorRef α sh) := do
  let cpu := do
    let t0 ← s.tape.get
    let (t1, id) ← okOrThrow (Runtime.Autograd.Tape.logSoftmax (t := t0) (s := sh) x.id)
    s.tape.set t1
    pure { id := id }
  let cuda := do
    let t0 ← s.cudaTape.get
    let (t1, id) ← okOrThrow (Runtime.Autograd.Cuda.Tape.logSoftmax (t := t0) (s := sh) x.id)
    s.cudaTape.set t1
    pure (some { id := id })
  dispatchCudaOpt (α := α) s "log_softmax" cpu cuda

/-- Record elementwise softplus. PyTorch: `torch.nn.functional.softplus`. -/
def softplus {α : Type} (s : EagerSession α) [Context α] [DecidableEq Shape]
  {sh : Shape} (x : TensorRef α sh) : IO (TensorRef α sh) := do
  let cpu := do
    let t0 ← s.tape.get
    let (t1, id) ← okOrThrow (Runtime.Autograd.Tape.softplus (t := t0) (s := sh) x.id)
    s.tape.set t1
    pure { id := id }
  let cuda := do
    let t0 ← s.cudaTape.get
    let (t1, id) ← okOrThrow (Runtime.Autograd.Cuda.Tape.softplus (t := t0) (s := sh) x.id)
    s.cudaTape.set t1
    pure (some { id := id })
  dispatchCudaOpt (α := α) s "softplus" cpu cuda

/-- Record elementwise exponential. PyTorch: `torch.exp`. -/
def exp {α : Type} (s : EagerSession α) [Context α] [DecidableEq Shape]
  {sh : Shape} (x : TensorRef α sh) : IO (TensorRef α sh) := do
  let cpu := do
    let t0 ← s.tape.get
    let (t1, id) ← okOrThrow (Runtime.Autograd.Tape.exp (t := t0) (s := sh) x.id)
    s.tape.set t1
    pure { id := id }
  let cuda := do
    let t0 ← s.cudaTape.get
    let (t1, id) ← okOrThrow <|
      Runtime.Autograd.Cuda.Tape.exp (t := t0) (s := sh) x.id
    s.cudaTape.set t1
    pure (some { id := id })
  dispatchCudaOpt (α := α) s "exp" cpu cuda

/-- Record elementwise log. PyTorch: `torch.log`. -/
def log {α : Type} (s : EagerSession α) [Context α] [DecidableEq Shape]
  {sh : Shape} (x : TensorRef α sh) : IO (TensorRef α sh) := do
  let cpu := do
    let t0 ← s.tape.get
    let (t1, id) ← okOrThrow (Runtime.Autograd.Tape.log (t := t0) (s := sh) x.id)
    s.tape.set t1
    pure { id := id }
  let cuda := do
    let t0 ← s.cudaTape.get
    let (t1, id) ← okOrThrow <|
      Runtime.Autograd.Cuda.Tape.log (t := t0) (s := sh) x.id
    s.cudaTape.set t1
    pure (some { id := id })
  dispatchCudaOpt (α := α) s "log" cpu cuda

/-- Record elementwise inverse `1/x`. PyTorch: `torch.reciprocal`. -/
def inv {α : Type} (s : EagerSession α) [Context α] [DecidableEq Shape]
  {sh : Shape} (x : TensorRef α sh) : IO (TensorRef α sh) := do
  let cpu := do
    let t0 ← s.tape.get
    let (t1, id) ← okOrThrow (Runtime.Autograd.Tape.inv (t := t0) (s := sh) x.id)
    s.tape.set t1
    pure { id := id }
  let cuda := do
    let t0 ← s.cudaTape.get
    let (t1, id) ← okOrThrow <|
      Runtime.Autograd.Cuda.Tape.inv (t := t0) (s := sh) x.id
    s.cudaTape.set t1
    pure (some { id := id })
  dispatchCudaOpt (α := α) s "inv" cpu cuda

/--
Record elementwise log with epsilon guard.

PyTorch comparison: `torch.log(torch.clamp(x, min=ε))`.
-/
def safeLog {α : Type} [CudaBridge.TensorConv α] (s : EagerSession α) [Context α] [DecidableEq Shape]
  {sh : Shape} (x : TensorRef α sh) (ε : α := Numbers.epsilon) : IO (TensorRef α sh) := do
  let cpu := do
    let t0 ← s.tape.get
    let (t1, id) ← okOrThrow (Runtime.Autograd.Tape.safeLog (t := t0) (s := sh) x.id (ε := ε))
    s.tape.set t1
    pure { id := id }
  let cuda := do
    let epsF ← CudaBridge.TensorConv.toFloat (α := α) ε
    let t0 ← s.cudaTape.get
    let (t1, id) ← okOrThrow <|
      Runtime.Autograd.Cuda.Tape.safeLog (t := t0) (s := sh) x.id epsF
    s.cudaTape.set t1
    pure (some { id := id })
  dispatchCudaOpt (α := α) s "safe_log" cpu cuda

/-- Sum-reduce all elements to a scalar. PyTorch: `x.sum()`. -/
def sum {α : Type} (s : EagerSession α) [Add α] [Zero α] [DecidableEq Shape]
  {sh : Shape} (x : TensorRef α sh) : IO (TensorRef α Shape.scalar) := do
  let cpu := do
    let t0 ← s.tape.get
    let (t1, id) ← okOrThrow (Runtime.Autograd.Tape.sum (t := t0) (s := sh) x.id)
    s.tape.set t1
    pure { id := id }
  let cuda := do
    let t0 ← s.cudaTape.get
    let (t1, id) ← okOrThrow <|
      Runtime.Autograd.Cuda.Tape.sum (t := t0) (s := sh) x.id
    s.cudaTape.set t1
    pure (some { id := id })
  dispatchCudaOpt (α := α) s "sum" cpu cuda

/-- Flatten a tensor to a 1D vector. PyTorch: `torch.flatten`. -/
def flatten {α : Type} (s : EagerSession α) [Inhabited α] [DecidableEq Shape] {sh : Shape}
  (x : TensorRef α sh) : IO (TensorRef α (.dim (Shape.size sh) .scalar)) := do
  let cpu := do
    let t0 ← s.tape.get
    let (t1, id) ← okOrThrow (Runtime.Autograd.Tape.flatten (t := t0) (s := sh) x.id)
    s.tape.set t1
    pure { id := id }
  let cuda := do
    let t0 ← s.cudaTape.get
    let (t1, id) ← okOrThrow <|
      Runtime.Autograd.Cuda.Tape.flatten (t := t0) (s := sh) x.id
    s.cudaTape.set t1
    pure (some { id := id })
  dispatchCudaOpt (α := α) s "flatten" cpu cuda

/--
Reshape a tensor while preserving total number of elements.

PyTorch comparison: `torch.reshape` / `view` (when valid).
-/
def reshape {α : Type} (s : EagerSession α) [Inhabited α] [DecidableEq Shape] {sh1 sh2 : Shape}
  (x : TensorRef α sh1) (h : Shape.size sh1 = Shape.size sh2) : IO (TensorRef α sh2) := do
  let cpu := do
    let t0 ← s.tape.get
    let (t1, id) ← okOrThrow (Runtime.Autograd.Tape.reshape (t := t0) (s₁ := sh1) (s₂ := sh2) x.id h)
    s.tape.set t1
    pure { id := id }
  let cuda := do
    let t0 ← s.cudaTape.get
    let (t1, id) ← okOrThrow <|
      Runtime.Autograd.Cuda.Tape.reshape (t := t0) (s₁ := sh1) (s₂ := sh2) x.id h
    s.cudaTape.set t1
    pure (some { id := id })
  dispatchCudaOpt (α := α) s "reshape" cpu cuda

/-- Transpose a 2D matrix. PyTorch: `x.t()` / `x.transpose(0,1)`. -/
def transpose2d {α : Type} (s : EagerSession α) [DecidableEq Shape] {m n : Nat}
  (x : TensorRef α (.dim m (.dim n .scalar))) : IO (TensorRef α (.dim n (.dim m .scalar))) := do
  let cpu := do
    let t0 ← s.tape.get
    let (t1, id) ← okOrThrow (Runtime.Autograd.Tape.transpose2d (t := t0) (m := m) (n := n) x.id)
    s.tape.set t1
    pure { id := id }
  let cuda := do
    let t0 ← s.cudaTape.get
    let (t1, id) ← okOrThrow <|
      Runtime.Autograd.Cuda.Tape.transpose2d (t := t0) (m := m) (n := n) x.id
    s.cudaTape.set t1
    pure (some { id := id })
  dispatchCudaOpt (α := α) s "transpose2d" cpu cuda

/-- Swap two adjacent axes at a given depth. PyTorch analogue: `x.transpose(dim, dim+1)`. -/
def swapAdjacentAtDepth {α : Type} (s : EagerSession α) [DecidableEq Shape] {sh : Shape}
  (depth : Nat) (x : TensorRef α sh) : IO (TensorRef α (sh.swapAdjacentAtDepth depth)) := do
  let cpu := do
    let t0 ← s.tape.get
    let (t1, id) ← okOrThrow (Runtime.Autograd.Tape.swapAdjacentAtDepth (t := t0) (s := sh) depth
      x.id)
    s.tape.set t1
    pure { id := id }
  let cuda := do
    let t0 ← s.cudaTape.get
    let (t1, id) ← okOrThrow <|
      Runtime.Autograd.Cuda.Tape.swapAdjacentAtDepth (t := t0) (s := sh) depth x.id
    s.cudaTape.set t1
    pure (some { id := id })
  dispatchCudaOpt (α := α) s "swapAdjacentAtDepth" cpu cuda

/-- Permute a 3D tensor `(a,b,c) → (b,c,a)`. PyTorch: `x.permute(1,2,0)`. -/
def transpose3dFirstToLast {α : Type} (s : EagerSession α) [DecidableEq Shape] {a b c : Nat}
  (x : TensorRef α (.dim a (.dim b (.dim c .scalar)))) :
  IO (TensorRef α (.dim b (.dim c (.dim a .scalar)))) := do
  let cpu := do
    let t0 ← s.tape.get
    let (t1, id) ← okOrThrow (Runtime.Autograd.Tape.transpose3dFirstToLast (t := t0) (a := a) (b :=
      b) (c := c) x.id)
    s.tape.set t1
    pure { id := id }
  let cuda := do
    let t0 ← s.cudaTape.get
    let (t1, id) ← okOrThrow <|
      Runtime.Autograd.Cuda.Tape.transpose3dFirstToLast (t := t0) (a := a) (b := b) (c := c) x.id
    s.cudaTape.set t1
    pure (some { id := id })
  dispatchCudaOpt (α := α) s "transpose3d_first_to_last" cpu cuda

/-- Permute a 3D tensor `(a,b,c) → (c,a,b)`. PyTorch: `x.permute(2,0,1)`. -/
def transpose3dLastToFirst {α : Type} (s : EagerSession α) [DecidableEq Shape] {a b c : Nat}
  (x : TensorRef α (.dim a (.dim b (.dim c .scalar)))) :
  IO (TensorRef α (.dim c (.dim a (.dim b .scalar)))) := do
  let cpu := do
    let t0 ← s.tape.get
    let (t1, id) ← okOrThrow (Runtime.Autograd.Tape.transpose3dLastToFirst (t := t0) (a := a) (b :=
      b) (c := c) x.id)
    s.tape.set t1
    pure { id := id }
  let cuda := do
    let t0 ← s.cudaTape.get
    let (t1, id) ← okOrThrow <|
      Runtime.Autograd.Cuda.Tape.transpose3dLastToFirst (t := t0) (a := a) (b := b) (c := c) x.id
    s.cudaTape.set t1
    pure (some { id := id })
  dispatchCudaOpt (α := α) s "transpose3d_last_to_first" cpu cuda

/-- Swap the last two axes of a 3D tensor `(a,b,c) → (a,c,b)`. PyTorch: `x.transpose(1,2)`. -/
def transpose3dLastTwo {α : Type} (s : EagerSession α) [DecidableEq Shape] {a b c : Nat}
  (x : TensorRef α (.dim a (.dim b (.dim c .scalar)))) :
  IO (TensorRef α (.dim a (.dim c (.dim b .scalar)))) := do
  let cpu := do
    let t0 ← s.tape.get
    let (t1, id) ← okOrThrow (Runtime.Autograd.Tape.transpose3dLastTwo (t := t0) (a := a) (b := b)
      (c := c) x.id)
    s.tape.set t1
    pure { id := id }
  let cuda := do
    let t0 ← s.cudaTape.get
    let (t1, id) ← okOrThrow <|
      Runtime.Autograd.Cuda.Tape.transpose3dLastTwo (t := t0) (a := a) (b := b) (c := c) x.id
    s.cudaTape.set t1
    pure (some { id := id })
  dispatchCudaOpt (α := α) s "transpose3d_last_two" cpu cuda

/-- Broadcast a tensor to a larger shape. PyTorch: implicit broadcasting / `expand`. -/
def broadcastTo {α : Type} (s : EagerSession α) [Inhabited α] [Add α] [Zero α] [DecidableEq Shape]
  {sh1 sh2 : Shape} (cb : Shape.CanBroadcastTo sh1 sh2) (x : TensorRef α sh1) : IO (TensorRef α sh2)
    := do
  let cpu := do
    let t0 ← s.tape.get
    let (t1, id) ← okOrThrow (Runtime.Autograd.Tape.broadcastTo (α := α) (t := t0) (s₁ := sh1) (s₂ :=
      sh2) cb x.id)
    s.tape.set t1
    pure { id := id }
  let cuda := do
    let t0 ← s.cudaTape.get
    let (t1, id) ← okOrThrow <|
      Runtime.Autograd.Cuda.Tape.broadcastTo (t := t0) (s₁ := sh1) (s₂ := sh2) cb x.id
    s.cudaTape.set t1
    pure (some { id := id })
  dispatchCudaOpt (α := α) s "broadcastTo" cpu cuda

/-- Sum-reduce along `axis`. PyTorch: `torch.sum(x, dim=axis)`. -/
def reduceSum {α : Type} (s : EagerSession α) [Add α] [Zero α] [Inhabited α] [DecidableEq Shape]
  {sh : Shape} (axis : Nat) [valid : Shape.valid_axis_inst axis sh] [wf : Shape.WellFormed sh]
  (x : TensorRef α sh) : IO (TensorRef α (shapeAfterSum sh axis)) := do
  let cpu := do
    let t0 ← s.tape.get
    let (t1, id) ← okOrThrow (Runtime.Autograd.Tape.reduceSum (t := t0) (s := sh) axis x.id)
    s.tape.set t1
    pure { id := id }
  let cuda := do
    let t0 ← s.cudaTape.get
    let (t1, id) ← okOrThrow <|
      Runtime.Autograd.Cuda.Tape.reduceSum (s := sh) axis (t := t0) x.id
    s.cudaTape.set t1
    pure (some { id := id })
  dispatchCudaOpt (α := α) s "reduce_sum" cpu cuda

/-- Mean-reduce along `axis`. PyTorch: `torch.mean(x, dim=axis)`. -/
def reduceMean {α : Type} (s : EagerSession α) [Context α] [Inhabited α] [DecidableEq Shape]
  {sh : Shape} (axis : Nat) [valid : Shape.valid_axis_inst axis sh] [wf : Shape.WellFormed sh]
  (x : TensorRef α sh) : IO (TensorRef α (shapeAfterSum sh axis)) := do
  let cpu := do
    let t0 ← s.tape.get
    let (t1, id) ← okOrThrow (Runtime.Autograd.Tape.reduceMean (t := t0) (s := sh) axis x.id)
    s.tape.set t1
    pure { id := id }
  let cuda := do
    let t0 ← s.cudaTape.get
    let (t1, id) ← okOrThrow <|
      Runtime.Autograd.Cuda.Tape.reduceMean (s := sh) axis (t := t0) x.id
    s.cudaTape.set t1
    pure (some { id := id })
  dispatchCudaOpt (α := α) s "reduce_mean" cpu cuda

/-- Gather a scalar from a 1D vector with a `Fin n` index. PyTorch: `x[i]`. -/
def gatherScalar {α : Type} (s : EagerSession α) [Zero α] [DecidableEq Shape]
  {n : Nat} (x : TensorRef α (.dim n .scalar)) (i : Fin n) : IO (TensorRef α Shape.scalar) := do
  let cpu := do
    let t0 ← s.tape.get
    let (t1, id) ← okOrThrow (Runtime.Autograd.Tape.gatherScalar (t := t0) (n := n) x.id i)
    s.tape.set t1
    pure { id := id }
  let cuda := do
    let t0 ← s.cudaTape.get
    let (t1, id) ← okOrThrow <|
      Runtime.Autograd.Cuda.Tape.gatherScalar (t := t0) (n := n) x.id i
    s.cudaTape.set t1
    pure (some { id := id })
  dispatchCudaOpt (α := α) s "gather_scalar" cpu cuda

/-- Gather a row from a 2D tensor with a `Fin rows` index. PyTorch: `x[i]` for 2D tensors. -/
def gatherRow {α : Type} (s : EagerSession α) [Zero α] [DecidableEq Shape]
  {rows cols : Nat} (x : TensorRef α (.dim rows (.dim cols .scalar))) (i : Fin rows) :
  IO (TensorRef α (.dim cols .scalar)) := do
  let cpu := do
    let t0 ← s.tape.get
    let (t1, id) ← okOrThrow (Runtime.Autograd.Tape.gatherRow (t := t0) (rows := rows) (cols := cols)
      x.id i)
    s.tape.set t1
    pure { id := id }
  let cuda := do
    let t0 ← s.cudaTape.get
    let (t1, id) ← okOrThrow <|
      Runtime.Autograd.Cuda.Tape.gatherRow (t := t0) (rows := rows) (cols := cols) x.id i
    s.cudaTape.set t1
    pure (some { id := id })
  dispatchCudaOpt (α := α) s "gather_row" cpu cuda

/-- Gather a scalar from a 1D vector with a raw `Nat` index (totalized by the tape op). -/
def gatherScalarNat {α : Type} (s : EagerSession α) [Zero α] [DecidableEq Shape]
  {n : Nat} (x : TensorRef α (.dim n .scalar)) (i : Nat) : IO (TensorRef α Shape.scalar) := do
  let cpu := do
    let t0 ← s.tape.get
    let (t1, id) ← okOrThrow (Runtime.Autograd.Tape.gatherScalarNat (t := t0) (n := n) x.id i)
    s.tape.set t1
    pure { id := id }
  let cuda := do
    let t0 ← s.cudaTape.get
    let (t1, id) ← okOrThrow <|
      Runtime.Autograd.Cuda.Tape.gatherScalarNat (t := t0) (n := n) x.id i
    s.cudaTape.set t1
    pure (some { id := id })
  dispatchCudaOpt (α := α) s "gather_scalar_nat" cpu cuda

/-- Dynamic gather scalar using an index stored in `NatRef`. -/
def gatherScalarRef {α : Type} (s : EagerSession α) [Zero α] [DecidableEq Shape]
  {n : Nat} (x : TensorRef α (.dim n .scalar)) (i : NatRef) : IO (TensorRef α Shape.scalar) := do
  let idx ← getNat (α := α) s i
  gatherScalarNat (α := α) s (n := n) x idx

/-- Dynamic gather row using an index stored in `NatRef` (out-of-range gives a zero row). -/
def gatherRowRef {α : Type} [CudaBridge.TensorConv α] (s : EagerSession α) [Zero α] [DecidableEq Shape]
  {rows cols : Nat} (x : TensorRef α (.dim rows (.dim cols .scalar))) (i : NatRef) :
  IO (TensorRef α (.dim cols .scalar)) := do
  let idx ← getNat (α := α) s i
  if h : idx < rows then
    gatherRow (α := α) s (rows := rows) (cols := cols) x ⟨idx, h⟩
  else
    -- total: out-of-bounds labels map to a zero row
    const (α := α) s (sh := .dim cols .scalar) (fill (0 : α) (.dim cols .scalar)) (name := none)

/-- Gather `k` scalars using an explicit index tensor. PyTorch analogue: `gather` / advanced
  indexing. -/
def gatherVecNat {α : Type} (s : EagerSession α) [Add α] [Zero α] [DecidableEq Shape]
  {n k : Nat} (x : TensorRef α (.dim n .scalar)) (idx : Tensor Nat (.dim k .scalar)) :
  IO (TensorRef α (.dim k .scalar)) := do
  let cpu := do
    let t0 ← s.tape.get
    let (t1, id) ← okOrThrow (Runtime.Autograd.Tape.gatherVecNat (t := t0) (n := n) (k := k) x.id
      idx)
    s.tape.set t1
    pure { id := id }
  let cuda := do
    let t0 ← s.cudaTape.get
    let (t1, id) ← okOrThrow <|
      Runtime.Autograd.Cuda.Tape.gatherVecNat (t := t0) (n := n) (k := k) x.id idx
    s.cudaTape.set t1
    pure (some { id := id })
  dispatchCudaOpt (α := α) s "gather_vec_nat" cpu cuda

/-- Gather `k` rows using an explicit index tensor. PyTorch: `index_select(dim=0, index=...)`. -/
def gatherRowsNat {α : Type} (s : EagerSession α) [Add α] [Zero α] [DecidableEq Shape]
  {rows cols k : Nat} (x : TensorRef α (.dim rows (.dim cols .scalar))) (idx : Tensor Nat (.dim k
    .scalar)) :
  IO (TensorRef α (.dim k (.dim cols .scalar))) := do
  let cpu := do
    let t0 ← s.tape.get
    let (t1, id) ← okOrThrow (Runtime.Autograd.Tape.gatherRowsNat (t := t0) (rows := rows) (cols :=
      cols) (k := k) x.id idx)
    s.tape.set t1
    pure { id := id }
  let cuda := do
    let t0 ← s.cudaTape.get
    let (t1, id) ← okOrThrow <|
      Runtime.Autograd.Cuda.Tape.gatherRowsNat (t := t0) (rows := rows) (cols := cols) (k := k) x.id idx
    s.cudaTape.set t1
    pure (some { id := id })
  dispatchCudaOpt (α := α) s "gather_rows_nat" cpu cuda

/-- Gather `k` scalars using indices stored in the nat-environment (`NatVecRef`). -/
def gatherVecRef {α : Type} (s : EagerSession α) [Add α] [Zero α] [DecidableEq Shape]
  {n k : Nat} (x : TensorRef α (.dim n .scalar)) (idx : NatVecRef k) :
  IO (TensorRef α (.dim k .scalar)) := do
  let it ← getNatVec (α := α) (k := k) s idx
  gatherVecNat (α := α) s (n := n) (k := k) x it

/-- Gather `k` rows using indices stored in the nat-environment (`NatVecRef`). -/
def gatherRowsRef {α : Type} (s : EagerSession α) [Add α] [Zero α] [DecidableEq Shape]
  {rows cols k : Nat} (x : TensorRef α (.dim rows (.dim cols .scalar))) (idx : NatVecRef k) :
  IO (TensorRef α (.dim k (.dim cols .scalar))) := do
  let it ← getNatVec (α := α) (k := k) s idx
  gatherRowsNat (α := α) s (rows := rows) (cols := cols) (k := k) x it

/-- Scatter-add into a vector: return a copy of `x` with `x[i] += v`. -/
def scatterAddVec {α : Type} (s : EagerSession α) [Add α] [Zero α] [DecidableEq Shape]
  {n : Nat} (x : TensorRef α (.dim n .scalar)) (v : TensorRef α Shape.scalar) (i : Fin n) :
  IO (TensorRef α (.dim n .scalar)) := do
  let cpu := do
    let t0 ← s.tape.get
    let (t1, id) ← okOrThrow (Runtime.Autograd.Tape.scatterAddVec (t := t0) (n := n) x.id v.id i)
    s.tape.set t1
    pure { id := id }
  let cuda := do
    let t0 ← s.cudaTape.get
    let (t1, id) ← okOrThrow <|
      Runtime.Autograd.Cuda.Tape.scatterAddVec (t := t0) (n := n) x.id v.id i
    s.cudaTape.set t1
    pure (some { id := id })
  dispatchCudaOpt (α := α) s "scatter_add_vec" cpu cuda

/-- Scatter-add into a matrix row: return a copy of `x` with `x[i,:] += v`. -/
def scatterAddRow {α : Type} (s : EagerSession α) [Add α] [Zero α] [DecidableEq Shape]
  {rows cols : Nat}
  (x : TensorRef α (.dim rows (.dim cols .scalar))) (v : TensorRef α (.dim cols .scalar)) (i : Fin
    rows) :
  IO (TensorRef α (.dim rows (.dim cols .scalar))) := do
  let cpu := do
    let t0 ← s.tape.get
    let (t1, id) ← okOrThrow (Runtime.Autograd.Tape.scatterAddRow (t := t0) (rows := rows) (cols :=
      cols) x.id v.id i)
    s.tape.set t1
    pure { id := id }
  let cuda := do
    let t0 ← s.cudaTape.get
    let (t1, id) ← okOrThrow <|
      Runtime.Autograd.Cuda.Tape.scatterAddRow (t := t0) (rows := rows) (cols := cols) x.id v.id i
    s.cudaTape.set t1
    pure (some { id := id })
  dispatchCudaOpt (α := α) s "scatter_add_row" cpu cuda

/--
Fully-connected linear layer `y = w x + b` (matvec).

If `opts.fastKernels` is enabled, uses a runtime-only fast kernel implementation.
PyTorch comparison: `torch.nn.functional.linear(x, weight=w, bias=b)`.
-/
def linear {α : Type} (s : EagerSession α) [Inhabited α] [Add α] [Mul α] [Zero α] [DecidableEq
  Shape] [Runtime.Autograd.FastKernels.FastMatmul α]
  {inDim outDim : Nat}
  (w : TensorRef α (.dim outDim (.dim inDim .scalar)))
  (b : TensorRef α (.dim outDim .scalar))
  (x : TensorRef α (.dim inDim .scalar)) : IO (TensorRef α (.dim outDim .scalar)) := do
  let cpu := do
    let t0 ← s.tape.get
    let (t1, id) ←
      if s.opts.fastKernels then
        okOrThrow (Runtime.Autograd.Tape.linearFast (useGpu := s.opts.useGpu) (t := t0)
          (gpuPrecision := s.opts.fastGpuMatmulPrecision)
          (inDim := inDim) (outDim := outDim) w.id b.id x.id)
      else
        okOrThrow (Runtime.Autograd.Tape.linear (t := t0) (inDim := inDim) (outDim := outDim) w.id
          b.id x.id)
    s.tape.set t1
    pure { id := id }
  let cuda := do
    let t0 ← s.cudaTape.get
    let (t1, id) ← okOrThrow <|
      Runtime.Autograd.Cuda.Tape.linear (t := t0) (outDim := outDim) (inDim := inDim) w.id b.id x.id
    s.cudaTape.set t1
    pure (some { id := id })
  dispatchCudaOpt (α := α) s "linear" cpu cuda

/--
Mean-squared-error loss returning a scalar.

If `opts.fastKernels` is enabled, uses a runtime-only fast kernel implementation.
PyTorch comparison: `torch.nn.functional.mse_loss(..., reduction=\"mean\")`.
-/
def mseLoss {α : Type} [CudaBridge.TensorConv α] (s : EagerSession α)
  [Inhabited α] [Add α] [Sub α] [Mul α] [Div α] [Zero α] [One α] [Coe Nat α] [DecidableEq Shape]
  {sh : Shape} (yhat target : TensorRef α sh) : IO (TensorRef α Shape.scalar) := do
  let cpu := do
    let t0 ← s.tape.get
    let (t1, id) ←
      if s.opts.fastKernels then
        okOrThrow (Runtime.Autograd.Tape.mseLossFast (t := t0) (s := sh) yhat.id target.id)
      else
        okOrThrow (Runtime.Autograd.Tape.mseLoss (t := t0) (s := sh) yhat.id target.id)
    s.tape.set t1
    pure { id := id }
  let cuda := do
    let t0 ← s.cudaTape.get
    let (t1, id) ← okOrThrow <|
      Runtime.Autograd.Cuda.Tape.mseLoss (t := t0) (s := sh) yhat.id target.id
    s.cudaTape.set t1
    pure (some { id := id })
  dispatchCudaOpt (α := α) s "mse_loss" cpu cuda

/-- Layer normalization over embedding dimension. PyTorch: `nn.LayerNorm` / `functional.layer_norm`.
  -/
def layerNorm {α : Type} (s : EagerSession α) [Context α]
  [DecidableRel ((· > ·) : α → α → Prop)] [DecidableEq Shape]
  {seqLen embedDim : Nat} (h_seq_pos : seqLen > 0) (h_embed_pos : embedDim > 0)
  (x : TensorRef α (.dim seqLen (.dim embedDim .scalar)))
  (gamma : TensorRef α (.dim embedDim .scalar))
  (beta : TensorRef α (.dim embedDim .scalar)) : IO (TensorRef α (.dim seqLen (.dim embedDim
    .scalar))) := do
  let cpu := do
    let t0 ← s.tape.get
    let (t1, id) ← okOrThrow (Runtime.Autograd.Tape.layerNorm (t := t0)
      (seqLen := seqLen) (embedDim := embedDim) (h_seq_pos := h_seq_pos) (h_embed_pos := h_embed_pos)
      x.id gamma.id beta.id)
    s.tape.set t1
    pure { id := id }
  let cuda := do
    let t0 ← s.cudaTape.get
    let (t1, id) ← okOrThrow (Runtime.Autograd.Cuda.Tape.layerNorm (t := t0)
      (seqLen := seqLen) (embedDim := embedDim) (h_seq_pos := h_seq_pos) (h_embed_pos := h_embed_pos)
      x.id gamma.id beta.id)
    s.cudaTape.set t1
    pure (some { id := id })
  dispatchCudaOpt (α := α) s "layer_norm" cpu cuda

/-- BatchNorm for channel-first images `(C,H,W)` (no batch axis). PyTorch: `nn.BatchNorm2d`
  (conceptually). -/
def batchnormChannelFirst {α : Type} (s : EagerSession α) [Context α]
  [DecidableRel ((· > ·) : α → α → Prop)] [DecidableEq Shape]
  {channels height width : Nat} (h_c : channels > 0) (h_h : height > 0) (h_w : width > 0)
  (x : TensorRef α (.dim channels (.dim height (.dim width .scalar))))
  (gamma : TensorRef α (.dim channels .scalar))
  (beta : TensorRef α (.dim channels .scalar)) : IO (TensorRef α (.dim channels (.dim height (.dim
    width .scalar)))) := do
  let cpu := do
    let t0 ← s.tape.get
    let (t1, id) ← okOrThrow (Runtime.Autograd.Tape.batchnormChannelFirst (t := t0)
      (channels := channels) (height := height) (width := width) (h_c := h_c) (h_h := h_h) (h_w :=
        h_w)
      x.id gamma.id beta.id)
    s.tape.set t1
    pure { id := id }
  let cuda := do
    let t0 ← s.cudaTape.get
    let (t1, id) ← okOrThrow (Runtime.Autograd.Cuda.Tape.batchnormChannelFirst (t := t0)
      (channels := channels) (height := height) (width := width) (h_c := h_c) (h_h := h_h)
      (h_w := h_w)
      x.id gamma.id beta.id)
    s.cudaTape.set t1
    pure (some { id := id })
  dispatchCudaOpt (α := α) s "batchnorm_channel_first" cpu cuda

/-- Multi-head self-attention (typed, proof-friendly). PyTorch: `nn.MultiheadAttention`
  (conceptually). -/
def multiHeadAttention {α : Type} (s : EagerSession α) [Context α]
  [DecidableRel ((· > ·) : α → α → Prop)] [DecidableEq Shape]
  {n numHeads dModel headDim : Nat} (h1 : n ≠ 0)
  (wq : TensorRef α (.dim dModel (.dim (numHeads * headDim) .scalar)))
  (wk : TensorRef α (.dim dModel (.dim (numHeads * headDim) .scalar)))
  (wv : TensorRef α (.dim dModel (.dim (numHeads * headDim) .scalar)))
  (wo : TensorRef α (.dim (numHeads * headDim) (.dim dModel .scalar)))
  (x : TensorRef α (.dim n (.dim dModel .scalar)))
  (mask : Option (Tensor Bool (.dim n (.dim n .scalar))) := none) :
  IO (TensorRef α (.dim n (.dim dModel .scalar))) := do
  let cpu := do
    let t0 ← s.tape.get
    let (t1, id) ← okOrThrow (Runtime.Autograd.Tape.multiHeadAttention (t := t0)
      (n := n) (numHeads := numHeads) (dModel := dModel) (headDim := headDim) (h1 := h1)
      wq.id wk.id wv.id wo.id x.id mask)
    s.tape.set t1
    pure { id := id }
  let cuda := do
    let t0 ← s.cudaTape.get
    let (t1, id) ← okOrThrow (Runtime.Autograd.Cuda.Tape.multiHeadAttention (t := t0)
      (n := n) (numHeads := numHeads) (dModel := dModel) (headDim := headDim) (h1 := h1)
      wq.id wk.id wv.id wo.id x.id (mask := mask) (useFlash := s.opts.fastKernels))
    s.cudaTape.set t1
    pure (some { id := id })
  dispatchCudaOpt (α := α) s "multi_head_attention" cpu cuda

/--
N-D convolution for channels-first tensors `(inC, spatial...)` (no batch axis).

This is the generic counterpart to `conv2d`.
PyTorch comparison: `torch.nn.functional.conv{d}d` specialized to a single sample.
-/
def conv {α : Type} (s : EagerSession α) [Context α]
  [DecidableEq Shape]
  {d inC outC : Nat}
  {kernel stride padding : Vector Nat d}
  {inSpatial : Vector Nat d}
  {hInC : inC ≠ 0} {hKernel : ∀ i : Fin d, kernel.get i ≠ 0}
  (w : TensorRef α (Shape.ofList (outC :: inC :: kernel.toList)))
  (b : TensorRef α (.dim outC .scalar))
  (x : TensorRef α (Shape.ofList (inC :: inSpatial.toList))) :
  IO (TensorRef α
    (Shape.ofList (outC :: (Spec.convOutSpatial inSpatial kernel stride padding).toList))) := do
  let cpu := do
    let t0 ← s.tape.get
    let (t1, id) ← okOrThrow (Runtime.Autograd.Tape.conv (t := t0)
      (d := d) (inC := inC) (outC := outC)
      (kernel := kernel) (stride := stride) (padding := padding) (inSpatial := inSpatial)
      w.id b.id x.id)
    s.tape.set t1
    pure { id := id }
  let cuda := do
    let t0 ← s.cudaTape.get
    let (t1, id) ← okOrThrow (Runtime.Autograd.Cuda.Tape.conv (t := t0)
      (d := d) (inC := inC) (outC := outC)
      (kernel := kernel) (stride := stride) (padding := padding) (inSpatial := inSpatial)
      w.id b.id x.id
      (hInC := hInC) (hKernel := hKernel))
    s.cudaTape.set t1
    pure (some { id := id })
  dispatchCudaOpt (α := α) s "conv" cpu cuda

/-- 2D convolution for channel-first images `(C,H,W)` (no batch axis). PyTorch:
  `torch.nn.functional.conv2d`. -/
def conv2d {α : Type} (s : EagerSession α) [Context α]
  [DecidableRel ((· > ·) : α → α → Prop)] [DecidableEq Shape]
  {inC outC kH kW stride padding inH inW : Nat}
  {h1 : inC ≠ 0} {h2 : kH ≠ 0} {h3 : kW ≠ 0}
  (kernel : TensorRef α (.dim outC (.dim inC (.dim kH (.dim kW .scalar)))))
  (bias : TensorRef α (.dim outC .scalar))
  (input : TensorRef α (.dim inC (.dim inH (.dim inW .scalar)))) :
  IO (TensorRef α (.dim outC (.dim ((inH + 2 * padding - kH) / stride + 1)
    (.dim ((inW + 2 * padding - kW) / stride + 1) .scalar)))) := do
  let cpu := do
    let t0 ← s.tape.get
    let (t1, id) ← okOrThrow (Runtime.Autograd.Tape.conv2d (t := t0)
      (inC := inC) (outC := outC) (kH := kH) (kW := kW) (stride := stride) (padding := padding)
      (inH := inH) (inW := inW) (h1 := h1) (h2 := h2) (h3 := h3)
      kernel.id bias.id input.id)
    s.tape.set t1
    pure { id := id }
  let cuda := do
    let t0 ← s.cudaTape.get
    let (t1, id) ← okOrThrow <|
      Runtime.Autograd.Cuda.Tape.conv2d (t := t0)
        (inC := inC) (outC := outC) (kH := kH) (kW := kW) (stride := stride) (padding := padding)
        (inH := inH) (inW := inW)
        (h1 := h1) (h2 := h2) (h3 := h3)
        kernel.id bias.id input.id
    s.cudaTape.set t1
    pure (some { id := id })
  dispatchCudaOpt (α := α) s "conv2d" cpu cuda

/--
N-D transpose convolution for channels-first tensors `(inC, spatial...)` (no batch axis).

This is the generic counterpart to `conv_transpose2d`.
PyTorch comparison: `torch.nn.functional.conv_transpose{d}d` specialized to a single sample.
-/
def convTranspose {α : Type} (s : EagerSession α) [Context α]
  [DecidableEq Shape]
  {d inC outC : Nat}
  {kernel stride padding : Vector Nat d}
  {inSpatial : Vector Nat d}
  {hInC : inC ≠ 0} {hKernel : ∀ i : Fin d, kernel.get i ≠ 0}
  (w : TensorRef α (Shape.ofList (inC :: outC :: kernel.toList)))
  (b : TensorRef α (.dim outC .scalar))
  (x : TensorRef α (Shape.ofList (inC :: inSpatial.toList))) :
  IO (TensorRef α
    (Shape.ofList (outC :: (Spec.convTransposeOutSpatial inSpatial kernel stride padding).toList)))
    := do
  let cpu := do
    let t0 ← s.tape.get
    let (t1, id) ← okOrThrow (Runtime.Autograd.Tape.convTranspose (t := t0)
      (d := d) (inC := inC) (outC := outC)
      (kernel := kernel) (stride := stride) (padding := padding) (inSpatial := inSpatial)
      w.id b.id x.id)
    s.tape.set t1
    pure { id := id }
  let cuda := do
    let t0 ← s.cudaTape.get
    let (t1, id) ← okOrThrow (Runtime.Autograd.Cuda.Tape.convTranspose (t := t0)
      (d := d) (inC := inC) (outC := outC)
      (kernel := kernel) (stride := stride) (padding := padding) (inSpatial := inSpatial)
      w.id b.id x.id
      (hInC := hInC) (hKernel := hKernel))
    s.cudaTape.set t1
    pure (some { id := id })
  dispatchCudaOpt (α := α) s "conv_transpose" cpu cuda

/-- 2D transpose convolution for channel-first images `(C,H,W)` (no batch axis). PyTorch:
  `torch.nn.functional.conv_transpose2d`. -/
def convTranspose2d {α : Type} (s : EagerSession α) [Context α]
  [DecidableEq Shape]
  {inC outC kH kW stride padding inH inW : Nat}
  {h1 : inC ≠ 0} {h2 : kH ≠ 0} {h3 : kW ≠ 0}
  (kernel : TensorRef α (.dim inC (.dim outC (.dim kH (.dim kW .scalar)))))
  (bias : TensorRef α (.dim outC .scalar))
  (input : TensorRef α (.dim inC (.dim inH (.dim inW .scalar)))) :
  IO (TensorRef α (.dim outC (.dim ((inH - 1) * stride - 2 * padding + kH)
    (.dim ((inW - 1) * stride - 2 * padding + kW) .scalar)))) := do
  let cpu := do
    let t0 ← s.tape.get
    let (t1, id) ← okOrThrow (Runtime.Autograd.Tape.convTranspose2d (t := t0)
      (inC := inC) (outC := outC) (kH := kH) (kW := kW) (stride := stride) (padding := padding)
      (inH := inH) (inW := inW) (h1 := h1) (h2 := h2) (h3 := h3)
      kernel.id bias.id input.id)
    s.tape.set t1
    pure { id := id }
  let cuda := do
    let t0 ← s.cudaTape.get
    let (t1, id) ← okOrThrow <|
      Runtime.Autograd.Cuda.Tape.convTranspose2d (t := t0)
        (inC := inC) (outC := outC) (kH := kH) (kW := kW) (stride := stride) (padding := padding)
        (inH := inH) (inW := inW)
        (h1 := h1) (h2 := h2) (h3 := h3)
        kernel.id bias.id input.id
    s.cudaTape.set t1
    pure (some { id := id })
  dispatchCudaOpt (α := α) s "conv_transpose2d" cpu cuda

/-- Alias for `conv2d` (compat shorthand). -/
abbrev conv2dCompat {α : Type} := conv2d (α := α)

/-- 2D matrix multiplication. PyTorch: `torch.matmul` for 2D tensors. -/
def matmul {α : Type} (s : EagerSession α) [Context α]
  [DecidableRel ((· > ·) : α → α → Prop)] [DecidableEq Shape]
  {m n p : Nat}
  (a : TensorRef α (.dim m (.dim n .scalar)))
  (b : TensorRef α (.dim n (.dim p .scalar))) :
  IO (TensorRef α (.dim m (.dim p .scalar))) := do
  let cpu := do
    let t0 ← s.tape.get
    let (t1, id) ←
      if s.opts.fastKernels then
        okOrThrow (Runtime.Autograd.Tape.matmulFast (useGpu := s.opts.useGpu)
          (gpuPrecision := s.opts.fastGpuMatmulPrecision)
          (t := t0) (m := m) (n := n) (p := p) a.id b.id)
      else
        okOrThrow (Runtime.Autograd.Tape.matmul (t := t0) (m := m) (n := n) (p := p) a.id b.id)
    s.tape.set t1
    pure { id := id }
  let cuda := do
    let t0 ← s.cudaTape.get
    let (t1, id) ← okOrThrow <|
      Runtime.Autograd.Cuda.Tape.matmul (t := t0) (m := m) (n := n) (p := p) a.id b.id
    s.cudaTape.set t1
    pure (some { id := id })
  dispatchCudaOpt (α := α) s "matmul" cpu cuda

/-- Batched matrix multiplication. PyTorch: `torch.bmm`. -/
def bmm {α : Type} (s : EagerSession α) [Add α] [Mul α] [Zero α] [DecidableEq Shape]
  {batch m n p : Nat}
  (a : TensorRef α (.dim batch (.dim m (.dim n .scalar))))
  (b : TensorRef α (.dim batch (.dim n (.dim p .scalar)))) :
  IO (TensorRef α (.dim batch (.dim m (.dim p .scalar)))) := do
  let cpu := do
    let t0 ← s.tape.get
    let (t1, id) ← okOrThrow (Runtime.Autograd.Tape.bmm (α := α) (t := t0) (batch := batch) (m := m)
      (n := n) (p := p) a.id b.id)
    s.tape.set t1
    pure { id := id }
  let cuda := do
    let t0 ← s.cudaTape.get
    let (t1, id) ← okOrThrow <|
      Runtime.Autograd.Cuda.Tape.bmm (t := t0) (batch := batch) (m := m) (n := n) (p := p) a.id b.id
    s.cudaTape.set t1
    pure (some { id := id })
  dispatchCudaOpt (α := α) s "bmm" cpu cuda

/-- Concatenate two vectors along dim 0. PyTorch: `torch.cat([a,b], dim=0)`. -/
def concatVectors {α : Type} (s : EagerSession α) [Context α]
  [DecidableRel ((· > ·) : α → α → Prop)] [DecidableEq Shape]
  {n m : Nat}
  (a : TensorRef α (.dim n .scalar))
  (b : TensorRef α (.dim m .scalar)) :
  IO (TensorRef α (.dim (n + m) .scalar)) := do
  let cpu := do
    let t0 ← s.tape.get
    let (t1, id) ← okOrThrow (Runtime.Autograd.Tape.concatVectors (t := t0) (n := n) (m := m) a.id
      b.id)
    s.tape.set t1
    pure { id := id }
  let cuda := do
    let t0 ← s.cudaTape.get
    let (t1, id) ← okOrThrow <|
      Runtime.Autograd.Cuda.Tape.concatVectors (t := t0) (n := n) (m := m) a.id b.id
    s.cudaTape.set t1
    pure (some { id := id })
  dispatchCudaOpt (α := α) s "concat_vectors" cpu cuda

/-- Concatenate along dim 0 for tensors with leading dimension. PyTorch: `torch.cat(..., dim=0)`. -/
def concatDim0 {α : Type} (s : EagerSession α) [DecidableEq Shape]
  {n m : Nat} {sh : Shape}
  (a : TensorRef α (.dim n sh))
  (b : TensorRef α (.dim m sh)) :
  IO (TensorRef α (.dim (n + m) sh)) := do
  let cpu := do
    let t0 ← s.tape.get
    let (t1, id) ← okOrThrow (Runtime.Autograd.Tape.concatDim0 (α := α) (t := t0) (n := n) (m := m)
      (s := sh) a.id b.id)
    s.tape.set t1
    pure { id := id }
  let cuda := do
    let t0 ← s.cudaTape.get
    let (t1, id) ← okOrThrow <|
      Runtime.Autograd.Cuda.Tape.concatDim0 (t := t0) (n := n) (m := m) (s := sh) a.id b.id
    s.cudaTape.set t1
    pure (some { id := id })
  dispatchCudaOpt (α := α) s "concat_dim0" cpu cuda

/-- Slice along dim 0: `x[start:start+len]`. PyTorch: standard slicing. -/
def sliceRange0 {α : Type} (s : EagerSession α) [Zero α] [DecidableEq Shape]
  {n : Nat} {sh : Shape}
  (x : TensorRef α (.dim n sh)) (start len : Nat) (h : len + start ≤ n) :
  IO (TensorRef α (.dim len sh)) := do
  let cpu := do
    let t0 ← s.tape.get
    let (t1, id) ← okOrThrow (Runtime.Autograd.Tape.sliceRange0 (α := α) (t := t0) (n := n) (s := sh)
      x.id start len h)
    s.tape.set t1
    pure { id := id }
  let cuda := do
    let t0 ← s.cudaTape.get
    let (t1, id) ← okOrThrow <|
      Runtime.Autograd.Cuda.Tape.sliceRange0 (t := t0) (n := n) (s := sh) x.id start len h
    s.cudaTape.set t1
    pure (some { id := id })
  dispatchCudaOpt (α := α) s "slice_range0" cpu cuda

/--
N-D max pooling for channels-first tensors `(C, spatial...)` (no batch axis).

PyTorch comparison: `torch.nn.functional.max_pool1d` / `max_pool2d` / `max_pool3d` depending on the
spatial rank `d`.
-/
def maxPool {α : Type} (s : EagerSession α) [Context α] [DecidableEq Shape]
  {d C : Nat} {inSpatial kernel stride padding : Vector Nat d}
  {hKernel : ∀ i : Fin d, kernel.get i ≠ 0}
  (x : TensorRef α (Shape.ofList (C :: inSpatial.toList))) :
  IO (TensorRef α
    (Shape.ofList (C :: (Spec.poolOutSpatialPad inSpatial kernel stride padding).toList))) := do
  let cpu := do
    let t0 ← s.tape.get
    let (t1, id) ← okOrThrow (Runtime.Autograd.Tape.maxPool (t := t0)
      (d := d) (C := C)
      (inSpatial := inSpatial) (kernel := kernel) (stride := stride) (padding := padding)
      (hKernel := hKernel) x.id)
    s.tape.set t1
    pure { id := id }
  let cuda := do
    let t0 ← s.cudaTape.get
    let (t1, id) ← okOrThrow (Runtime.Autograd.Cuda.Tape.maxPool (t := t0)
      (d := d) (C := C)
      (inSpatial := inSpatial) (kernel := kernel) (stride := stride) (padding := padding)
      (hKernel := hKernel) x.id)
    s.cudaTape.set t1
    pure (some { id := id })
  dispatchCudaOpt (α := α) s "max_pool" cpu cuda

/--
N-D average pooling for channels-first tensors `(C, spatial...)` (no batch axis).

PyTorch comparison: `torch.nn.functional.avg_pool1d` / `avg_pool2d` / `avg_pool3d` depending on the
spatial rank `d`.
-/
def avgPool {α : Type} (s : EagerSession α) [Context α] [DecidableEq Shape]
  {d C : Nat} {inSpatial kernel stride padding : Vector Nat d}
  (hKernel : ∀ i : Fin d, kernel.get i ≠ 0)
  (x : TensorRef α (Shape.ofList (C :: inSpatial.toList))) :
  IO (TensorRef α
    (Shape.ofList (C :: (Spec.poolOutSpatialPad inSpatial kernel stride padding).toList))) := do
  let cpu := do
    let t0 ← s.tape.get
    let (t1, id) ← okOrThrow (Runtime.Autograd.Tape.avgPool (t := t0)
      (d := d) (C := C)
      (inSpatial := inSpatial) (kernel := kernel) (stride := stride) (padding := padding)
      hKernel x.id)
    s.tape.set t1
    pure { id := id }
  let cuda := do
    let t0 ← s.cudaTape.get
    let (t1, id) ← okOrThrow (Runtime.Autograd.Cuda.Tape.avgPool (t := t0)
      (d := d) (C := C)
      (inSpatial := inSpatial) (kernel := kernel) (stride := stride) (padding := padding)
      hKernel x.id)
    s.cudaTape.set t1
    pure (some { id := id })
  dispatchCudaOpt (α := α) s "avg_pool" cpu cuda

/--
N-D smooth max pooling (log-sum-exp surrogate) for channels-first tensors `(C, spatial...)`.

This is a differentiable approximation to max pooling; PyTorch does not expose it as a single
primitive, but it can be emulated with `logsumexp` over local windows.
-/
def smoothMaxPool {α : Type} [CudaBridge.TensorConv α] (s : EagerSession α) [Context α]
  [DecidableEq Shape]
  {d C : Nat} {inSpatial kernel stride padding : Vector Nat d}
  {hKernel : ∀ i : Fin d, kernel.get i ≠ 0}
  (x : TensorRef α (Shape.ofList (C :: inSpatial.toList))) (beta : α) :
  IO (TensorRef α
    (Shape.ofList (C :: (Spec.poolOutSpatialPad inSpatial kernel stride padding).toList))) := do
  let cpu := do
    let t0 ← s.tape.get
    let (t1, id) ← okOrThrow (Runtime.Autograd.Tape.smoothMaxPool (t := t0)
      (d := d) (C := C)
      (inSpatial := inSpatial) (kernel := kernel) (stride := stride) (padding := padding)
      (hKernel := hKernel) x.id beta)
    s.tape.set t1
    pure { id := id }
  let cuda := do
    let betaF ← CudaBridge.TensorConv.toFloat (α := α) beta
    let t0 ← s.cudaTape.get
    let (t1, id) ← okOrThrow (Runtime.Autograd.Cuda.Tape.smoothMaxPool (t := t0)
      (d := d) (C := C)
      (inSpatial := inSpatial) (kernel := kernel) (stride := stride) (padding := padding)
      (hKernel := hKernel) x.id betaF)
    s.cudaTape.set t1
    pure (some { id := id })
  dispatchCudaOpt (α := α) s "smooth_max_pool" cpu cuda

/-- 2D max-pooling (no batch axis). PyTorch: `torch.nn.functional.max_pool2d`. -/
def maxPool2d {α : Type} (s : EagerSession α) [Context α] [DecidableEq Shape]
  {kH kW inH inW inC stride : Nat} {h1 : kH ≠ 0} {h2 : kW ≠ 0}
  (x : TensorRef α (.dim inC (.dim inH (.dim inW .scalar)))) :
  IO (TensorRef α (.dim inC (.dim ((inH - kH) / stride + 1) (.dim ((inW - kW) / stride + 1)
    .scalar)))) := do
  let cpu := do
    let t0 ← s.tape.get
    let (t1, id) ← okOrThrow (Runtime.Autograd.Tape.maxPool2d (t := t0)
      (kH := kH) (kW := kW) (inH := inH) (inW := inW) (inC := inC) (stride := stride)
      (h1 := h1) (h2 := h2) x.id)
    s.tape.set t1
    pure { id := id }
  let cuda := do
    let inCU32 : UInt32 := UInt32.ofNat inC
    let inHU32 : UInt32 := UInt32.ofNat inH
    let inWU32 : UInt32 := UInt32.ofNat inW
    let kHU32 : UInt32 := UInt32.ofNat kH
    let kWU32 : UInt32 := UInt32.ofNat kW
    let strideU32 : UInt32 := UInt32.ofNat stride
    let paddingU32 : UInt32 := 0
    let outSh : Shape :=
      .dim inC (.dim ((inH - kH) / stride + 1) (.dim ((inW - kW) / stride + 1) .scalar))
    let t0 ← s.cudaTape.get
    let (t1, id) ← okOrThrow <|
      Runtime.Autograd.Cuda.Tape.unary (t := t0) "max_pool2d" x.id
        (.dim inC (.dim inH (.dim inW .scalar))) outSh
        (forward := fun xBuf =>
          Runtime.Autograd.Cuda.torchleanMaxPool2dFwdCuda xBuf inCU32 inHU32 inWU32 kHU32 kWU32 strideU32 paddingU32)
        (backward := fun xBuf dLdy =>
          Runtime.Autograd.Cuda.torchleanMaxPool2dBwdCuda xBuf dLdy inCU32 inHU32 inWU32 kHU32 kWU32 strideU32 paddingU32)
    s.cudaTape.set t1
    pure (some { id := id })
  dispatchCudaOpt (α := α) s "max_pool2d" cpu cuda

/-- 2D max-pooling with padding (no batch axis). PyTorch: `max_pool2d(..., padding=...)`. -/
def maxPool2dPad {α : Type} (s : EagerSession α) [Context α] [DecidableEq Shape]
  {kH kW inH inW inC stride padding : Nat} {h1 : kH ≠ 0} {h2 : kW ≠ 0}
  (x : TensorRef α (.dim inC (.dim inH (.dim inW .scalar)))) :
  IO (TensorRef α (.dim inC (.dim ((inH + 2 * padding - kH) / stride + 1)
    (.dim ((inW + 2 * padding - kW) / stride + 1) .scalar)))) := do
  let cpu := do
    let t0 ← s.tape.get
    let (t1, id) ← okOrThrow (Runtime.Autograd.Tape.maxPool2dPad (t := t0)
      (kH := kH) (kW := kW) (inH := inH) (inW := inW) (inC := inC) (stride := stride) (padding :=
        padding)
      (h1 := h1) (h2 := h2) x.id)
    s.tape.set t1
    pure { id := id }
  let cuda := do
    let inCU32 : UInt32 := UInt32.ofNat inC
    let inHU32 : UInt32 := UInt32.ofNat inH
    let inWU32 : UInt32 := UInt32.ofNat inW
    let kHU32 : UInt32 := UInt32.ofNat kH
    let kWU32 : UInt32 := UInt32.ofNat kW
    let strideU32 : UInt32 := UInt32.ofNat stride
    let paddingU32 : UInt32 := UInt32.ofNat padding
    let outSh : Shape :=
      .dim inC (.dim ((inH + 2 * padding - kH) / stride + 1)
        (.dim ((inW + 2 * padding - kW) / stride + 1) .scalar))
    let t0 ← s.cudaTape.get
    let (t1, id) ← okOrThrow <|
      Runtime.Autograd.Cuda.Tape.unary (t := t0) "max_pool2d_pad" x.id
        (.dim inC (.dim inH (.dim inW .scalar))) outSh
        (forward := fun xBuf =>
          Runtime.Autograd.Cuda.torchleanMaxPool2dFwdCuda xBuf inCU32 inHU32 inWU32 kHU32 kWU32 strideU32 paddingU32)
        (backward := fun xBuf dLdy =>
          Runtime.Autograd.Cuda.torchleanMaxPool2dBwdCuda xBuf dLdy inCU32 inHU32 inWU32 kHU32 kWU32 strideU32 paddingU32)
    s.cudaTape.set t1
    pure (some { id := id })
  dispatchCudaOpt (α := α) s "max_pool2d_pad" cpu cuda

/-- Alias for `max_pool2d_pad` (PyTorch-style shorthand). -/
abbrev maxPoolPad {α : Type} := maxPool2dPad (α := α)

/-- Smooth max-pooling (softmax pooling). Not a standard PyTorch primitive; see
  `Torch.LinkedSession.smooth_max_pool2d`. -/
def smoothMaxPool2d {α : Type} [CudaBridge.TensorConv α] (s : EagerSession α) [Context α]
  [DecidableEq Shape]
  {kH kW inH inW inC stride : Nat} {h1 : kH ≠ 0} {h2 : kW ≠ 0}
  (x : TensorRef α (.dim inC (.dim inH (.dim inW .scalar)))) (beta : α) :
  IO (TensorRef α (.dim inC (.dim ((inH - kH) / stride + 1) (.dim ((inW - kW) / stride + 1)
    .scalar)))) := do
  let cpu := do
    let t0 ← s.tape.get
    let (t1, id) ← okOrThrow (Runtime.Autograd.Tape.smoothMaxPool2d (t := t0)
      (kH := kH) (kW := kW) (inH := inH) (inW := inW) (inC := inC) (stride := stride)
      (h1 := h1) (h2 := h2) x.id beta)
    s.tape.set t1
    pure { id := id }
  let cuda := do
    let betaF ← CudaBridge.TensorConv.toFloat (α := α) beta
    let t0 ← s.cudaTape.get
    let (t1, id) ← okOrThrow <|
      Runtime.Autograd.Cuda.Tape.smoothMaxPool2d (t := t0)
        (kH := kH) (kW := kW) (inH := inH) (inW := inW) (inC := inC) (stride := stride)
        (h1 := h1) (h2 := h2) x.id betaF
    s.cudaTape.set t1
    pure (some { id := id })
  dispatchCudaOpt (α := α) s "smooth_max_pool2d" cpu cuda

/-- 2D average-pooling (no batch axis). PyTorch: `torch.nn.functional.avg_pool2d`. -/
def avgPool2d {α : Type} (s : EagerSession α) [Context α] [DecidableEq Shape]
  {kH kW inH inW inC stride : Nat} (h1 : kH ≠ 0) (h2 : kW ≠ 0)
  (x : TensorRef α (.dim inC (.dim inH (.dim inW .scalar)))) :
  IO (TensorRef α (.dim inC (.dim ((inH - kH) / stride + 1) (.dim ((inW - kW) / stride + 1)
    .scalar)))) := do
  let cpu := do
    let t0 ← s.tape.get
    let (t1, id) ← okOrThrow (Runtime.Autograd.Tape.avgPool2d (t := t0)
      (kH := kH) (kW := kW) (inH := inH) (inW := inW) (inC := inC) (stride := stride)
      (h1 := h1) (h2 := h2) x.id)
    s.tape.set t1
    pure { id := id }
  let cuda := do
    let t0 ← s.cudaTape.get
    let (t1, id) ← okOrThrow (Runtime.Autograd.Cuda.Tape.avgPool2d (t := t0)
      (kH := kH) (kW := kW) (inH := inH) (inW := inW) (inC := inC) (stride := stride)
      h1 h2 x.id)
    s.cudaTape.set t1
    pure (some { id := id })
  dispatchCudaOpt (α := α) s "avg_pool2d" cpu cuda

/-- 2D average-pooling with padding (no batch axis). PyTorch: `avg_pool2d(..., padding=...)`. -/
def avgPool2dPad {α : Type} (s : EagerSession α) [Context α] [DecidableEq Shape]
  {kH kW inH inW inC stride padding : Nat} (h1 : kH ≠ 0) (h2 : kW ≠ 0)
  (x : TensorRef α (.dim inC (.dim inH (.dim inW .scalar)))) :
  IO (TensorRef α (.dim inC (.dim ((inH + 2 * padding - kH) / stride + 1)
    (.dim ((inW + 2 * padding - kW) / stride + 1) .scalar)))) := do
  let cpu := do
    let t0 ← s.tape.get
    let (t1, id) ← okOrThrow (Runtime.Autograd.Tape.avgPool2dPad (t := t0)
      (kH := kH) (kW := kW) (inH := inH) (inW := inW) (inC := inC) (stride := stride) (padding :=
        padding)
      (h1 := h1) (h2 := h2) x.id)
    s.tape.set t1
    pure { id := id }
  let cuda := do
    let t0 ← s.cudaTape.get
    let (t1, id) ← okOrThrow (Runtime.Autograd.Cuda.Tape.avgPool2dPad (t := t0)
      (kH := kH) (kW := kW) (inH := inH) (inW := inW) (inC := inC) (stride := stride) (padding :=
        padding)
      h1 h2 x.id)
    s.cudaTape.set t1
    pure (some { id := id })
  dispatchCudaOpt (α := α) s "avg_pool2d_pad" cpu cuda

/-- Alias for `avg_pool2d_pad` (PyTorch-style shorthand). -/
abbrev avgPoolPad {α : Type} := avgPool2dPad (α := α)

/--
Run reverse-mode backprop on the CUDA tape, returning device gradients for all tape entries.

This is the CUDA analogue of `backwardDenseAll`, but it does *not* download gradients back to the
host. This is primarily useful for implementing GPU-native optimizer steps.
-/
def backwardDenseAllCuda {α : Type} [CudaBridge.TensorConv α] (s : EagerSession α) [Add α] [Zero α]
  [DecidableEq Shape]
  {sh : Shape} (out : TensorRef α sh) (seed : Tensor α sh) :
  IO (Array Runtime.Autograd.Cuda.AnyBuffer) := do
  if Options.device s.opts != .cuda then
    throw <| IO.userError "torch: backwardDenseAllCuda called on non-CUDA eager session"
  let t ← s.cudaTape.get
  let seedAny ← CudaBridge.TensorConv.toAnyBuffer (α := α) (s := sh) seed
  okOrThrow <|
    Runtime.Autograd.Cuda.Tape.backwardDenseAll (t := t) (outId := out.id) (seed := seedAny)

/-- Convenience wrapper for scalar losses on CUDA: backward with seed `1` (device buffers). -/
def backwardScalarDenseAllCuda {α : Type} [CudaBridge.TensorConv α] (s : EagerSession α) [Add α]
  [Zero α] [One α] [DecidableEq Shape]
  (loss : TensorRef α Shape.scalar) : IO (Array Runtime.Autograd.Cuda.AnyBuffer) := do
  backwardDenseAllCuda (α := α) s (sh := Shape.scalar) loss (Tensor.scalar (1 : α))

/--
Run reverse-mode backprop and return a dense gradient array for all tape entries.

`seed` is the upstream gradient for `out` (like PyTorch's `backward(gradient=...)`).
-/
def backwardDenseAll {α : Type} [CudaBridge.TensorConv α] (s : EagerSession α) [Add α] [Zero α]
  [DecidableEq Shape]
  {sh : Shape} (out : TensorRef α sh) (seed : Tensor α sh) :
  IO (Array (Runtime.AnyTensor α)) := do
  if Options.device s.opts == .cuda then
    let gradsDev ← backwardDenseAllCuda (α := α) s (sh := sh) out seed
    gradsDev.mapM (fun g => CudaBridge.TensorConv.ofAnyBuffer (α := α) g)
  else
    let t ← s.tape.get
    okOrThrow (Runtime.Autograd.Tape.backwardDenseAll (t := t) (outId := out.id)
      (seed := Runtime.Autograd.AnyTensor.mk seed))

/--
Convenience wrapper for scalar losses: run backward with seed `1`.

PyTorch comparison: `loss.backward()` for a scalar loss.
-/
def backwardScalarDenseAll {α : Type} [CudaBridge.TensorConv α] (s : EagerSession α) [Add α]
  [Zero α] [One α] [DecidableEq Shape]
  (loss : TensorRef α Shape.scalar) : IO (Array (Runtime.AnyTensor α)) := do
  backwardDenseAll (α := α) s (sh := Shape.scalar) loss (Tensor.scalar (1 : α))

/--
Extract the gradient for a particular `TensorRef` from a dense gradient array.
-/
def grad {α : Type} {sh : Shape} [DecidableEq Shape]
  (grads : Array (Runtime.AnyTensor α)) (x : TensorRef α sh) : IO (Tensor α sh) := do
  let gAny ← match grads[x.id]? with
    | some g => pure g
    | none => throw <| IO.userError "torch: gradient array out of bounds"
  if h : gAny.s = sh then
    pure (Tensor.castShape gAny.t h)
  else
    throw <| IO.userError
      s!"torch: grad shape mismatch (expected {Shape.pretty sh}, got {Shape.pretty gAny.s})"

/--
Apply an SGD update to all parameters recorded via `use`.

PyTorch comparison: `for p in params: p.data -= lr * p.grad`.
-/
def sgdStepAll {α : Type} [CudaBridge.TensorConv α] (s : EagerSession α)
  [Sub α] [Mul α] [Add α] [Zero α] [DecidableEq Shape]
  (lr : α) (grads : Array (Runtime.AnyTensor α)) : IO Unit := do
  if Options.device s.opts == .cuda then
    let lrF ← CudaBridge.TensorConv.toFloat (α := α) lr
    let t0 ← s.cudaTape.get
    let m ← s.paramsByLeaf.get
    for (id, p) in m.toList do
      let gAny ← match grads[id]? with
        | some g => pure g
        | none => throw <| IO.userError "torch: gradient array out of bounds during SGD"
      if hs : gAny.s = p.s then
        let gT : Tensor α p.s := Tensor.castShape gAny.t hs
        let gDev ← CudaBridge.TensorConv.toAnyBuffer (α := α) (s := p.s) gT
        let pBuf ← okOrThrow <|
          Runtime.Autograd.Cuda.Tape.requireValue (t := t0) (id := id) (s := p.s)
        let updatedDev : Runtime.Autograd.Cuda.AnyBuffer :=
          { s := p.s, buf := Runtime.Autograd.Cuda.Buffer.axpy pBuf gDev.buf (-lrF) }
        p.setCuda updatedDev
      else
        throw <| IO.userError "torch: internal grad shape mismatch during SGD"
  else
    let m ← s.paramsByLeaf.get
    for (id, p) in m.toList do
      let gAny ← match grads[id]? with
        | some g => pure g
        | none => throw <| IO.userError "torch: gradient array out of bounds during SGD"
      if hs : gAny.s = p.s then
        let pv ← p.get
        if hp : pv.s = p.s then
          let pvT : Tensor α p.s := Tensor.castShape pv.t hp
          let gT : Tensor α p.s := Tensor.castShape gAny.t hs
          let updated : Tensor α p.s :=
            Tensor.materialize <| subSpec pvT (scaleSpec (α := α) (s := p.s) gT lr)
          p.set (Runtime.Autograd.AnyTensor.mk updated)
        else
          throw <| IO.userError "torch: internal param shape mismatch"
      else
        throw <| IO.userError "torch: internal grad shape mismatch during SGD"

/--
Apply an SGD update to all parameters recorded via `use`, using CUDA device gradients.

This avoids downloading the full dense gradient array and keeps updated parameters in each
`Param`'s CUDA mirror. Host tensors are synchronized later by explicit parameter readback.
-/
def sgdStepAllCuda {α : Type} [CudaBridge.TensorConv α] (s : EagerSession α) [DecidableEq Shape]
  (lr : α) (grads : Array Runtime.Autograd.Cuda.AnyBuffer) : IO Unit := do
  if Options.device s.opts != .cuda then
    throw <| IO.userError "torch: sgdStepAllCuda called on non-CUDA eager session"
  let lrF ← CudaBridge.TensorConv.toFloat (α := α) lr
  let t0 ← s.cudaTape.get
  let m ← s.paramsByLeaf.get
  for (id, p) in m.toList do
    let gAny ← match grads[id]? with
      | some g => pure g
      | none => throw <| IO.userError "torch: gradient array out of bounds during SGD"
    if _hs : gAny.s = p.s then
      let pBuf ← okOrThrow <|
        Runtime.Autograd.Cuda.Tape.requireValue (t := t0) (id := id) (s := p.s)
      let updatedDev : Runtime.Autograd.Cuda.AnyBuffer :=
        { s := p.s, buf := Runtime.Autograd.Cuda.Buffer.axpy pBuf gAny.buf (-lrF) }
      p.setCuda updatedDev
    else
      throw <| IO.userError "torch: internal grad shape mismatch during SGD"

/-- Device-side Adam moment buffers for one parameter leaf. -/
structure CudaAdamParamState where
  /-- First moment buffer. -/
  m : Runtime.Autograd.Cuda.Buffer
  /-- Second moment buffer. -/
  v : Runtime.Autograd.Cuda.Buffer
  /-- Adam step counter for this parameter. -/
  t : Nat

abbrev CudaAdamState := Std.HashMap Nat CudaAdamParamState

/--
Apply an Adam update to all parameters recorded via `use`, using CUDA device gradients.

This is the CUDA analogue of the generic `TorchLean.Optim.adam` path.  It keeps Adam moments as
device buffers and keeps updated parameters in each `Param`'s CUDA mirror, so the next CUDA forward
can reuse them without a host upload. Host tensors are synchronized later by explicit readback.
-/
def adamStepAllCuda {α : Type} [CudaBridge.TensorConv α] (s : EagerSession α) [DecidableEq Shape]
    (stateRef : IO.Ref CudaAdamState)
    (lr beta1 beta2 epsilon : α)
    (grads : Array Runtime.Autograd.Cuda.AnyBuffer) : IO Unit := do
  if Options.device s.opts != .cuda then
    throw <| IO.userError "torch: adamStepAllCuda called on non-CUDA eager session"
  let lrF ← CudaBridge.TensorConv.toFloat (α := α) lr
  let beta1F ← CudaBridge.TensorConv.toFloat (α := α) beta1
  let beta2F ← CudaBridge.TensorConv.toFloat (α := α) beta2
  let epsF ← CudaBridge.TensorConv.toFloat (α := α) epsilon
  let oneMinusBeta1 := 1.0 - beta1F
  let oneMinusBeta2 := 1.0 - beta2F
  let t0 ← s.cudaTape.get
  let params ← s.paramsByLeaf.get
  let mut state ← stateRef.get
  for (id, p) in params.toList do
    let gAny ← match grads[id]? with
      | some g => pure g
      | none => throw <| IO.userError "torch: gradient array out of bounds during CUDA Adam"
    if _hs : gAny.s = p.s then
      let pBuf ← okOrThrow <|
        Runtime.Autograd.Cuda.Tape.requireValue (t := t0) (id := id) (s := p.s)
      let n := Runtime.Autograd.Cuda.Buffer.size pBuf
      let st :=
        match state.get? id with
        | some st => st
        | none =>
            { m := Runtime.Autograd.Cuda.Buffer.zeros n
              v := Runtime.Autograd.Cuda.Buffer.zeros n
              t := 0 }
      let t' := st.t + 1
      let mScaled := Runtime.Autograd.Cuda.Buffer.scale st.m beta1F
      let m' := Runtime.Autograd.Cuda.Buffer.axpy mScaled gAny.buf oneMinusBeta1
      let g2 := Runtime.Autograd.Cuda.Buffer.mul gAny.buf gAny.buf
      let vScaled := Runtime.Autograd.Cuda.Buffer.scale st.v beta2F
      let v' := Runtime.Autograd.Cuda.Buffer.axpy vScaled g2 oneMinusBeta2
      let mHatScale := 1.0 / (1.0 - Float.pow beta1F (Float.ofNat t'))
      let vHatScale := 1.0 / (1.0 - Float.pow beta2F (Float.ofNat t'))
      let mHat := Runtime.Autograd.Cuda.Buffer.scale m' mHatScale
      let vHat := Runtime.Autograd.Cuda.Buffer.scale v' vHatScale
      let sqrtVHat := Runtime.Autograd.Cuda.Buffer.sqrt vHat
      let epsBuf := Runtime.Autograd.Cuda.Buffer.full n epsF
      let denom :=
        Runtime.Autograd.Cuda.Buffer.add sqrtVHat epsBuf
      let update := Runtime.Autograd.Cuda.Buffer.div mHat denom
      let updatedDev : Runtime.Autograd.Cuda.AnyBuffer :=
        { s := p.s, buf := Runtime.Autograd.Cuda.Buffer.axpy pBuf update (-lrF) }
      p.setCuda updatedDev
      releaseCudaBuffer st.m
      releaseCudaBuffer st.v
      releaseCudaBuffer mScaled
      releaseCudaBuffer g2
      releaseCudaBuffer vScaled
      releaseCudaBuffer mHat
      releaseCudaBuffer vHat
      releaseCudaBuffer sqrtVHat
      releaseCudaBuffer epsBuf
      releaseCudaBuffer denom
      releaseCudaBuffer update
      state := state.insert id { m := m', v := v', t := t' }
    else
      throw <| IO.userError "torch: internal grad shape mismatch during CUDA Adam"
  for (id, st) in state.toList do
    if params.contains id then
      pure ()
    else
      releaseCudaBuffer st.m
      releaseCudaBuffer st.v
      state := state.erase id
  stateRef.set state

/--
Apply an AdamW update to all parameters recorded via `use`, using CUDA device gradients.

This mirrors `Optim.AdamW.update`: moments are formed from the raw gradient, weight decay is applied
directly to parameters, then the Adam update is applied. Like `adamStepAllCuda`, it keeps updated
parameter buffers resident on device and only synchronizes the host copy when readback is requested.
-/
def adamWStepAllCuda {α : Type} [CudaBridge.TensorConv α] (s : EagerSession α) [DecidableEq Shape]
    (stateRef : IO.Ref CudaAdamState)
    (lr weightDecay beta1 beta2 epsilon : α)
    (grads : Array Runtime.Autograd.Cuda.AnyBuffer) : IO Unit := do
  if Options.device s.opts != .cuda then
    throw <| IO.userError "torch: adamWStepAllCuda called on non-CUDA eager session"
  let lrF ← CudaBridge.TensorConv.toFloat (α := α) lr
  let wdF ← CudaBridge.TensorConv.toFloat (α := α) weightDecay
  let beta1F ← CudaBridge.TensorConv.toFloat (α := α) beta1
  let beta2F ← CudaBridge.TensorConv.toFloat (α := α) beta2
  let epsF ← CudaBridge.TensorConv.toFloat (α := α) epsilon
  let oneMinusBeta1 := 1.0 - beta1F
  let oneMinusBeta2 := 1.0 - beta2F
  let t0 ← s.cudaTape.get
  let params ← s.paramsByLeaf.get
  let mut state ← stateRef.get
  for (id, p) in params.toList do
    let gAny ← match grads[id]? with
      | some g => pure g
      | none => throw <| IO.userError "torch: gradient array out of bounds during CUDA AdamW"
    if _hs : gAny.s = p.s then
      let pBuf ← okOrThrow <|
        Runtime.Autograd.Cuda.Tape.requireValue (t := t0) (id := id) (s := p.s)
      let n := Runtime.Autograd.Cuda.Buffer.size pBuf
      let st :=
        match state.get? id with
        | some st => st
        | none =>
            { m := Runtime.Autograd.Cuda.Buffer.zeros n
              v := Runtime.Autograd.Cuda.Buffer.zeros n
              t := 0 }
      let t' := st.t + 1
      let mScaled := Runtime.Autograd.Cuda.Buffer.scale st.m beta1F
      let m' := Runtime.Autograd.Cuda.Buffer.axpy mScaled gAny.buf oneMinusBeta1
      let g2 := Runtime.Autograd.Cuda.Buffer.mul gAny.buf gAny.buf
      let vScaled := Runtime.Autograd.Cuda.Buffer.scale st.v beta2F
      let v' := Runtime.Autograd.Cuda.Buffer.axpy vScaled g2 oneMinusBeta2
      let mHatScale := 1.0 / (1.0 - Float.pow beta1F (Float.ofNat t'))
      let vHatScale := 1.0 / (1.0 - Float.pow beta2F (Float.ofNat t'))
      let mHat := Runtime.Autograd.Cuda.Buffer.scale m' mHatScale
      let vHat := Runtime.Autograd.Cuda.Buffer.scale v' vHatScale
      let sqrtVHat := Runtime.Autograd.Cuda.Buffer.sqrt vHat
      let epsBuf := Runtime.Autograd.Cuda.Buffer.full n epsF
      let denom :=
        Runtime.Autograd.Cuda.Buffer.add sqrtVHat epsBuf
      let update := Runtime.Autograd.Cuda.Buffer.div mHat denom
      let decayedParam := Runtime.Autograd.Cuda.Buffer.axpy pBuf pBuf (-(lrF * wdF))
      let updatedDev : Runtime.Autograd.Cuda.AnyBuffer :=
        { s := p.s, buf := Runtime.Autograd.Cuda.Buffer.axpy decayedParam update (-lrF) }
      p.setCuda updatedDev
      releaseCudaBuffer st.m
      releaseCudaBuffer st.v
      releaseCudaBuffer mScaled
      releaseCudaBuffer g2
      releaseCudaBuffer vScaled
      releaseCudaBuffer mHat
      releaseCudaBuffer vHat
      releaseCudaBuffer sqrtVHat
      releaseCudaBuffer epsBuf
      releaseCudaBuffer denom
      releaseCudaBuffer update
      releaseCudaBuffer decayedParam
      state := state.insert id { m := m', v := v', t := t' }
    else
      throw <| IO.userError "torch: internal grad shape mismatch during CUDA AdamW"
  for (id, st) in state.toList do
    if params.contains id then
      pure ()
    else
      releaseCudaBuffer st.m
      releaseCudaBuffer st.v
      state := state.erase id
  stateRef.set state

end EagerSession

end Internal

/-!
Imperative sessions live in:
- `Runtime.Autograd.TorchLean.Session` (unified eager/compiled, recommended),
- `Runtime.Autograd.Torch.Internal.SessionIR` (proof-linked imperative session, internal).

`torch.compile`-style wrapper (cached static graph).

This is a thin wrapper around the proof-compiled graph model (`GraphData`) and its proven-correct
reverse-mode accumulator (`GraphData.backpropCtx`). It compiles once, then you can call
`forward/backward` repeatedly with new inputs.

Note: this does *not* cache a `Runtime.Autograd.Tape` for reuse across different inputs.
The current tape compiler bakes the forward context into backward closures, so reusing a single
tape across changing inputs would be unsound without redesigning the runtime node API.
-/

/--
`torch.compile`-style wrapper for a scalar-valued computation over leaf context `Γ`.

This stores a *proved* node (`NodeData`) together with the preceding graph prefix so it can be
evaluated and differentiated without rebuilding the whole graph each time.
-/
structure CompiledScalar (α : Type) (Γ : List Shape) where
  /-- Shapes of internal SSA nodes preceding the scalar output node. -/
  ssPrev : List Shape
  /-- Proved graph prefix that computes all preceding SSA nodes. -/
  gPrev : Proofs.Autograd.Algebra.GraphData α Unit Γ ssPrev
  /-- Final scalar output node over the leaf context plus graph prefix. -/
  node : Proofs.Autograd.Algebra.NodeData α Unit (Γ ++ ssPrev) Shape.scalar

namespace CompiledScalar

/-- Convenience alias for the proved heterogeneous tensor list over a shape context. -/
abbrev TList (α : Type) (ss : List Shape) := Proofs.Autograd.Algebra.TList α ss

/-- Evaluate the scalar output for leaf values `x`. -/
def forward {α : Type} {Γ : List Shape}
  (c : CompiledScalar α Γ) (x : TList α Γ) : Tensor α Shape.scalar :=
  c.node.forward (Proofs.Autograd.Algebra.GraphData.eval (g := c.gPrev) x ()) ()

/-- Forward-mode Jacobian-vector product (JVP) at `x` with tangent `dx`. -/
def jvp {α : Type} {Γ : List Shape}
  (c : CompiledScalar α Γ) (x dx : TList α Γ) : Tensor α Shape.scalar :=
  let ctx := Proofs.Autograd.Algebra.GraphData.eval (g := c.gPrev) x ()
  let dctx := Proofs.Autograd.Algebra.GraphData.jvpCtx (g := c.gPrev) x dx ()
  c.node.jvp ctx dctx ()

/--
Reverse-mode backprop for a scalar output with implicit seed `1`.

Returns a `TList` of gradients aligned with the leaf context `Γ`.
-/
def backward {α : Type} [Add α] [Zero α] [One α]
  {Γ : List Shape} (c : CompiledScalar α Γ) (x : TList α Γ) : TList α Γ :=
  let ssPrev := c.ssPrev
  let full : Proofs.Autograd.Algebra.GraphData α Unit Γ (ssPrev ++ [Shape.scalar]) :=
    .snoc (ss := ssPrev) c.gPrev c.node
  let seedPrev : TList α (Γ ++ ssPrev) := TList.zero (α := α) (ss := Γ ++ ssPrev)
  let seed' : TList α ((Γ ++ ssPrev) ++ [Shape.scalar]) :=
    TList.snoc (α := α) (ss := Γ ++ ssPrev) (τ := Shape.scalar) seedPrev (Tensor.scalar (1 : α))
  let seed : TList α (Γ ++ (ssPrev ++ [Shape.scalar])) :=
    TList.cast (α := α) (h := List.append_assoc Γ ssPrev [Shape.scalar]) seed'
  Proofs.Autograd.Algebra.GraphData.backpropCtx (α := α) (Δ := Unit) (Γ := Γ) (g := full) x () seed

/--
Reverse-mode backprop for a scalar output with an explicit scalar seed.

PyTorch comparison: `loss.backward(gradient=seedOut)` for a scalar loss.
-/
def backwardWithSeed {α : Type} [Add α] [Zero α]
    {Γ : List Shape} (c : CompiledScalar α Γ) (x : TList α Γ) (seedOut : α) : TList α Γ :=
  let ssPrev := c.ssPrev
  let full : Proofs.Autograd.Algebra.GraphData α Unit Γ (ssPrev ++ [Shape.scalar]) :=
    .snoc (ss := ssPrev) c.gPrev c.node
  let seedPrev : TList α (Γ ++ ssPrev) := TList.zero (α := α) (ss := Γ ++ ssPrev)
  let seed' : TList α ((Γ ++ ssPrev) ++ [Shape.scalar]) :=
    TList.snoc (α := α) (ss := Γ ++ ssPrev) (τ := Shape.scalar) seedPrev (Tensor.scalar seedOut)
  let seed : TList α (Γ ++ (ssPrev ++ [Shape.scalar])) :=
    TList.cast (α := α) (h := List.append_assoc Γ ssPrev [Shape.scalar]) seed'
  Proofs.Autograd.Algebra.GraphData.backpropCtx (α := α) (Δ := Unit) (Γ := Γ) (g := full) x () seed

end CompiledScalar

/-!
`torch.compile`-style wrapper for tensor-valued outputs.

This is the same idea as `CompiledScalar`, but parameterized by an arbitrary output shape `τ`.
It supports:
- `forward` (evaluate the output),
- `jvp` (forward-mode JVP, provided all ops supply `jvp`),
- `vjpWithSeed` (reverse-mode VJP with an explicit cotangent seed at the output).
-/

/--
`torch.compile`-style wrapper for a tensor-valued output of shape `τ`.

This generalizes `CompiledScalar` to arbitrary output shapes and provides forward-mode JVP and
reverse-mode VJP (with explicit seed).
-/
structure CompiledOut (α : Type) (Γ : List Shape) (τ : Shape) where
  /-- Shapes of internal SSA nodes preceding the output node. -/
  ssPrev : List Shape
  /-- Proved graph prefix that computes all preceding SSA nodes. -/
  gPrev : Proofs.Autograd.Algebra.GraphData α Unit Γ ssPrev
  /-- Final output node over the leaf context plus graph prefix. -/
  node : Proofs.Autograd.Algebra.NodeData α Unit (Γ ++ ssPrev) τ

namespace CompiledOut

/-- Convenience alias for the proved heterogeneous tensor list over a shape context. -/
abbrev TList (α : Type) (ss : List Shape) := Proofs.Autograd.Algebra.TList α ss

/-- Evaluate the output tensor for leaf values `x`. -/
def forward {α : Type} {Γ : List Shape} {τ : Shape}
  (c : CompiledOut α Γ τ) (x : TList α Γ) : Tensor α τ :=
  c.node.forward (Proofs.Autograd.Algebra.GraphData.eval (g := c.gPrev) x ()) ()

/-- Forward-mode Jacobian-vector product (JVP) at `x` with tangent `dx`. -/
def jvp {α : Type} {Γ : List Shape} {τ : Shape}
  (c : CompiledOut α Γ τ) (x dx : TList α Γ) : Tensor α τ :=
  let ctx := Proofs.Autograd.Algebra.GraphData.eval (g := c.gPrev) x ()
  let dctx := Proofs.Autograd.Algebra.GraphData.jvpCtx (g := c.gPrev) x dx ()
  c.node.jvp ctx dctx ()

/--
Reverse-mode vector-Jacobian product (VJP) with an explicit output cotangent seed.

This is the tensor-valued analogue of `CompiledScalar.backwardWithSeed`.
PyTorch comparison: `out.backward(gradient=seedOut)` (for a tensor output).
-/
def vjpWithSeed {α : Type} [Add α] [Zero α]
    {Γ : List Shape} {τ : Shape}
    (c : CompiledOut α Γ τ) (x : TList α Γ) (seedOut : Tensor α τ) : TList α Γ :=
  let ssPrev := c.ssPrev
  let full : Proofs.Autograd.Algebra.GraphData α Unit Γ (ssPrev ++ [τ]) :=
    .snoc (ss := ssPrev) c.gPrev c.node
  let seedPrev : TList α (Γ ++ ssPrev) := TList.zero (α := α) (ss := Γ ++ ssPrev)
  let seed' : TList α ((Γ ++ ssPrev) ++ [τ]) :=
    TList.snoc (α := α) (ss := Γ ++ ssPrev) (τ := τ) seedPrev seedOut
  let seed : TList α (Γ ++ (ssPrev ++ [τ])) :=
    TList.cast (α := α) (h := List.append_assoc Γ ssPrev [τ]) seed'
  Proofs.Autograd.Algebra.GraphData.backpropCtx (α := α) (Δ := Unit) (Γ := Γ) (g := full) x () seed

end CompiledOut

/--
Compile a scalar-output graph builder into a `CompiledScalar`.

The builder is expressed in the `Compiled.GraphM` monad. We expect it to produce at least one node
and return a variable of scalar shape.
-/
def compileScalar {α : Type} [DecidableEq Shape] {Γ : List Shape}
  (build : Runtime.Autograd.Compiled.GraphM.M α Γ (Runtime.Autograd.Compiled.GraphM.Var
    Shape.scalar)) :
  Runtime.Autograd.Result (CompiledScalar α Γ) := do
  let (outVar, st) ← Runtime.Autograd.Compiled.GraphM.run (α := α) (Γ := Γ) build
  match st with
  | ⟨_ss, g⟩ =>
      match g with
      | .nil =>
          .error "torch.compile: graph produced no nodes (need a scalar output node)"
      | .snoc (ss := ssPrev) (τ := τ) gPrev node =>
          match τ with
          | .scalar =>
              let expectedOutId := Γ.length + ssPrev.length
              if _h : outVar.id = expectedOutId then
                .ok { ssPrev := ssPrev, gPrev := gPrev, node := node }
              else
                .error
                  (s!"torch.compile: output Var is not the last node (got id={outVar.id}, " ++
                    s!"expected id={expectedOutId})")
          | _ =>
              .error "torch.compile: output node is not scalar (expected Shape.scalar)"

/--
Compile a tensor-output graph builder into a `CompiledOut`.

We require that the returned `Var τ` is the *last* node produced by the builder, so the wrapper can
store the prefix graph and final output node cleanly.
-/
def compileOut {α : Type} [DecidableEq Shape] {Γ : List Shape} {τ : Shape}
  (build : Runtime.Autograd.Compiled.GraphM.M α Γ (Runtime.Autograd.Compiled.GraphM.Var τ)) :
  Runtime.Autograd.Result (CompiledOut α Γ τ) := do
  let (outVar, st) ← Runtime.Autograd.Compiled.GraphM.run (α := α) (Γ := Γ) build
  match st with
  | ⟨_ss, g⟩ =>
      match g with
      | .nil =>
          .error "torch.compileOut: graph produced no nodes (need an explicit output node)"
      | .snoc (ss := ssPrev) (τ := τ') gPrev node =>
          let expectedOutId := Γ.length + ssPrev.length
          if _hOut : outVar.id = expectedOutId then
            if hτ : τ' = τ then
              match hτ with
              | rfl => .ok { ssPrev := ssPrev, gPrev := gPrev, node := node }
            else
              .error
                (s!"torch.compileOut: output node shape mismatch (expected " ++
                  s!"{Shape.pretty τ}, got {Shape.pretty τ'})")
          else
            .error
              (s!"torch.compileOut: output Var is not the last node (got " ++
                s!"id={outVar.id}, expected id={expectedOutId})")

/-
Backend-generic "one API" layer

The eager backend builds a runtime tape each iteration.
The `GraphM` authoring API provides a proof-compiled model.

The definitions below let you write a single model/loss once (as a polymorphic program over a
small `Ops` interface) and then choose:
- `backend := .eager`    (build a tape each iteration)
- `backend := .compiled` (compile once, run many)
-/

namespace Proofs.Autograd.Algebra.TList

/--
Append two `TList`s.

This is a small utility for bridging between curried APIs and list-of-shapes APIs.
-/
def append {α : Type} : {ss₁ ss₂ : List Shape} → TList α ss₁ → TList α ss₂ → TList α (ss₁ ++ ss₂)
  | [], _ss₂, .nil, ys => ys
  | _s :: ss₁, ss₂, .cons x xs, ys => .cons x (append (ss₁ := ss₁) (ss₂ := ss₂) xs ys)

/--
Split a `TList α (ss₁ ++ ss₂)` into its left and right parts.

This is the inverse of `TList.append`.
-/
def splitAppend {α : Type} : {ss₁ ss₂ : List Shape} → TList α (ss₁ ++ ss₂) → TList α ss₁ × TList α
  ss₂
  | [], _ss₂, xs => (.nil, xs)
  | _s :: ss₁, ss₂, .cons x xs =>
      let (xs₁, xs₂) := splitAppend (α := α) (ss₁ := ss₁) (ss₂ := ss₂) xs
      (.cons x xs₁, xs₂)

end Proofs.Autograd.Algebra.TList

namespace Curried

/--
Type of a curried function accepting one tensor argument per shape in `ss`.

For example, `Fn α [s₁, s₂] β` is `Tensor α s₁ → Tensor α s₂ → β`.
-/
def Fn (α : Type) : List Shape → Type → Type
  | [], β => β
  | s :: ss, β => Tensor α s → Fn α ss β

/-- Convert a function on `TList` inputs into its curried form. -/
def curry {α : Type} {β : Type} : {ss : List Shape} → (TList α ss → β) → Fn α ss β
  | [], f => f .nil
  | _s :: ss, f => fun x => curry (ss := ss) (fun xs => f (.cons x xs))

/-- Convert a curried function into a function on `TList` inputs. -/
def uncurry {α : Type} {β : Type} : {ss : List Shape} → Fn α ss β → TList α ss → β
  | [], f, .nil => f
  | _s :: ss, f, .cons x xs => uncurry (ss := ss) (f x) xs

end Curried

/-!
`RefList` is the reference-analogue of `TList`: a heterogeneous list of `Ref s` values indexed by
a shape list.

This is used to write backend-generic code over references (e.g. `TensorRef`s in eager mode, or
`GraphM.Var`s in compiled mode).
-/
/-- Reference-analogue of `TList`: a heterogeneous list of `Ref s` values indexed by shapes. -/
inductive RefList (Ref : Shape → Type) : List Shape → Type where
  | nil : RefList Ref []
  | cons {s : Shape} {ss : List Shape} : Ref s → RefList Ref ss → RefList Ref (s :: ss)

namespace RefList

/-- Append two `RefList`s. -/
def append {Ref : Shape → Type} : {ss₁ ss₂ : List Shape} → RefList Ref ss₁ → RefList Ref ss₂ →
  RefList Ref (ss₁ ++ ss₂)
  | [], _ss₂, .nil, ys => ys
  | _s :: ss₁, ss₂, .cons x xs, ys => .cons x (append (ss₁ := ss₁) (ss₂ := ss₂) xs ys)

/-- Split a `RefList Ref (ss₁ ++ ss₂)` into its left and right parts. -/
def split {Ref : Shape → Type} : {ss₁ ss₂ : List Shape} →
    RefList Ref (ss₁ ++ ss₂) → RefList Ref ss₁ × RefList Ref ss₂
  | [], _ss₂, xs => (.nil, xs)
  | _s :: ss₁, ss₂, .cons x xs =>
      let (l, r) := split (Ref := Ref) (ss₁ := ss₁) (ss₂ := ss₂) xs
      (.cons x l, r)

/-- Split a `RefList Ref (ss ++ [τ])` into its prefix and last element. -/
def splitAppend1 {Ref : Shape → Type} : {ss : List Shape} → {τ : Shape} →
    RefList Ref (ss ++ [τ]) → RefList Ref ss × Ref τ
  | [], _τ, .cons x .nil => (.nil, x)
  | _s :: ss, τ, .cons x xs =>
      let (l, last) := splitAppend1 (Ref := Ref) (ss := ss) (τ := τ) xs
      (.cons x l, last)

end RefList

/--
Type of a curried function over references, one `Ref s` argument per shape in `ss`.

This mirrors `Curried.Fn`, but for `Ref`-valued arguments (e.g. `TensorRef`s in eager mode or
`GraphM.Var`s in compiled mode).
-/
def CurriedRef (Ref : Shape → Type) : List Shape → Type → Type
  | [], β => β
  | s :: ss, β => Ref s → CurriedRef Ref ss β

namespace CurriedRef

/-- Uncurry a curried reference function to accept a `RefList`. -/
def uncurry {Ref : Shape → Type} {β : Type} : {ss : List Shape} → CurriedRef Ref ss β → RefList Ref
  ss → β
  | [], f, .nil => f
  | _s :: ss, f, .cons x xs => uncurry (ss := ss) (f x) xs

/-- Curry a reference function that consumes a `RefList`. -/
def curry {Ref : Shape → Type} {β : Type} : {ss : List Shape} → (RefList Ref ss → β) → CurriedRef
  Ref ss β
  | [], f => f .nil
  | _s :: ss, f => fun x => curry (ss := ss) (fun xs => f (.cons x xs))

/--
Apply a curried reference function to a `GraphM.VarList`.

This is a convenience for the compiled backend, where leaves/inputs are represented as `Var`s.
-/
def applyVarList {Γ : List Shape} {β : Type} :
    CurriedRef (fun s => Runtime.Autograd.Compiled.GraphM.Var s) Γ β →
      Runtime.Autograd.Compiled.GraphM.VarList Γ → β
  | f, .nil => f
  | f, .cons v vs => applyVarList (Γ := _) (β := β) (f v) vs

end CurriedRef

/--
Backend-generic interface for building and executing tensor programs.

This typeclass lets you write a single model/loss once (polymorphic over `Ops m α`) and then choose:
- an eager backend that executes immediately on a runtime tape, or
- a compiled backend that records proved IR (`GraphM`) for later compilation/proofs.

Each method corresponds to a Tensor op; implementations are expected to match the semantics of the
corresponding `Runtime.Autograd.Tape.*` / `Compiled.GraphM.*` operator.
-/
class Ops (m : Type → Type) (α : Type) [Context α] [DecidableEq Shape] where
  Ref : Shape → Type
  const : {s : Shape} → Tensor α s → m (Ref s)
  add : {s : Shape} → Ref s → Ref s → m (Ref s)
  sub : {s : Shape} → Ref s → Ref s → m (Ref s)
  mul : {s : Shape} → Ref s → Ref s → m (Ref s)
  scale : {s : Shape} → Ref s → α → m (Ref s)
  abs : {s : Shape} → Ref s → m (Ref s)
  sqrt : {s : Shape} → Ref s → m (Ref s)
  clamp : {s : Shape} → Ref s → α → α → m (Ref s)
  max : {s : Shape} → Ref s → Ref s → m (Ref s)
  min : {s : Shape} → Ref s → Ref s → m (Ref s)
  broadcastTo : {s₁ s₂ : Shape} → Shape.CanBroadcastTo s₁ s₂ → Ref s₁ → m (Ref s₂)
  reshape : {s₁ s₂ : Shape} → Ref s₁ → (h : Shape.size s₁ = Shape.size s₂) → m (Ref s₂)
  transpose2d {mDim nDim : Nat} : Ref (.dim mDim (.dim nDim .scalar)) → m (Ref (.dim nDim (.dim mDim
    .scalar)))
  transpose3dFirstToLast {a b c : Nat} :
    Ref (.dim a (.dim b (.dim c .scalar))) → m (Ref (.dim b (.dim c (.dim a .scalar))))
  transpose3dLastToFirst {a b c : Nat} :
    Ref (.dim a (.dim b (.dim c .scalar))) → m (Ref (.dim c (.dim a (.dim b .scalar))))
  transpose3dLastTwo {a b c : Nat} :
    Ref (.dim a (.dim b (.dim c .scalar))) → m (Ref (.dim a (.dim c (.dim b .scalar))))
  swapAdjacentAtDepth {s : Shape} : (depth : Nat) → Ref s → m (Ref (s.swapAdjacentAtDepth depth))
  reduceSum {s : Shape} (axis : Nat) [Shape.valid_axis_inst axis s] [Shape.WellFormed s] :
    Ref s → m (Ref (shapeAfterSum s axis))
  reduceMean {s : Shape} (axis : Nat) [Shape.valid_axis_inst axis s] [Shape.WellFormed s] :
    Ref s → m (Ref (shapeAfterSum s axis))
  gatherScalar {n : Nat} : Ref (.dim n .scalar) → Fin n → m (Ref Shape.scalar)
  gatherRow {rows cols : Nat} : Ref (.dim rows (.dim cols .scalar)) → Fin rows → m (Ref (.dim cols
    .scalar))
  gatherScalarNat {n : Nat} : Ref (.dim n .scalar) → Nat → m (Ref Shape.scalar)
  gatherVecNat {n k : Nat} : Ref (.dim n .scalar) → Tensor Nat (.dim k .scalar) → m (Ref (.dim k
    .scalar))
  gatherRowsNat {rows cols k : Nat} :
    Ref (.dim rows (.dim cols .scalar)) → Tensor Nat (.dim k .scalar) → m (Ref (.dim k (.dim cols
      .scalar)))
  scatterAddVec {n : Nat} : Ref (.dim n .scalar) → Ref Shape.scalar → Fin n → m (Ref (.dim n
    .scalar))
  scatterAddRow {rows cols : Nat} :
    Ref (.dim rows (.dim cols .scalar)) → Ref (.dim cols .scalar) → Fin rows → m (Ref (.dim rows
      (.dim cols .scalar)))
  matmul {mDim nDim pDim : Nat} :
    Ref (.dim mDim (.dim nDim .scalar)) →
    Ref (.dim nDim (.dim pDim .scalar)) →
    m (Ref (.dim mDim (.dim pDim .scalar)))
  bmm {batch mDim nDim pDim : Nat} :
    Ref (.dim batch (.dim mDim (.dim nDim .scalar))) →
    Ref (.dim batch (.dim nDim (.dim pDim .scalar))) →
    m (Ref (.dim batch (.dim mDim (.dim pDim .scalar))))
  concatVectors {nDim mDim : Nat} :
    Ref (.dim nDim .scalar) →
    Ref (.dim mDim .scalar) →
    m (Ref (.dim (nDim + mDim) .scalar))
  concatDim0 {nDim mDim : Nat} {s : Shape} :
    Ref (.dim nDim s) →
    Ref (.dim mDim s) →
    m (Ref (.dim (nDim + mDim) s))
  sliceRange0 {nDim : Nat} {s : Shape} :
    (start len : Nat) → (h : len + start ≤ nDim) →
    Ref (.dim nDim s) → m (Ref (.dim len s))
  maxPool {d C : Nat}
    {inSpatial kernel stride padding : Vector Nat d}
    {hKernel : ∀ i : Fin d, kernel.get i ≠ 0} :
    Ref (Shape.ofList (C :: inSpatial.toList)) →
    m (Ref (Shape.ofList (C :: (Spec.poolOutSpatialPad inSpatial kernel stride padding).toList)))
  avgPool {d C : Nat}
    {inSpatial kernel stride padding : Vector Nat d}
    (hKernel : ∀ i : Fin d, kernel.get i ≠ 0) :
    Ref (Shape.ofList (C :: inSpatial.toList)) →
    m (Ref (Shape.ofList (C :: (Spec.poolOutSpatialPad inSpatial kernel stride padding).toList)))
  smoothMaxPool {d C : Nat}
    {inSpatial kernel stride padding : Vector Nat d}
    {hKernel : ∀ i : Fin d, kernel.get i ≠ 0} :
    Ref (Shape.ofList (C :: inSpatial.toList)) →
    α →
    m (Ref (Shape.ofList (C :: (Spec.poolOutSpatialPad inSpatial kernel stride padding).toList)))
  maxPool2d {kH kW inH inW inC stride : Nat} {h1 : kH ≠ 0} {h2 : kW ≠ 0} :
    Ref (.dim inC (.dim inH (.dim inW .scalar))) →
    m (Ref (.dim inC (.dim ((inH - kH) / stride + 1) (.dim ((inW - kW) / stride + 1) .scalar))))
  maxPool2dPad {kH kW inH inW inC stride padding : Nat} {h1 : kH ≠ 0} {h2 : kW ≠ 0} :
    Ref (.dim inC (.dim inH (.dim inW .scalar))) →
    m (Ref (.dim inC (.dim ((inH + 2 * padding - kH) / stride + 1)
      (.dim ((inW + 2 * padding - kW) / stride + 1) .scalar))))
  smoothMaxPool2d {kH kW inH inW inC stride : Nat} {h1 : kH ≠ 0} {h2 : kW ≠ 0} :
    Ref (.dim inC (.dim inH (.dim inW .scalar))) →
    α →
    m (Ref (.dim inC (.dim ((inH - kH) / stride + 1) (.dim ((inW - kW) / stride + 1) .scalar))))
  avgPool2d {kH kW inH inW inC stride : Nat} (h1 : kH ≠ 0) (h2 : kW ≠ 0) :
    Ref (.dim inC (.dim inH (.dim inW .scalar))) →
    m (Ref (.dim inC (.dim ((inH - kH) / stride + 1) (.dim ((inW - kW) / stride + 1) .scalar))))
  avgPool2dPad {kH kW inH inW inC stride padding : Nat} (h1 : kH ≠ 0) (h2 : kW ≠ 0) :
    Ref (.dim inC (.dim inH (.dim inW .scalar))) →
    m (Ref (.dim inC (.dim ((inH + 2 * padding - kH) / stride + 1)
      (.dim ((inW + 2 * padding - kW) / stride + 1) .scalar))))
  relu : {s : Shape} → Ref s → m (Ref s)
  sigmoid : {s : Shape} → Ref s → m (Ref s)
  tanh : {s : Shape} → Ref s → m (Ref s)
  softmax : {s : Shape} → Ref s → m (Ref s)
  logSoftmax : {s : Shape} → Ref s → m (Ref s)
  softplus : {s : Shape} → Ref s → m (Ref s)
  exp : {s : Shape} → Ref s → m (Ref s)
  log : {s : Shape} → Ref s → m (Ref s)
  inv : {s : Shape} → Ref s → m (Ref s)
  detach : {s : Shape} → Ref s → m (Ref s)
  safeLog : {s : Shape} → Ref s → α → m (Ref s)
  sum : {s : Shape} → Ref s → m (Ref Shape.scalar)
  flatten : {s : Shape} → Ref s → m (Ref (.dim (Shape.size s) .scalar))
  linear {inDim outDim : Nat} :
    Ref (.dim outDim (.dim inDim .scalar)) →
    Ref (.dim outDim .scalar) →
    Ref (.dim inDim .scalar) →
    m (Ref (.dim outDim .scalar))
  mseLoss : {s : Shape} → Ref s → Ref s → m (Ref Shape.scalar)
  layerNorm {seqLen embedDim : Nat} (h_seq_pos : seqLen > 0) (h_embed_pos : embedDim > 0) :
    Ref (.dim seqLen (.dim embedDim .scalar)) →
    Ref (.dim embedDim .scalar) →
    Ref (.dim embedDim .scalar) →
    m (Ref (.dim seqLen (.dim embedDim .scalar)))
  batchnormChannelFirst {channels height width : Nat} (h_c : channels > 0) (h_h : height > 0) (h_w
    : width > 0) :
    Ref (.dim channels (.dim height (.dim width .scalar))) →
    Ref (.dim channels .scalar) →
    Ref (.dim channels .scalar) →
    m (Ref (.dim channels (.dim height (.dim width .scalar))))
  multiHeadAttention {n numHeads dModel headDim : Nat} (h1 : n ≠ 0) :
    Ref (.dim dModel (.dim (numHeads * headDim) .scalar)) →
    Ref (.dim dModel (.dim (numHeads * headDim) .scalar)) →
    Ref (.dim dModel (.dim (numHeads * headDim) .scalar)) →
    Ref (.dim (numHeads * headDim) (.dim dModel .scalar)) →
    Ref (.dim n (.dim dModel .scalar)) →
    Option (Tensor Bool (.dim n (.dim n .scalar))) →
    m (Ref (.dim n (.dim dModel .scalar)))
  conv {d inC outC : Nat}
    {kernel stride padding : Vector Nat d}
    {inSpatial : Vector Nat d}
    {hInC : inC ≠ 0} {hKernel : ∀ i : Fin d, kernel.get i ≠ 0} :
    Ref (Shape.ofList (outC :: inC :: kernel.toList)) →
    Ref (.dim outC .scalar) →
    Ref (Shape.ofList (inC :: inSpatial.toList)) →
    m (Ref (Shape.ofList (outC :: (Spec.convOutSpatial inSpatial kernel stride padding).toList)))
  convTranspose {d inC outC : Nat}
    {kernel stride padding : Vector Nat d}
    {inSpatial : Vector Nat d}
    {hInC : inC ≠ 0} {hKernel : ∀ i : Fin d, kernel.get i ≠ 0} :
    Ref (Shape.ofList (inC :: outC :: kernel.toList)) →
    Ref (.dim outC .scalar) →
    Ref (Shape.ofList (inC :: inSpatial.toList)) →
    m (Ref (Shape.ofList (outC ::
      (Spec.convTransposeOutSpatial inSpatial kernel stride padding).toList)))
  conv2d {inC outC kH kW stride padding inH inW : Nat}
    {h1 : inC ≠ 0} {h2 : kH ≠ 0} {h3 : kW ≠ 0} :
    Ref (.dim outC (.dim inC (.dim kH (.dim kW .scalar)))) →
    Ref (.dim outC .scalar) →
    Ref (.dim inC (.dim inH (.dim inW .scalar))) →
    m (Ref (.dim outC (.dim ((inH + 2 * padding - kH) / stride + 1)
      (.dim ((inW + 2 * padding - kW) / stride + 1) .scalar))))

  convTranspose2d {inC outC kH kW stride padding inH inW : Nat}
    {h1 : inC ≠ 0} {h2 : kH ≠ 0} {h3 : kW ≠ 0} :
    Ref (.dim inC (.dim outC (.dim kH (.dim kW .scalar)))) →
    Ref (.dim outC .scalar) →
    Ref (.dim inC (.dim inH (.dim inW .scalar))) →
    m (Ref (.dim outC (.dim ((inH - 1) * stride - 2 * padding + kH)
      (.dim ((inW - 1) * stride - 2 * padding + kW) .scalar))))

  /-
  Seeded RNG primitives (first-class in TorchLean graphs).

  These are deterministic functions of:
  - the provided `seed` (user-controlled), and
  - backend-specific internal counters (e.g. node id / call index).

  They intentionally *do not* rely on `IO` randomness so compiled graphs remain replayable.
  -/
  randUniform : {s : Shape} → (seed : Nat) → m (Ref s)
  bernoulliMask : {s : Shape} → Ref Shape.scalar → (seed : Nat) → m (Ref s)

section

variable {m : Type → Type} {α : Type} [Context α] [DecidableEq Shape] [Monad m] [Ops (m := m) (α :=
  α)]

/--
Reference type for the current `Ops` instance.

In eager mode this will typically be `TensorRef`; in compiled mode it will typically be
  `GraphM.Var`.
-/
abbrev Ref (s : Shape) : Type := Ops.Ref (m := m) (α := α) s

/-- Re-export of `Ops.const`. PyTorch: `torch.tensor(...)` / literal constants. -/
def const {s : Shape} (t : Tensor α s) : m (Ref (m := m) (α := α) s) := Ops.const (m := m) (α := α)
  t
/-- Re-export of `Ops.add`. PyTorch: `torch.add` / `+`. -/
def add {s : Shape} (a b : Ref (m := m) (α := α) s) : m (Ref (m := m) (α := α) s) := Ops.add (m :=
  m) (α := α) a b
/-- Re-export of `Ops.sub`. PyTorch: `torch.sub` / `-`. -/
def sub {s : Shape} (a b : Ref (m := m) (α := α) s) : m (Ref (m := m) (α := α) s) := Ops.sub (m :=
  m) (α := α) a b
/-- Re-export of `Ops.mul`. PyTorch: `torch.mul` / `*`. -/
def mul {s : Shape} (a b : Ref (m := m) (α := α) s) : m (Ref (m := m) (α := α) s) := Ops.mul (m :=
  m) (α := α) a b
/-- Re-export of `Ops.scale`. PyTorch: `x * c` for a scalar `c`. -/
def scale {s : Shape} (x : Ref (m := m) (α := α) s) (c : α) : m (Ref (m := m) (α := α) s) :=
  Ops.scale (m := m) (α := α) x c
/-- Re-export of `Ops.abs`. PyTorch: `torch.abs`. -/
def abs {s : Shape} (x : Ref (m := m) (α := α) s) : m (Ref (m := m) (α := α) s) := Ops.abs (m := m)
  (α := α) x
/-- Re-export of `Ops.sqrt`. PyTorch: `torch.sqrt`. -/
def sqrt {s : Shape} (x : Ref (m := m) (α := α) s) : m (Ref (m := m) (α := α) s) := Ops.sqrt (m :=
  m) (α := α) x
/-- Re-export of `Ops.clamp`. PyTorch: `torch.clamp`. -/
def clamp {s : Shape} (x : Ref (m := m) (α := α) s) (minVal maxVal : α) :
    m (Ref (m := m) (α := α) s) :=
  Ops.clamp (m := m) (α := α) (s := s) x minVal maxVal
/-- Re-export of `Ops.max`. PyTorch: `torch.maximum`. -/
def max {s : Shape} (a b : Ref (m := m) (α := α) s) : m (Ref (m := m) (α := α) s) := Ops.max (m :=
  m) (α := α) a b
/-- Re-export of `Ops.min`. PyTorch: `torch.minimum`. -/
def min {s : Shape} (a b : Ref (m := m) (α := α) s) : m (Ref (m := m) (α := α) s) := Ops.min (m :=
  m) (α := α) a b
/-- Re-export of `Ops.broadcastTo`. PyTorch: broadcasting / `expand`. -/
def broadcastTo {s₁ s₂ : Shape} (cb : Shape.CanBroadcastTo s₁ s₂)
  (x : Ref (m := m) (α := α) s₁) : m (Ref (m := m) (α := α) s₂) :=
  Ops.broadcastTo (m := m) (α := α) (s₁ := s₁) (s₂ := s₂) cb x
/-- Re-export of `Ops.reshape`. PyTorch: `reshape` / `view`. -/
def reshape {s₁ s₂ : Shape} (x : Ref (m := m) (α := α) s₁) (h : Shape.size s₁ = Shape.size s₂) :
  m (Ref (m := m) (α := α) s₂) :=
  Ops.reshape (m := m) (α := α) (s₁ := s₁) (s₂ := s₂) x h
/-- Re-export of `Ops.transpose2d`. PyTorch: `x.t()` / `transpose`. -/
def transpose2d {mDim nDim : Nat}
  (x : Ref (m := m) (α := α) (.dim mDim (.dim nDim .scalar))) :
  m (Ref (m := m) (α := α) (.dim nDim (.dim mDim .scalar))) :=
  Ops.transpose2d (m := m) (α := α) (mDim := mDim) (nDim := nDim) x
/-- Re-export of `Ops.transpose3d_first_to_last`. PyTorch: `permute(1,2,0)`. -/
def transpose3dFirstToLast {a b c : Nat}
  (x : Ref (m := m) (α := α) (.dim a (.dim b (.dim c .scalar)))) :
  m (Ref (m := m) (α := α) (.dim b (.dim c (.dim a .scalar)))) :=
  Ops.transpose3dFirstToLast (m := m) (α := α) (a := a) (b := b) (c := c) x
/-- Re-export of `Ops.transpose3d_last_to_first`. PyTorch: `permute(2,0,1)`. -/
def transpose3dLastToFirst {a b c : Nat}
  (x : Ref (m := m) (α := α) (.dim a (.dim b (.dim c .scalar)))) :
  m (Ref (m := m) (α := α) (.dim c (.dim a (.dim b .scalar)))) :=
  Ops.transpose3dLastToFirst (m := m) (α := α) (a := a) (b := b) (c := c) x
/-- Re-export of `Ops.transpose3d_last_two`. PyTorch: `transpose(1,2)`. -/
def transpose3dLastTwo {a b c : Nat}
  (x : Ref (m := m) (α := α) (.dim a (.dim b (.dim c .scalar)))) :
  m (Ref (m := m) (α := α) (.dim a (.dim c (.dim b .scalar)))) :=
  Ops.transpose3dLastTwo (m := m) (α := α) (a := a) (b := b) (c := c) x
/-- Re-export of `Ops.swapAdjacentAtDepth` (general adjacent-axis swap). -/
def swapAdjacentAtDepth {s : Shape} (depth : Nat)
  (x : Ref (m := m) (α := α) s) :
  m (Ref (m := m) (α := α) (s.swapAdjacentAtDepth depth)) :=
  Ops.swapAdjacentAtDepth (m := m) (α := α) (s := s) depth x
/-- Re-export of `Ops.reduce_sum`. PyTorch: `torch.sum(..., dim=axis)`. -/
def reduceSum {s : Shape} (axis : Nat) [Shape.valid_axis_inst axis s] [Shape.WellFormed s]
  (x : Ref (m := m) (α := α) s) : m (Ref (m := m) (α := α) (shapeAfterSum s axis)) :=
  Ops.reduceSum (m := m) (α := α) (s := s) axis x
/-- Re-export of `Ops.reduce_mean`. PyTorch: `torch.mean(..., dim=axis)`. -/
def reduceMean {s : Shape} (axis : Nat) [Shape.valid_axis_inst axis s] [Shape.WellFormed s]
  (x : Ref (m := m) (α := α) s) : m (Ref (m := m) (α := α) (shapeAfterSum s axis)) :=
  Ops.reduceMean (m := m) (α := α) (s := s) axis x
/-- Re-export of `Ops.gather_scalar`. PyTorch: `x[i]` (1D). -/
def gatherScalar {n : Nat}
  (x : Ref (m := m) (α := α) (.dim n .scalar)) (i : Fin n) : m (Ref (m := m) (α := α) Shape.scalar)
    :=
  Ops.gatherScalar (m := m) (α := α) (n := n) x i
/-- Re-export of `Ops.gather_row`. PyTorch: `x[i]` (2D row). -/
def gatherRow {rows cols : Nat}
  (x : Ref (m := m) (α := α) (.dim rows (.dim cols .scalar))) (i : Fin rows) :
  m (Ref (m := m) (α := α) (.dim cols .scalar)) :=
  Ops.gatherRow (m := m) (α := α) (rows := rows) (cols := cols) x i
/-- Re-export of `Ops.gather_scalar_nat` (index is a raw `Nat`). -/
def gatherScalarNat {n : Nat}
  (x : Ref (m := m) (α := α) (.dim n .scalar)) (i : Nat) : m (Ref (m := m) (α := α) Shape.scalar) :=
  Ops.gatherScalarNat (m := m) (α := α) (n := n) x i
/-- Re-export of `Ops.gather_vec_nat` (index tensor). -/
def gatherVecNat {n k : Nat}
  (x : Ref (m := m) (α := α) (.dim n .scalar)) (idx : Tensor Nat (.dim k .scalar)) :
  m (Ref (m := m) (α := α) (.dim k .scalar)) :=
  Ops.gatherVecNat (m := m) (α := α) (n := n) (k := k) x idx
/-- Re-export of `Ops.gather_rows_nat` (index tensor). -/
def gatherRowsNat {rows cols k : Nat}
  (x : Ref (m := m) (α := α) (.dim rows (.dim cols .scalar))) (idx : Tensor Nat (.dim k .scalar)) :
  m (Ref (m := m) (α := α) (.dim k (.dim cols .scalar))) :=
  Ops.gatherRowsNat (m := m) (α := α) (rows := rows) (cols := cols) (k := k) x idx
/-- Re-export of `Ops.scatter_add_vec`. -/
def scatterAddVec {n : Nat}
  (x : Ref (m := m) (α := α) (.dim n .scalar)) (v : Ref (m := m) (α := α) Shape.scalar) (i : Fin n)
    :
  m (Ref (m := m) (α := α) (.dim n .scalar)) :=
  Ops.scatterAddVec (m := m) (α := α) (n := n) x v i
/-- Re-export of `Ops.scatter_add_row`. -/
def scatterAddRow {rows cols : Nat}
  (x : Ref (m := m) (α := α) (.dim rows (.dim cols .scalar)))
  (v : Ref (m := m) (α := α) (.dim cols .scalar))
  (i : Fin rows) :
  m (Ref (m := m) (α := α) (.dim rows (.dim cols .scalar))) :=
  Ops.scatterAddRow (m := m) (α := α) (rows := rows) (cols := cols) x v i
/-- Re-export of `Ops.matmul`. PyTorch: `torch.matmul` for 2D tensors. -/
def matmul {mDim nDim pDim : Nat}
  (a : Ref (m := m) (α := α) (.dim mDim (.dim nDim .scalar)))
  (b : Ref (m := m) (α := α) (.dim nDim (.dim pDim .scalar))) :
  m (Ref (m := m) (α := α) (.dim mDim (.dim pDim .scalar))) :=
  Ops.matmul (m := m) (α := α) (mDim := mDim) (nDim := nDim) (pDim := pDim) a b
/-- Re-export of `Ops.bmm`. PyTorch: `torch.bmm`. -/
def bmm {batch mDim nDim pDim : Nat}
  (a : Ref (m := m) (α := α) (.dim batch (.dim mDim (.dim nDim .scalar))))
  (b : Ref (m := m) (α := α) (.dim batch (.dim nDim (.dim pDim .scalar)))) :
  m (Ref (m := m) (α := α) (.dim batch (.dim mDim (.dim pDim .scalar)))) :=
  Ops.bmm (m := m) (α := α) (batch := batch) (mDim := mDim) (nDim := nDim) (pDim := pDim) a b
/-- Re-export of `Ops.concat_vectors`. PyTorch: `torch.cat([a,b], dim=0)` for vectors. -/
def concatVectors {nDim mDim : Nat}
  (a : Ref (m := m) (α := α) (.dim nDim .scalar))
  (b : Ref (m := m) (α := α) (.dim mDim .scalar)) :
  m (Ref (m := m) (α := α) (.dim (nDim + mDim) .scalar)) :=
  Ops.concatVectors (m := m) (α := α) (nDim := nDim) (mDim := mDim) a b
/-- Re-export of `Ops.concat_dim0`. PyTorch: `torch.cat(..., dim=0)`. -/
def concatDim0 {nDim mDim : Nat} {s : Shape}
  (a : Ref (m := m) (α := α) (.dim nDim s))
  (b : Ref (m := m) (α := α) (.dim mDim s)) :
  m (Ref (m := m) (α := α) (.dim (nDim + mDim) s)) :=
  Ops.concatDim0 (m := m) (α := α) (nDim := nDim) (mDim := mDim) (s := s) a b
/-- Re-export of `Ops.slice_range0`. PyTorch: `x[start:start+len]` on the leading dimension. -/
def sliceRange0 {nDim : Nat} {s : Shape} (start len : Nat) (h : len + start ≤ nDim)
  (x : Ref (m := m) (α := α) (.dim nDim s)) :
  m (Ref (m := m) (α := α) (.dim len s)) :=
  Ops.sliceRange0 (m := m) (α := α) (nDim := nDim) (s := s) start len h x
/--
Re-export of `Ops.max_pool` (generic N-D max pooling, channels-first; no batch axis).

PyTorch comparison: `torch.nn.functional.max_pool1d` / `max_pool2d` / `max_pool3d` depending on the
spatial rank `d`.
-/
def maxPool {d C : Nat}
  {inSpatial kernel stride padding : Vector Nat d}
  {hKernel : ∀ i : Fin d, kernel.get i ≠ 0}
  (x : Ref (m := m) (α := α) (Shape.ofList (C :: inSpatial.toList))) :
  m (Ref (m := m) (α := α)
    (Shape.ofList (C :: (Spec.poolOutSpatialPad inSpatial kernel stride padding).toList))) :=
  Ops.maxPool (m := m) (α := α)
    (d := d) (C := C)
    (inSpatial := inSpatial) (kernel := kernel) (stride := stride) (padding := padding)
    (hKernel := hKernel)
    x
/--
Re-export of `Ops.avg_pool` (generic N-D average pooling, channels-first; no batch axis).

PyTorch comparison: `torch.nn.functional.avg_pool1d` / `avg_pool2d` / `avg_pool3d` depending on the
spatial rank `d`.
-/
def avgPool {d C : Nat}
  {inSpatial kernel stride padding : Vector Nat d}
  (hKernel : ∀ i : Fin d, kernel.get i ≠ 0)
  (x : Ref (m := m) (α := α) (Shape.ofList (C :: inSpatial.toList))) :
  m (Ref (m := m) (α := α)
    (Shape.ofList (C :: (Spec.poolOutSpatialPad inSpatial kernel stride padding).toList))) :=
  Ops.avgPool (m := m) (α := α)
    (d := d) (C := C)
    (inSpatial := inSpatial) (kernel := kernel) (stride := stride) (padding := padding)
    hKernel
    x
/--
Re-export of `Ops.smooth_max_pool` (generic N-D smooth max pooling, channels-first; no batch axis).

This is a differentiable approximation to max pooling; PyTorch does not expose it as a single
primitive.
-/
def smoothMaxPool {d C : Nat}
  {inSpatial kernel stride padding : Vector Nat d}
  {hKernel : ∀ i : Fin d, kernel.get i ≠ 0}
  (x : Ref (m := m) (α := α) (Shape.ofList (C :: inSpatial.toList)))
  (beta : α) :
  m (Ref (m := m) (α := α)
    (Shape.ofList (C :: (Spec.poolOutSpatialPad inSpatial kernel stride padding).toList))) :=
  Ops.smoothMaxPool (m := m) (α := α)
    (d := d) (C := C)
    (inSpatial := inSpatial) (kernel := kernel) (stride := stride) (padding := padding)
    (hKernel := hKernel)
    x beta
/-- Re-export of `Ops.max_pool2d`. PyTorch: `torch.nn.functional.max_pool2d`. -/
def maxPool2d {kH kW inH inW inC stride : Nat} {h1 : kH ≠ 0} {h2 : kW ≠ 0}
  (x : Ref (m := m) (α := α) (.dim inC (.dim inH (.dim inW .scalar)))) :
  m (Ref (m := m) (α := α) (.dim inC (.dim ((inH - kH) / stride + 1) (.dim ((inW - kW) / stride + 1)
    .scalar)))) :=
  Ops.maxPool2d (m := m) (α := α) (kH := kH) (kW := kW) (inH := inH) (inW := inW) (inC := inC)
    (stride := stride) (h1 := h1) (h2 := h2) x
/-- Re-export of `Ops.max_pool2d_pad`. PyTorch: `max_pool2d(..., padding=...)`. -/
def maxPool2dPad {kH kW inH inW inC stride padding : Nat} {h1 : kH ≠ 0} {h2 : kW ≠ 0}
  (x : Ref (m := m) (α := α) (.dim inC (.dim inH (.dim inW .scalar)))) :
  m (Ref (m := m) (α := α) (.dim inC (.dim ((inH + 2 * padding - kH) / stride + 1)
    (.dim ((inW + 2 * padding - kW) / stride + 1) .scalar)))) :=
  Ops.maxPool2dPad (m := m) (α := α) (kH := kH) (kW := kW) (inH := inH) (inW := inW) (inC := inC)
    (stride := stride) (padding := padding) (h1 := h1) (h2 := h2) x

/-- Alias for `max_pool2d_pad` (PyTorch-style shorthand). -/
abbrev maxPoolPad := @maxPool2dPad

/-- Re-export of `Ops.smooth_max_pool2d` (softmax pooling). -/
def smoothMaxPool2d {kH kW inH inW inC stride : Nat} {h1 : kH ≠ 0} {h2 : kW ≠ 0}
  (x : Ref (m := m) (α := α) (.dim inC (.dim inH (.dim inW .scalar)))) (beta : α) :
  m (Ref (m := m) (α := α) (.dim inC (.dim ((inH - kH) / stride + 1) (.dim ((inW - kW) / stride + 1)
    .scalar)))) :=
  Ops.smoothMaxPool2d (m := m) (α := α) (kH := kH) (kW := kW) (inH := inH) (inW := inW) (inC :=
    inC)
    (stride := stride) (h1 := h1) (h2 := h2) x beta
/-- Re-export of `Ops.avg_pool2d`. PyTorch: `torch.nn.functional.avg_pool2d`. -/
def avgPool2d {kH kW inH inW inC stride : Nat} (h1 : kH ≠ 0) (h2 : kW ≠ 0)
  (x : Ref (m := m) (α := α) (.dim inC (.dim inH (.dim inW .scalar)))) :
  m (Ref (m := m) (α := α) (.dim inC (.dim ((inH - kH) / stride + 1) (.dim ((inW - kW) / stride + 1)
    .scalar)))) :=
  Ops.avgPool2d (m := m) (α := α) (kH := kH) (kW := kW) (inH := inH) (inW := inW) (inC := inC)
    (stride := stride) h1 h2 x
/-- Re-export of `Ops.avg_pool2d_pad`. PyTorch: `avg_pool2d(..., padding=...)`. -/
def avgPool2dPad {kH kW inH inW inC stride padding : Nat} (h1 : kH ≠ 0) (h2 : kW ≠ 0)
  (x : Ref (m := m) (α := α) (.dim inC (.dim inH (.dim inW .scalar)))) :
  m (Ref (m := m) (α := α) (.dim inC (.dim ((inH + 2 * padding - kH) / stride + 1)
    (.dim ((inW + 2 * padding - kW) / stride + 1) .scalar)))) :=
  Ops.avgPool2dPad (m := m) (α := α) (kH := kH) (kW := kW) (inH := inH) (inW := inW) (inC := inC)
    (stride := stride) (padding := padding) h1 h2 x

/-- Alias for `avg_pool2d_pad` (PyTorch-style shorthand). -/
abbrev avgPoolPad := @avgPool2dPad
/-- Re-export of `Ops.relu`. -/
def relu {s : Shape} (x : Ref (m := m) (α := α) s) : m (Ref (m := m) (α := α) s) := Ops.relu (m :=
  m) (α := α) x
/-- Re-export of `Ops.sigmoid`. -/
def sigmoid {s : Shape} (x : Ref (m := m) (α := α) s) : m (Ref (m := m) (α := α) s) := Ops.sigmoid
  (m := m) (α := α) x
/-- Re-export of `Ops.tanh`. -/
def tanh {s : Shape} (x : Ref (m := m) (α := α) s) : m (Ref (m := m) (α := α) s) := Ops.tanh (m :=
  m) (α := α) x
/-- Re-export of `Ops.softmax`. -/
def softmax {s : Shape} (x : Ref (m := m) (α := α) s) : m (Ref (m := m) (α := α) s) := Ops.softmax
  (m := m) (α := α) x
/-- Re-export of `Ops.softplus`. -/
def softplus {s : Shape} (x : Ref (m := m) (α := α) s) : m (Ref (m := m) (α := α) s) := Ops.softplus
  (m := m) (α := α) x
/-- Re-export of `Ops.exp`. -/
def exp {s : Shape} (x : Ref (m := m) (α := α) s) : m (Ref (m := m) (α := α) s) := Ops.exp (m := m)
  (α := α) x
/-- Re-export of `Ops.log`. -/
def log {s : Shape} (x : Ref (m := m) (α := α) s) : m (Ref (m := m) (α := α) s) := Ops.log (m := m)
  (α := α) x
/-- Re-export of `Ops.inv` (reciprocal). -/
def inv {s : Shape} (x : Ref (m := m) (α := α) s) : m (Ref (m := m) (α := α) s) := Ops.inv (m := m)
  (α := α) x
/-- Re-export of `Ops.detach`. PyTorch: `x.detach()`. -/
def detach {s : Shape} (x : Ref (m := m) (α := α) s) : m (Ref (m := m) (α := α) s) :=
  Ops.detach (m := m) (α := α) x
/-- Re-export of `Ops.safe_log`. -/
def safeLog {s : Shape} (x : Ref (m := m) (α := α) s) (ε : α := Numbers.epsilon) :
  m (Ref (m := m) (α := α) s) :=
  Ops.safeLog (m := m) (α := α) (s := s) x ε
/-- Re-export of `Ops.rand_uniform` (deterministic seeded RNG). -/
def randUniform {s : Shape} (seed : Nat) : m (Ref (m := m) (α := α) s) :=
  Ops.randUniform (m := m) (α := α) (s := s) seed
/-- Re-export of `Ops.bernoulli_mask` (deterministic dropout-style mask). -/
def bernoulliMask {s : Shape} (keepProb : Ref (m := m) (α := α) Shape.scalar) (seed : Nat) :
    m (Ref (m := m) (α := α) s) :=
  Ops.bernoulliMask (m := m) (α := α) (s := s) keepProb seed

/--
Stable `log_softmax(x)` along the last axis.

This is a backend primitive with the standard max-shifted formulation
`x - max(x) - log(sum(exp(x - max(x))))`, matching PyTorch's numerical intent.  The optional
`ε` parameter is kept for source compatibility with existing TorchLean callers and is intentionally
ignored; callers that need an epsilon-smoothed logarithm should use `safeLog` explicitly.
-/
def logSoftmax {s : Shape} (x : Ref (m := m) (α := α) s) (ε : α := Numbers.epsilon) :
    m (Ref (m := m) (α := α) s) :=
  let _epsilonKeptForSourceCompatibility := ε
  Ops.logSoftmax (m := m) (α := α) (s := s) x

/-- SiLU / swish: `x * sigmoid(x)`. -/
def silu {s : Shape} (x : Ref (m := m) (α := α) s) : m (Ref (m := m) (α := α) s) := do
  let sx ← sigmoid (m := m) (α := α) (s := s) x
  mul (m := m) (α := α) (s := s) x sx

/--
GELU (approximation used by many ML frameworks):

`0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x^3)))`.

This is defined using existing primitives (`tanh`, `mul`, `add`, `scale`), so it works in eager,
compiled, and verifier-IR backends without introducing a new opcode.
-/
def gelu {s : Shape} (x : Ref (m := m) (α := α) s) : m (Ref (m := m) (α := α) s) := do
  let halfT : Tensor α s := Spec.fill (α := α) (Numbers.pointfive : α) s
  let oneT : Tensor α s := Spec.fill (α := α) (Numbers.one : α) s
  let c0 : α := ((44715 : Nat) : α) / ((1000000 : Nat) : α)
  let c1 : α := MathFunctions.sqrt (Numbers.two / MathFunctions.pi)
  let x2 ← mul (m := m) (α := α) (s := s) x x
  let x3 ← mul (m := m) (α := α) (s := s) x2 x
  let inner ← add (m := m) (α := α) (s := s) x (← scale (m := m) (α := α) (s := s) x3 c0)
  let t ← tanh (m := m) (α := α) (s := s) (← scale (m := m) (α := α) (s := s) inner c1)
  let oneRef ← const (m := m) (α := α) (s := s) oneT
  let onePlus ← add (m := m) (α := α) (s := s) oneRef t
  let mid ← mul (m := m) (α := α) (s := s) x onePlus
  let halfRef ← const (m := m) (α := α) (s := s) halfT
  mul (m := m) (α := α) (s := s) halfRef mid

/--
Global average pooling over the last two axes of a `C×H×W` tensor (channel-first).

Returns a vector `C`, averaging each channel over `H×W`.
-/
def globalAvgPool2dChw {c h w : Nat}
    (h_c_pos : c > 0) (h_h_pos : h > 0) (h_w_pos : w > 0)
    (x : Ref (m := m) (α := α) (.dim c (.dim h (.dim w .scalar)))) :
    m (Ref (m := m) (α := α) (.dim c .scalar)) := do
  let sCHW : Shape := .dim c (.dim h (.dim w .scalar))
  let _ : Shape.WellFormed sCHW := ⟨⟨h_c_pos, ⟨h_h_pos, ⟨h_w_pos, trivial⟩⟩⟩⟩
  let axisW : Nat := Shape.rank sCHW - 1
  have hrank : Shape.rank sCHW > 0 := by simp [sCHW, Shape.rank]
  let _ : Shape.valid_axis_inst axisW sCHW := Shape.validAxisLastAuto hrank
  let yCH ← reduceMean (m := m) (α := α) (s := sCHW) axisW x
  let sCH : Shape := shapeAfterSum sCHW axisW
  have hsCH : sCH = .dim c (.dim h .scalar) := by
    simp [sCH, sCHW, axisW, Shape.rank, shapeAfterSum]
  let _ : Shape.WellFormed sCH := by
    simpa [hsCH] using (show Shape.WellFormed (.dim c (.dim h .scalar)) from ⟨⟨h_c_pos, ⟨h_h_pos,
      trivial⟩⟩⟩)
  let axisH : Nat := Shape.rank sCH - 1
  have hrank2 : Shape.rank sCH > 0 := by simp [hsCH, Shape.rank]
  let _ : Shape.valid_axis_inst axisH sCH := Shape.validAxisLastAuto hrank2
  let yC ← reduceMean (m := m) (α := α) (s := sCH) axisH (by simpa [hsCH] using yCH)
  have hsC : shapeAfterSum sCH axisH = .dim c .scalar := by
    simp [hsCH, axisH, Shape.rank]
  return (by simpa [hsC] using yC)

/--
Global average pooling over the last two axes of an `N×C×H×W` tensor (PyTorch default layout).

Returns `N×C`, averaging each channel over `H×W` for each batch element.
-/
def globalAvgPool2dNchw {n c h w : Nat}
    (h_n_pos : n > 0) (h_c_pos : c > 0) (h_h_pos : h > 0) (h_w_pos : w > 0)
    (x : Ref (m := m) (α := α) (.dim n (.dim c (.dim h (.dim w .scalar))))) :
    m (Ref (m := m) (α := α) (.dim n (.dim c .scalar))) := do
  let sNCHW : Shape := .dim n (.dim c (.dim h (.dim w .scalar)))
  let _ : Shape.WellFormed sNCHW := ⟨⟨h_n_pos, ⟨h_c_pos, ⟨h_h_pos, ⟨h_w_pos, trivial⟩⟩⟩⟩⟩
  let axisW : Nat := Shape.rank sNCHW - 1
  have hrank : Shape.rank sNCHW > 0 := by simp [sNCHW, Shape.rank]
  let _ : Shape.valid_axis_inst axisW sNCHW := Shape.validAxisLastAuto hrank
  let yNCH ← reduceMean (m := m) (α := α) (s := sNCHW) axisW x
  let sNCH : Shape := shapeAfterSum sNCHW axisW
  have hsNCH : sNCH = .dim n (.dim c (.dim h .scalar)) := by
    simp [sNCH, sNCHW, axisW, Shape.rank]
  let _ : Shape.WellFormed sNCH := by
    simpa [hsNCH] using
      (show Shape.WellFormed (.dim n (.dim c (.dim h .scalar))) from ⟨⟨h_n_pos, ⟨h_c_pos, ⟨h_h_pos,
        trivial⟩⟩⟩⟩)
  let axisH : Nat := Shape.rank sNCH - 1
  have hrank2 : Shape.rank sNCH > 0 := by simp [hsNCH, Shape.rank]
  let _ : Shape.valid_axis_inst axisH sNCH := Shape.validAxisLastAuto hrank2
  let yNC ← reduceMean (m := m) (α := α) (s := sNCH) axisH (by simpa [hsNCH] using yNCH)
  have hsNC : shapeAfterSum sNCH axisH = .dim n (.dim c .scalar) := by
    simp [hsNCH, axisH, Shape.rank, shapeAfterSum]
  return (by simpa [hsNC] using yNC)
/-- Re-export of `Ops.sum`. PyTorch: `x.sum()`. -/
def sum {s : Shape} (x : Ref (m := m) (α := α) s) : m (Ref (m := m) (α := α) Shape.scalar) :=
  Ops.sum (m := m) (α := α) (s := s) x
/-- Re-export of `Ops.flatten`. PyTorch: `torch.flatten`. -/
def flatten {s : Shape} (x : Ref (m := m) (α := α) s) :
    m (Ref (m := m) (α := α) (.dim (Shape.size s) .scalar)) :=
  Ops.flatten (m := m) (α := α) (s := s) x

/-- Re-export of `Ops.linear`. PyTorch: `torch.nn.functional.linear`. -/
def linear {inDim outDim : Nat}
  (w : Ref (m := m) (α := α) (.dim outDim (.dim inDim .scalar)))
  (b : Ref (m := m) (α := α) (.dim outDim .scalar))
  (x : Ref (m := m) (α := α) (.dim inDim .scalar)) :
  m (Ref (m := m) (α := α) (.dim outDim .scalar)) :=
  Ops.linear (m := m) (α := α) (inDim := inDim) (outDim := outDim) w b x

/-- Re-export of `Ops.mse_loss`. PyTorch: `torch.nn.functional.mse_loss`. -/
def mseLoss {s : Shape} (yhat target : Ref (m := m) (α := α) s) :
  m (Ref (m := m) (α := α) Shape.scalar) :=
  Ops.mseLoss (m := m) (α := α) (s := s) yhat target

/-- Re-export of `Ops.layer_norm`. PyTorch: `nn.LayerNorm` / `functional.layer_norm`. -/
def layerNorm {seqLen embedDim : Nat} (h_seq_pos : seqLen > 0) (h_embed_pos : embedDim > 0)
  (x : Ref (m := m) (α := α) (.dim seqLen (.dim embedDim .scalar)))
  (gamma : Ref (m := m) (α := α) (.dim embedDim .scalar))
  (beta : Ref (m := m) (α := α) (.dim embedDim .scalar)) :
  m (Ref (m := m) (α := α) (.dim seqLen (.dim embedDim .scalar))) :=
  Ops.layerNorm (m := m) (α := α) (seqLen := seqLen) (embedDim := embedDim)
    h_seq_pos h_embed_pos x gamma beta

/-- Re-export of `Ops.batchnorm_channel_first`. PyTorch: `nn.BatchNorm2d` (conceptually). -/
def batchnormChannelFirst {channels height width : Nat} (h_c : channels > 0) (h_h : height > 0)
  (h_w : width > 0)
  (x : Ref (m := m) (α := α) (.dim channels (.dim height (.dim width .scalar))))
  (gamma : Ref (m := m) (α := α) (.dim channels .scalar))
  (beta : Ref (m := m) (α := α) (.dim channels .scalar)) :
  m (Ref (m := m) (α := α) (.dim channels (.dim height (.dim width .scalar)))) :=
  Ops.batchnormChannelFirst (m := m) (α := α) (channels := channels) (height := height) (width :=
    width)
    h_c h_h h_w x gamma beta

/-- Re-export of `Ops.multi_head_attention`. -/
def multiHeadAttention {n numHeads dModel headDim : Nat} (h1 : n ≠ 0)
  (wq : Ref (m := m) (α := α) (.dim dModel (.dim (numHeads * headDim) .scalar)))
  (wk : Ref (m := m) (α := α) (.dim dModel (.dim (numHeads * headDim) .scalar)))
  (wv : Ref (m := m) (α := α) (.dim dModel (.dim (numHeads * headDim) .scalar)))
  (wo : Ref (m := m) (α := α) (.dim (numHeads * headDim) (.dim dModel .scalar)))
  (x : Ref (m := m) (α := α) (.dim n (.dim dModel .scalar)))
  (mask : Option (Tensor Bool (.dim n (.dim n .scalar))) := none) :
  m (Ref (m := m) (α := α) (.dim n (.dim dModel .scalar))) :=
  Ops.multiHeadAttention (m := m) (α := α) (n := n) (numHeads := numHeads) (dModel := dModel)
    (headDim := headDim) h1 wq wk wv wo x mask

/--
Re-export of `Ops.conv` (generic N-D convolution, channels-first).

PyTorch comparison: `torch.nn.functional.conv{d}d` specialized to a single sample (no batch axis).
-/
def conv {d inC outC : Nat}
  {kernel stride padding : Vector Nat d}
  {inSpatial : Vector Nat d}
  {hInC : inC ≠ 0} {hKernel : ∀ i : Fin d, kernel.get i ≠ 0}
  (weight : Ref (m := m) (α := α) (Shape.ofList (outC :: inC :: kernel.toList)))
  (bias : Ref (m := m) (α := α) (.dim outC .scalar))
  (input : Ref (m := m) (α := α) (Shape.ofList (inC :: inSpatial.toList))) :
  m (Ref (m := m) (α := α)
    (Shape.ofList (outC :: (Spec.convOutSpatial inSpatial kernel stride padding).toList))) :=
  Ops.conv (m := m) (α := α)
    (d := d) (inC := inC) (outC := outC)
    (kernel := kernel) (stride := stride) (padding := padding) (inSpatial := inSpatial)
    (hInC := hInC) (hKernel := hKernel)
    weight bias input

/--
Re-export of `Ops.conv_transpose` (generic N-D transpose convolution, channels-first).

PyTorch comparison: `torch.nn.functional.conv_transpose{d}d` specialized to a single sample.
-/
def convTranspose {d inC outC : Nat}
  {kernel stride padding : Vector Nat d}
  {inSpatial : Vector Nat d}
  {hInC : inC ≠ 0} {hKernel : ∀ i : Fin d, kernel.get i ≠ 0}
  (weight : Ref (m := m) (α := α) (Shape.ofList (inC :: outC :: kernel.toList)))
  (bias : Ref (m := m) (α := α) (.dim outC .scalar))
  (input : Ref (m := m) (α := α) (Shape.ofList (inC :: inSpatial.toList))) :
  m (Ref (m := m) (α := α)
    (Shape.ofList (outC ::
      (Spec.convTransposeOutSpatial inSpatial kernel stride padding).toList))) :=
  Ops.convTranspose (m := m) (α := α)
    (d := d) (inC := inC) (outC := outC)
    (kernel := kernel) (stride := stride) (padding := padding) (inSpatial := inSpatial)
    (hInC := hInC) (hKernel := hKernel)
    weight bias input

/-- Re-export of `Ops.conv2d`. PyTorch: `torch.nn.functional.conv2d` (conceptually, no batch axis).
  -/
def conv2d {inC outC kH kW stride padding inH inW : Nat}
  {h1 : inC ≠ 0} {h2 : kH ≠ 0} {h3 : kW ≠ 0}
  (kernel : Ref (m := m) (α := α) (.dim outC (.dim inC (.dim kH (.dim kW .scalar)))))
  (bias : Ref (m := m) (α := α) (.dim outC .scalar))
  (input : Ref (m := m) (α := α) (.dim inC (.dim inH (.dim inW .scalar)))) :
  m (Ref (m := m) (α := α) (.dim outC (.dim ((inH + 2 * padding - kH) / stride + 1)
    (.dim ((inW + 2 * padding - kW) / stride + 1) .scalar)))) :=
  Ops.conv2d (m := m) (α := α) (inC := inC) (outC := outC) (kH := kH) (kW := kW)
    (stride := stride) (padding := padding) (inH := inH) (inW := inW) (h1 := h1) (h2 := h2) (h3 :=
      h3)
    kernel bias input

/-- Re-export of `Ops.conv_transpose2d`. PyTorch: `torch.nn.functional.conv_transpose2d`. -/
def convTranspose2d {inC outC kH kW stride padding inH inW : Nat}
  {h1 : inC ≠ 0} {h2 : kH ≠ 0} {h3 : kW ≠ 0}
  (kernel : Ref (m := m) (α := α) (.dim inC (.dim outC (.dim kH (.dim kW .scalar)))))
  (bias : Ref (m := m) (α := α) (.dim outC .scalar))
  (input : Ref (m := m) (α := α) (.dim inC (.dim inH (.dim inW .scalar)))) :
  m (Ref (m := m) (α := α) (.dim outC (.dim ((inH - 1) * stride - 2 * padding + kH)
    (.dim ((inW - 1) * stride - 2 * padding + kW) .scalar)))) :=
  Ops.convTranspose2d (m := m) (α := α)
    (inC := inC) (outC := outC) (kH := kH) (kW := kW)
    (stride := stride) (padding := padding) (inH := inH) (inW := inW)
    (h1 := h1) (h2 := h2) (h3 := h3)
    kernel bias input

/-- Alias for `conv2d` (compat shorthand). -/
abbrev conv2dCompat := @conv2d

end

/--
Monad used for the eager `Ops` instance: read an `Internal.EagerSession α` and execute in `IO`.

This is the backend that makes `Ops` programs execute immediately by mutating a hidden runtime tape.
-/
abbrev Internal.EagerM (α : Type) := ReaderT (Internal.EagerSession α) IO

/--
`Ops` instance for the eager Torch-style runtime.

This interprets `Ops` primitives by immediately executing them against the hidden mutable tape in
the current `Internal.EagerSession`.
-/
instance {α : Type} [Context α] [Internal.CudaBridge.TensorConv α] [DecidableEq Shape] :
    Ops (Internal.EagerM α) α where
  Ref := fun s => TensorRef α s
  const := fun {s} t => fun sess => Internal.EagerSession.const (α := α) sess (sh := s) t
  add := fun {s} a b => fun sess => Internal.EagerSession.add (α := α) sess (sh := s) a b
  sub := fun {s} a b => fun sess => Internal.EagerSession.sub (α := α) sess (sh := s) a b
  mul := fun {s} a b => fun sess => Internal.EagerSession.mul (α := α) sess (sh := s) a b
  scale := fun {s} x c => fun sess => Internal.EagerSession.scale (α := α) sess (sh := s) x c
  abs := fun {s} x => fun sess => Internal.EagerSession.abs (α := α) sess (sh := s) x
  sqrt := fun {s} x => fun sess => Internal.EagerSession.sqrt (α := α) sess (sh := s) x
  clamp := fun {s} x minVal maxVal => fun sess =>
    Internal.EagerSession.clamp (α := α) sess (sh := s) x minVal maxVal
  max := fun {s} a b => fun sess => Internal.EagerSession.max (α := α) sess (sh := s) a b
  min := fun {s} a b => fun sess => Internal.EagerSession.min (α := α) sess (sh := s) a b
  broadcastTo := fun {s₁ s₂} cb x => fun sess =>
    Internal.EagerSession.broadcastTo (α := α) sess (sh1 := s₁) (sh2 := s₂) cb x
  reshape := fun {s₁ s₂} x h => fun sess =>
    Internal.EagerSession.reshape (α := α) sess (sh1 := s₁) (sh2 := s₂) x h
  transpose2d := fun {mDim nDim} x => fun sess =>
    Internal.EagerSession.transpose2d (α := α) sess (m := mDim) (n := nDim) x
  transpose3dFirstToLast := fun {a b c} x => fun sess =>
    Internal.EagerSession.transpose3dFirstToLast (α := α) sess (a := a) (b := b) (c := c) x
  transpose3dLastToFirst := fun {a b c} x => fun sess =>
    Internal.EagerSession.transpose3dLastToFirst (α := α) sess (a := a) (b := b) (c := c) x
  transpose3dLastTwo := fun {a b c} x => fun sess =>
    Internal.EagerSession.transpose3dLastTwo (α := α) sess (a := a) (b := b) (c := c) x
  swapAdjacentAtDepth := fun {s} depth x => fun sess =>
    Internal.EagerSession.swapAdjacentAtDepth (α := α) sess (sh := s) depth x
  reduceSum := fun {s} axis => fun x => fun sess =>
    Internal.EagerSession.reduceSum (α := α) sess (sh := s) axis x
  reduceMean := fun {s} axis => fun x => fun sess =>
    Internal.EagerSession.reduceMean (α := α) sess (sh := s) axis x
  gatherScalar := fun {n} x i => fun sess =>
    Internal.EagerSession.gatherScalar (α := α) sess (n := n) x i
  gatherRow := fun {rows cols} x i => fun sess =>
    Internal.EagerSession.gatherRow (α := α) sess (rows := rows) (cols := cols) x i
  gatherScalarNat := fun {n} x i => fun sess =>
    Internal.EagerSession.gatherScalarNat (α := α) sess (n := n) x i
  gatherVecNat := fun {n k} x idx => fun sess =>
    Internal.EagerSession.gatherVecNat (α := α) sess (n := n) (k := k) x idx
  gatherRowsNat := fun {rows cols k} x idx => fun sess =>
    Internal.EagerSession.gatherRowsNat (α := α) sess (rows := rows) (cols := cols) (k := k) x idx
  scatterAddVec := fun {n} x v i => fun sess =>
    Internal.EagerSession.scatterAddVec (α := α) sess (n := n) x v i
  scatterAddRow := fun {rows cols} x v i => fun sess =>
    Internal.EagerSession.scatterAddRow (α := α) sess (rows := rows) (cols := cols) x v i
  matmul := fun {mDim nDim pDim} a b => fun sess =>
    Internal.EagerSession.matmul (α := α) sess (m := mDim) (n := nDim) (p := pDim) a b
  bmm := fun {batch mDim nDim pDim} a b => fun sess =>
    Internal.EagerSession.bmm (α := α) sess (batch := batch) (m := mDim) (n := nDim) (p := pDim) a b
  concatVectors := fun {nDim mDim} a b => fun sess =>
    Internal.EagerSession.concatVectors (α := α) sess (n := nDim) (m := mDim) a b
  concatDim0 := fun {nDim mDim} {s} a b => fun sess =>
    Internal.EagerSession.concatDim0 (α := α) sess (n := nDim) (m := mDim) (sh := s) a b
  sliceRange0 := fun {nDim} {s} start len h x => fun sess =>
    Internal.EagerSession.sliceRange0 (α := α) sess (n := nDim) (sh := s) x start len h
  maxPool := fun {d C} {inSpatial kernel stride padding} {hKernel} x => fun sess =>
    Internal.EagerSession.maxPool (α := α) sess
      (d := d) (C := C)
      (inSpatial := inSpatial) (kernel := kernel) (stride := stride) (padding := padding)
      (hKernel := hKernel)
      x
  avgPool := fun {d C} {inSpatial kernel stride padding} hKernel x => fun sess =>
    Internal.EagerSession.avgPool (α := α) sess
      (d := d) (C := C)
      (inSpatial := inSpatial) (kernel := kernel) (stride := stride) (padding := padding)
      hKernel
      x
  smoothMaxPool := fun {d C} {inSpatial kernel stride padding} {hKernel} x beta => fun sess =>
    Internal.EagerSession.smoothMaxPool (α := α) sess
      (d := d) (C := C)
      (inSpatial := inSpatial) (kernel := kernel) (stride := stride) (padding := padding)
      (hKernel := hKernel)
      x beta
  maxPool2d := fun {kH kW inH inW inC stride} {h1 h2} x => fun sess =>
    Internal.EagerSession.maxPool2d (α := α) sess
      (kH := kH) (kW := kW) (inH := inH) (inW := inW) (inC := inC) (stride := stride)
      (h1 := h1) (h2 := h2) x
  maxPool2dPad := fun {kH kW inH inW inC stride padding} {h1 h2} x => fun sess =>
    Internal.EagerSession.maxPool2dPad (α := α) sess
      (kH := kH) (kW := kW) (inH := inH) (inW := inW) (inC := inC) (stride := stride) (padding :=
        padding)
      (h1 := h1) (h2 := h2) x
  smoothMaxPool2d := fun {kH kW inH inW inC stride} {h1 h2} x beta => fun sess =>
    Internal.EagerSession.smoothMaxPool2d (α := α) sess
      (kH := kH) (kW := kW) (inH := inH) (inW := inW) (inC := inC) (stride := stride)
      (h1 := h1) (h2 := h2) x beta
  avgPool2d := fun {kH kW inH inW inC stride} h1 h2 x => fun sess =>
    Internal.EagerSession.avgPool2d (α := α) sess
      (kH := kH) (kW := kW) (inH := inH) (inW := inW) (inC := inC) (stride := stride)
      h1 h2 x
  avgPool2dPad := fun {kH kW inH inW inC stride padding} h1 h2 x => fun sess =>
    Internal.EagerSession.avgPool2dPad (α := α) sess
      (kH := kH) (kW := kW) (inH := inH) (inW := inW) (inC := inC) (stride := stride) (padding :=
        padding)
      h1 h2 x
  relu := fun {s} x => fun sess => Internal.EagerSession.relu (α := α) sess (sh := s) x
  sigmoid := fun {s} x => fun sess => Internal.EagerSession.sigmoid (α := α) sess (sh := s) x
  tanh := fun {s} x => fun sess => Internal.EagerSession.tanh (α := α) sess (sh := s) x
  softmax := fun {s} x => fun sess => Internal.EagerSession.softmax (α := α) sess (sh := s) x
  logSoftmax := fun {s} x => fun sess => Internal.EagerSession.logSoftmax (α := α) sess (sh := s) x
  softplus := fun {s} x => fun sess => Internal.EagerSession.softplus (α := α) sess (sh := s) x
  exp := fun {s} x => fun sess => Internal.EagerSession.exp (α := α) sess (sh := s) x
  log := fun {s} x => fun sess => Internal.EagerSession.log (α := α) sess (sh := s) x
  inv := fun {s} x => fun sess => Internal.EagerSession.inv (α := α) sess (sh := s) x
  detach := fun {s} x => fun sess => Internal.EagerSession.detach (α := α) sess (sh := s) x
  safeLog := fun {s} x ε => fun sess => Internal.EagerSession.safeLog (α := α) sess (sh := s) x (ε
    := ε)
  sum := fun {s} x => fun sess => Internal.EagerSession.sum (α := α) sess (sh := s) x
  flatten := fun {s} x => fun sess => Internal.EagerSession.flatten (α := α) sess (sh := s) x
  linear := fun {inDim outDim} w b x => fun sess =>
    Internal.EagerSession.linear (α := α) sess (inDim := inDim) (outDim := outDim) w b x
  mseLoss := fun {s} yhat target => fun sess => Internal.EagerSession.mseLoss (α := α) sess (sh :=
    s) yhat target
  layerNorm := fun {seqLen embedDim} hSeq hEmb x gamma beta => fun sess =>
    Internal.EagerSession.layerNorm (α := α) sess (seqLen := seqLen) (embedDim := embedDim)
      (h_seq_pos := hSeq) (h_embed_pos := hEmb) x gamma beta
  batchnormChannelFirst := fun {channels height width} hC hH hW x gamma beta => fun sess =>
    Internal.EagerSession.batchnormChannelFirst (α := α) sess
      (channels := channels) (height := height) (width := width) (h_c := hC) (h_h := hH) (h_w := hW)
      x gamma beta
  multiHeadAttention := fun {n numHeads dModel headDim} h1 wq wk wv wo x mask => fun sess =>
    Internal.EagerSession.multiHeadAttention (α := α) sess (n := n) (numHeads := numHeads)
      (dModel := dModel) (headDim := headDim) h1 wq wk wv wo x (mask := mask)
  conv := fun {d inC outC} {kernel stride padding} {inSpatial} {hInC hKernel} w b x => fun sess =>
    Internal.EagerSession.conv (α := α) sess
      (d := d) (inC := inC) (outC := outC)
      (kernel := kernel) (stride := stride) (padding := padding) (inSpatial := inSpatial)
      (hInC := hInC) (hKernel := hKernel)
      w b x
  convTranspose := fun {d inC outC} {kernel stride padding} {inSpatial} {hInC hKernel} w b x =>
    fun sess =>
      Internal.EagerSession.convTranspose (α := α) sess
        (d := d) (inC := inC) (outC := outC)
        (kernel := kernel) (stride := stride) (padding := padding) (inSpatial := inSpatial)
        (hInC := hInC) (hKernel := hKernel)
        w b x
  conv2d := fun {inC outC kH kW stride padding inH inW} {h1 h2 h3} kernel bias input => fun sess =>
    Internal.EagerSession.conv2d (α := α) sess (inC := inC) (outC := outC) (kH := kH) (kW := kW)
      (stride := stride) (padding := padding) (inH := inH) (inW := inW) (h1 := h1) (h2 := h2) (h3 :=
        h3)
      kernel bias input
  convTranspose2d := fun {inC outC kH kW stride padding inH inW} {h1 h2 h3} kernel bias input =>
    fun sess =>
      Internal.EagerSession.convTranspose2d (α := α) sess
        (inC := inC) (outC := outC) (kH := kH) (kW := kW)
        (stride := stride) (padding := padding) (inH := inH) (inW := inW)
        (h1 := h1) (h2 := h2) (h3 := h3)
        kernel bias input
  randUniform := fun {s} seed => fun sess =>
    Internal.EagerSession.randUniform (α := α) sess (sh := s) seed
  bernoulliMask := fun {s} keepProb seed => fun sess =>
    Internal.EagerSession.bernoulliMask (α := α) sess (sh := s) keepProb seed

/--
`Ops` instance for the compiled graph-building monad `GraphM`.

This interprets `Ops` primitives by *recording* typed IR nodes (rather than executing immediately).
See `Runtime.Autograd.Compiled.GraphM` and `Torch.LinkedSession` for how these graphs are later run.
-/
instance {α : Type} [Context α] [DecidableEq Shape] {Γ : List Shape} :
    Ops (Runtime.Autograd.Compiled.GraphM.M α Γ) α where
  Ref := fun s => Runtime.Autograd.Compiled.GraphM.Var s
  const := fun {s} t => Runtime.Autograd.Compiled.GraphM.const (α := α) (Γ := Γ) (s := s) t
  add := fun {s} a b => Runtime.Autograd.Compiled.GraphM.add (α := α) (Γ := Γ) (s := s) a b
  sub := fun {s} a b => Runtime.Autograd.Compiled.GraphM.sub (α := α) (Γ := Γ) (s := s) a b
  mul := fun {s} a b => Runtime.Autograd.Compiled.GraphM.mul (α := α) (Γ := Γ) (s := s) a b
  scale := fun {s} x c => Runtime.Autograd.Compiled.GraphM.scale (α := α) (Γ := Γ) (s := s) x c
  abs := fun {s} x => Runtime.Autograd.Compiled.GraphM.abs (α := α) (Γ := Γ) (s := s) x
  sqrt := fun {s} x => Runtime.Autograd.Compiled.GraphM.sqrt (α := α) (Γ := Γ) (s := s) x
  clamp := fun {s} x minVal maxVal =>
    Runtime.Autograd.Compiled.GraphM.clamp (α := α) (Γ := Γ) (s := s) x minVal maxVal
  max := fun {s} a b => Runtime.Autograd.Compiled.GraphM.max (α := α) (Γ := Γ) (s := s) a b
  min := fun {s} a b => Runtime.Autograd.Compiled.GraphM.min (α := α) (Γ := Γ) (s := s) a b
  broadcastTo := fun {s₁ s₂} cb x =>
    Runtime.Autograd.Compiled.GraphM.broadcastTo (α := α) (Γ := Γ) (s₁ := s₁) (s₂ := s₂) cb x
  reshape := fun {s₁ s₂} x h =>
    Runtime.Autograd.Compiled.GraphM.reshape (α := α) (Γ := Γ) (s₁ := s₁) (s₂ := s₂) x h
  transpose2d := fun {mDim nDim} x =>
    Runtime.Autograd.Compiled.GraphM.transpose2d (α := α) (Γ := Γ) (m := mDim) (n := nDim) x
  transpose3dFirstToLast := fun {a b c} x =>
    Runtime.Autograd.Compiled.GraphM.transpose3dFirstToLast (α := α) (Γ := Γ) (a := a) (b := b)
      (c := c) x
  transpose3dLastToFirst := fun {a b c} x =>
    Runtime.Autograd.Compiled.GraphM.transpose3dLastToFirst (α := α) (Γ := Γ) (a := a) (b := b)
      (c := c) x
  transpose3dLastTwo := fun {a b c} x =>
    Runtime.Autograd.Compiled.GraphM.transpose3dLastTwo (α := α) (Γ := Γ) (a := a) (b := b) (c :=
      c) x
  swapAdjacentAtDepth := fun {s} depth x =>
    Runtime.Autograd.Compiled.GraphM.swapAdjacentAtDepth (α := α) (Γ := Γ) (s := s) depth x
  reduceSum := fun {s} axis => fun x =>
    Runtime.Autograd.Compiled.GraphM.reduceSum (α := α) (Γ := Γ) (s := s) axis x
  reduceMean := fun {s} axis => fun x =>
    Runtime.Autograd.Compiled.GraphM.reduceMean (α := α) (Γ := Γ) (s := s) axis x
  gatherScalar := fun {n} x i =>
    Runtime.Autograd.Compiled.GraphM.gatherScalar (α := α) (Γ := Γ) (n := n) x i
  gatherRow := fun {rows cols} x i =>
    Runtime.Autograd.Compiled.GraphM.gatherRow (α := α) (Γ := Γ) (rows := rows) (cols := cols) x i
  gatherScalarNat := fun {n} x i =>
    Runtime.Autograd.Compiled.GraphM.gatherScalarNat (α := α) (Γ := Γ) (n := n) x i
  gatherVecNat := fun {n k} x idx =>
    Runtime.Autograd.Compiled.GraphM.gatherVecNat (α := α) (Γ := Γ) (n := n) (k := k) x idx
  gatherRowsNat := fun {rows cols k} x idx =>
    Runtime.Autograd.Compiled.GraphM.gatherRowsNat (α := α) (Γ := Γ) (rows := rows) (cols := cols)
      (k := k) x idx
  scatterAddVec := fun {n} x v i =>
    Runtime.Autograd.Compiled.GraphM.scatterAddVec (α := α) (Γ := Γ) (n := n) x v i
  scatterAddRow := fun {rows cols} x v i =>
    Runtime.Autograd.Compiled.GraphM.scatterAddRow (α := α) (Γ := Γ) (rows := rows) (cols := cols)
      x v i
  matmul := fun {mDim nDim pDim} a b =>
    Runtime.Autograd.Compiled.GraphM.matmul (α := α) (Γ := Γ) (m := mDim) (n := nDim) (p := pDim) a
      b
  bmm := fun {batch mDim nDim pDim} a b =>
    Runtime.Autograd.Compiled.GraphM.bmm (α := α) (Γ := Γ) (batch := batch) (m := mDim) (n := nDim)
      (p := pDim) a b
  concatVectors := fun {nDim mDim} a b =>
    Runtime.Autograd.Compiled.GraphM.concatVectors (α := α) (Γ := Γ) (n := nDim) (m := mDim) a b
  concatDim0 := fun {nDim mDim} {s} a b =>
    Runtime.Autograd.Compiled.GraphM.concatDim0 (α := α) (Γ := Γ) (n := nDim) (m := mDim) (s := s)
      a b
  sliceRange0 := fun {nDim} {s} start len h x =>
    Runtime.Autograd.Compiled.GraphM.sliceRange0 (α := α) (Γ := Γ) (n := nDim) (s := s) x start len
      h
  maxPool := fun {d C} {inSpatial kernel stride padding} {hKernel} x =>
    Runtime.Autograd.Compiled.GraphM.maxPool (α := α) (Γ := Γ)
      (d := d) (C := C)
      (inSpatial := inSpatial) (kernel := kernel) (stride := stride) (padding := padding)
      (hKernel := hKernel)
      x
  avgPool := fun {d C} {inSpatial kernel stride padding} hKernel x =>
    Runtime.Autograd.Compiled.GraphM.avgPool (α := α) (Γ := Γ)
      (d := d) (C := C)
      (inSpatial := inSpatial) (kernel := kernel) (stride := stride) (padding := padding)
      (hKernel := hKernel)
      x
  smoothMaxPool := fun {d C} {inSpatial kernel stride padding} {hKernel} x beta =>
    Runtime.Autograd.Compiled.GraphM.smoothMaxPool (α := α) (Γ := Γ)
      (d := d) (C := C)
      (inSpatial := inSpatial) (kernel := kernel) (stride := stride) (padding := padding)
      (hKernel := hKernel)
      x beta
  maxPool2d := fun {kH kW inH inW inC stride} {h1 h2} x =>
    Runtime.Autograd.Compiled.GraphM.maxPool2d (α := α) (Γ := Γ)
      (kH := kH) (kW := kW) (inH := inH) (inW := inW) (inC := inC) (stride := stride)
      (h1 := h1) (h2 := h2) x
  maxPool2dPad := fun {kH kW inH inW inC stride padding} {h1 h2} x =>
    Runtime.Autograd.Compiled.GraphM.maxPool2dPad (α := α) (Γ := Γ)
      (kH := kH) (kW := kW) (inH := inH) (inW := inW) (inC := inC) (stride := stride) (padding :=
        padding)
      (h1 := h1) (h2 := h2) x
  smoothMaxPool2d := fun {kH kW inH inW inC stride} {h1 h2} x beta =>
    Runtime.Autograd.Compiled.GraphM.smoothMaxPool2d (α := α) (Γ := Γ)
      (kH := kH) (kW := kW) (inH := inH) (inW := inW) (inC := inC) (stride := stride)
      (h1 := h1) (h2 := h2) x beta
  avgPool2d := fun {kH kW inH inW inC stride} h1 h2 x =>
    Runtime.Autograd.Compiled.GraphM.avgPool2d (α := α) (Γ := Γ)
      (kH := kH) (kW := kW) (inH := inH) (inW := inW) (inC := inC) (stride := stride)
      h1 h2 x
  avgPool2dPad := fun {kH kW inH inW inC stride padding} h1 h2 x =>
    Runtime.Autograd.Compiled.GraphM.avgPool2dPad (α := α) (Γ := Γ)
      (kH := kH) (kW := kW) (inH := inH) (inW := inW) (inC := inC) (stride := stride) (padding :=
        padding)
      h1 h2 x
  relu := fun {s} x => Runtime.Autograd.Compiled.GraphM.relu (α := α) (Γ := Γ) (s := s) x
  sigmoid := fun {s} x => Runtime.Autograd.Compiled.GraphM.sigmoid (α := α) (Γ := Γ) (s := s) x
  tanh := fun {s} x => Runtime.Autograd.Compiled.GraphM.tanh (α := α) (Γ := Γ) (s := s) x
  softmax := fun {s} x => Runtime.Autograd.Compiled.GraphM.softmax (α := α) (Γ := Γ) (s := s) x
  logSoftmax := fun {s} x => Runtime.Autograd.Compiled.GraphM.logSoftmax (α := α) (Γ := Γ) (s := s)
    x
  softplus := fun {s} x => Runtime.Autograd.Compiled.GraphM.softplus (α := α) (Γ := Γ) (s := s) x
  exp := fun {s} x => Runtime.Autograd.Compiled.GraphM.exp (α := α) (Γ := Γ) (s := s) x
  log := fun {s} x => Runtime.Autograd.Compiled.GraphM.log (α := α) (Γ := Γ) (s := s) x
  inv := fun {s} x => Runtime.Autograd.Compiled.GraphM.inv (α := α) (Γ := Γ) (s := s) x
  detach := fun {s} x => Runtime.Autograd.Compiled.GraphM.detach (α := α) (Γ := Γ) (s := s) x
  safeLog := fun {s} x ε => Runtime.Autograd.Compiled.GraphM.safeLog (α := α) (Γ := Γ) (s := s) x
    (ε := ε)
  sum := fun {s} x => Runtime.Autograd.Compiled.GraphM.sum (α := α) (Γ := Γ) (s := s) x
  flatten := fun {s} x => Runtime.Autograd.Compiled.GraphM.flatten (α := α) (Γ := Γ) (s := s) x
  linear := fun {inDim outDim} w b x =>
    Runtime.Autograd.Compiled.GraphM.linear (α := α) (Γ := Γ) (inDim := inDim) (outDim := outDim) w
      b x
  mseLoss := fun {s} yhat target =>
    Runtime.Autograd.Compiled.GraphM.mseLoss (α := α) (Γ := Γ) (s := s) yhat target
  layerNorm := fun {seqLen embedDim} hSeq hEmb x gamma beta =>
    Runtime.Autograd.Compiled.GraphM.layerNorm (α := α) (Γ := Γ) (seqLen := seqLen) (embedDim :=
      embedDim)
      (h_seq_pos := hSeq) (h_embed_pos := hEmb) x gamma beta
  batchnormChannelFirst := fun {channels height width} hC hH hW x gamma beta =>
    Runtime.Autograd.Compiled.GraphM.batchnormChannelFirst (α := α) (Γ := Γ)
      (channels := channels) (height := height) (width := width) (h_c := hC) (h_h := hH) (h_w := hW)
      x gamma beta
  multiHeadAttention := fun {n numHeads dModel headDim} h1 wq wk wv wo x mask =>
    Runtime.Autograd.Compiled.GraphM.multiHeadAttention (α := α) (Γ := Γ) (n := n) (numHeads :=
      numHeads)
      (dModel := dModel) (headDim := headDim) h1 wq wk wv wo x (mask := mask)
  conv := fun {d inC outC} {kernel stride padding} {inSpatial} {hInC hKernel} w b x =>
    Runtime.Autograd.Compiled.GraphM.conv (α := α) (Γ := Γ)
      (d := d) (inC := inC) (outC := outC)
      (kernel := kernel) (stride := stride) (padding := padding) (inSpatial := inSpatial)
      (hInC := hInC) (hKernel := hKernel)
      w b x
  convTranspose := fun {d inC outC} {kernel stride padding} {inSpatial} {hInC hKernel} w b x =>
    Runtime.Autograd.Compiled.GraphM.convTranspose (α := α) (Γ := Γ)
      (d := d) (inC := inC) (outC := outC)
      (kernel := kernel) (stride := stride) (padding := padding) (inSpatial := inSpatial)
      (hInC := hInC) (hKernel := hKernel)
      w b x
  conv2d := fun {inC outC kH kW stride padding inH inW} {h1 h2 h3} kernel bias input =>
    Runtime.Autograd.Compiled.GraphM.conv2d (α := α) (Γ := Γ) (inC := inC) (outC := outC) (kH := kH)
      (kW := kW)
      (stride := stride) (padding := padding) (inH := inH) (inW := inW) (h1 := h1) (h2 := h2) (h3 :=
        h3)
      kernel bias input
  convTranspose2d := fun {inC outC kH kW stride padding inH inW} {h1 h2 h3} kernel bias input =>
    Runtime.Autograd.Compiled.GraphM.convTranspose2d (α := α) (Γ := Γ)
      (inC := inC) (outC := outC) (kH := kH) (kW := kW)
      (stride := stride) (padding := padding) (inH := inH) (inW := inW)
      (h1 := h1) (h2 := h2) (h3 := h3)
      kernel bias input
  randUniform := fun {s} seed => do
    Runtime.Autograd.Compiled.GraphM.randUniform (α := α) (Γ := Γ) (s := s) (seed := seed)
  bernoulliMask := fun {s} keepProb seed => do
    Runtime.Autograd.Compiled.GraphM.bernoulliMask (α := α) (Γ := Γ) (s := s) keepProb (seed :=
      seed)

/--
Heterogeneous list of trainable parameters, indexed by a list of shapes.

This is the Torch front-end analogue of "a parameter vector" (like `model.parameters()` in PyTorch),
but with shapes tracked at the type level.
-/
inductive ParamList (α : Type) : List Shape → Type where
  | nil : ParamList α []
  | cons {s : Shape} {ss : List Shape} : Param α s → ParamList α ss → ParamList α (s :: ss)

namespace ParamList

/--
Materialize the SGD update `v - lr * g` in a single traversal.

This is used by `sgdStep_fast` as a runtime-performance optimization to avoid building deep thunk
chains when training for many steps.
-/
  def subScaleMaterialize {α : Type} [Sub α] [Mul α] :
    {s : Shape} → Tensor α s → Tensor α s → α → Tensor α s
  | .scalar, .scalar v, .scalar g, lr =>
      Tensor.scalar (v - (lr * g))
  | .dim n s', .dim fv, .dim fg, lr =>
      let arr : Array (Tensor α s') := Array.ofFn (fun i : Fin n => subScaleMaterialize (s := s')
        (fv i) (fg i) lr)
      Tensor.dim (fun i =>
        let hn : arr.size = n := by
          simp [arr]
        let hi : i.1 < arr.size :=
          Eq.ndrec (motive := fun m => i.1 < m) i.2 hn.symm
        arr[i.1]'hi)

/--
Allocate a fresh `ParamList` from an initial `TList` of parameter tensors.

Each tensor becomes an `IO.Ref` so it can be updated by optimizer steps.
-/
def ofTList {α : Type} {ss : List Shape} (xs : TList α ss) : IO (ParamList α ss) := do
  match xs with
  | .nil => pure .nil
  | .cons x xs =>
      let r ← IO.mkRef x
      let cudaValue ← IO.mkRef (none : Option Runtime.Autograd.Cuda.AnyBuffer)
      let hostCurrent ← IO.mkRef true
      let p : Param α _ := { value := r, cudaValue := cudaValue, hostCurrent := hostCurrent }
      let ps ← ofTList (α := α) xs
      pure (.cons p ps)

/--
Allocate a fresh `ParamList` from an initial `TList` of parameter tensors, with explicit
`requiresGrad` flags.

Returns an error when the flag list length does not match the parameter shape list length.
-/
def ofTListWithRequiresGrad {α : Type} :
    {ss : List Shape} → TList α ss → List Bool → IO (ParamList α ss)
  | [], .nil, [] => pure .nil
  | _s :: ss, .cons x xs, rg :: rgs => do
      let r ← IO.mkRef x
      let cudaValue ← IO.mkRef (none : Option Runtime.Autograd.Cuda.AnyBuffer)
      let hostCurrent ← IO.mkRef true
      let p : Param α _ :=
        { value := r, cudaValue := cudaValue, hostCurrent := hostCurrent, requiresGrad := rg }
      let ps ← ofTListWithRequiresGrad (α := α) (ss := ss) xs rgs
      pure (.cons p ps)
  | [], .nil, _ =>
      throw <| IO.userError "torch: requiresGrad list longer than parameter list"
  | _ :: _, .cons _ _, [] =>
      throw <| IO.userError "torch: requiresGrad list shorter than parameter list"

/-- Read the current parameter values as a `TList` aligned with the shape list. -/
def values {α : Type} : {ss : List Shape} → ParamList α ss → IO (TList α ss)
  | [], .nil => pure .nil
  | _s :: ss, .cons p ps => do
      let v ← p.value.get
      let vs ← values (α := α) (ss := ss) ps
      pure (.cons v vs)

/-- Read parameter values, synchronizing CUDA-resident mirrors first when necessary. -/
def valuesSynced {α : Type} [Internal.CudaBridge.TensorConv α] [DecidableEq Shape] :
    {ss : List Shape} → ParamList α ss → IO (TList α ss)
  | [], .nil => pure .nil
  | _s :: ss, .cons p ps => do
      Internal.syncParamCudaToHost (α := α) (sh := _s) p
      let v ← p.value.get
      let vs ← valuesSynced (α := α) (ss := ss) ps
      pure (.cons v vs)

/-- Overwrite the current parameter values from a `TList` aligned with the shape list. -/
def setValues {α : Type} : {ss : List Shape} → ParamList α ss → TList α ss → IO Unit
  | [], .nil, .nil => pure ()
  | _s :: ss, .cons p ps, .cons v vs => do
      Internal.setParamHostValue (α := α) (sh := _s) p v
      setValues (α := α) (ss := ss) ps vs

/--
Apply an SGD step `p := p - lr * g` to each parameter that has `requiresGrad = true`.

`gs` must be aligned with the parameter shapes.
-/
def sgdStep {α : Type} [Context α] : {ss : List Shape} → ParamList α ss → (lr : α) → TList α ss → IO
  Unit
  | [], .nil, _lr, .nil => pure ()
  | _s :: ss, .cons p ps, lr, .cons g gs => do
      if p.requiresGrad then
        let v ← p.value.get
        let updated : Tensor α _s :=
          -- `Tensor.materialize` prevents long training runs from building deep closure chains
          -- (important for Lean runtime performance).
          Tensor.materialize <| subSpec v (scaleSpec (α := α) (s := _s) g lr)
        Internal.setParamHostValue (α := α) (sh := _s) p updated
      sgdStep (α := α) (ss := ss) ps lr gs

/--
Like `sgdStep`, but uses a fully materialized update (`subScaleMaterialize`) for speed.

This is a runtime performance knob; mathematically it is equivalent to `sgdStep`.
-/
def sgdStepFast {α : Type} [Context α] : {ss : List Shape} → ParamList α ss → (lr : α) → TList α ss
  → IO Unit
  | [], .nil, _lr, .nil => pure ()
  | _s :: ss, .cons p ps, lr, .cons g gs => do
      if p.requiresGrad then
        let v ← p.value.get
        let updated : Tensor α _s :=
          -- Build a materialized tensor in one pass: `v - lr*g`.
          subScaleMaterialize (α := α) (s := _s) v g lr
        Internal.setParamHostValue (α := α) (sh := _s) p updated
      sgdStepFast (α := α) (ss := ss) ps lr gs

end ParamList

/--
Bundle a scalar-loss training loop for a fixed parameter pack and input signature.

This is intended for simple demos:
- `forward` computes a scalar loss,
- `backward` computes gradients w.r.t. parameters,
- `step` applies an optimizer update (typically SGD),
- `getParams` reads current parameter values.
-/
structure ScalarTrainer (α : Type) (paramShapes inputShapes : List Shape) where
  /-- Mutable trainable parameter pack. -/
  params : ParamList α paramShapes
  /-- Compute the scalar loss for a curried input pack. -/
  forward : Curried.Fn α inputShapes (IO (Tensor α Shape.scalar))
  /-- Compute gradients aligned with `paramShapes` for a curried input pack. -/
  backward : Curried.Fn α inputShapes (IO (TList α paramShapes))
  /-- Apply one SGD-style update for a curried input pack. -/
  step : α → Curried.Fn α inputShapes (IO Unit)
  /--
  Optional Adam update path.

  In eager CUDA mode this is a device-gradient/device-moment update path.  Other backends expose
  `none` and should use the generic optimizer wrappers.
  -/
  adamStep? : Option (α → α → α → α → Curried.Fn α inputShapes (IO Unit)) := none
  /--
  Optional AdamW update path.

  In eager CUDA mode this is a device-gradient/device-moment update path with decoupled weight
  decay. Other backends expose `none` and should use the generic optimizer wrappers.
  -/
  adamWStep? : Option (α → α → α → α → α → Curried.Fn α inputShapes (IO Unit)) := none
  /-- Read current parameter values, synchronizing device mirrors if needed. -/
  getParams : IO (TList α paramShapes)

namespace Internal

/--
Extract gradients (as a typed `TList`) for a list of eager `TensorRef`s from a dense gradient array.
-/
def gradsOfRefs {α : Type} [DecidableEq Shape] :
    {ss : List Shape} → Array (Runtime.AnyTensor α) → RefList (TensorRef α) ss → IO (TList α ss)
  | [], _grads, .nil => pure .nil
  | s :: ss, grads, .cons r rs => do
      let g ← Internal.EagerSession.grad (α := α) (sh := s) grads r
      let gs ← gradsOfRefs (α := α) (ss := ss) grads rs
      pure (.cons g gs)

/--
Record all parameters as tape leaves in an eager session, returning their corresponding
  `TensorRef`s.

This is the eager analogue of "using" a parameter pack during a forward pass.
-/
def useParams {α : Type} [CudaBridge.TensorConv α] [DecidableEq Shape] :
    {ss : List Shape} → ParamList α ss → EagerM α (RefList (TensorRef α) ss)
  | [], .nil => pure .nil
  | s :: ss, .cons p ps => fun sess => do
      let r ← Internal.EagerSession.use (α := α) (sh := s) sess p
      let rs ← useParams (α := α) (ss := ss) ps sess
      pure (.cons r rs)

/--
Record all input tensors as tape leaves in an eager session, returning their corresponding
  `TensorRef`s.
-/
def useInputs {α : Type} [CudaBridge.TensorConv α] [DecidableEq Shape] :
    {ss : List Shape} → TList α ss → EagerM α (RefList (TensorRef α) ss)
  | [], .nil => pure .nil
  | s :: ss, .cons x xs => fun sess => do
      let r ← Internal.EagerSession.input (α := α) (sh := s) sess x
      let rs ← useInputs (α := α) (ss := ss) xs sess
      pure (.cons r rs)

end Internal

/--
Build a `ScalarTrainer` from an initial parameter pack and a backend-generic loss definition.

`loss` is written once against the `Ops` interface over a concatenated context
`paramShapes ++ inputShapes`. Depending on `opts.backend`, we either:
- compile the loss once (compiled backend), or
- execute it eagerly by building a runtime tape each step (eager backend).
-/
def scalarTrainer {α : Type} [Context α] [Internal.CudaBridge.TensorConv α] [DecidableEq Shape]
    {paramShapes inputShapes : List Shape} (opts : Options := {})
    (initRequiresGrad : List Bool := List.replicate paramShapes.length true)
    (loss :
      ∀ {m : Type → Type}, [Monad m] → [Ops (m := m) (α := α)] →
        CurriedRef (fun s => Ops.Ref (m := m) (α := α) s) (paramShapes ++ inputShapes)
          (m (Ops.Ref (m := m) (α := α) Shape.scalar))) :
    Curried.Fn α paramShapes (IO (ScalarTrainer α paramShapes inputShapes)) :=
  Curried.curry (α := α) (ss := paramShapes) (β := IO (ScalarTrainer α paramShapes inputShapes))
    (fun initParams => do
    let ps ← ParamList.ofTListWithRequiresGrad (α := α) initParams initRequiresGrad
    match opts.backend with
    | .compiled =>
        let Γ : List Shape := paramShapes ++ inputShapes
        let build : Runtime.Autograd.Compiled.GraphM.M α Γ (Runtime.Autograd.Compiled.GraphM.Var
          Shape.scalar) := do
          let vs ← Runtime.Autograd.Compiled.GraphM.args (α := α) (Γ := Γ)
          CurriedRef.applyVarList (Γ := Γ) (β := Runtime.Autograd.Compiled.GraphM.M α Γ
            (Runtime.Autograd.Compiled.GraphM.Var Shape.scalar))
            (loss (m := Runtime.Autograd.Compiled.GraphM.M α Γ)) vs
        let compiled ← okOrThrow (compileScalar (α := α) (Γ := Γ) build)
        let ssFull : List Shape := compiled.ssPrev ++ [Shape.scalar]
        let fullGraph : Proofs.Autograd.Algebra.GraphData α Unit Γ ssFull :=
          .snoc (ss := compiled.ssPrev) (τ := Shape.scalar) compiled.gPrev compiled.node
        let outId : Nat := Runtime.Autograd.Compiled.outId (Γ := Γ) (ss := ssFull)

        let getScalarFromTape (t : Runtime.Autograd.Tape α) : IO (Tensor α Shape.scalar) := do
          let any ← match t.getValue? outId with
            | some v => pure v
            | none => throw <| IO.userError "torch.compile: missing output value in compiled tape"
          if h : any.s = Shape.scalar then
            pure (Tensor.castShape any.t h)
          else
            throw <| IO.userError
              s!"torch.compile: output shape mismatch (expected scalar, got {Shape.pretty any.s})"

        let rec gradsPrefix :
            {ss : List Shape} → Array (Runtime.AnyTensor α) → Nat → IO (TList α ss)
          | [], _grads, _off => pure .nil
          | s :: ss, grads, off => do
              let any ← match grads[off]? with
                | some v => pure v
                | none => throw <| IO.userError "torch.compile: gradient array too small"
              if h : any.s = s then
                let g : Tensor α s := Tensor.castShape any.t h
                let gs ← gradsPrefix (ss := ss) grads (off + 1)
                pure (.cons g gs)
              else
                throw <| IO.userError <|
                  s!"torch.compile: gradient shape mismatch at idx={off} (expected "
                    ++ s!"{Shape.pretty s}, got "
                    ++ s!"{Shape.pretty any.s})"

        let forward : Curried.Fn α inputShapes (IO (Tensor α Shape.scalar)) :=
          Curried.curry (α := α) (ss := inputShapes) (β := IO (Tensor α Shape.scalar)) (fun xs => do
            let pv ← ParamList.values (α := α) ps
            let args := Proofs.Autograd.Algebra.TList.append (α := α) (ss₁ := paramShapes) (ss₂ :=
              inputShapes) pv xs
            let (tape, _ctx) := Runtime.Autograd.Compiled.compile (α := α) (Γ := Γ) (ss := ssFull)
              fullGraph args
            getScalarFromTape tape)
        let backward : Curried.Fn α inputShapes (IO (TList α paramShapes)) :=
          Curried.curry (α := α) (ss := inputShapes) (β := IO (TList α paramShapes)) (fun xs => do
            let pv ← ParamList.values (α := α) ps
            let args := Proofs.Autograd.Algebra.TList.append (α := α) (ss₁ := paramShapes) (ss₂ :=
              inputShapes) pv xs
            let (tape, _ctx) := Runtime.Autograd.Compiled.compile (α := α) (Γ := Γ) (ss := ssFull)
              fullGraph args
            let grads ← okOrThrow (Runtime.Autograd.Compiled.backwardDenseAllFromOutput (α := α) (Γ
              := Γ) (ss := ssFull) tape)
            gradsPrefix (ss := paramShapes) grads 0)
        let step (lr : α) : Curried.Fn α inputShapes (IO Unit) :=
          Curried.curry (α := α) (ss := inputShapes) (β := IO Unit) (fun xs => do
            let g ← Curried.uncurry (α := α) (ss := inputShapes) (β := IO (TList α paramShapes))
              backward xs
            if opts.fastKernels then
              ParamList.sgdStepFast (α := α) (ss := paramShapes) ps lr g
            else
              ParamList.sgdStep (α := α) (ss := paramShapes) ps lr g)
        pure
          { params := ps
            forward := forward
            backward := backward
            step := step
            adamStep? := none
            adamWStep? := none
            getParams := ParamList.values (α := α) (ss := paramShapes) ps }
    | .eager =>
        let sess ← Internal.EagerSession.new (α := α) opts
        let adamStateRef ← IO.mkRef (Std.HashMap.emptyWithCapacity : Internal.EagerSession.CudaAdamState)
        let lossEager := loss (m := Internal.EagerM α)
        let forward : Curried.Fn α inputShapes (IO (Tensor α Shape.scalar)) :=
          Curried.curry (α := α) (ss := inputShapes) (β := IO (Tensor α Shape.scalar)) (fun xs => do
            sess.resetTape
            let lossRef ← (do
              let pRefs ← Internal.useParams (α := α) (ss := paramShapes) ps
              let xRefs ← Internal.useInputs (α := α) (ss := inputShapes) xs
              let allRefs := RefList.append (ss₁ := paramShapes) (ss₂ := inputShapes) pRefs xRefs
              CurriedRef.uncurry (ss := paramShapes ++ inputShapes) (lossEager) allRefs) |>.run sess
            Internal.EagerSession.getValue (α := α) sess (sh := Shape.scalar) lossRef)
        let backward : Curried.Fn α inputShapes (IO (TList α paramShapes)) :=
          Curried.curry (α := α) (ss := inputShapes) (β := IO (TList α paramShapes)) (fun xs => do
            sess.resetTape
            let (lossRef, pRefs) ← (do
              let pRefs ← Internal.useParams (α := α) (ss := paramShapes) ps
              let xRefs ← Internal.useInputs (α := α) (ss := inputShapes) xs
              let allRefs := RefList.append (ss₁ := paramShapes) (ss₂ := inputShapes) pRefs xRefs
              let lossRef ← CurriedRef.uncurry (ss := paramShapes ++ inputShapes) (lossEager) allRefs
              pure (lossRef, pRefs)) |>.run sess
            let grads ← Internal.EagerSession.backwardScalarDenseAll (α := α) sess lossRef
            Internal.gradsOfRefs (α := α) (ss := paramShapes) grads pRefs)
        let step (lr : α) : Curried.Fn α inputShapes (IO Unit) :=
          Curried.curry (α := α) (ss := inputShapes) (β := IO Unit) (fun xs => do
            if opts.useGpu then
              sess.resetTape
              let lossRef ← (do
                let pRefs ← Internal.useParams (α := α) (ss := paramShapes) ps
                let xRefs ← Internal.useInputs (α := α) (ss := inputShapes) xs
                let allRefs := RefList.append (ss₁ := paramShapes) (ss₂ := inputShapes) pRefs xRefs
                CurriedRef.uncurry (ss := paramShapes ++ inputShapes) (lossEager) allRefs) |>.run sess
              let gradsDev ← Internal.EagerSession.backwardScalarDenseAllCuda (α := α) sess lossRef
              Internal.EagerSession.sgdStepAllCuda (α := α) sess lr gradsDev
              Internal.EagerSession.releaseCudaAnyBufferArray gradsDev
              Internal.EagerSession.releaseCudaTapeAfterOptimizerStep sess
              sess.cudaTape.set Runtime.Autograd.Cuda.Tape.empty
              sess.paramsByLeaf.set (Std.HashMap.emptyWithCapacity)
              sess.nats.set #[]
              Internal.EagerSession.collectCudaAllocator
            else
              let g ← Curried.uncurry (α := α) (ss := inputShapes) (β := IO (TList α paramShapes))
                backward xs
              if opts.fastKernels then
                ParamList.sgdStepFast (α := α) (ss := paramShapes) ps lr g
              else
                ParamList.sgdStep (α := α) (ss := paramShapes) ps lr g)
        let adamStep? : Option (α → α → α → α → Curried.Fn α inputShapes (IO Unit)) :=
          if opts.useGpu then
            some (fun lr beta1 beta2 epsilon =>
              Curried.curry (α := α) (ss := inputShapes) (β := IO Unit) (fun xs => do
                sess.resetTape
                let lossRef ← (do
                  let pRefs ← Internal.useParams (α := α) (ss := paramShapes) ps
                  let xRefs ← Internal.useInputs (α := α) (ss := inputShapes) xs
                  let allRefs := RefList.append (ss₁ := paramShapes) (ss₂ := inputShapes) pRefs xRefs
                  CurriedRef.uncurry (ss := paramShapes ++ inputShapes) (lossEager) allRefs) |>.run sess
                let gradsDev ← Internal.EagerSession.backwardScalarDenseAllCuda (α := α) sess lossRef
                Internal.EagerSession.adamStepAllCuda (α := α) sess adamStateRef lr beta1 beta2
                  epsilon gradsDev
                Internal.EagerSession.releaseCudaAnyBufferArray gradsDev
                Internal.EagerSession.releaseCudaTapeAfterOptimizerStep sess
                sess.cudaTape.set Runtime.Autograd.Cuda.Tape.empty
                sess.paramsByLeaf.set (Std.HashMap.emptyWithCapacity)
                sess.nats.set #[]
                Internal.EagerSession.collectCudaAllocator))
          else
            none
        let adamWStep? : Option (α → α → α → α → α → Curried.Fn α inputShapes (IO Unit)) :=
          if opts.useGpu then
            some (fun lr weightDecay beta1 beta2 epsilon =>
              Curried.curry (α := α) (ss := inputShapes) (β := IO Unit) (fun xs => do
                sess.resetTape
                let lossRef ← (do
                  let pRefs ← Internal.useParams (α := α) (ss := paramShapes) ps
                  let xRefs ← Internal.useInputs (α := α) (ss := inputShapes) xs
                  let allRefs := RefList.append (ss₁ := paramShapes) (ss₂ := inputShapes) pRefs xRefs
                  CurriedRef.uncurry (ss := paramShapes ++ inputShapes) (lossEager) allRefs) |>.run sess
                let gradsDev ← Internal.EagerSession.backwardScalarDenseAllCuda (α := α) sess lossRef
                Internal.EagerSession.adamWStepAllCuda (α := α) sess adamStateRef lr weightDecay
                  beta1 beta2 epsilon gradsDev
                Internal.EagerSession.releaseCudaAnyBufferArray gradsDev
                Internal.EagerSession.releaseCudaTapeAfterOptimizerStep sess
                sess.cudaTape.set Runtime.Autograd.Cuda.Tape.empty
                sess.paramsByLeaf.set (Std.HashMap.emptyWithCapacity)
                sess.nats.set #[]
                Internal.EagerSession.collectCudaAllocator))
          else
            none
        pure
          { params := ps
            forward := forward
            backward := backward
            step := step
            adamStep? := adamStep?
            adamWStep? := adamWStep?
            getParams := ParamList.valuesSynced (α := α) (ss := paramShapes) ps })

end Torch
end Autograd
end Runtime
