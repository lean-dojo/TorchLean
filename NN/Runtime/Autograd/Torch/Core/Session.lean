/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Torch.Core.Types
public import NN.Floats.IEEEExec.Exec32.Compare
public import NN.Backend.Report

/-!
# Eager Session and CUDA Bridge

Session state, CUDA upload/download conversions, parameter synchronization, and tape lifecycle
helpers for the eager Torch-style runtime.
-/


@[expose] public section

namespace Runtime
namespace Autograd
namespace Torch

open Spec
open Tensor
open Proofs.Autograd.Algebra

/-!
## Internal Eager Session

The eager backend keeps one mutable CPU tape, one mutable CUDA tape, the non-differentiable `Nat`
environment, and the map from tape leaves back to trainable parameters. The public session-style API
lives in `Runtime.Autograd.TorchLean.Session`; this module owns the lower-level state it delegates
to.
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
    let b ← Runtime.Autograd.Cuda.Buffer.ofFloatArrayIO a
    pure { s := s, buf := b }
  ofAnyBuffer := fun any => do
    let a := Runtime.Autograd.Cuda.Buffer.toFloatArray any.buf
    if a.size != Shape.size any.s then
      throw <| IO.userError
        s!"torch: cuda: bad buffer length (expected {Shape.size any.s}, got {a.size})"
    let t : Tensor Float any.s :=
      Runtime.Autograd.Cuda.Convert.unflattenFloatUnsafe (s := any.s) a
    pure { s := any.s, t := t }
  toFloat := fun x => pure x

/-! #### Executable IEEE32 host scalar implementation -/

/--
Host-side conversion for TorchLean's executable IEEE-754 binary32 scalar.

`IEEE32Exec` is a Lean-defined bit-level scalar semantics, not the CUDA eager tape's float32 device
wire format. We allow scalar readback to `Float` so CPU examples can print predictions and
summaries, but CUDA upload/download remains an explicit unsupported boundary.
-/
instance (priority := 1000) : TensorConv TorchLean.Floats.IEEE754.IEEE32Exec where
  toAnyBuffer := fun {_s} _ =>
    throw <| IO.userError
      "torch: cuda: IEEE32Exec has host-side scalar conversion only; use Float for eager CUDA"
  ofAnyBuffer := fun _ =>
    throw <| IO.userError
      "torch: cuda: IEEE32Exec has host-side scalar conversion only; use Float for eager CUDA"
  toFloat := fun x => pure (TorchLean.Floats.IEEE754.IEEE32Exec.toFloat x)

/--
Generic CPU-preserving fallback for scalar types without a CUDA wire-format bridge.

Many TorchLean sessions are scalar-polymorphic on CPU, while the eager CUDA tape stores float32
buffers. This fallback keeps those CPU instantiations usable and fails loudly if a CUDA upload is
requested for a scalar type that has no declared float32 wire representation. Add a
higher-priority `TensorConv α` instance when a scalar type should be allowed onto the CUDA tape.
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

This synchronization point is explicit. Training hot paths keep parameters resident on device; public
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
  /-- CPU eager tape used when `Options.device opts = .cpu`. -/
  tape : IO.Ref (Runtime.Autograd.Tape α)
  /-- CUDA eager tape used when `Options.device opts = .cuda`. -/
  cudaTape : IO.Ref (Runtime.Autograd.Cuda.Tape)
  /-- Map from tape leaf ids to trainable parameter objects. -/
  paramsByLeaf : IO.Ref (Std.HashMap Nat (AnyParam α))
  /-- Non-differentiable integer inputs for dynamic indexing operations. -/
  nats : IO.Ref (Array Nat)
  /-- Accepted capsules already reported for this session, in first-use order. -/
  selectedBackends : IO.Ref (List NN.Backend.AcceptedKernel)

namespace EagerSession

/-- Select and validate the capsule that corresponds to this eager session's concrete executor. -/
def selectedCapsule {α : Type} (s : EagerSession α) (op : NN.Backend.BackendOp) :
    IO NN.Backend.KernelCapsule := do
  let planned ←
    match s.opts.planBackendOp op with
    | .ok planned => pure planned
    | .error msg => throw <| IO.userError s!"torch: backend planning failed for `{op.name}`: {msg}"
  let capsule := planned.capsule
  if s.opts.usesCuda then
    unless capsule.provider == NN.Backend.Provider.nativeCuda &&
        capsule.device == NN.Backend.Device.cuda do
      throw <| IO.userError <|
        s!"torch: CUDA op `{op.name}` selected capsule `{capsule.name}`, " ++
          "but this eager path is wired to native CUDA"
  else
    unless capsule.provider == NN.Backend.Provider.reference &&
        capsule.device == NN.Backend.Device.cpu do
      throw <| IO.userError <|
        s!"torch: CPU op `{op.name}` selected capsule `{capsule.name}`, " ++
          "but this eager path is wired to the portable reference provider"
  let selected ← s.selectedBackends.get
  unless selected.any (fun old => old.capsule.sameIdentity capsule) do
    s.selectedBackends.set (selected ++ [planned])
    if s.opts.showBackend then
      let row := NN.Backend.KernelAudit.ofPlannedKernel
        { op := planned.op, capsule := planned.capsule }
      for line in row.detailedReportLines do
        IO.println line
  pure capsule

/-- Accepted capsules actually selected by this eager session. -/
def backendSelections {α : Type} (s : EagerSession α) : IO (List NN.Backend.AcceptedKernel) :=
  s.selectedBackends.get

/-- Allocate a fresh eager session with an empty tape and empty side tables. -/
def new {α : Type} (opts : Options := {}) : IO (EagerSession α) := do
  try
    opts.validateForExecution
  catch e =>
    throw <| IO.userError s!"torch eager session: {e.toString}"
  let tape ← IO.mkRef Runtime.Autograd.Tape.empty
  let cudaTape ← IO.mkRef Runtime.Autograd.Cuda.Tape.empty
  let paramsByLeaf ← IO.mkRef (Std.HashMap.emptyWithCapacity)
  let nats ← IO.mkRef #[]
  let selectedBackends ← IO.mkRef ([] : List NN.Backend.AcceptedKernel)
  if opts.showBackend then
    IO.println "[TorchLean] backend capsules used:"
  pure
    { opts := opts
      tape := tape
      cudaTape := cudaTape
      paramsByLeaf := paramsByLeaf
      nats := nats
      selectedBackends := selectedBackends }

/-- Force-free a CUDA buffer allocation; the external finalizer is safe to call twice. -/
def releaseCudaBuffer (b : Runtime.Autograd.Cuda.Buffer) : IO Unit := do
  let released := Runtime.Autograd.Cuda.Buffer.release b
  AnyParam.observeCudaCleanupFlag released

/-- Force-release a shape-erased CUDA buffer. -/
def releaseCudaAnyBuffer (b : Runtime.Autograd.Cuda.AnyBuffer) : IO Unit :=
  releaseCudaBuffer b.buf

/-!
CUDA optimizers only need gradients for parameter leaves, not for every intermediate tape node.
`CudaGradMap` is the sparse representation used by long eager CUDA runs: keys are tape node ids
for parameter leaves and values are device-resident cotangents with the same shape as that leaf.
-/
abbrev CudaGradMap := Std.HashMap Nat Runtime.Autograd.Cuda.AnyBuffer

/--
Release current CUDA tape values that are not persistent parameter mirrors.

Eager CUDA training creates ephemeral buffers for forward values and backward workspace. Reset paths
call this before discarding the current tape snapshot, while persistent parameter mirrors remain
owned by their `Param` objects.
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

/-- Release a sparse CUDA gradient map after an optimizer has consumed it. -/
def releaseCudaGradMap (xs : CudaGradMap) : IO Unit := do
  for (_id, x) in xs.toList do
    releaseCudaAnyBuffer x

/-- Check that a shape-erased CUDA buffer has the number of elements promised by its shape. -/
def checkCudaAnyBufferSize (where_ : String)
    (x : Runtime.Autograd.Cuda.AnyBuffer) : IO Unit := do
  let expected := Shape.size x.s
  if _hExpected : expected ≤ UInt32.size then
    let got := Runtime.Autograd.Cuda.Buffer.size x.buf
    let expectedU32 : UInt32 := UInt32.ofNat expected
    if got != expectedU32 then
      throw <| IO.userError
        s!"torch: CUDA buffer size mismatch in {where_} \
           (shape={Shape.pretty x.s}, expected={expected}, got={got.toNat})"
  else
    throw <| IO.userError s!"torch: CUDA tensor too large in {where_}"
/-- Make an owned copy of a CUDA buffer after checking its shape metadata. -/
def ownedCudaAnyBuffer (where_ : String)
    (x : Runtime.Autograd.Cuda.AnyBuffer) : IO Runtime.Autograd.Cuda.AnyBuffer := do
  checkCudaAnyBufferSize where_ x
  pure { s := x.s, buf := Runtime.Autograd.Cuda.Buffer.copy x.buf }

/-- Ask the native allocator to return/free pages after a large CUDA eager step. -/
def collectCudaAllocator : IO Unit := do
  let released := Runtime.Autograd.Cuda.Buffer.collectAllocator true
  AnyParam.observeCudaCleanupFlag released

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
Retire the CUDA tape snapshot and side tables and ask the native allocator to return pages.

The caller is responsible for having already released the tape's reclaimable device buffers (via
`releaseCudaTapeAfterOptimizerStep` or `releaseCudaTapeNonParamValues`); this only drops the now-dead
tape state so the next step starts clean.
-/
def retireCudaTapeState {α : Type} (s : EagerSession α) : IO Unit := do
  s.cudaTape.set Runtime.Autograd.Cuda.Tape.empty
  s.paramsByLeaf.set (Std.HashMap.emptyWithCapacity)
  s.nats.set #[]
  collectCudaAllocator

/--
Run one eager training step inside a device-memory scope.

`resetTape` opens the scope; `body` builds the forward tape, runs backward, and applies the optimizer
(which writes each trainable parameter to a *fresh* persistent mirror); on exit every tape-owned
temporary is released and the tape snapshot retired, so the device working set stays flat across
steps regardless of step count.

This is the safe, tape-scoped replacement for a device arena: reclamation is keyed on **tape
membership**, not allocation epoch. The only buffers it releases are tape node values and workspace
(`cleanup`), which are runtime-private — callers hold `TensorRef`/`Param`, never a raw `Buffer` — so
it can never free a buffer the caller still holds. Persistent parameter mirrors are preserved by the
`paramsByLeaf` role check.

Exception-safe: if `body` raises before the optimizer completes, the parameters may still be their
live mirrors, so the scope falls back to `releaseCudaTapeNonParamValues` (which preserves *all*
parameter values) before re-raising — it never frees a live mirror on the error path.
-/
def withStep {α : Type} (s : EagerSession α) (body : IO Unit) : IO Unit := do
  s.resetTape
  try
    body
    -- Success: the optimizer has re-materialized trainable parameters into fresh mirrors, so the
    -- leaf buffers captured on the just-consumed tape are stale and can be reclaimed too.
    if s.opts.usesCuda then
      releaseCudaTapeAfterOptimizerStep s
      retireCudaTapeState s
  catch e =>
    -- Failure: the optimizer may not have run, so trainable leaves may still be the live parameter
    -- mirrors. Use the conservative release that preserves every parameter value.
    if s.opts.usesCuda then
      releaseCudaTapeNonParamValues s
      retireCudaTapeState s
    throw e

/--
Run one eager forward pass inside a device-memory scope.

`resetTape` opens the scope; `body` builds the forward tape and returns a result (typically a
`TensorRef`); `transfer` copies the survivors off-device — e.g. `EagerSession.getValue`, which
downloads to a host `Tensor` — *before* the scope releases every forward temporary and retires the
tape. Persistent parameter mirrors are preserved (`releaseCudaTapeNonParamValues`), so a session can
run an unbounded sequence of forward passes with a flat device working set.

This is the safe, tape-scoped replacement for `withCudaArena` on the inference/eval path. Because
`transfer` runs first, the caller leaves the scope holding host tensors, not device `Buffer`s, so the
release can never invalidate a buffer the caller still references. Exception-safe: the tape is
reclaimed even if `body` or `transfer` raises.
-/
def withForward {α β γ : Type} (s : EagerSession α)
    (transfer : β → IO γ) (body : IO β) : IO γ := do
  s.resetTape
  try
    let b ← body
    let out ← transfer b
    if s.opts.usesCuda then
      releaseCudaTapeNonParamValues s
      retireCudaTapeState s
    pure out
  catch e =>
    if s.opts.usesCuda then
      releaseCudaTapeNonParamValues s
      retireCudaTapeState s
    throw e

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
  let _ ← s.selectedCapsule .randUniform
  if Options.device s.opts == .cuda then
    let t0 ← s.cudaTape.get
    let counter := t0.size
    let key := Runtime.Autograd.TorchLean.Random.keyOf seed counter
    let n32 ← Runtime.Autograd.okOrThrow <|
      Runtime.Autograd.Cuda.AnyBuffer.natToU32Checked (Shape.size sh)
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
  let _ ← s.selectedCapsule .bernoulliMask
  let kpT ← getValue (α := α) s (sh := Shape.scalar) keepProb
  let keepProbVal : α :=
    match kpT with
    | Tensor.scalar v => v
  if Options.device s.opts == .cuda then
    let t0 ← s.cudaTape.get
    let counter := t0.size
    let key := Runtime.Autograd.TorchLean.Random.keyOf seed counter
    let n32 ← Runtime.Autograd.okOrThrow <|
      Runtime.Autograd.Cuda.AnyBuffer.natToU32Checked (Shape.size sh)
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

When `s.opts.trackGradients = false`, the parameter is still registered as a leaf so CUDA cleanup
can recognize persistent parameter buffers, but the leaf itself is marked non-differentiable.
-/
def use {α : Type} [CudaBridge.TensorConv α] (s : EagerSession α) {sh : Shape} [DecidableEq Shape]
  (p : Param α sh) : IO (TensorRef α sh) := do
  let requiresGrad := s.opts.trackGradients && p.requiresGrad
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
          (requires_grad := requiresGrad)
      s.cudaTape.set t1
      pure id
    else
      syncParamCudaToHost (α := α) (sh := sh) p
      let v ← p.value.get
      let t0 ← s.tape.get
      let (t1, id) :=
        Runtime.Autograd.Tape.leaf (t := t0) (s := sh)
          (value := v) (name := p.name) (requires_grad := requiresGrad)
      s.tape.set t1
      pure id
  s.paramsByLeaf.modify (fun m => m.insert id (AnyParam.ofParam p))
  pure { id := id }

/--
Record an external input tensor as a leaf on the tape.

PyTorch comparison: like introducing a tensor into the autograd graph with a chosen
`requires_grad` flag.

The session-level `trackGradients` flag is a final gate on the caller's requested `requiresGrad`.
This keeps inference helpers from accidentally building a trainable tape even when a lower-level
caller asks for a differentiable input.
-/
def input {α : Type} [CudaBridge.TensorConv α] (s : EagerSession α) {sh : Shape} [DecidableEq Shape]
  (v : Tensor α sh) (name : Option String := none) (requiresGrad : Bool := false) :
  IO (TensorRef α sh) := do
  let requiresGrad := s.opts.trackGradients && requiresGrad
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

end EagerSession

end Internal
end Torch
end Autograd
end Runtime
