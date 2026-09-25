/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Torch.Core.OptimizerCheckpoint.Schema
public import NN.Runtime.Autograd.Torch.Core.Session

/-!
# CUDA Adam Checkpoints

The eager trainer records parameter leaves before input leaves on every step. Their tape identifiers
are the stable zero-based parameter slots used by this internal format. The reader checks every slot
against the shared optimizer schema before installing any state.
-/

@[expose] public section

namespace Runtime
namespace Autograd
namespace Torch
namespace Internal

namespace EagerSession

/-- Device-side Adam moment buffers for one parameter leaf. -/
structure CudaAdamParamState where
  /-- First moment buffer. -/
  m : Runtime.Autograd.Cuda.Buffer
  /-- Second moment buffer. -/
  v : Runtime.Autograd.Cuda.Buffer
  /-- Adam step counter for this parameter. -/
  t : Nat

/-- Adam moment state keyed by the eager trainer's parameter-leaf slot. -/
abbrev CudaAdamState := Std.HashMap Nat CudaAdamParamState

/-- The update rule associated with a CUDA Adam-family state. -/
inductive CudaAdamKind where
  | adam
  | adamW
  deriving BEq, Repr

/-- Hyperparameters that determine the meaning of retained Adam-family moments. -/
structure CudaAdamConfig where
  kind : CudaAdamKind
  beta1 : Float
  beta2 : Float
  epsilon : Float
  weightDecay : Float

/-- Human-readable name used in checkpoint diagnostics. -/
def checkpointName : String :=
  "CUDA Adam checkpoint"

namespace CudaAdamConfig

/-- Compare configurations by exact floating-point bit pattern. -/
def sameBits (left right : CudaAdamConfig) : Bool :=
  left.kind == right.kind &&
    left.beta1.toBits == right.beta1.toBits &&
    left.beta2.toBits == right.beta2.toBits &&
    left.epsilon.toBits == right.epsilon.toBits &&
    left.weightDecay.toBits == right.weightDecay.toBits

/--
Reject invalid Adam-family hyperparameters before updating or restoring device state.

The denominator uses float32 arithmetic, so epsilon must remain finite and positive after the
same conversion performed by the native kernel.
-/
def validate (config : CudaAdamConfig) : Except String Unit := do
  unless config.beta1.isFinite && 0.0 ≤ config.beta1 && config.beta1 < 1.0 do
    throw s!"{checkpointName}: `beta1` must be finite and lie in [0, 1)"
  unless config.beta2.isFinite && 0.0 ≤ config.beta2 && config.beta2 < 1.0 do
    throw s!"{checkpointName}: `beta2` must be finite and lie in [0, 1)"
  let effectiveEpsilon := config.epsilon.toFloat32.toFloat
  unless effectiveEpsilon.isFinite && 0.0 < effectiveEpsilon do
    throw s!"{checkpointName}: `epsilon` must remain finite and positive in float32"
  match config.kind with
  | .adam =>
      unless config.weightDecay == 0.0 do
        throw s!"{checkpointName}: Adam checkpoints must have zero weight decay"
  | .adamW =>
      unless config.weightDecay.isFinite && 0.0 ≤ config.weightDecay do
        throw s!"{checkpointName}: AdamW weight decay must be finite and nonnegative"

end CudaAdamConfig

/-- Version 2 header, retained for layouts with independent trainable parameters. -/
def cudaAdamCheckpointFormat : CheckpointIO.Format where
  name := checkpointName
  magic := "TLADAMF".toUTF8
  version := 2

/-- Version 3 records storage sharing while retaining the original parameter-slot identifiers. -/
def cudaAdamAliasCheckpointFormat : CheckpointIO.Format :=
  { cudaAdamCheckpointFormat with version := 3 }

/-- Version 4 also restores the eager session's random-operation counter. -/
def cudaAdamRandomCheckpointFormat : CheckpointIO.Format :=
  { cudaAdamCheckpointFormat with version := 4 }

/--
Read a supported header and reject a legacy checkpoint when the destination shares parameters.

Version 2 contains no sharing information. Its per-slot histories cannot establish which moment
pair belongs to shared storage, so an aliased trainer must start a fresh history or load version 3
or later. Version 4 carries the random-operation counter as well.
The check precedes config parsing and all device allocation.
-/
def readCudaAdamCheckpointFormat
    (handle : IO.FS.Handle) (schema : OptimizerCheckpoint.ParameterSchema) :
    IO CheckpointIO.Format := do
  let magic ← CheckpointIO.readExact checkpointName handle cudaAdamCheckpointFormat.magic.size
  if magic != cudaAdamCheckpointFormat.magic then
    throw <| IO.userError s!"{checkpointName}: unsupported checkpoint family"
  let version ← CheckpointIO.readNat64 checkpointName handle
  match version with
  | 2 => do
      if schema.hasAliases then
        throw <| IO.userError
          s!"{checkpointName}: version 2 cannot restore shared parameter histories"
      pure cudaAdamCheckpointFormat
  | 3 => pure cudaAdamAliasCheckpointFormat
  | 4 => pure cudaAdamRandomCheckpointFormat
  | _ =>
      throw <| IO.userError
        s!"{checkpointName}: unsupported format version {version}; expected 2, 3, or 4"

/-- Release every device buffer owned by an Adam state map. -/
def releaseCudaAdamState (state : CudaAdamState) : IO Unit := do
  for (_, entry) in state.toList do
    releaseCudaBuffer entry.m
    releaseCudaBuffer entry.v

/--
Bind an in-memory moment map to one optimizer configuration.

Changing Adam to AdamW, or changing a moment-defining hyperparameter, requires a fresh state rather
than silently reinterpreting existing moments.
-/
def ensureCudaAdamConfig
    (configRef : IO.Ref (Option CudaAdamConfig)) (expected : CudaAdamConfig) : IO Unit := do
  match expected.validate with
  | .ok () => pure ()
  | .error message => throw <| IO.userError message
  match ← configRef.get with
  | none => configRef.set (some expected)
  | some actual =>
      unless actual.sameBits expected do
        throw <| IO.userError
          "torch: CUDA Adam state belongs to a different optimizer configuration"

/-- Serialize the hyperparameters as a tag plus four raw bit patterns.

Bits rather than decimal text: a checkpoint has to restore the same moments bit for bit, and a
decimal round trip of a `Float` is exactly where that guarantee would be lost. -/
def writeConfig (handle : IO.FS.Handle) (config : CudaAdamConfig) : IO Unit := do
  let kindTag := match config.kind with
    | .adam => 0
    | .adamW => 1
  CheckpointIO.writeNat64 checkpointName handle kindTag
  CheckpointIO.writeNat64 checkpointName handle config.beta1.toBits.toNat
  CheckpointIO.writeNat64 checkpointName handle config.beta2.toBits.toNat
  CheckpointIO.writeNat64 checkpointName handle config.epsilon.toBits.toNat
  CheckpointIO.writeNat64 checkpointName handle config.weightDecay.toBits.toNat

/-- Inverse of `writeConfig`. An unrecognized kind tag fails loudly rather than defaulting to Adam,
since silently reading AdamW moments as Adam moments would corrupt training quietly. -/
def readConfig (handle : IO.FS.Handle) : IO CudaAdamConfig := do
  let kind ← match ← CheckpointIO.readNat64 checkpointName handle with
    | 0 => pure CudaAdamKind.adam
    | 1 => pure CudaAdamKind.adamW
    | tag => throw <| IO.userError s!"{checkpointName}: invalid optimizer tag {tag}"
  let beta1 := Float.ofBits (UInt64.ofNat (← CheckpointIO.readNat64 checkpointName handle))
  let beta2 := Float.ofBits (UInt64.ofNat (← CheckpointIO.readNat64 checkpointName handle))
  let epsilon := Float.ofBits (UInt64.ofNat (← CheckpointIO.readNat64 checkpointName handle))
  let weightDecay :=
    Float.ofBits (UInt64.ofNat (← CheckpointIO.readNat64 checkpointName handle))
  let config : CudaAdamConfig := { kind, beta1, beta2, epsilon, weightDecay }
  match config.validate with
  | .ok () => pure config
  | .error message => throw <| IO.userError message

/--
Stream CUDA Adam or AdamW moments to disk.

The file includes the optimizer configuration and the complete ordered parameter schema. Shared
parameters use version 3 and one moment entry per representative; independent parameters keep the
version 2 encoding. Supplying the session counter selects version 4, which retains both storage
sharing and the next random key. Saving before the first Adam-family update is rejected because
no optimizer state has yet been defined.
-/
def writeCudaAdamStateFloat32
    (path : System.FilePath) (schema : OptimizerCheckpoint.ParameterSchema)
    (configRef : IO.Ref (Option CudaAdamConfig)) (stateRef : IO.Ref CudaAdamState)
    (randomCounter : Option Nat := none) : IO Unit := do
  unless schema.isWellFormed do
    throw <| IO.userError s!"{checkpointName}: malformed in-memory parameter schema"
  let config ← match ← configRef.get with
    | some config => pure config
    | none => throw <| IO.userError <|
        s!"{checkpointName}: no Adam or AdamW update has initialized optimizer state"
  match config.validate with
  | .ok () => pure ()
  | .error message => throw <| IO.userError message
  let state ← stateRef.get
  if state.size != schema.optimizerStateCount then
    throw <| IO.userError <|
      s!"{checkpointName}: expected {schema.optimizerStateCount} moment entries, got {state.size}"
  let entries := state.toList.mergeSort (fun left right => left.1 ≤ right.1)
  let format :=
    if randomCounter.isSome then cudaAdamRandomCheckpointFormat
    else if schema.hasAliases then cudaAdamAliasCheckpointFormat else cudaAdamCheckpointFormat
  CheckpointIO.writeAtomically path fun handle => do
    CheckpointIO.writeFormat format handle
    writeConfig handle config
    if let some counter := randomCounter then
      CheckpointIO.writeNat64 checkpointName handle counter
    schema.write format handle
    if format.version ≥ 3 then
      schema.writeRepresentatives format handle
    CheckpointIO.writeNat64 checkpointName handle entries.length
    for (id, entry) in entries do
      let shape ← match schema.shapes[id]? with
        | some shape => pure shape
        | none => throw <| IO.userError s!"{checkpointName}: invalid parameter id {id}"
      unless schema.isRepresentative id do
        throw <| IO.userError s!"{checkpointName}: parameter {id} does not own optimizer state"
      if entry.t == 0 then
        throw <| IO.userError s!"{checkpointName}: zero step counter for parameter {id}"
      let count := Spec.Shape.size shape
      if (Runtime.Autograd.Cuda.Buffer.size entry.m).toNat != count ||
          (Runtime.Autograd.Cuda.Buffer.size entry.v).toNat != count then
        throw <| IO.userError s!"{checkpointName}: moment-size mismatch for parameter {id}"
      let mBytes ← Runtime.Autograd.Cuda.Buffer.toFloat32BytesIO entry.m
      let vBytes ← Runtime.Autograd.Cuda.Buffer.toFloat32BytesIO entry.v
      if mBytes.size != count * 4 || vBytes.size != count * 4 then
        throw <| IO.userError s!"{checkpointName}: invalid float32 payload for parameter {id}"
      CheckpointIO.writeNat64 checkpointName handle id
      CheckpointIO.writeNat64 checkpointName handle entry.t
      CheckpointIO.writeNat64 checkpointName handle count
      handle.write mBytes
      handle.write vBytes

/-- Read a whole CUDA Adam checkpoint: metadata, then one moment pair per representative.

The state is built in a local map and only installed once every parameter has been read and checked
against `schema`, so a truncated file leaves the live optimizer untouched. -/
def readCheckpoint
    (handle : IO.FS.Handle) (schema : OptimizerCheckpoint.ParameterSchema)
    (expectedConfig : Option CudaAdamConfig := none) :
    IO (CudaAdamConfig × CudaAdamState × Option Nat) := do
  unless schema.isWellFormed do
    throw <| IO.userError s!"{checkpointName}: malformed expected parameter schema"
  let mut replacement : CudaAdamState := Std.HashMap.emptyWithCapacity
  try
    let format ← readCudaAdamCheckpointFormat handle schema
    let config ← readConfig handle
    let randomCounter ← if format.version ≥ 4 then
        some <$> CheckpointIO.readNat64 checkpointName handle
      else pure none
    if let some expected := expectedConfig then
      unless config.sameBits expected do
        throw <| IO.userError <|
          s!"{checkpointName}: checkpoint belongs to a different optimizer configuration"
    schema.readAndCheck format handle
    if format.version ≥ 3 then
      schema.readAndCheckRepresentatives format handle
    let entryCount ← CheckpointIO.readNat64 checkpointName handle
    if entryCount != schema.optimizerStateCount then
      throw <| IO.userError <|
        s!"{checkpointName}: expected {schema.optimizerStateCount} moment entries, got {entryCount}"
    for _ in [0:entryCount] do
      let id ← CheckpointIO.readNat64 checkpointName handle
      let step ← CheckpointIO.readNat64 checkpointName handle
      let count ← CheckpointIO.readNat64 checkpointName handle
      if replacement.contains id then
        throw <| IO.userError s!"{checkpointName}: duplicate parameter id {id}"
      let shape ← match schema.shapes[id]? with
        | some shape => pure shape
        | none => throw <| IO.userError s!"{checkpointName}: invalid parameter id {id}"
      unless schema.isRepresentative id do
        throw <| IO.userError s!"{checkpointName}: parameter {id} does not own optimizer state"
      if step = 0 then
        throw <| IO.userError s!"{checkpointName}: zero step counter for parameter {id}"
      if count != Spec.Shape.size shape then
        throw <| IO.userError <|
          s!"{checkpointName}: moment-size mismatch for parameter {id} " ++
            s!"(file={count}, expected={Spec.Shape.size shape})"
      let mBytes ← CheckpointIO.readExact checkpointName handle (count * 4)
      let vBytes ← CheckpointIO.readExact checkpointName handle (count * 4)
      let m ← Runtime.Autograd.Cuda.Buffer.ofFloat32BytesIO mBytes
      let v ← try
        Runtime.Autograd.Cuda.Buffer.ofFloat32BytesIO vBytes
      catch error =>
        releaseCudaBuffer m
        throw error
      replacement := replacement.insert id { m, v, t := step }
    let trailing ← handle.read 1
    if trailing.size != 0 then
      throw <| IO.userError s!"{checkpointName}: trailing bytes after final state entry"
    pure (config, replacement, randomCounter)
  catch error =>
    releaseCudaAdamState replacement
    throw error

/--
Restore CUDA Adam or AdamW moments after validating their optimizer and parameter metadata.

The old state remains live until the replacement has been parsed completely. Duplicate ids,
frozen or out-of-range parameters, zero step counters, trailing bytes, and truncated payloads are
rejected.
-/
def readCudaAdamStateFloat32
    (path : System.FilePath) (schema : OptimizerCheckpoint.ParameterSchema)
    (configRef : IO.Ref (Option CudaAdamConfig)) (stateRef : IO.Ref CudaAdamState)
    (rngCounter? : Option (IO.Ref Nat) := none) : IO Unit := do
  unless schema.isWellFormed do
    throw <| IO.userError s!"{checkpointName}: malformed in-memory parameter schema"
  let expectedConfig ← configRef.get
  let (config, replacement, randomCounter) ←
    IO.FS.withFile path IO.FS.Mode.read fun handle =>
      readCheckpoint handle schema expectedConfig
  if randomCounter.isSome && rngCounter?.isNone then
    releaseCudaAdamState replacement
    throw <| IO.userError s!"{checkpointName}: destination has no random-counter reference"
  match ← configRef.get with
  | some expected =>
      unless config.sameBits expected do
        releaseCudaAdamState replacement
        throw <| IO.userError <|
          s!"{checkpointName}: checkpoint belongs to a different optimizer configuration"
  | none => pure ()
  let previous ← stateRef.get
  stateRef.set replacement
  configRef.set (some config)
  match rngCounter?, randomCounter with
  | some counter, some value => counter.set value
  | some _, none =>
      IO.eprintln s!"{checkpointName}: legacy checkpoint has no random counter; \
        stochastic training cannot be resumed exactly"
  | _, _ => pure ()
  releaseCudaAdamState previous

end EagerSession
end Internal
end Torch
end Autograd
end Runtime
