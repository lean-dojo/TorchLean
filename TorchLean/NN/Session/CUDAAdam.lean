/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.Checkpoint.Schema
public import TorchLean.NN.Session.Lifecycle

/-!
# CUDA Adam Checkpoints

The eager trainer records parameter leaves before input leaves on every step. Their tape identifiers
are the stable zero-based parameter slots used by this internal format. The reader checks every slot
against the shared optimizer schema before installing any state.
-/

@[expose] public section

open TorchLean.Shape (size)

namespace TorchLean

namespace Session

/-- Device-side Adam moment buffers, step counter and bias-correction powers for one parameter
leaf. -/
structure CUDAAdamParamState where
  /-- First moment buffer. -/
  m : LibTorch.Buffer
  /-- Second moment buffer. -/
  v : LibTorch.Buffer
  /-- Adam step counter for this parameter. -/
  t : Nat
  /-- `beta1 ^ t` by the recurrence of `npowRec`, rounded in the parameter dtype. -/
  firstPower : Float
  /-- `beta2 ^ t` by the recurrence of `npowRec`, rounded in the parameter dtype. -/
  secondPower : Float

/-- Adam moment state keyed by the eager trainer's parameter-leaf slot. -/
abbrev CUDAAdamState := Std.HashMap Nat CUDAAdamParamState

/-- The update rule associated with a CUDA Adam-family state. -/
inductive CUDAAdamKind where
  | adam
  | adamW
  deriving BEq, Repr

/-- Hyperparameters that determine the meaning of retained Adam-family moments. -/
structure CUDAAdamConfig where
  /-- Dtype in which the kernel stores and updates the parameters and moments. The coefficient
  fields are host `Float` values for every dtype. -/
  dtype : LibTorch.Dtype := .float32
  /-- Whether the update rule is Adam or AdamW. -/
  kind : CUDAAdamKind
  /-- Decay rate of the first-moment estimate. -/
  beta1 : Float
  /-- Decay rate of the second-moment estimate. -/
  beta2 : Float
  /-- Term added to the denominator of the update. -/
  epsilon : Float
  /-- Weight-decay coefficient of AdamW, which `validate` requires to be zero for Adam. -/
  weightDecay : Float

/-- Human-readable name used in checkpoint diagnostics. -/
def checkpointName : String :=
  "CUDA Adam checkpoint"

namespace CUDAAdamConfig

/-- Compare configurations by exact floating-point bit pattern. -/
def sameBits (left right : CUDAAdamConfig) : Bool :=
  left.dtype == right.dtype && left.kind == right.kind &&
    left.beta1.toBits == right.beta1.toBits &&
    left.beta2.toBits == right.beta2.toBits &&
    left.epsilon.toBits == right.epsilon.toBits &&
    left.weightDecay.toBits == right.weightDecay.toBits

/-- `power * beta` rounded in the parameter dtype, one step of the recurrence of `npowRec`. -/
def nextPower (config : CUDAAdamConfig) (beta power : Float) : Float :=
  if config.dtype == .float32 then (power.toFloat32 * beta.toFloat32).toFloat else power * beta

/-- The bias-correction denominator `1 - power`, rounded in the parameter dtype. -/
def correctionDenominator (config : CUDAAdamConfig) (power : Float) : Float :=
  if config.dtype == .float32 then (1 - power.toFloat32).toFloat else 1 - power

/-- `step` applications of `nextPower` to `1`, stopping at the first power that `nextPower`
returns unchanged, since each later step returns it again. -/
def power (config : CUDAAdamConfig) (beta : Float) (step : Nat) : Float := Id.run do
  let mut current := 1
  for _ in [0:step] do
    let next := config.nextPower beta current
    if next.toBits == current.toBits then break
    current := next
  return current

/--
Reject invalid Adam-family hyperparameters before updating or restoring device state.

`beta1` and `beta2` must be finite and lie in `[0, 1)`. Epsilon must be finite and positive after
conversion to the parameter dtype, as in the native kernel. Adam needs zero weight decay and AdamW
a finite nonnegative one. The encoded dtype is rejected, since configured formats need typed
optimizer state.
-/
def validate (config : CUDAAdamConfig) : Except String Unit := do
  if config.dtype == .encoded then
    throw s!"{checkpointName}: configured formats require typed optimizer state"
  Context.requireWithin s!"{checkpointName}: `beta1`" (some 0) none config.beta1
  Context.requireWithin s!"{checkpointName}: `beta1`" none (some 1) config.beta1 (closed := false)
  Context.requireWithin s!"{checkpointName}: `beta2`" (some 0) none config.beta2
  Context.requireWithin s!"{checkpointName}: `beta2`" none (some 1) config.beta2 (closed := false)
  let effectiveEpsilon := match config.dtype with
    | .float32 => config.epsilon.toFloat32.toFloat
    | .float64 => config.epsilon
    | .encoded => config.epsilon
  Context.requireWithin s!"{checkpointName}: `epsilon` after conversion to {repr config.dtype}"
    (some 0) none effectiveEpsilon (closed := false)
  match config.kind with
  | .adam =>
      unless config.weightDecay == 0.0 do
        throw s!"{checkpointName}: Adam checkpoints must have zero weight decay"
  | .adamW =>
      Context.requireWithin s!"{checkpointName}: AdamW weight decay" (some 0) none
        config.weightDecay

end CUDAAdamConfig

/--
The CUDA Adam checkpoint format: the moment dtype, the optimizer configuration, an optional random
counter, the parameter schema with its storage sharing, then one moment pair per parameter that
owns optimizer state.
-/
def adamCheckpointFormat : Checkpoint.Format where
  name := checkpointName
  magic := "TLADAMF".toUTF8
  version := 5

/-- The recorded moment dtype: `0` binary32 and `1` binary64, the native code of each dtype.
Encoded moments are never written. -/
def dtypeCode : LibTorch.Dtype → Nat
  | .float32 => 0
  | .float64 => 1
  | .encoded => 2

/-- Release every device buffer owned by an Adam state map. -/
def releaseAdam (state : CUDAAdamState) : IO Unit := do
  for (_, entry) in state.toList do
    releaseCUDABuffer entry.m
    releaseCUDABuffer entry.v

/--
Bind an in-memory moment map to one optimizer configuration.

The first call validates and stores `expected`. A later call fails unless `expected` has the same
dtype, kind and coefficient bits (`sameBits`), so switching Adam to AdamW or changing a
hyperparameter needs a fresh state.
-/
def bindAdamConfig
    (configRef : IO.Ref (Option CUDAAdamConfig)) (expected : CUDAAdamConfig) : IO Unit := do
  IO.ofExcept expected.validate
  match ← configRef.get with
  | none => configRef.set (some expected)
  | some actual =>
      unless actual.sameBits expected do
        throw <| IO.userError
          "torch: CUDA Adam state belongs to a different optimizer configuration"

/-- Serialize the hyperparameters as a kind tag plus four raw bit patterns.

Raw bits give back every coefficient bit for bit, so a resumed run computes the same updates. A
decimal round trip of a `Float` can lose that. -/
def writeConfig (handle : IO.FS.Handle) (config : CUDAAdamConfig) : IO Unit := do
  let kindTag := match config.kind with
    | .adam => 0
    | .adamW => 1
  Checkpoint.writeNat64 checkpointName handle kindTag
  Checkpoint.writeNat64 checkpointName handle config.beta1.toBits.toNat
  Checkpoint.writeNat64 checkpointName handle config.beta2.toBits.toNat
  Checkpoint.writeNat64 checkpointName handle config.epsilon.toBits.toNat
  Checkpoint.writeNat64 checkpointName handle config.weightDecay.toBits.toNat

/-- Inverse of `writeConfig`, followed by `validate`. A kind tag other than `0` (Adam) or `1`
(AdamW) is an error, because reading AdamW moments as Adam moments would corrupt training
unnoticed. -/
def readConfig (handle : IO.FS.Handle)
    (dtype : LibTorch.Dtype := .float32) : IO CUDAAdamConfig := do
  let kind ← match ← Checkpoint.readNat64 checkpointName handle with
    | 0 => pure CUDAAdamKind.adam
    | 1 => pure CUDAAdamKind.adamW
    | tag => throw <| IO.userError s!"{checkpointName}: invalid optimizer tag {tag}"
  let beta1 := Float.ofBits (UInt64.ofNat (← Checkpoint.readNat64 checkpointName handle))
  let beta2 := Float.ofBits (UInt64.ofNat (← Checkpoint.readNat64 checkpointName handle))
  let epsilon := Float.ofBits (UInt64.ofNat (← Checkpoint.readNat64 checkpointName handle))
  let weightDecay :=
    Float.ofBits (UInt64.ofNat (← Checkpoint.readNat64 checkpointName handle))
  let config : CUDAAdamConfig := { dtype, kind, beta1, beta2, epsilon, weightDecay }
  IO.ofExcept config.validate
  pure config

/--
Write CUDA Adam or AdamW moments to `path` in `adamCheckpointFormat`, replacing the file
through `Checkpoint.writeAtomically`.

The file holds the moment dtype, the optimizer configuration, the random counter when one is given,
and the ordered parameter schema with its storage sharing, followed by one moment entry per
representative parameter. Saving fails when no Adam-family update or load has set the configuration,
and when the state disagrees with `schema` or the configured dtype or holds a zero step counter.
-/
def saveAdam
    (path : System.FilePath) (schema : Checkpoint.Schema)
    (configRef : IO.Ref (Option CUDAAdamConfig)) (stateRef : IO.Ref CUDAAdamState)
    (randomCounter : Option Nat := none) : IO Unit := do
  unless schema.isWellFormed do
    throw <| IO.userError s!"{checkpointName}: malformed in-memory parameter schema"
  let config ← match ← configRef.get with
    | some config => pure config
    | none => throw <| IO.userError <|
        s!"{checkpointName}: no Adam or AdamW update has initialized optimizer state"
  IO.ofExcept config.validate
  let state ← stateRef.get
  if state.size != schema.optimizerStateCount then
    throw <| IO.userError <|
      s!"{checkpointName}: expected {schema.optimizerStateCount} moment entries, got {state.size}"
  let entries := state.toList.mergeSort (fun left right => left.1 ≤ right.1)
  let format := adamCheckpointFormat
  Checkpoint.writeAtomically path fun handle => do
    Checkpoint.writeFormat format handle
    Checkpoint.writeNat64 checkpointName handle (dtypeCode config.dtype)
    writeConfig handle config
    Checkpoint.writeNat64 checkpointName handle (if randomCounter.isSome then 1 else 0)
    if let some counter := randomCounter then
      Checkpoint.writeNat64 checkpointName handle counter
    schema.write format handle
    schema.writeRepresentatives format handle
    Checkpoint.writeNat64 checkpointName handle entries.length
    for (id, entry) in entries do
      let shape ← match schema.shapes[id]? with
        | some shape => pure shape
        | none => throw <| IO.userError s!"{checkpointName}: invalid parameter id {id}"
      unless schema.isRepresentative id do
        throw <| IO.userError s!"{checkpointName}: parameter {id} does not own optimizer state"
      if entry.t == 0 then
        throw <| IO.userError s!"{checkpointName}: zero step counter for parameter {id}"
      let count := Shape.size shape
      unless LibTorch.Buffer.dtype entry.m == config.dtype &&
          LibTorch.Buffer.dtype entry.v == config.dtype do
        throw <| IO.userError s!"{checkpointName}: moment dtype mismatch for parameter {id}"
      if (LibTorch.Buffer.size entry.m).toNat != count ||
          (LibTorch.Buffer.size entry.v).toNat != count then
        throw <| IO.userError s!"{checkpointName}: moment-size mismatch for parameter {id}"
      let mBytes ← LibTorch.Buffer.toBytesIO entry.m
      let vBytes ← LibTorch.Buffer.toBytesIO entry.v
      if mBytes.size != count * config.dtype.bytes || vBytes.size != count * config.dtype.bytes then
        throw <| IO.userError s!"{checkpointName}: invalid moment payload for parameter {id}"
      Checkpoint.writeNat64 checkpointName handle id
      Checkpoint.writeNat64 checkpointName handle entry.t
      Checkpoint.writeNat64 checkpointName handle count
      handle.write mBytes
      handle.write vBytes

/-- Read a whole CUDA Adam checkpoint: metadata, then one moment pair per representative.

The moments go into a local map, returned only once every entry has been read and checked against
`schema` and no byte follows the last one. A truncated file therefore leaves the live optimizer
untouched. -/
def readCheckpoint
    (handle : IO.FS.Handle) (schema : Checkpoint.Schema)
    (expectedConfig : Option CUDAAdamConfig := none)
    (dtype : LibTorch.Dtype := .float32) :
    IO (CUDAAdamConfig × CUDAAdamState × Option Nat) := do
  unless schema.isWellFormed do
    throw <| IO.userError s!"{checkpointName}: malformed expected parameter schema"
  let replacementRef ← IO.mkRef (Std.HashMap.emptyWithCapacity : CUDAAdamState)
  try
    let format := adamCheckpointFormat
    Checkpoint.readFormat format handle
    let storedDtype ← match ← Checkpoint.readNat64 checkpointName handle with
      | 0 => pure LibTorch.Dtype.float32
      | 1 => pure LibTorch.Dtype.float64
      | _ => throw <| IO.userError s!"{checkpointName}: unsupported moment dtype"
    unless storedDtype == dtype do
      throw <| IO.userError s!"{checkpointName}: moment dtype differs from the destination"
    let config ← readConfig handle dtype
    let hasCounter ← match ← Checkpoint.readNat64 checkpointName handle with
      | 0 => pure false
      | 1 => pure true
      | _ => throw <| IO.userError s!"{checkpointName}: invalid random-counter flag"
    let randomCounter ← if hasCounter then
        some <$> Checkpoint.readNat64 checkpointName handle
      else pure none
    if let some expected := expectedConfig then
      unless config.sameBits expected do
        throw <| IO.userError <|
          s!"{checkpointName}: checkpoint belongs to a different optimizer configuration"
    schema.readAndCheck format handle
    schema.readRepresentatives format handle
    let entryCount ← Checkpoint.readNat64 checkpointName handle
    if entryCount != schema.optimizerStateCount then
      throw <| IO.userError <|
        s!"{checkpointName}: expected {schema.optimizerStateCount} moment entries, got {entryCount}"
    -- Parameters usually share one step count, so each count replays the recurrence once.
    let mut powers : Std.HashMap Nat (Float × Float) := {}
    for _ in [0:entryCount] do
      let id ← Checkpoint.readNat64 checkpointName handle
      let step ← Checkpoint.readNat64 checkpointName handle
      let count ← Checkpoint.readNat64 checkpointName handle
      if (← replacementRef.get).contains id then
        throw <| IO.userError s!"{checkpointName}: duplicate parameter id {id}"
      let shape ← match schema.shapes[id]? with
        | some shape => pure shape
        | none => throw <| IO.userError s!"{checkpointName}: invalid parameter id {id}"
      unless schema.isRepresentative id do
        throw <| IO.userError s!"{checkpointName}: parameter {id} does not own optimizer state"
      if step = 0 then
        throw <| IO.userError s!"{checkpointName}: zero step counter for parameter {id}"
      if count != Shape.size shape then
        throw <| IO.userError <|
          s!"{checkpointName}: moment-size mismatch for parameter {id} " ++
            s!"(file={count}, expected={Shape.size shape})"
      let mBytes ← Checkpoint.readExact checkpointName handle (count * dtype.bytes)
      let vBytes ← Checkpoint.readExact checkpointName handle (count * dtype.bytes)
      let m ← LibTorch.Buffer.ofBytesIO mBytes dtype
      let v ← try
        LibTorch.Buffer.ofBytesIO vBytes dtype
      catch error =>
        releaseCUDABuffer m
        throw error
      unless powers.contains step do
        powers := powers.insert step
          (config.power config.beta1 step, config.power config.beta2 step)
      let (firstPower, secondPower) := powers.getD step (1, 1)
      replacementRef.modify (·.insert id { m, v, t := step, firstPower, secondPower })
    let trailing ← handle.read 1
    if trailing.size != 0 then
      throw <| IO.userError s!"{checkpointName}: trailing bytes after final state entry"
    pure (config, ← replacementRef.get, randomCounter)
  catch error =>
    releaseAdam (← replacementRef.get)
    throw error

/--
Restore CUDA Adam or AdamW moments after validating their optimizer and parameter metadata.

The current state stays live until the replacement has been read completely, and its buffers are
released after the swap. Duplicate ids, frozen, aliased or out-of-range slots, zero step counters,
trailing bytes and truncated payloads are rejected. A stored random counter is restored into
`rngCounter?`, which must then be given. A checkpoint without a counter loads, with a warning when
`rngCounter?` is given.
-/
def loadAdam
    (path : System.FilePath) (schema : Checkpoint.Schema)
    (configRef : IO.Ref (Option CUDAAdamConfig)) (stateRef : IO.Ref CUDAAdamState)
    (rngCounter? : Option (IO.Ref Nat) := none)
    (dtype : LibTorch.Dtype := .float32) : IO Unit := do
  unless schema.isWellFormed do
    throw <| IO.userError s!"{checkpointName}: malformed in-memory parameter schema"
  let expectedConfig ← configRef.get
  let (config, replacement, randomCounter) ←
    IO.FS.withFile path IO.FS.Mode.read fun handle =>
      readCheckpoint handle schema expectedConfig dtype
  if randomCounter.isSome && rngCounter?.isNone then
    releaseAdam replacement
    throw <| IO.userError s!"{checkpointName}: destination has no random-counter reference"
  match ← configRef.get with
  | some expected =>
      unless config.sameBits expected do
        releaseAdam replacement
        throw <| IO.userError <|
          s!"{checkpointName}: checkpoint belongs to a different optimizer configuration"
  | none => pure ()
  let previous ← stateRef.get
  stateRef.set replacement
  configRef.set (some config)
  match rngCounter?, randomCounter with
  | some counter, some value => counter.set value
  | some _, none =>
      IO.eprintln s!"{checkpointName}: checkpoint has no random counter; \
        stochastic training cannot be resumed exactly"
  | _, _ => pure ()
  releaseAdam previous

end Session
end TorchLean
