/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

-- shake: keep-all

public import NN.Floats.Float32
public import NN.Runtime.Autograd.TorchLean.Dual
public import NN.Runtime.Scalar
public import NN.Spec.Core.Complex
public import NN.Spec.Core.Scalar
public import NN.Tensor.API

import Lean.Data.Json
import Mathlib.Algebra.Order.Algebra

/-!
# API Core

Public convenience API core on top of TorchLean's spec + runtime entrypoints.

This module holds the small foundational surface used by the higher-level API modules:

- scalar-role separation (`Semantics` vs `Runtime`)
- CLI helpers
- dtype dispatch
- tensor constructor re-exports

Most end-user code should not import this file directly; it is re-exported through `NN.API.Public`
and via the top-level `NN` facade.

### PyTorch Mapping

`DType` here plays a role similar to `torch.dtype` selection in Python, but it also encodes a key
TorchLean distinction:
- some scalar types are executable (you can `#eval` / run demos), and
- some scalar types are proof-only (e.g. `ℝ` or a noncomputable float model).
-/

@[expose] public section


namespace NN
namespace API

export _root_.Spec (SpecScalar SpecTensor SpecContext)

export _root_.Runtime
  (RuntimeScalar RuntimeTensor RuntimeNeuralScalar RuntimeNeuralTensor)

namespace Semantics

/--
Public name for TorchLean's generic scalar semantics contract.

Read this as:

- "the model/loss is allowed to do its math over `α`"
- not "this backend is executable"

That second role belongs to `API.Runtime.Scalar`.
-/
abbrev Scalar (α : Type) := Context α

/-- Host-side scalar ReLU for task definitions and synthetic targets. -/
def relu {α : Type} [Zero α] [Max α] (x : α) : α :=
  Max.max x 0

end Semantics

namespace Runtime

/--
Runtime conversion from host `Float` constants into a TorchLean scalar type.

PyTorch analogy:

- in Python tutorials, users write float literals directly and tensors inherit a runtime dtype;
- in TorchLean, examples often start from host `Float` literals and inject them into the chosen
  runtime scalar backend with `Runtime.ofFloat`.
-/
class Scalar (α : Type) where
  /-- Convert a host `Float` literal into this runtime scalar backend. -/
  ofFloat : Float → α

/-- Generic host-float injection for TorchLean scalar backends. -/
def ofFloat {α : Type} [Scalar α] (x : Float) : α :=
  Scalar.ofFloat x

/-- `Float` is already the host literal type, so injection is identity. -/
instance : Scalar Float where
  ofFloat := id

/-- Inject host `Float` literals into the executable IEEE-754 binary32 backend. -/
instance : Scalar TorchLean.Floats.IEEE754.IEEE32Exec where
  ofFloat := TorchLean.Floats.IEEE754.IEEE32Exec.ofFloat

/--
Inject host literals into the dual-number backend used by the runtime autograd engine.

We interpret a literal as a primal value with zero tangent/adjoint component.
-/
instance {α : Type} [Scalar α] [Zero α] :
    Scalar (_root_.Runtime.Autograd.TorchLean.Dual α) where
  ofFloat x := _root_.Runtime.Autograd.TorchLean.Dual.ofPrimal (ofFloat x)

/-- Inject host literals into TorchLean's parametric complex scalar (imaginary part defaults to 0). -/
instance {α : Type} [Scalar α] [Zero α] : Scalar (TorchLean.Complex α) where
  ofFloat x := ⟨Runtime.ofFloat (α := α) x, 0⟩

/-- Allow numeric literals like `0.1` to elaborate to any TorchLean runtime scalar backend. -/
@[default_instance low]
instance {α : Type} [Scalar α] : OfScientific α where
  ofScientific m s e := ofFloat (Float.ofScientific m s e)

end Runtime

namespace CLI

/--
Strip at most one occurrence of a `--key` flag from an argument list.

Accepted forms:
- `--key value`
- `--key=value`
-/
def takeFlagValueOnce (args : List String) (key : String) :
    Except String (Option String × List String) :=
  let eqPrefix := s!"--{key}="
  let keyTok := s!"--{key}"
  let rec go :
      List String → Option String → List String → Except String (Option String × List String)
    | [], found, acc => .ok (found, acc.reverse)
    | a :: rest, found, acc =>
        if a == keyTok then
          match rest with
          | [] => .error s!"{keyTok}: expected a value"
          | v :: rest' =>
              if found.isSome then
                .error s!"{keyTok}: duplicate flag"
              else
                go rest' (some v) acc
        else if a.startsWith eqPrefix then
          let v := (a.drop eqPrefix.length).toString
          if found.isSome then
            .error s!"{keyTok}: duplicate flag"
          else
            go rest (some v) acc
        else
          go rest found (a :: acc)
  go args none []

/-- Return true when `args` contains `--key value` or `--key=value`. -/
def hasFlagValue (args : List String) (key : String) : Bool :=
  let eqPrefix := s!"--{key}="
  let keyTok := s!"--{key}"
  args.any (fun a => a == keyTok || a.startsWith eqPrefix)

/-- Remove a no-value boolean flag once, returning whether it appeared. -/
partial def takeBoolFlagOnce (args : List String) (key : String) :
    Except String (Bool × List String) := do
  let keyTok := s!"--{key}"
  let rec go : List String → Bool → List String → Except String (Bool × List String)
    | [], seen, acc => pure (seen, acc.reverse)
    | a :: rest, seen, acc =>
        if a == keyTok then
          if seen then
            throw s!"{keyTok}: duplicate flag"
          else
            go rest true acc
        else
          go rest seen (a :: acc)
  go args false []

/-- Drop the leading `--` separator commonly used with `lean --run`. -/
def dropDashDash (args : List String) : List String :=
  match args with
  | "--" :: rest => rest
  | xs => xs

/-- Fail if there are any unconsumed CLI arguments. -/
def requireNoArgs (args : List String) : Except String Unit :=
  if args.isEmpty then
    .ok ()
  else
    .error s!"unexpected arguments: {args}"

/-- Like `takeFlagValueOnce`, but parse the value as a `Nat`. -/
def takeNatFlagOnce (args : List String) (key : String) :
    Except String (Option Nat × List String) := do
  let (value?, rest) ← takeFlagValueOnce args key
  match value? with
  | none => pure (none, rest)
  | some value =>
      match value.toNat? with
      | some n => pure (some n, rest)
      | none => throw s!"--{key}: expected a natural number, got `{value}`"

/-- Parse a JSON-style floating-point literal. Accepts the same decimal forms as `Lean.Json`. -/
def parseFloatLit (s : String) : Option Float :=
  match Lean.Json.parse s with
  | Except.ok (.num n) => some n.toFloat
  | _ => none

/-- Like `takeFlagValueOnce`, but parse the value as a `Float`. -/
def takeFloatFlagOnce (args : List String) (key : String) :
    Except String (Option Float × List String) := do
  let (value?, rest) ← takeFlagValueOnce args key
  match value? with
  | none => pure (none, rest)
  | some value =>
      match parseFloatLit value with
      | some x => pure (some x, rest)
      | none => throw s!"--{key}: expected a JSON-style float literal, got `{value}`"

/-- Like `takeFlagValueOnce`, but return the value as a `System.FilePath`. -/
def takePathFlagOnce (args : List String) (key : String) :
    Except String (Option System.FilePath × List String) := do
  let (value?, rest) ← takeFlagValueOnce args key
  pure (value?.map (fun s => (s : System.FilePath)), rest)

/-- Common training flags that appear across many demos: `--epochs` and `--batch`. -/
structure EpochBatch where
  /-- Number of epochs to train for. -/
  epochs : Nat
  /-- Batch size. -/
  batch : Nat

/--
Parse optional `--epochs` and `--batch` flags.

Returns the parsed values (falling back to the provided defaults) and the remaining args.
-/
def takeEpochBatch (args : List String) (defaultEpochs defaultBatch : Nat) :
    Except String (EpochBatch × List String) := do
  let (epochs?, args) ← takeNatFlagOnce args "epochs"
  let (batch?, args) ← takeNatFlagOnce args "batch"
  pure ({ epochs := epochs?.getD defaultEpochs, batch := batch?.getD defaultBatch }, args)

/--
Parse a `--steps` flag, but also accept `--epochs` as a synonym.

This is a small ergonomics helper for tutorials that use a step-based loop,
while still letting users pass `--epochs` out of habit from typical PyTorch training code.

If both `--steps` and `--epochs` are provided, this errors.
-/
def takeStepsOrEpochs (args : List String) (default : Nat) : Except String (Nat × List String) := do
  let (steps?, args) ← takeNatFlagOnce args "steps"
  let (epochs?, args) ← takeNatFlagOnce args "epochs"
  match steps?, epochs? with
  | some _, some _ =>
      throw "--steps and --epochs are mutually exclusive (pick one)"
  | some steps, none => pure (steps, args)
  | none, some epochs => pure (epochs, args)
  | none, none => pure (default, args)

/-- Parse an optional `--seed` flag (defaults to the provided value). -/
def takeSeed (args : List String) (default : Nat := 0) :
    Except String (Nat × List String) := do
  let (seed?, rest) ← takeNatFlagOnce args "seed"
  pure (seed?.getD default, rest)

end CLI

/--
Configuration for the `float32` dtype option.

We support both:
- proof-only float32 semantics (`mode = .fp32`, noncomputable), and
- executable IEEE-754 float32 semantics (`mode = .ieee754Exec`).
-/
structure Float32Config where
  /-- Which float32 semantics backend to use. -/
  mode : TorchLean.Floats.Float32Mode := .ieee754Exec
  deriving Repr, DecidableEq

/--
Scalar type choice for demos and small executables.

This is a *runtime* selection mechanism used by example programs; the core library itself is
parametric in the scalar type `α`.

### PyTorch Mapping

This corresponds loosely to choosing `dtype=` in PyTorch, but with additional "proof-only"
variants:
- `.float` uses Lean's builtin `Float` (executable, but its IEEE-754 behavior is trusted),
- `.float32` uses TorchLean's float32 model (either proof-only or executable),
- `.complex` uses TorchLean's parametric complex scalar over a float32 backend,
- `.real` uses `ℝ` (proof-only; not executable).
-/
inductive DType where
  | float
  | float32 (cfg : Float32Config)
  | complex (cfg : Float32Config)
  | real
  deriving Repr, DecidableEq

namespace DType

/-- Whether this dtype can be used in an executable (`IO`/`#eval`) context. -/
def isExecutable : DType → Bool
  | .float => true
  | .real => false
  | .float32 { mode := .fp32 } => false
  | .float32 _ => true
  | .complex { mode := .fp32 } => false
  | .complex _ => true

/-- Log a human-readable description of the chosen dtype to stdout. -/
def log : DType → IO Unit
  | .float =>
      IO.println "[TorchLean] dtype: Float (Lean `Float`, trusted runtime semantics)"
  | .real =>
      IO.println "[TorchLean] dtype: ℝ (proof-only; not executable)"
  | .float32 cfg =>
      TorchLean.Floats.logFloat32Mode cfg.mode
  | .complex cfg => do
      IO.println "[TorchLean] dtype: Complex Float32 (TorchLean.Complex over Float32)"
      TorchLean.Floats.logFloat32Mode cfg.mode

namespace Internal

/-- Parse a `float32` mode selector string into a `Float32Mode`. -/
def parseFloat32Mode (v : String) : Except String TorchLean.Floats.Float32Mode := do
  if v == "fp32" then
    pure .fp32
  else if v == "ieee32" || v == "ieee754" || v == "ieee32exec" || v == "ieee754exec" then
    pure .ieee754Exec
  else
    throw s!"unknown float32 mode {v} (supported: fp32 | ieee754exec)"

/-- Parse a dtype selector string into a `DType`. -/
def parseDTypeValue (v : String) : Except String DType := do
  if v == "float" then
    pure .float
  else if v == "float32" || v == "f32" then
    pure (.float32 {})
  else if v == "complex" || v == "complex32" || v == "cfloat32" || v == "c32" then
    pure (.complex {})
  else if v == "real" || v == "reals" || v == "ℝ" then
    pure .real
  else if v == "fp32" || v == "ieee32" || v == "ieee754" || v == "ieee32exec" || v == "ieee754exec"
    then
    pure (.float32 { mode := (← parseFloat32Mode v) })
  else if v.startsWith "float32:" then
    let m ← parseFloat32Mode ((v.drop 8).toString)
    pure (.float32 { mode := m })
  else if v.startsWith "f32:" then
    let m ← parseFloat32Mode ((v.drop 4).toString)
    pure (.float32 { mode := m })
  else if v.startsWith "complex:" then
    let m ← parseFloat32Mode ((v.drop 8).toString)
    pure (.complex { mode := m })
  else if v.startsWith "c32:" then
    let m ← parseFloat32Mode ((v.drop 4).toString)
    pure (.complex { mode := m })
  else
    throw s!"unknown --dtype {v} (supported: float | real | complex | fp32 | ieee754exec | f32:<mode> | c32:<mode>)"

end Internal

open Internal

/--
Parse and remove dtype flags from CLI arguments, using `default` when no dtype flags are provided.

This is the same parsing logic as `parseAndStrip`, but it lets higher-level runners choose a
different default dtype depending on context (e.g. CUDA eager requires `Float`).
-/
def parseAndStripWithDefault (args : List String) (default : DType) :
    Except String (DType × List String) := do
  let (dtypeV?, args1) ← CLI.takeFlagValueOnce args "dtype"
  let (modeV?, args2) ← CLI.takeFlagValueOnce args1 "float32-mode"
  match dtypeV?, modeV? with
  | some _, some _ =>
      throw "--dtype and --float32-mode are mutually exclusive (pick one)"
  | none, none =>
      pure (default, args2)
  | some dv, none =>
      pure (← parseDTypeValue dv, args2)
  | none, some mv =>
      let m ← parseFloat32Mode mv
      pure (.float32 { mode := m }, args2)

/--
Parse and remove dtype flags from CLI arguments.

Supported flags:
- `--dtype <value>` or `--dtype=<value>`
- `--float32-mode <mode>` or `--float32-mode=<mode>`

The two flags are mutually exclusive.
-/
def parseAndStrip (args : List String) : Except String (DType × List String) := do
  parseAndStripWithDefault args (.float32 {})

/--
Run `k` under the scalar type selected by `dt`.

If `dt` is proof-only, this returns an error rather than trying to execute noncomputable code.
-/
def withRuntime
    (dt : DType)
    (k : ∀ {α : Type}, [API.Semantics.Scalar α] → [DecidableEq Spec.Shape] → [ToString α] →
      [API.Runtime.Scalar α] → IO Unit) :
    IO (Except String Unit) := do
  match dt with
  | .float =>
      k (α := Float)
      pure (.ok ())
  | .real =>
      pure (.error
        "dtype=real is proof-only (noncomputable); use it in theorems, not in executables")
  | .complex { mode := .fp32 } =>
      pure (.error
        "dtype=complex:fp32 is proof-only (noncomputable); use it in theorems/verification proofs")
  | .complex { mode := .ieee754Exec } =>
      k (α := TorchLean.Complex (TorchLean.Floats.F32 .ieee754Exec))
      pure (.ok ())
  | .float32 { mode := .fp32 } =>
      pure (.error
        "float32-mode=fp32 is proof-only (noncomputable); use it in theorems/verification proofs")
  | .float32 { mode := .ieee754Exec } =>
      k (α := TorchLean.Floats.F32 .ieee754Exec)
      pure (.ok ())

/--
Run `k` under the scalar type selected by `dt`, passing an explicit cast function `Float → α`.

This is a convenient shape for small demos that want to construct tensors from `Float` lists.
-/
def withExec
    (dt : DType)
    (k : ∀ {α : Type}, [API.Semantics.Scalar α] → [DecidableEq Spec.Shape] → [ToString α] → (Float →
      α) → IO Unit) :
    IO (Except String Unit) :=
  withRuntime dt (fun {α} _ _ _ _ => k (α := α) (API.Runtime.ofFloat (α := α)))

end DType

namespace Tensor

export _root_.Spec.Tensor (scalar dim)

export _root_.NN.Tensor
  (shapeOfDims numelDims tensor1d oneHot oneHotNat tensor2d tensorND tensorDynND
   fillND zerosND onesND
   tensorF321d tensorF322d
   print)

export _root_.NN.Tensor.Shape (Vec Mat CHW OIHW NCHW)

end Tensor

end API
end NN
