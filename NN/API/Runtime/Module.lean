/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Core
public import NN.API.Rand
public import NN.API.TorchLean.ParamIO
public import NN.API.TorchLean.Schedulers
public import NN.GraphSpec.Models.TorchLean
public import NN.Runtime.Autograd.TorchLean
public import NN.Runtime.Autograd.Train.Dataset
public import NN.Runtime.RL
public import NN.Spec.RL.MDP
public import NN.Spec.RL.MarkovMDP
public import NN.Spec.RL.FiniteStochasticMDP
public import NN.API.Runtime.Core
public import NN.API.Runtime.Layers
public import NN.API.Runtime.Autograd

import Mathlib.Algebra.Order.Algebra
import NN.Spec.Autograd.AutogradSpec

@[expose] public section

namespace NN
namespace API
namespace TorchLean

/-!
# Runtime Modules and Executable Entrypoints

Scalar-module instantiation, runtime initialization, CLI execution config parsing, and executable
`main` helpers.
-/

namespace Models

/-!
## Model constructors (re-export)

This namespace re-exports ready-made model constructors (MLP/CNN/ResNet18/etc.) for runnable
examples and integration checks.

For compositional building blocks, prefer `API.TorchLean.NN` and `API.TorchLean.Layers`.
-/

export _root_.NN.GraphSpec.Models.TorchLean
  (mlp autoencoder cnn2 softmaxRegression mlpClassifier transformerBlock
   resnet18Model resnet18Program resnet18InitParams)
end Models

namespace Module

/-!
### ScalarModule API (Session-Like Interface)

The `ScalarModule` interface is the TorchLean equivalent of "instantiate a model, then do forward,
backward, and optimizer steps" in an imperative runtime.

This section mostly re-exports `Runtime.Autograd.TorchLean.Module.*` and adds small CLI-friendly
helpers (`Module.withModule` / `Module.withModuleRuntime`) that select dtype/backend from flags.
-/

export _root_.Runtime.Autograd.TorchLean.Module (ScalarModuleDef ScalarModule)
namespace RuntimeInit
export _root_.Runtime.Autograd.TorchLean.Module.RuntimeInit
  (FloatInit Plan xavierUniformForShape kaimingUniformForShape xavierLinearWeight
   kaimingLinearWeight)
end RuntimeInit
export _root_.Runtime.Autograd.TorchLean.Module.ScalarModule
  (create forward backward step initOptim stepWith params setParams trainSGD trainWith meanLoss)
export _root_.Runtime.Autograd.TorchLean.Module.ScalarModuleDef
  (instantiate instantiateFloatWithRuntimePlan instantiateFloatWithRuntimeInit)

/--
Instantiate a `ScalarModuleDef` under explicit Torch options (`backend`, `fastKernels`, `useGpu`,
etc.).

This is the most direct "runtime" entrypoint (used by the CPU/CUDA example binaries), since it
threads the same options record all the way down to the eager tape / CUDA tape selection.

The shorter `instantiate` entrypoint selects the backend and then uses default runtime options.
-/
def instantiateWithOptions
    {α : Type} [API.Semantics.Scalar α] [DecidableEq Spec.Shape]
    [_root_.Runtime.Autograd.Torch.Internal.CudaBridge.TensorConv α]
    {paramShapes inputShapes : List Spec.Shape}
    (defn : ScalarModuleDef paramShapes inputShapes)
    (cast : Float → α) (opts : Options) :
    IO (ScalarModule α paramShapes inputShapes) :=
  _root_.Runtime.Autograd.TorchLean.Module.ScalarModuleDef.instantiateWith
    (α := α) (paramShapes := paramShapes) (inputShapes := inputShapes) defn cast opts

/--
Instantiate a Float module with runtime-side parameter initializers.

This is the public runtime-initialization entrypoint. The initializer plan is indexed by the same
`paramShapes` list as the module, so Lean checks that every parameter has exactly one initializer.
In CUDA mode, supported initializers allocate device buffers directly instead of first constructing
every parameter as a large nested Lean tensor.
-/
def instantiateFloatWithRuntimePlanOptions
    {paramShapes inputShapes : List Spec.Shape}
    (defn : ScalarModuleDef paramShapes inputShapes)
    (opts : Options)
    (plan : RuntimeInit.Plan paramShapes) :
    IO (ScalarModule Float paramShapes inputShapes) :=
  _root_.Runtime.Autograd.TorchLean.Module.ScalarModuleDef.instantiateFloatWithRuntimePlan
    (paramShapes := paramShapes) (inputShapes := inputShapes) defn opts plan

/--
List-based wrapper for checkpoint/JSON boundaries.

If the caller has a statically known parameter list, prefer
`instantiateFloatWithRuntimePlanOptions`; this wrapper checks the list length before applying it.
-/
def instantiateFloatWithRuntimeInitOptions
    {paramShapes inputShapes : List Spec.Shape}
    (defn : ScalarModuleDef paramShapes inputShapes)
    (opts : Options)
    (inits : List RuntimeInit.FloatInit) :
    IO (ScalarModule Float paramShapes inputShapes) :=
  _root_.Runtime.Autograd.TorchLean.Module.ScalarModuleDef.instantiateFloatWithRuntimeInit
    (paramShapes := paramShapes) (inputShapes := inputShapes) defn opts inits

/--
Execution configuration parsed from CLI flags.

Supported flags (parsed by `ExecConfig.parseAndStrip`):
- `--dtype ...` / `--float32-mode ...` (see `NN.API.DType`)
- `--backend eager|compiled`
- `--cpu` / `--cuda` (eager device selection)
- `--fast-kernels` (eager-only performance hooks, no effect on compiled backend)
- `--fast-gpu-matmul-precision fp32|fp64` (fast-kernel CUDA matmul precision)
-/
structure ExecConfig where
  /-- Scalar dtype selection. -/
  dtype : DType := .float
  /-- Execution backend selection. -/
  backend : Backend := .eager
  /--
  Eager execution device selector.

  When `true` and `backend = .eager`, TorchLean uses the CUDA tape backend.
  -/
  useGpu : Bool := false
  /-- Enable runtime-only eager fast kernels (tight-loop implementations for a few hot ops). -/
  fastKernels : Bool := false
  /-- GPU precision for fast-kernel matmul over Lean `Float` tensors. -/
  fastGpuMatmulPrecision : GpuMatmulPrecision := .fp32
  deriving Repr, DecidableEq

namespace ExecConfig

/-- Parse a backend selector string into a runtime `Backend`. -/
def parseBackend (v : String) : Except String Backend := do
  if v == "eager" then
    pure .eager
  else if v == "compiled" then
    pure .compiled
  else
    throw s!"unknown --backend {v} (supported: eager | compiled)"

/-- Parse a fast-kernel CUDA matmul precision selector. -/
def parseFastGpuMatmulPrecision (v : String) : Except String GpuMatmulPrecision := do
  if v == "fp32" || v == "float32" then
    pure .fp32
  else if v == "fp64" || v == "float64" || v == "double" then
    pure .fp64
  else
    throw s!"unknown --fast-gpu-matmul-precision {v} (supported: fp32 | fp64)"

/--
Parse CLI flags handled by `ExecConfig` and return `(cfg, rest)`.

Consumed flags:
- `--backend eager|compiled` (at most once),
- `--cpu` / `--cuda` (boolean flags; last one wins; removed from `rest`),
- `--fast-kernels` (boolean flag; removed from `rest`).
- `--fast-gpu-matmul-precision fp32|fp64` (at most once).

All dtype/Float32 selection flags are delegated to `DType.parseAndStripWithDefault`.

Default dtype policy:
- If the user does not specify `--dtype` / `--float32-mode` and `--cuda` is present, default to
  `dtype=float` (CUDA eager supports `Float` upload/download).
- Otherwise default to `dtype=float32` (executable IEEE-754 float32 semantics).

When `--cuda` is selected, also enable the CUDA fast path (`--fast-kernels`) by default.
-/
def parseAndStripWithDefaultDType (args : List String) (defaultDType : DType) :
    Except String (ExecConfig × List String) := do
  let (dtype, args1) ← DType.parseAndStripWithDefault args defaultDType
  let (backend, args2) ←
    CLI.takeParsedFlagDefault args1 "backend" "eager" parseBackend
  let (fastGpuMatmulPrecision, args3) ←
    CLI.takeParsedFlagDefault args2 "fast-gpu-matmul-precision" "fp32" parseFastGpuMatmulPrecision
  let rec go (useGpu fastKernels : Bool) (acc : List String) :
      List String → (Bool × Bool × List String)
    | [] => (useGpu, fastKernels, acc.reverse)
    | a :: as =>
        if a == "--cuda" then
          go true fastKernels acc as
        else if a == "--cpu" then
          go false fastKernels acc as
        else if a == "--fast-kernels" then
          go useGpu true acc as
        else
          go useGpu fastKernels (a :: acc) as
  let (useGpu, fastKernels, rest) := go false false [] args3
  pure ({
    dtype := dtype,
    backend := backend,
    useGpu := useGpu,
    fastKernels := fastKernels || useGpu,
    fastGpuMatmulPrecision := fastGpuMatmulPrecision
  }, rest)

/-- Convert a parsed CLI execution config to runtime `Options`. -/
def toOptions (cfg : ExecConfig) (seed : Nat := 0) : Options :=
  { backend := cfg.backend
    seed := seed
    useGpu := cfg.useGpu
    fastKernels := cfg.fastKernels
    fastGpuMatmulPrecision := cfg.fastGpuMatmulPrecision }

/-- Parse CLI flags with the standard TorchLean default dtype policy. -/
def parseAndStrip (args : List String) : Except String (ExecConfig × List String) := do
  let defaultDType : DType := if args.contains "--cuda" then .float else .float32 {}
  parseAndStripWithDefaultDType args defaultDType

/-- Log the chosen execution config to stdout for reproducible runs. -/
def log (cfg : ExecConfig) : IO Unit := do
  DType.log cfg.dtype
  IO.println s!"[TorchLean] backend: {reprStr cfg.backend}"
  IO.println s!"[TorchLean] device: {if cfg.useGpu then "cuda" else "cpu"}"
  IO.println s!"[TorchLean] fastKernels: {cfg.fastKernels}"
  IO.println s!"[TorchLean] fastGpuMatmulPrecision: {reprStr cfg.fastGpuMatmulPrecision}"

end ExecConfig

/--
Parse runtime flags (`--dtype`, `--backend`, `--cpu|--cuda`, `--fast-kernels`,
`--fast-gpu-matmul-precision`) and choose an executable scalar `α`, then call `k` with:
- `cast : Float → α` for building inputs from literals
- `opts : Options` selecting the backend/kernel mode
- `rest : List String` containing the remaining CLI arguments

This is useful for scripts that need to build a dataset/loader (and maybe determine shapes/batch
sizes) before instantiating a concrete `ScalarModuleDef`.
-/
def withRuntime
    (args : List String)
    (k :
      ∀ {α : Type}, [API.Semantics.Scalar α] → [DecidableEq Spec.Shape] → [ToString α] →
        [API.Runtime.Scalar α] →
        (cast : Float → α) → (opts : Options) → (rest : List String) → IO Unit) :
    IO Unit := do
  let (cfg, rest) ←
    match ExecConfig.parseAndStrip args with
    | .ok v => pure v
    | .error msg => throw <| IO.userError msg
  ExecConfig.log cfg
  let opts : Options := ExecConfig.toOptions cfg
  match (← DType.withRuntime cfg.dtype (fun {α} _ _ _ _ => do
        k (α := α) (API.Runtime.ofFloat (α := α)) opts rest
      )) with
  | .ok () => pure ()
  | .error msg => throw <| IO.userError msg

/--
Instantiate a `ScalarModuleDef` under CLI runtime flags (`--dtype`, `--backend`, `--cpu|--cuda`,
  `--fast-kernels`, `--fast-gpu-matmul-precision`), then call a continuation.

This provides the cast function `Float → α` so call sites can build inputs from float literals.
-/
def withModule
    {paramShapes inputShapes : List Spec.Shape}
    (defn : ScalarModuleDef paramShapes inputShapes)
    (args : List String)
    (k :
      ∀ {α : Type}, [API.Semantics.Scalar α] → [DecidableEq Spec.Shape] → [ToString α] →
        (cast : Float → α) → ScalarModule α paramShapes inputShapes → (rest : List String) →
        IO Unit) :
    IO Unit := do
  let (cfg, rest) ←
    match ExecConfig.parseAndStrip args with
    | .ok v => pure v
    | .error msg => throw <| IO.userError msg
  ExecConfig.log cfg
  let opts : Options := ExecConfig.toOptions cfg
  match cfg.dtype with
  | .float =>
      -- Keep the Float branch explicit. If this path is hidden behind the scalar-polymorphic
      -- `DType.withExec` continuation, Lean can elaborate module construction with the generic
      -- fallback CUDA converter instead of the real Float upload bridge. That still compiles, but
      -- a CUDA training step later fails when it tries to upload a Float tensor.
      let m ← _root_.Runtime.Autograd.TorchLean.Module.ScalarModuleDef.instantiateWith
        (α := Float) (paramShapes := paramShapes) (inputShapes := inputShapes) defn id opts
      k (α := Float) id m rest
  | _ =>
      if cfg.useGpu then
        throw <| IO.userError "torch: eager CUDA module execution currently requires --dtype float"
      match (← DType.withExec cfg.dtype (fun {α} _ _ _ cast => do
            let m ← _root_.Runtime.Autograd.TorchLean.Module.ScalarModuleDef.instantiateWith
              (α := α) (paramShapes := paramShapes) (inputShapes := inputShapes)
              defn cast opts
            k (α := α) cast m rest
          )) with
      | .ok () => pure ()
      | .error msg => throw <| IO.userError msg

/--
Like `withModule`, but also provides an `API.Runtime.Scalar α` instance (for numeric literals).
-/
def withModuleRuntime
    {paramShapes inputShapes : List Spec.Shape}
    (defn : ScalarModuleDef paramShapes inputShapes)
    (args : List String)
    (k :
      ∀ {α : Type}, [API.Semantics.Scalar α] → [DecidableEq Spec.Shape] → [ToString α] →
        [API.Runtime.Scalar α] →
        ScalarModule α paramShapes inputShapes → (rest : List String) → IO Unit) :
    IO Unit := do
  let (cfg, rest) ←
    match ExecConfig.parseAndStrip args with
    | .ok v => pure v
    | .error msg => throw <| IO.userError msg
  ExecConfig.log cfg
  let opts : Options := ExecConfig.toOptions cfg
  match cfg.dtype with
  | .float =>
      -- Same reason as `withModule`: CUDA module construction should see `α = Float` directly, so
      -- the Float-specific `TensorConv` instance is selected before the runner is handed to user
      -- code.
      let m ← _root_.Runtime.Autograd.TorchLean.Module.ScalarModuleDef.instantiateWith
        (α := Float) (paramShapes := paramShapes) (inputShapes := inputShapes) defn id opts
      k (α := Float) m rest
  | _ =>
      if cfg.useGpu then
        throw <| IO.userError "torch: eager CUDA module execution currently requires --dtype float"
      match (← DType.withRuntime cfg.dtype (fun {α} _ _ _ _ => do
            let m ← _root_.Runtime.Autograd.TorchLean.Module.ScalarModuleDef.instantiateWith
              (α := α) (paramShapes := paramShapes) (inputShapes := inputShapes)
              defn (API.Runtime.ofFloat (α := α)) opts
            k (α := α) m rest
          )) with
      | .ok () => pure ()
      | .error msg => throw <| IO.userError msg

/-!
## Executable `main` Helpers

TorchLean has a lot of pure, type-indexed code (models live in `Type 2`), but runnable scripts still
want a "single entrypoint" that handles:
- parsing `--seed`,
- selecting an executable dtype/backend/device from flags,
- seeding TorchLean's global RNG stream (`API.rand`) so `nn.freshSeed`/`nn.withModel` are deterministic.
-/

/-- Options for `TorchLean.Module.run` (banner printing, trailing ok, etc.). -/
structure RunOptions where
  /-- Optional banner to print before executing the program. -/
  banner? : Option (Options → String) := none
  /-- Flush stdout after printing the banner (if present). -/
  flush : Bool := true
  /-- Print `"{exeName}: ok"` on success. -/
  printOk : Bool := false
deriving Inhabited

namespace RunOptions

/-- Print the configured executable banner, if one was supplied. -/
def printBanner (o : RunOptions) (opts : Options) : IO Unit := do
  match o.banner? with
  | none => pure ()
  | some banner =>
      IO.println (banner opts)
      if o.flush then
        (← IO.getStdout).flush

end RunOptions

/-- How `run` should select the scalar backend for an executable. -/
inductive RunAction where
  /--
  Allow `--dtype` selection (the continuation must work for all executable scalar backends).
  -/
  | any
      (k :
        ∀ {α : Type}, [API.Semantics.Scalar α] → [DecidableEq Spec.Shape] → [ToString α] →
          [API.Runtime.Scalar α] →
          (cast : Float → α) → (opts : Options) → (rest : List String) → IO Unit)
  /--
  Force the scalar backend to builtin `Float` (useful for Float-only IO bridges / CUDA upload paths).
  -/
  | float (k : (opts : Options) → (rest : List String) → IO Unit)

/-- Generic help text for executables built on `TorchLean.Module.run`. -/
def runUsage (exeName : String) : String :=
  String.intercalate "\n"
    [ s!"Usage: {exeName} [runtime flags] [command flags]"
    , ""
    , "Runtime flags:"
    , "  --cpu | --cuda"
    , "  --dtype float|ieee754exec"
    , "  --backend eager|compiled"
    , "  --seed N"
    , "  --fast-kernels"
    , "  --fast-gpu-matmul-precision fp32|tf32"
    , ""
    , "Use the example documentation or `lake exe torchlean --help` for command-specific flags."
    ]

/--
CLI entrypoint helper for executable `main` functions.

This parses:
- `--seed N` (via `API.CLI.takeSeed`), and
- runtime execution flags (`--dtype`, `--float32-mode`, `--backend`, `--cpu|--cuda`,
  `--fast-kernels`, `--fast-gpu-matmul-precision`),
then executes the chosen `RunAction`.

It also seeds TorchLean's global RNG stream (`API.rand`) so code that draws init seeds via
`API.nn.freshSeed`/`API.nn.withModel` is deterministic by default, matching the PyTorch pattern of
calling `torch.manual_seed` once in `main`.
-/
def run
    (exeName : String)
    (args : List String)
    (action : RunAction)
    (runOpts : RunOptions := {}) :
    IO UInt32 := do
  let args := API.CLI.dropDashDash args
  if args.contains "--help" || args.contains "-h" then
    IO.println (runUsage exeName)
    return 0
  let (seed, args) ←
    match API.CLI.takeSeed args 0 with
    | .ok v => pure v
    | .error msg => throw <| IO.userError s!"{exeName}: {msg}"

  _root_.NN.API.rand.manualSeed seed

  let printOk : IO Unit := do
    if runOpts.printOk then
      IO.println s!"{exeName}: ok"

  match action with
  | .any k =>
      withRuntime args (fun {α} _ _ _ _ cast opts rest => do
        -- Keep seed in the same `Options` record used by the Torch eager/compiled sessions so scripts
        -- can still follow the familiar pattern `nn.manualSeed opts.seed` when desired.
        let opts : Options := { opts with seed := seed }
        runOpts.printBanner opts
        k (α := α) cast opts rest
        printOk
      )
      pure 0
  | .float k =>
      let (cfg, rest) ←
        match ExecConfig.parseAndStripWithDefaultDType args .float with
        | .ok v => pure v
        | .error msg => throw <| IO.userError msg
      if cfg.dtype != .float then
        throw <| IO.userError s!"{exeName}: this program only supports `--dtype float`"
      ExecConfig.log cfg
      let opts : Options := ExecConfig.toOptions cfg seed
      runOpts.printBanner opts
      k opts rest
      printOk
      pure 0

end Module

end TorchLean
end API
end NN
