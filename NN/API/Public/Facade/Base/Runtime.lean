/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Public.NN
public import NN.API.Public.TensorPack
public import NN.API.Public.Seeded
public import NN.API.Public.Autograd
public import NN.API.Data
public import NN.API.Data.Transforms
public import NN.API.Runtime
public import NN.API.Models
public import NN.API.Public.NN.Transformer
public import NN.API.RL
public import NN.API.Rand
public import NN.API.Samples.Bands
public import NN.API.Text.Bpe
public import NN.MLTheory.CROWN.Flatbox
public import NN.MLTheory.CROWN.Graph
public import NN.Verification.TorchLean.Compile
public import NN.API.Public.Facade.Base.Root

/-!
# TorchLean Runtime Names

Dtype, backend, device, and runtime-selection helpers.
-/

@[expose] public section

namespace TorchLean

namespace Runtime

@[inherit_doc NN.API.DType]
abbrev DType := NN.API.DType

namespace DType

@[inherit_doc NN.API.DType.float]
abbrev float : DType :=
  NN.API.DType.float

@[inherit_doc NN.API.DType.float32]
abbrev float32 (cfg : NN.API.Float32Config := {}) : DType :=
  NN.API.DType.float32 cfg

@[inherit_doc NN.API.DType.complex]
abbrev complex (cfg : NN.API.Float32Config := {}) : DType :=
  NN.API.DType.complex cfg

@[inherit_doc NN.API.DType.real]
abbrev real : DType :=
  NN.API.DType.real

end DType

/-!
Scalar classes used after a runtime backend has been selected.

Most TorchLean code should let `Trainer` provide these instances implicitly. A few demos use
`Runtime.withOptions` / `Runtime.withOptionsNoCast` to run the same code under a CLI-selected scalar
backend. These classes stay under `TorchLean.Runtime` so the root namespace can stay focused on
models, data, and training.
-/

/--
Executable scalar support for TorchLean examples.

Use this when an example chooses a scalar backend or writes code that is polymorphic over the
selected runtime.
-/
abbrev Scalar := NN.API.Runtime.Scalar

/--
Scalar math used by TorchLean model and loss definitions.

Runtime-selected examples and verification demos use this when they need the same model code to run
under more than one scalar backend.
-/
abbrev SemanticScalar := NN.API.Semantics.Scalar

/-- Scalar operations needed to build and manipulate shape-indexed TorchLean tensors. -/
abbrev TensorScalar := _root_.Context

export NN.API.TorchLean (Ops RefTy Program)

@[inherit_doc _root_.Runtime.Autograd.Torch.Backend]
abbrev Backend := _root_.Runtime.Autograd.Torch.Backend

namespace Backend

@[inherit_doc _root_.Runtime.Autograd.Torch.Backend.eager]
abbrev eager : Backend :=
  _root_.Runtime.Autograd.Torch.Backend.eager

@[inherit_doc _root_.Runtime.Autograd.Torch.Backend.compiled]
abbrev compiled : Backend :=
  _root_.Runtime.Autograd.Torch.Backend.compiled

end Backend

@[inherit_doc _root_.Runtime.Autograd.Torch.Device]
abbrev Device := _root_.Runtime.Autograd.Torch.Device

namespace Device

@[inherit_doc _root_.Runtime.Autograd.Torch.Device.cpu]
abbrev cpu : Device :=
  _root_.Runtime.Autograd.Torch.Device.cpu

@[inherit_doc _root_.Runtime.Autograd.Torch.Device.cuda]
abbrev cuda : Device :=
  _root_.Runtime.Autograd.Torch.Device.cuda

end Device

@[inherit_doc NN.API.Runtime.ofFloat]
def ofFloat {α : Type} [TorchLean.Runtime.Scalar α] (x : Float) : α :=
  NN.API.Runtime.ofFloat x

/--
Parse the usual TorchLean runtime flags and run a `Float` callback.

Examples should use this or `ModelZoo.runFloat` instead of calling `TorchLean.Module.run` directly;
that lower-level dispatcher is what backs this wrapper.
-/
def runFloat
    (exeName : String) (args : List String)
    (banner : Options → String)
    (k : (opts : Options) → (rest : List String) → IO Unit)
    (printOk : Bool := true) : IO UInt32 :=
  NN.API.Common.runFloat exeName args banner k printOk

/--
Parse the standard TorchLean runtime flags and return the resulting `Options`.

This is the non-polymorphic sibling of `Runtime.withOptions`: examples that always run at `Float`
can still parse `--cpu`, `--cuda`, `--backend`, and `--dtype` without exposing a polymorphic
callback.
-/
def parseArgs (args : List String) (defaultDType : DType := .float) :
    Except String (Options × List String) := do
  let (cfg, rest) ←
    NN.API.TorchLean.Module.ExecConfig.parseAndStripWithDefaultDType args defaultDType
  pure (NN.API.TorchLean.Module.ExecConfig.toOptions cfg, rest)

/--
Run a scalar-polymorphic example under the dtype/backend selected by the usual TorchLean CLI flags.

The callback receives:
- `cast`, for turning Float literals into the selected executable scalar type,
- `rest`, the arguments left after runtime flags are stripped.

Prefer `Trainer` for model training. Use this for demos that really need the selected scalar type in
their own code, for example small autograd/runtime walkthroughs.
-/
def withSelected
    (args : List String)
    (k :
      ∀ {α : Type}, [SemanticScalar α] → [DecidableEq Shape] → [ToString α] →
        [Scalar α] → (cast : Float → α) → (rest : List String) → IO Unit) :
    IO Unit :=
  NN.API.TorchLean.Module.withRuntime args
    (fun {α} _ _ _ _ cast _opts rest => k (α := α) cast rest)

/--
Run a scalar-polymorphic example under the selected runtime when the callback does not need a
Float-cast function.

Prefer `Runtime.withOptionsNoCast` when the callback also needs parsed runtime options. This
exists for the rare case where only the selected scalar/backend instances and remaining CLI
arguments matter.
-/
def withSelectedNoCast
    (args : List String)
    (k :
      ∀ {α : Type}, [SemanticScalar α] → [DecidableEq Shape] → [ToString α] →
        [Scalar α] → (rest : List String) → IO Unit) :
    IO Unit :=
  NN.API.TorchLean.Module.withRuntime args
    (fun {α} _ _ _ _ _cast _opts rest => k (α := α) rest)

/--
Run an example under the selected runtime and pass through the parsed runtime options.

Use this when an example needs to inspect `--backend`, `--cuda`, or similar flags after TorchLean
has selected the scalar backend.
-/
def withOptions
    (args : List String)
    (k :
      ∀ {α : Type}, [SemanticScalar α] → [DecidableEq Shape] → [ToString α] →
        [Scalar α] → (cast : Float → α) → Options → (rest : List String) → IO Unit) :
    IO Unit :=
  NN.API.TorchLean.Module.withRuntime args
    (fun {α} _ _ _ _ cast opts rest => k (α := α) cast opts rest)

/--
Run an example under the selected runtime and pass through runtime options when the callback does
not need an explicit Float-cast function.
-/
def withOptionsNoCast
    (args : List String)
    (k :
      ∀ {α : Type}, [SemanticScalar α] → [DecidableEq Shape] → [ToString α] →
        [Scalar α] → Options → (rest : List String) → IO Unit) :
    IO Unit :=
  withOptions args (fun {α} _ _ _ _ _cast opts rest => k (α := α) opts rest)

/--
Run a verification or demo command under the selected runtime dtype.

This is the banner-printing sibling of `withSelected`; it matches the convention used by
`lake exe verify -- ...` commands.
-/
def runWithDType
    (title : String) (args : List String)
    (k : ∀ {α : Type}, [SemanticScalar α] → [DecidableEq Shape] → [ToString α] → [Scalar α] →
      IO Unit) :
    IO Unit :=
  NN.API.Common.runWithRuntimeDType title args
    (fun {α} _ _ _ _ => k (α := α))

end Runtime


end TorchLean
