/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Core.Tensor

/-!
# Spec modules (`NNModuleSpec`) and shape-safe composition

The `NN/Spec/Layers/*` files define reference layer specs: usually a parameter record plus a pure
`forward` (and sometimes explicit gradient formulas).

This file packages those specs into a small, uniform *module* interface:

`NNModuleSpec α inShape outShape`

which is just a `forward` function plus lightweight metadata (`kind`, `export_func`)
used by tooling (export/extraction) and by the runtime/IR pipeline described in
the TorchLean paper (`arXiv:2602.22631`).

We keep that metadata separate from the semantics: changing `kind`/`toPyTorch` should never change
what `forward` means.

`SpecChain` is a dependent composition operator that enforces intermediate shape agreement at
compile time, so you can build pipelines without runtime shape casts.

Mental model (PyTorch analogy):

- `NNModuleSpec` is like a compact, pure `nn.Module` with just a `forward`.
- `SpecChain` is like `nn.Sequential`, but shape-safe by construction.

Diagram:

```
  x : Tensor α s
      |
      v
  [m1 : s -> t]  forward
      |
      v
  [m2 : t -> u]  forward
      |
      v
  y : Tensor α u
```

The point of all this is practical: you want shape mistakes to show up as type errors, not as
runtime exceptions.
-/

@[expose] public section


namespace ModSpec

open Spec
open Tensor

/-- Export-related metadata carried alongside a spec module.

This is intentionally informal (mostly strings). It is **not** part of the math we prove about a
 model. We use it for demos and for "roughly equivalent PyTorch" pretty-printing.
-/
structure ExportFunctions where
  /-- A PyTorch-style rendering for docs/demos (metadata only). -/
  toPyTorch : String
  /-- Extra integer metadata used by some exporters (interpretation depends on `kind`). -/
  dimensions : Nat × Nat

/-- A pure module: from `inShape` to `outShape` without runtime state. -/
structure NNModuleSpec (α : Type) (inShape outShape : Shape) where
  /-- forward. -/
  forward : Tensor α inShape → Tensor α outShape
  /-- Tag used by export/extraction tooling (metadata only). -/
  kind    : String
  /-- Export-related metadata (metadata only). -/
  export_func  : ExportFunctions

/--
Dependent chain of spec modules ensuring intermediate shapes match at compile-time.
Use this for shape-safe composition without runtime casting.
-/
inductive SpecChain (α : Type) : Shape → Shape → Type where
| single {s t} (m : NNModuleSpec α s t) : SpecChain α s t
| comp   {s t u} (a : SpecChain α s t) (b : SpecChain α t u) : SpecChain α s u

namespace SpecChain

/-- Forward evaluation over a `SpecChain` by structural composition. -/
def forward {α : Type} {s t : Shape} (c : SpecChain α s t) : Tensor α s → Tensor α t :=
  match c with
  | single m    => fun x => m.forward x
  | comp a b    => fun x =>
      let y := forward a x
      forward b y

/-- Right-associative composition helper.

This is the ergonomic "append a module to a chain" operator used at call sites:

```
net : SpecChain α s t
net |>.compose_right m2 |>.compose_right m3
```
-/
def composeRight {α : Type} {s t u : Shape}
  (a : SpecChain α s t) (b : NNModuleSpec α t u) : SpecChain α s u :=
  SpecChain.comp a (SpecChain.single b)

/-- Extract the list of `kind` tags from a chain (left-to-right). -/
def extractOps {α : Type} {s t : Shape} : SpecChain α s t → List String
| single m    => [m.kind]
| comp a b    => extractOps a ++ extractOps b

/-- Extract `(kind, toPyTorch)` pairs from a chain (left-to-right). -/
def extractLayerInfo {α : Type} {s t : Shape} : SpecChain α s t → List (String × String)
| .single m =>
  let export_func := m.export_func
  [(m.kind, export_func.toPyTorch)]
| .comp a b => extractLayerInfo a ++ extractLayerInfo b

/-- Lift a module to apply independently over a leading sequence dimension.

This is a "map over time" helper for sequence models:

- input has shape `(seqLen, elemIn)`,
- output has shape `(seqLen, elemOut)`,
- and we apply the same module at each timestep.

In PyTorch terms: `torch.vmap`-style mapping, or a common pattern like:

```
ys = [m(x_t) for t in range(T)]
```

We keep this as a single helper so call sites stay small and the intent is obvious.
-/
def mapEach
  {α : Type} [Context α]
  {seqLen : Nat} {elemIn elemOut : Shape}
  (m : NNModuleSpec α elemIn elemOut) :
  NNModuleSpec α (Shape.dim seqLen elemIn) (Shape.dim seqLen elemOut) :=
{
  forward := fun t =>
    match t with
    | Tensor.dim f =>
      Tensor.dim fun i => m.forward (f i),
  kind := "map_each_" ++ m.kind,  -- tag to indicate sequence lifting (metadata only)
  export_func := m.export_func    -- reuse underlying export functions
}

end SpecChain

end ModSpec
