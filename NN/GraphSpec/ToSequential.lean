/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.GraphSpec.Chain.Syntax
public import NN.Runtime.Autograd.Model.Layers.Seq

/-!
# GraphSpec to Sequential Models

This module converts a GraphSpec `Chain` to a TorchLean `NN.Seq` when every primitive supplies a
`Layer` representation. The conversion packages deterministic parameter initialization and the
parameter ABI expected by the sequential trainer.

This is a partial adapter, not the general GraphSpec execution path. `Chain.toProgram` interprets
every supported GraphSpec primitive through the operation-polymorphic `TorchLean.Program`
interface, including primitives with no `Layer`. By contrast, `ToSequential.toSeq` returns
`Except String` and rejects a chain at the first primitive without a sequential-layer view.

The adapter threads a primitive occurrence index through the chain. Deterministic initializers use
that index to derive stable per-layer seeds; the index is not part of GraphSpec's mathematical
semantics.

Related modules:
- `NN.GraphSpec.Core` for the core DSL and its general `Chain.toProgram` translation.
- `NN/GraphSpec/README.md` for the relationship between GraphSpec and the runtime model API.

For `g : Chain ps σ τ`:

- `ToSequential.toSeq g` constructs a sequential model when every primitive has a `Layer`;
- `Chain.toProgram g` constructs the general operation-polymorphic program.
-/

@[expose] public section


namespace NN
namespace GraphSpec
namespace ToSequential

open Spec TorchLean
open TorchLean.Tensor

namespace Internal

/--
Lower a chain to a TorchLean `Seq`, threading a “layer occurrence index”.

The index is incremented for primitives with `countsAsLayer = true`.
-/
def toSeqFrom
    {ps : List Shape} {σ τ : Shape}
    (g : Chain ps σ τ) (i : Nat) :
    Except String (Runtime.Autograd.Model.Layers.Seq σ τ × Nat) :=
  match g with
  | .id s => do
      -- Identity becomes the identity sequential model.
      return (Runtime.Autograd.Model.Layers.Seq.id s, i)
  | .seq g₁ g₂ => do
      -- Sequential composition becomes sequential composition.
      let (s₁, i') ← toSeqFrom (ps := _) (σ := _) (τ := _) g₁ i
      let (s₂, i'') ← toSeqFrom (ps := _) (σ := _) (τ := _) g₂ i'
      return (Runtime.Autograd.Model.Layers.Seq.comp s₁ s₂, i'')
  | .prim p => do
      -- A primitive can only be lowered if it provides a `Layer`.
      match p.toLayerM? with
      | none =>
          .error <|
            s!"graphspec.toSeq: primitive `{p.name}` has no Seq lowering (missing toLayerM?); "
              ++
            "use `Chain.toProgram` if you only need execution"
      | some mk =>
          -- Thread a deterministic occurrence index for initialization.
          let i' := if p.countsAsLayer then i + 1 else i
          let ⟨l, _hps⟩ := mk i
          return (Runtime.Autograd.Model.Layers.Seq.fromLayer l, i')

end Internal

/--
Try to lower a GraphSpec chain into a `Runtime.Autograd.Model.Layers.Seq`.

Use this when you specifically want the `Seq` wrapper for training ergonomics. If all you need
is an executable program, prefer `Chain.toProgram`: it is the more general path and does not
require every primitive to have a `Layer` view.
-/
def toSeq
    {ps : List Shape} {σ τ : Shape}
    (g : Chain ps σ τ) :
    Except String (Runtime.Autograd.Model.Layers.Seq σ τ) :=
  match Internal.toSeqFrom (ps := ps) (σ := σ) (τ := τ) g 0 with
  | .ok (s, _i) => .ok s
  | .error e => .error e

end ToSequential
end GraphSpec
end NN
