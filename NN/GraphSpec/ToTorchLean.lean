/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.GraphSpec.Core

/-!
# GraphSpec → TorchLean.NN.Seq (sequential lowering)

This provides a *structural lowering* from GraphSpec graphs into TorchLean sequential models.

## What is this lowering for?

GraphSpec gives us two things already:

- a **pure** denotational semantics (`Interp.spec`) for proofs / reference reasoning, and
- a compiler to an **executable** TorchLean program (`Compile.torchProgram`) for running/training.

So why do we also lower to `TorchLean.NN.Seq`?

`TorchLean.NN.Seq` is a small “training ergonomics” wrapper around a sequence of `LayerDef`s:

- it packages a *parameter shape list* (`paramShapes`) and deterministic parameter initialization,
- it provides a sequential forward program,
- and it plugs easily into existing training-loop code that expects a
  sequential layer stack.

GraphSpec remains the source of truth for the model structure. `Seq` is a secondary view for the
subset of models that are layer stacks.

## Partial compilation: `Except String`

Important design decision:

- GraphSpec itself is a **graph language**, not “a training framework”.
- Not every GraphSpec primitive should be forced to be a `LayerDef`.

So the lowering returns `Except String`: it succeeds for graphs whose primitives provide
`Primitive.toLayerDefM?`, and fails otherwise.

## Deterministic initialization (occurrence index)

We thread a `Nat` counter to support deterministic, occurrence-indexed initialization. This gives
the lowering a stable RNG boundary without making global randomness part of the graph syntax.

For example, the default `Primitive.linear` uses:
- `seedW = 2*i`, `seedB = 2*i + 1`.

See also:
- `NN.GraphSpec.Core` for the core sequential DSL and its `Compile.torchProgram` compiler.
- `NN.GraphSpec/README.md` for the high-level “GraphSpec vs TorchLean” motivation.

### Example (informal)

If `g : Graph ps σ τ` is built from primitives that provide `toLayerDefM?`, then:

- `ToTorchLean.toSeq g : Except String (TorchLean.NN.Seq σ τ)` gives a sequential model with
  deterministic initialization, and
- `Compile.torchProgram g : TorchLean.Program α (ps ++ [σ]) τ` gives the general executable program
  view (works even when there is no `Seq` lowering).
-/

@[expose] public section


namespace NN
namespace GraphSpec
namespace ToTorchLean

open Spec
open NN.Tensor

/-- Convenience constructor for an error result in the `Except String` lowering pipeline. -/
def err {α : Sort _} (msg : String) : Except String α := .error msg

/--
Public alias for TorchLean's sequential model type used as the lowering target.

This keeps GraphSpec examples from exposing the lower-level
`Runtime.Autograd.TorchLean.NN.Seq` path directly. It is definitionally the same type as
`NN.API.nn.Sequential`, but avoids importing the full API facade back into GraphSpec.
-/
abbrev Sequential (σ τ : Shape) : Type 2 :=
  _root_.Runtime.Autograd.TorchLean.NN.Seq σ τ

/--
Lower a graph to a TorchLean `Seq`, threading a “layer occurrence index”.

The index is incremented for primitives with `countsAsLayer = true`.
-/
def toSeqAux
    {ps : List Shape} {σ τ : Shape}
    (g : Graph ps σ τ) (i : Nat) :
    Except String (Sequential σ τ × Nat) := do
  match g with
  | .id s =>
      -- Identity graph becomes identity sequential model.
      return (_root_.Runtime.Autograd.TorchLean.NN.Seq.id s, i)
  | .seq g₁ g₂ =>
      -- Sequential composition becomes sequential composition.
      let (s₁, i') ← toSeqAux (ps := _) (σ := _) (τ := _) g₁ i
      let (s₂, i'') ← toSeqAux (ps := _) (σ := _) (τ := _) g₂ i'
      return (_root_.Runtime.Autograd.TorchLean.NN.Seq.comp s₁ s₂, i'')
  | .prim p =>
      -- A primitive can only be lowered if it provides a `LayerDef`.
      match p.toLayerDefM? with
      | none =>
          err <|
            s!"graphspec.toSeq: primitive `{p.name}` has no Seq lowering (missing toLayerDefM?); "
              ++
            "use `Compile.torchProgram` if you only need execution"
      | some mk =>
          -- Thread a deterministic occurrence index for initialization.
          let i' := if p.countsAsLayer then i + 1 else i
          let ⟨l, _hps⟩ := mk i
          return (_root_.Runtime.Autograd.TorchLean.NN.seq1 l, i')

/--
Try to lower a sequential GraphSpec graph into a `TorchLean.NN.Seq`.

Use this when you specifically want the `Seq` wrapper for training ergonomics. If all you need
is an executable program, prefer `Compile.torchProgram`: it is the more general path and does not
require every primitive to have a `LayerDef` view.
-/
def toSeq
    {ps : List Shape} {σ τ : Shape}
    (g : Graph ps σ τ) :
    Except String (Sequential σ τ) :=
  match toSeqAux (ps := ps) (σ := σ) (τ := τ) g 0 with
  | .ok (s, _i) => .ok s
  | .error e => .error e

end ToTorchLean
end GraphSpec
end NN
