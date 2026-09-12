/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Model.Layers.Core

/-!
# Sequential GraphSpec syntax

This module defines the extensible primitive interface and the shape-indexed `Chain` language.
A chain records its ordered parameter ABI at the type level, and sequential composition
concatenates those parameter lists.
-/

@[expose] public section

namespace NN
namespace GraphSpec

open Spec TorchLean

/--
A primitive operation in the sequential GraphSpec language.

The pure `specFwd` interpretation and executable `program` share the same parameter, input, and
output shape indices. The optional layer conversion supports deterministic initialization and
conversion to `Runtime.Autograd.Model.Layers.Seq`.
-/
structure Primitive (ps : List Shape) (σ τ : Shape) where
  /-- Short name used mainly for debugging and error messages. -/
  name : String
  /-- Pure reference semantics of the primitive. -/
  specFwd :
    ∀ {α : Type 0}, [TorchLean.Storage α] → [Context α] →
      TorchLean.TensorPack α ps → TorchLean.Tensor α σ → TorchLean.Tensor α τ
  /--
  Executable TorchLean forward program, with parameters followed by the data input.
  -/
  program :
    ∀ {α : Type 0}, [TorchLean.Storage α] → [Context α] →
      Runtime.Autograd.Model.Program α (ps ++ [σ]) τ
  /--
  Optional conversion to a TorchLean layer, indexed by its occurrence in the surrounding chain.
  -/
  toLayerM? :
    Option (Nat → { l : Runtime.Autograd.Model.Layers.Layer σ τ // l.stateShapes = ps }) := none
  /-- Whether this primitive advances the layer-occurrence counter. -/
  countsAsLayer : Bool := false

/--
`Chain ps σ τ` is a sequential model from shape `σ` to shape `τ` whose parameters have shapes
`ps`, in order. Composition concatenates parameter ABIs; use `NN.GraphSpec.DAG` when explicit
sharing or multi-input nodes are required.
-/
inductive Chain : List Shape → Shape → Shape → Type 2 where
  /-- Identity chain: passes the input through unchanged and requires no parameters. -/
  | id (s : Shape) : Chain [] s s
  /-- Sequential composition. Parameter lists concatenate. -/
  | seq {ps₁ ps₂ : List Shape} {σ τ υ : Shape} :
      Chain ps₁ σ τ → Chain ps₂ τ υ → Chain (ps₁ ++ ps₂) σ υ
  /-- Embed a single primitive node in a chain. -/
  | prim {ps : List Shape} {σ τ : Shape} :
      Primitive ps σ τ → Chain ps σ τ

infixr:80 " >>> " => Chain.seq

end GraphSpec
end NN
