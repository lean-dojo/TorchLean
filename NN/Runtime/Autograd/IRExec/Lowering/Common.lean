/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.IR.Payload
public import NN.Proofs.Autograd.Tape.Algebra.Soundness
public import NN.Runtime.Autograd.IRExec.Core

/-!
# Shared Checked-Lowering Context

Common dependent context and result types used by the operation-family lowering modules.

Every lowering branch validates parents and shapes once, while the graph is being lowered, and
then builds a pure closure whose result type is the node's declared output shape. The closures
apply typed specification operators directly to typed parent values; they never call the dynamic
IR evaluator and never unwrap a runtime result, so the lowered graph contains no `panic!`.
-/

@[expose] public section

namespace Runtime
namespace Autograd
namespace IRExec

open Spec TorchLean
open TorchLean TorchLean.Tensor
open Proofs.Autograd.Algebra
-- Typed context indices come from `NN.Proofs.Autograd.Tape.Util.Idx`, the one place
-- `Idx` and `getIdx` are defined.
open Proofs (Idx getIdx)
open NN.IR

namespace Internal

/-- Data shared by checked lowerers for one IR node. -/
structure NodeLoweringContext (α : Type) [TorchLean.Storage α] [Context α] (Γ : List Shape) where
  /-- The IR graph being lowered. -/
  graph : NN.IR.Graph
  /-- External parameters keyed by node id. -/
  payload : Payload α
  /-- Position of the node in the graph. -/
  index : Nat
  /-- The IR node being lowered. -/
  node : NN.IR.Node
  /-- Build a typed index for a parent id at an expected shape. -/
  parentIdx : (pid : Nat) → (s : Shape) → Except String (Idx Γ s)

/-- The checked executable node produced for a lowering context. -/
abbrev NodeLoweringResult {α : Type} [TorchLean.Storage α] [Context α] {Γ : List Shape}
    (ctx : NodeLoweringContext α Γ) : Type :=
  Except String (ForwardNode α Γ ctx.node.outShape)

end Internal
end IRExec
end Autograd
end Runtime
