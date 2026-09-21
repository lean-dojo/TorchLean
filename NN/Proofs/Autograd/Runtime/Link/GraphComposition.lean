/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Proofs.Autograd.Runtime.Link.Checked
public import NN.Proofs.Autograd.Tape.Nodes.GraphComposition

/-!
# Checked differentiation of composed, proved graphs

`DGraph` already stores the primitive derivative proofs and preserves them under composition.
Selecting an output now gives the same typed graph that the runtime differentiates. The VJP
theorem uses the stored certificates, so a new composition needs no separate backpropagation
proof. Runtime domain validation is still allowed to reject an input.

These are exact-real semantics. They do not certify floating-point rounding or the nested-dual
implementation of higher derivatives.
-/

@[expose] public section

namespace Proofs.Autograd.DGraph

open Spec TorchLean

/-- Select an output of a proved graph for the checked runtime, preserving all node data. -/
def toTypedGraph {Γ ss : List Shape} {τ : Shape} (graph : DGraph Γ ss)
    (output : Idx (Γ ++ ss) τ) : Runtime.Autograd.Torch.TypedGraph ℝ Γ τ where
  nodeShapes := ss
  data := graph.g.toAlgebra.toData
  output := output

/-- A successful pullback of a composed graph inherits its stored primitive derivative proofs. -/
theorem vjpChecked_adjoint_fderiv {Γ ss : List Shape} {τ : Shape} (graph : DGraph Γ ss)
    (output : Idx (Γ ++ ss) τ) (inputs : TensorPack ℝ Γ) (seed : Tensor ℝ τ)
    (result : TensorPack ℝ Γ × Tensor ℝ τ)
    (checked : (graph.toTypedGraph output).vjpChecked inputs () seed = .ok result) :
    flattenCtx result.1 =
      (fderiv ℝ (fun x => tensorToVec ((graph.toTypedGraph output).forward
        (unflattenCtx x))) (flattenCtx inputs)).adjoint (tensorToVec seed) := by
  apply Runtime.Autograd.Torch.TypedGraphWithData.vjpChecked_adjoint_fderiv
    (graph.toTypedGraph output) graph.g.toAlgebra rfl inputs () seed _ result checked
  change GraphFDerivCorrectAt (graph.g.toAlgebra.toReal ()) (flattenCtx inputs)
  rw [Algebra.Graph.toAlgebra_toReal]
  exact graphFDerivCorrectAtOfCorrect graph.hg (flattenCtx inputs)

end Proofs.Autograd.DGraph
