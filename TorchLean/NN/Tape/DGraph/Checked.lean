/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.TypedGraph.WithData.Checked
public import TorchLean.NN.Tape.DGraph

/-!
# Checked differentiation of composed, proved graphs

`DGraph` stores the primitive derivative proofs and preserves them under composition. Its graph is
the algebraic tape graph at `α := ℝ` and `Δ := Unit`, so selecting an output (`toTypedGraph`) gives
the typed graph that the runtime differentiates, with no conversion. `vjpChecked_eq_fderiv`
uses the stored certificates, so a composed graph needs no separate backpropagation proof. The
theorem assumes a successful pullback: runtime domain validation may reject an input.

The semantics are exact reals, so floating-point rounding and the nested-dual implementation of
higher derivatives lie outside these results.
-/

@[expose] public section

namespace TorchLean.Tape.DGraph

/-- Select an output of a proved graph for the checked runtime, preserving all node data. -/
def toTypedGraph {Γ ss : List Shape} {τ : Shape} (graph : DGraph Γ ss)
    (output : Idx (Γ ++ ss) τ) : TypedGraph ℝ Γ τ where
  nodeShapes := ss
  data := graph.g.toData
  output := output

/-- If the checked pullback of a selected output succeeds, its input gradients are the adjoint of
Mathlib's derivative of that output's forward map, applied to the seed. The certificate `hg`
stored in the graph supplies the differentiability hypothesis at every input. -/
theorem vjpChecked_eq_fderiv {Γ ss : List Shape} {τ : Shape} (graph : DGraph Γ ss)
    (output : Idx (Γ ++ ss) τ) (inputs : TensorPack ℝ Γ) (seed : Tensor ℝ τ)
    (result : TensorPack ℝ Γ × Tensor ℝ τ)
    (checked : (graph.toTypedGraph output).vjpChecked inputs () seed = .ok result) :
    TensorPack.toVec result.1 =
      (fderiv ℝ (fun x => Tensor.toVec ((graph.toTypedGraph output).forward
        (TensorPack.ofVec x) ())) (TensorPack.toVec inputs)).adjoint (Tensor.toVec seed) := by
  apply TypedGraph.WithData.vjpChecked_eq_fderiv
    (graph.toTypedGraph output) graph.g rfl inputs () seed _ result checked
  exact RealGraph.FDerivCorrect.at graph.hg (TensorPack.toVec inputs)

end TorchLean.Tape.DGraph
