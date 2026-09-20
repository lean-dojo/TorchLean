/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.MLTheory.CROWN.Graph.Engine.CROWN.Run

/-!
# Upper affine bounds

The one-sided API projects the upper forms of the two-sided CROWN engine. Intermediate lower
bounds must be retained: multiplication by a negative weight or subtraction exchanges the two
bounds, even when the caller ultimately requests only an upper bound.
-/

public section

namespace NN.MLTheory.CROWN.Graph

open TorchLean

/-- Run CROWN and retain the upper affine form at each node.

Both sides are propagated internally so negative coefficients select the correct parent bound.
Rounded backends use the directed coefficient propagation shared with backward CROWN.
-/
def runAffine {α : Type} [Storage α] [Context α] [BoundOps α]
    (g : NN.IR.Graph) (ps : ParamStore α) (ctx : AffineCtx)
    (ibp : Array (Option (FlatBox α))) : Array (Option (FlatAffine α)) :=
  (runCROWN g ps ctx ibp).map fun entry =>
    entry.map fun bounds =>
      { inDim := bounds.inDim, outDim := bounds.outDim, aff := bounds.hiAff }

end NN.MLTheory.CROWN.Graph
