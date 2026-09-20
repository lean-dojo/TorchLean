/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.MLTheory.CROWN.Graph.Engine
public import NN.Tensor
/-!
# Shared CROWN Test Helpers
One helper, shared by the two suites that drive the flattened CROWN graph engine:
`NN/Tests/MLTheory/CROWNSoundnessGuardrails.lean` and
`NN/Tests/Runtime/Floats/RankPolymorphicLayerOps.lean`.
It lived in both of them, once as `pointBox` and once as `pointFlatBox`, with the same body. Two
names for one function is the version of duplication that is hardest to notice, because grepping for
either name finds exactly one definition and looks reassuring.
The generic bound-array matchers those two suites also shared are in `NN/Tests/Utils.lean` instead,
which imports nothing from the library. Only this helper needs `FlatBox`, and dragging the CROWN
import closure into the lightweight assertions module would put it in front of every runtime suite.
-/

@[expose] public section

namespace NN.Tests.MLTheory.Utils

open Spec TorchLean
open NN.MLTheory.CROWN

/--
The degenerate box `[x, x]` around a concrete tensor, flattened.

Guardrail tests feed a point rather than an interval because they are checking whether the engine
certifies a node at all, not how tight the certificate is.
-/
def pointFlatBox {s : Shape} (value : Tensor Float s) : FlatBox Float :=
  FlatBox.ofTensor (Tensor.flattenSpec value)

end NN.Tests.MLTheory.Utils
