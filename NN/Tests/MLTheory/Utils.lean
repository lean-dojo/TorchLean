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

Point boxes for the flattened CROWN graph engine. Generic bound-array assertions live in
`NN/Tests/Utils.lean`; keeping the CROWN-specific construction here lets other runtime suites use
those assertions without importing the verifier.
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
