/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Verify.RuntimeApprox.Roundoff.Coverage
public import TorchLean.Verify.RuntimeApprox.Roundoff.Rules.NormalizedRows -- shake: keep
public import TorchLean.Tensor.Op.Table.Rule

/-!
# Roundoff coverage of the layer and batch-statistics normalizations of the operator table

`#roundoff_coverage` reads the routes of each registered operator from its arm of
`TorchLean.Op.rule?`. The theorems are declared in the namespace `TorchLean.Coverage.Layers`,
which every module under `TorchLean/Verify/Coverage/Layers/` shares.

`#roundoff_propagation` declares the absolute propagation theorems of
`TorchLean.Normalization.normalize`, `TorchLean.Tensor.abs` and `TorchLean.Tensor.detach`, each an
error bound under real bounds on its inputs.

This is the one coverage module that imports the rules of
`TorchLean.Verify.RuntimeApprox.Roundoff.Rules.NormalizedRows`: no other module analyzes a program
that reads `Normalization.normalizedRows`, and each `roundoff` session normalizes every registered
rule.
-/

public section

namespace TorchLean.Coverage.Layers

-- Both layers broadcast tensors of shape `[rows]` at the shape `[rows, cols].eraseIdx 1`.
-- Unfolding `List.eraseIdx` in implicit arguments lets the rules read them at either shape.
attribute [local implicit_reducible] List.eraseIdx

-- `Op.batchNorm` is analyzed here, beside `Op.layerNorm`, because the tangents and gradients of
-- both read `Normalization.normalizedRows`, whose propagation theorems a namespace may declare only
-- once.
#roundoff_coverage
  layers := Op.layerNorm, Op.batchNorm

#roundoff_propagation
  absolute := Normalization.normalize, Tensor.abs, Tensor.detach

end TorchLean.Coverage.Layers
