/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.IRExec
public import NN.Runtime.Autograd.IRExec.Correctness.Common
public import NN.Runtime.Autograd.IRExec.Correctness.Ops

/-!
# Correctness

Correctness for the IR → executable SSA graph bridge.

This file collects per-op forward-correctness lemmas for
`Runtime.Autograd.IRExec.lowerToForwardGraph`. It is split out from `NN.Runtime.Autograd.IRExec`
so that routine builds of TorchLean's runtime do not have to import proof internals one file at
a time.

Reusable helper lemmas live in `NN.Runtime.Autograd.IRExec.Correctness.Common`.
Operator-step lemmas live under `...Correctness.Ops`, grouped by role:
- `...Correctness.Ops.Concat`
- `...Correctness.Ops.LinearAlgebra`
- `...Correctness.Ops.Loss`
- `...Correctness.Ops.Normalization`
- `...Correctness.Ops.Pooling`
- `...Correctness.Ops.Random`

The recursive end-to-end theorem lives in `...Correctness.SemanticEquivalence` and is not
imported here.

## Main definitions

- `NoRawLog`: side condition for the current theorem fragment, until raw-log positivity is carried
  through the end-to-end lowering proof.
- Per-operator lowering-step lemmas from `...Correctness.Ops`.

## Implementation notes

- Separating reusable infrastructure (`Common`) from op-specific steps keeps large correctness
  proofs maintainable.
- Import `...Correctness.SemanticEquivalence` explicitly when you want the recursive end-to-end
  theorem.

## References

- [PyTorch operator semantics (for op parity checks)](https://pytorch.org/docs/stable/index.html)

## Tags

correctness, ir, runtime, lowering, semantic equivalence
-/

@[expose] public section


namespace Runtime
namespace Autograd
namespace IRExec

open Spec TorchLean
open TorchLean TorchLean.Tensor
open Proofs.Autograd.Algebra
open NN.IR
open Internal

end IRExec
end Autograd
end Runtime
