/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Compiled.IRExec
public import NN.Runtime.Autograd.Compiled.IRExec.Correctness.Common
public import NN.Runtime.Autograd.Compiled.IRExec.Correctness.Ops

/-!
# Correctness

Correctness for the IR → executable SSA graph bridge.

This file collects per-op forward-correctness lemmas for `Runtime.Autograd.Compiled.execGraphOfIR`.
It is intentionally split out from `NN.Runtime.Autograd.Compiled.IRExec` so that routine builds of
TorchLean's runtime do not have to import proof internals one file at a time.

Reusable helper lemmas live in `NN.Runtime.Autograd.Compiled.IRExec.Correctness.Common`.
Operator-step lemmas live under `...Correctness.Ops`, grouped by role:
- `...Correctness.Ops.LinearAlgebra`
- `...Correctness.Ops.Loss`
- `...Correctness.Ops.Normalization`
- `...Correctness.Ops.Pooling`
- `...Correctness.Ops.Random`

The recursive end-to-end theorem lives in `...Correctness.SemanticEquivalence` and is intentionally not
imported here.

## Main definitions

- `NoMSELoss`: side condition used by the semantic equivalence theorem to state its exact fragment.
- Per-operator compiler-step lemmas from `...Correctness.Ops`.

## Implementation notes

- Separating "reusable infrastructure" (`Common`) from "op-specific steps" keeps the
  way to keep large correctness proofs maintainable.
- Import `...Correctness.SemanticEquivalence` explicitly when you want the recursive end-to-end theorem.

## References

- [PyTorch operator semantics (for op parity checks)](https://pytorch.org/docs/stable/index.html)

## Tags

correctness, ir, runtime, compilation, semantic equivalence
-/

@[expose] public section


namespace Runtime
namespace Autograd
namespace Compiled

open Spec
open Tensor
open Proofs.Autograd.Algebra
open NN.IR
open IRExec

end Compiled
end Autograd
end Runtime
