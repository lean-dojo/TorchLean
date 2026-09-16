/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Neural.Builders -- shake: keep

/-!
# Sequential Model Literals

TorchLean sequential models are shape-indexed (`Sequential σ τ`), so a plain `List` of layers
cannot describe a model the way PyTorch's `nn.Sequential([...])` does: every element would need
the same type. This file provides two list-shaped spellings that expand to ordinary composition
through `TorchLean.nn.compose`:

- `nn.Sequential![a, b, c]` runs each entry as a monadic layer builder and returns the composed
  model in that monad. This is the spelling used with the seeded builders of `NN.API.Seeded`.
- `nn.compose![a, b, c]` composes already-built layers or sequential models without any monad.

Both are scoped syntax in the `TorchLean` namespace, so they become available after
`open TorchLean`. The `!` suffix keeps `nn.Sequential` itself usable as a type name in expressions
such as `nn.Sequential σ τ`.
-/

@[expose] public section

namespace TorchLean

/--
`nn.Sequential![a, b, c]` builds a sequential model from monadic layer builders.

Each entry is run in order in the ambient monad and the results are composed left to right with
`TorchLean.nn.compose`, so the output shape of every entry must match the input shape of
the next. A single entry is converted to a `Sequential` model with
`TorchLean.nn.AsSequential.asSequential`. Entries may be layers or sequential models.

Example:
```lean
-- As close to `torch.nn.Sequential([...])` as a shape-indexed model can get: the entries are
-- builders, they run in order, and the shapes have to line up or the model does not compile.
def model : nn.Builder (nn.Sequential [2] [1]) :=
  nn.Sequential![
    nn.linear 2 8,
    nn.relu,
    nn.linear 8 1
  ]
```
-/
scoped syntax (name := nnSequentialBangLit) "nn.Sequential!" "[" term,+ "]" : term

macro_rules (kind := nnSequentialBangLit)
  | `(nn.Sequential![$a:term]) =>
      `(do
        let a ← ($a)
        pure (TorchLean.nn.AsSequential.asSequential a))
  | `(nn.Sequential![$a:term, $b:term]) =>
      `(do
        let a ← ($a)
        let b ← ($b)
        pure (TorchLean.nn.compose a b))
  | `(nn.Sequential![$a:term, $b:term, $rest:term,*]) =>
      `(do
        let a ← ($a)
        let bc ← (nn.Sequential![$b, $rest,*])
        pure (TorchLean.nn.compose a bc))

/--
`nn.compose![a, b, c]` composes already-built layers and sequential models without a monad.

Entries are composed left to right with `TorchLean.nn.compose`, so the output shape of
every entry must match the input shape of the next. A single entry is returned unchanged.

Example:
```lean
-- The same list spelling for models that are already built, with no monad in the way.
def stack (first : nn.Sequential [4] [8]) (second : nn.Sequential [8] [2]) :
    nn.Sequential [4] [2] :=
  nn.compose![first, second]
```
-/
scoped syntax (name := nnComposeBangLit) "nn.compose!" "[" term,+ "]" : term

macro_rules (kind := nnComposeBangLit)
  | `(nn.compose![$a:term]) => `($a)
  | `(nn.compose![$a:term, $b:term]) => `(TorchLean.nn.compose $a $b)
  | `(nn.compose![$a:term, $b:term, $rest:term,*]) =>
      `(TorchLean.nn.compose $a (nn.compose![$b, $rest,*]))

end TorchLean
