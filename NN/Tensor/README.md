# `NN.Tensor`

This folder is the user facing tensor API for TorchLean. It provides small constructors and helpers
for examples, tests, and small models while keeping Lean's shape and dtype checks.

For public use, prefer the curated library import or the tensor entrypoint:

```lean
import NN.Library
-- or, if you only want this subsystem:
import NN.Entrypoint.Tensor
```

`NN.Tensor.API` is the implementation leaf behind the entrypoint.

The key invariant is that semantics and proofs stay in the spec layer (`NN/Spec/*`), while this
layer stays focused on usability:

- DTypes are Lean types: you write `Tensor Float s`, `Tensor ℚ s`, and other scalar backends used by the project.
- Shapes are (usually) in the type: constructors like `tensor1d` remember `xs.length` in the result type.
- If you see `Tensor Float _`, the `_` asks Lean to infer the shape from the right hand side.
- When you truly need dynamic shapes, use `tensorND` (runtime dims + runtime length check) or
  `DynTensor` (store the shape as data instead of in the type).
- For constants, `tensorND!` and `tensorF!` trade a bit of macro expansion for cleaner literal code.
- `tensor!` accepts nested bracket syntax and flattens in row-major order, which is handy for
  handwritten examples.

Files:

- `../Entrypoint/Tensor.lean`: the stable tensor subsystem import.
- `API.lean`: the implementation leaf. Includes shape aliases, `tensor1d`/`tensor2d`, dynamic N-D
  constructors, padding friendly 2D constructors for ragged data, and a small printing API that
  refuses to print proof level scalar backends like `ℝ`.
