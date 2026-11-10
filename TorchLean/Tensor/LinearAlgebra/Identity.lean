/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Tensor.Datatype

/-!
# The identity matrix

`Tensor.identity n` is the `n × n` identity matrix over any scalar with `0` and `1`. Matrix
products are contractions: `Tensor.matmul` multiplies two matrices, and `Tensor.contract` with
`Shape.ContractValid.matVec` or `Shape.ContractValid.vecMat` multiplies a matrix and a vector.
-/

@[expose] public section


namespace TorchLean.Tensor

/-- The `n × n` identity matrix over any scalar with `0` and `1`. Its diagonal test compares
natural-number indices, so it needs no equality test on scalars. -/
def identity {α : Type} [TorchLean.Storage α] [Zero α] [One α]
    (n : Nat) : Tensor α [n, n] :=
  Tensor.Rep.ofFn fun coordinate =>
    if coordinate.1.val == coordinate.2.1.val then 1 else 0

end TorchLean.Tensor
