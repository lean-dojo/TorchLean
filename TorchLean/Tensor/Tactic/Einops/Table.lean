/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/
module

public import TorchLean.Tensor.Einops.Check.ParseShape
public import TorchLean.Tensor.Native.Construction
public import TorchLean.Tensor.Einops.Lowering.Einsum
public import TorchLean.Tensor.Einops.Lowering.Pack
public import TorchLean.Tensor.Einops.Lowering.Reduce
public import TorchLean.Tensor.Einops.Lowering.Reduce.View
public import TorchLean.Tensor.Einops.Lowering.TransformFusion

/-!
# The lowering table of `einops` and `einops?`

Each entry names a lowering, the number of its arguments, the positions of its runtime tensor
inputs, its `einops?` family, its denotation and its correctness theorem. `einops?` reads the
table to find, classify and report applications. `einops` reads it to recognize and rewrite them.
-/

public meta section

namespace TorchLean.Einops

open Lean

/-- A lowering that `einops` and `einops?` recognize, or the `parse_shape` metadata check. -/
structure Lowering where
  /-- The lowering constant. -/
  head : Name
  /-- The number of arguments of a complete application, implicit ones included. -/
  arity : Nat
  /-- The positions of the runtime tensor inputs among those arguments. -/
  tensorArguments : Array Nat
  /-- The family that `einops?` reports, absent outside the reported families. -/
  family? : Option String := none
  /-- The semantics that `correctness` identifies the lowering with. -/
  denotation : Name := .anonymous
  /-- Further semantic constants whose presence in a target shows the family. -/
  stages : Array Name := #[]
  /-- The correctness theorem of the lowering, anonymous for the metadata check. -/
  correctness : Name := .anonymous
  /-- Correctness theorems for one final checked operation, keyed by its report label. -/
  operationCorrectness : List (String × Name) := []

namespace Lowering

/-- Whether an expression applies the lowering to exactly `arity` arguments. -/
def isApp (entry : Lowering) (expression : Expr) : Bool :=
  expression.isAppOfArity entry.head entry.arity

/-- A declaration name as reports print it, relative to `TorchLean.Einops`. -/
def reportName (name : Name) : String :=
  (name.replacePrefix `TorchLean.Einops .anonymous).toString

/-- The fused shape transformation, one source read per output entry. -/
def transformTensorFused : Lowering where
  head := ``Lowering.transformFused
  arity := 9
  tensorArguments := #[8]
  correctness := ``Lowering.transformFused_correct
  operationCorrectness := [
    ("rearrange", ``Lowering.transformFused_eq_rearrange),
    ("repeat", ``Lowering.transformFused_eq_expand)]

/-- The rearrange lowering. -/
def rearrangeTensor : Lowering where
  head := ``Lowering.rearrange
  arity := 5
  tensorArguments := #[4]
  family? := some "rearrange"
  denotation := ``Semantics.denoteRearrange
  correctness := ``Lowering.rearrange_correct

/-- The repeat lowering. -/
def repeatTensor : Lowering where
  head := ``Lowering.expand
  arity := 5
  tensorArguments := #[4]
  family? := some "repeat"
  denotation := ``Semantics.denoteExpand
  correctness := ``Lowering.expand_correct

/-- The reduction lowering through an ordered fold with a finalizer. -/
def reduceFoldTensor : Lowering where
  head := ``Lowering.reduceFold
  arity := 11
  tensorArguments := #[10]
  family? := some "reduction"
  denotation := ``Semantics.denoteOrderedReduce
  correctness := ``Lowering.reduceFold_ordered_correct

/-- The reduction lowering through an ordered fold with a finalizer, read through a flat input
view. -/
def reduceFoldViewTensor : Lowering where
  head := ``Lowering.reduceFoldView
  arity := 13
  tensorArguments := #[10]
  family? := some "reduction"
  denotation := ``Semantics.denoteOrderedReduce
  correctness := ``Lowering.reduceFoldView_ordered_correct

/-- The reduction lowering through an ordered fold from the first value of each fiber. -/
def reduceNonemptyFoldTensor : Lowering where
  head := ``Lowering.reduceNonemptyFold
  arity := 7
  tensorArguments := #[6]
  family? := some "reduction"
  denotation := ``Semantics.denoteOrderedReduceNonempty
  correctness := ``Lowering.reduceNonemptyFold_correct

/-- The reduction lowering through an ordered fold from the first value of each fiber, read through
a flat input view. -/
def reduceNonemptyFoldViewTensor : Lowering where
  head := ``Lowering.reduceNonemptyFoldView
  arity := 9
  tensorArguments := #[6]
  family? := some "reduction"
  denotation := ``Semantics.denoteOrderedReduceNonempty
  correctness := ``Lowering.reduceNonemptyFoldView_correct

/-- The reduction lowering through an aggregate of each fiber. -/
def reduceTensor : Lowering where
  head := ``Lowering.reduce
  arity := 8
  tensorArguments := #[7]
  family? := some "reduction"
  denotation := ``Semantics.denoteReduce
  correctness := ``Lowering.reduce_correct

/-- The reduction lowering through an aggregate of each fiber, read through a flat input view. -/
def reduceViewTensor : Lowering where
  head := ``Lowering.reduceView
  arity := 10
  tensorArguments := #[7]
  family? := some "reduction"
  denotation := ``Semantics.denoteReduce
  correctness := ``Lowering.reduceView_correct

/-- The reduction lowering through an aggregate of each nonempty fiber. -/
def reduceNonemptyTensor : Lowering where
  head := ``Lowering.reduceNonempty
  arity := 9
  tensorArguments := #[8]
  family? := some "reduction"
  denotation := ``Semantics.denoteReduceNonempty
  correctness := ``Lowering.reduceNonempty_correct

/-- The einsum lowering through a generated kernel. -/
def einsumTensorKernel : Lowering where
  head := ``Lowering.einsumKernel
  arity := 12
  tensorArguments := #[7]
  family? := some "einsum"
  denotation := ``Semantics.denoteEinsum
  stages := #[``Semantics.einsumProduct]
  correctness := ``Lowering.einsumKernel_correct

/-- The fused einsum lowering. -/
def einsumTensor : Lowering where
  head := ``Lowering.einsum
  arity := 8
  tensorArguments := #[7]
  family? := some "einsum"
  denotation := ``Semantics.denoteEinsum
  correctness := ``Lowering.einsum_correct

/-- The pack lowering. -/
def packTensor : Lowering where
  head := ``Lowering.pack
  arity := 4
  tensorArguments := #[3]
  family? := some "pack/unpack"
  denotation := ``Semantics.denotePack
  correctness := ``Lowering.pack_correct

/-- The unpack lowering. -/
def unpackTensor : Lowering where
  head := ``Lowering.unpack
  arity := 4
  tensorArguments := #[3]
  family? := some "pack/unpack"
  denotation := ``Semantics.denoteUnpack
  correctness := ``Lowering.unpack_correct

/-- The `parse_shape` metadata check, which has no tensor input. -/
def checkParseShape : Lowering where
  head := ``ParseShape.check
  arity := 2
  tensorArguments := #[]

/-- The native kernel wrapper around a reference program, which `einops?` looks through. -/
def nativeTensorKernel : Lowering where
  head := ``Native.tensorKernel
  arity := 8
  tensorArguments := #[3]
  correctness := ``Native.tensorKernel_correct

/-- The applications that `einops?` reports, with each family contiguous and in report order. -/
def reported : Array Lowering := #[
  transformTensorFused, rearrangeTensor, repeatTensor, reduceFoldTensor, reduceFoldViewTensor,
  reduceNonemptyFoldTensor, reduceNonemptyFoldViewTensor, reduceTensor, reduceViewTensor,
  reduceNonemptyTensor, einsumTensorKernel, einsumTensor, packTensor, unpackTensor,
  checkParseShape]

end Lowering

end TorchLean.Einops
