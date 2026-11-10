/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.Ops
public import TorchLean.Tensor.Op.Table
public import TorchLean.Tensor.Random

/-!
# The pure interpreter

`Ops Id α` reads every reference as a tensor and runs a registered operator by the first output of
its rule `Op.rule?`. It is a total denotation of a model whose parameters and buffer values are
passed in. It differs from the checked interpreters (the eager tape, the typed graph, the verifier
recorder) in three ways. It runs neither the attribute check `Op.check` nor the value-domain check
of the rule, and it updates no buffers. Where no rule is available (a `custom` node, or an operator
whose optional capability the record lacks), it returns zeros, an arbitrary totalization.

`applyRandom` uses one fixed key on every call, the first key of the stream under its seed, and
runs an operator without a key unchanged. Every random call under one seed therefore gets the same
key, the one a fresh eager session gives its first random operation.
-/

@[expose] public section

namespace TorchLean

open TorchLean.Tensor

variable {α : Type} [Storage α] [Context α]

/-- The first output of the rule of `op` at the capability record `caps` on `args`, or the zero
tensor when no rule is available. The zero is an arbitrary totalization, unrelated to what a custom
node computes. -/
def Ops.pureApply {ins rest : List Port} {s : Shape} (op : Op ins (.value s :: rest))
    (caps : Op.Capabilities α)
    (args : PortArgs α (fun s => Tensor α s) (fun β _ s => Tensor β s) ins) : Tensor α s :=
  match @Op.rule? α _ _ caps _ _ op with
  | some rule => (rule.forward (args.values ins)).1
  | none => Tensor.full s 0

/-- The pure interpreter: references are tensors, `apply` is `Ops.pureApply` (zeros where no rule is
available), and `applyRandom` uses the first key under its seed on every call. -/
instance : Ops Id α where
  Ref := fun s => Tensor α s
  DataRef := fun β _ s => Tensor β s
  dataConst := fun x => x
  mapData := fun f x => f x
  const := fun t => pure t
  observe := fun x => pure (Tensor.item x)
  apply := fun {_ _ _} op [supported : Op.Supported α op] args =>
    pure (Ops.pureApply op supported.caps args)
  applyRandom := fun {_ _ _} op [supported : Op.Supported α op] seed args =>
    pure <| match op.withKey? with
      | some withKey => Ops.pureApply (withKey (Random.keyOf seed 0)) supported.caps args
      | none => Ops.pureApply op supported.caps args

end TorchLean
