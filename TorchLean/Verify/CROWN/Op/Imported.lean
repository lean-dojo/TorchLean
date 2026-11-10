/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.IR.Lowering
public import TorchLean.Verify.CROWN.Op.ExecCROWN

/-!
# Bound passes on imported graphs

The bound passes of `TorchLean.CROWN.Op` run on the operator graph of an import (`IR.Imported`),
which also holds the payload and the references of the source values. Each pass starts from a box
`xB` at graph input `0`. `Imported.runIBP` runs interval bound propagation and returns the box of
every node, and `Imported.outputBox?` reads the box of the first output from such a table.
`Imported.outputBoxCROWN?` gives the CROWN box of the first output and
`Imported.backwardObjectiveBox?` the backward CROWN box of an objective at that output.

A lowered graph is an import (`LoweredModel.toImported`, `TorchLean.NN.IR.Lowering`).
-/

@[expose] public section

namespace TorchLean.IR

section Bounds

open CROWN

variable {α : Type} [Storage α] [Context α]

/-- Interval bound propagation on the operator graph of an import, with the box `xB` at graph
input `0`. -/
def _root_.TorchLean.IR.Imported.runIBP [BoundOps α] [NonlinearBoundOps α]
    [Op.Capabilities α] (imported : IR.Imported α) (xB : FlatBox α) :
    Array (Option (FlatBox α)) :=
  Op.runIBP imported.graph imported.payload (Op.inputSeeds imported.graph [xB])

/-- The box of the first output of an import. -/
def _root_.TorchLean.IR.Imported.outputBox? (imported : IR.Imported α)
    (boxes : Array (Option (FlatBox α))) : Except String (FlatBox α) :=
  Op.outputBox? imported.graph boxes 0

variable [BoundOps α] [NonlinearBoundOps α] [Op.Capabilities α]

/-- The CROWN box of the first output of an import, with the box `xB` at graph input `0`. -/
def _root_.TorchLean.IR.Imported.outputBoxCROWN? (imported : IR.Imported α) (xB : FlatBox α) :
    Except String (FlatBox α) :=
  Op.outputBoxCROWN? imported.graph imported.payload xB 0

/-- The backward CROWN box of the objective `obj` at the first output of an import, with the box
`xB` at graph input `0` and the interval table `boxes` of its operator graph. -/
def _root_.TorchLean.IR.Imported.backwardObjectiveBox? (imported : IR.Imported α)
    (boxes : Array (Option (FlatBox α))) (xB : FlatBox α) (obj : Graph.FlatTensor α) :
    Except String (FlatBox α) :=
  Op.backwardObjectiveBox? imported.graph imported.payload boxes xB 0 obj

end Bounds

end TorchLean.IR
