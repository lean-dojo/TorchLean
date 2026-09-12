/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.MLTheory.CROWN.Graph.Engine.CROWN.Node
public import NN.MLTheory.CROWN.Graph.Engine.IBP
public import NN.MLTheory.CROWN.Graph.Engine.CROWN.Linear -- shake: keep

@[expose] public section

namespace NN.MLTheory.CROWN.Graph

open Spec TorchLean
open TorchLean.Tensor
open NN.MLTheory.CROWN
open NN.IR

variable {α : Type} [TorchLean.Storage α] [Context α]
variable [BoundOps α]
variable [NonlinearBoundOps α]

open BoundOps

/-!
# Running CROWN

Graph traversal and output-box evaluation for the forward CROWN pass.
-/

/--
Run the forward CROWN affine-bounds pass from previously computed node intervals.

Nodes without a justified affine transfer retain their IBP enclosure as a constant affine bound.
-/
def runCROWN (g : Graph) (ps : ParamStore α) (ctx : AffineCtx)
    (ibp : Array (Option (FlatBox α))) : Array (Option (FlatAffineBounds α)) :=
  let init := Array.replicate g.nodes.size none
  if crownGraphSemanticsSupported (α := α) g ps then
    (List.finRange g.nodes.size).foldl (fun acc i =>
      propagateCROWNNode (α:=α) g.nodes ps ibp acc ctx i) init
  else
    init

/-- Evaluate already-computed CROWN output affine bounds on an input box. -/
def evalCROWNOutputBox? (bounds : Array (Option (FlatAffineBounds α))) (xB : FlatBox α)
    (outputId inputDim : Nat) : Except String (FlatBox α) := do
  let outAff ←
    match bounds[outputId]? with
    | some (some outAff) => pure outAff
    | some none => throw s!"CROWN produced no affine bound at output node {outputId}"
    | none => throw s!"output node {outputId} is out of bounds for {bounds.size} CROWN entries"
  if hIn : outAff.inDim = inputDim then
    if hXB : xB.dim = inputDim then
      let outB := outAff.evalOnFlatBox xB (by simpa [hXB] using hIn.symm)
      pure { dim := outAff.outDim, lo := outB.lo, hi := outB.hi }
    else
      throw s!"input box dimension mismatch: got {xB.dim}, expected {inputDim}"
  else
    throw s!"CROWN input dimension mismatch: got {outAff.inDim}, expected {inputDim}"

/--
Run IBP, run forward CROWN, and evaluate the output affine bounds on the selected input box.

This is the common "forward CROWN output box" workflow. It keeps callers from open-coding the same
output-array lookup and input-dimension proof checks around `runCROWN`.
-/
@[noinline, nospecialize]
def outputBoxCROWN? (g : Graph) (ps : ParamStore α) (xB : FlatBox α)
    (inputId outputId inputDim : Nat) : Except String (FlatBox α) := do
  let ibp := runIBP (α := α) g ps
  let ctx : AffineCtx := { inputId := inputId, inputDim := inputDim }
  let crown := runCROWN (α := α) g ps ctx ibp
  evalCROWNOutputBox? (α := α) crown xB outputId inputDim

namespace ParamStore

/--
Run `outputBoxCROWN?` from an input-seeded parameter store.

This method form reads naturally at call sites that already thread a `ParamStore`.
-/
def outputBoxCROWN? (ps : ParamStore α) (g : Graph) (xB : FlatBox α)
    (inputId outputId inputDim : Nat) : Except String (FlatBox α) :=
  Graph.outputBoxCROWN? (α := α) g ps xB inputId outputId inputDim

end ParamStore

end NN.MLTheory.CROWN.Graph
