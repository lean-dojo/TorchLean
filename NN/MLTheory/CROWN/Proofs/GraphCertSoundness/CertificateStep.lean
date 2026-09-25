/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import Mathlib.Analysis.SpecialFunctions.Trigonometric.Arctan
import Mathlib.Tactic.Measurability.Init
public import NN.MLTheory.CROWN.BoundOps.Lawful
public import NN.MLTheory.CROWN.Graph.Engine.Base

/-!
# IBP Certificate Step

Safe, option-returning certificate-step semantics for the graph IBP checker. This is the proof
layer counterpart of the executable bound propagation step.
-/

@[expose] public section

namespace NN.MLTheory.CROWN.Graph

open Spec TorchLean
open TorchLean.Tensor
open NN.MLTheory.CROWN

namespace CertSoundness

noncomputable section

/-!
## The IBP “certificate step” (safe, total)

This is a safe version of `propagateIBPNode` tailored to the subset of ops we prove soundness for.
It defines what it means for a per-node certificate to be “locally well-formed”.

Important: the runtime implementation `propagateIBPNode` uses `get!` on parent boxes; it assumes
topological order and that earlier boxes exist. Here we avoid partiality by returning `none`
whenever parents are missing.
-/

/-- Safe lookup of the interval box recorded for node id `pid`, `none` when out of range. -/
def getBox? (cert : Array (Option (FlatBox ℝ))) (pid : Nat) : Option (FlatBox ℝ) :=
  if _h : pid < cert.size then cert[pid]! else none

/--
Safe per-node IBP step for the checker semantics.

This is the total (option-returning) analogue of the runtime `propagateIBPNode`, restricted to the
ops handled in the soundness development.
-/
def certStepNode? (nodes : Array Node) (ps : ParamStore ℝ) (cert : Array (Option (FlatBox ℝ))) (id :
  Nat) :
    Option (FlatBox ℝ) :=
  let node := nodes[id]!
  match node.kind with
  | .input =>
      ps.inputBoxes[id]?
  | .const _ =>
      match ps.constVals[id]? with
      | none => none
      | some v => some { dim := v.n, lo := v.v, hi := v.v }
  | .detach =>
      match NN.IR.unaryParent? node.parents with
      | some p1 => getBox? cert p1
      | none => none
  | .add =>
      match NN.IR.binaryParents? node.parents with
      | some (p1, p2) =>
          match getBox? cert p1, getBox? cert p2 with
          | some B1, some B2 =>
              if _h : B1.dim = B2.dim then
                some (boxAdd (α := ℝ) B1 B2)
              else
                none
          | _, _ => none
      | none => none
  | .sub =>
      match NN.IR.binaryParents? node.parents with
      | some (p1, p2) =>
          match getBox? cert p1, getBox? cert p2 with
          | some B1, some B2 =>
              if _h : B1.dim = B2.dim then
                some (boxSub (α := ℝ) B1 B2)
              else
                none
          | _, _ => none
      | none => none
  | .mulElem =>
      match NN.IR.binaryParents? node.parents with
      | some (p1, p2) =>
          match getBox? cert p1, getBox? cert p2 with
          | some B1, some B2 => boxMulElem (α := ℝ) B1 B2
          | _, _ => none
      | none => none
  | .relu =>
      match NN.IR.unaryParent? node.parents with
      | some p1 =>
          match getBox? cert p1 with
          | some B => some (boxRelu (α := ℝ) B)
          | none => none
      | none => none
  | .tanh =>
      match NN.IR.unaryParent? node.parents with
      | some p1 =>
          match getBox? cert p1 with
          | some Xin =>
              let yB := NN.MLTheory.CROWN.Runtime.Ops.IBP.tanh (α := ℝ) (ofFlatBox (α
                := ℝ) Xin)
              some (toFlatBox (α := ℝ) Xin.dim yB)
          | none => none
      | none => none
  | .sigmoid =>
      match NN.IR.unaryParent? node.parents with
      | some p1 =>
          match getBox? cert p1 with
          | some Xin =>
              let yB := NN.MLTheory.CROWN.Runtime.Ops.IBP.sigmoid (α := ℝ) (ofFlatBox
                (α := ℝ) Xin)
              some (toFlatBox (α := ℝ) Xin.dim yB)
          | none => none
      | none => none
  | .sin =>
      match NN.IR.unaryParent? node.parents with
      | some p1 =>
          match getBox? cert p1 with
          | some Xin =>
              let yB := NN.MLTheory.CROWN.Runtime.Ops.IBP.sin (α := ℝ) (ofFlatBox (α
                := ℝ) Xin)
              some (toFlatBox (α := ℝ) Xin.dim yB)
          | none => none
      | none => none
  | .softplus =>
      match NN.IR.unaryParent? node.parents with
      | some p1 =>
          match getBox? cert p1 with
          | some B => boxSoftplus? B
          | none => none
      | none => none
  | .safeLog =>
      match NN.IR.binaryParents? node.parents with
      | some (p1, p2) =>
          match getBox? cert p1, getBox? cert p2 with
          | some B, some epsilon => boxSafeLog? B epsilon
          | _, _ => none
      | none => none
  | .cos =>
      match NN.IR.unaryParent? node.parents with
      | some p1 =>
          match getBox? cert p1 with
          | some Xin =>
              let yB := NN.MLTheory.CROWN.Runtime.Ops.IBP.cos (α := ℝ) (ofFlatBox (α
                := ℝ) Xin)
              some (toFlatBox (α := ℝ) Xin.dim yB)
          | none => none
      | none => none
  | .linear =>
      match NN.IR.unaryParent? node.parents with
      | some p1 =>
          match getBox? cert p1 with
          | some Xin => ibpLinear (α := ℝ) id ps Xin
          | none => none
      | none => none
  | .matmul =>
      match NN.IR.unaryParent? node.parents with
      | some p1 =>
          match getBox? cert p1 with
          | some Xin => ibpMatmul (α := ℝ) id ps Xin
          | none => none
      | none =>
          match NN.IR.binaryParents? node.parents with
          | some (p1, p2) =>
              match getBox? cert p1, getBox? cert p2 with
              | some left, some right =>
                  ibpBinaryMatmul? nodes[p1]!.outShape nodes[p2]!.outShape left right
              | _, _ => none
          | none => none
  | .conv configuration => do
      let parent ← NN.IR.unaryParent? node.parents
      let parentNode ← nodes[parent]?
      let input ← getBox? cert parent
      ibpConvNode configuration parentNode.outShape node.outShape id ps input
  | .concat axis => do
      let layout ← concatNodeLayout? nodes node axis
      let parents ← node.parents.mapM (getBox? cert)
      concatFlatBoxes? layout parents
  | _ =>
      none

/-- A certificate is *locally consistent* if every node equals `certStepNode?` at that node. -/
def CertLocalOK (g : Graph) (ps : ParamStore ℝ) (cert : Array (Option (FlatBox ℝ))) : Prop :=
  cert.size = g.nodes.size ∧
  ∀ id : Nat, id < g.nodes.size → cert[id]! = certStepNode? g.nodes ps cert id

end

end CertSoundness

end NN.MLTheory.CROWN.Graph
