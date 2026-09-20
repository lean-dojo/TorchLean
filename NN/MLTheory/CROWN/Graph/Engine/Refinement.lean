/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.MLTheory.CROWN.Graph.Engine.IBP

/-!
# Input subdivision for graph bounds

Subdivision reduces dependency overestimation without changing layer transfer rules. Each split
covers the entire parent box, including the shared boundary. Both children must produce bounds;
otherwise the parent result is retained. Combining children takes their hull, never their
intersection. That hull can then be intersected with the independently computed parent bound.

This is an optional, bounded-cost refinement for any supported graph, not a new soundness claim
about the underlying scalar transfers. It cannot repair an unsound transfer or guarantee strict
improvement for every layer. In particular, input-independent range fallbacks may remain unchanged.
-/

@[expose] public section

namespace NN.MLTheory.CROWN.Graph

open Spec TorchLean
open NN.MLTheory.CROWN

variable {α : Type} [TorchLean.Storage α] [Context α]

namespace Refinement

/-- Reject reversed or unordered endpoints before combining boxes. -/
def valid (box : FlatBox α) : Bool :=
  (List.finRange box.dim).all fun i => decide (box.lo.getScalar i < box.hi.getScalar i) ||
    (box.lo.getScalar i == box.hi.getScalar i)

/-- Split one coordinate at a shared boundary, leaving every other endpoint unchanged. -/
def splitAt (box : FlatBox α) (axis : Fin box.dim) (cut : α) : FlatBox α × FlatBox α :=
  ({ box with hi := Tensor.ofFn fun i => if i = axis then cut else box.hi.getScalar i },
   { box with lo := Tensor.ofFn fun i => if i = axis then cut else box.lo.getScalar i })

/-- Split the widest coordinate that has a representable interior midpoint.

The halved-endpoint formula avoids overflow in `lo + hi` and `hi - lo`. The strict endpoint checks
reject stagnation, unordered values, and infinite midpoints. A box with no such coordinate stays
unsplit; there is no arbitrary epsilon perturbation of its domain.
-/
def split? (box : FlatBox α) : Option (FlatBox α × FlatBox α) := Id.run do
  if !valid box then return none
  let mut best : Option (Fin box.dim × α × α) := none
  for i in List.finRange box.dim do
    let lo := box.lo.getScalar i
    let hi := box.hi.getScalar i
    let cut := lo / 2 + hi / 2
    if lo < cut && cut < hi then
      let width := hi - lo
      match best with
      | none => best := some (i, cut, width)
      | some (_, _, oldWidth) =>
        if oldWidth < width then best := some (i, cut, width)
  return best.map fun (i, cut, _) => splitAt box i cut

/-- Combine compatible boxes by their coordinatewise hull or intersection.

An invalid intersection is rejected, never returned as an empty certificate. The comparison checks
also prevent NaNs from disappearing through endpoint selection on native floating-point backends.
-/
def combine? (hull : Bool) (a b : FlatBox α) : Option (FlatBox α) := do
  if !valid a || !valid b then none
  else if h : b.dim = a.dim then
    let blo : Tensor α [a.dim] := h ▸ b.lo
    let bhi : Tensor α [a.dim] := h ▸ b.hi
    let result : FlatBox α :=
      { dim := a.dim
        lo := Tensor.ofFn fun i =>
          if hull then BoundOps.min2 (a.lo.getScalar i) (blo.getScalar i)
          else BoundOps.max2 (a.lo.getScalar i) (blo.getScalar i)
        hi := Tensor.ofFn fun i =>
          if hull then BoundOps.max2 (a.hi.getScalar i) (bhi.getScalar i)
          else BoundOps.min2 (a.hi.getScalar i) (bhi.getScalar i) }
    if valid result then some result else none
  else none

/-- Refine any box-to-box enclosure procedure using at most `budget` binary splits.

There are at most `2 * budget + 1` calls to `bound`. The budget is shared between the children,
so it counts work rather than an exponentially growing depth. Zero budget returns the original
result exactly. A failed child, dimension mismatch, or inconsistent intersection falls back to
that result; successful siblings alone never stand in for the complete input domain.
-/
def bound (enclose : FlatBox α → Option (FlatBox α)) (input : FlatBox α)
    (budget : Nat) : Option (FlatBox α) :=
  let baseline := enclose input
  match budget with
  | 0 => baseline
  | remaining + 1 =>
    match split? input with
    | none => baseline
    | some (left, right) =>
      let leftBudget := remaining / 2
      let rightBudget := remaining - leftBudget
      match bound enclose left leftBudget, bound enclose right rightBudget with
      | some a, some b =>
        match combine? true a b with
        | none => baseline
        | some hull =>
          match baseline with
          | none => some hull
          | some original => (combine? false original hull).or baseline
      | _, _ => baseline
termination_by budget
 decreasing_by
  all_goals have := Nat.div_le_self remaining 2
  all_goals omega

/-- Disabling subdivision preserves the supplied enclosure procedure exactly. -/
@[simp] theorem bound_zero (enclose : FlatBox α → Option (FlatBox α)) (input : FlatBox α) :
    bound enclose input 0 = enclose input := by
  rw [bound]

end Refinement

/-- Check that every declared graph input has a shape-compatible, ordered seed box. -/
def validIBPInputs (g : Graph) (ps : ParamStore α) : Bool :=
  (List.finRange g.nodes.size).all fun i =>
    let node := g.nodes[i]
    match node.kind with
    | .input =>
      match ps.inputBoxes[i.val]? with
      | some box => node.id == i.val && box.dim == node.outShape.size && Refinement.valid box
      | none => false
    | _ => true

/-- Refine one graph output by subdividing the selected input box.

Every other input and parameter stays fixed. The same operation works for dense, convolutional,
recurrent, attention, and mixed graphs to the extent their operators are supported by `runIBP`.
The caller selects the input node and total split budget explicitly. Invalid node selections return
`none`; unsupported transfers retain the ordinary IBP failure behavior.
-/
def refinedIBPOutput? [BoundOps α] [NonlinearBoundOps α]
    (g : Graph) (ps : ParamStore α) (inputId outId budget : Nat) : Option (FlatBox α) := do
  if !validIBPInputs g ps then none else do
    let node ← g.nodes[inputId]?
    match node.kind with
    | .input => pure ()
    | _ => none
    let input ← ps.inputBoxes[inputId]?
    if input.dim != node.outShape.size then none
    else
      let enclose := fun box => do
        let output ← ((runIBP g (ps.seedInputBox inputId box))[outId]?).join
        let outNode ← g.nodes[outId]?
        if output.dim == outNode.outShape.size && Refinement.valid output then some output else none
      Refinement.bound enclose input budget

end NN.MLTheory.CROWN.Graph
