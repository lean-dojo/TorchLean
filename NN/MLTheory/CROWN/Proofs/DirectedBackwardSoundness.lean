/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.MLTheory.CROWN.Proofs.DirectedBackwardNode

/-!
# Soundness of the rounded backward objective

The executable reverse sweep encloses its output objective at every real graph point satisfying
the stored-parameter equations and the input IBP enclosures. The proof includes coefficient
rounding, repeated-parent accumulation, bias rounding, node discharge, and the crossing-zero
correction used when converting the input coefficient intervals to affine bounds.
-/

@[expose] public section

namespace NN.MLTheory.CROWN.Graph.DirectedBackward

open Spec TorchLean NN.IR
open TorchLean.Tensor
open NN.MLTheory.CROWN
open NN.MLTheory.CROWN.Graph.Internal
open scoped BigOperators

noncomputable section

variable {α : Type} [Storage α] [Context α] [BoundOps α] [LawfulBoundOps α]

local notation "value" => LawfulBoundOps.toReal (α := α)

/-- A successful rounded backward sweep returns affine bounds on the exact real output
objective. The hypotheses specify real node equations and directed scalar laws, never local
soundness of the verifier's backward steps. -/
theorem runDirectedBackwardObjective_encloses (hzero : value (0 : α) = 0)
    {g : Graph} {ps : ParamStore α} {ibp : Array (Option (FlatBox α))}
    {ctx : AffineCtx} {dims : Nat → Nat} {v : Nat → Nat → ℝ}
    (point : GraphPoint g.nodes ps ibp ctx dims v)
    (output : Nat) (houtput : output < g.nodes.size) (obj : FlatTensor α)
    (hdim : obj.n = dims output)
    {lower upper : AffineVec α ctx.inputDim 1}
    (hresult : runDirectedBackwardObjective g ps ctx ibp output obj = some (lower, upper)) :
    affineValue lower (fun i => v ctx.inputId i.val) ≤
        dot (dims output) (fun i => value (getAtOrZero obj.v [i])) (v output) ∧
      dot (dims output) (fun i => value (getAtOrZero obj.v [i])) (v output) ≤
        affineValue upper (fun i => v ctx.inputId i.val) := by
  let z := dot (dims output) (fun i => value (getAtOrZero obj.v [i])) (v output)
  let initial : DirectedBackwardState α :=
    { coeffs := (Array.replicate g.nodes.size none).set! output (some (pointCoeffBox obj))
      cstLo := 0, cstHi := 0 }
  let final := (List.finRange g.nodes.size).reverse.foldl
    (fun state i => directedBackwardNode g.nodes ps ibp ctx state i.val) initial
  have hinitial : SweepInvariant dims v ctx.inputId g.nodes.size g.nodes.size z initial :=
    initial_represents hzero dims v ctx.inputId g.nodes.size output houtput obj hdim
  have hfinal : SweepInvariant dims v ctx.inputId g.nodes.size 0 z final :=
    reverseSweep_preserves (directedBackwardNode g.nodes ps ibp ctx)
      (fun k state => SweepInvariant dims v ctx.inputId g.nodes.size k z state)
      g.nodes.size (fun k hk state h => backwardNode_preserves hzero point k hk state z h)
      g.nodes.size le_rfl initial hinitial
  let inputCoefficient := final.coeffs[ctx.inputId]!.getD
    { dim := ctx.inputDim
      lo := Tensor.full (α := α) (.dim ctx.inputDim .scalar) 0
      hi := Tensor.full (α := α) (.dim ctx.inputDim .scalar) 0 }
  simp only [runDirectedBackwardObjective, houtput, ↓reduceIte] at hresult
  change (if final.failed then none else
    (ibp[ctx.inputId]?).bind fun entry => entry.bind fun inputBox =>
      directedInputAffines ctx.inputDim inputBox inputCoefficient final.cstLo final.cstHi) =
        some (lower, upper) at hresult
  by_cases hfailed : final.failed = true
  · simp only [hfailed, ↓reduceIte] at hresult
    cases hresult
  simp only [hfailed] at hresult
  obtain ⟨entry, hentry, hresult⟩ := Option.bind_eq_some_iff.mp hresult
  obtain ⟨inputBox, hbox, hresult⟩ := Option.bind_eq_some_iff.mp hresult
  rw [hbox] at hentry
  have hlookup : ibp[ctx.inputId]! = some inputBox := by
    obtain ⟨hindex, hget⟩ := Array.getElem?_eq_some_iff.mp hentry
    simpa only [getElem!_pos (c := ibp) (i := ctx.inputId) hindex] using hget
  rcases hfinal with hf | ⟨hsize, f, c, hs, hz⟩
  · exact False.elim (hfailed hf)
  have hx : RowEncloses inputBox ctx.inputDim (v ctx.inputId) := by
    simpa only [point.input_dim] using point.ibp_encloses _ point.input_lt inputBox hlookup
  have ha : RowEncloses inputCoefficient ctx.inputDim (f ctx.inputId) :=
    inputRow_encloses hzero hs ctx.inputId ctx.inputDim
      (by simpa only [hsize] using point.input_lt) point.input_dim
  have hr := inputAffines_row_encloses hzero hx ha hs.2 hresult
  rw [frontier_zero, point.input_dim] at hz
  simpa only [hz] using hr

end

end NN.MLTheory.CROWN.Graph.DirectedBackward
