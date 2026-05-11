/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Proofs.RuntimeApprox.Graph.LinkAutogradAlgebra
public import NN.Proofs.RuntimeApprox.NF.BackwardOps

/-!
# NF End-To-End GraphData Bridge

End-to-end runtime→spec bridge for NF graphs executed as `GraphData`.

`NN.Proofs.RuntimeApprox.NF` provides per-op NF approximation lemmas and composes them over
`RevGraph` via `RevGraph.eval_approx` and `NFBackend.backprop_approx`.

This file links those results to the executable SSA/DAG form used by the proof-compiled runtime:
`Proofs.Autograd.Algebra.GraphData`.

In other words, this is where the abstract approximation graph model meets the executable graph
interpreter used elsewhere in TorchLean.
-/

@[expose] public section


namespace Proofs
namespace RuntimeApprox
namespace NFBackend

open Spec
open Tensor
open NN.MLTheory.Robustness.Spec

open LinkAutogradAlgebra
open Proofs.Autograd.Algebra

noncomputable section

open TorchLean.Floats
open Proofs.RuntimeRoundingApprox

variable {β : NeuralRadix} {fexp : ℤ → ℤ} [NeuralValidExp fexp]
variable {rnd : ℝ → ℤ} [NeuralValidRndToNearest rnd]

local notation "R" => TorchLean.Floats.NF β fexp rnd

omit [NeuralValidExp fexp] [NeuralValidRndToNearest rnd] in
/--
Executable forward-pass soundness for an NF `RevGraph` erased to `GraphData`.

The theorem says that evaluating the executable `GraphData` forward interpreter gives the same
runtime context covered by `RevGraph.eval_approx`, so the abstract graph approximation theorem
applies to the executable representation.
-/
theorem eval_approx_graphData {Γ : List Shape} {ss : List Shape}
    (g : RevGraph (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) Γ ss) :
    ∀ (xS : TList SpecScalar Γ) (xR : TList R Γ) (epsIn : EList Γ),
      approxCtx (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) xS xR epsIn →
      approxCtx (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
        (RevGraph.evalSpec g xS)
        (GraphData.eval (α := R) (Δ := Unit) (Γ := Γ) (ss := ss)
          (g := LinkAutogradAlgebra.RevGraph.toGraphData (α := R)
            (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) (Γ := Γ) (ss := ss) g)
          xR ())
        (RevGraph.evalBounds g epsIn xR) := by
  intro xS xR epsIn hx
  have h := RevGraph.eval_approx (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
      (Γ := Γ) (ss := ss) g xS xR epsIn hx
  have hEq :
      GraphData.eval (α := R) (Δ := Unit) (Γ := Γ) (ss := ss)
            (g := LinkAutogradAlgebra.RevGraph.toGraphData (α := R)
              (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) (Γ := Γ) (ss := ss) g)
            xR ()
        =
      RevGraph.evalRuntime (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) (Γ := Γ)
        (ss := ss) g xR :=
    LinkAutogradAlgebra.RevGraph.evalRuntime_of_toGraphData (α := R)
      (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) (Γ := Γ) (ss := ss) g xR
  simpa [hEq] using h

/--
Executable backward-pass soundness for an NF `RevGraph` erased to `GraphData`.

Given approximate inputs and approximate seed cotangents, executable `GraphData.backpropCtx`
approximates the real-spec reverse-mode result with the bound computed by the NF backend.
-/
theorem backprop_approx_graphData {Γ : List Shape} {ss : List Shape}
    (g : RevGraph (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) Γ ss) :
    ∀ (xS : TList SpecScalar Γ) (xR : TList R Γ) (epsIn : EList Γ)
      (seedS : TList SpecScalar (Γ ++ ss)) (seedR : TList R (Γ ++ ss)) (epsSeed : EList (Γ ++ ss)),
      approxCtx (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) xS xR epsIn →
      approxCtx (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) seedS seedR epsSeed
        →
        approxCtx (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
          (RevGraph.backpropSpec g xS seedS)
          (GraphData.backpropCtx (α := R) (Δ := Unit) (Γ := Γ) (ss := ss)
            (g := LinkAutogradAlgebra.RevGraph.toGraphData (α := R)
              (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) (Γ := Γ) (ss := ss) g)
            xR () seedR)
          (RevGraph.backpropBounds g epsIn xR epsSeed seedR (ctxAddBound (β := β) (fexp := fexp)
            (rnd := rnd))) := by
  intro xS xR epsIn seedS seedR epsSeed hx hseed
  have h := backprop_approx (β := β) (fexp := fexp) (rnd := rnd) (Γ := Γ) (ss := ss) g
      xS xR epsIn seedS seedR epsSeed hx hseed
  have hEq :
      GraphData.backpropCtx (α := R) (Δ := Unit) (Γ := Γ) (ss := ss)
            (g := LinkAutogradAlgebra.RevGraph.toGraphData (α := R)
              (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) (Γ := Γ) (ss := ss) g)
            xR () seedR
        =
      RevGraph.backpropRuntime (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) (Γ
        := Γ) (ss := ss) g xR seedR :=
    LinkAutogradAlgebra.RevGraph.backpropRuntime_of_toGraphData (α := R)
      (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) (Γ := Γ) (ss := ss) g xR seedR
  simpa [hEq] using h

end
end NFBackend
end RuntimeApprox
end Proofs
