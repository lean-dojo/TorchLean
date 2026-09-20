/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Proofs.RuntimeApprox.NF.BackwardOps.Backend
public import NN.Proofs.RuntimeApprox.Graph.BackwardApprox

/-!
# NF Backpropagation Approximation

The end-to-end theorem for the rounded NF reverse-mode backend.  A reverse graph built from sound
local nodes produces a runtime backpropagated context enclosed by the corresponding spec context.
-/
@[expose] public section

namespace Proofs
namespace RuntimeApprox

open Spec TorchLean
open TorchLean TorchLean.Tensor
open NN.MLTheory.Robustness.Spec
open Proofs.Autograd.Algebra

noncomputable section

namespace NFBackend

open FloatLib FloatLib.Numerics FloatLib.Floats.Formats
open Flocq
open Proofs.RuntimeRoundingApprox

variable {β : Radix} {fexp : ℤ → ℤ} [ValidExp fexp]
variable {rnd : ℝ → ℤ} [ValidRndToNearest rnd]

local notation "R" => NF β fexp rnd

-- ---------------------------------------------------------------------------
-- Global reverse-mode bound (specialized to NF accumulation)
-- ---------------------------------------------------------------------------

/--
End-to-end NF reverse-mode soundness for a well-typed reverse graph.

This is the main composition theorem: if each node in the graph has a sound `RevNode` instance,
then the whole backpropagated context is an `approxCtx` enclosure of the spec backpropagation.
-/
theorem backprop_approx {Γ : List Shape} {ss : List Shape}
    (g : RevGraph (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) Γ ss) :
    ∀ (xS : TorchLean.TensorPack SpecScalar Γ) (xR : TorchLean.TensorPack R Γ) (epsIn : EList Γ)
      (seedS : TorchLean.TensorPack SpecScalar (Γ ++ ss))
      (seedR : TorchLean.TensorPack R (Γ ++ ss)) (epsSeed : EList (Γ ++ ss)),
      approxCtx (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) xS xR epsIn →
      approxCtx (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) seedS seedR epsSeed
        →
        approxCtx (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
          (RevGraph.backpropSpec g xS seedS)
          (RevGraph.backpropRuntime g xR seedR)
          (RevGraph.backpropBounds g epsIn xR epsSeed seedR (ctxAddBound (β := β) (fexp := fexp)
            (rnd := rnd))) := by
  intro xS xR epsIn seedS seedR epsSeed hx hseed
  simpa using
    (RevGraph.backprop_approx (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) g
      (addBound := ctxAddBound (β := β) (fexp := fexp) (rnd := rnd))
      (addSound := fun {Δ} => approxCtx_add (β := β) (fexp := fexp) (rnd := rnd) (Δ := Δ))
      xS xR epsIn seedS seedR epsSeed hx hseed)

end NFBackend

end

end RuntimeApprox
end Proofs
