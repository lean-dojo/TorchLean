/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Verify.RuntimeApprox.NF.BackwardOps.Context
public import TorchLean.Verify.RuntimeApprox.Graph.Backward

/-!
# NF Backpropagation Approximation

The end-to-end theorem for the rounded NF reverse-mode backend. Given approximate inputs and
seeds, a graph built from local approximation contracts produces a rounded reverse-pass context
within the propagated bounds of the supplied spec reverse pass. Derivative correctness of that
spec pass is a separate premise when interpreting the result as a gradient guarantee.
-/
@[expose] public section

namespace TorchLean
namespace NF

open TorchLean.Tensor

noncomputable section

open FloatLib FloatLib.Numerics FloatLib.Floats.Formats
open Flocq

variable {β : Radix} {fexp : ℤ → ℤ} [ValidExp fexp]
variable {rnd : ℝ → ℤ} [ValidRndToNearest rnd]

local notation "R" => NF β fexp rnd

-- ---------------------------------------------------------------------------
-- Global reverse-mode bound (specialized to NF accumulation)
-- ---------------------------------------------------------------------------

/--
End-to-end NF reverse-mode soundness for a well-typed reverse graph.

The theorem composes the node-local approximation bounds with the NF context-addition bound
`ctxAddBound`, which charges the rounding of each addition of a node's contribution to the running
seed. Its premises are the input and seed approximations within `epsIn` and `epsSeed`.
-/
theorem backprop_approx {Γ : List Shape} {ss : List Shape}
    (g : RevGraph (α := R) (toSpec := NF.toReal) Γ ss) :
    ∀ (xS : TensorPack ℝ Γ) (xR : TensorPack R Γ) (epsIn : EList Γ)
      (seedS : TensorPack ℝ (Γ ++ ss))
      (seedR : TensorPack R (Γ ++ ss)) (epsSeed : EList (Γ ++ ss)),
      ApproxCtx (α := R) (toSpec := NF.toReal) xS xR epsIn →
      ApproxCtx (α := R) (toSpec := NF.toReal) seedS seedR epsSeed
        →
        ApproxCtx (α := R) (toSpec := NF.toReal)
          (RevGraph.backpropReal g xS seedS)
          (RevGraph.backpropRuntime g xR seedR)
          (RevGraph.backpropBounds g epsIn xR epsSeed seedR (ctxAddBound (β := β) (fexp := fexp)
            (rnd := rnd))) := by
  intro xS xR epsIn seedS seedR epsSeed hx hseed
  simpa using
    (RevGraph.backprop_approx (α := R) (toSpec := NF.toReal) g
      (addBound := ctxAddBound (β := β) (fexp := fexp) (rnd := rnd))
      (addSound := fun {Δ} => ApproxCtx.add (β := β) (fexp := fexp) (rnd := rnd) (Δ := Δ))
      xS xR epsIn seedS seedR epsSeed hx hseed)

end

end NF
end TorchLean
