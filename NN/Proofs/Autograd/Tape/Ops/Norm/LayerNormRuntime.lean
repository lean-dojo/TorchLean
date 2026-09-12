/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Proofs.Autograd.Tape.Ops.Norm.LayerNormFDeriv
public import NN.Runtime.Autograd.Engine.Core.Neural

/-!
# The eager LayerNorm rule

The eager tape checks the shapes of its input, scale, and bias, then saves those tensors in one
LayerNorm node. Its backward closure calls `Spec.layerNormBackward` and tags each cotangent with
the corresponding parent id.

These theorems inspect that recorded node. The forward value is the actual `Spec.layerNorm`
output, and applying its backward closure to a correctly shaped cotangent returns the three
blocks of the adjoint Fréchet derivative. The input-read hypotheses describe successful dynamic
shape checks; the derivative identity is proved from the real LayerNorm formula.

Parent ids may coincide. The closure still returns one contribution per argument; accumulation
of contributions into a shared parent belongs to the tape traversal. This file concerns the
exact real instantiation of the eager tape, whose tensor arithmetic has mathematical semantics.
-/

@[expose] public section

namespace Proofs.Autograd.LayerNorm

open Spec TorchLean

noncomputable section

variable {m n : Nat}

/-- Recording LayerNorm saves its specified forward value at the returned node id.

The successful lookups retain the original scale and bias tensors and the configured epsilon. -/
theorem eager_layerNorm_value (hm : 0 < m) (hn : 0 < n)
    (t : Runtime.Autograd.Tape ℝ) (xId gammaId betaId : Nat)
    (x : Tensor ℝ [m, n]) (gamma beta : Tensor ℝ [n]) (ε : ℝ)
    (hx : Runtime.Autograd.Tape.requireValue t xId = .ok x)
    (hgamma : Runtime.Autograd.Tape.requireValue t gammaId = .ok gamma)
    (hbeta : Runtime.Autograd.Tape.requireValue t betaId = .ok beta) :
    ((Runtime.Autograd.Tape.layerNorm hm hn t xId gammaId betaId ε).toOption.bind
      fun result => result.1.getValue? result.2) =
        some (SomeTensor.ofTensor (Spec.layerNorm x gamma beta hm hn ε)) := by
  simp only [Runtime.Autograd.Tape.layerNorm, hx, hgamma, hbeta,
    Bind.bind, Except.bind, Pure.pure, Except.pure]
  simp [Except.toOption,
    Runtime.Autograd.Tape.addNode, Runtime.Autograd.Tape.getValue?,
    Runtime.Autograd.Tape.getNode?]

/-- The stored eager backward closure returns the adjoint derivative in input, scale, bias order.

Reading the just-recorded node and invoking its closure both succeed. The resulting array keeps
the runtime parent ids, so the equality also checks the correspondence between argument slots
and cotangent destinations. -/
theorem eager_layerNorm_backward_eq_adjoint_fderiv (hm : 0 < m) (hn : 0 < n)
    {ε : ℝ} (hε : 0 < ε)
    (t : Runtime.Autograd.Tape ℝ) (xId gammaId betaId : Nat)
    (x gradOutput : Tensor ℝ [m, n]) (gamma beta : Tensor ℝ [n])
    (hx : Runtime.Autograd.Tape.requireValue t xId = .ok x)
    (hgamma : Runtime.Autograd.Tape.requireValue t gammaId = .ok gamma)
    (hbeta : Runtime.Autograd.Tape.requireValue t betaId = .ok beta) :
    ((Runtime.Autograd.Tape.layerNorm hm hn t xId gammaId betaId ε).toOption.bind
      fun result =>
        (result.1.getNode? result.2).map
          fun node => node.backward (SomeTensor.ofTensor gradOutput)) =
      let gradient :=
        (fderiv ℝ (specLayerNormVec hm hn ε) (packLN x gamma beta)).adjoint
          (tensorToVec gradOutput)
      some (.ok #[
        (xId, SomeTensor.ofTensor (specX gradient)),
        (gammaId, SomeTensor.ofTensor (specGamma gradient)),
        (betaId, SomeTensor.ofTensor (specBeta gradient))]) := by
  rw [adjoint_fderiv_layerNorm_eq_layerNormBackward hm hn hε]
  dsimp only
  rw [specX_packLN, specGamma_packLN, specBeta_packLN]
  simp only [Runtime.Autograd.Tape.layerNorm, hx, hgamma, hbeta,
    Bind.bind, Except.bind, Pure.pure, Except.pure]
  simp [Except.toOption,
    Runtime.Autograd.Tape.addNode, Runtime.Autograd.Tape.getNode?,
    Runtime.Autograd.Tape.requireGrad]

end

end Proofs.Autograd.LayerNorm
