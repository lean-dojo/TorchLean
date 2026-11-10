/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.Norm.LayerNorm.FDeriv
public import TorchLean.NN.Tape.Apply
public import TorchLean.Tensor.Op.Table.Normalization

/-!
# The eager LayerNorm rule

The eager tape records LayerNorm with `Tape.apply` of `Op.layerNorm`: it checks the shapes of the
input, scale, and bias, then saves those tensors and the frozen epsilon in one node. Its backward
closure calls `TorchLean.LayerNorm.backward` through the arm and tags each cotangent with the
corresponding parent id, in the arm's order.

The theorems here inspect that recorded node. The forward value is the `Tensor.layerNorm` output,
and applying its backward closure to a correctly shaped cotangent returns the blocks of the adjoint
of the Fréchet derivative for the parents that need a gradient. The input-read hypotheses describe
successful dynamic shape checks, and the derivative identity is proved from the LayerNorm formula
over the reals.

Parent ids may coincide, and the closure then returns one contribution per argument that needs a
gradient. The tape traversal sums the contributions that reach a shared parent. The theorems
concern the eager tape over `ℝ`, whose tensor arithmetic is exact real arithmetic.
-/

@[expose] public section

namespace TorchLean.LayerNorm

noncomputable section

variable {m n : Nat}

/-- When the input, scale, and bias reads succeed, recording LayerNorm with `Tape.apply` stores
`Tensor.layerNorm` over the last axis of the tensors read, with the frozen epsilon, at the
returned node id. -/
theorem eager_value (hm : 0 < m) (hn : 0 < n)
    (t : Tape (SomeTensor ℝ)) (xId gammaId betaId : Nat)
    (x : Tensor ℝ [m, n]) (gamma beta : Tensor ℝ [n]) (ε : ℝ)
    (hx : Tape.requireValue t xId = .ok x)
    (hgamma : Tape.requireValue t gammaId = .ok gamma)
    (hbeta : Tape.requireValue t betaId = .ok beta) :
    ((Tape.apply t "layer_norm"
        (Op.layerNorm hm hn (lead := [m]) (x := [m, n]) (Shape.size_vector m).symm rfl)
        (xId, gammaId, betaId, Tensor.scalar ε, ())).toOption.bind
      fun result => result.1.getValue? result.2.1) =
        some (SomeTensor.ofTensor (Tensor.layerNorm x [⟨1, Nat.one_lt_two⟩] gamma beta ε)) := by
  simp only [Tape.apply, Op.check, Op.rule?_layerNorm_matrix,
    Tape.Args.read, Tape.Arg.read, hx, hgamma, hbeta,
    Bind.bind, Except.bind, Pure.pure, Except.pure]
  simp [Except.toOption, Tape.record, Tape.valuePorts,
    Tape.Values.value?, Tape.prepared,
    Tape.aliasOf, Tape.Values.outs, Pure.pure, Except.pure,
    Tape.addNode, Tape.getValue?, Tape.size,
    Tape.getNode?, Tape.recordedNode, Tensor.item_scalar]

/-- For positive `ε`, the backward closure of the recorded node maps a cotangent to the input,
scale, and bias blocks of the adjoint of the LayerNorm derivative, kept for the parents that need
a gradient.

Reading the recorded node and invoking its closure both succeed. Each block carries its runtime
parent id, so the equality also checks that every argument slot sends its cotangent to the right
parent. -/
theorem eager_backward_eq_fderiv (hm : 0 < m) (hn : 0 < n)
    {ε : ℝ} (hε : 0 < ε)
    (t : Tape (SomeTensor ℝ)) (xId gammaId betaId : Nat)
    (x gradOutput : Tensor ℝ [m, n]) (gamma beta : Tensor ℝ [n])
    (hx : Tape.requireValue t xId = .ok x)
    (hgamma : Tape.requireValue t gammaId = .ok gamma)
    (hbeta : Tape.requireValue t betaId = .ok beta) :
    ((Tape.apply t "layer_norm"
        (Op.layerNorm hm hn (lead := [m]) (x := [m, n]) (Shape.size_vector m).symm rfl)
        (xId, gammaId, betaId, Tensor.scalar ε, ())).toOption.bind
      fun result =>
        (result.1.getNode? result.2.1).map
          fun node => node.backward (SomeTensor.ofTensor gradOutput)) =
      let gradient :=
        (fderiv ℝ (forwardVec ε) (pack x gamma beta)).adjoint
          (Tensor.toVec gradOutput)
      some (.ok (#[
        (xId, SomeTensor.ofTensor (inputX gradient)),
        (gammaId, SomeTensor.ofTensor (inputGamma gradient)),
        (betaId, SomeTensor.ofTensor (inputBeta gradient))].filter
          fun c => (t.getNode? c.1).any (·.requiresGrad))) := by
  rw [adjoint_fderiv_eq_backward hm hn hε]
  dsimp only
  rw [inputX_pack, inputGamma_pack, inputBeta_pack]
  simp only [Tape.apply, Op.check, Op.rule?_layerNorm_matrix,
    Tape.Args.read, Tape.Arg.read, hx, hgamma, hbeta,
    Bind.bind, Except.bind, Pure.pure, Except.pure]
  simp [Except.toOption, Tape.record, Tape.valuePorts,
    Tape.Values.value?, Tape.prepared,
    Tape.aliasOf, Tape.recordedNode,
    Tape.Values.outs, Pure.pure, Except.pure, Tape.size,
    Tape.addNode, Tape.getNode?,
    Tape.requireGrad, Tape.seed, Tape.emit,
    Tape.Grads.contributions, Tape.Args.ids, Tangents.select,
    Tape.Args.flags, Bind.bind, Except.bind, List.finRange,
    Tape.Grad.contribution?, Tape.Arg.id?, Port.select,
    List.filterMap, Tensor.item_scalar]
  by_cases h1 : Option.any (fun x => x.requiresGrad) t.nodes[xId]? = true <;>
  by_cases h2 : Option.any (fun x => x.requiresGrad) t.nodes[gammaId]? = true <;>
  by_cases h3 : Option.any (fun x => x.requiresGrad) t.nodes[betaId]? = true <;>
  simp [h1, h2, h3]

end

end TorchLean.LayerNorm
