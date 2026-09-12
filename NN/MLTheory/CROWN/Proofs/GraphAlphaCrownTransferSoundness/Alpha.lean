/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.MLTheory.CROWN.Proofs.GraphAlphaCrownTransferSoundness.Common
public import NN.MLTheory.CROWN.Proofs.GraphAlphaCrownTransferSoundness.Alpha.Basic
public import NN.MLTheory.CROWN.Proofs.GraphAlphaCrownTransferSoundness.Alpha.Leaf
public import NN.MLTheory.CROWN.Proofs.GraphAlphaCrownTransferSoundness.Alpha.Shape
public import NN.MLTheory.CROWN.Proofs.GraphAlphaCrownTransferSoundness.Alpha.Linear
public import NN.MLTheory.CROWN.Proofs.GraphAlphaCrownTransferSoundness.Alpha.ReLU
public import NN.MLTheory.CROWN.Proofs.GraphAlphaCrownTransferSoundness.Alpha.Fallback

/-!
# α-CROWN Graph Transfer Soundness

Pointwise soundness theorem for the graph-dialect `alphaCrownStepNode?` transfer rule. The proof
dispatches on the node kind to the per-operator lemmas in the `Alpha` subdirectory: leaf nodes
(`Alpha.Leaf`), value-preserving nodes (`Alpha.Shape`), linear nodes (`Alpha.Linear`), ReLU
(`Alpha.ReLU`), and the IBP-derived constant fallback (`Alpha.Fallback`).
-/

@[expose] public section

namespace NN.MLTheory.CROWN.Graph

open Spec TorchLean
open TorchLean.Tensor
open scoped BigOperators
open Proofs.TensorAlgebra

open NN.MLTheory.CROWN
open NN.MLTheory.CROWN.Cert

namespace AlphaCrownTransferSoundness

noncomputable section

open CrownCertSoundness
open CertSoundness

/-! ## Main transfer theorem -/

/--
Pointwise soundness of the graph-dialect α-CROWN transfer rule.

Fix a graph `g`, parameters `ps`, an input point `x`, and a locally consistent value semantics
array `vals` (that is, `vals[id]` agrees with evaluating node `id` from its parents' values).

Assume:

- the designated input node in `inputs` matches `x` (`InputsMatch`),
- the IBP boxes `ibp` enclose the semantic values in `vals` (`IBPEnclosesVals`), and
- the α parameters are well-formed (`AlphaOK`).

Then the concrete step function `alphaCrownStepNode?` satisfies the abstract
`CrownTransferSound` requirement: whenever every parent `p` is enclosed by its certificate entry,
the current node `id` is enclosed by the step-produced certificate entry as well.

This is the key lemma that lets `alphaCrownStepNode?` plug into the generic end-to-end checker
theorem in `NN.MLTheory.CROWN.Proofs.GraphCrownCertSoundness`.
-/
theorem alphaCrown_transfer_sound
    (g : Graph) (ps : ParamStore ℝ)
    (ibp : Array (Option (FlatBox ℝ)))
    (alpha : Array (Option (FlatTensor ℝ)))
    (cert : Array (Option (FlatAffineBounds ℝ)))
    (inputs : Std.HashMap Nat Val)
    (vals : Array (Option Val))
    (ctx : AffineCtx) (x : Tensor ℝ [ctx.inputDim])
    (htopo : TopoSorted g)
    (hsem : SemLocalOK (g := g) (ps := ps) (inputs := inputs) vals)
    (hinputs : InputsMatch (inputs := inputs) (ctx := ctx) x)
    (hibp : IBPEnclosesVals (ibp := ibp) (vals := vals))
    (halpha : AlphaOK (alpha := alpha)) :
    CrownTransferSound
      (g := g) (_ps := ps) (_inputs := inputs) (vals := vals)
      (ctx := ctx) (x := x)
      (step := stepAlpha g ps ibp alpha ctx) (cert := cert) := by
  intro id hid hparents
  have hpar : Alpha.ParentsEnclosed g cert vals ctx x id := Alpha.parentsEnclosed_of_match hparents
  split
  · next b v hs hv =>
    have hEvalSome : evalNode? g.nodes ps inputs vals id = some v :=
      Alpha.evalNode?_eq_some_of_semLocalOK hsem hid hv
    have hlt : id < vals.size := lt_of_lt_of_eq hid hsem.1.symm
    have hparLt : ∀ p : Nat, p ∈ (g.nodes[id]!).parents → p < vals.size := fun p hp =>
      lt_of_lt_of_eq (lt_trans (htopo id hid p hp) hid) hsem.1.symm
    -- Split by node kind, mirroring `alphaCrownStepNode?`.
    match hk : (g.nodes[id]!).kind with
    | .input => exact Alpha.input_sound hk hs hEvalSome hinputs
    | .const _ => exact Alpha.const_sound hk hs hEvalSome
    | .detach => exact Alpha.detach_sound hk hs hEvalSome hpar
    | .reshape _ _ => exact Alpha.reshape_flatten_sound (Or.inl ⟨_, _, hk⟩) hs hEvalSome hpar
    | .flatten _ => exact Alpha.reshape_flatten_sound (Or.inr ⟨_, hk⟩) hs hEvalSome hpar
    | .linear => exact Alpha.linear_sound hk hs hEvalSome hpar
    | .matmul => exact Alpha.matmul_sound hk hs hEvalSome hpar
    | .sum => exact Alpha.sum_sound hk hs hEvalSome hpar
    | .relu => exact Alpha.relu_sound hk hs hEvalSome hpar hparLt hibp halpha
    | .conv _ | .layernorm _ | .concat _ =>
      -- IBP fallback guarded by `crownNodeSemanticsSupported`.
      simp only [stepAlpha, alphaCrownStepNode?, hk] at hs
      split at hs
      · split at hs
        · next B0 hib => exact Alpha.fallback_sound hib (Option.some.inj hs) hv hlt hibp
        · cases hs
      · cases hs
    | .permute _ | .randUniform _ | .bernoulliMask _ | .add | .sub | .mulElem | .abs | .sqrt
    | .inv | .maxElem | .minElem | .maxPool _ | .avgPool _ | .broadcastTo _ _ | .reduceSum _
    | .reduceMean _ | .batchNormEval _ _ | .tanh | .sigmoid | .softplus | .safeLog
    | .exp | .log | .sin | .cos
    | .softmax _ | .hardMaskedSoftmax _ | .transpose _ _ | .mseLoss =>
      -- Unguarded IBP fallback.
      simp only [stepAlpha, alphaCrownStepNode?, hk] at hs
      split at hs
      · next B0 hib => exact Alpha.fallback_sound hib (Option.some.inj hs) hv hlt hibp
      · cases hs
  · trivial

end

end AlphaCrownTransferSoundness

end NN.MLTheory.CROWN.Graph
