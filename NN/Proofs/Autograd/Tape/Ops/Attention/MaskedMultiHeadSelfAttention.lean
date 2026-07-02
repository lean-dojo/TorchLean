/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Proofs.Autograd.Tape.Ops.Attention.MultiHeadSelfAttention
public import NN.Proofs.Autograd.Tape.Ops.Attention.MaskedScaledDotProduct

/-!
# Additive-Bias Multi-Head Attention Core

This module proves the head-wise fixed-score-bias attention core. The surrounding
projection/split/merge graph lives in `MultiHeadSelfAttention.lean`; the theorem here is the
reusable additive-bias replacement for the score/probability/value part of that graph:

`softmax(c · QKᵀ + bias) V`.

The bias has shape `(heads × seq × seq)`. This is not the boolean causal-mask semantics: boolean
attention masks in the spec/runtime path use hard masking, where blocked entries contribute zero
softmax numerator. This file is for intentional finite additive score biases.
-/

@[expose] public section

namespace Proofs
namespace Autograd

open Spec

open scoped BigOperators

noncomputable section

namespace MultiHeadAttention

open TapeNodes
open DGraph

universe u v

/-- Context for a masked multi-head attention core: `[Q_heads, Kᵀ_heads, V_heads]`. -/
abbrev ΓMaskedCore (n numHeads headDim : Nat) : List Shape :=
  [ HeadsShape n numHeads headDim
  , KtShape n numHeads headDim
  , HeadsShape n numHeads headDim
  ]

/-- Saved tensors for the fixed-bias multi-head attention core. -/
abbrev ssMaskedCore (n numHeads headDim : Nat) : List Shape :=
  [ ScoresShape n numHeads       -- QKᵀ
  , ScoresShape n numHeads       -- scaled scores
  , ScoresShape n numHeads       -- scaled scores plus fixed bias
  , ScoresShape n numHeads       -- probabilities
  , HeadsShape n numHeads headDim -- per-head output
  ]

/-- Query-head index in the masked attention core context. -/
def idxMaskedCoreQ {n numHeads headDim : Nat} {ss : List Shape} :
    Idx (ΓMaskedCore n numHeads headDim ++ ss) (HeadsShape n numHeads headDim) :=
  ⟨⟨0, by simp [ΓMaskedCore]⟩, by simp [ΓMaskedCore]⟩

/-- Transposed-key index in the masked attention core context. -/
def idxMaskedCoreKt {n numHeads headDim : Nat} {ss : List Shape} :
    Idx (ΓMaskedCore n numHeads headDim ++ ss) (KtShape n numHeads headDim) :=
  ⟨⟨1, by simp [ΓMaskedCore]⟩, by simp [ΓMaskedCore, KtShape]⟩

/-- Value-head index in the masked attention core context. -/
def idxMaskedCoreV {n numHeads headDim : Nat} {ss : List Shape} :
    Idx (ΓMaskedCore n numHeads headDim ++ ss) (HeadsShape n numHeads headDim) :=
  ⟨⟨2, by simp [ΓMaskedCore]⟩, by simp [ΓMaskedCore]⟩

/--
Proof-carrying masked multi-head attention core.

The fixed `bias` is added after scaling the score tensor and before the row-wise softmax.
-/
def maskedCoreDGraph {n numHeads headDim : Nat}
    (c : ℝ) (bias : Vec (Shape.size (ScoresShape n numHeads)) := 0) :
    DGraph (ΓMaskedCore n numHeads headDim) (ssMaskedCore n numHeads headDim) := by
  classical
  let dg0 : DGraph (ΓMaskedCore n numHeads headDim) [] := DGraph.nil

  let nodeScores :
      Node (ΓMaskedCore n numHeads headDim) (ScoresShape n numHeads) :=
    TapeNodes.Batched.matmul
      (Γ := ΓMaskedCore n numHeads headDim)
      (h := numHeads) (m := n) (n := headDim) (p := n)
      (A := idxMaskedCoreQ (n := n) (numHeads := numHeads) (headDim := headDim) (ss := []))
      (B := idxMaskedCoreKt (n := n) (numHeads := numHeads) (headDim := headDim) (ss := []))
  let dg1 :=
    DGraph.snoc (dg := dg0) (node := nodeScores)
      (hn := TapeNodes.Batched.matmulFderiv
        (Γ := ΓMaskedCore n numHeads headDim)
        (h := numHeads) (m := n) (n := headDim) (p := n)
        (A := idxMaskedCoreQ (n := n) (numHeads := numHeads) (headDim := headDim) (ss := []))
        (B := idxMaskedCoreKt (n := n) (numHeads := numHeads) (headDim := headDim) (ss := [])))

  let idxScores :
      Idx (ΓMaskedCore n numHeads headDim ++ [ScoresShape n numHeads])
        (ScoresShape n numHeads) :=
    Idx.last (Γ := ΓMaskedCore n numHeads headDim) (ss := []) (τ := ScoresShape n numHeads)
  let nodeScaled :
      Node (ΓMaskedCore n numHeads headDim ++ [ScoresShape n numHeads])
        (ScoresShape n numHeads) :=
    TapeNodes.scale
      (Γ := ΓMaskedCore n numHeads headDim ++ [ScoresShape n numHeads])
      (s := ScoresShape n numHeads) (idx := idxScores) c
  let dg2 :=
    DGraph.snoc (dg := dg1) (node := nodeScaled)
      (hn := TapeNodes.scaleFderiv
        (Γ := ΓMaskedCore n numHeads headDim ++ [ScoresShape n numHeads])
        (s := ScoresShape n numHeads) (idx := idxScores) (c := c))

  let idxScaled :
      Idx (ΓMaskedCore n numHeads headDim ++ [ScoresShape n numHeads, ScoresShape n numHeads])
        (ScoresShape n numHeads) :=
    Idx.last (Γ := ΓMaskedCore n numHeads headDim) (ss := [ScoresShape n numHeads])
      (τ := ScoresShape n numHeads)
  let nodeMasked :
      Node (ΓMaskedCore n numHeads headDim ++ [ScoresShape n numHeads, ScoresShape n numHeads])
        (ScoresShape n numHeads) :=
    TapeNodes.affine
      (Γ := ΓMaskedCore n numHeads headDim ++ [ScoresShape n numHeads, ScoresShape n numHeads])
      (sIn := ScoresShape n numHeads) (sOut := ScoresShape n numHeads)
      idxScaled
      (1 : Vec (Shape.size (ScoresShape n numHeads)) →L[ℝ]
        Vec (Shape.size (ScoresShape n numHeads)))
      bias
  let dg3 :=
    DGraph.snoc (dg := dg2) (node := nodeMasked)
      (hn := TapeNodes.affineFderiv
        (Γ := ΓMaskedCore n numHeads headDim ++ [ScoresShape n numHeads, ScoresShape n numHeads])
        (sIn := ScoresShape n numHeads) (sOut := ScoresShape n numHeads)
        idxScaled
        (1 : Vec (Shape.size (ScoresShape n numHeads)) →L[ℝ]
          Vec (Shape.size (ScoresShape n numHeads)))
        bias)

  let idxMasked :
      Idx (ΓMaskedCore n numHeads headDim ++
        [ScoresShape n numHeads, ScoresShape n numHeads, ScoresShape n numHeads])
        (ScoresShape n numHeads) :=
    Idx.last (Γ := ΓMaskedCore n numHeads headDim)
      (ss := [ScoresShape n numHeads, ScoresShape n numHeads])
      (τ := ScoresShape n numHeads)
  let nodeProbs :
      Node (ΓMaskedCore n numHeads headDim ++
        [ScoresShape n numHeads, ScoresShape n numHeads, ScoresShape n numHeads])
        (ScoresShape n numHeads) :=
    TapeNodes.Batched.softmaxLast
      (Γ := ΓMaskedCore n numHeads headDim ++
        [ScoresShape n numHeads, ScoresShape n numHeads, ScoresShape n numHeads])
      (h := numHeads) (m := n) (n := n) (idx := idxMasked)
  let dg4 :=
    DGraph.snoc (dg := dg3) (node := nodeProbs)
      (hn := TapeNodes.Batched.softmaxLastFderiv
        (Γ := ΓMaskedCore n numHeads headDim ++
          [ScoresShape n numHeads, ScoresShape n numHeads, ScoresShape n numHeads])
        (h := numHeads) (m := n) (n := n) (idx := idxMasked))

  let idxProbs :
      Idx (ΓMaskedCore n numHeads headDim ++
        [ScoresShape n numHeads, ScoresShape n numHeads, ScoresShape n numHeads,
          ScoresShape n numHeads])
        (ScoresShape n numHeads) :=
    Idx.last (Γ := ΓMaskedCore n numHeads headDim)
      (ss := [ScoresShape n numHeads, ScoresShape n numHeads, ScoresShape n numHeads])
      (τ := ScoresShape n numHeads)
  let nodeOut :
      Node (ΓMaskedCore n numHeads headDim ++
        [ScoresShape n numHeads, ScoresShape n numHeads, ScoresShape n numHeads,
          ScoresShape n numHeads])
        (HeadsShape n numHeads headDim) :=
    TapeNodes.Batched.matmul
      (Γ := ΓMaskedCore n numHeads headDim ++
        [ScoresShape n numHeads, ScoresShape n numHeads, ScoresShape n numHeads,
          ScoresShape n numHeads])
      (h := numHeads) (m := n) (n := n) (p := headDim)
      (A := idxProbs)
      (B := idxMaskedCoreV (n := n) (numHeads := numHeads) (headDim := headDim)
        (ss := [ScoresShape n numHeads, ScoresShape n numHeads, ScoresShape n numHeads,
          ScoresShape n numHeads]))
  let dg5 :=
    DGraph.snoc (dg := dg4) (node := nodeOut)
      (hn := TapeNodes.Batched.matmulFderiv
        (Γ := ΓMaskedCore n numHeads headDim ++
          [ScoresShape n numHeads, ScoresShape n numHeads, ScoresShape n numHeads,
            ScoresShape n numHeads])
        (h := numHeads) (m := n) (n := n) (p := headDim)
        (A := idxProbs)
        (B := idxMaskedCoreV (n := n) (numHeads := numHeads) (headDim := headDim)
          (ss := [ScoresShape n numHeads, ScoresShape n numHeads, ScoresShape n numHeads,
            ScoresShape n numHeads])))

  simpa using dg5

/-- Reverse-mode theorem for the fixed-bias multi-head attention core. -/
theorem maskedCore_backpropVec_eq_adjoint_fderiv
    {n numHeads headDim : Nat}
    (c : ℝ) (bias : Vec (Shape.size (ScoresShape n numHeads)) := 0)
    (xV : CtxVec (ΓMaskedCore n numHeads headDim))
    (seedV : CtxVec (ΓMaskedCore n numHeads headDim ++ ssMaskedCore n numHeads headDim)) :
    Graph.backpropVec
        (Γ := ΓMaskedCore n numHeads headDim)
        (ss := ssMaskedCore n numHeads headDim)
        (maskedCoreDGraph (n := n) (numHeads := numHeads) (headDim := headDim) c bias).g xV seedV
      =
    (fderiv ℝ
        (Graph.evalVec
          (Γ := ΓMaskedCore n numHeads headDim)
          (ss := ssMaskedCore n numHeads headDim)
          (maskedCoreDGraph (n := n) (numHeads := numHeads) (headDim := headDim) c bias).g)
        xV).adjoint seedV :=
  DGraph.backpropVec_eq_adjoint_fderiv
    (dg := maskedCoreDGraph (n := n) (numHeads := numHeads) (headDim := headDim) c bias) xV seedV

/--
Composition theorem for the additive-bias core after a projection/split stage.

`projectPack` is the mathematical interface exposed by a full MHA front half: it builds
`[Q_heads, Kᵀ_heads, V_heads]` from an outer context.  The theorem composes that front half with the
proved additive-bias attention core.
-/
theorem maskedCoreAfterProjection_hasFDerivAt
    {E : Type u} [NormedAddCommGroup E] [NormedSpace ℝ E]
    {n numHeads headDim : Nat}
    (c : ℝ) (bias : Vec (Shape.size (ScoresShape n numHeads)) := 0)
    (projectPack : E → CtxVec (ΓMaskedCore n numHeads headDim))
    (DprojectPack : E →L[ℝ] CtxVec (ΓMaskedCore n numHeads headDim))
    (x : E)
    (hProject : HasFDerivAt projectPack DprojectPack x) :
    HasFDerivAt
      (fun z : E =>
        Graph.evalVec
          (Γ := ΓMaskedCore n numHeads headDim)
          (ss := ssMaskedCore n numHeads headDim)
          (maskedCoreDGraph (n := n) (numHeads := numHeads) (headDim := headDim) c bias).g
          (projectPack z))
      ((fderiv ℝ
          (Graph.evalVec
            (Γ := ΓMaskedCore n numHeads headDim)
            (ss := ssMaskedCore n numHeads headDim)
            (maskedCoreDGraph (n := n) (numHeads := numHeads) (headDim := headDim) c bias).g)
          (projectPack x)).comp DprojectPack)
      x := by
  classical
  let core := maskedCoreDGraph (n := n) (numHeads := numHeads) (headDim := headDim) c bias
  rcases Graph.hasFDerivAt_evalVec_and_jvp
      (Γ := ΓMaskedCore n numHeads headDim)
      (ss := ssMaskedCore n numHeads headDim)
      (g := core.g) core.hg (projectPack x) with
    ⟨Dcore, hDcore, _hJcore⟩
  have hFderiv :
      fderiv ℝ
        (Graph.evalVec
          (Γ := ΓMaskedCore n numHeads headDim)
          (ss := ssMaskedCore n numHeads headDim)
          core.g)
        (projectPack x) = Dcore := by
    simpa using hDcore.fderiv
  rw [hFderiv]
  have hfun :
      (fun z : E =>
        Graph.evalVec
          (Γ := ΓMaskedCore n numHeads headDim)
          (ss := ssMaskedCore n numHeads headDim)
          core.g
          (projectPack z)) =
        (Graph.evalVec
          (Γ := ΓMaskedCore n numHeads headDim)
          (ss := ssMaskedCore n numHeads headDim)
          core.g ∘ projectPack) := by
    rfl
  exact (hDcore.comp x hProject).congr_of_eventuallyEq hfun.eventuallyEq

/--
Full masked-attention composition contract.

The theorem separates the already-proved pieces of a GPT-style attention block:

* a differentiable front half that projects/splits inputs into `Q`, `Kᵀ`, and `V`;
* the proved finite-mask split-head attention core;
* a differentiable back half that merges/project outputs or packages residual data for the caller.

Instantiating `projectPack` and `mergePack` with the concrete projection/split/merge graphs gives
the full masked-MHA differentiability statement without changing the core proof.
-/
theorem projectedMaskedAttention_hasFDerivAt
    {E : Type u} [NormedAddCommGroup E] [NormedSpace ℝ E]
    {F : Type v} [NormedAddCommGroup F] [NormedSpace ℝ F]
    {n numHeads headDim : Nat}
    (c : ℝ) (bias : Vec (Shape.size (ScoresShape n numHeads)) := 0)
    (projectPack : E → CtxVec (ΓMaskedCore n numHeads headDim))
    (DprojectPack : E →L[ℝ] CtxVec (ΓMaskedCore n numHeads headDim))
    (mergePack : CtxVec (ΓMaskedCore n numHeads headDim ++ ssMaskedCore n numHeads headDim) → F)
    (DmergePack : CtxVec (ΓMaskedCore n numHeads headDim ++ ssMaskedCore n numHeads headDim) →L[ℝ] F)
    (x : E)
    (hProject : HasFDerivAt projectPack DprojectPack x)
    (hMerge :
      HasFDerivAt mergePack DmergePack
        (Graph.evalVec
          (Γ := ΓMaskedCore n numHeads headDim)
          (ss := ssMaskedCore n numHeads headDim)
          (maskedCoreDGraph (n := n) (numHeads := numHeads) (headDim := headDim) c bias).g
          (projectPack x))) :
    HasFDerivAt
      (fun z : E =>
        mergePack
          (Graph.evalVec
            (Γ := ΓMaskedCore n numHeads headDim)
            (ss := ssMaskedCore n numHeads headDim)
            (maskedCoreDGraph (n := n) (numHeads := numHeads) (headDim := headDim) c bias).g
            (projectPack z)))
      (DmergePack.comp
        ((fderiv ℝ
          (Graph.evalVec
            (Γ := ΓMaskedCore n numHeads headDim)
            (ss := ssMaskedCore n numHeads headDim)
            (maskedCoreDGraph (n := n) (numHeads := numHeads) (headDim := headDim) c bias).g)
          (projectPack x)).comp DprojectPack))
      x :=
  hMerge.comp x
    (maskedCoreAfterProjection_hasFDerivAt
      (n := n) (numHeads := numHeads) (headDim := headDim)
      c bias projectPack DprojectPack x hProject)

end MultiHeadAttention

end

end Autograd
end Proofs
