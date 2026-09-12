/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import Mathlib.Analysis.SpecialFunctions.Pow.Real
public import Mathlib.Analysis.SpecialFunctions.Trigonometric.Arctan
public import Mathlib.Analysis.SpecialFunctions.Trigonometric.DerivHyp
import Mathlib.Tactic.Measurability.Init
-- These two are discussed in the overview rather than used by it, which is exactly the shape
-- `lake shake` reads as dead. They stay: the overview is the only route by which they get
-- typechecked.
public import NN.MLTheory.CROWN.Proofs.GraphIBPBasicTheorems
public import NN.MLTheory.CROWN.Proofs.GraphRuntimeBridge

/-!
# CROWN/LiRPA soundness proofs (overview)

This directory contains proof-level soundness results for the CROWN/LiRPA family of bound
propagation methods, plus “checker-style” theorems for certificate checking on our verifier-graph
dialect (`NN.MLTheory.CROWN.Graph`).

The core pattern repeated throughout the development is:

1. Define a *value semantics* for a supported graph dialect (typically over `ℝ`).
2. Define a *bound-propagation step function* (IBP, CROWN, α-CROWN, α/β-CROWN) that computes a
   certificate entry from parent certificate entries.
3. Prove a *transfer-rule soundness* lemma for the step function.
4. Use a topological-order induction to obtain an end-to-end “checker implies enclosure” theorem.
5. Connect the proof-side objects to the executable ones: `GraphRunibpEndToEnd` identifies the
   engine's `runIBP` with the proof-side pass, and `GraphRuntimeBridge` shows that the runtime
   evaluator `NN.IR.Graph.evalNode` at `ℝ` is reproduced by the proof-side value semantics.

## Relation to existing implementations

These theorems are designed to align with the standard producer/checker split used in practical
verifiers:

- PyTorch is a common *producer* runtime for bounds and certificates (via eager execution and
  `torch` tensor semantics).
- `auto_LiRPA` provides a widely used LiRPA engine and a reference implementation of α-CROWN and
  β-CROWN style bound propagation on general computational graphs.
- `alpha-beta-CROWN` builds on `auto_LiRPA` and adds branch-and-bound and split constraints for
  complete verification workflows.

In TorchLean, we aim to keep the trusted core small: theorems are stated against the Lean semantics,
and any external solver can be treated as an untrusted certificate producer.

## Status note

`NN.MLTheory.CROWN.Proofs.GraphAlphaCrownTransferSoundness` contains the (large) op-by-op
transfer proofs for the concrete α-CROWN / α/β-CROWN step functions. It is not imported
by this umbrella module so that the overview remains a fast orientation point; import it directly
when you are working on those transfer proofs.

## References (code)

- `auto_LiRPA`: https://github.com/Verified-Intelligence/auto_LiRPA
- `alpha-beta-CROWN`: https://github.com/Verified-Intelligence/alpha-beta-CROWN

## References (papers)

- CROWN: Z. Zhang et al., 2018 (linear bound propagation for robustness).
- α-CROWN: K. Xu et al., ICLR 2021 (optimized relaxations).
- β-CROWN: S. Wang et al., NeurIPS 2021 (split constraints + BaB acceleration).
-/

@[expose] public section
