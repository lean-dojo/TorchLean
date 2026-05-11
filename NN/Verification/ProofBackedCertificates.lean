/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Entrypoint.Verification

/-!
# Certificate theorem examples

Most files in this directory are runnable checkers: they parse an artifact, recompute the relevant
bounds, and reject the artifact if it disagrees with Lean. This file shows the complementary
theorem-backed layer. The public theorems below are the bridge from “the checker accepted this
certificate” to “the represented graph semantics is enclosed.”

The split is deliberate:

- executable examples are fast, concrete, and good for interoperability with JSON artifacts;
- theorem-backed results explain what those checks mean mathematically once their hypotheses are
  discharged for a supported graph and certificate format.

In other words, a JSON checker answers "does this artifact match Lean recomputation?" The theorem
handles below answer the stronger question: "assuming the local checker hypotheses, what semantic
fact follows?"
-/

@[expose] public section

namespace NN.Verification.ProofBackedCertificates

/-!
## Theorem handles

These aliases keep the important public theorem names visible in generated docs and catch
namespace/API mismatches during builds.
-/

/-- Local IBP certificate soundness: locally valid boxes enclose graph semantics. -/
noncomputable abbrev ibpCertificateSoundness :=
  NN.MLTheory.CROWN.Graph.CertSoundness.cert_encloses_semantics

/-- Concrete Lean `runIBP?` soundness against the recursive graph evaluator. -/
noncomputable abbrev runIBPSoundness :=
  NN.MLTheory.CROWN.Graph.CertSoundness.runIBP?_encloses_evalGraphRec

/-- Generic affine CROWN-family checker soundness over real semantics. -/
noncomputable abbrev crownCertificateSoundness :=
  NN.MLTheory.CROWN.Graph.CrownCertSoundness.crown_checker_encloses_semantics

/-- Schematic IEEE32Exec version used when the local transfer rule is supplied for floats. -/
noncomputable abbrev crownCertificateSoundnessIEEE32 :=
  NN.MLTheory.CROWN.Graph.CrownCertSoundness.crown_checker_encloses_semantics_ieee32exec

end NN.Verification.ProofBackedCertificates
