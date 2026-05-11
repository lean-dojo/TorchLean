/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import Mathlib.Data.Real.Basic
public import NN.MLTheory.CROWN.Core
public import NN.Spec.Core.Tensor
public import NN.Spec.Core.TensorOps

/-!
# Lyapunov oracle

We isolate the single oracle assumption used by TorchLean's Lyapunov/CROWN workflow:
`crown_oracle`.

Everything else in the Lyapunov verification workflow should reduce to ordinary theorems that *depend*
on this axiom, but do not introduce additional axioms.

Trust-boundary policy:
- Repo linting allowlists `axiom crown_oracle` only in this file.
- Downstream modules should `import NN.MLTheory.CROWN.Lyapunov.Oracle` (directly or via
  `NN.MLTheory.CROWN.Lyapunov.Verification`) rather than defining new trusted axioms.

References:
- Lyapunov stability is the classical certificate pattern: prove `V > 0` and `Vdot < 0` on a
  region.
- The numeric bound producer is CROWN-style affine/interval propagation; see Zhang et al. (CROWN,
  NeurIPS 2018) and Xu et al. (auto_LiRPA/α-CROWN).
-/

@[expose] public section

namespace NN.MLTheory.CROWN.Lyapunov

open _root_.Spec
open NN.MLTheory.CROWN

/-- Certificate for Lyapunov verification over a boxed region. -/
structure LyapunovCert (α : Type) [Context α] (n : Nat) where
  /-- Region on which the bounds are claimed. -/
  region : Box α (.dim n .scalar)
  /-- Lower bound for `V`. -/
  V_lo : α
  /-- Upper bound for `V`. -/
  V_hi : α
  /-- Lower bound for `V̇`. -/
  Vdot_lo : α
  /-- Upper bound for `V̇`. -/
  Vdot_hi : α

/-- A neural Lyapunov function specification.

In the oracle-backed workflow, `Vdot` is an externally supplied decay witness; this file does not
derive it from dynamics on its own. -/
structure NeuralLyapunov (α : Type) [Context α] (n : Nat) where
  /-- Candidate Lyapunov scalar field. -/
  V : Tensor α (.dim n .scalar) → α
  /-- Orbital derivative or decay witness associated with `V`. -/
  Vdot : Tensor α (.dim n .scalar) → α

variable {α : Type} [Context α] {n : Nat}

/--
Opaque witness produced by an external checker that is trusted by project policy for this workflow.
It attests that the numeric bounds packaged in `cert` are sound for the given `lyap`.

This witness is intentionally *not* constructible inside Lean. It represents an external
artifact/checker run (e.g. α/β-CROWN) that we treat as a trust boundary.
-/
opaque CrownOracleWitness (lyap : NeuralLyapunov α n) (cert : LyapunovCert α n) : Type

/--
Oracle axiom: if the trusted external witness is supplied for a certificate, then the certificate
bounds hold.

This is the only trusted axiom for the Lyapunov oracle approach.
-/
axiom crown_oracle (lyap : NeuralLyapunov α n) (cert : LyapunovCert α n)
    (_w : CrownOracleWitness lyap cert) :
  (∀ x, Box.contains cert.region x → cert.V_lo ≤ lyap.V x ∧ lyap.V x ≤ cert.V_hi) ∧
  (∀ x, Box.contains cert.region x → cert.Vdot_lo ≤ lyap.Vdot x ∧ lyap.Vdot x ≤ cert.Vdot_hi)

end NN.MLTheory.CROWN.Lyapunov
