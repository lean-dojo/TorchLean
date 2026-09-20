/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.MLTheory.CROWN.Core

/-!
# Lyapunov certificate semantics

This module defines the mathematical statement carried by a Lyapunov certificate. It does not
turn producer-reported bounds into a theorem. A checked workflow must prove `LyapunovCert.ValidFor`
from a verified graph evaluation or certificate checker; an external workflow must state that
validity assumption at its own trust boundary.

References:
- Lyapunov stability is the classical certificate pattern: prove `V > 0` and `Vdot < 0` on a
  region.
- The numeric bound producer is CROWN-style affine/interval propagation; see Zhang et al. (CROWN,
  NeurIPS 2018) and Xu et al. (auto_LiRPA/α-CROWN).
-/

@[expose] public section

namespace NN.MLTheory.CROWN.Lyapunov

open Spec TorchLean
open NN.MLTheory.CROWN

/-- Certificate for Lyapunov verification over a boxed region. -/
structure LyapunovCert (α : Type) [TorchLean.Storage α] [Context α] (n : Nat) where
  /-- Region on which the bounds are claimed. -/
  region : Box α (.dim n .scalar)
  /-- Lower bound for `V`. -/
  vLower : α
  /-- Upper bound for `V`. -/
  vUpper : α
  /-- Lower bound for `V̇`. -/
  derivativeLower : α
  /-- Upper bound for `V̇`. -/
  derivativeUpper : α

/-- A neural Lyapunov function specification.

`Vdot` is supplied by the application; this file does not derive it from dynamics on its own. -/
structure NeuralLyapunov (α : Type) [TorchLean.Storage α] [Context α] (n : Nat) where
  /-- Candidate Lyapunov scalar field. -/
  value : Tensor α [n] → α
  /-- Orbital derivative or decay witness associated with `V`. -/
  orbitalDerivative : Tensor α [n] → α

variable {α : Type} [TorchLean.Storage α] [Context α] {n : Nat}

/--
Proof object produced by a semantic certificate checker.

Parsing a certificate or checking the signs of its endpoints cannot construct this structure. Its
two fields require enclosure proofs for the actual functions named by `lyap` on the actual region
stored in `cert`.
-/
structure LyapunovCert.ValidFor (cert : LyapunovCert α n) (lyap : NeuralLyapunov α n) : Prop where
  /-- The checked interval for the Lyapunov candidate. -/
  valueBounds : ∀ x, Box.contains cert.region x →
    cert.vLower ≤ lyap.value x ∧ lyap.value x ≤ cert.vUpper
  /-- The checked interval for the orbital derivative. -/
  orbitalDerivativeBounds : ∀ x, Box.contains cert.region x →
    cert.derivativeLower ≤ lyap.orbitalDerivative x ∧
      lyap.orbitalDerivative x ≤ cert.derivativeUpper

end NN.MLTheory.CROWN.Lyapunov
