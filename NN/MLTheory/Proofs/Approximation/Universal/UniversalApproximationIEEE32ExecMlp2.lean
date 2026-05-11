/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.MLTheory.Proofs.Approximation.Universal.IEEE32ExecCore

/-!
# IEEE32Exec two-layer ReLU approximation bound

This file proves the reusable three-term error decomposition for executing a
single-hidden-layer ReLU MLP under concrete IEEE binary32 semantics.

The theorem separates the three mathematically different sources of error:

- **real approximation**: the ideal real-valued ReLU MLP approximates the target,
- **parameter quantization**: the real MLP is close to the real interpretation of the IEEE
  parameters, and
- **IEEE execution**: the executable graph, interpreted back into `ℝ`, is close to the real graph
  with those interpreted parameters.

This is the finite-dimensional analogue of the hinge-network executable bound in
`UniversalApproximationIEEE32Exec`.  The decomposition follows the standard numerical-analysis
pattern for floating-point algorithms: prove the real algorithm correct, bound data/parameter
rounding, and bound arithmetic rounding separately.  For background, see IEEE Std 754-2019,
Goldberg (1991), Higham (2002), and the ReLU density literature of Cybenko, Hornik, Leshno, and
Pinkus.
-/

@[expose] public section


namespace NN.MLTheory.Proofs.UniversalApproximation
namespace IEEE32ExecMLP2

open _root_.Spec
open _root_.Spec.Tensor
open NN.MLTheory.Proofs.ReLUMlpBridge
open TorchLean.Floats.IEEE754
open IEEE32ExecCore

noncomputable section

/-!
## Three-term bound

Read this as:

Given an IEEE32Exec input `xI`, let `xR` be its real interpretation (`toReal` elementwise).
If:

1) the target `f` is approximated by a real 2-layer ReLU MLP (error ≤ εApprox),
2) the real MLP is close to the real interpretation of the IEEE parameters (error ≤ εQ),
3) executing the IEEE MLP and then mapping to reals is close to the real interpretation
   of those IEEE parameters (error ≤ εR),

then the IEEE32Exec execution approximates `f` within εApprox + εQ + εR.
-/

theorem relu_mlp2_ieee32exec_threeTerm
    {n hidDim : Nat}
    (D : Set (Tensor IEEE32Exec (.dim n .scalar)))
    (f : TensorVec n → ℝ)
    (l1R : LinearSpec ℝ n hidDim) (l2R : LinearSpec ℝ hidDim 1)
    (l1I : LinearSpec IEEE32Exec n hidDim) (l2I : LinearSpec IEEE32Exec hidDim 1)
    (εApprox εQ εR : ℝ)
    (hApprox :
      ∀ xI ∈ D,
        let xR : TensorVec n := tensorToReal xI
        |f xR - mlpEvalNd (n := n) (hidDim := hidDim) l1R l2R xR| ≤ εApprox)
    (hQ :
      ∀ xI ∈ D,
        let xR : TensorVec n := tensorToReal xI
        |mlpEvalNd (n := n) (hidDim := hidDim) l1R l2R xR
          - mlpEvalNd (n := n) (hidDim := hidDim) (linearSpecToReal l1I) (linearSpecToReal l2I)
            xR| ≤ εQ)
    (hR :
      ∀ xI ∈ D,
        let xR : TensorVec n := tensorToReal xI
        |IEEE32Exec.toReal (mlpEvalNdIeee32exec (n := n) (hidDim := hidDim) l1I l2I xI)
          - mlpEvalNd (n := n) (hidDim := hidDim) (linearSpecToReal l1I) (linearSpecToReal l2I)
            xR| ≤ εR) :
    ∀ xI ∈ D,
      let xR : TensorVec n := tensorToReal xI
      |f xR - IEEE32Exec.toReal (mlpEvalNdIeee32exec (n := n) (hidDim := hidDim) l1I l2I xI)|
        ≤ εApprox + εQ + εR := by
  intro xI hxI
  classical
  -- Name the three intermediate values so the final bound reads as a textbook triangle argument.
  set xR : TensorVec n := tensorToReal xI
  set yU : ℝ := mlpEvalNd (n := n) (hidDim := hidDim) l1R l2R xR
  set yQ : ℝ := mlpEvalNd (n := n) (hidDim := hidDim) (linearSpecToReal l1I) (linearSpecToReal
    l2I) xR
  set yI : ℝ := IEEE32Exec.toReal (mlpEvalNdIeee32exec (n := n) (hidDim := hidDim) l1I l2I xI)
  -- Pull in the approximation, quantization, and IEEE execution hypotheses at this point.
  have h1 : |f xR - yU| ≤ εApprox := by
    simpa [xR, yU] using (hApprox xI hxI)
  have h2 : |yU - yQ| ≤ εQ := by
    simpa [xR, yU, yQ] using (hQ xI hxI)
  have h3 : |yI - yQ| ≤ εR := by
    have := (hR xI hxI)
    simpa [xR, yI, yQ, abs_sub_comm] using this
  -- Chain two triangle inequalities: first through the real approximant, then through the
  -- quantized real interpretation of the executable parameters.
  have hfyI : |f xR - yI| ≤ |f xR - yU| + (|yU - yQ| + |yI - yQ|) := by
    have hB : |yU - yI| ≤ |yU - yQ| + |yI - yQ| := by
      calc
        |yU - yI| ≤ |yU - yQ| + |yQ - yI| := by
          simpa using (abs_sub_le yU yQ yI)
        _ = |yU - yQ| + |yI - yQ| := by
          rw [abs_sub_comm yQ yI]
    calc
      |f xR - yI| ≤ |f xR - yU| + |yU - yI| := by
        simpa using (abs_sub_le (f xR) yU yI)
      _ ≤ |f xR - yU| + (|yU - yQ| + |yI - yQ|) := by
        linarith [hB]
  have : |f xR - yI| ≤ εApprox + εQ + εR := by
    calc
      |f xR - yI| ≤ |f xR - yU| + (|yU - yQ| + |yI - yQ|) := hfyI
      _ ≤ εApprox + (εQ + εR) := by
        exact add_le_add h1 (add_le_add h2 h3)
      _ = εApprox + εQ + εR := by
        ring
  simpa [xR, yI] using this

end
end IEEE32ExecMLP2
end NN.MLTheory.Proofs.UniversalApproximation
