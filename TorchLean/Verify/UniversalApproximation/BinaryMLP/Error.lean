/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Verify.UniversalApproximation.BinaryMLP

/-!
# Configured binary two-layer ReLU approximation bound

A single-hidden-layer ReLU MLP executed in any configured binary format and model codec
approximates a target within the sum of three error terms, one per source of error:

- **real approximation**: the ideal real-valued ReLU MLP approximates the target,
- **parameter quantization**: the real MLP is close to the real interpretation of the binary
  parameters, and
- **execution**: the executable graph, interpreted back into $\mathbb{R}$, is close to the
  real graph with those interpreted parameters.

The theorem is the `n`-input counterpart of the one-dimensional hinge-network bounds in
`TorchLean.Verify.UniversalApproximation.Binary`. Here all three bounds are hypotheses, while the
hinge file proves its execution term from IEEE rounding. The split is the usual one for
floating-point algorithms: prove the real algorithm correct, then bound parameter rounding and
arithmetic rounding separately. For background, see IEEE Std 754-2019, Goldberg (1991), Higham
(2002), and the ReLU density literature of Cybenko, Hornik, Leshno, and Pinkus.
-/

@[expose] public section

open FloatLib.Floats (ExecFloat)
open FloatLib.Floats.Formats.BinaryInterchange


namespace TorchLean.UniversalApproximation

open _root_.TorchLean.Tensor

noncomputable section

variable {format : FloatFormat} {plan : Configured.StoragePlan format} {code : Type}
    [ExecFloat.ModelCodec plan (Model format) code]

local notation "Value" => ExecFloat (Configured.Family format code plan)

/-!
## Three-term bound

For a configured binary input `xI` with real interpretation $x_R$ (`tensorToReal`, entrywise
`toReal`), suppose that

1. a real two-layer ReLU MLP approximates the target $f$ at $x_R$ with error at most
   $\varepsilon_{\mathrm{approx}}$,
2. that MLP differs from the real MLP on the real interpretation of the binary parameters by at
   most $\varepsilon_Q$, and
3. executing the binary MLP and mapping the output to $\mathbb{R}$ differs from the real MLP on
   those interpreted parameters by at most $\varepsilon_R$.

Then the binary execution approximates $f$ within
$\varepsilon_{\mathrm{approx}}+\varepsilon_Q+\varepsilon_R$.
-/

/-- On every input of `D`, the executed MLP read through `toReal` is within
`εApprox + εQ + εR` of `f`. The execution bound `hR` is a hypothesis about `Model.toReal` of the
executed output (which sends NaN and infinities to `0`), and the caller proves the rounding and
finiteness facts behind it. -/
theorem relu_mlp_binary_three_term
    {n hidDim : Nat}
    (D : Set (Tensor Value [n]))
    (f : Tensor ℝ [n] → ℝ)
    (l1R : Linear ℝ n hidDim) (l2R : Linear ℝ hidDim 1)
    (l1I : Linear Value n hidDim) (l2I : Linear Value hidDim 1)
    (εApprox εQ εR : ℝ)
    (hApprox :
      ∀ xI ∈ D,
        let xR : Tensor ℝ [n] := tensorToReal xI
        |f xR - mlpEval (n := n) (hidDim := hidDim) l1R l2R xR| ≤ εApprox)
    (hQ :
      ∀ xI ∈ D,
        let xR : Tensor ℝ [n] := tensorToReal xI
        |mlpEval (n := n) (hidDim := hidDim) l1R l2R xR
          - mlpEval (n := n) (hidDim := hidDim) (linearToReal l1I) (linearToReal l2I)
            xR| ≤ εQ)
    (hR :
      ∀ xI ∈ D,
        let xR : Tensor ℝ [n] := tensorToReal xI
        |(ExecFloat.Binary.toModel (mlpEval (n := n) (hidDim := hidDim) l1I l2I
          xI)).toReal
          - mlpEval (n := n) (hidDim := hidDim) (linearToReal l1I) (linearToReal l2I)
            xR| ≤ εR) :
    ∀ xI ∈ D,
      let xR : Tensor ℝ [n] := tensorToReal xI
      |f xR - (ExecFloat.Binary.toModel (mlpEval (n := n) (hidDim := hidDim) l1I l2I
        xI)).toReal|
        ≤ εApprox + εQ + εR := by
  intro xI hxI
  -- Name the three intermediate values so the final bound reads as a textbook triangle argument.
  set xR : Tensor ℝ [n] := tensorToReal xI
  set yU : ℝ := mlpEval (n := n) (hidDim := hidDim) l1R l2R xR
  set yQ : ℝ := mlpEval (n := n) (hidDim := hidDim) (linearToReal l1I) (linearToReal
    l2I) xR
  set yI : ℝ := (ExecFloat.Binary.toModel (mlpEval (n := n) (hidDim := hidDim) l1I l2I
    xI)).toReal
  have h1 : |f xR - yU| ≤ εApprox := hApprox xI hxI
  have h2 : |yU - yQ| ≤ εQ := hQ xI hxI
  have h3 : |yI - yQ| ≤ εR := hR xI hxI
  -- Chain two triangle inequalities: first through the real approximant, then through the
  -- quantized real interpretation of the executable parameters.
  calc
    |f xR - yI| ≤ |f xR - yU| + |yU - yI| := abs_sub_le (f xR) yU yI
    _ ≤ |f xR - yU| + (|yU - yQ| + |yQ - yI|) := add_le_add le_rfl (abs_sub_le yU yQ yI)
    _ = |f xR - yU| + (|yU - yQ| + |yI - yQ|) := by rw [abs_sub_comm yQ yI]
    _ ≤ εApprox + (εQ + εR) := add_le_add h1 (add_le_add h2 h3)
    _ = εApprox + εQ + εR := by ring

end
end TorchLean.UniversalApproximation
