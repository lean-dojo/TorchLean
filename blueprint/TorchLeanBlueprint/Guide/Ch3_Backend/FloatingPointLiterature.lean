import VersoManual
import VersoBlueprint

open Verso.Genre Manual

#doc (Manual) "Approximation and Floating Point References" =>
%%%
tag := "uat-fp-literature"
%%%

This page is a citation map for numerical claims in TorchLean. Many papers use similar words:
approximation, robustness, bounds, finite precision. The words are not enough. Before citing a
theorem, we need to know whether it is about real-valued networks, rounded-real FP32 semantics,
executable IEEE-754 behavior, or an abstract verifier enclosure.

The purpose is to avoid citing the right theorem for the wrong semantics. Citations next to a
TorchLean theorem should name whether the claim is over reals, `FP32`, `IEEE32Exec`, or a
verifier-style enclosure, and should name the quantity being bounded: pointwise error, image of a
set, or an interval overapproximation.

# Three Questions Before Citing A Result

For a "floating-point universal approximation" claim, three concrete checks apply:

1. Is the theorem about reals, about a rounded real model, or about actual executable IEEE-754 bits?
2. What is the approximation target (a pointwise function, a direct image of a set, or an interval enclosure)?
3. If the theorem is meant to feed a verifier, what semantics is the verifier using?

TorchLean keeps these cases separate in code:

- `FP32` is the rounded real model used in proofs.
- `IEEE32Exec` is the executable binary32 kernel.
- `NN.IR.Graph` is the shared "this is the computation" object used by execution and verification.

Two common citation chains are:

- real target function, classical approximation error, FP32 rounding model, then interval or LiRPA
  verification semantics;
- executable `IEEE32Exec` bits, then the `FP32` proof model on the finite bridge path.

# Definitions we use in this book

- `direct image`: the exact output set `f(B)` of a network over an input region `B`.
- `interval abstraction`: a sound overapproximation of `f(B)` (a box, or a box plus affine forms).
- `rounded target`: the same computation after fixing a scalar semantics (reals vs FP32 vs IEEE bits).
- `bridge theorem`: a statement that connects an executable semantics to a proof semantics on a
  stated "finite path" (no NaN/Inf, no overflow, etc).

The same feed-forward architecture can therefore be discussed at several incompatible levels of
precision. Papers and theorems should name the level explicitly so readers can tell whether the
claim concerns real valued idealization, a float model used for proofs, executable IEEE-754
behavior, or a verifier's overapproximation of outputs.

# Key References

## Standards And Numerical Analysis

Start with IEEE 754 / ISO 60559 for the machine arithmetic target. This is the source for binary
formats, special values, rounding attributes, exceptions, and the claim that conforming operations
have determined results under specified formats and rounding modes.

Goldberg and Higham are the tutorial and numerical-analysis references to cite when explaining why
floating-point operations are rounded operations, why evaluation order matters, and why forward
error bounds are the right language for many ML proofs.

## Proof Assistant Floating Point

Flocq is the closest proof-engineering precedent for TorchLean's `FP32` layer: it formalizes
floating-point arithmetic in Coq using reusable rounded-real models. FloatSpec is a Lean 4 project
with the same broad ambition for IEEE-style arithmetic and executable reference operations.

Gappa is the reference to cite when the point is automatic certification of floating-point bounds:
it automates interval/error propagation and can emit independently checkable proof artifacts.

CompCert is useful background for the compilation side of the story: it is a verified compiler with
machine floating-point models in its formal development.

## Classical Universal Approximation

Cybenko (1989), Hornik, Stinchcombe, and White (1989), Leshno et al. (1993), and Pinkus (1999).

These are the right citations for the classical density statement over reals, before rounding or
verification enter the discussion.

## Quantization And Numerical Precision

Sakr et al., *Analytical Guarantees on Numerical Precision of Deep Neural Networks* (ICML 2017), and
papers on quantized ReLU approximation. The PMLR page is
https://proceedings.mlr.press/v70/sakr17a.html.

These are the right citations when the question is "what does finite precision do to my error
budget?" rather than "can the architecture approximate at all?"

## Abstract Interpretation And Interval Semantics

Gehr et al., *AI2: Safety and Robustness Certification of Neural Networks with Abstract
Interpretation* (2018), Singh et al., *An Abstract Domain for Certifying Neural Networks* (POPL
2019), and the CROWN / LiRPA line of work.

This is the lineage behind interval boxes and affine relaxations. It is the background for
TorchLean's verification and bound propagation passes.

## Floating Point Neural Networks As Robust Approximators

Hwang, Saad, et al., *Floating-Point Neural Networks Are Provably Robust Universal Approximators*
([arXiv:2506.16065](https://arxiv.org/abs/2506.16065), CAV 2025).

Among the references here, this paper is the closest match to how TorchLean separates concerns: it
distinguishes classical approximation over the reals from finite precision effects and from the
interval-style quantities that verification tools compute.

# What TorchLean Can Cite Today

TorchLean already has a chain checked by Lean that is more concrete than a paper sketch:

- a classical 1D ReLU universal-approximation theorem,
- a hinge-network variant of that theorem,
- and an FP32-flavored version with an explicit rounding bound.

```
import NN.MLTheory.Proofs.Approximation
import NN.MLTheory.Proofs.Approximation.Universal.UniversalApproximationIEEE32Exec

#check relu_universal_approximation_Icc
#check relu_universal_approximation_Icc_hinge
#check relu_universal_approximation_Icc_fp32
```

Read the three theorems as a staged chain:

- `relu_universal_approximation_Icc` states the classical real valued universal approximation result.
- `relu_universal_approximation_Icc_hinge` is a constructive hinge-network variant that exposes the
  piecewise-linear structure used in later proofs.
- `relu_universal_approximation_Icc_fp32` layers TorchLean's `FP32` rounding model on
  top of that approximation theorem, with an explicit rounding error budget.

Together they illustrate the pattern the rest of the numerics chapters follow: establish the
real valued property, fix a scalar semantics, then package a compositional error or refinement
lemma that links the two.

# Which Citation Goes With Which Claim?

Use this map when writing a paper, proposal, or technical note:

- ReLU networks approximate continuous functions over reals: cite
  `relu_universal_approximation_Icc` and the classical universal approximation references.
- FP32 rounded networks approximate a real target with an error budget: cite
  `relu_universal_approximation_Icc_fp32`, Higham-style forward error analysis, and the TorchLean
  `FP32` semantics.
- Executable IEEE32 statements: cite the `relu_universal_approximation_Icc_ieee32exec_*` family
  together with IEEE 754 / ISO 60559 and the `IEEE32Exec` semantics.
- Interval or LiRPA enclosures: cite the IBP/CROWN objects together with abstract interpretation
  and LiRPA references.
- Proof-assistant float semantics: cite `FP32`, `IEEE32Exec`, and related work such as Flocq,
  FloatSpec, Gappa, and CompCert.

## There is also an executable IEEE32Exec theorem line

For statements phrased directly over the executable binary32 model instead of the rounded real
`FP32` layer, the repo also contains the start of that line in:

- [IEEE32Exec universal approximation theorem](https://github.com/lean-dojo/TorchLean/blob/main/NN/MLTheory/Proofs/Approximation/Universal/UniversalApproximationIEEE32Exec.lean)

The most visible theorem names there are:

```
import NN.MLTheory.Proofs.Approximation.Universal.UniversalApproximationIEEE32Exec

#check relu_universal_approximation_Icc_ieee32exec_minimal
#check relu_universal_approximation_Icc_ieee32exec_threeTerm
#check relu_universal_approximation_Icc_ieee32exec_twoTerm
```

That does not replace the `FP32` layer. It complements it: `FP32` is still the cleanest place to
package compositional rounding arguments, while the `IEEE32Exec` theorems support statements closer
to executable binary32 behavior itself.

# What A Full Paper Statement Still Has To Package

The remaining work is concrete theorem packaging. A paper-ready theorem has to quantify over:

- the real target function or real network being approximated;
- the scalar semantics used by the executable model (`FP32`, `IEEE32Exec`, or a runtime bridge);
- the classical approximation error `ε_appx`;
- the finite precision or rounding error `ε_fp32`;
- the interval, CROWN/LiRPA, or certificate error term `ε_verify`, if the statement is
  verifier side;
- the bridge hypotheses: finite values, no overflow, fixed reduction order, and any runtime/native
  agreement assumptions.

A paper-ready theorem usually has to spend three budgets: approximation error from the real model,
finite-precision error from the scalar semantics, and verification error from the enclosure method.
Those budgets should be named separately before they are added:

$$`\varepsilon_{\mathrm{total}}
=
\varepsilon_{\mathrm{appx}}
+
\varepsilon_{\mathrm{fp32}}
+
\varepsilon_{\mathrm{verify}}`

That is where the verification and floating-point chapters should be cited together.

A typical citation bundle for this combined topic is:

- `Floating-Point Semantics` for `FP32`, `IEEE32Exec`, and bridge theorems;
- `FP32 Soundness Notes` for the rounding model used in proofs and its theorem names;
- `Verification` for the interval / LiRPA / certificate side;
- [approximation proofs](https://github.com/lean-dojo/TorchLean/tree/main/NN/MLTheory/Proofs/Approximation/) for the universal-approximation
  theorems themselves.

## A practical citation map

When writing a paper, proposal, or technical note, a useful division is:

- classical approximation over reals:
  cite `relu_universal_approximation_Icc` and the classical UAT references;
- explicit `FP32` rounding on top of the approximation theorem:
  cite `relu_universal_approximation_Icc_fp32` plus the FP32 semantics and theorems;
- executable IEEE-754-flavored approximation statements:
  cite the `relu_universal_approximation_Icc_ieee32exec_*` family;
- interval or verifier side semantics:
  cite this bibliography together with `Verification` and the abstract-interpretation / LiRPA papers.

Related guide pages: *Floating-Point Semantics* (`FP32` vs `IEEE32Exec`), *FP32 Soundness Notes*
(transfer lemmas and theorem names), *Verification* (IBP/CROWN and certificates).

# References

- Hwang, Saad, et al. *Floating-Point Neural Networks Are Provably Robust Universal Approximators*.
  arXiv:2506.16065, CAV 2025. https://arxiv.org/abs/2506.16065
- Cybenko, G. *Approximation by superpositions of a sigmoidal function*. 1989.
- Hornik, Stinchcombe, and White. *Multilayer feedforward networks are universal approximators*.
  Neural Networks, 1989.
- Leshno et al. *Multilayer feedforward networks with a nonpolynomial activation function can
  approximate any function*. Neural Networks, 1993.
- Pinkus, A. *Approximation theory of the MLP model in neural networks*. Acta Numerica, 1999.
- Sakr et al. *Analytical Guarantees on Numerical Precision of Deep Neural Networks*. ICML 2017.
  [PMLR page](https://proceedings.mlr.press/v70/sakr17a.html)
- Gehr et al. *AI2: Safety and Robustness Certification of Neural Networks with Abstract
  Interpretation*. 2018.
- Singh et al. *An Abstract Domain for Certifying Neural Networks*. POPL 2019.
- Goldberg, D. *What Every Computer Scientist Should Know About Floating-Point Arithmetic*.
  [Oracle-hosted reprint](https://docs.oracle.com/cd/E19957-01/806-3568/ncg_goldberg.html)
- Higham, N. *Accuracy and Stability of Numerical Algorithms*.
  [SIAM book page](https://epubs.siam.org/doi/book/10.1137/1.9780898718027)
- IEEE Std 754-2019. Standard for Floating-Point Arithmetic.
  [IEEE 754-2019](https://standards.ieee.org/standard/754-2019/)
- ISO/IEC/IEEE 60559:2020. Floating-point arithmetic.
  [ISO/IEC/IEEE 60559:2020](https://standards.ieee.org/standard/60559-2020.html)
- Flocq project documentation. [Flocq](https://flocq.gitlabpages.inria.fr/)
- FloatSpec for Lean 4. [FloatSpec](https://reservoir.lean-lang.org/%40Beneficial-AI-Foundation/FloatSpec)
- Gappa, *Certifying floating-point implementations using Gappa*.
  [Gappa paper](https://arxiv.org/abs/0801.0523)
- CompCert commented development. [CompCert](https://compcert.org/doc/)
