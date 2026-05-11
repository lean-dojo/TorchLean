/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.MLTheory.LearningTheory.DifferentialPrivacy
public import NN.MLTheory.LearningTheory.Robustness
public import NN.MLTheory.LearningTheory.Stability

/-!
# Learning theory

This is the curated entrypoint for TorchLean's learning-theory layer. We group the files here around
three themes:

- robustness specifications and executable Float diagnostics;
- algorithmic stability definitions, dynamics stability, and a fully proved 1D ridge-regression
  stability theorem;
- differential privacy definitions and closure lemmas.

The files keep theorem statements explicit about their level. Specification files define
`Prop`-level notions; runtime files compute finite diagnostics; worked examples such as ridge
regression prove concrete bounds with explicit assumptions.

References:
- Bousquet and Elisseeff, "Stability and Generalization", JMLR 2002.
- Dwork and Roth, *The Algorithmic Foundations of Differential Privacy*, 2014.
- Goodfellow, Shlens, and Szegedy, "Explaining and Harnessing Adversarial Examples", ICLR 2015.
- Hardt, Recht, and Singer, "Train faster, generalize better", ICML 2016.
-/

@[expose] public section
