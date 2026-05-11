/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.MLTheory.LearningTheory.Stability.Core
public import NN.MLTheory.LearningTheory.Stability.Dynamics
public import NN.MLTheory.LearningTheory.Stability.RidgeRegression1D

/-!
# Algorithmic and dynamical stability

This entrypoint collects the stability chapter. We keep three pieces together because they are often
used in the same arguments, while still leaving each piece in its own source file:

- `Stability.Core` defines datasets, replace-one/remove-one perturbations, learning maps, losses,
  population error, empirical error, and standard algorithmic-stability notions.
- `Stability.Dynamics` gives the discrete-time dynamical-system vocabulary used by recurrent
  models, samplers, and stability diagnostics.
- `Stability.RidgeRegression1D` is the worked theorem development: a concrete, fully proved
  stability analysis for bounded one-dimensional ridge regression.

References:
- Bousquet and Elisseeff, "Stability and Generalization", JMLR 2002.
- Shalev-Shwartz et al., "Learnability, Stability and Uniform Convergence", JMLR 2010.
- Hardt, Recht, and Singer, "Train faster, generalize better: Stability of stochastic gradient
  descent", ICML 2016.
-/

@[expose] public section
