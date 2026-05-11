/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.MLTheory.LearningTheory.DifferentialPrivacy.Core

/-!
# Differential privacy

This entrypoint collects TorchLean's differential-privacy vocabulary. We keep the event-wise
definition and closure lemmas in `DifferentialPrivacy.Core` because the public API is deliberately
small: a randomized mechanism, an adjacency relation, `(ε, δ)`-DP, pure DP, monotonicity in `δ`,
and measurable post-processing.

The module is phrased using mathlib probability measures, so downstream users can instantiate the
same definitions for discrete mechanisms, continuous mechanisms, or randomized training procedures
without changing the statement of privacy.

References:
- Dwork, McSherry, Nissim, and Smith, "Calibrating Noise to Sensitivity in Private Data Analysis",
  TCC 2006.
- Dwork and Roth, *The Algorithmic Foundations of Differential Privacy*, 2014.
-/

@[expose] public section
