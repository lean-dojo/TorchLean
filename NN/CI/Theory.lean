/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module -- shake: keep-all (These imports define the CI build coverage.)

import NN.MLTheory.API
import NN.MLTheory.CROWN.Extras.FP32
import NN.MLTheory.CROWN.Lyapunov.TwoStage.Execution
import NN.MLTheory.CROWN.Lyapunov.TwoStage.Run
import NN.MLTheory.CROWN.Operators.Trigonometric
import NN.MLTheory.CROWN.Graph.Proofs.LayerNormDerivativeEnclosure
import NN.MLTheory.CROWN.Proofs.Conv
import NN.MLTheory.CROWN.Proofs.ConvDerivatives
import NN.MLTheory.CROWN.Proofs.ConvEnclosure
import NN.MLTheory.CROWN.Proofs.GraphAlphaCrownTransferSoundness
import NN.MLTheory.CROWN.Proofs.Overview
import NN.MLTheory.CROWN.Proofs.LayerNormEnclosure
import NN.MLTheory.CROWN.Propagation.Backward
import NN.MLTheory.CROWN.Tactics.CertificateWorkflow
import NN.MLTheory.LearningTheory.Stability.RidgeRegression1D.IEEE32Exec
import NN.Proofs
import NN.Proofs.Autograd.Overview
import NN.MLTheory.CROWN.Graph.Engine.Enclosure
import NN.API.RL.Markov

/-!
# Additional Theory Modules

This target covers maintained theorem, tactic, and workflow modules that are not re-exported from
the supported `NN.Proofs` and `NN.MLTheory` surfaces.
-/

@[expose] public section
