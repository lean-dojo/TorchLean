/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.MLTheory.CROWN.Proofs.GraphAlphaCrownTransferSoundness.Common
public import NN.MLTheory.CROWN.Proofs.GraphAlphaCrownTransferSoundness.Alpha
public import NN.MLTheory.CROWN.Proofs.GraphAlphaCrownTransferSoundness.Phased
public import NN.MLTheory.CROWN.Proofs.GraphAlphaCrownTransferSoundness.EndToEnd

/-!
# α-CROWN Graph Transfer Soundness

Soundness theorems for the concrete α-CROWN graph transfer rule over `ℝ`, without and with ReLU
phases.

The main results are:
- `AlphaCrownTransferSoundness.alphaCrown_transfer_sound`
- `AlphaCrownTransferSoundness.phasedCrown_transfer_sound`

These theorems show that the executable checker steps satisfy the abstract `CrownTransferSound`
interface used by the generic graph certificate soundness theorem.

`EndToEnd` discharges their `IBPEnclosesVals` hypothesis from the IBP soundness theorem and states
the fully composed enclosure corollaries
(`AlphaCrownTransferSoundness.alphaCrown_cert_encloses_semantics`,
`AlphaCrownTransferSoundness.phasedCrown_cert_encloses_semantics`).
-/
