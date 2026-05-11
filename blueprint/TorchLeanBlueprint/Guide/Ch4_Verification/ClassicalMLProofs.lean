import VersoManual
import VersoBlueprint

open Verso.Genre Manual

#doc (Manual) "Classical and Structural ML Proofs" =>
%%%
tag := "classical-ml-proofs"
%%%

Not every proof in TorchLean is about a modern runtime boundary. Some declarations formalize
classical ML theory: Hopfield energy, ReLU approximation components, and state space scan
or causality facts. These are valuable because they show TorchLean can host mathematical ML theory
directly, not only verify artifacts emitted by a runtime.

The common shape is small but powerful:

- *Hopfield networks*: finite Boolean states, weights, thresholds, and an energy function, with
  asynchronous updates that do not increase energy under symmetry assumptions.
- *ReLU approximation*: local gadgets, compact domains, and MLP bridges, so approximation
  components can be reused in later network theorems.
- *State space scans*: recurrent scan equations over lists, with prefix theorems saying future
  inputs do not affect past outputs.

# Hopfield Networks

The Hopfield proof island formalizes the classical energy argument. A state is a finite Boolean spin
assignment, parameters contain weights and thresholds, and the energy is the scalar quantity that
should not increase under an asynchronous update when the weights are symmetric and the diagonal is
zero.

The central assumptions are named in the
[Hopfield energy API](https://github.com/lean-dojo/TorchLean/blob/main/NN/MLTheory/Proofs/Hopfield/Energy.lean):

$$`\operatorname{SymmetricW}(p) :=
\forall i\,j,\; p.W(i,j)=p.W(j,i),
\qquad
\operatorname{DiagonalZero}(p) :=
\forall i,\; p.W(i,i)=0`

The energy has the classical quadratic form:

$$`E(x)=-\frac12 x^\top W x+\theta^\top x.`

The theorem `energy_updateAt_le` is the local statement:

> Updating one coordinate according to the Hopfield rule does not increase energy under symmetric
> weights and zero diagonal.

In symbols:

$$`E(x^{t+1})\le E(x^t),`

and away from threshold ties, when the state actually changes,

$$`E(x^{t+1})<E(x^t).`

The stronger theorem `energy_updateAt_lt_of_change_of_ne` says that when the coordinate really
changes and the net input is not exactly at threshold, the energy strictly decreases. The dynamics
file lifts this to trajectories through `energy_seqStates_succ_le` and
`energy_seqStates_le_start`.

The convergence proof then uses finiteness: a strictly decreasing energy path cannot cycle forever
through nontrivial state changes. The theorems `cycleUpdate_no_nontrivial_cycles`,
`cycleUpdate_exists_fixedpoint_le_card`, and `cycleUpdate_exists_fixedpoint_le_pow` are the checked
versions of that classical argument.

This is not a complete theory of all associative memory models. It is a named Hopfield vocabulary:
states, energy, update assumptions, progress lemmas, and finite fixed point style theorems that
later extensions can build on.

# ReLU Approximation Bridges

The ReLU approximation bridge is a library of reusable components, not a standalone model claim.
The [ReLU multiplication approximation API](https://github.com/lean-dojo/TorchLean/blob/main/NN/MLTheory/Proofs/ReLU/Approx/ReLUMulApprox.lean)
records a small network shaped approximation component for multiplication. The
[compact set approximation API](https://github.com/lean-dojo/TorchLean/blob/main/NN/MLTheory/Proofs/ReLU/Approximation/CompactSet.lean) gives
the language for approximation on compact domains. The
[ReLU MLP bridge API](https://github.com/lean-dojo/TorchLean/blob/main/NN/MLTheory/Proofs/ReLU/Bridge/ReLUMlpBridge.lean) connects those pieces
back to MLP style objects.

The theory shape is:

$$`\text{local ReLU gadget theorem}
+ \text{compact-domain hypotheses}
+ \text{bridge from gadget notation to MLP notation}
\Longrightarrow
\text{reusable approximation fact for later model proofs}`

This proof component is neither a runtime test nor a full model theorem. It is a reusable
mathematical part, and larger formalizations often depend on these quiet lemmas.

# State Space and Mamba Causality

State space models replace attention with recurrent scan structure, so the theorem we care about is
causality. If two input sequences agree on a prefix, then the produced outputs should agree on that
prefix. Future tokens should not affect past outputs.

The recurrence shape is:

$$`h_{t+1}=A_t h_t+B_t x_t,\qquad
y_t=C_t h_t+D_t x_t.`

The [state space scan API](https://github.com/lean-dojo/TorchLean/blob/main/NN/MLTheory/Proofs/StateSpace/Scan.lean) proves append theorems for
scalar affine scans and diagonal selective scans:

$$`\operatorname{outputs}\!\left(\operatorname{run}(prefix \mathbin{++} suffix)\right)
\text{ restricted to the prefix }
=
\operatorname{outputs}\!\left(\operatorname{run}(prefix)\right)`

Equivalently:

$$`x_{0:k}=x'_{0:k}
\quad\Longrightarrow\quad
y_{0:k}=y'_{0:k}.`

The [Mamba causality API](https://github.com/lean-dojo/TorchLean/blob/main/NN/MLTheory/Proofs/StateSpace/MambaCausality.lean) specializes that
idea to Mamba style computations. The theorem names are intentionally direct:
`diagonalS4_runList_append_outputs_prefix`, `compactMamba_runList_append_outputs_prefix`,
`selectiveMamba_runListAux_append_outputs_prefix`, and
`selectiveMamba_runList_append_outputs_prefix`.

Sequence model verification is not only about attention. Attention has masks, KV caches, and
positional encodings. State space models have scan order, recurrent state, and causality
conventions. The Mamba causality theorems give those concerns a place in the formal layer.

# What Carries Forward

This area provides reusable proof components:

- Hopfield theorems seed energy and convergence arguments.
- ReLU approximation lemmas seed MLP bridge arguments.
- State space scan theorems seed causality arguments.

They do not claim that every model in the zoo is fully verified. They give us named mathematical
objects that future model theorems can reuse.
