import VersoManual
import VersoBlueprint

open Verso.Genre Manual

#doc (Manual) "Two-Stage Verification Workflows" =>
%%%
tag := "twostage"
%%%

Two-stage workflows are for cases where search is too expensive or too specialized to do inside
Lean, but the final claim should still be checked as a small artifact.

The first stage is a producer: Python, Julia, α,β-CROWN, or a control script trains a model,
searches for a certificate, or partitions a domain. The second stage is a Lean checker: it parses
the artifact, checks the supported conditions, and records exactly what follows.

TorchLean does not vendor the Two-Stage / α,β-CROWN repository. The core TorchLean library and
`lake build` do not require that Python stack. If you want to run the external producer workflow,
clone the Two-Stage repository separately. TorchLean no longer carries that producer as a submodule.

The workflow has two stages, each with a few concrete steps:

```
Stage 1: external search or training
  PyTorch / Julia / α,β-CROWN
  -> checkpoint, bounds, leaf certificate, controller candidate

Stage 2: Lean post-check
  parse artifact
  check shape and schema
  replay supported computation
  validate certificate predicate
  state theorem or producer hypothesis
```

In repository terms, the stages are:

- Python training, Julia simulation, or α,β-CROWN produces a checkpoint, JSON artifact, or bound
  report.
- TorchLean parses the artifact into typed Lean data.
- A TorchLean checker replays the supported computation, checks structure, or records an explicit
  producer hypothesis.
- The final report says which part was proved in Lean, which part was recomputed, and which part
  remains external evidence.

# Scope

Below is how TorchLean connects Python training and α,β-CROWN runs to Lean replay and
checking. For flags and solver options on the Python side, follow the Two-Stage repository; those
details change independently of TorchLean.

To run a verifier tool, use `lake exe verify -- list` and the tool names in
[NN.Verification.CLI API](https://github.com/lean-dojo/TorchLean/blob/main/NN/Verification/CLI.lean), then read the sections here for
what Lean actually checks.

# What TorchLean Provides

TorchLean supports the Python-first verification pattern: an external producer writes an artifact,
then Lean parses, replays, or structurally checks it. The current Two Stage checker handles a small
leaf artifact JSON format exported from α,β-CROWN:
`abcrown-leaf` (see the certificate guide for what this does and does not check).

# Quick Run (From TorchLean)

The Two Stage checker is registered in the unified verification CLI, so the easiest place to
start is:

```
lake exe verify -- list
```

For debugging certificate export and import, the leaf checker is the usual starting point:

```
lake exe verify -- abcrown-leaf
```

# Minimal Setup (One-Time)

Clone the external producer only if you want to generate fresh α,β-CROWN / Two-Stage artifacts:

- `git clone https://github.com/Verified-Intelligence/Two-Stage_Neural_Controller_Training.git Two-Stage_Neural_Controller_Training`

The Python verifier dependencies are managed by the external Two-Stage / α,β-CROWN repositories;
TorchLean does not try to own that environment. Local α,β-CROWN runs follow those repositories'
instructions; artifact export is described in *Certificates*.

# What Is Actually Checked

The Lean pipelines re-evaluate models against TorchLean's semantics, including the chosen scalar
domain such as `IEEE32Exec`, and run TorchLean's bound propagation and checkers on the resulting
graphs. The `abcrown-leaf` JSON artifacts are structural leaf artifacts today: TorchLean can parse
them and check their declared shape and metadata; stronger certificates would additionally replay
or prove the bound computation.

For branch-and-bound verification, the theorem pattern is:

$$`B=\bigcup_{\ell=1}^m B_\ell,
\qquad
\forall \ell,\;\operatorname{Safe}(B_\ell)
\quad\Longrightarrow\quad
\operatorname{Safe}(B).`

For neural-controller workflows, the certificate target often has a Lyapunov shape:

$$`V(x)\ge 0,\qquad
\dot V(x)=\nabla V(x)\cdot f(x,u_\theta(x))\le -\alpha\|x\|^2.`

Certificates checked against the shared IR semantics: see the `RealCert`-style artifacts in
*Certificates*.

For the Lean declarations, open the [verification CLI](https://github.com/lean-dojo/TorchLean/blob/main/NN/Verification/CLI.lean) and the
[α,β-CROWN leaf checker](https://github.com/lean-dojo/TorchLean/blob/main/NN/Verification/Cert/AbCrownLeafCert.lean). The external producer
checkout is separate, and sample assets are kept in `NN/Examples/Verification/AbCrown/`.

# Reproducibility Notes

Two stage workflows depend on both the Lean checker and the external producer environment. The
short checklist that helps most is to pin the external producer commit, record the Python
environment used for the external verifier, and record the scalar backend selected for the Lean
replay.
That makes it much easier to compare a Lean replay against the original Python run later.

Broader verifier map: *Verification*. Float32 execution details: *Floating-Point Semantics*.

# References

- TorchLean paper (Two-Stage case study context): https://arxiv.org/abs/2602.22631
- α,β-CROWN codebase: https://github.com/Verified-Intelligence/alpha-beta-CROWN
- β-CROWN / α,β-CROWN paper: https://arxiv.org/abs/2103.06624
