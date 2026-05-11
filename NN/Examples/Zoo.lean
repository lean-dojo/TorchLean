/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Examples.Quickstart
public import NN.Examples.Quickstart.SimpleCnnTrain
public import NN.Examples.Quickstart.MinibatchMlpTrain
public import NN.Examples.Data.Loaders.Csv
public import NN.Examples.Data.Loaders.Npy
public import NN.Examples.Models
public import NN.Examples.Interop.PyTorch
public import NN.Examples.Advanced
public import NN.Examples.Verification
public import NN.Examples.Data.Loaders.Cifar10Images
public import NN.Examples.Quickstart.ResnetBasicblockTrain
public import NN.Verification.Cert.AbCrownLeafCert
public import NN.Verification.PINN.CLI
public import NN.Verification.PINN.Certificate
public import NN.Verification.PINN.Core
public import NN.Verification.PINN.DatasetCheck
public import NN.Verification.PINN.PdeAst
public import NN.Verification.PINN.PdeParse
public import NN.Verification.PINN.ResidualAffine
public import NN.Verification.ProofBackedCertificates
public import NN.Verification.Robustness.Digits
public import NN.Verification.ODE.Verify
public import NN.Examples.RL
public import NN.Examples.BugZoo.All

/-!
# `NN.Examples.Zoo`

Single umbrella for TorchLean examples.

The examples directory intentionally keeps only one root Lean entrypoint. Import this module when
you want to compile every maintained example module, including quickstarts, model examples,
interop examples, widgets, advanced tutorials, and verification examples.

Typical usage:

* Build the full example surface: `lake build NN.Examples.Zoo`
* Run model examples through the CLI:
  `lake exe torchlean mlp -- --steps 10`
* Run standalone executables directly when they intentionally define their own root `main`
  (for example `lake exe torchlean gpt_adder --steps 1000`).

The heavier examples remain in their subdirectories so users can still build or run one example at a
time. This umbrella avoids importing standalone executable roots that would collide on the global
Lean name `main`.
-/

@[expose] public section
