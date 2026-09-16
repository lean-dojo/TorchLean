/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Verification.Cert.AbCrownLeafCert
public import NN.Verification.ODE.Verify
public import NN.Verification.PINN.CLI
public import NN.Verification.PINN.Certificate
public import NN.Verification.PINN.DatasetCheck
public import NN.Verification.Robustness.Digits
public import NN.Verification.Robustness.MarginCertCLI
public import NN.Verification.Splines.PiecewiseLinearCLI
public import NN.Verification.Builtin.CrownOpsWorkflow
public import NN.Verification.Builtin.IBPWorkflow
public import NN.Verification.Builtin.MlpTrainVerifyWorkflow
public import NN.Verification.Builtin.TransformerIBPWorkflow
public import NN.Verification.VNNComp.MnistFC

/-!
# Verification Examples

Runnable and theorem-backed examples for TorchLean's verification library. The imports include the
checkers used by the bundled LiRPA, robustness, spline, VNN-COMP, ODE, PINN, and alpha-beta-CROWN
artifacts, together with workflows whose models originate in TorchLean.
-/

@[expose] public section
