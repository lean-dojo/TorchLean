/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Proofs.RuntimeApprox.Graph.NumericalCertificate.Certificate
public import NN.Proofs.RuntimeApprox.Graph.NumericalCertificate.Contracts
public import NN.Proofs.RuntimeApprox.Graph.NumericalCertificate.Enclosure

/-!
# Numerical certificates for TorchLean graphs

This module joins three existing parts of TorchLean without introducing another graph or another
floating-point representation:

* `NN.IR.Graph` remains the program being analysed;
* FloatLib supplies executable, outward-rounded binary32 intervals;
* `NN.Backend.KernelPlanAudit` records the selected kernel capsules.

A raw certificate is proof-free data that an application may construct or decode using its own
artifact format; this module does not prescribe a JSON schema. `check` does not trust its node
ranges. It reconstructs the canonical range trace from the graph and source
assumptions, checks every interval for finite ordered endpoints, replans the graph under the named
backend profile, and compares the result with the raw artifact. Successful checking returns a
`RegistryCheckedCertificate`, whose node ranges carry finite-endpoint and ordering proofs. This
executable check does not by itself prove enclosure of the exact-real graph denotation; that
evidence is the separate `ProvedRealEnclosure` value used by `RangeCheckedExecution.error_trace`.

The range trace deliberately starts with operations whose enclosure is already provided by the
sound interval core. Unsupported operations fail with the node id and operation name. They are
not assigned `[-inf,+inf]`, because that would turn a missing numerical theorem into an apparently
successful certificate.

The numerical conventions follow IEEE Std 754-2019. Outward-rounded interval propagation follows
IEEE Std 1788-2015 and the standard inclusion principle for interval arithmetic. For the error
model that composes local bounds across forward and reverse graphs, see `ForwardApprox.lean` and
`BackwardApprox.lean`; the organization follows the local-error/global-error distinction in
N. J. Higham, *Accuracy and Stability of Numerical Algorithms*, 2nd ed., 2002.
-/
