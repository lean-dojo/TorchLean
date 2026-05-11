# NN/Proofs

This directory contains TorchLean's proof library.

The proofs here connect:

* spec-layer definitions (`NN/Spec/*`),
* IR semantics (`NN/IR/*`),
* runtime execution models (`NN/Runtime/*`),
* and verification tooling (`NN/Verification/*`).

Entry points:

* `NN/Entrypoint/Proofs.lean` is the curated umbrella import.
* `NN/Proofs/Autograd/Overview.lean` provides orientation for the autograd proof stack.
