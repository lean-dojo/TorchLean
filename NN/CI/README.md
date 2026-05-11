# NN/CI

This directory holds proof and check targets that are meant for CI rather than the everyday
development loop.

Some proofs take long enough to elaborate that building them on every local edit would slow people
down. The umbrella modules here let CI build those checks explicitly, or on a schedule, while normal
development stays focused on the files being changed.

See:

* `NN/Runtime/Autograd/Compiled/IRExec/Correctness.lean` (runtime compiler correctness,
  including the semantic equivalence theorem)
* `NN/CI/All.lean` (CI umbrella for broad compile checks)
* `NN/CI/ComparatorAll.lean` (Comparator entrypoint; sandboxed checking for untrusted submissions)

If you run one of these locally and it appears to pause, Lean is often elaborating one large module
without intermediate progress output.
