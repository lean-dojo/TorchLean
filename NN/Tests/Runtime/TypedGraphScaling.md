The TypedGraph execution change replaces repeated full-context appends with an indexed tensor
context and certified local node programs. Saved backward execution uses an array of gradients
and a shared chronological history of uniform contributions. It preserves each dense scalar
addition, including the complete contribution of a node with repeated parents. Removing an
output gradient consumes its array before reading the remaining length, avoiding a copy on every
reverse step.

`NN.Runtime.Autograd.TypedGraph` exports `compileChecked` and `Compiled`. Ordinary checked and
pure TypedGraph VJPs use the saved execution path through proved compiler simplifications.
TypedGraph sessions and the typed trainer also use it directly. `Compiled.toTape`,
`lowerToTapeChecked`, and the raw dense Tape interface remain available for compatibility.
The raw Tape backward interface still materializes full prefix contributions.

Measurements below were taken on 2026-09-25 with native CPU executables, Lean
`leanprover/lean4:v4.34.0`, and an Intel Xeon Platinum 8275CL at 3.00 GHz. The baseline is
`0f845313599795032960d487708c309e1fb2f40e`; the changed source is the commit containing this report.
Both use the unchanged dependency cache at
`/mnt/build/torchlean-cleanup-review-20260925/.lake/packages`. Project builds are private to
`/dev/shm/torchlean-finish-20260925-150455/typedgraph`; binaries and working logs are in the sibling
`typedgraph-work` directory.

The fixture is a chain of `Float` (binary64) `relu [4]` nodes with input
`[-2.0, -0.0, 0.5, 3.0]` and an all-ones output seed. Each size runs in a separate process.
Construction, checked forward/lowering, and backward are timed separately; output hashes force
observation of every primal and gradient tensor. The first changed backward retains the compiled
handle for reuse. A second backward consumes its final owner and therefore includes disposal.
These are individual measurements on a shared host, not medians or a formal complexity result.
The primary compiled series partially overlapped the CPU test suite. A second compiled series,
also retained in `TypedGraphScalingResults.txt`, measured 33.33 ms lowering and 48.37 ms retained
backward at 16k; its other points also show host timing variation.

All times in these tables are milliseconds. Peak memory is whole-process GNU `time -v` maximum
RSS in MiB, including graph construction.

| Nodes | Baseline construct | Changed construct | Baseline lower | Changed lower | Baseline backward | Changed retained backward | Baseline peak MiB | Changed peak MiB |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1,000 | 10.04 | 9.39 | 30.33 | 0.61 | 224.30 | 1.69 | 128.64 | 89.78 |
| 2,000 | 40.42 | 39.28 | 118.32 | 1.47 | 910.06 | 3.84 | 289.29 | 138.71 |
| 4,000 | 164.55 | 155.81 | 479.35 | 3.90 | 3,954.98 | 7.45 | 933.01 | 324.98 |
| 8,000 | 669.58 | 628.88 | 1,936.55 | 7.37 | 19,369.61 | 14.13 | 3,506.92 | 1,058.79 |
| 16,000 | 3,545.39 | 3,013.47 | 32,312.11 | 25.33 | 124,642.42 | 44.41 | 13,792.24 | 4,007.66 |

The ordinary `Torch.TypedGraph.vjpChecked` API is measured separately, including a fresh checked
forward evaluation and input-gradient extraction on every call. Its first call retains the
recorded graph; its final call includes releasing the graph.

| Nodes | Public checked VJP | Public final VJP + disposal | Compiled final backward + disposal | Public peak MiB |
| ---: | ---: | ---: | ---: | ---: |
| 1,000 | 2.22 | 5.24 | 4.94 | 89.75 |
| 2,000 | 5.19 | 17.42 | 16.64 | 137.31 |
| 4,000 | 11.85 | 64.94 | 59.27 | 324.30 |
| 8,000 | 23.63 | 226.76 | 255.63 | 1,065.48 |
| 16,000 | 46.07 | 889.08 | 953.07 | 4,015.28 |

Construction and final disposal remain quadratic: the existing graph representation retains
full shape-prefix lists. At 16k nodes the changed compiled process already holds 4,093,532 KiB
RSS after construction, before lowering; lowering reaches 4,099,384 KiB and retained backward
4,104,832 KiB. These phase readings come from `/proc/self/status`, whose accounting can differ
slightly from GNU `time`. Releasing the last graph owner must reclaim this metadata. Consequently,
the measured reusable execution improves markedly, but the whole graph lifecycle and its peak
memory are not linear.

The compression boundary is explicit. `Storage.decEq?` supplies a certified equality test for
native `Float`, `Float32`, and `UInt8`; other storage instances default to `none` and use exact
dense backward. Even with equality, varying or otherwise noncompressible uniform histories can
require quadratic replay. Custom nodes without certified local preparation or compact
contributions retain materializing adapters. There is no size cutoff, zero-VJP assumption,
reassociation, or general linear-time guarantee for custom graphs. JVP scaling is not covered.

The formal equalities preserve the complete checked error/result and dense backward meanings.
In particular, `compileChecked_asLegacy` identifies the original Tape and full primal context;
`compileChecked_backwardDenseFrom_eq_tape` identifies every output gradient for an arbitrary
seed pack. `lowerToTapeChecked_eq_array`, `GraphData.backpropCtx_eq_saved`, and
`TypedGraphWithData.vjpChecked_eq_compiled` justify the compiler substitutions. Analytic node
round-trip theorem statements remain unchanged. The audited equalities depend only on
`propext`, `Classical.choice`, and `Quot.sound`.

Full-context binary64 primal and gradient hashes match baseline at all five sizes and on repeated
backward calls:

| Nodes | Primal hash | Gradient hash |
| ---: | ---: | ---: |
| 1,000 | 18300553107339986344 | 13835234076802139560 |
| 2,000 | 2930834670921170080 | 5004742299325283488 |
| 4,000 | 13280056868529673168 | 12719358714922046416 |
| 8,000 | 3448001696160945712 | 378798550107952688 |
| 16,000 | 1788165612113097456 | 13819532016633377520 |

`TypedGraphScalingRegression` additionally compares FP32 tensor bits directly against the raw
Tape engine. Cases cover reverse-order branch cancellation, repeated parents, input and
intermediate seeds, signed zero, subnormals, NaNs, a disconnected singular inverse with zero
cotangent, a custom nonzero VJP at zero seed, validation failures, repeated backward, public
checked/pure VJPs, and frozen session leaves. The branch test distinguishes reverse accumulation
from creation order; the repeated-parent test distinguishes adding the complete local sum from
scattering separately rounded terms. The regression runs from the maintained
`Floats.AllAutogradTests` suite.

The interpreted regression, native regression, and full curated native CPU suite passed.
Native project compilation completed 5,912 jobs, followed by the public root object and a
3,043-job proof/downstream build. The curated suite includes typed training and buffer updates.
CUDA execution was skipped because this build uses `cuda=false`. The main branch's newer
FloatLib revision still needs its integration build; this checkout deliberately retains the
original package cache.

These are the validation commands used in the isolated checkout. The saved response files list
the exact native link inputs: rebuilt private project objects and reused package objects. The
native suite was linked from `NN/Tests/Suite.lean` with those inputs; this does not claim a
`lake build nn_tests_suite` run.

```bash
cd /dev/shm/torchlean-finish-20260925-150455/typedgraph
tg_work=/dev/shm/torchlean-finish-20260925-150455/typedgraph-work

lake -Kcuda=false build $(cat "$tg_work/native-targets.txt")
lake -Kcuda=false build +NN.Runtime.Autograd.TypedGraph:c.o.export
lake -Kcuda=false build \
  NN.Proofs.Autograd.Runtime.Link.Checked \
  NN.Proofs.Autograd.Runtime.Link.HigherOrder \
  NN.Proofs.Autograd.Runtime.Link.HigherOrderFDeriv \
  NN.Proofs.Autograd.Runtime.Link.GraphComposition \
  NN.Proofs.Autograd.Training.StepAlgebra \
  NN.Proofs.RuntimeApprox.Graph.ForwardApprox \
  NN.Tests.Runtime.TypedGraphScaling \
  NN.Tests.Runtime.TypedGraphScalingRegressionMain

lake env lean --run NN/Tests/Runtime/TypedGraphScalingRegressionMain.lean
lake env lean --root="$tg_work" "$tg_work/TypedGraphAxioms.lean"
lake env lean -c "$tg_work/after.c" NN/Tests/Runtime/TypedGraphScaling.lean
lake env lean -c "$tg_work/regression.c" NN/Tests/Runtime/TypedGraphScalingRegressionMain.lean
lake env lean -c "$tg_work/suite.c" NN/Tests/Suite.lean
lake env leanc -O3 -o "$tg_work/after" @"$tg_work/after.rsp"
lake env leanc -O3 -o "$tg_work/regression" @"$tg_work/regression.rsp"
lake env leanc -O3 -o "$tg_work/suite" @"$tg_work/suite.rsp"
"$tg_work/regression"
"$tg_work/suite"

for tg_n in 1000 2000 4000 8000 16000; do
  /usr/bin/time -v "$tg_work/baseline" "$tg_n" >"$tg_work/before-$tg_n.log" 2>&1
  /usr/bin/time -v "$tg_work/after" "$tg_n" >"$tg_work/final-compiled-$tg_n.log" 2>&1
  /usr/bin/time -v "$tg_work/after" "$tg_n" public >"$tg_work/final-public-$tg_n.log" 2>&1
done
```

The baseline executable was built before source changes and remains in `typedgraph-work`.
The current benchmark's `legacy` mode uses the current proved lowering substitution and the
raw Tape backward; it is not a reconstruction of the old baseline lowering performance.
`TypedGraphScalingResults.txt` preserves the raw baseline, primary compiled/public, secondary
compiled, regression, axiom-audit, and curated-suite logs. Full build logs, native response files,
target lists, and executables remain in `typedgraph-work` for integration inspection.
