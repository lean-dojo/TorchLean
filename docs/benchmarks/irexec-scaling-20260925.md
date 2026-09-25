IRExec lowering scaling, 2026-09-25
=================================

Baseline: `0f845313599795032960d487708c309e1fb2f40e`, Lean 4.34.0, Linux x86_64.
All builds ran in the private tmpfs clone, reusing the baseline dependency symlink and copied
build cache. The baseline executable was built and measured before changing runtime sources.

`ForwardData` now wraps nodes indexed by reversed shape prefixes. Each new prefix shares its
predecessor's tail. The compiled lowering loop checks parents through a proof-carrying array
and constructs the chronological output list once at completion. Kernel-checked equalities
connect array lookup, lowering, and evaluation to their original typed meanings, including
every failure result. Operation closures and summation order are unchanged.

The generated C for both generic and Float-specialized lowering loops contains no list
reverse or append call. The dispatcher erases its chronological context argument. Its
`nospecialize` and `inline_if_reduce` attributes are both necessary: specialization otherwise
retains unused shape arguments and reconstructs chronological prefixes at each node.

Each graph contains one `[4]` input and the indicated number of ReLUs. Each sample runs in
a fresh native process. Three repetitions produce the medians below; all 53 raw samples,
including two additional execution checks, are in
[the TSV](irexec-scaling-20260925.tsv). Lowering and evaluation are forced into `IO.Ref`s
before their ending timestamps. Disposal clears the final graph reference after inspection
returns. Graph creation, output hashing, and inspection are outside the corresponding timers.
Peak RSS is the whole-process `/usr/bin/time` measurement, including runtime overhead;
the TSV also records live `/proc/self/status` RSS immediately after lowering. Memory units
are KiB in the TSV and MiB here.

| ReLUs | Lower before, ms | Lower after, ms | Dispose before, ms | Dispose after, ms | Peak before, MiB | Peak after, MiB |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1,000 | 9.070 | 0.569 | 3.007 | 0.034 | 79.51 | 66.17 |
| 2,000 | 35.350 | 0.738 | 12.155 | 0.066 | 127.51 | 68.25 |
| 4,000 | 145.619 | 1.145 | 49.918 | 0.136 | 311.79 | 67.48 |
| 8,000 | 576.223 | 1.905 | 199.828 | 0.278 | 1,046.05 | 68.26 |
| 16,000 | 2,352.590 | 4.180 | 826.625 | 0.763 | 3,987.46 | 77.20 |
| 32,000 | — | 7.626 | — | 1.715 | — | 81.75 |
| 64,000 | — | 14.723 | — | 3.981 | — | 91.39 |

Construction and actual destruction were also measured without tensor execution:

| ReLUs | Lower, ms | Dispose, ms | Peak, MiB |
| ---: | ---: | ---: | ---: |
| 64,000 | 14.929 | 2.888 | 85.50 |
| 128,000 | 30.600 | 6.955 | 100.21 |
| 256,000 | 61.488 | 15.865 | 137.50 |
| 512,000 | 121.683 | 32.832 | 202.25 |
| 1,024,000 | 247.650 | 65.209 | 338.21 |

The final large doublings approach twice the construction and disposal time. The empirical
growth guard passes both after runs and rejects the old implementation's lowering, disposal,
live RSS, and peak RSS growth. This guard is a regression check, not a formal complexity proof.

Validation commands, run from the private clone:

```bash
lake -Kcuda=false build \
  NN.Runtime.Autograd.IRExec.Correctness.SemanticEquivalence \
  NN.Tests.Runtime.Floats.TorchLeanIRExecEquivCheck

irexec_work=/dev/shm/torchlean-finish-20260925-150455/irexec-work
python3 scripts/checks/irexec_scaling.py \
  --sizes 1000 2000 4000 8000 16000 32000 64000 --repeats 3 \
  --assert-linear --output "$irexec_work/after.json"
python3 scripts/checks/irexec_scaling.py --binary .lake/build/bin/irexec_scaling \
  --lower-only --sizes 64000 128000 256000 512000 1024000 --repeats 3 \
  --assert-linear --output "$irexec_work/after-large.json"
python3 scripts/checks/repo_lint.py --fail-on-warn
git diff --check
```

All passed. The runner creates and removes its own temporary Lake executable configuration;
the root lakefile and global test registry are unchanged. Semantic checks compare every
intermediate Float bit against `NN.IR.Graph.denoteAll`, including heterogeneous shapes,
shared parents, typed-pack output, resumed prefixes, deterministic random nodes, payloads,
and exact malformed-context diagnostics. All 15 baseline execution hashes match the after
runs. The preserved pre-change executable was measured with:

```bash
for repeat in 1 2 3; do
  for n in 1000 2000 4000 8000 16000; do
    /usr/bin/time -f 'peak_kb=%M user_s=%U system_s=%S elapsed_s=%e' \
      "$irexec_work/irexec-scaling-baseline" "$n"
  done
done
```

The existing `Tests.Floats.TorchLeanIRExecEquivCheck.run` also passed as a native executable
(`torchlean_ir_exec_equiv_check: ok`). Its temporary driver imported the existing test module
and defined `public def main : IO Unit := Tests.Floats.TorchLeanIRExecEquivCheck.run`.
With a temporary Lake target named `irexec_existing`, validation used
`lake -f "$config" -Kcuda=false build irexec_existing` followed by
`.lake/build/bin/irexec_existing`.

An axiom audit ran `lake -Kcuda=false env lean -DwarningAsError=true "$irexec_work/Axioms.lean"`.
The file imported `NN.Runtime.Autograd.IRExec.Correctness.SemanticEquivalence` and printed
axioms for `Internal.ShapeArray.mkIdx_eq`, `ForwardData.eval_snoc`,
`Internal.ReverseData.evalArray_eq`, `Internal.buildFromArray_eq`,
`Internal.buildFrom_eq_buildFromWithArray`, and `denoteAll_eq_of_lowerToForwardGraph`
in namespace `Runtime.Autograd.IRExec`. Only Lean's existing `propext`, `Classical.choice`,
and `Quot.sound` occur; there are no new axioms or proof shortcuts.

There are two compatibility/performance limits. `ForwardData.nil`, `snoc`, and `eval` retain
their typed interfaces and equations, but `ForwardData` is now a structure, so external code
using its old inductive recursor needs adaptation. Generic operation dispatch also makes this
tiny-tensor evaluation workload slower: 16k-node evaluation is 7.925 ms versus 4.323 ms before.
It remains linear and bit-identical. Full execution passed at 128k and 256k nodes, the latter
taking 129.172 ms with 171.13 MiB peak RSS. Evaluation still uses native recursion and therefore
depends on stack capacity (10,240 KiB here); million-node measurements cover lowering and
disposal. Parent selection is constant time apart from shape comparison, and tensor operation
costs remain unchanged. Initializing from an existing public prefix takes linear time in that
prefix. Global test-suite registration remains for integration.
