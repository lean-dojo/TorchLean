import VersoManual

open Verso.Genre Manual

#doc (Manual) "External Tools and FFI" =>
%%%
tag := "external-tools-ffi"
%%%

TorchLean is not trying to make Lean do every job. Python can train models and load checkpoints.
Julia can run numerical search. Arb can produce high-precision interval evidence. CUDA can execute
kernels. External verifiers can optimize bounds. The important question is what comes back to Lean
and what Lean checks before using it.

The guiding pattern is producer and checker: external tools produce small artifacts; Lean checks
the parts that have a stated contract.

The pattern we use is:

1. Lean defines the semantic target or checker.
2. An external tool performs search, training, numeric enclosure, graph capture, plotting, or fast
   execution.
3. The tool returns a small artifact: JSON, raw bits, a typed buffer handle, a graph, weights, bounds,
   or a certificate.
4. Lean parses and validates the artifact against the contract it owns.
5. Any part that Lean did not check remains a named runtime or producer assumption.

CUDA is one instance of the broader pattern: an external producer may do heavy work, while Lean
checks the metadata, bits, certificate, or agreement condition that is small enough to validate.

# Ecosystem Boundary

The previous pages covered tensors, graphs, Float32, CUDA, and certificates. The missing bridge was
the ecosystem layer:

- how Lean launches external programs without making them part of the kernel;
- how Python bridges PyTorch checkpoints, graph capture, datasets, Gymnasium, and plotting;
- how Julia can produce numeric or spline/PINN artifacts that Lean checks later;
- how Arb through `python-flint` can produce high-precision interval evidence;
- how C/CUDA FFI differs from subprocess integration;
- how we decide what is proved, what is parsed, what is tested, and what is merely assumed.

That distinction matters because TorchLean is designed to work with numerical ecosystems. Lean
serves as the accountability layer; the other tools remain excellent at what they do, while the
boundary stays explicit.

# Two Kinds Of Boundary

TorchLean uses two main external boundaries.

The FFI path is direct and fast. Lean calls native symbols for CUDA buffers, kernels, cuBLAS, cuFFT,
allocation, and finalizers. This is the right path for runtime execution.

The subprocess path is slower but easier to audit. Lean launches Python, Julia, Arb, or a verifier,
then reads JSON, stdout, or a file. This is the right path for certificate producers, data
conversion, plotting, and external numeric search.

In practice:

- FFI is for runtime execution: CUDA buffers, kernels, BLAS/FFT calls, and fast tensor paths.
- Subprocesses are for producers: PyTorch export, Arb interval queries, Julia spline fitting,
  Gymnasium environments, external verifiers, dataset conversion, and plotting.

Both are useful. The question is which artifact returns to Lean and which contract Lean checks.

# Producer Roles

The common producer/checker roles are:

- PyTorch trains models, loads checkpoints, or captures graphs; Lean checks names, shapes, supported
  ops, and payload layout.
- Python data scripts convert and plot data; Lean checks dimensions, schemas, and tensor contracts.
- Julia produces numerical artifacts, splines, ODE/PINN data, or search results; Lean checks
  domains, coefficients, bounds, and certificate fields.
- Arb or `python-flint` produces high-precision intervals; Lean decodes exact rational bounds and
  can check stronger certificate formats when available.
- CUDA FFI produces buffers, bits, and values; Lean checks shape metadata and applies parity tests
  or runtime agreement assumptions.
- External verifiers produce bounds, slopes, leaves, or certificates; Lean replays or structurally
  checks the accepted artifact.

# Lean-Side Plumbing

The generic subprocess utilities live in `NN.Runtime.External`. They have a deliberately small job:
resolve an executable, check availability, run a command, capture stdout, and parse JSON.

```
import NN.Runtime.External

namespace Runtime.External.Process
#check resolveCmdFromEnv
#check isCmdAvailable
#check ensureCmdAvailable
#check runStdoutChecked
#check runJsonStdoutChecked
end Runtime.External.Process

namespace Runtime.External.Julia
#check resolveJuliaCmd
#check isAvailable
#check run
#check runJson
end Runtime.External.Julia
```

The interface is intentionally plain. Oracle wrappers share the same environment-variable
conventions, error shape, and JSON parsing path so readers can see where external execution enters
the trusted story.

# Python: PyTorch, Data, Gymnasium, And Producers

Python enters TorchLean in several different roles, which have different trust meanings.

For PyTorch interop, Python is the right loader and graph-capture tool. PyTorch checkpoints are
pickle/zip artifacts and PyTorch modules can contain Python object structure, so Lean should not
pretend to parse arbitrary `.pt` files directly. Instead, we generate small Python adapters that
emit TorchLean-readable JSON.

```
import NN.Runtime.PyTorch.Export.StateDict
import NN.Runtime.PyTorch.Export.TorchExport
import NN.Runtime.PyTorch.Import.Core
import NN.Runtime.PyTorch.Import.TorchExport

namespace Export.PyTorch.StateDict
#check generateJsonBridgeScript
end Export.PyTorch.StateDict

namespace Export.PyTorch.TorchExport
#check generateGraphBridgeScript
end Export.PyTorch.TorchExport

namespace Import.PyTorch
#check parseTensor
#check loadWeightsE
#check getTensorE
end Import.PyTorch

namespace Import.PyTorch.TorchExport
#check parseGraph
end Import.PyTorch.TorchExport
```

The decision here is conservative: Python may load the checkpoint or capture the graph, but Lean
checks the JSON shape, parameter names, supported op subset, and IR validators. Unsupported Python
operators fail closed. They do not get silently approximated as verified TorchLean operations.

Python also appears in:

- data preparation scripts such as tiny Shakespeare, TinyStories, CIFAR, and FNO Burgers helpers;
- plotting scripts, where Python renders outputs but does not prove anything about the model;
- Gymnasium RL environments, where Lean checks observation shape, action count, reward parsing, and
  transition records against a `Runtime.RL.Boundary.Contract`;
- external verification producers such as alpha,beta-CROWN / Two-Stage scripts that emit JSON
  certificates for Lean checks.

The same rule applies each time: Python can produce; Lean decides what part of the production is
accepted.

# Julia: Numeric Producers And Certificate Search

Julia is useful when the producing side wants high performance numerical code, differential
equation tooling, optimization, spline fitting, or GPU-heavy search. The wrapper is intentionally
thin and optional:

```
import NN.Runtime.External.Julia

namespace Runtime.External.Julia
#check resolveJuliaCmd
#check ensureAvailable
#check runJson
end Runtime.External.Julia
```

We resolve `TORCHLEAN_JULIA` when it is set, otherwise we use `julia` from `PATH`. Importing the
Lean wrapper does not require Julia to be installed; Julia is needed only when code actually runs the
IO action.

The trust rule is the same as for Python. A Julia script may fit a piecewise polynomial, search for
a candidate certificate, or produce PINN/spline residual data. Lean should then check the small
certificate data it needs: cell domains, polynomial coefficients, interval bounds, residual
inequalities, and shape conventions. Julia's optimizer, floating-point arithmetic, GPU scheduler,
and package stack remain external evidence unless Lean checks the returned certificate.

# Arb And python-flint

Arb is the example where external numerical strength is genuinely valuable. Through `python-flint`,
we can ask Arb/FLINT for rigorous ball-arithmetic enclosures at high precision. That is excellent
evidence, especially for transcendental functions or interval experiments, but it is still an
external oracle unless the returned certificate is independently checked in Lean.

```
import NN.Floats.Arb

namespace TorchLean.Floats.Arb
#check Query
#check MidRad10Exp
#check MidRad10Exp.toRatBounds
#check run
#check runExpr
#check runMLP
end TorchLean.Floats.Arb
```

The Arb boundary returns results with enough structure to become exact Lean data. The
`mid_rad_10exp` encoding represents a rational enclosure:

```
[(mid - rad) * 10^exp, (mid + rad) * 10^exp]
```

That lets us separate two statements:

- "Arb says this interval encloses the result" is an oracle statement.
- "This JSON payload decodes to these exact rational bounds" is something Lean can check.

For a stronger theorem, we need an additional Lean checker that proves the external enclosure
is valid for the specific expression, graph op, or certificate format. Until then, Arb is powerful
evidence, not Lean kernel proof.

# C And CUDA FFI

The C/CUDA FFI boundary sits below Python or Julia subprocesses. The native code does not
usually return a neat JSON certificate; it mutates buffers, launches kernels, and hands Lean opaque
external objects. That is why the CUDA contract is stricter about source maps, bit contracts, and
regression tests.

The useful declarations are the
[trusted CUDA bridge](https://github.com/lean-dojo/TorchLean/blob/main/NN/Runtime/Autograd/Engine/Cuda/Trusted.lean),
[buffer surface](https://github.com/lean-dojo/TorchLean/blob/main/NN/Runtime/Autograd/Engine/Cuda/Buffer.lean),
[native source map](https://github.com/lean-dojo/TorchLean/blob/main/NN/Runtime/Autograd/Engine/Cuda/NativeSources.lean),
[float32 contract](https://github.com/lean-dojo/TorchLean/blob/main/NN/Runtime/Autograd/Engine/Cuda/Float32Contract.lean), and
[kernel specs](https://github.com/lean-dojo/TorchLean/blob/main/NN/Runtime/Autograd/Engine/Cuda/KernelSpec.lean).

The reason we made `NativeSources` explicit is practical: when a Lean file says it calls
`flashAttentionFwd`, `bmm`, or an FFT/FNO kernel, the native source should be easy to find without
guessing which `.cu` file owns it.

# What Counts As Connected?

There are several levels of "connected to Lean", and they should not be confused.

- *Launched from Lean*: Lean starts a process or calls an FFI symbol. This is interface work.
- *Parsed by Lean*: Lean successfully reads JSON, raw bits, or a buffer handle. This checks format.
- *Shape-checked by Lean*: tensors, graphs, or observations match declared shapes and supported ops.
- *Replay-checked by Lean*: Lean recomputes the artifact's local condition, such as a certificate
  predicate or graph validator.
- *Proved in Lean*: a theorem in Lean connects the checked artifact to a mathematical claim.

Most external workflows should aim for replay-checked artifacts. Full proof is better when feasible,
but a replay checker is already much stronger than treating external stdout as truth.

# Next Improvements

This ecosystem boundary is usable, and the next improvements are concrete:

- More external producers should emit small certificate formats rather than logs or ad hoc JSON.
- JSON parsers should report precise context: field name, expected shape, actual shape, and
  supported op subset.
- New external tools should update the trust-boundary docs in the same commit.

The principle does not change: external tools can be powerful, and Lean is where the returned
artifact is checked against the claim we are willing to state.

# Where To Read Next

- *GPU and CUDA Boundaries* for the native FFI version of this pattern.
- *PyTorch Roundtrip* for Python graph and weight artifacts.
- *Verification Certificates* and *Two-Stage Workflows* for external producer / Lean checker
  verification.
- `TRUST_BOUNDARIES.md` for the current inventory of axioms, FFI code,
  oracle wrappers, and external numeric producers.
