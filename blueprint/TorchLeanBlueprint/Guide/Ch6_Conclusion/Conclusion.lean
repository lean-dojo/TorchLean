import VersoManual
import VersoBlueprint

open Verso.Genre Manual

#doc (Manual) "Conclusion" =>
%%%
tag := "conclusion"
%%%

TorchLean is still a research system, but the direction is clear. Modern ML systems increasingly
mix generated code, learned models, numerical kernels, exported graphs, and scientific claims. The
useful question is not only whether these systems run. It is whether we can say precisely what they
mean.

The guide has followed one design principle from start to finish:

> a model should be runnable as ML code, inspectable as an artifact, and connected to named
> semantics whenever a correctness, gradient, bound, or certificate claim is made.

That is the unifying point of the tensor API, runtime, graph IR, floating-point layers,
verification checkers, examples, and widgets.

The long-term value is especially clear in scientific computing. Neural operators, PINNs,
controllers, 3D perception systems, and RL agents are not isolated classifiers. They sit inside
larger mathematical and engineering arguments. A verified ML stack should help those arguments name
their assumptions, check their artifacts, and preserve meaning across runtimes.

# The Working Pattern

The main workflow has five steps.

1. Write the model or artifact in a typed form: tensors, parameters, graph nodes, certificates,
   datasets, or runtime logs.
2. Choose the semantics relevant to the question: real tensors, `FP32`, `IEEE32Exec`, graph
   denotation, autograd tapes, verifier domains, or external artifact formats.
3. Run the computation or importer: training loop, CUDA path, PyTorch bridge, graph compiler,
   verifier pass, or external producer.
4. Inspect the result with ordinary outputs, widgets, logs, graph views, or certificate reports.
5. Cite the strongest available support: executable run, Lean checker, theorem, or explicit
  assumption.

This is why TorchLean is organized as a framework rather than a collection of isolated examples.
The same model may appear as user-facing API code, an eager runtime computation, a graph artifact,
a verification input, and a theorem target.

# Levels Of Support

The guide uses different words for different levels of support:

- *Execution*: a command runs and produces concrete tensors, logs, samples, bounds, or reports.
- *Inspection*: the produced object is visible enough to debug, through printed output, widgets,
  graph views, training curves, or certificate diagnostics.
- *Checking*: Lean parses or recomputes an artifact and accepts or rejects a stated contract.
- *Theorem support*: a Lean theorem connects the checked contract to the intended semantic claim.
- *External assumption*: a Python exporter, CUDA kernel, dataset converter, solver, or native
  library remains outside the proved fragment and is named as part of the claim.

This distinction is practical. A loss curve can show that a run behaved sensibly. A certificate
checker can validate a bound artifact. A theorem can state why an accepted artifact implies a
semantic property. These are related, but they are not interchangeable.

# What The Chapters Contribute

Each part of the guide contributes one layer of that pattern:

- *Introduction* explains the problem TorchLean is built to address.
- *Building Models* gives the typed tensor, model, data, and training surface.
- *Runtime, Autograd, and Interop* explains how models execute, train, and cross the PyTorch
  boundary.
- *Semantics and Graphs* gives a shared IR and semantic target for runtimes, exporters, widgets, and
  verifiers.
- *Floating Point and Native Boundaries* separates real specifications, proof-side float models,
  executable IEEE behavior, CUDA, and external tools.
- *Verification and Certificates* explains how bounds, autograd proofs, runtime approximation,
  scientific certificates, and imported artifacts become checked claims.
- *Examples and Applications* shows the system in use: model zoo runs, generative models, RL
  examples, widgets, BugZoo case studies, and command-line workflows.

Together they give a path from ordinary model code to a precise statement about what has been run,
checked, proved, or assumed.

# How To State A TorchLean Result

When writing about TorchLean, the strongest statements have a simple form:

- name the object: model, graph, tensor, tape, certificate, parameter store, dataset, or external
  artifact;
- name the semantics: real, `FP32`, `IEEE32Exec`, graph denotation, verifier abstraction, or runtime
  execution path;
- name the support: command output, widget inspection, checker acceptance, theorem, or assumption;
- state the property: shape agreement, gradient correctness, robustness margin, approximation
  bound, certificate validity, or runtime behavior.

For example, a model-zoo run can establish that a particular training script executes and writes a
loss trace. A verifier command can establish that a graph artifact passes a bound checker. A
soundness theorem can establish that accepted bounds enclose the graph semantics for a supported
fragment. A CUDA or Python path can be used, but the external part remains part of the statement
unless a bridge theorem or replay checker covers it.

# The Extension Rule

The same rule applies when extending the project. A new feature is strongest when it adds one clear
bridge:

- a new model should expose its data boundary, parameter shapes, training loop, and logs;
- a new operator should name its spec and runtime behavior;
- a new graph pass should state what semantics it preserves or approximates;
- a new verifier should identify the abstract domain, checker, and soundness theorem surface;
- a new external integration should specify the artifact format and producer/checker contract;
- a new widget should render an existing Lean object rather than define a new semantic meaning.

That rule keeps TorchLean extensible without making the system vague. Every addition should make at
least one object easier to run, inspect, check, or prove.

# Closing

TorchLean is useful because it keeps two activities connected: building neural-network systems and
stating mathematical claims about them. The code can still look like ML code. The examples can still
train, log, save, load, and call native or external tools. The difference is that the important
objects have names in Lean, and claims about those objects can be tied to checkers, theorems, or
explicit assumptions.

The standard the project should preserve is simple: identify the object, identify the semantics,
and say what has been run, checked, proved, or assumed. If TorchLean makes that habit easier for
neural networks, then it has done something useful.
