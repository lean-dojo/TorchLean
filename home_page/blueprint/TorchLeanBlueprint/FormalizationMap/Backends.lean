import Verso
import VersoManual
import VersoBlueprint
import TorchLean.NN.Backend.Graph
import TorchLean.Verify.LibTorch.TreeSum
import TorchLean.NN.Trainer.Session
import TorchLean.NN.Trainer.Scheduler
import TorchLean.NN.PyTorch.Export.IR
import TorchLean.NN.IR.Loaded
import TorchLean.NN.PyTorch.Import.State
import TorchLean.NN.Loss
import TorchLean.NN.Ops.Pure
import TorchLean.NN.Ops.Tensor
import TorchLean.NN.Session.Random
import TorchLean.NN.ScalarTrainer.Loss
import TorchLean.Tensor.Op.Pointwise.Real
import TorchLean.Verify.Trainer

open Verso.Genre
open Verso.Genre.Manual
open Informal

open FloatLib.Floats (ExecFloat)
open FloatLib.Floats.Formats.BinaryInterchange (Model FloatFormat)

#doc (Manual) "Backends and Training" =>

Backend selection is a planning step over declared data. A capsule names the registered operator
with its attributes, provider, device, trust level, reduction policy, and the evidence behind its
four contract claims. The contract check turns a plan audit into an accept or reject decision under
an assurance policy, and eager execution runs the accepted capsule with an implementation of the
same identity. Nothing in this group proves a kernel correct: its nodes record what is checked and
what is assumed.

Selecting CUDA matmul, for example, involves a shape obligation, a row-major layout obligation, and
numerical obligations for forward and backward. Acceptance explains why this provider is allowed to
run, but the native result still depends on the implementation satisfying its recorded contract.
The training and import nodes below describe how values reach that boundary.

:::group "backend_selection"
Kernel contracts, providers, and dispatch.
:::

:::definition "backend_capsule_contracts" (parent := "backend_selection") (lean := "TorchLean.Backend.KernelCapsule")
A kernel capsule records a registered operator (its wire record, attributes included), provider,
device, trust level (`checked` or `trustedExternal`), forward and VJP support, four contract
descriptors (shape, layout, forward value, VJP), and a {uses "backend_numerical_policy"}[numerical
policy]. Each descriptor pairs a structured claim with its evidence: a shared definition, a runtime
guard, a test suite, a named trusted boundary, or `notApplicable`. A capsule states the contract
expected from an implementation, but it doesn't prove that implementation. A `KernelContract` holds
the rest of the record without the operator, provider, and device, and `KernelContract.capsule`
builds its capsule for one operator.
:::

:::definition "backend_numerical_policy" (parent := "backend_selection") (lean := "TorchLean.Backend.NumericalPolicy")
A capsule's numerical policy is its reduction order: `fixedLeft` for the left fold used by the
canonical tensor semantics, `fixedLeftOrExact` for that fold unless the checked arithmetic selects
exact replay, `implementationDefined` for native and library accumulations, or `notApplicable`.
Numerical certificates read this field so that a fixed-left range trace isn't reused for an
implementation-defined schedule. Rounding mode, subnormal handling, and multiply-add contraction
aren't recorded.

Even for the same matrix dimensions the distinction matters: changing the accumulation tree changes
the sequence of rounded additions. A shape proof can't substitute for the missing numerical-policy
match.
:::

:::definition "backend_contract_check" (parent := "backend_selection") (lean := "TorchLean.Backend.KernelPlan.checkContracts")
The contract check reads every obligation of a plan audit and rejects those whose evidence the
assurance policy doesn't accept. Shared definitions, runtime guards, test suites, and
`notApplicable` pass under the `checked` policy, and only the `external` policy admits a trusted
boundary. The check returns a `Contract.Check`: either `accepted` or `rejected` with the failing
obligation reports.
:::

:::definition "backend_accepted_kernel" (parent := "backend_selection") (lean := "TorchLean.Backend.AcceptedKernel")
An accepted kernel is a planned operation whose
{uses "backend_capsule_contracts"}[capsule] passed the complete kernel-policy gate, including the
{uses "backend_contract_check"}[evidence check]. Its proof field states
`PlannedKernel.acceptable policy = true`: operation identity, forward support, contract alignment,
trust, provider, device, and VJP mode must also agree. `AcceptedGraphKernelPlan` carries the
corresponding check for groups derived from its stored graph plan.
:::

:::definition "cuda_native_boundary" (parent := "backend_selection") (lean := "TorchLean.LibTorch.Buffer")
`LibTorch.Buffer` is an opaque handle to a contiguous buffer that retains its binary32 or binary64
dtype. In a CUDA build an ATen tensor sits behind the handle, while the default build reports
`.notLinked` and rejects buffer operations. Lean code can't inspect the native representation
directly. TorchLean owns the differentiation tape and calls ATen with LibTorch autograd recording
disabled.

A typed shape supplies a logical element count, and runtime validation compares it with the
handle's reported length. Gradient seeds and accumulated contributions must also retain the node's
dtype. This checks observable interface conditions without exposing the storage as a Lean array or
deriving its contents from the type.
:::

:::definition "backend_profile" (parent := "backend_selection") (lean := "TorchLean.Backend.Profile")
A backend profile stores a name, a kernel policy (device, provider preference, assurance policy,
and VJP mode), the devices and providers declared available, and the implementation tables that
offer its capsules. Planning operators or a graph validates those tables for duplicate names.

For the CPU, the maintained profile uses checked assurance. On CUDA the default profile,
`libTorchCUDA`, admits recorded LibTorch implementation assumptions as well as retained numerical
comparisons, and the separate `checkedCUDA` profile rejects operations without admissible checked
evidence. Neither profile proves the native implementation correct.
:::

:::definition "backend_provider_catalog" (parent := "backend_selection") (lean := "TorchLean.Backend.Registry.maintained")
Two implementation tables make up the maintained registry. Each offers a
{uses "backend_capsule_contracts"}[contract capsule] per registered operator: the LibTorch CUDA
table offers the capsule of each `LibTorch.Kernel.ofOp` entry's contract (every entry but
`detach`'s, which has none), and the reference CPU table offers the capsule of each operator with a
reference contract. These capsules are planning metadata, not executable handlers. Build
availability filters the CUDA table from CPU-only profiles.
:::

:::definition "checked_cpu_backend_profile" (parent := "backend_selection") (lean := "TorchLean.Backend.Profile.checkedCPU")
The checked CPU profile instantiates {uses "backend_profile"}[the profile record] with
{uses "backend_provider_catalog"}[the maintained implementation tables], CPU-only availability,
and the checked assurance policy.
:::

:::definition "backend_executable_binding" (parent := "backend_selection") (lean := "TorchLean.Backend.KernelCapsule.implements")
A CUDA entry runs a {uses "backend_capsule_contracts"}[selected capsule] only when its own capsule
names the same operator, provider, and device. A reference CPU capsule runs the operator's rule
(which carries no capsule). This check doesn't strengthen the selected capsule's evidence.

For instance, the reference CPU matmul doesn't run under the LibTorch CUDA matmul capsule, although
both declare matmul. Agreement on those labels permits dispatch without inspecting the
implementation's code.
:::

:::definition "backend_planning_and_gating" (parent := "backend_selection") (lean := "TorchLean.IR.Graph.accept")
Under a {uses "backend_profile"}[backend profile], graph acceptance validates the profile's
implementation tables, checks the operator graph (`TorchLean.IR.Graph.check`) and selects a capsule
for the registered operator of each node that needs a kernel. It then groups those choices and runs
the {uses "backend_contract_check"}[contract check] under the profile's assurance policy, returning
an {uses "backend_accepted_kernel"}[accepted graph plan] or the rejected obligations.
:::

:::definition "backend_eager_dispatch" (parent := "backend_selection") (lean := "TorchLean.Session.apply")
For one eager operator, the session selects an admitted capsule for that operator, attributes
included, and records the operator on its tape: with the operator's rule on the CPU tape, or with
its `LibTorch.Kernel.ofOp` entry on the CUDA tape after {uses "backend_executable_binding"}[the
identity check]. That rule or entry records the forward value and the VJP.

Each call selects under the session's profile, which stays fixed, and the first acceptance of each
capsule name is kept for reports. The session doesn't cache tensor results or cotangents: later
calls still execute with their current inputs, and the operator remains responsible for recording
dependencies on those inputs.
:::

:::definition "cuda_autograd_tape" (parent := "backend_selection") (lean := "TorchLean.Tape")
On CUDA the tape stores {uses "cuda_native_boundary"}[device buffers], parent ids, and local VJP
closures in evaluation order (it's the same tape as on the CPU, over another value type).
`requireValue`, `requireGrad`, and backward accumulation check shape tags and native buffer
lengths. Dense backward returns one buffer per node, while sparse backward retains owned buffers
only for selected node ids and requires the caller to release them.
:::

:::theorem "cuda_execution_contracts" (parent := "backend_selection") (lean := "TorchLean.LibTorch.Contract.Agreement.eq_of_isFinite")
Given the stated agreement hypothesis for one of FloatLib's operations and a finite native result,
the decoded native result equals FloatLib's result of that operation, in any format of the
configured binary family ({uses "executable_binary32"}[`ExecFloat.add`] for binary32 addition).
Both the hypothesis and the finiteness side condition remain visible in the theorem type.
:::

:::proof "cuda_execution_contracts"
We rewrite the supplied native result to {uses "executable_binary32"}[the executable result], using
finiteness to rule out the `NaN` case that the agreement hypothesis tolerates. Nothing here proves
the external kernel implementation from source.

The agreement premise permits different NaN encodings. Finiteness removes that alternative,
leaving equality, and so equal bits, for the result. `Agreement.abs_toReal_sub_le` in the
same module gives the real-valued error bound: on the operation's domain, in an IEEE format, a
finite native result lies within half an ulp of the exact result.
:::

:::group "training_runtime"
Scalar objectives and stateful supervised updates.
:::

:::definition "runtime_module_training" (parent := "training_runtime") (lean := "TorchLean.Objective")
`Objective` wraps a scalar trainer, its runtime options, and the selected host/device tensor
conversion. That trainer owns a shape-indexed mutable parameter pack and runs its scalar-loss
{uses "runtime_ops_program"}[program] through {uses "backend_eager_dispatch"}[eager execution].
Gradients are returned to callers, and generic optimizer state is passed to and returned from
update methods (it isn't stored here).

State shapes include persistent model buffers as well as trainable parameters. Differentiability
flags determine which entries receive gradients, so possessing a state tensor does not by itself
make that tensor an optimization variable.
:::

:::definition "supervised_training_state" (parent := "training_runtime") (lean := "TorchLean.Trainer.Session")
`Session` owns a supervised update action and step counter. `step` updates from one sample, and
`step samples (batch := true)` averages a nonempty array's gradients before one update. Both return
`Unit` unless `(loss := true)` requests the pre-update loss. `steps` reads the counter, prediction
and loss use evaluation mode, and `finish` takes an independent snapshot as a trained result.

A batch update and an evaluation may use the same parameters with different mode-dependent behavior
(dropout, for example). The step counter tracks updates, not every forward call made for prediction
or loss inspection.
:::

:::definition "supervised_training_constructor" (parent := "training_runtime") (lean := "TorchLean.Trainer.«open»")
Opening a trainer builds {uses "supervised_training_state"}[the stateful training loop] around
{uses "runtime_module_training"}[a scalar module]. It selects the trainer's optimizer, applies an
optional learning-rate schedule, and refreshes mode-dependent model buffers before each update.
:::

:::group "external_graph_bridges"
Checked import of captured PyTorch graph artifacts and state dicts, and the operation wire format.
:::

:::definition "pytorch_op_wire_format" (parent := "external_graph_bridges") (lean := "TorchLean.Op.encode")
A registered operator (`TorchLean.Op`) is written to an artifact as its codec record:
`TorchLean.Op.encode` gives the operator's tag and its implicit shapes and attributes, and
`TorchLean.Op.decode` reads a record back into an operator and rejects a malformed record. The
PyTorch, JAX and ONNX captures of `python/torchlean` write the same records, with the record
version and the operator codes of the op table `TorchLean.IR.table`.
:::

:::theorem "pytorch_op_wire_round_trip" (parent := "external_graph_bridges") (lean := "TorchLean.Op.decode_encode")
Every {uses "pytorch_op_wire_format"}[registered operator] decodes back from its own record, so no
two operators share a record.

The statement is about the codec: reading JSON, checking an operator's ports against the graph and
the operator's meaning are separate checks.
:::

:::proof "pytorch_op_wire_round_trip"
Case analysis over the {uses "pytorch_op_wire_format"}[operator constructors], where each case is
the operator's own round trip lemma.
:::

:::definition "pytorch_graph_import" (parent := "external_graph_bridges") (lean := "TorchLean.IR.load")
The loader reads the `"artifact"` of a capture document, a typed artifact of format `torchlean.ir`.
It decodes each record into a {uses "pytorch_op_wire_format"}[registered operator], checks each
reference against the port of the input that reads it, the element type of each step and the
outputs, and builds the payload of the constants under the reference runtime. It then requires the
program to give each value of the document's `declared_types` its declared element type and shape,
and refuses a document that names a frontend without them. A document without an artifact is
refused.
:::

:::theorem "pytorch_graph_import_well_shaped" (parent := "external_graph_bridges") (lean := "TorchLean.IR.load_ok")
Every program returned successfully by {uses "pytorch_graph_import"}[the loader] passes the
structural check `TorchLean.IR.Program.check` under the reference runtime, and its payload passes
`TorchLean.IR.Program.Payload.check`.

So downstream code can rely on the accepted program's shape equations. Numerical agreement with the
captured Python model still requires correct operation lowering and matching parameter payloads
(two different activations can satisfy the same shape check).
:::

:::proof "pytorch_graph_import_well_shaped"
The loader returns a program only after `TorchLean.IR.Artifact.load` checks it:
`TorchLean.IR.load_eq_ok` unfolds {uses "pytorch_graph_import"}[the loader] and rules out the
document without an artifact, and `TorchLean.IR.Artifact.load_ok` gives the two checks.
:::

:::theorem "pytorch_graph_import_declared" (parent := "external_graph_bridges") (lean := "TorchLean.IR.load_declared")
Every program returned successfully by {uses "pytorch_graph_import"}[the loader] gives each value
of the document's `declared_types` the element type and shape declared there. The declarations
come from the capture, so this catches a lowering that disagrees with the framework's metadata, not
a wrong declaration.
:::

:::proof "pytorch_graph_import_declared"
`TorchLean.IR.load_eq_ok` gives the successful `TorchLean.IR.Program.checkDeclared`, and
`TorchLean.IR.Program.checkDeclared_ok` turns it into the property of each declared value.
:::

:::definition "pytorch_state_import" (parent := "external_graph_bridges") (lean := "TorchLean.PyTorch.state")
The state importer reads a `torchlean.state` document, which `python/torchlean/state.py` writes
from a module's `state_dict()`, into the `TensorPack` of a TorchLean model. Tensor `i` comes from
the entry under the `i`-th PyTorch name. The entries must be exactly the names, each with the dtype
of the element type, the tensor's shape and one word per element, and every word must decode under
the element type's `Checkpoint.Encoding`.
:::

:::theorem "pytorch_state_round_trip" (parent := "external_graph_bridges") (lean := "TorchLean.PyTorch.state_stateDict")
{uses "pytorch_state_import"}[The importer] reads back the document that
`TorchLean.PyTorch.stateDict` writes for a state when the names have no repeats, there is one name
per tensor and the element type's decoder inverts its encoder.
`TorchLean.PyTorch.state_stateDict_scalar` discharges the last condition for every
`TorchLean.Kernel.Scalar`.
:::

:::proof "pytorch_state_round_trip"
The writer's keys are the distinct names, so each lookup of
{uses "pytorch_state_import"}[the importer] returns the entry written under that name, and each
entry decodes back to its tensor in row-major order.
:::
