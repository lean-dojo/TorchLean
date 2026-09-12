import Verso
import VersoManual
import VersoBlueprint
import NN.Backend
import NN.Runtime.Autograd.Engine
import NN.Runtime.Autograd.Torch
import NN.Runtime.Autograd.Model
import NN.Runtime.Autograd.Train
import NN.API.Trainer.Session
import NN.API.Trainer.Scheduler
import NN.Runtime.PyTorch

open Verso.Genre
open Verso.Genre.Manual
open Informal

#doc (Manual) "Backends and Training" =>

Backend selection is a planning step over declared data. A capsule names the operation, provider,
device, trust level, reduction policy, and the evidence behind its four contract claims. The
contract check turns a plan audit into an accept or reject decision under an assurance policy, and
eager execution binds the accepted capsule to a handler with the same identity. Nothing in this
group proves a kernel correct; the nodes record what is checked and what is assumed.

For example, selecting CUDA matmul involves a shape obligation, a row-major layout obligation,
and numerical obligations for forward and backward. Acceptance explains why this provider is
allowed to run. The native result still depends on the implementation satisfying its recorded
contract. The training and import nodes below describe how values reach that boundary.

:::group "backend_selection"
Kernel contracts, providers, and dispatch.
:::

:::definition "backend_capsule_contracts" (parent := "backend_selection") (lean := "NN.Backend.KernelCapsule")
A kernel capsule records an operation, provider, device, trust level (`checked` or
`trustedExternal`), forward and VJP support, four contract descriptors (shape, layout, forward
value, VJP), and a {uses "backend_numerical_policy"}[numerical policy]. Each descriptor pairs a
structured claim with its evidence: a runtime guard, a test suite, a named trusted boundary, or
`notApplicable`. The capsule states the contract expected from an implementation; it does not
prove that implementation.
:::

:::definition "backend_numerical_policy" (parent := "backend_selection") (lean := "NN.Backend.NumericalPolicy")
The numerical policy of a capsule is its reduction order: `fixedLeft` for the left fold used by the
canonical tensor semantics, `implementationDefined` for native and library accumulations, or
`notApplicable`. Numerical certificates read this field so that a fixed-left range trace is not
reused for an implementation-defined schedule. Rounding mode, subnormal handling, and multiply-add
contraction are not recorded.

The distinction matters even for the same matrix dimensions: changing the accumulation tree
changes the sequence of rounded additions. A shape proof cannot substitute for the missing
numerical-policy match.
:::

:::definition "backend_contract_check" (parent := "backend_selection") (lean := "NN.Backend.KernelPlanAudit.checkContracts")
The contract check reads every obligation of a plan audit and rejects those whose evidence the
assurance policy does not accept. The `checked` policy admits runtime guards, test suites, and
`notApplicable`; only the `external` policy admits a trusted boundary. The result is a
`ContractCheck`: either `accepted` or `rejected` with the failing obligation reports.
:::

:::definition "backend_accepted_kernel" (parent := "backend_selection") (lean := "NN.Backend.AcceptedKernel")
An accepted kernel is a planned operation whose
{uses "backend_capsule_contracts"}[capsule] passed the complete kernel-policy gate, including the
{uses "backend_contract_check"}[evidence check]. Its proof field states
`PlannedKernel.acceptable policy = true`: operation identity, forward support, contract alignment,
trust, provider, device, and VJP mode must also agree. `AcceptedGraphKernelPlan` carries the
corresponding check for groups derived from its stored graph plan.
:::

:::definition "cuda_native_boundary" (parent := "backend_selection") (lean := "Runtime.Autograd.Cuda.Buffer")
`Cuda.Buffer` is an opaque handle to a contiguous float32 buffer. A CUDA build stores device
memory behind the handle; the default stub keeps parity storage on the host. Lean code cannot
inspect either representation directly.

A typed shape supplies a logical element count; runtime validation compares it with the handle's
reported length. This checks an observable interface condition without exposing the storage as a
Lean array or deriving its contents from the type.
:::

:::definition "backend_profile" (parent := "backend_selection") (lean := "NN.Backend.BackendProfile")
A backend profile stores a name, a kernel policy (device, provider preference, assurance policy,
and VJP mode), the devices and providers declared available, and the capsule modules that form its
planning registry. Capsule modules are validated for duplicate names when a graph is planned.
:::

:::definition "backend_provider_catalog" (parent := "backend_selection") (lean := "NN.Backend.Registry.maintainedModules")
The maintained registry collects {uses "backend_capsule_contracts"}[contract capsules] contributed
by the attention, native CUDA, and reference modules. It contains planning metadata, not executable
handlers. Profiles add the separate LibTorch module when requested.
:::

:::definition "checked_cpu_backend_profile" (parent := "backend_selection") (lean := "NN.Backend.BackendProfile.checkedCpu")
The checked CPU profile instantiates {uses "backend_profile"}[the profile record] with
{uses "backend_provider_catalog"}[the maintained capsule modules], CPU-only availability, and the
checked assurance policy.
:::

:::definition "backend_executable_binding" (parent := "backend_selection") (lean := "NN.Backend.KernelCapsule.bind")
Binding a {uses "backend_capsule_contracts"}[selected capsule] to a handler checks that their
operation, provider, and device agree. The resulting executable kernel carries those identity
equalities; binding does not strengthen the capsule's evidence.

A CUDA capsule paired with a CPU handler is rejected even when both declare matmul. Agreement on
those labels permits dispatch; it does not inspect the handler's `IO` body.
:::

:::definition "backend_planning_and_gating" (parent := "backend_selection") (lean := "NN.Backend.BackendProfile.acceptGraph")
A {uses "backend_profile"}[backend profile] validates its configured capsule modules, requires
{uses "ir_structural_validation"}[a structurally well-formed IR graph], and selects a capsule for
each runtime-relevant node. It then groups those choices and runs the
{uses "backend_contract_check"}[contract check] under the profile's assurance policy, returning an
{uses "backend_accepted_kernel"}[accepted graph plan] or the rejected obligations.
:::

:::definition "backend_eager_dispatch" (parent := "backend_selection") (lean := "Runtime.Autograd.Torch.Internal.EagerSession.executeSelected")
For one eager operation, the session selects and caches an admitted capsule, finds a handler with
the same operation, provider, and device, and runs it through
{uses "backend_executable_binding"}[the checked binding]. This dispatcher does not append an
autograd node; each operation implementation records its own forward value and VJP.

Caching a selection reuses the admitted provider choice for this session, whose profile stays
fixed. It does not cache tensor results or cotangents: later calls still execute with their current
inputs, and the operation remains responsible for recording dependencies on those inputs.
:::

:::definition "cuda_autograd_tape" (parent := "backend_selection") (lean := "Runtime.Autograd.Cuda.Tape")
The CUDA tape stores {uses "cuda_native_boundary"}[device buffers], parent ids, and local VJP
closures in evaluation order. `requireValue`, `requireGrad`, and backward accumulation check shape
tags and native buffer lengths. Dense backward returns one buffer per node; sparse backward retains
owned buffers only for selected node ids and requires the caller to release them.
:::

:::theorem "cuda_execution_contracts" (parent := "backend_selection") (lean := "Runtime.Autograd.Cuda.Float32Contract.native_add_eq_ieee32_of_isFinite")
Given the stated native bit-agreement hypothesis and a finite native result, decoded native scalar
addition equals {uses "executable_binary32"}[`IEEE32Exec.add`]. Both the hypothesis and the
finiteness side condition remain visible in the theorem type.
:::

:::proof "cuda_execution_contracts"
The proof rewrites the supplied native result bits to
{uses "executable_binary32"}[the executable binary32 result], using finiteness to rule out the
`NaN` case the agreement hypothesis tolerates. It does not prove the external kernel implementation
from source.

The bit-agreement premise permits different NaN encodings. Finiteness removes that alternative,
leaving exact bit equality for the scalar result. Applying a real-valued error bound then needs
the separate theorem relating the executable binary32 operation to real arithmetic.
:::

:::group "training_runtime"
Scalar objectives and stateful supervised updates.
:::

:::definition "runtime_module_training" (parent := "training_runtime") (lean := "Runtime.Autograd.Model.Module.Objective")
`Objective` wraps a scalar trainer, its runtime options, and the selected host/device tensor
conversion. The trainer owns a shape-indexed mutable parameter pack and runs its scalar-loss
{uses "runtime_ops_program"}[program] through {uses "backend_eager_dispatch"}[eager execution] or
{uses "runtime_typed_graph_autograd"}[typed graph execution]. Gradients are returned to callers;
generic optimizer state is passed to and returned from update methods rather than stored here.

State shapes include persistent model buffers as well as trainable parameters. Differentiability
flags determine which entries receive gradients, so possessing a state tensor does not by itself
make that tensor an optimization variable.
:::

:::definition "supervised_training_state" (parent := "training_runtime") (lean := "TorchLean.Trainer.Session")
`Session` owns a supervised update action and step counter behind the public `step`, `stepBatch`,
and `steps` operations, together with evaluation-mode prediction and loss and the `finish`
operation that packages the live state as a trained result.

A batch update and an evaluation may use the same parameters with different mode-dependent
behavior, such as dropout. The step counter tracks updates; it is not a count of every forward
call made for prediction or loss inspection.
:::

:::definition "supervised_training_constructor" (parent := "training_runtime") (lean := "TorchLean.Trainer.«open»")
Opening a trainer builds {uses "supervised_training_state"}[the stateful training loop] around
{uses "runtime_module_training"}[a scalar module]. It selects the trainer's optimizer, applies an
optional learning-rate schedule, and refreshes mode-dependent model buffers before each update.
:::

:::group "external_graph_bridges"
Checked import of captured PyTorch graph artifacts and the operation wire format.
:::

:::definition "pytorch_op_wire_format" (parent := "external_graph_bridges") (lean := "Interop.PyTorch.Wire.parseOpTag?")
`NN.IR.OpTag` identifies each IR operation constructor. `Wire.opTag` gives its fixed v1 `kind`
string, and `Wire.parseOpTag?` reads that string back into a tag.
:::

:::theorem "pytorch_op_wire_round_trip" (parent := "external_graph_bridges") (lean := "Interop.PyTorch.Wire.parse_op_tag")
Every {uses "pytorch_op_wire_format"}[operation tag] parses back from its own wire string, so the
tag table is complete and collision free. `parse_op_kind_tag` states the same round trip starting
from an `OpKind` with payload.

The conclusion recovers the constructor tag, not all its attributes. Reconstructing an axis,
stride, shape, or tensor payload needs the corresponding parser and validation beyond this tag
identity.
:::

:::proof "pytorch_op_wire_round_trip"
Case analysis over the {uses "pytorch_op_wire_format"}[tag constructors]; each case is `rfl`.
:::

:::definition "pytorch_graph_import" (parent := "external_graph_bridges") (lean := "Import.PyTorch.TorchExport.parseGraph")
The `torch.export` adapter parses TorchLean's captured graph schema, lowers its supported values to
the shared IR, runs {uses "ir_shape_validation"}[shape validation], and checks that the named input
and output nodes exist.
:::

:::theorem "pytorch_graph_import_well_shaped" (parent := "external_graph_bridges") (lean := "Import.PyTorch.TorchExport.parseGraph_wellShaped")
Every graph returned successfully by {uses "pytorch_graph_import"}[the `torch.export` parser]
satisfies {uses "ir_shape_validation"}[TorchLean's executable shape check].

This permits downstream code to rely on the accepted graph's shape equations. Numerical
agreement with the captured Python model still requires correct operation lowering and matching
parameter payloads; two different activations can satisfy the same shape check.
:::

:::proof "pytorch_graph_import_well_shaped"
The proof unfolds {uses "pytorch_graph_import"}[the parser], rules out each rejected branch, and
returns the successful {uses "ir_shape_validation"}[shape check].
:::
