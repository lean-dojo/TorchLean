import VersoManual
import NN.IR.Check
-- The parsed graph is evaluated further down, which needs the reference semantics rather than
-- just the shape checker, and the tensor literal notation for the input it is evaluated on.
import NN.IR.Semantics
import NN.Tensor
import NN.Runtime.PyTorch.Import.TorchExport
import NN.Verification.Util.Json
import NN.Runtime.Autograd.Engine.Cuda.Buffer
import NN.Runtime.Autograd.Engine.Cuda.Tape
import TorchLeanBlueprint.Bib
import TorchLeanBlueprint.Roles

open Verso.Genre Manual
open Verso.Genre.Manual.InlineLean
open Lean
open Import.PyTorch.TorchExport
open Lean.Elab.Tactic.GuardMsgs.WhitespaceMode (lax)

#doc (Manual) "External Tools and FFI" =>
%%%
tag := "external-tools-and-ffi"
file := "Crossing-Lean___s-Boundary"
%%%

A captured PyTorch graph arrives as a document naming operations, edges, and tensor shapes.
Lean can parse those bytes, check the shapes, and evaluate the graph. None of those steps alone
establishes that the document describes the intended Python model. We can see the gap by changing
one activation name: both documents pass the shape checker, but their outputs differ.

There are two main boundaries, and they fail differently:

- a *subprocess* exchanges files, standard output, or JSON with another executable;
- an *FFI call* invokes a linked native symbol and may exchange opaque memory handles.

A subprocess hands over copied bytes that become ordinary Lean values after parsing. An FFI
may exchange opaque handles to native allocations, where native code can allocate, mutate, alias,
and free storage outside Lean's type guarantees. For a GPU tensor, the contract must account for
the allocation's lifetime as well as its shape and contents.

The JSON examples below build documents as Lean string literals and print the parser's verdict
during the page build.

# Subprocess Output Validation

The common process helper is small:

```
-- Require a successful process and one complete JSON
-- document on stdout.
def runJsonStdoutChecked
    (ctx : String)
    (cmd : String)
    (args : Array String)
    (cwd : Option String := some ".") :
    IO Json := do
  let stdout ← runStdoutChecked ctx cmd args cwd
  match Json.parse stdout with
  | .ok value => pure value
  | .error message =>
      throw <| IO.userError
        s!"{ctx}: JSON parse error: {message}\nstdout:\n{stdout}"
```

`runStdoutChecked` starts the process, captures its streams, and rejects a nonzero exit code with
the command, arguments, status, and standard error in the diagnostic. `runJsonStdoutChecked` then
requires all of standard output to be one JSON document.

Suppose Python prints:

```
{"format":"torchlean.bound.v1","lower":0.12,"upper":0.31}
```

Successful parsing establishes only that this text is valid JSON. A useful checker must still:

1. require the exact `format` string;
2. require finite numeric fields;
3. establish $`\mathrm{lower}\leq\mathrm{upper}`;
4. connect the interval to a particular graph, payload, input set, and output;
5. invoke a sound acceptance theorem.

The helper gives process failures and data failures separate locations. If Python exits with an
error, the caller sees that exit status and its standard error rather than attempting to interpret
a partial result as a certificate. If Python exits successfully but prints a progress banner before
the JSON, the single-document parse fails. A producer using this interface should put diagnostic
text on standard error and reserve standard output for the artifact.

The small bound document has no graph identifier, input coordinates, or output index. Even if
`0.12 ≤ 0.31` is checked, it remains an interval with no established connection to the requested
computation. Binding those fields to a particular query is part of acceptance, not an optional
annotation to add after a numerical inequality succeeds.

## Finite Fields After Numeric Conversion

JSON's grammar has no bound on exponents, so `1e999` is a syntactically valid number. The range of a
machine `Float` is finite, however. Converting this JSON field succeeds while overflowing to
infinity:

```lean (name := ffiInf)
-- Check finiteness after converting the syntactically valid
-- JSON number.
#eval show IO Unit from do
  match Json.parse "{\"upper\": 1e999}" with
  | .error e => IO.println s!"json error: {e}"
  | .ok j =>
    match j.getObjValAs? Float "upper" with
    | .error e => IO.println s!"field error: {e}"
    | .ok x =>
      IO.println s!"upper = {x}, finite = {x.isFinite}"
```
```leanOutput ffiInf
upper = inf, finite = false
```

The syntax check has succeeded, but the field no longer satisfies a schema requiring a finite
upper bound. Every finite value is below infinity, so an upper-bound test alone would accept
without imposing a finite constraint. Other inequalities and interval ordering can still fail.
Verification parsers therefore check finiteness after conversion, using the variants in
{src "NN/Verification/Util/Json.lean"}[`NN/Verification/Util/Json.lean`], such as
`parseFiniteFloat`, `parseFieldFiniteFloat`, and `expectFiniteFloatArray`, wherever the schema
promises finite claims.

Checking the original spelling for a literal infinity would miss `1e999`. The check must inspect
the converted value that later arithmetic consumes. For an array, one overflowing coordinate is
enough to violate a finite-box contract.

## Duplicate JSON Keys

Duplicate keys create a different problem at the parsing step. JSON's grammar does not forbid
repeating a key, and Lean's parser keeps the last occurrence:

```lean (name := ffiDup)
-- Print the parsed object to reveal which duplicate-key
-- value survived.
#eval show IO Unit from do
  match Json.parse "{\"upper\": 0.1, \"upper\": 9.9}" with
  | .error e => IO.println s!"json error: {e}"
  | .ok j =>
    IO.println s!"parsed  = {j.compress}"
    match j.getObjValAs? Float "upper" with
    | .error e => IO.println s!"field error: {e}"
    | .ok x => IO.println s!"checker read upper = {x}"
```
```leanOutput ffiDup
parsed  = {"upper":9.9}
checker read upper = 9.900000
```

The parsed object contains only `9.9`. A later check of that object cannot recover the earlier
`0.1` or detect the duplicate. A reader inspecting the original text might therefore remember a
different bound from the one the checker used.

The examples in this chapter print back the values they used, and the region example echoes `lo`
and `hi` in its accepting message. This lets a reader compare the checked values with the file.
Refusing duplicate keys outright would be another possible policy. Echoing recognized fields
does not itself detect duplicates or report unknown fields that the schema ignores.

A policy that rejects duplicates must operate while reading the text, or retain additional parsing
evidence. By the time a checker receives this `Json` object, the first occurrence is gone.

## Input Region Validation

Every certificate describing an input box needs finite, ordered endpoints. The shared
`parseBoxRegion` parser checks these conditions for all consumers. It accepts either endpoint
notation or center-and-radius notation and validates the result either way:

```lean (name := ffiRegionDef)
open NN.Verification.Json in
/-- Parse a box region and report the verdict. -/
def ffiRegion (text : String) : String :=
  match Json.parse text with
  | .error e => s!"json error: {e}"
  | .ok j =>
    match parseBoxRegion "region" j with
    | .error e => s!"rejected: {e}"
    | .ok box => s!"accepted: lo={box.lo}, hi={box.hi}"
```

Two well-formed documents, in the two notations:

```lean (name := ffiRegionOk)
-- Normalize both supported region notations to explicit
-- endpoints.
#eval show IO Unit from do
  IO.println (ffiRegion
    "{\"lo\": [0.1, 0.2], \"hi\": [0.3, 0.4]}")
  IO.println (ffiRegion
    "{\"center\": [0.5, 0.8], \"eps\": 0.1}")
```
```leanOutput ffiRegionOk (whitespace := lax)
accepted: lo=#[0.100000, 0.200000], hi=#[0.300000, 0.400000]
accepted: lo=#[0.400000, 0.700000], hi=#[0.600000, 0.900000]
```

The second line is the input box from
{ref "examples"}[the IBP lab], written the way a robustness query usually arrives: a center and a
radius. The parser normalizes it to endpoints once, so no downstream checker has to remember which
notation it was handed.

The following documents violate finiteness, endpoint order, completeness, and the choice of
notation, respectively:

```lean (name := ffiRegionBad)
-- Isolate nonfinite, reversed, incomplete, and conflicting
-- region descriptions.
#eval show IO Unit from do
  IO.println (ffiRegion "{\"lo\": [0.1], \"hi\": [1e999]}")
  IO.println (ffiRegion "{\"lo\": [0.5], \"hi\": [0.2]}")
  IO.println (ffiRegion "{\"lo\": [0.1, 0.2]}")
  IO.println (ffiRegion
    "{\"center\": [0.5], \"eps\": 0.1, \"lo\": [0.0],
      \"hi\": [1.0]}")
```
```leanOutput ffiRegionBad (whitespace := lax)
rejected: region.hi[0]: expected finite float
rejected: region: invalid interval at coordinate 0: [0.500000, 0.200000]
rejected: region: field `lo` requires a matching `hi` field
rejected: region: endpoint fields (`lo`, `hi`) cannot be
  combined with `center` or `eps`
```

In the last document, the center and radius describe `[0.4, 0.6]`, while the explicit endpoints
describe `[0.0, 1.0]`. Rejecting the mixed notation avoids choosing one of two different regions.
The fourth message was rewrapped to fit this page.

In the center form, radius `0.1` is subtracted from and added to each coordinate independently.
The result is a box around `[0.5, 0.8]`. The parser also checks the arrays against the declared
dimension when one is supplied.

That normalization uses host floating-point arithmetic. Acceptance establishes a well-formed box
of the resulting endpoint values; it does not by itself prove outward rounding relative to exact
real decimal inputs. A certificate whose theorem concerns an exact real region needs the
appropriate numeric interpretation or enclosure step as well. The four rejection messages above
localize structural and finite-value failures before such a theorem is considered.

# PyTorch Graph Capture

Run:

```terminal
# Capture the maintained Python probes and validate their
# exported IR documents.
lake exe pytorch_export_check
```

The command asks Python and `torch.export` to capture several small `nn.Module`s
{Informal.citep pytorch2019}[], emits `torchlean.ir.v1` JSON, parses each document in Lean, lowers
supported values to `NN.IR.Graph`, and runs the graph validators.

The maintained numerical probes cover `TinyMLP`, `TinyAffineLayerNorm`, `TinyLayerNormEpsilon`,
and `TinyBatchNormEpsilon`. Each captured graph is executed on a deterministic input and compared
with PyTorch. Generated-reference and state-dict checks run in the same command:

```terminal +output
== PyTorch nn.Module → TorchLean IR runtime check ==
generated Python float bit patterns and tiny epsilon: ok
generated reference code and state-dict round trip: ok
  numerical parity: ok (TinyMLP)
  numerical parity: ok (TinyAffineLayerNorm)
  numerical parity: ok (TinyLayerNormEpsilon)
  numerical parity: ok (TinyBatchNormEpsilon)
pytorch_export_check: ok
```

The parser's structural theorem is `parseGraph_wellShaped`. The examples below inspect that
statement and exercise malformed documents directly, independently of these numerical probes.

The two epsilon models make small normalization constants observable: a formatter that rounded
them to zero could change the computation without changing any shape. Generated-source and
state-dictionary checks exercise separate import paths. Each `ok` records the outcome of its
named probe, with the supplied parameters and deterministic input.

# Graph Exchange Format

I'll use a three-node graph with one input of shape `[1,4]`. The second node applies ReLU
elementwise, preserving that shape. The third sums all four entries to a scalar, whose shape is
`[]`. Parent IDs record that sequence:

```lean (name := ffiJson)
-- Record input, shape-preserving activation, and scalar
-- reduction as separate nodes.
def ffiGood : String := "{
  \"format\": \"torchlean.ir.v1\",
  \"input_id\": 0,
  \"output_ids\": [2],
  \"nodes\": [
    {\"id\": 0, \"kind\": \"input\",
     \"parents\": [], \"shape\": [1, 4]},
    {\"id\": 1, \"kind\": \"relu\",
     \"parents\": [0], \"shape\": [1, 4]},
    {\"id\": 2, \"kind\": \"sum\",
     \"parents\": [1], \"shape\": []}
  ]
}"
```

That string is a Lean value, so the importer can be run on it directly. One helper reports the
verdict:

```lean (name := ffiParseDef)
/-- Import a captured graph and report the verdict. -/
def ffiParse (text : String) : String :=
  match Json.parse text with
  | .error e => s!"json error: {e}"
  | .ok j =>
    match parseGraph j with
    | .error e => s!"rejected: {e}"
    | .ok cg =>
      s!"accepted: nodes={cg.graph.nodes.size}, " ++
        s!"input={cg.inputId}, outputs={cg.outputIds}"
```

```lean (name := ffiParseGood)
-- Report the designated interface as well as the accepted
-- node count.
#eval IO.println (ffiParse ffiGood)
```
```leanOutput ffiParseGood
accepted: nodes=3, input=0, outputs=#[2]
```

The report identifies the three accepted nodes, node `0` as the input, and node `2` as the output.
This fixture exercises the parser directly from a Lean string without starting Python. In a full
capture, the Python producer translates raw
FX or ATen names into stable TorchLean tags. The Lean parser recognizes a conservative list,
parses operation-specific fields, constructs candidate nodes, and validates the result. The op
kind strings come from the fixed v1 table in `NN.Runtime.PyTorch.Wire`. Its round-trip theorem says
that reading the spelling of an operator recovers its `NN.IR.OpTag` identity.

The complete path separates external capture from the checks Lean performs on its result:

```
PyTorch module
   ↓ external capture, treated as an untrusted producer
FX/value graph
   ↓ explicit JSON schema
Lean parser
   ↓ checked structural lowering
NN.IR.Graph
   ↓ denotation / verifier / exporter
TorchLean analysis
```

The first lowering step must preserve more than tensor shapes. For example, `nn.MultiheadAttention`
returns a tuple of attention output and attention weights. Treating that FX node as a single
tensor loses the distinction between its components. TorchLean first retains tuple shape metadata,
then lowers only supported tensor projections. The later Lean checks apply to the resulting
document, so their scope depends on exactly what the parser and graph validators recognize.

Parent identifiers and output identifiers serve different purposes. Parent edges establish the
dependencies needed to evaluate a node. The output list chooses which already-recorded values
form the external result, in return order. Both need validation because a well-shaped internal
node list alone would not make an out-of-range output identifier usable.

# The Parser's Structural Guarantee

`parseGraph_wellShaped` states the structural property of a successful parse:

```lean (name := ffiThm)
-- Success of this parser call supplies the structural
-- premise for downstream proofs.
#check @parseGraph_wellShaped
```
```leanOutput ffiThm (whitespace := lax)
@parseGraph_wellShaped : ∀ {j : Json} {cg : CapturedGraph},
  parseGraph j = Except.ok cg → cg.graph.WellShaped
```

The conclusion is the property `WellShaped`, defined by the executable checker's success:

```lean (name := ffiWellShaped)
-- The proposition uses the same shape-check verdict as
-- executable consumers.
#print NN.IR.Graph.WellShaped
```
```leanOutput ffiWellShaped
def NN.IR.Graph.WellShaped : NN.IR.Graph → Prop :=
fun g => g.checkShapes = Except.ok ()
```

If the importer returns a graph, the theorem establishes that the shared shape checker accepts
it. Downstream proofs can use this result without re-deriving well-shapedness or assuming that the
importer remembered to call the checker. Its premise mentions only the JSON value and successful
parsing, so its conclusion concerns that graph's structure. It makes no claim that Python captured
the intended module or translated every operator correctly
{Informal.citep necula1997}[].

## Shape Checking And Semantic Equality

Change one string in `ffiGood`, from `relu` to `sigmoid`, and evaluate both graphs on the same
input `[[1, -2, 3, -4]]`:

```lean (name := ffiMeaning)
-- Change only the activation to expose a semantic
-- difference that shapes cannot detect.
def ffiSigmoid : String :=
  ffiGood.replace "\"relu\"" "\"sigmoid\""

open TorchLean NN.IR in
/-- Import a captured graph, then run its denotation. -/
def ffiDenote (text : String) : String :=
  match Json.parse text with
  | .error e => s!"json error: {e}"
  | .ok j =>
    match parseGraph j with
    | .error e => s!"rejected: {e}"
    | .ok cg =>
      let x : Tensor Float [1, 4] := [[1, -2, 3, -4]]
      match Graph.denote (α := Float) (g := cg.graph)
          (payload := {})
          (input := Spec.SomeTensor.ofTensor x)
          (outputId := 2) with
      | .error e => s!"eval error: {e}"
      | .ok v => s!"accepted, output = {v.tensor}"

#eval do
  IO.println (ffiDenote ffiGood)
  IO.println (ffiDenote ffiSigmoid)
```
```leanOutput ffiMeaning (whitespace := lax)
accepted, output = 4.000000
accepted, output = 1.820822
```

In the first graph, ReLU turns the input into `[[1, 0, 3, 0]]`, whose sum is `4`. In the second,
sigmoid maps each entry into `(0,1)` before the sum, giving the displayed `1.820822`. Both
elementwise operations preserve `[1,4]`, so both documents pass `parseGraph` and satisfy
`parseGraph_wellShaped`. Distinguishing which graph represents the intended Python module requires
a statement relating capture to the module's semantics. The structural theorem does not supply
that relation.

The checker and proposition serve different callers in {srcDir "NN/IR"}[`NN/IR`]. Backends and
exporters use `Except String Unit` to obtain a readable error. Proofs use a `Prop` as a hypothesis
or conclusion. Defining the proposition as the checker's success gives both uses the same
definition of well-shapedness.

The theorem requires an equality for the actual parser call, tying its conclusion to the graph
returned from that document. It cannot choose between our two documents: both describe valid
functions. Identifying the intended activation belongs to the translation from the source program.

# Import Rejection

Each of the following changes one field in `ffiGood`, letting us trace the resulting rejection
to the check responsible for it:

```lean (name := ffiReject)
-- Change one schema or graph condition at a time to locate
-- each rejection.
#eval show IO Unit from do
  IO.println (ffiParse
    (ffiGood.replace "ir.v1" "ir.v2"))
  IO.println (ffiParse
    (ffiGood.replace "\"relu\"" "\"sort\""))
  IO.println (ffiParse
    (ffiGood.replace "[0], \"shape\": [1, 4]"
                     "[0], \"shape\": [1, 5]"))
  IO.println (ffiParse
    (ffiGood.replace "\"parents\": [0]" "\"parents\": [2]"))
```
```leanOutput ffiReject (whitespace := lax)
rejected: PyTorch graph import: unsupported format `torchlean.ir.v2`
  (expected `torchlean.ir.v1`)
rejected: PyTorch graph import: node[1]: unsupported TorchLean IR op
  kind `sort`
rejected: IR graph: node 1: outShape mismatch: inferred=[1, 4],
  declared=[1, 5] (Node(id=1, kind=relu, parents=#[0], outShape=[1, 5]))
rejected: PyTorch graph import: node[1]: parent id 2 is not < 1
```

The errors identify four stages of validation:

- the *format* rejection comes from the schema check, before any node is looked at;
- the *op kind* rejection comes from the conservative operator table, which is why adding
  `torch.sort` to the Python side without touching the Lean parser produces an explicit
  unsupported-operation failure rather than a partial import;
- the *shape mismatch* comes from `checkShapes`, which reports both the inferred and the
  declared shape, helping locate a mismatch without proving which side introduced it;
- the *parent id* rejection comes from the id discipline, and it is the one that keeps cycles out: a
  node may only refer to strictly earlier nodes, so a candidate containing a forward reference is
  rejected.

The four messages above were rewrapped to fit the page; the runtime prints each on one line.

The command's successful captures do not test every unsupported translation. Reflect padding
could become zero padding, scaled addition could lose its scale, a dtype cast could disappear, or
training-mode BatchNorm could be read as eval-mode BatchNorm. Each substitution can yield a graph
with correct shapes and different values. A producer must either implement the intended semantics
or reject the case, and that behavior needs tests in addition to the Lean shape check. Replacing
an unsupported node with an identity or silently dropping attention weights would have the same
problem.

# State Dictionaries And Graphs

A `state_dict` supplies named tensors. It does not describe data flow. A graph capture supplies
operations and edges. It may refer to parameters without carrying the full checkpoint provenance.

Round-trip import therefore has two obligations:

:::table +header
*
  * Artifact
  * Checks
*
  * graph
  * operator subset, IDs, parents, shapes, attributes, input/output IDs
*
  * state dictionary
  * key mapping, tensor shape and scalar decoding; finite-value checks depend on the importer
:::

Run:

```terminal
# Generate the default family artifacts for joint inspection
# of model and payload.
lake exe torchlean pytorch_roundtrip
```

This writes the generated MLP PyTorch artifacts under
`NN/Examples/Interop/PyTorch/MLP/`. Open the generated model and parameter files together: neither
one is a complete description of the executable network by itself. A graph with a well-shaped
convolution node contains no weights, and a checkpoint full of correctly shaped weights says nothing
about the order the operations run in. Reconstructing the network requires matching the graph's
parameter references to the tensors in the state dictionary.

# Semantic Guarantees For Imported Graphs

For an imported model, distinguish these three claims:

1. Python produced a document and exited successfully.
2. Lean parsed the document into a well-shaped supported graph.
3. The graph's denotation equals the original PyTorch module on all admissible inputs.

The capture experiment and `parseGraph_wellShaped` establish the second proposition, conditional on
the bytes received. The third needs a translation theorem for the supported producer or an explicit
trust assumption about capture and lowering.

Numerical parity tests provide execution evidence for that assumption. The deterministic linear
and normalization probes above can expose transposes, axis errors, and missing biases. Checking
mask translation would require an attention probe. Their conclusions concern the tested cases,
with no universal quantification over parameters and
inputs. Coverage-guided fuzzing {Informal.citep tensorfuzz2019}[] and generated well-typed graphs
{Informal.citep nnsmith2023}[] extend this approach by exploring cases beyond a fixed collection
of hand-written examples; both have found defects in production frameworks.

A graph/payload mismatch can survive both kinds of structural checks. Two checkpoints may supply
all the same names and dimensions but contain different learned values. Pairing the captured graph
with the wrong one then runs a well-shaped network whose prediction is unrelated to the intended
training artifact. Numerical probes can expose that mismatch, while a semantic claim must specify
which payload its graph denotation uses. The empty payload in `ffiDenote` is sufficient only
because its input, activation, and sum require no stored parameters.

# ONNX Import

`NN.Runtime.PyTorch.Export.ONNX` emits a Python adapter for a conservative static-shape ONNX
fragment. The producer handles elementwise operations, vector and broadcast matmul, reductions,
softmax, reshape and flatten, concat, transposes, `Gemm`, and inference BatchNorm. Convolution
records groups, dilation, stride, and explicit asymmetric padding. Its importer accepts a
channels-first sample or a batch of one; larger batches still need producer support even though
the IR convolution contract can represent them. It writes the same `torchlean.ir.v1` document
consumed by `parseGraph`, so both producers use the same Lean parser and structural checks.

ONNX protobuf parsing and shape inference therefore remain outside Lean. Lean checks the smaller IR
artifact it receives. Graph initializers and payload-backed nodes still need a matching payload
import; a well-shaped convolution node by itself does not contain authenticated weights.

# Native FFI And Memory Ownership

The parsed graph is ordinary Lean data. A CUDA buffer instead holds an opaque reference to memory
that native code can allocate, mutate, alias, or free. Parsing cannot establish that such an
allocation remains live.

The CUDA buffer boundary in
{src "NN/Runtime/Autograd/Engine/Cuda/Buffer.lean"}[`Buffer.lean`] contains declarations such as:

```
-- These declarations expose native ownership operations
-- without exposing pointer storage.
@[never_extract, extern "torchlean_cuda_buffer_of_float_array_io"]
opaque ofFloatArrayIO (a : @& FloatArray) : IO Buffer

@[never_extract, extern "torchlean_cuda_buffer_to_float_array_io"]
opaque toFloatArrayIO (b : @& Buffer) : IO FloatArray

@[never_extract, extern "torchlean_cuda_buffer_release_with_token"]
private opaque releaseWithToken (b : @& Buffer) (token : UInt32) : UInt32
```

The annotations control three aspects of the boundary: access to the representation, reference
ownership, and compiler treatment of calls.

Returning an opaque `Buffer`, including through `IO Buffer`, keeps the native pointer representation
hidden from Lean callers. The C implementation still determines whether allocation, copying,
finalization, and release are
correct. Hiding the representation prevents Lean callers from manipulating the pointer directly;
it does not verify the native implementation.

`@&` marks a borrowed argument {Informal.citep immutablebeans2019}[]. Lean's compiler uses reference
counting, and a borrowed parameter is one the callee may inspect during the call without owning:
native code must not consume the borrowed reference. Retaining it beyond the call requires taking
its own reference, with the matching release later. Getting this wrong produces a
use-after-free or a leak that no type in the Lean source mentions. Native translation units also
repeat critical length and geometry checks. The file-to-symbol map in
`NN.Runtime.Autograd.Engine.Cuda.Trusted` identifies the native implementations behind these
declarations.

`never_extract` prevents closed-term extraction and common-subexpression elimination for calls
to the marked declaration. It does not turn a pure signature into an effect type or establish
unique ownership. Cleanup also needs a live result dependency or an explicit effectful boundary.

## Checking Shape Metadata Against Storage

The shape-erased tape does not trust a shape tag by itself. `AnyBuffer` carries a shape and a native
buffer as separate fields, so nothing structural forces them to agree, and the check is explicit:

```lean (name := ffiValidate)
-- Check the native allocation length against the
-- shape-erased metadata.
open Runtime.Autograd.Cuda in
#check @AnyBuffer.validate
```
```leanOutput ffiValidate
AnyBuffer.validate : AnyBuffer → Result AnyBuffer
```

It returns a `Result`, not a `Prop` and not a `Bool`: a mismatch becomes an ordinary Lean error with
the expected and the native element count in the message. Before a buffer reaches a shape-indexed
kernel, `validate` checks that every axis and the total element count fit the CUDA `UInt32` ABI and
that the native buffer has exactly that many elements. Convolution and pooling wrappers additionally
reject zero strides and oversized dimension arrays, validate input, kernel, and output spatial
products independently, and check the returned native output length. Runtime natural-number gather
indices accept the full `UInt32` range and reject larger values before conversion.

These checks turn malformed runtime values into ordinary Lean errors. They establish agreement
between the shape metadata and storage size. A buffer of the right length can still contain
incorrect values, so numerical refinement remains a separate obligation.

`AnyBuffer.validate` returns the accepted buffer for downstream use. A `[2, 3]` tag requires six
native elements; that count cannot establish their row-major layout or the allocation's lifetime.
Those parts of the native contract remain necessary even after the size check succeeds.

The integer-width checks precede conversion because Lean natural numbers do not overflow at the
CUDA ABI's limit. A conversion that silently wrapped a large dimension could turn an apparently
valid tensor into a much smaller allocation request. Checking each axis as well as the product
also prevents a zero-sized axis from concealing another unrepresentable dimension.

# Allocation Effects In IO

Native allocation also raises a question about how the compiler may reuse a result. Compare the
pure primitive with the checked `IO` entry point for the same host upload:

```lean (name := ffiToken)
-- Compare the pure allocation primitive with the checked
-- IO upload.
open Runtime.Autograd.Cuda in
#check @Buffer.ofFloatArray
open Runtime.Autograd.Cuda in
#check @Buffer.ofFloatArrayIO
```
```leanOutput ffiToken
Buffer.ofFloatArray : FloatArray → Buffer
```
```leanOutput ffiToken
Buffer.ofFloatArrayIO : FloatArray → IO Buffer
```

The first signature does not express allocation effects or release-sensitive identity. It says the
result is a function of the input array, so
Lean is entitled to treat two uploads of the same array as one value: common subexpression
elimination is sound for pure functions. Allocation combined with explicit release cannot safely be
treated that way without ownership rules. Two uploads of the
same host array must produce independently owned buffers, or releasing one invalidates the other.

The effectful entry point declares the native call itself in `IO`:

```
-- Sequence the native upload and its allocation result in IO.
@[never_extract, extern "torchlean_cuda_buffer_of_float_array_io"]
opaque ofFloatArrayIO (a : @& FloatArray) : IO Buffer
```

`ofFloatArrayIO a` describes an action. Executing it performs the upload at that point in the
surrounding `IO` sequence. Executing it again with the same host array allocates a separate buffer.
The allocation effect belongs to this native call, rather than to a pure upload evaluated inside
`pure`. The borrowed host array remains available to the caller.

Failure is also part of the call's result. When device allocation runs out of memory, the allocator
releases unused cached blocks and retries. If that retry also runs out of memory, the action throws
`IO.Error.resourceExhausted`, which the caller can handle through the usual `IO` exception
mechanism.
The call has returned no new buffer, and the caller retains its host array and existing device
buffers. This is the checked upload contract; the `IO` type makes sequencing and failure visible,
while correct allocation and ownership still depend on the native implementation.

The release wrapper `releaseIO` still uses a token. It is
called only at an ownership boundary where no alias will be used again; session caches atomically
remove their published alias before calling it, and the native finalizer stays safe after explicit
release because the implementation nulls the pointer.

Both upload entry points copy the same host values, rounding each element to float32. The checked
`IO` entry point additionally exposes when allocation happens and how allocation failure returns
to the caller. These effects do not change the tensor's shape or intended contents. They also do
not establish unique ownership of the returned handle: Lean references to a `Buffer` remain
copyable.

After explicit release, every alias still refers to the native object whose payload was retired.
Copying a Lean reference does not bring the allocation back. The index in `Tensor α s` constrains
rank and dimensions, so it cannot detect use of a stale cached handle. Session code must remove
that handle before release, and consumers must respect the same lifetime.

# Workspaces And Backward

Some forward kernels produce an output plus intermediates needed by their VJP. TorchLean represents
that ownership explicitly:

```
-- Retain the intermediates needed by backward alongside the
-- forward value.
structure WithWorkspace where
  value : Buffer
  workspace : Array Buffer := #[]
```

The tape node retains the workspace until backward has consumed it. Afterwards the cleanup is
threaded through a value that is still used:

```lean (name := ffiWorkspace)
-- The returned keep buffer makes cleanup part of a used
-- result dependency.
open Runtime.Autograd.Cuda in
#check @Buffer.WithWorkspace.releaseWorkspaceThen
open Runtime.Autograd.Cuda in
#check @Buffer.WithWorkspace.releaseAllThen
```
```leanOutput ffiWorkspace (whitespace := lax)
Buffer.WithWorkspace.releaseWorkspaceThen :
  Buffer.WithWorkspace → Buffer → Buffer
```
```leanOutput ffiWorkspace (whitespace := lax)
Buffer.WithWorkspace.releaseAllThen :
  Buffer.WithWorkspace → Buffer → Buffer
```

Each helper takes a result-with-workspace and a buffer to keep, then returns the kept buffer.
Callers thread that return value into their next computation, keeping the release on a path whose
result is still needed. The type does not force callers to invoke the helper: they already have
the `keep` buffer and could use it directly. Cleanup therefore remains an ownership protocol that
the caller must follow, using a result dependency to retain the release operation.

For long training runs this prevents two forms of growth:

- GPU allocations waiting for Lean external-object finalizers;
- tape closures retaining workspaces after their VJP has run.

Allocator counters report live and peak bytes, allocation and free counts, wrapper counts, and
device free memory. These measurements help test the cleanup protocol: for example, live bytes
should not grow indefinitely when a fixed workload repeatedly releases its temporaries. The
measurements describe that run and do not prove the absence of native leaks.

The two workspace helpers retire different ownership sets. `releaseWorkspaceThen` keeps the
result's own value alive while releasing its auxiliaries. `releaseAllThen` also retires that value,
so the returned `keep` must belong to storage the caller can still use. Neither type proves that
those ownership sets are disjoint; the surrounding operation must establish the protocol.

Allocator telemetry can help distinguish retained payloads from retained Lean wrappers. A wrapper
may remain alive after explicit payload release, while a device allocator may retain freed blocks
in a cache. Looking only at driver free memory can therefore hide which layer retained memory.
Live payload bytes, wrapper counts, and cache behavior need to be interpreted together for the
particular repeated workload being inspected.

# Host Profiling

[LeanProfiler](https://github.com/lean-dojo/LeanProfiler) records named `IO` spans and writes both a
Perfetto-compatible trace and a JSON timing summary. It can sit around an existing TorchLean
runner after adding the separate LeanProfiler package dependency:

```
-- Instrument the existing IO runner; device completion
-- needs its own synchronization.
import LeanProfiler
open LeanProfiler

def main : IO Unit :=
  profileFromEnvironment "training" do
    span "model.run" runModel
```

With `LEAN_PROFILE=1`, the trace shows nesting and order while the summary groups repeated spans and
records process counters. The comparison command checks a new summary against a baseline with
explicit absolute and relative tolerances.

A host timer does not automatically measure asynchronous device work. The TorchLean integration
can wait for device completion before closing a span; that number is completion latency, not
per-kernel CUDA or CUPTI time. A short host call may merely queue a much longer GPU operation, so a
short host span can hide device work. It can also mean the relevant host work was not instrumented;
measure completion before assigning the cause.

# External Oracles And Certificates

TorchLean also calls arbitrary-precision or interval tools to propose bounds. A typical workflow is:

```
Lean writes an exact query
        ↓
Arb / python-flint computes an enclosure
        ↓
Lean parses rational midpoint-radius data
        ↓
a checker validates the enclosure or treats it as oracle evidence
```

The rational midpoint and radius preserve the returned numbers without an extra binary64
conversion at ingestion. This avoids the conversion step that overflowed in the `1e999` example.
It does not verify the external interval algorithm. That question belongs to the last step in the
diagram: if a proved checker replays and accepts the result, its theorem supplies the corresponding
proposition. If the result is only compared in a test, it remains oracle evidence.

The same producer and checker distinction applies to α,β-CROWN leaf dumps, PINN residual artifacts,
ODE enclosures, and geometry certificates. The external search may use a large implementation,
while a checker validates only the returned artifact. The schema determines what information
crosses the boundary; the acceptance theorem determines what can be concluded from it.

For each artifact, the report should identify the proposition the checker established and the
assumptions connecting its inputs to the requested computation. Exact midpoint-radius data
preserves the proposed interval; the acceptance theorem must still justify its enclosure claim.

# Evidence For Boundary Contracts

A useful boundary report may say:

- *shape:* proved by a typed constructor;
- *layout:* guarded by length and row-major checks;
- *value:* compared by a regression suite;
- *VJP:* delegated to a trusted external provider;
- *provenance:* native symbol `torchlean_cuda_buffer_matmul`.

A proof of shape safety supplies no arithmetic conclusion, and a source-file link identifies an
implementation without validating its behavior. Recording evidence field by field keeps each
claim attached to the check, test, or assumption that supports it.

# Boundary Failure Tests

The live blocks above exercise numeric parsing, region validation, graph acceptance, and graph
rejection during the page build. To exercise the external capture process and a CUDA training
call, run:

```terminal
# Exercise external graph capture and a separately
# configured CUDA runtime call.
lake exe pytorch_export_check
lake -R -K cuda=true exe torchlean quickstart_mlp \
  --device cuda --steps 2 --show-backend
```

Then deliberately break one condition:

1. add an unsupported PyTorch operation and observe import rejection;
2. change a JSON shape and observe `checkShapes` reject it;
3. request CUDA from a stub build and observe runtime availability rejection;
4. pass a wrong-size Q buffer to the LibTorch SDPA test and observe the Lean and native guard reject
   it.

The live blocks illustrate the first two rejection paths using JSON strings. The last two depend
on the build configuration and native library: availability is checked when the CUDA session is
created, and buffer dimensions are checked at the Lean and native call boundaries. Testing these
failures confirms where execution stops when an input violates a boundary condition.
