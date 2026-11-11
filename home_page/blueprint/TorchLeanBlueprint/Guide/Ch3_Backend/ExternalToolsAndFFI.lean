import VersoManual
import TorchLean.NN.IR.Loaded
import TorchLean.Verify.JSON
import TorchLean.CUDA.LibTorch.Buffer
import TorchLean.NN.LibTorch.Tape
import TorchLeanBlueprint.Bib
import TorchLeanBlueprint.Roles
import TorchLean.NN.IR.Builder
import TorchLean.Tensor.Op.Pointwise.Real

open Verso.Genre Manual
open Verso.Genre.Manual.InlineLean
open Lean
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

- A *subprocess* exchanges files, standard output, or JSON with another executable.
- An *FFI call* invokes a linked native symbol and may exchange opaque memory handles.

A subprocess hands over copied bytes that become ordinary Lean values after parsing. An FFI
may exchange opaque handles to native allocations, where native code can allocate, mutate, alias,
and free storage outside Lean's type guarantees. For a GPU tensor, the contract must account for
the allocation's lifetime as well as its shape and contents.

Each JSON example below builds its document as a Lean string literal and prints the parser's
verdict during the page build.

# Subprocess Output Validation

Our common process helper is small:

```
-- Require a successful process and one complete JSON
-- document on stdout.
def runJson
    (ctx : String)
    (cmd : String)
    (args : Array String)
    (cwd : Option String := some ".") :
    IO Json := do
  let stdout ← run ctx cmd args cwd
  match Json.parse stdout with
  | .ok value => pure value
  | .error message =>
      throw <| IO.userError
        s!"{ctx}: JSON parse error: {message}\nstdout:\n{stdout}"
```

`run` starts the process, captures its streams, and rejects a nonzero exit code (the diagnostic
carries the command, arguments, status, and standard error). `runJson` then requires all of
standard output to be one JSON document.

Suppose Python prints:

```
{"format":"torchlean.bound.v1","lower":0.12,"upper":0.31}
```

Successful parsing establishes only that this text is valid JSON. A useful checker must still:

1. require the exact `format` string
2. require finite numeric fields
3. establish $`\mathrm{lower}\leq\mathrm{upper}`
4. connect the interval to a particular graph, payload, input set, and output
5. invoke a sound acceptance theorem

The helper gives process failures and data failures separate locations. If Python exits with an
error, the caller sees that exit status and its standard error instead of attempting to interpret a
partial result as a certificate. If Python exits successfully but prints a progress banner before
the JSON, the single-document parse fails. A producer using this interface should put diagnostic
text on standard error and reserve standard output for the artifact.

That small bound document has no graph identifier, input coordinates, or output index, so even if
`0.12 ≤ 0.31` is checked, it remains an interval with no established connection to the requested
computation. Binding those fields to a particular query is part of acceptance, not an optional
annotation to add after a numerical inequality succeeds.

## Finite Fields After Numeric Conversion

JSON's grammar has no bound on exponents, so `1e999` is a syntactically valid number. But a machine
`Float` has a finite range. Converting this JSON field succeeds while overflowing to infinity:

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
Verification parsers therefore check finiteness after conversion, using `parseFiniteFloat` and its
array and matrix forms in {src "TorchLean/NN/Json.lean"}[`TorchLean/NN/Json.lean`]
wherever the schema promises finite claims.

Checking the original spelling for a literal infinity would miss `1e999`, so the check must inspect
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

Only `9.9` survives in the parsed object, and a later check of that object can't recover the
earlier `0.1` or detect the duplicate. So a reader inspecting the original text might remember a
different bound from the one the checker used.

This page's examples print back the values they used (the region example echoes `lo` and `hi` in
its accepting message), so a reader can compare the checked values with the file. Refusing
duplicate keys outright would be another possible policy. But echoing recognized fields doesn't by
itself detect duplicates or report unknown fields that the schema ignores.

A policy that rejects duplicates must operate while reading the text, or retain additional parsing
evidence. By the time a checker receives this `Json` object, the first occurrence is gone.

## Input Region Validation

Every certificate describing an input box needs finite, ordered endpoints. `BoxRegion.parse` rounds
each endpoint once outward into binary64 and checks these conditions on the rounded endpoints. The
leaf checker of the verification chapter reads its boxes with `BoxRegion.parseEndpoints` instead,
as the exact rationals of the decimals. `BoxRegion.parse` accepts either endpoint notation or
center-and-radius notation and validates the result either way:

```lean (name := ffiRegionDef)
open TorchLean.Json.BoxRegion in
/-- Parse a box region and report the verdict. -/
def ffiRegion (text : String) : String :=
  match TorchLean.Certificate.DecimalJson.parse text with
  | .error e => s!"json error: {e}"
  | .ok j =>
    match parse "region" j with
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

The second line is the input box from {ref "examples"}[the IBP lab], written the way a robustness
query usually arrives: a center and a radius. Our parser normalizes it to endpoints once, so no
downstream checker has to remember which notation it was handed.

These documents violate finiteness, endpoint order, completeness, and the choice of notation,
respectively:

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

In the last document, the center and radius describe `[0.4, 0.6]`, while the endpoint fields
describe `[0.0, 1.0]`. Rejecting the mixed notation avoids choosing one of two different regions.
We rewrapped the fourth message to fit this page.

In the center form, radius `0.1` is subtracted from and added to each coordinate independently,
giving a box around `[0.5, 0.8]`. When a declared dimension is supplied, the parser also checks the
arrays against it.

That normalization is exact on the decimals: the center form computes `0.5 - 0.1` and `0.5 + 0.1`
as rationals, and each endpoint is then rounded once outward, the lower one down and the upper one
up, so the accepted box contains the box the document states. Each endpoint goes through FloatLib's
directed rounding, for which `TorchLean.DecimalRounding.toEReal_convertDecimalText_down_le`,
`Model.toEReal_ofRat_down_le` and their upward forms prove the outward direction. All four
rejection messages above localize structural and finite-value failures before any certificate
theorem is considered.

# PyTorch Graph Capture

The Python package {srcDir "python/torchlean"}[`python/torchlean`] captures a supported
`nn.Module` with `torch.export` or FX {Informal.citep pytorch2019}[], and its JAX capture reads a
jaxpr. Both write a capture document whose `"artifact"` field holds a typed artifact of format
`torchlean.ir`. The format name, the record version, the operator codes and the pointwise forms
they write come from the op table that Lean computes (`TorchLean.IR.table`). Lean reads the document
of either capture with one loader, `TorchLean.IR.load`, whose structural theorem is
`TorchLean.IR.load_ok`. `TorchLean.IR.load_declared` states that the loaded program gives every
declared value its recorded element type and shape.

The examples below inspect that theorem and exercise malformed documents. Numerical agreement is
a separate question: shape validation cannot detect a missing affine bias or a changed
normalization epsilon.

Here the loader is {src "TorchLean/NN/IR/Loaded.lean"}[`TorchLean.IR.load`], and the artifact reader
it calls is {src "TorchLean/NN/IR/Artifact.lean"}[`TorchLean.IR.Artifact.ofJson`].

# Graph Exchange Format

We use a program of three steps. Step 0 reads the input, of shape `[1, 4]`. Step 1 applies ReLU
to each entry and keeps that shape. Step 2 sums all four entries to a scalar, of shape `[]`. A
step's id is its position in `"nodes"`, and the reference `[j, 0]` reads the first output of step
`j`. Each operator step holds the codec record of a registered operator. The record of the sum
holds the input shape, the code `0` of a sum, the summed dimensions and `keepdim`. The
`"scalars"` array names the element type of each step, and `"constants"` holds the words of each
constant step and `null` at the other steps:

```lean (name := ffiJson)
-- Record the input, a shape-preserving activation and a
-- scalar reduction as three float64 steps.
def ffiGood : String := "{\"artifact\": {
  \"format\": \"torchlean.ir\",
  \"nodes\": [
    {\"input\": {\"index\": 0,
       \"port\": {\"value\": [1, 4]}}},
    {\"op\": {\"record\": {\"version\": 3,
       \"tag\": \"relu\",
       \"fields\": [{\"shape\": [1, 4]}]},
     \"inputs\": [[0, 0]]}},
    {\"op\": {\"record\": {\"version\": 3,
       \"tag\": \"reduce\",
       \"fields\": [{\"shape\": [1, 4]}, {\"nat\": 0},
         {\"nats\": [0, 1]}, {\"bool\": false}]},
     \"inputs\": [[1, 0]]}}
  ],
  \"outputs\": [[2, 0]],
  \"scalars\": [\"float64\", \"float64\", \"float64\"],
  \"constants\": [null, null, null]
}}"
```

That string is a Lean value, so we can run the loader on it. One helper reports the verdict:

```lean (name := ffiParseDef)
/-- Load a capture document and report the verdict. -/
def ffiParse (text : String) : String :=
  match Json.parse text with
  | .error e => s!"json error: {e}"
  | .ok j =>
    match TorchLean.IR.load j with
    | .error e => s!"rejected: {e}"
    | .ok (program, _) =>
      s!"accepted: steps={program.nodes.size}, " ++
        s!"outputs={program.outputs.map (·.node)}"
```

```lean (name := ffiParseGood)
-- Report the designated outputs as well as the accepted
-- step count.
#eval IO.println (ffiParse ffiGood)
```
```leanOutput ffiParseGood
accepted: steps=3, outputs=#[2]
```

The report identifies the three accepted steps and step `2` as the output. This fixture exercises
the loader from a Lean string without starting Python. In a full capture, the Python producer
translates FX, ATen or jaxpr operations into the records of registered operators. The loader
decodes each record with `TorchLean.Op.decode`, which reads back the record that
`TorchLean.Op.encode` writes for every registered operator, and checks each reference against the
port of the operator input that reads it.

Across the complete path, external capture stays separate from the checks Lean performs on its
result:

```
PyTorch module or jaxpr
   ↓ external capture, treated as an untrusted producer
records of registered operators, with the op table's codes
   ↓ one JSON document of format torchlean.ir
Lean loader (TorchLean.IR.load)
   ↓ decoded records, checked references, ports and element types
typed program (TorchLean.IR.Program) and its payload
   ↓ evaluation / verifier / exporter
TorchLean analysis
```

A step can have several outputs, such as the values and the indices of a sort, and the reference
`[j, k]` reads output `k` of step `j`, so a tuple-valued operation is one step. The later Lean
checks apply to the decoded program, so their scope depends on what the loader and its checks
recognize.

Input references and output references serve different purposes. The `"inputs"` of a step
establish the dependencies needed to evaluate it. The `"outputs"` list chooses which recorded
values form the external result, in return order. Both need validation because a well-formed list
of steps alone would not make an out-of-range output reference usable.

`TorchLean.IR.Artifact.ofJson` checks the format name and decodes each operator record with
`Op.decode` and each element type, but it doesn't check references, ports or words against the
steps (that's `TorchLean.IR.Artifact.load`'s job). `TorchLean.IR.Artifact.ofJson_toJson` proves
that decoding the JSON of any artifact returns it, which is a statement about Lean `Json` values,
not about text.

# The Parser's Structural Guarantee

`TorchLean.IR.load` reads a capture document into a typed program and its payload under the
reference runtime, and `TorchLean.IR.load_ok` states the structural property of a successful load:

```lean (name := ffiThm)
-- Success of this loader call supplies the structural
-- premise for downstream proofs.
#check @TorchLean.IR.load_ok
```
```leanOutput ffiThm (whitespace := lax)
@TorchLean.IR.load_ok : ∀ {document : Json}
  {p : TorchLean.IR.Program}
  {P : TorchLean.IR.Program.Payload
    TorchLean.IR.Program.Machine.reference},
  TorchLean.IR.load document = Except.ok (p, P) →
    p.check TorchLean.IR.Program.Machine.reference =
        Except.ok () ∧
      P.check p = Except.ok ()
```

The conclusion is the success of two executable checkers. `TorchLean.IR.Program.check` checks the
element type of each step, that each reference names an output of an earlier step at a port that
the reading input accepts, the attributes of each operator (`TorchLean.Op.check`), and the outputs
and input indices. `TorchLean.IR.Program.Payload.check` checks that the payload holds a constant of
the step's type exactly at each constant step.

If the loader returns a program, the theorem establishes that both checkers accept it. Downstream
proofs can then use this result without re-running the checks or assuming that the loader
remembered to call the checkers. Its premise mentions only the JSON value and a successful load, so
its conclusion concerns that program's structure: it says nothing about whether Python captured the
intended module or translated every operator correctly {Informal.citep necula1997}[].

## Shape Checking And Semantic Equality

Change one string in `ffiGood`, from `relu` to `sigmoid`, and evaluate both programs on the same
input `[[1, -2, 3, -4]]`:

```lean (name := ffiMeaning)
-- Change only the activation to expose a semantic
-- difference that the checks cannot detect.
def ffiSigmoid : String :=
  ffiGood.replace "\"relu\"" "\"sigmoid\""

/-- The input `[[1, -2, 3, -4]]`: its element type, shape
and float64 words. -/
def ffiInput : Json :=
  Json.mkObj [("dtype", toJson "float64"),
    ("shape", toJson [1, 4]),
    ("bits", toJson (([1, -2, 3, -4] : List Float).map
      (·.toBits.toNat)))]

/-- Load a capture document, then evaluate it on
`ffiInput`. -/
def ffiDenote (text : String) : String :=
  match Json.parse text >>= TorchLean.IR.load with
  | .error e => s!"rejected: {e}"
  | .ok (program, payload) =>
    let result : Except String _ := do
      let x ← TorchLean.IR.loadValue .reference ffiInput
      let (outputs, _) ←
        program.eval .reference payload #[x]
      outputs.mapM TorchLean.IR.words
    match result with
    | .error e => s!"eval error: {e}"
    | .ok [(_, _, #[w])] =>
      s!"accepted, output = {Float.ofBits w.toUInt64}"
    | .ok _ => "unexpected outputs"

#eval do
  IO.println (ffiDenote ffiGood)
  IO.println (ffiDenote ffiSigmoid)
```
```leanOutput ffiMeaning (whitespace := lax)
accepted, output = 4.000000
accepted, output = 1.820822
```

In the first program, ReLU turns the input into `[[1, 0, 3, 0]]`, whose sum is `4`. In the second,
sigmoid maps these four finite inputs into `(0,1)` before the sum, giving the displayed `1.820822`.
Both elementwise operations keep the shape `[1, 4]`, so `TorchLean.IR.load` accepts both documents
and `TorchLean.IR.load_ok` applies to both. Distinguishing which program represents the intended
Python module requires a statement relating capture to the module's semantics, and the structural
theorem doesn't supply that relation.

In {srcDir "TorchLean/NN/IR"}[`TorchLean/NN/IR`], the checker serves two kinds of callers. Backends
and exporters use its `Except String Unit` result to obtain a readable error. Proofs state its
success (`p.check .reference = .ok ()`) as a hypothesis or conclusion, so both uses share one
definition of a well-formed program.

As a hypothesis, the theorem takes an equality for the actual loader call, which ties its
conclusion to the program returned from that document. But it can't choose between our two
documents: both describe valid functions. Identifying the intended activation belongs to the
translation from the source program.

# Import Rejection

Each of the following changes one field in `ffiGood`, letting us trace the resulting rejection
to the check responsible for it:

```lean (name := ffiReject)
-- Change one schema or program condition at a time to
-- locate each rejection.
#eval show IO Unit from do
  IO.println (ffiParse
    (ffiGood.replace "torchlean.ir" "unrecognized"))
  IO.println (ffiParse
    (ffiGood.replace "\"relu\"" "\"unique\""))
  IO.println (ffiParse
    (ffiGood.replace "{\"value\": [1, 4]}"
                     "{\"value\": [1, 5]}"))
  IO.println (ffiParse
    (ffiGood.replace "\"inputs\": [[0, 0]]"
                     "\"inputs\": [[2, 0]]"))
```
```leanOutput ffiReject (whitespace := lax)
rejected: IR import: format unrecognized, not torchlean.ir
rejected: IR import: the record unique does not decode
rejected: IR check: node 1: output 0 of node 0 has port
  { kind := TorchLean.Port.Kind.value, shape := [1, 5] }, which
  input port { kind := TorchLean.Port.Kind.value, shape := [1, 4] }
  does not accept
rejected: IR check: node 1: no earlier output 0 of node 2
```

These errors identify four stages of validation:

- The *schema* rejection comes before any step is looked at. An artifact in an unrecognized
  format is not read at all.
- The *operator* rejection comes from the codec, which decodes no record tagged `unique`. Adding
  `torch.unique` to the Python side without a registered Lean operator therefore fails at
  decoding, and no part of the document is loaded.
- The *port* rejection comes from `TorchLean.IR.Program.check`, which reports both the port that the
  input step writes and the port that the ReLU input accepts. It locates a mismatch without
  proving which side introduced it.
- The *forward reference* rejection is the one that keeps cycles out. A step may read only the
  outputs of strictly earlier steps, so a document whose step 1 reads step 2 is rejected.

We rewrapped the third message to fit the page. The runtime prints each message on one line.

Successful captures do not test every unsupported translation. Reflect padding
could become zero padding, scaled addition could lose its scale, a dtype cast could disappear, or
training-mode BatchNorm could be read as eval-mode BatchNorm. Each substitution can yield a graph
with correct shapes and different values. A producer must either implement the intended semantics
or reject the case, and that behavior needs tests in addition to the Lean shape check. Replacing
an unsupported node with an identity or silently dropping attention weights would have the same
problem.

# State Dictionaries And Graphs

A `state_dict` supplies named tensors, but it doesn't describe data flow. A graph capture supplies
operations and edges, and it may refer to parameters without carrying the full checkpoint
provenance.

Round-trip import therefore has two obligations:

:::table +header
*
  * Artifact
  * Checks
*
  * graph
  * operator records, references, ports, element types, inputs and outputs
*
  * state dictionary
  * names, dtypes, shapes and words, decoded at the element type, with no finite-value check
:::

Run:

```terminal
# Run the MLP on its PyTorch state dictionary.
scripts/lake.sh exe torchlean pytorch_roundtrip --model mlp
```

This reads `mlp.json` under `TorchLeanExamples/Interop/PyTorch/MLP/`, a PyTorch state dictionary
of format `torchlean.state`, into the model's `TensorPack Float` and prints the output of the
ordinary `nn` model on a fixed input. `--model cnn` and `--model transformer` run the other two
fixtures. When exchanging a graph and a separate state dictionary, inspect them together. A graph
with a well-shaped convolution node contains no weights, and a checkpoint full of correctly shaped
weights says nothing about the order the operations run in. Reconstructing the network requires
matching the graph's parameter references to the tensors in the state dictionary.

# Semantic Guarantees For Imported Graphs

For an imported model, distinguish these three claims:

1. Python produced a document and exited successfully.
2. Lean parsed the document into a well-shaped supported graph.
3. The graph's denotation equals the original PyTorch module on all admissible inputs.

Our capture experiment and the loader's checks (with `TorchLean.IR.load_ok`) establish the second
proposition, conditional on the bytes received. Establishing the third takes a translation theorem
for the supported producer or a stated trust assumption about capture and lowering.

Numerical parity tests provide execution evidence for that assumption. The deterministic linear and
normalization probes above can expose transposes, axis errors, and missing biases, though checking
mask translation would require an attention probe. Their conclusions concern the tested cases, with
no universal quantification over parameters and inputs. Coverage-guided fuzzing
{Informal.citep tensorfuzz2019}[] and generated well-typed graphs {Informal.citep nnsmith2023}[]
extend this approach by exploring cases beyond a fixed collection of hand-written examples (both
have found defects in production frameworks).

A graph/payload mismatch can survive both kinds of structural checks. Two checkpoints may supply
all the same names and dimensions but contain different learned values. Pairing the captured graph
with the wrong one then runs a well-shaped network whose prediction is unrelated to the intended
training artifact. Numerical probes can expose that mismatch, while a semantic claim must specify
which payload its graph's evaluation uses. `ffiDenote` passes the payload the loader built from
the document, which holds no constant because the input, activation, and sum read no stored
parameters.

# ONNX Import

The module `torchlean.onnx` of {srcDir "python/torchlean"}[`python/torchlean`] lowers a
conservative static-shape ONNX fragment. It handles elementwise operations with the broadcasting
of their ONNX version, vector and broadcast matmul, reductions, softmax, reshape and flatten,
concat, transposes, `Gemm`, inference BatchNorm, and convolution with groups, dilation, stride,
explicit asymmetric padding and any batch. It writes a typed artifact of format `torchlean.ir`:
each operator is the codec record of a registered operator, every step has the floating element
type of the ONNX value it computes, and each initializer the graph reads as data is a constant
holding its own words. The format name, the record version, the operator codes and the JSON of
each element type come from the op table. Lean decodes the artifact with
`TorchLean.IR.Artifact.ofJson`, checks it and builds its constants with
`TorchLean.IR.Artifact.load`, and runs it in those element types with `TorchLean.IR.Program.eval`.

So ONNX protobuf parsing and shape inference remain outside Lean, and Lean checks the smaller
artifact it receives: every record decodes, every input reads a port it accepts, and every word
lies in its type. A checked artifact holds the model's weights exactly as the file stores them, but
that doesn't establish that they're the intended ones.

# Native FFI And Memory Ownership

A parsed graph is ordinary Lean data, but a CUDA buffer holds an opaque reference to memory that
native code can allocate, mutate, alias, or free. Parsing can't establish that such an allocation
remains live.

In {src "TorchLean/CUDA/LibTorch/Buffer.lean"}[`Buffer.lean`], the CUDA buffer boundary contains
declarations such as:

```
-- These declarations expose native ownership operations
-- without exposing pointer storage.
@[never_extract, extern "torchlean_cuda_buffer_of_float_array_io"]
opaque ofFloatArrayIO (a : @& FloatArray) (dtype : Dtype := .float32) : IO Buffer

@[never_extract, extern "torchlean_cuda_buffer_to_float_array_io"]
opaque toFloatArrayIO (b : @& Buffer) : IO FloatArray

@[never_extract, extern "torchlean_cuda_buffer_release_io"]
opaque releaseIO (b : @& Buffer) : IO UInt32
```

These annotations control three aspects of the boundary: access to the representation, reference
ownership, and compiler treatment of calls.

Returning an opaque `Buffer` (including through `IO Buffer`) keeps the native pointer
representation hidden from Lean callers. Whether allocation, copying, finalization, and release are
correct is still up to the C implementation. Hiding the representation prevents Lean callers from
manipulating the pointer directly, but it doesn't verify the native implementation.

`@&` marks a borrowed argument {Informal.citep immutablebeans2019}[]. Lean's compiler uses
reference counting, and a borrowed parameter is one the callee may inspect during the call without
owning: native code must not consume the borrowed reference, and retaining it beyond the call
requires taking its own reference (with the matching release later). Getting this wrong produces a
use-after-free or a leak that no type in the Lean source mentions. Native translation units also
repeat critical length and geometry checks. For these declarations, the file-to-symbol map in
`TorchLean.CUDA.LibTorch.Trusted` identifies the native implementations.

`never_extract` prevents closed-term extraction and common-subexpression elimination for calls to
the marked declaration. But it doesn't turn a pure signature into an effect type or establish
unique ownership, so cleanup also needs a live result dependency or an effectful boundary.

## Checking Shape Metadata Against Storage

Our shape-erased tape doesn't trust a shape tag by itself. `SomeBuffer` carries a shape and a
native buffer as separate fields, so nothing structural forces them to agree, and a separate
function checks them:

```lean (name := ffiValidate)
-- Check the native allocation length against the
-- shape-erased metadata.
open TorchLean.LibTorch in
#check @SomeBuffer.validate
```
```leanOutput ffiValidate
SomeBuffer.validate : SomeBuffer → TorchLean.Result SomeBuffer
```

That function returns a `Result`, not a `Prop` and not a `Bool`: a mismatch becomes an ordinary
Lean error (with the expected and the native element count in the message). Before a buffer reaches
a shape-indexed kernel, `validate` checks that every axis and the total element count fit the CUDA
`UInt32` ABI and that the native buffer has exactly that many elements. Convolution and pooling
wrappers also reject zero strides and oversized dimension arrays, validate input, kernel, and
output spatial products independently, and check the returned native output length. Runtime
natural-number gather indices accept the full `UInt32` range and reject larger values before
conversion.

These checks turn malformed runtime values into ordinary Lean errors, and they establish agreement
between the shape metadata and storage size. But a buffer of the right length can still contain
incorrect values, so numerical refinement remains a separate obligation.

`SomeBuffer.validate` returns the accepted buffer for downstream use. A `[2, 3]` tag requires six
native elements, but that count can't establish their row-major layout or the allocation's
lifetime. Those parts of the native contract remain necessary even after the size check succeeds.

We run the integer-width checks before conversion because Lean natural numbers don't overflow at
the CUDA ABI's limit, and a conversion that silently wrapped a large dimension could turn an
apparently valid tensor into a much smaller allocation request. Checking each axis as well as the
product also prevents a zero-sized axis from concealing another unrepresentable dimension.

# Allocation Effects In IO

Native allocation also raises a question about how the compiler may reuse a result. Compare the
pure primitive with the checked `IO` entry point for the same host upload:

```lean (name := ffiToken)
-- Compare the pure allocation primitive with the checked
-- IO upload.
open TorchLean.LibTorch in
#check @Buffer.ofFloatArray
open TorchLean.LibTorch in
#check @Buffer.ofFloatArrayIO
```
```leanOutput ffiToken
@Buffer.ofFloatArray : FloatArray → optParam Dtype Dtype.float32 → Buffer
```
```leanOutput ffiToken
@Buffer.ofFloatArrayIO : FloatArray → optParam Dtype Dtype.float32 → IO Buffer
```

In both signatures the dtype argument selects binary32 or binary64 storage. Nothing in the first
signature expresses allocation effects or release-sensitive identity, and for an ordinary pure
function, common subexpression elimination can reuse the result of identical calls. But
`ofFloatArray` is marked `@[never_extract]`: in the pinned Lean compiler this suppresses both
closed-term extraction and common subexpression elimination for the primitive. So the declaration
already guards against those optimizations, though its pure type still doesn't express when the
upload happens, how allocation can fail, or when releasing a buffer is safe.

As shown above, the effectful entry point declares the native call itself in `IO`.

`ofFloatArrayIO a` describes an action. Executing it performs the upload at that point in the
surrounding `IO` sequence, and executing it again with the same host array allocates a separate
buffer. That allocation effect belongs to this native call, not to a pure upload evaluated inside
`pure`. And the borrowed host array remains available to the caller.

Failure is also part of the call's result. LibTorch owns the device allocator's retry policy.
Native allocation failures return `IO.Error.resourceExhausted`, which the caller can handle through
the usual `IO` exception mechanism. In that case the call has returned no new buffer, and the
caller retains its host array and existing device buffers. This is the checked upload contract: the
`IO` type makes sequencing and failure visible, while correct allocation and ownership still depend
on the native implementation.

`releaseIO` is an `IO` extern, called only at an ownership boundary where no alias will be used
again. Removing a cache entry is insufficient if a tape still retains
the same buffer. Parameter mirrors and their recorded snapshots therefore use ordinary Lean
reference counting. After explicit release, the native finalizer stays safe because the
implementation nulls the pointer.

Both upload entry points copy the same host values into `dtype`, rounding at `.float32`, and the
checked `IO` entry point also exposes when allocation happens and how allocation failure returns to
the caller. These effects don't change the tensor's shape or intended contents. Nor do they
establish unique ownership of the returned handle: Lean references to a `Buffer` remain copyable.

After explicit release, every alias still refers to the native object whose payload was retired,
and copying a Lean reference doesn't bring the allocation back. The index in `Tensor α s`
constrains rank and dimensions, so it can't detect use of a stale handle. Explicit release must
therefore retire all usable aliases (including those retained by a tape), while reference-counted
snapshots keep the allocation alive until their last owner is gone.

# Workspaces And Backward

Some forward kernels produce an output plus intermediates needed by their VJP. TorchLean records
that ownership in a structure:

```
-- Retain the intermediates needed by backward alongside the
-- forward value.
structure WithWorkspace where
  value : Buffer
  workspace : Array Buffer := #[]
```

A tape node retains the workspace until backward has consumed it, and afterwards the cleanup is
threaded through a value that's still used:

```lean (name := ffiWorkspace)
-- The returned keep buffer makes cleanup part of a used
-- result dependency.
open TorchLean.LibTorch in
#check @Buffer.WithWorkspace.releaseWorkspaceThen
open TorchLean.LibTorch in
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
Callers thread that return value into their next computation, which keeps the release on a path
whose result is still needed. Nothing in the type forces callers to invoke the helper, though: they
already have the `keep` buffer and could use it directly. So cleanup remains an ownership protocol
that the caller must follow, using a result dependency to retain the release operation.

For long training runs this prevents two forms of growth:

- GPU allocations waiting for Lean external-object finalizers
- tape closures retaining workspaces after their VJP has run

Allocator counters report logical live and peak payload bytes, payload ownership counts, wrapper
counts, native allocated and reserved bytes, and driver free memory. Logical payload bytes can
count shared tensor storage more than once, and they need not include native temporaries. These
measurements help test the cleanup protocol (for example, live bytes shouldn't grow indefinitely
when a fixed workload repeatedly releases its temporaries), but they describe the measured run and
don't prove the absence of native leaks.

These two workspace helpers retire different ownership sets. `releaseWorkspaceThen` keeps the
result's own value alive while releasing its auxiliaries, and `releaseAllThen` also retires that
value, so the returned `keep` must belong to storage the caller can still use. Neither type proves
that those ownership sets are disjoint, so the surrounding operation must establish the protocol.

Allocator telemetry can help distinguish retained payloads from retained Lean wrappers. A wrapper
may remain alive after explicit payload release, while a device allocator may retain freed blocks
in a cache, so looking only at driver free memory can hide which layer retained memory. Live
payload bytes, wrapper counts, and cache behavior need to be interpreted together for the
particular repeated workload being inspected.

# Host Profiling

[LeanProfiler](https://github.com/lean-dojo/LeanProfiler) records named `IO` spans and writes both
a Perfetto-compatible trace and a JSON timing summary. You can wrap it around an existing TorchLean
runner once you add the separate LeanProfiler package dependency:

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

A host timer doesn't automatically measure asynchronous device work. TorchLean's integration can
wait for device completion before closing a span, and that number is completion latency (not
per-kernel CUDA or CUPTI time). A short host call may merely queue a much longer GPU operation, so
a short host span can hide device work. It can also mean the relevant host work wasn't
instrumented, so measure completion before assigning the cause.

# External Oracles And Certificates

Arbitrary-precision or interval tools outside Lean can also propose bounds. A typical workflow is:

```
Lean writes an exact query
        ↓
Arb / python-flint computes an enclosure
        ↓
Lean parses rational midpoint-radius data
        ↓
a checker validates the enclosure or treats it as oracle evidence
```

Keeping the midpoint and radius rational preserves the returned numbers without an extra binary64
conversion at ingestion, which avoids the conversion step that overflowed in the `1e999` example.
None of this verifies the external interval algorithm. That question belongs to the last step in
the diagram: if a proved checker replays and accepts the result, its theorem supplies the
corresponding proposition. If the result is only compared in a test, it remains oracle evidence.

That same producer and checker distinction applies to α,β-CROWN leaf dumps, PINN residual
artifacts, ODE enclosures, and geometry certificates. An external search may use a large
implementation, while a checker validates only the returned artifact. The schema determines what
information crosses the boundary, and the acceptance theorem determines what can be concluded from
it.

For each artifact, the report should identify the proposition the checker established and the
assumptions connecting its inputs to the requested computation. Exact midpoint-radius data
preserves the proposed interval, but the acceptance theorem must still justify its enclosure claim.

# Evidence For Boundary Contracts

A useful boundary report may say:

- *shape:* proved by a typed constructor
- *layout:* guarded by length and row-major checks
- *value:* compared by a regression suite
- *VJP:* delegated to a trusted external provider
- *provenance:* native symbol `torchlean_cuda_buffer_bmm_with_transpose`

A proof of shape safety supplies no arithmetic conclusion, and a source-file link identifies an
implementation without validating its behavior. Recording evidence field by field keeps each
claim attached to the check, test, or assumption that supports it.

# Boundary Failure Tests

The live blocks above exercise numeric parsing, region validation, graph acceptance, and graph
rejection during the page build, and the earlier capture command exercises the external process. To
also exercise a CUDA training call, run:

```terminal
# Exercise a separately configured CUDA runtime call.
scripts/lake.sh -Kcuda=true exe torchlean quickstart_mlp \
  --device cuda --steps 2 --show-backend
```

Then deliberately break one condition:

1. Add an unsupported PyTorch operation and observe import rejection.
2. Change a port shape in the JSON and observe `TorchLean.IR.Program.check` reject it.
3. Request CUDA from a build without LibTorch and observe runtime availability rejection.
4. Pass a wrong-size Q buffer to the LibTorch attention test and observe the Lean composition's
   size guard reject it before matrix multiplication.

JSON strings in the live blocks illustrate the first two rejection paths, but the last two depend
on the build configuration and native library: availability is checked when the CUDA session is
created, and buffer dimensions are checked at the Lean and native call boundaries. Testing these
failures confirms where execution stops when an input violates a boundary condition.
