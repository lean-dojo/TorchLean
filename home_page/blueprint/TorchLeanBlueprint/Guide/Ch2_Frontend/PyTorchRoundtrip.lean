import VersoManual
import NN.API
import NN.API.Seeded
import NN.Runtime.Autograd.Model.Layers.Seq
import NN.Runtime.PyTorch.Import.MLP
import NN.Runtime.PyTorch.Import.Core
import NN.Runtime.PyTorch.Export.Core
import NN.Spec.Models.Mlp
import TorchLeanBlueprint.Bib
import TorchLeanBlueprint.Roles

open Verso.Genre Manual
open Verso.Genre.Manual.InlineLean
open TorchLean
open Import.PyTorch (StateDict parseTensor)
open Import.PyTorch (loadStateDict? unwrapParams getTensor?)

#doc (Manual) "PyTorch Interop" =>
%%%
tag := "pytorch-roundtrip"
file := "PyTorch-Round-Trip"
%%%

Suppose we train a `2 → 3 → 1` MLP in PyTorch {Informal.citep pytorch2019}[] and want to evaluate
its learned weights in Lean. Four tensors must cross over: two weight matrices and two biases.
Their names and nested dimensions tell the importer which field each tensor should fill.

The importer reconstructs known model families with specified parameter layouts. It does not
reconstruct arbitrary `nn.Module` objects, so the expected architecture must be supplied on the
Lean side.

The checked-in
{src "NN/Examples/Interop/PyTorch/Roundtrip.lean"}[round-trip program]
wires three family-specific importers: an MLP, a convolutional network, and a transformer encoder.
The shared JSON core parses named nested arrays, while each family chooses its accepted keys and
expected tensor shapes. Python supplies a named payload, and Lean either reconstructs that typed
family record or rejects it.

Named Lean blocks are checked while this page is built. Shell and Python transcripts are recorded
runs with separate reproduction commands; they are not executed automatically by the guide build.

# Round-Trip Contract

The round trip has five steps:

1. Lean defines the expected model family.
2. Lean exports matching PyTorch code.
3. Python trains or modifies the weights.
4. Python writes a named tensor payload.
5. The chosen Lean family importer looks up its required names and parses every nested array at the
   statically expected shape.

If any check fails, TorchLean rejects the artifact before constructing a model.

The shape of the boundary is intentionally closer to a `state_dict` contract than to Python object
serialization. PyTorch users are used to seeing names such as:

```
layer1.weight
layer1.bias
layer2.weight
layer2.bias
```

Each name must map to a particular parameter in the Lean family record.

The bridge delegates loading PyTorch's own serialization format to PyTorch, then exports the
required tensor payload as JSON. Nested numeric arrays keep the Lean parser small and make the
name-to-parameter mapping inspectable. The file is larger than a binary payload, however, and its
numbers are decimal text rather than stored bit patterns. The
{ref "fp32-soundness"}[float chapter] is where exact bit-level claims live; this chapter is about
names and shapes.

The expected architecture gives each name a job. For the `2 → 3 → 1` MLP, the first weight must
contain three output rows with two input coefficients each; its bias contains three values. The
second weight has one row of three coefficients, followed by one output bias. A file with four
arrays of the right total sizes could still attach them to the wrong meanings. Looking up names
and parsing their particular nested shapes establishes a more useful contract than flattening all
values and consuming them in file order.

The contract ends at model state. Learning rate, optimizer momentum, random-generator state, and
the training data are not reconstructed by these four arrays. That is enough to evaluate the
family's forward pass. Resuming the same training trajectory would require a broader artifact and
an explicit account of how each additional state component is restored.

# MLP Parameter Round Trip

The MLP family exposes the full exchange in four tensors. Export its Python modules from the
repository root:

```terminal
# Generate the MLP family files expected by the companion
# Python script.
lake exe torchlean pytorch_roundtrip --model mlp --action export
```

```
Exported MLP PyTorch files under NN/Examples/Interop/PyTorch/MLP/.
```

That step writes two readable reference modules, `TestMLP_PyTorch.py` and
`TestMLP_WithWeights.py`, whose class matches what the Lean importer expects to see back:
`Linear(2, 3)`, `ReLU`, `Linear(3, 1)`.

Step three and four are one Python script,
{src "NN/Examples/Interop/PyTorch/MLP/train_mlp.py"}[`train_mlp.py`]:

```terminal
# Train the one-pair MLP example and write its named
# parameter payload.
python3 NN/Examples/Interop/PyTorch/MLP/train_mlp.py
```

```
Model info: TestMLP(2→3→1, ReLU)
Initial output: tensor([[-0.0945]], grad_fn=<AddmmBackward0>)

Starting training...
Epoch 0, Loss: 1.197846
Epoch 50, Loss: 0.000000
Epoch 100, Loss: 0.000000
Epoch 150, Loss: 0.000000
Final output: tensor([[1.0000]], grad_fn=<AddmmBackward0>)
Saved trained weights to .../NN/Examples/Interop/PyTorch/MLP/mlp.json
Original (fresh init) output: tensor([[-0.0945]], grad_fn=<AddmmBackward0>)
Training improved output by: 1.094462
```

This run fits one input-target pair, $`x=(0.5,0.8)` and $`y=1`, using two hundred SGD steps at
learning rate $`0.1`. The fit supplies a modified parameter payload whose prediction can be
checked after import; it provides no evidence of generalization. The script calls
`torch.manual_seed(0)` before initialization, and repeating it in the recorded environment
regenerates the same `mlp.json`.

Step five reads the file back:

```terminal
# Load the exported parameter values into the expected Lean
# MLP family.
lake exe torchlean pytorch_roundtrip --model mlp --action import
```

```terminal +output
== MLP import example ==
Loaded: NN/Examples/Interop/PyTorch/MLP/mlp.json
Output (native tensor operations, Float):
[1.000000]
```

PyTorch finished training at `tensor([[1.0000]])`. Lean, reading only the JSON file and running its
own forward pass, prints `[1.000000]`. The imported parameters reproduce the observed output at
the displayed precision. One rounded output is not a proof that the
programs compute the same function on every input.

The complete exchanged payload is:

```
{
  "params": {
    "layers.0.weight": [
      [0.028833430260419846, 0.10097555816173553],
      [-0.08230451494455338, -0.07359390705823898],
      [-0.03851543739438057, 0.026815736666321754]
    ],
    "layers.0.bias": [
      0.057182908058166504, 0.07928895205259323, -0.00887440424412489
    ],
    "layers.2.weight": [
      [0.11024449020624161, -0.030221307650208473, -0.01965653896331787]
    ],
    "layers.2.bias": [0.9832008481025696]
  },
  "meta": {
    "format": "TorchLean.MLP",
    "pytorch_keys": "layers.*",
    "dtype": "float32"
  }
}
```

The output bias `layers.2.bias` is about $`0.983`. With one training point, that bias alone could
fit the target while the other parameters remained fixed; the fit does not identify a unique
parameter vector.

The decimal strings retain enough significant digits for finite binary32 values because Python's
`json.dump` prints the shortest decimal that round-trips the underlying binary64 value that
`tolist()` produced from a binary32 tensor. This serialization can preserve those numerical
values, while `meta.dtype` separately records their source dtype. The importer still constructs
host `Float` tensors; source dtype metadata does not select the arithmetic of the Lean forward pass.

The retained transcript's last label says `Training improved output by`. The current script labels
that same absolute change `Output change from initialization`. It is a change in a prediction,
not a validation score: moving a prediction by a larger amount is only helpful relative to the
chosen target. The initial and final values above make that interpretation possible without relying
on the older label.

# Import Validation

The importer is strict about required tensors, and it checks them before constructing the typed
family record used by the example.

The checks actually performed are:

- the root is a JSON object, optionally with a `params` object;
- every required family-specific key exists under one of the accepted naming conventions;
- every scalar leaf is a JSON number;
- every nested array has exactly the length required by the expected Lean shape.

JSON object order is irrelevant: the importer looks up names and constructs a fixed typed record.
The family loaders validate a supplied `meta.dtype` or per-tensor dtype metadata as float32.
`meta.format` is not an architecture proof, and extra parameter keys are ignored. Absent dtype
metadata is accepted for the older bare-payload format. Imported
numbers become host `Float` values, so this format does not preserve exact binary32 payload bits.

## Parser Examples

The two helpers that do the work are small and total. Their types say what they promise:

```lean (name := ptrTypes)
-- The requested shape is retained in the successful parse
-- result.
#check @parseTensor
#check @getTensor?
```

```leanOutput ptrTypes
parseTensor : (s : Shape) → Lean.Json → Option (Tensor Float s)
```

```leanOutput ptrTypes
getTensor? : StateDict → String → (s : Shape) → Option (Tensor Float s)
```

The shape argument to `parseTensor` appears in its result type. There is no
value of type `Tensor Float [3, 2]` that holds two rows of three, so the returned type records that
layout. The parser must still be reviewed for how it maps each
JSON position into the tensor; the type alone would also allow a reordered or fabricated tensor.

Take a payload in the same wrapper format the Python script writes:

```lean (name := ptrPayload)
-- Exercise the low-level wrapper helpers independently of
-- family dtype validation.
def ptrPayload : String :=
  r#"{ "params":
       { "layers.0.weight": [[1, 2], [3, 4], [5, 6]],
         "layers.0.bias": [0.5, -0.5, 0.25] },
     "meta": { "format": "TorchLean.MLP" } }"#

def ptrDict : Option StateDict := do
  let json ← (Lean.Json.parse ptrPayload).toOption
  return unwrapParams (← loadStateDict? json)
```

`unwrapParams` is what lets both accepted layouts work: a bare object of tensors, or an object with
a `params` field. It merges the remaining top-level fields, so `meta` stays visible to any loader
that wants to read provenance, while parameter entries win on a name collision. Metadata must never
be able to replace a tensor.

Now ask for the two tensors at the shapes the MLP family expects:

```lean (name := ptrGood)
-- Read each required name at its own expected nested shape.
#eval ptrDict.bind fun d =>
  getTensor? d "layers.0.weight" [3, 2]

#eval ptrDict.bind fun d => getTensor? d "layers.0.bias" [3]
```

```leanOutput ptrGood
some [[1.000000, 2.000000], [3.000000, 4.000000], [5.000000, 6.000000]]
```

```leanOutput ptrGood
some [0.500000, -0.500000, 0.250000]
```

Three ways to break it, and all three are caught:

```lean (name := ptrBad)
-- The same JSON, requested at the transposed shape.
#eval (ptrDict.bind fun d =>
  getTensor? d "layers.0.weight" [2, 3]).isSome

-- A misspelled key.
#eval (ptrDict.bind fun d =>
  getTensor? d "layers.0.wieght" [3, 2]).isSome

-- A leaf that is not a number.
#eval (parseTensor [3]
  (.arr #[.num 1, .str "nan", .num 3])).isSome
```

```leanOutput ptrBad
false
```

```leanOutput ptrBad
false
```

```leanOutput ptrBad
false
```

The first rejection compares the nested array structure with the requested shape. The payload has
three rows of length two, so it cannot satisfy a request for `[2, 3]`, even though both shapes
contain six values. No separate `"shape"` annotation is needed to make this check.

The other two failures need no dimension mismatch: one misspells `weight`, and the other supplies
a string where a JSON number is required. `"nan"` fails because it is a string.

These examples deliberately call the low-level dictionary and tensor helpers. They do not call
`loadWeights?`, which performs wrapper dtype validation for the family loaders. In particular,
`ptrPayload` demonstrates retaining a `meta.format` field while parsing tensors; its success is
not evidence that an arbitrary metadata object would pass a complete MLP load. Keeping this layer
visible helps locate a rejection before blaming the array parser.

## Import Error Diagnostics

Transposing `layers.0.weight` in `mlp.json` to two rows of three causes the family importer to
reject it, but the command does not identify the offending tensor:

```terminal +output
error: Failed to load MLP state dict
```

The exit code is 1. Deleting a required key or replacing a numeric leaf with the string `"nan"`
produces the same message. The current diagnostic distinguishes success from failure but does not
localize the cause.

The reason is visible in
{src "NN/Runtime/PyTorch/Import/MLP.lean"}[the MLP loader].
It is written in the `Option` monad, and it tries the two accepted key conventions in sequence:

```
-- Try each complete naming convention as one alternative.
tryKeys "fc1.weight" "fc1.bias" "fc2.weight" "fc2.bias" <|>
  tryKeys "layers.0.weight" "layers.0.bias"
          "layers.2.weight" "layers.2.bias"
```

`<|>` on `Option` cannot tell you why the left branch failed, because `none` carries no
information. Every distinct failure inside either branch arrives at the caller as the same absent
value, and the caller can only report that the load failed.

A diagnostic variant returning `Except String` could retain the missing key, expected shape, or
invalid leaf. The current `Option` interface discards that information, so inspecting the payload
is necessary to distinguish these failures.

## PyTorch Load Errors

PyTorch checks required keys and shapes when loading a `state_dict`. With
`torch.nn.Sequential(Linear(2,3), ReLU(), Linear(3,1))` and a checkpoint whose
first weight has been transposed, `load_state_dict` raises:

```
RuntimeError: Error(s) in loading state_dict for Sequential:
  size mismatch for 0.weight: copying a param with shape
  torch.Size([2, 3]) from checkpoint, the shape in current model
  is torch.Size([3, 2]).
```

and with a key removed:

```
RuntimeError: Error(s) in loading state_dict for Sequential:
  Missing key(s) in state_dict: "0.bias".
```

Called with `strict=False`, the same load returns
`_IncompatibleKeys(missing_keys=['0.bias'], unexpected_keys=[])` instead of raising, and the
parameter simply keeps its previous value.

The checks produce different kinds of results:

:::table +header
*
  * Property
  * PyTorch `load_state_dict`
  * TorchLean family importer
*
  * Input format
  * mapping, often obtained with `torch.load`
  * JSON object of nested arrays
*
  * Reading it requires
  * a Python interpreter
  * a Lean parser of a few dozen lines
*
  * Shape check
  * at load time, per parameter
  * at parse time, against the Lean type
*
  * Missing key
  * `RuntimeError`, or `missing_keys` when not strict
  * whole load returns `none`
*
  * Failure message
  * names the key and both shapes
  * one sentence, no localization
*
  * Element type
  * copied into destination dtype by default
  * decimal text becomes host `Float`
*
  * Result
  * a mutable module object
  * an immutable typed record
:::

A loaded TorchLean family record retains its parameter shapes in its type. Later graph lowering
and theorem statements can use those shapes directly. This guarantee concerns the record's
structure; it does not establish that a name or coordinate has the intended physical meaning.

The family loader combines all required lookups before returning a record. A missing bias
therefore cannot leave a freshly initialized Lean bias in place, as PyTorch's non-strict load
permits. Extra keys are still ignored, so success does not imply that every supplied field was used.

# Serialized Tensor Layouts

The imported record supplies this example's family-specific forward pass. Using it through another
model API requires an explicit adapter. Even before that step, the names must agree with one of
the chosen family's accepted conventions.

The MLP loader accepts two key conventions, `fc1.*`/`fc2.*` and `layers.0.*`/`layers.2.*`, because
PyTorch itself produces both. A module with named attributes gives `fc1.weight`; the same
architecture written as `nn.Sequential` gives `0.weight`, which becomes `layers.0.weight` once the
sequential block is held in an attribute called `layers`. The index gap between `0` and `2` is not
an error: position `1` is the `ReLU`, and activations have no parameters. The CNN and transformer
loaders have their own fixed key sets.

Here are both conventions carrying identical weights, plus a third payload that names one layer each
way:

```lean (name := ptrKeys)
open Import.PyTorch.MLP (load forward)

def ptrLinearKeys : String :=
  r#"{ "fc1.weight": [[1, 0], [0, 1]],
       "fc1.bias": [0, 0],
       "fc2.weight": [[1, 1]],
       "fc2.bias": [0.5] }"#

def ptrSeqKeys : String :=
  r#"{ "layers.0.weight": [[1, 0], [0, 1]],
       "layers.0.bias": [0, 0],
       "layers.2.weight": [[1, 1]],
       "layers.2.bias": [0.5] }"#

-- One layer named each way, in the same file.
def ptrMixedKeys : String :=
  r#"{ "fc1.weight": [[1, 0], [0, 1]],
       "fc1.bias": [0, 0],
       "layers.2.weight": [[1, 1]],
       "layers.2.bias": [0.5] }"#

def ptrRunMlp (s : String) : Option (Tensor Float [1]) := do
  let json ← (Lean.Json.parse s).toOption
  let p ← load 2 2 1 json
  return forward p [2.0, 3.0]

#eval ptrRunMlp ptrLinearKeys
#eval ptrRunMlp ptrSeqKeys
#eval ptrRunMlp ptrMixedKeys
```

```leanOutput ptrKeys
some [5.500000]
```
```leanOutput ptrKeys
some [5.500000]
```
```leanOutput ptrKeys
none
```

The identity weights and the `[1, 1]` output row determine the result directly: the hidden
vector is $`[2, 3]`, the ReLU leaves it alone, and the sum plus the $`0.5` bias is $`5.5`. Both
naming conventions construct the same tensors and produce this value.

The mixed payload fails because `load` is written as
`tryKeys "fc1.weight" ... <|> tryKeys "layers.0.weight" ...`, so each convention is tried as a
complete set. A payload that names its first layer `fc1` and its second `layers.2.weight` satisfies
neither alternative and is rejected, even though every tensor it needs is present under some name.
This loader requires a consistent complete naming convention. Supporting mixed conventions would
need an explicit conflict policy; shape checking cannot identify the intended parameter when two
candidate keys both have the expected shape.

This small exchange format handles known TorchLean families. Keep the full object in Python when a
training pipeline needs PyTorch's complete serialization graph, and export only the checked payload
that TorchLean understands.

The identity weights make the naming test easy to check by hand. They would not reveal an
accidental transpose of the first weight matrix, since transposing an identity changes nothing.
The transformer example below uses an asymmetric matrix to expose that mistake.

# Model Families

The three family examples exercise different parameter layouts and forward paths.

## MLP

The MLP contract consists of two weight matrices and two bias vectors. Their names and shapes
can be compared directly with the Python `state_dict` and Lean family record:

```terminal
# Run the complete MLP exchange using one family-specific
# schema.
lake exe torchlean pytorch_roundtrip --model mlp --action export
python3 NN/Examples/Interop/PyTorch/MLP/train_mlp.py
lake exe torchlean pytorch_roundtrip --model mlp --action import
```

Its forward pass runs on native tensor operations, which is why the import transcript above says
so.

## CNN

The CNN example adds convolutional weight layouts. Both sides must agree on which axes represent
output channels, input channels, and spatial kernel dimensions; matching the total number of
elements does not establish that agreement.

```terminal
# Exercise convolutional parameter layouts through the CNN
# exchange.
lake exe torchlean pytorch_roundtrip --model cnn --action export
python3 NN/Examples/Interop/PyTorch/CNN/train_cnn.py
lake exe torchlean pytorch_roundtrip --model cnn --action import
```

```terminal +output
== CNN import example ==
Loaded: NN/Examples/Interop/PyTorch/CNN/cnn.json
Output (executable `nn` module on CPU, Float):
[0.999962, 0.500014]
```

Here
{src "NN/Examples/Interop/PyTorch/CNN/train_cnn.py"}[`train_cnn.py`]
writes `NN/Examples/Interop/PyTorch/CNN/cnn.json`. The trained targets were $`1` and $`0.5`, and
the imported predictions are within about $`4\times10^{-5}` of those targets. Unlike the MLP path,
this family runs
through an ordinary executable `nn` module.

The residuals compare the imported model with its training targets. They do not compare CPU and
CUDA convolutions; this forward pass uses host `Float` on the CPU.

## Transformer (Encoder)

The transformer example adds four attention projections per block and normalization parameters.
The importer must know the name and orientation of each tensor
{Informal.citep transformer2017}[].

```terminal
# Exchange the encoder projections and normalization
# parameters.
lake exe torchlean pytorch_roundtrip --model transformer --action export
python3 NN/Examples/Interop/PyTorch/Transformer/train_transformer.py
lake exe torchlean pytorch_roundtrip --model transformer --action import
```

```terminal +output
== Transformer import example ==
Loaded: NN/Examples/Interop/PyTorch/Transformer/transformer_encoder.json
Output (executable `nn` module on CPU, Float):
[[0.999998, -0.999998]]
```

The companion
{src "NN/Examples/Interop/PyTorch/Transformer/train_transformer.py"}[`train_transformer.py`]
writes `NN/Examples/Interop/PyTorch/Transformer/transformer_encoder.json`. The same pattern scales
to larger families: Lean checks declared shapes and parameter layout, Python runs the training
loop, and the JSON payload transports only named parameters back across the boundary.

One caution that this family makes concrete: a projection matrix stored in the wrong orientation is
only caught by shape checking when the two dimensions differ. A square $`d\times d` attention
projection transposed by mistake still parses, as this two-dimensional example shows:

```lean (name := ptrSquare)
-- A 2x2 projection, and the same four numbers stored
-- with rows and columns swapped.
def ptrRowMajor : String :=
  r#"{ "params": { "w": [[1, 2], [3, 4]] } }"#

def ptrColMajor : String :=
  r#"{ "params": { "w": [[1, 3], [2, 4]] } }"#

def ptrReadW (s : String) :
    Option (Tensor Float [2, 2]) := do
  let json ← (Lean.Json.parse s).toOption
  let dict := unwrapParams (← loadStateDict? json)
  getTensor? dict "w" [2, 2]

-- Both parse. Shape checking cannot tell them apart.
#eval (ptrReadW ptrRowMajor).isSome
#eval (ptrReadW ptrColMajor).isSome

def ptrProbe : Tensor Float [2] := [1.0, 0.0]

#eval (ptrReadW ptrRowMajor).map (Tensor.matvec · ptrProbe)
#eval (ptrReadW ptrColMajor).map (Tensor.matvec · ptrProbe)
```

```leanOutput ptrSquare
true
```
```leanOutput ptrSquare
true
```
```leanOutput ptrSquare
some [1.000000, 3.000000]
```
```leanOutput ptrSquare
some [1.000000, 2.000000]
```

Both payloads parse, but matrix-vector multiplication produces different answers. Multiplying by
the first basis vector reads off a column, so the row-major payload gives $`[1, 3]` and the swapped
one gives $`[1, 2]`. Nothing here is a bug in `getTensor?`; the requested type was `[2, 2]` and both
payloads have that shape.

Shape checking detects incompatible dimensions; it cannot distinguish equally shaped orientation
conventions. Evaluating both sides on a shared input can expose such a mismatch. A
literal nested-array transpose of a non-square projection fails the shape check. Other value
permutations can preserve shape, so numerical checks remain useful for both square and non-square
parameters. Attention blocks are full of square
projections, which is exactly why this family is the one that makes the caution concrete.

# PyTorch Code Generation

The
{src "NN/Examples/DeepDives/TorchIRPyTorch.lean"}[IR export example]
does the reverse kind of work. It lowers a TorchLean model to the shared IR and emits runnable
PyTorch code for a curated set of architectures: `linear`, `mlp`, `sum`, `autoencoder`, `mha`,
`mha-mask`, and `transformer`.

```terminal
# Emit a standalone Python program for the seeded IR
# example.
lake exe torchlean torch_ir_pytorch --arch mlp > exported_model.py
python3 exported_model.py
```

The `mlp` architecture is three lines of TorchLean:

```
-- Fix the architecture before assigning seeded parameter
-- values.
def archMLP : nn.Builder (nn.Sequential [2] [1]) :=
  nn.Sequential![nn.linear 2 3, nn.relu, nn.linear 3 1]
```

and `nn.build 0 archMLP` turns that description into a model with concrete seeded parameters. The
emitted Python has no builder and no seed; the parameters are baked in as literals, and the forward
pass is the lowered IR written out one SSA binding at a time:

```
class TorchLeanMLP(nn.Module):
  def __init__(self):
    super().__init__()
    dtype = torch.float32
    self.const_2 = nn.Parameter(torch.tensor([
      float.fromhex('0x182a07c19a71ecp-54'),
      float.fromhex('0x1597e07783b3c0p-53'),
      float.fromhex('0x1f66b498260754p-53'),
      float.fromhex('0x1bf20b38c869acp-54'),
      float.fromhex('0x18795e185062c8p-53'),
      float.fromhex('-0x1ec171132474b8p-53')],
      dtype=dtype).reshape((3, 2)))
    self.register_buffer("const_5", torch.tensor(
      [0.0, 0.0, 0.0], dtype=dtype).reshape((3,)))
    self.const_11 = nn.Parameter(torch.tensor([
      float.fromhex('-0x10fbf4e603fdfdp-52'),
      float.fromhex('-0x170671f4bea0ecp-53'),
      float.fromhex('0x1accf23e183060p-53')],
      dtype=dtype).reshape((1, 3)))
    self.register_buffer("const_14", torch.tensor(
      [0.0], dtype=dtype).reshape((1,)))

  def forward(self, x):
    v0 = x
    v1 = v0.reshape((1, 2))
    v2 = self.const_2
    v3 = v2.transpose(0, 1)
    v4 = torch.matmul(v1, v3)
    v5 = self.const_5
    v6 = torch.broadcast_to(v5, (1, 3))
    v7 = v4 + v6
    v8 = v7.reshape((3,))
    v9 = torch.relu(v8)
    v10 = v9.reshape((1, 3))
    v11 = self.const_11
    v12 = v11.transpose(0, 1)
    v13 = torch.matmul(v10, v12)
    ...
```

The `transpose(0, 1)` calls are the orientation convention made explicit: TorchLean stores a linear
layer's matrix as `[out, in]` and the emitter writes the transpose that `x @ W.T` needs, rather
than silently reordering the numbers on the way out. The emitted file also trains itself when run,
which exercises forward, backward, and an optimizer step. In the excerpt, weight constants become
`nn.Parameter`s while the zero biases become registered buffers. A matching forward value
therefore does not establish that subsequent training updates the same set of parameters.
A CPU run
(`CUDA_VISIBLE_DEVICES=\"\" python3 exported_model.py`) reports:

```
loss 3.9950685501098633
```

The generated bindings let us follow one linear layer without guessing how Python interprets a
matrix. The input becomes a single row, the stored `[3, 2]` weight is transposed to `[2, 3]`, and
matrix multiplication produces three hidden features. Broadcasting the `[3]` bias adds one value
per feature before the row is reshaped back to a vector. Those transformations preserve the
intended feature order while adapting it to PyTorch's matrix interface.

That printed loss comes from the emitted file's training example. To compare its initial forward
pass with Lean, I'll use the same input and seeded parameters on both sides.

## Generated Model Forward Comparison

Evaluating both programs at the same input checks more than successful loading or matching
shapes. TorchLean can use the model's pure specification semantics for its half of the comparison.
Build the same architecture, take the seeded initial state, and apply the MLP specification forward
pass at $`x = (0.5, 0.8)`:

```lean (name := ptrLean)
-- Use the same seed and [out, in] weights as the exported
-- MLP.
def ptrArch : nn.Builder (nn.Sequential [2] [1]) :=
  nn.Sequential![nn.linear 2 3, nn.relu, nn.linear 3 1]

def ptrInput : Tensor Float [2] :=
  Tensor.ofFn fun i => [0.5, 0.8][i.val]!

def ptrLeanOut : Tensor Float [1] :=
  match Runtime.Autograd.Model.Layers.Seq.initState
      (m := nn.build 0 ptrArch) with
  | .cons w1 (.cons b1 (.cons w2 (.cons b2 .nil))) =>
      Examples.mlpForward (α := Float)
        { weights := w1, bias := b1 }
        { weights := w2, bias := b2 } ptrInput

#eval Runtime.Autograd.Model.Layers.Seq.initState
  (m := nn.build 0 ptrArch)

#eval ptrLeanOut
```

```leanOutput ptrLean
[[3, 2]:
 [[0.377565, 0.674790], [0.981287, 0.436648], [0.764815, -0.961113]],
 [3]:
 [0.000000, 0.000000, 0.000000],
 [1, 3]:
 [[-1.061513, -0.719537, 0.837518]],
 [1]:
 [0.000000]]
```

```leanOutput ptrLean
[-1.377817]
```

The four parameter tensors correspond to the four emitted constants in the same orientation.
Their six-decimal Lean display does not expose all stored bits. Using the exact generated literals,
the initial forward values at $`(0.5, 0.8)` are:

:::table +header
*
  * Where
  * Value
*
  * Lean, `Tensor Float`, seeded weights (display)
  * `-1.377817`
*
  * Emitted module, `torch.float32`
  * `-1.3778172731399536`
*
  * Same emitted parameters, NumPy float64
  * `-1.3778172042069539`
*
  * Emitted module with `dtype = torch.float64`
  * `-1.3778172042069539`
:::

The float32 row rounds the emitted parameters and inputs to binary32 and evaluates with binary32
operations. Its difference from the float64 rows therefore includes both conversion and arithmetic;
it does not isolate accumulation precision alone. Matching NumPy and PyTorch float64 outputs at
this point is a useful regression, not a theorem for arbitrary inputs or reduction schedules.

The emitter uses `Export.PyTorch.floatToPyString`, rather than the tensor display formatter. It
keeps short decimal strings only when they round-trip to the same bits; otherwise it emits an exact
hexadecimal Python expression. Small nonzero parameters therefore survive source generation:

```lean (name := ptrFmt)
-- Compare human-readable display with literals intended to
-- preserve stored values.
#eval toString (1e-9 : Float)
#eval Export.PyTorch.floatToPyString (1e-9 : Float)
#eval Export.PyTorch.floatToPyString (0.5 : Float)
```

```leanOutput ptrFmt
"0.000000"
```

```leanOutput ptrFmt
"float.fromhex('0x112e0be826d695p-82')"
```

```leanOutput ptrFmt
"0.500000"
```

Python evaluates the hexadecimal expression to the original binary64 value. The formatter also
preserves signed zero and spells infinities and NaN explicitly; it does not preserve NaN payloads.
An emitted `torch.tensor(..., dtype=torch.float32)` then intentionally rounds to its requested
storage dtype. Exact source literals preserve input values across serialization, but do not prove
that two frameworks use the same arithmetic, backward rules, or parameter/buffer classification.

Using the ordinary display string for `1e-9` would replace a nonzero parameter by zero. In
contrast, `0.5` already survives the short decimal path.

## JSON Transport And Bitwise Comparison

The JSON path has a separate numeric transport check. A PyTorch module with the architecture used
above was
initialized under `torch.manual_seed(7)` and dumped with `json.dumps`, which prints a float64 repr
of each float32 parameter:

```
# Keep the parameter payload fixed for the two arithmetic
# evaluations.
torch.manual_seed(7)
m = torch.nn.Sequential(torch.nn.Linear(2, 3), torch.nn.ReLU(), torch.nn.Linear(3, 1))
sd = m.state_dict()
payload = {"params": {
  "layers.0.weight": sd["0.weight"].tolist(),
  "layers.0.bias":   sd["0.bias"].tolist(),
  "layers.2.weight": sd["2.weight"].tolist(),
  "layers.2.bias":   sd["2.bias"].tolist()}}
```

The same run reports its prediction at $`(0.5,0.8)` two ways, once through the module in
`torch.float32` and once by evaluating the printed weights in NumPy `float64`, with the raw bit
pattern beside each value. Both bit strings below encode binary64 values; the first is the
binary32 result widened for reporting:

```
float32 forward : -0.052048876881599426 0xbfaaa62680000000
float64 forward : -0.05204887814426984  0xbfaaa6268ad8a440
```

Now the Lean half. The payload goes in verbatim, and the four tensors come out at the shapes the MLP
family expects:

```lean (name := ptrJson)
-- Import this explicit payload, then compare output bits as
-- well as rounded display.
def ptrTrained : String :=
  r#"{ "params":
   { "layers.0.weight":
     [[0.04938792809844017, -0.4259566068649292],
      [0.22515933215618134, 0.2218763530254364],
      [-0.37793222069740295, -0.10597917437553406]],
     "layers.0.bias":
     [-0.4142428934574127, 0.18347492814064026,
      -0.19047172367572784],
     "layers.2.weight":
     [[0.40560975670814514, 0.4098530411720276,
       0.05881491303443909]],
     "layers.2.bias": [-0.2461371123790741] } }"#

def ptrTrainedDict : Option StateDict := do
  let json ← (Lean.Json.parse ptrTrained).toOption
  return unwrapParams (← loadStateDict? json)

def ptrImported : Option (Tensor Float [1]) := do
  let d ← ptrTrainedDict
  let w1 ← getTensor? d "layers.0.weight" [3, 2]
  let b1 ← getTensor? d "layers.0.bias" [3]
  let w2 ← getTensor? d "layers.2.weight" [1, 3]
  let b2 ← getTensor? d "layers.2.bias" [1]
  return Examples.mlpForward (α := Float)
    { weights := w1, bias := b1 }
    { weights := w2, bias := b2 } ptrInput

/-- The bit pattern of a `Float`, in hex. Six printed
decimals cannot distinguish two nearby doubles, and this
comparison is about the last bit. -/
def ptrBits (x : Float) : String :=
  String.ofList (Nat.toDigits 16 x.toBits.toNat)

#eval ptrImported
#eval ptrImported.map fun y => ptrBits (y[0])
```

```leanOutput ptrJson
some [-0.052049]
```

```leanOutput ptrJson
some "bfaaa6268ad8a440"
```

Lean's output bits match the NumPy float64 result in this test. For lossless parameter transport,
the parser must recover each value written by `json.dumps`: a finite float32 value is exactly
representable in float64. Comparing the imported parameter bits directly would test that property
without relying on a forward pass, where unused or canceling parameters could conceal an error.

Against the widened float32 forward result, `bfaaa62680000000`, the two patterns share their first
nine hex digits, followed by zeros in the float32 row. That gap
does not come from source-literal truncation. Nor is it simply a final rounding of
the float64 answer, since the module rounded to binary32 after every operation rather than at
the end. Bounding such precision differences requires the operation hypotheses and error analysis in
{ref "fp32-soundness"}[the FP32 soundness chapter].

This bit equality is a result for one run. Even a three-term dot product can be sensitive to
cancellation, and a reference Lean loop, a NumPy BLAS call, and a fused CUDA kernel may accumulate
in different orders. {ref "spec-layer"}[The specification chapter] shows four numbers whose
sum already depends on that choice. The importer checks names and shapes and parses decimal numbers.
Bit preservation, finite-range
requirements, and evaluation parity require their own checks.

# Python Training And Parameter Export

The training loop in `train_mlp.py` uses PyTorch's standard loss, backward pass, and optimizer:

```
# Clear each iteration's gradients before differentiating
# the current loss.
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

x_train = torch.tensor([[0.5, 0.8]], dtype=torch.float32)
y_train = torch.tensor([[1.0]], dtype=torch.float32)

for epoch in range(200):
    optimizer.zero_grad()
    outputs = model(x_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()
```

The only TorchLean-aware part is the export, and it is a dictionary rebuild:

```
# Rename learned tensors into the exact keys accepted by the
# Lean family loader.
state_dict = model.state_dict()
new_format = {}
new_format['layers.0.weight'] = state_dict['fc1.weight'].tolist()
new_format['layers.0.bias'] = state_dict['fc1.bias'].tolist()
new_format['layers.2.weight'] = state_dict['fc2.weight'].tolist()
new_format['layers.2.bias'] = state_dict['fc2.bias'].tolist()
```

Those four lines implement the name mapping. A missing required key is rejected; a wrong mapping
that still supplies every required key at the expected shape may be accepted. Keep numerical
probes alongside the schema checks.

The optimizer loop clears accumulated gradients before each backward pass. Without that step, the
next update would use contributions retained from previous iterations as well as the current
loss. The export block runs after training and reads the resulting parameter values; it does not
serialize the autograd graph that produced them. Renaming `fc1.weight` to `layers.0.weight` changes
the transport key while preserving the nested numeric array. That explicit mapping is the place
to review which learned tensor fills each Lean field.

# Tensor Exchange Versus Model Import

It is useful to distinguish three interop layers:

- *tensor exchange* moves arrays between frameworks;
- *parameter import* fills a known TorchLean model family with checked weights;
- *semantic import* claims that a foreign program has the same meaning as a TorchLean graph.

DLPack belongs mostly to the first layer: it is a standard in-memory tensor exchange format used by
array and tensor libraries. It can help avoid extra copies, but it does not by itself tell Lean that
a Python model has the same architecture, parameter names, or proof semantics.

TorchLean's current round trip implements the second layer for three known families: the selected
loader checks required names and tensor shapes. Semantic import is stronger and requires an IR
denotation and a proof relating the foreign program to it. The numeric comparison in the previous
section checks one input; the
verification chapters are where a claim about all inputs in a region gets established.

# Interop Boundary Selection

:::table +header
*
  * If
  * Then
*
  * training belongs in Python, and named tensors should enter a known family
  * use the round trip
*
  * training state transitions must stay explicit and checkable
  * run the training loop in Lean
*
  * the model must be inspected as a graph or connected to a theorem
  * lower to IR
*
  * a reader needs to see a TorchLean model in PyTorch vocabulary
  * emit PyTorch from the IR
:::

An imported model can subsequently be lowered to IR or exported again; these choices can be combined.

# Guarantees And Limits

The family importers return typed parameters for a forward pass. When preparing an exchange,
keep these limits in view:

- arbitrary PyTorch training needs its own semantic or artifact bridge;
- the importer covers the supported artifact formats rather than the full PyTorch ecosystem;
- dtype metadata is validated when supplied, but architecture metadata, optimizer state, unlisted
  buffers, extra keys, and exact float bit patterns are not certified by these family loaders;
- malformed required tensors are rejected without a localized diagnostic, because the loader uses
  `Option`;
- emitted constants preserve finite binary64 values before the requested tensor dtype conversion;
- shape checking cannot distinguish two conventions that agree on shape, such as a transposed
  square matrix;
- binary32 behavior is handled by the floating point bridge rather than by the round trip format.

# References

- {Informal.citet torchlean2026}[] for the project overview and shared IR architecture.
- {Informal.citet pytorch2019}[] for PyTorch's design, including `state_dict` conventions.
- {Informal.citet transformer2017}[] for the encoder architecture used by the third family.
- PyTorch, [serialization notes](https://pytorch.org/docs/stable/notes/serialization.html), for
  `state_dict`-style save and load workflows.
- PyTorch, [`nn.Module`](https://pytorch.org/docs/stable/generated/torch.nn.Module.html), for
  parameter naming and module conventions.
- [DLPack documentation](https://dmlc.github.io/dlpack/latest/), for tensor exchange as distinct
  from model-family import.
