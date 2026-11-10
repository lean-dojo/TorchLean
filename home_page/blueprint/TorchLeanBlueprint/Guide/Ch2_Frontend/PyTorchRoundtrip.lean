import VersoManual
import TorchLean.NN.LoRA
import TorchLean.NN.Autograd.Function
import TorchLean.NN.Autograd.Frontend
import TorchLean.NN.Autograd.Differential
import TorchLean.NN.Autograd.Complex
import TorchLean.NN.CLI.Training.Command
import TorchLean.NN.Data.Synthetic
import TorchLean.NN.Data.CausalLM
import TorchLean.NN.Models.KAN
import TorchLean.NN.Models.CNN
import TorchLean.NN.Models.ResNet
import TorchLean.NN.Models.UNet
import TorchLean.NN.Models.Recurrent
import TorchLean.NN.Models.CausalTransformer.LanguageModel
import TorchLean.NN.Models.Mamba
import TorchLean.NN.Models.MaskedPatchReconstructor
import TorchLean.NN.Models.Diffusion.NoisePredictor
import TorchLean.NN.Models.FNO
import TorchLean.NN.Models.PPO
import TorchLean.NN.Builder.MoE
import TorchLean.NN.Sequential.Training
import TorchLean.NN.RL.CLI
import TorchLean.NN.RL.PPO.ActorCritic
import TorchLean.NN.SelfSupervised
import TorchLean.NN.Text.Vocabulary
import TorchLean.NN.Text.GPT2BPE
import TorchLean.NN.Text.Generation
import TorchLean.NN.Trainer.Constructor
import TorchLean.NN.Trainer.Train
import TorchLean.NN.Data.Image
import TorchLean.NN.Builder
import TorchLean.NN.Sequential
import TorchLean.NN.PyTorch.Import.State
import TorchLean.NN.PyTorch.Export
import TorchLean.NN.Models.MLP
import TorchLeanBlueprint.Bib
import TorchLeanBlueprint.Roles
import TorchLean.NN.Session.Random
import TorchLean.Verify.Trainer

open Verso.Genre Manual
open Verso.Genre.Manual.InlineLean
open TorchLean
open Lean.Elab.Tactic.GuardMsgs.WhitespaceMode (lax)

#doc (Manual) "PyTorch Interop" =>
%%%
tag := "pytorch-roundtrip"
file := "PyTorch-Round-Trip"
%%%

Suppose we train a `2 → 3 → 1` MLP in PyTorch {Informal.citep pytorch2019}[] and want to evaluate
its learned weights in Lean. Four tensors must cross over: two weight matrices and two biases.
Each one travels under the name that `state_dict()` gives it, such as `fc1.weight`, together with
its dtype, its shape and the binary words of its elements.

`PyTorch.state` reads such a state dict into the `TensorPack` of any TorchLean model. We give
it the state shapes of the model and one PyTorch name per tensor, and it either returns the state
with every tensor in place or fails with an error. It does not rebuild the PyTorch module itself,
so the architecture is defined on the Lean side.
Captured graphs have their own adapter and go through the shared IR parser.

The checked-in
{src "TorchLeanExamples/Interop/PyTorch/Roundtrip.lean"}[round-trip program]
reads three models this way: an MLP, a convolutional network and a transformer encoder block.
Python writes their state dicts with {src "python/torchlean/state.py"}[`torchlean.state`], and the
importer is {src "TorchLean/NN/PyTorch/Import/State.lean"}[`PyTorch.state`].

Named Lean blocks are checked while this page is built. Shell and Python transcripts are recorded
runs with their own reproduction commands, and the guide build does not execute them.

# Checking a captured computation

TorchLean's shared IR describes tensor programs, not only neural networks. An ordinary tensor
function can have inputs, constants, casts, array operations and results without layers, trainable
parameters or a loss. Both capture adapters use this interface, and what they can import depends on
operation coverage, whether or not there's a model class.

From Python, the entry point is `torchlean.verify(model, *inputs)`. It captures a PyTorch function
or module, translates its supported operations to TorchLean IR and asks Lean to check its structure
and element types, so you don't export a graph by hand. Add
`checks=("structure", "types", "compare")` to compare CPU reference execution with the frontend on
that input. Casts, tensor promotions, broadcasts and binary constants are all written out in the
artifact, and linear parameters are loaded at their declared precision from complete binary words.

Both frontends share the ordered tensor interface. PyTorch capture reads the user's argument order
from the export graph signature, separately from lifted parameters and constants, and unused
arguments stay in that interface. Each argument keeps its own shape and dtype (the IR doesn't
require one global scalar type for the model).

Each requested check is named in the report, along with its scope. A sample comparison isn't an
all-input theorem, and an unsupported request can't silently become a shape check. Frontend
equivalence, external kernels and captured physical device/layout constraints stay outside these
checks. Besides PyTorch, the same entry point accepts `frontend="jax"` or a typed artifact
dictionary from another producer. See the
{src "home_page/examples/verify/index.md"}[captured-computation example] for runnable code.

Operation coverage follows {ref "registered-operators"}[the operator table], whose chapter says how
the capture and the JAX importer translate a framework's operators and which operators they refuse.
For the attributes, element types and inputs the capture accepts, an imported entry is computed by
its registered operator (the capture refuses the others). Each entry states its arithmetic, and in
places that arithmetic is also PyTorch's: binary16 and bfloat16 elementary functions are computed
in binary32 and rounded once, `cumsum` accumulates in PyTorch's accumulation type, and the tanh
GELU follows the order of the kernel's DEFAULT level. Where eager PyTorch departs from IEEE
754-2019, the IR keeps the IEEE result: `clamp_min(-0.0, 0)` is `+0.0` (PyTorch's kernel keeps
`-0.0`), `0.1 / x` is the correctly rounded quotient (PyTorch multiplies by the reciprocal), and
`relu` returns `+0.0` at `-0.0` (eager PyTorch returns `-0.0`). Both the
{src "TorchLean/NN/IR/README.md"}[IR guide] and the
{src "TorchLean/NN/PyTorch/README.md"}[PyTorch interop guide] list these rules operator by
operator.

When we're ready to write a theorem, add `export="Proofs/Model.lean"` to that same call. The file
embeds the capture document and defines `Model.program` , its `torchlean.ir` program. Lean checks
its decoding and shape/type/interface proofs when we open or compile the module (export itself
doesn't launch a compiler). We can then import it and prove further properties, reusing the tensor
semantics and existing operation lemmas, and those mathematical proofs are separate from the check
report. Export doesn't capture the Python function again, and it never replaces an existing file.
With a `.json` filename, export saves the interchange artifact for inspection instead.

Element descriptors preserve configured binary and decimal formats, integer and posit widths,
complex component types and named representations. Their parameters are validated by the shared
decoder, which doesn't select an implementation. Typed custom calls also carry shape and
element-type signatures, which the checker compares with the graph (executing one requires a
supplied implementation). TorchLean's default reference executes binary arithmetic and Boolean and
arbitrary-width integer operations: layout operations, integer arithmetic modulo `2 ^ width`,
comparisons and conversions. Other element families and unimplemented custom calls are rejected.

# Captured graphs from JAX

Weights are not the only useful artifact we can bring into Lean. A captured tensor computation
also tells us which operations use those weights. The Jaxpr adapter in
{src "python/torchlean/jaxpr.py"}[`jaxpr.py`] lowers a static tensor computation to the same
`torchlean.ir` artifact as the PyTorch capture, in a capture document whose frontend is `jaxpr`.
The Lean entry point is `TorchLean.JAX.load`, from `TorchLean.NN.JAX`. It checks the
frontend and loads the artifact with `TorchLean.IR.load`, and `TorchLean.JAX.load_ok` states that
the program it returns passes `Program.check .reference` and its payload passes `Payload.check`.

Use `torchlean.verify(function, *inputs, frontend="jax")` directly: it captures the Jaxpr and
translates it to TorchLean IR inside the call. Arguments may have different shapes and element
types, and the capture preserves their positional order.
Each input step of the artifact records its argument's position, and the shared loader checks the
complete interface, including unused arguments. Execution requires the declared number, shapes and
types of tensor arguments. This interface is independent of the model architecture. Nested input
containers require their own explicit type rules.

For example, `jnp.sum(x * x)` becomes an input, an elementwise multiplication and a scalar sum. In
the companion {src "TorchLeanExamples/Interop/JAX.lean"}[Lean example], we prove that this graph's
real-valued evaluation is unchanged by any permutation of a vector's entries, for any vector
length. We first connect the graph evaluator to the tensor formula and then reuse the finite-sum
reindexing theorem. That's a property of the graph's Lean semantics, not a proof of the Python
translation or the XLA kernels.

On the Python side, the companion also saves capture documents for a small MLP, sums, broadcasts,
reshapes, a batched contraction and several returned tensors, and it reports unsupported entries
separately. To inspect those captures on their own, run
`python3 TorchLeanExamples/Interop/JAX/export.py /tmp/torchlean-jax-examples` and then
`scripts/lake.sh exe torchlean jaxpr /tmp/torchlean-jax-examples/mlp.json`. The runner loads each
document, runs its program on the recorded arguments under the reference runtime and compares every
output's element type, shape and interchange words with JAX's recorded output.

For mixed precision, a Jaxpr conversion is an explicit cast step that rounds once. Every constant
carries its complete words, whether binary, Boolean or integer. The reference runtime executes
each binary step with FloatLib at its declared precision, and `TorchLean.IR.Binary.cast_value`
connects its conversions to FloatLib's directed-rounding specification. Try `mixed_precision.json`,
`float16_sum.json` or `bfloat16_sum.json`: the runner compares their recorded results by
interchange word. These examples do not establish equivalence with every XLA execution or
reduction schedule. The reference runtime also executes Boolean and fixed-width integer
operations. Complex values, dynamic control flow and contractions with a separate accumulation
type are rejected.

JAX division imports as IEEE division. XLA's CPU code multiplies by the reciprocal of a broadcast
divisor, and of a constant divisor under `jax.jit`, which changes rounding and exceptional values.
When that rewrite reproduces every differing element, the failed `compare` entry lists it in
`consistent_with`. JAX's mean, LayerNorm and softmax expansions import. Square root imports as
IEEE's root at every operand, `Op.sqrt` at `.raw`.

# Round-Trip Contract

A round trip has five steps:

1. Lean defines the model, and with it the shapes of the model's state.
2. Python defines the same architecture as an `nn.Module`.
3. Python trains or modifies the weights.
4. `torchlean.state.write` writes the module's `state_dict()` as a state document.
5. `PyTorch.state` reads the document into the model's `TensorPack`. It checks every name,
   dtype and shape and decodes every word.

If any check fails, `PyTorch.state` returns an error and no state.

The boundary is PyTorch's own `state_dict` contract. A module with the attributes `fc1` and `fc2`
names its four tensors:

```
fc1.weight
fc1.bias
fc2.weight
fc2.bias
```

The caller gives the importer these names, one for each tensor of the Lean state and in the order
of that state. Tensor `i` of the state is read from the entry named `names[i]`.

Python writes each element as the integer value of its binary word. A float64 entry holds the
64-bit pattern of each element in row-major order, and a float32 entry the 32-bit pattern. Lean
decodes the words with `Checkpoint.Encoding` at the element type of the state, so negative zeros,
subnormals and every NaN payload that the element type keeps come back unchanged. Nothing is
converted on the way. The {ref "fp32-soundness"}[float chapter] covers what the arithmetic then
does with these values.

`PyTorch.stateDict` writes the same document from a Lean state. The theorem
`PyTorch.state_stateDict` proves that reading it back gives the original state when the
names have no repeats, there is one name per tensor and every word decodes to the scalar it
encodes. `PyTorch.state_stateDict_scalar` discharges the last condition for every element
type with an `Kernel.Scalar` instance, which covers `Float`, `Float32` and FloatLib's configured
binary formats such as `Binary64`. Both theorems are about the Lean side. The Python side is
checked by running it. The
{src "TorchLeanTest/Interop/PyTorchRoundtrip.lean"}[parity test] reads the three
checked-in state dicts with `torchlean.state.read` into the PyTorch modules and with
`PyTorch.state` into the Lean models, and compares the outputs of the two sides.

The Lean model gives each name a job. For the `2 → 3 → 1` MLP, the state shapes are `[3, 2]`,
`[3]`, `[1, 3]` and `[1]`: three output rows of two input coefficients, three biases, one row of
three coefficients and one output bias. A file with four tensors of the right sizes could
attach them to the wrong meanings. Reading each tensor by its name and comparing its stored shape
with the shape of the model rules out a reordered file and a transposed matrix of a different
shape. It can't rule out a wrong name list, so the names should come from the PyTorch module.

The contract ends at model state. The learning rate, the optimizer momentum, the random-generator
state and the training data are not part of the module's `state_dict()`. The state is enough to
evaluate the model's forward pass. Resuming the same training trajectory would need a larger
artifact and a rule for restoring each extra component.

# MLP Parameter Round Trip

The MLP shows the whole exchange in four tensors. Steps three and four are one Python script,
{src "TorchLeanExamples/Interop/PyTorch/MLP/train_mlp.py"}[`train_mlp.py`], run from the
repository root:

```terminal
# Train the one-pair MLP example and write its state dict.
python3 TorchLeanExamples/Interop/PyTorch/MLP/train_mlp.py
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
Original (fresh init) output: tensor([[-0.0945]], grad_fn=<AddmmBackward0>)
Output change from initialization: 1.094462
Saved the state dict to .../TorchLeanExamples/Interop/PyTorch/MLP/mlp.json
```

This run fits one input-target pair, $`x=(0.5,0.8)` and $`y=1`, using two hundred SGD steps at
learning rate $`0.1`. The fit gives a modified state whose prediction can be checked after import.
It provides no evidence of generalization. The script seeds the initialization with
`torch.manual_seed(0)`. With PyTorch 2.13 on the CPU, a second run writes a byte-identical
`mlp.json`, and the same holds for the CNN and transformer scripts.

Step five reads the file into the Lean MLP and evaluates it at the training input:

```terminal
# Read the state dict into the Lean MLP and run it.
scripts/lake.sh exe torchlean pytorch_roundtrip --model mlp
```

```terminal +output
== MLP import example ==
Loaded: TorchLeanExamples/Interop/PyTorch/MLP/mlp.json
Output (executable `nn` module on CPU, Float):
[1.000000]
```

The program builds the ordinary `nn` model
`nn.Sequential![nn.linear 2 3, nn.op .relu, nn.linear 3 1]` and instantiates it with the imported
state in place of its seeded initialization. PyTorch finished training at `tensor([[1.0000]])`.
Lean reads the JSON file, runs its own forward pass and prints `[1.000000]`. The full
value is 0.9999999112061961, the word `0x3fefffffd05445c4`, and the trained module converted to
float64 computes the same word at this input. One output is not a proof that the two programs
compute the same function on every input.

Here's the complete document:

```
{"format": "torchlean.state", "entries": {
  "fc1.weight": {"dtype": "float64", "shape": [3, 2],
    "words": [4583967891858128896, 4591940476440936448, 13813967122843303936,
              13813339457396408320, 13809083517394485248, 4583386331274543104]},
  "fc1.bias": {"dtype": "float64", "shape": [3],
    "words": [4588401746585321472, 4590377791781863424, 13799641404382642176]},
  "fc2.weight": {"dtype": "float64", "shape": [1, 3],
    "words": [4592608373383364608, 13807739957135015936, 13804694869882437632]},
  "fc2.bias": {"dtype": "float64", "shape": [1], "words": [4607031105491566592]}
}}
```

The script writes the float64 copy `model.double()`, so every entry has the dtype `float64`, the
dtype of Lean's `Float`. Widening binary32 to binary64 is exact, so these words hold the trained
float32 values. The output bias `fc2.bias` is the word `4607031105491566592`, which is
`0x3fef7661a0000000`, the binary64 value 0.9832008481025696. Its low 29 bits are zero, as for every
widened binary32 value. With one training point, that bias alone could fit the target while the
other parameters stayed fixed, and the fit does not identify a unique parameter vector.

The dtype of a document must equal the dtype of the Lean element type. Reading a `float32`
document into `TensorPack Float` fails, and the importer never converts. A state of `Float32`
elements reads the document that `state.write` writes from the float32 module itself.

`Output change from initialization` measures the absolute change in the prediction. Its usefulness
depends on the chosen target. The initial and final values above show that this prediction moved
toward `1`.

# Import Validation

`PyTorch.state shapes names document` returns either the state or the first error it
meets. It checks, in this order:

- There is one name for each tensor of the state.
- The keys of the document are `format` and `entries`, and its format is `torchlean.state`.
- The names have no repeats, and they are the keys of `entries`.
- Each entry has the keys `dtype`, `shape` and `words`.
- The dtype of each entry is the dtype of the element type, such as `float64` for `Float`.
- The shape of each entry is the shape of its state tensor.
- Each entry has one word per element, and every word decodes at the element type.

The last four checks run entry by entry, in the order of the names. JSON object order is
irrelevant, since the importer looks up names. An entry that the model does not name is an error,
as in PyTorch's strict `load_state_dict`, so a successful read uses every tensor of the document.
A word decodes when the element type gives it back unchanged. `Float.ofBits` and `Float32.ofBits`
turn every NaN word into `0x7ff8000000000000` and `0x7fc00000`, so a `Float` or `Float32` state
reads no other NaN word. FloatLib's `Binary64` and `Binary32` keep every word, NaN payloads
included.

## Importer Examples

The type of the importer says what a successful read returns:

```lean (name := ptrTypes)
-- The state shapes in the arguments appear in the result
-- type.
#check @PyTorch.state
```

```leanOutput ptrTypes (whitespace := lax)
@PyTorch.state : {α : Type} →
  [inst : Storage α] →
    [Checkpoint.Encoding α] →
      (shapes : List Shape) → List String → Lean.Json → Except String (TensorPack α shapes)
```

The shape list `shapes` appears in the result type `TensorPack α shapes`, so a successful read has
these shapes in this order. The type alone would also allow a reordered or fabricated state.
`PyTorch.state_stateDict` is the theorem about values: reading the document of a state
gives back that state.

Take the state of a `2 → 2 → 1` MLP with identity first weights, zero first biases, the second
weight row `[1, 1]` and the output bias `0.5`, written as a float64 state document:

```lean (name := ptrPayload)
-- The binary64 words of 1.0, 0.0 and 0.5.
def ptrPayload : String :=
  r#"{"format": "torchlean.state", "entries": {
  "fc1.weight": {"dtype": "float64", "shape": [2, 2],
    "words": [4607182418800017408, 0, 0,
      4607182418800017408]},
  "fc1.bias": {"dtype": "float64", "shape": [2],
    "words": [0, 0]},
  "fc2.weight": {"dtype": "float64", "shape": [1, 2],
    "words": [4607182418800017408, 4607182418800017408]},
  "fc2.bias": {"dtype": "float64", "shape": [1],
    "words": [4602678819172646912]}
}}"#

def ptrShapes : List Shape :=
  [[2, 2], [2], [1, 2], [1]]

def ptrNames : List String :=
  ["fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias"]

def ptrRead (names : List String) (s : String) :
    Except String (TensorPack Float ptrShapes) := do
  PyTorch.state ptrShapes names (← Lean.Json.parse s)

#eval ptrRead ptrNames ptrPayload
```

```leanOutput ptrPayload (whitespace := lax)
Except.ok [[2, 2]:
 [[1.000000, 0.000000], [0.000000, 1.000000]],
 [2]:
 [0.000000, 0.000000],
 [1, 2]:
 [[1.000000, 1.000000]],
 [1]:
 [0.500000]]
```

Each tensor of the state carries its shape in its type. `Pack.get` returns tensor `i` at the
type `Tensor Float (ptrShapes.get i)`, which is the type the forward pass needs:

```lean (name := ptrRun)
-- Evaluate the imported state at the input (2, 3).
def ptrRun (names : List String) (s : String) :
    Except String (Tensor Float [1]) := do
  let p ← ptrRead names s
  return MLP.forward (α := Float)
    { weights := p.get ⟨0, by decide⟩,
      bias := p.get ⟨1, by decide⟩ }
    { weights := p.get ⟨2, by decide⟩,
      bias := p.get ⟨3, by decide⟩ }
    [2.0, 3.0]

#eval ptrRun ptrNames ptrPayload
```

```leanOutput ptrRun
Except.ok [5.500000]
```

The identity weights and the `[1, 1]` output row determine the result: the hidden vector is
$`[2, 3]`, the ReLU leaves it unchanged, and the sum plus the bias $`0.5` is $`5.5`.

Each check has its own message. Five altered copies of the document and a short name list show
them:

```lean (name := ptrBad)
-- Print the error of a read, or ok.
def ptrCheck (names : List String) (s : String) : IO Unit :=
  match ptrRead names s with
  | .ok _ => IO.println "ok"
  | .error e => IO.println e

-- The second weight stored transposed, as [2, 1].
#eval ptrCheck ptrNames
  (ptrPayload.replace "[1, 2]" "[2, 1]")

-- A misspelled name.
#eval ptrCheck ptrNames
  (ptrPayload.replace "fc2.bias" "fc2.bais")

-- A float32 document for a state of Float elements.
#eval ptrCheck ptrNames
  (ptrPayload.replace "float64" "float32")

-- One word for two elements.
#eval ptrCheck ptrNames
  (ptrPayload.replace "\"words\": [0, 0]" "\"words\": [0]")

-- A NaN word that Float.ofBits would change.
#eval ptrCheck ptrNames
  (ptrPayload.replace "4602678819172646912"
    "9218868437227405313")

-- Three names for four tensors.
#eval ptrCheck (ptrNames.take 3) ptrPayload
```

```leanOutput ptrBad (whitespace := lax)
PyTorch state: size mismatch for fc2.weight: copying a param with
shape torch.Size([2, 1]) from checkpoint, the shape in current model
is torch.Size([1, 2]).
```

```leanOutput ptrBad (whitespace := lax)
PyTorch state: Missing key(s) in state_dict: "fc2.bias".
Unexpected key(s) in state_dict: "fc2.bais".
```

```leanOutput ptrBad (whitespace := lax)
PyTorch state: dtype mismatch for fc1.weight: copying a param with
dtype float32 from checkpoint, the dtype in current model is float64.
```

```leanOutput ptrBad (whitespace := lax)
PyTorch state: the entry fc1.bias of shape torch.Size([2]) has 1
word(s), expected 2.
```

```leanOutput ptrBad (whitespace := lax)
PyTorch state: fc2.bias[0]: Checkpoint: noncanonical native Float NaN
encoding
```

```leanOutput ptrBad (whitespace := lax)
PyTorch state: 3 name(s) for 4 tensor(s).
```

The first rejection compares the stored shape with the shape of the model, as PyTorch does. Both
shapes hold two values, so the words alone could not tell them apart. The misspelled name gives
both halves of PyTorch's strict message: the model's name is missing and the document's name is
unexpected. The float32 document fails at `fc1.weight`, the first name, since an entry's dtype is
compared ahead of its shape and words. The word `9218868437227405313` is `0x7ff0000000000001`, a
NaN with the payload 1. `Float.ofBits` would turn it into `0x7ff8000000000000`, so the importer
rejects the document.

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

with a key removed:

```
RuntimeError: Error(s) in loading state_dict for Sequential:
  Missing key(s) in state_dict: "0.bias".
```

and with an extra key:

```
RuntimeError: Error(s) in loading state_dict for Sequential:
  Unexpected key(s) in state_dict: "0.extra".
```

Called with `strict=False`, the load without `0.bias` raises no error. It returns
`_IncompatibleKeys(missing_keys=['0.bias'], unexpected_keys=[])`, and the parameter keeps its
value. A float64 state dict loads into the float32 module without a message, and the module stays
float32: PyTorch copies each tensor into the destination dtype.

Side by side, the two checks produce different kinds of results:

:::table +header
*
  * Property
  * PyTorch `load_state_dict`
  * `PyTorch.state`
*
  * Input format
  * mapping, often obtained with `torch.load`
  * `torchlean.state` JSON document of binary words
*
  * Reading it requires
  * a Python interpreter
  * a Lean reader and the scalar decoders
*
  * Shape check
  * at load time, per parameter
  * at load time, against the shapes in the Lean type
*
  * Missing key
  * `RuntimeError`, or `missing_keys` when not strict
  * an error naming every missing key
*
  * Unexpected key
  * `RuntimeError`, or `unexpected_keys` when not strict
  * an error naming every unexpected key
*
  * Failure message
  * names the key and both shapes
  * PyTorch's wording for key and shape errors, after `PyTorch state:`
*
  * Element type
  * copied into the destination dtype by default
  * must equal the dtype of the Lean element type, never converted
*
  * Result
  * a mutable module object
  * an immutable `TensorPack` whose type holds the shapes
:::

The shapes in the type of the state are available to graph lowering and to theorems about the
model. They describe the structure of the state and say nothing about whether a name or coordinate
has the intended physical meaning.

A read that fails returns no state, so a missing bias can't leave a freshly initialized Lean bias
in place, as PyTorch's non-strict load permits.

# Parameter Names

The importer reads the names the caller gives it and has no naming convention of its own. The
names come from the PyTorch module. A module with named attributes gives `fc1.weight`, and the same
architecture written as `nn.Sequential` gives `0.weight`. The index gap between `0` and `2` is not
an error: position `1` is the `ReLU`, and activations have no parameters. Nesting adds prefixes, so
a sequential block held in an attribute called `layers` gives `layers.0.weight`.

Here is the identity document under the names of `nn.Sequential`, read with its own names and with
the names of the module with attributes:

```lean (name := ptrKeys)
-- The same tensors under the names that nn.Sequential
-- gives them.
def ptrSeqPayload : String :=
  (ptrPayload.replace "fc1." "0.").replace "fc2." "2."

def ptrSeqNames : List String :=
  ["0.weight", "0.bias", "2.weight", "2.bias"]

#eval ptrRun ptrSeqNames ptrSeqPayload
#eval ptrCheck ptrNames ptrSeqPayload
```

```leanOutput ptrKeys
Except.ok [5.500000]
```

```leanOutput ptrKeys (whitespace := lax)
PyTorch state: Missing key(s) in state_dict: "fc1.weight", "fc1.bias",
"fc2.weight", "fc2.bias". Unexpected key(s) in state_dict: "0.bias",
"0.weight", "2.bias", "2.weight".
```

With matching names the read gives the same state and the same output $`5.5`. With the other
names every key is missing and every entry unexpected, and the message lists all of them. A
document whose names follow no single convention needs no special rule either, since every name in
the caller's list must be present and every entry must be named.

The identity weights make the naming test easy to check by hand. They would not reveal an
accidental transpose of the first weight matrix, since transposing an identity changes nothing.
The transformer example below shows that mistake with an asymmetric matrix.

# Example Models

The three examples exercise different state layouts and forward paths. Each runs the ordinary
executable `nn` model with the imported state in place of its seeded initialization.

## MLP

The MLP state consists of two weight matrices and two bias vectors, with the same names and shapes
in the Python `state_dict` and in the Lean model's state.

```terminal
# Train the MLP and read its state dict into the Lean MLP.
python3 TorchLeanExamples/Interop/PyTorch/MLP/train_mlp.py
scripts/lake.sh exe torchlean pytorch_roundtrip --model mlp
```

The second command prints the `[1.000000]` of the MLP round trip.

## CNN

The CNN example adds convolutional weight layouts. Both sides must agree on which axes hold output
channels, input channels and the spatial kernel dimensions, and matching the total number of
elements does not establish that agreement. PyTorch stores `conv1.weight` with the shape
`[2, 1, 3, 3]`, output channels first, and the Lean convolution's state has the same shape.

```terminal
# Train the CNN and read its state dict into the Lean CNN.
python3 TorchLeanExamples/Interop/PyTorch/CNN/train_cnn.py
scripts/lake.sh exe torchlean pytorch_roundtrip --model cnn
```

```terminal +output
== CNN import example ==
Loaded: TorchLeanExamples/Interop/PyTorch/CNN/cnn.json
Output (executable `nn` module on CPU, Float):
[0.999962, 0.500014]
```

Here
{src "TorchLeanExamples/Interop/PyTorch/CNN/train_cnn.py"}[`train_cnn.py`]
writes `TorchLeanExamples/Interop/PyTorch/CNN/cnn.json`, six entries from `conv1.weight` to
`fc.bias`. The training targets are $`1` and $`0.5`, and the imported predictions are within about
$`4\times10^{-5}` of those targets. The trained module in float64 computes 0.9999618741736017 and
0.5000140904038091 at the same input. Lean's first output is 0.9999618741736018, one unit in the
last place above, and its second output has the same word.

The residuals compare the imported model with its training targets and with PyTorch's float64
evaluation. They do not compare CPU and CUDA convolutions, since this forward pass uses host
`Float` on the CPU.

## Transformer (Encoder)

The transformer example adds four attention projections per block and normalization parameters.
Each of these tensors needs the right name and orientation
{Informal.citep transformer2017}[].

```terminal
# Initialize the encoder and read its state dict into the
# Lean encoder block.
python3 TorchLeanExamples/Interop/PyTorch/Transformer/export_transformer.py
scripts/lake.sh exe torchlean pytorch_roundtrip --model transformer
```

```terminal +output
== Transformer import example ==
Loaded: TorchLeanExamples/Interop/PyTorch/Transformer/transformer_encoder.json
Output (executable `nn` module on CPU, Float):
[[0.999998, -0.999998]]
```

The companion
{src "TorchLeanExamples/Interop/PyTorch/Transformer/export_transformer.py"}[`export_transformer.py`]
writes `TorchLeanExamples/Interop/PyTorch/Transformer/transformer_encoder.json` from seeded
initialization and does not train the encoder. Its twelve entries run from
`layers.0.mha.q_weight` to `layers.0.norm2.bias`. The four attention projections are bias-free
`(input, output)` matrices applied as `x @ weight`, the layout of TorchLean's executable attention
layer, so both sides store the same numbers in the same order. The module in float64 computes
0.9999983489980666 and -0.9999983489980665. Lean's first output has the same word, and its second
output is the exact negation of its first, one unit in the last place from PyTorch's.

The shape check catches a projection matrix stored in the wrong orientation when its two dimensions
differ, and misses it when they are equal. A module that computed its projections with `nn.Linear`
would store `(output, input)` matrices of the same square shape, and reading them would transpose
every projection without an error. This two-dimensional example shows the effect:

```lean (name := ptrSquare)
-- A 2x2 matrix, and the same four numbers stored with rows
-- and columns swapped: the words of 1, 2, 3, 4 and of
-- 1, 3, 2, 4.
def ptrRowMajor : String :=
  r#"{"format": "torchlean.state", "entries": {
  "w": {"dtype": "float64", "shape": [2, 2],
    "words": [4607182418800017408, 4611686018427387904,
              4613937818241073152, 4616189618054758400]}}}"#

def ptrColMajor : String :=
  r#"{"format": "torchlean.state", "entries": {
  "w": {"dtype": "float64", "shape": [2, 2],
    "words": [4607182418800017408, 4613937818241073152,
              4611686018427387904, 4616189618054758400]}}}"#

def ptrReadW (s : String) :
    Except String (TensorPack Float [[2, 2]]) := do
  PyTorch.state [[2, 2]] ["w"] (← Lean.Json.parse s)

-- Both read. The shape check cannot tell them apart.
#eval (ptrReadW ptrRowMajor).isOk
#eval (ptrReadW ptrColMajor).isOk

def ptrProbe : Tensor Float [2] := [1.0, 0.0]

def ptrApply (s : String) :
    Except String (Tensor Float [2]) := do
  let p ← ptrReadW s
  return Tensor.matvec (p.get ⟨0, by decide⟩) ptrProbe

#eval ptrApply ptrRowMajor
#eval ptrApply ptrColMajor
```

```leanOutput ptrSquare
true
```
```leanOutput ptrSquare
true
```
```leanOutput ptrSquare
Except.ok [1.000000, 3.000000]
```
```leanOutput ptrSquare
Except.ok [1.000000, 2.000000]
```

Both documents read, but matrix-vector multiplication produces different answers. Multiplying by
the first basis vector reads off a column, so the row-major document gives $`[1, 3]` and the
swapped one gives $`[1, 2]`. The model's shape is `[2, 2]` and both documents have it, so the
importer accepts both.

Evaluating both sides on a shared input can expose such a mismatch, and the parity test does that
for the three examples. Other permutations of the values also preserve shape, so numerical
comparison is useful for square and non-square parameters alike. Attention blocks have many square
projections, such as the four `[2, 2]` matrices of this encoder.

# PyTorch Code Generation

Going the other way, the {src "TorchLeanExamples/DeepDives/TorchIRPyTorch.lean"}[IR export example]
lowers a TorchLean model to the shared IR and emits runnable PyTorch code for a curated set of
architectures (`linear`, `mlp`, `sum`, `autoencoder`, `mha`, `mha-mask` and `transformer`).

```terminal
# Emit a standalone Python program for the seeded IR
# example.
scripts/lake.sh exe torchlean torch_ir_pytorch --arch mlp > exported_model.py
python3 exported_model.py
```

We can write the MLP with three layers:

```
-- Fix the architecture before assigning seeded parameter
-- values.
def mlp : nn.Builder (nn.Sequential [2] [1]) :=
  nn.Sequential![nn.linear 2 3, nn.op .relu, nn.linear 3 1]
```

and `nn.build 0 mlp` turns that description into a model with concrete seeded parameters. There's
no builder and no seed in the emitted Python. Its parameters are baked in as literals, and its
forward pass is the lowered IR written out one SSA binding at a time:

```
class TorchLeanMLP(nn.Module):
  def __init__(self):
    super().__init__()
    dtype = torch.float64
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
      float.fromhex('-0x100e8202b6cceap-52'),
      float.fromhex('-0x116eede4c8f307p-53'),
      float.fromhex('-0x18d0ae05a0e7a0p-56')],
      dtype=dtype).reshape((1, 3)))
    self.register_buffer("const_14", torch.tensor(
      [0.0], dtype=dtype).reshape((1,)))

  def forward(self, x):
    v0 = x
    v1 = torch.reshape(v0, shape=(1, 2))
    v2 = self.const_2
    v3 = torch.permute(v2, dims=(1, 0))
    v4 = torch.matmul(v1, v3)
    v5 = self.const_5
    v6 = v5.reshape((1, 3,)).expand((1, 3))
    v7 = torch.add(v4, v6)
    v8 = torch.reshape(v7, shape=(3,))
    v9 = torch.relu(v8) + 0.0
    v10 = torch.reshape(v9, shape=(1, 3))
    v11 = self.const_11
    v12 = torch.permute(v11, dims=(1, 0))
    v13 = torch.matmul(v10, v12)
    ...
```

Those `torch.permute(..., dims=(1, 0))` calls make the orientation convention visible: TorchLean
stores a linear layer's matrix as `[out, in]`, and the emitter writes the transpose that `x @ W.T`
needs (it doesn't silently reorder the numbers on the way out). When run, the emitted file also
trains itself, which exercises forward, backward and an optimizer step. In the excerpt, weight
constants become `nn.Parameter`s while the zero biases become registered buffers, so a matching
forward value doesn't establish that subsequent training updates the same set of parameters. A CPU
run (`CUDA_VISIBLE_DEVICES="" python3 exported_model.py`, PyTorch 2.8.0 on an Intel Xeon) reports:

```
loss 0.8064298643505659
```

The module computes in binary64, the emitter's default dtype, and its training example draws `x` and
`y` with `torch.randn` after `torch.manual_seed(0)`. The printed value is the mean squared error of
the first forward pass, taken before the optimizer step. Its low bits can depend on the order in
which the CPU's matrix kernel rounds the products and sums.

With the generated bindings we can follow one linear layer without guessing how Python interprets a
matrix. First the input becomes a single row, then the stored `[3, 2]` weight is transposed to
`[2, 3]`, and matrix multiplication produces three hidden features. Broadcasting the `[3]` bias adds
one value per feature before the row is reshaped back to a vector. Those transformations preserve
the intended feature order while adapting it to PyTorch's matrix interface.

That printed loss comes from the emitted file's training example. To compare its initial forward
pass with Lean, we use the same input and seeded parameters on both sides.

## Generated Model Forward Comparison

Evaluating both programs at the same input checks more than successful loading or matching
shapes. TorchLean can use the model's pure specification semantics for its half of the comparison.
Build the same architecture, take the seeded initial state, and apply the MLP specification forward
pass at $`x = (0.5, 0.8)`:

```lean (name := ptrLean)
-- Use the same seed and [out, in] weights as the exported
-- MLP.
def ptrArch : nn.Builder (nn.Sequential [2] [1]) :=
  nn.Sequential![nn.linear 2 3, nn.op .relu, nn.linear 3 1]

def ptrInput : Tensor Float [2] :=
  Tensor.ofFn fun i => [0.5, 0.8][i.val]!

def ptrLeanOut : Tensor Float [1] :=
  match nn.Sequential.initState
      (m := nn.build 0 ptrArch) with
  | .cons w1 (.cons b1 (.cons w2 (.cons b2 .nil))) =>
      MLP.forward (α := Float)
        { weights := w1, bias := b1 }
        { weights := w2, bias := b2 } ptrInput

#eval nn.Sequential.initState
  (m := nn.build 0 ptrArch)

#eval ptrLeanOut
```

```leanOutput ptrLean
[[3, 2]:
 [[0.377565, 0.674790], [0.981287, 0.436648], [0.764815, -0.961113]],
 [3]:
 [0.000000, 0.000000, 0.000000],
 [1, 3]:
 [[-1.003542, -0.544791, -0.096934]],
 [1]:
 [0.000000]]
```

```leanOutput ptrLean
[-1.188799]
```

Those four parameter tensors correspond to the four emitted constants in the same orientation, but
their six-decimal Lean display doesn't expose all stored bits. Using the exact generated literals,
the initial forward values at $`(0.5, 0.8)` are:

:::table +header
*
  * Where
  * Value
*
  * Lean, `Tensor Float`, seeded weights (display)
  * `-1.188799`
*
  * Emitted module with `dtype = torch.float32`
  * `-1.1887991428375244`
*
  * Same emitted parameters, NumPy float64
  * `-1.1887990892460092`
*
  * Emitted module, `torch.float64`
  * `-1.1887990892460092`
:::

In the float32 row, the emitted parameters and inputs are rounded to binary32 and evaluated with
binary32 operations. So that row's difference from the float64 rows includes both conversion and
arithmetic, and it doesn't isolate accumulation precision alone. Matching NumPy and PyTorch float64
outputs at this point is a useful regression, not a theorem for arbitrary inputs or reduction
schedules.

Instead of the tensor display formatter, the emitter uses `PyTorch.floatLiteral`. That
function keeps short decimal strings only when they round-trip to the same bits, and otherwise it
emits an exact hexadecimal Python expression. So small nonzero parameters survive source
generation:

```lean (name := ptrFmt)
-- Compare human-readable display with literals intended to
-- preserve stored values.
#eval toString (1e-9 : Float)
#eval PyTorch.floatLiteral (1e-9 : Float)
#eval PyTorch.floatLiteral (0.5 : Float)
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
preserves signed zero and spells out infinities and NaN, though it doesn't preserve NaN payloads.
An emitted `torch.tensor(..., dtype=torch.float32)` then intentionally rounds to its requested
storage dtype. Exact source literals preserve input values across serialization, but they don't
prove that two frameworks use the same arithmetic, backward rules or parameter/buffer
classification.

Using the ordinary display string for `1e-9` would replace a nonzero parameter by zero. In
contrast, `0.5` already survives the short decimal path.

## JSON Transport And Bitwise Comparison

The state document gets its own numeric transport check. A PyTorch module with the architecture
used above is initialized under `torch.manual_seed(7)`, evaluated at $`(0.5, 0.8)` in
`torch.float32`, converted to float64 with `m.double()`, evaluated again and written:

```
# Seed the parameters, evaluate the module in float32 and in
# float64, then write the float64 copy.
torch.manual_seed(7)
m = torch.nn.Sequential(torch.nn.Linear(2, 3), torch.nn.ReLU(), torch.nn.Linear(3, 1))
with torch.no_grad():
    y32 = m(torch.tensor([[0.5, 0.8]])).item()
    m.double()
    y64 = m(torch.tensor([[0.5, 0.8]], dtype=torch.float64)).item()
state.write("seed7.json", m)
```

Printed with their binary64 words in hex, the float32 result widened for the report, the two
results are:

```
float32 forward : -0.052048876881599426 0xbfaaa62680000000
float64 forward : -0.05204887814426984  0xbfaaa6268ad8a440
```

NumPy, evaluating the widened parameters in float64, gives the same float64 word. The Lean half
reads `seed7.json` under the names of `nn.Sequential` and runs the MLP specification at the same
input:

```lean (name := ptrJson)
-- Read the seed-7 state dict, then compare the output word
-- as well as the rounded display.
def ptrTrained : String :=
  r#"{"format": "torchlean.state", "entries": {
  "0.weight": {"dtype": "float64", "shape": [3, 2],
    "words": [4587278371582377984, 13824717008836493312,
              4597280240165715968, 4597161958376996864,
              13823851878405898240, 13815673061841567744]},
  "0.bias": {"dtype": "float64", "shape": [3],
    "words": [13824505993335144448, 4595778401234583552,
              13819402524215476224]},
  "2.weight": {"dtype": "float64", "shape": [1, 3],
    "words": [4600978435714580480, 4601054875931901952,
              4588636943289417728]},
  "2.bias": {"dtype": "float64", "shape": [1],
    "words": [13821408081206050816]}
}}"#

def ptrImported : Except String (Tensor Float [1]) := do
  let json ← Lean.Json.parse ptrTrained
  let p ← PyTorch.state (α := Float)
    [[3, 2], [3], [1, 3], [1]]
    ["0.weight", "0.bias", "2.weight", "2.bias"] json
  return MLP.forward (α := Float)
    { weights := p.get ⟨0, by decide⟩,
      bias := p.get ⟨1, by decide⟩ }
    { weights := p.get ⟨2, by decide⟩,
      bias := p.get ⟨3, by decide⟩ }
    ptrInput

/-- The bit pattern of a `Float`, in hex. Six printed
decimals cannot distinguish two nearby doubles, and this
comparison is about the last bit. -/
def ptrBits (x : Float) : String :=
  String.ofList (Nat.toDigits 16 x.toBits.toNat)

#eval ptrImported
#eval ptrImported.map fun y => ptrBits (y[0])
```

```leanOutput ptrJson
Except.ok [-0.052049]
```

```leanOutput ptrJson
Except.ok "bfaaa6268ad8a440"
```

Lean's output word matches the float64 results of PyTorch and NumPy in this test. The importer
decodes each word to the `Float` with that word, so the four tensors hold the float32 parameters
of the module unchanged. Writing the Lean state back with `PyTorch.stateDict` and comparing
words tests the transport without a forward pass, where unused or canceling parameters could hide
an error.

Against the widened float32 result, `bfaaa62680000000`, the two words share their first nine hex
digits, and the rest of the float32 word is zero, as in every widened binary32 value. In this run
the float32 result is the float64 result rounded to binary32. The module gets there another way,
rounding every intermediate tensor to binary32, and on other inputs the two can differ. Bounding
such differences requires the operation hypotheses and error analysis in
{ref "fp32-soundness"}[the FP32 soundness chapter].

This word equality is a result for one run. Even a three-term dot product can be sensitive to
cancellation, and a reference Lean loop, a NumPy BLAS call and a LibTorch operation may accumulate
in different orders. {ref "spec-layer"}[The specification chapter] shows four numbers whose sum
depends on that choice. The importer checks names, dtypes and shapes and keeps every word.
Finite-range requirements and evaluation parity need their own checks.

# Python Training And State Export

The training loop in `train_mlp.py` uses PyTorch's standard loss, backward pass and optimizer:

```
# One SGD step per epoch on the single training pair.
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

The part that knows about TorchLean is one call:

```
# Write the float64 copy of the trained state dict.
out_path = THIS_DIR / "mlp.json"
state.write(out_path, model.double())
```

Nothing is renamed. The Lean program lists the PyTorch name of each state tensor in the order of
the Lean state, so `fc1.weight` keeps its name from the module to the importer.

Each step starts with `optimizer.zero_grad()`. Without that call, the backward pass would add the
gradient of the step's loss to the gradients accumulated by earlier steps. The export runs after
training and reads the resulting parameter values. It does not serialize the autograd graph that
produced them.

# Tensor Exchange Versus Model Import

It helps to separate three interop layers:

- *Tensor exchange* moves arrays between frameworks.
- *State import* fills the state of a TorchLean model with checked tensors.
- *Semantic import* claims that a foreign program has the same meaning as a TorchLean graph.

DLPack belongs mostly to the first layer. It is a standard in-memory tensor exchange format used
by array and tensor libraries. It can avoid extra copies, but it does not tell Lean that a Python
model has the same architecture, parameter names or proof semantics.

The round trip implements the second layer for any TorchLean model, given the PyTorch name of each
state tensor. Semantic import is stronger and requires an IR denotation and a proof relating the
foreign program to it. The numeric comparisons above each check one input, and the verification
chapters establish claims about all inputs in a region.

# Interop Boundary Selection

:::table +header
*
  * If
  * Then
*
  * training belongs in Python, and the trained state should enter a Lean model
  * use the round trip
*
  * every training step must be checkable in Lean
  * run the training loop in Lean
*
  * the model must be inspected as a graph or connected to a theorem
  * lower to IR
*
  * a reader needs to see a TorchLean model in PyTorch vocabulary
  * emit PyTorch from the IR
:::

An imported model can be lowered to IR or exported again, and these choices combine.

# Limits

- Arbitrary PyTorch training needs its own semantic or artifact bridge. The round trip carries the
  state that training produces.
- The importer reads `torchlean.state` documents. A `torch.save` checkpoint of a `state_dict`
  converts in one line, `state.write("model.json", torch.load("model.pt"))`, since `write` accepts
  a mapping from names to tensors as well as a module.
- One state has one element type, and every entry must carry its dtype.
  `BatchNorm2d(2).state_dict()` holds four `float32` entries and the `int64` counter
  `num_batches_tracked`, so its document fits no single state. A mapping that leaves the counter
  out writes the four `float32` entries.

# References

- {Informal.citet torchlean2026}[] for the project overview and shared IR architecture.
- {Informal.citet pytorch2019}[] for PyTorch's design, including `state_dict` conventions.
- {Informal.citet transformer2017}[] for the encoder architecture used by the third example.
- PyTorch, [serialization notes](https://pytorch.org/docs/stable/notes/serialization.html), for
  `state_dict`-style save and load workflows.
- PyTorch, [`nn.Module`](https://pytorch.org/docs/stable/generated/torch.nn.Module.html), for
  parameter naming and module conventions.
- [DLPack documentation](https://dmlc.github.io/dlpack/latest/), for tensor exchange as distinct
  from state import.
