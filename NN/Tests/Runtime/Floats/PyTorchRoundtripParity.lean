/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Json
public import NN.Runtime.PyTorch.Import.CNN
public import NN.Runtime.PyTorch.Import.MLP
public import NN.Runtime.PyTorch.Export.MLP
public import NN.Examples.Interop.PyTorch.Roundtrip
public import NN.Runtime.PyTorch.Import.Transformer
public import NN.Tests.Runtime.Floats.Utils

/-!
# PyTorch Roundtrip Parity Checks

Compares PyTorch and TorchLean evaluation of the checked-in MLP, CNN and transformer weights on the
round-trip examples' fixed inputs. Separate checks cover parameter-wrapper precedence, dtype
metadata rejection, and generated class initialization and summaries. Numerical parity is skipped
when Python cannot import PyTorch; importer checks still run. These cases do not establish general
import/export equivalence.
-/

@[expose] public section

open Lean
open Spec TorchLean
open TorchLean TorchLean.Tensor
open Tests.Utils
open Tests.Floats.Utils

namespace Tests
namespace Floats
namespace PyTorchRoundtripParity

def workDir : System.FilePath :=
  TorchLean.External.Process.directory "pytorch_roundtrip_parity"

def parityScriptPath : System.FilePath :=
  workDir / "compute_parity.py"

def parityScript : String :=
  String.intercalate "\n"
    [ "import json"
    , "import math"
    , "from pathlib import Path"
    , "import torch"
    , "import torch.nn as nn"
    , "import torch.nn.functional as F"
    , ""
    , "ROOT = Path('NN/Examples/Interop/PyTorch')"
    , ""
    , "def params(path):"
    , "    return json.loads(path.read_text())['params']"
    , ""
    , "def run_mlp():"
    , "    p = params(ROOT / 'MLP/mlp.json')"
    , "    fc1 = nn.Linear(2, 3)"
    , "    fc2 = nn.Linear(3, 1)"
    , "    with torch.no_grad():"
    , "        fc1.weight.copy_(torch.tensor(p['layers.0.weight'], dtype=torch.float32))"
    , "        fc1.bias.copy_(torch.tensor(p['layers.0.bias'], dtype=torch.float32))"
    , "        fc2.weight.copy_(torch.tensor(p['layers.2.weight'], dtype=torch.float32))"
    , "        fc2.bias.copy_(torch.tensor(p['layers.2.bias'], dtype=torch.float32))"
    , "        x = torch.tensor([[0.5, 0.8]], dtype=torch.float32)"
    , "        return fc2(F.relu(fc1(x))).flatten().tolist()"
    , ""
    , "def run_cnn():"
    , "    p = params(ROOT / 'CNN/cnn.json')"
    , "    conv1 = nn.Conv2d(1, 2, 3, padding=1)"
    , "    conv2 = nn.Conv2d(2, 2, 3, padding=1)"
    , "    fc = nn.Linear(8, 2)"
    , "    with torch.no_grad():"
    , "        conv1.weight.copy_(torch.tensor(p['conv1.weight'], dtype=torch.float32))"
    , "        conv1.bias.copy_(torch.tensor(p['conv1.bias'], dtype=torch.float32))"
    , "        conv2.weight.copy_(torch.tensor(p['conv2.weight'], dtype=torch.float32))"
    , "        conv2.bias.copy_(torch.tensor(p['conv2.bias'], dtype=torch.float32))"
    , "        fc.weight.copy_(torch.tensor(p['fc.weight'], dtype=torch.float32))"
    , "        fc.bias.copy_(torch.tensor(p['fc.bias'], dtype=torch.float32))"
    , "        x = torch.arange(1, 65, dtype=torch.float32).reshape(1, 1, 8, 8)"
    , "        x = F.max_pool2d(F.relu(conv1(x)), 2, 2)"
    , "        x = F.max_pool2d(F.relu(conv2(x)), 2, 2)"
    , "        return fc(x.reshape(x.shape[0], -1)).flatten().tolist()"
    , ""
    , "class TinyMHA(nn.Module):"
    , "    def __init__(self, p):"
    , "        super().__init__()"
    , "        self.head_dim = 2"
    , "        self.q = nn.Linear(2, 2, bias=False)"
    , "        self.k = nn.Linear(2, 2, bias=False)"
    , "        self.v = nn.Linear(2, 2, bias=False)"
    , "        self.o = nn.Linear(2, 2, bias=False)"
    , "        with torch.no_grad():"
    , "            self.q.weight.copy_(torch.tensor(p['Wq'], dtype=torch.float32).t())"
    , "            self.k.weight.copy_(torch.tensor(p['Wk'], dtype=torch.float32).t())"
    , "            self.v.weight.copy_(torch.tensor(p['Wv'], dtype=torch.float32).t())"
    , "            self.o.weight.copy_(torch.tensor(p['Wo'], dtype=torch.float32).t())"
    , "    def forward(self, x):"
    , "        q, k, v = self.q(x), self.k(x), self.v(x)"
    , "        attn = torch.softmax(q @ k.transpose(-2, -1) / math.sqrt(self.head_dim), dim=-1)"
    , "        return self.o(attn @ v)"
    , ""
    , "def run_transformer():"
    , "    p = params(ROOT / 'Transformer/transformer_encoder.json')"
    , "    mha = TinyMHA(p)"
    , "    norm1 = nn.LayerNorm(2)"
    , "    fc1 = nn.Linear(2, 2)"
    , "    fc2 = nn.Linear(2, 2)"
    , "    norm2 = nn.LayerNorm(2)"
    , "    with torch.no_grad():"
    , "        fc1.weight.copy_(torch.tensor(p['W1'], dtype=torch.float32))"
    , "        fc1.bias.copy_(torch.tensor(p['b1'], dtype=torch.float32))"
    , "        fc2.weight.copy_(torch.tensor(p['W2'], dtype=torch.float32))"
    , "        fc2.bias.copy_(torch.tensor(p['b2'], dtype=torch.float32))"
    , "        norm1.weight.copy_(torch.tensor(p['norm1_gamma'], dtype=torch.float32))"
    , "        norm1.bias.copy_(torch.tensor(p['norm1_beta'], dtype=torch.float32))"
    , "        norm2.weight.copy_(torch.tensor(p['norm2_gamma'], dtype=torch.float32))"
    , "        norm2.bias.copy_(torch.tensor(p['norm2_beta'], dtype=torch.float32))"
    , "        x = torch.full((1, 1, 2), 1.5, dtype=torch.float32)"
    , "        x = norm1(x + mha(x))"
    , "        return norm2(x + fc2(F.relu(fc1(x)))).flatten().tolist()"
    , ""
    , "print(json.dumps({'mlp': run_mlp(), 'cnn': run_cnn(), 'transformer': run_transformer()}))"
    ]

def leanMlp : IO (Array Float) := do
  let y ← NN.Examples.Interop.PyTorch.Roundtrip.mlpOutput
  pure #[Tensor.getScalar y ⟨0, by decide⟩]

def leanCnn : IO (Array Float) := do
  let y ← NN.Examples.Interop.PyTorch.Roundtrip.cnnOutput
  pure #[Tensor.getScalar y ⟨0, by decide⟩, Tensor.getScalar y ⟨1, by decide⟩]

def leanTransformer : IO (Array Float) := do
  let y ← NN.Examples.Interop.PyTorch.Roundtrip.transformerOutput
  pure #[Tensor.get2 y ⟨0, by decide⟩ ⟨0, by decide⟩,
    Tensor.get2 y ⟨0, by decide⟩ ⟨1, by decide⟩]

def parseJson! (source : String) : IO Json :=
  match Json.parse source with
  | .ok json => pure json
  | .error error =>
      throw <| IO.userError s!"pytorch_roundtrip_parity: invalid test JSON: {error}"

/-- Check the wrapper rules before running the optional Python parity process. -/
def checkImporterBoundaries : IO Unit := do
  let collision ← parseJson! "{\"params\":{\"weight\":[1.0]},\"weight\":[9.0]}"
  let some weights := Import.PyTorch.loadWeights? collision
    | throw <| IO.userError "pytorch_roundtrip_parity: rejected valid wrapped weights"
  let some weight := Import.PyTorch.getTensor? weights "weight" [1]
    | throw <| IO.userError "pytorch_roundtrip_parity: failed to parse wrapped weight"
  unless Tensor.getScalar weight ⟨0, by decide⟩ == 1.0 do
    throw <| IO.userError "pytorch_roundtrip_parity: wrapper field shadowed a parameter"

  let float32 ← parseJson!
    "{\"params\":{\"weight\":[1.0]},\"meta\":{\"weight\":{\"dtype\":\"torch.float32\"}}}"
  if (Import.PyTorch.loadWeights? float32).isNone then
    throw <| IO.userError "pytorch_roundtrip_parity: rejected float32 metadata"

  let float64 ← parseJson!
    "{\"params\":{\"weight\":[1.0]},\"meta\":{\"weight\":{\"dtype\":\"torch.float64\"}}}"
  if (Import.PyTorch.loadWeights? float64).isSome then
    throw <| IO.userError "pytorch_roundtrip_parity: accepted unsupported float64 metadata"

/-- Execute the emitted class skeleton and both MLP variants, including their summary API. -/
def checkGeneratedPythonContract : IO Unit := do
  let source := Export.PyTorch.MLP.script
      (inputWidth := 2) (hiddenWidth := 3) (outputWidth := 2) "Fixture" ++ "\n" ++
    Export.PyTorch.base "Base" "Regression fixture" ++ "\n" ++
    String.intercalate "\n"
      [ "assert '_initialize_layers' in Base.__dict__"
      , "try:"
      , "    Base()"
      , "except NotImplementedError:"
      , "    pass"
      , "else:"
      , "    raise AssertionError('base initialization must require an override')"
      , "class Concrete(Base):"
      , "    def _initialize_layers(self):"
      , "        self.layer = nn.Linear(2, 2)"
      , "    def forward(self, x):"
      , "        return self.layer(x)"
      , "assert Concrete()(torch.zeros(1, 2)).shape == (1, 2)"
      , "for cls, count in [(Fixture, 3), (FixtureWithSoftmax, 4)]:"
      , "    model = cls()"
      , "    info = model.get_model_info()"
      , "    assert info['model_name'] == cls.__name__"
      , "    assert info['layer_count'] == count"
      , "    assert info['input_dim'] == 2 and info['hidden_dim'] == 3"
      , "    assert info['output_dim'] == 2"
      , "    assert info['input_shape'] == (2,) and info['output_shape'] == (2,)"
      , "    assert len(info['operation_types']) == count"
      , "    assert model(torch.zeros(1, 2)).shape == (1, 2)"
      , "    print_model_summary(model)"
      , "assert torch.allclose(FixtureWithSoftmax()(torch.zeros(1, 2)).sum(1), torch.ones(1))"
      ]
  let path := workDir / "generated_contract.py"
  IO.FS.writeFile path source
  let _ ← TorchLean.External.Process.run
    (ctx := "pytorch_generated_contract") (cmd := "python3") (args := #[path.toString])
    (cwd := some ".")
  pure ()

def run : IO Unit := do
  IO.println "pytorch_roundtrip_parity: begin"
  checkImporterBoundaries
  if !(← pythonHasTorch) then
    IO.println "pytorch_roundtrip_parity: skipped (python package `torch` not installed)"
    return ()
  IO.FS.createDirAll workDir
  checkGeneratedPythonContract
  IO.FS.writeFile parityScriptPath parityScript
  let out ← TorchLean.External.Process.run
    (ctx := "pytorch_roundtrip_parity")
    (cmd := "python3")
    (args := #[parityScriptPath.toString])
    (cwd := some ".")
  let pyJson ←
    match Json.parse out with
    | .ok j => pure j
    | .error e => throw (IO.userError s!"pytorch_roundtrip_parity: bad Python JSON: {e}\n{out}")
  let pyMlp ←
    match jsonFloatArrayField pyJson "mlp" with
    | .ok xs => pure xs
    | .error e => throw (IO.userError e)
  let pyCnn ←
    match jsonFloatArrayField pyJson "cnn" with
    | .ok xs => pure xs
    | .error e => throw (IO.userError e)
  let pyTransformer ←
    match jsonFloatArrayField pyJson "transformer" with
    | .ok xs => pure xs
    | .error e => throw (IO.userError e)
  assertArrayApprox "pytorch_roundtrip_parity: mlp" (← leanMlp) pyMlp
  assertArrayApprox "pytorch_roundtrip_parity: cnn" (← leanCnn) pyCnn
  assertArrayApprox "pytorch_roundtrip_parity: transformer" (← leanTransformer) pyTransformer
  IO.println "pytorch_roundtrip_parity: ok"

end PyTorchRoundtripParity
end Floats
end Tests
