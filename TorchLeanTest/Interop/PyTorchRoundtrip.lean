/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import Lean.Data.Json.Parser
public import TorchLean.NN.PyTorch.Export
public import TorchLeanExamples.Interop.PyTorch.Roundtrip
public import TorchLeanTest.Harness.Interop

/-!
# PyTorch Roundtrip Parity Checks

Compares PyTorch and TorchLean evaluation of the checked-in MLP, CNN and transformer state dicts on
the round-trip examples' fixed inputs, both in float64. PyTorch loads each state dict with
`torchlean.state.read` into the reference module of the script that wrote it, and TorchLean reads
it with `TorchLean.PyTorch.state` into the `nn` model of
`TorchLeanExamples/Interop/PyTorch/Roundtrip.lean` by the same names. A separate check covers the
initialization of the generated base class. When Python cannot import PyTorch, no check runs. The
round trip is tested on these three models, not proved.
-/

@[expose] public section

open Lean
open TorchLean.Tensor
open TorchLean.Test.Harness

namespace TorchLean.Test
namespace PyTorchRoundtrip

/-- Scratch directory of these checks, `.lake/build/torchlean_pytorch_roundtrip_parity`. -/
def workDir : System.FilePath :=
  Subprocess.directory "pytorch_roundtrip_parity"

/-- Where `run` writes `parityScript` before running it with `python3`. -/
def parityScriptPath : System.FilePath :=
  workDir / "compute_parity.py"

/-- The PyTorch side: each reference module in float64 with the state dict its script wrote,
evaluated on the fixed input of `TorchLeanExamples/Interop/PyTorch/Roundtrip.lean`. -/
def parityScript : String :=
  String.intercalate "\n"
    [ "import importlib.util"
    , "import json"
    , "from pathlib import Path"
    , "import sys"
    , ""
    , "sys.path.insert(0, str(Path('python').resolve()))"
    , "import torch"
    , "from torchlean import state"
    , ""
    , "ROOT = Path('TorchLeanExamples/Interop/PyTorch')"
    , ""
    , "def output(script, cls, fixture, x):"
    , "    path = ROOT / script"
    , "    spec = importlib.util.spec_from_file_location(path.stem, path)"
    , "    module = importlib.util.module_from_spec(spec)"
    , "    spec.loader.exec_module(module)"
    , "    model = getattr(module, cls)().double()"
    , "    model.load_state_dict(state.read(ROOT / fixture))"
    , "    model.eval()"
    , "    with torch.no_grad():"
    , "        return model(x).flatten().tolist()"
    , ""
    , "mlp = output('MLP/train_mlp.py', 'TestMLP', 'MLP/mlp.json',"
    , "             torch.tensor([[0.5, 0.8]], dtype=torch.float64))"
    , "cnn = output('CNN/train_cnn.py', 'TestCNN', 'CNN/cnn.json',"
    , "             torch.arange(1, 65, dtype=torch.float64).reshape(1, 1, 8, 8))"
    , "transformer = output('Transformer/export_transformer.py', 'TinyTransformerEncoder',"
    , "                     'Transformer/transformer_encoder.json',"
    , "                     torch.full((1, 1, 2), 1.5, dtype=torch.float64))"
    , "print(json.dumps({'mlp': mlp, 'cnn': cnn, 'transformer': transformer}))"
    ]

/-- The entry of `Examples.PyTorchRoundtrip.mlpOutput`, the Lean output of the MLP fixture at
`[0.5, 0.8]`. -/
def leanMLP : IO (Array Float) := do
  let y ← Examples.PyTorchRoundtrip.mlpOutput
  pure #[Tensor.getScalar y ⟨0, by decide⟩]

/-- The two entries of `Examples.PyTorchRoundtrip.cnnOutput`, the Lean output of the CNN fixture
at the input `1..64`. -/
def leanCNN : IO (Array Float) := do
  let y ← Examples.PyTorchRoundtrip.cnnOutput
  pure #[Tensor.getScalar y ⟨0, by decide⟩, Tensor.getScalar y ⟨1, by decide⟩]

/-- The two entries of `Examples.PyTorchRoundtrip.transformerOutput`, the Lean output of the
encoder fixture at `[[1.5, 1.5]]`. -/
def leanTransformer : IO (Array Float) := do
  let y ← Examples.PyTorchRoundtrip.transformerOutput
  pure #[Tensor.get2 y ⟨0, by decide⟩ ⟨0, by decide⟩,
    Tensor.get2 y ⟨0, by decide⟩ ⟨1, by decide⟩]

/-- Execute the emitted class skeleton. -/
def checkGeneratedPythonContract : IO Unit := do
  let source := PyTorch.imports ++ "\n" ++
    PyTorch.base "Base" "Regression fixture" ++ "\n" ++
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
      ]
  let path := workDir / "generated_contract.py"
  IO.FS.writeFile path source
  let _ ← Subprocess.run
    (ctx := "pytorch_generated_contract") (cmd := "python3") (args := #[path.toString])
    (cwd := some ".")
  pure ()

/-- Skips when Python cannot import PyTorch. Otherwise runs `checkGeneratedPythonContract` and
`parityScript`, and compares the MLP, CNN and transformer outputs of the script with `leanMLP`,
`leanCNN` and `leanTransformer` by `assertArrayApprox`. -/
def run : IO Unit := do
  IO.println "pytorch_roundtrip_parity: begin"
  if !(← pythonHasTorch) then
    IO.println "pytorch_roundtrip_parity: skipped (python package `torch` not installed)"
    return ()
  IO.FS.createDirAll workDir
  checkGeneratedPythonContract
  IO.FS.writeFile parityScriptPath parityScript
  let out ← Subprocess.run
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
  assertArrayApprox "pytorch_roundtrip_parity: mlp" (← leanMLP) pyMlp
  assertArrayApprox "pytorch_roundtrip_parity: cnn" (← leanCNN) pyCnn
  assertArrayApprox "pytorch_roundtrip_parity: transformer" (← leanTransformer) pyTransformer
  IO.println "pytorch_roundtrip_parity: ok"

end PyTorchRoundtrip
end TorchLean.Test
