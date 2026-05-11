/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import Lean.Data.Json
public import NN.Runtime.External.Process
public import NN.Runtime.PyTorch.Export.TorchExport
public import NN.Runtime.PyTorch.Import.TorchExport

/-!
# PyTorch `nn.Module` → TorchLean IR Smoke Test

This executable example exercises the model-agnostic PyTorch graph bridge:

1. write the generated Python graph-capture adapter;
2. write several compact supported PyTorch models (MLP/CNN/CNN head/attention/MHA blocks);
3. capture them to `torchlean.ir.v1` JSON;
4. parse and validate each JSON artifact back into `NN.IR.Graph`; and
5. exercise a real `torch.save(model.state_dict())` checkpoint reload path; and
6. run deliberate unsupported-op cases to confirm the bridge reports unsupported semantics clearly.

The example is intentionally small. Its job is not to benchmark PyTorch; it is to test the trust
boundary we care about: PyTorch may produce an artifact, but TorchLean accepts it only after parsing
and validating the explicit IR JSON.

Run:

```bash
lake exe torchlean pytorch_export_smoke
```
-/

@[expose] public section

namespace NN.Examples.Interop.PyTorch.TorchExportSmoke

open Lean

def workDir : System.FilePath :=
  ".lake/build/torchlean_pytorch_export_smoke"

def bridgePath : System.FilePath :=
  workDir / "export_torchlean_graph.py"

def modelPath : System.FilePath :=
  workDir / "tiny_models.py"

def checkpointPath : System.FilePath :=
  workDir / "tiny_mlp_state.pt"

def supportedModelSource : String :=
  String.intercalate "\n"
    [ "import torch"
    , "import torch.nn as nn"
    , "import torch.nn.functional as F"
    , "from pathlib import Path"
    , ""
    , "class TinyAddRelu(nn.Module):"
    , "    def forward(self, x):"
    , "        return torch.relu(x + x)"
    , ""
    , "class TinyMLP(nn.Module):"
    , "    def __init__(self):"
    , "        super().__init__()"
    , "        self.fc1 = nn.Linear(4, 3)"
    , "        self.fc2 = nn.Linear(3, 2)"
    , "    def forward(self, x):"
    , "        return self.fc2(torch.relu(self.fc1(x)))"
    , ""
    , "class TinyCheckpointMLP(TinyMLP):"
    , "    def __init__(self):"
    , "        super().__init__()"
    , "        path = Path(__file__).with_name('tiny_mlp_state.pt')"
    , "        state = torch.load(path, map_location='cpu', weights_only=True)"
    , "        self.load_state_dict(state)"
    , ""
    , "class TinyCNN(nn.Module):"
    , "    def __init__(self):"
    , "        super().__init__()"
    , "        self.conv = nn.Conv2d(1, 2, kernel_size=3, padding=1)"
    , "    def forward(self, x):"
    , "        return F.max_pool2d(torch.relu(self.conv(x)), kernel_size=2, stride=2)"
    , ""
    , "class TinyCNNHead(nn.Module):"
    , "    def __init__(self):"
    , "        super().__init__()"
    , "        self.conv = nn.Conv2d(1, 2, kernel_size=3, padding=1)"
    , "        self.fc = nn.Linear(32, 5)"
    , "    def forward(self, x):"
    , "        x = F.max_pool2d(torch.relu(self.conv(x)), kernel_size=2, stride=2)"
    , "        x = torch.flatten(x)"
    , "        return self.fc(x)"
    , ""
    , "class TinyNormSoftmax(nn.Module):"
    , "    def __init__(self):"
    , "        super().__init__()"
    , "        self.norm = nn.LayerNorm(4)"
    , "    def forward(self, x):"
    , "        return torch.softmax(self.norm(x), dim=-1)"
    , ""
    , "class TinyTransformerishBlock(nn.Module):"
    , "    def __init__(self):"
    , "        super().__init__()"
    , "        self.norm = nn.LayerNorm(4)"
    , "        self.fc1 = nn.Linear(4, 4)"
    , "        self.fc2 = nn.Linear(4, 4)"
    , "    def forward(self, x):"
    , "        y = self.fc2(torch.relu(self.fc1(self.norm(x))))"
    , "        return torch.softmax(x + y, dim=-1)"
    , ""
    , "class TinySelfAttentionOps(nn.Module):"
    , "    def forward(self, x):"
    , "        scores = torch.matmul(x, x.transpose(-2, -1))"
    , "        attn = torch.softmax(scores, dim=-1)"
    , "        return torch.matmul(attn, x)"
    , ""
    , "class UnsupportedSort(nn.Module):"
    , "    def forward(self, x):"
    , "        return torch.sort(x, dim=-1).values"
    , ""
    , "class TinySingleHeadMHA(nn.Module):"
    , "    def __init__(self):"
    , "        super().__init__()"
    , "        self.mha = nn.MultiheadAttention(embed_dim=4, num_heads=1, batch_first=True)"
    , "    def forward(self, x):"
    , "        y, _ = self.mha(x, x, x)"
    , "        return y"
    , ""
    , "class UnsupportedMultiHeadMHA(nn.Module):"
    , "    def __init__(self):"
    , "        super().__init__()"
    , "        self.mha = nn.MultiheadAttention(embed_dim=4, num_heads=2, batch_first=True)"
    , "    def forward(self, x):"
    , "        y, _ = self.mha(x, x, x)"
    , "        return y"
    , ""
    ]

def readJsonFile (path : System.FilePath) : IO Json := do
  let txt ← IO.FS.readFile path
  match Json.parse txt with
  | .ok j => pure j
  | .error e => throw <| IO.userError s!"bad JSON in {path}: {e}"

def runCapture (ctor : String) (outPath : System.FilePath) (shape : String) : IO String := do
  Runtime.External.Process.runStdoutChecked
    (ctx := s!"PyTorch graph capture ({ctor})")
    (cmd := "python")
    (args := #[bridgePath.toString, modelPath.toString, ctor, outPath.toString, "--example-shape", shape])
    (cwd := some ".")

/--
Write a real PyTorch checkpoint used by `TinyCheckpointMLP`.

This is intentionally a compact `state_dict` round-trip: Python owns `.pt` loading/saving, and the graph
bridge only sees the resulting initialized `nn.Module`. That mirrors the intended user workflow:
load trained weights in Python, capture the model graph, then ask Lean to validate the exported IR.
-/
def writeCheckpoint : IO Unit := do
  let _stdout ← Runtime.External.Process.runStdoutChecked
    (ctx := "PyTorch checkpoint fixture")
    (cmd := "python")
    (args := #[
      "-c",
      "import importlib.util, torch; " ++
      s!"spec=importlib.util.spec_from_file_location('tiny_models', '{modelPath}'); " ++
      "m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); " ++
      "torch.manual_seed(7); model=m.TinyMLP(); " ++
      s!"torch.save(model.state_dict(), '{checkpointPath}')"
    ])
    (cwd := some ".")
  pure ()

/--
Keep subprocess failures readable in the smoke output.

The full `IO.userError` still contains command, args, exit code, and stderr. The smoke only prints
the first informative lines so unsupported-op tests explain the boundary without hiding the signal.
-/
def compactFailure (err : IO.Error) : String :=
  String.intercalate "\n" ((err.toString.splitOn "\n").take 24)

/-- Run one supported capture path and parse the resulting graph in Lean. -/
def runSupportedCase (ctor shape : String) : IO Unit := do
  let outPath := workDir / s!"{ctor}.graph.json"
  let _stdout ← runCapture ctor outPath shape
  IO.println s!"captured supported graph: {ctor} ({shape})"
  let j ← readJsonFile outPath
  match Import.PyTorch.TorchExport.parseGraph j with
  | .ok cg =>
      IO.println s!"  accepted: nodes={cg.graph.nodes.size}, input={cg.inputId}, output={cg.outputId}"
      IO.println "  guarantee: WellShaped via parseGraph_wellShaped"
  | .error e =>
      throw <| IO.userError s!"TorchLean rejected supported PyTorch graph `{ctor}`:\n{e}"

/-- Run the supported capture paths and parse the resulting graphs in Lean. -/
def runSupported : IO Unit := do
  runSupportedCase "TinyAddRelu" "1,4"
  runSupportedCase "TinyMLP" "4"
  runSupportedCase "TinyCheckpointMLP" "4"
  runSupportedCase "TinyCNN" "1,8,8"
  runSupportedCase "TinyCNNHead" "1,8,8"
  runSupportedCase "TinyNormSoftmax" "4"
  runSupportedCase "TinyTransformerishBlock" "4"
  runSupportedCase "TinySelfAttentionOps" "1,2,4"
  runSupportedCase "TinySingleHeadMHA" "1,2,4"

/-- Run a deliberate unsupported op and require the Python bridge to reject it. -/
def runPythonUnsupportedCase (ctor shape msg : String) : IO Unit := do
  try
    let stdout ← runCapture ctor (workDir / s!"{ctor}.graph.json") shape
    throw <| IO.userError
      (s!"unsupported PyTorch model unexpectedly exported successfully:\n{stdout}")
  catch _err =>
    IO.println s!"unsupported-op rejection: ok ({ctor})"
    IO.println s!"  {msg}"
    IO.println s!"  failure detail:\n{compactFailure _err}"

/--
Run a PyTorch graph that can be captured as a value graph but must not lower into the tensor IR yet.

This is the intended behavior for container-valued PyTorch ops: the JSON keeps the tuple,
then Lean rejects tensor lowering with a semantic explanation instead of the Python bridge crashing
or incorrectly treating the tuple producer as a tensor op.
-/
def runLeanUnsupportedCase (ctor shape msg : String) : IO Unit := do
  let outPath := workDir / s!"{ctor}.graph.json"
  let _stdout ← runCapture ctor outPath shape
  let j ← readJsonFile outPath
  match Import.PyTorch.TorchExport.parseGraph j with
  | .ok _ =>
      throw <| IO.userError s!"unsupported PyTorch value graph unexpectedly lowered: {ctor}"
  | .error e =>
      IO.println s!"unsupported tensor-lowering rejection: ok ({ctor})"
      IO.println s!"  {msg}"
      IO.println s!"  failure detail:\n{e}"

/-- Run deliberate unsupported ops and require the Python bridge to reject them. -/
def runUnsupported : IO Unit := do
  runPythonUnsupportedCase "UnsupportedSort" "1,4"
    "rejected `torch.sort`, which is not in the current TorchLean IR import subset"
  runLeanUnsupportedCase "UnsupportedMultiHeadMHA" "1,2,4"
    "captured MultiheadAttention as a tuple-valued FX node, then rejected tensor lowering because multi-head splitting is not in the current tensor-IR lowering slice"

/-- Main smoke-test body. -/
def run : IO Unit := do
  IO.FS.createDirAll workDir
  IO.FS.writeFile bridgePath (Export.PyTorch.TorchExport.generateGraphBridgeScript {})
  IO.FS.writeFile modelPath supportedModelSource
  writeCheckpoint
  IO.println "== PyTorch nn.Module → TorchLean IR smoke =="
  runSupported
  runUnsupported
  IO.println "pytorch_export_smoke: ok"

/-- Entrypoint used by `lake exe torchlean pytorch_export_smoke`. -/
def main (_args : List String) : IO UInt32 := do
  run
  pure 0

end NN.Examples.Interop.PyTorch.TorchExportSmoke
