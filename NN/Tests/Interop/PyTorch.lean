/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Core.ExternalProcess
public import NN.Runtime.PyTorch.Export.IRPyTorch
public import NN.Runtime.PyTorch.Export.TorchExport
public import NN.Runtime.PyTorch.Import.TorchExport
public import NN.API.CLI.Command
public import NN.API.Json
public import NN.Runtime.PyTorch.Export.MLP
public import NN.Runtime.PyTorch.Export.Transformer
public import NN.Runtime.PyTorch.Import.Transformer

/-!
# PyTorch numerical boundary checks

Check exact scalar emission, nonsymmetric matrix orientation, normalization epsilon, and
representative imported-model execution against PyTorch. Generated fixtures use no downloaded data.

Run `lake exe pytorch_export_check`.
-/

@[expose] public section

namespace NN.Tests.Interop.PyTorch

open Lean
open TorchLean

/-- Command-line help for the PyTorch export bridge check. -/
def usage : String :=
  String.intercalate "\n"
    [ "TorchLean PyTorch export bridge check"
    , ""
    , "Usage:"
    , "  lake exe pytorch_export_check"
    , ""
    , "Checks exact float emission, parameter orientation, and imported numerical execution."
    ]

/-- Scratch directory for the generated Python and the captured graphs. -/
def workDir : System.FilePath :=
  TorchLean.External.Process.artifactWorkDir "pytorch_export_check"

/-- Where the export bridge script is written. -/
def bridgePath : System.FilePath :=
  workDir / "export_torchlean_graph.py"

/-- A second generated bridge that exercises FX without attempting `torch.export`. -/
def fxBridgePath : System.FilePath :=
  workDir / "export_torchlean_fx_graph.py"

/-- Where the generated PyTorch model definitions are written. -/
def modelPath : System.FilePath :=
  workDir / "tiny_models.py"

/--
Python source for the models that TorchLean's importer is expected to handle.

The command writes these fixtures into its build directory. Python and PyTorch are required;
no external model files or datasets are needed.
-/
def supportedModelSource : String :=
  r#"import torch
import torch.nn as nn
import torch.nn.functional as F
torch.manual_seed(0)

class TinyMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 3)
        self.fc2 = nn.Linear(3, 2)
    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))

class TinyAffineLayerNorm(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(4, eps=0.25)
        with torch.no_grad():
            self.norm.weight.copy_(torch.tensor([0.5, 1.0, 1.5, 2.0]))
            self.norm.bias.copy_(torch.tensor([-1.0, -0.5, 0.5, 1.0]))
    def forward(self, x):
        return self.norm(x)

class TinyLayerNormEpsilon(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(4, eps=1e-4, elementwise_affine=False)
    def forward(self, x):
        return self.norm(x)

class TinyBatchNormEpsilon(nn.Module):
    def __init__(self):
        super().__init__()
        self.bn = nn.BatchNorm2d(2, eps=1e-4)
        with torch.no_grad():
            self.bn.weight.copy_(torch.tensor([1.5, 0.5]))
            self.bn.bias.copy_(torch.tensor([-0.25, 0.75]))
            self.bn.running_mean.copy_(torch.tensor([0.5, -1.0]))
            self.bn.running_var.copy_(torch.tensor([2.0, 3.0]))
    def forward(self, x):
        return self.bn(x)

class TinyFunctionalOperators(nn.Module):
    def forward(self, x):
        y = torch.exp(x)
        return (torch.log(y), torch.sin(x), torch.cos(x), torch.sqrt(y),
                torch.reciprocal(y), torch.abs(x), torch.tanh(x), torch.sigmoid(x),
                torch.add(x, y), torch.sub(x, y), torch.mul(x, y),
                torch.maximum(x, y), torch.minimum(x, y), torch.relu(x),
                torch.softmax(x, dim=1), F.softmax(x, 1, 3, None),
                torch.sum(x, dim=1), torch.mean(x, dim=0, keepdim=True),
                torch.reshape(x, (3, 2)), torch.flatten(x),
                torch.permute(x, (1, 0)), torch.transpose(x, dim0=0, dim1=1),
                torch.cat((x, y), dim=1), torch.matmul(x, torch.transpose(x, 0, 1)))

class TinyMethodOperators(nn.Module):
    def forward(self, x):
        y = x.exp()
        return (y.log(), x.sin(), x.cos(), y.sqrt(), y.reciprocal(), x.abs(),
                x.tanh(), x.sigmoid(), x.add(y), x.sub(y), x.mul(y),
                x.maximum(y), x.minimum(y), x.relu(), x.softmax(1),
                x.sum(1), x.mean(0, keepdim=True), x.reshape(3, 2), x.view(3, 2),
                x.flatten(), x.permute(1, 0), x.transpose(0, 1), x.contiguous(),
                x.matmul(x.transpose(0, 1)))
"#

/-- Capture one model to the checked IR artifact format. -/
def runCapture (ctor : String) (outPath : System.FilePath) (shape : String)
    (scriptPath : System.FilePath := bridgePath) (requireTorchExport : Bool := false) : IO String :=
  TorchLean.External.Process.runStdoutChecked
    (ctx := s!"PyTorch graph capture ({ctor})") (cmd := "python3")
    (args := #[scriptPath.toString, modelPath.toString, ctor, outPath.toString,
      "--example-shape", shape] ++ if requireTorchExport then #["--require-torch-export"] else #[])
    (cwd := some ".")

/-- Compare every output of an imported graph against PyTorch on a deterministic input. -/
def checkNumericalParity
    (ctor shape : String) (artifactPath : System.FilePath) : IO Unit := do
  let json ← TorchLean.Json.readFile artifactPath
  let captured ← IO.ofExcept (Import.PyTorch.TorchExport.parseGraph json)
  let payload ← IO.ofExcept (Import.PyTorch.TorchExport.parsePayload json)
  let python ← TorchLean.External.Process.runStdoutChecked
    (ctx := s!"PyTorch numerical parity ({ctor})")
    (cmd := "python3")
    (args := #[
      "-c",
      "import importlib.util,json,math,torch; " ++
      s!"spec=importlib.util.spec_from_file_location('tiny_models','{modelPath}'); " ++
      "m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); " ++
      s!"model=m.{ctor}().eval(); shape=tuple(int(x) for x in '{shape}'.split(',')); " ++
      "x=torch.arange(math.prod(shape),dtype=torch.float32).reshape(shape)/7.0; " ++
      "result=model(x); outputs=list(result) if isinstance(result,tuple) else [result]; " ++
      "print(json.dumps({'input':x.tolist()," ++
        "'outputs':[value.detach().tolist() for value in outputs]}))"
    ])
    (cwd := some ".")
  let reference ← IO.ofExcept (Lean.Json.parse python.trimAscii.toString)
  let inputJson ← IO.ofExcept (reference.getObjVal? "input")
  let outputJsons ← IO.ofExcept ((← IO.ofExcept (reference.getObjVal? "outputs")).getArr?)
  let inputNode ← IO.ofExcept (captured.graph.getNode captured.inputId)
  let some inputTensor := Import.PyTorch.parseTensor inputNode.outShape inputJson
    | throw <| IO.userError "numerical parity input shape mismatch"
  unless outputJsons.size = captured.outputIds.size do
    throw <| IO.userError <|
      s!"numerical parity output count mismatch for {ctor}: " ++
        s!"PyTorch returned {outputJsons.size}, graph declares {captured.outputIds.size}"
  for (outputId, outputJson) in captured.outputIds.zip outputJsons do
    let outputNode ← IO.ofExcept (captured.graph.getNode outputId)
    let some expectedTensor := Import.PyTorch.parseTensor outputNode.outShape outputJson
      | throw <| IO.userError
          s!"numerical parity output shape mismatch for {ctor}, node {outputId}"
    let actual ← IO.ofExcept (NN.IR.Graph.denote (α := Float) captured.graph payload
      (Spec.SomeTensor.ofTensor inputTensor) outputId)
    let actualTensor : Tensor Float outputNode.outShape ←
      if hShape : actual.shape = outputNode.outShape then
        pure (hShape ▸ actual.tensor)
      else
        throw <| IO.userError
          s!"numerical parity produced the wrong output shape for {ctor}, node {outputId}"
    let errors := Tensor.map2Spec (fun value expected => Float.abs (value - expected))
      actualTensor expectedTensor
    unless Tensor.allSpec (fun error => error ≤ 2e-5) errors do
      throw <| IO.userError
        s!"numerical parity mismatch for {ctor}, node {outputId}: absolute error exceeds 2e-5"
  IO.println s!"  numerical parity: ok ({ctor})"

/-- Capture one model and compare imported execution against PyTorch. -/
def runSupportedCase (ctor shape : String)
    (scriptPath : System.FilePath := bridgePath)
    (requireTorchExport : Bool := false) : IO Unit := do
  let outPath := workDir / s!"{ctor}.graph.json"
  discard <| runCapture ctor outPath shape scriptPath requireTorchExport
  checkNumericalParity ctor shape outPath

/--
Exercise the generated bridge entrypoint on neighboring names, overloads, and FX callables.
Rejection must happen before an artifact is written; supported cases also check their wire kinds.
-/
def classificationRegressionSource : String :=
  r#"import importlib.util
import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

root = Path(sys.argv[1])
def load(name):
    spec = importlib.util.spec_from_file_location(name, root / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

class Unary(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x):
        return self.fn(x)

def unused_inplace(x):
    x.relu_()
    return x

negative = {
    "log1p": torch.log1p, "log2": torch.log2, "log10": torch.log10,
    "expm1": torch.expm1, "exp2": torch.exp2, "sinh": torch.sinh, "cosh": torch.cosh,
    "log_softmax": lambda x: F.log_softmax(x, dim=1),
    "aten_log_softmax": lambda x: torch.ops.aten._log_softmax.default(x, 1, False),
    "softmax_dtype": lambda x: torch.softmax(x, 1, dtype=torch.float64),
    "functional_softmax_dtype": lambda x: F.softmax(x, 1, 3, torch.float64),
    "method_softmax_dtype": lambda x: x.softmax(1, dtype=torch.float64),
    "view_dtype": lambda x: x.view(torch.int32),
    "aten_view_dtype": lambda x: torch.ops.aten.view.dtype(x, torch.int32),
    "relu_inplace": lambda x: F.relu(x, inplace=True),
    "method_inplace": lambda x: x.relu_(),
    "unused_inplace_result": unused_inplace,
    "out_overload": lambda x: torch.ops.aten.exp.out(x, out=x),
    "scalar_add": lambda x: torch.add(x, 1.0),
    "scaled_add": lambda x: torch.add(x, x, alpha=2),
    "sort_tuple": lambda x: torch.sort(x),
    "tensor_getitem": lambda x: x[0],
    "pool_indices": lambda x: F.max_pool1d(x, 2, return_indices=True),
}
positive = {
    "exp": (torch.exp, "exp"),
    "log": (torch.log, "log"),
    "sin": (torch.sin, "sin"),
    "cos": (torch.cos, "cos"),
    "torch_softmax": (lambda x: torch.softmax(x, 1), "softmax"),
    "functional_softmax": (lambda x: F.softmax(x, 1), "softmax"),
    "method_softmax": (lambda x: x.softmax(1), "softmax"),
    "aten_softmax": (lambda x: torch.ops.aten._softmax.default(x, 1, False), "softmax"),
    "max_pool": (lambda x: F.max_pool1d(x, 2), "max_pool"),
    "avg_pool": (lambda x: F.avg_pool1d(x, 2), "avg_pool"),
}
for name, strict in (("export_torchlean_graph", True), ("export_torchlean_fx_graph", False)):
    bridge = load(name)
    rejected_models = [(label, Unary(fn).eval()) for label, fn in negative.items()]
    rejected_models.append(("training_batch_norm", nn.Sequential(nn.BatchNorm1d(4)).train()))
    for label, model in rejected_models:
        path = root / (name + "_" + label + ".json")
        path.unlink(missing_ok=True)
        try:
            bridge.export_torchlean_graph_json(
                model, torch.ones(2, 4), str(path), strict)
        except NotImplementedError:
            pass
        else:
            raise AssertionError("accepted unsupported operator: " + name + "/" + label)
        assert not path.exists(), "rejected operation wrote an artifact"
    for label, (fn, kind) in positive.items():
        path = root / (name + "_" + label + ".json")
        graph = bridge.export_torchlean_graph_json(
            Unary(fn).eval(), torch.ones(2, 4), str(path), strict)
        output = graph["nodes"][graph["output_ids"][0]]
        assert path.exists() and output["kind"] == kind, (label, output)
print("generated bridge classification: both capture paths passed")
"#

/-- Run schema rejection and callable preservation through both generated Python adapters. -/
def runClassificationChecks : IO Unit := do
  let output ← TorchLean.External.Process.runStdoutChecked
    (ctx := "generated PyTorch bridge classification") (cmd := "python3")
    (args := #["-c", classificationRegressionSource, workDir.toString])
  IO.println output.trimAscii.toString

/-- Execute exported scalar/tensor expressions and a tiny-epsilon normalization in Python. -/
def runFloatCodegenChecks : IO Unit := do
  let mut cases : Array UInt64 := #[0, 0x8000000000000000, 1, 0x8000000000000001,
    0x000fffffffffffff, 0x0010000000000000, 0x7fefffffffffffff, 0xffefffffffffffff,
    0x7ff0000000000000, 0xfff0000000000000, 0x7ff8000000000000]
  cases := Tensor.foldl (fun bits value => bits.push value.toBits) cases
    ([1.0e-9, -1.0e-9, 0.1, 0.5, 1.23456789, 1.0e300, 1.0e-300] : Tensor Float [7])
  let mut seed : UInt64 := 1739
  for _ in [0:256] do
    seed := seed * 6364136223846793005 + 1442695040888963407
    cases := cases.push seed
  let checks := cases.map fun bits =>
    s!"check({Export.PyTorch.floatToPyString (Float.ofBits bits)}, {bits})"
  let matrix : Tensor Float [2, 4] :=
    [[1.0e-9, Float.ofBits 0x8000000000000000, Float.ofBits 1, 0.1],
      [Float.ofBits 0x7fefffffffffffff, -1.0e-300, 1.23456789, -0.5]]
  let expected := String.intercalate ", "
    ((matrix.to (Array Float)).toList.map (fun value => toString value.toBits))
  let graph : NN.IR.Graph := { nodes := #[
    { id := 0, parents := #[], kind := .input, outShape := [1] },
    { id := 1, parents := #[0], kind := .batchNormEval 0 1, outShape := [1] }] }
  let parameters : NN.IR.BatchNormEvalParams Float :=
    { c := 1, gamma := [1], beta := [0], mean := [0], var := [0], eps := 1.0e-9 }
  let store : NN.MLTheory.CROWN.Graph.ParamStore Float :=
    { batchNormEval :=
        ({} : Std.HashMap Nat (NN.IR.BatchNormEvalParams Float)).insert 1 parameters }
  let model ← IO.ofExcept (Export.IRPyTorch.emit graph store 0 1
    { className := "TinyEpsilon", dtypeExpr := "torch.float64", includeTrainingSkeleton := false })
  let source := String.intercalate "\n"
    ([ "import struct, math, torch"
     , "def check(value, expected):"
     , "  if expected & 0x7ff0000000000000 == 0x7ff0000000000000 and expected & 0xfffffffffffff:"
     , "    assert math.isnan(value)"
     , "  else:"
     , "    assert struct.unpack('>Q', struct.pack('>d', value))[0] == expected"
     , s!"matrix = {Export.PyTorch.tensorToPyString matrix}"
     , s!"transposed = {Export.PyTorch.transposedMatrixTensorToPy matrix}"
     , s!"expected = [{expected}]"
     , "for value, bits in zip(sum(matrix, []), expected): check(value, bits)"
     , "for i in range(2):"
     , "  for j in range(4): check(transposed[j][i], expected[i * 4 + j])"
     , "tensor_values = torch.tensor(matrix, dtype=torch.float64).flatten().tolist()"
     , "for value, bits in zip(tensor_values, expected):"
     , "  check(value, bits)"
     ] ++ checks.toList ++
     [ model
     , "actual = TinyEpsilon()(torch.ones(1, dtype=torch.float64))"
     , "expected = torch.rsqrt(torch.tensor([1e-9], dtype=torch.float64))"
     , "assert torch.equal(actual, expected), (actual, expected)"
     ])
  let path := workDir / "float_source_roundtrip.py"
  IO.FS.writeFile path source
  discard <| TorchLean.External.Process.runStdoutChecked
    (ctx := "exported Python float expressions") (cmd := "python3") (args := #[path.toString])
  IO.println "generated Python float bit patterns and tiny epsilon: ok"

/-- Execute generated reference models and check both state-dict naming and matrix orientation. -/
def runReferenceCodegenChecks : IO Unit := do
  let matrix : Tensor Float [2, 2] := [[1.0, 2.0], [3.0, 4.0]]
  let bias : Tensor Float [2] := [0.25, 0.5]
  IO.FS.writeFile (workDir / "reference_mlp.py")
    (Export.PyTorch.MLP.withParameters matrix bias matrix bias "ReferenceMLP" .sequential)
  IO.FS.writeFile (workDir / "reference_transformer.py")
    (Export.PyTorch.Transformer.withParameters 1 2 1 2
      matrix matrix matrix matrix matrix matrix bias bias bias bias bias bias
      "ReferenceTransformer")
  let source := String.intercalate "\n"
    [ "import json, pathlib, sys, torch"
    , "root = pathlib.Path(sys.argv[1])"
    , "def load(name):"
    , "  namespace = {'__name__': name}"
    , "  exec((root / (name + '.py')).read_text(), namespace)"
    , "  return namespace"
    , "m = load('reference_mlp')"
    , "model = m['load_mlp_weights'](m['ReferenceMLP']())"
    , "assert torch.equal(model.fc1.weight, torch.tensor([[1., 2.], [3., 4.]]))"
    , "m = load('reference_transformer')"
    , "model = m['load_transformer_weights'](m['ReferenceTransformer']())"
    , "print(json.dumps({'params': {k: v.tolist() for k, v in model.state_dict().items()}}))"
    ]
  let output ← TorchLean.External.Process.runStdoutChecked
    (ctx := "generated PyTorch reference models") (cmd := "python3")
    (args := #["-c", source, workDir.toString])
  let json ← IO.ofExcept (Json.parse output.trimAscii.toString)
  let some parameters := Import.PyTorch.Transformer.load 2 2 json
    | throw <| IO.userError "generated Transformer state dict was rejected"
  for weight in #[parameters.queryWeight, parameters.keyWeight,
      parameters.valueWeight, parameters.outputWeight,
      parameters.feedForwardInputWeight, parameters.feedForwardOutputWeight] do
    unless Tensor.allSpec id (Tensor.map2Spec (· == ·) weight matrix) do
      throw <| IO.userError "generated Transformer state dict changed matrix orientation"
  IO.println "generated reference code and state-dict round trip: ok"

/-- Main runtime-check body. -/
def run : IO Unit := do
  IO.FS.createDirAll workDir
  IO.FS.writeFile bridgePath (Export.PyTorch.TorchExport.generateGraphBridgeScript {})
  IO.FS.writeFile fxBridgePath
    (Export.PyTorch.TorchExport.generateGraphBridgeScript { preferTorchExport := false })
  IO.FS.writeFile modelPath supportedModelSource
  IO.println "== PyTorch nn.Module → TorchLean IR runtime check =="
  runFloatCodegenChecks
  runReferenceCodegenChecks
  runClassificationChecks
  for ctor in #["TinyFunctionalOperators", "TinyMethodOperators"] do
    runSupportedCase ctor "2,3" bridgePath true
    runSupportedCase ctor "2,3" fxBridgePath
  runSupportedCase "TinyMLP" "4"
  runSupportedCase "TinyAffineLayerNorm" "2,4"
  runSupportedCase "TinyLayerNormEpsilon" "2,4"
  runSupportedCase "TinyBatchNormEpsilon" "1,2,4,4"
  IO.println "pytorch_export_check: ok"

/-- Entrypoint used by `lake exe pytorch_export_check`. -/
def main (args : List String) : IO UInt32 := do
  let args := TorchLean.CLI.dropDashDash args
  if TorchLean.CLI.hasHelp args then
    IO.println usage
    return 0
  TorchLean.CLI.requireNoArgs "pytorch_export_check" args
  run
  pure 0

end NN.Tests.Interop.PyTorch
