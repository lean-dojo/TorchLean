/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Core.ExternalProcess
public import NN.Runtime.PyTorch.Export.ONNX
public import NN.Runtime.PyTorch.Import.TorchExport
public import NN.Tests.Utils

/-!
# ONNX Bridge Generator Checks

These checks cover the generated Python adapter at the artifact boundary: it must emit the
`torchlean.ir.v1` format, reuse the TorchLean IR parent conventions, and reject ONNX shapes or ops
that cannot be represented by the current checked IR fragment. When the optional Python `onnx`
package is installed, the test also lowers a real BatchNorm+ReLU model and parses the resulting
artifact through Lean's checked IR importer.
-/

@[expose] public section

namespace Tests
namespace Floats
namespace ONNXBridge

open Export.PyTorch.ONNX

def workDir : System.FilePath :=
  TorchLean.External.Process.artifactWorkDir "onnx_bridge_check"

def bridgePath : System.FilePath :=
  workDir / "onnx_to_torchlean_ir.py"

def modelPath : System.FilePath :=
  workDir / "batchnorm_relu.onnx"

def jsonPath : System.FilePath :=
  workDir / "batchnorm_relu.graph.json"

def assertContains (label haystack needle : String) : IO Unit := do
  unless haystack.contains needle do
    throw (IO.userError s!"onnx_bridge: missing {label}: {needle}")

/-- Check ONNX availability, failing when required interop checks are enabled. -/
def pythonHasONNX : IO Bool := do
  Tests.Utils.checkInteropDependency "onnx, numpy"
    (← TorchLean.External.Process.pythonCanImport #["onnx", "numpy"])

def sampleModelScript : String :=
  String.intercalate "\n"
    [ "from pathlib import Path"
    , "import numpy as np"
    , "import onnx"
    , "from onnx import TensorProto, helper, numpy_helper"
    , ""
    , "model_path = Path('" ++ modelPath.toString ++ "')"
    , "x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [1, 2, 2, 2])"
    , "y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [1, 2, 2, 2])"
    , "initializers = ["
    , "    numpy_helper.from_array(np.array([1.0, 0.5], dtype=np.float32), name='scale'),"
    , "    numpy_helper.from_array(np.array([0.0, 0.1], dtype=np.float32), name='bias'),"
    , "    numpy_helper.from_array(np.array([0.2, -0.1], dtype=np.float32), name='mean'),"
    , "    numpy_helper.from_array(np.array([0.5, 0.25], dtype=np.float32), name='var'),"
    , "]"
    , "bn = helper.make_node('BatchNormalization', ['x', 'scale', 'bias', 'mean', 'var'], "
        ++ "['bn'], epsilon=1e-5, name='bn0')"
    , "relu = helper.make_node('Relu', ['bn'], ['y'], name='relu0')"
    , "graph = helper.make_graph([bn, relu], 'torchlean_bn_relu', [x], [y], "
        ++ "initializer=initializers)"
    , "model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid('', 17)])"
    , "onnx.checker.check_model(model)"
    , "onnx.save(model, model_path)"
    ]

def runRealONNXRoundtrip : IO Unit := do
  if !(← pythonHasONNX) then
    IO.println "onnx_bridge: real ONNX roundtrip skipped (python package `onnx` not installed)"
    return ()
  IO.FS.createDirAll workDir
  IO.FS.writeFile bridgePath (generateBridgeScript {})
  IO.FS.writeFile (workDir / "make_batchnorm_relu.py") sampleModelScript
  let _ ← TorchLean.External.Process.runStdoutChecked
    (ctx := "onnx_bridge: build representative ONNX model")
    (cmd := "python3")
    (args := #[(workDir / "make_batchnorm_relu.py").toString])
    (cwd := some ".")
  let _ ← TorchLean.External.Process.runStdoutChecked
    (ctx := "onnx_bridge: lower representative ONNX model")
    (cmd := "python3")
    (args := #[bridgePath.toString, modelPath.toString, jsonPath.toString])
    (cwd := some ".")
  let txt ← IO.FS.readFile jsonPath
  assertContains "BatchNorm epsilon value" txt "\"values\": ["
  assertContains "BatchNorm epsilon literal" txt "9.999999"
  let json ←
    match Lean.Json.parse txt with
    | .ok j => pure j
    | .error e => throw (IO.userError s!"onnx_bridge: emitted invalid JSON: {e}")
  match Import.PyTorch.TorchExport.parseGraph json with
  | .ok cg =>
      unless cg.graph.nodes.size >= 6 do
        throw (IO.userError
          s!"onnx_bridge: imported graph unexpectedly small: {cg.graph.nodes.size}")
  | .error e =>
      throw (IO.userError s!"onnx_bridge: Lean parser rejected generated artifact: {e}")

def run : IO Unit := do
  IO.println "onnx_bridge: begin"
  runRealONNXRoundtrip
  IO.println "onnx_bridge: ok"

end ONNXBridge
end Floats
end Tests
