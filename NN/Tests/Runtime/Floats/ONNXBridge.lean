/-
Copyright (c) 2026 TorchLean contributors
Released under MIT license as described in the file LICENSE.
Authors: TorchLean contributors
-/
module

public import NN.Core.ExternalProcess
public import NN.IR.Semantics
public import NN.Runtime.PyTorch.Export.ONNX
public import NN.Runtime.PyTorch.Import.TorchExport
public import NN.Tests.Utils

/-!
# ONNX artifact round trip

Export one small Conv → BatchNorm → Softmax model to ONNX, run the generated bridge, and load its
graph and payload through Lean's checked importer. Compare the resulting tensor with ONNX's
reference evaluator. Unequal channel and spatial extents make a misplaced normalization axis
visible. This covers the generated adapter, not a general import-equivalence theorem.
-/

public section

namespace Tests.Floats.ONNXBridge

open TorchLean

private def workDir : System.FilePath := TorchLean.External.Process.directory "onnx_bridge_check"

private def modelScript : String := "\n".intercalate
  [ "import json, sys"
  , "from pathlib import Path"
  , "import numpy as np"
  , "import onnx"
  , "from onnx import TensorProto, helper, numpy_helper"
  , "from onnx.reference import ReferenceEvaluator"
  , "work = Path(sys.argv[1])"
  , "x = ((np.arange(18, dtype=np.float32) * 0.37) % 2.3 - 1.1).reshape(1, 2, 3, 3)"
  , "def constant(name, value):"
  , "    return numpy_helper.from_array(np.asarray(value, dtype=np.float32), name=name)"
  , "initializers = ["
  , "    constant('weight', np.linspace(-0.4, 0.6, 24).reshape(3, 2, 2, 2)),"
  , "    constant('conv_bias', [-0.2, 0.1, 0.3]),"
  , "    constant('scale', [0.5, 1.2, 1.7]), constant('bias', [-0.1, 0.3, -0.2]),"
  , "    constant('mean', [0.2, -0.4, 0.1]), constant('var', [0.5, 0.25, 1.2])]"
  , "nodes = ["
  , "    helper.make_node('Conv', ['x', 'weight', 'conv_bias'], ['conv']),"
  , "    helper.make_node('BatchNormalization', ['conv', 'scale', 'bias', 'mean', 'var'],"
  , "                     ['normalized'], epsilon=0.01),"
  , "    helper.make_node('Softmax', ['normalized'], ['y'], axis=1)]"
  , "graph = helper.make_graph(nodes, 'torchlean_roundtrip',"
  , "    [helper.make_tensor_value_info('x', TensorProto.FLOAT, [1, 2, 3, 3])],"
  , "    [helper.make_tensor_value_info('y', TensorProto.FLOAT, [1, 3, 2, 2])],"
  , "    initializer=initializers)"
  , "model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid('', 17)])"
  , "onnx.checker.check_model(model)"
  , "onnx.save(model, work / 'model.onnx')"
  , "loaded = onnx.load(work / 'model.onnx')"
  , "assert [n.op_type for n in loaded.graph.node] == ['Conv', 'BatchNormalization', 'Softmax']"
  , "(y,) = ReferenceEvaluator(loaded).run(None, {'x': x})"
  , "(work / 'reference.json').write_text(json.dumps({'input': x.tolist(), 'output': y.tolist()}))"
  ]

private def readJson (path : System.FilePath) : IO Lean.Json := do
  IO.ofExcept (Lean.Json.parse (← IO.FS.readFile path))

/-- Exercise the actual generated ONNX bridge and compare all output entries. -/
def run : IO Unit := do
  let available ← TorchLean.External.Process.pythonCanImport #["onnx", "numpy"]
  unless ← Tests.Utils.checkInteropDependency "onnx, numpy" available do
    IO.println "ONNX round trip: skipped (onnx or numpy is unavailable)"
    return
  IO.FS.createDirAll workDir
  let fixture := workDir / "make_model.py"
  let bridge := workDir / "onnx_to_torchlean_ir.py"
  let artifact := workDir / "model.graph.json"
  IO.FS.writeFile fixture modelScript
  IO.FS.writeFile bridge (Export.PyTorch.ONNX.script {})
  discard <| TorchLean.External.Process.run "ONNX fixture" "python3"
    #[fixture.toString, workDir.toString]
  discard <| TorchLean.External.Process.run "ONNX bridge" "python3"
    #[bridge.toString, (workDir / "model.onnx").toString, artifact.toString]
  let json ← readJson artifact
  let captured ← IO.ofExcept (Import.PyTorch.TorchExport.parseGraph json)
  let payload ← IO.ofExcept (Import.PyTorch.TorchExport.parsePayload json)
  let reference ← readJson (workDir / "reference.json")
  let inputNode ← IO.ofExcept (captured.graph.getNode captured.inputId)
  let inputJson ← IO.ofExcept (reference.getObjVal? "input")
  let some input := Import.PyTorch.parseTensor inputNode.outShape inputJson
    | throw <| IO.userError "ONNX round trip: input shape mismatch"
  let some outputId := captured.outputIds[0]?
    | throw <| IO.userError "ONNX round trip: missing output"
  unless captured.outputIds.size == 1 do
    throw <| IO.userError "ONNX round trip: unexpected output count"
  let outputNode ← IO.ofExcept (captured.graph.getNode outputId)
  unless outputNode.outShape == [1, 3, 2, 2] do
    throw <| IO.userError "ONNX round trip: exported output shape changed"
  let outputJson ← IO.ofExcept (reference.getObjVal? "output")
  let some expected := Import.PyTorch.parseTensor outputNode.outShape outputJson
    | throw <| IO.userError "ONNX round trip: reference shape mismatch"
  let actual ← IO.ofExcept <| NN.IR.Graph.denote (α := Float) captured.graph payload
    (Spec.SomeTensor.ofTensor input) outputId
  if h : actual.shape = outputNode.outShape then
    let values : Tensor Float outputNode.outShape := h ▸ actual.tensor
    unless Tensor.foldl (· && ·) true
        (TorchLean.Tensor.Internal.Rep.zipWith
          (fun x y => x.isFinite && y.isFinite && Float.abs (x - y) ≤ 1e-5)
          values expected) do
      throw <| IO.userError "ONNX round trip: output differs from the ONNX reference"
  else
    throw <| IO.userError "ONNX round trip: evaluated output has the wrong shape"
  IO.println "ONNX Conv/BatchNorm/Softmax round trip: ok"

end Tests.Floats.ONNXBridge
