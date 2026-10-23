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
visible. Rejection checks cover domains, integer data, and square-root preconditions, while
integer shape/axis metadata remains accepted. This covers the generated adapter, not a general
import-equivalence theorem.
-/

public section

namespace Tests.Floats.ONNXBridge

open TorchLean

private def workDir : System.FilePath := TorchLean.External.Process.directory "onnx_bridge_check"

private def modelScript : String := "\n".intercalate
  [ "import importlib.util, json, sys"
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
  , "spec = importlib.util.spec_from_file_location('onnx_bridge', work / 'onnx_to_torchlean_ir.py')"
  , "bridge = importlib.util.module_from_spec(spec)"
  , "spec.loader.exec_module(bridge)"
  , "def check_case(label, nodes, *, dtype=TensorProto.FLOAT, shape=(1,),"
  , "               initializers=(), domain=None, opset=17, output_dtype=None, error=None):"
  , "    graph = helper.make_graph(nodes, label,"
  , "        [helper.make_tensor_value_info('x', dtype, shape)],"
  , "        [helper.make_tensor_value_info('y', output_dtype or dtype, shape)],"
  , "        initializer=initializers)"
  , "    opsets = [helper.make_operatorsetid('', opset)]"
  , "    if domain: opsets.append(helper.make_operatorsetid(domain, 1))"
  , "    model = helper.make_model(graph, opset_imports=opsets)"
  , "    onnx.checker.check_model(model, full_check=True)"
  , "    path = work / (label + '.onnx')"
  , "    onnx.save(model, path)"
  , "    try:"
  , "        artifact = bridge.export_onnx_torchlean_graph_json(path, work / (label + '.json'))"
  , "    except RuntimeError as exc:"
  , "        assert error and error in str(exc), (label, str(exc))"
  , "    else:"
  , "        assert error is None, label + ': unsupported semantics were accepted'"
  , "        return artifact"
  , "check_case('raw_sqrt', [helper.make_node('Sqrt', ['x'], ['y'])],"
  , "           error='structurally nonnegative')"
  , "check_case('integer_add', [helper.make_node('Add', ['x', 'x'], ['y'])],"
  , "           dtype=TensorProto.UINT8, error='only floating-point data')"
  , "check_case('integer_direct_output', [], output_dtype=TensorProto.INT64,"
  , "    initializers=[numpy_helper.from_array(np.array([9007199254740993], dtype=np.int64),"
  , "                                         name='y')], error='only floating-point data')"
  , "check_case('custom_add',"
  , "    [helper.make_node('Add', ['x', 'x'], ['y'], domain='custom.example')],"
  , "           domain='custom.example', error='registered semantics')"
  , "check_case('safe_sqrt', [helper.make_node('Abs', ['x'], ['a']),"
  , "                         helper.make_node('Sqrt', ['a'], ['y'])])"
  , "for op, name, value in [('Reshape', 'shape', [1]), ('ReduceSum', 'axes', [0])]:"
  , "    metadata = numpy_helper.from_array(np.asarray(value, dtype=np.int64), name=name)"
  , "    check_case('integer_metadata_' + op, [helper.make_node(op, ['x', name], ['y'])],"
  , "               initializers=[metadata])"
  , "check_case('negative_variance',"
  , "    [helper.make_node('BatchNormalization', ['x', 'scale', 'bias', 'mean', 'var'], ['y'])],"
  , "    shape=(1, 1), initializers=[constant('scale', [1]), constant('bias', [0]),"
  , "        constant('mean', [0]), constant('var', [-1])], error='structurally nonnegative')"
  , "for is_test in (0, 1):"
  , "    check_case('legacy_batchnorm_' + str(is_test),"
  , "        [helper.make_node('BatchNormalization', ['x', 'scale', 'bias', 'mean', 'var'],"
  , "                          ['y'], is_test=is_test)], shape=(1, 1), opset=6,"
  , "        initializers=[constant('scale', [1]), constant('bias', [0]),"
  , "                      constant('mean', [0]), constant('var', [1])],"
  , "        error='legacy training mode' if not is_test else None)"
  , "collision = check_case('helper_name_collision', ["
  , "    helper.make_node('Add', ['x', 'x'], ['g__torchlean_gemm_matmul']),"
  , "    helper.make_node('Gemm', ['x', 'weight', 'bias'], ['g']),"
  , "    helper.make_node('Add', ['g__torchlean_gemm_matmul', 'g'], ['y'])],"
  , "    shape=(1, 1), initializers=[constant('weight', [[3]]), constant('bias', [[1]])])"
  , "original = next(n for n in collision['nodes']"
  , "                if n['kind'] == 'add' and n['parents'] == [0, 0])"
  , "result = collision['nodes'][collision['output_ids'][0]]"
  , "assert result['parents'][0] == original['id'], 'generated helper shadowed a source value'"
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
