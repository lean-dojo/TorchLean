/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Core.ExternalProcess
public import NN.IR.Semantics
public import NN.Runtime.PyTorch.Export.ONNX
public import NN.Runtime.PyTorch.Import.TorchExport
public import NN.Tests.Utils

/-!
# ONNX Bridge Generator Checks

These checks cover the generated Python adapter at the artifact boundary: it must emit the
`torchlean.ir.v1` format, reuse the TorchLean IR parent conventions, and reject ONNX shapes or ops
that cannot be represented by the current checked IR fragment. When the optional Python `onnx`
package is installed, the test also lowers real ONNX models, parses the resulting artifacts
through Lean's checked IR importer, and compares the imported graph's output against the ONNX
reference evaluator. The parity models cover the ONNX defaults the adapter has to reproduce:
per-channel BatchNorm on rank-4 inputs, reductions with attribute or input axes and the default
`keepdims = 1`, the 2-D `Flatten`, `Softmax` before and after opset 13, `Conv` with bias, strides,
dilations, asymmetric padding and groups, and rejection of NaN initializers, which JSON cannot
carry.
-/

@[expose] public section

namespace Tests
namespace Floats
namespace ONNXBridge

open TorchLean
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

/-- ONNX models whose imported graphs must reproduce the ONNX reference evaluator. -/
def parityCases : List String :=
  [ "bn_c3_w4", "bn_c2_w2", "reduce_sum_axes_attr", "reduce_mean_all", "reduce_mean_all_keepdims"
  , "reduce_sum_axes_input", "flatten_axis2", "softmax_opset11", "softmax_opset11_last"
  , "softmax_opset13", "conv2d_bias", "conv2d_strided_padded", "conv1d_grouped_nobias" ]

/-- Build the parity models, their reference outputs, and a model with a NaN initializer. -/
def parityModelScript : String :=
  String.intercalate "\n"
    [ "import json"
    , "from pathlib import Path"
    , "import numpy as np"
    , "import onnx"
    , "from onnx import TensorProto, helper, numpy_helper"
    , "from onnx.reference import ReferenceEvaluator"
    , ""
    , "work = Path('" ++ workDir.toString ++ "')"
    , ""
    , "def save(name, nodes, in_shape, out_shape, inits, opset):"
    , "    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, in_shape)"
    , "    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, out_shape)"
    , "    graph = helper.make_graph(nodes, name, [x], [y], initializer=inits)"
    , "    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid('', opset)])"
    , "    onnx.checker.check_model(model)"
    , "    onnx.save(model, work / f'{name}.onnx')"
    , "    return model"
    , ""
    , "def case(name, nodes, in_shape, out_shape, inits=(), opset=17, reference=None):"
    , "    model = save(name, nodes, in_shape, out_shape, list(inits), opset)"
    , "    size = int(np.prod(in_shape))"
    , "    x = ((np.arange(size, dtype=np.float32) * 0.37) % 2.3 - 1.1).reshape(in_shape)"
    , "    if reference is None:"
    , "        (y,) = ReferenceEvaluator(model).run(None, {'x': x})"
    , "    else:"
    , "        y = reference(x)"
    , "    ref = {'input': x.tolist(), 'output': np.asarray(y, dtype=np.float64).tolist()}"
    , "    (work / f'{name}.ref.json').write_text(json.dumps(ref))"
    , ""
    , "# onnx.reference runs Softmax-11 with the opset-13 meaning,"
    , "# so compute the 2-D coercion here."
    , "def softmax_coerced(axis):"
    , "    def run(x):"
    , "        flat = x.reshape(int(np.prod(x.shape[:axis])), -1).astype(np.float64)"
    , "        e = np.exp(flat - flat.max(axis=1, keepdims=True))"
    , "        return (e / e.sum(axis=1, keepdims=True)).reshape(x.shape)"
    , "    return run"
    , ""
    , "def f32(name, values):"
    , "    return numpy_helper.from_array(np.array(values, dtype=np.float32), name=name)"
    , ""
    , "def bn(name, channels, in_shape):"
    , "    inits = [f32('scale', np.linspace(0.5, 1.5, channels)),"
    , "             f32('bias', np.linspace(-0.2, 0.3, channels)),"
    , "             f32('mean', np.linspace(0.1, -0.4, channels)),"
    , "             f32('var', np.linspace(0.25, 2.0, channels))]"
    , "    node = helper.make_node('BatchNormalization', ['x', 'scale', 'bias', 'mean', 'var'],"
    , "                            ['y'], epsilon=1e-3)"
    , "    case(name, [node], in_shape, in_shape, inits)"
    , ""
    , "bn('bn_c3_w4', 3, [2, 3, 2, 4])"
    , "bn('bn_c2_w2', 2, [1, 2, 3, 2])"
    , "case('reduce_sum_axes_attr', [helper.make_node('ReduceSum', ['x'], ['y'], axes=[1])],"
    , "     [2, 3, 4], [2, 1, 4], opset=11)"
    , "case('reduce_mean_all', [helper.make_node('ReduceMean', ['x'], ['y'], keepdims=0)],"
    , "     [2, 3, 4], [], opset=11)"
    , "case('reduce_mean_all_keepdims', [helper.make_node('ReduceMean', ['x'], ['y'])],"
    , "     [2, 3, 4], [1, 1, 1], opset=11)"
    , "axes = numpy_helper.from_array(np.array([-1, 0], dtype=np.int64), name='axes')"
    , "case('reduce_sum_axes_input',"
    , "     [helper.make_node('ReduceSum', ['x', 'axes'], ['y'], keepdims=0)],"
    , "     [2, 3, 4], [3], [axes], opset=13)"
    , "case('flatten_axis2', [helper.make_node('Flatten', ['x'], ['y'], axis=2)], [2, 3, 4],"
    , "     [6, 4])"
    , "case('softmax_opset11', [helper.make_node('Softmax', ['x'], ['y'])], [2, 3, 4], [2, 3, 4],"
    , "     opset=11, reference=softmax_coerced(1))"
    , "case('softmax_opset11_last', [helper.make_node('Softmax', ['x'], ['y'], axis=-1)],"
    , "     [2, 3, 4], [2, 3, 4], opset=11)"
    , "case('softmax_opset13', [helper.make_node('Softmax', ['x'], ['y'], axis=1)], [2, 3, 4],"
    , "     [2, 3, 4])"
    , "def kernel(name, shape):"
    , "    size = int(np.prod(shape))"
    , "    return f32(name, (np.linspace(-1.0, 1.0, size) * 0.7).reshape(shape))"
    , ""
    , "case('conv2d_bias',"
    , "     [helper.make_node('Conv', ['x', 'w', 'b'], ['y'])],"
    , "     [1, 1, 4, 4], [1, 2, 3, 3], [kernel('w', [2, 1, 2, 2]), f32('b', [0.25, -0.5])])"
    , "case('conv2d_strided_padded',"
    , "     [helper.make_node('Conv', ['x', 'w', 'b'], ['y'], strides=[2, 1],"
    , "                       dilations=[1, 2], pads=[1, 0, 0, 1])],"
    , "     [1, 2, 5, 5], [1, 3, 2, 2],"
    , "     [kernel('w', [3, 2, 3, 3]), f32('b', [0.1, 0.0, -0.3])])"
    , "case('conv1d_grouped_nobias',"
    , "     [helper.make_node('Conv', ['x', 'w'], ['y'], group=2, pads=[1, 1])],"
    , "     [1, 4, 6], [1, 4, 6], [kernel('w', [4, 2, 3])])"
    , "save('nan_initializer', [helper.make_node('Add', ['x', 'c'], ['y'])], [2], [2],"
    , "     [f32('c', [1.0, float('nan')])], 17)"
    ]

/-- Read and parse one JSON file written by the bridge or the model builder. -/
def readJson (path : System.FilePath) : IO Lean.Json := do
  match Lean.Json.parse (← IO.FS.readFile path) with
  | .ok j => pure j
  | .error e => throw (IO.userError s!"onnx_bridge: {path} is not valid JSON: {e}")

/-- Run the generated bridge on `<name>.onnx`, returning the process result. -/
def lowerModel (name : String) : IO IO.Process.Output :=
  IO.Process.output
    { cmd := "python3"
      args := #[bridgePath.toString, (workDir / s!"{name}.onnx").toString,
        (workDir / s!"{name}.graph.json").toString]
      cwd := some "." }

/-- Lower one parity model and compare Lean's execution of the imported graph with ONNX. -/
def checkParityCase (name : String) : IO Unit := do
  let out ← lowerModel name
  unless out.exitCode == 0 do
    throw (IO.userError s!"onnx_bridge: {name}: lowering failed: {out.stderr}")
  let json ← readJson (workDir / s!"{name}.graph.json")
  let captured ← match Import.PyTorch.TorchExport.parseGraph json with
    | .ok cg => pure cg
    | .error e => throw (IO.userError s!"onnx_bridge: {name}: Lean parser rejected graph: {e}")
  let payload ← match Import.PyTorch.TorchExport.parsePayload json with
    | .ok p => pure p
    | .error e => throw (IO.userError s!"onnx_bridge: {name}: payload import failed: {e}")
  let reference ← readJson (workDir / s!"{name}.ref.json")
  let inputJson ← IO.ofExcept (reference.getObjVal? "input")
  let outputJson ← IO.ofExcept (reference.getObjVal? "output")
  let inputNode ← IO.ofExcept (captured.graph.getNode captured.inputId)
  let some inputTensor := Import.PyTorch.parseTensor inputNode.outShape inputJson
    | throw (IO.userError s!"onnx_bridge: {name}: reference input shape mismatch")
  let some outputId := captured.outputIds[0]?
    | throw (IO.userError s!"onnx_bridge: {name}: imported graph has no output")
  let outputNode ← IO.ofExcept (captured.graph.getNode outputId)
  let some expected := Import.PyTorch.parseTensor outputNode.outShape outputJson
    | throw (IO.userError
        s!"onnx_bridge: {name}: ONNX output does not have shape {repr outputNode.outShape}")
  let actual ← match NN.IR.Graph.denote (α := Float) captured.graph payload
      (Spec.SomeTensor.ofTensor inputTensor) outputId with
    | .ok value => pure value
    | .error e => throw (IO.userError s!"onnx_bridge: {name}: evaluation failed: {e}")
  let actualTensor : Tensor Float outputNode.outShape ←
    if hShape : actual.shape = outputNode.outShape then
      pure (hShape ▸ actual.tensor)
    else
      throw (IO.userError s!"onnx_bridge: {name}: evaluation produced the wrong shape")
  let errors := Tensor.map2Spec (fun value target => Float.abs (value - target))
    actualTensor expected
  unless Tensor.allSpec (fun error => error ≤ 1e-5) errors do
    throw (IO.userError s!"onnx_bridge: {name}: output differs from the ONNX reference")

/-- Lower every parity model, then check that NaN initializers are refused before JSON output. -/
def runParityChecks : IO Unit := do
  if !(← pythonHasONNX) then
    IO.println "onnx_bridge: ONNX parity checks skipped (python package `onnx` not installed)"
    return ()
  IO.FS.createDirAll workDir
  IO.FS.writeFile bridgePath (generateBridgeScript {})
  IO.FS.writeFile (workDir / "make_parity_models.py") parityModelScript
  let _ ← TorchLean.External.Process.runStdoutChecked
    (ctx := "onnx_bridge: build parity models")
    (cmd := "python3")
    (args := #[(workDir / "make_parity_models.py").toString])
    (cwd := some ".")
  for name in parityCases do
    checkParityCase name
  let nanOut ← lowerModel "nan_initializer"
  if nanOut.exitCode == 0 then
    throw (IO.userError "onnx_bridge: a NaN initializer was written to the JSON artifact")
  assertContains "NaN rejection message" nanOut.stderr "cannot carry NaN"
  IO.println s!"onnx_bridge: {parityCases.length} ONNX parity cases ok"

def run : IO Unit := do
  IO.println "onnx_bridge: begin"
  runRealONNXRoundtrip
  runParityChecks
  IO.println "onnx_bridge: ok"

end ONNXBridge
end Floats
end Tests
