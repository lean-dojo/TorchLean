/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/
module

public import TorchLean.NN.Subprocess
public import TorchLean.NN.IR.Artifact
public import TorchLeanTest.IR.Capture
public import TorchLeanTest.Harness

/-!
# ONNX artifact round trip

Build small ONNX models and run the ONNX bridge `python3 -m torchlean.onnx`, which reads Lean's op
table (`TorchLean.IR.table`, written to the work directory when the test runs). Then check every
artifact it writes: decoding its JSON and writing the artifact again gives the same JSON, and
`Artifact.load` accepts it. For the models with a reference, the inputs and outputs have the element
types of their ONNX values, and the reference runtime computes the outputs from the inputs' words.
The constant step that the bridge's sources name for each floating initializer holds its element
type, shape and words.

* The binary32 models agree with the reference within `1e-5` per entry. They include Conv, BatchNorm
  and Softmax written by the bridge's command line, every elementwise operator with broadcast and
  three-operand forms, matrix products with vectors and broadcast batches, both reductions, and
  Flatten, Gemm and Reshape. The rest are grouped, strided, padded and dilated convolutions of a
  batch of two with batch normalization, the same convolutions followed by Softmax with opset 11's
  coercion, and a convolution whose weight the graph computes from a second input.
* Sums and products at binary64, binary16 and bfloat16, BatchNorm's default epsilon at binary64,
  special words at binary32 and bfloat16, and a binary64 initializer returned beside a binary32
  result give the reference's words exactly. So do opset 6 broadcasting, Concat's default axis at
  opset 3, and a sum and a mean over two axes. IEEE maximum and minimum give them up to NaN
  payloads.

The bridge refuses an operator of a custom domain, integer data, an empty mean, shapes that opset 6
does not broadcast and BatchNorm with `is_test=0` at opset 6. It accepts integer shape and axis
metadata, a square root of any operand and BatchNorm with a negative variance. The command line's
`--sources` file has one entry per step, and a source value named like a generated helper keeps its
own step. This covers the ONNX bridge, not a general import-equivalence theorem.
-/

public section

namespace TorchLean.Test.ONNX

open Lean TorchLean.IR

private def workDir : System.FilePath := Subprocess.directory "onnx_bridge_check"

private def modelScript : String := r#"import json, math, sys
sys.dont_write_bytecode = True
from pathlib import Path
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from onnx.reference import ReferenceEvaluator
import torchlean.onnx as bridge
work = Path(sys.argv[1])
manifest = []
def element_type(array):
    return helper.np_dtype_to_tensor_dtype(array.dtype)
def words(array):
    array = np.ascontiguousarray(array).reshape(-1)
    return [int(w) for w in array.view({2: np.uint16, 4: np.uint32, 8: np.uint64}[array.itemsize])]
def constant(name, value, dtype=np.float32):
    return numpy_helper.from_array(np.asarray(value, dtype=dtype), name=name)
def node(op, inputs, output, **attributes):
    return helper.make_node(op, inputs, [output], **attributes)
def model(label, nodes, inputs, outputs, initializers, opset=17, compare=1e-5, reference=None,
          write=True):
    # Save the model, write its artifact and the source of each step (unless the command line
    # writes them) and its reference: each input's and output's words and ONNX element type, the
    # output shapes, and the name, words, shape and element type of each floating initializer
    # (an integer one is metadata).
    # `inputs` maps names to arrays, `outputs` names to (shape, element type). The outputs are
    # onnx.reference's unless `reference` computes them from the inputs (where onnx.reference
    # departs from the ONNX schema; see the callers).
    graph = helper.make_graph(nodes, label,
        [helper.make_tensor_value_info(n, element_type(x), x.shape) for n, x in inputs.items()],
        [helper.make_tensor_value_info(n, t, s) for n, (s, t) in outputs.items()],
        initializer=initializers)
    onnx_model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid('', opset)])
    onnx.checker.check_model(onnx_model, full_check=True)
    onnx.save(onnx_model, work / (label + '.onnx'))
    ys = reference(**inputs) if reference else ReferenceEvaluator(onnx_model).run(None, inputs)
    (work / (label + '.reference.json')).write_text(json.dumps({
        'inputs': [words(x) for x in inputs.values()],
        'input_types': [element_type(x) for x in inputs.values()],
        'outputs': [words(y) for y in ys], 'shapes': [list(np.shape(y)) for y in ys],
        'output_types': [t for (_, t) in outputs.values()],
        'constants': [{'name': i.name, 'words': words(numpy_helper.to_array(i)),
                       'shape': list(i.dims), 'type': i.data_type} for i in initializers
                      if i.data_type in bridge.ELEMENT_TYPES]}))
    if write:
        sources = []
        bridge.export_onnx_torchlean_graph_json(work / (label + '.onnx'), work / (label + '.json'),
                                                sources)
        (work / (label + '.sources.json')).write_text(json.dumps(sources))
    manifest.append({'label': label, 'compare': compare})
F32 = TensorProto.FLOAT
x = ((np.arange(18, dtype=np.float32) * 0.37) % 2.3 - 1.1).reshape(1, 2, 3, 3)
model('model', [node('Conv', ['x', 'weight', 'conv_bias'], 'conv'),
    node('BatchNormalization', ['conv', 'scale', 'bias', 'mean', 'var'], 'normalized',
         epsilon=0.01),
    node('Softmax', ['normalized'], 'y', axis=1)], {'x': x}, {'y': ([1, 3, 2, 2], F32)},
    [constant('weight', np.linspace(-0.4, 0.6, 24).reshape(3, 2, 2, 2)),
     constant('conv_bias', [-0.2, 0.1, 0.3]),
     constant('scale', [0.5, 1.2, 1.7]), constant('bias', [-0.1, 0.3, -0.2]),
     constant('mean', [0.2, -0.4, 0.1]), constant('var', [0.5, 0.25, 1.2])], write=False)
# Every elementwise operator, a broadcast operand, Max of three operands, Concat and Transpose.
unary = ['Relu', 'Tanh', 'Sigmoid', 'Exp', 'Sin', 'Cos', 'Abs']
x = np.linspace(-1.1, 1.3, 6, dtype=np.float32).reshape(2, 3)
model('elementwise', [node(op, ['x'], op.lower()) for op in unary] + [
    node('Log', ['exp'], 'log'), node('Sqrt', ['abs'], 'sqrt'),
    node('Reciprocal', ['exp'], 'reciprocal'), node('Add', ['x', 'offset'], 'add'),
    node('Sub', ['relu', 'tanh'], 'sub'), node('Mul', ['sigmoid', 'sin'], 'mul'),
    node('Max', ['cos', 'abs', 'offset'], 'max'), node('Min', ['sqrt', 'reciprocal'], 'min'),
    node('Concat', [op.lower() for op in unary] + ['log', 'sqrt', 'reciprocal', 'add', 'sub',
         'mul', 'max', 'min'], 'joined', axis=0),
    node('Transpose', ['joined'], 'y')], {'x': x}, {'y': ([3, 30], F32)},
    [constant('offset', [0.25, -0.5, 1.0])])
# Matrix products with broadcast batches and vectors, the reductions, Flatten, Gemm, Reshape.
x = np.linspace(-1.0, 1.2, 24, dtype=np.float32).reshape(2, 3, 4)
model('tensor', [node('MatMul', ['x', 'w'], 'product'),
    node('MatMul', ['v', 'product'], 'left'), node('MatMul', ['product', 'u'], 'right'),
    node('Softmax', ['product'], 'softmax', axis=1),
    node('ReduceSum', ['softmax', 'axes02'], 'sum', keepdims=0),
    node('ReduceMean', ['product'], 'mean', axes=[2], keepdims=1),
    node('Sub', ['product', 'mean'], 'centered'),
    node('Flatten', ['centered'], 'rows', axis=2),
    node('Gemm', ['rows', 'b', 'c'], 'gemm', transB=1),
    node('Reshape', ['gemm', 'flat'], 'g'), node('Reshape', ['sum', 'flat'], 's'),
    node('Reshape', ['left', 'flat'], 'l'), node('Reshape', ['right', 'flat'], 'r'),
    node('Concat', ['g', 's', 'l', 'r'], 'y', axis=0)], {'x': x}, {'y': ([43], F32)},
    [constant('w', np.linspace(-0.5, 0.5, 20).reshape(4, 5)),
     constant('v', [0.3, -0.2, 0.9]), constant('u', [1.0, -1.0, 0.5, 0.25, 2.0]),
     constant('axes02', [0, 2], np.int64),
     constant('b', np.linspace(1.0, -1.0, 20).reshape(4, 5)), constant('c', [0.5]),
     constant('flat', [-1], np.int64)])
# A batch of two grouped, strided, dilated and padded convolutions, then batch normalization of
# the rank-4 result (at opset 17: onnx.reference takes batch statistics at opsets 9 to 13); the
# convolutions without bias before Softmax's coercion of opset 11, which onnx.reference skips
# (it normalizes along the axis only), so its reference follows the schema.
x = np.sin(np.arange(200, dtype=np.float32)).reshape(2, 4, 5, 5)
kernel = np.cos(np.arange(108)).reshape(6, 2, 3, 3).astype(np.float32)
conv = dict(group=2, strides=[2, 1], pads=[1, 1, 0, 1], dilations=[1, 2])
model('batched', [node('Conv', ['x', 'kernel', 'conv_bias'], 'conv', **conv),
    node('BatchNormalization', ['conv', 'scale', 'bias', 'mean', 'var'], 'y', epsilon=1e-3)],
    {'x': x}, {'y': ([2, 6, 2, 3], F32)},
    [constant('kernel', kernel), constant('conv_bias', np.linspace(-0.2, 0.3, 6)),
     constant('scale', np.linspace(0.5, 1.5, 6)), constant('bias', np.linspace(-0.3, 0.3, 6)),
     constant('mean', np.linspace(-0.1, 0.1, 6)), constant('var', np.linspace(0.2, 1.2, 6))])
def coerced_softmax(z):
    rows = z.reshape(z.shape[0], -1)
    e = np.exp(rows - rows.max(axis=1, keepdims=True))
    return (e / e.sum(axis=1, keepdims=True)).reshape(z.shape)
convolved = ReferenceEvaluator(helper.make_model(helper.make_graph(
    [node('Conv', ['x', 'kernel'], 'y', **conv)], 'conv',
    [helper.make_tensor_value_info('x', F32, x.shape)],
    [helper.make_tensor_value_info('y', F32, [2, 6, 2, 3])],
    initializer=[constant('kernel', kernel)]))).run(None, {'x': x})[0]
model('coerced', [node('Conv', ['x', 'kernel'], 'conv', **conv),
    node('Softmax', ['conv'], 'y', axis=1)], {'x': x}, {'y': ([2, 6, 2, 3], F32)},
    [constant('kernel', kernel)], opset=11,
    reference=lambda x: [coerced_softmax(convolved)])
# A convolution whose weight is computed in the graph from a second graph input.
w = np.cos(np.arange(36, dtype=np.float32)).reshape(2, 2, 3, 3)
model('computed', [node('Mul', ['w', 'half'], 'weight'),
    node('Conv', ['x', 'weight'], 'y', pads=[1, 1, 1, 1])],
    {'x': np.sin(np.arange(50, dtype=np.float32)).reshape(1, 2, 5, 5), 'w': w},
    {'y': ([1, 2, 5, 5], F32)}, [constant('half', [0.5])])
# Exact words at binary64, binary16 and bfloat16: sums and products round once in the source
# format.
for label, dtype in [('double', np.float64), ('half', np.float16),
                     ('bfloat', helper.tensor_dtype_to_np_dtype(TensorProto.BFLOAT16))]:
    x = np.array([0.1, -2.5, 3.0e-5, 7.0], dtype=dtype)
    model(label, [node('Add', ['x', 'c'], 'sum'), node('Mul', ['sum', 'x'], 'y')], {'x': x},
          {'y': ([4], element_type(x))}, [constant('c', [0.2, 1.0 / 3.0, -1.0e-5, 65.0], dtype)],
          compare='words')
# BatchNormalization's default epsilon is the binary32 attribute 1e-5f, at binary64 too: the
# normalizations with and without the explicit default agree word for word.
x = np.array([[1.0, 2.0], [3.0, -4.0]], dtype=np.float64).reshape(2, 2, 1)
bn = ['x', 'scale', 'bias', 'mean', 'var']
model('epsilon', [node('BatchNormalization', bn, 'default'),
    node('BatchNormalization', bn, 'explicit', epsilon=1e-5),
    node('Sub', ['default', 'explicit'], 'y')], {'x': x}, {'y': ([2, 2, 1], TensorProto.DOUBLE)},
    [constant('scale', [1.0, 1.0], np.float64), constant('bias', [0.0, 0.0], np.float64),
     constant('mean', [0.0, 0.0], np.float64), constant('var', [0.0, 0.0], np.float64)],
    compare='words')
# Signed zeros, a subnormal, infinities and NaN payloads survive in binary32 and bfloat16.
for label, dtype, bits in [('exceptional', np.float32, [0x80000000, 0x00000001, 0x7f800000,
                                                        0xff800000, 0x7fc00123, 0xffa00001]),
                           ('exceptional_bf16', helper.tensor_dtype_to_np_dtype(
                               TensorProto.BFLOAT16), [0x8000, 0x0001, 0x7f80, 0xff80, 0x7fc3,
                                                       0xffa1])]:
    word = {np.dtype(np.float32): np.uint32}.get(np.dtype(dtype), np.uint16)
    special = np.array(bits, dtype=word).view(dtype)
    x = np.array([1.5, -0.0], dtype=dtype)
    model(label, [node('Concat', ['x', 'special'], 'y', axis=0)], {'x': x},
          {'y': ([8], element_type(x))}, [numpy_helper.from_array(special, name='special')],
          compare='words', reference=lambda x, special=special: [np.concatenate([x, special])])
# Mixed element types: a binary32 result beside a binary64 initializer returned directly.
model('mixed', [node('Relu', ['x'], 'y')], {'x': np.array([-1.0, 2.0], dtype=np.float32)},
      {'y': ([2], F32), 'c': ([1], TensorProto.DOUBLE)},
      [constant('c', [1.0 + 2.0 ** -30], np.float64)], compare='words')
# Opset 6 broadcasting: B placed from `axis`, or along the trailing axes without it.
x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
model('legacy', [node('Add', ['x', 'b'], 'sum', broadcast=1, axis=0),
    node('Mul', ['sum', 'c'], 'y', broadcast=1)], {'x': x}, {'y': ([2, 2], F32)},
    [constant('b', [10.0, 20.0]), constant('c', [1.0, -1.0])], opset=6, compare='words',
    reference=lambda x: [(x + np.array([[10.0], [20.0]], dtype=np.float32)) *
                         np.array([[1.0, -1.0]], dtype=np.float32)])
# IEEE maximum and minimum: NaN wins, +0 is above -0, whatever the operand order.
def ieee(a, b, larger):
    if math.isnan(a) or math.isnan(b):
        return math.nan
    if a == b == 0 and larger:
        return 0.0 if math.copysign(1, a) > 0 or math.copysign(1, b) > 0 else -0.0
    if a == b == 0:
        return -0.0 if math.copysign(1, a) < 0 or math.copysign(1, b) < 0 else 0.0
    return max(a, b) if larger else min(a, b)
x = np.array([math.nan, -0.0, 0.0, 1.0, -1.0, 0.0], dtype=np.float32)
c = np.array([1.0, 0.0, -0.0, math.nan, -2.0, 0.0], dtype=np.float32)
model('extrema', [node('Max', ['x', 'c'], 'a'), node('Max', ['c', 'x'], 'b'),
    node('Min', ['x', 'c'], 'i'), node('Min', ['c', 'x'], 'j'),
    node('Concat', ['a', 'b', 'i', 'j'], 'y', axis=0)], {'x': x}, {'y': ([24], F32)},
    [numpy_helper.from_array(c, name='c')], compare='nan',
    reference=lambda x: [np.array([ieee(float(p), float(q), larger)
                                   for larger, (s, t) in [(True, (x, c)), (True, (c, x)),
                                                          (False, (x, c)), (False, (c, x))]
                                   for p, q in zip(s, t)], dtype=np.float32)])
# One fold over both axes and one division by the entry count, as onnx.reference computes the
# sum 1 and the mean 0.25: a chain over one axis at a time gives 0 for both.
model('folded', [node('ReduceSum', ['x', 'axes'], 'y', keepdims=0),
    node('ReduceMean', ['x'], 'm', axes=[0, 1], keepdims=0)],
      {'x': np.array([[2.0 ** 25, 1.0], [-2.0 ** 25, 1.0]], dtype=np.float32)},
      {'y': ([], F32), 'm': ([], F32)}, [constant('axes', [0, 1], np.int64)], compare='words')
# Concat's default axis is 1 before opset 4.
model('concat_v1', [node('Concat', ['x', 'x'], 'y')], {'x': np.array([[5.0]], dtype=np.float32)},
      {'y': ([1, 2], F32)}, [], opset=3, compare='words',
      reference=lambda x: [np.concatenate([x, x], axis=1)])
def check_case(label, nodes, *, dtype=TensorProto.FLOAT, shape=(1,), out_shape=None,
               initializers=(), domain=None, opset=17, output_dtype=None, error=None,
               sources=None):
    graph = helper.make_graph(nodes, label,
        [helper.make_tensor_value_info('x', dtype, shape)],
        [helper.make_tensor_value_info('y', output_dtype or dtype, out_shape or shape)],
        initializer=initializers)
    opsets = [helper.make_operatorsetid('', opset)]
    if domain: opsets.append(helper.make_operatorsetid(domain, 1))
    onnx_model = helper.make_model(graph, opset_imports=opsets)
    onnx.checker.check_model(onnx_model, full_check=True)
    path = work / (label + '.onnx')
    onnx.save(onnx_model, path)
    sources = [] if sources is None else sources
    try:
        artifact = bridge.export_onnx_torchlean_graph_json(path, work / (label + '.json'),
                                                          sources)
    except RuntimeError as exc:
        assert error and error in str(exc), (label, str(exc))
    else:
        assert error is None, label + ': unsupported semantics were accepted'
        (work / (label + '.sources.json')).write_text(json.dumps(sources))
        manifest.append({'label': label, 'compare': None})
        return artifact
check_case('raw_sqrt', [helper.make_node('Sqrt', ['x'], ['y'])])
check_case('integer_add', [helper.make_node('Add', ['x', 'x'], ['y'])],
           dtype=TensorProto.UINT8, error='only floating-point data')
check_case('integer_direct_output', [], output_dtype=TensorProto.INT64,
    initializers=[numpy_helper.from_array(np.array([9007199254740993], dtype=np.int64),
                                         name='y')], error='only floating-point data')
check_case('custom_add',
    [helper.make_node('Add', ['x', 'x'], ['y'], domain='custom.example')],
           domain='custom.example', error='registered semantics')
check_case('safe_sqrt', [helper.make_node('Abs', ['x'], ['a']),
                         helper.make_node('Sqrt', ['a'], ['y'])])
for op, name, value in [('Reshape', 'shape', [1]), ('ReduceSum', 'axes', [0])]:
    metadata = numpy_helper.from_array(np.asarray(value, dtype=np.int64), name=name)
    check_case('integer_metadata_' + op, [helper.make_node(op, ['x', name], ['y'], keepdims=1)
                                          if op == 'ReduceSum' else
                                          helper.make_node(op, ['x', name], ['y'])],
               initializers=[metadata])
check_case('empty_mean', [helper.make_node('ReduceMean', ['x'], ['y'], axes=[1], keepdims=0)],
           shape=(2, 0), out_shape=(2,), error='empty fiber')
check_case('negative_variance',
    [helper.make_node('BatchNormalization', ['x', 'scale', 'bias', 'mean', 'var'], ['y'])],
    shape=(1, 1), initializers=[constant('scale', [1]), constant('bias', [0]),
        constant('mean', [0]), constant('var', [-1])])
for is_test in (0, 1):
    check_case('batchnorm_is_test_' + str(is_test),
        [helper.make_node('BatchNormalization', ['x', 'scale', 'bias', 'mean', 'var'],
                          ['y'], is_test=is_test)], shape=(1, 1), opset=6,
        initializers=[constant('scale', [1]), constant('bias', [0]),
                      constant('mean', [0]), constant('var', [1])],
        error='is_test=0' if not is_test else None)
check_case('legacy_shapes', [helper.make_node('Add', ['x', 'b'], ['y'])], shape=(2, 2),
           opset=6, initializers=[constant('b', [1.0, 2.0])], error='broadcast is 0')
check_case('legacy_singleton', [helper.make_node('Add', ['x', 'b'], ['y'], broadcast=1, axis=0)],
           shape=(2, 3), opset=6, initializers=[constant('b', [[1.0], [2.0]])],
           error='neither one element')
check_case('legacy_extrema', [helper.make_node('Max', ['x', 'b'], ['y'])], shape=(2, 2),
           opset=6, initializers=[constant('b', [1.0, 2.0])], error='broadcasts from opset 8')
# A source value whose name looks like a generated helper's keeps its own step, and each step
# names its source.
sources = []
collision = check_case('helper_name_collision', [
    helper.make_node('Add', ['x', 'x'], ['g__torchlean_gemm_matmul'], name='encoder/block1/add'),
    helper.make_node('Gemm', ['x', 'weight', 'bias'], ['g']),
    helper.make_node('Add', ['g__torchlean_gemm_matmul', 'g'], ['y'])],
    shape=(1, 1), initializers=[constant('weight', [[3]]), constant('bias', [[1]])],
    sources=sources)
steps = collision['nodes']
original = next(i for i, n in enumerate(steps)
                if 'op' in n and n['op']['inputs'] == [[0, 0], [0, 0]])
output = collision['outputs'][0][0]
assert steps[output]['op']['inputs'][0] == [original, 0], 'a helper step shadowed a source value'
assert sources[0] == ('input', None, 'x'), sources
assert sources[original] == ('Add', 'encoder/block1/add', 'g__torchlean_gemm_matmul'), sources
assert sources[output] == ('Add', '', 'y') and ('initializer', None, 'weight') in sources, sources
(work / 'manifest.json').write_text(json.dumps(manifest))
"#

/-- How an output entry is compared with its reference: by its word, by its word up to the NaN
payload (two binary32 NaNs agree), or as a binary32 value within a tolerance (two NaNs agree). -/
private inductive Compare where
  | words
  | nanWords
  | within (tolerance : Float)

/-- Whether word `a` agrees with the reference word `b`. -/
private def Compare.agree (compare : Compare) (a b : Nat) : Bool :=
  let x := Float32.ofBits a.toUInt32
  let y := Float32.ofBits b.toUInt32
  match compare with
  | .words => a == b
  | .nanWords => a == b || (x.isNaN && y.isNaN)
  | .within tolerance => (x.isNaN && y.isNaN) || Float.abs (x.toFloat - y.toFloat) ≤ tolerance

/-- The element type of a floating ONNX type code (`TensorProto.DataType`). -/
private def elemType? : Nat → Option DType
  | 1 => some .float32
  | 10 => some .float16
  | 11 => some .float64
  | 16 => some .bfloat16
  | _ => none

/-- The comparison a manifest entry names: none for an artifact that is only loaded. -/
private def Compare.ofJson? : Json → Option (Option Compare)
  | .null => some none
  | .str "words" => some (some .words)
  | .str "nan" => some (some .nanWords)
  | .num n => some (some (.within n.toFloat))
  | _ => none

/-- The bit words of a loaded floating value at a value port, the Booleans as 0 and 1, the integer
words, and the components of index tuples, in order. -/
private def words? : Program.Value .reference → Option (List Nat)
  | ⟨⟨⟨.value, s⟩, some (.binary f)⟩, x⟩ =>
      some ((@Tensor.Rep.data _ (Program.Machine.reference.storage (.binary f)) s
        x).toList.map FloatLib.Floats.ExecFloat.Binary.toNatBits)
  | ⟨⟨⟨.bool, _⟩, _⟩, x⟩ => some ((Tensor.Rep.data x).toList.map (if · then 1 else 0))
  | ⟨⟨⟨.integer _ _, _⟩, _⟩, x⟩ => some ((Tensor.Rep.data x).toList.map (·.toNat))
  | ⟨⟨⟨.index bounds, _⟩, _⟩, x⟩ =>
      some ((Tensor.Rep.data x).toList.flatMap (Port.IndexTuple.components bounds))
  | _ => none

/-- Check the artifact `label`: its JSON is the JSON of its decoding, `Artifact.load` accepts it,
and its sources file names one source per step. With a comparison, the inputs also have the
reference element types, the loaded constant of the step that the sources file names for each
floating initializer has the initializer's element type, shape and words, and the run on the
reference inputs' words gives outputs of the reference element types and shapes that agree with the
reference. -/
private def check (label : String) (compare : Option Compare) : IO Unit := do
  let fail := fun (message : String) => IO.userError s!"ONNX round trip, {label}: {message}"
  let json ← Json.readFile (workDir / (label ++ ".json"))
  let artifact ← IO.ofExcept (Artifact.ofJson json)
  unless artifact.toJson.compress == json.compress do
    throw (fail "the JSON is not the JSON of its decoding")
  let (p, P) ← IO.ofExcept (artifact.load .reference ⟨fun _ => none, fun _ => none⟩)
  let sources ←
    IO.ofExcept ((← Json.readFile (workDir / (label ++ ".sources.json"))).getArr?)
  unless sources.size == p.nodes.size do throw (fail "the sources do not name every step")
  let some compare := compare | return
  let reference ← Json.readFile (workDir / (label ++ ".reference.json"))
  let inputs ← IO.ofExcept (reference.getObjValAs? (List (Array Nat)) "inputs")
  let inputTypes ← IO.ofExcept (reference.getObjValAs? (List Nat) "input_types")
  let outputs ← IO.ofExcept (reference.getObjValAs? (List (List Nat)) "outputs")
  let outputTypes ← IO.ofExcept (reference.getObjValAs? (List Nat) "output_types")
  let shapes ← IO.ofExcept (reference.getObjValAs? (List (List Nat)) "shapes")
  let constants ← IO.ofExcept (reference.getObjValAs? (List Json) "constants")
  for c in constants do
    let name ← IO.ofExcept (c.getObjValAs? String "name")
    let words ← IO.ofExcept (c.getObjValAs? (List Nat) "words")
    let shape ← IO.ofExcept (c.getObjValAs? (List Nat) "shape")
    let type ← IO.ofExcept (c.getObjValAs? Nat "type")
    let some i := sources.findIdx? (· == Json.arr #[.str "initializer", .null, .str name])
      | throw (fail s!"no step holds the initializer {name}")
    let some (some v) := P.consts[i]? | throw (fail s!"the step of {name} is not a constant")
    unless v.typed.scalar == elemType? type && v.typed.port.shape == shape &&
        words? v == some words do
      throw (fail s!"the constant of {name} has another element type, shape or words")
  let args ← (inputs.zip inputTypes).zipIdx.mapM fun ((words, type), k) => do
    let some i := p.input? k | throw (fail s!"no input {k}")
    let some (.node (.input _ port)) := p.nodes[i]? | throw (fail s!"input {k} is not a step")
    unless p.scalar i == elemType? type do throw (fail s!"input {k} has another element type")
    let t : Program.Typed := ⟨port, p.scalar i⟩
    pure (⟨t, ← IO.ofExcept (t.decode .reference words)⟩ : Program.Value .reference)
  let (ys, _) ← IO.ofExcept (p.eval .reference P args.toArray)
  unless ys.length == outputs.length && ys.length == shapes.length &&
      ys.length == outputTypes.length do
    throw (fail "the output count differs")
  for (((y, expected), shape), type) in ((ys.zip outputs).zip shapes).zip outputTypes do
    unless y.typed.scalar == elemType? type do throw (fail "an output has another element type")
    unless y.typed.port.shape == shape do throw (fail "an output shape differs")
    let some words := words? y | throw (fail "an output has no words")
    unless words.length == expected.length &&
        (words.zip expected).all fun (a, b) => compare.agree a b do
      throw (fail "an output differs from the reference")

/-- Run the ONNX fixture and the bridge's command line, then check each artifact of the manifest. -/
def run : IO Unit := do
  let available ← Subprocess.pythonCanImport #["onnx", "numpy"]
  unless ← Harness.checkInteropDependency "onnx, numpy" available do
    IO.println "ONNX round trip: skipped (onnx or numpy is unavailable)"
    return
  IO.FS.createDirAll workDir
  let table := workDir / "op_table.json"
  IR.Capture.writeOpTable table
  let fixture := workDir / "make_model.py"
  IO.FS.writeFile fixture modelScript
  discard <| IR.Capture.runPython "ONNX fixture"
    #[fixture.toString, workDir.toString] table
  discard <| IR.Capture.runPython "ONNX bridge"
    #["-m", "torchlean.onnx", (workDir / "model.onnx").toString, (workDir / "model.json").toString,
      "--sources", (workDir / "model.sources.json").toString] table
  let steps ← IO.ofExcept
    ((← Json.readFile (workDir / "model.json")).getObjValAs? (Array Json) "nodes")
  let sources ← IO.ofExcept ((← Json.readFile (workDir / "model.sources.json")).getArr?)
  unless sources.size == steps.size &&
      sources[0]? == some (Json.arr #[.str "input", .null, .str "x"]) do
    throw (IO.userError "ONNX round trip: the --sources file does not name each step's source")
  let entries ← IO.ofExcept ((← Json.readFile (workDir / "manifest.json")).getArr?)
  for entry in entries do
    let label ← IO.ofExcept (entry.getObjValAs? String "label")
    let some compare := Compare.ofJson? (entry.getObjValD "compare")
      | throw (IO.userError s!"ONNX round trip, {label}: unknown comparison")
    check label compare
  IO.println s!"ONNX round trip: {entries.size} artifacts ok"

end TorchLean.Test.ONNX
