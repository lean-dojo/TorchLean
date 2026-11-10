/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.IR.Builder
public import TorchLean.NN.PyTorch.Export.IR
public import TorchLeanTest.Harness.Interop

/-!
# IRPyTorch Parity at Special Values

Runs the module that `TorchLean.PyTorch.emit` writes for SiLU and both GELU formulas in binary64 at
`±∞`, `±0`, a NaN and four finite points. It compares the result with the graph's evaluation in
`Float`, where `erf` below supplies the error function that `Float` lacks. The words agree at `±∞`
and `±0`, both results are NaNs at the NaN, and elsewhere they agree to a relative `1e-12`. The
autograd slopes of each formula at `-∞` and `+∞` must be `0` and `1`. `runCases` then runs the
modules of `cases` in binary64 and binary32. They are the extrema at every pair and `relu` at each
of `±0`, `±1`, `±∞` and a NaN, a product whose regrouped partial products overflow, a mean over an
empty axis, `topk` at ties and the draws of key `0`. Their words must equal `Graph.eval`'s in
`Float` and `Float32`, a NaN matching any NaN. Both checks are skipped when Python cannot import
PyTorch.
-/

@[expose] public section

open Lean
open TorchLean.Test.Harness

namespace TorchLean.Test
namespace IRPyTorch

/-- The points: `-∞`, `+∞`, `-0`, `+0`, a NaN and four finite inputs. -/
def points : Array Float := #[-1 / 0, 1 / 0, -0.0, 0, 0 / 0, 1, -1, 30, -30]

/-- `erf x` in binary64: `±1` where `|x| ≥ 6`, to which `erf` rounds, and otherwise the series
`2 / √π · exp (-x²) · Σ (2x²)ⁿ x / (2n + 1)!!` to `n = 128`, whose terms have the sign of `x`. -/
def erf (x : Float) : Float := Id.run do
  if x ≥ 6 then return 1
  if x ≤ -6 then return -1
  let mut term := x
  let mut sum := x
  for n in [0:128] do
    term := term * (2 * x * x) / (2 * n.toFloat + 3)
    sum := sum + term
  return 1.1283791670955126 * Float.exp (-(x * x)) * sum

/-- An input `[9]` read by SiLU, the tanh GELU and the exact GELU, whose results are concatenated
in that order. -/
def graph : Except String IR.Graph :=
  let b : IR.Graph.Builder := {}
  let (b, x) := b.input 0 (.value [9])
  let (b, silu) := b.op (Op.silu (s := [9])) [x]
  let (b, geluTanh) := b.op (Op.gelu (s := [9]) .tanh) [x]
  let (b, gelu) := b.op (Op.gelu (s := [9]) .none) [x]
  let (b, y) := b.op (Op.cat (s := [9]) ⟨0, by decide⟩ [9, 9, 9])
    (silu ++ geluTanh ++ gelu)
  b.build #[y.headD x]

/-- The graph's outputs at `points` in `Float`, with `erf` for the error function. -/
def reference : Except String (List (Op.Value Float)) :=
  letI : Erf Float := ⟨erf⟩
  graph >>= fun g => g.eval ({} : IR.Payload Float)
    [⟨.value [9], Tensor.generateFlat [9] fun i => points[i]!⟩]

/-- The scratch directory of the check. -/
def workDir : System.FilePath :=
  Subprocess.directory "irpytorch_parity"

/-- The Python file the check runs: the emitted module, then `script`. -/
def scriptPath : System.FilePath :=
  workDir / "parity.py"

/-- The module at `points` in binary64 and the autograd slopes of each formula at `-∞` and `+∞`,
printed as words. -/
def script : String :=
  String.intercalate "\n"
    [ "import json"
    , "import struct"
    , s!"WORDS = {(points.map (·.toBits.toNat)).toList}"
    , "n = len(WORDS)"
    , "x = torch.tensor([struct.unpack('<d', struct.pack('<Q', w))[0] for w in WORDS],"
    , "    dtype=torch.float64, requires_grad=True)"
    , "y = ExportedIRModel()(x)"
    , "slopes = []"
    , "for k in range(3):"
    , "    (g,) = torch.autograd.grad(y[k * n:(k + 1) * n].sum(), x, retain_graph=True)"
    , "    slopes += g[:2].tolist()"
    , "def words(values):"
    , "    return [struct.unpack('<Q', struct.pack('<d', v))[0] for v in values]"
    , "print(json.dumps({'values': words(y.detach().tolist()), 'slopes': words(slopes)}))"
    ]

/-! ## Selections, a product over axes, an empty mean and the random nodes -/

/-- The selection points: `±0`, `±1`, `±∞` and a NaN. -/
def specials : Array Float := #[0, -0.0, 1, -1, 1 / 0, -1 / 0, 0 / 0]

/-- An input `[7]` placed along each axis of `[7, 7]`, so that entry `(i, j)` pairs entries `i`
and `j`. The graph concatenates `maximum`, `minimum`, `fmax` and `fmin` of the pairs, `amax` and
`amin` over each pair and the `relu` of the first placement, in that order. -/
def extremaGraph : Except String IR.Graph :=
  let b : IR.Graph.Builder := {}
  let (b, x) := b.input 0 (.value [7])
  let (b, a) := b.op (Op.broadcast (s := [7]) (t := [7, 7])
    (fun _ => ⟨0, by decide⟩) (by decide)) [x]
  let (b, c) := b.op (Op.broadcast (s := [7]) (t := [7, 7])
    (fun _ => ⟨1, by decide⟩) (by decide)) [x]
  let (b, mx) := b.op (Op.max (s := [7, 7])) (a ++ c)
  let (b, mn) := b.op (Op.min (s := [7, 7])) (a ++ c)
  let (b, fx) := b.op (Op.fmax (s := [7, 7])) (a ++ c)
  let (b, fn) := b.op (Op.fmin (s := [7, 7])) (a ++ c)
  let (b, a1) := b.op (Op.broadcast (s := [7]) (t := [7, 7, 1])
    (fun _ => ⟨0, by decide⟩) (by decide)) [x]
  let (b, c1) := b.op (Op.broadcast (s := [7]) (t := [7, 7, 1])
    (fun _ => ⟨1, by decide⟩) (by decide)) [x]
  let (b, pairs) := b.op (Op.cat (s := [7, 7, 1]) ⟨2, by decide⟩ [1, 1]) (a1 ++ c1)
  let (b, ax) := b.op (Op.reduce (s := [7, 7, 2]) .max [⟨2, by decide⟩] false
    (by decide)) pairs
  let (b, an) := b.op (Op.reduce (s := [7, 7, 2]) .min [⟨2, by decide⟩] false
    (by decide)) pairs
  let (b, r) := b.op (Op.relu (s := [7, 7])) a
  let (b, y) := b.op (Op.cat (s := [7, 7]) ⟨0, by decide⟩ [7, 7, 7, 7, 7, 7, 7])
    (mx ++ mn ++ fx ++ fn ++ ax ++ an ++ r)
  b.build #[y.headD x]

/-- An input `[2, 3, 2]` read by the product over axes `0` and `2`, kept with extent one. -/
def productGraph : Except String IR.Graph :=
  let b : IR.Graph.Builder := {}
  let (b, x) := b.input 0 (.value [2, 3, 2])
  let (b, y) := b.op (Op.reduce (s := [2, 3, 2]) .prod
    [⟨0, by decide⟩, ⟨2, by decide⟩] true (by decide)) [x]
  b.build #[y.headD x]

/-- An input `[2, 0]` read by its mean over axis `1`. -/
def meanGraph : Except String IR.Graph :=
  let b : IR.Graph.Builder := {}
  let (b, x) := b.input 0 (.value [2, 0])
  let (b, y) := b.op (Op.mean (s := [2, 0]) [⟨1, by decide⟩] false) [x]
  b.build #[y.headD x]

/-- An input `[9]` read by its `8` largest and its `4` smallest entries, concatenated as the
largest values, their indices, the smallest values and their indices. -/
def topkGraph : Except String IR.Graph :=
  let b : IR.Graph.Builder := {}
  let (b, x) := b.input 0 (.value [9])
  let (b, large) := b.op (Op.topk (s := [9]) ⟨0, by decide⟩ 8 true (by decide)) [x]
  let (b, small) := b.op (Op.topk (s := [9]) ⟨0, by decide⟩ 4 false (by decide)) [x]
  let (b, i) := b.op (Op.cast (s := [8]) (.index 9)) (large.drop 1)
  let (b, j) := b.op (Op.cast (s := [4]) (.index 9)) (small.drop 1)
  let (b, y) := b.op (Op.cat (s := [8]) ⟨0, by decide⟩ [8, 8, 4, 4])
    (large.take 1 ++ i ++ small.take 1 ++ j)
  b.build #[y.headD x]

/-- A scalar input `p` broadcast to a scalar, then the uniform draws of `[2, 3]` from key `0` and
the Bernoulli mask of `[2, 3]` from key `0` with keep probability `p`, concatenated. -/
def drawsGraph : Except String IR.Graph :=
  let b : IR.Graph.Builder := {}
  let (b, p) := b.input 0 (.value .scalar)
  let (b, q) := b.op (Op.broadcast (s := .scalar) (t := .scalar) (fun i => i.elim0)
    (Shape.BroadcastDims.scalar .scalar)) [p]
  let (b, u) := b.op (Op.randUniform (s := [2, 3]) 0) []
  let (b, m) := b.op (Op.bernoulliMask (s := [2, 3]) 0) q
  let (b, y) := b.op (Op.cat (s := [2, 3]) ⟨0, by decide⟩ [2, 2]) (u ++ m)
  b.build #[y.headD p]

/-- The first draw of key `0`, a keep probability equal to the draw it meets. -/
def firstDraw : Float := Random.sampleUnit (Random.sampleNat 0 0) (2 ^ 32)

/-- A graph whose exported module is run, the shape of its input and its inputs, each run in
binary64 and, entry by entry through `Float.toFloat32`, in binary32. -/
structure Case where
  /-- The graph. -/
  graph : Except String IR.Graph
  /-- The shape of its input. -/
  shape : Shape
  /-- The inputs, each in row-major order. -/
  inputs : List (Array Float)

/-- The cases: the extrema and `relu` at every pair of `specials`, the product at fibers whose
regrouped partial products overflow (`2⁻¹⁰²³ · 1 · 2¹⁰²³ · 2` in binary64,
`2⁻¹⁴⁹ · 1 · 2¹²⁷ · 2` in binary32) and the mean over an empty axis. They end with `topk` at ties
of `+0` and `-0`, of `1`, of `3` and of two NaNs, and with the draws at keep probabilities `±0`,
`1`, `1 / 2`, a NaN, `-1`, `2`, `+∞`, the first draw and `3 / 4`. -/
def cases : List Case :=
  [ ⟨extremaGraph, [7], [specials]⟩
  , ⟨productGraph, [2, 3, 2],
      [ #[Float.ofBits 0x0008000000000000, 1, 1, 2, -1, 0, Float.ofBits 0x7fe0000000000000, 2, 3,
          4, 1, -0.0]
      , #[Float.ofBits 0x36a0000000000000, 1, 1, 2, -1, 0, Float.ofBits 0x47e0000000000000, 2, 3,
          4, 1, -0.0] ]⟩
  , ⟨meanGraph, [2, 0], [#[]]⟩
  , ⟨topkGraph, [9], [#[1, 0 / 0, 0, 3, -0.0, 1, 0 / 0, -1 / 0, 3]]⟩
  , ⟨drawsGraph, .scalar,
      [0, -0.0, 1, 0.5, 0 / 0, -1, 2, 1 / 0, firstDraw, 0.75].map fun p => #[p]⟩ ]

/-- The words of the value output of `g` in `α` at the entries `x` of an input of shape `s`. -/
def evalWords {α : Type} [Storage α] [Context α] [Op.Capabilities α] (word : α → Nat)
    (g : IR.Graph) (s : Shape) (x : Array α) :
    Except String (Array Nat) := do
  match ← g.eval ({} : IR.Payload α) [⟨.value s, Tensor.generateFlat s fun i => x[i]!⟩] with
  | [⟨⟨.value, _⟩, y⟩] => pure ((Tensor.to y (Array α)).map word)
  | _ => throw "the graph has not one value output"

/-- The Python program of `cases`: the module of each case in binary64 and in binary32, then a
script that prints the output words of each run. It comes with the words `Graph.eval` gives for
each run in `Float` or `Float32` and whether the run is in binary64. -/
def casesProgram : Except String (String × Array (Array Nat × Bool)) := do
  let mut modules := ""
  let mut runs : Array String := #[]
  let mut expected : Array (Array Nat × Bool) := #[]
  for (c, k) in cases.zipIdx do
    let g ← c.graph
    for (bits, dtype) in [(64, "torch.float64"), (32, "torch.float32")] do
      let name := s!"Case{k}F{bits}"
      modules := modules ++ (← PyTorch.emit ⟨g, {}, #[]⟩
        { className := name, dtypeExpr := dtype, includeTrainingSkeleton := false,
          learnableConsts := false }) ++ "\n\n"
      for x in c.inputs do
        let (words, want) ←
          if bits == 64 then
            (x.map (·.toBits.toNat), ·) <$> evalWords (·.toBits.toNat) g c.shape x
          else
            let y := x.map Float.toFloat32
            (y.map (·.toBits.toNat), ·) <$> evalWords (·.toBits.toNat) g c.shape y
        runs := runs.push s!"('{name}', {bits}, {c.shape}, {words.toList})"
        expected := expected.push (want, bits == 64)
  let script := String.intercalate "\n"
    [ "import json"
    , "import struct"
    , s!"RUNS = [{", ".intercalate runs.toList}]"
    , "FORMATS = {64: ('<Q', '<d', torch.float64), 32: ('<I', '<f', torch.float32)}"
    , "out = []"
    , "for name, bits, shape, words in RUNS:"
    , "    word, value, dtype = FORMATS[bits]"
    , "    x = torch.tensor([struct.unpack(value, struct.pack(word, w))[0] for w in words],"
    , "        dtype=dtype).reshape(shape)"
    , "    y = globals()[name]()(x).reshape(-1).tolist()"
    , "    out.append([struct.unpack(word, struct.pack(value, v))[0] for v in y])"
    , "print(json.dumps({'runs': out}))"
    ]
  pure (modules ++ script, expected)

/-- The Python file `runCases` runs. -/
def casesPath : System.FilePath :=
  workDir / "cases.py"

/-- Run `casesProgram` and compare the words of each run with `Graph.eval`'s, a NaN matching any
NaN. -/
def runCases : IO Unit := do
  let (program, expected) ← IO.ofExcept casesProgram
  IO.FS.createDirAll workDir
  IO.FS.writeFile casesPath program
  let json ← Subprocess.runJson (ctx := "irpytorch_parity")
    (cmd := "python3") (args := #[casesPath.toString])
  let runs ← IO.ofExcept (json.getObjValAs? (Array (Array Nat)) "runs")
  unless runs.size == expected.size do
    throw (IO.userError s!"irpytorch_parity: {runs.size} runs, {expected.size} expected")
  for j in [0:runs.size] do
    let (want, wide) := expected[j]!
    let nan (w : Nat) : Bool :=
      if wide then (Float.ofBits w.toUInt64).isNaN else (Float32.ofBits w.toUInt32).isNaN
    unless runs[j]!.size == want.size &&
        (runs[j]!.zip want).all fun (a, b) => a == b || (nan a && nan b) do
      throw (IO.userError s!"irpytorch_parity: run {j}: PyTorch {runs[j]!}, Lean {want}")

/-- Compare the emitted module with `reference` at `points` and its slopes at `-∞` and `+∞` with
`0` and `1`, then run `runCases`. -/
def run : IO Unit := do
  IO.println "irpytorch_parity: begin"
  if !(← pythonHasTorch) then
    IO.println "irpytorch_parity: skipped (python package `torch` not installed)"
    return ()
  let source ← IO.ofExcept <| graph >>= fun g =>
    PyTorch.emit ⟨g, {}, #[]⟩ (options := { includeTrainingSkeleton := false })
  let expected ← match reference with
    | .ok [⟨⟨.value, _⟩, y⟩] => pure (Tensor.to y (Array Float))
    | .ok _ => throw (IO.userError "irpytorch_parity: the graph has not one value output")
    | .error e => throw (IO.userError s!"irpytorch_parity: {e}")
  IO.FS.createDirAll workDir
  IO.FS.writeFile scriptPath (source ++ "\n\n" ++ script)
  let json ← Subprocess.runJson (ctx := "irpytorch_parity")
    (cmd := "python3") (args := #[scriptPath.toString])
  let values ← IO.ofExcept (json.getObjValAs? (Array Nat) "values")
  let slopes ← IO.ofExcept (json.getObjValAs? (Array Nat) "slopes")
  unless values.size = expected.size do
    throw (IO.userError s!"irpytorch_parity: {values.size} values, {expected.size} expected")
  for j in [0:values.size] do
    let a := Float.ofBits (values[j]!).toUInt64
    let b := expected[j]!
    let p := points[j % points.size]!
    let agree :=
      if p.isNaN then a.isNaN && b.isNaN
      else if p.isInf || p == 0 then a.toBits == b.toBits
      else decide (Float.abs (a - b) ≤ 1e-12 * max a.abs b.abs)
    unless agree do
      throw (IO.userError s!"irpytorch_parity: entry {j}: PyTorch {a} ({a.toBits}), \
        Lean {b} ({b.toBits})")
  unless slopes.map (fun w => Float.ofBits w.toUInt64) == #[0, 1, 0, 1, 0, 1] do
    throw (IO.userError s!"irpytorch_parity: slopes {slopes} at -∞ and +∞, expected 0 and 1")
  runCases
  IO.println "irpytorch_parity: ok"

end IRPyTorch
end TorchLean.Test
