/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.Subprocess
public import TorchLean.NN.IR.Loaded
public import TorchLean.Tensor.Reduce.Fold
public import TorchLean.Tensor.Layers.MoE
public import TorchLeanTest.IR.Capture
public import TorchLeanTest.Harness
public import TorchLean.Verify.Capture

/-!
# A PyTorch mixture of experts through TorchLean IR

Mixtral's MoE block in static-shape form, written in ordinary PyTorch (`torch.topk` and `scatter`
for the routing, a slice of the gates per expert), is captured with `torch.export` by
`python3 -m torchlean.export`. The exporter reads Lean's op table (`IR.table`, written to
the work directory when the test runs), and `IR.load` loads the capture document. The
routing becomes one `Op.topKMask` step. A mask built from float64 zeros over float32 scores has no
such encoding, and the capture refuses it.

* Lean's reference interpreter, which evaluates binary32 as IEEE 754 prescribes, agrees with
  PyTorch's output to within `8 · m / 2^24` for the largest output magnitude `m`, four to eight
  units in the last place of `m`. Entries near zero, where the two experts' terms cancel, are held
  to the same absolute bound. A different routing decision would change an entry by the size of an
  expert's output.
* Evaluated in `Float`, the artifact's operator graph computes `TorchLean.MoE.forward` bit for
  bit, with the router and the experts read from the artifact's weights.
-/

@[expose] public section

namespace TorchLean.Test.MoEImport

open Lean TorchLean.Tensor
open Test.IR

/-- Scratch directory of this check, `.lake/build/torchlean_moe_import`. -/
def workDir : System.FilePath :=
  Subprocess.directory "moe_import"

/-- The PyTorch model file: six tokens of width 8, four experts of two layers with 16 hidden units,
two experts per token, built by the seeded constructors `dense` and `wide` that the capture calls.
Run as a script on a capture document, it adds the input and PyTorch's output as binary32 words. -/
def modelScript : String := "\n".intercalate
  [ "import json, sys"
  , "from pathlib import Path"
  , "import numpy as np"
  , "import torch, torch.nn as nn, torch.nn.functional as F"
  , "class DenseMoE(nn.Module):"
  , "    def __init__(self, width=8, hidden=16, experts=4, k=2):"
  , "        super().__init__()"
  , "        self.k = k"
  , "        self.gate = nn.Linear(width, experts, bias=False)"
  , "        self.experts = nn.ModuleList("
  , "            nn.Sequential(nn.Linear(width, hidden), nn.ReLU(), nn.Linear(hidden, width))"
  , "            for _ in range(experts))"
  , "    def forward(self, x):"
  , "        probabilities = F.softmax(self.gate(x), dim=-1)"
  , "        _, chosen = torch.topk(probabilities, self.k, dim=-1)"
  , "        mask = torch.zeros_like(probabilities).scatter(-1, chosen, 1.0)"
  , "        kept = probabilities * mask"
  , "        gates = kept / kept.sum(dim=-1, keepdim=True)"
  , "        out = torch.zeros_like(x)"
  , "        for e, expert in enumerate(self.experts):"
  , "            out = out + gates[:, e:e + 1] * expert(x)"
  , "        return out"
  , "class WideMask(DenseMoE):"
  , "    def forward(self, x):"
  , "        probabilities = F.softmax(self.gate(x), dim=-1)"
  , "        _, chosen = torch.topk(probabilities, self.k, dim=-1)"
  , "        zeros = torch.zeros_like(probabilities, dtype=torch.float64)"
  , "        return zeros.scatter(-1, chosen, 1.0)"
  , "# The capture draws its example input after the constructor, as the script below does."
  , "def dense():"
  , "    torch.manual_seed(0)"
  , "    return DenseMoE()"
  , "def wide():"
  , "    torch.manual_seed(0)"
  , "    return WideMask()"
  , "def tensor(t):"
  , "    t = t.detach().contiguous().to(torch.float32)"
  , "    words = t.numpy().view(np.uint32).reshape(-1)"
  , "    return {'dtype': 'float32', 'shape': list(t.shape), 'bits': [int(w) for w in words]}"
  , "if __name__ == '__main__':"
  , "    model = dense().eval()"
  , "    x = torch.randn(6, 8)"
  , "    path = Path(sys.argv[1])"
  , "    document = json.loads(path.read_text())"
  , "    document['arguments'] = [tensor(x)]"
  , "    document['reference_outputs'] = [tensor(model(x))]"
  , "    path.write_text(json.dumps(document))"
  ]

/-- The binary32 words of the first tensor of an array field. -/
def firstWords (json : Json) (field : String) : Except String (Array Nat) := do
  let some first := (← (← json.getObjVal? field).getArr?)[0]? | throw s!"no {field}"
  (← (← first.getObjVal? "bits").getArr?).mapM fun word => word.getNat?

/-- The value of a binary32 word. -/
def ofWord (w : Nat) : Float := (Float32.ofBits w.toUInt32).toFloat

/-- The operator graph of a capture document's artifact, loaded by `IR.load`, whose steps are
operator-graph nodes and whose scalars are binary32, each constant read in `Float` from the binary32
words that `IR.words` gives back. -/
def graphOf (document : Json) : Except String (IR.Graph × IR.Payload Float) := do
  let (program, payload) ← IR.load document
  let some (graph, scalar) := program.graph?
    | throw "MoE import: the artifact has a step outside an operator graph"
  unless scalar == some .float32 do throw "MoE import: the artifact is not binary32"
  let consts ← graph.nodes.mapIdxM fun i n => do
    match n, (payload.consts[i]?).join with
    | .const ⟨.value, s⟩, some v =>
        let (_, _, words) ← IR.words v
        unless words.size == s.size do throw s!"MoE import: constant {i} has {words.size} words"
        let value : Op.Value Float :=
          ⟨⟨.value, s⟩, Tensor.generateFlat s fun j => ofWord words[j]!⟩
        pure (some value)
    | .const _, _ => throw s!"MoE import: constant {i} is not a value tensor with words"
    | _, _ => pure none
  pure (graph, { const? := fun m => (consts[m]?).join })

/-- `x ↦ W x + b` for the `k`-th `Op.linear` node of `g`, where `W` and `b` are the constants of
`P` that its first two inputs read, which is the operator's computation on one row. -/
def linearOf (g : IR.Graph) (P : IR.Payload Float) (k inDim outDim : Nat) :
    Except String (Tensor Float [inDim] → Tensor Float [outDim]) := do
  let linears := g.nodes.filterMap fun
    | .op ⟨_, _, .linear ..⟩ refs => some refs
    | _ => none
  let some refs := linears[k]? | throw s!"MoE import: no linear node {k}"
  let constant := fun (j : Nat) => (refs[j]?).bind fun ref => P.const? ref.node
  match constant 0, constant 1 with
  | some ⟨⟨.value, s⟩, W⟩, some ⟨⟨.value, t⟩, b⟩ =>
      if h : s = [outDim, inDim] ∧ t = [outDim] then
        let W : Tensor Float [outDim, inDim] := h.1 ▸ W
        let b : Tensor Float [outDim] := h.2 ▸ b
        pure fun v => Tensor.contract W v _ _ [outDim] .matVec + b
      else throw s!"MoE import: linear node {k} is {repr s} with a bias {repr t}"
  | _, _ => throw s!"MoE import: linear node {k} reads no weight and bias constants"

/-- Equal as bit patterns, entry by entry. -/
def sameBits {s : Shape} (x y : Tensor Float s) : Bool :=
  Tensor.foldl (fun acc pair => acc && pair.1.toBits == pair.2.toBits) true
    (Tensor.zipWith (fun a b => (a, b)) x y)

/-- When Python cannot import torch and numpy, skip, or fail under `TORCHLEAN_REQUIRE_INTEROP=1`.
Otherwise capture `modelScript`, check that the `wide` routing is refused, and compare the
import with PyTorch's output and with `TorchLean.MoE.forward`. -/
def run : IO Unit := do
  let available ← Subprocess.pythonCanImport #["torch", "numpy"]
  unless ← Harness.checkInteropDependency "torch, numpy" available do
    IO.println "PyTorch mixture of experts through TorchLean IR: skipped (torch is unavailable)"
    return
  IO.FS.createDirAll workDir
  let table := workDir / "op_table.json"
  Capture.writeOpTable table
  let model := (workDir / "moe_model.py").toString
  IO.FS.writeFile model modelScript
  let capture (ctor : String) (path : System.FilePath) : Array String :=
    #["-m", "torchlean.export", model, ctor, path.toString, "--example-shape", "6,8",
      "--require-torch-export"]
  discard <| Capture.runPython "MoE export" (capture "dense" (workDir / "moe.json")) table
  discard <| Capture.runPython "MoE reference" #[model, (workDir / "moe.json").toString] table
  let document ← Json.readFile (workDir / "moe.json")
  -- Routing whose zeros are wider than the scores has no `Op.topKMask` step, and the capture
  -- refuses it. The capture of `dense` from the same file succeeded above.
  let wide ← Capture.python (capture "wide" (workDir / "wide.json")) table
  if wide.exitCode == 0 then
    throw <| IO.userError "MoE import: a float64 mask over float32 scores was exported"
  -- The routing is one top-2 mask.
  let (graph, payload) ← IO.ofExcept (graphOf document)
  let masks := graph.nodes.filter fun
    | .op ⟨_, _, .topKMask _ 2⟩ _ => true
    | _ => false
  unless masks.size == 1 do
    throw <| IO.userError s!"MoE import: expected one top-2 mask, found {masks.size}"
  -- Lean's binary32 reference execution against PyTorch's output.
  let report := Capture.check document #["structure", "types", "execute"]
  unless report.getObjValAs? Bool "ok" == .ok true do
    throw <| IO.userError s!"MoE import: reference execution failed: {report.compress}"
  let lean ← IO.ofExcept (firstWords report "outputs")
  let torch ← IO.ofExcept (firstWords document "reference_outputs")
  unless lean.size == torch.size && lean.size == 48 do
    throw <| IO.userError "MoE import: output sizes differ"
  let largest := torch.foldl (fun acc w => max acc (ofWord w).abs) 0
  let unit := largest * Float.ofScientific 1 true 0 / 16777216
  let worst := (lean.zip torch).foldl (fun acc (a, b) => max acc ((ofWord a - ofWord b).abs)) 0
  unless worst ≤ 8 * unit do
    throw <| IO.userError
      s!"MoE import: Lean and PyTorch differ by {worst / unit} units of the largest output"
  -- The artifact's graph in `Float` against the spec's layer with the same weights.
  let inputWords ← IO.ofExcept (firstWords document "arguments")
  let x : Tensor Float [6, 8] := Tensor.generate [6, 8] fun c =>
    (Float32.ofBits (inputWords[8 * c.getD 0 0 + c.getD 1 0]!).toUInt32).toFloat
  let actual ← match graph.eval payload [⟨.value [6, 8], x⟩] with
    | .ok [⟨⟨.value, s⟩, y⟩] =>
        if h : s = [6, 8] then pure (h ▸ y)
        else throw <| IO.userError s!"MoE import: the output has shape {repr s}"
    | .ok _ => throw <| IO.userError "MoE import: expected one value output"
    | .error e => throw <| IO.userError s!"MoE import: {e}"
  let linears := graph.nodes.filter fun
    | .op ⟨_, _, .linear ..⟩ _ => true
    | _ => false
  unless linears.size == 9 do
    throw <| IO.userError s!"MoE import: expected 9 linear layers, found {linears.size}"
  let router ← IO.ofExcept (linearOf graph payload 0 8 4)
  let firsts ← IO.ofExcept ((List.range 4).mapM fun e => linearOf graph payload (1 + 2 * e) 8 16)
  let seconds ← IO.ofExcept ((List.range 4).mapM fun e => linearOf graph payload (2 + 2 * e) 16 8)
  let experts : Fin 4 → Tensor Float [8] → Tensor Float [8] := fun e v =>
    (seconds[e.val]!) (Tensor.relu ((firsts[e.val]!) v))
  let expected : Tensor Float [6, 8] := Tensor.dim fun t =>
    MoE.forward 2 router experts (Tensor.unstack x t)
  unless sameBits actual expected do
    throw <| IO.userError "MoE import: the artifact's graph differs from TorchLean.MoE.forward"
  IO.println s!"PyTorch mixture of experts through TorchLean IR: one top-2 mask, \
    within {(worst / unit).ceil} units of PyTorch, TorchLean.MoE.forward bit for bit"

end TorchLean.Test.MoEImport
