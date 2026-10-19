/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Core.ExternalProcess
public import NN.Runtime.PyTorch.Export.TorchExport
public import NN.Runtime.PyTorch.Import.TorchExport
public import NN.Spec.Layers.MoE
public import NN.Tests.Utils
public import NN.Verification.IR

/-!
# A PyTorch mixture of experts through TorchLean IR

Mixtral's MoE block in static-shape form, written in ordinary PyTorch (`torch.topk` and `scatter`
for the routing, a slice of the gates per expert), is captured with `torch.export` by the
generated adapter and imported as TorchLean IR, where the routing becomes one `top_k_mask` node.

* Lean's reference interpreter, which evaluates binary32 as IEEE 754 prescribes, agrees with
  PyTorch's output to within a few units in the last place of the largest output; entries near
  zero, where the two experts' terms cancel, are held to the same absolute bound. A different
  routing decision would change an entry by the size of an expert's output.
* Evaluated in `Float`, the imported graph computes `Spec.MoE.forwardSpec` bit for bit, with the
  router and the experts read from the artifact's weights.
-/

@[expose] public section

namespace NN.Tests.IR.MoEImport

open Lean Spec TorchLean TorchLean.Tensor

def workDir : System.FilePath :=
  TorchLean.External.Process.directory "moe_import"

/-- The PyTorch model and the export driver: six tokens of width 8, four experts of two layers
with 16 hidden units, two experts per token. The artifact carries the input and PyTorch's output
as binary32 words. -/
def driverScript : String := "\n".intercalate
  [ "import importlib.util, json, sys"
  , "from pathlib import Path"
  , "import numpy as np"
  , "import torch, torch.nn as nn, torch.nn.functional as F"
  , "work = Path(sys.argv[1])"
  , "spec = importlib.util.spec_from_file_location('adapter', work / 'export_torchlean.py')"
  , "adapter = importlib.util.module_from_spec(spec)"
  , "spec.loader.exec_module(adapter)"
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
  , "def tensor(t):"
  , "    t = t.detach().contiguous().to(torch.float32)"
  , "    words = t.numpy().view(np.uint32).reshape(-1)"
  , "    return {'dtype': 'float32', 'shape': list(t.shape), 'bits': [int(w) for w in words]}"
  , "torch.manual_seed(0)"
  , "model = DenseMoE().eval()"
  , "x = torch.randn(6, 8)"
  , "path = work / 'moe.json'"
  , "adapter.export_torchlean_graph_json(model, x, str(path), True)"
  , "artifact = json.loads(path.read_text())"
  , "artifact['arguments'] = [tensor(x)]"
  , "artifact['reference_outputs'] = [tensor(model(x))]"
  , "path.write_text(json.dumps(artifact))"
  ]

def readJson (path : System.FilePath) : IO Json := do
  IO.ofExcept (Json.parse (← IO.FS.readFile path))

/-- The binary32 words of the first tensor of an array field. -/
def firstWords (json : Json) (field : String) : Except String (Array Nat) := do
  let some first := (← (← json.getObjVal? field).getArr?)[0]? | throw s!"no {field}"
  (← (← first.getObjVal? "bits").getArr?).mapM fun word => word.getNat?

/-- The value of a binary32 word. -/
def ofWord (w : Nat) : Float := (Float32.ofBits w.toUInt32).toFloat

/-- `x ↦ W x + b` for the linear payload of node `id`, as the IR evaluates it on each row. -/
def linearOf (payload : NN.IR.Payload Float) (id inDim outDim : Nat) :
    Except String (Tensor Float [inDim] → Tensor Float [outDim]) := do
  let some p := payload.linear? id | throw s!"MoE import: node {id} has no linear payload"
  if h : p.inDim = inDim ∧ p.outDim = outDim then
    let W : Tensor Float [outDim, inDim] := Tensor.castShape p.W (by rw [h.1, h.2])
    let b : Tensor Float [outDim] := Tensor.castShape p.b (by rw [h.2])
    pure fun v => Tensor.addSpec (Spec.matVecMulSpec W v) b
  else
    throw s!"MoE import: linear node {id} is {p.outDim} by {p.inDim}"

/-- Equal as bit patterns, entry by entry. -/
def sameBits {s : Shape} (x y : Tensor Float s) : Bool :=
  Tensor.foldlSpec (fun acc pair => acc && pair.1.toBits == pair.2.toBits) true
    (Tensor.map2Spec (fun a b => (a, b)) x y)

def run : IO Unit := do
  let available ← TorchLean.External.Process.pythonCanImport #["torch", "numpy"]
  unless ← Tests.Utils.checkInteropDependency "torch, numpy" available do
    IO.println "PyTorch mixture of experts through TorchLean IR: skipped (torch is unavailable)"
    return
  IO.FS.createDirAll workDir
  IO.FS.writeFile (workDir / "export_torchlean.py") (Export.PyTorch.TorchExport.script {})
  IO.FS.writeFile (workDir / "make_moe.py") driverScript
  discard <| TorchLean.External.Process.run "MoE export" "python3"
    #[(workDir / "make_moe.py").toString, workDir.toString]
  let artifact ← readJson (workDir / "moe.json")
  -- The routing is one top-2 mask.
  let captured ← IO.ofExcept (Import.PyTorch.TorchExport.parseGraph artifact)
  let masks := captured.graph.nodes.filter fun node => node.kind == .topKMask 2
  unless masks.size == 1 do
    throw <| IO.userError s!"MoE import: expected one top-2 mask, found {masks.size}"
  -- Lean's binary32 reference execution against PyTorch's output.
  let report := NN.Verification.IR.check artifact #["structure", "types", "execute"]
  unless report.getObjValAs? Bool "ok" == .ok true do
    throw <| IO.userError s!"MoE import: reference execution failed: {report.compress}"
  let lean ← IO.ofExcept (firstWords report "outputs")
  let torch ← IO.ofExcept (firstWords artifact "reference_outputs")
  unless lean.size == torch.size && lean.size == 48 do
    throw <| IO.userError "MoE import: output sizes differ"
  let largest := torch.foldl (fun acc w => max acc (ofWord w).abs) 0
  let unit := largest * Float.ofScientific 1 true 0 / 16777216
  let worst := (lean.zip torch).foldl (fun acc (a, b) => max acc ((ofWord a - ofWord b).abs)) 0
  unless worst ≤ 8 * unit do
    throw <| IO.userError
      s!"MoE import: Lean and PyTorch differ by {worst / unit} units of the largest output"
  -- The imported graph in `Float` against the spec's layer with the same weights.
  let payload ← IO.ofExcept (Import.PyTorch.TorchExport.parsePayload artifact)
  let inputWords ← IO.ofExcept (firstWords artifact "arguments")
  let x : Tensor Float [6, 8] := Tensor.generate [6, 8] fun c =>
    (Float32.ofBits (inputWords[8 * c.getD 0 0 + c.getD 1 0]!).toUInt32).toFloat
  let values ← IO.ofExcept (captured.graph.denoteAll payload (Spec.SomeTensor.mk [6, 8] x))
  let some outputId := captured.outputIds[0]? | throw <| IO.userError "MoE import: no output"
  let some output := values[outputId]? | throw <| IO.userError "MoE import: missing output value"
  let actual ← IO.ofExcept (NN.IR.Graph.expectShape (expected := [6, 8]) output)
  let linears := (captured.graph.nodes.filter fun node => node.kind == .linear).map (·.id)
  unless linears.size == 9 do
    throw <| IO.userError s!"MoE import: expected 9 linear layers, found {linears.size}"
  let router ← IO.ofExcept (linearOf payload linears[0]! 8 4)
  let firsts ← IO.ofExcept ((List.range 4).mapM fun e => linearOf payload linears[1 + 2 * e]! 8 16)
  let seconds ← IO.ofExcept ((List.range 4).mapM fun e => linearOf payload linears[2 + 2 * e]! 16 8)
  let experts : Fin 4 → Tensor Float [8] → Tensor Float [8] := fun e v =>
    (seconds[e.val]!) (Activation.reluSpec ((firsts[e.val]!) v))
  let expected : Tensor Float [6, 8] := Tensor.dim fun t =>
    Spec.MoE.forwardSpec 2 router experts (Tensor.unstack x t)
  unless sameBits actual expected do
    throw <| IO.userError "MoE import: the imported graph differs from Spec.MoE.forwardSpec"
  IO.println s!"PyTorch mixture of experts through TorchLean IR: one top-2 mask, \
    within {(worst / unit).ceil} units of PyTorch, Spec.MoE.forwardSpec bit for bit"

end NN.Tests.IR.MoEImport
