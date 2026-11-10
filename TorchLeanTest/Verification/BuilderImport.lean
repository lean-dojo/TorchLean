/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLeanExamples.Verification.ODE.Verify
public import TorchLeanExamples.Verification.PINN

/-!
# Checkpoints in the PyTorch state-dictionary layout build operator graphs

`run` reads PINN checkpoints in the PyTorch state-dictionary layout and builds their graphs of
registered operators: the bundled `zero_mlp`, `one_mlp` and `zero_siren` files under
`TorchLeanExamples/Verification/ODE/`, and a tanh and a ReLU checkpoint written here. The ODE
corridor graphs with their tangent graphs are built from the MLP, SIREN and tanh files on the
direct path and from the MLP and tanh files on the TorchLean model path. Each must load and build
(`Graph.Builder.build`, which runs `Graph.check`), and `run` fails when one is refused.

Whether another checkpoint builds depends on its own graph, payload and outputs, and a built graph
says nothing about a certificate's soundness.
-/

@[expose] public section

namespace TorchLean.Test.BuilderImport

/-- The number of nodes of a built graph of registered operators. -/
def builtSize {α : Type} [Storage α] (model : Except String (IR.Graph × IR.Payload α)) :
    Except String Nat :=
  model.map (·.1.nodes.size)

/-- A PINN checkpoint with two tanh hidden layers of width 3. -/
def tanhCheckpoint : String :=
  "{\"meta\": {\"activation\": \"tanh\"}, " ++
  "\"layers.0.weight\": [[0.5], [-0.25], [1.0]], \"layers.0.bias\": [0.1, 0.0, -0.1], " ++
  "\"layers.1.weight\": [[0.3, -0.2, 0.1], [0.0, 0.4, -0.5], [0.2, 0.2, 0.2]], " ++
  "\"layers.1.bias\": [0.0, 0.1, 0.0], " ++
  "\"layers.2.weight\": [[1.0, -1.0, 0.5]], \"layers.2.bias\": [0.0]}"

/-- A PINN checkpoint with one ReLU hidden layer of width 2. -/
def reluCheckpoint : String :=
  "{\"meta\": {\"activation\": \"relu\"}, " ++
  "\"layers.0.weight\": [[1.0], [-1.0]], \"layers.0.bias\": [0.0, 0.5], " ++
  "\"layers.1.weight\": [[0.5, 0.25]], \"layers.1.bias\": [0.0]}"

/-- The PINN graph of a checkpoint file. -/
def pinnCheckpoint (path : System.FilePath) : IO (Except String Nat) := do
  let j ← Json.readFile path
  match Examples.PINN.Checkpoint.ofJson? j with
  | some sd => pure (builtSize (Examples.PINN.Checkpoint.model (α := Float) (fun x => x) sd))
  | none => pure (.error s!"cannot load {path}")

/-- The ODE corridor graph with its tangent graph of a checkpoint file. -/
def odeCheckpoint (backend : Examples.ODE.ModelBackend)
    (path : System.FilePath) : IO (Except String Nat) := do
  let m ← Examples.ODE.loadModelFloat backend path.toString
  pure (.ok m.graph.nodes.size)

/-- Every checkpoint case, by name, with the number of nodes of its operator graph. -/
def cases (tanhPath reluPath : System.FilePath) : List (String × IO (Except String Nat)) :=
  let ode : System.FilePath := "TorchLeanExamples/Verification/ODE"
  [ ("PINN checkpoint zero_mlp", pinnCheckpoint (ode / "zero_mlp.json")),
    ("PINN checkpoint one_mlp", pinnCheckpoint (ode / "one_mlp.json")),
    ("PINN checkpoint zero_siren", pinnCheckpoint (ode / "zero_siren.json")),
    ("PINN checkpoint tanh", pinnCheckpoint tanhPath),
    ("PINN checkpoint ReLU", pinnCheckpoint reluPath),
    ("ODE direct zero_mlp", odeCheckpoint .direct (ode / "zero_mlp.json")),
    ("ODE direct one_mlp", odeCheckpoint .direct (ode / "one_mlp.json")),
    ("ODE direct zero_siren", odeCheckpoint .direct (ode / "zero_siren.json")),
    ("ODE direct tanh", odeCheckpoint .direct tanhPath),
    ("ODE torchlean zero_mlp", odeCheckpoint .torchlean (ode / "zero_mlp.json")),
    ("ODE torchlean one_mlp", odeCheckpoint .torchlean (ode / "one_mlp.json")),
    ("ODE torchlean tanh", odeCheckpoint .torchlean tanhPath) ]

/-- Write the tanh and ReLU checkpoints under `.lake/build/tmp/builder-import`, build every
case of `cases` and print its node count, and fail when a case is refused. -/
def run : IO Unit := do
  let dir : System.FilePath := ".lake/build/tmp/builder-import"
  IO.FS.createDirAll dir
  let tanhPath := dir / "tanh_mlp.json"
  IO.FS.writeFile tanhPath tanhCheckpoint
  let reluPath := dir / "relu_mlp.json"
  IO.FS.writeFile reluPath reluCheckpoint
  let mut refused : Array String := #[]
  for (name, size) in cases tanhPath reluPath do
    match ← (try size catch e => pure (.error (toString e))) with
    | .ok n => IO.println s!"  {name}: {n} operator nodes"
    | .error e => refused := refused.push s!"{name}: {e}"
  unless refused.isEmpty do
    throw <| IO.userError s!"builder graphs refused: {refused}"

end TorchLean.Test.BuilderImport