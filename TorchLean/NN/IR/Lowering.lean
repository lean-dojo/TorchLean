/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.IR.Imported
public import TorchLean.NN.IR.Record

/-!
# Lowering to the verifier IR

`lowerForward` runs a `TorchLean.Program` with the recording interpreter `RecordM`, then builds the
graph of registered operators of the recorded typed graph and checks its structure
(`TorchLean.IR.Graph.check`). A successful result is executable and can be handed to the verifier
passes on graphs of registered operators (`TorchLean.CROWN.Op`), which may refuse transfers they do
not support. Its graph returns the recorded graph's typed value at the output on every input
(`lowerForward_eq_ok`), and no theorem relates it to the source program.

- `LoweredModel`: the graph of registered operators of a recorded typed graph as an import
  (`TorchLean.IR.Imported`: the graph, its payload and the reference of every recorded value), with
  the references of the graph input and of the output.
-/

@[expose] public section


namespace TorchLean.IR

/-- The lowering of a TorchLean forward model to the verifier IR: an import whose graph's one
output is the lowered output, with the references of the graph input and of the output. -/
structure LoweredModel (α : Type) [Storage α] extends IR.Imported α where
  /-- The graph input. -/
  input : IR.Ref
  /-- The output. -/
  output : IR.Ref

open TorchLean.Tensor

/--
Lower a TorchLean forward model with one distinguished input, supplied as its last argument: the
graph of registered operators of the typed graph that `recordForward` records, after the structural
check `IR.Graph.check`.

Success means that every encountered operation was supported and that the graph passed the check.
The lowered graph returns the recorded typed graph's value at the output (`lowerForward_eq_ok`),
and no theorem relates it to the source program.
-/
@[noinline, nospecialize]
def lowerForward
    {α : Type} [Storage α] [Context α]
    {paramShapes : List Shape} {inShape outShape : Shape}
    (model : Program α (paramShapes ++ [inShape]) outShape)
    (params : TensorPack α paramShapes) :
    Except String (LoweredModel α) :=
  match recordForward model params with
  | .error e => .error e
  | .ok ⟨st, out⟩ =>
      let r := st.graph.toOp (fun _ t => t) #[out.1]
      match r.graph.check with
      | .error e => .error s!"TorchLean→IR: produced an ill-formed graph: {e}"
      | .ok () =>
          .ok { graph := r.graph, payload := r.payload, ids := r.ids, input := ⟨0, 0⟩
                output := r.ids[out.1.val]?.getD ⟨0, 0⟩ }

/-- A successful lowering is the graph of registered operators of the graph that `recordForward`
records, and it returns that graph's typed value at the output, for every input. -/
theorem lowerForward_eq_ok
    {α : Type} [Storage α] [Context α] [Op.Capabilities α]
    {paramShapes : List Shape} {inShape outShape : Shape}
    {model : Program α (paramShapes ++ [inShape]) outShape}
    {params : TensorPack α paramShapes} {lowered : LoweredModel α}
    (h : lowerForward model params = .ok lowered) :
    ∃ r, recordForward model params = .ok r ∧
      lowered.graph = (r.1.graph.toOp (fun _ t => t) #[r.2.1]).graph ∧
      lowered.payload = (r.1.graph.toOp (fun _ t => t) #[r.2.1]).payload ∧
      ∀ x : Tensor α inShape, lowered.graph.eval lowered.payload [⟨.value inShape, x⟩] =
        .ok [⟨r.1.ports.get r.2.1, Values.get (r.1.graph.eval (fun _ t => t) x) r.2.1⟩] := by
  unfold lowerForward at h
  split at h
  · cases h
  · rename_i st out hr
    dsimp only at h
    split at h
    · cases h
    · injection h with h
      subst h
      exact ⟨⟨st, out⟩, hr, rfl, rfl, fun x => OpGraph.eval_toOp _ st.graph #[out.1] x⟩

end TorchLean.IR
