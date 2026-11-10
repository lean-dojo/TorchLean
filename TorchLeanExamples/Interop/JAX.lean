/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.CLI.Parser
public import TorchLean.NN.IR.Builder
public import TorchLean.Tensor.Op.Table.Elementwise
public import TorchLean.Tensor.Op.Table.Reduce
public import TorchLean.Tensor.LinearAlgebra.Real
public import TorchLean.NN.JAX
public import TorchLean.Tensor.Context.Real
public import TorchLean.Tensor.Reduce.Fold
public meta import TorchLean.Tensor.Einops.Elab.Einsum
public meta import TorchLean.Tensor.Einops.Elab.Pack
public meta import TorchLean.Tensor.Literal
public meta import TorchLean.Tensor.Einops.Elab.Transform
public meta import TorchLean.Tensor.Tactic.Einops.Report

/-!
# From Jaxpr to a permutation-invariance proof

The Python companion captures `sum(x * x)` and several other tensor computations as capture
documents. `run` loads each with `TorchLean.JAX.load`, runs its program on the recorded arguments
under the reference runtime and compares the interchange words of every output with the recorded JAX
output. For `sum(x * x)`, a graph built in Lean comes with a proof over the reals that it gives the
same value on a vector and on every permutation of its entries, at every vector length. `check`
reports each imported graph that equals it.

The proof uses exact real addition, and a floating-point sum can round differently when its terms
are reordered. Neither the proof nor the numerical checks certify the Python adapter or XLA.
-/

@[expose] public section

namespace TorchLean.Examples.Jaxpr

open scoped BigOperators

/-- The registered-operator graph of `jax.numpy.sum(x * x)`: graph input `0` at `shape`, the
elementwise product of the input with itself, and its sum over every axis without `keepdim`. -/
def sumSquaresGraph (shape : Shape) : Except String IR.Graph :=
  let (b, x) := ({} : IR.Graph.Builder).input 0 (.value shape)
  let (b, p) := b.op (Op.mul (s := shape)) [x, x]
  let (b, y) := b.op (Op.reduce (s := shape) .sum (List.finRange shape.rank) false
    (fun h => absurd h (by decide))) p
  b.build y.toArray

/-- The real value of the captured computation: the sum of the squared entries of `x`. -/
noncomputable def sumSquares {shape : Shape} (x : Tensor ℝ shape) : ℝ :=
  Tensor.sum (x * x)

/-- Reordering vector entries leaves the exact real sum of their squares unchanged. -/
theorem sumSquares_permutation {n : Nat} (permutation : Equiv.Perm (Fin n))
    (x y : Tensor ℝ [n]) (h : ∀ i, y[i] = x[permutation i]) :
    sumSquares y = sumSquares x := by
  unfold sumSquares
  rw [Tensor.sum_vec, Tensor.sum_vec]
  have hSquare (z : Tensor ℝ [n]) (i : Fin n) :
      Tensor.getScalar (z * z) i = z[i] * z[i] := Tensor.getScalar_mul z z i
  simp_rw [hSquare]
  calc
    (∑ i : Fin n, y[i] * y[i]) = ∑ i : Fin n, x[permutation i] * x[permutation i] := by
      apply Finset.sum_congr rfl
      intro i _
      rw [h i]
    _ = ∑ i, x[i] * x[i] := Equiv.sum_comp permutation (fun i : Fin n => x[i] * x[i])

/-- `sumSquaresGraph` passes the structural check and builds three nodes: the input, its product
with itself and the sum of that product over every axis, which is the graph's only output. -/
theorem sumSquaresGraph_eq (shape : Shape) :
    sumSquaresGraph shape = .ok ⟨#[.input 0 (.value shape),
      .op ⟨_, _, Op.mul (s := shape)⟩ #[⟨0, 0⟩, ⟨0, 0⟩],
      .op ⟨_, _, Op.reduce (s := shape) .sum (List.finRange shape.rank) false
        (fun h => absurd h (by decide))⟩ #[⟨1, 0⟩]], #[⟨2, 0⟩]⟩ := by
  refine IR.Graph.Builder.build_ok rfl ⟨fun i hi => ?_, fun r hr => ?_⟩
  · rcases i with _ | _ | _ | i
    · trivial
    · exact ⟨.cons ⟨_, ⟨Nat.zero_lt_one, _, rfl, rfl⟩, Port.accepts_self _⟩
        (.cons ⟨_, ⟨Nat.zero_lt_one, _, rfl, rfl⟩, Port.accepts_self _⟩ .nil), rfl⟩
    · exact ⟨.cons ⟨_, ⟨Nat.one_lt_two, _, rfl, rfl⟩, Port.accepts_self _⟩ .nil,
        Op.reduce_check shape⟩
    · exact absurd hi (Nat.not_lt.2 (Nat.le_add_left 3 i))
  · obtain rfl : r = ⟨2, 0⟩ := by simpa using hr
    exact ⟨_, Nat.lt_succ_self 2, _, rfl, rfl⟩

/-- At every shape, the built graph, evaluated over the reals with the empty payload and input
`x`, returns the single scalar `sumSquares x`. -/
theorem sumSquaresGraph_eval {shape : Shape} (x : Tensor ℝ shape) :
    (sumSquaresGraph shape >>= fun g => g.eval {} [⟨.value shape, x⟩]) =
      .ok [⟨.value .scalar, Tensor.scalar (sumSquares x)⟩] := by
  have h0 : (IR.Node.input 0 (.value shape)).eval ({} : IR.Payload ℝ)
      [⟨.value shape, x⟩] #[] 0 = .ok [⟨.value shape, x⟩] := by
    simp only [IR.Node.eval, List.getElem?_cons_zero, Port.read_self]
    rfl
  obtain ⟨E1, h1, s1, r1, -⟩ := IR.runNodes_cons_of_eval h0 _
  have r1 : (⟨0, 0⟩ : IR.Ref).read E1 = .ok ⟨.value shape, x⟩ := r1
  obtain ⟨E2, h2, s2, r2, -⟩ := IR.runNodes_cons_of_eval
    (IR.Node.eval_binary _ rfl rfl Op.rule?_mul r1 r1) _
  have r2 : (⟨1, 0⟩ : IR.Ref).read E2 = .ok ⟨.value shape, x * x⟩ := by
    rw [s1] at r2
    exact r2
  obtain ⟨E3, h3, -, r3, -⟩ := IR.runNodes_cons_of_eval
    (IR.Node.eval_unary _ (Op.reduce_check shape) rfl (Op.rule?_reduce _ _ _ _) r2) _
  have r3 : (⟨2, 0⟩ : IR.Ref).read E3 = .ok ⟨.value .scalar, Tensor.scalar (sumSquares x)⟩ := by
    rw [s2, s1] at r3
    exact r3.trans (congrArg Except.ok (Op.value_eq_scalar _ (Shape.reduce_finRange shape)))
  refine IR.bind_eq_ok.2 ⟨_, sumSquaresGraph_eq shape,
    IR.bind_eq_ok.2 ⟨E3, ?_, IR.mapM_eq_ok_iff.2 (.cons r3 .nil)⟩⟩
  show IR.runNodes _ _ _ [_, _, _] = _
  rw [h1, h2, h3, IR.runNodes_nil]

/-- Evaluating the built graph at the reals on a vector and on any permutation of its entries
gives the same result. -/
theorem sumSquaresGraph_permutation {n : Nat} (permutation : Equiv.Perm (Fin n))
    (x y : Tensor ℝ [n]) (h : ∀ i, y[i] = x[permutation i]) :
    (sumSquaresGraph [n] >>= fun g => g.eval {} [⟨.value [n], y⟩]) =
      (sumSquaresGraph [n] >>= fun g => g.eval {} [⟨.value [n], x⟩]) := by
  rw [sumSquaresGraph_eval, sumSquaresGraph_eval, sumSquares_permutation permutation x y h]

/-- Load a Jaxpr capture document, run its program on its arguments under the reference runtime and
compare the element type, shape and interchange words of every output with the recorded JAX output.
Also report whether the program is the vector graph that `sumSquaresGraph_permutation` covers.

This checks the recorded arguments, not all inputs or XLA's reduction schedule. The companion
chooses arguments whose reductions are exact at their precision.
-/
def check (document : Lean.Json) : IO Unit := do
  let (program, payload) ← IO.ofExcept (JAX.load document)
  let arguments ← IO.ofExcept (IR.loadArguments .reference document)
  let (outputs, flags) ← IO.ofExcept (program.eval .reference payload arguments)
  let outputs := outputs.toArray
  let expected ← IO.ofExcept
    ((← IO.ofExcept (document.getObjVal? "reference_outputs")).getArr?)
  unless outputs.size == expected.size do
    throw (IO.userError "Jaxpr example: output count mismatch")
  for i in [:outputs.size] do
    let some output := outputs[i]?
      | throw (IO.userError "Jaxpr example: missing output")
    let some referenceJson := expected[i]?
      | throw (IO.userError "Jaxpr example: missing reference output")
    let (dtype, shape, actual) ← IO.ofExcept (IR.words output)
    let (dtype', shape', reference) ← IO.ofExcept
      (IR.loadValue .reference referenceJson >>= IR.words)
    unless dtype == dtype' && shape == shape' do
      throw (IO.userError
        s!"Jaxpr example: result {i} is {dtype} at {repr shape}, JAX's {dtype'} at {repr shape'}")
    unless actual == reference do
      throw (IO.userError s!"Jaxpr example: result {i} differs in its interchange words")
    IO.println s!"  output {i}: {dtype} at shape {repr shape}, bit-for-bit match for this input"
  IO.println s!"  conversion flags: {repr flags}"
  -- Matching the complete graph connects this document to the general theorem above.
  if let some (graph, _) := program.graph? then
    let input := graph.nodes.findSome? fun
      | .input _ ⟨.value, s⟩ => some s
      | _ => none
    if let some shape := input then
      if (sumSquaresGraph shape).toOption == some graph && shape.rank == 1 then
        IO.println "  matches the graph covered by sumSquaresGraph_permutation"
  IO.println s!"  checked {program.nodes.size} typed IR steps"

/-- Check one or more capture documents produced by the Python companion. With `--help` or `-h`,
print the usage line instead. -/
def run (paths : List String) : IO Unit := do
  let usage := "usage: torchlean jaxpr <document.json> ..."
  if CLI.hasHelp paths then
    IO.println usage
    return
  if paths.isEmpty then
    throw (IO.userError usage)
  for path in paths do
    IO.println path
    let document ← Json.readFile path
    check document

end TorchLean.Examples.Jaxpr
