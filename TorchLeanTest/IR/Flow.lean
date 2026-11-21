/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.IR.Flow.Reals
public import FloatLib.Floats.Formats.BinaryInterchange.Analysis.StandardModel
public import TorchLean.Verify.Capture

/-!
# Branches and loops of IR graphs

Flows of `IR.Flow.graphs Float` on signed 32-bit integers, checked by `run` in the test suite.
`gcd` is Euclid's algorithm: a `whileLoop` whose test graph computes `b ≠ 0` and whose body graph
maps `[a, b]` to `[b, a % b]`. Its trip count depends on the operands, so it runs only as a loop.
On `[12, 18]` its test holds before 3 iterations and not after them, the premises of
`Flow.eval_whileLoop` at `k = 3`, and it returns `[6, 0]` with fuel 3 or more and fails with fuel 2.
A `scan` of the body runs it a fixed number of times. A `cond` whose unselected branch reads a
missing input returns the selected branch's state, where evaluating every branch first
(`Flow.condAll`) fails; an index past the end or negative selects the last branch, and a Boolean
index selects branch 0 or 1.

The same graphs as the blocks of typed programs (`Flow.programs`) make capture documents of format
version `Flow.formatVersion`, which `Capture.check` decodes, loads and runs: `gcd` with fuel 64 and
with fuel 2, a `scan` and a `cond`. A declared type names the block that holds its value, and one
naming a block the flow does not have fails the load. The `types` check fails a flow whose blocks
do not compose (`Capture.flowTypes`): the test followed by the body, which takes two values, a
loop whose test returns two values, a loop whose body returns one value to its test, and the body
after a `cond` one of whose branches is the test. It passes a `cond` one of whose branches skips,
followed by `gcd`.

`square` is a flow of `Flow.Reals` blocks: a `whileLoop` that squares a real while a counter,
decreased by two each iteration, is positive. From `[θ, 3]` it runs twice. `square_hasDerivAt`
proves, for every real `θ₀`, that near `θ₀` it returns a real `f θ` with `f θ₀ = θ₀ ^ 4` and
derivative `4 θ₀ ^ 3` at `θ₀`, from the dual's run on `[(θ₀, 1), (3, 0)]` by `Follows.whileLoop` and
`Reals.dual_follows`. `square_rounded` proves, for every format `fmt` and real `θ`, that its run
with `roundAt fmt` rounding and the exact rounding errors as bounds returns `y`, the rounded square
of the rounded square `x` of `θ`, with bound `2 |x| |x - θ²| + |x - θ²|² + |y - x²|`, and that the
exact run returns `θ ^ 4` within that bound of `y`, by `Follows.whileLoop` and
`Reals.rounded_follows`. Each is an instance of their premises.
-/

@[expose] public section

namespace TorchLean.Test.Flow

open IR IR.Flow

/-- The port of a signed 32-bit integer scalar. -/
abbrev int32 : Port := .integer true 32 .scalar

/-- The 32-bit word of `n`. -/
def word (n : Int) : Op.Value Float := ⟨int32, Tensor.scalar (BitVec.ofInt 32 n)⟩

/-- The integers a run returns, or `none` where it fails. -/
def ints (r : Except String (List (Op.Value Float))) : Option (Array Int) :=
  (r.bind (·.toArray.mapM toIndex)).toOption

/-- `[a, b] ↦ [b, a % b]`, with Python's `%`. -/
def euclid : Graph where
  nodes := #[.input 0 int32, .input 1 int32,
    .op ⟨_, _, Op.remainder (s := .scalar) (.integer true 32)⟩ #[⟨0, 0⟩, ⟨1, 0⟩]]
  outputs := #[⟨1, 0⟩, ⟨2, 0⟩]

/-- `[a, b] ↦ b ≠ 0`, the zero a constant. -/
def nonzero : Graph where
  nodes := #[.input 0 int32, .input 1 int32, .const int32,
    .op ⟨_, _, Op.ne (s := .scalar) (.integer true 32)⟩ #[⟨1, 0⟩, ⟨2, 0⟩]]
  outputs := #[⟨3, 0⟩]

/-- The payload of `nonzero`: the zero at node 2. -/
def zero : Payload Float where
  const? k := if k = 2 then some (word 0) else none

/-- A graph that reads input 5, which a state of two values does not have. -/
def missing : Graph where
  nodes := #[.input 5 int32]
  outputs := #[⟨0, 0⟩]

/-- The body as a flow. -/
def step : Flow (Graph × Payload Float) := .block (euclid, {})

/-- The test as a flow. -/
def test : Flow (Graph × Payload Float) := .block (nonzero, zero)

/-- Euclid's algorithm. -/
def gcd : Flow (Graph × Payload Float) := .whileLoop test step

/-- A branch on two of `missing` and the body. -/
def branches : List (Flow (Graph × Payload Float)) := [.block (missing, {}), step]

/-- The checks, by name. -/
def checks : List (String × Bool) :=
  let L := graphs Float
  let s := [word 12, word 18]
  [ ("the graphs pass the check", euclid.check matches .ok () && nonzero.check matches .ok ()),
    ("gcd 12 18 with fuel 3", ints (gcd.eval L 3 s) == some #[6, 0]),
    ("gcd 12 18 with fuel 64", ints (gcd.eval L 64 s) == some #[6, 0]),
    ("gcd -4 6", ints (gcd.eval L 64 [word (-4), word 6]) == some #[2, 0]),
    ("gcd 12 18 with fuel 2 fails", gcd.eval L 2 s matches .error "while: out of fuel"),
    -- The premises of `eval_whileLoop` at `k = 3`, and the unrolled loop.
    ("the test holds before 3 iterations", (List.range 3).all fun j =>
      ((step.unroll j).eval L 3 s >>= holds L 3 test) matches .ok true),
    ("the test fails after 3 iterations",
      ((step.unroll 3).eval L 3 s >>= holds L 3 test) matches .ok false),
    ("the unrolled gcd", ints ((step.unroll 3).eval L 3 s) == some #[6, 0]),
    ("a scan of 2", ints ((Flow.scan 2 step).eval L 0 s) == some #[12, 6]),
    ("cond at 1", ints ((Flow.cond branches).eval L 0 (word 1 :: s)) == some #[18, 12]),
    ("cond past the end", ints ((Flow.cond branches).eval L 0 (word 7 :: s)) == some #[18, 12]),
    ("cond at -1", ints ((Flow.cond branches).eval L 0 (word (-1) :: s)) == some #[18, 12]),
    ("cond at true", ints ((Flow.cond branches).eval L 0
      (⟨.bool .scalar, Tensor.scalar true⟩ :: s)) == some #[18, 12]),
    ("cond at 0 fails", ints ((Flow.cond branches).eval L 0 (word 0 :: s)) == none),
    ("every branch first fails", ints (condAll L 0 branches (word 1 :: s)) == none),
    -- Where every branch is defined, `cond` and `condAll` agree.
    ("cond and condAll at 0", ints ((Flow.cond [step, .skip]).eval L 0 (word 0 :: s)) ==
      ints (condAll L 0 [step, .skip] (word 0 :: s))),
    ("cond at 1 skips", ints ((Flow.cond [step, .skip]).eval L 0 (word 1 :: s)) == some #[12, 18])]

/-! ## Typed programs in capture documents -/

open Lean

/-- The artifact of a graph whose ports are integer or Boolean, with the words of its constants. -/
def artifactOf (g : Graph) (constants : Array (Option (Array Nat))) : Artifact where
  program :=
    { nodes := g.nodes.map .node, outputs := g.outputs, scalars := g.nodes.map fun _ => none }
  constants

/-- The body as a block of a flow artifact. -/
def stepBlock : Flow Artifact := .block (artifactOf euclid #[none, none, none])

/-- The test as a block of a flow artifact. -/
def testBlock : Flow Artifact := .block (artifactOf nonzero #[none, none, some #[0], none])

/-- Euclid's algorithm as a flow artifact. -/
def gcdBlocks : Flow Artifact := .whileLoop testBlock stepBlock

/-- The raw IR document of `flow` with the fuel `fuel`, on signed 32-bit scalars, with the
`declared_types` `declared` where given. -/
def document (flow : Flow Artifact) (fuel : Nat) (arguments : Array Int)
    (declared : Option Json := none) : Json :=
  Json.mkObj ([("artifact", ({ flow, fuel } : FlowArtifact).toJson),
    ("arguments", Json.arr (arguments.map fun n => Json.mkObj [
      ("dtype", toJson (DType.integer true (some 32))), ("shape", toJson ([] : List Nat)),
      ("bits", toJson [(BitVec.ofInt 32 n).toNat])]))] ++
    (declared.map fun d => [("declared_types", d)]).getD [])

/-- A declared signed 32-bit scalar at the output `ref` of block `block`. -/
def declared (block : Nat) (ref : Ref) : Json :=
  Json.arr #[Json.mkObj [("node", toJson 0), ("block", toJson block), ("ref", ref.toJson),
    ("origin", toJson "remainder"), ("source", toJson "capture"),
    ("dtype", toJson (DType.integer true (some 32))), ("shape", toJson ([] : List Nat))]]

/-- The report of the structure, type and execution checks. -/
def report (document : Json) : Json :=
  Capture.check document #["structure", "types", "execute"]

/-- The words of each output of a passing report, or `none` where a check failed. -/
def words (document : Json) : Option Json :=
  let r := report document
  if r.getObjValD "ok" == .bool true then
    some (Json.arr (((r.getObjValD "outputs").getArr?.toOption.getD #[]).map (·.getObjValD "bits")))
  else none

/-- The message of the check `name` of a report. -/
def message (document : Json) (name : String) : String :=
  let checks := ((report document).getObjValD "checks").getArr?.toOption.getD #[]
  ((checks.find? (·.getObjValD "name" == toJson name)).map fun c =>
    (c.getObjValD "message").getStr?.toOption.getD "").getD ""

/-- The checks of flow artifacts, by name. -/
def documentChecks : List (String × Bool) :=
  let artifact := (document gcdBlocks 64 #[12, 18]).getObjValD "artifact"
  [ ("the flow artifact round trip", match FlowArtifact.ofJson artifact with
      | .ok a => a.toJson == artifact
      | .error _ => false),
    ("gcd 12 18 as typed programs",
      words (document gcdBlocks 64 #[12, 18]) == some (toJson [[6], [0]])),
    ("gcd 12 18 with fuel 2 fails",
      message (document gcdBlocks 2 #[12, 18]) "execute" == "while: out of fuel"),
    ("a scan of 2", words (document (.scan 2 stepBlock) 0 #[12, 18]) == some (toJson [[12], [6]])),
    ("cond at 0", words (document (.cond [stepBlock, .skip]) 0 #[0, 12, 18]) ==
      some (toJson [[18], [12]])),
    ("cond at 1 skips", words (document (.cond [stepBlock, .skip]) 0 #[1, 12, 18]) ==
      some (toJson [[12], [18]])),
    ("a declared type in block 1",
      words (document gcdBlocks 64 #[12, 18] (declared 1 ⟨2, 0⟩)) == some (toJson [[6], [0]])),
    ("a declared type in no block", message (document gcdBlocks 64 #[12, 18] (declared 5 ⟨2, 0⟩))
      "types" == "IR types: node 0 (remainder): no block 5"),
    -- The body given the test's one result.
    ("blocks that do not compose", message (document (.seq testBlock stepBlock) 0 #[12, 18])
      "types" == "IR types: block 1 takes 2 values, given 1"),
    ("a test of two values", message (document (.whileLoop stepBlock stepBlock) 64 #[12, 18])
      "types" == "IR types: the test of the while loop before block 0 returns 2 values"),
    -- The test given the body's one result, on the second iteration.
    ("a body its test cannot follow", message (document (.whileLoop testBlock testBlock) 64
      #[12, 18]) "types" == "IR types: block 0 takes 2 values, given 1"),
    -- The body given the result of the branch that is the test, which this run takes.
    ("a branch the next block cannot follow", message
      (document (.seq (.cond [stepBlock, testBlock]) stepBlock) 0 #[1, 12, 18]) "types" ==
        "IR types: block 2 takes 2 values, given 1"),
    ("gcd after a cond that skips at 1",
      words (document (.seq (.cond [stepBlock, .skip]) gcdBlocks) 64 #[1, 12, 18]) ==
        some (toJson [[6], [0]]))]

/-- Fail at the first check that does not hold. -/
def run : IO Unit := do
  for (name, ok) in checks ++ documentChecks do
    unless ok do
      throw <| IO.userError s!"IR flow: {name}"
  IO.println s!"IR flows: {checks.length + documentChecks.length} checks passed"

section Reals

open IR.Flow.Reals Topology FloatLib.Floats.Formats.BinaryInterchange Model

/-- Square the real at position 0 while the real at position 1 is positive, decreasing it by two
each iteration. -/
def square : Flow (List Out) :=
  .whileLoop (.block [.lt (.const 0) (.var 1)])
    (.block [.real (.mul (.var 0) (.var 0)), .real (.add (.var 1) (.const (-2)))])

/-- Near every `θ₀`, `square` on `[θ, 3]` returns a real `f θ` with value `θ₀ ^ 4` and derivative
`4 θ₀ ^ 3` at `θ₀`. -/
theorem square_hasDerivAt (θ₀ : ℝ) :
    ∃ f g : ℝ → ℝ, (∀ᶠ θ in 𝓝 θ₀,
      Flow.eval lang 2 square [.inl θ, .inl 3] = .ok [.inl (f θ), .inl (g θ)]) ∧
      f θ₀ = θ₀ ^ 4 ∧ HasDerivAt f (4 * θ₀ ^ 3) θ₀ := by
  have hF : Tracks (𝓝 θ₀) (Near θ₀) (fun θ : ℝ => ([.inl θ, .inl 3] : List (ℝ ⊕ Bool)))
      [(.inl θ₀, ContinuousLinearMap.id ℝ ℝ), (.inl 3, 0)] :=
    tracks_map (vs := [fun θ => (.inl θ : ℝ ⊕ Bool), fun _ => .inl 3])
      (.cons ⟨id, .of_forall fun _ => rfl, rfl, hasFDerivAt_id θ₀⟩
        (.cons ⟨fun _ => 3, .of_forall fun _ => rfl, rfl, hasFDerivAt_const _ _⟩ .nil))
  have he : Flow.eval (dual ℝ) 2 square [(.inl θ₀, ContinuousLinearMap.id ℝ ℝ), (.inl 3, 0)] =
      .ok [(.inl (θ₀ ^ 4), (4 * θ₀ ^ 3) • ContinuousLinearMap.id ℝ ℝ), (.inl (-1), 0)] := by
    norm_num [Flow.eval, square, runWhile, holdsOf, dual, Lang.dual, Out.dual, Term.dual, lang,
      Bind.bind, Except.bind, Pure.pure, Except.pure]
    refine ⟨by ring, ?_⟩
    ext
    simp
    ring
  obtain ⟨_, _, _, G, hG, vs, hvs, hD⟩ := (dual_follows θ₀).whileLoop hF he
  rcases hD with _ | ⟨⟨f, hf, hx, hd⟩, _ | ⟨⟨g, hg, -, -⟩, _ | _⟩⟩
  refine ⟨f, g, ((hG.and hvs).and (hf.and hg)).mono fun θ ⟨⟨e₁, e₂⟩, e₃, e₄⟩ => ?_, hx, ?_⟩
  · exact e₁.1.trans (by rw [e₂]; simp [e₃, e₄])
  · simpa using hd.hasDerivAt

/-- In every format `fmt`, `square` on `[θ, 3]` with each constant, sum and product rounded by
`roundAt fmt` and its exact rounding error as its bound returns the rounded square `y` of the
rounded square `x` of `θ` with bound `b`, and the exact run returns `θ ^ 4`, within `b` of `y`. -/
theorem square_rounded (fmt : FloatFormat) (θ : ℝ) :
    let x := roundAt fmt (θ * θ)
    let y := roundAt fmt (x * x)
    let b := 2 * |x| * |x - θ * θ| + |x - θ * θ| ^ 2 + |y - x * x|
    Flow.eval (rounded (roundAt fmt ·) fun z => |roundAt fmt z - z|) 2 square
        [(.inl θ, 0), (.inl 3, 0)] = .ok [(.inl y, b), (.inl (-1), 0)] ∧
      Flow.eval lang 2 square [.inl θ, .inl 3] = .ok [.inl (θ ^ 4), .inl (-1)] ∧
      |θ ^ 4 - y| ≤ b := by
  intro x y b
  have h₂ : roundAt fmt (-2) = -2 := by
    simpa using roundAt_intCast_of_abs_le fmt (n := -2) (by
      simpa using pow_le_pow_right₀ (by norm_num : (1 : ℝ) ≤ 2) (Nat.le_add_left 1 fmt.fracWidth))
  have hF : Tracks ⊤ Within (fun _ : Unit => ([.inl θ, .inl 3] : List (ℝ ⊕ Bool)))
      [(.inl θ, 0), (.inl 3, 0)] :=
    tracks_map (vs := [fun _ => (.inl θ : ℝ ⊕ Bool), fun _ => .inl 3])
      (.cons ⟨fun _ => θ, fun _ => rfl, fun _ => by simp⟩
        (.cons ⟨fun _ => 3, fun _ => rfl, fun _ => by simp⟩ .nil))
  have he : Flow.eval (rounded (roundAt fmt ·) fun z => |roundAt fmt z - z|) 2 square
      [(.inl θ, 0), (.inl 3, 0)] = .ok [(.inl y, b), (.inl (-1), 0)] := by
    norm_num [Flow.eval, square, runWhile, holdsOf, rounded, Lang.dual, Out.round, Term.round,
      lang, h₂, Bind.bind, Except.bind, Pure.pure, Except.pure, x, y, b]
    ring
  have hx : Flow.eval lang 2 square [.inl θ, .inl 3] = .ok [.inl (θ ^ 4), .inl (-1)] := by
    norm_num [Flow.eval, square, runWhile, holdsOf, lang, Out.eval, Term.eval, Bind.bind,
      Except.bind, Pure.pure, Except.pure]
    ring
  obtain ⟨_, _, _, G, hG, vs, hvs, hD⟩ :=
    (rounded_follows Unit fun z => le_refl |roundAt fmt z - z|).whileLoop hF he
  rcases hD with _ | ⟨⟨f, hf, hb⟩, _ | ⟨⟨g, hg, -⟩, _ | _⟩⟩
  have e : Flow.eval lang 2 square [.inl θ, .inl 3] = .ok (G ()) :=
    (Filter.eventually_top.1 hG ()).1
  rw [hx, Filter.eventually_top.1 hvs ()] at e
  simp only [List.map, hf (), hg (), Except.ok.injEq, List.cons.injEq, Sum.inl.injEq] at e
  refine ⟨he, hx, ?_⟩
  rw [e.1]
  exact hb ()

end Reals

end TorchLean.Test.Flow
