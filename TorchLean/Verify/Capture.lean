/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.IR.Loaded

/-!
# Checking captured tensor computations

The verification entry point checks a typed IR artifact and, optionally, executes supplied
arguments or compares supplied reference outputs. The result states the scope of each check: a
passing check covers the supplied arguments only, and the checks neither prove the frontend
translation nor validate external kernels.

A capture document holds its typed artifact under the key `artifact`, and every check reads it
through `TorchLean.IR.load`: `TorchLean.IR.Artifact.ofJson` decodes the artifact and
`TorchLean.IR.Artifact.load` checks its program (`TorchLean.IR.Program.check`) and its payload
(`TorchLean.IR.Program.Payload.check`). The `structure` and `types` findings report that check. The
arguments and reference outputs are tensors that `TorchLean.IR.loadValue` reads.

The report preserves the document's `capture_constraints` as unchecked metadata. Structure,
type and example-execution checks do not establish these conditions for all inputs.

`compare` treats two NaN results as equal whatever their sign and payload, which the IR leaves
unspecified. `compare_bitwise` requires equal words for NaNs too. Any other difference between the
IR's results and the reference fails. For a Jaxpr document whose reference XLA's CPU backend
computed, the checker then runs the program a second time as XLA's compiled CPU code computes it,
where this is exactly emulable. The emulated behaviours are reciprocals of broadcast and constant
divisors, `jax.jit`'s division and power rewrites when the document records a `jax.jit` reference,
and the flushing of subnormal values. The emulation recognises the steps it rewrites by their
`TorchLean.Op` operators, and every step of the emulated program runs by
`TorchLean.IR.Program.step`. A value that XLA's own `rsqrt` or power function computes counts as
reproduced within the window of `Behaviour.window`, measured in the format the function computes in.
If the second run reproduces every differing element, the failed finding lists the behaviours in
`consistent_with`. That field is advice: an importer error with the same effect on these arguments
cannot be ruled out, and some importer errors coincide with XLA's own rewrites. Under `jax.jit`,
listing them also requires the document's primitive-by-primitive reference, when there is one, to
agree with the IR or to be reproduced without `jax.jit`'s rewrites. The checker trusts the
document's record of how XLA compiled the reference, so a forged record can at most add this advice
to a failure.

An unknown request is reported as unsupported and makes the report fail. Certificate checkers
and proof APIs are separate tools with their own contracts.
-/

@[expose] public section

open TorchLean.Shape (size)

namespace TorchLean

open Lean TorchLean.IR
open Program (Value Machine Typed)
open FloatLib.Floats.Formats.BinaryInterchange

namespace XLA

/-- One named check with its outcome and an explanation of its scope. -/
def _root_.TorchLean.Capture.finding (name status message : String)
    (extra : List (String × Json) := []) : Json :=
  Json.mkObj ([("name", toJson name), ("status", toJson status), ("message", toJson message)] ++
    extra)

/-- FloatLib's classification of one binary word: whether it is a NaN and whether it is
subnormal, under the format's exceptional-value encoding. -/
def _root_.TorchLean.Capture.binaryWordClass (format : FloatFormat) (word : Nat) : Bool × Bool :=
  let model : Model format := Model.ofNatBits word
  (Model.isNaN model, Model.isSubnormal model)

/-- Whether two words of `dtype` agree: equal, or both NaNs unless `bitwise` is set. -/
def _root_.TorchLean.Capture.agree (bitwise : Bool) (dtype : DType) (a b : Nat) : Bool :=
  a == b || !bitwise && match dtype with
    | .binary format => (Capture.binaryWordClass format a).1 && (Capture.binaryWordClass format b).1
    | _ => false

/-- The elements of one result that disagree with its reference. -/
structure _root_.TorchLean.Capture.Mismatch where
  /-- Positions of the elements that disagree under the requested NaN policy. -/
  elements : Array Nat := #[]
  /-- Elements where both words are NaNs with a different sign or payload. -/
  nanWords : Nat := 0
  /-- Elements of the result. -/
  total : Nat := 0

/-- Compare one result with its reference. Two NaNs agree unless `bitwise` is set. A result of
index tuples fails, since a reference tensor has an element type and holds no index tuples. -/
def _root_.TorchLean.Capture.compareResult (bitwise : Bool)
    (output expected : Value Machine.reference) :
    Except String Capture.Mismatch := do
  if let .index bounds := output.typed.port.kind then
    throw s!"index tuples below {bounds}, and a reference tensor holds no index tuples"
  unless output.typed.dtype? == expected.typed.dtype? &&
      output.typed.port.shape == expected.typed.port.shape do
    throw "a different element type or shape"
  let (dtype, _, actual) ← IR.words output
  let (_, _, wanted) ← IR.words expected
  let mut result : Capture.Mismatch := { total := actual.size }
  for index in [:actual.size] do
    let a := actual[index]!
    let b := wanted[index]!
    if a != b then
      if Capture.agree false dtype a b then
        result := { result with nanWords := result.nanWords + 1 }
      unless Capture.agree bitwise dtype a b do
        result := { result with elements := result.elements.push index }
  pure result

/-! ## XLA's documented CPU behaviour

The behaviours, how the document says XLA compiled the reference, and the emulated program that
`compare` runs for a differing Jaxpr result.
-/

/-- A documented way in which XLA's compiled CPU code computes differently from the IR, with the
code that a failed comparison lists in `consistent_with` and the description of
`TorchLean/NN/IR/README.md`. -/
structure Behaviour where
  /-- Stable code in the report. -/
  code : String
  /-- What XLA does differently. -/
  description : String
  deriving BEq

/-- A division by a broadcast divisor, or under `jax.jit` by a constant. -/
def Behaviour.divisorReciprocal : Behaviour :=
  ⟨"divisor_reciprocal", "division by a broadcast divisor, or under jax.jit by a constant, as \
    multiplication by the divisor's rounded reciprocal"⟩

/-- The simplifier's division rewrites under `jax.jit`. -/
def Behaviour.rewrittenDivision : Behaviour :=
  ⟨"rewritten_division", "jax.jit's rewrites of a division of a quotient, or by a quotient, \
    square root, reciprocal square root or power"⟩

/-- Constant exponents computed as products under `jax.jit`. -/
def Behaviour.powerProducts : Behaviour :=
  ⟨"pow_products", "jax.jit's products for the constant exponents 2 and 3 and reciprocal for -1"⟩

/-- Subnormal values flushed to zero. -/
def Behaviour.subnormal : Behaviour :=
  ⟨"subnormal", "subnormal operands and results flushed to zero in bfloat16, binary32 and \
    binary64"⟩

/-- XLA's own reciprocal square root. -/
def Behaviour.reciprocalSquareRoot : Behaviour :=
  ⟨"rsqrt", "XLA's own reciprocal square root"⟩

/-- XLA's own power function. -/
def Behaviour.power : Behaviour :=
  ⟨"pow", "XLA's own power function"⟩

/--
The window, in units in the last place of `dtype`, within which a value that XLA's own `rsqrt` or
power function computes in `dtype` counts as reproduced: the largest distance measured between
XLA's function and the IR's in that format, 2 for `rsqrt` and 1 for the power, in binary32 and
binary64. XLA's power is within 1 ulp of the IR's for integral exponents and where `|y log x| < 1`.
Elsewhere FloatLib's power is farther from it, and such a result is not reproduced. XLA computes
binary16 and bfloat16 `rsqrt` and powers in binary32 and rounds once, so the window applies to the
binary32 value before that rounding, and a node of a reduced format has none.
-/
def Behaviour.window (behaviour : Behaviour) (dtype : DType) : Option Nat :=
  if !(dtype == DType.float32 || dtype == DType.float64) then none
  else if behaviour == reciprocalSquareRoot then some 2
  else if behaviour == power then some 1
  else none

/-- How XLA compiled a Jaxpr reference, as the document records it. -/
structure Compilation where
  /-- The reference came from `jax.jit`, which compiles the whole program as one computation. -/
  jit : Bool := false
  /-- Otherwise, for each step, the computation XLA compiled it in: the JAX importer's index of
  the primitive, or of the nested `jax.jit` call, that produced it. -/
  computation : Array (Option Nat) := #[]
  /-- The computations that are nested `jax.jit` calls. -/
  jitComputations : Array Nat := #[]

namespace Compilation

/-- Read `reference_execution`, `jit_computations` and the computation of each step from the
document's array `computation`, or, with `perPrimitive`, describe the primitive-by-primitive
reference, where each primitive, inside nested `jax.jit` calls too, is a computation of its own,
identified by the step's entry of the array `primitive_index`. A missing array or entry, and an
entry that is not a natural number, read as no computation. -/
def read (document : Json) (perPrimitive : Bool := false) : Compilation :=
  let field := if perPrimitive then "primitive_index" else "computation"
  let jit := !perPrimitive && document.getObjValD "reference_execution" == .str "jit"
  let jitComputations := if perPrimitive then #[] else
    match document.getObjValD "jit_computations" with
    | .arr ids => ids.filterMap (·.getNat?.toOption)
    | _ => #[]
  let computation := match document.getObjValD field with
    | .arr entries => entries.map (·.getNat?.toOption)
    | _ => #[]
  { jit, computation, jitComputations }

/-- Whether step `id` is compiled by `jax.jit`. -/
def jitted (compilation : Compilation) (id : Nat) : Bool :=
  compilation.jit ||
    (((compilation.computation[id]?).join).map compilation.jitComputations.contains).getD false

/-- Whether `jax.jit` compiles any part of the program. -/
def usesJit (compilation : Compilation) : Bool :=
  compilation.jit || !compilation.jitComputations.isEmpty

end Compilation

/-- The outputs a step reads, in order: an operator's inputs, a conversion's source and a call's
arguments. An input and a constant read none. -/
def operands : Program.Node → Array IR.Ref
  | .node (.op _ refs) | .call _ refs _ _ => refs
  | .cast _ source _ => #[source]
  | .node _ => #[]

/-- The step that reads output `f r` wherever `node` reads an output `r`. -/
def remap (f : IR.Ref → IR.Ref) : Program.Node → Program.Node
  | .node (.op o refs) => .node (.op o (refs.map f))
  | .cast rounding source port => .cast rounding (f source) port
  | .call name args signature port => .call name (args.map f) signature port
  | node => node

/-- A remapped step reads the remapped outputs of the step, in the same order. -/
theorem operands_remap (f : IR.Ref → IR.Ref) (node : Program.Node) :
    operands (remap f node) = (operands node).map f := by
  rcases node with (_ | _ | _) | _ | _ <;> simp [operands, remap]

/-- The program as XLA's CPU code computes it, built step by step from the IR program. -/
structure Emulation where
  /-- The reference came from `jax.jit`. -/
  jit : Bool
  /-- Steps of the emulated program. -/
  nodes : Array Program.Node := #[]
  /-- Scalar of each step. -/
  scalars : Array (Option DType) := #[]
  /-- Values of the copied constants and of the ones that the rewrites add. -/
  constants : Array (Option (Value Machine.reference)) := #[]
  /-- The IR step each copied step comes from, for its payload entries. -/
  origin : Array (Option Nat) := #[]
  /-- The computation each step belongs to, as in `Compilation`. -/
  computation : Array (Option Nat) := #[]
  /-- Whether `jax.jit` folds each step at compile time: it is computed from constants alone
  within one `jax.jit` computation. -/
  folded : Array Bool := #[]
  /-- The documented behaviour that produced each step, with the IR step it rewrites. -/
  site : Array (Option (Behaviour × Nat)) := #[]

namespace Emulation

/-- Whether step `id` belongs to `computation`. Under `jax.jit` every step does. -/
def within (emulation : Emulation) (computation : Option Nat) (id : Nat) : Bool :=
  emulation.jit || match (emulation.computation[id]?).join, computation with
    | some a, some b => a == b
    | _, _ => false

/-- Whether step `id` is folded at compile time. -/
def isFolded (emulation : Emulation) (id : Nat) : Bool :=
  emulation.folded[id]?.getD false

/-- The element type of output `r`: that of its port at its step's scalar
(`Program.Typed.dtype?`), none for a missing output or an index port. Nothing is proved about
it here. -/
def dtypeAt (emulation : Emulation) (r : IR.Ref) : Option DType := do
  let port ← (emulation.nodes[r.node]?).bind (·.outs[r.out]?)
  (Typed.ofPort (emulation.scalars[r.node]?).join port).dtype?

/-- The step producing output `r` when it belongs to `computation`. -/
def operation? (emulation : Emulation) (computation : Option Nat) (r : IR.Ref) :
    Option Program.Node := do
  let node ← emulation.nodes[r.node]?
  guard (emulation.within computation r.node)
  pure node

/-- Whether the divisor `r` is a broadcast computed in `computation`, past reshapes, permutations
and detaches. -/
def broadcastIn (emulation : Emulation) (computation : Option Nat) (r : IR.Ref) : Bool :=
    Id.run do
  let mut current := r.node
  for _ in [:emulation.nodes.size] do
    let some node := emulation.nodes[current]? | return false
    unless emulation.within computation current do return false
    match node, (operands node)[0]? with
    | .node (.op ⟨_, _, .broadcast ..⟩ _), _ => return true
    | .node (.op ⟨_, _, .reshape ..⟩ _), some parent
    | .node (.op ⟨_, _, .permute ..⟩ _), some parent
    | .node (.op ⟨_, _, .detach⟩ _), some parent =>
        current := parent.node
    | _, _ => return false
  return false

end Emulation

/-- The IR step being emulated, its computation, and the documented behaviour that the steps
emitted for it emulate, if any. -/
structure Site where
  /-- The IR step. -/
  node : Nat
  /-- Its computation. -/
  computation : Option Nat
  /-- Whether `jax.jit` compiles it. -/
  jitted : Bool
  /-- The behaviour that the emitted steps emulate. -/
  difference : Option Behaviour := none

/-- Construction of the emulated program. -/
abbrev Build := StateT Emulation (Except String)

/-- Add the step `node` at the scalar `scalar` for `site` and return its first output. Under
`jax.jit` a constant is folded, and so is a step that reads only folded outputs of its computation
and draws no random numbers. -/
def emit (node : Program.Node) (scalar : Option DType) (site : Site)
    (constant : Option (Value Machine.reference) := none) (copy : Bool := false) :
    Build IR.Ref := do
  let emulation ← get
  let folded := match node with
    | .node (.const _) => site.jitted
    | .node (.input ..) | .node (.op ⟨_, _, .randUniform _⟩ _) | .node (.op ⟨_, _, .randn _⟩ _)
    | .node (.op ⟨_, _, .bernoulliMask _⟩ _) => false
    | _ => site.jitted && !(operands node).isEmpty && (operands node).all fun r =>
        emulation.isFolded r.node && emulation.within site.computation r.node
  let id := emulation.nodes.size
  set { emulation with
    nodes := emulation.nodes.push node
    scalars := emulation.scalars.push scalar
    constants := emulation.constants.push constant
    origin := emulation.origin.push (if copy then some site.node else none)
    computation := emulation.computation.push site.computation
    folded := emulation.folded.push folded
    site := emulation.site.push (site.difference.map (·, site.node)) }
  pure { node := id }

/-- The word of the integer `n` rounded once to `format`. -/
def integerWord (format : FloatFormat) (n : Int) : Nat :=
  FloatLib.Floats.ExecFloat.Binary.toNatBits (Reference.integerToBinary format .nearest n).1

/-- A constant of ones. -/
def ones (dtype : DType) (shape : Shape) (site : Site) : Build IR.Ref := do
  let .binary format := dtype | throw s!"XLA emulation: {dtype} has no binary one"
  let typed : Typed := ⟨.value shape, some dtype⟩
  let value ← typed.decode Machine.reference
    (Array.replicate (Shape.size shape) (integerWord format 1))
  emit (.node (.const (.value shape))) (some dtype) site (constant := some ⟨typed, value⟩)

/-- Round output `r` to `target`. -/
def castTo (r : IR.Ref) (target : DType) (site : Site) : Build IR.Ref := do
  let some port := ((← get).nodes[r.node]?).bind (·.outs[r.out]?)
    | throw s!"XLA emulation: missing node {r.node}"
  emit (.cast .nearest r (.value port.shape)) (some target) site

/-- Formats in which `jax.jit` rewrites quotients of quotients and quotients by quotients, each
step rounded to the format. XLA computes bfloat16 in binary32, where the conversions between
operations block the rewrites. -/
def rewritesIn (dtype : DType) : Bool :=
  dtype == DType.float16 || dtype == DType.float32 || dtype == DType.float64

/-- Formats in which the emulation rewrites divisions by square roots, reciprocal square roots
and powers. The IR computes binary16 `rsqrt` and `pow` in binary32, so these patterns are not
binary16 operations there. -/
def rootRewritesIn (dtype : DType) : Bool :=
  dtype == DType.float32 || dtype == DType.float64

/--
A division as XLA's CPU code computes it. A broadcast divisor, or under `jax.jit` a constant one,
becomes multiplication by its reciprocal rounded to the format, and bfloat16 computes both in
binary32 and rounds the product once. Otherwise, under `jax.jit`, the simplifier's rewrites
`(x / y) / z = x / (y * z)` and `x / (y / z) = (x * z) / y` apply in `rewritesIn` formats, then
`x / sqrt y = x * rsqrt y`, `x / rsqrt y = x * sqrt y` and `x / y ^ c = x * y ^ -c` in
`rootRewritesIn` formats, in that order, and the division a rewrite produces is computed by the
same rules.
-/
def emitDivision (fuel : Nat) (num den : IR.Ref) (shape : Shape) (dtype : DType)
    (site : Site) : Build IR.Ref := do
  let emulation ← get
  let divide (a b : IR.Ref) (d : DType) (s : Site) : Build IR.Ref :=
    emit (.node (.op ⟨_, _, Op.div (s := shape) .none .scalar⟩ #[a, b])) (some d) s
  let multiply (a b : IR.Ref) (d : DType) (s : Site) : Build IR.Ref :=
    emit (.node (.op ⟨_, _, Op.mul (s := shape) .scalar⟩ #[a, b])) (some d) s
  let constant := site.jitted && emulation.isFolded den.node &&
    emulation.within site.computation den.node
  if emulation.broadcastIn site.computation den || constant then
    let site := { site with difference := some Behaviour.divisorReciprocal }
    if dtype == DType.bfloat16 then
      let wide ← castTo num DType.float32 site
      let divisor ← castTo den DType.float32 site
      let one ← ones DType.float32 shape site
      let reciprocal ← divide one divisor DType.float32 site
      let product ← multiply wide reciprocal DType.float32 site
      castTo product DType.bfloat16 site
    else
      let one ← ones dtype shape site
      let reciprocal ← divide one den dtype site
      multiply num reciprocal dtype site
  else
    match fuel with
    | 0 => divide num den dtype site
    | fuel + 1 =>
      unless site.jitted && rewritesIn dtype do
        return ← divide num den dtype site
      let rewritten := { site with difference := some Behaviour.rewrittenDivision }
      let quotient := emulation.operation? site.computation
      if let some (.node (.op ⟨_, _, .div .none .scalar⟩ #[x, y])) := quotient num then
        let product ← multiply y den dtype rewritten
        return ← emitDivision fuel x product shape dtype rewritten
      if let some (.node (.op ⟨_, _, .div .none .scalar⟩ #[y, z])) := quotient den then
        let product ← multiply num z dtype rewritten
        return ← emitDivision fuel product y shape dtype rewritten
      unless rootRewritesIn dtype do
        return ← divide num den dtype site
      match emulation.operation? site.computation den with
      | some (.node (.op ⟨_, _, .sqrt .raw⟩ #[y])) =>
          let root ← emit (.node (.op ⟨_, _, Op.rsqrt (s := shape)⟩ #[y]))
            (some dtype) rewritten
          multiply num root dtype rewritten
      | some (.node (.op ⟨_, _, .rsqrt⟩ #[y])) =>
          let root ← emit (.node (.op ⟨_, _, Op.sqrt (s := shape) .raw⟩ #[y]))
            (some dtype) rewritten
          multiply num root dtype rewritten
      | some (.node (.op ⟨_, _, .pow .scalar⟩ #[y, c])) =>
          if emulation.isFolded c.node && emulation.within site.computation c.node then
            let negated ← emit (.node (.op ⟨_, _, Op.neg (s := shape) .scalar⟩ #[c]))
              (some dtype) rewritten
            let raised ← emit
              (.node (.op ⟨_, _, Op.pow (s := shape) .scalar⟩ #[y, negated]))
              (some dtype) rewritten
            multiply num raised dtype rewritten
          else divide num den dtype site
      | _ => divide num den dtype site

/-- The exponent `2`, `3` or `-1` that `jax.jit` computes as products or a reciprocal, when every
element of a binary16, binary32 or binary64 exponent is that integer. -/
def productExponent (dtype : DType) (exponent : Value Machine.reference) : Option Int := do
  guard (dtype == DType.float16 || dtype == DType.float32 || dtype == DType.float64)
  let .binary format := dtype | none
  let (_, _, words) ← (IR.words exponent).toOption
  guard !words.isEmpty
  [2, 3, -1].find? fun n => words.all (· == integerWord format n)

/-- The binary16 operand of a binary32 output: a conversion of a binary16 value, as the JAX
importer writes the base of a binary16 power. -/
def Emulation.halfSource (emulation : Emulation) (r : IR.Ref) : Option IR.Ref := do
  guard (emulation.dtypeAt r == some DType.float32)
  let some (.cast _ source _) := emulation.nodes[r.node]? | none
  guard (emulation.dtypeAt source == some DType.float16)
  pure source

/-- Build the emulated program from the IR program, its payload and the outputs `steps` of its
run, returning the emulated step that replaces each IR step. Under `jax.jit` a constant exponent
2, 3 or -1 becomes products or a reciprocal, each rounded to the format. XLA rounds every product
of a binary16 base to binary16, so the binary32 power that the JAX importer writes for it is
computed in binary16 and widened. -/
def emulate (program : IR.Program) (payload : Program.Payload Machine.reference)
    (compilation : Compilation) (steps : Array (List (Value Machine.reference))) :
    Except String (Emulation × Array Nat) := do
  let fuel := program.nodes.size
  let build : Build (Array Nat) := do
    let mut map : Array Nat := #[]
    for node in program.nodes do
      let i := map.size
      let site : Site :=
        { node := i, computation := (compilation.computation[i]?).join
          jitted := compilation.jitted i }
      let copied := remap (fun r => { r with node := map[r.node]! }) node
      let copy : Build IR.Ref :=
        emit copied (program.scalar i) site (constant := (payload.consts[i]?).join) (copy := true)
      let emulation ← get
      let replacement ← match copied, program.scalar i with
        | .node (.op ⟨_, _, .div (s := shape) .none .scalar⟩ #[num, den]), some (.binary format) =>
            emitDivision fuel num den shape (.binary format) site
        | .node (.op ⟨_, _, .pow (s := shape) .scalar⟩ #[base, exponent]), some dtype =>
            let products := { site with difference := some Behaviour.powerProducts }
            let constant := site.jitted && emulation.isFolded exponent.node &&
              emulation.within site.computation exponent.node && !emulation.isFolded base.node
            let integer := if constant then
                ((operands node)[1]? >>= fun r => (r.read steps).toOption) >>=
                  productExponent dtype
              else none
            let (operand, format) := match Emulation.halfSource emulation base with
              | some source => (source, DType.float16)
              | none => (base, dtype)
            let multiply (a b : IR.Ref) : Build IR.Ref :=
              emit (.node (.op ⟨_, _, Op.mul (s := shape) .scalar⟩ #[a, b]))
                (some format) products
            match integer with
            | none => copy
            | some n => do
                let result ← match n with
                  | 2 => multiply operand operand
                  | 3 => do
                      let square ← multiply operand operand
                      multiply operand square
                  | _ => do
                      let one ← ones format shape products
                      emitDivision fuel one operand shape format products
                if format == dtype then pure result else castTo result dtype products
        | _, _ => copy
      map := map.push replacement.node
    pure map
  let (map, emulation) ← build.run { jit := compilation.jit }
  pure (emulation, map)

/-- Whether XLA's CPU code flushes subnormal values of `dtype`: bfloat16, binary32 and binary64.
It computes binary16 in binary32, where binary16 subnormals are normal. -/
def flushes (dtype : DType) : Bool :=
  dtype == DType.bfloat16 || dtype == DType.float32 || dtype == DType.float64

/-- Operations whose subnormal operands and results XLA's CPU code flushes to zero, as measured.
Negation, `abs`, `sin`, layout operations and binary32 and bfloat16 `tanh` keep them. -/
def flushingOperation (node : Program.Node) (dtype : DType) : Bool :=
  match node with
  | .node (.op ⟨_, _, .add _⟩ _) | .node (.op ⟨_, _, .sub _⟩ _) | .node (.op ⟨_, _, .mul _⟩ _)
  | .node (.op ⟨_, _, .max _⟩ _) | .node (.op ⟨_, _, .min _⟩ _)
  | .node (.op ⟨_, _, .sqrt .raw⟩ _) | .node (.op ⟨_, _, .log⟩ _) | .node (.op ⟨_, _, .exp⟩ _)
  | .node (.op ⟨_, _, .sigmoid⟩ _) | .node (.op ⟨_, _, .reduce ..⟩ _)
  | .node (.op ⟨_, _, .mean ..⟩ _) | .node (.op ⟨_, _, .contract ..⟩ _)
  | .node (.op ⟨_, _, .matmul ..⟩ _) | .node (.op ⟨_, _, .div .none _⟩ _)
  | .node (.op ⟨_, _, .rsqrt⟩ _) => true
  | .node (.op ⟨_, _, .tanh⟩ _) => dtype == DType.float64
  | _ => false

/-- Whether step `id`, `node`, flushes its operands and its result, where a step that `jax.jit`
folds flushes neither. A conversion from binary32 or bfloat16 to binary64 flushes its operand, and
one from binary64 to binary32 or bfloat16 its result. -/
def Emulation.flushing (emulation : Emulation) (id : Nat) (node : Program.Node) : Bool × Bool :=
  if emulation.isFolded id then (false, false) else
  let dtype := (emulation.dtypeAt { node := id }).getD .bool
  match node with
  | .cast _ source _ =>
      let source := emulation.dtypeAt source
      (dtype == DType.float64 && (source == some DType.float32 || source == some DType.bfloat16),
        source == some DType.float64 && (dtype == DType.float32 || dtype == DType.bfloat16))
  | _ =>
      let flush := flushes dtype && flushingOperation node dtype
      (flush, flush)

/-- Replace the subnormal elements of a bfloat16, binary32 or binary64 value by zero of the same
sign, reporting whether any element changed. -/
def flushValue (value : Value Machine.reference) :
    Except String (Value Machine.reference × Bool) := do
  let some (.binary format) := value.typed.dtype? | return (value, false)
  unless flushes (.binary format) do return (value, false)
  let (_, _, words) ← IR.words value
  let sign := 2 ^ (format.bitWidth - 1)
  let flushed := words.map fun word =>
    if (Capture.binaryWordClass format word).2 then word / sign * sign else word
  if flushed == words then return (value, false)
  return (⟨value.typed, ← value.typed.decode Machine.reference flushed⟩, true)

/-- The emulated program with the outputs `outputs`: the emulation's steps and their scalars.
Nothing is proved about it here, and `Emulated.run` checks it (`Program.check`). -/
def Emulation.program (emulation : Emulation) (outputs : Array IR.Ref) : IR.Program :=
  { nodes := emulation.nodes, outputs, scalars := emulation.scalars }

/-- The payload of the emulated program: the emulation's constants, and the custom operations and
calls of the IR's payload `P` read through the origins of the copied steps. Nothing is proved
about it here, and `Emulated.run` checks it (`Program.Payload.check`). -/
def Emulation.payload (emulation : Emulation) (P : Program.Payload Machine.reference) :
    Program.Payload Machine.reference where
  consts := emulation.constants
  customs := emulation.origin.map fun origin => origin >>= fun i => (P.customs[i]?).join
  calls := emulation.origin.map fun origin => origin >>= fun i => (P.calls[i]?).join

/-- The emulated program with its payload, the documented behaviours applied to each step, and
the arguments. -/
structure Emulated where
  /-- The documented behaviours applied to each step. -/
  emulation : Emulation
  /-- The emulated program. -/
  program : IR.Program
  /-- Its payload. -/
  payload : Program.Payload Machine.reference
  /-- The arguments. -/
  arguments : Array (Value Machine.reference)

/-- Evaluate step `i` of the emulated program, `node`, by `Program.step` on the outputs
`earlier` of the preceding steps, with its operands and outputs flushed as `flushing` says,
returning its outputs and whether flushing changed a word. Nothing is proved about it here, and
`Emulated.run` calls it only on a program and payload that passed their checks. -/
def Emulated.evalStep (emulated : Emulated) (earlier : Array (List (Value Machine.reference)))
    (i : Nat) (node : Program.Node) :
    Except String (List (Value Machine.reference) × Bool) := do
  let (inputs, output) := Emulation.flushing emulated.emulation i node
  let mut available := earlier
  let mut flushed := false
  if inputs then
    for r in operands node do
      let (value, touched) ← flushValue (← r.read available)
      available := available.modify r.node (·.set r.out value)
      flushed := flushed || touched
  let (outputs, _) ← emulated.program.step Machine.reference emulated.payload emulated.arguments
    available i node
  if output then
    let results ← outputs.mapM flushValue
    pure (results.map (·.1), flushed || results.any (·.2))
  else pure (outputs, flushed)

/-- Run the emulated program after checking it (`Program.check`) and its payload
(`Program.Payload.check`), returning the outputs of its steps and whether flushing changed a
word of each step's outputs. -/
def Emulated.run (emulated : Emulated) :
    Except String (Array (List (Value Machine.reference)) × Array Bool) := do
  let program := emulated.program
  program.check Machine.reference
  emulated.payload.check program
  unless emulated.arguments.size = program.arity do
    throw s!"IR: {emulated.arguments.size} arguments for {program.arity} inputs"
  let mut steps : Array (List (Value Machine.reference)) := #[]
  let mut changed : Array Bool := #[]
  for node in program.nodes do
    let (outputs, touched) ← emulated.evalStep steps steps.size node
    steps := steps.push outputs
    changed := changed.push touched
  pure (steps, changed)

/-- A result that XLA's own function computes, and the window within which it counts as
reproduced. -/
structure Approximation where
  /-- XLA's function. -/
  behaviour : Behaviour
  /-- The output of the emulated `rsqrt` or `pow` step. -/
  source : IR.Ref
  /-- The window, in units in the last place of the source's format, or why there is none. -/
  ulps : Except String Nat
  /-- The outputs from the source to the result, in order, the result last. -/
  path : Array IR.Ref

/-- Whether output `r` is an argument, reached through layout operations and conversions that do
not round. -/
def Emulation.exactArgument (emulation : Emulation) (r : IR.Ref) : Bool := Id.run do
  let mut current := r
  for _ in [:emulation.nodes.size] do
    let some node := emulation.nodes[current.node]? | return false
    match node, (operands node)[0]? with
    | .node (.input ..), _ => return true
    | .node (.op ⟨_, _, .broadcast ..⟩ _), some parent
    | .node (.op ⟨_, _, .reshape ..⟩ _), some parent
    | .node (.op ⟨_, _, .permute ..⟩ _), some parent
    | .node (.op ⟨_, _, .detach⟩ _), some parent =>
        current := parent
    | .cast _ parent _, _ =>
        match emulation.dtypeAt current, emulation.dtypeAt parent with
        | some (.binary target), some (.binary source) =>
            if source.fracWidth ≤ target.fracWidth && source.expWidth ≤ target.expWidth then
              current := parent
            else return false
        | _, _ => return false
    | _, _ => return false
  return false

/-- XLA's own function computing result `output`: an `rsqrt` or `pow` step, reached through layout
operations, negation and conversions between binary formats, with its window when the step is of
binary32 or binary64. A power has a window only when both its operands are arguments
(`exactArgument`), so that XLA's function and the IR's receive the same values: a power with a
small exponent hides a large error in its base, and one with a large base a small error in its
exponent. -/
def Emulation.approximation (emulation : Emulation) (output : IR.Ref) :
    Option Approximation := Id.run do
  let mut current := output
  let mut path : Array IR.Ref := #[]
  for _ in [:emulation.nodes.size] do
    let some node := emulation.nodes[current.node]? | return none
    let dtype := (emulation.dtypeAt current).getD .bool
    let measured (behaviour : Behaviour) : Except String Nat :=
      match Behaviour.window behaviour dtype with
      | some ulps => .ok ulps
      | none => .error s!"it computes in {dtype.describe}, where no window was measured"
    match node, (operands node)[0]? with
    | .node (.op ⟨_, _, .rsqrt⟩ _), _ =>
        return some { behaviour := Behaviour.reciprocalSquareRoot, source := current,
                      ulps := measured Behaviour.reciprocalSquareRoot, path := path.reverse }
    | .node (.op ⟨_, _, .pow _⟩ refs), _ =>
        let ulps := if refs.all (exactArgument emulation) then measured Behaviour.power
          else .error "its operands are not both arguments of the program"
        return some { behaviour := Behaviour.power, source := current, ulps, path := path.reverse }
    | .node (.op ⟨_, _, .broadcast ..⟩ _), some parent
    | .node (.op ⟨_, _, .reshape ..⟩ _), some parent
    | .node (.op ⟨_, _, .permute ..⟩ _), some parent
    | .node (.op ⟨_, _, .detach⟩ _), some parent
    | .node (.op ⟨_, _, .neg _⟩ _), some parent =>
        path := path.push current
        current := parent
    | .cast _ parent _, _ =>
        match emulation.dtypeAt current, emulation.dtypeAt parent with
        | some (.binary _), some (.binary _) =>
            path := path.push current
            current := parent
        | _, _ => return none
    | _, _ => return none
  return none

/-- The words `ulps` steps below and above `word` in the order of the magnitudes of an IEEE
`format`, keeping its sign and staying finite: the window never crosses zero or reaches an
infinity. An infinity or a NaN is its own window. -/
def windowEnds (format : FloatFormat) (ulps word : Nat) : Nat × Nat :=
  let sign := 2 ^ (format.bitWidth - 1)
  let magnitude := word % sign
  let infinity := (2 ^ format.expWidth - 1) * 2 ^ format.fracWidth
  if magnitude ≥ infinity then (word, word)
  else
    let base := word - magnitude
    (base + (magnitude - ulps), base + min (magnitude + ulps) (infinity - 1))

/-- The ends of the window around each element of the approximation's source, carried to the
result through its path: rounding is monotone, and negation and layout operations move the ends
with the value. -/
def windowAt (emulated : Emulated) (steps : Array (List (Value Machine.reference)))
    (approximation : Approximation) : Except String (Option (Array Nat × Array Nat)) := do
  let .ok ulps := approximation.ulps | return none
  let .ok value := approximation.source.read steps
    | throw s!"XLA emulation: missing node {approximation.source.node}"
  let some (.binary format) := value.typed.dtype? | return none
  let (_, _, words) ← IR.words value
  let ends := words.map (windowEnds format ulps)
  let mut low : Value Machine.reference :=
    ⟨value.typed, ← value.typed.decode Machine.reference (ends.map (·.1))⟩
  let mut high : Value Machine.reference :=
    ⟨value.typed, ← value.typed.decode Machine.reference (ends.map (·.2))⟩
  for r in approximation.path do
    let some node := emulated.program.nodes[r.node]?
      | throw s!"XLA emulation: missing node {r.node}"
    let some operand := (operands node)[0]?
      | throw s!"XLA emulation: node {r.node} has no operand"
    let earlier := steps.extract 0 r.node
    let through (bound : Value Machine.reference) : Except String (Value Machine.reference) := do
      let (outputs, _) ← emulated.evalStep
        (earlier.modify operand.node (·.set operand.out bound)) r.node node
      let some result := outputs[r.out]? | throw s!"XLA emulation: missing node {r.node}"
      pure result
    low ← through low
    high ← through high
  pure (some ((← IR.words low).2.2, (← IR.words high).2.2))

/-- Whether the reference word `wanted` lies between the window's ends `low` and `high` at the
result and has the sign and the finiteness of the emulated word: a difference in the sign of zero
or between a finite value and an infinity is outside every window. -/
def inWindow (format : FloatFormat) (emulated wanted low high : Nat) : Bool :=
  let x : Model format := Model.ofNatBits wanted
  let y : Model format := Model.ofNatBits emulated
  if (Capture.binaryWordClass format emulated).1 ||
    (Capture.binaryWordClass format wanted).1 then false
  else if x.signBit != y.signBit || x.isInf != y.isInf then false
  else
    let sign := 2 ^ (format.bitWidth - 1)
    let ordinal (word : Nat) : Int := if word ≥ sign then -((word - sign : Nat) : Int) else word
    let (a, b) := (ordinal low, ordinal high)
    decide (min a b ≤ ordinal wanted) && decide (ordinal wanted ≤ max a b)

/-- The steps that step `output` depends on, itself included. -/
def dependencies (nodes : Array Program.Node) (output : Nat) : Array Bool := Id.run do
  let mut needed := (Array.replicate nodes.size false).set! output true
  for offset in [:nodes.size] do
    let id := nodes.size - 1 - offset
    unless needed[id]! do continue
    let some node := nodes[id]? | continue
    for r in operands node do
      needed := needed.set! r.node true
  return needed

/-- A word in hexadecimal. -/
def hex (word : Nat) : String := "0x" ++ String.ofList (Nat.toDigits 16 word)

/-- The window used for a result: XLA's function, its width and format, and the differing
elements within it. -/
structure WindowUse where
  /-- XLA's function. -/
  behaviour : Behaviour
  /-- Units in the last place. -/
  ulps : Nat
  /-- The format the function computes in. -/
  dtype : DType
  /-- The result's format, which the window's ends are rounded to. -/
  result : DType
  /-- Differing elements within the window. -/
  elements : Nat

/-- What the emulation of XLA's CPU code reproduces of one differing result. -/
structure Reproduction where
  /-- Position of the result. -/
  result : Nat
  /-- Its differing elements. -/
  mismatch : Capture.Mismatch
  /-- Documented behaviours applied to nodes the result depends on, each with the IR nodes it
  rewrites, or for flushing the emulated nodes where it changed a word. -/
  applied : Array (Behaviour × Nat)
  /-- Differing elements whose words the emulated program returns. -/
  reproduced : Nat
  /-- The window of XLA's own function computing the result, if one applies. -/
  window : Option WindowUse
  /-- XLA's own function computing the result when no window applies, and why. -/
  withheld : Option (Behaviour × String) := none
  /-- Elements not reproduced, with the IR's, the emulation's and the reference's words. -/
  notReproduced : Array (Nat × Nat × Nat × Nat)

namespace Reproduction

/-- The behaviours the reproduction used: those applied, and XLA's function if its window
holds a differing element. -/
def used (entry : Reproduction) : Array Behaviour :=
  let applied := entry.applied.map (·.1)
  match entry.window with
  | some use => if use.elements > 0 then applied.push use.behaviour else applied
  | none => applied

/-- The report entry of a differing result. -/
def toJson (entry : Reproduction) : Json :=
  let applied := entry.applied.map fun (d, count) =>
    Json.mkObj [("kind", Lean.toJson d.code), ("nodes", Lean.toJson count)]
  let window := match entry.window with
    | some use => Json.mkObj [("kind", Lean.toJson use.behaviour.code),
        ("ulps", Lean.toJson use.ulps), ("format", Lean.toJson use.dtype.describe),
        ("elements", Lean.toJson use.elements)]
    | none => Json.null
  Json.mkObj [
    ("result", Lean.toJson entry.result), ("elements", Lean.toJson entry.mismatch.total),
    ("differing", Lean.toJson entry.mismatch.elements.size),
    ("nan_words", Lean.toJson entry.mismatch.nanWords),
    ("emulated", Lean.toJson applied), ("reproduced", Lean.toJson entry.reproduced),
    ("window", window), ("not_reproduced", Lean.toJson entry.notReproduced.size),
    ("examples", Lean.toJson ((entry.notReproduced.extract 0 3).map fun (element, ir, emulated,
        wanted) => Json.mkObj [("element", Lean.toJson element), ("ir", Lean.toJson ir),
          ("emulated", Lean.toJson emulated), ("reference", Lean.toJson wanted)]))]

/-- The behaviours applied, in words. -/
def behaviours (entry : Reproduction) : String :=
  ", ".intercalate (entry.applied.toList.map fun (d, count) =>
    s!"{d.description} ({count} {if count == 1 then "node" else "nodes"})")

/-- The window, in words. -/
def _root_.TorchLean.XLA.WindowUse.describe (use : WindowUse) : String :=
  let rounded := if use.result == use.dtype then "" else
    s!", with both ends rounded to {use.result.describe}"
  s!"{use.ulps} {use.dtype.describe} {if use.ulps == 1 then "ulp" else "ulps"} of the emulated \
    value of {use.behaviour.description}{rounded}"

/-- One sentence for the report message. -/
def describe (entry : Reproduction) : String :=
  let counts := s!"result {entry.result}: {entry.mismatch.elements.size} of \
    {entry.mismatch.total} elements differ"
  let reproduced := if entry.reproduced > 0 then
      [s!"emulating {entry.behaviours} reproduces {entry.reproduced} of them"] else []
  let within := match entry.window with
    | some use => if use.elements > 0 then
        [s!"{use.elements} {if use.elements == 1 then "is" else "are"} within \
          {WindowUse.describe use}"] else []
    | none => []
  if entry.notReproduced.isEmpty then
    counts ++ "; " ++ "; ".intercalate (reproduced ++ within)
  else
    let emulated := if entry.applied.isEmpty then [] else [s!"after emulating {entry.behaviours}"]
    let allowed := match entry.window, entry.withheld with
      | some use, _ => [s!"allowing {WindowUse.describe use}"]
      | none, some (behaviour, reason) =>
          [s!"with no window for {behaviour.description}, since {reason}"]
      | none, none => []
    let tried := if emulated.isEmpty && allowed.isEmpty then
        "no documented XLA behaviour applies to the operations it depends on"
      else " and ".intercalate (emulated ++ allowed)
    let examples := ", ".intercalate ((entry.notReproduced.extract 0 3).toList.map
      fun (element, ir, emulated, wanted) =>
        s!"element {element} (IR {hex ir}, emulated {hex emulated}, reference {hex wanted})")
    counts ++ s!"; {tried}, {entry.notReproduced.size} are not reproduced, e.g. {examples}"

end Reproduction

/-- What the emulated program reproduces of result `index`'s differing elements. -/
def reproduceResult (bitwise : Bool) (emulated : Emulated)
    (steps : Array (List (Value Machine.reference))) (changed : Array Bool) (output : IR.Ref)
    (index : Nat) (ir reference : Value Machine.reference) (mismatch : Capture.Mismatch) :
    Except String Reproduction := do
  let emulation := emulated.emulation
  let .ok value := output.read steps | throw s!"XLA emulation: missing result {index}"
  let (_, _, irWords) ← IR.words ir
  let (_, _, wanted) ← IR.words reference
  let (dtype, _, words) ← IR.words value
  let approximated := Emulation.approximation emulation output
  let ends ← match approximated with
    | some approximation => windowAt emulated steps approximation
    | none => pure none
  let mut reproduced := 0
  let mut within := 0
  let mut notReproduced : Array (Nat × Nat × Nat × Nat) := #[]
  for element in mismatch.elements do
    let a := words[element]!
    let b := wanted[element]!
    let close := match ends, dtype with
      | some (low, high), .binary format => inWindow format a b low[element]! high[element]!
      | _, _ => false
    if Capture.agree bitwise dtype a b then reproduced := reproduced + 1
    else if close then within := within + 1
    else notReproduced := notReproduced.push (element, irWords[element]!, a, b)
  let needed := dependencies emulation.nodes output.node
  let mut sites : Array (Behaviour × Nat) := #[]
  let mut flushed := 0
  for id in [:emulation.nodes.size] do
    unless needed[id]! do continue
    if let some site := (emulation.site[id]?).join then
      unless sites.contains site do sites := sites.push site
    if changed[id]?.getD false then flushed := flushed + 1
  let mut applied : Array (Behaviour × Nat) := #[]
  for (difference, _) in sites do
    unless applied.any (·.1 == difference) do
      applied := applied.push (difference, (sites.filter (·.1 == difference)).size)
  if flushed > 0 then applied := applied.push (Behaviour.subnormal, flushed)
  let window := match approximated, ends with
    | some approximation, some _ =>
        some { behaviour := approximation.behaviour, ulps := approximation.ulps.toOption.getD 0,
               dtype := (emulation.dtypeAt approximation.source).getD .bool, result := dtype,
               elements := within }
    | _, _ => none
  let withheld := match approximated with
    | some { behaviour, ulps := .error reason, .. } => some (behaviour, reason)
    | _ => none
  pure { result := index, mismatch, applied, reproduced, window, withheld, notReproduced }

/-- Run the emulation of XLA's CPU code that `compilation` describes on the IR program `program`,
its payload and the outputs `steps` of its run, and report what it reproduces of each differing
result. -/
def reproduce (bitwise : Bool) (compilation : Compilation) (program : IR.Program)
    (payload : Program.Payload Machine.reference) (arguments : Array (Value Machine.reference))
    (steps : Array (List (Value Machine.reference)))
    (results reference : Array (Value Machine.reference))
    (differing : Array (Nat × Capture.Mismatch)) :
    Except String (Array Reproduction) := do
  let (emulation, map) ← emulate program payload compilation steps
  let outputs := program.outputs.map fun r => { r with node := map[r.node]! }
  let emulated : Emulated :=
    { emulation, arguments, program := emulation.program outputs
      payload := emulation.payload payload }
  let (emulatedSteps, changed) ← emulated.run
  differing.mapM fun (index, mismatch) => do
    let some output := outputs[index]? | throw s!"XLA emulation: no result {index}"
    let some ir := results[index]? | throw s!"XLA emulation: no IR result {index}"
    let some wanted := reference[index]? | throw s!"XLA emulation: no reference {index}"
    reproduceResult bitwise emulated emulatedSteps changed output index ir wanted mismatch

/-- Compare the IR's results with the reference tensors in `field`, returning the references and
the elements that differ. -/
def _root_.TorchLean.Capture.mismatches (bitwise : Bool) (document : Json) (field : String)
    (results : Array (Value Machine.reference)) :
    Except String (Array (Value Machine.reference) × Array (Nat × Capture.Mismatch)) := do
  let reference ← (← (← document.getObjVal? field).getArr?).mapM
    (IR.loadValue Machine.reference)
  unless results.size == reference.size do
    throw s!"IR comparison: {field} has {reference.size} results, the program {results.size}"
  let found ← (results.zip reference).mapIdxM fun i (output, expected) =>
    match Capture.compareResult bitwise output expected with
    | .ok mismatch => pure mismatch
    | .error message => throw s!"IR comparison: result {i} has {message}"
  pure (reference, (found.mapIdx fun i m => (i, m)))

/-- The behaviours a set of reproductions used, each once, in order. -/
def usedBehaviours (entries : Array Reproduction) : Array Behaviour :=
  entries.foldl (init := #[]) fun found entry =>
    entry.used.foldl (init := found) fun found behaviour =>
      if found.contains behaviour then found else found.push behaviour

/-- Their codes. -/
def behaviourCodes (entries : Array Reproduction) : Array String :=
  (usedBehaviours entries).map (·.code)

/-- What the primitive-by-primitive reference says about a mismatch with a `jax.jit` reference:
whether the IR agrees with it, or the emulation of XLA's CPU code without `jax.jit` reproduces it,
with a sentence and report entries. -/
def _root_.TorchLean.Capture.eagerCheck (bitwise : Bool) (document : Json) (program : IR.Program)
    (payload : Program.Payload Machine.reference) (arguments : Array (Value Machine.reference))
    (steps : Array (List (Value Machine.reference))) (results : Array (Value Machine.reference)) :
    Bool × String × Json :=
  let outcome : Except String (Bool × String × Json) := do
    let (reference, found) ← Capture.mismatches bitwise document "eager_reference_outputs" results
    let differing := found.filter (!·.2.elements.isEmpty)
    if differing.isEmpty then
      return (true, "the IR's results equal the primitive-by-primitive reference",
        Json.mkObj [("status", "agrees")])
    let compilation := Compilation.read document (perPrimitive := true)
    let entries ← reproduce bitwise compilation program payload arguments steps results
      reference differing
    let summary := "; ".intercalate (entries.toList.map Reproduction.describe)
    let json := Json.mkObj [("results", toJson (entries.map Reproduction.toJson))]
    if entries.all (·.notReproduced.isEmpty) then
      return (true, s!"against the primitive-by-primitive reference, {summary}",
        json.setObjVal! "status" "consistent" |>.setObjVal! "consistent_with"
          (toJson (behaviourCodes entries)))
    return (false, s!"against the primitive-by-primitive reference, which XLA computed without \
      jax.jit's rewrites, {summary}", json.setObjVal! "status" "not_reproduced")
  match outcome with
  | .ok result => result
  | .error message => (false, s!"the primitive-by-primitive reference could not be compared: \
      {message}", Json.mkObj [("status", "unreadable")])

/-- The `compare` or `compare_bitwise` finding for the results `results` of the program `program`
with the payload `payload`, whose run gave the outputs `steps`: any difference fails. For a Jaxpr
document with a CPU reference, a failure whose differing elements the emulation of XLA's CPU code
reproduces carries the behaviours in `consistent_with` as advice. With a `jax.jit` reference this
also needs the document's primitive-by-primitive reference, when it has one, to agree with the IR
or be reproduced without `jax.jit`. -/
def _root_.TorchLean.Capture.compareFinding (name : String) (document : Json) (program : IR.Program)
    (payload : Program.Payload Machine.reference) (arguments : Array (Value Machine.reference))
    (steps : Array (List (Value Machine.reference))) (results : Array (Value Machine.reference)) :
    Bool × Json :=
  let bitwise := name == "compare_bitwise"
  match Capture.mismatches bitwise document "reference_outputs" results with
  | .error message => (false, Capture.finding name "failed" message)
  | .ok (reference, found) =>
      let nanWords := found.foldl (· + ·.2.nanWords) 0
      let differing := found.filter (!·.2.elements.isEmpty)
      let counts := "; ".intercalate (differing.toList.map fun (i, m) =>
        s!"result {i}: {m.elements.size} of {m.total} elements differ")
      let failure := "IR comparison: results differ in their interchange words: "
      let platform := document.getObjValD "reference_platform"
      if differing.isEmpty then
        let message :=
          if bitwise then "Interchange words, including NaN signs and payloads, agree for the \
            supplied arguments only."
          else s!"Interchange words agree for the supplied arguments only. Two NaN results \
            agree whatever their sign and payload ({nanWords} such elements); request \
            `compare_bitwise` to compare their words too."
        (true, Capture.finding name "compared" message)
      else if (document.getObjValD "frontend").getStr?.toOption != some "jaxpr" then
        (false, Capture.finding name "failed" (failure ++ counts))
      else if platform != .str "cpu" then
        (false, Capture.finding name "failed"
          s!"{failure}{counts}. The artifact does not record that \
          XLA's CPU backend computed the reference (reference_platform: {platform.compress}), so \
          XLA's CPU behaviour is not emulated.")
      else
        let compilation := Compilation.read document
        match reproduce bitwise compilation program payload arguments steps results reference
            differing with
        | .error message =>
            (false, Capture.finding name "failed"
              s!"{failure}{counts}; the emulation of XLA's CPU code \
              could not run: {message}")
        | .ok entries =>
            let report := ("results", toJson (entries.map Reproduction.toJson))
            let summary := "; ".intercalate (entries.toList.map Reproduction.describe)
            if !entries.all (·.notReproduced.isEmpty) then
              (false, Capture.finding name "failed" (failure ++ summary) [report])
            else
              let codes := behaviourCodes entries
              let descriptions := ", ".intercalate
                ((usedBehaviours entries).toList.map (·.description))
              let advice := s!"{failure}{summary}. This is consistent with XLA's documented CPU \
                behaviour ({descriptions}), but an importer error with the same effect on these \
                arguments cannot be ruled out, so the comparison fails."
              let hasEager := (document.getObjVal? "eager_reference_outputs").toOption.isSome
              if !compilation.usesJit then
                (false, Capture.finding name "failed" advice
                  [("consistent_with", toJson codes), report])
              else if !hasEager then
                (false, Capture.finding name "failed"
                  (advice ++ " No primitive-by-primitive reference \
                  was recorded to compare the IR with XLA's results without jax.jit's rewrites.")
                  [("consistent_with", toJson codes), report])
              else
                let (consistent, sentence, eager) :=
                  Capture.eagerCheck bitwise document program payload arguments steps results
                if consistent then
                  (false, Capture.finding name "failed"
                    (advice ++ s!" Without jax.jit, {sentence}.")
                    [("consistent_with", toJson codes), report, ("eager_reference", eager)])
                else
                  (false, Capture.finding name "failed"
                    s!"{failure}{summary}. Emulating XLA's CPU code \
                    under jax.jit reproduces the reference, but {sentence}, so the mismatch is not \
                    attributed to XLA's documented behaviour."
                    [report, ("eager_reference", eager)])

/-- Serialize one computed result without a lossy numerical display conversion. A Boolean,
integer or scalar result has its element type, shape and words (`IR.words`). A result of index
tuples has `"element_type": "index"`, a null `dtype`, its port (`Port.toJson`), the shape of the
integer tensor that holds its tuples (`Port.irShape`) and the components of its tuples in that
layout. -/
def _root_.TorchLean.Capture.resultJson : Value Machine.reference → Except String Json
  | ⟨⟨⟨.index bounds, s⟩, _⟩, x⟩ => pure <| Json.mkObj [
      ("element_type", toJson "index"), ("dtype", .null), ("port", Port.toJson ⟨.index bounds, s⟩),
      ("shape", toJson (Port.irShape ⟨.index bounds, s⟩)),
      ("bits", toJson ((Tensor.Rep.data x).toList.flatMap (Port.IndexTuple.components bounds)))]
  | value => do
      let (dtype, shape, words) ← IR.words value
      pure <| Json.mkObj [
        ("element_type", toJson dtype.describe), ("dtype", toJson dtype),
        ("shape", toJson shape), ("bits", toJson words)]

/-- The JSON of a run's conversion status: each of its seven flags, raised or not. The flags are
read by destructuring the status, so a flag added to `ConversionStatus` cannot be left out. -/
def _root_.TorchLean.Capture.flagsJson (flags : FloatLib.Floats.ExecFloat.ConversionStatus) :
    Json :=
  let ⟨inexact, overflow, underflow, saturated, wrapped, mappedSpecial, invalid⟩ := flags
  Json.mkObj [("inexact", toJson inexact), ("overflow", toJson overflow),
    ("underflow", toJson underflow), ("saturated", toJson saturated), ("wrapped", toJson wrapped),
    ("mapped_special", toJson mappedSpecial), ("invalid", toJson invalid)]

end XLA

/-- Run requested checks, returning failures and unsupported requests as report entries.

`execute`, `compare` and `compare_bitwise` concern only `arguments` stored in the document. A
comparison can expose legitimate differences in reduction or elementary-function implementations,
so a mismatch alone does not identify a bug, but it always fails.

After a run, the report's `outputs` holds each result as `Capture.resultJson` writes it, and
`conversion_flags` the run's complete conversion status (`Capture.flagsJson`). A result that
`Capture.resultJson` cannot write fails `execute`, `compare` and `compare_bitwise` with its message.
-/
def Capture.check (document : Json) (requests : Array String := #["structure", "types"]) :
    Json := Id.run do
  let mut findings : Array Json := #[]
  let mut ok := true
  let mut outputs : Json := .null
  let mut conversionFlags : Json := .null
  let loaded := IR.load document
  let execution : Except String _ := do
    unless requests.any (fun request =>
        request == "execute" || request == "compare" || request == "compare_bitwise") do
      throw "Execution was not requested."
    let (program, payload) ← loaded
    let arguments ← IR.loadArguments Machine.reference document
    let (steps, flags) ← program.run Machine.reference payload arguments
    let results ← program.outputs.mapM (·.read steps)
    let written ← results.mapIdxM fun i result =>
      (Capture.resultJson result).mapError (s!"IR: result {i} cannot be serialized: " ++ ·)
    pure (program, payload, arguments, steps, results, flags, written)
  if let .ok (_, _, _, _, _, flags, written) := execution then
    outputs := toJson written
    conversionFlags := flagsJson flags
  for request in requests do
    let (passed, entry) :=
      match request with
      | "structure" | "types" =>
          match loaded with
          | .ok _ => (true, finding request "checked"
              "Declared dependencies, shapes, types and casts are consistent.")
          | .error message => (false, finding request "failed" message)
      | "execute" | "compare" | "compare_bitwise" =>
          match execution with
          | .error message => (false, finding request "failed" message)
          | .ok (program, payload, arguments, steps, results, _, _) =>
              if request == "execute" then
                (true, finding request "executed"
                  "CPU reference execution completed for the supplied arguments.")
              else
                compareFinding request document program payload arguments steps results
      | _ =>
          (false, finding request "unsupported"
            s!"No checker for `{request}`; no proof was produced.")
    ok := ok && passed
    findings := findings.push entry
  if requests.isEmpty then
    ok := false
    findings := #[finding "request" "failed" "At least one check must be requested."]
  return Json.mkObj [
    ("format", toJson "torchlean.verification"), ("ok", toJson ok),
    ("checks", toJson findings), ("outputs", outputs), ("conversion_flags", conversionFlags),
    ("capture_constraints", (document.getObjVal? "capture_constraints").toOption.getD (.arr #[])),
    ("capture_constraints_status", toJson "not_checked"),
    ("frontend_equivalence", toJson "not_checked"),
    ("frontend_device_layout_constraints", toJson "not_checked"),
    ("external_kernels", toJson "not_checked")]

end TorchLean
