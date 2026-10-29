/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.IR
public import NN.Runtime.PyTorch.Export.TorchExport

/-!
# Checking captured tensor computations

This entry point checks a typed IR artifact and optionally executes supplied
arguments or compares supplied reference outputs. The result states the scope of
each check. Passing these checks is not a theorem about all inputs, a proof of the
frontend translation, or a validation of external kernels.

The report preserves the artifact's `capture_constraints` as unchecked metadata. Structure,
type and example-execution checks do not establish these conditions for all inputs.

`compare` treats two NaN results as equal whatever their sign and payload, which the IR leaves
unspecified; `compare_bitwise` requires equal words for NaNs too. Any other difference between the
IR's results and the reference fails. For a Jaxpr artifact whose reference XLA's CPU backend
computed, the checker then runs the program a second time as XLA's compiled CPU code computes it,
where this is exactly emulable: reciprocals of broadcast and constant divisors, `jax.jit`'s
division and power rewrites when the artifact records a `jax.jit` reference, and the flushing of
subnormal values. A value that XLA's own `rsqrt` or power function computes counts as reproduced
within the window of `Internal.window`, measured in the format the function computes in. If the
second run reproduces every differing element, the failed finding lists the behaviours in
`consistent_with`. That field is advice, not an explanation: an importer error with the same
effect on these arguments cannot be ruled out, and some importer errors coincide with XLA's own
rewrites. Under `jax.jit` it also requires the artifact's primitive-by-primitive reference, when
there is one, to agree with the IR or to be reproduced without `jax.jit`'s rewrites. The checker
trusts the artifact's record of how the reference was computed, so a forged record can at most
add this advice to a failure.

Unknown requests are reported as unsupported. They must not disappear from a
successful report. Existing certificate checkers and mathematical proof APIs
remain separate tools with their own contracts.
-/

@[expose] public section

namespace NN.Verification.IR

open Lean NN.IR
open FloatLib.Floats.Formats.BinaryInterchange

namespace Internal

/-- One named check with its outcome and an explanation of its scope. -/
def finding (name status message : String) (extra : List (String × Json) := []) : Json :=
  Json.mkObj ([("name", toJson name), ("status", toJson status), ("message", toJson message)] ++
    extra)

/-- Read a supported tensor from complete element words, with no host-float conversion. -/
def value (json : Json) : Except String (Interpreter.Value Interpreter.reference) :=
  Import.IR.parseValue Interpreter.reference Reference.load json

/-- FloatLib's classification of one binary word: whether it is a NaN and whether it is
subnormal, under the format's exceptional-value encoding. -/
def binaryWordClass (format : FloatFormat) (word : Nat) : Bool × Bool :=
  let model : Model format := Model.ofNatBits word
  (Model.isNaN model, Model.isSubnormal model)

/-- Whether two words of `dtype` agree: equal, or both NaNs unless `bitwise` is set. -/
def agree (bitwise : Bool) (dtype : DType) (a b : Nat) : Bool :=
  a == b || !bitwise && match dtype with
    | .binary format => (binaryWordClass format a).1 && (binaryWordClass format b).1
    | _ => false

/-- The elements of one result that disagree with its reference. -/
structure Mismatch where
  /-- Positions of the elements that disagree under the requested NaN policy. -/
  elements : Array Nat := #[]
  /-- Elements where both words are NaNs with a different sign or payload. -/
  nanWords : Nat := 0
  /-- Elements of the result. -/
  total : Nat := 0

/-- Compare one result with its reference. Two NaNs agree unless `bitwise` is set. -/
def compareResult (bitwise : Bool) (output expected : Interpreter.Value Interpreter.reference) :
    Except String Mismatch := do
  unless output.dtype == expected.dtype && output.shape == expected.shape do
    throw "a different element type or shape"
  let actual ← Reference.words output
  let wanted ← Reference.words expected
  let mut result : Mismatch := { total := actual.size }
  for index in [:actual.size] do
    let a := actual[index]!
    let b := wanted[index]!
    if a != b then
      if agree false output.dtype a b then
        result := { result with nanWords := result.nanWords + 1 }
      unless agree bitwise output.dtype a b do
        result := { result with elements := result.elements.push index }
  pure result

/-! ## XLA's documented CPU behaviour

The behaviours, how the artifact says XLA compiled the reference, and the emulated program that
`compare` runs for a differing Jaxpr result.
-/

/-- A documented way in which XLA's compiled CPU code computes differently from the IR, with the
code that a failed comparison lists in `consistent_with` and the description of
`NN/IR/README.md`. -/
structure XlaBehaviour where
  /-- Stable code in the report. -/
  code : String
  /-- What XLA does differently. -/
  description : String
  deriving BEq

/-- A division by a broadcast divisor, or under `jax.jit` by a constant. -/
def divisorReciprocal : XlaBehaviour :=
  ⟨"divisor_reciprocal", "division by a broadcast divisor, or under jax.jit by a constant, as \
    multiplication by the divisor's rounded reciprocal"⟩

/-- The simplifier's division rewrites under `jax.jit`. -/
def rewrittenDivision : XlaBehaviour :=
  ⟨"rewritten_division", "jax.jit's rewrites of a division of a quotient, or by a quotient, \
    square root, reciprocal square root or power"⟩

/-- Constant exponents computed as products under `jax.jit`. -/
def powerProducts : XlaBehaviour :=
  ⟨"pow_products", "jax.jit's products for the constant exponents 2 and 3 and reciprocal for -1"⟩

/-- Subnormal values flushed to zero. -/
def subnormal : XlaBehaviour :=
  ⟨"subnormal", "subnormal operands and results flushed to zero in bfloat16, binary32 and \
    binary64"⟩

/-- XLA's own reciprocal square root. -/
def reciprocalSquareRoot : XlaBehaviour :=
  ⟨"rsqrt", "XLA's own reciprocal square root"⟩

/-- XLA's own power function. -/
def power : XlaBehaviour :=
  ⟨"pow", "XLA's own power function"⟩

/--
The window, in units in the last place of `dtype`, within which a value that XLA's own `rsqrt` or
power function computes in `dtype` counts as reproduced: the largest distance measured between
XLA's function and the IR's in that format, 2 for `rsqrt` and 1 for the power, in binary32 and
binary64. XLA's power is within 1 ulp of the IR's for integral exponents and where `|y log x| < 1`;
elsewhere FloatLib's power is farther from it, and such a result is not reproduced. XLA computes
binary16 and bfloat16 `rsqrt` and powers in binary32 and rounds once, so the window applies to the
binary32 value, before that rounding; a node of a reduced format has none.
-/
def window (behaviour : XlaBehaviour) (dtype : DType) : Option Nat :=
  if !(dtype == DType.float32 || dtype == DType.float64) then none
  else if behaviour == reciprocalSquareRoot then some 2
  else if behaviour == power then some 1
  else none

/-- How XLA compiled a Jaxpr reference, as the artifact records it. -/
structure Compilation where
  /-- The reference came from `jax.jit`, which compiles the whole program as one computation. -/
  jit : Bool := false
  /-- Otherwise, for each node, the computation XLA compiled it in: the JAX importer's index of
  the primitive, or of the nested `jax.jit` call, that produced it. -/
  computation : Array (Option Nat) := #[]
  /-- The computations that are nested `jax.jit` calls. -/
  jitComputations : Array Nat := #[]

namespace Compilation

/-- Read `reference_execution`, `jit_computations` and each node's `computation`; or, with
`perPrimitive`, describe the primitive-by-primitive reference, where each primitive, inside nested
`jax.jit` calls too, is a computation of its own, identified by the node's `primitive_index`. A raw
node that the lowering expands owns every node up to its result, as in the type assignment of
`Import.IR.Internal.parse`. -/
def read (artifact : Json) (perPrimitive : Bool := false) : Except String Compilation := do
  let (raw, captured, _) ← Import.IR.Internal.parse artifact
  let field := if perPrimitive then "primitive_index" else "computation"
  let jit := !perPrimitive && artifact.getObjValD "reference_execution" == .str "jit"
  let jitComputations := if perPrimitive then #[] else
    match artifact.getObjValD "jit_computations" with
    | .arr ids => ids.filterMap (·.getNat?.toOption)
    | _ => #[]
  let mut computation : Array (Option Nat) := #[]
  for node in raw.nodes do
    let some (some id) := captured.rawToTensor[node.id]? | continue
    let unit := (node.raw.get? field).bind (·.getNat?.toOption)
    for _ in [computation.size:id + 1] do
      computation := computation.push unit
  pure { jit, computation, jitComputations }

/-- Whether node `id` is compiled by `jax.jit`. -/
def jitted (compilation : Compilation) (id : Nat) : Bool :=
  compilation.jit ||
    (((compilation.computation[id]?).join).map compilation.jitComputations.contains).getD false

/-- Whether `jax.jit` compiles any part of the program. -/
def usesJit (compilation : Compilation) : Bool :=
  compilation.jit || !compilation.jitComputations.isEmpty

end Compilation

/-- The program as XLA's CPU code computes it, built node by node from the IR program. -/
structure Emulation where
  /-- The reference came from `jax.jit`. -/
  jit : Bool
  /-- Nodes of the emulated program. -/
  nodes : Array Node := #[]
  /-- Element type of each node. -/
  dtypes : Array DType := #[]
  /-- Casts and registered operations, aligned with `nodes`. -/
  primitives : Array (Option Primitive) := #[]
  /-- Values of the copied constants and of the ones that the rewrites add. -/
  constants : Array (Option (Interpreter.Value Interpreter.reference)) := #[]
  /-- The IR node each copied node comes from, for its payload entries. -/
  origin : Array (Option Nat) := #[]
  /-- The computation each node belongs to, as in `Compilation`. -/
  computation : Array (Option Nat) := #[]
  /-- Whether `jax.jit` folds each node at compile time: it is computed from constants alone
  within one `jax.jit` computation. -/
  folded : Array Bool := #[]
  /-- The documented behaviour that produced each node, with the IR node it rewrites. -/
  site : Array (Option (XlaBehaviour × Nat)) := #[]

namespace Emulation

/-- Whether node `id` belongs to `computation`. Under `jax.jit` every node does. -/
def within (emulation : Emulation) (computation : Option Nat) (id : Nat) : Bool :=
  emulation.jit || match (emulation.computation[id]?).join, computation with
    | some a, some b => a == b
    | _, _ => false

/-- Whether node `id` is folded at compile time. -/
def isFolded (emulation : Emulation) (id : Nat) : Bool :=
  emulation.folded[id]?.getD false

/-- The operation and operands of node `id` when it belongs to `computation`. -/
def operation? (emulation : Emulation) (computation : Option Nat) (id : Nat) :
    Option (OpKind × Array Nat) := do
  let node ← emulation.nodes[id]?
  guard (emulation.within computation id)
  pure (node.kind, node.parents)

/-- Whether a divisor is a broadcast computed in `computation`, past reshapes and permutations. -/
def broadcastIn (emulation : Emulation) (computation : Option Nat) (id : Nat) : Bool := Id.run do
  let mut current := id
  for _ in [:emulation.nodes.size] do
    let some node := emulation.nodes[current]? | return false
    unless emulation.within computation current do return false
    match node.kind, node.parents[0]? with
    | .broadcastTo _ _, _ => return true
    | .reshape _ _, some parent | .permute _, some parent | .transpose _ _, some parent
    | .flatten _, some parent | .detach, some parent => current := parent
    | _, _ => return false
  return false

end Emulation

/-- The IR node being emulated, its computation, and the documented behaviour that the new nodes
emulate, if any. -/
structure Site where
  /-- The IR node. -/
  node : Nat
  /-- Its computation. -/
  computation : Option Nat
  /-- Whether `jax.jit` compiles it. -/
  jitted : Bool
  /-- The behaviour the new nodes emulate. -/
  difference : Option XlaBehaviour := none

/-- Construction of the emulated program. -/
abbrev Build := StateT Emulation (Except String)

/-- Add a node for `site`, returning its id. -/
def emit (kind : OpKind) (parents : Array Nat) (shape : Spec.Shape) (dtype : DType) (site : Site)
    (primitive : Option Primitive := none)
    (constant : Option (Interpreter.Value Interpreter.reference) := none) (copy : Bool := false) :
    Build Nat := do
  let emulation ← get
  let folded := match kind with
    | .const _ => site.jitted
    | .input | .randUniform _ | .bernoulliMask _ => false
    | _ => site.jitted && !parents.isEmpty && parents.all fun parent =>
        emulation.isFolded parent && emulation.within site.computation parent
  let id := emulation.nodes.size
  set { emulation with
    nodes := emulation.nodes.push { id, parents, kind, outShape := shape }
    dtypes := emulation.dtypes.push dtype
    primitives := emulation.primitives.push primitive
    constants := emulation.constants.push constant
    origin := emulation.origin.push (if copy then some site.node else none)
    computation := emulation.computation.push site.computation
    folded := emulation.folded.push folded
    site := emulation.site.push (site.difference.map (·, site.node)) }
  pure id

/-- The word of the integer `n` rounded once to `format`. -/
def integerWord (format : FloatFormat) (n : Int) : Nat :=
  FloatLib.Floats.ExecFloat.Binary.toNatBits (Reference.integerToBinary format .nearest n).1

/-- A constant of ones. -/
def ones (dtype : DType) (shape : Spec.Shape) (site : Site) : Build Nat := do
  let .binary format := dtype | throw s!"XLA emulation: {dtype} has no binary one"
  let value ← Reference.load dtype shape
    (Array.replicate (Spec.Shape.size shape) (integerWord format 1))
  emit (.const shape) #[] shape dtype site (constant := some value)

/-- Round node `id` to `target`. -/
def castTo (id : Nat) (target : DType) (site : Site) : Build Nat := do
  let some node := (← get).nodes[id]? | throw s!"XLA emulation: missing node {id}"
  emit (.custom "cast" [node.outShape] node.outShape) #[id] node.outShape target site
    (primitive := some (.cast .nearest))

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
becomes multiplication by its reciprocal rounded to the format; bfloat16 computes both in binary32
and rounds the product once. Otherwise, under `jax.jit`, the simplifier's rewrites
`(x / y) / z = x / (y * z)` and `x / (y / z) = (x * z) / y` apply in `rewritesIn` formats, then
`x / sqrt y = x * rsqrt y`, `x / rsqrt y = x * sqrt y` and `x / y ^ c = x * y ^ -c` in
`rootRewritesIn` formats, in that order, and the division a rewrite produces is computed by the
same rules.
-/
def emitDivision (fuel : Nat) (num den : Nat) (shape : Spec.Shape) (dtype : DType)
    (site : Site) : Build Nat := do
  let emulation ← get
  let divide : OpKind := .pointwise (.div .none)
  let constant := site.jitted && emulation.isFolded den && emulation.within site.computation den
  if emulation.broadcastIn site.computation den || constant then
    let site := { site with difference := some divisorReciprocal }
    if dtype == DType.bfloat16 then
      let wide ← castTo num DType.float32 site
      let divisor ← castTo den DType.float32 site
      let one ← ones DType.float32 shape site
      let reciprocal ← emit divide #[one, divisor] shape DType.float32 site
      let product ← emit .mulElem #[wide, reciprocal] shape DType.float32 site
      castTo product DType.bfloat16 site
    else
      let one ← ones dtype shape site
      let reciprocal ← emit divide #[one, den] shape dtype site
      emit .mulElem #[num, reciprocal] shape dtype site
  else
    match fuel with
    | 0 => emit divide #[num, den] shape dtype site
    | fuel + 1 =>
      unless site.jitted && rewritesIn dtype do
        return ← emit divide #[num, den] shape dtype site
      let rewritten := { site with difference := some rewrittenDivision }
      let quotient := emulation.operation? site.computation
      if let some (.pointwise (.div .none), #[x, y]) := quotient num then
        let product ← emit .mulElem #[y, den] shape dtype rewritten
        return ← emitDivision fuel x product shape dtype rewritten
      if let some (.pointwise (.div .none), #[y, z]) := quotient den then
        let product ← emit .mulElem #[num, z] shape dtype rewritten
        return ← emitDivision fuel product y shape dtype rewritten
      unless rootRewritesIn dtype do
        return ← emit divide #[num, den] shape dtype site
      match emulation.operation? site.computation den with
      | some (.sqrt, #[y]) =>
          let root ← emit (.pointwise .rsqrt) #[y] shape dtype rewritten
          emit .mulElem #[num, root] shape dtype rewritten
      | some (.pointwise .rsqrt, #[y]) =>
          let root ← emit .sqrt #[y] shape dtype rewritten
          emit .mulElem #[num, root] shape dtype rewritten
      | some (.pointwise .pow, #[y, c]) =>
          if emulation.isFolded c && emulation.within site.computation c then
            let negated ← emit (.pointwise .neg) #[c] shape dtype rewritten
            let raised ← emit (.pointwise .pow) #[y, negated] shape dtype rewritten
            emit .mulElem #[num, raised] shape dtype rewritten
          else emit divide #[num, den] shape dtype site
      | _ => emit divide #[num, den] shape dtype site

/-- The exponent `2`, `3` or `-1` that `jax.jit` computes as products or a reciprocal, when every
element of a binary16, binary32 or binary64 exponent is that integer. -/
def productExponent (dtype : DType) (exponent : Interpreter.Value Interpreter.reference) :
    Option Int := do
  guard (dtype == DType.float16 || dtype == DType.float32 || dtype == DType.float64)
  let .binary format := dtype | none
  let words ← (Reference.words exponent).toOption
  guard !words.isEmpty
  [2, 3, -1].find? fun n => words.all (· == integerWord format n)

/-- The binary16 operand of a binary32 node: a conversion of a binary16 value, as the JAX importer
writes the base of a binary16 power. -/
def halfSource (emulation : Emulation) (id : Nat) : Option Nat := do
  guard (emulation.dtypes[id]? == some DType.float32)
  let some (.cast _) := (emulation.primitives[id]?).join | none
  let source ← (emulation.nodes[id]?) >>= (·.parents[0]?)
  guard (emulation.dtypes[source]? == some DType.float16)
  pure source

/-- Build the emulated program from the IR program and its node values, returning the emulated
node of each IR node. Under `jax.jit` a constant exponent 2, 3 or -1 becomes products or a
reciprocal, each rounded to the format; XLA rounds every product of a binary16 base to binary16,
so the binary32 power that the JAX importer writes for it is computed in binary16 and widened. -/
def emulate (program : Program) (compilation : Compilation)
    (values : Array (Interpreter.Value Interpreter.reference)) :
    Except String (Emulation × Array Nat) := do
  let fuel := program.graph.nodes.size
  let steps : Build (Array Nat) := do
    let mut map : Array Nat := #[]
    for node in program.graph.nodes do
      let dtype ← program.dtype node.id
      let site : Site :=
        { node := node.id, computation := (compilation.computation[node.id]?).join
          jitted := compilation.jitted node.id }
      let parents := node.parents.map fun parent => map[parent]!
      let copy : Build Nat :=
        emit node.kind parents node.outShape dtype site (primitive := program.primitive? node.id)
          (constant := match node.kind with | .const _ => values[node.id]? | _ => none)
          (copy := true)
      let emulation ← get
      let id ← match node.kind, parents, dtype with
        | .pointwise (.div .none), #[num, den], .binary _ =>
            emitDivision fuel num den node.outShape dtype site
        | .pointwise .pow, #[base, exponent], _ =>
            let products := { site with difference := some powerProducts }
            let constant := site.jitted && emulation.isFolded exponent &&
              emulation.within site.computation exponent && !emulation.isFolded base
            let integer := if constant then
                (node.parents[1]? >>= (values[·]?)) >>= productExponent dtype
              else none
            let (operand, format) := match halfSource emulation base with
              | some source => (source, DType.float16)
              | none => (base, dtype)
            match integer with
            | none => copy
            | some n => do
                let result ← match n with
                  | 2 => emit .mulElem #[operand, operand] node.outShape format products
                  | 3 => do
                      let square ← emit .mulElem #[operand, operand] node.outShape format products
                      emit .mulElem #[operand, square] node.outShape format products
                  | _ => do
                      let one ← ones format node.outShape products
                      emitDivision fuel one operand node.outShape format products
                if format == dtype then pure result else castTo result dtype products
        | _, _, _ => copy
      map := map.push id
    pure map
  let (map, emulation) ← steps.run { jit := compilation.jit }
  pure (emulation, map)

/-- Whether XLA's CPU code flushes subnormal values of `dtype`: bfloat16, binary32 and binary64.
It computes binary16 in binary32, where binary16 subnormals are normal. -/
def flushes (dtype : DType) : Bool :=
  dtype == DType.bfloat16 || dtype == DType.float32 || dtype == DType.float64

/-- Operations whose subnormal operands and results XLA's CPU code flushes to zero, as measured.
Negation, `abs`, `sin`, layout operations and binary32 and bfloat16 `tanh` keep them. -/
def flushingOperation (kind : OpKind) (dtype : DType) : Bool :=
  match kind with
  | .add | .sub | .mulElem | .maxElem | .minElem | .sqrt | .log | .exp | .sigmoid | .sum
  | .reduceSum _ | .matmul | .pointwise (.div .none) | .pointwise .rsqrt => true
  | .tanh => dtype == DType.float64
  | _ => false

/-- Whether a node flushes its operands and its result. A node that `jax.jit` folds flushes
neither; a conversion from binary32 or bfloat16 to binary64 flushes its operand, and one from
binary64 to binary32 or bfloat16 its result. -/
def flushing (emulation : Emulation) (node : Node) : Bool × Bool :=
  if emulation.isFolded node.id then (false, false) else
  let dtype := emulation.dtypes[node.id]?.getD .bool
  match node.kind, (emulation.primitives[node.id]?).join with
  | .custom _ _ _, some (.cast _) =>
      let source := node.parents[0]? >>= (emulation.dtypes[·]?)
      (dtype == DType.float64 && (source == some DType.float32 || source == some DType.bfloat16),
        source == some DType.float64 && (dtype == DType.float32 || dtype == DType.bfloat16))
  | kind, _ =>
      let flush := flushes dtype && flushingOperation kind dtype
      (flush, flush)

/-- Replace the subnormal elements of a bfloat16, binary32 or binary64 value by zero of the same
sign, reporting whether any element changed. -/
def flushValue (value : Interpreter.Value Interpreter.reference) :
    Except String (Interpreter.Value Interpreter.reference × Bool) := do
  let .binary format := value.dtype | return (value, false)
  unless flushes value.dtype do return (value, false)
  let words ← Reference.words value
  let sign := 2 ^ (format.bitWidth - 1)
  let flushed := words.map fun word =>
    if (binaryWordClass format word).2 then word / sign * sign else word
  if flushed == words then return (value, false)
  return (← Reference.load value.dtype value.shape flushed, true)

/-- The IR payload, read through the origins of the emulated nodes. -/
def emulationPayload (emulation : Emulation) (payload : Interpreter.Payload Interpreter.reference) :
    Interpreter.Payload Interpreter.reference where
  constant? id := (emulation.constants[id]?).join
  homogeneous dtype :=
    letI := Interpreter.reference.storage dtype
    let original := payload.homogeneous dtype
    let origin (id : Nat) : Option Nat := (emulation.origin[id]?).join
    { custom? := fun id => origin id >>= original.custom?
      const? := fun id => origin id >>= original.const?
      linear? := fun id => origin id >>= original.linear?
      conv? := fun id => origin id >>= original.conv?
      batchNormEval? := fun id => origin id >>= original.batchNormEval?
      layerNorm? := fun id => origin id >>= original.layerNorm? }
  call := payload.call

/-- The emulated program and its interface, as a `Program`, with the payload read through it. -/
structure Emulated where
  /-- The documented behaviours applied to each node. -/
  emulation : Emulation
  /-- The emulated program. -/
  program : Program
  /-- Its payload. -/
  payload : Interpreter.Payload Interpreter.reference
  /-- The arguments. -/
  arguments : Array (Interpreter.Value Interpreter.reference)

/-- Evaluate one emulated node on `values`, flushing its operands and its result as `flushing`
says, and report whether flushing changed a word. -/
def Emulated.evalNode (emulated : Emulated)
    (values : Array (Interpreter.Value Interpreter.reference)) (node : Node) :
    Except String (Interpreter.Value Interpreter.reference × Bool) := do
  let (inputs, output) := flushing emulated.emulation node
  let mut operands := values
  let mut flushed := false
  if inputs then
    for parent in node.parents do
      let some value := values[parent]? | throw s!"XLA emulation: missing parent {parent}"
      let (value, touched) ← flushValue value
      if touched then
        operands := operands.set! parent value
        flushed := true
  let (value, _) ← Interpreter.reference.evalNode emulated.program emulated.payload
    emulated.arguments operands node
  let (value, touched) ← if output then flushValue value else pure (value, false)
  pure (value, flushed || touched)

/-- Run the emulated program, returning every node's value and whether flushing changed a word
of it. -/
def Emulated.run (emulated : Emulated) :
    Except String (Array (Interpreter.Value Interpreter.reference) × Array Bool) := do
  emulated.program.check
  let mut values : Array (Interpreter.Value Interpreter.reference) := #[]
  let mut changed : Array Bool := #[]
  for node in emulated.program.graph.nodes do
    let (value, touched) ← emulated.evalNode values node
    values := values.push value
    changed := changed.push touched
  pure (values, changed)

/-- A result that XLA's own function computes, and the window within which it counts as
reproduced. -/
structure Approximation where
  /-- XLA's function. -/
  behaviour : XlaBehaviour
  /-- The emulated `rsqrt` or `pow` node. -/
  source : Nat
  /-- The window, in units in the last place of the source node's format, or why there is none. -/
  ulps : Except String Nat
  /-- The nodes from the source to the result, in order, the result last. -/
  path : Array Nat

/-- Whether node `id` is an argument, reached through layout operations and conversions that do
not round. -/
def exactArgument (emulation : Emulation) (id : Nat) : Bool := Id.run do
  let mut current := id
  for _ in [:emulation.nodes.size] do
    let some node := emulation.nodes[current]? | return false
    match node.kind, node.parents[0]? with
    | .input, _ => return true
    | .broadcastTo _ _, some parent | .reshape _ _, some parent | .permute _, some parent
    | .transpose _ _, some parent | .flatten _, some parent | .detach, some parent =>
        current := parent
    | .custom _ _ _, some parent =>
        match (emulation.primitives[current]?).join, emulation.dtypes[current]?,
            emulation.dtypes[parent]? with
        | some (.cast _), some (.binary target), some (.binary source) =>
            if source.fracWidth ≤ target.fracWidth && source.expWidth ≤ target.expWidth then
              current := parent
            else return false
        | _, _, _ => return false
    | _, _ => return false
  return false

/-- XLA's own function computing result `id`: an `rsqrt` or `pow` node, reached through layout
operations, negation and conversions between binary formats, with its window when the node is of
binary32 or binary64. A power has a window only when both its operands are arguments
(`exactArgument`), so that XLA's function and the IR's receive the same values: a power with a
small exponent hides a large error in its base, and one with a large base a small error in its
exponent. -/
def approximation (emulation : Emulation) (id : Nat) : Option Approximation := Id.run do
  let mut current := id
  let mut path : Array Nat := #[]
  for _ in [:emulation.nodes.size] do
    let some node := emulation.nodes[current]? | return none
    let dtype := emulation.dtypes[current]?.getD .bool
    let measured (behaviour : XlaBehaviour) : Except String Nat :=
      match window behaviour dtype with
      | some ulps => .ok ulps
      | none => .error s!"it computes in {dtype.describe}, where no window was measured"
    match node.kind, node.parents[0]? with
    | .pointwise .rsqrt, _ =>
        return some { behaviour := reciprocalSquareRoot, source := current,
                      ulps := measured reciprocalSquareRoot, path := path.reverse }
    | .pointwise .pow, _ =>
        let ulps := if node.parents.all (exactArgument emulation) then measured power
          else .error "its operands are not both arguments of the program"
        return some { behaviour := power, source := current, ulps, path := path.reverse }
    | .broadcastTo _ _, some parent | .reshape _ _, some parent | .permute _, some parent
    | .transpose _ _, some parent | .flatten _, some parent | .detach, some parent
    | .pointwise .neg, some parent =>
        path := path.push current
        current := parent
    | .custom _ _ _, some parent =>
        match (emulation.primitives[current]?).join, emulation.dtypes[current]?,
            emulation.dtypes[parent]? with
        | some (.cast _), some (.binary _), some (.binary _) =>
            path := path.push current
            current := parent
        | _, _, _ => return none
    | _, _ => return none
  return none

/-- Whether a word of `format` has its sign bit set. -/
def signBit (format : FloatFormat) (word : Nat) : Bool :=
  word / 2 ^ (format.bitWidth - 1) % 2 == 1

/-- Whether a word of `format` is an infinity. -/
def isInfinite (format : FloatFormat) (word : Nat) : Bool :=
  Model.isInf (Model.ofNatBits word : Model format)

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

/-- The ends of the window around each element of the approximation's source node, carried to
the result through its path: rounding is monotone, and negation and layout operations move the
ends with the value. -/
def windowAt (emulated : Emulated) (values : Array (Interpreter.Value Interpreter.reference))
    (approximation : Approximation) : Except String (Option (Array Nat × Array Nat)) := do
  let .ok ulps := approximation.ulps | return none
  let some value := values[approximation.source]?
    | throw s!"XLA emulation: missing node {approximation.source}"
  let .binary format := value.dtype | return none
  let ends := (← Reference.words value).map (windowEnds format ulps)
  let mut low ← Reference.load value.dtype value.shape (ends.map (·.1))
  let mut high ← Reference.load value.dtype value.shape (ends.map (·.2))
  for id in approximation.path do
    let some node := emulated.program.graph.nodes[id]? | throw s!"XLA emulation: missing node {id}"
    let some parent := node.parents[0]? | throw s!"XLA emulation: node {id} has no operand"
    low := (← emulated.evalNode (values.set! parent low) node).1
    high := (← emulated.evalNode (values.set! parent high) node).1
  pure (some (← Reference.words low, ← Reference.words high))

/-- Whether the reference word `wanted` lies between the window's ends `low` and `high` at the
result and has the sign and the finiteness of the emulated word: a difference in the sign of zero
or between a finite value and an infinity is outside every window. -/
def inWindow (format : FloatFormat) (emulated wanted low high : Nat) : Bool :=
  if (binaryWordClass format emulated).1 || (binaryWordClass format wanted).1 then false
  else if signBit format wanted != signBit format emulated then false
  else if isInfinite format wanted != isInfinite format emulated then false
  else
    let sign := 2 ^ (format.bitWidth - 1)
    let ordinal (word : Nat) : Int := if word ≥ sign then -((word - sign : Nat) : Int) else word
    let (a, b) := (ordinal low, ordinal high)
    decide (min a b ≤ ordinal wanted) && decide (ordinal wanted ≤ max a b)

/-- The nodes that node `output` depends on, itself included. -/
def dependencies (nodes : Array Node) (output : Nat) : Array Bool := Id.run do
  let mut needed := (Array.replicate nodes.size false).set! output true
  for offset in [:nodes.size] do
    let id := nodes.size - 1 - offset
    unless needed[id]! do continue
    let some node := nodes[id]? | continue
    for parent in node.parents do
      needed := needed.set! parent true
  return needed

/-- A word in hexadecimal. -/
def hex (word : Nat) : String := "0x" ++ String.ofList (Nat.toDigits 16 word)

/-- The window used for a result: XLA's function, its width and format, and the differing
elements within it. -/
structure WindowUse where
  /-- XLA's function. -/
  behaviour : XlaBehaviour
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
  mismatch : Mismatch
  /-- Documented behaviours applied to nodes the result depends on, each with the IR nodes it
  rewrites, or for flushing the emulated nodes where it changed a word. -/
  applied : Array (XlaBehaviour × Nat)
  /-- Differing elements whose words the emulated program returns. -/
  reproduced : Nat
  /-- The window of XLA's own function computing the result, if one applies. -/
  window : Option WindowUse
  /-- XLA's own function computing the result when no window applies, and why. -/
  withheld : Option (XlaBehaviour × String) := none
  /-- Elements not reproduced, with the IR's, the emulation's and the reference's words. -/
  notReproduced : Array (Nat × Nat × Nat × Nat)

namespace Reproduction

/-- The behaviours the reproduction used: those applied, and XLA's function if its window
holds a differing element. -/
def used (entry : Reproduction) : Array XlaBehaviour :=
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
def windowText (use : WindowUse) : String :=
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
          {windowText use}"] else []
    | none => []
  if entry.notReproduced.isEmpty then
    counts ++ "; " ++ "; ".intercalate (reproduced ++ within)
  else
    let emulated := if entry.applied.isEmpty then [] else [s!"after emulating {entry.behaviours}"]
    let allowed := match entry.window, entry.withheld with
      | some use, _ => [s!"allowing {windowText use}"]
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
def reproduceResult (bitwise : Bool) (emulated : Emulated) (changed : Array Bool)
    (values : Array (Interpreter.Value Interpreter.reference)) (output index : Nat)
    (ir reference : Interpreter.Value Interpreter.reference) (mismatch : Mismatch) :
    Except String Reproduction := do
  let emulation := emulated.emulation
  let some value := values[output]? | throw s!"XLA emulation: missing result {index}"
  let irWords ← Reference.words ir
  let wanted ← Reference.words reference
  let words ← Reference.words value
  let approximated := approximation emulation output
  let ends ← match approximated with
    | some approximation => windowAt emulated values approximation
    | none => pure none
  let mut reproduced := 0
  let mut within := 0
  let mut notReproduced : Array (Nat × Nat × Nat × Nat) := #[]
  for element in mismatch.elements do
    let a := words[element]!
    let b := wanted[element]!
    let close := match ends, value.dtype with
      | some (low, high), .binary format => inWindow format a b low[element]! high[element]!
      | _, _ => false
    if agree bitwise value.dtype a b then reproduced := reproduced + 1
    else if close then within := within + 1
    else notReproduced := notReproduced.push (element, irWords[element]!, a, b)
  let needed := dependencies emulation.nodes output
  let mut sites : Array (XlaBehaviour × Nat) := #[]
  let mut flushed := 0
  for node in emulation.nodes do
    unless needed[node.id]! do continue
    if let some site := (emulation.site[node.id]?).join then
      unless sites.contains site do sites := sites.push site
    if changed[node.id]?.getD false then flushed := flushed + 1
  let mut applied : Array (XlaBehaviour × Nat) := #[]
  for (difference, _) in sites do
    unless applied.any (·.1 == difference) do
      applied := applied.push (difference, (sites.filter (·.1 == difference)).size)
  if flushed > 0 then applied := applied.push (subnormal, flushed)
  let window := match approximated, ends with
    | some approximation, some _ =>
        let dtype := emulation.dtypes[approximation.source]?.getD .bool
        some { behaviour := approximation.behaviour, ulps := approximation.ulps.toOption.getD 0,
               dtype, result := value.dtype, elements := within }
    | _, _ => none
  let withheld := match approximated with
    | some { behaviour, ulps := .error reason, .. } => some (behaviour, reason)
    | _ => none
  pure { result := index, mismatch, applied, reproduced, window, withheld, notReproduced }

/-- Run the emulation of XLA's CPU code that `compilation` describes and report what it
reproduces of each differing result. -/
def reproduce (bitwise : Bool) (compilation : Compilation) (program : Program)
    (payload : Interpreter.Payload Interpreter.reference)
    (arguments values results reference : Array (Interpreter.Value Interpreter.reference))
    (differing : Array (Nat × Mismatch)) : Except String (Array Reproduction) := do
  let (emulation, map) ← emulate program compilation values
  let emulated : Emulated :=
    { emulation, arguments, payload := emulationPayload emulation payload
      program := { graph := { nodes := emulation.nodes }, dtypes := emulation.dtypes
                   inputs := program.inputs.map (map[·]!), outputs := program.outputs.map (map[·]!)
                   primitives := emulation.primitives } }
  let (emulatedValues, changed) ← emulated.run
  differing.mapM fun (index, mismatch) => do
    let some output := emulated.program.outputs[index]? | throw s!"XLA emulation: no result {index}"
    let some ir := results[index]? | throw s!"XLA emulation: no IR result {index}"
    let some wanted := reference[index]? | throw s!"XLA emulation: no reference {index}"
    reproduceResult bitwise emulated changed emulatedValues output index ir wanted mismatch

/-- Compare the IR's results with the reference tensors in `field`, returning the references and
the elements that differ. -/
def mismatches (bitwise : Bool) (artifact : Json) (field : String)
    (results : Array (Interpreter.Value Interpreter.reference)) :
    Except String (Array (Interpreter.Value Interpreter.reference) × Array (Nat × Mismatch)) := do
  let reference ← (← (← artifact.getObjVal? field).getArr?).mapM value
  unless results.size == reference.size do
    throw s!"IR comparison: {field} has {reference.size} results, the program {results.size}"
  let found ← (results.zip reference).mapIdxM fun i (output, expected) =>
    match compareResult bitwise output expected with
    | .ok mismatch => pure mismatch
    | .error message => throw s!"IR comparison: result {i} has {message}"
  pure (reference, (found.mapIdx fun i m => (i, m)))

/-- The behaviours a set of reproductions used, each once, in order. -/
def usedBehaviours (entries : Array Reproduction) : Array XlaBehaviour :=
  entries.foldl (init := #[]) fun found entry =>
    entry.used.foldl (init := found) fun found behaviour =>
      if found.contains behaviour then found else found.push behaviour

/-- Their codes. -/
def behaviourCodes (entries : Array Reproduction) : Array String :=
  (usedBehaviours entries).map (·.code)

/-- What the primitive-by-primitive reference says about a mismatch with a `jax.jit` reference:
whether the IR agrees with it, or the emulation of XLA's CPU code without `jax.jit` reproduces it,
with a sentence and report entries. -/
def eagerCheck (bitwise : Bool) (artifact : Json) (program : Program)
    (payload : Interpreter.Payload Interpreter.reference)
    (arguments values results : Array (Interpreter.Value Interpreter.reference)) :
    Bool × String × Json :=
  let outcome : Except String (Bool × String × Json) := do
    let (reference, found) ← mismatches bitwise artifact "eager_reference_outputs" results
    let differing := found.filter (!·.2.elements.isEmpty)
    if differing.isEmpty then
      return (true, "the IR's results equal the primitive-by-primitive reference",
        Json.mkObj [("status", "agrees")])
    let compilation ← Compilation.read artifact (perPrimitive := true)
    let entries ← reproduce bitwise compilation program payload arguments values results
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

/-- The `compare` or `compare_bitwise` finding for computed results and every node value. Any
difference fails. For a Jaxpr artifact with a CPU reference, a failure whose differing elements
the emulation of XLA's CPU code reproduces carries the behaviours in `consistent_with`, as advice:
an importer error with the same effect cannot be ruled out. With a `jax.jit` reference, the
artifact's primitive-by-primitive reference, when it has one, must also agree with the IR or be
reproduced without `jax.jit`. -/
def compareFinding (name : String) (artifact : Json) (program : Program)
    (payload : Interpreter.Payload Interpreter.reference)
    (arguments values results : Array (Interpreter.Value Interpreter.reference)) : Bool × Json :=
  let bitwise := name == "compare_bitwise"
  match mismatches bitwise artifact "reference_outputs" results with
  | .error message => (false, finding name "failed" message)
  | .ok (reference, found) =>
      let nanWords := found.foldl (· + ·.2.nanWords) 0
      let differing := found.filter (!·.2.elements.isEmpty)
      let counts := "; ".intercalate (differing.toList.map fun (i, m) =>
        s!"result {i}: {m.elements.size} of {m.total} elements differ")
      let failure := "IR comparison: results differ in their interchange words: "
      let platform := artifact.getObjValD "reference_platform"
      if differing.isEmpty then
        let message :=
          if bitwise then "Interchange words, including NaN signs and payloads, agree for the \
            supplied arguments only."
          else s!"Interchange words agree for the supplied arguments only. Two NaN results \
            agree whatever their sign and payload ({nanWords} such elements); request \
            `compare_bitwise` to compare their words too."
        (true, finding name "compared" message)
      else if (artifact.getObjValD "frontend").getStr?.toOption != some "jaxpr" then
        (false, finding name "failed" (failure ++ counts))
      else if platform != .str "cpu" then
        (false, finding name "failed" s!"{failure}{counts}. The artifact does not record that \
          XLA's CPU backend computed the reference (reference_platform: {platform.compress}), so \
          XLA's CPU behaviour is not emulated.")
      else
        let outcome : Except String (Compilation × Array Reproduction) := do
          let compilation ← Compilation.read artifact
          let entries ← reproduce bitwise compilation program payload arguments values results
            reference differing
          pure (compilation, entries)
        match outcome with
        | .error message =>
            (false, finding name "failed" s!"{failure}{counts}; the emulation of XLA's CPU code \
              could not run: {message}")
        | .ok (compilation, entries) =>
            let report := ("results", toJson (entries.map Reproduction.toJson))
            let summary := "; ".intercalate (entries.toList.map Reproduction.describe)
            if !entries.all (·.notReproduced.isEmpty) then
              (false, finding name "failed" (failure ++ summary) [report])
            else
              let codes := behaviourCodes entries
              let descriptions := ", ".intercalate
                ((usedBehaviours entries).toList.map (·.description))
              let advice := s!"{failure}{summary}. This is consistent with XLA's documented CPU \
                behaviour ({descriptions}), but an importer error with the same effect on these \
                arguments cannot be ruled out, so the comparison fails."
              let hasEager := (artifact.getObjVal? "eager_reference_outputs").toOption.isSome
              if !compilation.usesJit then
                (false, finding name "failed" advice [("consistent_with", toJson codes), report])
              else if !hasEager then
                (false, finding name "failed" (advice ++ " No primitive-by-primitive reference \
                  was recorded to compare the IR with XLA's results without jax.jit's rewrites.")
                  [("consistent_with", toJson codes), report])
              else
                let (consistent, sentence, eager) :=
                  eagerCheck bitwise artifact program payload arguments values results
                if consistent then
                  (false, finding name "failed" (advice ++ s!" Without jax.jit, {sentence}.")
                    [("consistent_with", toJson codes), report, ("eager_reference", eager)])
                else
                  (false, finding name "failed" s!"{failure}{summary}. Emulating XLA's CPU code \
                    under jax.jit reproduces the reference, but {sentence}, so the mismatch is not \
                    attributed to XLA's documented behaviour."
                    [report, ("eager_reference", eager)])

/-- Serialize computed results for inspection without a lossy numerical display conversion. -/
def result (value : Interpreter.Value Interpreter.reference) : Except String Json := do
  pure <| Json.mkObj [
    ("element_type", toJson value.dtype.describe), ("dtype", toJson value.dtype),
    ("shape", toJson value.shape), ("bits", toJson (← Reference.words value))]

end Internal

/-- Run requested checks, returning explicit failures and unsupported requests as report entries.

`execute`, `compare` and `compare_bitwise` concern only `arguments` stored in the artifact. A
comparison uses complete binary words, except that `compare` lets two NaNs agree whatever their
sign and payload. It can expose legitimate differences in reduction or elementary-function
implementations; a mismatch alone does not identify a bug, and it always fails. For a Jaxpr
artifact, the failure lists in `consistent_with` the documented behaviours of XLA's CPU code whose
emulation reproduces every differing element (see the module documentation).
-/
def check (artifact : Json) (requests : Array String := #["structure", "types"]) : Json := Id.run do
  let mut findings : Array Json := #[]
  let mut ok := true
  let mut outputs : Json := .null
  let mut conversionFlags : Json := .null
  let parsed := Import.IR.parseProgram artifact
  let execution : Except String _ := do
    unless requests.any (fun request =>
        request == "execute" || request == "compare" || request == "compare_bitwise") do
      throw "Execution was not requested."
    let program ← parsed
    let payload ← Import.IR.parsePayload Interpreter.reference Reference.load artifact
    let arguments ← (← (← artifact.getObjVal? "arguments").getArr?).mapM Internal.value
    let (values, flags) ← Interpreter.reference.run program payload arguments
    let results ← program.outputs.mapM fun id =>
      match values[id]? with
      | some value => pure value
      | none => throw s!"IR interpreter: missing result node {id}"
    pure (program, payload, arguments, values, results, flags)
  if let .ok (_, _, _, _, results, flags) := execution then
    match results.mapM Internal.result with
    | .ok results => outputs := toJson results
    | .error _ => pure ()
    conversionFlags := Json.mkObj [
      ("inexact", toJson flags.inexact), ("overflow", toJson flags.overflow),
      ("underflow", toJson flags.underflow), ("invalid", toJson flags.invalid)]
  for request in requests do
    let (passed, entry) :=
      match request with
      | "structure" | "types" =>
          match parsed with
          | .ok _ => (true, Internal.finding request "checked"
              "Declared dependencies, shapes, types and casts are consistent.")
          | .error message => (false, Internal.finding request "failed" message)
      | "execute" | "compare" | "compare_bitwise" =>
          match execution with
          | .error message => (false, Internal.finding request "failed" message)
          | .ok (program, payload, arguments, values, results, _) =>
              if request == "execute" then
                (true, Internal.finding request "executed"
                  "CPU reference execution completed for the supplied arguments.")
              else
                Internal.compareFinding request artifact program payload arguments values results
      | _ =>
          (false, Internal.finding request "unsupported"
            s!"No checker for `{request}`; no proof was produced.")
    ok := ok && passed
    findings := findings.push entry
  if requests.isEmpty then
    ok := false
    findings := #[Internal.finding "request" "failed" "At least one check must be requested."]
  return Json.mkObj [
    ("format", toJson "torchlean.verification.v1"), ("ok", toJson ok),
    ("checks", toJson findings), ("outputs", outputs), ("conversion_flags", conversionFlags),
    ("capture_constraints", (artifact.getObjVal? "capture_constraints").toOption.getD (.arr #[])),
    ("capture_constraints_status", toJson "not_checked"),
    ("frontend_equivalence", toJson "not_checked"),
    ("frontend_device_layout_constraints", toJson "not_checked"),
    ("external_kernels", toJson "not_checked")]

/-- CLI: check an artifact as JSON, or emit the existing PyTorch capture adapter. -/
def run (arguments : List String) : IO Unit := do
  match arguments with
  | ["--export-pytorch", path] =>
      IO.FS.writeFile path Export.PyTorch.TorchExport.script
  | path :: requests =>
      let artifact ← IO.ofExcept (Json.parse (← IO.FS.readFile path))
      let requests := if requests.isEmpty then #["structure", "types"] else requests.toArray
      let report := check artifact requests
      IO.println report.compress
      unless (← IO.ofExcept ((← IO.ofExcept (report.getObjVal? "ok")).getBool?)) do
        throw (IO.userError "one or more IR checks failed; see the JSON report")
  | [] => throw (IO.userError
      "usage: verify ir <artifact.json> [structure|types|execute|compare|compare_bitwise]")

end NN.Verification.IR
