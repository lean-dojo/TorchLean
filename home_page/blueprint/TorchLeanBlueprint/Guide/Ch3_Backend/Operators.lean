import VersoManual
-- The operator table with its rules, invocation and codec, the CUDA table that keys on it, and
-- the bracket syntax of the example tensors.
import TorchLean.Tensor.Op.Run
import TorchLean.Tensor.Op.Table.Elementwise
import TorchLean.Tensor.Op.Table.Word
import TorchLean.Tensor.Op.Codec
import TorchLean.Tensor.Literal
import TorchLean.NN.LibTorch.Kernel
-- The certificates `#op_certs` prints for `tanh` and `linear`.
import TorchLean.NN.Op.Certificates
import TorchLean.NN.Op.Pointwise
import TorchLeanBlueprint.Bib
-- `TorchLeanBlueprint.OpTable` brings in every registered certificate and the definitions its
-- tables read.
import TorchLeanBlueprint.OpTable
import TorchLeanBlueprint.Roles

open Verso.Genre Manual
open Verso.Genre.Manual.InlineLean
open TorchLean
open Lean.Elab.Tactic.GuardMsgs.WhitespaceMode (lax)

/-! Guide page: the operator table, with its rules, CUDA entries and certificates. -/

#doc (Manual) "Registered Operators" =>
%%%
tag := "registered-operators"
file := "The-Operator-Table"
%%%

The value of `sqrt(-1)` distinguishes two graph meanings. IEEE square root returns a NaN;
`sqrt(max x 0)` returns zero at that input. A checker that clamps while the model executes the raw
operation would verify a different function. `Op.sqrt .raw` therefore names the IEEE square root,
and the program must request the clamped form explicitly.

`TorchLean.Op` names the operations. Except for `custom`, `Op.rule?` gives each operation a rule
for its forward values and pullbacks at a chosen scalar type. A custom node instead carries a
checked program that supplies its local meaning. The tape and graph record those same operator
names, so the checker can work with the same Lean rules. The CUDA backend uses the names to
select its implementations.

The table lives in {src "TorchLean/Tensor/Op/Table.lean"}[`TorchLean/Tensor/Op/Table.lean`], its
rules in {src "TorchLean/Tensor/Op/Table/Rule.lean"}[`TorchLean/Tensor/Op/Table/Rule.lean`] and the
equation lemmas of each operator family beside them. Lean elaborates every Lean block and computes
the operator counts and vocabulary tables from those definitions in the same build.

# One Constructor Per Operator

Sorting needs to give us both the sorted values and the positions they came from. ReLU only
returns its values. We want Lean to see that difference at the operator itself, because a
parent node needs to know what it can read. Each constructor in `TorchLean.Op` gives its
attributes and lists the ports for its inputs and outputs.

```lean (name := regPorts)
-- An elementwise operator, a sort with an index output, a
-- selection with a Boolean condition, and a clamp whose
-- bounds are frozen scalars.
#check (Op.relu (s := [3]))
#check (Op.sort (s := [3]) ⟨0, by decide⟩ false)
#check (Op.select (s := [2]))
#check (Op.clamp (s := [2]))
```

```leanOutput regPorts (whitespace := lax)
Op.relu : Op [Port.value [3]] [Port.value [3]]
```

```leanOutput regPorts (whitespace := lax)
Op.sort ⟨0, ⋯⟩ false :
  Op [Op.Element.port Port.Kind.value Op.Element.scalar [3]]
    [Op.Element.port Port.Kind.value Op.Element.scalar [3],
      Port.index [Shape.extent [3] ⟨0, ⋯⟩] [3]]
```

```leanOutput regPorts (whitespace := lax)
Op.select : Op
  [Port.bool [2], Op.Element.port Port.Kind.value Op.Element.scalar [2],
    Op.Element.port Port.Kind.value Op.Element.scalar [2]]
  [Op.Element.port Port.Kind.value Op.Element.scalar [2]]
```

```leanOutput regPorts (whitespace := lax)
Op.clamp : Op [Port.value [2], Port.frozen [], Port.frozen []]
  [Port.value [2]]
```

Shapes are attributes, so `Op.relu (s := [3])` and `Op.relu (s := [4])` are different operator
values, as two PyTorch calls with different sizes are different nodes of an exported graph. Here
the ascending `sort` along axis `0` returns the sorted values and, at its second output, an index
tensor whose entries are indices below the extent of that axis. `select` is PyTorch's
`torch.where`, with its condition at a Boolean port, and the ports of its data are those of its
element argument (described in `Ports And Element Sorts`). Both bounds of `clamp` are frozen scalar
inputs. An operator value doesn't know the scalar type, so a scalar attribute such as an epsilon,
a slope or a clamp
bound travels as data, at a port that never receives a gradient.

The table has {opCount}[] constructors, one for each operator. Operators that share a shape rule
are separate constructors when they compute different functions: `Op.max` returns the IEEE 754-2019
maximum, which propagates a NaN, and `Op.fmax` the maximum that ignores a NaN. Likewise
`Op.crossEntropy` takes logits and `Op.crossEntropyProbabilities` takes probabilities. A mode of
one operation is an attribute. `Op.div` takes `torch.div`'s `rounding_mode`, and `Op.sqrt` at
`.raw` (the default) is the IEEE square root, a NaN below zero, while at `.clampNegative` it's the
root of `max x 0`. Where neither operand of `Op.max` is larger (a NaN included), each receives half
of the gradient. PyTorch's `maximum` gives both the whole gradient at a NaN.

## Ports And Element Sorts

A port (`TorchLean.Port`) is an element sort and a shape, and the element sorts are:

```lean (name := regDtype)
#print Port.Kind
```

```leanOutput regDtype (whitespace := lax)
inductive TorchLean.Port.Kind : Type
number of parameters: 0
constructors:
TorchLean.Port.Kind.value : Port.Kind
TorchLean.Port.Kind.frozen : Port.Kind
TorchLean.Port.Kind.detached : Port.Kind
TorchLean.Port.Kind.index : List ℕ → Port.Kind
TorchLean.Port.Kind.bool : Port.Kind
TorchLean.Port.Kind.integer : Bool → ℕ → Port.Kind
```

A `value` port holds tensors of the model's scalar and receives a gradient when its parent needs
one. A `frozen` port holds the scalar without a gradient: an attribute, or a value the operator
reads without differentiating it, such as the operands of a comparison. A `detached` port reads a
recorded node without sending it a contribution, as PyTorch's `detach` does for the teacher of a
distillation loss. An `index` port holds index tuples below the listed bounds, a `bool` port
Booleans, and an `integer signed width` port words of `width` bits of an integer type, two's
complement when `signed`. Index, Boolean and integer ports let an operator take an index tensor
(`gather`, `scatter`), a mask (`select`) or integer words (the bitwise operators), or return one
(`sort`, `topk`, the comparisons, `encodeIndex`). A list of scalar tensor shapes can't say that.

For `reshape`, `permute`, `broadcast`, `flip`, `slice`, `pad`, `cat`, `gather`, `scatter`, `select`
and `sort` (`Op.Element`), and for `add`, `sub`, `mul`, `div`, `neg`, `pow`, `remainder`, `max`,
`min` and the comparisons (`Op.NumericElement`), the element argument defaults to `.scalar`, the
model's scalar. At that default, `Op.Element.port` gives the scalar's port of the operand's role:
the data ports of `select` in `regPorts` are value ports. At `.bool` or `.integer signed width` the
operator
moves Booleans or integer words as it moves scalars, or computes PyTorch's integer meaning with the
functions of `TorchLean.Word` (wrapping `+`, `-`, `*` and negation, the order of the type), without
gradients. An integer division or remainder fails its check at a zero divisor, an integer true
division (`div .none`) has no rule, and a Boolean scatter has a rule only as a replacement. An
ascending `sort` puts `false` before `true` (`Bool.toNat`) and orders integer words by their values
(`Word.toInteger`), signed or unsigned as the element type says.

For a list of ports, the values are nested pairs ending in `Unit`, so the inputs of `Op.mul` are
`(x, y, ())`. Tangents and cotangents exist for value ports only: every other port has `Unit`
there. Elementwise operators take equal shapes, and broadcasting is the separate operator
`broadcast`: where PyTorch broadcasts the operands of `add` implicitly, the capture writes the
broadcast as its own node.

# The Rule Of An Operator


Once an operator's ports line up, the run still needs to compute the output and, in
training, turn incoming sensitivities into input gradients. Those calculations live
together in the rule selected by `Op.rule? op` for the scalar type. A rule supplies the
following pieces.

- `forward`, from the values of the input ports to the values of the output ports
- `backward`, the dense pullback, a gradient for every value input
- `prepare`, the node a recorder builds once from the inputs and the runtime flags of the inputs
  that need a gradient, whose values and pullback share what the forward pass computed (the weights
  of `attentionFromScores`, the output of `logSoftmax`)
- `validate`, a check of the input values: the smooth max pool rejects a zero or nonfinite `beta`,
  an integer `div` with a rounding mode and an integer `remainder` a zero divisor, `encodeIndex` a
  component its words cannot hold and `decodeIndex` a word outside its bound
- the order in which a recorder emits the contributions, an alias rule for two inputs that name one
  node, and a forward-mode rule where one is registered

In every rule, the fields `prepare_value` and `prepare_eq` state that the prepared values are the
forward values and that the prepared pullback is the dense one at each active value input and
nothing at the others. A recorder calls `Op.Rule.check`, which runs `validate` and then `prepare`,
and never `forward`, so a value the pullback needs is computed once.

Each arm of `Op.rule?` is restated by an equation lemma named after its constructor, `Op.rule?_<op>`
(the two forms of `cast` have `Op.rule?_cast_bool` and `Op.rule?_cast_index`). So a proof about one
operator rewrites with its lemma instead of unfolding the whole table:

```lean (name := regSpecRelu)
#check @Op.rule?_relu
```

```leanOutput regSpecRelu (whitespace := lax)
@Op.rule?_relu : ∀ {α : Type} [inst : Storage α] [inst_1 : Context α]
  [inst_2 : Op.Capabilities α] {s : Shape},
  Op.relu.rule? =
    some
      { forward := fun xs => (Tensor.map Activation.relu xs.1, ()),
        backward := fun xs g =>
          (Tensor.map Activation.reluDeriv xs.1 * g.1, ()),
        prepare_value := ⋯, prepare_eq := ⋯, order_perm := ⋯,
        jvp? := some fun xs dxs =>
          (Tensor.map Activation.reluDeriv xs.1 * dxs.1, ()) }
```

A rule is an ordinary Lean value, and running its forward map is evaluation:

```lean (name := regForward)
-- The forward map of ReLU at Float, on a vector with a
-- negative entry and a zero.
#eval
  let x : Tensor Float [3] := [-1.5, 0, 2]
  (Op.rule? (α := Float) (Op.relu (s := [3]))).map fun r =>
    (r.forward (x, ())).1
```

```leanOutput regForward (whitespace := lax)
some [0.000000, 0.000000, 2.000000]
```

## The Same Operator at Two Precisions

An operator can have the same shape and still give different results when we change its scalar.
At $`2^{25}`, the next larger binary32 value is four units away. Adding one rounds back
to $`2^{25}`. Binary64 can keep the one. The two calls request the same `Op.add` at both scalar
types and run each rule on the same one-entry tensors.

```lean (name := regPrecision)
#eval
  let x : Tensor Float32 [1] := [33554432]
  let y : Tensor Float32 [1] := [1]
  let rule := Op.rule? (α := Float32) (Op.add (s := [1]))
  rule.map fun r => (r.forward (x, y, ())).1

#eval
  let x : Tensor Float [1] := [33554432]
  let y : Tensor Float [1] := [1]
  let rule := Op.rule? (α := Float) (Op.add (s := [1]))
  rule.map fun r => (r.forward (x, y, ())).1
```

```leanOutput regPrecision (whitespace := lax)
some [33554432.000000]
```

```leanOutput regPrecision (whitespace := lax)
some [33554433.000000]
```

When you read a numerical claim about a graph, check the scalar type and where conversions happen.
The shapes alone can't tell you which of these values to expect.

## Prepared Nodes And Runtime Flags

PyTorch's backward formulas receive `needs_input_grad` and can skip the products no input needs.
A TorchLean rule receives the same information through `prepare`. Here the product `x * y` is
prepared with only `x` needing a gradient, then pulled back along the cotangent `[1, 1]`:

```lean (name := regPrepared)
-- Prepare the product with the flags (true, false) and pull
-- back one cotangent.
#eval (Op.rule? (α := Float) (Op.mul (s := [2]))).map
  fun r =>
    let x : Tensor Float [2] := [3, -1]
    let y : Tensor Float [2] := [2, 5]
    let g : Tensor Float [2] := [1, 1]
    let p := r.prepare (x, y, ()) (true, false, ())
    (p.value.1, p.pullback (g, ()))
```

```leanOutput regPrepared (whitespace := lax)
some ([6.000000, -5.000000], some [2.000000, 5.000000], none, ())
```

The prepared value is the elementwise product, and the pullback sends `g * y = [2, 5]` to `x` and
`none` to `y`, whose flag is off. `TorchLean.Tape.apply` passes the parents' flags the same
way, so on the CPU tape only parents that need a gradient receive a contribution.

## Scalar Capabilities

Some operators need an operation that not every scalar type has. `Rounding` gives the integral
roundings, the rounded divisions, `fmod` and `remainder`. `Erf` gives the error function (for `erf`
and the exact GELU), and `InverseTrig` the inverse trigonometric functions `atan`, `asin`, `acos`
and `atan2`. `Cbrt` gives the cube root, `NextAfter` C's `nextafter` and `Bits` the interchange word
of a binary format (for `bitcast`). `Special` gives the special functions of `torch.special`, of one
argument and of two, and the regularized incomplete beta function of three (`lax.betainc`; PyTorch
has no such function), with their partial derivatives of every order (for `special`, whose orders of
`digamma` are `polygamma` and whose order `[1, 0]` of `igamma` is `igamma_grad_a`). Only the reals
and their dual numbers have `Special`, so `special` is specified over the reals only: it has no
rule at the floating-point scalars, executing it in binary32 or binary64 is refused, and none of
its roundoff analyses is better than unsupported. Its derivative certificate needs the inputs in
the domain `SpecialFn.smoothDomain` on which `SpecialFn.contDiffAt_real` proves the function smooth:
the points with both arguments positive for `igamma` and `igammac`, the points `(s, q)` with
`s > 1` and `q > 0` for `zeta`, the points `(a, b, x)` with `a` and `b` positive and `x ∈ (0, 1)`
for `betainc`, the points off the poles for `lgamma` and `digamma`, the open intervals of `erfinv`
and `ndtri`, the points off `0` for `i0e` and `i1e`, and every point for the others.
`Frexp` gives C's `frexp` and `ldexp` (for `frexp`), which the reals have, and `Fma` the fused
multiply-add rounded once (for `fma`), which the reals, the configured binary formats, `Float` and
`Float32` have.
`ComplexFunctions` gives the complex functions of `jax.lax`, the principal power and their
derivatives on pairs of parts (for `complex` and `complexPow`): `ComplexFn.value` and Mathlib's
`z ^ w` over the reals, and FloatLib's `ExecComplex` functions and power, each part rounded once
(the power where FloatLib certifies it), in the configured binary formats. `Op.Capabilities α`
bundles these ten capabilities, found by instance synthesis. An operator that needs a missing
capability has no rule at that scalar:

```lean (name := regCapabilities)
-- Float has the integral roundings and the inverse
-- trigonometric functions, but no error function.
#eval ((Op.rule? (α := Float) (Op.floor (s := [2]))).isSome,
  (Op.rule? (α := Float) (Op.atan2 (s := [2]))).isSome,
  (Op.rule? (α := Float) (Op.erf (s := [2]))).isSome)
```

```leanOutput regCapabilities (whitespace := lax)
(true, true, false)
```

For `erf` at `Float`, the table returns `none` instead of substituting an approximation. `Float`
has `Rounding`, `InverseTrig` and `Fma` but no `Erf`, `Cbrt`, `NextAfter`, `Bits`, `Special`,
`Frexp` or `ComplexFunctions`. The configured binary formats of FloatLib have every capability but
`Erf` and `Special`, and `Frexp` only in an IEEE format whose exponent bias is at least two and
whose exponents of finite nonzero values fit the int32 word `frexp` writes
(`configuredBinaryFrexpCapability`), as in binary16, bfloat16, binary32, binary64 and float8
`e5m2`, but not in the float8 formats `e4m3`, `e4m3fnuz` and `e5m2fnuz`, whose encodings are not
IEEE. `Op.need` names the capability an operator's rule reads, and `Op.isSome_rule?_iff` proves
that an operator for which `Op.hasRule` holds has a rule at a scalar exactly when the scalar
supplies that capability. `Op.needsCapability` names the operators whose rule reads a capability,
and `Op.rule?_capabilities` proves that every other rule is the same for all capability instances.

# Checking And Invoking

`Op.check` checks the attributes that the types do not, such as the window of a pooling (one
parameter per pooled axis, positive kernel extents and strides) and distinct axes for `reduce` and
`mean`:

```lean (name := regCheck)
-- A max pool with a zero kernel extent, and a sum over a
-- repeated axis.
#eval (Op.pool (s := [1, 4]) .max [⟨1, by decide⟩]
  [0] [1] [0] [0] [1] false).check
#eval (Op.reduce (s := [2, 3]) .sum
  [⟨1, by decide⟩, ⟨1, by decide⟩] false
  (by intro h; cases h <;> contradiction)).check
```

```leanOutput regCheck (whitespace := lax)
Except.error "autograd: max_pool requires kernel_size > 0 on every pooled axis"
```

```leanOutput regCheck (whitespace := lax)
Except.error "reduce requires distinct axes"
```

`Op.invoke` runs one operator on erased values. An erased value, `Op.Value α`, is a port paired
with a tensor at that port. An invocation checks in stages: the attributes, each input against its
port, the rule's value-domain check, then the prepared node, whose values it returns. A sort
returns two values, the second at an index port, and an input at the wrong shape is refused before
any arithmetic runs:

```lean (name := regInvoke)
-- One input vector, shown through a printer of erased
-- values.
def regInput : List (Op.Value Float) :=
  [⟨.value [3], ([-1.5, 0, 2] : Tensor Float [3])⟩]

def regShow (r : Except String (List (Op.Value Float))) :
    Except String (List String) :=
  r.map fun vs => vs.map fun
    | ⟨⟨.value, _⟩, x⟩ => s!"{x}"
    | ⟨p, _⟩ => s!"a value at {repr p}"

#eval regShow
  (Op.invoke (Op.relu (s := [3])) (.ok regInput))
#eval regShow
  (Op.invoke (Op.sort (s := [3]) ⟨0, by decide⟩ false)
    (.ok regInput))
#eval regShow
  (Op.invoke (Op.relu (s := [2])) (.ok regInput))
```

```leanOutput regInvoke (whitespace := lax)
Except.ok ["[0.000000, 0.000000, 2.000000]"]
```

```leanOutput regInvoke (whitespace := lax)
Except.ok ["[-1.500000, 0.000000, 2.000000]",
  "a value at { kind := TorchLean.Port.Kind.index [3], shape := [3] }"]
```

```leanOutput regInvoke (whitespace := lax)
Except.error "op: an input of port { kind := TorchLean.Port.Kind.value, shape := [3] } where
  { kind := TorchLean.Port.Kind.value, shape := [2] } is expected"
```

`TorchLean.IR.Graph.run` evaluates a straight-line list of such nodes forward with the same
checks. It's the forward half of a recorder, while the reverse sweep (with its gradient checks)
belongs to the tape.
`Op.invoke_eq_ok` states what a successful invocation returns: the rule's forward values on the
inputs read at their ports.

A custom node is the one constructor without an arm. `Op.custom` carries a name and the shapes of
its inputs and output, and the node's checked program lives in its graph's payload, not in the
operator value. `Op.customRule` builds the rule from that program: its forward map evaluates the
program, and its pullback is the one the caller supplies. That pullback is an executable policy,
with no claim that it's a derivative of the program.

# Operator Certificates

A theorem about an operator's arm can be registered as a certificate of that operator with the
attribute `@[op_cert Op.<operator> <kind>]`, followed by the input positions it concerns when the
others are held fixed. The kinds keep apart what the theorems state:

- `algebraic`: the pullback is the adjoint of a selected tangent map, with no claim that the map is
  a derivative.
- `derivative`: the tangent map is the Fréchet derivative and the pullback its adjoint, on the
  domain and shapes the theorem states.
- `convention`: a rule fixed by an automatic-differentiation convention, such as `detach` or a
  subgradient.
- `prepared`: a prepared pullback returns the dense one, beyond the `prepare_eq` field.
- `roundoff`: a floating-point error bound of a route.
- `enclosure`: the interval transfer of the graph numerical certificate encloses the operator's real
  forward value.

On registration, the attribute checks that the declaration is a theorem whose statement mentions
the operator's arm of `Op.rule?`, and, when the operator's input list reduces to a literal list,
that every listed position is an input. For `derivative` it checks the statement against one
proposition, `DerivCert f x δ v`: the operator's forward `f` in the listed inputs is
differentiable at the point `x`, and the stored backward `v` of `δ` is the adjoint of its
derivative. For the other kinds it doesn't check that the statement says what its kind says.
`#op_certs` prints the registrations of one operator:

```lean (name := regCerts)
#op_certs Op.tanh
#op_certs Op.linear
#op_certs Op.cat
```

```leanOutput regCerts (whitespace := lax)
TorchLean.Op.tanh: [algebraic TorchLean.AdjointOp.tanh_op,
  derivative TorchLean.Op.tanh_backward_eq_adjoint]
```

```leanOutput regCerts (whitespace := lax)
TorchLean.Op.linear: [algebraic TorchLean.AdjointOp.linear_op (inputs [2])]
```

```leanOutput regCerts (whitespace := lax)
TorchLean.Op.cat: no certificates
```

Of these, the linear certificate concerns input `2` (the layer's input), with the weight and bias
held fixed. Across the table, {opCertKinds}[] certificates are registered. The gradient
certificates are in {srcDir "TorchLean/NN/Op"}[`TorchLean/NN/Op`] and the enclosures
in {src "TorchLean/Verify/NumericalCertificate/Enclosure.lean"}[the graph
numerical certificate]. {ref "autograd-proofs"}[Autograd Proofs] explains the adjoint laws and
Fréchet derivatives the gradient kinds name.

# The Wire Codec

An operator value carries shapes, axes as `Fin` values and proofs, which a file cannot hold.
`Op.encode` writes an erased operator, `Σ ins outs, Op ins outs`, as a wire record `Op.Wire`: the
format version, the constructor's name as its tag, and its implicit shapes and attributes in binder
order, one `Op.Field` each:

```lean (name := regEncode)
#eval Op.encode
  ⟨_, _, Op.softmax (s := [2, 3]) ⟨1, by decide⟩⟩
```

```leanOutput regEncode (whitespace := lax)
{ version := 5, tag := "softmax",
  fields := [TorchLean.Op.Field.shape [2, 3], TorchLean.Op.Field.nat 1] }
```

Proof attributes are not sent. `Op.decode` checks every attribute and decides every proof
attribute, so a record whose axis is outside the rank names no operator:

```lean (name := regDecode)
-- Softmax of a rank-2 shape along axis 1, then axis 2.
#eval
  let wire (axis : Nat) : Op.Wire :=
    { version := Op.Wire.formatVersion,
      tag := "softmax"
      fields := [.shape [2, 3], .nat axis] }
  ((Op.decode (wire 1)).isSome, (Op.decode (wire 2)).isSome)
```

```leanOutput regDecode (whitespace := lax)
(true, false)
```

Decoding inverts encoding for every constructor:

```lean (name := regDecodeEncode)
#check @Op.decode_encode
```

```leanOutput regDecodeEncode (whitespace := lax)
Op.decode_encode : ∀ (op : (ins : List Port) × (outs : List Port) × Op ins outs),
  Op.decode (Op.encode op) = some op
```

Equality of erased operators is decided through the records (`Op.instDecidableEqErased`), and the
ports are recomputed from the attributes, so two operators at different shapes have different
records. `Op.decode` doesn't run `Op.check`: a decoded pooling with an ill-formed window is an
operator value, which a recorder rejects before reading its input. A record names the operator and
its attributes, not what it computes at a scalar type, so an artifact whose results must be
reproduced also names the TorchLean revision that evaluates it.

# One Table For Every Frontend

Each part of TorchLean that runs, records or plans an operation takes the operator itself:

:::table +header
*
  * Consumer
  * Through the table
*
  * CPU tape
  * `TorchLean.Tape.apply` records an operator through `TorchLean.Op.rule?`, with the attribute and
    value checks, the parents' flags and the alias rule
*
  * CUDA tape
  * `LibTorch.Kernel.ofOp op` is the CUDA implementation of `op` or `none`, and
    `LibTorch.Tape.apply` records through it
*
  * Eager session
  * `Session.apply` records an operator whose first output is a value port on the session's
    CPU or CUDA tape, under the capsule the profile selects, and returns a reference to that
    output
*
  * Verifier recorder
  * `TorchLean.IR.record` appends an operator's steps to the typed graph
    `TorchLean.IR.OpGraph`, for the operators of the verifier fragment
*
  * Graph IR
  * `TorchLean.IR.Graph` is a graph whose nodes are registered operators, and `Graph.eval` invokes
    each one with `Op.invoke`, a custom operator by its program from the payload
*
  * Artifacts
  * a `torchlean.ir` artifact (`TorchLean.IR.Artifact`) is a typed program with the words of its
    constants, and it stores each operator as its `Op.encode` record, which `Artifact.ofJson`
    decodes with `Op.decode`
*
  * Backend planning
  * a capsule's `op` field is the operator's `Op.Wire` record
*
  * Roundoff analysis
  * `#roundoff_coverage` analyzes the routes of an operator's arm of `Op.rule?`: its forward
    values, tangents and gradients
:::

Model code written against the `Ops` interface (`TorchLean.Ops`) reaches the table
through its instances: the eager instance records each operation with `Session.apply`, and the
recorder instance behind `IR.lowerForward` appends it with `record`. That recorder covers the
verifier fragment only. Most operators are one step of themselves, but `scale`, `clamp`, attention,
multi-head attention over leading axes and coordinate selection are recorded through several steps,
and `conv` and `pool` are taken only in the layouts and windows the recorder supports. The recorder
refuses random operators, GELU, training batch normalization and every operator that isn't
admissible (`Op.Admissible`: its attribute check passes, its rule checks no input value and reads
no optional capability, and it has a rule at every scalar). `IR.OpGraph.eval_toOp` proves that the
graph of registered operators of a typed graph returns the typed values at its outputs, for every
input. No theorem
relates the recorded graph to the source program.

{ref "runtime-autograd"}[Runtime and Autograd] builds a CPU tape with `Tape.apply` by hand,
{ref "gpu-and-cuda"}[GPU and CUDA] follows the CUDA entries, and {ref "graphs-and-ir"}[Graph IR]
imports a graph into operators and runs it.

The tables of the two tapes need not agree on which operators they cover:

```lean (name := regKernels)
-- The CUDA table has an entry for ReLU and none for
-- `Op.fmax`.
#eval
  ((TorchLean.LibTorch.Kernel.ofOp
      (Op.relu (s := [2]))).isSome,
    (TorchLean.LibTorch.Kernel.ofOp
      (Op.fmax (s := [2]))).isSome)
```

```leanOutput regKernels (whitespace := lax)
(true, false)
```

At a scalar with the capabilities it needs, every operator for which `Op.hasRule` holds has a CPU
rule (`Op.isSome_rule?_iff`), but an eager CUDA session refuses an operator without an entry
(`Session.cudaEntry`). A CUDA entry implements its operator under a recorded contract, whose
evidence doesn't make the foreign implementation a proved refinement. So an entry need not return
the rule's values at every input: {ref "gpu-and-cuda"}[GPU and CUDA] describes the contracts and
their evidence.

# Operator Specifications And Coverage

The following fourteen groups cover all {opCount}[] registered operators, exactly once. They are
documentation groups rather than separate implementations: constructors and attributes live in
{src "TorchLean/Tensor/Op/Table.lean"}[the operator table], executable rules in
{src "TorchLean/Tensor/Op/Table/Rule.lean"}[the rule table], and equation lemmas in the files beside
it. The public tensor functions provide the formulas from which those rules are built. We follow
PyTorch and JAX where an imported program needs their convention, IEEE 754 where exceptional
floating-point values matter, and the cited papers for named algorithms. A different choice is
recorded explicitly instead of hidden behind a familiar name.

Every group ends with a generated table. The build computes its rows from the constructors, using
the model's scalar for an element argument where the constructor has one:

- Rule: the capability of the scalar that the rule reads (`Op.need`), and the attributes at which
  the operator has no rule (`Op.hasRule`).
- CUDA: whether the LibTorch table has an entry (`TorchLean.LibTorch.Kernel.ofOp`).
- Gradient: the kinds of the certificates registered for the pullback at each value input
  (`TorchLean.Op.certificates`), and whether the rule has a forward mode (`Op.Rule.jvp?`), with the
  kinds of the certificates registered for it.
- Bounds: which bound propagations have an arm for the operator: the certificate checker
  (`TorchLean.CROWN.Op.cert?`), interval propagation (`TorchLean.CROWN.Op.ibp?`) and the
  backward sweep of CROWN (`TorchLean.CROWN.Op.backwardRule`). The cell ends in `enclosure` when an
  `enclosure` certificate is registered for the operator.

A cell says whether its property holds at every attribute, at some or at none, and a propagation
has an arm for an operator when its definition returns a result at some attributes and inputs. An
arm is a rule, not a theorem. We prove the checker's step against the real value semantics
`TorchLean.CROWN.Op.evalNode?`, and `TorchLean.CROWN.Op.ibp?_encloses` proves that the boxes of the
interval rule enclose the real values for the operators of `Directed.Supported`.
`TorchLean.CROWN.Op.backwardRule_sound` proves the rules of the CROWN sweep sound at a node whose
real equations hold. In a successful real run, `TorchLean.CROWN.Op.outputBoxCROWN?_of_run` proves
that the CROWN box of a graph output encloses its value, and `TorchLean.CROWN.Op.runCROWN_encloses`
proves that the affine bounds of full CROWN bound the values of their nodes. Without an arm of its
own, the backward sweep discharges the coefficient against the operator's box (constants and most
operators) or ends (`gather`, `scatter`, `pad`, `flip`, `contract`, `logSoftmax`, the cumulative and
sorting operators, and `custom`). ReLU has no arm because the forward pass relaxes it with α slopes
(`alphaStep?`).

## Scalar Arithmetic, Rounding, And Exceptional Values

The scalar formulas come from {src "TorchLean/Tensor/Elementwise.lean"}[`Tensor.Elementwise`],
{src "TorchLean/Tensor/Arithmetic.lean"}[`Tensor.Arithmetic`] and
{src "TorchLean/Tensor/Numeric.lean"}[`Tensor.Numeric`]. Their operator equations are in
{src "TorchLean/Tensor/Op/Table/Elementwise.lean"}[`Op.Table.Elementwise`]. `div` records
PyTorch's `rounding_mode`; `nextafter`, `frexp`, `fma` and the rounding operators retain their
standard scalar meanings.

Several distinctions here came from numerical failures rather than API taste. Raw square root
returns NaN below zero, whereas the clamped mode computes $`\sqrt{\max(x,0)}` and has zero stored
gradient when $`x \le 0`. The logarithm pullback evaluates $`g/x` directly, and reciprocal uses
$`(-g/x)/x`; forming $`1/x` or $`x^2` first can overflow before the final result is representable.
`smoothAbs` scales before squaring, so large finite terms do not overflow merely because their
unscaled squares do. IEEE `max` and `min` propagate NaNs and order signed zeros; `fmax` and `fmin`
are the NaN-ignoring variants. Guarded operations have separate names because changing a domain or
zero convention changes the function being verified.

```operators
add sub mul div neg abs sqrt rsqrt cbrt reciprocal
square pow scale fmod remainder hypot copysign
nextafter frexp fma lerp addcmul addcdiv nanToNum fullLike
safeDiv safeInv safeLog smoothAbs
max min fmax fmin clamp
floor ceil round trunc frac sign
```

## Transcendentals, Special Functions, And Complex Arithmetic

These rules use {src "TorchLean/Tensor/Numeric.lean"}[the scalar numeric interface],
{src "TorchLean/Tensor/Special/Fn.lean"}[the special-function interface] and
{src "TorchLean/Tensor/ComplexFn.lean"}[the complex formulas]. A complex tensor stores real and
imaginary components on a final axis of length two. This representation keeps shapes explicit and
lets the same graph distinguish real, complex and configured floating-point interpretations.
Functions with restricted real domains preserve their raw exceptional values; a guarded operation
uses a guarded name rather than silently changing `log`, `acosh` or `atanh`.

```operators
exp exp2 expm1 log log2 log10 log1p
sin cos tan asin acos atan atan2
sinh cosh asinh acosh atanh erf
special complex complexPow
```

## Activations And Probability Transforms

Activation formulas are in
{src "TorchLean/Tensor/Layers/Activation.lean"}[`Tensor.Layers.Activation`], and normalized
exponentials in {src "TorchLean/Tensor/Layers/Softmax.lean"}[`Tensor.Layers.Softmax`]. `gelu`
records PyTorch's `approximate` choice instead of treating its exact and tanh forms as one
operation. `softplus` uses $`x+\log(1+e^{-x})` on the positive side and
$`\log(1+e^x)` otherwise, avoiding overflow and cancellation in the naive formula. Softmax and
log-softmax subtract a row pivot before exponentiating. That stable algebra is part of the
operator meaning, not a compiler rewrite. Masked softmax also records the mask and all-masked-row
convention explicitly.

```operators
relu sigmoid tanh softplus silu gelu elu leakyReLU
softmax logSoftmax maskedSoftmax
```

## Predicates, Selection, Words, And Conversion

Comparisons and logic have Boolean outputs; `select` is the tensor form of `where`, and `cast`
converts Booleans or indices to the graph's scalar element. Word operations are specified in
{src "TorchLean/Tensor/Word.lean"}[`Tensor.Word`], with equation lemmas in
{src "TorchLean/Tensor/Op/Table/Compare.lean"}[`Op.Table.Compare`] and
{src "TorchLean/Tensor/Op/Table/Word.lean"}[`Op.Table.Word`]. Shift direction, word width,
bitcasts and index encoding are data in the IR rather than assumptions made by a backend.
Predicates make NaN and infinity tests visible to proofs and importers.

```operators
eq ne lt le gt ge isNaN isInf isFinite
logicalNot logicalAnd logicalOr logicalXor
select cast
bitwiseAnd bitwiseOr bitwiseXor bitwiseNot
bitwiseLeftShift shiftRightArithmetic shiftRightLogical
populationCount countLeadingZeros mulhi bitcast
encodeIndex decodeIndex
```

## Reductions, Scans, And Order Statistics

Reference reductions are defined in {src "TorchLean/Tensor/Reduce.lean"}[`Tensor.Reduce`], scans in
{src "TorchLean/Tensor/Cumulative.lean"}[`Tensor.Cumulative`] and sorting in
{src "TorchLean/Tensor/Sort.lean"}[`Tensor.Sort`]. A reduction order can be a left fold, pairwise
tree, blocked tree or explicit tree. That choice is observable in floating-point arithmetic: the
same values can round differently when their additions are regrouped. Exact real operations agree
when associativity and commutativity apply, while FloatLib bounds depend on tree depth. Maximum and
minimum reductions start from the first entry, propagate NaNs through the scalar extrema, and split
the stored pullback evenly among tied extrema. Empty-axis restrictions and returned indices are
part of the registered port contract.

```operators
cumsum cumprod cummax cummin sort topk argmax argmin
reduce mean
```

## Indexing And Layout

The formulas in {src "TorchLean/Tensor/Movement.lean"}[`Tensor.Movement`] and
{src "TorchLean/Tensor/Scatter.lean"}[`Tensor.Scatter`] define how coordinates move. `gather`
supports arbitrary gather axes and batch dimensions; `scatter` records replacement, sum, product,
maximum or minimum as an attribute. `reshape`, `permute` and `broadcast` change the coordinate map,
not the scalar formula. Keeping these as typed operators lets Lean reject an impossible shape
before a backend sees a buffer.

```operators
gather scatter indices indexAt slice embedding
reshape permute broadcast flip pad cat
```

## Products And Affine Maps

Contractions are specified in {src "TorchLean/Tensor/Contract.lean"}[`Tensor.Contract`] and affine
maps in {src "TorchLean/Tensor/Layers/Linear.lean"}[`Tensor.Layers.Linear`], with graph equations in
{src "TorchLean/Tensor/Op/Table/Linear.lean"}[`Op.Table.Linear`]. `matmul` and `scaledMatmul`
support broadcast batch prefixes; `linear` applies over leading axes. A contraction records its
reduction order, so a proof can refer to the same multiplication and addition schedule that the
reference evaluator uses.

```operators
contract matmul scaledMatmul linear
```

## Convolution And Pooling

Window extraction, convolution, transposed convolution and pooling are defined in
{src "TorchLean/Tensor/Window.lean"}[`Tensor.Window`], with registered equations in
{src "TorchLean/Tensor/Op/Table/Conv.lean"}[`Op.Table.Conv`]. Stride, padding, dilation, groups,
pooling kind and reduction order are attributes. Maximum pooling uses the same winning entry in its
forward and pullback rules; its NaN policy follows the registered maximum rather than a foreign
kernel's incidental comparison. `smoothMaxPool` is separate because it is a different
differentiable function, not a mode of hard maximum pooling.

```operators
conv convTranspose pool smoothMaxPool
```

## Normalization

Normalization is specified in
{src "TorchLean/Tensor/Layers/Normalization.lean"}[`Tensor.Layers.Normalization`] and connected to
the graph table by {src "TorchLean/Tensor/Op/Table/Normalization.lean"}[the normalization
equations]. Layer normalization scales a centered row before squaring it and adjusts epsilon by the
same scale; analogous compositions implement RMSNorm, batch-statistics BatchNorm and GroupNorm
without extra registered operators. This avoids overflow in a row such as `[-3e38, 3e38]` while
preserving the real formula. `batchNormRunning` is distinct from `batchNorm`: one consumes stored
statistics, while the other computes them from the batch and can update state.

```operators
layerNorm batchNorm batchNormRunning
```

## Attention

The definitions in {src "TorchLean/Tensor/Layers/Attention.lean"}[`Tensor.Layers.Attention`]
separate score formation, normalization and value mixing. `attentionFromScores` begins after score
formation; `multiheadAttention` includes the head layout and projections. Their equations live in
{src "TorchLean/Tensor/Op/Table/Attention.lean"}[`Op.Table.Attention`]. Flash-style or fused
backend code is not declared equivalent merely because it has the same name: its contract must
account for the mask, scale, reduction order and precision.

```operators
attention attentionFromScores multiheadAttention
```

## Losses

Loss formulas are in {src "TorchLean/Tensor/Layers/Loss.lean"}[`Tensor.Layers.Loss`] and their
axis-aware variants in {src "TorchLean/Tensor/Layers/Loss/Axis.lean"}[`Loss.Axis`]. Registered
equations are in {src "TorchLean/Tensor/Op/Table/Loss.lean"}[`Op.Table.Loss`]. Cross-entropy from
logits uses log-softmax rather than the logarithm of rounded probabilities. Log-cosh uses
$`\log(1+2\sinh^2(a/2))` for small residuals and
$`a+\log(1+e^{-2a})-\log 2` in the large branch, avoiding near-zero cancellation and large
overflow. Probability-input and logits-input cross-entropies remain different constructors so an
importer cannot quietly apply the wrong domain assumptions.

```operators
mseLoss maeLoss huberLoss hingeLoss poissonLoss
logCoshLoss binaryCrossEntropy cosineLoss
crossEntropy crossEntropyProbabilities distill
```

## Mask Construction And Randomness

Random draws are pure functions of explicit keys, as defined in
{src "TorchLean/Tensor/Random.lean"}[`Tensor.Random`]. Dropout takes its mask as an input, while
`bernoulliMask`, `randUniform` and `randn` produce keyed samples. This separation keeps randomness
out of implicit global state and lets a graph proof state which sample it is about. `topKMask` is
deterministic and shares the ordering conventions of `topk`.

```operators
dropout topKMask bernoulliMask randUniform randn
```

## Spectral And State-Space Operators

Real FFTs are defined in {src "TorchLean/Tensor/Layers/RealFFT.lean"}[`Tensor.Layers.RealFFT`], and
the state-space scan in
{src "TorchLean/Tensor/Layers/StateSpace.lean"}[`Tensor.Layers.StateSpace`]. Their registered
equations use the same explicit tensor layouts. `rfft` and `irfft` expose normalization and complex
layout to the graph, while `selectiveScan` exposes the recurrence instead of treating a fused
state-space kernel as an opaque call.

```operators
rfft irfft selectiveScan
```

## Graph Structure And Custom Programs

`detach` returns the same forward value while cutting the pullback. `custom` is the one registered
constructor without a built-in table rule: a checked node program supplies its node-local
semantics. This lets a frontend carry a supported tensor computation without pretending that an
arbitrary foreign function already has a TorchLean proof or backend refinement.

```operators
detach custom
```

Together the generated tables list all {opListed}[] constructors. The `operators` expander rejects
unknown or repeated names, and the final count fails the documentation build if a constructor is
missing. Rule, CUDA, gradient and bound cells therefore come from current source rather than a
manually maintained support claim.

# Import Coverage

The vocabulary tables cover every TorchLean operator. PyTorch and JAX each have a larger vocabulary,
so import coverage also depends on the frontend, dtype, and attributes.

## Where a Cast Goes

Consider two versions of a PyTorch square. Both return a float16 tensor of the same shape, and both
pass a type check. In the first, the cast rounds the input to one before we square it. In the
second, the input survives the multiplication in float32 and rounds up when converted at the end.
After [setting up the checker](https://lean-dojo.github.io/TorchLean/examples/verify/), try both.

```
# Python / PyTorch
import torch
import torchlean

def round_first(x):
    y = x.to(torch.float16)
    return y * y

def round_last(x):
    return (x * x).to(torch.float16)

x = torch.tensor([1.0003], dtype=torch.float32)
for square in (round_first, round_last):
    report = torchlean.verify(square, x, checks=("types", "compare"))
    print(square(x).item(), report["ok"])
    assert report["ok"], report
```

```
1.0 True
1.0009765625 True
```

Each `True` says the Lean reference matched PyTorch for that function and input. The imported
graphs keep the cast in different places, so we can see why the results differ and state an error
bound for either calculation. The
[JAX walkthrough](https://lean-dojo.github.io/TorchLean/examples/jaxpr/) follows the same idea
from a sum of squares to a trained classifier.

## Which Operations Import

Both the PyTorch capture and the JAX importer translate a framework's operators onto this table. An
operator becomes an IR node whose kind projects onto the registered operator or, where an operand
or result is an index or a Boolean beside a scalar, a custom node holding the operator's record. An
imported operator is computed by its registered operator for the attributes, element types and
inputs the capture accepts, and the capture refuses the others. Some imports also fix a policy.
Cumulative sums and products run in PyTorch's widened accumulation type, and `topk` fails at
execution where tied entries would decide its selection. A runtime scatter folds the updates of a
repeated target in the row-major order of its index tensor, and an in-place `index_put_` is
accepted only into fresh storage. For a training batch norm, the buffer updates are computed out of
place and returned after the program's outputs. And `randn` is a deterministic Box-Muller transform
of two uniform draws, one clamped away from zero. Operator by operator, the
{src "TorchLean/NN/IR/README.md"}[IR guide] and the
{src "TorchLean/NN/PyTorch/README.md"}[PyTorch interop guide] say where the import writes
the arithmetic of a framework's kernel and where it departs from eager PyTorch or XLA.

An operator outside a translation is refused with an error that names it. On the PyTorch side, the
capture has no translation for the backward operators of a joint graph, since TorchLean derives
gradients from the registered rules. Nor does it translate `_local_scalar_dense`, which reads a
tensor
as a Python scalar. Operators with uninitialized contents (`empty` and its variants) hold the fill
PyTorch documents for its deterministic mode, and the strided view `as_strided` gathers the
entries it reads, recording that the inputs must be laid out as at capture. It also refuses
results whose shape depends on the data, `nonzero` and Boolean mask indices among them (the bounded
`nonzero_static` imports), and control flow other than `torch.cond`, `while_loop`, `map` and
`invoke_subgraph`. The JAX importer rejects control flow other than `cond`, `scan` and `while`, an
`accuracy` request on a transcendental primitive and nondefault dot precisions, and it has no
translation for multi-device collectives, random key primitives or host callbacks.

Input-independent creations fold to constants the framework computes (`arange`, `full`, `iota`),
and `clone`, `alias` and `copy` pass their operand through.
