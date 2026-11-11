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

#doc (Manual) "Registered Operators" =>
%%%
tag := "registered-operators"
file := "The-Operator-Table"
%%%

PyTorch registers an operator once. `native_functions.yaml` declares `relu` with its schema,
`derivatives.yaml` gives its backward formula, and eager execution, autograd and `torch.export` all
refer to that one entry. TorchLean has the same arrangement in Lean. Here the inductive type
`TorchLean.Op` names every registered operator, and `Op.rule?` gives each one its rule: the forward
map, the pullback and the node a recorder builds. The CPU tape, the CUDA tape, the eager session,
the verifier's recorder, the graph IR, backend planning and the roundoff analysis all key on these
two definitions, so each operator is defined once for all of them.

The table lives in {src "TorchLean/Tensor/Op/Table.lean"}[`TorchLean/Tensor/Op/Table.lean`], its
rules in {src "TorchLean/Tensor/Op/Table/Rule.lean"}[`TorchLean/Tensor/Op/Table/Rule.lean`] and the
equation lemmas of each operator family beside them. Every Lean block below is elaborated while
this page is built, and the counts in the text and the tables of the vocabulary are read off the
definitions in the same build.

# One Constructor Per Operator

A constructor of `TorchLean.Op` names one operator together with its attributes. Its type lists
the operator's input ports and output ports:

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
Op.sort ⟨0, ⋯⟩ false : Op [Port.value [3]]
  [Port.value [3], Port.index [Shape.extent [3] ⟨0, ⋯⟩] [3]]
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
element argument (described below). Both bounds of `clamp` are frozen scalar inputs. An operator
value doesn't know the scalar type, so a scalar attribute such as an epsilon, a slope or a clamp
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

For `reshape`, `permute`, `broadcast`, `flip`, `slice`, `pad`, `cat`, `gather`, `scatter` and
`select` (`Op.Element`), and for `add`, `sub`, `mul`, `div`, `neg`, `pow`, `remainder`, `max`,
`min` and the comparisons (`Op.NumericElement`), the element argument defaults to `.scalar`, the
model's scalar. At that default, `Op.Element.port` gives the scalar's port of the operand's role:
the data ports of `select` above are value ports. At `.bool` or `.integer signed width` the
operator moves Booleans or integer words as it moves scalars, or computes PyTorch's integer meaning
with the functions of `TorchLean.Word` (wrapping `+`, `-`, `*` and negation, the order of the type),
without gradients. An integer division or remainder fails its check at a zero divisor, an integer
true division (`div .none`) has no rule, and a Boolean scatter has a rule only as a replacement.

For a list of ports, the values are nested pairs ending in `Unit`, so the inputs of `Op.mul` are
`(x, y, ())`. Tangents and cotangents exist for value ports only: every other port has `Unit`
there. Elementwise operators take equal shapes, and broadcasting is the separate operator
`broadcast`: where PyTorch broadcasts the operands of `add` implicitly, the capture writes the
broadcast as its own node.

# The Rule Of An Operator

`Op.rule? op` is the rule of `op` at a scalar `α`, an `Option (Op.Rule α ins outs)`. A rule holds:

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
of a binary format (for `bitcast`). `Op.Capabilities α` bundles these six capabilities, found by
instance synthesis. An operator that needs a missing capability has
no rule at that scalar:

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
has `Rounding` and `InverseTrig` but no `Erf`, `Cbrt`, `NextAfter` or `Bits`, and the configured
binary formats of FloatLib have every capability but `Erf`. `Op.need` names the capability an
operator's rule reads, and `Op.isSome_rule?_iff` proves that an operator for which `Op.hasRule`
holds has a rule at a scalar exactly when the scalar supplies that capability. `Op.needsCapability`
names the operators whose rule reads a capability, and `Op.rule?_capabilities` proves that every
other rule is the same for all capability instances.

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
{ version := 3, tag := "softmax",
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

# The Vocabulary

Each family below is a table with one row per operator, computed while this page is built from
the definitions at the operator's constructor, with its element argument at the model's scalar
where it has one:

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

Elementwise arithmetic, with PyTorch's `rounding_mode` on `div`, the cube root, C's `nextafter`
and the guarded forms `safeDiv`, `safeInv`, `safeLog` and `smoothAbs`:

```operators
add sub mul div neg abs sqrt rsqrt cbrt reciprocal
square pow scale fmod remainder hypot copysign
nextafter lerp addcmul addcdiv nanToNum fullLike
safeDiv safeInv safeLog smoothAbs
```

Exponentials, logarithms, and trigonometric and hyperbolic functions:

```operators
exp exp2 expm1 log log2 log10 log1p
sin cos tan asin acos atan atan2
sinh cosh asinh acosh atanh erf
```

Roundings and signs:

```operators
floor ceil round trunc frac sign
```

Extrema, of which `max` and `min` propagate a NaN, `fmax` and `fmin` ignore one, and `clamp`
reads its bounds at frozen ports:

```operators
max min fmax fmin clamp
```

Activations, with PyTorch's `approximate` on `gelu`:

```operators
relu sigmoid tanh softplus silu gelu elu leakyReLU
```

Comparisons, predicates and logic, whose outputs are Booleans, with `select` on a Boolean
condition and `cast`, which reads Booleans or indices as scalars:

```operators
eq ne lt le gt ge isNaN isInf isFinite
logicalNot logicalAnd logicalOr logicalXor
select cast
```

Integer words: the bitwise operators, the shifts, `populationCount`, `countLeadingZeros` and
`mulhi` compute the functions of `TorchLean.Word`, `bitcast` reads each word at another sort of the
same width, and `encodeIndex` and `decodeIndex` convert index tuples to and from words in the IR's
layout (`Port.irShape`):

```operators
bitwiseAnd bitwiseOr bitwiseXor bitwiseNot
bitwiseLeftShift shiftRightArithmetic shiftRightLogical
populationCount countLeadingZeros mulhi bitcast
encodeIndex decodeIndex
```

Scans and orderings, of which `cummax`, `cummin`, `sort`, `topk`, `argmax` and `argmin` return
indices:

```operators
cumsum cumprod cummax cummin sort topk argmax argmin
```

Indexing, with `gather` over any axes and batch dimensions and `scatter` as a replacement, sum,
product, maximum or minimum:

```operators
gather scatter indices indexAt slice embedding
```

Data movement:

```operators
reshape permute broadcast flip pad cat
```

Reductions and contractions, `reduce` being a sum, product, maximum or minimum over any axes:

```operators
reduce mean contract
```

Products, `matmul` and `scaledMatmul` with broadcast batch prefixes and `linear` over leading
axes:

```operators
matmul scaledMatmul linear
```

Convolution and pooling:

```operators
conv convTranspose pool smoothMaxPool
```

Normalization, `layerNorm` over the last axis, `batchNorm` with batch statistics and
`batchNormRunning` with running statistics along any axis:

```operators
layerNorm batchNorm batchNormRunning
```

Softmax and attention:

```operators
softmax logSoftmax maskedSoftmax
attention attentionFromScores multiheadAttention
```

Losses:

```operators
mseLoss maeLoss huberLoss hingeLoss poissonLoss
logCoshLoss binaryCrossEntropy cosineLoss
crossEntropy crossEntropyProbabilities distill
```

Masks and random draws, each draw under a key, with the mask of `dropout` an input:

```operators
dropout topKMask bernoulliMask randUniform randn
```

Spectral transforms and the selective scan:

```operators
rfft irfft selectiveScan
```

Graph structure:

```operators
detach custom
```

Together the tables list all {opListed}[] constructors.

# Import Coverage

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
gradients from the rules above. Nor does it translate operators with uninitialized contents, the
strided aliasing view `as_strided` or `_local_scalar_dense`, which reads a tensor as a Python
scalar. It also refuses results whose shape depends on the data, `nonzero` and Boolean mask indices
among them (the bounded `nonzero_static` imports). The JAX importer rejects control flow, complex
inputs and outputs, an `accuracy` request on a transcendental primitive and nondefault dot
precisions, and it has no translation for multi-device collectives, random key primitives or host
callbacks.

Input-independent creations fold to constants the framework computes (`arange`, `full`, `iota`),
and `clone`, `alias` and `copy` pass their operand through.
