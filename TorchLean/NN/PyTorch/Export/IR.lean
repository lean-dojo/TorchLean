/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.PyTorch.Export
public import TorchLean.NN.IR.Imported

/-!
# PyTorch code generation

Operator graph to PyTorch code generation.

`emit imported` writes a standalone PyTorch `nn.Module` for the operator graph of an import
(`TorchLean.IR.Imported Float`, which a `IR.LoweredModel` extends). Its constants, read from the
payload, become parameters and buffers, and each node becomes a few lines of `forward`, whose
argument `x` is graph input `0` and whose result is the graph's first output.

The emitter is an extraction layer for round-trip examples, which run or train the module in
Python and may import its weights back into Lean. The graph's evaluation, `TorchLean.IR.Graph.eval`,
is the reference semantics, and the emitted Python lies outside the proof boundary.

Output `k` of node `i` is the Python local `v{i}`, or `v{i}_{k}` for `k > 0`. An operator node is
one row of `opLines`, keyed on its `TorchLean.Op` constructor: one call of a PyTorch function on its
inputs in port order with its attributes as keywords, or the operator's own formula (the stable
sigmoid, SiLU and softplus, both GELU formulas, layouts, normalizations, masked softmax,
selections). SiLU and GELU return their limits, `+∞` at `+∞` and `-0` at `-∞`, and evaluate their
formula at zero there, so that no autograd slope is a NaN. The emitted slope at `-∞` is then `+0`,
where the operator's derivative gives `-0`. Boolean outputs stay Boolean and index outputs are
`torch.long` tensors, as PyTorch returns them.

A constant that `forward` reads as a tensor, or the output, is the attribute `const_{i}`. It is an
`nn.Parameter` when it is the weight or bias of a linear map or convolution or the scale or shift
of a normalization, or when `learnableConsts` holds and its `Float` entries differ. A constant read
as running statistics is a buffer, as is every other constant. A normalization's epsilon that is a
constant is written as a literal.

The extrema `maximum`, `minimum`, `fmax`, `fmin`, `amax` and `amin` keep PyTorch's value and
gradient and give a zero result the sign IEEE 754-2019 gives it, as PyTorch returns either zero at
a tie of `-0` and `+0`. `relu` adds `+0`, so it returns `+0` at `-0` as `Activation.relu` does. A
product over axes multiplies each fiber's entries in row-major order, one Python loop step per
entry, because `torch.prod` regroups them and a regrouped partial product can overflow where
`Graph.eval`'s left fold doesn't. A mean over an empty fiber is the sum `+0`, as `Tensor.mean`
divides an empty fiber's sum by one and `torch.mean` returns a NaN there. The mean squared error
of empty inputs is the sum `+0` too, where `mse_loss` returns a NaN. `topk` is the first `k`
entries of the stable sort, as `torch.topk` doesn't keep the index order of ties. A random node
reads the buffer `draws_{i}`, which holds the draws of the exported key (`Random.uniform` at
the node's shape, exact in binary64 and rounded once into the module's dtype). A Bernoulli mask
compares its keep probability with them as `Random.keepBit` does. A truncating division is
the floor of the exact quotient `|a| / |b|` with the sign of `a / b`, as the operator computes
it, since `rounding_mode='trunc'` truncates the rounded quotient: `10` at `(1, 0.1)`, where
the operator gives `9`.

The rows don't match `Graph.eval` everywhere. Sums, means, matrix products, contractions,
convolutions and cumulative sums and products are PyTorch's own, so their rounding can differ from
`Graph.eval`'s left folds. At a NaN operand `torch.maximum` and `torch.minimum` give the whole
gradient to both operands, where the operators give each half. At a NaN extremum `torch.amax` and
`torch.amin` give every entry of the fiber a NaN gradient, where the reductions split it among the
NaN entries. At an infinite or NaN upstream gradient they give a NaN gradient to the entries that
don't equal the extremum, where the reductions give them zero. The gradient of a product over axes
is autograd's pass back through the loop. It multiplies the upstream gradient and the other entries
of the fiber in another grouping than the operator's pullback, so their rounding and overflow can
differ. A scaled product is the product, then the scale. When a product or partial sum of finite
inputs overflows or is subnormal, a product of nonzero factors rounds to zero or a nonzero term is
absorbed, `Graph.eval` rounds the exact coefficients once (`scaledProductsSlice`) and the emitted
product doesn't. The gradient of a pad's value is autograd's sum of the upstream gradient at the
padding entries, in its own order. The emitted gradient of `abs` is torch's `grad * x.sgn()`,
which at seed `1` is `+0` at `-0` and at a NaN, where the operator's pullback sends `-0` and a
NaN. A replacing scatter is `index_put` with `accumulate=False`. Where several updates land on
one entry, torch leaves the written value unspecified and sends the upstream gradient to each
of them, while the operator keeps the last update and sends the gradient only to it. The emitted
mean squared error is `mse_loss`, whose sum is PyTorch's own, and when every difference `d` is
finite but the sum of their squares overflows, `Tensor.meanSquaredError` adds the terms
`d * (d / n)` over the `n` entries instead: at the binary64 differences `[2e154, 0, 0, 0]` it gives
`1e308`, where `mse_loss` gives `+∞`.

Emission fails with an `Except String` message on a graph without an output or without input `0`,
or with an output reference to no node output. It also fails on another graph input, a missing
constant, an integer constant that `forward` reads and an operator with the wrong number of inputs.
It refuses `custom`, a convolution, pooling or scatter outside the translated forms, and every
operator without a row, whose message names the node and the codec tag. It also refuses forms
that PyTorch rejects or evaluates differently: a convolution without input or output channels,
with a zero dilation or without a window position along a spatial axis, a three-axis average
pooling with an input extent below the kernel's, a contraction whose einsum labels run past the
52 letters and a concatenation of no inputs. The labels are the letters `a` to `z` and `A` to `Z`
by index: axis `i` of the left operand gets letter `i`, and axis `j` of the right operand gets its
partner's letter when a contracted or batch pair names it and letter `r + j` when it is free, where
`r` is the left operand's rank. So the emitter refuses a contraction with `r > 52` or with a free
right axis `j` at `r + j ≥ 52`, even when its distinct labels fit in 52 letters: two rank-30
operands contracted over their first 25 axes need 35 labels and are refused.
`Graph.Builder.build` and the import check the graph, and the emitter does not repeat the check.

PyTorch's ONNX exporter and `torch.export` capture a PyTorch program as a graph for execution in
other runtimes. The emitter goes the other way and prints a graph as readable Python.
-/

public section

namespace TorchLean
namespace PyTorch

open TorchLean.Tensor
open IR

/-- Flatten a tensor and render it as a Python list literal (used for
  `torch.tensor([...]).reshape(...)`). -/
def flat {s : Shape} (t : Tensor Float s) : String :=
  tensorLiteral (Tensor.reshape (α := Float) t [_] (Shape.size_vector _).symm)

/-- Return `true` iff every array entry equals the first one, vacuously for an empty array. -/
def allEq (xs : Array Float) : Bool :=
  match xs[0]? with
  | none => true
  | some x => xs.all (fun y => y == x)

/-- Options controlling emission: the class name, the dtype, the training skeleton and whether
constants with differing entries become parameters. -/
structure Options where
  /-- Class name to use in the emitted Python source. -/
  className : String := "ExportedIRModel"
  /-- Python expression for the dtype of the emitted tensors, by default `torch.float64`, the
  binary64 format of the payload's `Float`. -/
  dtypeExpr : String := "torch.float64"
  /-- If true, append a training skeleton: an optimizer helper, a training step and a `__main__`
    block that runs one step on random inputs. -/
  includeTrainingSkeleton : Bool := true
  /-- If true, a value, frozen or detached constant whose entries differ becomes an
    `nn.Parameter`, unless an input reads it as running statistics. -/
  learnableConsts : Bool := true
  deriving Repr

/-- Emit a `torch.tensor([...]).reshape(...)` expression from a flat Python list literal and a
`Shape`. -/
def tensor (flatList : String) (shape : Shape) (dtypeVar : String := "dtype") : String :=
  s!"torch.tensor({flatList}, dtype={dtypeVar}).reshape({shapeLiteral shape})"

/-- The Python local of output `r`: `v{node}`, or `v{node}_{out}` for a later output. -/
def name (r : Ref) : String :=
  if r.out = 0 then s!"v{r.node}" else s!"v{r.node}_{r.out}"

/-- Python expressions as a tuple: `()` for none, as `(,)` is not Python. -/
private def tuple (xs : List String) : String :=
  if xs.isEmpty then "()" else s!"({", ".intercalate xs},)"

/-- The `__init__` expression of a constant, with its entries when it has a value, frozen or
detached port. An integer constant is refused. -/
private def constExpr : Op.Value Float → Except String (String × Option (Array Float))
  | ⟨⟨.value, s⟩, t⟩ => pure (tensor (flat t) s, some (Tensor.to t (Array Float)))
  | ⟨⟨.frozen, s⟩, t⟩ => pure (tensor (flat t) s, some (Tensor.to t (Array Float)))
  | ⟨⟨.detached, s⟩, t⟩ => pure (tensor (flat t) s, some (Tensor.to t (Array Float)))
  | ⟨⟨.bool, s⟩, t⟩ =>
      let entries := (Tensor.to t (Array Bool)).map boolLiteral
      pure (s!"torch.tensor([{", ".intercalate entries.toList}], dtype=torch.bool)" ++
        s!".reshape({shapeLiteral s})", none)
  | ⟨⟨.index bounds, s⟩, t⟩ =>
      let entries := (Tensor.to t (Array (Port.IndexTuple bounds))).flatMap fun i =>
        (Port.IndexTuple.components bounds i).toArray
      let listed := ", ".intercalate (entries.toList.map toString)
      pure (s!"torch.tensor([{listed}], dtype=torch.long)" ++
        s!".reshape({shapeLiteral (Port.irShape ⟨.index bounds, s⟩)})", none)
  | ⟨⟨.integer _ _, _⟩, _⟩ => throw "IR→PyTorch: an integer constant has no export"

/-- The Python literal of a scalar constant at a value or frozen port. -/
private def scalarLiteral? : Op.Value Float → Option String
  | ⟨⟨.value, .scalar⟩, t⟩ => some (floatLiteral (item t))
  | ⟨⟨.frozen, .scalar⟩, t⟩ => some (floatLiteral (item t))
  | _ => none

/-- How an operator reads its input `k`: as the weight or bias of a linear map or convolution or
the scale or shift of a normalization, as running statistics, as an epsilon written as a literal
when it is a constant, or as a tensor. -/
private inductive Use where
  | parameter | statistics | literal | tensor
  deriving BEq

/-- The use of input `k` of an operator. -/
private def Use.of : Operator → Nat → Use
  | ⟨_, _, .linear⟩, k | ⟨_, _, .conv ..⟩, k => if k < 2 then .parameter else .tensor
  | ⟨_, _, .layerNorm ..⟩, k =>
      if k = 1 ∨ k = 2 then .parameter else if k = 3 then .literal else .tensor
  | ⟨_, _, .batchNormRunning _⟩, k =>
      if k = 1 ∨ k = 2 then .parameter else if k = 3 ∨ k = 4 then .statistics
      else if k = 5 then .literal else .tensor
  | _, _ => .tensor

/-- The uses of the first output of every node by the operator nodes of the graph. -/
private def uses (g : Graph) : Std.HashMap Nat (List Use) :=
  g.nodes.foldl (fun acc n => match n with
    | .op o refs => (List.range refs.size).foldl (fun acc k => match refs[k]? with
        | some r => if r.out = 0 then acc.insert r.node (Use.of o k :: acc.getD r.node []) else acc
        | none => acc) acc
    | _ => acc) {}

/-- The `__init__` line of the buffer `draws_{i}`: the draws `Random.uniform key` at the shape
`s` that `Graph.eval` makes for a random node of key `key`, in binary64. -/
private def drawsLine (i : Nat) (s : Shape) (key : UInt64) : String :=
  indent 4 <| s!"self.register_buffer(\"draws_{i}\", " ++
    s!"{tensor (flat (Random.uniform (α := Float) key (s := s))) s})"

/-- The constants that `forward` reads, and the `__init__` lines that set them. A constant read at
an input other than a literal epsilon, or the output, is a parameter or a buffer, and the draws of
each random node are the buffer `draws_{i}` (`drawsLine`). -/
private def bindings (g : Graph) (P : Payload Float) (options : Options) (output : Ref) :
    Except String (Std.HashSet Nat × Array String) := do
  let uses := uses g
  let mut bound : Std.HashSet Nat := {}
  let mut lines : Array String := #[]
  for i in [0:g.nodes.size] do
    match g.nodes[i]? with
    | some (.const _) =>
        let us := uses.getD i []
        if output == ⟨i, 0⟩ || us.any (· != .literal) then
          let some v := P.const? i | throw s!"IR→PyTorch: missing constant for node {i}"
          let (expr, entries?) ← constExpr v
          let learn := match entries? with
            | none => false
            | some entries => !us.contains .statistics &&
                (us.contains .parameter || (options.learnableConsts && !allEq entries))
          lines := lines.push <| indent 4 <|
            if learn then s!"self.const_{i} = nn.Parameter({expr})"
            else s!"self.register_buffer(\"const_{i}\", {expr})"
          bound := bound.insert i
    | some (.op ⟨_, _, .randUniform (s := s) key⟩ _) => lines := lines.push (drawsLine i s key)
    | some (.op ⟨_, _, .bernoulliMask (s := s) key⟩ _) => lines := lines.push (drawsLine i s key)
    | _ => pure ()
  pure (bound, lines)

/-- The index of axis `d` in a slice: `start:stop:step` for a sliced axis, `:` for another. -/
private def sliceEntry (dims : List Nat) (starts stops steps : List Nat) (d : Nat) : String :=
  match dims.idxOf? d with
  | some k => s!"{starts[k]?.getD 0}:{stops[k]?.getD 0}:{steps[k]?.getD 1}"
  | none => ":"

/-- The PyTorch function of pooling `op` over `r` trailing axes, if it has one. -/
private def poolFunction? : Tensor.PoolOp → Nat → Option (String × String)
  | .max, r => if 1 ≤ r ∧ r ≤ 3 then some (s!"max_pool{r}d", "") else none
  | .avg countPad, r =>
      if 1 ≤ r ∧ r ≤ 3 then some (s!"avg_pool{r}d", s!", count_include_pad={boolLiteral countPad}")
      else none
  | _, _ => none

/-- The `forward` lines of operator node `id`, whose inputs are the Python expressions `xs` in port
order, where `literal k` is input `k` written as a literal when it is a scalar constant and `dtype`
is the Python dtype of values: one row per `TorchLean.Op` constructor, a call of one PyTorch
function or the operator's own formula. -/
private def opLines (id : Nat) (dtype : String) (literal : Nat → Option String)
    (xs : Array String) (o : Operator) : Except String (Array String) :=
  let out := s!"v{id}"
  let x (k : Nat) : String := xs.getD k ""
  let line (rhs : String) : Array String := #[indent 4 s!"{out} = {rhs}"]
  let call (f : String) (keywords : List String := []) : Array String :=
    line s!"{f}({", ".intercalate (xs.toList ++ keywords)})"
  let pair (rhs : String) : Array String := #[indent 4 s!"{out}, {out}_1 = {rhs}"]
  -- PyTorch's extrema return either zero at a tie of `-0` and `+0`, depending on the loop. Adding
  -- `+0` to the result, or to its negation where `negative` holds, gives a zero result the sign
  -- IEEE 754 gives it and keeps the value and the gradient of the call `f`.
  let extremum (f negative : String) : Array String :=
    #[indent 4 s!"_m{id} = {f}",
      indent 4 s!"{out} = torch.where({negative}, -((-_m{id}) + 0.0), _m{id} + 0.0)"]
  -- `body` computes `_y` from `_xs`, the input with zero at `±∞`, and the limits `+∞` and `-0`
  -- are selected there, as a formula evaluated at `±∞` would give autograd a NaN slope.
  let withLimits (body : Array String) : Array String :=
    #[indent 4 s!"_x = {x 0}", indent 4 "_inf = torch.isinf(_x)",
      indent 4 "_xs = torch.where(_inf, torch.zeros_like(_x), _x)"] ++ body ++
    #[indent 4 <| s!"{out} = torch.where(_inf, " ++
      "torch.where(_x > 0, _x, -torch.zeros_like(_x)), _y)"]
  let refuse (what : String) : Except String (Array String) :=
    throw s!"IR→PyTorch: node {id}: {what}"
  match o with
  | ⟨_, _, .relu⟩ =>
      -- `torch.relu` returns `-0` at `-0`, where `Activation.relu` returns `+0`. Adding `+0` gives
      -- that zero and keeps every other value and the slope.
      pure (line s!"torch.relu({x 0}) + 0.0")
  | ⟨_, _, .tanh⟩ => pure (call "torch.tanh")
  | ⟨_, _, .exp⟩ => pure (call "torch.exp")
  | ⟨_, _, .abs⟩ => pure (call "torch.abs")
  | ⟨_, _, .sin⟩ => pure (call "torch.sin")
  | ⟨_, _, .cos⟩ => pure (call "torch.cos")
  | ⟨_, _, .neg⟩ => pure (call "torch.neg")
  | ⟨_, _, .log⟩ => pure (call "torch.log")
  | ⟨_, _, .square⟩ => pure (call "torch.square")
  | ⟨_, _, .reciprocal⟩ => pure (call "torch.reciprocal")
  | ⟨_, _, .detach⟩ => pure (call "torch.detach")
  | ⟨_, _, .rsqrt⟩ => pure (call "torch.rsqrt")
  | ⟨_, _, .exp2⟩ => pure (call "torch.exp2")
  | ⟨_, _, .expm1⟩ => pure (call "torch.expm1")
  | ⟨_, _, .log2⟩ => pure (call "torch.log2")
  | ⟨_, _, .log10⟩ => pure (call "torch.log10")
  | ⟨_, _, .log1p⟩ => pure (call "torch.log1p")
  | ⟨_, _, .tan⟩ => pure (call "torch.tan")
  | ⟨_, _, .sinh⟩ => pure (call "torch.sinh")
  | ⟨_, _, .cosh⟩ => pure (call "torch.cosh")
  | ⟨_, _, .asinh⟩ => pure (call "torch.asinh")
  | ⟨_, _, .acosh⟩ => pure (call "torch.acosh")
  | ⟨_, _, .atanh⟩ => pure (call "torch.atanh")
  | ⟨_, _, .add⟩ => pure (call "torch.add")
  | ⟨_, _, .sub⟩ => pure (call "torch.sub")
  | ⟨_, _, .mul⟩ => pure (call "torch.mul")
  | ⟨_, _, .max⟩ =>
      pure (extremum s!"torch.maximum({x 0}, {x 1})"
        s!"torch.signbit({x 0}) & torch.signbit({x 1})")
  | ⟨_, _, .min⟩ =>
      pure (extremum s!"torch.minimum({x 0}, {x 1})"
        s!"torch.signbit({x 0}) | torch.signbit({x 1})")
  | ⟨_, _, .fmax⟩ =>
      pure (extremum s!"torch.fmax({x 0}, {x 1})" <|
        s!"(torch.signbit({x 0}) | torch.isnan({x 0})) & " ++
          s!"(torch.signbit({x 1}) | torch.isnan({x 1}))")
  | ⟨_, _, .fmin⟩ =>
      pure (extremum s!"torch.fmin({x 0}, {x 1})" <|
        s!"(torch.signbit({x 0}) & ~torch.isnan({x 0})) | " ++
          s!"(torch.signbit({x 1}) & ~torch.isnan({x 1}))")
  | ⟨_, _, .hypot⟩ => pure (call "torch.hypot")
  | ⟨_, _, .copysign⟩ => pure (call "torch.copysign")
  | ⟨_, _, .pow⟩ => pure (call "torch.pow")
  | ⟨_, _, .div .none⟩ => pure (call "torch.div")
  | ⟨_, _, .div .trunc⟩ =>
      -- `rounding_mode='trunc'` truncates the rounded quotient, `10` at `(1, 0.1)`, where the
      -- operator floors the exact `|a| / |b|`, `9`, and copies the sign of `a / b`. An infinite
      -- quotient is kept, as torch's floor division gives a NaN at an infinite dividend.
      pure #[indent 4 s!"_q{id} = torch.div({x 0}, {x 1}, rounding_mode='trunc')",
        indent 4 <| s!"{out} = torch.where(torch.isinf(_q{id}), _q{id}, torch.copysign(" ++
          s!"torch.div(torch.abs({x 0}), torch.abs({x 1}), rounding_mode='floor'), _q{id}))"]
  | ⟨_, _, .div .floor⟩ => pure (call "torch.div" ["rounding_mode='floor'"])
  | ⟨_, _, .eq⟩ => pure (call "torch.eq")
  | ⟨_, _, .ne⟩ => pure (call "torch.ne")
  | ⟨_, _, .lt⟩ => pure (call "torch.lt")
  | ⟨_, _, .le⟩ => pure (call "torch.le")
  | ⟨_, _, .gt⟩ => pure (call "torch.gt")
  | ⟨_, _, .ge⟩ => pure (call "torch.ge")
  | ⟨_, _, .isNaN⟩ => pure (call "torch.isnan")
  | ⟨_, _, .isInf⟩ => pure (call "torch.isinf")
  | ⟨_, _, .isFinite⟩ => pure (call "torch.isfinite")
  | ⟨_, _, .logicalNot⟩ => pure (call "torch.logical_not")
  | ⟨_, _, .logicalAnd⟩ => pure (call "torch.logical_and")
  | ⟨_, _, .logicalOr⟩ => pure (call "torch.logical_or")
  | ⟨_, _, .logicalXor⟩ => pure (call "torch.logical_xor")
  | ⟨_, _, .select⟩ => pure (call "torch.where")
  | ⟨_, _, .gelu .tanh⟩ =>
      pure (withLimits #[indent 4 "_y = torch.nn.functional.gelu(_xs, approximate='tanh')"])
  | ⟨_, _, .gelu .none⟩ =>
      pure (withLimits #[indent 4 "_y = torch.nn.functional.gelu(_xs, approximate='none')"])
  | ⟨_, _, .cumsum d⟩ => pure (call "torch.cumsum" [s!"dim={d.val}"])
  | ⟨_, _, .cumprod d⟩ => pure (call "torch.cumprod" [s!"dim={d.val}"])
  | ⟨_, _, .softmax d⟩ => pure (call "torch.softmax" [s!"dim={d.val}"])
  | ⟨_, _, .logSoftmax d⟩ => pure (call "torch.log_softmax" [s!"dim={d.val}"])
  | ⟨_, _, .reshape (t := t) _⟩ => pure (call "torch.reshape" [s!"shape={shapeLiteral t}"])
  | ⟨_, _, .permute perm _ _⟩ =>
      pure (call "torch.permute" [s!"dims={shapeLiteral (List.ofFn fun i => (perm i).val)}"])
  | ⟨_, _, .flip ds⟩ => pure (call "torch.flip" [s!"dims={shapeLiteral (ds.map (·.val))}"])
  | ⟨_, _, .cat d _⟩ =>
      if xs.isEmpty then refuse "a concatenation of no inputs has no PyTorch translation"
      else pure (line s!"torch.cat([{", ".intercalate xs.toList}], dim={d.val})")
  | ⟨_, _, .matmul ..⟩ => pure (call "torch.matmul")
  | ⟨_, _, .mseLoss (s := s)⟩ =>
      -- `Tensor.meanSquaredError` divides an empty input's sum `+0` by one, where `mse_loss` gives
      -- a NaN.
      pure (call "torch.nn.functional.mse_loss" (if s.size = 0 then ["reduction='sum'"] else []))
  | ⟨_, _, .linear⟩ => pure (line s!"F.linear({x 2}, {x 0}, {x 1})")
  | ⟨_, _, .sigmoid⟩ =>
      -- Select the exponent before exp: abs would give the emitted autograd program
      -- a zero slope at the origin.
      pure #[indent 4 s!"_x = {x 0}",
        indent 4 "_z = torch.exp(torch.where(_x > 0, -_x, _x))",
        indent 4 s!"{out} = torch.where(_x > 0, 1 / (1 + _z), _z / (1 + _z))"]
  | ⟨_, _, .silu⟩ =>
      -- A minimum with zero would split the gradient at a tie and change higher derivatives.
      pure <| withLimits
        #[indent 4 "_r = torch.exp(torch.where(_xs > 0, torch.zeros_like(_xs), _xs) * 0.5)",
          indent 4 "_d = 1 + _r * _r",
          indent 4 "_z = torch.exp(torch.where(_xs > 0, -_xs, _xs))",
          indent 4 "_y = torch.where(_xs > 0, _xs * (1 / (1 + _z)), ((_xs * _r) * _r) / _d)"]
  | ⟨_, _, .softplus⟩ =>
      -- Select the exponent's argument before calling exp: PyTorch may evaluate both tensor
      -- branches, so a selection between two exponentials would overflow.
      pure #[indent 4 s!"_positive = {x 0} > 0",
        indent 4 s!"_tail = torch.log1p(torch.exp(torch.where(_positive, -{x 0}, {x 0})))",
        indent 4 s!"{out} = torch.where(_positive, {x 0} + _tail, _tail)"]
  | ⟨_, _, .safeLog⟩ =>
      pure #[indent 4 s!"_positive = {x 0} > 0",
        indent 4 s!"_tail = torch.log1p(torch.exp(torch.where(_positive, -{x 0}, {x 0})))",
        indent 4 s!"_softplus = torch.where(_positive, {x 0} + _tail, _tail)",
        indent 4 s!"{out} = torch.log(_softplus + {x 1})"]
  | ⟨_, _, .sqrt .raw⟩ => pure (call "torch.sqrt")
  | ⟨_, _, .sqrt .clampNegative⟩ =>
      pure (line s!"torch.sqrt(torch.where({x 0} <= 0, torch.zeros_like({x 0}), {x 0}))")
  | ⟨_, _, .maskedSoftmax d⟩ =>
      pure #[indent 4 s!"masked{id} = {x 0}.masked_fill(~{x 1}, float('-inf'))",
        indent 4 s!"probs{id} = torch.softmax(masked{id}, dim={d.val})",
        indent 4 <| s!"{out} = torch.where({x 1}.any(dim={d.val}, keepdim=True), probs{id}, " ++
          s!"torch.zeros_like(probs{id}))"]
  | ⟨_, _, .layerNorm (embedDim := e) (lead := lead) ..⟩ =>
      -- The specification's formula (`Tensor.layerNorm`): each centred row is divided by its scale,
      -- the maximum of its largest magnitude and `sqrt |epsilon|`, before it is squared. The
      -- variance of the scaled row centres it again.
      let d := lead.length
      let view := ", ".intercalate ((List.range (d + 1)).map fun a =>
        if a = d then toString e else "1")
      let epsilon := (literal 3).getD (x 3)
      pure #[indent 4 s!"_c{id} = {x 0} - {x 0}.mean(dim=({d},), keepdim=True)",
        indent 4 <| s!"_e{id} = torch.as_tensor({epsilon}, dtype=_c{id}.dtype, " ++
          s!"device=_c{id}.device)",
        indent 4 <| s!"_s{id} = torch.maximum(_c{id}.abs().amax(dim=({d},), keepdim=True), " ++
          s!"_e{id}.abs().sqrt())",
        indent 4 s!"_u{id} = _c{id} / _s{id}",
        indent 4 s!"_w{id} = _u{id} - _u{id}.mean(dim=({d},), keepdim=True)",
        indent 4 s!"_v{id} = (_w{id} * _w{id}).mean(dim=({d},), keepdim=True)",
        indent 4 <| s!"{out} = _u{id} / torch.sqrt(torch.clamp(_v{id} + _e{id} / _s{id} / " ++
          s!"_s{id}, min=0)) * {x 1}.reshape(({view},)) + {x 2}.reshape(({view},))"]
  | ⟨_, _, .batchNormRunning (s := s) d⟩ =>
      pure #[indent 4 s!"_x = {x 0}",
        indent 4 s!"_view = [1] * {s.rank}",
        indent 4 s!"_view[{d.val}] = {s.extent d}",
        indent 4 s!"_mean = {x 3}.reshape(_view)",
        indent 4 s!"_var = {x 4}.reshape(_view)",
        indent 4 s!"_gamma = {x 1}.reshape(_view)",
        indent 4 s!"_beta = {x 2}.reshape(_view)",
        indent 4 s!"_eps = {(literal 5).getD (x 5)}",
        -- `Tensor.batchNorm`: the variance clamped at zero, plus epsilon, clamped at zero
        -- again under the root, each clamp the comparison `v <= 0` (it maps -0 to +0).
        indent 4 "_var = torch.where(_var <= 0, torch.zeros_like(_var), _var) + _eps",
        indent 4 "_std = torch.sqrt(torch.where(_var <= 0, torch.zeros_like(_var), _var))",
        indent 4 s!"{out} = (_x - _mean) / _std * _gamma + _beta"]
  | ⟨_, _, .conv (s := s) (k := k) (r := r) dims stride padLow padHigh dilation groups
      (lead := lead) ..⟩ =>
      let channelsFirst := (List.finRange (r + 2)).all fun p =>
        (dims.input p).val == p.val && (dims.kernel p).val == p.val
      -- PyTorch rejects a zero dilation and a spatial axis without a window position, where the
      -- output extent is zero: an empty kernel, a zero stride or a padded input shorter than the
      -- dilated kernel.
      let windows := (List.finRange r).all fun i =>
        0 < dilation i && 0 < Shape.slidingWindowOutDimDilated (s.getD (i.val + 2) 0)
          (k.getD (i.val + 2) 0) (stride i) (dilation i) (padLow i) (padHigh i)
      if !channelsFirst || groups ≠ 1 || r = 0 || r > 3 || s.getD 1 0 = 0 then
        refuse s!"a convolution other than channels first with one group, input channels and \
          one to three spatial axes has no PyTorch translation"
      else if k.getD 0 0 = 0 || !windows then
        refuse "a convolution without output channels, with a zero dilation or without a window \
          position along a spatial axis has no PyTorch translation"
      else
        let inC := s.getD 1 0
        let outC := k.getD 0 0
        let pads := ((List.ofFn padLow).zip (List.ofFn padHigh)).reverse.flatMap
          fun (lo, hi) => [toString lo, toString hi]
        pure #[indent 4 s!"_x = {x 2}",
          indent 4 s!"_prefix = list(_x.shape[:{lead.length}])",
          indent 4 s!"_spatial = list(_x.shape[{lead.length + 1}:])",
          -- The batch extent is explicit, since `-1` is ambiguous for an input without entries.
          indent 4 s!"_x = _x.reshape((_x.shape[:{lead.length}].numel(), {inC}, *_spatial))",
          indent 4 s!"_x = F.pad(_x, ({", ".intercalate pads}))",
          indent 4 <| s!"_y = F.conv{r}d(_x, {x 0}, {x 1}, " ++
            s!"stride={shapeLiteral (List.ofFn stride)}, padding=0, " ++
            s!"dilation={shapeLiteral (List.ofFn dilation)}, groups=1)",
          indent 4 s!"{out} = _y.reshape((*_prefix, {outC}, *_y.shape[2:]))"]
  | ⟨_, _, .pool (s := s) pop dims kernel stride padLow padHigh dilation ceilMode⟩ =>
      let r := dims.length
      let trailing := dims.map (·.val) == List.range' (s.rank - r) r
      match poolFunction? pop r with
      | some (fn, extra) =>
          -- `avg_pool3d` rejects an input extent below the kernel's, even where the padded
          -- windows are valid.
          let small := r == 3 && (pop matches .avg _) &&
            ((List.range' (s.rank - r) r).zip kernel).any fun (a, k) => s.getD a 0 < k
          if trailing && padLow == padHigh && dilation.all (· == 1) && !ceilMode && !small then
            pure #[indent 4 s!"_x = {x 0}",
              indent 4 s!"_prefix = list(_x.shape[:-{r}])",
              indent 4 s!"_spatial = list(_x.shape[-{r}:])",
              indent 4 "_x = _x.reshape((-1, 1, *_spatial))",
              indent 4 <| s!"_y = F.{fn}(_x, kernel_size={shapeLiteral kernel}, " ++
                s!"stride={shapeLiteral stride}, padding={shapeLiteral padLow}{extra})",
              indent 4 s!"{out} = _y.reshape((*_prefix, *_y.shape[2:]))"]
          else refuse "a pooling other than over trailing axes with symmetric padding, unit \
            dilation, floor mode and, for a three-axis average, input extents at least the \
            kernel's has no PyTorch translation"
      | none => refuse "this pooling has no PyTorch function"
  | ⟨_, _, .reduce (s := s) rop dims keepdim _⟩ =>
      let listed := ", ".intercalate (dims.map fun d => toString d.val)
      let over := s!"dim=({listed},), keepdim={boolLiteral keepdim}"
      -- `torch.sum` over an empty tuple of dimensions would reduce every dimension.
      if dims.isEmpty then
        pure <| line <|
          match rop with
          | .sum => s!"torch.zeros_like({x 0}) + {x 0}"
          | .prod => s!"torch.ones_like({x 0}) * {x 0}"
          | .max | .min => x 0
      else
        match rop with
        | .sum => pure (line s!"torch.sum({x 0}, {over})")
        | .max =>
            pure (extremum s!"torch.amax({x 0}, {over})" s!"torch.signbit({x 0}).all({over})")
        | .min =>
            pure (extremum s!"torch.amin({x 0}, {over})" s!"torch.signbit({x 0}).any({over})")
        | .prod =>
            -- `torch.prod` regroups the factors, so a partial product can overflow or underflow
            -- where `Tensor.reduce`'s row-major left fold from `1` doesn't.
            let others := s.otherAxes dims
            let reduced := s.sortedAxes dims
            let lead := others.map s.extent
            let n := (reduced.map s.extent).foldl (· * ·) 1
            let order := shapeLiteral ((others ++ reduced).map (·.val))
            pure #[indent 4 <| s!"_f{id} = {x 0}.permute({order})" ++
                s!".reshape({shapeLiteral (lead ++ [n])})",
              indent 4 <| s!"_p{id} = torch.ones({shapeLiteral lead}, dtype={x 0}.dtype, " ++
                s!"device={x 0}.device)",
              indent 4 s!"for _j in range({n}):",
              indent 6 s!"_p{id} = _p{id} * _f{id}[..., _j]",
              indent 4 s!"{out} = _p{id}.reshape({shapeLiteral (s.reduce dims keepdim)})"]
  | ⟨_, _, .mean (s := s) dims keepdim⟩ =>
      let listed := ", ".intercalate (dims.map fun d => toString d.val)
      let over := s!"dim=({listed},), keepdim={boolLiteral keepdim}"
      -- `Tensor.mean` divides an empty fiber's sum `+0` by one, where `torch.mean` gives a NaN.
      pure <| line <|
        if dims.isEmpty then s!"torch.zeros_like({x 0}) + {x 0}"
        else if dims.any (s.extent · == 0) then s!"torch.sum({x 0}, {over})"
        else s!"torch.mean({x 0}, {over})"
  | ⟨_, _, .broadcast (s := s) (t := t) dims _⟩ =>
      let view := (List.range t.rank).map fun j =>
        match (List.finRange s.rank).find? fun i => (dims i).val = j with
        | some i => toString (s.extent i)
        | none => "1"
      pure (line s!"{x 0}.reshape({tuple view}).expand({shapeLiteral t})")
  | ⟨_, _, .slice (s := s) dims starts stops steps _⟩ =>
      let entries := (List.range s.rank).map
        (sliceEntry (dims.map (·.val)) starts stops steps)
      -- At rank 0 the index is `()`, as `x[,]` is not Python.
      let index := if entries.isEmpty then "()" else s!"{", ".intercalate entries},"
      pure (line s!"{x 0}[{index}]")
  | ⟨_, _, .pad (s := s) low high interior⟩ =>
      let windows := (List.finRange s.rank).map fun d =>
        let lo := low d
        let step := interior d + 1
        -- The stop `lo + n * step`, for the axis's extent `n`, ends the slice after its last
        -- entry and selects none at `n = 0`. Python clamps a stop past the end.
        s!"slice({lo}, {lo} + {x 0}.shape[{d.val}] * {step}, {step})"
      -- The fill is a copy of `value`, so that `value` receives the upstream gradient of the
      -- entries the input doesn't overwrite, as `Op.pad`'s pullback gives it.
      pure #[indent 4 s!"{out} = {x 1}.expand({shapeLiteral (s.pad low high interior)}).clone()",
        indent 4 s!"{out}[{tuple windows}] = {x 0}"]
  | ⟨_, _, .gather (s := s) (t := t) dims batchDims _⟩ =>
      let rank := s.rank
      let q := t.rank
      let used := dims.map (·.val) ++ batchDims.map (·.1.val)
      let rest := (List.range rank).filter (· ∉ used)
      let ones (count : Nat) := ", ".intercalate (List.replicate count "1")
      let entry (a : Nat) : String :=
        match (dims.map (·.val)).idxOf? a with
        | some k => s!"_idx{id}[..., {k}].reshape((*_idx{id}.shape[:-1], {ones rest.length}))"
        | none =>
            match batchDims.find? (·.1.val = a) with
            | some (_, b) =>
                let view := (List.range (q + rest.length)).map fun j =>
                  if j = b.val then toString (t.extent b) else "1"
                s!"torch.arange({t.extent b}, device={x 0}.device)" ++
                  s!".reshape(({", ".intercalate view},))"
            | none =>
                let m := rest.idxOf a
                let view := (List.range (q + rest.length)).map fun j =>
                  if j = q + m then toString (s.getD a 0) else "1"
                s!"torch.arange({s.getD a 0}, device={x 0}.device)" ++
                  s!".reshape(({", ".intercalate view},))"
      let index := if dims.length = 1 then s!"{x 1}.unsqueeze(-1)" else x 1
      let read := s!"{x 0}[{tuple ((List.range rank).map entry)}]"
      -- Without indexed axes only the batch entries carry index axes, so the read has extent 1 on
      -- the other index axes and is expanded to the result's shape.
      let result :=
        if dims.isEmpty then
          s!"{read}.expand({shapeLiteral (t ++ s.eraseAxes (dims ++ batchDims.map Prod.fst))})"
        else read
      pure #[indent 4 s!"_idx{id} = {index}", indent 4 s!"{out} = {result}"]
  | ⟨_, _, .scatter (s := s) (t := t) rule dims _⟩ =>
      let rank := s.rank
      let q := t.rank
      let rest := (List.range rank).filter (· ∉ dims.map (·.val))
      let ones (count : Nat) := ", ".intercalate (List.replicate count "1")
      let entry (a : Nat) : String :=
        match (dims.map (·.val)).idxOf? a with
        | some k => s!"_idx{id}[..., {k}].reshape((*_idx{id}.shape[:-1], {ones rest.length}))"
        | none =>
            let m := rest.idxOf a
            let view := (List.range (q + rest.length)).map fun j =>
              if j = q + m then toString (s.getD a 0) else "1"
            s!"torch.arange({s.getD a 0}, device={x 0}.device)" ++
              s!".reshape(({", ".intercalate view},))"
      let index := if dims.length = 1 then s!"{x 1}.unsqueeze(-1)" else x 1
      -- index_put writes or adds. A product, maximum or minimum has no index_put form. It rejects
      -- an empty index tuple, and without indexed axes the updates of a nonscalar index tensor
      -- don't broadcast to the entries they write.
      match rule with
      | .replace | .sum =>
          if dims.isEmpty && (rank == 0 || q != 0) then
            refuse "a scatter without indexed axes at rank 0 or from a nonscalar index tensor is \
              not exported"
          else
            let accumulate := match rule with
              | .sum => "True"
              | _ => "False"
            pure #[indent 4 s!"_idx{id} = {index}",
              indent 4 <| s!"{out} = {x 0}.index_put({tuple ((List.range rank).map entry)}, " ++
                s!"{x 2}, accumulate={accumulate})"]
      | _ => refuse "a scatter product, maximum or minimum is not exported"
  | ⟨_, _, .contract (s := s) (t := t) dims batchDims _⟩ =>
      -- One einsum letter per axis: batch pairs and contracted pairs share a letter.
      let letters := "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ".toList
      let pairs := (batchDims ++ dims).map fun (i, j) => (i.val, j.val)
      let rightLetter (j : Nat) : Char :=
        match pairs.find? (·.2 = j) with
        | some (i, _) => letters.getD i 'a'
        | none => letters.getD (s.rank + j) 'a'
      let leftLetters := String.ofList ((List.range s.rank).map (letters.getD · 'a'))
      let rightLetters := String.ofList ((List.range t.rank).map rightLetter)
      let batchLetters := batchDims.map fun (i, _) => letters.getD i.val 'a'
      let leftFree := (List.range s.rank).filter fun i => ¬ pairs.any (·.1 = i)
      let rightFree := (List.range t.rank).filter fun j => ¬ pairs.any (·.2 = j)
      let outLetters := String.ofList (batchLetters ++ leftFree.map (letters.getD · 'a') ++
        rightFree.map rightLetter)
      -- Past the last letter `getD` gives `'a'`, a label already in use.
      if letters.length < s.rank || rightFree.any (letters.length ≤ s.rank + ·) then
        refuse "a contraction whose einsum labels run past the 52 letters is not exported"
      else
        pure (line s!"torch.einsum('{leftLetters},{rightLetters}->{outLetters}', {x 0}, {x 1})")
  | ⟨_, _, .indices s d⟩ =>
      let view := (List.finRange s.rank).map fun a =>
        if a = d then toString (s.extent d) else "1"
      pure (line <| s!"torch.arange({s.extent d}, device=x.device)" ++
        s!".reshape(({", ".intercalate view},)).expand({shapeLiteral s}).contiguous()")
  | ⟨_, _, .cast _⟩ => pure (line s!"{x 0}.to({dtype})")
  | ⟨_, _, .topKMask d k⟩ =>
      -- `MoE.topKMask`: along `dim`, entry j beats entry i when it is larger, or when
      -- neither is larger and j < i. An entry is kept when fewer than k entries beat it.
      pure #[indent 4 s!"_x{id} = {x 0}.movedim({d.val}, -1)",
        indent 4 s!"_xi{id} = _x{id}.unsqueeze(-1)",
        indent 4 s!"_xj{id} = _x{id}.unsqueeze(-2)",
        indent 4 s!"_ix{id} = torch.arange(_x{id}.shape[-1], device={x 0}.device)",
        indent 4 (s!"_beats{id} = (_xj{id} > _xi{id}) | " ++
          s!"((_ix{id}.unsqueeze(0) < _ix{id}.unsqueeze(1)) & ~(_xi{id} > _xj{id}))"),
        indent 4 <| s!"{out} = (_beats{id}.sum(-1) < {k}).to({x 0}.dtype)" ++
          s!".movedim(-1, {d.val})"]
  | ⟨_, _, .sort d descending⟩ =>
      pure (pair s!"torch.sort({x 0}, dim={d.val}, descending={boolLiteral descending}, \
        stable=True)")
  | ⟨_, _, .topk d k largest _⟩ =>
      -- `torch.topk` documents no tie order, and `Tensor.topk` is the first `k` entries of the
      -- stable sort.
      pure #[indent 4 <| s!"_v{id}, _i{id} = torch.sort({x 0}, dim={d.val}, " ++
          s!"descending={boolLiteral largest}, stable=True)",
        indent 4 <| s!"{out}, {out}_1 = _v{id}.narrow({d.val}, 0, {k}), " ++
          s!"_i{id}.narrow({d.val}, 0, {k})"]
  | ⟨_, _, .cummax d⟩ => pure (pair s!"torch.cummax({x 0}, dim={d.val})")
  | ⟨_, _, .cummin d⟩ => pure (pair s!"torch.cummin({x 0}, dim={d.val})")
  | ⟨_, _, .randUniform ..⟩ => pure (line s!"self.draws_{id}")
  | ⟨_, _, .bernoulliMask ..⟩ =>
      -- `Random.keepBit`: zero unless the probability is above zero, one from one up, and
      -- otherwise one where the probability is above the draw.
      pure (line <| s!"torch.where({x 0} > 0, torch.where({x 0} < 1, " ++
        s!"({x 0} > self.draws_{id}).to({dtype}), 1), 0)")
  | ⟨_, _, .scaledMatmul ..⟩ =>
      -- The product, then the scale, as in the reference `scaledProductsSlice`. The reference
      -- sums each dot product left to right (`torch.matmul` fixes no order) and, on finite
      -- inputs, replays the exact coefficients when an intermediate leaves the range or a nonzero
      -- product is absorbed by the running sum. This line does not replay.
      pure (line s!"torch.matmul({x 0}, {x 1}) * {x 2}")
  | ⟨_, _, .custom name ..⟩ => refuse s!"custom {name} has no Python body translation"
  | o' => refuse s!"{(Op.encode o').tag} has no PyTorch translation"

/-- The literal of input `k` of a node reading `refs`, when it is a scalar constant. -/
private def scalarArg (g : Graph) (P : Payload Float) (refs : Array Ref) (k : Nat) :
    Option String :=
  match refs[k]? with
  | some r =>
      match g.nodes[r.node]?, r.out with
      | some (.const _), 0 => (P.const? r.node).bind scalarLiteral?
      | _, _ => none
  | none => none

/-- The body of `forward`: graph input `0` is `x`, a constant that `bindings` binds its attribute,
each operator node its row of `opLines`, then the return of `output`. -/
private def forward (g : Graph) (P : Payload Float) (bound : Std.HashSet Nat) (output : Ref)
    (dtype : String) : Except String (Array String) := do
  let mut lines : Array String := #[]
  for i in [0:g.nodes.size] do
    match g.nodes[i]? with
    | some (.input 0 _) => lines := lines.push (indent 4 s!"v{i} = x")
    | some (.input index _) =>
        throw s!"IR→PyTorch: node {i}: graph input {index} has no argument; forward takes one \
          tensor"
    | some (.const _) =>
        if bound.contains i then lines := lines.push (indent 4 s!"v{i} = self.const_{i}")
    | some (.op o refs) =>
        unless refs.size = o.1.length do
          throw s!"IR→PyTorch: node {i}: {(Op.encode o).tag} cannot take {refs.size} inputs"
        lines := lines ++ (← opLines i dtype (scalarArg g P refs) (refs.map name) o)
    | none => pure ()
  pure (lines.push (indent 4 s!"return {name output}"))

/--
Emit a standalone PyTorch `nn.Module` class for the operator graph of an import: imports, a class
whose `__init__` sets the constants, a `forward` method for the graph, and, when
`includeTrainingSkeleton` holds, a compact training script.
-/
def emit (imported : IR.Imported Float) (options : Options := {}) : Except String String := do
  let g := imported.graph
  let some output := g.outputs[0]? | throw "IR→PyTorch: the graph has no output"
  let some inputPort := g.nodes.findSome? (fun | .input 0 p => some p | _ => none)
    | throw "IR→PyTorch: the graph has no input 0"
  let some outputPort := (g.nodes[output.node]?).bind (·.outs[output.out]?)
    | throw s!"IR→PyTorch: no output {output.out} of node {output.node}"
  let inputShape := inputPort.shape
  let outputShape := outputPort.shape

  let (bound, initParamLines) ← bindings g imported.payload options output
  let forwardBody ← forward g imported.payload bound output options.dtypeExpr

  let imports : Array String :=
    #[ "import torch"
    , "import torch.nn as nn"
    , "import torch.nn.functional as F"
    , ""
    ]

  let classHeader : Array String :=
    #[ s!"class {options.className}(nn.Module):"
    , indent 2 "def __init__(self):"
    , indent 4 "super().__init__()"
    , indent 4 s!"dtype = {options.dtypeExpr}"
    ]

  let classForwardHeader : Array String :=
    #[ ""
    , indent 2 "def forward(self, x):"
    ]

  let classLines : Array String :=
    classHeader ++ initParamLines ++ classForwardHeader ++ forwardBody

  let helpers : Array String :=
    if !options.includeTrainingSkeleton then
      #[]
    else
      let outIsLoss : Bool :=
        match g.nodes[output.node]? with
        | some (.op ⟨_, _, .mseLoss⟩ _) => true
        | _ => false
      let xTuple := shapeLiteral inputShape
      let yTuple := shapeLiteral outputShape
      let (trainStepDef, mainLines) :=
        if outIsLoss then
          ( #[ ""
            , "def train_step(model: nn.Module, x: torch.Tensor, opt=None):"
            , indent 2 "model.train()"
            , indent 2 "if opt is not None:"
            , indent 4 "opt.zero_grad(set_to_none=True)"
            , indent 2 "loss = model(x)"
            , indent 2 "if opt is not None:"
            , indent 4 "loss.backward()"
            , indent 4 "opt.step()"
            , indent 2 "return float(loss.detach().cpu())"
            ]
          , #[ ""
            , "if __name__ == '__main__':"
            , indent 2 "torch.manual_seed(0)"
            , indent 2 "device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')"
            , indent 2 s!"model = {options.className}().to(device)"
            , indent 2 "opt = make_optimizer(model, kind='adam', lr=1e-3)"
            , indent 2 s!"x = torch.randn({xTuple}, dtype={options.dtypeExpr}, device=device)"
            , indent 2 "loss = train_step(model, x, opt)"
            , indent 2 "print('loss', loss)"
            ] )
        else
          ( #[ ""
            , "def train_step(model: nn.Module, x: torch.Tensor, y: torch.Tensor, opt=None):"
            , indent 2 "model.train()"
            , indent 2 "if opt is not None:"
            , indent 4 "opt.zero_grad(set_to_none=True)"
            , indent 2 "out = model(x)"
            , indent 2 "loss = F.mse_loss(out, y, reduction='mean')"
            , indent 2 "if opt is not None:"
            , indent 4 "loss.backward()"
            , indent 4 "opt.step()"
            , indent 2 "return float(loss.detach().cpu())"
            ]
          , #[ ""
            , "if __name__ == '__main__':"
            , indent 2 "torch.manual_seed(0)"
            , indent 2 "device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')"
            , indent 2 s!"model = {options.className}().to(device)"
            , indent 2 "opt = make_optimizer(model, kind='adam', lr=1e-3)"
            , indent 2 s!"x = torch.randn({xTuple}, dtype={options.dtypeExpr}, device=device)"
            , indent 2 s!"y = torch.randn({yTuple}, dtype={options.dtypeExpr}, device=device)"
            , indent 2 "loss = train_step(model, x, y, opt)"
            , indent 2 "print('loss', loss)"
            ] )

      #[ ""
      , "def make_optimizer(model: nn.Module, kind: str = 'adam', lr: float = 1e-3):"
      , indent 2 "params = [p for p in model.parameters() if p.requires_grad]"
      , indent 2 "if len(params) == 0:"
      , indent 4 "return None"
      , indent 2 "if kind == 'sgd':"
      , indent 4 "return torch.optim.SGD(params, lr=lr)"
      , indent 2 "if kind == 'adam':"
      , indent 4 "return torch.optim.Adam(params, lr=lr)"
      , indent 2 "raise ValueError(f'unknown optimizer kind: {kind}')"
      ] ++ trainStepDef ++ mainLines

  pure (joinLines (imports ++ classLines ++ helpers))

end PyTorch
end TorchLean
