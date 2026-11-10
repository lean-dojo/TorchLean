/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.IR.DType
public import TorchLean.NN.IR.Graph

/-!
# The pointwise operators as fragments of registered operators

`fragment? op s` is the fragment of registered operators (`TorchLean.Op`) that computes the
pointwise operator `op` (`PointwiseOp`) at the shape `s` on operands at the scalar. Its nodes
come in evaluation order, each reading operands, the zero tensor or an earlier node (`Arg`), and its
value is the first output of the last node. `Fragment.nodes` places a fragment in a graph of
registered operators. The word operators, `cbrt` and `nextafter` have no fragment at the scalar.

`form? op operand d s` is the fragment of a pointwise node with operands of element type `operand`
and a result of element type `d`, its steps at the element of their type (`element?`,
`TorchLean.Op.atElement`). Beyond `fragment?` it has the logical operators of Booleans (`logical?`)
and the word operators, `cbrt` and `nextafter` at their element types (`typed?`). The op table that
the Python captures read (`TorchLean.IR.table`) lists its forms. `typed?_isSome_of_integerOnly`
shows that every word operator has a registered operator at every integer type of known width.

`map_indices_eq_fwd`, `pad_eq_fwd` and `contract_eq_fwd` state that `Tensor.indices` read at the
scalar, `Tensor.pad` and `Tensor.contract` are the forwards (`fwd`) of `cast` after `indices`, of
`pad` and of `contract`. No theorem here relates a fragment to the PyTorch operator it stands for:
`TorchLean.Test.Pointwise` checks typed artifacts with one pointwise node against reference words,
most from eager PyTorch.
-/

@[expose] public section

namespace TorchLean.IR.Pointwise

/-! ## The forwards of `indices`, `pad` and `contract` -/

section Forwards

variable {α : Type} [Storage α] [Context α] [Op.Capabilities α] {s : Shape}

/-- The forward of an operator whose rule exists at every scalar with `Context`. -/
abbrev fwd {ins outs : List Port} (op : Op ins outs) (xs : Values α ins)
    (h : (op.rule? (α := α)).isSome := by rfl) : Values α outs :=
  ((op.rule? (α := α)).get h).forward xs

/-- `Tensor.indices` read at the scalar is the forward of `Op.cast` after `Op.indices`. -/
theorem map_indices_eq_fwd (shape : Shape) (d : Fin shape.rank) :
    Tensor.map (fun j => ((j.val : Nat) : α)) (Tensor.indices shape d) =
      (fwd (Op.cast (s := shape) (.index (shape.extent d)))
        ((fwd (α := α) (Op.indices shape d) ()).1, ())).1 :=
  rfl

/-- `Tensor.pad` padding with the entry of `value` is the forward of `Op.pad`. -/
theorem pad_eq_fwd (x : Tensor α s) (low high interior : Fin s.rank → Nat) (value : Tensor α []) :
    Tensor.pad x low high interior value.item = (fwd (Op.pad low high interior) (x, value, ())).1 :=
  rfl

/-- `Tensor.contract` is the forward of `Op.contract`. -/
theorem contract_eq_fwd {t : Shape} (x : Tensor α s) (y : Tensor α t)
    (dims batchDims : List (Fin s.rank × Fin t.rank)) (h : Shape.ContractValid s t dims batchDims) :
    Tensor.contract x y dims batchDims _ h rfl =
      (fwd (Op.contract dims batchDims h) (x, y, ())).1 :=
  rfl

end Forwards

/-! ## The table -/

/-- Where an input of a fragment's node reads from: operand `k` of the pointwise operator, the
zero tensor at the operands' shape, or the first output of node `j` of the fragment. -/
inductive Arg where
  /-- Operand `k` of the pointwise operator. -/
  | operand (k : Nat)
  /-- The zero tensor at the operands' shape. -/
  | zero
  /-- The first output of node `j` of the fragment. -/
  | node (j : Nat)
  deriving DecidableEq, Repr, Inhabited

/-- A fragment lists registered operators in evaluation order, each with one argument per input
port, and its value is the first output of the last one. -/
abbrev Fragment := List (Operator × List Arg)

/-- The fragment of the pointwise operator `op` at the shape `s`, on operands and with a result at
the scalar, or `none` for the word operators, `cbrt` and `nextafter`, which have registered
operators only at an integer type or a binary format.

`Op.cast .bool` reads a Boolean result back as the scalar. The condition of `select` and the
operands of `logicalAnd`, `logicalOr` and `logicalXor` are read as `Op.ne` with zero, and the
operand of `logicalNot` as `Op.eq` with zero. -/
def fragment? (op : PointwiseOp) (s : Shape) : Option Fragment :=
  let unary := fun (o : Operator) => some [(o, [Arg.operand 0])]
  let binary := fun (o : Operator) => some [(o, [Arg.operand 0, .operand 1])]
  let cast : Operator := ⟨_, _, Op.cast (s := s) .bool⟩
  let ne : Operator := ⟨_, _, Op.ne (s := s)⟩
  let select : Operator := ⟨_, _, Op.select (s := s)⟩
  let predicate := fun (o : Operator) => some [(o, [Arg.operand 0]), (cast, [.node 0])]
  let comparison := fun (o : Operator) =>
    some [(o, [Arg.operand 0, .operand 1]), (cast, [.node 0])]
  let logical := fun (o : Operator) =>
    some [(ne, [Arg.operand 0, .zero]), (ne, [.operand 1, .zero]), (o, [.node 0, .node 1]),
      (cast, [.node 2])]
  match op with
  | .neg => unary ⟨_, _, Op.neg (s := s)⟩
  | .sign => unary ⟨_, _, Op.sign (s := s)⟩
  | .square => unary ⟨_, _, Op.square (s := s)⟩
  | .exp2 => unary ⟨_, _, Op.exp2 (s := s)⟩
  | .expm1 => unary ⟨_, _, Op.expm1 (s := s)⟩
  | .log2 => unary ⟨_, _, Op.log2 (s := s)⟩
  | .log10 => unary ⟨_, _, Op.log10 (s := s)⟩
  | .log1p => unary ⟨_, _, Op.log1p (s := s)⟩
  | .tan => unary ⟨_, _, Op.tan (s := s)⟩
  | .sinh => unary ⟨_, _, Op.sinh (s := s)⟩
  | .cosh => unary ⟨_, _, Op.cosh (s := s)⟩
  | .asinh => unary ⟨_, _, Op.asinh (s := s)⟩
  | .acosh => unary ⟨_, _, Op.acosh (s := s)⟩
  | .atanh => unary ⟨_, _, Op.atanh (s := s)⟩
  | .rsqrt => unary ⟨_, _, Op.rsqrt (s := s)⟩
  | .erf => unary ⟨_, _, Op.erf (s := s)⟩
  | .atan => unary ⟨_, _, Op.atan (s := s)⟩
  | .asin => unary ⟨_, _, Op.asin (s := s)⟩
  | .acos => unary ⟨_, _, Op.acos (s := s)⟩
  | .floor => unary ⟨_, _, Op.floor (s := s)⟩
  | .ceil => unary ⟨_, _, Op.ceil (s := s)⟩
  | .round => unary ⟨_, _, Op.round (s := s)⟩
  | .trunc => unary ⟨_, _, Op.trunc (s := s)⟩
  | .frac => unary ⟨_, _, Op.frac (s := s)⟩
  | .gelu a => unary ⟨_, _, Op.gelu (s := s) a⟩
  | .div r => binary ⟨_, _, Op.div (s := s) r⟩
  | .pow => binary ⟨_, _, Op.pow (s := s)⟩
  | .atan2 => binary ⟨_, _, Op.atan2 (s := s)⟩
  | .fmod => binary ⟨_, _, Op.fmod (s := s)⟩
  | .remainder => binary ⟨_, _, Op.remainder (s := s)⟩
  | .hypot => binary ⟨_, _, Op.hypot (s := s)⟩
  | .copysign => binary ⟨_, _, Op.copysign (s := s)⟩
  | .maximum => binary ⟨_, _, Op.max (s := s)⟩
  | .minimum => binary ⟨_, _, Op.min (s := s)⟩
  | .fmax => binary ⟨_, _, Op.fmax (s := s)⟩
  | .fmin => binary ⟨_, _, Op.fmin (s := s)⟩
  | .lerp => some [(⟨_, _, Op.lerp (s := s)⟩, [.operand 0, .operand 1, .operand 2])]
  | .isNaN => predicate ⟨_, _, Op.isNaN (s := s)⟩
  | .isInf => predicate ⟨_, _, Op.isInf (s := s)⟩
  | .isFinite => predicate ⟨_, _, Op.isFinite (s := s)⟩
  | .eq => comparison ⟨_, _, Op.eq (s := s)⟩
  | .ne => comparison ne
  | .lt => comparison ⟨_, _, Op.lt (s := s)⟩
  | .le => comparison ⟨_, _, Op.le (s := s)⟩
  | .gt => comparison ⟨_, _, Op.gt (s := s)⟩
  | .ge => comparison ⟨_, _, Op.ge (s := s)⟩
  | .logicalNot => some [(⟨_, _, Op.eq (s := s)⟩, [.operand 0, .zero]), (cast, [.node 0])]
  | .logicalAnd => logical ⟨_, _, Op.logicalAnd (s := s)⟩
  | .logicalOr => logical ⟨_, _, Op.logicalOr (s := s)⟩
  | .logicalXor => logical ⟨_, _, Op.logicalXor (s := s)⟩
  | .select => some [(ne, [.operand 0, .zero]), (select, [.node 0, .operand 1, .operand 2])]
  | .clamp => some [(⟨_, _, Op.max (s := s)⟩, [.operand 0, .operand 1]),
      (⟨_, _, Op.min (s := s)⟩, [.node 0, .operand 2])]
  | .addcmul => some [(⟨_, _, Op.mul (s := s)⟩, [.operand 3, .operand 1]),
      (⟨_, _, Op.mul (s := s)⟩, [.node 0, .operand 2]),
      (⟨_, _, Op.add (s := s)⟩, [.operand 0, .node 1])]
  | .addcdiv => some [(⟨_, _, Op.div (s := s)⟩, [.operand 1, .operand 2]),
      (⟨_, _, Op.mul (s := s)⟩, [.operand 3, .node 0]),
      (⟨_, _, Op.add (s := s)⟩, [.operand 0, .node 1])]
  | .nanToNum => some [(⟨_, _, Op.isNaN (s := s)⟩, [.operand 0]),
      (⟨_, _, Op.isInf (s := s)⟩, [.operand 0]), (⟨_, _, Op.gt (s := s)⟩, [.operand 0, .zero]),
      (select, [.node 2, .operand 2, .operand 3]), (select, [.node 1, .node 3, .operand 0]),
      (select, [.node 0, .operand 1, .node 4])]
  | .clampMin => some [(⟨_, _, Op.gt (s := s)⟩, [.operand 1, .operand 0]),
      (⟨_, _, Op.eq (s := s)⟩, [.operand 1, .operand 1]),
      (select, [.node 1, .operand 0, .operand 1]), (select, [.node 0, .operand 1, .node 2])]
  | .cbrt | .nextafter | .bitwiseAnd | .bitwiseOr | .bitwiseXor | .bitwiseNot | .bitwiseLeftShift
  | .shiftRightArithmetic | .shiftRightLogical | .populationCount | .countLeadingZeros | .mulhi =>
      none

/-- Whether a node of the fragment reads the zero tensor. -/
def Fragment.usesZero (f : Fragment) : Bool :=
  f.any fun (_, args) => args.contains .zero

/-- The nodes of fragment `f` with its first node at position `base`, operand `k` read at
`operands[k]` and the zero tensor at `zero`, or `none` if an operand is missing. -/
def Fragment.nodes (f : Fragment) (operands : List Ref) (zero : Ref) (base : Nat) :
    Option (List Node) :=
  f.mapM fun (o, args) => do
    let refs ← args.mapM fun
      | .operand k => operands[k]?
      | .zero => some zero
      | .node j => some ⟨base + j, 0⟩
    pure (.op o refs.toArray)

/-! ## Forms at element types -/

/-- Whether `op` acts on the words of integers only: the bitwise operators, the shifts, the
bit counts and `mulhi`. -/
def integerOnly : PointwiseOp → Bool
  | .bitwiseAnd | .bitwiseOr | .bitwiseXor | .bitwiseNot | .bitwiseLeftShift
    | .shiftRightArithmetic | .shiftRightLogical | .populationCount | .countLeadingZeros
    | .mulhi => true
  | _ => false

/-- Whether PyTorch gives `op` a Boolean result: the comparisons `eq ne lt le gt ge`, the
floating-point class predicates `isnan isinf isfinite` and the logical operators. -/
def returnsBool : PointwiseOp → Bool
  | .eq | .ne | .lt | .le | .gt | .ge | .isNaN | .isInf | .isFinite => true
  | .logicalNot | .logicalAnd | .logicalOr | .logicalXor => true
  | _ => false

/-- The registered operator of a logical operator of Booleans at the shape `s`, and `none` for
every other operator. -/
def logical? (op : PointwiseOp) (s : Shape) : Option Operator :=
  match op with
  | .logicalNot => some ⟨_, _, Op.logicalNot (s := s)⟩
  | .logicalAnd => some ⟨_, _, Op.logicalAnd (s := s)⟩
  | .logicalOr => some ⟨_, _, Op.logicalOr (s := s)⟩
  | .logicalXor => some ⟨_, _, Op.logicalXor (s := s)⟩
  | _ => none

/-- The registered operator of a typed pointwise node of `op` whose operands and result have the
element type `dtype`, at the shape `s`: a word operator at an integer type of known width, `cbrt`
or `nextafter` at a binary format, and `none` for every other operator and type. -/
def typed? (op : PointwiseOp) (dtype : IR.DType) (s : Shape) : Option Operator :=
  match op, dtype with
  | .bitwiseAnd, .integer signed (some width) =>
      some ⟨_, _, Op.bitwiseAnd (signed := signed) (width := width) (s := s)⟩
  | .bitwiseOr, .integer signed (some width) =>
      some ⟨_, _, Op.bitwiseOr (signed := signed) (width := width) (s := s)⟩
  | .bitwiseXor, .integer signed (some width) =>
      some ⟨_, _, Op.bitwiseXor (signed := signed) (width := width) (s := s)⟩
  | .bitwiseNot, .integer signed (some width) =>
      some ⟨_, _, Op.bitwiseNot (signed := signed) (width := width) (s := s)⟩
  | .bitwiseLeftShift, .integer signed (some width) =>
      some ⟨_, _, Op.bitwiseLeftShift (signed := signed) (width := width) (s := s)⟩
  | .shiftRightArithmetic, .integer signed (some width) =>
      some ⟨_, _, Op.shiftRightArithmetic (signed := signed) (width := width) (s := s)⟩
  | .shiftRightLogical, .integer signed (some width) =>
      some ⟨_, _, Op.shiftRightLogical (signed := signed) (width := width) (s := s)⟩
  | .populationCount, .integer signed (some width) =>
      some ⟨_, _, Op.populationCount (signed := signed) (width := width) (s := s)⟩
  | .countLeadingZeros, .integer signed (some width) =>
      some ⟨_, _, Op.countLeadingZeros (signed := signed) (width := width) (s := s)⟩
  | .mulhi, .integer signed (some width) =>
      some ⟨_, _, Op.mulhi (signed := signed) (width := width) (s := s)⟩
  | .cbrt, .binary _ => some ⟨_, _, Op.cbrt (s := s)⟩
  | .nextafter, .binary _ => some ⟨_, _, Op.nextafter (s := s)⟩
  | _, _ => none

/-- Every word operator of `integerOnly` has a registered operator (`typed?`) at every integer
type of known width. -/
theorem typed?_isSome_of_integerOnly {op : PointwiseOp} (h : integerOnly op)
    (signed : Bool) (width : Nat) (s : Shape) :
    (typed? op (.integer signed (some width)) s).isSome := by
  cases op <;> first | rfl | simp [integerOnly] at h

/-- The fragment `f` without its last node where that node is `Op.cast .bool`, which leaves the
Booleans that a comparison, class predicate or logical fragment casts to the scalar, and `none`
otherwise. -/
def Fragment.boolean? (f : Fragment) : Option Fragment :=
  match f.getLast? with
  | some (⟨_, _, .cast .bool⟩, _) => some f.dropLast
  | _ => none

/-- The element of the data ports of an operator at the element type `d`: Booleans, an integer
type of positive width, or the scalar at a scalar element type, and `none` otherwise. -/
def element? (d : IR.DType) : Option Op.Element :=
  match d with
  | .bool => some .bool
  | .integer signed (some width) => if 0 < width then some (.integer signed width) else none
  | d => if d.isScalar then some .scalar else none

/-- The fragment `f` with each operator at the element `e` (`TorchLean.Op.atElement`), and `none`
where one has no form there. -/
def fragmentAt (e : Op.Element) (f : Fragment) : Option Fragment :=
  f.mapM fun (o, args) => (Op.atElement e o).map (·, args)

/-- The fragment of a typed pointwise node of `op` at shape `s` from operands of element type
`operand` to a result of element type `d`, with the element type its steps run at, or `none` at a
typing without one. It is the operator of `typed?` if there is one, else `where` with a Boolean
condition at the element of `d`, else chosen by the operand type. At a scalar it is the fragment of
`fragment?` at that scalar, without its final cast where a comparison, class predicate or logical
operator returns Booleans. At Booleans it is a logical operator. At an integer type it is negation,
a division, `pow` or `remainder` at that type, `clamp_min` as `Op.max`, or a comparison or logical
operator returning Booleans (a logical one through comparisons with zero). -/
def form? (op : PointwiseOp) (operand d : IR.DType) (s : Shape) :
    Option (Fragment × IR.DType) :=
  let single := fun (o : Operator) =>
    ([(o, (List.range op.arity).map Arg.operand)], d)
  let boolean := (fragment? op s).bind Fragment.boolean?
  if let some o := typed? op d s then some (single o)
  else if op = .select ∧ operand = .bool then
    ((element? d).bind (Op.atElement · ⟨_, _, Op.select (s := s)⟩)).map single
  else match element? operand, element? d with
    | some .scalar, _ =>
        if op = .select then
          if operand = d then (fragment? op s).map (·, d) else none
        else if returnsBool op ∧ d = .bool then boolean.map (·, operand)
        else if d = operand then (fragment? op s).map (·, operand)
        else none
    | some .bool, some .bool =>
        if returnsBool op then (logical? op s).map single else none
    | some (.integer _ _), some e@(.integer _ _) =>
        match op with
        | .clampMin => (Op.atElement e ⟨_, _, Op.max (s := s)⟩).map single
        | .neg | .div _ | .pow | .remainder =>
            ((fragment? op s).bind (fragmentAt e)).map (·, d)
        | _ => none
    | some e@(.integer _ _), some .bool =>
        if returnsBool op then (boolean.bind (fragmentAt e)).map (·, d) else none
    | _, _ => none

end TorchLean.IR.Pointwise
