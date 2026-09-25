/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.MLTheory.CROWN.Proofs.DirectedBackwardTransfers

/-!
# Real graph equations for directed backward CROWN

The rounded verifier targets the real graph described by its stored scalars. The equations here
mention node values, parameter values, and coordinate maps. They impose no condition on the
verifier's coefficient arithmetic or its returned bounds.

Nodes discharged through IBP need only their given value enclosure. Direct affine and structural
rules use their ordinary real equations. Connecting those equations to a separately scheduled
floating-point model evaluation additionally requires a rounding-error argument for that runtime.

The core backward theorems take a `GraphPoint`, whose `ibp_encloses` field states that each
forward box encloses the real node value. `GraphPoint.ofRunIBPAll` in `DirectedIBPFullBackward`
derives this field from input enclosures and `RealNodeEquation` for every graph operation,
including spatial convolution, structural tensor operations, normalization, and random
realizations. It uses the complete forward theorem `runIBP_encloses_all` and the backend's
directed arithmetic laws. The original `GraphPoint.ofRunIBP` remains available for the core
operation family selected by `ibpForwardSupported`.
-/

@[expose] public section

namespace NN.MLTheory.CROWN.Graph.DirectedBackward

open Spec TorchLean NN.IR
open TorchLean.Tensor
open NN.MLTheory.CROWN
open NN.MLTheory.CROWN.Graph.Internal
open scoped BigOperators

noncomputable section

variable {α : Type} [Storage α] [Context α] [BoundOps α] [LawfulBoundOps α]

local notation "value" => LawfulBoundOps.toReal (α := α)

/-- The shape and coordinate equations of a stored affine node. -/
def LinearEquation (dims : Nat → Nat) (v : Nat → Nat → ℝ) (id pid : Nat)
    {m n : Nat} (W : Tensor α [m, n]) (b : Tensor α [m]) : Prop :=
  dims id = m ∧ dims pid = n ∧ ∀ i : Fin m, v id i.val =
    (∑ j : Fin n, value (Spec.get2 W i j) * v pid j.val) + value (b.getScalar i)

/-- The shape and coordinate equations of a value-preserving unary node. -/
def CopyEquation (dims : Nat → Nat) (v : Nat → Nat → ℝ) (id pid : Nat) : Prop :=
  dims pid = dims id ∧ ∀ i : Fin (dims id), v id i.val = v pid i.val

/-- The shape and coordinate equations of a pointwise unary node computing `f`. -/
def UnaryEquation (dims : Nat → Nat) (v : Nat → Nat → ℝ) (id pid : Nat) (f : ℝ → ℝ) : Prop :=
  dims pid = dims id ∧ ∀ i : Fin (dims id), v id i.val = f (v pid i.val)

/-- The shape and coordinate equations of a pointwise binary node computing `f`. -/
def BinaryEquation (dims : Nat → Nat) (v : Nat → Nat → ℝ) (id p q : Nat)
    (f : ℝ → ℝ → ℝ) : Prop :=
  dims p = dims id ∧ dims q = dims id ∧
    ∀ i : Fin (dims id), v id i.val = f (v p i.val) (v q i.val)

/-- The inverse coordinate map identifies every output coordinate with its source coordinate. -/
def PermutationEquation (dims : Nat → Nat) (v : Nat → Nat → ℝ) (id pid : Nat)
    (outputShape : Shape) (forwardPerm : Array Nat) : Prop :=
  dims pid = dims id ∧
    ∀ inverse, (OpContracts.inversePerm forwardPerm).toOption = some inverse →
      ∀ perm, flatAxisPermutation? outputShape inverse (dims id) = some perm →
        Function.Bijective perm ∧
          ∀ i : Fin (dims id), v id (perm i).val = v pid i.val

/-- Real coordinate equations for a node, using the same successful parameter/layout lookups as
the executable dispatcher. Failed lookups require no equation because they reject the sweep.
Pointwise nonlinear nodes state their real function. Inputs, random nodes, and the operations
with a layout or reduction semantics not written here (pools, broadcasts, axis reductions,
BatchNorm, LayerNorm, softmax, `mseLoss`, `safeLog`) get `True`. -/
def NodeEquation (nodes : Array Node) (ps : ParamStore α)
    (ibp : Array (Option (FlatBox α))) (dims : Nat → Nat) (v : Nat → Nat → ℝ)
    (id : Nat) : Prop :=
  let node := nodes[id]!
  match node.kind with
  | .const _ =>
      ∀ stored, ps.constVals[id]? = some stored →
        dims id = stored.n ∧ ∀ i : Fin stored.n,
          v id i.val = value (stored.v.getScalar i)
  | .detach | .reshape _ _ | .flatten _ =>
      ∀ p, unaryParent? node.parents = some p → CopyEquation dims v id p
  | .add =>
      ∀ p q, binaryParents? node.parents = some (p, q) →
        dims p = dims id ∧ dims q = dims id ∧
          ∀ i : Fin (dims id), v id i.val = v p i.val + v q i.val
  | .sub =>
      ∀ p q, binaryParents? node.parents = some (p, q) →
        dims p = dims id ∧ dims q = dims id ∧
          ∀ i : Fin (dims id), v id i.val = v p i.val - v q i.val
  | .linear =>
      ∀ p, unaryParent? node.parents = some p →
        ∀ config, ps.linearWB[id]? = some config →
          LinearEquation dims v id p config.w config.b
  | .matmul =>
      ∀ p, unaryParent? node.parents = some p →
        ∀ config, ps.matmulW[id]? = some config →
          LinearEquation dims v id p config.w
            (Tensor.full (α := α) (.dim config.m .scalar) 0)
  | .conv configuration =>
      ∀ p, unaryParent? node.parents = some p →
        ∀ config, ps.convCfg[id]? = some config →
          ∀ parent, nodes[p]? = some parent →
            ∀ leading,
              planConvTransfer? configuration config parent.outShape node.outShape = some leading →
                LinearEquation dims v id p
                  (affOfConv (α := α) config leading).A (affOfConv (α := α) config leading).c
  | .sum =>
      ∀ p, unaryParent? node.parents = some p →
        ∀ box, ibp[p]! = some box →
          dims id = 1 ∧ dims p = box.dim ∧ v id 0 = ∑ i : Fin (dims p), v p i.val
  | .concat axis =>
      ∀ layout, concatBackwardLayout? nodes ibp node axis = some layout →
        dims id = layout.outputShape.size ∧ node.parents.size = layout.lengths.length ∧
          (∀ parent : Fin layout.lengths.length,
            dims (node.parents[parent.val]!) = (layout.parentShape parent).size) ∧
          ∀ (parent : Fin layout.lengths.length) (i : Fin (layout.parentShape parent).size),
            v id (layout.flatEquiv ⟨parent, i⟩).val = v (node.parents[parent.val]!) i.val
  | .transpose axis₁ axis₂ =>
      ∀ p, unaryParent? node.parents = some p →
        ∀ perm,
          (OpContracts.transposePerm nodes[p]!.outShape.rank axis₁ axis₂).toOption = some perm →
            PermutationEquation dims v id p node.outShape perm
  | .permute perm =>
      ∀ p, unaryParent? node.parents = some p → PermutationEquation dims v id p node.outShape perm
  | .relu => ∀ p, unaryParent? node.parents = some p → UnaryEquation dims v id p (max · 0)
  | .abs => ∀ p, unaryParent? node.parents = some p → UnaryEquation dims v id p (|·|)
  | .sqrt => ∀ p, unaryParent? node.parents = some p → UnaryEquation dims v id p Real.sqrt
  | .inv => ∀ p, unaryParent? node.parents = some p → UnaryEquation dims v id p (·⁻¹)
  | .exp => ∀ p, unaryParent? node.parents = some p → UnaryEquation dims v id p Real.exp
  | .log => ∀ p, unaryParent? node.parents = some p → UnaryEquation dims v id p Real.log
  | .tanh => ∀ p, unaryParent? node.parents = some p → UnaryEquation dims v id p Real.tanh
  | .sigmoid =>
      ∀ p, unaryParent? node.parents = some p →
        UnaryEquation dims v id p (fun x => 1 / (1 + Real.exp (-x)))
  | .softplus =>
      ∀ p, unaryParent? node.parents = some p →
        UnaryEquation dims v id p (fun x => Real.log (1 + Real.exp x))
  | .sin => ∀ p, unaryParent? node.parents = some p → UnaryEquation dims v id p Real.sin
  | .cos => ∀ p, unaryParent? node.parents = some p → UnaryEquation dims v id p Real.cos
  | .mulElem =>
      ∀ p q, binaryParents? node.parents = some (p, q) → BinaryEquation dims v id p q (· * ·)
  | .maxElem =>
      ∀ p q, binaryParents? node.parents = some (p, q) → BinaryEquation dims v id p q max
  | .minElem =>
      ∀ p q, binaryParents? node.parents = some (p, q) → BinaryEquation dims v id p q min
  | _ => True

/-- A topologically ordered real graph point with enclosed IBP values. Each direct node rule
satisfies its ordinary coordinate equation; all other active nodes may use their IBP enclosure.
`GraphPoint.ofRunIBP` builds one from a rounded `runIBP` pass on supported graphs. -/
structure GraphPoint (nodes : Array Node) (ps : ParamStore α)
    (ibp : Array (Option (FlatBox α))) (ctx : AffineCtx)
    (dims : Nat → Nat) (v : Nat → Nat → ℝ) : Prop where
  /-- The designated input exists in the graph. -/
  input_lt : ctx.inputId < nodes.size
  /-- Its width agrees with the affine interface. -/
  input_dim : dims ctx.inputId = ctx.inputDim
  /-- The designated input is an input node. -/
  input_kind : nodes[ctx.inputId]!.kind = .input
  /-- Stored node identifiers agree with their array positions. -/
  node_id : ∀ id, id < nodes.size → nodes[id]!.id = id
  /-- Every parent precedes its consumer. Repeated parents are allowed. -/
  parent_lt : ∀ id, id < nodes.size → ∀ p ∈ nodes[id]!.parents, p < id
  /-- Every available interval encloses the corresponding real node value. -/
  ibp_encloses : ∀ id, id < nodes.size → ∀ box, ibp[id]! = some box →
    RowEncloses box (dims id) (v id)
  /-- Directly propagated nodes obey their real coordinate equations. -/
  equation : ∀ id, id < nodes.size → NodeEquation nodes ps ibp dims v id

end

end NN.MLTheory.CROWN.Graph.DirectedBackward
