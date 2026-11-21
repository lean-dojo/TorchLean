/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.Dual.Special
public import TorchLean.NN.IR.Builder
public import TorchLean.NN.Op.Special
public import TorchLean.NN.PyTorch.Export.IR
public import TorchLean.Tensor.Op.Need
public meta import TorchLean.NN.IR.Builder
public meta import TorchLean.NN.PyTorch.Export.IR
public meta import TorchLean.Tensor.Op.Need

/-!
# The special-function operator at a scalar without `Special`

`Float` has no `Special` instance, so `Op.special orders f` needs the `Special` capability and has
no rule there. Its wire record is tagged `special` with the `torch.special` name of `f`, the orders
and the shape, and the name reads back to `f` for every member of `SpecialFn` of one, two and three
arguments. A record without the orders, or with a single order in place of the list, is refused,
and so are the records tagged `special2` and `betainc`. The PyTorch exporter writes orders all `0`
as the `torch.special` call on the operands, order `[n]` of digamma as
`torch.special.polygamma(n, x)`, and refuses every other order and `betainc`, since PyTorch has no
incomplete beta function (`emitted`).

Over the reals, nested dual numbers compute higher derivatives of the special functions:
`erfc_second_derivative` is the second derivative of `erfc` at `1`, `4 exp (-1) / √π`, read from
the dual number `⟨⟨1, 1⟩, ⟨1, 0⟩⟩`, `jet_erfc` is the jet of `erfc` at every order, and
`jet_smoothDomain` the jet of every function of one argument on its smooth domain.
`erfc_backward_eq_adjoint` gives the slope map of `Op.special [k] SpecialFn.erfc` and the adjoint
of its derivative as its pullback at every order `k` and every input, from
`Op.special₁_backward_eq_adjoint`, as every derivative of `erfc` is differentiable.
`deriv_du_eq_fderiv` checks that dual numbers in the arguments of the derivative of orders `k` of a
special function of any number of arguments carry that derivative's derivative along the tangents,
where the function is smooth. `contDiffAt_real_points` instantiates the smooth domains of the
functions of two and three arguments at a point each.
-/

@[expose] public section

namespace TorchLean.Test.Special

#guard ((Op.special [0] SpecialFn.erfc (s := [2])).rule? (α := Float)).isNone
#guard (Op.special [0] SpecialFn.lgamma (s := [2])).need == .special
#guard (Op.encode ⟨_, _, Op.special [0] SpecialFn.besselI0e (s := [3])⟩).tag == "special"
#guard [1, 2, 3].all fun n =>
  (SpecialFn.all n).all fun f => SpecialFn.ofTorchName? n f.torchName == some f
#guard (SpecialFn.all 1).length == 14 && (SpecialFn.all 2).length == 3 &&
  (SpecialFn.all 3).length == 1
#guard ((Op.special [0, 0] SpecialFn.igamma (s := [2])).rule? (α := Float)).isNone
#guard (Op.special [0, 0] SpecialFn.zeta (s := [2])).need == .special
#guard ((Op.special [0, 0, 0] SpecialFn.betainc (s := [2])).rule? (α := Float)).isNone

#guard (Op.encode ⟨_, _, Op.special [2] SpecialFn.erfc (s := [2])⟩).fields ==
  [.name "erfc", .nats [2], .shape [2]]
#guard Op.encode ⟨_, _, Op.special [1, 0, 2] SpecialFn.betainc (s := [2])⟩ ==
  ⟨Op.Wire.formatVersion, "special", [.name "betainc", .nats [1, 0, 2], .shape [2]]⟩
#guard ((Op.decode ⟨Op.Wire.formatVersion, "special",
    [.name "gammainc", .nats [1, 0], .shape [2]]⟩).map Op.encode) ==
  some ⟨Op.Wire.formatVersion, "special", [.name "gammainc", .nats [1, 0], .shape [2]]⟩
#guard (Op.decode ⟨Op.Wire.formatVersion, "special", [.name "erfc", .shape [2]]⟩).isNone
#guard (Op.decode ⟨Op.Wire.formatVersion, "special", [.name "erfc", .nat 0, .shape [2]]⟩).isNone
#guard (Op.decode ⟨Op.Wire.formatVersion, "special",
  [.name "gammainc", .nats [0], .shape [2]]⟩).isNone
#guard (Op.decode ⟨Op.Wire.formatVersion, "special2",
  [.name "gammainc", .nat 0, .nat 0, .shape [2]]⟩).isNone
#guard (Op.decode ⟨Op.Wire.formatVersion, "betainc", [.nat 0, .nat 0, .nat 0, .shape [2]]⟩).isNone

/-- The module `PyTorch.emit` writes for `o` applied to input 0, `v0` of shape `[3]`, and for
every other operand to `v1 = erfc v0`, or its refusal. -/
def emitted {ins : List Port} (o : Op ins [.value [3]]) : Except String String := do
  let b : IR.Graph.Builder := {}
  let (b, x) := b.input 0 (.value [3])
  let (b, y) :=
    if ins.length > 1 then b.op (Op.special [0] SpecialFn.erfc (s := [3])) [x] else (b, [])
  let (b, z) := b.op o (x :: List.replicate (ins.length - 1) (y.headD x))
  PyTorch.emit ⟨← b.build #[z.headD x], {}, #[]⟩ (options := { includeTrainingSkeleton := false })

/-- Whether `text` occurs in the module, or in the refusal when `refused` holds. -/
def mentions (result : Except String String) (text : String) (refused := false) : Bool :=
  match result, refused with
  | .ok module, false | .error module, true => (module.splitOn text).length > 1
  | _, _ => false

#guard mentions (emitted (Op.special [0] SpecialFn.erfc (s := [3]))) "v1 = torch.special.erfc(v0)"
#guard mentions (emitted (Op.special [2] SpecialFn.digamma (s := [3])))
  "v1 = torch.special.polygamma(2, v0)"
#guard mentions (emitted (Op.special [1] SpecialFn.erfc (s := [3])))
  "no function for the derivative of orders [1] of erfc" (refused := true)
#guard mentions (emitted (Op.special [0, 0] SpecialFn.igamma (s := [3])))
  "v2 = torch.special.gammainc(v0, v1)"
#guard mentions (emitted (Op.special [0, 0] SpecialFn.zeta (s := [3])))
  "v2 = torch.special.zeta(v0, v1)"
#guard mentions (emitted (Op.special [1, 0] SpecialFn.igamma (s := [3])))
  "no function for the derivative of orders [1, 0] of gammainc" (refused := true)
#guard mentions (emitted (Op.special [0, 2] SpecialFn.zeta (s := [3])))
  "no function for the derivative of orders [0, 2] of zeta" (refused := true)
#guard mentions (emitted (Op.special [0, 0, 0] SpecialFn.betainc (s := [3])))
  "betainc has no PyTorch translation" (refused := true)

open Real in
/-- The second derivative of `erfc` at `1` through two nested dual layers. -/
theorem erfc_second_derivative :
    (Special.deriv SpecialFn.erfc 0 fun _ => (⟨⟨1, 1⟩, ⟨1, 0⟩⟩ : Dual (Dual ℝ))).du.du =
      4 / √π * Real.exp (-1) := by
  have e (k : Fin 1 → ℕ) : Special.deriv SpecialFn.erfc k (fun _ => (1 : ℝ)) =
      deriv^[k 0] erfc 1 :=
    SpecialFn.deriv_eq_iterate _ k _
  have h := (((hasDerivAt_pow 2 (1 : ℝ)).fun_neg.exp).const_mul (2 / √π)).fun_neg
  show Special.deriv SpecialFn.erfc (0 + Pi.single 0 1 + Pi.single 0 1) (fun _ => (1 : ℝ)) * 1 *
      1 + Special.deriv SpecialFn.erfc (0 + Pi.single 0 1) (fun _ => (1 : ℝ)) * 0 = _
  rw [e, e]
  show deriv (deriv erfc) 1 * 1 * 1 + deriv erfc 1 * 0 = _
  rw [deriv_erfc, h.deriv]
  norm_num
  ring

/-- Seeded nested duals compute the jet of `erfc` at every order. -/
theorem jet_erfc {n : Nat} (directions : Fin n → ℝ) (x : ℝ) :
    Dual.jet directions erfc x =
      Special.deriv SpecialFn.erfc 0 fun _ => Dual.Nested.seed directions x := by
  simpa only [SpecialFn.deriv_zero, SpecialFn.real] using
    Dual.jet_special directions .erfc (contDiff_erfc.contDiffAt (x := x))

/-- At every order `k` and every input `x`, the forward of `Op.special [k] SpecialFn.erfc` has the
slope map of order `k + 1` as derivative at `x`, and its pullback of `δ` is the adjoint of that
derivative applied to `δ`. -/
theorem erfc_backward_eq_adjoint {s : Shape} (k : ℕ) (x δ : Tensor ℝ s) :
    HasFDerivAt (fun y => (((Op.special [k] SpecialFn.erfc (s := s)).rule? (α := ℝ) |>.get
          rfl).forward (y, ())).1)
        (Op.slopeCLM fun c => Special.deriv SpecialFn.erfc (fun _ => k + 1) fun _ => x c) x ∧
      (((Op.special [k] SpecialFn.erfc (s := s)).rule? (α := ℝ) |>.get rfl).backward (x, ())
          (δ, ())).1 =
        (fderiv ℝ (fun y => (((Op.special [k] SpecialFn.erfc (s := s)).rule? (α := ℝ) |>.get
          rfl).forward (y, ())).1) x).adjoint δ :=
  Op.special₁_backward_eq_adjoint .erfc k x δ fun _ =>
    ((contDiff_erfc.iterate_deriv k).differentiable (by simp)).differentiableAt

/-- Seeded nested duals compute the jet of every special function of one argument on its smooth
domain. -/
theorem jet_smoothDomain {n : Nat} (directions : Fin n → ℝ) (f : SpecialFn 1) {x : ℝ}
    (hx : (fun _ => x) ∈ f.smoothDomain) :
    Dual.jet directions (fun t => Special.deriv f 0 fun _ => t) x =
      Special.deriv f 0 fun _ => Dual.Nested.seed directions x :=
  Dual.jet_special directions f (SpecialFn.contDiffAt_real₁ hx)

/-- `Special.leftSum` adds the same terms as `Finset.sum`. -/
private theorem leftSum_eq_sum {n : ℕ} (t : Fin n → ℝ) : Special.leftSum t = ∑ i, t i := by
  induction n with
  | zero => simp [Special.leftSum]
  | succ n ih =>
    cases n with
    | zero => simp [Special.leftSum]
    | succ n =>
      show Special.leftSum (fun i : Fin (n + 1) => t i.castSucc) + t (Fin.last (n + 1)) = _
      rw [ih]
      exact (Fin.sum_univ_castSucc t).symm

open scoped ContDiff in
/-- Where the real special function `f` is smooth at `x`, dual numbers in its arguments, with
tangents `dx`, carry the derivative of its derivative of orders `k` along `dx`. -/
theorem deriv_du_eq_fderiv {n : ℕ} (f : SpecialFn n) (k : Fin n → ℕ) {x : Fin n → ℝ}
    (dx : Fin n → ℝ) (h : ContDiffAt ℝ ∞ f.real x) :
    (Special.deriv f k fun i => (⟨x i, dx i⟩ : Dual ℝ)).du =
      fderiv ℝ (Special.deriv f k) x dx := by
  rw [Dual.deriv_du, (SpecialFn.hasFDerivAt_deriv f k h).fderiv, leftSum_eq_sum]
  simp [smul_eq_mul]

open scoped ContDiff in
/-- `igamma` and `igammac` are smooth at `(1, 1)`, `zeta` at `(2, 1)` and `betainc` at
`(1, 1, 1 / 2)`, points of their smooth domains. -/
theorem contDiffAt_real_points :
    ContDiffAt ℝ ∞ SpecialFn.igamma.real ![1, 1] ∧ ContDiffAt ℝ ∞ SpecialFn.igammac.real ![1, 1] ∧
      ContDiffAt ℝ ∞ SpecialFn.zeta.real ![2, 1] ∧
      ContDiffAt ℝ ∞ SpecialFn.betainc.real ![1, 1, 1 / 2] :=
  ⟨SpecialFn.contDiffAt_real (by norm_num [SpecialFn.smoothDomain]),
    SpecialFn.contDiffAt_real (by norm_num [SpecialFn.smoothDomain]),
    SpecialFn.contDiffAt_real (by norm_num [SpecialFn.smoothDomain]),
    SpecialFn.contDiffAt_real (by norm_num [SpecialFn.smoothDomain])⟩

end TorchLean.Test.Special
