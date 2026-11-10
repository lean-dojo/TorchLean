/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Verify.RuntimeApprox.OnlineSoftmax.Binary32
public import TorchLean.Verify.RuntimeApprox.TransformerBlock.Nearest

/-!
# GPT-2-Small Block Rounding Bound

`sqrt_sum_sq_block_sub_le` evaluates the transformer block bound at the sizes of
GPT-2 small in `binary32`: width 768, 12 heads of width 64, hidden width 3072 and at most 1024 keys
per token. The stage inequalities are checked by `norm_num` and `linarith` from rational bounds on
the square roots, exponentials and summation constants that occur. The file applies the generic
bound `Arithmetic.sqrt_sum_sq_block_sub_le` of `TorchLean.Verify.RuntimeApprox.TransformerBlock`
to `nearestContract`, which `TorchLean.Verify.RuntimeApprox.TransformerBlock.Nearest` builds for
nearest rounding in any IEEE format.
-/

@[expose] public section

namespace TorchLean.Examples.GPT2BlockBound

open FloatLib.Floats FloatLib.Floats.Formats.BinaryInterchange
open TransformerBlock
open TorchLean (var)

/-- In `binary32` with unrounded stores, a GPT-2-small block (width 768, 12 heads of width 64,
hidden width 3072, at most 1024 keys per token, `ε = 10⁻⁵`, `τ = 1/8`) is within `0.538` of the
exact block at every token in Euclidean norm, given these bounds on the exact stages and weights. -/
theorem sqrt_sum_sq_block_sub_le {ν : Type*} {act : ℝ → ℝ}
    (hact : ∀ t, |act t| ≤ |t|) {La : ℝ} (hLa0 : 0 ≤ La) (hLa : La ≤ 2.11)
    (hlip : ∀ s t, |act s - act t| ≤ La * |s - t|)
    (P : Parameters (Fin 768) (Fin 12) (Fin 64) (Fin 3072)) {keys : ν → Finset ν}
    (hkeys : ∀ t, (keys t).Nonempty) (hcard : ∀ t, (keys t).card ≤ 1024)
    (x : ν → Fin 768 → ℝ) {a : ν → Fin 768 → ℝ}
    (ha : a = fun t => layerNorm P.γ1 P.β1 (1 / 100000) (x t))
    {q k v : ν → Fin 12 × Fin 64 → ℝ} (hq : q = fun t c => P.bq c + ∑ j, a t j * P.wq j c)
    (hk : k = fun t c => P.bk c + ∑ j, a t j * P.wk j c)
    (hv : v = fun t c => P.bv c + ∑ j, a t j * P.wv j c)
    {o : ν → Fin 12 × Fin 64 → ℝ} (ho : o = heads (1 / 8) keys q k v)
    {s : ν → Fin 768 → ℝ} (hs : s = fun t i => x t i + (P.bo i + ∑ c, o t c * P.wo c i))
    {b : ν → Fin 768 → ℝ} (hb : b = fun t => layerNorm P.γ2 P.β2 (1 / 100000) (s t))
    {r : ν → Fin 3072 → ℝ} (hr : r = fun t c => P.b1 c + ∑ j, b t j * P.w1 j c)
    (hX : ∀ t, √(∑ i, x t i ^ 2) ≤ 29.8) (hσlo1 : ∀ t, 0.92 ≤ √(var (x t) + 1 / 100000))
    (hσhi1 : ∀ t, √(var (x t) + 1 / 100000) ≤ 1.08) (hYa : ∀ t, √(∑ i, a t i ^ 2) ≤ 28.3)
    (hQ : ∀ t z, √(∑ i, q t (z, i) ^ 2) ≤ 6) (hK : ∀ t z, √(∑ i, k t (z, i) ^ 2) ≤ 6)
    (hV : ∀ t z, √(∑ i, v t (z, i) ^ 2) ≤ 6) (hO : ∀ t, √(∑ c, o t c ^ 2) ≤ 15.5)
    (hZ : ∀ t, √(∑ i, s t i ^ 2) ≤ 29.8) (hσlo2 : ∀ t, 0.92 ≤ √(var (s t) + 1 / 100000))
    (hσhi2 : ∀ t, √(var (s t) + 1 / 100000) ≤ 1.08) (hYb : ∀ t, √(∑ i, b t i ^ 2) ≤ 28.4)
    (hR : ∀ t, √(∑ c, r t c ^ 2) ≤ 32.5) (hYm : ∀ t, √(∑ c, act (r t c) ^ 2) ≤ 19.7)
    (hY : ∀ t, √(∑ i, block act P (1 / 100000) (1 / 8) keys x t i ^ 2) ≤ 29.8)
    (hγ1 : ∀ j, |P.γ1 j| ≤ 1.3) (hγ2 : ∀ j, |P.γ2 j| ≤ 1.3)
    (hWq : ∀ z, LinearBound (fun j i => P.wq j (z, i)) (fun i => P.bq (z, i)) 0.72 4.5 0.2)
    (hWk : ∀ z, LinearBound (fun j i => P.wk j (z, i)) (fun i => P.bk (z, i)) 0.72 4.5 0.2)
    (hWv : ∀ z, LinearBound (fun j i => P.wv j (z, i)) (fun i => P.bv (z, i)) 0.72 4.5 0.2)
    (hWo : LinearBound P.wo P.bo 0.23 3.2 0.53) (hW1 : LinearBound P.w1 P.b1 1.67 30.8 1.11)
    (hW2 : LinearBound P.w2 P.b2 0.34 6.3 0.56) (t : ν) :
    √(∑ i, ((nearest FloatFormat.binary32 id act).block P (1 / 100000) (1 / 8) hkeys x t i -
      block act P (1 / 100000) (1 / 8) keys x t i) ^ 2) ≤ 0.538 := by
  have hu : Model.unitRoundoffAt FloatFormat.binary32 = 16777216⁻¹ := by
    rw [OnlineSoftmax.unitRoundoff_binary32]; norm_num
  have hη : Model.underflowErrorAt FloatFormat.binary32 =
      1427247692705959881058285969449495136382746624⁻¹ := by
    rw [OnlineSoftmax.underflow_binary32]; norm_num
  obtain ⟨C, hCu, hCη, hCum, hCηm, hCus, hCηs, hCue, hCηe, hCua, hCηa, hCLa, hCg, hCh⟩ :
      ∃ C : (nearest FloatFormat.binary32 id act).Contract act, C.u = 16777216⁻¹ ∧
        C.η = 1427247692705959881058285969449495136382746624⁻¹ ∧ C.um = 16777216⁻¹ ∧
        C.ηm = 1427247692705959881058285969449495136382746624⁻¹ ∧ C.us = 0 ∧ C.ηs = 0 ∧
        C.ue = 16777216⁻¹ ∧ C.ηe = 1427247692705959881058285969449495136382746624⁻¹ ∧
        C.ua = 16777216⁻¹ ∧ C.ηa = 1427247692705959881058285969449495136382746624⁻¹ ∧
        C.la = La ∧ (∀ n, C.g n = (1 + 16777216⁻¹) ^ n - 1) ∧
        ∀ n, C.h n = n * (1 + 16777216⁻¹) ^ n * 1427247692705959881058285969449495136382746624⁻¹ :=
    ⟨nearestContract FloatFormat.binary32 le_rfl le_rfl abs_id_sub_le hact hLa0 hlip, hu, hη, hu,
      hη, rfl, rfl, hu, hη, hu, hη, rfl,
      fun n => by show (1 + Model.unitRoundoffAt _) ^ n - 1 = _; rw [hu],
      fun n => by
        show n * (1 + Model.unitRoundoffAt _) ^ n * Model.underflowErrorAt _ = _
        rw [hu, hη]⟩
  have hη0 : (0 : ℝ) ≤ 1427247692705959881058285969449495136382746624⁻¹ := by norm_num
  have g64 : C.g 64 ≤ 3.82e-6 := by
    rw [hCg]; exact (one_add_pow_sub_one_le le_rfl (by norm_num)).trans (by norm_num)
  have g768 : C.g 768 ≤ 4.58e-5 := by
    rw [hCg]; exact (one_add_pow_sub_one_le le_rfl (by norm_num)).trans (by norm_num)
  have g769 : C.g 769 ≤ 4.59e-5 := by
    rw [hCg]; exact (one_add_pow_sub_one_le le_rfl (by norm_num)).trans (by norm_num)
  have g3073 : C.g 3073 ≤ 1.84e-4 := by
    rw [hCg]; exact (one_add_pow_sub_one_le le_rfl (by norm_num)).trans (by norm_num)
  have h64 : C.h 64 ≤ 4.5e-44 := by
    rw [hCh]; exact (mul_one_add_pow_mul_le le_rfl (by norm_num) hη0).trans (by norm_num)
  have h768 : C.h 768 ≤ 5.4e-43 := by
    rw [hCh]; exact (mul_one_add_pow_mul_le le_rfl (by norm_num) hη0).trans (by norm_num)
  have h769 : C.h 769 ≤ 5.4e-43 := by
    rw [hCh]; exact (mul_one_add_pow_mul_le le_rfl (by norm_num) hη0).trans (by norm_num)
  have h3073 : C.h 3073 ≤ 2.2e-42 := by
    rw [hCh]; exact (mul_one_add_pow_mul_le le_rfl (by norm_num) hη0).trans (by norm_num)
  have hg0 := C.g_nonneg
  have hh0 := C.h_nonneg
  obtain ⟨s768, hs768⟩ : ∃ s, √(768 : ℝ) = s := ⟨_, rfl⟩
  obtain ⟨i768, hi768⟩ : ∃ i, s768⁻¹ = i := ⟨_, rfl⟩
  obtain ⟨s12, hs12⟩ : ∃ s, √(12 : ℝ) = s := ⟨_, rfl⟩
  obtain ⟨s3072, hs3072⟩ : ∃ s, √(3072 : ℝ) = s := ⟨_, rfl⟩
  have hs768u : s768 ≤ 27.72 := hs768 ▸ Real.sqrt_le_iff.2 ⟨by norm_num, by norm_num⟩
  have hs768l : 27.71 ≤ s768 := hs768 ▸ (Real.le_sqrt (by norm_num) (by norm_num)).2 (by norm_num)
  have hs768p : 0 < s768 := by linarith
  have hi768u : i768 ≤ 0.03609 := by
    rw [← hi768, inv_le_comm₀ hs768p (by norm_num)]; linarith
  have hi768p : 0 ≤ i768 := hi768 ▸ (inv_pos.2 hs768p).le
  have hs12u : s12 ≤ 3.465 := hs12 ▸ Real.sqrt_le_iff.2 ⟨by norm_num, by norm_num⟩
  have hs12p : 0 ≤ s12 := hs12 ▸ Real.sqrt_nonneg _
  have hs3072u : s3072 ≤ 55.43 := hs3072 ▸ Real.sqrt_le_iff.2 ⟨by norm_num, by norm_num⟩
  have hs3072p : 0 ≤ s3072 := hs3072 ▸ Real.sqrt_nonneg _
  have hs64 : √(64 : ℝ) = 8 := by
    rw [show (64 : ℝ) = 8 ^ 2 by norm_num]; exact Real.sqrt_sq (by norm_num)
  have hdiv : ∀ y : ℝ, y / s768 = y * i768 := fun y => by rw [div_eq_mul_inv, hi768]
  have p1 := mul_le_mul g768 hs768u hs768p.le (by norm_num)
  have p2 := mul_le_mul g769 hs12u hs12p (by norm_num)
  have p3 := mul_le_mul g769 hs768u hs768p.le (by norm_num)
  have p4 := mul_le_mul h769 hs768u hs768p.le (by norm_num)
  have p5 := mul_le_mul g769 hs3072u hs3072p (by norm_num)
  have p6 := mul_le_mul h769 hs3072u hs3072p (by norm_num)
  have p7 := mul_le_mul g3073 hs768u hs768p.le (by norm_num)
  have p8 := mul_le_mul h3073 hs768u hs768p.le (by norm_num)
  obtain ⟨Eδ, hEδ⟩ : ∃ E, Real.exp (1.85e-5 : ℝ) = E := ⟨_, rfl⟩
  have hEδu : Eδ ≤ 1.00001851 := hEδ ▸
    (Real.exp_bound_div_one_sub_of_interval (by norm_num) (by norm_num)).trans (by norm_num)
  obtain ⟨Es, hEs⟩ : ∃ E, Real.exp (1.79e-5 : ℝ) = E := ⟨_, rfl⟩
  have hEsu : Es ≤ 1.00001791 := hEs ▸
    (Real.exp_bound_div_one_sub_of_interval (by norm_num) (by norm_num)).trans (by norm_num)
  obtain ⟨T, hT⟩ : ∃ T, 2 * (Real.exp (0.0187 : ℝ) - 1) / (Real.exp 0.0187 + 1) = T :=
    ⟨_, rfl⟩
  have hTu : T ≤ 1.01905636 - 1 := hT ▸ (two_mul_exp_sub_one_div_le (by norm_num)).trans
    (sub_le_sub_right ((Real.exp_bound_div_one_sub_of_interval (by norm_num)
      (by norm_num)).trans (by norm_num)) 1)
  have hcard' : ∀ t, ((keys t).card : ℝ) ≤ 1024 := fun t => by exact_mod_cast hcard t
  refine Arithmetic.sqrt_sum_sq_block_sub_le C P (by norm_num) (by norm_num) hkeys x ha hq hk hv
    ho hs hb hr hX hσlo1 hσhi1 hYa hQ hK hV hO hZ hσlo2 hσhi2 hYb hR hYm hY hγ1 hγ2 hWq hWk hWv
    hWo hW1 hW2 (gN := 6.11e-5) (hN := 7.2e-43) (N := 1024) ?_ ?_ hcard' (ep1 := 0.00897)
    (ea := 0.00899) (eq := 0.0124) (ek := 0.0124) (ev := 0.0124) (ω := 0.0187)
    (es := 1.79e-5) (δ := 1.85e-5) (θ := 1.86e-5) (eL := 7.99e-5) (eN := 0.000481)
    (eo := 0.128) (ep := 0.105) (ez := 0.106) (ep2 := 0.00915) (eb := 0.31) (er := 0.56)
    (em := 1.19) (ef := 0.43) (ey := 0.538) (by norm_num) ?hLN1 ?hea ?heq ?hek ?hev
    (by norm_num) ?hes ?hδ ?hθ ?heL (by norm_num) ?heN ?heo ?hep ?hez ?hσlo2pos ?hLN2 ?heb ?her
    ?hem ?hef ?hey t
  · intro t; rw [hCg]
    exact (one_add_pow_sub_one_le (hcard t) (by norm_num)).trans (by norm_num)
  · intro t; rw [hCh]
    exact (mul_one_add_pow_mul_le (hcard t) (by norm_num) hη0).trans (by norm_num)
  case hLN1 =>
    unfold LayerNormCertificate
    refine ⟨4.95e-5, 0.00138, 0.124, 0.000162, 0.000163, 0.000178, 0.00688, ?_, ?_, ?_, ?_, ?_,
      ?_, by norm_num, ?_, ?_⟩ <;>
      simp only [Fintype.card_fin, Nat.cast_ofNat, hCu, hCη, hs768, hdiv] <;> norm_num <;>
      linarith
  case hLN2 =>
    refine LayerNormCertificate.mono (σlo' := 0.916) (σhi' := 1.084) ?_ C.u_nonneg C.η_nonneg
      (hg0 _) (hh0 _) (by norm_num) (by norm_num) ?_ ?_ ?_
    · unfold LayerNormCertificate
      refine ⟨4.97e-5, 0.00139, 0.125, 0.000164, 0.000165, 0.000181, 0.00702, ?_, ?_, ?_, ?_,
        ?_, ?_, by norm_num, ?_, ?_⟩ <;>
        simp only [Fintype.card_fin, Nat.cast_ofNat, hCu, hCη, hs768, hdiv] <;> norm_num <;>
        linarith
    all_goals
      simp only [Fintype.card_fin, Nat.cast_ofNat, hs768, hdiv]
      norm_num
      linarith
  case hea =>
    simp only [Fintype.card_fin, Nat.cast_ofNat, hCu, hCη, hCus, hCηs, hs768]
    norm_num; linarith
  case heq =>
    simp only [Fintype.card_fin, Nat.cast_ofNat, hCum, hCηm, hCus, hCηs, hs64]
    norm_num; linarith
  case hek =>
    simp only [Fintype.card_fin, Nat.cast_ofNat, hCum, hCηm, hCus, hCηs, hs64]
    norm_num; linarith
  case hev =>
    simp only [Fintype.card_fin, Nat.cast_ofNat, hCum, hCηm, hCus, hCηs, hs64]
    norm_num; linarith
  case hes =>
    simp only [Fintype.card_fin, Nat.cast_ofNat, hCu, hCη, hCum, hCηm]
    norm_num; linarith
  case hδ =>
    simp only [hCu, hCη]
    norm_num
  case hθ =>
    simp only [hCue, hEδ]
    norm_num; linarith
  case heL =>
    simp only [hCηe, hEs]
    norm_num; linarith
  case heN =>
    simp only [Fintype.card_fin, Nat.cast_ofNat, hCum, hCηm, hCηe, hs64, hEs]
    norm_num; linarith
  case heo =>
    simp only [Fintype.card_fin, Nat.cast_ofNat, hCu, hCη, hCus, hCηs, hs64, hT]
    norm_num; linarith
  case hep =>
    simp only [Fintype.card_fin, Fintype.card_prod, Nat.cast_ofNat, hCum, hCηm, hCus, hCηs,
      hs768, hs12]
    norm_num; linarith
  case hez =>
    simp only [Fintype.card_fin, Nat.cast_ofNat, hCu, hCη, hCus, hCηs, hs768]
    norm_num; linarith
  case hσlo2pos =>
    simp only [Fintype.card_fin, Nat.cast_ofNat, hs768, hdiv]
    norm_num; linarith
  case heb =>
    simp only [Fintype.card_fin, Nat.cast_ofNat, hCu, hCη, hCus, hCηs, hs768]
    norm_num; linarith
  case her =>
    simp only [Fintype.card_fin, Nat.cast_ofNat, hCum, hCηm, hCus, hCηs, hs3072]
    norm_num; linarith
  case hem =>
    simp only [Fintype.card_fin, Nat.cast_ofNat, hCus, hCηs, hCua, hCηa, hCLa, hs3072]
    norm_num; linarith
  case hef =>
    simp only [Fintype.card_fin, Nat.cast_ofNat, hCum, hCηm, hCus, hCηs, hs768]
    norm_num; linarith
  case hey =>
    simp only [Fintype.card_fin, Nat.cast_ofNat, hCu, hCη, hCus, hCηs, hs768]
    norm_num; linarith

end TorchLean.Examples.GPT2BlockBound
