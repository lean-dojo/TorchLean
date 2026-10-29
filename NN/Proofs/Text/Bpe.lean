/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Text.Bpe

/-!
# Correctness of GPT-2 BPE merges

The merge passes below are the ones executed by `GPT2BPE.Internal.encodeFragment`. Selected pairs
have minimum stored rank. Merging changes token boundaries while preserving the concatenated
byte-escaped text, independently of the rank table.
-/

@[expose] public section

namespace TorchLean.text.GPT2BPE.Internal

/-- Replacing adjacent pairs by their concatenation preserves the represented text. -/
theorem join_apply_merge (left right : String) (pieces : List String) :
    String.join (applyMerge left right pieces) = String.join pieces := by
  induction pieces using applyMerge.induct left right with
  | case1 first second rest h ih =>
      simp only [applyMerge, h, ↓reduceIte, String.join_cons, ih, String.append_assoc]
  | case2 first second rest h ih =>
      simp [applyMerge, h, ih]
  | case3 pieces h => simp [applyMerge]

/-- Every fuel-bounded BPE run preserves the text of its input pieces. -/
theorem join_bpe_loop (tokenizer : Tokenizer) (fuel : Nat) (pieces : List String) :
    String.join (bpeLoop tokenizer fuel pieces) = String.join pieces := by
  induction fuel generalizing pieces with
  | zero => rfl
  | succ fuel ih =>
      simp only [bpeLoop]
      split
      · rfl
      · rw [ih, join_apply_merge]

/-- A merge pass never increases the number of pieces. -/
theorem apply_merge_length_le (left right : String) (pieces : List String) :
    (applyMerge left right pieces).length ≤ pieces.length := by
  induction pieces using applyMerge.induct left right with
  | case1 first second rest h ih =>
      simp only [applyMerge, h, ↓reduceIte, List.length_cons]
      omega
  | case2 first second rest h ih =>
      simp only [applyMerge, h, Bool.false_eq_true, ↓reduceIte, List.length_cons]
      simp only [List.length_cons] at ih
      omega
  | case3 pieces h => simp [applyMerge]

private theorem pair_infix_cons_iff (left right first second : String) (rest : List String) :
    [left, right] <:+: first :: second :: rest ↔
      (left = first ∧ right = second) ∨ [left, right] <:+: second :: rest := by
  rw [List.infix_cons_iff]
  simp

private theorem best_merge_spec_aux (tokenizer : Tokenizer) (pieces : List String) :
    match bestMerge? tokenizer pieces with
    | none => ∀ left right rank, [left, right] <:+: pieces →
        mergeRank? tokenizer left right = some rank → False
    | some candidate =>
        [candidate.left, candidate.right] <:+: pieces ∧
        mergeRank? tokenizer candidate.left candidate.right = some candidate.rank ∧
        ∀ left right rank, [left, right] <:+: pieces →
          mergeRank? tokenizer left right = some rank → candidate.rank ≤ rank := by
  induction pieces with
  | nil => simp [bestMerge?]
  | cons first tail ih =>
      cases tail with
      | nil => simp [bestMerge?, List.infix_cons_iff]
      | cons second rest =>
          cases hr : mergeRank? tokenizer first second with
          | none =>
              cases ht : bestMerge? tokenizer (second :: rest) with
              | none =>
                  simp only [bestMerge?, hr, Option.map_none, ht]
                  simp only [ht] at ih
                  intro left right rank hadj havail
                  rcases (pair_infix_cons_iff left right first second rest).mp hadj with
                    ⟨rfl, rfl⟩ | htail
                  · simp [hr] at havail
                  · exact ih left right rank htail havail
              | some candidate =>
                  simp only [bestMerge?, hr, Option.map_none, ht]
                  simp only [ht] at ih
                  refine ⟨List.infix_cons ih.1, ih.2.1, ?_⟩
                  intro left right rank hadj havail
                  rcases (pair_infix_cons_iff left right first second rest).mp hadj with
                    ⟨rfl, rfl⟩ | htail
                  · simp [hr] at havail
                  · exact ih.2.2 left right rank htail havail
          | some rank =>
              cases ht : bestMerge? tokenizer (second :: rest) with
              | none =>
                  simp only [bestMerge?, hr, Option.map_some, ht]
                  simp only [ht] at ih
                  refine ⟨⟨[], rest, rfl⟩, trivial, ?_⟩
                  intro left right otherRank hadj havail
                  rcases (pair_infix_cons_iff left right first second rest).mp hadj with
                    ⟨rfl, rfl⟩ | htail
                  · simp only [hr, Option.some.injEq] at havail
                    omega
                  · exact False.elim (ih left right otherRank htail havail)
              | some candidate =>
                  simp only [bestMerge?, hr, Option.map_some, ht]
                  simp only [ht] at ih
                  by_cases hle : rank ≤ candidate.rank
                  · simp only [ite_eq_left hle]
                    refine ⟨⟨[], rest, rfl⟩, hr, ?_⟩
                    intro left right otherRank hadj havail
                    rcases (pair_infix_cons_iff left right first second rest).mp hadj with
                      ⟨rfl, rfl⟩ | htail
                    · simp only [hr, Option.some.injEq] at havail
                      omega
                    · exact hle.trans (ih.2.2 left right otherRank htail havail)
                  · simp only [ite_eq_right hle]
                    refine ⟨List.infix_cons ih.1, ih.2.1, ?_⟩
                    intro left right otherRank hadj havail
                    rcases (pair_infix_cons_iff left right first second rest).mp hadj with
                      ⟨rfl, rfl⟩ | htail
                    · simp only [hr, Option.some.injEq] at havail
                      omega
                    · exact ih.2.2 left right otherRank htail havail

/-- The selected merge is an adjacent pair with its stored rank, minimal among all ranked adjacent
pairs in the current pieces. -/
theorem best_merge_spec (tokenizer : Tokenizer) (pieces : List String)
    (candidate : MergeCandidate) (h : bestMerge? tokenizer pieces = some candidate) :
    [candidate.left, candidate.right] <:+: pieces ∧
    mergeRank? tokenizer candidate.left candidate.right = some candidate.rank ∧
    ∀ left right rank, [left, right] <:+: pieces →
      mergeRank? tokenizer left right = some rank → candidate.rank ≤ rank := by
  have hs := best_merge_spec_aux tokenizer pieces
  simpa only [h] using hs

/-- A merge selected from the current adjacent pairs strictly decreases the piece count. -/
theorem apply_merge_length_lt_of_best_merge (tokenizer : Tokenizer)
    (pieces : List String) (candidate : MergeCandidate)
    (h : bestMerge? tokenizer pieces = some candidate) :
    (applyMerge candidate.left candidate.right pieces).length < pieces.length := by
  induction pieces generalizing candidate with
  | nil => simp [bestMerge?] at h
  | cons first tail ih =>
      cases tail with
      | nil => simp [bestMerge?] at h
      | cons second rest =>
          have head_progress :
              (applyMerge first second (first :: second :: rest)).length <
                (first :: second :: rest).length := by
            simp only [applyMerge, beq_self_eq_true, Bool.and_self, ↓reduceIte,
              List.length_cons]
            have := apply_merge_length_le first second rest
            omega
          have tail_progress (selected : MergeCandidate)
              (ht : bestMerge? tokenizer (second :: rest) = some selected) :
              (applyMerge selected.left selected.right (first :: second :: rest)).length <
                (first :: second :: rest).length := by
            simp only [applyMerge]
            split
            · simp only [List.length_cons]
              have := apply_merge_length_le selected.left selected.right rest
              omega
            · simp only [List.length_cons]
              have htail := ih selected ht
              simp only [List.length_cons] at htail
              omega
          simp only [bestMerge?] at h
          cases hr : mergeRank? tokenizer first second with
          | none =>
              simp only [hr, Option.map_none] at h
              exact tail_progress candidate h
          | some rank =>
              simp only [hr, Option.map_some] at h
              cases ht : bestMerge? tokenizer (second :: rest) with
              | none =>
                  simp only [ht, Option.some.injEq] at h
                  subst candidate
                  exact head_progress
              | some selected =>
                  simp only [ht] at h
                  split at h
                  · simp only [Option.some.injEq] at h
                    subst candidate
                    exact head_progress
                  · simp only [Option.some.injEq] at h
                    subst candidate
                    exact tail_progress selected ht

/-- With at least `pieces.length - 1` rounds, BPE exhausts all applicable merges. -/
theorem best_merge_bpe_loop_eq_none (tokenizer : Tokenizer) (fuel : Nat)
    (pieces : List String) (hlen : pieces.length ≤ fuel + 1) :
    bestMerge? tokenizer (bpeLoop tokenizer fuel pieces) = none := by
  induction fuel generalizing pieces with
  | zero =>
      cases pieces with
      | nil => rfl
      | cons first rest =>
          cases rest with
          | nil => rfl
          | cons second rest =>
              simp only [List.length_cons] at hlen
              omega
  | succ fuel ih =>
      simp only [bpeLoop]
      split
      · assumption
      · rename_i candidate hcandidate
        apply ih
        have hprogress := apply_merge_length_lt_of_best_merge tokenizer pieces candidate hcandidate
        omega

private theorem forall₂_token_id_of_mapM (tokenizer : Tokenizer)
    (pieces : List String) (ids : List Nat)
    (h : pieces.mapM (fun piece =>
      match tokenId? tokenizer piece with
      | some id => pure id
      | none => throw s!"BPE piece is absent from vocabulary: {repr piece}") =
        (Except.ok ids : Except String (List Nat))) :
    List.Forall₂ (fun piece id => tokenId? tokenizer piece = some id) pieces ids := by
  induction pieces generalizing ids with
  | nil =>
      simp only [List.mapM_nil, pure, Except.pure, Except.ok.injEq] at h
      subst ids
      exact .nil
  | cons piece pieces ih =>
      cases hp : tokenId? tokenizer piece with
      | none =>
          simp only [List.mapM_cons, hp] at h
          change Except.error _ = Except.ok ids at h
          cases h
      | some id =>
          simp only [List.mapM_cons, hp] at h
          change Except.map (List.cons id) _ = Except.ok ids at h
          cases ht : (pieces.mapM (fun piece =>
              match tokenId? tokenizer piece with
              | some id => pure id
              | none => throw s!"BPE piece is absent from vocabulary: {repr piece}") :
                Except String (List Nat)) with
          | error message => simp [ht, Except.map] at h
          | ok tail =>
              simp only [ht, Except.map, Except.ok.injEq] at h
              subst ids
              exact .cons hp (ih tail ht)

/-- Successful fragment encoding returns vocabulary ids for a merge-complete partition of the
original byte-escaped text. This statement does not assume that the vocabulary lookup is invertible.
-/
theorem encode_fragment_spec (tokenizer : Tokenizer) (fragment : String) (ids : Array Nat)
    (h : encodeFragment tokenizer fragment = .ok ids) :
    ∃ pieces : List String,
      String.join pieces = byteEncode fragment ∧
      (∀ left right, [left, right] <:+: pieces →
        mergeRank? tokenizer left right = none) ∧
      List.Forall₂ (fun piece id => tokenId? tokenizer piece = some id) pieces ids.toList := by
  let escaped := byteEncode fragment
  let pieces := bpeLoop tokenizer escaped.length (escaped.toList.map String.singleton)
  have hjoin : String.join pieces = escaped := by
    dsimp only [pieces]
    rw [join_bpe_loop]
    apply String.toList_injective
    simp [List.flatMap_map]
  have hcomplete : ∀ left right, [left, right] <:+: pieces →
      mergeRank? tokenizer left right = none := by
    have hnone : bestMerge? tokenizer pieces = none := by
      apply best_merge_bpe_loop_eq_none
      simp [String.length_toList]
    have hmissing := best_merge_spec_aux tokenizer pieces
    simp only [hnone] at hmissing
    intro left right hadj
    cases hr : mergeRank? tokenizer left right with
    | none => rfl
    | some rank => exact False.elim (hmissing left right rank hadj hr)
  change (do
    let result ← pieces.mapM (fun piece =>
      match tokenId? tokenizer piece with
      | some id => pure id
      | none => throw s!"BPE piece is absent from vocabulary: {repr piece}")
    pure result.toArray) = Except.ok ids at h
  cases hm : (pieces.mapM (fun piece =>
      match tokenId? tokenizer piece with
      | some id => pure id
      | none => throw s!"BPE piece is absent from vocabulary: {repr piece}") :
        Except String (List Nat)) with
  | error message =>
      rw [hm] at h
      contradiction
  | ok result =>
      rw [hm] at h
      change Except.ok result.toArray = Except.ok ids at h
      have hid := Except.ok.inj h
      subst ids
      refine ⟨pieces, hjoin, hcomplete, ?_⟩
      simpa using forall₂_token_id_of_mapM tokenizer pieces result hm

end TorchLean.text.GPT2BPE.Internal
