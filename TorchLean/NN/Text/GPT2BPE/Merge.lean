/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.Text.GPT2BPE
public import Batteries.Data.List.Basic

/-!
# Correctness of GPT-2 BPE merges

The merge passes proved correct here are the ones `GPT2BPE.encodeFragment` runs. Each
pass merges every occurrence of the adjacent pair with minimum stored rank. Merging moves token
boundaries but keeps the concatenated byte-escaped text, whatever the rank table, and the fuel
`encodeFragment` passes (one round per symbol) leaves no ranked adjacent pair.
-/

@[expose] public section

namespace TorchLean.Text.GPT2BPE

/-- Replacing adjacent pairs by their concatenation preserves the represented text. -/
theorem join_applyMerge (left right : String) (pieces : List String) :
    String.join (applyMerge left right pieces) = String.join pieces := by
  induction pieces using applyMerge.induct left right with
  | case1 first second rest h ih =>
      simp only [applyMerge, h, ↓reduceIte, String.join_cons, ih, String.append_assoc]
  | case2 first second rest h ih =>
      simp [applyMerge, h, ih]
  | case3 pieces h => simp [applyMerge]

/-- Every fuel-bounded BPE run preserves the text of its input pieces. -/
theorem join_applyMerges (tokenizer : Tables) (fuel : Nat) (pieces : List String) :
    String.join (applyMerges tokenizer fuel pieces) = String.join pieces := by
  induction fuel generalizing pieces with
  | zero => rfl
  | succ fuel ih =>
      simp only [applyMerges]
      split
      · rfl
      · rw [ih, join_applyMerge]

/-- A merge pass never increases the number of pieces. -/
theorem length_applyMerge_le (left right : String) (pieces : List String) :
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

private theorem bestMerge?_spec_aux (tokenizer : Tables) (pieces : List String) :
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
                    · exact Nat.le_trans hle (ih.2.2 left right otherRank htail havail)
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
theorem bestMerge?_spec (tokenizer : Tables) (pieces : List String)
    (candidate : MergeRank) (h : bestMerge? tokenizer pieces = some candidate) :
    [candidate.left, candidate.right] <:+: pieces ∧
    mergeRank? tokenizer candidate.left candidate.right = some candidate.rank ∧
    ∀ left right rank, [left, right] <:+: pieces →
      mergeRank? tokenizer left right = some rank → candidate.rank ≤ rank := by
  have hs := bestMerge?_spec_aux tokenizer pieces
  simpa only [h] using hs

/-- A merge selected from the current adjacent pairs strictly decreases the piece count. -/
theorem length_applyMerge_lt (tokenizer : Tables)
    (pieces : List String) (candidate : MergeRank)
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
            have := length_applyMerge_le first second rest
            omega
          have tail_progress (selected : MergeRank)
              (ht : bestMerge? tokenizer (second :: rest) = some selected) :
              (applyMerge selected.left selected.right (first :: second :: rest)).length <
                (first :: second :: rest).length := by
            simp only [applyMerge]
            split
            · simp only [List.length_cons]
              have := length_applyMerge_le selected.left selected.right rest
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
theorem bestMerge?_applyMerges_eq_none (tokenizer : Tables) (fuel : Nat)
    (pieces : List String) (hlen : pieces.length ≤ fuel + 1) :
    bestMerge? tokenizer (applyMerges tokenizer fuel pieces) = none := by
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
      simp only [applyMerges]
      split
      · assumption
      · rename_i candidate hcandidate
        apply ih
        have hprogress := length_applyMerge_lt tokenizer pieces candidate hcandidate
        omega

private theorem forall₂_tokenId?_of_mapM (tokenizer : Tables)
    (pieces : List String) (ids : List Nat)
    (h : pieces.mapM (requireTokenId tokenizer) = (Except.ok ids : Except String (List Nat))) :
    List.Forall₂ (fun piece id => tokenId? tokenizer piece = some id) pieces ids := by
  induction pieces generalizing ids with
  | nil =>
      simp only [List.mapM_nil, pure, Except.pure, Except.ok.injEq] at h
      subst ids
      exact .nil
  | cons piece pieces ih =>
      cases hp : tokenId? tokenizer piece with
      | none =>
          simp only [List.mapM_cons, requireTokenId, hp] at h
          change Except.error _ = Except.ok ids at h
          cases h
      | some id =>
          simp only [List.mapM_cons, requireTokenId, hp] at h
          change Except.map (List.cons id) _ = Except.ok ids at h
          cases ht : (pieces.mapM (requireTokenId tokenizer) : Except String (List Nat)) with
          | error message => simp [ht, Except.map] at h
          | ok tail =>
              simp only [ht, Except.map, Except.ok.injEq] at h
              subst ids
              exact .cons hp (ih tail ht)

/-- Successful fragment encoding returns vocabulary ids for a merge-complete partition of the
original byte-escaped text. Each id is the `tokenId?` lookup of its piece, so the vocabulary need
not be invertible.
-/
theorem encodeFragment_spec (tokenizer : Tables) (fragment : String) (ids : Array Nat)
    (h : encodeFragment tokenizer fragment = .ok ids) :
    ∃ pieces : List String,
      String.join pieces = byteEncode fragment ∧
      (∀ left right, [left, right] <:+: pieces →
        mergeRank? tokenizer left right = none) ∧
      List.Forall₂ (fun piece id => tokenId? tokenizer piece = some id) pieces ids.toList := by
  let escaped := byteEncode fragment
  let pieces := applyMerges tokenizer escaped.length (escaped.toList.map String.singleton)
  have hjoin : String.join pieces = escaped := by
    dsimp only [pieces]
    rw [join_applyMerges]
    apply String.toList_injective
    simp [List.flatMap_map]
  have hcomplete : ∀ left right, [left, right] <:+: pieces →
      mergeRank? tokenizer left right = none := by
    have hnone : bestMerge? tokenizer pieces = none := by
      apply bestMerge?_applyMerges_eq_none
      simp [String.length_toList]
    have hmissing := bestMerge?_spec_aux tokenizer pieces
    simp only [hnone] at hmissing
    intro left right hadj
    cases hr : mergeRank? tokenizer left right with
    | none => rfl
    | some rank => exact False.elim (hmissing left right rank hadj hr)
  change (do
    let result ← pieces.mapM (requireTokenId tokenizer)
    pure result.toArray) = Except.ok ids at h
  cases hm : (pieces.mapM (requireTokenId tokenizer) : Except String (List Nat)) with
  | error message =>
      rw [hm] at h
      contradiction
  | ok result =>
      rw [hm] at h
      change Except.ok result.toArray = Except.ok ids at h
      have hid := Except.ok.inj h
      subst ids
      refine ⟨pieces, hjoin, hcomplete, ?_⟩
      simpa using forall₂_tokenId?_of_mapM tokenizer pieces result hm

end TorchLean.Text.GPT2BPE
