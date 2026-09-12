/-
Copyright (c) 2026 TorchLean contributors
Released under MIT license as described in the file LICENSE.
Authors: TorchLean contributors
-/
module

public import NN.Tensor.Internal.Syntax.Parser.Expression.Config
public import NN.Tensor.Internal.Syntax.Render
public import NN.Tensor.Internal.Syntax.Diagnostic -- shake: keep
public import NN.Tensor.Internal.Syntax.Lexer -- shake: keep

/-!
# Expression token state

This module owns duplicate detection, group state, token decoding, and the
canonicalization invariants preserved by each state transition.
-/

@[expose] public section

namespace TorchLean.Tensor.Internal.Syntax

namespace Parser.Impl

/-- Duplicate-detection identity for a named axis or the single ellipsis marker. -/
inductive AxisKey where
  /-- Duplicate-detection key for a user-named axis. -/
  | named (name : String)
  /-- Duplicate-detection key shared by all ellipsis occurrences. -/
  | ellipsis
deriving DecidableEq

/-- Accumulated grouping and duplicate-detection state for one expression. -/
structure ExpressionState where
  /-- Axis identities already accepted in this expression. -/
  seen : List AxisKey := []
  /-- The currently open parenthesized group, if any. -/
  group : Option (Span × List (Located Axis)) := none
  /-- Completed physical axes in reverse source order. -/
  axesRev : List CompositeAxis := []

/--
Normalize decimal words to their canonical natural-number spelling while
leaving structural tokens unchanged.
-/
def canonicalTokenKind (policy : IdentifierPolicy) : TokenKind → TokenKind
  | .word text =>
      if policy.isDecimal text then
        match policy.toNat? text with
        | some value => .word (toString value)
        | none => .word text
      else
        .word text
  | token => token

/-- Canonicalizing decimal words does not change whether a token is an arrow. -/
theorem canonicalTokenKind_beq_arrow (policy : IdentifierPolicy)
    (token : TokenKind) :
    (canonicalTokenKind policy token == .arrow) = (token == .arrow) := by
  rw [Bool.eq_iff_iff, beq_iff_eq, beq_iff_eq]
  cases token with
  | word text =>
      simp only [canonicalTokenKind]
      split
      · split <;> simp
      · simp
  | leftParen | rightParen | ellipsis | arrow | comma | star =>
      simp [canonicalTokenKind]

/-- Canonicalizing decimal words preserves and reflects comma tokens. -/
theorem canonicalTokenKind_eq_comma (policy : IdentifierPolicy)
    (token : TokenKind) :
    canonicalTokenKind policy token = .comma ↔ token = .comma := by
  cases token with
  | word text =>
      simp only [canonicalTokenKind]
      split
      · split <;> simp
      · simp
  | leftParen => simp [canonicalTokenKind]
  | rightParen => simp [canonicalTokenKind]
  | ellipsis => simp [canonicalTokenKind]
  | arrow => simp [canonicalTokenKind]
  | comma => simp [canonicalTokenKind]
  | star => simp [canonicalTokenKind]

/-- Canonical token replacement cannot introduce an arrow into an arrow-free stream. -/
theorem findArrow_eq_none_of_canonical
    (policy : IdentifierPolicy) (sourceTokens targetTokens : List Token)
    (hTokens :
      targetTokens.map Located.value =
        sourceTokens.map fun token =>
          canonicalTokenKind policy token.value)
    (hFind :
      sourceTokens.find? (fun token => token.value == .arrow) = none) :
    targetTokens.find? (fun token => token.value == .arrow) = none := by
  rw [List.find?_eq_none] at hFind ⊢
  intro targetToken hTargetToken
  have hTargetValue :
      targetToken.value ∈ targetTokens.map Located.value :=
    List.mem_map_of_mem hTargetToken
  rw [hTokens] at hTargetValue
  obtain ⟨sourceToken, hSourceToken, hValue⟩ :=
    List.mem_map.mp hTargetValue
  rw [← hValue, canonicalTokenKind_beq_arrow]
  exact hFind sourceToken hSourceToken

/-- Construct a source-located duplicate-axis diagnostic. -/
def duplicateDiagnostic (span : Span) : Diagnostic :=
  { code := .duplicateAxis
    message := "axis occurs more than once in an indexing expression"
    span }

/--
Record one named axis or ellipsis while enforcing the expression's duplicate
policy.
-/
def registerKey (config : ExpressionConfig) (key : AxisKey)
    (span : Span) (seen : List AxisKey) : Result (List AxisKey) :=
  let repeatedUnderscore :=
    config.allowUnderscore && key == .named "_"
  if key == .ellipsis && key ∈ seen then
    .error
      { code := .duplicateEllipsis
        message := "an expression may contain only one ellipsis"
        span }
  else if key ∈ seen && !config.allowDuplicates && !repeatedUnderscore then
    .error (duplicateDiagnostic span)
  else
    .ok (key :: seen)

/-- Successful duplicate registration is independent of the diagnostic source span. -/
theorem registerKey_eq_ok_at_span (config : ExpressionConfig)
    (key : AxisKey) (sourceSpan targetSpan : Span)
    (seen nextSeen : List AxisKey)
    (hRegister :
      registerKey config key sourceSpan seen = .ok nextSeen) :
    registerKey config key targetSpan seen = .ok nextSeen := by
  unfold registerKey at hRegister ⊢
  split <;> simp_all
  split <;> simp_all

/-- Enforce einops underscore placement while permitting the wildcard `_`. -/
def hasForbiddenUnderscore (config : ExpressionConfig) (name : String) : Bool :=
  if config.allowUnderscore && name == "_" then
    false
  else
    name.startsWith "_" || name.endsWith "_"

/--
Interpret one word token as a named, anonymous, or unit axis and update
duplicate tracking.
-/
def decodeWord (policy : IdentifierPolicy) (config : ExpressionConfig)
    (token : Token) (state : ExpressionState) :
    Result (Located Axis × ExpressionState) := do
  let .word text := token.value
    | .error
        { code := .unexpectedToken
          message := "expected an axis identifier"
          span := token.span }
  if policy.isDecimal text then
    let some value := policy.toNat? text
      | .error
          { code := .invalidAnonymousAxis
            message := "anonymous axis is not a natural number"
            span := token.span }
    if value == 0 then
      .error
        { code := .invalidAnonymousAxis
          message := "anonymous axes must have positive length"
          span := token.span }
    else if value == 1 then
      .ok (⟨.unit, token.span⟩, state)
    else
      let axis := Axis.anonymous value token.span.offset
      .ok (⟨axis, token.span⟩, state)
  else if !policy.isIdentifier text || hasForbiddenUnderscore config text then
    .error
      { code := .invalidIdentifier
        message := "invalid axis identifier"
        span := token.span }
  else
    let seen ← registerKey config (.named text) token.span state.seen
    .ok (⟨.named text, token.span⟩, { state with seen })

/-- Decode and register a bare ellipsis token. -/
def decodeEllipsis (config : ExpressionConfig) (token : Token)
    (state : ExpressionState) : Result (Located Axis × ExpressionState) := do
  let seen ← registerKey config .ellipsis token.span state.seen
  .ok (⟨.ellipsis, token.span⟩, { state with seen })

/-- Append an axis to the open group or as a new top-level physical axis. -/
def pushAxis (axis : Located Axis) (state : ExpressionState) : ExpressionState :=
  match state.group with
  | none =>
      { state with
        axesRev :=
          { axes := [axis], parenthesized := false, span := axis.span } :: state.axesRev }
  | some (openSpan, axesRev) =>
      { state with group := some (openSpan, axis :: axesRev) }

namespace ExpressionState

/-- Render the parser state's completed axes and any currently open group. -/
def tokenKinds (state : ExpressionState) : List TokenKind :=
  state.axesRev.reverse.flatMap CompositeAxis.tokenKinds ++
    match state.group with
    | none => []
    | some (_, axesRev) =>
        .leftParen :: axesRev.reverse.map fun axis => axis.value.tokenKind

/-- Appending an axis to parser state appends its token kind to the rendered state. -/
theorem tokenKinds_pushAxis (axis : Located Axis)
    (state : ExpressionState) :
    (pushAxis axis state).tokenKinds =
      state.tokenKinds ++ [axis.value.tokenKind] := by
  cases hGroup : state.group with
  | none =>
      simp [tokenKinds, pushAxis, hGroup, CompositeAxis.tokenKinds]
  | some group =>
      obtain ⟨openSpan, axesRev⟩ := group
      simp [tokenKinds, pushAxis, hGroup]

/--
Relate parser states that differ only in source spans and canonical decimal
spelling.
-/
def CanonicalEq (left right : ExpressionState) : Prop :=
  left.seen = right.seen ∧
    left.tokenKinds = right.tokenKinds ∧
      left.group.isSome = right.group.isSome

/-- Pushing canonically equal axes preserves canonical parser-state equivalence. -/
theorem canonicalEq_pushAxis (leftAxis rightAxis : Located Axis)
    (leftState rightState : ExpressionState)
    (hState : CanonicalEq leftState rightState)
    (hAxis : leftAxis.value.tokenKind = rightAxis.value.tokenKind) :
    CanonicalEq (pushAxis leftAxis leftState) (pushAxis rightAxis rightState) := by
  rcases hState with ⟨hSeen, hKinds, hGroup⟩
  refine ⟨?_, ?_, ?_⟩
  · cases hLeft : leftState.group <;>
      cases hRight : rightState.group <;>
        simpa [pushAxis, hLeft, hRight] using hSeen
  · rw [tokenKinds_pushAxis, tokenKinds_pushAxis, hKinds, hAxis]
  · cases hLeft : leftState.group <;>
      cases hRight : rightState.group <;>
        simp [pushAxis, hLeft, hRight] at hGroup ⊢

/-- Replacing equal duplicate-tracking lists preserves canonical state equivalence. -/
theorem canonicalEq_setSeen (leftState rightState : ExpressionState)
    (leftSeen rightSeen : List AxisKey)
    (hState : CanonicalEq leftState rightState)
    (hSeen : leftSeen = rightSeen) :
    CanonicalEq
      { leftState with seen := leftSeen }
      { rightState with seen := rightSeen } := by
  rcases hState with ⟨_, hKinds, hGroup⟩
  exact ⟨hSeen, hKinds, hGroup⟩

/-- Opening corresponding empty groups preserves canonical state equivalence. -/
theorem canonicalEq_openGroup (leftState rightState : ExpressionState)
    (leftSpan rightSpan : Span)
    (hState : CanonicalEq leftState rightState)
    (hLeftGroup : leftState.group = none)
    (hRightGroup : rightState.group = none) :
    CanonicalEq
      { leftState with group := some (leftSpan, []) }
      { rightState with group := some (rightSpan, []) } := by
  refine ⟨hState.1, ?_, rfl⟩
  simpa [tokenKinds, hLeftGroup, hRightGroup, List.append_assoc] using
    congrArg (fun kinds => kinds ++ [.leftParen]) hState.2.1

/-- Closing corresponding groups preserves canonical token structure despite span changes. -/
theorem canonicalEq_closeGroup (leftState rightState : ExpressionState)
    (leftOpenSpan rightOpenSpan leftCloseSpan rightCloseSpan : Span)
    (leftAxesRev rightAxesRev : List (Located Axis))
    (hState : CanonicalEq leftState rightState)
    (hLeftGroup : leftState.group = some (leftOpenSpan, leftAxesRev))
    (hRightGroup : rightState.group = some (rightOpenSpan, rightAxesRev)) :
    CanonicalEq
      { leftState with
        group := none
        axesRev :=
          { axes := leftAxesRev.reverse
            parenthesized := true
            span := Span.join leftOpenSpan leftCloseSpan } ::
            leftState.axesRev }
      { rightState with
        group := none
        axesRev :=
          { axes := rightAxesRev.reverse
            parenthesized := true
            span := Span.join rightOpenSpan rightCloseSpan } ::
            rightState.axesRev } := by
  refine ⟨hState.1, ?_, rfl⟩
  simpa [tokenKinds, hLeftGroup, hRightGroup, CompositeAxis.tokenKinds,
      List.append_assoc] using
    congrArg (fun kinds => kinds ++ [.rightParen]) hState.2.1

end ExpressionState

/-- Successful word decoding appends the token's canonical kind to parser state. -/
theorem tokenKinds_pushAxis_decodeWord (policy : IdentifierPolicy)
    (config : ExpressionConfig) (token : Token) (state : ExpressionState)
    (axis : Located Axis) (nextState : ExpressionState)
    (hDecode : decodeWord policy config token state = .ok (axis, nextState)) :
    (pushAxis axis nextState).tokenKinds =
      state.tokenKinds ++ [canonicalTokenKind policy token.value] := by
  cases hValue : token.value with
  | word text =>
      by_cases hDecimal : policy.isDecimal text = true
      · cases hNat : policy.toNat? text with
        | none =>
            simp [decodeWord, hValue, hDecimal, hNat] at hDecode
        | some value =>
            by_cases hZero : value = 0
            · simp [decodeWord, hValue, hDecimal, hNat, hZero] at hDecode
            · by_cases hOne : value = 1
              · simp [decodeWord, hValue, hDecimal, hNat, hOne] at hDecode
                rcases hDecode with ⟨rfl, rfl⟩
                rw [ExpressionState.tokenKinds_pushAxis]
                simp [canonicalTokenKind, hDecimal, hNat, hOne,
                  Axis.tokenKind]
                apply String.ext
                simp [Nat.toList_repr, Nat.toDigits_of_lt_base]
              · simp [decodeWord, hValue, hDecimal, hNat, hZero, hOne] at hDecode
                rcases hDecode with ⟨rfl, rfl⟩
                rw [ExpressionState.tokenKinds_pushAxis]
                simp [canonicalTokenKind, hDecimal, hNat, Axis.tokenKind]
      · by_cases hInvalid :
          (!policy.isIdentifier text ||
            hasForbiddenUnderscore config text) = true
        · simp [decodeWord, hValue, hDecimal, hInvalid] at hDecode
        · cases hRegister :
            registerKey config (.named text) token.span state.seen with
          | error diagnostic =>
              simp [decodeWord, hValue, hDecimal, hInvalid, hRegister] at hDecode
              change
                (Except.error diagnostic :
                  Result (Located Axis × ExpressionState)) =
                    .ok (axis, nextState) at hDecode
              contradiction
          | ok seen =>
              simp [decodeWord, hValue, hDecimal, hInvalid, hRegister] at hDecode
              rcases hDecode with ⟨rfl, rfl⟩
              rw [ExpressionState.tokenKinds_pushAxis]
              simp [canonicalTokenKind, hDecimal, Axis.tokenKind,
                ExpressionState.tokenKinds]
  | leftParen => simp [decodeWord, hValue] at hDecode
  | rightParen => simp [decodeWord, hValue] at hDecode
  | ellipsis => simp [decodeWord, hValue] at hDecode
  | arrow => simp [decodeWord, hValue] at hDecode
  | comma => simp [decodeWord, hValue] at hDecode
  | star => simp [decodeWord, hValue] at hDecode

/-- Decoding a canonical word spelling succeeds with an equivalent next parser state. -/
theorem decodeWord_canonical
    (config : ExpressionConfig) (sourceToken targetToken : Token)
    (sourceState targetState : ExpressionState)
    (sourceAxis : Located Axis) (sourceNextState : ExpressionState)
    (hValue :
      targetToken.value =
        canonicalTokenKind IdentifierPolicy.pythonUnicode sourceToken.value)
    (hState : ExpressionState.CanonicalEq sourceState targetState)
    (hDecode :
      decodeWord IdentifierPolicy.pythonUnicode config sourceToken sourceState =
        .ok (sourceAxis, sourceNextState)) :
    ∃ targetAxis targetNextState,
      decodeWord IdentifierPolicy.pythonUnicode config targetToken targetState =
          .ok (targetAxis, targetNextState) ∧
        ExpressionState.CanonicalEq
          (pushAxis sourceAxis sourceNextState)
          (pushAxis targetAxis targetNextState) := by
  cases hSourceValue : sourceToken.value with
  | word text =>
      by_cases hDecimal :
          IdentifierPolicy.pythonUnicode.isDecimal text = true
      · cases hNat :
          IdentifierPolicy.pythonUnicode.toNat? text with
        | none =>
            simp [decodeWord, hSourceValue, hDecimal, hNat] at hDecode
        | some value =>
            by_cases hZero : value = 0
            · simp [decodeWord, hSourceValue, hDecimal, hNat, hZero] at hDecode
            · by_cases hOne : value = 1
              · simp [decodeWord, hSourceValue, hDecimal, hNat, hOne] at hDecode
                rcases hDecode with ⟨rfl, rfl⟩
                have hTargetValue :
                    targetToken.value = .word (toString value) := by
                  simpa [canonicalTokenKind, hSourceValue, hDecimal, hNat]
                    using hValue
                refine
                  ⟨⟨.unit, targetToken.span⟩, targetState, ?_, ?_⟩
                · have hTargetDecimal :=
                    IdentifierPolicy.pythonUnicode_isDecimal_toString value
                  have hTargetNat :=
                    IdentifierPolicy.pythonUnicode_toNat?_toString value
                  simp only [decodeWord, hTargetValue]
                  rw [hTargetDecimal, hTargetNat]
                  simp [hOne]
                · exact
                    ExpressionState.canonicalEq_pushAxis _ _
                      sourceState targetState hState rfl
              · simp [decodeWord, hSourceValue, hDecimal, hNat, hZero, hOne]
                  at hDecode
                rcases hDecode with ⟨rfl, rfl⟩
                have hTargetValue :
                    targetToken.value = .word (toString value) := by
                  simpa [canonicalTokenKind, hSourceValue, hDecimal, hNat]
                    using hValue
                refine
                  ⟨⟨.anonymous value targetToken.span.offset, targetToken.span⟩,
                    targetState, ?_, ?_⟩
                · have hTargetDecimal :=
                    IdentifierPolicy.pythonUnicode_isDecimal_toString value
                  have hTargetNat :=
                    IdentifierPolicy.pythonUnicode_toNat?_toString value
                  simp only [decodeWord, hTargetValue]
                  rw [hTargetDecimal, hTargetNat]
                  simp [hZero, hOne]
                · exact
                    ExpressionState.canonicalEq_pushAxis _ _
                      sourceState targetState hState rfl
      · by_cases hInvalid :
          (!IdentifierPolicy.pythonUnicode.isIdentifier text ||
            hasForbiddenUnderscore config text) = true
        · simp [decodeWord, hSourceValue, hDecimal, hInvalid] at hDecode
        · cases hRegister :
            registerKey config (.named text) sourceToken.span
              sourceState.seen with
          | error diagnostic =>
              simp [decodeWord, hSourceValue, hDecimal, hInvalid, hRegister]
                at hDecode
              change
                (Except.error diagnostic :
                    Result (Located Axis × ExpressionState)) =
                  .ok (sourceAxis, sourceNextState) at hDecode
              contradiction
          | ok nextSeen =>
              simp [decodeWord, hSourceValue, hDecimal, hInvalid, hRegister]
                at hDecode
              rcases hDecode with ⟨rfl, rfl⟩
              have hTargetValue :
                  targetToken.value = .word text := by
                simpa [canonicalTokenKind, hSourceValue, hDecimal] using hValue
              have hTargetRegister :
                  registerKey config (.named text) targetToken.span
                      targetState.seen =
                    .ok nextSeen := by
                rw [← hState.1]
                exact
                  registerKey_eq_ok_at_span config (.named text)
                    sourceToken.span targetToken.span sourceState.seen nextSeen
                    hRegister
              refine
                ⟨⟨.named text, targetToken.span⟩,
                  { targetState with seen := nextSeen }, ?_, ?_⟩
              · simp [decodeWord, hTargetValue, hDecimal, hInvalid,
                  hTargetRegister]
                rfl
              · apply ExpressionState.canonicalEq_pushAxis
                · exact
                    ExpressionState.canonicalEq_setSeen _ _ nextSeen nextSeen
                      hState rfl
                · rfl
  | leftParen => simp [decodeWord, hSourceValue] at hDecode
  | rightParen => simp [decodeWord, hSourceValue] at hDecode
  | ellipsis => simp [decodeWord, hSourceValue] at hDecode
  | arrow => simp [decodeWord, hSourceValue] at hDecode
  | comma => simp [decodeWord, hSourceValue] at hDecode
  | star => simp [decodeWord, hSourceValue] at hDecode

/-- Successful ellipsis decoding appends exactly one ellipsis token kind. -/
theorem tokenKinds_pushAxis_decodeEllipsis (config : ExpressionConfig)
    (token : Token) (state : ExpressionState) (axis : Located Axis)
    (nextState : ExpressionState)
    (hDecode : decodeEllipsis config token state = .ok (axis, nextState)) :
    (pushAxis axis nextState).tokenKinds =
      state.tokenKinds ++ [.ellipsis] := by
  cases hRegister :
      registerKey config .ellipsis token.span state.seen with
  | error diagnostic =>
      simp [decodeEllipsis, hRegister] at hDecode
      change
        (Except.error diagnostic : Result (Located Axis × ExpressionState)) =
          .ok (axis, nextState) at hDecode
      contradiction
  | ok seen =>
      simp [decodeEllipsis, hRegister] at hDecode
      rcases hDecode with ⟨rfl, rfl⟩
      rw [ExpressionState.tokenKinds_pushAxis]
      simp [Axis.tokenKind, ExpressionState.tokenKinds]

/-- Re-decoding an ellipsis at a new span preserves canonical parser state. -/
theorem decodeEllipsis_canonical
    (config : ExpressionConfig) (sourceToken targetToken : Token)
    (sourceState targetState : ExpressionState)
    (sourceAxis : Located Axis) (sourceNextState : ExpressionState)
    (hState : ExpressionState.CanonicalEq sourceState targetState)
    (hDecode :
      decodeEllipsis config sourceToken sourceState =
        .ok (sourceAxis, sourceNextState)) :
    ∃ targetAxis targetNextState,
      decodeEllipsis config targetToken targetState =
          .ok (targetAxis, targetNextState) ∧
        ExpressionState.CanonicalEq
          (pushAxis sourceAxis sourceNextState)
          (pushAxis targetAxis targetNextState) := by
  cases hRegister :
      registerKey config .ellipsis sourceToken.span sourceState.seen with
  | error diagnostic =>
      simp [decodeEllipsis, hRegister] at hDecode
      change
        (Except.error diagnostic : Result (Located Axis × ExpressionState)) =
          .ok (sourceAxis, sourceNextState) at hDecode
      contradiction
  | ok nextSeen =>
      simp [decodeEllipsis, hRegister] at hDecode
      rcases hDecode with ⟨rfl, rfl⟩
      have hTargetRegister :
          registerKey config .ellipsis targetToken.span targetState.seen =
            .ok nextSeen := by
        rw [← hState.1]
        exact
          registerKey_eq_ok_at_span config .ellipsis sourceToken.span
            targetToken.span sourceState.seen nextSeen hRegister
      refine
        ⟨⟨.ellipsis, targetToken.span⟩,
          { targetState with seen := nextSeen }, ?_, ?_⟩
      · simp [decodeEllipsis, hTargetRegister]
        rfl
      · apply ExpressionState.canonicalEq_pushAxis
        · exact
            ExpressionState.canonicalEq_setSeen _ _ nextSeen nextSeen
              hState rfl
        · rfl


end Parser.Impl

end TorchLean.Tensor.Internal.Syntax
