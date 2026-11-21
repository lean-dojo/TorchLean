// Lexical highlighting for ordinary Lean 4 fences. Checked Verso spans retain their hover data.
(function () {
  const keywords = new Set((
    "abbrev axiom by calc case cases class def deriving do else end example export extends " +
    "for from fun have if import in include inductive instance let match module mutable mut " +
    "namespace noncomputable notation opaque open partial private protected public return " +
    "section set_option structure termination_by then theorem try universe universes unsafe " +
    "variable variables where while with unless decreasing_by syntax macro macro_rules elab " +
    "scoped infix infixl infixr prefix postfix #check #eval #print #reduce #guard #synth"
  ).split(/\s+/));
  const tactics = new Set((
    "apply assumption change constructor contradiction decide exact ext first fun_prop " +
    "induction intro intros linarith native_decide norm_num obtain omega positivity rcases " +
    "refine rfl ring rintro rw simp simpa subst trivial unfold aesop nlinarith op_deriv " +
    "autograd roundoff converges"
  ).split(/\s+/));
  const identifier = /^[#]?[\p{L}_][\p{L}\p{N}_'!?]*/u;
  const number = /^(?:0[xX][\da-fA-F]+|0[bB][01]+|\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)/;

  function highlight(code) {
    if (code.dataset.leanHighlighted || code.closest(".hl.lean")) return;
    const text = code.textContent;
    const output = document.createDocumentFragment();
    let offset = 0;
    let declaration = false;
    function emit(value, kind) {
      const node = kind ? document.createElement("span") : document.createTextNode(value);
      if (kind) { node.className = "tl-lean-" + kind; node.textContent = value; }
      output.appendChild(node);
      offset += value.length;
    }
    while (offset < text.length) {
      const rest = text.slice(offset);
      if (rest.startsWith("--")) {
        const end = rest.indexOf("\n");
        emit(end < 0 ? rest : rest.slice(0, end), "comment");
      } else if (rest.startsWith("/-")) {
        let end = 2, depth = 1;
        while (end < rest.length && depth) {
          if (rest.startsWith("/-", end)) { depth++; end += 2; }
          else if (rest.startsWith("-/", end)) { depth--; end += 2; }
          else end++;
        }
        emit(rest.slice(0, end), "comment");
      } else if (/^'(?:\\.|[^'\\])'/u.test(rest)) {
        emit(/^'(?:\\.|[^'\\])'/u.exec(rest)[0], "string");
      } else if (rest[0] === '"') {
        let end = 1;
        while (end < rest.length) {
          if (rest[end] === "\\") end += 2;
          else if (rest[end++] === '"') break;
        }
        emit(rest.slice(0, end), "string");
      } else if (number.test(rest)) {
        emit(number.exec(rest)[0], "number");
      } else if (identifier.test(rest)) {
        const word = identifier.exec(rest)[0];
        const kind = keywords.has(word) ? "keyword" : tactics.has(word) ? "tactic" :
          declaration ? "name" : /^(?:[\p{Lu}]|[ℕℤℚℝℂ])/u.test(word) ? "type" : null;
        emit(word, kind);
        declaration = /^(?:def|theorem|abbrev|opaque|class|structure|inductive)$/.test(word);
      } else {
        const symbol = /^[^\p{L}\p{N}_\s"]/u.exec(rest);
        emit(symbol ? symbol[0] : rest[0], symbol ? "operator" : null);
      }
    }
    code.replaceChildren(output);
    code.dataset.leanHighlighted = "true";
  }

  function highlightAll() {
    document.querySelectorAll("pre code.language-lean, .language-lean pre code").forEach(highlight);
    document.querySelectorAll("main pre:not([class])").forEach(highlightRaw);
  }

  const rawKeywords = new Set((
    "as assert await break class continue def del elif else except finally for from global if " +
    "import in is lambda nonlocal pass raise return try while with yield True False None " +
    "lake git python python3 curl bash export echo cd lean torchlean verify " +
    "abbrev axiom by calc class def deriving do end example fun have if import in inductive " +
    "instance let match namespace open partial private return section structure theorem then where with"
  ).split(/\s+/));

  // Verso's ordinary fences render as bare <pre> elements. They are useful for
  // incomplete signatures and Python transcripts that cannot be Lean-elaborated.
  function highlightRaw(pre) {
    if (pre.dataset.tlHighlighted || pre.querySelector("code")) return;
    const source = pre.textContent;
    const code = document.createElement("code");
    code.className = "tl-raw-code";
    const tokens = /(?:--|\/\/|#(?!check\b|eval\b|print\b|guard\b))[^\n]*|"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'|(?:0[xX][\da-fA-F]+|\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)|[#]?[\p{L}_][\p{L}\p{N}_'!?]*/gu;
    let cursor = 0;
    for (const match of source.matchAll(tokens)) {
      const start = match.index;
      if (start > cursor) code.append(document.createTextNode(source.slice(cursor, start)));
      const value = match[0];
      let kind = "";
      if (value.startsWith("--") || value.startsWith("//") || value.startsWith("# ")) kind = "comment";
      else if (value.startsWith('"') || value.startsWith("'")) kind = "string";
      else if (/^(?:0[xX]|\d)/.test(value)) kind = "number";
      else if (rawKeywords.has(value) || keywords.has(value)) kind = "keyword";
      else if (/^[\p{Lu}ℕℤℚℝℂ]/u.test(value)) kind = "type";
      const node = kind ? document.createElement("span") : document.createTextNode(value);
      if (kind) { node.className = "tl-raw-" + kind; node.textContent = value; }
      code.append(node);
      cursor = start + value.length;
    }
    if (cursor < source.length) code.append(document.createTextNode(source.slice(cursor)));
    pre.replaceChildren(code);
    pre.dataset.tlHighlighted = "true";
  }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", highlightAll);
  else highlightAll();
  window.addEventListener("load", highlightAll);
})();
