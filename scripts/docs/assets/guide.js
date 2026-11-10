(function () {
  const pages = TORCHLEAN_GUIDE_PAGES;
  const storageKey = "torchlean-guide-read-v1";
  const script = document.currentScript;
  const rootUrl = new URL(".", script ? script.src : window.location.href);

  function installDocstringMath() {
    if (!window.marked || !window.katex) return;
    // Imported declarations use Markdown docstrings, rendered at window.load.
    // Tokenize math before Markdown consumes TeX escapes or underscores.
    const rules = [
      {
        name: "torchleanDisplayMath",
        level: "block",
        pattern: /^\$\$[ \t]*\n([\s\S]+?)\n\$\$[ \t]*(?:\n|$)/,
        displayMode: true,
      },
      {
        name: "torchleanInlineMath",
        level: "inline",
        pattern: /^\$(?!\$)((?:\\.|[^$\\\n])+?)\$(?!\$)/,
        displayMode: false,
      },
    ];
    window.marked.use({extensions: rules.map((rule) => ({
      name: rule.name,
      level: rule.level,
      start(src) {
        return rule.displayMode ? src.search(/^\$\$[ \t]*\n/m) : src.indexOf("$");
      },
      tokenizer(src) {
        const match = rule.pattern.exec(src);
        if (match) return {type: rule.name, raw: match[0], text: match[1]};
      },
      renderer(token) {
        const rendered = window.katex.renderToString(token.text, {
          throwOnError: false,
          displayMode: rule.displayMode,
        });
        return rule.displayMode
          ? '<div class="tl-docstring-math">' + rendered + "</div>"
          : rendered;
      },
    }))});
  }

  function normalizedCurrentPage() {
    const here = new URL(window.location.href);
    const rootPath = rootUrl.pathname.endsWith("/") ? rootUrl.pathname : rootUrl.pathname + "/";
    if (!here.pathname.startsWith(rootPath)) return null;
    let rel = decodeURIComponent(here.pathname.slice(rootPath.length));
    if (rel === "") rel = "index.html";
    else if (rel.endsWith("/")) rel += "index.html";
    else if (!rel.endsWith(".html")) rel += "/index.html";
    return rel;
  }

  function loadReadSet() {
    try {
      return new Set(JSON.parse(localStorage.getItem(storageKey) || "[]"));
    } catch (_) {
      return new Set();
    }
  }

  function saveReadSet(set) {
    try {
      localStorage.setItem(storageKey, JSON.stringify(Array.from(set).sort()));
    } catch (_) {
      /* localStorage can be unavailable in privacy modes, and the page works without it. */
    }
  }

  function mountGuideNav() {
    const header = document.querySelector("body > header");
    if (!header || header.querySelector(".tl-guide-nav")) return;

    const titleWrapper = header.querySelector(".header-title-wrapper");
    const guideHome = rootUrl.href;
    const siteRoot = new URL("../", rootUrl);
    const links = [
      ["Main site", "./"],
      ["Installation", "installation/"],
      ["Examples", "examples/"],
      ["API Reference", "docs/"],
      ["Graphs", "graphs/"],
      ["Tools", "tools/"],
    ];

    document.querySelectorAll(".header-title").forEach((a) => {
      a.setAttribute("href", siteRoot.href);
      a.setAttribute("aria-label", "Back to the TorchLean main site");
    });

    document.querySelectorAll(".toc-title").forEach((a) => {
      a.setAttribute("href", guideHome);
    });

    const nav = document.createElement("nav");
    nav.className = "tl-guide-nav";
    nav.setAttribute("aria-label", "TorchLean site navigation");
    for (const [label, path] of links) {
      const a = document.createElement("a");
      a.href = new URL(path, siteRoot).href;
      a.textContent = label;
      nav.appendChild(a);
    }
    if (titleWrapper) {
      titleWrapper.appendChild(nav);
    } else {
      const search = header.querySelector("#search-wrapper");
      header.insertBefore(nav, search || null);
    }
  }

  function polishGuideHomepage() {
    if (normalizedCurrentPage() !== "index.html") return;

    const wrapper = document.querySelector("main .content-wrapper");
    const topNext = wrapper && wrapper.querySelector(":scope > .prev-next-buttons");
    if (topNext) topNext.remove();

    const duplicateContents = wrapper && wrapper.querySelector(":scope > section > section");
    if (duplicateContents && duplicateContents.querySelector("ol.section-toc")) {
      duplicateContents.remove();
    }

    const titlePermalink = document.querySelector("main .titlepage > h1 .permalink-widget");
    if (titlePermalink) titlePermalink.remove();
  }

  function scrollRatio() {
    const doc = document.documentElement;
    const max = Math.max(1, doc.scrollHeight - window.innerHeight);
    return Math.max(0, Math.min(1, window.scrollY / max));
  }

  function mountProgress() {
    const current = normalizedCurrentPage();
    if (!current || pages.indexOf(current) === -1) return;

    const root = document.createElement("div");
    root.className = "tl-reading-progress";
    root.setAttribute("role", "status");
    root.setAttribute("aria-live", "polite");
    root.innerHTML =
      '<div class="tl-reading-progress-track" aria-hidden="true">' +
      '<div class="tl-reading-progress-bar"></div></div>' +
      '<div class="tl-reading-progress-label"></div>';
    document.body.prepend(root);
    document.body.classList.add("tl-progress-mounted");

    const bar = root.querySelector(".tl-reading-progress-bar");
    const label = root.querySelector(".tl-reading-progress-label");
    const read = loadReadSet();

    function maybeMarkRead(ratio) {
      const doc = document.documentElement;
      const max = doc.scrollHeight - window.innerHeight;
      if (ratio >= 0.72 || max < 700) {
        if (!read.has(current)) {
          read.add(current);
          saveReadSet(read);
        }
      }
    }

    function update() {
      const ratio = scrollRatio();
      maybeMarkRead(ratio);
      const pct = Math.round(ratio * 100);
      bar.style.width = pct + "%";
      label.textContent = "Guide progress: " + read.size + "/" + pages.length + " sections read • " + pct + "% of this page";
    }

    update();
    window.addEventListener("scroll", update, { passive: true });
    window.addEventListener("resize", update);
    window.setTimeout(update, 1200);
  }

  function shouldSkipTextNode(node) {
    let el = node.parentElement;
    while (el) {
      const tag = el.tagName;
      if (tag === "A" || tag === "CODE" || tag === "PRE" || tag === "SCRIPT" ||
          tag === "STYLE" || tag === "TEXTAREA" || tag === "INPUT") {
        return true;
      }
      el = el.parentElement;
    }
    return false;
  }

  function autoLinkBareUrls() {
    const root = document.querySelector("main");
    if (!root) return;
    const urlRe = /\bhttps?:\/\/[^\s<>"']*[^\s<>"'.,;:!?)\]]/g;
    const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
    const nodes = [];
    while (true) {
      const node = walker.nextNode();
      if (!node) break;
      if (!shouldSkipTextNode(node) && urlRe.test(node.nodeValue)) {
        nodes.push(node);
      }
      urlRe.lastIndex = 0;
    }
    for (const node of nodes) {
      const text = node.nodeValue;
      const frag = document.createDocumentFragment();
      let last = 0;
      text.replace(urlRe, (match, offset) => {
        if (offset > last) frag.appendChild(document.createTextNode(text.slice(last, offset)));
        const a = document.createElement("a");
        a.href = match;
        a.textContent = match;
        a.className = "tl-auto-link";
        frag.appendChild(a);
        last = offset + match.length;
        return match;
      });
      if (last < text.length) frag.appendChild(document.createTextNode(text.slice(last)));
      node.parentNode.replaceChild(frag, node);
    }
  }

  function replaceAsciiArrowsInText() {
    const root = document.querySelector("main");
    if (!root) return;
    const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
    const nodes = [];
    while (true) {
      const node = walker.nextNode();
      if (!node) break;
      if (!shouldSkipTextNode(node) && node.nodeValue.includes("->")) {
        nodes.push(node);
      }
    }
    for (const node of nodes) {
      node.nodeValue = node.nodeValue.replace(/->/g, "→");
    }
  }

  function externalLinksOpenInNewTabs() {
    const here = window.location.origin;
    document.querySelectorAll('a[href^="http://"], a[href^="https://"]').forEach((a) => {
      let url;
      try { url = new URL(a.href); } catch (_) { return; }
      if (url.origin !== here) {
        a.target = "_blank";
        a.rel = "noopener noreferrer";
      }
    });
    document.querySelectorAll('a[href*="/docs/"], a[href^="../docs/"], a[href^="../../docs/"], a[href^="../../../docs/"]').forEach((a) => {
      a.target = "_blank";
      a.rel = "noopener noreferrer";
    });
  }

  function enhanceRouteLists() {
    document.querySelectorAll("main ol").forEach((ol) => {
      const text = ol.textContent || "";
      if (!text.includes("I want to") || !(text.includes("→") || text.includes("->"))) return;
      ol.classList.add("torchlean-route-list");
      const walker = document.createTreeWalker(ol, NodeFilter.SHOW_TEXT);
      const nodes = [];
      while (true) {
        const node = walker.nextNode();
        if (!node) break;
        if (!shouldSkipTextNode(node) && (node.nodeValue.includes("→") || node.nodeValue.includes("->"))) {
          nodes.push(node);
        }
      }
      for (const node of nodes) {
        const pieces = node.nodeValue.split(/(→|->)/g);
        const frag = document.createDocumentFragment();
        for (const piece of pieces) {
          if (piece === "→" || piece === "->") {
            const span = document.createElement("span");
            span.className = "tl-route-arrow";
            span.textContent = "→";
            frag.appendChild(span);
          } else if (piece) {
            frag.appendChild(document.createTextNode(piece));
          }
        }
        node.parentNode.replaceChild(frag, node);
      }
    });
  }

  function groupLeanExamples() {
    const heading = (text, output = false) => {
      const el = document.createElement("div");
      el.className = "tl-example-heading" + (output ? " tl-output-heading" : "");
      const label = document.createElement("span");
      label.textContent = text;
      el.appendChild(label);
      return el;
    };
    const appendOutput = (panel, output) => {
      const label = output.classList.contains("error") ? "Error output"
        : output.classList.contains("warning") ? "Warning output" : "Output";
      panel.appendChild(heading(label, true));
      panel.appendChild(output);
    };
    document.querySelectorAll("main code.hl.lean.block").forEach((code) => {
      if (code.closest(".tl-lean-example, details.bp_code_block")) return;
      const panel = document.createElement("div");
      panel.className = "tl-lean-example";
      code.before(panel);
      const title = heading("Code");
      const copy = document.createElement("button");
      copy.type = "button";
      copy.className = "tl-code-action";
      copy.textContent = "Copy code";
      copy.addEventListener("click", async () => {
        const source = code.cloneNode(true);
        source.querySelectorAll(".hover-container").forEach((el) => el.remove());
        try {
          await navigator.clipboard.writeText(source.textContent || "");
          copy.textContent = "Copied";
        } catch (_) {
          copy.textContent = "Copy unavailable";
        }
        window.setTimeout(() => { copy.textContent = "Copy code"; }, 1100);
      });
      title.appendChild(copy);
      panel.appendChild(title);
      panel.appendChild(code);
      while (panel.nextElementSibling?.matches("pre.lean-output")) {
        appendOutput(panel, panel.nextElementSibling);
      }
    });
    // Some chapters discuss a result before displaying it. Label these in place,
    // because moving them across prose would change the chapter's reading order.
    document.querySelectorAll("main pre.lean-output").forEach((output) => {
      if (output.closest(".tl-lean-example")) return;
      const panel = document.createElement("div");
      panel.className = "tl-lean-example";
      output.before(panel);
      appendOutput(panel, output);
    });
  }

  function simplifyPageNavigation() {
    const page = normalizedCurrentPage();
    if (!page || page.split("/").length < 3) return;
    const panels = document.querySelectorAll("#toc .split-toc");
    const local = panels[panels.length - 1];
    if (local && local.querySelector(".title .current")) local.remove();
  }

  function enhanceTerminals() {
    document.querySelectorAll('main .tl-terminal[data-output="false"]').forEach((panel) => {
      const pre = panel.querySelector(":scope > pre");
      const heading = panel.querySelector(":scope > .tl-terminal-heading");
      if (!pre || !heading) return;
      const next = panel.nextElementSibling;
      if (next && next.matches('.tl-terminal[data-output="true"]')) panel.appendChild(next);
      const button = document.createElement("button");
      button.type = "button";
      button.className = "tl-copy-command";
      button.textContent = "Copy command";
      button.addEventListener("click", async () => {
        try {
          await navigator.clipboard.writeText(pre.textContent);
          button.textContent = "Copied";
        } catch (_) {
          const selection = window.getSelection();
          const range = document.createRange();
          range.selectNodeContents(pre);
          selection.removeAllRanges();
          selection.addRange(range);
          button.textContent = "Command selected";
        }
        window.setTimeout(() => { button.textContent = "Copy command"; }, 1100);
      });
      heading.appendChild(button);
    });
  }

  function addCopyButtons() {
    document.querySelectorAll("main pre").forEach((pre) => {
      // Imported docstrings become prose at window.load, after this pass.
      if (pre.closest(".tl-code-wrap, .tl-lean-example, .tl-terminal")
          || pre.matches(".lean-output, .docstring")) return;
      const wrap = document.createElement("div");
      wrap.className = "tl-code-wrap";
      pre.parentNode.insertBefore(wrap, pre);
      wrap.appendChild(pre);
      const button = document.createElement("button");
      button.type = "button";
      button.className = "tl-copy-code";
      button.textContent = "Copy";
      button.addEventListener("click", async () => {
        const text = pre.innerText;
        try {
          await navigator.clipboard.writeText(text);
          button.textContent = "Copied";
          window.setTimeout(() => { button.textContent = "Copy"; }, 1100);
        } catch (_) {
          button.textContent = "Select";
          window.setTimeout(() => { button.textContent = "Copy"; }, 1100);
        }
      });
      wrap.appendChild(button);
    });
  }

  function wrapTables() {
    document.querySelectorAll("main table.tabular").forEach((table) => {
      if (table.closest(".tl-table-wrap")) return;
      const wrap = document.createElement("div");
      wrap.className = "tl-table-wrap";
      const firstRow = table.querySelector("tr");
      const columns = firstRow ? firstRow.children.length : 0;
      if (columns > 2) {
        wrap.classList.add("tl-table-wide");
        if (columns > 3) wrap.classList.add("tl-table-very-wide");
        const hint = document.createElement("div");
        hint.className = "tl-table-hint";
        hint.textContent = "Scroll horizontally to see every column →";
        wrap.appendChild(hint);
      }
      table.parentNode.insertBefore(wrap, table);
      wrap.appendChild(table);
    });
  }

  function codeTextOf(code) {
    if (!code) return "";
    return code.innerText || code.textContent || "";
  }

  function openInLeanLive(code) {
    const url = "https://live.lean-lang.org/#code=" + encodeURIComponent(code);
    window.open(url, "_blank", "noopener,noreferrer");
  }

  function openLeanCodePanels() {
    document.querySelectorAll("main details.bp_code_block").forEach((details) => {
      details.open = true;
    });
  }

  function addLeanCodePanelActions() {
    document.querySelectorAll("main details.bp_code_block").forEach((details) => {
      const summary = details.querySelector(":scope > summary");
      const code = details.querySelector(":scope > code.hl.lean.block");
      if (!summary || !code || summary.querySelector(".tl-code-actions")) return;

      const actions = document.createElement("span");
      actions.className = "tl-code-actions";
      actions.addEventListener("click", (event) => event.stopPropagation());

      const copy = document.createElement("button");
      copy.type = "button";
      copy.className = "tl-code-action";
      copy.textContent = "Copy";
      copy.title = "Copy this Lean snippet for local use in the TorchLean repository.";
      copy.addEventListener("click", async () => {
        try {
          await navigator.clipboard.writeText(codeTextOf(code));
          copy.textContent = "Copied";
          window.setTimeout(() => { copy.textContent = "Copy"; }, 1100);
        } catch (_) {
          copy.textContent = "Select";
          window.setTimeout(() => { copy.textContent = "Copy"; }, 1100);
        }
      });

      const live = document.createElement("button");
      live.type = "button";
      live.className = "tl-code-action tl-code-action-live";
      live.textContent = "Live ↪";
      live.title = "Open in live.lean-lang.org. TorchLean-specific snippets still need the local TorchLean project to typecheck.";
      live.addEventListener("click", () => openInLeanLive(codeTextOf(code)));

      actions.appendChild(copy);
      actions.appendChild(live);
      summary.appendChild(actions);
    });
  }


  function moveDisplayMathPunctuation() {
    document.querySelectorAll("main code.math.display, main code.bp_math.display").forEach((math) => {
      const tail = math.nextSibling;
      if (!tail || tail.nodeType !== Node.TEXT_NODE) return;
      const match = tail.nodeValue.match(/^([.,;:])(?=\s|$)/);
      if (!match) return;
      math.textContent = math.textContent.trimEnd() + match[1];
      tail.nodeValue = tail.nodeValue.slice(match[1].length);
    });
  }


  function addHeadingAnchors() {
    document.querySelectorAll("main h1[id], main h2[id], main h3[id], main h4[id], main h5[id], main h6[id]").forEach((heading) => {
      if (heading.querySelector(".tl-heading-anchor, .permalink-widget")) return;
      const a = document.createElement("a");
      a.className = "tl-heading-anchor";
      a.href = "#" + heading.id;
      a.setAttribute("aria-label", "Link to this section");
      a.textContent = "§";
      heading.appendChild(a);
    });
  }

  // This deferred script runs after parsing but before DOMContentLoaded. Move
  // sentence punctuation into display math before Verso's KaTeX listener runs,
  // so it stays beside the equation instead of becoming a detached text line.
  moveDisplayMathPunctuation();
  installDocstringMath();

  document.addEventListener("DOMContentLoaded", () => {
    mountGuideNav();
    polishGuideHomepage();
    replaceAsciiArrowsInText();
    autoLinkBareUrls();
    externalLinksOpenInNewTabs();
    enhanceRouteLists();
    wrapTables();
    groupLeanExamples();
    simplifyPageNavigation();
    enhanceTerminals();
    addCopyButtons();
    openLeanCodePanels();
    addLeanCodePanelActions();
    addHeadingAnchors();
    mountProgress();
  });
})();
