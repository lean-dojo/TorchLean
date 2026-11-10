(function () {
  const key = "torchlean-theme";
  const root = document.documentElement;

  function readStorage(name) {
    try {
      return window.localStorage ? window.localStorage.getItem(name) : null;
    } catch (_err) {
      return null;
    }
  }

  function writeStorage(name, value) {
    try {
      if (window.localStorage) {
        window.localStorage.setItem(name, value);
      }
    } catch (_err) {
      // The toggle still works when localStorage is unavailable.
    }
  }

  function preferredTheme() {
    const stored = readStorage(key);
    if (stored === "light" || stored === "dark") {
      return stored;
    }
    return window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches
      ? "dark"
      : "light";
  }

  function applyTheme(theme) {
    root.setAttribute("data-theme", theme);
    writeStorage("theme", theme);
    const button = document.querySelector("[data-doc-theme-toggle]");
    if (button) {
      const isDark = theme === "dark";
      button.textContent = isDark ? "Light" : "Dark";
      button.setAttribute("aria-label", isDark ? "Use light theme" : "Use dark theme");
      button.setAttribute("title", isDark ? "Use light theme" : "Use dark theme");
    }
    document.querySelectorAll("iframe.navframe").forEach(function (frame) {
      try {
        if (frame.contentDocument) {
          frame.contentDocument.documentElement.setAttribute("data-theme", theme);
        }
      } catch (_err) {
        // The frame is same-origin in local and docs builds. Skip it if a browser blocks access.
      }
    });
  }

  applyTheme(preferredTheme());

  window.addEventListener("DOMContentLoaded", function () {
    applyTheme(preferredTheme());
    document.querySelectorAll("iframe.navframe").forEach(function (frame) {
      frame.addEventListener("load", function () {
        applyTheme(root.getAttribute("data-theme") || preferredTheme());
      });
    });
    const button = document.querySelector("[data-doc-theme-toggle]");
    if (!button) {
      return;
    }
    button.addEventListener("click", function () {
      const next = root.getAttribute("data-theme") === "dark" ? "light" : "dark";
      writeStorage(key, next);
      applyTheme(next);
    });
  });
})();
