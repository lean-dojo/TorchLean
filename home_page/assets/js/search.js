(function () {
  var root = document.documentElement;
  var themeButton = document.querySelector("[data-theme-toggle]");
  var themeColor = document.querySelector('meta[name="theme-color"]');

  function storedTheme() {
    try {
      return window.localStorage.getItem("torchlean-theme");
    } catch (error) {
      return null;
    }
  }

  function saveTheme(theme) {
    try {
      window.localStorage.setItem("torchlean-theme", theme);
    } catch (error) {
      // The toggle should still work when localStorage is unavailable.
    }
  }

  function setTheme(theme) {
    root.setAttribute("data-theme", theme);
    if (themeColor) {
      themeColor.setAttribute("content", theme === "dark" ? "#101418" : "#ffffff");
    }
    if (themeButton) {
      themeButton.setAttribute("aria-label", theme === "dark" ? "Use light theme" : "Use dark theme");
    }
  }

  var initialTheme = storedTheme();
  if (initialTheme === "dark" || initialTheme === "light") {
    setTheme(initialTheme);
  } else {
    setTheme("light");
  }

  if (themeButton) {
    themeButton.addEventListener("click", function () {
      var nextTheme = root.getAttribute("data-theme") === "dark" ? "light" : "dark";
      setTheme(nextTheme);
      saveTheme(nextTheme);
    });
  }

  var openButton = document.querySelector("[data-search-open]");
  var overlay = document.querySelector("[data-search-overlay]");
  var closeButton = document.querySelector("[data-search-close]");
  var input = document.querySelector("[data-search-input]");
  var results = document.querySelector("[data-search-results]");

  if (!openButton || !overlay || !closeButton || !input || !results) {
    return;
  }

  var index = [];
  var loaded = false;

  function siteRoot() {
    var scripts = document.getElementsByTagName("script");
    for (var i = scripts.length - 1; i >= 0; i -= 1) {
      var src = scripts[i].getAttribute("src") || "";
      var marker = "/assets/js/search.js";
      var at = src.indexOf(marker);
      if (at >= 0) {
        return src.slice(0, at);
      }
    }
    return "";
  }

  function loadIndex() {
    if (loaded) {
      return Promise.resolve(index);
    }
    return fetch(siteRoot() + "/search-index.json")
      .then(function (response) {
        if (!response.ok) {
          throw new Error("search index unavailable");
        }
        return response.json();
      })
      .then(function (items) {
        index = items;
        loaded = true;
        return index;
      })
      .catch(function () {
        results.innerHTML = '<p class="search-empty">Search is unavailable in this build.</p>';
        return [];
      });
  }

  function normalize(value) {
    return String(value || "").toLowerCase();
  }

  function score(item, terms) {
    var title = normalize(item.title);
    var section = normalize(item.section);
    var text = normalize(item.text);
    var total = 0;

    terms.forEach(function (term) {
      if (title.indexOf(term) >= 0) {
        total += 8;
      }
      if (section.indexOf(term) >= 0) {
        total += 4;
      }
      if (text.indexOf(term) >= 0) {
        total += 2;
      }
    });

    return total;
  }

  function render(items, query) {
    if (!query.trim()) {
      results.innerHTML = '<p class="search-empty">Type to search the guide, examples, API docs, and CUDA notes.</p>';
      return;
    }

    var terms = normalize(query).split(/\s+/).filter(Boolean);
    var ranked = items
      .map(function (item) {
        return { item: item, score: score(item, terms) };
      })
      .filter(function (entry) {
        return entry.score > 0;
      })
      .sort(function (a, b) {
        return b.score - a.score || a.item.title.localeCompare(b.item.title);
      })
      .slice(0, 8);

    if (ranked.length === 0) {
      results.innerHTML = '<p class="search-empty">No results for "' + query.replace(/</g, "&lt;") + '".</p>';
      return;
    }

    results.innerHTML = ranked
      .map(function (entry) {
        var item = entry.item;
        return (
          '<a class="search-result" href="' + item.url + '">' +
          '<span>' + item.section + '</span>' +
          '<strong>' + item.title + '</strong>' +
          '<em>' + item.text + '</em>' +
          '</a>'
        );
      })
      .join("");
  }

  function openSearch() {
    overlay.hidden = false;
    document.body.classList.add("search-active");
    loadIndex().then(function (items) {
      render(items, input.value);
      input.focus();
      input.select();
    });
  }

  function closeSearch() {
    overlay.hidden = true;
    document.body.classList.remove("search-active");
    openButton.focus();
  }

  openButton.addEventListener("click", openSearch);
  closeButton.addEventListener("click", closeSearch);
  overlay.addEventListener("click", function (event) {
    if (event.target === overlay) {
      closeSearch();
    }
  });
  input.addEventListener("input", function () {
    render(index, input.value);
  });
  document.addEventListener("keydown", function (event) {
    if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "k") {
      event.preventDefault();
      openSearch();
    }
    if (event.key === "Escape" && !overlay.hidden) {
      closeSearch();
    }
  });
})();
