(function () {
  var appearance = "system";
  try {
    var stored = window.localStorage.getItem("vostavo.appearance.v1");
    if (stored === "light" || stored === "dark" || stored === "system") appearance = stored;
  } catch (_) {
    // First paint follows the operating system when storage is blocked.
  }
  var dark = appearance === "dark" || appearance === "system" &&
    typeof window.matchMedia === "function" && window.matchMedia("(prefers-color-scheme: dark)").matches;
  var theme = dark ? "dark" : "light";
  document.documentElement.dataset.theme = theme;
  document.documentElement.style.colorScheme = theme;
})();
