(() => {
  const year = document.querySelector("[data-year]");
  if (year) year.textContent = String(new Date().getFullYear());

  document.querySelectorAll("[data-copy]").forEach((btn) => {
    btn.addEventListener("click", async () => {
      const id = btn.getAttribute("data-copy");
      const block = id ? document.getElementById(id) : null;
      const text = block?.textContent ?? "";
      try {
        await navigator.clipboard.writeText(text);
        btn.classList.add("copied");
        btn.textContent = "Copied";
        setTimeout(() => {
          btn.classList.remove("copied");
          btn.textContent = "Copy";
        }, 1400);
      } catch {
        btn.textContent = "Failed";
      }
    });
  });

  document.querySelectorAll("[data-tabs]").forEach((root) => {
    const buttons = root.querySelectorAll(".tab");
    const panels = root.querySelectorAll(".tab-panel");
    buttons.forEach((btn) => {
      btn.addEventListener("click", () => {
        const target = btn.getAttribute("data-tab");
        buttons.forEach((b) => b.setAttribute("aria-selected", String(b === btn)));
        panels.forEach((p) => {
          p.hidden = p.getAttribute("data-panel") !== target;
        });
      });
    });
  });

  const modal = document.getElementById("article-modal");
  const content = document.getElementById("article-content");
  let articleHtml = null;
  let lastFocus = null;

  const openers = document.querySelectorAll("[data-open-article]");
  const closers = document.querySelectorAll("[data-close-article]");

  async function loadArticle() {
    if (articleHtml) {
      content.innerHTML = articleHtml;
      return;
    }
    content.innerHTML = '<p class="article-loading">Loading article…</p>';
    try {
      const res = await fetch("./paper_unabridged_full.md");
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const md = await res.text();
      if (typeof marked === "undefined") {
        throw new Error("Markdown renderer unavailable");
      }
      marked.setOptions({ gfm: true, breaks: false });
      articleHtml = marked.parse(md);
      content.innerHTML = articleHtml;
    } catch (err) {
      content.innerHTML = `<p class="article-error">Could not load the article. <a href="./paper_unabridged_full.md" target="_blank" rel="noopener">Open the Markdown file</a> instead.<br><small>${String(err.message || err)}</small></p>`;
    }
  }

  function openModal() {
    if (!modal) return;
    lastFocus = document.activeElement;
    modal.hidden = false;
    document.body.classList.add("modal-open");
    loadArticle();
    const closeBtn = modal.querySelector("[data-close-article].btn");
    (closeBtn || modal).focus?.();
  }

  function closeModal() {
    if (!modal || modal.hidden) return;
    modal.hidden = true;
    document.body.classList.remove("modal-open");
    if (lastFocus && typeof lastFocus.focus === "function") lastFocus.focus();
  }

  openers.forEach((el) => el.addEventListener("click", openModal));
  closers.forEach((el) => el.addEventListener("click", closeModal));

  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") closeModal();
  });
})();
