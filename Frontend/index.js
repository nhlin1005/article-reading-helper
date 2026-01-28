// Frontend/index.js

// ====== config: change these if your HTML uses different IDs ======
const ID_FILE_INPUT = "pdf-input";      // <input type="file">
const ID_TOP_N_INPUT = "ai-top-n";      // <input>, optional
const ID_ANALYZE_BTN = "analyze-btn";   // <button>
const ID_STATUS = "status";             // <div> for messages
const ID_RESULTS = "results";           // <div> for vocab list

document.addEventListener("DOMContentLoaded", () => {
  const fileInput = document.getElementById(ID_FILE_INPUT);
  const topNInput = document.getElementById(ID_TOPN_INPUT_SAFE());
  const analyzeBtn = document.getElementById(ID_ANALYZE_BTN);
  const statusEl = document.getElementById(ID_STATUS);
  const resultsEl = document.getElementById(ID_RESULTS);

  if (!fileInput || !analyzeBtn || !statusEl || !resultsEl) {
    console.error("Some expected DOM elements not found. Check IDs in index.js and index.html.");
    return;
  }

  // In case ai-top-n doesn't exist in HTML, treat as undefined
  function getTopNValue() {
    if (!topNInput) return undefined;
    const raw = topNInput.value.trim();
    if (!raw) return undefined;
    return raw;
  }

  function setStatus(msg, type = "info") {
    // type: "info" | "error" | "success"
    if (!statusEl) return;
    statusEl.textContent = msg;
    statusEl.className = ""; // reset
    statusEl.classList.add("status", `status-${type}`);
  }

  function renderResults(words, wordData) {
    if (!resultsEl) return;

    if (!words || words.length === 0) {
      resultsEl.innerHTML = "<p>No keywords found.</p>";
      return;
    }

    // Simple two-column layout: word list + details
    const listItems = words
      .map(
        (w) =>
          `<li class="word-item" data-word="${encodeURIComponent(
            w
          )}">${w}</li>`
      )
      .join("");

    const firstWord = words[0];
    const first = wordData[firstWord] || { meaning: "", example: "" };

    resultsEl.innerHTML = `
      <div class="results-layout">
        <div class="word-list">
          <h3>Selected Words (${words.length})</h3>
          <ul>${listItems}</ul>
        </div>
        <div class="word-detail" id="word-detail">
          <h3 id="detail-word">${firstWord}</h3>
          <p><strong>Meaning:</strong> <span id="detail-meaning">${first.meaning || "(no meaning found)"}</span></p>
          <p><strong>Example:</strong> <span id="detail-example">${first.example || "(no example sentence found)"}</span></p>
        </div>
      </div>
    `;

    // Click handler: when user clicks a word, update the right panel
    resultsEl.querySelectorAll(".word-item").forEach((li) => {
      li.addEventListener("click", () => {
        const word = decodeURIComponent(li.getAttribute("data-word"));
        const data = wordData[word] || {};
        const wEl = document.getElementById("detail-word");
        const mEl = document.getElementById("detail-meaning");
        const eEl = document.getElementById("detail-example");
        if (wEl) wEl.textContent = word;
        if (mEl) mEl.textContent = data.meaning || "(no meaning found)";
        if (eEl) eEl.textContent = data.example || "(no example sentence found)";
      });
    });
  }

  async function callBackend(file, aiTopNRaw) {
    const formData = new FormData();
    formData.append("pdf", file);
    if (aiTopNRaw !== undefined) {
      formData.append("ai_top_n", aiTopNRaw);
    }

    const resp = await fetch("/api/extract_keywords", {
      method: "POST",
      body: formData,
    });

    if (!resp.ok) {
      const txt = await resp.text();
      throw new Error(`Server error (${resp.status}): ${txt}`);
    }

    return resp.json();
  }

  analyzeBtn.addEventListener("click", async () => {
    const file = fileInput.files && fileInput.files[0];
    if (!file) {
      setStatus("Please choose a PDF file first.", "error");
      return;
    }
    if (file.type !== "application/pdf") {
      setStatus("File must be a PDF.", "error");
      return;
    }

    const topNRaw = getTopNValue();

    setStatus("Uploading PDF and running keyword extraction…", "info");
    analyzeBtn.disabled = true;
    analyzeBtn.textContent = "Processing…";

    try:
      const data = await callBackend(file, topNRaw);
      if (data.error) {
        throw new Error(data.error);
      }
      setStatus(
        `Done! Found ${data.words.length} keywords. (Folder: ${data.readingFolder})`,
        "success"
      );
      renderResults(data.words, data.wordData || {});
    } catch (err) {
      console.error(err);
      setStatus(`Error: ${err.message}`, "error");
      resultsEl.innerHTML = "";
    } finally {
      analyzeBtn.disabled = false;
      analyzeBtn.textContent = "Analyze";
    }
  });
});

// Helper because topNInput might not exist at all
function ID_TOPN_INPUT_SAFE() {
  return ID_TOP_N_INPUT;
}
