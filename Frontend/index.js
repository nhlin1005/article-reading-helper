// Frontend/index.js

let currentKeywords = [];       // ordered list of words from backend
let wordData = {};              // { word: { meaning, example } }
let studyDeck = [];             // [{ word, meaning, example }]

// ---------- DOM helpers ----------
function $(id) {
  return document.getElementById(id);
}

function setStatus(msg) {
  const el = $("statusText");
  if (el) el.textContent = msg;
}

let processingTimer = null;
let currentStep = 0;
const TOTAL_STEPS = 40;  // just an approximate total “steps”

function startProcessingCounter() {
  const el = $("statusText");
  currentStep = 0;
  if (!el) return;

  clearInterval(processingTimer);
  el.textContent = `Processing... (0/${TOTAL_STEPS})`;

  processingTimer = setInterval(() => {
    if (currentStep < TOTAL_STEPS) {
      currentStep += 1;
    }
    el.textContent = `Processing... (${currentStep}/${TOTAL_STEPS})`;
  }, 1000); // update every second
}

function stopProcessingCounter(finalMsg) {
  clearInterval(processingTimer);
  processingTimer = null;
  if (finalMsg) {
    const el = $("statusText");
    if (el) el.textContent = finalMsg;
  }
}


// ---------- Rendering ----------
function renderKeywords() {
  const listEl = $("keywordList");
  if (!listEl) return;

  listEl.innerHTML = "";
  if (!currentKeywords.length) {
    listEl.innerHTML = "<p>No keywords yet.</p>";
    return;
  }

  for (const w of currentKeywords) {
    const li = document.createElement("li");
    li.className = "keyword-item";

    const textSpan = document.createElement("span");
    textSpan.textContent = w;

    const addBtn = document.createElement("button");
    addBtn.textContent = "★";
    addBtn.title = "Add to Study Deck";
    addBtn.onclick = () => addKeywordToDeck(w);

    li.appendChild(textSpan);
    li.appendChild(addBtn);
    listEl.appendChild(li);
  }
}

function renderDeck() {
  const wordEl = $("deckWord");
  const defEl = $("deckDefinition");
  const exEl = $("deckExample");
  const counterEl = $("deckCounter");

  if (!wordEl || !defEl || !exEl || !counterEl) return;

  if (!studyDeck.length) {
    wordEl.textContent = "(no word)";
    defEl.textContent = "No definition available.";
    exEl.textContent = "No example available.";
    counterEl.textContent = "0 / 0";
    return;
  }

  const idx = parseInt(wordEl.dataset.index || "0", 10) || 0;
  const card = studyDeck[Math.min(idx, studyDeck.length - 1)];
  wordEl.dataset.index = String(idx);

  wordEl.textContent = card.word;
  defEl.textContent = card.meaning || "No definition available.";
  exEl.textContent = card.example || "No example available.";
  counterEl.textContent = `${idx + 1} / ${studyDeck.length}`;
}

function showNextCard() {
  const wordEl = $("deckWord");
  if (!wordEl || !studyDeck.length) return;
  let idx = parseInt(wordEl.dataset.index || "0", 10) || 0;
  idx = (idx + 1) % studyDeck.length;
  wordEl.dataset.index = String(idx);
  renderDeck();
}

function showPrevCard() {
  const wordEl = $("deckWord");
  if (!wordEl || !studyDeck.length) return;
  let idx = parseInt(wordEl.dataset.index || "0", 10) || 0;
  idx = (idx - 1 + studyDeck.length) % studyDeck.length;
  wordEl.dataset.index = String(idx);
  renderDeck();
}

// ---------- API helpers ----------
async function uploadPdfAndExtractKeywords(pdfFile, aiTopN) {
  const form = new FormData();
  form.append("pdf", pdfFile);
  form.append("ai_top_n", String(aiTopN));

  const resp = await fetch("/api/extract_keywords", {
    method: "POST",
    body: form,
  });

  if (!resp.ok) {
    const text = await resp.text();
    throw new Error(`Server error: ${resp.status} ${text}`);
  }

  return resp.json();
}

async function apiLookupWord(word) {
  const resp = await fetch("/api/lookup_word", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ word }),
  });

  if (!resp.ok) {
    console.error("lookup failed", resp.status);
    return {
      word,
      meaning: "",
      example: "",
    };
  }

  try {
    const data = await resp.json();
    return {
      word: data.word || word,
      meaning: data.meaning || "",
      example: data.example || "",
    };
  } catch (e) {
    console.error("failed to parse lookup response", e);
    return {
      word,
      meaning: "",
      example: "",
    };
  }
}

// ---------- Logic ----------
async function addKeywordToDeck(word) {
  // 1) try the existing data from /api/extract_keywords
  let meaning = (wordData[word] && wordData[word].meaning) || "";
  let example = (wordData[word] && wordData[word].example) || "";

  // 2) if still missing, ask backend to look it up
  if (!meaning || !example) {
    try {
      const lookedUp = await apiLookupWord(word);
      if (lookedUp.meaning) meaning = lookedUp.meaning;
      if (lookedUp.example) example = lookedUp.example;
      // cache into wordData so next time is instant
      wordData[word] = {
        meaning,
        example,
      };
    } catch (e) {
      console.error("lookup error", e);
    }
  }

  // 3) add to deck (avoid duplicates)
  if (!studyDeck.some((c) => c.word === word)) {
    studyDeck.push({
      word,
      meaning,
      example,
    });
  }

  // keep deck in a stable order (alphabetical by word)
  studyDeck.sort((a, b) => a.word.localeCompare(b.word));

  $("deckWord").dataset.index = "0";
  renderDeck();
}

// Manual add from Study Deck “+” button
async function handleManualAdd() {
  const wInput = $("manualWord");
  const dInput = $("manualDef");
  const eInput = $("manualEx");
  if (!wInput) return;

  const word = wInput.value.trim();
  if (!word) return;

  let meaning = dInput ? dInput.value.trim() : "";
  let example = eInput ? eInput.value.trim() : "";

  // if user didn’t type meaning/example, ask backend to fill in
  if (!meaning || !example) {
    try {
      const lookedUp = await apiLookupWord(word);
      if (!meaning && lookedUp.meaning) meaning = lookedUp.meaning;
      if (!example && lookedUp.example) example = lookedUp.example;
    } catch (e) {
      console.error("manual lookup error", e);
    }
  }

  if (!studyDeck.some((c) => c.word === word)) {
    studyDeck.push({
      word,
      meaning,
      example,
    });
    studyDeck.sort((a, b) => a.word.localeCompare(b.word));
  }

  wInput.value = "";
  if (dInput) dInput.value = "";
  if (eInput) eInput.value = "";

  $("deckWord").dataset.index = "0";
  renderDeck();
}

// main upload handler
async function handleUploadClick() {
  const fileInput = $("pdfInput");
  const topNInput = $("aiTopNInput");
  if (!fileInput || !fileInput.files.length) {
    alert("Please choose a PDF first.");
    return;
  }

  const pdfFile = fileInput.files[0];
  const aiTopN = parseFloat((topNInput && topNInput.value) || "0.1") || 0.1;

  try {
    startProcessingCounter();
    const data = await uploadPdfAndExtractKeywords(pdfFile, aiTopN);

    // words: [...], wordData: {word: {meaning, example}}
    currentKeywords = Array.isArray(data.words) ? data.words.slice() : [];
    // make output words order stable / alphabetical
    currentKeywords.sort((a, b) => a.localeCompare(b));

    wordData = data.wordData || {};
    setStatus(`Got ${currentKeywords.length} keywords.`);
    renderKeywords();

    // clear previous deck when a new article is uploaded
    studyDeck = [];
    renderDeck();
  } catch (e) {
    console.error(e);
    setStatus("Error: " + e.message);
  } finally {
    stopProcessingCounter();
  }
}

// ---------- Wire up events ----------
window.addEventListener("DOMContentLoaded", () => {
  const uploadBtn = $("uploadBtn");
  const nextBtn = $("deckNext");
  const prevBtn = $("deckPrev");
  const addManualBtn = $("manualAddBtn");

  if (uploadBtn) uploadBtn.onclick = handleUploadClick;
  if (nextBtn) nextBtn.onclick = showNextCard;
  if (prevBtn) prevBtn.onclick = showPrevCard;
  if (addManualBtn) addManualBtn.onclick = handleManualAdd;

  renderKeywords();
  renderDeck();
});
