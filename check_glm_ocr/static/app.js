const dropzone = document.getElementById("upload-area");
const fileInput = document.getElementById("file-input");
const statusArea = document.getElementById("status-area");
const progressBar = document.getElementById("progress-bar");
const statusText = document.getElementById("status-text");
const downloadBtn = document.getElementById("download-btn");
const resetBtn = document.getElementById("reset-btn");
const previewArea = document.getElementById("preview-area");
const sourceEl = document.getElementById("markdown-source");
const previewEl = document.getElementById("markdown-preview");

let currentFilename = "document";

marked.setOptions({ gfm: true, breaks: false });

function renderPreview() {
  previewEl.innerHTML = marked.parse(sourceEl.value || "");
  renderMathInElement(previewEl, {
    delimiters: [
      { left: "$$", right: "$$", display: true },
      { left: "$", right: "$", display: false },
      { left: "\\(", right: "\\)", display: false },
      { left: "\\[", right: "\\]", display: true },
    ],
    throwOnError: false,
  });
}

sourceEl.addEventListener("input", renderPreview);

function resetUI() {
  fileInput.value = "";
  statusArea.classList.add("hidden");
  previewArea.classList.add("hidden");
  downloadBtn.classList.add("hidden");
  resetBtn.classList.add("hidden");
  sourceEl.value = "";
  previewEl.innerHTML = "";
  progressBar.value = 0;
  progressBar.max = 1;
}

dropzone.addEventListener("click", () => fileInput.click());

dropzone.addEventListener("dragover", (e) => {
  e.preventDefault();
  dropzone.classList.add("dragover");
});

dropzone.addEventListener("dragleave", () => dropzone.classList.remove("dragover"));

dropzone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropzone.classList.remove("dragover");
  if (e.dataTransfer.files.length > 0) {
    startConversion(e.dataTransfer.files[0]);
  }
});

fileInput.addEventListener("change", () => {
  if (fileInput.files.length > 0) {
    startConversion(fileInput.files[0]);
  }
});

resetBtn.addEventListener("click", resetUI);

downloadBtn.addEventListener("click", () => {
  const blob = new Blob([sourceEl.value], { type: "text/markdown" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  const base = currentFilename.replace(/\.[^/.]+$/, "");
  a.href = url;
  a.download = `${base}.md`;
  a.click();
  URL.revokeObjectURL(url);
});

async function startConversion(file) {
  currentFilename = file.name;
  resetUI();
  statusArea.classList.remove("hidden");
  previewArea.classList.remove("hidden");
  statusText.textContent = "Uploading...";

  const formData = new FormData();
  formData.append("file", file);

  let response;
  try {
    response = await fetch("/api/convert", { method: "POST", body: formData });
  } catch (err) {
    statusText.textContent = `Error: ${err.message}`;
    return;
  }

  if (!response.ok || !response.body) {
    statusText.textContent = `Error: server returned ${response.status}`;
    return;
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    let boundary;
    while ((boundary = buffer.indexOf("\n\n")) !== -1) {
      const rawEvent = buffer.slice(0, boundary);
      buffer = buffer.slice(boundary + 2);
      handleSSEEvent(rawEvent);
    }
  }
}

function handleSSEEvent(rawEvent) {
  let eventType = "message";
  let dataLine = "";
  for (const line of rawEvent.split("\n")) {
    if (line.startsWith("event:")) {
      eventType = line.slice(6).trim();
    } else if (line.startsWith("data:")) {
      dataLine += line.slice(5).trim();
    }
  }
  if (!dataLine) return;

  let payload;
  try {
    payload = JSON.parse(dataLine);
  } catch {
    return;
  }

  if (eventType === "start") {
    progressBar.max = payload.total;
    progressBar.value = 0;
    statusText.textContent = `Processing page 0 / ${payload.total}...`;
  } else if (eventType === "page") {
    progressBar.max = payload.total;
    progressBar.value = payload.page;
    statusText.textContent = `Processing page ${payload.page} / ${payload.total}...`;
    if (payload.total > 1) {
      sourceEl.value += `${sourceEl.value ? "\n\n" : ""}<!-- page ${payload.page} -->\n\n${payload.markdown}`;
    } else {
      sourceEl.value = payload.markdown;
    }
    renderPreview();
  } else if (eventType === "done") {
    sourceEl.value = payload.markdown;
    renderPreview();
    statusText.textContent = "Done.";
    downloadBtn.classList.remove("hidden");
    resetBtn.classList.remove("hidden");
  } else if (eventType === "error") {
    statusText.textContent = `Error: ${payload.message}`;
    resetBtn.classList.remove("hidden");
  }
}
