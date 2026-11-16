const form = document.querySelector("#composer");
const message = document.querySelector("#message");
const model = document.querySelector("#model");
const send = document.querySelector("#send");
const count = document.querySelector("#count");
const error = document.querySelector("#error");
const chat = document.querySelector("#chat-log");
const timings = document.querySelector("#timings");
const emptyChat = document.querySelector("#empty-chat");
const emptyTiming = document.querySelector("#empty-timing");
const clear = document.querySelector("#clear");
const shortNames = {int8: "BERT INT8", bert: "BERT FP32", mlp: "Word2vec MLP", tfidf: "TF-IDF"};
let sequence = 0;

function append(parent, tag, value, className) {
  const node = document.createElement(tag);
  node.textContent = value;
  if (className) node.className = className;
  parent.appendChild(node);
  return node;
}

function appendResult(data) {
  sequence += 1;
  emptyChat.hidden = emptyTiming.hidden = true;
  const card = append(chat, "article", "", "chat-message");
  const meta = append(card, "div", "", "message-meta");
  append(meta, "span", String(sequence).padStart(2, "0"), "message-number");
  append(meta, "strong", data.label === "clean" ? "Clean" : "Offensive", data.label);
  append(meta, "span", `${(100 * data.confidence).toFixed(1)}% confidence`, "muted");
  append(card, "p", data.display_text, data.label === "offensive" ? "masked" : "");
  append(card, "span", `${data.model_name} · Lexicon ${data.rule_flag ? "flagged" : "clear"}`, "model-detail");
  chat.scrollTop = chat.scrollHeight;

  const row = document.createElement("tr");
  append(row, "td", String(sequence).padStart(2, "0"));
  append(row, "td", shortNames[data.model]);
  append(row, "td", `${data.latency_ms.toFixed(2)} ms`);
  timings.prepend(row);
}

message.addEventListener("input", () => {
  count.textContent = `${message.value.length} / 600`;
  error.textContent = "";
});
message.addEventListener("keydown", event => {
  if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
    event.preventDefault();
    if (!send.disabled) form.requestSubmit();
  }
});
clear.addEventListener("click", () => {
  chat.querySelectorAll(".chat-message").forEach(node => node.remove());
  timings.replaceChildren();
  emptyChat.hidden = emptyTiming.hidden = false;
  sequence = 0;
});
form.addEventListener("submit", async event => {
  event.preventDefault();
  const text = message.value.trim();
  if (!text) {
    error.textContent = "Enter a message first.";
    return;
  }
  send.disabled = model.disabled = clear.disabled = message.disabled = true;
  send.textContent = "Checking…";
  error.textContent = "";
  try {
    const request = {text, model: model.value};
    let data;
    if (globalThis.localModerate) {
      data = await globalThis.localModerate(request);
    } else {
      const response = await fetch("/api/moderate", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify(request),
      });
      data = await response.json();
      if (!response.ok) throw new Error(typeof data.detail === "string" ? data.detail : "Invalid message.");
    }
    appendResult(data);
    message.value = "";
    count.textContent = "0 / 600";
  } catch (failure) {
    error.textContent = failure.message || "Moderation failed. Try again.";
  } finally {
    send.disabled = model.disabled = clear.disabled = message.disabled = false;
    send.textContent = "Send";
    message.focus();
  }
});
