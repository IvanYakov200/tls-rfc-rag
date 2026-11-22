const el = id => document.getElementById(id);
const add = (parent, tag, text, cls) => {
  const node = document.createElement(tag);
  if (text !== undefined) node.textContent = text;
  if (cls) node.className = cls;
  parent.append(node);
  return node;
};
let live = false;
let busy = false;
function controls() {
  el("send").disabled = busy || !live || !el("question").value.trim();
  el("question").disabled = busy || !live;
}
function show(row, saved) {
  const box = el(saved ? "example-answer" : "answer");
  box.replaceChildren();
  add(box, "div", saved ? "Сохранённый пример · русская редакция" : "Живой ответ · " + row.total_s + " с", "source");
  add(box, "h2", row.question);
  if (row.status === "insufficient") {
    add(box, "p", "В корпусе недостаточно доказательств для полного ответа.");
  }
  for (const claim of row.claims) {
    const section = add(box, "section", undefined, "claim");
    add(section, "p", claim.text);
    const details = add(section, "details");
    add(details, "summary", "Цитаты из RFC · " + claim.citations.length);
    for (const c of claim.citations) {
      const quote = add(details, "blockquote");
      add(quote, "small", "RFC " + c.rfc + " · § " + c.section + " · байты [" + c.start_byte + ", " + c.end_byte + ")");
      quote.append(document.createTextNode(c.quote));
    }
  }
}
function highlight(active) {
  for (const button of el("examples").children) button.setAttribute("aria-pressed", String(button === active));
}
for (const button of el("sections").children) {
  button.onclick = () => {
    for (const tab of el("sections").children) {
      const active = tab === button;
      el(tab.getAttribute("aria-controls")).hidden = !active;
      tab.setAttribute("aria-pressed", String(active));
    }
  };
}
async function init() {
  const res = await fetch("/api/examples");
  if (!res.ok) throw Error();
  const data = await res.json();
  live = data.live;
  for (const row of data.examples) {
    const button = add(el("examples"), "button", row.title);
    button.onclick = () => { highlight(button); show(row, true); };
  }
  if (!live) el("notice").textContent = "В этом режиме доступны только примеры вопросов.";
  controls();
}
el("question").oninput = controls;
el("form").onsubmit = async event => {
  event.preventDefault();
  if (busy || !live || !el("question").value.trim()) return;
  busy = true;
  controls();
  el("answer").replaceChildren();
  el("notice").textContent = "Ищем фрагменты RFC и готовим ответ…";
  try {
    const res = await fetch("/api/ask", {method:"POST", headers:{"Content-Type":"application/json"},
      body:JSON.stringify({question:el("question").value.trim()})});
    const data = await res.json();
    if (!res.ok) throw Error(data.error);
    show(data, false);
    el("notice").textContent = "";
  } catch (err) {
    el("notice").textContent = err.message || "Не удалось получить ответ.";
  } finally { busy = false; controls(); }
};
init().catch(() => {
  const message = "Не удалось подключиться к локальному чату. Обновите страницу.";
  el("notice").textContent = message;
  el("example-notice").textContent = message;
  controls();
});
