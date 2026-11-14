/* SeedDrift web page: talks to the SeedDrift API (services/api). No build step. */
"use strict";

const CATS = [
  ["mammal", "疑似哺乳动物来源", "var(--c-mammal)"],
  ["prokaryote", "疑似原核来源", "var(--c-prokaryote)"],
  ["eukaryote", "疑似其他真核来源", "var(--c-eukaryote)"],
  ["virus", "疑似病毒 / 噬菌体来源", "var(--c-virus)"],
  ["native", "本谱系", "var(--c-native)"],
  ["orphan", "外部无同源", "var(--c-orphan)"],
  ["repeat", "仅重复结构命中", "var(--c-repeat)"],
];
const CAT = Object.fromEntries(CATS.map(([k, l, c]) => [k, { label: l, color: c }]));
const FOREIGN = new Set(["mammal", "prokaryote", "eukaryote", "virus"]);
const $ = (s, el = document) => el.querySelector(s);
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

const state = { api: "", current: null, timer: null, genes: [], filter: new Set(), sort: ["AI", -1], query: "" };

function defaultApi() {
  const saved = (() => { try { return localStorage.getItem("seeddrift.api"); } catch (e) { return null; } })();
  if (saved) return saved;
  if (location.protocol.startsWith("http") && location.port === "8600") return location.origin;
  return "http://localhost:8600";
}

async function api(path, opts) {
  const r = await fetch(state.api.replace(/\/$/, "") + path, opts);
  if (!r.ok) {
    let msg = r.status + " " + r.statusText;
    try { const j = await r.json(); msg = j.detail || msg; } catch (e) { /* not JSON */ }
    throw new Error(msg);
  }
  return r.json();
}

/* -- connection ------------------------------------------------------------------------------ */
async function checkHealth() {
  const dot = $("#health");
  try {
    const h = await api("/api/health");
    const tools = h.tools || {}, fold = h.esmfold || {};
    const toolsOk = tools.mode === "in-process" || (tools.diamond && tools.reference);
    const foldOk = fold.mode === "in-process" || fold.ready;
    dot.className = "dot " + (toolsOk && foldOk ? "ok" : toolsOk ? "warn" : "bad");
    dot.title = [
      "API " + h.api,
      "检索服务：" + (tools.error ? "不可用（" + tools.error + "）" : toolsOk ? "正常" : "缺少数据或工具"),
      "ESMFold：" + (fold.error ? "不可用" : fold.ready || fold.mode ? "就绪" : fold.loading ? "模型加载中" : "未就绪"),
    ].join("\n");
  } catch (e) {
    dot.className = "dot bad";
    dot.title = "连不上 " + state.api + "：" + e.message;
  }
}

/* -- form ------------------------------------------------------------------------------------ */
let tab = "file";
function setupForm() {
  document.querySelectorAll(".tabs button").forEach((b) => b.addEventListener("click", (ev) => {
    ev.preventDefault();
    tab = b.dataset.tab;
    document.querySelectorAll(".tabs button").forEach((x) => x.classList.toggle("on", x === b));
    document.querySelectorAll(".pane").forEach((p) => p.classList.toggle("hidden", p.dataset.pane !== tab));
  }));
  $("#demo").addEventListener("change", showDemoNote);
  $("#form").addEventListener("submit", async (ev) => {
    ev.preventDefault();
    const fd = new FormData();
    $("#form-err").textContent = "";
    if (tab === "file") {
      const f = $("#file").files[0];
      if (!f) { $("#form-err").textContent = "请选择一个文件"; return; }
      fd.append("file", f);
    } else if (tab === "url") {
      if (!$("#url").value.trim()) { $("#form-err").textContent = "请填写 URL"; return; }
      fd.append("url", $("#url").value.trim());
    } else {
      fd.append("demo", $("#demo").value);
    }
    fd.append("name", $("#name").value.trim());
    fd.append("self_taxon", $("#self").value.trim());
    fd.append("structure", $("#structure").checked ? "true" : "false");
    $("#submit").disabled = true;
    try {
      const r = await api("/api/jobs", { method: "POST", body: fd });
      await loadJobs();
      openJob(r.id);
    } catch (e) {
      $("#form-err").textContent = "提交失败：" + e.message;
    } finally {
      $("#submit").disabled = false;
    }
  });
}

let demos = [];
async function loadDemos() {
  try { demos = await api("/api/demos"); } catch (e) { demos = []; }
  $("#demo").innerHTML = demos.map((d) => `<option value="${esc(d.id)}">${esc(d.name)}</option>`).join("");
  showDemoNote();
}
function showDemoNote() {
  const d = demos.find((x) => x.id === $("#demo").value);
  $("#demo-note").textContent = d ? d.note || "" : "没有可用的示例";
}

/* -- jobs ------------------------------------------------------------------------------------ */
const STATUS = { queued: "排队中", running: "运行中", done: "完成", failed: "失败" };
async function loadJobs() {
  let jobs = [];
  try { jobs = await api("/api/jobs"); } catch (e) { $("#jobs").innerHTML = `<li class="muted">${esc(e.message)}</li>`; return; }
  $("#jobs").innerHTML = jobs.map((j) => `
    <li data-id="${esc(j.id)}" class="${state.current === j.id ? "on" : ""}">
      <div class="jn">${esc(j.name)}</div>
      <div class="js"><span class="st-${esc(j.status)}">${STATUS[j.status] || esc(j.status)}</span>
        · ${esc(j.created || "")}${j.status === "running" ? " · " + Math.round((j.progress || 0) * 100) + "%" : ""}</div>
    </li>`).join("") || `<li class="muted">还没有分析</li>`;
  document.querySelectorAll("#jobs li[data-id]").forEach((li) => li.addEventListener("click", () => openJob(li.dataset.id)));
}

async function openJob(id) {
  state.current = id;
  clearTimeout(state.timer);
  document.querySelectorAll("#jobs li").forEach((li) => li.classList.toggle("on", li.dataset.id === id));
  const node = $("#tpl-result").content.cloneNode(true);
  $("#content").replaceChildren(node);
  state.genes = []; state.filter = new Set(); state.query = "";
  await refreshJob(id);
}

async function refreshJob(id) {
  if (state.current !== id) return;
  let j;
  try { j = await api("/api/jobs/" + id); } catch (e) { $(".rerr").textContent = e.message; return; }
  $(".rname").textContent = j.name || id;
  $(".rmeta").textContent = `任务 ${id} · 创建于 ${j.created || ""}` + (j.params && j.params.self_taxon ? ` · 自身类群：${j.params.self_taxon}` : "");
  const prog = $(".progress");
  if (j.status === "queued" || j.status === "running") {
    prog.classList.remove("hidden");
    $(".bar span").style.width = Math.round((j.progress || 0) * 100) + "%";
    $(".pmsg").textContent = (j.message || STATUS[j.status]) + (j.status === "running" ? " …" : "");
    state.timer = setTimeout(() => { refreshJob(id); loadJobs(); }, 3000);
    return;
  }
  prog.classList.add("hidden");
  if (j.status === "failed") { $(".rerr").textContent = "分析失败：" + (j.error || "未知错误"); return; }
  const base = state.api.replace(/\/$/, "") + "/api/jobs/" + id + "/files/";
  $(".downloads").innerHTML = ["genes.tsv", "summary.json", "proteins.faa", "hits.tsv"]
    .map((f) => `<a href="${base + f}" download>${f}</a>`).join("");
  state.genes = await api("/api/jobs/" + id + "/genes");
  renderResult(j.summary || {});
}

/* -- result ---------------------------------------------------------------------------------- */
function renderResult(s) {
  $(".body").classList.remove("hidden");
  const cats = s.categories || {}, strong = s.strong || {};
  $(".cards").innerHTML = CATS.map(([k, l, c]) => `
    <div class="stat" style="--c:${c}" data-cat="${k}">
      <div class="n">${(cats[k] || 0).toLocaleString()}</div><div class="l">${l}</div>
      <div class="s">${FOREIGN.has(k) ? "强证据（AI ≥ 45）" + (strong[k] || 0) : "&nbsp;"}</div>
    </div>`).join("");
  document.querySelectorAll(".stat").forEach((el) => el.addEventListener("click", () => toggleFilter(el.dataset.cat, true)));
  const notes = [];
  notes.push(`共 ${(s.n_genes || 0).toLocaleString()} 个基因，用时 ${fmtTime(s.seconds)}。`);
  if (s.self_name) notes.push(`自身类群：${s.self_name}（taxid ${s.self_taxid}），比对时排除；本谱系按 ${s.home_group} 计算 Alien Index。`);
  if (s.note) notes.push(s.note + "。");
  if (s.single_species) notes.push(`其中 ${s.single_species} 个真核来源候选只有一个物种支持，可能是该物种基因组被细菌污染，需谨慎。`);
  $(".note").textContent = notes.join(" ");
  const groups = Object.entries(s.donor_groups || {});
  const gmax = Math.max(1, ...groups.map(([, n]) => n));
  $(".groups").innerHTML = groups.length ? groups.map(([g, n]) => `
    <div class="hbar"><span>${esc(g)}</span><div class="t"><span style="width:${(100 * n / gmax).toFixed(1)}%"></span></div><span class="k">${n}</span></div>`).join("")
    : `<p class="muted">没有外来候选</p>`;
  const st = Object.entries(s.structural || {}).sort((a, b) => b[1] - a[1]);
  const smax = Math.max(1, ...st.map(([, n]) => n));
  $(".structs").innerHTML = !s.structure ? `<p class="muted">这次没有做结构检索</p>` : st.length ? st.map(([g, n]) => `
    <div class="hbar"><span>${esc(structLabel(g))}</span><div class="t"><span style="width:${(100 * n / smax).toFixed(1)}%;background:var(--sky)"></span></div><span class="k">${n}</span></div>`).join("")
    : `<p class="muted">没有需要做结构检索的孤儿基因</p>`;
  renderMap();
  renderFilters();
  renderTable();
  $(".search").addEventListener("input", (ev) => { state.query = ev.target.value.trim().toLowerCase(); renderTable(); });
}

function structLabel(g) {
  return { "no structural match": "无同折叠匹配", "repeat": "仅重复结构", "skipped": "过长，未预测" }[g] || g;
}
function fmtTime(sec) {
  if (!sec && sec !== 0) return "—";
  return sec < 90 ? sec + " 秒" : Math.round(sec / 60) + " 分钟";
}

function renderFilters() {
  $(".filters").innerHTML = CATS.map(([k, l, c]) =>
    `<button class="chip ${state.filter.has(k) ? "on" : ""}" style="--c:${c}" data-cat="${k}">${l}</button>`).join("");
  document.querySelectorAll(".chip").forEach((b) => b.addEventListener("click", () => toggleFilter(b.dataset.cat)));
}
function toggleFilter(k, only) {
  if (only) state.filter = new Set(state.filter.size === 1 && state.filter.has(k) ? [] : [k]);
  else state.filter.has(k) ? state.filter.delete(k) : state.filter.add(k);
  renderFilters();
  renderTable();
  if (only) $(".genes").scrollIntoView({ behavior: "smooth", block: "start" });
}

const COLS = [
  ["gene_id", "基因"], ["category", "判定"], ["AI", "AI"], ["donor_group", "供体类群"], ["support", "支持物种"],
  ["best_taxon", "最像的外部物种"], ["best_pident", "一致度 %"], ["best_home_taxon", "本谱系最近"],
  ["length", "长度"], ["struct_group", "结构类群"],
];
const NUM = new Set(["AI", "support", "best_pident", "length"]);

function renderTable() {
  const tb = $(".genes tbody"), th = $(".genes thead");
  th.innerHTML = "<tr>" + COLS.map(([k, l]) => `<th data-k="${k}">${l}${state.sort[0] === k ? (state.sort[1] > 0 ? " ▲" : " ▼") : ""}</th>`).join("") + "</tr>";
  th.querySelectorAll("th").forEach((h) => h.addEventListener("click", () => {
    const k = h.dataset.k;
    state.sort = [k, state.sort[0] === k ? -state.sort[1] : NUM.has(k) ? -1 : 1];
    renderTable();
  }));
  let rows = state.genes.filter((r) => !state.filter.size || state.filter.has(r.category));
  if (state.query) rows = rows.filter((r) => [r.gene_id, r.best_taxon, r.donor_group, r.struct_name, r.mammal_hit]
    .some((x) => String(x || "").toLowerCase().includes(state.query)));
  const [k, d] = state.sort;
  rows.sort((a, b) => {
    const x = NUM.has(k) ? parseFloat(a[k]) : a[k], y = NUM.has(k) ? parseFloat(b[k]) : b[k];
    const xa = x === "" || Number.isNaN(x), ya = y === "" || Number.isNaN(y);
    if (xa !== ya) return xa ? 1 : -1;
    return x < y ? -d : x > y ? d : 0;
  });
  const LIMIT = 1500;
  tb.innerHTML = rows.slice(0, LIMIT).map((r, i) => `
    <tr class="row" data-i="${i}">
      <td>${esc(r.gene_id)}</td>
      <td><span class="tagc" style="--c:${(CAT[r.category] || {}).color}">${esc((CAT[r.category] || {}).label || r.category)}</span>
        ${r.strength === "strong" ? "强" : ""}</td>
      <td>${esc(r.AI)}</td><td>${esc(r.donor_group)}</td>
      <td class="${FOREIGN.has(r.category) && r.support === "1" ? "warnc" : ""}">${esc(r.support === "0" ? "" : r.support)}</td>
      <td>${esc(r.best_taxon)}</td><td>${esc(r.best_pident)}</td><td>${esc(r.best_home_taxon)}</td>
      <td>${esc(r.length)}</td><td>${esc(structLabel(r.struct_group || ""))}</td>
    </tr>`).join("");
  $(".more").textContent = rows.length > LIMIT ? `只显示前 ${LIMIT} 行（共 ${rows.length} 行），完整结果请下载 genes.tsv` : `${rows.length} 行`;
  tb.querySelectorAll("tr.row").forEach((tr) => tr.addEventListener("click", () => toggleDetail(tr, rows[+tr.dataset.i])));
}

function toggleDetail(tr, r) {
  if (tr.nextElementSibling && tr.nextElementSibling.classList.contains("detail")) { tr.nextElementSibling.remove(); return; }
  const items = [
    ["最佳外部命中", r.best_hit ? `${r.best_hit} · ${r.best_taxon}（${r.best_group}）· 一致度 ${r.best_pident}% · E = ${r.best_evalue}` : "无"],
    ["本谱系最近命中", r.best_home_taxon ? `${r.best_home_taxon} · 一致度 ${r.best_home_pident}%` : "本谱系内没有命中"],
    ["哺乳动物命中", r.mammal_hit || "无"],
    ["外部命中数", `${r.n_outside}（其中重复驱动 ${r.repeat_hits}，已剔除）`],
    ["重复区段", r.repeats || "无"],
    ["位置", r.contig ? `${r.contig}:${r.start}-${r.end} (${r.strand})` : "—"],
  ];
  if (r.struct_group) items.push(["结构检索", r.struct_group === "no structural match" ? `无同折叠匹配（pLDDT ${r.plddt}）`
    : `${structLabel(r.struct_group)} · ${r.struct_name || ""} · ${r.struct_taxon || ""} · TM ${r.struct_tm} · pLDDT ${r.plddt}`]);
  if (r.struct_mammal) items.push(["结构上的哺乳动物匹配", r.struct_mammal]);
  const d = document.createElement("tr");
  d.className = "detail";
  d.innerHTML = `<td colspan="${COLS.length}"><dl>${items.map(([a, b]) => `<dt>${a}</dt><dd>${esc(b)}</dd>`).join("")}</dl></td>`;
  tr.after(d);
}

function renderMap() {
  const rows = state.genes.filter((r) => r.contig && r.start);
  if (!rows.length) return;
  $(".map-card").classList.remove("hidden");
  const len = {};
  rows.forEach((r) => { len[r.contig] = Math.max(len[r.contig] || 0, +r.end); });
  const contigs = Object.entries(len).sort((a, b) => b[1] - a[1]).slice(0, 12);
  const W = 1000, L = 120, rowH = 26, max = contigs[0][1];
  let svg = `<svg viewBox="0 0 ${W + L + 10} ${contigs.length * rowH + 10}" width="100%">`;
  contigs.forEach(([c, n], i) => {
    const y = i * rowH + 14, w = (W * n) / max;
    svg += `<text x="0" y="${y + 4}">${esc(c.slice(0, 16))}</text>`;
    svg += `<rect x="${L}" y="${y - 2}" width="${w}" height="4" rx="2" fill="var(--line)"/>`;
    rows.filter((r) => r.contig === c && r.category !== "native").forEach((r) => {
      const x = L + (W * +r.start) / max, col = (CAT[r.category] || {}).color || "var(--stone)";
      const h = FOREIGN.has(r.category) ? 18 : 8;
      svg += `<rect x="${x.toFixed(1)}" y="${y - h / 2}" width="2" height="${h}" fill="${col}"><title>${esc(r.gene_id)} · ${esc((CAT[r.category] || {}).label)}</title></rect>`;
    });
  });
  $(".map").innerHTML = svg + "</svg>" +
    `<p class="muted" style="font-size:12px">高的竖线是外来候选，矮的是孤儿或重复基因；成簇出现的外来候选往往是基因组岛。</p>`;
}

/* -- start ------------------------------------------------------------------------------------ */
function init() {
  state.api = defaultApi();
  $("#api").value = state.api;
  $("#api").addEventListener("change", () => {
    state.api = $("#api").value.trim();
    try { localStorage.setItem("seeddrift.api", state.api); } catch (e) { /* private mode */ }
    boot();
  });
  setupForm();
  boot();
  setInterval(checkHealth, 30000);
}
function boot() { checkHealth(); loadDemos(); loadJobs(); }
init();
