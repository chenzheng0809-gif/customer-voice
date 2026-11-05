/* 客户对话智能分析看板 —— 纯 vanilla JS，无框架依赖。
 *
 * 结构说明（也方便你讲解）：
 *   1. api()     —— 统一封装 fetch，避免各处重复写
 *   2. render*() —— 每个视图一个函数，只负责把数据变成 DOM
 *   3. init()    —— 事件绑定 + 首屏加载
 * 所有数据都来自后端 SQLite，不在前端硬编码任何数字。
 */

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => [...document.querySelectorAll(sel)];

const SENT_COLOR = { 负面: "var(--neg)", 中性: "var(--neu)", 正面: "var(--pos)" };
const SENT_CLASS = { 负面: "neg", 中性: "neu", 正面: "pos" };

/** 统一的请求封装：出错时抛出带信息的 Error，由调用方决定怎么提示 */
async function api(path, opts) {
  const res = await fetch(path, opts);
  const data = await res.json();
  if (!res.ok || data.error) throw new Error(data.error || `HTTP ${res.status}`);
  return data;
}

/** 把数字格式化成百分比 */
const pct = (x) => `${(x * 100).toFixed(1)}%`;

/* ------------------------------------------------------------ 标签页 */
function bindTabs() {
  $$(".tab").forEach((btn) => {
    btn.addEventListener("click", () => {
      $$(".tab").forEach((b) => b.classList.remove("active"));
      $$(".panel").forEach((p) => p.classList.remove("active"));
      btn.classList.add("active");
      $(`#tab-${btn.dataset.tab}`).classList.add("active");
    });
  });
}

/* ------------------------------------------------------------ 总览 */
async function renderOverview() {
  const d = await api("/api/overview");
  const { counts } = d;

  // KPI 卡片
  const kpis = [
    ["会话数", counts.sessions],
    ["消息总数", counts.messages],
    ["客户发言", counts.customer_msgs],
    ["模型版本", counts.models],
  ];
  $("#kpi").innerHTML = kpis
    .map(([k, v]) => `<div class="card"><div class="v">${v}</div><div class="k">${k}</div></div>`)
    .join("");

  // 情感条形图
  const total = d.sentiment.reduce((s, r) => s + r.n, 0);
  const max = Math.max(...d.sentiment.map((r) => r.n));
  $("#sent-bars").innerHTML = d.sentiment
    .map((r) => `
      <div class="bar-row">
        <span class="label">${r.label}</span>
        <span class="track">
          <span class="fill" style="width:${(r.n / max) * 100}%;
                background:${SENT_COLOR[r.label]}"></span>
        </span>
        <span class="num">${r.n} · ${pct(r.n / total)}</span>
      </div>`)
    .join("");

  // 意图 × 情感 热力表
  const intents = d.intent.map((r) => r.label);
  const sentiments = ["负面", "中性", "正面"];
  const map = new Map(d.cross.map((r) => [`${r.intent}|${r.sentiment}`, r.n]));
  const maxCell = Math.max(...d.cross.map((r) => r.n));

  let html = `<div class="hdr"></div>`;
  sentiments.forEach((s) => (html += `<div class="hdr">${s}</div>`));
  intents.forEach((it) => {
    html += `<div class="hdr" style="text-align:right;padding-right:8px">${it}</div>`;
    sentiments.forEach((s) => {
      const n = map.get(`${it}|${s}`) || 0;
      // alpha 随数值上升，但保留最低可见度
      const a = 0.08 + (n / maxCell) * 0.55;
      html += `<div class="cell" style="background:rgba(79,140,255,${a.toFixed(3)})">${n}</div>`;
    });
  });
  $("#cross-heat").style.gridTemplateColumns = `72px repeat(${sentiments.length}, 1fr)`;
  $("#cross-heat").innerHTML = html;
}

/* ------------------------------------------------------------ 会话列表 */
async function renderSessions(riskOnly = false) {
  const rows = await api(`/api/sessions${riskOnly ? "?risk=1" : ""}`);
  const tbody = $("#sess-table tbody");
  tbody.innerHTML = rows
    .map(
      (r) => `
      <tr data-sid="${r.session_id}">
        <td>${r.session_id}</td>
        <td class="num">${r.msgs}</td>
        <td class="num">${r.neg}</td>
        <td class="num" style="color:${r.neg_ratio >= 0.5 ? "var(--neg)" : "inherit"}">
          ${pct(r.neg_ratio)}
        </td>
        <td><button class="link">查看</button></td>
      </tr>`
    )
    .join("");
  $("#sess-detail").innerHTML = "";

  // 事件委托：只绑一次，而不是给每行绑定（行可能有上百个）
  tbody.onclick = (e) => {
    const tr = e.target.closest("tr");
    if (tr) showSession(tr.dataset.sid);
  };
}

async function showSession(sid) {
  const d = await api(`/api/session/${sid}`);
  const html = d.messages
    .map((m) => {
      const p = d.predictions[m.msg_id] || {};
      const tags = [];
      if (m.role === "customer") {
        if (m.intent) tags.push(`<span class="tag">金标意图：${m.intent}</span>`);
        if (m.sentiment) tags.push(`<span class="tag ${SENT_CLASS[m.sentiment]}">金标情感：${m.sentiment}</span>`);
        if (p.intent) tags.push(`<span class="tag pred">预测意图：${p.intent.label} (${pct(p.intent.confidence)})</span>`);
        if (p.sentiment) tags.push(`<span class="tag pred">预测情感：${p.sentiment.label} (${pct(p.sentiment.confidence)})</span>`);
      }
      return `
        <div class="msg ${m.role}">
          <span class="who">${m.role === "customer" ? "客户" : "客服"}</span>
          <span class="bubble">${escapeHtml(m.text)}
            ${tags.length ? `<div class="tags">${tags.join("")}</div>` : ""}
          </span>
        </div>`;
    })
    .join("");
  $("#sess-detail").innerHTML = `<h2 style="font-size:14px;color:var(--muted);margin-bottom:14px">
    ${sid} 对话明细</h2>${html}`;
}

/** 用户输入要转义，否则话术里的 < 会破坏 DOM —— 前端安全基本功 */
function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  })[c]);
}

/* ------------------------------------------------------------ 模型对比 */
async function renderModels() {
  const rows = await api("/api/models");
  const grouped = new Map();
  rows.forEach((r) => {
    const key = `${r.model_version}`;
    if (!grouped.has(key)) grouped.set(key, { model: r.model_version, task: r.task });
    grouped.get(key)[r.metric] = r.value;
  });

  $("#model-table tbody").innerHTML = [...grouped.values()]
    .map(
      (r) => `
      <tr>
        <td>${r.model}</td>
        <td>${r.task === "intent" ? "意图分类" : "情感分析"}</td>
        <td class="num">${r.accuracy != null ? pct(r.accuracy) : "-"}</td>
        <td class="num">${r.macro_f1 != null ? r.macro_f1.toFixed(4) : "-"}</td>
      </tr>`
    )
    .join("");

  // 混淆矩阵
  const conf = await api("/api/confusion");
  const labels = [...new Set(conf.map((c) => c.gold))].sort();
  const map = new Map(conf.map((c) => [`${c.gold}|${c.pred}`, c.n]));
  const rowSum = {};
  labels.forEach((g) => {
    rowSum[g] = labels.reduce((s, p) => s + (map.get(`${g}|${p}`) || 0), 0);
  });

  let html = `<div class="hdr"></div>`;
  labels.forEach((l) => (html += `<div class="hdr">预测 ${l}</div>`));
  labels.forEach((g) => {
    html += `<div class="hdr" style="text-align:right;padding-right:8px">金标 ${g}</div>`;
    labels.forEach((p) => {
      const n = map.get(`${g}|${p}`) || 0;
      const ratio = rowSum[g] ? n / rowSum[g] : 0;
      const bg = g === p
        ? `rgba(63,185,80,${(0.1 + ratio * 0.6).toFixed(3)})`
        : `rgba(229,83,75,${(0.05 + ratio * 0.6).toFixed(3)})`;
      html += `<div class="cell" style="background:${bg}">${n}</div>`;
    });
  });
  const el = $("#confusion");
  el.style.gridTemplateColumns = `84px repeat(${labels.length}, 1fr)`;
  el.innerHTML = html;
}

/* ------------------------------------------------------------ Badcase */
async function renderBadcases() {
  const rows = await api("/api/badcases");
  $("#badcase-list").innerHTML = rows.length
    ? rows
        .map((r) => {
          const intentOk = r.pred_intent === r.gold_intent;
          const sentOk = r.pred_sent === r.gold_sent;
          return `
          <div class="badcase">
            <div class="text">${escapeHtml(r.text)}</div>
            <div class="meta">
              <span class="${intentOk ? "right" : "wrong"}">
                意图：金标 ${r.gold_intent} / 预测 ${r.pred_intent} (${pct(r.intent_conf)})
              </span>
              <span class="${sentOk ? "right" : "wrong"}">
                情感：金标 ${r.gold_sent} / 预测 ${r.pred_sent} (${pct(r.sent_conf)})
              </span>
            </div>
          </div>`;
        })
        .join("")
    : `<p class="hint">没有 badcase（或还没跑模型）</p>`;
}

/* ------------------------------------------------------------ 在线分析 */
function bindAnalyze() {
  $("#analyze-btn").addEventListener("click", async () => {
    const text = $("#analyze-input").value.trim();
    if (!text) return;
    $("#analyze-out").innerHTML = `<span class="hint">分析中…</span>`;
    try {
      const d = await api("/api/analyze", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
      });
      $("#analyze-out").innerHTML = `
        <div class="item">
          <div class="k">意图</div>
          <div class="v">${d.intent.label} <span class="hint">${pct(d.intent.confidence)}</span></div>
        </div>
        <div class="item">
          <div class="k">情感</div>
          <div class="v" style="color:${SENT_COLOR[d.sentiment.label]}">
            ${d.sentiment.label} <span class="hint">${pct(d.sentiment.confidence)}</span>
          </div>
        </div>`;
    } catch (e) {
      $("#analyze-out").innerHTML = `<span class="wrong">${e.message}</span>`;
    }
  });
}

/* ------------------------------------------------------------ 启动 */
async function init() {
  bindTabs();
  bindAnalyze();
  try {
    await renderOverview();
  } catch (e) {
    console.error(e);
  }
  // 其余视图懒加载：切到对应标签时才请求，减少首屏压力
  $$(".tab").forEach((btn) => {
    btn.addEventListener("click", async () => {
      const name = btn.dataset.tab;
      if (name === "sessions") await renderSessions($("#risk-only").checked);
      if (name === "models") await renderModels();
      if (name === "badcases") await renderBadcases();
    });
  });
  $("#risk-only").addEventListener("change", (e) => renderSessions(e.target.checked));
}

init();