"use strict";

const $ = id => document.getElementById(id);
const metrics = {
  manhattan: "Manhattan lấy trung bình chênh lệch tuyệt đối sau chuẩn hóa.",
  sqeuclidean: "Euclid bình phương lấy trung bình bình phương chênh lệch sau chuẩn hóa.",
  huber: "Huber bình phương chênh lệch nhỏ và tăng tuyến tính khi chênh lệch vượt 1.",
  cosine: "Cosine so sánh hướng của các vector sau chuẩn hóa."
};
const groupNames = {
  A: "Dịch chuyển toàn người", B: "Cử động theo vùng", C: "Phối hợp hai phía",
  D: "Diễn biến theo thời gian", E: "Hình chiếu và mức xám", F: "Thay đổi pixel"
};
const verdicts = { similar: "Tương tự", different: "Khác", unsure: "Chưa chắc" };
const initial = new URLSearchParams(location.search);
const state = {
  query: initial.get("query"), metric: Object.hasOwn(metrics, initial.get("metric")) ? initial.get("metric") : "manhattan",
  d: initial.get("d") !== "0", other: initial.get("other") === "1", labels: false, viewer: false,
  videos: [], visible: [], request: null
};
const num = new Intl.NumberFormat("vi-VN", { minimumFractionDigits: 4, maximumFractionDigits: 4 });
const pct = new Intl.NumberFormat("vi-VN", { style: "percent", maximumFractionDigits: 0 });
const seconds = new Intl.NumberFormat("vi-VN", { maximumFractionDigits: 2 });
const encoded = id => id.split("/").map(encodeURIComponent).join("/");
const alias = video => `${video.person} · ${video.setting}`;

function el(tag, text, className) {
  const element = document.createElement(tag);
  if (text != null) element.textContent = text;
  if (className) element.className = className;
  return element;
}

async function json(url, options) {
  const response = await fetch(url, options);
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || "Không thể đọc dữ liệu.");
  return data;
}

function errorMessage(text = "") {
  $("error").textContent = text;
  $("error").hidden = !text;
}

function updateRatingCount(count) {
  $("rating-count").textContent = `${count} cặp đã đánh giá`;
}

function toggleLabels() {
  $("show-labels").setAttribute("aria-pressed", String(state.labels));
  $("show-labels").textContent = state.labels ? "Ẩn nhãn thật" : "Hiện nhãn thật";
  document.querySelectorAll(".label").forEach(node => { node.hidden = !state.labels; });
  renderList();
}

function addOptions(id, values) {
  const options = values.map(value => {
    const option = el("option", value);
    option.value = value;
    return option;
  });
  $(id).append(...options);
}

function renderList() {
  const tokens = $("filter-text").value.toLowerCase().split(/[\s,]+/).filter(Boolean);
  const action = $("filter-action").value;
  const person = $("filter-person").value;
  const complete = $("filter-complete").checked;
  state.visible = state.videos.filter(video =>
    tokens.every(token => video.id.toLowerCase().includes(token))
    && (!action || video.action === action) && (!person || video.person === person)
    && (!complete || video.dimensions === 324)
  );
  $("library-count").textContent = `${state.visible.length} / ${state.videos.length}`;
  $("random").disabled = !state.visible.length;
  const list = $("video-list");
  const previousScroll = list.scrollTop;
  const items = state.visible.map(video => {
    const button = el("button", null, "video-option");
    button.type = "button";
    button.dataset.id = video.id;
    if (video.id === state.query) button.setAttribute("aria-current", "true");
    const status = el("small");
    const dot = el("span", null, video.dimensions === 324 ? "dot" : "dot partial");
    dot.setAttribute("aria-hidden", "true");
    status.append(dot, video.dimensions === 324 ? "Đủ 324 chiều" : `${video.dimensions}/324 chiều`);
    button.append(el("strong", state.labels ? `${alias(video)} · ${video.action}` : alias(video)), status);
    return button;
  });
  list.replaceChildren(...(items.length ? items : [el("p", "Không tìm thấy video khớp bộ lọc.", "empty")]));
  list.scrollTop = previousScroll;
}

function highlightQuery(scroll) {
  for (const button of $("video-list").children) {
    const active = button.dataset.id === state.query;
    if (active) {
      button.setAttribute("aria-current", "true");
      if (scroll) {
        const list = $("video-list");
        const itemRect = button.getBoundingClientRect();
        const listRect = list.getBoundingClientRect();
        if (itemRect.top < listRect.top || itemRect.bottom > listRect.bottom) {
          list.scrollTop += itemRect.top - listRect.top;
        }
      }
    } else {
      button.removeAttribute("aria-current");
    }
  }
}

function player(video, description) {
  const node = el("video");
  node.controls = true;
  node.playsInline = true;
  node.preload = "none";
  node.poster = `/poster/${encoded(video.id)}`;
  node.src = `/media/${encoded(video.id)}`;
  node.setAttribute("aria-label", description);
  node.addEventListener("play", () => {
    document.querySelectorAll("video").forEach(other => { if (other !== node) other.pause(); });
  });
  return node;
}

function label(video) {
  const node = el("p", `${video.action} · ${video.id.split("/").at(-1)}`, "label");
  node.hidden = !state.labels;
  return node;
}

function detailsFor(result) {
  const details = el("details");
  const table = el("table", null, "group-table");
  const header = table.createTHead().insertRow();
  for (const text of ["Nhóm", "Chiều chung", "Trọng số", "Khoảng cách"]) {
    const cell = el("th", text);
    cell.scope = "col";
    header.append(cell);
  }
  const rows = table.createTBody();
  for (const group of result.groups) {
    const row = rows.insertRow();
    row.append(el("td", `${group.key} · ${groupNames[group.key]}`),
               el("td", `${group.shared}/${group.size}`),
               el("td", pct.format(group.weight)),
               el("td", group.weight && group.distance != null ? num.format(group.distance) : "—"));
  }
  const distance = el("p", `Khoảng cách = tổng (khoảng cách nhóm × trọng số) / tổng trọng số = ${num.format(result.distance)}.`, "calculation");
  const score = el("p", `Điểm = 1 / (1 + ${num.format(result.distance)}) ≈ ${num.format(result.score)}.`, "calculation");
  details.append(el("summary", "Xem 6 nhóm đặc trưng và phép tính"), table, distance, score);
  return details;
}

function ratingControls(result, query) {
  const container = el("div", null, "rating");
  container.setAttribute("role", "group");
  container.setAttribute("aria-label", "Nhận xét mức tương đồng");
  let verdict = result.verdict;
  const buttons = Object.entries(verdicts).map(([id, text]) => {
    const button = el("button", text, "button");
    button.type = "button";
    button.dataset.verdict = id;
    button.setAttribute("aria-pressed", String(id === verdict));
    button.addEventListener("click", async () => {
      const next = id === verdict ? null : id;
      buttons.forEach(item => { item.disabled = true; });
      try {
        const data = await json("/api/rating", {
          method: "POST", headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ query: query.id, candidate: result.video.id, verdict: next })
        });
        verdict = next;
        buttons.forEach(item => item.setAttribute("aria-pressed", String(item.dataset.verdict === verdict)));
        updateRatingCount(data.rating_count);
        $("announcement").textContent = next ? "Đã lưu nhận xét." : "Đã bỏ nhận xét.";
      } catch (error) {
        errorMessage(error.message);
      } finally {
        buttons.forEach(item => { item.disabled = false; });
      }
    });
    return button;
  });
  container.append(...buttons);
  return container;
}

function renderResults(data) {
  const query = data.query;
  $("selected-name").textContent = alias(query);
  $("selected-meta").textContent = `${query.dimensions}/324 chiều đo được · ${data.candidates} video ứng viên`;
  $("position").textContent = `Truy vấn ${state.videos.findIndex(video => video.id === query.id) + 1} / ${state.videos.length}`;
  const head = el("div", null, "query-heading");
  head.append(el("span", "Q", "query-mark"), el("h3", "Video truy vấn"));
  $("query-card").replaceChildren(head, player(query, "Phát video truy vấn"),
    el("p", `${seconds.format(query.frames / query.fps)} giây · ${query.fps} fps · ${query.valid_pairs} cặp khung hình hợp lệ`, "query-stats"),
    label(query));
  const cards = data.results.map((result, i) => {
    const video = result.video;
    const card = el("article", null, "result-card");
    const info = el("div", null, "result-info");
    const heading = el("div", null, "result-heading");
    const score = el("span", num.format(result.score), "score");
    score.setAttribute("aria-label", `Điểm tương đồng ${score.textContent}`);
    heading.append(el("span", i + 1, "rank"), el("h3", alias(video)), score);
    const d = result.groups.find(group => group.key === "D");
    const status = d.weight ? `D chung ${d.shared}/${d.size} · ` : "";
    info.append(heading, el("p", status + `Khoảng cách ${num.format(result.distance)}`, "meta"), label(video));
    if (!state.viewer) info.append(ratingControls(result, query));
    card.append(player(video, `Phát video hạng ${i + 1}`), info, detailsFor(result));
    return card;
  });
  $("results-title").textContent = `${cards.length} video gần nhất`;
  $("results").replaceChildren(...(cards.length ? cards : [el("p", "Không có ứng viên đủ dữ liệu để xếp hạng.", "empty")]));
  if (!state.viewer) updateRatingCount(data.rating_count);
  $("announcement").textContent = `Đã tìm ${cards.length} video gần nhất cho ${alias(query)}.`;
}

function updateSettings() {
  $("include-d").checked = state.d;
  $("other-people").checked = state.other;
  $("metric-note").textContent = metrics[state.metric] + (state.d ? " Trọng số D theo số chiều hợp lệ chung." : " Tính trên A, B, C, E, F.");
  document.querySelectorAll("[data-metric]").forEach(button =>
    button.setAttribute("aria-pressed", String(button.dataset.metric === state.metric)));
}

async function search() {
  if (!state.query) return;
  state.request?.abort();
  const request = new AbortController();
  state.request = request;
  const params = new URLSearchParams({ query: state.query, metric: state.metric, d: state.d ? "1" : "0", other: state.other ? "1" : "0" });
  history.replaceState(null, "", `?${params}`);
  updateSettings();
  errorMessage();
  $("results").setAttribute("aria-busy", "true");
  try {
    const data = await json(`/api/search?${params}`, { signal: request.signal });
    if (state.request === request) renderResults(data);
  } catch (error) {
    if (error.name !== "AbortError") errorMessage(`Tìm kiếm thất bại. ${error.message}`);
  } finally {
    if (state.request === request) $("results").removeAttribute("aria-busy");
  }
}

$("filter-text").addEventListener("input", renderList);
for (const input of ["filter-action", "filter-person", "filter-complete"]) $(input).addEventListener("change", renderList);
$("video-list").addEventListener("click", event => {
  const option = event.target.closest("button[data-id]");
  if (!option || state.query === option.dataset.id) return;
  state.query = option.dataset.id;
  highlightQuery(false);
  search();
});
$("random").addEventListener("click", () => {
  const choices = state.visible.length > 1 ? state.visible.filter(video => video.id !== state.query) : state.visible;
  if (!choices.length) return;
  state.query = choices[Math.floor(Math.random() * choices.length)].id;
  highlightQuery(true);
  search();
});
$("show-labels").addEventListener("click", () => { state.labels = !state.labels; toggleLabels(); });
$("include-d").addEventListener("change", event => { state.d = event.target.checked; search(); });
$("other-people").addEventListener("change", event => { state.other = event.target.checked; search(); });
document.querySelectorAll("[data-metric]").forEach(button => {
  button.addEventListener("click", () => {
    if (state.metric === button.dataset.metric) return;
    state.metric = button.dataset.metric;
    search();
  });
});

async function initialize() {
  updateSettings();
  try {
    const data = await json("/api/videos");
    state.videos = data.videos;
    state.viewer = data.viewer === true;
    $("rating-count").hidden = state.viewer;
    $("export-ratings").hidden = state.viewer;
    if (!state.videos.some(video => video.id === state.query)) state.query = data.default;
    const people = [...new Set(state.videos.map(video => video.person))].sort();
    const actions = [...new Set(state.videos.map(video => video.action))].sort();
    addOptions("filter-person", people);
    addOptions("filter-action", actions);
    $("dataset-summary").textContent = `KTH · ${state.videos.length} video · ${people.length} người · ${actions.length} hành động`;
    renderList();
    highlightQuery(true);
    search();
  } catch (error) {
    errorMessage(`Không thể tải kho video. ${error.message}`);
  }
}

initialize();
