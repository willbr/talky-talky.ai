// Recording splitter: correct proposed sentence / word / phoneme boundaries
// for one sentence of a recording at a time, then save to harvest copies.
//
// The server proposes (DTW against a synthesised reference). Dragging an edge
// and letting go pins it and re-proposes the rest of the sentence around it.

const SPEC_H = 150;
const WORD_H = 24;
const PH_H = 20;
const HIT_PX = 6;
const MIN_MS = 5;
const PIN_COLOUR = '#e8b04a';
const MIN_VIEW_MS = 120;
const WHEEL_ZOOM = 0.0015;   // per wheel-delta unit; ~1.2x per notch

let deps = null;
let S = null;   // splitter state; null when closed
let windowListenersAttached = false;

export function initSplitter(d) {
  deps = d;
}

export function closeSplitter() {
  if (S && S.container) S.container.remove();
  S = null;
}

async function getJson(url, opts) {
  const r = await fetch(url, opts);
  const body = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(body.error || r.statusText);
  return body;
}

function postJson(url, body) {
  return getJson(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body || {}),
  });
}

export async function openSplitter(recordingId, sentenceIdx = 0) {
  if (deps.beforeOpen) deps.beforeOpen();
  closeSplitter();
  let rec;
  try {
    rec = await getJson(`/api/recordings/${encodeURIComponent(recordingId)}`);
  } catch (e) {
    deps.setStatus(`Splitter: ${e.message}`, true);
    return;
  }
  S = {
    rec, idx: Math.max(0, Math.min(sentenceIdx ?? 0, rec.sentences.length - 1)),
    data: null, pins: [], edited: new Set(), drag: null, view: null, detail: null,
    audioCtx: null, audioBuf: null, container: null, canvas: null, els: {},
  };
  buildDom();
  loadAudio(recordingId);
  await loadSentence(S.idx, {});
}

async function loadAudio(recordingId) {
  try {
    const r = await fetch(`/api/recordings/${encodeURIComponent(recordingId)}/audio`);
    const buf = await r.arrayBuffer();
    const ctx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 22050 });
    const decoded = await ctx.decodeAudioData(buf);
    if (S && S.rec.id === recordingId) { S.audioCtx = ctx; S.audioBuf = decoded; }
  } catch (e) {
    deps.setStatus(`Splitter audio: ${e.message}`, true);
  }
}

function play(startMs, endMs) {
  if (!S || !S.audioBuf) return;
  if (S.audioCtx.state !== 'running') S.audioCtx.resume();
  const src = S.audioCtx.createBufferSource();
  src.buffer = S.audioBuf;
  src.connect(S.audioCtx.destination);
  src.start(0, startMs / 1000, Math.max(0.01, (endMs - startMs) / 1000));
}

async function loadSentence(idx, body) {
  if (!S) return;
  S.idx = idx;
  S.els.title.textContent = 'proposing…';
  try {
    const data = await postJson(
      `/api/recordings/${encodeURIComponent(S.rec.id)}/sentences/${idx}/propose`, body);
    if (!S) return;
    const sameSentence = S.data && S.data.sentence.idx === idx;
    S.data = data;
    S.rec.sentences[idx] = data.sentence;
    // Re-proposals (pins, text) keep the zoom; a new sentence starts fitted.
    if (!sameSentence || !S.view) { S.view = [...data.view_ms]; S.detail = null; }
    scheduleDetail();
    if (!body.pins) { S.pins = []; S.edited = new Set(); }
    // Saved alignments come back with their stored edited flags.
    for (const w of data.words) if (w.edited) S.edited.add(`w:${w.idx}`);
    for (const p of data.phonemes) if (p.edited) S.edited.add(`p:${p.idx}`);
  } catch (e) {
    deps.setStatus(`Propose: ${e.message}`, true);
  }
  render();
}

// --- DOM ------------------------------------------------------------------------

function button(text, title, onClick, cls = '') {
  const b = document.createElement('button');
  b.textContent = text;
  b.title = title;
  if (cls) b.className = cls;
  b.addEventListener('click', onClick);
  return b;
}

function buildDom() {
  const c = document.createElement('section');
  c.id = 'splitter';

  const header = document.createElement('div');
  header.className = 'splitter-header';
  const title = document.createElement('h3');
  const progress = document.createElement('span');
  progress.className = 'splitter-progress';
  const takes = document.createElement('select');
  takes.title = 'Other takes of this text';
  takes.addEventListener('change', () => openSplitter(takes.value, 0));
  const nav = document.createElement('span');
  nav.className = 'splitter-nav';
  const prev = button('◀', 'Previous sentence', () => S.idx > 0 && loadSentence(S.idx - 1, {}));
  const pos = document.createElement('span');
  const next = button('▶', 'Next sentence',
    () => S.idx < S.rec.sentences.length - 1 && loadSentence(S.idx + 1, {}));
  nav.append(prev, pos, next);
  const close = button('×', 'Close', closeSplitter, 'dismiss');
  header.append(title, progress, takes, nav, close);

  const textRow = document.createElement('div');
  textRow.className = 'splitter-text';
  const text = document.createElement('input');
  text.type = 'text';
  text.title = 'Fix the text if you misread it, then press Enter to re-propose';
  text.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') { e.preventDefault(); text.blur(); }
  });
  text.addEventListener('change', () => {
    if (S && S.data && text.value.trim() && text.value.trim() !== S.data.sentence.text) {
      loadSentence(S.idx, { text: text.value.trim() });
    }
  });
  const badge = document.createElement('span');
  badge.className = 'splitter-badge';
  textRow.append(text, badge);

  const canvas = document.createElement('canvas');
  canvas.height = SPEC_H + WORD_H + PH_H;
  canvas.className = 'splitter-canvas';

  const actions = document.createElement('div');
  actions.className = 'actions';
  const playBtn = button('▶ Sentence', 'Play the whole sentence', () => {
    const w = S.data && S.data.words;
    if (w && w.length) play(w[0].start_ms, w[w.length - 1].end_ms);
  });
  const repropose = button('Re-propose', 'Discard edits and pins; propose fresh', () => {
    loadSentence(S.idx, { fresh: true });
  });
  const exclude = button('Exclude', 'Misread or unusable: no copies from this sentence', toggleExclude);
  const save = button('Save sentence', 'Store these boundaries and cut copies', saveSentence, 'save');
  actions.append(playBtn, repropose, exclude, save);

  const zoom = document.createElement('div');
  zoom.className = 'splitter-zoom';
  const zoomLabel = document.createElement('span');
  zoomLabel.className = 'splitter-zoom-label';
  const zoomHint = document.createElement('span');
  zoomHint.className = 'splitter-zoom-hint';
  zoomHint.textContent = 'wheel to zoom · drag background to pan · click a word to hear it';
  zoom.append(
    button('−', 'Zoom out', () => S && S.data && zoomCentre(1.6)),
    button('＋', 'Zoom in', () => S && S.data && zoomCentre(1 / 1.6)),
    button('Fit', 'Show the whole sentence', fitView),
    zoomLabel, zoomHint,
  );

  c.append(header, textRow, zoom, canvas, actions);
  deps.mount.appendChild(c);
  S.container = c;
  S.canvas = canvas;
  S.els = { title, progress, takes, prev, next, pos, text, badge, exclude, save, zoomLabel };
  attachCanvas(canvas);
}

function render() {
  if (!S) return;
  const { rec, idx, els } = S;
  const sents = rec.sentences;
  const saved = sents.filter((s) => s.status === 'saved').length;
  const takeNo = rec.takes.indexOf(rec.id) + 1;
  els.title.textContent = `Splitting take ${takeNo}/${rec.takes.length}`;
  els.progress.textContent = `${saved}/${sents.length} saved`;
  els.pos.textContent = ` sentence ${idx + 1}/${sents.length} `;
  els.prev.disabled = idx === 0;
  els.next.disabled = idx >= sents.length - 1;
  els.takes.hidden = rec.takes.length < 2;
  els.takes.innerHTML = '';
  rec.takes.forEach((id, i) => {
    const o = document.createElement('option');
    o.value = id;
    o.textContent = `take ${i + 1}`;
    o.selected = id === rec.id;
    els.takes.appendChild(o);
  });
  const sent = sents[idx];
  if (document.activeElement !== els.text) els.text.value = sent.text;
  els.badge.textContent = sent.status;
  els.badge.dataset.status = sent.status;
  els.exclude.textContent = sent.status === 'excluded' ? 'Include' : 'Exclude';
  els.save.disabled = !S.data || sent.status === 'excluded';
  draw();
}

// --- drawing --------------------------------------------------------------------

function xOf(ms) {
  const [v0, v1] = S.view;
  return ((ms - v0) / Math.max(1, v1 - v0)) * S.canvas.width;
}

function msOf(x) {
  const [v0, v1] = S.view;
  return v0 + (x / S.canvas.width) * (v1 - v0);
}

// --- zoom -------------------------------------------------------------------------

function setView(v0, v1) {
  const dur = S.rec.duration_ms;
  let span = Math.max(MIN_VIEW_MS, Math.min(dur, v1 - v0));
  v0 = Math.max(0, Math.min(dur - span, v0));
  S.view = [v0, v0 + span];
  draw();
  scheduleDetail();
}

function zoomAt(ms, factor) {
  const [v0, v1] = S.view;
  const span = Math.max(MIN_VIEW_MS, Math.min(S.rec.duration_ms, (v1 - v0) * factor));
  const frac = (ms - v0) / (v1 - v0);
  setView(ms - frac * span, ms - frac * span + span);
}

function zoomCentre(factor) {
  zoomAt((S.view[0] + S.view[1]) / 2, factor);
}

function fitView() {
  if (S && S.data) setView(...S.data.view_ms);
}

// A sharper spectrogram for the visible span (the proposal's own covers the
// whole sentence at ~12 ms per column, which turns to blocks when zoomed).
let detailTimer = null;

function scheduleDetail() {
  clearTimeout(detailTimer);
  detailTimer = setTimeout(fetchDetail, 120);
}

async function fetchDetail() {
  if (!S || !S.data || !S.view) return;
  const base = S.data.spectrogram;
  const [v0, v1] = S.view;
  const [b0, b1] = S.data.view_ms;
  const span = v1 - v0;
  if (span > 0.6 * (b1 - b0) && v0 >= b0 && v1 <= b1) return;   // base is sharp enough
  const d = S.detail;
  if (d && d.start_ms <= v0 && d.end_ms >= v1 && d.view_span < span * 1.5) return;
  const margin = span * 0.25;
  const q = new URLSearchParams({
    start_ms: Math.max(0, Math.round(v0 - margin)),
    end_ms: Math.min(S.rec.duration_ms, Math.round(v1 + margin)),
    width: Math.round(S.canvas.width * 1.5),
    lo: base.lo, hi: base.hi,
  });
  const recId = S.rec.id;
  try {
    const spec = await getJson(`/api/recordings/${encodeURIComponent(recId)}/spectrogram?${q}`);
    if (!S || S.rec.id !== recId || !spec.n_frames) return;
    spec.bytes = atob(spec.data);
    spec.end_ms = spec.start_ms + spec.duration_ms;
    spec.view_span = span;
    S.detail = spec;
    draw();
  } catch (e) { /* keep the coarse spectrogram */ }
}

function isPinned(tier, idx, edge) {
  return S.pins.some((p) => p.tier === tier && p.idx === idx && p.edge === edge);
}

function draw() {
  const canvas = S.canvas;
  const cssW = canvas.clientWidth || 800;
  if (canvas.width !== cssW) canvas.width = cssW;
  const w = canvas.width;
  const ctx = canvas.getContext('2d');
  ctx.fillStyle = '#1a1b1e';
  ctx.fillRect(0, 0, w, canvas.height);
  if (!S.data) return;

  // Spectrogram of the view: the sharp detail where it covers, else the
  // proposal's coarse one; columns outside both stay dark.
  const base = S.data.spectrogram;
  if (base && base.n_frames && !base.bytes) {
    base.bytes = atob(base.data);
    base.start_ms = S.data.view_ms[0];
  }
  const sources = [S.detail, base].filter((s) => s && s.n_frames);
  if (sources.length) {
    const img = ctx.createImageData(w, SPEC_H);
    const rowBin = [];
    for (let y = 0; y < SPEC_H; y++) rowBin.push((SPEC_H - 1 - y) / SPEC_H);
    for (let x = 0; x < w; x++) {
      const ms = msOf(x + 0.5);
      let src = null;
      let f = -1;
      for (const s of sources) {
        const fr = Math.round((ms - s.start_ms - (s.centre_ms || 0)) / s.hop_ms);
        if (fr >= 0 && fr < s.n_frames) { src = s; f = fr; break; }
      }
      if (!src) {
        for (let y = 0; y < SPEC_H; y++) img.data.set([26, 27, 30, 255], (y * w + x) * 4);
        continue;
      }
      const nB = src.n_freqs;
      for (let y = 0; y < SPEC_H; y++) {
        const b = Math.min(nB - 1, Math.floor(rowBin[y] * nB));
        const v = src.bytes.charCodeAt(f * nB + b);
        const i = (y * w + x) * 4;
        img.data[i] = (v * 143) >> 8;
        img.data[i + 1] = (v * 223) >> 8;
        img.data[i + 2] = (v * 155) >> 8;
        img.data[i + 3] = 255;
      }
    }
    ctx.putImageData(img, 0, 0);
  }
  S.els.zoomLabel.textContent = `${((S.view[1] - S.view[0]) / 1000).toFixed(2)} s visible`;

  // Bands: grey background = pause / outside any word.
  ctx.fillStyle = '#2a2c31';
  ctx.fillRect(0, SPEC_H, w, WORD_H + PH_H);
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  for (const wd of S.data.words) {
    const x0 = xOf(wd.start_ms);
    const x1 = xOf(wd.end_ms);
    ctx.fillStyle = S.edited.has(`w:${wd.idx}`) ? '#3a4a6a' : '#2a3550';
    ctx.fillRect(x0, SPEC_H, x1 - x0, WORD_H);
    ctx.fillStyle = wd.in_dict === false ? '#e6a0a0' : '#ffffff';
    ctx.font = 'bold 11px ui-monospace, monospace';
    ctx.fillText(wd.label, (x0 + x1) / 2, SPEC_H + WORD_H / 2, Math.max(4, x1 - x0 - 2));
  }
  for (const p of S.data.phonemes) {
    const x0 = xOf(p.start_ms);
    const x1 = xOf(p.end_ms);
    ctx.fillStyle = S.edited.has(`p:${p.idx}`) ? '#2c3548' : '#1f2533';
    ctx.fillRect(x0, SPEC_H + WORD_H, x1 - x0, PH_H);
    ctx.fillStyle = '#b8c0c8';
    ctx.font = '10px ui-monospace, monospace';
    ctx.fillText(p.label, (x0 + x1) / 2, SPEC_H + WORD_H + PH_H / 2, Math.max(4, x1 - x0 - 2));
  }

  // Phoneme edges inside words: thin.
  ctx.lineWidth = 1;
  for (const p of S.data.phonemes) {
    const prev = S.data.phonemes[p.idx - 1];
    if (!prev || prev.parent_idx !== p.parent_idx) continue;
    ctx.strokeStyle = isPinned('phoneme', p.idx, 'start') ? PIN_COLOUR : 'rgba(255,255,255,0.35)';
    const x = xOf(p.start_ms) + 0.5;
    ctx.beginPath();
    ctx.moveTo(x, SPEC_H * 0.5);
    ctx.lineTo(x, canvas.height);
    ctx.stroke();
  }
  // Word edges: full height, brighter.
  ctx.lineWidth = 2;
  for (const wd of S.data.words) {
    for (const edge of ['start', 'end']) {
      ctx.strokeStyle = isPinned('word', wd.idx, edge) ? PIN_COLOUR : '#e8e8e8';
      const x = xOf(edge === 'start' ? wd.start_ms : wd.end_ms);
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, SPEC_H + WORD_H);
      ctx.stroke();
    }
  }
}

// --- interaction ------------------------------------------------------------------

function wordPhonemes(wi) {
  return S.data.phonemes.filter((p) => p.parent_idx === wi);
}

function hitEdge(x, y) {
  let best = null;
  let bestD = HIT_PX + 1;
  const consider = (handle, ms) => {
    const d = Math.abs(xOf(ms) - x);
    if (d < bestD) { bestD = d; best = handle; }
  };
  if (y >= SPEC_H + WORD_H) {
    for (const p of S.data.phonemes) {
      const prev = S.data.phonemes[p.idx - 1];
      if (prev && prev.parent_idx === p.parent_idx) consider({ tier: 'phoneme', idx: p.idx, edge: 'start' }, p.start_ms);
    }
  }
  if (!best) {
    for (const wd of S.data.words) {
      consider({ tier: 'word', idx: wd.idx, edge: 'start' }, wd.start_ms);
      consider({ tier: 'word', idx: wd.idx, edge: 'end' }, wd.end_ms);
    }
  }
  return best;
}

function hitSegment(x, y) {
  const ms = msOf(x);
  if (y >= SPEC_H + WORD_H) {
    return S.data.phonemes.find((p) => p.start_ms <= ms && ms < p.end_ms) || null;
  }
  return S.data.words.find((wd) => wd.start_ms <= ms && ms < wd.end_ms) || null;
}

function applyDrag(h, ms) {
  const W = S.data.words;
  const P = S.data.phonemes;
  if (h.tier === 'phoneme') {
    const p = P[h.idx];
    const prev = P[h.idx - 1];
    ms = Math.max(prev.start_ms + MIN_MS, Math.min(p.end_ms - MIN_MS, ms));
    p.start_ms = prev.end_ms = Math.round(ms);
    S.edited.add(`p:${p.idx}`).add(`p:${prev.idx}`);
    return;
  }
  const wd = W[h.idx];
  const phs = wordPhonemes(h.idx);
  const v0 = 0;
  const v1 = S.rec.duration_ms;
  if (h.edge === 'start') {
    const lo = h.idx > 0 ? W[h.idx - 1].end_ms : v0;
    const hi = (phs.length ? phs[0].end_ms : wd.end_ms) - MIN_MS;
    wd.start_ms = Math.round(Math.max(lo, Math.min(hi, ms)));
    if (phs.length) { phs[0].start_ms = wd.start_ms; S.edited.add(`p:${phs[0].idx}`); }
  } else {
    const lo = (phs.length ? phs[phs.length - 1].start_ms : wd.start_ms) + MIN_MS;
    const hi = h.idx < W.length - 1 ? W[h.idx + 1].start_ms : v1;
    wd.end_ms = Math.round(Math.max(lo, Math.min(hi, ms)));
    if (phs.length) { phs[phs.length - 1].end_ms = wd.end_ms; S.edited.add(`p:${phs[phs.length - 1].idx}`); }
  }
  S.edited.add(`w:${wd.idx}`);
}

function attachCanvas(canvas) {
  canvas.addEventListener('mousedown', (e) => {
    if (!S || !S.data) return;
    const r = canvas.getBoundingClientRect();
    const x = e.clientX - r.left;
    const y = e.clientY - r.top;
    const h = hitEdge(x, y);
    S.drag = { handle: h, moved: false, x, y, view: [...S.view] };
    canvas.style.cursor = h ? 'ew-resize' : 'grabbing';
  });
  canvas.addEventListener('mousemove', (e) => {
    if (!S || !S.data || S.drag) return;
    const r = canvas.getBoundingClientRect();
    canvas.style.cursor = hitEdge(e.clientX - r.left, e.clientY - r.top) ? 'ew-resize' : 'grab';
  });
  canvas.addEventListener('wheel', (e) => {
    if (!S || !S.data) return;
    e.preventDefault();
    const r = canvas.getBoundingClientRect();
    const scale = e.deltaMode === 1 ? 40 : 1;   // lines → px
    const dx = e.deltaX * scale;
    const dy = e.deltaY * scale;
    if (e.shiftKey || Math.abs(dx) > Math.abs(dy)) {
      const msPerPx = (S.view[1] - S.view[0]) / canvas.width;
      const d = (Math.abs(dx) > Math.abs(dy) ? dx : dy) * msPerPx;
      setView(S.view[0] + d, S.view[1] + d);
    } else {
      zoomAt(msOf(e.clientX - r.left), Math.exp(dy * WHEEL_ZOOM));
    }
  }, { passive: false });
  if (windowListenersAttached) return;
  windowListenersAttached = true;
  window.addEventListener('mousemove', (e) => {
    if (!S || !S.drag) return;
    const r = S.canvas.getBoundingClientRect();
    const rawX = e.clientX - r.left;
    if (Math.abs(rawX - S.drag.x) < 3 && !S.drag.moved) return;
    S.drag.moved = true;
    if (!S.drag.handle) {
      // Pan: the point under the mouse stays under the mouse.
      const [v0, v1] = S.drag.view;
      const d = ((rawX - S.drag.x) / S.canvas.width) * (v1 - v0);
      setView(v0 - d, v1 - d);
      return;
    }
    const x = Math.max(0, Math.min(S.canvas.width, rawX));
    applyDrag(S.drag.handle, msOf(x));
    draw();
  });
  window.addEventListener('mouseup', () => {
    if (!S || !S.drag) return;
    const { handle, moved, x, y } = S.drag;
    S.drag = null;
    S.canvas.style.cursor = '';
    if (moved && !handle) return;   // a pan
    if (!moved) {
      const seg = hitSegment(x, y);
      if (seg) play(seg.start_ms, seg.end_ms);
      return;
    }
    const seg = handle.tier === 'word' ? S.data.words[handle.idx] : S.data.phonemes[handle.idx];
    const ms = handle.edge === 'start' ? seg.start_ms : seg.end_ms;
    S.pins = S.pins.filter((p) => !(p.tier === handle.tier && p.idx === handle.idx && p.edge === handle.edge));
    S.pins.push({ ...handle, ms });
    loadSentence(S.idx, { pins: S.pins });
  });
  window.addEventListener('resize', () => S && S.data && draw());
}

async function toggleExclude() {
  const sent = S.rec.sentences[S.idx];
  try {
    const { sentence } = await postJson(
      `/api/recordings/${encodeURIComponent(S.rec.id)}/sentences/${S.idx}/exclude`,
      { excluded: sent.status !== 'excluded' });
    S.rec.sentences[S.idx] = sentence;
    deps.setStatus(sentence.status === 'excluded'
      ? `Sentence ${S.idx + 1} excluded — its copies were removed`
      : `Sentence ${S.idx + 1} included again — save it to cut copies`);
    await deps.refreshInventory();
  } catch (e) {
    deps.setStatus(`Exclude: ${e.message}`, true);
  }
  render();
}

async function saveSentence() {
  if (!S || !S.data) return;
  const words = S.data.words.map((w) => ({
    label: w.label, start_ms: w.start_ms, end_ms: w.end_ms, edited: S.edited.has(`w:${w.idx}`),
  }));
  const phonemes = S.data.phonemes.map((p) => ({
    label: p.label, start_ms: p.start_ms, end_ms: p.end_ms, parent_idx: p.parent_idx,
    edited: S.edited.has(`p:${p.idx}`),
  }));
  try {
    const res = await postJson(
      `/api/recordings/${encodeURIComponent(S.rec.id)}/sentences/${S.idx}/save`, { words, phonemes });
    S.rec.sentences[S.idx] = res.sentence;
    deps.setStatus(`Saved sentence ${S.idx + 1}: ${res.saved_units} copies`);
    await deps.refreshInventory();
    const nextTodo = S.rec.sentences.findIndex((s, i) => i > S.idx && s.status === 'proposed');
    if (nextTodo >= 0) { await loadSentence(nextTodo, {}); return; }
  } catch (e) {
    deps.setStatus(`Save: ${e.message}`, true);
  }
  render();
}
