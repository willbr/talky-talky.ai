import { initSplitter, openSplitter, closeSplitter } from './splitter.js';
import { initMeter, startMeter, finishMeter } from './meter.js';

const SAMPLE_RATE = 22050;

const els = {
  recordText: document.getElementById('record-text'),
  takeHint: document.getElementById('take-hint'),
  bookLength: document.getElementById('book-length'),
  splitterMount: document.getElementById('splitter-mount'),
  recordBtn: document.getElementById('record-btn'),
  saveBtn: document.getElementById('save-btn'),
  discardBtn: document.getElementById('discard-btn'),
  preview: document.getElementById('preview'),
  recordMeter: document.getElementById('record-meter'),
  recordMeterLabel: document.getElementById('record-meter-label'),
  recordStatus: document.getElementById('record-status'),
  inventoryChips: document.getElementById('inventory-chips'),
  sentenceInput: document.getElementById('sentence-input'),
  synthBtn: document.getElementById('synth-btn'),
  output: document.getElementById('output'),
  skippedList: document.getElementById('skipped-list'),
  phonemeGrid: document.getElementById('phoneme-grid'),
  recordPane: document.getElementById('record-pane'),
  statusStrip: document.getElementById('status-strip'),
  tabBtns: document.querySelectorAll('.tab-btn'),
  tabPanes: document.querySelectorAll('.tab-pane'),
  bookSelect: document.getElementById('book-select'),
  pickSentenceBtn: document.getElementById('pick-sentence-btn'),
  phonemesOnlyToggle: document.getElementById('phonemes-only-toggle'),
  renderedWords: document.getElementById('rendered-words'),
  diphoneStats: document.getElementById('diphone-stats'),
  compareBtn: document.getElementById('compare-btn'),
  compareOutput: document.getElementById('compare-output'),
  compareOutputNormal: document.getElementById('compare-output-normal'),
  compareOutputPhonemes: document.getElementById('compare-output-phonemes'),
  outputSpec: document.getElementById('output-spec'),
  compareOutputNormalSpec: document.getElementById('compare-output-normal-spec'),
  compareOutputPhonemesSpec: document.getElementById('compare-output-phonemes-spec'),
  referenceOutput: document.getElementById('reference-output'),
  referenceOutputAudio: document.getElementById('reference-output-audio'),
  referenceOutputSpec: document.getElementById('reference-output-spec'),
};

const PHONEMES_ONLY_KEY = 'tt.phonemesOnly';
els.phonemesOnlyToggle.checked = localStorage.getItem(PHONEMES_ONLY_KEY) === '1';
els.phonemesOnlyToggle.addEventListener('change', () => {
  localStorage.setItem(PHONEMES_ONLY_KEY, els.phonemesOnlyToggle.checked ? '1' : '0');
});

let wordCount = 0;
let diphoneCount = 0;
let recordedPhonemeCount = 0;
let totalPhonemeCount = 0;

function updateStatusStrip() {
  els.statusStrip.textContent =
    `${wordCount} word${wordCount === 1 ? '' : 's'} · ${recordedPhonemeCount}/${totalPhonemeCount} phonemes`
    + ` · ${diphoneCount} diphone${diphoneCount === 1 ? '' : 's'}`;
}

function activateTab(tabId) {
  for (const b of els.tabBtns) b.classList.toggle('active', b.dataset.tab === tabId);
  for (const p of els.tabPanes) p.classList.toggle('active', p.id === tabId);
}

for (const btn of els.tabBtns) {
  btn.addEventListener('click', () => activateTab(btn.dataset.tab));
}

let audioCtx = null;
let workletNode = null;
let mediaStream = null;
let currentWavBlob = null;
let currentPhoneme = null;   // null => recording for a word; non-null => recording for this phoneme
let isRecording = false;

// Phoneme editor state — null when closed.
let editor = null;

async function ensureAudio() {
  if (audioCtx) return;
  audioCtx = new AudioContext({ sampleRate: SAMPLE_RATE });
  await audioCtx.audioWorklet.addModule('/static/recorder-worklet.js');
}

async function startRecording() {
  await ensureAudio();
  // A context created or left idle without a user gesture can sit
  // suspended, and a suspended context captures nothing.
  if (audioCtx.state !== 'running') await audioCtx.resume();
  mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });
  const source = audioCtx.createMediaStreamSource(mediaStream);
  workletNode = new AudioWorkletNode(audioCtx, 'recorder-processor');
  source.connect(workletNode);
  workletNode.port.postMessage('start');
  els.preview.removeAttribute('src');
  startMeter(audioCtx, source, mediaStream.getAudioTracks()[0]);
  setStatus('recording…');
  setButtons({ recording: true, hasWav: false });
}

function stopRecording() {
  return new Promise((resolve) => {
    workletNode.port.onmessage = (e) => {
      const float32 = e.data;
      mediaStream.getTracks().forEach((t) => t.stop());
      workletNode.disconnect();
      mediaStream = null;
      workletNode = null;
      resolve(float32);
    };
    workletNode.port.postMessage('stop');
  });
}

function floatsToInt16(floats) {
  const out = new Int16Array(floats.length);
  for (let i = 0; i < floats.length; i++) {
    let s = Math.max(-1, Math.min(1, floats[i]));
    out[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
  }
  return out;
}

function encodeWav(int16, sampleRate) {
  const numChannels = 1;
  const bytesPerSample = 2;
  const blockAlign = numChannels * bytesPerSample;
  const byteRate = sampleRate * blockAlign;
  const dataSize = int16.length * bytesPerSample;
  const buffer = new ArrayBuffer(44 + dataSize);
  const view = new DataView(buffer);

  const writeString = (offset, str) => {
    for (let i = 0; i < str.length; i++) view.setUint8(offset + i, str.charCodeAt(i));
  };

  writeString(0, 'RIFF');
  view.setUint32(4, 36 + dataSize, true);
  writeString(8, 'WAVE');
  writeString(12, 'fmt ');
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true); // PCM
  view.setUint16(22, numChannels, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, byteRate, true);
  view.setUint16(32, blockAlign, true);
  view.setUint16(34, 16, true); // bits per sample
  writeString(36, 'data');
  view.setUint32(40, dataSize, true);

  let offset = 44;
  for (let i = 0; i < int16.length; i++, offset += 2) view.setInt16(offset, int16[i], true);

  return new Blob([buffer], { type: 'audio/wav' });
}

function setStatus(text, isError = false) {
  els.recordStatus.textContent = text;
  els.recordStatus.style.color = isError ? 'var(--error)' : 'var(--muted)';
}

function setButtons({ recording, hasWav }) {
  els.recordBtn.textContent = recording ? '■ Stop' : '● Record';
  els.recordBtn.disabled = false;
  els.saveBtn.disabled = !hasWav;
  els.discardBtn.disabled = !hasWav;
}

async function refreshInventory() {
  await Promise.all([refreshWordInventory(), refreshPhonemeInventory()]);
}

async function refreshWordInventory() {
  const r = await fetch('/api/inventory');
  const { words, diphones } = await r.json();
  wordCount = words.length;
  diphoneCount = diphones || 0;
  updateStatusStrip();
  els.inventoryChips.innerHTML = '';
  for (const w of words) {
    const chip = document.createElement('span');
    chip.className = 'chip';
    const label = document.createElement('span');
    label.textContent = w;
    chip.appendChild(label);

    const edit = document.createElement('button');
    edit.className = 'chip-edit';
    edit.textContent = '✎';
    edit.title = `"${w}" copies`;
    edit.addEventListener('click', async (e) => {
      e.stopPropagation();
      await toggleUnitsPanel(chip, 'word', w);
    });
    chip.appendChild(edit);

    els.inventoryChips.appendChild(chip);
  }
}

async function refreshPhonemeInventory() {
  const r = await fetch('/api/phoneme-inventory');
  const { phonemes } = await r.json();
  totalPhonemeCount = phonemes.length;
  recordedPhonemeCount = phonemes.filter((p) => p.recorded).length;
  updateStatusStrip();
  els.phonemeGrid.innerHTML = '';
  for (const { symbol, example, recorded } of phonemes) {
    const chip = document.createElement('span');
    chip.className = 'phoneme-chip' + (recorded ? ' recorded' : '');
    chip.dataset.symbol = symbol;
    chip.dataset.example = example;

    const symEl = document.createElement('span');
    symEl.className = 'phoneme-symbol';
    symEl.textContent = symbol;

    const exEl = document.createElement('span');
    exEl.className = 'phoneme-example';
    exEl.textContent = example;

    chip.appendChild(symEl);
    chip.appendChild(document.createTextNode(' '));
    chip.appendChild(exEl);

    chip.addEventListener('click', () => armPhoneme(symbol, example));

    if (recorded) {
      const edit = document.createElement('button');
      edit.className = 'chip-edit';
      edit.textContent = '✎';
      edit.title = `${symbol} copies`;
      edit.addEventListener('click', async (e) => {
        e.stopPropagation();
        await toggleUnitsPanel(chip, 'phoneme', symbol);
      });
      chip.appendChild(edit);
    }

    if (currentPhoneme === symbol) chip.classList.add('armed');
    els.phonemeGrid.appendChild(chip);
  }
}

function describeUnit(u) {
  const bits = [];
  if (u.kind === 'word') {
    bits.push(u.source === 'sentence' ? 'from sentence' : u.source);
  } else {
    const from = u.source_word || 'word';
    bits.push(u.source === 'isolated' ? 'isolated' : `from "${from}"`);
  }
  const ctx = [];
  if (u.prev) ctx.push(`prev=${u.prev}`);
  if (u.next) ctx.push(`next=${u.next}`);
  if (u.position) ctx.push(u.position);
  if (ctx.length) bits.push(`(${ctx.join(' ')})`);
  return `${bits.join(' ')} · ${u.duration_ms} ms`;
}

// Toggle the inline copies panel under a word or phoneme chip. Only one
// panel is open at a time across both grids.
async function toggleUnitsPanel(chip, kind, label) {
  const existing = chip.querySelector('.variants-panel');
  for (const open of document.querySelectorAll('.variants-panel')) {
    open.parentElement.classList.remove('expanded');
    open.remove();
  }
  if (existing) return;   // we were toggling it off

  let body;
  try {
    const r = await fetch(`/api/units?kind=${kind}&label=${encodeURIComponent(label)}`);
    if (!r.ok) {
      setStatus(`Copies: ${r.statusText}`, true);
      return;
    }
    body = await r.json();
  } catch (e) {
    setStatus(`Copies failed: ${e.message}`, true);
    return;
  }

  chip.classList.add('expanded');
  const panel = document.createElement('div');
  panel.className = 'variants-panel';
  panel.addEventListener('click', (e) => e.stopPropagation());

  if (!body.units.length) {
    const empty = document.createElement('div');
    empty.className = 'variant-empty';
    empty.textContent = 'no copies';
    panel.appendChild(empty);
    chip.appendChild(panel);
    return;
  }

  for (const u of body.units) {
    const row = document.createElement('div');
    row.className = 'variant-row';

    const meta = document.createElement('span');
    meta.className = 'variant-meta';
    meta.textContent = describeUnit(u);
    meta.title = `${u.id} · ${meta.textContent}`;
    row.appendChild(meta);

    const action = (text, title, cls, fn) => {
      const b = document.createElement('button');
      b.className = `variant-action${cls ? ' ' + cls : ''}`;
      b.textContent = text;
      b.title = title;
      b.addEventListener('click', (e) => { e.stopPropagation(); fn(); });
      row.appendChild(b);
    };

    action('▶', 'Play', '', () => playUnit(u.id));
    if (u.source === 'sentence') {
      action('✎', 'Open the sentence it was cut from in the splitter', '', () => {
        openSplitter(u.source_recording_id, u.source_sentence_idx);
      });
    } else {
      action('✎', kind === 'word' ? 'Trim / split into phonemes' : 'Trim', '', () => {
        if (kind === 'word') openExtractEditor(label, u.id);
        else openPhonemeEditor(label, u.id);
      });
    }
    action('×', 'Delete', 'delete', async () => {
      if (!window.confirm(`Delete this copy of "${label}"?`)) return;
      const r = await fetch(`/api/units/${encodeURIComponent(u.id)}`, { method: 'DELETE' });
      if (!r.ok) {
        const err = await r.json().catch(() => ({}));
        setStatus(`Delete: ${err.error || r.statusText}`, true);
        return;
      }
      await refreshInventory();
    });

    panel.appendChild(row);
  }
  chip.appendChild(panel);
}

let unitPlayer = null;

function playUnit(unitId) {
  if (unitPlayer) unitPlayer.pause();
  unitPlayer = new Audio(`/api/units/${encodeURIComponent(unitId)}/audio`);
  unitPlayer.play().catch((e) => setStatus(`Play failed: ${e.message}`, true));
}

function armPhoneme(symbol, example) {
  currentPhoneme = symbol;
  els.recordText.value = '';   // clear any half-typed word
  const hint = example ? ` — as in "${example}"` : '';
  setStatus(`Recording phoneme ${symbol}${hint} — press Record`);
  // Re-paint chips so the new selection is highlighted.
  for (const chip of els.phonemeGrid.querySelectorAll('.phoneme-chip')) {
    chip.classList.toggle('armed', chip.dataset.symbol === symbol);
  }
}

function disarmPhoneme() {
  currentPhoneme = null;
  for (const chip of els.phonemeGrid.querySelectorAll('.phoneme-chip')) {
    chip.classList.remove('armed');
  }
}

els.recordBtn.addEventListener('click', async () => {
  if (isRecording) {
    isRecording = false;
    const floats = await stopRecording();
    const check = finishMeter(floats);
    const int16 = floatsToInt16(floats);
    currentWavBlob = encodeWav(int16, SAMPLE_RATE);
    els.preview.src = URL.createObjectURL(currentWavBlob);
    setStatus(check.message, check.verdict !== 'ok');
    setButtons({ recording: false, hasWav: true });
    return;
  }
  try {
    await startRecording();
    isRecording = true;
  } catch (err) {
    isRecording = false;
    setStatus(`mic unavailable: ${err.message || err.name}`, true);
    setButtons({ recording: false, hasWav: false });
  }
});

els.discardBtn.addEventListener('click', () => {
  currentWavBlob = null;
  els.preview.removeAttribute('src');
  setStatus('discarded');
  setButtons({ recording: false, hasWav: false });
});

async function saveWord(word, blob) {
  const form = new FormData();
  form.append('word', word);
  form.append('audio', blob, 'audio.wav');

  // Every save appends a new copy; nothing is ever overwritten.
  const r = await fetch('/api/record', { method: 'POST', body: form });
  if (!r.ok) {
    const err = await r.json().catch(() => ({}));
    setStatus(`save failed: ${err.error || r.statusText}`, true);
    return;
  }
  const { word: saved, unit_id: unitId } = await r.json();
  setStatus(`saved "${saved}"`);
  els.recordText.value = '';

  // After a successful word save, automatically offer phoneme extraction
  // on the copy that was just recorded.
  openExtractEditor(saved, unitId);
}

function wordCountOf(text) {
  return text.split(/\s+/).filter((t) => /[\p{L}\p{N}]/u.test(t)).length;
}

// A sentence or passage: stored whole, then split in the splitter.
async function saveRecording(text, blob) {
  const form = new FormData();
  form.append('text', text);
  form.append('audio', blob, 'audio.wav');
  let r;
  try {
    r = await fetch('/api/recordings', { method: 'POST', body: form });
  } catch (e) {
    setStatus(`save failed: ${e.message}`, true);
    return false;
  }
  if (!r.ok) {
    const err = await r.json().catch(() => ({}));
    setStatus(`save failed: ${err.error || r.statusText}`, true);
    return false;
  }
  const { id, sentences } = await r.json();
  setStatus(`saved recording — ${sentences.length} sentence${sentences.length === 1 ? '' : 's'} to check`);
  defaultSentence = text;
  updateTakeHint();
  await openSplitter(id, 0);
  return true;
}

async function savePhoneme(symbol, blob) {
  const form = new FormData();
  form.append('phoneme', symbol);
  form.append('audio', blob, 'audio.wav');

  const r = await fetch('/api/record/phoneme', { method: 'POST', body: form });
  if (!r.ok) {
    const err = await r.json().catch(() => ({}));
    setStatus(`save failed: ${err.error || r.statusText}`, true);
    return;
  }
  const { phoneme: saved } = await r.json();
  setStatus(`saved phoneme ${saved}`);
  disarmPhoneme();
}

els.saveBtn.addEventListener('click', async () => {
  if (!currentWavBlob) { setStatus('record something first', true); return; }

  if (currentPhoneme) {
    await savePhoneme(currentPhoneme, currentWavBlob);
  } else {
    const text = els.recordText.value.trim();
    if (!text) { setStatus('type what you said first', true); return; }
    if (wordCountOf(text) === 1) await saveWord(text, currentWavBlob);
    else if (!(await saveRecording(text, currentWavBlob))) return;
  }

  currentWavBlob = null;
  els.preview.removeAttribute('src');
  setButtons({ recording: false, hasWav: false });
  await refreshInventory();
});

async function postSynth(sentence, phonemesOnly) {
  const r = await fetch('/api/synthesize', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ sentence, phonemes_only: phonemesOnly }),
  });
  if (!r.ok) {
    const err = await r.json().catch(() => ({}));
    throw new Error(err.error || r.statusText);
  }
  return r.json();
}

function setAudioFromBase64(audioEl, b64) {
  if (!b64) {
    audioEl.removeAttribute('src');
    return;
  }
  const bin = atob(b64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  const blob = new Blob([bytes], { type: 'audio/wav' });
  audioEl.src = URL.createObjectURL(blob);
}

async function synthesise() {
  ensureSentence();
  const sentence = els.sentenceInput.value.trim();
  if (!sentence) return;

  // Single-output mode hides the comparison row.
  els.compareOutput.hidden = true;
  els.output.hidden = false;

  let result;
  try {
    result = await postSynth(sentence, els.phonemesOnlyToggle.checked);
  } catch (e) {
    els.skippedList.textContent = `synth failed: ${e.message}`;
    return;
  }
  const { audio: b64, skipped, units, spectrogram } = result;
  showDiphoneStats([['', result.diphones]]);

  setAudioFromBase64(els.output, b64);
  drawSpectrogram(els.outputSpec, els.output, spectrogram, units);
  if (b64) {
    els.output.play().catch(() => { /* user gesture is the click on synth */ });
  }

  renderRenderedWords(units);
  renderSkipped(skipped);
  renderReferenceForSentence(sentence);
}

// "Joins: 12 of 30 from real diphones" — how much of the sub-word audio
// came from recorded transitions rather than glued phoneme halves.
function showDiphoneStats(rows) {
  const parts = [];
  for (const [name, d] of rows) {
    if (!d) continue;
    const total = d.real + d.fallback;
    if (!total) continue;
    parts.push(`${name}${d.real} of ${total} joins from recorded diphones`);
  }
  els.diphoneStats.textContent = parts.join(' · ');
  els.diphoneStats.hidden = parts.length === 0;
}

async function compareSynth() {
  ensureSentence();
  const sentence = els.sentenceInput.value.trim();
  if (!sentence) return;

  // Comparison mode hides the single output player and its spectrogram.
  els.output.hidden = true;
  els.outputSpec.hidden = true;
  els.compareOutput.hidden = false;

  let normal;
  let phonemes;
  try {
    [normal, phonemes] = await Promise.all([
      postSynth(sentence, false),
      postSynth(sentence, true),
    ]);
  } catch (e) {
    els.skippedList.textContent = `synth failed: ${e.message}`;
    return;
  }

  showDiphoneStats([['Normal: ', normal.diphones], ['No word copies: ', phonemes.diphones]]);
  setAudioFromBase64(els.compareOutputNormal, normal.audio);
  setAudioFromBase64(els.compareOutputPhonemes, phonemes.audio);
  drawSpectrogram(els.compareOutputNormalSpec, els.compareOutputNormal, normal.spectrogram, normal.units);
  drawSpectrogram(els.compareOutputPhonemesSpec, els.compareOutputPhonemes, phonemes.spectrogram, phonemes.units);

  // Edit chips come from the normal-mode render: those are the word copies
  // that were actually heard (phonemes-only mode uses none).
  renderRenderedWords(normal.units);
  // Skip chips: union of both modes (anything that failed somewhere).
  const skippedUnion = Array.from(new Set([...normal.skipped, ...phonemes.skipped]));
  renderSkipped(skippedUnion);
  renderReferenceForSentence(sentence);
}

async function renderReferenceForSentence(sentence) {
  let result;
  try {
    const r = await fetch('/api/sentence/audio', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ sentence }),
    });
    if (!r.ok) {
      els.referenceOutput.hidden = true;
      return;
    }
    result = await r.json();
  } catch (e) {
    els.referenceOutput.hidden = true;
    return;
  }
  if (!result.found) {
    els.referenceOutput.hidden = true;
    return;
  }
  els.referenceOutput.hidden = false;
  setAudioFromBase64(els.referenceOutputAudio, result.audio);
  // Reference recordings have no word-level annotation — pass an empty
  // units list so the spectrogram renders without label decorations.
  drawSpectrogram(els.referenceOutputSpec, els.referenceOutputAudio, result.spectrogram, []);
}

function drawSpectrogram(canvas, audioEl, spec, units) {
  if (!canvas) return;
  if (!spec || !spec.n_frames || !spec.duration_ms) {
    canvas.hidden = true;
    canvas.onclick = null;
    return;
  }
  canvas.hidden = false;

  canvas.onclick = (e) => {
    if (!audioEl || !audioEl.src) return;
    const rect = canvas.getBoundingClientRect();
    const fraction = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
    audioEl.currentTime = (fraction * spec.duration_ms) / 1000;
    audioEl.play().catch(() => { /* user gesture is the click */ });
  };

  const cssW = canvas.clientWidth || 600;
  if (canvas.width !== cssW) canvas.width = cssW;
  const w = canvas.width;
  const h = canvas.height;
  const wordLabelH = 16;
  const phonemeLabelH = 14;
  const hasUnits = !!(units && units.length);
  const hasPhonemes = hasUnits && units.some((u) => u.phonemes && u.phonemes.length);
  const labelH = hasUnits ? wordLabelH + (hasPhonemes ? phonemeLabelH : 0) : 0;
  const specH = h - labelH;

  const bin = atob(spec.data);
  const mag = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) mag[i] = bin.charCodeAt(i);

  const nFrames = spec.n_frames;
  const nFreqs = spec.n_freqs;

  const ctx = canvas.getContext('2d');
  ctx.fillStyle = '#1a1b1e';
  ctx.fillRect(0, 0, w, h);

  const img = ctx.createImageData(w, specH);
  const pixels = img.data;
  for (let x = 0; x < w; x++) {
    const frame = Math.min(nFrames - 1, Math.floor((x / w) * nFrames));
    const frameBase = frame * nFreqs;
    for (let y = 0; y < specH; y++) {
      // y=0 is top → high freq; y=specH-1 is bottom → low freq.
      const freq = Math.min(nFreqs - 1, Math.floor(((specH - 1 - y) / specH) * nFreqs));
      const v = mag[frameBase + freq];
      const i = (y * w + x) * 4;
      pixels[i] = (v * 143) >> 8;       // r ramps to ~143 (#8f)
      pixels[i + 1] = (v * 223) >> 8;   // g ramps to ~223 (#df)
      pixels[i + 2] = (v * 155) >> 8;   // b ramps to ~155 (#9b)
      pixels[i + 3] = 255;
    }
  }
  ctx.putImageData(img, 0, labelH);

  if (!units || !units.length) return;
  const duration = spec.duration_ms;

  // Two-band label strip: word labels on top, phoneme labels below.
  ctx.fillStyle = 'rgba(0, 0, 0, 0.45)';
  ctx.fillRect(0, 0, w, labelH);
  ctx.textBaseline = 'middle';
  ctx.textAlign = 'center';

  // Phoneme boundaries: thin dashed lines spanning the full canvas, drawn first
  // so word boundaries (solid, brighter) sit on top of them at coincident edges.
  ctx.save();
  ctx.strokeStyle = 'rgba(255, 255, 255, 0.25)';
  ctx.lineWidth = 1;
  ctx.setLineDash([3, 3]);
  ctx.font = '10px ui-monospace, "Cascadia Mono", monospace';
  ctx.fillStyle = '#b8c0c8';
  for (const u of units) {
    if (!u.phonemes || !u.phonemes.length) continue;
    for (let j = 1; j < u.phonemes.length; j++) {
      const ph = u.phonemes[j];
      const x = (ph.start_ms / duration) * w;
      ctx.beginPath();
      ctx.moveTo(x + 0.5, wordLabelH);
      ctx.lineTo(x + 0.5, h);
      ctx.stroke();
    }
    for (const ph of u.phonemes) {
      const xc = ((ph.start_ms + ph.end_ms) / 2 / duration) * w;
      ctx.fillText(ph.symbol, xc, wordLabelH + phonemeLabelH / 2);
    }
  }
  ctx.restore();

  // Word boundaries: solid, brighter, and word labels in the top band.
  ctx.strokeStyle = 'rgba(255, 255, 255, 0.55)';
  ctx.lineWidth = 1;
  ctx.font = '11px ui-monospace, "Cascadia Mono", monospace';
  ctx.fillStyle = '#e6e6e6';
  for (const u of units) {
    const x0 = (u.start_ms / duration) * w;
    const x1 = (u.end_ms / duration) * w;
    ctx.beginPath();
    ctx.moveTo(x0 + 0.5, 0);
    ctx.lineTo(x0 + 0.5, h);
    ctx.stroke();
    ctx.fillText(u.token, (x0 + x1) / 2, wordLabelH / 2);
  }
  const xLast = (units[units.length - 1].end_ms / duration) * w;
  ctx.beginPath();
  ctx.moveTo(xLast - 0.5, 0);
  ctx.lineTo(xLast - 0.5, h);
  ctx.stroke();
}

// One chip per word occurrence that was rendered from a word copy, in
// sentence order. Each opens the editor on exactly the copy that was heard.
function renderRenderedWords(units) {
  els.renderedWords.innerHTML = '';
  const heard = (units || []).filter((u) => u.unit_id);
  if (!heard.length) return;

  const label = document.createElement('span');
  label.className = 'rendered-label';
  label.textContent = `Edit (${heard.length}):`;
  els.renderedWords.appendChild(label);

  for (const u of heard) {
    const chip = document.createElement('span');
    chip.className = 'chip rendered';

    const lbl = document.createElement('span');
    lbl.textContent = u.token;
    chip.appendChild(lbl);

    const edit = document.createElement('button');
    edit.className = 'chip-edit';
    edit.textContent = '✎';
    const fromSentence = u.source === 'sentence';
    edit.title = fromSentence
      ? `Open the sentence this "${u.token}" was cut from`
      : `Edit this copy of "${u.token}"`;
    edit.addEventListener('click', (e) => {
      e.stopPropagation();
      activateTab('train-tab');
      if (fromSentence) openSplitter(u.source_recording_id, u.source_sentence_idx);
      else openExtractEditor(u.token, u.unit_id);
    });
    chip.appendChild(edit);

    els.renderedWords.appendChild(chip);
  }
}

function renderSkipped(skipped) {
  els.skippedList.innerHTML = '';
  if (!skipped.length) return;

  const label = document.createElement('span');
  label.className = 'skipped-label';
  label.textContent = `Skipped — click to record (${skipped.length}):`;
  els.skippedList.appendChild(label);

  for (const w of skipped) {
    const chip = document.createElement('span');
    chip.className = 'chip missing';
    chip.textContent = w;
    chip.addEventListener('click', () => {
      activateTab('train-tab');
      els.recordText.value = w;
      els.recordText.focus();
    });
    els.skippedList.appendChild(chip);
  }
}

els.synthBtn.addEventListener('click', synthesise);
els.compareBtn.addEventListener('click', compareSynth);

els.recordText.addEventListener('input', () => {
  if (currentPhoneme && els.recordText.value.trim()) disarmPhoneme();
  clearTimeout(takeHintTimer);
  takeHintTimer = setTimeout(updateTakeHint, 400);
});

// "Open recording" when the text already has takes (or is a sentence of one).
let takeHintTimer = null;
let takeHintTarget = null;

async function updateTakeHint() {
  const text = els.recordText.value.trim();
  takeHintTarget = null;
  els.takeHint.hidden = true;
  if (wordCountOf(text) < 2) return;
  let matches;
  try {
    const r = await fetch(`/api/recordings/find?text=${encodeURIComponent(text)}`);
    ({ matches } = await r.json());
  } catch (e) { return; }
  if (text !== els.recordText.value.trim() || !matches.length) return;
  takeHintTarget = matches[0];
  els.takeHint.textContent = `Open recording (${matches.length} take${matches.length === 1 ? '' : 's'})`;
  els.takeHint.hidden = false;
}

els.takeHint.addEventListener('click', () => {
  if (takeHintTarget) openSplitter(takeHintTarget.recording_id, takeHintTarget.sentence_idx ?? 0);
});

async function refreshBooks() {
  const r = await fetch('/api/books');
  const { books } = await r.json();
  els.bookSelect.innerHTML = '';
  const any = document.createElement('option');
  any.value = 'any';
  any.textContent = 'any';
  els.bookSelect.appendChild(any);
  for (const b of books) {
    const opt = document.createElement('option');
    opt.value = b;
    opt.textContent = b.replace(/-/g, ' ');
    els.bookSelect.appendChild(opt);
  }
  els.pickSentenceBtn.disabled = books.length === 0;
}

els.pickSentenceBtn.addEventListener('click', async () => {
  const r = await fetch('/api/books/sample', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ book: els.bookSelect.value, length: els.bookLength.value }),
  });
  if (!r.ok) return;
  const picked = await r.json();
  els.recordText.value = picked.sentence;
  disarmPhoneme();
  updateTakeHint();
  if (picked.new_diphones !== undefined) {
    setStatus(`this passage adds ${picked.new_diphones} diphone${picked.new_diphones === 1 ? '' : 's'}`
      + ` you don't have yet (you have ${picked.have})`);
  }
});

// The Speak box always starts with a sentence: the most recent reference
// recording (so synth vs. reference is one click away), else a fallback.
const FALLBACK_SENTENCE = 'hello world';
let defaultSentence = FALLBACK_SENTENCE;

async function loadDefaultSentence() {
  try {
    const r = await fetch('/api/recordings');
    const { recordings } = await r.json();
    if (recordings.length) defaultSentence = recordings[recordings.length - 1].text;
  } catch (e) { /* keep the fallback */ }
  ensureSentence();
}

function ensureSentence() {
  if (!els.sentenceInput.value.trim()) els.sentenceInput.value = defaultSentence;
}

initMeter(els.recordMeter, els.recordMeterLabel);

initSplitter({
  mount: els.splitterMount,
  setStatus,
  refreshInventory,
  beforeOpen: () => { closeExtractEditor(); activateTab('train-tab'); },
});

refreshInventory();
refreshBooks();
loadDefaultSentence();
setButtons({ recording: false, hasWav: false });

async function openExtractEditor(word, unitId = null) {
  closeExtractEditor();
  closeSplitter();

  let prop;
  try {
    const r = await fetch('/api/extract/propose', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(unitId ? { unit_id: unitId } : { word }),
    });
    if (!r.ok) {
      const err = await r.json().catch(() => ({}));
      setStatus(`Extract: ${err.error || r.statusText}`, true);
      return;
    }
    prop = await r.json();
  } catch (e) {
    setStatus(`Extract failed: ${e.message}`, true);
    return;
  }

  editor = {
    mode: 'word',
    word: prop.word,
    unitId: prop.unit_id,
    audioBase64: prop.audio,
    audioDurationMs: prop.audio_duration_ms,
    phonemes: prop.phonemes,
    phonemeSpansMs: prop.spans.map((s) => ({ start_ms: s.start_ms, end_ms: s.end_ms })),
    audioBuffer: null,
    audioCtx: null,
    container: null,
    canvas: null,
    samples: null,
    draggingHandle: null,   // null | { phonemeIdx, kind: 'start' | 'end' }
    viewStartMs: 0,
    viewEndMs: prop.audio_duration_ms,
    trimBtn: null,
    trimSaveBtn: null,
  };
  applyTrim();

  buildEditorDom();
  decodeWaveform();
}

async function openPhonemeEditor(symbol, unitId = null) {
  closeExtractEditor();
  closeSplitter();

  let prop;
  try {
    const payload = unitId ? { symbol, unit_id: unitId } : { symbol };
    const r = await fetch('/api/phoneme/propose', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    if (!r.ok) {
      const err = await r.json().catch(() => ({}));
      setStatus(`Phoneme edit: ${err.error || r.statusText}`, true);
      return;
    }
    prop = await r.json();
  } catch (e) {
    setStatus(`Phoneme edit failed: ${e.message}`, true);
    return;
  }

  // Reuse the editor scaffolding with a single-span "phoneme" sitting
  // across the full recording. The user drags the two outer handles to
  // mark the trim range and clicks Save trim.
  editor = {
    mode: 'phoneme',
    phonemeSymbol: prop.symbol,
    unitId: prop.unit_id,
    phonemeExample: prop.example,
    audioBase64: prop.audio,
    audioDurationMs: prop.audio_duration_ms,
    phonemes: [],
    phonemeSpansMs: [{ start_ms: 0, end_ms: prop.audio_duration_ms }],
    audioBuffer: null,
    audioCtx: null,
    container: null,
    canvas: null,
    samples: null,
    draggingHandle: null,
    viewStartMs: 0,
    viewEndMs: prop.audio_duration_ms,
    trimBtn: null,
    trimSaveBtn: null,
  };
  buildEditorDom();
  decodeWaveform();
}

function applyTrim() {
  if (!editor) return;
  const spans = editor.phonemeSpansMs;
  const first = spans[0].start_ms;
  const last = spans[spans.length - 1].end_ms;
  const span = Math.max(20, last - first);
  const margin = Math.round(span * 0.05);
  editor.viewStartMs = Math.max(0, first - margin);
  editor.viewEndMs = Math.min(editor.audioDurationMs, last + margin);
}

function closeExtractEditor() {
  if (!editor) return;
  if (editor.container && editor.container.parentNode) {
    editor.container.parentNode.removeChild(editor.container);
  }
  editor = null;
}

function buildEditorDom() {
  const container = document.createElement('div');
  container.id = 'phoneme-editor';

  const header = document.createElement('div');
  header.className = 'editor-header';
  const title = document.createElement('h3');
  if (editor.mode === 'phoneme') {
    const ex = editor.phonemeExample ? ` (${editor.phonemeExample})` : '';
    title.textContent = `Trimming phoneme "${editor.phonemeSymbol}"${ex}`;
  } else {
    title.textContent = `Extracting phonemes from "${editor.word}"`;
  }

  const playAllBtn = document.createElement('button');
  playAllBtn.className = 'play-all';
  playAllBtn.textContent = '▶ Play';
  playAllBtn.title = 'Play the trimmed selection (between the outer boundaries)';
  playAllBtn.addEventListener('click', () => {
    const spans = editor.phonemeSpansMs;
    playSlice(spans[0].start_ms, spans[spans.length - 1].end_ms);
  });

  const trimBtn = document.createElement('button');
  trimBtn.className = 'trim';
  trimBtn.textContent = 'Fit';
  trimBtn.title = 'Toggle between trimmed-to-selection view and the full recording';
  trimBtn.addEventListener('click', toggleTrim);
  editor.trimBtn = trimBtn;

  const dismiss = document.createElement('button');
  dismiss.className = 'dismiss';
  dismiss.textContent = '×';
  dismiss.addEventListener('click', closeExtractEditor);

  const controls = document.createElement('div');
  controls.className = 'editor-controls';
  controls.appendChild(playAllBtn);
  controls.appendChild(trimBtn);
  if (editor.mode !== 'phoneme') {
    const recalcBtn = document.createElement('button');
    recalcBtn.className = 'recalc';
    recalcBtn.textContent = 'Recalc';
    recalcBtn.title = 'Redistribute internal boundaries by phoneme duration priors, keeping the current endpoints';
    recalcBtn.addEventListener('click', recalcBoundaries);
    controls.appendChild(recalcBtn);
  }
  controls.appendChild(dismiss);

  header.appendChild(title);
  header.appendChild(controls);
  container.appendChild(header);

  const canvas = document.createElement('canvas');
  canvas.width = 800;
  canvas.height = 120;
  container.appendChild(canvas);
  editor.canvas = canvas;

  for (let i = 0; i < editor.phonemes.length; i++) {
    const ph = editor.phonemes[i];
    const row = document.createElement('div');
    row.className = 'row';
    row.dataset.index = String(i);

    const checkbox = document.createElement('input');
    checkbox.type = 'checkbox';
    checkbox.checked = !ph.recorded;
    checkbox.dataset.index = String(i);

    const sym = document.createElement('span');
    sym.className = 'symbol';
    sym.textContent = ph.symbol;

    const ex = document.createElement('span');
    ex.className = 'ex';
    ex.textContent = ph.example;

    const play = document.createElement('button');
    play.className = 'play';
    play.textContent = '▶';
    play.title = `Play ${ph.symbol}`;
    play.addEventListener('click', (e) => {
      e.preventDefault();
      const span = editor.phonemeSpansMs[i];
      playSlice(span.start_ms, span.end_ms);
    });

    const duration = document.createElement('span');
    duration.className = 'duration';
    const initSpan = editor.phonemeSpansMs[i];
    duration.textContent = `${initSpan.end_ms - initSpan.start_ms} ms`;

    row.appendChild(checkbox);
    row.appendChild(sym);
    row.appendChild(ex);
    row.appendChild(play);
    row.appendChild(duration);

    if (ph.recorded) {
      const note = document.createElement('span');
      note.className = 'note';
      note.textContent = 'already recorded';
      row.appendChild(note);
    }

    container.appendChild(row);
  }

  const actions = document.createElement('div');
  actions.className = 'actions';
  if (editor.mode !== 'phoneme') {
    const saveBtn = document.createElement('button');
    saveBtn.textContent = 'Save phonemes';
    saveBtn.className = 'save';
    saveBtn.addEventListener('click', () => savePhonemesFromEditor());
    actions.appendChild(saveBtn);
  }
  const saveTrimBtn = document.createElement('button');
  saveTrimBtn.textContent = 'Save trim';
  saveTrimBtn.className = 'save-trim';
  saveTrimBtn.title = editor.mode === 'phoneme'
    ? 'Crop the phoneme recording to the outer boundaries'
    : 'Crop the word recording to the outer boundaries';
  saveTrimBtn.addEventListener('click', () => saveTrimFromEditor());
  editor.trimSaveBtn = saveTrimBtn;
  const cancelBtn = document.createElement('button');
  cancelBtn.textContent = 'Cancel';
  cancelBtn.className = 'cancel';
  cancelBtn.addEventListener('click', closeExtractEditor);
  actions.appendChild(saveTrimBtn);
  actions.appendChild(cancelBtn);
  container.appendChild(actions);
  updateTrimEnabled();

  els.recordPane.insertAdjacentElement('afterend', container);
  editor.container = container;
  attachCanvasDrag();
}

async function decodeWaveform() {
  if (!editor) return;
  const bin = atob(editor.audioBase64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  // Parse WAV header manually — samples are little-endian int16 starting at offset 44.
  const samplesArr = new Int16Array(bytes.buffer, 44, (bytes.length - 44) / 2);
  editor.samples = samplesArr;
  drawWaveform();
}

function drawWaveform() {
  if (!editor || !editor.canvas || !editor.samples) return;
  const canvas = editor.canvas;
  const ctx = canvas.getContext('2d');
  const cssW = canvas.clientWidth;
  if (cssW > 0 && canvas.width !== cssW) {
    canvas.width = cssW;
  }
  const w = canvas.width;
  const h = canvas.height;
  const mid = h / 2;

  ctx.fillStyle = '#1a1b1e';
  ctx.fillRect(0, 0, w, h);

  const samples = editor.samples;
  // Use overall peak for amplitude normalisation so trim doesn't change
  // the visual scale of the loud region.
  let peak = 0;
  for (let i = 0; i < samples.length; i++) {
    const a = Math.abs(samples[i]);
    if (a > peak) peak = a;
  }
  const threshold = peak * 0.02;

  const viewStart = editor.viewStartMs;
  const viewEnd = editor.viewEndMs;
  const viewSpan = Math.max(1, viewEnd - viewStart);

  // Sample range covered by the current view.
  const sampleStart = Math.max(0, Math.floor(
    viewStart * samples.length / editor.audioDurationMs));
  const sampleEnd = Math.min(samples.length, Math.ceil(
    viewEnd * samples.length / editor.audioDurationMs));
  const visibleLen = Math.max(1, sampleEnd - sampleStart);
  const bucketSize = visibleLen / w;

  for (let x = 0; x < w; x++) {
    const start = sampleStart + Math.floor(x * bucketSize);
    const end = sampleStart + Math.floor((x + 1) * bucketSize);
    let bucketPeak = 0;
    for (let i = start; i < end; i++) {
      const a = Math.abs(samples[i]);
      if (a > bucketPeak) bucketPeak = a;
    }
    const y = (bucketPeak / 32768) * (h / 2);
    const isLoud = bucketPeak >= threshold;
    ctx.fillStyle = isLoud ? '#8fdf9b' : '#4a4d54';
    ctx.fillRect(x, mid - y, 1, Math.max(1, y * 2));
  }

  // Build the list of "non-phoneme" regions in absolute ms: pre-first,
  // any internal gaps between consecutive phoneme spans, and post-last.
  const spans = editor.phonemeSpansMs;
  const gaps = [];
  gaps.push([0, spans[0].start_ms]);
  for (let i = 0; i < spans.length - 1; i++) {
    gaps.push([spans[i].end_ms, spans[i + 1].start_ms]);
  }
  gaps.push([spans[spans.length - 1].end_ms, editor.audioDurationMs]);

  ctx.fillStyle = 'rgba(0, 0, 0, 0.55)';
  for (const [g0, g1] of gaps) {
    if (g1 <= g0) continue;
    const x0 = Math.max(0, ((g0 - viewStart) / viewSpan) * w);
    const x1 = Math.min(w, ((g1 - viewStart) / viewSpan) * w);
    if (x1 > x0) ctx.fillRect(x0, 0, x1 - x0, h);
  }

  ctx.strokeStyle = '#4ea8de';
  ctx.lineWidth = 2;
  // Draw both handles for every span (2N lines, possibly coincident when
  // adjacent phonemes are contiguous).
  for (const span of spans) {
    for (const b of [span.start_ms, span.end_ms]) {
      if (b < viewStart || b > viewEnd) continue;
      const x = ((b - viewStart) / viewSpan) * w;
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, h);
      ctx.stroke();
    }
  }
}

function recalcBoundaries() {
  if (!editor) return;
  const n = editor.phonemes.length;
  if (n === 0) return;
  const spans = editor.phonemeSpansMs;
  const first = spans[0].start_ms;
  const last = spans[n - 1].end_ms;
  const totalRange = Math.max(0, last - first);

  const priors = editor.phonemes.map((p) => (p.prior_ms || 100));
  const totalPrior = priors.reduce((a, b) => a + b, 0);
  if (totalPrior === 0) return;

  // Redistribute contiguously between current outer endpoints — recalc
  // intentionally collapses any internal gaps; if the user wants gaps
  // back, they re-introduce them with drag.
  const newSpans = [];
  let cumulative = 0;
  for (let i = 0; i < n; i++) {
    const startFrac = cumulative / totalPrior;
    cumulative += priors[i];
    const endFrac = cumulative / totalPrior;
    newSpans.push({
      start_ms: first + Math.round(startFrac * totalRange),
      end_ms: first + Math.round(endFrac * totalRange),
    });
  }

  editor.phonemeSpansMs = newSpans;
  drawWaveform();
  updateRowDurations();
}

function toggleTrim() {
  if (!editor) return;
  const isTrimmed = editor.viewStartMs > 0 || editor.viewEndMs < editor.audioDurationMs;
  if (isTrimmed) {
    editor.viewStartMs = 0;
    editor.viewEndMs = editor.audioDurationMs;
    if (editor.trimBtn) editor.trimBtn.textContent = 'Trim';
  } else {
    applyTrim();
    if (editor.trimBtn) editor.trimBtn.textContent = 'Fit';
  }
  drawWaveform();
}

function attachCanvasDrag() {
  const canvas = editor.canvas;
  const HANDLE_PX = 8;   // hit-test radius around a boundary

  const pxToMs = (x) => {
    const span = Math.max(1, editor.viewEndMs - editor.viewStartMs);
    return editor.viewStartMs + (x / canvas.width) * span;
  };
  const msToPx = (ms) => {
    const span = Math.max(1, editor.viewEndMs - editor.viewStartMs);
    return ((ms - editor.viewStartMs) / span) * canvas.width;
  };

  canvas.addEventListener('mousedown', (e) => {
    if (!editor) return;
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    let nearest = null;
    let nearestPx = Infinity;
    const spans = editor.phonemeSpansMs;
    for (let i = 0; i < spans.length; i++) {
      for (const kind of ['start', 'end']) {
        const ms = kind === 'start' ? spans[i].start_ms : spans[i].end_ms;
        const dist = Math.abs(msToPx(ms) - x);
        if (dist < nearestPx) {
          nearestPx = dist;
          nearest = { phonemeIdx: i, kind };
        }
      }
    }
    if (nearest && nearestPx <= HANDLE_PX) {
      editor.draggingHandle = nearest;
    }
  });

  window.addEventListener('mousemove', (e) => {
    if (!editor || !editor.draggingHandle) return;
    const rect = canvas.getBoundingClientRect();
    const x = Math.max(0, Math.min(canvas.width, e.clientX - rect.left));
    let ms = pxToMs(x);

    const { phonemeIdx: i, kind } = editor.draggingHandle;
    const spans = editor.phonemeSpansMs;
    let minMs, maxMs;
    if (kind === 'start') {
      minMs = i > 0 ? spans[i - 1].end_ms : 0;
      maxMs = spans[i].end_ms - 5;
    } else {
      minMs = spans[i].start_ms + 5;
      maxMs = i < spans.length - 1 ? spans[i + 1].start_ms : editor.audioDurationMs;
    }
    ms = Math.max(minMs, Math.min(maxMs, ms));

    if (kind === 'start') spans[i].start_ms = Math.round(ms);
    else spans[i].end_ms = Math.round(ms);

    drawWaveform();
    updateRowDurations();
  });

  window.addEventListener('mouseup', () => {
    if (editor) editor.draggingHandle = null;
  });
}

function updateRowDurations() {
  if (!editor || !editor.container) return;
  const rows = editor.container.querySelectorAll('.row');
  for (const row of rows) {
    const i = parseInt(row.dataset.index, 10);
    const span = editor.phonemeSpansMs[i];
    const durEl = row.querySelector('.duration');
    if (durEl) durEl.textContent = `${span.end_ms - span.start_ms} ms`;
  }
  updateTrimEnabled();
}

function updateTrimEnabled() {
  if (!editor || !editor.trimSaveBtn) return;
  const spans = editor.phonemeSpansMs;
  const first = spans[0].start_ms;
  const last = spans[spans.length - 1].end_ms;
  editor.trimSaveBtn.disabled = (first === 0 && last === editor.audioDurationMs);
}

async function saveTrimFromEditor() {
  if (!editor) return;
  const spans = editor.phonemeSpansMs;
  const start = spans[0].start_ms;
  const end = spans[spans.length - 1].end_ms;
  const isPhoneme = editor.mode === 'phoneme';
  const label = isPhoneme ? editor.phonemeSymbol : editor.word;

  const ok = window.confirm(
    `Trim "${label}" to ${start}–${end} ms? This overwrites the recording.`
  );
  if (!ok) return;

  const endpoint = isPhoneme ? '/api/phoneme/trim' : '/api/extract/trim';
  const unitId = editor.unitId;
  const body = isPhoneme
    ? { symbol: label, unit_id: unitId, start_ms: start, end_ms: end }
    : { unit_id: unitId, start_ms: start, end_ms: end };

  let r;
  try {
    r = await fetch(endpoint, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
  } catch (e) {
    setStatus(`Trim failed: ${e.message}`, true);
    return;
  }
  if (!r.ok) {
    const err = await r.json().catch(() => ({}));
    setStatus(`Trim: ${err.error || r.statusText}`, true);
    return;
  }
  const { duration_ms } = await r.json();
  setStatus(`Trimmed ${label} → ${duration_ms} ms`);
  closeExtractEditor();
  if (isPhoneme) {
    await openPhonemeEditor(label, unitId);
  } else {
    await openExtractEditor(label, unitId);
  }
}

async function ensureEditorAudioBuffer() {
  if (!editor) return null;
  if (editor.audioBuffer) return editor.audioBuffer;
  const bin = atob(editor.audioBase64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  const ctx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 22050 });
  editor.audioCtx = ctx;
  editor.audioBuffer = await ctx.decodeAudioData(bytes.buffer.slice(0));
  return editor.audioBuffer;
}

async function playSlice(startMs, endMs) {
  const buffer = await ensureEditorAudioBuffer();
  if (!buffer) return;
  const src = editor.audioCtx.createBufferSource();
  src.buffer = buffer;
  src.connect(editor.audioCtx.destination);
  const startSec = startMs / 1000;
  const durationSec = Math.max(0.01, (endMs - startMs) / 1000);
  src.start(0, startSec, durationSec);
}

async function savePhonemesFromEditor() {
  if (!editor) return;
  const spans = editor.phonemeSpansMs.map((s) => ({ start_ms: s.start_ms, end_ms: s.end_ms }));
  const checkboxes = editor.container.querySelectorAll('.row input[type=checkbox]');
  const saveFlags = Array.from(checkboxes).map((cb) => cb.checked);

  let r;
  try {
    r = await fetch('/api/extract/save', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ unit_id: editor.unitId, spans, save: saveFlags }),
    });
  } catch (e) {
    setStatus(`Save phonemes failed: ${e.message}`, true);
    return;
  }
  if (!r.ok) {
    const err = await r.json().catch(() => ({}));
    setStatus(`Save phonemes: ${err.error || r.statusText}`, true);
    return;
  }
  const { saved, skipped } = await r.json();
  const savedSymbols = saved.map((s) => s.symbol);
  let msg = `Saved phonemes: ${savedSymbols.length ? savedSymbols.join(', ') : 'none'}`;
  if (skipped.length) {
    const skipStr = skipped.map((s) => `${s.symbol} (${s.reason})`).join(', ');
    msg += ` — skipped: ${skipStr}`;
  }
  setStatus(msg);
  closeExtractEditor();
  await refreshInventory();
}
