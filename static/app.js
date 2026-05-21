const SAMPLE_RATE = 22050;

const els = {
  wordInput: document.getElementById('word-input'),
  recordBtn: document.getElementById('record-btn'),
  stopBtn: document.getElementById('stop-btn'),
  saveBtn: document.getElementById('save-btn'),
  discardBtn: document.getElementById('discard-btn'),
  preview: document.getElementById('preview'),
  recordStatus: document.getElementById('record-status'),
  inventoryChips: document.getElementById('inventory-chips'),
  sentenceInput: document.getElementById('sentence-input'),
  synthBtn: document.getElementById('synth-btn'),
  output: document.getElementById('output'),
  skippedList: document.getElementById('skipped-list'),
  phonemeGrid: document.getElementById('phoneme-grid'),
  recordPane: document.getElementById('record-pane'),
};

let audioCtx = null;
let workletNode = null;
let mediaStream = null;
let currentWavBlob = null;
let currentPhoneme = null;   // null => recording for a word; non-null => recording for this phoneme

// Phoneme editor state — null when closed.
let editor = null;

async function ensureAudio() {
  if (audioCtx) return;
  audioCtx = new AudioContext({ sampleRate: SAMPLE_RATE });
  await audioCtx.audioWorklet.addModule('/static/recorder-worklet.js');
}

async function startRecording() {
  await ensureAudio();
  mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });
  const source = audioCtx.createMediaStreamSource(mediaStream);
  workletNode = new AudioWorkletNode(audioCtx, 'recorder-processor');
  source.connect(workletNode);
  workletNode.port.postMessage('start');
  setStatus('recording…');
  setButtons({ record: false, stop: true, save: false, discard: false });
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

function setButtons({ record, stop, save, discard }) {
  els.recordBtn.disabled = !record;
  els.stopBtn.disabled = !stop;
  els.saveBtn.disabled = !save;
  els.discardBtn.disabled = !discard;
}

async function refreshInventory() {
  await Promise.all([refreshWordInventory(), refreshPhonemeInventory()]);
}

async function refreshWordInventory() {
  const r = await fetch('/api/inventory');
  const { words } = await r.json();
  els.inventoryChips.innerHTML = '';
  for (const w of words) {
    const chip = document.createElement('span');
    chip.className = 'chip';
    chip.textContent = w;
    els.inventoryChips.appendChild(chip);
  }
}

async function refreshPhonemeInventory() {
  const r = await fetch('/api/phoneme-inventory');
  const { phonemes } = await r.json();
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
    if (currentPhoneme === symbol) chip.classList.add('armed');
    els.phonemeGrid.appendChild(chip);
  }
}

function armPhoneme(symbol, example) {
  currentPhoneme = symbol;
  els.wordInput.value = '';   // clear any half-typed word
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
  try {
    await startRecording();
  } catch (err) {
    setStatus('mic permission denied or unavailable', true);
    setButtons({ record: true, stop: false, save: false, discard: false });
  }
});

els.stopBtn.addEventListener('click', async () => {
  const floats = await stopRecording();
  const int16 = floatsToInt16(floats);
  currentWavBlob = encodeWav(int16, SAMPLE_RATE);
  els.preview.src = URL.createObjectURL(currentWavBlob);
  setStatus(`captured ${(floats.length / SAMPLE_RATE).toFixed(2)}s — review then Save`);
  setButtons({ record: true, stop: false, save: true, discard: true });
});

els.discardBtn.addEventListener('click', () => {
  currentWavBlob = null;
  els.preview.removeAttribute('src');
  setStatus('discarded');
  setButtons({ record: true, stop: false, save: false, discard: false });
});

async function saveWord(word, blob) {
  const form = new FormData();
  form.append('word', word);
  form.append('audio', blob, 'audio.wav');

  let r = await fetch('/api/record', { method: 'POST', body: form });
  if (r.status === 409) {
    if (!confirm(`"${word}" already exists. Overwrite?`)) return;
    r = await fetch('/api/record?overwrite=true', { method: 'POST', body: form });
  }
  if (!r.ok) {
    const err = await r.json().catch(() => ({}));
    setStatus(`save failed: ${err.error || r.statusText}`, true);
    return;
  }
  const { word: saved } = await r.json();
  setStatus(`saved "${saved}"`);
  els.wordInput.value = '';
}

async function savePhoneme(symbol, blob) {
  const form = new FormData();
  form.append('phoneme', symbol);
  form.append('audio', blob, 'audio.wav');

  let r = await fetch('/api/record/phoneme', { method: 'POST', body: form });
  if (r.status === 409) {
    if (!confirm(`Phoneme ${symbol} already recorded. Overwrite?`)) return;
    r = await fetch('/api/record/phoneme?overwrite=true', { method: 'POST', body: form });
  }
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
    const word = els.wordInput.value.trim();
    if (!word) { setStatus('enter a word first', true); return; }
    await saveWord(word, currentWavBlob);
  }

  currentWavBlob = null;
  els.preview.removeAttribute('src');
  setButtons({ record: true, stop: false, save: false, discard: false });
  await refreshInventory();
});

async function synthesise() {
  const sentence = els.sentenceInput.value.trim();
  if (!sentence) return;

  const r = await fetch('/api/synthesize', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ sentence }),
  });
  if (!r.ok) {
    const err = await r.json().catch(() => ({}));
    els.skippedList.textContent = `synth failed: ${err.error || r.statusText}`;
    return;
  }
  const { audio: b64, skipped } = await r.json();

  if (b64) {
    const bin = atob(b64);
    const bytes = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
    const blob = new Blob([bytes], { type: 'audio/wav' });
    els.output.src = URL.createObjectURL(blob);
    els.output.play().catch(() => { /* user gesture is the click on synth */ });
  } else {
    els.output.removeAttribute('src');
  }

  renderSkipped(skipped);
}

function renderSkipped(skipped) {
  els.skippedList.innerHTML = '';
  if (!skipped.length) return;

  const label = document.createElement('span');
  label.textContent = 'Skipped (click to record):';
  label.style.color = 'var(--muted)';
  label.style.marginRight = '0.4rem';
  els.skippedList.appendChild(label);

  for (const w of skipped) {
    const chip = document.createElement('span');
    chip.className = 'chip missing';
    chip.textContent = w;
    chip.addEventListener('click', () => {
      els.wordInput.value = w;
      els.wordInput.focus();
    });
    els.skippedList.appendChild(chip);
  }
}

els.synthBtn.addEventListener('click', synthesise);

els.wordInput.addEventListener('input', () => {
  if (currentPhoneme && els.wordInput.value.trim()) disarmPhoneme();
});

refreshInventory();
setButtons({ record: true, stop: false, save: false, discard: false });

async function openExtractEditor(word) {
  closeExtractEditor();

  let prop;
  try {
    const r = await fetch('/api/extract/propose', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ word }),
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
    word: prop.word,
    audioBase64: prop.audio,
    audioDurationMs: prop.audio_duration_ms,
    phonemes: prop.phonemes,
    boundariesMs: prop.boundaries_ms.slice(),
    audioBuffer: null,
    audioCtx: null,
    container: null,
    canvas: null,
    samples: null,
    draggingI: null,
  };

  buildEditorDom();
  decodeWaveform();
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
  title.textContent = `Extracting phonemes from "${editor.word}"`;
  const dismiss = document.createElement('button');
  dismiss.className = 'dismiss';
  dismiss.textContent = '×';
  dismiss.addEventListener('click', closeExtractEditor);
  header.appendChild(title);
  header.appendChild(dismiss);
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

    const duration = document.createElement('span');
    duration.className = 'duration';
    duration.textContent = `${editor.boundariesMs[i + 1] - editor.boundariesMs[i]} ms`;

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
  const saveBtn = document.createElement('button');
  saveBtn.textContent = 'Save phonemes';
  saveBtn.className = 'save';
  const cancelBtn = document.createElement('button');
  cancelBtn.textContent = 'Cancel';
  cancelBtn.className = 'cancel';
  cancelBtn.addEventListener('click', closeExtractEditor);
  actions.appendChild(saveBtn);
  actions.appendChild(cancelBtn);
  container.appendChild(actions);

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
  let peak = 0;
  for (let i = 0; i < samples.length; i++) {
    const a = Math.abs(samples[i]);
    if (a > peak) peak = a;
  }
  const threshold = peak * 0.02;
  const bucketSize = samples.length / w;

  for (let x = 0; x < w; x++) {
    const start = Math.floor(x * bucketSize);
    const end = Math.floor((x + 1) * bucketSize);
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

  ctx.strokeStyle = '#4ea8de';
  ctx.lineWidth = 2;
  for (const b of editor.boundariesMs) {
    const x = (b / editor.audioDurationMs) * w;
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, h);
    ctx.stroke();
  }
}

function attachCanvasDrag() {
  const canvas = editor.canvas;
  const HANDLE_PX = 8;   // hit-test radius around a boundary

  canvas.addEventListener('mousedown', (e) => {
    if (!editor) return;
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    let nearestI = -1;
    let nearestPx = Infinity;
    for (let i = 0; i < editor.boundariesMs.length; i++) {
      const bx = (editor.boundariesMs[i] / editor.audioDurationMs) * canvas.width;
      const dist = Math.abs(bx - x);
      if (dist < nearestPx) {
        nearestPx = dist;
        nearestI = i;
      }
    }
    if (nearestI >= 0 && nearestPx <= HANDLE_PX) {
      editor.draggingI = nearestI;
    }
  });

  window.addEventListener('mousemove', (e) => {
    if (!editor || editor.draggingI === null) return;
    const rect = canvas.getBoundingClientRect();
    const x = Math.max(0, Math.min(canvas.width, e.clientX - rect.left));
    let ms = (x / canvas.width) * editor.audioDurationMs;
    const i = editor.draggingI;
    const minMs = i > 0 ? editor.boundariesMs[i - 1] + 5 : 0;
    const maxMs = i < editor.boundariesMs.length - 1
      ? editor.boundariesMs[i + 1] - 5
      : editor.audioDurationMs;
    ms = Math.max(minMs, Math.min(maxMs, ms));
    editor.boundariesMs[i] = Math.round(ms);
    drawWaveform();
    updateRowDurations();
  });

  window.addEventListener('mouseup', () => {
    if (editor) editor.draggingI = null;
  });
}

function updateRowDurations() {
  if (!editor || !editor.container) return;
  const rows = editor.container.querySelectorAll('.row');
  for (const row of rows) {
    const i = parseInt(row.dataset.index, 10);
    const dur = editor.boundariesMs[i + 1] - editor.boundariesMs[i];
    const durEl = row.querySelector('.duration');
    if (durEl) durEl.textContent = `${dur} ms`;
  }
}
