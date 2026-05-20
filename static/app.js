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
};

let audioCtx = null;
let workletNode = null;
let mediaStream = null;
let currentWavBlob = null;
let currentPhoneme = null;   // null => recording for a word; non-null => recording for this phoneme

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
  for (const { symbol, recorded } of phonemes) {
    const chip = document.createElement('span');
    chip.className = 'phoneme-chip' + (recorded ? ' recorded' : '');
    chip.textContent = symbol;
    chip.dataset.symbol = symbol;
    chip.addEventListener('click', () => armPhoneme(symbol));
    if (currentPhoneme === symbol) chip.classList.add('armed');
    els.phonemeGrid.appendChild(chip);
  }
}

function armPhoneme(symbol) {
  currentPhoneme = symbol;
  els.wordInput.value = '';   // clear any half-typed word
  setStatus(`Recording phoneme ${symbol} — press Record`);
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
