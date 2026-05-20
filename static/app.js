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
};

let audioCtx = null;
let workletNode = null;
let mediaStream = null;
let currentWavBlob = null;

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

els.saveBtn.addEventListener('click', async () => {
  const word = els.wordInput.value.trim();
  if (!word) { setStatus('enter a word first', true); return; }
  if (!currentWavBlob) { setStatus('record something first', true); return; }

  const form = new FormData();
  form.append('word', word);
  form.append('audio', currentWavBlob, 'audio.wav');

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
  currentWavBlob = null;
  els.preview.removeAttribute('src');
  els.wordInput.value = '';
  setButtons({ record: true, stop: false, save: false, discard: false });
  await refreshInventory();
});

refreshInventory();
setButtons({ record: true, stop: false, save: false, discard: false });
