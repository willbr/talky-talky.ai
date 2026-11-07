// Recording level meter: a live envelope of the take while recording, and a
// verdict on the captured audio afterwards (silent / quiet / clipping / ok).

const SILENT_PEAK = 0.003;   // ≈ −50 dBFS: nothing reached the mic
const QUIET_PEAK = 0.05;     // ≈ −26 dBFS: audible only after normalising
const CLIP_PEAK = 0.99;

let analyser = null;
let canvas = null;
let label = null;
let raf = 0;
let history = [];            // per-animation-frame peak, 0..1
let startedAt = 0;
let deviceName = '';

export function initMeter(canvasEl, labelEl) {
  canvas = canvasEl;
  label = labelEl;
}

export function dbfs(peak) {
  return peak > 0 ? 20 * Math.log10(peak) : -Infinity;
}

export function startMeter(audioCtx, source, track) {
  analyser = audioCtx.createAnalyser();
  analyser.fftSize = 1024;
  source.connect(analyser);
  history = [];
  startedAt = performance.now();
  deviceName = (track && track.label) || 'default microphone';
  canvas.hidden = false;
  label.hidden = false;
  const buf = new Float32Array(analyser.fftSize);
  const tick = () => {
    analyser.getFloatTimeDomainData(buf);
    let peak = 0;
    for (let i = 0; i < buf.length; i++) peak = Math.max(peak, Math.abs(buf[i]));
    history.push(peak);
    drawEnvelope(history, true);
    const secs = (performance.now() - startedAt) / 1000;
    const recent = Math.max(...history.slice(-30));
    let msg = `● ${secs.toFixed(1)} s · ${deviceName}`;
    if (secs > 1.5 && Math.max(...history) < SILENT_PEAK) msg += ' · no sound from this mic yet';
    else if (recent >= CLIP_PEAK) msg += ' · too loud!';
    label.textContent = msg;
    label.dataset.state = secs > 1.5 && Math.max(...history) < SILENT_PEAK ? 'bad' : 'live';
    raf = requestAnimationFrame(tick);
  };
  raf = requestAnimationFrame(tick);
}

// Stop the live meter and draw the final take from its actual samples.
// Returns { peak, verdict, message, isError }.
export function finishMeter(floats) {
  cancelAnimationFrame(raf);
  if (analyser) analyser.disconnect();
  analyser = null;

  const cols = Math.max(1, canvas.clientWidth || 800);
  const per = Math.max(1, Math.floor(floats.length / cols));
  const env = [];
  let peak = 0;
  for (let i = 0; i < floats.length; i += per) {
    let p = 0;
    const end = Math.min(floats.length, i + per);
    for (let j = i; j < end; j++) p = Math.max(p, Math.abs(floats[j]));
    env.push(p);
    peak = Math.max(peak, p);
  }
  drawEnvelope(env, false);

  const secs = floats.length / 22050;
  const db = Number.isFinite(dbfs(peak)) ? `${dbfs(peak).toFixed(0)} dB` : 'silence';
  let verdict;
  let message;
  if (floats.length === 0) {
    verdict = 'bad';
    message = 'Nothing was captured: the audio engine never ran. Reload the page and try again.';
  } else if (peak < SILENT_PEAK) {
    verdict = 'bad';
    message = `Silent take (${secs.toFixed(1)} s, peak ${db}). No sound reached "${deviceName}"; check it's the right input and isn't muted.`;
  } else if (peak < QUIET_PEAK) {
    verdict = 'warn';
    message = `Very quiet take (${secs.toFixed(1)} s, peak ${db}). Move closer or raise the mic level; it will be boosted on save but noise comes with it.`;
  } else if (peak >= CLIP_PEAK) {
    verdict = 'warn';
    message = `Clipped take (${secs.toFixed(1)} s, peak ${db}). Too loud; back off a little and re-record.`;
  } else {
    verdict = 'ok';
    message = `Captured ${secs.toFixed(1)} s, peak ${db}. Review, then Save.`;
  }
  label.textContent = `${secs.toFixed(1)} s · peak ${db} · ${deviceName}`;
  label.dataset.state = verdict;
  return { peak, verdict, message, isError: verdict === 'bad' };
}

function drawEnvelope(env, live) {
  const w = canvas.clientWidth || 800;
  if (canvas.width !== w) canvas.width = w;
  const h = canvas.height;
  const ctx = canvas.getContext('2d');
  ctx.fillStyle = '#1a1b1e';
  ctx.fillRect(0, 0, w, h);
  // −26 dB and clip guides
  ctx.strokeStyle = 'rgba(255,255,255,0.08)';
  for (const level of [QUIET_PEAK, CLIP_PEAK]) {
    const y = h / 2 - (level * h) / 2;
    ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(0, h - y); ctx.lineTo(w, h - y); ctx.stroke();
  }
  // Live: newest on the right, one column per frame, scrolling.
  const cols = live ? env.slice(-w) : env;
  const step = live ? 1 : w / Math.max(1, cols.length);
  const x0 = live ? w - cols.length : 0;
  for (let i = 0; i < cols.length; i++) {
    const p = Math.min(1, cols[i]);
    ctx.fillStyle = p >= CLIP_PEAK ? '#ef6262' : p < QUIET_PEAK ? '#6b7d8f' : '#4ea8de';
    const bar = Math.max(1, p * h);
    ctx.fillRect(x0 + i * step, (h - bar) / 2, Math.max(1, step), bar);
  }
}
