class RecorderProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.buffers = [];
    this.recording = false;
    this.port.onmessage = (e) => {
      if (e.data === 'start') {
        this.buffers = [];
        this.recording = true;
      } else if (e.data === 'stop') {
        this.recording = false;
        let total = 0;
        for (const b of this.buffers) total += b.length;
        const out = new Float32Array(total);
        let off = 0;
        for (const b of this.buffers) { out.set(b, off); off += b.length; }
        this.port.postMessage(out, [out.buffer]);
        this.buffers = [];
      }
    };
  }

  process(inputs) {
    if (this.recording && inputs[0] && inputs[0][0]) {
      this.buffers.push(new Float32Array(inputs[0][0]));
    }
    return true;
  }
}

registerProcessor('recorder-processor', RecorderProcessor);
