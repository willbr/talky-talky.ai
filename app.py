"""Flask app factory and routes for talky-talky.ai."""
from __future__ import annotations

import io
import logging
import wave

import numpy as np
from flask import Flask, jsonify, render_template, request

import audio
import inventory

log = logging.getLogger("talky")


def create_app() -> Flask:
    flask_app = Flask(__name__, static_folder="static", template_folder="templates")
    logging.basicConfig(level=logging.INFO)

    @flask_app.route("/")
    def index() -> str:
        return render_template("index.html")

    @flask_app.route("/api/inventory")
    def api_inventory():
        return jsonify({"words": inventory.list_words()})

    @flask_app.route("/api/record", methods=["POST"])
    def api_record():
        word_raw = request.form.get("word", "")
        word = inventory.normalise(word_raw)
        if not word:
            log.warning("record: empty word after normalisation: %r", word_raw)
            return jsonify({"error": "word is empty after normalisation"}), 400

        audio_file = request.files.get("audio")
        if audio_file is None:
            return jsonify({"error": "missing audio field"}), 400

        raw = audio_file.read()
        try:
            with wave.open(io.BytesIO(raw), "rb") as r:
                rate = r.getframerate()
                channels = r.getnchannels()
                sampwidth = r.getsampwidth()
                n_frames = r.getnframes()
                frames = r.readframes(n_frames)
        except wave.Error as e:
            log.warning("record: invalid WAV: %s", e)
            return jsonify({"error": "invalid WAV file"}), 400

        if rate != audio.SAMPLE_RATE or channels != audio.CHANNELS or sampwidth != audio.SAMPLE_WIDTH:
            log.warning(
                "record: format mismatch rate=%s channels=%s sampwidth=%s",
                rate, channels, sampwidth,
            )
            return jsonify({"error": "expected 16-bit mono PCM at 22050 Hz"}), 400

        samples = np.frombuffer(frames, dtype=np.int16)
        samples = audio.trim_silence(samples)
        if len(samples) < int(audio.SAMPLE_RATE * audio.MIN_LENGTH_SECONDS):
            log.warning("record: too short after trim: %s samples", len(samples))
            return jsonify({"error": "recording too short"}), 400

        samples = audio.peak_normalise(samples)

        overwrite = request.args.get("overwrite") == "true"
        if inventory.has(word) and not overwrite:
            return jsonify({"error": "word exists", "existing": word}), 409

        out = io.BytesIO()
        with wave.open(out, "wb") as w:
            w.setnchannels(audio.CHANNELS)
            w.setsampwidth(audio.SAMPLE_WIDTH)
            w.setframerate(audio.SAMPLE_RATE)
            w.writeframes(samples.tobytes())
        inventory.save(word, out.getvalue())

        log.info("record: saved %r (%d samples)", word, len(samples))
        return jsonify({"word": word}), 200

    return flask_app


if __name__ == "__main__":
    create_app().run(debug=True, port=5000)
