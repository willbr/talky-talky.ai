"""Flask app factory and routes for talky-talky.ai."""
from __future__ import annotations

import base64
import io
import logging
import wave

import numpy as np
from flask import Flask, jsonify, render_template, request

import audio
import books
import extraction
import inventory
import phonemes
import synth

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

    @flask_app.route("/api/record/phoneme", methods=["POST"])
    def api_record_phoneme():
        phoneme_raw = request.form.get("phoneme", "").strip()
        if not phoneme_raw:
            return jsonify({"error": "phoneme is required"}), 400
        phoneme = phoneme_raw.upper()
        if phoneme not in phonemes.PHONEMES_SET:
            return jsonify({
                "error": "unknown phoneme",
                "allowed": list(phonemes.PHONEMES),
            }), 400

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
            log.warning("record/phoneme: invalid WAV: %s", e)
            return jsonify({"error": "invalid WAV file"}), 400

        if rate != audio.SAMPLE_RATE or channels != audio.CHANNELS or sampwidth != audio.SAMPLE_WIDTH:
            log.warning(
                "record/phoneme: format mismatch rate=%s channels=%s sampwidth=%s",
                rate, channels, sampwidth,
            )
            return jsonify({"error": "expected 16-bit mono PCM at 22050 Hz"}), 400

        samples = np.frombuffer(frames, dtype=np.int16)
        samples = audio.trim_silence(samples)
        if len(samples) < int(audio.SAMPLE_RATE * audio.MIN_LENGTH_SECONDS):
            log.warning("record/phoneme: too short after trim: %s samples", len(samples))
            return jsonify({"error": "recording too short"}), 400

        samples = audio.peak_normalise(samples)

        overwrite = request.args.get("overwrite") == "true"
        if inventory.has_phoneme(phoneme) and not overwrite:
            return jsonify({"error": "phoneme exists", "phoneme": phoneme}), 409

        out = io.BytesIO()
        with wave.open(out, "wb") as w:
            w.setnchannels(audio.CHANNELS)
            w.setsampwidth(audio.SAMPLE_WIDTH)
            w.setframerate(audio.SAMPLE_RATE)
            w.writeframes(samples.tobytes())
        inventory.save_phoneme(phoneme, out.getvalue())

        log.info("record/phoneme: saved %r (%d samples)", phoneme, len(samples))
        return jsonify({"phoneme": phoneme}), 200

    @flask_app.route("/api/phoneme-inventory")
    def api_phoneme_inventory():
        recorded = set(inventory.list_recorded_phonemes())
        return jsonify({
            "phonemes": [
                {
                    "symbol": s,
                    "example": phonemes.PHONEME_EXAMPLES[s],
                    "recorded": s in recorded,
                }
                for s in phonemes.PHONEMES
            ]
        })

    @flask_app.route("/api/extract/propose", methods=["POST"])
    def api_extract_propose():
        data = request.get_json(silent=True) or {}
        word_raw = data.get("word", "")
        word = inventory.normalise(word_raw)
        if not word:
            return jsonify({"error": "word is required"}), 400
        if not inventory.has(word):
            return jsonify({"error": "word not recorded", "word": word}), 404

        phs = phonemes.word_to_phonemes(word)
        if phs is None:
            return jsonify({"error": "word not in dictionary", "word": word}), 400

        samples, _ = audio.read_wav(inventory.path(word))
        audio_duration_ms = int(len(samples) * 1000 / audio.SAMPLE_RATE)
        boundaries_ms = extraction.propose_boundaries(samples, phs)
        wav_bytes = inventory.path(word).read_bytes()
        audio_b64 = base64.b64encode(wav_bytes).decode("ascii")

        phoneme_payload = [
            {
                "symbol": p,
                "example": phonemes.PHONEME_EXAMPLES.get(p, ""),
                "recorded": inventory.has_phoneme(p),
                "prior_ms": extraction.PHONEME_DURATION_MS.get(
                    p.upper(), extraction.DEFAULT_DURATION_MS,
                ),
            }
            for p in phs
        ]

        log.info("extract/propose: word=%r phonemes=%d duration=%dms",
                 word, len(phs), audio_duration_ms)
        return jsonify({
            "word": word,
            "audio_duration_ms": audio_duration_ms,
            "audio": audio_b64,
            "phonemes": phoneme_payload,
            "boundaries_ms": boundaries_ms,
        }), 200

    @flask_app.route("/api/extract/save", methods=["POST"])
    def api_extract_save():
        data = request.get_json(silent=True) or {}
        word_raw = data.get("word", "")
        word = inventory.normalise(word_raw)
        if not word:
            return jsonify({"error": "word is required"}), 400
        if not inventory.has(word):
            return jsonify({"error": "word not recorded", "word": word}), 404

        boundaries_ms = data.get("boundaries_ms")
        save_flags = data.get("save")
        if not isinstance(boundaries_ms, list):
            return jsonify({"error": "boundaries_ms is required"}), 400
        if not isinstance(save_flags, list):
            return jsonify({"error": "save is required"}), 400
        if len(boundaries_ms) < 2:
            return jsonify({"error": "boundaries_ms must have at least two entries"}), 400
        if len(save_flags) != len(boundaries_ms) - 1:
            return jsonify({"error": "save length must equal boundaries_ms length minus one"}), 400

        for i in range(len(boundaries_ms) - 1):
            if boundaries_ms[i + 1] < boundaries_ms[i]:
                return jsonify({"error": "boundaries must be non-decreasing"}), 400

        phs = phonemes.word_to_phonemes(word)
        if phs is None:
            return jsonify({"error": "word not in dictionary"}), 400
        if len(phs) != len(save_flags):
            return jsonify({"error": "phoneme count mismatch"}), 400

        samples, _ = audio.read_wav(inventory.path(word))
        audio_duration_ms = int(len(samples) * 1000 / audio.SAMPLE_RATE)
        for b in boundaries_ms:
            if not isinstance(b, (int, float)) or b < 0 or b > audio_duration_ms + 5:
                return jsonify({"error": "boundary out of range"}), 400

        min_slice_samples = int(audio.SAMPLE_RATE * 0.030)
        saved: list[str] = []
        skipped: list[dict] = []
        for i, should_save in enumerate(save_flags):
            if not should_save:
                continue
            symbol = phs[i]
            slice_arr = audio.slice_samples(samples, int(boundaries_ms[i]), int(boundaries_ms[i + 1]))
            if len(slice_arr) < min_slice_samples:
                skipped.append({"symbol": symbol, "reason": "too short"})
                continue
            normalised = audio.peak_normalise(slice_arr)

            out = io.BytesIO()
            with wave.open(out, "wb") as w:
                w.setnchannels(audio.CHANNELS)
                w.setsampwidth(audio.SAMPLE_WIDTH)
                w.setframerate(audio.SAMPLE_RATE)
                w.writeframes(normalised.tobytes())
            inventory.save_phoneme(symbol, out.getvalue())
            saved.append(symbol)

        log.info("extract/save: word=%r saved=%s skipped=%s", word, saved, skipped)
        return jsonify({"saved": saved, "skipped": skipped}), 200

    @flask_app.route("/api/sentence/record", methods=["POST"])
    def api_sentence_record():
        sentence = request.form.get("sentence", "").strip()
        if not sentence:
            return jsonify({"error": "sentence is required"}), 400

        audio_file = request.files.get("audio")
        if audio_file is None:
            return jsonify({"error": "missing audio field"}), 400

        raw = audio_file.read()
        try:
            with wave.open(io.BytesIO(raw), "rb") as r:
                rate = r.getframerate()
                channels = r.getnchannels()
                sampwidth = r.getsampwidth()
        except wave.Error as e:
            log.warning("sentence/record: invalid WAV: %s", e)
            return jsonify({"error": "invalid WAV file"}), 400

        if rate != audio.SAMPLE_RATE or channels != audio.CHANNELS or sampwidth != audio.SAMPLE_WIDTH:
            return jsonify({"error": "expected 16-bit mono PCM at 22050 Hz"}), 400

        sid = inventory.save_sentence(sentence, raw)
        log.info("sentence/record: saved id=%s text=%r", sid, sentence)
        return jsonify({"id": sid}), 200

    @flask_app.route("/api/sentences")
    def api_sentences():
        return jsonify({"sentences": inventory.list_sentences()})

    @flask_app.route("/api/books")
    def api_books():
        return jsonify({"books": books.list_books()})

    @flask_app.route("/api/books/sample", methods=["POST"])
    def api_books_sample():
        data = request.get_json(silent=True) or {}
        result = books.sample_sentence(data.get("book"))
        if result is None:
            return jsonify({"error": "no sentence available"}), 404
        book, sentence = result
        return jsonify({"book": book, "sentence": sentence}), 200

    @flask_app.route("/api/synthesize", methods=["POST"])
    def api_synthesize():
        data = request.get_json(silent=True) or {}
        sentence = data.get("sentence")
        if sentence is None:
            return jsonify({"error": "missing 'sentence' field"}), 400
        if not sentence.strip():
            return jsonify({"error": "sentence is empty"}), 400

        wav_bytes, skipped = synth.synthesise(sentence)
        encoded = base64.b64encode(wav_bytes).decode("ascii") if wav_bytes else None
        log.info("synthesize: %r resolved=%s skipped=%s",
                 sentence, "yes" if wav_bytes else "no", skipped)
        return jsonify({"audio": encoded, "skipped": skipped}), 200

    return flask_app


if __name__ == "__main__":
    create_app().run(debug=True, port=5000)
