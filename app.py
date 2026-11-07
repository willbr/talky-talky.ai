"""Flask app factory and routes for talky-talky.ai."""
from __future__ import annotations

import base64
import io
import logging
import wave

import numpy as np
from flask import Flask, jsonify, render_template, request, send_file

import audio
import books
import extraction
import inventory
import phonemes
import spectrogram
import store
import synth

log = logging.getLogger("talky")

PHONEME_MIN_SLICE_SECONDS = 0.030


def _read_upload_wav(file, route: str):
    """Validate an uploaded WAV. Returns (samples, None) or (None, error response)."""
    if file is None:
        return None, (jsonify({"error": "missing audio field"}), 400)
    try:
        with wave.open(io.BytesIO(file.read()), "rb") as r:
            rate, channels, sampwidth = r.getframerate(), r.getnchannels(), r.getsampwidth()
            frames = r.readframes(r.getnframes())
    except (wave.Error, EOFError) as e:
        log.warning("%s: invalid WAV: %s", route, e)
        return None, (jsonify({"error": "invalid WAV file"}), 400)
    if rate != audio.SAMPLE_RATE or channels != audio.CHANNELS or sampwidth != audio.SAMPLE_WIDTH:
        log.warning("%s: format mismatch rate=%s channels=%s sampwidth=%s",
                    route, rate, channels, sampwidth)
        return None, (jsonify({"error": "expected 16-bit mono PCM at 22050 Hz"}), 400)
    return np.frombuffer(frames, dtype=np.int16), None


def _trimmed_take(samples: np.ndarray, route: str):
    """Silence-trim + peak-normalise a fresh recording. Returns (wav_bytes, err)."""
    samples = audio.trim_silence(samples)
    if len(samples) < int(audio.SAMPLE_RATE * audio.MIN_LENGTH_SECONDS):
        log.warning("%s: too short after trim: %s samples", route, len(samples))
        return None, (jsonify({"error": "recording too short"}), 400)
    return audio.encode_wav(audio.peak_normalise(samples)), None


def _duration_ms(samples: np.ndarray) -> int:
    return int(len(samples) * 1000 / audio.SAMPLE_RATE)


def _spectrogram_payload(samples: np.ndarray) -> dict:
    spec = spectrogram.compute(samples)
    return {
        "data": base64.b64encode(spec["data"]).decode("ascii"),
        "n_frames": spec["n_frames"],
        "n_freqs": spec["n_freqs"],
        "hop_ms": spec["hop_ms"],
        "duration_ms": spec["duration_ms"],
    }


def _word_unit(data: dict):
    """Resolve the word copy a request targets: `unit_id` if given, else the
    most recent copy of `word`. Returns (unit, None) or (None, error)."""
    unit_id = data.get("unit_id")
    if unit_id is not None:
        unit = store.get_unit(unit_id)
        if unit is None or unit["kind"] != "word":
            return None, (jsonify({"error": "unit not found", "unit_id": unit_id}), 404)
        return unit, None
    word = inventory.normalise(str(data.get("word", "")))
    if not word:
        return None, (jsonify({"error": "word is required"}), 400)
    units = store.list_units("word", word)
    if not units:
        return None, (jsonify({"error": "word not recorded", "word": word}), 404)
    return units[-1], None


def _phoneme_unit(symbol: str, unit_id):
    """Resolve the phoneme copy a request targets (most recent by default)."""
    if unit_id is not None:
        unit = store.get_unit(unit_id)
        if unit is None or unit["kind"] != "phoneme" or unit["label"] != symbol:
            return None, (jsonify({"error": "unit not found", "symbol": symbol,
                                   "unit_id": unit_id}), 404)
        return unit, None
    units = store.list_units("phoneme", symbol)
    if not units:
        return None, (jsonify({"error": "phoneme not recorded", "symbol": symbol}), 404)
    return units[-1], None


def _symbol_from(data: dict):
    symbol = str(data.get("symbol", "")).strip().upper()
    if not symbol:
        return None, (jsonify({"error": "symbol is required"}), 400)
    if symbol not in phonemes.PHONEMES_SET:
        return None, (jsonify({"error": "unknown phoneme", "symbol": symbol}), 400)
    return symbol, None


def _trim_range(data: dict, samples: np.ndarray, min_seconds: float):
    """Validate start/end and return (normalised_slice, None) or (None, error)."""
    start_ms, end_ms = data.get("start_ms"), data.get("end_ms")
    if not isinstance(start_ms, (int, float)) or not isinstance(end_ms, (int, float)):
        return None, (jsonify({"error": "start_ms and end_ms are required"}), 400)
    if end_ms <= start_ms:
        return None, (jsonify({"error": "end_ms must be greater than start_ms"}), 400)
    if start_ms < 0 or end_ms > _duration_ms(samples) + 5:
        return None, (jsonify({"error": "trim out of range"}), 400)
    slice_arr = audio.slice_samples(samples, int(start_ms), int(end_ms))
    if len(slice_arr) < int(audio.SAMPLE_RATE * min_seconds):
        return None, (jsonify({"error": "trim too short"}), 400)
    return audio.peak_normalise(slice_arr), None


def create_app() -> Flask:
    flask_app = Flask(__name__, static_folder="static", template_folder="templates")
    logging.basicConfig(level=logging.INFO)

    @flask_app.route("/")
    def index() -> str:
        return render_template("index.html")

    @flask_app.route("/api/inventory")
    def api_inventory():
        return jsonify({"words": store.list_labels("word")})

    @flask_app.route("/api/record", methods=["POST"])
    def api_record():
        word_raw = request.form.get("word", "")
        word = inventory.normalise(word_raw)
        if not word:
            log.warning("record: empty word after normalisation: %r", word_raw)
            return jsonify({"error": "word is empty after normalisation"}), 400
        samples, err = _read_upload_wav(request.files.get("audio"), "record")
        if err:
            return err
        wav_bytes, err = _trimmed_take(samples, "record")
        if err:
            return err
        unit_id = store.add_unit("word", word, wav_bytes, source="isolated")
        log.info("record: saved %r unit=%s", word, unit_id)
        return jsonify({"word": word, "unit_id": unit_id}), 200

    @flask_app.route("/api/record/phoneme", methods=["POST"])
    def api_record_phoneme():
        phoneme = request.form.get("phoneme", "").strip().upper()
        if not phoneme:
            return jsonify({"error": "phoneme is required"}), 400
        if phoneme not in phonemes.PHONEMES_SET:
            return jsonify({"error": "unknown phoneme", "allowed": list(phonemes.PHONEMES)}), 400
        samples, err = _read_upload_wav(request.files.get("audio"), "record/phoneme")
        if err:
            return err
        wav_bytes, err = _trimmed_take(samples, "record/phoneme")
        if err:
            return err
        unit_id = store.add_unit("phoneme", phoneme, wav_bytes, source="isolated")
        log.info("record/phoneme: saved %r unit=%s", phoneme, unit_id)
        return jsonify({"phoneme": phoneme, "unit_id": unit_id}), 200

    @flask_app.route("/api/phoneme-inventory")
    def api_phoneme_inventory():
        recorded = set(store.list_labels("phoneme"))
        return jsonify({
            "phonemes": [
                {"symbol": s, "example": phonemes.PHONEME_EXAMPLES[s], "recorded": s in recorded}
                for s in phonemes.PHONEMES
            ]
        })

    # --- unit copies ------------------------------------------------------------

    @flask_app.route("/api/units", methods=["GET"])
    def api_units():
        kind = request.args.get("kind", "")
        if kind not in store.KINDS:
            return jsonify({"error": "kind must be 'word' or 'phoneme'"}), 400
        label = store.canonical_label(kind, request.args.get("label", ""))
        if not label:
            return jsonify({"error": "label is required"}), 400
        if kind == "phoneme" and label not in phonemes.PHONEMES_SET:
            return jsonify({"error": "unknown phoneme", "symbol": label}), 400
        return jsonify({"kind": kind, "label": label, "units": store.list_units(kind, label)}), 200

    @flask_app.route("/api/units/<unit_id>/audio", methods=["GET"])
    def api_unit_audio(unit_id):
        if store.get_unit(unit_id) is None:
            return jsonify({"error": "unit not found"}), 404
        # Absolute path: send_file resolves relative paths against the app
        # root, not the working directory INVENTORY_DIR is relative to.
        return send_file(store.unit_path(unit_id).resolve(), mimetype="audio/wav")

    @flask_app.route("/api/units/<unit_id>", methods=["DELETE"])
    def api_unit_delete(unit_id):
        if not store.delete_unit(unit_id):
            return jsonify({"error": "unit not found", "unit_id": unit_id}), 404
        log.info("units: deleted %s", unit_id)
        return ("", 204)

    # --- word editor ------------------------------------------------------------

    @flask_app.route("/api/extract/propose", methods=["POST"])
    def api_extract_propose():
        unit, err = _word_unit(request.get_json(silent=True) or {})
        if err:
            return err
        word = unit["label"]
        phs = phonemes.word_to_phonemes(word)
        if phs is None:
            return jsonify({"error": "word not in dictionary", "word": word}), 400

        path = store.unit_path(unit["id"])
        samples, _ = audio.read_wav(path)
        boundaries_ms = extraction.propose_boundaries(samples, phs)
        recorded = set(store.list_labels("phoneme"))
        return jsonify({
            "word": word,
            "unit_id": unit["id"],
            "audio_duration_ms": _duration_ms(samples),
            "audio": base64.b64encode(path.read_bytes()).decode("ascii"),
            "phonemes": [
                {
                    "symbol": p,
                    "example": phonemes.PHONEME_EXAMPLES.get(p, ""),
                    "recorded": p in recorded,
                    "prior_ms": extraction.PHONEME_DURATION_MS.get(
                        p.upper(), extraction.DEFAULT_DURATION_MS),
                }
                for p in phs
            ],
            "spans": [
                {"start_ms": int(boundaries_ms[i]), "end_ms": int(boundaries_ms[i + 1])}
                for i in range(len(phs))
            ],
        }), 200

    @flask_app.route("/api/extract/save", methods=["POST"])
    def api_extract_save():
        data = request.get_json(silent=True) or {}
        unit, err = _word_unit(data)
        if err:
            return err
        word = unit["label"]

        spans = data.get("spans")
        save_flags = data.get("save")
        if not isinstance(spans, list) or not spans:
            return jsonify({"error": "spans is required"}), 400
        if not isinstance(save_flags, list):
            return jsonify({"error": "save is required"}), 400
        if len(save_flags) != len(spans):
            return jsonify({"error": "save length must equal spans length"}), 400
        for s in spans:
            if (not isinstance(s, dict)
                    or not isinstance(s.get("start_ms"), (int, float))
                    or not isinstance(s.get("end_ms"), (int, float))):
                return jsonify({"error": "each span must have start_ms and end_ms"}), 400
            if s["end_ms"] <= s["start_ms"]:
                return jsonify({"error": "span end_ms must be greater than start_ms"}), 400
        for prev, curr in zip(spans, spans[1:]):
            if curr["start_ms"] < prev["end_ms"]:
                return jsonify({"error": "spans must not overlap"}), 400

        phs = phonemes.word_to_phonemes(word)
        if phs is None:
            return jsonify({"error": "word not in dictionary"}), 400
        if len(phs) != len(save_flags):
            return jsonify({"error": "phoneme count mismatch"}), 400

        samples, _ = audio.read_wav(store.unit_path(unit["id"]))
        duration_ms = _duration_ms(samples)
        for s in spans:
            if s["start_ms"] < 0 or s["end_ms"] > duration_ms + 5:
                return jsonify({"error": "span out of range"}), 400

        min_slice = int(audio.SAMPLE_RATE * PHONEME_MIN_SLICE_SECONDS)
        saved: list[dict] = []
        skipped: list[dict] = []
        for i, should_save in enumerate(save_flags):
            if not should_save:
                continue
            symbol = phs[i]
            slice_arr = audio.slice_samples(samples, int(spans[i]["start_ms"]), int(spans[i]["end_ms"]))
            if len(slice_arr) < min_slice:
                skipped.append({"symbol": symbol, "reason": "too short"})
                continue
            unit_id = store.add_unit(
                "phoneme", symbol, audio.encode_wav(audio.peak_normalise(slice_arr)),
                source="word", source_unit_id=unit["id"], source_word=word,
                prev=phs[i - 1] if i > 0 else None,
                next=phs[i + 1] if i < len(phs) - 1 else None,
            )
            saved.append({"symbol": symbol, "unit_id": unit_id})

        log.info("extract/save: word=%r unit=%s saved=%s skipped=%s",
                 word, unit["id"], saved, skipped)
        return jsonify({"saved": saved, "skipped": skipped}), 200

    @flask_app.route("/api/extract/trim", methods=["POST"])
    def api_extract_trim():
        data = request.get_json(silent=True) or {}
        unit, err = _word_unit(data)
        if err:
            return err
        samples, _ = audio.read_wav(store.unit_path(unit["id"]))
        trimmed, err = _trim_range(data, samples, audio.MIN_LENGTH_SECONDS)
        if err:
            return err
        duration_ms = store.replace_unit_audio(unit["id"], audio.encode_wav(trimmed))
        log.info("extract/trim: word=%r unit=%s -> %dms", unit["label"], unit["id"], duration_ms)
        return jsonify({"word": unit["label"], "unit_id": unit["id"],
                        "duration_ms": duration_ms}), 200

    # --- phoneme editor -----------------------------------------------------------

    @flask_app.route("/api/phoneme/propose", methods=["POST"])
    def api_phoneme_propose():
        data = request.get_json(silent=True) or {}
        symbol, err = _symbol_from(data)
        if err:
            return err
        unit, err = _phoneme_unit(symbol, data.get("unit_id"))
        if err:
            return err
        path = store.unit_path(unit["id"])
        samples, _ = audio.read_wav(path)
        return jsonify({
            "symbol": symbol,
            "unit_id": unit["id"],
            "example": phonemes.PHONEME_EXAMPLES.get(symbol, ""),
            "audio_duration_ms": _duration_ms(samples),
            "audio": base64.b64encode(path.read_bytes()).decode("ascii"),
        }), 200

    @flask_app.route("/api/phoneme/trim", methods=["POST"])
    def api_phoneme_trim():
        data = request.get_json(silent=True) or {}
        symbol, err = _symbol_from(data)
        if err:
            return err
        unit, err = _phoneme_unit(symbol, data.get("unit_id"))
        if err:
            return err
        samples, _ = audio.read_wav(store.unit_path(unit["id"]))
        # Phonemes are short — allow trims down to the extraction slice floor.
        trimmed, err = _trim_range(data, samples, PHONEME_MIN_SLICE_SECONDS)
        if err:
            return err
        duration_ms = store.replace_unit_audio(unit["id"], audio.encode_wav(trimmed))
        log.info("phoneme/trim: symbol=%r unit=%s -> %dms", symbol, unit["id"], duration_ms)
        return jsonify({"symbol": symbol, "unit_id": unit["id"], "duration_ms": duration_ms}), 200

    # --- reference sentences -------------------------------------------------------

    @flask_app.route("/api/sentence/record", methods=["POST"])
    def api_sentence_record():
        sentence = request.form.get("sentence", "").strip()
        if not sentence:
            return jsonify({"error": "sentence is required"}), 400
        samples, err = _read_upload_wav(request.files.get("audio"), "sentence/record")
        if err:
            return err
        # Saved untrimmed and un-normalised: references are raw training data.
        sid = store.save_sentence(sentence, audio.encode_wav(samples))
        log.info("sentence/record: saved id=%s text=%r", sid, sentence)
        return jsonify({"id": sid}), 200

    @flask_app.route("/api/sentences")
    def api_sentences():
        return jsonify({"sentences": store.list_sentences()})

    @flask_app.route("/api/sentence/audio", methods=["POST"])
    def api_sentence_audio():
        data = request.get_json(silent=True) or {}
        sentence = data.get("sentence")
        if not isinstance(sentence, str) or not sentence.strip():
            return jsonify({"error": "sentence is required"}), 400
        wav_path = store.find_sentence_by_text(sentence)
        if wav_path is None:
            return jsonify({"found": False}), 200
        samples, _ = audio.read_wav(wav_path)
        return jsonify({
            "found": True,
            "id": wav_path.stem,
            "audio": base64.b64encode(wav_path.read_bytes()).decode("ascii"),
            "audio_duration_ms": _duration_ms(samples),
            "spectrogram": _spectrogram_payload(samples),
        }), 200

    # --- books & synthesis -----------------------------------------------------------

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
        phonemes_only = bool(data.get("phonemes_only", False))

        wav_bytes, skipped, units = synth.synthesise(sentence, phonemes_only=phonemes_only)
        spec_payload = None
        if wav_bytes is not None:
            samples, _ = audio.read_wav(io.BytesIO(wav_bytes))
            spec_payload = _spectrogram_payload(samples)

        log.info("synthesize: %r phonemes_only=%s resolved=%s skipped=%s",
                 sentence, phonemes_only, "yes" if wav_bytes else "no", skipped)
        return jsonify({
            "audio": base64.b64encode(wav_bytes).decode("ascii") if wav_bytes else None,
            "skipped": skipped,
            "units": units,
            "spectrogram": spec_payload,
        }), 200

    return flask_app


if __name__ == "__main__":
    create_app().run(debug=True, port=5000)
