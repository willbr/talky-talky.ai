# talky-talky.ai

Basic concatenative text-to-speech in Python.

Concatenative TTS synthesises speech by stitching together small pre-recorded
audio units (phonemes, diphones, or words) rather than generating audio with a
neural model. It is the pre-deep-learning approach — simple, transparent, and
characterful.

## Pipeline

1. **Record** — capture an inventory of speech units from a human voice.
2. **Select** — for a target sentence, pick the best matching unit for each
   position.
3. **Stitch** — concatenate the selected audio and smooth the joins.

## Target text

Synthesis targets are drawn from public-domain books on
[gutenberg.org](https://www.gutenberg.org):

- Treasure Island
- The Iliad — Homer
- Alice's Adventures in Wonderland
- The Wonderful Wizard of Oz
- Peter Pan
- The Wind in the Willows
- The Wisdom of Father Brown
- Moby Dick
- Pride and Prejudice — Jane Austen
- Frankenstein

The books are *what we synthesise*, not what we record. The recording inventory
is a much smaller seed corpus designed to cover the unit set.

## Roadmap

1. Flask web UI for recording the inventory.
2. Flask web UI for synthesising target text.

## Getting started

Requires [uv](https://docs.astral.sh/uv/) and a modern desktop browser (Chrome
or Firefox) that supports the Web Audio API and AudioWorklet.

```powershell
uv sync --group dev
uv run flask --app app run
```

Open `http://127.0.0.1:5000`. Record a few words, then type a sentence to
synthesise it.

Run tests with `uv run pytest`.

## Phoneme fallback

If a sentence contains a word you haven't recorded directly, the synthesiser
tries to stitch it from individually-recorded phoneme samples. The bundled
[CMU Pronouncing Dictionary](data/cmudict.dict) provides the
grapheme-to-phoneme mapping. The phoneme inventory has 39 ARPABET symbols,
each shown as a chip below the record/synthesise panes — click any unrecorded
(red) chip to record that phoneme.

A word is synthesised via phonemes only if **every** required phoneme is
recorded. Words missing one or more phonemes, or words not in the dictionary,
are listed as skipped under the synthesise pane.

## Phoneme extraction

Recording a word also offers to chop it into per-phoneme slices. After you
save a word, an editor appears showing the waveform with auto-proposed
phoneme boundaries (duration-weighted from CMUdict's phoneme sequence). Drag
the boundary lines to align them with what you actually said, click ▶ to
preview each slice, untick any you don't want, then click *Save phonemes* —
the selected slices are written to the phoneme inventory.

The proposed boundaries are crude (linguistic priors only — no acoustic
alignment). Expect to drag every boundary by hand for usable phoneme audio.
The ✎ icon on each word chip re-opens the editor later.

## todo

- allow multiple copies of words
- allow multiple copies of phonemes
- /init
- use https://github.com/tinygrad/tinygrad
- use tinygrad webgpu
- synthesize phonemes
- text -> wav -> ai to tidy wav up -> new wav
- text -> ai -> wav
- do we really need all these tests?

