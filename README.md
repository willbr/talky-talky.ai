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

Not yet runnable. Setup instructions will land with the first Flask app.
