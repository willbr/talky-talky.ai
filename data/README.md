# Bundled data

## cmudict.dict

The CMU Pronouncing Dictionary, mirrored from
<https://github.com/cmusphinx/cmudict/blob/master/cmudict.dict>.

Format: one entry per line, `<word> <phoneme1> <phoneme2> …`. Phonemes are
ARPABET with stress digit suffixes (0/1/2). Multiple pronunciations appear
as `word(2)`, `word(3)` etc. — talky-talky's loader keeps only the first
pronunciation per word.

Licensed under a BSD-style permissive license; see the upstream repository
for the canonical text.
