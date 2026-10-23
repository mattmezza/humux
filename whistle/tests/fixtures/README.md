# Offline STT test sample

`test-stt.ogg` contains real human speech: “zero, one, two”. It is bundled for API and offline container smoke tests; no speech synthesis
or network access is needed to obtain test audio.

Source: [Free Spoken Digit Dataset](https://github.com/Jakobovski/free-spoken-digit-dataset),
by Zohar Jackson and contributors, revision
`26eb9aaf76e81b692f806f9140c2d2777410d7a1`, files
`recordings/{0,1,2}_jackson_0.wav` (speaker: Jackson).
The source recordings and this adapted clip are licensed under
[Creative Commons Attribution-ShareAlike 4.0 International](https://creativecommons.org/licenses/by-sa/4.0/).
This license applies to the audio asset, independently of the application license.

Adaptation: concatenate the three 8 kHz mono recordings in digit order, insert
0.2 seconds of silence after zero and one, resample to 16 kHz, encode as Ogg Opus:

```sh
ffmpeg -i 0_jackson_0.wav -i 1_jackson_0.wav -i 2_jackson_0.wav \
  -filter_complex '[0:a]apad=pad_dur=0.2[a0];[1:a]apad=pad_dur=0.2[a1];[a0][a1][2:a]concat=n=3:v=0:a=1' \
  -ar 16000 -c:a libopus test-stt.ogg
```

Transcription spelling and punctuation vary by model; the UI displays the result
rather than requiring an exact string match.
