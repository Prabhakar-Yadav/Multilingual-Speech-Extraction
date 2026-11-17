"""Detailed band-energy analysis to decide if the audio is salvageable.

If most energy is low-frequency RUMBLE/HUM (<300 Hz) but there is still some
energy in the speech band (300-3400 Hz), a high-pass + normalize can help.
If the speech band is truly empty, no cleaning can recover the words.
"""
import sys
import numpy as np
from faster_whisper.audio import decode_audio

path = sys.argv[1] if len(sys.argv) > 1 else "1.mp3.mpeg"
sr = 16000
x = np.asarray(decode_audio(path, sampling_rate=sr), dtype=np.float32)

spec = np.abs(np.fft.rfft(x * np.hanning(len(x)))) ** 2
freqs = np.fft.rfftfreq(len(x), 1 / sr)
total = spec.sum() + 1e-12

bands = [(0, 100), (100, 300), (300, 1000), (1000, 2000),
         (2000, 3400), (3400, 4000), (4000, 8000)]
print(f"file: {path}   duration: {len(x)/sr:.1f}s")
print("band-energy distribution:")
for lo, hi in bands:
    e = spec[(freqs >= lo) & (freqs < hi)].sum() / total * 100
    bar = "#" * int(e / 2)
    print(f"  {lo:>4}-{hi:<4} Hz : {e:5.1f}%  {bar}")

speech_band = spec[(freqs >= 300) & (freqs < 3400)].sum() / total * 100
rumble = spec[freqs < 300].sum() / total * 100
print(f"\nrumble (<300Hz)         : {rumble:.1f}%")
print(f"speech band (300-3400Hz): {speech_band:.1f}%")
if rumble > 70 and speech_band > 5:
    print(">> Likely RECOVERABLE: low-freq noise dominates but speech band has energy. Try high-pass + normalize.")
elif speech_band < 2:
    print(">> Likely NOT recoverable: speech band is essentially empty.")
else:
    print(">> Borderline: cleaning may help a little; expect partial results.")
