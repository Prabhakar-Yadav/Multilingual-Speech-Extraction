"""Diagnose audio quality to understand why ASR hallucinates.

Decodes the file to 16 kHz mono and reports loudness, silence ratio, clipping,
and how much energy sits in the high-frequency band. Heavily compressed
(16 kbps) audio loses the high-frequency detail consonants need, which makes
Whisper hallucinate even though it detects the language.
"""
import sys
import numpy as np
from faster_whisper.audio import decode_audio

path = sys.argv[1] if len(sys.argv) > 1 else "1.mp3.mpeg"
sr = 16000
x = decode_audio(path, sampling_rate=sr)          # float32 mono in [-1, 1]
x = np.asarray(x, dtype=np.float32)

dur = len(x) / sr
rms = float(np.sqrt(np.mean(x**2)))
peak = float(np.max(np.abs(x)))

# frame-wise silence ratio (30 ms frames)
fl = int(0.03 * sr)
frames = x[: len(x) // fl * fl].reshape(-1, fl)
fr_rms = np.sqrt(np.mean(frames**2, axis=1) + 1e-12)
silence_ratio = float(np.mean(fr_rms < (rms * 0.2)))

# spectrum: fraction of energy above 3 kHz (consonant band)
spec = np.abs(np.fft.rfft(x * np.hanning(len(x))))
freqs = np.fft.rfftfreq(len(x), 1 / sr)
e_total = float(np.sum(spec**2)) + 1e-12
e_hi = float(np.sum(spec[freqs > 3000] ** 2))
hi_ratio = e_hi / e_total
# effective top frequency: highest freq holding meaningful energy
cum = np.cumsum(spec**2) / e_total
top_freq = float(freqs[np.searchsorted(cum, 0.99)])

print(f"duration        : {dur:.1f}s")
print(f"RMS loudness    : {rms:.4f}  (peak {peak:.3f})")
print(f"silence ratio   : {silence_ratio*100:.0f}% of frames near-silent")
print(f"energy >3kHz    : {hi_ratio*100:.1f}%  (low => muffled, consonants lost)")
print(f"99% energy below: {top_freq:.0f} Hz  (telephone ~3400Hz; speech needs ~8kHz)")
