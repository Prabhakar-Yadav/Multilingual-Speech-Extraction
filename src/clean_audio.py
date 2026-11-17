"""Clean low-quality audio before ASR.

Removes <250 Hz rumble/hum (which held ~31% of this file's energy and masks the
real speech), then normalizes. Cannot recreate the missing 1-3.4 kHz consonant
band, but improves signal-to-noise for whatever speech is present.

    python src/clean_audio.py <in_audio> <out.wav>
"""
import sys
import wave
import numpy as np
from faster_whisper.audio import decode_audio

inp = sys.argv[1]
out = sys.argv[2] if len(sys.argv) > 2 else "outputs/cleaned.wav"
sr = 16000

x = np.asarray(decode_audio(inp, sampling_rate=sr), dtype=np.float32)

# FFT high-pass: zero everything below 250 Hz (rumble/hum/DC).
X = np.fft.rfft(x)
freqs = np.fft.rfftfreq(len(x), 1 / sr)
X[freqs < 250] = 0.0
y = np.fft.irfft(X, n=len(x)).astype(np.float32)

# Peak-normalize to make the quiet speech as loud as possible without clipping.
peak = float(np.max(np.abs(y))) + 1e-9
y = (y / peak) * 0.95

pcm = (y * 32767.0).astype(np.int16)
with wave.open(out, "wb") as w:
    w.setnchannels(1)
    w.setsampwidth(2)
    w.setframerate(sr)
    w.writeframes(pcm.tobytes())

print(f"wrote {out}  ({len(y)/sr:.1f}s, 16kHz mono, rumble<250Hz removed, normalized)")
