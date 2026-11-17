"""Torch-free: decode the 4 client mp3 files to 16kHz waveforms, save to pickle
(so the torch transcription process never imports librosa - that segfaults with torch)."""
import pickle, numpy as np, librosa
from pathlib import Path

B = Path(__file__).resolve().parent
DATA = B.parents[1] / "data"
wavs = []
for i in [1, 2, 3, 4]:
    w = librosa.load(str(DATA / f"{i}.mp3.mpeg"), sr=16000)[0].astype(np.float32)
    wavs.append(w)
    print(f"file {i}: {len(w)/16000:.1f}s", flush=True)
(B / "data").mkdir(exist_ok=True)
pickle.dump(wavs, open(B / "data" / "four_wavs.pkl", "wb"))
print("saved four_wavs.pkl", flush=True)
