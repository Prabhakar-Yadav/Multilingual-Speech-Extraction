# English ASR + Medical Extraction — on-prem, CPU-only (16 GB)

**Goal:** English doctor–patient audio → transcript → medical terms.
**Constraints:** 100% on-prem at runtime, CPU-only, 16 GB RAM, no data leaves the machine.
(API keys allowed for *training/testing* only, never at runtime.)

---

## Why this is far easier than Bengali

| | Bengali (previous) | **English (this)** |
|---|---|---|
| Best achievable WER | ~24–31% (research ceiling) | **~6%** |
| Training data available | scarce | massive |
| CPU-optimised models | few | many, mature |
| Medical NER models | none | **OpenMed: 2,000+, Apache-2.0** |

We were fighting a genuinely hard language before. English is a solved-ish problem — the
job here is **assembling the right proven parts**, not pushing a research frontier.

---

## Stage 1 — Speech to text

### ✅ Recommended: **NVIDIA Parakeet TDT 0.6B v3** (ONNX / int8)
Beats Whisper on **both** accuracy and CPU speed:

| Model | WER (Open ASR Leaderboard) | CPU speed |
|---|---|---|
| **Parakeet TDT 0.6B v3** | **6.32%** | **RTF 0.033 (~30× real-time)** |
| Whisper large-v3 | 7.44% | ~4× slower on CPU |
| Whisper small | ~8.5% | ~8× real-time |

**Why it's fast:** a transducer decoder avoids Whisper's per-token autoregressive cost.
A 5-minute call transcribes in **~10 seconds** on CPU.

### Backup: **faster-whisper** (`distil-large-v3`, int8)
- CTranslate2 backend, ~2× faster than vanilla Whisper on CPU
- Use if Parakeet's tooling gives trouble, or if multilingual is needed later

### Runtime
`onnxruntime` (int8 quantized) — pure CPU, no GPU, no CUDA.

---

## Stage 2 — Medical term extraction

### ✅ Recommended: **OpenMed** (the release you shared)
- **2,000+ Apache-2.0 clinical models**, ONNX exports for CPU/browser/Android
- **State-of-the-art: best micro-F1 on 10 of 12 biomedical NER benchmarks**
- Covers diseases, chemicals/drugs, genes, anatomy, pharmacology, oncology
- **Plus PHI de-identification: 55+ PHI types** — big for HIPAA/patient privacy
- Runs fully offline, small enough for a phone

**This replaces the LLM extractor entirely.** Instead of an LLM that hallucinates terms
(our Bengali problem: F1 0.42, invented symptoms), a purpose-built NER model *tags spans
in the actual text* — it structurally **cannot invent** a term that isn't there.

### Models to evaluate
| Purpose | Model |
|---|---|
| Disease/condition NER | `OpenMed/OpenMed-NER-*Disease*` |
| Drug/chemical NER | `OpenMed/OpenMed-NER-*Chem*` / pharma |
| Anatomy | `OpenMed/OpenMed-NER-*Anatomy*` |
| PHI de-identification | `OpenMed/OpenMed-PII-*` |

Model browser: https://huggingface.co/spaces/OpenMed/openmed-ner-models

---

## Proposed pipeline

```
audio (wav/mp3)
   │
   ├─▶ [1] Parakeet TDT ONNX int8      → English transcript      (~6% WER, CPU)
   │
   ├─▶ [2] OpenMed PHI de-identify     → strip patient names/IDs  (privacy first)
   │
   └─▶ [3] OpenMed clinical NER        → medicines / conditions / procedures / tests
                                          (span-tagged = cannot hallucinate)
```

**Memory budget (16 GB CPU):**
| Component | RAM |
|---|---|
| Parakeet TDT int8 | ~1.5 GB |
| OpenMed NER (int8) | ~0.3 GB each |
| Runtime + audio buffers | ~2 GB |
| **Total** | **~4 GB** ✅ comfortable |

---

## Lessons carried over from the Bengali work

1. **Always hold out a test set** the model never trains on — and never tune the
   vocabulary by peeking at it (contamination made an earlier 84% meaningless).
2. **WER is an error rate** — lower is better. Report word **and** character accuracy.
3. **Keep every checkpoint**; save the *best*, not the last (we lost a best model to
   `save_total_limit`).
4. **Audio quality dominates.** No model recovers speech that isn't in the signal —
   the client's 4 calls were band-limited (99% energy <1 kHz) and every model failed
   equally. Push for good recordings up front.
5. **Prefer purpose-built extractors over LLMs** for closed vocabularies — no hallucination.

---

## Next steps

1. Benchmark **Parakeet TDT vs faster-whisper** on sample English medical audio (WER + speed)
2. Shortlist the OpenMed NER models for our term categories
3. Wire the 3-stage pipeline, measure end-to-end on held-out audio
4. Define the honest scorecard: ASR WER/CER + NER precision/recall/F1

**Sources:** Parakeet vs Whisper CPU benchmarks (snailtext, localaimaster, modelslab),
OpenMed NER paper (arXiv 2508.01630), OpenMed HF org + GitHub (maziyarpanahi/openmed).
