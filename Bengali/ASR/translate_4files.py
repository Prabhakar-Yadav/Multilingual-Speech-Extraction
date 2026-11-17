"""Translate the fine-tuned Bengali transcripts to English with local NLLB-200
(no API - PHI stays on machine). Collapses repetition, chunks long text."""
import json, torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

BASR = Path(__file__).resolve().parent
NLLB = str(BASR.parent / "models" / "nllb-200-distilled-600M")
DEV = "cuda" if torch.cuda.is_available() else "cpu"

def collapse(text):
    out = []
    for w in text.split():
        if not out or out[-1] != w:
            out.append(w)
    return " ".join(out)

tok = AutoTokenizer.from_pretrained(NLLB, src_lang="ben_Beng")
model = AutoModelForSeq2SeqLM.from_pretrained(NLLB).to(DEV)
eng_id = tok.convert_tokens_to_ids("eng_Latn")

def translate(text):
    words = collapse(text).split()
    chunks = [" ".join(words[i:i + 35]) for i in range(0, len(words), 35)]
    outs = []
    for c in chunks:
        enc = tok(c, return_tensors="pt", truncation=True, max_length=200).to(DEV)
        with torch.no_grad():
            g = model.generate(**enc, forced_bos_token_id=eng_id, max_length=200)
        outs.append(tok.batch_decode(g, skip_special_tokens=True)[0])
    return " ".join(outs)

results = json.loads((BASR / "results_with_terms.json").read_text(encoding="utf-8"))
for r in results:
    r["english_translation"] = translate(r["finetuned_vaani"])
    print(f"===== {r['file']} =====\n{r['english_translation']}\n", flush=True)

(BASR / "results_final.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
print("saved -> results_final.json", flush=True)
