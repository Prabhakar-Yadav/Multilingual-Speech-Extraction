"""
LLM-as-a-Verifier (core-method replication) applied to VaultStack.

Instead of asking an LLM for a discrete 1-5 judgement, we read the LOGITS over the
scoring digits and take their EXPECTATION -> a continuous, calibrated score. This is
the paper's central idea. We implement its three scaling axes:
  1. GRANULARITY        : 0-9 scale (finer than yes/no) for better separation
  2. REPEATED EVALUATION: score each item several times (paraphrased) and average -> less variance
  3. CRITERIA DECOMPOSITION: split "is it grounded?" into sub-questions and combine

Use case: for each medical term the extractor produced, score how strongly the
transcript supports it; DROP low-scoring (hallucinated) terms.

    .\.venv-train\Scripts\python.exe -u Bengali_ASR\verifier.py
"""
import json, torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

VS = Path(__file__).resolve().parents[2]
VERIFIER = str(VS / "Hindi" / "models" / "base-3b-4bit")   # Llama-3.2-3B-Instruct 4-bit as the verifier
MAXSCORE = 9
THRESH = 4.5

tok = AutoTokenizer.from_pretrained(VERIFIER)
model = AutoModelForCausalLM.from_pretrained(VERIFIER, device_map={"": 0}, dtype=torch.bfloat16)
model.eval()
DIGIT_IDS = [tok.encode(str(d), add_special_tokens=False)[-1] for d in range(MAXSCORE + 1)]

def _expected_score(question):
    """The paper's core trick: expectation over the scoring-digit logits -> continuous score."""
    msgs = [{"role": "system", "content": "You are a strict grounding verifier. Reply with ONE digit 0-9 only."},
            {"role": "user", "content": question}]
    enc = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True).to(model.device)
    with torch.no_grad():
        logits = model(**enc).logits[0, -1]
    probs = torch.softmax(logits[DIGIT_IDS].float(), dim=-1)
    return float(sum(p * i for i, p in enumerate(probs)))

def verify_term(transcript, term, category):
    cat = category[:-1] if category.endswith("s") else category
    # CRITERIA DECOMPOSITION (2 sub-questions) x REPEATED EVALUATION (paraphrases)
    questions = [
        f'Transcript:\n"""{transcript}"""\n\nScore 0-9: is the {cat} "{term}" explicitly mentioned in this transcript? '
        f"(0 = not present at all, 9 = clearly stated). Score:",
        f'In the conversation below, how strongly is "{term}" ({cat}) actually discussed about the patient, '
        f'not just guessed? 0-9.\nConversation: {transcript}\nScore:',
    ]
    scores = [_expected_score(q) for q in questions]
    return sum(scores) / len(scores), scores

src = Path(__file__).resolve().parent / "results_final.json"
res = json.loads(src.read_text(encoding="utf-8"))
for r in res:
    transcript = r.get("english_translation", "")          # verify against English (verifier is English-strong)
    terms = json.loads(r["medical_terms"])
    verified = {k: [] for k in terms}
    print(f"\n===== {r['file']} =====", flush=True)
    for cat, items in terms.items():
        for t in items:
            score, subs = verify_term(transcript, t, cat)
            keep = score >= THRESH
            print(f"  [{cat:10}] {t:22} score={score:.2f}  {'KEEP' if keep else 'DROP (hallucinated)'}", flush=True)
            if keep:
                verified[cat].append(t)
    r["verified_terms"] = json.dumps(verified, ensure_ascii=False)
    print(f"  -> kept: {r['verified_terms']}", flush=True)

(Path(__file__).resolve().parent / "results_verified.json").write_text(
    json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
print("\nsaved -> results_verified.json", flush=True)
