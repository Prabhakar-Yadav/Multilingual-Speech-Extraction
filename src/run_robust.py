"""Robust extraction: re-uses the saved Groq transcripts (Bengali + English) and
feeds BOTH to Llama-3.3-70B with a drug-name-focused prompt. Rationale: Bengali
medical speech writes English drug names phonetically (প্যারাসিটামল=Paracetamol);
the English translation usually drops them, but the Bengali transcript keeps them.

    python src/run_robust.py [api_key]
"""
import os, sys, json
from pathlib import Path
from groq import Groq

B = Path(__file__).resolve().parent.parent
key = (sys.argv[1] if len(sys.argv) > 1 else os.environ.get("GROQ_API_KEY", "")).strip()
if not key:
    sys.exit("ERROR: pass Groq key or set GROQ_API_KEY")
client = Groq(api_key=key)

SYS = (
"You are a clinical NLP extractor for Bengali doctor-patient phone calls. You are given "
"TWO views of the SAME call: a Bengali transcript and its English translation.\n"
"CRITICAL: Bengali medical speech contains English DRUG NAMES written phonetically in "
"Bengali script (e.g. প্যারাসিটামল=Paracetamol, অ্যাজিথ্রোমাইসিন=Azithromycin, "
"ওমিপ্রাজল=Omeprazole, অ্যামলোডিপিন=Amlodipine, মেটফরমিন=Metformin, ওষুধ/অসুধ=medicine). "
"The English translation often DROPS these names. Scan the BENGALI text carefully and "
"recover any drug names to their standard English spelling.\n"
"Return STRICT JSON with keys: medicines, diagnoses, procedures, tests, symptoms. "
"Each entry is an object {term, heard_as, confidence} where heard_as is the source snippet "
"(Bengali or English) and confidence is 0.0-1.0. Use empty arrays if nothing is clearly "
"present. NEVER invent drug names that are not phonetically supported by the text.")

for i in ["1", "2", "3", "4"]:
    p = B / "outputs" / i / "groq_result.json"
    if not p.exists():
        print(f"{i}: no groq_result.json, skip", flush=True); continue
    d = json.loads(p.read_text(encoding="utf-8"))
    user = f"BENGALI TRANSCRIPT:\n{d.get('bengali','')}\n\nENGLISH TRANSLATION:\n{d.get('english_text','')}"
    try:
        resp = client.chat.completions.create(model="llama-3.3-70b-versatile",
            messages=[{"role": "system", "content": SYS}, {"role": "user", "content": user}],
            response_format={"type": "json_object"}, temperature=0)
        terms = json.loads(resp.choices[0].message.content)
        (B / "outputs" / i / "robust_terms.json").write_text(json.dumps(
            {"engine": "groq/llama-3.3-70b · dual-language", "terms": terms},
            ensure_ascii=False, indent=2), encoding="utf-8")
        meds = terms.get("medicines", [])
        print(f"\n=== {i} ===\nMEDICINES: {json.dumps(meds, ensure_ascii=False)}\n"
              f"DIAGNOSES: {json.dumps(terms.get('diagnoses', []), ensure_ascii=False)}", flush=True)
    except Exception as ex:
        print(f"{i}: ERROR {ex}", flush=True)
print("\nALL_DONE", flush=True)
