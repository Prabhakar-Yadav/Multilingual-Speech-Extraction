"""Extract medical terms from a transcript using a Groq-hosted LLM (Llama 3.3 70B)
— much stronger than the local NER on noisy ASR text, and it won't fragment names.

Needs GROQ_API_KEY.   python src/extract_groq.py <in.json|txt> <out.json> [api_key]
The prompt forbids inventing drug names — empty arrays if nothing is clearly present.
"""
import os, sys, json
from pathlib import Path
from groq import Groq

inp = sys.argv[1]
out = sys.argv[2] if len(sys.argv) > 2 else "outputs/groq_terms.json"
key = (sys.argv[3] if len(sys.argv) > 3 else os.environ.get("GROQ_API_KEY", "")).strip()
if not key:
    sys.exit("ERROR: set GROQ_API_KEY or pass key as 3rd arg.")

raw = Path(inp).read_text(encoding="utf-8")
text = json.loads(raw).get("english_text", raw) if inp.endswith(".json") else raw

SYS = ("You are a clinical NLP extractor. The input is a (possibly noisy, "
       "machine-transcribed) Bengali->English doctor-patient phone conversation. "
       "Extract ONLY medical terms that are genuinely present. Return STRICT JSON "
       "with keys: medicines, diagnoses, procedures, tests, symptoms (arrays of "
       "strings). Normalize obvious drug names to their standard spelling. If the "
       "text is too garbled or a field has nothing, use an empty array. NEVER "
       "invent or guess drug names that are not clearly indicated.")

client = Groq(api_key=key)
resp = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[{"role": "system", "content": SYS}, {"role": "user", "content": text}],
    response_format={"type": "json_object"}, temperature=0.0)
terms = json.loads(resp.choices[0].message.content)
Path(out).write_text(json.dumps({"engine": "groq/llama-3.3-70b", "terms": terms},
                     ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(terms, ensure_ascii=False, indent=2))
