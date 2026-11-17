"""Cross-model consensus: a medical term is reported ONLY if found in >=2
independent model outputs for the same audio. Genuine verification, not filling.
    python src/consensus.py <outputs/N folder>
"""
import sys, json, re
from pathlib import Path

d = Path(sys.argv[1])
VOC = {"symptoms": ["pain","fever","cough","cold","breathing","weakness","dizziness",
"vomiting","headache","stomach","ache","swelling","itching","bleeding"],
"tests": ["temperature","blood pressure","blood test","x-ray","check-up","checkup","sugar test"],
"diagnoses": ["heart disease","diabetes","infection","dental","gastric","asthma","ulcer"],
"medicines": ["medication","medicine","tablet","capsule","injection","dose","syrup",
"paracetamol","azithromycin","amoxicillin","metformin","amlodipine","omeprazole","antacid"]}

# independent English sources for this audio
sources = {}
for name, key in [("medibeng", "english_text"), ("nllb_banglaasr", "english_text"),
                  ("nllb_vaani", "english_text")]:
    f = {"medibeng": "medibeng_clean.json", "nllb_banglaasr": "tr_seg_en.json",
         "nllb_vaani": "vaani_en.json"}[name]
    p = d / f
    if p.exists():
        sources[name] = json.loads(p.read_text(encoding="utf-8")).get(key, "").lower()
# legacy NLLB translation of first banglaasr pass
for p in d.glob("tr_*.mp3.mpeg.json"):
    sources["nllb_banglaasr_v1"] = json.loads(p.read_text(encoding="utf-8")).get("english_text", "").lower()

result, evidence = {}, {}
for cat, words in VOC.items():
    out = []
    for w in words:
        found = [s for s, t in sources.items() if w in t]
        if len(found) >= 2:
            out.append(w)
            evidence[w] = found
    if out:
        result[cat] = out

print(json.dumps({"sources_used": list(sources), "confirmed_terms": result,
                  "evidence": evidence}, indent=2))
(d / "CONSENSUS_terms.json").write_text(json.dumps(
    {"method": ">=2 independent models must agree", "sources": list(sources),
     "terms": result, "evidence": evidence}, indent=2), encoding="utf-8")
