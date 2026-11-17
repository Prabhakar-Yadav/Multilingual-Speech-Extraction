"""Gazetteer + fuzzy matcher: catches drug/test names even when ASR/translation
mangles them (NER needs clean words; this doesn't).
    python src/extract_fuzzy.py <translation.json> <out.json>
"""
import sys, json, re
from pathlib import Path
from rapidfuzz import fuzz

MEDS = ["paracetamol","azithromycin","amoxicillin","metformin","amlodipine","atenolol",
"omeprazole","pantoprazole","cetirizine","montelukast","salbutamol","ibuprofen","diclofenac",
"aspirin","atorvastatin","losartan","telmisartan","insulin","glimepiride","ciprofloxacin",
"ofloxacin","metronidazole","doxycycline","cefixime","ranitidine","domperidone","ondansetron",
"levothyroxine","prednisolone","dexamethasone","calcium","vitamin d","iron","folic acid",
"clopidogrel","ramipril","frusemide","spironolactone","alprazolam","gabapentin"]
TESTS = ["x-ray","blood test","ecg","ultrasound","ct scan","mri","blood sugar","blood pressure",
"urine test","endoscopy","biopsy","hemoglobin","thyroid test","lipid profile"]
DIAG = ["pneumonia","diabetes","hypertension","asthma","tuberculosis","typhoid","dengue",
"malaria","anemia","arthritis","migraine","gastritis","ulcer","thyroid","cholesterol",
"fever","cough","cold","diarrhea","vomiting","jaundice","stroke","heart attack","cataract",
"dental caries","toothache","infection"]
PROC = ["surgery","operation","stitches","dressing","injection","vaccination","dialysis",
"chemotherapy","root canal","extraction","implant","angioplasty","bypass"]

def match(text, vocab, thr=85):
    text = text.lower()
    words = re.findall(r"[a-z]+(?: [a-z]+)?", text)
    hits = []
    for v in vocab:
        if v in text:
            hits.append((v, 100)); continue
        best = max((fuzz.ratio(v, w) for w in words if abs(len(w)-len(v)) <= 3), default=0)
        if best >= thr:
            hits.append((v, int(best)))
    return [{"term": t, "confidence": c} for t, c in sorted(hits, key=lambda x: -x[1])]

inp, out = sys.argv[1], sys.argv[2]
d = json.loads(Path(inp).read_text(encoding="utf-8"))
en = d.get("english_text", "")
res = {"medicines": match(en, MEDS), "tests": match(en, TESTS),
       "diagnoses": match(en, DIAG), "procedures": match(en, PROC)}
Path(out).write_text(json.dumps({"engine": "fuzzy-gazetteer", "terms": res},
                     ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(res, ensure_ascii=False))
