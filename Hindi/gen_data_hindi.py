"""PRACTICE: synthetic HINDI doctor-patient follow-up transcripts (Devanagari) with
EXACT English medical-term labels. Same technique as the Bengali/English Phase-1
fine-tune, but in Hindi - to validate the fine-tuning workflow. Local, no PHI.

    python Hindi/gen_data_hindi.py [n]
"""
import sys, json, random
from pathlib import Path

OUT = Path(__file__).resolve().parent / "data"
OUT.mkdir(parents=True, exist_ok=True)

SYS = ("You are a clinical NLP extractor for doctor-patient call transcripts (Hindi). "
       "Extract ONLY medical terms actually present. Return STRICT JSON with keys: "
       "medicines, diagnoses, procedures, tests, symptoms - arrays of standard ENGLISH "
       "terms. Empty arrays if none. Do NOT invent terms.")

# (Hindi surface form, English label)
MED = [("पैरासिटामोल", "Paracetamol"), ("एज़िथ्रोमाइसिन", "Azithromycin"),
    ("एमोक्सिसिलिन", "Amoxicillin"), ("मेटफॉर्मिन", "Metformin"), ("एमलोडिपिन", "Amlodipine"),
    ("ओमेप्राज़ोल", "Omeprazole"), ("सेटिरिज़िन", "Cetirizine"), ("इबुप्रोफेन", "Ibuprofen"),
    ("एटोरवास्टेटिन", "Atorvastatin"), ("इंसुलिन", "Insulin"), ("सिप्रोफ्लोक्सासिन", "Ciprofloxacin"),
    ("डोलो", "Paracetamol"), ("पैन्टोप्राज़ोल", "Pantoprazole"), ("लोसार्टन", "Losartan")]
DIAG = [("निमोनिया", "pneumonia"), ("मधुमेह", "diabetes"), ("उच्च रक्तचाप", "high blood pressure"),
    ("दमा", "asthma"), ("दांत का संक्रमण", "tooth infection"), ("गैस्ट्राइटिस", "gastritis"),
    ("एनीमिया", "anemia"), ("गठिया", "arthritis"), ("माइग्रेन", "migraine"),
    ("टाइफाइड", "typhoid"), ("डेंगू", "dengue"), ("थायराइड", "thyroid disorder"),
    ("किडनी स्टोन", "kidney stone"), ("दिल की बीमारी", "heart disease")]
SYMP = [("बुखार", "fever"), ("खांसी", "cough"), ("सीने में दर्द", "chest pain"),
    ("सिरदर्द", "headache"), ("दांत में दर्द", "toothache"), ("जोड़ों में दर्द", "joint pain"),
    ("कमजोरी", "weakness"), ("उल्टी", "vomiting"), ("सांस लेने में तकलीफ", "shortness of breath"),
    ("पेट दर्द", "stomach pain"), ("चक्कर आना", "dizziness"), ("पैर में दर्द", "leg pain"),
    ("कमर दर्द", "back pain"), ("गले में खराश", "sore throat"), ("भूख न लगना", "loss of appetite")]
PROC = [("रूट कैनाल", "root canal"), ("दांत निकालना", "tooth extraction"),
    ("फिजियोथेरेपी", "physiotherapy"), ("छोटी सर्जरी", "minor surgery"),
    ("ड्रेसिंग", "wound dressing"), ("इंजेक्शन", "injection"), ("डायलिसिस", "dialysis"),
    ("टांके", "stitches"), ("नेबुलाइज़ेशन", "nebulization"), ("बायोप्सी", "biopsy")]
TEST = [("खून की जांच", "blood test"), ("एक्स-रे", "X-ray"), ("सीबीसी", "complete blood count"),
    ("ब्लड शुगर टेस्ट", "blood sugar test"), ("यूरिन टेस्ट", "urine test"), ("ईसीजी", "ECG"),
    ("अल्ट्रासाउंड", "ultrasound"), ("एमआरआई", "MRI"), ("थायराइड टेस्ट", "thyroid test"),
    ("सीटी स्कैन", "CT scan")]
NAMES = ["शर्मा जी", "गुप्ता जी", "रीना देवी", "अहमद साहब", "वर्मा जी", "सुनीता जी", "रमेश जी"]
FILLER = ["जी", "हाँ हाँ", "अच्छा", "ठीक है", "देखिए", "हम्म"]


def pick(lst, lo, hi):
    return random.sample(lst, random.randint(lo, hi))


def hin(p): return [x[0] for x in p]
def eng(p): return [x[1] for x in p]


def build(meds, diags, procs, tests, symps):
    name = random.choice(NAMES)
    parts = [random.choice([
        f"नमस्ते, क्या मैं {name} के घर बात कर रहा हूँ? मैं क्लिनिक से फॉलो-अप के लिए कॉल कर रहा हूँ।",
        f"हैलो, क्या यह {name} हैं? मैं डॉक्टर साहब के यहाँ से फॉलो-अप के लिए बात कर रहा हूँ।"])]
    if diags:
        parts.append(f"कुछ दिन पहले आप {', '.join(hin(diags))} के लिए डॉक्टर को दिखाने आए थे। अब कैसा लग रहा है?")
    else:
        parts.append("कुछ दिन पहले आप डॉक्टर को दिखाने आए थे। अब तबीयत कैसी है?")
    parts.append(f"अभी भी थोड़ा {', और '.join(hin(symps))} है।" if symps else "अब काफी आराम है, कोई दिक्कत नहीं।")
    if meds:
        d = random.choice(["दिन में दो बार", "खाने के बाद", "सुबह-शाम", "पाँच दिन तक"])
        parts.append(f"डॉक्टर ने {', '.join(hin(meds))} {d} लेने को कहा है। दवा जारी रखिए।")
    if procs:
        parts.append(f"हमने आपके लिए {', '.join(hin(procs))} की सलाह दी है।")
    if tests:
        parts.append(f"अगली बार आने से पहले {', और '.join(hin(tests))} करवा लीजिए।")
    parts.append("घर में बाकी सब ठीक हैं? जी हाँ। ठीक है, स्वस्थ रहिए, ज़रूरत हो तो फिर दिखाने आइए।")
    mid = parts[1:-1]
    for i in range(len(mid)):
        if random.random() < 0.3:
            mid[i] = f"{random.choice(FILLER)}, {mid[i]}"
    random.shuffle(mid)
    return " ".join([parts[0]] + mid + [parts[-1]])


def make_rows(n, seed=11):
    random.seed(seed)
    rows = []
    for _ in range(n):
        if random.random() < 0.15:
            meds = diags = procs = tests = symps = []
            t = (f"नमस्ते, क्या मैं {random.choice(NAMES)} के घर बात कर रहा हूँ? क्लिनिक से फॉलो-अप कॉल है। "
                 "अब तबीयत कैसी है? जी, अब बिल्कुल ठीक हूँ, कोई दिक्कत नहीं। अच्छा, घर में सब ठीक? जी हाँ। "
                 "ठीक है, धन्यवाद, स्वस्थ रहिए।")
        else:
            meds = pick(MED, 0, 2) if random.random() < 0.55 else []
            diags = pick(DIAG, 0, 2) if random.random() < 0.7 else []
            procs = pick(PROC, 0, 2) if random.random() < 0.5 else []
            tests = pick(TEST, 0, 2) if random.random() < 0.55 else []
            symps = pick(SYMP, 1, 3)
            t = build(meds, diags, procs, tests, symps)
        target = json.dumps({"medicines": eng(meds), "diagnoses": eng(diags),
                             "procedures": eng(procs), "tests": eng(tests),
                             "symptoms": eng(symps)}, ensure_ascii=False)
        rows.append({"messages": [{"role": "system", "content": SYS},
                                  {"role": "user", "content": t},
                                  {"role": "assistant", "content": target}]})
    return rows


def write_test_set(n=400, seed=4321):
    """Fixed shared test set for model comparison (same samples for every model)."""
    rows = make_rows(n, seed=seed)
    p = OUT / "test400.jsonl"
    p.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
    return p


if __name__ == "__main__":
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 1200
    rows = make_rows(N, seed=11)
    random.shuffle(rows)
    val = rows[: max(1, N // 10)]
    train = rows[max(1, N // 10):]
    (OUT / "train.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in train), encoding="utf-8")
    (OUT / "val.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in val), encoding="utf-8")
    print(f"wrote {len(train)} train + {len(val)} val (Hindi) to {OUT}")
