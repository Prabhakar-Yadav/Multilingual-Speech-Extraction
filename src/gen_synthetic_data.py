"""v2 synthetic generator: richer, messier, more varied doctor-patient follow-up
transcripts (mimics real Whisper-translation output: repetition, filler, informal
phrasing) with EXACT term labels. More procedure/test coverage + partial/empty cases
so the model generalises beyond clean templates. Local, no PHI.

    python src/gen_synthetic_data.py [n] [out_dir]
"""
import sys, json, random
from pathlib import Path

random.seed(7)
N = int(sys.argv[1]) if len(sys.argv) > 1 else 1500
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(__file__).resolve().parent.parent / "data"
OUT.mkdir(parents=True, exist_ok=True)

SYS = ("You are a clinical NLP extractor for doctor-patient call transcripts. "
       "Extract ONLY medical terms actually present in the text. Return STRICT JSON "
       "with exactly these keys: medicines, diagnoses, procedures, tests, symptoms - "
       "each an array of strings. Empty arrays if none. Do NOT invent terms.")

MEDICINES = ["Paracetamol", "Azithromycin", "Amoxicillin", "Metformin", "Amlodipine",
    "Omeprazole", "Pantoprazole", "Cetirizine", "Ibuprofen", "Atorvastatin", "Losartan",
    "Insulin", "Ciprofloxacin", "Metronidazole", "Doxycycline", "Cefixime", "Ranitidine",
    "Domperidone", "Ondansetron", "Levothyroxine", "Prednisolone", "Clopidogrel",
    "Aspirin", "Telmisartan", "Montelukast", "Salbutamol", "Diclofenac", "Folic acid"]
DIAGNOSES = ["pneumonia", "diabetes", "hypertension", "asthma", "tooth infection",
    "gastritis", "urinary tract infection", "anemia", "arthritis", "migraine",
    "bronchitis", "typhoid", "dengue", "high blood pressure", "thyroid disorder",
    "tooth decay", "kidney stone", "heart disease", "ulcer", "sinus infection"]
SYMPTOMS = ["fever", "cough", "chest pain", "headache", "toothache", "joint pain",
    "fatigue", "vomiting", "shortness of breath", "stomach pain", "dizziness",
    "leg pain", "back pain", "sore throat", "nausea", "swelling", "weakness",
    "loss of appetite", "sleeplessness", "blurred vision", "burning urination"]
PROCEDURES = ["root canal", "tooth extraction", "physiotherapy", "minor surgery",
    "wound dressing", "injection", "tooth filling", "cataract surgery", "dialysis",
    "biopsy", "stitches", "plaster cast", "nebulization", "dental cleaning"]
TESTS = ["blood test", "X-ray", "complete blood count", "blood sugar test",
    "urine test", "ECG", "ultrasound", "MRI", "lipid profile", "thyroid test",
    "CT scan", "blood pressure check", "HbA1c test", "stool test"]
NAMES = ["Mr. Das", "Mrs. Khatun", "Mr. Roy", "Naskar", "Mr. Sheikh", "Mrs. Begum",
         "Mr. Mondal", "Mrs. Sinha", "Mr. Ghosh", "the patient", "Rahim", "Sumitra"]
FILLER = ["you know", "okay", "yes yes", "hmm", "I see", "right", "actually", "see"]


def maybe_repeat(s):
    return (s + " " + s) if random.random() < 0.15 else s


def sym_phrase(symps):
    t = random.choice([
        f"I still have {', and '.join(symps)}.",
        f"There is some {', and some '.join(symps)}.",
        f"The patient is complaining of {', '.join(symps)}.",
        f"He says he has {', '.join(symps)} since a few days.",
    ])
    return maybe_repeat(t)


def med_phrase(meds):
    d = random.choice(["once a day", "twice a day", "after meals", "for five days",
                       "in the morning", "at night", "three times a day"])
    return random.choice([
        f"The doctor prescribed {', '.join(meds)} to be taken {d}.",
        f"We have given {' and '.join(meds)}, {d}. Please continue it.",
        f"He was advised {', '.join(meds)}. Take it {d} and come back if it finishes.",
        f"Keep taking the {', '.join(meds)} {d}.",
    ])


def proc_phrase(procs):
    return random.choice([
        f"We have advised {', '.join(procs)} for this.",
        f"The doctor suggested {' and '.join(procs)}.",
        f"You will need {', '.join(procs)}.",
        f"They did {', '.join(procs)} during the visit.",
    ])


def test_phrase(tests):
    return random.choice([
        f"Please get a {', and a '.join(tests)} done before the next visit.",
        f"We have asked for a {', '.join(tests)}.",
        f"The {', and the '.join(tests)} should be done.",
        f"He needs to do a {', '.join(tests)}.",
    ])


def pick(lst, lo, hi):
    return random.sample(lst, random.randint(lo, hi))


def build(meds, diags, procs, tests, symps):
    name = random.choice(NAMES)
    parts = [random.choice([
        f"Hello, am I speaking with {name}'s house? This is a follow-up call from the clinic.",
        f"Hello hello, is this {name}? I am calling from the doctor's chamber for a follow-up.",
        f"Namaskar, am I talking to {name}'s family? This is from the hospital."])]
    if diags:
        parts.append(random.choice([
            f"A few days ago you saw the doctor for {', '.join(diags)}. How are you now?",
            f"You came to us regarding {', and '.join(diags)}. Are you feeling better?",
            f"The doctor had diagnosed {', '.join(diags)}. How is it now?"]))
    else:
        parts.append("A few days ago you visited the doctor. How are you feeling now?")
    if symps:
        parts.append(sym_phrase(symps))
    else:
        parts.append(random.choice(["I am feeling much better now.", "No problems now, all good."]))
    if meds:
        parts.append(med_phrase(meds))
    if procs:
        parts.append(proc_phrase(procs))
    if tests:
        parts.append(test_phrase(tests))
    parts.append(random.choice([
        "Is everyone else at home keeping well? Yes. Okay, stay healthy, come for a follow-up if needed.",
        "Are the others at home fine? Yes, all fine. Good, thank you for your time.",
        "Okay, you come and show again if there is any problem. Stay well."]))
    # sprinkle filler into the middle for realism
    mid = parts[1:-1]
    for i in range(len(mid)):
        if random.random() < 0.3:
            mid[i] = f"{random.choice(FILLER)}, {mid[i]}"
    random.shuffle(mid)
    return " ".join([parts[0]] + mid + [parts[-1]])


rows = []
for i in range(N):
    r = random.random()
    if r < 0.15:                                   # empty call
        meds = diags = procs = tests = symps = []
        transcript = (f"Hello, am I speaking with {random.choice(NAMES)}'s house? Follow-up call from "
                      "the clinic. How are you now? I am completely fine, no problems at all. "
                      "Good, everyone at home well? Yes. Okay, thank you, stay healthy.")
    else:                                          # partial-to-full coverage (varied)
        meds = pick(MEDICINES, 0, 2) if random.random() < 0.55 else []
        diags = pick(DIAGNOSES, 0, 2) if random.random() < 0.7 else []
        procs = pick(PROCEDURES, 0, 2) if random.random() < 0.55 else []
        tests = pick(TESTS, 0, 2) if random.random() < 0.55 else []
        symps = pick(SYMPTOMS, 1, 3)
        transcript = build(meds, diags, procs, tests, symps)
    target = json.dumps({"medicines": meds, "diagnoses": diags, "procedures": procs,
                         "tests": tests, "symptoms": symps}, ensure_ascii=False)
    rows.append({"messages": [
        {"role": "system", "content": SYS},
        {"role": "user", "content": transcript},
        {"role": "assistant", "content": target}]})

random.shuffle(rows)
val = rows[: max(1, N // 10)]
train = rows[max(1, N // 10):]
(OUT / "train.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in train), encoding="utf-8")
(OUT / "val.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in val), encoding="utf-8")
print(f"wrote {len(train)} train + {len(val)} val examples (v2, richer) to {OUT}")
print("\n--- sample transcript ---\n", rows[0]["messages"][1]["content"])
print("--- sample label ---\n", rows[0]["messages"][2]["content"])
