"""Synthetic BENGALI clinic follow-up transcripts -> exact English medical-term labels.

v4 fixes (measured against the real client calls):
  1. MULTI-SURFACE vocab: each English term has MANY real Bengali forms.
     The v3 misses were ALL surface mismatches:
       root canal  -> real speech says আরসিটি, we only taught রুট ক্যানাল
       eye drops   -> real speech says ড্রপ,   we only taught চোখের ড্রপ
       joint pain  -> real speech says জয়েন্টের ব্যথা, we taught জয়েন্টে ব্যথা
  2. Procedures appear in BILLING-COMPLAINT context (like the real calls:
     "why ₹450 to pull a tooth / three cards?"), not only "doctor advised X".
  3. 40% NEGATIVE calls (fees/feedback/follow-up only) -> empty JSON.

    python Bengali/gen_data_bengali.py [n]
"""
import sys, json, random
from pathlib import Path

OUT = Path(__file__).resolve().parent / "data"
OUT.mkdir(parents=True, exist_ok=True)

SYS = ("You are a clinical NLP extractor for doctor-patient call transcripts (Bengali). "
       "Extract ONLY medical terms actually present. Return STRICT JSON with keys: "
       "medicines, diagnoses, procedures, tests, symptoms - arrays of standard ENGLISH "
       "terms. Empty arrays if none. Do NOT invent terms.")

# english label -> [real Bengali surface forms actually heard in speech]
MED = {
    "eye drops": ["চোখের ড্রপ", "ড্রপ", "আই ড্রপ", "ড্রপটা"],
    "ear drops": ["কানের ড্রপ"],          # bare "ড্রপ" is ambiguous -> reserve it for eye drops
    "Paracetamol": ["প্যারাসিটামল", "ডোলো"],
    "cough syrup": ["কাশির সিরাপ", "সিরাপ"],
    "painkiller": ["ব্যথার ওষুধ", "পেইনকিলার"],
    "ointment": ["মলম", "অয়েন্টমেন্ট"],
    "antibiotic": ["অ্যান্টিবায়োটিক"],
    "Azithromycin": ["অ্যাজিথ্রোমাইসিন"],
    "Amoxicillin": ["অ্যামোক্সিসিলিন"],
    "Metformin": ["মেটফরমিন"],
    "Omeprazole": ["ওমিপ্রাজল", "প্যান্টোপ্রাজল"],
    "Insulin": ["ইনসুলিন"],
    "calcium": ["ক্যালসিয়াম", "ক্যালসিয়াম ট্যাবলেট"],
    "vitamin": ["ভিটামিন"],
}
DIAG = {
    "paralysis": ["প্যারালাইসিস", "প্যারালিসিস", "পক্ষাঘাত", "প্যারালাইসিস টাইপের"],
    "cataract": ["ছানি", "চোখে ছানি"],
    "bone decay": ["হাড় ক্ষয়", "হাড়ের ক্ষয়"],
    "tooth infection": ["দাঁতের সংক্রমণ", "দাঁতে ইনফেকশন"],
    "gum infection": ["মাড়ির সংক্রমণ", "মাড়িতে ইনফেকশন"],
    "diabetes": ["ডায়াবেটিস", "সুগার"],
    "high blood pressure": ["উচ্চ রক্তচাপ", "প্রেসার", "হাই প্রেসার"],
    "asthma": ["হাঁপানি", "অ্যাজমা"],
    "arthritis": ["বাত", "আর্থ্রাইটিস"],
    "gastritis": ["গ্যাস্ট্রাইটিস", "গ্যাসের সমস্যা"],
    "anemia": ["রক্তাল্পতা"],
    "migraine": ["মাইগ্রেন"],
    "heart disease": ["হৃদরোগ", "হার্টের সমস্যা"],
}
PROC = {
    "tooth extraction": ["দাঁত তোলা", "দাঁত তুলে ফেলা", "দাঁত তোলানো", "দাঁতটা তোলা"],
    "root canal": ["আরসিটি", "রুট ক্যানাল", "আর সি টি", "দাঁতের আরসিটি"],
    "dental crown": ["ক্যাপ বসানো", "ক্যাপ", "দাঁতে ক্যাপ বসানো"],
    "dental bridge": ["দাঁত বাঁধানো"],
    "physiotherapy": ["ফিজিওথেরাপি", "ফিজিও"],
    "injection": ["ইনজেকশন", "ইঞ্জেকশন"],
    "eye surgery": ["চোখের অপারেশন", "চোখে অপারেশন"],
    "cataract surgery": ["ছানি অপারেশন"],
    "scaling": ["স্কেলিং", "দাঁত স্কেলিং"],
    "wound dressing": ["ড্রেসিং"],
    "nebulization": ["নেবুলাইজেশন"],
    "stitches": ["সেলাই"],
}
SYMP = {
    "joint pain": ["জয়েন্টের ব্যথা", "জয়েন্টে ব্যথা", "গাঁটে ব্যথা", "জয়েন্টের ব্যথা যন্ত্রণা"],
    "toothache": ["দাঁতে ব্যথা", "দাঁতের ব্যথা"],
    "eye pain": ["চোখে ব্যথা"],
    "blurred vision": ["ঝাপসা দেখা", "চোখে ঝাপসা"],
    "fever": ["জ্বর"],
    "cough": ["কাশি"],
    "chest pain": ["বুকে ব্যথা"],
    "headache": ["মাথাব্যথা", "মাথা ব্যথা"],
    "weakness": ["দুর্বলতা"],
    "vomiting": ["বমি"],
    "shortness of breath": ["শ্বাসকষ্ট"],
    "stomach pain": ["পেটে ব্যথা", "পেট ব্যথা"],
    "dizziness": ["মাথা ঘোরা"],
    "back pain": ["কোমরে ব্যথা"],
    "leg pain": ["পায়ে ব্যথা"],
    "swelling": ["ফোলা", "ফুলে যাওয়া"],
    "loss of appetite": ["খিদে না পাওয়া"],
    "sore throat": ["গলা ব্যথা"],
}
TEST = {
    "blood test": ["রক্ত পরীক্ষা", "ব্লাড টেস্ট"],
    "X-ray": ["এক্স-রে"],
    "eye test": ["চোখ পরীক্ষা", "চোখের পরীক্ষা"],
    "dental X-ray": ["দাঁতের এক্স-রে"],
    "blood sugar test": ["সুগার টেস্ট", "ব্লাড সুগার টেস্ট"],
    "urine test": ["প্রস্রাব পরীক্ষা"],
    "ECG": ["ইসিজি"],
    "ultrasound": ["আল্ট্রাসাউন্ড"],
    "CT scan": ["সিটি স্ক্যান"],
}

NAMES = ["দাস বাবু", "খাতুন", "রায় বাবু", "নস্কর", "শেখ সাহেব", "বেগম", "মণ্ডল", "সিনহা", "ঘোষ বাবু"]
LOCS = ["সীতাপুর মার্কেটে", "গোচরণ মোড়ে", "বাজারের ব্যাংকের নিচে"]   # short (Bengali = 3-4 tokens/word)
FILLER = ["হ্যাঁ", "হ্যাঁ হ্যাঁ", "আচ্ছা", "ঠিক আছে", "দেখুন", "হুম"]

# procedures discussed INSIDE a billing complaint - exactly how the real calls talk
BILL = [
    "{p} করার পরে সাড়ে চারশো টাকা নিয়েছে, বোর্ডে তো ১৫০ টাকা লেখা আছে।",
    "{p} এর জন্য আলাদা খরচ আছে, ১৫০ টাকার টিকিটের মধ্যে হয় না।",
    "{p} করাতে গিয়ে অনেক টাকা খরচ হয়ে গেল, আগে জানতাম না।",
    "{p} এর ফি নিয়ে একটু খারাপ লেগেছে, তিনটে কার্ড করতে হলো।",
]
CHAT = [   # non-medical chatter - the real calls are full of this
    "এখানকার ডাক্তার আর স্টাফের ব্যবহার কেমন ছিল? খুব ভালো ব্যবহার।",
    "বাড়ির বাকি সবাই কেমন আছেন? সবাই ভালো আছে।",
    "১৫০ টাকার টিকিটে ডাক্তার দেখানো আর ওষুধ দুটোই হয়ে যায়।",
    "পরের মাসে আবার ক্যাম্প আছে, চলে আসবেন।",
    "সময় দেওয়ার জন্য ধন্যবাদ, ভালো থাকবেন।",
]
NEG = [
    "এখন শরীর কেমন? একদম ভালো আছি, কোনো অসুবিধা নেই।",
    "শরীর এখন ঠিক আছে তো? হ্যাঁ, অনেকটা ভালো।",
    "আগের ওষুধই চালিয়ে যাবেন, নতুন করে কিছু লাগবে না।",
    "ফি নিয়ে চিন্তা নেই, খুব কম খরচ।",
    "উনি এখন বাড়িতে নেই, পরে ফোন করবেন।",
]


def keys(d, lo, hi):
    return random.sample(list(d.keys()), random.randint(lo, hi))
def surf(d, ks):
    return [random.choice(d[k]) for k in ks]


def build(meds, diags, procs, tests, symps):
    name, loc = random.choice(NAMES), random.choice(LOCS)
    parts = [f"নমস্কার, {name} এর বাড়ি? {loc} ডাক্তারখানা থেকে ফলো-আপ কল।"]
    if diags:
        parts.append(f"কিছুদিন আগে আপনি {', '.join(surf(DIAG, diags))} এর জন্য ডাক্তার দেখিয়েছিলেন। এখন কেমন আছেন?")
    else:
        parts.append("কিছুদিন আগে আপনি ডাক্তার দেখিয়েছিলেন। এখন শরীর কেমন?")
    parts.append(f"এখনও একটু {', আর '.join(surf(SYMP, symps))} আছে।" if symps else "এখন অনেকটা ভালো আছি।")
    if procs:
        ps = surf(PROC, procs)
        if random.random() < 0.55:
            parts.append(random.choice(BILL).format(p=ps[0]))
            if len(ps) > 1:
                parts.append(f"{', '.join(ps[1:])} এর কথাও ডাক্তারবাবু বলেছিলেন।")
        else:
            parts.append(f"ডাক্তারবাবু আপনার {', '.join(ps)} করেছিলেন।")
    if meds:
        parts.append(f"{', '.join(surf(MED, meds))} ঠিকমতো চালিয়ে যাচ্ছেন তো?")
    if tests:
        parts.append(f"পরের বার আসার আগে {', আর '.join(surf(TEST, tests))} করিয়ে নেবেন।")
    parts += random.sample(CHAT, random.randint(0, 1))     # trimmed: keep sequences short
    mid = parts[1:]
    for i in range(len(mid)):
        if random.random() < 0.25:
            mid[i] = f"{random.choice(FILLER)}, {mid[i]}"
    random.shuffle(mid)
    return " ".join([parts[0]] + mid)


def build_negative():
    name, loc = random.choice(NAMES), random.choice(LOCS)
    open_ = f"নমস্কার, {name} এর বাড়ি? {loc} ডাক্তারখানা থেকে ফলো-আপ কল।"
    mid = random.sample(NEG, random.randint(1, 2)) + random.sample(CHAT, random.randint(0, 1))
    random.shuffle(mid)
    return " ".join([open_] + mid)


def make_rows(n, seed=11):
    random.seed(seed)
    rows = []
    for _ in range(n):
        if random.random() < 0.40:                       # 40% negative -> empty
            meds = diags = procs = tests = symps = []
            t = build_negative()
        else:
            meds = keys(MED, 0, 2) if random.random() < 0.5 else []
            diags = keys(DIAG, 0, 1) if random.random() < 0.55 else []
            procs = keys(PROC, 0, 2) if random.random() < 0.6 else []   # procedure-heavy, like real calls
            tests = keys(TEST, 0, 1) if random.random() < 0.4 else []
            symps = keys(SYMP, 0, 2) if random.random() < 0.65 else []
            if not (meds or diags or procs or tests or symps):
                procs = keys(PROC, 1, 1)
            t = build(meds, diags, procs, tests, symps)
        target = json.dumps({"medicines": meds, "diagnoses": diags, "procedures": procs,
                             "tests": tests, "symptoms": symps}, ensure_ascii=False)
        rows.append({"messages": [{"role": "system", "content": SYS},
                                  {"role": "user", "content": t},
                                  {"role": "assistant", "content": target}]})
    return rows


def write_test_set(n=400, seed=4321):
    rows = make_rows(n, seed=seed)
    (OUT / "test400.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")


if __name__ == "__main__":
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 1200
    rows = make_rows(N, seed=11)
    random.shuffle(rows)
    val = rows[: max(1, N // 10)]
    train = rows[max(1, N // 10):]
    (OUT / "train.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in train), encoding="utf-8")
    (OUT / "val.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in val), encoding="utf-8")
    print(f"wrote {len(train)} train + {len(val)} val (Bengali v4, multi-surface)")
    print("\n--- sample ---\n", rows[0]["messages"][1]["content"][:300])
    print(rows[0]["messages"][2]["content"])
