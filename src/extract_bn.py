"""Crude Bengali->Latin romanizer + fuzzy drug match on raw Bengali transcript.
    python src/extract_bn.py <bn_transcript.json> <out.json>
"""
import sys, json, re
from pathlib import Path
from rapidfuzz import fuzz

C = {'ক':'k','খ':'kh','গ':'g','ঘ':'gh','ঙ':'n','চ':'ch','ছ':'chh','জ':'j','ঝ':'jh','ঞ':'n',
'ট':'t','ঠ':'th','ড':'d','ঢ':'dh','ণ':'n','ত':'t','থ':'th','দ':'d','ধ':'dh','ন':'n',
'প':'p','ফ':'f','ব':'b','ভ':'bh','ম':'m','য':'y','র':'r','ল':'l','শ':'s','ষ':'s','স':'s',
'হ':'h','ড়':'r','ঢ়':'rh','য়':'y','ৎ':'t','অ':'a','আ':'a','ই':'i','ঈ':'i','উ':'u','ঊ':'u',
'এ':'e','ঐ':'oi','ও':'o','ঔ':'ou','া':'a','ি':'i','ী':'i','ু':'u','ূ':'u','ে':'e','ৈ':'oi',
'ো':'o','ৌ':'ou','্':'','ং':'ng','ঃ':'','ঁ':''}
MEDS = ["paracetamol","azithromycin","amoxicillin","metformin","amlodipine","atenolol",
"omeprazole","pantoprazole","cetirizine","montelukast","salbutamol","ibuprofen","diclofenac",
"aspirin","atorvastatin","losartan","insulin","ciprofloxacin","metronidazole","doxycycline",
"cefixime","ranitidine","domperidone","ondansetron","calcium","vitamin","antibiotic",
"antacid","gas er oshudh","painkiller"]

def roman(t):
    return "".join(C.get(ch, ch if ch.isascii() else " ") for ch in t)

inp, out = sys.argv[1], sys.argv[2]
d = json.loads(Path(inp).read_text(encoding="utf-8"))
r = roman(d.get("transcript", "")).lower()
words = re.findall(r"[a-z]{4,}", r)
grams = words + [" ".join(p) for p in zip(words, words[1:])]
hits = []
for v in MEDS:
    best, bw = 0, ""
    for w in grams:
        if abs(len(w) - len(v)) > 4: continue
        s = fuzz.ratio(v, w)
        if s > best: best, bw = s, w
    if best >= 78: hits.append({"drug": v, "heard_as": bw, "confidence": int(best)})
res = sorted(hits, key=lambda x: -x["confidence"])
Path(out).write_text(json.dumps({"engine": "bn-romanized-fuzzy", "medicines": res},
                     ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(res, ensure_ascii=False))
