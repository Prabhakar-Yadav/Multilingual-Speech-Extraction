"""Compare API models on the SAME fixed 400 Hindi test set (OpenAI-compatible).
One model per run; results accumulate in Hindi/compare/summary.csv.

  python Hindi/compare_models.py --provider groq    --model llama-3.3-70b-versatile --key gsk_...
  python Hindi/compare_models.py --provider gemini  --model gemini-2.5-flash       --key ...
  python Hindi/compare_models.py --provider sarvam  --model sarvam-m               --key ...
  python Hindi/compare_models.py --provider openrouter --model anthropic/claude-3.5-haiku --key ...

Via OpenRouter one key reaches GPT / Claude / Gemini / Qwen / Llama. Use --base-url
for any other OpenAI-compatible endpoint.
"""
import os, sys, json, time, csv, argparse, difflib
from pathlib import Path
from openai import OpenAI

B = Path(__file__).resolve().parent
TEST = B / "data" / "test400.jsonl"
OUTDIR = B / "compare"; OUTDIR.mkdir(exist_ok=True)
FIELDS = ["medicines", "diagnoses", "procedures", "tests", "symptoms"]
PRESETS = {
    "openai": "https://api.openai.com/v1",
    "groq": "https://api.groq.com/openai/v1",
    "gemini": "https://generativelanguage.googleapis.com/v1beta/openai/",
    "sarvam": "https://api.sarvam.ai/v1",
    "together": "https://api.together.xyz/v1",
    "fireworks": "https://api.fireworks.ai/inference/v1",
    "openrouter": "https://openrouter.ai/api/v1",
    "mistral": "https://api.mistral.ai/v1",
    "cerebras": "https://api.cerebras.ai/v1",          # free, fast
    "nvidia": "https://integrate.api.nvidia.com/v1",   # free credits
    "github": "https://models.github.ai/inference",    # free (GitHub token)
}
sys.path.insert(0, str(B))
from gen_data_hindi import MED, DIAG, SYMP, PROC, TEST as TVOCAB, eng
_v = lambda lst: ", ".join(sorted(set(eng(lst))))
SYS = ("You are a clinical NLP extractor for Hindi doctor-patient call transcripts. "
       "Extract ONLY medical terms actually present. Return STRICT JSON with keys "
       "medicines, diagnoses, procedures, tests, symptoms - arrays of standard ENGLISH terms. "
       "Empty arrays if none. Do NOT invent terms. Use ONLY these exact English term names "
       "when they apply (so all outputs are comparable):\n"
       f"medicines: {_v(MED)}\ndiagnoses: {_v(DIAG)}\nprocedures: {_v(PROC)}\n"
       f"tests: {_v(TVOCAB)}\nsymptoms: {_v(SYMP)}")


def load_env_keys():
    """Read ../.env lines like 'GROQ API KEY= gsk_...' -> {'groq': 'gsk_...'}."""
    keys, envp = {}, B.parent / ".env"
    if envp.exists():
        for line in envp.read_text(encoding="utf-8").splitlines():
            if "=" in line and not line.strip().startswith("#"):
                name, val = line.split("=", 1)
                prov = name.strip().split()[0].lower()
                if val.strip():
                    keys[prov] = val.strip()
    return keys


def sim(a, b):
    return difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio()


def parse_json(text):
    if "{" in text and "}" in text:
        text = text[text.find("{"): text.rfind("}") + 1]
    return json.loads(text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider"); ap.add_argument("--base-url"); ap.add_argument("--model", required=True)
    ap.add_argument("--key"); ap.add_argument("--label"); ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--delay", type=float, default=2.1, help="seconds between calls (stay under RPM limit)")
    a = ap.parse_args()
    base = a.base_url or PRESETS.get(a.provider or "")
    key = a.key or load_env_keys().get(a.provider or "") or os.environ.get("API_KEY")
    if not base or not key:
        sys.exit("Need --provider (or --base-url) and --key")
    label = a.label or f"{a.provider or 'custom'}:{a.model}"

    if not TEST.exists():
        sys.path.insert(0, str(B)); from gen_data_hindi import write_test_set; write_test_set()
    rows = [json.loads(l) for l in TEST.read_text(encoding="utf-8").splitlines()][:a.n]
    client = OpenAI(base_url=base, api_key=key)

    exact = typos = missed = extra = badjson = fails = 0
    cases = []
    for i, r in enumerate(rows, 1):
        user, gold = r["messages"][1]["content"], r["messages"][2]["content"]
        time.sleep(a.delay)                  # pace BEFORE every call to stay under the RPM limit
        pred = None
        for attempt in range(2):
            try:
                resp = client.chat.completions.create(model=a.model, temperature=0, max_tokens=300,
                    messages=[{"role": "system", "content": SYS}, {"role": "user", "content": user}])
                pred = resp.choices[0].message.content
                break
            except Exception as e:
                if attempt == 1:
                    print(f"  [{i}] api fail: {type(e).__name__}: {str(e)[:80]}", flush=True)
                else:
                    time.sleep(5)
        if pred is None:
            fails += 1; continue
        try:
            pj, gj = parse_json(pred), json.loads(gold)
        except Exception:
            badjson += 1; cases.append({"i": i, "kind": "bad_json", "pred": pred}); continue
        if pj == gj:
            exact += 1; continue
        for k in FIELDS:
            g = [x for x in gj.get(k, [])]
            p = [x for x in pj.get(k, []) if isinstance(x, str)]
            gs, ps = {x.lower() for x in g}, {x.lower() for x in p}
            og, op = [x for x in g if x.lower() not in ps], [x for x in p if x.lower() not in gs]
            for x in og:
                typos += any(sim(x, y) >= 0.8 for y in op)
                missed += not any(sim(x, y) >= 0.8 for y in op)
            for y in op:
                extra += not any(sim(x, y) >= 0.8 for x in og)
        cases.append({"i": i, "hindi": user, "pred": pj, "gold": gj})
        if i % 25 == 0:
            print(f"  {i}/{len(rows)}  exact so far {exact}", flush=True)

    n = len(rows) - fails
    res = {"label": label, "model": a.model, "tested": n, "exact": exact,
           "accuracy_pct": round(100 * exact / max(n, 1), 1), "api_fails": fails,
           "errors": {"typos": typos, "missed": missed, "extra": extra, "bad_json": badjson},
           "cases": cases}
    safe = label.replace("/", "_").replace(":", "_")
    (OUTDIR / f"{safe}.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    sm = OUTDIR / "summary.csv"; new = not sm.exists()
    with open(sm, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["model", "tested", "exact", "accuracy_%", "typos", "missed", "extra", "bad_json", "api_fails"])
        w.writerow([label, n, exact, res["accuracy_pct"], typos, missed, extra, badjson, fails])
    print(f"\n{label}: {exact}/{n} = {res['accuracy_pct']}%  |  typos {typos}  missed {missed}  "
          f"extra {extra}  bad_json {badjson}  api_fails {fails}")


if __name__ == "__main__":
    main()
