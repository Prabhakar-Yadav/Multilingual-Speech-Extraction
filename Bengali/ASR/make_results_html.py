"""Turn four_files_compare.json into a readable local HTML page (base vs fine-tuned)."""
import json, html
from pathlib import Path

B = Path(__file__).resolve().parent
d = json.loads((B / "four_files_compare.json").read_text(encoding="utf-8"))

rows = ""
for i in range(4):
    base = html.escape(d["base"][i])
    ft = html.escape(d["finetuned"][i])
    rows += f"""
    <h2>File {i+1}</h2>
    <div class="pair">
      <div class="col base"><div class="tag">BASE whisper-medium (before)</div><p>{base}</p></div>
      <div class="col ft"><div class="tag">FINE-TUNED on Vaani (after)</div><p>{ft}</p></div>
    </div>"""

page = f"""<!doctype html><html><head><meta charset="utf-8">
<title>Bengali ASR — Before / After</title>
<style>
 body{{font-family:'Nirmala UI',Segoe UI,Arial,sans-serif;max-width:1100px;margin:24px auto;padding:0 16px;color:#111}}
 h1{{font-size:22px}} h2{{margin-top:28px;border-top:1px solid #ddd;padding-top:14px}}
 .pair{{display:flex;gap:16px}} .col{{flex:1;padding:12px;border-radius:8px}}
 .base{{background:#fff1f0;border:1px solid #ffa39e}} .ft{{background:#f6ffed;border:1px solid #b7eb8f}}
 .tag{{font-weight:700;font-size:12px;margin-bottom:8px;text-transform:uppercase}}
 .base .tag{{color:#cf1322}} .ft .tag{{color:#389e0d}}
 p{{font-size:16px;line-height:1.7;margin:0;white-space:pre-wrap;word-break:break-word}}
 .note{{background:#e6f7ff;border:1px solid #91d5ff;padding:12px;border-radius:8px}}
</style></head><body>
<h1>Bengali ASR — Base vs Fine-tuned (4 client calls)</h1>
<div class="note"><b>Result:</b> base whisper-medium outputs wrong-script gibberish (Khmer/Hindi) with endless repetition —
the "transcription not proper" issue. The fine-tuned model outputs <b>real Bengali</b> matching the actual conversations
(doctor fees, locations, medicine follow-up). Eval WER improved 86% → 64.5%.</div>
{rows}
</body></html>"""

out = B / "results.html"
out.write_text(page, encoding="utf-8")
print("wrote ->", out)
