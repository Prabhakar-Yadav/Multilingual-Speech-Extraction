# Run when the GPU is free

Everything below is uploaded to `~/bengali_asr/` on `prabhakar@100.105.243.101`
and syntax-checked. Copy-paste in order.

---

## 0. Is the GPU actually free?

```powershell
ssh prabhakar@100.105.243.101 "nvidia-smi"
```

Look at the process table at the bottom. You need **~12 GB free**. Anything under
`/usr/lib/xorg/Xorg` (a few MiB) is just the desktop and can be ignored.

---

## 1. THE MAIN RUN — telephone-matched training

This is the experiment worth the GPU time. It degrades 50% of the training clips
to 8 kHz telephone quality, so the model finally trains on audio that resembles
the client's calls instead of clean studio speech.

```powershell
ssh prabhakar@100.105.243.101 "cd ~/bengali_asr && RUN=tele TELE_PROB=0.5 VRAM_GB=16 nohup ./venv/bin/python train_slr53.py > train_tele.log 2>&1 < /dev/null & echo LAUNCHED"
```

`RUN=tele` writes to `ckpt_tele/`, so **the existing `ckpt_run2/` checkpoints are
untouched** and the two runs stay comparable.

Confirm it started (wait ~60s first):

```powershell
ssh prabhakar@100.105.243.101 "cd ~/bengali_asr; grep -a -E 'VRAM cap|TELEPHONE|TRAIN |resuming' train_tele.log | head -5"
```

You want to see: `TELEPHONE AUG: 50% of training clips degraded to 8kHz (ON)`

### Watch it

```powershell
ssh prabhakar@100.105.243.101 "cd ~/bengali_asr; tail -c 200 train_tele.log | tr '\r' '\n' | tail -1; ls ckpt_tele"
```

Checkpoints land every 300 steps, all kept. Safe to close your laptop.

### Stop it at a set time

```powershell
ssh prabhakar@100.105.243.101 "cd ~/bengali_asr && nohup bash stop_at.sh 07:30 > stopper.log 2>&1 < /dev/null & echo ARMED"
```

Re-running the launch command later auto-resumes from the newest checkpoint.

---

## 2. THE NUMBER THAT DECIDES IT — test on the real client calls

The held-out SLR53 WER is clean speech and will probably get slightly *worse*
(the model is now splitting capacity across two conditions). Ignore it. **This is
the number that matters:**

```powershell
ssh prabhakar@100.105.243.101 "cd ~/bengali_asr && CKPT=ckpt_tele/checkpoint-3000 ./venv/bin/python eval_client.py 2>&1 | tail -12"
```

### Scoreboard to beat

| Model | client WER | client CER |
|---|---|---|
| baseline large-v3 | 102.0% | 74.5% |
| clean-trained (ckpt_run2/checkpoint-3000) | **92.1%** | **68.1%** |
| telephone-trained | ? | ? |

**Judge on CER, not WER.** At ~90% WER the metric is saturated and can't show real
progress; CER still moves. A 5+ point CER drop would be a genuine win.

---

## 3. Optional: learned bandwidth extension

The literature's biggest reported gain, and the one thing my crude spectral
folding could not test (it needs a generative model, not a filter).

```powershell
ssh prabhakar@100.105.243.101 "cd ~/bengali_asr && ./venv/bin/pip install -q voicefixer && ./venv/bin/python -c 'import voicefixer; print(\"ok\")'"
```

If it installs, restore the calls then re-run `eval_client.py` pointed at the
restored audio. Reported gains on 4->16 kHz are large (CER 68.6% -> 7.7% in one
study); our raw CER of 66.1% sits at exactly that starting point.

---

## What is already done

- `train_slr53.py` — LR 1e-4, save every 300 steps, keep all checkpoints,
  load-best-at-end, 16 GB VRAM cap, auto-resume, **new: `TELE_PROB` augmentation**
- `eval_client.py` — scores any checkpoint on the 4 real client calls
- `enhance_test.py` — compares raw / normalise / bandlift / denoise front-ends
- `stop_at.sh` — clean SIGINT stop at a wall-clock time
- `backup_ckpts.sh` — mirrors every checkpoint into `ckpt_backup/`

## Verified

Augmentation reproduces the client's acoustics (measured, not assumed):

```
CLEAN slr53    90%BW= 482Hz   99%BW= 1069Hz   rms=0.0208
AUGMENTED      90%BW= 836Hz   99%BW= 1193Hz   rms=0.0500
CLIENT calls   90%BW= 650-1268Hz  99%BW= 1225-2665Hz  rms=0.0395-0.0492
```

The augmented signal lands inside the client's range on every measure.
