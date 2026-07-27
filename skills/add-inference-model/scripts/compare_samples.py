#!/usr/bin/env python3
"""Compare ai-toolkit training preview samples against inference-server outputs.

This is the actual pass/fail gate for a new pipeline: if the inference pipeline
reproduces the trainer's preview for the same prompt/seed/steps/guidance, the
conditioning, schedule, CFG normalization and LoRA application are all correct.

Usage:

  # 1. train, then pull the final-step samples and the LoRA off the training box
  # 2. replay the same config through the inference server:
  #      python scripts/request_samples_from_config.py \
  #        --config scripts/krea2_parity/train_krea2_parity.yaml \
  #        --server http://<gpu-box>:8000 \
  #        --lora-file krea2_parity_20260723.safetensors --wait
  # 3. compare:
  python scripts/krea2_parity/compare_samples.py \
      --training-dir  ./samples_from_training \
      --inference-dir ./inference_output \
      --out           ./krea2_parity_report

Pairing: both sides are sorted by their numeric prompt index. ai-toolkit writes
samples as <name>_<step>_<idx>.jpg; the inference server writes
<request_id>_output_<idx>.jpg. Pass --training-glob / --inference-glob to
override if your filenames differ.

Interpreting the numbers -- these are NOT bit-exact comparisons. bf16
nondeterminism, attention-kernel selection and VAE decode differences between two
machines put a small floor under every metric. Rough guidance from other models
in this repo:

  PSNR > 30 dB, LPIPS < 0.05   same image, different arithmetic  -> PASS
  PSNR 20-30 dB                same composition, visible drift   -> investigate
  PSNR < 20 dB                 different image                   -> real bug

A systematic failure across ALL prompts points at a global parameter (guidance
normalization, mu schedule, step count). A failure on SOME prompts points at
conditioning (prompt template, token slicing).
"""

import argparse
import json
import os
import re
import sys
from glob import glob
from typing import List, Optional, Tuple

try:
    import numpy as np
except ImportError:
    sys.exit("numpy is required: pip install numpy")

try:
    from PIL import Image
except ImportError:
    sys.exit("Pillow is required: pip install pillow")


def _index_of(path: str) -> int:
    """Trailing integer in the filename, used to pair the two sides."""
    stem = os.path.splitext(os.path.basename(path))[0]
    nums = re.findall(r"\d+", stem)
    return int(nums[-1]) if nums else -1


def collect(directory: str, pattern: Optional[str]) -> List[str]:
    if pattern:
        files = sorted(glob(os.path.join(directory, pattern)))
    else:
        files = sorted(
            f
            for ext in ("jpg", "jpeg", "png", "webp")
            for f in glob(os.path.join(directory, f"*.{ext}"))
        )
    if not files:
        raise SystemExit(f"No images found in {directory!r} (pattern={pattern!r})")
    return sorted(files, key=_index_of)


def load(path: str, size: Optional[Tuple[int, int]] = None) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    if size is not None and img.size != size:
        print(f"  ! size mismatch {img.size} vs {size}, resizing: {os.path.basename(path)}")
        img = img.resize(size, Image.LANCZOS)
    return np.asarray(img).astype(np.float32) / 255.0


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a - b) ** 2))
    if mse <= 1e-12:
        return float("inf")
    return float(10.0 * np.log10(1.0 / mse))


def ssim_gray(a: np.ndarray, b: np.ndarray) -> float:
    """Global SSIM on the luma channel. Cheap, dependency-free, good enough to
    separate 'same image' from 'different image'."""
    ga = a @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    gb = b @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    mu_a, mu_b = ga.mean(), gb.mean()
    va, vb = ga.var(), gb.var()
    cov = ((ga - mu_a) * (gb - mu_b)).mean()
    c1, c2 = 0.01**2, 0.03**2
    return float(
        ((2 * mu_a * mu_b + c1) * (2 * cov + c2))
        / ((mu_a**2 + mu_b**2 + c1) * (va + vb + c2))
    )


def lpips_fn():
    """Optional perceptual metric; returns None when lpips/torch is unavailable."""
    try:
        import lpips
        import torch
    except ImportError:
        return None
    net = lpips.LPIPS(net="alex", verbose=False)

    def fn(a: np.ndarray, b: np.ndarray) -> float:
        import torch

        def t(x):
            return torch.from_numpy(x).permute(2, 0, 1)[None] * 2 - 1

        with torch.no_grad():
            return float(net(t(a), t(b)).item())

    return fn


def contact_sheet(pairs, out_path: str, thumb: int = 384) -> None:
    """training | inference | absolute difference, one row per prompt."""
    rows = []
    for tr_path, inf_path, _ in pairs:
        tr = Image.open(tr_path).convert("RGB")
        inf = Image.open(inf_path).convert("RGB").resize(tr.size, Image.LANCZOS)
        diff_arr = np.abs(
            np.asarray(tr).astype(np.int16) - np.asarray(inf).astype(np.int16)
        ).astype(np.uint8)
        # amplify so small drift is visible at a glance
        diff = Image.fromarray(np.clip(diff_arr.astype(np.int16) * 4, 0, 255).astype(np.uint8))
        h = thumb
        w = int(tr.width * thumb / tr.height)
        rows.append([im.resize((w, h), Image.LANCZOS) for im in (tr, inf, diff)])

    if not rows:
        return
    cw, ch = rows[0][0].size
    sheet = Image.new("RGB", (cw * 3, ch * len(rows)), "black")
    for r, row in enumerate(rows):
        for c, im in enumerate(row):
            sheet.paste(im, (c * cw, r * ch))
    sheet.save(out_path, quality=92)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--training-dir", required=True, help="ai-toolkit preview samples")
    ap.add_argument("--inference-dir", required=True, help="inference-server outputs")
    ap.add_argument("--training-glob", default=None)
    ap.add_argument("--inference-glob", default=None)
    ap.add_argument("--out", default="./parity_report")
    ap.add_argument("--psnr-pass", type=float, default=30.0)
    ap.add_argument("--psnr-warn", type=float, default=20.0)
    ap.add_argument("--label", default="krea2", help="model id, for the report header")
    args = ap.parse_args()

    train_files = collect(args.training_dir, args.training_glob)
    inf_files = collect(args.inference_dir, args.inference_glob)

    if len(train_files) != len(inf_files):
        print(
            f"! count mismatch: {len(train_files)} training vs {len(inf_files)} "
            f"inference images; comparing the first {min(len(train_files), len(inf_files))}"
        )
    n = min(len(train_files), len(inf_files))
    os.makedirs(args.out, exist_ok=True)

    lp = lpips_fn()
    if lp is None:
        print("(lpips not installed -- reporting PSNR/SSIM only; pip install lpips for a perceptual metric)")

    pairs, results = [], []
    for i in range(n):
        tr_path, inf_path = train_files[i], inf_files[i]
        tr = load(tr_path)
        inf = load(inf_path, size=Image.open(tr_path).size)
        rec = {
            "index": i,
            "training": os.path.basename(tr_path),
            "inference": os.path.basename(inf_path),
            "psnr": psnr(tr, inf),
            "ssim": ssim_gray(tr, inf),
            "mae": float(np.mean(np.abs(tr - inf))),
        }
        if lp is not None:
            rec["lpips"] = lp(tr, inf)
        rec["verdict"] = (
            "PASS" if rec["psnr"] >= args.psnr_pass
            else "INVESTIGATE" if rec["psnr"] >= args.psnr_warn
            else "FAIL"
        )
        results.append(rec)
        pairs.append((tr_path, inf_path, rec))

    sheet_path = os.path.join(args.out, "contact_sheet.jpg")
    contact_sheet(pairs, sheet_path)

    verdicts = [r["verdict"] for r in results]
    overall = "FAIL" if "FAIL" in verdicts else ("INVESTIGATE" if "INVESTIGATE" in verdicts else "PASS")
    mean_psnr = float(np.mean([r["psnr"] for r in results if np.isfinite(r["psnr"])] or [0]))

    lines = [
        f"# Krea 2 training-sample vs inference parity — `{args.label}`",
        "",
        f"**Overall: {overall}**  ({verdicts.count('PASS')}/{len(verdicts)} pass, "
        f"mean PSNR {mean_psnr:.2f} dB)",
        "",
        "| # | verdict | PSNR (dB) | SSIM | MAE | training | inference |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in results:
        lines.append(
            f"| {r['index']} | {r['verdict']} | {r['psnr']:.2f} | {r['ssim']:.4f} | "
            f"{r['mae']:.4f} | `{r['training']}` | `{r['inference']}` |"
        )
    if lp is not None:
        lines += ["", "| # | LPIPS |", "|---|---|"]
        lines += [f"| {r['index']} | {r['lpips']:.4f} |" for r in results]
    lines += [
        "",
        "![contact sheet](contact_sheet.jpg)",
        "",
        "Columns: training sample | inference output | |difference| x4.",
        "",
        "## How to read a failure",
        "",
        "- **All prompts fail together** -> a global parameter is wrong: guidance",
        "  normalization (the trainer applies `max(0, cfg - 1)`), the mu schedule,",
        "  or the step count.",
        "- **Some prompts fail** -> conditioning: prompt template, the 34-token",
        "  prefix slice, or the selected Qwen3-VL hidden layers.",
        "- **Everything is close but uniformly soft/noisy** -> quantization or dtype",
        "  mismatch; confirm the training run had `quantize: false`.",
    ]
    report = os.path.join(args.out, "report.md")
    with open(report, "w") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join(args.out, "results.json"), "w") as f:
        json.dump({"overall": overall, "mean_psnr": mean_psnr, "results": results}, f, indent=2)

    print("\n".join(lines[:6 + len(results)]))
    print(f"\nreport:       {report}")
    print(f"contact sheet: {sheet_path}")
    return 0 if overall != "FAIL" else 1


if __name__ == "__main__":
    raise SystemExit(main())
