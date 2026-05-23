"""
modal_tweak.py — Run predict-tweak + ERRANT scoring on Modal GPU.

Sweeps keep_confidence (kc) and min_error_prob (mep) over a grid,
scores each output against the BEA19-dev reference M2, and prints
a ranked table of results so you can pick the best parameters.

Usage
-----
    # Full grid sweep (kc x mep in {0.0, 0.1, ..., 0.9})
    modal run modal_tweak.py \
        --checkpoint stage3/best

    # Narrow sweep around a known good region
    modal run modal_tweak.py \
        --checkpoint stage3/best \
        --kc_min 0.1 --kc_max 0.5 \
        --mep_min 0.4 --mep_max 0.8 \
        --step 0.1

    # Just re-score already-generated outputs (skip inference)
    modal run modal_tweak.py \
        --checkpoint stage3/best \
        --score_only
"""

import os
from pathlib import Path
import modal

# ── Infrastructure (mirrors modal_train.py) ───────────────────────────────────

app    = modal.App("gector-tweak")
volume = modal.Volume.from_name("gector-data", create_if_missing=False)
MOUNT  = "/gector-data"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "torch>=2.6.0",
        "transformers>=4.49.0",
        "accelerate>=1.3.0",
        "huggingface-hub>=0.28.1",
        "python-levenshtein>=0.26.1",
        "errant",
        "numpy",
    )
    .run_commands(
        # spacy model required by ERRANT
        "python -m spacy download en_core_web_sm",
        "pip install --no-cache-dir git+https://github.com/phamvanvuhoan/My-Gector.git",
    )
    .add_local_file("modal_eval.py", "/root/modal_eval.py")  # reuse annotate helper
)

# ── Shared paths ──────────────────────────────────────────────────────────────

DATA      = f"{MOUNT}/data"
HF_CACHE  = f"{MOUNT}/hf_cache"
CKPT_BASE = f"{MOUNT}/checkpoints"
TWEAK_BASE = f"{MOUNT}/tweak_outputs"

SRC_FILE  = f"{DATA}/bea19-dev/src_aligned.txt"   # extracted from ref M2
REF_M2    = f"{DATA}/bea19-dev/ref.m2"
VERB_FILE = f"{DATA}/verb-form-vocab.txt"


# ── Remote function ───────────────────────────────────────────────────────────

@app.function(
    image    = image,
    gpu      = "T4",          # tweak sweep doesn't need A100
    cpu      = 4,
    memory   = 8192,
    volumes  = {MOUNT: volume},
    timeout  = 7200,          # 2 h — full 10×10 grid takes ~45 min on T4
)
def run_tweak(
    checkpoint:  str,         # relative to CKPT_BASE, e.g. "stage3/best"
    kc_min:      float = 0.0,
    kc_max:      float = 1.0,
    mep_min:     float = 0.0,
    mep_max:     float = 1.0,
    step:        float = 0.1,
    n_iteration: int   = 5,
    batch_size:  int   = 64,
    score_only:  bool  = False,
    beta:        float = 0.5,
):
    import subprocess, sys, json
    import numpy as np
    import torch
    from transformers import AutoTokenizer
    from gector import GECToR, predict, load_verb_dict

    os.environ["HF_HOME"] = HF_CACHE
    volume.reload()  # make sure latest files are visible

    restore_dir = os.path.join(CKPT_BASE, checkpoint)
    out_dir     = os.path.join(TWEAK_BASE, checkpoint.replace("/", "_"))
    hyp_dir     = os.path.join(out_dir, "hyp")
    m2_dir      = os.path.join(out_dir, "m2")
    os.makedirs(hyp_dir, exist_ok=True)
    os.makedirs(m2_dir,  exist_ok=True)

    # ── Build kc/mep grid ────────────────────────────────────────────────────
    kc_values  = [round(v, 2) for v in np.arange(kc_min,  kc_max,  step)]
    mep_values = [round(v, 2) for v in np.arange(mep_min, mep_max, step)]
    pairs      = [(kc, mep) for kc in kc_values for mep in mep_values]
    print(f"Grid: {len(kc_values)} kc × {len(mep_values)} mep = {len(pairs)} configs")

    srcs = Path(SRC_FILE).read_text().rstrip("\n").split("\n")
    print(f"Source sentences: {len(srcs):,}")

    # ── Load model once ───────────────────────────────────────────────────────
    if not score_only:
        print(f"\nLoading model from {restore_dir} ...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model     = GECToR.from_pretrained(restore_dir).eval().to(device)
        tokenizer = AutoTokenizer.from_pretrained(restore_dir)
        encode, decode = load_verb_dict(VERB_FILE)

    # ── Inference sweep ───────────────────────────────────────────────────────
    if not score_only:
        print("\n" + "=" * 60)
        print("Step 1/2 — Inference sweep")
        print("=" * 60)
        for kc, mep in pairs:
            tag      = f"kc{kc}_mep{mep}"
            hyp_path = os.path.join(hyp_dir, f"{tag}.txt")
            if os.path.exists(hyp_path):
                print(f"  SKIP (exists): {tag}")
                continue
            print(f"  {tag} ...", end=" ", flush=True)
            corrected = predict(
                model, tokenizer, srcs, encode, decode,
                keep_confidence = kc,
                min_error_prob  = mep,
                n_iteration     = n_iteration,
                batch_size      = batch_size,
            )
            Path(hyp_path).write_text("\n".join(corrected) + "\n")
            print("done")
        volume.commit()

    # ── ERRANT annotation ─────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Step 2/2 — ERRANT annotation + scoring")
    print("=" * 60)

    import errant as errant_lib
    annotator = errant_lib.load("en")

    # Locate compare_m2.py inside installed errant
    errant_pkg_dir   = Path(errant_lib.__file__).parent
    compare_script   = errant_pkg_dir / "commands" / "compare_m2.py"

    def _count_m2_blocks(m2_path: str) -> int:
        text = Path(m2_path).read_text()
        return sum(1 for b in text.split("\n\n") if b.strip().startswith("S "))

    def annotate(hyp_path: str, m2_path: str) -> None:
        """Write hypothesis M2 from (src, hyp) parallel files."""
        hyps = Path(hyp_path).read_text().rstrip("\n").split("\n")

        if len(hyps) != len(srcs):
            raise RuntimeError(
                f"Line count mismatch in {hyp_path}: "
                f"hyp={len(hyps)} src={len(srcs)}"
            )

        with open(m2_path, "w") as f:
            for i, (src, hyp) in enumerate(zip(srcs, hyps)):
                # An empty hypothesis means the model deleted everything —
                # treat it as identical to src so ERRANT still emits a block.
                if not hyp.strip():
                    hyp = src

                orig  = annotator.parse(src)
                cor   = annotator.parse(hyp)
                edits = annotator.annotate(orig, cor)
                f.write(f"S {src}\n")
                if edits:
                    for e in edits:
                        f.write(e.to_m2() + "\n")
                else:
                    f.write("A -1 -1|||noop|||-NONE-|||REQUIRED|||-NONE-|||0\n")
                f.write("\n")

        # Validate block count matches reference before we try to score
        ref_blocks = _count_m2_blocks(REF_M2)
        hyp_blocks = _count_m2_blocks(m2_path)
        if hyp_blocks != ref_blocks:
            raise RuntimeError(
                f"M2 block count mismatch after annotation: "
                f"hyp={hyp_blocks} ref={ref_blocks}. "
                f"Check {hyp_path} for unexpected empty/malformed lines."
            )

    def score(m2_path: str) -> dict:
        """Run compare_m2.py and parse P/R/F."""
        cmd    = [sys.executable, str(compare_script),
                  "-hyp", m2_path, "-ref", REF_M2, "-b", str(beta)]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(result.stderr)
        # Parse the numeric data row
        for line in result.stdout.split("\n"):
            parts = line.strip().split()
            if len(parts) >= 6 and _is_numeric(parts[0]):
                return {
                    "TP": int(parts[0]),   "FP": int(parts[1]),
                    "FN": int(parts[2]),   "P":  float(parts[3]),
                    "R":  float(parts[4]), "F":  float(parts[5]),
                }
        raise RuntimeError(f"Could not parse compare_m2 output:\n{result.stdout}")

    def _is_numeric(s):
        try: float(s); return True
        except ValueError: return False

    # ── Score all configs ─────────────────────────────────────────────────────
    results = []
    results_path = os.path.join(out_dir, "results.json")

    # Load any already-scored results to allow resuming
    if os.path.exists(results_path):
        with open(results_path) as f:
            results = json.load(f)
    scored_tags = {r["tag"] for r in results}

    for kc, mep in pairs:
        tag      = f"kc{kc}_mep{mep}"
        hyp_path = os.path.join(hyp_dir, f"{tag}.txt")
        m2_path  = os.path.join(m2_dir,  f"{tag}.m2")

        if not os.path.exists(hyp_path):
            print(f"  SKIP (no hyp file): {tag}")
            continue

        if tag in scored_tags:
            print(f"  SKIP (already scored): {tag}")
            continue

        # Annotate if M2 not yet written, or if an existing one has wrong
        # block count (e.g. from a previous failed/interrupted annotation run)
        needs_annotate = not os.path.exists(m2_path)
        if not needs_annotate:
            if _count_m2_blocks(m2_path) != _count_m2_blocks(REF_M2):
                print(f"  Re-annotating {tag} (bad block count) ...", end=" ", flush=True)
                os.remove(m2_path)
                needs_annotate = True

        if needs_annotate:
            print(f"  Annotating {tag} ...", end=" ", flush=True)
            annotate(hyp_path, m2_path)
            print("done", end=" ", flush=True)
        else:
            print(f"  Scoring   {tag} ...", end=" ", flush=True)

        metrics = score(m2_path)
        metrics["tag"] = tag
        metrics["kc"]  = kc
        metrics["mep"] = mep
        results.append(metrics)
        print(f"F{beta}={metrics['F']:.4f}  P={metrics['P']:.4f}  R={metrics['R']:.4f}")

        # Save incrementally so a crash doesn't lose everything
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)

    volume.commit()

    # ── Print ranked table ────────────────────────────────────────────────────
    if not results:
        print("No results to display.")
        return

    results_sorted = sorted(results, key=lambda r: r["F"], reverse=True)

    print("\n" + "=" * 72)
    print(f"  Results ranked by F{beta}  |  checkpoint: {checkpoint}")
    print("=" * 72)
    print(f"  {'Rank':<5} {'kc':>5} {'mep':>5} {'TP':>7} {'FP':>7} {'FN':>7} "
          f"{'P':>7} {'R':>7} {'F'+str(beta):>7}")
    print(f"  {'-'*68}")
    for rank, r in enumerate(results_sorted[:20], 1):   # top-20
        marker = " ←" if rank == 1 else ""
        print(f"  {rank:<5} {r['kc']:>5.1f} {r['mep']:>5.1f} "
              f"{r['TP']:>7} {r['FP']:>7} {r['FN']:>7} "
              f"{r['P']:>7.4f} {r['R']:>7.4f} {r['F']:>7.4f}{marker}")
    print("=" * 72)

    best = results_sorted[0]
    print(f"\n  Best config:  --keep_confidence {best['kc']}  "
          f"--min_error_prob {best['mep']}")
    print(f"  Best F{beta}:    {best['F']:.4f}")
    print(f"\n  Full results saved → {results_path}")

    return results_sorted


# ── Local entrypoints ─────────────────────────────────────────────────────────

@app.local_entrypoint()
def tweak(
    checkpoint:  str   = "stage3/best",
    kc_min:      float = 0.0,
    kc_max:      float = 1.0,
    mep_min:     float = 0.0,
    mep_max:     float = 1.0,
    step:        float = 0.1,
    n_iteration: int   = 5,
    batch_size:  int   = 64,
    score_only:  bool  = False,
    beta:        float = 0.5,
):
    """
    Full sweep — inference + annotation + scoring.

    Examples
    --------
    Full 10×10 grid:
        modal run modal_tweak.py::tweak --checkpoint stage3/best

    Narrow sweep:
        modal run modal_tweak.py::tweak \\
            --checkpoint stage3/best \\
            --kc_min 0.1 --kc_max 0.6 \\
            --mep_min 0.4 --mep_max 0.8

    Re-score only (hyp files already exist on volume):
        modal run modal_tweak.py::tweak \\
            --checkpoint stage3/best \\
            --score_only
    """
    run_tweak.remote(
        checkpoint  = checkpoint,
        kc_min      = kc_min,
        kc_max      = kc_max,
        mep_min     = mep_min,
        mep_max     = mep_max,
        step        = step,
        n_iteration = n_iteration,
        batch_size  = batch_size,
        score_only  = score_only,
        beta        = beta,
    )


@app.local_entrypoint()
def download_results(
    checkpoint: str = "stage3/best",
    local_dir:  str = "outputs/tweak",
):
    """
    Download the results.json and ranked table from the volume.

        modal run modal_tweak.py::download_results --checkpoint stage3/best
    """
    import json

    out_dir      = os.path.join(TWEAK_BASE, checkpoint.replace("/", "_"))
    results_path = os.path.join(out_dir, "results.json")

    local_out = Path(local_dir)
    local_out.mkdir(parents=True, exist_ok=True)
    dest = local_out / "results.json"

    print(f"Downloading {results_path} → {dest} ...")
    with dest.open("wb") as f:
        for chunk in volume.read_file(results_path):
            f.write(chunk)

    results = json.loads(dest.read_text())
    results_sorted = sorted(results, key=lambda r: r["F"], reverse=True)

    beta = 0.5
    print(f"\n{'Rank':<5} {'kc':>5} {'mep':>5} {'P':>8} {'R':>8} {'F0.5':>8}")
    print("-" * 42)
    for rank, r in enumerate(results_sorted[:20], 1):
        marker = " ←" if rank == 1 else ""
        print(f"{rank:<5} {r['kc']:>5.1f} {r['mep']:>5.1f} "
              f"{r['P']:>8.4f} {r['R']:>8.4f} {r['F']:>8.4f}{marker}")

    best = results_sorted[0]
    print(f"\nBest: --keep_confidence {best['kc']} --min_error_prob {best['mep']}")
    print(f"F0.5: {best['F']:.4f}")