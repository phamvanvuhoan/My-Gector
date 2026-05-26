"""
modal_eval.py — ERRANT evaluation of trained GECToR checkpoints on Modal.

Pipeline (runs entirely inside the Modal container):
    1. GECToR inference → hyp.txt  (GPU, T4)
    2. errant_parallel  → hyp.m2   (CPU, same container)
    3. errant_compare   → metrics  (CPU, same container)

All artefacts are written to the volume under:
    /gector-data/eval/<run_name>/hyp.txt
    /gector-data/eval/<run_name>/hyp.m2
    /gector-data/eval/<run_name>/errant_results.json
    /gector-data/eval/<run_name>/tweak_results.tsv   (if --tweak)

Data expected on the volume (upload once with upload_eval_data):
    /gector-data/data/bea19_dev.src          plain-text source sentences
    /gector-data/data/bea19_dev.m2           reference M2 (gold edits)
    /gector-data/data/verb-form-vocab.txt    already there from training setup

Usage
-----
    # Upload BEA19-dev data to the volume (one-time):
    modal run modal_eval.py::upload_eval_data \\
        --src  data/bea19_dev.src \\
        --m2   data/ABCN.dev.gold.bea19.m2

    # Evaluate a specific stage checkpoint:
    modal run modal_eval.py::run_eval \\
        --stage 3 --which best

    # Evaluate with custom kc / mep:
    modal run modal_eval.py::run_eval \\
        --stage 3 --which best \\
        --keep_confidence 0.3 --min_error_prob 0.6

    # Evaluate any arbitrary checkpoint path on the volume:
    modal run modal_eval.py::run_eval \\
        --restore_dir /gector-data/checkpoints/stage3/best

    # Full kc×mep grid search (0..0.9 × 0..0.9, step 0.1):
    modal run modal_eval.py::run_tweak \\
        --stage 3 --which best

    # Download results to your machine:
    modal run modal_eval.py::download_results --stage 3
"""

import os
from pathlib import Path

import modal

# ── Infrastructure (mirrors modal_train.py exactly) ───────────────────────────

app    = modal.App("gector-eval")
volume = modal.Volume.from_name("gector-data", create_if_missing=False)
MOUNT  = "/gector-data"

DATA      = f"{MOUNT}/data"
SAVE_BASE = f"{MOUNT}/checkpoints"
EVAL_BASE = f"{MOUNT}/eval"
HF_CACHE  = f"{MOUNT}/hf_cache"

# ── Image ─────────────────────────────────────────────────────────────────────
# Extends the training image with ERRANT + spaCy English model.
# We bake the spaCy model into the image so it never re-downloads at eval time.

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .env({"FORCE_REBUILD": "2024-05-40"})
    .pip_install(
        "torch>=2.6.0",
        "transformers>=4.49.0",
        "accelerate>=1.3.0",
        "huggingface-hub>=0.28.1",
        "python-levenshtein>=0.26.1",
        # ERRANT and its NLP deps
        "errant",
        "spacy>=3.0.0",
    )
    .run_commands(
        # Install your gector package
        "pip install --no-cache-dir git+https://github.com/phamvanvuhoan/My-Gector.git",
        # Bake in the spaCy English model so container startup is fast
        "python -m spacy download en_core_web_sm",
    )
    .add_local_file("modal_eval.py", "/root/modal_eval.py")
)


# ── Core eval function (GPU) ──────────────────────────────────────────────────

@app.function(
    image   = image,
    gpu     = "T4",
    cpu     = 4,
    memory  = 8192,
    volumes = {MOUNT: volume},
    timeout = 3600,   # inference on 4k BEA19-dev sentences is fast; 1h is generous
)
def evaluate(
    restore_dir:      str,
    src:              str   = f"{DATA}/bea19_dev.src",
    ref_m2:           str   = f"{DATA}/bea19_dev.m2",
    run_name:         str   = "eval",
    keep_confidence:  float = 0.1,
    min_error_prob:   float = 0.1,
    n_iteration:      int   = 5,
    batch_size:       int   = 128,
    beam_size:        int   = 1,
    beta:             float = 0.5,
    verb_file:        str   = f"{DATA}/verb-form-vocab.txt",
    # Re-use existing hyp.txt (skips inference, re-annotates + re-scores)
    skip_inference:   bool  = False,
) -> dict:
    """
    Run the full inference → annotation → scoring pipeline.
    Returns the metrics dict: {"P": ..., "R": ..., "F0.5": ...}
    """
    import json
    import subprocess
    import sys
    import torch
    from pathlib import Path
    from transformers import AutoTokenizer
    from gector import GECToR, predict, load_verb_dict, beam_predict

    os.environ["HF_HOME"] = HF_CACHE
    volume.reload()   # make sure we see the latest volume state

    out_dir = Path(EVAL_BASE) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    hyp_path = str(out_dir / "hyp.txt")
    hyp_m2   = str(out_dir / "hyp.m2")

    # ── Step 1: inference ────────────────────────────────────────────────────
    if skip_inference and Path(hyp_path).exists():
        print(f"Skipping inference — reusing {hyp_path}")
    else:
        print(f"\n{'='*60}")
        print("Step 1: GECToR inference")
        print(f"  restore_dir     : {restore_dir}")
        print(f"  keep_confidence : {keep_confidence}")
        print(f"  min_error_prob  : {min_error_prob}")
        print(f"  n_iteration     : {n_iteration}")
        print(f"  beam_size       : {beam_size}")
        print(f"{'='*60}")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device: {device}")

        model     = GECToR.from_pretrained(restore_dir).eval().to(device)
        tokenizer = AutoTokenizer.from_pretrained(restore_dir)
        encode, decode = load_verb_dict(verb_file)

        srcs = Path(src).read_text().rstrip().splitlines()
        print(f"Source sentences: {len(srcs):,}")

        if beam_size > 1:
            corrected = beam_predict(
                model, tokenizer, srcs, encode, decode,
                beam_size       = beam_size,
                keep_confidence = keep_confidence,
                min_error_prob  = min_error_prob,
                n_iteration     = n_iteration,
                batch_size      = batch_size,
            )
        else:
            corrected = predict(
                model, tokenizer, srcs, encode, decode,
                keep_confidence = keep_confidence,
                min_error_prob  = min_error_prob,
                n_iteration     = n_iteration,
                batch_size      = batch_size,
            )

        # Sanity check
        if len(corrected) != len(srcs):
            raise ValueError(
                f"Line count mismatch: src={len(srcs)}, hyp={len(corrected)}"
            )

        Path(hyp_path).write_text("\n".join(corrected) + "\n")
        print(f"Hypotheses written → {hyp_path}")

    # ── Step 2: ERRANT annotation ─────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Step 2: ERRANT annotation (errant_parallel)")
    print(f"{'='*60}")

    result = subprocess.run(
        ["errant_parallel", "-orig", src, "-cor", hyp_path, "-out", hyp_m2],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        # Fallback to module invocation (some envs don't install console scripts)
        result = subprocess.run(
            [sys.executable, "-m", "errant.commands.parallel_to_m2",
             "-orig", src, "-cor", hyp_path, "-out", hyp_m2],
            capture_output=True, text=True,
        )
    if result.returncode != 0:
        raise RuntimeError(f"errant_parallel failed:\n{result.stderr}")
    print(f"M2 written → {hyp_m2}")

    # ── Step 3: ERRANT scoring ────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Step 3: ERRANT scoring (errant_compare)")
    print(f"{'='*60}")

    result = subprocess.run(
        ["errant_compare", "-hyp", hyp_m2, "-ref", ref_m2, "-b", str(beta)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        result = subprocess.run(
            [sys.executable, "-m", "errant.commands.compare_m2",
             "-hyp", hyp_m2, "-ref", ref_m2, "-b", str(beta)],
            capture_output=True, text=True,
        )
    if result.returncode != 0:
        raise RuntimeError(f"errant_compare failed:\n{result.stderr}")

    print(result.stdout)
    metrics = _parse_errant_output(result.stdout, beta)

    # ── Pretty print ──────────────────────────────────────────────────────────
    fb_key = f"F{beta}"
    print(f"\n{'='*60}")
    print("ERRANT Results")
    print(f"  restore_dir    : {restore_dir}")
    print(f"  keep_confidence: {keep_confidence}  min_error_prob: {min_error_prob}")
    print(f"{'='*60}")
    if "TP" in metrics:
        print(f"  TP  : {metrics['TP']}")
        print(f"  FP  : {metrics['FP']}")
        print(f"  FN  : {metrics['FN']}")
    print(f"  P   : {metrics.get('P', 'n/a')}")
    print(f"  R   : {metrics.get('R', 'n/a')}")
    print(f"  {fb_key}: {metrics.get(fb_key, 'n/a')}")
    print(f"{'='*60}")

    # ── Persist results ───────────────────────────────────────────────────────
    result_data = {
        "restore_dir":      restore_dir,
        "src":              src,
        "ref_m2":           ref_m2,
        "keep_confidence":  keep_confidence,
        "min_error_prob":   min_error_prob,
        "n_iteration":      n_iteration,
        "beam_size":        beam_size,
        "beta":             beta,
        "metrics":          metrics,
    }
    result_path = out_dir / "errant_results.json"
    result_path.write_text(json.dumps(result_data, indent=2))
    print(f"\nResults saved → {result_path}")

    volume.commit()
    return metrics


# ── Tweak grid search (CPU — no GPU needed, hyp.txt must already exist) ───────

@app.function(
    image   = image,
    cpu     = 4,
    memory  = 4096,
    volumes = {MOUNT: volume},
    timeout = 7200,   # 100-cell grid × ~20s per cell ≈ 30 min; 2h is safe
)
def tweak_grid(
    restore_dir:   str,
    src:           str   = f"{DATA}/bea19_dev.src",
    ref_m2:        str   = f"{DATA}/bea19_dev.m2",
    run_name:      str   = "tweak",
    kc_min:        float = 0.0,
    kc_max:        float = 1.0,
    mep_min:       float = 0.0,
    mep_max:       float = 1.0,
    step:          float = 0.1,
    n_iteration:   int   = 5,
    batch_size:    int   = 128,
    beta:          float = 0.5,
    verb_file:     str   = f"{DATA}/verb-form-vocab.txt",
) -> dict:
    """
    Grid-search keep_confidence × min_error_prob.

    Because each (kc, mep) pair requires a separate inference pass, this
    function spawns one `evaluate` GPU task per grid cell in parallel using
    Modal's native parallelism (.map). Results are collected and the best
    (kc, mep) is returned.

    The grid is kc ∈ [kc_min, kc_max) × mep ∈ [mep_min, mep_max) with
    the given step, giving up to 10×10 = 100 cells by default.
    """
    import json
    import numpy as np

    volume.reload()

    kc_vals  = [round(v, 2) for v in np.arange(kc_min,  kc_max,  step)]
    mep_vals = [round(v, 2) for v in np.arange(mep_min, mep_max, step)]

    # Build list of (kc, mep, run_name) triples
    cells = [
        (kc, mep, f"{run_name}/kc{kc}_mep{mep}")
        for kc  in kc_vals
        for mep in mep_vals
    ]
    print(f"Grid: {len(kc_vals)} × {len(mep_vals)} = {len(cells)} cells")

    # Spawn all inference+eval tasks in parallel
    inputs = [
        dict(
            restore_dir     = restore_dir,
            src             = src,
            ref_m2          = ref_m2,
            run_name        = cell_name,
            keep_confidence = kc,
            min_error_prob  = mep,
            n_iteration     = n_iteration,
            batch_size      = batch_size,
            beta            = beta,
            verb_file       = verb_file,
        )
        for kc, mep, cell_name in cells
    ]

    # evaluate.map runs all cells concurrently (Modal schedules the GPU tasks)
    all_metrics = list(evaluate.map(inputs, kwargs={}))

    # Collect results
    fb_key  = f"F{beta}"
    results = []
    for (kc, mep, _), metrics in zip(cells, all_metrics):
        score = metrics.get(fb_key, 0.0)
        results.append((kc, mep, metrics, score))
        print(f"  kc={kc:.1f}  mep={mep:.1f}  "
              f"P={metrics.get('P', 0):.4f}  "
              f"R={metrics.get('R', 0):.4f}  "
              f"{fb_key}={score:.4f}")

    # Best
    best = max(results, key=lambda x: x[3])
    print(f"\nBest: kc={best[0]}  mep={best[1]}  {fb_key}={best[3]:.4f}")

    # Save TSV summary
    out_dir  = Path(EVAL_BASE) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    tsv_path = out_dir / "tweak_results.tsv"
    with open(tsv_path, "w") as f:
        f.write(f"kc\tmep\tP\tR\t{fb_key}\n")
        for kc, mep, m, _ in results:
            f.write(f"{kc}\t{mep}\t"
                    f"{m.get('P', 0):.4f}\t"
                    f"{m.get('R', 0):.4f}\t"
                    f"{m.get(fb_key, 0):.4f}\n")
    print(f"Grid results → {tsv_path}")

    volume.commit()
    return {"best_kc": best[0], "best_mep": best[1], fb_key: best[3]}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _parse_errant_output(output: str, beta: float) -> dict:
    """
    Parse errant_compare tabular output into a metrics dict.

    ERRANT can produce two formats depending on version:

    6-column (most common):
        TP    FP    FN    Prec    Rec    F_0.5
        1234  567   890   0.6851  0.3690  0.5762

    3-column:
        Prec   Rec    F_0.5
        68.51  36.90  57.62
    """
    fb_key  = f"F{beta}"
    metrics = {}
    for line in output.splitlines():
        parts = line.strip().split()
        if len(parts) not in (3, 6):
            continue
        try:
            floats = [float(p) for p in parts]
        except ValueError:
            continue
        if len(floats) == 6:
            metrics = {
                "TP": int(floats[0]), "FP": int(floats[1]), "FN": int(floats[2]),
                "P":  floats[3],      "R":  floats[4],      fb_key: floats[5],
            }
        elif len(floats) == 3:
            metrics = {"P": floats[0], "R": floats[1], fb_key: floats[2]}
    return metrics


# ── Local entrypoints ─────────────────────────────────────────────────────────

@app.local_entrypoint()
def upload_eval_data(
    src: str = "data/bea19_dev.src",
    m2:  str = "data/ABCN.dev.gold.bea19.m2",
):
    """
    Upload BEA19-dev evaluation data to the volume (run once).

        modal run modal_eval.py::upload_eval_data \\
            --src data/bea19_dev.src \\
            --m2  data/ABCN.dev.gold.bea19.m2
    """
    print("Uploading BEA19-dev eval data to volume 'gector-data' ...")
    with volume.batch_upload() as batch:
        for local, remote in [
            (src, "data/bea19_dev.src"),
            (m2,  "data/bea19_dev.m2"),
        ]:
            if not Path(local).exists():
                print(f"  SKIP (not found): {local}")
                continue
            size_mb = Path(local).stat().st_size / 1e6
            print(f"  {local} ({size_mb:.1f} MB) → {remote}")
            batch.put_file(local, remote)
    print("✓ Upload complete.")


@app.local_entrypoint()
def run_eval(
    stage:            int   = 3,
    which:            str   = "best",       # "best" or "last"
    restore_dir:      str   = "",           # override: use any volume path directly
    keep_confidence:  float = 0.1,
    min_error_prob:   float = 0.1,
    n_iteration:      int   = 5,
    batch_size:       int   = 128,
    beam_size:        int   = 1,
    beta:             float = 0.5,
    src:              str   = "",           # override default bea19_dev.src
    ref_m2:           str   = "",           # override default bea19_dev.m2
):
    """
    Evaluate a stage checkpoint.

        modal run modal_eval.py::run_eval --stage 3 --which best
        modal run modal_eval.py::run_eval --stage 3 --keep_confidence 0.3 --min_error_prob 0.6
    """
    ckpt = restore_dir or f"{SAVE_BASE}/stage{stage}/{which}"
    name = (
        restore_dir.replace("/", "_").strip("_")
        if restore_dir
        else f"stage{stage}_{which}"
    )
    run_name = f"{name}_kc{keep_confidence}_mep{min_error_prob}"

    kwargs = dict(
        restore_dir     = ckpt,
        run_name        = run_name,
        keep_confidence = keep_confidence,
        min_error_prob  = min_error_prob,
        n_iteration     = n_iteration,
        batch_size      = batch_size,
        beam_size       = beam_size,
        beta            = beta,
    )
    if src:
        kwargs["src"] = src
    if ref_m2:
        kwargs["ref_m2"] = ref_m2

    print(f"Launching eval: {ckpt}")
    metrics = evaluate.remote(**kwargs)

    fb_key = f"F{beta}"
    print(f"\n{'='*60}")
    print(f"Final: P={metrics.get('P','?')}  R={metrics.get('R','?')}  {fb_key}={metrics.get(fb_key,'?')}")
    print(f"{'='*60}")


@app.local_entrypoint()
def run_tweak(
    stage:       int   = 3,
    which:       str   = "best",
    restore_dir: str   = "",
    kc_min:      float = 0.0,
    kc_max:      float = 1.0,
    mep_min:     float = 0.0,
    mep_max:     float = 1.0,
    step:        float = 0.1,
    n_iteration: int   = 5,
    batch_size:  int   = 128,
    beta:        float = 0.5,
):
    """
    Grid-search kc × mep, running all cells in parallel on Modal.

        modal run modal_eval.py::run_tweak --stage 3 --which best
    """
    ckpt = restore_dir or f"{SAVE_BASE}/stage{stage}/{which}"
    name = (
        restore_dir.replace("/", "_").strip("_")
        if restore_dir
        else f"stage{stage}_{which}"
    )

    print(f"Launching tweak grid: {ckpt}")
    result = tweak_grid.remote(
        restore_dir = ckpt,
        run_name    = f"{name}_tweak",
        kc_min      = kc_min,
        kc_max      = kc_max,
        mep_min     = mep_min,
        mep_max     = mep_max,
        step        = step,
        n_iteration = n_iteration,
        batch_size  = batch_size,
        beta        = beta,
    )

    fb_key = f"F{beta}"
    print(f"\nBest: kc={result['best_kc']}  mep={result['best_mep']}  {fb_key}={result[fb_key]:.4f}")


@app.local_entrypoint()
def download_results(
    stage:     int = 3,
    which:     str = "best",
    local_dir: str = "outputs/errant_eval",
):
    """
    Download eval results from the volume to your machine.

        modal run modal_eval.py::download_results --stage 3
    """
    remote_dir = f"eval/stage{stage}_{which}"
    local_root = Path(local_dir)
    local_root.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {remote_dir} → {local_dir} ...")
    try:
        entries = list(volume.listdir(remote_dir, recursive=True))
    except Exception as e:
        print(f"ERROR: {e}")
        return

    for entry in entries:
        dest = local_root / Path(entry.path).name
        with dest.open("wb") as f:
            for chunk in volume.read_file(entry.path):
                f.write(chunk)
        print(f"  {dest}")

    print("✓ Done.")