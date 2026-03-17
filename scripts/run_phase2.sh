#!/usr/bin/env bash
set -euo pipefail

# Phase 2 batch runner for SOCOFing / fingerprint experiments
# ------------------------------------------------------------
# What this script does
#   1) Train baseline (v3) and MixStyle (v3_mixstyle) for multiple seeds
#   2) Evaluate each trained checkpoint on unseen CR and unseen Zcut
#   3) Summarize JSON metrics into CSV files
#
# Why this script uses a wrapper to set seeds
#   The current Python scripts do NOT expose a --seed flag for training.
#   This bash script seeds Python / NumPy / PyTorch externally before
#   launching each run, so you can still do fair multi-seed comparisons.
#
# Usage examples
#   bash run_phase2.sh                # train + eval + summarize
#   bash run_phase2.sh train          # train only
#   bash run_phase2.sh eval           # eval only (needs best.pt already present)
#   bash run_phase2.sh summarize      # rebuild CSV summaries only
#
# Optional overrides from shell
#   PROJECT_ROOT=/home/congpx/fingerprint bash run_phase2.sh
#   TRAIN_SEVERITIES=real,easy,medium bash run_phase2.sh
#   USE_PRETRAINED=1 USE_AMP=1 bash run_phase2.sh
#   CUDA_VISIBLE_DEVICES=0 bash run_phase2.sh

MODE="${1:-all}"

# -----------------------------
# Config: paths
# -----------------------------
PROJECT_ROOT="${PROJECT_ROOT:-/home/congpx/fingerprint}"
PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/data/SOCOFing/SOCOFing}"
INDEX_CSV="${INDEX_CSV:-${PROJECT_ROOT}/splits/index_v3.csv}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/outputs_phase2}"
LOG_ROOT="${LOG_ROOT:-${OUTPUT_ROOT}/logs}"
mkdir -p "${OUTPUT_ROOT}" "${LOG_ROOT}"

# -----------------------------
# Config: experiment design
# -----------------------------
SEEDS=(1 2 3)
TRAIN_ALT_TYPES="${TRAIN_ALT_TYPES:-Obl}"
# Keep this EXPLICIT so base and mix are apples-to-apples.
# If you want to compare against your existing mix run that used only real,easy,medium,
# set TRAIN_SEVERITIES=real,easy,medium before launching.
TRAIN_SEVERITIES="${TRAIN_SEVERITIES:-real,easy,medium,hard}"
# For seenObl -> unseen{CR,Zcut}, using CR,Zcut for validation selection is reasonable.
# Set to empty string if you want validation over ALL altered types.
VAL_ALT_TYPES="${VAL_ALT_TYPES:-CR,Zcut}"
PROBES=(CR Zcut)
PROBE_SEVERITIES="${PROBE_SEVERITIES:-easy,medium,hard}"

# -----------------------------
# Config: train hyper-parameters
# -----------------------------
EPOCHS="${EPOCHS:-30}"
BATCH="${BATCH:-128}"
EVAL_BATCH="${EVAL_BATCH:-256}"
WORKERS="${WORKERS:-8}"
LR="${LR:-1e-4}"
EMB_DIM="${EMB_DIM:-256}"
USE_PRETRAINED="${USE_PRETRAINED:-1}"
USE_AMP="${USE_AMP:-1}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

# -----------------------------
# Config: MixStyle hyper-parameters
# -----------------------------
MIX_P="${MIX_P:-0.5}"
MIX_ALPHA="${MIX_ALPHA:-0.3}"
MIX_LAYER="${MIX_LAYER:-layer1}"

# -----------------------------
# Internal helpers
# -----------------------------
BASE_SCRIPT="${PROJECT_ROOT}/run_socofing_v3.py"
MIX_SCRIPT="${PROJECT_ROOT}/run_socofing_v3_mixstyle.py"

require_file() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    echo "[ERR] Missing file: $path" >&2
    exit 1
  fi
}

require_dir() {
  local path="$1"
  if [[ ! -d "$path" ]]; then
    echo "[ERR] Missing directory: $path" >&2
    exit 1
  fi
}

run_seeded_python() {
  # Usage:
  #   SEED=1 run_seeded_python /path/to/script.py train ...args...
  local script="$1"
  shift
  SEED="${SEED:?SEED is required}" "${PYTHON_BIN}" - "$script" "$@" <<'PY'
import os, sys, random, runpy
import numpy as np
import torch

seed = int(os.environ.get("SEED", "1"))
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# Keep speed reasonable for long batch runs.
if hasattr(torch.backends, "cudnn"):
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

script = sys.argv[1]
args = sys.argv[2:]
sys.argv = [script] + args
runpy.run_path(script, run_name="__main__")
PY
}

common_train_args() {
  local -n _arr=$1
  _arr=(
    train
    --data_root "$DATA_ROOT"
    --index "$INDEX_CSV"
    --epochs "$EPOCHS"
    --batch "$BATCH"
    --eval_batch "$EVAL_BATCH"
    --workers "$WORKERS"
    --lr "$LR"
    --emb_dim "$EMB_DIM"
    --train_severities "$TRAIN_SEVERITIES"
    --train_alt_types "$TRAIN_ALT_TYPES"
  )
  if [[ -n "$VAL_ALT_TYPES" ]]; then
    _arr+=(--val_alt_types "$VAL_ALT_TYPES")
  fi
  if [[ "$USE_PRETRAINED" == "1" ]]; then
    _arr+=(--pretrained)
  fi
  if [[ "$USE_AMP" == "1" ]]; then
    _arr+=(--amp)
  fi
}

train_one() {
  local method="$1"   # base | mix
  local seed="$2"
  local run_name="v3_obl_${method}_s${seed}"
  local outdir="${OUTPUT_ROOT}/${run_name}"
  local log="${LOG_ROOT}/${run_name}_train.log"
  local ckpt="${outdir}/checkpoints/best.pt"

  if [[ "$SKIP_EXISTING" == "1" && -f "$ckpt" ]]; then
    echo "[SKIP] train ${run_name} (best.pt exists)"
    return 0
  fi

  local script
  local args=()
  common_train_args args
  args+=(--outdir "$outdir")

  if [[ "$method" == "base" ]]; then
    script="$BASE_SCRIPT"
  elif [[ "$method" == "mix" ]]; then
    script="$MIX_SCRIPT"
    args+=(--mix_p "$MIX_P" --mix_alpha "$MIX_ALPHA" --mix_layer "$MIX_LAYER")
  else
    echo "[ERR] Unknown method: ${method}" >&2
    exit 1
  fi

  echo "[RUN] train ${run_name}"
  echo "      script=$script"
  echo "      outdir=$outdir"
  echo "      train_alt_types=$TRAIN_ALT_TYPES train_severities=$TRAIN_SEVERITIES val_alt_types=${VAL_ALT_TYPES:-ALL} seed=$seed"
  mkdir -p "$outdir"
  SEED="$seed" run_seeded_python "$script" "${args[@]}" 2>&1 | tee "$log"
}

eval_one() {
  local method="$1"
  local seed="$2"
  local probe="$3"    # CR | Zcut
  local run_name="v3_obl_${method}_s${seed}"
  local outdir="${OUTPUT_ROOT}/${run_name}"
  local ckpt="${outdir}/checkpoints/best.pt"
  local out_json="${outdir}/test_unseen${probe}.json"
  local log="${LOG_ROOT}/${run_name}_eval_${probe}.log"

  if [[ ! -f "$ckpt" ]]; then
    echo "[WARN] skip eval ${run_name} on ${probe} because checkpoint not found: $ckpt"
    return 0
  fi
  if [[ "$SKIP_EXISTING" == "1" && -f "$out_json" ]]; then
    echo "[SKIP] eval ${run_name} on ${probe} (${out_json} exists)"
    return 0
  fi

  local script
  if [[ "$method" == "base" ]]; then
    script="$BASE_SCRIPT"
  elif [[ "$method" == "mix" ]]; then
    script="$MIX_SCRIPT"
  else
    echo "[ERR] Unknown method: ${method}" >&2
    exit 1
  fi

  echo "[RUN] eval ${run_name} on unseen ${probe}"
  SEED="$seed" run_seeded_python "$script" eval \
    --data_root "$DATA_ROOT" \
    --index "$INDEX_CSV" \
    --ckpt "$ckpt" \
    --split test \
    --batch "$EVAL_BATCH" \
    --workers "$WORKERS" \
    --probe_alt_types "$probe" \
    --probe_severities "$PROBE_SEVERITIES" \
    --out_json "$out_json" \
    2>&1 | tee "$log"
}

summarize_results() {
  local detail_csv="${OUTPUT_ROOT}/phase2_detail.csv"
  local summary_csv="${OUTPUT_ROOT}/phase2_summary.csv"
  local pretty_txt="${OUTPUT_ROOT}/phase2_summary.txt"

  "${PYTHON_BIN}" - "$OUTPUT_ROOT" "$detail_csv" "$summary_csv" "$pretty_txt" <<'PY'
import json, math, statistics, sys
from pathlib import Path
import pandas as pd

output_root = Path(sys.argv[1])
detail_csv = Path(sys.argv[2])
summary_csv = Path(sys.argv[3])
pretty_txt = Path(sys.argv[4])

rows = []
for run_dir in sorted(output_root.glob("v3_obl_*_s*")):
    parts = run_dir.name.split("_")
    if len(parts) < 5:
        continue
    method = parts[2]  # base / mix
    seed = int(parts[-1].replace("s", ""))
    for probe in ["CR", "Zcut"]:
        f = run_dir / f"test_unseen{probe}.json"
        if not f.exists():
            continue
        data = json.loads(f.read_text())
        for sev, m in data.items():
            if not isinstance(m, dict) or "rank1" not in m:
                continue
            rows.append({
                "run": run_dir.name,
                "method": method,
                "seed": seed,
                "probe": probe,
                "severity": sev,
                "n_probe": m.get("n_probe"),
                "rank1": m.get("rank1"),
                "rank5": m.get("rank5"),
                "eer": m.get("eer"),
                "tar@far=0.01": m.get("tar@far=0.01"),
                "tar@far=0.001": m.get("tar@far=0.001"),
            })

if not rows:
    print("[WARN] No JSON evaluation files found. Nothing to summarize.")
    pd.DataFrame([]).to_csv(detail_csv, index=False)
    pd.DataFrame([]).to_csv(summary_csv, index=False)
    pretty_txt.write_text("No evaluation JSON files found.\n")
    sys.exit(0)

df = pd.DataFrame(rows).sort_values(["method", "probe", "severity", "seed"])
df.to_csv(detail_csv, index=False)

agg_rows = []
metrics = ["rank1", "rank5", "eer", "tar@far=0.01", "tar@far=0.001"]
for (method, probe, severity), g in df.groupby(["method", "probe", "severity"]):
    row = {"method": method, "probe": probe, "severity": severity, "n_runs": len(g)}
    for metric in metrics:
        vals = [float(x) for x in g[metric].dropna().tolist()]
        row[f"{metric}_mean"] = sum(vals) / len(vals) if vals else math.nan
        row[f"{metric}_std"] = statistics.stdev(vals) if len(vals) >= 2 else 0.0
    agg_rows.append(row)

summary = pd.DataFrame(agg_rows).sort_values(["probe", "severity", "method"])
summary.to_csv(summary_csv, index=False)

lines = []
lines.append("Phase 2 summary (mean ± std across seeds)\n")
for probe in ["CR", "Zcut"]:
    lines.append(f"[Probe = unseen {probe}]")
    sub = summary[summary["probe"] == probe]
    if sub.empty:
        lines.append("  No data")
        lines.append("")
        continue
    for severity in ["easy", "medium", "hard"]:
        ss = sub[sub["severity"] == severity]
        if ss.empty:
            continue
        lines.append(f"  Severity = {severity}")
        for _, r in ss.iterrows():
            lines.append(
                f"    {r['method']:<4} | "
                f"rank1 {r['rank1_mean']:.4f}±{r['rank1_std']:.4f} | "
                f"rank5 {r['rank5_mean']:.4f}±{r['rank5_std']:.4f} | "
                f"eer {r['eer_mean']:.4f}±{r['eer_std']:.4f} | "
                f"tar@0.001 {r['tar@far=0.001_mean']:.4f}±{r['tar@far=0.001_std']:.4f}"
            )
        lines.append("")

pretty_txt.write_text("\n".join(lines) + "\n")
print(f"[OK] wrote detail  -> {detail_csv}")
print(f"[OK] wrote summary -> {summary_csv}")
print(f"[OK] wrote text    -> {pretty_txt}")
print("\n".join(lines))
PY
}

print_config() {
  cat <<EOF
[CONFIG]
  MODE              = ${MODE}
  PROJECT_ROOT      = ${PROJECT_ROOT}
  PYTHON_BIN        = ${PYTHON_BIN}
  DATA_ROOT         = ${DATA_ROOT}
  INDEX_CSV         = ${INDEX_CSV}
  OUTPUT_ROOT       = ${OUTPUT_ROOT}
  TRAIN_ALT_TYPES   = ${TRAIN_ALT_TYPES}
  TRAIN_SEVERITIES  = ${TRAIN_SEVERITIES}
  VAL_ALT_TYPES     = ${VAL_ALT_TYPES:-ALL}
  PROBES            = ${PROBES[*]}
  PROBE_SEVERITIES  = ${PROBE_SEVERITIES}
  SEEDS             = ${SEEDS[*]}
  EPOCHS            = ${EPOCHS}
  BATCH             = ${BATCH}
  EVAL_BATCH        = ${EVAL_BATCH}
  WORKERS           = ${WORKERS}
  LR                = ${LR}
  EMB_DIM           = ${EMB_DIM}
  USE_PRETRAINED    = ${USE_PRETRAINED}
  USE_AMP           = ${USE_AMP}
  MIX_P             = ${MIX_P}
  MIX_ALPHA         = ${MIX_ALPHA}
  MIX_LAYER         = ${MIX_LAYER}
  SKIP_EXISTING     = ${SKIP_EXISTING}
EOF
}

# -----------------------------
# Sanity checks
# -----------------------------
require_dir "$DATA_ROOT"
require_file "$INDEX_CSV"
require_file "$BASE_SCRIPT"
require_file "$MIX_SCRIPT"
print_config

# -----------------------------
# Main
# -----------------------------
case "$MODE" in
  all)
    for seed in "${SEEDS[@]}"; do
      train_one base "$seed"
      train_one mix  "$seed"
    done
    for seed in "${SEEDS[@]}"; do
      for probe in "${PROBES[@]}"; do
        eval_one base "$seed" "$probe"
        eval_one mix  "$seed" "$probe"
      done
    done
    summarize_results
    ;;
  train)
    for seed in "${SEEDS[@]}"; do
      train_one base "$seed"
      train_one mix  "$seed"
    done
    ;;
  eval)
    for seed in "${SEEDS[@]}"; do
      for probe in "${PROBES[@]}"; do
        eval_one base "$seed" "$probe"
        eval_one mix  "$seed" "$probe"
      done
    done
    ;;
  summarize)
    summarize_results
    ;;
  *)
    echo "Usage: bash run_phase2.sh [all|train|eval|summarize]" >&2
    exit 1
    ;;
esac

# Notes:
# - This script covers the CORE phase-2 matrix (baseline vs mixstyle, multi-seed, unseen CR/Zcut).
# - It does NOT yet do checkpoint-criterion ablation because the current Python scripts only save
#   one best checkpoint (best val mean rank1). For that next step, we should patch the Python code
#   to save per-epoch checkpoints or save multiple 'best' checkpoints by criterion.
