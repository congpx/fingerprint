#!/usr/bin/env bash
set -euo pipefail

# Phase 5: final 5-seed confirmation on unseen CR
# Models compared:
#   1) base_ref        = baseline from v3 (no MixStyle)
#   2) mix_ref         = previous MixStyle winner from Phase 3 (layer1, p=0.5, a=0.3)
#   3) mix_best        = current MixStyle winner from Phase 4 (layer1, p=0.7, a=0.3)
# Optional env overrides are supported, but default usage is simply:
#   bash /home/congpx/fingerprint/run_phase5_final_confirm.sh

MODE="${1:-all}"   # all | train | eval | summarize

PROJECT_ROOT="/home/congpx/fingerprint"
PYTHON_BIN="${PYTHON_BIN:-python}"

DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/data/SOCOFing/SOCOFing}"
INDEX_CSV="${INDEX_CSV:-${PROJECT_ROOT}/splits/index_v3.csv}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/outputs_phase5_final}"

SEEDS_STR="${SEEDS:-1 2 3 4 5}"
read -r -a SEEDS_ARR <<< "${SEEDS_STR}"

TRAIN_ALT_TYPES="${TRAIN_ALT_TYPES:-Obl}"
TRAIN_SEVERITIES="${TRAIN_SEVERITIES:-real,easy,medium,hard}"
VAL_ALT_TYPES="${VAL_ALT_TYPES:-CR}"
PROBES_STR="${PROBES:-CR}"
read -r -a PROBES_ARR <<< "${PROBES_STR}"

EPOCHS="${EPOCHS:-30}"
BATCH="${BATCH:-128}"
EVAL_BATCH="${EVAL_BATCH:-256}"
WORKERS="${WORKERS:-8}"
LR="${LR:-1e-4}"
EMB_DIM="${EMB_DIM:-256}"
USE_PRETRAINED="${USE_PRETRAINED:-1}"
USE_AMP="${USE_AMP:-1}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

# Reference MixStyle config from Phase 3
MIX_REF_P="${MIX_REF_P:-0.5}"
MIX_REF_ALPHA="${MIX_REF_ALPHA:-0.3}"
MIX_REF_LAYER="${MIX_REF_LAYER:-layer1}"

# Best MixStyle config from Phase 4
MIX_BEST_P="${MIX_BEST_P:-0.7}"
MIX_BEST_ALPHA="${MIX_BEST_ALPHA:-0.3}"
MIX_BEST_LAYER="${MIX_BEST_LAYER:-layer1}"

mkdir -p "${OUTPUT_ROOT}"

print_config() {
  echo "[CONFIG]"
  echo "  MODE              = ${MODE}"
  echo "  PROJECT_ROOT      = ${PROJECT_ROOT}"
  echo "  PYTHON_BIN        = ${PYTHON_BIN}"
  echo "  DATA_ROOT         = ${DATA_ROOT}"
  echo "  INDEX_CSV         = ${INDEX_CSV}"
  echo "  OUTPUT_ROOT       = ${OUTPUT_ROOT}"
  echo "  TRAIN_ALT_TYPES   = ${TRAIN_ALT_TYPES}"
  echo "  TRAIN_SEVERITIES  = ${TRAIN_SEVERITIES}"
  echo "  VAL_ALT_TYPES     = ${VAL_ALT_TYPES}"
  echo "  PROBES            = ${PROBES_STR}"
  echo "  SEEDS             = ${SEEDS_STR}"
  echo "  EPOCHS            = ${EPOCHS}"
  echo "  BATCH             = ${BATCH}"
  echo "  EVAL_BATCH        = ${EVAL_BATCH}"
  echo "  WORKERS           = ${WORKERS}"
  echo "  LR                = ${LR}"
  echo "  EMB_DIM           = ${EMB_DIM}"
  echo "  USE_PRETRAINED    = ${USE_PRETRAINED}"
  echo "  USE_AMP           = ${USE_AMP}"
  echo "  SKIP_EXISTING     = ${SKIP_EXISTING}"
  echo "  MIX_REF           = p=${MIX_REF_P}, a=${MIX_REF_ALPHA}, layer=${MIX_REF_LAYER}"
  echo "  MIX_BEST          = p=${MIX_BEST_P}, a=${MIX_BEST_ALPHA}, layer=${MIX_BEST_LAYER}"
}

run_train_base() {
  local seed="$1"
  local run_name="base_ref_s${seed}"
  local out_dir="${OUTPUT_ROOT}/${run_name}"
  local ckpt="${out_dir}/checkpoints/best.pt"

  if [[ "${SKIP_EXISTING}" = "1" && -f "${ckpt}" ]]; then
    echo "[SKIP] train ${run_name} (best.pt exists)"
    return
  fi

  mkdir -p "${out_dir}"
  echo "[RUN] train ${run_name}"
  PYTHONHASHSEED="${seed}" CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  ${PYTHON_BIN} "${PROJECT_ROOT}/run_socofing_v3.py" train \
    --data_root "${DATA_ROOT}" \
    --index "${INDEX_CSV}" \
    --outdir "${out_dir}" \
    --train_alt_types "${TRAIN_ALT_TYPES}" \
    --val_alt_types "${VAL_ALT_TYPES}" \
    --train_severities "${TRAIN_SEVERITIES}" \
    --epochs "${EPOCHS}" \
    --batch "${BATCH}" \
    --eval_batch "${EVAL_BATCH}" \
    --workers "${WORKERS}" \
    --lr "${LR}" \
    --emb_dim "${EMB_DIM}" \
    $( [[ "${USE_PRETRAINED}" = "1" ]] && echo "--pretrained" ) \
    $( [[ "${USE_AMP}" = "1" ]] && echo "--amp" )
}

run_train_mix() {
  local seed="$1"
  local variant="$2"  # ref | best

  local mix_p mix_alpha mix_layer run_name
  if [[ "${variant}" = "ref" ]]; then
    mix_p="${MIX_REF_P}"
    mix_alpha="${MIX_REF_ALPHA}"
    mix_layer="${MIX_REF_LAYER}"
    run_name="mix_ref_s${seed}"
  else
    mix_p="${MIX_BEST_P}"
    mix_alpha="${MIX_BEST_ALPHA}"
    mix_layer="${MIX_BEST_LAYER}"
    run_name="mix_best_s${seed}"
  fi

  local out_dir="${OUTPUT_ROOT}/${run_name}"
  local ckpt="${out_dir}/checkpoints/best.pt"

  if [[ "${SKIP_EXISTING}" = "1" && -f "${ckpt}" ]]; then
    echo "[SKIP] train ${run_name} (best.pt exists)"
    return
  fi

  mkdir -p "${out_dir}"
  echo "[RUN] train ${run_name} [p=${mix_p}, a=${mix_alpha}, layer=${mix_layer}]"
  PYTHONHASHSEED="${seed}" CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  ${PYTHON_BIN} "${PROJECT_ROOT}/run_socofing_v3_mixstyle.py" train \
    --data_root "${DATA_ROOT}" \
    --index "${INDEX_CSV}" \
    --outdir "${out_dir}" \
    --train_alt_types "${TRAIN_ALT_TYPES}" \
    --val_alt_types "${VAL_ALT_TYPES}" \
    --train_severities "${TRAIN_SEVERITIES}" \
    --epochs "${EPOCHS}" \
    --batch "${BATCH}" \
    --eval_batch "${EVAL_BATCH}" \
    --workers "${WORKERS}" \
    --lr "${LR}" \
    --emb_dim "${EMB_DIM}" \
    --mix_p "${mix_p}" \
    --mix_alpha "${mix_alpha}" \
    --mix_layer "${mix_layer}" \
    $( [[ "${USE_PRETRAINED}" = "1" ]] && echo "--pretrained" ) \
    $( [[ "${USE_AMP}" = "1" ]] && echo "--amp" )
}

run_eval_one() {
  local run_name="$1"
  local script="$2"
  local probe="$3"

  local run_dir="${OUTPUT_ROOT}/${run_name}"
  local ckpt="${run_dir}/checkpoints/best.pt"
  local out_json="${run_dir}/test_unseen${probe}.json"

  if [[ ! -f "${ckpt}" ]]; then
    echo "[WARN] missing ckpt: ${ckpt}"
    return
  fi

  if [[ "${SKIP_EXISTING}" = "1" && -f "${out_json}" ]]; then
    echo "[SKIP] eval ${run_name} on ${probe} (${out_json} exists)"
    return
  fi

  echo "[RUN] eval ${run_name} on unseen ${probe}"
  ${PYTHON_BIN} "${PROJECT_ROOT}/${script}" eval \
    --data_root "${DATA_ROOT}" \
    --index "${INDEX_CSV}" \
    --ckpt "${ckpt}" \
    --split test \
    --batch "${EVAL_BATCH}" \
    --workers "${WORKERS}" \
    --probe_alt_types "${probe}" \
    --out_json "${out_json}"
}

summarize_results() {
  ${PYTHON_BIN} - <<'PY'
import csv, json, math, statistics
from pathlib import Path

root = Path("/home/congpx/fingerprint/outputs_phase5_final")
rows = []

# Expected run names: base_ref_s1, mix_ref_s3, mix_best_s5
for run_dir in sorted(root.glob("*_s*")):
    if not run_dir.is_dir():
        continue
    name = run_dir.name
    parts = name.split("_")
    if len(parts) < 3:
        continue
    method = "_".join(parts[:-1])
    try:
        seed = int(parts[-1].replace("s", ""))
    except ValueError:
        continue

    for jf in sorted(run_dir.glob("test_unseen*.json")):
        probe = jf.stem.replace("test_unseen", "")
        data = json.loads(jf.read_text())
        for severity, m in data.items():
            if not isinstance(m, dict):
                continue
            rows.append({
                "run": name,
                "method": method,
                "seed": seed,
                "probe": probe,
                "severity": severity,
                "rank1": m.get("rank1"),
                "rank5": m.get("rank5"),
                "eer": m.get("eer"),
                "tar001": m.get("tar@far=0.001"),
            })

if not rows:
    print("[WARN] No JSON evaluation files found. Nothing to summarize.")
    raise SystemExit(0)

detail_csv = root / "phase5_final_detail.csv"
with detail_csv.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["run","method","seed","probe","severity","rank1","rank5","eer","tar001"])
    writer.writeheader()
    writer.writerows(rows)

summary = {}
for r in rows:
    key = (r["method"], r["probe"], r["severity"])
    summary.setdefault(key, {"rank1": [], "eer": [], "tar001": []})
    for metric in ["rank1", "eer", "tar001"]:
        v = r.get(metric)
        if v is not None:
            summary[key][metric].append(v)

summary_rows = []
for (method, probe, severity), vals in sorted(summary.items()):
    summary_rows.append({
        "method": method,
        "probe": probe,
        "severity": severity,
        "rank1_mean": statistics.mean(vals["rank1"]) if vals["rank1"] else None,
        "rank1_std": statistics.stdev(vals["rank1"]) if len(vals["rank1"]) > 1 else 0.0,
        "eer_mean": statistics.mean(vals["eer"]) if vals["eer"] else None,
        "eer_std": statistics.stdev(vals["eer"]) if len(vals["eer"]) > 1 else 0.0,
        "tar001_mean": statistics.mean(vals["tar001"]) if vals["tar001"] else None,
        "tar001_std": statistics.stdev(vals["tar001"]) if len(vals["tar001"]) > 1 else 0.0,
    })

summary_csv = root / "phase5_final_summary.csv"
with summary_csv.open("w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "method","probe","severity",
            "rank1_mean","rank1_std",
            "eer_mean","eer_std",
            "tar001_mean","tar001_std",
        ],
    )
    writer.writeheader()
    writer.writerows(summary_rows)

# Ranking on hard/CR: primary tar001 desc, secondary eer asc, tertiary rank1 desc
hard_cr = [r for r in summary_rows if r["probe"] == "CR" and r["severity"] == "hard"]
hard_cr_sorted = sorted(hard_cr, key=lambda x: (-x["tar001_mean"], x["eer_mean"], -x["rank1_mean"]))

summary_txt = root / "phase5_final_summary.txt"
with summary_txt.open("w") as f:
    f.write("=== HARD / unseen CR ===\n")
    for r in hard_cr_sorted:
        f.write(
            f'{r["method"]:>8} | '
            f'rank1={r["rank1_mean"]:.4f}±{r["rank1_std"]:.4f} | '
            f'eer={r["eer_mean"]:.4f}±{r["eer_std"]:.4f} | '
            f'tar001={r["tar001_mean"]:.4f}±{r["tar001_std"]:.4f}\n'
        )
    if hard_cr_sorted:
        best = hard_cr_sorted[0]
        f.write("\n=== WINNER ===\n")
        f.write(
            f'{best["method"]} (probe={best["probe"]}, severity={best["severity"]}) -> '
            f'rank1={best["rank1_mean"]:.4f}±{best["rank1_std"]:.4f}, '
            f'eer={best["eer_mean"]:.4f}±{best["eer_std"]:.4f}, '
            f'tar001={best["tar001_mean"]:.4f}±{best["tar001_std"]:.4f}\n'
        )

print(f"[OK] wrote {detail_csv}")
print(f"[OK] wrote {summary_csv}")
print(f"[OK] wrote {summary_txt}")
PY
}

print_config

if [[ "${MODE}" = "all" || "${MODE}" = "train" ]]; then
  for seed in "${SEEDS_ARR[@]}"; do
    run_train_base "${seed}"
    run_train_mix "${seed}" ref
    run_train_mix "${seed}" best
  done
fi

if [[ "${MODE}" = "all" || "${MODE}" = "eval" ]]; then
  for seed in "${SEEDS_ARR[@]}"; do
    for probe in "${PROBES_ARR[@]}"; do
      run_eval_one "base_ref_s${seed}" "run_socofing_v3.py" "${probe}"
      run_eval_one "mix_ref_s${seed}" "run_socofing_v3_mixstyle.py" "${probe}"
      run_eval_one "mix_best_s${seed}" "run_socofing_v3_mixstyle.py" "${probe}"
    done
  done
fi

if [[ "${MODE}" = "all" || "${MODE}" = "summarize" ]]; then
  summarize_results
fi
