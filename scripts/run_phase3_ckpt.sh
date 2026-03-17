#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-all}"   # all | train | eval | summarize

PROJECT_ROOT="/home/congpx/fingerprint"
PYTHON_BIN="python"

DATA_ROOT="${PROJECT_ROOT}/data/SOCOFing/SOCOFing"
INDEX_CSV="${PROJECT_ROOT}/splits/index_v3.csv"
OUTPUT_ROOT="${PROJECT_ROOT}/outputs_phase3"

SEEDS=(1 2 3)

TRAIN_ALT_TYPES="Obl"
TRAIN_SEVERITIES="real,easy,medium,hard"
VAL_ALT_TYPES="CR"
PROBES=("CR")
PROBE_SEVERITIES="easy,medium,hard"

EPOCHS=30
BATCH=128
EVAL_BATCH=256
WORKERS=8
LR="1e-4"
EMB_DIM=256
USE_PRETRAINED=1
USE_AMP=1

MIX_P=0.5
MIX_ALPHA=0.3
MIX_LAYER="layer1"

SKIP_EXISTING=1

mkdir -p "${OUTPUT_ROOT}"

echo "[CONFIG]"
echo "  MODE             = ${MODE}"
echo "  PROJECT_ROOT     = ${PROJECT_ROOT}"
echo "  DATA_ROOT        = ${DATA_ROOT}"
echo "  INDEX_CSV        = ${INDEX_CSV}"
echo "  OUTPUT_ROOT      = ${OUTPUT_ROOT}"
echo "  TRAIN_ALT_TYPES  = ${TRAIN_ALT_TYPES}"
echo "  TRAIN_SEVERITIES = ${TRAIN_SEVERITIES}"
echo "  VAL_ALT_TYPES    = ${VAL_ALT_TYPES}"
echo "  PROBES           = ${PROBES[*]}"
echo "  PROBE_SEVERITIES = ${PROBE_SEVERITIES}"
echo "  SEEDS            = ${SEEDS[*]}"
echo "  EPOCHS           = ${EPOCHS}"
echo "  BATCH            = ${BATCH}"
echo "  EVAL_BATCH       = ${EVAL_BATCH}"
echo "  WORKERS          = ${WORKERS}"
echo "  LR               = ${LR}"
echo "  EMB_DIM          = ${EMB_DIM}"
echo "  MIX_P            = ${MIX_P}"
echo "  MIX_ALPHA        = ${MIX_ALPHA}"
echo "  MIX_LAYER        = ${MIX_LAYER}"

run_train() {
  local method="$1"   # base | mix
  local seed="$2"

  local run_name="v3_obl_${method}_s${seed}"
  local out_dir="${OUTPUT_ROOT}/${run_name}"
  local ckpt_best="${out_dir}/checkpoints/best.pt"

  if [[ "${SKIP_EXISTING}" = "1" && -f "${ckpt_best}" ]]; then
    echo "[SKIP] train ${run_name} (best.pt exists)"
    return
  fi

  mkdir -p "${out_dir}"

  if [[ "${method}" = "base" ]]; then
    local script="${PROJECT_ROOT}/run_socofing_v3.py"
    echo "[RUN] train ${run_name}"
    PYTHONHASHSEED="${seed}" \
    CUBLAS_WORKSPACE_CONFIG=:4096:8 \
    ${PYTHON_BIN} "${script}" train \
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
  else
    local script="${PROJECT_ROOT}/run_socofing_v3_mixstyle.py"
    echo "[RUN] train ${run_name}"
    PYTHONHASHSEED="${seed}" \
    CUBLAS_WORKSPACE_CONFIG=:4096:8 \
    ${PYTHON_BIN} "${script}" train \
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
      --mix_p "${MIX_P}" \
      --mix_alpha "${MIX_ALPHA}" \
      --mix_layer "${MIX_LAYER}" \
      $( [[ "${USE_PRETRAINED}" = "1" ]] && echo "--pretrained" ) \
      $( [[ "${USE_AMP}" = "1" ]] && echo "--amp" )
  fi
}

run_eval_ckpts() {
  local method="$1"
  local seed="$2"

  local run_name="v3_obl_${method}_s${seed}"
  local run_dir="${OUTPUT_ROOT}/${run_name}"

  if [[ "${method}" = "base" ]]; then
    local script="${PROJECT_ROOT}/run_socofing_v3.py"
  else
    local script="${PROJECT_ROOT}/run_socofing_v3_mixstyle.py"
  fi

  for probe in "${PROBES[@]}"; do
    for ckpt_name in best_mean_rank1.pt best_hard_rank1.pt best_hard_tar001.pt; do
      local ckpt="${run_dir}/checkpoints/${ckpt_name}"
      local tag="${ckpt_name%.pt}"
      local out_json="${run_dir}/test_unseen${probe}_${tag}.json"

      if [[ ! -f "${ckpt}" ]]; then
        echo "[WARN] missing ckpt: ${ckpt}"
        continue
      fi

      if [[ "${SKIP_EXISTING}" = "1" && -f "${out_json}" ]]; then
        echo "[SKIP] eval ${run_name} ${probe} ${tag}"
        continue
      fi

      echo "[RUN] eval ${run_name} on unseen ${probe} using ${tag}"
      ${PYTHON_BIN} "${script}" eval \
        --data_root "${DATA_ROOT}" \
        --index "${INDEX_CSV}" \
        --ckpt "${ckpt}" \
        --split test \
        --batch "${EVAL_BATCH}" \
        --workers "${WORKERS}" \
        --probe_alt_types "${probe}" \
        --out_json "${out_json}"
    done
  done
}

summarize_results() {
  ${PYTHON_BIN} - <<'PY'
import json, csv, statistics
from pathlib import Path

root = Path("/home/congpx/fingerprint/outputs_phase3")
rows = []

for run_dir in sorted(root.glob("v3_obl_*_s*")):
    parts = run_dir.name.split("_")
    if len(parts) != 4:
        continue

    method = parts[2]
    seed = int(parts[3].replace("s", ""))

    for jf in run_dir.glob("test_unseenCR_*.json"):
        crit = jf.stem.replace("test_unseenCR_", "")
        data = json.loads(jf.read_text())

        for sev, m in data.items():
            if not isinstance(m, dict):
                continue
            rows.append({
                "run": run_dir.name,
                "method": method,
                "seed": seed,
                "criterion": crit,
                "severity": sev,
                "rank1": m.get("rank1"),
                "rank5": m.get("rank5"),
                "eer": m.get("eer"),
                "tar001": m.get("tar@far=0.001"),
            })

if not rows:
    print("[WARN] No JSON evaluation files found. Nothing to summarize.")
    raise SystemExit(0)

detail_csv = root / "phase3_ckpt_detail.csv"
with detail_csv.open("w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["run","method","seed","criterion","severity","rank1","rank5","eer","tar001"]
    )
    writer.writeheader()
    writer.writerows(rows)

summary = {}
for r in rows:
    key = (r["method"], r["criterion"], r["severity"])
    summary.setdefault(key, {"rank1": [], "eer": [], "tar001": []})
    for k in ["rank1", "eer", "tar001"]:
        v = r.get(k)
        if v is not None:
            summary[key][k].append(v)

summary_rows = []
for (method, criterion, severity), vals in sorted(summary.items()):
    row = {
        "method": method,
        "criterion": criterion,
        "severity": severity,
        "rank1_mean": statistics.mean(vals["rank1"]) if vals["rank1"] else None,
        "rank1_std": statistics.stdev(vals["rank1"]) if len(vals["rank1"]) > 1 else 0.0,
        "eer_mean": statistics.mean(vals["eer"]) if vals["eer"] else None,
        "eer_std": statistics.stdev(vals["eer"]) if len(vals["eer"]) > 1 else 0.0,
        "tar001_mean": statistics.mean(vals["tar001"]) if vals["tar001"] else None,
        "tar001_std": statistics.stdev(vals["tar001"]) if len(vals["tar001"]) > 1 else 0.0,
    }
    summary_rows.append(row)

summary_csv = root / "phase3_ckpt_summary.csv"
with summary_csv.open("w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "method","criterion","severity",
            "rank1_mean","rank1_std",
            "eer_mean","eer_std",
            "tar001_mean","tar001_std"
        ]
    )
    writer.writeheader()
    writer.writerows(summary_rows)

txt = root / "phase3_ckpt_summary.txt"
with txt.open("w") as f:
    f.write("=== HARD / unseen CR ===\n")
    hard_rows = [r for r in summary_rows if r["severity"] == "hard"]
    for r in sorted(hard_rows, key=lambda x: (x["method"], x["criterion"])):
        f.write(
            f'{r["method"]:>4} | {r["criterion"]:<18} | '
            f'rank1={r["rank1_mean"]:.4f}±{r["rank1_std"]:.4f} | '
            f'eer={r["eer_mean"]:.4f}±{r["eer_std"]:.4f} | '
            f'tar001={r["tar001_mean"]:.4f}±{r["tar001_std"]:.4f}\n'
        )

print(f"[OK] wrote {detail_csv}")
print(f"[OK] wrote {summary_csv}")
print(f"[OK] wrote {txt}")
PY
}

if [[ "${MODE}" = "all" || "${MODE}" = "train" ]]; then
  for seed in "${SEEDS[@]}"; do
    run_train base "${seed}"
    run_train mix "${seed}"
  done
fi

if [[ "${MODE}" = "all" || "${MODE}" = "eval" ]]; then
  for seed in "${SEEDS[@]}"; do
    run_eval_ckpts base "${seed}"
    run_eval_ckpts mix "${seed}"
  done
fi

if [[ "${MODE}" = "all" || "${MODE}" = "summarize" ]]; then
  summarize_results
fi
