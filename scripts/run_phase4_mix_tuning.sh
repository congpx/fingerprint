#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-all}"   # all | screen_train | screen_eval | screen_summarize | confirm_train | confirm_eval | confirm_summarize | summarize_all

PROJECT_ROOT="${PROJECT_ROOT:-/home/congpx/fingerprint}"
PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/data/SOCOFing/SOCOFing}"
INDEX_CSV="${INDEX_CSV:-${PROJECT_ROOT}/splits/index_v3.csv}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/outputs_phase4_mix_tuning}"

TRAIN_ALT_TYPES="${TRAIN_ALT_TYPES:-Obl}"
TRAIN_SEVERITIES="${TRAIN_SEVERITIES:-real,easy,medium,hard}"
VAL_ALT_TYPES="${VAL_ALT_TYPES:-CR}"
PROBE_ALT_TYPES="${PROBE_ALT_TYPES:-CR}"
SCREEN_PROBE_SEVERITIES="${SCREEN_PROBE_SEVERITIES:-hard}"
CONFIRM_PROBE_SEVERITIES="${CONFIRM_PROBE_SEVERITIES:-hard}"

EPOCHS="${EPOCHS:-30}"
BATCH="${BATCH:-128}"
EVAL_BATCH="${EVAL_BATCH:-256}"
WORKERS="${WORKERS:-8}"
LR="${LR:-1e-4}"
EMB_DIM="${EMB_DIM:-256}"
USE_PRETRAINED="${USE_PRETRAINED:-1}"
USE_AMP="${USE_AMP:-1}"

# screening: 6 configs, seed=1 only
SCREEN_SEED="${SCREEN_SEED:-1}"
# confirm: only top-2 configs, default seeds 1,2,3 (seed1 can be reused from screening)
CONFIRM_SEEDS_STR="${CONFIRM_SEEDS:-1 2 3}"
IFS=' ' read -r -a CONFIRM_SEEDS <<< "${CONFIRM_SEEDS_STR}"

# Explicit top-2 configs for confirm. Format: name:p:alpha:layer,name:p:alpha:layer
# Leave empty to auto-pick from screen summary by highest tar001, then lowest eer, then highest rank1.
TOP2_CONFIGS="${TOP2_CONFIGS:-}"

SKIP_EXISTING="${SKIP_EXISTING:-1}"

mkdir -p "${OUTPUT_ROOT}"

SCRIPT_TRAIN="${PROJECT_ROOT}/run_socofing_v3_mixstyle.py"

screen_configs=(
  "mix_l1_p03_a03:0.3:0.3:layer1"
  "mix_l1_p05_a03:0.5:0.3:layer1"
  "mix_l1_p07_a03:0.7:0.3:layer1"
  "mix_l2_p05_a03:0.5:0.3:layer2"
  "mix_l3_p05_a03:0.5:0.3:layer3"
  "mix_l1_p05_a01:0.5:0.1:layer1"
)

log_config() {
  echo "[CONFIG]"
  echo "  MODE                  = ${MODE}"
  echo "  PROJECT_ROOT          = ${PROJECT_ROOT}"
  echo "  DATA_ROOT             = ${DATA_ROOT}"
  echo "  INDEX_CSV             = ${INDEX_CSV}"
  echo "  OUTPUT_ROOT           = ${OUTPUT_ROOT}"
  echo "  TRAIN_ALT_TYPES       = ${TRAIN_ALT_TYPES}"
  echo "  TRAIN_SEVERITIES      = ${TRAIN_SEVERITIES}"
  echo "  VAL_ALT_TYPES         = ${VAL_ALT_TYPES}"
  echo "  PROBE_ALT_TYPES       = ${PROBE_ALT_TYPES}"
  echo "  SCREEN_PROBE_SEV      = ${SCREEN_PROBE_SEVERITIES}"
  echo "  CONFIRM_PROBE_SEV     = ${CONFIRM_PROBE_SEVERITIES}"
  echo "  SCREEN_SEED           = ${SCREEN_SEED}"
  echo "  CONFIRM_SEEDS         = ${CONFIRM_SEEDS[*]}"
  echo "  EPOCHS                = ${EPOCHS}"
  echo "  BATCH                 = ${BATCH}"
  echo "  EVAL_BATCH            = ${EVAL_BATCH}"
  echo "  WORKERS               = ${WORKERS}"
  echo "  LR                    = ${LR}"
  echo "  EMB_DIM               = ${EMB_DIM}"
  echo "  USE_PRETRAINED        = ${USE_PRETRAINED}"
  echo "  USE_AMP               = ${USE_AMP}"
  echo "  SKIP_EXISTING         = ${SKIP_EXISTING}"
  if [[ -n "${TOP2_CONFIGS}" ]]; then
    echo "  TOP2_CONFIGS          = ${TOP2_CONFIGS}"
  else
    echo "  TOP2_CONFIGS          = <auto from screen summary>"
  fi
}

train_one() {
  local name="$1"
  local p="$2"
  local alpha="$3"
  local layer="$4"
  local seed="$5"

  local run_dir="${OUTPUT_ROOT}/${name}_s${seed}"
  local ckpt="${run_dir}/checkpoints/best.pt"

  if [[ "${SKIP_EXISTING}" = "1" && -f "${ckpt}" ]]; then
    echo "[SKIP] train ${name}_s${seed} (best.pt exists)"
    return
  fi

  mkdir -p "${run_dir}"
  echo "[RUN] train ${name}_s${seed} (p=${p}, alpha=${alpha}, layer=${layer})"

  PYTHONHASHSEED="${seed}" \
  CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  "${PYTHON_BIN}" "${SCRIPT_TRAIN}" train \
    --data_root "${DATA_ROOT}" \
    --index "${INDEX_CSV}" \
    --outdir "${run_dir}" \
    --train_alt_types "${TRAIN_ALT_TYPES}" \
    --val_alt_types "${VAL_ALT_TYPES}" \
    --train_severities "${TRAIN_SEVERITIES}" \
    --epochs "${EPOCHS}" \
    --batch "${BATCH}" \
    --eval_batch "${EVAL_BATCH}" \
    --workers "${WORKERS}" \
    --lr "${LR}" \
    --emb_dim "${EMB_DIM}" \
    --mix_p "${p}" \
    --mix_alpha "${alpha}" \
    --mix_layer "${layer}" \
    $( [[ "${USE_PRETRAINED}" = "1" ]] && echo "--pretrained" ) \
    $( [[ "${USE_AMP}" = "1" ]] && echo "--amp" )
}

eval_one() {
  local name="$1"
  local seed="$2"
  local probe_sev="$3"
  local run_dir="${OUTPUT_ROOT}/${name}_s${seed}"
  local ckpt="${run_dir}/checkpoints/best.pt"
  local out_json="${run_dir}/test_unseen${PROBE_ALT_TYPES}_${probe_sev}.json"

  if [[ ! -f "${ckpt}" ]]; then
    echo "[WARN] missing ckpt: ${ckpt}"
    return
  fi
  if [[ "${SKIP_EXISTING}" = "1" && -f "${out_json}" ]]; then
    echo "[SKIP] eval ${name}_s${seed} (${probe_sev})"
    return
  fi

  echo "[RUN] eval ${name}_s${seed} on unseen ${PROBE_ALT_TYPES} severity=${probe_sev}"
  "${PYTHON_BIN}" "${SCRIPT_TRAIN}" eval \
    --data_root "${DATA_ROOT}" \
    --index "${INDEX_CSV}" \
    --ckpt "${ckpt}" \
    --split test \
    --batch "${EVAL_BATCH}" \
    --workers "${WORKERS}" \
    --probe_alt_types "${PROBE_ALT_TYPES}" \
    --probe_severities "${probe_sev}" \
    --out_json "${out_json}"
}

summarize_glob() {
  local glob_prefix="$1"   # e.g. mix_* or mix_l1_p05_a03
  local out_prefix="$2"    # e.g. screen or confirm
  "${PYTHON_BIN}" - <<PY
import json, csv, statistics
from pathlib import Path

root = Path(r"${OUTPUT_ROOT}")
glob_prefix = r"${glob_prefix}"
out_prefix = r"${out_prefix}"
rows = []

for run_dir in sorted(root.glob(f"{glob_prefix}_s*")):
    parts = run_dir.name.rsplit("_s", 1)
    if len(parts) != 2:
        continue
    cfg = parts[0]
    try:
        seed = int(parts[1])
    except ValueError:
        continue
    # collect all per-run jsons for unseen CR
    for jf in sorted(run_dir.glob(f"test_unseen${PROBE_ALT_TYPES}_*.json")):
        tag = jf.stem.replace(f"test_unseen${PROBE_ALT_TYPES}_", "")
        data = json.loads(jf.read_text())
        for sev, m in data.items():
            if not isinstance(m, dict):
                continue
            rows.append({
                "config": cfg,
                "seed": seed,
                "tag": tag,
                "severity": sev,
                "rank1": m.get("rank1"),
                "rank5": m.get("rank5"),
                "eer": m.get("eer"),
                "tar001": m.get("tar@far=0.001"),
            })

detail_csv = root / f"phase4_{out_prefix}_detail.csv"
with detail_csv.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["config","seed","tag","severity","rank1","rank5","eer","tar001"])
    writer.writeheader()
    writer.writerows(rows)

summary = {}
for r in rows:
    key = (r["config"], r["tag"], r["severity"])
    summary.setdefault(key, {"rank1": [], "eer": [], "tar001": []})
    for k in ["rank1", "eer", "tar001"]:
        v = r.get(k)
        if v is not None:
            summary[key][k].append(v)

summary_rows = []
for (cfg, tag, sev), vals in sorted(summary.items()):
    summary_rows.append({
        "config": cfg,
        "tag": tag,
        "severity": sev,
        "rank1_mean": statistics.mean(vals["rank1"]) if vals["rank1"] else None,
        "rank1_std": statistics.stdev(vals["rank1"]) if len(vals["rank1"]) > 1 else 0.0,
        "eer_mean": statistics.mean(vals["eer"]) if vals["eer"] else None,
        "eer_std": statistics.stdev(vals["eer"]) if len(vals["eer"]) > 1 else 0.0,
        "tar001_mean": statistics.mean(vals["tar001"]) if vals["tar001"] else None,
        "tar001_std": statistics.stdev(vals["tar001"]) if len(vals["tar001"]) > 1 else 0.0,
    })

summary_csv = root / f"phase4_{out_prefix}_summary.csv"
with summary_csv.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["config","tag","severity","rank1_mean","rank1_std","eer_mean","eer_std","tar001_mean","tar001_std"])
    writer.writeheader()
    writer.writerows(summary_rows)

txt = root / f"phase4_{out_prefix}_summary.txt"
with txt.open("w") as f:
    f.write(f"=== {out_prefix.upper()} / unseen ${PROBE_ALT_TYPES} ===\n")
    hard_rows = [r for r in summary_rows if r["severity"] == "hard"]
    hard_rows = sorted(hard_rows, key=lambda r: (-r["tar001_mean"], r["eer_mean"], -r["rank1_mean"]))
    for r in hard_rows:
        f.write(
            f'{r["config"]:<18} | tag={r["tag"]:<8} | '
            f'rank1={r["rank1_mean"]:.4f}±{r["rank1_std"]:.4f} | '
            f'eer={r["eer_mean"]:.4f}±{r["eer_std"]:.4f} | '
            f'tar001={r["tar001_mean"]:.4f}±{r["tar001_std"]:.4f}\n'
        )

print(f"[OK] wrote {detail_csv}")
print(f"[OK] wrote {summary_csv}")
print(f"[OK] wrote {txt}")
PY
}

auto_pick_top2() {
  local summary_csv="${OUTPUT_ROOT}/phase4_screen_summary.csv"
  if [[ ! -f "${summary_csv}" ]]; then
    echo "[ERROR] ${summary_csv} not found. Run screen_summarize first or set TOP2_CONFIGS." >&2
    exit 1
  fi
  "${PYTHON_BIN}" - <<PY
import csv
from pathlib import Path
p = Path(r"${summary_csv}")
rows = []
with p.open() as f:
    rd = csv.DictReader(f)
    for r in rd:
        if r["severity"] != "hard":
            continue
        # screening uses single seed -> tag usually 'hard'
        rows.append(r)
rows = sorted(rows, key=lambda r: (-float(r["tar001_mean"]), float(r["eer_mean"]), -float(r["rank1_mean"]), r["config"]))
# unique configs only
picked = []
seen = set()
for r in rows:
    if r["config"] in seen:
        continue
    picked.append(r["config"])
    seen.add(r["config"])
    if len(picked) == 2:
        break
print(",".join(picked))
PY
}

run_screen_train() {
  for cfg in "${screen_configs[@]}"; do
    IFS=':' read -r name p alpha layer <<< "${cfg}"
    train_one "${name}" "${p}" "${alpha}" "${layer}" "${SCREEN_SEED}"
  done
}

run_screen_eval() {
  for cfg in "${screen_configs[@]}"; do
    IFS=':' read -r name p alpha layer <<< "${cfg}"
    eval_one "${name}" "${SCREEN_SEED}" "${SCREEN_PROBE_SEVERITIES}"
  done
}

run_confirm_train() {
  local selected="${TOP2_CONFIGS}"
  if [[ -z "${selected}" ]]; then
    selected="$(auto_pick_top2)"
    echo "[AUTO] top-2 from screening = ${selected}"
  fi
  IFS=',' read -r -a chosen <<< "${selected}"
  for chosen_cfg in "${chosen[@]}"; do
    local found=0
    for cfg in "${screen_configs[@]}"; do
      IFS=':' read -r name p alpha layer <<< "${cfg}"
      if [[ "${name}" = "${chosen_cfg}" ]]; then
        found=1
        for seed in "${CONFIRM_SEEDS[@]}"; do
          train_one "${name}" "${p}" "${alpha}" "${layer}" "${seed}"
        done
      fi
    done
    if [[ "${found}" = "0" ]]; then
      echo "[ERROR] unknown config in TOP2_CONFIGS: ${chosen_cfg}" >&2
      exit 1
    fi
  done
}

run_confirm_eval() {
  local selected="${TOP2_CONFIGS}"
  if [[ -z "${selected}" ]]; then
    selected="$(auto_pick_top2)"
    echo "[AUTO] top-2 from screening = ${selected}"
  fi
  IFS=',' read -r -a chosen <<< "${selected}"
  for chosen_cfg in "${chosen[@]}"; do
    for seed in "${CONFIRM_SEEDS[@]}"; do
      eval_one "${chosen_cfg}" "${seed}" "${CONFIRM_PROBE_SEVERITIES}"
    done
  done
}

log_config

case "${MODE}" in
  all)
    run_screen_train
    run_screen_eval
    summarize_glob "mix_*" "screen"
    run_confirm_train
    run_confirm_eval
    summarize_glob "mix_*" "confirm"
    summarize_glob "mix_*" "all"
    ;;
  screen_train)
    run_screen_train
    ;;
  screen_eval)
    run_screen_eval
    ;;
  screen_summarize)
    summarize_glob "mix_*" "screen"
    ;;
  confirm_train)
    run_confirm_train
    ;;
  confirm_eval)
    run_confirm_eval
    ;;
  confirm_summarize)
    summarize_glob "mix_*" "confirm"
    ;;
  summarize_all)
    summarize_glob "mix_*" "all"
    ;;
  *)
    echo "Unknown MODE=${MODE}" >&2
    echo "Use: all | screen_train | screen_eval | screen_summarize | confirm_train | confirm_eval | confirm_summarize | summarize_all" >&2
    exit 1
    ;;
esac
