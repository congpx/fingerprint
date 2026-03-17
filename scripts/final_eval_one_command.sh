#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-all}"   # all | eval | summarize

PROJECT_ROOT="/home/congpx/fingerprint"
PYTHON_BIN="python"

DATA_ROOT="${PROJECT_ROOT}/data/SOCOFing/SOCOFing"
INDEX_CSV="${PROJECT_ROOT}/splits/index_v3.csv"
WINNER_ROOT="${PROJECT_ROOT}/outputs_phase5_final"
OUT_ROOT="${PROJECT_ROOT}/outputs_final_eval"

SEEDS=(1 2 3 4 5)
RUN_PREFIX="mix_best"
CKPT_NAME="best.pt"

EVAL_BATCH=256
WORKERS=8
SKIP_EXISTING=1

mkdir -p "${OUT_ROOT}"

print_config() {
  echo "[CONFIG]"
  echo "  MODE         = ${MODE}"
  echo "  PROJECT_ROOT = ${PROJECT_ROOT}"
  echo "  DATA_ROOT    = ${DATA_ROOT}"
  echo "  INDEX_CSV    = ${INDEX_CSV}"
  echo "  WINNER_ROOT  = ${WINNER_ROOT}"
  echo "  OUT_ROOT     = ${OUT_ROOT}"
  echo "  RUN_PREFIX   = ${RUN_PREFIX}"
  echo "  CKPT_NAME    = ${CKPT_NAME}"
  echo "  SEEDS        = ${SEEDS[*]}"
  echo "  EVAL_BATCH   = ${EVAL_BATCH}"
  echo "  WORKERS      = ${WORKERS}"
}

run_eval_standard() {
  local seed="$1"
  local probe="$2"   # CR | Zcut
  local run_dir="${WINNER_ROOT}/${RUN_PREFIX}_s${seed}"
  local ckpt="${run_dir}/checkpoints/${CKPT_NAME}"
  local out_json="${OUT_ROOT}/${RUN_PREFIX}_s${seed}_unseen${probe}.json"

  if [[ ! -f "${ckpt}" ]]; then
    echo "[WARN] missing ckpt: ${ckpt}"
    return
  fi

  if [[ "${SKIP_EXISTING}" = "1" && -f "${out_json}" ]]; then
    echo "[SKIP] unseen ${probe} for ${RUN_PREFIX}_s${seed}"
    return
  fi

  echo "[RUN] unseen ${probe} for ${RUN_PREFIX}_s${seed}"
  ${PYTHON_BIN} "${PROJECT_ROOT}/run_socofing_v3_mixstyle.py" eval \
    --data_root "${DATA_ROOT}" \
    --index "${INDEX_CSV}" \
    --ckpt "${ckpt}" \
    --split test \
    --batch "${EVAL_BATCH}" \
    --workers "${WORKERS}" \
    --probe_alt_types "${probe}" \
    --out_json "${out_json}"
}

run_eval_fullgallery() {
  local seed="$1"
  local run_dir="${WINNER_ROOT}/${RUN_PREFIX}_s${seed}"
  local ckpt="${run_dir}/checkpoints/${CKPT_NAME}"
  local out_json="${OUT_ROOT}/${RUN_PREFIX}_s${seed}_fullgallery.json"

  if [[ ! -f "${ckpt}" ]]; then
    echo "[WARN] missing ckpt: ${ckpt}"
    return
  fi

  if [[ "${SKIP_EXISTING}" = "1" && -f "${out_json}" ]]; then
    echo "[SKIP] full-gallery for ${RUN_PREFIX}_s${seed}"
    return
  fi

  echo "[RUN] full-gallery for ${RUN_PREFIX}_s${seed}"
  ${PYTHON_BIN} "${PROJECT_ROOT}/eval_fullgallery_compat.py" \
    --data_root "${DATA_ROOT}" \
    --index "${INDEX_CSV}" \
    --ckpt "${ckpt}" \
    --probe_split test \
    --batch "${EVAL_BATCH}" \
    --workers "${WORKERS}" \
    --out_json "${out_json}"
}

summarize_results() {
  ${PYTHON_BIN} - <<'PY'
import csv, json, statistics
from pathlib import Path

out_root = Path("/home/congpx/fingerprint/outputs_final_eval")
rows = []

# unseen CR / unseen Zcut
for jf in sorted(out_root.glob("mix_best_s*_unseen*.json")):
    stem = jf.stem
    parts = stem.split("_")
    # mix_best_s1_unseenCR
    if len(parts) < 4:
        continue
    seed = int(parts[2].replace("s", ""))
    probe = parts[3].replace("unseen", "")
    data = json.loads(jf.read_text())
    for sev, m in data.items():
        if not isinstance(m, dict):
            continue
        rows.append({
            "eval_type": probe,
            "seed": seed,
            "severity": sev,
            "rank1": m.get("rank1"),
            "rank5": m.get("rank5"),
            "eer": m.get("eer"),
            "tar001": m.get("tar@far=0.001"),
        })

# full-gallery
for jf in sorted(out_root.glob("mix_best_s*_fullgallery.json")):
    stem = jf.stem
    # mix_best_s1_fullgallery
    parts = stem.split("_")
    if len(parts) < 4:
        continue
    seed = int(parts[2].replace("s", ""))
    data = json.loads(jf.read_text())
    for sev, m in data.items():
        if not isinstance(m, dict):
            continue
        rows.append({
            "eval_type": "fullgallery",
            "seed": seed,
            "severity": sev,
            "rank1": m.get("rank1"),
            "rank5": m.get("rank5"),
            "eer": m.get("eer"),
            "tar001": m.get("tar@far=0.001"),
        })

if not rows:
    print("[WARN] No JSON files found. Nothing to summarize.")
    raise SystemExit(0)

detail_csv = out_root / "final_eval_detail.csv"
with detail_csv.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["eval_type","seed","severity","rank1","rank5","eer","tar001"])
    writer.writeheader()
    writer.writerows(rows)

summary = {}
for r in rows:
    key = (r["eval_type"], r["severity"])
    summary.setdefault(key, {"rank1": [], "rank5": [], "eer": [], "tar001": []})
    for k in ["rank1","rank5","eer","tar001"]:
        v = r.get(k)
        if v is not None:
            summary[key][k].append(v)

summary_rows = []
for (eval_type, severity), vals in sorted(summary.items()):
    summary_rows.append({
        "eval_type": eval_type,
        "severity": severity,
        "rank1_mean": statistics.mean(vals["rank1"]) if vals["rank1"] else None,
        "rank1_std": statistics.stdev(vals["rank1"]) if len(vals["rank1"]) > 1 else 0.0,
        "rank5_mean": statistics.mean(vals["rank5"]) if vals["rank5"] else None,
        "rank5_std": statistics.stdev(vals["rank5"]) if len(vals["rank5"]) > 1 else 0.0,
        "eer_mean": statistics.mean(vals["eer"]) if vals["eer"] else None,
        "eer_std": statistics.stdev(vals["eer"]) if len(vals["eer"]) > 1 else 0.0,
        "tar001_mean": statistics.mean(vals["tar001"]) if vals["tar001"] else None,
        "tar001_std": statistics.stdev(vals["tar001"]) if len(vals["tar001"]) > 1 else 0.0,
    })

summary_csv = out_root / "final_eval_summary.csv"
with summary_csv.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=[
        "eval_type","severity",
        "rank1_mean","rank1_std",
        "rank5_mean","rank5_std",
        "eer_mean","eer_std",
        "tar001_mean","tar001_std",
    ])
    writer.writeheader()
    writer.writerows(summary_rows)

summary_txt = out_root / "final_eval_summary.txt"
with summary_txt.open("w") as f:
    for eval_type in ["CR", "Zcut", "fullgallery"]:
        f.write(f"=== {eval_type} ===\n")
        subset = [r for r in summary_rows if r["eval_type"] == eval_type]
        for r in sorted(subset, key=lambda x: ["easy","medium","hard"].index(x["severity"])):
            f.write(
                f'{r["severity"]:<6} | '
                f'rank1={r["rank1_mean"]:.4f}±{r["rank1_std"]:.4f} | '
                f'rank5={r["rank5_mean"]:.4f}±{r["rank5_std"]:.4f} | '
                f'eer={r["eer_mean"]:.4f}±{r["eer_std"]:.4f} | '
                f'tar001={r["tar001_mean"]:.4f}±{r["tar001_std"]:.4f}\n'
            )
        f.write("\n")

    hard = {r["eval_type"]: r for r in summary_rows if r["severity"] == "hard"}
    f.write("=== HARD-FOCUS ===\n")
    for eval_type in ["CR", "Zcut", "fullgallery"]:
        if eval_type in hard:
            r = hard[eval_type]
            f.write(
                f'{eval_type:<11} | rank1={r["rank1_mean"]:.4f}±{r["rank1_std"]:.4f} | '
                f'eer={r["eer_mean"]:.4f}±{r["eer_std"]:.4f} | '
                f'tar001={r["tar001_mean"]:.4f}±{r["tar001_std"]:.4f}\n'
            )

print(f"[OK] wrote {detail_csv}")
print(f"[OK] wrote {summary_csv}")
print(f"[OK] wrote {summary_txt}")
PY
}

print_config

if [[ "${MODE}" = "all" || "${MODE}" = "eval" ]]; then
  for seed in "${SEEDS[@]}"; do
    run_eval_standard "${seed}" "CR"
    run_eval_standard "${seed}" "Zcut"
    run_eval_fullgallery "${seed}"
  done
fi

if [[ "${MODE}" = "all" || "${MODE}" = "summarize" ]]; then
  summarize_results
fi
