#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-all}"   # all | eval | summarize

PROJECT_ROOT="/home/congpx/fingerprint"
PYTHON_BIN="python"

DATA_ROOT="${PROJECT_ROOT}/data/SOCOFing/SOCOFing"
INDEX_CSV="${PROJECT_ROOT}/splits/index_v3.csv"

PHASE6_ROOT="${PROJECT_ROOT}/outputs_phase6_triplet"
OUTPUT_ROOT="${PROJECT_ROOT}/outputs_phase6_final_eval"

SEEDS=(1 2 3)
RUN_TAG="${RUN_TAG:-w20_m20}"
WINNER_PREFIX="mix_triplet_${RUN_TAG}_s"

BATCH=256
WORKERS=8
SKIP_EXISTING=1

mkdir -p "${OUTPUT_ROOT}"

echo "[CONFIG]"
echo "  MODE          = ${MODE}"
echo "  PROJECT_ROOT  = ${PROJECT_ROOT}"
echo "  DATA_ROOT     = ${DATA_ROOT}"
echo "  INDEX_CSV     = ${INDEX_CSV}"
echo "  PHASE6_ROOT   = ${PHASE6_ROOT}"
echo "  OUTPUT_ROOT   = ${OUTPUT_ROOT}"
echo "  SEEDS         = ${SEEDS[*]}"
echo "  RUN_TAG       = ${RUN_TAG}"
echo "  BATCH         = ${BATCH}"
echo "  WORKERS       = ${WORKERS}"

run_eval_standard() {
  local seed="$1"
  local run_name="${WINNER_PREFIX}${seed}"
  local ckpt="${PHASE6_ROOT}/${run_name}/checkpoints/best.pt"

  if [[ ! -f "${ckpt}" ]]; then
    echo "[WARN] missing checkpoint: ${ckpt}"
    return
  fi

  for probe in CR Zcut; do
    local out_json="${OUTPUT_ROOT}/${run_name}_test_unseen${probe}.json"

    if [[ "${SKIP_EXISTING}" = "1" && -f "${out_json}" ]]; then
      echo "[SKIP] standard eval ${run_name} unseen ${probe}"
      continue
    fi

    echo "[RUN] standard eval ${run_name} unseen ${probe}"
    ${PYTHON_BIN} "${PROJECT_ROOT}/run_socofing_v3_mixstyle_triplet.py" eval \
      --data_root "${DATA_ROOT}" \
      --index "${INDEX_CSV}" \
      --ckpt "${ckpt}" \
      --split test \
      --batch "${BATCH}" \
      --workers "${WORKERS}" \
      --probe_alt_types "${probe}" \
      --out_json "${out_json}"
  done
}

run_eval_fullgallery() {
  local seed="$1"
  local run_name="${WINNER_PREFIX}${seed}"
  local ckpt="${PHASE6_ROOT}/${run_name}/checkpoints/best.pt"

  if [[ ! -f "${ckpt}" ]]; then
    echo "[WARN] missing checkpoint: ${ckpt}"
    return
  fi

  local out_all="${OUTPUT_ROOT}/${run_name}_fullgallery_ALL.json"
  if [[ "${SKIP_EXISTING}" != "1" || ! -f "${out_all}" ]]; then
    echo "[RUN] full-gallery ${run_name} ALL"
    ${PYTHON_BIN} "${PROJECT_ROOT}/eval_fullgallery_compat.py" \
      --data_root "${DATA_ROOT}" \
      --index "${INDEX_CSV}" \
      --ckpt "${ckpt}" \
      --probe_split test \
      --batch "${BATCH}" \
      --workers "${WORKERS}" \
      --out_json "${out_all}"
  else
    echo "[SKIP] full-gallery ${run_name} ALL"
  fi

  for probe in CR Zcut; do
    local out_json="${OUTPUT_ROOT}/${run_name}_fullgallery_${probe}.json"

    if [[ "${SKIP_EXISTING}" = "1" && -f "${out_json}" ]]; then
      echo "[SKIP] full-gallery ${run_name} ${probe}"
      continue
    fi

    echo "[RUN] full-gallery ${run_name} ${probe}"
    ${PYTHON_BIN} "${PROJECT_ROOT}/eval_fullgallery_compat.py" \
      --data_root "${DATA_ROOT}" \
      --index "${INDEX_CSV}" \
      --ckpt "${ckpt}" \
      --probe_split test \
      --probe_alt_types "${probe}" \
      --batch "${BATCH}" \
      --workers "${WORKERS}" \
      --out_json "${out_json}"
  done
}

summarize_results() {
  ${PYTHON_BIN} - <<'PY'
import csv
import json
import statistics
from pathlib import Path

root = Path("/home/congpx/fingerprint/outputs_phase6_final_eval")

rows = []
for jf in sorted(root.glob("*.json")):
    stem = jf.stem
    parts = stem.split("_")
    if len(parts) < 7:
        continue

    seed_token = None
    for p in parts:
        if p.startswith("s") and p[1:].isdigit():
            seed_token = p
            break
    if seed_token is None:
        continue
    seed = int(seed_token[1:])
    run_name = stem[:stem.index(f"_{seed_token}") + len(seed_token)]

    if "_test_unseen" in stem:
        eval_type = "standard"
        probe = stem.split("_test_unseen", 1)[1]
    elif "_fullgallery_" in stem:
        eval_type = "fullgallery"
        probe = stem.split("_fullgallery_", 1)[1]
    else:
        continue

    data = json.loads(jf.read_text())
    for severity, m in data.items():
        if not isinstance(m, dict):
            continue
        rows.append({
            "run": run_name,
            "seed": seed,
            "eval_type": eval_type,
            "probe": probe,
            "severity": severity,
            "rank1": m.get("rank1"),
            "rank5": m.get("rank5"),
            "eer": m.get("eer"),
            "tar001": m.get("tar@far=0.001"),
            "tar01": m.get("tar@far=0.01"),
            "n_probe": m.get("n_probe"),
            "n_gallery": m.get("n_gallery"),
            "source_json": jf.name,
        })

detail_csv = root / "phase6_final_eval_detail.csv"
with detail_csv.open("w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "run","seed","eval_type","probe","severity",
            "rank1","rank5","eer","tar001","tar01",
            "n_probe","n_gallery","source_json"
        ]
    )
    writer.writeheader()
    writer.writerows(rows)

summary = {}
for r in rows:
    key = (r["eval_type"], r["probe"], r["severity"])
    summary.setdefault(key, {"rank1": [], "eer": [], "tar001": []})
    for k in ["rank1", "eer", "tar001"]:
        v = r.get(k)
        if v is not None:
            summary[key][k].append(v)

summary_rows = []
for (eval_type, probe, severity), vals in sorted(summary.items()):
    summary_rows.append({
        "eval_type": eval_type,
        "probe": probe,
        "severity": severity,
        "rank1_mean": statistics.mean(vals["rank1"]) if vals["rank1"] else None,
        "rank1_std": statistics.stdev(vals["rank1"]) if len(vals["rank1"]) > 1 else 0.0,
        "eer_mean": statistics.mean(vals["eer"]) if vals["eer"] else None,
        "eer_std": statistics.stdev(vals["eer"]) if len(vals["eer"]) > 1 else 0.0,
        "tar001_mean": statistics.mean(vals["tar001"]) if vals["tar001"] else None,
        "tar001_std": statistics.stdev(vals["tar001"]) if len(vals["tar001"]) > 1 else 0.0,
    })

summary_csv = root / "phase6_final_eval_summary.csv"
with summary_csv.open("w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "eval_type","probe","severity",
            "rank1_mean","rank1_std",
            "eer_mean","eer_std",
            "tar001_mean","tar001_std",
        ]
    )
    writer.writeheader()
    writer.writerows(summary_rows)

txt = root / "phase6_final_eval_summary.txt"
with txt.open("w") as f:
    f.write("=== STANDARD EVAL ===\n")
    for probe in ["CR", "Zcut"]:
        f.write(f"\n[{probe}]\n")
        for sev in ["easy", "medium", "hard"]:
            row = next((r for r in summary_rows if r["eval_type"] == "standard" and r["probe"] == probe and r["severity"] == sev), None)
            if row:
                f.write(
                    f"  {sev:<6} rank1={row['rank1_mean']:.4f}±{row['rank1_std']:.4f} | "
                    f"eer={row['eer_mean']:.4f}±{row['eer_std']:.4f} | "
                    f"tar001={row['tar001_mean']:.4f}±{row['tar001_std']:.4f}\n"
                )

    f.write("\n=== FULL-GALLERY EVAL ===\n")
    for probe in ["ALL", "CR", "Zcut"]:
        f.write(f"\n[{probe}]\n")
        for sev in ["easy", "medium", "hard"]:
            row = next((r for r in summary_rows if r["eval_type"] == "fullgallery" and r["probe"] == probe and r["severity"] == sev), None)
            if row:
                f.write(
                    f"  {sev:<6} rank1={row['rank1_mean']:.4f}±{row['rank1_std']:.4f} | "
                    f"eer={row['eer_mean']:.4f}±{row['eer_std']:.4f} | "
                    f"tar001={row['tar001_mean']:.4f}±{row['tar001_std']:.4f}\n"
                )

print(f"[OK] wrote {detail_csv}")
print(f"[OK] wrote {summary_csv}")
print(f"[OK] wrote {txt}")
PY
}

if [[ "${MODE}" = "all" || "${MODE}" = "eval" ]]; then
  for seed in "${SEEDS[@]}"; do
    run_eval_standard "${seed}"
    run_eval_fullgallery "${seed}"
  done
fi

if [[ "${MODE}" = "all" || "${MODE}" = "summarize" ]]; then
  summarize_results
fi
