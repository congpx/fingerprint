#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-all}"   # all | train | eval | summarize

PROJECT_ROOT="/home/congpx/fingerprint"
PYTHON_BIN="${PYTHON_BIN:-python}"

DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/data/SOCOFing/SOCOFing}"
INDEX_CSV="${INDEX_CSV:-${PROJECT_ROOT}/splits/index_v3.csv}"

# Winner from Phase 5 used as warm start
PHASE5_ROOT="${PHASE5_ROOT:-${PROJECT_ROOT}/outputs_phase5_final}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/outputs_phase6_triplet}"

SEEDS_STR="${SEEDS:-1 2 3}"
read -r -a SEEDS_ARR <<< "${SEEDS_STR}"

TRAIN_ALT_TYPES="${TRAIN_ALT_TYPES:-Obl}"
TRAIN_SEVERITIES="${TRAIN_SEVERITIES:-real,easy,medium,hard}"
VAL_ALT_TYPES="${VAL_ALT_TYPES:-CR}"
PROBES_STR="${PROBES:-CR}"
read -r -a PROBES_ARR <<< "${PROBES_STR}"

# Fine-tuning schedule
EPOCHS="${EPOCHS:-12}"
WORKERS="${WORKERS:-8}"
LR="${LR:-5e-5}"
EMB_DIM="${EMB_DIM:-256}"
EVAL_BATCH="${EVAL_BATCH:-256}"
USE_AMP="${USE_AMP:-1}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

# Winner MixStyle config from Phase 5
MIX_P="${MIX_P:-0.7}"
MIX_ALPHA="${MIX_ALPHA:-0.3}"
MIX_LAYER="${MIX_LAYER:-layer1}"

# Triplet fine-tune config
TRIPLET_WEIGHT="${TRIPLET_WEIGHT:-0.2}"
TRIPLET_MARGIN="${TRIPLET_MARGIN:-0.2}"
BATCH_IDENTITIES="${BATCH_IDENTITIES:-32}"
BATCH_INSTANCES="${BATCH_INSTANCES:-4}"

RUN_TAG="w$(printf '%02d' "$(python - <<PY
print(int(float('${TRIPLET_WEIGHT}')*100))
PY
)")_m$(printf '%02d' "$(python - <<PY
print(int(float('${TRIPLET_MARGIN}')*100))
PY
)")"

mkdir -p "${OUTPUT_ROOT}"

print_config() {
  echo "[CONFIG]"
  echo "  MODE              = ${MODE}"
  echo "  PROJECT_ROOT      = ${PROJECT_ROOT}"
  echo "  DATA_ROOT         = ${DATA_ROOT}"
  echo "  INDEX_CSV         = ${INDEX_CSV}"
  echo "  PHASE5_ROOT       = ${PHASE5_ROOT}"
  echo "  OUTPUT_ROOT       = ${OUTPUT_ROOT}"
  echo "  SEEDS             = ${SEEDS_STR}"
  echo "  TRAIN_ALT_TYPES   = ${TRAIN_ALT_TYPES}"
  echo "  TRAIN_SEVERITIES  = ${TRAIN_SEVERITIES}"
  echo "  VAL_ALT_TYPES     = ${VAL_ALT_TYPES}"
  echo "  PROBES            = ${PROBES_STR}"
  echo "  EPOCHS            = ${EPOCHS}"
  echo "  LR                = ${LR}"
  echo "  EVAL_BATCH        = ${EVAL_BATCH}"
  echo "  MIXSTYLE          = layer=${MIX_LAYER}, p=${MIX_P}, alpha=${MIX_ALPHA}"
  echo "  TRIPLET           = weight=${TRIPLET_WEIGHT}, margin=${TRIPLET_MARGIN}"
  echo "  PK SAMPLER        = P=${BATCH_IDENTITIES}, K=${BATCH_INSTANCES}"
  echo "  RUN_TAG           = ${RUN_TAG}"
}

run_train_one() {
  local seed="$1"
  local run_name="mix_triplet_${RUN_TAG}_s${seed}"
  local out_dir="${OUTPUT_ROOT}/${run_name}"
  local ckpt="${out_dir}/checkpoints/best.pt"
  local init_ckpt="${PHASE5_ROOT}/mix_best_s${seed}/checkpoints/best.pt"

  if [[ ! -f "${init_ckpt}" ]]; then
    echo "[ERROR] missing init checkpoint: ${init_ckpt}"
    exit 1
  fi

  if [[ "${SKIP_EXISTING}" = "1" && -f "${ckpt}" ]]; then
    echo "[SKIP] train ${run_name} (best.pt exists)"
    return
  fi

  mkdir -p "${out_dir}"
  echo "[RUN] train ${run_name} from ${init_ckpt}"
  ${PYTHON_BIN} "${PROJECT_ROOT}/run_socofing_v3_mixstyle_triplet.py" train \
    --data_root "${DATA_ROOT}" \
    --index "${INDEX_CSV}" \
    --outdir "${out_dir}" \
    --epochs "${EPOCHS}" \
    --workers "${WORKERS}" \
    --lr "${LR}" \
    --emb_dim "${EMB_DIM}" \
    --train_severities "${TRAIN_SEVERITIES}" \
    --train_alt_types "${TRAIN_ALT_TYPES}" \
    --val_alt_types "${VAL_ALT_TYPES}" \
    $( [[ "${USE_AMP}" = "1" ]] && echo "--amp" ) \
    --mix_p "${MIX_P}" \
    --mix_alpha "${MIX_ALPHA}" \
    --mix_layer "${MIX_LAYER}" \
    --init_ckpt "${init_ckpt}" \
    --triplet_weight "${TRIPLET_WEIGHT}" \
    --triplet_margin "${TRIPLET_MARGIN}" \
    --batch_identities "${BATCH_IDENTITIES}" \
    --batch_instances "${BATCH_INSTANCES}" \
    --eval_batch "${EVAL_BATCH}" \
    --seed "${seed}"
}

run_eval_one() {
  local seed="$1"
  local run_name="mix_triplet_${RUN_TAG}_s${seed}"
  local run_dir="${OUTPUT_ROOT}/${run_name}"
  local ckpt="${run_dir}/checkpoints/best.pt"

  if [[ ! -f "${ckpt}" ]]; then
    echo "[WARN] missing ckpt: ${ckpt}"
    return
  fi

  for probe in "${PROBES_ARR[@]}"; do
    local out_json="${run_dir}/test_unseen${probe}.json"
    if [[ "${SKIP_EXISTING}" = "1" && -f "${out_json}" ]]; then
      echo "[SKIP] eval ${run_name} on ${probe} (${out_json} exists)"
      continue
    fi
    echo "[RUN] eval ${run_name} on unseen ${probe}"
    ${PYTHON_BIN} "${PROJECT_ROOT}/run_socofing_v3_mixstyle_triplet.py" eval \
      --data_root "${DATA_ROOT}" \
      --index "${INDEX_CSV}" \
      --ckpt "${ckpt}" \
      --split test \
      --batch "${EVAL_BATCH}" \
      --workers "${WORKERS}" \
      --probe_alt_types "${probe}" \
      --out_json "${out_json}"
  done
}

summarize_results() {
  ${PYTHON_BIN} - <<'PY'
import csv, json, statistics
from pathlib import Path

root = Path('/home/congpx/fingerprint/outputs_phase6_triplet')
phase5 = Path('/home/congpx/fingerprint/outputs_phase5_final')
rows = []

# New triplet runs
for run_dir in sorted(root.glob('mix_triplet_*_s*')):
    name = run_dir.name
    parts = name.split('_')
    try:
        seed = int(parts[-1].replace('s', ''))
    except Exception:
        continue
    method = '_'.join(parts[:-1])
    for jf in sorted(run_dir.glob('test_unseen*.json')):
        probe = jf.stem.replace('test_unseen', '')
        data = json.loads(jf.read_text())
        for severity, m in data.items():
            if not isinstance(m, dict):
                continue
            rows.append({
                'run': name,
                'method': method,
                'seed': seed,
                'probe': probe,
                'severity': severity,
                'rank1': m.get('rank1'),
                'rank5': m.get('rank5'),
                'eer': m.get('eer'),
                'tar001': m.get('tar@far=0.001'),
            })

# Reference winner from Phase 5, same seeds only if available
for run_dir in sorted(phase5.glob('mix_best_s*')):
    name = run_dir.name
    try:
        seed = int(name.split('_')[-1].replace('s', ''))
    except Exception:
        continue
    for jf in sorted(run_dir.glob('test_unseen*.json')):
        probe = jf.stem.replace('test_unseen', '')
        data = json.loads(jf.read_text())
        for severity, m in data.items():
            if not isinstance(m, dict):
                continue
            rows.append({
                'run': name,
                'method': 'mix_best_ref',
                'seed': seed,
                'probe': probe,
                'severity': severity,
                'rank1': m.get('rank1'),
                'rank5': m.get('rank5'),
                'eer': m.get('eer'),
                'tar001': m.get('tar@far=0.001'),
            })

if not rows:
    print('[WARN] No JSON evaluation files found. Nothing to summarize.')
    raise SystemExit(0)

# Restrict to seeds available in triplet runs for fair comparison
triplet_seeds = sorted({r['seed'] for r in rows if r['method'].startswith('mix_triplet_')})
rows = [r for r in rows if not (r['method'] == 'mix_best_ref' and r['seed'] not in triplet_seeds)]

# Save detail
root.mkdir(parents=True, exist_ok=True)
detail_csv = root / 'phase6_triplet_detail.csv'
with detail_csv.open('w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['run','method','seed','probe','severity','rank1','rank5','eer','tar001'])
    writer.writeheader()
    writer.writerows(rows)

summary = {}
for r in rows:
    key = (r['method'], r['probe'], r['severity'])
    summary.setdefault(key, {'rank1': [], 'eer': [], 'tar001': []})
    for metric in ['rank1','eer','tar001']:
        v = r.get(metric)
        if v is not None:
            summary[key][metric].append(v)

summary_rows = []
for (method, probe, severity), vals in sorted(summary.items()):
    summary_rows.append({
        'method': method,
        'probe': probe,
        'severity': severity,
        'rank1_mean': statistics.mean(vals['rank1']) if vals['rank1'] else None,
        'rank1_std': statistics.stdev(vals['rank1']) if len(vals['rank1']) > 1 else 0.0,
        'eer_mean': statistics.mean(vals['eer']) if vals['eer'] else None,
        'eer_std': statistics.stdev(vals['eer']) if len(vals['eer']) > 1 else 0.0,
        'tar001_mean': statistics.mean(vals['tar001']) if vals['tar001'] else None,
        'tar001_std': statistics.stdev(vals['tar001']) if len(vals['tar001']) > 1 else 0.0,
    })

summary_csv = root / 'phase6_triplet_summary.csv'
with summary_csv.open('w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['method','probe','severity','rank1_mean','rank1_std','eer_mean','eer_std','tar001_mean','tar001_std'])
    writer.writeheader()
    writer.writerows(summary_rows)

# Write text report focused on hard / CR
lines = []
lines.append('=== Phase 6: CR hard comparison ===')
for r in summary_rows:
    if r['probe'] != 'CR' or r['severity'] != 'hard':
        continue
    lines.append(
        f"{r['method']:<28} | rank1={r['rank1_mean']:.4f}±{r['rank1_std']:.4f} | "
        f"eer={r['eer_mean']:.4f}±{r['eer_std']:.4f} | "
        f"tar001={r['tar001_mean']:.4f}±{r['tar001_std']:.4f}"
    )

trip = next((r for r in summary_rows if r['method'].startswith('mix_triplet_') and r['probe']=='CR' and r['severity']=='hard'), None)
ref = next((r for r in summary_rows if r['method']=='mix_best_ref' and r['probe']=='CR' and r['severity']=='hard'), None)
if trip and ref:
    lines.append('')
    lines.append('=== Delta triplet - reference (CR hard) ===')
    lines.append(f"delta_rank1 = {trip['rank1_mean'] - ref['rank1_mean']:+.4f}")
    lines.append(f"delta_eer   = {trip['eer_mean'] - ref['eer_mean']:+.4f}")
    lines.append(f"delta_tar001= {trip['tar001_mean'] - ref['tar001_mean']:+.4f}")

text_path = root / 'phase6_triplet_summary.txt'
text_path.write_text('\n'.join(lines))
print(f'[OK] wrote {detail_csv}')
print(f'[OK] wrote {summary_csv}')
print(f'[OK] wrote {text_path}')
PY
}

print_config

if [[ "${MODE}" = "all" || "${MODE}" = "train" ]]; then
  for seed in "${SEEDS_ARR[@]}"; do
    run_train_one "${seed}"
  done
fi

if [[ "${MODE}" = "all" || "${MODE}" = "eval" ]]; then
  for seed in "${SEEDS_ARR[@]}"; do
    run_eval_one "${seed}"
  done
fi

if [[ "${MODE}" = "all" || "${MODE}" = "summarize" ]]; then
  summarize_results
fi
