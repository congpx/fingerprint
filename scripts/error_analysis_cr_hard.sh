#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/congpx/fingerprint"
DATA_ROOT="${PROJECT_ROOT}/data/SOCOFing/SOCOFing"
INDEX_CSV="${PROJECT_ROOT}/splits/index_v3.csv"
RUN_ROOT="${PROJECT_ROOT}/outputs_phase5_final"
OUTPUT_ROOT="${PROJECT_ROOT}/outputs_error_analysis_cr_hard"
SEEDS="1 2 3 4 5"
RUN_PREFIX="mix_best_s"
CKPT_NAME="best.pt"
PROBE_SPLIT="test"
PROBE_ALT_TYPE="CR"
PROBE_SEVERITY="hard"
GALLERY_MODE="same_split_real"   # same_split_real | all_real
BATCH=256
WORKERS=8
TOPK_FAILS=50
TOPK_MONTAGE=20
IMG_SIZE=224

mkdir -p "${OUTPUT_ROOT}"

echo "[CONFIG]"
echo "  PROJECT_ROOT    = ${PROJECT_ROOT}"
echo "  DATA_ROOT       = ${DATA_ROOT}"
echo "  INDEX_CSV       = ${INDEX_CSV}"
echo "  RUN_ROOT        = ${RUN_ROOT}"
echo "  OUTPUT_ROOT     = ${OUTPUT_ROOT}"
echo "  SEEDS           = ${SEEDS}"
echo "  RUN_PREFIX      = ${RUN_PREFIX}"
echo "  CKPT_NAME       = ${CKPT_NAME}"
echo "  PROBE_SPLIT     = ${PROBE_SPLIT}"
echo "  PROBE_ALT_TYPE  = ${PROBE_ALT_TYPE}"
echo "  PROBE_SEVERITY  = ${PROBE_SEVERITY}"
echo "  GALLERY_MODE    = ${GALLERY_MODE}"
echo "  BATCH           = ${BATCH}"
echo "  WORKERS         = ${WORKERS}"
echo "  TOPK_FAILS      = ${TOPK_FAILS}"
echo "  TOPK_MONTAGE    = ${TOPK_MONTAGE}"

python - <<'PY'
import json
import math
import os
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.models as models

PROJECT_ROOT = Path("/home/congpx/fingerprint")
DATA_ROOT = PROJECT_ROOT / "data/SOCOFing/SOCOFing"
INDEX_CSV = PROJECT_ROOT / "splits/index_v3.csv"
RUN_ROOT = PROJECT_ROOT / "outputs_phase5_final"
OUTPUT_ROOT = PROJECT_ROOT / "outputs_error_analysis_cr_hard"
SEEDS = [1, 2, 3, 4, 5]
RUN_PREFIX = "mix_best_s"
CKPT_NAME = "best.pt"
PROBE_SPLIT = "test"
PROBE_ALT_TYPE = "CR"
PROBE_SEVERITY = "hard"
GALLERY_MODE = "same_split_real"
BATCH = 256
WORKERS = 8
TOPK_FAILS = 50
TOPK_MONTAGE = 20
IMG_SIZE = 224


class SOCOFingDS(Dataset):
    def __init__(self, df: pd.DataFrame, root: Path, tf):
        self.df = df.reset_index(drop=True)
        self.root = root
        self.tf = tf

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        r = self.df.iloc[idx]
        img = Image.open(self.root / r["relpath"]).convert("L").convert("RGB")
        x = self.tf(img)
        return x, r["finger_id"], r["relpath"]


def collate(batch):
    x = torch.stack([b[0] for b in batch], 0)
    ids = [b[1] for b in batch]
    rels = [b[2] for b in batch]
    return x, ids, rels


class MixStyle(nn.Module):
    def __init__(self, p=0.5, alpha=0.3, eps=1e-6):
        super().__init__()
        self.p = p
        self.alpha = alpha
        self.eps = eps
        self.beta = torch.distributions.Beta(alpha, alpha)

    def forward(self, x):
        if not self.training or self.p <= 0.0:
            return x
        if torch.rand(1).item() > self.p:
            return x
        B = x.size(0)
        mu = x.mean(dim=[2, 3], keepdim=True)
        var = x.var(dim=[2, 3], keepdim=True, unbiased=False)
        sig = (var + self.eps).sqrt()
        x_norm = (x - mu) / sig
        perm = torch.randperm(B, device=x.device)
        mu2, sig2 = mu[perm], sig[perm]
        lam = self.beta.sample((B, 1, 1, 1)).to(x.device)
        mu_mix = lam * mu + (1 - lam) * mu2
        sig_mix = lam * sig + (1 - lam) * sig2
        return x_norm * sig_mix + mu_mix


class MixFingerprintNet(nn.Module):
    def __init__(self, num_classes: int, emb_dim: int = 256, mix_p: float = 0.0, mix_alpha: float = 0.3, mix_layer: str = "layer1"):
        super().__init__()
        b = models.resnet18(weights=None)
        feat_dim = b.fc.in_features
        b.fc = nn.Identity()
        self.mix = MixStyle(p=mix_p, alpha=mix_alpha)
        self.mix_layer = mix_layer
        self.b = b
        self.emb = nn.Linear(feat_dim, emb_dim)
        self.bn = nn.BatchNorm1d(emb_dim)
        self.cls = nn.Linear(emb_dim, num_classes)

    def forward(self, x):
        x = self.b.conv1(x)
        x = self.b.bn1(x)
        x = self.b.relu(x)
        x = self.b.maxpool(x)
        x = self.b.layer1(x)
        if self.mix_layer == "layer1":
            x = self.mix(x)
        x = self.b.layer2(x)
        if self.mix_layer == "layer2":
            x = self.mix(x)
        x = self.b.layer3(x)
        if self.mix_layer == "layer3":
            x = self.mix(x)
        x = self.b.layer4(x)
        x = self.b.avgpool(x)
        x = torch.flatten(x, 1)
        emb = self.bn(self.emb(x))
        emb = F.normalize(emb, dim=1)
        logits = self.cls(emb)
        return emb, logits


def get_arg(args_obj, key, default=None):
    if args_obj is None:
        return default
    if isinstance(args_obj, dict):
        return args_obj.get(key, default)
    return getattr(args_obj, key, default)


def build_model(ckpt, device):
    state = ckpt["model"]
    label_map = ckpt["label_map"]
    args = ckpt.get("args", None)
    emb_dim = get_arg(args, "emb_dim", 256)
    mix_alpha = get_arg(args, "mix_alpha", 0.3)
    mix_layer = get_arg(args, "mix_layer", "layer1")
    model = MixFingerprintNet(
        num_classes=len(label_map),
        emb_dim=emb_dim,
        mix_p=0.0,
        mix_alpha=mix_alpha,
        mix_layer=mix_layer,
    ).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


@torch.no_grad()
def extract_embeddings(model, loader, device):
    all_embs, all_ids, all_rels = [], [], []
    for x, ids, rels in tqdm(loader, desc="extract", leave=False):
        x = x.to(device, non_blocking=True)
        emb, _ = model(x)
        all_embs.append(emb.cpu())
        all_ids.extend(ids)
        all_rels.extend(rels)
    embs = torch.cat(all_embs, 0).numpy()
    return embs, all_ids, all_rels


def make_loader(df, tf):
    ds = SOCOFingDS(df, DATA_ROOT, tf)
    return DataLoader(ds, batch_size=BATCH, shuffle=False, num_workers=WORKERS, pin_memory=True, collate_fn=collate)


def safe_open_image(path: Path, size=(224, 224)):
    img = Image.open(path).convert("RGB")
    return img.resize(size)


def draw_caption(draw, xy, text, fill=(255, 255, 255)):
    draw.text(xy, text, fill=fill)


def build_contact_sheet(rows_df: pd.DataFrame, out_path: Path, title: str, n_items: int = 20):
    rows_df = rows_df.head(n_items).copy()
    if len(rows_df) == 0:
        return

    cell_w, cell_h = 224, 224
    header_h = 36
    footer_h = 54
    gap = 10
    cols = 3
    total_w = cols * cell_w + (cols + 1) * gap
    total_h = gap + 30 + len(rows_df) * (header_h + cell_h + footer_h + gap)
    canvas = Image.new("RGB", (total_w, total_h), color=(18, 18, 18))
    draw = ImageDraw.Draw(canvas)
    draw.text((gap, gap), title, fill=(255, 255, 0))
    y = gap + 30

    for _, r in rows_df.iterrows():
        probe_img = safe_open_image(DATA_ROOT / r["probe_relpath"], (cell_w, cell_h))
        true_img = safe_open_image(DATA_ROOT / r["genuine_relpath"], (cell_w, cell_h))
        pred_img = safe_open_image(DATA_ROOT / r["top1_relpath"], (cell_w, cell_h))

        draw.text((gap, y), f"seed={int(r['seed'])} rank_true={int(r['rank_true'])} gap={r['score_gap']:.4f}", fill=(255,255,255))
        y += header_h

        x_positions = [gap, gap*2 + cell_w, gap*3 + cell_w*2]
        canvas.paste(probe_img, (x_positions[0], y))
        canvas.paste(true_img, (x_positions[1], y))
        canvas.paste(pred_img, (x_positions[2], y))
        draw.text((x_positions[0], y + cell_h + 4), "probe (CR hard)", fill=(255,255,255))
        draw.text((x_positions[1], y + cell_h + 4), f"true real\n{r['finger_id']}", fill=(0,255,0))
        draw.text((x_positions[2], y + cell_h + 4), f"top1 wrong\n{r['top1_finger_id']}", fill=(255,128,128))
        y += cell_h + footer_h + gap

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def main():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    tf = T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    df = pd.read_csv(INDEX_CSV)
    df["finger_id"] = df["finger_id"].astype(str)
    df["alt_type"] = df["alt_type"].fillna("")

    if GALLERY_MODE == "same_split_real":
        gal_df = df[(df["split"] == PROBE_SPLIT) & (df["severity"] == "real")].copy()
    elif GALLERY_MODE == "all_real":
        gal_df = df[df["severity"] == "real"].copy()
    else:
        raise ValueError(f"Unsupported GALLERY_MODE={GALLERY_MODE}")

    probe_df = df[
        (df["split"] == PROBE_SPLIT)
        & (df["severity"] == PROBE_SEVERITY)
        & (df["alt_type"] == PROBE_ALT_TYPE)
    ].copy()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    gal_loader = make_loader(gal_df, tf)
    all_fail_rows = []
    summary_rows = []

    for seed in SEEDS:
        run_name = f"{RUN_PREFIX}{seed}"
        ckpt_path = RUN_ROOT / run_name / "checkpoints" / CKPT_NAME
        if not ckpt_path.exists():
            print(f"[WARN] missing checkpoint: {ckpt_path}")
            continue

        print(f"[RUN] error analysis for {run_name}")
        ckpt = torch.load(ckpt_path, map_location="cpu")
        model = build_model(ckpt, device)

        gal_embs, gal_ids, gal_rels = extract_embeddings(model, gal_loader, device)
        gal_embs = gal_embs / np.linalg.norm(gal_embs, axis=1, keepdims=True).clip(1e-12)

        pr_loader = make_loader(probe_df, tf)
        pr_embs, pr_ids, pr_rels = extract_embeddings(model, pr_loader, device)
        pr_embs = pr_embs / np.linalg.norm(pr_embs, axis=1, keepdims=True).clip(1e-12)

        sims = pr_embs @ gal_embs.T
        order = np.argsort(-sims, axis=1)

        rows = []
        n_correct = 0
        for i, (pid, prel) in enumerate(zip(pr_ids, pr_rels)):
            sim_row = sims[i]
            ord_row = order[i]
            top1_idx = int(ord_row[0])
            top1_fid = gal_ids[top1_idx]
            top1_rel = gal_rels[top1_idx]
            top1_score = float(sim_row[top1_idx])

            genuine_idxs = np.where(np.array(gal_ids) == pid)[0]
            if len(genuine_idxs) == 0:
                continue
            genuine_scores = sim_row[genuine_idxs]
            best_genuine_local = int(np.argmax(genuine_scores))
            genuine_idx = int(genuine_idxs[best_genuine_local])
            genuine_score = float(sim_row[genuine_idx])
            genuine_rel = gal_rels[genuine_idx]

            rank_true = int(np.where(ord_row == genuine_idx)[0][0]) + 1
            correct = int(top1_fid == pid)
            n_correct += correct

            rows.append({
                "seed": seed,
                "run": run_name,
                "probe_split": PROBE_SPLIT,
                "probe_alt_type": PROBE_ALT_TYPE,
                "probe_severity": PROBE_SEVERITY,
                "gallery_mode": GALLERY_MODE,
                "probe_relpath": prel,
                "finger_id": pid,
                "top1_finger_id": top1_fid,
                "top1_relpath": top1_rel,
                "genuine_relpath": genuine_rel,
                "top1_score": top1_score,
                "genuine_score": genuine_score,
                "score_gap": top1_score - genuine_score,
                "rank_true": rank_true,
                "correct_top1": correct,
            })

        run_df = pd.DataFrame(rows)
        run_out_dir = OUTPUT_ROOT / run_name
        run_out_dir.mkdir(parents=True, exist_ok=True)

        detail_csv = run_out_dir / "cr_hard_all_probes.csv"
        run_df.to_csv(detail_csv, index=False)

        fail_df = run_df[run_df["correct_top1"] == 0].copy().sort_values(["score_gap", "rank_true", "top1_score"], ascending=[False, False, False])
        fail_df.head(TOPK_FAILS).to_csv(run_out_dir / "cr_hard_top_failures.csv", index=False)
        build_contact_sheet(
            fail_df,
            run_out_dir / "cr_hard_top_failures_montage.jpg",
            title=f"{run_name} | CR hard | top failure cases",
            n_items=TOPK_MONTAGE,
        )

        rank1 = float(n_correct / max(1, len(run_df)))
        summary_rows.append({
            "seed": seed,
            "run": run_name,
            "n_probe": int(len(run_df)),
            "n_fail": int((run_df["correct_top1"] == 0).sum()),
            "rank1": rank1,
            "mean_rank_true_fail": float(fail_df["rank_true"].mean()) if len(fail_df) else 1.0,
            "mean_gap_fail": float(fail_df["score_gap"].mean()) if len(fail_df) else 0.0,
            "max_gap_fail": float(fail_df["score_gap"].max()) if len(fail_df) else 0.0,
        })
        all_fail_rows.append(fail_df)
        print(f"[OK] {run_name}: n_probe={len(run_df)} n_fail={len(fail_df)} rank1={rank1:.4f}")

    if not summary_rows:
        raise SystemExit("No successful runs found.")

    pd.DataFrame(summary_rows).to_csv(OUTPUT_ROOT / "per_seed_summary.csv", index=False)

    all_fail_df = pd.concat(all_fail_rows, ignore_index=True) if all_fail_rows else pd.DataFrame()
    if len(all_fail_df):
        all_fail_df.to_csv(OUTPUT_ROOT / "all_failures_across_seeds.csv", index=False)

        agg = (
            all_fail_df.groupby(["probe_relpath", "finger_id"], as_index=False)
            .agg(
                fail_count=("seed", "count"),
                seeds=("seed", lambda x: ",".join(map(str, sorted(set(x))))),
                mean_rank_true=("rank_true", "mean"),
                max_rank_true=("rank_true", "max"),
                mean_gap=("score_gap", "mean"),
                max_gap=("score_gap", "max"),
                example_top1_relpath=("top1_relpath", "first"),
                example_genuine_relpath=("genuine_relpath", "first"),
                example_top1_finger_id=("top1_finger_id", "first"),
            )
            .sort_values(["fail_count", "mean_gap", "max_rank_true"], ascending=[False, False, False])
        )
        agg.to_csv(OUTPUT_ROOT / "consensus_hard_failures.csv", index=False)

        montage_df = agg.rename(columns={
            "example_top1_relpath": "top1_relpath",
            "example_genuine_relpath": "genuine_relpath",
            "example_top1_finger_id": "top1_finger_id",
        }).copy()
        montage_df["seed"] = 0
        montage_df["rank_true"] = montage_df["mean_rank_true"]
        montage_df["score_gap"] = montage_df["mean_gap"]
        build_contact_sheet(
            montage_df,
            OUTPUT_ROOT / "consensus_hard_failures_montage.jpg",
            title="Consensus CR hard failures across seeds",
            n_items=TOPK_MONTAGE,
        )

    summary_txt = OUTPUT_ROOT / "summary.txt"
    with open(summary_txt, "w") as f:
        f.write("=== CR HARD ERROR ANALYSIS (mix_best) ===\n")
        for row in summary_rows:
            f.write(
                f"seed={row['seed']} | n_probe={row['n_probe']} | n_fail={row['n_fail']} | "
                f"rank1={row['rank1']:.4f} | mean_rank_true_fail={row['mean_rank_true_fail']:.2f} | "
                f"mean_gap_fail={row['mean_gap_fail']:.4f} | max_gap_fail={row['max_gap_fail']:.4f}\n"
            )
        if len(all_fail_df):
            f.write("\n=== CONSENSUS TOP FAILURES ===\n")
            top = pd.read_csv(OUTPUT_ROOT / "consensus_hard_failures.csv").head(20)
            for _, r in top.iterrows():
                f.write(
                    f"fail_count={int(r['fail_count'])} | mean_gap={r['mean_gap']:.4f} | max_rank_true={int(r['max_rank_true'])} | "
                    f"probe={r['probe_relpath']} | true={r['finger_id']} | pred={r['example_top1_finger_id']} | seeds={r['seeds']}\n"
                )

    print(f"[OK] wrote {OUTPUT_ROOT / 'per_seed_summary.csv'}")
    if len(all_fail_df):
        print(f"[OK] wrote {OUTPUT_ROOT / 'all_failures_across_seeds.csv'}")
        print(f"[OK] wrote {OUTPUT_ROOT / 'consensus_hard_failures.csv'}")
        print(f"[OK] wrote {OUTPUT_ROOT / 'consensus_hard_failures_montage.jpg'}")
    print(f"[OK] wrote {summary_txt}")


if __name__ == "__main__":
    main()
PY
