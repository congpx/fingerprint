#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/congpx/fingerprint"
PYTHON_BIN="python"

DATA_ROOT="${PROJECT_ROOT}/data/SOCOFing/SOCOFing"
INDEX_CSV="${PROJECT_ROOT}/splits/index_v3.csv"
PHASE6_ROOT="${PROJECT_ROOT}/outputs_phase6_triplet"
OUTPUT_ROOT="${PROJECT_ROOT}/outputs_phase6_error_analysis"

RUN_TAG="${RUN_TAG:-w20_m20}"
SEEDS="${SEEDS:-1 2 3}"
BATCH="${BATCH:-256}"
WORKERS="${WORKERS:-8}"

mkdir -p "${OUTPUT_ROOT}"

echo "[CONFIG]"
echo "  PROJECT_ROOT = ${PROJECT_ROOT}"
echo "  DATA_ROOT    = ${DATA_ROOT}"
echo "  INDEX_CSV    = ${INDEX_CSV}"
echo "  PHASE6_ROOT  = ${PHASE6_ROOT}"
echo "  OUTPUT_ROOT  = ${OUTPUT_ROOT}"
echo "  RUN_TAG      = ${RUN_TAG}"
echo "  SEEDS        = ${SEEDS}"
echo "  BATCH        = ${BATCH}"
echo "  WORKERS      = ${WORKERS}"

"${PYTHON_BIN}" - <<'PY'
from pathlib import Path
from collections import Counter
import json
import math
import os
import statistics

import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.models as models

PROJECT_ROOT = Path("/home/congpx/fingerprint")
DATA_ROOT = PROJECT_ROOT / "data/SOCOFing/SOCOFing"
INDEX_CSV = PROJECT_ROOT / "splits/index_v3.csv"
PHASE6_ROOT = PROJECT_ROOT / "outputs_phase6_triplet"
OUTPUT_ROOT = PROJECT_ROOT / "outputs_phase6_error_analysis"
RUN_TAG = os.environ.get("RUN_TAG", "w20_m20")
SEEDS = [int(x) for x in os.environ.get("SEEDS", "1 2 3").split()]
BATCH = int(os.environ.get("BATCH", "256"))
WORKERS = int(os.environ.get("WORKERS", "8"))

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

class SOCOFingEvalDS(Dataset):
    def __init__(self, df: pd.DataFrame, data_root: Path, tf):
        self.df = df.reset_index(drop=True)
        self.root = data_root
        self.tf = tf

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        img = Image.open(self.root / r["relpath"]).convert("L").convert("RGB")
        x = self.tf(img)
        return x, r["finger_id"], r["relpath"]

def collate(batch):
    x = torch.stack([b[0] for b in batch], 0)
    ids = [b[1] for b in batch]
    rels = [b[2] for b in batch]
    return x, ids, rels

class MixStyle(nn.Module):
    def __init__(self, p=0.7, alpha=0.3, eps=1e-6):
        super().__init__()
        self.p = p
        self.alpha = alpha
        self.eps = eps
        self.beta = torch.distributions.Beta(alpha, alpha)

    def forward(self, x):
        if (not self.training) or self.p <= 0.0:
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
    def __init__(self, num_classes: int, emb_dim: int = 256, mix_p: float = 0.7, mix_alpha: float = 0.3, mix_layer: str = "layer1"):
        super().__init__()
        b = models.resnet18(weights=None)
        feat_dim = b.fc.in_features
        b.fc = nn.Identity()
        self.b = b
        self.mix = MixStyle(p=mix_p, alpha=mix_alpha)
        self.mix_layer = mix_layer
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

@torch.no_grad()
def extract(model, loader, device):
    model.eval()
    embs, ids, rels = [], [], []
    for x, fids, rps in loader:
        x = x.to(device, non_blocking=True)
        e, _ = model(x)
        embs.append(e.cpu())
        ids.extend(fids)
        rels.extend(rps)
    embs = torch.cat(embs, 0).numpy()
    return embs, ids, rels

def build_model_from_ckpt(ckpt, device):
    state = ckpt["model"]
    label_map = ckpt["label_map"]
    args = ckpt.get("args", {}) or {}
    emb_dim = args.get("emb_dim", 256)
    mix_p = args.get("mix_p", 0.7)
    mix_alpha = args.get("mix_alpha", 0.3)
    mix_layer = args.get("mix_layer", "layer1")

    model = MixFingerprintNet(
        num_classes=len(label_map),
        emb_dim=emb_dim,
        mix_p=mix_p,
        mix_alpha=mix_alpha,
        mix_layer=mix_layer,
    ).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model

def cosine_rank_analysis(probe_embs, probe_ids, probe_rels, gal_embs, gal_ids, gal_rels):
    probe_embs = probe_embs / np.linalg.norm(probe_embs, axis=1, keepdims=True).clip(1e-12)
    gal_embs = gal_embs / np.linalg.norm(gal_embs, axis=1, keepdims=True).clip(1e-12)
    sims = probe_embs @ gal_embs.T
    order = np.argsort(-sims, axis=1)
    rows = []

    for i, (pid, prel) in enumerate(zip(probe_ids, probe_rels)):
        ranking = order[i]
        ranked_ids = [gal_ids[j] for j in ranking]
        ranked_rels = [gal_rels[j] for j in ranking]
        ranked_scores = [float(sims[i, j]) for j in ranking]

        true_positions = [k for k, gid in enumerate(ranked_ids) if gid == pid]
        rank_true = true_positions[0] + 1 if true_positions else None
        top1_id = ranked_ids[0]
        top1_rel = ranked_rels[0]
        top1_score = ranked_scores[0]

        # first genuine match
        true_rel = None
        true_score = None
        if true_positions:
            k = true_positions[0]
            true_rel = ranked_rels[k]
            true_score = ranked_scores[k]

        rows.append({
            "probe_relpath": prel,
            "probe_finger_id": pid,
            "rank_true": rank_true,
            "is_fail_top1": int(top1_id != pid),
            "top1_relpath": top1_rel,
            "top1_finger_id": top1_id,
            "top1_score": top1_score,
            "genuine_relpath": true_rel,
            "genuine_score": true_score,
            "score_gap_top1_minus_true": (top1_score - true_score) if (true_score is not None) else None,
        })
    return pd.DataFrame(rows)

def make_figure2(consensus_csv: Path, output_png: Path, output_pdf: Path):
    df = pd.read_csv(consensus_csv)

    if len(df) == 0:
        print("[WARN] consensus_hard_failures.csv is empty; skip Figure 2.")
        return

    # choose representative cases
    near = df[df["fail_count"] >= 2].sort_values(["mean_rank_true", "fail_count"], ascending=[True, False]).head(2)
    cata = df[df["fail_count"] >= 2].sort_values(["mean_rank_true", "fail_count"], ascending=[False, False]).head(2)

    selected = []
    for _, r in near.iterrows():
        selected.append((r, "Near-miss"))
    for _, r in cata.iterrows():
        if r["probe_relpath"] not in [x[0]["probe_relpath"] for x in selected]:
            selected.append((r, "Catastrophic mismatch"))

    # ensure 4 rows if possible
    if len(selected) < 4:
        remaining = df.sort_values(["fail_count", "mean_rank_true"], ascending=[False, False])
        used = {x[0]["probe_relpath"] for x in selected}
        for _, r in remaining.iterrows():
            if r["probe_relpath"] in used:
                continue
            tag = "Catastrophic mismatch" if r["mean_rank_true"] > 20 else "Near-miss"
            selected.append((r, tag))
            used.add(r["probe_relpath"])
            if len(selected) == 4:
                break

    n_rows = len(selected)
    fig, axes = plt.subplots(n_rows, 3, figsize=(10.2, 2.7 * n_rows))
    if n_rows == 1:
        axes = np.array([axes])

    col_titles = ["Probe (CR-hard)", "Genuine gallery match", "Top-1 false match"]
    for j, t in enumerate(col_titles):
        axes[0, j].set_title(t, fontsize=12, fontweight="bold", pad=10)

    for i, (row, case_type) in enumerate(selected):
        probe_path = DATA_ROOT / row["probe_relpath"]
        genuine_path = DATA_ROOT / row["example_genuine_relpath"]
        false_path = DATA_ROOT / row["example_top1_relpath"]

        imgs = [
            Image.open(probe_path).convert("L"),
            Image.open(genuine_path).convert("L"),
            Image.open(false_path).convert("L"),
        ]

        for j, img in enumerate(imgs):
            ax = axes[i, j]
            ax.imshow(img, cmap="gray")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_linewidth(1.2)
                spine.set_edgecolor("black")

        subtitle = (
            f"{case_type} | fail_count={int(row['fail_count'])}, "
            f"mean rank_true={float(row['mean_rank_true']):.1f}"
        )
        axes[i, 0].text(
            -0.02, 1.08,
            subtitle,
            transform=axes[i, 0].transAxes,
            ha="left", va="bottom",
            fontsize=10.2, fontweight="bold"
        )
        axes[i, 0].text(0.5, -0.08, "altered probe", transform=axes[i, 0].transAxes,
                        ha="center", va="top", fontsize=8.8)
        axes[i, 1].text(0.5, -0.08, "correct match", transform=axes[i, 1].transAxes,
                        ha="center", va="top", fontsize=8.8, color="green")
        axes[i, 2].text(0.5, -0.08, "false top-1", transform=axes[i, 2].transAxes,
                        ha="center", va="top", fontsize=8.8, color="firebrick")

    plt.tight_layout(h_pad=1.6, w_pad=1.3)
    plt.savefig(output_png, dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.savefig(output_pdf, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"[OK] wrote {output_png}")
    print(f"[OK] wrote {output_pdf}")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tf = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    df = pd.read_csv(INDEX_CSV)
    df["finger_id"] = df["finger_id"].astype(str)

    gal_df = df[(df["split"] == "test") & (df["severity"] == "real")].copy()
    probe_df = df[(df["split"] == "test") & (df["severity"] == "hard") & (df["alt_type"] == "CR")].copy()

    gal_loader = DataLoader(
        SOCOFingEvalDS(gal_df, DATA_ROOT, tf),
        batch_size=BATCH,
        shuffle=False,
        num_workers=WORKERS,
        pin_memory=True,
        collate_fn=collate,
    )

    all_fails = []
    seed_rows = []

    for seed in SEEDS:
        run_name = f"mix_triplet_{RUN_TAG}_s{seed}"
        ckpt_path = PHASE6_ROOT / run_name / "checkpoints" / "best.pt"
        if not ckpt_path.exists():
            print(f"[WARN] missing checkpoint: {ckpt_path}")
            continue

        ckpt = torch.load(ckpt_path, map_location="cpu")
        model = build_model_from_ckpt(ckpt, device)

        probe_loader = DataLoader(
            SOCOFingEvalDS(probe_df, DATA_ROOT, tf),
            batch_size=BATCH,
            shuffle=False,
            num_workers=WORKERS,
            pin_memory=True,
            collate_fn=collate,
        )

        gal_embs, gal_ids, gal_rels = extract(model, gal_loader, device)
        probe_embs, probe_ids, probe_rels = extract(model, probe_loader, device)

        per_probe = cosine_rank_analysis(probe_embs, probe_ids, probe_rels, gal_embs, gal_ids, gal_rels)
        per_probe["seed"] = seed

        seed_dir = OUTPUT_ROOT / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        per_probe.to_csv(seed_dir / "cr_hard_all_probes.csv", index=False)

        fails = per_probe[per_probe["is_fail_top1"] == 1].copy()
        fails = fails.sort_values(["rank_true", "score_gap_top1_minus_true"], ascending=[False, False], na_position="last")
        fails.to_csv(seed_dir / "cr_hard_failures.csv", index=False)

        fail_rate = len(fails) / max(1, len(per_probe))
        seed_rows.append({
            "seed": seed,
            "n_probe": len(per_probe),
            "n_fail": len(fails),
            "fail_rate": fail_rate,
        })

        all_fails.append(fails)

    if len(all_fails) == 0:
        raise RuntimeError("No failure data generated. Check checkpoints and paths.")

    per_seed_summary = pd.DataFrame(seed_rows)
    per_seed_summary.to_csv(OUTPUT_ROOT / "per_seed_summary.csv", index=False)

    all_fails_df = pd.concat(all_fails, ignore_index=True)
    all_fails_df.to_csv(OUTPUT_ROOT / "all_failures_across_seeds.csv", index=False)

    agg_rows = []
    for probe_rel, sub in all_fails_df.groupby("probe_relpath"):
        top1_counts = Counter(sub["top1_relpath"].tolist())
        best_top1_rel, _ = top1_counts.most_common(1)[0]
        example_row = sub.iloc[0]
        agg_rows.append({
            "probe_relpath": probe_rel,
            "probe_finger_id": example_row["probe_finger_id"],
            "fail_count": int(len(sub)),
            "mean_rank_true": float(sub["rank_true"].mean()),
            "max_rank_true": int(sub["rank_true"].max()),
            "mean_score_gap_top1_minus_true": float(sub["score_gap_top1_minus_true"].mean()),
            "example_genuine_relpath": example_row["genuine_relpath"],
            "example_top1_relpath": best_top1_rel,
            "same_hand": int(str(example_row["probe_relpath"]).split("_")[2] == str(best_top1_rel).split("_")[2]) if len(str(best_top1_rel).split("_")) > 2 else None,
        })

    consensus = pd.DataFrame(agg_rows).sort_values(["fail_count", "mean_rank_true"], ascending=[False, False])
    consensus.to_csv(OUTPUT_ROOT / "consensus_hard_failures.csv", index=False)

    # summary txt
    lines = []
    lines.append("=== Phase 6 CR-hard Error Analysis ===")
    lines.append("")
    for _, r in per_seed_summary.iterrows():
        lines.append(
            f"seed {int(r['seed'])}: n_probe={int(r['n_probe'])}, "
            f"n_fail={int(r['n_fail'])}, fail_rate={float(r['fail_rate']):.4f}"
        )
    lines.append("")
    lines.append(f"Total failure records across seeds: {len(all_fails_df)}")
    lines.append(f"Unique failed probes: {consensus['probe_relpath'].nunique()}")
    lines.append("")
    lines.append("Top consensus hard failures:")
    for _, r in consensus.head(10).iterrows():
        lines.append(
            f"  {Path(r['probe_relpath']).name}: fail_count={int(r['fail_count'])}, "
            f"mean_rank_true={float(r['mean_rank_true']):.1f}, max_rank_true={int(r['max_rank_true'])}"
        )

    (OUTPUT_ROOT / "summary.txt").write_text("\n".join(lines))

    make_figure2(
        OUTPUT_ROOT / "consensus_hard_failures.csv",
        OUTPUT_ROOT / "figure2_cr_hard_examples.png",
        OUTPUT_ROOT / "figure2_cr_hard_examples.pdf",
    )

    print(f"[OK] wrote {OUTPUT_ROOT / 'per_seed_summary.csv'}")
    print(f"[OK] wrote {OUTPUT_ROOT / 'all_failures_across_seeds.csv'}")
    print(f"[OK] wrote {OUTPUT_ROOT / 'consensus_hard_failures.csv'}")
    print(f"[OK] wrote {OUTPUT_ROOT / 'summary.txt'}")

if __name__ == "__main__":
    main()
PY

echo
echo "[DONE] Outputs written to: ${OUTPUT_ROOT}"
echo "  - per_seed_summary.csv"
echo "  - all_failures_across_seeds.csv"
echo "  - consensus_hard_failures.csv"
echo "  - summary.txt"
echo "  - figure2_cr_hard_examples.png"
