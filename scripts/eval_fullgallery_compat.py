import argparse, json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.models as models
from sklearn.metrics import roc_curve


class SOCOFingEvalDS(Dataset):
    def __init__(self, df, data_root: Path, tf):
        self.df = df.reset_index(drop=True)
        self.root = data_root
        self.tf = tf

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        img = Image.open(self.root / r["relpath"]).convert("L").convert("RGB")
        x = self.tf(img)
        return x, r["finger_id"]


def collate(batch):
    x = torch.stack([b[0] for b in batch], 0)
    ids = [b[1] for b in batch]
    return x, ids


class BaseFingerprintNet(nn.Module):
    def __init__(self, num_classes: int, emb_dim: int = 256):
        super().__init__()
        backbone = models.resnet18(weights=None)
        feat_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.emb = nn.Linear(feat_dim, emb_dim)
        self.bn = nn.BatchNorm1d(emb_dim)
        self.cls = nn.Linear(emb_dim, num_classes)

    def forward(self, x):
        feat = self.backbone(x)
        emb = self.bn(self.emb(feat))
        emb = F.normalize(emb, dim=1)
        logits = self.cls(emb)
        return emb, logits


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


@torch.no_grad()
def extract(model, loader, device):
    model.eval()
    embs, ids = [], []
    for x, fids in tqdm(loader, desc="extract", leave=False):
        x = x.to(device, non_blocking=True)
        e, _ = model(x)
        embs.append(e.cpu())
        ids.extend(fids)
    embs = torch.cat(embs, 0).numpy()
    return embs, ids


def rank_metrics(probe_embs, probe_ids, gal_embs, gal_ids, topk=(1, 5)):
    probe_embs = probe_embs / np.linalg.norm(probe_embs, axis=1, keepdims=True).clip(1e-12)
    gal_embs = gal_embs / np.linalg.norm(gal_embs, axis=1, keepdims=True).clip(1e-12)
    sims = probe_embs @ gal_embs.T
    order = np.argsort(-sims, axis=1)
    out = {}
    for k in topk:
        hits = 0
        for i, fid in enumerate(probe_ids):
            top = [gal_ids[j] for j in order[i, :k]]
            hits += int(fid in top)
        out[f"rank{k}"] = hits / max(1, len(probe_ids))
    return out, sims


def eer_tar(probe_ids, gal_ids, sims, fars=(0.01, 0.001)):
    y_true, y_score = [], []
    for i, pid in enumerate(probe_ids):
        same = np.array([1 if pid == gid else 0 for gid in gal_ids], dtype=np.int32)
        y_true.append(same)
        y_score.append(sims[i])
    y_true = np.concatenate(y_true)
    y_score = np.concatenate(y_score)

    fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=1)
    fnr = 1 - tpr
    idx = int(np.nanargmin(np.abs(fnr - fpr)))

    out = {"eer": float((fpr[idx] + fnr[idx]) / 2)}
    for far in fars:
        valid = np.where(fpr <= far)[0]
        out[f"tar@far={far}"] = float(tpr[valid[-1]]) if len(valid) > 0 else 0.0
    return out


def build_model_from_ckpt(ckpt, device):
    state = ckpt["model"]
    label_map = ckpt["label_map"]
    args = ckpt.get("args", {}) or {}
    emb_dim = args.get("emb_dim", 256)

    is_mix = any(k.startswith("b.") for k in state.keys())

    if is_mix:
        model = MixFingerprintNet(
            num_classes=len(label_map),
            emb_dim=emb_dim,
            mix_p=0.0,
            mix_alpha=args.get("mix_alpha", 0.3),
            mix_layer=args.get("mix_layer", "layer1"),
        ).to(device)
    else:
        model = BaseFingerprintNet(
            num_classes=len(label_map),
            emb_dim=emb_dim,
        ).to(device)

    model.load_state_dict(state, strict=True)
    model.eval()
    return model, is_mix


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--index", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--probe_split", default="test", choices=["val", "test"])
    ap.add_argument("--probe_alt_types", default="")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out_json", default="")
    args = ap.parse_args()

    df = pd.read_csv(args.index)
    df["finger_id"] = df["finger_id"].astype(str)
    root = Path(args.data_root)

    ckpt = torch.load(args.ckpt, map_location="cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, is_mix = build_model_from_ckpt(ckpt, device)

    tf = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    gal_df = df[df["severity"] == "real"].copy()
    gal_loader = DataLoader(
        SOCOFingEvalDS(gal_df, root, tf),
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        collate_fn=collate,
    )
    gal_embs, gal_ids = extract(model, gal_loader, device)

    probe_alt_types = [x.strip() for x in args.probe_alt_types.split(",") if x.strip()]
    out = {}
    for sev in ["easy", "medium", "hard"]:
        pr_df = df[(df["split"] == args.probe_split) & (df["severity"] == sev)].copy()
        if probe_alt_types:
            pr_df = pr_df[pr_df["alt_type"].isin(probe_alt_types)].copy()

        if len(pr_df) == 0:
            out[sev] = {"n_probe": 0}
            continue

        pr_loader = DataLoader(
            SOCOFingEvalDS(pr_df, root, tf),
            batch_size=args.batch,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True,
            collate_fn=collate,
        )
        pr_embs, pr_ids = extract(model, pr_loader, device)
        rm, sims = rank_metrics(pr_embs, pr_ids, gal_embs, gal_ids)
        sm = eer_tar(pr_ids, gal_ids, sims)
        out[sev] = {"n_gallery": len(gal_ids), "n_probe": len(pr_ids), **rm, **sm}

    model_type = "mixstyle_v3" if is_mix else "base_v2v3"
    probe_tag = probe_alt_types if probe_alt_types else "ALL"
    print(f"[FULL-GALLERY EVAL] model={model_type} probe_split={args.probe_split} probe_alt_types={probe_tag} gallery=ALL_REAL")
    for sev in ["easy", "medium", "hard"]:
        m = out[sev]
        if "rank1" not in m:
            print(f"  {sev}: no data")
            continue
        print(
            f"  {sev}: n_gallery={m['n_gallery']} n_probe={m['n_probe']} "
            f"rank1={m['rank1']:.4f} rank5={m['rank5']:.4f} "
            f"eer={m['eer']:.4f} tar@far=0.01={m['tar@far=0.01']:.4f} tar@far=0.001={m['tar@far=0.001']:.4f}"
        )

    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_json, "w") as f:
            json.dump(out, f, indent=2)
        print("[OK] wrote ->", args.out_json)


if __name__ == "__main__":
    main()
