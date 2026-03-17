import argparse
import csv
import json
import math
import os
import random
from pathlib import Path
from typing import Dict, Optional, List

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler
import torchvision.transforms as T
import torchvision.models as models
from sklearn.metrics import roc_curve

ALT_CODES = {"Obl":"Obl","CR":"CR","Zcut":"Zcut","ZCut":"Zcut","ZCUT":"Zcut"}


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _parse_list(s: str):
    s = (s or "").strip()
    if not s:
        return None
    return [x.strip() for x in s.split(",") if x.strip()]


def find_dirs(data_root: Path):
    real = data_root / "Real"
    altered = data_root / "Altered"
    easy = (altered / "Altered-Easy") if altered.exists() else (data_root / "Altered-Easy")
    medium = (altered / "Altered-Medium") if altered.exists() else (data_root / "Altered-Medium")
    hard = (altered / "Altered-Hard") if altered.exists() else (data_root / "Altered-Hard")
    return {
        "real": real if real.exists() else None,
        "easy": easy if easy.exists() else None,
        "medium": medium if medium.exists() else None,
        "hard": hard if hard.exists() else None,
    }


def parse_name(p: Path):
    stem = p.stem
    if "__" in stem:
        sid, rest = stem.split("__", 1)
    else:
        toks = stem.split("_")
        sid, rest = toks[0], "_".join(toks[1:])
    toks = rest.split("_")
    gender = toks[0] if len(toks) > 0 else None
    hand = toks[1] if len(toks) > 1 else None
    finger = toks[2] if len(toks) > 2 else None
    alt_type = None
    if len(toks) >= 1 and toks[-1] in ALT_CODES:
        alt_type = ALT_CODES[toks[-1]]
    return {
        "subject_id": sid.zfill(3),
        "gender": gender,
        "hand": hand,
        "finger": finger,
        "alt_type": alt_type,
    }


def scan_images(folder: Path, data_root: Path, severity: str):
    exts = {".bmp", ".BMP", ".png", ".PNG", ".jpg", ".JPG", ".jpeg", ".JPEG"}
    rows = []
    for p in folder.rglob("*"):
        if p.is_file() and p.suffix in exts:
            meta = parse_name(p)
            meta.update({
                "relpath": str(p.relative_to(data_root)),
                "severity": severity,
                "is_altered": severity != "real",
            })
            rows.append(meta)
    return rows


def build_index(data_root: Path, out_csv: Path, seed: int = 42):
    dirs = find_dirs(data_root)
    if not dirs["real"]:
        raise FileNotFoundError(f"Không thấy Real trong {data_root}")
    rows = []
    rows += scan_images(dirs["real"], data_root, "real")
    for sev in ["easy", "medium", "hard"]:
        if dirs[sev]:
            rows += scan_images(dirs[sev], data_root, sev)
    df = pd.DataFrame(rows)
    df["finger_id"] = df["subject_id"].astype(str) + "|" + df["hand"].astype(str) + "|" + df["finger"].astype(str)

    rng = np.random.default_rng(seed)
    subjects = sorted(df["subject_id"].unique().tolist())
    rng.shuffle(subjects)
    n = len(subjects)
    n_train = int(0.70 * n)
    n_val = int(0.15 * n)
    train = set(subjects[:n_train])
    val = set(subjects[n_train:n_train + n_val])
    test = set(subjects[n_train + n_val:])

    def split_of(sid):
        if sid in train:
            return "train"
        if sid in val:
            return "val"
        return "test"

    df["split"] = df["subject_id"].map(split_of)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print("[OK] wrote index ->", out_csv)
    return df


def make_transforms():
    train_tf = T.Compose([
        T.Resize((224, 224)),
        T.RandomRotation(8),
        T.RandomResizedCrop(224, scale=(0.90, 1.0)),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    test_tf = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    return train_tf, test_tf


class SOCOFingDataset(Dataset):
    def __init__(self, df, root: Path, tf, label_map: Optional[Dict[str, int]] = None):
        self.df = df.reset_index(drop=True)
        self.root = root
        self.tf = tf
        self.label_map = label_map

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        img = Image.open(self.root / r["relpath"]).convert("L").convert("RGB")
        x = self.tf(img)
        item = {
            "image": x,
            "finger_id": r["finger_id"],
            "severity": r["severity"],
            "alt_type": r.get("alt_type", None),
            "relpath": r["relpath"],
        }
        if self.label_map is not None:
            item["label"] = self.label_map[r["finger_id"]]
        return item


def collate_train(batch):
    x = torch.stack([o["image"] for o in batch], 0)
    y = torch.tensor([o["label"] for o in batch], dtype=torch.long)
    return x, y


def collate_eval(batch):
    x = torch.stack([o["image"] for o in batch], 0)
    fids = [o["finger_id"] for o in batch]
    sevs = [o["severity"] for o in batch]
    alts = [o.get("alt_type", None) for o in batch]
    rps = [o["relpath"] for o in batch]
    return x, fids, sevs, alts, rps


class PKBatchSampler(Sampler[List[int]]):
    def __init__(self, labels: List[int], p: int = 32, k: int = 4, seed: int = 42):
        self.labels = list(labels)
        self.p = p
        self.k = k
        self.seed = seed
        self.label_to_indices = {}
        for idx, y in enumerate(self.labels):
            self.label_to_indices.setdefault(int(y), []).append(idx)
        self.identities = sorted(self.label_to_indices.keys())
        self.batch_size = self.p * self.k
        self.num_samples = len(self.labels)

    def __iter__(self):
        rng = np.random.default_rng(self.seed + np.random.randint(0, 10_000_000))
        ids = self.identities.copy()
        rng.shuffle(ids)
        batches = math.ceil(self.num_samples / self.batch_size)
        for _ in range(batches):
            chosen = rng.choice(ids, size=min(self.p, len(ids)), replace=False)
            batch = []
            for ident in chosen:
                pool = self.label_to_indices[int(ident)]
                replace = len(pool) < self.k
                picks = rng.choice(pool, size=self.k, replace=replace)
                batch.extend([int(i) for i in picks])
            rng.shuffle(batch)
            yield batch

    def __len__(self):
        return math.ceil(self.num_samples / self.batch_size)


def _norm(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True).clip(min=1e-12)


@torch.no_grad()
def extract(model, loader, device):
    model.eval()
    embs, fids, sevs, alts, rps = [], [], [], [], []
    for x, _f, _s, _a, _r in tqdm(loader, desc="extract", leave=False):
        x = x.to(device, non_blocking=True)
        e, _ = model(x)
        embs.append(e.cpu())
        fids.extend(_f)
        sevs.extend(_s)
        alts.extend(_a)
        rps.extend(_r)
    embs = torch.cat(embs, 0).numpy()
    return embs, fids, sevs, alts, rps


def rank_metrics(probe_embs, probe_ids, gal_embs, gal_ids, topk=(1, 5)):
    probe_embs = _norm(probe_embs)
    gal_embs = _norm(gal_embs)
    sims = probe_embs @ gal_embs.T
    order = np.argsort(-sims, axis=1)
    out = {}
    for k in topk:
        hits = 0
        for i, pid in enumerate(probe_ids):
            top = [gal_ids[j] for j in order[i, :k]]
            hits += int(pid in top)
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


@torch.no_grad()
def eval_retrieval(model, df_split, root: Path, batch, workers, device, probe_alt=None, probe_sev=None):
    _, test_tf = make_transforms()
    gal_df = df_split[df_split.severity == "real"].copy()
    gal_loader = DataLoader(
        SOCOFingDataset(gal_df, root, test_tf, None),
        batch_size=batch,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        collate_fn=collate_eval,
    )
    gal_embs, gal_ids, _, _, _ = extract(model, gal_loader, device)
    out = {}
    sevs = probe_sev or ["easy", "medium", "hard"]
    for sev in sevs:
        pr_df = df_split[df_split.severity == sev].copy()
        if probe_alt:
            pr_df = pr_df[pr_df.alt_type.isin(probe_alt)].copy()
        if len(pr_df) == 0:
            out[sev] = {"n_probe": 0}
            continue
        pr_loader = DataLoader(
            SOCOFingDataset(pr_df, root, test_tf, None),
            batch_size=batch,
            shuffle=False,
            num_workers=workers,
            pin_memory=True,
            collate_fn=collate_eval,
        )
        pr_embs, pr_ids, _, _, _ = extract(model, pr_loader, device)
        rm, sims = rank_metrics(pr_embs, pr_ids, gal_embs, gal_ids)
        sm = eer_tar(pr_ids, gal_ids, sims)
        out[sev] = {"n_gallery": len(gal_ids), "n_probe": len(pr_ids), **rm, **sm}
    return out


def filter_by_alt(df, alt_types):
    if not alt_types:
        return df
    alt_types = set(alt_types)
    real = df[df.severity == "real"]
    alt = df[(df.severity != "real") & (df.alt_type.isin(alt_types))]
    return pd.concat([real, alt], ignore_index=True)


class MixStyle(nn.Module):
    def __init__(self, p=0.5, alpha=0.3, eps=1e-6):
        super().__init__()
        self.p = p
        self.alpha = alpha
        self.eps = eps

    def forward(self, x):
        if (not self.training) or (np.random.rand() > self.p):
            return x
        B, C, H, W = x.size()
        mu = x.mean(dim=[2, 3], keepdim=True)
        var = x.var(dim=[2, 3], keepdim=True, unbiased=False)
        sig = (var + self.eps).sqrt()
        x_norm = (x - mu) / sig

        perm = torch.randperm(B, device=x.device)
        mu2, sig2 = mu[perm], sig[perm]
        lam = torch.distributions.Beta(self.alpha, self.alpha).sample((B, 1, 1, 1)).to(x.device)
        mu_mix = lam * mu + (1 - lam) * mu2
        sig_mix = lam * sig + (1 - lam) * sig2
        return x_norm * sig_mix + mu_mix


class FingerprintNet(nn.Module):
    def __init__(self, num_classes: int, emb_dim: int = 256, pretrained: bool = False, mix_p: float = 0.5, mix_alpha: float = 0.3, mix_layer: str = "layer1"):
        super().__init__()
        if pretrained:
            try:
                b = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            except Exception:
                b = models.resnet18(weights=None)
        else:
            b = models.resnet18(weights=None)
        self.mix = MixStyle(p=mix_p, alpha=mix_alpha)
        self.mix_layer = mix_layer
        feat_dim = b.fc.in_features
        b.fc = nn.Identity()
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


def save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def append_history(path: Path, row: Dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def load_init_ckpt(model: nn.Module, ckpt_path: Optional[str]):
    if not ckpt_path:
        return
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"[INIT] loaded from {ckpt_path}")
    if missing:
        print(f"[INIT] missing keys: {len(missing)}")
    if unexpected:
        print(f"[INIT] unexpected keys: {len(unexpected)}")


def batch_hard_triplet_loss(emb: torch.Tensor, labels: torch.Tensor, margin: float = 0.2):
    # compute in float32 for AMP stability
    emb = emb.float()
    sim = emb @ emb.t()
    dist = (1.0 - sim).clamp_min(0.0)

    N = labels.size(0)
    labels = labels.view(N, 1)
    mask_pos = labels.eq(labels.t())
    mask_pos.fill_diagonal_(False)
    mask_neg = ~labels.eq(labels.t())

    hardest_pos = dist.masked_fill(~mask_pos, float("-inf")).max(dim=1).values
    hardest_neg = dist.masked_fill(~mask_neg, float("inf")).min(dim=1).values

    valid = mask_pos.any(dim=1) & mask_neg.any(dim=1)
    if valid.sum().item() == 0:
        return emb.new_tensor(0.0), 0

    loss = F.relu(hardest_pos[valid] - hardest_neg[valid] + margin)
    return loss.mean(), int(valid.sum().item())


def train(args):
    set_seed(args.seed)
    root = Path(args.data_root)
    df = pd.read_csv(args.index)
    df["finger_id"] = df["finger_id"].astype(str)

    train_sev = _parse_list(args.train_severities) or ["real", "easy", "medium", "hard"]
    train_alt = _parse_list(args.train_alt_types)
    val_alt = _parse_list(args.val_alt_types)

    train_df = df[(df.split == "train") & (df.severity.isin(train_sev))].copy()
    train_df = filter_by_alt(train_df, train_alt)

    fids = sorted(train_df.finger_id.unique().tolist())
    if not fids:
        raise RuntimeError("Train rỗng sau filter.")
    label_map = {fid: i for i, fid in enumerate(fids)}
    train_df["label"] = train_df["finger_id"].map(label_map).astype(int)

    train_tf, _ = make_transforms()
    train_ds = SOCOFingDataset(train_df, root, train_tf, label_map)
    pk_sampler = PKBatchSampler(
        labels=train_df["label"].tolist(),
        p=args.batch_identities,
        k=args.batch_instances,
        seed=args.seed,
    )
    train_loader = DataLoader(
        train_ds,
        batch_sampler=pk_sampler,
        num_workers=args.workers,
        pin_memory=True,
        collate_fn=collate_train,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FingerprintNet(
        num_classes=len(fids),
        emb_dim=args.emb_dim,
        pretrained=args.pretrained,
        mix_p=args.mix_p,
        mix_alpha=args.mix_alpha,
        mix_layer=args.mix_layer,
    ).to(device)
    load_init_ckpt(model, args.init_ckpt)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    ce_loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp)

    outdir = Path(args.outdir)
    (outdir / "checkpoints").mkdir(parents=True, exist_ok=True)
    history_path = outdir / "history.csv"
    best_tar = -1.0
    best_rank1 = -1.0
    best_eer = 1e9

    for epoch in range(1, args.epochs + 1):
        model.train()
        run_total = 0.0
        run_ce = 0.0
        run_tri = 0.0
        run_valid_triplets = 0
        pbar = tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}")
        for x, y in pbar:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", enabled=args.amp):
                emb, logits = model(x)
                ce = ce_loss_fn(logits, y)
            tri, n_valid = batch_hard_triplet_loss(emb, y, margin=args.triplet_margin)
            loss = ce + args.triplet_weight * tri
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            run_total += float(loss.item())
            run_ce += float(ce.item())
            run_tri += float(tri.item())
            run_valid_triplets += n_valid
            pbar.set_postfix(loss=f"{run_total/max(1,len(pbar)):.4f}", ce=f"{run_ce/max(1,len(pbar)):.4f}", tri=f"{run_tri/max(1,len(pbar)):.4f}")

        val_df = df[df.split == "val"].copy()
        metrics = eval_retrieval(model, val_df, root, args.eval_batch, args.workers, device, probe_alt=val_alt)
        hard = metrics.get("hard", {})
        mean_rank1 = float(np.mean([metrics[s]["rank1"] for s in ["easy", "medium", "hard"] if "rank1" in metrics[s]]))
        hard_rank1 = float(hard.get("rank1", 0.0))
        hard_tar = float(hard.get("tar@far=0.001", 0.0))
        hard_eer = float(hard.get("eer", 1.0))

        row = {
            "epoch": epoch,
            "train_loss": run_total / max(1, len(train_loader)),
            "train_ce": run_ce / max(1, len(train_loader)),
            "train_triplet": run_tri / max(1, len(train_loader)),
            "train_valid_triplets": run_valid_triplets,
            "val_mean_rank1": mean_rank1,
            "val_hard_rank1": hard_rank1,
            "val_hard_tar001": hard_tar,
            "val_hard_eer": hard_eer,
        }
        append_history(history_path, row)

        print(
            f"\n[VAL] epoch={epoch} mean_rank1={mean_rank1:.4f} hard_rank1={hard_rank1:.4f} "
            f"hard_tar001={hard_tar:.4f} hard_eer={hard_eer:.4f} "
            f"triplet(w={args.triplet_weight}, m={args.triplet_margin})"
        )

        is_better = (hard_tar > best_tar + 1e-12) or \
                    (abs(hard_tar - best_tar) <= 1e-12 and hard_rank1 > best_rank1 + 1e-12) or \
                    (abs(hard_tar - best_tar) <= 1e-12 and abs(hard_rank1 - best_rank1) <= 1e-12 and hard_eer < best_eer - 1e-12)
        if is_better:
            best_tar, best_rank1, best_eer = hard_tar, hard_rank1, hard_eer
            payload = {"model": model.state_dict(), "label_map": label_map, "args": vars(args)}
            torch.save(payload, outdir / "checkpoints" / "best.pt")
            save_json(metrics, outdir / "best_val_metrics.json")
            print("[SAVE] best ->", outdir / "checkpoints" / "best.pt")

    print(f"[DONE] best hard_tar001={best_tar:.4f} hard_rank1={best_rank1:.4f} hard_eer={best_eer:.4f}")


def evaluate(args):
    root = Path(args.data_root)
    df = pd.read_csv(args.index)
    df["finger_id"] = df["finger_id"].astype(str)

    ckpt = torch.load(args.ckpt, map_location="cpu")
    label_map = ckpt["label_map"]
    ckpt_args = ckpt.get("args", {})
    emb_dim = ckpt_args.get("emb_dim", 256)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FingerprintNet(
        num_classes=len(label_map),
        emb_dim=emb_dim,
        pretrained=False,
        mix_p=0.0,
        mix_alpha=ckpt_args.get("mix_alpha", 0.3),
        mix_layer=ckpt_args.get("mix_layer", "layer1"),
    ).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    probe_alt = _parse_list(args.probe_alt_types)
    probe_sev = _parse_list(args.probe_severities)
    split_df = df[df.split == args.split].copy()
    metrics = eval_retrieval(model, split_df, root, args.batch, args.workers, device, probe_alt=probe_alt, probe_sev=probe_sev)

    print(f"[EVAL] split={args.split} probe_alt_types={probe_alt or 'ALL'}")
    for sev in (probe_sev or ["easy", "medium", "hard"]):
        m = metrics.get(sev, {})
        if "rank1" not in m:
            print(" ", sev, ": no data")
            continue
        print(
            f"  {sev}: n_probe={m['n_probe']} rank1={m['rank1']:.4f} rank5={m['rank5']:.4f} "
            f"eer={m['eer']:.4f} tar@far=0.001={m['tar@far=0.001']:.4f}"
        )

    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump(metrics, f, indent=2)
        print("[OK] wrote metrics ->", args.out_json)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    ap_i = sub.add_parser("index")
    ap_i.add_argument("--data_root", required=True)
    ap_i.add_argument("--out", default="splits/index_v3.csv")
    ap_i.add_argument("--seed", type=int, default=42)

    ap_t = sub.add_parser("train")
    ap_t.add_argument("--data_root", required=True)
    ap_t.add_argument("--index", required=True)
    ap_t.add_argument("--outdir", default="outputs/exp_v3_mix_triplet")
    ap_t.add_argument("--epochs", type=int, default=12)
    ap_t.add_argument("--workers", type=int, default=8)
    ap_t.add_argument("--lr", type=float, default=5e-5)
    ap_t.add_argument("--emb_dim", type=int, default=256)
    ap_t.add_argument("--train_severities", type=str, default="real,easy,medium,hard")
    ap_t.add_argument("--train_alt_types", type=str, default="Obl")
    ap_t.add_argument("--val_alt_types", type=str, default="CR")
    ap_t.add_argument("--pretrained", action="store_true")
    ap_t.add_argument("--amp", action="store_true")
    ap_t.add_argument("--mix_p", type=float, default=0.7)
    ap_t.add_argument("--mix_alpha", type=float, default=0.3)
    ap_t.add_argument("--mix_layer", type=str, default="layer1", choices=["layer1", "layer2", "layer3"])
    ap_t.add_argument("--init_ckpt", type=str, default="")
    ap_t.add_argument("--triplet_weight", type=float, default=0.2)
    ap_t.add_argument("--triplet_margin", type=float, default=0.2)
    ap_t.add_argument("--batch_identities", type=int, default=32)
    ap_t.add_argument("--batch_instances", type=int, default=4)
    ap_t.add_argument("--eval_batch", type=int, default=256)
    ap_t.add_argument("--seed", type=int, default=1)

    ap_e = sub.add_parser("eval")
    ap_e.add_argument("--data_root", required=True)
    ap_e.add_argument("--index", required=True)
    ap_e.add_argument("--ckpt", required=True)
    ap_e.add_argument("--split", type=str, default="test", choices=["val", "test"])
    ap_e.add_argument("--batch", type=int, default=256)
    ap_e.add_argument("--workers", type=int, default=8)
    ap_e.add_argument("--probe_alt_types", type=str, default="")
    ap_e.add_argument("--probe_severities", type=str, default="")
    ap_e.add_argument("--out_json", type=str, default="")

    args = ap.parse_args()
    if args.cmd == "index":
        build_index(Path(args.data_root), Path(args.out), seed=args.seed)
    elif args.cmd == "train":
        train(args)
    elif args.cmd == "eval":
        evaluate(args)


if __name__ == "__main__":
    main()
