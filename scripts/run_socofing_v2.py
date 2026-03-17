import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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


ALT_CODES = {"Obl": "Obl", "CR": "CR", "Zcut": "Zcut", "ZCut": "Zcut", "ZCUT": "Zcut"}


def find_dirs(data_root: Path) -> Dict[str, Optional[Path]]:
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


def parse_name(p: Path) -> Dict[str, Optional[str]]:
    """
    Expected examples:
      001__M_Left_index_finger.BMP
      001__M_Left_index_finger_Obl.BMP
    """
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


def scan_images(folder: Path, data_root: Path, severity: str) -> List[Dict]:
    exts = {".bmp", ".BMP", ".png", ".PNG", ".jpg", ".JPG", ".jpeg", ".JPEG"}
    rows = []
    for p in folder.rglob("*"):
        if p.is_file() and p.suffix in exts:
            meta = parse_name(p)
            meta.update(
                {
                    "relpath": str(p.relative_to(data_root)),
                    "severity": severity,
                    "is_altered": severity != "real",
                }
            )
            rows.append(meta)
    return rows


def build_index(data_root: Path, out_csv: Path, seed: int = 42) -> pd.DataFrame:
    dirs = find_dirs(data_root)
    if not dirs["real"]:
        raise FileNotFoundError(f"Không thấy thư mục Real trong {data_root}")

    rows = []
    rows += scan_images(dirs["real"], data_root, "real")
    for sev in ["easy", "medium", "hard"]:
        if dirs[sev]:
            rows += scan_images(dirs[sev], data_root, sev)

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("Không quét được ảnh nào. Kiểm tra lại dataset.")

    df["finger_id"] = (
        df["subject_id"].astype(str) + "|" + df["hand"].astype(str) + "|" + df["finger"].astype(str)
    )

    # split theo subject để train/test không lẫn người
    rng = np.random.default_rng(seed)
    subjects = sorted(df["subject_id"].unique().tolist())
    rng.shuffle(subjects)

    n = len(subjects)
    n_train = int(0.70 * n)
    n_val = int(0.15 * n)
    train_subjects = set(subjects[:n_train])
    val_subjects = set(subjects[n_train : n_train + n_val])
    test_subjects = set(subjects[n_train + n_val :])

    def assign_split(sid: str) -> str:
        if sid in train_subjects:
            return "train"
        if sid in val_subjects:
            return "val"
        return "test"

    df["split"] = df["subject_id"].map(assign_split)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    print(f"[OK] wrote index -> {out_csv}")
    print(df.groupby(["severity", "split"]).size())
    print("\n[Subjects]")
    print("train:", len(train_subjects), "val:", len(val_subjects), "test:", len(test_subjects))
    return df


class SOCOFingDataset(Dataset):
    def __init__(self, df: pd.DataFrame, data_root: Path, transform, label_map: Optional[Dict[str, int]] = None):
        self.df = df.reset_index(drop=True)
        self.data_root = data_root
        self.transform = transform
        self.label_map = label_map

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        p = self.data_root / row["relpath"]
        img = Image.open(p).convert("L").convert("RGB")
        x = self.transform(img)

        item = {
            "image": x,
            "finger_id": row["finger_id"],
            "severity": row["severity"],
            "relpath": row["relpath"],
        }

        if self.label_map is not None:
            item["label"] = self.label_map[row["finger_id"]]
        return item


def collate_train(batch):
    x = torch.stack([b["image"] for b in batch], dim=0)
    y = torch.tensor([b["label"] for b in batch], dtype=torch.long)
    return x, y


def collate_eval(batch):
    x = torch.stack([b["image"] for b in batch], dim=0)
    finger_ids = [b["finger_id"] for b in batch]
    severities = [b["severity"] for b in batch]
    relpaths = [b["relpath"] for b in batch]
    return x, finger_ids, severities, relpaths


class FingerprintNet(nn.Module):
    def __init__(self, num_classes: int, emb_dim: int = 256, pretrained: bool = False):
        super().__init__()
        if pretrained:
            try:
                backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            except Exception:
                backbone = models.resnet18(weights=None)
        else:
            backbone = models.resnet18(weights=None)

        feat_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.emb = nn.Linear(feat_dim, emb_dim)
        self.bn = nn.BatchNorm1d(emb_dim)
        self.cls = nn.Linear(emb_dim, num_classes)

    def forward(self, x):
        feat = self.backbone(x)
        emb = self.emb(feat)
        emb = self.bn(emb)
        emb = F.normalize(emb, dim=1)
        logits = self.cls(emb)
        return emb, logits


def make_transforms():
    train_tf = T.Compose(
        [
            T.Resize((224, 224)),
            T.RandomRotation(8),
            T.RandomResizedCrop(224, scale=(0.90, 1.0)),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    test_tf = T.Compose(
        [
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    return train_tf, test_tf


@torch.no_grad()
def extract_embeddings(model, loader, device):
    model.eval()
    embs = []
    finger_ids = []
    severities = []
    relpaths = []

    for x, fids, sevs, rps in tqdm(loader, desc="extract", leave=False):
        x = x.to(device, non_blocking=True)
        emb, _ = model(x)
        embs.append(emb.cpu())
        finger_ids.extend(fids)
        severities.extend(sevs)
        relpaths.extend(rps)

    embs = torch.cat(embs, dim=0).numpy()
    return embs, finger_ids, severities, relpaths


def compute_rank_metrics(probe_embs, probe_ids, gallery_embs, gallery_ids, topk=(1, 5)):
    probe_embs = probe_embs / np.linalg.norm(probe_embs, axis=1, keepdims=True).clip(min=1e-12)
    gallery_embs = gallery_embs / np.linalg.norm(gallery_embs, axis=1, keepdims=True).clip(min=1e-12)

    sims = probe_embs @ gallery_embs.T  # cosine similarity
    order = np.argsort(-sims, axis=1)

    results = {}
    for k in topk:
        k = min(k, len(gallery_ids))
        hits = 0
        for i, fid in enumerate(probe_ids):
            top_ids = [gallery_ids[j] for j in order[i, :k]]
            if fid in top_ids:
                hits += 1
        results[f"rank{k}"] = hits / max(1, len(probe_ids))
    return results, sims


def compute_eer_and_tar(probe_ids, gallery_ids, sims, far_targets=(0.01, 0.001)):
    y_true = []
    y_score = []

    for i, pfid in enumerate(probe_ids):
        same = np.array([1 if pfid == gfid else 0 for gfid in gallery_ids], dtype=np.int32)
        score = sims[i]
        y_true.append(same)
        y_score.append(score)

    y_true = np.concatenate(y_true)
    y_score = np.concatenate(y_score)

    fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=1)
    fnr = 1.0 - tpr
    idx = int(np.nanargmin(np.abs(fnr - fpr)))
    eer = float((fpr[idx] + fnr[idx]) / 2.0)

    out = {"eer": eer}
    for far in far_targets:
        valid = np.where(fpr <= far)[0]
        tar = float(tpr[valid[-1]]) if len(valid) > 0 else 0.0
        out[f"tar@far={far}"] = tar
    return out


@torch.no_grad()
def evaluate_retrieval(model, df_split, data_root: Path, batch: int, workers: int, device):
    _, test_tf = make_transforms()

    gallery_df = df_split[df_split["severity"] == "real"].copy()
    if len(gallery_df) == 0:
        raise RuntimeError("Gallery rỗng: split này không có ảnh real.")

    gallery_ds = SOCOFingDataset(gallery_df, data_root, test_tf, label_map=None)
    gallery_loader = DataLoader(
        gallery_ds,
        batch_size=batch,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        collate_fn=collate_eval,
    )
    gallery_embs, gallery_ids, _, _ = extract_embeddings(model, gallery_loader, device)

    all_metrics = {}
    for sev in ["easy", "medium", "hard"]:
        probe_df = df_split[df_split["severity"] == sev].copy()
        if len(probe_df) == 0:
            all_metrics[sev] = {"n_probe": 0}
            continue

        probe_ds = SOCOFingDataset(probe_df, data_root, test_tf, label_map=None)
        probe_loader = DataLoader(
            probe_ds,
            batch_size=batch,
            shuffle=False,
            num_workers=workers,
            pin_memory=True,
            collate_fn=collate_eval,
        )

        probe_embs, probe_ids, _, _ = extract_embeddings(model, probe_loader, device)
        rank_metrics, sims = compute_rank_metrics(probe_embs, probe_ids, gallery_embs, gallery_ids, topk=(1, 5))
        score_metrics = compute_eer_and_tar(probe_ids, gallery_ids, sims)

        metrics = {
            "n_gallery": len(gallery_ids),
            "n_probe": len(probe_ids),
            **rank_metrics,
            **score_metrics,
        }
        all_metrics[sev] = metrics

    return all_metrics


def save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def train(args):
    data_root = Path(args.data_root)
    df = pd.read_csv(args.index)
    df["finger_id"] = df["finger_id"].astype(str)

    train_sev = [s.strip() for s in args.train_severities.split(",")]
    train_df = df[(df["split"] == "train") & (df["severity"].isin(train_sev))].copy()
    val_df = df[df["split"] == "val"].copy()

    train_finger_ids = sorted(train_df["finger_id"].unique().tolist())
    if len(train_finger_ids) == 0:
        raise RuntimeError("Train split rỗng.")
    label_map = {fid: i for i, fid in enumerate(train_finger_ids)}

    train_tf, _ = make_transforms()
    train_ds = SOCOFingDataset(train_df, data_root, train_tf, label_map=label_map)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        collate_fn=collate_train,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FingerprintNet(
        num_classes=len(train_finger_ids),
        emb_dim=args.emb_dim,
        pretrained=args.pretrained,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    scaler = torch.amp.GradScaler("cuda", enabled=args.amp)

    outdir = Path(args.outdir)
    (outdir / "checkpoints").mkdir(parents=True, exist_ok=True)

    best_score = -1.0
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0

        pbar = tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}")
        for x, y in pbar:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", enabled=args.amp):
                _, logits = model(x)
                loss = criterion(logits, y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running += loss.item()
            pbar.set_postfix(loss=running / max(1, len(pbar)))

        # validation: retrieval trên subject chưa thấy
        val_metrics = evaluate_retrieval(
            model=model,
            df_split=val_df,
            data_root=data_root,
            batch=args.eval_batch,
            workers=args.workers,
            device=device,
        )

        val_rank1_mean = np.mean(
            [val_metrics[s]["rank1"] for s in ["easy", "medium", "hard"] if "rank1" in val_metrics[s]]
        )

        row = {"epoch": epoch, "train_loss": running / max(1, len(train_loader)), "val_rank1_mean": float(val_rank1_mean)}
        for sev in ["easy", "medium", "hard"]:
            for k, v in val_metrics[sev].items():
                row[f"val_{sev}_{k}"] = v
        history.append(row)

        print(f"\n[VAL] epoch={epoch} mean_rank1={val_rank1_mean:.4f}")
        for sev in ["easy", "medium", "hard"]:
            m = val_metrics[sev]
            if "rank1" in m:
                print(
                    f"  {sev}: rank1={m['rank1']:.4f} rank5={m['rank5']:.4f} "
                    f"eer={m['eer']:.4f} tar@far=0.01={m['tar@far=0.01']:.4f}"
                )

        if val_rank1_mean > best_score:
            best_score = float(val_rank1_mean)
            ckpt = {
                "model": model.state_dict(),
                "label_map": label_map,
                "args": vars(args),
            }
            torch.save(ckpt, outdir / "checkpoints" / "best.pt")
            save_json(val_metrics, outdir / "best_val_metrics.json")
            pd.DataFrame(history).to_csv(outdir / "history.csv", index=False)
            print(f"[SAVE] best -> {outdir / 'checkpoints' / 'best.pt'}")

    pd.DataFrame(history).to_csv(outdir / "history.csv", index=False)
    print(f"[DONE] best val mean rank1 = {best_score:.4f}")


def evaluate(args):
    data_root = Path(args.data_root)
    df = pd.read_csv(args.index)
    df["finger_id"] = df["finger_id"].astype(str)

    ckpt = torch.load(args.ckpt, map_location="cpu")
    label_map = ckpt["label_map"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FingerprintNet(
        num_classes=len(label_map),
        emb_dim=ckpt["args"].get("emb_dim", 256),
        pretrained=False,
    ).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    split_df = df[df["split"] == args.split].copy()
    metrics = evaluate_retrieval(
        model=model,
        df_split=split_df,
        data_root=data_root,
        batch=args.batch,
        workers=args.workers,
        device=device,
    )

    print(f"[EVAL] split={args.split}")
    for sev in ["easy", "medium", "hard"]:
        m = metrics[sev]
        if "rank1" not in m:
            print(f"  {sev}: no data")
            continue
        print(
            f"  {sev}: n_probe={m['n_probe']} rank1={m['rank1']:.4f} rank5={m['rank5']:.4f} "
            f"eer={m['eer']:.4f} tar@far=0.01={m['tar@far=0.01']:.4f} "
            f"tar@far=0.001={m['tar@far=0.001']:.4f}"
        )

    if args.out_json:
        save_json(metrics, Path(args.out_json))
        print(f"[OK] wrote metrics -> {args.out_json}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    ap_i = sub.add_parser("index")
    ap_i.add_argument("--data_root", required=True)
    ap_i.add_argument("--out", default="splits/index_v2.csv")
    ap_i.add_argument("--seed", type=int, default=42)

    ap_t = sub.add_parser("train")
    ap_t.add_argument("--data_root", required=True)
    ap_t.add_argument("--index", required=True)
    ap_t.add_argument("--outdir", default="outputs/exp_v2")
    ap_t.add_argument("--epochs", type=int, default=30)
    ap_t.add_argument("--batch", type=int, default=128)
    ap_t.add_argument("--eval_batch", type=int, default=256)
    ap_t.add_argument("--workers", type=int, default=8)
    ap_t.add_argument("--lr", type=float, default=1e-4)
    ap_t.add_argument("--emb_dim", type=int, default=256)
    ap_t.add_argument("--train_severities", type=str, default="real,easy,medium,hard")
    ap_t.add_argument("--pretrained", action="store_true")
    ap_t.add_argument("--amp", action="store_true")

    ap_e = sub.add_parser("eval")
    ap_e.add_argument("--data_root", required=True)
    ap_e.add_argument("--index", required=True)
    ap_e.add_argument("--ckpt", required=True)
    ap_e.add_argument("--split", type=str, default="test", choices=["val", "test"])
    ap_e.add_argument("--batch", type=int, default=256)
    ap_e.add_argument("--workers", type=int, default=8)
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
