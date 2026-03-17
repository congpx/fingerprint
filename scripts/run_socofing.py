import argparse
import os
import re
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict, Tuple, List

import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.models as models

ALT_CODES = {"Obl": "Obl", "CR": "CR", "Zcut": "Zcut", "ZCut": "Zcut", "ZCUT": "Zcut"}

def _find_dirs(data_root: Path) -> Dict[str, Optional[Path]]:
    # Support both layouts:
    # data_root/Real + data_root/Altered/Altered-Easy ...
    # or data_root/Real + data_root/Altered-Easy ...
    real = data_root / "Real"
    altered = data_root / "Altered"
    easy = (altered / "Altered-Easy") if altered.exists() else (data_root / "Altered-Easy")
    med  = (altered / "Altered-Medium") if altered.exists() else (data_root / "Altered-Medium")
    hard = (altered / "Altered-Hard") if altered.exists() else (data_root / "Altered-Hard")
    return {
        "real": real if real.exists() else None,
        "easy": easy if easy.exists() else None,
        "medium": med if med.exists() else None,
        "hard": hard if hard.exists() else None,
    }

def parse_name(p: Path) -> Dict[str, Optional[str]]:
    """
    Handle both common patterns:
    - 100__M_Left_thumb_finger.BMP
    - 100__M_Left_thumb_finger_CR.BMP
    - 001_M_Left_little_finger_Obl.bmp
    """
    stem = p.stem
    if "__" in stem:
        sid, rest = stem.split("__", 1)
    else:
        parts = stem.split("_")
        sid, rest = parts[0], "_".join(parts[1:])

    toks = rest.split("_")
    gender = toks[0] if len(toks) > 0 else None
    hand   = toks[1] if len(toks) > 1 else None
    finger = toks[2] if len(toks) > 2 else None  # thumb/index/middle/ring/little

    alt = None
    if len(toks) >= 1 and toks[-1] in ALT_CODES:
        alt = ALT_CODES[toks[-1]]

    return {
        "subject_id": sid.zfill(3),
        "gender": gender,
        "hand": hand,
        "finger": finger,
        "alt_type": alt,  # None for real
    }

def scan_images(folder: Path, data_root: Path, severity: str) -> List[Dict]:
    rows = []
    exts = {".bmp", ".BMP", ".png", ".PNG", ".jpg", ".JPG", ".jpeg", ".JPEG"}
    for p in folder.rglob("*"):
        if p.is_file() and p.suffix in exts:
            meta = parse_name(p)
            meta.update({
                "relpath": str(p.relative_to(data_root)),
                "severity": severity,              # real/easy/medium/hard
                "is_altered": severity != "real",
            })
            rows.append(meta)
    return rows

def build_index(data_root: Path, out_csv: Path, seed: int = 42) -> pd.DataFrame:
    dirs = _find_dirs(data_root)
    if not dirs["real"]:
        raise FileNotFoundError(f"Không thấy thư mục Real trong {data_root}")

    rows = []
    rows += scan_images(dirs["real"], data_root, "real")
    for sev in ["easy", "medium", "hard"]:
        if dirs[sev]:
            rows += scan_images(dirs[sev], data_root, sev)

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("Không quét được ảnh nào. Kiểm tra lại đường dẫn dataset.")

    # split theo (subject_id, hand, finger) dựa trên ảnh REAL
    real_df = df[df["severity"] == "real"].copy()
    real_df["key"] = real_df["subject_id"].astype(str) + "|" + real_df["hand"].astype(str) + "|" + real_df["finger"].astype(str)

    rng = np.random.default_rng(seed)
    key_to_split: Dict[str, str] = {}

    # mỗi subject có ~10 ngón; chia 8 train / 1 val / 1 test theo key
    for sid, grp in real_df.groupby("subject_id"):
        keys = grp["key"].unique().tolist()
        rng.shuffle(keys)
        if len(keys) < 3:
            # fallback
            for k in keys:
                key_to_split[k] = "train"
            continue
        train_keys = keys[: max(1, int(0.8 * len(keys)))]
        remain = keys[len(train_keys):]
        val_keys = remain[:1]
        test_keys = remain[1:2] if len(remain) > 1 else remain[:]
        for k in train_keys: key_to_split[k] = "train"
        for k in val_keys:   key_to_split[k] = "val"
        for k in test_keys:  key_to_split[k] = "test"

    # gán split cho toàn bộ ảnh (real + altered) theo key
    df["key"] = df["subject_id"].astype(str) + "|" + df["hand"].astype(str) + "|" + df["finger"].astype(str)
    df["split"] = df["key"].map(key_to_split).fillna("train")

    # label = subject_id (để closed-set classification baseline)
    df.to_csv(out_csv, index=False)
    print(f"[OK] Wrote index: {out_csv} | rows={len(df)}")
    print(df.groupby(["severity","split"]).size())
    return df

class SOCOFingDataset(Dataset):
    def __init__(self, df: pd.DataFrame, data_root: Path, transform):
        self.df = df.reset_index(drop=True)
        self.data_root = data_root
        self.transform = transform

    def __len__(self): return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        p = self.data_root / row["relpath"]
        img = Image.open(p).convert("L")  # grayscale
        # convert to 3-channel to use standard torchvision backbones
        img = img.convert("RGB")
        x = self.transform(img)
        y = int(row["label"])
        return x, y

def make_model(num_classes: int, pretrained: bool):
    # weights download may fail if no internet; pretrained=False is safe default
    if pretrained:
        try:
            m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        except Exception:
            m = models.resnet18(weights=None)
    else:
        m = models.resnet18(weights=None)
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m

@torch.no_grad()
def eval_acc(model, loader, device):
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        pred = torch.argmax(logits, dim=1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return correct / max(1, total)

def train(args):
    data_root = Path(args.data_root)
    df = pd.read_csv(args.index)

    # label encode subjects (closed-set baseline)
    subjects = sorted(df["subject_id"].unique().tolist())
    sid2label = {sid: i for i, sid in enumerate(subjects)}
    df["label"] = df["subject_id"].map(sid2label).astype(int)
    num_classes = len(subjects)

    # dataset splits: train/val từ REAL, test sẽ đánh giá riêng từng severity
    train_df = df[(df["split"] == "train") & (df["severity"] == "real")].copy()
    val_df   = df[(df["split"] == "val")   & (df["severity"] == "real")].copy()
    if len(train_df) == 0 or len(val_df) == 0:
        raise RuntimeError("Split train/val rỗng. Hãy chạy lại bước index hoặc kiểm tra dataset.")

    transform_train = T.Compose([
        T.Resize((224,224)),
        T.RandomRotation(8),
        T.RandomResizedCrop(224, scale=(0.9, 1.0)),
        T.ToTensor(),
        T.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]),
    ])
    transform_test = T.Compose([
        T.Resize((224,224)),
        T.ToTensor(),
        T.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]),
    ])

    train_ds = SOCOFingDataset(train_df, data_root, transform_train)
    val_ds   = SOCOFingDataset(val_df, data_root, transform_test)

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=args.workers, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                              num_workers=args.workers, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = make_model(num_classes, pretrained=args.pretrained).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    outdir = Path(args.outdir)
    (outdir / "checkpoints").mkdir(parents=True, exist_ok=True)

    best = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}")
        running = 0.0
        for x, y in pbar:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=args.amp):
                logits = model(x)
                loss = loss_fn(logits, y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            running += loss.item()
            pbar.set_postfix(loss=running / max(1, len(pbar)))

        val_acc = eval_acc(model, val_loader, device)
        print(f"[VAL] epoch={epoch} acc={val_acc:.4f}")

        if val_acc > best:
            best = val_acc
            ckpt = {
                "model": model.state_dict(),
                "sid2label": sid2label,
                "args": vars(args),
            }
            torch.save(ckpt, outdir / "checkpoints" / "best.pt")
            print(f"[SAVE] best -> {outdir / 'checkpoints' / 'best.pt'}")

    print(f"[DONE] best val acc={best:.4f}")

@torch.no_grad()
def eval_all(args):
    data_root = Path(args.data_root)
    df = pd.read_csv(args.index)

    ckpt = torch.load(args.ckpt, map_location="cpu")
    sid2label = ckpt["sid2label"]
    df = df[df["subject_id"].isin(list(sid2label.keys()))].copy()
    df["label"] = df["subject_id"].map(sid2label).astype(int)
    num_classes = len(sid2label)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = make_model(num_classes, pretrained=False).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    transform = T.Compose([
        T.Resize((224,224)),
        T.ToTensor(),
        T.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]),
    ])

    def _eval(sev: str):
        sub = df[(df["split"] == "test") & (df["severity"] == sev)].copy()
        if len(sub) == 0:
            return None
        ds = SOCOFingDataset(sub, data_root, transform)
        loader = DataLoader(ds, batch_size=args.batch, shuffle=False,
                            num_workers=args.workers, pin_memory=True)
        acc = eval_acc(model, loader, device)
        return acc, len(sub)

    for sev in ["real", "easy", "medium", "hard"]:
        r = _eval(sev)
        if r is None:
            print(f"[TEST] {sev}: (no data)")
        else:
            acc, n = r
            print(f"[TEST] {sev}: acc={acc:.4f} (n={n})")

def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    ap_i = sub.add_parser("index")
    ap_i.add_argument("--data_root", required=True)
    ap_i.add_argument("--out", default="splits/index.csv")
    ap_i.add_argument("--seed", type=int, default=42)

    ap_t = sub.add_parser("train")
    ap_t.add_argument("--data_root", required=True)
    ap_t.add_argument("--index", required=True)
    ap_t.add_argument("--outdir", default="outputs/exp1")
    ap_t.add_argument("--epochs", type=int, default=20)
    ap_t.add_argument("--batch", type=int, default=64)
    ap_t.add_argument("--lr", type=float, default=1e-4)
    ap_t.add_argument("--workers", type=int, default=8)
    ap_t.add_argument("--pretrained", action="store_true")
    ap_t.add_argument("--amp", action="store_true")

    ap_e = sub.add_parser("eval")
    ap_e.add_argument("--data_root", required=True)
    ap_e.add_argument("--index", required=True)
    ap_e.add_argument("--ckpt", required=True)
    ap_e.add_argument("--batch", type=int, default=128)
    ap_e.add_argument("--workers", type=int, default=8)

    args = ap.parse_args()
    if args.cmd == "index":
        build_index(Path(args.data_root), Path(args.out), seed=args.seed)
    elif args.cmd == "train":
        train(args)
    elif args.cmd == "eval":
        eval_all(args)

if __name__ == "__main__":
    main()
