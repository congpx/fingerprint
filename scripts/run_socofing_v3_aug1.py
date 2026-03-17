import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

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
    return {"real": real if real.exists() else None,
            "easy": easy if easy.exists() else None,
            "medium": medium if medium.exists() else None,
            "hard": hard if hard.exists() else None}

def parse_name(p: Path) -> Dict[str, Optional[str]]:
    stem = p.stem
    if "__" in stem:
        sid, rest = stem.split("__", 1)
    else:
        toks = stem.split("_")
        sid, rest = toks[0], "_".join(toks[1:])

    toks = rest.split("_")
    gender = toks[0] if len(toks) > 0 else None
    hand   = toks[1] if len(toks) > 1 else None
    finger = toks[2] if len(toks) > 2 else None

    alt_type = None
    if len(toks) >= 1 and toks[-1] in ALT_CODES:
        alt_type = ALT_CODES[toks[-1]]

    return {"subject_id": sid.zfill(3), "gender": gender, "hand": hand, "finger": finger, "alt_type": alt_type}

def scan_images(folder: Path, data_root: Path, severity: str) -> List[Dict]:
    exts = {".bmp", ".BMP", ".png", ".PNG", ".jpg", ".JPG", ".jpeg", ".JPEG"}
    rows = []
    for p in folder.rglob("*"):
        if p.is_file() and p.suffix in exts:
            meta = parse_name(p)
            meta.update({"relpath": str(p.relative_to(data_root)),
                         "severity": severity,
                         "is_altered": severity != "real"})
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
        raise RuntimeError("Không quét được ảnh nào. Kiểm tra dataset.")

    df["finger_id"] = df["subject_id"].astype(str) + "|" + df["hand"].astype(str) + "|" + df["finger"].astype(str)

    # split theo subject: 70/15/15
    rng = np.random.default_rng(seed)
    subjects = sorted(df["subject_id"].unique().tolist())
    rng.shuffle(subjects)
    n = len(subjects)
    n_train = int(0.70 * n)
    n_val   = int(0.15 * n)
    train_s = set(subjects[:n_train])
    val_s   = set(subjects[n_train:n_train+n_val])
    test_s  = set(subjects[n_train+n_val:])

    def split_of(sid: str) -> str:
        if sid in train_s: return "train"
        if sid in val_s:   return "val"
        return "test"

    df["split"] = df["subject_id"].map(split_of)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    print(f"[OK] wrote index -> {out_csv}")
    print(df.groupby(["severity","split"]).size())
    print("\n[Subjects]", "train:", len(train_s), "val:", len(val_s), "test:", len(test_s))
    print("\n[alt_type counts]")
    print(df[df.severity!="real"].alt_type.value_counts(dropna=False))
    return df

def make_transforms():
    train_tf = T.Compose([
        T.Resize((224,224)),
        T.RandomRotation(12),
        T.RandomResizedCrop(224, scale=(0.90, 1.0)),
        T.ToTensor(),
        T.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]),
    ])
    test_tf = T.Compose([
        T.Resize((224,224)),
        T.ToTensor(),
        T.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]),
    ])
    return train_tf, test_tf

class SOCOFingDataset(Dataset):
    def __init__(self, df: pd.DataFrame, data_root: Path, transform, label_map: Optional[Dict[str,int]]=None):
        self.df = df.reset_index(drop=True)
        self.root = data_root
        self.tf = transform
        self.label_map = label_map

    def __len__(self): return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        img = Image.open(self.root / r["relpath"]).convert("L").convert("RGB")
        x = self.tf(img)
        item = {"image": x, "finger_id": r["finger_id"], "severity": r["severity"], "alt_type": r.get("alt_type", None), "relpath": r["relpath"]}
        if self.label_map is not None:
            item["label"] = self.label_map[r["finger_id"]]
        return item

def collate_train(batch):
    x = torch.stack([b["image"] for b in batch], 0)
    y = torch.tensor([b["label"] for b in batch], dtype=torch.long)
    return x, y

def collate_eval(batch):
    x = torch.stack([b["image"] for b in batch], 0)
    fids = [b["finger_id"] for b in batch]
    sevs = [b["severity"] for b in batch]
    alts = [b.get("alt_type", None) for b in batch]
    rps  = [b["relpath"] for b in batch]
    return x, fids, sevs, alts, rps

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
        self.bn  = nn.BatchNorm1d(emb_dim)
        self.cls = nn.Linear(emb_dim, num_classes)

    def forward(self, x):
        feat = self.backbone(x)
        emb = self.bn(self.emb(feat))
        emb = F.normalize(emb, dim=1)
        logits = self.cls(emb)
        return emb, logits

@torch.no_grad()
def extract_embeddings(model, loader, device):
    model.eval()
    embs=[]; fids=[]; sevs=[]; alts=[]; rps=[]
    for x, _f, _s, _a, _r in tqdm(loader, desc="extract", leave=False):
        x = x.to(device, non_blocking=True)
        e, _ = model(x)
        embs.append(e.cpu())
        fids.extend(_f); sevs.extend(_s); alts.extend(_a); rps.extend(_r)
    embs = torch.cat(embs, 0).numpy()
    return embs, fids, sevs, alts, rps

def _norm(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True).clip(min=1e-12)

def compute_rank_metrics(probe_embs, probe_ids, gal_embs, gal_ids, topk=(1,5)):
    probe_embs = _norm(probe_embs); gal_embs = _norm(gal_embs)
    sims = probe_embs @ gal_embs.T
    order = np.argsort(-sims, axis=1)
    out={}
    for k in topk:
        hits=0
        for i,pid in enumerate(probe_ids):
            top = [gal_ids[j] for j in order[i,:k]]
            hits += int(pid in top)
        out[f"rank{k}"] = hits / max(1,len(probe_ids))
    return out, sims

def compute_eer_and_tar(probe_ids, gal_ids, sims, far_targets=(0.01,0.001)):
    y_true=[]; y_score=[]
    for i,pid in enumerate(probe_ids):
        same = np.array([1 if pid==gid else 0 for gid in gal_ids], dtype=np.int32)
        y_true.append(same)
        y_score.append(sims[i])
    y_true=np.concatenate(y_true); y_score=np.concatenate(y_score)
    fpr,tpr,_ = roc_curve(y_true, y_score, pos_label=1)
    fnr = 1 - tpr
    idx = int(np.nanargmin(np.abs(fnr - fpr)))
    out={"eer": float((fpr[idx] + fnr[idx]) / 2.0)}
    for far in far_targets:
        valid = np.where(fpr <= far)[0]
        tar = float(tpr[valid[-1]]) if len(valid)>0 else 0.0
        out[f"tar@far={far}"] = tar
    return out

def _parse_list(s: str) -> Optional[List[str]]:
    s = (s or "").strip()
    if not s:
        return None
    return [x.strip() for x in s.split(",") if x.strip()]

def filter_by_alt(df: pd.DataFrame, alt_types: Optional[List[str]]) -> pd.DataFrame:
    # Áp dụng filter alt_type cho altered; luôn giữ real
    if not alt_types:
        return df
    alt_types = set(alt_types)
    real = df[df["severity"]=="real"]
    alt  = df[(df["severity"]!="real") & (df["alt_type"].isin(alt_types))]
    return pd.concat([real, alt], axis=0, ignore_index=True)

@torch.no_grad()
def evaluate_retrieval(model, df_split, data_root: Path, batch: int, workers: int, device,
                       probe_alt_types: Optional[List[str]]=None,
                       probe_severities: Optional[List[str]]=None):
    _, test_tf = make_transforms()

    # gallery = real trong df_split
    gallery_df = df_split[df_split["severity"]=="real"].copy()
    if len(gallery_df)==0:
        raise RuntimeError("Gallery rỗng (không có real).")

    gal_loader = DataLoader(SOCOFingDataset(gallery_df, data_root, test_tf, None),
                            batch_size=batch, shuffle=False, num_workers=workers, pin_memory=True,
                            collate_fn=collate_eval)
    gal_embs, gal_ids, _, _, _ = extract_embeddings(model, gal_loader, device)

    out={}
    sevs = probe_severities or ["easy","medium","hard"]
    for sev in sevs:
        probe_df = df_split[df_split["severity"]==sev].copy()
        if probe_alt_types:
            probe_df = probe_df[probe_df["alt_type"].isin(probe_alt_types)].copy()
        if len(probe_df)==0:
            out[sev]={"n_probe":0}; continue

        pr_loader = DataLoader(SOCOFingDataset(probe_df, data_root, test_tf, None),
                               batch_size=batch, shuffle=False, num_workers=workers, pin_memory=True,
                               collate_fn=collate_eval)
        pr_embs, pr_ids, _, _, _ = extract_embeddings(model, pr_loader, device)
        rm, sims = compute_rank_metrics(pr_embs, pr_ids, gal_embs, gal_ids, topk=(1,5))
        sm = compute_eer_and_tar(pr_ids, gal_ids, sims)
        out[sev]={"n_gallery": len(gal_ids), "n_probe": len(pr_ids), **rm, **sm}
    return out

def save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path,"w",encoding="utf-8") as f:
        json.dump(obj,f,indent=2,ensure_ascii=False)

def train(args):
    data_root = Path(args.data_root)
    df = pd.read_csv(args.index)
    df["finger_id"] = df["finger_id"].astype(str)

    train_sev = _parse_list(args.train_severities) or ["real","easy","medium","hard"]
    train_alt = _parse_list(args.train_alt_types)  # None = all
    val_alt   = _parse_list(args.val_alt_types)    # None = all

    train_df = df[(df["split"]=="train") & (df["severity"].isin(train_sev))].copy()
    train_df = filter_by_alt(train_df, train_alt)  # real luôn giữ

    # label theo finger_id của train
    train_fids = sorted(train_df["finger_id"].unique().tolist())
    if len(train_fids)==0:
        raise RuntimeError("Train rỗng sau khi filter severities/alt_types.")
    label_map = {fid:i for i,fid in enumerate(train_fids)}

    train_tf, _ = make_transforms()
    train_loader = DataLoader(SOCOFingDataset(train_df, data_root, train_tf, label_map),
                              batch_size=args.batch, shuffle=True,
                              num_workers=args.workers, pin_memory=True,
                              collate_fn=collate_train)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FingerprintNet(num_classes=len(train_fids), emb_dim=args.emb_dim, pretrained=args.pretrained).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp)

    outdir = Path(args.outdir)
    (outdir/"checkpoints").mkdir(parents=True, exist_ok=True)

    best = -1.0
    history = []

    for epoch in range(1, args.epochs+1):
        model.train()
        running=0.0
        pbar=tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}")
        for x,y in pbar:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", enabled=args.amp):
                _, logits = model(x)
                loss = loss_fn(logits, y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            running += loss.item()
            pbar.set_postfix(loss=running/max(1,len(pbar)))

        # validation retrieval on VAL subjects
        val_df = df[df["split"]=="val"].copy()
        metrics = evaluate_retrieval(model, val_df, data_root,
                                     batch=args.eval_batch, workers=args.workers, device=device,
                                     probe_alt_types=val_alt)
        # mean rank1 across available severities
        ranks = [metrics[s]["rank1"] for s in ["easy","medium","hard"] if "rank1" in metrics[s]]
        mean_rank1 = float(np.mean(ranks)) if len(ranks)>0 else 0.0

        row={"epoch": epoch, "train_loss": running/max(1,len(train_loader)), "val_mean_rank1": mean_rank1}
        for sev in ["easy","medium","hard"]:
            for k,v in metrics.get(sev,{}).items():
                row[f"val_{sev}_{k}"]=v
        history.append(row)

        print(f"\n[VAL] epoch={epoch} mean_rank1={mean_rank1:.4f} (val_alt_types={val_alt or 'ALL'})")
        for sev in ["easy","medium","hard"]:
            m = metrics.get(sev,{})
            if "rank1" in m:
                print(f"  {sev}: rank1={m['rank1']:.4f} rank5={m['rank5']:.4f} eer={m['eer']:.4f} tar@far=0.001={m['tar@far=0.001']:.4f}")

        if mean_rank1 > best:
            best = mean_rank1
            torch.save({"model": model.state_dict(), "label_map": label_map, "args": vars(args)},
                       outdir/"checkpoints"/"best.pt")
            save_json(metrics, outdir/"best_val_metrics.json")
            pd.DataFrame(history).to_csv(outdir/"history.csv", index=False)
            print(f"[SAVE] best -> {outdir/'checkpoints'/'best.pt'}")

    pd.DataFrame(history).to_csv(outdir/"history.csv", index=False)
    print(f"[DONE] best val mean rank1 = {best:.4f}")

def evaluate(args):
    data_root = Path(args.data_root)
    df = pd.read_csv(args.index)
    df["finger_id"] = df["finger_id"].astype(str)

    ckpt = torch.load(args.ckpt, map_location="cpu")
    label_map = ckpt["label_map"]
    emb_dim = ckpt["args"].get("emb_dim", 256)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FingerprintNet(num_classes=len(label_map), emb_dim=emb_dim, pretrained=False).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    probe_alt = _parse_list(args.probe_alt_types)      # None = all
    probe_sev = _parse_list(args.probe_severities)     # None = default
    split_df = df[df["split"]==args.split].copy()

    metrics = evaluate_retrieval(model, split_df, data_root,
                                 batch=args.batch, workers=args.workers, device=device,
                                 probe_alt_types=probe_alt, probe_severities=probe_sev)

    print(f"[EVAL] split={args.split} probe_alt_types={probe_alt or 'ALL'}")
    for sev in (probe_sev or ["easy","medium","hard"]):
        m = metrics.get(sev,{})
        if "rank1" not in m:
            print(f"  {sev}: no data"); continue
        print(f"  {sev}: n_probe={m['n_probe']} rank1={m['rank1']:.4f} rank5={m['rank5']:.4f} "
              f"eer={m['eer']:.4f} tar@far=0.01={m['tar@far=0.01']:.4f} tar@far=0.001={m['tar@far=0.001']:.4f}")

    if args.out_json:
        save_json(metrics, Path(args.out_json))
        print(f"[OK] wrote metrics -> {args.out_json}")

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
    ap_t.add_argument("--outdir", default="outputs/exp_v3")
    ap_t.add_argument("--epochs", type=int, default=30)
    ap_t.add_argument("--batch", type=int, default=128)
    ap_t.add_argument("--eval_batch", type=int, default=256)
    ap_t.add_argument("--workers", type=int, default=8)
    ap_t.add_argument("--lr", type=float, default=1e-4)
    ap_t.add_argument("--emb_dim", type=int, default=256)
    ap_t.add_argument("--train_severities", type=str, default="real,easy,medium,hard")
    ap_t.add_argument("--train_alt_types", type=str, default="")    # e.g. "Obl"
    ap_t.add_argument("--val_alt_types", type=str, default="")      # e.g. "CR,Zcut"
    ap_t.add_argument("--pretrained", action="store_true")
    ap_t.add_argument("--amp", action="store_true")

    ap_e = sub.add_parser("eval")
    ap_e.add_argument("--data_root", required=True)
    ap_e.add_argument("--index", required=True)
    ap_e.add_argument("--ckpt", required=True)
    ap_e.add_argument("--split", type=str, default="test", choices=["val","test"])
    ap_e.add_argument("--batch", type=int, default=256)
    ap_e.add_argument("--workers", type=int, default=8)
    ap_e.add_argument("--probe_alt_types", type=str, default="")    # e.g. "Zcut"
    ap_e.add_argument("--probe_severities", type=str, default="")   # e.g. "hard"
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
