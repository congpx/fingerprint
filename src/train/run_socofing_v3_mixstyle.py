import argparse, json
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

ALT_CODES = {"Obl":"Obl","CR":"CR","Zcut":"Zcut","ZCut":"Zcut","ZCUT":"Zcut"}

def _parse_list(s: str):
    s = (s or "").strip()
    if not s: return None
    return [x.strip() for x in s.split(",") if x.strip()]

def find_dirs(data_root: Path):
    real = data_root/"Real"
    altered = data_root/"Altered"
    easy = (altered/"Altered-Easy") if altered.exists() else (data_root/"Altered-Easy")
    medium = (altered/"Altered-Medium") if altered.exists() else (data_root/"Altered-Medium")
    hard = (altered/"Altered-Hard") if altered.exists() else (data_root/"Altered-Hard")
    return {"real": real if real.exists() else None,
            "easy": easy if easy.exists() else None,
            "medium": medium if medium.exists() else None,
            "hard": hard if hard.exists() else None}

def parse_name(p: Path):
    stem = p.stem
    if "__" in stem:
        sid, rest = stem.split("__",1)
    else:
        toks = stem.split("_")
        sid, rest = toks[0], "_".join(toks[1:])
    toks = rest.split("_")
    gender = toks[0] if len(toks)>0 else None
    hand   = toks[1] if len(toks)>1 else None
    finger = toks[2] if len(toks)>2 else None
    alt_type = None
    if len(toks)>=1 and toks[-1] in ALT_CODES:
        alt_type = ALT_CODES[toks[-1]]
    return {"subject_id": sid.zfill(3), "gender": gender, "hand": hand, "finger": finger, "alt_type": alt_type}

def scan_images(folder: Path, data_root: Path, severity: str):
    exts={".bmp",".BMP",".png",".PNG",".jpg",".JPG",".jpeg",".JPEG"}
    rows=[]
    for p in folder.rglob("*"):
        if p.is_file() and p.suffix in exts:
            meta=parse_name(p)
            meta.update({"relpath": str(p.relative_to(data_root)),
                         "severity": severity,
                         "is_altered": severity!="real"})
            rows.append(meta)
    return rows

def build_index(data_root: Path, out_csv: Path, seed: int=42):
    dirs=find_dirs(data_root)
    if not dirs["real"]:
        raise FileNotFoundError(f"Không thấy Real trong {data_root}")
    rows=[]
    rows += scan_images(dirs["real"], data_root, "real")
    for sev in ["easy","medium","hard"]:
        if dirs[sev]:
            rows += scan_images(dirs[sev], data_root, sev)
    df=pd.DataFrame(rows)
    df["finger_id"]=df["subject_id"].astype(str)+"|"+df["hand"].astype(str)+"|"+df["finger"].astype(str)

    rng=np.random.default_rng(seed)
    subjects=sorted(df["subject_id"].unique().tolist())
    rng.shuffle(subjects)
    n=len(subjects)
    n_train=int(0.70*n); n_val=int(0.15*n)
    train=set(subjects[:n_train]); val=set(subjects[n_train:n_train+n_val]); test=set(subjects[n_train+n_val:])
    def split_of(sid):
        if sid in train: return "train"
        if sid in val: return "val"
        return "test"
    df["split"]=df["subject_id"].map(split_of)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print("[OK] wrote index ->", out_csv)
    print(df.groupby(["severity","split"]).size())
    print("\n[alt_type counts]\n", df[df.severity!="real"].alt_type.value_counts(dropna=False))
    return df

def make_transforms():
    train_tf=T.Compose([
        T.Resize((224,224)),
        T.RandomRotation(8),
        T.RandomResizedCrop(224, scale=(0.90,1.0)),
        T.ToTensor(),
        T.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]),
    ])
    test_tf=T.Compose([
        T.Resize((224,224)),
        T.ToTensor(),
        T.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]),
    ])
    return train_tf, test_tf

class SOCOFingDataset(Dataset):
    def __init__(self, df, root: Path, tf, label_map: Optional[Dict[str,int]]=None):
        self.df=df.reset_index(drop=True)
        self.root=root; self.tf=tf; self.label_map=label_map
    def __len__(self): return len(self.df)
    def __getitem__(self,i):
        r=self.df.iloc[i]
        img=Image.open(self.root/r["relpath"]).convert("L").convert("RGB")
        x=self.tf(img)
        item={"image": x, "finger_id": r["finger_id"], "severity": r["severity"], "alt_type": r.get("alt_type",None), "relpath": r["relpath"]}
        if self.label_map is not None:
            item["label"]=self.label_map[r["finger_id"]]
        return item

def collate_train(b):
    x=torch.stack([o["image"] for o in b],0)
    y=torch.tensor([o["label"] for o in b],dtype=torch.long)
    return x,y

def collate_eval(b):
    x=torch.stack([o["image"] for o in b],0)
    fids=[o["finger_id"] for o in b]
    sevs=[o["severity"] for o in b]
    alts=[o.get("alt_type",None) for o in b]
    rps=[o["relpath"] for o in b]
    return x,fids,sevs,alts,rps

def _norm(x):
    return x/np.linalg.norm(x,axis=1,keepdims=True).clip(min=1e-12)

@torch.no_grad()
def extract(model, loader, device):
    model.eval()
    embs=[]; fids=[]; sevs=[]; alts=[]; rps=[]
    for x,_f,_s,_a,_r in tqdm(loader, desc="extract", leave=False):
        x=x.to(device, non_blocking=True)
        e,_=model(x)
        embs.append(e.cpu())
        fids.extend(_f); sevs.extend(_s); alts.extend(_a); rps.extend(_r)
    embs=torch.cat(embs,0).numpy()
    return embs,fids,sevs,alts,rps

def rank_metrics(probe_embs, probe_ids, gal_embs, gal_ids, topk=(1,5)):
    probe_embs=_norm(probe_embs); gal_embs=_norm(gal_embs)
    sims=probe_embs@gal_embs.T
    order=np.argsort(-sims,axis=1)
    out={}
    for k in topk:
        hits=0
        for i,pid in enumerate(probe_ids):
            top=[gal_ids[j] for j in order[i,:k]]
            hits += int(pid in top)
        out[f"rank{k}"]=hits/max(1,len(probe_ids))
    return out,sims

def eer_tar(probe_ids, gal_ids, sims, fars=(0.01,0.001)):
    y_true=[]; y_score=[]
    for i,pid in enumerate(probe_ids):
        same=np.array([1 if pid==gid else 0 for gid in gal_ids],dtype=np.int32)
        y_true.append(same); y_score.append(sims[i])
    y_true=np.concatenate(y_true); y_score=np.concatenate(y_score)
    fpr,tpr,_=roc_curve(y_true,y_score,pos_label=1)
    fnr=1-tpr
    idx=int(np.nanargmin(np.abs(fnr-fpr)))
    out={"eer": float((fpr[idx]+fnr[idx])/2)}
    for far in fars:
        valid=np.where(fpr<=far)[0]
        out[f"tar@far={far}"]= float(tpr[valid[-1]]) if len(valid)>0 else 0.0
    return out

@torch.no_grad()
def eval_retrieval(model, df_split, root: Path, batch, workers, device, probe_alt=None, probe_sev=None):
    _, test_tf = make_transforms()
    gal_df=df_split[df_split.severity=="real"].copy()
    gal_loader=DataLoader(SOCOFingDataset(gal_df,root,test_tf,None), batch_size=batch, shuffle=False,
                          num_workers=workers, pin_memory=True, collate_fn=collate_eval)
    gal_embs, gal_ids, _,_,_ = extract(model, gal_loader, device)
    out={}
    sevs = probe_sev or ["easy","medium","hard"]
    for sev in sevs:
        pr_df=df_split[df_split.severity==sev].copy()
        if probe_alt:
            pr_df=pr_df[pr_df.alt_type.isin(probe_alt)].copy()
        if len(pr_df)==0:
            out[sev]={"n_probe":0}; continue
        pr_loader=DataLoader(SOCOFingDataset(pr_df,root,test_tf,None), batch_size=batch, shuffle=False,
                             num_workers=workers, pin_memory=True, collate_fn=collate_eval)
        pr_embs, pr_ids, _,_,_ = extract(model, pr_loader, device)
        rm,sims=rank_metrics(pr_embs, pr_ids, gal_embs, gal_ids)
        sm=eer_tar(pr_ids, gal_ids, sims)
        out[sev]={"n_gallery": len(gal_ids), "n_probe": len(pr_ids), **rm, **sm}
    return out

def filter_by_alt(df, alt_types):
    if not alt_types: return df
    alt_types=set(alt_types)
    real=df[df.severity=="real"]
    alt=df[(df.severity!="real") & (df.alt_type.isin(alt_types))]
    return pd.concat([real,alt], ignore_index=True)

class MixStyle(nn.Module):
    def __init__(self, p=0.5, alpha=0.3, eps=1e-6):
        super().__init__()
        self.p=p; self.alpha=alpha; self.eps=eps
    def forward(self, x):
        if (not self.training) or (np.random.rand() > self.p):
            return x
        B,C,H,W = x.size()
        mu = x.mean(dim=[2,3], keepdim=True)
        var = x.var(dim=[2,3], keepdim=True, unbiased=False)
        sig = (var + self.eps).sqrt()
        x_norm = (x - mu) / sig

        perm = torch.randperm(B, device=x.device)
        mu2, sig2 = mu[perm], sig[perm]

        lam = torch.distributions.Beta(self.alpha, self.alpha).sample((B,1,1,1)).to(x.device)
        mu_mix = lam*mu + (1-lam)*mu2
        sig_mix = lam*sig + (1-lam)*sig2
        return x_norm * sig_mix + mu_mix

class FingerprintNet(nn.Module):
    def __init__(self, num_classes: int, emb_dim: int=256, pretrained: bool=False, mix_p: float=0.5, mix_alpha: float=0.3, mix_layer: str="layer1"):
        super().__init__()
        if pretrained:
            try:
                b=models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            except Exception:
                b=models.resnet18(weights=None)
        else:
            b=models.resnet18(weights=None)

        self.mix = MixStyle(p=mix_p, alpha=mix_alpha)
        self.mix_layer = mix_layer

        feat_dim=b.fc.in_features
        b.fc=nn.Identity()
        self.b=b
        self.emb=nn.Linear(feat_dim, emb_dim)
        self.bn=nn.BatchNorm1d(emb_dim)
        self.cls=nn.Linear(emb_dim, num_classes)

    def forward(self, x):
        # manual forward to inject mixstyle
        x = self.b.conv1(x); x = self.b.bn1(x); x = self.b.relu(x); x = self.b.maxpool(x)
        x = self.b.layer1(x)
        if self.mix_layer=="layer1": x = self.mix(x)
        x = self.b.layer2(x)
        if self.mix_layer=="layer2": x = self.mix(x)
        x = self.b.layer3(x)
        if self.mix_layer=="layer3": x = self.mix(x)
        x = self.b.layer4(x)
        x = self.b.avgpool(x)
        x = torch.flatten(x, 1)

        emb = self.bn(self.emb(x))
        emb = F.normalize(emb, dim=1)
        logits = self.cls(emb)
        return emb, logits

def save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(obj, open(path,"w"), indent=2)

def compute_selection_scores(metrics):
    ranks = [metrics[s]["rank1"] for s in ["easy","medium","hard"] if "rank1" in metrics.get(s,{})]
    mean_rank1 = float(np.mean(ranks)) if ranks else -1.0
    hard = metrics.get("hard", {})
    hard_rank1 = float(hard.get("rank1", -1.0))
    hard_tar001 = float(hard.get("tar@far=0.001", -1.0))
    return {
        "mean_rank1": mean_rank1,
        "hard_rank1": hard_rank1,
        "hard_tar001": hard_tar001,
    }


def save_best_checkpoint(outdir: Path, model: nn.Module, label_map, args, metrics, criterion: str, score: float):
    ckpt_dir = outdir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"best_{criterion}.pt"
    payload = {
        "model": model.state_dict(),
        "label_map": label_map,
        "args": vars(args),
        "selection_metric": criterion,
        "selection_score": float(score),
    }
    torch.save(payload, ckpt_path)
    save_json({
        "selection_metric": criterion,
        "selection_score": float(score),
        "metrics": metrics,
    }, outdir / f"best_val_metrics_{criterion}.json")

    if criterion == "mean_rank1":
        torch.save(payload, ckpt_dir / "best.pt")
        save_json(metrics, outdir / "best_val_metrics.json")

def train(args):
    root=Path(args.data_root)
    df=pd.read_csv(args.index)
    df["finger_id"]=df["finger_id"].astype(str)

    train_sev=_parse_list(args.train_severities) or ["real","easy","medium","hard"]
    train_alt=_parse_list(args.train_alt_types)
    val_alt=_parse_list(args.val_alt_types)

    train_df=df[(df.split=="train") & (df.severity.isin(train_sev))].copy()
    train_df=filter_by_alt(train_df, train_alt)

    fids=sorted(train_df.finger_id.unique().tolist())
    if not fids: raise RuntimeError("Train rỗng sau filter.")
    label_map={fid:i for i,fid in enumerate(fids)}

    train_tf,_=make_transforms()
    train_loader=DataLoader(SOCOFingDataset(train_df,root,train_tf,label_map), batch_size=args.batch, shuffle=True,
                            num_workers=args.workers, pin_memory=True, collate_fn=collate_train)

    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model=FingerprintNet(num_classes=len(fids), emb_dim=args.emb_dim, pretrained=args.pretrained,
                         mix_p=args.mix_p, mix_alpha=args.mix_alpha, mix_layer=args.mix_layer).to(device)

    opt=torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_fn=nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler=torch.amp.GradScaler("cuda", enabled=args.amp)

    outdir=Path(args.outdir); (outdir/"checkpoints").mkdir(parents=True, exist_ok=True)
    best_scores={"mean_rank1": -1.0, "hard_rank1": -1.0, "hard_tar001": -1.0}

    for epoch in range(1, args.epochs+1):
        model.train()
        run=0.0
        pbar=tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}")
        for x,y in pbar:
            x=x.to(device, non_blocking=True)
            y=y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", enabled=args.amp):
                _,logits=model(x)
                loss=loss_fn(logits,y)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
            run += loss.item()
            pbar.set_postfix(loss=run/max(1,len(pbar)))

        val_df=df[df.split=="val"].copy()
        metrics=eval_retrieval(model, val_df, root, args.eval_batch, args.workers, device, probe_alt=val_alt)
        scores=compute_selection_scores(metrics)
        mean=scores["mean_rank1"]

        print(f"\n[VAL] epoch={epoch} mean_rank1={mean:.4f} mix(p={args.mix_p},a={args.mix_alpha},layer={args.mix_layer}) val_alt={val_alt or 'ALL'}")
        print(f"    selection: hard_rank1={scores['hard_rank1']:.4f} hard_tar@far=0.001={scores['hard_tar001']:.4f}")
        for criterion, score in scores.items():
            if score > best_scores[criterion]:
                best_scores[criterion] = score
                save_best_checkpoint(outdir, model, label_map, args, metrics, criterion, score)
                print(f"[SAVE] best_{criterion}.pt (score={score:.4f})")

    print("[DONE] best scores = " + ", ".join([f"{k}={v:.4f}" for k,v in best_scores.items()]))

def evaluate(args):
    root=Path(args.data_root)
    df=pd.read_csv(args.index)
    df["finger_id"]=df["finger_id"].astype(str)

    ckpt=torch.load(args.ckpt, map_location="cpu")
    label_map=ckpt["label_map"]
    emb_dim=ckpt["args"].get("emb_dim",256)

    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model=FingerprintNet(num_classes=len(label_map), emb_dim=emb_dim, pretrained=False,
                         mix_p=0.0, mix_alpha=0.3, mix_layer="layer1").to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    probe_alt=_parse_list(args.probe_alt_types)
    probe_sev=_parse_list(args.probe_severities)
    split_df=df[df.split==args.split].copy()

    metrics=eval_retrieval(model, split_df, root, args.batch, args.workers, device, probe_alt=probe_alt, probe_sev=probe_sev)

    print(f"[EVAL] split={args.split} probe_alt_types={probe_alt or 'ALL'}")
    for sev in (probe_sev or ["easy","medium","hard"]):
        m=metrics.get(sev,{})
        if "rank1" not in m:
            print(" ",sev,": no data"); continue
        print(f"  {sev}: n_probe={m['n_probe']} rank1={m['rank1']:.4f} rank5={m['rank5']:.4f} eer={m['eer']:.4f} tar@far=0.001={m['tar@far=0.001']:.4f}")

    if args.out_json:
        json.dump(metrics, open(args.out_json,"w"), indent=2)
        print("[OK] wrote metrics ->", args.out_json)

def main():
    ap=argparse.ArgumentParser()
    sub=ap.add_subparsers(dest="cmd", required=True)

    ap_i=sub.add_parser("index")
    ap_i.add_argument("--data_root", required=True)
    ap_i.add_argument("--out", default="splits/index_v3.csv")
    ap_i.add_argument("--seed", type=int, default=42)

    ap_t=sub.add_parser("train")
    ap_t.add_argument("--data_root", required=True)
    ap_t.add_argument("--index", required=True)
    ap_t.add_argument("--outdir", default="outputs/exp_v3_mix")
    ap_t.add_argument("--epochs", type=int, default=30)
    ap_t.add_argument("--batch", type=int, default=128)
    ap_t.add_argument("--eval_batch", type=int, default=256)
    ap_t.add_argument("--workers", type=int, default=8)
    ap_t.add_argument("--lr", type=float, default=1e-4)
    ap_t.add_argument("--emb_dim", type=int, default=256)
    ap_t.add_argument("--train_severities", type=str, default="real,easy,medium")
    ap_t.add_argument("--train_alt_types", type=str, default="")
    ap_t.add_argument("--val_alt_types", type=str, default="")
    ap_t.add_argument("--pretrained", action="store_true")
    ap_t.add_argument("--amp", action="store_true")
    ap_t.add_argument("--mix_p", type=float, default=0.5)
    ap_t.add_argument("--mix_alpha", type=float, default=0.3)
    ap_t.add_argument("--mix_layer", type=str, default="layer1", choices=["layer1","layer2","layer3"])

    ap_e=sub.add_parser("eval")
    ap_e.add_argument("--data_root", required=True)
    ap_e.add_argument("--index", required=True)
    ap_e.add_argument("--ckpt", required=True)
    ap_e.add_argument("--split", type=str, default="test", choices=["val","test"])
    ap_e.add_argument("--batch", type=int, default=256)
    ap_e.add_argument("--workers", type=int, default=8)
    ap_e.add_argument("--probe_alt_types", type=str, default="")
    ap_e.add_argument("--probe_severities", type=str, default="")
    ap_e.add_argument("--out_json", type=str, default="")

    args=ap.parse_args()
    if args.cmd=="index":
        build_index(Path(args.data_root), Path(args.out), seed=args.seed)
    elif args.cmd=="train":
        train(args)
    elif args.cmd=="eval":
        evaluate(args)

if __name__=="__main__":
    main()
