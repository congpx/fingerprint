import math
import os
import time
import json
from pathlib import Path
from typing import Dict, Any, List

import pandas as pd
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.models as models


PROJECT_ROOT = Path("/home/congpx/fingerprint")
DATA_ROOT = PROJECT_ROOT / "data/SOCOFing/SOCOFing"
INDEX_CSV = PROJECT_ROOT / "splits/index_v3.csv"

PHASE5_ROOT = PROJECT_ROOT / "outputs_phase5_final"
PHASE6_ROOT = PROJECT_ROOT / "outputs_phase6_triplet"

OUTPUT_DIR = PROJECT_ROOT / "outputs_computational_cost"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = DEVICE.type == "cuda"

N_INF_IMAGES = int(os.environ.get("N_INF_IMAGES", "128"))
INF_WARMUP = int(os.environ.get("INF_WARMUP", "20"))
INF_REPEATS = int(os.environ.get("INF_REPEATS", "100"))

TRAIN_WARMUP_STEPS = int(os.environ.get("TRAIN_WARMUP_STEPS", "2"))
TRAIN_MEASURE_STEPS = int(os.environ.get("TRAIN_MEASURE_STEPS", "10"))

MODELS = {
    "base_ref": {
        "ckpt": PHASE5_ROOT / "base_ref_s1" / "checkpoints" / "best.pt",
        "notes": "baseline retrieval model",
        "training_mode": "ce",
    },
    "mix_best": {
        "ckpt": PHASE5_ROOT / "mix_best_s1" / "checkpoints" / "best.pt",
        "notes": "MixStyle at layer1 (p=0.7, alpha=0.3)",
        "training_mode": "ce",
    },
    "mix_triplet": {
        "ckpt": PHASE6_ROOT / "mix_triplet_w20_m20_s1" / "checkpoints" / "best.pt",
        "notes": "same inference backbone, triplet only affects training",
        "training_mode": "ce_triplet",
    },
}


def ckpt_args_to_dict(args_obj: Any) -> Dict[str, Any]:
    if args_obj is None:
        return {}
    if isinstance(args_obj, dict):
        return args_obj
    if hasattr(args_obj, "__dict__"):
        return vars(args_obj)
    return {}


def parse_listish(v, default=None):
    if v is None:
        return default
    if isinstance(v, (list, tuple)):
        return list(v)
    if isinstance(v, str):
        return [x.strip() for x in v.split(",") if x.strip()]
    return default


def sizeof_mb(path: Path) -> float:
    return path.stat().st_size / (1024 ** 2)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def maybe_sync():
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()


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


def build_model_from_ckpt(ckpt: Dict[str, Any]) -> nn.Module:
    state = ckpt["model"]
    label_map = ckpt["label_map"]
    args = ckpt_args_to_dict(ckpt.get("args", {}))
    emb_dim = int(args.get("emb_dim", 256))

    is_mix = any(k.startswith("b.") for k in state.keys())
    if is_mix:
        model = MixFingerprintNet(
            num_classes=len(label_map),
            emb_dim=emb_dim,
            mix_p=float(args.get("mix_p", 0.7)),
            mix_alpha=float(args.get("mix_alpha", 0.3)),
            mix_layer=str(args.get("mix_layer", "layer1")),
        )
    else:
        model = BaseFingerprintNet(
            num_classes=len(label_map),
            emb_dim=emb_dim,
        )

    model.load_state_dict(state, strict=True)
    return model.to(DEVICE)


TRANSFORM = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])


class FingerprintLabelDataset(Dataset):
    def __init__(self, df: pd.DataFrame, label_map: Dict[str, int]):
        self.df = df.reset_index(drop=True)
        self.label_map = {str(k): int(v) for k, v in label_map.items()}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        img = Image.open(DATA_ROOT / r["relpath"]).convert("L").convert("RGB")
        x = TRANSFORM(img)
        y = self.label_map[str(r["finger_id"])]
        return x, y


class FingerprintImageDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        img = Image.open(DATA_ROOT / r["relpath"]).convert("L").convert("RGB")
        x = TRANSFORM(img)
        return x


def batch_hard_triplet_loss(emb: torch.Tensor, labels: torch.Tensor, margin: float = 0.2):
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
        return emb.new_tensor(0.0)

    loss = F.relu(hardest_pos[valid] - hardest_neg[valid] + margin)
    return loss.mean()


@torch.no_grad()
def benchmark_inference(model: nn.Module, images: List[torch.Tensor]) -> Dict[str, float]:
    model.eval()
    imgs = [img.unsqueeze(0).to(DEVICE, non_blocking=True) for img in images[:N_INF_IMAGES]]

    for i in range(min(INF_WARMUP, len(imgs))):
        with torch.autocast(device_type="cuda", enabled=USE_AMP):
            _ = model(imgs[i])
    maybe_sync()

    n = min(INF_REPEATS, len(imgs))
    start = time.perf_counter()
    for i in range(n):
        with torch.autocast(device_type="cuda", enabled=USE_AMP):
            _ = model(imgs[i])
    maybe_sync()
    end = time.perf_counter()
    latency_ms = (end - start) * 1000.0 / n

    batch_imgs = torch.cat(imgs[:min(32, len(imgs))], dim=0)
    for _ in range(10):
        with torch.autocast(device_type="cuda", enabled=USE_AMP):
            _ = model(batch_imgs)
    maybe_sync()

    start = time.perf_counter()
    for _ in range(30):
        with torch.autocast(device_type="cuda", enabled=USE_AMP):
            _ = model(batch_imgs)
    maybe_sync()
    end = time.perf_counter()
    throughput = (30 * batch_imgs.size(0)) / (end - start)

    return {
        "latency_ms_per_image_bs1": latency_ms,
        "throughput_img_per_s_bs32": throughput,
    }


def benchmark_training(model: nn.Module, loader: DataLoader, training_mode: str, epochs: int, lr: float, batch_size: int) -> Dict[str, float]:
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    ce_loss = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler(device="cuda", enabled=USE_AMP)

    step_times = []
    max_steps = TRAIN_WARMUP_STEPS + TRAIN_MEASURE_STEPS
    num_steps = 0

    for x, y in loader:
        x = x.to(DEVICE, non_blocking=True)
        y = y.to(DEVICE, non_blocking=True)

        maybe_sync()
        t0 = time.perf_counter()

        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", enabled=USE_AMP):
            emb, logits = model(x)
            loss = ce_loss(logits, y)
        if training_mode == "ce_triplet":
            tri = batch_hard_triplet_loss(emb, y, margin=0.2)
            loss = loss + 0.2 * tri

        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()

        maybe_sync()
        t1 = time.perf_counter()

        if num_steps >= TRAIN_WARMUP_STEPS:
            step_times.append(t1 - t0)

        num_steps += 1
        if num_steps >= max_steps:
            break

    avg_step_s = float(np.mean(step_times))
    steps_per_epoch = math.ceil(len(loader.dataset) / batch_size)
    epoch_time_min = avg_step_s * steps_per_epoch / 60.0
    total_train_time_h = epoch_time_min * epochs / 60.0

    return {
        "avg_train_step_s": avg_step_s,
        "steps_per_epoch": steps_per_epoch,
        "epoch_time_min_est": epoch_time_min,
        "total_train_time_h_est": total_train_time_h,
    }


def filter_train_df(df: pd.DataFrame, ckpt_args: Dict[str, Any]) -> pd.DataFrame:
    train_alt_types = parse_listish(ckpt_args.get("train_alt_types"), default=["Obl"])
    train_severities = parse_listish(ckpt_args.get("train_severities"), default=["real", "easy", "medium", "hard"])

    out = df[df["split"] == "train"].copy()
    out["finger_id"] = out["finger_id"].astype(str)

    allowed_sev = set(train_severities)
    out = out[out["severity"].isin(allowed_sev)]

    if train_alt_types:
        altered_mask = out["severity"] != "real"
        out = pd.concat([
            out[~altered_mask],
            out[altered_mask & out["alt_type"].isin(train_alt_types)]
        ], ignore_index=True)

    return out.reset_index(drop=True)


def select_inf_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df[(df["split"] == "test") & (df["severity"] == "hard")].copy()
    if len(out) < N_INF_IMAGES:
        out = df[df["split"] == "test"].copy()
    return out.head(N_INF_IMAGES).reset_index(drop=True)


def main():
    print(f"[INFO] Device: {DEVICE}")
    print(f"[INFO] Output dir: {OUTPUT_DIR}")

    df = pd.read_csv(INDEX_CSV)
    df["finger_id"] = df["finger_id"].astype(str)

    results = []

    for model_name, spec in MODELS.items():
        ckpt_path = spec["ckpt"]
        if not ckpt_path.exists():
            print(f"[WARN] missing checkpoint: {ckpt_path}")
            continue

        print(f"[RUN] {model_name}")
        ckpt = torch.load(ckpt_path, map_location="cpu")
        args = ckpt_args_to_dict(ckpt.get("args", {}))
        label_map = {str(k): int(v) for k, v in ckpt["label_map"].items()}
        model = build_model_from_ckpt(ckpt)

        params = count_params(model)
        ckpt_mb = sizeof_mb(ckpt_path)

        inf_df = select_inf_df(df)
        inf_ds = FingerprintImageDataset(inf_df)
        inf_images = [inf_ds[i] for i in range(len(inf_ds))]
        inf_stats = benchmark_inference(model, inf_images)

        train_df = filter_train_df(df, args)
        batch_size = int(args.get("batch", 128))
        workers = int(args.get("workers", 8))
        epochs = int(args.get("epochs", 12 if model_name == "mix_triplet" else 30))
        lr = float(args.get("lr", 5e-5 if model_name == "mix_triplet" else 1e-4))

        train_loader = DataLoader(
            FingerprintLabelDataset(train_df, label_map),
            batch_size=batch_size,
            shuffle=True,
            num_workers=min(workers, 8),
            pin_memory=(DEVICE.type == "cuda"),
        )

        train_stats = benchmark_training(
            model,
            train_loader,
            training_mode=spec["training_mode"],
            epochs=epochs,
            lr=lr,
            batch_size=batch_size,
        )

        results.append({
            "model": model_name,
            "params_million": params / 1e6,
            "ckpt_size_mb": ckpt_mb,
            "inference_ms_per_image_bs1": inf_stats["latency_ms_per_image_bs1"],
            "throughput_img_per_s_bs32": inf_stats["throughput_img_per_s_bs32"],
            "avg_train_step_s": train_stats["avg_train_step_s"],
            "steps_per_epoch": train_stats["steps_per_epoch"],
            "epoch_time_min_est": train_stats["epoch_time_min_est"],
            "total_train_time_h_est": train_stats["total_train_time_h_est"],
            "batch_size": batch_size,
            "epochs": epochs,
            "notes": spec["notes"],
            "device": str(DEVICE),
            "amp": USE_AMP,
        })

        del model
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    if not results:
        raise RuntimeError("No results collected.")

    out_df = pd.DataFrame(results)
    out_df.to_csv(OUTPUT_DIR / "computational_cost.csv", index=False)

    latex_lines = []
    for _, r in out_df.iterrows():
        latex_lines.append(
            f"{r['model']} & "
            f"{r['params_million']:.2f}M / {r['ckpt_size_mb']:.1f} MB & "
            f"{r['inference_ms_per_image_bs1']:.2f} ms & "
            f"{r['notes']} \\\\"
        )
    (OUTPUT_DIR / "computational_cost_table_rows.txt").write_text("\n".join(latex_lines))

    summary_lines = []
    summary_lines.append("=== Computational Cost Summary ===")
    summary_lines.append(f"Device: {DEVICE}")
    summary_lines.append(f"AMP: {USE_AMP}")
    summary_lines.append(f"Inference benchmark: batch size 1, {N_INF_IMAGES} preloaded test images, warmup={INF_WARMUP}, repeats={INF_REPEATS}")
    summary_lines.append(f"Training benchmark: {TRAIN_MEASURE_STEPS} measured steps after {TRAIN_WARMUP_STEPS} warmup steps")
    summary_lines.append("")
    for _, r in out_df.iterrows():
        summary_lines.append(f"[{r['model']}]")
        summary_lines.append(f"  params           : {r['params_million']:.2f} M")
        summary_lines.append(f"  checkpoint size  : {r['ckpt_size_mb']:.2f} MB")
        summary_lines.append(f"  infer latency    : {r['inference_ms_per_image_bs1']:.2f} ms / image (bs=1)")
        summary_lines.append(f"  infer throughput : {r['throughput_img_per_s_bs32']:.2f} img/s (bs=32)")
        summary_lines.append(f"  train step       : {r['avg_train_step_s']:.4f} s / step")
        summary_lines.append(f"  steps / epoch    : {int(r['steps_per_epoch'])}")
        summary_lines.append(f"  epoch time est   : {r['epoch_time_min_est']:.2f} min")
        summary_lines.append(f"  total train est  : {r['total_train_time_h_est']:.2f} h for {int(r['epochs'])} epochs")
        summary_lines.append(f"  notes            : {r['notes']}")
        summary_lines.append("")
    (OUTPUT_DIR / "computational_cost_summary.txt").write_text("\n".join(summary_lines))

    json.dump(results, open(OUTPUT_DIR / "computational_cost.json", "w"), indent=2)

    para = []
    para.append("The computational cost analysis was conducted on Ubuntu using an NVIDIA GeForce RTX 5060 GPU with 12 GB memory. ")
    para.append("For inference, latency was measured with batch size 1 on preloaded test images, while throughput was additionally measured with batch size 32. ")
    para.append("For training, the reported cost corresponds to an estimated epoch time obtained from measured training-step latency on the training split, using the original batch size and number of epochs of each model configuration. ")
    for _, r in out_df.iterrows():
        para.append(
            f"The {r['model']} model contains {r['params_million']:.2f}M parameters, "
            f"occupies {r['ckpt_size_mb']:.1f} MB on disk, and requires {r['inference_ms_per_image_bs1']:.2f} ms per image at inference. "
            f"Its estimated training time is {r['epoch_time_min_est']:.2f} minutes per epoch, corresponding to approximately {r['total_train_time_h_est']:.2f} hours for the full training schedule."
        )
    (OUTPUT_DIR / "computational_cost_paragraph.txt").write_text(" ".join(para))

    print(f"[OK] wrote {OUTPUT_DIR / 'computational_cost.csv'}")
    print(f"[OK] wrote {OUTPUT_DIR / 'computational_cost_summary.txt'}")
    print(f"[OK] wrote {OUTPUT_DIR / 'computational_cost_table_rows.txt'}")
    print(f"[OK] wrote {OUTPUT_DIR / 'computational_cost_paragraph.txt'}")


if __name__ == "__main__":
    main()
