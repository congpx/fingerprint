from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from PIL import Image

# =========================
# Paths
# =========================
PROJECT_ROOT = Path("/home/congpx/fingerprint")
DATA_ROOT = PROJECT_ROOT / "data/SOCOFing/SOCOFing"
CSV_PATH = PROJECT_ROOT / "outputs_phase6_error_analysis" / "consensus_hard_failures.csv"

OUT_PNG = PROJECT_ROOT / "figure2_cr_hard_examples_v2.png"
OUT_PDF = PROJECT_ROOT / "figure2_cr_hard_examples_v2.pdf"

# =========================
# Load CSV
# =========================
df = pd.read_csv(CSV_PATH)

required_cols = [
    "probe_relpath",
    "fail_count",
    "mean_rank_true",
    "example_genuine_relpath",
    "example_top1_relpath",
]
for c in required_cols:
    if c not in df.columns:
        raise ValueError(f"Missing required column: {c}")

if len(df) == 0:
    raise ValueError("consensus_hard_failures.csv is empty.")

# =========================
# Select representative cases
# =========================
# Near-miss: low mean_rank_true but repeated across seeds
near_df = df[df["fail_count"] >= 2].sort_values(
    ["mean_rank_true", "fail_count"], ascending=[True, False]
)

# Catastrophic mismatch: high mean_rank_true, repeated across seeds
cata_df = df[df["fail_count"] >= 2].sort_values(
    ["mean_rank_true", "fail_count"], ascending=[False, False]
)

selected = []
used = set()

# 2 near-miss
for _, r in near_df.iterrows():
    if r["probe_relpath"] in used:
        continue
    selected.append((r, "Near-miss"))
    used.add(r["probe_relpath"])
    if len([x for x in selected if x[1] == "Near-miss"]) == 2:
        break

# 2 catastrophic mismatch
for _, r in cata_df.iterrows():
    if r["probe_relpath"] in used:
        continue
    selected.append((r, "Catastrophic mismatch"))
    used.add(r["probe_relpath"])
    if len([x for x in selected if x[1] == "Catastrophic mismatch"]) == 2:
        break

if len(selected) < 4:
    # fallback: fill remaining from consensus list
    for _, r in df.sort_values(["fail_count", "mean_rank_true"], ascending=[False, False]).iterrows():
        if r["probe_relpath"] in used:
            continue
        tag = "Catastrophic mismatch" if float(r["mean_rank_true"]) > 20 else "Near-miss"
        selected.append((r, tag))
        used.add(r["probe_relpath"])
        if len(selected) == 4:
            break

# Keep order: near-miss first, catastrophic second
selected = sorted(selected, key=lambda x: (0 if x[1] == "Near-miss" else 1, x[0]["mean_rank_true"]))

# =========================
# Figure layout
# =========================
n_rows = len(selected)

fig = plt.figure(figsize=(10.8, 2.45 * n_rows + 1.0), facecolor="white")
gs = GridSpec(
    nrows=n_rows + 1,
    ncols=4,
    figure=fig,
    height_ratios=[0.28] + [1.0] * n_rows,
    width_ratios=[1.7, 1.0, 1.0, 1.0],
    hspace=0.55,
    wspace=0.18,
)

# =========================
# Column headers
# =========================
headers = ["", "Probe (CR-hard)", "Genuine gallery match", "Top-1 false match"]
for j in range(4):
    ax = fig.add_subplot(gs[0, j])
    ax.axis("off")
    if headers[j]:
        ax.text(
            0.5, 0.25, headers[j],
            ha="center", va="center",
            fontsize=11, fontweight="bold"
        )

# =========================
# Helper
# =========================
def open_gray(path: Path):
    return Image.open(path).convert("L")

def shorten_probe_name(relpath: str) -> str:
    name = Path(relpath).stem
    return name.replace("_CR", "")

# =========================
# Render rows
# =========================
for i, (row, case_type) in enumerate(selected, start=1):
    probe_path = DATA_ROOT / row["probe_relpath"]
    genuine_path = DATA_ROOT / row["example_genuine_relpath"]
    false_path = DATA_ROOT / row["example_top1_relpath"]

    probe_img = open_gray(probe_path)
    genuine_img = open_gray(genuine_path)
    false_img = open_gray(false_path)

    # Left annotation cell
    ax_info = fig.add_subplot(gs[i, 0])
    ax_info.axis("off")

    probe_name = shorten_probe_name(row["probe_relpath"])
    fail_count = int(row["fail_count"])
    mean_rank = float(row["mean_rank_true"])

    if case_type == "Near-miss":
        color = "#1f4e79"
        title = f"Near-miss case"
        desc = (
            f"Probe: {probe_name}\n"
            f"fail_count = {fail_count}\n"
            f"mean rank_true = {mean_rank:.1f}\n"
            f"Genuine remains close in rank\nbut is not retrieved at top-1."
        )
    else:
        color = "#7f0000"
        title = f"Catastrophic mismatch"
        desc = (
            f"Probe: {probe_name}\n"
            f"fail_count = {fail_count}\n"
            f"mean rank_true = {mean_rank:.1f}\n"
            f"Severe corruption pushes the\ntrue match far down the ranking."
        )

    ax_info.text(
        0.02, 0.85, title,
        ha="left", va="top",
        fontsize=12.2, fontweight="bold", color=color
    )
    ax_info.text(
        0.02, 0.55, desc,
        ha="left", va="top",
        fontsize=10.1, linespacing=1.35
    )

    # Images
    imgs = [probe_img, genuine_img, false_img]
    labels = ["Probe", "Genuine", "False Top-1"]
    label_colors = ["black", "green", "firebrick"]

    for j, (img, lab, lab_color) in enumerate(zip(imgs, labels, label_colors), start=1):
        ax = fig.add_subplot(gs[i, j])
        ax.imshow(img, cmap="gray", vmin=0, vmax=255)
        ax.set_xticks([])
        ax.set_yticks([])

        # nicer border
        for spine in ax.spines.values():
            spine.set_linewidth(1.4)
            spine.set_edgecolor("black")

        ax.text(
            0.5, -0.08, lab,
            transform=ax.transAxes,
            ha="center", va="top",
            fontsize=9.5,
            color=lab_color,
            fontweight="bold" if j > 1 else None
        )

# =========================
# Save
# =========================
plt.tight_layout(pad=0.3)
plt.savefig(OUT_PNG, dpi=300, bbox_inches="tight", pad_inches=0.02)
plt.savefig(OUT_PDF, bbox_inches="tight", pad_inches=0.02)

print(f"[OK] wrote {OUT_PNG}")
print(f"[OK] wrote {OUT_PDF}")
plt.show()