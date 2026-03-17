from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

# ===== Paths =====
DATA_ROOT = Path("/home/congpx/fingerprint/data/SOCOFing/SOCOFing")
CSV_PATH = Path("/mnt/data/consensus_hard_failures.csv")  # đổi nếu cần
OUT_PNG = Path("/home/congpx/fingerprint/figure2_cr_hard_examples.png")
OUT_PDF = Path("/home/congpx/fingerprint/figure2_cr_hard_examples.pdf")

# ===== Selected representative cases =====
SELECTED = [
    ("58__M_Left_ring_finger_CR.BMP", "Near-miss"),
    ("171__M_Left_middle_finger_CR.BMP", "Near-miss"),
    ("404__M_Left_middle_finger_CR.BMP", "Catastrophic mismatch"),
    ("484__M_Left_middle_finger_CR.BMP", "Catastrophic mismatch"),
]

df = pd.read_csv(CSV_PATH)

def basename(x: str) -> str:
    return Path(x).name

df["probe_name"] = df["probe_relpath"].apply(basename)

rows = []
for probe_name, case_type in SELECTED:
    sub = df[df["probe_name"] == probe_name]
    if len(sub) == 0:
        raise FileNotFoundError(f"Không tìm thấy case {probe_name} trong CSV")
    row = sub.iloc[0].copy()
    row["case_type"] = case_type
    rows.append(row)

# ===== Plot =====
n_rows = len(rows)
fig, axes = plt.subplots(n_rows, 3, figsize=(10.2, 11.5))
fig.patch.set_facecolor("white")

col_titles = ["Probe (CR-hard)", "Genuine gallery match", "Top-1 false match"]
for j, t in enumerate(col_titles):
    axes[0, j].set_title(t, fontsize=13, fontweight="bold", pad=10)

for i, row in enumerate(rows):
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

    probe_id = row["probe_name"].replace(".BMP", "")
    subtitle = (
        f"{row['case_type']} | probe: {probe_id}\n"
        f"fail_count={int(row['fail_count'])}, "
        f"mean rank_true={row['mean_rank_true']:.1f}"
    )

    # text at left side of the row
    axes[i, 0].text(
        -0.02, 1.08,
        subtitle,
        transform=axes[i, 0].transAxes,
        ha="left", va="bottom",
        fontsize=10.5,
        fontweight="bold"
    )

    # small labels below images
    axes[i, 0].text(0.5, -0.08, "altered probe", transform=axes[i, 0].transAxes,
                    ha="center", va="top", fontsize=9)
    axes[i, 1].text(0.5, -0.08, "correct match", transform=axes[i, 1].transAxes,
                    ha="center", va="top", fontsize=9, color="green")
    axes[i, 2].text(0.5, -0.08, "false top-1", transform=axes[i, 2].transAxes,
                    ha="center", va="top", fontsize=9, color="firebrick")

plt.tight_layout(h_pad=2.0, w_pad=1.6)
plt.savefig(OUT_PNG, dpi=300, bbox_inches="tight", pad_inches=0.02)
plt.savefig(OUT_PDF, bbox_inches="tight", pad_inches=0.02)
print(f"[OK] wrote {OUT_PNG}")
print(f"[OK] wrote {OUT_PDF}")
plt.show()