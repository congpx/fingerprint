import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch, Polygon


def add_down_block_arrow(ax, x_center, y_top, y_tip,
                         shaft_w=0.36, head_w=1.45, head_h=0.9,
                         facecolor="#FFD966", edgecolor="black",
                         alpha=0.20, linestyle="--", lw=1.5, zorder=1):
    y_head_base = y_tip + head_h
    pts = [
        (x_center - shaft_w / 2, y_top),
        (x_center + shaft_w / 2, y_top),
        (x_center + shaft_w / 2, y_head_base),
        (x_center + head_w / 2, y_head_base),
        (x_center, y_tip),
        (x_center - head_w / 2, y_head_base),
        (x_center - shaft_w / 2, y_head_base),
    ]
    arrow = Polygon(
        pts,
        closed=True,
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=lw,
        linestyle=linestyle,
        alpha=alpha,
        zorder=zorder
    )
    ax.add_patch(arrow)
    return arrow


def add_box(ax, center, width, height, text,
            facecolor="#f8f9fa", edgecolor="black",
            fontsize=11, lw=1.4, linestyle="-", zorder=2):
    x = center[0] - width / 2
    y = center[1] - height / 2
    rect = Rectangle(
        (x, y), width, height,
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=lw,
        linestyle=linestyle,
        zorder=zorder
    )
    ax.add_patch(rect)
    ax.text(
        center[0], center[1], text,
        ha="center", va="center",
        fontsize=fontsize,
        family="sans-serif",
        zorder=zorder + 1
    )
    return rect


def add_two_line_box(ax, center, width, height,
                     line1, line2,
                     facecolor="#f8f9fa", edgecolor="black",
                     fontsize1=11, fontsize2=8,
                     italic2=True, lw=1.4, linestyle="-", zorder=2):
    x = center[0] - width / 2
    y = center[1] - height / 2
    rect = Rectangle(
        (x, y), width, height,
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=lw,
        linestyle=linestyle,
        zorder=zorder
    )
    ax.add_patch(rect)

    ax.text(
        center[0], center[1] + 0.18,
        line1,
        ha="center", va="center",
        fontsize=fontsize1,
        family="sans-serif",
        zorder=zorder + 1
    )

    ax.text(
        center[0], center[1] - 0.18,
        line2,
        ha="center", va="center",
        fontsize=fontsize2,
        style="italic" if italic2 else "normal",
        family="sans-serif",
        zorder=zorder + 1
    )
    return rect


def box_anchor(center, width, height, side):
    x, y = center
    if side == "top":
        return (x, y + height / 2)
    if side == "bottom":
        return (x, y - height / 2)
    if side == "left":
        return (x - width / 2, y)
    if side == "right":
        return (x + width / 2, y)
    raise ValueError("side must be top/bottom/left/right")


def add_arrow(ax, p1, p2, style="-", lw=1.4, color="black",
              connectionstyle="arc3,rad=0.0",
              arrowstyle="-|>", mutation_scale=18, zorder=3):
    arr = FancyArrowPatch(
        p1, p2,
        arrowstyle=arrowstyle,
        mutation_scale=mutation_scale,
        linewidth=lw,
        linestyle=style,
        color=color,
        connectionstyle=connectionstyle,
        zorder=zorder
    )
    ax.add_patch(arr)
    return arr


def add_elbow_arrow(ax, p1, p2, mid_x=None, mid_y=None,
                    style="-", lw=1.4, color="black",
                    arrowstyle="-|>", mutation_scale=18, zorder=3):
    x1, y1 = p1
    x2, y2 = p2

    if mid_x is not None:
        ax.plot([x1, mid_x], [y1, y1], linestyle=style, linewidth=lw, color=color, zorder=zorder)
        ax.plot([mid_x, mid_x], [y1, y2], linestyle=style, linewidth=lw, color=color, zorder=zorder)
        add_arrow(
            ax,
            (mid_x, y2),
            p2,
            style=style,
            lw=lw,
            color=color,
            connectionstyle="arc3,rad=0.0",
            arrowstyle=arrowstyle,
            mutation_scale=mutation_scale,
            zorder=zorder
        )
    elif mid_y is not None:
        ax.plot([x1, x1], [y1, mid_y], linestyle=style, linewidth=lw, color=color, zorder=zorder)
        ax.plot([x1, x2], [mid_y, mid_y], linestyle=style, linewidth=lw, color=color, zorder=zorder)
        add_arrow(
            ax,
            (x2, mid_y),
            p2,
            style=style,
            lw=lw,
            color=color,
            connectionstyle="arc3,rad=0.0",
            arrowstyle=arrowstyle,
            mutation_scale=mutation_scale,
            zorder=zorder
        )
    else:
        raise ValueError("Either mid_x or mid_y must be provided.")


def main():
    fig, ax = plt.subplots(figsize=(10, 9.2))
    ax.set_xlim(0, 14)
    ax.set_ylim(2.15, 18)
    ax.axis("off")

    # ===== Colors =====
    c_input = "#F2F2F2"
    c_pre = "#EDEDED"
    c_backbone = "#D9EAF7"
    c_mix = "#FFF2CC"
    c_embed = "#D9EAD3"
    c_gallery = "#FCE5CD"
    c_probe = "#F4CCCC"
    c_match = "#D9D2E9"
    c_output = "#E8F1FB"

    c_ckpt = "#FFE599"
    c_sampler = "#F9CB9C"
    c_triplet = "#F4CCCC"
    c_loss = "#CFE2F3"
    c_update = "#D9EAD3"

    container_face = "#FBFBFB"
    container_edge = "#666666"

    # ===== Containers =====
    top_container = Rectangle(
        (1.0, 8.0), 12.0, 8.8,
        facecolor=container_face,
        edgecolor=container_edge,
        linewidth=1.5,
        linestyle="-",
        zorder=0
    )
    ax.add_patch(top_container)
    ax.text(1.2, 16.45, "Final Retrieval Model", fontsize=13, fontweight="bold", va="center")

    bottom_container = Rectangle(
        (1.0, 2.25), 12.0, 4.55,
        facecolor=container_face,
        edgecolor=container_edge,
        linewidth=1.5,
        linestyle="--",
        zorder=0
    )
    ax.add_patch(bottom_container)
    ax.text(1.2, 6.32, "Training-Time Metric Fine-Tuning",
            fontsize=13, fontweight="bold", va="center")

    ax.text(5.9, 16.85, "Solid arrows: inference + training path", fontsize=10)
    ax.text(5.9, 17.20, "Dashed arrows: training-only path", fontsize=10)

    # ===== Top section =====
    input_c = (7.0, 15.2)
    pre_c = (7.0, 13.8)
    backbone_c = (7.0, 12.3)
    mix_c = (7.0, 10.8)

    gallery_c = (3.0, 9.0)
    probe_c = (7.0, 9.0)
    embed_c = (11.0, 9.0)

    match_c = (7.0, 7.55)

    w_big, h_big = 4.8, 0.95
    w_mid, h_mid = 3.5, 0.95
    w_small, h_small = 2.6, 0.95

    # tăng ngang cho retrieval / verification
    w_match, h_match = 5.8, 0.90

    add_box(ax, input_c, w_big, h_big,
            "Input Fingerprint\n(real or altered)",
            facecolor=c_input, fontsize=12)

    add_box(ax, pre_c, w_big, h_big,
            "Preprocessing\nresize 224×224 + normalization",
            facecolor=c_pre, fontsize=12)

    add_box(ax, backbone_c, w_big, h_big,
            "Stage 1: ResNet18 Backbone\nCross-Entropy Training",
            facecolor=c_backbone, fontsize=12)

    add_box(ax, mix_c, w_big, h_big,
            "Stage 2: MixStyle at layer1\np = 0.7, α = 0.3",
            facecolor=c_mix, fontsize=12, lw=1.8)

    add_box(ax, gallery_c, w_mid, h_mid,
            "Gallery Branch\nreal fingerprints",
            facecolor=c_gallery, fontsize=11)

    add_box(ax, probe_c, w_mid, h_mid,
            "Probe Branch\nunseen CR / Zcut",
            facecolor=c_probe, fontsize=11)

    add_box(ax, embed_c, w_mid, h_mid,
            "Embedding Head\n256-D + L2 normalization",
            facecolor=c_embed, fontsize=11)

    add_box(ax, match_c, w_match, h_match,
            "Retrieval / Verification\ncosine similarity → Rank-1 / EER / TAR@FAR",
            facecolor=c_match, fontsize=11.2)

    # Top arrows
    add_arrow(ax, box_anchor(input_c, w_big, h_big, "bottom"), box_anchor(pre_c, w_big, h_big, "top"))
    add_arrow(ax, box_anchor(pre_c, w_big, h_big, "bottom"), box_anchor(backbone_c, w_big, h_big, "top"))
    add_arrow(ax, box_anchor(backbone_c, w_big, h_big, "bottom"), box_anchor(mix_c, w_big, h_big, "top"))

    add_arrow(ax, box_anchor(mix_c, w_big, h_big, "bottom"), box_anchor(gallery_c, w_mid, h_mid, "top"))
    add_arrow(ax, box_anchor(mix_c, w_big, h_big, "bottom"), box_anchor(probe_c, w_mid, h_mid, "top"))
    add_arrow(ax, box_anchor(mix_c, w_big, h_big, "bottom"), box_anchor(embed_c, w_mid, h_mid, "top"))

    add_arrow(ax, box_anchor(gallery_c, w_mid, h_mid, "bottom"), box_anchor(match_c, w_match, h_match, "top"))
    add_arrow(ax, box_anchor(probe_c, w_mid, h_mid, "bottom"), box_anchor(match_c, w_match, h_match, "top"))
    add_arrow(ax, box_anchor(embed_c, w_mid, h_mid, "bottom"), box_anchor(match_c, w_match, h_match, "top"))

    # ===== Bottom section =====
    best_ckpt_c = (2.6, 5.15)
    sampler_c = (2.6, 3.55)
    triplet_c = (6.6, 5.15)
    loss_c = (6.6, 3.55)
    update_c = (10.8, 4.35)

    # giảm ngang, tăng cao
    w_train_small, h_train_small = 3.0, 1.05
    w_train_mid, h_train_mid = 4.0, 1.05

    add_box(ax, best_ckpt_c, w_train_small, h_train_small,
            "Best MixStyle Model\n(selected from validation)",
            facecolor=c_ckpt, fontsize=10.8)

    add_two_line_box(ax, sampler_c, w_train_small, h_train_small,
                     "PK Sampler", "P = 32 identities, K = 4 samples",
                     facecolor=c_sampler, fontsize1=10.8, fontsize2=8, italic2=True)

    add_box(ax, triplet_c, w_train_mid, h_train_mid,
            "Batch-Hard Triplet Mining\nhardest positive / hardest negative",
            facecolor=c_triplet, fontsize=10.8)

    add_box(ax, loss_c, w_train_mid, h_train_mid,
            "Stage 3: Fine-Tuning Loss\nCross-Entropy + λ·Triplet\nλ = 0.2, margin = 0.2",
            facecolor=c_loss, fontsize=10.6)

    add_box(ax, update_c, w_train_small, h_train_small,
            "Updated Final Model\nMixStyle + Triplet",
            facecolor=c_update, fontsize=10.8)

    # arrows bottom
    add_arrow(ax, box_anchor(best_ckpt_c, w_train_small, h_train_small, "bottom"),
              box_anchor(sampler_c, w_train_small, h_train_small, "top"))

    add_arrow(ax, box_anchor(best_ckpt_c, w_train_small, h_train_small, "right"),
              box_anchor(triplet_c, w_train_mid, h_train_mid, "left"))

    add_arrow(ax, box_anchor(sampler_c, w_train_small, h_train_small, "right"),
              box_anchor(loss_c, w_train_mid, h_train_mid, "left"))

    add_arrow(ax, box_anchor(triplet_c, w_train_mid, h_train_mid, "bottom"),
              box_anchor(loss_c, w_train_mid, h_train_mid, "top"))

    add_arrow(ax, box_anchor(triplet_c, w_train_mid, h_train_mid, "right"),
              box_anchor(update_c, w_train_small, h_train_small, "left"))

    add_arrow(ax, box_anchor(loss_c, w_train_mid, h_train_mid, "right"),
              box_anchor(update_c, w_train_small, h_train_small, "left"))

    # training-only dashed block arrow from MixStyle to fine-tuning container
    mix_bottom = box_anchor(mix_c, w_big, h_big, "bottom")
    triplet_top = box_anchor(triplet_c, w_train_mid, h_train_mid, "top")

    add_down_block_arrow(
        ax,
        x_center=mix_c[0],
        y_top=mix_bottom[1],
        y_tip=triplet_top[1],
        shaft_w=1.2,
        head_w=2.4,
        head_h=0.7,
        facecolor="#FFD966",
        edgecolor="black",
        alpha=0.30,
        linestyle="--",
        lw=1.5,
        zorder=1
    )

    ax.text(
        mix_c[0] + 1.25,
        triplet_top[1] + 0.18,
        "training-only path",
        fontsize=9,
        style="italic",
        color="black",
        va="bottom",
        ha="left"
    )

   # dashed feedback arrow: straight up, then left, with explicit visible arrow head
    update_top = box_anchor(update_c, w_train_small, h_train_small, "top")
    match_right = box_anchor(match_c, w_match, h_match, "right")

    mid_y_feedback = 7.55
    x_vert = update_top[0]
    x_end = match_right[0] + 0.18   # stop slightly outside the box, then draw arrow into it

    # vertical dashed segment
    ax.plot(
        [x_vert, x_vert],
        [update_top[1], mid_y_feedback],
        linestyle="--",
        linewidth=1.4,
        color="black",
        zorder=3
    )

    # horizontal dashed segment
    ax.plot(
        [x_vert, x_end],
        [mid_y_feedback, mid_y_feedback],
        linestyle="--",
        linewidth=1.4,
        color="black",
        zorder=3
    )

    # explicit arrow head at the end
    add_arrow(
        ax,
        (x_end, mid_y_feedback),
        match_right,
        style="--",
        lw=1.4,
        color="black",
        connectionstyle="arc3,rad=0.0",
        arrowstyle="-|>",
        mutation_scale=18,
        zorder=4
    )

    # label
    ax.text(
        x_vert + 0.12,
        (update_top[1] + mid_y_feedback) / 2,
        "final inference model",
        fontsize=9,
        rotation=90,
        alpha=0.9,
        va="center",
        ha="left"
    )

    ax.set_title(
        "Overall Pipeline of the Proposed Altered Fingerprint Retrieval Framework",
        fontsize=15, fontweight="bold", pad=10
    )

    plt.tight_layout()
    plt.savefig("pipeline.png", dpi=300, bbox_inches="tight", pad_inches=0.03)
    plt.savefig("pipeline.pdf", bbox_inches="tight", pad_inches=0.03)
    plt.show()


if __name__ == "__main__":
    main()