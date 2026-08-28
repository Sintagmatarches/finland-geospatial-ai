"""Rebuild summary figures from committed run histories and dataset manifest."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt


def main() -> None:
    figures = Path("reports/figures")
    figures.mkdir(parents=True, exist_ok=True)
    histories = {
        "E1 U-Net RGB": Path("artifacts/checkpoints/unet-rgb-best.history.json"),
        "E2 U-Net RGB+NIR": Path("artifacts/checkpoints/unet-best.history.json"),
        "E3 TinyDeepLab RGB+NIR": Path("artifacts/checkpoints/deeplab-best.history.json"),
        "E4 U-Net weighted CE": Path("artifacts/checkpoints/unet-ce-best.history.json"),
    }
    fig, axis = plt.subplots(figsize=(8, 5))
    for label, path in histories.items():
        rows = json.loads(path.read_text())
        axis.plot(
            [int(row["epoch"]) for row in rows],
            [row["val_miou"] for row in rows],
            marker="o",
            markersize=3,
            label=label,
        )
    axis.set(xlabel="Epoch", ylabel="Tampere validation mIoU", xticks=range(1, 13))
    axis.set_title("Declared experiment matrix — validation only")
    axis.grid(alpha=0.2)
    axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figures / "training-curves.png", dpi=160)
    plt.close(fig)

    manifest = json.loads(Path("artifacts/dataset-manifest-v1.json").read_text())
    names = [manifest["class_mapping"]["classes"][str(i)]["name"] for i in range(6)]
    fig, axis = plt.subplots(figsize=(9, 4.5))
    x = list(range(6))
    width = 0.25
    for offset, split in enumerate(("train", "val", "test")):
        counts = manifest["class_counts_by_split"][split]
        total = sum(counts.values())
        axis.bar(
            [value + (offset - 1) * width for value in x],
            [100 * counts[str(index)] / total for index in x],
            width=width,
            label=split,
        )
    axis.set_xticks(x, names, rotation=25, ha="right")
    axis.set(ylabel="Valid pixels (%)", title="Class distribution by held-out geography")
    axis.legend()
    axis.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(figures / "class-distribution.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
