"""Consistent chart styling for every figure in the project."""
import matplotlib.pyplot as plt

PRIMARY = "#512A7A"
PALETTE = ["#512A7A", "#2A9D8F", "#E9A23B", "#D1495B", "#4A7FC1", "#8C8C8C"]


def apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 110,
            "savefig.dpi": 160,
            "savefig.bbox": "tight",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.titleweight": "bold",
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "font.family": "DejaVu Sans",
            "axes.prop_cycle": plt.cycler(color=PALETTE),
        }
    )
