"""Optional matplotlib charts (install with: pip install 'statarb[plot]')."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import pandas as pd

from .metrics import drawdown_series


def plot_equity(returns: Mapping[str, pd.Series], path: str | Path | None = None, title: str = ""):
    """Compounded equity curves (top) and drawdowns (bottom)."""
    try:
        import matplotlib

        if path is not None:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("matplotlib is not installed: pip install 'statarb[plot]'") from exc

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(12, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )
    for name, r in returns.items():
        r = r.dropna()
        ax1.plot((1 + r).cumprod(), label=name)
        ax2.plot(drawdown_series(r) * 100, label=name)
    ax1.set_ylabel("Growth of $1")
    ax1.set_yscale("log")
    ax1.legend()
    ax1.set_title(title)
    ax2.set_ylabel("Drawdown (%)")
    ax2.axhline(0, color="k", lw=0.5)
    fig.tight_layout()
    if path is not None:
        fig.savefig(path, dpi=120)
        plt.close(fig)
    return fig
