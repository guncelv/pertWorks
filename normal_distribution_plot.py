"""Reusable plotter for normal-distribution graphs.

This module provides a small, focused API for drawing Gaussian bell curves
with three common customizations:

1. Fully configurable horizontal-axis labels (raw values, z-scores, sigma
   units, percentiles, or any custom mapping).
2. Vertical lines at any value(s) the user chooses.
3. Hatched shaded regions under the curve to visualize probabilities.

Run this file directly to render a small gallery of example plots.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm


@dataclass
class VerticalLine:
    """A vertical line to draw on top of the distribution."""

    x: float
    label: str | None = None
    color: str = "black"
    linestyle: str = "--"
    linewidth: float = 1.5
    alpha: float = 0.9


@dataclass
class HatchedRegion:
    """A shaded/hatched area under the curve between two x-values.

    Either bound may be ``None`` / ``+-inf`` to hatch an open tail.
    The probability mass of the region is computed automatically and can be
    shown in the legend via ``show_probability=True``.
    """

    lower: float | None = None
    upper: float | None = None
    label: str | None = None
    hatch: str = "///"
    edgecolor: str = "C0"
    facecolor: str = "none"
    alpha: float = 0.6
    show_probability: bool = True


class NormalDistributionPlot:
    """Build and render a customizable normal distribution chart.

    Parameters
    ----------
    mean, std:
        Parameters of the normal distribution (``std`` must be positive).
    x_range:
        Optional ``(xmin, xmax)`` window. If omitted, the window defaults to
        ``mean +/- 4*std``.
    num_points:
        Number of samples used to draw the curve.
    """

    def __init__(
        self,
        mean: float = 0.0,
        std: float = 1.0,
        x_range: tuple[float, float] | None = None,
        num_points: int = 1000,
    ) -> None:
        if std <= 0:
            raise ValueError("std must be strictly positive")
        self.mean = float(mean)
        self.std = float(std)
        self.num_points = int(num_points)
        if x_range is None:
            self.x_range = (mean - 4 * std, mean + 4 * std)
        else:
            if x_range[0] >= x_range[1]:
                raise ValueError("x_range must satisfy xmin < xmax")
            self.x_range = (float(x_range[0]), float(x_range[1]))

        self._vlines: list[VerticalLine] = []
        self._regions: list[HatchedRegion] = []

        self._xtick_positions: Sequence[float] | None = None
        self._xtick_labels: Sequence[str] | None = None

    def add_vertical_line(
        self,
        x: float,
        label: str | None = None,
        color: str = "black",
        linestyle: str = "--",
        linewidth: float = 1.5,
        alpha: float = 0.9,
    ) -> "NormalDistributionPlot":
        """Add a vertical line at ``x``. Returns ``self`` for chaining."""
        self._vlines.append(
            VerticalLine(
                x=float(x),
                label=label,
                color=color,
                linestyle=linestyle,
                linewidth=linewidth,
                alpha=alpha,
            )
        )
        return self

    def add_hatched_region(
        self,
        lower: float | None = None,
        upper: float | None = None,
        label: str | None = None,
        hatch: str = "///",
        edgecolor: str = "C0",
        facecolor: str = "none",
        alpha: float = 0.6,
        show_probability: bool = True,
    ) -> "NormalDistributionPlot":
        """Shade the area under the curve between ``lower`` and ``upper``.

        ``None`` (or +-inf) means "open tail"; e.g. ``lower=None, upper=1.96``
        shades ``(-inf, 1.96]``. Returns ``self`` for chaining.
        """
        self._regions.append(
            HatchedRegion(
                lower=lower,
                upper=upper,
                label=label,
                hatch=hatch,
                edgecolor=edgecolor,
                facecolor=facecolor,
                alpha=alpha,
                show_probability=show_probability,
            )
        )
        return self

    def set_xticks(
        self,
        positions: Sequence[float],
        labels: Sequence[str] | None = None,
    ) -> "NormalDistributionPlot":
        """Set explicit tick positions (and optional labels) on the x axis."""
        positions = list(positions)
        if labels is not None:
            labels = list(labels)
            if len(labels) != len(positions):
                raise ValueError("labels must have the same length as positions")
        self._xtick_positions = positions
        self._xtick_labels = labels
        return self

    def set_xticks_by_sigma(
        self,
        sigma_values: Iterable[float] = (-3, -2, -1, 0, 1, 2, 3),
        as_sigma_labels: bool = False,
    ) -> "NormalDistributionPlot":
        """Place ticks at multiples of ``std`` away from the mean.

        If ``as_sigma_labels`` is True, labels look like ``-2σ``, ``μ``, ``+σ``;
        otherwise the numerical x values are shown.
        """
        sigma_values = list(sigma_values)
        positions = [self.mean + k * self.std for k in sigma_values]
        if as_sigma_labels:
            labels = []
            for k in sigma_values:
                if k == 0:
                    labels.append("μ")
                elif k == 1:
                    labels.append("μ+σ")
                elif k == -1:
                    labels.append("μ−σ")
                else:
                    sign = "+" if k > 0 else "−"
                    labels.append(f"μ{sign}{abs(k):g}σ")
            return self.set_xticks(positions, labels)
        return self.set_xticks(positions)

    def set_xticks_by_mapping(
        self,
        positions: Sequence[float],
        label_fn: Callable[[float], str],
    ) -> "NormalDistributionPlot":
        """Use any callable to transform each tick position into a label.

        Example: show cumulative probabilities at each tick::

            plot.set_xticks_by_mapping(
                [-2, -1, 0, 1, 2],
                lambda x: f"{norm.cdf(x):.2f}",
            )
        """
        labels = [label_fn(p) for p in positions]
        return self.set_xticks(positions, labels)

    def _pdf(self, x: np.ndarray) -> np.ndarray:
        return norm.pdf(x, loc=self.mean, scale=self.std)

    def _probability(self, lower: float | None, upper: float | None) -> float:
        lo = -np.inf if lower is None else lower
        hi = np.inf if upper is None else upper
        return float(
            norm.cdf(hi, loc=self.mean, scale=self.std)
            - norm.cdf(lo, loc=self.mean, scale=self.std)
        )

    def plot(
        self,
        ax: plt.Axes | None = None,
        title: str | None = None,
        xlabel: str = "x",
        ylabel: str = "Probability density",
        curve_color: str = "C0",
        curve_linewidth: float = 2.0,
        show_legend: bool = True,
        grid: bool = True,
    ) -> plt.Axes:
        """Render the distribution and return the matplotlib ``Axes``."""
        if ax is None:
            _, ax = plt.subplots(figsize=(9, 5))

        xmin, xmax = self.x_range
        x = np.linspace(xmin, xmax, self.num_points)
        y = self._pdf(x)

        ax.plot(x, y, color=curve_color, linewidth=curve_linewidth, label="N(μ, σ²)")

        for region in self._regions:
            lo = xmin if region.lower is None else max(region.lower, xmin)
            hi = xmax if region.upper is None else min(region.upper, xmax)
            if hi <= lo:
                continue
            mask = (x >= lo) & (x <= hi)
            xr = x[mask]
            yr = y[mask]
            prob = self._probability(region.lower, region.upper)
            label = region.label
            if region.show_probability:
                prob_str = f"P = {prob:.4f}"
                label = f"{label} ({prob_str})" if label else prob_str
            ax.fill_between(
                xr,
                0,
                yr,
                facecolor=region.facecolor,
                edgecolor=region.edgecolor,
                hatch=region.hatch,
                alpha=region.alpha,
                linewidth=0.0,
                label=label,
            )

        for vline in self._vlines:
            ax.axvline(
                vline.x,
                color=vline.color,
                linestyle=vline.linestyle,
                linewidth=vline.linewidth,
                alpha=vline.alpha,
                label=vline.label,
            )

        if self._xtick_positions is not None:
            ax.set_xticks(list(self._xtick_positions))
            if self._xtick_labels is not None:
                ax.set_xticklabels(list(self._xtick_labels))

        ax.set_xlim(xmin, xmax)
        ax.set_ylim(bottom=0)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if title is None:
            title = f"Normal distribution (μ={self.mean:g}, σ={self.std:g})"
        ax.set_title(title)
        if grid:
            ax.grid(True, linestyle=":", alpha=0.5)
        if show_legend:
            ax.legend(loc="best", framealpha=0.9)
        return ax


def _demo() -> None:
    """Render a small gallery showing the main features."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    p1 = NormalDistributionPlot(mean=0, std=1)
    p1.set_xticks_by_sigma(as_sigma_labels=True)
    p1.add_vertical_line(0, label="mean (μ)", color="black")
    p1.add_hatched_region(
        lower=-1, upper=1, label="within 1σ", hatch="///", edgecolor="C1"
    )
    p1.plot(ax=axes[0, 0], title="Standard normal with σ-labels")

    p2 = NormalDistributionPlot(mean=100, std=15, x_range=(40, 160))
    p2.set_xticks([55, 70, 85, 100, 115, 130, 145])
    p2.add_vertical_line(130, label="IQ = 130 (gifted cutoff)", color="crimson")
    p2.add_hatched_region(
        lower=130, upper=None, label="IQ ≥ 130", hatch="\\\\\\", edgecolor="crimson"
    )
    p2.add_hatched_region(
        lower=85, upper=115, label="average range", hatch="...", edgecolor="C2"
    )
    p2.plot(ax=axes[0, 1], title="IQ scores (μ=100, σ=15)", xlabel="IQ score")

    p3 = NormalDistributionPlot(mean=0, std=1)
    p3.set_xticks_by_mapping(
        [-2, -1, 0, 1, 2],
        lambda z: f"z={z:+g}\nCDF={norm.cdf(z):.3f}",
    )
    p3.add_vertical_line(1.96, label="z = 1.96", color="purple")
    p3.add_vertical_line(-1.96, color="purple", linestyle=":")
    p3.add_hatched_region(
        lower=-1.96, upper=1.96, label="95% CI", hatch="xx", edgecolor="purple"
    )
    p3.plot(ax=axes[1, 0], title="Custom tick labels via a mapping function")

    p4 = NormalDistributionPlot(mean=50, std=8, x_range=(20, 80))
    p4.set_xticks([26, 34, 42, 50, 58, 66, 74])
    p4.add_vertical_line(42, label="lower bound", color="darkorange")
    p4.add_vertical_line(66, label="upper bound", color="darkorange")
    p4.add_hatched_region(lower=None, upper=42, label="left tail", hatch="//", edgecolor="gray")
    p4.add_hatched_region(lower=66, upper=None, label="right tail", hatch="\\\\", edgecolor="gray")
    p4.add_hatched_region(lower=42, upper=66, label="central", hatch="++", edgecolor="C0")
    p4.plot(ax=axes[1, 1], title="Tails and central mass with probabilities")

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    _demo()
