#!/usr/bin/env python3
"""Plot configurable normal distribution charts with annotations."""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np


@dataclass(frozen=True)
class VerticalLine:
    value: float
    label: str | None = None
    color: str | None = None


@dataclass(frozen=True)
class HatchedRegion:
    left: float
    right: float
    label: str | None = None


def normal_pdf(x: np.ndarray, mean: float, std: float) -> np.ndarray:
    coefficient = 1.0 / (std * math.sqrt(2.0 * math.pi))
    exponent = -0.5 * ((x - mean) / std) ** 2
    return coefficient * np.exp(exponent)


def normal_cdf(value: float, mean: float, std: float) -> float:
    z = (value - mean) / (std * math.sqrt(2.0))
    return 0.5 * (1.0 + math.erf(z))


def parse_float_list(raw: str) -> list[float]:
    try:
        return [float(item.strip()) for item in raw.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Could not parse comma-separated float list: {raw}"
        ) from exc


def parse_vertical_line(raw: str) -> VerticalLine:
    parts = [part.strip() for part in raw.split(":", maxsplit=2)]
    if not 1 <= len(parts) <= 3:
        raise argparse.ArgumentTypeError(
            "Vertical line format must be value[:label[:color]]."
        )

    try:
        value = float(parts[0])
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Vertical line value must be numeric: {parts[0]}"
        ) from exc

    label = parts[1] if len(parts) >= 2 and parts[1] else None
    color = parts[2] if len(parts) == 3 and parts[2] else None
    return VerticalLine(value=value, label=label, color=color)


def parse_hatched_region(raw: str) -> HatchedRegion:
    parts = [part.strip() for part in raw.split(":", maxsplit=2)]
    if len(parts) < 2:
        raise argparse.ArgumentTypeError(
            "Hatched region format must be left:right[:label]."
        )

    try:
        left = float(parts[0])
        right = float(parts[1])
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Hatched bounds must be numeric in: {raw}"
        ) from exc

    if left > right:
        left, right = right, left

    label = parts[2] if len(parts) == 3 and parts[2] else None
    return HatchedRegion(left=left, right=right, label=label)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create a normal distribution graph with custom x-axis values, "
            "vertical lines, and hatched probability regions."
        )
    )
    parser.add_argument("--mean", type=float, default=0.0, help="Distribution mean.")
    parser.add_argument(
        "--std",
        type=float,
        default=1.0,
        help="Distribution standard deviation (must be > 0).",
    )
    parser.add_argument("--x-min", type=float, default=-4.0, help="Lower x-axis limit.")
    parser.add_argument("--x-max", type=float, default=4.0, help="Upper x-axis limit.")
    parser.add_argument(
        "--points",
        type=int,
        default=1000,
        help="Number of points used to draw the curve.",
    )
    parser.add_argument(
        "--x-values",
        type=parse_float_list,
        default=None,
        help="Comma-separated x-axis tick values, e.g. -3,-2,-1,0,1,2,3.",
    )
    parser.add_argument(
        "--vline",
        action="append",
        default=[],
        type=parse_vertical_line,
        help=(
            "Add a vertical line. Repeat this flag as needed. "
            "Format: value[:label[:color]]"
        ),
    )
    parser.add_argument(
        "--hatch",
        action="append",
        default=[],
        type=parse_hatched_region,
        help=(
            "Add a hatched probability region. Repeat as needed. "
            "Format: left:right[:label]"
        ),
    )
    parser.add_argument(
        "--hatch-pattern",
        type=str,
        default="///",
        help='Matplotlib hatch pattern, e.g. "///", "\\\\\\\\", "xx".',
    )
    parser.add_argument(
        "--hatch-color",
        type=str,
        default="tab:green",
        help="Color for hatched probability areas.",
    )
    parser.add_argument(
        "--hatch-alpha",
        type=float,
        default=0.25,
        help="Transparency for hatched probability areas.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Normal Distribution",
        help="Chart title.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="normal_distribution.png",
        help="Output image path.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the chart interactively in addition to saving.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.std <= 0:
        parser.error("--std must be greater than 0.")
    if args.points < 2:
        parser.error("--points must be at least 2.")
    if args.x_min >= args.x_max:
        parser.error("--x-min must be lower than --x-max.")

    x = np.linspace(args.x_min, args.x_max, args.points)
    y = normal_pdf(x, mean=args.mean, std=args.std)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x, y, color="tab:blue", linewidth=2.0, label="Normal PDF")

    if args.x_values:
        ax.set_xticks(args.x_values)

    for line in args.vline:
        line_kwargs = {"linestyle": "--", "linewidth": 1.75}
        if line.color:
            line_kwargs["color"] = line.color
        ax.axvline(
            line.value,
            label=line.label if line.label else f"x = {line.value:g}",
            **line_kwargs,
        )

    for region in args.hatch:
        mask = (x >= region.left) & (x <= region.right)
        probability = normal_cdf(region.right, args.mean, args.std) - normal_cdf(
            region.left, args.mean, args.std
        )
        label = (
            region.label
            if region.label
            else f"P({region.left:g} <= X <= {region.right:g}) = {probability:.4f}"
        )
        ax.fill_between(
            x[mask],
            y[mask],
            0,
            facecolor=args.hatch_color,
            edgecolor=args.hatch_color,
            alpha=args.hatch_alpha,
            hatch=args.hatch_pattern,
            label=label,
        )

    ax.set_title(args.title)
    ax.set_xlabel("X")
    ax.set_ylabel("Density")
    ax.set_xlim(args.x_min, args.x_max)
    ax.grid(alpha=0.25)

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc="best")

    fig.tight_layout()
    fig.savefig(args.output, dpi=200)
    print(f"Saved chart to {args.output}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
