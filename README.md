# Normal Distribution Plotter

A small Python utility for drawing **normal-distribution (Gaussian) bell
curves** with three practical customizations:

1. **Custom horizontal-axis labels** – show raw values, z-scores, σ-units,
   or any custom mapping.
2. **Vertical lines** at any value(s) you choose.
3. **Hatched shaded regions** under the curve to display probabilities.

The output is deliberately minimal: only the curve, any hatched regions,
any vertical lines, and the horizontal-axis tick markers are drawn. No
title, no axis labels, no legend, no grid, and no y-axis are rendered.

It is implemented with `numpy`, `scipy.stats.norm`, and `matplotlib`.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Quick start

```python
from normal_distribution_plot import NormalDistributionPlot
import matplotlib.pyplot as plt

plot = NormalDistributionPlot(mean=100, std=15, x_range=(40, 160))

plot.set_xticks([55, 70, 85, 100, 115, 130, 145])

plot.add_vertical_line(130, color="crimson")

plot.add_hatched_region(lower=130, upper=None, hatch="\\\\\\", edgecolor="crimson")
plot.add_hatched_region(lower=85, upper=115, hatch="...", edgecolor="C2")

plot.plot()
plt.show()
```

## API overview

### `NormalDistributionPlot(mean=0.0, std=1.0, x_range=None, num_points=1000)`

Creates a plotter for `N(mean, std^2)`. If `x_range` is omitted it defaults
to `mean ± 4·std`.

### Vertical lines

```python
plot.add_vertical_line(x, color="black",
                       linestyle="--", linewidth=1.5, alpha=0.9)
```

Call as many times as you like — each call appends a new vertical line.

### Hatched / shaded regions

```python
plot.add_hatched_region(lower=None, upper=None,
                        hatch="///", edgecolor="C0",
                        facecolor="none", alpha=0.6)
```

- Use `lower=None` for a left-open tail (`-∞`) and `upper=None` for a
  right-open tail (`+∞`).
- Popular hatch patterns: `"/"`, `"\\"`, `"|"`, `"-"`, `"+"`, `"x"`,
  `"o"`, `"O"`, `"."`, `"*"` — and any combination/repetition for density.

### Custom x-axis tick labels

Three convenience helpers are provided:

```python
plot.set_xticks([55, 70, 85, 100, 115, 130, 145])

plot.set_xticks_by_sigma(sigma_values=(-3, -2, -1, 0, 1, 2, 3),
                         as_sigma_labels=True)

plot.set_xticks_by_mapping([-2, -1, 0, 1, 2],
                           lambda z: f"{z:+g}")
```

### Rendering

```python
plot.plot(ax=None, curve_color="C0", curve_linewidth=2.0)
```

Returns the `matplotlib.axes.Axes` so you can tweak it further if you
really want to — by default the plot shows only the curve, hatched
regions, vertical lines, and horizontal-axis tick markers.

## Demo gallery

Running the module directly renders four example plots that exercise every
feature:

```bash
python normal_distribution_plot.py
```
