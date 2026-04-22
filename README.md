# Normal Distribution Plotter

A small, focused Python utility for drawing **normal-distribution (Gaussian)
bell curves** with three practical customizations:

1. **Custom horizontal-axis labels** – show raw values, z-scores, σ-units,
   cumulative probabilities, or any custom mapping.
2. **Vertical lines** at any value(s) you choose (cutoffs, thresholds,
   critical values…).
3. **Hatched shaded regions** under the curve to display probabilities.

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

plot.add_vertical_line(130, label="IQ = 130 (gifted cutoff)", color="crimson")

plot.add_hatched_region(lower=130, upper=None,
                        label="IQ ≥ 130", hatch="\\\\\\", edgecolor="crimson")
plot.add_hatched_region(lower=85, upper=115,
                        label="average range", hatch="...", edgecolor="C2")

plot.plot(title="IQ scores", xlabel="IQ score")
plt.show()
```

## API overview

### `NormalDistributionPlot(mean=0.0, std=1.0, x_range=None, num_points=1000)`

Creates a plotter for `N(mean, std^2)`. If `x_range` is omitted it defaults
to `mean ± 4·std`.

### Vertical lines

```python
plot.add_vertical_line(x, label=None, color="black",
                       linestyle="--", linewidth=1.5, alpha=0.9)
```

Call as many times as you like — each call appends a new vertical line.

### Hatched / shaded regions

```python
plot.add_hatched_region(lower=None, upper=None, label=None,
                        hatch="///", edgecolor="C0", facecolor="none",
                        alpha=0.6, show_probability=True)
```

- Use `lower=None` for a left-open tail (`-∞`) and `upper=None` for a
  right-open tail (`+∞`).
- When `show_probability=True`, the exact probability mass of the region
  (`P(lower ≤ X ≤ upper)`) is appended to the legend label.
- Popular hatch patterns: `"/"`, `"\\"`, `"|"`, `"-"`, `"+"`, `"x"`,
  `"o"`, `"O"`, `"."`, `"*"` — and any combination/repetition for density.

### Custom x-axis tick labels

Three convenience helpers are provided:

```python
plot.set_xticks([55, 70, 85, 100, 115, 130, 145])

plot.set_xticks_by_sigma(sigma_values=(-3, -2, -1, 0, 1, 2, 3),
                         as_sigma_labels=True)

plot.set_xticks_by_mapping([-2, -1, 0, 1, 2],
                           lambda z: f"CDF={norm.cdf(z):.3f}")
```

### Rendering

```python
plot.plot(ax=None, title=None, xlabel="x",
          ylabel="Probability density",
          curve_color="C0", curve_linewidth=2.0,
          show_legend=True, grid=True)
```

Returns the `matplotlib.axes.Axes` so you can tweak it further.

## Demo gallery

Running the module directly renders four example plots that exercise every
feature:

```bash
python normal_distribution_plot.py
```
