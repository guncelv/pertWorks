# pertWorks

## Normal distribution plotting utility

This repository now includes `normal_distribution_plot.py`, a command-line tool to
generate normal distribution graphs with:

- custom values displayed on the horizontal axis (`--x-values`)
- vertical marker lines at any chosen value (`--vline`)
- hatched regions to visualize probabilities (`--hatch`)

### Requirements

Install dependencies:

```bash
pip install matplotlib numpy
```

### Basic usage

```bash
python normal_distribution_plot.py
```

This produces `normal_distribution.png`.

### Example with custom x-axis values, vertical lines, and hatched areas

```bash
python normal_distribution_plot.py \
  --mean 100 \
  --std 15 \
  --x-min 40 \
  --x-max 160 \
  --x-values 40,55,70,85,100,115,130,145,160 \
  --vline 85:"Lower threshold":tab:red \
  --vline 115:"Upper threshold":tab:orange \
  --hatch 70:100:"P(70 <= X <= 100)" \
  --hatch 100:130:"P(100 <= X <= 130)" \
  --output normal_distribution_example.png
```

### CLI help

```bash
python normal_distribution_plot.py --help
```
