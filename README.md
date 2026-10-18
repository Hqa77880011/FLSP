# FLSP

Implementation of **FLSP: Fault-Localization-Guided Selective Protection for Quantized Inference**.

## Installation

Python >=3.10.

```bash
python -m pip install -e .
```

Inspection environment: Python 3.13.5, NumPy 2.1.3, PyTorch 2.7.1+cu118, SciPy 1.15.3 and scikit-learn 1.6.1. Checks cover Python syntax, JSON/TOML parsing and CLI options against parser definitions; training and evaluation were not run.

## Data

The bundled two-fault, two-group protection profile supplies conditional risk, unavailability, costs, intensity and resource caps. These are finite model coefficients, not measured device fault rates.

## Run

```bash
flsp execute --input examples/finite_profile.json --output runs/full --variant full
```

## Ablations

Use the same input profile for each variant: `cost_aware`.

```bash
flsp execute --input examples/finite_profile.json --output runs/cost_aware --variant cost_aware
```

## Outputs

`runs/full/placement.json` stores chosen protection actions, risk, costs and feasibility. Input and output paths are controlled by `--input` and `--output`.
