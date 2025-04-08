# Synthetic ODT Benchmark Dataset

This repository provides code to generate synthetic datasets based on **Complete Orthogonal Balanced Oblique Decision Trees (COB-ODTs)**. These datasets are intentionally designed to challenge classical tree learners and kernel methods, while being learnable by deep non-linear models like DLGN and ReLU networks.

We open source this generator and encourage its use as a **benchmark challenge dataset** for learning discontinuous functions and evaluating feature learning models.

---

## Motivation

Decision trees are often used as interpretable models, but struggle when data is not axis-aligned. **Oblique Decision Trees (ODTs)**, which split based on arbitrary hyperplanes, introduce complex discontinuities that are hard to learn using traditional greedy methods like CART.

To expose this learning challenge, we generate data labelled by a COB-ODT:
- Internal nodes split data with **random orthogonal hyperplanes**.
- Labels at the leaves are assigned such that **sibling leaves get opposite signs**.
- Data points are sampled **uniformly from the surface of a unit sphere** in ℝ<sup>d</sup>.

This makes the resulting label function **discontinuous**, yet **learnable with non-linear models**.

---

## Dataset Variants

We provide 3 synthetic datasets with increasing complexity:

| Dataset | Input Dimension (d) | Number of Points | Description         |
|---------|---------------------|------------------|---------------------|
| SDI     | 20                  | 40,000           | Moderate difficulty |
| SDII    | 100                 | 60,000           | Higher dimension    |
| SDIII   | 500                 | 100,000          | Very high-dim       |

You can configure and generate these using generate_odt_data.py

---

## How to Run

Install dependencies (only `numpy` needed):

```bash
pip install -r requirements.txt

Run the generator script:

python generate_odt_data.py