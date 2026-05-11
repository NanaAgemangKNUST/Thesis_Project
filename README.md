# Federated Learning for Cross-Border Malaria Surveillance
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18722519.svg)](https://doi.org/10.5281/zenodo.18722519)

> Privacy-preserving collaborative machine learning for malaria prevalence prediction across West African health systems

This repository contains the implementation and experimental code for our research on **federated learning for cross-border malaria surveillance** in West Africa, addressing the critical challenge of data heterogeneity while maintaining data sovereignty compliance with the Ghana Data Protection Act, Nigeria Data Protection Act 2023, and the African Union Malabo Convention.

## 📋 Table of Contents
- [Overview](#overview)
- [Key Features](#key-features)
- [Research Questions](#research-questions)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Dataset](#dataset)
- [Methods Compared](#methods-compared)
- [Experimental Protocol](#experimental-protocol)
- [Repository Structure](#repository-structure)
- [Configuration](#configuration)
- [Results](#results)
- [Citation](#citation)
- [License](#license)
- [Acknowledgments](#acknowledgments)

---

## 🔬 Overview

Malaria surveillance in West Africa faces two conflicting imperatives:

1. **Regional collaboration** for effective cross-border disease tracking
2. **Data sovereignty** mandated by national data protection regulations

This work demonstrates that **federated learning (FL)** can enable privacy-preserving collaboration while handling extreme data heterogeneity across countries. We compare **8 methods** — including centralized and local-only baselines and 6 federated learning algorithms — on real malaria surveillance data from Ghana, Mali, Nigeria, and Burkina Faso.

### The Challenge: Data Heterogeneity

West African malaria data exhibits **22-fold variation** in prevalence:
- **Coastal regions (Ghana):** 2.3% prevalence (hypoendemic)
- **Sahelian regions (Mali):** >50% prevalence (hyperendemic)

This extreme heterogeneity violates the i.i.d. assumption of standard federated learning, making algorithm selection critical.

### Our Contribution

- **Comprehensive comparison** of 8 methods (2 baselines + 6 FL algorithms)
- **Rigorous statistical validation** with 15 random seeds (~70% statistical power for effect size d=0.5 at α=0.05)
- **Privacy-utility tradeoff analysis** quantifying accuracy cost of privacy via centralized baseline
- **Real-world evaluation** on ~1,500 community clusters across 4 countries (DHS survey data)
- **Supplementary analyses** covering MOON hyperparameter sensitivity, normality-aware statistical tests with Bonferroni correction, calibration diagnostics, and preprocessing sensitivity
- **Reproducible implementation** with complete hyperparameter search documentation

---

## ✨ Key Features

### 🔐 Privacy-Preserving
- No raw data sharing across borders
- Compliant with African data protection regulations
- Local model training with gradient/weight sharing only

### 📊 Comprehensive Evaluation
- **8 methods compared:** Local-Only, Centralized, FedAvg, SCAFFOLD, FedProx+Per, SCAFFOLD+Per, IFCA, MOON
- **15 random seeds** for robust statistical conclusions (~70% power)
- **Paired statistical tests** with Shapiro-Wilk normality checks and Bonferroni correction
- **Calibration diagnostics:** reliability diagrams, Expected Calibration Error (ECE), WHO-threshold accuracy
- **Real malaria data:** ~1,500 community clusters from DHS surveys

### 🎯 Rigorous Hyperparameter Protocol
- Grid search over 9 hyperparameter dimensions on 20% held-out validation per country
- Selection seed set [42, 123, 456] for efficiency; full evaluation on 15 seeds
- All search spaces and selected values documented in `ComprehensiveFLConfig` and `HyperparameterSearchSpace`

### 📈 Advanced FL Methods
- **MOON:** Model-contrastive learning with projection head for heterogeneous data
- **SCAFFOLD:** Control variates to reduce client drift
- **IFCA:** Iterative federated clustering (K=2 epidemiological regimes)
- **Personalization:** Country-specific prediction heads (FedProx+Per, SCAFFOLD+Per)

---

## 🎯 Research Questions

| RQ | Question | Summary |
|----|----------|---------|
| **RQ1** | Can FL improve over local training for malaria surveillance? | Yes — contrastive and personalized methods outperform local-only training while preserving privacy |
| **RQ2** | Which FL methods best handle West African data heterogeneity? | MOON and personalized variants (FedProx+Per, SCAFFOLD+Per) are most robust to the 22-fold prevalence range |
| **RQ3** | Can advanced FL methods achieve privacy with minimal accuracy cost? | Yes — the best FL methods close most of the gap to the privacy-violating centralized ceiling |
| **RQ4** | What is the privacy-utility tradeoff vs centralized training? | Quantified exactly; see `table1_overall_results_ranked.csv` and the privacy-gap analysis in `comprehensive_all_methods_results.json` |

---

## 🔧 Installation

### Prerequisites

- Python 3.8+
- PyTorch 2.0+ (CPU or CUDA)

### Install dependencies

```bash
git clone https://github.com/NanaAgyemangKNUST/Thesis_Project.git
cd Thesis_Project
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118  # or cpu
pip install pandas numpy scipy matplotlib seaborn scikit-learn
```

### Verify install

```bash
python -c "import torch, pandas, scipy, seaborn; print('OK')"
```

---

## 🚀 Quick Start

### 1. Run the main experiment (all 8 methods, 15 seeds)

```bash
python comprehensive_fl_all_methods_with_centralized.py
```

Results are saved to `results/comprehensive_all_methods/`.

> **Note:** If real DHS data is not yet in `data/processed_improved/`, the script falls back to synthetic data automatically so you can verify the pipeline end-to-end.

### 2. Run supplementary analyses

```bash
# Supp. Analysis 1 — MOON sensitivity & ablation (Tables S1, S2)
python supp_analysis_1_moon_sensitivity.py

# Supp. Analysis 2 — statistical tests, calibration, threshold accuracy (Tables S3–S6)
python supp_analysis_2_stats_calibration.py
# Use synthetic data if main results JSON is not yet available:
python supp_analysis_2_stats_calibration.py --synthetic

# Supp. Analysis 3 — preprocessing sensitivity: filtered vs. unfiltered clusters (Table S7)
python supp_analysis_3_preprocessing_sensitivity.py
```

### 3. Generate publication figures (legacy script)

```bash
python visualize_results.py
```

> This script requires prior outputs from the main pipeline in `results/`.

---

## 📂 Dataset

Data are derived from **Demographic and Health Surveys (DHS)** for four West African countries:

| Country | Approx. Clusters | Baseline Prevalence | Epidemiological Zone |
|---------|-----------------|--------------------|--------------------|
| Ghana | ~400 | 28% | Coastal / hypoendemic |
| Mali | ~350 | 35–50%+ | Sahelian / hyperendemic |
| Nigeria | ~450 | 20–32% | Mixed |
| Burkina Faso | ~300 | 30–38% | Sahelian |

**Expected data layout:**

```
data/
└── processed_improved/
    ├── Ghana_clusters.csv
    ├── Mali_clusters.csv
    ├── Nigeria_clusters.csv
    └── Burkina_Faso_clusters.csv
```

Each CSV contains one row per community cluster with columns including `prevalence`, `positive_tests`, `total_tests`, `cluster_id`, `country`, `survey_year`, and environmental/socioeconomic covariates.

**Exclusion criterion:** Clusters with prevalence ∈ {0.0, 1.0} **and** `total_tests < 25` are removed as statistically unreliable. Supplementary Analysis 3 confirms method rankings are stable regardless of this filter.

**Data access:** DHS data are publicly available (registration required) at [dhsprogram.com](https://dhsprogram.com). Raw data cannot be redistributed; processed cluster-level CSVs are available upon reasonable request subject to DHS terms.

---

## 🧪 Methods Compared

| # | Method | Type | Key Idea |
|---|--------|------|----------|
| 1 | **Local-Only** | Baseline | Each country trains independently — privacy lower bound |
| 2 | **Centralized** | Baseline | All raw data pooled — privacy-violating accuracy ceiling |
| 3 | **FedAvg** | FL | Weighted average of client weights each round |
| 4 | **SCAFFOLD** | FL | Control variates correct client drift |
| 5 | **SCAFFOLD+Per** | FL + Personalization | SCAFFOLD shared layers + country-specific heads |
| 6 | **FedProx+Per** | FL + Personalization | Proximal regularization (μ=0.1) + personal heads |
| 7 | **IFCA** | FL + Clustering | K=2 clusters reflecting Sahelian vs. Coastal regimes |
| 8 | **MOON** | FL + Contrastive | Model-contrastive loss (μ=1.0, τ=0.5) via projection head |

---

## 🔬 Experimental Protocol

### Architecture

All methods use a shared MLP backbone `[input → 64 → 32]` with LayerNorm, ReLU, and Dropout(0.3). Personalized methods add a `[16 → 1]` country-specific head. MOON adds a `[64-dim]` projection head for contrastive learning.

### Hyperparameter Selection

Grid search was conducted offline over the following ranges, using 20% held-out validation per country and seeds [42, 123, 456]:

| Hyperparameter | Search Space | Selected |
|---|---|---|
| Learning rate | [0.001, 0.005, 0.01, 0.02] | 0.01 (FL), 0.005 (fine-tune) |
| Dropout | [0.2, 0.3, 0.4, 0.5] | 0.3 |
| Batch size | [16, 32, 64] | 32 |
| FL local epochs | [1, 3, 5] | 3 |
| Weight decay | [0.0, 0.001, 0.01] | 0.01 |
| FedProx μ | [0.01, 0.1, 1.0] | 0.1 |
| MOON μ | [0.5, 1.0, 2.0, 5.0] | 1.0 |
| MOON τ | [0.1, 0.5, 1.0] | 0.5 |
| IFCA K | [2, 3, 4] | 2 |

Selection criterion: minimum average validation MAE across countries.

### Training Configuration

- FL rounds: 80 | FL local epochs per round: 3 | Fine-tune epochs: 30
- Local-only epochs: 100 | Centralized epochs: 100 (with scheduler)
- Evaluation seeds: 15 (provides ~70% power for d=0.5 at α=0.05)
- Data split: 80% train / 20% test per country per seed

### Statistical Tests

- Paired t-test (primary, reported always)
- Shapiro-Wilk normality test on pairwise MAE differences (C-STA-001)
- Wilcoxon signed-rank test where normality is violated
- Bonferroni correction across all comparisons
- See `supp_analysis_2_stats_calibration.py` for full details

---

## 🗂️ Repository Structure

```
Thesis_Project/
│
├── comprehensive_fl_all_methods_with_centralized.py   # Main experiment: 8 methods × 15 seeds
├── new_pipeline.py                                     # Corrected pipeline (bug-fixed version)
├── supp_analysis_1_moon_sensitivity.py                # MOON μ/τ sensitivity + ablation
├── supp_analysis_2_stats_calibration.py               # Statistical tests, calibration, ECE
├── supp_analysis_3_preprocessing_sensitivity.py       # Filtered vs. unfiltered cluster sensitivity
├── visualize_results.py                               # Legacy figure generator
│
├── data/
│   └── processed_improved/                            # Filtered DHS cluster CSVs (not distributed)
│
└── results/
    └── comprehensive_all_methods/
        ├── comprehensive_all_methods_results.json     # Full raw results + hyperparameter metadata
        ├── figures/
        │   ├── fig1_all_methods_comparison.png/pdf    # Overall MAE ranking
        │   ├── fig2_per_country_all_methods.png/pdf   # Per-country breakdown
        │   ├── fig3_improvement_over_fedavg.png/pdf   # Relative improvement
        │   └── fig4_method_ranking.png/pdf            # Composite ranking
        ├── tables/
        │   ├── table1_overall_results_ranked.csv
        │   ├── table2_per_country_all_methods.csv
        │   ├── table3_tests_vs_local.csv
        │   ├── table4_tests_vs_fedavg.csv
        │   └── table5_ifca_clusters.csv
        └── supplementary/
            ├── tables/
            │   ├── supp_table_S1_moon_sensitivity.csv
            │   ├── supp_table_S2_moon_ablation.csv
            │   ├── supp_table_S3_normality_tests.csv
            │   ├── supp_table_S4_wilcoxon_tests.csv
            │   ├── supp_table_S5_calibration_ece.csv
            │   └── supp_table_S6_threshold_accuracy.csv
            └── figures/
                ├── supp_fig_S1_moon_sensitivity_heatmap.png/pdf
                ├── supp_fig_S3_reliability_diagrams.png/pdf
                ├── supp_fig_S4_threshold_roc.png/pdf
                └── supp_fig_S5_preprocessing_sensitivity.png/pdf
```

---

## ⚙️ Configuration

Key parameters in `ComprehensiveFLConfig` (top of `comprehensive_fl_all_methods_with_centralized.py`):

```python
config = ComprehensiveFLConfig(
    data_dir     = 'data/processed_improved',
    results_dir  = 'results/comprehensive_all_methods',
    seeds        = [42, 123, 456, 789, 2024, 1337, 7777, 8888,
                    9999, 1111, 2222, 3333, 4444, 5555, 6666],  # 15 seeds
    num_rounds   = 80,
    fl_local_epochs = 3,
    fedprox_mu   = 0.1,
    moon_mu      = 1.0,
    moon_temperature = 0.5,
    num_clusters = 2,       # IFCA: Sahelian vs. Coastal
    local_lr     = 0.01,
    dropout      = 0.3,
    batch_size   = 32,
)
```

GPU is used automatically when available (`torch.cuda.is_available()`).

---

## 📊 Results

All quantitative results are generated reproducibly by running the scripts above. Key output files:

- **`table1_overall_results_ranked.csv`** — Overall MAE/RMSE/R² for all 8 methods, ranked
- **`table2_per_country_all_methods.csv`** — Per-country breakdown for all methods
- **`comprehensive_all_methods_results.json`** — Full raw per-seed arrays, statistical test results, privacy-gap analysis, and hyperparameter metadata

For figures and tables used in the submitted manuscript, see `results/comprehensive_all_methods/figures/` and `results/comprehensive_all_methods/tables/`.

---

## 📖 Citation

If you use this code or data pipeline in your work, please cite:

```bibtex
@article{agyemang2026federated,
  title   = {Federated Learning for Privacy-Preserving Cross-Border Malaria
             Surveillance in West Africa},
  author  = {Agyemang, Ebenezer Nana},
  journal = {Discover Artificial Intelligence},
  year    = {2026},
  doi     = {10.5281/zenodo.18722519}
}
```

Zenodo archive: [https://doi.org/10.5281/zenodo.18722519](https://doi.org/10.5281/zenodo.18722519)

---

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

- **Supervisor:** Dr. Eric Osei Opoku, Department of Computer Science, KNUST
- **Institution:** Kwame Nkrumah University of Science and Technology (KNUST), Kumasi, Ghana
- **Data:** Demographic and Health Surveys (DHS) Program — Ghana, Mali, Nigeria, Burkina Faso
- **Regulatory framework:** Ghana Data Protection Act, Nigeria Data Protection Act 2023, African Union Malabo Convention

> **Author:** Agyemang Ebenezer Nana | MSc Computer Science, KNUST | January 2026
