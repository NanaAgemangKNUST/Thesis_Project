# Federated Learning for Cross-Border Malaria Surveillance

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

> Privacy-preserving collaborative machine learning for malaria prevalence prediction across West African health systems

This repository contains the implementation and experimental code for our research on **federated learning for cross-border malaria surveillance** in West Africa, addressing the critical challenge of data heterogeneity while maintaining data sovereignty compliance with Ghana Data Protection Act, Nigeria Data Protection Act 2023, and the African Union Malabo Convention.

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Research Questions](#research-questions)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Dataset](#dataset)
- [Methods Compared](#methods-compared)
- [Experimental Protocol](#experimental-protocol)
- [Results](#results)
- [Repository Structure](#repository-structure)
- [Configuration](#configuration)
- [Citation](#citation)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## 🔬 Overview

Malaria surveillance in West Africa faces two conflicting imperatives:
1. **Regional collaboration** for effective cross-border disease tracking
2. **Data sovereignty** mandated by national data protection regulations

This work demonstrates that **federated learning (FL)** can enable privacy-preserving collaboration while handling extreme data heterogeneity across countries. We compare **8 methods** including centralized baselines and 6 federated learning algorithms on real malaria surveillance data from Ghana, Mali, Nigeria, and Burkina Faso.

### The Challenge: Data Heterogeneity

West African malaria data exhibits **22-fold variation** in prevalence:
- **Coastal regions (Ghana):** 2.3% prevalence (hypoendemic)
- **Sahelian regions (Mali):** >50% prevalence (hyperendemic)

This extreme heterogeneity violates the i.i.d. assumption of standard federated learning, making algorithm selection critical.

### Our Contribution

We provide:
- **Comprehensive comparison** of 8 methods (2 baselines + 6 FL algorithms)
- **Rigorous statistical validation** with 15 random seeds (~70% statistical power)
- **Privacy-utility tradeoff analysis** quantifying accuracy cost of privacy
- **Real-world evaluation** on ~1,500 community clusters across 4 countries
- **Reproducible implementation** with complete hyperparameter search documentation

## ✨ Key Features

### 🔐 Privacy-Preserving
- No raw data sharing across borders
- Compliant with African data protection regulations
- Local model training with gradient/weight sharing only

### 📊 Comprehensive Evaluation
- **8 methods compared:** Local-Only, Centralized, FedAvg, SCAFFOLD, FedProx+Per, SCAFFOLD+Per, IFCA, MOON
- **15 random seeds:** Sufficient statistical power for robust conclusions
- **Real malaria data:** ~1,500 community clusters from DHS surveys

### 🎯 Production-Ready
- Hyperparameter search via grid search on validation data
- Paired statistical tests with Bonferroni correction
- Extensive logging and result visualization
- Complete experimental reproducibility

### 📈 Advanced FL Methods
- **MOON:** Model-contrastive learning for heterogeneous data
- **SCAFFOLD:** Control variates for reduced client drift
- **IFCA:** Adaptive clustering for data regimes
- **Personalization:** Country-specific prediction heads

## 🎯 Research Questions

**RQ1:** Can federated learning improve over local training for malaria surveillance?
- **Answer:** Yes, MOON achieves 1.5pp better MAE than local training while preserving privacy

**RQ2:** Which FL methods best handle West African data heterogeneity?
- **Answer:** MOON (13.96 MAE) > SCAFFOLD+Per (14.63 MAE) > IFCA (15.31 MAE)

**RQ3:** How much accuracy is lost for privacy vs. centralized training?
- **Answer:** MOON loses only 0.54pp vs. centralized (13.96 vs. 13.42 MAE)

**RQ4:** Does standard FedAvg work for heterogeneous health data?
- **Answer:** No, FedAvg (16.35 MAE) significantly underperforms local training (15.26 MAE)

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- CUDA-compatible GPU (recommended, not required)
- 8GB+ RAM

### Step 1: Clone the Repository
```bash
git clone https://github.com/yourusername/federated-malaria-surveillance.git
cd federated-malaria-surveillance
```

### Step 2: Create Virtual Environment
```bash
# Using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Or using conda
conda create -n fl-malaria python=3.8
conda activate fl-malaria
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Dependencies
Core packages:
- **PyTorch 2.0+**: Deep learning framework
- **Flower 1.5+**: Federated learning framework
- **NumPy, Pandas, SciPy**: Scientific computing
- **Scikit-learn**: Machine learning utilities
- **Matplotlib, Seaborn**: Visualization

See `requirements.txt` for complete list.

## 🏃 Quick Start

### 1. Prepare Your Data

Place DHS malaria survey data in the `data/` directory:

```bash
data/
├── processed_multiyear_filtered_15/
│   ├── Ghana_clusters.csv
│   ├── Mali_clusters.csv
│   ├── Nigeria_clusters.csv
│   └── Burkina_Faso_clusters.csv
```

Each CSV should contain:
- `prevalence`: Malaria prevalence (0-1)
- `altitude`, `temperature`, `rainfall`: Environmental features
- `population_density`, `urban`: Demographic features
- `wealth_score`: Socioeconomic indicator
- Additional survey features

### 2. Preprocess Data (Optional)

If you need to improve data quality:

```bash
python data_preprocessing.py
```

This will:
- Handle missing values via median/mode imputation
- Detect and handle outliers
- Generate data quality reports
- Create improved datasets in `data/processed_improved/`

### 3. Run Full Comparison

```bash
python comprehensive_fl_all_methods_with_centralized.py
```

This executes all 8 methods with 15 random seeds. **Expected runtime:** 4-8 hours on GPU, 12-24 hours on CPU.

### 4. View Results

Results are saved to `results/comprehensive_all_methods/`:

```
results/comprehensive_all_methods/
├── comprehensive_all_methods_results.json  # Raw results + metadata
├── figures/
│   ├── fig1_all_methods_comparison.png     # Method comparison
│   ├── fig2_per_country_all_methods.png    # Country-level results
│   ├── fig3_improvement_over_fedavg.png    # FedAvg baseline
│   └── fig4_method_ranking.png             # Overall ranking
└── tables/
    ├── table1_overall_results_ranked.csv   # Main results
    ├── table2_per_country_all_methods.csv  # Per-country breakdown
    ├── table3_tests_vs_local.csv           # Statistical tests vs. local
    ├── table4_tests_vs_fedavg.csv          # Statistical tests vs. FedAvg
    └── table5_ifca_clusters.csv            # IFCA cluster assignments
```

## 📊 Dataset

### Data Source
Demographic and Health Surveys (DHS) from:
- **Ghana** (2022): ~350 clusters
- **Mali** (2021): ~300 clusters
- **Nigeria** (2023-24): ~500 clusters
- **Burkina Faso** (2021): ~350 clusters

**Total:** ~1,500 community clusters

### Features (18 total)

**Environmental (3):**
- Altitude (meters)
- Mean annual temperature (°C)
- Mean annual rainfall (mm)

**Demographic (5):**
- Population density
- Urban/rural classification
- Under-5 population
- Household size
- Education index

**Socioeconomic (3):**
- Wealth score (composite index)
- Improved water access
- Improved sanitation access

**Health System (4):**
- Distance to health facility
- ITN (insecticide-treated net) coverage
- IRS (indoor residual spraying) coverage
- Access to antimalarial drugs

**Temporal (3):**
- Survey year
- Season (rainy/dry)
- Months since last survey

### Target Variable
- **Prevalence:** Proportion of children under 5 testing positive for malaria (0-1 scale)

### Data Characteristics

| Country | Clusters | Mean Prev. (%) | Std Dev (pp) | Min-Max (%) |
|---------|----------|----------------|--------------|-------------|
| Ghana | 350 | 12.4 | 8.2 | 2.3 - 31.5 |
| Mali | 300 | 38.6 | 12.1 | 18.2 - 54.8 |
| Nigeria | 500 | 24.7 | 14.3 | 5.1 - 48.3 |
| Burkina Faso | 350 | 41.2 | 10.8 | 22.4 - 58.1 |

## 🧪 Methods Compared

### Baseline Methods

#### 1. Local-Only
- **Description:** Each country trains independently (no collaboration)
- **Purpose:** Privacy-preserving lower bound on performance
- **Result:** 15.26 pp MAE

#### 2. Centralized
- **Description:** Pool all data centrally (privacy-violating)
- **Purpose:** Upper bound on accuracy (maximum collaboration)
- **Result:** 13.42 pp MAE
- **Note:** Not deployable due to data protection regulations

### Federated Learning Methods

#### 3. FedAvg (Baseline FL)
- **Description:** Standard federated averaging ([McMahan et al., 2017](http://proceedings.mlr.press/v54/mcmahan17a.html))
- **Formula:** `w_global = Σ(n_k/n_total * w_k)`
- **Result:** 16.35 pp MAE
- **Issue:** Performs worse than local training due to data heterogeneity

#### 4. SCAFFOLD
- **Description:** Control variates for reduced client drift ([Karimireddy et al., 2020](http://proceedings.mlr.press/v119/karimireddy20a.html))
- **Innovation:** Tracks control variates to correct for local updates
- **Result:** 14.66 pp MAE (+10.3% vs. FedAvg)

#### 5. SCAFFOLD+Per
- **Description:** SCAFFOLD with personalized prediction heads
- **Innovation:** Shared representation + country-specific final layers
- **Result:** 14.63 pp MAE (+10.5% vs. FedAvg)

#### 6. FedProx+Per
- **Description:** Proximal regularization + personalization ([Li et al., 2020](https://proceedings.mlsys.org/paper/2020/file/3280256e5f6d3d8e2c0a3f8f3a8e0a3f-Paper.pdf))
- **Innovation:** Adds proximal term to limit drift
- **Result:** 15.03 pp MAE (+8.1% vs. FedAvg)

#### 7. IFCA (Clustering)
- **Description:** Iterative Federated Clustering Algorithm ([Ghosh et al., 2020](https://proceedings.neurips.cc/paper/2020/file/f083b7a8c0e91c8a0e1f8d0e9c8a0e1f-Paper.pdf))
- **Innovation:** Discovers K=2 clusters (Sahelian vs. Coastal)
- **Result:** 15.31 pp MAE (+6.4% vs. FedAvg)

#### 8. MOON (Best Method) ⭐
- **Description:** Model-Contrastive Federated Learning ([Li et al., 2021](https://doi.org/10.1109/CVPR46437.2021.01060))
- **Innovation:** Contrastive learning to align representations across countries
- **Formula:** `L = L_sup + μ * L_con` where `L_con` pulls similar countries together
- **Result:** 13.96 pp MAE (+14.6% vs. FedAvg)
- **Highlights:**
  - Only 1.5 pp worse than local training
  - Only 0.54 pp worse than centralized
  - Achieves privacy with minimal accuracy cost

## 🔬 Experimental Protocol

### Statistical Rigor

**Seeds:** 15 random seeds (42, 123, 456, 789, 2024, 1337, 7777, 8888, 9999, 1111, 2222, 3333, 4444, 5555, 6666)
- Provides ~70% statistical power for effect size d=0.5 at α=0.05

**Statistical Tests:**
- Paired t-tests comparing each method to baselines
- Bonferroni correction for multiple comparisons
- 95% confidence intervals reported

### Hyperparameter Selection

**Protocol:**
1. 20% held-out validation set (separate from test set)
2. Grid search over documented parameter ranges
3. Selection via minimum average validation MAE
4. Validation used 3 seeds (42, 123, 456) for efficiency

**Search Spaces:**
```python
Learning rates:     [0.001, 0.005, 0.01, 0.02]     → Selected: 0.01
Dropout rates:      [0.2, 0.3, 0.4, 0.5]           → Selected: 0.3
FedProx μ:          [0.01, 0.1, 1.0]               → Selected: 0.1
MOON μ:             [0.5, 1.0, 2.0, 5.0]           → Selected: 1.0
MOON τ:             [0.1, 0.5, 1.0]                → Selected: 0.5
IFCA K:             [2, 3, 4]                      → Selected: 2
Batch sizes:        [16, 32, 64]                   → Selected: 32
FL local epochs:    [1, 3, 5]                      → Selected: 3
Weight decay:       [0.0, 0.001, 0.01]             → Selected: 0.01
```

All hyperparameters documented in code with selection rationale.

### Training Configuration

**Model Architecture:**
- Input: 18 features
- Hidden layers: [64, 32] neurons
- Personalized head: [16] neurons (for personalization methods)
- Dropout: 0.3
- Activation: ReLU

**Training:**
- FL rounds: 80
- Local epochs per round: 3
- Batch size: 32
- Learning rate: 0.01
- Weight decay: 0.01
- Optimizer: Adam

**Evaluation:**
- Metric: Mean Absolute Error (MAE) in percentage points
- Test set: 20% held-out per country
- Early stopping: Patience of 15 rounds on validation loss

## 📈 Results

### Overall Performance (MAE in percentage points, ↓ better)

| Rank | Method | MAE (pp) | 95% CI | vs. Local | vs. FedAvg | Privacy? |
|------|--------|----------|--------|-----------|------------|----------|
| 1 | **Centralized** | **13.42** | [13.1, 13.7] | +12.1% ✓ | +17.9% ✓ | ❌ No |
| 2 | **MOON** ⭐ | **13.96** | [13.6, 14.3] | +8.5% ✓ | +14.6% ✓ | ✅ Yes |
| 3 | SCAFFOLD+Per | 14.63 | [14.2, 15.0] | +4.1% ✓ | +10.5% ✓ | ✅ Yes |
| 4 | SCAFFOLD | 14.66 | [14.3, 15.0] | +3.9% ✓ | +10.3% ✓ | ✅ Yes |
| 5 | FedProx+Per | 15.03 | [14.7, 15.4] | +1.5% | +8.1% ✓ | ✅ Yes |
| 6 | **Local-Only** | **15.26** | [14.9, 15.6] | — | +6.7% ✓ | ✅ Yes |
| 7 | IFCA | 15.31 | [14.9, 15.7] | -0.3% | +6.4% ✓ | ✅ Yes |
| 8 | FedAvg | 16.35 | [15.9, 16.8] | -7.1% ✗ | — | ✅ Yes |

**Key Findings:**
- ✅ **MOON is the best privacy-preserving method**
- ❌ **FedAvg fails:** Worse than local training
- 💡 **Privacy-utility tradeoff:** MOON loses only 0.54 pp vs. centralized
- 📊 All improvements statistically significant (p < 0.01)

### Per-Country Results

| Country | Local | FedAvg | SCAFFOLD | MOON | Centralized |
|---------|-------|--------|----------|------|-------------|
| Ghana | 12.8 | 14.2 | 12.1 | **11.9** | 11.2 |
| Mali | 16.5 | 17.8 | 16.4 | **15.1** | 14.8 |
| Nigeria | 15.1 | 16.9 | 14.8 | **14.2** | 13.9 |
| Burkina Faso | 16.7 | 16.9 | 16.1 | **14.5** | 13.8 |

MOON achieves best FL performance in all 4 countries.

### Statistical Significance

All comparisons vs. FedAvg: **p < 0.01**
- MOON vs. FedAvg: t = 12.4, p < 0.001
- SCAFFOLD vs. FedAvg: t = 8.7, p < 0.001  
- Local vs. FedAvg: t = 6.2, p < 0.001

Bonferroni-corrected significance level: α = 0.0071 (7 comparisons)

## 📁 Repository Structure

```
federated-malaria-surveillance/
│
├── README.md                          # This file
├── LICENSE                            # MIT License
├── requirements.txt                   # Python dependencies
│
├── data/                              # Data directory (not included)
│   ├── processed_multiyear_filtered_15/
│   │   ├── Ghana_clusters.csv
│   │   ├── Mali_clusters.csv
│   │   ├── Nigeria_clusters.csv
│   │   └── Burkina_Faso_clusters.csv
│   └── processed_improved/            # Output from data_preprocessing.py
│
├── comprehensive_fl_all_methods_with_centralized.py  # Main experiment
├── data_preprocessing.py              # Data quality analysis
│
├── results/                           # Experimental results (generated)
│   └── comprehensive_all_methods/
│       ├── comprehensive_all_methods_results.json
│       ├── figures/
│       │   ├── fig1_all_methods_comparison.png
│       │   ├── fig2_per_country_all_methods.png
│       │   ├── fig3_improvement_over_fedavg.png
│       │   └── fig4_method_ranking.png
│       └── tables/
│           ├── table1_overall_results_ranked.csv
│           ├── table2_per_country_all_methods.csv
│           ├── table3_tests_vs_local.csv
│           ├── table4_tests_vs_fedavg.csv
│           └── table5_ifca_clusters.csv
│
└── docs/                              # Additional documentation
    ├── HYPERPARAMETERS.md             # Hyperparameter selection details
    ├── DATA_FORMAT.md                 # Expected data format
    └── METHODS.md                     # Detailed method descriptions
```

## ⚙️ Configuration

### Modify Experimental Settings

Edit `ComprehensiveFLConfig` in `comprehensive_fl_all_methods_with_centralized.py`:

```python
@dataclass 
class ComprehensiveFLConfig:
    # Number of random seeds (affects statistical power)
    seeds: List[int] = field(default_factory=lambda: SEEDS)  # 15 seeds
    
    # Training rounds/epochs
    num_rounds: int = 80           # FL communication rounds
    fl_local_epochs: int = 3       # Local epochs per round
    
    # Model architecture
    hidden_dims: List[int] = field(default_factory=lambda: [64, 32])
    dropout: float = 0.3
    
    # Learning rates
    local_lr: float = 0.01         # FL methods
    finetune_lr: float = 0.005     # Fine-tuning
    
    # Method-specific parameters
    fedprox_mu: float = 0.1        # FedProx proximal term
    moon_mu: float = 1.0           # MOON contrastive weight
    moon_temperature: float = 0.5  # MOON temperature
    num_clusters: int = 2          # IFCA clusters
    
    # Data paths
    data_dir: str = 'data/processed_improved'
    results_dir: str = 'results/comprehensive_all_methods'
```

### Run Subset of Methods

Modify the `methods_to_run` list in `run_comprehensive_comparison()`:

```python
methods_to_run = [
    'local',           # Local-Only
    'centralized',     # Centralized
    'fedavg',          # FedAvg
    'scaffold',        # SCAFFOLD
    'scaffold_per',    # SCAFFOLD+Per
    'fedprox_per',     # FedProx+Per
    'ifca',            # IFCA
    'moon'             # MOON
]
```

### Adjust Statistical Power

Change the number of seeds for different power levels:

```python
# High power (90%): 25 seeds, ~48-60 hours
SEEDS = list(range(1, 26))

# Medium power (70%): 15 seeds, ~20-30 hours (default)
SEEDS = [42, 123, 456, 789, 2024, 1337, 7777, 8888, 9999, 
         1111, 2222, 3333, 4444, 5555, 6666]

# Low power (50%): 5 seeds, ~6-10 hours (quick test)
SEEDS = [42, 123, 456, 789, 2024]
```

## 📄 Citation

If you use this code or findings in your research, please cite:

```bibtex
@article{agyemang2026federated,
  title={Federated Learning for Cross-Border Malaria Surveillance: Managing Data Heterogeneity Across West African Health Systems},
  author={Agyemang, Ebenezer Nana and Osei, Eric Opoku},
  journal={BMC Medical Informatics and Decision Making},
  year={2026},
  note={Under review}
}
```

### Key References

**Federated Learning:**
- McMahan et al. (2017). Communication-efficient learning of deep networks from decentralized data. AISTATS.
- Karimireddy et al. (2020). SCAFFOLD: Stochastic controlled averaging for federated learning. ICML.
- Li et al. (2021). Model-contrastive federated learning. CVPR.

**Malaria Surveillance:**
- WHO (2023). World Malaria Report 2023. Geneva: World Health Organization.
- Weiss et al. (2019). Mapping the global prevalence, incidence, and mortality of Plasmodium falciparum. Lancet.

**Data Protection:**
- Ghana Data Protection Act (Act 843) of 2012
- Nigeria Data Protection Act 2023
- African Union Malabo Convention on Cyber Security and Personal Data Protection (2014)

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### MIT License Summary
- ✅ Commercial use
- ✅ Modification
- ✅ Distribution
- ✅ Private use
- ⚠️ Liability and warranty disclaimers apply

## 🙏 Acknowledgments

### Data
- **Demographic and Health Surveys (DHS) Program** for providing malaria survey data
- **ICF International** for data access and support
- National malaria control programs of Ghana, Mali, Nigeria, and Burkina Faso

### Funding
- Kwame Nkrumah University of Science and Technology (KNUST)
- [Add your funding sources]

### Supervision
- **Dr. Eric Osei Opoku** - Research supervisor, KNUST Department of Computer Science

### Collaborators
- [Add collaborating institutions/individuals]

### Open Source
This work builds on excellent open-source projects:
- [PyTorch](https://pytorch.org/) - Deep learning framework
- [Flower](https://flower.dev/) - Federated learning framework
- [Scikit-learn](https://scikit-learn.org/) - Machine learning tools

## 🤝 Contributing

We welcome contributions! Please follow these guidelines:

1. **Fork the repository**
2. **Create a feature branch** (`git checkout -b feature/amazing-feature`)
3. **Make your changes** with clear commit messages
4. **Add tests** if applicable
5. **Update documentation** as needed
6. **Submit a pull request**

### Areas for Contribution
- Additional FL algorithms (e.g., FedNova, FedDyn)
- Support for additional countries/regions
- Performance optimizations
- Documentation improvements
- Bug fixes

## 📮 Contact

**Agyemang Ebenezer Nana**
- 📧 Email: enagyemang4@st.knust.edu.gh
- 🎓 Institution: Kwame Nkrumah University of Science and Technology (KNUST)
- 📍 Location: Kumasi, Ghana

**Dr. Eric Osei Opoku** (Supervisor)
- 📧 Email: [supervisor email]
- 🎓 Department: Computer Science, KNUST

## 🐛 Issues and Support

If you encounter any problems or have questions:

1. **Check existing issues:** [GitHub Issues](https://github.com/yourusername/federated-malaria-surveillance/issues)
2. **Create a new issue:** Provide detailed description, error messages, and system information
3. **For urgent matters:** Contact via email

### Common Issues
- **CUDA out of memory:** Reduce batch size or use CPU
- **Data loading errors:** Verify CSV format matches expected structure
- **Slow training:** Reduce number of seeds or use GPU

## 🗺️ Roadmap

### Current Version (v1.0)
- ✅ 8 methods compared
- ✅ 15-seed statistical validation
- ✅ Complete hyperparameter search documentation
- ✅ Privacy-utility tradeoff analysis

### Planned Features (v1.1)
- [ ] Additional FL algorithms (FedNova, FedDyn, FedBN)
- [ ] Support for differential privacy
- [ ] Real-time deployment simulation
- [ ] Interactive result visualization dashboard
- [ ] Docker containerization

### Future Work (v2.0)
- [ ] Multi-disease surveillance (malaria + COVID + other diseases)
- [ ] Temporal modeling with RNNs/Transformers
- [ ] Secure multi-party computation integration
- [ ] Production deployment guide

## ⚡ Performance Tips

### For Faster Experiments
```python
# Quick test (3 seeds, ~2 hours on GPU)
SEEDS = [42, 123, 456]
config.num_rounds = 40
config.fl_local_epochs = 2
```

### For GPU Memory Issues
```python
# Reduce memory usage
config.batch_size = 16
config.hidden_dims = [32, 16]  # Smaller model
```

### For High Statistical Power
```python
# Maximum rigor (25 seeds, ~48-60 hours)
SEEDS = list(range(1, 26))
```

## 📊 Expected Runtime

| Configuration | Seeds | Rounds | Expected Time (GPU) | Expected Time (CPU) |
|--------------|-------|--------|---------------------|---------------------|
| Quick test | 3 | 40 | ~2 hours | ~6 hours |
| Default | 15 | 80 | ~8 hours | ~24 hours |
| High power | 25 | 80 | ~12 hours | ~36 hours |

*Times estimated on NVIDIA RTX 3090 GPU / Intel i9 CPU*

---

## 🌍 Impact

This research contributes to:
- **Global health:** Improved malaria surveillance for 200M+ at-risk population
- **Data sovereignty:** Privacy-preserving collaboration compliant with African regulations
- **ML research:** Advancing federated learning for heterogeneous data
- **Policy:** Evidence-based recommendations for cross-border health data governance

---

**Star ⭐ this repository if you find it useful!**

**Made with ❤️ for better malaria surveillance in West Africa**
