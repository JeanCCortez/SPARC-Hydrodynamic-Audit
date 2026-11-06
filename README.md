# SPARC Catalog Kinematic Audit: Parameter-Free Hydrodynamic Limit

This repository contains the complete, reproducible Python pipeline utilized to mathematically audit the Radial Acceleration Relation (RAR) of 175 late-type galaxies from the SPARC database.

The codebase is designed to test a rigid, zero-parameter ($k=0$) phenomenological hydrodynamic limit against standard parameterized dark matter halo models (such as $\Lambda$CDM NFW profiles). By locking the stellar mass-to-light ratio ($\Upsilon_d = 0.5\,M_\odot/L_\odot$) universally and enforcing an absolute macroscopic acceleration threshold ($a_0 = 1.200 \times 10^{-10}\ \text{m/s}^2$), this pipeline isolates the intrinsic physical law from the instrumental noise floor without algorithmic overfitting.

## 🛠️ Requirements & Dependencies

The pipeline is written in Python 3 and relies exclusively on standard, open-source scientific libraries. No proprietary software is required.

- **Python:** 3.8 or higher
- **NumPy:** `>= 1.21.0`
- **Pandas:** `>= 1.3.0`
- **SciPy:** `>= 1.7.0` (used for curve fitting and blind optimization)
- **Matplotlib:** `>= 3.4.0`

You can install all dependencies via pip:

```bash
pip install numpy pandas scipy matplotlib
```

## 📂 Data Acquisition (Crucial Step)

To ensure absolute transparency and prevent any claims of data manipulation, the raw SPARC data files are deliberately not included in this repository. You must download the unaltered empirical data directly from the official source.

1. Download the SPARC catalog (Mass Models) from the official database: [SPARC Database](http://astroweb.cwru.edu/SPARC/)
2. Extract the `.dat` rotation curve files.
3. Place all 175 `.dat` files into the `data/Rotmod_LTG/` directory of this project.

## ⚙️ Pipeline Execution & Script Architecture

Once the data is in place, execute the scripts sequentially from the root directory. Each script is independent and audits a specific layer of the kinematic data:

### `01_sparc_instrumental_noise_baseline.py`

**Function:** Parses all `.dat` files to extract the absolute instrumental noise floor of the SPARC sample (velocity uncertainty $\sigma_V$ vs observed velocity).

**Output:** Terminal log establishing the $\sim 7\%$ observational limit.

### `02_sparc_raw_kinematic_scatter.py`

**Function:** Calculates the raw global $\chi^2$ and Mean Absolute Error (MAE) across all 3,389 unbinned radial points using the rigid $k=0$ hydrodynamic equation.

### `03_sparc_binned_rar_extraction.py`

**Function:** Groups empirical data into 30 logarithmic baryonic acceleration bins to nullify localized distance/inclination artifacts, revealing the core physical law at a $2.65\%$ residual error.

### `04_sparc_comparative_bic_panel.py`

**Function:** Fits a 2-parameter NFW dark matter profile against the 0-parameter hydrodynamic ansatz. Calculates the Bayesian Information Criterion (BIC) for representative galaxies to evaluate predictive economy.

**Output:** Generates `sparc_comparative_panel.jpg` in the `/outputs` folder.

### `05_sparc_residual_distribution.py`

**Function:** Computes the signed percentage residual error for every raw data point and generates the global distribution histogram, isolating the $17.39\%$ raw geometric/instrumental scatter.

**Output:** Generates `sparc_real_global_residuals.png` in the `/outputs` folder.

### `06_sparc_montecarlo_cross_validation.py`

**Function:** A strict statistical stress-test. Executes 1,000 blind Monte Carlo iterations, bisecting the catalog into training and testing sets to mathematically prove the convergence of the mass-to-light ratio ($\Upsilon_d \approx 0.41\,M_\odot/L_\odot$) without data leakage.

## 📜 Author & License

**Jean C. Cortez** — Independent Researcher (Rio de Janeiro, Brazil).

This project is open-source. All analytical pipelines, residual minimization algorithms, and error propagations are provided for strict computational reproducibility and peer review.
