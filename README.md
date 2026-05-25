# Credit Default Risk Analysis
### Vitto DS Intern Assessment

**Dataset:** UCI Credit Card Default · 30,000 clients · Taiwan, Apr–Sep 2005  
**Source:** https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients

---

## Setup

```bash
# 1. Clone the repository
git clone https://github.com/Stokesy-dev/Credit-default-risk.git
cd Credit-default-risk

# 2. Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Register the kernel
python -m ipykernel install --user --name vitto-venv --display-name "Python (vitto)"

# 5. Open the notebook
jupyter lab credit_default_risk.ipynb
```

> The dataset (`UCI_Credit_Card.csv`) must be in the same directory as the notebook.
> It is loaded via a relative path — no path changes needed.

---

## Methodology Summary

### 1. Exploratory Data Analysis
- Profiled shape, dtypes, null counts, and class balance (~22% default rate)
- Plotted distributions of `LIMIT_BAL`, `AGE`, and `PAY_0`; flagged right-skew and undocumented PAY codes
- Compared default rates across demographic segments (sex, education, marital status, age band)
- Visualised repayment delay patterns (`PAY_0`–`PAY_6`) by default outcome via heatmap
- Produced a full correlation matrix; identified top 5 features correlated with default

### 2. Feature Engineering

| Feature | Formula | Rationale |
|---------|---------|-----------|
| `AVG_UTIL_RATE` | mean(BILL_AMTx_clipped / LIMIT_BAL) over 6 months | Sustained high utilisation signals financial stress |
| `AVG_PAY_RATIO` | mean(PAY_AMTx / BILL_AMTx) where BILL_AMTx > 0 | Chronic under-payment predicts inability to service debt |
| `TOTAL_DELAY_MONTHS` | count(PAY_x > 0) over 6 months | Cumulative delinquency: stronger signal than any single month |

**Data quality decisions:**
- `EDUCATION` values {0, 5, 6} remapped to 4 (Other); `MARRIAGE` value {0} remapped to 3 (Other)
- Negative `BILL_AMT` values (overpayments/credit balances) clipped to 0 in ratio denominators; raw columns preserved
- `AVG_PAY_RATIO` imputed as 1.0 for clients with no bills (no outstanding balance = fully paid)
- `PAY_x` values of -2 (no consumption) and -1 (paid duly) are both treated as non-delinquent

### 3. Model Development

**Class imbalance (~22% default)** addressed with `class_weight='balanced'` (Logistic Regression) and `scale_pos_weight` (XGBoost). SMOTE was explicitly avoided to prevent data leakage inside cross-validation folds.

**Validation:** Stratified 80/20 train/test split; Stratified 5-fold CV on training set only.

| Model | Precision | Recall | F1 | AUC-ROC | CV AUC |
|-------|-----------|--------|----|---------|--------|
| Logistic Regression | 0.470 | 0.563 | 0.512 | 0.745 | 0.758 ± 0.008 |
| **XGBoost** | **0.471** | **0.625** | **0.537** | **0.778** | **0.782 ± 0.008** |

XGBoost selected as final model (higher recall and AUC-ROC).

### 4. Key Findings

**Top 5 predictive features (XGBoost gain-based importance):**

| Rank | Feature | Importance | Interpretation |
|------|---------|-----------|----------------|
| 1 | `TOTAL_DELAY_MONTHS` | 0.318 | Cumulative months with any payment delay |
| 2 | `PAY_0` | 0.147 | Most recent month payment status |
| 3 | `PAY_2` | 0.130 | Second most recent month |
| 4 | `PAY_4` | 0.030 | Fourth month status |
| 5 | `PAY_AMT2` | 0.020 | Second month payment amount |

**Default rate by cumulative delay months:**

| Delay months | Default rate |
|-------------|-------------|
| 0 | 11.7% |
| 1 | 29.8% |
| 3 | 50.9% |
| 6 | 70.3% |

### 5. SQL Bonus
Three business questions answered via SQLite:
- Default rate by education level
- Average credit limit by default status and sex
- Default rate by `TOTAL_DELAY_MONTHS` bucket

---

## Assumptions

1. `EDUCATION` values 0, 5, 6 are undocumented — treated as "Other" (category 4)
2. `MARRIAGE` value 0 is undocumented — treated as "Other" (category 3)
3. `PAY_x = -2` (no consumption) and `PAY_x = -1` (paid duly) are both treated as non-delinquent
4. Negative `BILL_AMT` values represent overpayments (credit balances), not data errors
5. `AVG_PAY_RATIO` for clients with zero bills across all 6 months is imputed as 1.0 (no debt = fully paid)

---

## Repository Structure

```
Credit-default-risk/
├── credit_default_risk.ipynb   # Main analysis notebook (run top-to-bottom)
├── UCI_Credit_Card.csv         # Dataset (30,000 records)
├── requirements.txt            # Pinned Python dependencies
├── README.md                   # This file
├── writeup.md                  # 1–2 page technical write-up
├── PRD.md                      # Product requirements document
├── plot_01_distributions.png
├── plot_02_demographic_default_rates.png
├── plot_03_delay_heatmap.png
├── plot_04_correlation_heatmap.png
├── plot_05_roc_curve.png
├── plot_06_confusion_matrix.png
└── plot_07_feature_importance.png
```

---

*This is a confidential assessment issued by Vitto. Do not distribute or publish publicly.*
