# PRD: Credit Default Risk Analysis — Vitto DS Intern Assignment

## Problem Statement

Vitto needs to evaluate a candidate's ability to build a production-ready credit-risk signal pipeline
from raw borrower data. The candidate must demonstrate end-to-end data science competency —
from exploratory analysis through feature engineering, classification modelling, and clear
communication of findings — as if presenting to a real credit risk team.

The dataset (UCI Credit Card Default, 30,000 clients, Taiwan 2005) contains known data quality
issues (undocumented EDUCATION values, ambiguous PAY_x codes, negative BILL_AMT values) that
must be handled correctly and documented explicitly.

---

## Solution

Deliver a single, self-contained Jupyter notebook that:

1. Loads, profiles, and cleans the dataset
2. Performs structured exploratory data analysis
3. Engineers three domain-specific features
4. Trains and compares a Logistic Regression baseline and an XGBoost classifier
5. Validates using stratified 5-fold cross-validation
6. Surfaces the top predictive features and communicates findings to a non-technical audience
7. Answers three business questions via SQLite as a bonus

The notebook must run end-to-end without errors on a clean Python environment.

---

## User Stories

1. As a credit risk reviewer, I want to see the dataset shape, dtypes, null counts, and class balance reported upfront, so that I can immediately assess data quality.
2. As a credit risk reviewer, I want distributions of `LIMIT_BAL`, `AGE`, and `PAY_0` plotted with anomalies flagged, so that I understand the borrower population.
3. As a credit risk reviewer, I want default rates compared across `SEX`, `EDUCATION`, `MARRIAGE`, and `AGE` bands, so that I can spot demographic risk signals.
4. As a credit risk reviewer, I want repayment delay patterns across `PAY_0` to `PAY_6` visualised by default outcome, so that I can see how delay behaviour differs between defaulters and non-defaulters.
5. As a credit risk reviewer, I want a correlation heatmap with the top 5 features associated with default identified, so that I know where to focus modelling effort.
6. As a credit risk reviewer, I want `AVG_UTIL_RATE` engineered as `mean(BILL_AMTx / LIMIT_BAL)` across 6 months, so that I have a credit utilisation signal.
7. As a credit risk reviewer, I want `AVG_PAY_RATIO` engineered as `mean(PAY_AMTx / BILL_AMTx where BILL_AMTx > 0)`, so that I have a repayment consistency signal.
8. As a credit risk reviewer, I want `TOTAL_DELAY_MONTHS` engineered as `count(PAY_x > 0)`, so that I have a cumulative delinquency signal.
9. As a credit risk reviewer, I want negative `BILL_AMT` values (overpayments) explicitly flagged and clipped to 0 before ratio calculations, so that the feature engineering is economically sound.
10. As a credit risk reviewer, I want `AVG_PAY_RATIO` imputed as `1.0` for clients with no bills across all 6 months, so that clients with no outstanding balance are not penalised.
11. As a credit risk reviewer, I want undocumented `EDUCATION` values (0, 5, 6) remapped to the existing 'other' category (4), so that the encoding is clean and consistent.
12. As a credit risk reviewer, I want undocumented `MARRIAGE` value (0) remapped to the existing 'other' category (3), so that the encoding is clean and consistent.
13. As a credit risk reviewer, I want `EDUCATION` and `MARRIAGE` one-hot encoded with `drop='first'` after remapping, so that multicollinearity is avoided in Logistic Regression.
14. As a credit risk reviewer, I want all feature choices justified in a markdown cell, so that I understand the reasoning behind every engineering decision.
15. As a credit risk reviewer, I want the `PAY_x` ambiguity between -2 (no consumption) and -1 (paid duly) documented in a markdown cell, so that I know the candidate understood the data dictionary.
16. As a credit risk reviewer, I want a Logistic Regression baseline trained with `class_weight='balanced'` and Precision, Recall, F1, and AUC-ROC reported, so that I have a reference point for model performance.
17. As a credit risk reviewer, I want an XGBoost classifier trained with `scale_pos_weight` to address the ~22% default imbalance, so that the model is not biased toward the majority class.
18. As a credit risk reviewer, I want both models compared side-by-side on Precision, Recall, F1, and AUC-ROC, so that I can evaluate the improvement from the baseline.
19. As a credit risk reviewer, I want stratified 5-fold cross-validation run on the training set only, reporting mean ± std AUC-ROC for both models, so that performance estimates are robust and unbiased.
20. As a credit risk reviewer, I want the final model evaluated on a held-out test set (80/20 stratified split), so that the reported metrics reflect true out-of-sample performance.
21. As a credit risk reviewer, I want the top 5 predictive features from the best model listed with importance scores, so that I know which signals matter most.
22. As a credit risk reviewer, I want an ROC curve with both LR and XGBoost plotted on the same axes, so that I can visually compare model discrimination.
23. As a credit risk reviewer, I want a confusion matrix for the final model, so that I can understand the trade-off between false positives and false negatives.
24. As a credit risk reviewer, I want all plots to be publication quality (labelled axes, readable fonts, clear titles, consistent style), so that the analysis is presentable to a lending team.
25. As a lending manager, I want a 200–300 word plain-English summary of findings with zero jargon, so that I can understand the key insights without a data science background.
26. As a lending manager, I want two concrete, operational actions my credit team can take based on the findings, so that the analysis translates directly into business decisions.
27. As a data analyst, I want the dataset loaded into SQLite and three business questions answered with SQL queries, so that I can see the candidate's SQL fluency alongside Python skills.
28. As a notebook reviewer, I want all imports in the first cell and a single `RANDOM_STATE = 42` constant used everywhere, so that the notebook is fully reproducible.
29. As a notebook reviewer, I want the dataset loaded via a relative path, so that the notebook runs on any machine without path changes.
30. As a notebook reviewer, I want the notebook to run end-to-end without errors on a clean Python environment, so that I can evaluate it without debugging setup issues.

---

## Implementation Decisions

### Notebook Structure
- Single Jupyter notebook (`.ipynb`) saved alongside `UCI_Credit_Card.csv` in the project root
- Sections numbered to match the assignment: `00 Setup`, `01 EDA`, `02 Feature Engineering`, `03 Modelling`, `04 Validation`, `05 Insights`, `06 SQL Bonus`
- All imports in cell 1; `RANDOM_STATE = 42` defined once and referenced everywhere

### Data Splitting & Validation
- **80/20 stratified train/test split** performed before any modelling or CV
- **Stratified 5-fold CV** (`StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`) run exclusively on the training set
- Final ROC curve and confusion matrix computed on the held-out test set

### Class Imbalance
- Use `class_weight='balanced'` on Logistic Regression
- Use `scale_pos_weight = (n_negative / n_positive)` on XGBoost
- SMOTE explicitly excluded — avoids data leakage risk inside CV folds

### Negative `BILL_AMT` Handling
- Raw `BILL_AMT` columns preserved for EDA and flagging
- Clipped to 0 only when used as denominators in `AVG_UTIL_RATE` and `AVG_PAY_RATIO`
- Markdown cell flags the overpayment phenomenon and documents the clipping decision

### `AVG_PAY_RATIO` Edge Case
- Clients where `BILL_AMTx > 0` is never true across all 6 months → `AVG_PAY_RATIO` imputed as `1.0`
- Rationale: no outstanding balance = fully paid up, not a poor payer

### `PAY_x` Encoding
- Kept as raw numeric; no ordinal re-encoding applied
- Markdown cell documents -2 (no consumption) vs -1 (paid duly) distinction
- Both values ≤ 0 mean "not delinquent" — no ordinal economic meaning between them
- `TOTAL_DELAY_MONTHS = count(PAY_x > 0)` correctly excludes both

### EDUCATION & MARRIAGE Encoding
- Step 1: Remap `EDUCATION {0, 5, 6} → 4`, `MARRIAGE {0} → 3` (join existing 'other' categories)
- Step 2: One-hot encode both with `drop='first'` to prevent multicollinearity in Logistic Regression

### Tree-Based Model
- XGBoost chosen over Random Forest
- Gain-based feature importance is more informative for a lending context
- Industry-standard choice in FinTech credit risk

### SQL Bonus — Three Business Questions
1. Default rate by education level
2. Average credit limit by default status and sex
3. Default rate by `TOTAL_DELAY_MONTHS` bucket

### Plot Stack
| # | Plot | Purpose |
|---|------|---------|
| 1 | Distribution of `LIMIT_BAL`, `AGE`, `PAY_0` (3-panel) | EDA |
| 2 | Default rate by `SEX`, `EDUCATION`, `MARRIAGE`, `AGE` band | Demographic risk |
| 3 | Repayment delay heatmap: `PAY_0`–`PAY_6` by default outcome | Delay patterns |
| 4 | Correlation heatmap | Feature relationships |
| 5 | ROC curve — LR vs XGBoost on same axes | Model comparison |
| 6 | Confusion matrix for final model | Error analysis |

- seaborn for statistical/EDA plots; matplotlib for model evaluation plots
- Consistent style (e.g. `seaborn-v0_8-whitegrid`) applied across all plots

### Bonus Tasks Excluded
- SHAP explanations: out of scope
- Fairness check (FPR by SEX/EDUCATION): out of scope
- Streamlit app: out of scope

---

## Testing Decisions

Because this is a data science notebook (not a software package), automated unit tests are not part
of the deliverable. The following in-notebook validation checks serve as the effective test suite:

- **End-to-end run**: Notebook must execute top-to-bottom without errors on a clean environment
- **CV vs test consistency**: CV AUC-ROC (mean ± std) should be in a plausible range relative to final test AUC-ROC — a large gap signals leakage or overfitting
- **Class balance assertion**: After splitting, assert default rate in train and test sets is approximately equal (stratification check)
- **Feature shape assertion**: Assert engineered feature DataFrame has no unexpected nulls before modelling
- **`AVG_PAY_RATIO` imputation check**: Assert no NaN values remain after imputation

---

## Out of Scope

- SHAP explanations (beeswarm / waterfall plots)
- Fairness check (False Positive Rate by SEX / EDUCATION groups)
- Streamlit UI for client default probability prediction
- Hyperparameter tuning beyond defaults
- Deployment or productionisation of any model
- Any modification to the raw `UCI_Credit_Card.csv` file

---

## Further Notes

- The assignment is **confidential** — the notebook must not be published publicly
- The non-technical summary and two concrete actions must be written *after* seeing model results, derived from the actual top 5 XGBoost features
- `PAY_0` is expected to dominate feature importance — the two recommended lending actions should be calibrated against whichever features the model actually surfaces
