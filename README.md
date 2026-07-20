# Handling Missing Data — Imputation Strategies Compared (Titanic Dataset)

A practical, side-by-side comparison of **five missing-value handling strategies** — column dropping, mode imputation, median imputation, arbitrary-value imputation, and KNN imputation — plus categorical encoding methods, demonstrated on the classic Titanic dataset (**891 passengers, 19.9% of ages missing, 77.1% of cabins missing**).

## Objective

Missing data is the first real decision in any ML pipeline, and the wrong choice silently corrupts every model downstream. This project demonstrates each mainstream strategy, shows *when* each is appropriate, and — crucially — visualizes what each one does to the underlying distribution, so the trade-offs stop being theoretical.

## Dataset

| Property | Detail |
|---|---|
| Records | 891 passengers × 12 columns |
| Missing values | **Age: 177 (19.87%) · Cabin: 687 (77.10%) · Embarked: 2 (0.22%)** |
| Age profile | Mean 29.7 · Median 28.0 · Std 14.5 · Range 0.42–80 years |
| Categoricals | Sex (577 M / 314 F), Embarked (S: 644, C: 168, Q: 77) |

## Strategies Demonstrated

### Notebook 1 — Core Imputation Strategies (`Missing_Data_Strategies.ipynb`)

| Strategy | Applied to | Rationale |
|---|---|---|
| **Drop the column** | Cabin (77.1% missing) | Beyond ~60–70% missing, imputation manufactures data; dropping is the honest choice |
| **Mode imputation** | Embarked (2 missing) | Categorical with a dominant class — filled with 'S' (644 → 646) |
| **Median imputation** | Age (177 missing) | The boxplot showed outliers (max 80 vs median 28), so **median beats mean** — the mean (29.7) is pulled upward by the tail |
| **Arbitrary values (9999 / −1)** | Age (demo) | KDE plots overlay the original vs imputed distributions, making the distortion visible — arbitrary values create artificial spikes that flag "missingness" to tree models but wreck distance-based ones |

Plus a leakage-aware variant: a 75/25 train–test split (668/223) performed *before* imputation, so fill values are learned from training data only.

### Notebook 2 — KNN Imputation & Encoding (`KNN_Imputation_and_Encoding.ipynb`)

- **KNNImputer (k=5)** fills 133 missing training ages from each passenger's 5 nearest neighbours — preserving distribution shape rather than spiking the median
- **One-Hot Encoding** (`pd.get_dummies`) with reference categories dropped to avoid the dummy-variable trap
- **Label Encoding** two ways: `sklearn.LabelEncoder` and pandas `.cat.codes`

## Decision Guide — Which Strategy When?

- **>60–70% missing** → drop the column (Cabin)
- **Categorical, few gaps** → mode (Embarked)
- **Numeric with outliers** → median, not mean (Age)
- **Numeric, relationships matter** → KNN imputation (best fidelity, highest cost)
- **Tree models + informative missingness** → arbitrary value as a "missing" flag
- **Always** → split before imputing to prevent leakage

## Tech Stack

`Python` · `pandas` · `NumPy` · `scikit-learn` (KNNImputer, LabelEncoder, train_test_split) · `seaborn` / `matplotlib`

## Repository Structure

```
├── Missing_Data_Strategies.ipynb        # Drop / mode / median / arbitrary + KDE comparison
├── KNN_Imputation_and_Encoding.ipynb    # KNNImputer + One-Hot & Label Encoding
├── Titanic-Dataset.csv                  # Dataset (891 passengers)
└── README.md
```

## How to Run

```bash
pip install pandas numpy scikit-learn seaborn matplotlib
jupyter notebook
```

---

*Author: Kailas Wadje — MSc Data Science & AI, University of Liverpool*
