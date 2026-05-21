# OpenADMET PXR pEC50 Simple Baseline

This repository contains a clean and simple baseline pipeline for the OpenADMET PXR activity prediction task.  
The goal is to predict **pEC50** values for molecules from their chemical structures.

The pipeline is intentionally kept compact:

- read clean train/test CSV files
- generate molecular features from SMILES
- train several classical regression models
- evaluate them on a validation split
- blend model predictions using validation MAE
- retrain on the full training set
- generate a final submission CSV

The main entry point is:

```bash
python main.py
```

---

## 1. Project Structure

Recommended folder layout:

```text
project_root/
  main.py
  requirements.txt
  README.md

  data/
    openadmet_train_clean.csv
    openadmet_test_clean.csv

  outputs/
    validation_metrics.csv
    validation_predictions.csv
    model_weights.csv
    test_predictions.csv

  submissions/
    submission_clean_simple.csv
```

The `outputs/` and `submissions/` folders are created automatically when the script runs.

---

## 2. Task Description

The script trains a regression model to predict molecular **pEC50** values for PXR activity.

The final submission file has three columns:

```text
SMILES,Molecule Name,pEC50
```

where:

- `SMILES` is the molecule structure string from the test set
- `Molecule Name` is the molecule identifier
- `pEC50` is the predicted activity value

The final predictions are clipped to the range:

```text
2.0 <= pEC50 <= 8.0
```

---

## 3. Input Files

By default, the script expects the following files:

```text
data/openadmet_train_clean.csv
data/openadmet_test_clean.csv
```

You can also specify custom paths with command-line arguments.

### 3.1 Training CSV

The training CSV should contain these columns:

| Column | Required | Meaning |
|---|---:|---|
| `Molecule Name` | Yes | Molecule ID |
| `SMILES` | Yes | Original SMILES string |
| `canonical_smiles` | Recommended | Canonical SMILES string |
| `pEC50` | Yes | Target value |

If `canonical_smiles` is missing, the script will try to create it from `SMILES` using RDKit.

### 3.2 Test CSV

The test CSV should contain these columns:

| Column | Required | Meaning |
|---|---:|---|
| `Molecule Name` | Yes | Molecule ID |
| `SMILES` | Yes | Original SMILES string |
| `canonical_smiles` | Recommended | Canonical SMILES string |

If `canonical_smiles` is missing, the script will try to create it from `SMILES`.

### 3.3 Optional Column

If the input CSV contains:

```text
is_valid_mol
```

then only rows where `is_valid_mol == True` are kept.

---

## 4. Installation

Create or activate your Python environment, then install dependencies:

```bash
pip install -r requirements.txt
```

A minimal `requirements.txt` should include:

```text
numpy
pandas
scipy
scikit-learn
rdkit
```

If `rdkit` cannot be installed with pip in your environment, try:

```bash
pip install rdkit-pypi
```

On some systems, RDKit installation may depend on your Python version and platform.

---

## 5. Quick Start

Put the train and test CSV files into the `data/` folder:

```text
data/openadmet_train_clean.csv
data/openadmet_test_clean.csv
```

Then run:

```bash
python main.py
```

After the run finishes, the final submission will be saved to:

```text
submissions/submission_clean_simple.csv
```

---

## 6. Running with Custom File Paths

You can pass custom train/test files:

```bash
python main.py ^
  --train_csv path/to/train.csv ^
  --test_csv path/to/test.csv
```

On Linux/macOS:

```bash
python main.py \
  --train_csv path/to/train.csv \
  --test_csv path/to/test.csv
```

---

## 7. Command-Line Arguments

| Argument | Default | Meaning |
|---|---:|---|
| `--train_csv` | `data/openadmet_train_clean.csv` | Path to training CSV |
| `--test_csv` | `data/openadmet_test_clean.csv` | Path to test CSV |
| `--seed` | `42` | Random seed |
| `--val_size` | `0.2` | Validation split ratio |
| `--n_bits` | `2048` | Morgan fingerprint bit length |
| `--radius` | `2` | Morgan fingerprint radius |
| `--knn_k` | `8` | Number of neighbors for Tanimoto kNN |

Example:

```bash
python main.py --seed 123 --val_size 0.2 --n_bits 2048 --radius 2 --knn_k 8
```

---

## 8. Feature Generation

The script uses two types of molecular features.

### 8.1 Morgan Fingerprint

Morgan fingerprints are generated with RDKit:

```text
radius = 2
n_bits = 2048
```

This is similar to ECFP4-style circular fingerprints.

The fingerprint is a binary vector representing local molecular substructures.

### 8.2 RDKit 2D Descriptors

The script also computes RDKit molecular descriptors using:

```python
Descriptors._descList
```

These descriptors include common 2D molecular properties, such as molecular weight, logP-like descriptors, polar surface area-related descriptors, atom counts, ring counts, and other structural descriptors.

Constant descriptor columns are removed before model training.

---

## 9. Models

The script trains four model branches.

### 9.1 ExtraTreesRegressor

A tree ensemble model that is often strong for molecular fingerprint features.

Main settings:

```text
n_estimators = 600
max_features = sqrt
min_samples_leaf = 2
```

### 9.2 RandomForestRegressor

Another tree ensemble model.

Main settings:

```text
n_estimators = 400
max_features = sqrt
min_samples_leaf = 2
```

### 9.3 RidgeCV

A linear regression model with L2 regularization.

The script tests several alpha values:

```text
1e-3 to 1e3
```

The Ridge model uses:

```text
SimpleImputer
StandardScaler
RidgeCV
```

### 9.4 Tanimoto kNN

A similarity-based model using Morgan fingerprints.

For every validation or test molecule, the script finds the most similar training molecules using Tanimoto similarity, then predicts a weighted average of their pEC50 values.

Default:

```text
k = 8
```

If all similarities are near zero, the prediction falls back to the training-set mean.

---

## 10. Validation Strategy

The script uses a random train/validation split:

```text
train: 80%
validation: 20%
```

The split is controlled by:

```text
--val_size
--seed
```

Validation is used to:

1. estimate model performance
2. compute model blending weights
3. save validation predictions for inspection

The validation split is not used for final model training.  
After validation, the final models are retrained on the full training dataset.

---

## 11. Metrics

The script reports these metrics:

| Metric | Meaning |
|---|---|
| `MAE` | Mean Absolute Error |
| `RMSE` | Root Mean Squared Error |
| `R2` | Coefficient of determination |
| `RAE` | Relative Absolute Error |
| `Spearman` | Spearman rank correlation |
| `Kendall` | Kendall rank correlation |

The metrics are saved to:

```text
outputs/validation_metrics.csv
```

---

## 12. Model Blending

After validation, each model receives a weight based on its validation MAE.

The weighting rule is:

```text
weight_i = (1 / MAE_i) / sum(1 / MAE_j)
```

So models with lower validation MAE receive larger weights.

The weights are saved to:

```text
outputs/model_weights.csv
```

The final blended prediction is:

```text
prediction = sum(model_prediction_i * weight_i)
```

---

## 13. Output Files

### 13.1 validation_metrics.csv

Path:

```text
outputs/validation_metrics.csv
```

Contains validation metrics for each model and the blended model.

Example columns:

```text
model,MAE,RMSE,R2,RAE,Spearman,Kendall
```

---

### 13.2 validation_predictions.csv

Path:

```text
outputs/validation_predictions.csv
```

Contains validation-set ground truth and model predictions.

Example columns:

```text
Molecule Name
SMILES
y_true
pred_extratrees
pred_randomforest
pred_ridge
pred_tanimoto_knn
pred_blend
```

This file is useful for error analysis.

---

### 13.3 model_weights.csv

Path:

```text
outputs/model_weights.csv
```

Contains the blending weights used for final prediction.

Example columns:

```text
extratrees,randomforest,ridge,tanimoto_knn
```

---

### 13.4 test_predictions.csv

Path:

```text
outputs/test_predictions.csv
```

Contains detailed test-set predictions from each model branch and the final blend.

Example columns:

```text
Molecule Name
SMILES
pred_extratrees
pred_randomforest
pred_ridge
pred_tanimoto_knn
pred_blend
```

This file is useful for comparing models and checking whether one model is making unusual predictions.

---

### 13.5 submission_clean_simple.csv

Path:

```text
submissions/submission_clean_simple.csv
```

This is the final submission file.

Columns:

```text
SMILES
Molecule Name
pEC50
```

---

## 14. Full Pipeline

The script follows this workflow:

```text
1. Read train/test CSV
2. Clean input rows
3. Canonicalize SMILES if needed
4. Build Morgan fingerprints
5. Build RDKit 2D descriptors
6. Split training data into train/validation
7. Train ExtraTrees, RandomForest, Ridge
8. Run Tanimoto kNN
9. Evaluate all models on validation set
10. Compute inverse-MAE blend weights
11. Save validation outputs
12. Refit models on all training data
13. Predict test data
14. Blend test predictions
15. Clip final pEC50 to [2.0, 8.0]
16. Save final submission
```

---

## 15. Reproducibility

The script sets random seeds for:

```text
Python random
NumPy random
scikit-learn model random_state
```

Default seed:

```text
42
```

To reproduce the same validation split and model outputs, use the same input files and the same seed:

```bash
python main.py --seed 42
```

---

## 16. Notes on Data Cleaning

The script does not perform aggressive data correction.

It only does basic cleaning:

- checks required columns
- creates `canonical_smiles` if missing
- removes invalid molecules during feature generation
- drops rows with missing SMILES
- drops training rows with missing `pEC50`
- optionally filters by `is_valid_mol == True`

The script assumes the input files are already reasonably clean.

---

## 17. Troubleshooting

### 17.1 Cannot find SMILES column

Error example:

```text
Cannot find SMILES column. Expected 'canonical_smiles' or 'SMILES'.
```

Check that your CSV contains at least one of:

```text
canonical_smiles
SMILES
```

---

### 17.2 Missing columns

Error example:

```text
Missing columns: Molecule Name, pEC50
```

Check that the training CSV contains:

```text
Molecule Name
SMILES
pEC50
```

The test CSV should contain:

```text
Molecule Name
SMILES
```

---

### 17.3 RDKit installation problem

Try:

```bash
pip install rdkit-pypi
```

or use a Python version where RDKit wheels are available.

---

### 17.4 Script runs but submission has fewer rows than expected

This usually means some molecules failed RDKit parsing or were filtered out by `is_valid_mol`.

Check:

```text
[data] train=... test=...
```

printed at the start of the run.

Also check whether your CSV contains an `is_valid_mol` column.

---

### 17.5 Validation score changes between runs

Make sure the same seed is used:

```bash
python main.py --seed 42
```

Also make sure the input CSV files have not changed.

---

## 18. Suggested Next Steps

This script is designed as a stable baseline. Good next steps include:

- inspect `validation_predictions.csv` for high-error molecules
- compare per-model predictions in `test_predictions.csv`
- try scaffold-based or analog-based validation splits
- tune `knn_k`, `n_bits`, and model hyperparameters
- add carefully validated molecular features only when they improve validation reliability

---

## 19. License and Usage

This code is provided as a simple research baseline for molecular activity prediction.  
Before submitting predictions to any challenge or benchmark, make sure your input data and submission format follow the official rules.
