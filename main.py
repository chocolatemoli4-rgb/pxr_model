#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Clean and simple OpenADMET PXR pEC50 baseline.

What this version keeps:
  - official clean train/test csv
  - Morgan/ECFP4 fingerprint
  - RDKit 2D descriptors
  - ExtraTrees, RandomForest, Ridge
  - Tanimoto kNN
  - validation metrics
  - final full-train submission

What this version removes:
  - GNN / torch-geometric
  - mol2vec
  - ChEMBL/external pretraining
  - single-concentration/counter-assay features
  - complicated multi-stage blending

Run:
  python main.py

Outputs:
  outputs/validation_predictions.csv
  outputs/validation_metrics.csv
  outputs/model_weights.csv
  submissions/submission_clean_simple.csv
"""

from __future__ import annotations

import argparse
import json
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors

from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "outputs"
SUB_DIR = ROOT / "submissions"

TRAIN_CSV = DATA_DIR / "openadmet_train_clean.csv"
TEST_CSV = DATA_DIR / "openadmet_test_clean.csv"

ID_COL = "Molecule Name"
SMILES_COL = "canonical_smiles"
RAW_SMILES_COL = "SMILES"
TARGET_COL = "pEC50"

DESC_LIST = Descriptors._descList


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def canonicalize_smiles(smiles: str) -> str | None:
    try:
        mol = Chem.MolFromSmiles(str(smiles))
        if mol is None:
            return None
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return None


def pick_smiles_column(df: pd.DataFrame) -> str:
    if SMILES_COL in df.columns:
        return SMILES_COL
    if RAW_SMILES_COL in df.columns:
        return RAW_SMILES_COL
    raise ValueError(f"Cannot find SMILES column. Expected {SMILES_COL!r} or {RAW_SMILES_COL!r}.")


def clean_input(df: pd.DataFrame, is_train: bool) -> pd.DataFrame:
    df = df.copy()
    smiles_col = pick_smiles_column(df)
    if smiles_col != SMILES_COL:
        df[SMILES_COL] = df[smiles_col].map(canonicalize_smiles)

    if "is_valid_mol" in df.columns:
        df = df[df["is_valid_mol"] == True].copy()

    required = [ID_COL, RAW_SMILES_COL, SMILES_COL]
    if is_train:
        required.append(TARGET_COL)
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError("Missing columns: " + ", ".join(missing))

    subset = [SMILES_COL] + ([TARGET_COL] if is_train else [])
    df = df.dropna(subset=subset).reset_index(drop=True)
    return df


def calc_morgan_bits(smiles: str, radius: int, n_bits: int) -> np.ndarray | None:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    arr = np.zeros((n_bits,), dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def calc_rdkit2d(smiles: str) -> np.ndarray | None:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    values = []
    for _, func in DESC_LIST:
        try:
            values.append(float(func(mol)))
        except Exception:
            values.append(np.nan)
    return np.asarray(values, dtype=np.float32)


def build_features(df: pd.DataFrame, radius: int, n_bits: int) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    fps, descs, keep_idx = [], [], []

    for i, smi in enumerate(df[SMILES_COL].values):
        fp = calc_morgan_bits(smi, radius=radius, n_bits=n_bits)
        desc = calc_rdkit2d(smi)
        if fp is None or desc is None:
            continue
        fps.append(fp)
        descs.append(desc)
        keep_idx.append(i)

    if not keep_idx:
        raise RuntimeError("No valid molecules after feature generation.")

    df2 = df.iloc[keep_idx].reset_index(drop=True)
    X_fp = np.vstack(fps).astype(np.float32)
    X_desc = np.vstack(descs).astype(np.float32)
    return df2, X_fp, X_desc


def remove_constant_columns_fit(X: np.ndarray) -> np.ndarray:
    X0 = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X0.std(axis=0) > 1e-12


def make_X(X_fp: np.ndarray, X_desc: np.ndarray, keep_desc: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    if keep_desc is None:
        keep_desc = remove_constant_columns_fit(X_desc)
    X = np.hstack([X_fp, X_desc[:, keep_desc]]).astype(np.float32)
    X[~np.isfinite(X)] = np.nan
    return X, keep_desc


def tanimoto_knn_predict(train_fp: np.ndarray, y_train: np.ndarray, query_fp: np.ndarray, k: int = 8) -> np.ndarray:
    # train_fp/query_fp are binary numpy arrays.
    train_bool = train_fp.astype(bool)
    query_bool = query_fp.astype(bool)
    preds = []

    for q in query_bool:
        inter = np.logical_and(train_bool, q).sum(axis=1).astype(float)
        union = np.logical_or(train_bool, q).sum(axis=1).astype(float)
        sim = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)

        kk = min(k, len(y_train))
        idx = np.argpartition(-sim, kk - 1)[:kk]
        s = sim[idx]
        y = y_train[idx]

        if float(s.sum()) <= 1e-12:
            preds.append(float(np.mean(y_train)))
        else:
            preds.append(float(np.sum(y * s) / np.sum(s)))

    return np.asarray(preds, dtype=float)


def make_models(seed: int) -> dict:
    # One feature matrix, three ordinary regressors.
    return {
        "extratrees": make_pipeline(
            SimpleImputer(strategy="median"),
            ExtraTreesRegressor(
                n_estimators=600,
                random_state=seed,
                n_jobs=-1,
                max_features="sqrt",
                min_samples_leaf=2,
            ),
        ),
        "randomforest": make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestRegressor(
                n_estimators=400,
                random_state=seed,
                n_jobs=-1,
                max_features="sqrt",
                min_samples_leaf=2,
            ),
        ),
        "ridge": make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(with_mean=False),
            RidgeCV(alphas=np.logspace(-3, 3, 13)),
        ),
    }


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    baseline = np.full_like(y_true, np.mean(y_true), dtype=float)
    denom = np.sum(np.abs(y_true - baseline))

    out = {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
        "RAE": float(np.sum(np.abs(y_true - y_pred)) / denom) if denom > 0 else np.nan,
        "Spearman": float(spearmanr(y_true, y_pred).correlation),
        "Kendall": float(kendalltau(y_true, y_pred).correlation),
    }
    return out


def inverse_mae_weights(metric_df: pd.DataFrame) -> dict[str, float]:
    # Stable, simple weighting: better MAE gets larger weight.
    rows = metric_df[metric_df["model"] != "mean_blend"].copy()
    score = 1.0 / np.maximum(rows["MAE"].values.astype(float), 1e-8)
    score = score / score.sum()
    return dict(zip(rows["model"].values, score.astype(float)))


def weighted_blend(preds: dict[str, np.ndarray], weights: dict[str, float]) -> np.ndarray:
    out = None
    for name, w in weights.items():
        if name not in preds:
            continue
        part = np.asarray(preds[name], dtype=float) * float(w)
        out = part if out is None else out + part
    if out is None:
        raise RuntimeError("No predictions available for blend.")
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", default=str(TRAIN_CSV))
    parser.add_argument("--test_csv", default=str(TEST_CSV))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val_size", type=float, default=0.2)
    parser.add_argument("--n_bits", type=int, default=2048)
    parser.add_argument("--radius", type=int, default=2)
    parser.add_argument("--knn_k", type=int, default=8)
    args = parser.parse_args()

    seed_everything(args.seed)
    OUT_DIR.mkdir(exist_ok=True)
    SUB_DIR.mkdir(exist_ok=True)

    train_raw = pd.read_csv(args.train_csv)
    test_raw = pd.read_csv(args.test_csv)

    train_df = clean_input(train_raw, is_train=True)
    test_df = clean_input(test_raw, is_train=False)

    print(f"[data] train={len(train_df)} test={len(test_df)}")

    all_df = pd.concat([train_df, test_df], axis=0, ignore_index=True)
    all_df, X_fp_all, X_desc_all = build_features(all_df, radius=args.radius, n_bits=args.n_bits)

    n_train = len(train_df)
    train_df = all_df.iloc[:n_train].reset_index(drop=True)
    test_df = all_df.iloc[n_train:].reset_index(drop=True)
    X_fp_train_all = X_fp_all[:n_train]
    X_fp_test = X_fp_all[n_train:]
    X_desc_train_all = X_desc_all[:n_train]
    X_desc_test = X_desc_all[n_train:]

    y_all = train_df[TARGET_COL].values.astype(float)

    tr_idx, va_idx = train_test_split(
        np.arange(len(train_df)),
        test_size=args.val_size,
        random_state=args.seed,
        shuffle=True,
    )

    keep_desc = remove_constant_columns_fit(X_desc_train_all[tr_idx])
    X_all, _ = make_X(X_fp_train_all, X_desc_train_all, keep_desc)
    X_test, _ = make_X(X_fp_test, X_desc_test, keep_desc)

    X_tr, X_va = X_all[tr_idx], X_all[va_idx]
    y_tr, y_va = y_all[tr_idx], y_all[va_idx]

    val_preds = {}
    metric_rows = []

    print("[fit] validation models")
    for name, model in make_models(args.seed).items():
        model.fit(X_tr, y_tr)
        pred = model.predict(X_va).astype(float)
        val_preds[name] = pred
        row = {"model": name, **metrics(y_va, pred)}
        metric_rows.append(row)
        print(f"  {name:12s} MAE={row['MAE']:.4f} R2={row['R2']:.4f} Spearman={row['Spearman']:.4f}")

    pred_knn = tanimoto_knn_predict(X_fp_train_all[tr_idx], y_tr, X_fp_train_all[va_idx], k=args.knn_k)
    val_preds["tanimoto_knn"] = pred_knn
    row = {"model": "tanimoto_knn", **metrics(y_va, pred_knn)}
    metric_rows.append(row)
    print(f"  {'tanimoto_knn':12s} MAE={row['MAE']:.4f} R2={row['R2']:.4f} Spearman={row['Spearman']:.4f}")

    metric_df = pd.DataFrame(metric_rows)
    weights = inverse_mae_weights(metric_df)
    val_blend = weighted_blend(val_preds, weights)
    blend_row = {"model": "mean_blend", **metrics(y_va, val_blend)}
    metric_df = pd.concat([metric_df, pd.DataFrame([blend_row])], ignore_index=True)

    print("[blend weights]")
    for k, v in weights.items():
        print(f"  {k:12s} {v:.4f}")
    print(f"[validation blend] MAE={blend_row['MAE']:.4f} R2={blend_row['R2']:.4f} Spearman={blend_row['Spearman']:.4f}")

    metric_df.to_csv(OUT_DIR / "validation_metrics.csv", index=False)
    pd.DataFrame([weights]).to_csv(OUT_DIR / "model_weights.csv", index=False)

    val_out = train_df.iloc[va_idx][[ID_COL, RAW_SMILES_COL, TARGET_COL]].copy()
    val_out = val_out.rename(columns={TARGET_COL: "y_true"})
    for name, pred in val_preds.items():
        val_out[f"pred_{name}"] = pred
    val_out["pred_blend"] = val_blend
    val_out.to_csv(OUT_DIR / "validation_predictions.csv", index=False)

    # Final submission: refit on all official train data, then predict official test.
    print("[fit] full-train models for final submission")
    keep_desc_full = remove_constant_columns_fit(X_desc_train_all)
    X_train_full, _ = make_X(X_fp_train_all, X_desc_train_all, keep_desc_full)
    X_test_full, _ = make_X(X_fp_test, X_desc_test, keep_desc_full)

    test_preds = {}
    for name, model in make_models(args.seed).items():
        model.fit(X_train_full, y_all)
        test_preds[name] = model.predict(X_test_full).astype(float)

    test_preds["tanimoto_knn"] = tanimoto_knn_predict(X_fp_train_all, y_all, X_fp_test, k=args.knn_k)
    test_blend = weighted_blend(test_preds, weights)

    sub = test_df[[RAW_SMILES_COL, ID_COL]].copy()
    sub[TARGET_COL] = np.clip(test_blend, 2.0, 8.0)
    sub = sub[[RAW_SMILES_COL, ID_COL, TARGET_COL]]
    sub.to_csv(SUB_DIR / "submission_clean_simple.csv", index=False)

    detail = test_df[[ID_COL, RAW_SMILES_COL]].copy()
    for name, pred in test_preds.items():
        detail[f"pred_{name}"] = pred
    detail["pred_blend"] = test_blend
    detail.to_csv(OUT_DIR / "test_predictions.csv", index=False)

    print(f"[saved] {OUT_DIR / 'validation_metrics.csv'}")
    print(f"[saved] {OUT_DIR / 'validation_predictions.csv'}")
    print(f"[saved] {OUT_DIR / 'test_predictions.csv'}")
    print(f"[saved] {SUB_DIR / 'submission_clean_simple.csv'}")


if __name__ == "__main__":
    main()
