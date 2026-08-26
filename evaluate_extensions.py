"""Standalone evaluation for baselines + new GBR variants.

Run:  python evaluate_extensions.py

Compares Linear, Ridge-GS (from preprocessing_pipeline) and the two new GBR
pipelines, persists the best performer, and reports permutation importance.
"""
from preprocessing_pipeline import (
    create_preprocessor,
    linear_model_pipeline,
    ridge_gridCv_pipeline,
    evaluate_model,
    gbr_gridCv_pipeline,
    create_native_gbr_pipeline,
    load_training_data,
)
from model_registry import save_model

import numpy as np
from sklearn.inspection import permutation_importance


def compare_all_models():
    X_train, X_test, Y_train, Y_test = load_training_data()

    original_pre = create_preprocessor()
    models = {
        "Linear": linear_model_pipeline(original_pre),
        "Ridge-GS": ridge_gridCv_pipeline(original_pre),
        "GBR-GS": gbr_gridCv_pipeline(original_pre),
        "GBR-Native": create_native_gbr_pipeline(),
    }

    results = {}
    best_name, best_r2 = None, -np.inf
    for name, model in models.items():
        print(f"\n=== Training {name} ===")
        model.fit(X_train, Y_train)
        metrics = evaluate_model(model, X_test, Y_test)
        results[name] = metrics
        print(
            f"{name}: R2={metrics['r2']:.4f}  RMSE={metrics['rmse']:.4f}  "
            f"MAE={metrics['mae']:.4f}"
        )
        if metrics["r2"] > best_r2:
            best_r2, best_name = metrics["r2"], name

    print("\n=== Best model ===")
    print(f"{best_name} (R2={best_r2:.4f})")

    best_model = models[best_name]
    fitted = best_model.best_estimator_ if hasattr(best_model, "best_estimator_") else best_model
    save_model(fitted, f"best_{best_name.replace('-', '_').lower()}")

    print("\n=== Permutation importance (best model) ===")
    r = permutation_importance(
        fitted, X_test, Y_test, n_repeats=10, random_state=42, scoring="r2"
    )
    for feat, imp, std in sorted(
        zip(X_test.columns, r.importances_mean, r.importances_std),
        key=lambda t: t[1],
        reverse=True,
    ):
        print(f"  {feat:12s} {imp:.4f} +/- {std:.4f}")

    return results, best_name


if __name__ == "__main__":
    compare_all_models()
