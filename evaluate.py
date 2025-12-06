import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import os
from sklearn.metrics import classification_report, recall_score, roc_auc_score, accuracy_score
from sklearn.dummy import DummyClassifier

def evaluate_model_metrics(model_grid, X_test, y_test, model_name, output_dir="outputs"):
    """
    1. Prints best params and CV score (Notebook logic).
    2. Prints Test report (Notebook logic).
    3. SAVES metrics.json and predictions.csv (Assignment requirement).
    """
    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # --- Notebook Logic: Print Best Params ---
    print(f"\nBest {model_name} params:", model_grid.best_params_)
    print(f"Best CV recall ({model_name}):", model_grid.best_score_)

    best_model = model_grid.best_estimator_
    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]

    # --- Notebook Logic: Print Test Performance ---
    print(f"\n=== {model_name}: Test performance ===")
    print(classification_report(y_test, y_pred, digits=3))
    
    acc = accuracy_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    
    print(f"Test recall ({model_name}):", rec)

    # --- Assignment Logic: Save Outputs ---
    
    # 1. Save Metrics JSON
    metrics = {
        "model": model_name,
        "best_params": model_grid.best_params_,
        "best_cv_recall": model_grid.best_score_,
        "test_accuracy": acc,
        "test_recall_fatal": rec,
        "test_roc_auc": auc
    }
    safe_name = model_name.lower().replace(" ", "_")
    metrics_path = f"{output_dir}/metrics_{safe_name}.json"
    
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"   [Saved] Metrics to {metrics_path}")

    # 2. Save Predictions CSV
    pred_df = pd.DataFrame({
        "actual": y_test,
        "predicted_class": y_pred,
        "predicted_proba_fatal": y_proba
    })
    csv_path = f"{output_dir}/predictions_{safe_name}.csv"
    pred_df.to_csv(csv_path, index=False)
    print(f"   [Saved] Predictions to {csv_path}")
    
    return y_pred, y_proba

def plot_feature_importance(best_rf, numeric_features, categorical_features):
    """
    Aggregates and plots feature importance (Notebook Cell 17).
    """
    print("\n=== Feature importances (Aggregated by original variable) ===")
    best_preprocessor = best_rf.named_steps["preprocess"]
    best_rf_model = best_rf.named_steps["clf"]

    # 1. Get detailed feature names (Encoded)
    cat_pipeline = best_preprocessor.named_transformers_["cat"]
    ohe = cat_pipeline.named_steps["encoder"]
    cat_feature_names = ohe.get_feature_names_out(categorical_features)

    all_feature_names = list(numeric_features) + list(cat_feature_names)
    importances = best_rf_model.feature_importances_

    feat_imp_raw = pd.DataFrame({
        "encoded_feature": all_feature_names,
        "importance": importances
    })

    # Helper map function
    def map_to_original(encoded_name, numeric_list, categorical_list):
        if encoded_name in numeric_list:
            return encoded_name
        for cat in sorted(categorical_list, key=len, reverse=True):
            if encoded_name.startswith(f"{cat}_"):
                return cat
        return encoded_name

    feat_imp_raw["original_feature"] = feat_imp_raw["encoded_feature"].apply(
        lambda x: map_to_original(x, numeric_features, categorical_features)
    )

    feat_imp_grouped = (
        feat_imp_raw.groupby("original_feature")["importance"]
        .sum()
        .reset_index()
        .sort_values("importance", ascending=False)
    )

    print(feat_imp_grouped.head(20))

    # Plot
    top_k = 20
    top_feat = feat_imp_grouped.head(top_k).sort_values("importance", ascending=True)

    plt.figure(figsize=(10, 8))
    plt.barh(top_feat["original_feature"], top_feat["importance"])
    plt.xlabel("Total Feature Importance (Sum of One-Hot Parts)")
    plt.title(f"Top {top_k} Aggregated Random Forest Importances")
    plt.tight_layout()
    # plt.show() # Commented out so it doesn't block script execution if no UI
    plt.savefig("outputs/feature_importance.png") # Save instead of show
    print("   [Saved] Plot to outputs/feature_importance.png")

def run_final_comparison(X_train, y_train, X_test, y_test, 
                        y_pred_logit, y_proba_logit, 
                        y_pred_rf, y_proba_rf):
    """
    Runs baselines and prints the comparison table (Notebook Cell 18).
    Also saves the table to CSV.
    """
    # 1. Baseline: Majority Class
    y_pred_baseline = np.zeros_like(y_test)
    baseline_acc = accuracy_score(y_test, y_pred_baseline)
    baseline_recall_fatal = 0.0
    baseline_roc_auc = 0.5 

    print("\n=== BASELINE MODEL (Majority Rule) ===")
    print(f"Accuracy: {baseline_acc:.4f}")
    print(f"Recall (fatal): {baseline_recall_fatal:.4f}")
    print(f"ROC AUC: {baseline_roc_auc:.4f}")

    # 2. Baseline: Dummy Stratified
    dummy = DummyClassifier(strategy="stratified", random_state=69)
    dummy.fit(X_train, y_train)

    y_pred_dummy = dummy.predict(X_test)
    y_proba_dummy = dummy.predict_proba(X_test)[:, 1]

    dummy_acc = accuracy_score(y_test, y_pred_dummy)
    dummy_recall_fatal = recall_score(y_test, y_pred_dummy, pos_label=1)
    dummy_roc_auc = roc_auc_score(y_test, y_proba_dummy)

    print("\n=== DUMMY BASELINE (Random Stratified) ===")
    print(f"Accuracy: {dummy_acc:.4f}")
    print(f"Recall (fatal): {dummy_recall_fatal:.4f}")
    print(f"ROC AUC: {dummy_roc_auc:.4f}")

    # 3. Compile Metrics
    logit_acc = accuracy_score(y_test, y_pred_logit)
    logit_recall_fatal = recall_score(y_test, y_pred_logit, pos_label=1)
    logit_roc = roc_auc_score(y_test, y_proba_logit)

    rf_acc = accuracy_score(y_test, y_pred_rf)
    rf_recall_fatal = recall_score(y_test, y_pred_rf, pos_label=1)
    rf_roc = roc_auc_score(y_test, y_proba_rf)

    # 4. Final Table
    comparison_df = pd.DataFrame({
        "Model": [
            "Baseline (Majority Rule)",
            "Baseline (Random Stratified)",
            "Lasso Logistic Regression",
            "Random Forest",
        ],
        "Accuracy": [baseline_acc, dummy_acc, logit_acc, rf_acc],
        "Recall (Fatal)": [baseline_recall_fatal, dummy_recall_fatal, logit_recall_fatal, rf_recall_fatal],
        "ROC AUC": [baseline_roc_auc, dummy_roc_auc, logit_roc, rf_roc],
    })

    print("\n=== FINAL MODEL COMPARISON (Test Set) ===")
    print(comparison_df.round(4))
    
    # Save the table too (Good practice)
    comparison_df.round(4).to_csv("outputs/final_model_comparison.csv", index=False)
    print("   [Saved] Comparison table to outputs/final_model_comparison.csv")