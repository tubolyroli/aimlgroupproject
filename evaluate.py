import pandas as pd
import json
import os
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score, classification_report

def evaluate_and_save(model, X_test, y_test, model_name, output_dir="outputs"):
    """
    Calculates metrics, prints a report, and saves results to disk.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"\n=== Evaluating {model_name} ===")
    
    # Predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred, pos_label=1)
    auc = roc_auc_score(y_test, y_proba)

    print(f"Accuracy: {acc:.4f}")
    print(f"Recall (Fatal): {rec:.4f}")
    print(f"ROC AUC: {auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, digits=3))

    # Save Metrics
    metrics = {
        "model": model_name,
        "accuracy": acc,
        "recall_fatal": rec,
        "roc_auc": auc
    }
    
    metrics_path = f"{output_dir}/metrics_{model_name.lower().replace(' ', '_')}.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"Metrics saved to {metrics_path}")

    # Save Predictions (CSV)
    results_df = pd.DataFrame({
        "actual": y_test,
        "predicted_class": y_pred,
        "predicted_proba_fatal": y_proba
    })
    
    csv_path = f"{output_dir}/predictions_{model_name.lower().replace(' ', '_')}.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"Predictions saved to {csv_path}")
    
    return metrics