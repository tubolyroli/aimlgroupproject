import argparse
import pandas as pd
from data import load_and_process_data
from features import get_preprocessor
from models import get_lasso_pipeline, get_rf_pipeline
from evaluate import evaluate_and_save

def main():
    parser = argparse.ArgumentParser(description="Run UK Road Safety Analysis")
    parser.add_argument("--data_path", type=str, default="data", help="Path to folder containing csv files")
    parser.add_argument("--seed", type=int, default=69, help="Random seed for reproducibility")
    args = parser.parse_args()

    # 1. Load Data
    print("--- STEP 1: LOADING & SPLITTING DATA ---")
    X_train, X_test, y_train, y_test = load_and_process_data(args.data_path, args.seed)

    # 2. Setup Features
    print("--- STEP 2: SETUP PREPROCESSING ---")
    preprocessor = get_preprocessor(X_train)

    # 3. Train Baseline (Lasso)
    print("--- STEP 3: TRAINING LASSO LOGISTIC REGRESSION ---")
    lasso_model = get_lasso_pipeline(preprocessor, C=0.1) # C=0.1 based on analysis
    lasso_model.fit(X_train, y_train)

    # 4. Evaluate Lasso
    lasso_metrics = evaluate_and_save(lasso_model, X_test, y_test, "Lasso Logistic Regression")

    # 5. Train Random Forest (Optional, but good for comparison)
    print("--- STEP 5: TRAINING RANDOM FOREST ---")
    rf_model = get_rf_pipeline(preprocessor)
    rf_model.fit(X_train, y_train)

    # 6. Evaluate RF
    rf_metrics = evaluate_and_save(rf_model, X_test, y_test, "Random Forest")

    print("\n--- DONE ---")
    print("Check 'outputs/' folder for results.")

if __name__ == "__main__":
    main()