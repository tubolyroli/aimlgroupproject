import argparse
from data import load_and_preprocess_data
from features import get_preprocessors
from models import train_lasso_cv, train_rf_cv
from evaluate import evaluate_model_metrics, plot_feature_importance, run_final_comparison

def main():
    parser = argparse.ArgumentParser(description="Run Full UK Road Safety Analysis")
    parser.add_argument("--data_path", type=str, default="data", help="Path to folder containing csv files")
    parser.add_argument("--seed", type=int, default=69, help="Random seed for reproducibility")
    args = parser.parse_args()

    # 1. Load & Split Data
    print("--- STEP 1: PROCESSING DATA ---")
    X_train, X_test, y_train, y_test = load_and_preprocess_data(args.data_path, args.seed)

    # 2. Setup Preprocessors
    print("--- STEP 2: SETUP PREPROCESSORS ---")
    preprocess_logit, preprocess_tree, num_feats, cat_feats = get_preprocessors(X_train)

    # 3. Train & Evaluate Lasso
    print("--- STEP 3: TRAINING LASSO (GridSearch) ---")
    lasso_grid = train_lasso_cv(X_train, y_train, preprocess_logit, seed=args.seed)
    y_pred_logit, y_proba_logit = evaluate_model_metrics(lasso_grid, X_test, y_test, "Lasso Logistic Regression")

    # 4. Train & Evaluate Random Forest
    print("--- STEP 4: TRAINING RANDOM FOREST (GridSearch) ---")
    # Note: RF uses specific seed=68 internally as per notebook, but CV uses args.seed
    rf_grid = train_rf_cv(X_train, y_train, preprocess_tree, cv_seed=args.seed)
    y_pred_rf, y_proba_rf = evaluate_model_metrics(rf_grid, X_test, y_test, "Random Forest")

    # 5. Feature Importance
    print("--- STEP 5: FEATURE IMPORTANCE ---")
    plot_feature_importance(rf_grid.best_estimator_, num_feats, cat_feats)

    # 6. Final Comparison
    print("--- STEP 6: FINAL COMPARISON ---")
    run_final_comparison(X_train, y_train, X_test, y_test,
                         y_pred_logit, y_proba_logit,
                         y_pred_rf, y_proba_rf)
    
    print("\n--- ANALYSIS COMPLETE ---")

if __name__ == "__main__":
    main()