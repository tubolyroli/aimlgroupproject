from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

def get_lasso_pipeline(preprocessor, C=0.1):
    """
    Returns a Pipeline with Lasso (L1) Logistic Regression.
    C=0.1 is chosen based on experimental tuning.
    class_weight='balanced' is critical for the fatal accident imbalance.
    """
    clf = LogisticRegression(
        penalty="l1",
        solver="liblinear",
        max_iter=2000,
        class_weight="balanced",
        C=C,
        random_state=69
    )
    
    return Pipeline(steps=[
        ("preprocess", preprocessor),
        ("clf", clf)
    ])

def get_rf_pipeline(preprocessor):
    """
    Returns a Pipeline with Random Forest.
    """
    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=68
    )
    
    return Pipeline(steps=[
        ("preprocess", preprocessor),
        ("clf", clf)
    ])