from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, GridSearchCV

def train_lasso_cv(X_train, y_train, preprocessor, seed=69):
    """
    Runs GridSearchCV for Lasso Logistic Regression.
    """
    logit = LogisticRegression(
        penalty="l1",
        solver="liblinear",
        max_iter=2000,
        class_weight="balanced"
    )

    logit_pipeline = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("clf", logit),
    ])

    logit_param_grid = {
        "clf__C": [0.001, 0.01, 0.1, 1.0],
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    logit_grid = GridSearchCV(
        estimator=logit_pipeline,
        param_grid=logit_param_grid,
        scoring="recall",
        n_jobs=-1,
        cv=cv,
        verbose=1,
    )

    print("\n=== Fitting Lasso (L1) Logistic Regression with CV ===")
    logit_grid.fit(X_train, y_train)
    return logit_grid

def train_rf_cv(X_train, y_train, preprocessor, cv_seed=69, rf_seed=68):
    """
    Runs GridSearchCV for Random Forest.
    """
    rf = RandomForestClassifier(
        random_state=rf_seed,
        class_weight="balanced_subsample",
        n_jobs=-1,
    )

    rf_pipeline = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("clf", rf),
    ])

    rf_param_grid = {
        "clf__n_estimators": [200],
        "clf__max_depth": [10, 20, 30],
        "clf__min_samples_leaf": [1, 5, 10],
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=cv_seed)

    rf_grid = GridSearchCV(
        estimator=rf_pipeline,
        param_grid=rf_param_grid,
        scoring="recall",
        n_jobs=-1,
        cv=cv,
        verbose=1,
    )

    print("\n=== Fitting Random Forest with CV ===")
    rf_grid.fit(X_train, y_train)
    return rf_grid