import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer

def to_string(x):
    """Helper function to force-cast data to strings"""
    return x.astype(str)

def get_preprocessors(X_train):
    """
    Defines the preprocessing pipelines for Logistic Regression and Trees.
    Returns: preprocess_logit, preprocess_tree, numeric_features, categorical_features
    """
    numeric_features = X_train.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X_train.columns.difference(numeric_features).tolist()

    print(f"Numeric features: {len(numeric_features)}")
    print(f"Categorical features: {len(categorical_features)}")

    # 1. Numeric Pipeline
    num_logit_pipe = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    # 2. Categorical Pipeline
    cat_logit_pipe = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('cast_to_str', FunctionTransformer(to_string, validate=False)),
        ('encoder', OneHotEncoder(handle_unknown="ignore"))
    ])

    # Full Column Transformer for Logit
    preprocess_logit = ColumnTransformer(
        transformers=[
            ("num", num_logit_pipe, numeric_features),
            ("cat", cat_logit_pipe, categorical_features),
        ]
    )

    # Column Transformer for Trees (no scaling, just imputation)
    preprocess_tree = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy='mean'), numeric_features),
            ("cat", Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                ('cast_to_str', FunctionTransformer(to_string, validate=False)),
                ('encoder', OneHotEncoder(handle_unknown="ignore"))
            ]), categorical_features),
        ]
    )

    return preprocess_logit, preprocess_tree, numeric_features, categorical_features