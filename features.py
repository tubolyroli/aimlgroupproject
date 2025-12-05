import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer

def to_string(x):
    """Helper to force-cast data to strings to prevent mixed-type errors in OneHotEncoder"""
    return x.astype(str)

def get_preprocessor(X_train):
    """
    Returns a ColumnTransformer that:
    1. Imputes and scales numeric features.
    2. Imputes and OneHotEncodes categorical features.
    """
    # Identify feature types
    numeric_features = X_train.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X_train.columns.difference(numeric_features).tolist()

    print(f"Preprocessing: {len(numeric_features)} numeric, {len(categorical_features)} categorical.")

    # Numeric Pipeline
    num_pipe = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    # Categorical Pipeline
    cat_pipe = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('cast_to_str', FunctionTransformer(to_string, validate=False)),
        ('encoder', OneHotEncoder(handle_unknown="ignore"))
    ])

    # Combine
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipe, numeric_features),
            ("cat", cat_pipe, categorical_features),
        ]
    )
    
    return preprocessor