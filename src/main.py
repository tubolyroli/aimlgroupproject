# %%
#### Imports ####
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import classification_report, roc_auc_score, accuracy_score, recall_score
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

# %%
#### Load raw data ####

try:
    root = Path(__file__).resolve().parent
except NameError:
    root = Path.cwd() # works if you run it as a script and in notebooks/VS Code too

data_dir = root.parent / "data" # from src folder goes to main

collision_raw = pd.read_csv(data_dir / "collision.csv", low_memory=False)
casualty_raw = pd.read_csv(data_dir / "casualty.csv", low_memory=False)
vehicle_raw = pd.read_csv(data_dir / "vehicle.csv",  low_memory=False)
# to ensure no mixed types, low_memory=False, it does not process the files in chunks

# Number of rows and columns in our 3 csv-s
print("collision_raw:", collision_raw.shape)
print("casualty_raw:", casualty_raw.shape)
print("vehicle_raw:", vehicle_raw.shape)

# %%
#### We construct a binary target Y from collision_severity (where 1 = fatal, 2 = serious, 3 = slight) ####

y_collision = (collision_raw["collision_severity"] == 1).astype(int) # transform the boolean values to 0 and 1
collision_raw["target_fatal"] = y_collision # create a new column with the dependent variable

print("\nTarget distribution (0=non-fatal, 1=fatal):")
print(y_collision.value_counts(normalize=True)) # proportion of fatal and non-fatal collisions (~1.5% fatal)

# %%
#### Columns to DROP (IDs, constants, duplicates, leakage) ####

## From collision.csv
cols_to_drop_collision = [
    # IDs and constants (we keep collision_index for merging)
    "collision_year",
    "collision_ref_no",
    "location_easting_osgr",
    "location_northing_osgr",
    "local_authority_district",
    "local_authority_ons_district",
    "local_authority_highway",
    "local_authority_highway_current",
    # historic/duplicate coding
    "junction_detail_historic",
    "pedestrian_crossing_human_control_historic",
    "pedestrian_crossing_physical_facilities_historic",
    "carriageway_hazards_historic",
    # location has extremely many distinct values so we drop it
    "lsoa_of_accident_location",
    # other outcome variants (we already used collision_severity)
    "enhanced_severity_collision",
    "collision_injury_based",
    "collision_adjusted_severity_serious",
    "collision_adjusted_severity_slight",
    # we will use collision_severity only via target_fatal, so we drop that too
    "collision_severity",
]

## From casualty.csv
cols_to_drop_casualty = [
    # IDs and constants (we keep collision_index for merging)
    "collision_year",
    "collision_ref_no",
    "casualty_reference",
    # location has extremely many distinct values so we drop it
    "lsoa_of_casualty",
    # severity/outcome features
    "casualty_severity",
    "enhanced_casualty_severity",
    "casualty_injury_based",
    "casualty_adjusted_severity_serious",
    "casualty_adjusted_severity_slight",
    # age: we keep the banded version instead
    "age_of_casualty",
]

## From vehicle.csv
cols_to_drop_vehicle = [
    # IDs and constants (we keep collision_index for merging)
    "collision_year",
    "collision_ref_no",
    # historic/duplicate coding
    "vehicle_manoeuvre_historic",
    "vehicle_location_restricted_lane_historic",
    "journey_purpose_of_driver_historic",
    # location has extremely many distinct values so we drop it
    "lsoa_of_driver",
    # too many categories for a baseline model, we drop it
    "generic_make_model",
    # age: we keep the banded version instead
    "age_of_driver",
]

# %%
#### Create cleaned tables ####

collision_feat = collision_raw.drop(columns=cols_to_drop_collision)
casualty_feat = casualty_raw.drop(columns=cols_to_drop_casualty)
vehicle_feat = vehicle_raw.drop(columns=cols_to_drop_vehicle)

# Number of rows and columns in our 3 cleaned csv-s
print("collision_feat:", collision_feat.shape)
print("casualty_feat:", casualty_feat.shape)
print("vehicle_feat:", vehicle_feat.shape)


# %%
#### Build collision-level modelling table ####
# Base unit: ONE ROW PER COLLISION (collision_index)
# We aggregate casualty and vehicle info to collision level first

# 1) Aggregate casualty-level info to collision level
casualty_agg = (
    casualty_feat
    .groupby("collision_index")
    .agg( # creating new, summarizing columns for the casualty aggregation
        n_casualties=("casualty_class", "size"), # number of casualties
        n_pedestrians=("casualty_class", lambda s: (s == 3).sum()), # number of pedestrians
        mean_casualty_age_band=("age_band_of_casualty", "mean"), # average age band of casualties in the collision
        mean_casualty_imd=("casualty_imd_decile", "mean"), # average deprivation decile of casualties in the collision
        max_casualty_distance_band=("casualty_distance_banding", "max"), 
        # maximum distance band over casualties in the collision-"worst distance"
    )
    .reset_index() # turns collision_index back into a regular column instead of an index
)

print(casualty_agg.shape)

# 2) Aggregate vehicle-level info to collision level
vehicle_agg = (
    vehicle_feat
    .groupby("collision_index")
    .agg(
        n_vehicles=("vehicle_reference", "size"), # number of vehicles
        mean_vehicle_age=("age_of_vehicle", "mean"), # average of age of vehicles involved in the casualties of the collision
        mean_engine_cc=("engine_capacity_cc", "mean"), # average of engine size of vehicles involved in the casualties of the collision
        share_left_hand_drive=("vehicle_left_hand_drive", lambda s: (s == 1).mean()), # fraction of vehicles that are left-hand drive
        share_escooter=("escooter_flag", lambda s: (s == 1).mean()), # share of vehicles that are e-scooters
    )
    .reset_index() # turns collision_index back into a regular column instead of an index
)

print(vehicle_agg.shape)

# 3) Merge everything to collision level
# Start from collision_feat (1 row per collision), then add casualty + vehicle aggregates.
model_df = (
    collision_feat
    .merge(casualty_agg, on="collision_index", how="left")
    .merge(vehicle_agg, on="collision_index", how="left")
)

print(model_df.shape)

# %%
#### We need to transform time: from HH:MM to useful and more interpretable numerical/categorical features ####

# Parse time column where the format is 'HH:MM'
model_df["time_parsed"] = pd.to_datetime(model_df["time"], format="%H:%M", errors="coerce")
model_df["hour_of_day"] = model_df["time_parsed"].dt.hour

def time_band(h): # function to define categories of time
    if pd.isna(h):
        return "unknown"
    h = int(h)
    if 0 <= h < 6:
        return "night" # 00:00–05:59
    elif 6 <= h < 10:
        return "morning_peak" # 06:00–09:59
    elif 10 <= h < 16:
        return "daytime" # 10:00–15:59
    elif 16 <= h < 20:
        return "evening_peak" # 16:00–19:59
    else:
        return "late_evening" # 20:00–23:59

model_df["time_band"] = model_df["hour_of_day"].apply(time_band)

# %%
#### We need to transform date too: from YYYY-MM-DD to numeric features ####

model_df["date_parsed"] = pd.to_datetime(model_df["date"], format="%d/%m/%Y", errors="coerce") # it is NaT if there is an error

model_df["day_of_week"] = model_df["date_parsed"].dt.dayofweek # 0=Monday, ..., 6=Sunday
model_df["month"] = model_df["date_parsed"].dt.month # 1-12
model_df["is_weekend"] = model_df["day_of_week"].isin([5, 6]).astype(int) # 1 if the day is Saturday/Sunday, 0 otherwise

# We drop raw minute-level time/date to avoid huge one-hot encodings later
model_df = model_df.drop(columns=["time", "time_parsed", "date", "date_parsed"])

# %%
#### Now we split into X (features) and y (target) ####

# Our target variable is collision-level fatal indicator
y = model_df["target_fatal"]

# Let's drop target and pure IDs from features
X = model_df.drop(
    columns=[
        "target_fatal",
        "collision_index",
        "vehicle_reference",
    ],
    errors="ignore",
) # errors="ignore": if columns wouldn't exist just ignore them

print("X:", X.shape)
print("y:", y.shape)

# %%
#### Train-test split ####

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=69
) # 80-20% split, stratify=y keeps the same class proportion (fatal/non-fatal) in train and test set

print("X_train:", X_train.shape, " y_train:", y_train.shape)
print("X_test: ", X_test.shape, " y_test: ", y_test.shape)

# %%
#### Preprocessing some definitions ####

numeric_features = X.select_dtypes(include=["number", "bool"]).columns.tolist() # list of numeric features
categorical_features = X.columns.difference(numeric_features).tolist() # everything else: list of categorical features

print(len(numeric_features)) # number of num features
print(len(categorical_features)) # number of cat features

# For logistic regression: scale numeric, one-hot encode categorical features
preprocess_logit = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ]
) # OneHotEncoder is needed to feed the categorical variables to the LR 

# For our decision tree model: pass numeric, one-hot encode categorical variables
preprocess_tree = ColumnTransformer(
    transformers=[
        ("num", "passthrough", numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ]
) # OneHotEncoder is needed to feed the categorical variables to the decision tree

# %%
#### We chose a LASSO Logistic Regression (L1) + 5-fold cross-validation ####
# LASSO to zero out unnecessary features instead of ridge based on our discussion with prof. Wachs

# Lasso = L1 penalty
logit = LogisticRegression(
    penalty="l1", # lasso: can drive some coefficients exactly to zero
    solver="liblinear", # supports L1 + class_weight for binary classification
    max_iter=2000, # allow enough iterations to converge
    class_weight="balanced", # reweights classes inversely to their frequency; important for rare fatal events
 )

logit_pipeline = Pipeline( # treat preprocessing + classifier as one model
    steps=[
        ("preprocess", preprocess_logit), # StandardScaler + OneHotEncoder (defined earlier)
        ("clf", logit), # classifier
    ]
) 

# Tune regularization strength C (smaller C means stronger regularization)
logit_param_grid = {
    "clf__C": [0.01, 0.1, 1.0],  # inverse of regularization strength (smaller C -> stronger regularization, sparser model)
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=69)  # cross-validation setup

logit_grid = GridSearchCV(
    estimator=logit_pipeline, # the full pipeline we are tuning
    param_grid=logit_param_grid, # dictionary of hyperparameters to search over
    scoring="roc_auc", # instead of accuracy, it is better for our data (imbalanced)
    n_jobs=-1, # all CPU
    cv=cv, # StratifiedKFold from earlier
    verbose=1,
)

print("\n=== Fitting Lasso (L1) Logistic Regression with CV (collision-level) ===")
logit_grid.fit(X_train, y_train)  # try each C and run the 5-fold CV, compute roc_auc and pick the best

print("\nBest lasso-logit params:", logit_grid.best_params_)
print("Best CV ROC AUC (lasso-logit):", logit_grid.best_score_)

best_logit = logit_grid.best_estimator_ # final chosen logistic model
y_pred_logit = best_logit.predict(X_test) # vector of hard class labels (fatal or non-fatal)
y_proba_logit = best_logit.predict_proba(X_test)[:, 1] # compute metrics (e.g. ROC AUC)

print("\n=== Lasso Logistic Regression: Test performance ===")
print(classification_report(y_test, y_pred_logit, digits=3))
print("Test ROC AUC (lasso-logit):", roc_auc_score(y_test, y_proba_logit))

# %%
#### LASSO-based feature selection ####

# 1) Extract trained preprocessor + classifier from the best LASSO pipeline
best_logit_pipeline = best_logit
best_preprocessor_logit = best_logit_pipeline.named_steps["preprocess"]
lasso_clf = best_logit_pipeline.named_steps["clf"]

# 2) Reconstruct feature names AFTER preprocessing
# Numeric features: passed through StandardScaler (so 1/1 mapping)
num_feature_names_logit = numeric_features

# Categorical features: expanded by OneHotEncoder, we need to get them back
ohe_logit = best_preprocessor_logit.named_transformers_["cat"]
cat_feature_names_logit = ohe_logit.get_feature_names_out(categorical_features)

# Combined feature space: [scaled numerics] + [one-hot categoricals]
all_feature_names_logit = list(num_feature_names_logit) + list(cat_feature_names_logit)

# 3) Get LASSO coefficients (for the positive class)
coefs = lasso_clf.coef_.ravel()  # shape (n_features,)

lasso_coef_df = (
    pd.DataFrame({
        "feature": all_feature_names_logit,
        "coef": coefs
    })
    .assign(abs_coef=lambda df: df["coef"].abs())
)

# 4) Select effectively non-zero coefficients
# we won't do exact zero, calculating with tiny numerical noise
eps = 1e-6
selected_lasso = lasso_coef_df[lasso_coef_df["abs_coef"] > eps].copy()
selected_lasso = selected_lasso.sort_values("abs_coef", ascending=False)

print("\n=== LASSO feature selection results ===")
print(f"Total preprocessed features: {lasso_coef_df.shape[0]}")
print(f"Selected (non-zero) features: {selected_lasso.shape[0]}")
print("\nTop 30 selected features by |coef|:")
print(selected_lasso.head(30))

# %%
#### LASSO-based feature selection and refitting a simpler logistic model ####

# 1) Extract trained preprocessor + LASSO classifier from the best pipeline
best_logit_pipeline = best_logit
preprocessor_logit = best_logit_pipeline.named_steps["preprocess"]
lasso_clf = best_logit_pipeline.named_steps["clf"]

# 2) Transform X_train and X_test with the fitted preprocessor
X_train_trans = preprocessor_logit.transform(X_train)
X_test_trans = preprocessor_logit.transform(X_test)

# 3) Get LASSO coefficients and select non-zero ones
coefs = lasso_clf.coef_.ravel()        # shape (n_features_after_preprocessing,)
nonzero_idx = np.where(coefs != 0)[0]  # indices of selected features

print(f"\nLASSO selected {len(nonzero_idx)} out of {len(coefs)} preprocessed features.")

# 4) Reduce the transformed feature matrices to the selected columns only
X_train_sel = X_train_trans[:, nonzero_idx]
X_test_sel = X_test_trans[:, nonzero_idx]

# 5) Refit a simpler logistic regression on the selected features (L2-regularized)
simple_logit = LogisticRegression(
    penalty="l2",
    solver="liblinear",
    max_iter=2000,
    class_weight="balanced",
)

simple_logit.fit(X_train_sel, y_train)

y_pred_simple = simple_logit.predict(X_test_sel)
y_proba_simple = simple_logit.predict_proba(X_test_sel)[:, 1]

print("\n=== Logistic regression on LASSO-selected features ===")
print(classification_report(y_test, y_pred_simple, digits=3))
print("Test ROC AUC (simple logit on selected features):", roc_auc_score(y_test, y_proba_simple))

# %%
#### Random Forest + CV ####

rf = RandomForestClassifier(
    random_state=68,
    class_weight="balanced_subsample", # rebalance classes when building each tree; important because fatals are rare
    n_jobs=-1, # all CPU
)

rf_pipeline = Pipeline(
    steps=[
        ("preprocess", preprocess_tree),
        ("clf", rf),
    ]
)

rf_param_grid = {
    "clf__n_estimators": [200], # no. of trees in the forest, we fix it at 200
    # (decent-size forest, usually stable, without wasting too much computational power)
    "clf__max_depth": [10, 20, 30], # maximum depth each individual tree can grow to
    # if smaller, strongly regularized -> less overfitting
    "clf__min_samples_leaf": [1, 5, 10], # minimum number of samples allowed in a leaf node
    # if smaller, bigger flexibility, highest overfitting risk
}  # hyperparameters chosen based on previous results and computational limits

rf_grid = GridSearchCV(
    estimator=rf_pipeline, # the full pipeline we are tuning
    param_grid=rf_param_grid, # dictionary of hyperparameters to search over
    scoring="roc_auc", # instead of accuracy, it is better for our data (imbalanced)
    n_jobs=-1, # all CPU
    cv=cv, # StratifiedKFold from earlier
    verbose=1,
)

print("\n=== Fitting Random Forest with CV (collision-level) ===")
rf_grid.fit(X_train, y_train) # try each hyperparameter and run the 5-fold CV, compute roc_auc and pick the best

print("\nBest random forest params:", rf_grid.best_params_)
print("Best CV ROC AUC (random forest):", rf_grid.best_score_)

best_rf = rf_grid.best_estimator_ # final chosen random forest model
y_pred_rf = best_rf.predict(X_test) # vector of hard class labels (fatal or non-fatal)
y_proba_rf = best_rf.predict_proba(X_test)[:, 1] # compute metrics (e.g. ROC AUC)

print("\n=== Random Forest: Test performance ===")
print(classification_report(y_test, y_pred_rf, digits=3))
print("Test ROC AUC (RF):", roc_auc_score(y_test, y_proba_rf))

# %%
#### Feature importances from best Random Forest + visualization ####

print("\n=== Top feature importances from Random Forest (collision-level) ===")

best_preprocessor = best_rf.named_steps["preprocess"] 
best_rf_model = best_rf.named_steps["clf"]
# access the random forest pieces' names

# Numeric feature names (pass-through)
num_feature_names = numeric_features

# Categorical feature names from the OneHotEncoder
ohe = best_preprocessor.named_transformers_["cat"]
cat_feature_names = ohe.get_feature_names_out(categorical_features)

# Combined in the same order as ColumnTransformer output: numerical then categorical
all_feature_names = list(num_feature_names) + list(cat_feature_names)

importances = best_rf_model.feature_importances_

feat_imp = (
    pd.DataFrame({"feature": all_feature_names, "importance": importances})
    .sort_values("importance", ascending=False)
)  # sorted table of feature importances from the random forest

print(feat_imp.head(20))  # top 20 most important features

# ---------- TOP 20 MOST IMPORTANT FEATURES PLOT ----------

top_k = 20
top_feat = feat_imp.head(top_k).sort_values("importance", ascending=True)
top_feat_desc = feat_imp.head(top_k)
print("\nTop 20 most important features:")
print(top_feat_desc)

plt.figure(figsize=(10, 8))
plt.barh(top_feat["feature"], top_feat["importance"])
plt.xlabel("Feature importance (Gini importance)")
plt.title(f"Top {top_k} Random Forest feature importances")
plt.tight_layout()
plt.show()

# ---------- BOTTOM 20 LEAST IMPORTANT FEATURES PLOT ----------

bottom_k = 20
bottom_feat = (
    feat_imp.tail(bottom_k) # take the smallest ones
    .sort_values("importance", ascending=True)
)

print("\nBottom 20 least important features:")
print(bottom_feat)

plt.figure(figsize=(10, 8))
plt.barh(bottom_feat["feature"], bottom_feat["importance"])
plt.xlabel("Feature importance (Gini importance)")
plt.title(f"Bottom {bottom_k} Random Forest feature importances")
plt.tight_layout()
plt.show()

# %%
#### Baseline models and our models - COMPARISON ####

# Baseline 1: always predict non-fatal (0 = majority class)
y_pred_baseline = np.zeros_like(y_test) # vector of zeros
# Use a constant probability equal to the observed fatal rate in y_test
baseline_pos_rate = y_test.mean()
y_proba_baseline = np.repeat(baseline_pos_rate, len(y_test))

baseline_acc = accuracy_score(y_test, y_pred_baseline)
baseline_recall_fatal = recall_score(y_test, y_pred_baseline, pos_label=1)
baseline_roc_auc = roc_auc_score(y_test, y_proba_baseline)

print("\n=== BASELINE MODEL (always predict non-fatal) ===")
print("Accuracy:", baseline_acc)
print("Recall (fatal class):", baseline_recall_fatal)
print("ROC AUC:", baseline_roc_auc)

from sklearn.dummy import DummyClassifier
# Baseline 2: random predictions respecting class imbalance
dummy = DummyClassifier(strategy="stratified", random_state=69)
dummy.fit(X_train, y_train)

y_pred_dummy = dummy.predict(X_test)
y_proba_dummy = dummy.predict_proba(X_test)[:, 1]

dummy_acc = accuracy_score(y_test, y_pred_dummy)
dummy_recall_fatal = recall_score(y_test, y_pred_dummy, pos_label=1)
dummy_roc_auc = roc_auc_score(y_test, y_proba_dummy)

print("\n=== DUMMY BASELINE (stratified, random) ===")
print("Accuracy:", dummy_acc)
print("Recall (fatal class):", dummy_recall_fatal)
print("ROC AUC:", dummy_roc_auc)

# Metrics for the models
logit_acc = accuracy_score(y_test, y_pred_logit)
logit_recall_fatal = recall_score(y_test, y_pred_logit, pos_label=1)
logit_roc = roc_auc_score(y_test, y_proba_logit)

simple_acc = accuracy_score(y_test, y_pred_simple)
simple_recall_fatal = recall_score(y_test, y_pred_simple, pos_label=1)
simple_roc = roc_auc_score(y_test, y_proba_simple)

rf_acc = accuracy_score(y_test, y_pred_rf)
rf_recall_fatal = recall_score(y_test, y_pred_rf, pos_label=1)
rf_roc = roc_auc_score(y_test, y_proba_rf)

# Comparison table
comparison_df = pd.DataFrame({
    "model": [
        "Baseline (always non-fatal)",
        "Dummy (stratified random)",
        "Lasso logistic regression",
        "Lasso logistic regression (with feature exclusion)",
        "Random forest",
    ],
    "accuracy": [
        baseline_acc,
        dummy_acc,
        logit_acc,
        simple_acc,
        rf_acc,
    ],
    "roc_auc": [
        baseline_roc_auc,
        dummy_roc_auc,
        logit_roc,
        simple_roc,
        rf_roc,
    ],
    "recall_fatal": [
        baseline_recall_fatal,
        dummy_recall_fatal,
        logit_recall_fatal,
        simple_recall_fatal,
        rf_recall_fatal,
    ],
})
print("\n=== MODEL COMPARISON (test set) ===")
print(comparison_df)
# %%
