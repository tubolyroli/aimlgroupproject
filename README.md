# AIML Group Project – UK Road Safety Analysis: Predicting Fatal Collisions

## Project Overview
**Research Question:** Can we predict the fatality of traffic collisions in the UK based on environmental conditions, vehicle characteristics, and driver demographics?

Traffic accidents are a leading cause of non-natural death. While most accidents are minor, a small fraction (~1.5%) are fatal. This project builds a machine learning pipeline to identify the key risk factors associated with these fatal outcomes, prioritizing **recall** (catching as many fatal cases as possible) over simple accuracy.

## Data Source
The dataset is derived from the [**Department for Transport (UK) Road Safety Data**](https://www.gov.uk/government/statistics/reported-road-casualties-great-britain-annual-report-2024) (STATS19) for the year 2024.
It consists of three linked CSV files:
1.  `collision.csv`: Event details (location, time, weather, road conditions).
2.  `vehicle.csv`: Vehicle details (type, age, engine size) and driver demographics (age, sex, socioeconomic decile).
3.  `casualty.csv`: Details on people injured (severity, age, pedestrian status).

**Data Access & Setup:**
1.  Create a folder named `data/` in the root of this repository.
2.  Place the three raw CSV files (`collision.csv`, `vehicle.csv`, `casualty.csv`) inside that folder.

## Environment Setup
This project requires **Python 3.8+**.

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/tubolyroli/aimlgroupproject
    cd aimlgroupproject
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage
To run the full end-to-end analysis (cleaning, processing, training, and evaluation), run the following command:

```bash
python main.py --data_path data --seed 69
```

### Arguments
* `--data_path`: The directory path where the script looks for the input CSV files (`collision.csv`, `vehicle.csv`, `casualty.csv`).
    * *Default:* `data` (expects a folder named 'data' in the same directory as the script).
    * *Example:* `--data_path "C:/Downloads/my_project_data"`
* `--seed`: The random seed used for data splitting and model initialization. Setting this ensures the results are exactly the same every time you run the script.
    * *Default:* `69`

### Outputs

After execution, the script generates an `outputs/` folder containing:

* `metrics_lasso_logistic_regression.json`: Performance scores.

* `predictions_lasso_logistic_regression.csv`: Raw predictions for further analysis.

* Console logs detailing feature importance and cross-validation scores.

## Project Structure
* `main.py`: **Entry Point**. Orchestrates the entire pipeline.
* `data.py`: Loads raw CSVs, aggregates casualty/vehicle data to the collision level, and handles Train/Test splitting.
* `features.py`: Contains preprocessing pipelines (Imputation, Scaling, One-Hot Encoding).
* `models.py`: Defines the model architectures (Lasso Logistic Regression & Random Forest).
* `evaluate.py`: Calculates performance metrics and saves outputs to the `outputs/` folder.

## Methodology

### 1. Data Preprocessing & Engineering
The raw data consists of three relational tables: *Collisions*, *Vehicles*, and *Casualties*. Since our objective is to predict the severity of a **collision event**, we performed the following transformations:
* **Aggregation:** We aggregated *Vehicle* and *Casualty* data to the *Collision* level. This involved creating summary features such as `share_male_drivers`, `mean_vehicle_age`, and `number_of_pedestrians`.
* **Feature Engineering:** We derived temporal features (e.g., `time_band` for "Rush Hour" vs. "Night") and parsed demographic data.
* **Handling Missingness:** We used a Pipeline approach to prevent data leakage. Numeric features were imputed with the mean, while categorical features (e.g., driver demographics in hit-and-run cases) were imputed with a constant "missing" marker.
* **Encoding:** Categorical variables were transformed using One-Hot Encoding, resulting in a high-dimensional feature space (~80 features).

### 2. Modeling Strategy
Given the extreme class imbalance (~1.5% fatal accidents), standard accuracy optimization would yield a trivial model that predicts "Non-Fatal" for every case. We addressed this via:
* **Class Weighting:** We applied `class_weight='balanced'` to all models, effectively penalizing the misclassification of fatal accidents significantly more than non-fatal ones.
* **Metric Selection:** We optimized for **Recall (Sensitivity)** on the Fatal class to maximize the detection of life-threatening accidents.

We trained and compared three models using **5-Fold Stratified Cross-Validation**:
1.  **Baseline (Majority Rule):** A naive model predicting the most frequent class (Non-Fatal).
2.  **Lasso Logistic Regression (L1):** A linear model with L1 regularization to perform automatic feature selection on the sparse One-Hot encoded data.
3.  **Random Forest:** An ensemble of decision trees to capture non-linear interactions.

---

## Results

### Model Performance Comparison
We evaluated the models on a held-out Test Set (20% split). The results highlight the "Accuracy Paradox" inherent in imbalanced safety data.

| Model | Accuracy | Recall (Fatal) | ROC AUC |
| :--- | :--- | :--- | :--- |
| **Baseline (Majority Rule)** | **98.5%** | 0.0% | 0.50 |
| **Random Forest** | 76.2% | 64.0% | 0.80 |
| **Lasso Logistic Regression** | 71.0% | **77.7%** | **0.83** |

### Key Findings
1.  **The Accuracy Trade-off:** The Baseline model achieved near-perfect accuracy (98.5%) but failed to identify a single fatal accident (Recall = 0%). To create a useful safety tool, we accepted a lower accuracy (71.0%) in the Lasso model to achieve a **Recall of 77.7%**. This means our model successfully flags nearly **4 out of 5** fatal crashes.
2.  **Linear vs. Non-Linear:** The Lasso Logistic Regression outperformed the Random Forest in both Recall (+13.7%) and ROC AUC (+0.03). This suggests that for this specific high-dimensional, sparse dataset, the global optimization of the linear model was more robust to noise than the local splits of the Random Forest.
3.  **Risk Drivers:** Feature importance analysis revealed that **environmental context** (unlit roads at night, high speed limits) and **vulnerable road users** (pedestrians, motorcycles) are the strongest predictors of fatality, rather than vehicle mechanics alone.

## Expected Runtime
* **Total Runtime:** ~ 5-10 minutes on a standard laptop.

## Authors
* Ádám Burkus
* Attila Sztreborny
* Bendegúz Birkmayer
* Bojta Rácz
* Roland Tuboly