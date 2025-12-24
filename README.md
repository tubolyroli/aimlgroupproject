[![smoke-test](https://github.com/tubolyroli/aimlgroupproject/actions/workflows/ci.yml/badge.svg)](https://github.com/tubolyroli/aimlgroupproject/actions/workflows/ci.yml)
# AIML Group Project – UK Road Safety Analysis: Predicting Fatal Collisions

## Project Overview
**Research Question:** Can we predict the fatality of traffic collisions in the UK based on environmental conditions, vehicle characteristics, and driver demographics?

**Why it matters:** fatal collisions are rare, so this project focuses on catching as many fatal cases as possible by prioritizing **recall** over accuracy.

## Data Source
The dataset is derived from the [**Department for Transport (UK) Road Safety Data**](https://www.gov.uk/government/statistical-data-sets/road-safety-open-data) (STATS19) for the year 2024.
It consists of three linked CSV files:
1.  `collision.csv`: event details (location, time, weather, road conditions)
2.  `vehicle.csv`: vehicle details (type, age, engine size) and driver demographics (age, sex, socioeconomic decile)
3.  `casualty.csv`: details on people injured (severity, age, pedestrian status)

## Data Download
### Option A: Download manually
1. Download the three CSV files from the official source.
2. Create a folder named `data/` in the repo root.
3. Place the files inside `data/` and rename them exactly to:
- `data/collision.csv`
- `data/vehicle.csv`
- `data/casualty.csv`

### Option B: Download via script
Run:
```bash
python scripts/data_download.py
```

## Data License
The dataset is published under the [**Open Government Licence v3.0**](https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/?utm_source=chatgpt.com). This repository does not redistribute the raw STATS19 data files.

## Environment Setup
Requires **Python 3.10+**.

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/tubolyroli/aimlgroupproject
    cd aimlgroupproject
    ```

2.  **Create environment and install dependencies:**
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # Windows: .venv\Scripts\activate
    pip install -r requirements.txt
    ```
Dependencies are pinned in requirements.txt for reproducibility.

## Usage
To run the full end-to-end analysis (cleaning, processing, training, and evaluation), run the following command:

```bash
python scripts/main.py --data_path data --seed 69
```

### Arguments
* `--data_path`: directory containing the input CSV files (`collision.csv`, `vehicle.csv`, `casualty.csv`).
    * *Default:* `data` (expects a folder named 'data' in the same directory as the script).
* `--seed`: random seed for splitting and model initialization.
    * *Default:* `69`

### Outputs

After execution, the script generates an `outputs/` folder containing:

* `metrics_lasso_logistic_regression.json`

* `predictions_lasso_logistic_regression.csv`

* Console logs detailing feature importance and cross-validation scores

## Project Structure
* `scripts/main.py`: orchestrates the entire pipeline
* `scripts/data.py`: loads CSVs, aggregates to collision level, train/test splitting
* `scripts/features.py`: contains preprocessing pipelines (imputation, scaling, one hot encoding)
* `scripts/models.py`: defines the model architectures (Lasso Logistic Regression & Random Forest)
* `scripts/evaluate.py`: calculates performance metrics and saves outputs to the `outputs/` folder

## Methodology

### 1. Data Preprocessing & Engineering
* Join and aggregate relational tables to collision level
* Engineer time features and summary statistics
* Handle missingness inside sklearn Pipelines
* One hot encode categorical variables

### 2. Modeling Strategy
* Extreme class imbalance handled via class weights
* Model selection optimized for fatal class recall
* Models compared with stratified cross validation:
    * baseline majority rule
    * L1 logistic regression
    * random forest

## Results

Test set results (single 80/20 split, seed 69):

| Model | Accuracy | Recall (Fatal) | ROC AUC |
| :--- | :--- | :--- | :--- |
| **Baseline (Majority Rule)** | **98.5%** | 0.0% | 0.50 |
| **Random Forest** | 76.2% | 64.0% | 0.80 |
| **Lasso Logistic Regression** | 71.0% | **77.7%** | **0.83** |

### Key takeaways:
* Accuracy is misleading in rare event prediction, recall is the priority metric
* Lasso performed best in this sparse high dimensional setting
* Strong predictors include lighting, speed limit, and vulnerable road users

## Limitations
* Observational data, no causal claims
* Performance may shift across years and regions
* Threshold choice changes false positive rate materially

## Runtime and resource usage
Full pipeline run on STATS19 2024 data.

Environment:
- Model: MacBook Air
- Chip: Apple M4
- Memory: 16 GB
- Python version: 3.13.5

Observed:
- **Wall clock runtime (real):** ~403 s (≈ 6 min 43 s)
- **CPU time:** 2077 s user + 40 s sys (multi-core execution, ~5× parallelism on average)
- **Peak RAM (maximum resident set size):** 696,483,840 bytes (≈ 664 MB)

Command used:
```bash
/usr/bin/time -l python scripts/main.py --data_path data --seed 69
```

## License
Code is licensed under the MIT License. See `LICENSE`.
Data is licensed separately under the Open Government Licence and is not redistributed in this repository.

## Authors
* Ádám Burkus
* Attila Sztreborny
* Bendegúz Birkmayer
* Bojta Rácz
* Roland Tuboly
