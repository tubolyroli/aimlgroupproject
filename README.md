# AIML Group Project – UK Road Safety Analysis: Predicting Fatal Collisions

## Project Overview
**Research Question:** Can we predict the fatality of traffic collisions in the UK based on environmental conditions, vehicle characteristics, and driver demographics?

Traffic accidents are a leading cause of non-natural death. While most accidents are minor, a small fraction (~1.5%) are fatal. This project builds a machine learning pipeline to identify the key risk factors associated with these fatal outcomes, prioritizing **recall** (catching as many fatal cases as possible) over simple accuracy.

## Data Source
The dataset is derived from the **Department for Transport (UK) Road Safety Data** (STATS19) for the year 2024.
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
    git clone <your-repo-url>
    cd <your-repo-folder>
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

## Project Structure
* `main.py`: **Entry Point**. Orchestrates the entire pipeline.
* `data.py`: Loads raw CSVs, aggregates casualty/vehicle data to the collision level, and handles Train/Test splitting.
* `features.py`: Contains preprocessing pipelines (Imputation, Scaling, One-Hot Encoding).
* `models.py`: Defines the model architectures (Lasso Logistic Regression & Random Forest).
* `evaluate.py`: Calculates performance metrics and saves outputs to the `outputs/` folder.

## Methodology & Results
We compare three approaches:
1.  **Baseline (Majority Rule):** Always predicts "Non-Fatal." (Accuracy: ~98.5%, Recall: 0%).
2.  **Lasso Logistic Regression (L1):** Our primary model. It handles high-dimensional sparse data and performs feature selection.
3.  **Random Forest:** A non-linear benchmark.

**Key Findings:**
While the Baseline has high accuracy, it is useless for safety planning. The **Lasso Logistic Regression** achieves a **Recall of ~78%** on the fatal class (ROC AUC ~0.83), successfully identifying the vast majority of high-risk accidents despite the extreme class imbalance.

## Expected Runtime
* **Total Runtime:** < 2 minutes on a standard laptop.
* **Memory Usage:** < 2GB RAM.

## Authors
* Ádám Burkus
* Attila Sztreborny
* Bendegúz Birkmayer
* Bojta Rácz
* Roland Tuboly