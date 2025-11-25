# AIML Group Project – Road Collision Severity Modelling

This repository contains our Applied Machine Learning group project.  
We build models to predict whether a road collision is **fatal vs. non-fatal** using UK collision, casualty, and vehicle data.

## Project structure

```text
.
├── src/
│   └── main.py              # main modelling script (LASSO + Random Forest + baselines)
├── data/
│   ├── collision.csv
│   ├── casualty.csv
│   └── vehicle.csv
│   └── variables_description.csv
├── requirements.txt
├── .gitignore
└── README.md