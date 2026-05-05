# AutoML Dashboard

A browser based Automated Machine Learning platform for non expert users, built with Python and Streamlit.

## Features

- Upload any CSV dataset or use built in demos
- Automatic classification and regression task detection
- Trains and compares up to 10 ML models simultaneously
- Interactive visualizations: distributions, correlation heatmap, radar chart, feature importances
- Hyperparameter tuning via Optuna
- Predict on new data using the best trained model
- Export results as CSV or download the trained model

## Project Structure

```
automl-dashboard/
├── app.py                  # Main Streamlit application
├── requirements.txt
├── .streamlit/
│   └── config.toml         # Theme configuration
└── src/
    ├── preprocessing.py    # Data cleaning and feature engineering
    ├── model_trainer.py    # Model registry, training, evaluation
    ├── visualizations.py   # Plotly chart generation
    └── utils.py            # Helper functions and demo datasets
```

## Running Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deployment

Deployed on Streamlit Community Cloud at: [https://share.streamlit.io/sv3005/automl-dashboard](https://automldashboard.streamlit.app)

## Tech Stack

- Streamlit 1.32+
- scikit-learn 1.3+
- XGBoost 2.0+
- LightGBM 4.0+
- Plotly 5.18+
- Optuna 3.0+
- pandas 2.0+

## Author

Sushmitha Vashist | NYU | sv3005@nyu.edu  
Advanced Project | Spring 2026  
Advisor: Prof. Azeez Bhavnagarwala
