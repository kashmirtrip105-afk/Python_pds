# Water Control & Quality Monitoring System

A Python + Streamlit water-quality monitoring and analytics project (CIPAT
course project, GTU BE Computer Engineering - Python for Data Science).

See **AUDIT_REPORT.md** for a full account of the code audit performed on
this project: every bug found, how it was fixed, and how each was verified.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Run the CLI application

```bash
python main.py            # interactive menu
python main.py --full     # run the entire pipeline (collect -> ... -> ML) in one shot
python main.py --live     # stream simulated real-time readings
```

Run `python main.py --full` at least once before launching the Streamlit
app's Machine Learning tab, since it trains and saves the model files the
dashboard loads (`data/processed/water_quality_classifier.pkl` and its
scaler).

## Run the Streamlit dashboard

```bash
python -m streamlit run streamlit_app.py
```

## Project layout

```
main.py                  CLI entry point (menu-driven, maps to CIPAT topics A-R)
streamlit_app.py         Interactive dashboard (5 tabs)
config.py                Central configuration (paths, thresholds, station info)
modules/
  custom_utils.py        Custom exceptions, logging, decorators, data structures
  data_collector.py      Live API fetch, web scraping, synthetic data generation
  data_preprocessing.py  Cleaning, outlier handling, feature engineering, WQI
  database_manager.py    SQLite CRUD (sensor readings + interactive CRUD records)
  statistical_analysis.py Descriptive stats, distribution fitting, probability
  inference.py            Sampling, confidence intervals, hypothesis testing
  visualization.py        Matplotlib/Seaborn exploratory figures
  ml_model.py              Regression (pH forecasting) + classification (contamination)
data/                     Raw/processed CSVs, SQLite database, trained model artifacts
reports/figures/          Generated PNG charts
logs/                     Application log file
```

