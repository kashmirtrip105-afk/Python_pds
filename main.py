"""
main.py
========
Water Quality Monitoring Analytics - CIPAT Project (Real-Time Application)
Course: Python for Data Science (BE05000231)

Entry point that ties every module together into one menu-driven CLI
application. Run:  python main.py

Menu options map directly onto the CIPAT syllabus topics (A-R):
  1. Collect data              (F. Web Scraping, C. Libraries, E. Exceptions)
  2. Preprocess data           (I. NumPy, K. Pandas, L. Data Preprocessing)
  3. Store / manage records    (H. CRUD, G. File Handling)
  4. Statistical analysis      (J. SciPy, N. Statistics, O. Probability)
  5. Sampling & hypothesis tests(P. Sampling & Inference, Q. Hypothesis Testing)
  6. Generate visualizations   (M. Data Visualization)
  7. Train ML models           (R. Machine Learning)
  8. Run FULL pipeline (1-7)
  9. LIVE real-time monitor (streams simulated sensor readings)
  0. Exit
"""

import os
import sys
import time

import pandas as pd

import config
from modules import (custom_utils, data_collector, data_preprocessing,
                      database_manager, inference, ml_model,
                      statistical_analysis, visualization)
from modules.custom_utils import (DataFetchError, DataValidationError,
                                   DatabaseError, ReadingStack, get_logger)

logger = get_logger("main")

undo_stack = ReadingStack()

MENU = {
    "1": "Collect real-time water quality data (Indian river sources + fallback)",
    "2": "Preprocess & clean data",
    "3": "CRUD demo on SQLite database",
    "4": "Statistical & probability analysis",
    "5": "Sampling & hypothesis testing",
    "6": "Generate visualizations",
    "7": "Train machine learning models",
    "8": "Run FULL pipeline (steps 1-7)",
    "9": "LIVE real-time monitor (streams new readings)",
    "0": "Exit",
}


def print_menu():
    print("\n" + "=" * 65)
    print(f"  WATER QUALITY MONITORING  -  {config.STATION_NAME}")
    print("=" * 65)
    for key, label in MENU.items():
        print(f"  [{key}] {label}")
    print("=" * 65)


def step_collect() -> pd.DataFrame:
    print("\n>> Collecting real-time data (live Indian API -> scraping -> synthetic fallback)...")
    try:
        df = data_collector.collect_dataset()
    except DataFetchError as exc:
        logger.error("Data collection failed entirely: %s", exc)
        print(f"   Data collection failed: {exc}")
        sys.exit(1)
    print(f"   Collected {len(df)} rows -> {config.RAW_CSV_PATH}")
    return df


def step_preprocess(raw_df: pd.DataFrame = None) -> pd.DataFrame:
    print("\n>> Preprocessing data...")
    if raw_df is None:
        if not os.path.exists(config.RAW_CSV_PATH):
            print("   No raw data found - run step 1 first.")
            return None
        raw_df = pd.read_csv(config.RAW_CSV_PATH)
    try:
        clean_df = data_preprocessing.preprocess_pipeline(raw_df)
    except DataValidationError as exc:
        print(f"   Preprocessing failed: {exc}")
        return None
    print(f"   Cleaned dataset: {clean_df.shape[0]} rows x {clean_df.shape[1]} cols "
          f"-> {config.PROCESSED_CSV_PATH}")
    return clean_df


def step_crud_demo(clean_df: pd.DataFrame = None):
    print("\n>> Running CRUD demo on SQLite database...")
    try:
        if clean_df is None:
            clean_df = pd.read_csv(config.PROCESSED_CSV_PATH)
        cols = ["timestamp", "station", "temperature_c", "ph", "dissolved_oxygen_mgl",
                "conductivity_us_cm", "turbidity_ntu", "contamination_event"]
        db_ready_df = clean_df[cols].copy()
        db_ready_df["timestamp"] = pd.to_datetime(db_ready_df["timestamp"]).astype(str)
        inserted = database_manager.bulk_insert(db_ready_df)
        print(f"   CREATE: inserted {inserted} new rows")

        latest_ts = db_ready_df["timestamp"].max()
        record = database_manager.read_by_timestamp(latest_ts)
        print(f"   READ:   {latest_ts} -> {record}")

        updated = database_manager.update_record(latest_ts, {"turbidity_ntu": 2.5})
        undo_stack.push(("update", latest_ts, record))
        print(f"   UPDATE: {updated} row(s) updated (turbidity set to 2.5)")

        total = len(database_manager.read_all())
        print(f"   Total rows currently in DB: {total}")

        deleted = database_manager.delete_record("1999-01-01 00:00:00")
        print(f"   DELETE: {deleted} row(s) deleted for a demo non-existent timestamp")

    except DatabaseError as exc:
        print(f"   Database operation failed: {exc}")


def step_statistics(clean_df: pd.DataFrame = None):
    print("\n>> Running statistical & probability analysis...")
    if clean_df is None:
        clean_df = pd.read_csv(config.PROCESSED_CSV_PATH)
    numeric_cols = ["ph", "temperature_c", "dissolved_oxygen_mgl",
                     "conductivity_us_cm", "turbidity_ntu", "water_quality_index"]

    summary = statistical_analysis.descriptive_summary(clean_df, numeric_cols)
    print("\n   Descriptive statistics (pH row):")
    print("  ", summary.loc["ph"].to_dict())

    corr = statistical_analysis.correlation_matrix(clean_df, numeric_cols)
    print(f"\n   Correlation (turbidity vs dissolved oxygen): "
          f"{corr.loc['turbidity_ntu', 'dissolved_oxygen_mgl']}")

    fit = statistical_analysis.fit_normal_distribution(clean_df["ph"])
    print(f"   Normal-distribution fit for pH: {fit}")

    p_contam = statistical_analysis.probability_of_contamination(clean_df)
    p_contam_night = statistical_analysis.probability_of_contamination(clean_df, time_band="Night")
    print(f"   P(contamination) = {p_contam}   |   P(contamination | Night) = {p_contam_night}")

    poisson_model = statistical_analysis.poisson_contamination_event_model(clean_df)
    print(f"   Poisson contamination-event model: lambda="
          f"{poisson_model['lambda_avg_events_per_day']}")
    return summary, corr


def step_inference(clean_df: pd.DataFrame = None):
    print("\n>> Sampling & hypothesis testing...")
    if clean_df is None:
        clean_df = pd.read_csv(config.PROCESSED_CSV_PATH)

    sample = inference.stratified_sample_by_time_band(clean_df, frac=0.25)
    print(f"   Stratified sample drawn: {len(sample)} rows (from {len(clean_df)} total)")

    ci = inference.mean_confidence_interval(clean_df["ph"])
    print(f"   95% CI for mean pH: [{ci['ci_lower']}, {ci['ci_upper']}] "
          f"(sample mean={ci['mean']})")

    morning = clean_df.loc[clean_df["time_band"] == "Morning", "ph"]
    night = clean_df.loc[clean_df["time_band"] == "Night", "ph"]
    t_test = inference.two_sample_ttest(morning, night)
    print(f"   Morning vs Night pH t-test: {t_test['conclusion']} (p={t_test['p_value']})")

    chi2 = inference.chi_square_independence(clean_df, "time_band", "contamination_event")
    print(f"   Time-band vs contamination chi-square test: {chi2['conclusion']} "
          f"(p={chi2['p_value']})")

    band_groups = [clean_df.loc[clean_df["time_band"] == b, "ph"]
                    for b in clean_df["time_band"].unique()]
    anova = inference.one_way_anova(*band_groups)
    print(f"   ANOVA across time bands: {anova['conclusion']} (p={anova['p_value']})")


def step_visualize(clean_df: pd.DataFrame = None):
    print("\n>> Generating visualizations...")
    if clean_df is None:
        clean_df = pd.read_csv(config.PROCESSED_CSV_PATH)
    paths = visualization.generate_all_figures(clean_df)
    for p in paths:
        print(f"   Saved: {p}")


def step_ml(clean_df: pd.DataFrame = None):
    print("\n>> Training machine learning models...")
    if clean_df is None:
        clean_df = pd.read_csv(config.PROCESSED_CSV_PATH)

    reg_results = ml_model.train_ph_regressor(clean_df)
    for name, res in reg_results.items():
        print(f"   [Regression] {name}: MAE={res['mae']}  RMSE={res['rmse']}  R2={res['r2']}")
    best_reg_name = min(reg_results, key=lambda n: reg_results[n]["mae"])
    visualization.plot_ml_actual_vs_predicted(
        reg_results[best_reg_name]["y_test"], reg_results[best_reg_name]["y_pred"],
        best_reg_name, target_label="pH",
    )

    clf_results = ml_model.train_contamination_classifier(clean_df)
    for name, res in clf_results.items():
        print(f"   [Classification] {name}: Accuracy={res['accuracy']}")
    best_clf_name = max(clf_results, key=lambda n: clf_results[n]["accuracy"])

    os.makedirs(config.DATA_PROCESSED_DIR, exist_ok=True)
    ml_model.save_model(
        reg_results[best_reg_name]["model"],
        os.path.join(config.DATA_PROCESSED_DIR, "best_ph_model.pkl"),
    )
    # Save the best contamination CLASSIFIER + its scaler too. The Streamlit
    # "Machine Learning & Quality Predictor" tab does potability/contamination
    # classification, so it needs this model (not the pH regressor above) and
    # the matching StandardScaler to correctly transform live user inputs.
    ml_model.save_model(
        clf_results[best_clf_name]["model"],
        os.path.join(config.DATA_PROCESSED_DIR, "water_quality_classifier.pkl"),
    )
    ml_model.save_model(
        clf_results[best_clf_name]["scaler"],
        os.path.join(config.DATA_PROCESSED_DIR, "water_quality_classifier_scaler.pkl"),
    )
    print(f"   Best classifier ({best_clf_name}) + scaler saved for the Streamlit predictor.")
    return reg_results, clf_results


def step_live_monitor(iterations: int = config.LIVE_DEFAULT_ITERATIONS):
    """
    REAL-TIME APPLICATION MODE: streams new simulated sensor readings every
    LIVE_POLL_SECONDS, checks each one against safety thresholds, stores it
    in the database immediately (CRUD in real time), and raises an alert
    the instant a contamination event is detected - this is the part that
    makes the project a genuine real-time monitoring application rather
    than a one-shot batch script.
    """
    print(f"\n>> Starting LIVE monitor for {iterations} readings "
          f"(every {config.LIVE_POLL_SECONDS}s)... Press Ctrl+C to stop early.\n")
    thresholds = data_collector.get_alert_thresholds()
    recent = custom_utils.RecentReadingsBuffer(maxlen=10)
    database_manager.init_db()

    try:
        for i in range(1, iterations + 1):
            reading = data_collector.generate_single_live_reading()
            recent.push(reading["ph"])

            db_record = {**reading, "timestamp": reading["timestamp"].strftime("%Y-%m-%d %H:%M:%S.%f")}
            try:
                database_manager.create_record(db_record)
            except DatabaseError as exc:
                logger.warning("Skipped duplicate live reading: %s", exc)

            alerts = []
            for param, (low, high) in thresholds.items():
                if param in reading and not (low <= reading[param] <= high):
                    alerts.append(f"{param}={reading[param]} outside [{low}, {high}]")

            status = "🔴 ALERT" if alerts else "🟢 OK"
            print(f"[{i:02d}/{iterations}] {reading['timestamp'].strftime('%H:%M:%S')} | "
                  f"pH={reading['ph']:<5} DO={reading['dissolved_oxygen_mgl']:<5} "
                  f"Turb={reading['turbidity_ntu']:<6} Temp={reading['temperature_c']:<5} | "
                  f"{status}  (rolling avg pH={recent.rolling_average():.2f})")
            if alerts:
                for a in alerts:
                    print(f"          ⚠️  {a}")

            time.sleep(config.LIVE_POLL_SECONDS)
    except KeyboardInterrupt:
        print("\n   Live monitor stopped by user.")

    print(f"\n   Live monitor finished. All readings were saved to {config.DB_PATH}")


def run_full_pipeline():
    raw_df = step_collect()
    clean_df = step_preprocess(raw_df)
    step_crud_demo(clean_df)
    step_statistics(clean_df)
    step_inference(clean_df)
    step_visualize(clean_df)
    step_ml(clean_df)
    print("\n✅ Full pipeline completed successfully.")
    print(f"   Figures  : {config.FIGURES_DIR}")
    print(f"   Database : {config.DB_PATH}")
    print(f"   Log file : {config.LOG_FILE_PATH}")


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--full":
        run_full_pipeline()
        return
    if len(sys.argv) > 1 and sys.argv[1] == "--live":
        step_live_monitor()
        return

    while True:
        print_menu()
        choice = input("Select an option: ").strip()
        try:
            if choice == "1":
                step_collect()
            elif choice == "2":
                step_preprocess()
            elif choice == "3":
                step_crud_demo()
            elif choice == "4":
                step_statistics()
            elif choice == "5":
                step_inference()
            elif choice == "6":
                step_visualize()
            elif choice == "7":
                step_ml()
            elif choice == "8":
                run_full_pipeline()
            elif choice == "9":
                step_live_monitor()
            elif choice == "0":
                print("Goodbye!")
                break
            else:
                print("Invalid option, please choose 0-9.")
        except FileNotFoundError as exc:
            print(f"   Required file missing - run earlier steps first. ({exc})")
        except Exception as exc:  # noqa: BLE001 - top-level safety net
            logger.exception("Unexpected error in menu option %s", choice)
            print(f"   Unexpected error: {exc}")


if __name__ == "__main__":
    main()
