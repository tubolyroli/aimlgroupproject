import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def load_and_process_data(data_folder="data", random_state=69):
    """
    Loads raw CSVs, aggregates casualty/vehicle data to collision level,
    cleans features, and splits into Train/Test sets.
    """
    # 1. Load Raw Data
    print("Loading data...")
    try:
        collision_raw = pd.read_csv(f"{data_folder}/collision.csv")
        casualty_raw = pd.read_csv(f"{data_folder}/casualty.csv")
        vehicle_raw = pd.read_csv(f"{data_folder}/vehicle.csv")
    except FileNotFoundError:
        raise FileNotFoundError(f"Could not find CSV files in folder '{data_folder}'. Check path.")

    # 2. Define Target (Fatal = 1)
    # collision_severity: 1=Fatal, 2=Serious, 3=Slight
    y_all = (collision_raw["collision_severity"] == 1).astype(int)
    collision_raw["target_fatal"] = y_all

    # 3. Columns to Drop (IDs, leakage, constant, or too granular)
    cols_drop_coll = [
        "collision_year", "collision_ref_no", "location_easting_osgr", "location_northing_osgr",
        "local_authority_district", "local_authority_ons_district", "local_authority_highway",
        "local_authority_highway_current", "junction_detail_historic", 
        "pedestrian_crossing_human_control_historic", "pedestrian_crossing_physical_facilities_historic",
        "carriageway_hazards_historic", "lsoa_of_accident_location", "enhanced_severity_collision",
        "collision_injury_based", "collision_adjusted_severity_serious", "collision_adjusted_severity_slight",
        "collision_severity"
    ]
    
    # Aggregation Dictionaries
    # Casualties: Aggregate to collision level
    casualty_agg = casualty_raw.groupby("collision_index").agg(
        n_casualties=("casualty_reference", "count"),
        n_pedestrians=("casualty_class", lambda s: (s == 3).sum()),
        mean_casualty_age_band=("age_band_of_casualty", "mean"),
        mean_casualty_imd=("casualty_imd_decile", "mean"),
        share_male_casualties=("sex_of_casualty", lambda s: (s == 1).mean()), # Added demographic
    ).reset_index()

    # Vehicles: Aggregate to collision level
    vehicle_agg = vehicle_raw.groupby("collision_index").agg(
        n_vehicles=("vehicle_reference", "count"),
        mean_vehicle_age=("age_of_vehicle", "mean"),
        mean_engine_cc=("engine_capacity_cc", "mean"),
        share_left_hand_drive=("vehicle_left_hand_drive", lambda s: (s == 1).mean()),
        share_escooter=("escooter_flag", lambda s: (s == 1).mean()),
        # Added Driver Demographics
        share_male_drivers=("sex_of_driver", lambda s: (s == 1).mean()),
        mean_driver_age_band=("age_band_of_driver", "mean"),
        mean_driver_imd=("driver_imd_decile", "mean"),
    ).reset_index()

    # 4. Merge and Drop
    collision_feat = collision_raw.drop(columns=cols_drop_coll)
    
    model_df = (
        collision_feat
        .merge(casualty_agg, on="collision_index", how="left")
        .merge(vehicle_agg, on="collision_index", how="left")
    )

    # 5. Feature Engineering: Time and Date
    model_df["time_parsed"] = pd.to_datetime(model_df["time"], format="%H:%M", errors="coerce")
    model_df["hour_of_day"] = model_df["time_parsed"].dt.hour
    
    def get_time_band(h):
        if pd.isna(h): return "unknown"
        h = int(h)
        if 0 <= h < 6: return "night"
        elif 6 <= h < 10: return "morning_peak"
        elif 10 <= h < 16: return "daytime"
        elif 16 <= h < 20: return "evening_peak"
        else: return "late_evening"

    model_df["time_band"] = model_df["hour_of_day"].apply(get_time_band)

    model_df["date_parsed"] = pd.to_datetime(model_df["date"], format="%d/%m/%Y", errors="coerce")
    model_df["day_of_week"] = model_df["date_parsed"].dt.dayofweek
    model_df["month"] = model_df["date_parsed"].dt.month
    model_df["is_weekend"] = model_df["day_of_week"].isin([5, 6]).astype(int)

    # Drop intermediate date/time columns
    model_df = model_df.drop(columns=["time", "time_parsed", "date", "date_parsed"])

    # 6. Type Casting
    numeric_cols = [
        "number_of_vehicles", "number_of_casualties", "n_pedestrians", 
        "mean_vehicle_age", "mean_engine_cc", "longitude", "latitude",
        "n_casualties", "n_vehicles", "mean_casualty_age_band", "mean_casualty_imd",
        "share_male_casualties", "share_male_drivers", "mean_driver_age_band", "mean_driver_imd"
    ]
    # Ensure they exist (some might be missing if csv is old version, handle safely)
    actual_num_cols = [c for c in numeric_cols if c in model_df.columns]
    
    model_df[actual_num_cols] = model_df[actual_num_cols].apply(pd.to_numeric, errors="coerce")
    
    # Remaining are categorical
    cat_cols = model_df.columns.difference(actual_num_cols + ["target_fatal", "collision_index", "vehicle_reference"])
    model_df[cat_cols] = model_df[cat_cols].astype("category")

    # 7. Split X and y
    y = model_df["target_fatal"]
    X = model_df.drop(columns=["target_fatal", "collision_index", "vehicle_reference"], errors="ignore")

    print(f"Data processed. Shape: {X.shape}, Fatal Rate: {y.mean():.4f}")

    # 8. Train/Test Split
    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=random_state)