import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def load_and_preprocess_data(data_folder="data", seed=69):
    """
    Loads raw CSVs, performs aggregation and cleaning exactly as defined in the notebook.
    Returns X_train, X_test, y_train, y_test.
    """
    # Cell 4: Load raw data
    try:
        collision_raw = pd.read_csv(f"{data_folder}/collision.csv")
        casualty_raw = pd.read_csv(f"{data_folder}/casualty.csv")
        vehicle_raw = pd.read_csv(f"{data_folder}/vehicle.csv")
    except FileNotFoundError:
        raise FileNotFoundError(f"Could not find CSV files in '{data_folder}/'. Please check the path.")

    # Cell 5: Target Construction
    y_collision = (collision_raw["collision_severity"] == 1).astype(int)
    collision_raw["target_fatal"] = y_collision

    # Cell 6: Columns to Drop
    cols_to_drop_collision = [
        "collision_year", "collision_ref_no", "location_easting_osgr", "location_northing_osgr",
        "local_authority_district", "local_authority_ons_district", "local_authority_highway",
        "local_authority_highway_current", "junction_detail_historic",
        "pedestrian_crossing_human_control_historic", "pedestrian_crossing_physical_facilities_historic",
        "carriageway_hazards_historic", "lsoa_of_accident_location", "enhanced_severity_collision",
        "collision_injury_based", "collision_adjusted_severity_serious", "collision_adjusted_severity_slight",
        "collision_severity",
    ]

    cols_to_drop_casualty = [
        "collision_year", "collision_ref_no", "casualty_reference", "lsoa_of_casualty",
        "casualty_severity", "enhanced_casualty_severity", "casualty_injury_based",
        "casualty_adjusted_severity_serious", "casualty_adjusted_severity_slight", "age_of_casualty",
    ]

    cols_to_drop_vehicle = [
        "collision_year", "collision_ref_no", "vehicle_manoeuvre_historic",
        "vehicle_location_restricted_lane_historic", "journey_purpose_of_driver_historic",
        "lsoa_of_driver", "generic_make_model", "age_of_driver",
    ]

    # Cell 7: Cleaned tables
    collision_feat = collision_raw.drop(columns=cols_to_drop_collision)
    casualty_feat = casualty_raw.drop(columns=cols_to_drop_casualty)
    vehicle_feat = vehicle_raw.drop(columns=cols_to_drop_vehicle)

    # Cell 8: Aggregation to collision level
    casualty_agg = (
        casualty_feat.groupby("collision_index")
        .agg(
            n_pedestrians=("casualty_class", lambda s: (s == 3).sum()),
            mean_casualty_age_band=("age_band_of_casualty", "mean"),
            mean_casualty_imd=("casualty_imd_decile", "mean"),
            max_casualty_distance_band=("casualty_distance_banding", "max"),
        )
        .reset_index()
    )

    vehicle_agg = (
        vehicle_feat.groupby("collision_index")
        .agg(
            mean_vehicle_age=("age_of_vehicle", "mean"),
            mean_engine_cc=("engine_capacity_cc", "mean"),
            share_left_hand_drive=("vehicle_left_hand_drive", lambda s: (s == 1).mean()),
            share_escooter=("escooter_flag", lambda s: (s == 1).mean()),
        )
        .reset_index()
    )

    model_df = (
        collision_feat.merge(casualty_agg, on="collision_index", how="left")
        .merge(vehicle_agg, on="collision_index", how="left")
    )

    # Cell 9: Time Transform
    model_df["time_parsed"] = pd.to_datetime(model_df["time"], format="%H:%M", errors="coerce")
    model_df["hour_of_day"] = model_df["time_parsed"].dt.hour

    def time_band(h):
        if pd.isna(h): return "unknown"
        h = int(h)
        if 0 <= h < 6: return "night"
        elif 6 <= h < 10: return "morning_peak"
        elif 10 <= h < 16: return "daytime"
        elif 16 <= h < 20: return "evening_peak"
        else: return "late_evening"

    model_df["time_band"] = model_df["hour_of_day"].apply(time_band)

    # Cell 10: Date Transform
    model_df["date_parsed"] = pd.to_datetime(model_df["date"], format="%d/%m/%Y", errors="coerce")
    model_df["day_of_week"] = model_df["date_parsed"].dt.dayofweek
    model_df["month"] = model_df["date_parsed"].dt.month
    model_df["is_weekend"] = model_df["day_of_week"].isin([5, 6]).astype(int)
    model_df = model_df.drop(columns=["time", "time_parsed", "date", "date_parsed"])

    # Cell 11: Type Conversion
    numeric_cols = [
        "number_of_vehicles", "number_of_casualties", "n_pedestrians",
        "mean_vehicle_age", "mean_engine_cc", "longitude", "latitude",
    ]
    model_df[numeric_cols] = model_df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    
    cat_cols = model_df.columns.difference(numeric_cols)
    model_df[cat_cols] = model_df[cat_cols].astype("category")

    # Cell 12: Split X and y
    y = model_df["target_fatal"]
    X = model_df.drop(columns=["target_fatal", "collision_index", "vehicle_reference"], errors="ignore")

    print(f"Data processed. X shape: {X.shape}, y shape: {y.shape}")

    # Cell 13: Train Test Split
    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=seed)