import pandas as pd
import numpy as np
import re
from pathlib import Path


def normalize_col(col: str) -> str:
    """
    Normalize column names:
    - strip spaces
    - lowercase
    - remove special chars
    - keep only alphanumerics
    """
    col = str(col).strip().lower()
    col = re.sub(r"[^a-z0-9]+", "", col)
    return col

def find_column(df: pd.DataFrame, aliases):
    norm_map = {normalize_col(c): c for c in df.columns}
    for alias in aliases:
        key = normalize_col(alias)
        if key in norm_map:
            return norm_map[key]
    return None

def safe_to_numeric(series):
    return pd.to_numeric(series, errors="coerce")

def clip_outliers(series, lower_q=0.01, upper_q=0.99):

    if series.dropna().empty:
        return series
    low = series.quantile(lower_q)
    high = series.quantile(upper_q)
    return series.clip(low, high)


def load_data(
    sensor_path=r"C:\Users\LENOVO\Desktop\All-Data-SensorParser.xlsx",
    daily_path=r"C:\Users\LENOVO\Desktop\DailyAverageSensedData1.xlsx",
    use_daily_average=True,
    add_rl_bins=False
):
    """
    Returns a cleaned dataframe for RL irrigation.

    Expected RL-ready columns:
    - temp
    - humidity
    - pressure
    - wind
    - par
    - soil
    - eto
    - etr (optional)
    - water (optional target/label if present)
    - date (if available)
    """

    sensor_path = Path(sensor_path)
    daily_path = Path(daily_path)

    if not sensor_path.exists():
        raise FileNotFoundError(f"Sensor file not found: {sensor_path}")
    if not daily_path.exists():
        raise FileNotFoundError(f"Daily average file not found: {daily_path}")

    df_sensor = pd.read_excel(sensor_path)
    df_daily = pd.read_excel(daily_path)

    # Clean column names
    df_sensor.columns = [str(c).strip() for c in df_sensor.columns]
    df_daily.columns = [str(c).strip() for c in df_daily.columns]

    # -----------------------------
    # Try to get usable columns from daily average file first
    # -----------------------------
    source_df = df_daily.copy() if use_daily_average else df_sensor.copy()

    # Common aliases based on your shown columns
    aliases = {
        "date": ["Date", "date", "DAY", "Timestamp"],
        "temp": ["SA01-TC", "SAP01-TC", "TC", "TEMP", "Temperature"],
        "humidity": ["SA01-HUM", "SAP01-HUM", "HUM", "RH", "Humidity"],
        "pressure": ["SA01-PRES", "SAP01-PRES", "PRES", "Pressure"],
        "wind": ["ANE", "Wind", "WindSpeed", "WS"],
        "par": ["PAR", "SolarRadiation", "Radiation"],
        "soil": ["SA01-SOIL", "SAP01-SOIL", "SOIL", "SoilMoisture", "SOIL_C", "SOIL_B"],
        "eto": ["SA01-PM ETo", "SAP01-PM ETo", "ETo", "PMETo", "ETo_PM"],
        "etr": ["SA01-PM ETr", "SAP01-PM ETr", "ETr", "PMETr", "ETr_PM"],
        "water_sa01": ["WaterSA01", "SA01-Water", "IrrigationSA01"],
        "water_sap01": ["WaterSAP01", "SAP01-Water", "IrrigationSAP01"],
    }

    selected = {}

    for standard_name, candidates in aliases.items():
        col = find_column(source_df, candidates)
        if col is not None:
            selected[standard_name] = source_df[col]
        else:
            selected[standard_name] = None

    # If daily average file is missing a useful column, try sensor file
    # (this is helpful if one file has more complete data than the other)
    for standard_name, candidates in aliases.items():
        if selected[standard_name] is None:
            col = find_column(df_sensor, candidates)
            if col is not None:
                selected[standard_name] = df_sensor[col]

    # Build final dataframe
    out = pd.DataFrame()

    for k, v in selected.items():
        if v is not None:
            out[k] = v

    # Keep date if present
    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"], errors="coerce", dayfirst=True)

    # Convert numeric columns
    for col in out.columns:
        if col != "date":
            out[col] = safe_to_numeric(out[col])

    # Drop fully empty columns
    out = out.dropna(axis=1, how="all")

    # Remove rows where all usable features are missing
    feature_cols = [c for c in out.columns if c != "date"]
    out = out.dropna(subset=feature_cols, how="all").copy()

    # Clip extreme sensor noise
    for col in feature_cols:
        out[col] = clip_outliers(out[col])

    # Sort by date if available
    if "date" in out.columns:
        out = out.sort_values("date").reset_index(drop=True)

    # Handle missing values:
    # 1) interpolate numeric values
    # 2) forward fill
    # 3) backward fill
    num_cols = [c for c in out.columns if c != "date"]
    out[num_cols] = out[num_cols].interpolate(method="linear", limit_direction="both")
    out[num_cols] = out[num_cols].ffill().bfill()

    # -----------------------------
    # Derive RL-friendly features
    # -----------------------------
    # If ETo exists and ETr exists, create a demand signal
    if "eto" in out.columns and "etr" in out.columns:
        out["water_demand"] = (out["eto"] + out["etr"]) / 2.0
    elif "eto" in out.columns:
        out["water_demand"] = out["eto"]
    elif "etr" in out.columns:
        out["water_demand"] = out["etr"]
    else:
        out["water_demand"] = np.nan

    # Create a simple normalized soil dryness index if soil exists
    if "soil" in out.columns:
        soil_min = out["soil"].min()
        soil_max = out["soil"].max()
        if pd.notna(soil_min) and pd.notna(soil_max) and soil_max != soil_min:
            out["soil_norm"] = (out["soil"] - soil_min) / (soil_max - soil_min)
        else:
            out["soil_norm"] = out["soil"]
    else:
        out["soil_norm"] = np.nan

    # Optional: RL bins for tabular Q-learning
    if add_rl_bins:
        def bin_series(s, bins=5):
            if s.dropna().empty:
                return s
            return pd.qcut(s.rank(method="first"), q=bins, labels=False, duplicates="drop")

        # Only bin columns useful for state representation
        for col in ["temp", "humidity", "pressure", "wind", "par", "soil", "water_demand"]:
            if col in out.columns:
                out[f"{col}_bin"] = bin_series(out[col], bins=5)

    # Final cleanup: drop rows where crucial columns are still missing
    critical = [c for c in ["temp", "humidity", "par", "soil", "water_demand"] if c in out.columns]
    if critical:
        out = out.dropna(subset=critical).reset_index(drop=True)

    return out


# -----------------------------
# Quick test
# -----------------------------
if __name__ == "__main__":
    df = load_data(add_rl_bins=True)
    print("Columns:", df.columns.tolist())
    print(df.head())