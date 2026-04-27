import pandas as pd
from datetime import datetime
from zoneinfo import ZoneInfo
from Data import PreferredTimezone

def merge_features_with_target(
    preferred_timezone: PreferredTimezone = PreferredTimezone.UTC
) -> pd.DataFrame:
    """
    Merge day-ahead electricity prices (target) with public power time series (features)
    into a single, hourly-aligned dataset.

    The function:
      1) Loads the target CSV (day-ahead prices) and parses the `timestamp` column and resamples it to hourly freq - prices by October 2025 were in 15-min freq.
      2) Loads the features CSV (public power production/load) and parses the `timestamp` column.
      3) Resamples the feature time series to hourly frequency using the mean aggregation.
      4) Renames feature columns to a consistent lowercase format.
      5) Merges target and hourly features on `timestamp` using an inner join (keeps only
         timestamps present in both datasets).
      6) Optionally converts the merged timestamps from UTC to Europe/Berlin (CET/CEST).
      7) Writes the merged dataset to a CSV file and returns it as a DataFrame.

    Notes:
      - All parsing is done in UTC first (tz-aware), which makes resampling and merging robust.
      - If `preferred_timezone` is CET, the conversion uses the IANA timezone
        "Europe/Berlin", which automatically handles CET/CEST daylight saving changes.

    Parameters
    ----------
    preferred_timezone : PreferredTimezone, default PreferredTimezone.CET
        Controls the timezone of the returned `timestamp` column:
          - PreferredTimezone.UTC: keep timestamps in UTC
          - PreferredTimezone.CET: convert timestamps to Europe/Berlin (CET/CEST)

    Returns
    -------
    pd.DataFrame
        A merged DataFrame containing:
          - `timestamp` (tz-aware datetime; UTC or Europe/Berlin depending on preference)
          - target price columns from the target CSV
          - hourly feature columns: `solar`, `wind onshore`, `wind offshore`, `load`

    Raises
    ------
    FileNotFoundError
        If one of the input CSV paths does not exist.
    ValueError
        If timestamps cannot be parsed or resampling/merge operations fail due to invalid data.

    Side Effects
    ------------
    Writes a CSV to `data_csv/` named like:
        dataset_timezone_<preferred_timezone>_<YYYYMMDDTHHMM>.csv
    """

    target_data = pd.read_csv(
        "data_csv/day_ahead_prices_DE-LU_20260224T1502.csv",
        parse_dates=["timestamp"],
    )

    target_data["timestamp"] = pd.to_datetime(target_data["timestamp"], utc=True)

    target_data_hourly = (target_data.set_index("timestamp")["price"].resample("1h").mean().round(2).reset_index())

    features_data = pd.read_csv(
        "data_csv/public_power_for_chosen_production_types_de_2020-01-01_to_2025-12-31_20260224T1618+0100.csv",
        parse_dates=["timestamp"],
    )
    features_data["timestamp"] = pd.to_datetime(features_data["timestamp"], utc=True)


    feature_cols = ["Solar", "Wind onshore", "Wind offshore", "Load", "Cross border electricity trading"]
    hourly_features_data = (
        features_data
        .set_index("timestamp")[feature_cols]
        .resample("1h")
        .mean().round(2)
        .reset_index()
    )
    hourly_features_data = hourly_features_data.rename(columns={"Solar": "solar",
                                                                "Wind onshore": "wind onshore",
                                                                "Wind offshore": "wind offshore",
                                                                "Load": "load",
                                                                "Cross border electricity trading": "cross border electricity trading"})

    complete_data = pd.merge(target_data_hourly, hourly_features_data, on="timestamp", how="inner")

    ts = datetime.now(ZoneInfo("Europe/Berlin")).strftime("%Y%m%dT%H%M")
    if preferred_timezone == PreferredTimezone.CET:
        berlin = ZoneInfo("Europe/Berlin")
        complete_data["timestamp"] = complete_data["timestamp"].dt.tz_convert(berlin)
        complete_data.to_csv(f"data_csv/dataset_CET_{ts}.csv", index=False)
    else:
        complete_data.to_csv(f"data_csv/dataset_UTC_{ts}.csv", index=False)

    return complete_data


if __name__ == "__main__":
    merge_features_with_target(preferred_timezone=PreferredTimezone.CET)
    merge_features_with_target(preferred_timezone=PreferredTimezone.UTC)