import pandas as pd
import matplotlib.pyplot as plt
from zoneinfo import ZoneInfo
from pandas.plotting import autocorrelation_plot


def statistical_analysis_of_data(
    path: str = "data_csv/dataset_CET_20260226T2242.csv",
    tz: str = "Europe/Berlin",
    target_col: str = "price",
    feature_cols: list[str] | None = None,
) -> pd.DataFrame:
    """
    Basic initial EDA for an energy time-series dataset:
      - Load CSV, parse timestamps (tz-aware)
      - Print dataset info, missing values, describe()
      - Add time features (Year, Month, Hour, Weekday, DayOfYear)
      - Group summaries by Year/Month/Hour/Weekday
      - Correlation analysis (including target correlations)
      - Plots: time series, distributions, monthly averages, correlation heatmap, autocorrelation

    Returns the enriched DataFrame (with time features).
    """
    if feature_cols is None:
        # default set (adjust to your actual column names)
        feature_cols = ["solar", "wind onshore", "wind offshore", "load"]

    # Load
    dataset = pd.read_csv(path)

    # Timestamp parsing (robust for tz-aware strings)
    dataset["timestamp"] = (
        pd.to_datetime(dataset["timestamp"], utc=True, errors="coerce")
        .dt.tz_convert(ZoneInfo(tz))
    )
    dataset = dataset.dropna(subset=["timestamp"]).sort_values("timestamp")

    # Basic info
    print("INFO")
    print(dataset.info())

    print("\n MISSING VALUES")
    print(dataset.isna().sum().sort_values(ascending=False))

    print("\nDESCRIBE (NUMERIC)")
    print(dataset.describe())

    # Time features
    dataset["Year"] = dataset["timestamp"].dt.year
    dataset["Month"] = dataset["timestamp"].dt.month
    dataset["Hour"] = dataset["timestamp"].dt.hour
    dataset["Weekday"] = dataset["timestamp"].dt.weekday  # 0=Mon ... 6=Sun
    dataset["DayOfYear"] = dataset["timestamp"].dt.dayofyear

    #  Grouped summaries
    print("\nYEARLY MEANS")
    print(dataset.groupby("Year").mean(numeric_only=True))

    print("\nMONTHLY MEANS (Year, Month)")
    print(dataset.groupby(["Year", "Month"]).mean(numeric_only=True))

    print("\n HOURLY MEANS (0-23)")
    print(dataset.groupby("Hour").mean(numeric_only=True))

    print("\WEEKDAY MEANS (0=Mon..6=Sun) ")
    print(dataset.groupby("Weekday").mean(numeric_only=True))

    # Target-specific quick stats
    if target_col in dataset.columns:
        print("\n TARGET QUICK CHECKS ")
        neg_rate = (dataset[target_col] < 0).mean()
        print(f"Share of negative {target_col}: {neg_rate:.3%}")

    #  Correlation analysis
    print("\n CORRELATIONS ")
    corr = dataset.corr(numeric_only=True)
    if target_col in corr.columns:
        print(f"\nTop correlations with '{target_col}':")
        print(corr[target_col].sort_values(ascending=False))

    # Plots
    # 1) Time series plot for target + selected features
    plot_cols = [c for c in [target_col] if c in dataset.columns]
    if plot_cols:
        plt.figure()
        dataset.set_index("timestamp")[plot_cols].plot()
        plt.title("Time series")
        plt.xlabel("timestamp")
        plt.tight_layout()
        plt.show()

    plot_cols = [c for c in [*feature_cols] if c in dataset.columns]
    if plot_cols:
        plt.figure()
        dataset.set_index("timestamp")[plot_cols].plot()
        plt.title("Time series")
        plt.xlabel("timestamp")
        plt.tight_layout()
        plt.show()

    # 2) Distribution of target
    if target_col in dataset.columns:
        plt.figure()
        dataset[target_col].hist(bins=60)
        plt.title(f"Distribution of {target_col}")
        plt.xlabel(target_col)
        plt.ylabel("count")
        plt.tight_layout()
        plt.show()

    # 3) Monthly average of target
    if target_col in dataset.columns:
        monthly = dataset.set_index("timestamp")[target_col].resample("1M").mean()
        plt.figure()
        monthly.plot()
        plt.title(f"Monthly average {target_col}")
        plt.xlabel("timestamp")
        plt.ylabel(target_col)
        plt.tight_layout()
        plt.show()

    # 4) Correlation heatmap
    if not corr.empty:
        plt.figure()
        plt.imshow(corr.values)
        plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
        plt.yticks(range(len(corr.index)), corr.index)
        plt.title("Correlation matrix (numeric)")
        plt.colorbar()
        plt.tight_layout()
        plt.show()

    # 5) Autocorrelation of target
    if target_col in dataset.columns:
        plt.figure()
        autocorrelation_plot(dataset[target_col].dropna())
        plt.title(f"Autocorrelation: {target_col}")
        plt.tight_layout()
        plt.show()

    return dataset


if __name__ == "__main__":
    complete_data = statistical_analysis_of_data()
