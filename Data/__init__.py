BIDDING_ZONE = "DE-LU"
START_DATE = "2020-01-01"
END_DATE = "2025-12-31"
TIMEZONE = "Europe/Berlin"
COUNTRY = "de"
FORECAST_TYPE = "day-ahead"
PRODUCTION_TYPE = ["Solar", "Wind onshore", "Wind offshore", "Load", "Cross border electricity trading"]

from enum import Enum

class PreferredTimezone(str, Enum):
    UTC = "utc"
    CET = "Europe/Berlin"