import csv
import os
import pandas as pd

CLEANING_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(CLEANING_DIR)
RAW_DATA_PATH = os.path.join(PROJECT_DIR, "data", "hotel_bookings.csv")
CLEANED_DATA_PATH = os.path.join(CLEANING_DIR, "cleaned_hotel_bookings.csv")


def _load_mapping(filename: str) -> dict[str, str]:
    path = os.path.join(CLEANING_DIR, filename)
    mapping: dict[str, str] = {}
    if os.path.exists(path):
        with open(path, mode="r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                code = (row.get("code") or "").strip()
                name = (row.get("name") or "").strip()
                if code:
                    mapping[code] = name
    return mapping


COUNTRY_NAMES: dict[str, str] = _load_mapping("countries.csv")
MEAL_PLAN_NAMES: dict[str, str] = _load_mapping("meal_plans.csv")
ROOM_TYPE_NAMES: dict[str, str] = _load_mapping("room_types.csv")
DISTRIBUTION_CHANNEL_NAMES: dict[str, str] = _load_mapping("distribution_channels.csv")


def format_country(code: str) -> str:
    return COUNTRY_NAMES.get(code, code)


def format_meal_plan(code: str) -> str:
    return MEAL_PLAN_NAMES.get(code, code)


def format_room_type(code: str) -> str:
    return ROOM_TYPE_NAMES.get(code, code)


def format_distribution_channel(code: str) -> str:
    return DISTRIBUTION_CHANNEL_NAMES.get(code, code)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["children"] = d["children"].fillna(0)
    d["country"] = d["country"].fillna("Unknown")
    d = d.drop(
        columns=[
            "agent",
            "company",
            "reservation_status",
            "reservation_status_date",
            "assigned_room_type",
        ],
        errors="ignore",
    )
    d = d[(d["adults"] + d["children"] + d["babies"]) > 0]
    d = d[(d["stays_in_weekend_nights"] + d["stays_in_week_nights"]) > 0]
    d = d[(d["adr"] >= 0) & (d["adr"] < 1000)]
    return d.drop_duplicates().reset_index(drop=True)


def clean_and_save(
    raw_path: str = RAW_DATA_PATH,
    cleaned_path: str = CLEANED_DATA_PATH,
) -> pd.DataFrame:
    raw_df = pd.read_csv(raw_path)
    cleaned_df = clean(raw_df)
    os.makedirs(os.path.dirname(cleaned_path), exist_ok=True)
    cleaned_df.to_csv(cleaned_path, index=False)
    print(f"Cleaned data: {len(raw_df)} -> {len(cleaned_df)} rows. Saved to {cleaned_path}")
    return cleaned_df
