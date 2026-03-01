import pandas as pd
from pathlib import Path


# Raw Data Loaders

def load_events(data_path: str) -> pd.DataFrame:
    """
    Load events.csv from the given directory.

    Returns the raw events dataframe with timestamp converted to datetime.
    """
    base_path = Path(data_path)
    file_path = base_path / "events.csv"

    if not file_path.exists():
        raise FileNotFoundError(f"events.csv not found at: {file_path}")

    df = pd.read_csv(file_path)

    # RetailRocket timestamps are in milliseconds
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")

    return df


def load_item_properties(data_path: str) -> pd.DataFrame:
    """
    Load and concatenate item_properties_part1 and part2.
    Timestamp is converted to datetime.
    """
    base_path = Path(data_path)
    file_1 = base_path / "item_properties_part1.csv"
    file_2 = base_path / "item_properties_part2.csv"

    if not file_1.exists():
        raise FileNotFoundError(f"item_properties_part1.csv not found at: {file_1}")

    if not file_2.exists():
        raise FileNotFoundError(f"item_properties_part2.csv not found at: {file_2}")

    df1 = pd.read_csv(file_1)
    df2 = pd.read_csv(file_2)

    df = pd.concat([df1, df2], ignore_index=True)

    # Convert timestamp (milliseconds → datetime)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")

    return df


# Transformation Helpers


def extract_transactions(events_df: pd.DataFrame) -> pd.DataFrame:
    """
    Return only transaction rows from events dataframe.
    Keeps visitorid, itemid, timestamp and other columns.
    """
    transactions_df = (
        events_df[events_df["event"] == "transaction"]
        .copy()
        .reset_index(drop=True)
    )

    return transactions_df


def extract_price_table(item_properties_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract price rows from item properties.
    Keeps itemid, timestamp, and numeric price value.
    """
    price_df = (
        item_properties_df[item_properties_df["property"] == "price"]
        [["itemid", "timestamp", "value"]]
        .copy()
        .reset_index(drop=True)
    )

    # Convert price to numeric; invalid entries become NaN
    price_df["value"] = pd.to_numeric(price_df["value"], errors="coerce")

    # Drop rows with missing itemid (defensive safeguard)
    price_df = price_df.dropna(subset=["itemid"])

    return price_df


def get_latest_price_per_item(price_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each itemid, keep the most recent price based on timestamp.
    """
    latest_price_df = (
        price_df
        .sort_values("timestamp", ascending=False)
        .drop_duplicates(subset="itemid", keep="first")
        .reset_index(drop=True)
    )

    return latest_price_df


def compute_user_revenue(
    transactions_df: pd.DataFrame,
    latest_price_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Compute total revenue per user.

    Transactions are left-joined with price table.
    Missing prices are treated as 0.
    """
    merged_df = transactions_df.merge(
        latest_price_df,
        on="itemid",
        how="left"
    )

    # If price lookup fails, treat as 0 revenue
    merged_df["value"] = merged_df["value"].fillna(0)

    revenue_df = (
        merged_df
        .groupby("visitorid")["value"]
        .sum()
        .reset_index()
        .rename(columns={"value": "revenue"})
    )

    return revenue_df


def build_conversion_table(
    events_df: pd.DataFrame,
    transactions_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Build user-level conversion table.

    A user is marked as converted if they have at least one transaction.
    """
    # All exposed users (denominator)
    all_users = (
        events_df[["visitorid"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    # Users who converted (numerator)
    converted_users = transactions_df["visitorid"].unique()

    all_users["converted"] = (
        all_users["visitorid"]
        .isin(converted_users)
        .astype(int)
    )

    return all_users


# Orchestration


def build_user_level_dataset(data_path: str) -> pd.DataFrame:
    """
    Full pipeline.

    Returns user-level dataset:
        visitorid | converted | revenue
    """

    # Load raw data
    events_df = load_events(data_path)
    item_props_df = load_item_properties(data_path)

    # Extract transactions
    transactions_df = extract_transactions(events_df)

    # Build conversion table (denominator preserved)
    conversion_df = build_conversion_table(events_df, transactions_df)

    # Build revenue table
    price_df = extract_price_table(item_props_df)
    latest_price_df = get_latest_price_per_item(price_df)
    revenue_df = compute_user_revenue(transactions_df, latest_price_df)

    # Merge conversion + revenue
    user_df = conversion_df.merge(
        revenue_df,
        on="visitorid",
        how="left"
    )

    # Non-purchasing users have zero revenue
    user_df["revenue"] = user_df["revenue"].fillna(0)

    return user_df