import pandas as pd
from datetime import datetime


def parse_date(s):
    for fmt in ("%d-%m-%Y", "%m-%d-%Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(str(s).strip(), fmt)
        except Exception:
            pass
    return None


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    for col in ["Sales", "Units", "Gross Profit", "Cost"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    return df


def compute_lead_times(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Order_Date_parsed"] = df["Order Date"].apply(parse_date)
    df["Ship_Date_parsed"] = df["Ship Date"].apply(parse_date)
    df["Lead_Time"] = (df["Ship_Date_parsed"] - df["Order_Date_parsed"]).dt.days.fillna(0).astype(int)
    return df
