import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from utils.factory_info import (
    PRODUCT_FACTORY, FACTORY_COORDS, REGION_OFFSETS, SHIP_MODE_OFFSETS
)

FACTORY_LT_OFFSETS = {
    "Lot's O' Nuts":     -10,
    "Wicked Choccy's":     5,
    "Sugar Shack":        60,
    "Secret Factory":     70,
    "The Other Factory": -50,
}

FACTORY_PROFIT_OFFSETS = {
    "Lot's O' Nuts":      0.0,
    "Wicked Choccy's":    2.0,
    "Sugar Shack":       -3.0,
    "Secret Factory":     1.0,
    "The Other Factory": -5.0,
}

FACTORIES = list(FACTORY_COORDS.keys())


class FactoryPredictor:
    def __init__(self):
        self._product_base_lt: dict = {}
        self._product_base_profit: dict = {}
        self._model = None
        self._encoders: dict = {}
        self._metrics: dict = {}

    # ── Training ────────────────────────────────────────────────────────────
    def fit(self, df: pd.DataFrame):
        grp = df.groupby("Product Name")
        self._product_base_lt = grp["Lead_Time"].mean().to_dict()
        self._product_base_profit = grp["Gross Profit"].sum().to_dict()
        self._train_model(df)

    def _train_model(self, df: pd.DataFrame):
        cat_cols = ["Product Name", "Ship Mode", "Region"]
        self._encoders = {c: LabelEncoder().fit(df[c].fillna("Unknown")) for c in cat_cols}

        features = pd.DataFrame({
            "product":   self._encoders["Product Name"].transform(df["Product Name"].fillna("Unknown")),
            "ship_mode": self._encoders["Ship Mode"].transform(df["Ship Mode"].fillna("Unknown")),
            "region":    self._encoders["Region"].transform(df["Region"].fillna("Unknown")),
            "sales":     df["Sales"],
            "units":     df["Units"],
        })
        target = df["Lead_Time"]

        X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

        rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)

        self._model = rf
        self._metrics = {
            "MAE":  round(mean_absolute_error(y_test, y_pred), 2),
            "RMSE": round(np.sqrt(mean_squared_error(y_test, y_pred)), 2),
            "R2":   round(r2_score(y_test, y_pred), 4),
        }

    @property
    def metrics(self) -> dict:
        return self._metrics

    # ── Prediction ──────────────────────────────────────────────────────────
    def predict_lead_time(
        self,
        product: str,
        factory: str,
        region: str = "All",
        ship_mode: str = "All",
        opt_priority: int = 50,
    ) -> float:
        base = self._product_base_lt.get(product, 1320)
        fac_offset = FACTORY_LT_OFFSETS.get(factory, 0)
        reg_offset = REGION_OFFSETS.get(region, 0)
        ship_offset = SHIP_MODE_OFFSETS.get(ship_mode, 0)
        speed_bias = (50 - opt_priority) * 0.5
        return round(base + fac_offset + reg_offset + ship_offset + speed_bias, 1)

    def estimate_profit_impact(self, product: str, factory: str) -> float:
        return FACTORY_PROFIT_OFFSETS.get(factory, 0.0)

    # ── Recommendations ─────────────────────────────────────────────────────
    def generate_recommendations(self, df: pd.DataFrame, opt_priority: int = 50) -> list[dict]:
        recs = []
        for prod in df["Product Name"].unique():
            cur_fac = PRODUCT_FACTORY.get(prod, "Unknown")
            cur_lt = self.predict_lead_time(prod, cur_fac, "All", "All", opt_priority)
            cur_profit = self._product_base_profit.get(prod, 0)

            best_fac, best_lt = cur_fac, cur_lt
            for f in FACTORIES:
                if f == cur_fac:
                    continue
                lt = self.predict_lead_time(prod, f, "All", "All", opt_priority)
                if lt < best_lt:
                    best_lt, best_fac = lt, f

            if best_fac == cur_fac:
                continue

            lt_gain = (cur_lt - best_lt) / cur_lt * 100
            profit_delta = self.estimate_profit_impact(prod, best_fac)
            profit_risk = "Low" if profit_delta >= 0 else ("Medium" if profit_delta >= -3 else "High")
            confidence = min(95, max(55, 85 - abs(profit_delta) * 3 + lt_gain * 0.5))

            recs.append({
                "Product":              prod,
                "Current Factory":      cur_fac,
                "Recommended Factory":  best_fac,
                "LT Gain %":           round(lt_gain, 1),
                "Profit Impact %":     round(profit_delta, 1),
                "Profit Risk":         profit_risk,
                "Confidence %":        round(confidence, 1),
            })

        return sorted(recs, key=lambda x: (-x["LT Gain %"], x["Profit Impact %"]))
