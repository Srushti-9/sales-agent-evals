"""Generate additional realistic sales rows and append them to the demo Parquet.

The bundled dataset covers 2021-11 through 2024-03. This script extends it with
reproducible, realistic rows for every day after the last present date up to a
cutoff (default 2026-09-04), learning the store / SKU / price / promo structure
from the existing data so the new rows share its shape.

Reproducible: a fixed --seed makes the output deterministic. Idempotent: it only
appends dates strictly after the current max date, so re-running never
duplicates rows.

    python scripts/generate_data.py                 # extend to 2026-09-04
    python scripts/generate_data.py --until 2025-12-31 --seed 7
"""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd

DATA_FILE = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "Store_Sales_Price_Elasticity_Promotions_Data.parquet"
)

DTYPES = {
    "Store_Number": "int16",
    "SKU_Coded": "int32",
    "Product_Class_Code": "int16",
    "Qty_Sold": "int16",
    "Total_Sale_Value": "float32",
    "On_Promo": "int8",
}


def _learn_catalog(df: pd.DataFrame):
    """Derive the SKU catalog, per-SKU unit price, store weights, and rates."""
    unit = df["Total_Sale_Value"] / df["Qty_Sold"]
    sku = (
        df.assign(unit=unit)
        .groupby("SKU_Coded")
        .agg(
            product_class=("Product_Class_Code", "first"),
            price_mean=("unit", "mean"),
            price_std=("unit", "std"),
            weight=("SKU_Coded", "size"),
        )
        .reset_index()
    )
    sku["price_std"] = sku["price_std"].fillna(0.0).clip(lower=0.0)

    store_numbers = np.sort(df["Store_Number"].unique())
    store_weights = (
        df["Store_Number"].value_counts().reindex(store_numbers).to_numpy(float)
    )
    store_weights /= store_weights.sum()

    # Empirical Qty_Sold distribution (heavily skewed toward 1).
    qty_vals, qty_counts = np.unique(df["Qty_Sold"].to_numpy(), return_counts=True)
    qty_probs = qty_counts / qty_counts.sum()

    promo_rate = float((df["On_Promo"] == 1).mean())
    rows_per_day = int(round(len(df) / df["Sold_Date"].nunique()))

    return sku, store_numbers, store_weights, qty_vals, qty_probs, promo_rate, rows_per_day


def _seasonal_factor(day: dt.date) -> float:
    """Mild year-over-year growth + a Nov/Dec holiday bump."""
    growth = 1.0 + 0.06 * (day.year - 2024)  # ~6% per year after 2024
    holiday = 1.35 if day.month in (11, 12) else 1.0
    return growth * holiday


def generate(until: dt.date, seed: int) -> pd.DataFrame:
    df = pd.read_parquet(DATA_FILE)
    last_date = max(df["Sold_Date"])
    start = last_date + dt.timedelta(days=1)
    if start > until:
        print(f"Nothing to do: data already reaches {last_date} (>= {until}).")
        return df

    rng = np.random.default_rng(seed)
    (
        sku,
        store_numbers,
        store_weights,
        qty_vals,
        qty_probs,
        promo_rate,
        rows_per_day,
    ) = _learn_catalog(df)

    sku_idx = np.arange(len(sku))
    sku_weights = sku["weight"].to_numpy(float)
    sku_weights /= sku_weights.sum()

    days = pd.date_range(start, until, freq="D").date
    chunks = []
    for day in days:
        n = max(1, int(round(rows_per_day * _seasonal_factor(day))))

        picks = rng.choice(sku_idx, size=n, p=sku_weights)
        chosen = sku.iloc[picks]

        qty = rng.choice(qty_vals, size=n, p=qty_probs).astype("int16")

        price = rng.normal(
            chosen["price_mean"].to_numpy(), chosen["price_std"].to_numpy()
        )
        price = np.clip(price, 0.01, None)
        total = (price * qty).astype("float32")

        chunks.append(
            pd.DataFrame(
                {
                    "Store_Number": rng.choice(
                        store_numbers, size=n, p=store_weights
                    ).astype("int16"),
                    "SKU_Coded": chosen["SKU_Coded"].to_numpy("int32"),
                    "Product_Class_Code": chosen["product_class"].to_numpy("int16"),
                    "Sold_Date": [day] * n,
                    "Qty_Sold": qty,
                    "Total_Sale_Value": total,
                    "On_Promo": (rng.random(n) < promo_rate).astype("int8"),
                }
            )
        )

    new = pd.concat(chunks, ignore_index=True)
    combined = pd.concat([df, new], ignore_index=True)
    combined = combined.sort_values("Sold_Date").reset_index(drop=True)
    for col, dtype in DTYPES.items():
        combined[col] = combined[col].astype(dtype)

    print(f"Existing rows: {len(df):,} (through {last_date})")
    print(f"Generated:     {len(new):,} rows, {start} -> {until}")
    print(f"Total:         {len(combined):,} rows")
    return combined


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--until", default="2026-09-04", help="Inclusive end date (YYYY-MM-DD)")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for reproducibility")
    args = parser.parse_args()

    until = dt.date.fromisoformat(args.until)
    combined = generate(until, args.seed)
    combined.to_parquet(DATA_FILE, index=False)
    print(f"Wrote {DATA_FILE}")


if __name__ == "__main__":
    main()
