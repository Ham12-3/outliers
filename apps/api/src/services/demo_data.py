"""Demo data generation with realistic patterns and anomalies."""
import random
from datetime import date, timedelta
from typing import List, Tuple

import numpy as np
from sqlalchemy.orm import Session

from ..database import RawDailyMetric

# Store names for realistic IDs
STORE_PREFIXES = ["LONDON", "MANCHESTER", "BIRMINGHAM", "LEEDS", "BRISTOL"]
SKU_CATEGORIES = ["ELEC", "FOOD", "CLOTH", "HOME", "SPORT"]


def generate_demo_data(
    db: Session,
    stores: int = 5,
    skus: int = 50,
    days: int = 120,
) -> int:
    """
    Generate realistic synthetic data with various anomaly patterns.

    Anomaly types injected:
    1. Sudden shrinkage: on_hand drops without corresponding sold (theft/damage)
    2. Demand spike: sold jumps significantly for a few days
    3. Supplier issue: delivered drops to zero for a week
    4. Return surge: unusually high returns
    5. Price anomaly: unexpected price changes during promo

    Returns:
        Total number of rows inserted
    """
    end_date = date.today()
    start_date = end_date - timedelta(days=days)

    store_ids = [f"{STORE_PREFIXES[i % len(STORE_PREFIXES)]}-{i+1:03d}" for i in range(stores)]

    # Generate SKU IDs with categories
    sku_ids = []
    for i in range(skus):
        category = SKU_CATEGORIES[i % len(SKU_CATEGORIES)]
        sku_ids.append(f"{category}-{i+1:05d}")

    # SKU base parameters (simulating different product characteristics)
    sku_params = {}
    for sku_id in sku_ids:
        sku_params[sku_id] = {
            "base_price": random.uniform(5, 200),
            "base_sold": random.randint(5, 50),
            "base_on_hand": random.randint(50, 500),
            "base_delivered": random.randint(10, 40),
            "seasonality": random.uniform(0.5, 1.5),  # Seasonal factor
            "weekend_factor": random.uniform(1.0, 1.8),  # Weekend sales multiplier
        }

    # Pre-plan anomalies for each store-sku combination
    anomalies = _plan_anomalies(store_ids, sku_ids, start_date, days)

    rows_to_insert = []
    total_rows = 0

    for store_id in store_ids:
        for sku_id in sku_ids:
            params = sku_params[sku_id]
            on_hand = params["base_on_hand"]
            store_sku_anomalies = anomalies.get((store_id, sku_id), [])

            for day_offset in range(days):
                current_date = start_date + timedelta(days=day_offset)
                day_of_week = current_date.weekday()

                # Check for active anomalies
                active_anomaly = None
                for anomaly in store_sku_anomalies:
                    if anomaly["start"] <= current_date <= anomaly["end"]:
                        active_anomaly = anomaly
                        break

                # Base calculations with natural variance
                is_weekend = day_of_week >= 5
                weekend_mult = params["weekend_factor"] if is_weekend else 1.0

                # Seasonal pattern (simple sine wave over the period)
                seasonal = 1 + 0.3 * np.sin(2 * np.pi * day_offset / 30) * params["seasonality"]

                # Calculate sold with natural variance
                base_sold = int(params["base_sold"] * weekend_mult * seasonal)
                sold = max(0, int(np.random.normal(base_sold, base_sold * 0.2)))

                # Calculate delivered (usually on certain days)
                delivered = 0
                if day_of_week in [1, 3]:  # Tuesday and Thursday deliveries
                    delivered = max(0, int(np.random.normal(params["base_delivered"], 5)))

                # Calculate returns (small percentage of recent sales)
                returned = max(0, int(np.random.poisson(sold * 0.03)))

                # Promo flag (occasional promotions)
                promo_flag = random.random() < 0.1

                # Apply anomaly effects
                if active_anomaly:
                    anomaly_type = active_anomaly["type"]

                    if anomaly_type == "shrinkage":
                        # Sudden inventory drop without explanation
                        if current_date == active_anomaly["start"]:
                            on_hand = max(0, on_hand - random.randint(50, 150))
                        sold = max(0, int(sold * 0.7))  # Lower sales due to stock issues

                    elif anomaly_type == "demand_spike":
                        # Dramatic increase in sales
                        sold = int(sold * random.uniform(3, 5))
                        if promo_flag:
                            sold = int(sold * 1.5)

                    elif anomaly_type == "supplier_issue":
                        # No deliveries
                        delivered = 0

                    elif anomaly_type == "return_surge":
                        # High returns
                        returned = int(sold * random.uniform(0.2, 0.4))

                    elif anomaly_type == "price_anomaly":
                        # Price doesn't match promo expectation
                        promo_flag = True

                # Update on_hand (simplified inventory equation)
                on_hand = max(0, on_hand + delivered - sold + returned)

                # Price with some variance
                price = params["base_price"]
                if promo_flag and not (active_anomaly and active_anomaly["type"] == "price_anomaly"):
                    price = price * 0.8  # 20% discount on promo

                rows_to_insert.append(
                    RawDailyMetric(
                        date=current_date,
                        store_id=store_id,
                        sku_id=sku_id,
                        on_hand=on_hand,
                        sold=sold,
                        delivered=delivered,
                        returned=returned,
                        price=round(price, 2),
                        promo_flag=promo_flag,
                    )
                )
                total_rows += 1

                # Batch insert every 5000 rows
                if len(rows_to_insert) >= 5000:
                    db.bulk_save_objects(rows_to_insert)
                    db.commit()
                    rows_to_insert = []

    # Insert remaining rows
    if rows_to_insert:
        db.bulk_save_objects(rows_to_insert)
        db.commit()

    return total_rows


def _plan_anomalies(
    store_ids: List[str],
    sku_ids: List[str],
    start_date: date,
    days: int,
) -> dict:
    """
    Pre-plan anomalies across store-sku combinations.

    Ensures anomalies are distributed realistically:
    - Not too many anomalies
    - Anomalies spread across different periods
    - Some stores/SKUs have multiple anomalies, most have none
    """
    anomalies = {}
    anomaly_types = ["shrinkage", "demand_spike", "supplier_issue", "return_surge", "price_anomaly"]

    # Target about 5-10% of store-sku combinations to have anomalies
    total_combinations = len(store_ids) * len(sku_ids)
    num_anomalies = int(total_combinations * random.uniform(0.05, 0.10))

    for _ in range(num_anomalies):
        store_id = random.choice(store_ids)
        sku_id = random.choice(sku_ids)
        anomaly_type = random.choice(anomaly_types)

        # Anomaly timing - prefer recent dates (more interesting for demo)
        # Weight towards last 60 days
        if random.random() < 0.7:
            day_offset = random.randint(days - 60, days - 7)
        else:
            day_offset = random.randint(14, days - 7)

        anomaly_start = start_date + timedelta(days=day_offset)

        # Duration varies by type
        if anomaly_type == "shrinkage":
            duration = 1  # Single day event
        elif anomaly_type == "demand_spike":
            duration = random.randint(2, 5)
        elif anomaly_type == "supplier_issue":
            duration = random.randint(5, 10)
        elif anomaly_type == "return_surge":
            duration = random.randint(3, 7)
        else:  # price_anomaly
            duration = random.randint(1, 3)

        anomaly_end = anomaly_start + timedelta(days=duration - 1)

        key = (store_id, sku_id)
        if key not in anomalies:
            anomalies[key] = []

        anomalies[key].append(
            {
                "type": anomaly_type,
                "start": anomaly_start,
                "end": anomaly_end,
            }
        )

    return anomalies
