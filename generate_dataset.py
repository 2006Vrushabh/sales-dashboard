"""
generate_dataset.py
-------------------
Creates the sample sales dataset (data/sales_dataset.csv) used as the default
dataset in the Data Mining mini project.

The dataset is generated ON PURPOSE with small "defects" (missing values and
duplicate rows) so that the Data Preprocessing page of the Streamlit app has
something real to clean. This makes the viva demonstration meaningful.

Run once:  python data/generate_dataset.py
"""

import numpy as np
import pandas as pd

# A fixed random seed means the dataset is the SAME every time it is generated.
# This is important so results shown in the report match the deployed app.
RANDOM_SEED = 42
NUM_RECORDS = 10000

rng = np.random.default_rng(RANDOM_SEED)


# ----------------------------------------------------------------------------
# 1. Reference data (products, prices, regions, cities)
# ----------------------------------------------------------------------------

# Each product is described as: (Product, Category, min price, max price)
PRODUCT_CATALOG = [
    ("Laptop",            "Electronics", 35000, 95000),
    ("Smartphone",        "Electronics", 9000,  75000),
    ("Headphones",        "Electronics", 700,   9000),
    ("Tablet",            "Electronics", 8000,  45000),
    ("Smart Watch",       "Electronics", 1500,  25000),
    ("Bluetooth Speaker", "Electronics", 900,   12000),

    ("Office Chair",      "Furniture",   3500,  18000),
    ("Study Table",       "Furniture",   4000,  22000),
    ("Bookshelf",         "Furniture",   2500,  14000),
    ("Sofa",              "Furniture",   12000, 60000),
    ("Bed Frame",         "Furniture",   9000,  48000),

    ("T-Shirt",           "Clothing",    300,   1600),
    ("Jeans",             "Clothing",    800,   3500),
    ("Jacket",            "Clothing",    1200,  7000),
    ("Formal Shirt",      "Clothing",    600,   2800),
    ("Sneakers",          "Clothing",    1200,  9000),

    ("Rice Bag",          "Groceries",   400,   1800),
    ("Cooking Oil",       "Groceries",   150,   900),
    ("Tea Pack",          "Groceries",   120,   750),
    ("Coffee Pack",       "Groceries",   200,   1400),
    ("Snacks Combo",      "Groceries",   100,   650),

    ("Notebook Set",      "Stationery",  120,   800),
    ("Pen Pack",          "Stationery",  50,    400),
    ("Printer Paper",     "Stationery",  200,   1200),
    ("Art Kit",           "Stationery",  300,   2500),
    ("Backpack",          "Stationery",  600,   4500),
]

# Region -> list of cities in that region (keeps City consistent with Region)
REGION_CITIES = {
    "North":   ["Delhi", "Jaipur", "Lucknow", "Chandigarh"],
    "South":   ["Bengaluru", "Chennai", "Hyderabad", "Kochi"],
    "East":    ["Kolkata", "Patna", "Bhubaneswar", "Guwahati"],
    "West":    ["Mumbai", "Pune", "Ahmedabad", "Surat"],
    "Central": ["Nagpur", "Bhopal", "Indore", "Raipur"],
}

# Typical profit margin per category (used to compute the Profit column)
CATEGORY_MARGIN = {
    "Electronics": 0.12,
    "Furniture":   0.22,
    "Clothing":    0.35,
    "Groceries":   0.08,
    "Stationery":  0.25,
}

CUSTOMER_TYPES = ["Regular", "Premium", "New", "Corporate"]
PAYMENT_MODES = ["UPI", "Credit Card", "Debit Card", "Cash", "Net Banking", "EMI"]


# ----------------------------------------------------------------------------
# 2. Order dates with simple seasonality
# ----------------------------------------------------------------------------

def generate_order_dates(n):
    """
    Pick n random dates between 01-Jan-2023 and 31-Dec-2024.

    Months are NOT equally likely: October and November get a higher weight so
    the Monthly Sales Trend line chart shows a realistic festive-season peak.
    """
    all_days = pd.date_range("2023-01-01", "2024-12-31", freq="D")

    # Weight for each month (index 0 = January ... index 11 = December)
    month_weight = [0.8, 0.8, 0.9, 1.0, 1.0, 0.9, 0.9, 1.0, 1.1, 1.6, 1.5, 1.2]
    weights = np.array([month_weight[d.month - 1] for d in all_days])
    weights = weights / weights.sum()

    chosen = rng.choice(all_days, size=n, p=weights)
    return pd.to_datetime(chosen)


# ----------------------------------------------------------------------------
# 3. Build the main table
# ----------------------------------------------------------------------------

def build_dataset(n=NUM_RECORDS):
    """Return a clean DataFrame of n sales records."""

    # -- Product / Category / Unit_Price -----------------------------------
    product_index = rng.integers(0, len(PRODUCT_CATALOG), size=n)
    products, categories, unit_prices = [], [], []

    for i in product_index:
        name, category, low, high = PRODUCT_CATALOG[i]
        products.append(name)
        categories.append(category)
        # Round the price to 2 decimals so it looks like real currency
        unit_prices.append(round(float(rng.uniform(low, high)), 2))

    # -- Region / City ------------------------------------------------------
    regions = rng.choice(
        list(REGION_CITIES.keys()),
        size=n,
        p=[0.22, 0.26, 0.16, 0.24, 0.12],   # West and South sell slightly more
    )
    cities = [rng.choice(REGION_CITIES[r]) for r in regions]

    # -- Quantity -----------------------------------------------------------
    # Cheap items (Groceries, Stationery) are bought in larger quantities.
    quantities = []
    for category in categories:
        if category in ("Groceries", "Stationery"):
            quantities.append(int(rng.integers(1, 16)))
        elif category == "Clothing":
            quantities.append(int(rng.integers(1, 7)))
        else:
            quantities.append(int(rng.integers(1, 4)))

    # -- Discount -----------------------------------------------------------
    discounts = rng.choice(
        [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
        size=n,
        p=[0.30, 0.20, 0.18, 0.13, 0.10, 0.06, 0.03],
    )

    # -- Assemble the DataFrame --------------------------------------------
    df = pd.DataFrame({
        "Order_ID": [f"ORD{100001 + i}" for i in range(n)],
        "Order_Date": generate_order_dates(n),
        "Product": products,
        "Category": categories,
        "Quantity": quantities,
        "Unit_Price": unit_prices,
        "Discount": discounts,
        "Region": regions,
        "City": cities,
        "Customer_Type": rng.choice(CUSTOMER_TYPES, size=n,
                                    p=[0.45, 0.20, 0.22, 0.13]),
        "Payment_Mode": rng.choice(PAYMENT_MODES, size=n,
                                   p=[0.34, 0.18, 0.16, 0.14, 0.11, 0.07]),
    })

    # -- Derived columns: Sales and Profit ----------------------------------
    # Sales = Quantity x Unit_Price x (1 - Discount)
    df["Sales"] = (df["Quantity"] * df["Unit_Price"] * (1 - df["Discount"])).round(2)

    # Profit = Sales x (category margin + random noise) - discount penalty.
    # The noise and the penalty let a few orders end up with a LOSS, which is
    # realistic and makes the Sales-vs-Profit scatter plot more interesting.
    base_margin = df["Category"].map(CATEGORY_MARGIN).to_numpy()
    noise = rng.normal(0, 0.05, size=n)
    effective_margin = base_margin + noise - (df["Discount"].to_numpy() * 0.35)
    df["Profit"] = (df["Sales"] * effective_margin).round(2)

    # Put the columns in a readable order
    df = df[[
        "Order_ID", "Order_Date", "Product", "Category", "Quantity",
        "Unit_Price", "Discount", "Sales", "Region", "City",
        "Customer_Type", "Payment_Mode", "Profit",
    ]]

    return df


# ----------------------------------------------------------------------------
# 4. Inject realistic "dirt" so preprocessing has real work to do
# ----------------------------------------------------------------------------

def inject_missing_values(df):
    """Blank out a small percentage of values in selected columns."""
    n = len(df)
    missing_plan = {
        "Discount":      0.030,   # 3.0% missing  -> numeric, fill with mean/median
        "Customer_Type": 0.025,   # 2.5% missing  -> categorical, fill with mode
        "City":          0.020,   # 2.0% missing  -> categorical, fill with mode
        "Profit":        0.015,   # 1.5% missing  -> numeric
        "Quantity":      0.010,   # 1.0% missing  -> numeric
    }
    for column, fraction in missing_plan.items():
        rows = rng.choice(n, size=int(n * fraction), replace=False)
        df.loc[rows, column] = np.nan
    return df


def inject_duplicates(df, num_duplicates=120):
    """Copy some existing rows and append them, so duplicates really exist."""
    rows = rng.choice(len(df), size=num_duplicates, replace=False)
    duplicated_rows = df.iloc[rows].copy()
    df = pd.concat([df, duplicated_rows], ignore_index=True)
    # Shuffle so the duplicates are not all sitting at the bottom
    df = df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    return df


# ----------------------------------------------------------------------------
# 5. Main
# ----------------------------------------------------------------------------

if __name__ == "__main__":
    data = build_dataset(NUM_RECORDS)
    data = inject_missing_values(data)
    data = inject_duplicates(data)

    data.to_csv("sales_dataset.csv", index=False)

    print("Dataset created: sales_dataset.csv")
    print("Rows:", len(data), "| Columns:", data.shape[1])
    print("Total missing values:", int(data.isna().sum().sum()))
    print("Duplicate rows:", int(data.duplicated().sum()))
