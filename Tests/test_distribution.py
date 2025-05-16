import pandas as pd

def test_income_distribution(df):
    lower = 50_000
    upper = 10_000_000
    ratio_within = df["Income"].between(lower, upper).mean()

    print(f"Income within [{lower}, {upper}]: {ratio_within:.2%}")

    if ratio_within < 0.98:
        raise AssertionError("❌ Income outliers exceed acceptable threshold.")



def test_house_ownership_distribution(df):
    expected_categories = {"rented", "owned", "norent_noown"}
    unique_values = set(df["House_Ownership"].unique())

    print(f"Unique values in House_Ownership: {unique_values}")

    if not unique_values.issubset(expected_categories):
        unexpected = unique_values - expected_categories
        raise AssertionError(f"❌ Unexpected values in House_Ownership: {unexpected}")
