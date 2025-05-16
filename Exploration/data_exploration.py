import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import kagglehub
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
import os
from helper_functions import get_data_path

def get_data ():
    dataset_path = kagglehub.dataset_download("rohit265/loan-approval-dataset")

    json_files = [f for f in os.listdir(dataset_path) if f.endswith(".json")]
    if not json_files:
        raise FileNotFoundError("No JSON files found in dataset folder.")
    json_path = os.path.join(dataset_path, json_files[0])

    with open(json_path, "r") as f:
        data = json.load(f)

    df = pd.DataFrame(data)
    return df

def get_info(df):
    pd.set_option('display.max_columns', None)
    print(df.head())
    print(df.describe(include='all'))
    for col in df.columns:
        null_count = df[col].isnull().sum()
        dtype = df[col].dtype
        print(f"{col:35} | Nulls: {null_count:5} | Type: {dtype}")


# NOT NEEDED --> OLD
# Checking variable importance to decide which columns are important and should therefore be tested for 0 values by variable importance of random forrest
def get_variable_importance(df):
    X = df.drop(columns=['Risk_Flag'])  # Features
    y = df['Risk_Flag']  # Target label

    X = pd.get_dummies(X, drop_first=True)

    # Fill remaining missing values (required to fit model)
    X = X.fillna(-999)


    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,  # 20% test set
        random_state=42,  # For reproducibility
        stratify=y  # Ensures class balance
    )

    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    # Get feature importance
    importance = model.feature_importances_

    # Match with feature names
    feature_importance = pd.Series(importance, index=X_train.columns)
    feature_importance = feature_importance.sort_values(ascending=False)

    # Print top 10
    print(feature_importance.head(10))

def preprocess_raw_data(df):
    # Drop ID column since not needed
    if 'SK_ID_CURR' in df.columns:
        df = df.drop(columns=['SK_ID_CURR'])

    # Drop columns with >40% missing values
    #missing_thresh = 0.4
    #df = df.loc[:, df.isnull().mean() < missing_thresh]

    # One-hot encode categoricals
    df = pd.get_dummies(df, drop_first=True)

    return df

def plot_income_distribution(df):
    plt.figure(figsize=(12, 6))
    sns.histplot(df["Income"], bins=100, kde=True)
    plt.title("Income Distribution")
    plt.xlabel("Income")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.show()

def summarize_income(df):
    income = df["Income"]
    print("Income Summary Statistics:")
    print(f"Min:        {income.min():,.0f}")
    print(f"Max:        {income.max():,.0f}")
    print(f"Mean:       {income.mean():,.0f}")
    print(f"Median:     {income.median():,.0f}")
    print(f"Std Dev:    {income.std():,.0f}")
    print(f"25th pct:   {income.quantile(0.25):,.0f}")
    print(f"75th pct:   {income.quantile(0.75):,.0f}")
    print(f"95th pct:   {income.quantile(0.95):,.0f}")
    print(f"99th pct:   {income.quantile(0.99):,.0f}")

#df = get_data()
#get_info(df)
#get_variable_importance(df)
#df.to_csv('C:/Users/vikho/PycharmProjects/ML_Ops_T1/Data/master_dataset.csv', index=False)
df = pd.read_csv(get_data_path("master_dataset.csv"))
plot_income_distribution(df)
summarize_income(df)

