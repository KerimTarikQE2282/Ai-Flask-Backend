import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

def train_model():
    # Load datasets
    np.random.seed(42)
    sales_data = pd.read_csv('./Untitled spreadsheet - Sheet1.csv')
    USD_data = pd.read_csv('./USD_Historical_Data.csv')

    # Preprocess USD data
    USD_data = USD_data.drop('Vol.', axis=1)
    USD_data.rename(columns={'Date': 'New_Date'}, inplace=True)

    # Merge datasets
    merged_sales_data = USD_data.merge(sales_data, on='New_Date')
    merged_sales_data.drop(columns=["New_Date", "Open", "Total Price", "High", "Low"], inplace=True)

    # Extract month and day of the week
    def extract_month(date_str):
        date_obj = datetime.strptime(date_str, '%d %b, %Y')
        return date_obj.strftime('%B')

    def get_day_of_week(date_str):
        date_obj = datetime.strptime(date_str, '%d %b, %Y')
        return date_obj.strftime('%A')

    merged_sales_data['Day_of_the_week'] = merged_sales_data['Sale Date'].apply(get_day_of_week)
    merged_sales_data['Sale_Month'] = merged_sales_data['Sale Date'].apply(extract_month)

    # Convert 'Change %' to numeric
    merged_sales_data['Change %'] = merged_sales_data['Change %'].str.rstrip('%').astype(float) / 100.0

    # Prepare features and target variable
    X = merged_sales_data.drop("Quantity", axis=1)
    y = merged_sales_data["Quantity"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define categorical features
    categorical_features = ["Customer Name", "Price", "Product Code", "Sale Date", 
                            "Unit Price", "Day_of_the_week", "Sale_Month"]

    # Create the pipeline
    pipeline = Pipeline(steps=[
        ('imputer', ColumnTransformer(
            transformers=[
                ('num_imputer', SimpleImputer(strategy='mean'), ["Price", "Unit Price"]),
                ('cat_imputer', SimpleImputer(strategy='constant', fill_value='missing'), categorical_features)
            ],
            remainder='passthrough'
        )),
        ('one_hot', OneHotEncoder(sparse=False, handle_unknown='ignore')),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
    ])

    # Fit the pipeline to the training data
    pipeline.fit(X_train, y_train)

if __name__ == "__main__":
    # Check if today is the first of the month
    if datetime.now().day == 1:
        train_model()