from flask import Flask, request, jsonify
from algorithms.apriori import run_apriori
from algorithms.eclat import run_eclat
from utils import SetEncoder
import json
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from flask_cors import CORS
app = Flask(__name__)

# Enable CORS globally for all routes
CORS(app)
# Apriori route
@app.route('/apriori', methods=['GET'])
def apriori_route():
    file_path = request.args.get('file_path', 'sample_sheet.csv')
    min_support = float(request.args.get('min_support', 0.01))  # Default min support is 0.01

    # Call the Apriori function
    apriori_results = run_apriori(file_path, min_support)

    # Prepare the response
    results = {
        "frequent_itemsets": apriori_results
    }

    return app.response_class(
        response=json.dumps(results, cls=SetEncoder),
        status=200,
        mimetype='application/json'
    )

# Eclat route
@app.route('/eclat', methods=['GET'])
def eclat_route():
    file_path = request.args.get('file_path', 'sample_sheet.csv')
    min_support = float(request.args.get('min_support', 0.01))  # Default min support is 0.01

    # Call the Eclat function
    eclat_results = run_eclat(file_path, min_support)

    # Prepare the response
    results = {
        "frequent_itemsets": eclat_results
    }

    return app.response_class(
        response=json.dumps(results, cls=SetEncoder),
        status=200,
        mimetype='application/json'
    )

pipeline = None

def train_model():
    global pipeline  # Make sure this is set globally
    
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
        ('one_hot', OneHotEncoder( handle_unknown='ignore')),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
    ])

    # Fit the pipeline to the training data
    pipeline.fit(X_train, y_train)

# Route to predict quantity
@app.route('/sckikit', methods=['POST'])
def sckikit_learn_route():
    global pipeline  # Access global pipeline
    
    # Ensure the pipeline is trained before making predictions
    if pipeline is None:
        return jsonify({"error": "Model not trained. Please train the model first."})

    # Get the data from the request
    data = request.get_json()

    try:
        # Convert the incoming data into a DataFrame
        df = pd.DataFrame(data, index=[0])

        # Convert 'Change %' to numeric in the prediction data
        df['Change %'] = df['Change %'].str.rstrip('%').astype(float) / 100.0

        # Make predictions using the trained pipeline
        predicted_data = pipeline.predict(df)

        # Return the predicted quantity as a JSON response
        return jsonify({"Predicted Quantity": predicted_data.tolist()})

    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/TrainSckikit', methods=['POST'])
def train_sckikit_learn():
    try:
        train_model()  # Call the function that trains the model
        return jsonify({"message": "Model training completed successfully!"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

