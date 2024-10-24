import pandas as pd
import json

# Function to fetch data from a CSV file
def fetch_data_from_csv(file_path):
    data = pd.read_csv(file_path)
    return data

# Function to convert data into transactions
def convert_to_transactions(data):
    transactions = data.groupby('Customer Name')['Product Code'].apply(list).tolist()
    return transactions

# Function to perform one-hot encoding on transactions
def one_hot_encode(transactions):
    df = pd.DataFrame(transactions)
    melted = df.melt(var_name='transaction', value_name='product')
    melted = melted.dropna()
    one_hot = pd.crosstab(melted['transaction'], melted['product'])
    one_hot = (one_hot > 0).astype(int)
    return one_hot

# Custom JSON encoder to handle frozenset objects
class SetEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, frozenset):
            return list(obj)
        return json.JSONEncoder.default(self, obj)
