import pandas as pd
from mlxtend.frequent_patterns import apriori
from utils import fetch_data_from_csv, convert_to_transactions, one_hot_encode

def run_apriori(file_path, min_support):
    # Fetch data from CSV and process transactions
    data = fetch_data_from_csv(file_path)
    transactions = convert_to_transactions(data)
    one_hot_encoded_df = one_hot_encode(transactions)

    # Run Apriori algorithm
    apriori_results = apriori(one_hot_encoded_df, min_support=min_support, use_colnames=True)
    apriori_results_list = apriori_results.to_dict('records')

    # Sort the itemsets by support in descending order
    apriori_results_list = sorted(apriori_results_list, key=lambda x: x['support'], reverse=True)

    return apriori_results_list
