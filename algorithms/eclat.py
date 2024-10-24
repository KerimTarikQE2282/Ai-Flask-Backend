import pandas as pd
from utils import fetch_data_from_csv, convert_to_transactions, one_hot_encode

def run_eclat(file_path, min_support):
    # Fetch data from CSV and process transactions
    data = fetch_data_from_csv(file_path)
    transactions = convert_to_transactions(data)
    one_hot_encoded_df = one_hot_encode(transactions)

    # Run ECLAT algorithm (a simplified version of Apriori focusing on itemsets)
    frequent_itemsets = eclat(one_hot_encoded_df, min_support)

    return frequent_itemsets

# ECLAT algorithm function
def eclat(one_hot_encoded_df, min_support):
    frequent_items = []
    support_count = one_hot_encoded_df.sum()
    total_transactions = len(one_hot_encoded_df)

    # Calculate support for each itemset
    for item, count in support_count.items():
        support = count / total_transactions
        if support >= min_support:
            frequent_items.append({
                "itemset": [item],
                "support": support
            })

    return frequent_items
