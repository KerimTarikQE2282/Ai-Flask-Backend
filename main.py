from flask import Flask, request, jsonify
from algorithms.apriori import run_apriori
from algorithms.eclat import run_eclat
from utils import SetEncoder
import json

app = Flask(__name__)

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

if __name__ == "__main__":
    app.run(debug=True)
