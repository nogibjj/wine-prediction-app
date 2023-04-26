import json
import boto3
from flask import Flask, request, jsonify
import pandas as pd
import re
import io
import numpy as np

app = Flask(__name__)

credentials = pd.read_csv('FINALprojApp_accessKeys.csv')

# Configure AWS credentials
AWS_ACCESS_KEY_ID = credentials['Access key ID'][0]
AWS_SECRET_ACCESS_KEY = credentials['Secret access key'][0]
AWS_REGION = 'us-east-1'
SAGEMAKER_ENDPOINT_NAME = 'sagemaker-xgboost-2023-04-25-18-33-28-097-endpoint'

# Initialize the SageMaker runtime client
sagemaker_client = boto3.client('sagemaker-runtime', region_name=AWS_REGION, aws_access_key_id=AWS_ACCESS_KEY_ID, aws_secret_access_key=AWS_SECRET_ACCESS_KEY)

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    if request.method == 'POST':
        input_data = request.get_json()
        predictions = invoke_sagemaker_endpoint(input_data)
        return jsonify(predictions)
    else:
        input_data = '14.23,1.71,2.43,15.6,127,2.8,3.06,0.28,2.29,5.64,1.04,3.92,1065'
        predictions = invoke_sagemaker_endpoint(input_data)
        return jsonify(predictions)

def invoke_sagemaker_endpoint(input_data):
    # csv_StringIO = io.StringIO(input_data)
    # test_df = pd.read_csv(csv_StringIO, sep=",", header=None)
    # csv_file = io.StringIO()
    # test_df.to_csv(csv_file, sep=",", header=False, index=False)
    # payload = csv_file.getvalue()
    response = sagemaker_client.invoke_endpoint(
        EndpointName=SAGEMAKER_ENDPOINT_NAME,
        ContentType="text/csv",
        Body=input_data
    )
    predictions = re.split(",|\n",response['Body'].read().decode("utf-8"))
    predictions.pop()
    
    return predictions

if __name__ == '__main__':
    app.run(debug=True)
