import json
import logging
import os
from confluent_kafka import Producer
import time
import requests
import ccloud_lib
import pandas as pd
import Model_fraud_detection.Fraud_predict

# FastAPI fraud_predict
URL = "http://127.0.0.1:8000/file"
# API Jedha
urlApi = "https://real-time-payments-api.herokuapp.com/current-transactions"
path = os.path.dirname(__file__)

#Initialize configurations from "python.config" file. It contains configuration to connect to the cluster
CONF = ccloud_lib.read_ccloud_config(path+"/python.config")

TOPIC = "my_fraud_detection_topic"

#Create Topic
ccloud_lib.create_topic(CONF, TOPIC)

# # Create Producer instance
producer_conf = ccloud_lib.pop_schema_registry_params_from_config(CONF)
producer = Producer(producer_conf)
delivered_records = 0

# Callback called acked (triggered by poll() or flush())
# when a message has been successfully delivered or
# permanently failed delivery (after retries).
def acked(err, msg):
    global delivered_records
    # Delivery report handler called on successful or failed delivery of message
    if err is not None:
        print("Failed to deliver message: {}".format(err))
    else:
        delivered_records += 1
        print("Produced record to topic {} partition [{}] @ offset {}".format(msg.topic(), msg.partition(), msg.offset()))
try:
    
    while True:
        record_key="Fraud detection datas"

        # Récupération des informations de transactions de paiements de l'API fournit par jedha
        response = requests.request("GET", urlApi)
        response_text=json.loads(response.text)
        
        # Convert to Json
        response_json=json.loads(response_text)     

        # Convert to Dataframe
        result = pd.DataFrame(data=response_json["data"], columns=response_json["columns"]) 

        result=result.rename(columns = {'current_time':'trans_date_trans_time'})
        result['trans_date_trans_time'] = pd.to_datetime(result['trans_date_trans_time'], infer_datetime_format=True, unit="ms")
        
        # Insert date and hour at the first column and delete it from the last column
        result.insert(0, 'trans_date_trans_time', result.pop('trans_date_trans_time'))

        # Convert data to csv
        result.to_csv("fraudTestAPI.csv")

        # Call the predict API (Fraud_predict.py) with the previous fraudTestAPI.csv
        Model_fraud_detection.Fraud_predict

        # Get the prediction from the predict API with the fraud result at the last column
        resp = requests.request("GET", URL)
        print(resp.text)

        # This will actually send data to the created topic
        producer.produce(
                TOPIC,     
                key=record_key,
                value=resp.text,
                on_delivery=acked
        )

        # Jedha provide the API that send 5 transactions per minute so we got a request every 12 secondes
        logging.info("Wait 12 secondes to get the next request")
        time.sleep(12)

except KeyboardInterrupt:
    pass
finally:
    producer.flush() # Finish producing the latest event before stopping the whole script