
# Code inspired from Confluent Cloud official examples library
# https://github.com/confluentinc/examples/blob/7.1.1-post/clients/cloud/python/producer.py

#from confluent_kafka import Producer
import json
import logging
import os
from confluent_kafka import Producer
import time
import requests
import ccloud_lib
from datetime import datetime
import pandas as pd
import Model_fraud_detection.Fraud_predict

URL = "http://127.0.0.1:8000/file"
urlApi = "https://real-time-payments-api.herokuapp.com/current-transactions"
path = os.path.dirname(__file__)
# # Initialize configurations from "python.config" file
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

        response = requests.request("GET", urlApi)
        response_text=json.loads(response.text)
        response_json=json.loads(response_text)     

        result = pd.DataFrame(data=response_json["data"], columns=response_json["columns"]) 

        result=result.rename(columns = {'current_time':'trans_date_trans_time'})
        result['trans_date_trans_time'] = pd.to_datetime(result['trans_date_trans_time'], infer_datetime_format=True, unit="ms")
        result.insert(0, 'trans_date_trans_time', result.pop('trans_date_trans_time'))

        result.to_csv("fraudTestAPI.csv")

        #time.sleep(25)
        Model_fraud_detection.Fraud_predict
        resp = requests.request("GET", URL)
        print(resp.text)

        producer.produce(
                TOPIC,     
                key=record_key,
                value=resp.text,
                on_delivery=acked
        )
        logging.info("Wait 25 secondes to get the next request")
        time.sleep(25)

except KeyboardInterrupt:
    pass
finally:
    producer.flush()