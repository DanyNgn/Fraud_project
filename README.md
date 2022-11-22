Automatic Fraud Detection Project
==================================

Fraud is a huge issue among financial institutions. In the EU in 2019, the European Central Bank estimated that fraudulent credit card transactions amounted more €1 billion.
AI can really help solve this issue by detecting fraudulent payments in a very precise manner. This usecase is actually now one of the most famous one among Data Scientists.
However, eventhough we managed to build powerful algorithms, the hard thing is now to use them in production. This means predict fraudulent payment in real-time and respond appropriately. 

Project goals
==============
The goal as a team of Machine Learning Engineers is to:

- Create an fraudulent payment detector using any AI library that you know of.

- Create an infrastructure that ingest real-time payment

- Automatically classify each payment and send back this prediction in real-time to a notification center 

Données sensibles
====================
Information of python.config file is sensitive and need to be completed with your own API key downloadable from your Confluent account
Open python.config file and replace {{ CLUSTER_API_KEY }} and {{ CLUSTER_API_SECRET }} by your public and secret API keys


How to Install and Run the Project
====================================

1) Clone the repository on a folder:
 
<pre><code>git clone https://github.com/DanyNgn/Fraud_project.git</code></pre>

2) Open your IDE and go to "Project" folder

3) Run the follwing command to start the API Fraud_predict.py:

<pre><code>uvicorn Fraud_predict:app --reload</code></pre>

4) Subscribe to Confluent: https://confluent.cloud/

5) Create API keys in your Confluent account

6) On a second terminal, run the producer with this following command to produce transaction datas from the Jedha API:

<pre><code>python .\producer_fraudtest.py</code></pre>

7) On a third terminal, run the consumer to use datas from topic:

<pre><code>python .\consumer_fraudtest.py</code></pre>

8) Create a bucket at Amazon and save the API keys from your account

9) Open Confluent and go to Data Integration -> Connectors and choose "S3 Sink connector"

10) Choose your Fraud topic previously created in producer_fraudtest.py and connect your S3 bucket with your API keys

11) We can see that datas are sent to S3 bucket during the consuming in real time

Extra:

12) Sign up to Zapier Application and connect it to S3 bucket to see notification sent to an email if a fraud is detected

How the architecture looks like:
![image](https://user-images.githubusercontent.com/6365217/203176444-2268d00e-61ad-4725-a9ee-86cc56fe72ca.png)




Credit:
===========================================
Team project: Edouard, Linda, Dany, Samba
