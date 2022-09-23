# Keep in memory test 
# https://discuss.streamlit.io/t/how-to-add-records-to-a-dataframe-using-python-and-streamlit/19164/6
#¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨#
#            import            #
#______________________________#

# Import of librairies

from asyncio.log import logger
from geopy.distance import geodesic
import pandas as pd
import numpy as np # Not always necessary
import matplotlib.pyplot as plt # Not always necessary
import seaborn as sns # Not always necessary
from sklearn.model_selection import train_test_split # libraie énorme donc j'importe que les fonctions utiles
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import plot_confusion_matrix
from sklearn.impute import SimpleImputer
from sklearn.tree import plot_tree
from fastapi import FastAPI, File, UploadFile
import requests

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import( OneHotEncoder, StandardScaler, LabelEncoder )
import joblib
import os

#¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨#
#         definitions          #
#______________________________#

app = FastAPI()

def preprocessorPipeline(X):
    numeric_features = X.select_dtypes([np.number]).columns 
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')), 
        ('scaler', StandardScaler())
    ])
    
    categorical_features = X.select_dtypes("object").columns
    categorical_transformer = Pipeline(
        steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(drop='first'))
        ])
        
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    preprocessor.fit_transform(X) # Preprocessing influenceur

    return preprocessor

def distance(row):
    try: 
        return (geodesic(row['geometry'], row['merch_geometry']).km) 
    except:
        return np.nan


def processusDataset(df_fraud_detection):
    df_fraud_detection['dob']= pd.to_datetime(df_fraud_detection['dob'])
    df_fraud_detection['age'] = (df_fraud_detection['trans_date_trans_time'] - df_fraud_detection['dob']).astype('<m8[Y]')
    df_fraud_detection['age'] = df_fraud_detection['age'].astype(int)
    df_fraud_detection['street_name'] = [ x.split(" ") for x in df_fraud_detection['street'] ]
    df_fraud_detection['street_name'] = [ ' '.join(x[1:3]) for x in df_fraud_detection['street_name'] ]
    df_fraud_detection["hour"] = df_fraud_detection['trans_date_trans_time'].dt.hour.astype(int)
    df_fraud_detection['dayofweek'] = df_fraud_detection['trans_date_trans_time'].dt.dayofweek.astype(int)
    df_fraud_detection['month'] = df_fraud_detection['trans_date_trans_time'].dt.month.astype(int)
    df_fraud_detection['year'] = df_fraud_detection['trans_date_trans_time'].dt.year.astype(int)
    df_fraud_detection['dayofyear'] = df_fraud_detection['trans_date_trans_time'].dt.dayofyear.astype(int)
    df_fraud_detection['dayofmonth'] = df_fraud_detection['trans_date_trans_time'].dt.day.astype(int)
    df_fraud_detection['weekofyear'] = df_fraud_detection['trans_date_trans_time'].dt.weekofyear.astype(int)
    df_fraud_detection['card_issuer_MMI'] = [ f'mmi{str(x)[0:1]}' for x in df_fraud_detection['cc_num']]
    df_fraud_detection['card_issuer_Bank'] = [ int(str(x)[1:6]) for x in df_fraud_detection['cc_num']]
    df_fraud_detection["merchant_"] = [ x.replace('fraud_', '') for x in df_fraud_detection["merchant"] ]
    df_fraud_detection['geometry'] = list(zip(df_fraud_detection['lat'], df_fraud_detection['long']))
    df_fraud_detection['merch_geometry'] = list(zip(df_fraud_detection['merch_lat'], df_fraud_detection['merch_long']))
    df_fraud_detection['distance'] = df_fraud_detection.apply(lambda row: distance(row), axis = 1 )
    df_fraud_detection['zip'] = df_fraud_detection['zip'].astype(str)
    df_fraud_detection['distance'] = (df_fraud_detection['distance']).astype(float,2)
    df_fraud_detection['amt'] = df_fraud_detection['amt'].astype(float,2)
    df_fraud_detection = df_fraud_detection.drop(columns=['trans_date_trans_time','merchant','first','last','street','city','state','zip','job','dob','trans_num','merchant_','geometry','merch_geometry','cc_num','street_name','year'])
    return df_fraud_detection


#¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨#
#        Execution Code        #
#______________________________#

@app.get("/file")
def main():

    path = os.path.dirname(__file__)
    ## This dataset is a formated dataset EDA for training model 
    dataset_original = pd.read_csv(path+'/formatedDataset.csv')
    ## This is the new dataset we want to guess the result 
    dataset_from_api = pd.read_csv(path+"/../fraudTestAPI.csv", index_col=0, parse_dates=[1])# <---- Consumer envoie le dataset ici. 
    ## We apply the same transformations than the first dataset 
    dataset_from_api = processusDataset(dataset_from_api)


    # The target colum name 
    target_name = "is_fraud"
    # create the preprocessor form the originals columns 
    preprocessor = preprocessorPipeline(dataset_original.drop(columns= [target_name]))
    # Split the dataset_from_api before guess the target with the model 
    # Y = dataset_from_api[:][target_name] # The Y is just a verif here
    X = dataset_from_api.drop(columns= [target_name])
    X_numpy = preprocessor.transform(X)

    ## This is the model train on this dataset (This is a LogisticRegression)
    model = joblib.load(path+"/model.joblib")
    ## Prediction
    Y = model.predict(X_numpy) # Prédictions on test set 

    finalDataset = X
    finalDataset['is_fraud'] = Y

    print(finalDataset.to_dict())
    return(finalDataset.to_dict())
try:
    main()
except:
    logger.error("An error happens with api server")