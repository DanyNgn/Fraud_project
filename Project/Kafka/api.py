from typing import Union
from fastapi import FastAPI
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

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import( OneHotEncoder, StandardScaler, LabelEncoder )
import joblib
import os

import Fraud_predict

app = FastAPI()




@app.get("/")
def start_prediction():
    print(type(Fraud_predict.predict.main))
    return {"message": 0}

try:
    start_prediction()
except:
    print("Hello")