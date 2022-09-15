from flask import Flask,jsonify,request
import pandas as pd
import numpy as np
import math
import pickle
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import log_loss
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score

import flask

app = Flask(__name__)

#############################################################################
def get_categorical_features(dataset):
    """Returns a list of all the categorical features present in the dataset"""
    categorical_features = []
    for feature in dataset.columns:
        if dataset[feature].dtypes == 'O':
            categorical_features.append(feature)
    return categorical_features

def identify_features_na(data):
    """This function takes a dataset as input and return a list of 
    columns for which contain a null value"""
    features_with_na = []
    for column in data.columns:
        if data[column].isnull().sum() > 1:
            features_with_na.append(column)
    return features_with_na   

def replace_categorical_feature_na(data, categorical_feature_nan):
    dataset = data.copy()
    dataset[categorical_feature_nan] = dataset[categorical_feature_nan].fillna('Missing')
    return dataset

def get_numerical_features(dataset):
    """Returns a list of all the numerical features present in the dataset"""
    numerical_features = []
    for feature in dataset.columns:
        if dataset[feature].dtypes != 'O':
            numerical_features.append(feature)
    return numerical_features

def encode_categorical_data(data, encoders_dict):
    encoded_data = []
    flag = 0
    for column, encoder in encoders_dict.items():
        if flag == 0:
            encoded_data = encoder.transform(data[column].values.reshape(-1,1)).toarray()
            flag = 1
        else:
            encoded_data = np.hstack((encoded_data,encoder.transform(data[column].values.reshape(-1,1)).toarray()))
    return encoded_data

def introduce_feature_interactions(data, columns):
    """This function randomly picks three features from the list of features provided to it
    and performs various feature interactions on them to generate a set of new features"""
    new_features = []
    
    for features in columns:
        
        feature_1 = data[:,features[0]]
        feature_2 = data[:,features[1]]
        feature_3 = data[:,features[2]]

        new_features.append(feature_1 * feature_2)
        new_features.append(feature_1 + feature_2)
        new_features.append(feature_1 - feature_2)
        new_features.append(feature_1 * feature_2 + feature_3)
        new_features.append(feature_1 + feature_2 - feature_3)
        new_features.append(feature_1 - feature_2 * feature_3)
        
        
    new_features = np.array(new_features)
    return new_features.T

def correlation(dataset, threshold):
    column = set()
    cd = dataset.corr()
    for i in range(len(cd.columns)):
        for j in range(i):
            if abs(cd.iloc[i,j]) >threshold:
                col_name = cd.columns[i]
                column.add(col_name)
    return column
#############################################################################

@app.route('/')
def hello_world():
    return 'Hello World'

@app.route('/predict')
def final_function_1():
    data = pd.read_csv('homesite-quote-conversion/test.csv').sample(100)
    #handle missing values
    features_categorical = get_categorical_features(data.loc[:,identify_features_na(data)])
    data = replace_categorical_feature_na(data, features_categorical)
    features_numerical = get_numerical_features(data.loc[:,identify_features_na(data)])
    data[['PersonalField84_nan', 'PropertyField29_nan']] = np.where(data[features_numerical].isnull(),1,0)
    data[features_numerical] = data[features_numerical].fillna(100)
    #handle date
    data['Original_Quote_Date'] = pd.to_datetime(data['Original_Quote_Date'], format='%Y-%m-%d')
    data['Original_Quote_Day'] = data['Original_Quote_Date'].apply(lambda x: x.day)
    data['Original_Quote_Month'] = data['Original_Quote_Date'].apply(lambda x: x.month)
    data['Original_Quote_Quater'] = data['Original_Quote_Date'].apply(lambda x: math.ceil(x.month/3))
    data['Original_Quote_Year'] = data['Original_Quote_Date'].apply(lambda x: x.year)
    #dropping date, quote_number
    quote_number = data['QuoteNumber']
    data.drop(columns = ['QuoteNumber','Original_Quote_Date'], axis = 1, inplace = True)
    #onehotencodecategoricaldata
    encoders_dict = pickle.load(open('pickle_files/encoders_dict.pkl','rb'))
    encoded_categorical_data = encode_categorical_data(data,encoders_dict)
    data.drop(labels = list(encoders_dict.keys()), axis = 1 , inplace = True)
    data = np.hstack((data.to_numpy(),encoded_categorical_data))
    #removing constant,scaling data
    vr = pickle.load(open('pickle_files/constant_features.pkl','rb'))
    data = data[:,vr.get_support()]
    scaler = pickle.load(open('pickle_files/feature_scaling.pkl','rb'))
    data = scaler.transform(data)
    #feature interactions and removing correlated features
    columns = pickle.load(open('pickle_files/feature_interactions.pkl','rb'))
    new_features =  introduce_feature_interactions(data,columns)
    data = np.hstack((data,new_features))
    dataset = pd.DataFrame(data)
    final_features = pickle.load(open('pickle_files/final_features.pkl','rb'))
    data = data[:,final_features]
    #loading model
    model = pickle.load(open('pickle_files/best_model_cc.pkl','rb'))
    y_pred = np.argmax(model.predict_proba(data), axis = -1)
    predictions = pd.DataFrame(data = zip(quote_number,y_pred), columns = ['QuoteNumber','QuoteConversion_Flag'])
    return predictions.to_json(orient='split' , index = False)

@app.route('/evaluate_metric')
def final_function_2():
    data = pd.read_csv('homesite-quote-conversion/train.csv').sample(100)
    labels = data['QuoteConversion_Flag']
    data.drop(columns = ['QuoteConversion_Flag'], axis = 1, inplace = True)
    #handle missing values
    features_categorical = get_categorical_features(data.loc[:,identify_features_na(data)])
    data = replace_categorical_feature_na(data, features_categorical)
    features_numerical = get_numerical_features(data.loc[:,identify_features_na(data)])
    data[['PersonalField84_nan', 'PropertyField29_nan']] = np.where(data[features_numerical].isnull(),1,0)
    data[features_numerical] = data[features_numerical].fillna(100)
    #handle date
    data['Original_Quote_Date'] = pd.to_datetime(data['Original_Quote_Date'], format='%Y-%m-%d')
    data['Original_Quote_Day'] = data['Original_Quote_Date'].apply(lambda x: x.day)
    data['Original_Quote_Month'] = data['Original_Quote_Date'].apply(lambda x: x.month)
    data['Original_Quote_Quater'] = data['Original_Quote_Date'].apply(lambda x: math.ceil(x.month/3))
    data['Original_Quote_Year'] = data['Original_Quote_Date'].apply(lambda x: x.year)
    #dropping date, quote_number
    quote_number = data['QuoteNumber']
    data.drop(columns = ['QuoteNumber','Original_Quote_Date'], axis = 1, inplace = True)
    #onehotencodecategoricaldata
    encoders_dict = pickle.load(open('pickle_files/encoders_dict.pkl','rb'))
    encoded_categorical_data = encode_categorical_data(data,encoders_dict)
    data.drop(labels = list(encoders_dict.keys()), axis = 1 , inplace = True)
    data = np.hstack((data.to_numpy(),encoded_categorical_data))
    #removing constant,scaling data
    vr = pickle.load(open('pickle_files/constant_features.pkl','rb'))
    data = data[:,vr.get_support()]
    scaler = pickle.load(open('pickle_files/feature_scaling.pkl','rb'))
    data = scaler.transform(data)
    #feature interactions and removing correlated features
    columns = pickle.load(open('pickle_files/feature_interactions.pkl','rb'))
    new_features =  introduce_feature_interactions(data,columns)
    data = np.hstack((data,new_features))
    dataset = pd.DataFrame(data)
    final_features = pickle.load(open('pickle_files/final_features.pkl','rb'))
    data = data[:,final_features]
    #loading model
    model = pickle.load(open('pickle_files/best_model_cc.pkl','rb'))
    y_pred_proba = model.predict_proba(data)
    #performance metrics
    auc = roc_auc_score(labels, y_pred_proba[:,1])
    ll = log_loss(labels,y_pred_proba)
    f1 = f1_score(labels, np.argmax(y_pred_proba, axis = -1))
    performance_metrics = {"AUC": auc, "LOG LOSS": ll, "F1_SCORE": f1}
    return pd.DataFrame(performance_metrics,  index=[0]).to_json(orient='split' , index = False)

if __name__ == '__main__':
    app.run(host = '0.0.0.0', port = 8080)
