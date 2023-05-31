import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, k_means
import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix,precision_score,recall_score,f1_score
import warnings
warnings.filterwarnings('ignore')
import joblib
import sys
from mlxtend.feature_selection import SequentialFeatureSelector
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing, svm
from sklearn.model_selection import cross_validate as cross_validation
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
import pickle
import lime
import lime.lime_tabular
from pydantic import BaseModel
import json
from fastapi import FastAPI

class Data(BaseModel):
    Age                        :      int
    DistanceFromHome           :     int
    EnvironmentSatisfaction    :     int
    Gender                     :     int
    HourlyRate                 :     int
    JobInvolvement             :     int
    JobLevel                   :     int
    JobSatisfaction            :     int
    MonthlyIncome              :     int
    NumCompaniesWorked         :     int
    OverTime                   :     int
    PercentSalaryHike          :     int
    PerformanceRating          :     int
    RelationshipSatisfaction   :     int
    StockOptionLevel           :     int
    TotalWorkingYears          :     int
    TrainingTimesLastYear      :     int
    WorkLifeBalance            :     int
    YearsAtCompany             :     int
    YearsInCurrentRole         :     int
    YearsSinceLastPromotion    :     int
    YearsWithCurrManager       :     int
    AVG_POS                    :     float
    AVG_NEG                    :     float
    BusinessTravel_Non_Travel  :     int
    BusinessTravel_Travel_Frequently:int
    BusinessTravel_Travel_Rarely    :int
    MaritalStatus_Divorced     :     int
    MaritalStatus_Married      :     int
    MaritalStatus_Single       :     int
    JobRole_AESP               :     int
    JobRole_Corporate          :     int
    JobRole_Director           :     int
    JobRole_ESP                :     int
    JobRole_Manager            :     int
    JobRole_Sales              :     int
    
# Cluster                    :     int

# loading models 
with open('modelM.pkl', 'rb') as file: 
    model = pickle.load(file)

with open('model.pkl', 'rb') as file: 
    forest = pickle.load(file)
app = FastAPI()
# uvicorn main:app --reload

@app.post("/getresult")
async def create_item(data: Data):
    data = data.__dict__
    new_df = pd.DataFrame([data], columns=data.keys())

    if new_df['JobRole_AESP'][0] == 1:
        new_df['AVG_POS'][0] = 0.7752242894597106
        new_df['AVG_NEG'][0] = 0.4793618006709342
    elif new_df['JobRole_Corporate'][0] == 1:
        new_df['AVG_POS'][0] = 0.7599373072814598
        new_df['AVG_NEG'][0] = 0.43415875065227527
    elif new_df['JobRole_Director'][0] == 1:
        new_df['AVG_POS'][0] = 0.8168275354546495
        new_df['AVG_NEG'][0] = 0.48619210150354775
    elif new_df['JobRole_ESP'][0] == 1:
        new_df['AVG_POS'][0] = 0.7752242894597106
        new_df['AVG_NEG'][0] = 0.4793618006709342
    elif new_df['JobRole_Manager'][0] == 1:
        new_df['AVG_POS'][0] = 0.7930518632056192
        new_df['AVG_NEG'][0] = 0.45369301825472763
    else:
        new_df['AVG_POS'][0] = 0.8514087075560257
        new_df['AVG_NEG'][0] = 0.5108685644568297

    predictions = model.predict(new_df.values)
    df_prediction = pd. DataFrame(predictions, columns=['Cluster'])
    new_df['Cluster'] = df_prediction['Cluster']
    
    forest_prediction = forest.predict(new_df)
    forest_prediction = list(forest_prediction)
    result = int(forest_prediction[0])

    print('result -> ', result)

    explainer = lime.lime_tabular.LimeTabularExplainer(new_df.values,feature_names = new_df.columns,class_names = ['Stayed','Left'],kernel_width = 5)
    predict_rf = lambda x: forest.predict_proba(x).astype(float)
    chosen_instance = new_df.loc[[0]].values[0]
    exp = explainer.explain_instance(chosen_instance,predict_rf,num_features = 10)

    factors = []
    for factor in exp.as_list():
        factors.append(factor[0].split(" ")[0])
    
    print('exp list -> ', factors)

    value = {
        "attrition":result, 
        "factors": factors, 
        "probabilities": {
            "staying_percentage": exp.predict_proba[0],
            "leaving_percentage": exp.predict_proba[1]
        }
    }

    print('value -> ', value)

    return value

@app.get("/hello/")
async def root():
    return {"message": "Hello World"}