import streamlit as st
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import warnings
warnings.filterwarnings("ignore")

st.title('Obesity Classifier AI')
#classifier = joblib.load('data/model_obesity.xgb')

best_model = XGBClassifier(learning_rate=.1, n_estimators=200, max_depth=7, min_chil_weight=1, gamma=0, subsample=.8, colsample_bytree=.6)
best_model.load_model('data/model_obesity_alex.model')
# steps = [("scaler", StandardScaler()),
#     ("xgb", model)]
# best_model = Pipeline(steps)


def user_input_features():

    Gender = st.sidebar.selectbox('Gender: (Male: 1, Female: 0)',(0,1))
    Age = st.sidebar.slider(min_value=1, label="Age (0-100)")
    family_history_with_overweight = st.sidebar.selectbox('Has a family member suffered or suffers from being overweight? (Yes: 1, No: 0)',(0,1))
    FAVC = st.sidebar.selectbox('Do you eat high caloric food frequently? (Yes: 1, No: 0)',(0,1))
    FCVC = st.sidebar.selectbox('Do you usually eat vegetables in your meals? (Never: 0, Sometimes: 1, Always: 2)',(0,1,2))
    NCP = st.sidebar.selectbox('How many main meals do you have daily?',(1,2,3,4))
    CAEC = st.sidebar.selectbox('Do you eat any food between meals? (No: 0, Sometimes: 1, Frequently: 2, Always: 3)',(0,1,2,3))
    SMOKE = st.sidebar.selectbox('Do you smoke? (Yes: 1, No: 0)',(0,1))
    CH2O = st.sidebar.selectbox('How much water do you drink daily? (less than a liter: 1, between 1-2 liters: 2, more than 2 liters: 3)',(1,2,3))
    SCC = st.sidebar.selectbox('Do you monitor the calories you eat daily? (Yes: 1, No: 0)',(0,1))
    FAF = st.sidebar.selectbox('How often do you have physical activity? ("None: 0, 1-2 days: 1, 2-4 days: 2, 4-5 days: 3)',(0,1,2,3))
    TUE = st.sidebar.selectbox('How much time do you use technological devices such as cell phone, videogames, television, computer and others? (0-2 hours: 0, 3-5 hours: 1, more than 5 hours: 2)',(0,1,2))
    CALC = st.sidebar.selectbox('How often do you drink alcohol? (Never: 0, Sometimes: 1, Frequently: 2, Always: 3)',(0,1,2,3))
    MTRANS = st.sidebar.selectbox('Which transportation do you usually use? (car: 0, bike: 1, motorbike: 2, public transport: 3, walking: 4, )',(0,1,2,3,4))

    data = {'Gender':[Gender], 
            'Age':[np.log(Age)], 
            'family_history_with_overweight':[family_history_with_overweight], 
            'FAVC':[FAVC], 
            'FCVC':[FCVC], 
            'NCP':[NCP], 
            'CAEC':[CAEC], 
            'SMOKE':[SMOKE], 
            'CH2O':[CH2O], 
            'SCC':[SCC], 
            'FAF':[FAF],
            'TUE':[TUE],
            'CALC':[CALC],
            'MTRANS':[MTRANS],
            }


    features = pd.DataFrame(data)

    return features

X_train = pd.read_csv('data/cleaned_train_obesity', index_col=0)
X_test = pd.read_csv('data/cleaned_test_obesity',index_col=0)
y_test = pd.read_csv('data/y_test_obesity',index_col=0)

scaler = StandardScaler().fit(X_train)
st.write(best_model.score(scaler.transform(X_test),y_test))
input_df = scaler.transform(user_input_features())
# xgb.DMatrix(input_df.values, feature_names=input_df.columns)


def prediction(): 
    # prediction = best_model.predict(xgb.DMatrix(input_df.values, feature_names=input_df.columns))
    prediction = best_model.predict(input_df)
    if prediction == 0:
        st.success("Underweight")
    if prediction == 1:
        st.success("Normal")
    if prediction == 2:
        st.success("Overweight I")
    if prediction == 3:
        st.success("Overweight II")
    if prediction == 4:
        st.success("Obese I")
    if prediction == 5:
        st.success("Obese II")
    if prediction == 6:
        st.success("Obese III")

st.button("Predict", on_click=prediction)