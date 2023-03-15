import streamlit as st
import pandas as pd
import numpy as np
# import joblib
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

    Gender = st.select_slider('Gender',\
                              options=("Male", "Female"), key=0)
    Age = st.slider(min_value=1, label="Age (0-100)", key=1)
    family_history_with_overweight = st.select_slider('Has a family member suffered or suffers from being overweight?',\
                                                      options=('No', 'Yes'), key=2)
    FAVC = st.select_slider('Do you eat high caloric food frequently?',\
                            options=('No', 'Yes'), key=3)
    FCVC = st.select_slider('Do you usually eat vegetables in your meals?',\
                            options=("Never", "Sometimes", "Always"), key=4)
    NCP = st.select_slider('How many main meals do you have daily?',\
                           options=('1','2','3','4'), key=5)
    CAEC = st.select_slider('Do you eat any food between meals?',\
                            options=('No', 'Sometimes', 'Frequently', 'Always'), key=6)
    SMOKE = st.select_slider('Do you smoke?',\
                             options=('No','Yes'), key=7)
    CH2O = st.select_slider('How much water do you drink daily?',\
                            options=('less than a liter', 'between 1-2 liters', 'more than 2 liters'), key=8)
    SCC = st.select_slider('Do you monitor the calories you eat daily?',\
                            options=('No', 'Yes'), key=9)
    FAF = st.select_slider('How often do you have physical activity?',\
                            options=("None", "1-2 days", "2-4 days", "4-5 days"), key=10)
    TUE = st.select_slider('How much time do you use technological devices such as cell phone, videogames, television, computer and others?',\
                           options=('0-2 hours', '3-5 hours', 'more than 5 hours'), key=11)
    CALC = st.select_slider('How often do you drink alcohol?',\
                        options=('Never', 'Sometimes', 'Frequently', 'Always'), key=12)
    MTRANS = st.select_slider('Which transportation do you usually use?',\
                              options=('car', 'bike', 'motorbike', 'public transport', 'walking'), key=13)

    data = {
            'Gender':[1 if 'Male' else 0], 
            'Age':[np.log(Age)],
            'family_history_with_overweight':[1 if 'Yes' else 0], 
            'FAVC':[1 if 'Yes' else 0], 
            'FCVC':[0 if FCVC=='Never' else 1 if FCVC=="Sometimes" else 2], 
            'NCP':[int(NCP)], 
            'CAEC':[0 if CAEC=='No' else 1 if CAEC=='Sometimes' else 2 if CAEC=='Frequently' else 3], 
            'SMOKE':[1 if 'Yes' else 0], 
            'CH2O':[1 if CH2O=='less than a liter' else 2 if 'between 1-2 liters: 2' else 3 ], 
            'SCC':[1 if 'Yes' else 0], 
            'FAF':[0 if FAF=='None' else 1 if FAF=='1-2 days' else 2 if FAF=='2-4 days' else 3],
            'TUE':[0 if TUE=='0-2 hours' else 1 if TUE=='3-5 hours' else 2 if TUE=='more than 5 hours' else 3],
            'CALC':[0 if CALC=='Never' else 1 if CALC=='Sometimes' else 2 if CALC=='Frequently' else 3],
            'MTRANS':[0 if MTRANS=='car' else 1 if MTRANS=='bike' else 2 if MTRANS=='motorbike' else 3 \
                        if MTRANS=='public transport' else 4],
            }


    features = pd.DataFrame(data)

    return features



X_train = pd.read_csv('data/cleaned_train_obesity', index_col=0)
X_test = pd.read_csv('data/cleaned_test_obesity',index_col=0)
y_test = pd.read_csv('data/y_test_obesity',index_col=0)

scaler = StandardScaler().fit(X_train)
# st.write(best_model.score(scaler.transform(X_test),y_test))
with st.expander("Parameters", expanded=True):
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

with st.container():
    st.button("Predict", on_click=prediction)