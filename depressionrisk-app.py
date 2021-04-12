import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier

st.write("""
# Depression Risk Assessment
This app predicts **if you are at a greater risk for depression**, based on your lifestyle and physical attributes.
The model is built using the “National Health and Nutrition Examination Survey (NHANES), 2005-2006” survey data. 
The NHANES survey design is a stratified, multistage probability sample of the civilian noninstitutionalized United States population.
""")
 
st.sidebar.header('Tell us about yourself')

st.sidebar.markdown("""
[Sample CSV input file](https://github.com/jaindx/UB-CSE587-Depression-Risk-Assessment/blob/master/Sample_Data.csv)
""")

# Collects user input features into dataframe
uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
else:
    def user_input_features():
        age = st.sidebar.slider('Age', 0,120,27)
        gender = st.sidebar.selectbox('Gender',('Male','Female'))
        height_inches = st.sidebar.slider('Height (inches)', 40,100,60)
        weight_pounds = st.sidebar.slider('Weight (lbs)', 60,700,150)
        num_drinks_days = st.sidebar.slider('No. of days you had a drink in the past 1 year', 0,365,20)
        num_ppl_family = st.sidebar.slider('No. of people in the family', 1,10,2)
        data = {'AGE_ADJUDICATED': age,
                'BMI': (weight_pounds/(height_inches*height_inches))*703,
                'DAYS_DRINK_12MONTHS': num_drinks_days,
                'NUMBER_PEOPLE_FAMILY': num_ppl_family,
                'GENDER': 1 if gender == 'Male' else 2}
        features = pd.DataFrame(data, index=[0])
        return features
    input_df = user_input_features()
    
data_raw = pd.read_csv('Final_DF.csv')
final_df = data_raw.drop(columns=['RISKY_GROUP'])
df = pd.concat([input_df,final_df],axis=0)

df = df[:1] # Selects only the first row (the user input data)

# Displays the user input features
st.subheader('Your data')

if uploaded_file is not None:
    st.write(df)
else:
    st.write('Awaiting CSV file to be uploaded. Currently using the user input data (shown below).')
    st.write(df)
    
# Reads in saved classification model
load_clf = pickle.load(open('depressionrisk_clf.pkl', 'rb'))

# Apply model to make predictions
prediction = load_clf.predict(df)
prediction_proba = load_clf.predict_proba(df)

st.subheader('Prediction')
st.write(prediction)

if prediction == 0:
    st.write('**You are at lower risk for depression..!!**')
else:
    st.write('**You are at higher risk for depression..!!**')

st.subheader('Prediction Probability')
st.write(prediction_proba)