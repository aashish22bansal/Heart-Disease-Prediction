import streamlit as st
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler 
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go






#the model
pickle_in= open("model.pk1","rb")
classifier=pickle.load(pickle_in)

st.set_page_config(layout="wide")

#first section
st.markdown(""" # Coronary Heart Disease (CHD) Prediction """, unsafe_allow_html=True) 
#objective 
st.divider()
st.subheader('Goal:')
st.write('The goal of this project is to predict whether a candidate will suffer from heart disease and help people identify high-risk candidates and prevent adverse selection.')  
st.divider()

#predict the label (i.e. 0, 1)
def prediction_label(x):
    prediction_label=classifier.predict(x)
    return prediction_label
#predict the probability (%)   
def prediction_proba(x):
    prediction_proba=classifier.predict_proba(x)[:,1]
    return prediction_proba

    
def main():
    scaler = MinMaxScaler()
    
    #markdown
    st.markdown(
        """
        # Please answer the following medical questions :male-doctor::
        """ 
        ,unsafe_allow_html=True) 

    
    #male
    male = 0
    gender_class = ['Male', 'Female']
    gender = st.radio('Gender',gender_class)
    if gender == 'Male':
        male = 1 
    else:
        male = 0
    
    #age 
    age = 0
    age = st.slider('Age',min_value=0, max_value=100, value=0, step=1)

    #cigsPerDay
    cigsPerDay = st.slider('Daily Cigarette Consumption',min_value=0, max_value=40, value=0, step=None)
    
    #BPMeds
    BPMeds = 0
    BPMeds_answers=['Yes', 'No']
    BPMeds_response = st.radio('Have you taken Blood Pressure Medicine before?',BPMeds_answers)
    if BPMeds_response == 'Yes':
        BPMeds = 1 
    else:
        BPMeds = 0
        
    #prevalentStroke
    prevalentStroke = 0
    prevalentStroke_answers=['Yes', 'No']
    prevalentStroke_response = st.radio('Have you experienced stroke in the past?',prevalentStroke_answers)
    if  prevalentStroke_response == 'Yes':
        prevalentStroke = 1 
    else:
        prevalentStroke = 0
    
    
    #prevalentHyp
    prevalentHyp = 0
    prevalentHyp_answers=['Yes', 'No']
    prevalentHyp_response = st.radio('Do you have any hypertensive heart disease, such as high blood pressure?',prevalentHyp_answers)
    if prevalentHyp_response == 'Yes':
        prevalentHyp = 1 
    else:
        prevalentHyp = 0
    

    
    #diabetes
    diabetes=0
    diabetes_answers=['Yes', 'No']
    diabetes_response = st.radio('Do you have diabetes?',diabetes_answers)
    if diabetes_response == 'Yes':
        diabetes = 1 
    else:
        diabetes = 0
    
    #totChol
    totChol = st.slider('What is your total cholesterol level? \
                        (according to  Johns Hopkins Medicine, \
                            ranges for total cholesterol in adults: \
                            Normal: Less than 200 mg/dL. \
                            Borderline high: 200 to 239 mg/dL.\
                            High: At or above 240 mg/dL)',min_value=50, max_value=400, value=0, step=10)
    

    #sysBP
    sysBP = st.slider('What is your systolic blood pressure level? \
                        (according to American Heart Association (AHA), \
                        Normal: less than 120 mmHg, Elevated: 120-129 mmHg, \
                        Stage 1 hypertension: 130-139 mmHg, \
                        Stage 2 hypertension: 140 mmHg or higher)',min_value=50, max_value=200, value=0, step=10)
    


    #diaBP
    diaBP = 0
    diaBP = st.slider('what is the range for diastolic blood pressure? \
                        (according to American Heart Association (AHA), \
                        Normal: less than 80 mmHg, Elevated: 80-89 mmHg,\
                        Stage 1 hypertension: 90-99 mmHg, \
                        Stage 2 hypertension: 100 mmHg or higher)',min_value=50, max_value=140, value=0, step=10)
    

    #BMI
    BMI = 0
    BMI = st.slider('What is your Body Mass Index(BMI)? \
                    (according to World Health Organizatio - WHO,\
                        Underweight: less than 18.5, Normal weight: 18.5-24.9, \
                        Overweight: 25-29.9, Obesity class I: 30-34.9, Obesity class II: 35-39.9,\
                        Obesity class III: 40 or higher)',min_value=10, max_value=50, value=0, step=1)


    #heartRate
    heartRate = 0 
    heartRate = st.slider('What is your heart rate per minute (bpm)? \
                            (according to Bristish Heart Foundation, \
                            a normal resting heart rate should be between 60 to 100 beats per minute)'
                            ,min_value=40, max_value=150, value=0, step=None)
    
        
    
    #input one dimensional array
    X_test_continuous= {'age' : age, 'cigsPerDay': cigsPerDay,'totChol':totChol,'sysBP': sysBP, 'diaBP': diaBP, 'BMI': BMI,'heartRate': heartRate}
    df_test = pd.DataFrame(X_test_continuous, index=['test'])
    
    #scale all
    #fit 
    df = pd.read_csv('data/framingham.csv')
    scaler = scaler.fit(df[['age', 'cigsPerDay', 'totChol', 'sysBP', 'diaBP', 'BMI', 'heartRate']])
    #scale
    scaled_value = scaler.transform(df_test)
    
    # define variable
    list_= list()
    label=""
    proba=""
    
    # get scaled values
    age_sc, cigsPerDay_sc, totChol_sc, sysBP_sc, diaBP_sc, BMI_sc, heartRate_sc = scaled_value.flatten()
    age, cigsPerDay, totChol, sysBP, diaBP, BMI, heartRate =  age_sc, cigsPerDay_sc, totChol_sc, sysBP_sc, diaBP_sc, BMI_sc, heartRate_sc
    list_= [male, age, cigsPerDay, BPMeds, prevalentStroke, prevalentHyp, diabetes, totChol, sysBP, diaBP, BMI, heartRate]
     
    #page layout (the table and the radar chart)
    col1, col2 = st.columns([0.65,1])
    
    #variable
    list_conti = np.array([age, cigsPerDay, totChol, sysBP, diaBP, BMI, heartRate])
    
    #if user click "predict"
    st.subheader('Result: ')
    st.write('Please click \'Predict\' button after you finish inputing your information')
    if st.button("Predict"):
        label = prediction_label(np.array(list_).reshape(1,-1))
        proba = np.round(prediction_proba(np.array(list_).reshape(1,-1)) * 100, 2)
        #st.success('The result is {}'.format(label))
        st.warning(f'Your probability of having CHD in the next 10 years is {proba}%')
     
    #last section
    st.divider()
    

if __name__ == '__main__':
    main()
