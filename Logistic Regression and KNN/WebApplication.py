
# load python packages for web app
import pickle
import streamlit as st
from streamlit_option_menu import option_menu # pip install streamlit-option-menu

import numpy as np
import pandas as pd
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px

# @st.cache


# ***************************************************************************************
# **** load model, scaler function, model metrics created in individual problem sets ****
# ***************************************************************************************

# load pickle files for each of the problem set. 
# Pickle files contain logistic regression model, kNN model, scaler function, model metrics
pickle_in = open('rr_assignment3_p1_files', 'rb')
p1_files = pickle.load(pickle_in)
p1_lr_classifier = p1_files[0]
p1_knn_classifier = p1_files[1]
p1_scaler = p1_files[2]
p1_lr_model_metrics=p1_files[3]
p1_knn_model_metrics=p1_files[4]

pickle_in = open('rr_assignment3_p2_files', 'rb')
p2_files = pickle.load(pickle_in)
p2_lr_classifier = p2_files[0]
p2_knn_classifier = p2_files[1]
p2_scaler = p2_files[2]
p2_lr_model_metrics=p2_files[3]
p2_knn_model_metrics=p2_files[4]

pickle_in = open('rr_assignment3_p3_files', 'rb')
p3_files = pickle.load(pickle_in)
p3_lr_classifier = p3_files[0]
p3_knn_classifier = p3_files[1]
p3_scaler = p3_files[2]
p3_lr_model_metrics=p3_files[3]
p3_knn_model_metrics=p3_files[4]

# set page title
st.set_page_config(page_title='CIS 508 - Problem Set 3')

# page configuration
hide_st_style="""
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
</style>
"""
st.markdown(hide_st_style, unsafe_allow_html=True)

# add title to the page
st.title("CIS 508 - PROBLEM SET 3")

# ************************************************
# **** functions used in the web applications ****
# ************************************************

@st.cache

# function to select the model based on selected dataset and classifier
def model_selection(dataset_name,classifier_name):
    if dataset_name == 'System Administrator':
        if classifier_name == 'Logistic Regression':
            model = p1_lr_classifier
        elif classifier_name == 'kNN':
            model = p1_knn_classifier
    elif dataset_name == 'Flight Delay':
        if classifier_name == 'Logistic Regression':
            model = p2_lr_classifier
        elif classifier_name == 'kNN':
            model = p2_knn_classifier
    elif dataset_name == 'Loan Approval':
        if classifier_name == 'Logistic Regression':
            model = p3_lr_classifier
        elif classifier_name == 'kNN':
            model = p3_knn_classifier
    
    return model

# function to obtain the default probability threshold for the logistic and kNN models.
def model_params_default_value(dataset_name,classifier_name):
    default_params = dict()
    if dataset_name == 'System Administrator':
        if classifier_name == 'Logistic Regression':
            default_params['th']=0.46
        else:
            default_params['th']=0.4
            default_params['k']=9
    elif dataset_name == 'Flight Delay':
        if classifier_name == 'Logistic Regression':
            default_params['th']=0.51
        else:
            default_params['th']=0.4
            default_params['k']=19
    elif dataset_name == 'Loan Approval':
        if classifier_name == 'Logistic Regression':
            default_params['th']=0.5
        else:
            default_params['th']=0.3
            default_params['k']=7
    
    return default_params
    
# function to set model's probability threshold for logistic and kNN model
def model_params(dataset_name,classifier_name):
    params = dict()
    default_params = model_params_default_value(dataset_name,classifier_name)
    if classifier_name == 'Logistic Regression':
        th = st.sidebar.slider("Select Probability Threshold for the Model",0.0,1.0,value=default_params['th'],step=0.01)
        params['th']=th
    elif classifier_name == 'kNN':
        th = st.sidebar.slider("Select Probability Threshold for the Model",0.0,1.0,value=default_params['th'],step=0.1)
        params['th']=th
        k = st.sidebar.slider("Optimal value of k for the model",1,21,value=default_params['k'],step=2,disabled=True)
        params['k']=k
    
    return params

# function for creating a dashboard displaying the performance of a selected model
def plot_perfomance_metrics(model_metrics,model_parameters):
    
    accuracy = "{:.0%}".format(model_metrics[model_metrics['threshold']>=model_parameters['th']]['Accuracy'].values[0])
    precision = "{:.0%}".format(model_metrics[model_metrics['threshold']>=model_parameters['th']]['Precision'].values[0])
    recall = "{:.0%}".format(model_metrics[model_metrics['threshold']>=model_parameters['th']]['Recall'].values[0])
    F1 = "{:.0%}".format(model_metrics[model_metrics['threshold']>=model_parameters['th']]['F1'].values[0])
    fp_rate = "{:.0%}".format(model_metrics[model_metrics['threshold']>=model_parameters['th']]['FP_rate'].values[0])
    fn_rate = "{:.0%}".format(model_metrics[model_metrics['threshold']>=model_parameters['th']]['FN_rate'].values[0])
    
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    col1.metric(label="ACCURACY", value=accuracy)
    col2.metric(label="PRECISION", value=precision)
    col3.metric(label="RECALL", value=recall)
    col4.metric(label="F1 SCORE", value=F1)
    col5.metric(label="FP RATE", value=fp_rate)
    col6.metric(label="FN RATE", value=fn_rate)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=model_metrics['threshold'], y=model_metrics['Precision'],
                        mode='lines',
                        name='Precision'))
    fig.add_trace(go.Scatter(x=model_metrics['threshold'], y=model_metrics['Recall'],
                        mode='lines',
                        name='Recall'))
    fig.add_trace(go.Scatter(x=model_metrics['threshold'], y=model_metrics['F1'],
                        mode='lines', name='F1'))
    fig.add_trace(go.Scatter(x=model_metrics['threshold'], y=model_metrics['Accuracy'],
                        mode='lines', name='Accuracy'))
    fig.update_layout(legend_title_text = "Performance Measures")
    fig.update_xaxes(title_text="Threshold")
    

#     fig.show()
    st.plotly_chart(fig, use_container_width=True)

def model_performance_metrics(dataset_name,classifier_name,model_parameters):
    if dataset_name == 'System Administrator':
        if classifier_name == 'Logistic Regression':
            model_metrics = p1_lr_model_metrics
        elif classifier_name == 'kNN':
            model_metrics = p1_knn_model_metrics
    elif dataset_name == 'Flight Delay':
        if classifier_name == 'Logistic Regression':
            model_metrics = p2_lr_model_metrics
        elif classifier_name == 'kNN':
            model_metrics = p2_knn_model_metrics
    elif dataset_name == 'Loan Approval':
        if classifier_name == 'Logistic Regression':
            model_metrics = p3_lr_model_metrics
        elif classifier_name == 'kNN':
            model_metrics = p3_knn_model_metrics
    
    return model_metrics

# function for creating a user input template for the system administrator dataset
def get_data_system_administrator():
    # display selection of years of experience and training level in same row
    col1, col2 = st.columns(2)
    # select years of experience object
    years_of_experience = col1.number_input("Number of years of experience",
                              min_value=0.0,
                              max_value=20.0,
                              value=5.0,
                              step=0.1,
                             )
    # normalize user input using the scaler used during training of model 
    years_of_experience = p1_scaler.transform([[years_of_experience]])
#     st.write(years_of_experience[0][0])
    # select training level object
    training_level_4=0
    training_level_6=0
    training_level_8=0

    training_level = col2.selectbox("Training Level",("Level 4","Level 6","Level 8"))
    if training_level == 'Level 4':
        training_level_4=1
    elif training_level == 'Level 6':
        training_level_6=1
    elif training_level == 'Level 8':
        training_level_8=1

    return [[years_of_experience[0][0],training_level_4,training_level_6,training_level_8]]

# function for creating a user input template for the flight delay dataset
def get_data_flight_delay():
    
    # display selection of departure time and carrier in same row
    col1, col2 =st.columns(2)
    
    # select scheduled departure time
    scheduled_departure_time = col1.number_input("Scheduled Departure Time",
                              min_value=0.00,
                              max_value=23.99,
                              value=10.00,
                              step=0.01,
                             )
    # normalize user input using the scaler used during training of model 
    scheduled_departure_time = p2_scaler.transform([[scheduled_departure_time]])
    
    # select carrier object
    carrier_delta=0
    carrier_us=0
    carrier_envoy=0
    carrier_continental=0
    carrier_discovery=0
    carrier_other=0
    
    carrier = col2.selectbox("Carrier",("Delta","US","Envoy","Continental","Discovery","Other"))
    if carrier == 'Delta':
        carrier_delta=1
    elif carrier == 'US':
        carrier_us=1
    elif carrier == 'Envoy':
        carrier_envoy=1
    elif carrier == 'Continental':
        carrier_continental=1
    elif carrier == 'Discovery':
        carrier_discovery=1
    elif carrier == 'Other':
        carrier_other=1
    
    # display selection of departure time and carrier in same row
    col1, col2 =st.columns(2)
    
    # select destination
    dest_jfk=0
    dest_ewr=0
    dest_lga=0
    
    destination = col1.selectbox("Destination Airport",("John F. Kennedy Airport",
                                                             "Newark Liberty International Airport",
                                                             "LaGuardia Airport"))
    if destination == 'John F. Kennedy Airport':
        dest_jfk=1
    elif destination == 'Newark Liberty International Airport':
        dest_ewr=1
    elif destination == 'LaGuardia Airport':
        dest_lga=1
    
    # select origin
    origin_dca=0
    origin_iad=0
    origin_bwi=0
    
    origin = col2.selectbox("Origin Airport",("Ronald Reagan Washington National Airport",
                                                   "Dulles International Airport",
                                                   "Baltimore/Washington International Airport"))
    if origin == 'Ronald Reagan Washington National Airport':
        origin_dca=1
    elif origin == 'Dulles International Airport':
        origin_iad=1
    elif origin == 'Baltimore/Washington International Airport':
        origin_bwi=1
    
    # display selection of weather and day name in same row
    col1, col2 =st.columns(2)
    
    # select weather
    bad_weather=0
    
    weather = col1.selectbox("Weather",("Cloudy/Rainy","Clear Sky"))
    if weather == 'Cloudy/Rainy':
        bad_weather=1
    
    # select day_name
    Monday=0
    Tuesday=0
    Wednesday=0
    Thursday=0
    Friday=0
    Saturday=0
    Sunday=0
    
    day_name = col2.selectbox("WeekDay Name",("Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"))
    if day_name == 'Monday':
        Monday=1
    elif day_name == 'Tuesday':
        Tuesday=1
    elif day_name == 'Wednesday':
        Wednesday=1
    elif day_name == 'Thursday':
        Thursday=1
    elif day_name == 'Friday':
        Friday=1
    elif day_name == 'Saturday':
        Saturday=1
    elif day_name == 'Sunday':
        Sunday=1 
        
    return [[scheduled_departure_time[0][0],carrier_delta,carrier_us,carrier_envoy,carrier_continental,
             carrier_discovery,carrier_other,dest_jfk,dest_ewr,dest_lga,origin_dca,origin_iad,origin_bwi,
             bad_weather,Monday,Tuesday,Wednesday,Thursday,Friday,Saturday,Sunday]]

# function for creating a user input template for the loan approval dataset
def get_data_loan_approval():
    
    col1, col2 =st.columns(2)
    # select years employed object
    years_employed = col1.number_input("Years of Employment",
                              min_value=0.00,
                              max_value=30.00,
                              value=10.00,
                              step=0.01,
                             )
    
    emp_industrial=0
    emp_materials = 0
    emp_consumer_services = 0
    emp_healthcare = 0
    emp_financials = 0
    emp_utilities = 0
    emp_education = 0
    
    # select job sector object
    job_sector = col2.selectbox("Job Sector",("Industrial","Materials","Consumer Services","Healthcare","Financials",
                                            "Utilities","Education"))
    if job_sector == 'Industrial':
        emp_industrial=1
    elif job_sector == 'Materials':
        emp_materials=1
    elif job_sector == 'Consumer Services':
        emp_consumer_services=1
    elif job_sector == 'Healthcare':
        emp_healthcare=1
    elif job_sector == 'Financials':
        emp_financials=1
    elif job_sector == 'Utilities':
        emp_utilities=1
    elif job_sector == 'Education':
        emp_education=1
        
    col1, col2 =st.columns(2)
    
    # select employed yes/no object
    employed=0
    
    employed_field = col1.radio("Employed",("Yes","No"),horizontal=True)
    if employed_field == 'Yes':
        employed=1
    
    # select salary yes/no object
    salaried=0
    
    salaried_field = col2.radio("Salaried",("Yes","No"),horizontal=True)
    if salaried_field == 'Yes':
        salaried=1
    
    col1, col2 =st.columns(2)
    
    # select scheduled departure time
    age = col1.number_input("Age",
                              min_value=0.00,
                              max_value=85.00,
                              value=30.00,
                              step=0.01,
                             )
    # select ehnicity
    ethnicity_white = 0
    ethnicity_black = 0
    ethnicity_latino = 0
    ethnicity_asian = 0
    ethnicity_other = 0

    ethinicity = col2.selectbox("Ethnicity",("White","Black","Latino","Asian","Others"))
    
    if ethinicity == 'White':
        ethnicity_white=1
    elif ethinicity == 'Black':
        ethnicity_black=1
    elif ethinicity == 'Latino':
        ethnicity_latino=1
    elif ethinicity == 'Asian':
        ethnicity_asian=1
    elif ethinicity == 'Others':
        ethnicity_other=1
    
    col1, col2 =st.columns(2)
    
    # select gender male/female
    gender=0
    
    gender_field = col1.radio("Gender",("Male","Female"),horizontal=True)
    if gender_field == 'Male':
        gender=1
        
    married=0
    
    # select married yes/no object
    married_field = col2.radio("Married",("Yes","No"),horizontal=True)
    if married_field == 'Yes':
        married=1
    
    col1, col2 =st.columns(2)
    
    # select dept
    debt = col1.number_input("Dept",
                              min_value=0.00,
                              max_value=30.00,
                              value=15.00,
                              step=0.01,
                             )
    
    credit_score = col2.number_input("Credit Score",
                              min_value=0,
                              max_value=100,
                              value=30,
                              step=1,
                             )
    
    col1, col2 =st.columns(2)
    
    # select bank customer yes/no object
    bank_customer=0
    
    bank_customer_field = col1.radio("Bank Customer",("Yes","No"),horizontal=True)
    if bank_customer_field == 'Yes':
        bank_customer=1
    
    # select prior default yes/no object
    prior_default=0
    
    prior_default_field = col2.radio("Prior Default",("Yes","No"),horizontal=True)
    if prior_default_field == 'Yes':
        prior_default=1
    
    
    col1, col2 =st.columns(2)
    
 # select citizen object
    citizen_bybirth = 0
    citizen_other = 0
    citizen_temporary = 0

    citizen_type = col1.selectbox("Citizen Type",("By Birth","Temporary","Others"))
    
    if citizen_type == 'By Birth':
        citizen_bybirth=1
    elif citizen_type == 'Temporary':
        citizen_temporary=1
    elif citizen_type == 'Others':
        citizen_other=1
    
    # select drivers license yes/no object
    drivers_license=0
    
    drivers_license_field = col2.selectbox("Driving License",("Yes","No"))
    if drivers_license_field == 'Yes':
        drivers_license=1
    
        
    # normalize user input using the scaler used during training of model 
    age,debt,years_employed = p3_scaler.transform([[age,debt,years_employed]])[0]
    
    return [[gender,age,debt,married,bank_customer,emp_industrial,emp_materials,
             emp_consumer_services,emp_healthcare,emp_financials,emp_utilities,emp_education,
             ethnicity_white,ethnicity_black,ethnicity_latino,ethnicity_asian,ethnicity_other,
             years_employed,prior_default,employed,credit_score,drivers_license,citizen_bybirth,citizen_other,citizen_temporary,
             salaried]]
    
# function to display the user input template based on the dataset selection
def get_data(dataset_name):
    if dataset_name == 'System Administrator':
        new_data = get_data_system_administrator()
    elif dataset_name == 'Flight Delay':
        new_data = get_data_flight_delay()
    elif dataset_name == 'Loan Approval':
        new_data = get_data_loan_approval()
    
    return new_data
  
# function returns the message if model predicts 1 as taget value
def message_1(dataset_name):
    if dataset_name == 'System Administrator':
        message = 'Task will be completed'
    elif dataset_name == 'Flight Delay':
        message = 'Flight will be delayed'
    elif dataset_name == 'Loan Approval':
        message = 'Loan will be granted'
    
    return message

# function returns the message if model predicts 0 as taget value
def message_0(dataset_name):
    if dataset_name == 'System Administrator':
        message = 'Task will not be completed'
    elif dataset_name == 'Flight Delay':
        message = 'Flight will be on time'
    elif dataset_name == 'Loan Approval':
        message = 'Loan will be denied'
    
    return message

# function to predict the target variable based on selection of dataset, classifier, and model parameters
def prediction(classifier_name,dataset_name,new_data,model_parameters):
    model = model_selection(dataset_name,classifier_name)
    predicted_probability = model.predict_proba(new_data)[0][1]
    mess_1 = message_1(dataset_name)
    mess_0 = message_0(dataset_name)
       
    if predicted_probability < model_parameters['th']:
        pred = mess_0
        st.error(pred)
    else:
        pred = mess_1
        st.success(pred)
    
    st.write('Predicted probability is ',round(predicted_probability*100,2),'%')
    st.write('Threshold probability is ',round(model_parameters['th']*100,2),'%')

    return pred

# ****************************
# **** create page layout ****
# ****************************

st.sidebar.image('https://nbmbaa.org/wp-content/uploads/2018/05/AZ-WPC-logo.png')

st.sidebar.write('# ASSIGNMENT 3')

dataset_name = st.sidebar.selectbox("Select Dataset",("Loan Approval","System Administrator","Flight Delay"))

classifier_name = st.sidebar.selectbox("Select Classifier",("Logistic Regression","kNN"))

page_title = dataset_name + ' : ' + classifier_name

st.write(' ##',page_title)

if dataset_name == 'System Administrator':
    st.write('*Predict if the task will be completed or not*')

elif dataset_name == 'Flight Delay':
    st.write('*Predict if the flight will be delayed or not*')

elif dataset_name == 'Loan Approval':
    st.write('*Predict if the loan will be approved or not*')

st.write('*Select model / Update threshold from sidebar*')

selected = option_menu(menu_title=None,
                       options = ["Predict New Data","Model Performance"],
                       icons = ["pencil-fill","bar-chart-fill"], # https://icons.getbootstrap.com/
                       orientation ="horizontal"
                      )

model_parameters= model_params(dataset_name,classifier_name)
model_metrics=model_performance_metrics(dataset_name,classifier_name,model_parameters)

st.sidebar.markdown("""---""")

# section to predict the output of new data
if selected =='Predict New Data':

    # new data layout
    new_data = get_data(dataset_name)
    # predict button to predict the output based on user input
    if st.button('Predict'):
        result = prediction(classifier_name,dataset_name,new_data,model_parameters)
    #     st.success(result)

    st.markdown("""---""")

# section to view dashboard of performance metrics
else:
    plot_perfomance_metrics(model_metrics,model_parameters)

    
