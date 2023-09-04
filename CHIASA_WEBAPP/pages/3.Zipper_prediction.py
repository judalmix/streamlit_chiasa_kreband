#imports

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import StringIO
import pickle
import numpy
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from xgboost.sklearn import XGBClassifier
import shap
from streamlit_shap import st_shap



#functions
def load_model(filename): #load the model from notebook
    with open(filename, 'rb') as file:
        data = pickle.load(file)
    return data
def quartils_results(new_prediction):
    if new_prediction==0:
        st.write('The prediction is that the zipper will be sold in the 1st Quartile.')
        st.write('This means that the quantity sold of this zipper will be less or equal than:', quartils[0])
        st.write('')
    elif new_prediction==1:
        st.write('The prediction is that the zipper will be sold in the 2nd Quartile.')
        st.write('This means that the quantity sold of this zipper will be between:', quartils[0], 'and', quartils[1])
        st.write('')
    elif new_prediction==2:
        st.write('The prediction is that the zipper will be sold in the 3rd Quartile.')
        st.write('This means that the quantity sold of this zipper will be between:', quartils[1], 'and', quartils[2])
        st.write('')
    else: 
        st.write('The prediction is that the zipper will be sold in the 4th Quartile.')
        st.write('This means that the quantity solf of this zipper will be at least:', quartils[3])
        st.write('')

def apply_regression(data,nom_ultima_col):
    data=data.drop('Quartils',axis=1)
    model=LinearRegression()
    y=data.loc[: , nom_ultima_col]
    x=data.loc[: , data.drop(nom_ultima_col, axis=1).columns]
    X_train, X_test, Y_train, Y_test= train_test_split(x,y,train_size=0.8)
    regression_model=model

    model_fit=regression_model.fit(X_train, Y_train)
    predict = model_fit.predict(X_test)
    relative_error=np.abs(predict-Y_test)/np.abs(Y_test)
    relative_error_mean=relative_error.mean()

    return regression_model, X_test, x, Y_test,relative_error_mean

def multiclass_classification(data,nom_ultima_col):
    data=data.drop(nom_ultima_col,axis=1)
    model_xgboost=XGBClassifier()
    y2=data.loc[: , 'Quartils']
    x2=data.loc[: , data.drop('Quartils', axis=1).columns]
    X_train2, X_test2, Y_train2, Y_test2= train_test_split(x2,y2,train_size=0.8, test_size=0.2)
    model_xgboost_multiclass = model_xgboost.fit(X_train2, Y_train2)
    predict_xgb = model_xgboost_multiclass.predict(X_test2)
    
    return  model_xgboost,X_train2, X_test2, Y_train2, Y_test2,x2,y2

def array_to_dataset(idx,num_cols,new_df):
    new_data=pd.DataFrame()
    j=0
    for i in new_df.columns: 
        if j<num_cols:
            new_data[i]=[idx[j]]
            new_data.i=new_data.astype(int)
        j=j+1
    return new_data




def zippers_model(new_df, multiclass_model, regression_model, X_test_regression_reduced,x_regression_reduced, relative_error_mean_reduced,nom_ultima_col):
    num = st.slider('Insert the ID of the zipper you would like to predict', min_value=0, max_value=new_df.shape[0],key='10000')
    st.write('The ID selected is ', num)
    idx_inicial=np.int(num)
    aux=new_df.iloc[idx_inicial]
    aux1=aux.drop(['Quartils',nom_ultima_col])
    new_data=array_to_dataset(aux1,len(aux1),new_df)
    st.write(' ')

    multiclass_prediction_button=st.button('See MULTICLASS prediction',key='99')
    regression_prediction_button=st.button('See REGRESSION prediction',key='98')

    if multiclass_prediction_button:
        #multiclass
        new_prediction_multiclass= multiclass_model.predict(new_data)
        st.write('')
        st.write('The zipper was sold in this quartile: ', aux['Quartils']+1 )
        results_multiclass_prediction=quartils_results(new_prediction_multiclass)
    if regression_prediction_button:
        st.write('')
        results_linear_regression = regression_model.predict(new_data) 
        relative_error=((np.abs(results_linear_regression-aux[nom_ultima_col]))/(np.abs(aux[nom_ultima_col])))*100
        if relative_error>=relative_error_mean_reduced:
            st.write('The linear regression is not giving an acurate prediction for this zipper.')
            st.write('Is better to rely on the MULTICLASS CLASSIFICATION.')
        else:
            st.write('The prediction of the quantity of zippers that will be sold in the target variable is: ')
            st.write(results_linear_regression)
    #idx_shap_plot=input('Introduce a integer for see the individual shap plots (HAS TO BE BETWEEN 0-135): ')
    #idx_shap_plot=int(idx_shap_plot)
    #individual_shap_plot(regression_model,X_test_regression_reduced,x_regression_reduced,idx_shap_plot)

def model(new_df,regression_model_reduced):
    new_df=new_df.drop('Quartils',axis=1)
    target=new_df.iloc[:,7:]
    not_target=new_df.iloc[:,:7]
    #model=LinearRegression()
    model_fit=regression_model_reduced.fit(not_target, target)
    prediction=model_fit.predict(not_target)
    for i in range(prediction.shape[0]):
        for j in range(prediction.shape[1]):
            if prediction[i][j]<=0:
                prediction[i][j]=0
    new_future_sales=pd.DataFrame(prediction)
    not_target = not_target.reset_index(drop=True)
    result = pd.concat([not_target, new_future_sales],axis=1)
    new_cols = {}
    for i, col in enumerate(result.columns[7:]):
        new_col_name = f'{i+1}{"st" if i%10==0 and i!=10 else ""} prediction'
        new_cols[col] = new_col_name 
    df_encoded = result.rename(columns=new_cols)
    
    df_encoded_0=df_encoded.copy()

    decodings = {}
    
    for col in df_encoded.columns[:7]:
        if col in values_dict:
            categories, _ = values_dict[col]
            decodings[col] = dict(enumerate(_))
            df_encoded[col] = df_encoded[col].map(lambda x: decodings[col][x])

    return  df_encoded_0,df_encoded

def convert_to_string(dataframe):
    num_initial_cols = 7
    # Get the number of columns that are integers
    num_int_cols = dataframe.shape[1] - num_initial_cols
    # Create a list of new column names that are strings
    new_col_names = [str(i+num_initial_cols) for i in range(num_int_cols)]
    # Rename the integer columns using the new column names
    dataframe.columns.values[num_initial_cols:num_initial_cols + num_int_cols] = new_col_names
    for i in range(len(new_col_names)):
        df=dataframe.rename(columns={num_initial_cols:new_col_names[0]})
        num_initial_cols=num_initial_cols+1
    return df

if 'diccionari' in st.session_state:
    values_dict=st.session_state['diccionari']  
if 'function' in st.session_state:
    encoding_data=st.session_state['function']
if 'function2' in st.session_state:
    group_by_months=st.session_state['function2']
if 'function3' in st.session_state:
    rename_columns=st.session_state['function3']
if 'numero' in st.session_state:
    numero=st.session_state['numero']
if 'data_processed' in st.session_state:
    dataframe=st.session_state['data_processed']


dataset=st.session_state.data

#data_covid_encoded=st.session_state.dataa_covid
#df_not_encoded_covid=st.session_state.data_covid_not_encoded

    
    

#starting plotting
st.title('Zipper prediction with ML')
st.write('')
st.write("We are going to see some predictions of our datasets using Machine Learning and Shap plots")
st.write('You will be able to see do two different types of prediction:')
st.write('1-You will be able to see the prediction of one zipper for the next month.')
st.write('2-Prediction for all the data. ')
st.write('')
st.write('')
st.write('Here you can find the prediction and the explainability of the model. ')

df=convert_to_string(dataset)
nom_ultima_col = df.columns[-1]
new_df= df[(df[nom_ultima_col]>0)]
quartils= numpy.quantile(new_df[nom_ultima_col], [0.25,0.5,0.75,1])
q=[]
for i in range(len(new_df)):
    if new_df[nom_ultima_col].iloc[i]<=quartils[0]:
        q.append(0)
    elif new_df[nom_ultima_col].iloc[i]>quartils[0] and new_df[nom_ultima_col].iloc[i]<=quartils[1]:
        q.append(1)
    elif new_df[nom_ultima_col].iloc[i]>quartils[1] and  new_df[nom_ultima_col].iloc[i]<=quartils[2]:
        q.append(2)
    else:
        q.append(3)
new_df['Quartils']=q

model_regression,X_test_regression_reduced,x_regression_reduced,Y_test_regression_reduced, relative_error_mean_reduced=apply_regression(new_df,nom_ultima_col)
model_xgboost,X_train2, X_test2, Y_train2, Y_test2,x2,y2=multiclass_classification(new_df,nom_ultima_col)



with st.expander("Prediction"):
       
    tab1,tab2= st.tabs(["Prediction of a zipper already in the dataset", " Prediction Dataset for the next months"])
    with tab1: 
        st.write('You will do a prediction with a zipper already in the dataset!')
        st.write('Here you can find the dataset:  ')
        zippers_model(new_df, model_xgboost, model_regression, X_test_regression_reduced,x_regression_reduced, relative_error_mean_reduced,nom_ultima_col)

    with tab2: 
        st.write('Prediction for the next months: ')
        st.write('')
        df_encoded,df_not_encoded=model(new_df,LinearRegression())
        st.write('')
        st.write('The prediction of the dataset NOT ENCODED is: ')
        st.write(df_not_encoded)

shap_values_multiclass = shap.TreeExplainer(model_xgboost).shap_values(X_test2)
#explainer = shap.Explainer(model_regression.predict, X_test_regression_reduced)
#shap_values_regressor = explainer(X_test_regression_reduced)

explainer = shap.Explainer(model_regression.predict,X_test_regression_reduced )
shap_values_regressor = explainer(X_test_regression_reduced)

#explainer = shap.Explainer(model.predict, X_test)
#shap_values = explainer(X_test)

with st.expander("Explainable SHAP Plots"):
    tab1,tab2=st.tabs(["MULTICLASS Plots", "Linear Regression Plots"])
    with tab1: 
        st.write('SHAP PLOT for MULTICLASS: ')
        #shap.summary_plot(shap_values_multiclass, X_test2)
        st_shap(shap.summary_plot(shap_values_multiclass,X_test2))
        

    with tab2: 
        tab1,tab2,tab3= st.tabs([ "Summary Plot", "Bar plot","Individual Plots"])
        with tab1: 
            st.write("The x-axis represents the shap values and the y-axis are the most important features for the output of the model. Moreover, this plot shows how the different values of the variables contribute to the model with a positive and negative effect in order to predict the quantity of zipper sold.")
            st.write('')
            st_shap(shap.summary_plot(shap_values_regressor, X_test_regression_reduced))
        with tab2: 
            st.write("The x-axis represents the shap values and the y-axis represents the most relevant features for the Linear Regression model.")
            st.write('')
            st_shap(shap.plots.bar(shap_values_regressor,max_display=10))
        with tab3: 
            st.write(' The x-axis represents the shap values and the y-axis represents the most relevant features for the ID selected from the test data of the output model.')
            st.write('We can also see the differnt values for each characteristic next to the feature name. The various values are from our test data with the ID picked. Those variables that will have a negative influence on the number of zippers sold with these features are noted in blue, while those that will raise the number of zippers sold are marked in red.')
            idx_shap_plot = st.slider('Choose an ID to see the individual SHAP Plot. ', 0, X_test_regression_reduced.shape[0], X_test_regression_reduced.shape[1],key='100000')
            idx_shap_plot=np.int(idx_shap_plot)
            st_shap(shap.plots.waterfall(shap_values_regressor[idx_shap_plot]))
            

        


