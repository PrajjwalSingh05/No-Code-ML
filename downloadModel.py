

import streamlit as st
import numpy as np
import pandas as pd
from joblib import load
from PIL import Image
import tensorflow as tf
import streamlit as st
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import  matplotlib.pyplot as plt
from  sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import SelectKBest,chi2,RFE
from sklearn.tree import ExtraTreeClassifier,ExtraTreeRegressor
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier
from sklearn.model_selection import train_test_split,GridSearchCV
from imblearn.over_sampling import SMOTE,ADASYN

from sklearn.preprocessing import StandardScaler,OneHotEncoder,LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import  make_pipeline,Pipeline
from sklearn.metrics import r2_score,mean_absolute_error
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import f_regression
from sklearn.metrics import confusion_matrix,classification_report

from joblib import dump

class download_model:
    def download_model(slef,model):
        dump(model, 'created_model.jb')
    def download_random_forest_regressor(self,X,y,paramlist):
        input_col,Icriterion,Imaxfeature,Isamples_split,In_estimator=paramlist
        preprocessor= self.data_preprocessor(X,y)
        xtrain,xtest,ytrain,ytest=train_test_split(X,y,test_size=0.2,random_state=45)
  
        # # ****************Feature Seclortot**********************************************
        # feature_selector = Pipeline(
        #             steps=[("preprocessor", preprocessor),
        #             ("feature", SelectKBest(f_regression,k=i))])
        # feature_selector.fit(xtrain,ytrain)
        # ****************Model Seclortot**********************************************
        model_selector = Pipeline(
                    steps=[("preprocessor", preprocessor),
                    ("feature", SelectKBest(f_regression,k=input_col)),
                    ("classifier", RandomForestRegressor(criterion=Icriterion,max_features=Imaxfeature,min_samples_split=Isamples_split,
                    n_estimators=In_estimator))]
                )
        model_selector.fit(xtrain,ytrain)
        #****************************Result Generation ******************************
        ypred=model_selector.predict(xtest)
        result=r2_score(ytest,ypred)
                # print(f"R2score is without estimator {result}")
        st.write(f"Accuracy with without parameter{result}")
        return model_selector
            
        #*********************************Working on features****************************
                
    def download_interface_random_forest_regression(self):
        with st.form("Form1"):    
                parameter_list=[]
# ***********************************************Uploading File***************************************************
                file = st.file_uploader("Upload an image",type=["csv","xlsx"])
                prediction_column=st.text_input("Enter predictiion")
                st.write(prediction_column)
                # input_col=st.slider ("Number of input",3,1000)
                input_col=st.number_input("Input column",1,1000 ,help="Step up is 5")
                # parameter_list.append(input_col)

                criterion=st.selectbox("Choose Criterion",["absolute_error","squared_error"])
# ***********************************************Creating Max  Feature Parameter**********************************
                maxfeature=st.selectbox("Max_features",["sqrt","log2","None"])
           
# ***********************************************Creating Max  Min_Sample_split Parameter*************************                
                # st.title("min_samples_split")
                samples_split=st.number_input("Min_samples_split",1,20 ,help="Step up is 1")
# ***********************************************Creating n_estimator Parameter***********************************
                # st.title("n_estimator")
                n_estimators=st.number_input("N_estimator",100,1000 ,help="Step up is 5")
                submit = st.form_submit_button("Submit")
# ***********************************************Clicking On submit button Parameter******************************
                if submit:
                    print("submiting")
                    print("submiting")
                    print("submiting")
                    print("submiting")
                    print("submiting")
                    print("submiting")
                    parameter_list.append(input_col)
                    parameter_list.append(criterion)
                    parameter_list.append(maxfeature)
                    parameter_list.append(samples_split)
                    parameter_list.append(n_estimators)
                    st.write(type(parameter_list))
                    st.write(type(parameter_list))
                    st.write("working on download interface")
                    st.write("working on download interface")
                    st.write("working on download interface")
                    if file is None:
                        st.warning("Please Upload a file")
                    data=self.data_extract(file)
                    maxcol=data.shape[1]-2
                    st.write("Inside extretree")
                    x=data.drop(columns=prediction_column)
                    y=data[prediction_column]
# ***********************************************Updating criterion list******************************************
                    parameter_list=tuple(parameter_list)
                    st.write("working on download interface")
                    st.write("working on download interface")
                    st.write("working on download interface")
                    print("working on download interface")
                    print("working on download interface")
                    print("working on download interface")
                    print("working on download interface")
                    print("working on download interface")
                    print("working on download interface")
                    print("working on download interface")
                    print("working on download interface")
                    print("working on download interface")
# ***********************************************Updating MAx Feaute  list****************************************                  
                    # model=RandomForestRegressor
                    # if self.parameter_checkup(input_col,maxcol,x,y):
                    model=self.download_random_forest_regressor(x,y,parameter_list)
                    self.download_model(model)
                else:
                     print("submit not working")
                     print("submit not working")
                     print("submit not working")
                     print("submit not working")
                # download_model_button=st.download_button("Download Your model ",download_model)