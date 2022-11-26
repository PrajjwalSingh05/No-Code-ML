

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
import pickle

class download_model():
    

    def download_random_forest_regressor(self,X,y,paramlist):
        #****************************Unpacking Tuple **************************************************
        input_col,Icriterion,Imaxfeature,Isamples_split,In_estimator=paramlist
        #****************************Data Preprocessing******************* ******************************
        preprocessor= self.data_preprocessor(X,y)
        xtrain,xtest,ytrain,ytest=train_test_split(X,y,test_size=0.2,random_state=45)
       
        #*********************************************Creating Model***************************************
        model_selector = Pipeline(
                    steps=[("preprocessor", preprocessor),
                    ("feature", SelectKBest(f_regression,k=input_col)),
                    ("classifier", RandomForestRegressor(criterion=Icriterion,max_features=Imaxfeature,min_samples_split=Isamples_split,
                    n_estimators=In_estimator))]
                )
        model_selector.fit(xtrain,ytrain)
        #*********************************************Result Generation ***************************************
        ypred=model_selector.predict(xtest)
        result=r2_score(ytest,ypred)
        st.write(f"Accuracy with without parameter{result}")
        return model_selector           
    def download_ExtraTreeRegressor(self,X,y,paramlist):

        #****************************Unpacking Tuple **************************************************
        input_col,Icriterion,Imin_samples_split,In_estimator=paramlist
        #****************************Data Preprocessing******************* ******************************
        preprocessor= self.data_preprocessor(X,y)
        xtrain,xtest,ytrain,ytest=train_test_split(X,y,test_size=0.2,random_state=45)
       
        #*********************************************Creating Model***************************************
        model_selector = Pipeline(
                    steps=[("preprocessor", preprocessor),
                    ("feature", SelectKBest(f_regression,k=input_col)),
                    ("classifier", RandomForestRegressor(criterion=Icriterion,min_samples_split=Imin_samples_split,
                    n_estimators=In_estimator))]
                )
        model_selector.fit(xtrain,ytrain)
        ypred=model_selector.predict(xtest)
        result=r2_score(ytest,ypred)
        #*********************************************Result Generation ***************************************
        st.write(f"Accuracy with without parameter{result}")
        return model_selector       
    def download_randomforest_classifier(self,X,y,paramlist):
        #    parameter_list=[]
        #****************************Unpacking Tuple **************************************************
        input_col,Icriterion,Imax_features,Imin_samples_split,In_estimator=paramlist
        #****************************Data Preprocessing******************* ******************************
        preprocessor= self.data_preprocessor(X,y)
        xtrain,xtest,ytrain,ytest=train_test_split(X,y,test_size=0.2,random_state=45)
        #*********************************************Creating Model***************************************
        model_selector = Pipeline(
                    steps=[("preprocessor", preprocessor),
                    ("feature", SelectKBest(f_regression,k=input_col)),
                    ("classifier", RandomForestClassifier(criterion=Icriterion,min_samples_split=Imin_samples_split,
                    n_estimators=In_estimator))]
                )
        model_selector.fit(xtrain,ytrain)

        #*********************************************Generating result **************************************

        ypred=model_selector.predict(xtest)
        st.write("Confusion Matrix is :")
        st.write(confusion_matrix(ypred,ytest))
        st.write("CLassification Report is :")
        st.write(classification_report(ypred,ytest))
        return model_selector
    def download_extratree_classifier(self,X,y,paramlist):
        #    parameter_list=[]
# **********************************************************Tuple Unpacking/***********************************
        input_col,Imax_features,Icriterion,Imax_depth,Imin_samples_split=paramlist
        preprocessor= self.data_preprocessor(X,y)
        xtrain,xtest,ytrain,ytest=train_test_split(X,y,test_size=0.2,random_state=45)
# **********************************************************Model Creating***********************************
        model_selector = Pipeline(
                    steps=[("preprocessor", preprocessor),
                    ("feature", SelectKBest(f_regression,k=input_col)),
                    ("classifier", RandomForestClassifier(criterion=Icriterion,min_samples_split=Imin_samples_split,
                    ))]
                )
        model_selector.fit(xtrain,ytrain)
        ypred=model_selector.predict(xtest)
# ********************************************************************Generating result**************************
        st.write("Confusion Matrix is :")
        st.write(confusion_matrix(ypred,ytest))
        st.write("CLassification Report is :")
        st.write(classification_report(ypred,ytest))
        return model_selector
           
    def download_interface_ExtraTreeRegressor(self):
         
                parameter_list=[]
                st.markdown("#### You Are Using Download Extra Tree Regressor ")
# ************************************File Uploader********************************************************
                file = st.file_uploader("Upload Your File ",type=["csv","xlsx"])
                prediction_column=st.text_input("Enter predictiion Column ")
                st.write(prediction_column)
# ******************************************Creating Interface************************************************
                input_col=st.slider ("Number of Input Column ",3,1000)               
                criterion=st.selectbox("Choose a Criterion",["squared_error", "absolute_error"])
                max_depth=st.number_input("Range Of max_depth",min_value=1,max_value=50)
                n_estimators=st.number_input("n_estimators",min_value=100,max_value=5000)
                min_samples_split=st.number_input("Range Of min_samples_split",min_value=2,max_value=50)
                btn=st.button("Model Train")
                if btn:
# *****************************************Updating Parameter*************************
                    parameter_list.append(input_col)
                    parameter_list.append(criterion)
                    parameter_list.append(min_samples_split)
                    parameter_list.append(max_depth)
                    if file is None:
                            st.warning("Please Upload a file")
# **************************************************Reading Data file*****************************
                    data=self.data_extract(file)
                    maxcol=data.shape[1]-2
                    # st.write("Inside extretree")
                    x=data.drop(columns=prediction_column)
                    y=data[prediction_column]
                    parameter_list=tuple(parameter_list)   
                    if self.parameter_checkup(input_col,maxcol,x,y):
                            model_file1=self.download_ExtraTreeRegressor(x,y,parameter_list)
                           
                    try:
                                
                                btn=st.download_button("Download Model",data=pickle.dumps(model_file1),file_name="model.pkl", )
                    except Exception as e:
        
                              pass
    def download_interface_random_forest_regression(self):
                    st.markdown("#### You Are Using Download Random Forest Regression")
                    parameter_list=[]
    # ***********************************************Uploading File***************************************************
                    file = st.file_uploader("Upload Your File ",type=["csv","xlsx"])
                    prediction_column=st.text_input("Enter Predictiion Column")
                    # st.write(prediction_column)
                   
                    input_col=st.number_input("Input column",1,1000 ,help="Step up is 5")
            

                    criterion=st.selectbox("Choose Criterion",["absolute_error","squared_error"])
    # ***********************************************Creating Max  Feature Parameter**********************************
                    maxfeature=st.selectbox("Max_features",["sqrt","log2","None"])
            
    # ***********************************************Creating Max  Min_Sample_split Parameter*************************                
           
                    samples_split=st.number_input("Min_samples_split",3,20 ,help="Step up is 1")
    # ***********************************************Creating n_estimator Parameter***********************************
            
                    n_estimators=st.number_input("N_estimator",100,1000 ,help="Step up is 5")
                   
# ***********************************************Clicking On submit button Parameter******************************
                    btn=st.button("Model Train")
                    if btn:
# *************************************Updating Parameter*********************************
                        parameter_list.append(input_col)
                        parameter_list.append(criterion)
                        parameter_list.append(maxfeature)
                        parameter_list.append(samples_split)
                        parameter_list.append(n_estimators)
                        # st.write(type(parameter_list))
                        # st.write(type(parameter_list))
                       
                        if file is None:
                            st.warning("Please Upload a file")
                        data=self.data_extract(file)
                        maxcol=data.shape[1]-2
                        st.write("Inside extretree")
                        x=data.drop(columns=prediction_column)
                        y=data[prediction_column]
    # ***********************************************Updating Parameter list******************************************
                        parameter_list=tuple(parameter_list)
                       
    # ***********************************************Updating MAx Feaute  list****************************************                  
                        # model=RandomForestRegressor
                        if self.parameter_checkup(input_col,maxcol,x,y):
                            model_file1=self.download_random_forest_regressor(x,y,parameter_list)
                        try:
                              
                            btn=st.download_button("Download Model",data=pickle.dumps(model_file1),file_name="model.pkl", )
                        except Exception as e:
        
                              pass
    def download_interface_ExtraTreeClassifier(self):
                """It Genarte an Download Interface For ExtraTreeClassifier"""

                parameter_list=[]
                st.markdown("#### You Are Using Download Extra Tree Classifie ")
# ****************************************Upload File************************************************
                file = st.file_uploader("Upload Your File ",type=["csv","xlsx"])
                prediction_column=st.text_input("Enter Predictiion Column")
                st.write(prediction_column)
# ******************************************Interface*******************************************
                input_col=st.slider ("Number of Input Column",3,1000)
                max_features=st.selectbox("Choose a Max Features",["auto", "sqrt", "log2"])
            
                criterion=st.selectbox("Choose a Criterion",["gini","entropy","log_loss"])
                max_depth=st.number_input("Input Max Depth " ,min_value=1,max_value=1000)
                min_samples_split=st.number_input("Input Sample Split ",min_value=3,max_value=50)
               
                btn=st.button("Model Train")
                if btn:
                    if file is None:
                        st.warning("Please Upload a file")


                    data=self.data_extract(file)
                    maxcol=data.shape[1]-2
                    x=data.drop(columns=prediction_column)
                    le=LabelEncoder()
                    y=le.fit_transform(data[prediction_column])

# **********************************************UPdating Parameter*************************************
                    parameter_list.append(input_col)
                    parameter_list.append(max_features)
                    parameter_list.append(criterion)
                    parameter_list.append(max_depth)
                    parameter_list.append(min_samples_split)
                    parameter_list=tuple(parameter_list)
                    if self.parameter_checkup(input_col,maxcol,x,y):
                      model_file=self.download_extratree_classifier (x,y,parameter_list)
                      try:
                            btn=st.download_button("Download Model",data=pickle.dumps(model_file),file_name="model.pkl", )
                      except Exception as e:
        
                              pass
                       
    def download_interface_randomforest_classifier(self):
                """It Genarte an interface for  download Random Forest Classifier """
                st.markdown(" ### You Are Using   dwonlaod Random Forest Classifier")
                parameter_list=[]
# ***************************File Uploader***********************************************************
                file = st.file_uploader("Upload an image",type=["csv","xlsx"])
                prediction_column=st.text_input("Enter predictiion")
                st.write(prediction_column)
                input_col=st.slider ("Number of input",3,1000)

                criterion=st.selectbox("Choose a Criterion",["gini","entropy","log_loss"])
                max_features=st.selectbox("Choose a Max Features",["sqrt","log2",None])

                max_depth=st.number_input("Input Max Depth " ,min_value=1,max_value=1000)
                min_samples_split=st.number_input("Input Sample Split ",min_value=2,max_value=50)
                n_estimator=st.number_input("Input n Estimator",min_value=100,max_value=5000)
                # submit = st.form_submit_button("Submit")
                # if submit:
                btn=st.button("Model Train")
                if btn:
                    if file is None:
                        st.warning("Please Upload a file")
                    data=self.data_extract(file)
                    maxcol=data.shape[1]-2
                    x=data.drop(columns=prediction_column)
                    le=LabelEncoder()
                   


                    y=le.fit_transform(data[prediction_column])
# **********************************************Updating Parameter******************************
                    parameter_list.append(input_col)
                    parameter_list.append(criterion)
                    parameter_list.append(max_features)
                    parameter_list.append(min_samples_split)
                    parameter_list.append(n_estimator)
                    parameter_list=tuple(parameter_list)
                    if self.parameter_checkup(input_col,maxcol,x,y):
                            # st.write("inside if function")
                        #  self.random_forest_classifier(x,y,1,input_col)
                           model_file=self.download_randomforest_classifier  (x,y,parameter_list)
                    try:
                              
                            btn=st.download_button("Download Model",data=pickle.dumps(model_file),file_name="model.pkl", )
                    except Exception as e:
        
                              pass