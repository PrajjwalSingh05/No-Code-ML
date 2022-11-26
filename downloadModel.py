

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
    # global model_file
    # def __init__(self):
    #      model_file=""
    # def download_model(self,model):
    #     # dump(model, 'created_model.jb')
    #      st.download_button("Download Model",data=pickle.dumps(model),file_name="model.pkl")

    def download_random_forest_regressor(self,X,y,paramlist):
        input_col,Icriterion,Imaxfeature,Isamples_split,In_estimator=paramlist
        preprocessor= self.data_preprocessor(X,y)
        xtrain,xtest,ytrain,ytest=train_test_split(X,y,test_size=0.2,random_state=45)
       
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
        st.write(f"Accuracy with without parameter{result}")
        return model_selector           
    def download_ExtraTreeRegressor(self,X,y,paramlist):
        input_col,Icriterion,Imin_samples_split,In_estimator=paramlist
        preprocessor= self.data_preprocessor(X,y)
        xtrain,xtest,ytrain,ytest=train_test_split(X,y,test_size=0.2,random_state=45)
       
        model_selector = Pipeline(
                    steps=[("preprocessor", preprocessor),
                    ("feature", SelectKBest(f_regression,k=input_col)),
                    ("classifier", RandomForestRegressor(criterion=Icriterion,min_samples_split=Imin_samples_split,
                    n_estimators=In_estimator))]
                )
        model_selector.fit(xtrain,ytrain)
        #****************************Result Generation ******************************
        ypred=model_selector.predict(xtest)
        result=r2_score(ytest,ypred)
        st.write(f"Accuracy with without parameter{result}")
        return model_selector       
    def download_randomforest_classifier(self,X,y,paramlist):
        #    parameter_list=[]
        input_col,Icriterion,Imax_features,Imin_samples_split,In_estimator=paramlist
        preprocessor= self.data_preprocessor(X,y)
        xtrain,xtest,ytrain,ytest=train_test_split(X,y,test_size=0.2,random_state=45)
        model_selector = Pipeline(
                    steps=[("preprocessor", preprocessor),
                    ("feature", SelectKBest(f_regression,k=input_col)),
                    ("classifier", RandomForestClassifier(criterion=Icriterion,min_samples_split=Imin_samples_split,
                    n_estimators=In_estimator))]
                )
        model_selector.fit(xtrain,ytrain)
        ypred=model_selector.predict(xtest)
        # print(confusion_matrix(ypred,ytest))
        #  print(classification_report(ypred,ytest))
        st.write("Confusion Matrix is :")
        st.write(confusion_matrix(ypred,ytest))
        st.write("CLassification Report is :")
        st.write(classification_report(ypred,ytest))
        return model_selector
    def download_extratree_classifier(self,X,y,paramlist):
        #    parameter_list=[]
        input_col,Imax_features,Icriterion,Imax_depth,Imin_samples_split=paramlist
        preprocessor= self.data_preprocessor(X,y)
        xtrain,xtest,ytrain,ytest=train_test_split(X,y,test_size=0.2,random_state=45)
        model_selector = Pipeline(
                    steps=[("preprocessor", preprocessor),
                    ("feature", SelectKBest(f_regression,k=input_col)),
                    ("classifier", RandomForestClassifier(criterion=Icriterion,min_samples_split=Imin_samples_split,
                    ))]
                )
        model_selector.fit(xtrain,ytrain)
        ypred=model_selector.predict(xtest)
        # print(confusion_matrix(ypred,ytest))
        #  print(classification_report(ypred,ytest))
        st.write("Confusion Matrix is :")
        st.write(confusion_matrix(ypred,ytest))
        st.write("CLassification Report is :")
        st.write(classification_report(ypred,ytest))
        return model_selector
           
    def download_interface_ExtraTreeRegressor(self):
         
                parameter_list=[]
                file = st.file_uploader("Upload an image",type=["csv","xlsx"])
                prediction_column=st.text_input("Enter predictiion")
                st.write(prediction_column)
                input_col=st.slider ("Number of input",3,1000)
                st.title("Criterion")
                criterion=st.selectbox("Choose a Criterion",["squared_error", "absolute_error"])
                st.title("Splitter")
                max_depth=st.number_input("Range Of max_depth",min_value=1,max_value=50)
                n_estimators=st.number_input("n_estimators",min_value=100,max_value=5000)
                min_samples_split=st.number_input("Range Of min_samples_split",min_value=1,max_value=50)
                btn=st.button("Model Train")
                if btn:
                    parameter_list.append(input_col)
                    parameter_list.append(criterion)
                    parameter_list.append(min_samples_split)
                    parameter_list.append(max_depth)
                        # parameter_list.append(n_estimators)
                    if file is None:
                            st.warning("Please Upload a file")
                    data=self.data_extract(file)
                    maxcol=data.shape[1]-2
                    st.write("Inside extretree")
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
                    # st.download_button("Download Model",data=pickle.dumps(model_file),file_name="model.pkl", )
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
                    samples_split=st.number_input("Min_samples_split",3,20 ,help="Step up is 1")
    # ***********************************************Creating n_estimator Parameter***********************************
                    # st.title("n_estimator")
                    n_estimators=st.number_input("N_estimator",100,1000 ,help="Step up is 5")
                    # submit = st.form_submit_button("Submit")
# ***********************************************Clicking On submit button Parameter******************************
                    btn=st.button("Model Train")
                    if btn:
                        parameter_list.append(input_col)
                        parameter_list.append(criterion)
                        parameter_list.append(maxfeature)
                        parameter_list.append(samples_split)
                        parameter_list.append(n_estimators)
                        st.write(type(parameter_list))
                        st.write(type(parameter_list))
                       
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
           """It Genarte an interface for ExtraTreeClassifier """
           with st.form("Form1"):
                parameter_list=[]
                file = st.file_uploader("Upload an image",type=["csv","xlsx"])
                prediction_column=st.text_input("Enter predictiion")
                st.write(prediction_column)
                input_col=st.slider ("Number of input",3,1000)
                # criterion=["criterion"]
                # max_depth_list=["max_depth"]
                # # n_estimator_list=["n_estimator"]
                # max_features=["max_features"]
                max_features=st.selectbox("Choose a max_features",["auto", "sqrt", "log2"])
                # min_samples_split_list=["min_samples_split"]
                criterion=st.selectbox("Choose a Criterion",["gini","entropy","log_loss"])
                max_depth=st.number_input("Input Max Depth " ,min_value=1,max_value=1000)
                min_samples_split=st.number_input("Input Sample Split ",min_value=3,max_value=50)
                submit = st.form_submit_button("Submit")
                if submit:
                    if file is None:
                        st.warning("Please Upload a file")


                    data=self.data_extract(file)
                    maxcol=data.shape[1]-2
                    x=data.drop(columns=prediction_column)
                    le=LabelEncoder()
                    y=le.fit_transform(data[prediction_column])
                    # criterion=criterion+criterion_list
                    # max_features=max_features+max_features_list
                    # for i in range(1,n_estimator,5):
                    #       n_estimator_list.append(i)
                    # for i in range(1,min_samples_split,5):
                    #       min_samples_split_list.append(i)
                    # self.creating_hyper_paramerter(parms,min_samples_split_list)
                    # # self.creating_hyper_paramerter(parms,n_estimator_list)
                    # self.creating_hyper_paramerter(parms,max_features)
                          
                          
                    # self.creating_hyper_paramerter(parms,criterion)
                    # self.creating_hyper_paramerter(parms,criterion)
                    parameter_list.append(input_col)
                    parameter_list.append(max_features)
                    parameter_list.append(criterion)
                    parameter_list.append(max_depth)
                    # parameter_list.append(max_features)
                    parameter_list.append(min_samples_split)
                    parameter_list=tuple(parameter_list)
                    if self.parameter_checkup(input_col,maxcol,x,y):
                          
                      model_file=self.download_extratree_classifier (x,y,parameter_list)
                      try:
                              
                            btn=st.download_button("Download Model",data=pickle.dumps(model_file),file_name="model.pkl", )
                      except Exception as e:
        
                              pass
                       
                        # self.extra_tree_classifier(x,y,1,parameter_list)
    def download_interface_randomforest_classifier(self):
           """It Genarte an interface for  download Random Forest Classifier """
           with st.form("Form1"):
                parameter_list=[]
                # criterion=["criterion"]
                # max_depth_list=["max_depth"]
                # n_estimator_list=["n_estimator"]
                # max_features=["max_features"]
                # min_samples_split_list=["min_samples_split"]
                file = st.file_uploader("Upload an image",type=["csv","xlsx"])
                prediction_column=st.text_input("Enter predictiion")
                st.write(prediction_column)
                input_col=st.slider ("Number of input",3,1000)
                # criterion_list=st.multi
                criterion=st.selectbox("Choose a Criterion",["gini","entropy","log_loss"])
                max_features=st.selectbox("Choose a max_features",["sqrt","log2",None])

                max_depth=st.number_input("Input Max Depth " ,min_value=1,max_value=1000)
                min_samples_split=st.number_input("Input Sample Split ",min_value=1,max_value=50)
                n_estimator=st.number_input("input n Estimator",min_value=100,max_value=5000)
                submit = st.form_submit_button("Submit")
                if submit:
                    if file is None:
                        st.warning("Please Upload a file")
                    data=self.data_extract(file)
                    maxcol=data.shape[1]-2
                    x=data.drop(columns=prediction_column)
                    le=LabelEncoder()
                    # criterion=criterion+criterion_list
                    # max_features=max_features+max_features_list
                    # for i in range(1,n_estimator,5):
                    #       n_estimator_list.append(i)
                    # for i in range(1,min_samples_split,5):
                    #       min_samples_split_list.append(i)
                    # self.creating_hyper_paramerter(parms,min_samples_split_list)
                    # self.creating_hyper_paramerter(parms,n_estimator_list)
                    # self.creating_hyper_paramerter(parms,max_features)
                          
                          
                    # self.creating_hyper_paramerter(parms,criterion)
                    # self.creating_hyper_paramerter(parms,criterion)


                    y=le.fit_transform(data[prediction_column])
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