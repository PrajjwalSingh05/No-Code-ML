
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

# ****************************************Self Made file*****************************
from default_model import *
from all_model import *
from downloadModel import * 



class all_function(CreatedModels,Defaul_model,download_model):
        
    def introduction(self):
        """it is a Function to give a personal deatils about the creator """
        # st.image('vg.gif', width=None)
        st.markdown("""
            
        Name : Prajjwal Singh
        \nQualification : Bachelor ofComputer Application(BCA)
        \nStream : Computer Science
        \nUniversity : University of Lucknow
        \nLocation : Lucknow, INDIA
        \nThis Project Perfrom Object dection 
            
            - The Libraries I used in Project are:
                Matplotlib Explore here
                Sklearn Explore Here
                Streamlit Explore here
                Pandas 
                Tensorflow

            - Their Following Tasks are Implemented in the Project:
                Data Preparation and Cleaning
                Model design
                Best Feature Selectio 
                References and Future Work
                
        """)
    def parameter_checkup(self,input_col,maxcol,x,y):

        """This is a function to check that input iternation does not exceeed availble column in filr """
        if input_col<=maxcol:
                    st.write("insidce paramter function")
                    # model_search(x,y,3,input_col)
                    return 1
        else:
                    st.warning(f"Number of column in file is less than{input_col}{maxcol} ")
    def data_extract(self,file):
        """Function to read the instered file of document type CSV and Excel"""
        try:

                    data=pd.read_csv(file)
                    st.write("Your Original Dataset is -:")
                    st.write(data)
        except Exception as e:
                    print(e)
                    data=pd.read_excel(file)  
                    st.write("Your Original Dataset is -:")
                    st.write(data)
        return data
    def slidebar(self,options):
            sidebar = st.sidebar

            sidebar.title('Select Model ')

            selOptionmodel = sidebar.selectbox("Select an Option",options)
            return selOptionmodel
    # def formmaker(self):
    #        with st.form("Form1"):
            
    #             file = st.file_uploader("Upload an image",type=["csv","xlsx"])
    #             prediction_column=st.text_input("Enter predictiion")
    #             st.write(prediction_column)
    #             input_col=st.slider ("Number of input",3,1000)
                
    #             submit = st.form_submit_button("Submit")
    #             if submit:
    #                 if file is None:
    #                     st.warning("Please Upload a file")


    #                 data=self.data_extract(file)
    #                 return data
    #                 # st.write(f"The Maximum col are :{maxcol}"
                
    def interface_randomforest_classifier(self):
           st.markdown("#### Using Random Forest Classifier")
           """It Genarte an interface for Random Forest Classifier """
           with st.form("RandomForestClassfierForm"):
                parms={}
# *********************************************************Creating Parameter List ***********************************************
                criterion=["criterion"]
                max_depth_list=["max_depth"]
                n_estimator_list=["n_estimators"]
                max_features=["max_features"]
                min_samples_split_list=["min_samples_split"]
# *********************************************************File Uploader and interface***********************************************
                file = st.file_uploader("Upload Your File ",type=["csv","xlsx"])
                prediction_column=st.text_input("Enter predictiion Column")
                # st.write(prediction_column)
                input_col=st.slider ("Number of input Column",3,1000)
                criterion_list=st.multiselect("Choose a Criterion",["gini","entropy","log_loss"])
                max_features_list=st.multiselect("Choose a max_features",["sqrt","log2",None])

                max_depth=st.slider("Input Max Depth " ,min_value=2,max_value=1000)
                min_samples_split=st.slider("Input Sample Split ",min_value=3,max_value=50)
                n_estimator=st.slider("Input n Estimator",min_value=105,max_value=5000)
                submit = st.form_submit_button("Submit")
                if submit:
                    if file is None:
                        st.warning("Please Upload a file")
                    data=self.data_extract(file)
                    maxcol=data.shape[1]-2
                    x=data.drop(columns=prediction_column)
                    le=LabelEncoder()
                    y=le.fit_transform(data[prediction_column])
                
# *********************************************************Updating Parameter List ***********************************************
                    criterion=criterion+criterion_list
                    max_features=max_features+max_features_list
                    for i in range(1,n_estimator,5):
                          n_estimator_list.append(i)
                    for i in range(1,min_samples_split,1):
                          min_samples_split_list.append(i)

# *********************************************************Creating Params ***********************************************
                    self.creating_hyper_paramerter(parms,min_samples_split_list)
                    self.creating_hyper_paramerter(parms,n_estimator_list)
                    self.creating_hyper_paramerter(parms,max_features)
                    self.creating_hyper_paramerter(parms,criterion)
                    self.creating_hyper_paramerter(parms,criterion)

                    if self.parameter_checkup(input_col,maxcol,x,y):
                         
                         self.random_forest_classifier(x,y,1,input_col,parms)
    def interface_ExtraTreeClassifier(self):
           """It Genarte an interface for ExtraTreeClassifier """
           st.markdown("#### Using ExtraTreeClassifier ")
           with st.form("ExtraTreeClassifierForm"):
                parms={}
                file = st.file_uploader("Upload Your File ",type=["csv","xlsx"])
                prediction_column=st.text_input("Enter predictiion Column ")
                st.write(prediction_column)
                input_col=st.slider ("Number of input",3,1000)
# *********************************************************Creating Parameter List ***********************************************
                criterion=["criterion"]
                max_depth_list=["max_depth"]
                max_features=["max_features"]
                max_features_list=st.multiselect("Choose a max_features",["auto", "sqrt", "log2"])
                min_samples_split_list=["min_samples_split"]
                criterion_list=st.multiselect("Choose a Criterion",["gini","entropy","log_loss"])
                max_depth=st.slider("Input Max Depth " ,min_value=1,max_value=1000)
                min_samples_split=st.slider("Input Sample Split ",min_value=3,max_value=50)
                submit = st.form_submit_button("Submit")
                if submit:
                    if file is None:
                        st.warning("Please Upload a file")
                    data=self.data_extract(file)
                    maxcol=data.shape[1]-2
                    x=data.drop(columns=prediction_column)
                    le=LabelEncoder()
                    y=le.fit_transform(data[prediction_column])

# *********************************************************Updating Parameter List ***********************************************
                    criterion=criterion+criterion_list
                    max_features=max_features+max_features_list
                    for i in range(1,min_samples_split,1):
                          min_samples_split_list.append(i)

# *********************************************************Creating Params ***********************************************
                    self.creating_hyper_paramerter(parms,min_samples_split_list)
                    self.creating_hyper_paramerter(parms,max_features)
                    self.creating_hyper_paramerter(parms,criterion)

# *********************************************************Checking input < predcition***********************************************
                    if self.parameter_checkup(input_col,maxcol,x,y):
                       
                        self.extra_tree_classifier(x,y,1,input_col,parms)
    def creating_hyper_paramerter(self,parms,column):
                try:
                    parms["classifier__"+column[0]]=column[1:]
                    return parms
                except Exception as e:
                      print(Exception)
                               
    def interface_ExtraTreeRegressor(self):
           """It Genarte an interface for ExtraTreeRegressor """
           st.markdown("#### Using Extra Tree Regressor ")
           with st.form("ExtraTreeRegressor"):
                parms={}
# *******************************************File Uploader**************************************************
                file = st.file_uploader("Upload an image",type=["csv","xlsx"])
                prediction_column=st.text_input("Enter Predictiion Column")
                st.write(prediction_column)
                input_col=st.slider ("Number of input Column",3,1000)

# *******************************************Creating Parameter**************************************************

                criterion=["criterion"]
                splitter=["splitter"]
                min_samples_split=["min_samples_split"]
                # st.title("Criterion")
                criterion_list=st.multiselect("Choose a Criterion",["squared_error","friedman_mse"])
                # st.title("Splitter")
                splitter_list=st.multiselect("Choose a splitter",["random","best"])
                min_samples=st.slider("Range Of minSample",min_value=3,max_value=50)
                submit = st.form_submit_button("Submit")
                if submit:
                    
                    if file is None:
                        st.warning("Please Upload a file")
                    data=self.data_extract(file)
                    maxcol=data.shape[1]-2
                    # st.write("Inside extretree")
                    x=data.drop(columns=prediction_column)
                    y=data[prediction_column]
# **************************************Updating Paramter list**************************
                    criterion=criterion+criterion_list
                    splitter=splitter+splitter_list
                    for i in range(1,min_samples):
                          min_samples_split.append(i)
                    
                    self.creating_hyper_paramerter(parms,criterion)
                    self.creating_hyper_paramerter(parms,splitter)
                    self.creating_hyper_paramerter(parms,min_samples_split)
                    if self.parameter_checkup(input_col,maxcol,x,y):
                         self.Extra_tree_Regressor(x,y,1,input_col,parms)
    def interface_random_forest_regression(self):
           
           """It Genarte an interface for random_forest_regression """
           st.markdown("#### Using Random Forest Regression  ")
           with st.form("Randomforestregression"):
            
                parms={}
# ***********************************************Creating list***************************************************************
                criterion=["criterion"]
                n_estimator_list=["n_estimators"]
                max_features=["max_features"]
                min_samples_split=["min_samples_split"]
    
# ***********************************************Uploading File***************************************************************
                file = st.file_uploader("Upload an image",type=["csv","xlsx"])
                prediction_column=st.text_input("Enter predictiion Column")
                st.write(prediction_column)
                input_col=st.slider ("Number of Input Column",3,1000)
# ***********************************************Creating Ctiterion Parameter***************************************************************
                
                # st.title("criterion")
                # st.markdown("### Criterion")
                criterion_list=st.multiselect("Choose Criterion",["absolute_error","squared_error"])
               
# ***********************************************Creating Max  Feature Parameter***************************************************************
                # st.title("max_features")
                maxfeature_list=st.multiselect("Choose Max Feature",["sqrt","log2","None"])
           
# ***********************************************Creating Max  Min_Sample_split Parameter***************************************************************
                
                
                # st.title("min_samples_split")
                samples_split=st.slider ("Number of Min Sample Split",min_value=3,max_value=20 ,help="Step up is 1")
# ***********************************************Creating n_estimator Parameter***************************************************************
                # st.title("n_estimator")
                n_estimators=st.slider ("Number of N_estimator",min_value=105,max_value=1000 ,help="Step up is 5")
                
                submit = st.form_submit_button("Submit")
# ***********************************************Clicking On submit button Parameter***************************************************************
                if submit:
                    if file is None:
                        st.warning("Please Upload a file")


                    data=self.data_extract(file)
                    maxcol=data.shape[1]-2
                    st.write("Inside extretree")
                    x=data.drop(columns=prediction_column)
                    y=data[prediction_column]
# ***********************************************Updating criterion list***************************************************************
                 
                    criterion=criterion+criterion_list
                    st.write(criterion)
                    parms=self.creating_hyper_paramerter(parms,criterion)
                    # st.write("New Cretertion are")
# ***********************************************Updating MAx Feaute  list***************************************************************
                 
                    max_features=max_features+maxfeature_list
                    parms=self.creating_hyper_paramerter(parms,max_features)
# ***********************************************Updating N estimator list***************************************************************
                    for i in range(100,n_estimators,5):
                        n_estimator_list.append(i)
                    parms=self.creating_hyper_paramerter(parms,n_estimator_list)
# ***********************************************Updating Min MAx Sample list***************************************************************
                    for i in range(1,samples_split):
                        min_samples_split.append(i)
                    parms=self.creating_hyper_paramerter(parms,min_samples_split)
                    st.write(parms)
                   
         
                    if self.parameter_checkup(input_col,maxcol,x,y):
             
                         self.random_forest_regression(x,y,1,input_col,parms)
    def interfacr_deafult_classifier(self):
          st.markdown("## Default Platform For Classification  Probelm")
          with st.form("Default Classifier"):               
# ***********************************************Uploading File***************************************************************
                file = st.file_uploader("Upload Your File ",type=["csv","xlsx"])
                prediction_column=st.text_input("Enter predictiion Column")
                st.write(prediction_column)
                input_col=st.slider ("Number of input Column ",3,1000)
# ***********************************************Creating Ctiterion Parameter***************************************************************
                model=st.selectbox("choose your model",("RandomForestClassifier","ExtraTreeClassifier"))
                self.model_name=model
                submit = st.form_submit_button("Submit")
#  **********************************Clicking On submit button Parameter***************************************************************
                if submit:
                    if file is None:
                        st.warning("Please Upload a file")
                    data=self.data_extract(file)
                    maxcol=data.shape[1]-2
                    # st.write("Inside extretree")
                    
                    x=data.drop(columns=prediction_column)
                    le=LabelEncoder()
                    y=le.fit_transform(data[prediction_column])
                    if self.parameter_checkup(input_col,maxcol,x,y):
                        self.default_model_classifier(x,y,1,input_col,model)
    def interface_deafult_regressor(self):
            #    st.title("Default Platform For Regression Probelm")
               st.markdown("## Default Platform For Regression Probelm")
               with st.form("Default Regression"):               
# ***********************************************Uploading File***************************************************************
                file = st.file_uploader("Upload Your File ",type=["csv","xlsx"])
                prediction_column=st.text_input("Enter predictiion Column")
                # st.write(prediction_column)
                input_col=st.slider ("Number of Column to be used",3,1000)
# ***********************************************Creating Ctiterion Parameter***************************************************************
                model=st.selectbox("Choose your model",("RandomForestRegressor","ExtraTreeRegressor"))
                self.model_name=model
                submit = st.form_submit_button("Submit")
# ***********************************************Clicking On submit button Parameter***************************************************************
                if submit:
                    if file is None:
                        st.warning("Please Upload a file")
                    data=self.data_extract(file)
                    maxcol=data.shape[1]-2
                    # st.write("Inside extretree")
                    x=data.drop(columns=prediction_column)
                    y=data[prediction_column]
                    if self.parameter_checkup(input_col,maxcol,x,y):
                        self.default_model_regression(x,y,1,input_col,model)
# ***********************************************Updating criterion list***************************************************************

    def execute(self):
        data_options = ["None","Regression","classifier",]

        sidebar = st.sidebar
# **********************************Download slidebar *****************************************************************
        sidebar.title("Download Your Custom Model")
        download_option=["None","Random Forest Regressor","Extra Tree Regressor","Random Forest Classfier","Extra Tree Classfier"]
        customdownload=sidebar.selectbox("Choose Your Model",download_option)

        if customdownload==download_option[1]:
             
              self.download_interface_random_forest_regression()
           
        elif customdownload==download_option[2]:
            #   self.interface_custom_download_model()
              self.download_interface_ExtraTreeRegressor()
        elif customdownload==download_option[3]:
              self.download_interface_randomforest_classifier()
        elif customdownload==download_option[4]:
              self.download_interface_ExtraTreeClassifier()
        else:
              pass 
# ***********************************************Regression and classifier sliderbar ************************************************
        sidebar.title('Select Data Type  ')

        selOptionmodel = sidebar.selectbox("Select an Option", data_options)
# ****************************************************************Inside Regression slideBar***********************************
        if(selOptionmodel==data_options[1]):
            model_options = ["None","Random Forest Reggression","Extra Tree Regressor"]
            seloptionmodelregression=self.slidebar(model_options)
            if seloptionmodelregression==model_options[0]:
                
                  self.interface_deafult_regressor()

            elif seloptionmodelregression==model_options[1]:
                    self.interface_random_forest_regression()
            elif seloptionmodelregression==model_options[2]:
                    self.interface_ExtraTreeRegressor()
        elif(selOptionmodel==data_options[2]):
               model_options = ["None ","Random Forest Classifier","Extra Tree Classifier"]
          
               selOptionmodel=self.slidebar(model_options)
    # ********************************************Inside Classfication Slide Bar********************************************
               if selOptionmodel==model_options[0]:
                     self.interfacr_deafult_classifier()
               elif selOptionmodel==model_options[1]:
                    self.interface_randomforest_classifier()
               elif selOptionmodel==model_options[2]:
                    self.interface_ExtraTreeClassifier()
                    # st.write("Inside reandom forest classifer")
