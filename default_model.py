

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

from all_model import *

class Defaul_model:
    def __init__(self) :
         model_name=""
    def data_preprocessor(self,X,y):
        """Function to Prepocess the data """
        numeric_transformer = Pipeline(
                steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
            )

        categorical_transformer = OneHotEncoder(handle_unknown="ignore")
        preprocessor = ColumnTransformer(
                transformers=[
                    ("numeric", numeric_transformer, X.select_dtypes(np.number).columns.tolist()),
                    ("category", categorical_transformer,X.select_dtypes("object").columns.tolist()),
                ]
            )
        return preprocessor
   
    def default_model(self,X,y,start,end,model):
        Listing=[]
        preprocessor= self.data_preprocessor(X,y)
        xtrain,xtest,ytrain,ytest=train_test_split(X,y,test_size=0.2,random_state=45)
    
        for i in range(start,end):
        # ****************Feature Seclortot**********************************************
                if self.model=="RandomForestRegressor":
                    model_selector = Pipeline(
                                steps=[("preprocessor", preprocessor),
                                ("feature", SelectKBest(f_regression,k=i)),
                                ("classifier", RandomForestRegressor())]
                            )
                    parms={
                "classifier__n_estimators":[100,150,200,250,270,300],
                "classifier__criterion":["squared_error", "absolute_error", "poisson"],
                'classifier__min_samples_split':[1,2,3,4,5,7,8,9,10],
                'classifier__max_features':["sqrt", "log2", None],
                         
                    }
                elif self.model_name=="ExtraTreeRegressor":
                    parms={
                "classifier__splitter":["random", "best"],
                "classifier__max_features":["sqrt", "log2", None],
                'classifier__max_depth':[10,15,20,25,30],
            } 
                    model_selector = Pipeline(
                            steps=[("preprocessor", preprocessor),
                            ("feature", SelectKBest(f_regression,k=i)),
                            ("classifier", ExtraTreeRegressor())]
                        )
                else:
                    print("Not working deafult model")
                    print("Not working deafult model")
                    print("Not working deafult model")
                    print("Not working deafult model")
                    print("Not working deafult model")
                    print("Not working deafult model")
                    print("Not working deafult model")
                    print("Not working deafult model")
                    st.write("No modek qorking")
                    st.write("No modek qorking")
                    st.write("No modek qorking")
                    st.write("No modek qorking")
                    st.write("No modek qorking")
                    st.write("No modek qorking")
                feature_selector = Pipeline(
                    steps=[("preprocessor", preprocessor),
                    ("feature", SelectKBest(f_regression,k=i))])
                feature_selector.fit(xtrain,ytrain)
        # ****************Model Seclortot**********************************************
                # model_selector = Pipeline(
                #     steps=[("preprocessor", preprocessor),
                #     ("feature", SelectKBest(f_regression,k=i)),
                #     ("classifier", RandomForestRegressor())]
                # )
                model_selector.fit(xtrain,ytrain)
        # *********************************Hyper Parametet***********************************
                grid=GridSearchCV(model_selector,parms,cv=4,n_jobs=-1,verbose=3)
                grid.fit(xtrain,ytrain)
                feature=grid.best_params_
                model=grid.best_estimator_
                ypred_model=model.predict(xtest)
                result_model=r2_score(ytest,ypred_model)
                st.markdown("_"*200)
                st.write(f"Accuracy with best parameter{result_model}")
                # print(f"R2score of Result Modelwith best estimator is : {result_model}")
        #****************************Result Generation ******************************
                ypred=model_selector.predict(xtest)
                result=r2_score(ytest,ypred)
                # print(f"R2score is without estimator {result}")
                st.write(f"Accuracy with without parameter{result}")
            
        #*********************************Working on features****************************
                xopt=feature_selector.get_feature_names_out()
                feature_selection=[]
                for x in xopt:
                    feature_selection.append(x.split("__")[1])
            
                st.write("The feature Selection are as follow-:")
                st.write(feature_selection)
                st.write("Hypre Paramerter are as follow-:")
                st.write(feature)
                # print(feature_selection)
                # st.write("************************")
                # print(f"***********************--{i}******************")
                # print("**********new********************")

        # ********************Colecting Data--***********************************************
                Listing.append({
                    "i":i,
                    "Error":result,
                    "Error_model":result_model,
                    "columns":feature_selection,
                    "parameter":feature
                })