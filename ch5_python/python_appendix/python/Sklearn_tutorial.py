# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# # Sklearn Modeling Procedure
# ### Scikit-learn requries Python >=3.6
# ## Step 1: Data import 
# ### You can load the data from sklearn.datasets API. The dataset loader is for small and Toy dataset, while the dataset fetcher is for huge and Real World dataset. Here are the examples to load small data and fetch large data. 

# %%
from sklearn import datasets
iris = datasets.load_iris()  
faces = datasets.fetch_olivetti_faces()

# %% [markdown]
# ### We can also have a function to download the data, this is to keep tack of the up-to-date data and allows for automation of dara fetching process. Below presented the code to fetch zip file from web and extract all .csv files to your working directory.

# %%
import os
from zipfile import ZipFile
import urllib

Root_Download = './master/..'
url_download = Root_Download +'Data.zip'
Data_Path = os.path.join ('Data')

def fetch_Data (url = url_download, Path =Data_Path):
    os.makedirs(Path, exist_ok=True)
    zip_path = os.path.join (Path, 'Dataset.zip')
    filepath, _ = urllib.request.urlretrieve(url, zip_path)
    with zipfile.ZipFile(filepath, 'r') as zip:
        zip.extractall(Path)

# %% [markdown]
# #### Once we have dataset downloaded, let us write a function to load our data.

# %%
import pandas as pd 

def data_loader (data_path, data):
    csv_path = os.path.join (data_path, data)
    return pd.read_csv(csv_path)

# %% [markdown]
# #### Next I will walk you through preprocessing using one of our dataset, COVID_19.csv, and use head(10) to see the first 10 rows in the dataset. 

# %%
os.chdir('/Users/rachelzeng/full-stack-data-scientist/ch5_python')

import pandas as pd
Data_Path =os.path.join ('Data')
data_name = 'COVID_19_preprocess.csv'
COVID = data_loader (data_path=Data_Path, data= data_name)
COVID_copy = COVID.copy()
COVID.head(10)

#Suppose we are interested in predicting whether a suspected patient is infected with COVID 19. In order to make a robust model, we need to ensure our model can perform well on the new data. Assume we have a set of data, then this set should be divided into training and testing data, training data is what need to be fed into and testing data is what the model has not seen, and needed to be tested. We will talk more about it in Appendix for Machine Learning

target_name = 'COVID?'
target = COVID.pop (target_name)
# Split Training and Testing data
from sklearn.model_selection import train_test_split
Train_Data, Test_Data, target_train, target_test = train_test_split(COVID, target, test_size=0.2, random_state =1)


# Deal with Missing Values
from sklearn.impute import SimpleImputer
Null_col = COVID.columns[ COVID.isnull().any()]
print(COVID[Null_col].isnull().sum())

print(COVID.loc[pd.isnull(COVID).any(axis = 1),:].head())



import numpy as np
from sklearn.impute import SimpleImputer

Mean_imp = SimpleImputer(missing_values= np.nan, strategy = 'mean')
COVID['tmp'] = Mean_imp.fit_transform (COVID['tmp'].values.reshape(-1,1)) 
# .reshape(1,-1) indicate transform to array with 1 row, and do not care about how many column. 
COVID[COVID['tmp'].isnull()] 

# Find index of rows with missing values
row_null = COVID.loc[pd.isnull(COVID).any(axis = 1),:].index.values
COVID = COVID.drop(row_null)


# Attributes data type
COVID[['Travel?', 'CC', 'DC']] = COVID[['Travel?', 'CC', 'DC']].astype('bool')
COVID[['RS',  'Sex']] = COVID[['RS', 'Sex']].astype('category')
Bool_attribute = list(list(COVID.select_dtypes(include=['bool']).columns))
Num_attribute = list(COVID.select_dtypes(include=['number']).columns)
Cat_attribute = list(COVID.select_dtypes(include=['category']).columns)
Bool_attribute, Num_attribute,Cat_attribute

# Label Encoder
#Method 1: Manually
from sklearn.preprocessing import LabelEncoder

label_enc = LabelEncoder()
label_encoded = label_enc.fit_transform(COVID['RS'])

RS_col = pd.get_dummies(COVID['RS'], prefix='RS')
COVID_1 = pd.concat([COVID, RS_col],axis=1)
COVID_1.drop(['RS'], axis =1, inplace = True)
print(COVID_1.head() )


# Method 2：Using Sklearn Preprocessing
Resident_cat = COVID[['RS']]

from sklearn.preprocessing import OneHotEncoder
OH_enc = OneHotEncoder()
label = [['Canadian' , 1], ['Non-Canadian',2], ['Permanent Resident', 3], ['Refugee',4]]
print(OH_enc.fit(label))
print(OH_enc.categories_)

Residency_encoded = OH_enc.fit_transform(Resident_cat)
print(Residency_encoded.toarray())

# Let us deal with missing data in Train and Test data
null_train = Train_Data.loc[pd.isnull(Train_Data).any(axis = 1),:].index.values
Train_Data = Train_Data.drop(null_train)
null_test = Test_Data.loc[pd.isnull(Test_Data).any(axis = 1),:].index.values
Test_Data = Test_Data.drop(null_test)


target_train= target_train.drop(null_train)
target_test = target_test.drop(null_test)

#Transformation Pipeline
# As you can see there are still few columns needed to be executed. Here, a very useful technique in sklearn is the Pipeline class. In general, Pipeline groups a set of activaties together to a perform task, this can manage and fixed the sequence of all steps, in which makes it easy to reuse paramter sets on new data. In the data preprcoessing, let us build a pipeline with sequences of "fit and transform", to deal with each distinct data type.

#custome transformer
from sklearn.pipeline import Pipeline
class drop_na(object):
    def __init__(self,attribute_name=True):
        self.attribute_name = attribute_name
  
    def transform(self, X):
        X_cp = X.copy()
        #null = np.argwhere(np.isnan(X.cp))
        X_cp = X_cp[~np.isnan(X_cp).any(axis=1)]
        return X_cp 
     
    def fit (self, X_cp, y=None):
        return self

class target_encode(object):
    def __init__(self, attribute_name=True):
        self.attribute_name = attribute_name
  
    def transform(self, X):
        X_cp = X.copy()
        X_cp = np.where (X_cp == True, 1, X_cp)
        return X_cp
    def fit (self, X_cp, y=None):
        return self

# Pipeline for target
def target_transformation (target):
    cat_pip = Pipeline([
    ('imputer_na', drop_na()), # Handle missing data 
    ('imputer_Bool', target_encode())])

    target = cat_pip.transform(np.asarray(target).reshape(1,-1))[0]
    return target

target_train = target_transformation (target_train)
target_test = target_transformation (target_test)

target_test
# Pipeline for numerical attributes
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer # helps perform different transformations for different columns
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer


def Attribute_pip (Data):
    Num_attribute = list(Data.select_dtypes(include=['number']).columns)
    Obj_attribute = list(Data.select_dtypes(include=['category','object','bool' ]).columns) 

    numerical_pip = Pipeline([
    ('imputer', SimpleImputer(strategy= 'mean')), # Handle missing data 
    ('std_scaler', StandardScaler())])
    Full_pip = ColumnTransformer([
    ('num_age_temp' , numerical_pip, Num_attribute),
    ('cat_Bool', OneHotEncoder(), Obj_attribute)])

    transformed = Full_pip.fit_transform(Data)
    return transformed



list(Train_Data.select_dtypes(include=['object','bool' ]).columns)

# Full pipeline to deal with both numerical and categorical attributes

Trained_transformed = Attribute_pip (Train_Data)
Tested_transformed = Attribute_pip (Test_Data)

# Modeling

from sklearn.linear_model import LogisticRegression
from sklearn import neighbors 
from sklearn.metrics import classification_report

classification = LogisticRegression(solver = 'lbfgs' , multi_class= 'multinomial')
classification.fit(Trained_transformed, target_train)
#predict = classification.predict(Tested_transformed)
#print(classification_report(target_test, predict))


# Save Model
#os.chdir('/Users/rachelzeng/dsbook/Saved_models')
import joblib
#joblib.dump (classification, 'sk_classification.pkl')

# Reuse Model 
#Reuse_clf = joblib.load ( 'sk_classification.pkl' )

# print the score:
#score = Reuse_clf.score(Tested_transformed,target_test)
#print("The test score is {0:.2f} %".format(100 * score))

#Not suprisingly, using Logistic Regression with all attribute columns resulted in poor rate of predictive score. We will revisit this problem in Machine Learning section with additional knowledge on the subject of classification and random sampling, such as feature selections, bootstraping and implement random forest classifier to improve the prediction accuracy.
