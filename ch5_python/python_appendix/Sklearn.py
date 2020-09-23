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
os.chdir('/Users/rachelzeng/dsbook')

import pandas as pd
Data_Path =os.path.join ('Data')
data_name = 'COVID_19.csv'
COVID = data_loader (data_path=Data_Path, data= data_name)
COVID_copy = COVID.copy()
COVID.head(10)

# %% [markdown]
# #### This is a small dataset features 200 rows and 9 columns, notice that it is a simulated data for better ilustrate, thus no real meaning apply to the result.
# 
# #### Suppose that this data is collected from COVID 19 Assessment Center, it listed 200 suspected patient waiting for testing.
# 
# #### Next, there are some useful methods that provide quick desription of the data. dataset.info() will show you number and name of the columns with associate data type.
# 
# #### You will find that this dataset contain 200 instances, 9 attributes.

# %%
print(COVID.info())

# %% [markdown]
# #### The describe() method is showing the summary of statistics of each attributes, such as mean, standard deviation, min, and max. Knowing them you will have a general idea of how does the distirbution of each attribute look like. Notice that when you use describe() method, it only return the summary of statistics for Age and tmp attributes, it is becuase describe() only works for numerical data.

# %%
COVID.describe()

# %% [markdown]
# ## Wait for a minute, so what Problem, exactly, are we solving here? For a given project, the very first step is actually to understand the purpose of the study and what is optimal result return to the user in assisting them to make decision?
# 
# ### Suppose we are interested in predicting whether a suspected patient is infected with COVID 19. In order to make a robust model, we need to ensure our model can perform well on the new data. Assume we have a set of data, then this set should be divided into training and testing data, training data is what need to be fed into and testing data is what the model has not seen, and needed to be tested. We will talk more about it in Appendix for Machine Learning.
# 
# 

# %%
target_name = 'COVID_19?'
target = COVID.pop (target_name)
COVID.head()


# %%
from sklearn.model_selection import train_test_split
Train_Data, Test_Data, target_train, target_test = train_test_split(COVID, target, test_size=0.2, random_state =1)
Train_Data.info()

# %% [markdown]
# ### We simply use train_test_split function from Sklearn to do random spliting. test_size = 0.2 indicates the proportion of testing samples, ie. if we have 100 samples, then data set has 20 samples. random_state allows you to set random gnerator seed, ie. this will garurantee to get same result each time you run the experiment. (otherwise put 0 or nothing.) 
# 
# ### Using the random smapling like the one above is fine if we have a large dataset, but since we are working with only 200 samples, we need to ensure that the samples selected for training data is representative of the whole. A better way is to conduct stratified sampling. 
# 
# ### Stratify is a way to maintain the distirbution of pre-split classes. For example, suppose we have 100 suspected patients, 80 suspected patients are infected, 20 are not. If we need 75 training samples, in order to preserve the distribution, we must have 60 infected and 15 uninfected in the sample. (Similar to testing data) 
# 
# ### Let us see how to do this using sklearn.

# %%
from sklearn.model_selection import train_test_split
Train_Data, Test_Data, target_train, target_test  = train_test_split(COVID,target,test_size=0.2, random_state=1,stratify=target)
Train_Data.info()

# %% [markdown]
# ## Deal with Missing Values
# ### The very first step is to clean our data, if we use the Sklearn datasets like Iris, Boston housing price etc., these datasets are high quality with no missing values. However, often time we need to deal with dirty data, it contain missing values, noise; Specially when we are dealing with clinical dataset, inaccurate data due to measurement error, and needs to deal with variety of data types and sources. Low quality of dara will cause us great trouble in the later stage, thus let us adopt sklearn.preprocessing methods to improve the quality of our data.

# %%
from sklearn.impute import SimpleImputer
Null_col = COVID.columns[ COVID.isnull().any()]
COVID[Null_col].isnull().sum()

# %%
COVID.loc[pd.isnull(COVID).any(axis = 1),:].head()

# %% [markdown]
# #### Here is a list of index of rows with missing values

# %%
Null_row = COVID.loc[pd.isnull(COVID).any(axis = 1),:].index.values # return index of rows with NaN in dataframe
print(Null_row) 

# %% [markdown]
# ### There are severeal ways to deal with missing values. The simplest way is to remove the entire sample or variable from the data depending how many missing values in a sample or a variable(attribute). If it contians more than a certian percentage of missing value, then delete the missing entries or the entire column. In our case, we do not have too much missing value for the attribues, so we can delete the missing entries. 
# %% [markdown]
# ####  However, discard entire rows and/or columns may comes at the expense of potentially valuable data, and left with fewer samples. A better way to deal with numerical attribute, like tmp, is by inferring them from the data. \textit{Sklearn.SimpleImputer} provides strategies to estimate the missing values, such as fill the NaN with mean, medium or most frequent of the column where missing value is located.

# %%
COVID['tmp']=COVID['tmp'].apply(pd.to_numeric, errors='coerce')
import numpy as np
from sklearn.impute import SimpleImputer
Mean_imp = SimpleImputer(missing_values= np.nan, strategy = 'mean')
COVID['tmp'] = Mean_imp.fit_transform (COVID['tmp'].values.reshape(-1,1)) 
# .reshape(1,-1) indicate transform to array with 1 row, and do not care about how many column. 
COVID[COVID['tmp'].isnull()] 

# %% [markdown]
# #### We replece NaN values in tmp with the mean, and drop rows which contain missing values.
# 

# %%
row_null = COVID.loc[pd.isnull(COVID).any(axis = 1),:].index.values
COVID = COVID.drop(row_null)

# %%
COVID[['Travel?', 'CC', 'DC']] = COVID[['Travel?', 'CC', 'DC']].astype('bool')
COVID[['RS','Sex']] = COVID[['RS','Sex']].astype('category')
Bool_attribute = list(list(COVID.select_dtypes(include=['bool']).columns))
Num_attribute = list(COVID.select_dtypes(include=['number']).columns)
Cat_attribute = list(COVID.select_dtypes(include=['category']).columns)
print(Bool_attribute, Num_attribute,Cat_attribute)

# %% [markdown]
# ### In order to run numerical analyses, we will need to convert values from categorical to numerical by assigning muerical codes to them. As we can see column: RS, Education level and Sex are categorical data type and the column: Travel? CC, Dry Vough and COVID_19? are boolean, which means it only returns True or False. 
# 
# ### There are two ways to encode numerical data to numbers, LabelEncoder() or OneHotEncoder(). So what are the differences? 
# 
# ##  LabelEncoder() vs. OneHotEncoder()
# 

from random import seed
import random
seed(1)
Status = ['Canadian','Non-Canadain', 'Permanent Resident','Refugee']
Random_residency = random.choices(Status, k=197)
print(Random_residency[:8])

from sklearn.preprocessing import LabelEncoder
label_enc = LabelEncoder()
label_encoded_res = label_enc.fit_transform(Random_residency)
label_encoded_res

import scipy
from sklearn.preprocessing import OneHotEncoder
OH_enc = OneHotEncoder()
Resi_col = pd.DataFrame(Random_residency)
Resi_encoded = OH_enc.fit_transform(Resi_col).toarray()
print(Resi_encoded)

# %%
from sklearn.preprocessing import LabelEncoder

label_enc = LabelEncoder()
label_encoded = label_enc.fit_transform(COVID['RS'])
label_encoded

# %%
#Method 1:
RS_col = pd.get_dummies(COVID['RS'], prefix='RS')
COVID_1 = pd.concat([COVID, RS_col],axis=1)
COVID_1.drop(['RS'], axis =1, inplace = True)
COVID_1.head()


# %%
# Method 2：Using Sklearn Preprocessing
Resident_cat = COVID[['RS']]
Resident_cat.head(4)


# %%
from sklearn.preprocessing import OneHotEncoder
OH_enc = OneHotEncoder()
label = [['Canadian' , 1], ['Non-Canadian',2], ['Permanent Resident', 3], ['Refugee',4]]
OH_enc.fit(label)



# %%
Residency_encoded = OH_enc.fit_transform(Resident_cat)

# %%
Residency_encoded.toarray()

# %%
null_train = Train_Data.loc[pd.isnull(Train_Data).any(axis = 1),:].index.values
Train_Data = Train_Data.drop(null_train)
null_test = Test_Data.loc[pd.isnull(Test_Data).any(axis = 1),:].index.values
Test_Data = Test_Data.drop(null_test)


target_train= target_train.drop(null_train)
target_test = target_test.drop(null_test)

# %%
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

cat_pip = Pipeline([
    ('imputer_na', drop_na()), 
    ('imputer_Bool', target_encode()),
])

target_train = cat_pip.transform(np.asarray(target_train).reshape(1,-1))
target_test = cat_pip.transform(np.asarray(target_test).reshape(1,-1))
target_train = target_train[0]
target_train

# %% [markdown]
# ## A quick comment on three common data preprocessing tools: 
# ### StandardScaler: The data transofrmed into a standard normal distribution with a mean of 0 and a variance of 1 (Z-score).
# ### RobustScaler: Similar to StandardScaler. Instead of using mean and variance, it uses the median and quartile. Therefore, this will get rid of the outliers directly.
# ### MinMaxScaler: Simple method to shift and scale the data so that the values returned within the range of 0 and 1, we can do this by subract the min value and divide by the the difference between max and min values.
# ### Normalizer: It will first convert the samples into Euclidean distance of 1, the data distribution thus turn into a circle with a radius of 1. It is used when we are interested in only the distance and direction of the data not the value itself.

# %%
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer 
from sklearn.preprocessing import StandardScaler

numerical_pip = Pipeline([
    ('imputer', SimpleImputer(missing_values=np.nan, strategy= 'mean')), # Handle missing data 
    ('std_scaler', StandardScaler()),
])



# %%
COVID = data_loader (data_path=Data_Path, data= data_name)
Full_pip = ColumnTransformer([
    ('num_age_temp' , numerical_pip, Num_attribute),
    ('cat_Bool', OneHotEncoder(),Cat_attribute)
])
COVID_copy = COVID.copy()
COVID_copy['tmp'] = COVID_copy['tmp'].apply(pd.to_numeric, errors = 'coerce')
COVID_copy= COVID_copy.dropna()
target = COVID_copy.pop('COVID_19?')
COVID_transformed = Full_pip.fit_transform(COVID_copy)

Trained_transformed, Tested_transformed,y_train,  y_test  = train_test_split(COVID_transformed, target, test_size=0.2, random_state =1)

# %% [markdown]
# ### In addition, pipeline can be also used to chain classification estimator, but do keep in mind that estimator should comes after all the transfromers.
# ###  Pipline vs. featureUnion: featureUnion is another way to chian transformer objects into one, different from Pipeline, the transformers are applied in parallel while Pipeline execute the transformers in order.
# %% [markdown]
# # Modeling
# 
# ### Finally, enough for the suffer. Here comes the most exicting part, let us build a model to predict whether a suspected patient is infected with COVID 19. Sklearn provides packages for supervised(labeled target) and unsupervised learning(unlabeled target),  these models are called "Estimator" which is used to do prediction or regression. In general the estimators will also have the following functions: 
# ### 1. fit() : fed in attribute data and target to train the model, other parameters like batch size, learning rate etc. On the other hand, fit() in preprcoessing can be used to caculates mean and variance, and accept data trasnformation method. 
# ### 2. predict() : used for data prediction, fed in the input data and it wll return a prediction labels in numpy array. We usually feed the testing set into predict() and then compare with the true test labels.
# ### 3. score() : used to calculate the accuracy of the model, thus it range between 0 and 1. Notice that this is the most basic indicator to evaluate the performance of the model, there are other indicators such as recall rate or precision rate. Under certain circumstances, having one indicator is not enough to judge whether a model is a good model.
# %% [markdown]
# ## fit() and predict() with LogisticRegression.
# ### We also import classification_report, this will return a more comprehensive report of classification.

# %%
from sklearn.linear_model import LogisticRegression
from sklearn import neighbors 
from sklearn.metrics import accuracy_score

classification = LogisticRegression(solver = 'lbfgs' , multi_class= 'multinomial')
classification.fit(Trained_transformed, y_train)
predict = classification.predict(Tested_transformed)
print(accuracy_score(y_test, predict))
# %% [markdown]
# ## Use joblib to save our model
# 
# ### Note: it will save directly to your current working directory. 

# %%
os.chdir('C:\\Users\\rache\\Downloads\\dsbook\\Saved_models')


# %%
import joblib
joblib.dump (classification, 'sk_classification.pkl')


# %%
Reuse_clf = joblib.load ( 'sk_classification.pkl' )

