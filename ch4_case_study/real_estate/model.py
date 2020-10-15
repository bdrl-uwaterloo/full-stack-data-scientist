from sklearn.ensemble import RandomForestRegressor 
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import LabelEncoder, OneHotEncoder,StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import time
import pickle
import argparse
import os
import sys
import zipfile
from inflect import engine
from sklearn.metrics import explained_variance_score
#https://www.kaggle.com/camnugent/california-housing-prices

# base class for our real estate predictor
# it defines the common variables and functions for its two children:
# real_estate_predictor_trainer 
# and real_estate__predictor_evaluator

class real_estate_predictor():
    def __init__(self):
        # number of trees
        self.tree_num = 3000

        self.min_leaf = 5

        self.rf_model_path = 'model/random_forest.pickle'

        self.linear_model_path = 'model/linear.pickle'

        self.train_data = 'data/train_set.csv'

        self.test_data = 'data/test_set.csv'

        self.onehotencoder_path =  'model/onehotencoder.pickle'

        self.scalar_path =  'model/scalar.pickle'

        self.zip_file_path = 'data/california-housing-prices.zip'

        self.train = False

        try:
            un_zip = zipfile.ZipFile(self.zip_file_path, 'r')
            for item in un_zip.namelist():
                if item =='housing.csv':
                    un_zip.extract(item, 'data/')
        
            df = pd.read_csv('data/housing.csv', index_col=False)
            train, test = train_test_split(df, test_size=0.05,shuffle=True,random_state=2020)
            #print(train.info())
            train.to_csv(self.train_data, index = None)
            test.to_csv(self.test_data, index = None )
        except IOError:
            print("Zip file not found")    


    def extract_feature_for_housing(self, input_path):
        df = pd.read_csv(input_path,index_col=False)
        data = df.copy()
        y = data.pop('median_house_value')
        X= data # features
        bedroom_per_house = X['total_bedrooms']/X['households']
        member_per_house = X['population']/X['households']
        room_per_house = X['total_rooms']/X['households']
        X['bedroom_per_house'] =bedroom_per_house
        X['member_per_house'] = member_per_house
        X['room_per_house'] =room_per_house
        X= X.drop(['total_bedrooms','total_rooms', 'households','population'], axis=1)
        Num_attribute = list(X.select_dtypes(include=['number']).columns)
        if self.train ==True:
            
            enc = OneHotEncoder().fit(X[['ocean_proximity']])
            ocean_cat = enc.transform(X[['ocean_proximity']])
            ocean= ocean_cat.toarray()

            numerical_pip = Pipeline([
                ('imputer', SimpleImputer(strategy= 'mean')),#missing data
                ('std_scaler', StandardScaler())]) #standardized
            feature= numerical_pip.fit_transform(X[Num_attribute])
            #Full_pip = ColumnTransformer([
            # ('num' , numerical_pip, Num_attribute),
            # ('onehote' , OneHotEncoder(), ['ocean_proximity'])])

            with open(self.scalar_path, 'wb') as f:
                pickle.dump(numerical_pip, f)
            # save the onehot encoder
            with open(self.onehotencoder_path, 'wb') as f:
                pickle.dump(enc, f)
            
        else:
            with open(self.onehotencoder_path, 'rb') as f:
                self.onehotencoder = pickle.load(f)
            with open(self.scalar_path, 'rb') as f:
                self.scalar_path = pickle.load(f)
            ocean_cat = self.onehotencoder.transform(X[['ocean_proximity']])
            ocean = ocean_cat.toarray()
            feature = self.scalar_path.transform(X[Num_attribute])
        
        feature = np.concatenate((feature,ocean),axis=1)
        return (feature,y)


class real_estate_predictor_evaluator(real_estate_predictor):
    def __init__(self, model_to_use):
        super().__init__()
        self.train=False
        self.model_to_use = model_to_use

        if(model_to_use == 'random_forest'):
            # load the random forest
            try:
                with open(self.rf_model_path, 'rb') as f:
                    self.model = pickle.load(f)
            except FileNotFoundError as error:
                sys.exit("random forest model file not found, please train the random forest model first")
        else:
            # load the linear model
            try:
                with open(self.linear_model_path, 'rb') as f:
                    self.model = pickle.load(f)
            except FileNotFoundError as error:
                sys.exit("linear regression model file not found, please train the linear regression model first")
    def  evaluate(self, test_path):
        try:
            feature = self.extract_feature_for_housing(test_path)
        except FileNotFoundError as error:
            sys.exit('{} could not be found'.format(test_path))
        X, y= feature[0], feature[1]
        y_pred = self.model.predict(X)
        num =0
        for results in y_pred:
            num = num+1
            print("The " +engine().ordinal(num)+ " house's predicted median price is: {}".format(round(results,2)))
        return y_pred

# class for training
class real_estate_predictor_trainer(real_estate_predictor):
    def __init__(self):
        super().__init__()
        self.train=True    
        self.X, self.y = self.extract_feature_for_housing(self.train_data)
        #self.X = feature[0]
        #self.y = feature[1]
    

    def train_random_forest(self):
        ran_for = RandomForestRegressor(n_estimators = self.tree_num, criterion= 'mse', min_samples_leaf=self.min_leaf, random_state=2020)

        # split the training data to training portion and testing portion
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=2020)
        
        start = time.time()
        ran_for.fit(X_train,y_train)
        end = time.time()
        runing_time = end-start
        print('time cost: %.5f sec' %runing_time)
        # print the accuracy on the testing data set
        print('the R2 score of random forest is: {}'.format(ran_for.score(X_test, y_test)))
        # save the model
        with open(self.rf_model_path, 'wb') as f:
            pickle.dump(ran_for, f)

    def train_linear(self):
        # define a linear regression
        reg = LinearRegression()

        # split the training data to training portion and testing portion
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=2020)

        # fit the linear regression
        start = time.time()
        reg.fit(X_train, y_train)
        end = time.time()
        runing_time = end-start
        print('time cost: %.5f sec' %runing_time)
        # print the accuracy on the testing data set
        print('the R2 score of linear regression is: {}'.format(reg.score(X_test, y_test)))

        # save the model
        with open(self.linear_model_path, 'wb') as f:
            pickle.dump(reg, f)
        

if __name__ == '__main__':

    # command line arguments parsing
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--operation", required=True, help="choose operations", choices=['train', 'evaluate'], default='train')
    ap.add_argument("-m", "--model", required=False, help="which model to be trained/used", choices=['random_forest', 'linear'], default='random_forest')
    ap.add_argument("-f", "--file", required=False, help="file to evaluate")
    args = vars(ap.parse_args())
    
    if(args['operation'] == 'train'):
        # user selected `train`, so we train the model
        trainer = real_estate_predictor_trainer()
        if(args['model'] == 'random_forest'):
            trainer.train_random_forest()
        else:
            trainer.train_linear()
    else:
        # user selected `evaluate`, so we use the trained model
        if(args['file'] is None):
            sys.exit('you have to specify an audio file with the -f option')
        # create an evaulator using the model specified from the command line
        evaluator = real_estate_predictor_evaluator(args['model'])
        # evaluate the audio file to calculate the caller id
        evaluator.evaluate(args['file'])


'''
usage(need to remove it here and write to a README.md file): 
1. CREATE A CONDA ENVIRONMENT:  
    conda create --name housing python=3.7

2. INSTALL DEPENDENCIES 
    pip install -r requirements.txt

3. TRAIN THE MODEL

3.1 train the model using random forest
    python model.py -o train -m random_forest

3.2 train the model using linear:
    python model.py -o train -m linear

4. TEST(EVALUATE) THE MODEL
4.1 test the keras feedforward neural network model:
    python model.py -o evaluate -m random_forest -f data/test_set.csv

4.2 test the knn model:
    python model.py -o evaluate -m linear -f data/test_set.csv
   
'''