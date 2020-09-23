# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
import numpy as np
import matplotlib.pyplot as plt 

# import functions and classes from Sklearn.py
import sys

sys.path.append('\\Users\\rachelzeng\\dsbook\\code\\python_appendix')  
print(sys.path)


# %%
import os 

print(sys.path)
from Sklearn_tutorial import data_loader, target_transformation, Attribute_pip

import pandas as pd
os.chdir('/Users/rachelzeng/dsbook')
Data_Path =os.path.join ('Data')
data_name = 'COVID_19.csv'
COVID = data_loader (data_path=Data_Path, data= data_name)


# %%
COVID.info()


# %%
print(COVID.head())


# %%

target_name = 'COVID_19?'
Irrelavent = COVID.pop ('ID')
target = COVID.pop (target_name)

COVID[['Travel?', 'Close contact', 'Dry Cough']] = COVID[['Travel?', 'Close contact', 'Dry Cough']].astype('bool')

COVID[['Residency Status',  'Sex']] = COVID[['Residency Status', 'Sex']].astype('category')


# %%
# Split Training and Testing data
from sklearn.model_selection import train_test_split

Train_Data, Test_Data, target_train, target_test  = train_test_split(COVID,target,test_size=0.2, random_state= 1,stratify=target)
Train_Data.info()


# %%
# Preprocessing 
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from Sklearn_tutorial import data_loader, target_transformation, Attribute_pip

null_train = Train_Data.loc[pd.isnull(Train_Data).any(axis = 1),:].index.values
Train_Data = Train_Data.drop(null_train)
null_test = Test_Data.loc[pd.isnull(Test_Data).any(axis = 1),:].index.values
Test_Data = Test_Data.drop(null_test)
target_train= target_train.drop(null_train)
target_test = target_test.drop(null_test)

Trained_transformed = Attribute_pip (Train_Data)
Tested_transformed = Attribute_pip (Test_ßData)
target_train = target_transformation (target_train)
target_test = target_transformation (target_test)
print(target_train)
print(Trained_transformed)


# %%
col_name = ['Age','Temperature', 'Sex-M', 'Sex-F', 'Residency Status-C ','Residency Status-NC', 'Travel Y', 'Travel N', 'Close contact Y', 'Close contact N' ,'Dry Cough-Y','Dry Cough- N']
print(Trained_transformed[1])


# %%


from sklearn.tree import DecisionTreeClassifier
Single_tree = DecisionTreeClassifier(criterion = 'entropy',random_state= 1).fit(Trained_transformed,target_train)
print(Single_tree)


# %%

from sklearn.metrics import precision_score
from sklearn.metrics import classification_report
predict = Single_tree.predict(Tested_transformed)
print('precision_ Score on the test data: {0:.2f} %'.format(100 *((precision_score)(y_true= target_test, y_pred=predict))))
print(classification_report(target_test, predict))


# %%

tree.plot_tree(Single_tree, filled= True)
os.chdir('/Users/rachelzeng/dsbook/fig')
import graphviz # pip install graphviz (brew install graphviz on MAC solves the problem of system path)
graph_data = tree.export_graphviz(Single_tree, out_file=None, 
                    feature_names=col_name, 
                     class_names=['infected', 'noninfected'] ,
                     filled=True, rounded=True,  
                     special_characters=True) 
graph = graphviz.Source(graph_data) 
graph.format = 'png'
graph.render("COVID_19_DT_py", view = 'True') 


# %%
# Feature Importantce
print(*zip(col_name ,Single_tree.feature_importances_))


# %%

from sklearn.metrics import accuracy_score
predict_train = Single_tree.predict(Trained_transformed)
print('accuracy_ Score on the train data: {0:.2f} %'.format(100 *((accuracy_score)(y_true= target_train, y_pred=predict_train ))))

predict_test = Single_tree.predict(Tested_transformed)
print('accuracy_ Score on the test data: {0:.2f} %'.format(100 *((accuracy_score)(y_true= target_test, y_pred=predict_test ))))


# %%

# Pre-pruning
Single_tree_pruned = DecisionTreeClassifier(criterion = 'entropy', random_state= 1, min_samples_split=15, max_depth=4).fit(Trained_transformed,target_train)
score_prun = Single_tree_pruned.score(Tested_transformed,target_test )
print('accuracy_ Score of pre-pruned Tree on the test data: {0:.2f} %'.format(100 *score_prun))

# New graph:
tree.plot_tree(Single_tree_pruned, filled= True)

graph_data = tree.export_graphviz(Single_tree_pruned, out_file=None, 
                    feature_names=col_name, 
                     class_names=['infected', 'noninfected'] ,
                     filled=True, rounded=True,  
                     special_characters=True) 
graph = graphviz.Source(graph_data) 
graph.format = 'png'
graph.render("COVID_19_DTpruned_py", view=True) 


# %%
print(*zip(col_name ,Single_tree_pruned.feature_importances_))


# %%
# Post pruning 
prune_path = Single_tree.cost_complexity_pruning_path(Trained_transformed, target_train)
ccp_alphas, impurities = prune_path.ccp_alphas, prune_path.impurities

fig, ax = plt.subplots()
ax.plot(ccp_alphas[:-1], impurities[:-1], marker='o', drawstyle="steps-post")
ax.set_xlabel("effective alpha")
ax.set_ylabel("total impurity of leaves")
ax.set_title("Total Impurity vs effective alpha for training set")

plt.show()

clfs = []
for ccp_alpha in ccp_alphas:
    clf = DecisionTreeClassifier(criterion = 'entropy',random_state=1, ccp_alpha=ccp_alpha)
    clf.fit(Trained_transformed, target_train)
    clfs.append(clf)
print("Number of nodes in the last tree is: {} with ccp_alpha: {}".format(
      clfs[-1].tree_.node_count, ccp_alphas[-1]))

clfs = clfs[:-1]
ccp_alphas = ccp_alphas[:-1]

node_counts = [clf.tree_.node_count for clf in clfs]
depth = [clf.tree_.max_depth for clf in clfs]
fig, ax = plt.subplots(2, 1)
ax[0].plot(ccp_alphas, node_counts, marker='o', drawstyle="steps-post")
ax[0].set_xlabel("alpha")
ax[0].set_ylabel("number of nodes")
ax[0].set_title("Number of nodes vs alpha")
ax[1].plot(ccp_alphas, depth, marker='o', drawstyle="steps-post")
ax[1].set_xlabel("alpha")
ax[1].set_ylabel("depth of tree")
ax[1].set_title("Depth vs alpha")
fig.tight_layout()

train_scores = [clf.score(Trained_transformed, target_train) for clf in clfs]
test_scores = [clf.score(Tested_transformed, target_test) for clf in clfs]
print(test_scores)

fig, ax = plt.subplots()
ax.set_xlabel("alpha")
ax.set_ylabel("accuracy")
ax.set_title("Accuracy vs alpha for training and testing sets")
ax.plot(ccp_alphas, train_scores, marker='o', label="train",
        drawstyle="steps-post")
ax.plot(ccp_alphas, test_scores, marker='o', label="test",
        drawstyle="steps-post")
ax.legend()
plt.show()

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

classes = ['Healthy','Infected']
colors = [ 'forestgreen','slateblue']
scatter = plt.scatter(Trained_transformed[:,0], Trained_transformed[:,1], c= target_train, cmap = mcolors.ListedColormap(colors))
plt.legend(handles=scatter.legend_elements()[0], labels=classes)
plt.xlabel('Scaled - Age')
plt.ylabel('Scaled -Temperature')
plt.show()

# %%
A_tree = DecisionTreeClassifier(criterion = 'entropy',random_state= 1).fit(Trained_transformed[:,[1,2]],target_train)
tree_prediction = A_tree.predict(Tested_transformed[:,[1,2]])

from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
bag_clf =  BaggingClassifier(DecisionTreeClassifier(), n_estimators= 300, max_samples= 0.6, bootstrap= True, n_jobs = -1 )
bag_classifier = bag_clf.fit(Trained_transformed, target_train )
ensemble_prediction = bag_classifier.predict(Tested_transformed)
print(accuracy_score(target_test, tree_prediction))
print(accuracy_score(target_test, ensemble_prediction)}

# %%
# Random Forest Classifier 
from sklearn.ensemble import RandomForestClassifier
rand_clf = RandomForestClassifier(n_estimators= 300, max_leaf_nodes=20, n_jobs=-1 )
forest_classifier = rand_clf.fit(Trained_transformed, target_train )
ensemble_prediction_forest = forest_classifier.predict(Tested_transformed)
accuracy_score(target_test, ensemble_prediction_forest)


# %%



