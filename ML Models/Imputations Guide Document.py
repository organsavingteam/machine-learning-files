from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.naive_bayes import GaussianNB
from scipy.interpolate import interp1d
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn import preprocessing
from sklearn.manifold import TSNE 
from sklearn import metrics
from scipy import stats
from sklearn import tree
from sklearn import svm
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import graphviz 
import xlsxwriter
import pandas
import numpy 
import umap
import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin'

raw_data=pandas.read_excel('/Users/Frank Yu/Desktop/ImpMay29.xlsx')
raw_data=raw_data.iloc[0:195,0:47]
raw_data['Aortic Cann.'] = np.where((raw_data['DA Cann.'].isin(['1'])),1,np.where((raw_data['INA Cann.'].isin(['1'])),1,0))
x=raw_data.iloc[0:186,1:43]
y_RA=raw_data.iloc[0:186,43]
y_INA=raw_data.iloc[0:186,44]
y_fem=raw_data.iloc[0:186,45]
y_DA=raw_data.iloc[0:186,46]
y_AC=raw_data.iloc[0:186,47]
x_impute=raw_data.iloc[186:195,1:43]
cannulation = [y_RA, y_INA, y_fem, y_DA, y_AC]

mean = x['AGE'].mean()
sd = x['AGE'].std()
x['Normalized Age'] = x['AGE'].apply(lambda x: (x-mean)/sd)
x.drop(['AGE'], axis=1)
cannulation_type = ["RA", "INA", "femoral", "DA", "AC"]
counter = 0
row_counter = 43

# KNN Imputation 
# print("KNN Imputation")
# for z in cannulation:
#     X_train, X_test, y_train, y_test = train_test_split(x, z, test_size=0.20, random_state=0)
#     X = X_train
#     y = y_train

#     neigh = KNeighborsClassifier(n_neighbors=2)
#     neigh.fit(X, y)

#     y_pred = neigh.predict(X_test)
#     print("KNN imputation for ",cannulation_type[counter])
#     print(metrics.classification_report(y_test, y_pred))

#     counter +=1

# SVM Imputation
# for z in cannulation:
#     X_train, X_test, y_train, y_test = train_test_split(x, z, test_size=0.20, random_state=0)
#     SVMmodel = svm.SVC(gamma='scale', decision_function_shape='ovo')
#     SVMmodel.fit(X_train,y_train)
#     y_pred=SVMmodel.predict(X_test)

#     print("SVM Imputation for ",cannulation_type[counter])
#     print(metrics.classification_report(y_test, y_pred))

#     counter +=1


# General Parameter Optimization for Decision Tree
# criterion = ['gini', 'entropy']
# splitter = ['best', 'random']
# max_features = ['auto', 'sqrt','log2', None ]
# max_depth = [int(x) for x in numpy.linspace(50, 200, num = 11)]
# min_samples_split = [5, 10, 15]
# min_samples_leaf = [1, 2, 4, 6]
# random_grid = {'criterion': criterion,
#                'splitter': splitter,
#                'max_features': max_features,
#                'max_depth': max_depth,
#                'min_samples_split': min_samples_split,
#                'min_samples_leaf': min_samples_leaf,
#               }
# print(random_grid)

# X_train, X_test, y_train, y_test = train_test_split(x, y_fem, test_size=0.20, random_state=0)

# rash = tree.DecisionTreeClassifier()
# rf_random = RandomizedSearchCV(estimator = rash, param_distributions = random_grid, n_iter = 20, cv = 5, verbose=2, random_state=42, n_jobs = -1)
# rf_random.fit(X_train, y_train)

# print(rf_random.best_params_)

# # Parameter Fine Tuning
# X_train, X_test, y_train, y_test = train_test_split(x, y_fem, test_size=0.20, random_state=0)
# from sklearn.model_selection import GridSearchCV
# # Create the parameter grid based on the results of random search 
# param_grid = {
#     'criterion': ['gini'],
#     'splitter': ['random'],
#     'max_features': ['auto'],
#     'max_depth': [30, 40, 50, 60, 70, 80],
#     'min_samples_split': [7, 8, 9, 10, 11, 12, 13],
#     'min_samples_leaf': [2, 3, 4, 5, 6, 7, 8, 9],
# }
# # Create a based model
# rf = tree.DecisionTreeClassifier()
# # Instantiate the grid search model
# grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 
#                           cv = 3, n_jobs = -1, verbose = 2)

# # Fit the grid search to the data
# grid_search.fit(X_train, y_train)
# print(grid_search.best_params_)

X_train, X_test, y_train, y_test = train_test_split(x, y_fem, test_size=0.20, random_state=0)
treemodel = tree.DecisionTreeClassifier(criterion='gini', max_depth=70, max_features='auto', min_samples_leaf=2, min_samples_split=12, splitter='random')
treemodel.fit(X_train,y_train)
y_pred=treemodel.predict(X_test)

print("Decision tree Imputation for ",cannulation_type[counter])
print(metrics.classification_report(y_test, y_pred))

# Decision Tree Imputation
# for z in cannulation:
#     X_train, X_test, y_train, y_test = train_test_split(x, z, test_size=0.20, random_state=0)
#     treemodel = tree.DecisionTreeClassifier(criterion='gini')
#     treemodel.fit(X_train,y_train)
#     y_pred=treemodel.predict(X_test)
#     raw_data[187:195, row_counter] = y_pred

#     dot_data = tree.export_graphviz(treemodel.fit(X_train,y_train), out_file=None, 
#         filled=True, rounded=True,  
#         special_characters=True) 
#     graph = graphviz.Source(dot_data)
#     graph.render(cannulation_type[counter])
#     print("Decision tree Imputation for ",cannulation_type[counter])
#     print(metrics.classification_report(y_test, y_pred))

#     counter +=1
#     row_counter +=1

# # Number of trees in random forest
# n_estimators = [int(x) for x in numpy.linspace(start = 100, stop = 1000, num = 100)]
# # Number of features to consider at every split
# max_features = ['auto', 'sqrt','log2', None ]
# # Maximum number of levels in tree
# max_depth = [int(x) for x in numpy.linspace(50, 200, num = 11)]
# max_depth.append(None)
# # Minimum number of samples required to split a node
# min_samples_split = [5, 10, 15]
# # Minimum number of samples required at each leaf node
# min_samples_leaf = [1, 2, 4, 6]
# # Method of selecting samples for training each tree
# bootstrap = [True, False]
# # Create the random grid
# random_grid = {'n_estimators': n_estimators,
#                'max_features': max_features,
#                'max_depth': max_depth,
#                'min_samples_split': min_samples_split,
#                'min_samples_leaf': min_samples_leaf,
#                'bootstrap': bootstrap
#               }
# print(random_grid)

# X_train, X_test, y_train, y_test = train_test_split(x, y_fem, test_size=0.20, random_state=0)

# rash = RandomForestClassifier()
# rf_random = RandomizedSearchCV(estimator = rash, param_distributions = random_grid, n_iter = 20, cv = 5, verbose=2, random_state=42, n_jobs = -1)
# rf_random.fit(X_train, y_train)

# print(rf_random.best_params_)

 
# X_train, X_test, y_train, y_test = train_test_split(x, y_fem, test_size=0.20, random_state=0)
# from sklearn.model_selection import GridSearchCV
# # Create the parameter grid based on the results of random search 
# param_grid = {
#     'bootstrap': [True],
#     'max_depth': [80,90,100,110,120],
#     'max_features': ['sqrt'],
#     'min_samples_leaf': [1, 2, 3],
#     'min_samples_split': [8, 10, 12],
#     'n_estimators': [900,950,1000]
# }
# # Create a based model
# rf = RandomForestClassifier()
# # Instantiate the grid search model
# grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 
#                           cv = 3, n_jobs = -1, verbose = 2)

# # Fit the grid search to the data
# grid_search.fit(X_train, y_train)
# print(grid_search.best_params_)


# # Random Forest Imputation
# for z in cannulation:
#     X_train, X_test, y_train, y_test = train_test_split(x, z, test_size=0.20, random_state=0)
#     model = RandomForestClassifier(bootstrap= True,
#         max_depth= 80,
#         max_features= 'sqrt',
#         min_samples_leaf=2,
#         min_samples_split=8,
#         n_estimators=900)
#     model.fit(X_train, y_train)
#     y_pred = model.predict(X_test)
#     print(metrics.classification_report(y_test, y_pred))
#     print(y_pred)
#     print(y_test)
