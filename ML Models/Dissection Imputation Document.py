from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
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
import sys
import re
import json
import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin'

# Instantiation
raw_data = pandas.read_excel('/Users/Frank Yu/Desktop/ImpMay29.xlsx')
raw_data = raw_data.iloc[0:195, 0:47]
raw_data['Aortic Cann.'] = np.where((raw_data['DA Cann.'].isin(
    ['1'])), 1, np.where((raw_data['INA Cann.'].isin(['1'])), 1, 0))
x = raw_data.iloc[0:186, 1:43]
y_RA = raw_data.iloc[0:186, 43]
y_INA = raw_data.iloc[0:186, 44]
y_fem = raw_data.iloc[0:186, 45]
y_DA = raw_data.iloc[0:186, 46]
y_AC = raw_data.iloc[0:186, 47]
x_impute = raw_data.iloc[187:195, 1:43]

# Definitions
def normalizedAge(x):
    mean = x['AGE'].mean()
    sd = x['AGE'].std()
    x['Normalized Age'] = x['AGE'].apply(lambda y: (y - mean)/sd)
    x.drop(['AGE'], axis=1)
    return x

def listToString(s):
    str1 = ""
    for ele in s:
        str1 += ele
    return str1

def generateSmallNumberArray(x):
    res = []
    neg_counter = 6
    pos_counter = 6
    neg_x = x
    pos_x = x + 1
    while(neg_x > 0 and neg_counter > 0):
        res.append(neg_x)
        neg_x -= 1
        neg_counter -= 1
    while(pos_x > 0 and pos_counter > 0):
        res.append(pos_x)
        pos_x += 1
        pos_counter -= 1
    return res

def generateLargeNumberArray(x):
    res = []
    neg_counter = 5
    pos_counter = 5
    neg_x = x
    pos_x = x + 10
    while(neg_x > 0 and neg_counter > 0):
        res.append(neg_x)
        neg_x -= 10
        neg_counter -= 1
    while(pos_x > 0 and pos_counter > 0):
        res.append(pos_x)
        pos_x += 10
        pos_counter -= 1
    return res

def dt_optimization_parameters(type_of_cannulation, column_name):
    X_train, X_test, y_train, y_test = train_test_split(x, type_of_cannulation, test_size=0.20, random_state=0)
    normalizedAge(X_train)
    normalizedAge(X_test)
    # Random search
    criterion = ['gini', 'entropy']
    splitter = ['best', 'random']
    max_features = ['auto', 'sqrt', 'log2', None]
    max_depth = [int(x) for x in numpy.linspace(50, 200, num=150)]
    min_samples_split = [1, 2, 3, 4, 5, 6, 7, 8, 9,
                         10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    min_samples_leaf = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    random_grid = {'criterion': criterion,
                   'splitter': splitter,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   }

    clf = tree.DecisionTreeClassifier()
    clf_random = RandomizedSearchCV(estimator=clf, param_distributions=random_grid,
                                    n_iter=30, cv=5, verbose=2, random_state=42, n_jobs=-1)
    clf_random.fit(X_train, y_train)

    temp = json.dumps(clf_random.best_params_)
    criterion = re.findall('"criterion": "(.+)"}', temp)
    splitter = re.findall('"splitter": "(.+)", "min_samples_split"', temp)
    max_features = re.findall('"max_features": "(.+)", "max_depth"', temp)
    max_depth = generateLargeNumberArray(
        int(''.join(str(e) for e in re.findall('"max_depth": (.+), "criterion"', temp))))
    min_samples_split = generateSmallNumberArray(int(''.join(str(
        e) for e in re.findall('"min_samples_split": (.+), "min_samples_leaf"', temp))))
    min_samples_leaf = generateSmallNumberArray(int(''.join(
        str(e) for e in re.findall('"min_samples_leaf": (.+), "max_features"', temp))))

    # Grid Search
    param_grid = {
        'criterion': criterion,
        'splitter': splitter,
        'max_features': max_features,
        'max_depth': max_depth,
        'min_samples_split': min_samples_split,
        'min_samples_leaf': min_samples_leaf,
    }
    df = tree.DecisionTreeClassifier()
    clf = GridSearchCV(estimator=df, param_grid=param_grid,
                       cv=4, n_jobs=-1, verbose=2)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    # Store the Parameters
    temp = json.dumps(clf.best_params_)
    splitter = ''.join(re.findall('"splitter": "(.+)"}', temp))
    max_features = ''.join(re.findall(
        '"max_features": "(.+)", "min_samples_leaf"', temp))
    max_depth = ''.join(re.findall('"max_depth": (.+), "max_features"', temp))
    min_samples_split = ''.join(re.findall('"min_samples_split": (.+), "splitter"', temp))
    min_samples_leaf = ''.join(re.findall('"min_samples_leaf": (.+), "min_samples_split"', temp))

    file = open('Decision Tree Parameter.txt', 'a')
    file.write("The following parameters are for "+
         column_name+"\n")
    file.write("Splitter: "+ splitter+"\n")
    file.write("Max Features: "+ max_features+"\n")
    file.write("Max Depth: "+ max_depth+"\n")
    file.write("Min Samples Split: "+ min_samples_split+"\n")
    file.write("Min Samples Leaf: "+ min_samples_leaf+"\n")
    file.write(metrics.classification_report(y_test, y_pred))
    file.close()

    # Train the model with the entire data set
    normalizedAge(x)
    clf.fit(x, type_of_cannulation)

    return clf

def dt_imputation(type_of_cannulation, column_name):
    y_pred = dt_optimization_parameters(
        type_of_cannulation, column_name).predict(x_impute)
    raw_data.iloc[187:195, raw_data.columns.get_loc(column_name)] = y_pred
    print(y_pred)

def export_excel(dataframe, name):
    name = name + ".xlsx"
    dataframe.to_excel(name)

file = open('Decision Tree Parameter.txt', 'w')
file.close()
cannulation = [("Fem. Cann.", y_fem), ("RA Cann.", y_RA), ("Aortic Cann.", y_DA)]
normalizedAge(x_impute)

for a, b in cannulation:
    dt_imputation(b, a)

export_excel(raw_data, "new File")