from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler 
from eli5.sklearn import PermutationImportance
from sklearn.decomposition import PCA 
from sklearn import preprocessing
from sklearn.manifold import TSNE
from scipy.stats import chi2_contingency
from scipy.stats import chi2
from sklearn import metrics
from sklearn import tree
from scipy import stats
from sklearn import svm
import matplotlib.pyplot as plt
import seaborn as sns 
import pandas as pd
import numpy as np
import numpy
import umap
import json
import eli5
import re
import umap.umap_ as umap

# tSNE Definition: Pass the dataframe as data, pass the group as a list or none, pass the colors as list or none, pass the labels as a list or none
def binary_tsne(data, group_data, group, colours, label):
    tsne = TSNE(n_components = 2, random_state = 42, perplexity = 40, n_iter = 5000)
    values = tsne.fit_transform(data)

    if type(group_data) != pd.core.frame.DataFrame:
        plt.scatter(values[:,0],values[:,1])
        plt.title('tSNE')
        plt.show()
        return(values)

    else:
        df = pd.DataFrame(columns = ['dim1', 'dim2', 'group'])
        df['dim1'] = values[:,0]
        df['dim2'] = values[:,1]
        df['group'] = group_data[group]

        fig = plt.figure(figsize = (8,8))
        fig = fig.add_subplot(1,1,1)
        fig.set_title('tSNE', fontsize = 20)
        labels = [0,1]
        colors = colours

        for x, y in zip(labels, colors):
            indiciesToKeep = df['group'] == x
            fig.scatter(df.loc[indiciesToKeep, 'dim1'],
                        df.loc[indiciesToKeep, 'dim2'],
                        c = y,
                        s = 50)

        fig.legend(label)
        fig.grid()
        plt.show()
        return(values)

def binary_UMAP(data, group_data, group, colours, label):
    reducer = umap.UMAP(random_state=42)
    values = reducer.fit_transform(data)

    if type(group_data) != pd.core.frame.DataFrame:
        plt.scatter(values[:,0],values[:,1])
        plt.title('UMAP')
        plt.show()
        return(values)
    
    else:
        df = pd.DataFrame(columns = ['dim1', 'dim2', 'group'])
        df['dim1'] = values[:,0]
        df['dim2'] = values[:,1]
        df['group'] = group_data[group]

        fig = plt.figure(figsize = (8,8))
        fig = fig.add_subplot(1,1,1)
        fig.set_title('UMAP', fontsize = 20)
        labels = [0,1]
        colors = colours

        for x, y in zip(labels, colors):
            indiciesToKeep = df['group'] == x
            fig.scatter(df.loc[indiciesToKeep, 'dim1'],
                        df.loc[indiciesToKeep, 'dim2'],
                        c = y,
                        s = 50)
        
        fig.legend(label)
        fig.grid()
        plt.show()
        return(values)

def tsne(data, group_data, group, colours, label):
    tsne = TSNE(n_components = 2, random_state = 42, perplexity = 40, n_iter = 5000)
    values = tsne.fit_transform(data)

    df = pd.DataFrame(columns = ['dim1', 'dim2', 'group'])
    df['dim1'] = values[:,0]
    df['dim2'] = values[:,1]
    df['group'] = group_data[group]

    fig = plt.figure(figsize = (8,8))
    fig = fig.add_subplot(1,1,1)
    fig.set_title('tSNE', fontsize = 20)
    labels = range(0, len(label))
    colors = colours

    for x, y in zip(labels, colors):
        indiciesToKeep = df['group'] == x
        fig.scatter(df.loc[indiciesToKeep, 'dim1'],
                    df.loc[indiciesToKeep, 'dim2'],
                    c = y,
                    s = 50)

    fig.legend(label)
    fig.grid()
    plt.show()
    return(values)

def UMAP(data, group_data, group, colours, label):
    reducer = umap.UMAP(random_state=42)
    values = reducer.fit_transform(data)

    df = pd.DataFrame(columns = ['dim1', 'dim2', 'group'])
    df['dim1'] = values[:,0]
    df['dim2'] = values[:,1]
    df['group'] = group_data[group]

    fig = plt.figure(figsize = (8,8))
    fig = fig.add_subplot(1,1,1)
    fig.set_title('UMAP', fontsize = 20)
    labels = range(0, len(label))
    colors = colours

    for x, y in zip(labels, colors):
        indiciesToKeep = df['group'] == x
        fig.scatter(df.loc[indiciesToKeep, 'dim1'],
                    df.loc[indiciesToKeep, 'dim2'],
                    c = y,
                    s = 50)

    fig.legend(label)
    fig.grid()
    plt.show()   
    return(values)

def normalization(dataframe, column, name_of_new_column):
    data = dataframe.iloc[:,column]
    means = data.mean(axis=0)
    stddev = data.std(axis=0)

    data_norm = (data - means)/stddev
    dataframe[name_of_new_column] = data_norm

def find_points(scatter_data, x_min, x_max, y_min, y_max):
    points = len(scatter_data)
    dataframe = []
    original_points = []
    for x in range(points):
        if scatter_data[x][0] >= x_min:
            if scatter_data[x][0] <= x_max:
                if scatter_data[x][1] >= y_min:
                    if scatter_data[x][1] <= y_max:
                        dataframe.append(scatter_data[x].tolist())
                        original_points.append(x)
    dataframe = np.array(dataframe)
    dictionary = {original_points[x]: dataframe[x] for x in range(len(original_points))}
    return dictionary, dataframe, original_points

def identify_clusters(scatter_data):
    num_clusters = int(input("How many clusters do you see in your data? (num only)"))
    x_min = []
    x_max = []
    y_min = []
    y_max = []
    counter = 0
    text_counter = 1
    dataframe = pd.DataFrame(columns = ['Patient Number', 'Cluster'])
    
    for x in range(num_clusters):
        print("Enter the dimensions of cluster ",text_counter)
        x_min.append(float(input("What is the x min of this cluster")))
        x_max.append(float(input("What is the x max of this cluster")))
        y_min.append(float(input("What is the y min of this cluster")))
        y_max.append(float(input("What is the y max of this cluster")))
        text_counter += 1
        counter += 1

    for y in range(num_clusters):
        for x in range(len(scatter_data)):
            if scatter_data[x][0] >= x_min[y]:
                if scatter_data[x][0] <= x_max[y]:
                    if scatter_data[x][1] >= y_min[y]:
                        if scatter_data[x][1] <= y_max[y]:
                            dataframe = dataframe.append({'Patient Number': x, 'Cluster': y}, ignore_index = True)
    return dataframe

def match_data(original_data, new_data):
    unique_identifier = input("What is the unique identifer column name? Hint: the columns must have the same name")
    merged_data = pd.merge(original_data, new_data, on = unique_identifier, how = 'inner')
    return merged_data

def normalizedAge(x):
    mean = x['AGE'].mean()
    sd = x['AGE'].std()
    x['Normalized Age'] = x['AGE'].apply(lambda y: (y - mean)/sd)
    x.drop(['AGE'], axis=1)
    return x

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

def dt_optimization_parameters(data, type_of_cannulation):
    X_train, X_test, y_train, y_test = train_test_split(data, type_of_cannulation, test_size=0.20, random_state=0)
    # Random search
    criterion = ['gini', 'entropy']
    splitter = ['best', 'random']
    max_features = ['auto', 'sqrt', 'log2']
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
    print(metrics.classification_report(y_test, y_pred))

    # Store the Parameters
    temp = json.dumps(clf.best_params_)
    splitter = ''.join(re.findall('"splitter": "(.+)"}', temp))
    max_features = ''.join(re.findall(
        '"max_features": "(.+)", "min_samples_leaf"', temp))
    max_depth = ''.join(re.findall('"max_depth": (.+), "max_features"', temp))
    min_samples_split = ''.join(re.findall('"min_samples_split": (.+), "splitter"', temp))
    min_samples_leaf = ''.join(re.findall('"min_samples_leaf": (.+), "min_samples_split"', temp))
    
    clf.fit(data, type_of_cannulation)
    
    return clf

def make_dictionary(columns_name, scores):
    colnames = list(columns_name)
    fd = zip(colnames, scores)
    dictionary = dict(fd)

    return dictionary

def chi_square(dataframe, groups):
    columns = dataframe.columns.tolist()
    for x in columns:
        print(x)
        data_crosstab = pd.crosstab(dataframe[x], dataframe[groups], margins = False)
        stat, p, dof, expected = chi2_contingency(data_crosstab)
        print('dof=%d' % dof)
        print(expected)
        # interpret test-statistic
        prob = 0.95
        critical = chi2.ppf(prob, dof)
        print('probability=%.3f, critical=%.3f, stat=%.3f' % (prob, critical, stat))
        if abs(stat) >= critical:
            print('Dependent (reject H0)')
        else:
            print('Independent (fail to reject H0)')
        # interpret p-value
        alpha = 1.0 - prob
        print('significance=%.3f, p=%.3f' % (alpha, p))
        if p <= alpha:
            print('Dependent (reject H0)')
        else:
            print('Independent (fail to reject H0)')

def model_svm(xdata, ydata):
    X_train, X_test, y_train, y_test = train_test_split(xdata, ydata, test_size=0.20, random_state=0)
    SVMmodel = svm.SVC(gamma = 'scale', decision_function_shape = 'ovo', kernel = 'linear')
    SVMmodel.fit(X_train, y_train)
    y_pred = SVMmodel.predict(X_test)
    
    print(metrics.classification_report(y_test, y_pred))
    
    return SVMmodel

def PCA_analysis(xdata, ydata):
    scalar = StandardScaler() 
  
    # fitting 
    scalar.fit(xdata) 
    scaled_data = scalar.transform(xdata) 
    
    pca = PCA(n_components = 2) 
    pca.fit(scaled_data) 
    x_pca = pca.transform(scaled_data) 
    
    plt.figure(figsize =(8, 6)) 
  
    plt.scatter(x_pca[:, 0], x_pca[:, 1], c = ydata, cmap ='plasma') 

    # labeling x and y axes 
    plt.xlabel('First Principal Component') 
    plt.ylabel('Second Principal Component') 
    
    df_comp = pd.DataFrame(pca.components_, columns = xdata.columns)
    plt.figure(figsize =(14, 6)) 

    # plotting heatmap 
    sns.heatmap(df_comp)
            
raw_data=pd.read_excel('/Users/Frank Yu/Desktop/Updated Again.xlsx')
normalization(raw_data, 1, 'AGE')
pre_op = raw_data.iloc[0:195,1:43]
intra_op = raw_data.iloc[0:195,43:47]
post_op = raw_data.iloc[0:195, 47:68]
post_op_notdied=post_op.iloc[:,0:len(post_op.columns)-1]
pre_intra_op = raw_data.iloc[0:195, 1:46]

post_op_clusters = identify_clusters(binary_UMAP(post_op, 'none', 'none', 'none','none'))

data = match_data(raw_data, post_op_clusters)
# Drop the last 25 columns in the database
cols = []
counter = 0
for x in range(47):
    cols.append(counter)
    counter+=1
data.drop(data.columns[cols], axis = 1, inplace = True)
data_wo_clusters = data.drop(['Cluster'], axis = 1)
clusters = data.iloc[0:194, 20]
clusters = clusters.astype(str).astype(int)

# Pass the data through decision tree algorithm
model = dt_optimization_parameters(data_wo_clusters, clusters)

model

middle_cluster = identify_clusters(binary_UMAP(post_op, 'none', 'none', 'none','none'))
data = match_data(raw_data, middle_cluster)
# Drop the last 25 columns in the database
cols = []
counter = 0
for x in range(47):
    cols.append(counter)
    counter+=1
data.drop(data.columns[cols], axis = 1, inplace = True)
data_wo_clusters = data.drop(['Cluster'], axis = 1)
y_pred = model.predict(data_wo_clusters)

print(y_pred)

from sklearn.ensemble import RandomForestClassifier
# def fitmodel(x,y):
#     # Number of trees in random forest
#     n_estimators = [int(x) for x in numpy.linspace(start = 100, stop = 1000, num = 100)]
#     # Number of features to consider at every split
#     max_features = ['auto', 'sqrt','log2']
#     # Maximum number of levels in tree
#     max_depth = [int(x) for x in numpy.linspace(50, 200, num = 11)]
#     max_depth.append(None)
#     # Minimum number of samples required to split a node
#     min_samples_split = [5, 10, 15]
#     # Minimum number of samples required at each leaf node
#     min_samples_leaf = [1, 2, 4, 6]
#     # Method of selecting samples for training each tree
#     bootstrap = [True, False]
#     # Create the random grid
#     random_grid = {'n_estimators': n_estimators,
#                    'max_features': max_features,
#                    'max_depth': max_depth,
#                    'min_samples_split': min_samples_split,
#                    'min_samples_leaf': min_samples_leaf,
#                    'bootstrap': bootstrap
#                   }
#     clf = RandomForestClassifier()
#     clf_random = RandomizedSearchCV(estimator=clf, param_distributions=random_grid, scoring= 'accuracy',
#                                         n_iter=30, cv=5, verbose=2, random_state=42, n_jobs=-1)
#     clf_random.fit(x,y)
#     temp = clf_random.best_params_
#     temp_text = json.dumps(clf_random.best_params_)
    
#     best_parameters=temp
    
#     return best_parameters

# post_op_clusters = identify_clusters(binary_UMAP(post_op, 'none', 'none', 'none','none'))

# data = match_data(raw_data, post_op_clusters)
# # Drop the last 25 columns in the database
# cols = []
# counter = 0
# for x in range(47):
#     cols.append(counter)
#     counter+=1
# data.drop(data.columns[cols], axis = 1, inplace = True)
# data_wo_clusters = data.drop(['Cluster'], axis = 1)
# clusters = data.iloc[0:194, 20]
# clusters = clusters.astype(str).astype(int)
# best_parameters = fitmodel(X_train, y_train)

# model = RandomForestClassifier(bootstrap=str(best_parameters['bootstrap']),
#      max_depth=best_parameters['max_depth'],
#      max_features= str(best_parameters['max_features']),
#      min_samples_leaf= best_parameters['min_samples_leaf'],
#      min_samples_split= best_parameters['min_samples_split'],
#      n_estimators= best_parameters['n_estimators'])
# for x in range(20):
#     X_train, X_test, y_train, y_test = train_test_split(data_wo_clusters, clusters, test_size=0.20, random_state=None)
#     model = RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
#                        criterion='gini', max_depth=None, max_features='auto',
#                        max_leaf_nodes=None, max_samples=None,
#                        min_impurity_decrease=0.0, min_impurity_split=None,
#                        min_samples_leaf=1, min_samples_split=2,
#                        min_weight_fraction_leaf=0.0, n_estimators=100,
#                        n_jobs=None, oob_score=False, random_state=None,
#                        verbose=0, warm_start=False)
#     model.fit(X_train, y_train)
#     y_pred = model.predict(X_test)
#     print(metrics.classification_report(y_test, y_pred)) 
#     print(model.get_params)
dictionary = make_dictionary(list(data_wo_clusters.columns), model.feature_importances_*100)
dictionary
