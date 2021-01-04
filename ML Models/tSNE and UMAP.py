from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
from sklearn import preprocessing
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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
    pre_op[name_of_new_column] = data_norm

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
        x_min.append(int(input("What is the x min of this cluster")))
        x_max.append(int(input("What is the x max of this cluster")))
        y_min.append(int(input("What is the y min of this cluster")))
        y_max.append(int(input("What is the y max of this cluster")))
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

raw_data=pd.read_excel('/Users/Frank Yu/Desktop/Updated Again.xlsx')
pre_op = raw_data.iloc[0:195,1:43]
intra_op = raw_data.iloc[0:195,43:47]
post_op = raw_data.iloc[0:195, 47:68]
post_op_notdied=post_op.iloc[:,0:len(post_op.columns)-2]
normalization(pre_op, 1, 'AGE') 

# # tSNE and UMAP for preop, intraop, and postop data
# print('tSNE generation for preop introp and postop data')
# for x in [pre_op, intra_op, post_op]:
#     binary_tsne(x, 'none', 'none', 'none', 'none')
#     binary_UMAP(x, 'none', 'none', 'none', 'none')

# # preop vs. cannulation strategy
# binary_UMAP(pre_op, intra_op, 'Fem. Cann.', ['blue','red'], ('No can','Fem can'))
# binary_UMAP(pre_op, intra_op, 'Aortic Cann.', ['blue','red'], ('No can','Aortic can'))
# binary_UMAP(pre_op, intra_op, 'RA Cann.', ['blue','red'], ('No can','RA can'))
# tsne(pre_op,intra_op,'Cann',['grey','red','black','green','purple','blue','yellow'], ('RA', 'RA & Fem', 'RA & Ao', 'RA, Fem, & Ao', 'Fem', 'Fem & Ao', 'Ao'))
# UMAP(pre_op,intra_op,'Cann',['grey','red','black','green','purple','blue','yellow'], ('RA', 'RA & Fem', 'RA & Ao', 'RA, Fem, & Ao', 'Fem', 'Fem & Ao', 'Ao'))

# # preop vs. died
# binary_tsne(pre_op, post_op, 'DEATH', ['blue','red'], ('No Death', 'Death'))
# binary_UMAP(pre_op, post_op, 'DEATH', ['blue','red'], ('No Death', 'Death'))

# # postop vs. type of dissection
# binary_tsne(post_op, pre_op, 'Dissection Type 1', ['blue', 'red'], ['Dissection 1', 'Dissection 2'])
# binary_UMAP(post_op, pre_op, 'Dissection Type 1', ['blue', 'red'], ['Dissection 1', 'Dissection 2'])

# # post_op_notdied vs. died
# binary_tsne(post_op_notdied, post_op, 'DEATH', ['blue', 'red'], ['Death','No Death'])
# binary_UMAP(post_op_notdied, post_op, 'DEATH', ['blue', 'red'], ['Death','No Death'])

# # intra_op and died
# binary_tsne(intra_op, post_op, 'DEATH', ['blue', 'red'], ['Death','No Death'])
# binary_UMAP(intra_op, post_op, 'DEATH', ['blue', 'red'], ['Death','No Death'])
# binary_tsne(intra_op, post_op, 'DIED', ['blue', 'red'], ['Death','No Death'])
# dataframe = binary_UMAP(intra_op, post_op, 'DIED', ['blue', 'red'], ['Death','No Death'])

# # find a group of data
# partial_scatter_points, original_locations = find_points(dataframe, -10, 0, -10, 0)
# plt.scatter(partial_scatter_points[:,0],partial_scatter_points[:,1])
# plt.show()
# print(partial_scatter_points)
# print(original_locations)