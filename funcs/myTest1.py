# 鸢尾花分类
'''
@ four features: sepal length, sepal width, petal length, petal width
@ three categories: setosa, versicolor, virginica
@ data shape: (150, 4)
'''

def IrisData_read():
    from sklearn.datasets import load_iris
    iris_dataset = load_iris()
    # print(type(iris_dataset)) # <class 'sklearn.utils.Bunch'> similar to dict
    # print('keys of iris_dataset:',iris_dataset.keys()) 
    # dict_keys(['data', 'target', 'frame', 'DESCR', 'feature_names', 'target_names', 'filename', 'data_module'])
    # print('description of iris_dataset:',iris_dataset['DESCR'])
    # print('type of data:',type(iris_dataset['data'])) # <class 'numpy.ndarray'>
    # print('shape of data:',iris_dataset['data'].shape) # (150, 4)
    # print('head of data:',iris_dataset['data'][:5])
    # print('frame of data:',iris_dataset['frame'])
    # print('data_module of data:',iris_dataset['data_module'])
    # print('filename of data:',iris_dataset['filename'])
    return iris_dataset, iris_dataset['data'], iris_dataset['target']

def IrisData_split(data, target):
    from sklearn.model_selection import train_test_split
    # split data into train and test 8:2
    x_train, x_test, y_train, y_test = train_test_split(
        data, 
        target, 
        test_size=0.25, 
        random_state=42)
    # print('x_train shape:',x_train.shape)
    # print('x_test shape:',x_test.shape)
    # print('y_train shape:',y_train.shape)
    # print('y_test shape:',y_test.shape)
    return x_train, x_test, y_train, y_test

def IrisData_visualization(data, target, feature_names):
    import pandas as pd
    import matplotlib.pyplot as plt
    dataframe = pd.DataFrame(data, columns=feature_names)
    # transform data to pd.DataFrame then use pd.plotting.scatter_matrix
    pd.plotting.scatter_matrix(
        dataframe, 
        c=target, 
        figsize=(15, 15), 
        marker='o', 
        hist_kwds={'bins': 20}, 
        s=60, 
        alpha=.8, 
        diagonal='hist'
        )
    plt.show()

def IrisData_knc(x_train, y_train, x_test, y_test):
    from sklearn.neighbors import KNeighborsClassifier
    import numpy as np
    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(x_train, y_train)# knc model training
    # evaluate knc model
    y_pred = knn.predict(x_test)
    print('test set score: {:.2f}'.format(np.mean(y_pred == y_test)))
    print('test set accuracy: {:.2f}'.format(knn.score(x_test, y_test)))






