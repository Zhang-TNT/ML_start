# forge 二分类算法
import mglearn.datasets


def ForgeData_read():
    import mglearn
    X_forge, y_forge = mglearn.datasets.make_forge()
    # print('type of X_forge:', type(X_forge))
    # print('shape of X_forge:', X_forge.shape)
    # print('shape of y_forge:', y_forge.shape)
    return X_forge, y_forge

def ForgeData_visualization(X_data, y_data):
    import matplotlib.pyplot as plt
    mglearn.discrete_scatter(X_data[:,0], X_data[:,1], y_data)
    plt.legend(['Class 0', 'Class 1'])
    plt.xlabel("Feature 0")
    plt.ylabel("Feature 1")
    plt.show()

# wave 回归算法
def WaveData_read():
    import mglearn
    X_wave, y_wave = mglearn.datasets.make_wave(n_samples=40)
    # print('type of X_wave:', type(X_wave))
    # print('shape of X_wave:', X_wave.shape)
    # print('shape of y_wave:', y_wave.shape)
    return X_wave, y_wave

def WaveData_visualization(X_data, y_data):
    import matplotlib.pyplot as plt
    plt.plot(X_data, y_data, 'o')
    plt.xlabel("Feature")
    plt.ylabel("Target")
    plt.show()

# breast cancer 分类算法
'''
thirty features
two categories: malignant, benign
'''
def BreastCancerData_read():
    from sklearn.datasets import load_breast_cancer
    import numpy as np
    cancer_dataset = load_breast_cancer()
    # print(type(cancer_dataset)) # <class 'sklearn.utils.Bunch'>
    # print('keys of cancer_dataset:',cancer_dataset.keys())
    # dict_keys(['data', 'target', 'frame', 'DESCR', 'feature_names', 'target_names', 'filename', 'data_module'])
    # print('description of cancer_dataset:',cancer_dataset['DESCR'])
    # print('type of data:',type(cancer_dataset['data'])) # <class 'numpy.ndarray'>
    # print('shape of data:',cancer_dataset['data'].shape) # (569, 30)
    # print('shape of feature_names:', cancer_dataset['feature_names'].shape)
    # print('head of data:',cancer_dataset['data'][:5])
    # print('frame of data:',cancer_dataset['frame'])
    # print('data_module of data:',cancer_dataset['data_module'])
    # print('filename of data:',cancer_dataset['filename'])
    print('Sample counts per class:\n{}'.format(
        {n:v for n, v in zip(cancer_dataset['target_names'], np.bincount(cancer_dataset['target']))}
    ))
    return cancer_dataset, cancer_dataset['data'], cancer_dataset['target']


# diabetes 回归算法
'''

'''
def DiabetesData_read():
    from sklearn.datasets import load_diabetes
    diabetes_dataset = load_diabetes()
    # print(type(diabetes_dataset)) # <class 'sklearn.utils.Bunch'>
    # print('keys of diabetes_dataset:',diabetes_dataset.keys())
    # dict_keys(['data', 'target', 'frame', 'DESCR', 'feature_names', 'target_names', 'filename', 'data_module'])
    # print('description of diabetes_dataset:',diabetes_dataset['DESCR'])
    # print('type of data:',type(diabetes_dataset['data'])) # <class 'numpy.ndarray'>
    print('shape of data:',diabetes_dataset['data'].shape) # (442, 10)
    # print('head of data:',diabetes_dataset['data'][:5])
    # print('frame of data:',diabetes_dataset['frame'])
    # print('data_module of data:',diabetes_dataset['data_module'])
    # print('filename of data:',diabetes_dataset['filename'])
    return diabetes_dataset, diabetes_dataset['data'], diabetes_dataset['target']



