# third party import
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.datasets import load_breast_cancer
from sklearn.datasets import fetch_california_housing
from sklearn.datasets import make_blobs
from sklearn.datasets import make_moons
from sklearn.datasets import make_circles
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import export_graphviz
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from IPython.display import display
import mglearn
import graphviz
import seaborn as sns
import matplotlib.pyplot as plt
# local import


def dataSet_load():
    # 导出Iris数据集，load_iris()返回的iris对象是一个Bunch对象，类似字典
    dataset = load_iris()
    return dataset


def dataSet_info(dataSet):
    # 输出数据集属性
    print('Keys of dataSet: \n{}'.format(dataSet.keys()))
    # 输出数据集各健对应的值
    print('First five rows of dataSet: \n{}'.format(dataSet['data'][: 5]))
    print('Shape of dataSet: {}'.format(dataSet['data'].shape))
    print('Type of dataSet: {}'.format(type(dataSet['data'])))
    print('Target of dataSet: \n{}'.format(dataSet['target']))
    print('Frame of dataSet: \n{}'.format(dataSet['frame']))
    print('Target_names of dataSet: \n{}'.format(dataSet['target_names']))
    print('Descr of dataSet: \n{}'.format(dataSet['DESCR']))
    print('Feature_names of dataSet: \n{}'.format(dataSet['feature_names']))
    print('Filename of dataSet: \n{}'.format(dataSet['filename']))
    print('Data_module of dataSet: \n{}'.format(dataSet['data_module']))


def dataSet_split(dataSet):
    # 将数据集的75%划分为训练集余下划分为测试集，其中X表示数据，y表示标签/结果
    X_train, X_test, y_train, y_test = train_test_split(
        dataSet['data'], dataSet['target'], random_state=0)
    # 参数random_state指定随机数生成器种子
    return X_train, X_test, y_train, y_test


def dataSet_frame(dataSet):
    # 将数据集转化为dataframe
    dataSetframe = pd.DataFrame(
        dataSet['data'], columns=dataSet['feature_names'])
    # 添加一列数据集的标签
    class_mapping = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
    dataSetframe['species'] = dataSet['target']
    dataSetframe['species'] = dataSetframe['species'].map(class_mapping)
    return dataSetframe


def dataSet_plot(dataSetframe):
    # 数据可视化
    sns.set_style('whitegrid')  # 设置绘图风格
    # 绘制密度图，反映数据集中特征变量分布情况
    dataSetframe.plot(kind='kde')
    plt.title('iris数据比例图', fontproperties='SimSun')
    plt.xlabel('Value/cm')
    plt.ylabel('Percentage/%')

    dataSetframe.plot.area(stacked=False)  # 绘制面积图
    plt.title('iris数据集四种特征分布图', fontproperties='SimSun')
    plt.xlabel('Sample/n')
    plt.ylabel('Value/cm')
    # 绘制四特征直方图
    dataSetframe.hist(bins=20)
    # 绘制箱型图
    sns.boxplot(y=dataSetframe['sepal length (cm)'], x=dataSetframe['species'])
    dataSetframe.plot(kind='box')
    # # 绘制小提琴图
    # fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    # sns.violinplot(y=dataSetframe['Petal.Length'], data=dataSetframe, palette="ocean_r", ax=axes[0, 0])
    # sns.violinplot(y=dataSetframe['Petal.Width'], data=dataSetframe, palette="ocean_r", ax=axes[0, 1])
    # sns.violinplot(y=dataSetframe['Sepal.Length'], data=dataSetframe, palette="ocean_r", ax=axes[1, 0])
    # sns.violinplot(y=dataSetframe['Sepal.Width'], data=dataSetframe, palette="ocean_r", ax=axes[1, 1])
    # # 绘制两两特征图
    # pd.plotting.scatter_matrix(dataSetframe, marker='o', hist_kwds={'bins': 20}, s=20, alpha=.9,
    #                            c='species', cmap=mglearn.cm3)
    # 绘制两两特征图
    sns.pairplot(dataSetframe, diag_kind='kde',
                 markers=["o", "s", "D"], hue='species')
    plt.show()


def kNN_classify():
    # 实例化KNeighborsClassifier类，返回一个类对象
    kNN_object = KNeighborsClassifier(n_neighbors=5)
    return kNN_object


def testData_generation():
    # sklearn中的KNN分类器只接受numpy数组，所以这里需要先将数据转化为numpy数组，输入数组必须是二维数组
    new_data = np.array([[15, 12.9, 11, 10.2]])
    return new_data


def IrisDataset_multiClass(X_train, X_test, y_train, y_test):
    gbrt = GradientBoostingClassifier(random_state=0, learning_rate=0.01)
    gbrt.fit(X_train, y_train)
    # decision_function
    print('Decision function shape: {}'.format(
        gbrt.decision_function(X_test).shape))
    print('Decision function: {}'.format(gbrt.decision_function(X_test)[:6]))
    print('Argmax of decision function:\n{}'.format(
        np.argmax(gbrt.decision_function(X_test), axis=1)))
    print('Predictions:\n{}'.format(gbrt.predict(X_test)))
    # predict_proba
    print('Predicted probabilities: \n{}'.format(
        gbrt.predict_proba(X_test)[:6]))
    print('Argmax of predicted probabilities:\n{}'.format(
        np.argmax(gbrt.predict_proba(X_test), axis=1)))
    print('Predictions:\n{}'.format(gbrt.predict(X_test)))


def forgeDataset_load():
    # 导入forge数据集
    dataset = mglearn.datasets.make_forge()
    print(type(dataset))
    # print('forge dataset: \n{}'.format(dataset)) # 元组类型数据
    forge_data = dataset[0]
    forge_target = dataset[1]
    plt.figure()
    mglearn.discrete_scatter(forge_data[:, 0], forge_data[:, 1], forge_target)
    plt.legend(["Class 0", "Class 1"], loc=4)
    plt.xlabel("First feature")
    plt.ylabel("Second feature")
    plt.show()
    return forge_data, forge_target


def forgeDataset_split(dataset, target):
    data_Train, data_Test, target_Train, target_Test = train_test_split(
        dataset, target)
    return data_Train, data_Test, target_Train, target_Test


def forgeDataset_knnClassifier():
    clf = KNeighborsClassifier(n_neighbors=3)
    return clf


def forgeDataset_knnVisualization(dataset, target):
    fig, axes = plt.subplots(1, 3, figsize=(10, 3))
    for n_neighbor, ax in zip([1, 3, 9], axes):
        # the fit method returns the object self, so we can instantiate
        clf = KNeighborsClassifier(n_neighbors=n_neighbor).fit(dataset, target)
        mglearn.plots.plot_2d_separator(
            clf, dataset, fill=True, eps=0.5, ax=ax, alpha=.4)
        mglearn.discrete_scatter(dataset[:, 0], dataset[:, 1], target, ax=ax)
        ax.set_title("k = {}".format(n_neighbor))
        ax.set_xlabel("feature 0")
        ax.set_ylabel("feature 1")
    axes[0].legend(loc=3)
    plt.show()


def forgeDataset_linearRegression(dataset, target):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3))
    for model, ax in zip([LinearSVC(), LogisticRegression()], axes):
        clf = model.fit(dataset, target)
        mglearn.plots.plot_2d_separator(
            clf, dataset, fill=True, eps=0.5, ax=ax, alpha=.4)
        mglearn.discrete_scatter(dataset[:, 0], dataset[:, 1], target, ax=ax)
        ax.set_title("{}".format(clf.__class__.__name__))
        ax.set_xlabel("feature 0")
        ax.set_ylabel("feature 1")
    axes[0].legend()
    plt.show()


def forgeDataset_SVC(X, y):
    svm = SVC(kernel='rbf', C=10.0, gamma=0.7).fit(X, y)
    mglearn.plots.plot_2d_separator(svm, X, fill=True, eps=0.5, alpha=.7)
    mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
    # draw the support vector
    sv = svm.support_vectors_
    sv_labels = svm.dual_coef_.ravel() > 0
    mglearn.discrete_scatter(
        sv[:, 0], sv[:, 1], sv_labels, s=15, markeredgewidth=3)
    plt.title('svm')
    plt.xlabel('Feature 0')
    plt.ylabel('Feature 1')
    plt.show()


def waveDataset_load():
    X, y = mglearn.datasets.make_wave(n_samples=40)
    plt.plot(X, y, 'o')
    plt.ylim(-3, 3)
    plt.xlabel("Feature")
    plt.ylabel("Target")
    plt.show()
    return X, y


def waveDataset_split(dataset, target):
    data_Train, data_Test, target_Train, target_Test = train_test_split(
        dataset, target)
    return data_Train, data_Test, target_Train, target_Test


def waveDataset_knnRegressor():
    reg = KNeighborsRegressor(n_neighbors=3)
    return reg


def waveDataset_knnVisualization(dataset_train, dataset_test, target_train, target_test):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    # 在二维空间上创建1000个数据点，-3到3之间均匀分布，形状为（1000, 1）
    line = np.linspace(-3, 3, 1000).reshape(-1, 1)
    for n_neighbor, ax in zip([1, 3, 9], axes):
        # the fit method returns the object self, so we can instantiate
        reg = KNeighborsRegressor(n_neighbors=n_neighbor)
        reg.fit(dataset_train, target_train)
        ax.plot(line, reg.predict(line))
        ax.plot(dataset_train, target_train, '^',
                c=mglearn.cm2(0), markersize=8)
        ax.plot(dataset_test, target_test, 'v', c=mglearn.cm2(1), markersize=8)
        ax.set_title(
            "k = {}\n train score: {:.2f} test score: {:.2f}".format(
                n_neighbor, reg.score(dataset_train, target_train),
                reg.score(dataset_test, target_test)))
        ax.set_xlabel("Feature")
        ax.set_ylabel("Target")

    axes[0].legend(["Model predictions", "Training data/target",
                   "Test data/target"], loc="best")
    plt.show()


def waveDataset_linearRegression(dataset, target):
    # mglearn.plots.plot_linear_regression_wave()
    lr = LinearRegression().fit(dataset, target)
    print("lr.coef_: {}".format(lr.coef_))              # 权重或斜率
    print("lr.intercept_: {}".format(lr.intercept_))    # 截距
    return lr


def breastCancerdataset_load():
    dataset = load_breast_cancer()
    print(type(dataset))
    print('shape of breast_cancer: \n{}'.format(dataset.data.shape))
    print("keys of breast_cancer: \n{}".format(dataset.keys()))
    print('Sample counts per class: \n{}'.format(
        {n: v for n, v in zip(dataset.target_names, np.bincount(dataset.target))}))
    print('Feature names: \n{}'.format(dataset.feature_names))
    print('Head of dataset: \n{}'.format(dataset.data[:5]))
    return dataset


def breastCancerdataset_visualization(dataset):
    datasetFrame = pd.DataFrame(dataset.data, columns=dataset.feature_names)
    # add target
    class_mapping = {0: 'malignant', 1: 'benign'}
    datasetFrame['target'] = dataset.target
    datasetFrame['target'] = datasetFrame['target'].map(class_mapping)
    # radius
    radius = datasetFrame[['mean radius',
                           'radius error', 'worst radius', 'target']]
    pp1 = sns.pairplot(radius, hue='target',
                       diag_kind='kde', markers=['o', 's'])
    pp1.fig.suptitle('radius aspects')
    plt.show()
    # texture
    texture = datasetFrame[['mean texture',
                            'texture error', 'worst texture', 'target']]
    pp2 = sns.pairplot(texture, hue='target',
                       diag_kind='kde', markers=['v', '^'])
    pp2.fig.suptitle('texture aspects')
    plt.show()
    # perimeter
    perimeter = datasetFrame[['mean perimeter',
                              'perimeter error', 'worst perimeter', 'target']]
    pp3 = sns.pairplot(perimeter, hue='target',
                       diag_kind='kde', markers=['<', '>'])
    pp3.fig.suptitle('perimeter aspects')
    plt.show()
    # area
    area = datasetFrame[['mean area', 'area error', 'worst area', 'target']]
    pp4 = sns.pairplot(area, hue='target', diag_kind='kde', markers=['o', 's'])
    pp4.fig.suptitle('area aspects')
    plt.show()
    # smoothness
    smoothness = datasetFrame[['mean smoothness',
                               'smoothness error', 'worst smoothness', 'target']]
    pp5 = sns.pairplot(smoothness, hue='target',
                       diag_kind='kde', markers=['o', 's'])
    pp5.fig.suptitle('smoothness aspects')
    plt.show()
    # compactness
    compactness = datasetFrame[['mean compactness',
                                'compactness error', 'worst compactness', 'target']]
    pp6 = sns.pairplot(compactness, hue='target',
                       diag_kind='kde', markers=['o', 's'])
    pp6.fig.suptitle('compactness aspects')
    plt.show()
    # concavity
    concavity = datasetFrame[['mean concavity',
                              'concavity error', 'worst concavity', 'target']]
    pp7 = sns.pairplot(concavity, hue='target',
                       diag_kind='kde', markers=['o', 's'])
    pp7.fig.suptitle('concavity aspects')
    plt.show()
    # concave points
    concave_points = datasetFrame[[
        'mean concave points', 'concave points error', 'worst concave points', 'target']]
    pp8 = sns.pairplot(concave_points, hue='target',
                       diag_kind='kde', markers=['o', 's'])
    pp8.fig.suptitle('concave points aspects')
    plt.show()
    # symmetry
    symmetry = datasetFrame[['mean symmetry',
                             'symmetry error', 'worst symmetry', 'target']]
    pp9 = sns.pairplot(symmetry, hue='target',
                       diag_kind='kde', markers=['o', 's'])
    pp9.fig.suptitle('symmetry aspects')
    plt.show()
    # fractal dimension
    fractal_dimension = datasetFrame[[
        'mean fractal dimension', 'fractal dimension error', 'worst fractal dimension', 'target']]
    pp10 = sns.pairplot(fractal_dimension, hue='target',
                        diag_kind='kde', markers=['o', 's'])
    pp10.fig.suptitle('fractal dimension')
    plt.show()


def breastCancerdataset_split(dataset):
    data_Train, data_Test, target_Train, target_Test = train_test_split(
        dataset.data, dataset.target, random_state=0)
    print("data_Train shape: {}".format(data_Train.shape))
    print("data_Test shape: {}".format(data_Test.shape))
    print("target_Train shape: {}".format(target_Train.shape))
    print("target_Test shape: {}".format(target_Test.shape))
    return data_Train, data_Test, target_Train, target_Test


def breastCancerdataset_knnClassifier(X_train, y_train, X_test, y_test):
    training_accuracy = []
    test_accuracy = []
    # try n_neighbor from 1 to 10
    neighbor_setting = range(1, 11)
    for k in neighbor_setting:
        # build the model
        clf = KNeighborsClassifier(n_neighbors=k)
        clf.fit(X_train, y_train)
        # record training set accuracy
        training_accuracy.append(clf.score(X_train, y_train))
        # record generalization accuracy
        test_accuracy.append(clf.score(X_test, y_test))
    # plot the results
    plt.plot(neighbor_setting, training_accuracy, label="training accuracy")
    plt.plot(neighbor_setting, test_accuracy, label="test accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("n_neighbors")
    plt.legend()
    plt.show()


def breastCancerdataset_linearRegression(dataset, target):
    lr = LinearRegression().fit(dataset, target)
    print("lr.coef_: {}".format(lr.coef_))              # 权重或斜率
    print("lr.intercept_: {}".format(lr.intercept_))    # 截距
    return lr


def breastCancerdataset_RidgeRegression(dataset, target):
    ridge = Ridge(alpha=1).fit(dataset, target)
    ridge10 = Ridge(alpha=10).fit(dataset, target)
    ridge01 = Ridge(alpha=0.1).fit(dataset, target)
    return ridge, ridge10, ridge01


def breastCancerdataset_RidgeVisualization(ridge, ridge10, ridge01, lr):
    plt.plot(ridge.coef_, 's', label="Ridge alpha=1")
    plt.plot(ridge10.coef_, '^', label="Ridge alpha=10")
    plt.plot(ridge01.coef_, 'v', label="Ridge alpha=0.1")
    plt.plot(lr.coef_, 'o', label='LinearRegression')
    plt.xlabel("Coefficient index")
    plt.ylabel("Coefficient magnitude")
    plt.hlines(0, 0, len(lr.coef_))
    plt.legend()
    plt.show()


def breastCancerdataset_learningCurve():
    mglearn.plots.plot_ridge_n_samples()
    plt.show()


def breastCancerdataset_lassoRegression(dataset, target):
    lasso = Lasso().fit(dataset, target)
    lasso001 = Lasso(alpha=0.01, max_iter=100000).fit(dataset, target)
    lasso00001 = Lasso(alpha=0.0001, max_iter=100000).fit(dataset, target)
    print('number of features used: {}'.format(np.sum(lasso.coef_ != 0)))
    print('number of features used: {}'.format(np.sum(lasso001.coef_ != 0)))
    print('number of features used: {}'.format(np.sum(lasso00001.coef_ != 0)))
    return lasso, lasso001, lasso00001


def breastCancerdataset_lassoVisualization(lasso, lasso001, lasso00001, lr):
    plt.plot(lasso.coef_, 's', label="Lasso alpha=1")
    plt.plot(lasso001.coef_, '^', label="Lasso alpha=0.01")
    plt.plot(lasso00001.coef_, 'v', label='Lasso alpha=0.0001')
    plt.plot(lr.coef_, 'o', label='LinearRegression')
    plt.xlabel("Coefficient index")
    plt.ylabel("Coefficient magnitude")
    plt.hlines(0, 0, len(lasso.coef_))
    plt.legend()
    plt.show()


def breastCancerdataset_logisticRegression(X_train, y_train, X_test, y_test):
    logreg = LogisticRegression().fit(X_train, y_train)
    logreg100 = LogisticRegression(C=100).fit(X_train, y_train)
    logreg001 = LogisticRegression(C=0.01).fit(X_train, y_train)
    print('trainingSet C=1 score: {:.3f}'.format(
        logreg.score(X_train, y_train)))
    print('testSet C=1 score: {:.3f}'.format(logreg.score(X_test, y_test)))
    print('trainingSet C=100 score: {:.3f}'.format(
        logreg100.score(X_train, y_train)))
    print('testSet C=100 score: {:.3f}'.format(
        logreg100.score(X_test, y_test)))
    print('trainingSet C=0.01 score: {:.3f}'.format(
        logreg001.score(X_train, y_train)))
    print('testSet C=0.01 score: {:.3f}'.format(
        logreg001.score(X_test, y_test)))
    return logreg, logreg100, logreg001


def breastCancerdataset_logisticRegressionVisualization(dataset, logreg, logreg100, logreg001):
    plt.plot(logreg.coef_.T, 'o', label="C=1")
    plt.plot(logreg100.coef_.T, '^', label="C=100")
    plt.plot(logreg001.coef_.T, 'v', label="C=0.01")
    plt.xticks(range(dataset.data.shape[1]),
               dataset.feature_names, rotation=90)
    plt.hlines(0, 0, dataset.data.shape[1])
    plt.xlabel("Feature")
    plt.ylabel("Coefficient magnitude")
    plt.legend()
    plt.show()


def breastCancerdataset_decisionTreeClassifier(X_train, y_train, X_test, y_test):
    tree = DecisionTreeClassifier(random_state=0, max_depth=4)
    tree.fit(X_train, y_train)
    print("Accuracy on training set: {:.3f}".format(
        tree.score(X_train, y_train)))
    print("Accuracy of test set: {:.3f}".format(tree.score(X_test, y_test)))
    return tree


def breastCancerdataset_decisionTreeGraphviz(tree, dataset):
    export_graphviz(tree, out_file="tree.dot", class_names=["malignant", "benign"],
                    feature_names=dataset.feature_names, impurity=False, filled=True)
    with open("tree.dot") as f:
        dot_graph = f.read()
    g = graphviz.Source(dot_graph)
    g.render("tree")


def breastCancerdataset_treeFeatureImportance(tree, dataset):
    print("Feature importances:\n{}".format(tree.feature_importances_))
    n_features = dataset.data.shape[1]
    plt.barh(range(n_features), tree.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), dataset.feature_names)
    plt.xlabel("Feature importance")
    plt.ylabel("Feature")
    plt.ylim(-1, n_features)
    plt.show()


def breastCancerdataset_randomForestClassifier(X_train, y_train, X_test, y_test):
    forest = RandomForestClassifier(n_estimators=100, random_state=0)
    forest.fit(X_train, y_train)
    print("Accuracy on training set: {:.3f}".format(
        forest.score(X_train, y_train)))
    print('Accuracy on test set: {:.3f}'.format(forest.score(X_test, y_test)))
    return forest


def breastCancerdataset_GradientBoostingClassifier(X_train, y_train, X_test, y_test):
    gbrt = GradientBoostingClassifier(random_state=0, max_depth=1)
    gbrt.fit(X_train, y_train)
    print('Accuracy on training set: {:.3f}'.format(
        gbrt.score(X_train, y_train)))
    print('Accuracy on test set: {:.3f}'.format(gbrt.score(X_test, y_test)))
    return gbrt


def breastCancerdataset_SVM(X_train, y_train, X_test, y_test):
    svc = SVC(C=100)
    svc.fit(X_train, y_train)
    print('Accuracy on training set: {:.3f}'.format(
        svc.score(X_train, y_train)))
    print('Accuracy on test set: {:.3f}'.format(svc.score(X_test, y_test)))
    plt.plot(X_train.min(axis=0), 'o', label="min")
    plt.plot(X_train.max(axis=0), 'v', label="max")
    plt.xlabel("Feature index")
    plt.ylabel("Feature magnitude")
    plt.yscale("log")
    plt.show()
    return svc


def breastCancerdataset_prprocessing(data):
    min_on_dataset = data.min(axis=0)
    # range of each feature
    range_on_dataset = (data - min_on_dataset).max(axis=0)
    # standardize dataset
    data_std = (data - min_on_dataset) / range_on_dataset
    return data_std


def breastCancerdataset_neuralNetworkClassifier(X_train, y_train, X_test, y_test):
    mlp_train = MLPClassifier(random_state=0).fit(X_train, y_train)
    print('Accuracy on training set: {:.3f}'.format(
        mlp_train.score(X_train, y_train)))
    print('Accuracy on test set: {:.3f}'.format(
        mlp_train.score(X_test, y_test)))
    return mlp_train


def breastCancerdataset_dataProcessing(X_train, X_test):
    mean_on_train = X_train.mean(
        axis=0)                                # 每个特征的平均值
    # 每个特征的标准差
    std_on_train = X_train.std(axis=0)
    X_train_scaled = (X_train - mean_on_train) / \
        std_on_train           # 数据减去平均值除以标准差
    # 数据减去平均值除以标准差（使用训练集的平均值和标准差）
    X_test_scaled = (X_test - mean_on_train) / std_on_train
    return X_train_scaled, X_test_scaled


def breastCancerdataset_scaledDatasetresult(X_train_scaled, X_test_scaled, y_train, y_test, dataset):
    mlp = MLPClassifier(random_state=0, max_iter=1000,
                        alpha=1).fit(X_train_scaled, y_train)
    print('Accuracy on scaled training set: {:.3f}'.format(
        mlp.score(X_train_scaled, y_train)))
    print('Accuracy on scaled test set: {:.3f}'.format(
        mlp.score(X_test_scaled, y_test)))
    plt.figure(figsize=(20, 5))
    plt.imshow(mlp.coefs_[0], interpolation='none', cmap='viridis')
    plt.yticks(range(30), dataset.feature_names)
    plt.xlabel("Columns in weight matrix")
    plt.ylabel("Input feature")
    plt.colorbar()
    plt.show()


def breastCancerdataset_preprocessingMinMaxScaler(X_train, X_test):
    scaler = MinMaxScaler()
    # using fit() to fit the scaler
    scaler.fit(X_train)
    # using transform() to scale the training data
    X_train_scaled = scaler.transform(X_train)
    print('X_train_scaled shape: {}'.format(X_train_scaled.shape))
    print(
        'per-feature minimum before scaling: \n{}'.format(X_train.min(axis=0)))
    print(
        'per-feature maximum before scaling: \n{}'.format(X_train.max(axis=0)))
    print('per-feature minimum after scaling: \n{}'.format(
        X_train_scaled.min(axis=0)))
    print('per-feature maximum after scaling: \n{}'.format(
        X_train_scaled.max(axis=0)))
    # using transform() to scale the test data
    X_test_scaled = scaler.transform(X_test)
    print('per-feature minimum after scaling: \n{}'.format(X_test_scaled.min(
        axis=0)))
    print('per-feature maximum after scaling: \n{}'.format(X_test_scaled.max(
        axis=0)))


def californiaHousingdataset_load():
    dataset = fetch_california_housing()
    print("keys of california_housing: \n{}".format(dataset.keys()))
    print("shape of california_housing: \n{}".format(dataset.data.shape))
    return dataset


def californiaHousingdataset_split(dataset):
    data_Train, data_Test, target_Train, target_Test = train_test_split(
        dataset.data, dataset.target, random_state=0)
    return data_Train, data_Test, target_Train, target_Test


def caliHousdataset_linearRegression(dataset, target):
    lr = LinearRegression().fit(dataset, target)
    print("lr.coef_: {}".format(lr.coef_))              # 权重或斜率
    print("lr.intercept_: {}".format(lr.intercept_))    # 截距
    return lr


def caliHousdataset_RidgeRegression(dataset, target):
    ridge = Ridge(alpha=10).fit(dataset, target)
    print("ridge.coef_: {}".format(ridge.coef_))              # 权重或斜率
    print("ridge.intercept_: {}".format(ridge.intercept_))    # 截距
    return ridge


def caliHousdataset_visualization(ridge):
    plt.plot(ridge.coef_, 's', label='Ridge alpha = 1')
    plt.xlabel('Coefficient index')
    plt.ylabel('Coefficient magnitude')
    plt.hlines(0, 0, len(ridge.coef_))
    plt.ylim(-25, 25)
    plt.legend()
    plt.show()


def blobsDataset_load():
    X, y = make_blobs(random_state=4, centers=5, n_samples=50, cluster_std=2)
    # y = y % 2
    print('shape of X: {}'.format(X.shape))
    print('shape of y: {}'.format(y.shape))
    return X, y


def blobsDataset_dataVisualization(X, y):
    # plt.scatter(X[:, 0], X[:, 1], c=y, s=15, cmap='rainbow')
    mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
    plt.title('blobs dataset')
    plt.xlabel("Feature 0")
    plt.ylabel("Feature 1")
    # plt.legend(["Class 0", "Class 1", "Class 2", "Class 3"])
    plt.show()


def blobsDataset_linearSVC_classifier(X, y):
    linear_svm = LinearSVC().fit(X, y)
    print('Coefficient shape: {}'.format(linear_svm.coef_.shape))
    print('Intercept shape: {}'.format(linear_svm.intercept_.shape))
    return linear_svm


def blobsDataset_linearSVC_classifierVisualization(linear_svm, X, y):
    mglearn.plots.plot_2d_separator(linear_svm, X)
    mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
    # line = np.linspace(-15, 15)
    # plt.plot(line, -(line * linear_svm.coef_[1, 0] + linear_svm.intercept_[0])
    #          / linear_svm.coef_[1, 1], 'k-')
    # for coef, intercept, color in zip(linear_svm.coef_, linear_svm.intercept_, ['b', 'r', 'g']):
    #     plt.plot(line, -(line * coef[0] + intercept) / coef[1], color)
    plt.xlabel("Feature 0")
    plt.ylabel("Feature 1")
    plt.title('linear_svm classifier')
    # plt.legend(['Class 0', 'Class 1', 'Class 2', 'Line class 0', 'Line class 1', 'Line class 2'], loc=(1.01, 0.3))
    plt.show()


def blobsDataset_linearSVC_2dclassifierVisualization(linear_svm, X, y):
    # mglearn.plots.plot_2d_classification(linear_svm, X, fill=True, alpha=.7)
    mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
    line = np.linspace(-15, 15)
    plt.plot(line, -(line * linear_svm.coef_[0, 0] +
             linear_svm.intercept_[0]) / linear_svm.coef_[0, 1], 'k-')
    # for coef, intercept, color in zip(linear_svm.coef_, linear_svm.intercept_, ['b', 'r', 'g']):
    #     plt.plot(line, -(line * coef[0] + intercept) / coef[1], color)
    plt.legend(['Class 0', 'Class 1', 'Class 2'])
    plt.xlabel("Feature 0")
    plt.ylabel("Feature 1")
    plt.show()


def blobsDataset_3dExpand(X, y):
    X_new = np.hstack([X, X[:, 1:] ** 2])
    print('X_new.shape: {}'.format(X_new.shape))
    # 3d visualization
    ax = plt.figure().add_subplot(projection='3d')
    ax.view_init(elev=-150, azim=-30)
    # all the points y == 0
    mask = y == 0
    ax.scatter(X_new[mask, 0],
               X_new[mask, 1],
               X_new[mask, 2],
               c='b', s=60, edgecolors='black')
    ax.scatter(X_new[~mask, 0],
               X_new[~mask, 1],
               X_new[~mask, 2],
               c='r', s=60, marker='^')
    ax.set_xlabel("Feature 0")
    ax.set_ylabel("Feature 1")
    ax.set_zlabel("Feature 1 ** 2")
    plt.title('3d visualization of blobs dataset')
    plt.show()
    return X_new


def blobsDataset_linearSVC_3dclassifier(X, y):
    linear_svm_3d = LinearSVC().fit(X, y)
    return linear_svm_3d


def blobsDataset_linearSVC_3dclassifierVisualization(linear_svm_3d, X, y):
    coef, intercept = linear_svm_3d.coef_.ravel(), linear_svm_3d.intercept_
    print('shape of coef: {}'.format(coef.shape))
    print('shape of intercept: {}'.format(intercept.shape))
    # display linear decision boundary
    ax = plt.figure().add_subplot(projection='3d')
    ax.view_init(elev=-150, azim=-30)
    xx = np.linspace(X[:, 0].min() - 2, X[:, 0].max() + 2, 50)
    yy = np.linspace(X[:, 1].min() - 2, X[:, 1].max() + 2, 50)
    XX, YY = np.meshgrid(xx, yy)
    print('shape of XX: {}'.format(XX.shape))
    print('shape of YY: {}'.format(YY.shape))
    ZZ = (coef[0] * XX + coef[1] * YY + intercept[0]) / -coef[2]
    ax.plot_surface(XX, YY, ZZ, rstride=8, cstride=8, alpha=0.3, color='b')
    mask = y == 0
    ax.scatter(X[mask, 0],
               X[mask, 1],
               X[mask, 2], c='b', s=60)
    ax.scatter(X[~mask, 0],
               X[~mask, 1],
               X[~mask, 2], c='r',
               marker='^', s=60)
    ax.set_xlabel("Feature 0")
    ax.set_ylabel("Feature 1")
    ax.set_zlabel("Feature 1 ** 2")
    plt.title('3d visualization of blobs dataset')
    plt.show()


def blobsDataset_linearSVC_orignalFeature(linear_svm_3d, X, y):
    xx = np.linspace(X[:, 0].min() - 2, X[:, 0].max() + 2, 50)
    yy = np.linspace(X[:, 1].min() - 2, X[:, 1].max() + 2, 50)
    XX, YY = np.meshgrid(xx, yy)
    ZZ = YY ** 2
    dec = linear_svm_3d.decision_function(
        np.c_[XX.ravel(), YY.ravel(), ZZ.ravel()])
    plt.contourf(XX, YY, dec.reshape(XX.shape), levels=[dec.min(), 0, dec.max()],
                 cmap=mglearn.cm2, alpha=0.5)
    mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
    plt.xlabel('Feature 0')
    plt.ylabel('Feature 1')
    plt.show()


def blobsDataset_split(X):
    X_train, X_test = train_test_split(X, random_state=5, test_size=.1)
    # plot training and test sets
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    axes[0].scatter(X_train[:, 0], X_train[:, 1],
                    c=mglearn.cm2(0), label="training set", s=60)
    axes[0].scatter(X_test[:, 0], X_test[:, 1], c=mglearn.cm2(
        1), label="test set", s=60, marker='^')
    axes[0].legend(loc="upper left")
    axes[0].set_title('Orignal dataset')
    # scale the data using MinMaxScaler
    scaler = MinMaxScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    # plot the scaled training and test sets
    axes[1].scatter(X_train_scaled[:, 0], X_train_scaled[:, 1],
                    c=mglearn.cm2(0), label='training set', s=60)
    axes[1].scatter(X_test_scaled[:, 0], X_test_scaled[:, 1],
                    c=mglearn.cm2(1), label='test set', s=60, marker='^')
    axes[1].set_title('Scaled dataset')
    # rescale the test set separately
    test_scaler = MinMaxScaler()
    test_scaler.fit(X_test)
    X_test_scaled_badly = test_scaler.transform(X_test)
    # show new scalings
    axes[2].scatter(X_train_scaled[:, 0], X_train_scaled[:, 1],
                    c=mglearn.cm2(0), label='training set', s=60)
    axes[2].scatter(X_test_scaled_badly[:, 0], X_test_scaled_badly[:, 1],
                    c=mglearn.cm2(1), label='test set', s=60, marker='^')
    axes[2].set_title('Improperly scaled dataset')
    for ax in axes:
        ax.set_xlabel("Feature 0")
        ax.set_ylabel("Feature 1")
    plt.show()


def sampleDataset_load():
    X = np.array([[0, 1, 0, 1], [1, 0, 1, 1], [0, 0, 0, 1], [1, 0, 1, 0]])
    y = np.array([0, 1, 0, 1])
    return X, y


def sampleDataset_unique(X, y):
    counts = {}
    for label in np.unique(y):
        counts[label] = X[y == label].sum(axis=0)
    print("Feature counts: {}".format(counts))


def tree_example():
    tree = mglearn.plots.plot_tree_not_monotone()
    display(tree)
    plt.show()


def ramPriceDataset_load():
    dataset = pd.read_csv("./dataSet/ram_price.csv")
    print('shape of dataset: {}'.format(dataset.shape))
    return dataset


def ramPriceDataset_visualization(dataset):
    plt.semilogy(dataset["date"], dataset["price"])
    plt.xlabel("Year")
    plt.ylabel("Price")
    plt.show()


def ramPriceDataset_regressionContrast(dataset):
    # data split based on date 2000
    data_train = dataset[dataset.date < 2000]
    data_test = dataset[dataset.date >= 2000]
    # target based on date
    data_train_nparray = data_train.date.to_numpy()
    # transform to 2D array (333, 1)
    X_train = data_train_nparray[:, np.newaxis]
    print('X_train shape: {}'.format(X_train.shape))
    y_train = np.log(data_train.price)
    tree = DecisionTreeRegressor().fit(X_train, y_train)
    linear_reg = LinearRegression().fit(X_train, y_train)
    # prediction
    X_all = dataset.date.to_numpy()[:, np.newaxis]
    print('X_all shape: {}'.format(X_all.shape))
    pred_tree = tree.predict(X_all)
    pred_lr = linear_reg.predict(X_all)
    # Logarithmic transformation convert
    price_tree = np.exp(pred_tree)
    price_lr = np.exp(pred_lr)
    # visualization
    plt.semilogy(data_train.date, data_train.price, 'b', label="Training data")
    plt.semilogy(data_test.date, data_test.price, 'r', label="Test data")
    plt.semilogy(dataset.date, price_tree, 'bo',
                 label="Tree prediction", markersize=2)
    plt.semilogy(dataset.date, price_lr, 'ro',
                 label="Linear prediction", markersize=2)
    plt.legend()
    plt.show()


def moonsDataset_load():
    X, y = make_moons(n_samples=100, noise=0.25, random_state=3)
    print('X shape: {}'.format(X.shape))
    print('y shape: {}'.format(y.shape))
    mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
    return X, y


def moonsDataset_split(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, random_state=42)
    print('X_train shape: {}'.format(X_train.shape))
    print('X_test shape: {}'.format(X_test.shape))
    return X_train, X_test, y_train, y_test


def moonsDataset_randomForestClassifier(X_train, y_train, X_test, y_test):
    forest = RandomForestClassifier(n_estimators=5, random_state=2)
    forest.fit(X_train, y_train)
    fig, axes = plt.subplots(2, 3, figsize=(
        20, 10))  # figsize: (width, height)
    for i, (ax, tree) in enumerate(zip(axes.ravel(), forest.estimators_)):
        ax.set_title("Tree {}".format(i))
        mglearn.plots.plot_tree_partition(X_train, y_train, tree, ax=ax)
    print('Training score: {}'.format(forest.score(X_train, y_train)))
    print('Test score: {}'.format(forest.score(X_test, y_test)))
    mglearn.plots.plot_2d_separator(
        forest, X_train, fill=True, ax=axes[-1, -1], alpha=.4)
    axes[-1, -1].set_title("Random Forest")
    mglearn.discrete_scatter(X_train[:, 0], X_train[:, 1], y_train)
    plt.show()


def moonsDataset_neuralNetworkClassifier(X_train, y_train, X_test, y_test):
    plt.figure(1)
    mlp = MLPClassifier(solver='lbfgs', random_state=0, hidden_layer_sizes=[
        10, 10], activation='tanh').fit(X_train, y_train)
    mglearn.plots.plot_2d_separator(mlp, X_train, fill=True, alpha=.3)
    mglearn.discrete_scatter(X_train[:, 0], X_train[:, 1], y_train)
    plt.title("MLP Classifier with training dataset")
    plt.xlabel("Feature 0")
    plt.ylabel("Feature 1")
    plt.figure(2)
    mlp_test = MLPClassifier(solver='lbfgs', random_state=0, hidden_layer_sizes=[
        10, 10], activation='tanh').fit(X_test, y_test)
    mglearn.plots.plot_2d_separator(mlp_test, X_test, fill=True, alpha=.3)
    mglearn.discrete_scatter(X_test[:, 0], X_test[:, 1], y_test)
    plt.title("MLP Classifier with test dataset")
    plt.xlabel("Feature 0")
    plt.ylabel("Feature 1")
    plt.show()


def moonsDataset_differentAlpha(X_train, y_train, X_test, y_test):
    fig, axes = plt.subplots(4, 4, figsize=(20, 8))
    for axx, n_hidden_nodes in zip(axes, [10, 100]):
        for ax, alpha in zip(axx, [0.0001, 0.01, 0.1, 1]):
            mlp = MLPClassifier(solver='lbfgs', random_state=0, hidden_layer_sizes=[
                n_hidden_nodes, n_hidden_nodes], alpha=alpha).fit(X_train, y_train)
            mglearn.plots.plot_2d_separator(
                mlp, X_train, fill=True, alpha=.3, ax=ax)
            mglearn.discrete_scatter(
                X_train[:, 0], X_train[:, 1], y_train, ax=ax)
            ax.set_title("n_hidden=[{}, {}], alpha={:.4f}".format(
                n_hidden_nodes, n_hidden_nodes, alpha))
    plt.show()


def moonsDataset_sameParameter(X_train, y_train):
    fig, axes = plt.subplots(2, 4, figsize=(20, 8))
    for i, ax in enumerate(axes.ravel()):
        mlp = MLPClassifier(solver='lbfgs', random_state=i, hidden_layer_sizes=[
            100, 100]).fit(X_train, y_train)
        mglearn.plots.plot_2d_separator(
            mlp, X_train, fill=True, alpha=.3, ax=ax)
        mglearn.discrete_scatter(X_train[:, 0], X_train[:, 1], y_train, ax=ax)
    plt.show()


def handcraftedDataset_load():
    X, y = mglearn.tools.make_handcrafted_dataset()
    print('X shape: {}'.format(X.shape))
    print('y shape: {}'.format(y.shape))
    return X, y


def handcraftedDataset_LinearSVC(X, y):
    svm = SVC(kernel='rbf', C=10.0, gamma=0.1).fit(X, y)
    mglearn.plots.plot_2d_separator(svm, X, fill=False, eps=0.5, alpha=.7)
    mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
    sv = svm.support_vectors_
    sv_labels = svm.dual_coef_.ravel() > 0
    mglearn.discrete_scatter(
        sv[:, 0], sv[:, 1], sv_labels, s=15, markeredgewidth=3)
    plt.title('svm for handcrafted dataset')
    plt.xlabel('Feature 0')
    plt.ylabel('Feature 1')
    plt.show()


def handcraftedDataset_SVCparameters(X, y):
    fig, axes = plt.subplots(3, 3, figsize=(15, 10))
    for ax, C in zip(axes, [-1, 0, 3]):
        for a, gamma in zip(ax, range(-1, 2)):
            mglearn.plots.plot_svm(log_C=C, log_gamma=gamma, ax=a)

    axes[0, 0].legend(['class 0', 'class 1', 'sv class 0',
                      'sv class 1'], ncol=4, loc=(.9, 1.2))
    plt.show()


def circlesDataset_load():
    X, y = make_circles(noise=0.2, factor=0.5, random_state=1)
    print('X shape: {}'.format(X.shape))
    print('y shape: {}'.format(y.shape))
    print('X head: \n{}'.format(X[:5]))
    print('y head: \n{}'.format(y[:5]))
    mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
    plt.title('circles dataset')
    plt.xlabel('Feature 0')
    plt.ylabel('Feature 1')
    plt.show()
    return X, y


def circlesDataset_categoryTransform(y):
    y_named = np.array(['blue', 'red'])[y]
    return y_named


def circlesDataset_split(X, y_named, y):
    X_train, X_test, y_train_named, y_test_named, y_train, y_test = train_test_split(
        X, y_named, y, random_state=0)
    print('X_test shape: {}'.format(X_test.shape))
    return X_train, X_test, y_train_named, y_test_named, y_train, y_test


def circlesDataset_GBC_model(X, X_train, X_test, y_train, y_test, y_train_named):
    gbrt = GradientBoostingClassifier(random_state=0)
    gbrt.fit(X_train, y_train_named)
    # decision_function
    print('descision function shape: {}'.format(
        gbrt.decision_function(X_test).shape))
    print('Decision function: \n{}'.format(gbrt.decision_function(X_test)[:6]))
    print('Thresholded decision function:\n{}'.format(
        gbrt.decision_function(X_test) > 0))
    print('Predictions:\n{}'.format(gbrt.predict(X_test)))
    # predict_proba
    print('predict_proba shape: {}'.format(gbrt.predict_proba(X_test).shape))
    print('Predicted probabilities:\n{}'.format(
        gbrt.predict_proba(X_test)[:6]))
    # transform the boolean output into 0/1
    greater_zero = (gbrt.decision_function(X_test) > 0).astype(int)
    # use 0 and 1 as indices
    pred = gbrt.classes_[greater_zero]
    print('pred shape: {}'.format(pred.shape))
    print('pred:\n{}'.format(pred))
    print('pred is equal to predictions: {}'.format(
        np.all(pred == gbrt.predict(X_test))))
    # decision_function visualization
    decision_function = gbrt.decision_function(X_test)
    print('Decision function minimum: {:.2f} maximum: {:.2f}'.format(
        np.min(decision_function), np.max(decision_function)))
    fig1, axes1 = plt.subplots(1, 2, figsize=(13, 5))
    mglearn.tools.plot_2d_separator(
        gbrt, X, ax=axes1[0], alpha=.4, fill=True, cm=mglearn.cm2)
    scores_image = mglearn.tools.plot_2d_scores(
        gbrt, X, ax=axes1[1], alpha=.5, cm=mglearn.ReBl)
    for ax in axes1:
        mglearn.discrete_scatter(
            X_test[:, 0], X_test[:, 1], y_test, ax=ax, markers='^')
        mglearn.discrete_scatter(
            X_train[:, 0], X_train[:, 1], y_train, ax=ax, markers='o')
        ax.set_xlabel("Feature 0")
        ax.set_ylabel("Feature 1")
        if ax == axes1[0]:
            ax.set_title("Decision boundary")
        else:
            ax.set_title("Decision function")
    cbar = plt.colorbar(scores_image, ax=axes1.tolist())
    axes1[0].legend(['Test class 0', 'Test class 1',
                    'Train class 0', 'Train class 1'], ncol=4, loc=(.1, 1.1))
    # predict_proba visualization
    fig2, axes2 = plt.subplots(1, 2, figsize=(13, 5))
    mglearn.tools.plot_2d_separator(
        gbrt, X, ax=axes2[0], alpha=.4, cm=mglearn.cm2, fill=True)
    scores_image = mglearn.tools.plot_2d_scores(
        gbrt, X, ax=axes2[1], alpha=.5, cm=mglearn.ReBl, function='predict_proba')
    for ax in axes2:
        mglearn.discrete_scatter(
            X_test[:, 0], X_test[:, 1], y_test, ax=ax, markers='^')
        mglearn.discrete_scatter(
            X_train[:, 0], X_train[:, 1], y_train, ax=ax, markers='o')
        ax.set_xlabel("Feature 0")
        ax.set_ylabel("Feature 1")
        if ax == axes2[0]:
            ax.set_title("Decision boundary")
        else:
            ax.set_title("Predicted probability")
    cbar = plt.colorbar(scores_image, ax=axes2.tolist())
    axes2[0].legend(['Test class 0', 'Test class 1',
                    'Train class 0', 'Train class 1'], ncol=4, loc=(.1, 1.1))
    plt.show()


def dataSet_scaling_visualization():
    mglearn.plots.plot_scaling()
    plt.show()


def dataSet_PCA_visualization():
    mglearn.plots.plot_pca_illustration()
    plt.show()
