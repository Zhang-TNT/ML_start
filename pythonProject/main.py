# standard import
# =============================================================================
# import os
# =============================================================================
# local import
from myModule import myFunction


def main():
    # # 导出Iris数据集，load_iris()返回的iris对象是一个Bunch对象，类似字典
    # iris_dataset = myFunction.dataSet_load()
    # # 输出数据集属性
    # myFunction.dataSet_info(iris_dataset)
    # # 将数据集转化为dataframe
    # iris_dataFrame = myFunction.dataSet_frame(iris_dataset)
    # # 以表格的形式显示数据
    # print(iris_dataFrame)
    # print(iris_dataFrame.info())
    # # 划分训练集和测试集，其中X表示数据，y表示标签
    # X_train, X_test, y_train, y_test = myFunction.dataSet_split(iris_dataset)
    # # 数据可视化
    # # myFunction.dataSet_plot(iris_dataFrame)
    # # 模型训练
    # knn = myFunction.kNN_classify()
    # knn.fit(X_train, y_train)
    # # 模型预测
    # X_new = myFunction.testData_generation()
    # print('X_new.shape: {}'.format(X_new.shape))
    # prediction = knn.predict(X_new)
    # print('Prediction: {}'.format(prediction))
    # print(type(prediction))
    # print('Predicted target name: {}'.format(iris_dataset['target_names'][prediction]))
    # # 模型评估
    # y_pred = knn.predict(X_test)
    # print('Test set predictions:\n {}'.format(y_pred))
    # print('y_test: {}'.format(y_test))
    # print('Test set score: {:.2f}'.format(sum(y_pred == y_test) / len(y_test)))
    # print('Test set score: {:.2f}'.format(knn.score(X_test, y_test))) # 使用score方法评估
    # myFunction.IrisDataset_multiClass(X_train, X_test, y_train, y_test)

    # # forge数据集
    # forge_X, forge_y = myFunction.forgeDataset_load()
    # print('forge data shape: {}'.format(forge_X.shape))
    # forgeData_train, forgeData_test, forgeTarget_train, forgeTarget_test = myFunction.forgeDataset_split(forge_X, forge_y)
    # forgeData_clf = myFunction.forgeDataset_knnClassifier()
    # forgeData_clf.fit(forgeData_train, forgeTarget_train)                                               # 拟合数据集
    # print('testSet predictions: {}'.format(forgeData_clf.predict(forgeData_test)))                      # 预测测试集
    # print('testSet accuracy: {:.2f}'.format(forgeData_clf.score(forgeData_test, forgeTarget_test)))     # 测试集准确率
    # myFunction.forgeDataset_knnVisualization(forge_X, forge_y)
    # myFunction.forgeDataset_linearRegression(forge_X, forge_y)
    # myFunction.forgeDataset_SVC(forge_X, forge_y)

    # # wave数据集
    # wave_X, wave_y = myFunction.waveDataset_load()
    # print('wave data shape: {}'.format(wave_X.shape))
    # print('wave target shape: {}'.format(wave_y.shape))
    # waveData_train, waveData_test, waveTarget_train, waveTarget_test = myFunction.waveDataset_split(wave_X, wave_y)
    # waveData_reg = myFunction.waveDataset_knnRegressor()
    # waveData_reg.fit(waveData_train, waveTarget_train)
    # print('testSet accuracy: {:.2f}'.format(waveData_reg.score(waveData_test, waveTarget_test)))
    # myFunction.waveDataset_knnVisualization(waveData_train, waveData_test, waveTarget_train, waveTarget_test)
    # myFunction.waveDataset_linearRegression(waveData_train, waveTarget_train)
    # waveData_lr = myFunction.waveDataset_linearRegression(waveData_train, waveTarget_train)
    # print('testSet accuracy: {:.2f}'.format(waveData_lr.score(waveData_test, waveTarget_test)))
    # print('trainSet accuracy: {:.2f}'.format(waveData_lr.score(waveData_train, waveTarget_train)))

    # breast cancer数据集
    # breastCancer_dataset = myFunction.breastCancerdataset_load()
    # breastCancerdataset_train, breastCancerdataset_test, breastCancertarget_train, breastCancertarget_test = myFunction.breastCancerdataset_split(
    # breastCancer_dataset)
    # myFunction.breastCancerdataset_visualization(breastCancer_dataset)
    # myFunction.breastCancerdataset_knnClassifier(breastCancerdataset_train, breastCancertarget_train, breastCancerdataset_test, breastCancertarget_test)
    # breastCancerdataset_lr = myFunction.breastCancerdataset_linearRegression(breastCancerdataset_train, breastCancertarget_train)
    # print('testSet accuracy: {:.2f}'.format(breastCancerdataset_lr.score(breastCancerdataset_test, breastCancertarget_test)))
    # print('trainSet accuracy: {:.2f}'.format(breastCancerdataset_lr.score(breastCancerdataset_train, breastCancertarget_train)))
    # breastCancerdataset_ridge, breastCancerdataset_ridge10, breastCancerdataset_ridge01 = myFunction.breastCancerdataset_RidgeRegression(breastCancerdataset_train, breastCancertarget_train)
    # print('testSet accuracy: {:.2f}'.format(breastCancerdataset_ridge.score(breastCancerdataset_test, breastCancertarget_test)))
    # print('trainSet accuracy: {:.2f}'.format(breastCancerdataset_ridge.score(breastCancerdataset_train, breastCancertarget_train)))
    # myFunction.breastCancerdataset_RidgeVisualization(breastCancerdataset_ridge, breastCancerdataset_ridge10, breastCancerdataset_ridge01, breastCancerdataset_lr)
    # myFunction.breastCancerdataset_learningCurve()
    # breastCancerdataset_lasso, breastCancerdataset_lasso001, breastCancerdataset_lasso00001= myFunction.breastCancerdataset_lassoRegression(breastCancerdataset_train, breastCancertarget_train)
    # print('testSet accuracy: {:.2f}'.format(breastCancerdataset_lasso.score(breastCancerdataset_test, breastCancertarget_test)))
    # print('trainSet accuracy: {:.2f}'.format(breastCancerdataset_lasso.score(breastCancerdataset_train, breastCancertarget_train)))
    # print('testSet accuracy: {:.2f}'.format(breastCancerdataset_lasso001.score(breastCancerdataset_test, breastCancertarget_test)))
    # print('trainSet accuracy: {:.2f}'.format(breastCancerdataset_lasso001.score(breastCancerdataset_train, breastCancertarget_train)))
    # print('testSet accuracy: {:.2f}'.format(breastCancerdataset_lasso00001.score(breastCancerdataset_test, breastCancertarget_test)))
    # print('trainSet accuracy: {:.2f}'.format(breastCancerdataset_lasso00001.score(breastCancerdataset_train, breastCancertarget_train)))
    # myFunction.breastCancerdataset_lassoVisualization(breastCancerdataset_lasso, breastCancerdataset_lasso001, breastCancerdataset_lasso00001, breastCancerdataset_lr)
    # breastCancerdataset_logreg, breastCancerdataset_logreg100, breastCancerdataset_logreg001 = myFunction.breastCancerdataset_logisticRegression(breastCancerdataset_train, breastCancertarget_train, breastCancerdataset_test, breastCancertarget_test)
    # myFunction.breastCancerdataset_logisticRegressionVisualization(breastCancer_dataset, breastCancerdataset_logreg, breastCancerdataset_logreg100, breastCancerdataset_logreg001)
    # breastCancer_tree = myFunction.breastCancerdataset_decisionTreeClassifier(breastCancerdataset_train, breastCancertarget_train, breastCancerdataset_test, breastCancertarget_test)
    # myFunction.breastCancerdataset_decisionTreeGraphviz(breastCancer_tree, breastCancer_dataset)
    # myFunction.breastCancerdataset_treeFeatureImportance(breastCancer_tree, breastCancer_dataset)
    # myFunction.tree_example()
    # breastCancer_forest = myFunction.breastCancerdataset_randomForestClassifier(breastCancerdataset_train, breastCancertarget_train, breastCancerdataset_test, breastCancertarget_test)
    # myFunction.breastCancerdataset_treeFeatureImportance(breastCancer_forest, breastCancer_dataset)
    # breastCancer_gbrt = myFunction.breastCancerdataset_GradientBoostingClassifier(breastCancerdataset_train, breastCancertarget_train, breastCancerdataset_test, breastCancertarget_test)
    # myFunction.breastCancerdataset_treeFeatureImportance(breastCancer_gbrt, breastCancer_dataset)
    # breastCancerdataset_SVC = myFunction.breastCancerdataset_SVM(
    # breastCancerdataset_train, breastCancertarget_train, breastCancerdataset_test, breastCancertarget_test)
    # breastCancerdataset_train_std = myFunction.breastCancerdataset_prprocessing(breastCancerdataset_train)
    # breastCancerdataset_test_std = myFunction.breastCancerdataset_prprocessing(breastCancerdataset_test)
    # breastCancerdataset_SVC_std = myFunction.breastCancerdataset_SVM(breastCancerdataset_train_std, breastCancertarget_train, breastCancerdataset_test_std, breastCancertarget_test)
    # breastCancerdataset_mlp = myFunction.breastCancerdataset_neuralNetworkClassifier(breastCancerdataset_train, breastCancertarget_train, breastCancerdataset_test, breastCancertarget_test)
    # breastCancerdataset_train_scaled, breastCancerdataset_test_scaled = myFunction.breastCancerdataset_dataProcessing(breastCancerdataset_train, breastCancerdataset_test)
    # myFunction.breastCancerdataset_scaledDatasetresult(breastCancerdataset_train_scaled, breastCancerdataset_test_scaled, breastCancertarget_train, breastCancertarget_test, breastCancer_dataset)
    # myFunction.breastCancerdataset_preprocessingMinMaxScaler(
    #     breastCancerdataset_train, breastCancerdataset_test)

    # circles数据集
    # circles_X, circles_y = myFunction.circlesDataset_load()
    # circles_y_named = myFunction.circlesDataset_categoryTransform(circles_y)
    # circles_X_train, circles_X_test, circles_y_train_named, circles_y_test_named, circles_y_train, circles_y_test = myFunction.circlesDataset_split(circles_X, circles_y_named, circles_y)
    # myFunction.circlesDataset_GBC_model(circles_X, circles_X_train, circles_X_test, circles_y_train, circles_y_test, circles_y_train_named)

    # california_housing数据集
    # californiaHousing_dataset = myFunction.californiaHousingdataset_load()
    # print('target names: {}'.format(californiaHousing_dataset.feature_names))
    # caliHousdataset_train, caliHousdataset_test, caliHoustarget_train, caliHoustarget_test = myFunction.californiaHousingdataset_split(californiaHousing_dataset)
    # caliHousdataset_lr = myFunction.caliHousdataset_linearRegression(caliHousdataset_train, caliHoustarget_train)
    # print('testSet accuracy: {:.2f}'.format(caliHousdataset_lr.score(caliHousdataset_test, caliHoustarget_test)))
    # print('trainSet accuracy: {:.2f}'.format(caliHousdataset_lr.score(caliHousdataset_train, caliHoustarget_train)))
    # caliHousdataset_ridge = myFunction.caliHousdataset_RidgeRegression(caliHousdataset_train, caliHoustarget_train)
    # print('testSet accuracy: {:.2f}'.format(caliHousdataset_ridge.score(caliHousdataset_test, caliHoustarget_test)))
    # print('trainSet accuracy: {:.2f}'.format(caliHousdataset_ridge.score(caliHousdataset_train, caliHoustarget_train)))
    # myFunction.caliHousdataset_visualization(caliHousdataset_ridge)

    # blobs数据集
    # blobs_X, blobs_y = myFunction.blobsDataset_load()
    # myFunction.blobsDataset_dataVisualization(blobs_X, blobs_y)
    # blobsDataset_svm = myFunction.blobsDataset_linearSVC_classifier(blobs_X, blobs_y)
    # myFunction.blobsDataset_linearSVC_classifierVisualization(blobsDataset_svm, blobs_X, blobs_y)
    # myFunction.blobsDataset_linearSVC_2dclassifierVisualization(blobsDataset_svm, blobs_X, blobs_y)
    # blobs_X_3d = myFunction.blobsDataset_3dExpand(blobs_X, blobs_y)
    # blobsDataset_svm_3d = myFunction.blobsDataset_linearSVC_classifier(blobs_X_3d, blobs_y)
    # myFunction.blobsDataset_linearSVC_3dclassifierVisualization(blobsDataset_svm_3d, blobs_X_3d, blobs_y)
    # myFunction.blobsDataset_linearSVC_orignalFeature(blobsDataset_svm_3d, blobs_X_3d, blobs_y)
    # myFunction.blobsDataset_split(blobs_X)

    # BernoulliNB classifier
    # dataSet_X, dataSet_y = myFunction.sampleDataset_load()
    # print('shape of dataSet_X: {}'.format(dataSet_X.shape))
    # print('shape of dataSet_y: {}'.format(dataSet_y.shape))
    # # myFunction.sampleDataset_unique(dataSet_X, dataSet_y)
    # print('X[0]: {}'.format(dataSet_X[0]))

    # ram price数据集
    # ramPrice_dataset = myFunction.ramPriceDataset_load()
    # myFunction.ramPriceDataset_visualization(ramPrice_dataset)
    # myFunction.ramPriceDataset_regressionContrast(ramPrice_dataset)

    # moons数据集
    # moons_X, moons_y = myFunction.moonsDataset_load()
    # moonsData_train, moonsData_test, moonsTarget_train, moonsTarget_test = myFunction.moonsDataset_split(moons_X, moons_y)
    # myFunction.moonsDataset_randomForestClassifier(moonsData_train, moonsTarget_train, moonsData_test, moonsTarget_test)
    # myFunction.moonsDataset_neuralNetworkClassifier(moonsData_train, moonsTarget_train, moonsData_test, moonsTarget_test)
    # myFunction.moonsDataset_differentAlpha(moonsData_train, moonsTarget_train, moonsData_test, moonsTarget_test)
    # myFunction.moonsDataset_sameParameter(moonsData_train, moonsTarget_train)

    # handcraft数据集
    # handcraft_X, handcraft_y = myFunction.handcraftedDataset_load()
    # myFunction.handcraftedDataset_LinearSVC(handcraft_X, handcraft_y)
    # myFunction.handcraftedDataset_SVCparameters(handcraft_X, handcraft_y)

    # dataset scaling
    # myFunction.dataSet_scaling_visualization()
    myFunction.dataSet_PCA_visualization()


if __name__ == '__main__':
    main()
