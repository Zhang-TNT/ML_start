# encoding: utf-8
# @File  : main.py
# @Author: wenwen
# @Desc : 
# @Date  :  2025/04/20

# private import
# from funcs import myFunc
# from funcs import myTest1
from funcs import myTest2


def main():
    # irisOrignal, irisData, irisTarget = myTest1.IrisData_read()
    # # myTest1.IrisData_split(irisData, irisTarget)
    # X_train, X_test, y_train, y_test = myTest1.IrisData_split(irisData, irisTarget)
    # # myTest1.IrisData_visualization(X_train, y_train, irisOrignal['feature_names'])
    # myTest1.IrisData_visualization(X_train, y_train, irisOrignal['feature_names'])
    # myTest1.IrisData_knc(X_train, y_train, X_test, y_test)

    # X_forge, y_forge = myTest2.ForgeData_read()
    # myTest2.ForgeData_visualization(X_forge, y_forge)

    # X_wave, y_wave = myTest2.WaveData_read()
    # myTest2.WaveData_visualization(X_wave, y_wave)

    # breastcancerOrignal, breastcancerData, breastcancerTarget = myTest2.BreastCancerData_read()

    diabetesOrignal, diabetesData, diabetesTarget = myTest2.DiabetesData_read()

if __name__ == '__main__':
    main()


