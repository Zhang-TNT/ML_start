# encoding: utf-8
# @File  : main.py
# @Author: wenwen
# @Desc : 
# @Date  :  2025/04/20

# private import
# from funcs import myFunc
from funcs import myTest1


def main():
    # myFunc.myFunc_read()
    irisOrignal, irisData, irisTarget = myTest1.IrisData_read()
    # myTest1.IrisData_split(irisData, irisTarget)
    X_train, X_test, y_train, y_test = myTest1.IrisData_split(irisData, irisTarget)
    # myTest1.IrisData_visualization(X_train, y_train, irisOrignal['feature_names'])
    myTest1.IrisData_visualization(X_train, y_train, irisOrignal['feature_names'])
    myTest1.IrisData_knc(X_train, y_train, X_test, y_test)

if __name__ == '__main__':
    main()


