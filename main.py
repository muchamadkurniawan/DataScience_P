from dataset import class_dataset
from classifier import *
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from feature_selection import *
from sklearn.model_selection import KFold
import statistics
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import cross_val_score

import numpy as np
if __name__ == '__main__':
    # load_dataset
    # list dataset:
    # 2D
    # diabetes
    # glcm
    # dissimilarity
    # asmaDataset
    data = class_dataset(dataset_name="asmaDataset")
    # print("data freme :\n", data.df)
    print("feature :\n", data.X)
    print("class :\n", data.y)

    #missing value


    #preprocesing (normalisasi)
    select = feature_selection(data.X, data.y)
    # # data.X = select.linearSVC()
    # # data.X = select.tree_based()
    # # data.X = select.pearson_correletion(0.35)
    # # data.X = select.selectKbest_Anova(4)
    # # data.X = select.selectKbest_Chi2(4)
    # # data.X = select.selectKbest_regression(4)
    data.X = select.RFE_SVC(4)

    # # random split dataset
    # X_train, X_test, y_train, y_test = train_test_split(
    #     data.X, data.y, test_size=0.33, random_state=42)
    # DT = class_DecisionTree(X_train, y_train)
    # DT.model()
    # y_pred = DT.predict(X_test)
    # accRSD = accuracy_score(y_test, y_pred)


    # Kfold cross validation
    # # kfold with NB
    # data.y = np.array(data.y)
    # # print(data.y)
    # kf = KFold(n_splits=3, random_state=None, shuffle=True)
    # accAll = []
    # for train_index, test_index in kf.split(data.X):
    #     # print("train index : ", train_index)
    #     # print("test index :",test_index)
    #     X_train, X_test = data.X[train_index,:], data.X[test_index,:]
    #     # print("xtrain ", X_train)
    #     # print("xtest ", X_test)
    #     y_train, y_test = data.y[train_index], data.y[test_index]
    #     # print("len train :", len(y_train))
    #     # print("len test :",len(y_test))
    #     # model KNN
    #     print("--------------------------------")
    #     DT = class_DecisionTree(X_train, y_train)
    #     DT.model()
    #     y_pred = DT.predict(X_test)
    #     # print("compare :", y_pred, y_test)
    #     acc = accuracy_score(y_test, y_pred)
    #     print("accuracy: ", acc)
    #     # print("accuracy : ", acc)
    #     accAll.append(acc)
    # print("akurasi rata-rata (Kfold) : ", statistics.mean(accAll))
    # print("ACC Random split dataset: ", accRSD)
    #
    # # scores = cross_val_score(DT, X, y, cv=5, scoring='accuracy')
    # loo = LeaveOneOut()
    # loo.get_n_splits(data.X)
    # accLOO= []
    # for train_index, test_index in loo.split(data.X):
    #     X_train, X_test = data.X[train_index, :], data.X[test_index, :]
    #     y_train, y_test = data.y[train_index], data.y[test_index]
    #     DT = class_DecisionTree(X_train, y_train)
    #     DT.model()
    #     y_pred = DT.predict(X_test)
    #     acc = accuracy_score(y_test, y_pred)
    #     # print("accuracy: ", acc)
    #     accLOO.append(acc)
    # print("akurasi rata-rata (LOOCV) : ", statistics.mean(accLOO))
    #



    # ## model NB
    # print("--------------------------------")
    # nb = class_NB(X_train, y_train)
    # nb.model()
    # y_pred = nb.predict(X_test)
    # acc = accuracy_score(y_test, y_pred)
    # print("accuracy: ",acc)
    #
    # ##  Model Random Forest
    # print("--------------------------------")
    # RF = class_RandomForest(X_train, y_train)
    # RF.model()
    # y_pred = RF.predict(X_test)
    # acc = accuracy_score(y_test, y_pred)
    # print("accuracy: ", acc)
    #
    # ## Model Decesion Tree
    # print("--------------------------------")
    # DT = class_DecisionTree(X_train, y_train)
    # DT.model()
    # y_pred = DT.predict(X_test)
    # acc = accuracy_score(y_test, y_pred)
    # print("accuracy: ", acc)
    #
    # ## model SVM
    # print("--------------------------------")
    # SVM = class_SVM(X_train, y_train)
    # SVM.model()
    # y_pred = SVM.predict(X_test)
    # acc = accuracy_score(y_test, y_pred)
    # print("accuracy: ", acc)
    #
    # ## model MLP
    # print("--------------------------------")
    # mlp = class_MLP(X_train, y_train)
    # mlp.model()
    # y_pred = mlp.predict(X_test)
    # acc = accuracy_score(y_test, y_pred)
    # print("accuracy: ", acc)
    #
    # ## model Adaboost - MLP
    # print("--------------------------------")
    # adaboost = class_adaboostSVM(X_train, y_train)
    # adaboost.model()
    # y_pred = adaboost.predict(X_test)
    # acc = accuracy_score(y_test, y_pred)
    # print("accuracy: ", acc)
    #
    # ## Linear Regression
    # print("--------------------------------")
    # LR = classLinearRegression(X_train, y_train)
    # LR.model()
    # y_pred = LR.predict(X_test)
    # acc = accuracy_score(y_test, y_pred)
    # print("accuracy: ", acc)

    # # #evaluasi
    # #
    # #