from dataset import class_dataset
from classifier import *
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from feature_selection import *

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
    # print("feature :\n", data.X)
    # print("class :\n", data.y)

    #missing value


    #preprocesing (normalisasi)
    select = feature_selection(data.X, data.y)
    # data.X = select.linearSVC()
    # data.X = select.tree_based()
    data.X = select.pearson_correletion(0.35)
    print(data.X.shape)

    # random split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        data.X, data.y, test_size=0.33, random_state=42)

    # model KNN
    knn = class_KNN(X_train, y_train)
    knn.model()
    y_pred = knn.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print("accuracy : ",acc)

    ## model NB
    nb = class_NB(X_train, y_train)
    nb.model()
    y_pred = nb.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print("accuracy NB: ",acc)

    #
    # #evaluasi
    #
    #
