from dataset import class_dataset
from classifier import class_KNN
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

import numpy as np
if __name__ == '__main__':
    #load dataset
    # glcm : all fitur
    # dissimilarity : dis fitur
    data = class_dataset(dataset_name="dissimilarity")

    #missing value


    #preprocesing (normalisasi)


    #model
    X_train, X_test, y_train, y_test = train_test_split(
        data.X, data.y, test_size = 0.33, random_state = 42)
    knn = class_KNN(X_train, y_train)
    # knn.viewDataset()
    knn.model()
    # data_testing = np.array(data.X[0])
    # print(np.array(data_testing[0]))
    y_pred = knn.predict(X_test)
    # print("y prediksi L", y_pred)
    # print("y actual ",data.y)
    acc = accuracy_score(y_test, y_pred)
    print("accuracy : ",acc)
    #evaluasi


