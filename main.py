from dataset import class_dataset
from classifier import class_KNN
import numpy as np
if __name__ == '__main__':
    #load dataset
    data = class_dataset(dataset_name="glcm")

    #missing value


    #preprocesing (normalisasi)


    #model
    knn = class_KNN(data.X, data.y)
    # knn.viewDataset()
    knn.model()
    data_testing = np.array(data.X[0])
    # print(np.array(data_testing[0]))
    knn.predict(data.X)

    #evaluasi


