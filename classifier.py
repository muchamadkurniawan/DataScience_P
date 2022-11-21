from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import numpy as np

class class_NB:
    X = []
    y = []
    model = None

    def __init__(self, data, label):
        print("Naive Bayes classifier")
        self.X = data
        self.y = label

    def viewDataset(self):
        print("data :", self.X)
        print("label :", self.y)

    def model(self):
        # print("in model X :", self.X)
        # print(len(self.X))
        self.model = GaussianNB()
        self.model.fit(np.array(self.X), self.y)

    def predict(self, data_testing):
        # print("prediksi : ", self.model.predict(data_testing))
        return self.model.predict(data_testing)

class class_KNN:
    X = []
    y = []
    model = None

    def __init__(self, data, label):
        print("KNN classifier")
        self.X = data
        self.y = label

    def viewDataset(self):
        print("data :", self.X)
        print("label :", self.y)

    def model(self):
        # print("in model X :", self.X)
        # print(len(self.X))
        self.model = KNeighborsClassifier(n_neighbors=10)
        self.model.fit(np.array(self.X), self.y)

    def predict(self, data_testing):
        # print("prediksi : ", self.model.predict(data_testing))
        return self.model.predict(data_testing)