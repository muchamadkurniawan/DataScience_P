from sklearn.feature_selection import SelectFromModel
from sklearn.svm import LinearSVC
from sklearn.ensemble import ExtraTreesClassifier
from scipy.stats import pearsonr
import numpy as np

class feature_selection:
    X = []
    y = []

    def __init__(self, X, y):
        self.X = X
        self.y = y

    def linearSVC(self):
        print("feature selection: linear SVC")
        lsvc = LinearSVC(
            C=0.01, penalty="l2", dual=False).fit(self.X, self.y)
        model = SelectFromModel(lsvc, prefit=True)
        X_new = model.transform(self.X)
        print(X_new.shape)
        return X_new

    def tree_based(self):
        print("feature selection: tree based")
        clf = ExtraTreesClassifier(n_estimators=30,criterion="entropy")
        clf = clf.fit(self.X, self.y)
        # print(clf.feature_importances_)
        model = SelectFromModel(clf, prefit=True)
        X_new = model.transform(self.X)
        print(X_new.shape)
        return X_new

    def pearson_correletion(self, threshold):
        print("feature selection: pearson correlation")
        # print("len X ",len(self.X[:,0]))
        # print("len y ",len(self.y))
        person=[]
        for i in range(len(self.X[0])):
            corr, _ = pearsonr(self.X[:,i], self.y)
            person.append(corr)
        # print(person)
        th = threshold
        ind = []
        newX = []
        for i in range(len(person)):
            if person[i]>th or person[i]<-th:
                ind.append(i)
                x = []
                for j in range(len(self.X)):
                    x.append(self.X[j][i])
                newX.append(x)
        # print(np.array(newX))
        # print(ind)
        return np.array(newX).transpose()
