# -*- coding:utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA

class IrisData:

    def __init__(self):
        self.iris = load_iris()
        self.x = self.iris.data
        self.y = self.iris.target
        pca = PCA(n_components=2)
        self.x_pca = pca.fit_transform(self.x)

    def scatter_iris_2d(self):
        colors = ["r", "b", "y"]
        self.c = [colors[self.y[i]] for i in range(len(self.y))]
        plt.scatter(self.x_pca[:, 0], self.x_pca[:, 1], c=self.c)

    def x(self):
        return self.x_pca

if __name__ == "__main__":
    iris = IrisData()
    iris.scatter_iris_2d()
    plt.show()
