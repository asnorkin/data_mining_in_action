from dmia.classifiers import BinaryBoostingClassifier
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from dmia.utils import plot_surface
import numpy as np


def main():
    X, y = make_classification(n_samples=500, n_features=2,
                               n_informative=2, n_redundant=0, n_repeated=0,
                               n_classes=2, n_clusters_per_class=2,
                               flip_y=0.05, class_sep=0.8, random_state=241)
    y = 2 * (y - 0.5)

    clf = BinaryBoostingClassifier(n_estimators=100).fit(X, y)
    plot_surface(X, y, clf)

if __name__ == "__main__":
    main()