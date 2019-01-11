from .algorithm import Algorithm
from sklearn.neighbors import KNeighborsClassifier


def classifier_wrapper(*args, **kwargs):
    print("Supplied argumnets: ",kwargs)
    return KNeighborsClassifier(**kwargs)


class NearestNeighbor(Algorithm):

    def __init__ (self, training, target, verbose = False):
        super().__init__("Nearest Neighbor", training, target, verbose)

    def discovery_run(self):
        # Default
        print("Default settings")
        self.train(classifier_wrapper(n_neighbors=5, weights="uniform", algorithm="auto", leaf_size=30, p=2, metric="minkowski"))

        for n_neighbors in range(1, 11):
            for leaf_size in range(1, 51):
                for p in range(1,4):
                    self.train(classifier_wrapper(n_neighbors=n_neighbors, weights="uniform", algorithm="auto", leaf_size=leaf_size, p=p, metric="minkowski"))

    def execute(self):
        # Default
        print("Default settings")
        self.train(classifier_wrapper(n_neighbors=5, weights="uniform", algorithm="auto", leaf_size=30, p=2, metric="minkowski"))

        for n_neighbors in range(1, 11):
            for leaf_size in range(1, 51):
                for p in range(1,4):
                    self.train(classifier_wrapper(n_neighbors=n_neighbors, weights="uniform", algorithm="auto", leaf_size=leaf_size, p=p, metric="minkowski"))

