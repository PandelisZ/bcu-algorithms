from .algorithm import Algorithm
from sklearn.neighbors import KNeighborsClassifier


def classifier_wrapper(*args, **kwargs):
    """Wrapper for the KNeighborsClassifier class in order to keep track of arguments

    Returns:
        [object] -- KNeighborsClassifier instance with only keyword augments passed to it
    """

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
    def leaf_size_run(self):
        # Default
        print("Default settings")
        self.train(classifier_wrapper(n_neighbors=5, weights="uniform", algorithm="auto", leaf_size=30, p=2, metric="minkowski"))

        for n_neighbors in range(1, 11, 2):
            for leaf_size in range(20, 50, 5):
                    self.train(classifier_wrapper(n_neighbors=n_neighbors, weights="uniform", algorithm="auto", leaf_size=leaf_size, p=1, metric="minkowski"))

    def weights_algorithm_run(self):
        for weights in ['uniform', 'distance']:
            for algorithm in ['auto', 'ball_tree', 'kd_tree', 'brute']:
                    self.train(classifier_wrapper(n_neighbors=2, weights=weights, algorithm=algorithm, leaf_size=30, p=1, metric="minkowski"))


    def execute(self):
        # Default
        print("Default settings")
        self.train(classifier_wrapper(n_neighbors=5, weights="uniform", algorithm="auto", leaf_size=30, p=2, metric="minkowski"))

        for n_neighbors in range(1, 11):
            for leaf_size in range(1, 51):
                for p in range(1,4):
                    self.train(classifier_wrapper(n_neighbors=n_neighbors, weights="uniform", algorithm="auto", leaf_size=leaf_size, p=p, metric="minkowski"))

