from .algorithm import Algorithm
from sklearn.neighbors import KNeighborsClassifier


def classifier_wrapper(*args, **kwargs):
    print("Supplied argumnets: ",kwargs)
    return KNeighborsClassifier(**kwargs)


class NearestNeighbor(Algorithm):

    def __init__ (self, training, target, verbose = False):
        super().__init__("Nearest Neighbor", training, target, verbose)

    def execute(self):
        self.train(classifier_wrapper(n_neighbors=3))
        self.train(classifier_wrapper(n_neighbors=3))
        self.train(classifier_wrapper(n_neighbors=3))
        self.train(classifier_wrapper(n_neighbors=3))
        self.train(classifier_wrapper(n_neighbors=3))
        self.train(classifier_wrapper(n_neighbors=3))
