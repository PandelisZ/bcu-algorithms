from .algorithm import Algorithm
from sklearn.neighbors import KNeighborsClassifier

class NearestNeighbor(Algorithm):

    def __init__ (self, training, target, verbose = False):
        super().__init__("Nearest Neighbor", training, target, verbose)

    def train(self, training_x, training_y):
        model = KNeighborsClassifier(n_neighbors=3)
        return model.fit(training_x, training_y)
