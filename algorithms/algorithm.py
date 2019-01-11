import datetime
import time
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

def timer(func):
    """Time the execution time of a function

    Arguments:
        func {function} -- The function to wrap

    Returns:
        None
    """
    def timer_wraper(*args, **kwargs):
        start = time.time()
        func(*args, **kwargs)
        end = time.time()
        runtime = (end - start)
        #Minutes, seconds, hours, minutes
        m, s = divmod(runtime, 60)
        h, m = divmod(m, 60)
        print("    Execution time: %d:%02d:%02d (H:MM:SS)" % (h, m, s))
    return timer_wraper


class Algorithm:

    def __init__ (self, name, training, target, verbose = False, ):
        self.name = name
        self.data_training = training
        self.data_target = target
        self.VERBOSE = verbose

    def print_results(self, f1_score, precision_score, recall_score):
        """Print the algorithms score for various facets

        Arguments:
            f1_score {array} -- F1 Score
            precision_score {array} -- Precision Score
            recall_score {array} -- Recall Score
        """
        print("Algorithm: %s" % self.name)
        self.print_score(f1_score, "F1 Score")
        self.print_score(precision_score, "Precision Score")
        self.print_score(recall_score, "Recall Score")

    def print_score(self, score_array, label):
        """Helper function for pretty printing the scores

        Arguments:
            score_array {array} -- Array of ints
            label {[type]} -- Score label
        """
        avg = np.mean(score_array)
        std = np.std(score_array)
        print("    {:s} - Mean: {:f} - Standard Deviation: {:f}".format(label, avg, std))

    def score_model(self, model, test_training, test_target):
        """Evaluates model and gives f1-score. Generic to all algorithms

        Arguments:
            model {object} -- Algorithm model
            test_training {object} -- X training data
            test_target {object} -- y expected data

        Returns:
            [array] -- The scoring of the algorithms accuracy
        """

        target_prediction = model.predict(test_training)
        from sklearn.metrics import classification_report
        if(self.VERBOSE):
            print(classification_report(test_target, target_prediction))

        return [
            f1_score(test_target, target_prediction, average='weighted'),
            precision_score(test_target, target_prediction, average='weighted'),
            recall_score(test_target, target_prediction, average='weighted')
        ]

    @timer
    def train(self, algorithm):
        """Main training function. This is the entry for a training process in this class

        Arguments:
            algorithm {object} -- This is the instanciated algorithm object to call .fit()
                                  onto in order to get the model
        """

        kfold = StratifiedKFold(10, True, 1)
        f1_score = []
        precision_score = []
        recall_score = []
        for train, test in kfold.split(self.data_training, self.data_target):
            model = algorithm.fit(self.data_training.iloc[train], self.data_target.iloc[train])
            scores = self.score_model(model, self.data_training.iloc[test], self.data_target.iloc[test])
            f1_score.append(scores[0])
            precision_score.append(scores[1])
            recall_score.append(scores[2])

        self.print_results(f1_score, precision_score, recall_score)

