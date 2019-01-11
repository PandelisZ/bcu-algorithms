import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

class Algorithm:

    def __init__ (self, name, training, target, verbose = False, ):
        self.name = name
        self.data_training = training
        self.data_target = target
        self.VERBOSE = verbose

    def print_results(self, f1_score, precision_score, recall_score):
        print("Algorithm: %s" % self.name)
        self.print_score(f1_score, "F1 Score")
        self.print_score(precision_score, "Precision Score")
        self.print_score(recall_score, "Recall Score")

    def print_score(self, score_array, label):
        avg = np.mean(score_array)
        std = np.std(score_array)
        print("    {:s} - Mean: {:f} - Standard Deviation: {:f}".format(label, avg, std))

    #Evaluates model and gives f1-score. Generic to all algorithms
    def score_model(self, model, test_training, test_target):
        target_prediction = model.predict(test_training)
        from sklearn.metrics import classification_report
        if(self.VERBOSE):
            print(classification_report(test_target, target_prediction))

        return [
            f1_score(test_training, target_prediction, average='weighted'),
            precision_score(test_target, target_prediction, average='weighted'),
            recall_score(test_training, target_prediction, average='weighted')
        ]

    def execute(self):
        kfold = StratifiedKFold(10, True, 1)
        for train, test in kfold.split(self.data_training, self.data_target):
            model = self.train(self.data_training.iloc[train], self.data_target.iloc[train])
            f1_score, precision_score, recall_score = self.score_model(model, self.data_training.iloc[test], self.data_target.iloc[test])

        self.print_results(f1_score, precision_score, recall_score)

    # def train(self, training_x, training_y):
    #     pass
