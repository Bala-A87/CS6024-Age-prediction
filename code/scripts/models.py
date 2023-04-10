from sklearn.base import clone, BaseEstimator
import numpy as np

class VotingClassifier:
    def __init__(self, bin_size:int, estimator: BaseEstimator):
        self.bin_size = bin_size
        self.models = [None]*bin_size
        for i in range(bin_size):
            self.models[i] = clone(estimator)

    def train(self, data, labels):
        for i in range(self.bin_size):
            binned_labels = labels.apply(lambda x: (x - i) // self.bin_size)
            self.models[i].fit(data, binned_labels)

    def predict(self, data):
        predictions = [self.models[i].predict(data) for i in range(self.bin_size)]
        age_predictions = np.zeros((data.shape[0], 121)) # 120 is the max age
        for i in range(self.bin_size):
            for j in range(data.shape[0]):
                for age in range(i + self.bin_size*predictions[i][j], i+self.bin_size + self.bin_size*predictions[i][j]):
                    age_predictions[j][age] += 1
        return age_predictions.argmax(axis=1)
    