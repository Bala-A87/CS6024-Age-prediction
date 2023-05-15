from sklearn.base import clone, BaseEstimator
import numpy as np

class VotingClassifier:
    def __init__(self, bin_size:int, estimator: BaseEstimator):
        self.bin_size = bin_size
        self.models = [None]*bin_size
        for i in range(bin_size):
            self.models[i] = clone(estimator)

    def fit(self, data: np.array, labels: np.array):
        for i in range(self.bin_size):
            # binned_labels = labels.apply(lambda x: (x - i) // self.bin_size)
            binned_labels = (labels - i) // self.bin_size
            self.models[i].fit(data, binned_labels)

    def predict(self, data: np.array):
        predictions = [self.models[i].predict(data) for i in range(self.bin_size)]
        age_predictions = np.zeros((data.shape[0], 121)) # 120 is the max age
        for i in range(self.bin_size):
            for j in range(data.shape[0]):
                for age in range(i + self.bin_size*predictions[i][j], i+self.bin_size + self.bin_size*predictions[i][j]):
                    age_predictions[j][age] += 1
        return age_predictions.argmax(axis=1)

class HierarchicalModel():
    def __init__(
        self,
        num_bins: int,
        bin_predictor: BaseEstimator,
        output_predictor: BaseEstimator
    ) -> None:
        self.num_bins = num_bins
        self.binner = clone(bin_predictor)
        self.predictors = []
        for _ in range(num_bins):
            self.predictors.append(clone(output_predictor))
    
    def fit(self, X: np.array, y: np.array):
        y_sort = np.sort(y)
        self.bin_endpts = []
        for bin in range(1, self.num_bins):
            self.bin_endpts.append(y_sort[int(bin * len(y) / self.num_bins)])
        y_transformed_list = []
        for i in range(len(y)):
            bin = 0
            for j, endpt in enumerate(self.bin_endpts):
                if endpt > y[i]:
                    bin = j
                    break
            y_transformed_list.append(bin)
        y_transformed = np.array(y_transformed_list)
        self.binner.fit(X, y_transformed)
        for bin in range(self.num_bins):
            X_bin = X[y_transformed == bin]
            y_bin = y[y_transformed == bin]
            if len(X_bin.shape) == 1:
                X_bin = X_bin.reshape(1, -1)
            if len(y_bin) > 0:
                self.predictors[bin].fit(X_bin, y_bin)
    
    def predict(self, X: np.array):
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
        pred_bins = self.binner.predict(X)
        preds = []
        for i in range(len(X)):
            preds.append(self.predictors[pred_bins[i]].predict(np.array(X[i]).reshape(1, -1)))
        preds = np.array(preds).reshape(-1,)
        return preds
    