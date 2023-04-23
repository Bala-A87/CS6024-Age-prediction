from argparse import ArgumentParser
import pandas as pd
from timeit import default_timer as timer
from scripts.models import VotingClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVC, SVR
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_absolute_error, median_absolute_error, r2_score 
import numpy as np
import matplotlib.pyplot as plt
from seaborn import regplot

parser = ArgumentParser()
parser.add_argument('-e', '--estimator', type=str, choices=['voting', 'svm', 'linreg'], default='voting', help='Estimator to use')
parser.add_argument('-b', '--base', type=str, choices=['lda', 'logreg', 'svm', 'knn', 'tree'], default='lda', help='Base estimator to use with voting estimator')
parser.add_argument('-C', type=float, default=1., help='Value of C for SVC in the voting ensemble')
args = parser.parse_args()

data, labels = pd.read_csv('../data/data_proc.csv'), pd.read_csv('../data/labels.csv')

if args.base == 'lda':
    base_estimator = LDA(solver='eigen', shrinkage='auto')
elif args.base == 'logreg':
    base_estimator = LogisticRegression()
elif args.base == 'svm':
    base_estimator = SVC(kernel='poly', degree=2, C=args.C, coef0=1.)
elif args.base == 'knn':
    base_estimator = KNeighborsClassifier()
else:
    base_estimator = DecisionTreeClassifier()

def get_model():
    if args.estimator == 'voting':
        model = VotingClassifier(20, base_estimator)
    elif args.estimator == 'svm':
        model = SVR(kernel='linear')
    else:
        model = LinearRegression()
    return model

desc_name = f'voting_{args.base}_{args.C}' if args.estimator == 'voting' else args.estimator

# log_path = '../logs/logs_ensemble.txt'
log_path = '../logs/logs_svm_ensemble_cv.txt'

def add_log(log: str, end: str = '\n'):
    with open(log_path, mode='a') as f:
        f.write(log + end)

add_log('*'*100, '\n\n')
add_log(desc_name)

splitter = LeaveOneOut()
true_labels = []
pred_labels = []
train_time, eval_time = 0.0, 0.0

for fold, (train_ind, val_ind) in enumerate(splitter.split(data)):
    model = get_model()
    train_timer = timer()
    if args.estimator == 'voting':
        model.train(data.iloc[train_ind], labels.iloc[train_ind])
    else:
        model.fit(data.iloc[train_ind], labels.iloc[train_ind])
    train_time += timer() - train_timer
    eval_timer = timer()
    pred = model.predict(data.iloc[val_ind]).reshape(-1,)
    eval_time += timer() - eval_timer
    true_labels.append(labels.iloc[val_ind].values[0])
    pred_labels.append(pred[0])
    add_log(f'[Fold {fold+1}/{splitter.get_n_splits(data)}] ==> True label: {true_labels[-1]}, Prediction: {pred_labels[-1]}')

add_log(f'Average train time: {(train_time / splitter.get_n_splits(data)):.6f} s, average eval time: {(eval_time / splitter.get_n_splits(data)):.6f} s')

mean_error = mean_absolute_error(np.array(true_labels), np.array(pred_labels))
median_error = median_absolute_error(np.array(true_labels), np.array(pred_labels))
r2 = r2_score(np.array(true_labels), np.array(pred_labels))

add_log(f'Mean absolute error: {mean_error}')
add_log(f'Median absolute error: {median_error}')
add_log(f'R2 score: {r2}', end='\n\n')


plt.figure(figsize=(6, 6))
regplot(x=true_labels, y=pred_labels)
plt.title(desc_name)
plt.xlabel('True ages')
plt.ylabel('Predictions')
plt.savefig(f'../plots/svm_ensemble_cv/{desc_name}.png')
