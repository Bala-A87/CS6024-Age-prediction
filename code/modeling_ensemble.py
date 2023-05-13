from argparse import ArgumentParser
from timeit import default_timer as timer
from scripts.get_data import get_processed_data
from scripts.models import VotingClassifier
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVC, SVR
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_absolute_error as mae, median_absolute_error as med, r2_score as r2
import numpy as np
from warnings import catch_warnings, simplefilter

parser = ArgumentParser()
parser.add_argument('-e', '--estimator', type=str, choices=['voting', 'svm', 'linreg'], default='voting', help='Estimator to use')
parser.add_argument('-bs', '--bin_size', type=int, default=20, help='Bin size for voting classifier')
parser.add_argument('-b', '--base', type=str, choices=['lda', 'logreg', 'svm'], default='lda', help='Base estimator to use with voting estimator')
parser.add_argument('-C', type=float, default=1., help='Value of C for SVC in the voting ensemble')

args = parser.parse_args()

data, labels = get_processed_data(corr_thresh=0.9, minmax_scale=True, var_thresh=0.01)
data_arr, labels_arr = np.array(data), np.array(labels)
print('Data loaded')

if args.base == 'logreg':
    base_estimator = LogisticRegression()
elif args.base == 'svm':
    base_estimator = SVC(kernel='linear', C=args.C)
else:
    base_estimator = LDA(solver='eigen', shrinkage='auto')

def get_model():
    if args.estimator == 'svm':
        model = SVR(kernel='linear')
    elif args.estimator == 'linreg':
        model = LinearRegression()
    else:
        model = VotingClassifier(args.bin_size, base_estimator)
    return model

desc_name = f'{args.estimator}_{args.base}' if args.estimator == 'voting' else args.estimator

log_path = f'../logs/{desc_name}.txt'

def add_log(log: str, end: str = '\n'):
    with open(log_path, mode='a') as f:
        f.write(log + end)

add_log('*'*100, '\n\n')
add_log(desc_name+str(args.C) if args.estimator == 'voting' and args.base == 'svm' else desc_name)

splitter = LeaveOneOut()
true_labels = []
pred_labels = []
train_time, eval_time = 0.0, 0.0

with catch_warnings():
    for fold, (train_ind, val_ind) in enumerate(splitter.split(data_arr)):
        simplefilter('ignore')
        model = get_model()
        X_train = data_arr[train_ind]
        X_val = data_arr[val_ind]
        Y_train = labels_arr[train_ind]
        Y_val = labels_arr[val_ind]
        feat_sel = SelectKBest(score_func=f_regression, k=1000)
        X_train_sel = feat_sel.fit_transform(X_train, Y_train)
        X_val_sel = feat_sel.transform(X_val)
        
        start_time = timer()
        model.fit(X_train_sel, Y_train)
        train_time += timer() - start_time

        start_time = timer()
        pred = model.predict(X_val_sel).reshape(-1,)
        eval_time += timer() - start_time

        true_labels.append(Y_val[0])
        pred_labels.append(pred[0])

add_log(f'Average train time: {(train_time / splitter.get_n_splits(data_arr)):.6f} s, average eval time: {(eval_time / splitter.get_n_splits(data_arr)):.6f} s')

mae_val = mae(np.array(true_labels), np.array(pred_labels))
med_val = med(np.array(true_labels), np.array(pred_labels))
r2_val = r2(np.array(true_labels), np.array(pred_labels))

add_log(f'Mean absolute error: {mae_val}')
add_log(f'Median absolute error: {med_val}')
add_log(f'R2 score: {r2_val}', end='\n\n')
