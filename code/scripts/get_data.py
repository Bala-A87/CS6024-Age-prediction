import pandas as pd
from typing import Tuple, Union, Callable
import numpy as np
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_regression, RFE
from sklearn.linear_model import LinearRegression
from pathlib import Path
import os

dataset = pd.read_csv('../data/dataset.tsv', sep='\t')
data = dataset.drop(columns=['age'], inplace=False)
labels = dataset['age']

const_cols = data.columns[data.std() == 0.0]
data_non_zero_var = data.drop(columns=const_cols, inplace=False)

def get_processed_data(
    log_transform: bool = True,
    stability_coef: float = 1e-5,
    corr_thresh: float = 0.95,
    minmax_scale: bool = False,
    var_thresh: float = None,
    k_to_select: Union[int, str] = 'all',
    selection_metric: Union[Callable, str] = f_regression,
    cache_path: str = '../data/'
) -> Tuple[pd.DataFrame, pd.Series]: 
    """
    Returns processed data from the age prediction dataset after some feature elimination.

    Args:
        log_transform (bool, optional): Whether to convert the data to log values.
            Defaults to True.
        stability_coef (float, optional): Small constant to add while log transforming to handle zeroes.
            Defaults to 1e-5.
        corr_thresh (float, optional): Maximum correlation between features to allow. Any feature with larger correlation in the upper triangle of the correlation matrix will be dropped. Set to None to keep all features.
            Defaults to 0.95.
        minmax_scale (bool, optional): Whether to scale the data so that each column has values between 0.0 and 1.0.
            Defaults to False.
        var_thresh (float, optional): The minimum required variance of a feature to keep (after optional scaling). Set to None to keep all features.
            Defaults to None.
        k_to_select (int or str, optional): The number of features to select, after the above processing. Set to 'all' to select all features.
            Defaults to 'all'.
        selection_metric (Callable or str, optional): The selection metric to use to choose the top `k_to_select` features. Ideally a function from sklearn.feature_selection usable in sklearn.feature_selection.SelectKBest. Pass 'recursive' if recursive feature elimination using linear regression is to be used.
            Defaults to sklearn.feature_selection.f_regression.
        cache_path (str, optional): Directory to save processed data in for quicker access. Not cached if None is passed.
            Defaults to '../data/'.
        
    Returns:
        (data_proc, labels) (Tuple[pd.DataFrame, pd.Series]): The data after processing and the true labels.
    """
    if Path(cache_path+'data_clean.tsv').is_file() and Path(cache_path+'labels.tsv').is_file():
        data_proc = pd.read_csv(cache_path+'data_clean.tsv', sep='\t')
        labels_proc = pd.read_csv(cache_path+'labels.tsv', sep='\t')
    else:
        data_proc = data_non_zero_var.copy()
        labels_proc = labels.copy()

        if log_transform:
            data_proc = np.log2(data_proc + stability_coef)

        if corr_thresh is not None:
            corr_matrix = data_proc.corr().abs()
            upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            corr_cols = [column for column in upper.columns if any(upper[column] > corr_thresh)]
            data_proc.drop(columns=corr_cols, inplace=True)

        if minmax_scale:
            data_min = data_proc.min()
            data_max = data_proc.max()
            data_proc = (data_proc - data_min)/(data_max - data_min)
        
        if var_thresh is not None:
            thresh = VarianceThreshold(var_thresh)
            data_signif_var = thresh.fit_transform(data_proc)
            cols_signif_var = thresh.get_feature_names_out(data_proc.columns)
            data_proc = pd.DataFrame(data=data_signif_var, columns=cols_signif_var)
        
        if k_to_select != 'all' and selection_metric is not None:
            if selection_metric == 'recursive':
                feature_selector = RFE(estimator=LinearRegression(), n_features_to_select=k_to_select, step=0.1)
                data_selected = feature_selector.fit_transform(data_proc, labels_proc)
                cols_selected = feature_selector.get_feature_names_out(data_proc.columns)
                data_proc = pd.DataFrame(data=data_selected, columns=cols_selected)
            else:
                feature_selector = SelectKBest(score_func=selection_metric, k=k_to_select)
                data_selected = feature_selector.fit_transform(data_proc, labels_proc)
                cols_selected = feature_selector.get_feature_names_out(data_proc.columns)
                data_proc = pd.DataFrame(data=data_selected, columns=cols_selected)
    
    if cache_path is not None:
        if not Path(cache_path).is_dir():
            os.mkdir(cache_path)
        data_proc.to_csv(cache_path+'data_clean.tsv', sep='\t', index=None)
        labels_proc.to_csv(cache_path+'labels.tsv', sep='\t', index=None)

    return data_proc, labels_proc
