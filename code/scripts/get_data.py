import pandas as pd

dataset = pd.read_csv('../data/dataset.tsv', sep='\t')
data = dataset.drop(columns=['age'], inplace=False)
labels = dataset['age']

const_cols = data.columns[data.std() == 0.0]
data_non_zero_var = data.drop(columns=const_cols, inplace=False)

def get_clean_data():
    return data_non_zero_var, labels
