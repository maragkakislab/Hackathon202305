import numpy as np
import pandas as pd


def train_test_split_df(df, train_portion=0.8, verbose=False):
    """
    Splits dataframe into train and test split based on unique genes
    """
    gene_names = df['name'].unique()
    gene_count = len(gene_names)
    train_gene_names = gene_names[:int(gene_count*train_portion)]
    test_gene_names = gene_names[int(gene_count*train_portion):]

    if(verbose):
        print('num of unique train genes',len(train_gene_names))
        print('num of unique test genes',len(test_gene_names))

    assert len(np.intersect1d(train_gene_names, test_gene_names)) == 0
    
    train_df = df[df['name'].isin(train_gene_names)]
    test_df = df[df['name'].isin(test_gene_names)]
    
    if(verbose):
        print('num of train rows',len(train_df))
        print('num of test rows',len(test_df))
    
    return train_df, test_df

