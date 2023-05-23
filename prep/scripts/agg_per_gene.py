import argparse
import numpy as np
import pandas as pd

parser= argparse.ArgumentParser(description="aggregate transcript statistics per gene")
parser.add_argument("-f", "--file", help="file containing transcript level information")
parser.add_argument("-g", "--group_col", help="column name to group by")
parser.add_argument("-c", "--counts_col", help="column name to calculate weights")
parser.add_argument("-t", "--agg_type", help="type of aggregation to perform: max, mean")
parser.add_argument("-o", "--output_file", help="path and name of output file")


args = parser.parse_args()
                                

def weighted_mean(df, weight_col, agg_col):
    weighted_mean = np.average(df[agg_col], weights=df[weight_col])
    return weighted_mean

def load_csv(fpath):
    df = pd.read_csv(fpath, sep=",", header=0)
    return df

def save_csv(df, out):
    df.to_csv(out, index=False)

df = load_csv(args.file)
                                
if args.agg_type == "mean":
    ## TO DO ## add dropped columns warning
    cols = df.select_dtypes(include=['float64']).columns
    df = df.groupby(args.group_col).apply(
        lambda x: pd.Series(
        {cols[i]:weighted_mean(x, args.counts_col,cols[i]) for i in range(0,len(cols))}))
    df = df["sum_"+args.counts_col] = df.groupby(args.group_col)[args.counts_col].sum().values

elif args.agg_type == "max":
    df = df.loc[df.groupby(args.group_col)[args.counts_col].idxmax()]

else:
    print("pick either mean or max for agg_type.")
    quit()
                                
save_csv(df, args.output_file)

